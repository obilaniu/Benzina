/* Includes */
#include <cuda.h>
#include <cuda_runtime.h>

#include "benzina/benzina.h"
#include "kernels.h"


/* Defines */



/* CUDA kernels */

BENZINA_PLUGIN_STATIC __device__ uchar3 ycbcr2rgb(uchar3 yuv, unsigned colorMatrix){
	unsigned char Y = yuv.x, Cb = yuv.y, Cr = yuv.z, R, G, B;
	switch(colorMatrix){
		case 0:
			/* See ITU-T Rec. T.871 (JFIF), Section 7 */
			R = min(max(0.0f, round(Y                        +1.402000f*(Cr-128.0f))), 255.0f);
			G = min(max(0.0f, round(Y -0.344136f*(Cb-128.0f) -0.714136f*(Cr-128.0f))), 255.0f);
			B = min(max(0.0f, round(Y +1.772000f*(Cb-128.0f)                       )), 255.0f);
		break;
		default:
			R = Y;
			G = Cb;
			B = Cr;
		break;
	}
	
	return make_uchar3(R,G,B);
}

/**
 * @brief CUDA post-processing kernel
 */

BENZINA_PLUGIN_HIDDEN __global__ void
__launch_bounds__(1024, 2)
nvdecodePostprocKernel(void*    dstPtr,
                       unsigned dstH,
                       unsigned dstW,
                       float    OOB0,
                       float    OOB1,
                       float    OOB2,
                       float    B0,
                       float    B1,
                       float    B2,
                       float    S0,
                       float    S1,
                       float    S2,
                       float    H00,
                       float    H01,
                       float    H02,
                       float    H10,
                       float    H11,
                       float    H12,
                       float    H20,
                       float    H21,
                       float    H22,
                       unsigned colorMatrix,
                       void*    srcPtr,
                       unsigned srcPitch,
                       unsigned srcH,
                       unsigned srcW){
	/* Compute Destination Coordinates */
	const unsigned x = blockDim.x*blockIdx.x + threadIdx.x;
	const unsigned y = blockDim.y*blockIdx.y + threadIdx.y;
	if(x >= dstW || y >= dstH){return;}
	
	/* Compute Destination R,G,B Pointers */
	float* const dstPtr0 = (float*)dstPtr + 0*dstH*dstW + y*dstW + x;
	float* const dstPtr1 = (float*)dstPtr + 1*dstH*dstW + y*dstW + x;
	float* const dstPtr2 = (float*)dstPtr + 2*dstH*dstW + y*dstW + x;
	
	/* Compute Source Coordinates in homogeneous coordinates, then prespective-divide: X = Hx */
	const float _X = H00*x + H01*y + H02;
	const float _Y = H10*x + H11*y + H12;
	const float _W = H20*x + H21*y + H22;
	const float X  = _X/_W;
	const float Y  = _Y/_W;
	
	/* Out of Bounds Check */
	if(X<0 || X>srcW-1 || Y<0 || Y>srcH-1 || isnan(X) || isnan(Y)){
		*dstPtr0 = OOB0;
		*dstPtr1 = OOB1;
		*dstPtr2 = OOB2;
		return;
	}
	
	/**
	 * YUV420 Pixel Quad Fetch from NV12
	 * 
	 * In NV12 YUV420, the pixel data is laid out as follows: A plane of luma
	 * (Y) samples first, then an interleaved plane of chroma (U,V) samples,
	 * subsampled 2x vertically and 2x horizontally. In ASCII art:
	 * 
	 * YYYYYYYYYYYYYYYY
	 * YYYYYYYYYYYYYYYY
	 * YYYYYYYYYYYYYYYY
	 * YYYYYYYYYYYYYYYY
	 * UVUVUVUVUVUVUVUV
	 * UVUVUVUVUVUVUVUV
	 * 
	 * We will perform linear interpolation between sample points. For this we
	 * need the coordinates and values of 4 samples about P = (X,Y), which we
	 * number 0,1,2,3. The coordinates of luma are (clX*, clY*). The coordinates
	 * of chroma are (ccX*, ccY*). In ASCII art:
	 * 
	 *  (0,0)
	 *      +-----------> X
	 *      |
	 *      |
	 *      |
	 *      v           0  1
	 *      Y             P
	 *                  2  3
	 */
	
	const unsigned clX0 = floor(X),          clY0 = floor(Y);
	const unsigned clX1 = min(clX0+1, srcW), clY1 = clY0;
	const unsigned clX2 = clX0,              clY2 = min(clY0+1, srcH);
	const unsigned clX3 = clX1,              clY3 = clY2;
	
	const unsigned ccX0 = clX0/2,            ccY0 = clY0/2;
	const unsigned ccX1 = clX1/2,            ccY1 = clY1/2;
	const unsigned ccX2 = clX2/2,            ccY2 = clY2/2;
	const unsigned ccX3 = clX3/2,            ccY3 = clY3/2;
	
	const float    Xf   = X-clX0,            Yf   = Y-clY0;
	
	#define LUMAAT(X,Y)     ((uchar1*)((uchar1*)srcPtr + (Y)       *srcPitch +   (X)))
	#define CHROMAAT(X,Y)   ((uchar2*)((uchar1*)srcPtr + (srcH+(Y))*srcPitch + 2*(X)))
	const uchar1   vl0  = *LUMAAT  (clX0, clY0);
	const uchar1   vl1  = *LUMAAT  (clX1, clY1);
	const uchar1   vl2  = *LUMAAT  (clX2, clY2);
	const uchar1   vl3  = *LUMAAT  (clX3, clY3);
	const uchar2   vc0  = *CHROMAAT(ccX0, ccY0);
	const uchar2   vc1  = *CHROMAAT(ccX1, ccY1);
	const uchar2   vc2  = *CHROMAAT(ccX2, ccY2);
	const uchar2   vc3  = *CHROMAAT(ccX3, ccY3);
	#undef LUMAAT
	#undef CHROMAAT
	
	/**
	 * YUV -> RGB colorspace conversion of the 4 sample points.
	 */
	
	const uchar3   v0   = ycbcr2rgb(make_uchar3(vl0.x, vc0.x, vc0.y), colorMatrix);
	const uchar3   v1   = ycbcr2rgb(make_uchar3(vl1.x, vc1.x, vc1.y), colorMatrix);
	const uchar3   v2   = ycbcr2rgb(make_uchar3(vl2.x, vc2.x, vc2.y), colorMatrix);
	const uchar3   v3   = ycbcr2rgb(make_uchar3(vl3.x, vc3.x, vc3.y), colorMatrix);
	
	/**
	 * Linear Interpolation between the sample points.
	 */
	
	#define LERP(a,b,alpha) make_float3((1.0f-(alpha))*a.x + ((alpha))*b.x, \
	                                    (1.0f-(alpha))*a.y + ((alpha))*b.y, \
	                                    (1.0f-(alpha))*a.z + ((alpha))*b.z)
	const float3   v    = LERP(LERP(v0, v1, Xf),
	                           LERP(v2, v3, Xf), Yf);
	#undef LERP
	
	/**
	 * Scaling, Biasing, Write-out.
	 */
	
	*dstPtr0 = v.x*S0 + B0;
	*dstPtr1 = v.y*S1 + B1;
	*dstPtr2 = v.z*S2 + B2;
}

/**
 * C wrapper function to invoke CUDA C++ kernel.
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodePostprocKernelInvoker(cudaStream_t cudaStream,
                                                          void*        dstPtr,
                                                          unsigned     dstH,
                                                          unsigned     dstW,
                                                          float        OOB0,
                                                          float        OOB1,
                                                          float        OOB2,
                                                          float        B0,
                                                          float        B1,
                                                          float        B2,
                                                          float        S0,
                                                          float        S1,
                                                          float        S2,
                                                          float        H00,
                                                          float        H01,
                                                          float        H02,
                                                          float        H10,
                                                          float        H11,
                                                          float        H12,
                                                          float        H20,
                                                          float        H21,
                                                          float        H22,
                                                          unsigned     colorMatrix,
                                                          void*        srcPtr,
                                                          unsigned     srcPitch,
                                                          unsigned     srcH,
                                                          unsigned     srcW){
	dim3 Db = {                32,                 32, 1},
	     Dg = {(dstW+Db.x-1)/Db.x, (dstH+Db.y-1)/Db.y, 1};
	nvdecodePostprocKernel<<<Dg, Db, 0, cudaStream>>>(dstPtr,
	                                                  dstH,
	                                                  dstW,
	                                                  OOB0,
	                                                  OOB1,
	                                                  OOB2,
	                                                  B0,
	                                                  B1,
	                                                  B2,
	                                                  S0,
	                                                  S1,
	                                                  S2,
	                                                  H00,
	                                                  H01,
	                                                  H02,
	                                                  H10,
	                                                  H11,
	                                                  H12,
	                                                  H20,
	                                                  H21,
	                                                  H22,
	                                                  colorMatrix,
	                                                  srcPtr,
	                                                  srcPitch,
	                                                  srcH,
	                                                  srcW);
	return 0;
}
