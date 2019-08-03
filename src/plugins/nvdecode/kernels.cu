/* Includes */
#include <cuda.h>
#include <cuda_runtime.h>

#include "benzina/benzina-old.h"
#include "kernels.h"


/* Defines */



/* CUDA kernels */

/**
 * YCbCr to RGB colorspace conversion kernel
 */

BENZINA_PLUGIN_STATIC __device__ float3 ycbcr2rgb(float3 ycbcr, unsigned colorMatrix){
	float Kr, Kg, Kb, Y = ycbcr.x, Cb = ycbcr.y, Cr = ycbcr.z, R, G, B;
	switch(colorMatrix){
		case 0:
			/**
			 * See ITU-T Rec. T.871 (JFIF), Section 7
			 * 
			 * Uses ITU-R BT.601-6-625 recommentation
			 *   Kr = 0.299
			 *   Kg = 0.587
			 *   Kb = 0.114
			 * but full scale
			 *    Y,Cb,Cr in [0, 255]
			 */
			R = Y                        +1.402000f*(Cr-128.0f);
			G = Y -0.344136f*(Cb-128.0f) -0.714136f*(Cr-128.0f);
			B = Y +1.772000f*(Cb-128.0f)                       ;
			R = min(max(0.0f, round(R)), 255.0f);
			G = min(max(0.0f, round(G)), 255.0f);
			B = min(max(0.0f, round(B)), 255.0f);
		break;
		case 1:
			/**
			 * ITU-R BT.601-6-625 recommentation, with head/footroom
			 *    Y       in [16,235]
			 *    Cb,Cr   in [16,240]
			 */
			Kr = 0.299f; Kg = 0.587f; Kb = 0.114f;
			Y = 255.0f/219.0f*(Y - 16.0f);
			Cb= 255.0f/112.0f*(Cb-128.0f);
			Cr= 255.0f/112.0f*(Cr-128.0f);
			R = Y                        +         (1.0f-Kr)*Cr;
			G = Y - (Kb/Kg)*(1.0f-Kb)*Cb - (Kr/Kg)*(1.0f-Kr)*Cr;
			B = Y +         (1.0f-Kb)*Cb;
			R = min(max(0.0f, round(R)), 255.0f);
			G = min(max(0.0f, round(G)), 255.0f);
			B = min(max(0.0f, round(B)), 255.0f);
		break;
		case 2:
			/**
			 * ITU-R BT.709 recommentation, with head/footroom
			 *    Y       in [16,235]
			 *    Cb,Cr   in [16,240]
			 */
			Kr = 0.2126f; Kg = 0.7152f; Kb = 0.0722f;
			Y = 255.0f/219.0f*(Y - 16.0f);
			Cb= 255.0f/112.0f*(Cb-128.0f);
			Cr= 255.0f/112.0f*(Cr-128.0f);
			R = Y                        +         (1.0f-Kr)*Cr;
			G = Y - (Kb/Kg)*(1.0f-Kb)*Cb - (Kr/Kg)*(1.0f-Kr)*Cr;
			B = Y +         (1.0f-Kb)*Cb;
			R = min(max(0.0f, round(R)), 255.0f);
			G = min(max(0.0f, round(G)), 255.0f);
			B = min(max(0.0f, round(B)), 255.0f);
		break;
		case 3:
			/**
			 * ITU-R BT.2020 recommentation, with head/footroom
			 *    Y       in [16,235]
			 *    Cb,Cr   in [16,240]
			 */
			Kr = 0.2627f; Kg = 0.6780f; Kb = 0.0593f;
			Y = 255.0f/219.0f*(Y - 16.0f);
			Cb= 255.0f/112.0f*(Cb-128.0f);
			Cr= 255.0f/112.0f*(Cr-128.0f);
			R = Y                        +         (1.0f-Kr)*Cr;
			G = Y - (Kb/Kg)*(1.0f-Kb)*Cb - (Kr/Kg)*(1.0f-Kr)*Cr;
			B = Y +         (1.0f-Kb)*Cb;
			R = min(max(0.0f, round(R)), 255.0f);
			G = min(max(0.0f, round(G)), 255.0f);
			B = min(max(0.0f, round(B)), 255.0f);
		break;
		default:
			R = Y;
			G = Cb;
			B = Cr;
		break;
	}
	
	return make_float3(R,G,B);
}

/**
 * @brief CUDA post-processing kernel
 */

BENZINA_PLUGIN_STATIC __global__ void
__launch_bounds__(1024, 2)
nvdecodePostprocKernelTex2D(void* __restrict__       dstPtr,
                            unsigned                 dstH,
                            unsigned                 dstW,
                            float                    OOB0,
                            float                    OOB1,
                            float                    OOB2,
                            float                    B0,
                            float                    B1,
                            float                    B2,
                            float                    S0,
                            float                    S1,
                            float                    S2,
                            float                    H00,
                            float                    H01,
                            float                    H02,
                            float                    H10,
                            float                    H11,
                            float                    H12,
                            float                    H20,
                            float                    H21,
                            float                    H22,
                            unsigned                 colorMatrix,
                            cudaTextureObject_t      srcYPlane,
                            cudaTextureObject_t      srcCPlane,
                            unsigned                 srcH,
                            unsigned                 srcW){
	/* Compute Destination Coordinates */
	const unsigned x = blockDim.x*blockIdx.x + threadIdx.x;
	const unsigned y = blockDim.y*blockIdx.y + threadIdx.y;
	if(x >= dstW || y >= dstH){return;}
	
	/* Compute Destination R,G,B Pointers */
	float* const __restrict__ dstPtr0 = (float*)dstPtr + 0*dstH*dstW + y*dstW + x;
	float* const __restrict__ dstPtr1 = (float*)dstPtr + 1*dstH*dstW + y*dstW + x;
	float* const __restrict__ dstPtr2 = (float*)dstPtr + 2*dstH*dstW + y*dstW + x;
	
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
	 * 
	 * Under JFIF, chroma samples are exactly in the center of every luma quad:
	 * 
	 *         L   L   L   L   L   L   L   L   L   L   L   L
	 *                                                      
	 *           C       C       C       C       C       C  
	 *                                                      
	 *         L   L   L   L   L   L   L   L   L   L   L   L
	 *                                                      
	 *                                                      
	 *                                                      
	 *         L   L   L   L   L   L   L   L   L   L   L   L
	 *                                                      
	 *           C       C       C       C       C       C  
	 *                                                      
	 *         L   L   L   L   L   L   L   L   L   L   L   L
	 *                                                      
	 *                                                      
	 *                                                      
	 *         L   L   L   L   L   L   L   L   L   L   L   L
	 *                                                      
	 *           C       C       C       C       C       C  
	 *                                                      
	 *         L   L   L   L   L   L   L   L   L   L   L   L
	 */
	
	const float  Xc   = 0.5f*X-0.25f, Yc   = 0.5f*Y-0.25f;
	
	const float  clX0 = floorf(X),    clY0 = floorf(Y);
	const float  clX1 = clX0+1,       clY1 = clY0;
	const float  clX2 = clX0,         clY2 = clY0+1;
	const float  clX3 = clX1,         clY3 = clY2;
	const float  ccX0 = floorf(Xc),   ccY0 = floorf(Yc);
	const float  ccX1 = ccX0+1,       ccY1 = ccY0;
	const float  ccX2 = ccX0,         ccY2 = ccY0+1;
	const float  ccX3 = ccX1,         ccY3 = ccY2;
	
	const float  Xlf  = X  - clX0,    Ylf  = Y  - clY0;
	const float  Xcf  = Xc - ccX0,    Ycf  = Yc - ccY0;
	
	const uchar1 vl0  = tex2D<uchar1>(srcYPlane, clX0, clY0);
	const uchar1 vl1  = tex2D<uchar1>(srcYPlane, clX1, clY1);
	const uchar1 vl2  = tex2D<uchar1>(srcYPlane, clX2, clY2);
	const uchar1 vl3  = tex2D<uchar1>(srcYPlane, clX3, clY3);
	const uchar2 vc0  = tex2D<uchar2>(srcCPlane, ccX0, ccY0);
	const uchar2 vc1  = tex2D<uchar2>(srcCPlane, ccX1, ccY1);
	const uchar2 vc2  = tex2D<uchar2>(srcCPlane, ccX2, ccY2);
	const uchar2 vc3  = tex2D<uchar2>(srcCPlane, ccX3, ccY3);
	
	/**
	 * Linear Interpolation between the sample points.
	 */
	
	#define LERP(a,b,alpha) ((1.0f-(alpha))*(a) + ((alpha))*(b))
	const float3 s    = make_float3(LERP(LERP(vl0.x, vl1.x, Xlf),
	                                     LERP(vl2.x, vl3.x, Xlf), Ylf),
	                                LERP(LERP(vc0.x, vc1.x, Xcf),
	                                     LERP(vc2.x, vc3.x, Xcf), Ycf),
	                                LERP(LERP(vc0.y, vc1.y, Xcf),
	                                     LERP(vc2.y, vc3.y, Xcf), Ycf));
	#undef LERP
	
	/**
	 * YCbCr -> RGB colorspace conversion of the interpolated point.
	 */
	
	const float3 v    = ycbcr2rgb(s, colorMatrix);
	
	/**
	 * Biasing, Scaling, Write-out.
	 */
	
	*dstPtr0 = (v.x-B0)*S0;
	*dstPtr1 = (v.y-B1)*S1;
	*dstPtr2 = (v.z-B2)*S2;
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
	dim3 Db, Dg;
	Db.x =                 32; Db.y =                 32; Db.z = 1;
	Dg.x = (dstW+Db.x-1)/Db.x; Dg.y = (dstH+Db.y-1)/Db.y; Dg.z = 1;
	
	/* Specify texture */
	struct cudaResourceDesc yResDesc, cResDesc;
	struct cudaTextureDesc  yTexDesc, cTexDesc;
	cudaTextureObject_t     yTexObj,  cTexObj;
	
	memset(&yResDesc, 0, sizeof(yResDesc));
	memset(&yTexDesc, 0, sizeof(yTexDesc));
	memset(&cResDesc, 0, sizeof(cResDesc));
	memset(&cTexDesc, 0, sizeof(cTexDesc));
	yTexObj = 0;
	cTexObj = 0;
	
	yResDesc.resType                  = cudaResourceTypePitch2D;
	yResDesc.res.pitch2D.desc.f       = cudaChannelFormatKindUnsigned;
	yResDesc.res.pitch2D.desc.x       = 8;
	yResDesc.res.pitch2D.desc.y       = 0;
	yResDesc.res.pitch2D.desc.z       = 0;
	yResDesc.res.pitch2D.desc.w       = 0;
	yResDesc.res.pitch2D.devPtr       = (void*)(srcPtr);
	yResDesc.res.pitch2D.height       = srcH;
	yResDesc.res.pitch2D.width        = srcW;
	yResDesc.res.pitch2D.pitchInBytes = srcPitch;
	
	yTexDesc.addressMode[0]           = cudaAddressModeClamp;
	yTexDesc.addressMode[1]           = cudaAddressModeClamp;
	yTexDesc.filterMode               = cudaFilterModePoint;
	yTexDesc.readMode                 = cudaReadModeElementType;
	yTexDesc.sRGB                     = 0;
	yTexDesc.normalizedCoords         = 0;
	
	cResDesc.resType                  = cudaResourceTypePitch2D;
	cResDesc.res.pitch2D.desc.f       = cudaChannelFormatKindUnsigned;
	cResDesc.res.pitch2D.desc.x       = 8;
	cResDesc.res.pitch2D.desc.y       = 8;
	cResDesc.res.pitch2D.desc.z       = 0;
	cResDesc.res.pitch2D.desc.w       = 0;
	cResDesc.res.pitch2D.devPtr       = (void*)((char*)srcPtr + srcH*srcPitch);
	cResDesc.res.pitch2D.height       = srcH/2;
	cResDesc.res.pitch2D.width        = srcW/2;
	cResDesc.res.pitch2D.pitchInBytes = srcPitch;
	
	cTexDesc.addressMode[0]           = cudaAddressModeClamp;
	cTexDesc.addressMode[1]           = cudaAddressModeClamp;
	cTexDesc.filterMode               = cudaFilterModePoint;
	cTexDesc.readMode                 = cudaReadModeElementType;
	cTexDesc.sRGB                     = 0;
	cTexDesc.normalizedCoords         = 0;
	
	cudaCreateTextureObject(&yTexObj, &yResDesc, &yTexDesc, NULL);
	cudaCreateTextureObject(&cTexObj, &cResDesc, &cTexDesc, NULL);
	nvdecodePostprocKernelTex2D<<<Dg, Db, 0, cudaStream>>>(dstPtr,
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
	                                                       yTexObj,
	                                                       cTexObj,
	                                                       srcH,
	                                                       srcW);
	cudaDestroyTextureObject(yTexObj);
	cudaDestroyTextureObject(cTexObj);
	
	return 0;
}
