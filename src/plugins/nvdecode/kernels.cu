/* Includes */
#include <cuda.h>
#include <cuda_runtime.h>

#include "benzina/benzina.h"
#include "kernels.h"


/* Defines */



/* CUDA kernels */

/**
 * @brief CUDA post-processing kernel
 */

BENZINA_PLUGIN_HIDDEN __global__ void nvdecodePostprocKernel(void*    dstPtr,
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
	unsigned ix = blockDim.x*blockIdx.x + threadIdx.x;
	unsigned iy = blockDim.y*blockIdx.y + threadIdx.y;
	if(ix >= dstW || iy >= dstH){return;}
	float*         dstPtrR = (float*)        dstPtr + 0*dstH*dstW + iy*dstW     + ix;
	float*         dstPtrG = (float*)        dstPtr + 1*dstH*dstW + iy*dstW     + ix;
	float*         dstPtrB = (float*)        dstPtr + 2*dstH*dstW + iy*dstW     + ix;
	unsigned char* srcY    = (unsigned char*)srcPtr               + iy*srcPitch + ix;
	*dstPtrR = *srcY;
	*dstPtrG = *srcY;
	*dstPtrB = *srcY;
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
