/* Includes */
#include <cuda.h>
#include <cuda_runtime.h>

#include "benzina/benzina.h"
#include "kernels.h"


/* Defines */



/* CUDA kernels */

/**
 * @brief CUDA post-processing kernel
 * @param x
 */

BENZINA_PLUGIN_HIDDEN __global__ void nvdecodePostprocKernel(float* x){
	
}

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
	dim3 Dg = {1,1,1}, Db = {1,1,1};
	nvdecodePostprocKernel<<<Dg, Db, 0, cudaStream>>>((float*)dstPtr);
	return 0;
}
