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

extern "C" BENZINA_PLUGIN_HIDDEN void nvdecodePostprocKernelInvoker(void){
	dim3 Dg = {1,1,1}, Db = {1,1,1};
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	nvdecodePostprocKernel<<<Dg, Db, 0, stream>>>(NULL);
}
