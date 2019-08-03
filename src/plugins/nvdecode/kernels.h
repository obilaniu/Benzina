/* Include Guard */
#ifndef SRC_PLUGINS_NVDECODE_KERNELS_H
#define SRC_PLUGINS_NVDECODE_KERNELS_H

/**
 * Includes
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "benzina/benzina-old.h"



/* Defines */




/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif



/* Function Prototypes */
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
                                                          unsigned     srcW);



/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

