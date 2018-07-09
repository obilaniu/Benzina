/* Include Guard */
#ifndef INCLUDE_BENZINA_PLUGINS_NVDECODE_H
#define INCLUDE_BENZINA_PLUGINS_NVDECODE_H

/**
 * Includes
 */

#include <stddef.h>
#include <stdint.h>
#include "benzina/benzina.h"



/* Defines */
#define BENZINA_DATALOADER_ITER_SUCCESS       0
#define BENZINA_DATALOADER_ITER_FAILED        1
#define BENZINA_DATALOADER_ITER_INITFAILED    2
#define BENZINA_DATALOADER_ITER_ALREADYINITED 3
#define BENZINA_DATALOADER_ITER_INVALIDARGS   4
#define BENZINA_DATALOADER_ITER_INTERNAL      5
#define BENZINA_DATALOADER_ITER_OVERFLOW      6
#define BENZINA_DATALOADER_ITER_STOPPED       7



/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif



/* Data Structures Forward Declarations and Typedefs */
typedef struct BENZINA_PLUGIN_NVDECODE_VTABLE BENZINA_PLUGIN_NVDECODE_VTABLE;



/* Plugin Function Table */
struct BENZINA_PLUGIN_NVDECODE_VTABLE{
	int (*alloc)                   (void** ctx, const BENZINA_DATASET* dataset);
	int (*init)                    (void*  ctx);
	int (*retain)                  (void*  ctx);
	int (*release)                 (void*  ctx);
	int (*defineBatch)             (void*  ctx);
	int (*submitBatch)             (void*  ctx, const void*  tokenIn);
	int (*waitBatch)               (void*  ctx, const void** tokenOut, int block, double timeout);
	
	int (*defineSample)            (void*  ctx, uint64_t datasetIndex, void* dstPtr);
	int (*submitSample)            (void*  ctx);
	
	int (*setBuffer)               (void*  ctx,
	                                const char* deviceId,
	                                void*       devicePtr,
	                                uint32_t    multibuffering,
	                                uint32_t    batchSize,
	                                uint32_t    outputHeight,
	                                uint32_t    outputWidth);
	
	int (*setDefaultBias)          (void*  ctx, const float* B);
	int (*setDefaultScale)         (void*  ctx, const float* S);
	int (*setDefaultOOBColor)      (void*  ctx, const float* OOB);
	int (*selectDefaultColorMatrix)(void*  ctx, uint32_t     matrix);
	
	int (*setHomography)           (void*  ctx, const float* H);
	int (*setBias)                 (void*  ctx, const float* B);
	int (*setScale)                (void*  ctx, const float* S);
	int (*setOOBColor)             (void*  ctx, const float* OOB);
	int (*selectColorMatrix)       (void*  ctx, uint32_t     matrix);
};


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

