/* Include Guard */
#ifndef INCLUDE_BENZINA_PLUGINS_NVDECODE_H
#define INCLUDE_BENZINA_PLUGINS_NVDECODE_H

/**
 * Includes
 */

#include <stddef.h>
#include <stdint.h>
#include <errno.h>
#include "benzina/benzina-old.h"



/* Defines */
#define BENZINA_DATALOADER_ITER_SUCCESS       0
#define BENZINA_DATALOADER_ITER_INTR          EINTR
#define BENZINA_DATALOADER_ITER_IO            EIO
#define BENZINA_DATALOADER_ITER_NOMEM         ENOMEM
#define BENZINA_DATALOADER_ITER_FAILED        10001
#define BENZINA_DATALOADER_ITER_INITFAILED    10002
#define BENZINA_DATALOADER_ITER_ALREADYINITED 10003
#define BENZINA_DATALOADER_ITER_INVALIDARGS   10004
#define BENZINA_DATALOADER_ITER_INTERNAL      10005
#define BENZINA_DATALOADER_ITER_OVERFLOW      10006
#define BENZINA_DATALOADER_ITER_STOPPED       10007



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
	int (*startHelpers)            (void*  ctx);
	int (*stopHelpers)             (void*  ctx);
	int (*defineBatch)             (void*  ctx);
	int (*submitBatch)             (void*  ctx, const void*  tokenIn);
	int (*waitBatch)               (void*  ctx, const void** tokenOut, int block, double timeout);
	int (*peekToken)               (void*  ctx, uint64_t i, int clear, const void** tokenOut);
	
	int (*defineSample)            (void*  ctx, uint64_t datasetIndex, void* dstPtr, void* sample, uint64_t* location, uint64_t* config_location);
	int (*submitSample)            (void*  ctx);
	
	int (*getNumPushes)            (void*  ctx, uint64_t* out);
	int (*getNumPulls)             (void*  ctx, uint64_t* out);
	int (*getMultibuffering)       (void*  ctx, uint64_t* out);
	
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

