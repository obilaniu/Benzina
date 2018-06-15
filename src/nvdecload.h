/* Include Guard */
#ifndef SRC_NVDECLOAD_H
#define SRC_NVDECLOAD_H

/**
 * Includes
 */

#include <stdint.h>
#include <cuda.h>
#include <semaphore.h>
#include "benzina/benzina.h"



/* Defines */




/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif



/* Data Structures Forward Declarations and Typedefs */
typedef struct NVDECLOAD_PCB NVDECLOAD_PCB;
typedef struct NVDECLOAD_RQ  NVDECLOAD_RQ;

/**
 * @brief NVDEC Loader Process Control Block.
 * 
 * The shared-memory process control block for communication between the parent
 * and worker process.
 */

struct NVDECLOAD_PCB{
	uint64_t       version;
	uint64_t       status;
	uint64_t       pcbSize;
	uint64_t       pageSize;
	uint32_t       batchSize;
	uint32_t       multiBuffering;
	uint32_t       targetWidth;
	uint32_t       targetHeight;
	int32_t        parentPID;
	int32_t        workerPID;
	uint8_t        workerUUID[16];
	CUuuid         deviceUUID;
	CUipcMemHandle deviceMemHandle;
	uint64_t       deviceMemOffset;
	uint64_t       metaOffset;
	sem_t          masterSem;
};

/**
 * @brief NVDEC Request.
 */

struct NVDECLOAD_RQ{
	uint64_t dsIndex;
	uint64_t memOffset;
	uint32_t batchId;
	uint32_t colorMatrix;
	float    H[3][3];
	float    B[3];
	float    OOBColor[3];
};



/* Function Prototypes */




/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

