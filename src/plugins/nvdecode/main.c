/* Includes */
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <dynlink_cuviddec.h>
#include <dynlink_nvcuvid.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "benzina/benzina.h"
#include "benzina/plugins/nvdecode.h"
#include "kernels.h"


/* Defines */



/* Data Structures Forward Declarations and Typedefs */
typedef enum   NVDECODE_STATUS NVDECODE_STATUS;
typedef struct NVDECODE_RQ     NVDECODE_RQ;
typedef struct NVDECODE_BATCH  NVDECODE_BATCH;
typedef struct NVDECODE_CTX    NVDECODE_CTX;



/* Data Structure & Enum Definitions */

/**
 * @brief A structure containing the parameters and status of an individual
 *        request for image loading.
 */

struct NVDECODE_RQ{
	uint64_t        datasetIndex;/* Dataset index. */
	float           H[3][3];     /* Homography */
	float           B   [3];     /* Bias */
	float           S   [3];     /* Scale */
	float           OOB [3];     /* Out-of-bond color */
	uint32_t        colorMatrix; /* Color matrix selection */
	CUVIDPICPARAMS* picParams;   /* Picture parameters; From data.nvdecode file. */
	uint8_t*        data;        /* Image payload;      From data.bin. */
};

/**
 * @brief A structure containing batch status data.
 */

struct NVDECODE_BATCH{
	uint64_t size;
	uint64_t completed;
	uint64_t startIndex;
};

/**
 * @brief The NVDECODE context struct.
 * 
 * Terminology:
 * 
 *   - Contex: This structure. Manages a pipeline of image decoding.
 *   - Job:    A unit of work comprising a compressed image read, its decoding
 *             and postprocessing.
 *   - Batch:  A group of jobs.
 *   - Lock:   The context's Big Lock, controlling access to everything.
 *             Must NOT be held more than momentarily.
 */

struct NVDECODE_CTX{
	/**
	 * All-important dataset
	 */
	
	const BENZINA_DATASET* dataset;
	int                    datasetFd;
	int                    datasetAuxFd;
	
	/**
	 * Status & Reference Count
	 */
	
	uint64_t        refCnt;
	NVDECODE_BATCH* batch;
	NVDECODE_RQ*    request;
	
	/**
	 * Threaded Pipeline.
	 */
	
	int             threadsRunning;
	int             threadsStopping;
	pthread_mutex_t lock;
	uint64_t        cntSubmitted;
	pthread_cond_t  condSubmitted;
	struct{
		pthread_t thrd;
		int       err;
	} reader;
	uint64_t        cntRead;
	pthread_cond_t  condRead;
	struct{
		pthread_t thrd;
		int       err;
	} feeder;
	uint64_t        cntFed;
	pthread_cond_t  condFed;
	struct{
		pthread_t thrd;
		int       err;
	} worker;
	uint64_t        cntDecoded;
	cudaStream_t    cudaStream;
	uint64_t        cntCompleted;
	pthread_cond_t  condCompleted;
	uint64_t        cntAcknowledged;
	uint64_t        cntBatch;
	pthread_cond_t  condBatch;
	
	/* Tensor geometry */
	int      deviceOrd;/* Ordinal number of device. */
	void*    outputPtr;
	uint64_t multibuffering;
	uint64_t batchSize;
	uint64_t totalSlots;
	uint64_t outputHeight;
	uint64_t outputWidth;
	
	/* Default image transform parameters */
	struct{
		float    B  [3];/* Bias */
		float    S  [3];/* Scale */
		float    OOB[3];/* Out-of-bond color. */
		uint32_t colorMatrix;
	} defaults;
	
	/* NVDECODE state */
	void*                  cuvidHandle;
	tcuvidGetDecoderCaps*  cuvidGetDecoderCaps;
	tcuvidCreateDecoder*   cuvidCreateDecoder;
	tcuvidDecodePicture*   cuvidDecodePicture;
	tcuvidMapVideoFrame*   cuvidMapVideoFrame;
	tcuvidUnmapVideoFrame* cuvidUnmapVideoFrame;
	tcuvidDestroyDecoder*  cuvidDestroyDecoder;
	CUVIDDECODECAPS        decoderCaps;
	CUVIDDECODECREATEINFO  decoderInfo;
	CUvideodecoder         decoder;
	CUVIDPICPARAMS*        picParams;
	uint64_t               picParamTruncLen;
};



/* Static Function Prototypes */
BENZINA_PLUGIN_STATIC int   nvdecodeStartHelpers      (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeStopHelpers       (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdInit    (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdContinue(NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC void* nvdecodeReaderThrdMain    (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdInit    (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdContinue(NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC void* nvdecodeFeederThrdMain    (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdInit    (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdContinue(NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC void* nvdecodeWorkerThrdMain    (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC void  nvdecodeWorkerThrdCallback(cudaStream_t  stream,
                                                       cudaError_t   status,
                                                       NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeSetDevice         (NVDECODE_CTX* ctx, const char* deviceId);




/* Static Function Definitions */

/**
 * @brief Launch helper threads.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx
 * @return 0 if threads already running or started successfully.
 *         !0 if threads not already running and could not be started.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeStartHelpers      (NVDECODE_CTX* ctx){
	pthread_attr_t attr;
	int ret = 0;
	
	if(ctx->threadsRunning){return 0;}
	
	ctx->threadsRunning  = 0;
	ctx->threadsStopping = 0;
	ctx->cntSubmitted    = 0;
	ctx->cntRead         = 0;
	ctx->cntFed          = 0;
	ctx->cntDecoded      = 0;
	ctx->cntCompleted    = 0;
	ctx->cntAcknowledged = 0;
	ctx->cntBatch        = 0;
	ctx->picParams       = calloc(ctx->totalSlots, sizeof(*ctx->picParams));
	ret |= pthread_attr_init(&attr);
	ret |= pthread_attr_setstacksize(&attr, 64*1024);
	ret |= pthread_create(&ctx->reader.thrd, &attr, (void*)nvdecodeReaderThrdMain, ctx);
	ret |= pthread_create(&ctx->feeder.thrd, &attr, (void*)nvdecodeFeederThrdMain, ctx);
	ret |= pthread_create(&ctx->worker.thrd, &attr, (void*)nvdecodeWorkerThrdMain, ctx);
	ret |= pthread_attr_destroy(&attr);
	ctx->threadsRunning  = !ret;
	
	return !ctx->threadsRunning;
}

/**
 * @brief Stop helper threads.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx
 * @return 0 if threads not running or successfully stopped.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeStopHelpers       (NVDECODE_CTX* ctx){
	int ret = 0;
	
	if(!ctx->threadsRunning){return 0;}
	
	ctx->threadsStopping = 1;
	ret |= pthread_cond_broadcast(&ctx->condSubmitted);
	ret |= pthread_cond_broadcast(&ctx->condRead);
	ret |= pthread_cond_broadcast(&ctx->condFed);
	ret |= pthread_cond_broadcast(&ctx->condCompleted);
	ret |= pthread_cond_broadcast(&ctx->condBatch);
	ret |= pthread_mutex_unlock(&ctx->lock);
	ret |= pthread_join(ctx->reader.thrd, 0);
	ret |= pthread_join(ctx->feeder.thrd, 0);
	ret |= pthread_join(ctx->worker.thrd, 0);
	ret |= pthread_mutex_lock(&ctx->lock);
	ctx->threadsRunning  = 0;
	ctx->threadsStopping = 0;
	
	return ret;
}

/**
 * @brief Main routine of the reader thread.
 * 
 * Does I/O as and when jobs are submitted, asynchronously from decoder thread.
 * For every job submitted, two reads are performed:
 *   - On data.bin,      for the image data payload.
 *   - On data.nvdecode, for the precomputed decode parameters.
 * 
 * @param [in]  ctx  The decoding context.
 * @return NULL.
 */

BENZINA_PLUGIN_STATIC void* nvdecodeReaderThrdMain    (NVDECODE_CTX* ctx){
	NVDECODE_RQ* rq;
	uint64_t     dr, di;
	int          readsDone;
	struct{
		int fd; size_t off; size_t len; void* ptr; ssize_t lenRead;
	} rd0 = {0}, rd1 = {0};
	
	/**
	 * Initialization.
	 * 
	 * The reader thread sets for itself the selected device.
	 */
	
	pthread_mutex_lock(&ctx->lock);
	if(nvdecodeReaderThrdInit(ctx)){
		while(nvdecodeReaderThrdContinue(ctx)){
			/**
			 * Obtain a handle for the request being processed right now. Extract
			 * from it the parameters for the I/O read requests we must satisfy.
			 */
			
			dr      = ctx->cntRead % ctx->totalSlots;
			rq      = &ctx->request[dr];
			di      = rq->datasetIndex;
			rd0.fd  = ctx->datasetFd;
			if(benzinaDatasetGetElement(ctx->dataset, di, &rd0.off, &rd0.len) != 0){
				/* Error! How to deal with it? */
			}
			rd0.ptr = malloc(rd0.len);
			if(!rd0.ptr){
				/* Error! How to deal with it? */
			}
			rd1.fd  = ctx->datasetAuxFd;
			rd1.len = ctx->picParamTruncLen;
			rd1.off = ctx->picParamTruncLen*dr;
			rd1.ptr = &ctx->picParams[dr];
			
			
			/**
			 * Drop lock for blocking reads, then reacquire the lock.
			 * Allows other work to proceed in parallel.
			 */
			
			pthread_mutex_unlock(&ctx->lock);
			rd0.lenRead = pread(rd0.fd, rd0.ptr, rd0.len, rd0.off);
			rd1.lenRead = pread(rd1.fd, rd1.ptr, rd1.len, rd1.off);
			pthread_mutex_lock(&ctx->lock);
			
			
			/**
			 * If the reads were successful, increment counter. Otherwise, indicate
			 * error. Either way, signal everyone.
			 */
			
			readsDone = rd0.lenRead==(ssize_t)rd0.len ||
			            rd1.lenRead==(ssize_t)rd1.len;
			if(readsDone){
				ctx->cntRead++;
			}else{
				ctx->reader.err |= 1;
			}
			pthread_cond_broadcast(&ctx->condRead);
		}
	}
	pthread_mutex_unlock(&ctx->lock);
	pthread_exit(NULL);
}

/**
 * @brief Initialize reader thread state.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The context whose reader thread is initializing.
 * @return Whether (!0) or not (0) initialization was successful.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdInit    (NVDECODE_CTX* ctx){
	return !!ctx;
}

/**
 * @brief Determine whether the reader thread should shut down or do more work.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The
 * @return Whether (!0) or not (0) there is work to do.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdContinue(NVDECODE_CTX* ctx){
	do{
		if(ctx->cntRead == ctx->cntSubmitted){
			if(ctx->threadsStopping){
				break;   /* FIXME: Exit path. */
			}else{
				continue;/* FIXME: No work to do; First entry, or spurious wakeup */
			}
		}
		return 1;
	}while(pthread_cond_wait(&ctx->condSubmitted, &ctx->lock) == 0);
	return 0;
}

/**
 * @brief Main routine of the feeder thread.
 * 
 * Feeds the data read by the reader thread into the decoders.
 * 
 * @param [in]  ctx  The decoding context.
 * @return NULL.
 */

BENZINA_PLUGIN_STATIC void* nvdecodeFeederThrdMain    (NVDECODE_CTX* ctx){
	uint64_t dr = 0;
	
	pthread_mutex_lock(&ctx->lock);
	if(nvdecodeFeederThrdInit(ctx)){
		while(nvdecodeFeederThrdContinue(ctx)){
			
			
			pthread_mutex_unlock(&ctx->lock);
			ctx->cuvidDecodePicture(ctx->decoder, &ctx->picParams[dr]);
			pthread_mutex_lock(&ctx->lock);
			ctx->cntFed++;
			pthread_cond_broadcast(&ctx->condFed);
		}
	}
	pthread_mutex_unlock(&ctx->lock);
	pthread_exit(NULL);
}

/**
 * @brief Initialize feeder thread state.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The context whose feeder thread is initializing.
 * @return Whether (!0) or not (0) initialization was successful.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdInit    (NVDECODE_CTX* ctx){
	if(cudaSetDevice(ctx->deviceOrd) != cudaSuccess){
		ctx->feeder.err |= 1;
		return 0;
	}
	
	/***********************************************************************************************************/
	//! VIDEO_DECODER
	//!
	//! In order to minimize decode latencies, there should be always at least 2 pictures in the decode
	//! queue at any time, in order to make sure that all decode engines are always busy.
	//!
	//! Overall data flow:
	//!  - cuvidGetDecoderCaps(...)
	//!  - cuvidCreateDecoder(...)
	//!  - For each picture:
	//!    + cuvidDecodePicture(N)
	//!    + cuvidMapVideoFrame(N-4)
	//!    + do some processing in cuda
	//!    + cuvidUnmapVideoFrame(N-4)
	//!    + cuvidDecodePicture(N+1)
	//!    + cuvidMapVideoFrame(N-3)
	//!    + ...
	//!  - cuvidDestroyDecoder(...)
	//!
	//! NOTE:
	//! - When the cuda context is created from a D3D device, the D3D device must also be created
	//!   with the D3DCREATE_MULTITHREADED flag.
	//! - There is a limit to how many pictures can be mapped simultaneously (ulNumOutputSurfaces)
	//! - cuvidDecodePicture may block the calling thread if there are too many pictures pending
	//!   in the decode queue
	/***********************************************************************************************************/
	
	memset(&ctx->decoderCaps, 0, sizeof(ctx->decoderCaps));
	ctx->decoderCaps.nBitDepthMinus8 = 0;
	ctx->decoderCaps.eChromaFormat   = cudaVideoChromaFormat_420;
	ctx->decoderCaps.eCodecType      = cudaVideoCodec_H264;
	if(ctx->cuvidGetDecoderCaps(&ctx->decoderCaps) != CUDA_SUCCESS){
		ctx->feeder.err |= 1;
		return 0;
	}
	if(!ctx->decoderCaps.bIsSupported){
		ctx->feeder.err |= 1;
		return 0;
	}
	
	if(ctx->cuvidCreateDecoder(&ctx->decoder, &ctx->decoderInfo) != CUDA_SUCCESS){
		ctx->feeder.err |= 1;
		return 0;
	}
	
	return 1;
}

/**
 * @brief Determine whether the feeder thread should shut down or do more work.
 * 
 * @param [in]  ctx  The
 * @return Whether (!0) or not (0) there is work to do.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdContinue(NVDECODE_CTX* ctx){
	do{
		if(ctx->cntFed == ctx->cntRead){
			if(ctx->threadsStopping){
				break;   /* FIXME: Exit path. */
			}else{
				continue;/* FIXME: No work to do; First entry, or spurious wakeup */
			}
		}
		return 1;
	}while(pthread_cond_wait(&ctx->condRead, &ctx->lock) == 0);
	return 0;
}

/**
 * @brief Main routine of the worker thread.
 * 
 * Accepts the data payloads
 * 
 * @param [in]  ctx  The decoding context.
 * @return NULL.
 */

BENZINA_PLUGIN_STATIC void* nvdecodeWorkerThrdMain    (NVDECODE_CTX* ctx){
	CUVIDPROCPARAMS    vpp;
	unsigned long long devPtr;
	unsigned           pitch;
	uint64_t           picIdx = 0;
	
	pthread_mutex_lock(&ctx->lock);
	if(nvdecodeWorkerThrdInit(ctx)){
		while(nvdecodeWorkerThrdContinue(ctx)){
			pthread_mutex_unlock(&ctx->lock);
			vpp.output_stream = ctx->cudaStream;
			ctx->cuvidMapVideoFrame(ctx->decoder, picIdx, &devPtr, &pitch, &vpp);
			nvdecodePostprocKernelInvoker(/* ctx->cudaStream */);
			cudaStreamAddCallback(ctx->cudaStream,
			                      (cudaStreamCallback_t)nvdecodeWorkerThrdCallback,
			                      ctx,
			                      0);
			ctx->cuvidUnmapVideoFrame(ctx->decoder, devPtr);
			pthread_mutex_lock(&ctx->lock);
		}
	}
	pthread_mutex_unlock(&ctx->lock);
	pthread_exit(NULL);
}

/**
 * @brief Initialize worker thread state.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The context whose worker thread is initializing.
 * @return Whether (!0) or not (0) initialization was successful.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdInit    (NVDECODE_CTX* ctx){
	if(cudaSetDevice(ctx->deviceOrd) != cudaSuccess){
		ctx->worker.err |= 1;
		return 0;
	}
	
	return 1;
}

/**
 * @brief Determine whether the worker thread should shut down or do more work.
 * 
 * @param [in]  ctx  The
 * @return Whether (!0) or not (0) there is work to do.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdContinue(NVDECODE_CTX* ctx){
	do{
		if(ctx->cntDecoded == ctx->cntFed){
			if(ctx->threadsStopping){
				break;   /* FIXME: Exit path. */
			}else{
				continue;/* FIXME: No work to do; First entry, or spurious wakeup */
			}
		}
		return 1;
	}while(pthread_cond_wait(&ctx->condFed, &ctx->lock) == 0);
	return 0;
}

/**
 * @brief Post-processing Callback
 * @param [in]   stream The stream onto which this callback had been scheduled.
 * @param [in]   status The error status of this device or stream.
 * @param [in]   ctx    The context on which this callback is being executed.
 * @return 
 */

BENZINA_PLUGIN_STATIC void  nvdecodeWorkerThrdCallback(cudaStream_t  stream,
                                                       cudaError_t   status,
                                                       NVDECODE_CTX* ctx){
	(void)stream;
	
	if(status == cudaSuccess){
		pthread_mutex_lock(&ctx->lock);
		ctx->cntCompleted++;
		pthread_cond_broadcast(&ctx->condCompleted);
		if(0){
			pthread_cond_broadcast(&ctx->condBatch);
		}
		pthread_mutex_unlock(&ctx->lock);
	}else{
		
	}
}

/**
 * @brief Set the device this context will use.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx       The context for which the device is to be set.
 * @param [in]  deviceId  A string identifying uniquely the device to be used.
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeSetDevice         (NVDECODE_CTX* ctx, const char*   deviceId){
	int ret, deviceCount=0, i=-1;
	char* s;
	struct cudaDeviceProp prop;
	
	
	/* Forbid changing device ordinal while threads running. */
	if(ctx->threadsRunning){return BENZINA_DATALOADER_ITER_ALREADYINITED;}
	
	
	/* Determine maximum device ordinal. */
	ret = cudaGetDeviceCount(&deviceCount);
	if(ret != cudaSuccess){return ret;}
	
	
	/* Select a device ordinal i by one of several identification string schemes. */
	if      (strncmp(deviceId, "cuda:", strlen("cuda:")) == 0){
		if(deviceId[strlen("cuda:")] == '\0'){
			return BENZINA_DATALOADER_ITER_INVALIDARGS;
		}
		i = strtoull(deviceId, &s, 10);
		if(*s != '\0')      {return BENZINA_DATALOADER_ITER_INVALIDARGS;}
		if(i >= deviceCount){return BENZINA_DATALOADER_ITER_INVALIDARGS;}
	}else if(strncmp(deviceId, "pci:",  strlen("pci:"))  == 0){
		if(cudaDeviceGetByPCIBusId(&i, deviceId+strlen("pci:")) != cudaSuccess){
			return BENZINA_DATALOADER_ITER_INVALIDARGS;
		}
	}else{
		return BENZINA_DATALOADER_ITER_INVALIDARGS;
	}
	
	
	/**
	 * Verify that the device satisfies several important requirements by
	 * inspecting its properties.
	 * 
	 * In particular, we require an NVDECODE engine, which is available only on
	 * compute-capability 3.0 and up devices, and compute-mode access from
	 * multiple host threads.
	 */
	
	if(cudaGetDeviceProperties(&prop, i) != cudaSuccess){
		return BENZINA_DATALOADER_ITER_INTERNAL;
	}
	if(prop.major        < 3                         ||
	   prop.computeMode == cudaComputeModeProhibited ||
	   prop.computeMode == cudaComputeModeExclusive){
		return BENZINA_DATALOADER_ITER_INVALIDARGS;
	}
	
	
	/* We accept the device ordinal. */
	ctx->deviceOrd = i;
	return 0;
}



/* Function Definitions */

/**
 * @brief Allocate iterator context from dataset.
 * @param [in]  dataset  The dataset over which this iterator will iterate.
 *                       Must be non-NULL and compatible.
 * @return A pointer to the context, if successful; NULL otherwise.
 */

BENZINA_PLUGIN_HIDDEN NVDECODE_CTX* nvdecodeAlloc(const BENZINA_DATASET* dataset){
	NVDECODE_CTX* ctx = NULL;
	
	if(!dataset){
		return NULL;
	}
	
	ctx = calloc(1, sizeof(*ctx));
	if(!ctx){
		return NULL;
	}
	
	if(pthread_mutex_init(&ctx->lock, NULL)      ){goto fail_lock;}
	if(pthread_cond_init (&ctx->condSubmitted, 0)){goto fail_submitted;}
	if(pthread_cond_init (&ctx->condRead,      0)){goto fail_read;}
	if(pthread_cond_init (&ctx->condFed,       0)){goto fail_fed;}
	if(pthread_cond_init (&ctx->condCompleted, 0)){goto fail_completed;}
	if(pthread_cond_init (&ctx->condBatch,     0)){goto fail_batch;}
	
	ctx->dataset   =  dataset;
	ctx->refCnt    =  1;
	ctx->deviceOrd = -1;
	
	return ctx;
	
	                pthread_cond_destroy (&ctx->condBatch);
	fail_batch:     pthread_cond_destroy (&ctx->condCompleted);
	fail_completed: pthread_cond_destroy (&ctx->condFed);
	fail_fed:       pthread_cond_destroy (&ctx->condRead);
	fail_read:      pthread_cond_destroy (&ctx->condSubmitted);
	fail_submitted: pthread_mutex_destroy(&ctx->lock);
	fail_lock:      free(ctx);
	return NULL;
}

/**
 * @brief Initialize iterator context using its current properties.
 * 
 * The current properties of the iterator will be frozen and will be
 * unchangeable afterwards.
 * 
 * @param [in]  ctx  The iterator context to initialize.
 * @return 0 if successful in initializing; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodeInit(NVDECODE_CTX* ctx){
	return nvdecodeStartHelpers(ctx);
}

/**
 * @brief Increase reference count of the iterator.
 * 
 * @param [in]  ctx  The iterator context whose reference-count is to be increased.
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodeRetain(NVDECODE_CTX* ctx){
	if(!ctx){return 0;}
	
	pthread_mutex_lock(&ctx->lock);
	ctx->refCnt++;
	pthread_mutex_unlock(&ctx->lock);
	
	return 0;
}

/**
 * @brief Decrease reference count of the iterator. Destroy iterator if its
 *        reference count drops to 0.
 * 
 * @param [in]  ctx  The iterator context whose reference-count is to be decreased.
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodeRelease(NVDECODE_CTX* ctx){
	if(!ctx){return 0;}
	
	pthread_mutex_lock(&ctx->lock);
	if(--ctx->refCnt > 0){
		pthread_mutex_unlock(&ctx->lock);
		return 0;
	}
	
	/**
	 * FIXME:
	 * 
	 * At this present time the mutex is held, but the reference count is 0.
	 * This makes us responsible for the destruction of the object.
	 * We must do the following:
	 * 
	 *   - Signal all helper threads to shut down.
	 *   - Block on thread completion
	 *   - Deallocate resources
	 *   - Free this pointer
	 */
	
	nvdecodeStopHelpers(ctx);
	pthread_mutex_unlock(&ctx->lock);
	
	pthread_cond_destroy (&ctx->condBatch);
	pthread_cond_destroy (&ctx->condCompleted);
	pthread_cond_destroy (&ctx->condFed);
	pthread_cond_destroy (&ctx->condRead);
	pthread_cond_destroy (&ctx->condSubmitted);
	pthread_mutex_destroy(&ctx->lock);
	
	memset(ctx, 0, sizeof(*ctx));
	free(ctx);
	
	return 0;
}

/**
 * @brief Close and push a batch of work into the pipeline.
 * 
 * @param [in]  ctx    The iterator context in which.
 * @param [in]  token  User data that will be retrieved at the corresponding
 *                     pullBatch().
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodePushBatch(NVDECODE_CTX* ctx, const void* token){
	(void)ctx;
	(void)token;
	return 0;
}

/**
 * @brief Pull a completed batch of work from the pipeline.
 * 
 * @param [in]  ctx      The iterator context in which.
 * @param [out] token    User data that was submitted at the corresponding
 *                       pushBatch().
 * @param [in]  timeout  A maximum amount of time to wait for the batch of data,
 *                       in seconds. If timeout <= 0, wait indefinitely.
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodePullBatch(NVDECODE_CTX* ctx, const void** token, double timeout){
	(void)ctx;
	(void)token;
	(void)timeout;
	return 0;
}

/**
 * @brief Stop the pipeline.
 * 
 * Work already in the pipeline completes, but no more may be submitted.
 * 
 * @param [in]  ctx
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodeStop(NVDECODE_CTX* ctx){
	pthread_mutex_lock(&ctx->lock);
	pthread_mutex_unlock(&ctx->lock);
	return 0;
}

/**
 * @brief Is the iterator in an abnormal, error condition?
 * 
 * @param [in]  ctx
 * @return 0 if no errors; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodeHasError(NVDECODE_CTX* ctx){
	pthread_mutex_lock(&ctx->lock);
	pthread_mutex_unlock(&ctx->lock);
	return 0;
}

/**
 * @brief Begin defining a new job.
 * 
 * @param [in]  ctx  
 * @param [in]  i    Index into dataset.
 * @return 0 if no errors; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodeDefineJob(NVDECODE_CTX* ctx, uint64_t i){
	pthread_mutex_lock(&ctx->lock);
	pthread_mutex_unlock(&ctx->lock);
	(void)i;
	return 0;
}

/**
 * @brief Submit current job.
 * 
 * @param [in]  ctx
 * @return 0 if no errors; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodeSubmitJob(NVDECODE_CTX* ctx){
	pthread_mutex_lock(&ctx->lock);
	pthread_mutex_unlock(&ctx->lock);
	return 0;
}

/**
 * @brief Set Buffer and Geometry.
 * 
 * Provide to the context the target output buffer it should use.
 * 
 * @param [in]  ctx
 * @param [in]  deviceId
 * @param [in]  devicePtr
 * @param [in]  multibuffering
 * @param [in]  batchSize
 * @param [in]  outputHeight
 * @param [in]  outputWidth
 * @return Zero if successful; Non-zero otherwise.
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodeSetBuffer(NVDECODE_CTX* ctx,
                                              const char*   deviceId,
                                              void*         outputPtr,
                                              uint32_t      multibuffering,
                                              uint32_t      batchSize,
                                              uint32_t      outputHeight,
                                              uint32_t      outputWidth){
	int ret;
	
	pthread_mutex_lock(&ctx->lock);
	if(ctx->threadsRunning){
		ret = BENZINA_DATALOADER_ITER_ALREADYINITED;
	}else if(!outputPtr){
		ret = BENZINA_DATALOADER_ITER_INVALIDARGS;
	}else{
		ret = nvdecodeSetDevice(ctx, deviceId);
		if(ret == 0){
			ctx->outputPtr      = outputPtr;
			ctx->multibuffering = multibuffering;
			ctx->batchSize      = batchSize;
			ctx->totalSlots     = ctx->multibuffering*ctx->batchSize;
			ctx->outputHeight   = outputHeight;
			ctx->outputWidth    = outputWidth;
			ret = BENZINA_DATALOADER_ITER_SUCCESS;
		}
	}
	pthread_mutex_unlock(&ctx->lock);
	
	return ret;
}



BENZINA_PLUGIN_HIDDEN int   nvdecodeSetDefaultBias          (NVDECODE_CTX* ctx,
                                                             float*        B){
	memcpy(ctx->defaults.B, B, sizeof(ctx->defaults.B));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int   nvdecodeSetDefaultScale         (NVDECODE_CTX* ctx,
                                                             float*        S){
	memcpy(ctx->defaults.S, S, sizeof(ctx->defaults.S));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int   nvdecodeSetDefaultOOBColor      (NVDECODE_CTX* ctx,
                                                             float*        OOB){
	memcpy(ctx->defaults.OOB, OOB, sizeof(ctx->defaults.OOB));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int   nvdecodeSelectDefaultColorMatrix(NVDECODE_CTX* ctx,
                                                             uint32_t      colorMatrix){
	ctx->defaults.colorMatrix = colorMatrix;
	return 0;
}
BENZINA_PLUGIN_HIDDEN int   nvdecodeSetHomography           (NVDECODE_CTX* ctx,
                                                             float*        H){
	(void)ctx;
	(void)H;
	//memcpy(ctx->defaults.H, H, sizeof(ctx->defaults.H));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int   nvdecodeSetBias                 (NVDECODE_CTX* ctx,
                                                             float*        B){
	(void)ctx;
	(void)B;
	//memcpy(ctx->defaults.B, B, sizeof(ctx->defaults.B));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int   nvdecodeSetScale                (NVDECODE_CTX* ctx,
                                                             float*        S){
	(void)ctx;
	(void)S;
	//memcpy(ctx->defaults.S, S, sizeof(ctx->defaults.S));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int   nvdecodeSetOOBColor             (NVDECODE_CTX* ctx,
                                                             float*        OOB){
	(void)ctx;
	(void)OOB;
	//memcpy(ctx->defaults.OOB, OOB, sizeof(ctx->defaults.OOB));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int   nvdecodeSelectColorMatrix       (NVDECODE_CTX* ctx,
                                                             uint32_t      colorMatrix){
	(void)ctx;
	(void)colorMatrix;
	//ctx->defaults.colorMatrix = colorMatrix;
	return 0;
}



/**
 * Exported Function Table.
 */

BENZINA_PLUGIN_PUBLIC BENZINA_PLUGIN_NVDECODE_VTABLE VTABLE = {
	.alloc                    = (void*)nvdecodeAlloc,
	.init                     = (void*)nvdecodeInit,
	.retain                   = (void*)nvdecodeRetain,
	.release                  = (void*)nvdecodeRelease,
	.pushBatch                = (void*)nvdecodePushBatch,
	.pullBatch                = (void*)nvdecodePullBatch,
	.stop                     = (void*)nvdecodeStop,
	.hasError                 = (void*)nvdecodeHasError,
	
	.defineJob                = (void*)nvdecodeDefineJob,
	.submitJob                = (void*)nvdecodeSubmitJob,
	
	.setBuffer                = (void*)nvdecodeSetBuffer,
	
	.setDefaultBias           = (void*)nvdecodeSetDefaultBias,
	.setDefaultScale          = (void*)nvdecodeSetDefaultScale,
	.setDefaultOOBColor       = (void*)nvdecodeSetDefaultOOBColor,
	.selectDefaultColorMatrix = (void*)nvdecodeSelectDefaultColorMatrix,
	
	.setHomography            = (void*)nvdecodeSetHomography,
	.setBias                  = (void*)nvdecodeSetBias,
	.setScale                 = (void*)nvdecodeSetScale,
	.setOOBColor              = (void*)nvdecodeSetOOBColor,
	.selectColorMatrix        = (void*)nvdecodeSelectColorMatrix,
};
