/* Includes */
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <dynlink_cuviddec.h>
#include <dynlink_nvcuvid.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
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
typedef struct NVDECODE_RQ     NVDECODE_RQ;
typedef struct NVDECODE_BATCH  NVDECODE_BATCH;
typedef struct NVDECODE_CTX    NVDECODE_CTX;



/* Data Structure & Enum Definitions */

/**
 * @brief Thread status.
 */

typedef enum NVDECODE_THRD_STATUS{
	THRD_NOT_RUNNING = 0, /* Thread hasn't been spawned. */
	THRD_SPAWNED,         /* Thread has been spawned with pthread_create(). */
	THRD_INITED,          /* Thread initialized successfully, waiting for others to do so as well. */
	THRD_RUNNING,         /* Thread running. */
	THRD_EXITING,         /* Thread was spawned/running but has now been asked to exit. */
} NVDECODE_THRD_STATUS;

/**
 * @brief A structure containing the parameters and status of an individual
 *        request for image loading.
 */

struct NVDECODE_RQ{
	NVDECODE_BATCH* batch;       /* Batch to which this request belongs. */
	uint64_t        datasetIndex;/* Dataset index. */
	float*          devPtr;      /* Target destination on device. */
	float           H[3][3];     /* Homography */
	float           B   [3];     /* Bias */
	float           S   [3];     /* Scale */
	float           OOB [3];     /* Out-of-bond color */
	uint32_t        colorMatrix; /* Color matrix selection */
	uint8_t*        data;        /* Image payload;      From data.bin. */
	CUVIDPICPARAMS* picParams;   /* Picture parameters; From data.nvdecode. */
};

/**
 * @brief A structure containing batch status data.
 */

struct NVDECODE_BATCH{
	uint64_t    startIndex;
	uint64_t    stopIndex;
	const void* token;
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
	const char*            datasetRoot;
	size_t                 datasetLen;
	int                    datasetBinFd;
	int                    datasetProtobufFd;
	struct stat            datasetProtobufStat;
	int                    datasetNvdecodeFd;
	
	/**
	 * Reference Count
	 */
	
	uint64_t        refCnt;
	
	/**
	 * Threaded Pipeline.
	 */
	
	pthread_mutex_t lock;
	struct{
		uint64_t cntLifecycles;/* # of times helper threads spawned. */
		uint64_t cntSubmitted; /* # of images previously submitted. */
		uint64_t cntCompleted; /* # of images previously completed. */
		uint64_t cntBatchSub;  /* # of image batches previously submitted. */
		uint64_t cntBatchAck;  /* # of image batches previously acknowledged. */
		pthread_cond_t cond;
	} master;
	struct{
		NVDECODE_THRD_STATUS status;
		uint64_t cnt;/* # of images previously read */
		pthread_t thrd;
		pthread_cond_t cond;
	} reader;
	struct{
		NVDECODE_THRD_STATUS status;
		uint64_t cnt;/* # of images previously pushed into decoder */
		pthread_t thrd;
		pthread_cond_t cond;
	} feeder;
	struct{
		NVDECODE_THRD_STATUS status;
		uint64_t cnt;/* # of images previously pulled out of decoder */
		pthread_t thrd;
		pthread_cond_t cond;
		cudaStream_t cudaStream;
	} worker;
	
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
	void*                    cuvidHandle;
	tcuvidGetDecoderCaps*    cuvidGetDecoderCaps;
	tcuvidCreateDecoder*     cuvidCreateDecoder;
	tcuvidDecodePicture*     cuvidDecodePicture;
	tcuvidMapVideoFrame64*   cuvidMapVideoFrame64;
	tcuvidUnmapVideoFrame64* cuvidUnmapVideoFrame64;
	tcuvidDestroyDecoder*    cuvidDestroyDecoder;
	CUVIDDECODECAPS          decoderCaps;
	CUVIDDECODECREATEINFO    decoderInfo;
	CUvideodecoder           decoder;
	uint32_t                 decoderInited;
	uint32_t                 decoderRefCnt;
	CUVIDPICPARAMS*          picParams;
	uint64_t                 picParamTruncLen;
	uint32_t                 mallocRefCnt;
	NVDECODE_BATCH*          batch;
	NVDECODE_RQ*             request;
};



/* Static Function Prototypes */
BENZINA_PLUGIN_STATIC int   nvdecodeAbstime            (struct timespec* ts, double dt);
BENZINA_PLUGIN_STATIC int   nvdecodeMasterThrdGetSubmRq(NVDECODE_CTX* ctx, NVDECODE_RQ**    rqOut);
BENZINA_PLUGIN_STATIC int   nvdecodeMasterThrdGetSubmBt(NVDECODE_CTX* ctx, NVDECODE_BATCH** batchOut);
BENZINA_PLUGIN_STATIC int   nvdecodeMasterThrdGetRetrBt(NVDECODE_CTX* ctx, NVDECODE_BATCH** batchOut);
BENZINA_PLUGIN_STATIC int   nvdecodeHelpersStart       (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeHelpersStop        (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeHelpersAllStatusIs (NVDECODE_CTX* ctx, NVDECODE_THRD_STATUS status);
BENZINA_PLUGIN_STATIC int   nvdecodeHelpersAnyStatusIs (NVDECODE_CTX* ctx, NVDECODE_THRD_STATUS status);
BENZINA_PLUGIN_STATIC int   nvdecodeHelpersFailing     (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdInit     (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdAwaitAll (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdContinue (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdCore     (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC void* nvdecodeReaderThrdMain     (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdSetStatus(NVDECODE_CTX* ctx, NVDECODE_THRD_STATUS status);
BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdGetCurrRq(NVDECODE_CTX* ctx, NVDECODE_RQ** rqOut);
BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdInit     (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdAwaitAll (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdContinue (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdCore     (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC void* nvdecodeFeederThrdMain     (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdSetStatus(NVDECODE_CTX* ctx, NVDECODE_THRD_STATUS status);
BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdGetCurrRq(NVDECODE_CTX* ctx, NVDECODE_RQ** rqOut);
BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdInit     (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdAwaitAll (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdContinue (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdCore     (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC void* nvdecodeWorkerThrdMain     (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC void  nvdecodeWorkerThrdCallback (cudaStream_t  stream,
                                                        cudaError_t   status,
                                                        NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdSetStatus(NVDECODE_CTX* ctx, NVDECODE_THRD_STATUS status);
BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdGetCurrRq(NVDECODE_CTX* ctx, NVDECODE_RQ** rqOut);
BENZINA_PLUGIN_STATIC int   nvdecodeSetDevice          (NVDECODE_CTX* ctx, const char* deviceId);
BENZINA_PLUGIN_STATIC int   nvdecodeAllocNvcuvid       (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeAllocDataOpen      (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeAllocPBParse       (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeAllocThreading     (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeAllocCleanup       (NVDECODE_CTX* ctx, int ret);



/* Static Function Definitions */

/**
 * @brief Compute timespec for the absolute time NOW + dt.
 * @param [out] ts  The computed timespec.
 * @param [in]  dt  The time delta from NOW.
 * @return The return code from clock_gettime(CLOCK_REALTIME, ts).
 */

BENZINA_PLUGIN_STATIC int   nvdecodeAbstime            (struct timespec* ts, double dt){
	int    ret;
	double i=floor(dt), f=dt-i;
	
	ret = clock_gettime(CLOCK_REALTIME, ts);
	ts->tv_nsec += 1000000000*f;
	if(ts->tv_nsec >= 1000000000){
		ts->tv_nsec -= 1000000000;
		ts->tv_sec++;
	}
	ts->tv_sec += (uint64_t)i;
	
	return ret;
}

/**
 * @brief 
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx
 * @param [in]  batchOut
 * @return 
 */

BENZINA_PLUGIN_STATIC int   nvdecodeMasterThrdGetSubmBt(NVDECODE_CTX* ctx, NVDECODE_BATCH** batchOut){
	*batchOut = &ctx->batch[ctx->master.cntBatchSub % ctx->multibuffering];
	return 0;
}

/**
 * @brief 
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx
 * @param [in]  batchOut
 * @return 
 */

BENZINA_PLUGIN_STATIC int   nvdecodeMasterThrdGetRetrBt(NVDECODE_CTX* ctx, NVDECODE_BATCH** batchOut){
	*batchOut = &ctx->batch[ctx->master.cntBatchAck % ctx->multibuffering];
	return 0;
}

/**
 * @brief 
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx
 * @param [in]  rqOut
 * @return 
 */

BENZINA_PLUGIN_STATIC int   nvdecodeMasterThrdGetSubmRq(NVDECODE_CTX* ctx, NVDECODE_RQ**    rqOut){
	*rqOut = &ctx->request[ctx->master.cntSubmitted % ctx->totalSlots];
	return 0;
}

/**
 * @brief Launch helper threads.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx
 * @return 0 if threads already running or started successfully.
 *         !0 if threads not already running and could not be started.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeHelpersStart       (NVDECODE_CTX* ctx){
	uint64_t       cntLifecycles, i;
	pthread_attr_t attr;
	
	/**
	 * If the helper threads are not spawned, we spawn all of them.
	 * Otherwise, we wait for this event.
	 */
	
	if(nvdecodeHelpersAllStatusIs(ctx, THRD_NOT_RUNNING)){
		ctx->master.cntSubmitted = 0;
		ctx->master.cntCompleted = 0;
		ctx->master.cntBatchSub  = 0;
		ctx->master.cntBatchAck  = 0;
		ctx->reader.cnt          = 0;
		ctx->feeder.cnt          = 0;
		ctx->worker.cnt          = 0;
		ctx->decoderInited       = 0;
		ctx->decoderRefCnt       = 0;
		ctx->mallocRefCnt        = 0;
		memset(ctx->batch,   0, sizeof(*ctx->batch)   * ctx->multibuffering);
		memset(ctx->request, 0, sizeof(*ctx->request) * ctx->totalSlots);
		for(i=0;i<ctx->totalSlots;i++){
			ctx->request[i].picParams = &ctx->picParams[i];
			ctx->request[i].data      = NULL;
		}
		
		if(pthread_attr_init        (&attr)          != 0 ||
		   pthread_attr_setstacksize(&attr, 64*1024) != 0){
			return -1;
		}
		ctx->reader.status = pthread_create(&ctx->reader.thrd, &attr, (void*(*)(void*))nvdecodeReaderThrdMain, ctx) == 0 ? THRD_SPAWNED : THRD_NOT_RUNNING;
		ctx->feeder.status = pthread_create(&ctx->feeder.thrd, &attr, (void*(*)(void*))nvdecodeFeederThrdMain, ctx) == 0 ? THRD_SPAWNED : THRD_NOT_RUNNING;
		ctx->worker.status = pthread_create(&ctx->worker.thrd, &attr, (void*(*)(void*))nvdecodeWorkerThrdMain, ctx) == 0 ? THRD_SPAWNED : THRD_NOT_RUNNING;
		pthread_detach(ctx->reader.thrd);
		pthread_detach(ctx->feeder.thrd);
		pthread_detach(ctx->worker.thrd);
		pthread_attr_destroy(&attr);
		
		ctx->master.cntLifecycles++;
	}
	
	/**
	 * We wait for all threads to stabilize either to the all-RUNNING or
	 * all-NOT_RUNNING state.
	 * 
	 * If at least one thread
	 * 
	 *   1) Failed to spawn
	 *   2) Failed to initialize
	 *   3) Failed at runtime
	 *   4) Timed out
	 * 
	 * , all threads that did spawn and/or init and/or run will self-destruct
	 * promptly (THRD_* -> THRD_NOT_RUNNING) and signal us. The transition of
	 * the last helper thread in this lifecycle to NOT_RUNNING indicates the
	 * end of the lifecycle.
	 * 
	 * If, while we have released the mutex, the lifecycle number changes under
	 * our feet, we abandon entirely. We've slept so long that the next
	 * lifecycle has been entered. We
	 * exit straight out, since we had been waiting on a previous lifecycle
	 * and therefore these helpers may be of no use to us.
	 * 
	 * We do no cleanup in this situation since whichever events have
	 * transpired since have already arranged to clean up the resources for us.
	 */
	
	cntLifecycles = ctx->master.cntLifecycles;
	do{
		if(cntLifecycles != ctx->master.cntLifecycles       ){return -2;}
		if(nvdecodeHelpersAllStatusIs(ctx, THRD_NOT_RUNNING)){return -1;}
		if(nvdecodeHelpersAllStatusIs(ctx, THRD_RUNNING    )){return  0;}
	}while(pthread_cond_wait(&ctx->master.cond, &ctx->lock) == 0);
	return -3;
}

/**
 * @brief Stop helper threads.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx
 * @return 0 if threads not running or successfully stopped.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeHelpersStop        (NVDECODE_CTX* ctx){
	uint64_t cntLifecycles = ctx->master.cntLifecycles;
	do{
		if(cntLifecycles != ctx->master.cntLifecycles       ){return -2;}
		if(nvdecodeHelpersAllStatusIs(ctx, THRD_NOT_RUNNING)){
			pthread_cond_broadcast(&ctx->master.cond);
			return 0;
		}
		if(ctx->reader.status != THRD_NOT_RUNNING){nvdecodeReaderThrdSetStatus(ctx, THRD_EXITING);}
		if(ctx->feeder.status != THRD_NOT_RUNNING){nvdecodeFeederThrdSetStatus(ctx, THRD_EXITING);}
		if(ctx->worker.status != THRD_NOT_RUNNING){nvdecodeWorkerThrdSetStatus(ctx, THRD_EXITING);}
	}while(pthread_cond_wait(&ctx->master.cond, &ctx->lock) == 0);
	return -3;
}

/**
 * @brief Whether all helpers have the given status.
 * 
 * @param [in]  ctx
 * @param [in]  status
 * @return Whether (!0) or not (0) all helper threads have the given status.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeHelpersAllStatusIs (NVDECODE_CTX* ctx, NVDECODE_THRD_STATUS status){
	return ctx->reader.status == status &&
	       ctx->feeder.status == status &&
	       ctx->worker.status == status;
}

/**
 * @brief Whether any helpers have the given status.
 * 
 * @param [in]  ctx
 * @param [in]  status
 * @return Whether (!0) or not (0) any helper threads have the given status.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeHelpersAnyStatusIs (NVDECODE_CTX* ctx, NVDECODE_THRD_STATUS status){
	return ctx->reader.status == status ||
	       ctx->feeder.status == status ||
	       ctx->worker.status == status;
}

/**
 * @brief Whether the spawning of helpers is currently failing or not.
 * 
 * @param [in]  ctx
 * @param [in]  status
 * @return Whether (!0) or not (0) any helper threads are currently failing or failed.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeHelpersFailing     (NVDECODE_CTX* ctx){
	return nvdecodeHelpersAnyStatusIs(ctx, THRD_NOT_RUNNING) ||
	       nvdecodeHelpersAnyStatusIs(ctx, THRD_EXITING);
}

/**
 * @brief Maybe reap leftover malloc()'s from the reader.
 * @param [in]  ctx  The context in question.
 * @return 0.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeMaybeReapMallocs   (NVDECODE_CTX* ctx){
	uint64_t i;
	
	if(!--ctx->mallocRefCnt){
		for(i=0;i<ctx->totalSlots;i++){
			free(ctx->request[i].data);
			ctx->request[i].data = NULL;
		}
	}
	
	return 0;
}

/**
 * @brief Possibly destroy decoder, if no longer needed.
 * 
 * The feeder and worker threads share a decoder, but because either thread
 * may fail, the other must be ready to cleanup the decoder.
 * 
 * @param [in]  ctx  The context in question.
 * @return 0.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeMaybeReapDecoder   (NVDECODE_CTX* ctx){
	if(!--ctx->decoderRefCnt && ctx->decoderInited){
		ctx->cuvidDestroyDecoder(ctx->decoder);
		ctx->decoderInited = 0;
	}
	
	return 0;
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

BENZINA_PLUGIN_STATIC void* nvdecodeReaderThrdMain     (NVDECODE_CTX* ctx){
	pthread_mutex_lock(&ctx->lock);
	if(nvdecodeReaderThrdInit(ctx)){
		while(nvdecodeReaderThrdContinue(ctx)){
			nvdecodeReaderThrdCore(ctx);
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

BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdInit     (NVDECODE_CTX* ctx){
	if(nvdecodeHelpersFailing(ctx)){
		nvdecodeReaderThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
	
	ctx->mallocRefCnt++;
	
	nvdecodeReaderThrdSetStatus(ctx, THRD_INITED);
	if(nvdecodeReaderThrdAwaitAll(ctx)){
		nvdecodeReaderThrdSetStatus(ctx, THRD_RUNNING);
		return 1;
	}else{
		nvdecodeReaderThrdSetStatus(ctx, THRD_EXITING);
		nvdecodeMaybeReapMallocs(ctx);
		nvdecodeReaderThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
}

/**
 * @brief Wait for full initialization of all threads.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx
 * @return Whether (!0) or not (0) all threads reached INITED or RUNNING state.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdAwaitAll (NVDECODE_CTX* ctx){
	do{
		if(nvdecodeHelpersFailing(ctx)){return 0;}
		if(!nvdecodeHelpersAnyStatusIs(ctx, THRD_SPAWNED)){return 1;}
	}while(pthread_cond_wait(&ctx->reader.cond, &ctx->lock) == 0);
	return 0;
}

/**
 * @brief Determine whether the reader thread should shut down or do more work.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The
 * @return Whether (!0) or not (0) there is work to do.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdContinue (NVDECODE_CTX* ctx){
	do{
		if(ctx->reader.cnt >= ctx->master.cntSubmitted){
			if(nvdecodeHelpersFailing(ctx)){
				break;   /* FIXME: Exit path. */
			}else{
				continue;/* FIXME: No work to do; First entry, or spurious wakeup */
			}
		}
		return 1;
	}while(pthread_cond_wait(&ctx->reader.cond, &ctx->lock) == 0);
	
	nvdecodeReaderThrdSetStatus(ctx, THRD_EXITING);
	nvdecodeMaybeReapMallocs   (ctx);
	nvdecodeReaderThrdSetStatus(ctx, THRD_NOT_RUNNING);
	return 0;
}

/**
 * @brief Perform the core operation of the reader thread.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The context 
 * @return 0
 */

BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdCore     (NVDECODE_CTX* ctx){
	NVDECODE_RQ* rq;
	uint64_t     di;
	int          readsDone;
	struct{
		int fd; size_t off; size_t len; void* ptr; ssize_t lenRead;
	} rd0 = {0}, rd1 = {0};
	
	
	nvdecodeReaderThrdGetCurrRq(ctx, &rq);
	rq->data = NULL;
	di       = rq->datasetIndex;
	rd0.fd   = ctx->datasetBinFd;
	if(benzinaDatasetGetElement(ctx->dataset, di, &rd0.off, &rd0.len) != 0){
		nvdecodeReaderThrdSetStatus(ctx, THRD_EXITING); return 0;
	}
	rq->data = malloc(rd0.len);
	if(!rq->data){
		nvdecodeReaderThrdSetStatus(ctx, THRD_EXITING); return 0;
	}
	rd0.ptr  = rq->data;
	rd1.fd   = ctx->datasetNvdecodeFd;
	rd1.len  = ctx->picParamTruncLen;
	rd1.off  = ctx->picParamTruncLen*di;
	rd1.ptr  = rq->picParams;
	
	
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
	
	readsDone = (rd0.lenRead==(ssize_t)rd0.len) &&
	            (rd1.lenRead==(ssize_t)rd1.len);
	if(readsDone){
		ctx->reader.cnt++;
	}else{
		free(rq->data);
		rq->data = NULL;
		nvdecodeReaderThrdSetStatus(ctx, THRD_EXITING); return 0;
	}
	pthread_cond_broadcast(&ctx->feeder.cond);
	
	return 0;
}

/**
 * @brief Change reader thread's status.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The context whose reader thread's status is being changed.
 * @return 0
 */

BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdSetStatus(NVDECODE_CTX* ctx, NVDECODE_THRD_STATUS status){
	ctx->reader.status = status;
	pthread_cond_broadcast(&ctx->master.cond);
	pthread_cond_broadcast(&ctx->reader.cond);
	pthread_cond_broadcast(&ctx->feeder.cond);
	pthread_cond_broadcast(&ctx->worker.cond);
	return 0;
}

/**
 * @brief Get reader thread's current processing request.
 * @param [in]  ctx    The context in question.
 * @param [out] rqOut  The pointer to the request block.
 * @return 0
 */

BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdGetCurrRq(NVDECODE_CTX* ctx, NVDECODE_RQ** rqOut){
	*rqOut = &ctx->request[ctx->reader.cnt % ctx->totalSlots];
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

BENZINA_PLUGIN_STATIC void* nvdecodeFeederThrdMain     (NVDECODE_CTX* ctx){
	pthread_mutex_lock(&ctx->lock);
	if(nvdecodeFeederThrdInit(ctx)){
		while(nvdecodeFeederThrdContinue(ctx)){
			nvdecodeFeederThrdCore(ctx);
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

BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdInit     (NVDECODE_CTX* ctx){
	if(nvdecodeHelpersFailing(ctx)){
		nvdecodeFeederThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
	
	if(cudaSetDevice(ctx->deviceOrd) != cudaSuccess){
		nvdecodeFeederThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
	
	if(ctx->cuvidGetDecoderCaps){
		memset(&ctx->decoderCaps, 0, sizeof(ctx->decoderCaps));
		ctx->decoderCaps.eCodecType      = ctx->decoderInfo.CodecType;
		ctx->decoderCaps.eChromaFormat   = ctx->decoderInfo.ChromaFormat;
		ctx->decoderCaps.nBitDepthMinus8 = ctx->decoderInfo.bitDepthMinus8;
		if(ctx->cuvidGetDecoderCaps(&ctx->decoderCaps)               != CUDA_SUCCESS ||
		   !ctx->decoderCaps.bIsSupported                                            ||
		   ctx->decoderInfo.ulWidth  < ctx->decoderCaps.nMinWidth                    ||
		   ctx->decoderInfo.ulWidth  > ctx->decoderCaps.nMaxWidth                    ||
		   ctx->decoderInfo.ulHeight < ctx->decoderCaps.nMinHeight                   ||
		   ctx->decoderInfo.ulHeight > ctx->decoderCaps.nMaxHeight                   ||
		   ((ctx->decoderInfo.ulWidth*ctx->decoderInfo.ulHeight/256) > ctx->decoderCaps.nMaxMBCount)){
			nvdecodeFeederThrdSetStatus(ctx, THRD_NOT_RUNNING);
			return 0;
		}
	}
	if(ctx->cuvidCreateDecoder(&ctx->decoder, &ctx->decoderInfo) != CUDA_SUCCESS){
		nvdecodeFeederThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
	ctx->decoderInited = 1;
	ctx->decoderRefCnt++;
	
	ctx->mallocRefCnt++;
	
	nvdecodeFeederThrdSetStatus(ctx, THRD_INITED);
	if(nvdecodeFeederThrdAwaitAll(ctx)){
		nvdecodeFeederThrdSetStatus(ctx, THRD_RUNNING);
		return 1;
	}else{
		nvdecodeFeederThrdSetStatus(ctx, THRD_EXITING);
		nvdecodeMaybeReapDecoder(ctx);
		nvdecodeMaybeReapMallocs(ctx);
		nvdecodeFeederThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
}

/**
 * @brief Wait for full initialization of all threads.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx
 * @return Whether (!0) or not (0) all threads reached INITED or RUNNING state.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdAwaitAll (NVDECODE_CTX* ctx){
	do{
		if(nvdecodeHelpersFailing(ctx)){return 0;}
		if(!nvdecodeHelpersAnyStatusIs(ctx, THRD_SPAWNED)){return 1;}
	}while(pthread_cond_wait(&ctx->feeder.cond, &ctx->lock) == 0);
	return 0;
}

/**
 * @brief Determine whether the feeder thread should shut down or do more work.
 * 
 * @param [in]  ctx  The
 * @return Whether (!0) or not (0) there is work to do.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdContinue (NVDECODE_CTX* ctx){
	do{
		if(ctx->feeder.cnt >= ctx->reader.cnt){
			if(nvdecodeHelpersFailing(ctx)){
				break;   /* FIXME: Exit path. */
			}else{
				continue;/* FIXME: No work to do; First entry, or spurious wakeup */
			}
		}
		return 1;
	}while(pthread_cond_wait(&ctx->feeder.cond, &ctx->lock) == 0);
	
	/**
	 * If we are the last owners of the decoder handle, destroy it.
	 * 
	 * Normally, the feeder thread will never destroy the decoder. However, if
	 * the feeder thread spawns and initializes, but the worker thread spawns
	 * and fails to initialize, we must harvest the decoder ourselves. The
	 * reverse can also happen: The worker thread could spawn and initialize,
	 * and the feeder thread could spawn but fail to initialize. In that case,
	 * the worker thread must *not* destroy the decoder, since it wasn't
	 * initialized.
	 */
	
	nvdecodeFeederThrdSetStatus(ctx, THRD_EXITING);
	nvdecodeMaybeReapMallocs   (ctx);
	nvdecodeMaybeReapDecoder   (ctx);
	nvdecodeFeederThrdSetStatus(ctx, THRD_NOT_RUNNING);
	return 0;
}

/**
 * @brief Perform the core operation of the feeder thread.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The context 
 * @return 0
 */

BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdCore     (NVDECODE_CTX* ctx){
	NVDECODE_RQ*    rq;
	CUVIDPICPARAMS* pP;
	CUresult        ret;
	unsigned int    ZERO = 0;
	
	
	nvdecodeFeederThrdGetCurrRq(ctx, &rq);
	pP = rq->picParams;
	
	/**
	 * When we generated this dataset, we encoded the byte offset from
	 * the beginning of the H264 frame in the pointer field. We also
	 * must supply one slice offset of 0, since there is just one
	 * slice.
	 * 
	 * Patch up these pointers to valid values before supplying it to
	 * cuvidDecodePicture().
	 * 
	 * Also, set a CurrPicIdx value. Allegedly, it is always in the
	 * range [0, MAX_DECODE_SURFACES).
	 */
	
	pP->pBitstreamData    = rq->data+(uint64_t)pP->pBitstreamData;
	pP->pSliceDataOffsets = &ZERO;
	pP->CurrPicIdx        = ctx->feeder.cnt % ctx->decoderInfo.ulNumDecodeSurfaces;
	
	/**
	 * Drop mutex and possibly block attempting to decode image, then
	 * reacquire mutex.
	 */
	
	pthread_mutex_unlock(&ctx->lock);
	ret = ctx->cuvidDecodePicture(ctx->decoder, pP);
	pthread_mutex_lock(&ctx->lock);
	
	/* Release data. */
	free(rq->data);
	rq->data = NULL;
	if(ret != CUDA_SUCCESS){
		nvdecodeFeederThrdSetStatus(ctx, THRD_EXITING);
		return 0;
	}
	
	/* Bump counters and broadcast signal. */
	ctx->feeder.cnt++;
	pthread_cond_broadcast(&ctx->worker.cond);
	
	return 0;
}

/**
 * @brief Change feeder thread's status.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The context whose feeder thread's status is being changed.
 * @return 0
 */

BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdSetStatus(NVDECODE_CTX* ctx, NVDECODE_THRD_STATUS status){
	ctx->feeder.status = status;
	pthread_cond_broadcast(&ctx->master.cond);
	pthread_cond_broadcast(&ctx->reader.cond);
	pthread_cond_broadcast(&ctx->feeder.cond);
	pthread_cond_broadcast(&ctx->worker.cond);
	return 0;
}

/**
 * @brief Get feeder thread's current processing request.
 * @param [in]  ctx    The context in question.
 * @param [out] rqOut  The pointer to the request block.
 * @return 0
 */

BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdGetCurrRq(NVDECODE_CTX* ctx, NVDECODE_RQ** rqOut){
	*rqOut = &ctx->request[ctx->feeder.cnt % ctx->totalSlots];
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

BENZINA_PLUGIN_STATIC void* nvdecodeWorkerThrdMain     (NVDECODE_CTX* ctx){
	pthread_mutex_lock(&ctx->lock);
	if(nvdecodeWorkerThrdInit(ctx)){
		while(nvdecodeWorkerThrdContinue(ctx)){
			nvdecodeWorkerThrdCore(ctx);
		}
	}
	pthread_mutex_unlock(&ctx->lock);
	pthread_exit(NULL);
}

/**
 * @brief Initialize worker thread state.
 * 
 * Called with the lock held and status SPAWNED.
 * 
 * @param [in]  ctx  The context whose worker thread is initializing.
 * @return Whether (!0) or not (0) initialization was successful.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdInit     (NVDECODE_CTX* ctx){
	if(nvdecodeHelpersFailing(ctx)){
		nvdecodeWorkerThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
	
	if(cudaSetDevice   (ctx->deviceOrd)          != cudaSuccess ||
	   cudaStreamCreate(&ctx->worker.cudaStream) != cudaSuccess){
		nvdecodeWorkerThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
	
	ctx->decoderRefCnt++;
	
	nvdecodeWorkerThrdSetStatus(ctx, THRD_INITED);
	if(nvdecodeWorkerThrdAwaitAll(ctx)){
		nvdecodeWorkerThrdSetStatus(ctx, THRD_RUNNING);
		return 1;
	}else{
		nvdecodeWorkerThrdSetStatus(ctx, THRD_EXITING);
		nvdecodeMaybeReapDecoder(ctx);
		nvdecodeWorkerThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
}

/**
 * @brief Wait for full initialization of all threads.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx
 * @return Whether (!0) or not (0) all threads reached INITED or RUNNING state.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdAwaitAll (NVDECODE_CTX* ctx){
	do{
		if(nvdecodeHelpersFailing(ctx)){return 0;}
		if(!nvdecodeHelpersAnyStatusIs(ctx, THRD_SPAWNED)){return 1;}
	}while(pthread_cond_wait(&ctx->worker.cond, &ctx->lock) == 0);
	return 0;
}

/**
 * @brief Determine whether the worker thread should shut down or do more work.
 * 
 * @param [in]  ctx  The
 * @return Whether (!0) or not (0) there is work to do.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdContinue (NVDECODE_CTX* ctx){
	do{
		if(ctx->worker.cnt >= ctx->feeder.cnt){
			if(nvdecodeHelpersFailing(ctx)){
				break;   /* FIXME: Exit path. */
			}else{
				continue;/* FIXME: No work to do; First entry, or spurious wakeup */
			}
		}
		return 1;
	}while(pthread_cond_wait(&ctx->worker.cond, &ctx->lock) == 0);
	
	/**
	 * Destroy the decoder if we own the last reference to it.
	 * 
	 * Also, the worker thread is nominally responsible for the CUDA stream. We
	 * wait until work on the CUDA stream completes before exiting. We drop the
	 * lock while doing so, since the callbacks enqueued on that stream require
	 * the lock to work. We then reacquire the lock, set the status to
	 * NOT_RUNNING and exit.
	 */
	
	nvdecodeWorkerThrdSetStatus(ctx, THRD_EXITING);
	nvdecodeMaybeReapDecoder   (ctx);
	pthread_mutex_unlock       (&ctx->lock);
	cudaStreamSynchronize      (ctx->worker.cudaStream);
	cudaStreamDestroy          (ctx->worker.cudaStream);
	pthread_mutex_lock         (&ctx->lock);
	nvdecodeWorkerThrdSetStatus(ctx, THRD_NOT_RUNNING);
	return 0;
}

/**
 * @brief Perform the core operation of the worker thread.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The context 
 * @return 0
 */

BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdCore     (NVDECODE_CTX* ctx){
	NVDECODE_RQ*       rq;
	CUVIDPROCPARAMS    vpp;
	unsigned long long srcPtr;
	unsigned           pitch;
	uint64_t           picIdx = 0;
	CUresult           retMap;
	
	nvdecodeWorkerThrdGetCurrRq(ctx, &rq);
	memset(&vpp, 0, sizeof(vpp));
	vpp.progressive_frame = 1;
	vpp.second_field      = 0;
	vpp.top_field_first   = 0;
	vpp.unpaired_field    = 0;
	vpp.output_stream     = ctx->worker.cudaStream;
	picIdx                = ctx->worker.cnt % ctx->decoderInfo.ulNumDecodeSurfaces;
	
	/**
	 * Drop the mutex and block on the decoder, then perform CUDA ops
	 * on the returned data. Then, reacquire lock.
	 */
	
	pthread_mutex_unlock(&ctx->lock);
	retMap = ctx->cuvidMapVideoFrame64(ctx->decoder, picIdx, &srcPtr, &pitch, &vpp);
	if(retMap == CUDA_SUCCESS){
		/* From devPtr & pitch to rq->devPtr. */
		nvdecodePostprocKernelInvoker(ctx->worker.cudaStream,
		                              rq->devPtr,
		                              ctx->outputHeight,
		                              ctx->outputWidth,
		                              rq->OOB [0], rq->OOB [1], rq->OOB [2],
		                              rq->B   [0], rq->B   [1], rq->B   [2],
		                              rq->S   [0], rq->S   [1], rq->S   [2],
		                              rq->H[0][0], rq->H[0][1], rq->H[0][2],
		                              rq->H[1][0], rq->H[1][1], rq->H[1][2],
		                              rq->H[2][0], rq->H[2][1], rq->H[2][2],
		                              rq->colorMatrix,
		                              (void*)srcPtr,
		                              pitch,
		                              ctx->decoderInfo.ulHeight,
		                              ctx->decoderInfo.ulWidth);
		cudaStreamAddCallback(ctx->worker.cudaStream,
		                      (cudaStreamCallback_t)nvdecodeWorkerThrdCallback,
		                      ctx,
		                      0);
		ctx->cuvidUnmapVideoFrame64(ctx->decoder, srcPtr);
	}
	pthread_mutex_lock(&ctx->lock);
	
	if(retMap == CUDA_SUCCESS){
		ctx->worker.cnt++;
	}else{
		nvdecodeWorkerThrdSetStatus(ctx, THRD_EXITING); return 0;
	}
	
	return 0;
}

/**
 * @brief Post-processing Callback
 * @param [in]   stream The stream onto which this callback had been scheduled.
 * @param [in]   status The error status of this device or stream.
 * @param [in]   ctx    The context on which this callback is being executed.
 * @return 
 */

BENZINA_PLUGIN_STATIC void  nvdecodeWorkerThrdCallback (cudaStream_t  stream,
                                                        cudaError_t   status,
                                                        NVDECODE_CTX* ctx){
	(void)stream;
	
	pthread_mutex_lock(&ctx->lock);
	if(status == cudaSuccess){
		ctx->master.cntCompleted++;
		pthread_cond_broadcast(&ctx->master.cond);
	}else{
		nvdecodeWorkerThrdSetStatus(ctx, THRD_EXITING);
	}
	pthread_mutex_unlock(&ctx->lock);
}

/**
 * @brief Change worker thread's status.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The context whose worker thread's status is being changed.
 * @return 0
 */

BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdSetStatus(NVDECODE_CTX* ctx, NVDECODE_THRD_STATUS status){
	ctx->worker.status = status;
	pthread_cond_broadcast(&ctx->master.cond);
	pthread_cond_broadcast(&ctx->reader.cond);
	pthread_cond_broadcast(&ctx->feeder.cond);
	pthread_cond_broadcast(&ctx->worker.cond);
	return 0;
}

/**
 * @brief Get worker thread's current processing request.
 * @param [in]  ctx    The context in question.
 * @param [out] rqOut  The pointer to the request block.
 * @return 0
 */

BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdGetCurrRq(NVDECODE_CTX* ctx, NVDECODE_RQ** rqOut){
	*rqOut = &ctx->request[ctx->worker.cnt % ctx->totalSlots];
	return 0;
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

BENZINA_PLUGIN_STATIC int   nvdecodeSetDevice          (NVDECODE_CTX* ctx, const char*   deviceId){
	int ret, deviceCount=0, i=-1;
	char* s;
	struct cudaDeviceProp prop;
	
	
	/* Forbid changing device ordinal while threads running. */
	if(ctx->reader.status != THRD_NOT_RUNNING ||
	   ctx->feeder.status != THRD_NOT_RUNNING ||
	   ctx->worker.status != THRD_NOT_RUNNING){
		return BENZINA_DATALOADER_ITER_ALREADYINITED;
	}
	
	
	/* Determine maximum device ordinal. */
	ret = cudaGetDeviceCount(&deviceCount);
	if(ret != cudaSuccess){return ret;}
	
	
	/* Select a device ordinal i by one of several identification string schemes. */
	if      (strncmp(deviceId, "cuda:", strlen("cuda:")) == 0){
		if(deviceId[strlen("cuda:")] == '\0'){
			return BENZINA_DATALOADER_ITER_INVALIDARGS;
		}
		i = strtoull(deviceId+strlen("cuda:"), &s, 10);
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
 * @param [out] ctxOut   Output pointer for the context handle.
 * @param [in]  dataset  The dataset over which this iterator will iterate.
 *                       Must be non-NULL and compatible.
 * @return A pointer to the context, if successful; NULL otherwise.
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodeAlloc              (void** ctxOut, const BENZINA_DATASET* dataset){
	NVDECODE_CTX* ctx = NULL;
	const char*   datasetRoot = NULL;
	size_t        datasetLen;
	
	
	/**
	 * The ctxOut and dataset parameters cannot be NULL.
	 */
	
	if(!ctxOut){
		return -1;
	}
	*ctxOut = NULL;
	if(!dataset                                            ||
	   benzinaDatasetGetRoot  (dataset, &datasetRoot) != 0 ||
	   benzinaDatasetGetLength(dataset, &datasetLen)  != 0){
		return -1;
	}
	
	
	/**
	 * Allocate memory for context.
	 * 
	 * Also, initialize certain critical elements.
	 */
	
	*ctxOut = calloc(1, sizeof(*ctx));
	if(!*ctxOut){
		return -1;
	}else{
		ctx = (NVDECODE_CTX*)*ctxOut;
	}
	ctx->dataset           =  dataset;
	ctx->datasetRoot       =  datasetRoot;
	ctx->datasetLen        =  datasetLen;
	ctx->datasetBinFd      = -1;
	ctx->datasetProtobufFd = -1;
	ctx->datasetNvdecodeFd = -1;
	ctx->refCnt            =  1;
	ctx->deviceOrd         = -1;
	ctx->defaults.S[0]     = ctx->defaults.S[1] = ctx->defaults.S[2] = 1.0;
	ctx->picParams         = NULL;
	ctx->request           = NULL;
	ctx->batch             = NULL;
	ctx->cuvidHandle       = NULL;
	
	
	/**
	 * Tail-call into context initialization procedure.
	 */
	
	return nvdecodeAllocDataOpen(ctx);
}

/**
 * @brief Initialize context's dataset handles.
 * 
 * @param [in]   ctx  The context being initialized.
 * @return Error code.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeAllocDataOpen      (NVDECODE_CTX* ctx){
	struct stat   s0, s1, s2, s3, s4;
	int           dirfd;
	
	dirfd                  = open  (ctx->datasetRoot,       O_RDONLY|O_CLOEXEC|O_DIRECTORY);
	ctx->datasetBinFd      = openat(dirfd, "data.bin",      O_RDONLY|O_CLOEXEC);
	ctx->datasetProtobufFd = openat(dirfd, "data.protobuf", O_RDONLY|O_CLOEXEC);
	ctx->datasetNvdecodeFd = openat(dirfd, "data.nvdecode", O_RDONLY|O_CLOEXEC);
	if(ctx->datasetBinFd                                          < 0 ||
	   ctx->datasetProtobufFd                                     < 0 ||
	   ctx->datasetNvdecodeFd                                     < 0 ||
	   fstat  (ctx->datasetBinFd,      &s0)                       < 0 ||
	   fstatat(dirfd, "data.lengths",  &s1, 0)                    < 0 ||
	   fstat  (ctx->datasetProtobufFd, &ctx->datasetProtobufStat) < 0 ||
	   fstat  (ctx->datasetNvdecodeFd, &s2)                       < 0 ||
	   fstatat(dirfd, "README.md",     &s3, 0)                    < 0 ||
	   fstatat(dirfd, "SHA256SUMS",    &s4, 0)                    < 0 ||
	   s2.st_size % ctx->datasetLen                              != 0){
		close(dirfd);
		return nvdecodeAllocCleanup(ctx, -1);
	}
	close(dirfd);
	ctx->picParamTruncLen = s2.st_size / ctx->datasetLen;
	
	return nvdecodeAllocPBParse(ctx);
}

/**
 * @brief Parse protobuf description of dataset.
 * 
 * @param [in]   ctx  The context being initialized.
 * @return Error code.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeAllocPBParse       (NVDECODE_CTX* ctx){
	BENZINA_BUF bbuf;
	int         pbFd    = ctx->datasetProtobufFd;
	size_t      bufSize = ctx->datasetProtobufStat.st_size;
	uint64_t    dummy   = 0;
	uint32_t    tag, wire;
	
	if(benzinaBufInit       (&bbuf)                != 0 ||
	   benzinaBufEnsure     (&bbuf, bufSize)       != 0 ||
	   benzinaBufWriteFromFd(&bbuf, pbFd, bufSize) != 0){
		benzinaBufFini(&bbuf);
		return nvdecodeAllocCleanup(ctx, -1);
	}
	close(ctx->datasetProtobufFd);
	ctx->datasetProtobufFd = -1;
	benzinaBufSeek(&bbuf, 0, SEEK_SET);
	while(benzinaBufReadTagW(&bbuf, &tag, &wire) == 0){
		switch(tag){
			#define TAGCASE(tag, target)                        \
			    case tag:                                       \
			        if(benzinaBufReadvu64(&bbuf, &dummy) != 0){ \
			            benzinaBufFini(&bbuf);                  \
			            return nvdecodeAllocCleanup(ctx, -2);   \
			        }                                           \
			        target = dummy;                             \
			    break;
			TAGCASE(33554432, ctx->decoderInfo.ulWidth);
			TAGCASE(33554433, ctx->decoderInfo.ulHeight);
			TAGCASE(33554434, ctx->decoderInfo.ulNumDecodeSurfaces);
			TAGCASE(33554435, ctx->decoderInfo.CodecType);
			TAGCASE(33554436, ctx->decoderInfo.ChromaFormat);
			TAGCASE(33554438, ctx->decoderInfo.bitDepthMinus8);
			TAGCASE(33554439, ctx->decoderInfo.ulIntraDecodeOnly);
			TAGCASE(33554443, ctx->decoderInfo.display_area.left);
			TAGCASE(33554444, ctx->decoderInfo.display_area.top);
			TAGCASE(33554445, ctx->decoderInfo.display_area.right);
			TAGCASE(33554446, ctx->decoderInfo.display_area.bottom);
			TAGCASE(33554447, ctx->decoderInfo.OutputFormat);
			TAGCASE(33554448, ctx->decoderInfo.DeinterlaceMode);
			TAGCASE(33554449, ctx->decoderInfo.ulTargetWidth);
			TAGCASE(33554450, ctx->decoderInfo.ulTargetHeight);
			TAGCASE(33554451, ctx->decoderInfo.ulNumOutputSurfaces);
			TAGCASE(33554453, ctx->decoderInfo.target_rect.left);
			TAGCASE(33554454, ctx->decoderInfo.target_rect.top);
			TAGCASE(33554455, ctx->decoderInfo.target_rect.right);
			TAGCASE(33554456, ctx->decoderInfo.target_rect.bottom);
			#undef TAGCASE
			default:
				if(benzinaBufReadSkip(&bbuf, wire) != 0){
					benzinaBufFini(&bbuf);
					return nvdecodeAllocCleanup(ctx, -2);
				}
			break;
		}
	}
	benzinaBufFini(&bbuf);
	return nvdecodeAllocNvcuvid(ctx);
}

/**
 * @brief Initialize context handles to dynamically-loaded support libraries.
 * 
 * Dynamically attempt to open libnvcuvid.so.1, the basis for this
 * plugin's functionality.
 * 
 * Also retrieve pointers to several library functions.
 * 
 * @param [in]   ctx  The context being initialized.
 * @return Error code.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeAllocNvcuvid       (NVDECODE_CTX* ctx){
	ctx->cuvidHandle = dlopen("libnvcuvid.so.1", RTLD_LAZY);
	if(!ctx->cuvidHandle){
		return nvdecodeAllocCleanup(ctx, -1);
	}
	
	#define READ_SYMBOL(fn)  do{                                \
	       void* symPtr = dlsym(ctx->cuvidHandle, #fn);         \
	       ctx->fn = symPtr ? *(t##fn*)symPtr : (t##fn*)0;      \
	       if(!ctx->fn){                                        \
	           return nvdecodeAllocCleanup(ctx, -1);            \
	       }                                                    \
	    }while(0)
	READ_SYMBOL(cuvidGetDecoderCaps);
	READ_SYMBOL(cuvidCreateDecoder);
	READ_SYMBOL(cuvidDecodePicture);
	READ_SYMBOL(cuvidMapVideoFrame64);
	READ_SYMBOL(cuvidUnmapVideoFrame64);
	READ_SYMBOL(cuvidDestroyDecoder);
	#undef READ_SYMBOL
	
	return nvdecodeAllocThreading(ctx);
}

/**
 * @brief Initialize context's threading resources.
 * 
 * This includes the condition variables and the Big Lock, but does *not*
 * include launching helper threads.
 * 
 * @param [in]   ctx  The context being initialized.
 * @return Error code.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeAllocThreading     (NVDECODE_CTX* ctx){
	if(pthread_mutex_init(&ctx->lock,        0)){goto fail_lock;}
	if(pthread_cond_init (&ctx->master.cond, 0)){goto fail_master;}
	if(pthread_cond_init (&ctx->reader.cond, 0)){goto fail_reader;}
	if(pthread_cond_init (&ctx->feeder.cond, 0)){goto fail_feeder;}
	if(pthread_cond_init (&ctx->worker.cond, 0)){goto fail_worker;}
	
	return nvdecodeAllocCleanup(ctx,  0);
	
	             pthread_cond_destroy (&ctx->worker.cond);
	fail_worker: pthread_cond_destroy (&ctx->feeder.cond);
	fail_feeder: pthread_cond_destroy (&ctx->reader.cond);
	fail_reader: pthread_cond_destroy (&ctx->master.cond);
	fail_master: pthread_mutex_destroy(&ctx->lock);
	fail_lock:
	
	return nvdecodeAllocCleanup(ctx, -1);
}

/**
 * @brief Cleanup for context allocation.
 * 
 * @param [in]  ctx  The context being allocated.
 * @param [in]  ret  Return error code.
 * @return The value `ret`.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeAllocCleanup       (NVDECODE_CTX* ctx, int ret){
	if(ret == 0){
		return ret;
	}
	
	close(ctx->datasetBinFd);
	close(ctx->datasetProtobufFd);
	close(ctx->datasetNvdecodeFd);
	ctx->datasetBinFd      = -1;
	ctx->datasetProtobufFd = -1;
	ctx->datasetNvdecodeFd = -1;
	
	if(ctx->cuvidHandle){
		dlclose(ctx->cuvidHandle);
		ctx->cuvidHandle = NULL;
	}
	
	free(ctx);
	
	return ret;
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

BENZINA_PLUGIN_HIDDEN int   nvdecodeInit               (NVDECODE_CTX* ctx){
	int ret;
	pthread_mutex_lock(&ctx->lock);
	ret = nvdecodeHelpersStart(ctx);
	pthread_mutex_unlock(&ctx->lock);
	return ret;
}

/**
 * @brief Increase reference count of the iterator.
 * 
 * @param [in]  ctx  The iterator context whose reference-count is to be increased.
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodeRetain             (NVDECODE_CTX* ctx){
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

BENZINA_PLUGIN_HIDDEN int   nvdecodeRelease            (NVDECODE_CTX* ctx){
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
	
	nvdecodeHelpersStop  (ctx);
	pthread_mutex_unlock (&ctx->lock);
	
	pthread_cond_destroy (&ctx->worker.cond);
	pthread_cond_destroy (&ctx->feeder.cond);
	pthread_cond_destroy (&ctx->reader.cond);
	pthread_cond_destroy (&ctx->master.cond);
	pthread_mutex_destroy(&ctx->lock);
	
	close(ctx->datasetBinFd);
	close(ctx->datasetNvdecodeFd);
	
	if(ctx->cuvidHandle){
		dlclose(ctx->cuvidHandle);
	}
	
	free(ctx->picParams);
	free(ctx->request);
	free(ctx->batch);
	
	memset(ctx, 0, sizeof(*ctx));
	free(ctx);
	
	return 0;
}

/**
 * @brief Begin defining a batch of samples.
 * 
 * @param [in]  ctx       The iterator context in which.
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodeDefineBatch        (NVDECODE_CTX* ctx){
	NVDECODE_BATCH* batch;
	int ret = 0;
	
	pthread_mutex_lock(&ctx->lock);
	if(ctx->master.cntBatchSub-ctx->master.cntBatchAck >= ctx->multibuffering){
		ret = -1;
	}else{
		nvdecodeMasterThrdGetSubmBt(ctx, &batch);
		batch->startIndex = ctx->master.cntSubmitted;
		batch->stopIndex  = ctx->master.cntSubmitted;
		batch->token      = NULL;
		ret =  0;
	}
	pthread_mutex_unlock(&ctx->lock);
	
	return ret;
}

/**
 * @brief Close and push a batch of work into the pipeline.
 * 
 * @param [in]  ctx    The iterator context in which.
 * @param [in]  token  User data that will be retrieved at the corresponding
 *                     pullBatch().
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodeSubmitBatch        (NVDECODE_CTX* ctx, const void* token){
	NVDECODE_BATCH* batch;
	
	pthread_mutex_lock(&ctx->lock);
	nvdecodeMasterThrdGetSubmBt(ctx, &batch);
	batch->token = token;
	ctx->master.cntBatchSub++;
	pthread_mutex_unlock(&ctx->lock);
	
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

BENZINA_PLUGIN_HIDDEN int   nvdecodeWaitBatch(NVDECODE_CTX* ctx, const void** token, int block, double timeout){
	NVDECODE_BATCH* batch;
	struct timespec ts;
	uint64_t        cntLifecycles;
	int             ret = 0;
	
	*token = NULL;
	if(timeout > 0){
		nvdecodeAbstime(&ts, timeout);
	}
	
	pthread_mutex_lock(&ctx->lock);
	cntLifecycles = ctx->master.cntLifecycles;
	do{
		if(cntLifecycles != ctx->master.cntLifecycles){
			ret = -2;
			break;
		}
		if(ctx->master.cntBatchAck >= ctx->master.cntBatchSub){
			if(!block){
				ret = EAGAIN;
				break;
			}
			continue;
		}
		nvdecodeMasterThrdGetRetrBt(ctx, &batch);
		if(ctx->master.cntCompleted >= batch->stopIndex){
			*token = batch->token;
			ctx->master.cntBatchAck++;
			ret = 0;
			break;
		}
	}while((ret = (timeout > 0 ? pthread_cond_timedwait(&ctx->master.cond, &ctx->lock, &ts) :
	                             pthread_cond_wait     (&ctx->master.cond, &ctx->lock))) == 0);
	pthread_mutex_unlock(&ctx->lock);
	return ret;
}

/**
 * @brief Begin defining a new sample.
 * 
 * @param [in]  ctx     
 * @param [in]  i       Index into dataset.
 * @param [in]  dstPtr  Destination Pointer.
 * @return 0 if no errors; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodeDefineSample(NVDECODE_CTX* ctx, uint64_t i, void* dstPtr){
	NVDECODE_RQ*    rq;
	NVDECODE_BATCH* batch;
	int ret = 0;
	
	pthread_mutex_lock(&ctx->lock);
	nvdecodeMasterThrdGetSubmBt(ctx, &batch);
	nvdecodeMasterThrdGetSubmRq(ctx, &rq);
	if(batch->stopIndex-batch->startIndex >= ctx->batchSize){
		ret = -1;
	}else{
		batch->stopIndex++;
		rq->batch        = batch;
		rq->datasetIndex = i;
		rq->devPtr       = dstPtr;
		ret = 0;
	}
	pthread_mutex_unlock(&ctx->lock);
	
	return ret;
}

/**
 * @brief Submit current sample.
 * 
 * @param [in]  ctx
 * @return 0 if no errors; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodeSubmitSample(NVDECODE_CTX* ctx){
	pthread_mutex_lock(&ctx->lock);
	ctx->master.cntSubmitted++;
	pthread_cond_broadcast(&ctx->reader.cond);
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

BENZINA_PLUGIN_HIDDEN int   nvdecodeSetBuffer               (NVDECODE_CTX* ctx,
                                                             const char*   deviceId,
                                                             void*         outputPtr,
                                                             uint32_t      multibuffering,
                                                             uint32_t      batchSize,
                                                             uint32_t      outputHeight,
                                                             uint32_t      outputWidth){
	int ret;
	
	pthread_mutex_lock(&ctx->lock);
	if(!nvdecodeHelpersAllStatusIs(ctx, THRD_NOT_RUNNING)){
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
			ctx->picParams      = calloc(ctx->totalSlots,     sizeof(*ctx->picParams));
			ctx->request        = calloc(ctx->totalSlots,     sizeof(*ctx->request));
			ctx->batch          = calloc(ctx->multibuffering, sizeof(*ctx->batch));
			if(ctx->picParams && ctx->request && ctx->batch){
				ret = BENZINA_DATALOADER_ITER_SUCCESS;
			}else{
				free(ctx->picParams);
				free(ctx->request);
				free(ctx->batch);
				ctx->picParams = NULL;
				ctx->request   = NULL;
				ctx->batch     = NULL;
				ret = BENZINA_DATALOADER_ITER_INTERNAL;
			}
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
                                                             const float*  H){
	NVDECODE_RQ* rq = &ctx->request[ctx->master.cntSubmitted % ctx->totalSlots];
	if(H){
		memcpy(rq->H, H, sizeof(rq->H));
	}else{
		rq->H[0][0] = 1.0; rq->H[0][1] = 0.0; rq->H[0][2] = 0.0;
		rq->H[1][0] = 0.0; rq->H[1][1] = 1.0; rq->H[1][2] = 0.0;
		rq->H[2][0] = 0.0; rq->H[2][1] = 0.0; rq->H[2][2] = 1.0;
	}
	return 0;
}
BENZINA_PLUGIN_HIDDEN int   nvdecodeSetBias                 (NVDECODE_CTX* ctx,
                                                             const float*  B){
	NVDECODE_RQ* rq = &ctx->request[ctx->master.cntSubmitted % ctx->totalSlots];
	memcpy(rq->B, B?B:ctx->defaults.B, sizeof(rq->B));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int   nvdecodeSetScale                (NVDECODE_CTX* ctx,
                                                             const float*  S){
	NVDECODE_RQ* rq = &ctx->request[ctx->master.cntSubmitted % ctx->totalSlots];
	memcpy(rq->S, S?S:ctx->defaults.S, sizeof(rq->S));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int   nvdecodeSetOOBColor             (NVDECODE_CTX* ctx,
                                                             const float*  OOB){
	NVDECODE_RQ* rq = &ctx->request[ctx->master.cntSubmitted % ctx->totalSlots];
	memcpy(rq->OOB, OOB?OOB:ctx->defaults.OOB, sizeof(rq->OOB));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int   nvdecodeSelectColorMatrix       (NVDECODE_CTX* ctx,
                                                             uint32_t      colorMatrix){
	NVDECODE_RQ* rq = &ctx->request[ctx->master.cntSubmitted % ctx->totalSlots];
	rq->colorMatrix = colorMatrix;
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
	.defineBatch              = (void*)nvdecodeDefineBatch,
	.submitBatch              = (void*)nvdecodeSubmitBatch,
	.waitBatch                = (void*)nvdecodeWaitBatch,
	.defineSample             = (void*)nvdecodeDefineSample,
	.submitSample             = (void*)nvdecodeSubmitSample,
	
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




#if 0
	printf("\n");
	printf("****************\n");
	printf("PicWidthInMbs:            %d\n", picParams->PicWidthInMbs);
	printf("FrameHeightInMbs:         %d\n", picParams->FrameHeightInMbs);
	printf("CurrPicIdx:               %d\n", picParams->CurrPicIdx);
	printf("field_pic_flag:           %d\n", picParams->field_pic_flag);
	printf("bottom_field_flag:        %d\n", picParams->bottom_field_flag);
	printf("second_field:             %d\n", picParams->second_field);
	printf("nBitstreamDataLen:        %d\n", picParams->nBitstreamDataLen);
	printf("nNumSlices:               %d\n", picParams->nNumSlices);
	printf("ref_pic_flag:             %d\n", picParams->ref_pic_flag);
	printf("intra_pic_flag:           %d\n", picParams->intra_pic_flag);
	printf("num_ref_frames:           %d\n", picParams->CodecSpecific.h264.num_ref_frames);
	printf("entropy_coding_mode_flag: %d\n", picParams->CodecSpecific.h264.entropy_coding_mode_flag);
	printf("ref_pic_flag:             %d\n", picParams->CodecSpecific.h264.ref_pic_flag);
	printf("frame_num:                %d\n", picParams->CodecSpecific.h264.frame_num);
	printf("CurrFieldOrderCnt:        %d %d\n", picParams->CodecSpecific.h264.CurrFieldOrderCnt[0], picParams->CodecSpecific.h264.CurrFieldOrderCnt[1]);
	printf("DPB.PicIdx:               %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", picParams->CodecSpecific.h264.dpb[0].PicIdx, picParams->CodecSpecific.h264.dpb[1].PicIdx, picParams->CodecSpecific.h264.dpb[2].PicIdx, picParams->CodecSpecific.h264.dpb[3].PicIdx, picParams->CodecSpecific.h264.dpb[4].PicIdx, picParams->CodecSpecific.h264.dpb[5].PicIdx, picParams->CodecSpecific.h264.dpb[6].PicIdx, picParams->CodecSpecific.h264.dpb[7].PicIdx, picParams->CodecSpecific.h264.dpb[8].PicIdx, picParams->CodecSpecific.h264.dpb[9].PicIdx, picParams->CodecSpecific.h264.dpb[10].PicIdx, picParams->CodecSpecific.h264.dpb[11].PicIdx, picParams->CodecSpecific.h264.dpb[12].PicIdx, picParams->CodecSpecific.h264.dpb[13].PicIdx, picParams->CodecSpecific.h264.dpb[14].PicIdx, picParams->CodecSpecific.h264.dpb[15].PicIdx);
	printf("DPB.FrameIdx:             %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", picParams->CodecSpecific.h264.dpb[0].FrameIdx, picParams->CodecSpecific.h264.dpb[1].FrameIdx, picParams->CodecSpecific.h264.dpb[2].FrameIdx, picParams->CodecSpecific.h264.dpb[3].FrameIdx, picParams->CodecSpecific.h264.dpb[4].FrameIdx, picParams->CodecSpecific.h264.dpb[5].FrameIdx, picParams->CodecSpecific.h264.dpb[6].FrameIdx, picParams->CodecSpecific.h264.dpb[7].FrameIdx, picParams->CodecSpecific.h264.dpb[8].FrameIdx, picParams->CodecSpecific.h264.dpb[9].FrameIdx, picParams->CodecSpecific.h264.dpb[10].FrameIdx, picParams->CodecSpecific.h264.dpb[11].FrameIdx, picParams->CodecSpecific.h264.dpb[12].FrameIdx, picParams->CodecSpecific.h264.dpb[13].FrameIdx, picParams->CodecSpecific.h264.dpb[14].FrameIdx, picParams->CodecSpecific.h264.dpb[15].FrameIdx);
	printf("DPB.is_long_term:         %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", picParams->CodecSpecific.h264.dpb[0].is_long_term, picParams->CodecSpecific.h264.dpb[1].is_long_term, picParams->CodecSpecific.h264.dpb[2].is_long_term, picParams->CodecSpecific.h264.dpb[3].is_long_term, picParams->CodecSpecific.h264.dpb[4].is_long_term, picParams->CodecSpecific.h264.dpb[5].is_long_term, picParams->CodecSpecific.h264.dpb[6].is_long_term, picParams->CodecSpecific.h264.dpb[7].is_long_term, picParams->CodecSpecific.h264.dpb[8].is_long_term, picParams->CodecSpecific.h264.dpb[9].is_long_term, picParams->CodecSpecific.h264.dpb[10].is_long_term, picParams->CodecSpecific.h264.dpb[11].is_long_term, picParams->CodecSpecific.h264.dpb[12].is_long_term, picParams->CodecSpecific.h264.dpb[13].is_long_term, picParams->CodecSpecific.h264.dpb[14].is_long_term, picParams->CodecSpecific.h264.dpb[15].is_long_term);
	printf("DPB.not_existing:         %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", picParams->CodecSpecific.h264.dpb[0].not_existing, picParams->CodecSpecific.h264.dpb[1].not_existing, picParams->CodecSpecific.h264.dpb[2].not_existing, picParams->CodecSpecific.h264.dpb[3].not_existing, picParams->CodecSpecific.h264.dpb[4].not_existing, picParams->CodecSpecific.h264.dpb[5].not_existing, picParams->CodecSpecific.h264.dpb[6].not_existing, picParams->CodecSpecific.h264.dpb[7].not_existing, picParams->CodecSpecific.h264.dpb[8].not_existing, picParams->CodecSpecific.h264.dpb[9].not_existing, picParams->CodecSpecific.h264.dpb[10].not_existing, picParams->CodecSpecific.h264.dpb[11].not_existing, picParams->CodecSpecific.h264.dpb[12].not_existing, picParams->CodecSpecific.h264.dpb[13].not_existing, picParams->CodecSpecific.h264.dpb[14].not_existing, picParams->CodecSpecific.h264.dpb[15].not_existing);
	printf("DPB.used_for_reference:   %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", picParams->CodecSpecific.h264.dpb[0].used_for_reference, picParams->CodecSpecific.h264.dpb[1].used_for_reference, picParams->CodecSpecific.h264.dpb[2].used_for_reference, picParams->CodecSpecific.h264.dpb[3].used_for_reference, picParams->CodecSpecific.h264.dpb[4].used_for_reference, picParams->CodecSpecific.h264.dpb[5].used_for_reference, picParams->CodecSpecific.h264.dpb[6].used_for_reference, picParams->CodecSpecific.h264.dpb[7].used_for_reference, picParams->CodecSpecific.h264.dpb[8].used_for_reference, picParams->CodecSpecific.h264.dpb[9].used_for_reference, picParams->CodecSpecific.h264.dpb[10].used_for_reference, picParams->CodecSpecific.h264.dpb[11].used_for_reference, picParams->CodecSpecific.h264.dpb[12].used_for_reference, picParams->CodecSpecific.h264.dpb[13].used_for_reference, picParams->CodecSpecific.h264.dpb[14].used_for_reference, picParams->CodecSpecific.h264.dpb[15].used_for_reference);
	printf("DPB.FieldOrderCnt[0]:     %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", picParams->CodecSpecific.h264.dpb[0].FieldOrderCnt[0], picParams->CodecSpecific.h264.dpb[1].FieldOrderCnt[0], picParams->CodecSpecific.h264.dpb[2].FieldOrderCnt[0], picParams->CodecSpecific.h264.dpb[3].FieldOrderCnt[0], picParams->CodecSpecific.h264.dpb[4].FieldOrderCnt[0], picParams->CodecSpecific.h264.dpb[5].FieldOrderCnt[0], picParams->CodecSpecific.h264.dpb[6].FieldOrderCnt[0], picParams->CodecSpecific.h264.dpb[7].FieldOrderCnt[0], picParams->CodecSpecific.h264.dpb[8].FieldOrderCnt[0], picParams->CodecSpecific.h264.dpb[9].FieldOrderCnt[0], picParams->CodecSpecific.h264.dpb[10].FieldOrderCnt[0], picParams->CodecSpecific.h264.dpb[11].FieldOrderCnt[0], picParams->CodecSpecific.h264.dpb[12].FieldOrderCnt[0], picParams->CodecSpecific.h264.dpb[13].FieldOrderCnt[0], picParams->CodecSpecific.h264.dpb[14].FieldOrderCnt[0], picParams->CodecSpecific.h264.dpb[15].FieldOrderCnt[0]);
	printf("DPB.FieldOrderCnt[1]:     %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", picParams->CodecSpecific.h264.dpb[0].FieldOrderCnt[1], picParams->CodecSpecific.h264.dpb[1].FieldOrderCnt[1], picParams->CodecSpecific.h264.dpb[2].FieldOrderCnt[1], picParams->CodecSpecific.h264.dpb[3].FieldOrderCnt[1], picParams->CodecSpecific.h264.dpb[4].FieldOrderCnt[1], picParams->CodecSpecific.h264.dpb[5].FieldOrderCnt[1], picParams->CodecSpecific.h264.dpb[6].FieldOrderCnt[1], picParams->CodecSpecific.h264.dpb[7].FieldOrderCnt[1], picParams->CodecSpecific.h264.dpb[8].FieldOrderCnt[1], picParams->CodecSpecific.h264.dpb[9].FieldOrderCnt[1], picParams->CodecSpecific.h264.dpb[10].FieldOrderCnt[1], picParams->CodecSpecific.h264.dpb[11].FieldOrderCnt[1], picParams->CodecSpecific.h264.dpb[12].FieldOrderCnt[1], picParams->CodecSpecific.h264.dpb[13].FieldOrderCnt[1], picParams->CodecSpecific.h264.dpb[14].FieldOrderCnt[1], picParams->CodecSpecific.h264.dpb[15].FieldOrderCnt[1]);
	printf("fmo_aso_enable:           %d\n", picParams->CodecSpecific.h264.fmo_aso_enable);
	printf("num_slice_groups_minus1:  %d\n", picParams->CodecSpecific.h264.num_slice_groups_minus1);
	printf("fmo.pMb2SliceGroupMap:    %p\n", picParams->CodecSpecific.h264.fmo.pMb2SliceGroupMap);
	printf("****************\n");
	
	CUresult             result = CUDA_SUCCESS;
	CUVIDPROCPARAMS      procParams;
	memset(&procParams, 0, sizeof(procParams));
	procParams.progressive_frame = dispInfo->progressive_frame;
	procParams.second_field      = 0;
	procParams.top_field_first   = dispInfo->top_field_first;
	procParams.unpaired_field    = 0;
	
	unsigned long long devPtr = 0;
	unsigned int       pitch  = 0;
	result = u->cuvidMapVideoFrame64(u->decoder,
	                                 dispInfo->picture_index,
	                                 &devPtr,
	                                 &pitch,
	                                 &procParams);
	if(result != CUDA_SUCCESS){
		printf("Could not map picture successfully (%d)!\n", (int)result);
	}else{
		/* printf("Mapped picture successfully!\n"); */
	}
	
	result = u->cuvidUnmapVideoFrame64(u->decoder, devPtr);
	if(result != CUDA_SUCCESS){
		printf("Could not unmap picture successfully (%d)!\n", (int)result);
	}else{
		/* printf("Unmapped picture successfully!\n"); */
	}
#endif

#if 0
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
#endif

