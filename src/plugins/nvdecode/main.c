/* Includes */
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "cuviddec.h"

#include "benzina/benzina.h"
#include "benzina/plugins/nvdecode.h"
#include "kernels.h"


/* Defines */



/* Data Structures Forward Declarations and Typedefs */
typedef struct NVDECODE_RQ          NVDECODE_RQ;
typedef struct NVDECODE_BATCH       NVDECODE_BATCH;
typedef struct NVDECODE_READ_PARAMS NVDECODE_READ_PARAMS;
typedef struct NVDECODE_CTX         NVDECODE_CTX;



/* Data Structure & Enum Definitions */

/**
 * @brief Helper Thread Status.
 */

typedef enum NVDECODE_HLP_THRD_STATUS{
	THRD_NOT_RUNNING, /* Thread hasn't been spawned. */
	THRD_SPAWNED,     /* Thread has been spawned successfully with pthread_create(). */
	THRD_INITED,      /* Thread initialized successfully, waiting for others to
	                     do so as well. */
	THRD_RUNNING,     /* Thread running. */
	THRD_EXITING,     /* Thread is exiting. */
} NVDECODE_HLP_THRD_STATUS;


/**
 * @brief Context Status.
 */

typedef enum NVDECODE_CTX_STATUS{
	CTX_HELPERS_NOT_RUNNING, /* Context's helper threads are all not running. */
	CTX_HELPERS_RUNNING,     /* Context's helper threads are all running normally. */
	CTX_HELPERS_EXITING,     /* Context's helper threads are being asked to exit,
	                            or have begun doing so. */
} NVDECODE_CTX_STATUS;


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
 * @brief A structure containing the parameters for a disk read.
 */

struct NVDECODE_READ_PARAMS{
	int     fd;
	size_t  off;
	size_t  len;
	void*   ptr;
	ssize_t lenRead;
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
		NVDECODE_CTX_STATUS status;
		uint64_t lifecycle;
		struct{
			uint64_t batch;
			uint64_t token;
			uint64_t sample;
		} push;
		struct{
			uint64_t batch;
			uint64_t token;
			uint64_t sample;
		} pull;
		pthread_cond_t cond;
	} master;
	struct{
		NVDECODE_HLP_THRD_STATUS status;
		int err;
		uint64_t cnt;/* # of compressed images previously read from dataset. */
		pthread_t thrd;
		pthread_cond_t cond;
	} reader;
	struct{
		NVDECODE_HLP_THRD_STATUS status;
		int err;
		uint64_t cnt;/* # of compressed images previously pushed into decoder. */
		pthread_t thrd;
		pthread_cond_t cond;
	} feeder;
	struct{
		NVDECODE_HLP_THRD_STATUS status;
		int err;
		uint64_t cnt;/* # of decompressed images previously pulled out of decoder. */
		pthread_t thrd;
		pthread_cond_t cond;
		cudaStream_t cudaStream;
	} worker;
	
	/* Tensor geometry */
	int      deviceOrdinal;
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
	CUvideodecoder           decoder;
	uint32_t                 decoderInited;
	uint32_t                 decoderRefCnt;
	CUVIDDECODECAPS          decoderCaps;
	CUVIDDECODECREATEINFO    decoderInfo;
	CUVIDPICPARAMS*          picParams;
	uint64_t                 picParamTruncLen;
	uint32_t                 mallocRefCnt;
	NVDECODE_BATCH*          batch;
	NVDECODE_RQ*             request;
};



/* Static Function Prototypes */
BENZINA_PLUGIN_STATIC int   nvdecodeAbstime                (struct timespec* ts, double dt);
BENZINA_PLUGIN_STATIC int   nvdecodeSameLifecycle          (NVDECODE_CTX* ctx, uint64_t lifecycle);
BENZINA_PLUGIN_STATIC int   nvdecodeMasterThrdGetSubmRq    (NVDECODE_CTX* ctx, NVDECODE_RQ**    rqOut);
BENZINA_PLUGIN_STATIC int   nvdecodeMasterThrdGetSubmBt    (NVDECODE_CTX* ctx, NVDECODE_BATCH** batchOut);
BENZINA_PLUGIN_STATIC int   nvdecodeMasterThrdGetRetrBt    (NVDECODE_CTX* ctx, NVDECODE_BATCH** batchOut);
BENZINA_PLUGIN_STATIC int   nvdecodeMasterThrdSetStatus    (NVDECODE_CTX* ctx, NVDECODE_CTX_STATUS status);
BENZINA_PLUGIN_STATIC int   nvdecodeMasterThrdAwaitShutdown(NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeHelpersStart           (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeHelpersStop            (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeHelpersAllStatusIs     (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status);
BENZINA_PLUGIN_STATIC int   nvdecodeHelpersAnyStatusIs     (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status);
BENZINA_PLUGIN_STATIC int   nvdecodeHelpersShouldExitNow   (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeHelpersShouldExit      (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdInit         (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdAwaitAll     (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdContinue     (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdHasWork      (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdWait         (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdCore         (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdFillDataRd   (NVDECODE_CTX* ctx,
                                                            const NVDECODE_RQ*    rq,
                                                            NVDECODE_READ_PARAMS* rd);
BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdFillAuxRd    (NVDECODE_CTX* ctx,
                                                            const NVDECODE_RQ*    rq,
                                                            NVDECODE_READ_PARAMS* rd);
BENZINA_PLUGIN_STATIC void* nvdecodeReaderThrdMain         (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdSetStatus    (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status);
BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdGetCurrRq    (NVDECODE_CTX* ctx, NVDECODE_RQ** rqOut);
BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdInit         (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdAwaitAll     (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdContinue     (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdHasWork      (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdWait         (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdCore         (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC void* nvdecodeFeederThrdMain         (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdSetStatus    (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status);
BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdGetCurrRq    (NVDECODE_CTX* ctx, NVDECODE_RQ** rqOut);
BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdInit         (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdAwaitAll     (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdContinue     (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdHasWork      (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdWait         (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdCore         (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC void* nvdecodeWorkerThrdMain         (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC void  nvdecodeWorkerThrdCallback     (cudaStream_t  stream,
                                                            cudaError_t   status,
                                                            NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdSetStatus    (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status);
BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdGetCurrRq    (NVDECODE_CTX* ctx, NVDECODE_RQ** rqOut);
BENZINA_PLUGIN_STATIC int   nvdecodeSetDevice              (NVDECODE_CTX* ctx, const char* deviceId);
BENZINA_PLUGIN_STATIC int   nvdecodeAllocDataOpen          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeAllocPBParse           (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeAllocThreading         (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int   nvdecodeAllocCleanup           (NVDECODE_CTX* ctx, int ret);



/* Static Function Definitions */

/**
 * @brief Compute timespec for the absolute time NOW + dt.
 * @param [out] ts  The computed timespec.
 * @param [in]  dt  The time delta from NOW.
 * @return The return code from clock_gettime(CLOCK_REALTIME, ts).
 */

BENZINA_PLUGIN_STATIC int   nvdecodeAbstime                (struct timespec* ts, double dt){
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
 * @brief Are we still on the same lifecycle?
 * @param [in]  ctx
 * @param [in]  lifecycle
 * @return !0 if given lifecycle matches current one, 0 otherwise.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeSameLifecycle          (NVDECODE_CTX* ctx, uint64_t lifecycle){
	return ctx->master.lifecycle == lifecycle;
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

BENZINA_PLUGIN_STATIC int   nvdecodeMasterThrdGetSubmBt    (NVDECODE_CTX* ctx, NVDECODE_BATCH** batchOut){
	*batchOut = &ctx->batch[ctx->master.push.batch % ctx->multibuffering];
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

BENZINA_PLUGIN_STATIC int   nvdecodeMasterThrdGetRetrBt    (NVDECODE_CTX* ctx, NVDECODE_BATCH** batchOut){
	*batchOut = &ctx->batch[ctx->master.pull.batch % ctx->multibuffering];
	return 0;
}

/**
 * @brief Set master thread status.
 * @param [in]  ctx     The context in question.
 * @param [in]  status  The new status.
 * @return 
 */

BENZINA_PLUGIN_STATIC int   nvdecodeMasterThrdSetStatus    (NVDECODE_CTX* ctx, NVDECODE_CTX_STATUS status){
	ctx->master.status = status;
	pthread_cond_broadcast(&ctx->master.cond);
	pthread_cond_broadcast(&ctx->reader.cond);
	pthread_cond_broadcast(&ctx->feeder.cond);
	pthread_cond_broadcast(&ctx->worker.cond);
	return 0;
}

/**
 * @brief Wait for context to reach shutdown.
 * 
 * Called with the lock held. May release and reacquire lock.
 * 
 * @param [in]  ctx
 * @return 0 if desired status reached with no intervening helper thread lifecycle.
 *         !0 otherwise.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeMasterThrdAwaitShutdown(NVDECODE_CTX* ctx){
	uint64_t lifecycle = ctx->master.lifecycle;
	do{
		if(!nvdecodeSameLifecycle(ctx, lifecycle)){
			return -1;
		}
		if(ctx->master.status == CTX_HELPERS_NOT_RUNNING){
			return 0;
		}
	}while(pthread_cond_wait(&ctx->master.cond, &ctx->lock) == 0);
	return -3;
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

BENZINA_PLUGIN_STATIC int   nvdecodeMasterThrdGetSubmRq    (NVDECODE_CTX* ctx, NVDECODE_RQ**    rqOut){
	*rqOut = &ctx->request[ctx->master.push.sample % ctx->totalSlots];
	return 0;
}

/**
 * @brief Launch helper threads.
 * 
 * Called with the lock held. Must not be called from the helper threads.
 * 
 * @param [in]  ctx
 * @return 0 if threads already running or started successfully.
 *         !0 if threads exiting, or were not running and could not be started.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeHelpersStart           (NVDECODE_CTX* ctx){
	uint64_t       i;
	pthread_attr_t attr;
	
	switch(ctx->master.status){
		case CTX_HELPERS_EXITING: return -1;
		case CTX_HELPERS_RUNNING: return  0;
		default: break;
	}
	
	if(ctx->reader.err || ctx->feeder.err || ctx->worker.err){
		return -1;
	}
	
	memset(ctx->batch,   0, sizeof(*ctx->batch)   * ctx->multibuffering);
	memset(ctx->request, 0, sizeof(*ctx->request) * ctx->totalSlots);
	for(i=0;i<ctx->totalSlots;i++){
		ctx->request[i].picParams = &ctx->picParams[i];
		ctx->request[i].data      = NULL;
	}
	
	if(pthread_attr_init          (&attr)                          != 0){
		return -1;
	}
	if(pthread_attr_setstacksize  (&attr,                 64*1024) != 0 ||
	   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED) != 0){
		pthread_attr_destroy(&attr);
		return -1;
	}
	ctx->reader.status = pthread_create(&ctx->reader.thrd, &attr, (void*(*)(void*))nvdecodeReaderThrdMain, ctx) == 0 ? THRD_SPAWNED : THRD_NOT_RUNNING;
	ctx->feeder.status = pthread_create(&ctx->feeder.thrd, &attr, (void*(*)(void*))nvdecodeFeederThrdMain, ctx) == 0 ? THRD_SPAWNED : THRD_NOT_RUNNING;
	ctx->worker.status = pthread_create(&ctx->worker.thrd, &attr, (void*(*)(void*))nvdecodeWorkerThrdMain, ctx) == 0 ? THRD_SPAWNED : THRD_NOT_RUNNING;
	ctx->reader.err    = ctx->reader.status == THRD_NOT_RUNNING ? 1 : 0;
	ctx->feeder.err    = ctx->feeder.status == THRD_NOT_RUNNING ? 1 : 0;
	ctx->worker.err    = ctx->worker.status == THRD_NOT_RUNNING ? 1 : 0;
	pthread_attr_destroy(&attr);
	
	if(nvdecodeHelpersAllStatusIs(ctx, THRD_NOT_RUNNING)){
		return -1;
	}
	ctx->master.lifecycle++;
	if(nvdecodeHelpersAllStatusIs(ctx, THRD_SPAWNED)){
		nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_RUNNING);
	}else{
		nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_EXITING);
	}
	
	return 0;
}

/**
 * @brief Stop helper threads.
 * 
 * Called with the lock held. Must not be called from the helper threads.
 * May release and reacquire the lock.
 * 
 * @param [in]  ctx
 * @return 0 if threads not running or successfully stopped.
 *         !0 if lifecycle changes under our feet as we wait.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeHelpersStop        (NVDECODE_CTX* ctx){
	if(ctx->master.status == CTX_HELPERS_NOT_RUNNING){
		return 0;
	}
	
	nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_EXITING);
	return nvdecodeMasterThrdAwaitShutdown(ctx);
}

/**
 * @brief Whether all helpers have the given status.
 * 
 * @param [in]  ctx
 * @param [in]  status
 * @return Whether (!0) or not (0) all helper threads have the given status.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeHelpersAllStatusIs (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status){
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

BENZINA_PLUGIN_STATIC int   nvdecodeHelpersAnyStatusIs (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status){
	return ctx->reader.status == status ||
	       ctx->feeder.status == status ||
	       ctx->worker.status == status;
}

/**
 * @brief Whether all helpers should exit *immediately*.
 * 
 * @param [in]  ctx
 * @return 
 */

BENZINA_PLUGIN_STATIC int   nvdecodeHelpersShouldExitNow(NVDECODE_CTX* ctx){
	return ctx->reader.err ||
	       ctx->feeder.err ||
	       ctx->worker.err;
}

/**
 * @brief Whether all helpers should exit when the pipeline is empty.
 * 
 * @param [in]  ctx
 * @param [in]  now  Whether to exit *immediately* or after a pipeline flush.
 * @return 
 */

BENZINA_PLUGIN_STATIC int   nvdecodeHelpersShouldExit  (NVDECODE_CTX* ctx){
	return ctx->master.status == CTX_HELPERS_EXITING;
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
			if(ctx->request[i].data){
				free(ctx->request[i].data);
				ctx->request[i].data = NULL;
			}
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
		cuvidDestroyDecoder(ctx->decoder);
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
	if(nvdecodeHelpersShouldExitNow(ctx)){
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
		if(nvdecodeHelpersShouldExitNow(ctx)){return 0;}
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
		if(nvdecodeHelpersShouldExitNow(ctx)){
			break;
		}
		if(!nvdecodeReaderThrdHasWork(ctx)){
			if(nvdecodeHelpersShouldExit(ctx)){
				break;
			}else{
				continue;
			}
		}
		return 1;
	}while(nvdecodeReaderThrdWait(ctx));
	
	nvdecodeReaderThrdSetStatus(ctx, THRD_EXITING);
	nvdecodeMaybeReapMallocs   (ctx);
	nvdecodeReaderThrdSetStatus(ctx, THRD_NOT_RUNNING);
	return 0;
}

/**
 * @brief Does reader thread have work to do?
 * @param [in]  ctx  The context in question
 * @return !0 if thread has work to do; 0 otherwise.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdHasWork  (NVDECODE_CTX* ctx){
	return ctx->reader.cnt < ctx->master.push.sample;
}

/**
 * @brief Reader Wait.
 * @param [in]   ctx  The context
 * @return 1
 */

BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdWait     (NVDECODE_CTX* ctx){
	pthread_cond_wait(&ctx->reader.cond, &ctx->lock);
	return 1;
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
	NVDECODE_READ_PARAMS rd0 = {0}, rd1 = {0};
	NVDECODE_RQ*         rq;
	int                  readsDone;
	
	
	/* Get read parameters */
	nvdecodeReaderThrdGetCurrRq (ctx, &rq);
	if(nvdecodeReaderThrdFillDataRd(ctx, rq, &rd0) != 0 ||
	   nvdecodeReaderThrdFillAuxRd (ctx, rq, &rd1) != 0){
		return 0;
	}
	rq->data = rd0.ptr;
	
	
	/* Perform reads */
	pthread_mutex_unlock(&ctx->lock);
	rd0.lenRead = pread(rd0.fd, rd0.ptr, rd0.len, rd0.off);
	rd1.lenRead = pread(rd1.fd, rd1.ptr, rd1.len, rd1.off);
	pthread_mutex_lock(&ctx->lock);
	
	
	/* Handle any I/O problems */
	readsDone = (rd0.lenRead==(ssize_t)rd0.len) &&
	            (rd1.lenRead==(ssize_t)rd1.len);
	if(!readsDone){
		free(rq->data);
		rq->data = NULL;
		ctx->reader.err = 1;
		nvdecodeReaderThrdSetStatus(ctx, THRD_EXITING);
		return 0;
	}
	
	/* Otherwise, report success. */
	ctx->reader.cnt++;
	pthread_cond_broadcast(&ctx->feeder.cond);
	return 0;
}

/**
 * @brief Fill the dataset read parameters structure with the current sample's
 *        details.
 * @param [in]  ctx
 * @param [in]  rq
 * @param [out] rd
 * @return 0 if successful, !0 otherwise.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdFillDataRd(NVDECODE_CTX* ctx,
                                                         const NVDECODE_RQ*    rq,
                                                         NVDECODE_READ_PARAMS* rd){
	int ret;
	
	rd->ptr = NULL;
	ret = benzinaDatasetGetElement(ctx->dataset, rq->datasetIndex, &rd->off, &rd->len);
	if(ret != 0){
		ctx->reader.err = ret;
		nvdecodeReaderThrdSetStatus(ctx, THRD_EXITING);
		return ret;
	}
	rd->ptr = malloc(rd->len);
	if(!rd->ptr){
		ctx->reader.err = 1;
		nvdecodeReaderThrdSetStatus(ctx, THRD_EXITING);
		return 0;
	}
	rd->fd  = ctx->datasetBinFd;
	return 0;
}

/**
 * @brief Fill the auxiliary data read parameters structure with the current
 *        sample's details.
 * @param [in]  ctx
 * @param [in]  rq
 * @param [out] rd
 * @return 0 if successful, !0 otherwise.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdFillAuxRd (NVDECODE_CTX* ctx,
                                                         const NVDECODE_RQ*    rq,
                                                         NVDECODE_READ_PARAMS* rd){
	rd->fd  = ctx->datasetNvdecodeFd;
	rd->len = ctx->picParamTruncLen;
	rd->off = ctx->picParamTruncLen*rq->datasetIndex;
	rd->ptr = rq->picParams;
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

BENZINA_PLUGIN_STATIC int   nvdecodeReaderThrdSetStatus(NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status){
	ctx->reader.status = status;
	switch(status){
		case THRD_NOT_RUNNING:
			if(nvdecodeHelpersAllStatusIs(ctx, THRD_NOT_RUNNING)){
				nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_NOT_RUNNING);
			}else{
				nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_EXITING);
			}
		break;
		case THRD_EXITING:
			nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_EXITING);
		break;
		default:
		break;
	}
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
	int ret;
	
	if(nvdecodeHelpersShouldExitNow(ctx)){
		nvdecodeFeederThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
	
	ret = cudaSetDevice(ctx->deviceOrdinal);
	if(ret != cudaSuccess){
		ctx->feeder.err = ret;
		nvdecodeFeederThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
	
	#if NVDECODE >= 8001
	memset(&ctx->decoderCaps, 0, sizeof(ctx->decoderCaps));
	ctx->decoderCaps.eCodecType      = ctx->decoderInfo.CodecType;
	ctx->decoderCaps.eChromaFormat   = ctx->decoderInfo.ChromaFormat;
	ctx->decoderCaps.nBitDepthMinus8 = ctx->decoderInfo.bitDepthMinus8;
	ret = cuvidGetDecoderCaps(&ctx->decoderCaps);
	if(ret != CUDA_SUCCESS                                      ||
	   !ctx->decoderCaps.bIsSupported                           ||
	   ctx->decoderInfo.ulWidth  < ctx->decoderCaps.nMinWidth   ||
	   ctx->decoderInfo.ulWidth  > ctx->decoderCaps.nMaxWidth   ||
	   ctx->decoderInfo.ulHeight < ctx->decoderCaps.nMinHeight  ||
	   ctx->decoderInfo.ulHeight > ctx->decoderCaps.nMaxHeight  ||
	   ((ctx->decoderInfo.ulWidth*ctx->decoderInfo.ulHeight/256) > ctx->decoderCaps.nMaxMBCount)){
		ctx->feeder.err = ret;
		nvdecodeFeederThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
	#endif
	ret = cuvidCreateDecoder(&ctx->decoder, &ctx->decoderInfo);
	if(ret != CUDA_SUCCESS){
		ctx->feeder.err = ret;
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
		if(nvdecodeHelpersShouldExitNow(ctx)){return 0;}
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
		if(nvdecodeHelpersShouldExitNow(ctx)){
			break;
		}
		if(!nvdecodeFeederThrdHasWork(ctx)){
			if(nvdecodeHelpersShouldExit(ctx)){
				break;
			}else{
				continue;
			}
		}
		return 1;
	}while(nvdecodeFeederThrdWait(ctx));
	
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
 * @brief Does feeder thread have work to do?
 * @param [in]  ctx  The context in question
 * @return !0 if thread has work to do; 0 otherwise.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdHasWork  (NVDECODE_CTX* ctx){
	return ctx->feeder.cnt < ctx->reader.cnt;
}

/**
 * @brief Feeder Wait.
 * @param [in]   ctx  The context
 * @return 1
 */

BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdWait     (NVDECODE_CTX* ctx){
	pthread_cond_wait(&ctx->feeder.cond, &ctx->lock);
	return 1;
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
	ret = cuvidDecodePicture(ctx->decoder, pP);
	pthread_mutex_lock(&ctx->lock);
	
	/* Release data. */
	free(rq->data);
	rq->data = NULL;
	if(ret != CUDA_SUCCESS){
		ctx->feeder.err = ret;
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

BENZINA_PLUGIN_STATIC int   nvdecodeFeederThrdSetStatus(NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status){
	ctx->feeder.status = status;
	switch(status){
		case THRD_NOT_RUNNING:
			if(nvdecodeHelpersAllStatusIs(ctx, THRD_NOT_RUNNING)){
				nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_NOT_RUNNING);
			}else{
				nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_EXITING);
			}
		break;
		case THRD_EXITING:
			nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_EXITING);
		break;
		default:
		break;
	}
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
	int ret;
	
	if(nvdecodeHelpersShouldExitNow(ctx)){
		nvdecodeWorkerThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
	
	ret = cudaSetDevice(ctx->deviceOrdinal);
	if(ret != cudaSuccess){
		ctx->worker.err = ret;
		nvdecodeWorkerThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
	ret = cudaStreamCreate(&ctx->worker.cudaStream);
	if(ret != cudaSuccess){
		ctx->worker.err = ret;
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
		if(nvdecodeHelpersShouldExitNow(ctx)){return 0;}
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
		if(nvdecodeHelpersShouldExitNow(ctx)){
			break;
		}
		if(!nvdecodeWorkerThrdHasWork(ctx)){
			if(nvdecodeHelpersShouldExit(ctx)){
				break;
			}else{
				continue;
			}
		}
		return 1;
	}while(nvdecodeWorkerThrdWait(ctx));
	
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
 * @brief Does worker thread have work to do?
 * @param [in]  ctx  The context in question
 * @return !0 if thread has work to do; 0 otherwise.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdHasWork  (NVDECODE_CTX* ctx){
	return ctx->worker.cnt < ctx->feeder.cnt;
}

/**
 * @brief Worker Wait.
 * @param [in]   ctx  The context
 * @return 1
 */

BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdWait     (NVDECODE_CTX* ctx){
	pthread_cond_wait(&ctx->worker.cond, &ctx->lock);
	return 1;
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
	CUVIDPROCPARAMS    procParams;
	NVDECODE_RQ*       rq;
	unsigned long long srcPtr;
	unsigned           pitch;
	uint64_t           picIdx = 0;
	CUresult           ret;
	
	nvdecodeWorkerThrdGetCurrRq(ctx, &rq);
	memset(&procParams, 0, sizeof(procParams));
	procParams.progressive_frame = 1;
	procParams.second_field      = 0;
	procParams.top_field_first   = 0;
	procParams.unpaired_field    = 0;
	procParams.output_stream     = ctx->worker.cudaStream;
	picIdx = ctx->worker.cnt % ctx->decoderInfo.ulNumDecodeSurfaces;
	
	/**
	 * Drop the mutex and block on the decoder, then perform CUDA ops
	 * on the returned data. Then, reacquire lock.
	 */
	
	pthread_mutex_unlock(&ctx->lock);
	ret = cuvidMapVideoFrame(ctx->decoder, picIdx, &srcPtr, &pitch, &procParams);
	if(ret == CUDA_SUCCESS){
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
		cuvidUnmapVideoFrame(ctx->decoder, srcPtr);
	}
	pthread_mutex_lock(&ctx->lock);
	
	
	/* Handle errors. */
	if(ret != CUDA_SUCCESS){
		ctx->worker.err = ret;
		nvdecodeWorkerThrdSetStatus(ctx, THRD_EXITING);
		return 0;
	}
	
	
	/* Exit. */
	ctx->worker.cnt++;
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
		ctx->master.pull.sample++;
		pthread_cond_broadcast(&ctx->master.cond);
	}else{
		ctx->worker.err = 1;
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

BENZINA_PLUGIN_STATIC int   nvdecodeWorkerThrdSetStatus(NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status){
	ctx->worker.status = status;
	switch(status){
		case THRD_NOT_RUNNING:
			if(nvdecodeHelpersAllStatusIs(ctx, THRD_NOT_RUNNING)){
				nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_NOT_RUNNING);
			}else{
				nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_EXITING);
			}
		break;
		case THRD_EXITING:
			nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_EXITING);
		break;
		default:
		break;
	}
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
	if(!nvdecodeHelpersAllStatusIs(ctx, THRD_NOT_RUNNING)){
		return BENZINA_DATALOADER_ITER_ALREADYINITED;
	}
	
	
	/**
	 * If deviceId is NULL, select current device, whatever it may be. Otherwise,
	 * parse deviceId to figure out the device.
	 */
	if(!deviceId){
		ret = cudaGetDevice(&i);
		if(ret != cudaSuccess){return ret;}
	}else{
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
	ctx->deviceOrdinal = i;
	return 0;
}

/**
 * @brief Pull a completed batch of work from the pipeline.
 * 
 * Obviously, called with the lock held.
 * 
 * @param [in]  ctx      The iterator context in which.
 * @param [out] token    User data that was submitted at the corresponding
 *                       pushBatch().
 * @param [in]  block    Whether the wait should be blocking or not.
 * @param [in]  timeout  A maximum amount of time to wait for the batch of data,
 *                       in seconds. If timeout <= 0, wait indefinitely.
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_STATIC int   nvdecodeWaitBatchLocked    (NVDECODE_CTX*    ctx,
                                                        const void**     token,
                                                        int              block,
                                                        double           timeout){
	NVDECODE_BATCH* batch;
	struct timespec deadline;
	uint64_t        lifecycle;
	int             ret = 0;
	
	
	*token = NULL;
	if(timeout > 0){
		nvdecodeAbstime(&deadline, timeout);
	}
	
	lifecycle = ctx->master.lifecycle;
	do{
		if(!nvdecodeSameLifecycle(ctx, lifecycle)){return -2;}
		if(ctx->master.pull.batch >= ctx->master.push.batch){
			if(!block){return EAGAIN;}
			continue;
		}
		nvdecodeMasterThrdGetRetrBt(ctx, &batch);
		if(ctx->master.pull.sample >= batch->stopIndex){
			*token = batch->token;
			ctx->master.pull.batch++;
			ctx->master.pull.token++;
			return 0;
		}else{
			if(ctx->reader.err || ctx->feeder.err || ctx->worker.err){
				return -1;
			}
			if(!block){return EAGAIN;}
		}
	}while((ret = (timeout > 0 ? pthread_cond_timedwait(&ctx->master.cond, &ctx->lock, &deadline) :
	                             pthread_cond_wait     (&ctx->master.cond, &ctx->lock))) == 0);
	return ret;
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
	ctx->deviceOrdinal     = -1;
	ctx->defaults.S[0]     = ctx->defaults.S[1] = ctx->defaults.S[2] = 1.0;
	ctx->picParams         = NULL;
	ctx->request           = NULL;
	ctx->batch             = NULL;
	
	
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
	 * At this present time the mutex is held, but the reference count is 0.
	 * This makes us responsible for the destruction of the object.
	 * 
	 * Since we were the last to hold a reference to this context, we are
	 * guaranteed to succeed in tearing down the context's threads, due to
	 * there being no-one else to countermand the order. For the same reason,
	 * we are guaranteed that the current helper thread lifecycle is the last
	 * one, and a new one will not start under our feet while the lock is
	 * released.
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
	if(ctx->master.push.batch-ctx->master.pull.batch >= ctx->multibuffering){
		ret = -1;
	}else{
		nvdecodeMasterThrdGetSubmBt(ctx, &batch);
		batch->startIndex = ctx->master.push.sample;
		batch->stopIndex  = ctx->master.push.sample;
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

BENZINA_PLUGIN_HIDDEN int   nvdecodeSubmitBatch             (NVDECODE_CTX* ctx, const void* token){
	NVDECODE_BATCH* batch;
	
	pthread_mutex_lock(&ctx->lock);
	nvdecodeMasterThrdGetSubmBt(ctx, &batch);
	batch->token = token;
	ctx->master.push.batch++;
	ctx->master.push.token++;
	pthread_mutex_unlock(&ctx->lock);
	
	return 0;
}

/**
 * @brief Pull a completed batch of work from the pipeline.
 * 
 * @param [in]  ctx      The iterator context in which.
 * @param [out] token    User data that was submitted at the corresponding
 *                       pushBatch().
 * @param [in]  block    Whether the wait should be blocking or not.
 * @param [in]  timeout  A maximum amount of time to wait for the batch of data,
 *                       in seconds. If timeout <= 0, wait indefinitely.
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodeWaitBatch               (NVDECODE_CTX* ctx,
                                                             const void**  token,
                                                             int           block,
                                                             double        timeout){
	int ret;
	pthread_mutex_lock(&ctx->lock);
	ret = nvdecodeWaitBatchLocked(ctx, token, block, timeout);
	pthread_mutex_unlock(&ctx->lock);
	return ret;
}

/**
 * @brief Pull a token from the pipeline.
 * 
 * @param [in]  ctx      The iterator context in which.
 * @param [out] token    User data that was submitted at the corresponding
 *                       pushBatch().
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodeWaitToken               (NVDECODE_CTX* ctx, const void** token){
	int ret = 0;
	
	pthread_mutex_lock(&ctx->lock);
	if(ctx->master.pull.token < ctx->master.push.token){
		*token = ctx->batch[ctx->master.pull.token++ % ctx->multibuffering].token;
		if(ctx->master.pull.token == ctx->master.push.token){
			ctx->reader.err = 0;
			ctx->feeder.err = 0;
			ctx->worker.err = 0;
		}
		ret = 0;
	}else{
		*token = NULL;
		ret = EWOULDBLOCK;
	}
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

BENZINA_PLUGIN_HIDDEN int   nvdecodeDefineSample            (NVDECODE_CTX* ctx, uint64_t i, void* dstPtr){
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
 * @return 0 if submission successful and threads will soon handle it.
 *         !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodeSubmitSample            (NVDECODE_CTX* ctx){
	int ret;
	pthread_mutex_lock(&ctx->lock);
	if(ctx->master.status == CTX_HELPERS_EXITING){
		ret = nvdecodeMasterThrdAwaitShutdown(ctx);
		if(ret != 0){
			pthread_mutex_unlock(&ctx->lock);
			return -2;
		}
	}
	ctx->master.push.sample++;
	ret = nvdecodeHelpersStart(ctx);
	pthread_cond_broadcast(&ctx->reader.cond);
	pthread_mutex_unlock(&ctx->lock);
	return ret;
}

/**
 * @brief Retrieve the number of batch pushes into the context.
 * 
 * @param [in]  ctx
 * @param [out] out
 * @return 0
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodeGetNumPushes            (NVDECODE_CTX* ctx, uint64_t* out){
	pthread_mutex_lock(&ctx->lock);
	*out = ctx->master.push.batch;
	pthread_mutex_unlock(&ctx->lock);
	return 0;
}

/**
 * @brief Retrieve the number of batch pulls out of the context.
 * 
 * @param [in]  ctx
 * @param [out] out
 * @return 0
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodeGetNumPulls             (NVDECODE_CTX* ctx, uint64_t* out){
	pthread_mutex_lock(&ctx->lock);
	*out = ctx->master.pull.batch;
	pthread_mutex_unlock(&ctx->lock);
	return 0;
}

/**
 * @brief Retrieve the multibuffering depth of the context.
 * 
 * @param [in]  ctx
 * @param [out] out
 * @return 0
 */

BENZINA_PLUGIN_HIDDEN int   nvdecodeGetMultibuffering       (NVDECODE_CTX* ctx, uint64_t* out){
	pthread_mutex_lock(&ctx->lock);
	*out = ctx->multibuffering;
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
	NVDECODE_RQ* rq = &ctx->request[ctx->master.push.sample % ctx->totalSlots];
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
	NVDECODE_RQ* rq = &ctx->request[ctx->master.push.sample % ctx->totalSlots];
	memcpy(rq->B, B?B:ctx->defaults.B, sizeof(rq->B));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int   nvdecodeSetScale                (NVDECODE_CTX* ctx,
                                                             const float*  S){
	NVDECODE_RQ* rq = &ctx->request[ctx->master.push.sample % ctx->totalSlots];
	memcpy(rq->S, S?S:ctx->defaults.S, sizeof(rq->S));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int   nvdecodeSetOOBColor             (NVDECODE_CTX* ctx,
                                                             const float*  OOB){
	NVDECODE_RQ* rq = &ctx->request[ctx->master.push.sample % ctx->totalSlots];
	memcpy(rq->OOB, OOB?OOB:ctx->defaults.OOB, sizeof(rq->OOB));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int   nvdecodeSelectColorMatrix       (NVDECODE_CTX* ctx,
                                                             uint32_t      colorMatrix){
	NVDECODE_RQ* rq = &ctx->request[ctx->master.push.sample % ctx->totalSlots];
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
	.waitToken                = (void*)nvdecodeWaitToken,
	
	.defineSample             = (void*)nvdecodeDefineSample,
	.submitSample             = (void*)nvdecodeSubmitSample,
	
	.getNumPushes             = (void*)nvdecodeGetNumPushes,
	.getNumPulls              = (void*)nvdecodeGetNumPulls,
	.getMultibuffering        = (void*)nvdecodeGetMultibuffering,
	
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

