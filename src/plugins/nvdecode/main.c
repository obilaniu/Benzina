/* Includes */
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "cuviddec.h"

#include "benzina/benzina-old.h"
#include "benzina/plugins/nvdecode.h"
#include "benzina/iso/bmff-intops.h"
#include "benzina/itu/h26x.h"
#include "benzina/itu/h265.h"
#include "kernels.h"


/* Defines */



/* Data Structures Forward Declarations and Typedefs */
typedef struct timespec             TIMESPEC;
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
	CTX_HELPERS_JOINING,     /* Context's helper threads have exited, but must
	                            still be joined. */
} NVDECODE_CTX_STATUS;


/**
 * @brief A structure containing the parameters and status of an individual
 *        request for image loading.
 */

struct NVDECODE_RQ{
	NVDECODE_BATCH* batch;             /* Batch to which this request belongs. */
	uint64_t        datasetIndex;      /* Dataset index. */
	float*          devPtr;            /* Target destination on device. */
	void*           sample;            /* MP4 bytes. */
	uint64_t        location[2];       /* Image payload location. */
	uint64_t        config_location[2];/* Video configuration offset and length. */
	float           H[3][3];           /* Homography */
	float           B   [3];           /* Bias */
	float           S   [3];           /* Scale */
	float           OOB [3];           /* Out-of-bond color */
	uint32_t        colorMatrix;       /* Color matrix selection */
	uint8_t*        data;              /* Image payload; */
	uint8_t*        hvcCData;          /* hvcC payload; */
	CUVIDPICPARAMS* picParams;         /* Picture parameters. */
	TIMESPEC        T_s_submit;        /* Time this request was submitted. */
	TIMESPEC        T_s_start;         /* Time this request began processing. */
	TIMESPEC        T_s_read;          /* Time required for reading. */
	TIMESPEC        T_s_decode;        /* Time required for decoding. */
	TIMESPEC        T_s_postproc;      /* Time required for postprocessing. */
};

/**
 * @brief A structure containing batch status data.
 */

struct NVDECODE_BATCH{
	NVDECODE_CTX*   ctx;
	uint64_t        startIndex;  /* Number of first sample submitted. */
	uint64_t        stopIndex;   /*  */
	const void*     token;
	TIMESPEC        T_s_submit;  /* Time this request was submitted. */
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
 *   - Context: This structure. Manages a pipeline of image decoding.
 *   - Job:     A unit of work comprising a compressed image read, its decoding
 *              and postprocessing.
 *   - Batch:   A group of jobs.
 *   - Lock:    The context's Big Lock, controlling access to everything.
 *              Must NOT be held more than momentarily.
 */

struct NVDECODE_CTX{
	/**
	 * All-important dataset
	 */
	
	const BENZINA_DATASET* dataset;
	const char*            datasetFile;
	size_t                 datasetLen;
	int                    datasetFd;
	
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
BENZINA_PLUGIN_STATIC const void* nvdecodeReturnAndClear          (const void**  ptr);
BENZINA_PLUGIN_STATIC int         nvdecodeTimeMonotonic           (TIMESPEC*     t);
BENZINA_PLUGIN_STATIC int         nvdecodeTimeAdd                 (TIMESPEC*     t, const TIMESPEC* a, const TIMESPEC* b);
BENZINA_PLUGIN_STATIC int         nvdecodeTimeSub                 (TIMESPEC*     t, const TIMESPEC* a, const TIMESPEC* b);
BENZINA_PLUGIN_STATIC double      nvdecodeTimeToDouble            (const TIMESPEC* t);
BENZINA_PLUGIN_STATIC void        nvdecodeDoubleToTime            (TIMESPEC*     t, double d);
BENZINA_PLUGIN_STATIC int         nvdecodeSameLifecycle           (NVDECODE_CTX* ctx, uint64_t lifecycle);
BENZINA_PLUGIN_STATIC int         nvdecodeMasterThrdGetSubmRq     (NVDECODE_CTX* ctx, NVDECODE_RQ**    rqOut);
BENZINA_PLUGIN_STATIC int         nvdecodeMasterThrdGetSubmBt     (NVDECODE_CTX* ctx, NVDECODE_BATCH** batchOut);
BENZINA_PLUGIN_STATIC int         nvdecodeMasterThrdGetRetrBt     (NVDECODE_CTX* ctx, NVDECODE_BATCH** batchOut);
BENZINA_PLUGIN_STATIC int         nvdecodeMasterThrdSetStatus     (NVDECODE_CTX* ctx, NVDECODE_CTX_STATUS status);
BENZINA_PLUGIN_STATIC int         nvdecodeMasterThrdAwaitShutdown (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeHelpersStart            (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeHelpersStop             (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeHelpersJoin             (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeHelpersAllStatusIs      (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status);
BENZINA_PLUGIN_STATIC int         nvdecodeHelpersAnyStatusIs      (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status);
BENZINA_PLUGIN_STATIC int         nvdecodeHelpersShouldExitNow    (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeHelpersShouldExit       (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdInit          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdAwaitAll      (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdContinue      (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdHasWork       (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdWait          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdCore          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdFillDataRd    (NVDECODE_CTX* ctx,
                                                                   const NVDECODE_RQ*    rq,
                                                                   NVDECODE_READ_PARAMS* rd);
BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdFillConfigRd  (NVDECODE_CTX* ctx,
                                                                   const NVDECODE_RQ*    rq,
                                                                   NVDECODE_READ_PARAMS* rd);
BENZINA_PLUGIN_STATIC void*       nvdecodeReaderThrdMain          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdSetStatus     (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status);
BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdGetCurrRq     (NVDECODE_CTX* ctx, NVDECODE_RQ** rqOut);
BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdInit          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdAwaitAll      (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdContinue      (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdHasWork       (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdWait          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdCore          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC void*       nvdecodeFeederThrdMain          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdSetStatus     (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status);
BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdGetCurrRq     (NVDECODE_CTX* ctx, NVDECODE_RQ** rqOut);
BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdInit          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdAwaitAll      (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdContinue      (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdHasWork       (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdWait          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdCore          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC void*       nvdecodeWorkerThrdMain          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC void        nvdecodeWorkerThrdCallback      (cudaStream_t  stream,
                                                                   cudaError_t   status,
                                                                   NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdSetStatus     (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status);
BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdGetCurrRq     (NVDECODE_CTX* ctx, NVDECODE_RQ** rqOut);
BENZINA_PLUGIN_STATIC int         nvdecodeSetDevice               (NVDECODE_CTX* ctx, const char* deviceId);
BENZINA_PLUGIN_STATIC int         nvdecodeAllocDataOpen           (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeAllocThreading          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeAllocCleanup            (NVDECODE_CTX* ctx, int ret);


// Table 7-5
BENZINA_PLUGIN_STATIC const uint8_t DEFAULT_SCALING_LIST_16[16] = {
    16, 16, 16, 16,
    16, 16, 16, 16,
    16, 16, 16, 16,
    16, 16, 16, 16
};

// Table 7-6
BENZINA_PLUGIN_STATIC const uint8_t DEFAULT_SCALING_LIST_64[2][64] = {
    {16, 16, 16, 16, 16, 16, 16, 16,
     16, 16, 17, 16, 17, 16, 17, 18,
     17, 18, 18, 17, 18, 21, 19, 20,
     21, 20, 19, 21, 24, 22, 22, 24,
     24, 22, 22, 24, 25, 25, 27, 30,
     27, 25, 25, 29, 31, 35, 35, 31,
     29, 36, 41, 44, 41, 36, 47, 54,
     54, 47, 65, 70, 65, 88, 88, 115},

    {16, 16, 16, 16, 16, 16, 16, 16,
     16, 16, 17, 17, 17, 17, 17, 18,
     18, 18, 18, 18, 18, 20, 20, 20,
     20, 20, 20, 20, 24, 24, 24, 24,
     24, 24, 24, 24, 25, 25, 25, 25,
     25, 25, 25, 28, 28, 28, 28, 28,
     28, 33, 33, 33, 33, 33, 41, 41,
     41, 41, 54, 54, 54, 71, 71, 91}
};

BENZINA_PLUGIN_STATIC const uint8_t* DEFAULT_SCALING_LIST_4x4[6] = {
    DEFAULT_SCALING_LIST_16,
    DEFAULT_SCALING_LIST_16,
    DEFAULT_SCALING_LIST_16,
    DEFAULT_SCALING_LIST_16,
    DEFAULT_SCALING_LIST_16,
    DEFAULT_SCALING_LIST_16
};

BENZINA_PLUGIN_STATIC const uint8_t* DEFAULT_SCALING_LIST_8x8[6] = {
    DEFAULT_SCALING_LIST_64[0],
    DEFAULT_SCALING_LIST_64[0],
    DEFAULT_SCALING_LIST_64[0],
    DEFAULT_SCALING_LIST_64[1],
    DEFAULT_SCALING_LIST_64[1],
    DEFAULT_SCALING_LIST_64[1]
};

BENZINA_PLUGIN_STATIC const uint8_t** DEFAULT_SCALING_LIST_16x16 = DEFAULT_SCALING_LIST_8x8;

// According to spec "7.3.4 Scaling list data syntax",
// we just use scalingList32x32[0] and scalingList32x32[3];
// as spec "7.4.5 Scaling list data semantics",
// matrixId as the index of scalingList32x32 can be equal to 0 or 3;
// so only need to define scalingList32x32[2][] and have the index == matrixId / 3
BENZINA_PLUGIN_STATIC const uint8_t* DEFAULT_SCALING_LIST_32x32[2] = {
    DEFAULT_SCALING_LIST_64[0],
    DEFAULT_SCALING_LIST_64[1]
};


/* Static Function Definitions */

/**
 * @brief Read pointer at the specified location, return it and clear its source.
 * @param [in]  ptrPtr  The pointer to the pointer to be read, returned and cleared.
 * @return *ptrPtr
 */

BENZINA_PLUGIN_STATIC const void* nvdecodeReturnAndClear          (const void**  ptrPtr){
	const void* ptr = *ptrPtr;
	*ptrPtr = NULL;
	return ptr;
}

/**
 * @brief Get current monotonic time using high-resolution counter.
 * 
 * Monotonic time is unsettable and always-increasing (monotonic), but it may tick
 * slightly faster than or slower than 1s/s if a clock-slewing time adjustment is
 * in progress (such as commanded by adjtime() or NTP).
 * 
 * @param [out] t  The current monotonic time.
 * @return The return code from clock_gettime(CLOCK_MONOTONIC, t).
 */

BENZINA_PLUGIN_STATIC int         nvdecodeTimeMonotonic           (TIMESPEC* t){
	return clock_gettime(CLOCK_MONOTONIC, t);
}

/**
 * @brief Add times a and b together and store sum into t.
 * 
 * The output t is normalized such that tv_nsec is always in [0, 1e9), but the
 * tv_sec field is unconstrained.
 * 
 * t,a,b may all alias each other.
 * 
 * @param [out] t = a+b
 * @param [in]  a
 * @param [in]  b
 * @return 0
 */

BENZINA_PLUGIN_STATIC int         nvdecodeTimeAdd                 (TIMESPEC*       t,
                                                                   const TIMESPEC* a,
                                                                   const TIMESPEC* b){
	TIMESPEC an, bn, d;
	const int64_t GIGA = (int64_t)1*1000*1000*1000;
	
	an.tv_sec  = a->tv_nsec/GIGA;
	bn.tv_sec  = b->tv_nsec/GIGA;
	an.tv_nsec = a->tv_nsec - an.tv_sec*GIGA;
	bn.tv_nsec = b->tv_nsec - bn.tv_sec*GIGA;
	d.tv_sec   = a->tv_sec + an.tv_sec + b->tv_sec + bn.tv_sec;
	d.tv_nsec  = an.tv_nsec            + bn.tv_nsec;
	while(d.tv_nsec < 0){
		d.tv_sec  -=    1;
		d.tv_nsec += GIGA;
	}
	while(d.tv_nsec >= GIGA){
		d.tv_sec  +=    1;
		d.tv_nsec -= GIGA;
	}
	*t = d;
	
	return 0;
}

/**
 * @brief Subtract time b from a and store difference into t.
 * 
 * The output t is normalized such that tv_nsec is always in [0, 1e9), but the
 * tv_sec field is unconstrained.
 * 
 * t,a,b may all alias each other.
 * 
 * @param [out] t = a-b
 * @param [in]  a
 * @param [in]  b
 * @return 0
 */

BENZINA_PLUGIN_STATIC int         nvdecodeTimeSub                 (TIMESPEC*       t,
                                                                   const TIMESPEC* a,
                                                                   const TIMESPEC* b){
	TIMESPEC an, bn, d;
	const int64_t GIGA = (int64_t)1*1000*1000*1000;
	
	an.tv_sec  = a->tv_nsec/GIGA;
	bn.tv_sec  = b->tv_nsec/GIGA;
	an.tv_nsec = a->tv_nsec - an.tv_sec*GIGA;
	bn.tv_nsec = b->tv_nsec - bn.tv_sec*GIGA;
	d.tv_sec   = a->tv_sec + an.tv_sec - b->tv_sec - bn.tv_sec;
	d.tv_nsec  = an.tv_nsec            - bn.tv_nsec;
	while(d.tv_nsec < 0){
		d.tv_sec  -=    1;
		d.tv_nsec += GIGA;
	}
	while(d.tv_nsec >= GIGA){
		d.tv_sec  +=    1;
		d.tv_nsec -= GIGA;
	}
	*t = d;
	
	return 0;
}

/**
 * @brief Convert time to double.
 * @param [in] t  The time or time-delta to convert.
 * @return Double-precision floating-point value, in seconds.
 */

BENZINA_PLUGIN_STATIC double      nvdecodeTimeToDouble            (const TIMESPEC* t){
	TIMESPEC d;
	const int64_t GIGA = (int64_t)1*1000*1000*1000;
	
	d.tv_sec  = t->tv_nsec/GIGA;
	d.tv_nsec = t->tv_nsec - d.tv_sec*GIGA;
	d.tv_sec += t->tv_sec;
	
	while(d.tv_nsec < 0){
		d.tv_sec  -=    1;
		d.tv_nsec += GIGA;
	}
	
	/**
	 * The following code ensures that positive and negative times of equal magnitude
	 * render to the same-magnitude but oppositive-sign double-precision floating-point
	 * number, even after being canonicalized to d.tv_nsec in [0, 1e9). Otherwise,
	 * unpleasant surprises might occur when comparing such times.
	 */
	
	if(d.tv_sec < 0 && d.tv_nsec != 0){
		d.tv_nsec = GIGA - d.tv_nsec;
		d.tv_sec  =   -1 - d.tv_sec;
		return -d.tv_sec - 1e-9*d.tv_nsec;
	}else{
		return +d.tv_sec + 1e-9*d.tv_nsec;
	}
}

/**
 * @brief Convert double to time.
 * @param [out] t  The output time.
 * @param [in]  d  The double to convert.
 */

BENZINA_PLUGIN_STATIC void        nvdecodeDoubleToTime            (TIMESPEC* t, double d){
	double i=floor(d), f=d-i;
	const int64_t GIGA = (int64_t)1*1000*1000*1000;
	
	t->tv_nsec = GIGA*f;
	t->tv_sec  = i;
	if(t->tv_nsec >= GIGA){
		t->tv_nsec -= GIGA;
		t->tv_sec  +=    1;
	}
}

/**
 * @brief Are we still on the same lifecycle?
 * @param [in]  ctx
 * @param [in]  lifecycle
 * @return !0 if given lifecycle matches current one, 0 otherwise.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeSameLifecycle           (NVDECODE_CTX* ctx, uint64_t lifecycle){
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

BENZINA_PLUGIN_STATIC int         nvdecodeMasterThrdGetSubmBt     (NVDECODE_CTX* ctx, NVDECODE_BATCH** batchOut){
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

BENZINA_PLUGIN_STATIC int         nvdecodeMasterThrdGetRetrBt     (NVDECODE_CTX* ctx, NVDECODE_BATCH** batchOut){
	*batchOut = &ctx->batch[ctx->master.pull.batch % ctx->multibuffering];
	return 0;
}

/**
 * @brief Set master thread status.
 * @param [in]  ctx     The context in question.
 * @param [in]  status  The new status.
 * @return 
 */

BENZINA_PLUGIN_STATIC int         nvdecodeMasterThrdSetStatus     (NVDECODE_CTX* ctx, NVDECODE_CTX_STATUS status){
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

BENZINA_PLUGIN_STATIC int         nvdecodeMasterThrdAwaitShutdown (NVDECODE_CTX* ctx){
	uint64_t lifecycle = ctx->master.lifecycle;
	do{
		if(!nvdecodeSameLifecycle(ctx, lifecycle)){
			return -1;
		}
		if(ctx->master.status == CTX_HELPERS_JOINING){
			nvdecodeHelpersJoin(ctx);
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

BENZINA_PLUGIN_STATIC int         nvdecodeMasterThrdGetSubmRq     (NVDECODE_CTX* ctx, NVDECODE_RQ**    rqOut){
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

BENZINA_PLUGIN_STATIC int         nvdecodeHelpersStart            (NVDECODE_CTX* ctx){
	uint64_t       i;
	pthread_attr_t attr;
	sigset_t       oldset, allblockedset;
	
	switch(ctx->master.status){
		case CTX_HELPERS_NOT_RUNNING: break;
		case CTX_HELPERS_JOINING:     nvdecodeHelpersJoin(ctx); break;
		case CTX_HELPERS_EXITING:     return -1;
		case CTX_HELPERS_RUNNING:     return  0;
	}
	
	if(ctx->reader.err || ctx->feeder.err || ctx->worker.err){
		return -1;
	}
	
	memset(ctx->batch,   0, sizeof(*ctx->batch)   * ctx->multibuffering);
	memset(ctx->request, 0, sizeof(*ctx->request) * ctx->totalSlots);
	for(i=0;i<ctx->totalSlots;i++){
		ctx->request[i].picParams = &ctx->picParams[i];
		ctx->request[i].sample    = NULL;
		ctx->request[i].data      = NULL;
		ctx->request[i].hvcCData  = NULL;
	}
	
	if(pthread_attr_init          (&attr)                          != 0){
		return -1;
	}
	if(pthread_attr_setstacksize  (&attr,                 64*1024) != 0 ||
	   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE) != 0){
		pthread_attr_destroy(&attr);
		return -1;
	}
	sigfillset(&allblockedset);
	pthread_sigmask(SIG_SETMASK, &allblockedset, &oldset);
	ctx->reader.status = pthread_create(&ctx->reader.thrd, &attr, (void*(*)(void*))nvdecodeReaderThrdMain, ctx) == 0 ? THRD_SPAWNED : THRD_NOT_RUNNING;
	ctx->feeder.status = pthread_create(&ctx->feeder.thrd, &attr, (void*(*)(void*))nvdecodeFeederThrdMain, ctx) == 0 ? THRD_SPAWNED : THRD_NOT_RUNNING;
	ctx->worker.status = pthread_create(&ctx->worker.thrd, &attr, (void*(*)(void*))nvdecodeWorkerThrdMain, ctx) == 0 ? THRD_SPAWNED : THRD_NOT_RUNNING;
	pthread_sigmask(SIG_SETMASK, &oldset, NULL);
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

BENZINA_PLUGIN_STATIC int         nvdecodeHelpersStop             (NVDECODE_CTX* ctx){
	switch(ctx->master.status){
		case CTX_HELPERS_NOT_RUNNING:
		case CTX_HELPERS_JOINING:
			return nvdecodeHelpersJoin(ctx);
		default:
			nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_EXITING);
			return nvdecodeMasterThrdAwaitShutdown(ctx);
		case CTX_HELPERS_EXITING:
			return nvdecodeMasterThrdAwaitShutdown(ctx);
	}
}

/**
 * @brief Join helper threads.
 * 
 * Called with the lock held. Must not be called from the helper threads.
 * On successful return, all helper threads are truly no longer running and
 * have been joined, and the context is in state NOT_RUNNING.
 * 
 * Idempotent.
 * 
 * @param [in]  ctx
 * @return 0 if threads successfully joined, or not running in first place.
 *         !0 if threads were not ready to be joined.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeHelpersJoin             (NVDECODE_CTX* ctx){
	switch(ctx->master.status){
		case CTX_HELPERS_JOINING:
			pthread_join(ctx->reader.thrd, NULL);
			pthread_join(ctx->feeder.thrd, NULL);
			pthread_join(ctx->worker.thrd, NULL);
			nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_NOT_RUNNING);
			return 0;
		case CTX_HELPERS_NOT_RUNNING:
			return 0;
		default:
			return -1;
	}
}

/**
 * @brief Whether all helpers have the given status.
 * 
 * @param [in]  ctx
 * @param [in]  status
 * @return Whether (!0) or not (0) all helper threads have the given status.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeHelpersAllStatusIs      (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status){
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

BENZINA_PLUGIN_STATIC int         nvdecodeHelpersAnyStatusIs      (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status){
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

BENZINA_PLUGIN_STATIC int         nvdecodeHelpersShouldExitNow    (NVDECODE_CTX* ctx){
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

BENZINA_PLUGIN_STATIC int         nvdecodeHelpersShouldExit       (NVDECODE_CTX* ctx){
	return ctx->master.status == CTX_HELPERS_EXITING;
}

/**
 * @brief Maybe reap leftover malloc()'s from the reader.
 * @param [in]  ctx  The context in question.
 * @return 0.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeMaybeReapMallocs        (NVDECODE_CTX* ctx){
	uint64_t i;
	
	if(!--ctx->mallocRefCnt){
		for(i=0;i<ctx->totalSlots;i++){
			if(ctx->request[i].sample){
//				free(ctx->request[i].sample);
				ctx->request[i].sample = NULL;
			}
			if(ctx->request[i].data){
//				free(ctx->request[i].data);
				ctx->request[i].data = NULL;
			}
			if(ctx->request[i].hvcCData){
//				free(ctx->request[i].hvcCData);
				ctx->request[i].hvcCData = NULL;
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
 * Called with the lock held. Will release the lock and reacquire it.
 * 
 * @param [in]  ctx  The context in question.
 * @return 0.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeMaybeReapDecoder        (NVDECODE_CTX* ctx){
	CUvideodecoder decoder = ctx->decoder;
	
	if(!--ctx->decoderRefCnt && ctx->decoderInited){
		ctx->decoderInited = 0;
		
		/**
		 * We are forced to release the lock here, because deep inside
		 * cuvidDestroyDecoder(), there is a call to cuCtxSynchronize(). If
		 * we do not release the mutex, it is possible for deadlock to occur.
		 */
		
		pthread_mutex_unlock(&ctx->lock);
		cuvidDestroyDecoder (decoder);
		pthread_mutex_lock  (&ctx->lock);
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

BENZINA_PLUGIN_STATIC void*       nvdecodeReaderThrdMain          (NVDECODE_CTX* ctx){
	pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
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

BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdInit          (NVDECODE_CTX* ctx){
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

BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdAwaitAll      (NVDECODE_CTX* ctx){
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

BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdContinue      (NVDECODE_CTX* ctx){
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

BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdHasWork       (NVDECODE_CTX* ctx){
	return ctx->reader.cnt < ctx->master.push.sample;
}

/**
 * @brief Reader Wait.
 * @param [in]   ctx  The context
 * @return 1
 */

BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdWait          (NVDECODE_CTX* ctx){
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

BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdCore          (NVDECODE_CTX* ctx){
//	NVDECODE_READ_PARAMS rd0 = {0}, rd1 = {0};
	NVDECODE_RQ*         rq;
//	int                  readsDone;
	
	
	/* Get read parameters */
	nvdecodeReaderThrdGetCurrRq (ctx, &rq);

//	if(nvdecodeReaderThrdFillDataRd  (ctx, rq, &rd0) != 0 ||
//	   nvdecodeReaderThrdFillConfigRd(ctx, rq, &rd1) != 0){
//		return 0;
//	}
	
	
//	/* Perform reads */
//	pthread_mutex_unlock(&ctx->lock);
//	rd0.lenRead = pread(rd0.fd, rd0.ptr, rd0.len, rd0.off);
//	rd1.lenRead = pread(rd1.fd, rd1.ptr, rd1.len, rd1.off);
//	pthread_mutex_lock(&ctx->lock);
	
	
	/* Handle any I/O problems */
//	readsDone = (rd0.lenRead==(ssize_t)rd0.len) &&
//	            (rd1.lenRead==(ssize_t)rd1.len);
//	if(!readsDone){
//		free(rd0.ptr);
//		free(rd1.ptr);
//		ctx->reader.err = 1;
//		nvdecodeReaderThrdSetStatus(ctx, THRD_EXITING);
//		return 0;
//	}
	
	/* Otherwise, report success. */
	rq->data                         = &((uint8_t*)rq->sample)[rq->location[0]];
	rq->hvcCData                     = &((uint8_t*)rq->sample)[rq->config_location[0]];
	ctx->reader.cnt++;
    // Bitstream data
	rq->picParams->pBitstreamData    = &((const uint8_t*)rq->sample)[rq->location[0]];
	rq->picParams->nBitstreamDataLen = (uint32_t)rq->location[1];
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

BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdFillDataRd    (NVDECODE_CTX* ctx,
                                                                   const NVDECODE_RQ*    rq,
                                                                   NVDECODE_READ_PARAMS* rd){
	rd->ptr = NULL;
	rd->off = rq->location[0];
	rd->len = rq->location[1];
	rd->ptr = malloc(rd->len);
	if(!rd->ptr){
		ctx->reader.err = 1;
		nvdecodeReaderThrdSetStatus(ctx, THRD_EXITING);
		return 0;
	}
	rd->fd = ctx->datasetFd;
	return 0;
}

/**
 * @brief Fill the dataset video configuration parameters structure with the
 *        current sample's details.
 * @param [in]  ctx
 * @param [in]  rq
 * @param [out] rd
 * @return 0 if successful, !0 otherwise.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdFillConfigRd  (NVDECODE_CTX* ctx,
                                                                   const NVDECODE_RQ*    rq,
                                                                   NVDECODE_READ_PARAMS* rd){
	rd->ptr = NULL;
	rd->off = rq->config_location[0];
	rd->len = rq->config_location[1];
	rd->ptr = malloc(rd->len);
	if(!rd->ptr){
		ctx->reader.err = 1;
		nvdecodeReaderThrdSetStatus(ctx, THRD_EXITING);
		return 0;
	}
	rd->fd = ctx->datasetFd;
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

BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdSetStatus     (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status){
	ctx->reader.status = status;
	switch(status){
		case THRD_NOT_RUNNING:
			if(nvdecodeHelpersAllStatusIs(ctx, THRD_NOT_RUNNING)){
				if(ctx->master.status == CTX_HELPERS_EXITING){
					nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_JOINING);
				}else{
					nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_NOT_RUNNING);
				}
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

BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdGetCurrRq     (NVDECODE_CTX* ctx, NVDECODE_RQ** rqOut){
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

BENZINA_PLUGIN_STATIC void*       nvdecodeFeederThrdMain          (NVDECODE_CTX* ctx){
	pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
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

BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdInit          (NVDECODE_CTX* ctx){
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

BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdAwaitAll      (NVDECODE_CTX* ctx){
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

BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdContinue      (NVDECODE_CTX* ctx){
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

BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdHasWork       (NVDECODE_CTX* ctx){
	return ctx->feeder.cnt < ctx->reader.cnt &&
	       ctx->feeder.cnt < ctx->worker.cnt + ctx->decoderInfo.ulNumDecodeSurfaces;
}

/**
 * @brief Feeder Wait.
 * @param [in]   ctx  The context
 * @return 1
 */

BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdWait          (NVDECODE_CTX* ctx){
	pthread_cond_wait(&ctx->feeder.cond, &ctx->lock);
	return 1;
}

void skip_profile_tier_level(BENZ_ITU_H26XBS* bitstream, uint8_t max_sub_layers_minus1)
{
    /* Skips profile_tier_level() efficiently. */
    // As spec '7.3.3 Profile, tier and level syntax'
    // general_profile_space;                    2 bits
    // general_tier_flag;                        1 bit
    // general_profile_idc;                      5 bits
    // general_profile_compatibility_flag[32];   32 bits
    // general_progressive_source_flag;          1 bit
    // general_interlaced_source_flag;           1 bit
    // general_non_packed_constraint_flag;       1 bit
    // general_frame_only_constraint_flag;       1 bit
    // general_max_12bit_constraint_flag;        43 bits
    // general_max_10bit_constraint_flag;
    // general_max_8bit_constraint_flag;
    // general_max_422chroma_constraint_flag;
    // general_max_420chroma_constraint_flag;
    // general_max_monochrome_constraint_flag;
    // general_intra_constraint_flag;
    // general_one_picture_only_constraint_flag;
    // general_lower_bit_rate_constraint_flag;
    // general_reserved_zero_34bits;
    // general_reserved_zero_43bits;
    // general_inbld_flag;                       1 bit
    // general_reserved_zero_bit;
    // general_level_idc;                        8 bits
    // total                                     96 bits
    benz_itu_h26xbs_bigskip(bitstream, 96);
    if (max_sub_layers_minus1)
    {
        uint16_t sub_layer_x_present_flags = benz_itu_h26xbs_read_un(bitstream, 16);
        uint32_t sub_layer_profile_present_count = benz_popcnt64(sub_layer_x_present_flags & 0xAAA8);   // pop count odd bits
        uint32_t sub_layer_level_present_count = benz_popcnt64(sub_layer_x_present_flags & 0x5554);     // pop count even bits

        // As spec '7.3.3 Profile, tier and level syntax'
        // general_profile_space;                    2 bits
        // general_tier_flag;                        1 bit
        // general_profile_idc;                      5 bits
        // general_profile_compatibility_flag[32];   32 bits
        // general_progressive_source_flag;          1 bit
        // general_interlaced_source_flag;           1 bit
        // general_non_packed_constraint_flag;       1 bit
        // general_frame_only_constraint_flag;       1 bit
        // general_max_12bit_constraint_flag;        43 bits
        // general_max_10bit_constraint_flag;
        // general_max_8bit_constraint_flag;
        // general_max_422chroma_constraint_flag;
        // general_max_420chroma_constraint_flag;
        // general_max_monochrome_constraint_flag;
        // general_intra_constraint_flag;
        // general_one_picture_only_constraint_flag;
        // general_lower_bit_rate_constraint_flag;
        // general_reserved_zero_34bits;
        // general_reserved_zero_43bits;
        // general_inbld_flag;                       1 bit
        // general_reserved_zero_bit;
        // total                                     88 bits

        // general_level_idc;                        8 bits
        // total                                     8 bits
        benz_itu_h26xbs_bigskip(bitstream, 88 * sub_layer_profile_present_count +
                                            8 * sub_layer_level_present_count);
    }
}

void set_default_scaling_list(uint8_t* dstList, uint8_t* dstDcList, uint8_t sizeId, uint8_t matrixIdx)
{
    // Table 7-3-Specification of siezId
    switch (sizeId) {
    case 0: // 4x4
        memcpy(dstList, DEFAULT_SCALING_LIST_4x4[matrixIdx], 16);
        break;
    case 1: // 8x8
    case 2: // 16x16
        memcpy(dstList, DEFAULT_SCALING_LIST_8x8[matrixIdx], 64);
        break;
    case 3: // 32x32
        memcpy(dstList, DEFAULT_SCALING_LIST_32x32[matrixIdx], 64);
        break;
//    default:
//        ERROR("Can't get the scaling list by sizeId(%d)", sizeId);
//        return false;
    }

    if (sizeId > 1)
    {
        dstDcList[matrixIdx] = 16;
    }

//    return true;
}

void decode_scaling_list_data(CUVIDHEVCPICPARAMS* hevcPP, BENZ_ITU_H26XBS* bitstream)
{
#ifndef MIN
#define MIN(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })
#endif // MIN

    uint8_t* dstDcList = NULL;
    uint8_t* dstList = NULL;
    uint8_t* refList = NULL;

    size_t size = 64;
    uint8_t refMatrixIdx = 0;
    int scaling_list_pred_mode_flag = 0;
    uint8_t scaling_list_pred_matrix_id_delta = 0;
    uint8_t nextCoef;
    uint8_t coefNum;
    int16_t scaling_list_delta_coef;

    for (uint32_t sizeId = 0; sizeId < 4; sizeId++)
    {
        // as spec "7.3.4 Scaling list data syntax" and Table 7-4,
        // Since CUVIDHEVCPICPARAMS.ScalingList32x32[2], if sizeId == 3, we make matrixIdx range
        // within [0, 1] instead of [0, 5], thus making matrixId = matrixIdx * 3
        uint32_t maxMatrixIdx = (sizeId == 3) ? 2 : 6;
        for (uint32_t matrixIdx = 0; matrixIdx < maxMatrixIdx; matrixIdx++)
        {
            size = 64;
            // Table 7-3
            switch (sizeId)
            {
            case 0: // 4x4
                dstList = hevcPP->ScalingList4x4[matrixIdx];
                size = 16;
                break;
            case 1: // 8x8
                dstList = hevcPP->ScalingList8x8[matrixIdx];
                break;
            case 2: // 16x16
                dstList = hevcPP->ScalingList16x16[matrixIdx];
                dstDcList = hevcPP->ScalingListDCCoeff16x16;
                break;
            case 3: // 32x32
                dstList = hevcPP->ScalingList32x32[matrixIdx];
                dstDcList = hevcPP->ScalingListDCCoeff32x32;
            }

            scaling_list_pred_mode_flag = benz_itu_h26xbs_read_un(bitstream, 1);
            if (!scaling_list_pred_mode_flag)
            {
                // "7.4.5 Scaling list data semantics"
                scaling_list_pred_matrix_id_delta = benz_itu_h26xbs_read_ue(bitstream); // estimation [0, 5] or [0, 1] if sizeId == 3

                if (!scaling_list_pred_matrix_id_delta)
                {
                    set_default_scaling_list(dstList, dstDcList, sizeId, matrixIdx);
                }
                else
                {
                    //7-42
                    // Since CUVIDHEVCPICPARAMS.ScalingList32x32[2], if sizeId == 3, we make refMatrixIdx range
                    // within [0, 1] instead of [0, 5], thus making refMatrixId = refMatrixIdx * 3
                    refMatrixIdx = matrixIdx - scaling_list_pred_matrix_id_delta;
                    // get referrence list
                    switch (sizeId)
                    {
                    case 0: // 4x4
                        refList = hevcPP->ScalingList4x4[refMatrixIdx];
                        break;
                    case 1: // 8x8
                        refList = hevcPP->ScalingList8x8[refMatrixIdx];
                        break;
                    case 2: // 16x16
                        refList = hevcPP->ScalingList16x16[refMatrixIdx];
                        break;
                    case 3: // 32x32
                        refList = hevcPP->ScalingList32x32[refMatrixIdx];
                    }

                    for (uint32_t i = 0; i < size; i++)
                    {
                        dstList[i] = refList[i];
                    }

                    if (sizeId > 1)
                    {
                        dstDcList[matrixIdx] = dstDcList[refMatrixIdx];
                    }
                }
            }
            else
            {
                nextCoef = 8;
                coefNum = MIN(64, (1 << (4 + (sizeId << 1))));

                if (sizeId > 1)
                {
                    int32_t scaling_list_dc_coef_minus8;
                    scaling_list_dc_coef_minus8 = benz_itu_h26xbs_read_se(bitstream); // [-7, 247]
                    dstDcList[matrixIdx] = scaling_list_dc_coef_minus8 + 8;
                    nextCoef = dstDcList[matrixIdx];
                }

                for (uint32_t i = 0; i < coefNum; i++)
                {
                    scaling_list_delta_coef = benz_itu_h26xbs_read_se(bitstream); // [-128, 127]
                    nextCoef = (nextCoef + scaling_list_delta_coef + 256) % 256;
                    dstList[i] = nextCoef;
                }

                benz_itu_h26xbs_fill64b(bitstream);
            }
        }
    }

    benz_itu_h26xbs_fill64b(bitstream);

#ifdef MIN
#undef MIN
#endif // MIN
}

uint8_t get_sps_seq_parameter_set_id(const void* nalu, size_t nalubytelen)
{
    BENZ_ITU_H26XBS bitstream = {0};
    benz_itu_h26xbs_init(&bitstream, nalu, nalubytelen);

    benz_itu_h26xbs_skip_xn(&bitstream, 4); // sps_video_parameter_set_id
    uint8_t sps_max_sub_layers_minus1 = benz_itu_h26xbs_read_un(&bitstream, 3);
    benz_itu_h26xbs_skip_xn(&bitstream, 1); // sps_temporal_id_nesting_flag

    benz_itu_h26xbs_fill64b(&bitstream);
    skip_profile_tier_level(&bitstream, sps_max_sub_layers_minus1); // profile_tier_level( 1, sps_max_sub_layers_minus1 )

    return benz_itu_h26xbs_read_ue(&bitstream); // sps_seq_parameter_set_id [0, 15]
}

void decode_sps(CUVIDPICPARAMS* picParams, BENZ_ITU_H26XBS* bitstream){
    CUVIDHEVCPICPARAMS* hevcPP = (CUVIDHEVCPICPARAMS*)&picParams->CodecSpecific;

    benz_itu_h26xbs_skip_xn(bitstream, 4); // sps_video_parameter_set_id [0, 15]
    uint8_t sps_max_sub_layers_minus1 = benz_itu_h26xbs_read_un(bitstream, 3); // [0, 6]
    benz_itu_h26xbs_skip_xn(bitstream, 1); // sps_temporal_id_nesting_flag

//    ProfileTierLevel profile_tier_level;
//    profile_tier_level( 1, sps_max_sub_layers_minus1 )
    benz_itu_h26xbs_fill64b(bitstream);
    skip_profile_tier_level(bitstream, sps_max_sub_layers_minus1);

    benz_itu_h26xbs_skip_xe(bitstream); // sps_seq_parameter_set_id [0, 15]

    uint8_t chroma_format_idc = benz_itu_h26xbs_read_ue(bitstream); //[0, 3]
    int separate_colour_plane_flag = 0;
    if (chroma_format_idc == 3)
    {
        separate_colour_plane_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    }

//    uint8_t chroma_array_type;
    uint16_t pic_width_in_luma_samples = benz_itu_h26xbs_read_ue(bitstream);    // assumed to be within [0, 16382]
    uint16_t pic_height_in_luma_samples = benz_itu_h26xbs_read_ue(bitstream);   // assumed to be within [0, 16382]
    benz_itu_h26xbs_fill64b(bitstream);

    int conformance_window_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    if (conformance_window_flag)
    {
        benz_itu_h26xbs_skip_xe(bitstream);     // uint32_t conf_win_left_offset;   // assumed to be within [0, 16382]
        benz_itu_h26xbs_skip_xe(bitstream);     // uint32_t conf_win_right_offset;  // assumed to be within [0, 16382]
        benz_itu_h26xbs_skip_xe(bitstream);     // uint32_t conf_win_top_offset;    // assumed to be within [0, 16382]
        benz_itu_h26xbs_skip_xe(bitstream);     // uint32_t conf_win_bottom_offset; // assumed to be within [0, 16382]
        benz_itu_h26xbs_bigfill(bitstream);
    }

    uint8_t bit_depth_luma_minus8 = benz_itu_h26xbs_read_ue(bitstream); //[0, 8]
    uint8_t bit_depth_chroma_minus8 = benz_itu_h26xbs_read_ue(bitstream); //[0, 8]
    uint8_t log2_max_pic_order_cnt_lsb_minus4 = benz_itu_h26xbs_read_ue(bitstream); //[0, 12]

    int sps_sub_layer_ordering_info_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    for(int i = (sps_sub_layer_ordering_info_present_flag ? 0 : sps_max_sub_layers_minus1); i <= sps_max_sub_layers_minus1; i++)
    {
        benz_itu_h26xbs_skip_xe(bitstream); // sps_max_dec_pic_buffering_minus1[i] [0, 15]
        benz_itu_h26xbs_skip_xe(bitstream); // sps_max_num_reorder_pics[i] [0, 15]
        benz_itu_h26xbs_skip_xe(bitstream); // sps_max_latency_increase_plus1[i] [0, 0xFFFFFFFE]
        benz_itu_h26xbs_bigfill(bitstream);
    }

    uint8_t log2_min_luma_coding_block_size_minus3 = benz_itu_h26xbs_read_ue(bitstream); // estimation [1, 3] or [0, 8]
    uint8_t log2_diff_max_min_luma_coding_block_size = benz_itu_h26xbs_read_ue(bitstream); // estimation [0, 2] or [0, 8]
    uint8_t log2_min_transform_block_size_minus2 = benz_itu_h26xbs_read_ue(bitstream); // estimation [0, 8]
    uint8_t log2_diff_max_min_transform_block_size = benz_itu_h26xbs_read_ue(bitstream); // estimation [0, 8]
    uint8_t max_transform_hierarchy_depth_inter = benz_itu_h26xbs_read_ue(bitstream); // estimation [0, 8]
    uint8_t max_transform_hierarchy_depth_intra = benz_itu_h26xbs_read_ue(bitstream); // estimation [0, 8]
    benz_itu_h26xbs_fill64b(bitstream);

    uint8_t scaling_list_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    int sps_scaling_list_data_present_flag = 0;
    if (scaling_list_enabled_flag)
    {
        sps_scaling_list_data_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
        if (sps_scaling_list_data_present_flag)
        {
//            ScalingList scaling_list;
//            scaling_list_data( )
            benz_itu_h26xbs_fill64b(bitstream);
            decode_scaling_list_data(hevcPP, bitstream);
        }
    }

    int amp_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    int sample_adaptive_offset_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);

    int pcm_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    uint8_t pcm_sample_bit_depth_luma_minus1 = 0;
    uint8_t pcm_sample_bit_depth_chroma_minus1 = 0;
    uint8_t log2_min_pcm_luma_coding_block_size_minus3 = 0; // estimation [0, 8]
    uint8_t log2_diff_max_min_pcm_luma_coding_block_size = 0; // estimation [0, 8]
    int pcm_loop_filter_disabled_flag = 0;
    if (pcm_enabled_flag)
    {
        pcm_sample_bit_depth_luma_minus1 = benz_itu_h26xbs_read_un(bitstream, 4);
        pcm_sample_bit_depth_chroma_minus1 = benz_itu_h26xbs_read_un(bitstream, 4);
        log2_min_pcm_luma_coding_block_size_minus3 = benz_itu_h26xbs_read_ue(bitstream); // estimation [0,8]
        log2_diff_max_min_pcm_luma_coding_block_size = benz_itu_h26xbs_read_ue(bitstream); // estimation [0,8]
        pcm_loop_filter_disabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    }

    uint8_t num_short_term_ref_pic_sets = benz_itu_h26xbs_read_ue(bitstream); //[0, 64]
    // num_short_term_ref_pic_sets will always == 0 in our cases
//    for(int i = 0; i < num_short_term_ref_pic_sets; i++)
//    {
//        ShortTermRefPicSet short_term_ref_pic_set[64];
//        st_ref_pic_set( i )
//    }

    int long_term_ref_pics_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    // long_term_ref_pics_present_flag will always == 0 in our cases
//    if (long_term_ref_pics_present_flag)
//    {
//        uint8_t num_long_term_ref_pics_sps; //[0,32]
//        for (i = 0; i < num_long_term_ref_pics_sps; i++)
//        {
//            uint16_t lt_ref_pic_poc_lsb_sps[32];
//            bool used_by_curr_pic_lt_sps_flag[32];
//        }
//    }

    int temporal_mvp_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    int strong_intra_smoothing_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);

    // ... unnecessary vui_parameters and extension elements

    hevcPP->pic_width_in_luma_samples = pic_width_in_luma_samples;
    hevcPP->pic_height_in_luma_samples = pic_height_in_luma_samples;
    hevcPP->log2_min_luma_coding_block_size_minus3 = log2_min_luma_coding_block_size_minus3;
    hevcPP->log2_diff_max_min_luma_coding_block_size = log2_diff_max_min_luma_coding_block_size;
    hevcPP->log2_min_transform_block_size_minus2 = log2_min_transform_block_size_minus2;
    hevcPP->log2_diff_max_min_transform_block_size = log2_diff_max_min_transform_block_size;
    hevcPP->pcm_enabled_flag = pcm_enabled_flag;
    hevcPP->log2_min_pcm_luma_coding_block_size_minus3 = log2_min_pcm_luma_coding_block_size_minus3;
    hevcPP->log2_diff_max_min_pcm_luma_coding_block_size = log2_diff_max_min_pcm_luma_coding_block_size;
    hevcPP->pcm_sample_bit_depth_luma_minus1 = pcm_sample_bit_depth_luma_minus1;

    hevcPP->pcm_sample_bit_depth_chroma_minus1 = pcm_sample_bit_depth_chroma_minus1;
    hevcPP->pcm_loop_filter_disabled_flag = pcm_loop_filter_disabled_flag;
    hevcPP->strong_intra_smoothing_enabled_flag = strong_intra_smoothing_enabled_flag;
    hevcPP->max_transform_hierarchy_depth_intra = max_transform_hierarchy_depth_intra;
    hevcPP->max_transform_hierarchy_depth_inter = max_transform_hierarchy_depth_inter;
    hevcPP->amp_enabled_flag = amp_enabled_flag;
    hevcPP->separate_colour_plane_flag = separate_colour_plane_flag;
    hevcPP->log2_max_pic_order_cnt_lsb_minus4 = log2_max_pic_order_cnt_lsb_minus4;

    hevcPP->num_short_term_ref_pic_sets = num_short_term_ref_pic_sets;
    hevcPP->long_term_ref_pics_present_flag = long_term_ref_pics_present_flag;
    hevcPP->num_long_term_ref_pics_sps = 0;
    hevcPP->sps_temporal_mvp_enabled_flag = temporal_mvp_enabled_flag;
    hevcPP->sample_adaptive_offset_enabled_flag = sample_adaptive_offset_enabled_flag;
    hevcPP->scaling_list_enable_flag = scaling_list_enabled_flag;

    hevcPP->bit_depth_luma_minus8 = bit_depth_luma_minus8;
    hevcPP->bit_depth_chroma_minus8 = bit_depth_chroma_minus8;

	// Instead of following the computation given by 7-13, we assume nvdecode is expecting a ctbSizeY of 16
    uint32_t ctbSizeY = 16;
    // 7-15 (hacked to follow expectation)
    picParams->PicWidthInMbs = hevcPP->pic_width_in_luma_samples / ctbSizeY + 0.5;
    // 7-17 (hacked to follow expectation)
    picParams->FrameHeightInMbs = hevcPP->pic_height_in_luma_samples / ctbSizeY + 0.5;
}

void decode_pps(CUVIDPICPARAMS* picParams, BENZ_ITU_H26XBS* bitstream){
    CUVIDHEVCPICPARAMS* hevcPP = (CUVIDHEVCPICPARAMS*)&picParams->CodecSpecific;

    uint8_t pps_pic_parameter_set_id = benz_itu_h26xbs_read_ue(bitstream); //[0, 63]
    uint8_t pps_seq_parameter_set_id = benz_itu_h26xbs_read_ue(bitstream); //[0, 15]

    int dependent_slice_segments_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    int output_flag_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    uint8_t num_extra_slice_header_bits = benz_itu_h26xbs_read_un(bitstream, 3);
    int sign_data_hiding_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    int cabac_init_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    uint8_t num_ref_idx_l0_default_active_minus1 = benz_itu_h26xbs_read_ue(bitstream); //[0, 14]
    uint8_t num_ref_idx_l1_default_active_minus1 = benz_itu_h26xbs_read_ue(bitstream); //[0, 14]
    int8_t init_qp_minus26 = benz_itu_h26xbs_read_se(bitstream); // estimation [-26, 25]
    int constrained_intra_pred_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    int transform_skip_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);

    int cu_qp_delta_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    uint8_t diff_cu_qp_delta_depth = 0;
    if (cu_qp_delta_enabled_flag)
    {
        diff_cu_qp_delta_depth = benz_itu_h26xbs_read_ue(bitstream);    // estimation [0, 2] or [0, 8]
    }

    int8_t pps_cb_qp_offset = benz_itu_h26xbs_read_se(bitstream); //[-12, 12]
    int8_t pps_cr_qp_offset = benz_itu_h26xbs_read_se(bitstream); //[-12, 12]
    int slice_chroma_qp_offsets_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    int weighted_pred_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    int weighted_bipred_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    int transquant_bypass_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    int tiles_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    int entropy_coding_sync_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    benz_itu_h26xbs_bigfill(bitstream);

    uint8_t num_tile_columns_minus1 = 0;    // [0, 18] or assumed to be within [0, 1021]
    uint8_t num_tile_rows_minus1 = 0;       // [0, 20] or assumed to be within [0, 1021]
    int uniform_spacing_flag = 1;
    int loop_filter_across_tiles_enabled_flag = 1;
    if (tiles_enabled_flag)
    {
        // 7-10
        uint32_t minCbLog2SizeY = hevcPP->log2_min_luma_coding_block_size_minus3 + 3;
        // 7-11
        uint32_t ctbLog2SizeY = minCbLog2SizeY + hevcPP->log2_diff_max_min_luma_coding_block_size;
        // 7-13
        uint32_t ctbSizeY = 1 << ctbLog2SizeY;
        // 7-15
        uint32_t picWidthInCtbsY = (uint32_t)((double)hevcPP->pic_width_in_luma_samples / (double)ctbSizeY + 0.5);
        // 7-17
        uint32_t picHeightInCtbsY = (uint32_t)((double)hevcPP->pic_height_in_luma_samples / (double)ctbSizeY + 0.5);

        num_tile_columns_minus1 = benz_itu_h26xbs_read_ue(bitstream);   // [0, 20] or assumed to be within [0, 1021]
        num_tile_rows_minus1 = benz_itu_h26xbs_read_ue(bitstream);      // [0, 20] or assumed to be within [0, 1021]

        uniform_spacing_flag = benz_itu_h26xbs_read_un(bitstream, 1);
        if (uniform_spacing_flag)
        {
            uint8_t numCol = num_tile_columns_minus1 + 1;
            uint8_t numRow = num_tile_rows_minus1 + 1;
            for (int i = 0; i < numCol; i++)
            {
                hevcPP->column_width_minus1[i] = (i + 1) * picWidthInCtbsY / numCol - i * picWidthInCtbsY / numCol - 1;
            }
            for (int i = 0; i < numRow; i++)
            {
                hevcPP->row_height_minus1[i] = (i + 1) * picHeightInCtbsY / numRow - i * picHeightInCtbsY / numRow - 1;
            }
        }
//        else
//        {
//            for(int i = 0; i <= num_tile_columns_minus1; i++)
//            {
//                hevcPP->column_width_minus1[i] = benz_itu_h26xbs_read_ue(bitstream);    // assumed to be within [15, 1021]
//                benz_itu_h26xbs_fill64b(bitstream);
//            }
//            for(int i = 0; i <= num_tile_rows_minus1; i++)
//            {
//                hevcPP->row_height_minus1[i] = benz_itu_h26xbs_read_ue(bitstream);      // assumed to be within [15, 1021]
//                benz_itu_h26xbs_fill64b(bitstream);
//            }
//        }

        loop_filter_across_tiles_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    }

    int pps_loop_filter_across_slices_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);

    int deblocking_filter_control_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    int deblocking_filter_override_enabled_flag = 0;
    int pps_deblocking_filter_disabled_flag = 0;
    int8_t pps_beta_offset_div2 = 0;    //[-6, 6]
    int8_t pps_tc_offset_div2 = 0;      //[-6, 6]
    if (deblocking_filter_control_present_flag)
    {
        deblocking_filter_override_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);

        pps_deblocking_filter_disabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);
        if (!pps_deblocking_filter_disabled_flag)
        {
            pps_beta_offset_div2 = benz_itu_h26xbs_read_se(bitstream); //[-6, 6]
            pps_tc_offset_div2 = benz_itu_h26xbs_read_se(bitstream); //[-6, 6]
        }
    }

    int pps_scaling_list_data_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    benz_itu_h26xbs_fill64b(bitstream);
    if (pps_scaling_list_data_present_flag)
    {
//        ScalingList scaling_list;
//        scaling_list_data( )
        decode_scaling_list_data(hevcPP, bitstream);
    }

//    // When scaling_list_enabled_flag is equal to 1,
//    // sps_scaling_list_data_present_flag
//    // is equal to 0 and pps_scaling_list_data_present_flag is equal to 0, the
//    // default
//    // scaling list data are used to derive the array ScalingFactor as described
//    // in the
//    // scaling list data semantics as specified in clause 7.4.5.
//    if (sps->scaling_list_enabled_flag && !sps->sps_scaling_list_data_present_flag && !pps->pps_scaling_list_data_present_flag)
//    {
//        uint8_t* dstList = NULL;
//        uint8_t* dstDcList = NULL;
//        for (uint32_t sizeId = 0; sizeId < 4; sizeId++)
//        {
//            // as spec "7.3.4 Scaling list data syntax" and Table 7-4,
//            // Since CUVIDHEVCPICPARAMS.ScalingList32x32[2], if sizeId == 3, we make matrixIdx range
//            // within [0, 1] instead of [0, 5], thus making matrixId = matrixIdx * 3
//            uint32_t maxMatrixIdx = (sizeId == 3) ? 2 : 6;
//            for (uint32_t matrixIdx = 0; matrixIdx < maxMatrixIdx; matrixIdx++)
//            {
//                // Table 7-3
//                switch (sizeId)
//                {
//                case 0: // 4x4
//                    dstList = hevcPP->ScalingList4x4[matrixIdx];
//                    break;
//                case 1: // 8x8
//                    dstList = hevcPP->ScalingList8x8[matrixIdx];
//                    break;
//                case 2: // 16x16
//                    dstList = hevcPP->ScalingList16x16[matrixIdx];
//                    dstDcList = hevcPP->ScalingListDCCoeff16x16;
//                    break;
//                case 3: // 32x32
//                    dstList = hevcPP->ScalingList32x32[matrixIdx];
//                    dstDcList = hevcPP->ScalingListDCCoeff32x32;
//                }
//                set_default_scaling_list(dstList, dstDcList, sizeId, matrixIdx);
//            }
//        }
//    }

    int lists_modification_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    uint8_t log2_parallel_merge_level_minus2 = benz_itu_h26xbs_read_ue(bitstream);  // estimate [0, 4]
    int slice_segment_header_extension_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);

    // ... unnecessary extension elements

    hevcPP->dependent_slice_segments_enabled_flag = dependent_slice_segments_enabled_flag;
    hevcPP->slice_segment_header_extension_present_flag = slice_segment_header_extension_present_flag;
    hevcPP->sign_data_hiding_enabled_flag = sign_data_hiding_enabled_flag;
    hevcPP->cu_qp_delta_enabled_flag = cu_qp_delta_enabled_flag;
    hevcPP->diff_cu_qp_delta_depth = diff_cu_qp_delta_depth;
    hevcPP->init_qp_minus26 = init_qp_minus26;
    hevcPP->pps_cb_qp_offset = pps_cb_qp_offset;
    hevcPP->pps_cr_qp_offset = pps_cr_qp_offset;

    hevcPP->constrained_intra_pred_flag = constrained_intra_pred_flag;
    hevcPP->weighted_pred_flag = weighted_pred_flag;
    hevcPP->weighted_bipred_flag = weighted_bipred_flag;
    hevcPP->transform_skip_enabled_flag = transform_skip_enabled_flag;
    hevcPP->transquant_bypass_enabled_flag = transquant_bypass_enabled_flag;
    hevcPP->entropy_coding_sync_enabled_flag = entropy_coding_sync_enabled_flag;
    hevcPP->log2_parallel_merge_level_minus2 = log2_parallel_merge_level_minus2;
    hevcPP->num_extra_slice_header_bits = num_extra_slice_header_bits;

    hevcPP->loop_filter_across_tiles_enabled_flag = loop_filter_across_tiles_enabled_flag;
    hevcPP->loop_filter_across_slices_enabled_flag = pps_loop_filter_across_slices_enabled_flag;
    hevcPP->output_flag_present_flag = output_flag_present_flag;
    hevcPP->num_ref_idx_l0_default_active_minus1 = num_ref_idx_l0_default_active_minus1;
    hevcPP->num_ref_idx_l1_default_active_minus1 = num_ref_idx_l1_default_active_minus1;
    hevcPP->lists_modification_present_flag = lists_modification_present_flag;
    hevcPP->cabac_init_present_flag = cabac_init_present_flag;
    hevcPP->pps_slice_chroma_qp_offsets_present_flag = slice_chroma_qp_offsets_present_flag;

    hevcPP->deblocking_filter_override_enabled_flag = deblocking_filter_override_enabled_flag;
    hevcPP->pps_deblocking_filter_disabled_flag = pps_deblocking_filter_disabled_flag;
    hevcPP->pps_beta_offset_div2 = pps_beta_offset_div2;
    hevcPP->pps_tc_offset_div2 = pps_tc_offset_div2;
    hevcPP->tiles_enabled_flag = tiles_enabled_flag;
    hevcPP->uniform_spacing_flag = uniform_spacing_flag;
    hevcPP->num_tile_columns_minus1 = num_tile_columns_minus1;
    hevcPP->num_tile_rows_minus1 = num_tile_rows_minus1;
}

/**
 * @brief Perform the core operation of the feeder thread.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The context 
 * @return 0
 */

BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdCore          (NVDECODE_CTX* ctx){
#ifndef MAXSPSCOUNT
#define MAXSPSCOUNT 64
#endif // MAXSPSCOUNT
	static uint32_t ONE = 1;

	NVDECODE_RQ*    rq;
	CUVIDPICPARAMS* pP;
	CUresult        ret;
	
	
	nvdecodeFeederThrdGetCurrRq(ctx, &rq);
	pP = rq->picParams;
    CUVIDHEVCPICPARAMS* hevcPP = (CUVIDHEVCPICPARAMS*)&pP->CodecSpecific;
	
	const uint8_t* record = &rq->hvcCData[0];

    uint32_t configurationVersion = (uint32_t)record[0];                                           // 0
    uint32_t general_profile_space = (uint32_t)(record[1] >> 6);                                   // 1 : 11000000
    uint32_t general_tier_flag = (uint32_t)(record[1] >> 5 & 0x01);                                // 1 : 00100000
    uint32_t general_profile_idc = (uint32_t)(record[1] & 0x1f);                                   // 1 : 00011111
    uint32_t general_profile_compatibility_flags = benz_iso_bmff_as_u32(record + 2);               // 2 (2-5)
    uint64_t general_constraint_indicator_flags = benz_iso_bmff_as_u64(record + 6) >> 16;          // 6 (6-11)
    uint32_t general_level_idc = (uint32_t)record[12];                                             // 12
    // bits(4) reserved = ‘1111’b;
    uint32_t min_spatial_segmentation_idc = (uint32_t)benz_iso_bmff_as_u16(record + 13) & 0x0fff;  // 13 (13-14) : 00001111 11111111 ...
    // bits(6) reserved = ‘111111’b;
    uint32_t parallelismType = (uint32_t)(record[15] & 0x03);                                      // 15 : 00000011
    // bits(6) reserved = ‘111111’b;
    uint32_t chromaFormat = (uint32_t)(record[16] & 0x03);                                         // 16 : 00000011
    // bits(5) reserved = ‘11111’b;
    uint32_t bitDepthLumaMinus8 = (uint32_t)(record[17] & 0x07);                                   // 17 : 00000111
    // bits(5) reserved = ‘11111’b;
    uint32_t bitDepthChromaMinus8 = (uint32_t)(record[18] & 0x07);                                 // 18 : 00000011
    uint32_t avgFrameRate = (uint32_t)benz_iso_bmff_as_u16(record + 19);                           // 19 (19-20)
    uint32_t constantFrameRate = (uint32_t)(record[21] >> 6);                                      // 21 : 11000000
    uint32_t numTemporalLayers = (uint32_t)(record[21] >> 3 & 0x07);                               // 21 : 00111000
    uint32_t temporalIdNested = (uint32_t)(record[21] >> 2 & 0x01);                                // 21 : 00000100
    uint32_t lengthSizeMinusOne = (uint32_t)(record[21] & 0x03);                                   // 21 : 00000011
    uint32_t numOfArrays = (uint32_t)record[22];                                                   // 22
    record += 23;

    // Find PPS, SPSs and get SPS id
    uint8_t pps_sps_id = 0;
    const uint8_t* pps_location = 0;
    size_t pps_length = 0;
    const uint8_t* sps_locations[MAXSPSCOUNT] = {0};
    size_t sps_lengths[MAXSPSCOUNT] = {0};

    for (int i=0; i < numOfArrays; i++)
    {
        // In our case, array_completeness will always == 1
//        uint32_t array_completeness = (uint32_t)(record[0] >> 7);                                  // 0 : 10000000
        // bits(1) reserved = 0;
        uint32_t NAL_unit_type = (uint32_t)(record[0] & 0x3f);                                     // 0 : 00111111
        uint32_t numNalus = (uint32_t)benz_iso_bmff_as_u16(record + 1);                            // 1 (1-2)
        record += 3;

        uint8_t sps_id = 0;
        for (int j=0; j < numNalus; j++)
        {
            uint32_t nalUnitLength = (uint32_t)benz_iso_bmff_as_u16(record);                       // 0 (0-1)
            record += 2;

            // bits(8*nalUnitLength) nalUnit;
            switch(NAL_unit_type){
            case 32: break; //VPS_NUT
            case 33: //SPS_NUT
                //Skip the 2 header bytes
                sps_id = get_sps_seq_parameter_set_id(record + 2, nalUnitLength - 2);
                sps_locations[sps_id] = record + 2;
                sps_lengths[sps_id] = nalUnitLength - 2;
                break;
            case 34: //PPS_NUT
                pps_sps_id = benz_itu_h265_nalu_pps_get_sps_id(record);
                //Skip the 2 header bytes
                pps_location = record + 2;
                pps_length = nalUnitLength - 2;
            }
            record += nalUnitLength;
        }
    }

    BENZ_ITU_H26XBS bitstream = {0};

    // Decode SPS
    benz_itu_h26xbs_init(&bitstream, sps_locations[pps_sps_id], sps_lengths[pps_sps_id]);
    decode_sps(pP, &bitstream);

    // Decode PPS
    benz_itu_h26xbs_init(&bitstream, pps_location, pps_length);
    decode_pps(pP, &bitstream);

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
	
    benz_putbe32(rq->data, 0, 0x00000001); // 0x00 + annexb begin flag

    hevcPP->IrapPicFlag   = 1;
    hevcPP->IdrPicFlag    = 1;

	pP->CurrPicIdx        = ctx->feeder.cnt % ctx->decoderInfo.ulNumDecodeSurfaces;
    pP->field_pic_flag    = 0;
    pP->bottom_field_flag = 0;
    pP->second_field      = 0;
    // Bitstream data
    pP->nNumSlices        = 1;
	pP->pSliceDataOffsets = &ONE;   // offset to annexb begin flag
	
    pP->ref_pic_flag      = 1;
    pP->intra_pic_flag    = 1;
	
	/**
	 * Drop mutex and possibly block attempting to decode image, then
	 * reacquire mutex.
	 */
	
	pthread_mutex_unlock(&ctx->lock);
	ret = cuvidDecodePicture(ctx->decoder, pP);
	pthread_mutex_lock(&ctx->lock);
	
	/* Release data. */
//	free(rq->data);
//	free(rq->hvcCData);
	rq->sample   = NULL;
	rq->data     = NULL;
	rq->hvcCData = NULL;
	if(ret != CUDA_SUCCESS){
		ctx->feeder.err = ret;
		nvdecodeFeederThrdSetStatus(ctx, THRD_EXITING);
		return 0;
	}
	
	/* Bump counters and broadcast signal. */
	ctx->feeder.cnt++;
	pthread_cond_broadcast(&ctx->worker.cond);
	return 0;
#ifdef MAXSPSCOUNT
#undef MAXSPSCOUNT
#endif // MAXSPSCOUNT
}

/**
 * @brief Change feeder thread's status.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The context whose feeder thread's status is being changed.
 * @return 0
 */

BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdSetStatus    (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status){
	ctx->feeder.status = status;
	switch(status){
		case THRD_NOT_RUNNING:
			if(nvdecodeHelpersAllStatusIs(ctx, THRD_NOT_RUNNING)){
				if(ctx->master.status == CTX_HELPERS_EXITING){
					nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_JOINING);
				}else{
					nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_NOT_RUNNING);
				}
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

BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdGetCurrRq     (NVDECODE_CTX* ctx, NVDECODE_RQ** rqOut){
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

BENZINA_PLUGIN_STATIC void*       nvdecodeWorkerThrdMain          (NVDECODE_CTX* ctx){
	pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
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

BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdInit          (NVDECODE_CTX* ctx){
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

BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdAwaitAll      (NVDECODE_CTX* ctx){
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

BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdContinue      (NVDECODE_CTX* ctx){
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
	pthread_mutex_unlock       (&ctx->lock);
	cudaStreamSynchronize      (ctx->worker.cudaStream);
	cudaStreamDestroy          (ctx->worker.cudaStream);
	pthread_mutex_lock         (&ctx->lock);
	nvdecodeMaybeReapDecoder   (ctx);
	nvdecodeWorkerThrdSetStatus(ctx, THRD_NOT_RUNNING);
	return 0;
}

/**
 * @brief Does worker thread have work to do?
 * @param [in]  ctx  The context in question
 * @return !0 if thread has work to do; 0 otherwise.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdHasWork       (NVDECODE_CTX* ctx){
	return ctx->worker.cnt < ctx->feeder.cnt;
}

/**
 * @brief Worker Wait.
 * @param [in]   ctx  The context
 * @return 1
 */

BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdWait          (NVDECODE_CTX* ctx){
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

BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdCore          (NVDECODE_CTX* ctx){
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
	// In case feeder is waiting after worker, let him know a new surface is
	// available
	pthread_cond_broadcast(&ctx->feeder.cond);
	return 0;
}

/**
 * @brief Post-processing Callback
 * @param [in]   stream The stream onto which this callback had been scheduled.
 * @param [in]   status The error status of this device or stream.
 * @param [in]   ctx    The context on which this callback is being executed.
 * @return 
 */

BENZINA_PLUGIN_STATIC void        nvdecodeWorkerThrdCallback      (cudaStream_t  stream,
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

BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdSetStatus     (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status){
	ctx->worker.status = status;
	switch(status){
		case THRD_NOT_RUNNING:
			if(nvdecodeHelpersAllStatusIs(ctx, THRD_NOT_RUNNING)){
				if(ctx->master.status == CTX_HELPERS_EXITING){
					nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_JOINING);
				}else{
					nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_NOT_RUNNING);
				}
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

BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdGetCurrRq     (NVDECODE_CTX* ctx, NVDECODE_RQ** rqOut){
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

BENZINA_PLUGIN_STATIC int         nvdecodeSetDevice               (NVDECODE_CTX* ctx, const char*   deviceId){
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

BENZINA_PLUGIN_STATIC int         nvdecodeWaitBatchLocked         (NVDECODE_CTX*    ctx,
                                                                   const void**     token,
                                                                   int              block,
                                                                   double           timeout){
	NVDECODE_BATCH* batch;
	TIMESPEC        deadline, now;
	uint64_t        lifecycle;
	int             ret = 0;
	
	
	*token = NULL;
	if(timeout > 0){
		nvdecodeTimeMonotonic(&now);
		nvdecodeDoubleToTime (&deadline, timeout);
		nvdecodeTimeAdd      (&deadline, &now, &deadline);
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
			*token = nvdecodeReturnAndClear(&batch->token);
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


/* Plugin Interface Function Definitions */

/**
 * @brief Allocate iterator context from dataset.
 * @param [out] ctxOut   Output pointer for the context handle.
 * @param [in]  dataset  The dataset over which this iterator will iterate.
 *                       Must be non-NULL and compatible.
 * @return A pointer to the context, if successful; NULL otherwise.
 */

BENZINA_PLUGIN_HIDDEN int         nvdecodeAlloc                   (void** ctxOut, const BENZINA_DATASET* dataset){
	NVDECODE_CTX* ctx = NULL;
	const char*   datasetFile = NULL;
	size_t        datasetLen;
	
	
	/**
	 * The ctxOut and dataset parameters cannot be NULL.
	 */
	
	if(!ctxOut){
		return -1;
	}
	*ctxOut = NULL;
	if(!dataset                                            ||
	   benzinaDatasetGetFile  (dataset, &datasetFile) != 0 ||
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
	ctx->datasetFile       =  datasetFile;
	ctx->datasetLen        =  datasetLen;
	ctx->datasetFd         = -1;
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

BENZINA_PLUGIN_STATIC int         nvdecodeAllocDataOpen           (NVDECODE_CTX* ctx){
	struct stat s0;
	
	ctx->datasetFd = open(ctx->datasetFile, O_RDONLY|O_CLOEXEC);
	if(ctx->datasetFd             < 0 ||
	   fstat(ctx->datasetFd, &s0) < 0){
		return nvdecodeAllocCleanup(ctx, -1);
	}

	ctx->decoderInfo.ulWidth = 512;
    ctx->decoderInfo.ulHeight = 512;
    ctx->decoderInfo.ulNumDecodeSurfaces = 4;
//    ctx->decoderInfo.CodecType = cudaVideoCodec_H264;
    ctx->decoderInfo.CodecType = cudaVideoCodec_HEVC;
    ctx->decoderInfo.ChromaFormat = cudaVideoChromaFormat_420;
//    ctx->decoderInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    ctx->decoderInfo.bitDepthMinus8 = 0;
    ctx->decoderInfo.ulIntraDecodeOnly = 1;
    ctx->decoderInfo.ulMaxWidth = ctx->decoderInfo.ulWidth;
    ctx->decoderInfo.ulMaxHeight = ctx->decoderInfo.ulHeight;
    ctx->decoderInfo.display_area.left = 0;
    ctx->decoderInfo.display_area.top = 0;
    ctx->decoderInfo.display_area.right = ctx->decoderInfo.ulWidth;
    ctx->decoderInfo.display_area.bottom = ctx->decoderInfo.ulHeight;
    ctx->decoderInfo.OutputFormat = ctx->decoderInfo.bitDepthMinus8 > 0 ?
                                    cudaVideoSurfaceFormat_P016 :
                                    cudaVideoSurfaceFormat_NV12;
    ctx->decoderInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    ctx->decoderInfo.ulTargetWidth = ctx->decoderInfo.ulWidth;
    ctx->decoderInfo.ulTargetHeight = ctx->decoderInfo.ulHeight;
    ctx->decoderInfo.ulNumOutputSurfaces = 4;
//    u->decoderInfo.vidLock = NULL;
    ctx->decoderInfo.target_rect.left = 0;
    ctx->decoderInfo.target_rect.top = 0;
    ctx->decoderInfo.target_rect.right = ctx->decoderInfo.ulTargetWidth;
    ctx->decoderInfo.target_rect.bottom = ctx->decoderInfo.ulTargetWidth;

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

BENZINA_PLUGIN_STATIC int         nvdecodeAllocThreading          (NVDECODE_CTX* ctx){
	pthread_condattr_t condAttr;
	
	if(pthread_condattr_init    (&condAttr)                    != 0){goto fail_attr;}
	if(pthread_condattr_setclock(&condAttr, CLOCK_MONOTONIC)   != 0){goto fail_clock;}
	if(pthread_mutex_init       (&ctx->lock,                0) != 0){goto fail_lock;}
	if(pthread_cond_init        (&ctx->master.cond, &condAttr) != 0){goto fail_master;}
	if(pthread_cond_init        (&ctx->reader.cond, &condAttr) != 0){goto fail_reader;}
	if(pthread_cond_init        (&ctx->feeder.cond, &condAttr) != 0){goto fail_feeder;}
	if(pthread_cond_init        (&ctx->worker.cond, &condAttr) != 0){goto fail_worker;}
	
	pthread_condattr_destroy(&condAttr);
	
	return nvdecodeAllocCleanup(ctx,  0);
	
	
	             pthread_cond_destroy    (&ctx->worker.cond);
	fail_worker: pthread_cond_destroy    (&ctx->feeder.cond);
	fail_feeder: pthread_cond_destroy    (&ctx->reader.cond);
	fail_reader: pthread_cond_destroy    (&ctx->master.cond);
	fail_master: pthread_mutex_destroy   (&ctx->lock);
	fail_lock:   
	fail_clock:  pthread_condattr_destroy(&condAttr);
	fail_attr:
	
	return nvdecodeAllocCleanup(ctx, -1);
}

/**
 * @brief Cleanup for context allocation.
 * 
 * @param [in]  ctx  The context being allocated.
 * @param [in]  ret  Return error code.
 * @return The value `ret`.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeAllocCleanup            (NVDECODE_CTX* ctx, int ret){
	if(ret == 0){
		return ret;
	}
	
	close(ctx->datasetFd);
	ctx->datasetFd = -1;
	
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

BENZINA_PLUGIN_HIDDEN int         nvdecodeInit                    (NVDECODE_CTX* ctx){
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

BENZINA_PLUGIN_HIDDEN int         nvdecodeRetain                  (NVDECODE_CTX* ctx){
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

BENZINA_PLUGIN_HIDDEN int         nvdecodeRelease                 (NVDECODE_CTX* ctx){
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
	
	close(ctx->datasetFd);
	
	free(ctx->picParams);
	free(ctx->request);
	free(ctx->batch);
	
	memset(ctx, 0, sizeof(*ctx));
	free(ctx);
	
	return 0;
}

/**
 * @brief Ensure that this iterator context's helper threads are running.
 * 
 * This is not actually a very useful function. Technically, even if it returns
 * success, by the time it returns the threads may have shut down again already.
 * 
 * @param [in]  ctx  The iterator context whose helper threads are to be spawned.
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int         nvdecodeStartHelpers            (NVDECODE_CTX* ctx){
	int ret;
	
	pthread_mutex_lock(&ctx->lock);
	ret = nvdecodeHelpersStart(ctx);
	pthread_mutex_unlock(&ctx->lock);
	
	return ret;
}

/**
 * @brief Ensure that this iterator context's helper threads are stopped.
 * 
 * @param [in]  ctx  The iterator context whose helper threads are to be stopped.
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int         nvdecodeStopHelpers             (NVDECODE_CTX* ctx){
	int ret;
	
	pthread_mutex_lock(&ctx->lock);
	ret = nvdecodeHelpersStop(ctx);
	pthread_mutex_unlock(&ctx->lock);
	
	return ret;
}

/**
 * @brief Begin defining a batch of samples.
 * 
 * @param [in]  ctx       The iterator context in which.
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int         nvdecodeDefineBatch             (NVDECODE_CTX* ctx){
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

BENZINA_PLUGIN_HIDDEN int         nvdecodeSubmitBatch             (NVDECODE_CTX* ctx, const void* token){
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

BENZINA_PLUGIN_HIDDEN int         nvdecodeWaitBatch               (NVDECODE_CTX* ctx,
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
 * @brief Peek at a token from the pipeline.
 * 
 * @param [in]  ctx      The iterator context in question.
 * @param [in]  i        The index of the token wanted.
 * @param [in]  clear    Whether to clear (!0) the token from the buffer or not (0).
 * @param [out] token    User data that was submitted at the corresponding
 *                       pushBatch().
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int         nvdecodePeekToken               (NVDECODE_CTX* ctx,
                                                                   uint64_t      i,
                                                                   int           clear,
                                                                   const void**  token){
	NVDECODE_BATCH* batch = NULL;
	int ret = -1;
	
	pthread_mutex_lock(&ctx->lock);
	if(i >= ctx->master.pull.token &&
	   i <  ctx->master.push.token){
		batch = &ctx->batch[i % ctx->multibuffering];
		if(clear){
			*token = nvdecodeReturnAndClear(&batch->token);
		}else{
			*token = batch->token;
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

BENZINA_PLUGIN_HIDDEN int         nvdecodeDefineSample            (NVDECODE_CTX* ctx, uint64_t i, void* dstPtr,
                                                                   void* sample,
                                                                   uint64_t* location, uint64_t* config_location){
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
		rq->batch              = batch;
		rq->datasetIndex       = i;
		rq->devPtr             = dstPtr;
		rq->sample             = sample;
		rq->location[0]        = location[0];
		rq->location[1]        = location[1];
		rq->config_location[0] = config_location[0];
		rq->config_location[1] = config_location[1];
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

BENZINA_PLUGIN_HIDDEN int         nvdecodeSubmitSample            (NVDECODE_CTX* ctx){
	int ret;
	pthread_mutex_lock(&ctx->lock);
	switch(ctx->master.status){
		case CTX_HELPERS_JOINING:
		case CTX_HELPERS_EXITING:
			ret = nvdecodeMasterThrdAwaitShutdown(ctx);
			if(ret != 0){
				pthread_mutex_unlock(&ctx->lock);
				return -2;
			}
		break;
		default: break;
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

BENZINA_PLUGIN_HIDDEN int         nvdecodeGetNumPushes            (NVDECODE_CTX* ctx, uint64_t* out){
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

BENZINA_PLUGIN_HIDDEN int         nvdecodeGetNumPulls             (NVDECODE_CTX* ctx, uint64_t* out){
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

BENZINA_PLUGIN_HIDDEN int         nvdecodeGetMultibuffering       (NVDECODE_CTX* ctx, uint64_t* out){
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

BENZINA_PLUGIN_HIDDEN int         nvdecodeSetBuffer               (NVDECODE_CTX* ctx,
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

BENZINA_PLUGIN_HIDDEN int         nvdecodeSetDefaultBias          (NVDECODE_CTX* ctx,
                                                                   float*        B){
	memcpy(ctx->defaults.B, B, sizeof(ctx->defaults.B));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int         nvdecodeSetDefaultScale         (NVDECODE_CTX* ctx,
                                                                   float*        S){
	memcpy(ctx->defaults.S, S, sizeof(ctx->defaults.S));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int         nvdecodeSetDefaultOOBColor      (NVDECODE_CTX* ctx,
                                                                   float*        OOB){
	memcpy(ctx->defaults.OOB, OOB, sizeof(ctx->defaults.OOB));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int         nvdecodeSelectDefaultColorMatrix(NVDECODE_CTX* ctx,
                                                                   uint32_t      colorMatrix){
	ctx->defaults.colorMatrix = colorMatrix;
	return 0;
}
BENZINA_PLUGIN_HIDDEN int         nvdecodeSetHomography           (NVDECODE_CTX* ctx,
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
BENZINA_PLUGIN_HIDDEN int         nvdecodeSetBias                 (NVDECODE_CTX* ctx,
                                                                   const float*  B){
	NVDECODE_RQ* rq = &ctx->request[ctx->master.push.sample % ctx->totalSlots];
	memcpy(rq->B, B?B:ctx->defaults.B, sizeof(rq->B));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int         nvdecodeSetScale                (NVDECODE_CTX* ctx,
                                                                   const float*  S){
	NVDECODE_RQ* rq = &ctx->request[ctx->master.push.sample % ctx->totalSlots];
	memcpy(rq->S, S?S:ctx->defaults.S, sizeof(rq->S));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int         nvdecodeSetOOBColor             (NVDECODE_CTX* ctx,
                                                                   const float*  OOB){
	NVDECODE_RQ* rq = &ctx->request[ctx->master.push.sample % ctx->totalSlots];
	memcpy(rq->OOB, OOB?OOB:ctx->defaults.OOB, sizeof(rq->OOB));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int         nvdecodeSelectColorMatrix       (NVDECODE_CTX* ctx,
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
	.startHelpers             = (void*)nvdecodeStartHelpers,
	.stopHelpers              = (void*)nvdecodeStopHelpers,
	.defineBatch              = (void*)nvdecodeDefineBatch,
	.submitBatch              = (void*)nvdecodeSubmitBatch,
	.waitBatch                = (void*)nvdecodeWaitBatch,
	.peekToken                = (void*)nvdecodePeekToken,
	
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

