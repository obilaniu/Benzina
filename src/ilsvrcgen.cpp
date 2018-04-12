/**
 * ILSVRC Generator.
 */

/* Includes */
#include <cuda.h>
#include <cuda_runtime.h>
#include <hdf5.h>
#include <math.h>
#include <nvcuvid.h>
#include <omp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#ifdef __cplusplus
}
#endif


/* Defines */
#ifdef __GNUC__
#define UNUSED __attribute__((unused))
#else
#define UNUSED
#endif

#define ILSVRC_EXIT_SUCCESS        0
#define ILSVRC_EXIT_FAIL_ARGS      1
#define ILSVRC_EXIT_FAIL_CUDA      2
#define ILSVRC_EXIT_FAIL_FFMPEG    3
#define ILSVRC_EXIT_FAIL_HDF5      4
#define ILSVRC_EXIT_FAIL_OPENMP    5



/* Data Structure Forward Declarations and Typedefs */
struct DSET_CTX;
struct SPLIT_ATTR_TYPE;
struct DSET_THRD_CTX;

typedef struct DSET_CTX          DSET_CTX;
typedef struct SPLIT_ATTR_TYPE   SPLIT_ATTR_TYPE;
typedef struct DSET_THRD_CTX     DSET_THRD_CTX;



/* Data Structure Definitions */

/**
 * Per-thread context.
 */

struct DSET_THRD_CTX{
	DSET_CTX*             dsetCtx;
	int                   thrdNum;
	
	hsize_t               imgNum;
	hid_t                 imgFSpace, imgMSpace;
	hvl_t                 imgData[1];
	void*                 imgJPEG;
	hsize_t               imgJPEGSize;
	
	AVCodecContext*       jpegDecCtx;
	AVCodecContext*       h264EncCtx;
	AVDictionary*         jpegOpts;
	AVDictionary*         h264Opts;
	
	CUvideoparser         parser;
	CUVIDPARSERPARAMS     parserParams;
	
	int                   imgCnt;
};

/**
 * Split atrribute datatype, as present in original dataset.
 */

struct SPLIT_ATTR_TYPE{
	char     split[5];
	char     source[14];
	uint64_t start;
	uint64_t stop;
	void*    indices;
	char     available;
	char     comment;
};

/**
 * The dataset is structured as follows:
 * 
 * /
 *   data/
 *     x                        u1["length of concatenated h264"],  align=16MB
 *     y                        u8[1461406][5],                     align=16MB
 *        # [*][0]: Byte offset start, inclusive
 *        # [*][1]: Byte offset end,   exclusive
 *        # [*][2]: Class label (0-999)
 *        # [*][3]: Width  in pixels
 *        # [*][4]: Height in pixels
 *     splits                   u8[3][2],                           align=8
 *        # [0][0]: Training   Start Index (      0)
 *        # [0][1]: Training   End   Index (1261405)
 *        # [1][0]: Validation Start Index (1261406)
 *        # [1][1]: Validation End   Index (1311405)
 *        # [2][0]: Test       Start Index (1311406)
 *        # [2][1]: Test       End   Index (1461405)
 */

struct DSET_CTX{
	int              argc;
	char**           argv;
	struct{
		int              cudaDev;
		const char*      srcPath;
		const char*      dstPath;
	} args;
	
	AVCodec*         jpegDecoder;
	AVCodec*         h264Encoder;
	
	hid_t            srcFile;
	hid_t            srcFileSplit;
	hid_t            srcFileSplitSpace;
	hid_t            srcFileSplitType;
	hid_t            srcFileSplitNativeType;
	SPLIT_ATTR_TYPE* srcFileSplitNativePtr;
	hid_t            srcFileEncoded;
	hid_t            srcFileTargets;
	hid_t            srcFileEncodedSpace;
	hid_t            srcFileEncodedType;
	int              srcFileEncodedNDims;
	hsize_t          srcFileEncodedDims[1];
	hsize_t          nTotal, nTrain, nVal, nTest;
	
	int              numThrds;
	DSET_THRD_CTX*   thrds;
	
	omp_lock_t       exitLock;
	int              exiting;
	int              exitCode;
};



/* Static Function Prototypes */
static int    ilsvrcInit                    (DSET_CTX* ctx, int argc, char** argv);
static int    ilsvrcParseArg                (DSET_CTX* ctx, int argc, char** argv);
static int    ilsvrcCheckArgs               (DSET_CTX* ctx);
static int    ilsvrcInitCUDA                (DSET_CTX* ctx);
static int    ilsvrcInitFFmpeg              (DSET_CTX* ctx);
static int    ilsvrcInitHDF5                (DSET_CTX* ctx);
static int    ilsvrcInitOpenMP              (DSET_CTX* ctx);
static int    ilsvrcRun                     (DSET_CTX* ctx);
static int    ilsvrcWriteDst                (DSET_CTX* ctx);
static int    ilsvrcCleanup                 (DSET_CTX* ctx, int ret);
static int    ilsvrcCleanupf                (DSET_CTX*   ctx,
                                             int         ret,
                                             const char* fmt,
                                             ...);
static int    ilsvrcCleanupv                (DSET_CTX*   ctx,
                                             int         ret,
                                             const char* fmt,
                                             va_list     ap);
static int    ilsvrcThrdInit                (DSET_THRD_CTX* ctx);
static int    ilsvrcThrdInitHDF5            (DSET_THRD_CTX* ctx);
static int    ilsvrcThrdInitCUDA            (DSET_THRD_CTX* ctx);
static int    ilsvrcThrdInitFFmpeg          (DSET_THRD_CTX* ctx);
static int    ilsvrcThrdRequestExit         (DSET_THRD_CTX* ctx, int ret);
static int    ilsvrcThrdRequestExitf        (DSET_THRD_CTX* ctx,
                                             int            ret,
                                             const char*    fmt,
                                             ...);
static int    ilsvrcThrdRequestExitv        (DSET_THRD_CTX* ctx,
                                             int            ret,
                                             const char*    fmt,
                                             va_list        ap);
static int    ilsvrcThrdHasRequestedExit    (DSET_THRD_CTX* ctx);
static int    ilsvrcThrdGetRequestedExitCode(DSET_THRD_CTX* ctx);
static int    ilsvrcThrdIterBody            (DSET_THRD_CTX* ctx, size_t i);
static int    ilsvrcThrdIterFetchImg        (DSET_THRD_CTX* ctx);
static int    ilsvrcThrdIterFini            (DSET_THRD_CTX* ctx);
static int    ilsvrcThrdFini                (DSET_THRD_CTX* ctx);



/* Static Function Definitions */

/**
 * Initialize.
 */

static int    ilsvrcInit                    (DSET_CTX* ctx, int argc, char** argv){
	memset(ctx, 0, sizeof(*ctx));
	ctx->argc = argc;
	ctx->argv = argv;
	
	return ilsvrcParseArg(ctx, 1, argv);
}

/**
 * Parse one argument.
 */

static int    ilsvrcParseArg                (DSET_CTX* ctx, int argc, char** argv){
	switch(argc){
		case 1:  ctx->args.srcPath = argv[argc++]; break;
		case 2:  ctx->args.dstPath = argv[argc++]; break;
		default: argc++; break;
	}
	
	if(argc < ctx->argc){
		return ilsvrcParseArg(ctx, argc, argv);
	}else{
		return ilsvrcCheckArgs(ctx);
	}
}

/**
 * Check arguments valid.
 */

static int    ilsvrcCheckArgs               (DSET_CTX* ctx){
	struct stat src, dst;
	
	if(!ctx->args.srcPath)
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_ARGS,
		                      "Path to source dataset not provided!\n");
	
	if(!ctx->args.dstPath)
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_ARGS,
		                      "Path to destination dataset not provided!\n");
	
	if(!ctx->args.srcPath                   ||
	   stat(ctx->args.srcPath, &src) != 0   ||
	   !S_ISREG(src.st_mode)                ||
	   !((src.st_mode & S_IRUSR) == S_IRUSR))
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_ARGS,
		                      "Cannot read source dataset %s!\n",
		                      ctx->args.srcPath);
	
	if(stat(ctx->args.dstPath, &dst) == 0)
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_ARGS,
		                      "Destination dataset %s already exists!\n",
		                      ctx->args.dstPath);
	
	return ilsvrcInitCUDA(ctx);
}

/**
 * Initialize CUDA.
 */

static int    ilsvrcInitCUDA                (DSET_CTX* ctx){
	if(cuInit(0) != CUDA_SUCCESS)
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_CUDA,
		                      "Could not initialize CUDA!\n");
	
	if(cudaSetDevice(ctx->args.cudaDev) != cudaSuccess)
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_CUDA,
		                      "Could not select CUDA device %d!\n",
		                      ctx->args.cudaDev);
	
	return ilsvrcInitFFmpeg(ctx);
}

/**
 * Initialize FFmpeg.
 */

static int    ilsvrcInitFFmpeg              (DSET_CTX* ctx){
	avcodec_register_all();
	ctx->jpegDecoder = avcodec_find_decoder_by_name("mjpeg");
	ctx->h264Encoder = avcodec_find_encoder_by_name("libx264");
	
	if(!ctx->jpegDecoder)
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_FFMPEG,
		                      "Could not open JPEG decoder!\n");
	
	if(!ctx->h264Encoder)
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_FFMPEG,
		                      "Could not open x264 encoder!\n");
	
	return ilsvrcInitHDF5(ctx);
}

/**
 * Initialize HDF5 and open source HDF5 dataset.
 */

static int    ilsvrcInitHDF5                (DSET_CTX* ctx){
	int         numMembers, splitNDims;
	H5T_class_t attrClass;
	hsize_t     splitDims[1];
	size_t      startOff,       stopOff, elemSize;
	char*       startName,     *stopName;
	hid_t       startType,      stopType;
	htri_t      startTypeIsU64, stopTypeIsU64;
	int         startHasCorrectName, stopHasCorrectName;
	
	/* import h5py as H */
	if(H5open() < 0)
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_HDF5,
		                      "Could not initialize HDF5 library!\n");
	
	/* f = H.File(ctx->args.srcPath, "r") */
	ctx->srcFile = H5Fopen(ctx->args.srcPath, H5F_ACC_RDONLY, H5P_DEFAULT);
	if(ctx->srcFile < 0)
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_HDF5,
		                      "Could not open source dataset %s as HDF5 file!\n",
		                      ctx->args.srcPath);
	
	/**
	 * X = f["encoded_images"]
	 * Y = f["targets"]
	 */
	ctx->srcFileEncoded = H5Dopen(ctx->srcFile, "encoded_images", H5P_DEFAULT);
	ctx->srcFileTargets = H5Dopen(ctx->srcFile, "targets",        H5P_DEFAULT);
	if(ctx->srcFileEncoded < 0)
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_HDF5,
		                      "Could not open dataset \"/encoded_images\" within HDF5 file!\n");
	if(ctx->srcFileTargets < 0)
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_HDF5,
		                      "Could not open dataset \"/targets\" within HDF5 file!\n");
	
	/**
	 * X.shape
	 * X.dtype
	 */
	
	ctx->srcFileEncodedSpace = H5Dget_space(ctx->srcFileEncoded);
	ctx->srcFileEncodedType  = H5Dget_type (ctx->srcFileEncoded);
	if(ctx->srcFileEncodedSpace < 0||
	   ctx->srcFileEncodedType  < 0)
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_HDF5,
		                      "Could not get dataspace or datatype!\n");
	
	/* X.ndims */
	ctx->srcFileEncodedNDims = H5Sget_simple_extent_ndims(ctx->srcFileEncodedSpace);
	if(ctx->srcFileEncodedNDims != 1)
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_HDF5,
		                      "Dataset in unexpected format!\n");
	
	/* X.shape[:] */
	if(H5Sget_simple_extent_dims(ctx->srcFileEncodedSpace,
	                             ctx->srcFileEncodedDims,
	                             NULL) != ctx->srcFileEncodedNDims)
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_HDF5,
		                      "Could not get dataset dimensions!\n");
	
	/* f.attrs["split"] */
	ctx->srcFileSplit      = H5Aopen(ctx->srcFile, "split", H5P_DEFAULT);
	if(ctx->srcFileSplit < 0)
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_HDF5,
		                      "No splits attribute!\n");
	
	ctx->srcFileSplitType  = H5Aget_type(ctx->srcFileSplit);
	ctx->srcFileSplitNativeType = H5Tget_native_type   (ctx->srcFileSplitType,
	                                                    H5T_DIR_DEFAULT);
	ctx->srcFileSplitSpace = H5Aget_space              (ctx->srcFileSplit);
	attrClass              = H5Tget_class              (ctx->srcFileSplitNativeType);
	numMembers             = H5Tget_nmembers           (ctx->srcFileSplitNativeType);
	elemSize                   = H5Tget_size               (ctx->srcFileSplitNativeType);
	splitNDims             = H5Sget_simple_extent_ndims(ctx->srcFileSplitSpace);
	startOff               = H5Tget_member_offset      (ctx->srcFileSplitNativeType, 2);
	startType              = H5Tget_member_type        (ctx->srcFileSplitNativeType, 2);
	startName              = H5Tget_member_name        (ctx->srcFileSplitNativeType, 2);
	stopOff                = H5Tget_member_offset      (ctx->srcFileSplitNativeType, 3);
	stopType               = H5Tget_member_type        (ctx->srcFileSplitNativeType, 3);
	stopName               = H5Tget_member_name        (ctx->srcFileSplitNativeType, 3);
	startTypeIsU64         = H5Tequal                  (startType, H5T_STD_I64LE);
	stopTypeIsU64          = H5Tequal                  (stopType,  H5T_STD_I64LE);
	startHasCorrectName    = startName && strcmp       (startName, "start") == 0;
	stopHasCorrectName     = stopName  && strcmp       (stopName,  "stop")  == 0;
	H5Tclose(startType);
	H5Tclose(stopType);
	H5free_memory(startName);
	H5free_memory(stopName);
	
	if(splitNDims   !=            1 ||
	   attrClass    != H5T_COMPOUND ||
	   elemSize     !=           56 ||
	   numMembers   !=            7 ||
	   startOff     !=           24 ||
	   stopOff      !=           32 ||
	   !startTypeIsU64              ||
	   !stopTypeIsU64               ||
	   !startHasCorrectName         ||
	   !stopHasCorrectName)
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_HDF5,
		                      "The splits attribute is misshaped!\n");
	
	/* f.attrs["split"].shape == (9,) */
	if(H5Sget_simple_extent_dims(ctx->srcFileSplitSpace, splitDims, 0) != splitNDims ||
	   splitDims[0]                                                    !=          9)
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_HDF5,
		                      "Could not read dimensions of splits attribute!\n");
	
	/* Read f.attrs["split"] */
	ctx->srcFileSplitNativePtr = (SPLIT_ATTR_TYPE*)malloc(splitDims[0] * elemSize);
	if(!ctx->srcFileSplitNativePtr)
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_HDF5,
		                      "Failed to allocate memory for splits attribute data!\n");
	
	if(H5Aread(ctx->srcFileSplit,
	           ctx->srcFileSplitNativeType,
	           ctx->srcFileSplitNativePtr) < 0)
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_HDF5,
		                      "Failed to read splits attribute data!\n");
	
	/* Extract splits information. */
	ctx->nTrain = ctx->srcFileSplitNativePtr[0].stop - ctx->srcFileSplitNativePtr[0].start;
	ctx->nVal   = ctx->srcFileSplitNativePtr[3].stop - ctx->srcFileSplitNativePtr[3].start;
	ctx->nTest  = ctx->srcFileSplitNativePtr[6].stop - ctx->srcFileSplitNativePtr[6].start;
	ctx->nTotal = ctx->nTrain + ctx->nVal + ctx->nTest;
	
	printf("Source dataset path:   %s\n", ctx->args.srcPath);
	printf("Source dataset splits:\n");
	printf("\tTrain: %7llu\n", ctx->nTrain);
	printf("\tVal:   %7llu\n", ctx->nVal);
	printf("\tTest:  %7llu\n", ctx->nTest);
	printf("\tTotal: %7llu\n", ctx->nTotal);
	
	return ilsvrcInitOpenMP(ctx);
}

/**
 * Initialize OpenMP/threading/mutex-related state.
 */

static int    ilsvrcInitOpenMP              (DSET_CTX* ctx){
	omp_init_lock(&ctx->exitLock);
	
	ctx->numThrds = omp_get_max_threads();
	if(ctx->numThrds <= 0)
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_OPENMP,
		                      "Invalid number of OpenMP threads (%d)!\n",
		                      ctx->numThrds);
	
	printf("Using %d threads for conversion.\n", ctx->numThrds);
	
	ctx->thrds = (DSET_THRD_CTX*)calloc(ctx->numThrds, sizeof(*ctx->thrds));
	if(!ctx->thrds)
		return ilsvrcCleanupf(ctx, ILSVRC_EXIT_FAIL_OPENMP,
		                      "Failed to allocate thread workspaces!\n");
	
	return ilsvrcRun(ctx);
}

/**
 * Run dataset conversion (possibly in parallel)
 */

static int    ilsvrcRun                     (DSET_CTX* ctx){
	#pragma omp parallel num_threads(ctx->numThrds)
	{
		hsize_t        i;
		DSET_THRD_CTX* thrdCtx;
		int            thrdNum;
		
		thrdNum          = omp_get_thread_num();
		thrdCtx          = ctx->thrds + thrdNum;
		thrdCtx->thrdNum = thrdNum;
		thrdCtx->dsetCtx = ctx;
		
		ilsvrcThrdInit(thrdCtx);
		#pragma omp for ordered schedule(dynamic, 1)
		for(i=0;i<ctx->nTotal;i++){
			ilsvrcThrdIterBody(thrdCtx, i);
		}
		ilsvrcThrdFini(thrdCtx);
	}
	
	return ilsvrcWriteDst(ctx);
}

/**
 * Write destination HDF5 dataset.
 */

static int    ilsvrcWriteDst                (DSET_CTX* ctx){
	int i;
	
	for(i=0;i<ctx->numThrds;i++){
		printf("Thread %d exited with code %d\n", i,
		       ilsvrcThrdGetRequestedExitCode(ctx->thrds+i));
	}
	
	return ilsvrcCleanup(ctx, ILSVRC_EXIT_SUCCESS);
}

/**
 * Cleanup.
 */

static int    ilsvrcCleanup                 (DSET_CTX* ctx, int ret){
	H5Fclose(ctx->srcFile);
	H5Dclose(ctx->srcFileEncoded);
	H5Dclose(ctx->srcFileTargets);
	
	return ret;
}

/**
 * Cleanup.
 */

static int    ilsvrcCleanupf                (DSET_CTX*   ctx,
                                             int         ret,
                                             const char* fmt,
                                             ...){
	va_list ap;
	va_start(ap, fmt);
	return ilsvrcCleanupv(ctx, ret, fmt, ap);
}

/**
 * Cleanup.
 */

static int    ilsvrcCleanupv                (DSET_CTX*   ctx,
                                             int         ret,
                                             const char* fmt,
                                             va_list     ap){
	vfprintf(stderr, fmt, ap);
	fflush(stderr);
	va_end(ap);
	return ilsvrcCleanup(ctx, ret);
}

/**
 * Thread init.
 */

static int    ilsvrcThrdInit                (DSET_THRD_CTX* ctx){
	return ilsvrcThrdInitHDF5(ctx);
}

/**
 * Initialize per-thread HDF5 resources.
 */

static int    ilsvrcThrdInitHDF5            (DSET_THRD_CTX* ctx){
	const hsize_t ONE=1;
	
	/* Create file and memory selection dataspaces. */
	ctx->imgFSpace = H5Dget_space(ctx->dsetCtx->srcFileEncoded);
	ctx->imgMSpace = H5Screate_simple(1, &ONE, &ONE);
	if(ctx->imgFSpace < 0 ||
	   ctx->imgMSpace < 0)
		return ilsvrcThrdRequestExitf(ctx, ILSVRC_EXIT_FAIL_HDF5,
		                              "Could not allocate selection dataspaces!\n");
	
	return ilsvrcThrdInitCUDA(ctx);
}

/**
 * Initialize per-thread CUDA resources.
 */

static int    ilsvrcThrdInitCUDA            (DSET_THRD_CTX* ctx){
	//CUVIDSOURCEDATAPACKET packet;
	
	memset(&ctx->parserParams, 0, sizeof(ctx->parserParams));
	ctx->parserParams.CodecType              = cudaVideoCodec_H264;
	ctx->parserParams.ulMaxNumDecodeSurfaces = 8;
	ctx->parserParams.ulClockRate            = 0;
	ctx->parserParams.ulErrorThreshold       = 100;
	ctx->parserParams.ulMaxDisplayDelay      = 4;
	ctx->parserParams.pUserData              = ctx;
	ctx->parserParams.pfnSequenceCallback    = (PFNVIDSEQUENCECALLBACK)NULL;
	ctx->parserParams.pfnDecodePicture       = (PFNVIDDECODECALLBACK)  NULL;
	ctx->parserParams.pfnDisplayPicture      = (PFNVIDDISPLAYCALLBACK) NULL;
	ctx->parserParams.pExtVideoInfo          = NULL;
	
	cuvidCreateVideoParser (&ctx->parser, &ctx->parserParams);
	//cuvidParseVideoData    (ctx->parser, &ctx->packet);
	cuvidDestroyVideoParser(ctx->parser);
	
	return ilsvrcThrdInitFFmpeg(ctx);
}

/**
 * Initialize per-thread FFmpeg resources.
 */

static int    ilsvrcThrdInitFFmpeg          (DSET_THRD_CTX* ctx){
	int i;
	
	ctx->jpegDecCtx  = avcodec_alloc_context3(ctx->dsetCtx->jpegDecoder);
	ctx->h264EncCtx  = avcodec_alloc_context3(ctx->dsetCtx->h264Encoder);
	if(!ctx->jpegDecCtx)
		return ilsvrcThrdRequestExitf(ctx, ILSVRC_EXIT_FAIL_FFMPEG,
		                              "Could not allocate JPEG decoding context!\n");
	if(!ctx->h264EncCtx)
		return ilsvrcThrdRequestExitf(ctx, ILSVRC_EXIT_FAIL_FFMPEG,
		                              "Could not allocate h264 encoding context!\n");
	
	i = 0;
	av_dict_set    (&ctx->h264Opts, "x264-params",     "log=-1:chromaloc=1",   0);i++;
	av_dict_set_int(&ctx->h264Opts, "forced-idr",      1,                      0);i++;
	av_dict_set_int(&ctx->h264Opts, "crf",             13,                     0);i++;
	if(av_dict_count(ctx->h264Opts) != i)
		return ilsvrcThrdRequestExitf(ctx, ILSVRC_EXIT_FAIL_FFMPEG,
		                              "Failed to create options dictionary!\n");
	
	ctx->h264EncCtx->sample_aspect_ratio    = av_make_q(1, 1);
	ctx->h264EncCtx->time_base              = av_make_q(1, 25);
	ctx->h264EncCtx->pix_fmt                = AV_PIX_FMT_YUVJ420P;
	ctx->h264EncCtx->color_range            = AVCOL_RANGE_JPEG;
	ctx->h264EncCtx->color_trc              = AVCOL_TRC_BT709;
	ctx->h264EncCtx->color_primaries        = AVCOL_PRI_BT470BG;
	ctx->h264EncCtx->colorspace             = AVCOL_SPC_BT470BG;
	/* Unfortunately ignored by x264, passed as x264params chromaloc=1 above */
	ctx->h264EncCtx->chroma_sample_location = AVCHROMA_LOC_CENTER;
	ctx->h264EncCtx->width                  = 512;
	ctx->h264EncCtx->height                 = 512;
	ctx->h264EncCtx->gop_size               = 0;
	ctx->h264EncCtx->profile                = FF_PROFILE_H264_HIGH;
	
	if(avcodec_open2(ctx->jpegDecCtx, ctx->dsetCtx->jpegDecoder, &ctx->jpegOpts) < 0||
	   avcodec_open2(ctx->h264EncCtx, ctx->dsetCtx->h264Encoder, &ctx->h264Opts) < 0)
		return ilsvrcThrdRequestExitf(ctx, ILSVRC_EXIT_FAIL_FFMPEG,
		                              "Failed to create encoder/decoder contexts!\n");
	
	return 0;
}

/**
 * Request exit.
 */

static int    ilsvrcThrdRequestExit         (DSET_THRD_CTX* ctx, int ret){
	(void)ilsvrcThrdRequestExit;
	return ilsvrcThrdRequestExitf(ctx, ret, "");
}

/**
 * Request exit.
 */

static int    ilsvrcThrdRequestExitf        (DSET_THRD_CTX* ctx,
                                             int            ret,
                                             const char*    fmt,
                                             ...){
	va_list ap;
	va_start(ap, fmt);
	return ilsvrcThrdRequestExitv(ctx, ret, fmt, ap);
}

/**
 * Request exit.
 */

static int    ilsvrcThrdRequestExitv        (DSET_THRD_CTX* ctx,
                                             int            ret,
                                             const char*    fmt,
                                             va_list        ap){
	omp_set_lock  (&ctx->dsetCtx->exitLock);
	fprintf (stderr, "[Thrd %3d] ", ctx->thrdNum);
	vfprintf(stderr, fmt, ap);
	fflush(stderr);
	va_end(ap);
	
	ctx->dsetCtx->exiting  = 1;
	ctx->dsetCtx->exitCode = ret;
	omp_unset_lock(&ctx->dsetCtx->exitLock);
	return ret;
}

/**
 * Has a thread requested exit?
 */

static int    ilsvrcThrdHasRequestedExit    (DSET_THRD_CTX* ctx){
	int isRequestingExit;
	omp_set_lock  (&ctx->dsetCtx->exitLock);
	isRequestingExit = !!ctx->dsetCtx->exiting;
	omp_unset_lock(&ctx->dsetCtx->exitLock);
	return isRequestingExit;
}

/**
 * Get requested exit code.
 */

static int    ilsvrcThrdGetRequestedExitCode(DSET_THRD_CTX* ctx){
	int requestedExitCode;
	omp_set_lock  (&ctx->dsetCtx->exitLock);
	requestedExitCode = ctx->dsetCtx->exitCode;
	omp_unset_lock(&ctx->dsetCtx->exitLock);
	return requestedExitCode;
}

/**
 * Run thread body for iteration i.
 */

static int    ilsvrcThrdIterBody            (DSET_THRD_CTX* ctx, size_t i){
	ctx->imgNum = i;
	
#if 1
	/* FIXME: Delete when you stop experimenting. */
	if(ctx->thrdNum != 0 || ctx->imgCnt>0){
		return 0;
	}
#endif
	
	
	if(ilsvrcThrdHasRequestedExit(ctx)){
		return 0;
	}else{
		return ilsvrcThrdIterFetchImg(ctx);
	}
}

/**
 * Fetch JPEG image from dataset.
 */

static int    ilsvrcThrdIterFetchImg        (DSET_THRD_CTX* ctx){
	const hsize_t ZERO=0;
	
	/* Select elements for read in file and for write in memory dataspaces. */
	if(H5Sselect_elements(ctx->imgFSpace, H5S_SELECT_SET, 1, &ctx->imgNum) < 0 ||
	   H5Sselect_elements(ctx->imgMSpace, H5S_SELECT_SET, 1,        &ZERO) < 0)
		return ilsvrcThrdRequestExitf(ctx, ILSVRC_EXIT_FAIL_HDF5,
		                              "Could not select dataset element!\n");
	
	/* Perform actual read of variable-length array. */
	if(H5Dread(ctx->dsetCtx->srcFileEncoded,
	           ctx->dsetCtx->srcFileEncodedType,
	           ctx->imgMSpace,
	           ctx->imgFSpace,
	           H5P_DEFAULT,
	           ctx->imgData) < 0)
		return ilsvrcThrdRequestExitf(ctx, ILSVRC_EXIT_FAIL_HDF5,
		                              "Could not read image %llu!\n",
		                              ctx->imgNum);
	
	ctx->imgJPEGSize = ctx->imgData->len;
	
	/**
	 * Allocate working space for FFMPEG, including codec-required end padding
	 * of zeros.
	 */
	
	ctx->imgJPEG = calloc(1, ctx->imgData->len + AV_INPUT_BUFFER_PADDING_SIZE);
	if(!ctx->imgJPEG)
		return ilsvrcThrdRequestExitf(ctx, ILSVRC_EXIT_FAIL_HDF5,
		                              "Could not allocate buffer for JPEG image!\n");
	
	/* Copy to working space. */
	memcpy(ctx->imgJPEG, ctx->imgData->p, ctx->imgData->len);
	
	/* Free HDF5 VL array. */
	H5Dvlen_reclaim(ctx->dsetCtx->srcFileEncodedType,
	                ctx->imgMSpace,
	                H5P_DEFAULT,
	                ctx->imgData);
	
	/* Continue. */
	return ilsvrcThrdIterFini(ctx);
}

/**
 * Finish thread iteration.
 */

static int    ilsvrcThrdIterFini            (DSET_THRD_CTX* ctx){
	printf("%p[%llu] = {", ctx->imgJPEG, ctx->imgJPEGSize);
	for(size_t i=0;i<32;i++){
		printf("%c", ((char*)ctx->imgJPEG)[i]);
	}
	printf("}\n");
	free(ctx->imgJPEG);
	
	//printf("Thrd# %d printing unordered %llu\n", omp_get_thread_num(), i);
	
	/**
	 * But write it to destination in order.
	 */
	
	#pragma omp ordered
	{
		ctx->imgCnt++;
		//printf("Thrd# %d printing %llu\n", omp_get_thread_num(), i);
	}
	
	return 0;
}

/**
 * Tear down thread state.
 */

static int    ilsvrcThrdFini                (DSET_THRD_CTX* ctx){
	(void)ctx;
	return 0;
}



/* External Function Definitions */

/**
 * Main
 */

int   main(int argc, char* argv[]){
	DSET_CTX STACKCTX, *ctx=&STACKCTX;
	
	return ilsvrcInit(ctx, argc, argv);
}

