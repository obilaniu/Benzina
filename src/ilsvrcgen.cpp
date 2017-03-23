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
#define ILSVRC_EXIT_FAIL_FFMPEG    2
#define ILSVRC_EXIT_FAIL_HDF5      3
#define ILSVRC_EXIT_FAIL_OPENMP    4



/* Data Structure Forward Declarations and Typedefs */
struct DSET_CTX;
struct DSET_THRD_CTX;

typedef struct DSET_CTX      DSET_CTX;
typedef struct DSET_THRD_CTX DSET_THRD_CTX;



/* Data Structure Definitions */

/**
 * Per-thread context.
 */

typedef struct DSET_THRD_CTX{
	DSET_CTX* dsetCtx;
	int       thrdNum;
	
	AVCodecContext*  jpegDecCtx;
	AVCodecContext*  h264EncCtx;
	AVDictionary*    jpegOpts;
	AVDictionary*    h264Opts;
} DSET_THRD_CTX;


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
	const char*      srcPath;
	const char*      dstPath;
	
	AVCodec*         jpegDecoder;
	AVCodec*         h264Encoder;
	
	hid_t            srcFile;
	hid_t            srcFileSplit;
	hid_t            srcFileSplitType;
	hid_t            srcFileSplitSpace;
	H5A_info_t       srcFileSplitInfo;
	hid_t            srcFileEncoded;
	hid_t            srcFileTargets;
	hid_t            srcFileEncodedSpace;
	hid_t            srcFileEncodedType;
	int              srcFileEncodedNDims;
	hsize_t          srcFileEncodedDims[1];
	hsize_t          nTotal, nTrain, nVal, nTest;
	
	int              numThrds;
	DSET_THRD_CTX*   thrds;
	
	AVCodecContext*  jpegDecCtx;
	AVCodecContext*  h264EncCtx;
	AVDictionary*    jpegOpts;
	AVDictionary*    h264Opts;
};



/* Static Function Prototypes */
static int    checkArgs            (DSET_CTX* ctx, int argc, char* argv[]);
static int    initCUDA             (DSET_CTX* ctx);
static int    initFFmpeg           (DSET_CTX* ctx);
static int    openHDF5Src          (DSET_CTX* ctx);
static int    readSrc              (DSET_CTX* ctx);
static int    runConversion        (DSET_CTX* ctx);
static int    cleanup              (DSET_CTX* ctx, int ret);



/* Static Function Definitions */

/**
 * Check arguments.
 */

static int   checkArgs            (DSET_CTX* ctx, int argc, char* argv[]){
	struct stat src, dst;
	memset(ctx, 0, sizeof(*ctx));
	ctx->argc    = argc;
	ctx->argv    = argv;
	ctx->srcPath = argv[1];
	ctx->dstPath = argv[2];
	
	if(argc < 3){
		return cleanup(ctx, ILSVRC_EXIT_FAIL_ARGS);
	}
	
	if(stat(ctx->srcPath, &src) != 0        ||
	   !S_ISREG(src.st_mode)                ||
	   !((src.st_mode & S_IRUSR) == S_IRUSR)){
		fprintf(stderr, "Cannot read source dataset %s!\n", argv[1]);
		return cleanup(ctx, ILSVRC_EXIT_FAIL_ARGS);
	}
	
	if(stat(ctx->dstPath, &dst) == 0){
		fprintf(stderr, "Destination dataset %s already exists!\n", argv[2]);
		return cleanup(ctx, ILSVRC_EXIT_FAIL_ARGS);
	}
	
	return initCUDA(ctx);
}

/**
 * Initialize CUDA.
 */

static int   initCUDA             (DSET_CTX* ctx){
	cuInit(0);
	cudaSetDevice(0);
	
	CUvideoparser         parser;
	CUVIDPARSERPARAMS     parserParams;
	//CUVIDSOURCEDATAPACKET packet;
	
	memset(&parserParams, 0, sizeof(parserParams));
	parserParams.CodecType              = cudaVideoCodec_H264;
	parserParams.ulMaxNumDecodeSurfaces = 8;
	parserParams.ulClockRate            = 0;
	parserParams.ulErrorThreshold       = 100;
	parserParams.ulMaxDisplayDelay      = 4;
	parserParams.pUserData              = ctx;
	parserParams.pfnSequenceCallback    = (PFNVIDSEQUENCECALLBACK)NULL;
	parserParams.pfnDecodePicture       = (PFNVIDDECODECALLBACK)  NULL;
	parserParams.pfnDisplayPicture      = (PFNVIDDISPLAYCALLBACK) NULL;
	parserParams.pExtVideoInfo          = NULL;
	
	cuvidCreateVideoParser (&parser, &parserParams);
	//cuvidParseVideoData    (parser, &packet);
	cuvidDestroyVideoParser(parser);
	
	return initFFmpeg(ctx);
}

/**
 * Initialize FFmpeg.
 */

static int   initFFmpeg           (DSET_CTX* ctx){
	int i;
	
	avcodec_register_all();
	ctx->jpegDecoder = avcodec_find_decoder_by_name("mjpeg");
	ctx->h264Encoder = avcodec_find_encoder_by_name("libx264");
	if(!ctx->jpegDecoder){
		fprintf(stderr, "Could not open JPEG decoder!\n");
		return cleanup(ctx, ILSVRC_EXIT_FAIL_FFMPEG);
	}
	if(!ctx->h264Encoder){
		fprintf(stderr, "Could not open x264 encoder!\n");
		return cleanup(ctx, ILSVRC_EXIT_FAIL_FFMPEG);
	}
	
	ctx->jpegDecCtx  = avcodec_alloc_context3(ctx->jpegDecoder);
	ctx->h264EncCtx  = avcodec_alloc_context3(ctx->h264Encoder);
	if(!ctx->jpegDecCtx){
		fprintf(stderr, "Could not allocate JPEG decoding context!\n");
		return cleanup(ctx, ILSVRC_EXIT_FAIL_FFMPEG);
	}
	if(!ctx->h264EncCtx){
		fprintf(stderr, "Could not allocate h264 encoding context!\n");
		return cleanup(ctx, ILSVRC_EXIT_FAIL_FFMPEG);
	}
	
	/**
	 * H264 ENCODER SETTINGS
	 * 
	 * We must be extremely careful here. There's a lot of intricate detail. In
	 * particular, refer to:
	 * 
	 * - Rec. ITU-R BT.470-6    (11/1998)
	 * - Rec. ITU-R BT.601-7    (03/2011)
	 * - Rec. ITU-T T.871       (05/2011)
	 * - Rec. ITU-T H.264       (10/2016)
	 * 
	 * 
	 * The situation is as follows.
	 * 
	 *   - We're loading data from JPEG images and transcoding to h264 IDR frames.
	 * 
	 *   - JFIF (the interchange format for JPEG) requires YCbCr coding of image
	 *     data and references the color matrix of "625-line BT601-6", with
	 *     modifications that make use of full-range (256) quantization levels.
	 * 
	 *   - The most popular chroma subsampling method is YUV 4:2:0, meaning that
	 *     the U & V chroma samples are subsampled 2x in both horizontal and
	 *     vertical directions. In ImageNet there are also YUV 4:4:4-coded images.
	 * 
	 *   - JFIF's chroma samples are, if subsampled, offset as follows w.r.t. the
	 *     luma samples:
	 *         Hoff = H_downsample_factor/2 - 0.5
	 *         Voff = V_downsample_factor/2 - 0.5
	 * 
	 *   - Nvidia's NVDEC is only capable of decoding h264 High Profile 4.1
	 *     YUV420P in NV12 format, with unknown support for full-scale YUV.
	 * 
	 *   - FFmpeg will produce I-frame-only video if ctx->gop_size == 0.
	 * 
	 *   - FFmpeg won't make an encoder context unless it's been told a timebase,
	 *     width and height.
	 * 
	 *   - FFmpeg will force x264 to mark an I-frame as an IDR-frame
	 *     (Instantaneous Decoder Refresh) if the option forced_idr == 1.
	 * 
	 *   - x264 won't shut up unless its log-level is set to none (log=-1)
	 * 
	 *   - In H264 the above can be coded if:
	 *         video_full_range_flag               = 1 (pc/full range)
	 *         colour_description_present_flag     = 1
	 *         matrix_coefficients                 = 5 (Rec. ITU-R BT.601-6 625)
	 *         chroma_format_idc                   = 1 (YUV 4:2:0)
	 *         chroma_loc_info_present_flag        = 1
	 *         chroma_sample_loc_type_top_field    = 1 (center sample)
	 *         chroma_sample_loc_type_bottom_field = 1 (center sample)
	 *     Given that the colorspace is that of Rec. ITU-R BT.601-6 625 (PAL), a
	 *     reasonable guess is that the transfer characteristics and primaries are
	 *     also of that standard, even though they are unspecified in ImageNet:
	 *         colour_primaries                    = 5 (Rec. ITU-R BT.601-6 625)
	 *         transfer_characteristics            = 1 (Rec. ITU-R BT.601-6 625 is
	 *                                                  labelled "6", but "1", which
	 *                                                  corresponds to BT.709-5, is
	 *                                                  functionally equivalent and
	 *                                                  explicitly preferred by the
	 *                                                  H264 standard)
	 */
	
	i = 0;
	av_dict_set    (&ctx->h264Opts, "x264-params",     "log=-1:chromaloc=1",   0);i++;
	av_dict_set_int(&ctx->h264Opts, "forced-idr",      1,                      0);i++;
	av_dict_set_int(&ctx->h264Opts, "crf",             13,                     0);i++;
	if(av_dict_count(ctx->h264Opts) != i){
		fprintf(stderr, "Failed to create options dictionary!\n");
		return cleanup(ctx, ILSVRC_EXIT_FAIL_FFMPEG);
	}
	
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
	
	if(avcodec_open2(ctx->jpegDecCtx, ctx->jpegDecoder, &ctx->jpegOpts) < 0||
	   avcodec_open2(ctx->h264EncCtx, ctx->h264Encoder, &ctx->h264Opts) < 0){
		fprintf(stderr, "Failed to create encoder/decoder contexts!\n");
		return cleanup(ctx, ILSVRC_EXIT_FAIL_FFMPEG);
	}
	
	return openHDF5Src(ctx);
}

/**
 * Open source HDF5 dataset.
 */

static int   openHDF5Src           (DSET_CTX* ctx){
	if(H5open() < 0){
		fprintf(stderr, "Could not initialize HDF5 library!\n");
		return cleanup(ctx, ILSVRC_EXIT_FAIL_HDF5);
	}
	
	ctx->srcFile = H5Fopen(ctx->srcPath, H5F_ACC_RDONLY, H5P_DEFAULT);
	if(ctx->srcFile < 0){
		fprintf(stderr, "Could not open source dataset %s as HDF5 file!\n",
		        ctx->srcPath);
		return cleanup(ctx, ILSVRC_EXIT_FAIL_HDF5);
	}
	
	ctx->srcFileEncoded = H5Dopen(ctx->srcFile, "encoded_images", H5P_DEFAULT);
	ctx->srcFileTargets = H5Dopen(ctx->srcFile, "targets",        H5P_DEFAULT);
	if(ctx->srcFileEncoded < 0){
		fprintf(stderr, "Could not open dataset \"/encoded_images\" within HDF5 file!\n");
		return cleanup(ctx, ILSVRC_EXIT_FAIL_HDF5);
	}
	if(ctx->srcFileTargets < 0){
		fprintf(stderr, "Could not open dataset \"/targets\" within HDF5 file!\n");
		return cleanup(ctx, ILSVRC_EXIT_FAIL_HDF5);
	}
	
	ctx->srcFileEncodedSpace = H5Dget_space(ctx->srcFileEncoded);
	ctx->srcFileEncodedType  = H5Dget_type (ctx->srcFileEncoded);
	if(ctx->srcFileEncodedSpace < 0||
	   ctx->srcFileEncodedType  < 0){
		fprintf(stderr, "Could not get dataspace or datatype!\n");
		return cleanup(ctx, ILSVRC_EXIT_FAIL_HDF5);
	}
	
	ctx->srcFileEncodedNDims = H5Sget_simple_extent_ndims(ctx->srcFileEncodedSpace);
	if(ctx->srcFileEncodedNDims != 1){
		fprintf(stderr, "Dataset in unexpected format!\n");
		return cleanup(ctx, ILSVRC_EXIT_FAIL_HDF5);
	}
	if(H5Sget_simple_extent_dims(ctx->srcFileEncodedSpace,
	                             ctx->srcFileEncodedDims,
	                             NULL) != 1){
		fprintf(stderr, "Could not get dataset dimensions!\n");
		return cleanup(ctx, ILSVRC_EXIT_FAIL_HDF5);
	}
	
	ctx->srcFileSplit      = H5Aopen(ctx->srcFile, "split", H5P_DEFAULT);
	if(ctx->srcFileSplit < 0){
		fprintf(stderr, "No splits attribute!\n");
		return cleanup(ctx, ILSVRC_EXIT_FAIL_HDF5);
	}
	
	ctx->srcFileSplitType  = H5Aget_type    (ctx->srcFileSplit);
	ctx->srcFileSplitSpace = H5Aget_space   (ctx->srcFileSplit);
	int numMembers         = H5Tget_nmembers(ctx->srcFileSplitType);
	printf("Split datatype has %d members:\n", numMembers);
	for(int i=0;i<numMembers;i++){
		char*  name = H5Tget_member_name(ctx->srcFileSplitType, i);
		hid_t  type = H5Tget_member_type(ctx->srcFileSplitType, i);
		size_t size = H5Tget_size(type);
		printf("\t%s (%zu bytes)\n", name, size);
		H5free_memory(name);
	}
	
	if(H5Aget_info(ctx->srcFileSplit, &ctx->srcFileSplitInfo) < 0){
		fprintf(stderr, "Could not get splits attribute info!\n");
		return cleanup(ctx, ILSVRC_EXIT_FAIL_HDF5);
	}
	
	typedef struct SPLITATTRTYPE{
		char     split[5];
		char     source[14];
		uint64_t start;
		uint64_t stop;
		void*    indices;
		char     available;
		char     comment;
	} __attribute__((packed)) SPLITATTRTYPE;
	printf("Attribute size: %zu\n", sizeof(SPLITATTRTYPE));
	SPLITATTRTYPE* splits = (SPLITATTRTYPE*)malloc(ctx->srcFileSplitInfo.data_size);
	H5Aread(ctx->srcFileSplit, ctx->srcFileSplitType, splits);
	
	ctx->nTrain = splits[0].stop-splits[0].start;
	ctx->nVal   = splits[3].stop-splits[3].start;
	ctx->nTest  = splits[6].stop-splits[6].start;
	ctx->nTotal = ctx->nTrain + ctx->nVal + ctx->nTest;
	
	printf("Dataset splits:\n");
	printf("\tTrain: %7llu\n", ctx->nTrain);
	printf("\tVal:   %7llu\n", ctx->nVal);
	printf("\tTest:  %7llu\n", ctx->nTest);
	printf("\tTotal: %7llu\n", ctx->nTotal);
	
	return runConversion(ctx);
}

/**
 * Run dataset conversion (possibly in parallel)
 */

static int   runConversion        (DSET_CTX* ctx){
	ctx->numThrds = omp_get_max_threads();
	if(ctx->numThrds <= 0){
		fprintf(stderr, "Invalid number of OpenMP threads (%d)!\n", ctx->numThrds);
		return cleanup(ctx, ILSVRC_EXIT_FAIL_OPENMP);
	}
	printf("Using %d threads for conversion.\n", ctx->numThrds);
	
	ctx->thrds = (DSET_THRD_CTX*)calloc(ctx->numThrds, sizeof(*ctx->thrds));
	if(ctx->thrds == NULL){
		return cleanup(ctx, ILSVRC_EXIT_FAIL_OPENMP);
	}
	
	
	/**
	 * OpenMP-parallelizable section.
	 */
	
	#pragma omp parallel num_threads(ctx->numThrds)
	{
		hsize_t i;
		
		/**
		 * For each image,
		 */
		
		#pragma omp for schedule(dynamic, 1) ordered
		for(i=0;i<ctx->nTotal;i++){
			/**
			 * Process it in a possibly-unordered way
			 */
			
			//printf("Thrd# %d printing unordered %llu\n", omp_get_thread_num(), i);
			
			/**
			 * But write it to destination in order.
			 */
			
			#pragma omp ordered
			{
				//printf("Thrd# %d printing %llu\n", omp_get_thread_num(), i);
			}
		}
	}
	
	return readSrc(ctx);
}

/**
 * Read source HDF5 dataset.
 */

static int   readSrc              (DSET_CTX* ctx){
	printf("/encoded_images has rank %d, dims (%lld)\n",
	       ctx->srcFileEncodedNDims, ctx->srcFileEncodedDims[0]);
	
	int getClass = H5Tget_class(ctx->srcFileEncodedType);
	printf("/encoded_images class %d.\n", getClass);
	
	hid_t baseType = H5Tget_super(ctx->srcFileEncodedType);
	printf("/encoded_images has base type %d.\n", H5Tequal(baseType, H5T_STD_U8LE));
	
	const hsize_t batchSize = 5;
	hsize_t imgNum = 2, imgSize = 0, batchImgNum = 0;
	hvl_t   imgData[batchSize] = {0};
	herr_t  err = 0;
	err = H5Sselect_none(ctx->srcFileEncodedSpace);
	err = H5Sselect_elements(ctx->srcFileEncodedSpace, H5S_SELECT_APPEND, 1, &imgNum);
	err = H5Dvlen_get_buf_size(ctx->srcFileEncoded,
	                           ctx->srcFileEncodedType,
	                           ctx->srcFileEncodedSpace,
	                           &imgSize);
	printf("Image #%lld of /encoded_images has size %lld.\n", imgNum, imgSize);
	hid_t memSpace = H5Screate_simple(1, &batchSize, &batchSize);
	err = H5Sselect_none(memSpace);
	err = H5Sselect_elements(memSpace, H5S_SELECT_APPEND, 1, &batchImgNum);
	err = H5Dread(ctx->srcFileEncoded,
	              ctx->srcFileEncodedType,
	              memSpace,
	              ctx->srcFileEncodedSpace,
	              H5P_DEFAULT, imgData);
	printf("H5Dread err = %d\n", err);
	
	for(hsize_t i=0;i<batchSize;i++){
		fprintf(stdout, "%p[%zu] {", imgData[i].p, imgData[i].len);
		for(hsize_t j=0;imgData[i].p && j<32;j++){
			fprintf(stdout, "%02x", ((unsigned char*)imgData[i].p)[j]);
		}
		fprintf(stdout, "}\n");
	}
	
	H5Dvlen_reclaim(ctx->srcFileEncodedType,
	                memSpace,
	                H5P_DEFAULT,
	                imgData);
	
	
	return cleanup(ctx, ILSVRC_EXIT_SUCCESS);
}

/**
 * Cleanup.
 */

static int   cleanup              (DSET_CTX* ctx, int ret){
	H5Fclose(ctx->srcFile);
	H5Dclose(ctx->srcFileEncoded);
	H5Dclose(ctx->srcFileTargets);
	
	return ret;
}



/* External Function Definitions */

/**
 * Main
 */

int   main(int argc, char* argv[]){
	DSET_CTX STACKCTX, *ctx=&STACKCTX;
	
	return checkArgs(ctx, argc, argv);
}

