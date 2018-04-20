/* Includes */
#include <cuda.h>
#include <dlfcn.h>
#include <dynlink_cuviddec.h>
#include <dynlink_nvcuvid.h>
#include <pthread.h>

#include "benzina.h"


/* Defines */
#define CUVID_LIBRARY_NAME       "libnvcuvid.so"
#define DEFINE_CUVID_SYMBOL(fn)  static t##fn* fn = (t##fn*)NULL



/* Data Structure Definitions */
struct BENZINA_DATASET{
	const char* root;
	size_t      length;
	CUVIDDECODECREATEINFO info;
};



/* Static Function Declarations */
static void  benzinaInitOnce(void);



/* Global Variables & Constants. */
static pthread_once_t benzinaInitOnceControl = PTHREAD_ONCE_INIT;
static int            benzinaInitOnceStatus  = -1;
static void*          cuvidHandle            = NULL;
DEFINE_CUVID_SYMBOL  (cuvidCreateVideoParser);
DEFINE_CUVID_SYMBOL  (cuvidParseVideoData);
DEFINE_CUVID_SYMBOL  (cuvidDestroyVideoParser);
DEFINE_CUVID_SYMBOL  (cuvidGetDecoderCaps);
DEFINE_CUVID_SYMBOL  (cuvidCreateDecoder);
DEFINE_CUVID_SYMBOL  (cuvidDestroyDecoder);
DEFINE_CUVID_SYMBOL  (cuvidDecodePicture);
DEFINE_CUVID_SYMBOL  (cuvidMapVideoFrame64);
DEFINE_CUVID_SYMBOL  (cuvidUnmapVideoFrame64);


/* Static Function Definitions */
static void  benzinaInitOnce(void){
	/* libnvcuvid loading. */
	cuvidHandle = dlopen(CUVID_LIBRARY_NAME, RTLD_LAZY);
	if(!cuvidHandle){
		benzinaInitOnceStatus = -2; return;
	}
	#define INIT_CUVID_SYMBOL(fn)  fn = *(t##fn*)dlsym(cuvidHandle, #fn)
	INIT_CUVID_SYMBOL(cuvidCreateVideoParser);
	INIT_CUVID_SYMBOL(cuvidParseVideoData);
	INIT_CUVID_SYMBOL(cuvidDestroyVideoParser);
	INIT_CUVID_SYMBOL(cuvidGetDecoderCaps);
	INIT_CUVID_SYMBOL(cuvidCreateDecoder);
	INIT_CUVID_SYMBOL(cuvidDestroyDecoder);
	INIT_CUVID_SYMBOL(cuvidDecodePicture);
	INIT_CUVID_SYMBOL(cuvidMapVideoFrame64);
	INIT_CUVID_SYMBOL(cuvidUnmapVideoFrame64);
	#undef INIT_CUVID_SYMBOL
	
	/* Return with 0 indicating initialization success. */
	benzinaInitOnceStatus = 0;
}


/* Public Function Definitions */
int          benzinaInit(void){
	int    ret = pthread_once(&benzinaInitOnceControl, benzinaInitOnce);
	return ret < 0 ? ret : benzinaInitOnceStatus;
}

int          benzinaDatasetAlloc(BENZINA_DATASET** ctx){
	return -!(*ctx = malloc(sizeof(**ctx)));
}

int          benzinaDatasetInit (BENZINA_DATASET*  ctx, const char* root){
	int w, h;
	cudaVideoCodec         codecType;
	cudaVideoChromaFormat  chromaFormat;
	cudaVideoSurfaceFormat outputFormat;
	
	/* Clear old contents. */
	memset(ctx, 0, sizeof(*ctx));
	
	ctx->root   = root;
	
	
	/**
	 * We should read this from the path, but we support only one format
	 * right now.
	 */
	
	w            = 256;
	h            = 256;
	codecType    = cudaVideoCodec_H264;
	chromaFormat = cudaVideoChromaFormat_420;
	outputFormat = cudaVideoSurfaceFormat_NV12;
	
	/* We support only one dataset too. */
	ctx->length     = 1431167;
	
	/* Set the fields of future decoders created for this dataset. */
	ctx->info.ulWidth             = w;
	ctx->info.ulHeight            = h;
	ctx->info.ulNumDecodeSurfaces = 4;
	ctx->info.CodecType           = codecType;
	ctx->info.ChromaFormat        = chromaFormat;
	ctx->info.ulCreationFlags     = cudaVideoCreate_PreferCUVID;
	ctx->info.bitDepthMinus8      = 0;
	ctx->info.ulIntraDecodeOnly   = 1;
	ctx->info.display_area.left   = 0;
	ctx->info.display_area.top    = 0;
	ctx->info.display_area.right  = ctx->info.ulWidth;
	ctx->info.display_area.bottom = ctx->info.ulHeight;
	ctx->info.OutputFormat        = outputFormat;
	ctx->info.DeinterlaceMode     = cudaVideoDeinterlaceMode_Weave;
	ctx->info.ulTargetWidth       = ctx->info.ulWidth;
	ctx->info.ulTargetHeight      = ctx->info.ulHeight;
	ctx->info.ulNumOutputSurfaces = 4;
	ctx->info.vidLock             = NULL;
	ctx->info.target_rect.left    = 0;
	ctx->info.target_rect.top     = 0;
	ctx->info.target_rect.right   = ctx->info.ulTargetWidth;
	ctx->info.target_rect.bottom  = ctx->info.ulTargetHeight;
	
	/* Successfully initialized. */
	return 0;
}

int          benzinaDatasetNew  (BENZINA_DATASET** ctx, const char* root){
	int ret;
	
	ret = benzinaDatasetAlloc(ctx);
	if(ret != 0){return ret;}
	
	return benzinaDatasetInit(*ctx, root);
}

int          benzinaDatasetFini (BENZINA_DATASET*  ctx){
	memset(ctx, 0, sizeof(*ctx));
	return 0;
}

int          benzinaDatasetFree (BENZINA_DATASET*  ctx){
	benzinaDatasetFini(ctx);
	free(ctx);
	return 0;
}

int          benzinaDatasetGetLength(BENZINA_DATASET*  ctx, size_t* length){
	*length = ctx->length;
	return 0;
}

int          benzinaDatasetGetShape (BENZINA_DATASET*  ctx, size_t* w, size_t* h){
	*w = ctx->info.ulWidth;
	*h = ctx->info.ulHeight;
	return 0;
}

