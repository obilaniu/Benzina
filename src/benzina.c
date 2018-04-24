/* Includes */
#include <cuda.h>
#include <dlfcn.h>
#include <dynlink_cuviddec.h>
#include <dynlink_nvcuvid.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "benzina/benzina.h"


/* Defines */
#define CUVID_LIBRARY_NAME       "libnvcuvid.so"
#define DEFINE_CUVID_SYMBOL(fn)  t##fn* fn = (t##fn*)NULL



/* Data Structure Definitions */

/**
 * @brief Benzina Dataset.
 * 
 * A 
 */

struct BENZINA_DATASET{
	char*                 root;
	size_t                length;
	uint64_t*             lengths;
	uint64_t*             offsets;
	CUVIDDECODECREATEINFO info;
};



/* Static Function Declarations */
BENZINA_STATIC void  benzinaInitOnce(void);



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

/**
 * Perform initialization of libbenzina exactly once.
 */

BENZINA_STATIC void  benzinaInitOnce(void){
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

/**
 * @brief Read ProtoBuf varint.
 * 
 * @param [in]  fp   Stream to read from.
 * @param [out] val  Encoded value read.
 * @return Zero if successful; Non-zero otherwise (e.g. due to EOF).
 */

BENZINA_STATIC int   readVarInt(int fd, uint64_t* val){
	ssize_t       r;
	unsigned char c;
	int           i=0;
	
	*val = 0;
	do{
		r = read(fd, &c, 1);
		if(r <= 0){
			return -1;
		}
		*val |= (uint64_t)(c&0x7F) << 7*i++;
	}while((c&0x80) && (i<10));
	
	return 0;
}

BENZINA_STATIC int   benzinaDatasetInitFromProtoBuf(BENZINA_DATASET*  ctx,
                                                    int               dataprotobuffd){
	uint64_t tagw, tag, val;
	
	while(readVarInt(dataprotobuffd, &tagw) == 0){
		tag = tagw >> 3;
		switch(tag){
			#define TAGCASE(tagid, target)             \
			    case tagid:                            \
			        readVarInt(dataprotobuffd, &val);  \
			        target = val;                      \
			    break;
			
			TAGCASE(33554432, ctx->info.ulWidth);
			TAGCASE(33554433, ctx->info.ulHeight);
			TAGCASE(33554434, ctx->info.ulNumDecodeSurfaces);
			TAGCASE(33554435, ctx->info.CodecType);
			TAGCASE(33554436, ctx->info.ChromaFormat);
			TAGCASE(33554438, ctx->info.bitDepthMinus8);
			TAGCASE(33554439, ctx->info.ulIntraDecodeOnly);
			TAGCASE(33554443, ctx->info.display_area.left);
			TAGCASE(33554444, ctx->info.display_area.top);
			TAGCASE(33554445, ctx->info.display_area.right);
			TAGCASE(33554446, ctx->info.display_area.bottom);
			TAGCASE(33554447, ctx->info.OutputFormat);
			TAGCASE(33554448, ctx->info.DeinterlaceMode);
			TAGCASE(33554449, ctx->info.ulTargetWidth);
			TAGCASE(33554450, ctx->info.ulTargetHeight);
			TAGCASE(33554451, ctx->info.ulNumOutputSurfaces);
			TAGCASE(33554453, ctx->info.target_rect.left);
			TAGCASE(33554454, ctx->info.target_rect.top);
			TAGCASE(33554455, ctx->info.target_rect.right);
			TAGCASE(33554456, ctx->info.target_rect.bottom);
			
			#undef TAGCASE
		}
	}
	
	return 0;
}


/* Public Function Definitions */
int          benzinaInit(void){
	int    ret = pthread_once(&benzinaInitOnceControl, benzinaInitOnce);
	return ret < 0 ? ret : benzinaInitOnceStatus;
}

int          benzinaDatasetAlloc     (BENZINA_DATASET** ctx){
	return -!(*ctx = malloc(sizeof(**ctx)));
}

int          benzinaDatasetInit      (BENZINA_DATASET*  ctx, const char* root){
	struct stat databinstat, datalengthsstat, datanvdecodestat, dataprotobufstat,
	            READMEmdstat, SHA256SUMSstat;
	int     ret=0, dirfd=-1, datalengthsfd=-1, dataprotobuffd=-1;
	size_t  bytesRead=0, bytesLeft=0, i;
	ssize_t bytesChunk=0;
	
	
	/* Wipe previous contents. */
	memset(ctx, 0, sizeof(*ctx));
	
	/* Duplicate path to root. */
	ctx->root = strdup(root);
	if(!ctx->root){
		return -1;
	}
	
	/* Test the existence of the expected files in the dataset root. */
	dirfd = open(ctx->root, O_RDONLY|O_CLOEXEC|O_DIRECTORY);
	if(dirfd                                                 < 0 ||
	   fstatat(dirfd, "data.bin",      &databinstat,      0) < 0 ||
	   fstatat(dirfd, "data.lengths",  &datalengthsstat,  0) < 0 ||
	   fstatat(dirfd, "data.nvdecode", &datanvdecodestat, 0) < 0 ||
	   fstatat(dirfd, "data.protobuf", &dataprotobufstat, 0) < 0 ||
	   fstatat(dirfd, "README.md",     &READMEmdstat,     0) < 0 ||
	   fstatat(dirfd, "SHA256SUMS",    &SHA256SUMSstat,   0) < 0){
		ret = -1;
		goto abortprobe;
	}
	
	/**
	 * The data.lengths file is a simple array of uint64_t that defines how long
	 * each record in the data.bin file concatenation is. Because it is of fixed
	 * size per entry, we use it as the canonical source of information for the
	 * length of the dataset.
	 */
	
	ctx->length  = datalengthsstat.st_size/8;
	ctx->lengths = malloc(datalengthsstat.st_size);
	ctx->offsets = malloc(datalengthsstat.st_size);
	if(!ctx->lengths || !ctx->offsets){
		ret = -2;
		goto abortprobe;
	}
	
	/* Read fully data.lengths into our private buffer. */
	datalengthsfd = openat(dirfd, "data.lengths", O_RDONLY|O_CLOEXEC);
	if(datalengthsfd < 0){
		ret = -1;
		goto abortprobe;
	}
	for(bytesLeft=datalengthsstat.st_size, bytesRead=0; bytesLeft>0;){
		bytesChunk = pread(datalengthsfd,
		                   (char*)ctx->lengths+bytesRead,
		                   bytesLeft,
		                   bytesRead);
		if (bytesChunk <= 0){
			/* We should not end up at EOF with bytesLeft > 0. */
			ret = -3;
			goto abortprobe;
		}else{
			bytesRead += bytesChunk;
			bytesLeft -= bytesChunk;
		}
	}
	
	/**
	 * Simultaneously byte-transpose to native endianness and compute cumulative
	 * offset.
	 */
	
	uint64_t cumulative=0, rawLength=0;
	for(i=0;i<ctx->length;i++){
		rawLength       = ctx->lengths[i];
		#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
		rawLength       = __builtin_bswap64(rawLength);
		ctx->lengths[i] = rawLength;
		#endif
		ctx->offsets[i] = cumulative;
		cumulative     += rawLength;
	}
	
	/* Read ProtoBuf dataset description. */
	dataprotobuffd = openat(dirfd, "data.protobuf", O_RDONLY|O_CLOEXEC);
	if(dataprotobuffd                                       < 0 ||
	   benzinaDatasetInitFromProtoBuf(ctx, dataprotobuffd) != 0){
		ret = -1;
		goto abortprobe;
	}
	
	/* Some decisions are hardcoded. */
	ctx->info.ChromaFormat        = cudaVideoChromaFormat_420;
	ctx->info.ulCreationFlags     = cudaVideoCreate_PreferCUVID;
	ctx->info.bitDepthMinus8      = 0;
	ctx->info.ulIntraDecodeOnly   = 1;
	ctx->info.OutputFormat        = cudaVideoSurfaceFormat_NV12;
	ctx->info.DeinterlaceMode     = cudaVideoDeinterlaceMode_Weave;
	ctx->info.vidLock             = NULL;
	
	
	/* Return. */
	exitprobe:
	close(dirfd);
	close(datalengthsfd);
	close(dataprotobuffd);
	return ret;
	
	
	/* Abort path. */
	abortprobe:
	free(ctx->root);
	free(ctx->lengths);
	free(ctx->offsets);
	ctx->root    = NULL;
	ctx->lengths = NULL;
	ctx->offsets = NULL;
	goto exitprobe;
}

int          benzinaDatasetNew       (BENZINA_DATASET** ctx, const char* root){
	int ret = benzinaDatasetAlloc(ctx);
	return ret ? ret : benzinaDatasetInit(*ctx, root);
}

int          benzinaDatasetFini      (BENZINA_DATASET*  ctx){
	free(ctx->root);
	free(ctx->lengths);
	free(ctx->offsets);
	memset(ctx, 0, sizeof(*ctx));
	return 0;
}

int          benzinaDatasetFree      (BENZINA_DATASET*  ctx){
	if(ctx){
		benzinaDatasetFini(ctx);
		free(ctx);
	}
	return 0;
}

int          benzinaDatasetGetLength (BENZINA_DATASET*  ctx, size_t* length){
	*length = ctx->length;
	return 0;
}

int          benzinaDatasetGetShape  (BENZINA_DATASET*  ctx, size_t* w, size_t* h){
	*w = ctx->info.ulWidth;
	*h = ctx->info.ulHeight;
	return 0;
}

int          benzinaDatasetGetElement(BENZINA_DATASET*  ctx,
                                      size_t            i,
                                      size_t*           off,
                                      size_t*           len){
	if(!ctx->offsets || !ctx->lengths || i>=ctx->length){
		*off = -1;
		*len = -1;
		return -1;
	}else{
		*off = ctx->offsets[i];
		*len = ctx->lengths[i];
		return 0;
	}
}

