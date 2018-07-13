/* Includes */
#define _GNU_SOURCE
#define __HAVE_FLOAT128 0
#include <cuda.h>
#include <dlfcn.h>
#include <linux/limits.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "cuviddec.h"
#include "nvcuvid.h"



/* Data Structure Definitions */
struct UNIVERSE;
typedef struct UNIVERSE UNIVERSE;
struct UNIVERSE{
	/* Argument Parsing */
	const char*               fileH264Path;
	int                       fileH264Fd;
	struct stat               fileH264Stat;
	const uint8_t*            fileH264Data;
	int                       fileH264LengthsFd;
	int                       fileH264NvdecodePicparamsFd;
	struct stat               fileH264LengthsStat;
	const uint64_t*           fileH264LengthsData;
	
	/* CUDA */
	CUdevice                  cuDev;
	CUcontext                 cuCtx;
	
	/* NVDECODE */
	CUvideodecoder            decoder;
	CUVIDDECODECREATEINFO     decoderInfo;
	CUvideoparser             parser;
	CUVIDPARSERPARAMS         parserParams;
	
	/* Processing */
	long                      numProcessedImages;
	uint64_t*                 byteOffsets;
};
static UNIVERSE u;



/* Static Function Prototypes */



/* Static Function Definitions */

/**
 * @brief Sequence Callback
 */

static int   sequenceCb(UNIVERSE* u, CUVIDEOFORMAT *format){
	(void)u;
	(void)format;
	
	return 1;
}

/**
 * @brief Decode Callback
 */

static int   decodeCb  (UNIVERSE* u, CUVIDPICPARAMS* picParams){
	CUresult              result = CUDA_SUCCESS;
	const void*           haystack;
	size_t                haystackLen;
	const void*           needle;
	size_t                needleLen;
	uintptr_t             offset;
	size_t                bytesToWrite;
	
	/**
	 * Notes:
	 * 
	 * In our particular situation, each image is a single-slice IDR frame.
	 * Therefore, the following is true:
	 * 
	 *   - picParams->pBitstreamData       points to the 00 00 01 start code of the
	 *                                     VCL NAL unit (Here it will always be
	 *                                     type 5 - IDR). The first three bytes are
	 *                                     therefore always 00 00 01.
	 *   - picParams->nBitstreamDataLen    is always equal to the length of
	 *                                     the above-mentioned NAL unit, from the
	 *                                     beginning of its start code to the
	 *                                     beginning of the next start code.
	 *   - picParams->nNumSlices           is always equal to 1.
	 *   - picParams->pSliceDataOffsets[0] is always equal to 0.
	 * 
	 * But of course, serializing pointers is useless. We therefore choose to:
	 * 
	 *   - Record pSliceDataOffsets as NULL
	 *   - Record pBitstreamData    as the uintptr_t byte offset from the beginning
	 *                              of the h264 image file to the beginning of the
	 *                              start code of the slice.
	 * 
	 * When this structure will be loaded, pBitstreamData will have to be relocated
	 * and pSliceDataOffsets pointed to an integer 0.
	 * 
	 * Additionally, the CurrPicIdx is dynamically determined. Its value is an
	 * incrementing counter modulo u->decoderInfo.ulNumDecodeSurfaces .
	 * 
	 * We store only the first 968 bytes of the structure, which corresponds to
	 * offsetof(CUVIDPICPARAMS, CodecSpecific.h264.fmo), because all bytes beyond
	 * are exactly 0.
	 */
	
	haystack    = u->fileH264Data         + u->byteOffsets[u->numProcessedImages];
	haystackLen = u->fileH264Stat.st_size - u->byteOffsets[u->numProcessedImages];
	needle      = picParams->pBitstreamData;
	needleLen   = picParams->nBitstreamDataLen;
	offset      = memmem(haystack, haystackLen, needle, needleLen) - haystack;
	
	picParams->CurrPicIdx        = 0;
	picParams->pBitstreamData    = (void*)offset;
	picParams->pSliceDataOffsets = NULL;
	
	bytesToWrite = offsetof(CUVIDPICPARAMS, CodecSpecific.h264.fmo);
	if(write(u->fileH264NvdecodePicparamsFd, picParams, bytesToWrite) != (ssize_t)bytesToWrite){
		printf("[%7ld] Error: Incomplete write!\n",
		       u->numProcessedImages);
		fflush(stdout);
		exit(-1);
	}
	
	u->numProcessedImages++;
	
	return result == CUDA_SUCCESS;
}

/**
 * @brief Display Callback
 */

static int   displayCb (UNIVERSE* u, CUVIDPARSERDISPINFO* dispInfo){
	(void)u;
	(void)dispInfo;
	
	return 1;
}

/**
 * @brief Init CUDA & NVCUVID
 * 
 * Reference:
 * 
 * https://devtalk.nvidia.com/default/topic/417734/problem-using-nvcuvid-library-for-video-decoding/
 */

static int   initCUDA(UNIVERSE* u){
	CUresult      result;
	unsigned long w = 256, h = 256;
	
	if(cuInit(0)                    != CUDA_SUCCESS){
		printf("Could not initialize CUDA runtime!\n");
		goto exit_cuInit;
	}
	
	if(cuDeviceGet(&u->cuDev, 0)    != CUDA_SUCCESS){
		printf("Could not retrieve handle for GPU device 0!\n");
		goto exit_cuDeviceGet;
	}
	
	if(cuCtxCreate(&u->cuCtx,
	               CU_CTX_MAP_HOST,
	               u->cuDev)        != CUDA_SUCCESS){
		printf("Failed to create context with GPU device!\n");
		goto exit_cuCtxCreate;
	}
	
	if(cuCtxSetCurrent(u->cuCtx)    != CUDA_SUCCESS){
		printf("Failed to bind context!\n");
		goto exit_cuCtxSetCurrent;
	}
	
	memset(&u->decoderInfo, 0, sizeof(u->decoderInfo));
	u->decoderInfo.ulWidth             = w;
	u->decoderInfo.ulHeight            = h;
	u->decoderInfo.ulNumDecodeSurfaces = 4;
	u->decoderInfo.CodecType           = cudaVideoCodec_H264;
	u->decoderInfo.ChromaFormat        = cudaVideoChromaFormat_420;
	u->decoderInfo.ulCreationFlags     = cudaVideoCreate_PreferCUVID;
	u->decoderInfo.bitDepthMinus8      = 0;
	u->decoderInfo.ulIntraDecodeOnly   = 1;
	u->decoderInfo.display_area.left   = 0;
	u->decoderInfo.display_area.top    = 0;
	u->decoderInfo.display_area.right  = u->decoderInfo.ulWidth;
	u->decoderInfo.display_area.bottom = u->decoderInfo.ulHeight;
	u->decoderInfo.OutputFormat        = cudaVideoSurfaceFormat_NV12;
	u->decoderInfo.DeinterlaceMode     = cudaVideoDeinterlaceMode_Weave;
	u->decoderInfo.ulTargetWidth       = u->decoderInfo.ulWidth;
	u->decoderInfo.ulTargetHeight      = u->decoderInfo.ulHeight;
	u->decoderInfo.ulNumOutputSurfaces = 4;
	u->decoderInfo.vidLock             = NULL;
	u->decoderInfo.target_rect.left    = 0;
	u->decoderInfo.target_rect.top     = 0;
	u->decoderInfo.target_rect.right   = u->decoderInfo.ulTargetWidth;
	u->decoderInfo.target_rect.bottom  = u->decoderInfo.ulTargetHeight;
	result = cuvidCreateDecoder(&u->decoder, &u->decoderInfo);
	if(result != CUDA_SUCCESS){
		printf("Failed to create NVDEC decoder (%d)!\n", (int)result);
		goto exit_cuvidCreateDecoder;
	}
	
	memset(&u->parserParams, 0, sizeof(u->parserParams));
	u->parserParams.CodecType              = u->decoderInfo.CodecType;
	u->parserParams.ulMaxNumDecodeSurfaces = u->decoderInfo.ulNumDecodeSurfaces;
	u->parserParams.ulClockRate            = 0;
	u->parserParams.ulErrorThreshold       = 0;
	u->parserParams.ulMaxDisplayDelay      = 4;
	u->parserParams.pUserData              = u;
	u->parserParams.pfnSequenceCallback    = (PFNVIDSEQUENCECALLBACK)sequenceCb;
	u->parserParams.pfnDecodePicture       = (PFNVIDDECODECALLBACK)  decodeCb;
	u->parserParams.pfnDisplayPicture      = (PFNVIDDISPLAYCALLBACK) displayCb;
	result = cuvidCreateVideoParser(&u->parser, &u->parserParams);
	if(result != CUDA_SUCCESS){
		printf("Failed to create CUVID video parser (%d)!\n", (int)result);
		goto exit_cuvidCreateVideoParser;
	}
	
	
	return 0;
	
	
	exit_cuvidCreateVideoParser:
	cuvidDestroyDecoder(u->decoder);
	exit_cuvidCreateDecoder:
	exit_cuCtxSetCurrent:
	cuCtxDestroy(u->cuCtx);
	exit_cuCtxCreate:
	exit_cuDeviceGet:
	exit_cuInit:
	return -1;
}

/**
 * @brief Init memory-map of dataset.
 */

static int   initMmap(UNIVERSE* u){
	char fileH264LengthsPath[PATH_MAX];
	char fileH264NvdecodePicparamsPath[PATH_MAX];
	snprintf(fileH264LengthsPath,           sizeof(fileH264LengthsPath),           "%s.lengths",
	         u->fileH264Path);
	snprintf(fileH264NvdecodePicparamsPath, sizeof(fileH264NvdecodePicparamsPath), "%s.nvdecode.picparams",
	         u->fileH264Path);
	
	
	if      ((u->fileH264Fd        = open(u->fileH264Path,
	                                      O_RDONLY|O_CLOEXEC))    < 0){
		printf("Cannot open() file %s ...\n", u->fileH264Path);
		exit(-1);
	}else if((u->fileH264LengthsFd = open(fileH264LengthsPath,
	                                      O_RDONLY|O_CLOEXEC))    < 0){
		printf("Cannot open() file %s ...\n", fileH264LengthsPath);
		exit(-1);
	}else if((u->fileH264NvdecodePicparamsFd = open(fileH264NvdecodePicparamsPath,
	                                                O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC, 0644)) < 0){
		printf("Cannot open() file %s ... it must not already exist.\n", fileH264NvdecodePicparamsPath);
		exit(-1);
	}else if(fstat(u->fileH264Fd,        &u->fileH264Stat)        < 0){
		printf("Cannot stat() file %s ...\n", u->fileH264Path);
		exit(-1);
	}else if(fstat(u->fileH264LengthsFd, &u->fileH264LengthsStat) < 0){
		printf("Cannot stat() file %s ...\n", fileH264LengthsPath);
		exit(-1);
	}
	
	u->fileH264Data        = (const uint8_t *)mmap(NULL,
	                                               u->fileH264Stat.st_size,
	                                               PROT_READ,
	                                               MAP_SHARED,
	                                               u->fileH264Fd,
	                                               0);
	u->fileH264LengthsData = (const uint64_t*)mmap(NULL,
	                                               u->fileH264LengthsStat.st_size,
	                                               PROT_READ,
	                                               MAP_SHARED,
	                                               u->fileH264LengthsFd,
	                                               0);
	if(u->fileH264Data        == MAP_FAILED){
		printf("Cannot mmap dataset file %s!\n", u->fileH264Path);
		goto exit_mmap;
	}
	if(u->fileH264LengthsData == MAP_FAILED){
		printf("Cannot mmap dataset file %s!\n", fileH264LengthsPath);
		goto exit_mmap;
	}
	
	if(madvise((void*)u->fileH264Data,
	           u->fileH264Stat.st_size,
	           MADV_DONTDUMP) < 0){
		printf("Cannot madvise memory range of dataset!\n");
		goto exit_madvise;
	}
	if(madvise((void*)u->fileH264LengthsData,
	           u->fileH264LengthsStat.st_size,
	           MADV_DONTDUMP) < 0){
		printf("Cannot madvise memory range of dataset!\n");
		goto exit_madvise;
	}
	
	
	printf("Processing file %s ...\n", u->fileH264Path);
	return 0;
	
	
	exit_madvise:
	exit_mmap:
	return -1;
}

/**
 * @brief Run
 */

static int   run(UNIVERSE* u){
	CUVIDSOURCEDATAPACKET packet;
	CUresult              result;
	long           i, numImages;
	const uint8_t* p;
	
	if(initMmap(u) != 0){
		printf("Failed to initialize memory map!\n");
		goto exit_initMmap;
	}
	if(initCUDA(u) != 0){
		printf("Failed to initialize CUDA!\n");
		goto exit_initCUDA;
	}
	
	
	numImages      = u->fileH264LengthsStat.st_size/8;
	u->byteOffsets = calloc(numImages, sizeof(*u->byteOffsets));
	printf("Dataset File size:         %15lu\n", u->fileH264Stat.st_size);
	printf("Dataset Lengths File size: %15lu\n", u->fileH264LengthsStat.st_size);
	printf("# of dataset images:       %15lu\n", numImages);
	fflush(stdout);
	
	
	
	for(p=u->fileH264Data, i=0;i<numImages;p+=u->fileH264LengthsData[i], i++){
		if(i>0){
			u->byteOffsets[i] = u->byteOffsets[i-1]+u->fileH264LengthsData[i-1];
		}
		
		packet.flags        = 0;
		packet.payload_size = u->fileH264LengthsData[i];
		packet.payload      = p;
		packet.timestamp    = 0;
		result = cuvidParseVideoData(u->parser, &packet);
		if(result != CUDA_SUCCESS){
			goto exit_cuvidParseVideoData;
		}
	}
	packet.flags        = CUVID_PKT_ENDOFSTREAM;
	packet.payload_size = 0;
	packet.payload      = 0;
	packet.timestamp    = 0;
	result = cuvidParseVideoData(u->parser, &packet);
	if(result != CUDA_SUCCESS){
		goto exit_cuvidParseVideoData;
	}
	printf("# of processed images:     %15ld\n", u->numProcessedImages);
	
	
	
	return 0;
	
	
	exit_cuvidParseVideoData:
	exit_initCUDA:
	exit_initMmap:
	return -1;
}



/**
 * Main
 */

int   main(int argc, char* argv[]){
	int i;
	UNIVERSE* up = &u, *u = up;
	up->fileH264Fd        = -1;
	up->fileH264LengthsFd = -1;
	
	/**
	 * Argument parsing
	 */
	
	for(i=0;i<argc; i++){
		if(i+1 >= argc){break;}
		if(strcmp(argv[i], "--path") == 0){
			u->fileH264Path = argv[++i]; continue;
		}
	}
	
	/**
	 * Argument validation
	 */
	
	if(!u->fileH264Path){
		printf("No --path PATH/TO/FILE.h264 argument provided!\n");
		exit(-1);
	}
	
	/**
	 * Run
	 */
	
	return run(u);
}

