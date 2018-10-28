/* Includes */
#define _GNU_SOURCE
#define __HAVE_FLOAT128 0
#include <cuda.h>
#include <cuda_runtime.h>
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
	/* Arguments */
	struct{
		const char* path;
		int         device;
	} args;
	
	/* Argument Parsing */
	int                   fileH264Fd;
	struct stat           fileH264Stat;
	const uint8_t*        fileH264Data;
	
	/* NVDECODE */
	cudaStream_t          stream;
	CUvideodecoder        decoder;
	CUVIDDECODECREATEINFO decoderInfo;
	CUvideoparser         parser;
	CUVIDPARSERPARAMS     parserParams;
	
	/* Processing */
	long                  numDecodedImages;
	long                  numMappedImages;
};



/* Static Function Prototypes */



/* Static Function Definitions */

/**
 * @brief Sequence Callback
 */

static int   nvdtSequenceCb(UNIVERSE* u, CUVIDEOFORMAT *format){
	CUresult result;
	
	
	/* Get our cues from the CUVIDEOFORMAT struct */
	memset(&u->decoderInfo, 0, sizeof(u->decoderInfo));
	u->decoderInfo.ulWidth             = format->coded_width;
	u->decoderInfo.ulHeight            = format->coded_height;
	u->decoderInfo.ulNumDecodeSurfaces = u->parserParams.ulMaxNumDecodeSurfaces;
	u->decoderInfo.CodecType           = format->codec;
	u->decoderInfo.ChromaFormat        = format->chroma_format;
	u->decoderInfo.ulCreationFlags     = cudaVideoCreate_PreferCUVID;
	u->decoderInfo.bitDepthMinus8      = format->bit_depth_luma_minus8;
	u->decoderInfo.ulIntraDecodeOnly   = 1;
	u->decoderInfo.ulMaxWidth          = u->decoderInfo.ulWidth;
	u->decoderInfo.ulMaxHeight         = u->decoderInfo.ulHeight;
	u->decoderInfo.display_area.left   = 0;
	u->decoderInfo.display_area.top    = 0;
	u->decoderInfo.display_area.right  = u->decoderInfo.ulWidth;
	u->decoderInfo.display_area.bottom = u->decoderInfo.ulHeight;
	u->decoderInfo.OutputFormat        = format->bit_depth_luma_minus8 > 0 ?
	                                     cudaVideoSurfaceFormat_P016 :
	                                     cudaVideoSurfaceFormat_NV12;
	u->decoderInfo.DeinterlaceMode     = cudaVideoDeinterlaceMode_Weave;
	u->decoderInfo.ulTargetWidth       = u->decoderInfo.ulWidth;
	u->decoderInfo.ulTargetHeight      = u->decoderInfo.ulHeight;
	u->decoderInfo.ulNumOutputSurfaces = 4;
	u->decoderInfo.vidLock             = NULL;
	u->decoderInfo.target_rect.left    = 0;
	u->decoderInfo.target_rect.top     = 0;
	u->decoderInfo.target_rect.right   = u->decoderInfo.ulTargetWidth;
	u->decoderInfo.target_rect.bottom  = u->decoderInfo.ulTargetHeight;
	
	
	/* Print Image Size */
	if(!u->decoder){
		fprintf(stdout, "Dataset Coded Image Size:        %4lux%4lu\n",
		        u->decoderInfo.ulWidth, u->decoderInfo.ulHeight);
		fflush (stdout);
	}
	
	
	/* Initialize decoder only once */
	if(!u->decoder){
		result = cuvidCreateDecoder(&u->decoder, &u->decoderInfo);
		if(result != CUDA_SUCCESS){
			fprintf(stdout, "Failed to create NVDEC decoder (%d)!\n", (int)result);
			fflush (stdout);
			return 0;
		}
	}else{
		//cuvidDestroyDecoder(u->decoder);
		//return 1;
	}
	
	
	/* Exit */
	return 1;
}

/**
 * @brief Decode Callback
 */

static int   nvdtDecodeCb  (UNIVERSE* u, CUVIDPICPARAMS* picParams){
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
	
	if(cuvidDecodePicture(u->decoder, picParams) != CUDA_SUCCESS){
		fprintf(stdout, "Could not decode picture!\n");
		return -1;
	}
	u->numDecodedImages++;
	
	return 1;
}

/**
 * @brief Display Callback
 */

static int   nvdtDisplayCb (UNIVERSE* u, CUVIDPARSERDISPINFO* dispInfo){
	CUresult              result   = CUDA_SUCCESS;
	CUVIDPROCPARAMS       VPP      = {0};
	unsigned long long    devPtr   = 0;
	unsigned              devPitch = 0;
	
	
	/* Map frame */
	VPP.progressive_frame = dispInfo->progressive_frame;
	VPP.second_field      = dispInfo->repeat_first_field + 1;
	VPP.top_field_first   = dispInfo->top_field_first;
	VPP.unpaired_field    = dispInfo->repeat_first_field < 0;
	VPP.output_stream     = u->stream;
	result = cuvidMapVideoFrame(u->decoder, dispInfo->picture_index,
	                            &devPtr, &devPitch, &VPP);
	if(result != CUDA_SUCCESS){
		return 0;
	}
	
	/* Increment count of mapped frames */
	u->numMappedImages++;
	
	/* Unmap frame */
	result = cuvidUnmapVideoFrame(u->decoder, devPtr);
	if(result != CUDA_SUCCESS){
		return 0;
	}
	
	/* Exit */
	return 1;
}

/**
 * @brief Init CUDA & NVCUVID
 * 
 * Reference:
 * 
 * https://devtalk.nvidia.com/default/topic/417734/problem-using-nvcuvid-library-for-video-decoding/
 */

static int   nvdtInitCUDA(UNIVERSE* u){
	CUresult  result;
	
#if 1
	/* CUDA Runtime API */
	if(cudaSetDevice(u->args.device) != cudaSuccess){
		fprintf(stdout, "Could not set GPU device %d!\n", u->args.device);
		return -1;
	}
	cudaDeviceSynchronize();
#else
	CUdevice  cuDev;
	CUcontext cuCtx;
	
	/* CUDA Driver API */
	if(cuInit(0)                 != CUDA_SUCCESS){
		printf("Could not initialize CUDA runtime!\n");
		return -1;
	}
	
	if(cuDeviceGet(&cuDev, 0)    != CUDA_SUCCESS){
		printf("Could not retrieve handle for GPU device 0!\n");
		return -1;
	}
	
	if(cuDevicePrimaryCtxRetain(&cuCtx, cuDev) != CUDA_SUCCESS){
		printf("Failed to retain primary context!\n");
		return -1;
	}
	
	if(cuCtxSetCurrent(cuCtx)    != CUDA_SUCCESS){
		printf("Failed to bind context!\n");
		return -1;
	}
#endif
	
	if(cudaStreamCreateWithFlags(&u->stream, cudaStreamNonBlocking) != cudaSuccess){
		printf("Failed to create CUDA stream!\n");
		return -1;
	}
	
	memset(&u->parserParams, 0, sizeof(u->parserParams));
	u->parserParams.CodecType              = cudaVideoCodec_H264;
	u->parserParams.ulMaxNumDecodeSurfaces = 4;
	u->parserParams.ulClockRate            = 0;
	u->parserParams.ulErrorThreshold       = 0;
	u->parserParams.ulMaxDisplayDelay      = 4;
	u->parserParams.pUserData              = u;
	u->parserParams.pfnSequenceCallback    = (PFNVIDSEQUENCECALLBACK)nvdtSequenceCb;
	u->parserParams.pfnDecodePicture       = (PFNVIDDECODECALLBACK)  nvdtDecodeCb;
	u->parserParams.pfnDisplayPicture      = (PFNVIDDISPLAYCALLBACK) nvdtDisplayCb;
	result = cuvidCreateVideoParser(&u->parser, &u->parserParams);
	if(result != CUDA_SUCCESS){
		printf("Failed to create CUVID video parser (%d)!\n", (int)result);
		return -1;
	}
	
	return 0;
}

/**
 * @brief Init memory-map of dataset.
 */

static int   nvdtInitMmap(UNIVERSE* u){
	if      ((u->fileH264Fd = open(u->args.path, O_RDONLY|O_CLOEXEC)) < 0){
		printf("Cannot open() file %s ...\n", u->args.path);
		exit(-1);
	}else if(fstat(u->fileH264Fd, &u->fileH264Stat) < 0){
		printf("Cannot stat() file %s ...\n", u->args.path);
		exit(-1);
	}
	
	u->fileH264Data = (const uint8_t *)mmap(NULL,
	                                        u->fileH264Stat.st_size,
	                                        PROT_READ,
	                                        MAP_SHARED,
	                                        u->fileH264Fd,
	                                        0);
	if(u->fileH264Data == MAP_FAILED){
		printf("Cannot mmap dataset file %s!\n", u->args.path);
		goto exit_mmap;
	}
	
	if(madvise((void*)u->fileH264Data, u->fileH264Stat.st_size, MADV_DONTDUMP) < 0){
		printf("Cannot madvise memory range of dataset!\n");
		goto exit_madvise;
	}
	
	printf("Processing file %s ...\n", u->args.path);
	return 0;
	
	
	exit_madvise:
	exit_mmap:
	return -1;
}

/**
 * @brief Run
 */

static int   nvdtRun(UNIVERSE* u){
	CUVIDSOURCEDATAPACKET packet;
	CUresult              result;
	
	/* Initialize */
	if(nvdtInitMmap(u) != 0){
		fprintf(stdout, "Failed to initialize memory map!\n");
		return -1;
	}
	if(nvdtInitCUDA(u) != 0){
		fprintf(stdout, "Failed to initialize CUDA!\n");
		return -1;
	}
	
	fprintf(stdout, "Dataset File size:         %15lu\n", u->fileH264Stat.st_size);
	fflush (stdout);
	
	/* Feed entire dataset in one go. */
	packet.flags        = 0;
	packet.payload_size = u->fileH264Stat.st_size;
	packet.payload      = u->fileH264Data;
	packet.timestamp    = 0;
	result = cuvidParseVideoData(u->parser, &packet);
	if(result != CUDA_SUCCESS){
		return -1;
	}
	packet.flags        = CUVID_PKT_ENDOFSTREAM;
	packet.payload_size = 0;
	packet.payload      = NULL;
	packet.timestamp    = 0;
	result = cuvidParseVideoData(u->parser, &packet);
	if(result != CUDA_SUCCESS){
		return -1;
	}
	
	/* Exit. */
	printf("# of decoded images:       %15ld\n", u->numDecodedImages);
	printf("# of mapped  images:       %15ld\n", u->numMappedImages);
	return 0;
}



/**
 * Main
 */

int   main(int argc, char* argv[]){
	static UNIVERSE U = {0}, *u = &U;
	int i;
	
	/**
	 * Argument parsing
	 */
	
	u->args.device =    0;
	u->args.path   = NULL;
	u->fileH264Fd  =   -1;
	for(i=0;i<argc; i++){
		if(strcmp(argv[i], "--device") == 0){
			u->args.device = strtol(argv[++i]?argv[i]:"0", NULL, 0);
		}else{
			u->args.path   = argv[i];
		}
	}
	
	/**
	 * Argument validation
	 */
	
	if(!u->args.path){
		printf("No PATH/TO/FILE.h264 argument provided!\n");
		exit(-1);
	}
	
	/**
	 * Run
	 */
	
	return nvdtRun(u);
}

