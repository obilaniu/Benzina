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
#include <libavcodec/avcodec.h>
#include <libavutil/pixdesc.h>

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
    int                   fileH265Fd;
    struct stat           fileH265Stat;
    const uint8_t*        fileH265Data;
    
    /* FFmpeg */
    enum AVCodecID        codecID;
    AVCodec*              codec;
    AVCodecContext*       codecCtx;
    AVPacket*             packet;
    AVFrame*              frame;
    
    
    /* NVDECODE */
    cudaStream_t          stream;
    CUvideodecoder        decoder;
    CUVIDDECODECREATEINFO decoderInfo;
    CUvideoparser         parser;
    CUVIDPARSERPARAMS     parserParams;
    
    /* Processing */
    long                  numDecodedImages;
    long                  numMappedImages;
    
    /* Decoded Frame */
    uint8_t*              nvdecFramePtr;
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
     * Additionally, the CurrPicIdx is dynamically determined. Its value is an
     * incrementing counter modulo u->decoderInfo.ulNumDecodeSurfaces.
     * 
     * All bytes of the structure beyond the first 968 (which corresponds to
     * offsetof(CUVIDPICPARAMS, CodecSpecific.h264.fmo)) should be exactly 0.
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
    cudaError_t           err      = cudaSuccess;
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
    if(u->numMappedImages++ == 0){
        u->nvdecFramePtr = calloc(u->decoderInfo.ulTargetHeight*3/2,
                                  u->decoderInfo.ulTargetWidth);
        if(!u->nvdecFramePtr){
            fprintf(stdout, "Failed to allocate memory for frame!\n");
            exit(1);
        }
        err = cudaMemcpy2DAsync(u->nvdecFramePtr,
                                u->decoderInfo.ulTargetWidth,
                                (const void*)devPtr, devPitch,
                                u->decoderInfo.ulTargetWidth,
                                u->decoderInfo.ulTargetHeight*3/2,
                                cudaMemcpyDeviceToHost,
                                u->stream);
        if(err != cudaSuccess){
            fprintf(stdout, "Could not read out frame from memory (%d)!\n", (int)err);
            exit(1);
        }
    }
    
    /* Unmap frame */
    result = cuvidUnmapVideoFrame(u->decoder, devPtr);
    if(result != CUDA_SUCCESS){
        return 0;
    }
    
    /* Exit */
    return 1;
}

/**
 * @brief Init FFmpeg
 * 
 * Implies creating an H.264 decoder.
 */

static int   nvdtInitFFmpeg(UNIVERSE* u){
    int ret;
    #if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58,9,100)
    avcodec_register_all();
    #endif
    u->codecID  = AV_CODEC_ID_HEVC;
    u->codec    = avcodec_find_decoder  (u->codecID);
    u->codecCtx = avcodec_alloc_context3(u->codec);
    if(!u->codecCtx){
        fprintf(stdout, "Could not allocate FFmpeg decoder context!\n");
        return -1;
    }
    ret         = avcodec_open2         (u->codecCtx,
                                         u->codec,
                                         NULL);
    if(ret < 0){
        fprintf(stdout, "Could not open FFmpeg decoder context!\n");
        return -1;
    }
    u->packet   = av_packet_alloc();
    if(!u->packet){
        fprintf(stdout, "Failed to allocate packet object!\n");
        return -1;
    }
    u->frame    = av_frame_alloc();
    if(!u->frame){
        fprintf(stdout, "Error allocating frame!\n");
        return -1;
    }
    
    return 0;
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
    u->parserParams.CodecType              = cudaVideoCodec_HEVC;
    u->parserParams.ulMaxNumDecodeSurfaces = 20;
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
    if      ((u->fileH265Fd = open(u->args.path, O_RDONLY|O_CLOEXEC)) < 0){
        printf("Cannot open() file %s ...\n", u->args.path);
        exit(-1);
    }else if(fstat(u->fileH265Fd, &u->fileH265Stat) < 0){
        printf("Cannot stat() file %s ...\n", u->args.path);
        exit(-1);
    }
    
    u->fileH265Data = (const uint8_t *)mmap(NULL,
                                            u->fileH265Stat.st_size,
                                            PROT_READ,
                                            MAP_SHARED,
                                            u->fileH265Fd,
                                            0);
    if(u->fileH265Data == MAP_FAILED){
        printf("Cannot mmap dataset file %s!\n", u->args.path);
        goto exit_mmap;
    }
    
    if(madvise((void*)u->fileH265Data, u->fileH265Stat.st_size, MADV_DONTDUMP) < 0){
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
    CUresult              result = CUDA_SUCCESS;
    int                   ret    = 0, match = 0;
    int                   i, j;
    int                   w, h;
    
    /* Initialize */
    if(nvdtInitMmap(u) != 0){
        fprintf(stdout, "Failed to initialize memory map!\n");
        return -1;
    }
    if(nvdtInitCUDA(u) != 0){
        fprintf(stdout, "Failed to initialize CUDA!\n");
        return -1;
    }
    if(nvdtInitFFmpeg(u) != 0){
        fprintf(stdout, "Failed to initialize FFmpeg!\n");
        return -1;
    }
    
    fprintf(stdout, "Dataset File size:         %15lu\n", u->fileH265Stat.st_size);
    fflush (stdout);
    
    /* Feed entire dataset in one go to NVDECODE. */
    packet.flags        = 0;
    packet.payload_size = u->fileH265Stat.st_size;
    packet.payload      = u->fileH265Data;
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
    cudaDeviceSynchronize();
    cuvidDestroyVideoParser(u->parser);
    cuvidDestroyDecoder    (u->decoder);
    cudaDeviceSynchronize();
    
    /* Feed entire dataset in one go to FFmpeg. */
    u->packet->data = (void*)u->fileH265Data;
    u->packet->size = (int)  u->fileH265Stat.st_size;
    ret = avcodec_send_packet  (u->codecCtx, u->packet);
    if(ret != 0){
        fprintf(stdout, "Error pushing packet! (%d)\n", ret);
        return ret;
    }
    ret = avcodec_send_packet  (u->codecCtx, NULL);
    if(ret != 0){
        fprintf(stdout, "Error flushing decoder! (%d)\n", ret);
        return ret;
    }
    ret = avcodec_receive_frame(u->codecCtx, u->frame);
    if(ret != 0){
        fprintf(stdout, "Error pulling frame! (%d)\n",  ret);
        return ret;
    }
    w = u->decoderInfo.ulTargetWidth;
    h = u->decoderInfo.ulTargetHeight;
    
    /* Final Check and Printouts */
    fprintf(stdout, "# of decoded images:       %15ld\n", u->numDecodedImages);
    fprintf(stdout, "# of mapped  images:       %15ld\n", u->numMappedImages);
    for(i=0;i<h;i+=2){
        for(j=0;j<w;j+=2){
            uint8_t ny0 = u->nvdecFramePtr [0*h*w+(i+0)*w+(j+0)];
            uint8_t ny1 = u->nvdecFramePtr [0*h*w+(i+0)*w+(j+1)];
            uint8_t ny2 = u->nvdecFramePtr [0*h*w+(i+1)*w+(j+0)];
            uint8_t ny3 = u->nvdecFramePtr [0*h*w+(i+1)*w+(j+1)];
            uint8_t ncb = u->nvdecFramePtr [1*h*w+(i/2)*w+(j+0)];
            uint8_t ncr = u->nvdecFramePtr [1*h*w+(i/2)*w+(j+1)];
            uint8_t fy0 = u->frame->data[0][(i+0)*u->frame->linesize[0]+(j+0)];
            uint8_t fy1 = u->frame->data[0][(i+0)*u->frame->linesize[0]+(j+1)];
            uint8_t fy2 = u->frame->data[0][(i+1)*u->frame->linesize[0]+(j+0)];
            uint8_t fy3 = u->frame->data[0][(i+1)*u->frame->linesize[0]+(j+1)];
            uint8_t fcb = u->frame->data[1][(i/2)*u->frame->linesize[1]+(j/2)];
            uint8_t fcr = u->frame->data[2][(i/2)*u->frame->linesize[2]+(j/2)];
            if(ny0 != fy0 || ny1 != fy1 || ny2 != fy2 || ny3 != fy3){
                fprintf(stdout, "ERROR: NVDEC-decoded image luma does not bitwise match FFmpeg!\n");
                return 1;
            }
            if(ncb != fcb || ncr != fcr){
                fprintf(stdout, "ERROR: NVDEC-decoded image chroma does not bitwise match FFmpeg!\n");
                fprintf(stdout, "ERROR: CbCr NVDEC %d,%d != %d,%d FFmpeg\n", ncb, ncr, fcb, fcr);
                return 1;
            }
        }
    }
    fprintf(stdout, "SUCCESS: NVDEC-decoded image bitwise matches FFmpeg!\n");
    
    /* Exit. */
    return match ? 0 : 1;
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
    u->fileH265Fd  =   -1;
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

