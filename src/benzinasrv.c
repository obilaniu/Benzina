/* Includes */
#define _GNU_SOURCE
#include <cuda.h>
#include <dlfcn.h>
#include <dynlink_cuviddec.h>
#include <dynlink_nvcuvid.h>
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
#include <zmq.h>



/* Data Structure Definitions */
struct UNIVERSE;
struct THREADUNIVERSE;
typedef struct UNIVERSE       UNIVERSE;
typedef struct THREADUNIVERSE THREADUNIVERSE;

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
	
	/* NVDECODE */
	void*                     libnvcuvid;
	#define USE_SYMBOL(fn) t ## fn* fn
	USE_SYMBOL               (cuvidCreateVideoParser);
	USE_SYMBOL               (cuvidParseVideoData);
	USE_SYMBOL               (cuvidDestroyVideoParser);
	USE_SYMBOL               (cuvidGetDecoderCaps);
	USE_SYMBOL               (cuvidCreateDecoder);
	USE_SYMBOL               (cuvidDestroyDecoder);
	USE_SYMBOL               (cuvidDecodePicture);
	USE_SYMBOL               (cuvidMapVideoFrame64);
	USE_SYMBOL               (cuvidUnmapVideoFrame64);
	#undef USE_SYMBOL
	
	/* Threads */
	THREADUNIVERSE*           thrd;
};

struct THREADUNIVERSE{
	/* Universe */
	UNIVERSE*                 u;
	
	/* ZMQ */
	void*                     sock;
	
	/* CUDA */
	CUdevice                  cuDev;
	CUcontext                 cuCtx;
	
	/* NVDECODE */
	CUvideodecoder            decoder;
	CUVIDDECODECREATEINFO     decoderInfo;
	CUvideoparser             parser;
	CUVIDPARSERPARAMS         parserParams;
};

static UNIVERSE u;



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


/* Static Function Prototypes */



/* Static Function Definitions */



/**
 * Main
 */

int   main(int argc, char* argv[]){
	(void)argc;
	(void)argv;
	(void)u;
	
	return 0;
}

