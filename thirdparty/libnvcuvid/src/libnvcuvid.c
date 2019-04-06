#include "nvcuvid.h"
#include "cuviddec.h"


#ifdef __cplusplus
extern "C" {
#endif


/* nvcuvid.h */
#ifndef __APPLE__
CUresult       CUDAAPI cuvidCreateVideoSource   (CUvideosource* pObj, const char*    pszFileName,  CUVIDSOURCEPARAMS* pParams){}
CUresult       CUDAAPI cuvidCreateVideoSourceW  (CUvideosource* pObj, const wchar_t* pwszFileName, CUVIDSOURCEPARAMS* pParams){}
CUresult       CUDAAPI cuvidDestroyVideoSource  (CUvideosource  obj){}
CUresult       CUDAAPI cuvidSetVideoSourceState (CUvideosource  obj, cudaVideoState state){}
cudaVideoState CUDAAPI cuvidGetVideoSourceState (CUvideosource  obj){}
CUresult       CUDAAPI cuvidGetSourceVideoFormat(CUvideosource  obj, CUVIDEOFORMAT* pvidfmt, unsigned int flags){}
CUresult       CUDAAPI cuvidGetSourceAudioFormat(CUvideosource  obj, CUAUDIOFORMAT* paudfmt, unsigned int flags){}
#endif

CUresult       CUDAAPI cuvidCreateVideoParser   (CUvideoparser* pObj, CUVIDPARSERPARAMS* pParams){}
CUresult       CUDAAPI cuvidParseVideoData      (CUvideoparser  obj,  CUVIDSOURCEDATAPACKET* pPacket){}
CUresult       CUDAAPI cuvidDestroyVideoParser  (CUvideoparser  obj){}


/* cuviddec.h */
CUresult       CUDAAPI cuvidGetDecoderCaps      (CUVIDDECODECAPS* pdc){}
CUresult       CUDAAPI cuvidCreateDecoder       (CUvideodecoder* phDecoder, CUVIDDECODECREATEINFO* pdci){}
CUresult       CUDAAPI cuvidDestroyDecoder      (CUvideodecoder  hDecoder){}
CUresult       CUDAAPI cuvidDecodePicture       (CUvideodecoder  hDecoder, CUVIDPICPARAMS* pPicParams){}
CUresult       CUDAAPI cuvidGetDecodeStatus     (CUvideodecoder  hDecoder, int nPicIdx, CUVIDGETDECODESTATUS* pDecodeStatus){}
CUresult       CUDAAPI cuvidReconfigureDecoder  (CUvideodecoder  hDecoder, CUVIDRECONFIGUREDECODERINFO* pDecReconfigParams){}

#if !defined(__CUVID_DEVPTR64) || defined(__CUVID_INTERNAL)
CUresult       CUDAAPI cuvidMapVideoFrame       (CUvideodecoder  hDecoder, int nPicIdx, unsigned int* pDevPtr, unsigned int* pPitch, CUVIDPROCPARAMS* pVPP){}
CUresult       CUDAAPI cuvidUnmapVideoFrame     (CUvideodecoder  hDecoder, unsigned int DevPtr){}
#endif

#if defined(_WIN64) || defined(__LP64__) || defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
CUresult       CUDAAPI cuvidMapVideoFrame64     (CUvideodecoder  hDecoder, int nPicIdx, unsigned long long* pDevPtr, unsigned int* pPitch, CUVIDPROCPARAMS* pVPP){}
CUresult       CUDAAPI cuvidUnmapVideoFrame64   (CUvideodecoder  hDecoder, unsigned long long DevPtr){}
#endif

CUresult       CUDAAPI cuvidCtxLockCreate       (CUvideoctxlock* pLock, CUcontext ctx){}
CUresult       CUDAAPI cuvidCtxLockDestroy      (CUvideoctxlock  lck){}
CUresult       CUDAAPI cuvidCtxLock             (CUvideoctxlock  lck, unsigned int reserved_flags){}
CUresult       CUDAAPI cuvidCtxUnlock           (CUvideoctxlock  lck, unsigned int reserved_flags){}


#ifdef __cplusplus
}
#endif


