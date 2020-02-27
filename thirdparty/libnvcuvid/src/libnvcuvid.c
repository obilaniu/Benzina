#include "nvcuvid.h"
#include "cuviddec.h"


#ifdef __cplusplus
extern "C" {
#endif


/* nvcuvid.h */
#ifndef __APPLE__
CUresult       CUDAAPI cuvidCreateVideoSource   (CUvideosource* pObj, const char*    pszFileName,  CUVIDSOURCEPARAMS* pParams){
    return 0xDEADBEEF;
}
CUresult       CUDAAPI cuvidCreateVideoSourceW  (CUvideosource* pObj, const wchar_t* pwszFileName, CUVIDSOURCEPARAMS* pParams){
    return 0xDEADBEEF;
}
CUresult       CUDAAPI cuvidDestroyVideoSource  (CUvideosource  obj){
    return 0xDEADBEEF;
}
CUresult       CUDAAPI cuvidSetVideoSourceState (CUvideosource  obj, cudaVideoState state){
    return 0xDEADBEEF;
}
cudaVideoState CUDAAPI cuvidGetVideoSourceState (CUvideosource  obj){
    return 0xDEADBEEF;
}
CUresult       CUDAAPI cuvidGetSourceVideoFormat(CUvideosource  obj, CUVIDEOFORMAT* pvidfmt, unsigned int flags){
    return 0xDEADBEEF;
}
CUresult       CUDAAPI cuvidGetSourceAudioFormat(CUvideosource  obj, CUAUDIOFORMAT* paudfmt, unsigned int flags){
    return 0xDEADBEEF;
}
#endif

CUresult       CUDAAPI cuvidCreateVideoParser   (CUvideoparser* pObj, CUVIDPARSERPARAMS* pParams){
    return 0xDEADBEEF;
}
CUresult       CUDAAPI cuvidParseVideoData      (CUvideoparser  obj,  CUVIDSOURCEDATAPACKET* pPacket){
    return 0xDEADBEEF;
}
CUresult       CUDAAPI cuvidDestroyVideoParser  (CUvideoparser  obj){
    return 0xDEADBEEF;
}


/* cuviddec.h */
CUresult       CUDAAPI cuvidGetDecoderCaps      (CUVIDDECODECAPS* pdc){
    return 0xDEADBEEF;
}
CUresult       CUDAAPI cuvidCreateDecoder       (CUvideodecoder* phDecoder, CUVIDDECODECREATEINFO* pdci){
    return 0xDEADBEEF;
}
CUresult       CUDAAPI cuvidDestroyDecoder      (CUvideodecoder  hDecoder){
    return 0xDEADBEEF;
}
CUresult       CUDAAPI cuvidDecodePicture       (CUvideodecoder  hDecoder, CUVIDPICPARAMS* pPicParams){
    return 0xDEADBEEF;
}
CUresult       CUDAAPI cuvidGetDecodeStatus     (CUvideodecoder  hDecoder, int nPicIdx, CUVIDGETDECODESTATUS* pDecodeStatus){
    return 0xDEADBEEF;
}
CUresult       CUDAAPI cuvidReconfigureDecoder  (CUvideodecoder  hDecoder, CUVIDRECONFIGUREDECODERINFO* pDecReconfigParams){
    return 0xDEADBEEF;
}

#if !defined(__CUVID_DEVPTR64) || defined(__CUVID_INTERNAL)
CUresult       CUDAAPI cuvidMapVideoFrame       (CUvideodecoder  hDecoder, int nPicIdx, unsigned int* pDevPtr, unsigned int* pPitch, CUVIDPROCPARAMS* pVPP){
    return 0xDEADBEEF;
}
CUresult       CUDAAPI cuvidUnmapVideoFrame     (CUvideodecoder  hDecoder, unsigned int DevPtr){
    return 0xDEADBEEF;
}
#endif

#if defined(_WIN64) || defined(__LP64__) || defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
CUresult       CUDAAPI cuvidMapVideoFrame64     (CUvideodecoder  hDecoder, int nPicIdx, unsigned long long* pDevPtr, unsigned int* pPitch, CUVIDPROCPARAMS* pVPP){
    return 0xDEADBEEF;
}
CUresult       CUDAAPI cuvidUnmapVideoFrame64   (CUvideodecoder  hDecoder, unsigned long long DevPtr){
    return 0xDEADBEEF;
}
#endif

CUresult       CUDAAPI cuvidCtxLockCreate       (CUvideoctxlock* pLock, CUcontext ctx){
    return 0xDEADBEEF;
}
CUresult       CUDAAPI cuvidCtxLockDestroy      (CUvideoctxlock  lck){
    return 0xDEADBEEF;
}
CUresult       CUDAAPI cuvidCtxLock             (CUvideoctxlock  lck, unsigned int reserved_flags){
    return 0xDEADBEEF;
}
CUresult       CUDAAPI cuvidCtxUnlock           (CUvideoctxlock  lck, unsigned int reserved_flags){
    return 0xDEADBEEF;
}


#ifdef __cplusplus
}
#endif


