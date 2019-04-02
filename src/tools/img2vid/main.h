/* Include Guard */
#ifndef SRC_TOOLS_IMG2VID_MAIN_H
#define SRC_TOOLS_IMG2VID_MAIN_H


/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif


/**
 * Includes
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavformat/avformat.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>


/**
 * Defines
 */

#define i2vmessage(fp, f, ...)                                          \
    (i2vfprintf((fp), "%s:%d: " f, __FILE__, __LINE__, ## __VA_ARGS__))
#define i2vmessageexit(code, fp, f, ...)     \
    do{                                      \
        i2vmessage(fp, f, ## __VA_ARGS__);   \
        exit(code);                          \
    }while(0)



/** Data Structure Definitions */

/**
 * @brief Arguments struct.
 */

typedef struct{
    const char* urlIn;
    const char* urlOut;
    int         dashdashseen;
    struct{unsigned l,r,t,b;} crop;
    struct{
        unsigned w,h;
        int      chroma_fmt;
        int      superscale;
    } canvas;
    unsigned    crf;
    const char* x264params;
} ARGS;

/**
 * @brief State of the program.
 * 
 * There's lot's of crap here.
 */

typedef struct{
    /* Args */
    ARGS             args;
    
    /* Status */
    int              ret;
    
    /* FFmpeg Encode/Decode */
    struct{
        AVFormatContext*   formatCtx;
        int                formatStream;
        AVCodecParameters* codecPar;
        AVCodec*           codec;
        AVCodecContext*    codecCtx;
        AVPacket*          packet;
        AVFrame*           frame;
    } in;
    struct{
        struct{int l,r,t,b;}    crop;      /* Cropping applied to source image */
        struct{int l,r,t,b;}    pad;       /* Padding  applied to destination image */
        struct{int x,y,w,h;}    embed;     /* Embedded image's x,y offset and wxh size within frame. */
        struct{float x,y;}      scale;     /* Scale factor source/destination */
        struct{float x,y;}      lumaoff;   /* Offset applied to source luma accesses */
        struct{float x,y;}      chromaoff; /* Offset applied to source chroma accesses (if subsampled) */
        enum AVColorRange       range;     /* Chroma range. */
    } tx;
    struct{
        AVCodec*           codec;
        AVCodecContext*    codecCtx;
        AVPacket*          packet;
        AVDictionary*      codecCtxOpt;
        FILE*              fp;
    } out;
    struct{
        struct{
            AVFrame*           frame;
        } in;
        struct{
            AVFrame*           frame;
        } out;
    } filt;
} UNIVERSE;


/* Static Inline Function Definitions */

/**
 * @brief String Equality (Inequality)
 * @return 0 if the strings are the same (different); !0 otherwise.
 */

static inline int   streq        (const char* s, const char* t){
    return !strcmp(s,t);
}
static inline int   strne        (const char* s, const char* t){
    return !streq(s,t);
}

/**
 * @brief Quick print of status message.
 */

static inline int   i2vfprintf   (FILE*            fp,
                                  const char*      f,
                                  ...){
    va_list ap;
    int     ret=0;
    
    fp = fp ? fp : stderr;
    va_start(ap, f);
    ret = vfprintf(fp, f, ap);
    va_end  (ap);
    return ret;
}


/* Function Prototypes */



/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

