/* Includes */
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavformat/avformat.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#if _OPENMP
#include <omp.h>
#endif

#include "benzina/benzina.h"

#include "geometry.h"



/**
 * Defines
 */

#define i2hmessage(fp, f, ...)                                          \
    (i2hfprintf((fp), "%s:%d: " f, __FILE__, __LINE__, ## __VA_ARGS__))
#define i2hmessageexit(code, fp, f, ...)     \
    do{                                      \
        i2hmessage(fp, f, ## __VA_ARGS__);   \
        exit(code);                          \
    }while(0)



/* Data Structure Forward Declarations and Typedefs */
typedef struct ITEM     ITEM;
typedef struct IREF     IREF;
typedef struct UNIVERSE UNIVERSE;



/**
 * Data Structure Definitions
 */

/**
 * @brief Item to be added.
 */

struct ITEM{
    uint32_t    id;
    const char* path;
    const char* name;
    char        type[4];
    const char* mime;
    unsigned    primary        : 1;
    unsigned    hidden         : 1;
    unsigned    want_thumbnail : 1;
    uint32_t    width, height;
    AVFrame*    frame;
    AVPacket*   packet;
    IREF*       refs;
    ITEM*       thumbnail;
    ITEM*       grid;
    ITEM*       next;
};

/**
 * @brief Item reference.
 */

struct IREF{
    char      type[4];
    uint32_t  num_references;
    uint32_t* to_item_id;
    IREF*     next;
};

/**
 * @brief State of the program.
 * 
 * There's lots of crap here.
 */

struct UNIVERSE{
    /* Args */
    struct{
        char*       urlIn;
        const char* x264params;
    } args;
    
    /* HEIF */
    ITEM             item;
    ITEM*            items;
    
    /* Status */
    int              ret;
    
    /* FFmpeg Encode/Decode */
    struct{
        AVFormatContext*   formatCtx;
        int                formatStream;
        AVCodecParameters* codecPar;
        AVCodec*           codec;
        AVCodecContext*    codecCtx;
    } in;
    struct{
        BENZINA_GEOM       geom;
    } tx;
    struct{
        AVCodec*           codec;
        AVCodecContext*    codecCtx;
        AVPacket*          packet;
        uint32_t           tile_width, tile_height;
        unsigned           crf;
        unsigned           chroma_format;
        const char*        x264_params;
        const char*        x265_params;
        const char*        url;
    } out;
};



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

static inline int   i2hfprintf   (FILE*            fp,
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

/**
 * @brief Miscellaneous string testing and utility functions.
 */

static int strstartswith(const char* s, const char* t){
    size_t ls = strlen(s), lt = strlen(t);
    return ls<lt ? 0 : !memcmp(s, t, lt);
}
static int strendswith(const char* s, const char* e){
    size_t ls = strlen(s), le = strlen(e);
    return ls<le ? 0 : !strcmp(s+ls-le, e);
}
static int i2h_str2chromafmt(const char* s){
    if      (strcmp(s, "yuv400") == 0){return BENZINA_CHROMAFMT_YUV400;
    }else if(strcmp(s, "yuv420") == 0){return BENZINA_CHROMAFMT_YUV420;
    }else if(strcmp(s, "yuv422") == 0){return BENZINA_CHROMAFMT_YUV422;
    }else if(strcmp(s, "yuv444") == 0){return BENZINA_CHROMAFMT_YUV444;
    }else{
        return -100;
    }
}
static int av2benzinachromaloc(enum AVChromaLocation chroma_loc){
    if(chroma_loc == AVCHROMA_LOC_UNSPECIFIED)
        return BENZINA_CHROMALOC_CENTER;
    return (int)chroma_loc - 1;
}
static enum AVChromaLocation benzina2avchromaloc(int chroma_loc){
    return (enum AVChromaLocation)(chroma_loc + 1);
}

/**
 * @brief Fixup several possible problems in frame metadata.
 * 
 * Some pixel formats are deprecated in favour of a combination with other
 * attributes (such as color_range), so we fix them up.
 * 
 * The AV_CODEC_ID_MJPEG decoder still reports the pixel format using the
 * obsolete pixel-format codes AV_PIX_FMT_YUVJxxxP. This causes problems
 * while filtering with graph filters not aware of the deprecated pixel
 * formats.
 * 
 * Fix this up by replacing the deprecated pixel formats with the
 * corresponding AV_PIX_FMT_YUVxxxP code, and setting the color_range to
 * full-range: AVCOL_RANGE_JPEG.
 * 
 * The color primary, transfer function and colorspace data may all be
 * unspecified. Fill in with plausible data.
 * 
 * @param [in]  f   The AVFrame whose attributes are to be fixed up.
 * @return 0.
 */

static int i2h_fixup_frame(AVFrame* f){
    switch(f->format){
        #define FIXUPFORMAT(p)                         \
            case AV_PIX_FMT_YUVJ ## p:                 \
                f->format      = AV_PIX_FMT_YUV ## p;  \
                f->color_range = AVCOL_RANGE_JPEG;     \
            break
        FIXUPFORMAT(411P);
        FIXUPFORMAT(420P);
        FIXUPFORMAT(422P);
        FIXUPFORMAT(440P);
        FIXUPFORMAT(444P);
        default: break;
        #undef  FIXUPFORMAT
    }
    switch(f->color_primaries){
        case AVCOL_PRI_RESERVED0:
        case AVCOL_PRI_RESERVED:
        case AVCOL_PRI_UNSPECIFIED:
            f->color_primaries = AVCOL_PRI_BT709;
        break;
        default: break;
    }
    switch(f->color_trc){
        case AVCOL_TRC_RESERVED0:
        case AVCOL_TRC_RESERVED:
        case AVCOL_TRC_UNSPECIFIED:
            f->color_trc = AVCOL_TRC_IEC61966_2_1;
        break;
        default: break;
    }
    switch(f->colorspace){
        case AVCOL_SPC_RESERVED:
        case AVCOL_TRC_UNSPECIFIED:
            f->colorspace = AVCOL_SPC_BT470BG;
        break;
        default: break;
    }
    return 0;
}

/**
 * @brief Pick a canonical, losslessly compatible pixel format for given pixel
 *        format.
 * 
 * By losslessly converting to one of a few canonical formats, we minimize the
 * number of pixel formats to support.
 * 
 * @return The canonical destination pixel format for a given source format, or
 *         AV_PIX_FMT_NONE if not supported.
 */

static enum AVPixelFormat i2h_pick_pix_fmt(enum AVPixelFormat src){
    switch(src){
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUVA420P:
        case AV_PIX_FMT_GRAY8:
        case AV_PIX_FMT_YA8:
        case AV_PIX_FMT_MONOBLACK:
        case AV_PIX_FMT_MONOWHITE:
        case AV_PIX_FMT_NV12:
        case AV_PIX_FMT_NV21:
            return AV_PIX_FMT_YUV420P;
        case AV_PIX_FMT_YUV444P:
        case AV_PIX_FMT_YUVA444P:
            return AV_PIX_FMT_YUV444P;
        case AV_PIX_FMT_GBRP:
        case AV_PIX_FMT_0BGR:
        case AV_PIX_FMT_0RGB:
        case AV_PIX_FMT_ABGR:
        case AV_PIX_FMT_ARGB:
        case AV_PIX_FMT_BGR0:
        case AV_PIX_FMT_BGR24:
        case AV_PIX_FMT_BGRA:
        case AV_PIX_FMT_GBRAP:
        case AV_PIX_FMT_PAL8:
        case AV_PIX_FMT_RGB0:
        case AV_PIX_FMT_RGB24:
        case AV_PIX_FMT_RGBA:
            return AV_PIX_FMT_GBRP;
        case AV_PIX_FMT_YUV420P16LE:
        case AV_PIX_FMT_YUV420P16BE:
        case AV_PIX_FMT_YUVA420P16LE:
        case AV_PIX_FMT_YUVA420P16BE:
        case AV_PIX_FMT_YA16BE:
        case AV_PIX_FMT_YA16LE:
        case AV_PIX_FMT_GRAY16BE:
        case AV_PIX_FMT_GRAY16LE:
        case AV_PIX_FMT_P016BE:
        case AV_PIX_FMT_P016LE:
            return AV_PIX_FMT_YUV420P16LE;
        case AV_PIX_FMT_YUV444P16LE:
        case AV_PIX_FMT_YUV444P16BE:
        case AV_PIX_FMT_YUVA444P16LE:
        case AV_PIX_FMT_YUVA444P16BE:
        case AV_PIX_FMT_AYUV64BE:
        case AV_PIX_FMT_AYUV64LE:
            return AV_PIX_FMT_YUV444P16LE;
        case AV_PIX_FMT_GBRP16LE:
        case AV_PIX_FMT_GBRP16BE:
        case AV_PIX_FMT_GBRAP16LE:
        case AV_PIX_FMT_GBRAP16BE:
        case AV_PIX_FMT_BGRA64LE:
        case AV_PIX_FMT_BGRA64BE:
        case AV_PIX_FMT_RGBA64LE:
        case AV_PIX_FMT_RGBA64BE:
        case AV_PIX_FMT_BGR48BE:
        case AV_PIX_FMT_BGR48LE:
        case AV_PIX_FMT_RGB48BE:
        case AV_PIX_FMT_RGB48LE:
            return AV_PIX_FMT_GBRP16LE;
        case AV_PIX_FMT_YUV410P:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_YUV411P:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_UYYVYY411:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_YUV420P9LE:
        case AV_PIX_FMT_YUV420P9BE:
        case AV_PIX_FMT_YUVA420P9LE:
        case AV_PIX_FMT_YUVA420P9BE:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_YUV420P10LE:
        case AV_PIX_FMT_YUV420P10BE:
        case AV_PIX_FMT_YUVA420P10LE:
        case AV_PIX_FMT_YUVA420P10BE:
        case AV_PIX_FMT_P010LE:
        case AV_PIX_FMT_P010BE:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_YUV420P12LE:
        case AV_PIX_FMT_YUV420P12BE:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_YUV420P14LE:
        case AV_PIX_FMT_YUV420P14BE:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_GBRP9LE:
        case AV_PIX_FMT_GBRP9BE:
        case AV_PIX_FMT_GRAY9LE:
        case AV_PIX_FMT_GRAY9BE:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_GBRP10LE:
        case AV_PIX_FMT_GBRP10BE:
        case AV_PIX_FMT_GBRAP10LE:
        case AV_PIX_FMT_GBRAP10BE:
        case AV_PIX_FMT_GRAY10LE:
        case AV_PIX_FMT_GRAY10BE:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_GBRP12LE:
        case AV_PIX_FMT_GBRP12BE:
        case AV_PIX_FMT_GBRAP12LE:
        case AV_PIX_FMT_GBRAP12BE:
        case AV_PIX_FMT_GRAY12LE:
        case AV_PIX_FMT_GRAY12BE:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_GBRP14LE:
        case AV_PIX_FMT_GBRP14BE:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_YUV422P:
        case AV_PIX_FMT_YUVA422P:
        case AV_PIX_FMT_UYVY422:
        case AV_PIX_FMT_YVYU422:
        case AV_PIX_FMT_YUYV422:
        case AV_PIX_FMT_NV16:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_YUV422P9LE:
        case AV_PIX_FMT_YUV422P9BE:
        case AV_PIX_FMT_YUVA422P9LE:
        case AV_PIX_FMT_YUVA422P9BE:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_YUV422P10LE:
        case AV_PIX_FMT_YUV422P10BE:
        case AV_PIX_FMT_YUVA422P10LE:
        case AV_PIX_FMT_YUVA422P10BE:
        case AV_PIX_FMT_NV20LE:
        case AV_PIX_FMT_NV20BE:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_YUV422P12LE:
        case AV_PIX_FMT_YUV422P12BE:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_YUV422P14LE:
        case AV_PIX_FMT_YUV422P14BE:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_YUV422P16LE:
        case AV_PIX_FMT_YUV422P16BE:
        case AV_PIX_FMT_YUVA422P16LE:
        case AV_PIX_FMT_YUVA422P16BE:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_YUV440P:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_YUV440P10LE:
        case AV_PIX_FMT_YUV440P10BE:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_YUV440P12LE:
        case AV_PIX_FMT_YUV440P12BE:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_YUV444P9LE:
        case AV_PIX_FMT_YUV444P9BE:
        case AV_PIX_FMT_YUVA444P9LE:
        case AV_PIX_FMT_YUVA444P9BE:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_YUV444P10LE:
        case AV_PIX_FMT_YUV444P10BE:
        case AV_PIX_FMT_YUVA444P10LE:
        case AV_PIX_FMT_YUVA444P10BE:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_YUV444P12LE:
        case AV_PIX_FMT_YUV444P12BE:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_YUV444P14LE:
        case AV_PIX_FMT_YUV444P14BE:
            return AV_PIX_FMT_NONE;
        case AV_PIX_FMT_GBRPF32LE:
        case AV_PIX_FMT_GBRPF32BE:
        case AV_PIX_FMT_GBRAPF32LE:
        case AV_PIX_FMT_GBRAPF32BE:
            return AV_PIX_FMT_NONE;
        default:
            return AV_PIX_FMT_NONE;
#if FF_API_VAAPI
        case AV_PIX_FMT_VAAPI_MOCO:
        case AV_PIX_FMT_VAAPI_IDCT:
#endif
        case AV_PIX_FMT_CUDA:
        case AV_PIX_FMT_D3D11:
        case AV_PIX_FMT_D3D11VA_VLD:
        case AV_PIX_FMT_DRM_PRIME:
        case AV_PIX_FMT_MEDIACODEC:
        case AV_PIX_FMT_MMAL:
        case AV_PIX_FMT_OPENCL:
        case AV_PIX_FMT_QSV:
        case AV_PIX_FMT_VAAPI:
        case AV_PIX_FMT_VDPAU:
        case AV_PIX_FMT_VIDEOTOOLBOX:
        case AV_PIX_FMT_XVMC:
            return AV_PIX_FMT_NONE;
    }
}

/**
 * @brief Convert input frame to output with canonical pixel format.
 * 
 * @param [out]  out  Output frame, with canonical pixel format.
 * @param [in]   in   Input frame.
 * @return 0 if successful; !0 otherwise.
 */

static int i2h_canonicalize_pixfmt(AVFrame* out, AVFrame* in){
    enum AVPixelFormat canonical     = i2h_pick_pix_fmt(in->format);
    AVFilterGraph*     graph         = NULL;
    AVFilterInOut*     inputs        = NULL;
    AVFilterInOut*     outputs       = NULL;
    AVFilterContext*   buffersrcctx  = NULL;
    AVFilterContext*   buffersinkctx = NULL;
    const AVFilter*    buffersrc     = NULL;
    const AVFilter*    buffersink    = NULL;
    char               graphstr     [512] = {0};
    char               buffersrcargs[512] = {0};
    int                ret           = -EINVAL;
    
    
    /**
     * Fast Check
     */
    
    if(canonical == AV_PIX_FMT_NONE){
        i2hmessage(stderr, "Pixel format %s unsupported!\n",
                   av_get_pix_fmt_name(in->format));
        return -EINVAL;
    }
    
    
    /**
     * Graph Init
     */
    
    
    buffersrc  = avfilter_get_by_name("buffer");
    buffersink = avfilter_get_by_name("buffersink");
    if(!buffersrc || !buffersink){
        i2hmessage(stderr, "Failed to find buffer or buffersink filter objects!\n");
        return -EINVAL;
    }
    graph = avfilter_graph_alloc();
    if(!graph){
        i2hmessage(stderr, "Failed to allocate graph object!\n");
        return -ENOMEM;
    }
    avfilter_graph_set_auto_convert(graph, AVFILTER_AUTO_CONVERT_NONE);
    outputs = avfilter_inout_alloc();
    inputs  = avfilter_inout_alloc();
    if(!outputs   || !inputs){
        i2hmessage(stderr, "Failed to allocate inout objects!\n");
        avfilter_graph_free(&graph);
        avfilter_inout_free(&outputs);
        avfilter_inout_free(&inputs);
        return -ENOMEM;
    }
    inputs ->name       = av_strdup("out");
    inputs ->filter_ctx = NULL;
    inputs ->pad_idx    = 0;
    inputs ->next       = NULL;
    outputs->name       = av_strdup("in");
    outputs->filter_ctx = NULL;
    outputs->pad_idx    = 0;
    outputs->next       = NULL;
    if(!inputs->name || !outputs->name){
        i2hmessage(stderr, "Failed to allocate name for filter input/output!\n");
        ret = -ENOMEM;
        goto fail;
    }
    
    
    /**
     * The arguments for the buffersrc filter, buffersink filter, and the
     * creation of the graph must be done as follows. Even if other ways are
     * documented, they will *NOT* work.
     * 
     *   - buffersrc:  snprintf() a minimal set of arguments into a string.
     *   - buffersink: av_opt_set_int_list() a list of admissible pixel formats.
     *                   -> Avoided by use of the "format" filter in graph.
     *   - graph:      1) snprintf() the graph description into a string.
     *                 2) avfilter_graph_parse_ptr() parse it.
     *                 3) avfilter_graph_config() configure it.
     */
    
    snprintf(buffersrcargs, sizeof(buffersrcargs),
             "video_size=%dx%d:pix_fmt=%d:sar=%d/%d:time_base=1/30:frame_rate=30",
             in->width, in->height, in->format,
             in->sample_aspect_ratio.num,
             in->sample_aspect_ratio.den);
    
    snprintf(graphstr, sizeof(graphstr),
             "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=%s",
             av_get_pix_fmt_name(canonical));
    ret = avfilter_graph_create_filter(&outputs->filter_ctx,
                                       buffersrc,
                                       "in",
                                       buffersrcargs,
                                       NULL,
                                       graph);
    if(ret < 0){
        i2hmessage(stderr, "Failed to allocate buffersrc filter instance!\n");
        goto fail;
    }
    ret = avfilter_graph_create_filter(&inputs->filter_ctx,
                                       buffersink,
                                       "out",
                                       NULL,
                                       NULL,
                                       graph);
    if(ret < 0){
        i2hmessage(stderr, "Failed to allocate buffersink filter instance!\n");
        goto fail;
    }
    buffersrcctx  = outputs->filter_ctx;
    buffersinkctx = inputs ->filter_ctx;
    ret = avfilter_graph_parse_ptr(graph, graphstr, &inputs, &outputs, NULL);
    if(ret < 0){
        i2hmessage(stderr, "Error parsing graph! (%d)\n", ret);
        goto fail;
    }
    ret = avfilter_graph_config(graph, NULL);
    if(ret < 0){
        i2hmessage(stderr, "Error configuring graph! (%d)\n", ret);
        goto fail;
    }
    
#if 0
    char* str = avfilter_graph_dump(graph, "");
    if(str){
        i2hmessage(stdout, "%s\n", str);
    }
    i2hmessage(stdout, "%s\n", graphstr);
    av_free(str);
#endif
    
    
    /**
     * Push the input  frame into the graph.
     * Pull the to-be-filtered frame from the graph.
     * 
     * Flushing the input pipeline requires either closing the buffer source
     * or pushing in a NULL frame.
     */
    
    ret = av_buffersrc_add_frame_flags(buffersrcctx, in, AV_BUFFERSRC_FLAG_KEEP_REF);
    av_buffersrc_close(buffersrcctx, 0, 0);
    if(ret < 0){
        i2hmessage(stderr, "Error pushing frame into graph! (%d)\n", ret);
        goto fail;
    }
    ret = av_buffersink_get_frame(buffersinkctx, out);
    if(ret < 0){
        i2hmessage(stderr, "Error pulling frame from graph! (%d)\n", ret);
        goto fail;
    }
    
    
    /**
     * Destroy graph and return.
     */
    
    ret = 0;
    fail:
    avfilter_free      (buffersrcctx);
    avfilter_free      (buffersinkctx);
    avfilter_inout_free(&outputs);
    avfilter_inout_free(&inputs);
    avfilter_graph_free(&graph);
    return ret;
}

/**
 * @brief Return whether the stream is of image/video type (contains pixel data).
 * 
 * @param [in]  avfctx  Format context whose streams we are inspecting.
 * @param [in]  stream  Integer number of the stream we are inspecting.
 * @return 0 if not a valid stream or not of image/video type; 1 otherwise.
 */

static int i2h_is_imagelike_stream(AVFormatContext* avfctx, int stream){
    if(stream < 0 || stream >= (int)avfctx->nb_streams){
        return 0;
    }
    return avfctx->streams[stream]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO;
}

/**
 * @brief Convert to x264-expected string the integer code for color primaries,
 *        transfer characteristics and color space.
 * 
 * @return A string representation of the code, as named by x264; Otherwise NULL.
 */

static const char* i2h_x264_color_primaries(enum AVColorPrimaries color_primaries){
    switch(color_primaries){
        case AVCOL_PRI_BT709:         return "bt709";
        case AVCOL_PRI_BT470M:        return "bt470m";
        case AVCOL_PRI_BT470BG:       return "bt470bg";
        case AVCOL_PRI_SMPTE170M:     return "smpte170m";
        case AVCOL_PRI_SMPTE240M:     return "smpte240m";
        case AVCOL_PRI_FILM:          return "film";
        case AVCOL_PRI_BT2020:        return "bt2020";
        case AVCOL_PRI_SMPTE428:      return "smpte428";
        case AVCOL_PRI_SMPTE431:      return "smpte431";
        case AVCOL_PRI_SMPTE432:      return "smpte432";
        default:                      return NULL;
    }
}
static const char* i2h_x264_color_trc(enum AVColorTransferCharacteristic color_trc){
    switch(color_trc){
        case AVCOL_TRC_BT709:         return "bt709";
        case AVCOL_TRC_GAMMA22:       return "bt470m";
        case AVCOL_TRC_GAMMA28:       return "bt470bg";
        case AVCOL_TRC_SMPTE170M:     return "smpte170m";
        case AVCOL_TRC_SMPTE240M:     return "smpte240m";
        case AVCOL_TRC_LINEAR:        return "linear";
        case AVCOL_TRC_LOG:           return "log100";
        case AVCOL_TRC_LOG_SQRT:      return "log316";
        case AVCOL_TRC_IEC61966_2_4:  return "iec61966-2-4";
        case AVCOL_TRC_BT1361_ECG:    return "bt1361e";
        case AVCOL_TRC_IEC61966_2_1:  return "iec61966-2-1";
        case AVCOL_TRC_BT2020_10:     return "bt2020-10";
        case AVCOL_TRC_BT2020_12:     return "bt2020-12";
        case AVCOL_TRC_SMPTE2084:     return "smpte2084";
        case AVCOL_TRC_SMPTE428:      return "smpte428";
        case AVCOL_TRC_ARIB_STD_B67:  return "arib-std-b67";
        default:                      return NULL;
    }
}
static const char* i2h_x264_color_space(enum AVColorSpace color_space){
    switch(color_space){
        case AVCOL_SPC_BT709:              return "bt709";
        case AVCOL_SPC_FCC:                return "fcc";
        case AVCOL_SPC_BT470BG:            return "bt470bg";
        case AVCOL_SPC_SMPTE170M:          return "smpte170m";
        case AVCOL_SPC_SMPTE240M:          return "smpte240m";
        case AVCOL_SPC_RGB:                return "GBR";
        case AVCOL_SPC_YCGCO:              return "YCgCo";
        case AVCOL_SPC_BT2020_NCL:         return "bt2020nc";
        case AVCOL_SPC_BT2020_CL:          return "bt2020c";
        case AVCOL_SPC_CHROMA_DERIVED_NCL: return "chroma-derived-nc";
        case AVCOL_SPC_CHROMA_DERIVED_CL:  return "chroma-derived-c";
        case AVCOL_SPC_SMPTE2085:          return "smpte2085";
        default:                           return NULL;
    }
}

static int  i2h_cuda_filter(UNIVERSE* u, AVFrame* dst, AVFrame* src, BENZINA_GEOM* geom){
    return 0;
}

/**
 * @brief Configure encoder for encoding the given prototypical frame.
 * 
 * @param [in,out]  u     The universe we're operating over.
 * @param [in]      frame The frame we use as a template for encoding.
 * @return 0 if successful; !0 otherwise.
 */

static int  i2h_configure_encoder(UNIVERSE* u, AVFrame* frame){
    char          x264params[1024];
    const char*   colorprimStr;
    const char*   transferStr;
    const char*   colormatrixStr;
    AVDictionary* codecCtxOpt = NULL;
    
    
    /**
     * Find codec, then configure and create encoder context.
     * 
     * This may fail if FFmpeg was not configured with x264.
     */
    
    u->out.codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if(!u->out.codec)
        i2hmessageexit(EINVAL, stderr, "Cannot transcode to H264! Please rebuild with an "
                                       "FFmpeg configured with --enable-libx264.\n");
    
    u->out.codecCtx = avcodec_alloc_context3(u->out.codec);
    if(!u->out.codecCtx)
        i2hmessageexit(ENOMEM, stderr, "Could not allocate encoder!\n");
    
    u->out.codecCtx->width                  = frame->width;
    u->out.codecCtx->height                 = frame->height;
    u->out.codecCtx->time_base.num          =  1;  /* Nominal 30 fps */
    u->out.codecCtx->time_base.den          = 30;  /* Nominal 30 fps */
    u->out.codecCtx->gop_size               =  0;  /* Intra only */
    u->out.codecCtx->pix_fmt                = frame->format;
    u->out.codecCtx->chroma_sample_location = frame->chroma_location;
    u->out.codecCtx->color_range            = frame->color_range;
    u->out.codecCtx->color_primaries        = frame->color_primaries;
    u->out.codecCtx->color_trc              = frame->color_trc;
    u->out.codecCtx->colorspace             = frame->colorspace;
    u->out.codecCtx->sample_aspect_ratio    = frame->sample_aspect_ratio;
    u->out.codecCtx->flags                 |= AV_CODEC_FLAG_LOOP_FILTER;
    u->out.codecCtx->thread_count           = 1;
    u->out.codecCtx->slices                 = 1;
    colorprimStr   = i2h_x264_color_primaries(frame->color_primaries);
    colorprimStr   = colorprimStr   ? colorprimStr   : i2h_x264_color_primaries(
                                      frame->color_primaries = AVCOL_PRI_BT709);
    transferStr    = i2h_x264_color_trc      (frame->color_trc);
    transferStr    = transferStr    ? transferStr    : i2h_x264_color_trc(
                                      frame->color_trc       = AVCOL_TRC_IEC61966_2_1);
    colormatrixStr = i2h_x264_color_space    (frame->colorspace);
    colormatrixStr = colormatrixStr ? colormatrixStr : i2h_x264_color_space(
                                      frame->colorspace      = AVCOL_SPC_BT470BG);
    snprintf(x264params, sizeof(x264params),
             "ref=1:chromaloc=1:log=-1:slices-max=32:fullrange=on:stitchable=1:"
             "colorprim=%s:transfer=%s:colormatrix=%s%s%s",
             colorprimStr, transferStr, colormatrixStr,
             (u->args.x264params && *u->args.x264params) ? ":"                : "",
             (u->args.x264params)                        ? u->args.x264params : "");
    av_dict_set    (&codecCtxOpt, "profile",     "high444",    0);
    av_dict_set    (&codecCtxOpt, "preset",      "placebo",    0);
    av_dict_set    (&codecCtxOpt, "tune",        "stillimage", 0);
    av_dict_set_int(&codecCtxOpt, "forced-idr",  1,            0);
    av_dict_set_int(&codecCtxOpt, "crf",         u->out.crf,  0);
    av_dict_set    (&codecCtxOpt, "x264-params", x264params,   0);
    u->ret = avcodec_open2(u->out.codecCtx, u->out.codec, &codecCtxOpt);
    if(av_dict_count(codecCtxOpt)){
        AVDictionaryEntry* entry = NULL;
        i2hmessage(stderr, "Warning: codec ignored %d options!\n",
                           av_dict_count(codecCtxOpt));
        while((entry=av_dict_get(codecCtxOpt, "", entry, AV_DICT_IGNORE_SUFFIX))){
            i2hmessage(stderr, "    %20s: %s\n", entry->key, entry->value);
        }
    }
    av_dict_free(&codecCtxOpt);
    if(u->ret < 0)
        i2hmessageexit(u->ret, stderr, "Could not open encoder! (%d)\n", u->ret);
    
    
    /* Return */
    return u->ret = 0;
}

/**
 * @brief Release the universe's encoder context.
 * 
 * @param [in,out]  u     The universe we're operating over.
 */

static void i2h_deconfigure_encoder(UNIVERSE* u){
    avcodec_free_context(&u->out.codecCtx);
    u->out.codec = NULL;
}

/**
 * @brief Handle one frame.
 * 
 * This means:
 * 
 *   - Creating the filter graph specific to this frame
 *   - Invoking the filter graph, thus canonicalizing the pixel format
 *   - Solving the geometry problem for this frame.
 *   - Creating the canvas by executing custom CUDA code.
 *   - Encoding it in H.264
 *   - Surrounding it with metadata and an embedded payload.
 *   - Writing it out.
 */

static int  i2h_do_frame        (UNIVERSE*        u, AVFrame* frame){
    AVFrame*     graphfiltframe = NULL;
    AVFrame*     cudafiltframe  = NULL;
    
    /* Graph & CUDA Filtering Init */
    graphfiltframe = av_frame_alloc();
    if(!graphfiltframe)
        i2hmessageexit(ENOMEM, stderr, "Error allocating frame!\n");
    
    cudafiltframe = av_frame_alloc();
    if(!cudafiltframe)
        i2hmessageexit(ENOMEM, stderr, "Error allocating frame!\n");
    
    
    /**
     * Fixup and graph-filter to canonicalize input frame
     */
    
    i2h_fixup_frame(frame);
    u->ret = i2h_canonicalize_pixfmt(graphfiltframe, frame);
    if(u->ret < 0)
        i2hmessageexit(u->ret, stderr, "Failed to canonicalize pixel format! (%d)\n", u->ret);
    
    
    /**
     * Solve Geometry Problem.
     * 
     * Because Benzina's chroma_loc codepoints follow the ITU-T H.273 standard,
     * and FFmpeg's don't, there is a shift of +1 in the integer values of FFmpeg's
     * enum values. We cannot use raw FFmpeg chroma_loc, so we convert them.
     */
    
    u->tx.geom.i.source.w          = graphfiltframe->width;
    u->tx.geom.i.source.h          = graphfiltframe->height;
    u->tx.geom.i.source.chroma_loc = av2benzinachromaloc(graphfiltframe->chroma_location);
    switch(graphfiltframe->format){
        case AV_PIX_FMT_GRAY8:   u->tx.geom.i.source.chroma_fmt = BENZINA_CHROMAFMT_YUV400; break;
        case AV_PIX_FMT_YUV420P: u->tx.geom.i.source.chroma_fmt = BENZINA_CHROMAFMT_YUV420; break;
        case AV_PIX_FMT_YUV422P: u->tx.geom.i.source.chroma_fmt = BENZINA_CHROMAFMT_YUV422; break;
        case AV_PIX_FMT_GBRP:
        case AV_PIX_FMT_YUV444P: u->tx.geom.i.source.chroma_fmt = BENZINA_CHROMAFMT_YUV444; break;
        default:
            i2hmessageexit(EINVAL, stderr, "Unsupported pixel format %s!\n",
                                           av_get_pix_fmt_name(graphfiltframe->format));
    }
    u->tx.geom.i.canvas.w          = u->out.tile_width;
    u->tx.geom.i.canvas.h          = u->out.tile_height;
    u->tx.geom.i.canvas.chroma_fmt = u->out.chroma_format;
    u->tx.geom.i.canvas.superscale = 0;
    benzina_geom_solve(&u->tx.geom);
    
    
    /**
     * CUDA Filtering.
     */
    
    cudafiltframe->width           = u->out.tile_width;
    cudafiltframe->height          = u->out.tile_height;
    switch(u->out.chroma_format){
        case BENZINA_CHROMAFMT_YUV400: cudafiltframe->format = AV_PIX_FMT_GRAY8;   break;
        case BENZINA_CHROMAFMT_YUV420: cudafiltframe->format = AV_PIX_FMT_YUV420P; break;
        case BENZINA_CHROMAFMT_YUV422: cudafiltframe->format = AV_PIX_FMT_YUV422P; break;
        case BENZINA_CHROMAFMT_YUV444: cudafiltframe->format = AV_PIX_FMT_YUV444P; break;
        default:
            i2hmessageexit(EINVAL, stderr, "Unsupported canvas chroma_fmt %d!\n",
                                           u->out.chroma_format);
    }
    cudafiltframe->chroma_location = benzina2avchromaloc(u->tx.geom.o.chroma_loc);
    cudafiltframe->color_range     = graphfiltframe->color_range;
    cudafiltframe->color_primaries = graphfiltframe->color_primaries;
    cudafiltframe->color_trc       = graphfiltframe->color_trc;
    cudafiltframe->colorspace      = graphfiltframe->colorspace;
    u->ret = av_frame_get_buffer(cudafiltframe, 0);
    if(u->ret != 0)
        i2hmessageexit(ENOMEM, stderr, "Error allocating frame! (%d)\n", u->ret);
    
    u->ret = i2h_cuda_filter(u, cudafiltframe, graphfiltframe, &u->tx.geom);
    if(u->ret < 0)
        i2hmessageexit(u->ret, stderr, "Failed to filter image! (%d)\n", u->ret);
    
    
    /* Encoding Init */
    u->out.packet = av_packet_alloc();
    if(!u->out.packet)
        i2hmessageexit(ENOMEM, stderr, "Failed to allocate packet object!\n");
    
    
    /**
     * Push the filtered frame  into the encoder.
     * Pull the output packet from the encoder.
     * 
     * Flushing the input encode pipeline requires pushing in a NULL frame.
     */
    
    u->ret = i2h_configure_encoder(u, cudafiltframe);
    if(u->ret)
        i2hmessageexit(u->ret, stderr, "Failed to configure x264 encoder! (%d)\n", u->ret);
    
    cudafiltframe->pts = frame->best_effort_timestamp;
    u->ret = avcodec_send_frame(u->out.codecCtx, cudafiltframe);
    if(u->ret < 0)
        i2hmessageexit(u->ret, stderr, "Error pushing frame into encoder! (%d)\n", u->ret);
    
    u->ret = avcodec_send_frame(u->out.codecCtx, NULL);
    if(u->ret < 0)
        i2hmessageexit(u->ret, stderr, "Error flushing encoder input! (%d)\n", u->ret);
    
    u->ret = avcodec_receive_packet(u->out.codecCtx, u->out.packet);
    if(u->ret < 0)
        i2hmessageexit(u->ret, stderr, "Error pulling packet from encoder! (%d)\n", u->ret);
    
    i2h_deconfigure_encoder(u);
    
    
    /* Write Out */
#if 0
    fwrite(u->out.packet->data, 1, u->out.packet->size, u->out.fp);
    fflush(u->out.fp);
#endif
    
    
    /* Cleanup */
    av_packet_free(&u->out.packet);
    av_frame_free(&graphfiltframe);
    av_frame_free(&cudafiltframe);
    
    
    /* Return */
    return u->ret=0;
}

/**
 * Open and close FFmpeg.
 * 
 * In-between, handle frames.
 */

static int  i2h_main            (UNIVERSE* u){
    /* Input Init */
    u->ret              = avformat_open_input         (&u->in.formatCtx,
                                                       u->args.urlIn,
                                                       NULL, NULL);
    if(u->ret < 0)
        i2hmessageexit(u->ret, stderr, "Cannot open input media \"%s\"! (%d)\n", u->args.urlIn, u->ret);
    
    u->ret              = avformat_find_stream_info   (u->in.formatCtx, NULL);
    if(u->ret < 0)
        i2hmessageexit(EIO,    stderr, "Cannot understand input media \"%s\"! (%d)\n", u->args.urlIn, u->ret);
    
    u->in.formatStream  = av_find_default_stream_index(u->in.formatCtx);
    if(!i2h_is_imagelike_stream(u->in.formatCtx, u->in.formatStream))
        i2hmessageexit(EINVAL, stderr, "No image data in media \"%s\"!\n", u->args.urlIn);
    
    u->in.codecPar      = u->in.formatCtx->streams[u->in.formatStream]->codecpar;
    u->in.codec         = avcodec_find_decoder        (u->in.codecPar->codec_id);
    u->in.codecCtx      = avcodec_alloc_context3      (u->in.codec);
    if(!u->in.codecCtx)
        i2hmessageexit(EINVAL, stderr, "Could not allocate decoder!\n");
    
    u->ret              = avcodec_open2               (u->in.codecCtx,
                                                       u->in.codec,
                                                       NULL);
    if(u->ret < 0)
        i2hmessageexit(u->ret, stderr, "Could not open decoder! (%d)\n", u->ret);
    
    
    /**
     * Input packet-reading loop.
     */
    
    do{
        AVPacket* packet = av_packet_alloc();
        if(!packet)
            i2hmessageexit(ENOMEM, stderr, "Failed to allocate packet object!\n");
        
        u->ret = av_read_frame(u->in.formatCtx, packet);
        if(u->ret < 0)
            i2hmessageexit(u->ret, stderr, "Error or end-of-file reading media \"%s\" (%d)!\n",
                                           u->args.urlIn, u->ret);
        
        if(packet->stream_index != u->in.formatStream){
            av_packet_free(&packet);
        }else{
            u->ret = avcodec_send_packet(u->in.codecCtx, packet);
            av_packet_free(&packet);
            if(u->ret){
                i2hmessageexit(u->ret, stderr, "Error pushing packet! (%d)\n", u->ret);
            }else{
                AVFrame* frame = av_frame_alloc();
                if(!frame)
                    i2hmessageexit(ENOMEM, stderr, "Error allocating frame!\n");
                
                u->ret = avcodec_receive_frame(u->in.codecCtx, frame);
                if(u->ret)
                    i2hmessageexit(u->ret, stderr, "Error pulling frame! (%d)\n",  u->ret);
                
                u->ret = i2h_do_frame(u, frame);
                av_frame_free(&frame);
            }
            break;
        }
    }while(!u->ret);
    
    
    /* Cleanup */
    avcodec_free_context(&u->in.codecCtx);
    avformat_close_input(&u->in.formatCtx);
    
    
    /* Return */
    return u->ret;
}

/**
 * @brief Output a HEIF file with the given items.
 * 
 * @param [in]  u  The universe we're operating over.
 * @return 0 if successful; !0 otherwise.
 */

static int  i2h_do_output         (UNIVERSE*        u){
    (void)u;
    return 0;
}

/**
 * @brief Append the set of ID references to a linked list of them.
 * @param [in,out]  refs  The head of the linked list
 * @param [in,out]  iref  The set to be appended.
 * @return 0.
 */

static int  i2h_append_irefs      (IREF** refs, IREF* iref){
    iref->next = *refs;
    *refs = iref;
    return 0;
}

/**
 * @brief Parse individual arguments.
 * 
 * @param [in/out]  u        The universe we're operating over.
 * @param [in,out]  argv     A pointer to a pointer to the arguments array.
 * 
 * On entry,  **argv points just past the end of the option's name.
 * On return, **argv points to the next option.
 * 
 * @return !0 if successfully parsed, 0 otherwise.
 */

static int i2h_parse_shortopt(char*** argv){
    int ret = **argv            &&
             (**argv)[0] == '-' &&
             (**argv)[1] != '-' &&
             (**argv)[1] != '\0';
    if(ret){
        ++**argv;
    }
    return ret;
}
static int i2h_parse_longopt(char*** argv, const char* opt){
    size_t l   = strlen(opt);
    int    ret = strncmp(**argv, opt, l) == 0 &&
                       ((**argv)[l] == '\0' ||
                        (**argv)[l] == '=');
    if(ret){
        **argv += l;
    }
    return ret;
}
static int i2h_parse_consumeequals(char*** argv){
    switch(***argv){
        case '\0': return !!*++*argv;
        case '=':  return !!++**argv;
        default:   return 0;
    }
}
static int i2h_parse_noconsume(char*** argv){
    switch(***argv){
        case '\0': return (void)*++*argv, 1;
        case '=':  i2hmessageexit(EINVAL, stderr, "Option takes no arguments!\n");
        default:   return 1;
    }
}
static int i2h_parse_codec(UNIVERSE* u, char*** argv){
    const char* codec;
    
    if(!i2h_parse_consumeequals(argv)){
        i2hmessageexit(EINVAL, stderr, "Option --codec consumes an argument!\n");
    }
    
    codec = *(*argv)++;
    if      (strcmp(codec, "h264") == 0 ||
             strcmp(codec, "x264") == 0){
        codec = "libx264";
    }else if(strcmp(codec, "hevc") == 0 ||
             strcmp(codec, "h265") == 0 ||
             strcmp(codec, "x265") == 0){
        codec = "libx265";
    }
    
    u->out.codec = avcodec_find_encoder_by_name(codec);
    if(!u->out.codec)
        i2hmessageexit(EINVAL, stderr, "--codec '%s' unknown!\n", codec);
    
    return 1;
}
static int i2h_parse_tile(UNIVERSE* u, char*** argv){
    char tile_chroma_format[21] = "yuv420";
    
    if(!i2h_parse_consumeequals(argv)){
        i2hmessageexit(EINVAL, stderr, "Option --tile consumes an argument!\n");
    }
    
    switch(sscanf(*(*argv)++, "%"PRIu32":%"PRIu32":%20[yuv420]",
                  &u->out.tile_width,
                  &u->out.tile_height,
                  tile_chroma_format)){
        default:
        case 0: i2hmessageexit(EINVAL, stderr, "--tile requires an argument formatted 'width:height:chroma_format' !\n");
        case 1: u->out.tile_height = u->out.tile_width; break;
        case 2:
        case 3: break;
    }
    u->out.chroma_format = i2h_str2chromafmt(tile_chroma_format);
    if(u->out.tile_width  < 48 || u->out.tile_width  > 4096 || u->out.tile_width  % 16 != 0)
        i2hmessageexit(EINVAL, stderr, "Tile width must be 48-4096 pixels in multiples of 16!\n");
    if(u->out.tile_height < 48 || u->out.tile_height > 4096 || u->out.tile_height % 16 != 0)
        i2hmessageexit(EINVAL, stderr, "Tile height must be 48-4096 pixels in multiples of 16!\n");
    if(u->out.chroma_format > BENZINA_CHROMAFMT_YUV444)
        i2hmessageexit(EINVAL, stderr, "Unknown chroma format \"%s\"!\n", tile_chroma_format);
    
    return 1;
}
static int i2h_parse_x264_params(UNIVERSE* u, char*** argv){
    if(!i2h_parse_consumeequals(argv)){
        i2hmessageexit(EINVAL, stderr, "Option --x264-params consumes an argument!\n");
    }
    
    u->out.x264_params = *(*argv)++;
    return 1;
}
static int i2h_parse_x265_params(UNIVERSE* u, char*** argv){
    if(!i2h_parse_consumeequals(argv)){
        i2hmessageexit(EINVAL, stderr, "Option --x265-params consumes an argument!\n");
    }
    
    u->out.x265_params = *(*argv)++;
    return 1;
}
static int i2h_parse_crf(UNIVERSE* u, char*** argv){
    char* p;
    
    if(!i2h_parse_consumeequals(argv)){
        i2hmessageexit(EINVAL, stderr, "Option --crf consumes an argument!\n");
    }
    
    u->out.crf = strtoull(p=*(*argv)++, &p, 0);
    if(u->out.crf > 51 || *p){
        i2hmessageexit(EINVAL, stderr, "Error: --crf must be integer in range [0, 51]!\n");
    }
    return 1;
}
static int i2h_parse_hidden(UNIVERSE* u, char*** argv){
    if(!i2h_parse_noconsume(argv)){
        i2hmessageexit(EINVAL, stderr, "Option --hidden does not consume an argument!\n");
    }
    u->item.hidden         = 1;
    return 1;
}
static int i2h_parse_primary(UNIVERSE* u, char*** argv){
    if(!i2h_parse_noconsume(argv)){
        i2hmessageexit(EINVAL, stderr, "Option --primary does not consume an argument!\n");
    }
    u->item.primary        = 1;
    return 1;
}
static int i2h_parse_thumbnail(UNIVERSE* u, char*** argv){
    if(!i2h_parse_noconsume(argv)){
        i2hmessageexit(EINVAL, stderr, "Option --thumb does not consume an argument!\n");
    }
    u->item.want_thumbnail = 1;
    return 1;
}
static int i2h_parse_name(UNIVERSE* u, char*** argv){
    if(!i2h_parse_consumeequals(argv)){
        i2hmessageexit(EINVAL, stderr, "Option --name consumes an argument!\n");
    }
    
    u->item.name = *(*argv)++;
    return 1;
}
static int i2h_parse_ref(UNIVERSE* u, char*** argv){
    char* p, *s;
    uint32_t r    = 0;
    int      n    = 0;
    IREF*    iref = calloc(1, sizeof(*iref));
    
    /* Consume the = if it's there. */
    if(!i2h_parse_consumeequals(argv)){
        i2hmessageexit(EINVAL, stderr, "Option --ref consumes an argument!\n");
    }
    
    /* Extract the argument. */
    p = *(*argv)++;
    
    /* First 4 bytes are the type. */
    memcpy(iref->type, p, sizeof(iref->type));
    p += sizeof(iref->type);
    
    /* The type is separated from the IDs by a : */
    if(*p++ != ':'){
        i2hmessageexit(EINVAL, stderr, "Option --ref requires form 'type:id,id,id,...' !\n");
    }
    
    /* Count the number of IDs. */
    s = p;
    do{
        if(sscanf(s, " %"PRIu32"%*[, ]%n", &r, &n) == 1){
            iref->num_references++;
            s += n;
        }else{
            break;
        }
    }while(1);
    
    /* Allocate the array of IDs. */
    iref->to_item_id = calloc(iref->num_references,
                              sizeof(*iref->to_item_id));
    
    /* Reparse the IDs, this time saving them into the array. */
    s = p;
    for(r=0;r<iref->num_references;r++){
        sscanf(s, " %"PRIu32"%*[, ]%n", &iref->to_item_id[r], &n);
        s += n;
    }
    
    /* Append this set of references to the current item. */
    i2h_append_irefs(&u->item.refs, iref);
    
    return 1;
}
static int i2h_parse_mime(UNIVERSE* u, char*** argv){
    if(!i2h_parse_consumeequals(argv)){
        i2hmessageexit(EINVAL, stderr, "Option --mime consumes an argument!\n");
    }
    
    u->item.mime = *(*argv)++;
    return 1;
}
static int i2h_parse_item(UNIVERSE* u, char*** argv){
    char* p;
    char* q;
    int   id_set = 0;
    ITEM* i      = u->items;
    
    /* Consume the = if it's there. */
    if(!i2h_parse_consumeequals(argv)){
        i2hmessageexit(EINVAL, stderr, "Option --item consumes an argument!\n");
    }
    
    /* Extract the argument. */
    p = *(*argv)++;
    
    /* Parse the argument. */
    while(1){
        if      (strstartswith(p, "id=")){
            if(id_set){
                i2hmessageexit(EINVAL, stderr, "Item ID has already been set!\n");
            }
            u->item.id = strtoull(p+=3, &q, 0);
            if(p==q){
                i2hmessageexit(EINVAL, stderr, "id= requires a 32-bit unsigned integer item ID!\n");
            }else{
                p = q;
            }
            id_set = 1;
            for(;i;i=i->next){
                if(i->id == u->item.id){
                    i2hmessageexit(EINVAL, stderr, "Pre-existing item with same ID %"PRIu32"!\n", u->item.id);
                }
            }
        }else if(strstartswith(p, "type=")){
            memcpy(u->item.type, p+=5, 4);
            p += 4;
        }else if(strstartswith(p, "path=")){
            u->item.path = (p+=5);
            break;
        }else{
            i2hmessageexit(EINVAL, stderr, "Unknown key '%s'\n", p);
        }
        
        if(*p++ != ','){
            i2hmessageexit(EINVAL, stderr, "The keys of --item must be separated by ',' and the last key must be path= !\n");
        }
    }
    
    /**
     * Validate.
     * 
     * - Type codes must be exactly 4 bytes.
     * - When no id= key is given, automatically assign one.
     */
    
    if(strnlen(u->item.type, 4) != 4){
        i2hmessageexit(EINVAL, stderr, "The type= key must be exactly 4 ASCII characters!\n");
    }
    if(!id_set){
        if(!u->items || !u->items->id){
            u->item.id = 0;
        }else{
            for(i=u->items; i->next && i->id+1==i->next->id; i=i->next){}
            u->item.id = i->id+1;
        }
    }
    
#if 0
    fprintf(stdout, "id=%u,type=%4.4s,path=%s\n",
            u->item.id, u->item.type, u->item.path);
#endif
    
    return 1;
}
static int i2h_parse_output(UNIVERSE* u, char*** argv){
    if(!i2h_parse_consumeequals(argv)){
        i2hmessageexit(EINVAL, stderr, "Option --output consumes an argument!\n");
    }
    
    u->out.url = *(*argv)++;
    return 1;
}
static int i2h_parse_help(UNIVERSE* u, char*** argv){
    (void)u;
    (void)argv;
    
    fprintf(stdout,
        "Usage: benzina-image2heif [args]\n"
    );
    exit(0);
    
    return 0;
}

/**
 * @brief Parse the arguments, performing what they require.
 * 
 * The command-line is structured in several groups of flags:
 *   - An input group,  which terminates in -i or --item and declares an input file.
 *   - An output group, which terminates in -o and declares an output file.
 * 
 * There can be many input groups, declaring multiple items, but there can be at
 * most one output group.
 * 
 * @param [in,out]  u     The universe we're operating over.
 * @param [in,out]  argv  A pointer to a pointer to the arguments array.
 * 
 * @return Exit code.
 */

static int  i2h_parse_args      (UNIVERSE*        u,
                                 char***          argv){
    const char* s;
    while(**argv){
        if(i2h_parse_shortopt(argv)){
            /**
             * Iterate over the individual characters of a short option sequence
             * until we 1) fall off its end or 2) consume an argument.
             */
            for(s=**argv; (s == **argv) && (***argv); ++s){
                switch(*(**argv)++){
                    case 'H': i2h_parse_hidden   (u, argv); break;
                    case 'p': i2h_parse_primary  (u, argv); break;
                    case 't': i2h_parse_thumbnail(u, argv); break;
                    case 'h': i2h_parse_help     (u, argv); break;
                    case 'm': i2h_parse_mime     (u, argv); break;
                    case 'n': i2h_parse_name     (u, argv); break;
                    case 'r': i2h_parse_ref      (u, argv); break;
                    case 'i': i2h_parse_item     (u, argv); break;
                    case 'o': i2h_parse_output   (u, argv); break;
                    default:
                        i2hmessageexit(EINVAL, stderr, "Unknown short option '-%c' !\n", *--**argv);
                }
            }
            if(s == **argv){
                /* We fell off the end of a series of short options. */
                ++*argv;
            }
        }else if(i2h_parse_longopt(argv, "--codec")){
            i2h_parse_codec(u, argv);
        }else if(i2h_parse_longopt(argv, "--tile")){
            i2h_parse_tile(u, argv);
        }else if(i2h_parse_longopt(argv, "--x264-params")){
            i2h_parse_x264_params(u, argv);
        }else if(i2h_parse_longopt(argv, "--x265-params")){
            i2h_parse_x265_params(u, argv);
        }else if(i2h_parse_longopt(argv, "--crf")){
            i2h_parse_crf(u, argv);
        }else if(i2h_parse_longopt(argv, "--hidden")){
            i2h_parse_hidden(u, argv);
        }else if(i2h_parse_longopt(argv, "--primary")){
            i2h_parse_primary(u, argv);
        }else if(i2h_parse_longopt(argv, "--thumb")){
            i2h_parse_thumbnail(u, argv);
        }else if(i2h_parse_longopt(argv, "--mime")){
            i2h_parse_mime(u, argv);
        }else if(i2h_parse_longopt(argv, "--name")){
            i2h_parse_name(u, argv);
        }else if(i2h_parse_longopt(argv, "--ref")){
            i2h_parse_ref(u, argv);
        }else if(i2h_parse_longopt(argv, "--item")){
            i2h_parse_item(u, argv);
        }else if(i2h_parse_longopt(argv, "--output")){
            i2h_parse_output(u, argv);
        }else if(i2h_parse_longopt(argv, "--help")){
            i2h_parse_help(u, argv);
        }else{
            i2hmessageexit(EINVAL, stderr, "Unknown long option '%s' !\n", **argv);
        }
    }
    
    return i2h_do_output(u);
}

/**
 * @brief Initialize the universe to default values.
 * 
 * @param [in]  u  The universe we're operating over.
 * @return 0.
 */

static int  i2h_init_item       (ITEM*            item){
    item->id             = 0;
    item->path           = NULL;
    item->name           = "";
    memcpy(item->type, "pict", 4);
    item->mime           = "application/octet-stream";
    item->primary        = 0;
    item->hidden         = 0;
    item->want_thumbnail = 0;
    item->width          = 0;
    item->height         = 0;
    item->frame          = NULL;
    item->packet         = NULL;
    item->refs           = NULL;
    item->thumbnail      = NULL;
    item->grid           = NULL;
    item->next           = NULL;
    return 0;
}

/**
 * @brief Initialize the universe to default values.
 * 
 * @param [in]  u  The universe we're operating over.
 * @return 0.
 */

static int  i2h_init_universe   (UNIVERSE*        u){
    i2h_init_item(&u->item);
    u->items              = NULL;
    
    u->out.codec          = avcodec_find_encoder_by_name("libx265");
    u->out.codecCtx       = NULL;
    u->out.packet         = NULL;
    u->out.tile_width     = 512;
    u->out.tile_height    = 512;
    u->out.crf            = 10;
    u->out.chroma_format  = BENZINA_CHROMAFMT_YUV420;
    u->out.x264_params    = "";
    u->out.x265_params    = "";
    u->out.url            = NULL;
    
    return 0;
}

/**
 * Main
 * 
 * Supports the following arguments:
 * 
 *   "Sticky" Parameters (preserved across picture items until changed)
 *     --codec=S                Codec selection. Default "h265", synonym "hevc", alternative "h264".
 *     --tile=W:H[:F]           Tile configuration. Width:Height:Chroma Format. Default 512:512:yuv420.
 *     --x264-params=S          Additional x264 parameters. Default "".
 *     --x265-params=S          Additional x265 parameters. Default "".
 *     --crf=N                  CRF 0-51. Default 10.
 * 
 *   "Unsticky" Parameters (used for only one item)
 *     -H, --hidden             Set hidden bit. Default false. Incompatible with -p.
 *     -p, --primary            Mark as primary item. Default false. Incompatible with -h.
 *     -t, --thumb              Want (single-tile) thumbnail.
 *     -n, --name=S             Name of item.
 *     -m, --mime=S             MIME-type of item. Default "application/octet-stream".
 *     -r, --ref=type:id[,id]*  Typed reference. Can be specified multiple times.
 * 
 *   Item Declaration
 *     -i, --item=[id=N,][type=T,]path=S   Source file. Examples:
 *     -i path=image.png        Add item with next-lowest-numbered ID of type
 *                              "pict" from image.png
 *     -i id=4,path=image.png   Add item with ID=4 of type "pict" from image.png
 *     -i id=4,type=pict,path=image.png 
 *                              Add item with ID=4 of type "pict" from image.png
 *     --mime application/octet-stream -i type=mime,path=target.bin
 *                              Add item with next-lowest-numbered ID of MIME
 *                              type "application/octet-stream" from target.bin.
 * 
 *   Output Declaration
 *     -o, --output=S       Target output filename. Default "out.heif".
 * 
 *   Miscellaneous
 *     -h, --help           Print help and exit.
 */

int main(int argc, char* argv[]){
    static UNIVERSE ustack = {0}, *u = &ustack;
    (void)argc;
    argv++;
    i2h_init_universe(u);
    return i2h_parse_args(u, &argv);
}



#if 0
/**
 * Collection of CUDA kernels that accurately resamples and convert from
 * FFmpeg's better-supported pixel formats to full-range YUV420P.
 * 
 * 
 * Not supported:
 *   - Bayer Pattern:    bayer_*
 *   - Sub-8-bit RGB:    bgr4, rgb4, bgr8, rgb8, bgr4_byte, rgb4_byte,
 *                       bgr444be, bgr444le, rgb444be, rgb444le,
 *                       bgr555be, bgr555le, rgb555be, rgb555le,
 *                       bgr565be, bgr565le, rgb565be, rgb565le
 *   - Hardware formats: ...
 *   - XYZ:              xyz12be, xyz12le
 */



/* CUDA kernels */

/**
 * @brief Floating-point clamp
 */

static float  fclampf(float lo, float x, float hi){
    return fmaxf(lo, fminf(x, hi));
}

/**
 * @brief Elementwise math Functions
 */

static float3 lerps3(float3 a, float3 b, float m){
    return make_float3((1.0f-m)*a.x + (m)*b.x,
                       (1.0f-m)*a.y + (m)*b.y,
                       (1.0f-m)*a.z + (m)*b.z);
}
static float3 lerpv3(float3 a, float3 b, float3 m){
    return make_float3((1.0f-m.x)*a.x + (m.x)*b.x,
                       (1.0f-m.y)*a.y + (m.y)*b.y,
                       (1.0f-m.z)*a.z + (m.z)*b.z);
}

/**
 * @brief Forward/Backward Transfer Functions
 * 
 * Some of the backward functions are not yet implemented, because
 * deriving them is non-trivial.
 * 
 * Reference:
 *     ITU-T Rec. H.264   (10/2016)
 *     http://avisynth.nl/index.php/Colorimetry
 */

static float  fwXfrFn (float  L, unsigned fn){
    float a, b, g, c1, c2, c3, n, m, p;
    
    switch(fn){
        case BENZINA_XFR_BT_709:
        case BENZINA_XFR_SMPTE_ST_170M:
        case BENZINA_XFR_BT_2020_10BIT:
        case BENZINA_XFR_BT_2020_12BIT:
            a = 1.099296826809442f;
            b = 0.018053968510807f;
            return L >= b ? a*powf(L, 0.45f) - (a-1.f) : 4.500f*L;
        case BENZINA_XFR_GAMMA22: return powf(fmaxf(L, 0.f), 1.f/2.2f);
        case BENZINA_XFR_GAMMA28: return powf(fmaxf(L, 0.f), 1.f/2.8f);
        case BENZINA_XFR_SMPTE_ST_240M:
            a = 1.111572195921731f;
            b = 0.022821585529445f;
            return L >= b ? a*powf(L, 0.45f) - (a-1.f) : 4.0f*L;
        case BENZINA_XFR_LINEAR:  return L;
        case BENZINA_XFR_LOG20:   return L < sqrtf(1e-4) ? 0.f : 1.f + log10f(L)/2.0f;
        case BENZINA_XFR_LOG25:   return L < sqrtf(1e-5) ? 0.f : 1.f + log10f(L)/2.5f;
        case BENZINA_XFR_IEC_61966_2_4:
            a = 1.099296826809442f;
            b = 0.018053968510807f;
            if      (L >= +b){return +a*powf(+L, 0.45f) - (a-1.f);
            }else if(L <= -b){return -a*powf(-L, 0.45f) + (a-1.f);
            }else{            return 4.5f*L;
            }
        case BENZINA_XFR_BT_1361:
            a = 1.099296826809442f;
            b = 0.018053968510807f;
            g = b/4.f;
            if      (L >= +b){return +(a*powf(+1.f*L, 0.45f) - (a-1.f));
            }else if(L <= -g){return -(a*powf(-4.f*L, 0.45f) - (a-1.f))/4.f;
            }else{            return 4.5f*L;
            }
        case BENZINA_XFR_UNSPECIFIED:
        case BENZINA_XFR_IEC_61966_2_1:
        default:
            a = 1.055f;
            b = 0.0031308f;
            return L > b ? a*powf(L, 1.f/2.4f) - (a-1.f) : 12.92f*L;
        case BENZINA_XFR_BT_2100_PQ:
            c1 =  107.f/  128.f;
            c2 = 2413.f/  128.f;
            c3 = 2392.f/  128.f;
            m  = 2523.f/   32.f;
            n  = 2610.f/16384.f;
            p  = powf(fmaxf(L, 0.f), n);
            return powf((c1+c2*p)/(1.f+c3*p), m);
        case BENZINA_XFR_SMPTE_ST_428: return powf(48.f*fmaxf(L, 0.f)/52.37f, 1.f/2.6f);
    }
}
static float  bwXfrFn (float  V, unsigned fn){
    float a, b;
    
    switch(fn){
        case BENZINA_XFR_BT_709:
        case BENZINA_XFR_SMPTE_ST_170M:
        case BENZINA_XFR_BT_2020_10BIT:
        case BENZINA_XFR_BT_2020_12BIT:
            a = 0.099296826809442f;
            b = 0.08124285829863229f;
            return V >= b ? powf((V+a)/(1.f+a), 1.f/0.45f) : V/4.500f;
        case BENZINA_XFR_GAMMA22: return powf(V, 2.2f);
        case BENZINA_XFR_GAMMA28: return powf(V, 2.8f);
        case BENZINA_XFR_SMPTE_ST_240M:
            a = 1.111572195921731f;
            b = 0.09128634211778018f;
            return V >= b ? powf((V+a)/(1.f+a), 1.f/0.45f) : V/4.0f;
        case BENZINA_XFR_LINEAR:  return V;
        case BENZINA_XFR_LOG20:   return powf(10.f, (V-1.f)*2.0f);
        case BENZINA_XFR_LOG25:   return powf(10.f, (V-1.f)*2.5f);
        case BENZINA_XFR_IEC_61966_2_4:  return 0.0f;/* FIXME: Implement me! */
        case BENZINA_XFR_BT_1361:      return 0.0f;/* FIXME: Implement me! */
        case BENZINA_XFR_UNSPECIFIED:
        case BENZINA_XFR_IEC_61966_2_1:
        default:
            a = 0.055f;
            b = 0.04045f;
            return V >  b ? powf((V+a)/(1.f+a),      2.4f) : V/12.92f;
        case BENZINA_XFR_BT_2100_PQ:      return 0.0f;/* FIXME: Implement me! */
        case BENZINA_XFR_SMPTE_ST_428: return powf(V, 2.6f)*52.37f/48.f;
    }
}
static float3 fwXfrCol(float3   p,
                       unsigned cm,
                       unsigned fn,
                       unsigned fullRange,
                       unsigned YDepth,
                       unsigned CDepth){
    float Eg=p.x, Eb=p.y, Er=p.z;
    float Ey;
    float Epy, Eppb, Eppr;
    float Epr, Epg,  Epb;
    float Kr, Kb;
    float Y,  Cb, Cr;
    float R,  G,  B;
    float Nb, Pb, Nr, Pr;
    float YClamp, YSwing, YShift;
    float CClamp, CSwing, CShift;
    
    if(fullRange){
        switch(cm){
            case 0:
            case 8:
                YClamp=(1<<YDepth)-1;
                YSwing=(1<<YDepth)-1;
                YShift=0;
                CClamp=(1<<CDepth)-1;
                CSwing=(1<<CDepth)-1;
                CShift=0;
            break;
            default:
                YClamp=(1<<YDepth)-1;
                YSwing=(1<<YDepth)-1;
                YShift=0;
                CClamp=(1<<CDepth)-1;
                CSwing=(1<<CDepth)-1;
                CShift=(1<<(CDepth-1));
            break;
        }
    }else{
        switch(cm){
            case 0:
            case 8:
                YClamp=(1<<YDepth)-1;
                YSwing=219<<(YDepth-8);
                YShift= 16<<(YDepth-8);
                CClamp=(1<<CDepth)-1;
                CSwing=219<<(CDepth-8);
                CShift= 16<<(CDepth-8);
            break;
            default:
                YClamp=(1<<YDepth)-1;
                YSwing=219<<(YDepth-8);
                YShift= 16<<(YDepth-8);
                CClamp=(1<<CDepth)-1;
                CSwing=224<<(CDepth-8);
                CShift=(1<<(CDepth-1));
            break;
        }
    }
    switch(cm){
        case  1: Kr=0.2126, Kb=0.0722; break;
        case  4: Kr=0.30,   Kb=0.11;   break;
        case  5:
        case  6:
        default: Kr=0.299,  Kb=0.114;  break;
        case  7: Kr=0.212,  Kb=0.087;  break;
        case  9:
        case 10: Kr=0.2627, Kb=0.0593; break;
    }
    switch(cm){
        case  2:
            /* Unknown. Fall-through to common case. */
        case  1:
        case  4:
        case  5:
        case  6:
        case  7:
        case  9:
        default:
            Epr  = fwXfrFn(Er, fn);                                    /* E-1. Nominal range [0,1]. */
            Epg  = fwXfrFn(Eg, fn);                                    /* E-2. Nominal range [0,1]. */
            Epb  = fwXfrFn(Eb, fn);                                    /* E-3. Nominal range [0,1]. */
            Epy  = Kr*Epr + (1-Kr-Kb)*Epg + Kb*Epb;                    /* E-16 */
            Eppb = 0.5f*(Epb-Epy)/(1-Kb);                              /* E-17 */
            Eppr = 0.5f*(Epr-Epy)/(1-Kr);                              /* E-18 */
            Y    = fclampf(0, YSwing*Epy  + YShift, YClamp);           /* E-4 */
            Cb   = fclampf(0, CSwing*Eppb + CShift, CClamp);           /* E-5 */
            Cr   = fclampf(0, CSwing*Eppr + CShift, CClamp);           /* E-6 */
        break;
        case 10:
            Nb   = fwXfrFn(1-Kb, fn);                                  /* E-43 */
            Pb   = 1-fwXfrFn(Kb, fn);                                  /* E-44 */
            Nr   = fwXfrFn(1-Kr, fn);                                  /* E-45 */
            Pr   = 1-fwXfrFn(Kr, fn);                                  /* E-46 */
            Ey   = Kr*Er + (1-Kr-Kb)*Eg + Kb*Eb;                       /* E-37 */
            Epy  = fwXfrFn(Ey, fn);                                    /* E-38 */
            Epr  = fwXfrFn(Er, fn);                                    /* E-1. Nominal range [0,1]. */
            Epb  = fwXfrFn(Eb, fn);                                    /* E-3. Nominal range [0,1]. */
            Eppb = (Epb-Epy)<0 ? (Epb-Epy)/(2*Nb) : (Epb-Epy)/(2*Pb);  /* E-39, E-40 */
            Eppr = (Epr-Epy)<0 ? (Epr-Epy)/(2*Nr) : (Epr-Epy)/(2*Pr);  /* E-41, E-42 */
            Y    = fclampf(0, YSwing*Epy  + YShift, YClamp);           /* E-4 */
            Cb   = fclampf(0, CSwing*Eppb + CShift, CClamp);           /* E-5 */
            Cr   = fclampf(0, CSwing*Eppr + CShift, CClamp);           /* E-6 */
        break;
        case 11:
            Epr  = fwXfrFn(Er, fn);                                    /* E-1. Nominal range [0,1]. */
            Epg  = fwXfrFn(Eg, fn);                                    /* E-2. Nominal range [0,1]. */
            Epb  = fwXfrFn(Eb, fn);                                    /* E-3. Nominal range [0,1]. */
            Epy  = Epg;                                                /* E-47. Y'  */
            Eppb = (0.986566f*Epb - Epy) * 0.5f;                       /* E-48. D'z */
            Eppr = (Epr - 0.991902f*Epy) * 0.5f;                       /* E-49. D'x */
            Y    = fclampf(0, YSwing*Epy  + YShift, YClamp);           /* E-4 */
            Cb   = fclampf(0, CSwing*Eppb + CShift, CClamp);           /* E-5 */
            Cr   = fclampf(0, CSwing*Eppr + CShift, CClamp);           /* E-6 */
        break;
        case  0:
            Epr  = fwXfrFn(Er, fn);                                    /* E-1. Nominal range [0,1]. */
            Epg  = fwXfrFn(Eg, fn);                                    /* E-2. Nominal range [0,1]. */
            Epb  = fwXfrFn(Eb, fn);                                    /* E-3. Nominal range [0,1]. */
            R    = fclampf(0, YSwing*Epr + YShift, YClamp);            /* E-7 */
            G    = fclampf(0, CSwing*Epg + CShift, CClamp);            /* E-8 */
            B    = fclampf(0, CSwing*Epb + CShift, CClamp);            /* E-9 */
            Y    = G;                                                  /* E-19 */
            Cb   = B;                                                  /* E-20 */
            Cr   = R;                                                  /* E-21 */
        break;
        case  8:
            Epr  = fwXfrFn(Er, fn);                                    /* E-1. Nominal range [0,1]. */
            Epg  = fwXfrFn(Eg, fn);                                    /* E-2. Nominal range [0,1]. */
            Epb  = fwXfrFn(Eb, fn);                                    /* E-3. Nominal range [0,1]. */
            R    = fclampf(0, YSwing*Epr + YShift, YClamp);            /* E-7 */
            G    = fclampf(0, YSwing*Epg + YShift, CClamp);            /* E-8 */
            B    = fclampf(0, YSwing*Epb + YShift, CClamp);            /* E-9 */
            Y    = 0.5f*G + 0.25f*(R+B);                               /* E-22. Y  */
            Cb   = 0.5f*G - 0.25f*(R+B) + (1<<(CDepth-1));             /* E-23. Cg */
            Cr   = 0.5f*(R-B)           + (1<<(CDepth-1));             /* E-24. Co */
        break;
    }
    
    return make_float3(Y, Cb, Cr);
}
static float3 bwXfrCol(float3 p,
                       unsigned cm,
                       unsigned fn,
                       unsigned fullRange,
                       unsigned YDepth,
                       unsigned CDepth){
    float Y=p.x,  Cb=p.y, Cr=p.z;
    float Epy, Eppb, Eppr;
    float Epr, Epg,  Epb;
    float Er,  Eg,   Eb;
    float Ey;
    float Kr, Kb;
    float R,  G,  B,  t;
    float Nb, Pb, Nr, Pr;
    float YSwing, YShift;
    float CSwing, CShift;
    
    if(fullRange){
        switch(cm){
            case 0:
            case 8:
                YSwing=(1<<YDepth)-1;
                YShift=0;
                CSwing=(1<<CDepth)-1;
                CShift=0;
            break;
            default:
                YSwing=(1<<YDepth)-1;
                YShift=0;
                CSwing=(1<<CDepth)-1;
                CShift=(1<<(CDepth-1));
            break;
        }
    }else{
        switch(cm){
            case 0:
            case 8:
                YSwing=219<<(YDepth-8);
                YShift= 16<<(YDepth-8);
                CSwing=219<<(CDepth-8);
                CShift= 16<<(CDepth-8);
            break;
            default:
                YSwing=219<<(YDepth-8);
                YShift= 16<<(YDepth-8);
                CSwing=224<<(CDepth-8);
                CShift=(1<<(CDepth-1));
            break;
        }
    }
    switch(cm){
        case  1: Kr=0.2126, Kb=0.0722; break;
        case  4: Kr=0.30,   Kb=0.11;   break;
        case  5:
        case  6:
        default: Kr=0.299,  Kb=0.114;  break;
        case  7: Kr=0.212,  Kb=0.087;  break;
        case  9:
        case 10: Kr=0.2627, Kb=0.0593; break;
    }
    switch(cm){
        case  2:
            /* Unknown. Fall-through to common case. */
        case  1:
        case  4:
        case  5:
        case  6:
        case  7:
        case  9:
        default:
            Epy  = fclampf(0, (Y -YShift)/YSwing, 1);                  /* E-4 */
            Eppb = fclampf(0, (Cb-CShift)/CSwing, 1);                  /* E-5 */
            Eppr = fclampf(0, (Cr-CShift)/CSwing, 1);                  /* E-6 */
            Epr  = Epy + (1-Kr)*2*Eppr;                                /* E-16,17,18 Inverse */
            Epg  = Epy - (1-Kb)*Kb/(1-Kb-Kr)*2*Eppb                    /* E-16,17,18 Inverse */
                       - (1-Kr)*Kr/(1-Kb-Kr)*2*Eppr;                   /* E-16,17,18 Inverse */
            Epb  = Epy + (1-Kb)*2*Eppb;                                /* E-16,17,18 Inverse */
            Er   = bwXfrFn(Epr, fn);                                   /* E-1. Nominal range [0,1]. */
            Eg   = bwXfrFn(Epg, fn);                                   /* E-2. Nominal range [0,1]. */
            Eb   = bwXfrFn(Epb, fn);                                   /* E-3. Nominal range [0,1]. */
        break;
        case 10:
            Nb   = fwXfrFn(1-Kb, fn);                                  /* E-43 */
            Pb   = 1-fwXfrFn(Kb, fn);                                  /* E-44 */
            Nr   = fwXfrFn(1-Kr, fn);                                  /* E-45 */
            Pr   = 1-fwXfrFn(Kr, fn);                                  /* E-46 */
            Epy  = fclampf(0, (Y -YShift)/YSwing, 1);                  /* E-4 */
            Eppb = fclampf(0, (Cb-CShift)/CSwing, 1);                  /* E-5 */
            Eppr = fclampf(0, (Cr-CShift)/CSwing, 1);                  /* E-6 */
            Epb  = Eppb<0 ? Eppb*2*Nb+Epy : Eppb*2*Pb+Epy;             /* E-39, E-40 */
            Epr  = Eppr<0 ? Eppr*2*Nr+Epy : Eppr*2*Pr+Epy;             /* E-39, E-40 */
            Ey   = bwXfrFn(Epy, fn);                                   /* E-38 */
            Er   = bwXfrFn(Epr, fn);                                   /* E-1. Nominal range [0,1]. */
            Eb   = bwXfrFn(Epb, fn);                                   /* E-3. Nominal range [0,1]. */
            Eg   = (Ey - Kr*Er - Kb*Eb)/(1-Kr-Kb);                     /* E-37 */
        break;
        case 11:
            Epy  = fclampf(0, (Y -YShift)/YSwing, 1);                  /* E-4 */
            Eppb = fclampf(0, (Cb-CShift)/CSwing, 1);                  /* E-5 */
            Eppr = fclampf(0, (Cr-CShift)/CSwing, 1);                  /* E-6 */
            Epg  = Epy;                                                /* E-47. Y'  */
            Epb  = (2*Eppb + Epg)/0.986566f;                           /* E-48. D'z */
            Epr  = 2*Eppr + 0.991902f*Epg;                             /* E-49. D'x */
            Er   = bwXfrFn(Epr, fn);                                   /* E-1. Nominal range [0,1]. */
            Eg   = bwXfrFn(Epg, fn);                                   /* E-2. Nominal range [0,1]. */
            Eb   = bwXfrFn(Epb, fn);                                   /* E-3. Nominal range [0,1]. */
        break;
        case  0:
            G    = Y;                                                  /* E-19 */
            B    = Cb;                                                 /* E-20 */
            R    = Cr;                                                 /* E-21 */
            Epr  = fclampf(0, (R-YShift)/YSwing, 1);                   /* E-7 */
            Epg  = fclampf(0, (G-YShift)/YSwing, 1);                   /* E-8 */
            Epb  = fclampf(0, (B-YShift)/YSwing, 1);                   /* E-9 */
            Er   = bwXfrFn(Epr, fn);                                   /* E-1. Nominal range [0,1]. */
            Eg   = bwXfrFn(Epg, fn);                                   /* E-2. Nominal range [0,1]. */
            Eb   = bwXfrFn(Epb, fn);                                   /* E-3. Nominal range [0,1]. */
        break;
        case  8:
            t    = Y - (Cb-(1<<(CDepth-1)));                           /* E-25 */
            G    = fclampf(0, Y+(Cb-(1<<(CDepth-1))), ((1<<CDepth)-1));/* E-26 */
            B    = fclampf(0, t-(Cr-(1<<(CDepth-1))), ((1<<CDepth)-1));/* E-27 */
            R    = fclampf(0, t+(Cr-(1<<(CDepth-1))), ((1<<CDepth)-1));/* E-28 */
            Epr  = fclampf(0, (R-YShift)/YSwing, 1);                   /* E-7 */
            Epg  = fclampf(0, (G-YShift)/YSwing, 1);                   /* E-8 */
            Epb  = fclampf(0, (B-YShift)/YSwing, 1);                   /* E-9 */
            Er   = bwXfrFn(Epr, fn);                                   /* E-1. Nominal range [0,1]. */
            Eg   = bwXfrFn(Epg, fn);                                   /* E-2. Nominal range [0,1]. */
            Eb   = bwXfrFn(Epb, fn);                                   /* E-3. Nominal range [0,1]. */
        break;
    }
    
    return make_float3(Eg, Eb, Er);
}




/**
 * @brief Conversion Kernels
 */

/**
 * @brief Conversion from gbrp to yuv420p
 * 
 * Every thread handles a 2x2 pixel block, due to yuv420p's inherent chroma
 * subsampling.
 */

template<int  DST_CM         = AVCOL_SPC_BT470BG,
         int  SRC_CM         = AVCOL_SPC_BT470BG,
         int  FN             = AVCOL_TRC_IEC61966_2_1,
         bool SRC_FULL_SWING = true,
         bool DST_FULL_SWING = true,
         bool LERP           = true,
         unsigned SRC_YDEPTH = 8,
         unsigned SRC_CDEPTH = 8,
         unsigned DST_YDEPTH = 8,
         unsigned DST_CDEPTH = 8>
static __global__ void KERNEL_BOUNDS
i2hKernel_gbrp_to_yuv420p(void* __restrict const dstYPtr,
                          void* __restrict const dstCbPtr,
                          void* __restrict const dstCrPtr,
                          const unsigned         dstH,
                          const unsigned         dstW,
                          const unsigned         embedX, const unsigned embedY, const unsigned embedW, const unsigned embedH,
                          const unsigned         cropX,  const unsigned cropY,  const unsigned cropW,  const unsigned cropH,
                          const float            kx,     const float    ky,
                          const float            subX0,  const float    subY0,
                          const float            subX1,  const float    subY1,
                          const float            subX2,  const float    subY2,
                          const float            offX0,  const float    offY0,
                          const float            offX1,  const float    offY1,
                          const float            offX2,  const float    offY2,
                          cudaTextureObject_t    srcGPlane,
                          cudaTextureObject_t    srcBPlane,
                          cudaTextureObject_t    srcRPlane){
    /* Compute Destination Coordinates */
    const unsigned x = 2*(blockDim.x*blockIdx.x + threadIdx.x);
    const unsigned y = 2*(blockDim.y*blockIdx.y + threadIdx.y);
    if(x >= dstW || y >= dstH){return;}
    
    /* Compute Destination Y,U,V Pointers */
    uchar1* const __restrict__ dstPtr0 = (uchar1*)dstYPtr  + (y/1)*dstW/1 + (x/1);
    uchar1* const __restrict__ dstPtr1 = (uchar1*)dstCbPtr + (y/2)*dstW/2 + (x/2);
    uchar1* const __restrict__ dstPtr2 = (uchar1*)dstCrPtr + (y/2)*dstW/2 + (x/2);
    
    /**
     * Compute Master Quad Source Coordinates
     * 
     * The points (X0,Y0), (X1,Y1), (X2,Y2), (X3,Y3) represent the exact
     * coordinates at which we must sample for the Y samples 0,1,2,3 respectively.
     * The Cb/Cr sample corresponding to this quad is interpolated from the
     * samples drawn above.
     * 
     * The destination image is padded and the source image is cropped. This
     * means that when starting from the raw destination coordinates, we must:
     * 
     *    1. Subtract (embedX, embedY), since those are the destination embedding
     *       offsets in the target buffer.
     *    2. Clamp to (embedW-1, embedH-1), since those are the true destination
     *       bounds into the target buffer and any pixel beyond that limit
     *       should replicate the value found at the edge.
     *    3. Scale by the factors (kx, ky).
     *    4. Clamp to (cropW-1, cropH-1), since those are the bounds of the
     *       cropped source image.
     *    5. Add (cropX, cropY), since those are the source offsets into the
     *       cropped source buffer that should correspond to (embedX, embedY)
     *       in the destination buffer.
     */
    
    /* Plane 0: Y */
    const float X0p0=(fclampf(0, fclampf(0, x-embedX+0, embedW-1)*kx, cropW-1)+cropX)*subX0+offX0;
    const float X1p0=(fclampf(0, fclampf(0, x-embedX+1, embedW-1)*kx, cropW-1)+cropX)*subX0+offX0;
    const float X2p0=(fclampf(0, fclampf(0, x-embedX+0, embedW-1)*kx, cropW-1)+cropX)*subX0+offX0;
    const float X3p0=(fclampf(0, fclampf(0, x-embedX+1, embedW-1)*kx, cropW-1)+cropX)*subX0+offX0;
    const float Y0p0=(fclampf(0, fclampf(0, y-embedY+0, embedH-1)*ky, cropH-1)+cropY)*subY0+offY0;
    const float Y1p0=(fclampf(0, fclampf(0, y-embedY+0, embedH-1)*ky, cropH-1)+cropY)*subY0+offY0;
    const float Y2p0=(fclampf(0, fclampf(0, y-embedY+1, embedH-1)*ky, cropH-1)+cropY)*subY0+offY0;
    const float Y3p0=(fclampf(0, fclampf(0, y-embedY+1, embedH-1)*ky, cropH-1)+cropY)*subY0+offY0;
    /* Plane 1: Cb */
    const float X0p1=(fclampf(0, fclampf(0, x-embedX+0, embedW-1)*kx, cropW-1)+cropX)*subX1+offX1;
    const float X1p1=(fclampf(0, fclampf(0, x-embedX+1, embedW-1)*kx, cropW-1)+cropX)*subX1+offX1;
    const float X2p1=(fclampf(0, fclampf(0, x-embedX+0, embedW-1)*kx, cropW-1)+cropX)*subX1+offX1;
    const float X3p1=(fclampf(0, fclampf(0, x-embedX+1, embedW-1)*kx, cropW-1)+cropX)*subX1+offX1;
    const float Y0p1=(fclampf(0, fclampf(0, y-embedY+0, embedH-1)*ky, cropH-1)+cropY)*subY1+offY1;
    const float Y1p1=(fclampf(0, fclampf(0, y-embedY+0, embedH-1)*ky, cropH-1)+cropY)*subY1+offY1;
    const float Y2p1=(fclampf(0, fclampf(0, y-embedY+1, embedH-1)*ky, cropH-1)+cropY)*subY1+offY1;
    const float Y3p1=(fclampf(0, fclampf(0, y-embedY+1, embedH-1)*ky, cropH-1)+cropY)*subY1+offY1;
    /* Plane 2: Cr */
    const float X0p2=(fclampf(0, fclampf(0, x-embedX+0, embedW-1)*kx, cropW-1)+cropX)*subX2+offX2;
    const float X1p2=(fclampf(0, fclampf(0, x-embedX+1, embedW-1)*kx, cropW-1)+cropX)*subX2+offX2;
    const float X2p2=(fclampf(0, fclampf(0, x-embedX+0, embedW-1)*kx, cropW-1)+cropX)*subX2+offX2;
    const float X3p2=(fclampf(0, fclampf(0, x-embedX+1, embedW-1)*kx, cropW-1)+cropX)*subX2+offX2;
    const float Y0p2=(fclampf(0, fclampf(0, y-embedY+0, embedH-1)*ky, cropH-1)+cropY)*subY2+offY2;
    const float Y1p2=(fclampf(0, fclampf(0, y-embedY+0, embedH-1)*ky, cropH-1)+cropY)*subY2+offY2;
    const float Y2p2=(fclampf(0, fclampf(0, y-embedY+1, embedH-1)*ky, cropH-1)+cropY)*subY2+offY2;
    const float Y3p2=(fclampf(0, fclampf(0, y-embedY+1, embedH-1)*ky, cropH-1)+cropY)*subY2+offY2;
    
    
    /* Compute and output pixel depending on interpolation policy. */
    if(LERP){
        /**
         * We will perform linear interpolation manually between sample points.
         * For this reason we need the coordinates and values of 4 samples for
         * each of the 4 YUV pixels this thread is computing. In ASCII art:
         * 
         *  (0,0)
         *      +-----------> X
         *      |           Xb
         *      |
         *      |
         *      v Yb        0  1
         *      Y             P
         *                  2  3
         * 
         * We must consider three planes separately:
         * 
         *   - Plane 0 is luma                   or G
         *   - Plane 1 is chroma blue-difference or B
         *   - Plane 2 is chroma red -difference or R
         * 
         * If sample i's plane j sample is at (Xipj, Yipj), the quad used to
         * manually interpolate it has coordinates
         * 
         *   - (Xipjq0, Yipjq0)
         *   - (Xipjq1, Yipjq1)
         *   - (Xipjq2, Yipjq2)
         *   - (Xipjq3, Yipjq3)
         * 
         * And the interpolation within the quad is done using the scalar pair
         * 
         *   - (Xipjf, Yipjf)
         * 
         * or for an entire pixel, with the vector pair
         * 
         *   - (Xipf, Yipf)
         */
        
        const float X0p0b=floorf(X0p0), X0p0f=X0p0-X0p0b;
        const float X1p0b=floorf(X1p0), X1p0f=X1p0-X1p0b;
        const float X2p0b=floorf(X2p0), X2p0f=X2p0-X2p0b;
        const float X3p0b=floorf(X3p0), X3p0f=X3p0-X3p0b;
        const float X0p1b=floorf(X0p1), X0p1f=X0p1-X0p1b;
        const float X1p1b=floorf(X1p1), X1p1f=X1p1-X1p1b;
        const float X2p1b=floorf(X2p1), X2p1f=X2p1-X2p1b;
        const float X3p1b=floorf(X3p1), X3p1f=X3p1-X3p1b;
        const float X0p2b=floorf(X0p2), X0p2f=X0p2-X0p2b;
        const float X1p2b=floorf(X1p2), X1p2f=X1p2-X1p2b;
        const float X2p2b=floorf(X2p2), X2p2f=X2p2-X2p2b;
        const float X3p2b=floorf(X3p2), X3p2f=X3p2-X3p2b;
        
        const float Y0p0b=floorf(Y0p0), Y0p0f=Y0p0-Y0p0b;
        const float Y1p0b=floorf(Y1p0), Y1p0f=Y1p0-Y1p0b;
        const float Y2p0b=floorf(Y2p0), Y2p0f=Y2p0-Y2p0b;
        const float Y3p0b=floorf(Y3p0), Y3p0f=Y3p0-Y3p0b;
        const float Y0p1b=floorf(Y0p1), Y0p1f=Y0p1-Y0p1b;
        const float Y1p1b=floorf(Y1p1), Y1p1f=Y1p1-Y1p1b;
        const float Y2p1b=floorf(Y2p1), Y2p1f=Y2p1-Y2p1b;
        const float Y3p1b=floorf(Y3p1), Y3p1f=Y3p1-Y3p1b;
        const float Y0p2b=floorf(Y0p2), Y0p2f=Y0p2-Y0p2b;
        const float Y1p2b=floorf(Y1p2), Y1p2f=Y1p2-Y1p2b;
        const float Y2p2b=floorf(Y2p2), Y2p2f=Y2p2-Y2p2b;
        const float Y3p2b=floorf(Y3p2), Y3p2f=Y3p2-Y3p2b;
        
        
        /* Sample from texture at the computed coordinates. */
        const float3 srcPix00 = make_float3(tex2D<unsigned char>(srcGPlane, X0p0b+0, Y0p0b+0), tex2D<unsigned char>(srcBPlane, X0p1b+0, Y0p1b+0), tex2D<unsigned char>(srcRPlane, X0p2b+0, Y0p2b+0));
        const float3 srcPix01 = make_float3(tex2D<unsigned char>(srcGPlane, X0p0b+1, Y0p0b+0), tex2D<unsigned char>(srcBPlane, X0p1b+1, Y0p1b+0), tex2D<unsigned char>(srcRPlane, X0p2b+1, Y0p2b+0));
        const float3 srcPix02 = make_float3(tex2D<unsigned char>(srcGPlane, X0p0b+0, Y0p0b+1), tex2D<unsigned char>(srcBPlane, X0p1b+0, Y0p1b+1), tex2D<unsigned char>(srcRPlane, X0p2b+0, Y0p2b+1));
        const float3 srcPix03 = make_float3(tex2D<unsigned char>(srcGPlane, X0p0b+1, Y0p0b+1), tex2D<unsigned char>(srcBPlane, X0p1b+1, Y0p1b+1), tex2D<unsigned char>(srcRPlane, X0p2b+1, Y0p2b+1));
        const float3 srcPix10 = make_float3(tex2D<unsigned char>(srcGPlane, X1p0b+0, Y1p0b+0), tex2D<unsigned char>(srcBPlane, X1p1b+0, Y1p1b+0), tex2D<unsigned char>(srcRPlane, X1p2b+0, Y1p2b+0));
        const float3 srcPix11 = make_float3(tex2D<unsigned char>(srcGPlane, X1p0b+1, Y1p0b+0), tex2D<unsigned char>(srcBPlane, X1p1b+1, Y1p1b+0), tex2D<unsigned char>(srcRPlane, X1p2b+1, Y1p2b+0));
        const float3 srcPix12 = make_float3(tex2D<unsigned char>(srcGPlane, X1p0b+0, Y1p0b+1), tex2D<unsigned char>(srcBPlane, X1p1b+0, Y1p1b+1), tex2D<unsigned char>(srcRPlane, X1p2b+0, Y1p2b+1));
        const float3 srcPix13 = make_float3(tex2D<unsigned char>(srcGPlane, X1p0b+1, Y1p0b+1), tex2D<unsigned char>(srcBPlane, X1p1b+1, Y1p1b+1), tex2D<unsigned char>(srcRPlane, X1p2b+1, Y1p2b+1));
        const float3 srcPix20 = make_float3(tex2D<unsigned char>(srcGPlane, X2p0b+0, Y2p0b+0), tex2D<unsigned char>(srcBPlane, X2p1b+0, Y2p1b+0), tex2D<unsigned char>(srcRPlane, X2p2b+0, Y2p2b+0));
        const float3 srcPix21 = make_float3(tex2D<unsigned char>(srcGPlane, X2p0b+1, Y2p0b+0), tex2D<unsigned char>(srcBPlane, X2p1b+1, Y2p1b+0), tex2D<unsigned char>(srcRPlane, X2p2b+1, Y2p2b+0));
        const float3 srcPix22 = make_float3(tex2D<unsigned char>(srcGPlane, X2p0b+0, Y2p0b+1), tex2D<unsigned char>(srcBPlane, X2p1b+0, Y2p1b+1), tex2D<unsigned char>(srcRPlane, X2p2b+0, Y2p2b+1));
        const float3 srcPix23 = make_float3(tex2D<unsigned char>(srcGPlane, X2p0b+1, Y2p0b+1), tex2D<unsigned char>(srcBPlane, X2p1b+1, Y2p1b+1), tex2D<unsigned char>(srcRPlane, X2p2b+1, Y2p2b+1));
        const float3 srcPix30 = make_float3(tex2D<unsigned char>(srcGPlane, X3p0b+0, Y3p0b+0), tex2D<unsigned char>(srcBPlane, X3p1b+0, Y3p1b+0), tex2D<unsigned char>(srcRPlane, X3p2b+0, Y3p2b+0));
        const float3 srcPix31 = make_float3(tex2D<unsigned char>(srcGPlane, X3p0b+1, Y3p0b+0), tex2D<unsigned char>(srcBPlane, X3p1b+1, Y3p1b+0), tex2D<unsigned char>(srcRPlane, X3p2b+1, Y3p2b+0));
        const float3 srcPix32 = make_float3(tex2D<unsigned char>(srcGPlane, X3p0b+0, Y3p0b+1), tex2D<unsigned char>(srcBPlane, X3p1b+0, Y3p1b+1), tex2D<unsigned char>(srcRPlane, X3p2b+0, Y3p2b+1));
        const float3 srcPix33 = make_float3(tex2D<unsigned char>(srcGPlane, X3p0b+1, Y3p0b+1), tex2D<unsigned char>(srcBPlane, X3p1b+1, Y3p1b+1), tex2D<unsigned char>(srcRPlane, X3p2b+1, Y3p2b+1));
        
        
        /* Backward Colorspace and Transfer (EOTF). */
        const float3 linPix00 = bwXfrCol(srcPix00, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix01 = bwXfrCol(srcPix01, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix02 = bwXfrCol(srcPix02, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix03 = bwXfrCol(srcPix03, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix10 = bwXfrCol(srcPix10, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix11 = bwXfrCol(srcPix11, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix12 = bwXfrCol(srcPix12, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix13 = bwXfrCol(srcPix13, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix20 = bwXfrCol(srcPix20, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix21 = bwXfrCol(srcPix21, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix22 = bwXfrCol(srcPix22, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix23 = bwXfrCol(srcPix23, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix30 = bwXfrCol(srcPix30, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix31 = bwXfrCol(srcPix31, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix32 = bwXfrCol(srcPix32, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix33 = bwXfrCol(srcPix33, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        
        
        /* Linear Interpolation */
        const float3 X0pf     = make_float3(X0p0f, X0p1f, X0p2f);
        const float3 X1pf     = make_float3(X1p0f, X1p1f, X1p2f);
        const float3 X2pf     = make_float3(X2p0f, X2p1f, X2p2f);
        const float3 X3pf     = make_float3(X3p0f, X3p1f, X3p2f);
        const float3 Y0pf     = make_float3(Y0p0f, Y0p1f, Y0p2f);
        const float3 Y1pf     = make_float3(Y1p0f, Y1p1f, Y1p2f);
        const float3 Y2pf     = make_float3(Y2p0f, Y2p1f, Y2p2f);
        const float3 Y3pf     = make_float3(Y3p0f, Y3p1f, Y3p2f);
        const float3 linPixD0 = lerpv3(lerpv3(linPix00, linPix01, X0pf), lerpv3(linPix02, linPix03, X0pf), Y0pf);
        const float3 linPixD1 = lerpv3(lerpv3(linPix10, linPix11, X1pf), lerpv3(linPix12, linPix13, X1pf), Y1pf);
        const float3 linPixD2 = lerpv3(lerpv3(linPix20, linPix21, X2pf), lerpv3(linPix22, linPix23, X2pf), Y2pf);
        const float3 linPixD3 = lerpv3(lerpv3(linPix30, linPix31, X3pf), lerpv3(linPix32, linPix33, X3pf), Y3pf);
        const float3 linPixD4 = lerps3(lerps3(linPixD0, linPixD1, 0.5f), lerps3(linPixD2, linPixD3, 0.5f), 0.5f);
        
        
        /* Forward Colorspace and Transfer (OETF). */
        const float3 dstPixD0 = fwXfrCol(linPixD0, DST_CM, FN, DST_FULL_SWING, DST_YDEPTH, DST_CDEPTH);
        const float3 dstPixD1 = fwXfrCol(linPixD1, DST_CM, FN, DST_FULL_SWING, DST_YDEPTH, DST_CDEPTH);
        const float3 dstPixD2 = fwXfrCol(linPixD2, DST_CM, FN, DST_FULL_SWING, DST_YDEPTH, DST_CDEPTH);
        const float3 dstPixD3 = fwXfrCol(linPixD3, DST_CM, FN, DST_FULL_SWING, DST_YDEPTH, DST_CDEPTH);
        const float3 dstPixD4 = fwXfrCol(linPixD4, DST_CM, FN, DST_FULL_SWING, DST_YDEPTH, DST_CDEPTH);
        
        
        /* Store YUV. */
        dstPtr0[0*dstW/1/1+0]   = make_uchar1(roundf(fclampf(0.f, dstPixD0.x, (1<<DST_YDEPTH)-1)));
        dstPtr0[0*dstW/1/1+1]   = make_uchar1(roundf(fclampf(0.f, dstPixD1.x, (1<<DST_YDEPTH)-1)));
        dstPtr0[1*dstW/1/1+0]   = make_uchar1(roundf(fclampf(0.f, dstPixD2.x, (1<<DST_YDEPTH)-1)));
        dstPtr0[1*dstW/1/1+1]   = make_uchar1(roundf(fclampf(0.f, dstPixD3.x, (1<<DST_YDEPTH)-1)));
        dstPtr1[0*dstW/2/2+0]   = make_uchar1(roundf(fclampf(0.f, dstPixD4.y, (1<<DST_CDEPTH)-1)));
        dstPtr2[0*dstW/2/2+0]   = make_uchar1(roundf(fclampf(0.f, dstPixD4.z, (1<<DST_CDEPTH)-1)));
    }else{
        /**
         * We perform nearest-sample point fetch for 4 Y, 1 Cb and 1 Cr samples.
         * 
         * Since YUV420 is 2x2 chroma-subsampled, we select the top left
         * pixel's chroma information, given that we're forbidden to interpolate.
         * 
         * We must consider three planes separately:
         * 
         *   - Plane 0 is luma                   or G
         *   - Plane 1 is chroma blue-difference or B
         *   - Plane 2 is chroma red -difference or R
         */
        
        const float X0p0b=roundf(X0p0);
        const float X1p0b=roundf(X1p0);
        const float X2p0b=roundf(X2p0);
        const float X3p0b=roundf(X3p0);
        const float X0p1b=roundf(X0p1);
        const float X1p1b=roundf(X1p1);
        const float X2p1b=roundf(X2p1);
        const float X3p1b=roundf(X3p1);
        const float X0p2b=roundf(X0p2);
        const float X1p2b=roundf(X1p2);
        const float X2p2b=roundf(X2p2);
        const float X3p2b=roundf(X3p2);
        
        const float Y0p0b=roundf(Y0p0);
        const float Y1p0b=roundf(Y1p0);
        const float Y2p0b=roundf(Y2p0);
        const float Y3p0b=roundf(Y3p0);
        const float Y0p1b=roundf(Y0p1);
        const float Y1p1b=roundf(Y1p1);
        const float Y2p1b=roundf(Y2p1);
        const float Y3p1b=roundf(Y3p1);
        const float Y0p2b=roundf(Y0p2);
        const float Y1p2b=roundf(Y1p2);
        const float Y2p2b=roundf(Y2p2);
        const float Y3p2b=roundf(Y3p2);
        
        
        /* Sample from texture at the computed coordinates. */
        const float3 srcPix0 = make_float3(tex2D<unsigned char>(srcGPlane, X0p0b+0, Y0p0b+0), tex2D<unsigned char>(srcBPlane, X0p1b+0, Y0p1b+0), tex2D<unsigned char>(srcRPlane, X0p2b+0, Y0p2b+0));
        const float3 srcPix1 = make_float3(tex2D<unsigned char>(srcGPlane, X1p0b+1, Y1p0b+0), tex2D<unsigned char>(srcBPlane, X1p1b+1, Y1p1b+0), tex2D<unsigned char>(srcRPlane, X1p2b+1, Y1p2b+0));
        const float3 srcPix2 = make_float3(tex2D<unsigned char>(srcGPlane, X2p0b+0, Y2p0b+1), tex2D<unsigned char>(srcBPlane, X2p1b+0, Y2p1b+1), tex2D<unsigned char>(srcRPlane, X2p2b+0, Y2p2b+1));
        const float3 srcPix3 = make_float3(tex2D<unsigned char>(srcGPlane, X3p0b+1, Y3p0b+1), tex2D<unsigned char>(srcBPlane, X3p1b+1, Y3p1b+1), tex2D<unsigned char>(srcRPlane, X3p2b+1, Y3p2b+1));
        
        
        /* Backward Colorspace and Transfer (EOTF). */
        const float3 linPix0 = bwXfrCol(srcPix0, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix1 = bwXfrCol(srcPix1, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix2 = bwXfrCol(srcPix2, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        const float3 linPix3 = bwXfrCol(srcPix3, SRC_CM, FN, SRC_FULL_SWING, SRC_YDEPTH, SRC_CDEPTH);
        
        
        /* Forward Colorspace and Transfer (OETF). */
        const float3 dstPix0 = fwXfrCol(linPix0, DST_CM, FN, DST_FULL_SWING, DST_YDEPTH, DST_CDEPTH);
        const float3 dstPix1 = fwXfrCol(linPix1, DST_CM, FN, DST_FULL_SWING, DST_YDEPTH, DST_CDEPTH);
        const float3 dstPix2 = fwXfrCol(linPix2, DST_CM, FN, DST_FULL_SWING, DST_YDEPTH, DST_CDEPTH);
        const float3 dstPix3 = fwXfrCol(linPix3, DST_CM, FN, DST_FULL_SWING, DST_YDEPTH, DST_CDEPTH);
        
        
        /* Store YUV. */
        dstPtr0[0*dstW/1/1+0]   = make_uchar1(roundf(fclampf(0.f, dstPix0.x, (1<<DST_YDEPTH)-1)));
        dstPtr0[0*dstW/1/1+1]   = make_uchar1(roundf(fclampf(0.f, dstPix1.x, (1<<DST_YDEPTH)-1)));
        dstPtr0[1*dstW/1/1+0]   = make_uchar1(roundf(fclampf(0.f, dstPix2.x, (1<<DST_YDEPTH)-1)));
        dstPtr0[1*dstW/1/1+1]   = make_uchar1(roundf(fclampf(0.f, dstPix3.x, (1<<DST_YDEPTH)-1)));
        dstPtr1[0*dstW/2/2+0]   = make_uchar1(roundf(fclampf(0.f, dstPix0.y, (1<<DST_CDEPTH)-1)));
        dstPtr2[0*dstW/2/2+0]   = make_uchar1(roundf(fclampf(0.f, dstPix0.z, (1<<DST_CDEPTH)-1)));
    }
}
#endif

/**
 * Other References:
 * 
 *   - The "Continuum Magic Kernel Sharp" method:
 * 
 *                { 0.0             ,         x <= -1.5
 *                { 0.5(x+1.5)**2   , -1.5 <= x <= -0.5
 *         m(x) = { 0.75 - (x)**2   , -0.5 <= x <= +0.5
 *                { 0.5(x-1.5)**2   , +0.5 <= x <= +1.5
 *                { 0.0             , +1.5 <= x
 *     
 *     followed by the sharpening post-filter [-0.25, +1.5, -0.25].
 *     http://www.johncostella.com/magic/
 */
