/* Includes */
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include "main.h"
#include "utils.h"


/**
 * Defines
 */



/**
 * Public Functions.
 */

extern int i2v_fixup_frame(AVFrame* f){
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

extern enum AVPixelFormat i2v_pick_pix_fmt(enum AVPixelFormat src){
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

extern int i2v_canonicalize_pixfmt(AVFrame* out, AVFrame* in){
    enum AVPixelFormat canonical     = i2v_pick_pix_fmt(in->format);
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
        i2vmessage(stderr, "Pixel format %s unsupported!\n",
                   av_get_pix_fmt_name(in->format));
        return -EINVAL;
    }
    
    
    /**
     * Graph Init
     */
    
    
    buffersrc  = avfilter_get_by_name("buffer");
    buffersink = avfilter_get_by_name("buffersink");
    if(!buffersrc || !buffersink){
        i2vmessage(stderr, "Failed to find buffer or buffersink filter objects!\n");
        return -EINVAL;
    }
    graph = avfilter_graph_alloc();
    if(!graph){
        i2vmessage(stderr, "Failed to allocate graph object!\n");
        return -ENOMEM;
    }
    avfilter_graph_set_auto_convert(graph, AVFILTER_AUTO_CONVERT_NONE);
    outputs = avfilter_inout_alloc();
    inputs  = avfilter_inout_alloc();
    if(!outputs   || !inputs){
        i2vmessage(stderr, "Failed to allocate inout objects!\n");
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
        i2vmessage(stderr, "Failed to allocate name for filter input/output!\n");
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
        i2vmessage(stderr, "Failed to allocate buffersrc filter instance!\n");
        goto fail;
    }
    ret = avfilter_graph_create_filter(&inputs->filter_ctx,
                                       buffersink,
                                       "out",
                                       NULL,
                                       NULL,
                                       graph);
    if(ret < 0){
        i2vmessage(stderr, "Failed to allocate buffersink filter instance!\n");
        goto fail;
    }
    buffersrcctx  = outputs->filter_ctx;
    buffersinkctx = inputs ->filter_ctx;
    ret = avfilter_graph_parse_ptr(graph, graphstr, &inputs, &outputs, NULL);
    if(ret < 0){
        i2vmessage(stderr, "Error parsing graph! (%d)\n", ret);
        goto fail;
    }
    ret = avfilter_graph_config(graph, NULL);
    if(ret < 0){
        i2vmessage(stderr, "Error configuring graph! (%d)\n", ret);
        goto fail;
    }
    
#if 0
    char* str = avfilter_graph_dump(graph, "");
    if(str){
        i2vmessage(stdout, "%s\n", str);
    }
    i2vmessage(stdout, "%s\n", graphstr);
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
        i2vmessage(stderr, "Error pushing frame into graph! (%d)\n", ret);
        goto fail;
    }
    ret = av_buffersink_get_frame(buffersinkctx, out);
    if(ret < 0){
        i2vmessage(stderr, "Error pulling frame from graph! (%d)\n", ret);
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

extern int i2v_is_imagelike_stream(AVFormatContext* avfctx, int stream){
    if(stream < 0 || stream >= (int)avfctx->nb_streams){
        return 0;
    }
    return avfctx->streams[stream]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO;
}

extern const char* i2v_x264_color_primaries(enum AVColorPrimaries color_primaries){
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

extern const char* i2v_x264_color_trc(enum AVColorTransferCharacteristic color_trc){
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

extern const char* i2v_x264_color_space(enum AVColorSpace color_space){
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
