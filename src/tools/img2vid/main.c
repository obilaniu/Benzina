/* Includes */
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include "main.h"
#include "kernels.h"
#include "utils.h"
#include "benzina/bits.h"


/**
 * Defines
 */



/**
 * Functions.
 */

int      strendswith(const char* s, const char* e){
    size_t ls = strlen(s), le = strlen(e);
    return ls<le ? 0 : !strcmp(s+ls-le, e);
}
int      str2chromafmt(const char* s){
    if      (!strcmp(s, "yuv400")){return BENZINA_CHROMAFMT_YUV400;
    }else if(!strcmp(s, "yuv420")){return BENZINA_CHROMAFMT_YUV420;
    }else if(!strcmp(s, "yuv422")){return BENZINA_CHROMAFMT_YUV422;
    }else if(!strcmp(s, "yuv444")){return BENZINA_CHROMAFMT_YUV444;
    }else{
        return -100;
    }
}
int      str2canvaschromafmt(const char* s){
    return !strcmp(s, "yuv420super") ? BENZINA_CHROMAFMT_YUV420 : str2chromafmt(s);
}

/**
 * @brief Handle one frame.
 * 
 * This means creating the filter graph specific to this frame and invoking it.
 */

static int  i2v_do_frame        (UNIVERSE*        u){
    i2v_fixup_pix_fmt(u->in.frame);
    u->ret = i2v_canonicalize_pixfmt(u->filt.in.frame, u->in.frame);
    if(u->ret < 0){
        i2vmessage(stderr, "Failed to canonicalize pixel format! (%d)\n", u->ret);
        return u->ret;
    }
    
    
    /**
     * Manual filtering.
     */
    
    u->ret = i2vFilter(u);
    if(u->ret < 0){
        i2vmessage(stderr, "Failed to filter image! (%d)\n", u->ret);
        return u->ret;
    }
    
    
    /**
     * Push the filtered frame  into the encoder.
     * Pull the output packet from the encoder.
     * 
     * Flushing the input encode pipeline requires pushing in a NULL frame.
     */
    
    u->filt.out.frame->pts = u->in.frame->best_effort_timestamp;
    u->ret = avcodec_send_frame    (u->out.codecCtx, u->filt.out.frame);
    if(u->ret < 0){
        i2vmessage(stderr, "Error pushing frame into encoder! (%d)\n", u->ret);
        return u->ret;
    }
    u->ret = avcodec_send_frame    (u->out.codecCtx, NULL);
    if(u->ret < 0){
        i2vmessage(stderr, "Error flushing encoder input! (%d)\n", u->ret);
        return u->ret;
    }
    u->ret = avcodec_receive_packet(u->out.codecCtx, u->out.packet);
    if(u->ret < 0){
        i2vmessage(stderr, "Error pulling packet from encoder! (%d)\n", u->ret);
        return u->ret;
    }
    
#if 1
    fwrite(u->out.packet->data, 1, u->out.packet->size, u->out.fp);
    fclose(u->out.fp);
#endif
    
    return u->ret=0;
}

/**
 * Open and close FFmpeg.
 * 
 * In-between, handle frames.
 */

static int  i2v_main            (UNIVERSE* u){
    char        x264params[1024];
    const char* colorprimStr   = NULL;
    const char* transferStr    = NULL;
    const char* colormatrixStr = NULL;
    
    
    /**
     * FFmpeg setup for transcoding.
     */
    
    /* Input Init */
    u->in.packet        = av_packet_alloc();
    if(!u->in.packet){
        i2vmessage(stderr, "Failed to allocate packet object!\n");
        goto fail;
    }
    u->in.frame         = av_frame_alloc();
    if(!u->in.frame){
        i2vmessage(stderr, "Error allocating frame!\n");
        goto fail;
    }
    u->ret              = avformat_open_input         (&u->in.formatCtx,
                                                       u->args.urlIn,
                                                       NULL, NULL);
    if(u->ret < 0){
        i2vmessage(stderr, "Cannot open input media \"%s\"! (%d)\n", u->args.urlIn, u->ret);
        goto fail;
    }
    u->ret              = avformat_find_stream_info   (u->in.formatCtx, NULL);
    if(u->ret < 0){
        i2vmessage(stderr, "Cannot understand input media \"%s\"! (%d)\n", u->args.urlIn, u->ret);
        goto fail;
    }
    u->in.formatStream  = av_find_default_stream_index(u->in.formatCtx);
    if(!i2v_is_imagelike_stream(u->in.formatCtx, u->in.formatStream)){
        i2vmessage(stderr, "No image data in media \"%s\"!\n", u->args.urlIn);
        goto fail;
    }
    u->in.codecPar      = u->in.formatCtx->streams[u->in.formatStream]->codecpar;
    u->in.codec         = avcodec_find_decoder        (u->in.codecPar->codec_id);
    u->in.codecCtx      = avcodec_alloc_context3      (u->in.codec);
    if(!u->in.codecCtx){
        i2vmessage(stderr, "Could not allocate decoder!\n");
        goto fail;
    }
    u->ret              = avcodec_open2               (u->in.codecCtx,
                                                       u->in.codec,
                                                       NULL);
    if(u->ret < 0){
        i2vmessage(stderr, "Could not open decoder! (%d)\n", u->ret);
        goto fail;
    }
    colorprimStr   = i2v_x264_color_primaries(u->in.codecPar->color_primaries);
    colorprimStr   = colorprimStr   ? colorprimStr   : i2v_x264_color_primaries(
                                      u->in.codecPar->color_primaries = AVCOL_PRI_BT709);
    transferStr    = i2v_x264_color_trc      (u->in.codecPar->color_trc);
    transferStr    = transferStr    ? transferStr    : i2v_x264_color_trc(
                                      u->in.codecPar->color_trc       = AVCOL_TRC_IEC61966_2_1);
    colormatrixStr = i2v_x264_color_space    (u->in.codecPar->color_space);
    colormatrixStr = colormatrixStr ? colormatrixStr : i2v_x264_color_space(
                                      u->in.codecPar->color_space     = AVCOL_SPC_BT470BG);
    
    
    /* Output Init */
    u->out.packet       = av_packet_alloc();
    if(!u->out.packet){
        i2vmessage(stderr, "Failed to allocate packet object!\n");
        goto fail;
    }
    u->out.codec        = avcodec_find_encoder        (AV_CODEC_ID_H264);
    if(!u->out.codec){
        i2vmessage(stderr, "Cannot transcode to H264! Please rebuild with an "
                           "FFmpeg configured with --enable-libx264.\n");
        goto fail;
    }
    u->out.codecCtx     = avcodec_alloc_context3      (u->out.codec);
    if(!u->out.codecCtx){
        i2vmessage(stderr, "Could not allocate encoder!\n");
        goto fail;
    }
    u->out.codecCtx->width                  = u->args.canvas.w;
    u->out.codecCtx->height                 = u->args.canvas.h;
    u->out.codecCtx->time_base.num          =  1;
    u->out.codecCtx->time_base.den          = 30;
    u->out.codecCtx->gop_size               =  0;
    u->out.codecCtx->pix_fmt                = AV_PIX_FMT_YUV420P;
    u->out.codecCtx->chroma_sample_location = AVCHROMA_LOC_CENTER;
    u->out.codecCtx->color_range            = AVCOL_RANGE_JPEG;
    u->out.codecCtx->color_primaries        = u->in.codecPar->color_primaries;
    u->out.codecCtx->color_trc              = u->in.codecPar->color_trc;
    u->out.codecCtx->colorspace             = u->in.codecPar->color_space;
    u->out.codecCtx->sample_aspect_ratio    = u->in.codecPar->sample_aspect_ratio;
    u->out.codecCtx->flags                 |= AV_CODEC_FLAG_LOOP_FILTER;
    u->out.codecCtx->thread_count           = 1;
    u->out.codecCtx->slices                 = 1;
    snprintf(x264params, sizeof(x264params),
             "ref=1:chromaloc=1:log=-1:slices-max=32:fullrange=on:"
             "colorprim=%s:transfer=%s:colormatrix=%s%s%s",
             colorprimStr, transferStr, colormatrixStr,
             *u->args.x264params?":":"", u->args.x264params);
    av_dict_set    (&u->out.codecCtxOpt, "profile",     "high444",    0);
    av_dict_set    (&u->out.codecCtxOpt, "preset",      "placebo",    0);
    av_dict_set    (&u->out.codecCtxOpt, "tune",        "stillimage", 0);
    av_dict_set_int(&u->out.codecCtxOpt, "forced-idr",  1,            0);
    av_dict_set_int(&u->out.codecCtxOpt, "crf",         u->args.crf,  0);
    av_dict_set    (&u->out.codecCtxOpt, "x264-params", x264params,   0);
    u->ret              = avcodec_open2               (u->out.codecCtx,
                                                       u->out.codec,
                                                       &u->out.codecCtxOpt);
    if(av_dict_count(u->out.codecCtxOpt)){
        i2vmessage(stderr, "Warning: codec ignored %d options!\n",
                   av_dict_count(u->out.codecCtxOpt));
        AVDictionaryEntry* entry = NULL;
        while((entry=av_dict_get(u->out.codecCtxOpt, "", entry, AV_DICT_IGNORE_SUFFIX))){
            i2vmessage(stderr, "    %20s: %s\n", entry->key, entry->value);
        }
    }
    av_dict_free   (&u->out.codecCtxOpt);
    if(u->ret < 0){
        i2vmessage(stderr, "Could not open encoder! (%d)\n", u->ret);
        goto fail;
    }
    
    
    /* Manual Frame Init */
    u->filt.in.frame        = av_frame_alloc();
    if(!u->filt.in.frame){
        i2vmessage(stderr, "Error allocating frame!\n");
        goto fail;
    }
    u->filt.out.frame       = av_frame_alloc();
    if(!u->filt.out.frame){
        i2vmessage(stderr, "Error allocating frame!\n");
        goto fail;
    }
    u->filt.out.frame->width           = u->out.codecCtx->width;
    u->filt.out.frame->height          = u->out.codecCtx->height;
    u->filt.out.frame->format          = u->out.codecCtx->pix_fmt;
    u->filt.out.frame->color_range     = u->out.codecCtx->color_range;
    u->filt.out.frame->color_primaries = u->out.codecCtx->color_primaries;
    u->filt.out.frame->color_trc       = u->out.codecCtx->color_trc;
    u->filt.out.frame->colorspace      = u->out.codecCtx->colorspace;
    u->ret = av_frame_get_buffer(u->filt.out.frame, 0);
    if(u->ret != 0){
        i2vmessage(stderr, "Error allocating frame! (%d)\n", u->ret);
        return u->ret;
    }
    
    
    /**
     * Input packet-reading loop.
     */
    
    do{
        u->ret = av_read_frame(u->in.formatCtx, u->in.packet);
        if(u->ret < 0){
            i2vmessage(stderr, "Error or end-of-file reading media \"%s\" (%d)!\n",
                       u->args.urlIn, u->ret);
            goto fail;
        }
        if(u->in.packet->stream_index == u->in.formatStream){
            u->ret = avcodec_send_packet  (u->in.codecCtx, u->in.packet);
            if(u->ret){
                i2vmessage(stderr, "Error pushing packet! (%d)\n", u->ret);
            }else{
                u->ret = avcodec_receive_frame(u->in.codecCtx, u->in.frame);
                if(u->ret){
                    i2vmessage(stderr, "Error pulling frame! (%d)\n",  u->ret);
                }else{
                    u->ret = i2v_do_frame(u);
                }
            }
        }
        av_packet_unref(u->in.packet);
        if(u->in.packet->stream_index == u->in.formatStream){
            break;
        }
    }while(!u->ret);
    
    
    /**
     * FFmpeg finalization.
     */
    
    fail:
    avformat_close_input(&u->in.formatCtx);
    avcodec_close(u->in.codecCtx);
    avcodec_close(u->out.codecCtx);
    av_packet_free(&u->in.packet);
    return u->ret;
}

/**
 * @brief Quick arguments parsing.
 * 
 * Everything after a double-dash (--) is taken not to be an option.
 */

static void i2v_parse_args      (UNIVERSE*        u,
                                 int              argc,
                                 char*            argv[]){
    int i, dashdashseen=0, conv=0;
    char canvasChromaFmt[21] = "yuv420";
    
    
    /* Defaults */
    u->args.urlOut            =       NULL;
    u->args.canvas.w          =       1024;
    u->args.canvas.h          =        512;
    u->args.canvas.superscale =          1;
    u->args.canvas.chroma_fmt = BENZINA_CHROMAFMT_YUV420;
    u->args.crf               =         10;
    u->args.urlOut            = "out.h264";
    u->args.x264params        =         "";
    
    
    /* Parsing */
    for(i=1;i<argc;i++){
        if(dashdashseen){
            u->args.urlIn = u->args.urlIn ? u->args.urlIn : argv[i];
            break;
        }
        if(*argv[i] == '-'){
            if      (streq(argv[i], "--")){
                dashdashseen=1;
            }else if(streq(argv[i], "--help")){
                printf("%s\n"
                       //"    --crop     l:r:t:b    (0:0:0:0)\n"
                       "    --canvas   w:h:fmt    (1024:512:yuv420)\n"
                       "    --crf      10         (0-51)\n"
                       "    --x264     X264ARGS   (\"\")\n"
                       "    -o         OUTFNAME   (out.h264)\n",
                       argv[0]);
                exit(0);
            }else if(streq(argv[i], "--crop")){
                conv = sscanf(argv[++i] ? argv[i] : "0", "%u:%u:%u:%u",
                              &u->args.crop.l, &u->args.crop.r,
                              &u->args.crop.t, &u->args.crop.b);
                switch(conv){
                    case 1: u->args.crop.b = u->args.crop.t = u->args.crop.r = u->args.crop.l; break;
                    case 2: u->args.crop.b = u->args.crop.t = u->args.crop.r;                  break;
                    case 3: u->args.crop.b = u->args.crop.t;                                   break;
                    default: break;
                }
            }else if(streq(argv[i], "--canvas")){
                conv = sscanf(argv[++i] ? argv[i] : "1024:512:yuv420",
                              "%u:%u:%20[yuv420super]",
                              &u->args.canvas.w, &u->args.canvas.h,
                              canvasChromaFmt);
                switch(conv){
                    default:
                    case 0: break;
                    case 1: u->args.canvas.h = u->args.canvas.w; break;
                }
                u->args.canvas.superscale = strendswith(canvasChromaFmt, "super");
                u->args.canvas.chroma_fmt = str2canvaschromafmt(canvasChromaFmt);
                if(u->args.canvas.chroma_fmt < 0){
                    i2vmessageexit(1, stderr, "Unknown option \"%s\"!\n", argv[i]);
                }
            }else if(streq(argv[i], "--crf")){
                u->args.crf = strtoull(argv[++i] ? argv[i] : "10", NULL, 0);
            }else if(streq(argv[i], "--x264")){
                u->args.x264params = argv[++i] ? argv[i] : "";
            }else if(streq(argv[i], "-o")){
                u->args.urlOut = argv[++i] ? argv[i] : u->args.urlOut;
            }else{
                i2vmessageexit(1, stderr, "Unknown option \"%s\"!\n", argv[i]);
            }
        }else{
            u->args.urlIn = u->args.urlIn ? u->args.urlIn : argv[i];
        }
    }
    
    
    /* Early Validation of Arguments */
    if(!u->args.urlIn){
        i2vmessageexit(1, stderr, "No input media provided as argument!\n");
    }
    
    if(streq(u->args.urlOut, "-")){
        u->out.fp = stdout;
    }else{
        u->out.fp = fopen(u->args.urlOut, "a+b");
        if(!u->out.fp){
            i2vmessageexit(1, stderr, "Cannot open output media!\n");
        }
    }
    
    if(u->args.crf > 51){
        i2vmessageexit(1, stderr, "Invalid CRF %u! Must be in range [0,51].\n", u->args.crf);
    }
    
    if(u->args.canvas.w < 48 || u->args.canvas.w > 4096 || u->args.canvas.w % 16 != 0){
        i2vmessageexit(1, stderr, "Coded width must be 48-4096 pixels in multiples of 16!\n");
    }
    if(u->args.canvas.h < 48 || u->args.canvas.h > 4096 || u->args.canvas.h % 16 != 0){
        i2vmessageexit(1, stderr, "Coded height must be 48-4096 pixels in multiples of 16!\n");
    }
}

/**
 * Main
 */

int main(int argc, char* argv[]){
    static UNIVERSE ustack = {0}, *u = &ustack;
    i2v_parse_args(u, argc, argv);
    return i2v_main(u);
}
