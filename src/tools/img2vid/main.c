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
int      av2benzinachromaloc(enum AVChromaLocation chroma_loc){
    if(chroma_loc == AVCHROMA_LOC_UNSPECIFIED)
        return BENZINA_CHROMALOC_CENTER;
    return (int)chroma_loc - 1;
}
enum AVChromaLocation benzina2avchromaloc(int chroma_loc){
    return (enum AVChromaLocation)(chroma_loc + 1);
}

/**
 * @brief Configure encoder for encoding the given prototypical frame.
 * 
 * @param [in,out]  u     The universe we're operating over.
 * @param [in]      frame The frame we use as a template for encoding.
 * @return 0 if successful; !0 otherwise.
 */

static int  i2v_configure_encoder(UNIVERSE* u, AVFrame* frame){
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
        i2vmessageexit(EINVAL, stderr, "Cannot transcode to H264! Please rebuild with an "
                                       "FFmpeg configured with --enable-libx264.\n");
    
    u->out.codecCtx = avcodec_alloc_context3(u->out.codec);
    if(!u->out.codecCtx)
        i2vmessageexit(ENOMEM, stderr, "Could not allocate encoder!\n");
    
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
    colorprimStr   = i2v_x264_color_primaries(frame->color_primaries);
    colorprimStr   = colorprimStr   ? colorprimStr   : i2v_x264_color_primaries(
                                      frame->color_primaries = AVCOL_PRI_BT709);
    transferStr    = i2v_x264_color_trc      (frame->color_trc);
    transferStr    = transferStr    ? transferStr    : i2v_x264_color_trc(
                                      frame->color_trc       = AVCOL_TRC_IEC61966_2_1);
    colormatrixStr = i2v_x264_color_space    (frame->colorspace);
    colormatrixStr = colormatrixStr ? colormatrixStr : i2v_x264_color_space(
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
    av_dict_set_int(&codecCtxOpt, "crf",         u->args.crf,  0);
    av_dict_set    (&codecCtxOpt, "x264-params", x264params,   0);
    u->ret = avcodec_open2(u->out.codecCtx, u->out.codec, &codecCtxOpt);
    if(av_dict_count(codecCtxOpt)){
        AVDictionaryEntry* entry = NULL;
        i2vmessage(stderr, "Warning: codec ignored %d options!\n",
                           av_dict_count(codecCtxOpt));
        while((entry=av_dict_get(codecCtxOpt, "", entry, AV_DICT_IGNORE_SUFFIX))){
            i2vmessage(stderr, "    %20s: %s\n", entry->key, entry->value);
        }
    }
    av_dict_free(&codecCtxOpt);
    if(u->ret < 0)
        i2vmessageexit(u->ret, stderr, "Could not open encoder! (%d)\n", u->ret);
    
    
    /* Return */
    return u->ret = 0;
}

/**
 * @brief Release the universe's encoder context.
 * 
 * @param [in,out]  u     The universe we're operating over.
 */

static void i2v_deconfigure_encoder(UNIVERSE* u){
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

static int  i2v_do_frame        (UNIVERSE*        u, AVFrame* frame){
    AVFrame*     graphfiltframe = NULL;
    AVFrame*     cudafiltframe  = NULL;
    
    /* Graph & CUDA Filtering Init */
    graphfiltframe = av_frame_alloc();
    if(!graphfiltframe)
        i2vmessageexit(ENOMEM, stderr, "Error allocating frame!\n");
    
    cudafiltframe = av_frame_alloc();
    if(!cudafiltframe)
        i2vmessageexit(ENOMEM, stderr, "Error allocating frame!\n");
    
    
    /**
     * Fixup and graph-filter to canonicalize input frame
     */
    
    i2v_fixup_frame(frame);
    u->ret = i2v_canonicalize_pixfmt(graphfiltframe, frame);
    if(u->ret < 0)
        i2vmessageexit(u->ret, stderr, "Failed to canonicalize pixel format! (%d)\n", u->ret);
    
    
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
            i2vmessageexit(EINVAL, stderr, "Unsupported pixel format %s!\n",
                                           av_get_pix_fmt_name(graphfiltframe->format));
    }
    u->tx.geom.i.canvas.w          = u->args.canvas.w;
    u->tx.geom.i.canvas.h          = u->args.canvas.h;
    u->tx.geom.i.canvas.chroma_fmt = u->args.canvas.chroma_fmt;
    u->tx.geom.i.canvas.superscale = u->args.canvas.superscale;
    benzina_geom_solve(&u->tx.geom);
    
    
    /**
     * CUDA Filtering.
     */
    
    cudafiltframe->width           = u->args.canvas.w;
    cudafiltframe->height          = u->args.canvas.h;
    switch(u->args.canvas.chroma_fmt){
        case BENZINA_CHROMAFMT_YUV400: cudafiltframe->format = AV_PIX_FMT_GRAY8;   break;
        case BENZINA_CHROMAFMT_YUV420: cudafiltframe->format = AV_PIX_FMT_YUV420P; break;
        case BENZINA_CHROMAFMT_YUV422: cudafiltframe->format = AV_PIX_FMT_YUV422P; break;
        case BENZINA_CHROMAFMT_YUV444: cudafiltframe->format = AV_PIX_FMT_YUV444P; break;
        default:
            i2vmessageexit(EINVAL, stderr, "Unsupported canvas chroma_fmt %d!\n",
                                           u->args.canvas.chroma_fmt);
    }
    cudafiltframe->chroma_location = benzina2avchromaloc(u->tx.geom.o.chroma_loc);
    cudafiltframe->color_range     = graphfiltframe->color_range;
    cudafiltframe->color_primaries = graphfiltframe->color_primaries;
    cudafiltframe->color_trc       = graphfiltframe->color_trc;
    cudafiltframe->colorspace      = graphfiltframe->colorspace;
    u->ret = av_frame_get_buffer(cudafiltframe, 0);
    if(u->ret != 0)
        i2vmessageexit(ENOMEM, stderr, "Error allocating frame! (%d)\n", u->ret);
    
    u->ret = i2v_cuda_filter(u, cudafiltframe, graphfiltframe, &u->tx.geom);
    if(u->ret < 0)
        i2vmessageexit(u->ret, stderr, "Failed to filter image! (%d)\n", u->ret);
    
    
    /* Encoding Init */
    u->out.packet = av_packet_alloc();
    if(!u->out.packet)
        i2vmessageexit(ENOMEM, stderr, "Failed to allocate packet object!\n");
    
    
    /**
     * Push the filtered frame  into the encoder.
     * Pull the output packet from the encoder.
     * 
     * Flushing the input encode pipeline requires pushing in a NULL frame.
     */
    
    u->ret = i2v_configure_encoder(u, cudafiltframe);
    if(u->ret)
        i2vmessageexit(u->ret, stderr, "Failed to configure x264 encoder! (%d)\n", u->ret);
    
    cudafiltframe->pts = frame->best_effort_timestamp;
    u->ret = avcodec_send_frame(u->out.codecCtx, cudafiltframe);
    if(u->ret < 0)
        i2vmessageexit(u->ret, stderr, "Error pushing frame into encoder! (%d)\n", u->ret);
    
    u->ret = avcodec_send_frame(u->out.codecCtx, NULL);
    if(u->ret < 0)
        i2vmessageexit(u->ret, stderr, "Error flushing encoder input! (%d)\n", u->ret);
    
    u->ret = avcodec_receive_packet(u->out.codecCtx, u->out.packet);
    if(u->ret < 0)
        i2vmessageexit(u->ret, stderr, "Error pulling packet from encoder! (%d)\n", u->ret);
    
    i2v_deconfigure_encoder(u);
    
    
    /* Write Out */
#if 1
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

static int  i2v_main            (UNIVERSE* u){
    /* Input Init */
    u->ret              = avformat_open_input         (&u->in.formatCtx,
                                                       u->args.urlIn,
                                                       NULL, NULL);
    if(u->ret < 0)
        i2vmessageexit(u->ret, stderr, "Cannot open input media \"%s\"! (%d)\n", u->args.urlIn, u->ret);
    
    u->ret              = avformat_find_stream_info   (u->in.formatCtx, NULL);
    if(u->ret < 0)
        i2vmessageexit(EIO,    stderr, "Cannot understand input media \"%s\"! (%d)\n", u->args.urlIn, u->ret);
    
    u->in.formatStream  = av_find_default_stream_index(u->in.formatCtx);
    if(!i2v_is_imagelike_stream(u->in.formatCtx, u->in.formatStream))
        i2vmessageexit(EINVAL, stderr, "No image data in media \"%s\"!\n", u->args.urlIn);
    
    u->in.codecPar      = u->in.formatCtx->streams[u->in.formatStream]->codecpar;
    u->in.codec         = avcodec_find_decoder        (u->in.codecPar->codec_id);
    u->in.codecCtx      = avcodec_alloc_context3      (u->in.codec);
    if(!u->in.codecCtx)
        i2vmessageexit(EINVAL, stderr, "Could not allocate decoder!\n");
    
    u->ret              = avcodec_open2               (u->in.codecCtx,
                                                       u->in.codec,
                                                       NULL);
    if(u->ret < 0)
        i2vmessageexit(u->ret, stderr, "Could not open decoder! (%d)\n", u->ret);
    
    
    /**
     * Input packet-reading loop.
     */
    
    do{
        AVPacket* packet = av_packet_alloc();
        if(!packet)
            i2vmessageexit(ENOMEM, stderr, "Failed to allocate packet object!\n");
        
        u->ret = av_read_frame(u->in.formatCtx, packet);
        if(u->ret < 0)
            i2vmessageexit(u->ret, stderr, "Error or end-of-file reading media \"%s\" (%d)!\n",
                                           u->args.urlIn, u->ret);
        
        if(packet->stream_index != u->in.formatStream){
            av_packet_free(&packet);
        }else{
            u->ret = avcodec_send_packet(u->in.codecCtx, packet);
            av_packet_free(&packet);
            if(u->ret){
                i2vmessageexit(u->ret, stderr, "Error pushing packet! (%d)\n", u->ret);
            }else{
                AVFrame* frame = av_frame_alloc();
                if(!frame)
                    i2vmessageexit(ENOMEM, stderr, "Error allocating frame!\n");
                
                u->ret = avcodec_receive_frame(u->in.codecCtx, frame);
                if(u->ret)
                    i2vmessageexit(u->ret, stderr, "Error pulling frame! (%d)\n",  u->ret);
                
                u->ret = i2v_do_frame(u, frame);
                av_frame_free(&frame);
            }
            break;
        }
    }while(!u->ret);
    
    
    /* Cleanup */
    avcodec_free_context(&u->in.codecCtx);
    avformat_close_input(&u->in.formatCtx);
    if(u->emb.fp){
        fclose(u->emb.fp);
        u->emb.fp = NULL;
    }
    fclose(u->out.fp);
    u->out.fp = NULL;
    
    
    /* Return */
    return u->ret;
}

/**
 * @brief Perform the activities for one argument group.
 * 
 * @param [in,out]  u     The universe we're operating over.
 */

static void i2v_do_one_group(UNIVERSE* u){
    i2v_main(u);
}

/**
 * @brief Parse invidual arguments.
 * 
 * This will rearrange in-place argv to create arrays of strings, and prepare
 * the universe for processing.
 * 
 * @param [in,out] u       The universe we're operating over.
 * @param [in]     p       The current argument we're handling.
 * @param [out]    pOut    Pointer to next argument we should handle.
 */

static void i2v_parse_input (UNIVERSE* u, char** p, char*** pOut){
    if(p[1]){
        u->args.urlIn = p[1];
        p += 2;
    }else{
        i2vmessageexit(EINVAL, stderr, "Error: Missing argument to -i\n");
    }
    *pOut = p;
}
static void i2v_parse_embed (UNIVERSE* u, char** p, char*** pOut){
    if(p[1]){
        u->args.urlEmb = p[1];
        p += 2;
    }else{
        i2vmessageexit(EINVAL, stderr, "Error: Missing argument to -e\n");
    }
    *pOut = p;
}
static void i2v_parse_canvas(UNIVERSE* u, char** p, char*** pOut){
    char canvasChromaFmt[21] = "yuv420";
    if(p[1]){
        switch(sscanf(p[1], "%u:%u:%20[yuv420super]",
                      &u->args.canvas.w,
                      &u->args.canvas.h,
                      canvasChromaFmt)){
            default:
            case 0: break;
            case 1: u->args.canvas.h = u->args.canvas.w; break;
        }
        u->args.canvas.superscale = strendswith(canvasChromaFmt, "super");
        u->args.canvas.chroma_fmt = str2canvaschromafmt(canvasChromaFmt);
        if(u->args.canvas.w < 48 || u->args.canvas.w > 4096 || u->args.canvas.w % 16 != 0)
            i2vmessageexit(EINVAL, stderr, "Coded width must be 48-4096 pixels in multiples of 16!\n");
        if(u->args.canvas.h < 48 || u->args.canvas.h > 4096 || u->args.canvas.h % 16 != 0)
            i2vmessageexit(EINVAL, stderr, "Coded height must be 48-4096 pixels in multiples of 16!\n");
        if(u->args.canvas.chroma_fmt < 0)
            i2vmessageexit(EINVAL, stderr, "Unknown chroma format \"%s\"!\n", canvasChromaFmt);
        
        p += 2;
    }else{
        i2vmessageexit(EINVAL, stderr, "Error: Missing argument to --canvas\n");
    }
    *pOut = p;
}
static void i2v_parse_crf   (UNIVERSE* u, char** p, char*** pOut){
    char* s;
    if(p[1]){
        u->args.crf = strtoull(p[1], &s, 0);
        if((u->args.crf > 51) || (*s != '\0')){
            i2vmessageexit(EINVAL, stderr, "Error: --crf must be integer in range [0, 51]!\n");
        }
        p += 2;
    }else{
        i2vmessageexit(EINVAL, stderr, "Error: Missing argument to --crf\n");
    }
    *pOut = p;
}
static void i2v_parse_x264  (UNIVERSE* u, char** p, char*** pOut){
    if(p[1]){
        if(p[1][0] != '-'){
            u->args.x264params = p[1];
            p += 2;
        }else{
            p += 1;
        }
    }
    *pOut = p;
}
static void i2v_parse_output(UNIVERSE* u, char** p, char*** pOut){
    if(streq(*p, "-o")){
        if(!*++p){
            i2vmessageexit(EINVAL, stderr, "Error: Missing argument to -o\n");
        }
    }
    u->args.urlOut = *p++;
    *pOut = p;
}

/**
 * @brief Parse one group of arguments.
 * 
 * This will rearrange in-place argv to create arrays of strings, and prepare
 * the universe for processing.
 * 
 * @param [in,out] u       The universe we're operating over.
 * @param [in]     argv    The beginning of the parameters vector to consider.
 * @param [out]    argvOut The beginning of the next parameters vector to consider,
 *                         or NULL if there isn't one.
 */

static void i2v_parse_one_group(UNIVERSE* u, char** argv, char*** argvOut){
    /* Set defaults for argument group */
    u->args.urlIn             =                     NULL;
    u->args.urlEmb            =                     NULL;
    u->args.urlOut            =               "out.h264";
    u->args.canvas.w          =                     1024;
    u->args.canvas.h          =                      512;
    u->args.canvas.superscale =                        1;
    u->args.canvas.chroma_fmt = BENZINA_CHROMAFMT_YUV420;
    u->args.crf               =                       10;
    u->args.x264params        =                     NULL;
    
    /* Iterate over arguments group */
    while(*argv && u->ret == 0){
        if      (streq(*argv, "-i")){
            i2v_parse_input (u, argv, &argv);
        }else if(streq(*argv, "-e")){
            i2v_parse_embed (u, argv, &argv);
        }else if(streq(*argv, "--canvas")){
            i2v_parse_canvas(u, argv, &argv);
        }else if(streq(*argv, "--crf")){
            i2v_parse_crf   (u, argv, &argv);
        }else if(streq(*argv, "--x264")){
            i2v_parse_x264  (u, argv, &argv);
        }else if(streq(*argv, "-o") || **argv != '-'){
            i2v_parse_output(u, argv, &argv);
            break;
        }else{
            i2vmessageexit(EINVAL, stderr, "Error: Unknown option %s!\n", *argv);
        }
    }
    
    /* Exit */
    *argvOut = argv;
}

/**
 * @brief Validate parsed group arguments globally.
 * 
 * @param [in,out]  u     The universe we're operating over.
 */

static void i2v_validate_one_group(UNIVERSE* u){
    if(!u->args.urlIn || !*u->args.urlIn){
        i2vmessageexit(EINVAL, stderr, "No input media provided as argument!\n");
    }
    
    if(!u->args.urlEmb          ||
       streq(u->args.urlEmb, "")){
        u->emb.fp = NULL;
    }else{
        u->emb.fp = fopen(u->args.urlEmb, "r");
        if(!u->emb.fp)
            i2vmessageexit(EPERM, stderr, "Cannot open embedded data file!\n");
    }
    
    if(!u->args.urlOut           ||
       streq(u->args.urlOut, "") ||
       streq(u->args.urlOut, "-")){
        u->out.fp = stdout;
    }else{
        u->out.fp = fopen(u->args.urlOut, "a+b");
        if(!u->out.fp)
            i2vmessageexit(EPERM, stderr, "Cannot open output media!\n");
    }
}

/**
 * @brief Interpret the program, performing what the arguments require.
 * 
 * The command-line is structured in several groups of flags that precede
 * an output filename. If a group lacks an output filename, the result is
 * written to standard output.
 * 
 * @param [in,out]  u     The universe we're operating over.
 * @param [in,out]  argc  The parameters count.
 * @param [in,out]  argv  The parameters vector.
 * 
 * @return Exit code.
 */

static int  i2v_interpret_args  (UNIVERSE*        u,
                                 int              argc,
                                 char**           argv){
    (void)argc;
    while(*argv){
        i2v_parse_one_group(u, argv, &argv);
        i2v_validate_one_group(u);
        i2v_do_one_group(u);
    }
    return u->ret;
}

/**
 * Main
 */

int main(int argc, char* argv[]){
    static UNIVERSE ustack = {0}, *u = &ustack;
    return i2v_interpret_args(u, argc-1, argv+1);
}
