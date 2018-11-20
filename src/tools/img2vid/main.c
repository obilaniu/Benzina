/* Includes */
#include "main.h"
#include "kernels.h"


/**
 * Defines
 */



/**
 * Functions.
 */

/**
 * @brief Calculate the sampling offset from the origin within the source
 *        required to equalize the information contained in all destination
 *        sampling points.
 * 
 * @param [in]  dst  The size of the destination dimension, in samples.
 * @param [in]  src  The size of the source dimension, in samples.
 * @return Required sampling offset.
 */

static float i2vCalcSubsampleOff(uint64_t dst, uint64_t src){
	return 0.5*src/dst - 0.5;
}

/**
 * @brief Quick arguments parsing.
 * 
 * Everything after -- is taken not to be an option.
 */

static void i2vParseArgs        (UNIVERSE*        u,
                                 int              argc,
                                 char*            argv[]){
	int i, dashdashseen=0, conv=0;
	
	
	/* Defaults */
	u->args.embedMode     =   "square";
	u->args.size.w        =       1024;
	u->args.size.h        =       1024;
	u->args.crf           =         10;
	u->args.urlOut        = "out.h264";
	u->args.x264params    =         "";
	
	
	/* Parsing */
	for(i=1;i<argc;i++){
		if(dashdashseen){
			u->args.urlIn = argv[i];
			break;
		}
		if(*argv[i] == '-'){
			if      (streq(argv[i], "--")){
				dashdashseen=1;
			}else if(streq(argv[i], "--help")){
				printf("%s\n"
				       "    --mode     square   (square|simple)\n"
				       "    --crop     l:r:t:b  (0:0:0:0)\n"
				       "    --size     w:h      (1024x1024)\n"
				       "    --crf      10       (0-51)\n",
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
			}else if(streq(argv[i], "--size")){
				conv = sscanf(argv[++i] ? argv[i] : "1024", "%u%*[:x]%u",
				              &u->args.size.w, &u->args.size.h);
				switch(conv){
					case 1: u->args.size.h = u->args.size.w; break;
					default: break;
				}
			}else if(streq(argv[i], "--crf")){
				u->args.crf       = strtoull(argv[++i] ? argv[i] :   "10", NULL, 0);
			}else if(streq(argv[i], "--mode")){
				u->args.embedMode = argv[++i] ? argv[i] : "square";
			}else if(streq(argv[i], "--x264")){
				u->args.x264params = argv[++i] ? argv[i] : "";
			}else if(streq(argv[i], "-o")){
				u->args.urlOut    = argv[++i];
			}else{
				i2vmessageexit(1, stderr, "Unknown option \"%s\"!\n", argv[i]);
			}
		}else{
			u->args.urlIn = argv[i];
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
	
	if(u->args.size.w < 16 || u->args.size.w > 4096 || u->args.size.w % 16 != 0){
		i2vmessageexit(1, stderr, "Coded width must be 16-4096 pixels in multiples of 16!\n");
	}
	if(u->args.size.h < 16 || u->args.size.h > 4096 || u->args.size.h % 16 != 0){
		i2vmessageexit(1, stderr, "Coded height must be 16-4096 pixels in multiples of 16!\n");
	}
	
	if(strne(u->args.embedMode, "square") &&
	   strne(u->args.embedMode, "simple")){
		i2vmessageexit(1, stderr, "Unknown embedding mode \"%d\"!\n", u->args.embedMode);
	}
	
	if(streq(u->args.embedMode, "square") &&
	   u->args.size.w != u->args.size.h){
		i2vmessageexit(1, stderr, "Embedding mode is \"square\", but coded size is not a square!\n");
	}
}

/**
 * @brief Some pixel formats are deprecated in favour of a combination with
 *        other attributes (such as color_range), so we fix them up.
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
 * @param [in]  f   The AVFrame whose attributes are to be fixed up.
 * @return 0.
 */

static int  i2vFixupPixFmt      (AVFrame*         f){
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
	return 0;
}

/**
 * @brief Compute the transform parameters.
 * 
 * The essential question here is to determine the mapping from integer
 * destination pixel coordinates to the corresponding source coordinate.
 * 
 * In "simple" mode, this is the identity matrix. Images are effectively
 * aligned to the top left corner and cropped to the coded frame size if they
 * extend past it, or padded if they do not reach the frame size.
 * 
 * In "square" mode, the image's center is aligned with the frame center, and
 * the image scaled such that its center crop occupies the center 25% area of
 * the frame. The rest of the image extends EITHER top-and-bottom OR
 * left-and-right OR neither beyond the center area, being cropped if it
 * extends beyond frame limits and padded if it does not reach the frame size.
 * 
 * In both cases, it is necessary to find:
 * 
 *     u->tx.crop.{lrtb}:    The cropping applied to the source  image.
 *     u->tx.pad.{lrtb}:     The padding  applied to the embedded image to make
 *                           it fit within the coded frame.
 *     u->tx.embed.{xywh}:   The offset and size of the embedded image within
 *                           the coded frame.
 *     u->tx.scale.{xy}:     The scale factors for resizing.
 *     u->tx.lumaoff.{xy}:   The offsets applied to source luma accesses.
 *     u->tx.chromaoff.{xy}: The offsets applied to source chroma accesses.
 * 
 * @return 0 if successful; !0 if image cannot be validly transformed.
 */

static int  i2vComputeTx        (UNIVERSE*        u){
	int frmH = u->in.frame->height,
	    frmW = u->in.frame->width,
	    srcH = frmH - u->args.crop.t - u->args.crop.b,
	    srcW = frmW - u->args.crop.l - u->args.crop.r,
	    codH = u->out.codecCtx->coded_height,
	    codW = u->out.codecCtx->coded_width;
	
	if(srcH <= 0 || srcW <= 0){
		return -1;
	}
	
	u->tx.range       = u->in.frame->color_range;
	u->tx.crop.l      = u->args.crop.l;
	u->tx.crop.t      = u->args.crop.t;
	u->tx.crop.r      = u->args.crop.r;
	u->tx.crop.b      = u->args.crop.b;
	u->tx.pad.l       = 0;
	u->tx.pad.t       = 0;
	u->tx.pad.r       = 0;
	u->tx.pad.b       = 0;
	if      (streq(u->args.embedMode, "square")){
		if      (srcW >= srcH*2){
			/**
			 * Image much wider than tall. Crop L/R, Pad T/B.
			 * 
			 * Crop to aspect ratio 2:1, with height a multiple of 4, then pad
			 * top and bottom with a quarter-height for centering.
			 */
			
			u->tx.pad.t       = codH/4;
			u->tx.pad.b       = codH/4;
			u->tx.embed.w     = codW;
			u->tx.embed.h     = codH/2;
			u->tx.crop.l     += (srcW - srcH*2 + 0)/2;
			u->tx.crop.r     += (srcW - srcH*2 + 1)/2;
			srcW              = srcH*2;
		}else if(srcH >= srcW*2){
			/**
			 * Image much taller than wide. Crop T/B, Pad L/R.
			 * 
			 * Crop to aspect ratio 1:2, with width a multiple of 4, then pad
			 * left and right with a quarter-width for centering.
			 */
			
			u->tx.pad.l       = codW/4;
			u->tx.pad.r       = codW/4;
			u->tx.embed.w     = codW/2;
			u->tx.embed.h     = codH;
			u->tx.crop.t     += (srcH - srcW*2 + 0)/2;
			u->tx.crop.b     += (srcH - srcW*2 + 1)/2;
			srcH              = srcW*2;
		}else{
			/**
			 * Image of sane aspect ratio.  Pad T/B/L/R.
			 * 
			 * No cropping, but fudge embed.h and embed.w to get a multiple of 4.
			 * Then, pad left, right, top and bottom with multiples of 2 pixels
			 * for centering.
			 */
			
			u->tx.embed.w     = srcH>srcW ? codW/2 : (int64_t)codW*srcW/srcH/8*4;
			u->tx.embed.h     = srcW>srcH ? codH/2 : (int64_t)codH*srcH/srcW/8*4;
			u->tx.pad.l       = (codW-u->tx.embed.w)/2;
			u->tx.pad.r       = (codW-u->tx.embed.w)/2;
			u->tx.pad.t       = (codH-u->tx.embed.h)/2;
			u->tx.pad.b       = (codH-u->tx.embed.h)/2;
		}
		u->tx.embed.x     = u->tx.pad.l;
		u->tx.embed.y     = u->tx.pad.t;
		u->tx.scale.x     = (float)srcW/u->tx.embed.w;
		u->tx.scale.y     = (float)srcH/u->tx.embed.h;
		u->tx.lumaoff.x   = i2vCalcSubsampleOff(u->tx.embed.w/1, srcW);
		u->tx.lumaoff.y   = i2vCalcSubsampleOff(u->tx.embed.h/1, srcH);
		u->tx.chromaoff.x = i2vCalcSubsampleOff(u->tx.embed.w/2, srcW);
		u->tx.chromaoff.y = i2vCalcSubsampleOff(u->tx.embed.h/2, srcH);
	}else if(streq(u->args.embedMode, "simple")){
		if(srcW > codW){
			u->tx.crop.r += srcW-codW;
			u->tx.embed.w = codW;
		}else{
			u->tx.pad.r   = codW-srcW;
			u->tx.embed.w = srcW;
		}
		if(srcH > codH){
			u->tx.crop.b += srcH-codH;
			u->tx.embed.h = codH;
		}else{
			u->tx.pad.b   = codH-srcH;
			u->tx.embed.h = srcH;
		}
		u->tx.embed.x     = u->tx.pad.l;
		u->tx.embed.y     = u->tx.pad.t;
		u->tx.scale.x     = u->tx.scale.y     = 1.0f;
		u->tx.lumaoff.x   = u->tx.lumaoff.y   = 0.0f;
		u->tx.chromaoff.x = u->tx.chromaoff.y = 0.5f;
	}
	
	return 0;
}

/**
 * @brief Write out the compressed datum.
 * 
 * @return 0 if successful, !0 otherwise.
 */

static int  i2vWriteOut         (UNIVERSE*        u){
#if 1
	fwrite(u->out.packet->data, 1, u->out.packet->size, u->out.fp);
	fclose(u->out.fp);
#endif
	
	return u->ret=0;
}

/**
 * @brief Handle one frame.
 * 
 * This means creating the filter graph specific to this frame and invoking it.
 */

static int  i2vDoFrame          (UNIVERSE*        u){
	char graphstr     [512] = {0};
	char buffersrcargs[512] = {0};
	
	i2vFixupPixFmt(u->in.frame);
	if(i2vComputeTx(u) != 0){
		return -1;
	}
	
	/**
	 * The arguments for the buffersrc filter, buffersink filter, and the
	 * creation of the graph must be done as follows. Even if other ways are
	 * documented, they will *NOT* work.
	 * 
	 *   - buffersrc:  snprintf() a minimal set of arguments into a string.
	 *   - buffersink: av_opt_set_int_list() a list of admissible pixel formats.
	 *   - graph:      1) snprintf() the graph description into a string.
	 *                 2) avfilter_graph_parse_ptr() parse it.
	 *                 3) avfilter_graph_config() configure it.
	 */
	
	snprintf(buffersrcargs, sizeof(buffersrcargs),
	         "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:sar=%d/%d:frame_rate=%d",
	         u->in.frame->width, u->in.frame->height, u->in.frame->format,
	         u->out.codecCtx->time_base.num, u->out.codecCtx->time_base.den,
	         u->out.codecCtx->sample_aspect_ratio.num,
	         u->out.codecCtx->sample_aspect_ratio.den,
	         u->out.codecCtx->time_base.den);
	
	/**
	 * We filter the frame losslessly (but discarding alpha) using a graph to
	 * one of a reduced set of canonical formats:
	 * 
	 *   - AV_PIX_FMT_YUV420P for all 8-bit 4:2:0 YUV-like formats, B&W and grey images
	 *   - AV_PIX_FMT_YUV444P for all 8-bit 4:4:4 YUV-like formats
	 *   - AV_PIX_FMT_GBRP    for all RGB-like formats and the palette format
	 *   - ...
	 * 
	 * We then run one of a tiny few kernels to crop/pad/resize in a gamma-correct
	 * manner the frame to YUV420P.
	 */
	
	switch(u->in.frame->format){
		case AV_PIX_FMT_YUV420P:
		case AV_PIX_FMT_YUVA420P:
		case AV_PIX_FMT_GRAY8:
		case AV_PIX_FMT_YA8:
		case AV_PIX_FMT_MONOBLACK:
		case AV_PIX_FMT_MONOWHITE:
		case AV_PIX_FMT_NV12:
		case AV_PIX_FMT_NV21:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv420p");
		break;
		case AV_PIX_FMT_YUV444P:
		case AV_PIX_FMT_YUVA444P:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv444p");
		break;
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
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=gbrp");
		break;
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
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv420p16");
		break;
		case AV_PIX_FMT_YUV444P16LE:
		case AV_PIX_FMT_YUV444P16BE:
		case AV_PIX_FMT_YUVA444P16LE:
		case AV_PIX_FMT_YUVA444P16BE:
		case AV_PIX_FMT_AYUV64BE:
		case AV_PIX_FMT_AYUV64LE:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv444p16");
		break;
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
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=gbrp16");
		break;
		case AV_PIX_FMT_YUV410P:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv410p");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_YUV411P:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv411p");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_UYYVYY411:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=uyyvyy411");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_YUV420P9LE:
		case AV_PIX_FMT_YUV420P9BE:
		case AV_PIX_FMT_YUVA420P9LE:
		case AV_PIX_FMT_YUVA420P9BE:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv420p9");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_YUV420P10LE:
		case AV_PIX_FMT_YUV420P10BE:
		case AV_PIX_FMT_YUVA420P10LE:
		case AV_PIX_FMT_YUVA420P10BE:
		case AV_PIX_FMT_P010LE:
		case AV_PIX_FMT_P010BE:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv420p10");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_YUV420P12LE:
		case AV_PIX_FMT_YUV420P12BE:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv420p12");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_YUV420P14LE:
		case AV_PIX_FMT_YUV420P14BE:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv420p14");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_GBRP9LE:
		case AV_PIX_FMT_GBRP9BE:
		case AV_PIX_FMT_GRAY9LE:
		case AV_PIX_FMT_GRAY9BE:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=gbrp9");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_GBRP10LE:
		case AV_PIX_FMT_GBRP10BE:
		case AV_PIX_FMT_GBRAP10LE:
		case AV_PIX_FMT_GBRAP10BE:
		case AV_PIX_FMT_GRAY10LE:
		case AV_PIX_FMT_GRAY10BE:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=gbrp10");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_GBRP12LE:
		case AV_PIX_FMT_GBRP12BE:
		case AV_PIX_FMT_GBRAP12LE:
		case AV_PIX_FMT_GBRAP12BE:
		case AV_PIX_FMT_GRAY12LE:
		case AV_PIX_FMT_GRAY12BE:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=gbrp12");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_GBRP14LE:
		case AV_PIX_FMT_GBRP14BE:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=gbrp14");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_YUV422P:
		case AV_PIX_FMT_YUVA422P:
		case AV_PIX_FMT_UYVY422:
		case AV_PIX_FMT_YVYU422:
		case AV_PIX_FMT_YUYV422:
		case AV_PIX_FMT_NV16:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv422p");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_YUV422P9LE:
		case AV_PIX_FMT_YUV422P9BE:
		case AV_PIX_FMT_YUVA422P9LE:
		case AV_PIX_FMT_YUVA422P9BE:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv422p9");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_YUV422P10LE:
		case AV_PIX_FMT_YUV422P10BE:
		case AV_PIX_FMT_YUVA422P10LE:
		case AV_PIX_FMT_YUVA422P10BE:
		case AV_PIX_FMT_NV20LE:
		case AV_PIX_FMT_NV20BE:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv422p10");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_YUV422P12LE:
		case AV_PIX_FMT_YUV422P12BE:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv422p12");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_YUV422P14LE:
		case AV_PIX_FMT_YUV422P14BE:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv422p14");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_YUV422P16LE:
		case AV_PIX_FMT_YUV422P16BE:
		case AV_PIX_FMT_YUVA422P16LE:
		case AV_PIX_FMT_YUVA422P16BE:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv422p16");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_YUV440P:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv440p");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_YUV440P10LE:
		case AV_PIX_FMT_YUV440P10BE:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv440p10");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_YUV440P12LE:
		case AV_PIX_FMT_YUV440P12BE:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv440p12");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_YUV444P9LE:
		case AV_PIX_FMT_YUV444P9BE:
		case AV_PIX_FMT_YUVA444P9LE:
		case AV_PIX_FMT_YUVA444P9BE:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv444p9");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_YUV444P10LE:
		case AV_PIX_FMT_YUV444P10BE:
		case AV_PIX_FMT_YUVA444P10LE:
		case AV_PIX_FMT_YUVA444P10BE:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv444p10");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_YUV444P12LE:
		case AV_PIX_FMT_YUV444P12BE:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv444p12");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_YUV444P14LE:
		case AV_PIX_FMT_YUV444P14BE:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=yuv444p14");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		case AV_PIX_FMT_GBRPF32LE:
		case AV_PIX_FMT_GBRPF32BE:
		case AV_PIX_FMT_GBRAPF32LE:
		case AV_PIX_FMT_GBRAPF32BE:
			snprintf(graphstr, sizeof(graphstr),
			         "scale=in_range=jpeg:out_range=jpeg,format=pix_fmts=gbrpf32");
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		default:
			i2vmessage(stderr, "Pixel format %s unsupported!\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		break;
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
			i2vmessage(stderr, "Hardware pixel format %s unsupported and there are no plans to support it.\n",
			           av_get_pix_fmt_name(u->in.frame->format));
			return -1;
		break;
	}
	
	u->ret = avfilter_graph_create_filter(&u->graph.buffersrcCtx,
	                                      u->graph.buffersrc,
	                                      "in",
	                                      buffersrcargs,
	                                      NULL,
	                                      u->graph.graph);
	if(u->ret < 0){
		i2vmessage(stderr, "Failed to allocate buffersrc filter instance!\n");
		return -1;
	}
	u->ret = avfilter_graph_create_filter(&u->graph.buffersinkCtx,
	                                      u->graph.buffersink,
	                                      "out",
	                                      NULL,
	                                      NULL,
	                                      u->graph.graph);
	if(u->ret < 0){
		i2vmessage(stderr, "Failed to allocate buffersink filter instance!\n");
		return -1;
	}
	u->graph.inputs ->name       = av_strdup("out");
	u->graph.inputs ->filter_ctx = u->graph.buffersinkCtx;
	u->graph.inputs ->pad_idx    = 0;
	u->graph.inputs ->next       = NULL;
	u->graph.outputs->name       = av_strdup("in");
	u->graph.outputs->filter_ctx = u->graph.buffersrcCtx;
	u->graph.outputs->pad_idx    = 0;
	u->graph.outputs->next       = NULL;
	if(!u->graph.inputs->name || !u->graph.outputs->name){
		i2vmessage(stderr, "Failed to allocate name for filter input/output!\n");
		return u->ret=-1;
	}
	u->ret = avfilter_graph_parse_ptr(u->graph.graph,
	                                  graphstr,
	                                  &u->graph.inputs,
	                                  &u->graph.outputs,
	                                  NULL);
	if(u->ret < 0){
		i2vmessage(stderr, "Error parsing graph! (%d)\n", u->ret);
		return u->ret;
	}
	u->ret = avfilter_graph_config(u->graph.graph, NULL);
	if(u->ret < 0){
		i2vmessage(stderr, "Error configuring graph! (%d)\n", u->ret);
		return u->ret;
	}
	
#if 0
	char* str = avfilter_graph_dump(u->graph.graph, "");
	i2vmessage(stdout, "%s\n", str);
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
	
	u->ret = av_buffersrc_add_frame_flags(u->graph.buffersrcCtx,
	                                      u->in.frame,
	                                      AV_BUFFERSRC_FLAG_KEEP_REF);
	av_buffersrc_close(u->graph.buffersrcCtx, 0, 0);
	if(u->ret < 0){
		i2vmessage(stderr, "Error pushing frame into graph! (%d)\n", u->ret);
		return u->ret;
	}
	u->ret = av_buffersink_get_frame(u->graph.buffersinkCtx, u->filt.in.frame);
	if(u->ret < 0){
		i2vmessage(stderr, "Error pulling frame from graph! (%d)\n", u->ret);
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
	
	return i2vWriteOut(u);
}

/**
 * Handle one packet.
 */

static int  i2vDoPacket         (UNIVERSE* u){
	u->ret = avcodec_send_packet  (u->in.codecCtx, u->in.packet);
	if(u->ret){
		i2vmessage(stderr, "Error pushing packet! (%d)\n", u->ret);
		return u->ret;
	}
	u->ret = avcodec_receive_frame(u->in.codecCtx, u->in.frame);
	if(u->ret){
		i2vmessage(stderr, "Error pulling frame! (%d)\n",  u->ret);
		return u->ret;
	}
	
	return u->ret=i2vDoFrame(u);
}

/**
 * Check that input stream is of image type.
 */

static int  i2vIsImageLikeStream(AVFormatContext* avfctx,
                                 int              stream){
	if(stream < 0 || stream >= (int)avfctx->nb_streams){
		return 0;
	}
	return avfctx->streams[stream]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO;
}

/**
 * Open and close FFmpeg.
 * 
 * In-between, handle frames.
 */

static int  i2vMain             (UNIVERSE* u){
	int         c0, c1, c2, c3;
	char        x264params[1024];
	const char* colorprimStr   = NULL;
	const char* transferStr    = NULL;
	const char* colormatrixStr = NULL;
	#define UNPACK_ERR(r,c0,c1,c2,c3)   \
	    do{                             \
	        c0 = ((unsigned)-r) >>  0;  \
	        c1 = ((unsigned)-r) >>  8;  \
	        c2 = ((unsigned)-r) >> 16;  \
	        c3 = ((unsigned)-r) >> 24;  \
	    }while(0)
	#define MAYBE_PRINT_ERR(c0,c1,c2,c3)                      \
	    do{                                                   \
	        if(!((c0|c1|c2|c3) & 0x80)){                      \
	            fprintf(stderr, "(%c%c%c%c)\n", c0,c1,c2,c3); \
	        }                                                 \
	    }while(0)
	
	/**
	 * FFmpeg initialization.
	 */
	
	avformat_network_init();
	
	
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
		UNPACK_ERR(u->ret,c0,c1,c2,c3);
		MAYBE_PRINT_ERR(c0,c1,c2,c3);
		goto fail;
	}
	u->ret              = avformat_find_stream_info   (u->in.formatCtx, NULL);
	if(u->ret < 0){
		i2vmessage(stderr, "Cannot understand input media \"%s\"! (%d)\n", u->args.urlIn, u->ret);
		UNPACK_ERR(u->ret,c0,c1,c2,c3);
		MAYBE_PRINT_ERR(c0,c1,c2,c3);
		goto fail;
	}
	u->in.formatStream  = av_find_default_stream_index(u->in.formatCtx);
	if(!i2vIsImageLikeStream(u->in.formatCtx, u->in.formatStream)){
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
		UNPACK_ERR(u->ret,c0,c1,c2,c3);
		MAYBE_PRINT_ERR(c0,c1,c2,c3);
		goto fail;
	}
	switch(u->in.codecPar->color_primaries){
		default: u->in.codecPar->color_primaries = AVCOL_PRI_BT709;
		case AVCOL_PRI_BT709:         colorprimStr   = "bt709";         break;
		case AVCOL_PRI_BT470M:        colorprimStr   = "bt470m";        break;
		case AVCOL_PRI_BT470BG:       colorprimStr   = "bt470bg";       break;
		case AVCOL_PRI_SMPTE170M:     colorprimStr   = "smpte170m";     break;
		case AVCOL_PRI_SMPTE240M:     colorprimStr   = "smpte240m";     break;
		case AVCOL_PRI_FILM:          colorprimStr   = "film";          break;
		case AVCOL_PRI_BT2020:        colorprimStr   = "bt2020";        break;
		case AVCOL_PRI_SMPTE428:      colorprimStr   = "smpte428";      break;
		case AVCOL_PRI_SMPTE431:      colorprimStr   = "smpte431";      break;
		case AVCOL_PRI_SMPTE432:      colorprimStr   = "smpte432";      break;
	}
	switch(u->in.codecPar->color_trc){
		case AVCOL_TRC_BT709:         transferStr    = "bt709";         break;
		case AVCOL_TRC_GAMMA22:       transferStr    = "bt470m";        break;
		case AVCOL_TRC_GAMMA28:       transferStr    = "bt470bg";       break;
		case AVCOL_TRC_SMPTE170M:     transferStr    = "smpte170m";     break;
		case AVCOL_TRC_SMPTE240M:     transferStr    = "smpte240m";     break;
		case AVCOL_TRC_LINEAR:        transferStr    = "linear";        break;
		case AVCOL_TRC_LOG:           transferStr    = "log100";        break;
		case AVCOL_TRC_LOG_SQRT:      transferStr    = "log316";        break;
		case AVCOL_TRC_IEC61966_2_4:  transferStr    = "iec61966-2-4";  break;
		case AVCOL_TRC_BT1361_ECG:    transferStr    = "bt1361e";       break;
		default: u->in.codecPar->color_trc = AVCOL_TRC_IEC61966_2_1;
		case AVCOL_TRC_IEC61966_2_1:  transferStr    = "iec61966-2-1";  break;
		case AVCOL_TRC_BT2020_10:     transferStr    = "bt2020-10";     break;
		case AVCOL_TRC_BT2020_12:     transferStr    = "bt2020-12";     break;
		case AVCOL_TRC_SMPTE2084:     transferStr    = "smpte2084";     break;
		case AVCOL_TRC_SMPTE428:      transferStr    = "smpte428";      break;
	}
	switch(u->in.codecPar->color_space){
		case AVCOL_SPC_BT709:         colormatrixStr = "bt709";         break;
		case AVCOL_SPC_FCC:           colormatrixStr = "fcc";           break;
		default: u->in.codecPar->color_space = AVCOL_SPC_BT470BG;
		case AVCOL_SPC_BT470BG:       colormatrixStr = "bt470bg";       break;
		case AVCOL_SPC_SMPTE170M:     colormatrixStr = "smpte170m";     break;
		case AVCOL_SPC_SMPTE240M:     colormatrixStr = "smpte240m";     break;
		case AVCOL_SPC_RGB:           colormatrixStr = "GBR";           break;
		case AVCOL_SPC_YCGCO:         colormatrixStr = "YCgCo";         break;
		case AVCOL_SPC_BT2020_NCL:    colormatrixStr = "bt2020nc";      break;
		case AVCOL_SPC_BT2020_CL:     colormatrixStr = "bt2020c";       break;
		case AVCOL_SPC_SMPTE2085:     colormatrixStr = "smpte2085";     break;
	}
	
	
	
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
	u->out.codecCtx->width                  = u->args.size.w;
	u->out.codecCtx->height                 = u->args.size.h;
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
		UNPACK_ERR(u->ret,c0,c1,c2,c3);
		MAYBE_PRINT_ERR(c0,c1,c2,c3);
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
	
	
	/* Graph Init */
	u->graph.graph                          = avfilter_graph_alloc();
	u->graph.outputs                        = avfilter_inout_alloc();
	u->graph.inputs                         = avfilter_inout_alloc();
	u->graph.buffersrc                      = avfilter_get_by_name("buffer");
	u->graph.buffersink                     = avfilter_get_by_name("buffersink");
	if(!u->graph.graph){
		i2vmessage(stderr, "Failed to allocate graph object!\n");
		goto fail;
	}
	if(!u->graph.outputs   || !u->graph.inputs){
		i2vmessage(stderr, "Failed to allocate inout objects!\n");
		goto fail;
	}
	if(!u->graph.buffersrc || !u->graph.buffersink){
		i2vmessage(stderr, "Failed to find buffer or buffersink filter objects!\n");
		goto fail;
	}
	avfilter_graph_set_auto_convert(u->graph.graph, AVFILTER_AUTO_CONVERT_NONE);
	
	
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
			u->ret = i2vDoPacket(u);
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
	avformat_network_deinit();
	return u->ret;
	
	#undef UNPACK_ERR
	#undef MAYBE_PRINT_ERR
}

/**
 * Main
 */

int main(int argc, char* argv[]){
	static UNIVERSE ustack = {0}, *u = &ustack;
	i2vParseArgs(u, argc, argv);
	return i2vMain(u);
}
