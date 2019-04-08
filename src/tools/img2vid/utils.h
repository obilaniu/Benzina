/* Include Guard */
#ifndef SRC_TOOLS_IMG2VID_UTILS_H
#define SRC_TOOLS_IMG2VID_UTILS_H

/**
 * Includes
 */

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>

/**
 * Defines
 */




/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif


/* Public Function Declarations */

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

extern int i2v_fixup_frame(AVFrame* f);

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

extern enum AVPixelFormat i2v_pick_pix_fmt(enum AVPixelFormat src);

/**
 * @brief Convert input frame to output with canonical pixel format.
 * 
 * @param [out]  out  Output frame, with canonical pixel format.
 * @param [in]   in   Input frame.
 * @return 0 if successful; !0 otherwise.
 */

extern int i2v_canonicalize_pixfmt(AVFrame* out, AVFrame* in);

/**
 * @brief Return whether the stream is of image/video type (contains pixel data).
 * 
 * @param [in]  avfctx  Format context whose streams we are inspecting.
 * @param [in]  stream  Integer number of the stream we are inspecting.
 * @return 0 if not a valid stream or not of image/video type; 1 otherwise.
 */

extern int i2v_is_imagelike_stream(AVFormatContext* avfctx, int stream);

/**
 * @brief Convert to x264-expected string the integer code for color primaries,
 *        transfer characteristics and color space.
 * 
 * @return A string representation of the code, as named by x264; Otherwise NULL.
 */

extern const char* i2v_x264_color_primaries(enum AVColorPrimaries              color_primaries);
extern const char* i2v_x264_color_trc      (enum AVColorTransferCharacteristic color_trc);
extern const char* i2v_x264_color_space    (enum AVColorSpace                  color_space);


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

