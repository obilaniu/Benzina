/* Includes */
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavformat/avformat.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>

#include "benzina/benzina.h"
#include "benzina/itu/h264.h"
#include "benzina/itu/h265.h"
#include "benzina/itu/h26x.h"



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
typedef struct PACKETLIST PACKETLIST;
typedef struct ITEM       ITEM;
typedef struct IREF       IREF;
typedef struct IPRP       IPRP;
typedef struct UNIVERSE   UNIVERSE;



/**
 * Data Structure Definitions
 */

/**
 * @brief Singly-linked-list of AVPackets.
 */

struct PACKETLIST{
    PACKETLIST* next;
    AVPacket*   packet;
};

/**
 * @brief Item to be added.
 * 
 * The valid item types can be found at mp4ra.org, but because the website is
 * broken, look at the source directly:
 *     https://github.com/mp4ra/mp4ra.github.io/blob/master/CSV/item-types.csv
 * 
 * As of 2019-09-09 its contents are reproduced below:
 * 
 *     code,description,specification
 *     auvd,Auxiliary Video descriptor,ISO
 *     avc1,AVC image item,HEIF
 *     Exif,EXIF item,HEIF
 *     grid,Image item grid derivation,HEIF
 *     hvc1,HEVC image item,HEIF
 *     hvt1,HEVC tile image item,HEIF
 *     iden,Image item identity derivation,HEIF
 *     iovl,Image item overlay derivation,HEIF
 *     j2k1,Contiguous Codestream box as specified in Rec. ITU-T T.800 | ISO/IEC 15444-1,J2KHEIF
 *     jpeg,JPEG image item,HEIF
 *     jxs1,Images coded to the JPEG-XS coding format,JPXS
 *     lhv1,Layered HEVC image item,HEIF
 *     mint,Data integrity item,HEIF
 *     mime,Item identified by a MIME type,ISO
 * 
 * There is at least one row missing, approximately as follows:
 * 
 *     uri%20,Item identified by a URI type,ISO
 */

struct ITEM{
    ITEM*       next;
    uint32_t    id;
    char        type[4];
    const char* name;
    const char* mime_type;
    unsigned    primary        : 1;
    unsigned    hidden         : 1;
    
    /**
     * @brief Argument section.
     * 
     * Contains per-item fields that are not semantically part of a HEIF item.
     */
    
    struct{
        const char*           path;
        struct{uint32_t w,h;} tile;
        enum AVPixelFormat    pix_fmt;
        unsigned              want_thumbnail : 1;
    } args;
    
    /**
     * @brief Frontend section.
     * 
     * Contains per-item fields that are decompressed intermediaries on the way
     * to crafting a HEIF item.
     */
    
    union{
        struct{
            struct{uint32_t w,h;} grid_clap,  grid_ispe,
                                  thumb_clap, thumb_ispe;
            AVFrame* frame;
            AVFrame* grid_frame;
            AVFrame* thumb_frame;
            AVCodec* codec;
            unsigned is_source_jpeg   : 1;
            unsigned single_tile      : 1;
            unsigned single_tile_ltbx : 1;
        } pict;
    } frontend;
    
    /**
     * @brief Middleend section.
     * 
     * Contains per-item fields related to encoding/compression on the way to
     * crafting a HEIF item.
     */
    
    union{
        struct{
            AVCodecContext* codec_ctx;
            PACKETLIST*     packet_list;
            PACKETLIST*     thumb_packet_list;
            PACKETLIST*     grid_packet_list;
        } pict;
        struct{
            PACKETLIST*     packet_list;
        } binary;
    } middleend;
    
    /**
     * @brief Backend section.
     * 
     * Contains per-item fields that are semantically part of a HEIF item.
     */
    
    struct{
        struct{
            struct{uint32_t w,h;} grid_clap, ispe;
            AVFrame* frame;
        } pict;
        struct{
            struct{uint32_t w,h;} grid_clap, ispe;
            AVFrame* frame;
            ITEM* refs;
        } iden;
        struct{
            struct{uint32_t w,h;} grid_clap, ispe, tile;
            AVFrame* frame;
            ITEM* refs;
        } grid;
        struct{
            
        } Exif, mime, uri_;
    } backend;
    
    IREF*       refs;
    ITEM*       children;
};

/**
 * @brief Item reference.
 * 
 * The valid item references can be found at mp4ra.org, here:
 *     https://github.com/mp4ra/mp4ra.github.io/blob/master/CSV/item-references.csv
 * 
 * As of 2019-09-09 its contents are reproduced below:
 * 
 *     code,description,type,specification
 *     auxl,Auxiliary image item reference,item ref.,HEIF
 *     base,Pre-derived image item base reference, item ref.,HEIF
 *     dimg,Derived image item reference,item ref.,HEIF
 *     dpnd,Item coding dependency reference,item ref.,HEIF
 *     exbl,Scalable image item reference, item ref.,HEIF
 *     fdel,File delivery reference,item ref.,ISO
 *     grid,Image item grid reference,item ref.,HEIF
 *     iloc,Item data location,item ref.,ISO
 *     prem,Pre-Multiplied item,item ref.,MIAF
 *     thmb,Thumbnail image item reference,item ref.,HEIF
 */

struct IREF{
    IREF*     next;
    char      type[4];
    uint32_t  num_references;
    uint32_t* to_item_id;
};

/**
 * @brief Item property.
 * 
 * The valid item properties can be found at mp4ra.org, here:
 *     https://raw.githubusercontent.com/mp4ra/mp4ra.github.io/master/CSV/item-properties.csv
 * 
 * As of 2019-11-29 its contents are reproduced below:
 * 
 *     code,description,specification
 *     auxC,Auxiliary type,HEIF
 *     avcC,AVC Image item configuration,HEIF
 *     brnd,Item brand (type),ISO
 *     clap,Clean aperture,HEIF
 *     colr,Colour information,HEIF
 *     hvcC,HEVC Image item configuration,HEIF
 *     iaux,Item Auxiliary Information,ISO Common Encryption
 *     ienc,Item Encryption,ISO Common Encryption
 *     imir,Image mirroring,HEIF
 *     irot,Image rotation,HEIF
 *     ispe,Image spatial extents,HEIF
 *     jpgC,JPEG image item configuration,HEIF
 *     lhvC,Layered HEVC Image item configuration,HEIF
 *     lsel,Layer selection,HEIF
 *     oinf,HEVC operating point,HEIF
 *     pasp,Pixel aspect ratio,HEIF
 *     pixi,Pixel information,HEIF
 *     rloc,Relative location,HEIF
 *     subs,Subsample information,HEIF
 *     tols,Layered HEVC Target output layer set property,HEIF
 * 
 * Descriptive properties:
 *     auxC,avcC,colr,hvcC,ispe,jpgC,lvhC,lsel,oinf,pasp,pixi,rloc,subs,tols
 * 
 * Transformative properties:
 *     clap,imir,irot
 * 
 * Essential properties:
 *     auxC,avcC,hvcC,lsel
 */

struct IPRP{
    IPRP*     next;
    uint32_t  size;
    uint8_t   type[4];
    union{
        struct{
            uint32_t cleanApertureWidthN,  cleanApertureWidthD;
            uint32_t cleanApertureHeightN, cleanApertureHeightD;
            uint32_t horizOffN,            horizOffD;
            uint32_t vertOffN,             vertOffD;
        } clap;
        struct{
            uint8_t  axis;
        } imir;
        struct{
            uint8_t  angle;
        } irot;
        struct{
            uint32_t image_width, image_height;
        } ispe;
        struct{
            uint32_t hSpacing, vSpacing;
        } pasp;
        struct{
            uint8_t  type[4];
            uint16_t colour_primaries;
            uint16_t transfer_characteristics;
            uint16_t matrix_coefficients;
            uint8_t  full_range_flag;/* Flag is in bit 7: 0x80 or 0x00 */
        } colr;
        struct{
            uint8_t  num_channels;
        } pixi;
        struct{
            uint16_t layer_id;
        } lsel;
    } u;
    
    uint8_t raw[];
};

/**
 * @brief State of the program.
 * 
 * There's lots of crap here.
 */

struct UNIVERSE{
    /* HEIF */
    ITEM*            item;
    ITEM*            items;
    size_t           num_items;
    
    /* FFmpeg Encode/Decode */
    struct{
        struct{uint32_t w, h;} tile;
        unsigned     crf;
        unsigned     chroma_format;
        const char*  x264_params;
        const char*  x265_params;
        const char*  url;
        int          fd;
    } out;
};



/* Static Function Prototypes */
static int   i2h_frame_fixup                      (AVFrame* f);
static int   i2h_frame_apply_graph                (AVFrame* out, const char* graphstr, AVFrame* in);
static int   i2h_frame_canonicalize_pixfmt        (AVFrame* out, AVFrame* in);
static int   i2h_frame_scale_and_pad              (AVFrame*           dst,
                                                   AVFrame*           src,
                                                   enum AVPixelFormat dst_pix_fmt,
                                                   int                dst_width,
                                                   int                dst_height,
                                                   int                tile_width,
                                                   int                tile_height);

static int   i2h_packetlist_append                (PACKETLIST** list, AVPacket* packet);
static void  i2h_packetlist_free                  (PACKETLIST** list);

static ITEM* i2h_item_new                         (void);
static int   i2h_item_init                        (ITEM* item);
static int   i2h_item_insert                      (ITEM** items, ITEM* item);
static int   i2h_item_type_is                     (ITEM* item, const char* type);
static int   i2h_item_type_is_picture             (ITEM* item);
static int   i2h_item_picture_fe_read_frame       (ITEM* item);
static int   i2h_item_picture_fe_make_grid_frame  (ITEM* item);
static int   i2h_item_picture_fe_make_thumb_frame (ITEM* item);
static int   i2h_item_picture_me_encoder_configure(ITEM* item, UNIVERSE* u);
static int   i2h_item_picture_me_encoder_deconfigure(ITEM* item);
static int   i2h_item_picture_me_push_grid_frames (ITEM* item);
static int   i2h_item_picture_me_push_thumb_frames(ITEM* item);
static int   i2h_item_picture_me_push_frame       (ITEM* item, AVFrame* frame);
static int   i2h_item_picture_me_pull_packets     (ITEM* item);
static int   i2h_item_picture_me_append_packet    (ITEM* item, AVPacket* packet);
static int   i2h_item_picture_me_strip_sei        (ITEM* item);
static int   i2h_item_picture_me_split_packet_list(ITEM* item);
static int   i2h_item_binary_me_append_packet     (ITEM* item, AVPacket* packet);

static int   i2h_iref_append                      (IREF** refs, IREF* iref);

static int   i2h_universe_init                    (UNIVERSE* u);
static int   i2h_universe_item_find_free_id       (UNIVERSE* u, uint32_t* id_out);
static ITEM* i2h_universe_item_find_by_id         (UNIVERSE* u, uint32_t  id);
static int   i2h_universe_item_handle             (UNIVERSE* u, ITEM* item);
static int   i2h_universe_item_handle_picture     (UNIVERSE* u, ITEM* item);
static int   i2h_universe_item_handle_binary      (UNIVERSE* u, ITEM* item);
static int   i2h_universe_flatten                 (UNIVERSE* u);
static int   i2h_universe_do_output               (UNIVERSE* u);



/*************************
 *** Utility Functions ***
 *************************/

/* Static Function Definitions */

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

static inline int   streq        (const char* s, const char* t){
    return !strcmp(s,t);
}
static inline int   strne        (const char* s, const char* t){
    return !streq(s,t);
}
static inline int   strstartswith(const char* s, const char* t){
    size_t ls = strlen(s), lt = strlen(t);
    return ls<lt ? 0 : !memcmp(s, t, lt);
}
static inline int   strendswith  (const char* s, const char* e){
    size_t ls = strlen(s), le = strlen(e);
    return ls<le ? 0 : !strcmp(s+ls-le, e);
}
static int          i2h_str2chromafmt(const char* s){
    if      (strcmp(s, "yuv400") == 0){return BENZINA_CHROMAFMT_YUV400;
    }else if(strcmp(s, "yuv420") == 0){return BENZINA_CHROMAFMT_YUV420;
    }else if(strcmp(s, "yuv422") == 0){return BENZINA_CHROMAFMT_YUV422;
    }else if(strcmp(s, "yuv444") == 0){return BENZINA_CHROMAFMT_YUV444;
    }else{
        return -100;
    }
}

/**
 * @brief Write fully, possibly in multiple attempts, or abort.
 * 
 * @param [in]  fd   The file descriptor.
 * @param [in]  buf  The buffer to write out.
 * @param [in]  len  The buffer's length.
 * @param [in]  off  The offset in the file descriptor at which to write.
 */

static ssize_t writefully (int fd, const void* buf, ssize_t len){
    ssize_t tot=0, b=1;
    
    while(b != 0 && tot < len){
        b = write(fd, buf, len);
        if(b<0)
            i2hmessageexit(EIO, stderr, "Error writing to FD=%d!\n", fd);
        if(b==0)
            i2hmessageexit(EIO, stderr, "Failed to write to FD=%d!\n", fd);
        tot += b;
        buf  = (const void*)((const char*)buf + b);
        len -= b;
    }
    
    return tot;
}
static ssize_t pwritefully(int fd, const void* buf, ssize_t len, off_t off){
    ssize_t tot=0, b=1;
    
    while(b != 0 && tot < len){
        b = pwrite(fd, buf, len, off);
        if(b<0)
            i2hmessageexit(EIO, stderr, "Error writing to FD=%d!\n", fd);
        if(b==0)
            i2hmessageexit(EIO, stderr, "Failed to write to FD=%d!\n", fd);
        tot += b;
        buf  = (const void*)((const char*)buf + b);
        len -= b;
        off += b;
    }
    
    return tot;
}

/**
 * @brief Get length and write length.
 * @param [in]  u    The universe we're operating over.
 * @param [out] off  A pointer to the offset.
 * @return 0
 */

static int i2h_universe_fdtell(UNIVERSE* u, size_t* off){
    *off = lseek(u->out.fd, 0, SEEK_CUR);
    return 0;
}
static int i2h_universe_writelen(UNIVERSE* u, size_t off, size_t end){
    uint32_t len = end-off;
    
    if(end-off > UINT32_MAX)
        i2hmessageexit(EINVAL, stderr, "ISO box size too large (%zu > %zu)!\n",
                       end-off, (size_t)UINT32_MAX);
    
    len = benz_htobe32(len);
    
    return pwritefully(u->out.fd, (char*)&len, sizeof(len), off) != 4;
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
            return AV_PIX_FMT_YUV410P;
        case AV_PIX_FMT_YUV411P:
            return AV_PIX_FMT_YUV411P;
        case AV_PIX_FMT_UYYVYY411:
            return AV_PIX_FMT_UYYVYY411;
        case AV_PIX_FMT_YUV420P9LE:
        case AV_PIX_FMT_YUV420P9BE:
        case AV_PIX_FMT_YUVA420P9LE:
        case AV_PIX_FMT_YUVA420P9BE:
            return AV_PIX_FMT_YUV420P9LE;
        case AV_PIX_FMT_YUV420P10LE:
        case AV_PIX_FMT_YUV420P10BE:
        case AV_PIX_FMT_YUVA420P10LE:
        case AV_PIX_FMT_YUVA420P10BE:
        case AV_PIX_FMT_P010LE:
        case AV_PIX_FMT_P010BE:
            return AV_PIX_FMT_YUV420P10LE;
        case AV_PIX_FMT_YUV420P12LE:
        case AV_PIX_FMT_YUV420P12BE:
            return AV_PIX_FMT_YUV420P12LE;
        case AV_PIX_FMT_YUV420P14LE:
        case AV_PIX_FMT_YUV420P14BE:
            return AV_PIX_FMT_YUV420P14LE;
        case AV_PIX_FMT_GBRP9LE:
        case AV_PIX_FMT_GBRP9BE:
            return AV_PIX_FMT_GBRP9LE;
        case AV_PIX_FMT_GRAY9LE:
        case AV_PIX_FMT_GRAY9BE:
            return AV_PIX_FMT_GRAY9LE;
        case AV_PIX_FMT_GBRP10LE:
        case AV_PIX_FMT_GBRP10BE:
        case AV_PIX_FMT_GBRAP10LE:
        case AV_PIX_FMT_GBRAP10BE:
            return AV_PIX_FMT_GBRP10LE;
        case AV_PIX_FMT_GRAY10LE:
        case AV_PIX_FMT_GRAY10BE:
            return AV_PIX_FMT_GRAY10LE;
        case AV_PIX_FMT_GBRP12LE:
        case AV_PIX_FMT_GBRP12BE:
        case AV_PIX_FMT_GBRAP12LE:
        case AV_PIX_FMT_GBRAP12BE:
            return AV_PIX_FMT_GBRP12LE;
        case AV_PIX_FMT_GRAY12LE:
        case AV_PIX_FMT_GRAY12BE:
            return AV_PIX_FMT_GRAY12LE;
        case AV_PIX_FMT_GBRP14LE:
        case AV_PIX_FMT_GBRP14BE:
            return AV_PIX_FMT_GBRP14LE;
        case AV_PIX_FMT_YUV422P:
        case AV_PIX_FMT_YUVA422P:
        case AV_PIX_FMT_UYVY422:
        case AV_PIX_FMT_YVYU422:
        case AV_PIX_FMT_YUYV422:
        case AV_PIX_FMT_NV16:
            return AV_PIX_FMT_YUV422P;
        case AV_PIX_FMT_YUV422P9LE:
        case AV_PIX_FMT_YUV422P9BE:
        case AV_PIX_FMT_YUVA422P9LE:
        case AV_PIX_FMT_YUVA422P9BE:
            return AV_PIX_FMT_YUV422P9LE;
        case AV_PIX_FMT_YUV422P10LE:
        case AV_PIX_FMT_YUV422P10BE:
        case AV_PIX_FMT_YUVA422P10LE:
        case AV_PIX_FMT_YUVA422P10BE:
        case AV_PIX_FMT_NV20LE:
        case AV_PIX_FMT_NV20BE:
            return AV_PIX_FMT_YUV422P10LE;
        case AV_PIX_FMT_YUV422P12LE:
        case AV_PIX_FMT_YUV422P12BE:
            return AV_PIX_FMT_YUV422P12LE;
        case AV_PIX_FMT_YUV422P14LE:
        case AV_PIX_FMT_YUV422P14BE:
            return AV_PIX_FMT_YUV422P14LE;
        case AV_PIX_FMT_YUV422P16LE:
        case AV_PIX_FMT_YUV422P16BE:
        case AV_PIX_FMT_YUVA422P16LE:
        case AV_PIX_FMT_YUVA422P16BE:
            return AV_PIX_FMT_YUV422P16LE;
        case AV_PIX_FMT_YUV440P:
            return AV_PIX_FMT_YUV440P;
        case AV_PIX_FMT_YUV440P10LE:
        case AV_PIX_FMT_YUV440P10BE:
            return AV_PIX_FMT_YUV440P10LE;
        case AV_PIX_FMT_YUV440P12LE:
        case AV_PIX_FMT_YUV440P12BE:
            return AV_PIX_FMT_YUV440P12LE;
        case AV_PIX_FMT_YUV444P9LE:
        case AV_PIX_FMT_YUV444P9BE:
        case AV_PIX_FMT_YUVA444P9LE:
        case AV_PIX_FMT_YUVA444P9BE:
            return AV_PIX_FMT_YUV444P9LE;
        case AV_PIX_FMT_YUV444P10LE:
        case AV_PIX_FMT_YUV444P10BE:
        case AV_PIX_FMT_YUVA444P10LE:
        case AV_PIX_FMT_YUVA444P10BE:
            return AV_PIX_FMT_YUV444P10LE;
        case AV_PIX_FMT_YUV444P12LE:
        case AV_PIX_FMT_YUV444P12BE:
            return AV_PIX_FMT_YUV444P12LE;
        case AV_PIX_FMT_YUV444P14LE:
        case AV_PIX_FMT_YUV444P14BE:
            return AV_PIX_FMT_YUV444P14LE;
        case AV_PIX_FMT_GBRPF32LE:
        case AV_PIX_FMT_GBRPF32BE:
        case AV_PIX_FMT_GBRAPF32LE:
        case AV_PIX_FMT_GBRAPF32BE:
            return AV_PIX_FMT_GBRPF32LE;
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
 * @brief Return the pixel format corresponding to a given Benzina chroma format.
 * 
 * @param [in]  chroma format.
 * @return The chroma format in question.
 */

static enum AVPixelFormat i2h_benzina2avpixfmt(int chroma_format){
    switch(chroma_format){
        case BENZINA_CHROMAFMT_YUV400: return AV_PIX_FMT_GRAY8;
        case BENZINA_CHROMAFMT_YUV420: return AV_PIX_FMT_YUV420P;
        case BENZINA_CHROMAFMT_YUV422: return AV_PIX_FMT_YUV422P;
        case BENZINA_CHROMAFMT_YUV444: return AV_PIX_FMT_YUV444P;
        default:
            i2hmessageexit(EINVAL, stderr, "Unsupported chroma format \"%u\"!\n",
                                           chroma_format);
    }
}

/**
 * @brief Return the Benzina/ISO chroma location corresponding to a given
 *        FFmpeg chroma location.
 * 
 * @param [in] chroma_location  The chroma location to convert.
 * @return The converted code.
 */

static int i2h_av2benzinachromaloc(enum AVChromaLocation chroma_location){
    switch(chroma_location){
        case AVCHROMA_LOC_UNSPECIFIED:
        case AVCHROMA_LOC_NB:
        default:                       return BENZINA_CHROMALOC_CENTER;
        case AVCHROMA_LOC_LEFT:        return BENZINA_CHROMALOC_LEFT;
        case AVCHROMA_LOC_CENTER:      return BENZINA_CHROMALOC_CENTER;
        case AVCHROMA_LOC_TOPLEFT:     return BENZINA_CHROMALOC_TOPLEFT;
        case AVCHROMA_LOC_TOP:         return BENZINA_CHROMALOC_TOP;
        case AVCHROMA_LOC_BOTTOMLEFT:  return BENZINA_CHROMALOC_BOTTOMLEFT;
        case AVCHROMA_LOC_BOTTOM:      return BENZINA_CHROMALOC_BOTTOM;
    }
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
static const char* i2h_x264_color_trc      (enum AVColorTransferCharacteristic color_trc){
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
static const char* i2h_x264_color_space    (enum AVColorSpace color_space){
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

/**
 * @brief Check whether the selected codec is x264 or x265.
 * 
 * @param [in]  codec  The codec in question.
 * @return Whether it is x264/x265 or not.
 */

static inline int i2h_is_codec_x264(AVCodec* codec){
    return streq(codec->name, "libx264");
}
static inline int i2h_is_codec_x265(AVCodec* codec){
    return streq(codec->name, "libx265");
}

/**
 * @brief Return whether the stream is of image/video type (contains pixel data).
 * 
 * @param [in]  avfctx  Format context whose streams we are inspecting.
 * @param [in]  stream  Integer number of the stream we are inspecting.
 * @return 0 if not a valid stream or not of image/video type; 1 otherwise.
 */

static inline int i2h_is_imagelike_stream(AVFormatContext* avfctx, int stream){
    if(stream < 0 || stream >= (int)avfctx->nb_streams){
        return 0;
    }
    return avfctx->streams[stream]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO;
}



/*****************************
 *** Application Functions ***
 *****************************/

/***** Frame *****/

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
 * @param [in,out]  f   The AVFrame whose attributes are to be fixed up.
 * @return 0.
 */

static int i2h_frame_fixup                        (AVFrame* f){
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
 * @brief Apply a filter graph described by a string to the input frame,
 *        producing an output frame.
 * 
 * The graph must have one default input and one default output.
 * 
 * @param [out] out       Output frame.
 * @param [in]  graphstr  String filtergraph description.
 * @param [in]  in        Input frame. The function increments the reference
 *                        count of the frame.
 * @return 0 if successful; !0 otherwise.
 */

static int i2h_frame_apply_graph                  (AVFrame*    out,
                                                   const char* graphstr,
                                                   AVFrame*    in){
    AVFilterGraph*     graph         = NULL;
    AVFilterInOut*     inputs        = NULL;
    AVFilterInOut*     outputs       = NULL;
    AVFilterContext*   buffersrcctx  = NULL;
    AVFilterContext*   buffersinkctx = NULL;
    const AVFilter*    buffersrc     = NULL;
    const AVFilter*    buffersink    = NULL;
    char               buffersrcargs[512] = {0};
    int                ret           = -EINVAL;
    
    
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
 * @brief Convert input frame to output with canonical pixel format.
 * 
 * @param [out]  out  Output frame, with canonical pixel format.
 * @param [in]   in   Input frame.
 * @return 0 if successful; !0 otherwise.
 */

static int i2h_frame_canonicalize_pixfmt          (AVFrame* out, AVFrame* in){
    const char* colorrangestr;
    int mpegrange;
    enum AVPixelFormat canonical = i2h_pick_pix_fmt(in->format);
    char graphstr[512] = {0};
    
    if(canonical == AV_PIX_FMT_NONE){
        i2hmessage(stderr, "Pixel format %s unsupported!\n",
                   av_get_pix_fmt_name(in->format));
        return -EINVAL;
    }
    
    mpegrange     = in->color_range == AVCOL_RANGE_MPEG;
    colorrangestr = mpegrange ? "mpeg" : "jpeg";
    snprintf(graphstr, sizeof(graphstr),
             "scale=in_range=%s:out_range=%s,format=pix_fmts=%s",
             colorrangestr, colorrangestr, av_get_pix_fmt_name(canonical));
    
    return i2h_frame_apply_graph(out, graphstr, in);
}

/**
 * @brief Scale/Pad frame to even number of tiles.
 * 
 * @param [out]  dst         Output frame.
 * @param [in]   src         Input frame.
 * @param [in]   dst_pix_fmt Output pixel format. Almost surely either
 *                           AV_PIX_FMT_YUV420P or AV_PIX_FMT_YUV444P. Must be
 *                           supported by pad/fillborders filter.
 * @param [in]   dst_width   Target width  for scaling, before padding.
 * @param [in]   dst_height  Target height for scaling, before padding.
 * @param [in]   tile_width  Tile width.  Output will be padded to a multiple
 *                           of this width.
 * @param [in]   tile_height Tile height. Output will be padded to a multiple
 *                           of this height.
 * @return 0 if successful; !0 otherwise.
 */

static int i2h_frame_scale_and_pad                (AVFrame*           dst,
                                                   AVFrame*           src,
                                                   enum AVPixelFormat dst_pix_fmt,
                                                   int                dst_width,
                                                   int                dst_height,
                                                   int                tile_width,
                                                   int                tile_height){
    char        graphstr[1024] = {0}, *p;
    const char* src_range, *src_space, *dst_space, *dst_fmt_name;
    int         i, l, src_is_rgb, src_is_yuv444, dst_scaled;
    int         padded_width, padded_height;
    
    
    /* Preliminaries. Compute various facts about the graph. */
    src_is_rgb    = src->format == AV_PIX_FMT_GBRP;
    src_is_yuv444 = src->format == AV_PIX_FMT_YUV444P;
    src_range     = src->color_range == AVCOL_RANGE_MPEG ? "mpeg" : "jpeg";
    src_space     = av_color_space_name(src->colorspace);
    src_space     = src_space  ? src_space : "bt470bg";
    dst_space     = src_is_rgb ? "bt470bg" : src_space;
    dst_scaled    = src->width != dst_width || src->height != dst_height;
    dst_fmt_name  = av_get_pix_fmt_name(dst_pix_fmt);
    padded_width  = (dst_width  + tile_width  - 1) / tile_width  * tile_width;
    padded_height = (dst_height + tile_height - 1) / tile_height * tile_height;
    
    
    /* snprintf the graph. */
    p = graphstr;
    l = sizeof(graphstr);
    i = 0;
    i = snprintf(p, l, "scale=in_range=%s:out_range=%s:in_color_matrix=%s:"
                       "out_color_matrix=%s",
                 src_range, src_range, src_space, dst_space);
    p += i;
    l -= i;
    if(dst_scaled){
        i = snprintf(p, l, ":w=%d:h=%d", dst_width, dst_height);
        p += i;
        l -= i;
    }
    if(src_is_rgb || src_is_yuv444 || dst_scaled){
        i = snprintf(p, l, ",format=pix_fmts=yuv444p");
        p += i;
        l -= i;
    }
    i = snprintf(p, l, ",pad=%d:%d:0:0,fillborders=0:%d:0:%d:smear",
                 padded_width,
                 padded_height,
                 padded_width  - dst_width,
                 padded_height - dst_height);
    p += i;
    l -= i;
    i = snprintf(p, l, ",scale=in_range=%s:out_range=%s:in_color_matrix=%s:"
                       "out_color_matrix=%s,format=pix_fmts=%s",
                 src_range, src_range, dst_space, dst_space, dst_fmt_name);
    
    
    /* Apply and return. */
    return i2h_frame_apply_graph(dst, graphstr, src);
}



/***** Packet List *****/

/**
 * @brief Append a packet to the list.
 * 
 * @param [in,out]  List of packets to modify. Must be non-NULL.
 * @param [in]      Packet to append. Must be non-NULL.
 * @return 0 if successful, !0 otherwise.
 */

static int   i2h_packetlist_append                (PACKETLIST** list, AVPacket* packet){
    PACKETLIST* node;
    
    if(!list || !packet){return 1;}
    
    while(*list){list = &(*list)->next;}
    
    node = malloc(sizeof(*node));
    node->packet = packet;
    node->next   = NULL;
    
    *list = node;
    
    return 0;
}

/**
 * @brief Free a list of packets.
 * 
 * @param [in]  A list of packets to be freed. 
 */

static void  i2h_packetlist_free                  (PACKETLIST** list){
    PACKETLIST* node;
    PACKETLIST* next;
    
    if(!list){return;}
    
    for(node=*list; node; node=next){
        next = node->next;
        node->next = NULL;
        av_packet_free(&node->packet);
        free(node);
    }
    
    *list = node;
}



/***** Item *****/

/**
 * @brief Allocate new item.
 * 
 * @return The new item.
 */

static ITEM* i2h_item_new                         (void){
    ITEM* item = malloc(sizeof(*item));
    i2h_item_init(item);
    return item;
}

/**
 * @brief Initialize the item to default values.
 * 
 * @param [in]  item   The item to initialize.
 * @return 0.
 */

static int   i2h_item_init                        (ITEM* item){
    memset(item, 0, sizeof(*item));
    item->name           = "";
    item->mime_type      = "application/octet-stream";
    memcpy(item->type, "hvc1", 4);
    return 0;
}

/**
 * @brief Insert the item in sorted order in the list.
 * 
 * @param [in,out]  items  The head of the linked list of items.
 * @param [in,out]  item   The item to be inserted.
 * @return 0.
 */

static int   i2h_item_insert                      (ITEM** items, ITEM* item){
    while(*items && (*items)->id < item->id){
        items = &(*items)->next;
    }
    item->next = *items;
    *items     = item;
    return 0;
}

/**
 * @brief Check whether item is of given type.
 * 
 * @param [in]  item        The item whose type we are checking.
 * @param [in]  type        The type we are comparing against.
 * @return 1 if item is of given type; 0 otherwise.
 */

static int   i2h_item_type_is                     (ITEM* item, const char* type){
    return memcmp(item->type, type, sizeof(item->type)) == 0;
}

/**
 * @brief Check whether item is of picture type.
 * 
 * @param [in]  item        The item whose type we are checking.
 * @return 1 if pict-type item; 0 otherwise.
 */

static int   i2h_item_type_is_picture             (ITEM* item){
    return i2h_item_type_is(item, "jpeg") ||
           i2h_item_type_is(item, "avc1") ||
           i2h_item_type_is(item, "hvc1");
}

/**
 * @brief Read frame of picture at its URL.
 * 
 * @param [in,out]  item        The item for which we're reading the frame.
 * @return 0 if successful; !0 otherwise.
 */

static int   i2h_item_picture_fe_read_frame       (ITEM* item){
    int                ret;
    AVFormatContext*   format_ctx    = NULL;
    int                format_stream = 0;
    AVCodecParameters* codec_par     = NULL;
    AVCodec*           codec         = NULL;
    AVCodecContext*    codec_ctx     = NULL;
    AVPacket*          packet        = NULL;
    
    
    /* Input Init */
    ret   = avformat_open_input(&format_ctx, item->args.path, NULL, NULL);
    if(ret < 0)
        i2hmessageexit(ret, stderr, "Cannot open input media \"%s\"! (%d)\n", item->args.path, ret);
    
    ret   = avformat_find_stream_info(format_ctx, NULL);
    if(ret < 0)
        i2hmessageexit(EIO,    stderr, "Cannot understand input media \"%s\"! (%d)\n", item->args.path, ret);
    
    format_stream = av_find_default_stream_index(format_ctx);
    if(!i2h_is_imagelike_stream(format_ctx, format_stream))
        i2hmessageexit(EINVAL, stderr, "No image data in media \"%s\"!\n", item->args.path);
    
    item->frontend.pict.is_source_jpeg = format_ctx->iformat == av_find_input_format("jpeg");
    codec_par = format_ctx->streams[format_stream]->codecpar;
    codec     = avcodec_find_decoder(codec_par->codec_id);
    codec_ctx = avcodec_alloc_context3(codec);
    if(!codec_ctx)
        i2hmessageexit(EINVAL, stderr, "Could not allocate decoder!\n");
    
    ret = avcodec_open2(codec_ctx, codec, NULL);
    if(ret < 0)
        i2hmessageexit(ret, stderr, "Could not open decoder! (%d)\n", ret);
    
    
    /* Read first (only) frame. */
    do{
        packet = av_packet_alloc();
        if(!packet)
            i2hmessageexit(ENOMEM, stderr, "Failed to allocate packet object!\n");
        
        ret = av_read_frame(format_ctx, packet);
        if(ret < 0)
            i2hmessageexit(ret, stderr, "Error or end-of-file reading media \"%s\" (%d)!\n",
                                        item->args.path, ret);
        
        if(packet->stream_index != format_stream){
            av_packet_free(&packet);
        }else{
            ret = avcodec_send_packet(codec_ctx, packet);
            av_packet_free(&packet);
            if(ret)
                i2hmessageexit(ret, stderr, "Error pushing packet! (%d)\n", ret);
            
            item->frontend.pict.frame = av_frame_alloc();
            if(!item->frontend.pict.frame)
                i2hmessageexit(ENOMEM, stderr, "Error allocating frame!\n");
            
            ret = avcodec_receive_frame(codec_ctx, item->frontend.pict.frame);
            if(ret)
                i2hmessageexit(ret, stderr, "Error pulling frame! (%d)\n",  ret);
            break;
        }
    }while(!ret);
    
    
    /* Cleanup input reader objects */
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&format_ctx);
    
    
    /* Fixup and canonicalize pixel format. */
    i2h_frame_fixup(item->frontend.pict.frame);
    i2h_frame_canonicalize_pixfmt(item->frontend.pict.frame,
                                  item->frontend.pict.frame);
    
    
    /* Record image spatial-extent and clean-aperture information. */
    item->frontend.pict.thumb_ispe.w     = item->args.tile.w;
    item->frontend.pict.thumb_ispe.h     = item->args.tile.h;
    item->frontend.pict.grid_clap.w      = item->frontend.pict.frame->width;
    item->frontend.pict.grid_clap.h      = item->frontend.pict.frame->height;
    item->frontend.pict.grid_ispe.w      = (item->frontend.pict.grid_clap.w + item->args.tile.w - 1)/item->args.tile.w*item->args.tile.w;
    item->frontend.pict.grid_ispe.h      = (item->frontend.pict.grid_clap.h + item->args.tile.h - 1)/item->args.tile.h*item->args.tile.h;
    item->frontend.pict.single_tile      = item->frontend.pict.grid_clap.w <= item->args.tile.w &&
                                           item->frontend.pict.grid_clap.h <= item->args.tile.h;
    item->frontend.pict.single_tile_ltbx = (uint64_t)item->frontend.pict.grid_clap.w*item->args.tile.h >=
                                           (uint64_t)item->frontend.pict.grid_clap.h*item->args.tile.w;
    if(item->frontend.pict.single_tile){
        item->frontend.pict.thumb_clap.w  = item->frontend.pict.grid_clap.w;
        item->frontend.pict.thumb_clap.h  = item->frontend.pict.grid_clap.h;
    }else if(item->frontend.pict.single_tile_ltbx){
        item->frontend.pict.thumb_clap.w = item->args.tile.w;
        item->frontend.pict.thumb_clap.h = round(item->frontend.pict.thumb_clap.w * 1.0 *
                                                 item->frontend.pict.grid_clap.h /
                                                 item->frontend.pict.grid_clap.w);
    }else{
        item->frontend.pict.thumb_clap.h = item->args.tile.h;
        item->frontend.pict.thumb_clap.w = round(item->frontend.pict.thumb_clap.h * 1.0 *
                                                 item->frontend.pict.grid_clap.w /
                                                 item->frontend.pict.grid_clap.h);
    }
    
    
    /* Select codec. */
    if      (i2h_item_type_is(item, "avc1")){
        item->frontend.pict.codec = avcodec_find_encoder_by_name("libx264");
        if(!item->frontend.pict.codec)
            i2hmessageexit(EINVAL, stderr, "Cannot transcode! Please rebuild Benzina with an "
                                           "FFmpeg configured with --enable-libx264.\n");
    }else if(i2h_item_type_is(item, "hvc1")){
        item->frontend.pict.codec = avcodec_find_encoder_by_name("libx265");
        if(!item->frontend.pict.codec)
            i2hmessageexit(EINVAL, stderr, "Cannot transcode! Please rebuild Benzina with an "
                                           "FFmpeg configured with --enable-libx265.\n");
    }else{
        i2hmessageexit(EINVAL, stderr, "Unknown item type %4.4s!\n", item->type);
    }
    
    
    /* Exit */
    return ret;
}

/**
 * @brief Make padded grid frame for image item.
 * 
 * @param [in,out]  item    The item for which we're creating a padded grid frame.
 * @return 0 if successful; !0 otherwise.
 */

static int   i2h_item_picture_fe_make_grid_frame  (ITEM* item){
    item->frontend.pict.grid_frame = av_frame_alloc();
    if(!item->frontend.pict.grid_frame)
        i2hmessageexit(ENOMEM, stderr, "Error allocating frame!\n");
    
    return i2h_frame_scale_and_pad(item->frontend.pict.grid_frame,
                                   item->frontend.pict.frame,
                                   item->args.pix_fmt,
                                   item->frontend.pict.grid_clap.w,
                                   item->frontend.pict.grid_clap.h,
                                   item->args.tile.w,
                                   item->args.tile.h);
}

/**
 * @brief Make thumbnail frame for image item.
 * 
 * If the item does not require a thumbnail, skip.
 * 
 * @param [in,out]  item    The item for which we're creating a thumbnail frame.
 * @return 0 if successful; !0 otherwise.
 */

static int   i2h_item_picture_fe_make_thumb_frame (ITEM* item){
    if(!item->args.want_thumbnail)
        return 0;
    
    item->frontend.pict.thumb_frame = av_frame_alloc();
    if(!item->frontend.pict.thumb_frame)
        i2hmessageexit(ENOMEM, stderr, "Error allocating frame!\n");
    
    if(item->frontend.pict.single_tile){
        return av_frame_ref(item->frontend.pict.thumb_frame,
                            item->frontend.pict.grid_frame);
    }else{
        return i2h_frame_scale_and_pad(item->frontend.pict.thumb_frame,
                                       item->frontend.pict.frame,
                                       item->args.pix_fmt,
                                       item->frontend.pict.thumb_clap.w,
                                       item->frontend.pict.thumb_clap.h,
                                       item->args.tile.w,
                                       item->args.tile.h);
    }
}

/**
 * @brief Configure/Deconfigure encoder for encoding the given prototypical frame.
 * 
 * Uses both frame properties and sticky universe properties to configure the
 * encoder.
 * 
 * @param [in,out]  item        The item we're configuring for.
 * @param [in]      frame       The frame whose properties (except size) we use
 *                              as a template for encoding.
 * @param [in,out]  u           The universe whose sticky flags we are using.
 * @return 0 if successful; !0 otherwise.
 */

static int   i2h_item_picture_me_encoder_configure(ITEM*     item,
                                                   UNIVERSE* u){
    int           ret;
    char          x26x_params_str[1024];
    const char*   x26x_params;
    const char*   x26x_params_args;
    const char*   x26x_profile;
    const char*   colorrange_str;
    const char*   colorprim_str;
    const char*   transfer_str;
    const char*   colormatrix_str;
    AVDictionary* codec_ctx_opt = NULL;
    AVFrame*      frame = item->frontend.pict.frame;
    unsigned      crf = u->out.crf;
    
    
    /* Allocate context. */
    item->middleend.pict.codec_ctx = avcodec_alloc_context3(item->frontend.pict.codec);
    if(!item->middleend.pict.codec_ctx)
        i2hmessageexit(ENOMEM, stderr, "Could not allocate encoder!\n");
    
    
    /* Configure context. */
    colorprim_str   = i2h_x264_color_primaries(frame->color_primaries);
    colorprim_str   = colorprim_str   ? colorprim_str   : i2h_x264_color_primaries(AVCOL_PRI_BT709);
    transfer_str    = i2h_x264_color_trc      (frame->color_trc);
    transfer_str    = transfer_str    ? transfer_str    : i2h_x264_color_trc      (AVCOL_TRC_IEC61966_2_1);
    colormatrix_str = i2h_x264_color_space    (frame->colorspace);
    colormatrix_str = colormatrix_str ? colormatrix_str : i2h_x264_color_space    (AVCOL_SPC_BT470BG);
    if(i2h_is_codec_x264(item->frontend.pict.codec)){
        x26x_params      = "x264-params";
        x26x_params_args = u->out.x264_params ? u->out.x264_params : "";
        x26x_profile     = "high444";
        colorrange_str   = frame->color_range==AVCOL_RANGE_MPEG ? "on" : "off";
        snprintf(x26x_params_str, sizeof(x26x_params_str),
                 "ref=1:chromaloc=%d:log=-1:log-level=none:stitchable=1:slices-max=32:"
                 "fullrange=%s:colorprim=%s:transfer=%s:colormatrix=%s%s%s:"
                 "annexb=0",
                 i2h_av2benzinachromaloc(frame->chroma_location),
                 colorrange_str, colorprim_str, transfer_str, colormatrix_str,
                 *x26x_params_args ? ":"              : "",
                 *x26x_params_args ? x26x_params_args : "");
    }else{
        x26x_params      = "x265-params";
        x26x_params_args = u->out.x265_params ? u->out.x265_params : "";
        x26x_profile     = "main444-stillpicture";
        colorrange_str   = frame->color_range==AVCOL_RANGE_MPEG ? "limited" : "full";
        snprintf(x26x_params_str, sizeof(x26x_params_str),
                 "ref=1:chromaloc=%d:log=-1:log-level=none:info=0:"
                 "range=%s:colorprim=%s:transfer=%s:colormatrix=%s%s%s:"
                 "annexb=0:temporal-layers=0",
                 i2h_av2benzinachromaloc(frame->chroma_location),
                 colorrange_str, colorprim_str, transfer_str, colormatrix_str,
                 *x26x_params_args ? ":"              : "",
                 *x26x_params_args ? x26x_params_args : "");
    }
    item->middleend.pict.codec_ctx->width                  = item->args.tile.w;
    item->middleend.pict.codec_ctx->height                 = item->args.tile.h;
    item->middleend.pict.codec_ctx->time_base.num          =  1;  /* Nominal 30 fps */
    item->middleend.pict.codec_ctx->time_base.den          = 30;  /* Nominal 30 fps */
    item->middleend.pict.codec_ctx->gop_size               =  0;  /* Intra only */
    item->middleend.pict.codec_ctx->pix_fmt                = frame->format;
    item->middleend.pict.codec_ctx->chroma_sample_location = frame->chroma_location;
    item->middleend.pict.codec_ctx->color_range            = frame->color_range;
    item->middleend.pict.codec_ctx->color_primaries        = frame->color_primaries;
    item->middleend.pict.codec_ctx->color_trc              = frame->color_trc;
    item->middleend.pict.codec_ctx->colorspace             = frame->colorspace;
    item->middleend.pict.codec_ctx->sample_aspect_ratio    = frame->sample_aspect_ratio;
    item->middleend.pict.codec_ctx->flags                 |= AV_CODEC_FLAG_LOOP_FILTER |
                                                             AV_CODEC_FLAG_GLOBAL_HEADER;
    item->middleend.pict.codec_ctx->thread_count           = 1;
    item->middleend.pict.codec_ctx->thread_type            = FF_THREAD_FRAME;
    item->middleend.pict.codec_ctx->slices                 = 1;
    av_dict_set    (&codec_ctx_opt, "profile",    x26x_profile,    0);
    av_dict_set    (&codec_ctx_opt, "preset",     "placebo",       0);
    av_dict_set    (&codec_ctx_opt, "tune",       "ssim",          0);
    av_dict_set_int(&codec_ctx_opt, "forced-idr", 1,               0);
    av_dict_set_int(&codec_ctx_opt, "crf",        crf,             0);
    av_dict_set    (&codec_ctx_opt, x26x_params,  x26x_params_str, 0);
    
    
    /* Open context. */
    ret = avcodec_open2(item->middleend.pict.codec_ctx,
                        item->frontend.pict.codec,
                        &codec_ctx_opt);
    if(av_dict_count(codec_ctx_opt)){
        AVDictionaryEntry* entry = NULL;
        i2hmessage(stderr, "Warning: codec ignored %d options!\n",
                           av_dict_count(codec_ctx_opt));
        while((entry=av_dict_get(codec_ctx_opt, "", entry, AV_DICT_IGNORE_SUFFIX))){
            i2hmessage(stderr, "    %20s: %s\n", entry->key, entry->value);
        }
    }
    av_dict_free(&codec_ctx_opt);
    
    
    /* Exit */
    return ret;
}
static int   i2h_item_picture_me_encoder_deconfigure(ITEM* item){
    avcodec_free_context(&item->middleend.pict.codec_ctx);
    return 0;
}

/**
 * @brief Push picture grid/tile frames.
 * 
 * @param [in,out]  u           The universe we're operating over.
 * @param [in,out]  item        The item for which a grid and tile frames are to
 *                              be encoded.
 * @return 0 if successful; !0 otherwise.
 */

static int   i2h_item_picture_me_push_grid_frames (ITEM* item){
    int j, tile_w=item->args.tile.w;
    int i, tile_h=item->args.tile.h;
    int grid_rows=item->frontend.pict.grid_ispe.h/tile_h;
    int grid_cols=item->frontend.pict.grid_ispe.w/tile_w;
    int ret;
    
    if(item->args.want_thumbnail &&
       item->frontend.pict.single_tile){
        /**
         * The single tile to encode was already pushed in for
         * the thumbnail image.
         */
        
        return 0;
    }
    
    for(i=0;i<grid_rows;i++){
        for(j=0;j<grid_cols;j++){
            AVFrame* frame = av_frame_clone(item->frontend.pict.grid_frame);
            frame->crop_top    = tile_h * (i);
            frame->crop_bottom = tile_h * (grid_rows-i-1);
            frame->crop_left   = tile_w * (j);
            frame->crop_right  = tile_w * (grid_cols-j-1);
            
            ret = av_frame_apply_cropping(frame, 0);
            if(ret < 0)
                i2hmessageexit(ret, stderr, "Error applying cropping (%d)!\n", ret);
            
            ret = i2h_item_picture_me_push_frame(item, frame);
            if(ret != 0)
                i2hmessageexit(ret, stderr, "Error pushing grid tile into encoder (%d)!\n", ret);
            
            av_frame_free(&frame);
        }
    }
    
    return 0;
}
static int   i2h_item_picture_me_push_thumb_frames(ITEM* item){
    int ret;
    
    if(!item->args.want_thumbnail)
        return 0;
    
    ret = i2h_item_picture_me_push_frame(item, item->frontend.pict.thumb_frame);
    if(ret != 0)
        i2hmessageexit(ret, stderr, "Error pushing thumbnail tile into encoder (%d)!\n", ret);
    
    return ret;
}

/**
 * @brief Push a frame into the encoder with respect to this item.
 * @param [in,out] item    Item for which we push in frames.
 * @param [in]     frame   Frame to push in. If NULL, this is a flush frame
 *                         that signals the encoder to empty itself.
 * @return 0 if successful, !0 otherwise.
 */

static int   i2h_item_picture_me_push_frame       (ITEM* item, AVFrame* frame){
    int retpush, retpull=0;
    
    if(frame)
        frame->pts = frame->best_effort_timestamp;
    
    retpush = avcodec_send_frame(item->middleend.pict.codec_ctx, frame);
    
    if(retpush == AVERROR(EAGAIN)){
        /**
         * The encoder is still open (if it wasn't, we would have gotten AVERROR_EOF),
         * but needs to be emptied of input first.
         */
        
        retpull = i2h_item_picture_me_pull_packets(item);
        
        if(retpull == AVERROR_EOF)
            return 0;
        
        if(retpull == 0)
            i2hmessageexit(retpull, stderr, "Impossible error, pull packets ended early??? (%d)\n", retpull);
        
        if(retpull == AVERROR(ENOMEM))
            i2hmessageexit(retpull, stderr, "Pulling packets failed, out of memory! (%d)\n", retpull);
        
        if(retpull != AVERROR(EAGAIN))
            i2hmessageexit(retpull, stderr, "Error while pulling frames from encoder! (%d)\n", retpull);
        
        retpush = avcodec_send_frame(item->middleend.pict.codec_ctx, frame);
        
        if(retpush == AVERROR(EAGAIN))
            i2hmessageexit(retpush, stderr, "Impossible error, push=EAGAIN after pull=EAGAIN (%d)\n", retpush);
    }
    
    if(frame)
        return retpush;
    
    if(retpush != AVERROR_EOF && retpush != 0)
        return retpush;
    
    /**
     * If we get here, it's a flush frame and it successfully signalled to the
     * encoder to flush itself. Reap any remaining frames from the encoder.
     */
    
    retpull = i2h_item_picture_me_pull_packets(item);
    
    if(retpull == AVERROR_EOF)
        return 0;
    
    if(retpull == 0)
        i2hmessageexit(retpull, stderr, "Impossible error, pull packets ended early??? (%d)\n", retpull);
    
    if(retpull == AVERROR(ENOMEM))
        i2hmessageexit(retpull, stderr, "Pulling packets failed, out of memory! (%d)\n", retpull);
    
    if(retpull != AVERROR(EAGAIN))
        i2hmessageexit(retpull, stderr, "Error while pulling frames from encoder! (%d)\n", retpull);
    
    i2hmessageexit(retpull, stderr, "Impossible error, pull=EAGAIN after flush! (%d)\n", retpull);
}

/**
 * @brief Pull as many packets as possible from the encoder.
 * 
 * @param [in,out] item    Item for which to pull out packets.
 * @return A non-zero avcodec_receive_packet() error code representing the
 *         reason why no more frames can be retrieved. Usually will be
 *           - AVERROR(EAGAIN), when more input frames must be pushed into the encoder, or
 *           - AVERROR_EOF,     when the encoder has been fully emptied.
 */

static int   i2h_item_picture_me_pull_packets     (ITEM* item){
    AVPacket* packet;
    int       ret;
    
    do{
        packet = av_packet_alloc();
        if(!packet)
            i2hmessageexit(ENOMEM, stderr, "Failed to allocate packet object!\n");
        
        ret = avcodec_receive_packet(item->middleend.pict.codec_ctx, packet);
        if(ret == 0){
            i2h_item_picture_me_append_packet(item, packet);
        }else{
            av_packet_free(&packet);
        }
    }while(ret == 0);
    
    return ret;
}

/**
 * @brief Append packet to picture item's packet list.
 * 
 * @param [in,out] item    Item to which to append a packet.
 * @param [in]     packet  Packet to append.
 * @return 0 if successful, !0 otherwise.
 */

static int   i2h_item_picture_me_append_packet    (ITEM* item, AVPacket* packet){
    return i2h_packetlist_append(&item->middleend.pict.packet_list, packet);
}

/**
 * @brief Strip SEI NALUs from picture item's packet list.
 * 
 * @param [in,out] item    Item from which to strip SEI NALUs.
 * @return 0 if successful, !0 otherwise.
 */

static int   i2h_item_picture_me_strip_sei        (ITEM* item){
    PACKETLIST* l;
    AVPacket*   p;
    uint8_t*    srco;/* Pointer into original packet */
    uint8_t*    dstr;/* Pointer into in-place rewritten packet */
    uint8_t*    endo;/* Pointer to tail of original packet */
    uint64_t    sz, minnalusz, trimmed;
    int       (*is_sei)(const void*);
    
    int h264 = i2h_is_codec_x264(item->frontend.pict.codec),
        h265 = i2h_is_codec_x265(item->frontend.pict.codec);
    
    
    /**
     * We assume from now on here AVCC/HEVC 4-byte framing. That is, all
     * NALUs are prefixed with their size in bytes as a 32-bit
     * big-endian value. This size does *not* include the prefix itself.
     * 
     * The shortest framed AVCC-framed H.264 NALU out is 5 bytes: A 4-byte
     * prefix and a 1-byte header.
     * 
     * The shortest framed HEVC-framed H.265 NALU out is 6 bytes: A 4-byte
     * prefix and a 2-byte header.
     */
    
    if(h265){
        minnalusz = 6;
        is_sei    = benz_itu_h265_nalu_is_sei;
        return 0;
    }else if(h264){
        minnalusz = 5;
        /* is_sei    = benz_itu_h264_nalu_is_sei;*/
        return 0;/* TODO: H.264 SEI stripping support not yet implemented! */
    }else{
        return 0;
    }
    
    
    /**
     * For each packet separately, we strip out all SEI NALUs. There may be
     * multiple packets and multiple (SEI) NALUs per packet.
     */
    
    for(l=item->middleend.pict.packet_list; l; l=l->next){
        p    = l->packet;
        dstr = p->data;
        srco = dstr;
        endo = srco + p->size;
        
        for(; srco <= endo-minnalusz; srco += sz){
            sz  = benz_iso_bmff_as_u32(srco);
            sz += 4;
            if(is_sei(srco+4))
                continue;
            
            memmove(dstr, srco, sz);
            dstr += sz;
        }
        
        trimmed = srco-dstr;
        av_shrink_packet(p, p->size-trimmed);
    }
    
    return 0;
}

/**
 * @brief Split packet list between grid and thumbnail packets.
 * 
 * If a picture item is single-item and does want a thumbnail, then nominally
 * assign the packet to its thumbnail.
 * 
 * @param [in,out] item  Item for which to split the packet list.
 * @return 0 if successful, !0 otherwise.
 */

static int   i2h_item_picture_me_split_packet_list(ITEM* item){
    if(item->frontend.pict.single_tile){
        if(item->args.want_thumbnail){
            item->middleend.pict.grid_packet_list  = NULL;
            item->middleend.pict.thumb_packet_list = item->middleend.pict.packet_list;
        }else{
            item->middleend.pict.grid_packet_list  = item->middleend.pict.packet_list;
            item->middleend.pict.thumb_packet_list = NULL;
        }
    }else{
        if(item->args.want_thumbnail){
            item->middleend.pict.grid_packet_list  = item->middleend.pict.packet_list->next;
            item->middleend.pict.thumb_packet_list = item->middleend.pict.packet_list;
            item->middleend.pict.thumb_packet_list->next = NULL;
        }else{
            item->middleend.pict.grid_packet_list  = item->middleend.pict.packet_list;
            item->middleend.pict.thumb_packet_list = NULL;
        }
    }
    
    item->middleend.pict.packet_list = NULL;
    
    return 0;
}

/**
 * @brief Append packet to binary item's packet list.
 * 
 * @param [in,out] item    Item to which to append a packet.
 * @param [in]     packet  Packet to append.
 * @return 0 if successful, !0 otherwise.
 */

static int   i2h_item_binary_me_append_packet     (ITEM* item, AVPacket* packet){
    return i2h_packetlist_append(&item->middleend.binary.packet_list, packet);
}


/***** Item Reference *****/

/**
 * @brief Append the set of ID references to a linked list of them.
 * 
 * @param [in,out]  refs  The head of the linked list
 * @param [in,out]  iref  The set to be appended.
 * @return 0.
 */

static int  i2h_iref_append                       (IREF** refs, IREF* iref){
    iref->next = *refs;
    *refs = iref;
    return 0;
}



/***** Universe *****/

/**
 * @brief Initialize the universe to default values.
 * 
 * @param [in]  u  The universe we're operating over.
 * @return 0.
 */

static int  i2h_universe_init                     (UNIVERSE* u){
    u->items              = NULL;
    u->item               = i2h_item_new();
    u->num_items          = 0;
    
    u->out.tile.w         = 512;
    u->out.tile.h         = 512;
    u->out.crf            = 10;
    u->out.chroma_format  = BENZINA_CHROMAFMT_YUV420;
    u->out.x264_params    = "";
    u->out.x265_params    = "";
    u->out.url            = "out.heif";
    
    return 0;
}

/**
 * @brief Find an item by its ID.
 * @param [in]  u   The universe we're operating over.
 * @param [in]  id  The ID of the item we're looking for.
 * @return The item, if found; NULL otherwise.
 */

static ITEM* i2h_universe_item_find_by_id         (UNIVERSE* u, uint32_t id){
    ITEM* item;
    for(item=u->items; item; item=item->next){
        if(item->id == id){
            return item;
        }
    }
    
    return NULL;
}

/**
 * @brief Find lowest free item ID.
 * 
 * @param [in]  u       The universe we're operating over.
 * @param [out] id_out  The ID that was found.
 * @return 0 if no free ID available; !0 otherwise.
 */

static int  i2h_universe_item_find_free_id        (UNIVERSE* u, uint32_t* id_out){
    uint32_t id;
    ITEM*    item;
    
    if(!id_out){
        return 0;
    }
    
    if(!u->items){
        *id_out = 0;
        return 1;
    }
    
    for(id=0,item=u->items; item; id++,item=item->next){
        if(item->id != id){
            *id_out = id;
            return 1;
        }
    }
    
    return 0;
}

/**
 * @brief Append a new item to the list
 * @param [in]  u     The universe we're operating over.
 * @param [in]  item  The completed item we'd like to process.
 * @return 0 if successful; !0 otherwise.
 */

static int  i2h_universe_item_append              (UNIVERSE* u, ITEM* item){
    i2h_item_insert(&u->items, item);
    u->num_items++;
    return 0;
}

/**
 * @brief Handle the completion of a new item.
 * 
 * @param [in]  u     The universe we're operating over.
 * @param [in]  item  The completed item we'd like to process.
 * @return 0 if successful; !0 otherwise.
 */

static int  i2h_universe_item_handle              (UNIVERSE* u, ITEM* item){
    /**
     * The following item types may be specified via the command line:
     * 
     *   - avc1
     *   - Exif
     *   - hvc1
     *   - jpeg
     *   - mime
     *   - uri%20
     * 
     * The avc1/hvc1 types may be autosubstituted with the following, on an
     * as-needed basis:
     * 
     *   - iden
     *   - grid
     */
    
    if      (i2h_item_type_is_picture(item)){
        return i2h_universe_item_handle_picture(u, item);
    }else if(i2h_item_type_is(item, "mime") ||
             i2h_item_type_is(item, "Exif") ||
             i2h_item_type_is(item, "uri ")){
        return i2h_universe_item_handle_binary(u, item);
    }else{
        i2hmessageexit(EINVAL, stderr, "Unknown item type %4.4s!\n", item->type);
    }
    return 1;
}

/**
 * @brief Handle the completion of a new picture item
 * @param [in]  u     The universe we're operating over.
 * @param [in]  item  The completed item we'd like to process.
 * @return 0.
 */

static int  i2h_universe_item_handle_picture      (UNIVERSE* u, ITEM* item){
    /* FFmpeg state. */
    int ret;
    
    
    /**
     * PREP:
     *    1. Copy current desired tile properties:
     *        1.1 Width
     *        1.2 Height
     *        1.3 Pixel format
     */
    
    item->args.tile.w  = u->out.tile.w;
    item->args.tile.h  = u->out.tile.h;
    item->args.pix_fmt = i2h_benzina2avpixfmt(u->out.chroma_format);
    
    
    /**
     * FRONTEND:
     *    1. Read one frame
     *    2. Fixup and canonicalize the frame
     *    3. Make (padded) grid image.
     *    4. (Optional) Make thumbnail image.
     */
    
    ret = i2h_item_picture_fe_read_frame(item);
    if(ret != 0)
        i2hmessageexit(ret, stderr, "Failed to read frame from input media \"%s\"! (%d)\n", item->args.path, ret);
    ret = i2h_item_picture_fe_make_grid_frame(item);
    if(ret != 0)
        i2hmessageexit(ret, stderr, "Failed to make grid for input \"%s\"! (%d)\n", item->args.path, ret);
    ret = i2h_item_picture_fe_make_thumb_frame(item);
    if(ret != 0)
        i2hmessageexit(ret, stderr, "Failed to make thumbnail for input \"%s\"! (%d)\n", item->args.path, ret);
    
    
    /**
     * MIDDLEEND:
     *    1. Configure encoder.
     *    2. (Optional) Encode thumbnail tile.
     *    3. Encode grid tiles.
     *    4. Flush encoder.
     *    5. Strip SEI NALUs.
     *    6. Release encoder.
     *    7. Split/duplicate single list of packets between thumbnail and grid tiles.
     */
    
    ret = i2h_item_picture_me_encoder_configure(item, u);
    if(ret < 0)
        i2hmessageexit(ret, stderr, "Could not open encoder! (%d)\n", ret);
    ret = i2h_item_picture_me_push_thumb_frames(item);
    if(ret < 0)
        i2hmessageexit(ret, stderr, "Error pushing thumbnail frame into encoder! (%d)\n", ret);
    ret = i2h_item_picture_me_push_grid_frames(item);
    if(ret < 0)
        i2hmessageexit(ret, stderr, "Error pushing grid frames into encoder! (%d)\n", ret);
    ret = i2h_item_picture_me_push_frame(item, NULL);
    if(ret < 0)
        i2hmessageexit(ret, stderr, "Failed to flush encoder! (%d)\n", ret);
    ret = i2h_item_picture_me_strip_sei(item);
    if(ret < 0)
        i2hmessageexit(ret, stderr, "Failed to strip SEI NALUs! (%d)\n", ret);
    ret = i2h_item_picture_me_encoder_deconfigure(item);
    if(ret < 0)
        i2hmessageexit(ret, stderr, "Failed to deallocate encoder! (%d)\n", ret);
    ret = i2h_item_picture_me_split_packet_list(item);
    if(ret < 0)
        i2hmessageexit(ret, stderr, "Failed to split packet list! (%d)\n", ret);
    
    
#if 1
    fprintf(stdout, "ID %u: %ux%u %s %s\n",
            item->id, item->frontend.pict.frame->width, item->frontend.pict.frame->height,
            av_get_pix_fmt_name(item->frontend.pict.frame->format),
            item->frontend.pict.frame->color_range == AVCOL_RANGE_JPEG ? "jpeg" : "mpeg");
    fprintf(stdout, "Grid:  %ux%u %s %s\n",
            item->frontend.pict.grid_frame->width, item->frontend.pict.grid_frame->height,
            av_get_pix_fmt_name(item->frontend.pict.grid_frame->format),
            item->frontend.pict.grid_frame->color_range == AVCOL_RANGE_JPEG ? "jpeg" : "mpeg");
    if(item->args.want_thumbnail){
        fprintf(stdout, "Tile:  %ux%u %s %s (%d encoded bytes)\n",
                item->frontend.pict.thumb_frame->width, item->frontend.pict.thumb_frame->height,
                av_get_pix_fmt_name(item->frontend.pict.thumb_frame->format),
                item->frontend.pict.thumb_frame->color_range == AVCOL_RANGE_JPEG ? "jpeg" : "mpeg",
                item->middleend.pict.thumb_packet_list->packet->size);
    }
#endif
#if 0
    FILE* fp;
    fp = fopen("grid.yuv", "w");
    fwrite(item->grid_frame->data[0], item->grid_frame->width*item->grid_frame->height/1, 1, fp);
    fwrite(item->grid_frame->data[1], item->grid_frame->width*item->grid_frame->height/4, 1, fp);
    fwrite(item->grid_frame->data[2], item->grid_frame->width*item->grid_frame->height/4, 1, fp);
    fclose(fp);
    fp = fopen("tile.yuv", "w");
    fwrite(item->tile_frame->data[0], item->tile_frame->width*item->tile_frame->height/1, 1, fp);
    fwrite(item->tile_frame->data[1], item->tile_frame->width*item->tile_frame->height/4, 1, fp);
    fwrite(item->tile_frame->data[2], item->tile_frame->width*item->tile_frame->height/4, 1, fp);
    fclose(fp);
    exit(1);
    
    av_frame_free(&item->frame);
    av_packet_free(&packet);
#endif
    
    return 0;
}

/**
 * @brief Handle the completion of a new MIME item
 * @param [in]  u     The universe we're operating over.
 * @param [in]  item  The completed item we'd like to process.
 * @return 0.
 */

static int  i2h_universe_item_handle_binary       (UNIVERSE* u, ITEM* item){
    (void)u;
    
    int         fd, ret;
    struct stat buf;
    ssize_t     len;
    AVPacket*   packet;
    
    fd = open(item->args.path, O_RDONLY|O_CLOEXEC);
    if(fd<0)
        i2hmessageexit(EIO, stderr, "Error opening path '%s'!\n", item->args.path);
    
    if(fstat(fd, &buf) < 0)
        i2hmessageexit(EIO, stderr, "Cannot stat path '%s'!\n", item->args.path);
    
    if(buf.st_size > INT_MAX)
        i2hmessageexit(EINVAL, stderr, "Oversize file '%s' >%d bytes!\n", item->args.path, INT_MAX);
    
    packet = av_packet_alloc();
    if(!packet)
        i2hmessageexit(ENOMEM, stderr, "Cannot allocate packet!\n");
    
    ret = av_new_packet(packet, buf.st_size);
    if(ret != 0)
        i2hmessageexit(ENOMEM, stderr, "Cannot allocate %zu bytes of memory for packet!\n", (size_t)buf.st_size);
    
    len = read(fd, packet->data, packet->size);
    if(len != (ssize_t)packet->size)
        i2hmessageexit(EIO, stderr, "Failed to read '%s' fully!\n", item->args.path);
    
    close(fd);
    
    if(i2h_item_binary_me_append_packet(item, packet) != 0)
        i2hmessageexit(ENOMEM, stderr, "Failed to append packet to list!\n");
    
    return 0;
}

/**
 * @brief Transform the universe's tree of items into a single flat list ordered by ID.
 * 
 * @param [in]  u  The universe we're operating over.
 * @return 0 if successful; !0 otherwise.
 */

static int  i2h_universe_flatten                  (UNIVERSE* u){
    ITEM* i;
    ITEM* n;
    
    for(i=u->items; i; i=i->next){
        if(i2h_item_type_is_picture(i)){
            if(i->args.want_thumbnail){
                n = i2h_item_new();
                i2h_universe_item_find_free_id(u, &n->id);
                n->args.path = i->args.path;
                n->name = "";
                memcpy(n->type, i->type, 4);
                n->primary = 0;
                n->hidden = 0;
                n->args.want_thumbnail = 0;
                
            }
        }
    }
    
    return 0;
}

/**
 * @brief Output a HEIF file with the given items.
 * 
 * @param [in]  u  The universe we're operating over.
 * @return 0 if successful; !0 otherwise.
 */

static int  i2h_universe_do_output_i8                 (UNIVERSE*        u, uint8_t  v){
    writefully(u->out.fd, &v, sizeof(v));
    return 0;
}
static int  i2h_universe_do_output_i16                (UNIVERSE*        u, uint16_t v){
    v = benz_htobe16(v);
    writefully(u->out.fd, &v, sizeof(v));
    return 0;
}
static int  i2h_universe_do_output_i32                (UNIVERSE*        u, uint32_t v){
    v = benz_htobe32(v);
    writefully(u->out.fd, &v, sizeof(v));
    return 0;
}
static int  i2h_universe_do_output_i64                (UNIVERSE*        u, uint64_t v){
    v = benz_htobe64(v);
    writefully(u->out.fd, &v, sizeof(v));
    return 0;
}
static int  i2h_universe_do_output_bytes              (UNIVERSE*        u, const void* buf, size_t len){
    writefully(u->out.fd, buf, len);
    return 0;
}
static int  i2h_universe_do_output_ascii              (UNIVERSE*        u, const void* str){
    return i2h_universe_do_output_bytes(u, str, strlen(str));
}
static int  i2h_universe_do_output_asciz              (UNIVERSE*        u, const void* str){
    return i2h_universe_do_output_bytes(u, str, strlen(str)+1);/* *with* NUL-terminator */
}
static int  i2h_universe_do_output_box                (UNIVERSE*        u, const char name[static 4]){
    i2h_universe_do_output_i32  (u, 0);
    i2h_universe_do_output_bytes(u, name, 4);
    return 0;
}
static int  i2h_universe_do_output_fullbox            (UNIVERSE*        u, const char name[static 4], uint32_t version, uint32_t flags){
    version <<= 24;
    version  |= flags & 0x00FFFFFFU;
    i2h_universe_do_output_box(u, name);
    i2h_universe_do_output_i32(u, version);
    return 0;
}
static int  i2h_universe_do_output_mdat               (UNIVERSE*        u){
    ITEM* i;
    
    size_t start, end;
    i2h_universe_fdtell(u, &start);
    i2h_universe_do_output_box(u, "mdat");
    for(i=u->items; i; i=i->next){
        if(i2h_item_type_is_picture(i)){
            PACKETLIST* pkt;
            for(pkt=i->middleend.pict.thumb_packet_list;pkt;pkt=pkt->next){
                writefully(u->out.fd, pkt->packet->data, pkt->packet->size);
            }
            for(pkt=i->middleend.pict.grid_packet_list;pkt;pkt=pkt->next){
                writefully(u->out.fd, pkt->packet->data, pkt->packet->size);
            }
        }
    }
    i2h_universe_fdtell(u, &end);
    i2h_universe_writelen(u, start, end);
    return 0;
}
static int  i2h_universe_do_output_meta_hdlr          (UNIVERSE*        u, const char* track_name){
    size_t start, end;
    
    track_name = track_name ? track_name : "";
    
    i2h_universe_fdtell(u, &start);
    i2h_universe_do_output_fullbox(u, "hdlr", 0, 0);
    i2h_universe_do_output_i32  (u, 0);         /* predefined */
    i2h_universe_do_output_ascii(u, "pict");    /* handler */
    i2h_universe_do_output_i32  (u, 0);         /* reserved[0] */
    i2h_universe_do_output_i32  (u, 0);         /* reserved[1] */
    i2h_universe_do_output_i32  (u, 0);         /* reserved[2] */
    i2h_universe_do_output_asciz(u, track_name);/* name */
    i2h_universe_fdtell(u, &end);
    i2h_universe_writelen(u, start, end);
    return 0;
}
static int  i2h_universe_do_output_meta_pitm          (UNIVERSE*        u){
    ITEM* i;
    uint32_t primaryid = u->items ? u->items->id : 0;
    size_t start, end;
    i2h_universe_fdtell(u, &start);
    i2h_universe_do_output_fullbox(u, "pitm", 1, 0);
    for(i=u->items; i; i=i->next){
        if(i->primary){
            primaryid = i->id;
            break;
        }
    }
    i2h_universe_do_output_i32(u, primaryid);
    i2h_universe_fdtell(u, &end);
    i2h_universe_writelen(u, start, end);
    return 0;
}
static int  i2h_universe_do_output_meta_dinf_dref_url (UNIVERSE*        u, const char* url){
    size_t start, end;
    i2h_universe_fdtell(u, &start);
    i2h_universe_do_output_fullbox(u, "url ", 0, !url);
    if(url){
        /* url == NULL indicates "same file". */
        i2h_universe_do_output_asciz(u, url);
    }
    i2h_universe_fdtell(u, &end);
    i2h_universe_writelen(u, start, end);
    return 0;
}
static int  i2h_universe_do_output_meta_dinf_dref     (UNIVERSE*        u){
    size_t start, end;
    i2h_universe_fdtell(u, &start);
    i2h_universe_do_output_fullbox(u, "dref", 0, 0);
    i2h_universe_do_output_i32(u, 1);
    i2h_universe_do_output_meta_dinf_dref_url(u, NULL);
    i2h_universe_fdtell(u, &end);
    i2h_universe_writelen(u, start, end);
    return 0;
}
static int  i2h_universe_do_output_meta_dinf          (UNIVERSE*        u){
    size_t start, end;
    i2h_universe_fdtell(u, &start);
    i2h_universe_do_output_box(u, "dinf");
    i2h_universe_do_output_meta_dinf_dref(u);
    i2h_universe_fdtell(u, &end);
    i2h_universe_writelen(u, start, end);
    return 0;
}
static int  i2h_universe_do_output_meta_iloc          (UNIVERSE*        u){
    size_t start, end, i;
    i2h_universe_fdtell(u, &start);
    i2h_universe_do_output_fullbox(u, "iloc", 2, 0);
    i2h_universe_do_output_i16(u, 0x4400);
    
    i2h_universe_do_output_i32(u, u->num_items);
    for(i=0;i<u->num_items;i++){
        
    }
    i2h_universe_fdtell(u, &end);
    i2h_universe_writelen(u, start, end);
    return 0;
}
static int  i2h_universe_do_output_meta_infe          (UNIVERSE*        u){
    size_t start, end, i;
    i2h_universe_fdtell(u, &start);
    i2h_universe_do_output_fullbox(u, "infe", 1, 0);
    
    i2h_universe_do_output_i32(u, u->num_items);
    for(i=0;i<u->num_items;i++){
        
    }
    i2h_universe_fdtell(u, &end);
    i2h_universe_writelen(u, start, end);
    return 0;
}
static int  i2h_universe_do_output_meta_iinf          (UNIVERSE*        u){
    size_t start, end, i;
    i2h_universe_fdtell(u, &start);
    i2h_universe_do_output_fullbox(u, "iinf", 1, 0);
    
    i2h_universe_do_output_i32(u, u->num_items);
    for(i=0;i<u->num_items;i++){
        
    }
    i2h_universe_fdtell(u, &end);
    i2h_universe_writelen(u, start, end);
    return 0;
}
static int  i2h_universe_do_output_meta_iref          (UNIVERSE*        u){
    size_t start, end;
    i2h_universe_fdtell(u, &start);
    i2h_universe_do_output_fullbox(u, "iref", 1, 0);
    for(;0;){
        //SingleItemTypeReferenceBoxLarge
    }
    i2h_universe_fdtell(u, &end);
    i2h_universe_writelen(u, start, end);
    return 0;
}
static int  i2h_universe_do_output_meta_iprp_ipco     (UNIVERSE*        u){
    ITEM* i;
    size_t start, end;
    
    i2h_universe_fdtell(u, &start);
    i2h_universe_do_output_box(u, "ipco");
    i2h_universe_do_output_bytes(u, "\x00\x00\x00\x10pixi\x00\x00\x00\x00\x03\x08\x08\x08", 16);// Item Property 1: 3-plane, 8-bit color.
    for(i=u->items;i;i=i->next){
        if(i2h_item_type_is_picture(i)){
            // ItemProperty or ItemFullProperty
            av_pix_fmt_desc_get(i->frontend.pict.grid_frame->format);
        }
    }
    i2h_universe_fdtell(u, &end);
    i2h_universe_writelen(u, start, end);
    return 0;
}
static int  i2h_universe_do_output_meta_iprp_ipma     (UNIVERSE*        u){
    size_t start, end;
    i2h_universe_fdtell(u, &start);
    i2h_universe_do_output_fullbox(u, "ipma", 1, 1);
    for(;0;){
        // Associations
    }
    i2h_universe_fdtell(u, &end);
    i2h_universe_writelen(u, start, end);
    return 0;
}
static int  i2h_universe_do_output_meta_iprp          (UNIVERSE*        u){
    size_t start, end;
    i2h_universe_fdtell(u, &start);
    i2h_universe_do_output_box(u, "iprp");
    i2h_universe_do_output_meta_iprp_ipco(u);
    i2h_universe_do_output_meta_iprp_ipma(u);
    i2h_universe_fdtell(u, &end);
    i2h_universe_writelen(u, start, end);
    return 0;
}
static int  i2h_universe_do_output_meta               (UNIVERSE*        u){
    size_t start, end;
    i2h_universe_fdtell(u, &start);
    i2h_universe_do_output_fullbox(u, "meta", 0, 0);
    i2h_universe_do_output_meta_hdlr(u, "");
    i2h_universe_do_output_meta_pitm(u);
    i2h_universe_do_output_meta_dinf(u);
    i2h_universe_do_output_meta_iloc(u);
    i2h_universe_do_output_meta_iinf(u);
    i2h_universe_do_output_meta_iref(u);
    i2h_universe_do_output_meta_iprp(u);
    i2h_universe_fdtell(u, &end);
    i2h_universe_writelen(u, start, end);
    return 0;
}
static int  i2h_universe_do_output_ftyp               (UNIVERSE*        u){
    ITEM* i;
    int uses_h264=0, uses_h265=0;
    
    /* Sweep over the file to find codecs in use. */
    for(i=u->items; i; i=i->next){
        if(i2h_item_type_is_picture(i)){
            uses_h264 |= i2h_is_codec_x264(i->frontend.pict.codec);
            uses_h265 |= i2h_is_codec_x265(i->frontend.pict.codec);
        }
    }
    
    /* Write out header and compatible brands. */
    if(uses_h265 && uses_h264){
        writefully(u->out.fd, "\x00\x00\x00\x1c" "ftyp" "heic"
                              "\x00\x00\x00\x00" "mif1avciheic", 28);
    }else if(uses_h265){
        writefully(u->out.fd, "\x00\x00\x00\x18" "ftyp" "heic"
                              "\x00\x00\x00\x00" "mif1heic", 24);
    }else if(uses_h264){
        writefully(u->out.fd, "\x00\x00\x00\x18" "ftyp" "avci"
                              "\x00\x00\x00\x00" "mif1avci", 24);
    }else{
        writefully(u->out.fd, "\x00\x00\x00\x14" "ftyp" "mif1"
                              "\x00\x00\x00\x00" "mif1", 20);
    }
    
    return 0;
}
static int  i2h_universe_do_output                    (UNIVERSE*        u){
    #if 0
    ITEM* i;
    i = u->items;
    for(;i;i=i->next){
        printf("id=%3u,name=%s\n", i->id, i->name);
    }
    #endif
    
    /**
     * BACKEND:
     *    1. Create new items and fill them.
     *    2. If some item is not single-tile,
     *        2.1 Transmute to "grid".
     *        2.2 Compute grid references.
     *    3. Flatten Item Hierarchy
     *    4. Open, write out and close the file.
     */
    
    i2h_universe_flatten(u);
    
    /* Open File */
    u->out.fd = open(u->out.url, O_CREAT|O_RDWR|O_TRUNC|O_CLOEXEC, 0644);
    if(u->out.fd<0)
        i2hmessageexit(EIO, stderr, "Failed to open/create file %s for writing!\n",
                       u->out.url);
    
    i2h_universe_do_output_ftyp(u);
    i2h_universe_do_output_meta(u);
    i2h_universe_do_output_mdat(u);
    
    /* Close File */
    close(u->out.fd);
    return 0;
}



/***** Argument-parsing *****/

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

static int i2h_parse_shortopt   (char*** argv){
    int ret = **argv            &&
             (**argv)[0] == '-' &&
             (**argv)[1] != '-' &&
             (**argv)[1] != '\0';
    if(ret){
        ++**argv;
    }
    return ret;
}
static int i2h_parse_longopt    (char*** argv, const char* opt){
    size_t l   = strlen(opt);
    int    ret = strncmp(**argv, opt, l) == 0 &&
                       ((**argv)[l] == '\0' ||
                        (**argv)[l] == '=');
    if(ret){
        **argv += l;
    }
    return ret;
}
static int i2h_parse_consumestr (char*** argv, char** str_out){
    *str_out = *(*argv)++;
    return !!*str_out;
}
static int i2h_parse_consumecstr(char*** argv, const char** str_out){
    *str_out = *(*argv)++;
    return !!*str_out;
}
static int i2h_parse_consumeequals(char*** argv){
    switch(***argv){
        case '\0': return !!*++*argv;
        case '=':  return !!++**argv;
        default:   return 0;
    }
}
static int i2h_parse_noconsume  (char*** argv){
    switch(***argv){
        case '\0': return (void)*++*argv, 1;
        case '=':  i2hmessageexit(EINVAL, stderr, "Option takes no arguments!\n");
        default:   return 1;
    }
}
static int i2h_parse_tile       (UNIVERSE* u, char*** argv){
    char tile_chroma_format[21] = "yuv420";
    
    if(!i2h_parse_consumeequals(argv))
        i2hmessageexit(EINVAL, stderr, "Option --tile consumes an argument!\n");
    
    switch(sscanf(*(*argv)++, "%"PRIu32":%"PRIu32":%20[yuv420]",
                  &u->out.tile.w,
                  &u->out.tile.h,
                  tile_chroma_format)){
        default:
        case 0: i2hmessageexit(EINVAL, stderr, "--tile requires an argument formatted 'width:height:chroma_format' !\n");
        case 1: u->out.tile.h = u->out.tile.w; break;
        case 2:
        case 3: break;
    }
    u->out.chroma_format = i2h_str2chromafmt(tile_chroma_format);
    if(u->out.tile.w < 48 || u->out.tile.w > 4096 || u->out.tile.w % 16 != 0)
        i2hmessageexit(EINVAL, stderr, "Tile width must be 48-4096 pixels in multiples of 16!\n");
    if(u->out.tile.h < 48 || u->out.tile.h > 4096 || u->out.tile.h % 16 != 0)
        i2hmessageexit(EINVAL, stderr, "Tile height must be 48-4096 pixels in multiples of 16!\n");
    if(u->out.chroma_format > BENZINA_CHROMAFMT_YUV444)
        i2hmessageexit(EINVAL, stderr, "Unknown chroma format \"%s\"!\n", tile_chroma_format);
    if(u->out.chroma_format != BENZINA_CHROMAFMT_YUV420 &&
       u->out.chroma_format != BENZINA_CHROMAFMT_YUV444)
        i2hmessageexit(EINVAL, stderr, "The only supported chroma formats are yuv420 and yuv444!\n");
    
    return 1;
}
static int i2h_parse_x264_params(UNIVERSE* u, char*** argv){
    if(!i2h_parse_consumeequals(argv) || !i2h_parse_consumecstr(argv, &u->out.x264_params))
        i2hmessageexit(EINVAL, stderr, "Option --x264-params consumes an argument!\n");
    return 1;
}
static int i2h_parse_x265_params(UNIVERSE* u, char*** argv){
    if(!i2h_parse_consumeequals(argv) || !i2h_parse_consumecstr(argv, &u->out.x265_params))
        i2hmessageexit(EINVAL, stderr, "Option --x265-params consumes an argument!\n");
    return 1;
}
static int i2h_parse_crf        (UNIVERSE* u, char*** argv){
    char* p;
    
    if(!i2h_parse_consumeequals(argv) || !i2h_parse_consumestr(argv, &p))
        i2hmessageexit(EINVAL, stderr, "Option --crf consumes an argument!\n");
    
    u->out.crf = strtoull(p, &p, 0);
    if(u->out.crf > 51 || *p != '\0')
        i2hmessageexit(EINVAL, stderr, "Error: --crf must be integer in range [0, 51]!\n");
    
    return 1;
}
static int i2h_parse_hidden     (UNIVERSE* u, char*** argv){
    if(!i2h_parse_noconsume(argv))
        i2hmessageexit(EINVAL, stderr, "Option --hidden does not consume an argument!\n");
    u->item->hidden         = 1;
    return 1;
}
static int i2h_parse_primary    (UNIVERSE* u, char*** argv){
    if(!i2h_parse_noconsume(argv))
        i2hmessageexit(EINVAL, stderr, "Option --primary does not consume an argument!\n");
    u->item->primary        = 1;
    return 1;
}
static int i2h_parse_thumbnail  (UNIVERSE* u, char*** argv){
    if(!i2h_parse_noconsume(argv))
        i2hmessageexit(EINVAL, stderr, "Option --thumb does not consume an argument!\n");
    u->item->args.want_thumbnail = 1;
    return 1;
}
static int i2h_parse_name       (UNIVERSE* u, char*** argv){
    if(!i2h_parse_consumeequals(argv) || !i2h_parse_consumecstr(argv, &u->item->name))
        i2hmessageexit(EINVAL, stderr, "Option --name consumes an argument!\n");
    return 1;
}
static int i2h_parse_mime       (UNIVERSE* u, char*** argv){
    if(!i2h_parse_consumeequals(argv) || !i2h_parse_consumecstr(argv, &u->item->mime_type))
        i2hmessageexit(EINVAL, stderr, "Option --mime consumes an argument!\n");
    return 1;
}
static int i2h_parse_ref        (UNIVERSE* u, char*** argv){
    char* p, *s;
    uint32_t r    = 0;
    int      n    = 0;
    IREF*    iref = calloc(1, sizeof(*iref));
    
    /* Consume the = and argument if it's there. */
    if(!i2h_parse_consumeequals(argv) || !i2h_parse_consumestr(argv, &p))
        i2hmessageexit(EINVAL, stderr, "Option --ref consumes an argument!\n");
    
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
    i2h_iref_append(&u->item->refs, iref);
    
    return 1;
}
static int i2h_parse_item       (UNIVERSE* u, char*** argv){
    char* p, *q;
    int   id_set = 0;
    
    /* Consume the = and argument if it's there. */
    if(!i2h_parse_consumeequals(argv) || !i2h_parse_consumestr(argv, &p))
        i2hmessageexit(EINVAL, stderr, "Option --item consumes an argument!\n");
    
    /* Parse the argument. */
    while(1){
        if      (strstartswith(p, "id=")){
            if(id_set || id_set++)
                i2hmessageexit(EINVAL, stderr, "Item ID has already been set!\n");
            
            q = p+3;
            u->item->id = strtoull(q, &p, 0);
            if(p==q)
                i2hmessageexit(EINVAL, stderr, "id= requires a 32-bit unsigned integer item ID!\n");
            if(i2h_universe_item_find_by_id(u, u->item->id))
                i2hmessageexit(EINVAL, stderr, "Pre-existing item with same ID %"PRIu32"!\n", u->item->id);
        }else if(strstartswith(p, "type=")){
            p += 5;
            memcpy(u->item->type, p, 4);
            p += 4;
        }else if(strstartswith(p, "path=")){
            u->item->args.path = (p+=5);
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
    
    if(strnlen(u->item->type, 4) != 4)
        i2hmessageexit(EINVAL, stderr, "The type= key must be exactly 4 ASCII characters!\n");
    if(!id_set && !i2h_universe_item_find_free_id(u, &u->item->id))
        i2hmessageexit(EINVAL, stderr, "Too many items added, no item IDs remaining!\n");
    
    /* Process completed item. */
    i2h_universe_item_append(u, u->item);
    i2h_universe_item_handle(u, u->item);
    u->item = i2h_item_new();
    
    return 1;
}
static int i2h_parse_output     (UNIVERSE* u, char*** argv){
    if(!i2h_parse_consumeequals(argv) || !i2h_parse_consumecstr(argv, &u->out.url))
        i2hmessageexit(EINVAL, stderr, "Option --output consumes an argument!\n");
    return 1;
}
static int i2h_parse_help       (UNIVERSE* u, char*** argv){
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
    
    return i2h_universe_do_output(u);
}

/**
 * @brief Main Function
 * 
 * Supports the following arguments:
 * 
 *   "Sticky" Parameters (preserved across picture items until changed)
 *     --tile=W:H[:F]           Tile configuration. Width:Height:Chroma Format. Default 512:512:yuv420.
 *     --x264-params=S          Additional x264 parameters. Default "".
 *     --x265-params=S          Additional x265 parameters. Default "".
 *     --crf=N                  CRF 0-51. Default 10.
 * 
 *   "Unsticky" Parameters (used for only one item)
 *     -H, --hidden             Set hidden bit. Default false. Incompatible with -p.
 *     -p, --primary            Mark as primary item. Default false. Incompatible with -H.
 *     -t, --thumb              Want (single-tile) thumbnail.
 *     -n, --name=S             Name of item.
 *     -m, --mime=S             MIME-type of item. Default "application/octet-stream".
 *     -r, --ref=type:id[,id]*  Typed reference. Can be specified multiple times.
 * 
 *   Item Declaration
 *     -i, --item=[id=N,][type=T,]path=S   Source file. Examples:
 *     -i path=image.png        Add item with next-lowest-numbered ID of type
 *                              "hvc1" from image.png
 *     -i id=4,path=image.png   Add item with ID=4 of type "hvc1" from image.png
 *     -i id=4,type=hvc1,path=image.png 
 *                              Add item with ID=4 of type "hvc1" from image.png
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
    (void)argc, (void)argv++;
    
    i2h_universe_init(u);
    return i2h_parse_args(u, &argv);
}



#if 0
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

#if 0
aligned(8) class Box(unsigned int(32) boxtype, optional unsigned int(8)[16] extended_type){
    unsigned int(32) size;
    unsigned int(32) type = boxtype;
    if(size==1){
        unsigned int(64) largesize;
    }else if(size==0){
        // box extends to end of file
    }
    if(boxtype=='uuid'){
        unsigned int(8)[16] usertype = extended_type;
    }
}

aligned(8) class FullBox(unsigned int(32) boxtype, unsigned int(8) v, bit(24) f) extends Box(boxtype){
    unsigned int(8) version = v;
    bit(24)         flags   = f;
}

aligned(8) class FileTypeBox extends Box('ftyp'){
    unsigned int(32) major_brand;
    unsigned int(32) minor_version;
    unsigned int(32) compatible_brands[]; // to end of the box
}

aligned(8) class MetaBox(handler_type) extends FullBox('meta', version=0, 0){
    HandlerBox(handler_type) theHandler;
    PrimaryItemBox     primary_resource; // optional
    DataInformationBox file_locations;   // optional
    ItemLocationBox    item_locations;   // optional
    ItemProtectionBox  protections;      // optional
    ItemInfoBox        item_infos;       // optional
    IPMPControlBox     IPMP_control;     // optional
    ItemReferenceBox   item_refs;        // optional
    ItemDataBox        item_data;        // optional
    Box                other_boxes[];    // optional
    //  E.g. ItemPropertiesBox item_properties;
}

aligned(8) class ItemProperty(property_type) extends Box(property_type){}

aligned(8) class ItemFullProperty(property_type, version, flags) extends FullBox(property_type, version, flags){}

aligned(8) class ItemPropertyContainerBox extends Box('ipco'){
    properties ItemProperty()[]; // boxes derived from
                                 // ItemProperty or ItemFullProperty, to fill box
}

aligned(8) class ItemPropertyAssociation extends FullBox('ipma', version, flags){
    unsigned int(32) entry_count;
    for(i=0; i<entry_count; i++){
        if(version < 1)
            unsigned int(16) item_ID;
        else
            unsigned int(32) item_ID;
        unsigned int(8) association_count;
        for(i=0; i<association_count; i++){
            bit(1) essential;
            if(flags & 1)
                unsigned int(15) property_index;
            else
                unsigned int(7) property_index;
        }
    }
}

aligned(8) class ItemPropertiesBox extends Box('iprp'){
    ItemPropertyContainerBox  property_container;
    ItemPropertyAssociation   association[];
}

aligned(8) class ImageSpatialExtentsProperty extends ItemFullProperty('ispe', version=0, flags=0){
    unsigned int(32) image_width;
    unsigned int(32) image_height;
}

class PixelAspectRatioBox extends Box('pasp'){
    unsigned int(32) hSpacing;
    unsigned int(32) vSpacing;
}

class CleanApertureBox extends Box('clap'){
    unsigned int(32) cleanApertureWidthN;
    unsigned int(32) cleanApertureWidthD;
    unsigned int(32) cleanApertureHeightN;
    unsigned int(32) cleanApertureHeightD;
    unsigned int(32) horizOffN;
    unsigned int(32) horizOffD;
    unsigned int(32) vertOffN;
    unsigned int(32) vertOffD;
}

aligned(8) class ColourInformationBox extends Box('colr'){
    unsigned int(32) colour_type;
    if (colour_type == 'nclx'){/* on-screen colours */
        unsigned int(16) colour_primaries;
        unsigned int(16) transfer_characteristics;
        unsigned int(16) matrix_coefficients;
        unsigned int(1)  full_range_flag;
        unsigned int(7)  reserved = 0;
    }else if(colour_type == 'rICC'){
        ICC_profile; // restricted ICC profile
    }else if(colour_type == 'prof'){
        ICC_profile; // unrestricted ICC profile
    }
}

aligned(8) class PixelInformationProperty extends ItemFullProperty('pixi', version=0, flags=0){
    unsigned int(8) num_channels;
    for(i=0; i<num_channels; i++){
        unsigned int(8) bits_per_channel;
    }
}

aligned(8) class ImageRotation extends ItemProperty('irot'){
    unsigned int(6) reserved = 0;
    unsigned int(2) angle;
}

aligned(8) class ImageMirror extends ItemProperty('imir'){
    unsigned int(7) reserved = 0;
    unsigned int(1) axis;
}

aligned(8) class LayerSelectorProperty extends ItemProperty('lsel'){
    unsigned int(16) layer_id;
}

aligned(8) class MetaBox(handler_type) extends FullBox('meta', version = 0, 0){
    HandlerBox(handler_type) theHandler;
    PrimaryItemBox           primary_resource; // optional
    DataInformationBox       file_locations;   // optional
    ItemLocationBox          item_locations;   // optional
    ItemProtectionBox        protections;      // optional
    ItemInfoBox              item_infos;       // optional
    IPMPControlBox           IPMP_control;     // optional
    ItemReferenceBox         item_refs;        // optional
    ItemDataBox              item_data;        // optional
    Box                      other_boxes[];    // optional
}

aligned(8) class ItemLocationBox extends FullBox('iloc', version, 0){
    unsigned int(4) offset_size;
    unsigned int(4) length_size;
    unsigned int(4) base_offset_size;
    if((version == 1) || (version == 2)){
        unsigned int(4) index_size;
    }else{
        unsigned int(4) reserved;
    }
    if(version < 2){
        unsigned int(16) item_count;
    }else if(version == 2){
        unsigned int(32) item_count;
    }
    for(i=0; i<item_count; i++){
        if(version < 2){
            unsigned int(16) item_ID;
        }else if(version == 2){
            unsigned int(32) item_ID;
        }
        if((version == 1) || (version == 2)){
            unsigned int(12) reserved = 0;
            unsigned int(4)  construction_method;
        }
        unsigned int(16)                 data_reference_index;
        unsigned int(base_offset_size*8) base_offset;
        unsigned int(16)                 extent_count;
        for(j=0; j<extent_count; j++){
            if(((version == 1) || (version == 2)) && (index_size > 0)){
                unsigned int(index_size*8) extent_index;
            }
            unsigned int(offset_size*8) extent_offset;
            unsigned int(length_size*8) extent_length;
        }
    }
}

aligned(8) class ItemInfoBox extends FullBox('iinf', version, 0){
    if (version == 0) {
        unsigned int(16) entry_count;
    } else {
        unsigned int(32) entry_count;
    }
    ItemInfoEntry[entry_count] item_infos;
}

aligned(8) class ItemInfoEntry extends FullBox('infe', version, 0){
    if((version == 0) || (version == 1)){
        unsigned int(16) item_ID;
        unsigned int(16) item_protection_index
        string item_name;
        string content_type;
        string content_encoding; //optional
    }
    if(version == 1){
        unsigned int(32) extension_type; //optional
        ItemInfoExtension(extension_type); //optional
    }
    if(version >= 2){
        if(version == 2) {
            unsigned int(16) item_ID;
        }else if(version == 3){
            unsigned int(32) item_ID;
        }
        unsigned int(16) item_protection_index;
        unsigned int(32) item_type;
        string item_name;
        if(item_type=='mime'){
            string content_type;
            string content_encoding; //optional
        }else if(item_type == 'uri '){
            string item_uri_type;
        }
    }
}

aligned(8) class ItemDataBox extends Box('idat'){
    bit(8) data[];
}

aligned(8) class ItemReferenceBox extends FullBox('iref', version, 0){
    if(version==0){
        SingleItemTypeReferenceBox      references[];
    }else if(version==1){
        SingleItemTypeReferenceBoxLarge references[];
    }
}

aligned(8) class SingleItemTypeReferenceBox(referenceType) extends Box(referenceType){
    unsigned int(16) from_item_ID;
    unsigned int(16) reference_count;
    for(j=0; j<reference_count; j++){
        unsigned int(16) to_item_ID;
    }
}

aligned(8) class SingleItemTypeReferenceBoxLarge(referenceType) extends Box(referenceType){
    unsigned int(32) from_item_ID;
    unsigned int(16) reference_count;
    for(j=0; j<reference_count; j++){
        unsigned int(32) to_item_ID;
    }
}

aligned(8) class HEVCConfigurationBox extends Box('hvcC'){
    HEVCDecoderConfigurationRecord() HEVCConfig;
}

aligned(8) class HEVCSampleEntry() extends VisualSampleEntry('hvc1' or 'hev1'){
    HEVCConfigurationBox config;
    MPEG4BitRateBox();              // optional
    MPEG4ExtensionDescriptorsBox(); // optional
    extra_boxes          boxes[];   // optional
}

aligned(8) class HEVCDecoderConfigurationRecord {
    unsigned int(8)  configurationVersion = 1;
    unsigned int(2)  general_profile_space;
    unsigned int(1)  general_tier_flag;
    unsigned int(5)  general_profile_idc;
    unsigned int(32) general_profile_compatibility_flags;
    unsigned int(48) general_constraint_indicator_flags;
    unsigned int(8)  general_level_idc;
    bit(4) reserved = ‘1111’b;
    unsigned int(12) min_spatial_segmentation_idc;
    bit(6) reserved = ‘111111’b;
    unsigned int(2)  parallelismType;
    bit(6) reserved = ‘111111’b;
    unsigned int(2)  chromaFormat;
    bit(5) reserved = ‘11111’b;
    unsigned int(3)  bitDepthLumaMinus8;
    bit(5) reserved = ‘11111’b;
    unsigned int(3)  bitDepthChromaMinus8;
    bit(16)          avgFrameRate;
    bit(2)           constantFrameRate;
    bit(3)           numTemporalLayers;
    bit(1)           temporalIdNested;
    unsigned int(2)  lengthSizeMinusOne;
    unsigned int(8)  numOfArrays;
    
    for(j=0; j<numOfArrays; j++){
        bit(1)           array_completeness;
        unsigned int(1)  reserved = 0;
        unsigned int(6)  NAL_unit_type;
        unsigned int(16) numNalus;
        for(i=0; i<numNalus; i++){
            unsigned int(16)     nalUnitLength;
            bit(8*nalUnitLength) nalUnit;//Includes 2-byte nal_unit_header().
        }
    }
}
#endif
