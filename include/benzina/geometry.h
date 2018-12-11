/* Include Guard */
#ifndef INCLUDE_BENZINA_GEOMETRY_H
#define INCLUDE_BENZINA_GEOMETRY_H

/**
 * Includes
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "benzina/bits.h"
#include "benzina/visibility.h"



/* Defines */



/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif



/* Data Structures Forward Declarations and Typedefs */
typedef struct BENZINA_GEOM_ENCODE     BENZINA_GEOM_ENCODE;



/**
 * @brief The BENZINA_GEOM_ENCODE struct
 * 
 * Used to compute the geometric transformation that places a source image onto
 * the encoded canvas in as lossless a manner as possible.
 */

struct BENZINA_GEOM_ENCODE{
    struct{
        uint32_t w, h;
        int      chroma_loc;
    } src;
    struct{
        uint32_t w, h;
        int      chroma_format;
    } canvas;
};



/**
 * Static Function Definitions
 */

BENZINA_STATIC inline int          benzina_geom_enc_init                         (BENZINA_GEOM_ENCODE* geom,
                                                                                  uint32_t             src_w,
                                                                                  uint32_t             src_h,
                                                                                  int                  src_chroma_loc,
                                                                                  uint32_t             canvas_w,
                                                                                  uint32_t             canvas_h,
                                                                                  int                  canvas_chroma_format){
    geom->src.w                = src_w;
    geom->src.h                = src_h;
    geom->src.chroma_loc       = src_chroma_loc;
    geom->canvas.w             = canvas_w;
    geom->canvas.h             = canvas_h;
    geom->canvas.chroma_format = canvas_chroma_format;
    return 0;
}
BENZINA_STATIC inline uint32_t     benzina_geom_enc_subsx                        (BENZINA_GEOM_ENCODE* geom){
    switch(geom->canvas.chroma_format){
        case BENZINA_CHROMAFMT_YUV420:
        case BENZINA_CHROMAFMT_YUV422:
            return 2;
        default:
            return 1;
    }
}
BENZINA_STATIC inline uint32_t     benzina_geom_enc_subsy                        (BENZINA_GEOM_ENCODE* geom){
    switch(geom->canvas.chroma_format){
        case BENZINA_CHROMAFMT_YUV420:
            return 2;
        default:
            return 1;
    }
}
BENZINA_STATIC inline int          benzina_geom_enc_wholly_contained_untransposed(BENZINA_GEOM_ENCODE* geom){
    return geom->src.w <= geom->canvas.w &&
           geom->src.h <= geom->canvas.h;
}
BENZINA_STATIC inline int          benzina_geom_enc_wholly_contained_transposed  (BENZINA_GEOM_ENCODE* geom){
    return geom->src.w <= geom->canvas.h &&
           geom->src.h <= geom->canvas.w;
}
BENZINA_STATIC inline int          benzina_geom_enc_wholly_contained             (BENZINA_GEOM_ENCODE* geom){
    return benzina_geom_enc_wholly_contained_untransposed(geom) &&
           benzina_geom_enc_wholly_contained_transposed  (geom);
}
BENZINA_STATIC inline int          benzina_geom_enc_landscape                    (BENZINA_GEOM_ENCODE* geom){
    return geom->src.w >= geom->src.h;
}
BENZINA_STATIC inline int          benzina_geom_enc_landscape_canvas             (BENZINA_GEOM_ENCODE* geom){
    return geom->canvas.w >= geom->canvas.h;
}
BENZINA_STATIC inline int          benzina_geom_enc_transpose                    (BENZINA_GEOM_ENCODE* geom){
    if      (benzina_geom_enc_wholly_contained_untransposed(geom)){
        return 0;
    }else if(benzina_geom_enc_wholly_contained_transposed(geom)){
        return 1;
    }else{
        return benzina_geom_enc_landscape       (geom) !=
               benzina_geom_enc_landscape_canvas(geom);
    }
}
BENZINA_STATIC inline uint32_t     benzina_geom_enc_transw                       (BENZINA_GEOM_ENCODE* geom){
    return benzina_geom_enc_transpose(geom) ? geom->src.h : geom->src.w;
}
BENZINA_STATIC inline uint32_t     benzina_geom_enc_transh                       (BENZINA_GEOM_ENCODE* geom){
    return benzina_geom_enc_transpose(geom) ? geom->src.w : geom->src.h;
}
BENZINA_STATIC inline int          benzina_geom_enc_aspect_too_wide              (BENZINA_GEOM_ENCODE* geom){
    uint64_t transw  = benzina_geom_enc_transw(geom);
    uint64_t transh  = benzina_geom_enc_transh(geom);
    uint64_t canvasw = geom->canvas.w;
    uint64_t canvash = geom->canvas.h;
    
    return transw*canvash > canvasw*transh;
}
BENZINA_STATIC inline uint32_t     benzina_geom_enc_cropw                        (BENZINA_GEOM_ENCODE* geom){
    uint64_t w;
    uint64_t transw = benzina_geom_enc_transw(geom);
    uint64_t transh = benzina_geom_enc_transh(geom);
    uint64_t subsx  = benzina_geom_enc_subsx (geom);
    
    if(benzina_geom_enc_wholly_contained(geom)){
        /**
         * If both dimensions fit, cropping is unnecessary because
         * transposition did its job.
         */
        
        w = transw;
    }else if(transh <= transw){
        /**
         * If only one dimension is in excess, cropping it will bring down the
         * frame size within the canvas bounds while maintaining a frame
         * aspect ratio w:h >= (landscape) or <= (portrait) the canvas itself,
         * which is what was desired.
         */
        
        w = geom->canvas.w;
    }else{
        /**
         * Both dimensions are in excess. Cropping is therefore used to match
         * the canvas aspect ratio, and a resize will bring down the frame
         * size to exactly the canvas size.
         */
        
        if(benzina_geom_enc_aspect_too_wide(geom)){
            w = geom->canvas.w*transh / geom->canvas.h;
        }else{
            w = transw;
        }
    }
    
    return (w / subsx) * subsx;
}
BENZINA_STATIC inline uint32_t     benzina_geom_enc_croph                        (BENZINA_GEOM_ENCODE* geom){
    uint64_t h;
    uint64_t transw = benzina_geom_enc_transw(geom);
    uint64_t transh = benzina_geom_enc_transh(geom);
    uint64_t subsy  = benzina_geom_enc_subsy (geom);
    
    if(benzina_geom_enc_wholly_contained(geom)){
        /**
         * If both dimensions fit, cropping is unnecessary because
         * transposition did its job.
         */
        
        h = transh;
    }else if(transh <= transw){
        /**
         * If only one dimension is in excess, cropping it will bring down the
         * frame size within the canvas bounds while maintaining a frame
         * aspect ratio w:h >= (landscape) or <= (portrait) the canvas itself,
         * which is what was desired.
         */
        
        h = geom->canvas.h;
    }else{
        /**
         * Both dimensions are in excess. Cropping is therefore used to match
         * the canvas aspect ratio, and a resize will bring down the frame
         * size to exactly the canvas size.
         */
        
        if(benzina_geom_enc_aspect_too_wide(geom)){
            h = transh;
        }else{
            h = geom->canvas.h*transw / geom->canvas.w;
        }
    }
    
    return (h / subsy) * subsy;
}
BENZINA_STATIC inline uint32_t     benzina_geom_enc_cropx                        (BENZINA_GEOM_ENCODE* geom){
    uint64_t transw = benzina_geom_enc_transw(geom);
    uint64_t cropw  = benzina_geom_enc_cropw (geom);
    uint64_t subsx  = benzina_geom_enc_subsx (geom);
    
    /* Horizontal offset in pixels into the transposed image. */
    return ((transw-cropw) / (2*subsx)) * subsx;
}
BENZINA_STATIC inline uint32_t     benzina_geom_enc_cropy                        (BENZINA_GEOM_ENCODE* geom){
    uint64_t transh = benzina_geom_enc_transh(geom);
    uint64_t croph  = benzina_geom_enc_croph (geom);
    uint64_t subsy  = benzina_geom_enc_subsy (geom);
    
    /* Vertical offset in pixels into the transposed image. */
    return ((transh-croph) / (2*subsy)) * subsy;
}
BENZINA_STATIC inline int          benzina_geom_enc_vflip                        (BENZINA_GEOM_ENCODE* geom){
    (void)geom;
    return 0;
}
BENZINA_STATIC inline int          benzina_geom_enc_hflip                        (BENZINA_GEOM_ENCODE* geom){
    /**
     * We require a horizontal flip for transpositions of images with chromaLocs
     * bottom left (4) and bottom center (5) because the transposition will
     * bring the chroma samples to the right, and there exist no codes to
     * indicate top right and right center.
     * 
     * On the other hand, if, after the transposition, a horizontal flip is
     * performed, the top right and right center sample points will become
     * top left (2) and left center (0), both of them being legal values.
     */
    
    switch(geom->src.chroma_loc){
        case BENZINA_CHROMALOC_BOTTOM:
        case BENZINA_CHROMALOC_BOTTOMLEFT:
            return benzina_geom_enc_transpose(geom);
        default:
            return 0;
    }
}
BENZINA_STATIC inline int          benzina_geom_enc_canvas_chromaloc             (BENZINA_GEOM_ENCODE* geom){
    if(!benzina_geom_enc_transpose(geom)){
        return geom->src.chroma_loc;
    }
    
    switch(geom->src.chroma_loc){
        case BENZINA_CHROMALOC_LEFT:       return BENZINA_CHROMALOC_TOP;    /* Transposed */
        case BENZINA_CHROMALOC_CENTER:     return BENZINA_CHROMALOC_CENTER; /* Transposed */
        case BENZINA_CHROMALOC_TOPLEFT:    return BENZINA_CHROMALOC_TOPLEFT;/* Transposed */
        case BENZINA_CHROMALOC_TOP:        return BENZINA_CHROMALOC_LEFT;   /* Transposed */
        case BENZINA_CHROMALOC_BOTTOMLEFT: return BENZINA_CHROMALOC_TOPLEFT;/* Transposed, H-Flipped */
        case BENZINA_CHROMALOC_BOTTOM:     return BENZINA_CHROMALOC_LEFT;   /* Transposed, H-Flipped */
        default: return -1;
    }
}
BENZINA_STATIC inline int          benzina_geom_enc_resize                       (BENZINA_GEOM_ENCODE* geom){
    return benzina_geom_enc_cropw(geom) > geom->canvas.w ||
           benzina_geom_enc_croph(geom) > geom->canvas.h;
}



/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

