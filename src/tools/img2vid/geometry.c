/* Includes */
#include "geometry.h"



/* Static Function Definitions */
static inline void swapu32(uint32_t* a, uint32_t* b){
    uint32_t t = *a;
    *a = *b;
    *b = t;
}
static inline int chromasubx(int chroma_format){
    switch(chroma_format){
        case BENZINA_CHROMAFMT_YUV420:
        case BENZINA_CHROMAFMT_YUV422:
            return 2;
        default:
            return 1;
    }
}
static inline int chromasuby(int chroma_format){
    switch(chroma_format){
        case BENZINA_CHROMAFMT_YUV420:
            return 2;
        default:
            return 1;
    }
}


/* Public Function Definitions */

/**
 * @brief Solve the geometry problem for a given frame.
 * 
 * The geometry problem is the question of how precisely to transform any image
 * so that it fits within a specified canvas while minimizing information loss.
 * This is a non-trivial topic given the complex series of transformations and
 * resamplings that may be necessary to achieve this.
 * 
 * @param [in,out] geom  The struct with its input members set to valid values.
 *                       The output members will be set on successful return.
 * @return 0 if inputs valid; !0 if error encountered.
 */

extern int benzina_geom_solve(BENZINA_GEOM* geom){
    uint32_t subsx = chromasubx(geom->in.src.chroma_format);
    uint32_t subsy = chromasuby(geom->in.src.chroma_format);
    uint32_t subcx = chromasubx(geom->in.canvas.chroma_format);
    uint32_t subcy = chromasuby(geom->in.canvas.chroma_format);
    
    if(geom->in.src.w <= 0){return 1;}
    if(geom->in.src.h <= 0){return 1;}
    if(geom->in.src.w % subsx){return 1;}
    if(geom->in.src.h % subsy){return 1;}
    
    /**
     * 1. Transposition
     * 
     * Determine whether transposition is required. Transposition may cause
     * images to fit that otherwise wouldn't, or can preserve more of the image
     * than otherwise would have, if the image in question was portrait and the
     * canvas landscape or vice-versa.
     */
    
    geom->wholly_contained_untransposed = geom->in.src.w <= geom->in.canvas.w &&
                                          geom->in.src.h <= geom->in.canvas.h;
    geom->wholly_contained_transposed   = geom->in.src.w <= geom->in.canvas.h &&
                                          geom->in.src.h <= geom->in.canvas.w;
    geom->wholly_contained              = geom->wholly_contained_untransposed ||
                                          geom->wholly_contained_transposed;
    geom->landscape_src                 = geom->in.src.w    >= geom->in.src.h;
    geom->landscape_canvas              = geom->in.canvas.w >= geom->in.canvas.h;
    if      (geom->wholly_contained_untransposed){
        geom->transpose = 0;
    }else if(geom->wholly_contained_transposed){
        geom->transpose = 1;
    }else{
        geom->transpose = geom->landscape_src != geom->landscape_canvas;
    }
    geom->transw = geom->transpose ? geom->in.src.h : geom->in.src.w;
    geom->transh = geom->transpose ? geom->in.src.w : geom->in.src.h;
    
    /**
     * 2. V/H-Flip
     * 
     * Determine whether a vertical/horizontal flip is required to make
     * the chroma location valid again.
     */
    
    geom->vflip = 0;
    geom->hflip = 0;
    if(geom->in.src.chroma_format == BENZINA_CHROMAFMT_YUV420){
        switch(geom->in.src.chroma_loc){
            case BENZINA_CHROMALOC_BOTTOM:
            case BENZINA_CHROMALOC_BOTTOMLEFT:
                geom->hflip = geom->transpose;
            break;
            default: break;
        }
    }
    
    /**
     * 3. Chroma Location
     * 
     * Based on the transpose and flip configurations, determine the new
     * effective chroma location.
     * 
     * An invalid selection of h-flip is silently ignored.
     */
    
    geom->chroma_loc_canvas = BENZINA_CHROMALOC_TOPLEFT;
    if(geom->in.src.chroma_format == BENZINA_CHROMAFMT_YUV420){
        switch(geom->in.src.chroma_loc){
            case BENZINA_CHROMALOC_LEFT:
                if(geom->transpose){
                    geom->chroma_loc_canvas = geom->vflip ? BENZINA_CHROMALOC_BOTTOM :
                                                            BENZINA_CHROMALOC_TOP;
                }else{
                    geom->chroma_loc_canvas = BENZINA_CHROMALOC_LEFT;
                }
            break;
            case BENZINA_CHROMALOC_CENTER:
                geom->chroma_loc_canvas = BENZINA_CHROMALOC_CENTER;
            break;
            case BENZINA_CHROMALOC_TOPLEFT:
                geom->chroma_loc_canvas = geom->vflip ? BENZINA_CHROMALOC_BOTTOMLEFT :
                                                        BENZINA_CHROMALOC_TOPLEFT;
            break;
            case BENZINA_CHROMALOC_TOP:
                if(geom->transpose){
                    geom->chroma_loc_canvas = BENZINA_CHROMALOC_LEFT;
                }else{
                    geom->chroma_loc_canvas = geom->vflip ? BENZINA_CHROMALOC_BOTTOM :
                                                            BENZINA_CHROMALOC_TOP;
                }
            break;
            case BENZINA_CHROMALOC_BOTTOM:
                if(geom->transpose){
                    geom->chroma_loc_canvas = BENZINA_CHROMALOC_LEFT;
                }else{
                    geom->chroma_loc_canvas = geom->vflip ? BENZINA_CHROMALOC_TOP   :
                                                            BENZINA_CHROMALOC_BOTTOM;
                }
            case BENZINA_CHROMALOC_BOTTOMLEFT:
                if(geom->transpose){
                    geom->chroma_loc_canvas = geom->vflip ? BENZINA_CHROMALOC_BOTTOMLEFT :
                                                            BENZINA_CHROMALOC_TOPLEFT;
                }else{
                    geom->chroma_loc_canvas = geom->vflip ? BENZINA_CHROMALOC_TOPLEFT   :
                                                            BENZINA_CHROMALOC_BOTTOMLEFT;
                }
            break;
            default: return 1;
        }
    }
    
    /**
     * 4. Transposed Source Crop Window Size
     * 
     * The source image may or may not fit into the canvas after the
     * (optional) transposition.
     * 
     * - If it does fit with/without transposition, no cropping is necessary.
     * - If it does not fit, but just one dimension is to blame, and that
     *   dimension is the longer one, crop that dimension to canvas size.
     * - Otherwise, a full-blown crop & resize is required, for which we need
     *   a crop window within the source image with an AR that matches that
     *   of the canvas. Compute canvas-AR-corrected source width/height, then
     *   crop source to the lesser of corrected and original width/height.
     * 
     * Also crop a little extra to round down the window size to an integer
     * number of subsampling units.
     */
    
    if(geom->wholly_contained){
        geom->arcorw = geom->transw;
        geom->arcorh = geom->transh;
    }else if(geom->transw <= geom->in.canvas.w &&
             !geom->landscape_canvas){
        geom->arcorw = geom->transw;
        geom->arcorh = geom->in.canvas.h;
    }else if(geom->transh <= geom->in.canvas.h &&
             geom->landscape_canvas){
        geom->arcorw = geom->in.canvas.w;
        geom->arcorh = geom->transh;
    }else{
        geom->arcorw = (uint64_t)geom->transh*geom->in.canvas.w/geom->in.canvas.h;
        geom->arcorh = (uint64_t)geom->transw*geom->in.canvas.h/geom->in.canvas.w;
    }
    geom->cropw  = geom->arcorw < geom->transw ? geom->arcorw : geom->transw;
    geom->croph  = geom->arcorh < geom->transh ? geom->arcorh : geom->transh;
    geom->cropw -= geom->cropw % (geom->transpose ? subsy : subsx);
    geom->croph -= geom->croph % (geom->transpose ? subsx : subsy);
    geom->cropw -= geom->cropw % subcx;
    geom->croph -= geom->croph % subcy;
    
    /**
     * 5. Transposed Source Crop Window Offset
     * 
     * The transposed source's crop window size having been computed, we
     * center it within the transposed source.
     */
    
    geom->cropx  = (geom->transw-geom->cropw)/2;
    geom->cropy  = (geom->transh-geom->croph)/2;
    geom->cropx -= geom->cropx % (geom->transpose ? subsy : subsx);
    geom->cropy -= geom->cropy % (geom->transpose ? subsx : subsy);
    
    /**
     * 6. Embedded Image Size
     * 
     * Determine the embedded window size within the canvas. This involves
     * scaling the crow window (cropw, croph) down to fit entirely within
     * the canvas (geom->in.canvas.w, geom->in.canvas.h) if necessary.
     */
    
    if(geom->cropw <= geom->in.canvas.w &&
       geom->croph <= geom->in.canvas.h){
        geom->embedw = geom->cropw;
        geom->embedh = geom->croph;
    }else if((uint64_t)geom->croph*geom->in.canvas.w/geom->cropw > geom->in.canvas.h){
        geom->embedw = (uint64_t)geom->cropw*geom->in.canvas.h/geom->croph;
        geom->embedh = geom->in.canvas.h;
    }else{
        geom->embedw = geom->in.canvas.w;
        geom->embedh = (uint64_t)geom->croph*geom->in.canvas.w/geom->cropw;
    }
    geom->embedw -= geom->embedw % subcx;
    geom->embedh -= geom->embedh % subcy;
    
    /**
     * 7. Resize
     * 
     * Determine from the cropped and embedded image size above whether a
     * full resize is needed.
     */
    
    geom->resize = geom->embedw != geom->cropw || geom->embedh != geom->croph;
    
    return 0;
}

