/* Includes */
#include "geometry.h"



/* Static Function Definitions */
static inline uint32_t minu32 (uint32_t a, uint32_t b){return a<b?a:b;}
static inline uint32_t maxu32 (uint32_t a, uint32_t b){return a>b?a:b;}
static inline void     swapu32(uint32_t* a, uint32_t* b){
    uint32_t t = *a;
    *a = *b;
    *b = t;
}
static inline int      chromafmtsubx(int chroma_fmt){
    switch(chroma_fmt){
        case BENZINA_CHROMAFMT_YUV420:
        case BENZINA_CHROMAFMT_YUV422:
            return 2;
        default:
            return 1;
    }
}
static inline int      chromafmtsuby(int chroma_fmt){
    switch(chroma_fmt){
        case BENZINA_CHROMAFMT_YUV420:
            return 2;
        default:
            return 1;
    }
}
static inline int      chromaloctransp(int chroma_loc){
    switch(chroma_loc){
        case BENZINA_CHROMALOC_TOPLEFT:       return BENZINA_CHROMALOC_TOPLEFT;
        case BENZINA_CHROMALOC_TOP:           return BENZINA_CHROMALOC_LEFT;
        case BENZINA_CHROMALOC_TOPRIGHT:      return BENZINA_CHROMALOC_BOTTOMLEFT;
        case BENZINA_CHROMALOC_LEFT:          return BENZINA_CHROMALOC_TOP;
        case BENZINA_CHROMALOC_CENTER:        return BENZINA_CHROMALOC_CENTER;
        case BENZINA_CHROMALOC_RIGHT:         return BENZINA_CHROMALOC_BOTTOM;
        case BENZINA_CHROMALOC_BOTTOMLEFT:    return BENZINA_CHROMALOC_TOPRIGHT;
        case BENZINA_CHROMALOC_BOTTOM:        return BENZINA_CHROMALOC_RIGHT;
        case BENZINA_CHROMALOC_BOTTOMRIGHT:   return BENZINA_CHROMALOC_BOTTOMRIGHT;
        default:                              return chroma_loc;
    }
}

/* Public Function Definitions */

/**
 * @brief Retrieve the x coordinate, y coordinate, width and height of a
 *        rectangle.
 * @param [in]  r  The rectangle being queried.
 * @return The queried property.
 */

extern uint32_t rect2d_x(const BENZINA_RECT2D* r){
    return minu32(minu32(r->p[0].x, r->p[1].x),
                  minu32(r->p[2].x, r->p[3].x));
}
extern uint32_t rect2d_y(const BENZINA_RECT2D* r){
    return minu32(minu32(r->p[0].y, r->p[1].y),
                  minu32(r->p[2].y, r->p[3].y));
}
extern uint32_t rect2d_w(const BENZINA_RECT2D* r){
    return maxu32(maxu32(r->p[0].x, r->p[1].x),
                  maxu32(r->p[2].x, r->p[3].x)) - rect2d_x(r) + 1;
}
extern uint32_t rect2d_h(const BENZINA_RECT2D* r){
    return maxu32(maxu32(r->p[0].y, r->p[1].y),
                  maxu32(r->p[2].y, r->p[3].y)) - rect2d_y(r) + 1;
}

/**
 * @brief Scale a rectangle r by a rational a/b in the x and y dimensions.
 * @param [in]  r  The rectangle being scaled.
 * @param [in]  a  The numerator of the rational.
 * @param [in]  b  The denominator of the rational.
 */

extern void     rect2d_scalex(BENZINA_RECT2D* r, uint32_t a, uint32_t b){
    int i;
    uint32_t xmin = rect2d_x(r);
    for(i=0;i<4;i++){
        if(r->p[i].x == xmin){
            r->p[i].x = ((uint64_t)r->p[i].x  ) * a / b;
        }else{
            r->p[i].x = ((uint64_t)r->p[i].x+1) * a / b - 1;
        }
    }
}
extern void     rect2d_scaley(BENZINA_RECT2D* r, uint32_t a, uint32_t b){
    int i;
    uint32_t ymin = rect2d_y(r);
    for(i=0;i<4;i++){
        if(r->p[i].y == ymin){
            r->p[i].y = ((uint64_t)r->p[i].y  ) * a / b;
        }else{
            r->p[i].y = ((uint64_t)r->p[i].y+1) * a / b - 1;
        }
    }
}

/**
 * @brief Transpose, horizontally flip and vertically flip a rectangle r.
 * @param [in]  r  The rectangle being transformed.
 */

extern void     rect2d_transp(BENZINA_RECT2D* r){
    int i;
    for(i=0;i<4;i++){
        swapu32(&r->p[i].x, &r->p[i].y);
    }
}
extern void     rect2d_hflip (BENZINA_RECT2D* r){
    uint32_t xmin=rect2d_x(r), w=rect2d_w(r);
    uint32_t xmax=xmin+w-1;
    int i;
    for(i=0;i<4;i++){
        r->p[i].x = r->p[i].x == xmin ? xmax : xmin;
    }
}
extern void     rect2d_vflip (BENZINA_RECT2D* r){
    uint32_t ymin=rect2d_y(r), h=rect2d_h(r);
    uint32_t ymax=ymin+h-1;
    int i;
    for(i=0;i<4;i++){
        r->p[i].y = r->p[i].y == ymin ? ymax : ymin;
    }
}

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
    uint32_t subsx = chromafmtsubx(geom->i.source.chroma_fmt);
    uint32_t subsy = chromafmtsuby(geom->i.source.chroma_fmt);
    uint32_t subcx = chromafmtsubx(geom->i.canvas.chroma_fmt);
    uint32_t subcy = chromafmtsuby(geom->i.canvas.chroma_fmt);
    
    if(geom->i.canvas.w <= 0   ){return 1;}
    if(geom->i.canvas.h <= 0   ){return 1;}
    if(geom->i.canvas.w % subcx){return 1;}
    if(geom->i.canvas.h % subcy){return 1;}
    if(geom->i.source.w <= 0   ){return 1;}
    if(geom->i.source.h <= 0   ){return 1;}
    if(geom->i.source.w % subsx){return 1;}
    if(geom->i.source.h % subsy){return 1;}
    
    /**
     * 1. Init
     * 
     * We start with the identity transformation: Source and canvas
     * coordinate systems are the same, and match the *source* image.
     * 
     * Point 0: TL (0,0)
     * Point 1: TR (w-1,0)
     * Point 2: BR (w-1,h-1)
     * Point 3: BL (0,h-1)
     */
    
    geom->o.transpose     = geom->o.vflip = geom->o.hflip = 0;
    geom->o.chroma_loc    = geom->i.source.chroma_loc;
    geom->o.canvas.p[0].x = geom->o.source.p[0].x = 0;
    geom->o.canvas.p[0].y = geom->o.source.p[0].y = 0;
    geom->o.canvas.p[1].x = geom->o.source.p[1].x = geom->i.source.w-1;
    geom->o.canvas.p[1].y = geom->o.source.p[1].y = 0;
    geom->o.canvas.p[2].x = geom->o.source.p[2].x = geom->i.source.w-1;
    geom->o.canvas.p[2].y = geom->o.source.p[2].y = geom->i.source.h-1;
    geom->o.canvas.p[3].x = geom->o.source.p[3].x = 0;
    geom->o.canvas.p[3].y = geom->o.source.p[3].y = geom->i.source.h-1;
    
    /**
     * 2. Super Boost
     * 
     * In yuv420super mode, the canvas embedding is attempted at 2x the true
     * resolution to avoid chroma downsampling losses, but otherwise the
     * fitting logic is entirely the same, and this boost may, in fact, have
     * no effect in the end if scale-to-fit cancels it out.
     */
    
    if(geom->i.canvas.superscale){
        rect2d_scalex(&geom->o.canvas, subcx, 1);
        rect2d_scaley(&geom->o.canvas, subcy, 1);
    }
    
    /**
     * 3. Transposition
     * 
     * Determine whether transposition is required. Transposition may cause
     * images to fit that otherwise wouldn't, or can preserve more of the image
     * than otherwise would have, if the image in question was portrait and the
     * canvas landscape or vice-versa.
     */
    
    int wholly_contained_untransposed = rect2d_w(&geom->o.canvas) <= geom->i.canvas.w &&
                                        rect2d_h(&geom->o.canvas) <= geom->i.canvas.h;
    int wholly_contained_transposed   = rect2d_w(&geom->o.canvas) <= geom->i.canvas.h &&
                                        rect2d_h(&geom->o.canvas) <= geom->i.canvas.w;
    int landscape_source              = geom->i.source.w >= geom->i.source.h;
    int landscape_canvas              = geom->i.canvas.w >= geom->i.canvas.h;
    int portrait_canvas               = !landscape_canvas;
    geom->o.transpose                 = wholly_contained_untransposed ? 0 :
                                        wholly_contained_transposed   ? 1 :
                                        landscape_source != landscape_canvas;
    if(geom->o.transpose){
        rect2d_transp(&geom->o.canvas);
        geom->o.chroma_loc = chromaloctransp(geom->o.chroma_loc);
    }
    
    /**
     * 4. V/H-Flip
     * 
     * Determine whether a vertical/horizontal flip is required to make
     * the chroma location valid again.
     */
    
    if(geom->i.canvas.chroma_fmt == BENZINA_CHROMAFMT_YUV420){
        switch(geom->o.chroma_loc){
            default: break;
            case BENZINA_CHROMALOC_TOPRIGHT:    geom->o.chroma_loc = BENZINA_CHROMALOC_TOPLEFT;    geom->o.hflip = 1; goto hflip;
            case BENZINA_CHROMALOC_RIGHT:       geom->o.chroma_loc = BENZINA_CHROMALOC_LEFT;       geom->o.hflip = 1; goto hflip;
            case BENZINA_CHROMALOC_BOTTOMRIGHT: geom->o.chroma_loc = BENZINA_CHROMALOC_BOTTOMLEFT; geom->o.hflip = 1; goto hflip;
            hflip:
                rect2d_hflip(&geom->o.canvas);
            break;
        }
    }
    
    /**
     * 5. Source Crop
     * 
     * The source image may or may not fit into the canvas after the
     * (optional) transposition.
     * 
     * - If it does fit with/without transposition, no cropping is necessary.
     *    * A resize is not required.
     * - If it does not fit, but just one dimension is to blame, *and* the
     *   aspect ratio is more extreme than that of the canvas, crop the
     *   width (for a landscape canvas) or height (for a portrait canvas)
     *   to the canvas' width and height. The aspect ratio will still exceed
     *   that of the canvas, but will be less extreme, and the cropped frame
     *   fits entirely in the canvas.
     *    * A resize is not required.
     * - If it does not fit, and both dimensions are to blame, *and* the
     *   aspect ratio is more extreme than that of the canvas, crop the
     *   width (for a landscape canvas) or height (for a portrait canvas)
     *   to match the aspect ratio of the canvas.
     *    * A resize is required.
     * - Otherwise, it does not fit, at least one dimension is to blame,
     *   and the aspect ratio is equal to or less extreme than the canvas.
     *   No cropping is done, as we intend to put the entire frame within
     *   the canvas, but we'll need to scale down the destination port.
     *    * A resize is required.
     * 
     * Also crop a little extra to round down the window size to an integer
     * number of subsampling units.
     */
    
#if 0
    
#endif
    
    /**
     * 6. Resize
     * 
     * The source image, even after cropping, may not fit within the canvas.
     */
    
#if 0
    
#endif
    
    return 0;
}
