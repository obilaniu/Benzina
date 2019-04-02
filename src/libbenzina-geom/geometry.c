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
static inline int      chromalochflip (int chroma_loc){
    switch(chroma_loc){
        case BENZINA_CHROMALOC_TOPLEFT:       return BENZINA_CHROMALOC_TOPRIGHT;
        case BENZINA_CHROMALOC_LEFT:          return BENZINA_CHROMALOC_RIGHT;
        case BENZINA_CHROMALOC_BOTTOMLEFT:    return BENZINA_CHROMALOC_BOTTOMRIGHT;
        case BENZINA_CHROMALOC_TOPRIGHT:      return BENZINA_CHROMALOC_TOPLEFT;
        case BENZINA_CHROMALOC_RIGHT:         return BENZINA_CHROMALOC_LEFT;
        case BENZINA_CHROMALOC_BOTTOMRIGHT:   return BENZINA_CHROMALOC_BOTTOMLEFT;
        default:                              return chroma_loc;
    }
}
static inline int      chromalocvflip (int chroma_loc){
    switch(chroma_loc){
        case BENZINA_CHROMALOC_TOPLEFT:       return BENZINA_CHROMALOC_BOTTOMLEFT;
        case BENZINA_CHROMALOC_TOP:           return BENZINA_CHROMALOC_BOTTOM;
        case BENZINA_CHROMALOC_TOPRIGHT:      return BENZINA_CHROMALOC_BOTTOMRIGHT;
        case BENZINA_CHROMALOC_BOTTOMLEFT:    return BENZINA_CHROMALOC_TOPLEFT;
        case BENZINA_CHROMALOC_BOTTOM:        return BENZINA_CHROMALOC_TOP;
        case BENZINA_CHROMALOC_BOTTOMRIGHT:   return BENZINA_CHROMALOC_TOPRIGHT;
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
 * @brief Scale a rectangle r to width w or height h.
 * @param [in]  r      The rectangle being cropped.
 * @param [in]  mw,mh  The maximum width or height to crop the rectangle to.
 * @return The new width or height. This will be no greater than w or h.
 */

extern void     rect2d_cropw (BENZINA_RECT2D* r, uint32_t mw){
    uint32_t xmin = rect2d_x(r), w = rect2d_w(r);
    if(w <= mw){return;}
    
    int i;
    for(i=0;i<4;i++){
        r->p[i].x = xmin + minu32(r->p[i].x-xmin, mw-1);
    }
}
extern void     rect2d_croph (BENZINA_RECT2D* r, uint32_t mh){
    uint32_t ymin = rect2d_y(r), h = rect2d_h(r);
    if(h <= mh){return;}
    
    int i;
    for(i=0;i<4;i++){
        r->p[i].y = ymin + minu32(r->p[i].y-ymin, mh-1);
    }
}

/**
 * @brief Translate a rectangle r by (x,y) pixels.
 * @param [in]  r  The rectangle being translated
 * @param [in]  x  The number of pixels in the x dimension to translate by
 * @param [in]  y  The number of pixels in the y dimension to translate by
 */

extern void     rect2d_transl(BENZINA_RECT2D* r, uint32_t x, uint32_t y){
    int i;
    for(i=0;i<4;i++){
        r->p[i].x += x;
        r->p[i].y += y;
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
    
    if(geom->i.canvas.w < subcx){return 1;}
    if(geom->i.canvas.h < subcy){return 1;}
    if(geom->i.canvas.w % subcx){return 1;}
    if(geom->i.canvas.h % subcy){return 1;}
    if(geom->i.source.w < subsx){return 1;}
    if(geom->i.source.h < subsy){return 1;}
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
    
    const int wholly_contained_untransposed = rect2d_w(&geom->o.canvas) <= geom->i.canvas.w &&
                                              rect2d_h(&geom->o.canvas) <= geom->i.canvas.h;
    const int wholly_contained_transposed   = rect2d_w(&geom->o.canvas) <= geom->i.canvas.h &&
                                              rect2d_h(&geom->o.canvas) <= geom->i.canvas.w;
    const int landscape_source              = geom->i.source.w >= geom->i.source.h;
    const int landscape_canvas              = geom->i.canvas.w >= geom->i.canvas.h;
    const int portrait_canvas               = !landscape_canvas;
    geom->o.transpose = wholly_contained_untransposed ? 0 :
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
            case BENZINA_CHROMALOC_TOPRIGHT:
            case BENZINA_CHROMALOC_RIGHT:
            case BENZINA_CHROMALOC_BOTTOMRIGHT:
                geom->o.chroma_loc = chromalochflip(geom->o.chroma_loc);
                geom->o.hflip      = 1;
                rect2d_hflip(&geom->o.canvas);
            break;
        }
    }
    
    /**
     * 5. Source Crop
     * 6. Resize
     * 
     * The source image may or may not fit into the canvas after the
     * (optional) transposition.
     * 
     * - If it does fit with/without transposition, no cropping is necessary.
     *    * A resize is not required.
     * - If it does not fit, but just one dimension is to blame, *and* the
     *   aspect ratio is more extreme than that of the canvas, crop the
     *   width (for a landscape canvas) or height (for a portrait canvas)
     *   to EITHER canvas size OR canvas AR, whichever results in the larger
     *   image. The aspect ratio may still exceed that of the canvas, but will
     *   be less extreme.
     *    * A resize is not required the crop was to canvas size, but is
     *      required if the crop was to aspect ratio.
     * - Otherwise, it does not fit, at least one dimension is to blame,
     *   and the aspect ratio is equal to or less extreme than the canvas.
     *   No cropping is done, as we intend to put the entire frame within
     *   the canvas, but we'll need to scale down the destination port.
     *    * A resize is required.
     */
    
    uint64_t ocw = rect2d_w(&geom->o.canvas), osw = rect2d_w(&geom->o.source);
    uint64_t och = rect2d_h(&geom->o.canvas), osh = rect2d_h(&geom->o.source);
    uint64_t mcw = geom->i.canvas.w;
    uint64_t mch = geom->i.canvas.h;
    int      extreme_wide = ocw*mch > mcw*och;
    int      extreme_tall = och*mcw > mch*ocw;
    if      (ocw <= mcw && och <= mch){
    }else if(extreme_wide && landscape_canvas){
        if(och <= mch){
            /* Crop to canvas width. Embedding AR > Canvas AR. No Resize. */
            if(geom->o.transpose){
                /* Want: nsh/nsw = nsh/osw = mcw/och  ->  nsh = osw*mcw/och */
                rect2d_croph(&geom->o.source, mcw*osw/och);
                rect2d_cropw(&geom->o.canvas, mcw);
            }else{
                /* Want: nsw/nsh = nsw/osh = mcw/och  ->  nsw = osh*mcw/och */
                rect2d_cropw(&geom->o.source, mcw*osh/och);
                rect2d_cropw(&geom->o.canvas, mcw);
            }
        }else{
            /* Crop to canvas AR. Embedding AR = Canvas AR. Resize. */
            if(geom->o.transpose){
                /* Want: nsh/nsw = nsh/osw = mcw/mch  ->  nsh = osw*mcw/mch */
                rect2d_croph(&geom->o.source, mcw*osw/mch);
                rect2d_cropw(&geom->o.canvas, mcw);
                rect2d_croph(&geom->o.canvas, mch);
            }else{
                /* Want: nsw/nsh = nsw/osh = mcw/mch  ->  nsw = osh*mcw/mch */
                rect2d_cropw(&geom->o.source, mcw*osh/mch);
                rect2d_cropw(&geom->o.canvas, mcw);
                rect2d_croph(&geom->o.canvas, mch);
            }
        }
    }else if(extreme_tall && portrait_canvas){
        if(ocw <= mcw){
            /* Crop to canvas height. Embedding AR > Canvas AR. No Resize. */
            if(geom->o.transpose){
                /* Want: nsw/nsh = nsw/osh = mch/ocw  ->  nsw = osh*mch/ocw */
                rect2d_cropw(&geom->o.source, mch*osh/ocw);
                rect2d_croph(&geom->o.canvas, mch);
            }else{
                /* Want: nsh/nsw = nsh/osw = mch/ocw  ->  nsh = osw*mch/ocw */
                rect2d_croph(&geom->o.source, mch*osw/ocw);
                rect2d_croph(&geom->o.canvas, mch);
            }
        }else{
            /* Crop to canvas AR. Embedding AR = Canvas AR. Resize. */
            if(geom->o.transpose){
                /* Want: nsw/nsh = nsw/osh = mch/mcw  ->  nsw = osh*mch/mcw */
                rect2d_cropw(&geom->o.source, mch*osh/mcw);
                rect2d_croph(&geom->o.canvas, mch);
                rect2d_cropw(&geom->o.canvas, mcw);
            }else{
                /* Want: nsh/nsw = nsh/osw = mch/mcw  ->  nsh = osw*mch/mcw */
                rect2d_croph(&geom->o.source, mch*osw/mcw);
                rect2d_croph(&geom->o.canvas, mch);
                rect2d_cropw(&geom->o.canvas, mcw);
            }
        }
    }else if(landscape_canvas){
        /* Landscape image does not have extreme AR, but does not fit in landscape canvas. */
        rect2d_cropw(&geom->o.canvas, geom->o.transpose ? mch*osh/osw : mch*osw/osh);
        rect2d_croph(&geom->o.canvas, mch);
    }else{
        /* Portrait  image does not have extreme AR, but does not fit in portrait  canvas. */
        rect2d_croph(&geom->o.canvas, geom->o.transpose ? mcw*osw/osh : mcw*osh/osw);
        rect2d_cropw(&geom->o.canvas, mcw);
    }
    
    /**
     * 7. Translate
     * 
     * Once the sizes of the embedding into the canvas and the viewport into
     * the source image are determined, we
     * 
     *     1) Position the embedding at the top left of the canvas and
     *     2) Center the viewport within the source.
     * 
     * Alignment constraints due to chroma subsampling factors may need to be
     * taken into account.
     */
    
    uint32_t tsx = (geom->i.source.w - rect2d_w(&geom->o.source)) / (2*subsx) * subsx;
    uint32_t tsy = (geom->i.source.h - rect2d_h(&geom->o.source)) / (2*subsy) * subsy;
    rect2d_transl(&geom->o.source, tsx-rect2d_x(&geom->o.source), tsy-rect2d_y(&geom->o.source));
    rect2d_transl(&geom->o.canvas,    -rect2d_x(&geom->o.canvas),    -rect2d_y(&geom->o.canvas));
    
    return 0;
}
