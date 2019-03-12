/* Includes */
#include <stdio.h>
#include "geometry.h"


/**
 * Defines
 */




/**
 * Functions.
 */

/**
 * Utility Functions
 */

uint32_t minu32(uint32_t a, uint32_t b){return a<b?a:b;}
uint32_t maxu32(uint32_t a, uint32_t b){return a<b?a:b;}
uint32_t rect2dx(const BENZINA_RECT2D* r){
    return minu32(r->p[0].x, r->p[2].x);
}
uint32_t rect2dy(const BENZINA_RECT2D* r){
    return minu32(r->p[0].y, r->p[2].y);
}
uint32_t rect2dw(const BENZINA_RECT2D* r){
    return maxu32(r->p[0].x, r->p[2].x) - maxu32(r->p[0].x, r->p[2].x) + 1;
}
uint32_t rect2dh(const BENZINA_RECT2D* r){
    return maxu32(r->p[0].y, r->p[2].y) - maxu32(r->p[0].y, r->p[2].y) + 1;
}

/**
 * @brief Validate the geometry problem's solution.
 * @param [in]   geom  The geometry problem input and its solution
 * @return 0 if valid;
 *         1 if invalid chroma format;
 *         2 if oversize;
 *         3 if not rectangular;
 *         4 if misaligned.
 */

int validate(const BENZINA_GEOM* geom){
    uint32_t cx,cy,cw,ch,sx,sy,sw,sh;
    
    /* Chroma Location Check */
    switch(geom->out.chroma_loc){
        case BENZINA_CHROMALOC_TOPRIGHT:
        case BENZINA_CHROMALOC_RIGHT:
        case BENZINA_CHROMALOC_BOTTOMRIGHT:
            return 1;
    }
    
    /* Oversize Check */
    cx = rect2dx(&geom->out.canvas);
    cy = rect2dy(&geom->out.canvas);
    cw = rect2dw(&geom->out.canvas);
    ch = rect2dh(&geom->out.canvas);
    
    sx = rect2dx(&geom->out.source);
    sy = rect2dy(&geom->out.source);
    sw = rect2dw(&geom->out.source);
    sh = rect2dh(&geom->out.source);
    
    if(cx >= geom->in.canvas.w || cw > geom->in.canvas.w || cx+cw > geom->in.canvas.w ||
       cy >= geom->in.canvas.h || ch > geom->in.canvas.h || cy+ch > geom->in.canvas.h ||
       sx >= geom->in.source.w || sw > geom->in.source.w || sx+sw > geom->in.source.w ||
       sy >= geom->in.source.h || sh > geom->in.source.h || sy+sh > geom->in.source.h){
        return 2;
    }
    
    /* Rectangular Check */
    if(geom->out.canvas.p[0].x == geom->out.canvas.p[1].x){
        if(geom->out.canvas.p[2].x != geom->out.canvas.p[3].x ||
           geom->out.canvas.p[1].y != geom->out.canvas.p[2].y ||
           geom->out.canvas.p[0].y != geom->out.canvas.p[3].y){
            return 3;
        }
    }else{
        if(geom->out.canvas.p[0].y != geom->out.canvas.p[1].y ||
           geom->out.canvas.p[2].y != geom->out.canvas.p[3].y ||
           geom->out.canvas.p[0].x != geom->out.canvas.p[3].x ||
           geom->out.canvas.p[1].x != geom->out.canvas.p[2].x){
            return 3;
        }
    }
    if(geom->out.source.p[0].x == geom->out.source.p[1].x){
        if(geom->out.source.p[2].x != geom->out.source.p[3].x ||
           geom->out.source.p[1].y != geom->out.source.p[2].y ||
           geom->out.source.p[0].y != geom->out.source.p[3].y){
            return 3;
        }
    }else{
        if(geom->out.source.p[0].y != geom->out.source.p[1].y ||
           geom->out.source.p[2].y != geom->out.source.p[3].y ||
           geom->out.source.p[0].x != geom->out.source.p[3].x ||
           geom->out.source.p[1].x != geom->out.source.p[2].x){
            return 3;
        }
    }
    
    /* Alignment Check */
    if(geom->in.canvas.chroma_format == BENZINA_CHROMAFMT_YUV420 ||
       geom->in.canvas.chroma_format == BENZINA_CHROMAFMT_YUV422){
        if((cx|cw) & 1){return 4;}
    }
    if(geom->in.source.chroma_format == BENZINA_CHROMAFMT_YUV420 ||
       geom->in.source.chroma_format == BENZINA_CHROMAFMT_YUV422){
        if((sx|sw) & 1){return 4;}
    }
    if(geom->in.canvas.chroma_format == BENZINA_CHROMAFMT_YUV420){
        if((cy|ch) & 1){return 4;}
    }
    if(geom->in.source.chroma_format == BENZINA_CHROMAFMT_YUV420){
        if((sy|sh) & 1){return 4;}
    }
    
    return 0;
}


/**
 * Main
 */

int main(int argc, char* argv[]){
    /**
     * Parse Arguments.
     */
    
    BENZINA_GEOM geom = {0};
    int  transpose, hflip, vflip, crop=0, resize=0, invalid=0, ret;
    char canvasChromaFormat[20] = "yuv420";
    char sourceChromaFormat[20] = "yuv420";
    char sourceChromaLoc   [20] = "left";
    char canvasChromaLoc   [20] = "left";
    int  canvasConvs = argc >= 2 ? sscanf(argv[1], "%u:%u:%20[yuv420super]",
                                          &geom.in.canvas.w,
                                          &geom.in.canvas.h,
                                          canvasChromaFormat) : 0;
    int  sourceConvs = argc >= 3 ? sscanf(argv[2], "%u:%u:%20[yuv420]:%20[a-z]",
                                          &geom.in.source.w,
                                          &geom.in.source.h,
                                          sourceChromaFormat,
                                          sourceChromaLoc) : 0;
    switch(canvasConvs){
        case 0: geom.in.canvas.w = 1024;
        case 1: geom.in.canvas.h = geom.in.canvas.w;
        case 2: 
        default: break;
    }
    switch(sourceConvs){
        case 0: geom.in.source.w = 1024;
        case 1: geom.in.source.h = geom.in.source.w;
        case 2: 
        case 3:
        default: break;
    }
    
    
    /**
     * Print Geometry Problem that will be solved.
     */
    
    printf("                       INPUT\n");
    printf("       |  Width x Height | Chroma Format | Location\n");
    printf("-------+-----------------+---------------+---------\n");
    printf("Canvas |  %5u x %5u  | %13s |\n",    geom.in.canvas.w, geom.in.canvas.h, canvasChromaFormat);
    printf("Source |  %5u x %5u  | %13s | %s\n", geom.in.source.w, geom.in.source.h, sourceChromaFormat, sourceChromaLoc);
    
    
    /* Solve the problem. */
    benzina_geom_solve(&geom);
    ret       = validate(&geom);
    transpose = geom.out.transpose;
    hflip     = geom.out.hflip;
    vflip     = geom.out.vflip;
    crop      = rect2dw(&geom.out.source) != (transpose ? geom.in.source.h : geom.in.source.w) ||
                rect2dh(&geom.out.source) != (transpose ? geom.in.source.w : geom.in.source.h);
    resize    = rect2dw(&geom.out.source) != (transpose ? rect2dh : rect2dw)(&geom.out.canvas) ||
                rect2dh(&geom.out.source) != (transpose ? rect2dw : rect2dh)(&geom.out.canvas);
    invalid   = ret != 0;
    
    
    /**
     * Print solved Geometry Problem.
     * 
     * Flags:
     *     (T)ranspose
     *     (H)orizontal flip
     *     (V)ertical flip
     *     (C)rop
     *     (R)esize
     *     (I)nvalid
     */
    
    printf("\n\n");
    printf("                       OUTPUT\n");
    printf("Description     | Value\n");
    printf("----------------+---------------------------------------------\n");
    printf("Canvas Corners  | (%5u,%5u) -- (%5u,%5u) -- (%5u,%5u) -- (%5u,%5u) -- cycle\n", geom.out.canvas.p[0].x, geom.out.canvas.p[0].y, geom.out.canvas.p[1].x, geom.out.canvas.p[1].y, geom.out.canvas.p[2].x, geom.out.canvas.p[2].y, geom.out.canvas.p[3].x, geom.out.canvas.p[3].y);
    printf("Source Corners  | (%5u,%5u) -- (%5u,%5u) -- (%5u,%5u) -- (%5u,%5u) -- cycle\n", geom.out.source.p[0].x, geom.out.source.p[0].y, geom.out.source.p[1].x, geom.out.source.p[1].y, geom.out.source.p[2].x, geom.out.source.p[2].y, geom.out.source.p[3].x, geom.out.source.p[3].y);
    printf("Chroma Location | %13s -> %-13s (source -> canvas)\n", sourceChromaLoc, canvasChromaLoc);
    printf("Flags           | %c%c%c%c%c%c\n", transpose?'T':'-', hflip?'H':'-', vflip?'V':'-', crop?'C':'-', resize?'R':'-', invalid?'I':'-');
    
    return ret;
}
