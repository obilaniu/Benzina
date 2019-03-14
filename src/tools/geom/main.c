/* Includes */
#include <stdio.h>
#include <string.h>
#include "geometry.h"


/**
 * Defines
 */




/**
 * Functions.
 */

/* Utility Functions */
int      strendswith(const char* s, const char* e){
    size_t ls = strlen(s), le = strlen(e);
    return ls<le ? 0 : !strcmp(s+ls-le, e);
}
uint32_t minu32(uint32_t a, uint32_t b){return a<b?a:b;}
uint32_t maxu32(uint32_t a, uint32_t b){return a>b?a:b;}
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
int      str2chromaloc(const char* s){
    if      (!strcmp(s,       "left")){return BENZINA_CHROMALOC_LEFT;
    }else if(!strcmp(s,     "center")){return BENZINA_CHROMALOC_CENTER;
    }else if(!strcmp(s,    "topleft")){return BENZINA_CHROMALOC_TOPLEFT;
    }else if(!strcmp(s,        "top")){return BENZINA_CHROMALOC_TOP;
    }else if(!strcmp(s, "bottomleft")){return BENZINA_CHROMALOC_BOTTOMLEFT;
    }else if(!strcmp(s,     "bottom")){return BENZINA_CHROMALOC_BOTTOM;
    }else{
        return -1;
    }
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
    switch(geom->o.chroma_loc){
        case BENZINA_CHROMALOC_TOPRIGHT:
        case BENZINA_CHROMALOC_RIGHT:
        case BENZINA_CHROMALOC_BOTTOMRIGHT:
            return 1;
    }
    
    /* Oversize Check */
    cx = rect2d_x(&geom->o.canvas);
    cy = rect2d_y(&geom->o.canvas);
    cw = rect2d_w(&geom->o.canvas);
    ch = rect2d_h(&geom->o.canvas);
    
    sx = rect2d_x(&geom->o.source);
    sy = rect2d_y(&geom->o.source);
    sw = rect2d_w(&geom->o.source);
    sh = rect2d_h(&geom->o.source);
    
    if(cx >= geom->i.canvas.w || cw > geom->i.canvas.w || cx+cw > geom->i.canvas.w ||
       cy >= geom->i.canvas.h || ch > geom->i.canvas.h || cy+ch > geom->i.canvas.h ||
       sx >= geom->i.source.w || sw > geom->i.source.w || sx+sw > geom->i.source.w ||
       sy >= geom->i.source.h || sh > geom->i.source.h || sy+sh > geom->i.source.h){
        return 2;
    }
    
    /* Rectangular Check */
    if(geom->o.canvas.p[0].x == geom->o.canvas.p[1].x){
        if(geom->o.canvas.p[2].x != geom->o.canvas.p[3].x ||
           geom->o.canvas.p[1].y != geom->o.canvas.p[2].y ||
           geom->o.canvas.p[0].y != geom->o.canvas.p[3].y){
            return 3;
        }
    }else{
        if(geom->o.canvas.p[0].y != geom->o.canvas.p[1].y ||
           geom->o.canvas.p[2].y != geom->o.canvas.p[3].y ||
           geom->o.canvas.p[0].x != geom->o.canvas.p[3].x ||
           geom->o.canvas.p[1].x != geom->o.canvas.p[2].x){
            return 3;
        }
    }
    if(geom->o.source.p[0].x == geom->o.source.p[1].x){
        if(geom->o.source.p[2].x != geom->o.source.p[3].x ||
           geom->o.source.p[1].y != geom->o.source.p[2].y ||
           geom->o.source.p[0].y != geom->o.source.p[3].y){
            return 3;
        }
    }else{
        if(geom->o.source.p[0].y != geom->o.source.p[1].y ||
           geom->o.source.p[2].y != geom->o.source.p[3].y ||
           geom->o.source.p[0].x != geom->o.source.p[3].x ||
           geom->o.source.p[1].x != geom->o.source.p[2].x){
            return 3;
        }
    }
    
    /* Alignment Check */
    if(geom->i.canvas.chroma_fmt == BENZINA_CHROMAFMT_YUV420 ||
       geom->i.canvas.chroma_fmt == BENZINA_CHROMAFMT_YUV422){
        if((cx|cw) & 1){return 4;}
    }
    if(geom->i.source.chroma_fmt == BENZINA_CHROMAFMT_YUV420 ||
       geom->i.source.chroma_fmt == BENZINA_CHROMAFMT_YUV422){
        if((sx|sw) & 1){return 4;}
    }
    if(geom->i.canvas.chroma_fmt == BENZINA_CHROMAFMT_YUV420){
        if((cy|ch) & 1){return 4;}
    }
    if(geom->i.source.chroma_fmt == BENZINA_CHROMAFMT_YUV420){
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
    char canvasChromaFmt[21] = "yuv420";
    char sourceChromaFmt[21] = "yuv420";
    char sourceChromaLoc[21] = "left";
    char canvasChromaLoc[21] = "left";
    int  canvasConvs = argc >= 2 ? sscanf(argv[1], "%u:%u:%20[yuv420super]",
                                          &geom.i.canvas.w, &geom.i.canvas.h,
                                          canvasChromaFmt) : 0;
    int  sourceConvs = argc >= 3 ? sscanf(argv[2], "%u:%u:%20[yuv420]:%20[a-z]",
                                          &geom.i.source.w, &geom.i.source.h,
                                          sourceChromaFmt, sourceChromaLoc) : 0;
    switch(canvasConvs){
        default:
        case 0:
            geom.i.canvas.w          = 1024;
        case 1:
            geom.i.canvas.h          = geom.i.canvas.w;
        case 2:
        case 3:;
    }
    geom.i.canvas.chroma_fmt = str2canvaschromafmt(canvasChromaFmt);
    geom.i.canvas.superscale = strendswith(canvasChromaFmt, "super");
    switch(sourceConvs){
        default:
        case 0:
            geom.i.source.w          = 1024;
        case 1:
            geom.i.source.h          = geom.i.source.w;
        case 2:
        case 3:
        case 4:;
    }
    geom.i.source.chroma_fmt = str2chromafmt(sourceChromaFmt);
    geom.i.source.chroma_loc = str2chromaloc(sourceChromaLoc);
    
    /**
     * Print Geometry Problem that will be solved.
     */
    
    printf("                       INPUT\n");
    printf("       |  Width x Height | Chroma Format | Location\n");
    printf("-------+-----------------+---------------+---------\n");
    printf("Source |  %5u x %5u  | %13s | %s\n", geom.i.source.w, geom.i.source.h, sourceChromaFmt, sourceChromaLoc);
    printf("Canvas |  %5u x %5u  | %13s | %s\n", geom.i.canvas.w, geom.i.canvas.h, canvasChromaFmt, "N/A");
    printf("\n\n");
    
    
    /* Solve the problem. */
    ret       = benzina_geom_solve(&geom);
    if(ret){
        printf("Error in solver! benzina_geom_solve() == %d\n", ret);
        return ret;
    }
    
    /* Validate. */
    ret       = validate(&geom);
    transpose = geom.o.transpose;
    hflip     = geom.o.hflip;
    vflip     = geom.o.vflip;
    crop      = rect2d_w(&geom.o.source) != geom.i.source.w ||
                rect2d_h(&geom.o.source) != geom.i.source.h;
    resize    = rect2d_w(&geom.o.canvas) != (transpose ? rect2d_h(&geom.o.source) :
                                                         rect2d_w(&geom.o.source)) ||
                rect2d_h(&geom.o.canvas) != (transpose ? rect2d_w(&geom.o.source) :
                                                         rect2d_h(&geom.o.source));
    invalid   = ret != 0;
    switch(geom.o.chroma_loc){
        case BENZINA_CHROMALOC_TOPLEFT:     snprintf(canvasChromaLoc, sizeof(canvasChromaLoc), "%s", "topleft");     break;
        case BENZINA_CHROMALOC_TOP:         snprintf(canvasChromaLoc, sizeof(canvasChromaLoc), "%s", "top");         break;
        case BENZINA_CHROMALOC_TOPRIGHT:    snprintf(canvasChromaLoc, sizeof(canvasChromaLoc), "%s", "topright");    break;
        case BENZINA_CHROMALOC_LEFT:        snprintf(canvasChromaLoc, sizeof(canvasChromaLoc), "%s", "left");        break;
        case BENZINA_CHROMALOC_CENTER:      snprintf(canvasChromaLoc, sizeof(canvasChromaLoc), "%s", "center");      break;
        case BENZINA_CHROMALOC_RIGHT:       snprintf(canvasChromaLoc, sizeof(canvasChromaLoc), "%s", "right");       break;
        case BENZINA_CHROMALOC_BOTTOMLEFT:  snprintf(canvasChromaLoc, sizeof(canvasChromaLoc), "%s", "bottomleft");  break;
        case BENZINA_CHROMALOC_BOTTOM:      snprintf(canvasChromaLoc, sizeof(canvasChromaLoc), "%s", "bottom");      break;
        case BENZINA_CHROMALOC_BOTTOMRIGHT: snprintf(canvasChromaLoc, sizeof(canvasChromaLoc), "%s", "bottomright"); break;
        default:                            snprintf(canvasChromaLoc, sizeof(canvasChromaLoc), "%s", "unknown");     break;
    }
    
    
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
    
    printf("                       OUTPUT\n");
    printf("Description     | Value\n");
    printf("----------------+--------------------------------------------------------------------------\n");
    printf("Source Corners  | (%5u,%5u) -- (%5u,%5u) -- (%5u,%5u) -- (%5u,%5u) -- cycle\n", geom.o.source.p[0].x, geom.o.source.p[0].y, geom.o.source.p[1].x, geom.o.source.p[1].y, geom.o.source.p[2].x, geom.o.source.p[2].y, geom.o.source.p[3].x, geom.o.source.p[3].y);
    printf("Canvas Corners  | (%5u,%5u) -- (%5u,%5u) -- (%5u,%5u) -- (%5u,%5u) -- cycle\n", geom.o.canvas.p[0].x, geom.o.canvas.p[0].y, geom.o.canvas.p[1].x, geom.o.canvas.p[1].y, geom.o.canvas.p[2].x, geom.o.canvas.p[2].y, geom.o.canvas.p[3].x, geom.o.canvas.p[3].y);
    printf("Chroma Location | %13s -> %-13s (source -> canvas)\n", sourceChromaLoc, canvasChromaLoc);
    printf("Flags           | %c%c%c%c%c%c\n", transpose?'T':'-', hflip?'H':'-', vflip?'V':'-', crop?'C':'-', resize?'R':'-', invalid?'I':'-');
    
    return ret;
}
