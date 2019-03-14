/* Include Guard */
#ifndef SRC_TOOLS_GEOM_GEOMETRY_H
#define SRC_TOOLS_GEOM_GEOMETRY_H

/**
 * Includes
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "benzina/bits.h"



/* Defines */



/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif



/* Data Structures Forward Declarations and Typedefs */
typedef struct BENZINA_GEOM     BENZINA_GEOM;
typedef struct BENZINA_POINT2D  BENZINA_POINT2D;
typedef struct BENZINA_RECT2D   BENZINA_RECT2D;



/**
 * @brief A 2D point.
 */

struct BENZINA_POINT2D{
    uint32_t x, y;
};

/**
 * @brief A 2D rectangle, defined by its 4 corners.
 */

struct BENZINA_RECT2D{
    BENZINA_POINT2D p[4];
};

/**
 * @brief The BENZINA_GEOM_ENCODE struct
 * 
 * Used to compute the geometric transformation that places a source image onto
 * the canvas in as lossless a manner as possible.
 */

struct BENZINA_GEOM{
    /* Input */
    struct{
        struct{
            uint32_t w, h;
            int      chroma_fmt;
            int      chroma_loc;
        } source;
        struct{
            uint32_t w, h;
            int      chroma_fmt;
            int      superscale;
        } canvas;
    } i;
    
    /* Output */
    struct{
        int transpose, vflip, hflip;
        int chroma_loc;
        BENZINA_RECT2D source, canvas;
    } o;
};


/**
 * Public Function Definitions
 */

extern uint32_t rect2d_x(const BENZINA_RECT2D* r);
extern uint32_t rect2d_y(const BENZINA_RECT2D* r);
extern uint32_t rect2d_w(const BENZINA_RECT2D* r);
extern uint32_t rect2d_h(const BENZINA_RECT2D* r);
extern void     rect2d_scalex(BENZINA_RECT2D* r, uint32_t a, uint32_t b);
extern void     rect2d_scaley(BENZINA_RECT2D* r, uint32_t a, uint32_t b);
extern void     rect2d_transp(BENZINA_RECT2D* r);
extern void     rect2d_hflip (BENZINA_RECT2D* r);
extern void     rect2d_vflip (BENZINA_RECT2D* r);
extern void     rect2d_cropw (BENZINA_RECT2D* r, uint32_t w);
extern void     rect2d_croph (BENZINA_RECT2D* r, uint32_t h);
extern void     rect2d_transl(BENZINA_RECT2D* r, uint32_t x, uint32_t y);

extern int      benzina_geom_solve(BENZINA_GEOM* geom);


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

