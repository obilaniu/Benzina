/* Include Guard */
#ifndef SRC_TOOLS_IMG2VID_GEOMETRY_H
#define SRC_TOOLS_IMG2VID_GEOMETRY_H

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
typedef struct BENZINA_GEOM     BENZINA_GEOM;



/**
 * @brief The BENZINA_GEOM_ENCODE struct
 * 
 * Used to compute the geometric transformation that places a source image onto
 * the canvas in as lossless a manner as possible.
 */

struct BENZINA_GEOM{
    struct{
        struct{
            uint32_t w, h;
            int      chroma_loc;
            int      chroma_format;
        } src;
        struct{
            uint32_t w, h;
            int      chroma_format;
        } canvas;
    } in;
    
    unsigned wholly_contained_untransposed : 1;
    unsigned wholly_contained_transposed   : 1;
    unsigned wholly_contained              : 1;
    unsigned landscape_src                 : 1;
    unsigned landscape_canvas              : 1;
    unsigned transpose                     : 1;
    unsigned hflip                         : 1;
    unsigned vflip                         : 1;
    unsigned chroma_loc_canvas             : 3;
    unsigned resize                        : 1;
    uint32_t transw, transh;
    uint32_t arcorw, arcorh;
    uint32_t cropw,  croph;
    uint32_t cropx,  cropy;
    uint32_t embedw, embedh;
};



/**
 * Public Function Definitions
 */

extern int benzina_geom_solve(BENZINA_GEOM* geom);


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

