/* Include Guard */
#ifndef SRC_TOOLS_IMG2VID_KERNELS_H
#define SRC_TOOLS_IMG2VID_KERNELS_H


/**
 * Defines
 */




/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif



/**
 * Includes
 */

#include <libavutil/avutil.h>
#include "main.h"



/* Function Prototypes */
extern int  i2v_cuda_filter(UNIVERSE* u, AVFrame* dst, AVFrame* src, BENZINA_GEOM* geom);


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

