/* Include Guard */
#ifndef INCLUDE_BENZINA_BENZINA_H
#define INCLUDE_BENZINA_BENZINA_H

/**
 * Includes
 */

#include <stddef.h>
#include "visibility.h"



/* Defines */




/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif



/* Data Structures Forward Declarations and Typedefs */
typedef struct BENZINA_DATASET     BENZINA_DATASET;
typedef struct BENZINA_LOADER_ITER BENZINA_LOADER_ITER;



/* Function Prototypes */

/**
 * @brief Benzina Initialization.
 * 
 * @return Zero if successful; Non-zero if not successful.
 */

BENZINA_PUBLIC int          benzinaInit(void);


BENZINA_PUBLIC int          benzinaDatasetAlloc     (BENZINA_DATASET**     ctx);
BENZINA_PUBLIC int          benzinaDatasetInit      (BENZINA_DATASET*      ctx, const char*  path);
BENZINA_PUBLIC int          benzinaDatasetNew       (BENZINA_DATASET**     ctx, const char*  path);
BENZINA_PUBLIC int          benzinaDatasetFini      (BENZINA_DATASET*      ctx);
BENZINA_PUBLIC int          benzinaDatasetFree      (BENZINA_DATASET*      ctx);
BENZINA_PUBLIC int          benzinaDatasetGetLength (BENZINA_DATASET*      ctx, size_t* length);
BENZINA_PUBLIC int          benzinaDatasetGetShape  (BENZINA_DATASET*      ctx, size_t* w, size_t* h);
BENZINA_PUBLIC int          benzinaDatasetGetElement(BENZINA_DATASET*      ctx,
                                                     size_t                i,
                                                     size_t*               off,
                                                     size_t*               len);


BENZINA_PUBLIC int          benzinaLoaderIterAlloc  (BENZINA_LOADER_ITER** ctx);
BENZINA_PUBLIC int          benzinaLoaderIterInit   (BENZINA_LOADER_ITER*  ctx,
                                                     BENZINA_DATASET*      dataset,
                                                     int                   device,
                                                     size_t                pipelineDepth);
BENZINA_PUBLIC int          benzinaLoaderIterNew    (BENZINA_LOADER_ITER** ctx, const char*  path);
BENZINA_PUBLIC int          benzinaLoaderIterFini   (BENZINA_LOADER_ITER*  ctx);
BENZINA_PUBLIC int          benzinaLoaderIterFree   (BENZINA_LOADER_ITER*  ctx);


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

