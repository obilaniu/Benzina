/* Include Guard */
#ifndef INCLUDE_BENZINA_H
#define INCLUDE_BENZINA_H

/**
 * Includes
 */

#include <stddef.h>



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

int          benzinaInit(void);

int          benzinaDatasetAlloc    (BENZINA_DATASET**     ctx);
int          benzinaDatasetInit     (BENZINA_DATASET*      ctx, const char*  path);
int          benzinaDatasetNew      (BENZINA_DATASET**     ctx, const char*  path);
int          benzinaDatasetFini     (BENZINA_DATASET*      ctx);
int          benzinaDatasetFree     (BENZINA_DATASET*      ctx);
int          benzinaDatasetGetLength(BENZINA_DATASET*      ctx, size_t* length);
int          benzinaDatasetGetShape (BENZINA_DATASET*      ctx, size_t* w, size_t* h);

int          benzinaLoaderIterAlloc (BENZINA_LOADER_ITER** ctx);
int          benzinaLoaderIterInit  (BENZINA_LOADER_ITER*  ctx,
                                     BENZINA_DATASET*      dataset,
                                     int                   device,
                                     size_t                pipelineDepth);
int          benzinaLoaderIterNew   (BENZINA_LOADER_ITER** ctx, const char*  path);
int          benzinaLoaderIterFini  (BENZINA_LOADER_ITER*  ctx);
int          benzinaLoaderIterFree  (BENZINA_LOADER_ITER*  ctx);


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

