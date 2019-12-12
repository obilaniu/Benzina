/* Includes */
#include "benzina/ptrops.h"


/* Function Definitions */

/**
 * Materialize extern versions of bit-manipulation inline functions from the
 * header, in case:
 * 
 *   - Inlining somehow failed or
 *   - We want to dlsym() these symbols.
 */

BENZINA_PUBLIC BENZINA_EXTERN_INLINE const void* benz_ptr_addcv(const void* p, int64_t  off);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void*       benz_ptr_addnv(void*       p, int64_t  off);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int64_t     benz_ptr_diffv(const void* p, const void* q);

