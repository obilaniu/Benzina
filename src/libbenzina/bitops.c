/* Includes */
#include "benzina/bitops.h"


/* Function Definitions */

/**
 * Materialize extern versions of bit-manipulation inline functions from the
 * header, in case:
 * 
 *   - Inlining somehow failed or
 *   - We want to dlsym() these symbols.
 */

BENZINA_PUBLIC BENZINA_EXTERN_INLINE int benz_popcnt64(uint64_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int benz_clz64   (uint64_t x);

