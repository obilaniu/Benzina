/* Include Guard */
#ifndef INCLUDE_BENZINA_INTOPS_H
#define INCLUDE_BENZINA_INTOPS_H


/**
 * Includes
 */

#include <stdint.h>
#include "benzina/config.h"
#include "benzina/inline.h"
#include "benzina/visibility.h"


/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif


/**
 * Inline Function Definitions for Bit Manipulation
 */

#if __GNUC__
BENZINA_PUBLIC BENZINA_ATTRIBUTE_ALWAYSINLINE BENZINA_ATTRIBUTE_CONST BENZINA_INLINE unsigned benz_popcnt64(uint64_t x){return __builtin_popcountll(x);}
BENZINA_PUBLIC BENZINA_ATTRIBUTE_ALWAYSINLINE BENZINA_ATTRIBUTE_CONST BENZINA_INLINE unsigned benz_clz64(uint64_t x){return x ? __builtin_clzll(x) : 64;}
#else
BENZINA_PUBLIC BENZINA_ATTRIBUTE_ALWAYSINLINE BENZINA_ATTRIBUTE_CONST BENZINA_INLINE unsigned benz_popcnt64(uint64_t x){
    /* SWAR */
    x  = (x>>1 & 0x5555555555555555ULL) + (x & 0x5555555555555555ULL);
    x  = (x>>2 & 0x3333333333333333ULL) + (x & 0x3333333333333333ULL);
    x  = (x>>4 & 0x0F0F0F0F0F0F0F0FULL) + (x & 0x0F0F0F0F0F0F0F0FULL);
    /* Digit convolution, sum of all bytes in [63:56] */
    x *= 0x0101010101010101;
    return x >> 56;
}
BENZINA_PUBLIC BENZINA_ATTRIBUTE_ALWAYSINLINE BENZINA_ATTRIBUTE_CONST BENZINA_INLINE unsigned benz_clz64(uint64_t x){
    x |= x>> 1;
    x |= x>> 2;
    x |= x>> 4;
    x |= x>> 8;
    x |= x>>16;
    x |= x>>32;
    return 64-benz_popcnt64(x);
}
#endif

BENZINA_PUBLIC BENZINA_ATTRIBUTE_ALWAYSINLINE BENZINA_ATTRIBUTE_CONST BENZINA_INLINE uint64_t benz_ueto64(uint64_t x){
    int k = benz_clz64(x);
    x <<= k;
    x >>= 63-k;
    return x-1;
}
BENZINA_PUBLIC BENZINA_ATTRIBUTE_ALWAYSINLINE BENZINA_ATTRIBUTE_CONST BENZINA_INLINE int64_t  benz_seto64(uint64_t x){
    uint64_t v = benz_ueto64(x);
    v = v&1 ? v+1 : -v;
    return (int64_t)v>>1;
}


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

