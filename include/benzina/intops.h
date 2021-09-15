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

/**
 * @brief Operations for bit-counting on 64-bit integers.
 * 
 *   - popcnt: Count number of bits set (=1).
 *   - clz:    Count number of leading  zero bits (from most-significant  bit).
 *   - ctz:    Count number of trailing zero bits (from least-significant bit).
 * 
 * A GCC-specific builtin function is used where possible; Otherwise a
 * substitute C implementation is provided.
 */

#if __GNUC__
BENZINA_ATTRIBUTE_CONST
BENZINA_ATTRIBUTE_ALWAYSINLINE
BENZINA_PUBLIC  BENZINA_INLINE
unsigned benz_popcnt64(uint64_t x){
    return __builtin_popcountll(x);
}

BENZINA_ATTRIBUTE_CONST
BENZINA_ATTRIBUTE_ALWAYSINLINE
BENZINA_PUBLIC  BENZINA_INLINE
unsigned benz_clz64(uint64_t x){
    return x ? __builtin_clzll(x) : 64;
}

BENZINA_ATTRIBUTE_CONST
BENZINA_ATTRIBUTE_ALWAYSINLINE
BENZINA_PUBLIC  BENZINA_INLINE
unsigned benz_ctz64(uint64_t x){
    return x ? __builtin_ctzll(x) : 64;
}
#else
BENZINA_ATTRIBUTE_CONST
BENZINA_ATTRIBUTE_ALWAYSINLINE
BENZINA_PUBLIC  BENZINA_INLINE
unsigned benz_popcnt64(uint64_t x){
    /* SWAR */
    x  = (x>>1 & 0x5555555555555555ULL) + (x & 0x5555555555555555ULL);
    x  = (x>>2 & 0x3333333333333333ULL) + (x & 0x3333333333333333ULL);
    x  = (x>>4 & 0x0F0F0F0F0F0F0F0FULL) + (x & 0x0F0F0F0F0F0F0F0FULL);
    /* Digit convolution, sum of all bytes in [63:56] */
    x *= 0x0101010101010101;
    return x >> 56;
}

BENZINA_ATTRIBUTE_CONST
BENZINA_ATTRIBUTE_ALWAYSINLINE
BENZINA_PUBLIC  BENZINA_INLINE
unsigned benz_clz64(uint64_t x){
    x |= x>> 1;
    x |= x>> 2;
    x |= x>> 4;
    x |= x>> 8;
    x |= x>>16;
    x |= x>>32;
    return 64-benz_popcnt64(x);
}

BENZINA_ATTRIBUTE_CONST
BENZINA_ATTRIBUTE_ALWAYSINLINE
BENZINA_PUBLIC  BENZINA_INLINE
unsigned benz_ctz64(uint64_t x){
    x |= x<< 1;
    x |= x<< 2;
    x |= x<< 4;
    x |= x<< 8;
    x |= x<<16;
    x |= x<<32;
    return 64-benz_popcnt64(x);
}
#endif

/**
 * @brief Bit rotation on integers.
 * 
 *   - rol: Bit Rotate Left
 *   - ror: Bit Rotate Right
 * 
 * @param [in] x  An integer.
 * @return The same integer, rotated by c bits over its bitwidth.
 */

BENZINA_ATTRIBUTE_CONST
BENZINA_ATTRIBUTE_ALWAYSINLINE
BENZINA_PUBLIC  BENZINA_INLINE
uint64_t benz_rol64(uint64_t x, unsigned c){
    const unsigned BW = 64;
    c &= BW-1;
    return !c ? x : x<<c | x>>(BW-c);
}

BENZINA_ATTRIBUTE_CONST
BENZINA_ATTRIBUTE_ALWAYSINLINE
BENZINA_PUBLIC  BENZINA_INLINE
uint64_t benz_ror64(uint64_t x, unsigned c){
    const unsigned BW = 64;
    c &= BW-1;
    return !c ? x : x>>c | x<<(BW-c);
}

/**
 * @brief Exp-Golomb decoding on 64-bit shift registers.
 * 
 * The Exp-Golomb coded value is assumed to occupy the most significant bits of
 * the shift register.
 * 
 *   - ueto64: Unsigned Exp-Golomb-coded integer to 64-bit integer.
 *   - seto64: Signed   Exp-Golomb-coded integer to 64-bit integer.
 * 
 * @param [in] x  A 64-bit shift register value.
 * @return A signed or unsigned decoded integer from the shift register.
 */

BENZINA_ATTRIBUTE_CONST
BENZINA_ATTRIBUTE_ALWAYSINLINE
BENZINA_PUBLIC  BENZINA_INLINE
uint64_t benz_ueto64(uint64_t x){
    int k = benz_clz64(x);
    x <<= k;
    x >>= 63-k;
    return x-1;
}

BENZINA_ATTRIBUTE_CONST
BENZINA_ATTRIBUTE_ALWAYSINLINE
BENZINA_PUBLIC  BENZINA_INLINE
int64_t  benz_seto64(uint64_t x){
    uint64_t v = benz_ueto64(x);
    v = v&1 ? v+1 : -v;
    return (int64_t)v>>1;
}


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

