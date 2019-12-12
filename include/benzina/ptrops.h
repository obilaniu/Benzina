/* Include Guard */
#ifndef INCLUDE_BENZINA_PTROPS_H
#define INCLUDE_BENZINA_PTROPS_H


/**
 * Includes
 */

#include <limits.h>
#include <stdint.h>
#include "benzina/config.h"
#include "benzina/inline.h"
#include "benzina/visibility.h"


/* Static Asserts */
#if __STDC_VERSION__ >= 201112L
_Static_assert(CHAR_BIT         == 8, "Size of byte assumed to be 8 bits!");
_Static_assert(sizeof(uint8_t)  == 1, "Size of uint8_t assumed to be equal to that of 1 char!");
_Static_assert(sizeof(uint16_t) == 2, "Size of uint16_t assumed to be equal to that of 2 char!");
_Static_assert(sizeof(uint32_t) == 4, "Size of uint32_t assumed to be equal to that of 4 char!");
_Static_assert(sizeof(uint64_t) == 8, "Size of uint64_t assumed to be equal to that of 8 char!");
#endif


/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief Void pointer add/offset ((c)onst/(n)on-const)
 * 
 * @param [in]  p    (void) pointer to offset
 * @param [in]  off  Signed integer number of bytes.
 * @return p+off, as const/non-const void*
 */

BENZINA_PUBLIC BENZINA_ATTRIBUTE_CONST BENZINA_INLINE const void* benz_ptr_addcv(const void* p, int64_t  off){
    return (const void*)((const uint8_t*)p + off);
}
BENZINA_PUBLIC BENZINA_ATTRIBUTE_CONST BENZINA_INLINE void*       benz_ptr_addnv(void*       p, int64_t  off){
    return (      void*)((      uint8_t*)p + off);
}

/**
 * @brief Void pointer difference
 * @param [in]  p  First  pointer
 * @param [in]  q  Second pointer
 * @return p-q, as int64_t
 */

BENZINA_PUBLIC BENZINA_ATTRIBUTE_CONST BENZINA_INLINE int64_t     benz_ptr_diffv(const void* p, const void* q){
    return (int64_t)((const char*)p-(const char*)q);
}


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

