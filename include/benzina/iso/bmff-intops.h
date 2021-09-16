/* Include Guard */
#ifndef INCLUDE_BENZINA_ISO_BMFF_INTOPS_H
#define INCLUDE_BENZINA_ISO_BMFF_INTOPS_H


/**
 * Includes
 */

#include <stdint.h>
#include <string.h>
#include "benzina/endian.h"
#include "benzina/inline.h"
#include "benzina/visibility.h"
#include "benzina/ptrops.h"


/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif



/**
 * Inline Function Definitions for ISO BMFF support.
 */

/**
 * @brief Read at pointer as 8/16/32/64-bit unsigned integer.
 * 
 * Converts to native endianness on-the-fly.
 * 
 * @param [in]  p  Pointer at which to read integer
 * @return Integer.
 */

BENZINA_PUBLIC BENZINA_ATTRIBUTE_PURE BENZINA_INLINE uint8_t       benz_iso_bmff_as_u8               (const void* p){
    return benz_getbe8(p, 0);
}
BENZINA_PUBLIC BENZINA_ATTRIBUTE_PURE BENZINA_INLINE uint16_t      benz_iso_bmff_as_u16              (const void* p){
    return benz_getbe16(p, 0);
}
BENZINA_PUBLIC BENZINA_ATTRIBUTE_PURE BENZINA_INLINE uint32_t      benz_iso_bmff_as_u32              (const void* p){
    return benz_getbe32(p, 0);
}
BENZINA_PUBLIC BENZINA_ATTRIBUTE_PURE BENZINA_INLINE uint64_t      benz_iso_bmff_as_u64              (const void* p){
    return benz_getbe64(p, 0);
}

/**
 * @brief Read at pointer as 8/16/32/64-bit unsigned integer or an ASCII string.
 * 
 * Converts integers to native endianness on-the-fly and advances the pointer.
 * 
 * @param [in]      p     Pointer at which to read integer
 * @param [in,out]  pOut  On exit, points immediately after the value that was read.
 * @return Integer or string.
 */

BENZINA_PUBLIC                        BENZINA_INLINE uint8_t       benz_iso_bmff_get_u8              (const void* p, const void** pOut){
    uint8_t x = benz_iso_bmff_as_u8(p);
    *pOut = benz_ptr_addcv(p, sizeof(x));
    return x;
}
BENZINA_PUBLIC                        BENZINA_INLINE uint16_t      benz_iso_bmff_get_u16             (const void* p, const void** pOut){
    uint16_t x = benz_iso_bmff_as_u16(p);
    *pOut = benz_ptr_addcv(p, sizeof(x));
    return x;
}
BENZINA_PUBLIC                        BENZINA_INLINE uint32_t      benz_iso_bmff_get_u32             (const void* p, const void** pOut){
    uint32_t x = benz_iso_bmff_as_u32(p);
    *pOut = benz_ptr_addcv(p, sizeof(x));
    return x;
}
BENZINA_PUBLIC                        BENZINA_INLINE uint64_t      benz_iso_bmff_get_u64             (const void* p, const void** pOut){
    uint64_t x = benz_iso_bmff_as_u64(p);
    *pOut = benz_ptr_addcv(p, sizeof(x));
    return x;
}
BENZINA_PUBLIC                        BENZINA_INLINE const char*   benz_iso_bmff_get_asciz           (const void* p, const void** pOut){
    size_t len = strlen(p);
    *pOut = benz_ptr_addcv(p, len+1);
    return p;
}

/**
 * @brief Write at pointer an 8/16/32/64-bit unsigned integer, an ASCII string
 *        with or without its NUL terminator, or an array of bytes.
 * 
 * Converts integers from native endianness on-the-fly and advances the pointer.
 * 
 * @param [in]      p     Pointer at which to write integer
 * @param [in]      x     An integer to write.
 * @param [in]      str   A string to write.
 * @param [in]      buf   A buffer to write.
 * @param [in]      len   The buffer's length.
 * @param [in,out]  pOut  On exit, points immediately after the value that was read.
 * @return The pointer p, advanced by the number of bytes written.
 */

BENZINA_PUBLIC                        BENZINA_INLINE void*         benz_iso_bmff_put_u8              (void*       p, uint8_t  x){
    benz_putbe8(p, 0, x);
    return benz_ptr_addnv(p, sizeof(x));
}
BENZINA_PUBLIC                        BENZINA_INLINE void*         benz_iso_bmff_put_u16             (void*       p, uint16_t x){
    benz_putbe16(p, 0, x);
    return benz_ptr_addnv(p, sizeof(x));
}
BENZINA_PUBLIC                        BENZINA_INLINE void*         benz_iso_bmff_put_u32             (void*       p, uint32_t x){
    benz_putbe32(p, 0, x);
    return benz_ptr_addnv(p, sizeof(x));
}
BENZINA_PUBLIC                        BENZINA_INLINE void*         benz_iso_bmff_put_u64             (void*       p, uint64_t x){
    benz_putbe64(p, 0, x);
    return benz_ptr_addnv(p, sizeof(x));
}
BENZINA_PUBLIC                        BENZINA_INLINE void*         benz_iso_bmff_put_ascii           (void*       p, const char* str){
    size_t len = strlen(str);
    memcpy(p, str, len);
    return benz_ptr_addnv(p, len);
}
BENZINA_PUBLIC                        BENZINA_INLINE void*         benz_iso_bmff_put_asciz           (void*       p, const char* str){
    size_t len = strlen(str)+1;
    memcpy(p, str, len);
    return benz_ptr_addnv(p, len);
}
BENZINA_PUBLIC                        BENZINA_INLINE void*         benz_iso_bmff_put_bytes           (void*       p, const void* buf, uint64_t len){
    memcpy(p, buf, len);
    return benz_ptr_addnv(p, len);
}

/**
 * @brief Compare two 4-byte type codes for equality.
 * 
 * @param [in]  p  Pointer to first  type code.
 * @param [in]  q  Pointer to second type code.
 * @return 0 (false) if unequal; !0 (true) if equal.
 */

BENZINA_PUBLIC BENZINA_ATTRIBUTE_PURE BENZINA_INLINE int           benz_iso_bmff_is_type_equal       (const void* p, const void* q){
    return memcmp(p, q, sizeof(uint32_t)) == 0;
}


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

