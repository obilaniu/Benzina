/* Include Guard */
#ifndef INCLUDE_BENZINA_ISO_BMFF_BOX_H
#define INCLUDE_BENZINA_ISO_BMFF_BOX_H


/**
 * Includes
 */

#include <stdint.h>
#include <string.h>
#include "benzina/inline.h"
#include "benzina/ptrops.h"
#include "benzina/visibility.h"
#include "benzina/iso/bmff-types.h"
#include "benzina/iso/bmff-intops.h"


/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif


/**
 * Inline Function Definitions for ISO BMFF support.
 */

/**
 * @brief Get size of box.
 * 
 * @param [in]  p     Pointer to box.
 * @return Size of the box as reported in box header.
 * @note   Assumes the ability to read up to 16 bytes beyond p.
 */

BENZINA_PUBLIC BENZINA_ATTRIBUTE_PURE BENZINA_INLINE uint64_t      benz_iso_bmff_box_get_size_unsafe (const BENZ_ISO_BOX*     p){
    uint32_t size = benz_iso_bmff_as_u32(p);
    return size != 1 ? size : benz_iso_bmff_as_u64(benz_ptr_addcv(p, 8));
}

/**
 * @brief Get bounded size of box.
 * 
 * @param [in]  p     Pointer to box.
 * @param [in]  tail  Pointer to bound of box. Must be >= p.
 * @return Size of the box, if present and sane; Otherwise 0.
 */

BENZINA_PUBLIC BENZINA_ATTRIBUTE_PURE BENZINA_INLINE uint64_t      benz_iso_bmff_box_get_size        (const BENZ_ISO_BOX*     p, const void* tail){
    int64_t  max_size;
    uint32_t size32;
    
    max_size = benz_ptr_diffv(tail, p);
    if(max_size < 8)
        return 0;
    
    size32    = benz_iso_bmff_as_u32(p);
    if(size32 < 8)
        return benz_iso_bmff_box_get_size(p, tail);
    
    return size32>max_size ? 0 : size32;
}

/**
 * @brief Get bounded tail of box.
 * 
 * @param [in]  p     Pointer to box.
 * @param [in]  tail  Pointer to bound of box. Must be >= p.
 * @return p + bounded size of box, if present and sane; Otherwise p.
 */

BENZINA_PUBLIC BENZINA_ATTRIBUTE_PURE BENZINA_INLINE const void*   benz_iso_bmff_box_get_tail        (const BENZ_ISO_BOX*     p, const void* tail){
    return benz_ptr_addcv(p, benz_iso_bmff_box_get_size(p, tail));
}

/**
 * @brief Skip n contiguous boxes from a given starting point.
 * 
 * @param [in]  p     Pointer to box.
 * @param [in]  tail  Pointer to bound of box. Must be >= p.
 * @param [in[  n     Number of boxes to skip over.
 * @return Pointer to one byte past the end of the (n-1)'th box after this one.
 *         The case n=0 indicates "this" box; Therefore, skip(p, tail, 0)==p.
 */

BENZINA_PUBLIC BENZINA_ATTRIBUTE_PURE BENZINA_INLINE const void*   benz_iso_bmff_box_skip            (const BENZ_ISO_BOX*     p, const void* tail, uint64_t n){
    const BENZ_ISO_BOX* q;
    
    for(;n--;p=q){
        q = benz_iso_bmff_box_get_tail(p, tail);
        if(p==q)
            return NULL;
    }
    
    return p;
}

/**
 * @brief Get type of box, as a 4-byte string or integer.
 * 
 * @param [in]  p     Pointer to box.
 * @return Type code, as string or integer.
 */

BENZINA_PUBLIC BENZINA_ATTRIBUTE_PURE BENZINA_INLINE const char*   benz_iso_bmff_box_get_type_str    (const BENZ_ISO_BOX*     p){
    return benz_ptr_addcv(p, 4);
}
BENZINA_PUBLIC BENZINA_ATTRIBUTE_PURE BENZINA_INLINE uint32_t      benz_iso_bmff_box_get_type        (const BENZ_ISO_BOX*     p){
    return benz_iso_bmff_as_u32(benz_iso_bmff_box_get_type_str(p));
}

/**
 * @brief Check that box has given type.
 * 
 * @param [in]  p     Pointer to box.
 * @param [in]  type  Type to compare for equality.
 * @return 0 (false) if box does not have exactly the given type; !0 (true) otherwise.
 */

BENZINA_PUBLIC BENZINA_ATTRIBUTE_PURE BENZINA_INLINE int           benz_iso_bmff_box_is_type         (const BENZ_ISO_BOX*     p, const void* type){
    return benz_iso_bmff_is_type_equal(benz_iso_bmff_box_get_type_str(p), type);
}

/**
 * @brief Get extended type of "uuid" box.
 * 
 * @param [in]  p     Pointer to box.
 * @return Pointer to UUID of box if the box is of type "uuid"; NULL otherwise.
 */

BENZINA_PUBLIC BENZINA_ATTRIBUTE_PURE BENZINA_INLINE const char*   benz_iso_bmff_box_get_exttype     (const BENZ_ISO_BOX*     p){
    size_t offset = 8;
    if(!benz_iso_bmff_box_is_type(p, "uuid")){return NULL;}
    offset += benz_iso_bmff_as_u32(p)         == 1 ?  8 : 0;
    return benz_ptr_addcv(p, offset);
}

/**
 * @brief Get payload, flags and version of box as a pointer, integer and integer respectively.
 * 
 * The flags and version are applicable only to FullBoxes.
 * 
 * @param [in]  p     Pointer to box.
 * @return Pointer to UUID of box if the box is of type "uuid"; NULL otherwise.
 */

BENZINA_PUBLIC BENZINA_ATTRIBUTE_PURE BENZINA_INLINE const void*   benz_iso_bmff_box_get_payload     (const BENZ_ISO_BOX*     p, uint64_t offset){
    offset += benz_iso_bmff_as_u32(p)         == 1 ?  8 : 0;
    offset += benz_iso_bmff_box_is_type(p, "uuid") ? 16 : 0;
    return benz_ptr_addcv(p, offset);
}
BENZINA_PUBLIC BENZINA_ATTRIBUTE_PURE BENZINA_INLINE uint32_t      benz_iso_bmff_box_get_flagver     (const BENZ_ISO_FULLBOX* p){
    return benz_iso_bmff_as_u32(benz_iso_bmff_box_get_payload(p, 0));
}
BENZINA_PUBLIC BENZINA_ATTRIBUTE_PURE BENZINA_INLINE uint32_t      benz_iso_bmff_box_get_version     (const BENZ_ISO_FULLBOX* p){
    return benz_iso_bmff_box_get_flagver(p) >> 24;
}
BENZINA_PUBLIC BENZINA_ATTRIBUTE_PURE BENZINA_INLINE uint32_t      benz_iso_bmff_box_get_flags       (const BENZ_ISO_FULLBOX* p){
    return benz_iso_bmff_box_get_flagver(p) & 0xFFFFFFUL;
}

/**
 * @brief Write into a box a 32-bit size, a 64-bit size, a type code or flagver field.
 * 
 * @param [in]  p     Pointer to box.
 * @param [in]  size  Size of box being created.
 * @param [in]  type  Type of box being created.
 * @return Pointer to one-byte-past-the-end of what was just written.
 */

BENZINA_PUBLIC                        BENZINA_INLINE void*         benz_iso_bmff_box_put_size32      (BENZ_ISO_BOX*           p, uint32_t size){
    return benz_iso_bmff_put_u32(p, size);
}
BENZINA_PUBLIC                        BENZINA_INLINE void*         benz_iso_bmff_box_put_size64      (BENZ_ISO_BOX*           p, uint64_t size){
    benz_iso_bmff_box_put_size32(p, 1);
    return benz_iso_bmff_put_u64(benz_ptr_addnv(p, 8), size);
}
BENZINA_PUBLIC                        BENZINA_INLINE void*         benz_iso_bmff_box_put_size        (BENZ_ISO_BOX*           p, uint64_t size){
    return size <= 0xFFFFFFFFU ? benz_iso_bmff_box_put_size32(p, size):
                                 benz_iso_bmff_box_put_size64(p, size);
}
BENZINA_PUBLIC                        BENZINA_INLINE void*         benz_iso_bmff_box_put_type_str    (BENZ_ISO_BOX*           p, const void* type){
    memcpy(benz_ptr_addnv(p, 4), type, 4);
    return benz_ptr_addnv(p, 8);
}
BENZINA_PUBLIC                        BENZINA_INLINE void*         benz_iso_bmff_box_put_type        (BENZ_ISO_BOX*           p, uint32_t type){
    return benz_iso_bmff_put_u32(benz_ptr_addnv(p, 4), type);
}
BENZINA_PUBLIC                        BENZINA_INLINE void*         benz_iso_bmff_box_put_flagver     (BENZ_ISO_FULLBOX*       p, uint32_t flagver){
    return benz_iso_bmff_put_u32((void*)benz_iso_bmff_box_get_payload(p, 0), flagver);
}



/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

