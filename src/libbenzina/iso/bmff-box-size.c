/* Includes */
#include "benzina/ptrops.h"
#include "benzina/iso/bmff-intops.h"
#include "benzina/iso/bmff-types.h"





/* Function Definitions */

/**
 * @brief Get bounded size of box.
 * 
 * NOTE: THIS DEFINITION IS *NOT* AND *CANNOT* BE THE SAME AS THAT OF THE HEADER!
 * NOTE: THE HEADER DEFINITION IS INLINE PURE TAIL-SELF-RECURSIVE!
 * 
 * @param [in]  p     Pointer to box.
 * @param [in]  tail  Pointer to bound of box. Must be >= p.
 * @return Size of the box, if present and sane; Otherwise 0.
 */

BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t benz_iso_bmff_box_get_size        (const BENZ_ISO_BOX*     p, const void* tail){
    int64_t  max_size;
    uint32_t size32;
    uint64_t size64;
    
    max_size = benz_ptr_diffv(tail, p);
    if(max_size < 8)
        return 0;
    
    size32   = benz_iso_bmff_as_u32(p);
    if(size32 >= 8)
        return size32>max_size ? 0 : size32;
    if(size32 == 0)
        return max_size;
    if(size32 != 1)
        return 0;
    
    if(max_size < 16)
        return 0;
    
    size64   = benz_iso_bmff_as_u64(benz_ptr_addcv(p, 8));
    if(size64 < 16)
        return 0;
    if(size64 > (uint64_t)max_size)
        return 0;
    
    return size64;
}

