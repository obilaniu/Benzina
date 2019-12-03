/* Includes */
#include "benzina/iso/iso.h"



/* Defines */



/* Function Definitions */

/**
 * Materialize extern versions of ISO BMFF support inline functions from the
 * header, in case:
 * 
 *   - Inlining somehow failed or
 *   - We want to dlsym() these symbols.
 */

BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint8_t     benz_iso_as_u8                   (const void* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint16_t    benz_iso_as_u16                  (const void* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t    benz_iso_as_u32                  (const void* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_iso_as_u64                  (const void* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint8_t     benz_iso_get_u8                  (const void* p, const void** pOut);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint16_t    benz_iso_get_u16                 (const void* p, const void** pOut);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t    benz_iso_get_u32                 (const void* p, const void** pOut);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_iso_get_u64                 (const void* p, const void** pOut);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE const char* benz_iso_get_asciz               (const void* p, const void** pOut);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void*       benz_iso_put_u8                  (void*       p, uint8_t  x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void*       benz_iso_put_u16                 (void*       p, uint16_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void*       benz_iso_put_u32                 (void*       p, uint32_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void*       benz_iso_put_u64                 (void*       p, uint64_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void*       benz_iso_put_ascii               (void*       p, const char* str);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void*       benz_iso_put_asciz               (void*       p, const char* str);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void*       benz_iso_put_bytes               (void*       p, const void* buf, uint64_t len);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_iso_is_type_equal           (const void* p, const void* type);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_iso_box_get_size            (const BENZ_ISO_BOX*   p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_iso_box_get_size_bounded    (const BENZ_ISO_BOX*   p, uint64_t max_size);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE const char* benz_iso_box_get_type_str        (const BENZ_ISO_BOX*   p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t    benz_iso_box_get_type            (const BENZ_ISO_BOX*   p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_iso_box_is_type             (const BENZ_ISO_BOX*   p, const void* type);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE const char* benz_iso_box_get_exttype         (const BENZ_ISO_BOX*   p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE const void* benz_iso_box_get_payload         (const BENZ_ISO_BOX*   p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_iso_fullbox_get_size        (const BENZ_ISO_BOX*   p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_iso_fullbox_get_size_bounded(const BENZ_ISO_BOX*   p, uint64_t max_size);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE const char* benz_iso_fullbox_get_type_str    (const BENZ_ISO_BOX*   p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_iso_fullbox_get_type        (const BENZ_ISO_BOX*   p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_iso_fullbox_is_type         (const BENZ_ISO_BOX*   p, const void* type);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE const char* benz_iso_fullbox_get_exttype     (const BENZ_ISO_BOX*   p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t    benz_iso_fullbox_get_flagver     (const BENZ_ISO_BOX*   p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t    benz_iso_fullbox_get_version     (const BENZ_ISO_BOX*   p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t    benz_iso_fullbox_get_flags       (const BENZ_ISO_BOX*   p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE const void* benz_iso_fullbox_get_payload     (const BENZ_ISO_BOX*   p);

