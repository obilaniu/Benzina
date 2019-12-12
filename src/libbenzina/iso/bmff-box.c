/* Includes */
#include "benzina/iso/bmff-box.h"



/* Function Definitions */

/**
 * Materialize extern versions of ISO BMFF support inline functions from the
 * header, in case:
 * 
 *   - Inlining somehow failed or
 *   - We want to dlsym() these symbols.
 */

BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t      benz_iso_bmff_box_get_size_unsafe (const BENZ_ISO_BOX*     p);
/* Do *NOT* uncomment: BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t      benz_iso_bmff_box_get_size        (const BENZ_ISO_BOX*     p, const void* tail) */
BENZINA_PUBLIC BENZINA_EXTERN_INLINE const void*   benz_iso_bmff_box_get_tail        (const BENZ_ISO_BOX*     p, const void* tail);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE const void*   benz_iso_bmff_box_skip            (const BENZ_ISO_BOX*     p, const void* tail, uint64_t n);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE const char*   benz_iso_bmff_box_get_type_str    (const BENZ_ISO_BOX*     p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t      benz_iso_bmff_box_get_type        (const BENZ_ISO_BOX*     p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int           benz_iso_bmff_box_is_type         (const BENZ_ISO_BOX*     p, const void* type);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE const char*   benz_iso_bmff_box_get_exttype     (const BENZ_ISO_BOX*     p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE const void*   benz_iso_bmff_box_get_payload     (const BENZ_ISO_BOX*     p, uint64_t offset);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t      benz_iso_bmff_box_get_flagver     (const BENZ_ISO_FULLBOX* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t      benz_iso_bmff_box_get_version     (const BENZ_ISO_FULLBOX* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t      benz_iso_bmff_box_get_flags       (const BENZ_ISO_FULLBOX* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void*         benz_iso_bmff_box_put_size32      (BENZ_ISO_BOX*           p, uint32_t size);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void*         benz_iso_bmff_box_put_size64      (BENZ_ISO_BOX*           p, uint64_t size);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void*         benz_iso_bmff_box_put_size        (BENZ_ISO_BOX*           p, uint64_t size);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void*         benz_iso_bmff_box_put_type_str    (BENZ_ISO_BOX*           p, const void* type);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void*         benz_iso_bmff_box_put_type        (BENZ_ISO_BOX*           p, uint32_t type);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE void*         benz_iso_bmff_box_put_flagver     (BENZ_ISO_FULLBOX*       p, uint32_t flagver);

