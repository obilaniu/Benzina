/* Includes */
#include "benzina/iso.h"



/* Defines */



/* Function Definitions */

/**
 * Materialize extern versions of ISO BMFF support inline functions from the
 * header, in case:
 * 
 *   - Inlining somehow failed or
 *   - We want to dlsym() these symbols.
 */


BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint8_t     benz_iso_as_u8                   (const char* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint16_t    benz_iso_as_u16                  (const char* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t    benz_iso_as_u32                  (const char* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_iso_as_u64                  (const char* p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_iso_is_type_equal           (const char* p, const char* type);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_iso_box_get_size            (BENZ_BOX    p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_iso_box_get_size_bounded    (BENZ_BOX    p, uint64_t max_size);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE const char* benz_iso_box_get_type_str        (BENZ_BOX    p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t    benz_iso_box_get_type            (BENZ_BOX    p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_iso_box_is_type             (BENZ_BOX    p, const char* type);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE const char* benz_iso_box_get_exttype         (BENZ_BOX    p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE const char* benz_iso_box_get_payload         (BENZ_BOX    p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_iso_fullbox_get_size        (BENZ_BOX    p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_iso_fullbox_get_size_bounded(BENZ_BOX    p, uint64_t max_size);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE const char* benz_iso_fullbox_get_type_str    (BENZ_BOX    p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t    benz_iso_fullbox_get_type        (BENZ_BOX    p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE int         benz_iso_fullbox_is_type         (BENZ_BOX    p, const char* type);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE const char* benz_iso_fullbox_get_exttype     (BENZ_BOX    p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t    benz_iso_fullbox_get_flagver     (BENZ_BOX    p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t    benz_iso_fullbox_get_version     (BENZ_BOX    p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t    benz_iso_fullbox_get_flags       (BENZ_BOX    p);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE const char* benz_iso_fullbox_get_payload     (BENZ_BOX    p);

