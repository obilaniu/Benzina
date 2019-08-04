/* Includes */
#include "benzina/endian.h"


/* Defines */



/* Function Definitions */

/**
 * Materialize extern versions of endianness-support inline functions from the
 * header, in case:
 * 
 *   - Inlining somehow failed or
 *   - We want to dlsym() these symbols.
 */

BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint16_t benz_bswap16(uint16_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t benz_bswap32(uint32_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t benz_bswap64(uint64_t x);

BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint16_t benz_htobe16(uint16_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t benz_htobe32(uint32_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t benz_htobe64(uint64_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint16_t benz_htole16(uint16_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t benz_htole32(uint32_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t benz_htole64(uint64_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint16_t benz_be16toh(uint16_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t benz_be32toh(uint32_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t benz_be64toh(uint64_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint16_t benz_le16toh(uint16_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t benz_le32toh(uint32_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t benz_le64toh(uint64_t x);

