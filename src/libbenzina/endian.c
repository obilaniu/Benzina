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

BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint8_t  benz_getbe8 (const void* p, int64_t off);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint16_t benz_getbe16(const void* p, int64_t off);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t benz_getbe32(const void* p, int64_t off);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t benz_getbe64(const void* p, int64_t off);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint8_t  benz_getle8 (const void* p, int64_t off);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint16_t benz_getle16(const void* p, int64_t off);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t benz_getle32(const void* p, int64_t off);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t benz_getle64(const void* p, int64_t off);

BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint8_t  benz_putbe8 (void* p, int64_t off, uint8_t  x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint16_t benz_putbe16(void* p, int64_t off, uint16_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t benz_putbe32(void* p, int64_t off, uint32_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t benz_putbe64(void* p, int64_t off, uint64_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint8_t  benz_putle8 (void* p, int64_t off, uint8_t  x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint16_t benz_putle16(void* p, int64_t off, uint16_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint32_t benz_putle32(void* p, int64_t off, uint32_t x);
BENZINA_PUBLIC BENZINA_EXTERN_INLINE uint64_t benz_putle64(void* p, int64_t off, uint64_t x);

