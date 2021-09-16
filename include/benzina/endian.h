/* Include Guard */
#ifndef INCLUDE_BENZINA_ENDIAN_H
#define INCLUDE_BENZINA_ENDIAN_H


/**
 * Includes
 */

#include <stdint.h>
#include <string.h>
#include "benzina/config.h"
#include "benzina/inline.h"
#include "benzina/visibility.h"


/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif


/* Temporary defines...  */
#undef  BENZINA_PUBLIC_ENDIANOP
#undef  BENZINA_PUBLIC_ENDIANOP_CONST
#define BENZINA_PUBLIC_ENDIANOP        BENZINA_ATTRIBUTE_ALWAYSINLINE BENZINA_PUBLIC BENZINA_INLINE
#define BENZINA_PUBLIC_ENDIANOP_CONST  BENZINA_PUBLIC_ENDIANOP BENZINA_ATTRIBUTE_CONST

#undef  BENZ_PICK_BE_LE
#if BENZINA_BIG_ENDIAN
# define BENZ_PICK_BE_LE(A,B) A
#else
# define BENZ_PICK_BE_LE(A,B) B
#endif


/**
 * Inline Function Definitions for Endianness Support
 */

#if __GNUC__
BENZINA_PUBLIC_ENDIANOP_CONST uint16_t benz_bswap16(uint16_t x){return __builtin_bswap16(x);}
BENZINA_PUBLIC_ENDIANOP_CONST uint32_t benz_bswap32(uint32_t x){return __builtin_bswap32(x);}
BENZINA_PUBLIC_ENDIANOP_CONST uint64_t benz_bswap64(uint64_t x){return __builtin_bswap64(x);}
#else
BENZINA_PUBLIC_ENDIANOP_CONST uint16_t benz_bswap16(uint16_t x){return                     x <<  8 |              x >>  8;      }
BENZINA_PUBLIC_ENDIANOP_CONST uint32_t benz_bswap32(uint32_t x){return benz_bswap16(x >>  0) << 16 | benz_bswap16(x >> 16) << 0;}
BENZINA_PUBLIC_ENDIANOP_CONST uint64_t benz_bswap64(uint64_t x){return benz_bswap32(x >>  0) << 32 | benz_bswap32(x >> 32) << 0;}
#endif

BENZINA_PUBLIC_ENDIANOP_CONST uint16_t benz_htobe16(uint16_t x){return BENZ_PICK_BE_LE(x, benz_bswap16(x));}
BENZINA_PUBLIC_ENDIANOP_CONST uint32_t benz_htobe32(uint32_t x){return BENZ_PICK_BE_LE(x, benz_bswap32(x));}
BENZINA_PUBLIC_ENDIANOP_CONST uint64_t benz_htobe64(uint64_t x){return BENZ_PICK_BE_LE(x, benz_bswap64(x));}
BENZINA_PUBLIC_ENDIANOP_CONST uint16_t benz_htole16(uint16_t x){return BENZ_PICK_BE_LE(benz_bswap16(x), x);}
BENZINA_PUBLIC_ENDIANOP_CONST uint32_t benz_htole32(uint32_t x){return BENZ_PICK_BE_LE(benz_bswap32(x), x);}
BENZINA_PUBLIC_ENDIANOP_CONST uint64_t benz_htole64(uint64_t x){return BENZ_PICK_BE_LE(benz_bswap64(x), x);}
BENZINA_PUBLIC_ENDIANOP_CONST uint16_t benz_be16toh(uint16_t x){return BENZ_PICK_BE_LE(x, benz_bswap16(x));}
BENZINA_PUBLIC_ENDIANOP_CONST uint32_t benz_be32toh(uint32_t x){return BENZ_PICK_BE_LE(x, benz_bswap32(x));}
BENZINA_PUBLIC_ENDIANOP_CONST uint64_t benz_be64toh(uint64_t x){return BENZ_PICK_BE_LE(x, benz_bswap64(x));}
BENZINA_PUBLIC_ENDIANOP_CONST uint16_t benz_le16toh(uint16_t x){return BENZ_PICK_BE_LE(benz_bswap16(x), x);}
BENZINA_PUBLIC_ENDIANOP_CONST uint32_t benz_le32toh(uint32_t x){return BENZ_PICK_BE_LE(benz_bswap32(x), x);}
BENZINA_PUBLIC_ENDIANOP_CONST uint64_t benz_le64toh(uint64_t x){return BENZ_PICK_BE_LE(benz_bswap64(x), x);}

BENZINA_PUBLIC_ENDIANOP       uint8_t  benz_getbe8 (const void* p, int64_t off){
    uint8_t x;
    memcpy(&x, (char*)p+off, sizeof(x));
    return x;
}
BENZINA_PUBLIC_ENDIANOP       uint16_t benz_getbe16(const void* p, int64_t off){
    uint16_t x;
    memcpy(&x, (char*)p+off, sizeof(x));
    return benz_be16toh(x);
}
BENZINA_PUBLIC_ENDIANOP       uint32_t benz_getbe32(const void* p, int64_t off){
    uint32_t x;
    memcpy(&x, (char*)p+off, sizeof(x));
    return benz_be32toh(x);
}
BENZINA_PUBLIC_ENDIANOP       uint64_t benz_getbe64(const void* p, int64_t off){
    uint64_t x;
    memcpy(&x, (char*)p+off, sizeof(x));
    return benz_be64toh(x);
}
BENZINA_PUBLIC_ENDIANOP       uint8_t  benz_getle8 (const void* p, int64_t off){
    uint8_t x;
    memcpy(&x, (char*)p+off, sizeof(x));
    return x;
}
BENZINA_PUBLIC_ENDIANOP       uint16_t benz_getle16(const void* p, int64_t off){
    uint16_t x;
    memcpy(&x, (char*)p+off, sizeof(x));
    return benz_le16toh(x);
}
BENZINA_PUBLIC_ENDIANOP       uint32_t benz_getle32(const void* p, int64_t off){
    uint32_t x;
    memcpy(&x, (char*)p+off, sizeof(x));
    return benz_le32toh(x);
}
BENZINA_PUBLIC_ENDIANOP       uint64_t benz_getle64(const void* p, int64_t off){
    uint64_t x;
    memcpy(&x, (char*)p+off, sizeof(x));
    return benz_le64toh(x);
}

BENZINA_PUBLIC_ENDIANOP       uint8_t  benz_putbe8 (void* p, int64_t off, uint8_t  x){
    memcpy((char*)p+off, &x, sizeof(x));
    return x;
}
BENZINA_PUBLIC_ENDIANOP       uint16_t benz_putbe16(void* p, int64_t off, uint16_t x){
    uint16_t y = benz_htobe16(x);
    memcpy((char*)p+off, &y, sizeof(y));
    return x;
}
BENZINA_PUBLIC_ENDIANOP       uint32_t benz_putbe32(void* p, int64_t off, uint32_t x){
    uint32_t y = benz_htobe32(x);
    memcpy((char*)p+off, &y, sizeof(y));
    return x;
}
BENZINA_PUBLIC_ENDIANOP       uint64_t benz_putbe64(void* p, int64_t off, uint64_t x){
    uint64_t y = benz_htobe64(x);
    memcpy((char*)p+off, &y, sizeof(y));
    return x;
}
BENZINA_PUBLIC_ENDIANOP       uint8_t  benz_putle8 (void* p, int64_t off, uint8_t  x){
    memcpy((char*)p+off, &x, sizeof(x));
    return x;
}
BENZINA_PUBLIC_ENDIANOP       uint16_t benz_putle16(void* p, int64_t off, uint16_t x){
    uint16_t y = benz_htole16(x);
    memcpy((char*)p+off, &y, sizeof(y));
    return x;
}
BENZINA_PUBLIC_ENDIANOP       uint32_t benz_putle32(void* p, int64_t off, uint32_t x){
    uint32_t y = benz_htole32(x);
    memcpy((char*)p+off, &y, sizeof(y));
    return x;
}
BENZINA_PUBLIC_ENDIANOP       uint64_t benz_putle64(void* p, int64_t off, uint64_t x){
    uint64_t y = benz_htole64(x);
    memcpy((char*)p+off, &y, sizeof(y));
    return x;
}


/* Undo temporary defines */
#undef  BENZINA_PUBLIC_ENDIANOP
#undef  BENZINA_PUBLIC_ENDIANOP_CONST
#undef  BENZ_PICK_BE_LE


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

