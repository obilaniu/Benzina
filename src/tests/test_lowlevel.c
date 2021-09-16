/* Includes */
#include <stdio.h>
#include <stdlib.h>
#include <benzina/benzina.h>



/**
 * Defines
 */

#define tstmessageflush(fp, f, ...)               \
    do{                                           \
        fprintf((fp), (f), ## __VA_ARGS__);       \
        fflush((fp));                             \
    }while(0)
#define tstmessagetap(r, f, ...)                  \
    do{                                           \
        if((r)){                                  \
            fprintf(stdout, "ok ");               \
        }else{                                    \
            fprintf(stdout, "not ok ");           \
        }                                         \
        fprintf(stdout, (f), ## __VA_ARGS__);     \
        fprintf(stdout, "\n");                    \
        fflush(stdout);                           \
    }while(0)
#define tstmessage(fp, f, ...)                    \
    (fprintf((fp), (f), ## __VA_ARGS__))
#define tstmessageexit(code, fp, f, ...)          \
    do{                                           \
        tstmessageflush(fp, f, ## __VA_ARGS__);   \
        exit(code);                               \
    }while(0)



/**
 * Functions
 */

void test_symbols(void){
    int r;
    
    r = benz_version() == LIBBENZINA_VERSION_INT(LIBBENZINA_VERSION_MAJOR,
                                                 LIBBENZINA_VERSION_MINOR,
                                                 LIBBENZINA_VERSION_PATCH);
    tstmessagetap(r, "symbol version");
    
    r = strcmp(benz_license(), "MIT") == 0;
    tstmessagetap(r, "symbol license");
    
    r = strcmp(benz_configuration(), "") == 0;
    tstmessagetap(r, "symbol configuration");
}
void test_endian(void){
    uint16_t a16 = 0x0201ULL;
    uint16_t b16 = 0x0102ULL;
    uint32_t a32 = 0x04030201ULL;
    uint32_t b32 = 0x01020304ULL;
    uint64_t a64 = 0x0807060504030201ULL;
    uint64_t b64 = 0x0102030405060708ULL;
    uint8_t  c[8] = {1,2,3,4,5,6,7,8};
    uint8_t  d[8];
    
    tstmessagetap(a16 == benz_bswap16(b16), "bswap16");
    tstmessagetap(a32 == benz_bswap32(b32), "bswap32");
    tstmessagetap(a64 == benz_bswap64(b64), "bswap64");
    
    #if BENZINA_ENDIAN_LITTLE
    tstmessagetap(a16 == benz_le16toh(a16), "benz_le16toh");
    tstmessagetap(a32 == benz_le32toh(a32), "benz_le32toh");
    tstmessagetap(a64 == benz_le64toh(a64), "benz_le64toh");
    
    tstmessagetap(b16 == benz_be16toh(a16), "benz_be16toh");
    tstmessagetap(b32 == benz_be32toh(a32), "benz_be32toh");
    tstmessagetap(b64 == benz_be64toh(a64), "benz_be64toh");
    
    tstmessagetap(a16 == benz_htole16(a16), "benz_htole16");
    tstmessagetap(a32 == benz_htole32(a32), "benz_htole32");
    tstmessagetap(a64 == benz_htole64(a64), "benz_htole64");
    
    tstmessagetap(b16 == benz_htobe16(a16), "benz_htobe16");
    tstmessagetap(b32 == benz_htobe32(a32), "benz_htobe32");
    tstmessagetap(b64 == benz_htobe64(a64), "benz_htobe64");
    #else
    tstmessagetap(b16 == benz_le16toh(a16), "benz_le16toh");
    tstmessagetap(b32 == benz_le32toh(a32), "benz_le32toh");
    tstmessagetap(b64 == benz_le64toh(a64), "benz_le64toh");
    
    tstmessagetap(a16 == benz_be16toh(a16), "benz_be16toh");
    tstmessagetap(a32 == benz_be32toh(a32), "benz_be32toh");
    tstmessagetap(a64 == benz_be64toh(a64), "benz_be64toh");
    
    tstmessagetap(b16 == benz_htole16(a16), "benz_htole16");
    tstmessagetap(b32 == benz_htole32(a32), "benz_htole32");
    tstmessagetap(b64 == benz_htole64(a64), "benz_htole64");
    
    tstmessagetap(a16 == benz_htobe16(a16), "benz_htobe16");
    tstmessagetap(a32 == benz_htobe32(a32), "benz_htobe32");
    tstmessagetap(a64 == benz_htobe64(a64), "benz_htobe64");
    #endif
    
    tstmessagetap(0x1 == benz_getle8 (c,0), "getle8");
    tstmessagetap(a16 == benz_getle16(c,0), "getle16");
    tstmessagetap(a32 == benz_getle32(c,0), "getle32");
    tstmessagetap(a64 == benz_getle64(c,0), "getle64");
    
    tstmessagetap(0x1 == benz_getbe8 (c,0), "getbe8");
    tstmessagetap(b16 == benz_getbe16(c,0), "getbe16");
    tstmessagetap(b32 == benz_getbe32(c,0), "getbe32");
    tstmessagetap(b64 == benz_getbe64(c,0), "getbe64");
    
    tstmessagetap(0xF == benz_putle8 (d,0,0xF) && *d == 0xF,                "putle8");
    tstmessagetap(a16 == benz_putle16(d,0,a16) && benz_getle16(d,0) == a16, "putle16");
    tstmessagetap(a32 == benz_putle32(d,0,a32) && benz_getle32(d,0) == a32, "putle32");
    tstmessagetap(a64 == benz_putle64(d,0,a64) && benz_getle64(d,0) == a64, "putle64");
    
    tstmessagetap(0xF == benz_putbe8 (d,0,0xF) && *d == 0xF,                "putbe8");
    tstmessagetap(a16 == benz_putbe16(d,0,a16) && benz_getbe16(d,0) == a16, "putbe16");
    tstmessagetap(a32 == benz_putbe32(d,0,a32) && benz_getbe32(d,0) == a32, "putbe32");
    tstmessagetap(a64 == benz_putbe64(d,0,a64) && benz_getbe64(d,0) == a64, "putbe64");
}
void test_intops(void){
    tstmessagetap(benz_popcnt64(0xFFFFFFFFFFFFFFFFULL) == 64, "popcnt 1");
    tstmessagetap(benz_popcnt64(0x0000000000000000ULL) ==  0, "popcnt 2");
    tstmessagetap(benz_popcnt64(0xDEADBEEFCAFEC001ULL) == 38, "popcnt 3");
    
    tstmessagetap(benz_clz64(0xFFFFFFFFFFFFFFFFULL) ==  0, "clz 1");
    tstmessagetap(benz_clz64(0x0000000000000000ULL) == 64, "clz 2");
    tstmessagetap(benz_clz64(0x0000000000000001ULL) == 63, "clz 3");
    tstmessagetap(benz_clz64(0x0000000000000002ULL) == 62, "clz 4");
    tstmessagetap(benz_clz64(0x003AE3F0F80F05F0ULL) == 10, "clz 5");
    
    tstmessagetap(benz_ctz64(0xFFFFFFFFFFFFFFFFULL) ==  0, "ctz 1");
    tstmessagetap(benz_ctz64(0x0000000000000000ULL) == 64, "ctz 2");
    tstmessagetap(benz_ctz64(0x8000000000000000ULL) == 63, "ctz 3");
    tstmessagetap(benz_ctz64(0x4000000000000000ULL) == 62, "ctz 4");
    tstmessagetap(benz_ctz64(0x003AE3F0F80F05F0ULL) ==  4, "ctz 5");
    
    tstmessagetap(benz_ueto64(0x8000000000000000ULL) == 0, "ue 1");
    tstmessagetap(benz_ueto64(0xFFFFFFFFFFFFFFFFULL) == 0, "ue 2");
    tstmessagetap(benz_ueto64(0x4000000000000000ULL) == 1, "ue 3");
    tstmessagetap(benz_ueto64(0x5000000000000000ULL) == 1, "ue 4");
    tstmessagetap(benz_ueto64(0x5FFFFFFFFFFFFFFFULL) == 1, "ue 5");
    tstmessagetap(benz_ueto64(0x6000000000000000ULL) == 2, "ue 6");
    tstmessagetap(benz_ueto64(0x2000000000000000ULL) == 3, "ue 7");
    tstmessagetap(benz_ueto64(0x2800000000000000ULL) == 4, "ue 8");
    tstmessagetap(benz_ueto64(0x3000000000000000ULL) == 5, "ue 9");
    tstmessagetap(benz_ueto64(0x3800000000000000ULL) == 6, "ue 10");
    
    tstmessagetap(benz_seto64(0x8000000000000000ULL) == 0, "se 1");
    tstmessagetap(benz_seto64(0x4000000000000000ULL) ==+1, "se 2");
    tstmessagetap(benz_seto64(0x6000000000000000ULL) ==-1, "se 3");
    tstmessagetap(benz_seto64(0x2000000000000000ULL) ==+2, "se 4");
    tstmessagetap(benz_seto64(0x2800000000000000ULL) ==-2, "se 5");
}


/**
 * Main
 */

int main(void){
    tstmessageflush(stdout, "1..62\n");
    test_symbols();
    test_endian();
    test_intops();
}
