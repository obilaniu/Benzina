/* Includes */
#include "benzina/endian.h"
#include "benzina/intops.h"
#include "benzina/ptrops.h"
#include "benzina/siphash.h"
#include "internal.h"



/* Static Function Definitions */
BENZINA_STATIC inline void benz_siphash_init(uint64_t v[4], uint64_t k0, uint64_t k1){
    v[0] = k0 ^ 0x736f6d6570736575; /* "somepseu" */
    v[1] = k1 ^ 0x646f72616e646f6d; /* "dorandom" */
    v[2] = k0 ^ 0x6c7967656e657261; /* "lygenera" */
    v[3] = k1 ^ 0x7465646279746573; /* "tedbytes" */
}
BENZINA_STATIC inline void benz_siphash_rounds(uint64_t v[4], uint64_t m, unsigned r){
    uint64_t v0 = v[0];
    uint64_t v1 = v[1];
    uint64_t v2 = v[2];
    uint64_t v3 = v[3];
    
    v3 ^= m;
    
    while(r--){
        #define rol(v, c) v = benz_rol64(v, c)
        /*   V0    ;     V1    ;     V2    ;     V3    ; */
        /* Half-round 1/2 start { v0&v1 | v2&v3 } */
        v0 += v1   ;           ; v2 += v3  ;           ;
                   ;rol(v1, 13);           ;rol(v3, 16);
                   ;v1 ^= v0   ;           ;v3 ^= v2   ;
        rol(v0, 32);           ;           ;           ;
        /* Half-round 1/2 end   { v0&v1 | v2&v3 } */
        /* Half-round 2/2 start { v0&v3 | v1&v2 } */
        v0 += v3   ;           ; v2 += v1  ;           ;
                   ;rol(v1, 17);           ;rol(v3, 21);
                   ;v1 ^= v2   ;           ;v3 ^= v0   ;
                   ;           ;rol(v2, 32);           ;
        /* Half-round 2/2 end   { v0&v3 | v1&v2 } */
        /*   V0    ;     V1    ;     V2    ;     V3    ; */
        #undef  rol
    }
    
    v0 ^= m;
    
    v[0] = v0;
    v[1] = v1;
    v[2] = v2;
    v[3] = v3;
}
BENZINA_STATIC inline uint64_t benz_siphash_padword(const void* buf, uint64_t len){
    uint64_t m=0;
    memcpy(&m, benz_ptr_addcv(buf, len&~7), len&7);
    return benz_getle64(&m, 0) | len<<56;
}
BENZINA_STATIC inline uint64_t benz_siphash_finalize(uint64_t v[4], unsigned d){
    v[2] ^= 0xFFU;
    benz_siphash_rounds(v, 0, d);
    return v[0]^v[1]^v[2]^v[3];
}
BENZINA_STATIC inline uint64_t benz_siphash_digest_internal(const void* buf, uint64_t len,
                                                            uint64_t    k0,  uint64_t k1,
                                                            unsigned    c,   unsigned d){
    uint64_t l;
    uint64_t v[4];
    benz_siphash_init(v, k0, k1);
    for(l=0;l+7<len;l+=8)
        benz_siphash_rounds(v, benz_getle64(buf, l), c);
    benz_siphash_rounds(v, benz_siphash_padword(buf, len), c);
    return benz_siphash_finalize(v, d);
}



/* Public Function Definitions */
BENZINA_PUBLIC uint64_t benz_siphash_digestx(const void* buf, uint64_t len,
                                             uint64_t    k0,  uint64_t k1,
                                             unsigned    c,   unsigned d){
    return benz_siphash_digest_internal(buf, len, k0, k1, c, d);
}
BENZINA_PUBLIC uint64_t benz_siphash_digest (const void* buf, uint64_t len,
                                             uint64_t    k0,  uint64_t k1){
    return benz_siphash_digest_internal(buf, len, k0, k1, 2, 4);
}



/* Lua bindings for SipHash */

/**
 * @brief SipHash-c-d of a Lua string with a given key and round parameters.
 * 
 *   - siphash(data, {k | k0,k1} [, c=2 [, d=4]]):
 */

BENZINA_STATIC int benz_lua_siphash(lua_State* L){
    int         a=1;
    unsigned    c=2, d=4;
    const char* s, *k;
    size_t      len, klen;
    uint64_t    k0,  k1;
    
    
    /* Mandatory arguments: data and key. */
    if(!(s=lua_tolstring(L, a++, &len)))
        luaL_error(L, "The value to be hashed, `data`, must be a string!");
    
    if(lua_isinteger(L, a)){
        if(!lua_isinteger(L, a+1))
            return luaL_error(L, "If the key is provided in halves, then both "
                                 "the first half `k0` and the second half `k1` "
                                 "must be of integer type!");
        k0 = lua_tointeger(L, a++);
        k1 = lua_tointeger(L, a++);
    }else if((k=lua_tolstring(L, a, &klen))){
        if(klen != 16)
            return luaL_error(L, "The key `k` must be a 16-byte string!");
        a++;
        k0 = benz_getle64(k, 0);
        k1 = benz_getle64(k, 8);
    }else
        return luaL_error(L, "The key must be supplied as either two integers "
                             "or a 16-byte string!");
    
    
    /* Optional arguments: round counts c and d. */
    if(!lua_isnone(L, a)){
        if(!lua_isinteger(L, a))
            return luaL_error(L, "The compression round count `c` must be an "
                                 "integer! (default=2).");
        c = lua_tointeger(L, a++);
        if(!lua_isnone(L, a)){
            if(!lua_isinteger(L, a))
                return luaL_error(L, "The finalization round count `d` must "
                                     "be an integer! (default=4).");
            d = lua_tointeger(L, a++);
        }
    }
    
    
    /* Execute hashing operation and return 1 value on the stack. */
    lua_pushinteger(L, benz_siphash_digest_internal(s, len, k0, k1, c, d));
    return 1;
}

BENZINA_STATIC const luaL_Reg siphashlib[] = {
  {"siphash", benz_lua_siphash},
  {NULL,      NULL}
};
BENZINA_STATIC int luaopen_benzina_siphash(lua_State* L){
    luaL_newlib(L, siphashlib);
    return 1;
}
BENZINA_LUAOPEN_REGISTER("benzina.siphash", luaopen_benzina_siphash);
