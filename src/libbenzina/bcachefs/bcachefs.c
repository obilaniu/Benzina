/* Includes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "benzina/endian.h"
#include "benzina/intops.h"
#include "benzina/ptrops.h"
#include "benzina/siphash.h"
#include "benzina/bcachefs/bcachefs.h"
#include "internal.h"



/* Public Function Definitions */

/**
 * @brief Unpack a BcacheFS inode.
 * 
 * The input struct bch_inode has packed varint values in little-endian, while
 * the output struct bch_inode_unpacked is native-endian.
 * 
 * @param [out] u    Pointer to      unpacked inode struct.
 * @param [in]  p    Pointer to        packed inode struct.
 * @param [in]  end  Pointer to end of packed inode struct. Cannot be NULL.
 * 
 * @return The number of fields successfully parsed, else <0.
 */

BENZINA_PUBLIC int   benz_bch_inode_unpack     (bch_inode_unpacked* u,
                                                const bch_inode*    p,
                                                const void*         end){
    register unsigned oo, o;
    register int      new_varint, nr_fields;
    register int      varintc;
    register uint64_t varintw, f;
    const bch_u8* e = (const bch_u8*)end;
    const bch_u8* r = (const bch_u8*)&p->fields;
    bch_u8*       w = (      bch_u8*)&u->bi_atime;
    
    memset(u, 0, sizeof(*u));
    if(e<r)
        return -1;/* Parse error, end pointer behind field pointer! */
    
    u->bi_hash_seed = benz_getle64(&p->bi_hash_seed, 0);
    u->bi_flags     = benz_getle32(&p->bi_flags,     0);
    u->bi_mode      = benz_getle16(&p->bi_mode,      0);
    nr_fields       = (int)(u->bi_flags >> 24) & 127;
    new_varint      = !!(u->bi_flags & BCH_INODE_FLAG_new_varint);
    
    if(!new_varint)
        return -1;/* Parse error, old-style varint! */
    if(e-r < (ptrdiff_t)nr_fields)
        return -1;/* Parse error, end pointer far too short! At least 1 byte/field. */
    
    
    /**
     * Decode varint fields sequentially into respective struct member.
     * Wide fields are encoded as two varints.
     * 
     *   varintw  is a bitstring where bit-groups
     *     - `00` indicate naturally-aligned bch_u8  output
     *     - `01` indicate naturally-aligned bch_u16 output
     *     - `10` indicate naturally-aligned bch_u32 output
     *     - `11` indicate naturally-aligned bch_u64 output
     * 
     * We use this trick because expanding out the inode unpack using macros
     * produces far more code, making it likely an icache miss will occur.
     * 
     * WARNING: 29 out of 32 possible two-bit bit-groups are now used!
     */
    
    nr_fields = nr_fields<=25 ? nr_fields : 25;
    varintc = nr_fields<4 ? nr_fields*2 : nr_fields+4;
    varintw = 0b1010111101010101010000100000101010101011111111111111111111;
    for(o=0; varintc--; varintw>>=2){
        f  = benz_ctz64(*r+1)+1;
        r += f;
        if(luai_unlikely(r>e))
            return -1;
        
        f *= 6;
        f &= 0x3F;/* Can be elided on x86_64 */
        /* For field length:   9  8  7  6  5  4  3  2  1    */
        /* Shift right by: --  0  8 15 22 29 36 43 50 57 -- */
        f  = 000101726354453627100 >> f;
        f &= 0x3F;/* Can be elided on x86_64 */
        f  = benz_getle64(r,-8) >> f;
        
        oo = 1U << (varintw&3);
        o  = (o+oo-1)&-oo;/* Naturally-aligned to kk = 2^k */
        switch(oo){
            case 8:  *(bch_u64*)&w[o] = f; break;
            case 4:  *(bch_u32*)&w[o] = f; break;
            case 2:  *(bch_u16*)&w[o] = f; break;
            case 1:
            default: *(bch_u8* )&w[o] = f; break;
        }
        o += oo;
    }
    
    return nr_fields;
}

/**
 * @brief Unpack a BcacheFS inode's size field.
 * 
 * The input struct bch_inode has packed varint values in little-endian, while
 * the output integer is native-endian.
 * 
 * This function is faster than a full decode because it skips other fields
 * before decoding the inode's size field.
 * 
 * @param [out] bi_size  Pointer to 64-bit integer.
 * @param [in]  p        Pointer to        packed inode struct.
 * @param [in]  end      Pointer to end of packed inode struct. Cannot be NULL.
 * 
 * @return 0 if successfully parsed, else <0.
 */

/* NOTE: Step down to -O2 because at -O3 the loop gets unrolled unconditionally! */
#pragma GCC push_options
#pragma GCC optimize ("O2")
BENZINA_PUBLIC int   benz_bch_inode_unpack_size(bch_u64*            bi_size,
                                                const bch_inode*    p,
                                                const void*         end){
    register int      new_varint, nr_fields;
    register int      varintc;
    register uint32_t bi_flags;
    register uint64_t f;
    const bch_u8* e = (const bch_u8*)end;
    const bch_u8* r = (const bch_u8*)&p->fields;
    
    *bi_size = 0;/* Default is 0. */
    if(e<r)
        return -1;/* Parse error, end pointer behind field pointer! */
    
    bi_flags   = benz_getle32(&p->bi_flags, 0);
    nr_fields  = (int)(bi_flags >> 24) & 127;
    new_varint = !!(bi_flags & BCH_INODE_FLAG_new_varint);
    
    if(!new_varint)
        return -1;/* Parse error, old-style varint! */
    if(e-r < (ptrdiff_t)nr_fields)
        return -1;/* Parse error, end pointer far too short! At least 1 byte/field. */
    
    
    /**
     * The field bi_size is the 5th field and 9th varint in a v2-packed inode,
     * being preceded by four wide (double-varint) fields (the 96-bit timestamps).
     * 
     * Accordingly, check that the number of fields is at least 5, and if so
     * then scan up to the 9th varint.
     */
    
    if(nr_fields < 5)
        return  0;/* No size field encoded, default is 0. */
    
    for(varintc=0; varintc<9; varintc++){
        f  = benz_ctz64(*r+1)+1;
        r += f;
        if(luai_unlikely(r>e))
            return -1;
    }
    
    /**
     * Pointer r now points one byte past the end of the target varint.
     * Decode varint at current location.
     */
    
    f *= 6;
    f &= 0x3F;/* Can be elided on x86_64 */
    /* For field length:   9  8  7  6  5  4  3  2  1    */
    /* Shift right by: --  0  8 15 22 29 36 43 50 57 -- */
    f  = 000101726354453627100 >> f;
    f &= 0x3F;/* Can be elided on x86_64 */
    f  = benz_getle64(r,-8) >> f;
    *bi_size = f;
    
    return 0;
}
#pragma GCC pop_options



/* Lua bindings for BcacheFS */

/**
 * @brief Unpack BcacheFS inode struct.
 * 
 *   - inode_unpack(packed_inode):
 */

BENZINA_STATIC int benz_lua_bch_inode_unpack(lua_State* L){
    size_t len;
    const char* p;
    bch_inode_unpacked u;
    
    if(!(p=lua_tolstring(L, 1, &len)) || len<14)
        return luaL_error(L, "The packed inode must be a string at least 14 bytes long!");
    
    if(benz_bch_inode_unpack(&u, (const bch_inode*)p, p+len) < 0)
        return luaL_error(L, "Failed to unpack inode!");
    
    
    #define INODE_FIELDS()                               \
            INODE_FIELD_WIDE(bi_atime)                   \
            INODE_FIELD_WIDE(bi_ctime)                   \
            INODE_FIELD_WIDE(bi_mtime)                   \
            INODE_FIELD_WIDE(bi_otime)                   \
            INODE_FIELD     (bi_size)                    \
            INODE_FIELD     (bi_sectors)                 \
            INODE_FIELD     (bi_uid)                     \
            INODE_FIELD     (bi_gid)                     \
            INODE_FIELD     (bi_nlink)                   \
            INODE_FIELD     (bi_generation)              \
            INODE_FIELD     (bi_dev)                     \
            INODE_FIELD     (bi_data_checksum)           \
            INODE_FIELD     (bi_compression)             \
            INODE_FIELD     (bi_project)                 \
            INODE_FIELD     (bi_background_compression)  \
            INODE_FIELD     (bi_data_replicas)           \
            INODE_FIELD     (bi_promote_target)          \
            INODE_FIELD     (bi_foreground_target)       \
            INODE_FIELD     (bi_background_target)       \
            INODE_FIELD     (bi_erasure_code)            \
            INODE_FIELD     (bi_fields_set)              \
            INODE_FIELD     (bi_dir)                     \
            INODE_FIELD     (bi_dir_offset)              \
            INODE_FIELD     (bi_subvol)                  \
            INODE_FIELD     (bi_parent_subvol)           \
    
    #define INODE_FIELD(name)      +1
    #define INODE_FIELD_WIDE(name) +2
    
    lua_createtable(L, 0, +3+INODE_FIELDS());
    
    #undef  INODE_FIELD_WIDE
    #undef  INODE_FIELD
    
    
    #define INODE_FIELD(name)                          \
        do{                                            \
            lua_pushliteral(L, __BENZ_QUOTEME(name));  \
            lua_pushinteger(L, u.name);                \
            lua_settable   (L, 2);                     \
        }while(0);
    #define INODE_FIELD_WIDE(name)  \
        INODE_FIELD(name); INODE_FIELD(name ## _hi);
    
    INODE_FIELD(bi_hash_seed);
    INODE_FIELD(bi_flags);
    INODE_FIELD(bi_mode);
    INODE_FIELDS();
    
    #undef  INODE_FIELD_WIDE
    #undef  INODE_FIELD
    
    #undef  INODE_FIELDS
    
    
    return 1;
}

/**
 * @brief Unpack BcacheFS inode struct's bi_size field.
 * 
 *   - inode_unpack_size(packed_inode):
 */

BENZINA_STATIC int benz_lua_bch_inode_unpack_size(lua_State* L){
    size_t len;
    const char* p;
    bch_u64 bi_size;
    
    if(!(p=lua_tolstring(L, 1, &len)) || len<14)
        return luaL_error(L, "The packed inode must be a string at least 14 bytes long!");
    
    if(benz_bch_inode_unpack_size(&bi_size, (const bch_inode*)p, p+len) < 0)
        return luaL_error(L, "Failed to unpack inode bi_size!");
    
    lua_pushinteger(L, bi_size);
    return 1;
}

BENZINA_STATIC const luaL_Reg bcachefslib[] = {
    {"inode_unpack",      benz_lua_bch_inode_unpack},
    {"inode_unpack_size", benz_lua_bch_inode_unpack_size},
    {NULL,           NULL}
};
BENZINA_STATIC int luaopen_benzina_bcachefs(lua_State* L){
  luaL_newlib(L, bcachefslib);  /* new module */
  return 1;
}
BENZINA_LUAOPEN_REGISTER("benzina.bcachefs", luaopen_benzina_bcachefs)
