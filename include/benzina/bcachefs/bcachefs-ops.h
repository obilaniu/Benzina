/* Include Guard */
#ifndef INCLUDE_BENZINA_BCACHEFS_BCACHEFS_OPS_H
#define INCLUDE_BENZINA_BCACHEFS_BCACHEFS_OPS_H


/**
 * Includes
 */

#include "benzina/visibility.h"
#include "benzina/bcachefs/bcachefs-types.h"


/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif



/* Data Structures Forward Declarations and Typedefs */
typedef struct bch_inode_unpacked       bch_inode_unpacked;



/* Data Structure Definitions */

/**
 * @brief BcacheFS inode struct, unpacked.
 * 
 * This structure's layout is not normative!
 */

struct bch_inode_unpacked{
    /*          UNPACKED FORM             |  PACKED/ENCODED FORM  */
    /* Header (from struct bch_inode{}) */
    bch_u64 bi_hash_seed;              /* |  64-bit little-endian */
    bch_u32 bi_flags;                  /* |  32-bit little-endian */
    bch_u16 bi_mode;                   /* |  16-bit little-endian */
    bch_u16 _pad_0;
    
    /* Body */
    bch_u64 bi_atime, bi_atime_hi;     /* |  96-bit varint        */
    bch_u64 bi_ctime, bi_ctime_hi;     /* |  96-bit varint        */
    bch_u64 bi_mtime, bi_mtime_hi;     /* |  96-bit varint        */
    bch_u64 bi_otime, bi_otime_hi;     /* |  96-bit varint        */
    bch_u64 bi_size;                   /* |  64-bit varint        */
    bch_u64 bi_sectors;                /* |  64-bit varint        */
    bch_u32 bi_uid;                    /* |  32-bit varint        */
    bch_u32 bi_gid;                    /* |  32-bit varint        */
    bch_u32 bi_nlink;                  /* |  32-bit varint        */
    bch_u32 bi_generation;             /* |  32-bit varint        */
    bch_u32 bi_dev;                    /* |  32-bit varint        */
    bch_u8  bi_data_checksum;          /* |   8-bit varint        */
    bch_u8  bi_compression;            /* |   8-bit varint        */
    bch_u16 _pad_1;
    bch_u32 bi_project;                /* |  32-bit varint        */
    bch_u8  bi_background_compression; /* |   8-bit varint        */
    bch_u8  bi_data_replicas;          /* |   8-bit varint        */
    bch_u16 bi_promote_target;         /* |  16-bit varint        */
    bch_u16 bi_foreground_target;      /* |  16-bit varint        */
    bch_u16 bi_background_target;      /* |  16-bit varint        */
    bch_u16 bi_erasure_code;           /* |  16-bit varint        */
    bch_u16 bi_fields_set;             /* |  16-bit varint        */
    bch_u64 bi_dir;                    /* |  64-bit varint        */
    bch_u64 bi_dir_offset;             /* |  64-bit varint        */
    bch_u32 bi_subvol;                 /* |  32-bit varint        */
    bch_u32 bi_parent_subvol;          /* |  32-bit varint        */
};


/**
 * Public Function Definitions for BcacheFS support.
 */

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
                                                const void*         end);

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

BENZINA_PUBLIC int   benz_bch_inode_unpack_size(bch_u64*            bi_size,
                                                const bch_inode*    p,
                                                const void*         end);


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif

