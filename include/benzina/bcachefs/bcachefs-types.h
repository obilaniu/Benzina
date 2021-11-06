/* Include Guard */
#ifndef INCLUDE_BENZINA_BCACHEFS_BCACHEFS_TYPES_H
#define INCLUDE_BENZINA_BCACHEFS_BCACHEFS_TYPES_H


/**
 * Includes
 */

#include "benzina/endian.h"
#include "benzina/inline.h"
#include "benzina/visibility.h"




/* Extern "C" Guard */
#ifdef __cplusplus
extern "C" {
#endif



/* Defines */
#define BENZINA_ATTRIBUTE_PACKED_ALIGNED(n)   \
        BENZINA_ATTRIBUTE_PACKED              \
        BENZINA_ATTRIBUTE_ALIGNED(n)
#if __GNUC__ && __STDC_VERSION__ >= 199901L
# define BENZINA_DO_PRAGMA_(s) _Pragma(#s)
# define BENZINA_DO_PRAGMA(s) BENZINA_DO_PRAGMA_(s)
# define BENZINA_PRAGMA_POISON(name) BENZINA_DO_PRAGMA(GCC poison name)
#else
# define BENZINA_PRAGMA_POISON(name)
#endif

#define BCH_MIN_NR_BUCKETS                      (1 << 6)
#define BCH_JOURNAL_MIN_NR_BUCKETS                    8
#define BCH_SB_LAYOUT_SECTOR                          7
#define BCH_SB_SECTOR                                 8
#define BCH_SB_LABEL_SIZE                            32
#define BCH_BTREE_MAX_DEPTH                           4
#define BCH_REPLICAS_MAX                              4
#define BCH_BKEY_PTRS_MAX                            16



/**
 * Integer Typedefs (for documentary purposes)
 * 
 * Indeterminate/no endianness: bch_uN
 * Little-endian:               bch_leN
 * Big-endian:                  bch_beN
 * 
 * There is no type-checking for endianness; The type documents the
 * endianness of the value stored on-disk, while manipulations should always
 * be carried out in the native endianness of the machine.
 */

typedef uint8_t  bch_u8;
typedef uint16_t bch_u16;
typedef uint32_t bch_u32;
typedef uint64_t bch_u64;

typedef bch_u16  bch_le16;
typedef bch_u32  bch_le32;
typedef bch_u64  bch_le64;

typedef bch_u16  bch_be16;
typedef bch_u32  bch_be32;
typedef bch_u64  bch_be64;


/**
 * Enum Forward Declarations and Typedefs
 */

#ifndef __cplusplus
typedef enum bch_metadata_version            bch_metadata_version;
typedef enum bch_bkey_type                   bch_bkey_type;
typedef enum bch_bkey_fields                 bch_bkey_fields;
typedef enum bch_extent_entry_type           bch_extent_entry_type;
typedef enum bch_xattr_type                  bch_xattr_type;
typedef enum bch_quota_type                  bch_quota_type;
typedef enum bch_quota_counters              bch_quota_counters;
typedef enum bch_member_state                bch_member_state;
typedef enum bch_cache_replacement_policies  bch_cache_replacement_policies;
typedef enum bch_kdf_type                    bch_kdf_type;
typedef enum bch_error_actions               bch_error_actions;
typedef enum bch_str_hash_type               bch_str_hash_type;
typedef enum bch_str_hash_opts               bch_str_hash_opts;
typedef enum bch_csum_type                   bch_csum_type;
typedef enum bch_csum_opts                   bch_csum_opts;
typedef enum bch_compression_type            bch_compression_type;
typedef enum bch_compression_opts            bch_compression_opts;
typedef enum bch_sb_field_type               bch_sb_field_type;
typedef enum bch_jset_entry_type             bch_jset_entry_type;
typedef enum bch_btree_id                    bch_btree_id;
typedef enum bch_inode_flags                 bch_inode_flags;
#endif


/**
 * Data Structure Forward Declarations and Typedefs
 */

typedef union  bch_csum                 bch_csum;
typedef union  bch_uuid_le              bch_uuid_le;
typedef struct bch_member               bch_member;
typedef struct bch_key                  bch_key;
typedef struct bch_encrypted_key        bch_encrypted_key;
typedef struct bch_replicas_entry_v0    bch_replicas_entry_v0;
typedef struct bch_replicas_entry       bch_replicas_entry;
typedef struct bch_quota_counter        bch_quota_counter;
typedef struct bch_sb_quota_counter     bch_sb_quota_counter;
typedef struct bch_sb_quota_type        bch_sb_quota_type;
typedef struct bch_disk_group           bch_disk_group;
typedef struct bch_journal_seq_blacklist_entry
               bch_journal_seq_blacklist_entry;
typedef struct bch_jset_entry_dev_usage_type
               bch_jset_entry_dev_usage_type;

typedef struct bch_extent_ptr           bch_extent_ptr;
typedef struct bch_extent_crc32         bch_extent_crc32;
typedef struct bch_extent_crc64         bch_extent_crc64;
typedef struct bch_extent_crc128        bch_extent_crc128;
typedef struct bch_extent_stripe_ptr    bch_extent_stripe_ptr;
typedef struct bch_extent_reservation   bch_extent_reservation;
typedef union  bch_extent_entry         bch_extent_entry;

typedef struct bch_bpos_be              bch_bpos_be;
typedef struct bch_bpos_le              bch_bpos_le;
typedef union  bch_bpos                 bch_bpos;
typedef struct bch_bversion_be          bch_bversion_be;
typedef struct bch_bversion_le          bch_bversion_le;
typedef union  bch_bversion             bch_bversion;
typedef struct bch_bkey_be              bch_bkey_be;
typedef struct bch_bkey_le              bch_bkey_le;
typedef union  bch_bkey                 bch_bkey;
typedef struct bch_bkey_i               bch_bkey_i;
typedef struct bch_bkey_packed          bch_bkey_packed;
typedef struct bch_bkey_format          bch_bkey_format;

typedef struct bch_deleted              bch_deleted;
typedef struct bch_whiteout             bch_whiteout;
typedef struct bch_error                bch_error;
typedef struct bch_cookie               bch_cookie;
typedef struct bch_hash_whiteout        bch_hash_whiteout;
typedef struct bch_btree_ptr            bch_btree_ptr;
typedef struct bch_extent               bch_extent;
typedef struct bch_reservation          bch_reservation;
typedef struct bch_inode                bch_inode;
typedef struct bch_inode_generation     bch_inode_generation;
typedef struct bch_dirent               bch_dirent;
typedef struct bch_xattr                bch_xattr;
typedef struct bch_alloc                bch_alloc;
typedef struct bch_quota                bch_quota;
typedef struct bch_stripe               bch_stripe;
typedef struct bch_reflink_p            bch_reflink_p;
typedef struct bch_reflink_v            bch_reflink_v;
typedef struct bch_inline_data          bch_inline_data;
typedef struct bch_btree_ptr_v2         bch_btree_ptr_v2;
typedef struct bch_indirect_inline_data bch_indirect_inline_data;
typedef struct bch_alloc_v2             bch_alloc_v2;
typedef struct bch_subvolume            bch_subvolume;
typedef struct bch_snapshot             bch_snapshot;
typedef struct bch_inode_v2             bch_inode_v2;
typedef struct bch_alloc_v3             bch_alloc_v3;

typedef struct bch_jset_entry               bch_jset_entry;
typedef struct bch_jset_entry_blacklist     bch_jset_entry_blacklist;
typedef struct bch_jset_entry_blacklist_v2  bch_jset_entry_blacklist_v2;
typedef struct bch_jset_entry_usage         bch_jset_entry_usage;
typedef struct bch_jset_entry_data_usage    bch_jset_entry_data_usage;
typedef struct bch_jset_entry_clock         bch_jset_entry_clock;
typedef struct bch_jset_entry_dev_usage     bch_jset_entry_dev_usage;
typedef struct bch_jset                     bch_jset;

typedef struct bch_bset                 bch_bset;
typedef struct bch_btree_node           bch_btree_node;
typedef struct bch_btree_node_entry     bch_btree_node_entry;

typedef struct bch_sb_field_journal     bch_sb_field_journal;
typedef struct bch_sb_field_members     bch_sb_field_members;
typedef struct bch_sb_field_crypt       bch_sb_field_crypt;
typedef struct bch_sb_field_replicas_v0 bch_sb_field_replicas_v0;
typedef struct bch_sb_field_quota       bch_sb_field_quota;
typedef struct bch_sb_field_disk_groups bch_sb_field_disk_groups;
typedef struct bch_sb_field_clean       bch_sb_field_clean;
typedef struct bch_sb_field_replicas    bch_sb_field_replicas;
typedef struct bch_sb_field_journal_seq_blacklist
               bch_sb_field_journal_seq_blacklist;
typedef union  bch_sb_field             bch_sb_field;

typedef struct bch_sb_layout            bch_sb_layout;
typedef struct bch_sb                   bch_sb;



/* Enum Definitions */
enum bch_metadata_version{
    BCH_METADATA_VERSION_min                       =  9,
    BCH_METADATA_VERSION_new_versioning            = 10,
    BCH_METADATA_VERSION_bkey_renumber             = 10, /* Note: duplicate! */
    BCH_METADATA_VERSION_inode_btree_change        = 11,
    BCH_METADATA_VERSION_snapshot                  = 12,
    BCH_METADATA_VERSION_inode_backpointers        = 13,
    BCH_METADATA_VERSION_btree_ptr_sectors_written = 14,
    BCH_METADATA_VERSION_snapshot_2                = 15,
    BCH_METADATA_VERSION_reflink_p_fix             = 16,
    BCH_METADATA_VERSION_subvol_dirent             = 17,
    BCH_METADATA_VERSION_inode_v2                  = 18,
    BCH_METADATA_VERSION_max,
    BCH_METADATA_VERSION_current = BCH_METADATA_VERSION_max-1,
    BCH_METADATA_VERSION_NR      = BCH_METADATA_VERSION_max
};
enum bch_bkey_type{
    BCH_BKEY_deleted,                /* 0 */
    BCH_BKEY_whiteout,
    BCH_BKEY_error,
    BCH_BKEY_cookie,
    BCH_BKEY_hash_whiteout,          /* 4 */
    BCH_BKEY_btree_ptr,
    BCH_BKEY_extent,
    BCH_BKEY_reservation,
    BCH_BKEY_inode,                  /* 8 */
    BCH_BKEY_inode_generation,
    BCH_BKEY_dirent,
    BCH_BKEY_xattr,
    BCH_BKEY_alloc,                  /* 12 */
    BCH_BKEY_quota,
    BCH_BKEY_stripe,
    BCH_BKEY_reflink_p,
    BCH_BKEY_reflink_v,              /* 16 */
    BCH_BKEY_inline_data,
    BCH_BKEY_btree_ptr_v2,
    BCH_BKEY_indirect_inline_data,
    BCH_BKEY_alloc_v2,               /* 20 */
    BCH_BKEY_subvolume,
    BCH_BKEY_snapshot,
    BCH_BKEY_inode_v2,
    BCH_BKEY_alloc_v3,               /* 24 */
    BCH_BKEY_NR
};
enum bch_bkey_fields{
    BCH_BKEY_FIELD_inode,
    BCH_BKEY_FIELD_offset,
    BCH_BKEY_FIELD_snapshot,
    BCH_BKEY_FIELD_size,
    BCH_BKEY_FIELD_version_hi,
    BCH_BKEY_FIELD_version_lo,
    BCH_BKEY_FIELD_NR
};
enum bch_extent_entry_type{
    BCH_EXTENT_ENTRY_ptr,
    BCH_EXTENT_ENTRY_crc32,
    BCH_EXTENT_ENTRY_crc64,
    BCH_EXTENT_ENTRY_crc128,
    BCH_EXTENT_ENTRY_stripe_ptr,
    BCH_EXTENT_ENTRY_reservation,
    BCH_EXTENT_ENTRY_NR
};
enum bch_xattr_type{
    BCH_XATTR_user,
    BCH_XATTR_posix_acl_access,
    BCH_XATTR_posix_acl_default,
    BCH_XATTR_trusted,
    BCH_XATTR_security,
    BCH_XATTR_NR
};
enum bch_quota_type{
    BCH_QUOTA_usr,
    BCH_QUOTA_grp,
    BCH_QUOTA_prj,
    BCH_QUOTA_NR
};
enum bch_quota_counters{
    BCH_QUOTA_COUNTER_spc,               /*  0 */
    BCH_QUOTA_COUNTER_ino,
    BCH_QUOTA_COUNTER_NR
};
enum bch_member_state{
    BCH_MEMBER_STATE_rw,                 /*  0 */
    BCH_MEMBER_STATE_ro,
    BCH_MEMBER_STATE_failed,
    BCH_MEMBER_STATE_spare,
    BCH_MEMBER_STATE_NR
};
enum bch_cache_replacement_policies{
    BCH_CACHE_REPLACEMENT_lru,
    BCH_CACHE_REPLACEMENT_fifo,
    BCH_CACHE_REPLACEMENT_random,
    BCH_CACHE_REPLACEMENT_NR
};
enum bch_kdf_type{
    BCH_KDF_scrypt,     /*  0 */
    BCH_KDF_NR
};
enum bch_data_type{
    BCH_DATA_none,      /*  0 */
    BCH_DATA_sb,
    BCH_DATA_journal,
    BCH_DATA_btree,
    BCH_DATA_user,      /*  4 */
    BCH_DATA_cached,
    BCH_DATA_parity,
    BCH_DATA_NR
};
enum bch_error_actions{
    BCH_ERROR_ACTION_continue,           /*  0 */
    BCH_ERROR_ACTION_ro,
    BCH_ERROR_ACTION_panic,
    BCH_ERROR_ACTION_NR
};
enum bch_str_hash_type{
    BCH_STR_HASH_crc32c,                 /*  0 */
    BCH_STR_HASH_crc64,
    BCH_STR_HASH_siphash_old,
    BCH_STR_HASH_siphash,
    BCH_STR_HASH_NR
};
enum bch_str_hash_opts{
    BCH_STR_HASH_OPT_crc32c,             /*  0 */
    BCH_STR_HASH_OPT_crc64,
    BCH_STR_HASH_OPT_siphash,
    BCH_STR_HASH_OPT_NR
};
enum bch_csum_type{
    BCH_CSUM_none,                       /*  0 */
    BCH_CSUM_crc32c_nonzero,
    BCH_CSUM_crc64_nonzero,
    BCH_CSUM_chacha20_poly1305_80,
    BCH_CSUM_chacha20_poly1305_128,      /*  4 */
    BCH_CSUM_crc32c,
    BCH_CSUM_crc64,
    BCH_CSUM_xxhash,
    BCH_CSUM_NR,
};
enum bch_csum_opts{
    BCH_CSUM_OPT_none,                   /*  0 */
    BCH_CSUM_OPT_crc32c,
    BCH_CSUM_OPT_crc64,
    BCH_CSUM_OPT_xxhash,
    BCH_CSUM_OPT_NR
};
enum bch_compression_type{
    BCH_COMPRESSION_none,                /*  0 */
    BCH_COMPRESSION_lz4_old,
    BCH_COMPRESSION_gzip,
    BCH_COMPRESSION_lz4,
    BCH_COMPRESSION_zstd,                /*  4 */
    BCH_COMPRESSION_incompressible,
    BCH_COMPRESSION_NR
};
enum bch_compression_opts{
    BCH_COMPRESSION_OPT_none,            /*  0 */
    BCH_COMPRESSION_OPT_lz4,
    BCH_COMPRESSION_OPT_gzip,
    BCH_COMPRESSION_OPT_zstd,
    BCH_COMPRESSION_OPT_NR
};
enum bch_sb_field_type{
    BCH_SB_FIELD_journal,                /*  0 */
    BCH_SB_FIELD_members,
    BCH_SB_FIELD_crypt,
    BCH_SB_FIELD_replicas_v0,
    BCH_SB_FIELD_quota,                  /*  4 */
    BCH_SB_FIELD_disk_groups,
    BCH_SB_FIELD_clean,
    BCH_SB_FIELD_replicas,
    BCH_SB_FIELD_journal_seq_blacklist,  /*  8 */
    BCH_SB_FIELD_NR
};
enum bch_jset_entry_type{
    BCH_JSET_ENTRY_btree_keys,           /*  0 */
    BCH_JSET_ENTRY_btree_root,
    BCH_JSET_ENTRY_prio_ptrs,
    BCH_JSET_ENTRY_blacklist,
    BCH_JSET_ENTRY_blacklist_v2,         /*  4 */
    BCH_JSET_ENTRY_usage,
    BCH_JSET_ENTRY_data_usage,
    BCH_JSET_ENTRY_clock,
    BCH_JSET_ENTRY_dev_usage,            /*  8 */
    BCH_JSET_ENTRY_NR
};
enum bch_btree_id{
    BCH_BTREE_ID_extents,                /*  0 */
    BCH_BTREE_ID_inodes,
    BCH_BTREE_ID_dirents,
    BCH_BTREE_ID_xattrs,
    BCH_BTREE_ID_alloc,                  /*  4 */
    BCH_BTREE_ID_quotas,
    BCH_BTREE_ID_stripes,
    BCH_BTREE_ID_reflink,
    BCH_BTREE_ID_subvolumes,             /*  8 */
    BCH_BTREE_ID_snapshots,
    BCH_BTREE_ID_NR
};
enum bch_inode_flags{
    BCH_INODE_FLAG_sync              = (1UL <<  0),
    BCH_INODE_FLAG_immutable         = (1UL <<  1),
    BCH_INODE_FLAG_append            = (1UL <<  2),
    BCH_INODE_FLAG_nodump            = (1UL <<  3),
    BCH_INODE_FLAG_noatime           = (1UL <<  4),
    BCH_INODE_FLAG_i_size_dirty      = (1UL <<  5),
    BCH_INODE_FLAG_i_sectors_dirty   = (1UL <<  6),
    BCH_INODE_FLAG_unlinked          = (1UL <<  7),
    BCH_INODE_FLAG_backptr_untrusted = (1UL <<  8),
    BCH_INODE_FLAG_new_varint        = (1UL << 31),
};



/* Data Structures Definitions */

/**
 * BcacheFS Generic Structures
 */

union  bch_csum{
    bch_le32 crc32;                     /* CRC-32C (Castagnoli) */
    bch_le64 crc64;                     /* CRC-64  (ECMA-182) */
    bch_u8   chacha20_poly1305_80[10];  /* ChaCha20/Poly1305 AEAD (80-bit tag) */
    bch_u8   chacha20_poly1305_128[16]; /* ChaCha20/Poly1305 AEAD (128-bit tag) */
    bch_le64 generic[2];                /* Generic 64-bit word access */
    bch_u8   bytes[16];                 /* Generic byte access */
} BENZINA_ATTRIBUTE_ALIGNED(8);
union  bch_uuid_le{
    bch_u8   bytes[16];
};
struct bch_member{
    bch_uuid_le uuid;
    bch_le64    nbuckets;     /* device size */
    bch_le16    first_bucket; /* index of first bucket used */
    bch_le16    bucket_size;  /* sectors */
    bch_le32    pad;
    bch_le64    last_mount;   /* time_t */
    
    /**
     * 2x u64, little-endian byte-order
     *   Word   |    Bits     |   Name
     * =========|=============|==================
     * flags[0] | (lsb)   4   | state (enum bch_member_state)
     *          |         6   | unused
     *          |         4   | replacement
     *          |         1   | discard
     *          |         5   | data_allowed
     *          |         8   | group
     *          |         2   | durability
     */
    bch_le64    flags[2];
};
struct bch_key{
    bch_le64 key[4];
};
struct bch_encrypted_key{
    bch_u8  magic[8];/* "bch**key" */
    bch_key key;
};
struct bch_replicas_entry_v0{
    bch_u8 data_type;
    bch_u8 nr_devs;
    bch_u8 devs[];
} BENZINA_ATTRIBUTE_PACKED BENZINA_ATTRIBUTE_DEPRECATED;
struct bch_replicas_entry{
    bch_u8 data_type;
    bch_u8 nr_devs;
    bch_u8 nr_required;
    bch_u8 devs[];
} BENZINA_ATTRIBUTE_PACKED;
struct bch_quota_counter{
    bch_le64 hardlimit;
    bch_le64 softlimit;
};
struct bch_sb_quota_counter{
    bch_le32 timelimit;
    bch_le32 warnlimit;
};
struct bch_sb_quota_type{
    bch_le64 flags;
    bch_sb_quota_counter c[BCH_QUOTA_COUNTER_NR];
};
struct bch_disk_group{
    bch_u8   label[BCH_SB_LABEL_SIZE];
    
    /**
     * 2x u64, little-endian byte-order
     *   Word   |    Bits     |   Name
     * =========|=============|==================
     * flags[0] | (lsb)   1   | deleted
     *          |         5   | data_allowed
     *          |        18   | parent
     */
    bch_le64 flags[2];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct bch_journal_seq_blacklist_entry{
    bch_le64 start;
    bch_le64 end;
};
struct bch_jset_entry_dev_usage_type{
    bch_le64 buckets;
    bch_le64 sectors;
    bch_le64 fragmented;
} BENZINA_ATTRIBUTE_PACKED;


/**
 * BcacheFS Extent Entries
 */

struct bch_extent_ptr{
    /**
     * 1x u64, filesystem-endian byte-order
     *       Bits  |   Name
     * ============|==================
     * (lsb)   1   | type (=1)
     *         1   | cached
     *         1   | unused
     *         1   | reservation (pointer hasn't been written to, just reserved)
     *        44   | offset      (up to 8 petabytes)
     *         8   | dev
     * (msb)   8   | gen
     */
    bch_u64 type;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct bch_extent_crc32{
    /**
     * 1x u64, filesystem-endian byte-order
     *       Bits  |   Name
     * ============|==================
     * (lsb)   2   | type (=2)
     *         7   | compressed_size (minus-1)
     *         7   | uncompressed_size (minus-1)
     *         7   | offset
     *         1   | unused
     *         4   | csum_type
     *         4   | compression_type
     * (msb)  32   | csum
     */
    bch_u64 type;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct bch_extent_crc64{
    /**
     * 2x u64, filesystem-endian byte-order
     *       Bits  |   Name
     * ============|==================
     * (lsb)   3   | type (=4)
     *         9   | compressed_size (minus-1)
     *         9   | uncompressed_size (minus-1)
     *         9   | offset
     *        10   | nonce
     *         4   | csum_type
     *         4   | compression_type
     * (msb)  16   | csum_hi
     * ------------+------------------
     *        64   | csum_lo
     */
    bch_u64 type;
    bch_u64 csum_lo;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct bch_extent_crc128{
    /**
     * 3x u64, filesystem-endian byte-order
     *       Bits  |   Name
     * ============|==================
     * (lsb)   4   | type (=8)
     *        13   | compressed_size (minus-1)
     *        13   | uncompressed_size (minus-1)
     *        13   | offset
     *        13   | nonce
     *         4   | csum_type
     * (msb)   4   | compression_type
     * ------------+------------------
     *       128   | csum
     */
    bch_u64  type;
    bch_csum csum;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct bch_extent_stripe_ptr{
    /**
     * 1x u64, filesystem-endian byte-order
     *       Bits  |   Name
     * ============|==================
     * (lsb)   5   | type (=16)
     *         8   | block
     *         4   | redundancy
     * (msb)  47   | idx
     */
    bch_u64 type;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct bch_extent_reservation{
    /**
     * 1x u64, filesystem-endian byte-order
     *       Bits  |   Name
     * ============|==================
     * (lsb)   6   | type (=32)
     *        22   | unused
     *         4   | replicas
     * (msb)  32   | generation
     */
    bch_u64 type;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
union  bch_extent_entry{
    bch_u64                type;
    bch_extent_ptr         ptr;
    bch_extent_crc32       crc32;
    bch_extent_crc64       crc64;
    bch_extent_crc128      crc128;
    bch_extent_stripe_ptr  stripe_ptr;
    bch_extent_reservation reservation;
};


/**
 * BcacheFS Key Structures
 */

struct bch_bpos_be{
    bch_be64 inode;
    bch_be64 offset;
    bch_be32 snapshot;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(4);
struct bch_bpos_le{
    bch_le32 snapshot;
    bch_le64 offset;
    bch_le64 inode;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(4);
union  bch_bpos{
    struct bch_bpos_be be;
    struct bch_bpos_le le;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(4);
struct bch_bversion_be{
    bch_be32 hi;
    bch_be64 lo;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(4);
struct bch_bversion_le{
    bch_le64 lo;
    bch_le32 hi;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(4);
union  bch_bversion{
    struct bch_bversion_be be;
    struct bch_bversion_le le;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(4);
struct bch_bkey_be{
    bch_u8          u64s;       /* Size of combined key and value, in u64s */
    bch_u8          formatw;    /* Format of key (0 for format local to btree node).
                                   MSB: needs_whiteout */
    bch_u8          type;       /* Type of the value */
    bch_bpos_be     pos;
    bch_be32        size;       /* extent size, in sectors */
    bch_bversion_be version;
    bch_u8          pad[1];     /* XXX: The presence of this byte here misaligns
                                        everything for big-endian starting at pos! */
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct bch_bkey_le{
    bch_u8          u64s;       /* Size of combined key and value, in u64s */
    bch_u8          formatw;    /* Format of key (0 for format local to btree node).
                                   MSB: needs_whiteout */
    bch_u8          type;       /* Type of the value */
    bch_u8          pad[1];
    bch_bversion_le version;
    bch_le32        size;       /* extent size, in sectors */
    bch_bpos_le     pos;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
union  bch_bkey{
    bch_bkey_be be;
    bch_bkey_le le;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct bch_bkey_i{
    bch_bkey k;
    bch_u64  v[];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct bch_bkey_packed{
    bch_u8          u64s;       /* Size of combined key and value, in u64s */
    bch_u8          formatw;    /* Format of key (0 for format local to btree node).
                                   MSB: needs_whiteout */
    bch_u8          type;       /* Type of the value */
    bch_u8          pad[sizeof(bch_bkey)-3];/* Actually 8*this->u64s - 3 */
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct bch_bkey_format{
    bch_u8      key_u64s;
    bch_u8      nr_fields;
    bch_u8      bits_per_field[6];
    bch_le64    field_offset[6];
};


/**
 * BcacheFS Value Structures
 * 
 * In order of associated key type.
 */

struct /* 0x00 =  0 */ bch_deleted{
    bch_u64 data[0];/* Empty */
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8); BENZINA_PRAGMA_POISON(bch_deleted)
struct /* 0x01 =  1 */ bch_whiteout{
    bch_u64 data[0];/* Empty */
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8); BENZINA_PRAGMA_POISON(bch_whiteout)
struct /* 0x02 =  2 */ bch_error{
    bch_u64 data[0];/* Empty */
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8); BENZINA_PRAGMA_POISON(bch_error)
struct /* 0x03 =  3 */ bch_cookie{
    bch_le64 cookie;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct /* 0x04 =  4 */ bch_hash_whiteout{
    bch_u64 data[0];/* Empty */
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8); BENZINA_PRAGMA_POISON(bch_hash_whiteout)
struct /* 0x05 =  5 */ bch_btree_ptr{
    bch_extent_ptr start[0];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8)  BENZINA_ATTRIBUTE_DEPRECATED;
struct /* 0x06 =  6 */ bch_extent{
    bch_extent_entry start[0];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct /* 0x07 =  7 */ bch_reservation{
    bch_le32 generation;
    bch_u8   nr_replicas;
    bch_u8   pad[3];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct /* 0x08 =  8 */ bch_inode{
    bch_le64 bi_hash_seed;
    
    /**
     * 1x u32, little-endian byte-order
     *       Bits  |   Name
     * ============|==================
     * (lsb)   1   | BCH_INODE_SYNC
     *         1   | BCH_INODE_IMMUTABLE
     *         1   | BCH_INODE_APPEND
     *         1   | BCH_INODE_NODUMP
     *         1   | BCH_INODE_NOATIME
     *         1   | BCH_INODE_I_SIZE_DIRTY
     *         1   | BCH_INODE_I_SECTORS_DIRTY
     *         1   | BCH_INODE_UNLINKED
     *         1   | BCH_INODE_BACKPTR_UNTRUSTED
     *        11   | unused
     *         4   | str_hash
     *         7   | nr_fields
     * (msb)   1   | new_varint
     */
    bch_le32 bi_flags;
    
    /**
     * 1x u16, little-endian byte-order
     *       Bits  |   Name
     * ============|==================
     * (lsb)   3   | Other rwx permissions (S_I[RWX]USR)
     *         3   | Group rwx permissions (S_I[RWX]GRP)
     *         3   | Owner rwx permissions (S_I[RWX]OTH)
     *         1   | Sticky bit            (S_ISVRX)
     *         1   | Setgid bit            (S_ISGID)
     *         1   | Setuid bit            (S_ISUID)
     * (msb)   4   | Inode mode. Extractable with (stat.st_mode & S_IFMT).
     *             | Possible values (octal):
     *             |     S_IFSOCK   0140000   socket
     *             |     S_IFLNK    0120000   symbolic link
     *             |     S_IFREG    0100000   regular file
     *             |     S_IFBLK    0060000   block device
     *             |     S_IFDIR    0040000   directory
     *             |     S_IFCHR    0020000   character device
     *             |     S_IFIFO    0010000   FIFO
     * ------------+------------------
     * References: `man 7 inode`
     */
    bch_le16 bi_mode;
    bch_u8   fields[];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct /* 0x09 =  9 */ bch_inode_generation{
    bch_le32 bi_generation;
    bch_le32 pad;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct /* 0x0a = 10 */ bch_dirent{
    /**
     * Dirent Target
     * 
     *   - For non-subvolumes (d_type != DT_SUBVOL):
     *     Dirent maps name to inode number d_inum.
     * 
     *   - For subvolumes     (d_type == DT_SUBVOL):
     *     Dirent maps name to child subvolume id. Also records parent
     *     subvolume id.
     */
    union{
        bch_le64 d_inum;/* Target inode number */
        struct{
            bch_le32 d_child_subvol;
            bch_le32 d_parent_subvol;
        } subvol;
    };
    
    /*
     * Copy of mode bits 12-15 from the target inode - so userspace can get
     * the filetype without having to do a stat()
     */
    bch_u8   d_type;
    bch_u8   d_name[];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct /* 0x0b = 11 */ bch_xattr{
    bch_u8    x_type;      /* Type: enum bch_xattr_type */
    bch_u8    x_name_len;  /* Unit: bytes > 0, <=255 */
    bch_le16  x_val_len;   /* Unit: bytes <= 1995 */
    bch_u8    x_name[];    /* Not NUL-terminated */
    
    /**
     * bch_u8 x_val[] begins immediately after x_name[], with no preceding or
     * terminating NUL byte. If the length of bch_xattr including both .x_name
     * and .x_val arrays is not a multiple of 8, padding bytes are added to
     * round the size up to the next multiple of 8 bytes.
     * 
     * The maximum size of the bch_bkey, bch_xattr, x_name and x_val combined
     * must be less than or equal to 255 u64s (2040 bytes), and therefore the
     * maximum safe size for an x_val is:
     * 
     *     2040 bytes (a maximally-large key-value pair)
     *     - 40 bytes (a worst-case, uncompressed bch_bkey)
     *     -  4 bytes (the fixed header size of a bch_xattr)
     *     -  1 bytes (the shortest possible name for an xattr)
     *   = 1995 bytes.
     */
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct /* 0x0c = 12 */ bch_alloc{
    bch_u8 fields;
    bch_u8 gen;
    bch_u8 data[];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8)  BENZINA_ATTRIBUTE_DEPRECATED;
struct /* 0x0d = 13 */ bch_quota{
    bch_quota_counter c[BCH_QUOTA_COUNTER_NR];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct /* 0x0e = 14 */ bch_stripe{
    bch_le16       sectors;
    bch_u8         algorithm;
    bch_u8         nr_blocks;
    bch_u8         nr_redundant;

    bch_u8         csum_granularity_bits;
    bch_u8         csum_type;
    bch_u8         pad;

    bch_extent_ptr ptrs[];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct /* 0x0f = 15 */ bch_reflink_p{
    bch_le64 idx;
    bch_le32 front_pad;
    bch_le32 back_pad;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct /* 0x10 = 16 */ bch_reflink_v{
    bch_le64         refcount;
    bch_extent_entry start[];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct /* 0x11 = 17 */ bch_inline_data{
    bch_u8 data[0];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct /* 0x12 = 18 */ bch_btree_ptr_v2{
    bch_u64        mem_ptr;
    bch_le64       seq;
    bch_le16       sectors_written;
    
    /**
     * 1x u16, little-endian byte-order
     *     Bits     |   Name
     * =============|==================
     *  (lsb)   1   | ptr_range_updated
     */
    bch_le16       flags;
    bch_bpos       min_key;
    bch_extent_ptr start[];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct /* 0x13 = 19 */ bch_indirect_inline_data{
    bch_le64 refcount;
    bch_u8   data[];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct /* 0x14 = 20 */ bch_alloc_v2{
    bch_u8 nr_fields;
    bch_u8 gen;
    bch_u8 oldest_gen;
    bch_u8 data_type;
    bch_u8 data[];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct /* 0x15 = 21 */ bch_subvolume{
    /**
     * 1x u32, little-endian byte-order
     *     Bits     |   Name
     * =============|==================
     *  (lsb)   1   | ro
     *          1   | snap       (Is this subvolume a snapshot?)
     *          1   | unlinked   (Is this subvolume unlinked?)
     */
    bch_le32 flags;
    bch_le32 snapshot;
    bch_le64 inode;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct /* 0x16 = 22 */ bch_snapshot{
    /**
     * 1x u32, little-endian byte-order
     *     Bits     |   Name
     * =============|==================
     *  (lsb)   1   | deleted
     *          1   | subvol  (True if a subvolume points to this snapshot node)
     */
    bch_le32 flags;
    bch_le32 parent;
    bch_le32 children[2];
    bch_le32 subvol;
    bch_le32 pad;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct /* 0x17 = 23 */ bch_inode_v2{
    bch_le64 bi_journal_seq;
    bch_le64 bi_hash_seed;
    
    /**
     * 1x u32, little-endian byte-order
     *       Bits  |   Name
     * ============|==================
     * (lsb)   1   | BCH_INODE_SYNC
     *         1   | BCH_INODE_IMMUTABLE
     *         1   | BCH_INODE_APPEND
     *         1   | BCH_INODE_NODUMP
     *         1   | BCH_INODE_NOATIME
     *         1   | BCH_INODE_I_SIZE_DIRTY
     *         1   | BCH_INODE_I_SECTORS_DIRTY
     *         1   | BCH_INODE_UNLINKED
     *         1   | BCH_INODE_BACKPTR_UNTRUSTED
     *        11   | unused
     *         4   | str_hash
     *         7   | nr_fields
     */
    bch_le32 bi_flags;
    
    /**
     * 1x u16, little-endian byte-order
     *       Bits  |   Name
     * ============|==================
     * (lsb)   3   | Other rwx permissions (S_I[RWX]USR)
     *         3   | Group rwx permissions (S_I[RWX]GRP)
     *         3   | Owner rwx permissions (S_I[RWX]OTH)
     *         1   | Sticky bit            (S_ISVRX)
     *         1   | Setgid bit            (S_ISGID)
     *         1   | Setuid bit            (S_ISUID)
     * (msb)   4   | Inode mode. Extractable with (stat.st_mode & S_IFMT).
     *             | Possible values (octal):
     *             |     S_IFSOCK   0140000   socket
     *             |     S_IFLNK    0120000   symbolic link
     *             |     S_IFREG    0100000   regular file
     *             |     S_IFBLK    0060000   block device
     *             |     S_IFDIR    0040000   directory
     *             |     S_IFCHR    0020000   character device
     *             |     S_IFIFO    0010000   FIFO
     * ------------+------------------
     * References: `man 7 inode`
     */
    bch_le16 bi_mode;
    bch_u8   fields[];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct /* 0x18 = 24 */ bch_alloc_v3{
    bch_le64 bi_journal_seq;
    
    /**
     * 1x u32, little-endian byte-order
     *       Bits  |   Name
     * ============|==================
     */
    bch_le32 bi_flags;
    bch_u8   nr_fields;
    bch_u8   gen;
    bch_u8   oldest_gen;
    bch_u8   data_type;
    bch_u8   data[];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);



/**
 * BcacheFS Journal Set Entry Structures
 */

struct bch_jset_entry{
    bch_le16   u64s;    /* Exclusive of this header's word. */
    bch_u8     btree_id;/* Designates btree to which this jset relates (enum bch_btree_id) */
    bch_u8     level;
    bch_u8     type;    /* Designates what this jset holds (enum bch_jset_entry_type) */
    bch_u8     pad[3];
    bch_bkey_i start[0];
};
struct bch_jset_entry_blacklist{
    bch_jset_entry entry;
    bch_le64       seq;
};
struct bch_jset_entry_blacklist_v2{
    bch_jset_entry entry;
    bch_le64       start;
    bch_le64       end;
};
struct bch_jset_entry_usage{
    bch_jset_entry entry;
    bch_le64       v;
} BENZINA_ATTRIBUTE_PACKED;
struct bch_jset_entry_data_usage{
    bch_jset_entry entry;
    bch_le64       v;
    bch_replicas_entry r;
} BENZINA_ATTRIBUTE_PACKED;
struct bch_jset_entry_clock{
    bch_jset_entry entry;
    bch_u8         rw;
    bch_u8         pad[7];
    bch_le64       time;
} BENZINA_ATTRIBUTE_PACKED;
struct bch_jset_entry_dev_usage{
    bch_jset_entry entry;
    bch_le32       dev;
    bch_u32        pad;
    bch_le64       buckets_ec;
    bch_le64       buckets_unavailable;
    bch_jset_entry_dev_usage_type d[];
} BENZINA_ATTRIBUTE_PACKED;
struct bch_jset{
    bch_csum csum;
    bch_le64 magic;
    bch_le64 seq;
    bch_le32 version;
    
    /**
     * 1x u32, little-endian byte-order
     *   Word   |    Bits     |   Name
     * =========|=============|==================
     * flags[0] | (lsb)   4   | csum_type (enum bch_csum_type)
     *          |         1   | big_endian
     *          |         1   | no_flush
     */
    bch_le32 flags;
    bch_le32 u64s; /* Size of entries[] in u64s, exclusive of this header. */
    /* Start of encrypted area */
    bch_le16 _read_clock  BENZINA_ATTRIBUTE_DEPRECATED;
    bch_le16 _write_clock BENZINA_ATTRIBUTE_DEPRECATED;
    bch_le64 last_seq; /* Sequence number of oldest dirty journal entry */
    bch_jset_entry entries[];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);


/**
 * BcacheFS B+tree Node & Set Structures
 * 
 * A BcacheFS B+tree node's size is determined by the pointer into them;
 * Specifically, bch_btree_ptr_v2->sectors_written is the number of valid
 * sectors out of a maximum of bch_sb->btree_node_size.
 * 
 * B+tree nodes always begin with a struct bch_btree_node at offset 0 sectors;
 * thereafter follows a list of struct bch_btree_node_entry[], each
 * encapsulating one checksummed bch_bset. A bch_btree_node_entry always begins
 * on the next even bch_sb->block_size boundary rounding up from the previous
 * bch_btree_node_entry's tail. There is no limit on the number of bsets
 * within a B+tree node except the trivial btree_node_size/block_size.
 * 
 * With a default block_size of 4096 (8 sectors), an illustrative layout is
 * 
 *     bch_btree_ptr_v2->sectors_written == 48
 *     bch_btree_ptr_v2->start.offset
 *        ===>
 *             - Sector +  0: bch_btree_node         (empty bset)
 *                      +  1-7:   <skipped, zeros>
 *             - Sector +  8: bch_btree_node_entry   (bset 0, 31 sectors rounded up to 32)
 *                      + 38:     <last sector of bset 0>
 *                      + 39:     <skipped, zeros>
 *             - Sector + 40: bch_btree_node_entry   (bset 1, 5 sectors rounded up to 8)
 *                      + 44:     <last sector of bset 1>
 *                      + 45-47:  <skipped, zeros>
 */

struct bch_bset{
    bch_le64 seq;
    bch_le64 journal_seq;
    
    /**
     * 1x u32, little-endian byte-order
     *     Bits     |   Name
     * =============|==================
     *  (lsb)   4   | csum_type (enum bch_csum_type)
     *          1   | big_endian
     *          1   | separate_whiteouts
     *         10   | unused
     *  (msb)  16   | offset (in sectors, from btree node start)
     */
    bch_le32 flags;
    bch_le16 version;
    bch_le16 u64s; /* Size of start[] array, in u64s. */
    bch_bkey_packed start[];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct bch_btree_node{
    bch_csum csum;
    bch_le64 magic;
    
    /* Beginning of encrypted area: */
    
    /**
     * 1x u64, little-endian byte-order
     *     Bits     |   Name
     * =============|==================
     *  (lsb)   4   | id (enum bch_btree_id)
     *          4   | level (0-3)
     *          1   | new_extent_overwrite
     *         23   | unused
     *  (msb)  32   | node_seq
     */
    bch_le64 flags;
    bch_bpos min_key;/* Minimum, inclusive (closed interval) */
    bch_bpos max_key;/* Maximum, Inclusive (closed interval) */
    bch_extent_ptr _ptr BENZINA_ATTRIBUTE_DEPRECATED;
    bch_bkey_format format;
    bch_bset keys;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct bch_btree_node_entry{
    bch_csum csum;
    bch_bset keys;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);


/**
 * BcacheFS Superblock Field Structures
 * 
 * The u64s header size field is *inclusive* of itself (or, more precisely, the
 * 64-bit word within which it is found).
 */

struct bch_sb_field_journal{
    bch_le32 u64s;
    bch_le32 type;
    bch_le64 buckets[];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct bch_sb_field_members{
    bch_le32 u64s;
    bch_le32 type;
    bch_member members[];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct bch_sb_field_crypt{
    bch_le32 u64s;
    bch_le32 type;
    
    /**
     * 1x u64, little-endian byte-order
     *     Bits     |   Name
     * =============|==================
     *  (lsb)   4   | kdf_type (enum bch_kdf_type)
     */
    bch_le64 flags;
    
    /**
     * 1x u64, little-endian byte-order
     *     Bits     |   Name
     * =============|==================
     *  (lsb)  16   | scrypt_n (base-2 logarithm)
     *         16   | scrypt_r (base-2 logarithm)
     *         16   | scrypt_p (base-2 logarithm)
     */
    bch_le64 kdf_flags;
    bch_encrypted_key key;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct bch_sb_field_replicas_v0{
    bch_le32 u64s;
    bch_le32 type;
    bch_replicas_entry_v0 entries[] BENZINA_ATTRIBUTE_DEPRECATED;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8) BENZINA_ATTRIBUTE_DEPRECATED;
struct bch_sb_field_quota{
    bch_le32 u64s;
    bch_le32 type;
    bch_sb_quota_type q[BCH_QUOTA_NR];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct bch_sb_field_disk_groups{
    bch_le32 u64s;
    bch_le32 type;
    bch_disk_group entries[];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct bch_sb_field_clean{
    bch_le32 u64s;
    bch_le32 type;
    bch_le32 flags;
    bch_le16 _read_clock  BENZINA_ATTRIBUTE_DEPRECATED;
    bch_le16 _write_clock BENZINA_ATTRIBUTE_DEPRECATED;
    bch_le64 journal_seq;
    bch_jset_entry entries[];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct bch_sb_field_replicas{
    bch_le32 u64s;
    bch_le32 type;
    bch_replicas_entry entries[];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct bch_sb_field_journal_seq_blacklist{
    bch_le32 u64s;
    bch_le32 type;
    bch_journal_seq_blacklist_entry entries[];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
union  bch_sb_field{
    struct{
        bch_le32 u64s; /* Inclusive of this header's word. */
        bch_le32 type;
        bch_u64  data[];
    } generic;
    bch_sb_field_journal               journal;
    bch_sb_field_members               members;
    bch_sb_field_crypt                 crypt;
    bch_sb_field_replicas_v0           replicas_v0 BENZINA_ATTRIBUTE_DEPRECATED;
    bch_sb_field_quota                 quota;
    bch_sb_field_disk_groups           disk_groups;
    bch_sb_field_clean                 clean;
    bch_sb_field_replicas              replicas;
    bch_sb_field_journal_seq_blacklist journal_seq_blacklist;
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);


/**
 * BcacheFS Superblock Structures
 */

struct bch_sb_layout{
    bch_uuid_le   magic;            /* bcachefs superblock UUID */
    bch_u8        layout_type;
    bch_u8        sb_max_size_bits; /* base 2 of 512 byte sectors */
    bch_u8        nr_superblocks;
    bch_u8        pad[5];
    bch_le64      sb_offset[61];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);
struct bch_sb{
    bch_csum      csum;
    bch_le16      version;
    bch_le16      version_min;
    bch_le16      pad[2];
    bch_uuid_le   magic;
    bch_uuid_le   uuid;
    bch_uuid_le   user_uuid;
    bch_u8        label[BCH_SB_LABEL_SIZE];
    bch_le64      offset;
    bch_le64      seq;
    
    bch_le16      block_size;
    bch_u8        dev_idx;
    bch_u8        nr_devices;
    bch_le32      u64s;
    
    bch_le64      time_base_lo;
    bch_le32      time_base_hi;
    bch_le32      time_precision;
    
    
    /**
     * 8x u64, little-endian byte-order
     *   Word   |    Bits     |   Name
     * =========|=============|================================================
     * flags[0] | (lsb)   1   | initialized
     * flags[0] |         1   | clean
     * flags[0] |         6   | csum_type            (enum bch_csum_type)
     * flags[0] |         4   | error_action         (enum bch_error_actions)
     * flags[0] |        16   | btree_node_size      (Must be power-of-2 and >= `block_size`. **Min: `1`** (512B), **Max: `512`** (256K))
     * flags[0] |         5   | gc_reserve,   in %
     * flags[0] |         7   | root_reserve, in %
     * flags[0] |         4   | meta_csum_type       (enum bch_csum_type)
     * flags[0] |         4   | data_csum_type       (enum bch_csum_type)
     * flags[0] |         4   | meta_replicas_want   (< BCH_REPLICAS_MAX = 4)
     * flags[0] |         4   | data_replicas_want   (< BCH_REPLICAS_MAX = 4)
     * flags[0] |         1   | posix_acl
     * flags[0] |         1   | usrquota
     * flags[0] |         1   | grpquota
     * flags[0] |         1   | prjquota
     * flags[0] |         1   | has_errors
     * flags[0] |         1   | has_topology_errors
     * flags[0] |         1   | big_endian
     * ---------+-------------+------------------------------------------------
     * flags[1] | (lsb)   4   | str_hash_type        (enum bch_str_hash_type)
     * flags[1] |         4   | compression_type     (enum bch_compression_type)
     * flags[1] |         1   | inode_32bit
     * flags[1] |         1   | wide_macs
     * flags[1] |         4   | encryption_type      (if non-0, encryption enabled; overrides data/metadata csum type;
     *          |             |                       wide_macs determines BCH_CSUM_chacha20_poly1305_80 vs BCH_CSUM_chacha20_poly1305_128)
     * flags[1] |         6   | encoded_extent_max_bits
     * flags[1] |         4   | meta_replicas_required   (< BCH_REPLICAS_MAX = 4)
     * flags[1] |         4   | data_replicas_required   (< BCH_REPLICAS_MAX = 4)
     * flags[1] |        12   | promote_target
     * flags[1] |        12   | foreground_target
     * flags[1] | (msb)  12   | background_target
     * ---------+-------------+------------------------------------------------
     * flags[2] | (lsb)   4   | background_compression_type    (enum bch_compression_type)
     * flags[2] | (msb)  60   | gc_reserve_bytes
     * ---------+-------------+------------------------------------------------
     * flags[3] | (lsb)  16   | erasure_code
     * flags[3] |        12   | metadata_target
     * flags[3] |         1   | shard_inums
     * flags[3] |         1   | inodes_use_key_cache
     */
    bch_le64      flags   [8];
    
    /**
     * 2x u64, little-endian byte-order
     *     Word    |    Bits     |   Name
     * ============|=============|================================================
     * features[0] | (lsb)   1   | lz4
     * features[0] |         1   | gzip
     * features[0] |         1   | zstd
     * features[0] |         1   | atomic_nlink
     * features[0] |         1   | ec
     * features[0] |         1   | journal_seq_blacklist_v3
     * features[0] |         1   | reflink
     * features[0] |         1   | new_siphash
     * features[0] |         1   | inline_data
     * features[0] |         1   | new_extent_overwrite
     * features[0] |         1   | incompressible
     * features[0] |         1   | btree_ptr_v2
     * features[0] |         1   | extents_above_btree_updates
     * features[0] |         1   | btree_updates_journalled
     * features[0] |         1   | reflink_inline_data
     * features[0] |         1   | new_varint
     * features[0] |         1   | journal_no_flush
     * features[0] |         1   | alloc_v2
     * features[0] |         1   | extents_across_btree_nodes
     */
    bch_le64      features[2];
    
    /**
     * 2x u64, little-endian byte-order
     *    Word   |    Bits     |   Name
     * ==========|=============|================================================
     * compat[0] | (lsb)   1   | alloc_info
     * compat[0] |         1   | alloc_metadata
     * compat[0] |         1   | extents_above_btree_updates_done
     * compat[0] |         1   | bformat_overflow_done
     */
    bch_le64      compat  [2];
    
    bch_sb_layout layout;
    bch_sb_field  fields[];
} BENZINA_ATTRIBUTE_PACKED_ALIGNED(8);


/* End Extern "C" and Include Guard */
#ifdef __cplusplus
}
#endif
#endif
