# BcacheFS On-Disk Format


## 1. Intro

BcacheFS uses in several places the following units:

- Sectors: A unit of 512 bytes. Offsets are frequently expressed as an absolute number of sectors from the beginning of the filesystem.
- `u64s`:  A size expressed in full 64-bit/8-byte words.




## 2. Data Structures

### 2.1. `struct bch_sb`: The Superblock

|Offset | Size      | Type            |  Name           |   Ʉ   | Description                                                                    |
|-------|-----------|-----------------|-----------------|-------|--------------------------------------------------------------------------------|
|      0|   16      | `bch_csum`      | `csum`          |       | Superblock checksum                                                            |
|     16|    2      | `uint16`        | `version`       |       | On-disk format version                                                         |
|     18|    2      | `uint16`        | `version_min`   |       | Oldest metadata version this filesystem contains                               |
|      -|    -      | `uint16[2]`     | _padding_       |   -   |     -                                                                          |
|     24|   16      | `UUID`          | `magic`         |       | BcacheFS filesystem identifier (BCACHE_MAGIC)                                  |
|     40|   16      | `UUID`          | `uuid`          |       | Used for generating various magic numbers and identifying member devices. Never changes. Cannot be all-zeroes. |
|     56|   16      | `UUID`          | `user_uuid`     |       | User-visible UUID, may be changed. Cannot be all-zeroes.                                                       |
|     72|   32      | `char[32]`      | `label`         |       | Not necessarily NUL-terminated.                                                |
|    104|    8      | `uint64`        | `offset`        | **S** | Offset of this copy of the superblock, for consistency checking.               |
|    112|    8      | `uint64`        | `seq`           |       | Identifies most recent superblock. Incremented each time superblock is written.|
|    120|    2      | `uint16`        | `block_size`    | **S** | Must be power-of-2. **Min: `1`** (512B), **Max: `128`** (64K)                  |
|    122|    1      | `uint8`         | `dev_idx`       |       | Device index within array. Lower than `nr_devices`.                            |
|    123|    1      | `uint8`         | `nr_devices`    |       | Number of devices within array. For a single-device filesystem, `1`.           |
|    124|    4      | `uint32`        | `u64s`          | **U** | Total size of the `fields` array, in u64s.                                     |
|    128|    8      | `uint64`        | `time_base_lo`  |       |                                                                                |
|    136|    4      | `uint32`        | `time_base_hi`  |       |                                                                                |
|    140|    4      | `uint32`        | `time_precision`|       |                                                                                |
|    144|   64      | `uint64[8]`     | `flags`         |       | Bitfields                                                                      |
|    208|   16      | `uint64[2]`     | `features`      |       | Bitfields                                                                      |
|    224|   16      | `uint64[2]`     | `compat`        |       | Bitfields                                                                      |
|    240|  512      | `bch_sb_layout` | `layout`        |       |                                                                                |
|    752| 8x `u64s` | `bch_sb_field`  | `fields`        |       | Array of contiguous structures `bch_sb_field_*`.                               |

**Size: (752+8x u64s) bytes**

One copy of this data structure always begins at an `offset` of `BCH_SB_SECTOR` = **8 sectors** (4096 bytes) into the partition. It is the root data structure from which all other ones are located.


#### 2.1.1. Flags

| Location |Hi  |Lo  | Name                        |Bw  |   Ʉ   | Description                                                                       |
|----------|----|----|-----------------------------|----|-------|-----------------------------------------------------------------------------------|
|`flags[0]`| `1`| `0`|`initialized`                | `1`|       |                                                                                   |
|`flags[0]`| `2`| `1`|`clean`                      | `1`|       |                                                                                   |
|`flags[0]`| `8`| `2`|`csum_type`                  | `6`|       |                                                                                   |
|`flags[0]`|`12`| `8`|`error_action`               | `4`|       |                                                                                   |
|`flags[0]`|`28`|`12`|`btree_node_size`            |`16`| **S** | Must be power-of-2 and >= `block_size`. **Min: `1`** (512B), **Max: `512`** (256K)|
|`flags[0]`|`33`|`28`|`gc_reserve`                 | `5`|       |                                                                                   |
|`flags[0]`|`40`|`33`|`root_reserve`               | `7`|       |                                                                                   |
|`flags[0]`|`44`|`40`|`meta_csum_type`             | `4`|       |                                                                                   |
|`flags[0]`|`48`|`44`|`data_csum_type`             | `4`|       |                                                                                   |
|`flags[0]`|`52`|`48`|`meta_replicas_want`         | `4`|       | `< BCH_REPLICAS_MAX = 4`                                                          |
|`flags[0]`|`56`|`52`|`data_replicas_want`         | `4`|       | `< BCH_REPLICAS_MAX = 4`                                                          |
|`flags[0]`|`57`|`56`|`posix_acl`                  | `1`|       |                                                                                   |
|`flags[0]`|`58`|`57`|`usrquota`                   | `1`|       |                                                                                   |
|`flags[0]`|`59`|`58`|`grpquota`                   | `1`|       |                                                                                   |
|`flags[0]`|`60`|`59`|`prjquota`                   | `1`|       |                                                                                   |
|`flags[0]`|`61`|`60`|`has_errors`                 | `1`|       |                                                                                   |
|`flags[0]`|`62`|`61`|`has_topology_errors`        | `1`|       |                                                                                   |
|`flags[0]`|`63`|`62`|`big_endian`                 | `1`|       |                                                                                   |
|          |    |    |                             |    |       |                                                                                   |
|`flags[1]`| `4`| `0`|`str_hash_type`              | `4`|       |                                                                                   |
|`flags[1]`| `8`| `4`|`compression_type`           | `4`|       |                                                                                   |
|`flags[1]`| `9`| `8`|`inode_32bit`                | `1`|       |                                                                                   |
|`flags[1]`|`10`| `9`|`wide_macs`                  | `1`|       |                                                                                   |
|`flags[1]`|`14`|`10`|`encryption_type`            | `4`|       |                                                                                   |
|`flags[1]`|`20`|`14`|`encoded_extent_max_bits`    | `6`|       |                                                                                   |
|`flags[1]`|`24`|`20`|`meta_replicas_required`     | `4`|       | `< BCH_REPLICAS_MAX = 4`                                                          |
|`flags[1]`|`28`|`24`|`data_replicas_required`     | `4`|       | `< BCH_REPLICAS_MAX = 4`                                                          |
|`flags[1]`|`40`|`28`|`promote_target`             |`12`|       |                                                                                   |
|`flags[1]`|`52`|`40`|`foreground_target`          |`12`|       |                                                                                   |
|`flags[1]`|`64`|`52`|`background_target`          |`12`|       |                                                                                   |
|          |    |    |                             |    |       |                                                                                   |
|`flags[2]`| `4`| `0`|`background_compression_type`| `4`|       |                                                                                   |
|`flags[2]`|`64`| `4`|`gc_reserve_bytes`           |`60`|       |                                                                                   |
|          |    |    |                             |    |       |                                                                                   |
|`flags[3]`|`16`| `0`|`erasure_code`               |`16`|       |                                                                                   |
|`flags[3]`|`28`|`16`|`metadata_target`            |`12`|       |                                                                                   |
|`flags[3]`|`29`|`28`|`shard_inums`                | `1`|       |                                                                                   |
|`flags[3]`|`30`|`29`|`inodes_use_key_cache`       | `1`|       |                                                                                   |


#### 2.1.2. Features

| Location    |Hi  |Lo  | Name                        |Bw  |   Ʉ   | Description |
|-------------|----|----|-----------------------------|----|-------|-------------|
|`features[0]`| `1`| `0`|`lz4`                        | `1`|       |             |
|`features[0]`| `2`| `1`|`gzip`                       | `1`|       |             |
|`features[0]`| `3`| `2`|`zstd`                       | `1`|       |             |
|`features[0]`| `4`| `3`|`atomic_nlink`               | `1`|       |             |
|`features[0]`| `5`| `4`|`ec`                         | `1`|       |             |
|`features[0]`| `6`| `5`|`journal_seq_blacklist_v3`   | `1`|       |             |
|`features[0]`| `7`| `6`|`reflink`                    | `1`|       |             |
|`features[0]`| `8`| `7`|`new_siphash`                | `1`|       |             |
|`features[0]`| `9`| `8`|`inline_data`                | `1`|       |             |
|`features[0]`|`10`| `9`|`new_extent_overwrite`       | `1`|       |             |
|`features[0]`|`11`|`10`|`incompressible`             | `1`|       |             |
|`features[0]`|`12`|`11`|`btree_ptr_v2`               | `1`|       |             |
|`features[0]`|`13`|`12`|`extents_above_btree_updates`| `1`|       |             |
|`features[0]`|`14`|`13`|`btree_updates_journalled`   | `1`|       |             |
|`features[0]`|`15`|`14`|`reflink_inline_data`        | `1`|       |             |
|`features[0]`|`16`|`15`|`new_varint`                 | `1`|       |             |
|`features[0]`|`17`|`16`|`journal_no_flush`           | `1`|       |             |
|`features[0]`|`18`|`17`|`alloc_v2`                   | `1`|       |             |
|`features[0]`|`19`|`18`|`extents_across_btree_nodes` | `1`|       |             |


#### 2.1.3. Compat

| Location  |Hi  |Lo  | Name                             |Bw  |   Ʉ   | Description |
|-----------|----|----|----------------------------------|----|-------|-------------|
|`compat[0]`| `1`| `0`|`alloc_info`                      | `1`|       |             |
|`compat[0]`| `2`| `1`|`alloc_metadata`                  | `1`|       |             |
|`compat[0]`| `3`| `2`|`extents_above_btree_updates_done`| `1`|       |             |
|`compat[0]`| `4`| `3`|`bformat_overflow_done`           | `1`|       |             |


### 2.2. `union bch_csum`: Checksum

|Offset | Size      | Type            |  Name                   |Ʉ| Description                           |
|-------|-----------|-----------------|-------------------------|-|---------------------------------------|
|      0|         4 | `uint32`        | `crc32`                 | | CRC-32C (Castagnoli)                  |
|      0|         8 | `uint64`        | `crc64`                 | | CRC-64  (ECMA-182)                    |
|      0|        10 | `uint8[10]`     | `chacha20_poly1305_80`  | | ChaCha20/Poly1305 AEAD (80-bit tag)   |
|      0|        16 | `uint8[16]`     | `chacha20_poly1305_128` | | ChaCha20/Poly1305 AEAD (128-bit tag)  |
|      0|        16 | `uint64[2]`     | `generic`               | | Generic 64-bit words                  |
|      0|        16 | `uint8[16]`     | `bytes`                 | | Generic byte access                   |

**Size: 16 bytes**

The size of this structure is exactly 16 bytes in all circumstances. The 80-bit AEAD construction exists because in B+tree nodes, truncating at 80 bits can save 8 bytes and maintain the size of an extent at 32 bytes; However, in the superblock, no space saving is obtained.

Which of these checksum types is in use depends on `bch_sb->flags` in the superblock.




### 2.3. `UUID`

|Offset | Size      | Type            |  Name                   |Ʉ| Description  |
|-------|-----------|-----------------|-------------------------|-|--------------|
|      0|        16 | `uint8[16]`     | `bytes`                 | | 128-bit UUID |

**Size: 16 bytes**

A 128-bit UUID.




### 2.4. `struct bch_sb_layout`: Disk Layout.

|Offset  | Size      | Type          |  Name              |   Ʉ         | Description                          |
|--------|-----------|---------------|--------------------|-------------|----------------------------------------------|
|      0 |        16 | `UUID`        | `magic`            |             | BcacheFS superblock UUID                     |
|     16 |         1 | `uint8`       | `layout_type`      |             | Layout type (currently, only `0`)            |
|     17 |         1 | `uint8`       | `sb_max_size_bits` | **log2(S)** | Log base-2 of superblock size.               |
|     18 |         1 | `uint8`       | `nr_superblocks`   |             | Number of superblocks. Must be less than 61. |
|      - |         - | `uint8[5]`    | _padding_          |      -      | -                                            |
|     24 |       488 | `uint64[61]`  | `sb_offset`        |    **S**    | Offset of superblock i.                      |

**Size: 512 bytes**

One copy of this data structure is embedded within every superblock, and one copy is placed at an offset of `BCH_SB_LAYOUT_SECTOR` = **7 sectors** (3584 bytes) into the partition (immediately behind the master superblock).


