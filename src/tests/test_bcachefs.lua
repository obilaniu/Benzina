--[[
BcacheFS tests, using the Test Anything Protocol (TAP)
--]]

print('1..105')


-- Small utility function to save typing while defining tests
function tstmessagetap(name, cond)
  if cond then print('ok '..name)
  else         print('not ok '..name)
  end
end


bcachefs = require 'benzina.bcachefs'
siphash  = require 'benzina.siphash'.siphash



--[[
The following sequence of tests undertake the necessary steps to access a path
    /n09332890/n09332890_29876.JPEG
from scratch.


BcacheFS Inode 4096 Unpack

Root inode (4096) of a test filesystem. Corresponds to the root directory, "/".
Packed length: 6x u64s w/ pad.
--]]

t1 = {
    -- Fixed:
    bi_hash_seed              = 0x49865c96d772670d,
    bi_flags                  = 0x89300000,
    bi_mode                   = 0x41ed, -- octal 40755, dir drwxr-xr-x
    -- Varint v2:
    bi_atime                  = 51151231,
    bi_atime_hi               = 0,
    bi_ctime                  = 110127203799,
    bi_ctime_hi               = 0,
    bi_mtime                  = 110127203799,
    bi_mtime_hi               = 0,
    bi_otime                  = 51151231,
    bi_otime_hi               = 0,
    bi_size                   = 0,
    bi_sectors                = 0,
    bi_uid                    = 0,
    bi_gid                    = 0,
    bi_nlink                  = 7,
    bi_generation             = 0,
    bi_dev                    = 0,
    bi_data_checksum          = 0,
    bi_compression            = 0,
    bi_project                = 0,
    bi_background_compression = 0,
    bi_data_replicas          = 0,
    bi_promote_target         = 0,
    bi_foreground_target      = 0,
    bi_background_target      = 0,
    bi_erasure_code           = 0,
    bi_fields_set             = 0,
    bi_dir                    = 0,
    bi_dir_offset             = 0,
    bi_subvol                 = 0,
    bi_parent_subvol          = 0,
}

p1 =     "\x0d\x67\x72\xd7\x96\x5c\x86\x49" -- bch_inode{ hash_seed=0x49865c96d772670d,
p1 = p1.."\x00\x00\x30\x89\xed\x41\xf7\x17" --   flags=0x89300000 (new_varint, 9 fields, siphash), mode=0x41ed=40755
                                            --   VARINT LIST:
p1 = p1.."\xc8\x30\x00\xdf\x75\xf1\x05\x69" --     [f7 17 c8 30] [00]         atime
                                            --     [df 75 f1 05 69 06] [00]   ctime
p1 = p1.."\x06\x00\xdf\x75\xf1\x05\x69\x06" --     [df 75 f1 05 69 06] [00]   mtime
p1 = p1.."\x00\xf7\x17\xc8\x30\x00\x00\x00" --     [f7 17 c8 30] [00]         otime
                                            --     [00]                       size
                                            --     [00]                       sectors
p1 = p1.."\x00\x00\x0e\x00\x00\x00\x00\x00" --     [00]                       uid
                                            --     [00]                       gid
                                            --     [0e]                       nlink
                                            --     (not present)              generation, dev, data_checksum, ...
                                            --     (not present)              ..., dir, dir_offset, subvol, parent_subvol
                                            -- }

u1 = bcachefs.inode_unpack(p1)

for k, v in pairs(t1) do
  tstmessagetap(string.format('bcachefs inode 4096 unpack root (%s)', k:sub(4)), u1[k]==t1[k])
end

v1 = bcachefs.inode_unpack_size(p1)
tstmessagetap('bcachefs inode 4096 unpack root (size, fast)', v1 == t1.bi_size)



--[[
BcacheFS Dirent 4096 Seek

The root directory (inode 4096) on the test filesystem contains the following
dirents (directory entries):

    u64s 7 type dirent 4096:1356932970024394392:U32_MAX len 0 ver 0: file1 -> 4111 type reg
    u64s 8 type dirent 4096:1402452081417787627:U32_MAX len 0 ver 0: lost+found -> 4097 type dir
    u64s 8 type dirent 4096:2721615440840922396:U32_MAX len 0 ver 0: n09332890 -> 4109 type dir
    u64s 7 type dirent 4096:2786330024448471437:U32_MAX len 0 ver 0: dir -> 4098 type dir
    u64s 8 type dirent 4096:7040298706587502813:U32_MAX len 0 ver 0: n04467665 -> 4105 type dir
    u64s 8 type dirent 4096:7166071670834426273:U32_MAX len 0 ver 0: n02033041 -> 4101 type dir
    u64s 8 type dirent 4096:7611033946105185279:U32_MAX len 0 ver 0: n02445715 -> 4103 type dir
    u64s 8 type dirent 4096:8468282947345244644:U32_MAX len 0 ver 0: n04584207 -> 4107 type dir

We wish to seek to "n09332890". To obtain the "offset" of the dirent in the
root directory under the dirent B+tree, we hash the subdirectory name with
SipHash-2-4, using the containing directory's bi_hash_seed as the key.

Seeking to that offset, one obtains a dirent with a name that is indeed
"n09332890" and that points to inode 4109, a directory.
--]]

d1 = siphash("n09332890", t1.bi_hash_seed, 0) >> 1
tstmessagetap('bcachefs dirent 4096 seek / -> 4109 /n09332890', d1 == 2721615440840922396)



--[[
BcacheFS Inode 4109 Unpack

Directory inode (4109) of a test filesystem. Corresponds to the directory "/n09332890".
Packed length: 9x u64s w/ pad.
--]]

t2 = {
    -- Fixed:
    bi_hash_seed              = 0x895b187cf867b13b,
    bi_flags                  = 0x97300000,
    bi_mode                   = 0x41ed, -- octal 40755, dir drwxr-xr-x
    -- Varint v2:
    bi_atime                  = 110123203652,
    bi_atime_hi               = 0,
    bi_ctime                  = 110123203652,
    bi_ctime_hi               = 0,
    bi_mtime                  = 110123203652,
    bi_mtime_hi               = 0,
    bi_otime                  = 110123203652,
    bi_otime_hi               = 0,
    bi_size                   = 0,
    bi_sectors                = 0,
    bi_uid                    = 0,
    bi_gid                    = 0,
    bi_nlink                  = 0,
    bi_generation             = 0,
    bi_dev                    = 0,
    bi_data_checksum          = 0,
    bi_compression            = 0,
    bi_project                = 0,
    bi_background_compression = 0,
    bi_data_replicas          = 0,
    bi_promote_target         = 0,
    bi_foreground_target      = 0,
    bi_background_target      = 0,
    bi_erasure_code           = 0,
    bi_fields_set             = 0,
    bi_dir                    = 4096,
    bi_dir_offset             = 2721615440840922396,
    bi_subvol                 = 0,
    bi_parent_subvol          = 0,
}

p2 =     "\x3b\xb1\x67\xf8\x7c\x18\x5b\x89" -- bch_inode{ hash_seed=0x895b187cf867b13b,
p2 = p2.."\x00\x00\x30\x97\xed\x41\x1f\x11" --   flags=0x97300000 (new_varint, 23 fields, siphash), mode=0x41ed=40755
                                            --   VARINT LIST:
p2 = p2.."\xaf\xf6\x68\x06\x00\x1f\x11\xaf" --     [1f 11 af f6 68 06] [00]     atime
                                            --     [1f 11 af f6 68 06] [00]     ctime
p2 = p2.."\xf6\x68\x06\x00\x1f\x11\xaf\xf6" --     [1f 11 af f6 68 06] [00]     mtime
p2 = p2.."\x68\x06\x00\x1f\x11\xaf\xf6\x68" --     [1f 11 af f6 68 06] [00]     otime
p2 = p2.."\x06\x00\x00\x00\x00\x00\x00\x00" --     [00]                         size
                                            --     [00]                         sectors
                                            --     [00]                         uid
                                            --     [00]                         gid
                                            --     [00]                         nlink
                                            --     [00]                         generation
p2 = p2.."\x00\x00\x00\x00\x00\x00\x00\x00" --     [00]                         dev
                                            --     [00]                         data_checksum
                                            --     [00]                         compression
                                            --     [00]                         project
                                            --     [00]                         background_compression
                                            --     [00]                         data_replicas
                                            --     [00]                         promote_target
                                            --     [00]                         foreground_target
p2 = p2.."\x00\x00\x00\x01\x40\xff\x1c\x95" --     [00]                         background_target
                                            --     [00]                         erasure_code
                                            --     [00]                         fields_set
                                            --     [01 40]                      dir
                                            --     [ff 1c 95 8a de d2 1e c5 25] dir_offset
p2 = p2.."\x8a\xde\xd2\x1e\xc5\x25\x00\x00" --     (not present)                subvol
                                            --     (not present)                parent_subvol
                                            -- }

u2 = bcachefs.inode_unpack(p2)

for k, v in pairs(t2) do
  tstmessagetap(string.format('bcachefs inode 4109 unpack dir (%s)', k:sub(4)), u2[k]==t2[k])
end

v2 = bcachefs.inode_unpack_size(p2)
tstmessagetap('bcachefs inode 4109 unpack dir (size, fast)',      t2.bi_size       == v2)
tstmessagetap('bcachefs inode 4109 verify backlink (dir)',        t2.bi_dir        == 4096)
tstmessagetap('bcachefs inode 4109 verify backlink (dir_offset)', t2.bi_dir_offset == d1)



--[[
BcacheFS Dirent 4109 Seek

The subdirectory /n09332890 (inode 4109) on the test filesystem contains the
following dirents (directory entries):

    u64s 9 type dirent 4109:1788158112371519905:U32_MAX len 0 ver 0: n09332890_29876.JPEG -> 4110 type reg

We wish to seek to "n09332890_29876.JPEG". To obtain the "offset" of the
dirent in the directory under the dirent B+tree, we hash the file name
with SipHash-2-4, using the containing directory's bi_hash_seed as the key.

Seeking to that offset, one obtains a dirent with a name that is indeed
"n09332890_29876.JPEG" and that points to inode 4110, a regular file.
--]]

d2 = siphash("n09332890_29876.JPEG", t2.bi_hash_seed, 0) >> 1
tstmessagetap('bcachefs dirent 4109 seek /n09332890 -> 4110 /n09332890/n09332890_29876.JPEG',
              d2 == 1788158112371519905)



--[[
BcacheFS Inode Unpack Test 3

Example inode 4110 from the test filesystem. Corresponds to a regular file
"n09332890_29876.JPEG" of 144761 bytes. Packed length: 8x u64s w/ pad.
Unpacks to:
--]]

t3 = {
    -- Fixed:
    bi_hash_seed              = 0xcff099609d9f8b9c,
    bi_flags                  = 0x97300000,
    bi_mode                   = 0x81a4, -- octal 100644, reg -rw-r--r--
    -- Varint v2:
    bi_atime                  = 110123203652,
    bi_atime_hi               = 0,
    bi_ctime                  = 110127203799,
    bi_ctime_hi               = 0,
    bi_mtime                  = 110127203799,
    bi_mtime_hi               = 0,
    bi_otime                  = 110123203652,
    bi_otime_hi               = 0,
    bi_size                   = 144761,
    bi_sectors                = 288,
    bi_uid                    = 0,
    bi_gid                    = 0,
    bi_nlink                  = 0,
    bi_generation             = 0,
    bi_dev                    = 0,
    bi_data_checksum          = 0,
    bi_compression            = 0,
    bi_project                = 0,
    bi_background_compression = 0,
    bi_data_replicas          = 0,
    bi_promote_target         = 0,
    bi_foreground_target      = 0,
    bi_background_target      = 0,
    bi_erasure_code           = 0,
    bi_fields_set             = 0,
    bi_dir                    = 4109,
    bi_dir_offset             = 1788158112371519905,
    bi_subvol                 = 0,
    bi_parent_subvol          = 0,
}

p3 =     "\x9c\x8b\x9f\x9d\x60\x99\xf0\xcf" -- bch_inode{ hash_seed=0xcff099609d9f8b9c,
p3 = p3.."\x00\x00\x30\x97\xa4\x81\x1f\x11" --   flags=0x97300000 (new_varint, 23 fields, siphash), mode=0x81a4=100644
                                            --   VARINT LIST:
p3 = p3.."\xaf\xf6\x68\x06\x00\xdf\x75\xf1" --     [1f 11 af f6 68 06] [00]     atime
p3 = p3.."\x05\x69\x06\x00\xdf\x75\xf1\x05" --     [df 75 f1 05 69 06] [00]     ctime
p3 = p3.."\x69\x06\x00\x1f\x11\xaf\xf6\x68" --     [1f 11 af f6 68 06] [00]     mtime
p3 = p3.."\x06\x00\xcb\xab\x11\x81\x04\x00" --     [1f 11 af f6 68 06] [00]     otime
                                            --     [cb ab 11]                   size
                                            --     [81 04]                      sectors
                                            --     [00]                         uid
p3 = p3.."\x00\x00\x00\x00\x00\x00\x00\x00" --     [00]                         gid
                                            --     [00]                         nlink
                                            --     [00]                         generation
                                            --     [00]                         dev
                                            --     [00]                         data_checksum
                                            --     [00]                         compression
                                            --     [00]                         project
                                            --     [00]                         background_compression
p3 = p3.."\x00\x00\x00\x00\x00\x00\x35\x40" --     [00]                         data_replicas
                                            --     [00]                         promote_target
                                            --     [00]                         foreground_target
                                            --     [00]                         background_target
                                            --     [00]                         erasure_code
                                            --     [00]                         fields_set
                                            --     [35 40]                      dir
p3 = p3.."\xff\xa1\x31\x23\x42\x54\xd0\xd0" --     [ff a1 31 23 42 54 d0 d0 18] dir_offset
p3 = p3.."\x18\x00\x00\x00\x00\x00\x00\x00" --     (not present)                subvol
                                            --     (not present)                parent_subvol
                                            -- }

u3 = bcachefs.inode_unpack(p3)

for k, v in pairs(t3) do
  tstmessagetap(string.format('bcachefs inode 4110 unpack regular (%s)', k:sub(4)), u3[k]==t3[k])
end

v3 = bcachefs.inode_unpack_size(p3)
tstmessagetap('bcachefs inode 4110 unpack regular (size, fast)',  t3.bi_size       == v3)
tstmessagetap('bcachefs inode 4110 verify backlink (dir)',        t3.bi_dir        == 4109)
tstmessagetap('bcachefs inode 4110 verify backlink (dir_offset)', t3.bi_dir_offset == d2)
