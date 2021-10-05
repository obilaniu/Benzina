--[[
BcacheFS tests, using the Test Anything Protocol (TAP)
--]]

print('1..31')


-- Small utility function to save typing while defining tests
function tstmessagetap(name, cond)
  if cond then print('ok '..name)
  else         print('not ok '..name)
  end
end


bcachefs = require 'benz.bcachefs'


--[[
BcacheFS Inode Unpack Test 1

Example inode 4110 from a test filesystem. Corresponds to a regular file
"n09332890_29876.JPEG" of 144761 bytes. Packed length: 8x u64s w/ pad.
Unpacks to:
--]]

t = {
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
}

p =    "\x9c\x8b\x9f\x9d\x60\x99\xf0\xcf" -- bch_inode{ hash_seed=0xcff099609d9f8b9c,
p = p.."\x00\x00\x30\x97\xa4\x81\x1f\x11" --   flags=0x97300000 (new_varint, 23 fields, siphash), mode=0x81a4=100644
                                          --   VARINT LIST:
p = p.."\xaf\xf6\x68\x06\x00\xdf\x75\xf1" --     [1f 11 af f6 68 06] [00]     atime
p = p.."\x05\x69\x06\x00\xdf\x75\xf1\x05" --     [df 75 f1 05 69 06] [00]     ctime
p = p.."\x69\x06\x00\x1f\x11\xaf\xf6\x68" --     [1f 11 af f6 68 06] [00]     mtime
p = p.."\x06\x00\xcb\xab\x11\x81\x04\x00" --     [1f 11 af f6 68 06] [00]     otime
                                          --     [cb ab 11]                   size
                                          --     [81 04]                      sectors
                                          --     [00]                         uid
p = p.."\x00\x00\x00\x00\x00\x00\x00\x00" --     [00]                         gid
                                          --     [00]                         nlink
                                          --     [00]                         generation
                                          --     [00]                         dev
                                          --     [00]                         data_checksum
                                          --     [00]                         compression
                                          --     [00]                         project
                                          --     [00]                         background_compression
p = p.."\x00\x00\x00\x00\x00\x00\x35\x40" --     [00]                         data_replicas
                                          --     [00]                         promote_target
                                          --     [00]                         foreground_target
                                          --     [00]                         background_target
                                          --     [00]                         erasure_code
                                          --     [00]                         fields_set
                                          --     [35 40]                      dir
p = p.."\xff\xa1\x31\x23\x42\x54\xd0\xd0" --     [ff a1 31 23 42 54 d0 d0 18] dir_offset
p = p.."\x18\x00\x00\x00\x00\x00\x00\x00" --  }

u = bcachefs.inode_unpack(p)

for k, v in pairs(t) do
  tstmessagetap(string.format('bcachefs inode unpack regular 1 (%s)', k:sub(4)), u[k]==t[k])
end

v = bcachefs.inode_unpack_size(p)
tstmessagetap('bcachefs inode unpack regular 1 (bi_size, fast)', v == t.bi_size)
