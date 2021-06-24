.section .license, "a"
    .ascii  "luafilesystem license:\n\n"
    .incbin "./LICENSE"
    .ascii  "\n\n"

.section .rodata.str1.1, "aMS", @progbits, 1
.Llfs_module_name:
    .asciz "lfs"

.section .lua.open_array.preload, "aw"
    GASPOINTERWORD .Llfs_module_name, luaopen_lfs
