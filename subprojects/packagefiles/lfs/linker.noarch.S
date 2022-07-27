ASM_SECT_LICENSE_DECL
    .ascii  "luafilesystem license:\n\n"
    .incbin "./LICENSE"
    .ascii  "\n\n"

ASM_SECT_CSTRING_DECL
.Llfs_module_name:
    .asciz "lfs"

ASM_SECT_LUAOPENARRAY_DECL
#if BENZINA_OS_DARWIN
    ASM_PTR_DECL .Llfs_module_name, _luaopen_lfs
#else
    ASM_PTR_DECL .Llfs_module_name, luaopen_lfs
#endif
