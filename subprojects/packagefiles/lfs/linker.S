#if   __SIZEOF_POINTER__ == 4
# define POINTERWORD .4byte
#elif __SIZEOF_POINTER__ == 8
# define POINTERWORD .8byte
#else
# error "Adapt macro for strange pointer size!"
#endif


.section .license, "a"
    .ascii  "luafilesystem license:\n\n"
    .incbin "./LICENSE"
    .ascii  "\n\n"

.section .rodata.str1.1, "aMS", @progbits, 1
.Llfs_module_name:
    .asciz "lfs"

.section .lua.open_array.preload, "aw"
    POINTERWORD .Llfs_module_name, luaopen_lfs
