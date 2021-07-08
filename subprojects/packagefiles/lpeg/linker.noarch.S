ASM_SECT_LICENSE_DECL
    .ascii  "LPeg license:\n\n"
    .incbin "./LICENSE.md"
    .ascii  "\n\n"

ASM_SECT_CSTRING_DECL
.Llpeg_module_name:
    .asciz "lpeg"

ASM_SECT_LUAOPENARRAY_DECL
#if BENZINA_OS_DARWIN
    ASM_PTR_DECL .Llpeg_module_name, _luaopen_lpeg
#else
    ASM_PTR_DECL .Llpeg_module_name, luaopen_lpeg
#endif
