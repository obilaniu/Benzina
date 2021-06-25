.section .license, "a"
    .ascii  "LPeg license:\n\n"
    .incbin "./LICENSE.md"
    .ascii  "\n\n"

.section .rodata.str1.1, "aMS", @progbits, 1
.Llpeg_module_name:
    .asciz "lpeg"

.section .lua.open_array, "aw"
    GASPOINTERWORD .Llpeg_module_name, luaopen_lpeg
