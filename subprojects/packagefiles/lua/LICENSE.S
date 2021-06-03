.section .rodata.license, "aS"
.globl lua_license
lua_license:
    .incbin "./LICENSE.md"
    .byte 0
