.section .rodata.license, "aS"
.globl linenoise_license
linenoise_license:
    .incbin "./LICENSE"
    .byte 0
