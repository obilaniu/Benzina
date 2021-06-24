#!/usr/bin/env python
import os, sys, subprocess


if __name__ == "__main__":
    lz4     = sys.argv[1]
    out_S   = sys.argv[2] # Meson @OUTPUT0@
    out_lz4 = sys.argv[3] # Meson @OUTPUT1@
    D = {name: file for name, file in zip(sys.argv[4::2], sys.argv[5::2])}
    
    sortednames = sorted(D.keys())
    
    payload_unc = bytearray()
    with open(out_S, "w") as f:
        f.write('.section .rodata.str1.1, "aMS", @progbits, 1\n')
        for i, name in enumerate(sortednames):
            f.write(f'    .Lname{i}: .asciz "{name}"\n')
        f.write('\n\n')
        f.write('.section .bss\n')
        f.write('.align 4096\n')
        f.write('.Ldst_start:\n')
        for i, name in enumerate(sortednames):
            filecontents = open(D[name], "rb").read()
            payload_unc += bytearray(filecontents + b'\x00')
            f.write(f'.Lstart{i}: .skip {len(filecontents)}\n')
            f.write(f'.Lend{i}:   .skip 1\n')
        f.write('.Ldst_end:\n')
        f.write('    .skip 8\n')
        f.write('.align 4096\n')
        f.write('\n\n')
        f.write('.section .lua.text_array, "aw", @progbits\n')
        for i, name in enumerate(sortednames):
            f.write(f'    GASPOINTERWORD .Lname{i}, .Lstart{i}, .Lend{i}\n')
        f.write('\n\n')
        
        #
        # Execute compression now to hard-code file size into GAS!
        #
        subprocess.run(
            [lz4, '-f', '--best', '-', out_lz4],
            input  = payload_unc,
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
            check  = True,
        )
        incbinpath = os.path.relpath(out_lz4, os.path.dirname(out_S))
        incbinsize = os.stat(out_lz4, follow_symlinks=True).st_size
        
        f.write('.section .rodata.lz4, "a", @progbits\n')
        f.write('.Lsrc_start:\n')
        f.write(f'    .incbin "{incbinpath}", 0, {incbinsize}\n')
        f.write('.Lsrc_end:\n')
        f.write('    .8byte 0\n')
        f.write('\n\n')
        f.write('.section .lz4.decompress_array, "aw", @progbits\n')
        f.write('    GASPOINTERWORD .Lsrc_start, .Lsrc_end, .Ldst_start, .Ldst_end\n')
