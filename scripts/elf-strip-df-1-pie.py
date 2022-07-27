#!/usr/bin/env python
import shutil, os, sys
import elf

def strip_df_1_pie(f):
    min_phentsize, dynfmt, dynsz = \
        (56, 'QQ', 16) if f.elf_class_64 else (32, 'II',  8)
    
    assert f.e_type      == elf.ET_DYN
    assert f.e_version   == elf.EV_CURRENT
    assert f.e_phoff     >  0
    assert f.e_phentsize >= min_phentsize
    assert f.e_phnum     in range(1, 2**16-1)
    
    for off, phdr in f.iterphdr():
        if phdr.p_type == elf.PT_DYNAMIC:
            for doff in range(phdr.p_offset, phdr.p_offset+phdr.p_filesz, dynsz):
                d_tag, d_val = f.rdfmt(dynfmt, doff)
                if   d_tag == elf.DT_NULL:
                    break
                elif d_tag == elf.DT_FLAGS_1:
                    if d_val & elf.DF_1_PIE:
                        f.wrfmt(doff, dynfmt, d_tag, d_val & ~elf.DF_1_PIE)
                    break
            break

if __name__ == "__main__":
    path_benzina_unpatched   = sys.argv[1] # Meson @INPUT0@
    path_libbenzina_so_X_Y_Z = sys.argv[2] # Meson @OUTPUT0@
    
    if(os.path.islink (path_libbenzina_so_X_Y_Z) or
       os.path.lexists(path_libbenzina_so_X_Y_Z)):
        os.unlink     (path_libbenzina_so_X_Y_Z)
    
    shutil.copy2(path_benzina_unpatched, path_libbenzina_so_X_Y_Z)
    
    with elf.ELF(path_libbenzina_so_X_Y_Z) as f:
        strip_df_1_pie(f)
