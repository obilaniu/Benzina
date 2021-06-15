#!/usr/bin/env python
import shutil, sys
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
    #
    # Yes, this script does a really nasty thing. It modifies its input argv[1]
    # *first* and *in-place*, *then* copies it to the output argv[2].
    # This was done because there is no easier way to modify within Meson a
    # shared library after it has been build and without losing the ability
    # to use shared_library() to construct it.
    #
    with elf.ELF(sys.argv[1]) as f:
        strip_df_1_pie(f)
    try:
        shutil.copy2(sys.argv[1], sys.argv[2])
    except shutil.SameFileError:
        pass
