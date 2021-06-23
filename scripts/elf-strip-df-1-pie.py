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
    path_libbenzina_so_X_Y_Z = sys.argv[1] # Meson @INPUT0@  (*)
    path_libbenzina_so_X     = sys.argv[2] # Meson @OUTPUT0@
    path_libbenzina_so       = sys.argv[3] # Meson @OUTPUT1@
    libbenzina_so_X_Y_Z      = os.path.basename(path_libbenzina_so_X_Y_Z)
    libbenzina_so_X          = os.path.basename(path_libbenzina_so_X)
    
    #
    # (*)
    #
    # Yes, this script does a really nasty thing. It modifies its input argv[1]
    # *first* and *in-place*, *then* creates "unrelated" symlinks.
    # This was done because there is no easier way to modify within Meson a
    # shared library after it has been build and without losing the ability
    # to use shared_library() to construct it.
    #
    with elf.ELF(path_libbenzina_so_X_Y_Z) as f:
        strip_df_1_pie(f)
    
    #
    # Force-create/overwrite output symlinks:
    #     libbenzina.so.X.Y.Z <- libbenzina.so.X
    #     libbenzina.so.X     <- libbenzina.so
    #
    def overwrite_symlink(src, dst):
        if os.path.islink(dst) and os.readlink(dst) == src: return
        if os.path.exists(dst):    os.unlink(dst)
        os.symlink(src, dst)
    
    overwrite_symlink(libbenzina_so_X_Y_Z, path_libbenzina_so_X)
    overwrite_symlink(libbenzina_so_X,     path_libbenzina_so)
