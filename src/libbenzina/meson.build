#
# Library: libbenzina
#
# Contains a lot of the "generic" logic for data-loading, as well as ProtoBuf
# parsing code.
#
libbenzina_name      = 'benzina'
libbenzina_libname   = 'lib'+libbenzina_name
libbenzina_linkname  = libbenzina_libname+'.so'
libbenzina_soversion = meson.project_version().split('.')[0]
libbenzina_soname    = libbenzina_linkname+'.'+libbenzina_soversion
libbenzina_sonamearg = IS_HOST_DARWIN ? [] : ['-Wl,-soname,'+libbenzina_soname]
libbenzina_lua       = {}
libbenzina_lua      += project_lpeg    .get_variable('lpeg_lua')
libbenzina_lua      += project_penlight.get_variable('penlight_lua')
libbenzina_dict      = {
    'c_args':     ['-DBENZINA_IS_SHARED=1', '-DBENZINA_IS_BUILDING=1'],
    'link_args':  libbenzina_sonamearg+force_link_args+special_link_args,
    'link_with':  [],
    'link_whole': [],
    'link_deps':  [],
    'objs': [],
    'deps': [
        threads, libm, libdl,
        project_lz4      .get_variable('lz4_dep'),
        project_linenoise.get_variable('linenoise_dep'),
        project_lua      .get_variable('custom_lua_dep'),
        project_lfs      .get_variable('lfs_dep'),
        project_lpeg     .get_variable('lpeg_dep'),
        project_penlight .get_variable('penlight_dep'),
    ],
    'incs': [benzina_incs],
    'srcs': files(
        'benzina.c',
        'benzinaproto.c',
        'endian.c',
        'init.c',
        'intops.c',
        'ptrops.c',
        'version.c',
        'internal.noarch.S',
        'main.c',
    ),
}
subdir('iso')
subdir('itu')


#
# Pack Lua Code.
#
# Translate the libbenzina_lua dictionary of {package: files} to a command-line
# invocation of the packer script, which uses a native-built lz4 executable to
# compress the Lua code into an LZ4 frame. At runtime, the constructor and
# library initializer unpack themselves into statically-allocated .bss space.
#
# The packer is invoked as:
#     python lz4-lua-pack.py
#            build/path/to/lz4
#            out.S
#            out.lz4
#            [packagename, packagesource.lua]*
#
# While the LZ4 CLI tool itself is GPLv2, the liblz4.a library is BSD2 licensed
# and its output obviously is not licensed. Because the CLI tool is only used
# at build time and not incorporated into the project (whether statically or
# dynamically), we do not guard this portion of the build using ENABLE_GPL.
#

lua_input_list = []
foreach name, file: libbenzina_lua
  lua_input_list += [name, file]
endforeach
if lua_input_list.length() > 0
  packed_lua = custom_target(
      'LZ4-packed Lua code',
      command: [python, lz4_lua_pack_py, project_lz4.get_variable('lz4_exe_native'),
                '@OUTPUT0@', '@OUTPUT1@'] + lua_input_list,
      output:  ['packed-lua.S', 'packed-lua.lz4'],
  )
  libbenzina_dict += {'srcs': libbenzina_dict['srcs'] + [packed_lua[0]]}
endif


#
# Add Linker Script and Version Script.
#
# Linker version scripts are passed through to ld, and can modify the
# visibility of symbols.
#
# In Meson, it is not easy to add a linker version script to the build:
#
#   - With GNU ld.bfd, it is possible to pass a full, *implicit* linker script
#     with a VERSION clause to the linker as if it were an "object file".
#   - With the PGI 19.4 compiler's linker, this fails because a VERSION clause
#     is always included in the builtin script, and the combination of all
#     linker scripts may only declare at most one such section.
#   - With Google's ld.gold, a full implicit linker script is rejected, whereas
#     the more limited --version-script works=.
#   - With LLVM lld/ld.lld, only a subset of GNU ld.bfd's linker script syntax
#     is supported, but the more limited --version-script= works.
#
# On balance, therefore, it is preferable to use the --version-script= flag
# passed to the linker directly. Unfortunately, Meson does not support auto-
# formatting files() as strings yet, and link_args must be an array of strings.
# So we go about it the hard way - get the current source directory and
# construct the full absolute path to the linker version script, and add that
# manually as an implicit link (though not compile) dependency.
#

if cc.get_linker_id() in ['ld.bfd', 'ld.gold', 'ld.lld']
  libbenzina_lds_name = 'libbenzina.lds'
  libbenzina_ver_name = 'libbenzina.ver'
  libbenzina_lds_path = meson.current_source_dir() / libbenzina_lds_name
  libbenzina_ver_path = meson.current_source_dir() / libbenzina_ver_name
  libbenzina_lds_file = files(libbenzina_lds_name)
  libbenzina_ver_file = files(libbenzina_ver_name)
  libbenzina_dict += {
      'link_args': libbenzina_dict['link_args'] + [
          '-Wl,-T,'+libbenzina_lds_path,
          '-Wl,--version-script='+libbenzina_ver_path,
      ],
      'link_deps': libbenzina_dict['link_deps'] + [
          libbenzina_lds_file,
          libbenzina_ver_file,
      ],
  }
endif


#
# Build the sources of the shared library as static-pic, relink the static
# library as shared, patch it, symlink it, then declare a dependency.
#
# The reasons for this contribed process are manifold:
# 
#    - Since libbenzina.so is meant to *also* be a PIE executable, we use
#      executable() here, as the alternative through shared_library() runs into
#      trouble with Clang. Clang ignores the -pie flag after -shared has been
#      selected, whereas GCC accepts both.
#
#    - However, when compiling PIE executables (as opposed to PIC libraries),
#      Meson sets the -fPIE flag on sources. When this flag is used instead of
#      -fPIC, it allows the compiler to use inappropriate relocations that
#      translate into errors such as
#          init.c.o: warning: relocation against `stderr@@GLIBC_2.2.5' in
#                             read-only section `.text.startup'
#          ld: read-only segment has dynamic relocations
#      This is particularly likely with GCC. The solution is that all
#      *source files* *must* be gated through a Meson static_library(pic:true).
#
#    - But this is still not enough. The linker unconditionally sets an ELF
#      flag DF_1_PIE on PIE executables, which the GNU C library >= 2.30 checks
#      for to block their dlopen()'ing (this may be justified when thread-local
#      storage or copy relocations are used, but neither is a concern for us).
#      We choose to manually patch out that flag using a Python script. This
#      script modifies its input *in-place* for convenience's sake and also
#      generates the .so and .so.X symlinks.
#
# As a result of all this, we manage to declare libbenzina_sha with every type
# (static_library, executable, custom_target, declare_dependency) except the
# one we might truly have intended (shared_library).
#
if IS_HOST_DARWIN
  libbenzina_sha = shared_library(
      libbenzina_name,       libbenzina_dict['srcs'],
      include_directories:   libbenzina_dict['incs'],
      c_args:                libbenzina_dict['c_args'],
      dependencies:          libbenzina_dict['deps'],
      objects:               libbenzina_dict['objs'],
      link_args:             libbenzina_dict['link_args'],
      link_with:             libbenzina_dict['link_with'],
      link_whole:            libbenzina_dict['link_whole'],
      link_depends:          libbenzina_dict['link_deps'],
      gnu_symbol_visibility: 'hidden',
      build_rpath:           librpath,
      install_rpath:         librpath,
      install_dir:           libdir,
      install:               true,
  )
  libbenzina = declare_dependency(
      include_directories: benzina_incs,
      compile_args:        ['-DBENZINA_IS_SHARED=1'],
      link_with:           libbenzina_sha,
  )
  subdir_done()
endif
libbenzina_sha = declare_dependency(dependencies: libbenzina_dict['deps'])
libbenzina_sha = libbenzina_sha.partial_dependency(compile_args: true, includes: true)
libbenzina_sha = static_library( # *STATIC-PIC*! SEE NOTES ABOVE!
    libbenzina_libname,    libbenzina_dict['srcs'],
    include_directories:   libbenzina_dict['incs'],
    c_args:                libbenzina_dict['c_args'],
    dependencies:          libbenzina_sha,
    gnu_symbol_visibility: 'hidden',
    pic:                   true,
    install:               false,
)
libbenzina_sha = declare_dependency(dependencies: libbenzina_dict['deps'],
                                    link_whole:   libbenzina_sha)
libbenzina_sha = libbenzina_sha.partial_dependency(link_args: true, links: true)
libbenzina_sha = executable(
    libbenzina_libname,    [],   # *MUST BE EMPTY*! SEE NOTES ABOVE!
    objects:               libbenzina_dict['objs'],
    dependencies:          libbenzina_sha,
    link_args:             libbenzina_dict['link_args'],
    link_with:             libbenzina_dict['link_with'],
    link_whole:            libbenzina_dict['link_whole'],
    link_depends:          libbenzina_dict['link_deps'],
    name_suffix:           'so.'+meson.project_version(),
    gnu_symbol_visibility: 'hidden',
    export_dynamic:        true,
    pie:                   true,
    build_rpath:           librpath,
    install_rpath:         librpath,
    install_dir:           libdir,
    install:               true,
)
libbenzina_sha = custom_target(  # MODIFIES IN-PLACE!
    'patched '+libbenzina_libname,
    command:          [python, elf_strip_df_1_pie_py, '@INPUT0@', '@OUTPUT@'],
    input:            [libbenzina_sha],
    output:           [libbenzina_soname, libbenzina_linkname],
    depend_files:     [elf_py],
    build_by_default: true,
    install_dir:      libdir,
    install:          true,
)
libbenzina_sha = libbenzina_sha[1]
libbenzina = declare_dependency(
    include_directories: benzina_incs,
    compile_args:        ['-DBENZINA_IS_SHARED=1'],
    link_with:           libbenzina_sha,
)

