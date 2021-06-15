#
# Library: libbenzina
#
# Contains a lot of the "generic" logic for data-loading, as well as ProtoBuf
# parsing code.
#

libbenzina_dict     = {
    'c_args':     ['-DBENZINA_IS_SHARED=1', '-DBENZINA_IS_BUILDING=1'],
    'link_args':  ['-pie', '-Wl,-E'],
    'link_with':  [],
    'link_whole': [],
    'link_deps':  [],
    'objs': [],
    'deps': [
        threads, libm, libdl,
        project_linenoise.get_variable('linenoise_dep'),
        project_lua      .get_variable('custom_lua_dep'),
        project_lfs      .get_variable('lfs_dep'),
    ],
    'incs': [benzina_incs],
    'srcs': files('benzina.c',
                  'benzinaproto.c',
                  'endian.c',
                  'init.c',
                  'intops.c',
                  'ptrops.c',
                  'version.c',
                  'main.c')
}
subdir('iso')
subdir('itu')


#
# Add Linker Version Script.
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
  libbenzina_vers_name = 'libbenzina.version'
  libbenzina_vers_path = meson.current_source_dir() / 'libbenzina.version'
  libbenzina_dict += {
      'link_args': libbenzina_dict['link_args'] + [f'-Wl,--version-script=@libbenzina_vers_path@'],
      'link_deps': libbenzina_dict['link_deps'] + files(libbenzina_vers_name),
  }
endif


#
# Build the shared library, patch it, then declare a dependency.
#
libbenzina_sha = shared_library(
    'benzina', libbenzina_dict['srcs'],
    objects:               libbenzina_dict['objs'],
    include_directories:   libbenzina_dict['incs'],
    dependencies:          libbenzina_dict['deps'],
    c_args:                libbenzina_dict['c_args'],
    link_args:             libbenzina_dict['link_args'],
    link_with:             libbenzina_dict['link_with'],
    link_whole:            libbenzina_dict['link_whole'],
    link_depends:          libbenzina_dict['link_deps'],
    gnu_symbol_visibility: 'hidden',
    version:               meson.project_version(),
    build_rpath:           librpath,
    install_rpath:         librpath,
    install_dir:           libdir,
    install:               true,
)
libbenzina_sha = custom_target(
    'patched libbenzina',
    command: [python, elf_strip_df_1_pie_py, '@INPUT0@', '@OUTPUT0@'],
    input:   [libbenzina_sha],
    output:  'libbenzina-patched.so',
    install: false,
    depend_files: [elf_py],
    build_by_default: true,
)
libbenzina = declare_dependency(
    include_directories: benzina_incs,
    compile_args:        ['-DBENZINA_IS_SHARED=1'],
    link_with:           libbenzina_sha,
)

