#
# Library: libbenzina
#
# Contains a lot of the "generic" logic for data-loading, as well as ProtoBuf
# parsing code.
#
libbenzina_name      = 'benzina'
libbenzina_libname   = 'lib'+libbenzina_name
libbenzina_linkname  = libbenzina_libname+'.so'
libbenzina_fullname  = libbenzina_linkname+'.'+meson.project_version()
libbenzina_unpatched = libbenzina_fullname+'.unpatched'
libbenzina_soversion = meson.project_version().split('.')[0]
libbenzina_soname    = libbenzina_linkname+'.'+libbenzina_soversion
libbenzina_sonamearg = IS_HOST_DARWIN ? [] : ['-Wl,-soname,'+libbenzina_soname]
libbenzina_link_args = libbenzina_sonamearg+force_link_args+special_link_args
libbenzina_c_args    = ['-DBENZINA_IS_SHARED=1']
libbenzina_incs      = [benzina_incs]
libbenzina_srcs      = []
libbenzina_lua_srcs  = {}
libbenzina_lua_srcs += project_lpeg.get_variable('lpeg_lua')
libbenzina_link_only = []
libbenzina_deps      = [
    threads, dl, libm,
    project_lz4      .get_variable('lz4_dep'),
    project_linenoise.get_variable('linenoise_dep'),
    project_lua      .get_variable('custom_lua_dep'),
    project_lfs      .get_variable('lfs_dep'),
    project_lpeg     .get_variable('lpeg_dep'),
]
libbenzina_srcs += files(
    'benzina.c',
    'benzinaproto.c',
    'endian.c',
    'init.c',
    'intops.c',
    'ptrops.c',
    'siphash.c',
    'version.c',
    'internal.noarch.S',
    'main.c',
)
subdir('iso')
subdir('itu')
subdir('bcachefs')


#
# Pack Lua Code.
#
# Translate the Lua source dictionary of {package: files} to a command-line
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
foreach name, file: libbenzina_lua_srcs
  lua_input_list += [name, file]
endforeach
if lua_input_list.length() > 0
  lz4_exe = project_lz4.get_variable('lz4_exe_native')
  packed_lua = custom_target('LZ4-packed Lua code',
      command: [python, lz4_lua_pack_py, lz4_exe, '@OUTPUT0@', '@OUTPUT1@'] + lua_input_list,
      output:  ['packed-lua.S', 'packed-lua.lz4'],
  )
  libbenzina_srcs += [packed_lua[0]] # .S only! It includes the .lz4!
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
#     with a VERSION clause to the linker as if it were an "object file", or
#     by specifying the -Wl,-dT,linkerscript or -Wl,-T,linkerscript flags.
#   - With elderly GNU ld.bfd versions (<=2.30?), our linker script, which is
#     made up exclusively of SECTION {...} INSERT [AFTER|BEFORE] commands,
#     fails to build when passed with -Wl,-T,linkerscript, but does compile
#     when passed as -Wl,-dT,linkerscript.
#   - With newer GNU ld.bfd, -dT still works.
#   - With LLVM lld/ld.lld, only a subset of GNU ld.bfd's linker script syntax
#     is supported, thankfully including INSERTs.
#   - With LLVM lld/ld.lld, -dT is not supported at all but -T does work for
#     supplying linker scripts.
#   - With LLVM lld/ld.lld, --version-script= does work.
#   - With the PGI 19.4 compiler's linker, this fails because a VERSION clause
#     is always included in the builtin script, and the combination of all
#     linker scripts may only declare at most one such section.
#   - With Google's ld.gold, a full implicit linker script is rejected, whereas
#     the more limited --version-script= works.
#   - With Rui Ueyama's ld.mold, a very limited subset of the linker script
#     language is supported, whereas the more limited --version-script= works.
#
# Since we must absolutely use a linker script, we have elsewhere determined
# whether the default is ld.gold and if so, force the linker to either ld.bfd
# or ld.lld. Next, we use the --version-script= flag unconditionally, and the
# -dT or -T flags depending on whether we are forcing ld.bfd or ld.lld.
#
# It would be easiest if we could pass the version script and linker scripts
# to the linker directly. Unfortunately, Meson does not support auto-formatting
# files() as strings yet, and link_args must be an array of strings.
# So we go about it the hard way - get the current source directory and
# construct the full absolute path to the linker version script, and add that
# manually as an implicit link (though not compile) dependency.
#

if CCLD_FORCE_ID in ['ld.bfd', 'ld.lld']
  libbenzina_lds_name = 'libbenzina.lds'
  libbenzina_ver_name = 'libbenzina.ver'
  libbenzina_lds_path = meson.current_source_dir() / libbenzina_lds_name
  libbenzina_ver_path = meson.current_source_dir() / libbenzina_ver_name
  libbenzina_lds_file = files(libbenzina_lds_name)
  libbenzina_ver_file = files(libbenzina_ver_name)
  libbenzina_lds_flag = CCLD_FORCE_ID == 'ld.bfd' ? '-dT' : '-T'
  libbenzina_link_only += [libbenzina_lds_file, libbenzina_ver_file]
  libbenzina_link_args += [f'-Wl,@libbenzina_lds_flag@,@libbenzina_lds_path@',
                           f'-Wl,--version-script=@libbenzina_ver_path@']
endif


#
# Build the sources of the shared library as static-pic, relink the static
# library as shared, patch it, symlink it, then declare a dependency.
#
# The reasons for this contrived process are manifold:
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
#      We choose to manually patch out that flag using a Python script.
#
# As a result of all this, we manage to declare libbenzina_sha with every type
# (static_library, executable, custom_target, declare_dependency) except the
# one we might truly have intended (shared_library).
#
if IS_HOST_DARWIN
  libbenzina_sha = shared_library(
      libbenzina_name,       libbenzina_srcs,
      include_directories:   libbenzina_incs,
      c_args:                libbenzina_c_args + ['-DBENZINA_IS_BUILDING=1'],
      dependencies:          libbenzina_deps,
      link_args:             libbenzina_link_args,
      link_depends:          libbenzina_link_only,
      gnu_symbol_visibility: 'hidden',
      build_rpath:           librpath,
      install_rpath:         librpath,
      install_dir:           libdir,
      install:               true,
  )
else
  libbenzina_all_deps  =                     declare_dependency(dependencies: libbenzina_deps)
  libbenzina_comp_deps = libbenzina_all_deps.partial_dependency(compile_args: true, includes: true)
  libbenzina_link_deps = libbenzina_all_deps.partial_dependency(link_args:    true, links:    true)
  
  ## * Phase I:   Compile libbenzina.so.X.Y.Z.unpatched.a as PIC, but don't link any objects...
  benzina_sta    = static_library(
      libbenzina_unpatched,  libbenzina_srcs,
      include_directories:   libbenzina_incs,
      c_args:                libbenzina_c_args + ['-DBENZINA_IS_BUILDING=1'],
      dependencies:          libbenzina_comp_deps,
      gnu_symbol_visibility: 'hidden',
      name_prefix:           '',
      pic:                   true,
      prelink:               false,
  )
  
  ## * Phase II:  Link libbenzina.so.X.Y.Z.unpatched as PIE, but don't compile any sources...
  benzina_exe    = executable(
      libbenzina_unpatched,  [],
      objects:               benzina_sta.extract_all_objects(recursive: true),
      link_depends:          libbenzina_link_only,
      link_args:             libbenzina_link_args,
      dependencies:          libbenzina_link_deps,
      build_rpath:           librpath,
      gnu_symbol_visibility: 'hidden',
      name_prefix:           '',
      export_dynamic:        true,
      pie:                   true,
  )
  
  ## * Phase III: Patch libbenzina.so.X.Y.Z.unpatched as libbenzina.so.X.Y.Z, and install
  libbenzina_sha = custom_target(libbenzina_fullname,
      command:      [python, elf_strip_df_1_pie_py, '@INPUT@', '@OUTPUT@'],
      input:        [benzina_exe],
      output:       [libbenzina_fullname],
      depend_files: [elf_py],
      install_mode: 'rwxr-xr-x',
      install_dir:  libdir,
      install:      true,
  )
  
  ## * Phase IV:  Create symlink libbenzina.so.X -> libbenzina.so.X.Y.Z, but do not install
  libbenzina_sha = custom_target(libbenzina_soname,
      command: ['ln', '-sf', '@PLAINNAME@', '@OUTPUT@'],
      input:   libbenzina_sha,
      output:  libbenzina_soname,
      install: false,
  )
  
  ## * Phase V:   Create symlink libbenzina.so -> libbenzina.so.X, but do not install
  libbenzina_sha = custom_target(libbenzina_linkname,
      command: ['ln', '-sf', '@PLAINNAME@', '@OUTPUT@'],
      input:   libbenzina_sha,
      output:  libbenzina_linkname,
      install: false,
  )
  
  ## * Phase VI:  Install symlinks libbenzina.so -> libbenzina.so.X -> libbenzina.so.X.Y.Z
  install_symlink(libbenzina_soname,   install_dir: libdir,  pointing_to: libbenzina_fullname)
  install_symlink(libbenzina_linkname, install_dir: libdir,  pointing_to: libbenzina_soname)
endif

libbenzina = declare_dependency(
    include_directories: benzina_incs,
    compile_args:        ['-DBENZINA_IS_SHARED=1'],
    link_with:           libbenzina_sha,
)
