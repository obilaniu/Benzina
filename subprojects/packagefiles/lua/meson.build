project('Lua', 'c',
    version:         '5.4.3',
    meson_version:   '>=0.58.1',
    license:         'MIT',
    default_options: [
        'warning_level=2',
    ],
)


#
# System check
#
HOST_SYSTEM       = host_machine.system()
IS_HOST_LINUX     = HOST_SYSTEM == 'linux'
IS_HOST_DARWIN    = HOST_SYSTEM == 'darwin'
IS_HOST_SUNOS     = HOST_SYSTEM == 'sunos'
IS_HOST_PLAIN_BSD = HOST_SYSTEM in ['freebsd', 'netbsd', 'openbsd']
IS_HOST_BSD       = HOST_SYSTEM in ['dragonfly'] or HOST_SYSTEM.contains('bsd')
IS_HOST_POSIX     = HOST_SYSTEM not in ['windows', 'emscripten', 'android']


#
# Compiler, Dependencies & Libraries
#
cc          = meson.get_compiler('c')
threads     = dependency('threads')
linenoise   = subproject('linenoise').get_variable('linenoise_dep')
libm        = cc.find_library('m',  required: false)
libdl       = cc.find_library('dl', required: false)


#
# Source Code
#
# Lua's source code must be patched, but Meson does not make it entirely easy
# to do so. Meson does support *overlaying* files atop the source distribution,
# but this requires a copy of the entire file, even for a single-line change.
#
# We contort ourselves into a pretzel here to patch Lua's source code using
# patchfiles.
#
lua_inc = include_directories('.')
patch_script = files('apply-patches.sh')
patch_filenames = files([
    '000-lua-5.4.3-postrel.patch',     # Update from 5.4.3 with postrelease bugfixes
    '001-main-function-rename.patch',  # Allow renaming main()
    '002-luaopen-extra-preload.patch', # Allow preloading of statically-linked libraries.
    '003-searcher-static-C.patch',     # Allow demand-loading of statically-linked libraries.
    '004-searcher-static-Lua.patch',   # Allow demand-loading of statically-linked libraries.
])
core_filenames = [
    'lapi.c',    'lcode.c',
    'lctype.c',  'ldebug.c',
    'ldo.c',     'ldump.c',
    'lfunc.c',   'lgc.c',
    'llex.c',    'lmem.c',
    'lobject.c', 'lopcodes.c',
    'lparser.c', 'lstate.c',
    'lstring.c', 'ltable.c',
    'ltm.c',     'lundump.c',
    'lvm.c',     'lzio.c',
]
aux_filenames = ['lauxlib.c']
lib_filenames = [
   'lbaselib.c', 'lcorolib.c',
   'ldblib.c',   'liolib.c',
   'lmathlib.c', 'loadlib.c',
   'loslib.c',   'lstrlib.c',
   'ltablib.c',  'lutf8lib.c',
   'linit.c',
]
hdr_filenames = [
    'lapi.h',     'lauxlib.h',
    'lcode.h',    'lctype.h',
    'ldebug.h',   'ldo.h',
    'lfunc.h',    'lgc.h',
    'ljumptab.h', 'llex.h',
    'llimits.h',  'lmem.h',
    'lobject.h',  'lopcodes.h',
    'lopnames.h', 'lparser.h',
    'lprefix.h',  'lstate.h',
    'lstring.h',  'ltable.h',
    'ltm.h',      'luaconf.h',
    'lua.h',      'lua.hpp',
    'lualib.h',   'lundump.h',
    'lvm.h',      'lzio.h',
    # Custom:
    'luabenz.h',
]
lua_filenames  = ['lua.c']
luac_filenames = ['luac.c']
inputs         = []
outputs        = []
outvar         = []
foreach v: ['core','aux','lib','hdr','lua','luac']
  foreach f: get_variable(v+'_filenames')
    inputs  += ['src'/f]
    outputs += [f]
    outvar  += [v+'_files']
  endforeach
endforeach
all_patched_files = custom_target(
    'Lua patched files',
    command:      [patch_script, '@CURRENT_SOURCE_DIR@', '@OUTDIR@', '@INPUT@'],
    input:        files(inputs),
    output:       outputs,
    depend_files: patch_filenames,
    install:      false,
)
foreach i: range(inputs.length())
  set_variable(outvar[i], get_variable(outvar[i], []) + [all_patched_files[i]])
endforeach


#
# Compiler and Linker Flags
#
# Four macros of the utmost importance are defined here:
#   - LUA_USER_H: Lua can import a custom header if this is defined.
#         We set it to "luabenz.h".
#   - LUA*_EXTRA: We have patched Lua to optionally allow one to
#         statically link Lua code into the core. We enable these options.
#   - main_lua & main_luac: We rename main() to main_lua() and main_luac() for
#         lua.c and luac.c respectively, for the purpose of compiling both into
#         the same library.
#
c_args  = []
c_args += IS_HOST_POSIX ? ['-DLUA_USE_POSIX']  : []
c_args += IS_HOST_SUNOS ? ['-D_REENTRANT']     : []
c_args += libdl.found() ? ['-DLUA_USE_DLOPEN'] : []
c_args += ['-DLUA_USER_H="luabenz.h"']
custom_lua_sta = static_library(
    'lua',
    core_files+aux_files+lib_files+lua_files+luac_files,
    c_args: c_args + [
        '-DLUATEXT_EXTRA=__lua_text_array',
        '-DLUAOPEN_EXTRA=__lua_open_array',
        '-DLUAOPEN_PRELOAD_EXTRA=__lua_open_array_preload',
        '-Dmain_lua=main_lua',
        '-Dmain_luac=main_luac',
    ],
    pic:          true,
    name_prefix:  '',
    dependencies: [linenoise, threads],
    override_options: [
        'optimization=s',
        'buildtype=minsize',
    ],
)
custom_lua_dep = declare_dependency(
    dependencies:        [linenoise, threads],
    include_directories: lua_inc,
    compile_args:        c_args,
    link_with:           custom_lua_sta,
    link_whole:          [
        static_library('lua-linker', files('linker.noarch.S'), pic: true, name_prefix: ''),
    ],
)
