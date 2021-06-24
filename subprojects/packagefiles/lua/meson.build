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
subdir('src')


#
# Compiler and Linker Flags
#
# Four macros of the utmost importance are defined here:
#   - LUA_USER_H: Lua can import a custom header if this is defined.
#         We set it to "luabenz.h".
#   - LUA_USE_STATIC_LFS: We have patched Lua to optionally allow one to
#         statically link luafilesystem into the core. We enable this option.
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
        static_library('lua-linker', files('linker.S'), pic: true, name_prefix: ''),
    ],
)
