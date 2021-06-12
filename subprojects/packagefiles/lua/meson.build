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
# Compiler and Linker Flags, Part I
#
c_args  = []
c_args += IS_HOST_POSIX ? ['-DLUA_USE_POSIX']  : []
c_args += IS_HOST_SUNOS ? ['-D_REENTRANT']     : []


#
# Lua Compiler
#
luac_c_args = c_args
if cc.get_argument_syntax() == 'gcc'
  luac_c_args += cc.get_supported_arguments(['-ffunction-sections', '-fno-PIC'])
endif
luac = executable(
    'luac',
    core_files+aux_files+luac_files,
    c_args:           luac_c_args,
    pie:              false,
    dependencies:     [libm],
    override_options: [
        'optimization=s',
        'debug=false',
        'buildtype=minsize',
        'b_ndebug=true',
        'b_asneeded=true',
        'b_pie=false',
        'strip=true',
    ],
)
luac_gen = generator(luac, arguments: ['@INPUT@', '@EXTRA_ARGS@', '-o', '@OUTPUT@'],
                           output:    '@BASENAME@')


#
# Compiler and Linker Flags, Part II
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
c_args += libdl.found() ? ['-DLUA_USE_DLOPEN'] : []
c_args += ['-DLUA_USER_H="luabenz.h"']
custom_lua_sta = static_library(
    'lua',
    core_files+aux_files+lib_files+lua_files+luac_files+files('LICENSE.S'),
    c_args: c_args + [
        '-DLUA_USE_STATIC_LFS',
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
)
