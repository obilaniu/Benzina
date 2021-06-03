project('Lua', 'c',
    version:         '5.4.3',
    meson_version:   '>=0.58.0',
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
assert(get_option('c_std') not in ['c89', 'gnu89'], 'C89 mode not supported in this build!')


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
core_files = files(
    'src/lapi.c',
    'src/lcode.c',
    'src/lctype.c',
    'src/ldebug.c',
    'src/ldo.c',
    'src/ldump.c',
    'src/lfunc.c',
    'src/lgc.c',
    'src/llex.c',
    'src/lmem.c',
    'src/lobject.c',
    'src/lopcodes.c',
    'src/lparser.c',
    'src/lstate.c',
    'src/lstring.c',
    'src/ltable.c',
    'src/ltm.c',
    'src/lundump.c',
    'src/lvm.c',
    'src/lzio.c',
)
aux_files  = files('src/lauxlib.c')
ini_files  = files('src/linit.c')
lib_files  = files(
    'src/lbaselib.c',
    'src/lcorolib.c',
    'src/ldblib.c',
    'src/liolib.c',
    'src/lmathlib.c',
    'src/loadlib.c',
    'src/loslib.c',
    'src/lstrlib.c',
    'src/ltablib.c',
    'src/lutf8lib.c',
)
hdr_files  = files(
    'src/lua.h',
    'src/luaconf.h',
    'src/lualib.h',
    'src/lauxlib.h',
    'src/lua.hpp',
)
lua_files  = files('src/lua.c')
luac_files = files('src/luac.c')
lua_inc    = include_directories('src')


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
luac = executable('luac',
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
# Two macros of the utmost importance are defined here:
#   - LUA_USER_H: Lua can import a custom header if this is defined. We set it to "luabenz.h".
#   - main: We #define main to lua_main and luac_main for lua.c and luac.c respectively, for
#           the purpose of compiling both into the same library.
#
c_args += libdl.found() ? ['-DLUA_USE_DLOPEN'] : []
c_args += ['-DLUA_USER_H="luabenz.h"']
options = ['optimization=s', 'buildtype=minsize']
custom_lua_sta      = static_library('lua',
    core_files+aux_files+ini_files+lib_files+files('LICENSE.S'),
    name_prefix:  '',
    c_args:       c_args,
    pic:          true,
    dependencies: [linenoise, threads],
    override_options: options,
)
custom_lua_main_sta = static_library('lua_main',
    lua_files,
    name_prefix:  '',
    c_args:       c_args + ['-Dmain=lua_main'],
    pic:          true,
    dependencies: [linenoise, threads],
    override_options: options,
)
custom_luac_main_sta = static_library('luac_main',
    luac_files,
    name_prefix:  '',
    c_args:       c_args + ['-Dmain=luac_main'],
    pic:          true,
    dependencies: [linenoise, threads],
    override_options: options,
)
custom_linit_sta = static_library('linit',
    ini_files,
    name_prefix:  '',
    c_args:       c_args + [
        #
        # What follows is the ugliest compiler command-line -D function-macro
        # definition in the history of all mankind. It is a tortured way to
        # inject extra code inside luaL_openlibs() to open more libraries
        # without copying linit.c wholesale.
        #
        '-DluaL_openlibs(x)=luaL_openlibs(x){'+
        '    void luaL_openlibs2(x);'+
        '    luaL_openlibs2(L);'+
        '    int luaopen_lfs(x);'+
        '    luaL_requiref(L, "lfs", luaopen_lfs, 1);'+
        '    lua_pop(L, 1);'+
        '}'+
        'LUALIB_API void luaL_openlibs2(x)'
    ],
    pic:          true,
    dependencies: [linenoise, threads],
    override_options: options,
)


#
# Custom build of the Lua core, Lua interpreter and Lua compiler all-in-one.
#
custom_lua_dep = declare_dependency(
    dependencies:        [linenoise, threads],
    include_directories: lua_inc,
    compile_args:        c_args,
    link_with:           [
        custom_lua_sta,
        custom_lua_main_sta,
        custom_luac_main_sta,
        custom_linit_sta,
    ],
)
