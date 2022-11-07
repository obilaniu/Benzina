project('Lua', 'c',
    version:         '5.4.4',
    meson_version:   '>=0.64.0',
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
cc        = meson.get_compiler('c')
threads   = dependency('threads')
dl        = dependency('dl', required: false)
linenoise = subproject('linenoise').get_variable('linenoise_dep')
libm      = cc.find_library('m', required: false)


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
lua_filenames  = ['lua.c']
luac_filenames = ['luac.c']


#
# Compiler and Linker Flags
#
# A few macros of the utmost importance are defined here:
#   - LUA*_EXTRA: We have patched Lua to optionally allow one to
#         statically link Lua code into the core. We enable these options.
#   - main_lua & main_luac: We rename main() to main_lua() and main_luac() for
#         lua.c and luac.c respectively, for the purpose of compiling both into
#         the same library.
#
c_args  = []
c_args += IS_HOST_POSIX ? ['-DLUA_USE_POSIX']  : []
c_args += IS_HOST_SUNOS ? ['-D_REENTRANT']     : []
c_args += dl.found()    ? ['-DLUA_USE_DLOPEN'] : []

c_private_args  = [
    '-Dmain_lua=main_lua',
    '-Dmain_luac=main_luac',
    '-DLUATEXT_EXTRA=__lua_text_array',
    '-DLUAOPEN_EXTRA=__lua_open_array',
    '-DLUA_USE_THREADING',
    '-DLUA_USE_LINENOISE',
    '-DLUA_USE_LINENOISE_UTF8',
    '-DLUA_USE_LINENOISE_HISTCONTROL',
    '-ULUA_USE_READLINE',
]
c_private_args += IS_HOST_LINUX ? ['-D_DEFAULT_SOURCE'] : []

custom_lua_sta  = static_library('lua',
    files(core_filenames+aux_filenames+lib_filenames+lua_filenames+luac_filenames),
    dependencies:        [linenoise, threads, dl],
    c_args:              c_args+c_private_args,
    pic:                 true,
    name_prefix:         '',
    override_options:    [
        'optimization=s',
        'buildtype=minsize',
    ],
)
custom_lua_dep  = declare_dependency(
    dependencies:        [linenoise, threads, dl],
    include_directories: lua_inc,
    compile_args:        c_args,
    link_with:           custom_lua_sta,
    link_whole:          [
        static_library('lua-linker', files('linker.noarch.S'), pic: true, name_prefix: ''),
    ],
)
