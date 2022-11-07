project('lpeg', 'c',
    version:         '1.0.2',
    meson_version:   '>=0.64.0',
    license:         'MIT',
    default_options: [
        'warning_level=1',
    ],
)

custom_lua_dep = subproject('lua').get_variable('custom_lua_dep')
lpeg_inc = include_directories('.')
lpeg_sta = static_library(
    'lpeg',
    files('lpcap.c', 'lpcode.c', 'lpprint.c', 'lptree.c', 'lpvm.c'),
    include_directories: lpeg_inc,
    dependencies:        custom_lua_dep,
    pic:                 true,
    name_prefix:         '',
    override_options:    ['optimization=s', 'buildtype=minsize'],
)
lpeg_lua = {
    're': files('re.lua')
}

lpeg_dep = declare_dependency(
    dependencies:        custom_lua_dep,
    include_directories: lpeg_inc,
    link_with:           lpeg_sta,
    link_whole:          [
        static_library('lpeg-linker', files('linker.noarch.S'), pic: true, name_prefix: ''),
    ],
)
