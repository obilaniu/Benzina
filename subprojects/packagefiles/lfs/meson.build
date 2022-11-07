project('lfs', 'c',
    version:         '1.8.0',
    meson_version:   '>=0.64.0',
    license:         'MIT',
)

custom_lua_dep = subproject('lua').get_variable('custom_lua_dep')
lfs_inc = include_directories('src')
lfs_sta = static_library(
    'lfs',
    files('src/lfs.c'),
    include_directories: lfs_inc,
    dependencies:        custom_lua_dep,
    pic:                 true,
    name_prefix:         '',
    override_options:    ['optimization=s', 'buildtype=minsize'],
)

lfs_dep = declare_dependency(
    dependencies:        custom_lua_dep,
    include_directories: lfs_inc,
    link_with:           lfs_sta,
    link_whole:          [
        static_library('lfs-linker', files('linker.noarch.S'), pic: true, name_prefix: ''),
    ],
)
