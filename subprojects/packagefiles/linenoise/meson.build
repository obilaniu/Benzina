project('linenoise', 'c',
    version:         '1.0',
    meson_version:   '>=0.64.0',
    license:         'MIT',
    default_options: [
        'warning_level=0',
    ],
)

linenoise_inc = include_directories('.')
linenoise_sta = static_library(
    'linenoise',
    files('linenoise.c', 'encodings/utf8.c'),
    include_directories: linenoise_inc,
    pic:                 true,
    name_prefix:         '',
    override_options:    ['optimization=s', 'buildtype=minsize'],
)

linenoise_dep = declare_dependency(
    include_directories: linenoise_inc,
    link_with:           linenoise_sta,
    link_whole:          [
        static_library('linenoise-linker', files('linker.noarch.S'), pic: true, name_prefix: ''),
    ],
)
