project('lz4', 'c',
    version:         '1.9.4',
    meson_version:   '>=0.64.0',
    license:         'BSD',
    default_options: [
        'warning_level=2',
    ],
)


cc = meson.get_compiler('c')
size_opt_args = cc.get_supported_arguments(['-ffunction-sections'])


lz4_lib_files  = files(
    'lib/lz4.c',
    'lib/lz4hc.c',
    'lib/lz4frame.c',
    'lib/lz4file.c',
    'lib/xxhash.c',
)
lz4_prog_files = files(
    'programs/bench.c',
    'programs/datagen.c',
    'programs/lz4cli.c',
    'programs/lz4io.c',
)
lz4_inc        = include_directories('lib')
lz4_c_args     = [
    '-DXXH_NAMESPACE=LZ4_',   # Avoid xxhash symbol conflicts with an LZ4-specific prefix.
    '-DLZ4LIB_VISIBILITY=',   # Disable visibility declarations.
    '-DLZ4FLIB_VISIBILITY=',  # Disable visibility declarations.
]

lz4_exe_native = executable(
    'lz4',
    lz4_lib_files+lz4_prog_files,
    include_directories:   lz4_inc,
    c_args:                lz4_c_args,
    native:                true,
    override_options:      ['optimization=s', 'buildtype=minsize'],
)
lz4_sta = static_library(
    'lz4',
    lz4_lib_files,
    include_directories:   lz4_inc,
    c_args:                lz4_c_args+size_opt_args,
    gnu_symbol_visibility: 'hidden',
    native:                false,
    pic:                   true,
    override_options:      ['optimization=s', 'buildtype=minsize'],
)

lz4_dep = declare_dependency(
    include_directories:   lz4_inc,
    link_with:             lz4_sta,
    link_whole:            [
        static_library('lz4-linker', files('linker.noarch.S'), pic: true, name_prefix: ''),
    ],
)
