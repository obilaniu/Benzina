#
# Plugin: libbenzina-plugin-nvdecode
#
# A dynamically-loaded plugin implementing within itself the logic required to
# load images and transform them. The true workhorse. Reads:
#
#   - data.bin
#   - data.lengths
#   - data.nvdecode
#   - data.protobuf
#
# Also contains a CUDA kernel. Compiling it is painful in Meson. We must:
#   1) Compile every CUDA file using nvcc -c -Xcompiler -fPIC with a Meson
#      generator(),
#   2) Perform device code linking using nvcc -dlink -Xcompiler -fPIC with a
#      Meson custom_target(),
#   3) Add the CPU object code from 1) and the device code from 2) as
#      "source code" for a Meson shared_library().
#


### Dependency Check
if not nvcc.found()
  subdir_done()
endif


### CUDA code compilation
libbenzina_plugin_nvdecodeCuda_sta = static_library('pre-dlink',
    nvcc_compiler_generator.process(
        files('kernels.cu'),
        extra_args: [
            '--use_fast_math', '-Xcompiler', '-fPIC',
            '-Xptxas', '--warn-on-double-precision-use,-O3',
        ] + benzina_iflags + plugin_cdefs
    )
)
libbenzina_plugin_nvdecodeCuda_dep = declare_dependency(
    link_whole: libbenzina_plugin_nvdecodeCuda_sta,
    sources:    custom_target('libbenzina_plugin_nvdecodeCuda-dlink.o',
        command         : [nvcc, '-shared'] + nvcc_link_args + ['@INPUT@', '-o', '@OUTPUT@'],
        input           : libbenzina_plugin_nvdecodeCuda_sta.extract_all_objects(recursive: true),
        output          : 'libbenzina_plugin_nvdecodeCuda-dlink.o',
        build_by_default: true,
        install         : false,
    )
)



### C code compilation
libbenzina_plugin_nvdecode_srcs = files('main.c')
libbenzina_plugin_nvdecode_deps = [
    threads, dl, libm,
    libcuda, libcudart, libcudadevrt, libnvcuvid,
    libbenzina, plugin,
    libbenzina_plugin_nvdecodeCuda_dep
]
libbenzina_plugin_nvdecode_c_args = cc.get_supported_arguments(['-Wno-unused-variable', '-Wno-sign-compare'])
libbenzina_plugin_nvdecode_sha  = shared_library(
    'benzina-plugin-nvdecode',
    libbenzina_plugin_nvdecode_srcs,
    dependencies         : libbenzina_plugin_nvdecode_deps,
    gnu_symbol_visibility: 'hidden',
    c_args               : libbenzina_plugin_nvdecode_c_args,
    build_rpath          : librpath,
    install_rpath        : librpath,
    install              : true,
    install_dir          : libdir,
)

