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
libbenzina_plugin_nvdecodeCudaSrcs = files('kernels.cu')
libbenzina_plugin_nvdecodeCudaObjs = nvccCompGen.process(libbenzina_plugin_nvdecodeCudaSrcs, extra_args: [
    '-Xcompiler', '-fPIC',
    '-Xptxas', '--warn-on-double-precision-use,-O3',
    '--use_fast_math',
] + libbenzinacudaCDefs + libbenzinaCDefs + benzinaIFlags + pluginCDefs)
libbenzina_plugin_nvdecodeCudaSta  = static_library('pre-dlink', libbenzina_plugin_nvdecodeCudaObjs)
libbenzina_plugin_nvdecodeCudaDep  = declare_dependency(link_whole: libbenzina_plugin_nvdecodeCudaSta)
libbenzina_plugin_nvdecodeCudaObjs = custom_target ('libbenzina_plugin_nvdecodeCuda-dlink',
    command         : [nvcc, '-shared'] + nvccLinkArgs + ['@INPUT@', '-o', '@OUTPUT@'],
    input           : libbenzina_plugin_nvdecodeCudaSta,
    output          : ['@BASENAME@-dlink.o'],
    build_by_default: true,
    install         : false
)
libbenzina_plugin_nvdecodeCudaSta  = static_library('dlink', libbenzina_plugin_nvdecodeCudaObjs)
libbenzina_plugin_nvdecodeCudaDep  = declare_dependency(dependencies: libbenzina_plugin_nvdecodeCudaDep,
                                                        link_whole:   libbenzina_plugin_nvdecodeCudaSta)


### C code compilation
libbenzina_plugin_nvdecodeSrcs = files('main.c')
libbenzina_plugin_nvdecodeDeps = [threadsDep, libm, libdl,
                                  libcuda, libcudart, libcudadevrt, libnvcuvid,
                                  libbenzina, libbenzinacuda, pluginDep,
                                  libbenzina_plugin_nvdecodeCudaDep]
libbenzina_plugin_nvdecodeSha  = shared_library('benzina-plugin-nvdecode',
                                                libbenzina_plugin_nvdecodeSrcs,
                                                dependencies         : libbenzina_plugin_nvdecodeDeps,
                                                gnu_symbol_visibility: 'hidden',
                                                build_rpath          : librpath,
                                                install_rpath        : librpath,
                                                install              : true,
                                                install_dir          : libdir)

