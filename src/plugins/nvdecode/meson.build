#
# Plugin: libbenzina_plugin_nvdecode
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

### CUDA code compilation
libbenzina_plugin_nvdecodeCudaSrcs = files('kernels.cpp')
libbenzina_plugin_nvdecodeCudaObjs = nvccCompGen.process(libbenzina_plugin_nvdecodeCudaSrcs, extra_args: [
    '-Xcompiler', '-fPIC',
    '-Xptxas', '--warn-on-double-precision-use,-O3',
     '--use_fast_math', '-O3',
    '-I'+benzinaIncsDir,
])
libbenzina_plugin_nvdecodeCudaSta  = static_library('pre-dlink', libbenzina_plugin_nvdecodeCudaObjs)
libbenzina_plugin_nvdecodeCudaObjs = custom_target ('libbenzina_plugin_nvdecodeCuda-dlink',
    command         : [nvcc, '@INPUT@', '-dlink', '-o', '@OUTPUT@'],
    input           : libbenzina_plugin_nvdecodeCudaSta,
    output          : ['@BASENAME@-dlink.o'],
    build_by_default: true,
    install         : false
)
libbenzina_plugin_nvdecodeCudaSta  = static_library('post-dlink', libbenzina_plugin_nvdecodeCudaObjs,
                                                    link_whole:   libbenzina_plugin_nvdecodeCudaSta)


### C code compilation
libbenzina_plugin_nvdecodeSrcs     = files('main.c')
libbenzina_plugin_nvdecodeDeps     = [threadsDep, libm, libdl, librt, libcuda, libcudart, libcudadevrt, libnvcuvid]
libbenzina_plugin_nvdecodeIncs     = [cudaIncs, benzinaIncs]
libbenzina_plugin_nvdecodeCArgs    = pluginCArgs+nvcuvidCDefs
libbenzina_plugin_nvdecode         = shared_library('benzina_plugin_nvdecode',
                                                    libbenzina_plugin_nvdecodeSrcs,
                                                    include_directories  : libbenzina_plugin_nvdecodeIncs,
                                                    dependencies         : libbenzina_plugin_nvdecodeDeps,
                                                    c_args               : libbenzina_plugin_nvdecodeCArgs,
                                                    gnu_symbol_visibility: 'hidden',
                                                    link_with            : [libbenzinaSha],
                                                    link_whole           : [libbenzina_plugin_nvdecodeCudaSta],
                                                    build_rpath          : '$ORIGIN',
                                                    install_rpath        : '$ORIGIN',
                                                    install              : true,
                                                    install_dir          : libdir)



#
# Executable: benzina_nvdecode_gen
#
# An executable that takes data.bin and data.lengths files and generates the
# corresponding data.nvdecode file.
#

benzina_nvdecode_genSrcs  = files('nvdecodegen.c')
benzina_nvdecode_genDeps  = [threadsDep, libm, libdl, libcuda, libnvcuvid]
benzina_nvdecode_genIncs  = [cudaIncs, benzinaIncs]
benzina_nvdecode_genCArgs = [cudaIncs, benzinaIncs]
benzina_nvdecode_gen      = executable('benzina_nvdecode_gen', benzina_nvdecode_genSrcs,
                                       include_directories: benzina_nvdecode_genIncs,
                                       dependencies       : benzina_nvdecode_genDeps,
                                       c_args             : nvcuvidCDefs,
                                       install            : true,
                                       install_dir        : bindir)




