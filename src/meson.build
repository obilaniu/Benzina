#
# Library: libbenzina
#
# Contains most of the business logic for data-loading.
#

libbenzinaSrcs  = files('benzina.c', 'benzinaproto.c')
libbenzinaDeps  = [threadsDep, libm, libdl, librt]
libbenzinaIncs  = [cudaIncs, benzinaIncs]
libbenzinaCArgs = ['-DBENZINA_IS_BUILDING=1',
                   '-DBENZINA_IS_SHARED=1',
                   '-DBENZINA_PLUGIN_IS_SHARED=1']
if build_machine.system() != 'windows'
  libbenzinaCArgs += ['-fvisibility=hidden']
endif
libbenzinaSha   = shared_library('benzina',           libbenzinaSrcs,
                                 include_directories: libbenzinaIncs,
                                 dependencies       : libbenzinaDeps,
                                 c_args             : libbenzinaCArgs,
                                 build_rpath        : '$ORIGIN',
                                 install_rpath      : '$ORIGIN',
                                 install            : true,
                                 install_dir        : libdir)



#
# PLUGINS.
#
# So far, there is just one plugin.
#

pluginCArgs = ['-DBENZINA_PLUGIN_IS_BUILDING=1',
               '-DBENZINA_PLUGIN_IS_SHARED=1',
               '-DBENZINA_IS_SHARED=1']
if build_machine.system() != 'windows'
  pluginCArgs += ['-fvisibility=hidden']
endif

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

libbenzina_plugin_nvdecodeCudaSrcs = files('plugin_nvdecode_kernel.cu')
nvcc_compile                       = generator     (nvcc,
                                                    arguments       : ['@INPUT@', '-I'+benzinaIncsDir, '-Xcompiler', '-fPIC', '-c', '-o', '@OUTPUT@'],
                                                    output          : ['@BASENAME@.o'])
libbenzina_plugin_nvdecodeCudaObjs = nvcc_compile.process(libbenzina_plugin_nvdecodeCudaSrcs)
libbenzina_plugin_nvdecodeCudaCmd  = [nvcc, '-Xcompiler', '-fPIC', '@INPUT@', '-dlink', '-o', '@OUTPUT@']
libbenzina_plugin_nvdecodeCudaDObj = custom_target ('libbenzina_plugin_nvdecode_cuda',
                                                    command         : libbenzina_plugin_nvdecodeCudaCmd,
                                                    input           : libbenzina_plugin_nvdecodeCudaObjs,
                                                    output          : ['libbenzina_plugin_nvdecode_cuda-dlink.o'],
                                                    build_by_default: true,
                                                    install         : false)

libbenzina_plugin_nvdecodeSrcs     = [files('plugin_nvdecode.c'), libbenzina_plugin_nvdecodeCudaObjs, libbenzina_plugin_nvdecodeCudaDObj]
libbenzina_plugin_nvdecodeDeps     = [threadsDep, libm, libdl, librt, libcuda, libcudart, libcudadevrt]
libbenzina_plugin_nvdecodeIncs     = [cudaIncs, benzinaIncs]
libbenzina_plugin_nvdecodeCArgs    = pluginCArgs
libbenzina_plugin_nvdecode         = shared_library('benzina_plugin_nvdecode',
                                                    libbenzina_plugin_nvdecodeSrcs,
                                                    include_directories: libbenzina_plugin_nvdecodeIncs,
                                                    dependencies       : libbenzina_plugin_nvdecodeDeps,
                                                    c_args             : libbenzina_plugin_nvdecodeCArgs,
                                                    link_with          : [libbenzinaSha],
                                                    build_rpath        : '$ORIGIN',
                                                    install_rpath      : '$ORIGIN',
                                                    install            : true,
                                                    install_dir        : libdir)



#
# Executable: benzina_nvdecode_gen
#
# An executable that takes data.bin and data.lengths files and generates the
# corresponding data.nvdecode file.
#

benzina_nvdecode_genSrcs = files('nvdecodegen.c')
benzina_nvdecode_genDeps = [threadsDep, libm, libcuda, libdl]
benzina_nvdecode_genIncs = [cudaIncs, benzinaIncs]
benzina_nvdecode_gen     = executable('benzina_nvdecode_gen', benzina_nvdecode_genSrcs,
                                      include_directories: benzina_nvdecode_genIncs,
                                      dependencies       : benzina_nvdecode_genDeps,
                                      install            : true,
                                      install_dir        : bindir)




