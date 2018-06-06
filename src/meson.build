#
# Library: libbenzina
#
# Contains most of the business logic for data-loading.
#

libbenzinaSrcs     = files('benzina.c', 'benzinafork.c', 'benzinaproto.c')
libbenzinaDeps     = [threadsDep, libm, libdl, libcuda]
libbenzinaIncs     = [cudaIncs, benzinaIncs]
libbenzinaCArgs    = []
libbenzinaShaCArgs = libbenzinaCArgs+['-DBENZINA_SHAREDOBJECT']
if build_machine.system() == 'windows'
  libbenzinaShaCArgs += ['-DBENZINA_BUILDING_DLL']
else
  libbenzinaShaCArgs += ['-fvisibility=hidden']
endif

libbenzinaSha   = shared_library('benzina',           libbenzinaSrcs,
                                 include_directories: libbenzinaIncs,
                                 dependencies       : libbenzinaDeps,
                                 c_args             : libbenzinaShaCArgs,
                                 install            : true,
                                 install_dir        : libdir)

#
# Executable: benzina_fork_server
# Executable: benzina_fork_server_test
#
# A small intermediary process that services fork()+exec() requests, and a
# corresponding test executable for it.
#

benzina_fork_server      = executable('benzina_fork_server',
                                      files('forkserver.c'),
                                      include_directories: [benzinaIncs],
                                      link_with          : [libbenzinaSha],
                                      dependencies       : [libdl],
                                      install            : true,
                                      install_dir        : bindir)
benzina_fork_server_test = executable('benzina_fork_server_test',
                                      files('forkservertest.c'),
                                      include_directories: [benzinaIncs],
                                      link_with          : [libbenzinaSha],
                                      dependencies       : [libdl],
                                      install            : false,
                                      install_dir        : bindir)


#
# Executable: benzina_nvdec_gen
#
# An executable that takes data.bin and data.lengths files and generates the
# corresponding data.nvdecode file.
#

benzina_nvdec_genSrcs = files('nvdecgen.c')
benzina_nvdec_genDeps = [threadsDep, libm, libcuda, libdl]
benzina_nvdec_genIncs = [cudaIncs, benzinaIncs]
benzina_nvdec_gen     = executable('benzina_nvdec_gen', benzina_nvdec_genSrcs,
                                   include_directories: benzina_nvdec_genIncs,
                                   dependencies       : benzina_nvdec_genDeps,
                                   install            : true,
                                   install_dir        : bindir)


#
# Executable: benzina_nvdec_load
#
# The true image-loader workhorse. Reads:
#   - data.bin
#   - data.lengths
#   - data.nvdecode
#   - data.protobuf
# And communicates over shared memory the data with the requesting process.
#

benzina_nvdec_loadSrcs = files('nvdecload.cu')
benzina_nvdec_loadDeps = [threadsDep, libm, libcuda, libdl]
benzina_nvdec_loadIncs = [cudaIncs, benzinaIncs]
benzina_nvdec_loadCmd  = [nvcc, '@INPUT@', '-I'+benzinaIncsDir, libbenzinaSha.full_path(),
                          '-Xlinker', '-rpath=$ORIGIN', '-o', '@OUTPUT@']
benzina_nvdec_load     = custom_target('benzina_ncdec_load',
                                       command:          benzina_nvdec_loadCmd,
                                       depends:          [libbenzinaSha],
                                       input:            benzina_nvdec_loadSrcs,
                                       output:           ['benzina_nvdec_load'],
                                       build_by_default: true,
                                       install:          true,
                                       install_dir:      bindir)


