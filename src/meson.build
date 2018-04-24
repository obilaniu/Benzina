#
# Libraries
#

libbenzinaSrcs  = files('benzina.c')
libbenzinaDeps  = [threadsDep, libm, libcuda, libdl]
libbenzinaIncs  = [cudaIncs, benzinaIncs]
libbenzinaCArgs = ['-DBENZINA_SHAREDOBJECT']
if build_machine.system() == 'windows'
  libbenzinaCArgs += ['-DBENZINA_BUILDING_DLL']
else
  libbenzinaCArgs += ['-fvisibility=hidden']
endif

libbenzina      = shared_library('benzina',           libbenzinaSrcs,
                                 include_directories: libbenzinaIncs,
                                 dependencies       : libbenzinaDeps,
                                 c_args             : libbenzinaCArgs,
                                 install            : true,
                                 install_dir        : join_paths('benzina', 'libs'),)

#
# Executables
#

benzinanvdecgenSrcs = files('benzinanvdecgen.c')
benzinanvdecgenDeps = [threadsDep, libm, libcuda, libdl]
benzinanvdecgenIncs = [cudaIncs, benzinaIncs]
benzinanvdecgen     = executable('benzinanvdecgen',   benzinanvdecgenSrcs,
                                 include_directories: benzinanvdecgenIncs,
                                 dependencies       : benzinanvdecgenDeps)

