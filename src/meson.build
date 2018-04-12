#
# Executables
#

benzinasrvSrcs = files('benzinasrv.c')
benzinasrvDeps = [threadsDep, libm, libzmq, libcuda, libdl]
benzinasrvIncs = [cudaIncs, benzinaIncs]
benzinasrv     = executable('benzinasrv',        benzinasrvSrcs,
                            include_directories: benzinasrvIncs,
                            dependencies       : benzinasrvDeps)



benzinanvdecgenSrcs = files('benzinanvdecgen.c')
benzinanvdecgenDeps = [threadsDep, libm, libcuda, libdl]
benzinanvdecgenIncs = [cudaIncs, benzinaIncs]
benzinanvdecgen     = executable('benzinanvdecgen',   benzinanvdecgenSrcs,
                                 include_directories: benzinanvdecgenIncs,
                                 dependencies       : benzinanvdecgenDeps)

