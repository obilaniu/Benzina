#
# Executable: benzina_nvdecodetest
#
# An executable that takes a video file and attempts to supply it to NVDECODE.
# Allows one to check whether NVIDIA's decoders will ingest a particular
# bytestream.
#

benzina_nvdecodetestSrcs  = files('main.c')
benzina_nvdecodetestDeps  = [threadsDep, libm, libdl, librt,
                             libcuda, libcudart, libnvcuvid]
benzina_nvdecodetestIncs  = [cudaIncs, benzinaIncs]
benzina_nvdecodetestCArgs = [cudaIncs, benzinaIncs]
benzina_nvdecodetest      = executable('benzina_nvdecodetest',
                                       benzina_nvdecodetestSrcs,
                                       include_directories: benzina_nvdecodetestIncs,
                                       dependencies       : benzina_nvdecodetestDeps,
                                       c_args             : nvcuvidCDefs,
                                       install            : true,
                                       install_dir        : tooldir)




