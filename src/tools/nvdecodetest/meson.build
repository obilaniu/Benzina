#
# Executable: benzina-nvdecodetest
#
# An executable that takes a video file and attempts to supply it to NVDECODE.
# Allows one to check whether NVIDIA's decoders will ingest a particular
# bytestream.
#

benzina_nvdecodetestSrcs  = files('main.c')
benzina_nvdecodetestDeps  = [threadsDep, libm, libdl, librt,
                             libcuda, libcudart, libnvcuvid]
benzina_nvdecodetestCDefs = nvcuvidCDefs
benzina_nvdecodetestIncs  = [cudaIncs, benzinaIncs]
benzina_nvdecodetestCArgs = [cudaIncs, benzinaIncs]
benzina_nvdecodetest      = executable('benzina-nvdecodetest',
                                       benzina_nvdecodetestSrcs,
                                       include_directories: benzina_nvdecodetestIncs,
                                       dependencies       : benzina_nvdecodetestDeps,
                                       c_args             : benzina_nvdecodetestCDefs,
                                       link_with          : [libbenzinaSha],
                                       build_rpath        : toolrpath,
                                       install_rpath      : toolrpath,
                                       install            : true,
                                       install_dir        : tooldir)

