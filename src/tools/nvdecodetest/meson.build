#
# Executable: benzina-nvdecodetest
#
# An executable that takes a video file and attempts to supply it to NVDECODE.
# Allows one to check whether NVIDIA's decoders will ingest a particular
# bytestream.
#

benzina_nvdecodetestSrcs  = files('main.c')
benzina_nvdecodetestDeps  = [threadsDep, libm, libdl,
                             libavcodec, libavutil,
                             libcuda, libcudart, libnvcuvid,
                             libbenzina]
benzina_nvdecodetest      = executable('benzina-nvdecodetest',
                                       benzina_nvdecodetestSrcs,
                                       dependencies       : benzina_nvdecodetestDeps,
                                       link_with          : [],
                                       build_rpath        : toolrpath,
                                       install_rpath      : toolrpath,
                                       install            : true,
                                       install_dir        : tooldir)

