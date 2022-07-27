#
# Executable: benzina-nvdecodetest
#
# An executable that takes a video file and attempts to supply it to NVDECODE.
# Allows one to check whether NVIDIA's decoders will ingest a particular
# bytestream.
#

benzina_nvdecodetest_srcs = files('main.c')
benzina_nvdecodetest_deps = [threads, dl, libm, libavcodec, libavutil,
                             libcuda, libcudart, libnvcuvid, libbenzina]
benzina_nvdecodetest      = executable('benzina-nvdecodetest',
                                       benzina_nvdecodetest_srcs,
                                       dependencies       : benzina_nvdecodetest_deps,
                                       link_with          : [],
                                       build_rpath        : toolrpath,
                                       install_rpath      : toolrpath,
                                       install            : true,
                                       install_dir        : tooldir)

