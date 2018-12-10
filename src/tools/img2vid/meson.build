#
# Executable: benzina-img2vid
#
# An executable that takes an image and produces a 1-frame H.264 video clip in
# Datum format.
#


### License & Dependency Check
if not ENABLE_GPL or not HAVE_FFMPEG
  subdir_done()
endif


### CUDA code compilation
benzina_img2vidCudaSrcs = files('kernels.cpp')
benzina_img2vidCudaObjs = nvccCompGen.process(benzina_img2vidCudaSrcs, extra_args: [
    '-Xptxas', '--warn-on-double-precision-use,-O3', '-I', benzinaIncDir, '-I', ffmpegIncDir,
])
benzina_img2vidCudaSta  = static_library('pre-dlink', benzina_img2vidCudaObjs)
benzina_img2vidCudaObjs = custom_target ('benzina_img2vidCuda-dlink',
    command         : [nvcc, '@INPUT@', '-dlink', '-o', '@OUTPUT@'],
    input           : benzina_img2vidCudaSta,
    output          : ['@BASENAME@-dlink.o'],
    build_by_default: true,
    install         : false
)
benzina_img2vidCudaSta  = static_library('post-dlink', benzina_img2vidCudaObjs,
                                         link_whole:   benzina_img2vidCudaSta)


### C code compilation
benzina_img2vidSrcs     = files('main.c')
benzina_img2vidDeps     = [libavcodec, libavformat, libavfilter, libavutil, libswscale,
                           threadsDep, libm, libdl,
                           libcuda, libcudart, libcudadevrt,
                           libbenzina, libbenzinacuda]
benzina_img2vid         = executable('benzina-img2vid',   benzina_img2vidSrcs,
                                     dependencies       : benzina_img2vidDeps,
                                     link_whole         : [benzina_img2vidCudaSta],
                                     build_rpath        : toolrpath,
                                     install_rpath      : toolrpath,
                                     install            : true,
                                     install_dir        : tooldir)

