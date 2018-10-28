#
# Executable: benzina_img2vid
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
    '-Xptxas', '--warn-on-double-precision-use,-O3',
    '-I',  libavutil.get_pkgconfig_variable('includedir', default: '/usr/include/ffmpeg'),
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
                           threadsDep, libm, libdl, librt,
                           libcuda, libcudart, libcudadevrt]
benzina_img2vidIncs     = [cudaIncs]
benzina_img2vidCArgs    = []
benzina_img2vid         = executable('benzina_img2vid',   benzina_img2vidSrcs,
                                     include_directories: benzina_img2vidIncs,
                                     dependencies       : benzina_img2vidDeps,
                                     c_args             : benzina_img2vidCArgs,
                                     link_whole         : [benzina_img2vidCudaSta],
                                     install            : true,
                                     install_dir        : tooldir)

