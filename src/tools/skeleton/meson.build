#
# Executable: benzina-skeleton
#
# A skeleton for a Benzina tool. Meant to be copy-pasted and used as a template.
#


### Dependency Check
if not true
  subdir_done()
endif


### C code compilation
benzina_skeletonSrcs  = files('main.c')
benzina_skeletonDeps  = [libavcodec, libavformat, libavfilter, libavutil, libswscale,
                         threadsDep, libm, libdl,
                         libcuda, libcudart, libnvcuvid,
                         libbenzina, libbenzinacuda]
benzina_skeleton      = executable('benzina-skeleton',
                                   benzina_skeletonSrcs,
                                   include_directories: [],
                                   dependencies       : benzina_skeletonDeps,
                                   c_args             : [],
                                   build_rpath        : toolrpath,
                                   install_rpath      : toolrpath,
                                   install            : true,
                                   install_dir        : tooldir)

