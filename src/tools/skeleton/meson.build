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
benzina_skeleton_srcs = files('main.c')
benzina_skeleton_deps = [libavcodec, libavformat, libavfilter, libavutil, libswscale,
                         threads, dl, libm, libcuda, libcudart, libnvcuvid,
                         libbenzina]
benzina_skeleton      = executable('benzina-skeleton',
                                   benzina_skeleton_srcs,
                                   include_directories: [],
                                   dependencies       : benzina_skeleton_deps,
                                   c_args             : [],
                                   build_rpath        : toolrpath,
                                   install_rpath      : toolrpath,
                                   install            : true,
                                   install_dir        : tooldir)

