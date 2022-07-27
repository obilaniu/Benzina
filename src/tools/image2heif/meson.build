#
# Executable: benzina-image2heif
#
# An executable that takes an image and produces a HEIF equivalent.
#


### License & Dependency Check
if(not ENABLE_GPL          or
   IS_HOST_WINDOWS         or
   not libavcodec.found()  or
   not libavformat.found() or
   not libavfilter.found() or
   not libavutil.found()   or
   not libswscale.found())
  subdir_done()
endif


### C code compilation
benzina_image2heif_srcs = files('main.c')
benzina_image2heif_deps = [libavcodec, libavformat, libavfilter, libavutil, libswscale,
                           threads, dl, libm, libbenzina]
benzina_image2heif      = executable('benzina-image2heif', benzina_image2heif_srcs,
                                     dependencies        : benzina_image2heif_deps,
                                     build_rpath         : toolrpath,
                                     install_rpath       : toolrpath,
                                     install             : true,
                                     install_dir         : tooldir)

