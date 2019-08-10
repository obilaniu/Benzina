#
# Library: libbenzina-geom
#
# Contains geometry problem solvers.
#


### C code compilation
libbenzinageom_incdir = meson.current_source_dir()
libbenzinageom_incs   = include_directories('.')
libbenzinageom_srcs   = [files('geometry.c')]
libbenzinageom_deps   = [libm, libbenzina]
libbenzinageom_sta    = static_library('benzina-geom',        libbenzinageom_srcs,
                                       include_directories  : libbenzinageom_incs,
                                       dependencies         : libbenzinageom_deps,
                                       pic                  : true,
                                       install              : false)
libbenzinageom        = declare_dependency(include_directories: libbenzinageom_incs,
                                           link_whole         : libbenzinageom_sta,
                                           dependencies       : libbenzina.partial_dependency(includes: true))

