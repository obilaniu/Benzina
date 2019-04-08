#
# Library: libbenzina-geom
#
# Contains geometry problem solvers.
#


### C code compilation
libbenzinageomIncDir = meson.current_source_dir()
libbenzinageomIncs   = include_directories('.')
libbenzinageomSrcs   = [files('geometry.c')]
libbenzinageomDeps   = [libm, libbenzina]
libbenzinageomSta    = static_library('benzina-geom',        libbenzinageomSrcs,
                                      include_directories  : libbenzinageomIncs,
                                      dependencies         : libbenzinageomDeps,
                                      pic                  : true,
                                      install              : false)
libbenzinageom       = declare_dependency(include_directories: libbenzinageomIncs,
                                          link_whole         : libbenzinageomSta,
                                          dependencies       : libbenzina.partial_dependency(includes: true))

