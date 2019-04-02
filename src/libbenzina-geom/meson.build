#
# Library: libbenzina-geom
#
# Contains geometry problem solvers.
#


### C code compilation
libbenzinageomSrcs  = [files('geometry.c')]
libbenzinageomDeps  = [libm, libbenzina]
libbenzinageomSta   = static_library('benzina-geom',        libbenzinageomSrcs,
                                     dependencies         : libbenzinageomDeps,
                                     pic                  : true,
                                     install              : false)
libbenzinageom      = declare_dependency(include_directories: include_directories('.'),
                                         link_whole         : libbenzinageomSta,
                                         dependencies       : libbenzina.partial_dependency(includes: true))

