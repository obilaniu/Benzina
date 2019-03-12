#
# Executable: benzina-geom
#
# An executable that takes a canvas specification and an image specification,
# and prints out the solution to the geometry problem for that pair.
#


### C code compilation
benzina_geomSrcs     = files('main.c', 'geometry.c')
benzina_geomDeps     = [libm, libdl, libbenzina]
benzina_geom         = executable('benzina-geom', benzina_geomSrcs,
                                  dependencies  : benzina_geomDeps,
                                  build_rpath   : toolrpath,
                                  install_rpath : toolrpath,
                                  install       : false,
                                  install_dir   : tooldir)

