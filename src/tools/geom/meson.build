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



#
# A few tests using this executable
#
test('perfectfit444', benzina_geom, suite: 'geom', args: ['512:512:yuv444', '512:512:yuv444'])
test('perfectfit420', benzina_geom, suite: 'geom', args: ['512:512:yuv420', '512:512:yuv420'])
