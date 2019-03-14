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
test('chromafmt 4:0:0',         benzina_geom, suite: 'geom', args: ['512:512:yuv400',      '512:512:yuv400'])
test('chromafmt 4:2:0',         benzina_geom, suite: 'geom', args: ['512:512:yuv420',      '512:512:yuv420'])
test('chromafmt 4:2:0 super',   benzina_geom, suite: 'geom', args: ['512:512:yuv420super', '256:256:yuv420'])
test('chromafmt 4:4:4',         benzina_geom, suite: 'geom', args: ['512:512:yuv444',      '512:512:yuv444'])

test('chromaloc topleft',       benzina_geom, suite: 'geom', args: ['512:512', '512:512:yuv420:topleft'])
test('chromaloc top',           benzina_geom, suite: 'geom', args: ['512:512', '512:512:yuv420:top'])
test('chromaloc left',          benzina_geom, suite: 'geom', args: ['512:512', '512:512:yuv420:left'])
test('chromaloc center',        benzina_geom, suite: 'geom', args: ['512:512', '512:512:yuv420:center'])
test('chromaloc bottomleft',    benzina_geom, suite: 'geom', args: ['512:512', '512:512:yuv420:bottomleft'])
test('chromaloc bottom',        benzina_geom, suite: 'geom', args: ['512:512', '512:512:yuv420:bottom'])
test('chromaloc bottomright',   benzina_geom, suite: 'geom', args: ['512:512', '512:512:yuv420:bottomright'])

test('whole fit untrans',       benzina_geom, suite: 'geom', args: ['1920:1080', '1280:720'])
test('whole fit trans',         benzina_geom, suite: 'geom', args: ['1920:1080', '720:1280'])

test('transposed chroma hflip', benzina_geom, suite: 'geom', args: ['1920:1080', '720:1280:yuv420:bottomleft'])

test('xar untrans',             benzina_geom, suite: 'geom', args: ['1920:1080', '1920:2'])
test('xar trans',               benzina_geom, suite: 'geom', args: ['1920:1080', '2:1920'])

test('xar untrans crop',        benzina_geom, suite: 'geom', args: ['1920:1080', '4000:2'])
test('xar trans crop',          benzina_geom, suite: 'geom', args: ['1920:1080', '2:4000'])

test('untrans resize',          benzina_geom, suite: 'geom', args: ['1280:720', '1920:1080'])
test('trans resize',            benzina_geom, suite: 'geom', args: ['1280:720', '1080:1920'])

test('xar untrans resize crop', benzina_geom, suite: 'geom', args: ['1280:800', '1920:1080'])
test('xar trans resize crop',   benzina_geom, suite: 'geom', args: ['1280:800', '1080:1920'])

test('untrans resize pillar',   benzina_geom, suite: 'geom', args: ['1280:720', '1920:1200'])
test('trans resize pillar',     benzina_geom, suite: 'geom', args: ['1280:720', '1200:1920'])

