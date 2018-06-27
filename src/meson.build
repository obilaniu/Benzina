#
# Library: libbenzina
#
# Contains a lot of the "generic" logic for data-loading, as well as ProtoBuf
# parsing code.
#

libbenzinaSrcs  = files('benzina.c', 'benzinaproto.c')
libbenzinaDeps  = [threadsDep, libm, libdl, librt]
libbenzinaIncs  = [cudaIncs, benzinaIncs]
libbenzinaCArgs = ['-DBENZINA_IS_BUILDING=1',
                   '-DBENZINA_IS_SHARED=1',
                   '-DBENZINA_PLUGIN_IS_SHARED=1']
if build_machine.system() != 'windows'
  libbenzinaCArgs += ['-fvisibility=hidden']
endif
libbenzinaSha   = shared_library('benzina',           libbenzinaSrcs,
                                 include_directories: libbenzinaIncs,
                                 dependencies       : libbenzinaDeps,
                                 c_args             : libbenzinaCArgs,
                                 build_rpath        : '$ORIGIN',
                                 install_rpath      : '$ORIGIN',
                                 install            : true,
                                 install_dir        : libdir)


#
# Build selected plugins.
#
subdir('plugins')
