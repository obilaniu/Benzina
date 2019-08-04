#
# Library: libbenzina
#
# Contains a lot of the "generic" logic for data-loading, as well as ProtoBuf
# parsing code.
#

libbenzinaSrcs  = files('benzina.c',
                        'benzinaproto.c',
                        'endian.c',
                        'init.c',
                        'iso.c',
                        'version.c')
libbenzinaDeps  = [threadsDep, libm, libdl]
libbenzinaIncs  = [benzinaIncs]
libbenzinaCDefs = ['-DBENZINA_IS_SHARED=1']
libbenzinaLinkArgs = []
if cc.get_id() == 'gcc'
  libbenzinaLinkArgs += ['-Wl,--version-script='+meson.current_source_dir()/'libbenzina.map']
endif
libbenzinaSha   = shared_library('benzina',             libbenzinaSrcs,
                                 include_directories  : libbenzinaIncs,
                                 dependencies         : libbenzinaDeps,
                                 c_args               : libbenzinaCDefs + ['-DBENZINA_IS_BUILDING=1'],
                                 link_args            : libbenzinaLinkArgs,
                                 link_with            : [],
                                 gnu_symbol_visibility: 'hidden',
                                 build_rpath          : librpath,
                                 install_rpath        : librpath,
                                 install              : true,
                                 install_dir          : libdir)
libbenzina      = declare_dependency(include_directories: benzinaIncs,
                                     compile_args       : libbenzinaCDefs,
                                     link_with          : libbenzinaSha)

