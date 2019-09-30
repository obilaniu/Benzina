#
# Library: libbenzina
#
# Contains a lot of the "generic" logic for data-loading, as well as ProtoBuf
# parsing code.
#

libbenzina_srcs     = files('benzina.c',
                            'benzinaproto.c',
                            'endian.c',
                            'init.c',
                            'intops.c',
                            'iso.c',
                            'version.c')
subdir('itu')
libbenzina_map_name = 'libbenzina.map'
libbenzina_map_file = files(libbenzina_map_name)
libbenzina_map      = meson.current_source_dir()/libbenzina_map_name
libbenzina_deps     = [threads, libm, libdl]
libbenzina_incs     = [benzina_incs]
libbenzina_cdefs    = ['-DBENZINA_IS_SHARED=1']
libbenzina_linkargs = []
if cc.get_id() == 'gcc'
  libbenzina_linkargs += ['-Wl,--version-script='+libbenzina_map]
endif
libbenzina_sha = shared_library('benzina',             libbenzina_srcs,
                                include_directories  : libbenzina_incs,
                                dependencies         : libbenzina_deps,
                                c_args               : libbenzina_cdefs + ['-DBENZINA_IS_BUILDING=1'],
                                link_args            : libbenzina_linkargs,
                                link_with            : [],
                                link_depends         : libbenzina_map_file,
                                gnu_symbol_visibility: 'hidden',
                                version              : meson.project_version(),
                                build_rpath          : librpath,
                                install_rpath        : librpath,
                                install              : true,
                                install_dir          : libdir)
libbenzina     = declare_dependency(include_directories: benzina_incs,
                                    compile_args       : libbenzina_cdefs,
                                    link_with          : libbenzina_sha)

