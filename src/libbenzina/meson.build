#
# Library: libbenzina
#
# Contains a lot of the "generic" logic for data-loading, as well as ProtoBuf
# parsing code.
#

libbenzina_dict     = {
    'c_args':     ['-DBENZINA_IS_SHARED=1', '-DBENZINA_IS_BUILDING=1'],
    'link_args':  ['-pie', '-Wl,-E'],
    'link_with':  [],
    'link_whole': [],
    'link_deps':  [],
    'objs': [],
    'deps': [
        threads, libm, libdl,
        project_linenoise.get_variable('linenoise_dep'),
        project_lua      .get_variable('custom_lua_dep'),
        project_lfs      .get_variable('lfs_dep'),
    ],
    'incs': [benzina_incs],
    'srcs': files('benzina.c',
                  'benzinaproto.c',
                  'endian.c',
                  'init.c',
                  'intops.c',
                  'ptrops.c',
                  'version.c',
                  'main.c')
}

subdir('os')
subdir('iso')
subdir('itu')


#
# Define the shared library and dependency.
#

libbenzina_sha = shared_library('benzina',             libbenzina_dict['srcs'],
                                objects              : libbenzina_dict['objs'],
                                include_directories  : libbenzina_dict['incs'],
                                dependencies         : libbenzina_dict['deps'],
                                c_args               : libbenzina_dict['c_args'],
                                link_args            : libbenzina_dict['link_args'],
                                link_with            : libbenzina_dict['link_with'],
                                link_whole           : libbenzina_dict['link_whole'],
                                link_depends         : libbenzina_dict['link_deps'],
                                gnu_symbol_visibility: 'hidden',
                                version              : meson.project_version(),
                                build_rpath          : librpath,
                                install_rpath        : librpath,
                                install              : true,
                                install_dir          : libdir)
libbenzina     = declare_dependency(include_directories: benzina_incs,
                                    compile_args       : ['-DBENZINA_IS_SHARED=1'],
                                    link_with          : libbenzina_sha)

