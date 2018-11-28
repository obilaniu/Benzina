#
# Plugin: libbenzina-plugin-skeleton
#
# A skeleton for a Benzina plugin. Meant to be copy-pasted and used as a template.
#


libbenzina_plugin_skeletonSrcs = files('main.c')
libbenzina_plugin_skeletonDeps = [threadsDep, libm, libdl, librt,
                                  libcuda, libcudart, libcudadevrt, libnvcuvid,
                                  libbenzina, libbenzinacuda, pluginDep]
libbenzina_plugin_skeletonSha  = shared_library('benzina-plugin-skeleton',
                                                libbenzina_plugin_skeletonSrcs,
                                                dependencies         : libbenzina_plugin_skeletonDeps,
                                                gnu_symbol_visibility: 'hidden',
                                                build_rpath          : librpath,
                                                install_rpath        : librpath,
                                                install              : true,
                                                install_dir          : libdir)

