#
# Plugin: libbenzina-plugin-skeleton
#
# A skeleton for a Benzina plugin. Meant to be copy-pasted and used as a template.
#


### Dependency Check
if not true
  subdir_done()
endif


### C code compilation
libbenzina_plugin_skeletonSrcs = files('main.c')
libbenzina_plugin_skeletonDeps = [threads, dl, libm,
                                  libcuda, libcudart, libcudadevrt, libnvcuvid,
                                  libbenzina, plugin]
libbenzina_plugin_skeletonSha  = shared_library('benzina-plugin-skeleton',
                                                libbenzina_plugin_skeletonSrcs,
                                                dependencies         : libbenzina_plugin_skeletonDeps,
                                                gnu_symbol_visibility: 'hidden',
                                                build_rpath          : librpath,
                                                install_rpath        : librpath,
                                                install              : false,
                                                install_dir          : libdir)

