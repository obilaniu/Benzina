#
# PLUGINS
#
# Every plugin should be built against libbenzina. This entails providing the
# include directory, the library itself, and certain specific compile flags.
# A declared dependency allow us to add these components together quickly to
# new plugins.
#

pluginCDefs  = ['-DBENZINA_PLUGIN_IS_BUILDING=1',
                '-DBENZINA_PLUGIN_IS_SHARED=1'] + libbenzinaCDefs
pluginCDefs += libbenzinacuda.found() ? libbenzinacudaCDefs : []
pluginDep    = declare_dependency(dependencies: [libbenzina],
                                  compile_args: pluginCDefs)



#
# There is so far only one "real" plugin: nvdecode.
#
subdir('skeleton')         # Template.
subdir('nvdecode')

