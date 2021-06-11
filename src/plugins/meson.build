#
# PLUGINS
#
# Every plugin should be built against libbenzina. This entails providing the
# include directory, the library itself, and certain specific compile flags.
# A declared dependency allow us to add these components together quickly to
# new plugins.
#

plugin_cdefs  = ['-DBENZINA_PLUGIN_IS_BUILDING=1',
                 '-DBENZINA_PLUGIN_IS_SHARED=1']
plugin        = declare_dependency(dependencies: [libbenzina],
                                   compile_args: plugin_cdefs)



#
# There is so far only one "real" plugin: nvdecode.
#
subdir('skeleton')         # Template.
subdir('nvdecode')

