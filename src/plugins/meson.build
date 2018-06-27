#
# PLUGINS.
#

pluginCDefs = ['-DBENZINA_PLUGIN_IS_BUILDING=1',
              '-DBENZINA_PLUGIN_IS_SHARED=1',
              '-DBENZINA_IS_SHARED=1']
if build_machine.system() == 'windows'
  pluginCArgs = pluginCDefs
else
  pluginCArgs = pluginCDefs + ['-fvisibility=hidden']
endif



#
# So far, there is just one plugin: nvdecode.
#
subdir('nvdecode')
