#
# Documentation
#

sphinxbuild = find_program('sphinx-build', required: false, native: true)
if sphinxbuild.found()
  run_target('doc', command: [
      sphinxbuild, '-M', 'html',
      meson.current_source_dir()/'source',
      meson.current_build_dir(),
  ])
endif
