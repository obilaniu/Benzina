#
# Documentation
#

sphinx = find_program('sphinx-build')
if sphinx.found()
  custom_target('doc',
                build_by_default: false,
                command         : [sphinx, '-M', 'html',
                                   meson.current_source_dir(),
                                   meson.current_build_dir()],
                input           : [],
                output          : ['bla'])
endif

