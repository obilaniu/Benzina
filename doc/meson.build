#
# Documentation
#

sphinx = find_program('sphinx-build')

if sphinx.found()
  custom_target('doc',
                build_by_default: false,
                command         : [sphinx, '-M', 'html',
                                   join_paths(meson.current_source_dir(), 'source'),
                                   meson.current_build_dir()],
                input           : [],
                output          : ['html'])
endif
