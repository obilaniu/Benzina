project('Benzinadoc')

#
# Documentation
#

sphinx = find_program('sphinx-build')
sphinx_source_dir = 'source'

if sphinx.found()
  custom_target('doc',
                build_by_default: false,
                command         : [sphinx, '-M', 'html',
                                   meson.current_source_dir()+'/'+sphinx_source_dir,
                                   meson.current_build_dir()],
                input           : [],
                output          : ['html'])
endif
