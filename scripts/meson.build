#
# Python
#
# If building a Python package, we override the default install directories
# bindir, libdir and includedir.
#
pymodule = import('python')
PYINTERPRETER = get_option('py_interpreter')
if PYINTERPRETER == ''
  python = pymodule.find_installation()
else
  python = pymodule.find_installation(PYINTERPRETER)
endif

BUILDINGPYPKG = get_option('building_py_pkg')
if BUILDINGPYPKG
  message('Entering special Python-package build mode. Overriding bindir, libdir & includedir.')
  bindir     = join_paths('benzina', 'lib')
  libdir     = join_paths('benzina', 'lib')
  includedir = join_paths('benzina', 'include')
  tooldir    = join_paths(bindir, 'tools')
  librpath   = '$ORIGIN'
  toolrpath  = librpath+':'+join_paths(librpath, '..')
else
  bindir     = get_option('bindir')
  libdir     = get_option('libdir')
  includedir = get_option('includedir')
  tooldir    = join_paths(bindir, 'tools')
  librpath   = ''
  toolrpath  = ''
endif


#
# Important Python scripts used elsewhere
#
detectGPUpy  = files('detectgpus.py')

