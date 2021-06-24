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
  bindir     = 'benzina'/'lib'
  libdir     = 'benzina'/'lib'
  includedir = 'benzina'/'include'
  tooldir    = bindir/'tools'
  librpath   = '$ORIGIN'
  toolrpath  = librpath+':'+librpath/'..'
else
  bindir     = get_option('bindir')
  libdir     = get_option('libdir')
  includedir = get_option('includedir')
  tooldir    = bindir/'tools'
  librpath   = ''
  toolrpath  = ''
endif


#
# Declare GPU architecture auto-detection test source code
# for possible use later on.
#
detectgpus_c = files('detectgpus.c')


#
# ELF hacking tools
#
elf_py                = files('elf.py')
elf_strip_df_1_pie_py = files('elf-strip-df-1-pie.py')


#
# Pack LZ4-compressed Lua code inside ELF
#
lz4_lua_pack_py = files('lz4-lua-pack.py')
