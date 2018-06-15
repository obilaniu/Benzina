project('Benzina', 'c', 'cpp',
        version:                 '0.0.0',
        meson_version:           '>=0.45',
        license:                 'MIT',
        default_options:         [
            'c_std=gnu99',
            'warning_level=2',
        ],
)


#
# Dependencies & Libraries
#
# Library and Include Search Paths
# More configurable defaults could surely be provided.
#
cc          = meson.get_compiler('c')
threadsDep  = dependency('threads')
libm        = cc.find_library('m',    required: true)
libdl       = cc.find_library('dl',   required: true)
librt       = cc.find_library('rt',   required: true)


#
# CUDA Toolkit
#
CUDA_HOME    = get_option('cuda_home')
if get_option('cuda_runtime') == 'static'
  CUDA_RUNTIME = 'cudart_static'
else
  CUDA_RUNTIME = 'cudart'
endif
cudaLibDirs  = join_paths(CUDA_HOME, 'lib64')
cudaIncDirs  = join_paths(CUDA_HOME, 'include')
cudaIncs     = include_directories(cudaIncDirs)
libcuda      = cc.find_library('cuda',          required: true, dirs: cudaLibDirs)
libcudadevrt = cc.find_library('cudadevrt',     required: true, dirs: cudaLibDirs)
libcudart    = cc.find_library(CUDA_RUNTIME,    required: true, dirs: cudaLibDirs)
libnvcuvid   = cc.find_library('nvcuvid',       required: true, dirs: cudaLibDirs)
nvcc         = find_program(join_paths(CUDA_HOME, 'bin', 'nvcc'), required: true)


#
# If building a Python package, we override the default install directories
# bindir, libdir and includedir.
#
BUILDINGPYPKG = get_option('setuptools_driving_meson')
if BUILDINGPYPKG
  message('Entering special Python-package build mode. Overriding bindir, libdir & includedir.')
  bindir     = join_paths('benzina', 'libs')
  libdir     = join_paths('benzina', 'libs')
  includedir = join_paths('benzina', 'include')
else
  bindir     = get_option('bindir')
  libdir     = get_option('libdir')
  includedir = get_option('includedir')
endif


#
# Subdirectories.
#
subdir('include')
subdir('src')
