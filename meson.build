project('Benzina', 'c', 'cpp',
        version:                 '0.0.0',
        meson_version:           '>=0.46',
        license:                 'MIT',
        default_options:         [
            'c_std=gnu99',
            'warning_level=3',
        ],
)


#
# Import Meson Modules
#
pymodule = import('python')


#
# Python
#
# If building a Python package, we override the default install directories
# bindir, libdir and includedir.
#
PYINTERPRETER = get_option('py_interpreter')
if PYINTERPRETER == ''
  python = pymodule.find_installation()
else
  python = pymodule.find_installation(PYINTERPRETER)
endif

BUILDINGPYPKG = get_option('building_py_pkg')
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
# Compiler, Dependencies & Libraries
#
# Library and Include Search Paths
# More configurable defaults could surely be provided.
#
cc          = meson.get_compiler('c')
threadsDep  = dependency('threads')
libm        = cc.find_library('m',    required: false)
libdl       = cc.find_library('dl',   required: true)
librt       = cc.find_library('rt',   required: true)

# Relax a bit the warning flags on GCC & Clang. -Wpedantic is very noisy.
if cc.get_id() == 'gcc' or cc.get_id() == 'clang'
  add_project_arguments('-Wno-pedantic', language: 'c')
  add_project_arguments('-Wno-pedantic', language: 'cpp')
endif


#
# NVIDIA
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
libcuda      = cc.find_library('cuda',          required: true)
libcudart    = cc.find_library(CUDA_RUNTIME,    required: true, dirs: cudaLibDirs)
libcudadevrt = cc.find_library('cudadevrt',     required: true, dirs: cudaLibDirs)
libnvcuvid   = cc.find_library('nvcuvid',       required: true)
nvcc         = find_program(join_paths(CUDA_HOME, 'bin', 'nvcc'), required: true)

#
# NVDECODE/Video Codec SDK feature detection
#
if   cc.has_function('cuvidReconfigureDecoder', dependencies: [libnvcuvid])
  message('Found Video Codec SDK >=8.2 (NVIDIA Driver >=396.24 will be required)')
  nvcuvidCDefs = ['-DNVDECODE=8002']
elif cc.has_function('cuvidGetDecoderCaps',     dependencies: [libnvcuvid])
  message('Found Video Codec SDK ==8.1 (NVIDIA Driver >=390.25 will be required)')
  nvcuvidCDefs = ['-DNVDECODE=8001']
else
  message('Found Video Codec SDK <=8.0 (NVIDIA Driver >=378.13 will be required)')
  nvcuvidCDefs = ['-DNVDECODE=8000']
endif

#
# GPU presence detection and architecture selection
#
CUDA_ARCH    = get_option('cuda_arch')
detectGPUpy  = files(join_paths('scripts', 'detectgpus.py'))
detectGPU    = run_command(python, [detectGPUpy, nvcc, CUDA_ARCH])
if detectGPU.returncode() != 0
  warning('GPU detection failed:')
  warning(detectGPU.stdout())
  warning(detectGPU.stderr())
  cudaGenCodeFlags = []
else
  message(detectGPU.stderr())
  cudaGenCodeFlags = detectGPU.stdout().split()
endif


#
# Subdirectories.
#
subdir('include')
subdir('src')
