project('Benzina', 'c', 'cpp',
        version:                 '0.0.3',
        meson_version:           '>=0.48',
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
  tooldir    = join_paths(bindir, 'tools')
else
  bindir     = get_option('bindir')
  libdir     = get_option('libdir')
  includedir = get_option('includedir')
  tooldir    = join_paths(bindir, 'tools')
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
  add_project_arguments(['-Wno-pedantic',
                         '-Wno-implicit-fallthrough'], language: 'c')
  add_project_arguments(['-Wno-pedantic',
                         '-Wno-implicit-fallthrough'], language: 'cpp')
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
nvccPath      = join_paths(CUDA_HOME, 'bin', 'nvcc')
toolkitLibDir = join_paths(CUDA_HOME, 'lib64')
toolkitIncDir = join_paths(CUDA_HOME, 'include')
driverLibDirs = [toolkitLibDir, '/usr/lib64', '/usr/lib64/nvidia']
cudaIncs      = include_directories(toolkitIncDir)
libcuda       = cc.find_library('cuda',          required: true, dirs: driverLibDirs)
libcudart     = cc.find_library(CUDA_RUNTIME,    required: true, dirs: toolkitLibDir)
libcudadevrt  = cc.find_library('cudadevrt',     required: true, dirs: toolkitLibDir)
libnvcuvid    = cc.find_library('nvcuvid',       required: true, dirs: driverLibDirs)
nvcc          = find_program(nvccPath, 'nvcc', required: true)

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
if(get_option('buildtype') == 'debugoptimized' or
   get_option('buildtype') == 'debug')
  cudaGenCodeFlags += ['-g']
else
  cudaGenCodeFlags += ['-O3']
endif

#
# Compile generator.
#
nvccCompGen   = generator(nvcc,
    arguments: cudaGenCodeFlags+['-x', 'cu', '@EXTRA_ARGS@', '@INPUT@',
                                 '-c', '-o', '@OUTPUT@'],
    output:    ['@BASENAME@.o'],
)

#
# NVDECODE/Video Codec SDK feature detection
#
if   cc.has_function('cuvidReconfigureDecoder', dependencies: [libnvcuvid])
  message('Found Video Codec SDK >= 8.2')
  message('Info: Video Codec SDK 8.2 requires NVIDIA Driver >= 396.24 and mandatory CUDA Toolkit >= 8.0')
  nvcuvidCDefs = ['-DNVDECODE=8002']
elif cc.has_function('cuvidGetDecoderCaps',     dependencies: [libnvcuvid])
  message('Found Video Codec SDK == 8.0 or 8.1')
  message('Info: Video Codec SDK 8.1 requires NVIDIA Driver >= 390.25 and mandatory CUDA Toolkit >= 8.0')
  message('Info: Video Codec SDK 8.0 requires NVIDIA Driver >= 378.13 and optional  CUDA Toolkit >= 8.0')
  nvcuvidCDefs = ['-DNVDECODE=8000']
else
  message('Found Video Codec SDK <  8.0')
  message('Info: Video Codec SDK 7.1 requires NVIDIA Driver >= 375.20 and optional CUDA Toolkit >= 7.5')
  message('Info: Video Codec SDK 7.0 requires NVIDIA Driver >= 367.35 and optional CUDA Toolkit >= 7.5')
  nvcuvidCDefs = ['-DNVDECODE=7000']
endif


#
# FFmpeg
#
libavcodec  = dependency('libavcodec',   version: '>=58',  required: false)
libavformat = dependency('libavformat',  version: '>=58',  required: false)
libavfilter = dependency('libavfilter',  version: '>=7',   required: false)
libavutil   = dependency('libavutil',    version: '>=56',  required: false)
libswscale  = dependency('libswscale',   version: '>=5',   required: false)
HAVE_FFMPEG = (libavcodec.found()   and
               libavformat.found()  and
               libavfilter.found()  and
               libavutil.found()    and
               libswscale.found())
if HAVE_FFMPEG
  message('Enabling build with FFmpeg')
else
  message('Disabling build with FFmpeg')
endif


#
# Enable linking against GPL software?
#
ENABLE_GPL = get_option('enable_gpl')
if ENABLE_GPL
  message('Enabling GPL build.')
endif


#
# Subdirectories.
#
subdir('include')
subdir('src')
