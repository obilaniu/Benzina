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
  bindir     = join_paths('benzina', 'lib')
  libdir     = join_paths('benzina', 'lib')
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
# FFmpeg
#
libavcodec  = dependency('libavcodec',   version: '>=57',  required: false)
libavformat = dependency('libavformat',  version: '>=57',  required: false)
libavfilter = dependency('libavfilter',  version: '>=6',   required: false)
libavutil   = dependency('libavutil',    version: '>=55',  required: false)
libswscale  = dependency('libswscale',   version: '>=4',   required: false)
HAVE_FFMPEG = (libavcodec.found()   and
               libavformat.found()  and
               libavfilter.found()  and
               libavutil.found()    and
               libswscale.found())
if HAVE_FFMPEG
  message('Enabling build with FFmpeg')
  ffmpegIncDir = libavutil.get_pkgconfig_variable('includedir',
                                                  default: '/usr/include/ffmpeg')
else
  message('Disabling build with FFmpeg')
  ffmpegIncDir = '/usr/include/ffmpeg'
endif
ENABLE_GPL = get_option('enable_gpl')
if ENABLE_GPL
  message('Enabling build with GPL license.')
endif


#
# Third-party libraries.
#
subdir('thirdparty')


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
  cudaGenCodeFlags += ['-g', '-lineinfo']
else
  cudaGenCodeFlags += ['-O3']
endif


#
# Compile generator.
#
nvccCompGen   = generator(nvcc,
    arguments: cudaGenCodeFlags+['-x', 'cu', '@EXTRA_ARGS@', '@INPUT@',
                                 '-I', nvcuvidIncDir,
                                 '-c', '-o', '@OUTPUT@'],
    output:    ['@BASENAME@.o'],
)

#
# Subdirectories.
#
subdir('include')
subdir('src')
subdir('doc')
