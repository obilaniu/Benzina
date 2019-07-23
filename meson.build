project('Benzina', 'c', 'cpp',
        version:                 '0.0.3',
        meson_version:           '>=0.50.0',
        license:                 'MIT',
        default_options:         [
            'c_std=gnu99',
            'warning_level=3',
            'b_ndebug=if-release',
        ],
)


#
# System check
#
IS_CROSS        = meson.is_cross_build()
IS_NATIVE       = not IS_CROSS
IS_HOST_LE      = host_machine.endian()     == 'little'
IS_HOST_X86_64  = host_machine.cpu_family() == 'x86_64'
IS_HOST_AARCH64 = host_machine.cpu_family() == 'aarch64'
IS_HOST_PPC64   = host_machine.cpu_family() == 'ppc64'
IS_HOST_PPC64LE = IS_HOST_PPC64 and IS_HOST_LE
IS_HOST_LINUX   = host_machine.system()     == 'linux'
IS_HOST_DARWIN  = host_machine.system()     == 'darwin'
IS_HOST_WINDOWS = host_machine.system()     == 'windows'
if   not IS_HOST_X86_64 and not IS_HOST_PPC64LE and not IS_HOST_AARCH64
  warning('The CUDA Toolkit is not built for architectures other than x64_64, '+
          'ppc64le and aarch64. No CUDA acceleration of any kind whatsoever is '+
          'therefore possible, which defeats the point of Benzina. A pure CPU '+
          'fallback could be implemented in principle, but it would be very '+
          'slow, except on massively-multithreaded machines.')
endif
if   IS_HOST_DARWIN
  warning('Benzina is unsupported on Mac OS X/macOS/iOS because NVIDIA does not '+
          'provide required libraries (nvcuvid) on Apple platforms. Apple '+
          'instead mandates the use of its own mystery APIs. As a result, this '+
          'build is extremely likely to fail even if this check were disabled, '+
          'unless someone were to contribute Mac OS X-specific enablement code.')
elif IS_HOST_WINDOWS
  warning('Benzina is extremely unlikely to build or run on Windows due to '+
          'widespread use of Linuxisms and low-level system calls, but '+
          'Windows equivalents or approximations do exist and Benzina could '+
          'in principle be ported to that OS.')
endif


#
# Compiler, Dependencies & Libraries
#
# Library and Include Search Paths
# More configurable defaults could surely be provided.
#
cc          = meson.get_compiler('c')
cxx         = meson.get_compiler('cpp')
CC_SYNTAX   = cc.get_argument_syntax()
CXX_SYNTAX  = cxx.get_argument_syntax()
# Relax a bit the warning flags on GCC & Clang. -Wpedantic is very noisy.
if CC_SYNTAX == 'gcc' or CXX_SYNTAX == 'gcc'
  relaxWarnings    = ['-Wno-pedantic',
                      '-Wno-implicit-fallthrough',
                      '-Wno-missing-field-initializers',
                      '-Wno-missing-braces',
                      '-Wno-unused-function']
  relaxWarningsC   = cc .get_supported_arguments(relaxWarnings)
  relaxWarningsCxx = cxx.get_supported_arguments(relaxWarnings)
  add_project_arguments(relaxWarningsC,   language: 'c')
  add_project_arguments(relaxWarningsCxx, language: 'cpp')
endif
libm        = cc.find_library('m',  required: IS_HOST_LINUX)
libdl       = cc.find_library('dl', required: IS_HOST_LINUX)
librt       = cc.find_library('rt', required: IS_HOST_LINUX)
threadsDep  = dependency('threads')


#
# FFmpeg
#
libavcodec   = dependency('libavcodec',  version: '>=57', required: false, disabler: true)
libavformat  = dependency('libavformat', version: '>=57', required: false, disabler: true)
libavfilter  = dependency('libavfilter', version: '>=6',  required: false, disabler: true)
libavutil    = dependency('libavutil',   version: '>=55', required: false, disabler: true)
libswscale   = dependency('libswscale',  version: '>=4',  required: false, disabler: true)
ffmpegIncDir = []
ffmpegIFlags = []
ffmpegNames  = []
foreach L : [libavcodec, libavformat, libavfilter, libavutil, libswscale]
  if L.found()
    ffmpegNames += [L.name() + '-' + L.version()]#.split('.')[0]
    if L.type_name() == 'pkgconfig'
      foreach D : L.get_pkgconfig_variable('includedir',
                                           default: '/usr/include/ffmpeg').split(':')
        ffmpegIncDir += D in ffmpegIncDir ? [] : [D]
      endforeach
    endif
  endif
endforeach
foreach D : ffmpegIncDir
  ffmpegIFlags += ['-I'+D]
endforeach
if ffmpegNames.length() > 0
  message('Enabling build with FFmpeg (@0@)'.format(', '.join(ffmpegNames)))
else
  message('Disabling build with FFmpeg')
endif
ENABLE_GPL = get_option('enable_gpl')
if ENABLE_GPL
  message('Enabling build with GPL license.')
endif


#
# Perform setup work for Python and other third-party libraries.
# Then, build project's main artifacts: libraries, executables, documentation, ...
#
subdir('scripts')
subdir('thirdparty')
subdir('include')
subdir('src')
