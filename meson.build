project('Benzina', 'c', 'cpp',
        version:                 '0.0.4',
        meson_version:           '>=0.51.1',
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
HOST_SYSTEM     = host_machine.system()
IS_CROSS        = meson.is_cross_build()
IS_NATIVE       = not IS_CROSS
IS_HOST_LE      = host_machine.endian()     == 'little'
IS_HOST_BE      = host_machine.endian()     == 'big'
IS_HOST_X86_64  = host_machine.cpu_family() == 'x86_64'
IS_HOST_AARCH64 = host_machine.cpu_family() == 'aarch64'
IS_HOST_PPC64   = host_machine.cpu_family() == 'ppc64'
IS_HOST_PPC64LE = IS_HOST_PPC64 and IS_HOST_LE
IS_HOST_LINUX   = HOST_SYSTEM               == 'linux'
IS_HOST_DARWIN  = HOST_SYSTEM               == 'darwin'
IS_HOST_WINDOWS = HOST_SYSTEM               == 'windows'
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
# GPL check for permission
#
ENABLE_GPL = get_option('enable_gpl')
if ENABLE_GPL
  message('Enabling build with GPL license.')
endif


#
# Compiler, Dependencies & Libraries
#
# Library and Include Search Paths
# More configurable defaults could surely be provided.
#
cc          = meson.get_compiler('c')
cxx         = meson.get_compiler('cpp')
CC_ID       = cc.get_id()
CXX_ID      = cxx.get_id()
CC_SYNTAX   = cc.get_argument_syntax()
CXX_SYNTAX  = cxx.get_argument_syntax()
if CC_SYNTAX == 'gcc' or CXX_SYNTAX == 'gcc'
  relax_warnings     = ['-Wno-pedantic',
                        '-Wno-implicit-fallthrough',
                        '-Wno-missing-field-initializers',
                        '-Wno-missing-braces',
                        '-Wno-unused-function']
  relax_warnings_c   = cc .get_supported_arguments(relax_warnings)
  relax_warnings_cxx = cxx.get_supported_arguments(relax_warnings)
  add_project_arguments(relax_warnings_c,   language: 'c')
  add_project_arguments(relax_warnings_cxx, language: 'cpp')
endif
libm        = cc.find_library('m',  required: IS_HOST_LINUX)
libdl       = cc.find_library('dl', required: IS_HOST_LINUX)
librt       = cc.find_library('rt', required: IS_HOST_LINUX)
openmp      = dependency('openmp',  required: false)
threads     = dependency('threads')


#
# Add predefined macros identifying the:
#   - C & C++ compilers
#   - CPU family
#   - Operating System
#
MACRO_COMPILER_CC   = 'BENZINA_COMPILER_'
MACRO_COMPILER_CXX  = 'BENZINA_COMPILER_'
MACRO_CPU_FAMILY    = 'BENZINA_CPU_FAMILY_'
MACRO_OS            = 'BENZINA_OS_'
MACRO_COMPILER_CC  += {'clang-cl': 'clang',
                       'intel-cl': 'intel'}.get(CC_ID,  CC_ID)
MACRO_COMPILER_CXX += {'clang-cl': 'clang',
                       'intel-cl': 'intel'}.get(CXX_ID,  CXX_ID)
MACRO_CPU_FAMILY   += host_machine.cpu_family()
MACRO_OS           += host_machine.system()
MACRO_COMPILER_CC   = MACRO_COMPILER_CC .to_upper().underscorify()
MACRO_COMPILER_CXX  = MACRO_COMPILER_CXX.to_upper().underscorify()
MACRO_CPU_FAMILY    = MACRO_CPU_FAMILY  .to_upper().underscorify()
MACRO_OS            = MACRO_OS          .to_upper().underscorify()

add_project_arguments(['-D'+MACRO_COMPILER_CC+'=1',
                       '-DBENZINA_COMPILER_IDENTIFIED=1'], language: 'c')
add_project_arguments(['-D'+MACRO_COMPILER_CXX+'=1',
                       '-DBENZINA_COMPILER_IDENTIFIED=1'], language: 'cpp')

add_project_arguments(['-D'+MACRO_CPU_FAMILY+'=1',
                       '-DBENZINA_CPU_FAMILY_IDENTIFIED=1'], language: 'c')
add_project_arguments(['-D'+MACRO_CPU_FAMILY+'=1',
                       '-DBENZINA_CPU_FAMILY_IDENTIFIED=1'], language: 'cpp')

add_project_arguments(['-D'+MACRO_OS+'=1',
                       '-DBENZINA_OS_IDENTIFIED=1'], language: 'c')
add_project_arguments(['-D'+MACRO_OS+'=1',
                       '-DBENZINA_OS_IDENTIFIED=1'], language: 'cpp')


#
# Configuration Data
#
# Begin collecting configuration information as we configure the build.
#
configdata = configuration_data()
configdata.set  ('LIBBENZINA_VERSION_MAJOR', meson.project_version().split('.')[0].to_int(),
                 description: 'The major version of libbenzina.')
configdata.set  ('LIBBENZINA_VERSION_MINOR', meson.project_version().split('.')[1].to_int(),
                 description: 'The minor version of libbenzina.')
configdata.set  ('LIBBENZINA_VERSION_PATCH', meson.project_version().split('.')[2].to_int(),
                 description: 'The patch version of libbenzina.')
configdata.set10('BENZINA_ENDIAN_BIG',       IS_HOST_BE,
                 description: 'Whether host machine is big-endian (1) or not (0).')
configdata.set10('BENZINA_ENDIAN_LITTLE',    IS_HOST_LE,
                 description: 'Whether host machine is little-endian (1) or not (0).')


#
# FFmpeg
#
libavcodec    = dependency('libavcodec',  version: '>=57', required: false, disabler: true)
libavformat   = dependency('libavformat', version: '>=57', required: false, disabler: true)
libavfilter   = dependency('libavfilter', version: '>=6',  required: false, disabler: true)
libavutil     = dependency('libavutil',   version: '>=55', required: false, disabler: true)
libswscale    = dependency('libswscale',  version: '>=4',  required: false, disabler: true)
ffmpeg_incdir = []
ffmpeg_iflags = []
ffmpeg_names  = []
foreach L : [libavcodec, libavformat, libavfilter, libavutil, libswscale]
  if L.found()
    ffmpeg_names += [L.name() + '-' + L.version()]
    if L.type_name() == 'pkgconfig'
      foreach D : L.get_pkgconfig_variable('includedir',
                                           default: '/usr/include/ffmpeg').split(':')
        ffmpeg_incdir += D in ffmpeg_incdir ? [] : [D]
      endforeach
    endif
  endif
endforeach
foreach D : ffmpeg_incdir
  ffmpeg_iflags += ['-I'+D]
endforeach
if ffmpeg_names.length() > 0
  message('Enabling build with FFmpeg (@0@)'.format(', '.join(ffmpeg_names)))
else
  message('Disabling build with FFmpeg')
endif


#
# Perform setup work for Python and other third-party libraries.
# Then, build project's main artifacts: libraries, executables, documentation, ...
#
subdir('scripts')
subdir('thirdparty')
subdir('include')
subdir('src')
subdir('tests')
