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
# Perform setup work for Python and other third-party libraries.
# Then, build project's main artifacts: libraries, executables, documentation, ...
#
subdir('scripts')
subdir('thirdparty')
subdir('include')
subdir('src')
subdir('doc')
