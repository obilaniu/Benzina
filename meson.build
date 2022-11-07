project('Benzina', 'c', 'cpp',
    version:         files('scripts/VERSION.txt'),
    meson_version:   '>=0.64.0',
    license:         'MIT',
    default_options: [
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
# Initial Meson module imports
#
fs   = import('fs')
ss   = import('sourceset')
CUDA = import('unstable-cuda')


#
# Compiler checks
#
# We also set here some predefined macros for assembler-language compilations
# by the C and C++ compilers.
#
cc          = meson.get_compiler('c')
cxx         = meson.get_compiler('cpp')
CC_ID       = cc.get_id()
CXX_ID      = cxx.get_id()
CCLD_ID     = cc.get_linker_id()
CC_SYNTAX   = cc.get_argument_syntax()
CXX_SYNTAX  = cxx.get_argument_syntax()
VOIDP       = 'void*'
CHARP       = 'char*'
FUNCP       = 'void(*)(void)'
VOIDP_SIZE  = cc.sizeof(VOIDP)
assert(VOIDP_SIZE == cxx.sizeof(VOIDP), f'C and C++ compilers disagree on sizeof(@VOIDP@)!')
CHARP_SIZE  = cc.sizeof(CHARP)
assert(CHARP_SIZE == cxx.sizeof(CHARP), f'C and C++ compilers disagree on sizeof(@CHARP@)!')
FUNCP_SIZE  = cc.sizeof(FUNCP)
assert(CHARP_SIZE == cxx.sizeof(FUNCP), f'C and C++ compilers disagree on sizeof(@FUNCP@)!')
assert(VOIDP_SIZE == CHARP_SIZE,        f'Pointers to void and char not of equal size!')
assert(VOIDP_SIZE == FUNCP_SIZE,        f'Pointers to void and function not of equal size!')
CCLD_FORCE_ID = CCLD_ID


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
                       'intel-cl': 'intel'}.get(CXX_ID, CXX_ID)
MACRO_CPU_FAMILY   += host_machine.cpu_family()
MACRO_OS           += host_machine.system()
MACRO_COMPILER_CC   = MACRO_COMPILER_CC .to_upper().underscorify()
MACRO_COMPILER_CXX  = MACRO_COMPILER_CXX.to_upper().underscorify()
MACRO_CPU_FAMILY    = MACRO_CPU_FAMILY  .to_upper().underscorify()
MACRO_OS            = MACRO_OS          .to_upper().underscorify()
add_global_arguments([f'-D@MACRO_COMPILER_CC@=1'],  language: 'c')
add_global_arguments([f'-D@MACRO_COMPILER_CXX@=1'], language: 'cpp')
add_global_arguments([f'-D@MACRO_CPU_FAMILY@=1',
                      f'-D@MACRO_OS@=1',
                      f'-DBENZINA_COMPILER_IDENTIFIED=1',
                      f'-DBENZINA_CPU_FAMILY_IDENTIFIED=1',
                      f'-DBENZINA_OS_IDENTIFIED=1'], language: ['c', 'cpp'])


#
# Add predefined macros for use by the preprocessed assembly code, identifying:
#
#   - Pointer size
#   - Underscore prefixing
#   - Section/segment+section names and attributes
#
# The section names must be <=15 characters long due to
#   A) macOS's Mach-O binary format imposing a limitation of 16 characters and
#   B) macOS's Mach-O binary format using a naming convention of
#          __SEGM,__sect
#      which we adhere to by underscorifying the equivalent ELF section name
#      and adding an additional prefix underscore.
#
add_global_arguments([f'-DASM_PTR_SIZE=@VOIDP_SIZE@',
                      f'-DASM_PTR_DECL=.@VOIDP_SIZE@byte'], language: ['c', 'cpp'])
SECT_DICT = {
    'BSS':                 {'name': '.bss',            'elf_attr': '',                    'macho_segm': '__DATA', 'macho_attr': ''},
    'CSTRING':             {'name': '.rodata.str1.1',  'elf_attr': '"aMS", @progbits, 1', 'macho_segm': '__TEXT', 'macho_attr': 'cstring_literals'},
    'LICENSE':             {'name': '.license',        'elf_attr': '"a"',                 'macho_segm': '__TEXT', 'macho_attr': 'regular, no_dead_strip'},
    'RODATALZ4':           {'name': '.rodata.lz4',     'elf_attr': '"a"',                 'macho_segm': '__TEXT', 'macho_attr': 'regular, no_dead_strip'},
    'LZ4CMDARRAY':         {'name': '.lz4.cmd_array',  'elf_attr': '"aw"',                'macho_segm': '__DATA', 'macho_attr': 'regular, no_dead_strip'},
    'TOOLARRAY':           {'name': '.tool_array',     'elf_attr': '"aw"',                'macho_segm': '__DATA', 'macho_attr': 'regular, no_dead_strip'},
    'LUAOPENARRAY':        {'name': '.lua.open_array', 'elf_attr': '"aw"',                'macho_segm': '__DATA', 'macho_attr': 'regular, no_dead_strip'},
    'LUATEXTARRAY':        {'name': '.lua.text_array', 'elf_attr': '"aw"',                'macho_segm': '__DATA', 'macho_attr': 'regular, no_dead_strip'},
}
foreach sect, dict : SECT_DICT
  if IS_HOST_DARWIN
    sect_name = dict['macho_segm'] + ',_' + {
        'CSTRING': '.cstring',         # Mach-O uses __cstring for mergeable strings
    }.get(sect, dict['name']).underscorify()
    sect_attr = dict['macho_attr']
  else
    sect_name = dict['name']
    sect_attr = dict['elf_attr']
  endif
  sect_attr = sect_attr == '' ? sect_attr : ', '+sect_attr
  add_global_arguments([
      f'-DASM_SECT_@sect@_NAME="@sect_name@"',
      f'-DASM_SECT_@sect@_DECL=.section @sect_name@@sect_attr@',
  ], language: ['c', 'cpp'])
endforeach


#
# Build subprojects we will use.
#
project_lz4       = subproject('lz4')
project_linenoise = subproject('linenoise')
project_lua       = subproject('lua')
project_lfs       = subproject('lfs')
project_lpeg      = subproject('lpeg')


#
# GPL check for permission
#
ENABLE_GPL = get_option('enable_gpl')
if ENABLE_GPL
  message('Enabling build with GPL license.')
endif


#
# Dependencies & Libraries
#
# Library and Include Search Paths
# More configurable defaults could surely be provided.
#
if CC_SYNTAX == 'gcc' or CXX_SYNTAX == 'gcc'
  relax_warnings = [
      '-Wno-pedantic',
      '-Wno-implicit-fallthrough',
      '-Wno-missing-field-initializers',
      '-Wno-missing-braces',
      '-Wno-unused-function',
  ]
  relax_warnings_c   = cc .get_supported_arguments(relax_warnings)
  relax_warnings_cxx = cxx.get_supported_arguments(relax_warnings)
  add_project_arguments(relax_warnings_c,   language: 'c')
  add_project_arguments(relax_warnings_cxx, language: 'cpp')
endif
force_link_args = []
special_link_args = []
if CC_SYNTAX == 'gcc'
  if CCLD_ID == 'ld.gold' or CCLD_ID == 'ld.mold'
    if   cc.has_link_argument('-fuse-ld=bfd')
      warning(f'The @CCLD_ID@ linker does not support our linker script. '+
              f'Forcing instead GNU ld.bfd (-fuse-ld=bfd)')
      force_link_args += ['-fuse-ld=bfd']
      CCLD_FORCE_ID = 'ld.bfd'
    elif cc.has_link_argument('-fuse-ld=lld')
      warning(f'The @CCLD_ID@ linker does not support our linker script. '+
              f'Forcing instead LLVM ld.lld (-fuse-ld=lld)')
      force_link_args += ['-fuse-ld=lld']
      CCLD_FORCE_ID = 'ld.lld'
    else
      error  (f'The @CCLD_ID@ linker does not support our linker script, '+
              f'but we are unable to force-select an alternative with the '+
              f'compiler flag -fuse-ld=[bfd|lld]. Aborting.')
    endif
  endif
  #
  # Before we could carry out feature tests on the linker, we had to determine
  # whether and how to pick an alternative to the default ld. We have now done
  # so, and proceed to the feature tests, forcing the alternate linker if
  # required. We cannot use cc.get_supported_link_arguments([...]) here because
  # it tests linker arguments individually rather than in unison with any
  # -fuse-ld= flag determined above.
  #
  foreach link_arg: ['now', 'relro', 'combreloc', 'noseparate-code',
                     'noexecstack', 'nocopyreloc', 'text']
    if cc.has_multi_link_arguments(force_link_args+['-Wl,-z,'+link_arg])
      special_link_args += [link_arg]
    endif
  endforeach
  if special_link_args.length() > 0
    special_link_args = ['-Wl,-z,'+',-z,'.join(special_link_args)]
  else
    special_link_args = []
  endif
endif
libm        = cc.find_library('m',  required: IS_HOST_LINUX)
librt       = cc.find_library('rt', required: IS_HOST_LINUX)
dl          = dependency('dl',      required: IS_HOST_LINUX)
openmp      = dependency('openmp',  required: false)
threads     = dependency('threads')


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
    foreach D : L.get_variable('includedir', default_value: '/usr/include/ffmpeg').split(':')
      if fs.is_dir(D) and D not in ffmpeg_incdir
        ffmpeg_incdir += [D]
      endif
    endforeach
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
