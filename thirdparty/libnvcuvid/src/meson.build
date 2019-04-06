#
# Source Directory of libnvcuvid *stub*.
#
# This directory holds a renamed dynlink_cuda.h -> cuda.h header for the sole
# purpose of building the stub. It is not complete, and it must therefore
# *NOT* be made available for inclusion by any target except this stub. This
# implies not adding this directory to any other search paths except the one
# for this stub.
#

nvcuvidLinkArgs = []
nvcuvidCArgs    = ['-D__CUVID_INTERNAL']
nvcuvidKwargs   = IS_HOST_WINDOWS ? {} : {'soversion': '1'}
cuda_h          = configure_file(input:  dynlink_cuda_h,
                                 copy:   true,
                                 output: 'cuda.h')


#
# We now build the stub.
#
# *Exclusively* for this target, we add the current directory explictly to the
# include directories, since this directory contains an (incomplete) cuda.h.
#
# However, we do not add the current directory to the includes of the *users*
# of the stub.
#
# Various options are enabled that silence warnings and reduce library size,
# given that the library is only a stub and will never be used at runtime.
#
if CC_SYNTAX == 'gcc'
  nvcuvidCArgs    += ['-w',
                      '-fno-exceptions',
                      '-fno-unwind-tables',
                      '-fno-asynchronous-unwind-tables']
  nvcuvidLinkArgs += ['-nostartfiles',
                      '-Wl,-s,--build-id=none,--hash-style=sysv,-z,norelro']
endif
libnvcuvidSha   = shared_library('nvcuvid', [files('libnvcuvid.c'), cuda_h],
                                 include_directories: nvcuvidIncs + ['.'],
                                 c_args             : nvcuvidCArgs + nvcuvidCDefs,
                                 link_args          : nvcuvidLinkArgs,
                                 kwargs             : nvcuvidKwargs,
                                 install            : false,
                                 override_options   : ['warning_level=0',
                                                       'buildtype=release',
                                                       'optimization=s',
                                                       'debug=false',
                                                       'werror=false',
                                                       'strip=true',
                                                       'b_ndebug=true'])


#
# And we now declare a Meson dependency usable elsewhere. But before doing
# that, if the CUDA Toolkit is in fact available, we add its includes to
# make libnvcuvid fully functional as a stub.
#
if nvcc.found()
  nvcuvidIFlags  += ['-I'+toolkitIncDir]
  nvcuvidIncs    += [include_directories(toolkitIncDir)]
endif
libnvcuvid      = declare_dependency(include_directories: nvcuvidIncs,
                                     compile_args       : nvcuvidCDefs,
                                     link_with          : libnvcuvidSha)

