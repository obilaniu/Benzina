#
# Source Directory of libnvcuvid *stub*.
#
# This directory holds a renamed dynlink_cuda.h -> cuda.h header for the sole
# purpose of building the stub. It is not complete, and it must therefore
# *NOT* be made available for inclusion by any target except this stub. This
# implies not adding this directory to any other search paths except the one
# for this stub.
#

nvcuvid_linkargs = []
nvcuvid_cargs    = ['-D__CUVID_INTERNAL']
nvcuvid_kwargs   = IS_HOST_WINDOWS ? {} : {'soversion': '1'}
cuda_h           = fs.copyfile(dynlink_cuda_h, 'cuda.h')


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
if CC_SYNTAX == 'gcc' and IS_HOST_LINUX
  nvcuvid_cargs    += ['-w',
                      '-fno-exceptions',
                      '-fno-unwind-tables',
                      '-fno-asynchronous-unwind-tables']
  nvcuvid_linkargs += ['-nostartfiles',
                      '-Wl,-s,--build-id=none,--hash-style=sysv,-z,norelro']
endif
libnvcuvid_sha  = shared_library('nvcuvid', [files('libnvcuvid.c'), cuda_h],
                                 include_directories: nvcuvid_incs + ['.'],
                                 c_args             : nvcuvid_cargs + nvcuvid_cdefs,
                                 link_args          : nvcuvid_linkargs,
                                 kwargs             : nvcuvid_kwargs,
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
  nvcuvid_iflags  += ['-I'+toolkit_incdir]
  nvcuvid_incs    += [include_directories(toolkit_incdir)]
endif
libnvcuvid      = declare_dependency(include_directories: nvcuvid_incs,
                                     compile_args       : nvcuvid_cdefs,
                                     link_with          : libnvcuvid_sha)

