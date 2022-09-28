#
# Include Directory of libnvcuvid.
#
# This directory holds the unmodified, authentic Video Codec SDK headers, namely
#    - cuviddec.h
#    - nvcuvid.h
# These headers directly include cuda.h, a dynamic-link version of which called
# dynlink_cuda.h exists in this directory. This include/dynlink_cuda.h file is
# renamed, using configure_file(), to src/cuda.h in the *build-tree* directory
# The cuda.h file *MUST NOT* end up in either the build- or source-tree
# include/ directory, as this directory will be made available for inclusion by
# all other targets, and if a file called cuda.h is found in it, it will take
# precedence over the CUDA Toolkit's own, authentic cuda.h.
#

nvcuvid_incdir  = meson.current_source_dir()
nvcuvid_incs    = [include_directories('.')]
nvcuvid_iflags  = ['-I'+nvcuvid_incdir]
nvcuvid_cdefs   = []
dynlink_cuda_h  = files('dynlink_cuda.h')


#
# Infer Video Codec SDK version from NVIDIA driver version.
#
if NVDEC_VERSION == 'auto'
  if   NVIDIA_DRIVER.version_compare('>='+NVIDIA_DRIVER_11_1)
    NVDEC_VERSION = '11.1'
  elif NVIDIA_DRIVER.version_compare('>='+NVIDIA_DRIVER_11_0)
    NVDEC_VERSION = '11.0'
  elif NVIDIA_DRIVER.version_compare('>='+NVIDIA_DRIVER_10_0)
    NVDEC_VERSION = '10.0'
  elif NVIDIA_DRIVER.version_compare('>='+NVIDIA_DRIVER_9_1)
    NVDEC_VERSION = '9.1'
  elif NVIDIA_DRIVER.version_compare('>='+NVIDIA_DRIVER_9_0)
    NVDEC_VERSION = '9.0'
  elif NVIDIA_DRIVER.version_compare('>='+NVIDIA_DRIVER_8_2)
    NVDEC_VERSION = '8.2'
  elif NVIDIA_DRIVER.version_compare('>='+NVIDIA_DRIVER_8_1)
    NVDEC_VERSION = '8.1'
  elif NVIDIA_DRIVER.version_compare('>='+NVIDIA_DRIVER_8_0)
    NVDEC_VERSION = '8.0'
  endif
endif


#
# Depending on the Video Codec SDK version, add different compile flags.
# Also print an announcement and possibly a warning, depending on the version.
#
if   NVDEC_VERSION == 'unknown'
  warning('Found Video Codec SDK < 8.0, too old!')
elif NVDEC_VERSION.version_compare('>=11.1')
  message('Found Video Codec SDK >= 11.1, requires NVIDIA Driver >= '+NVIDIA_DRIVER_11_1+' and mandatory CUDA Toolkit >= 11.0')
  if CUDA_VERSION != 'unknown' and CUDA_VERSION.version_compare('<11.0')
    warning('CUDA Toolkit @0@ too old!'.format(CUDA_VERSION))
  endif
  nvcuvid_cdefs += ['-DNVDECODE=11001']
elif NVDEC_VERSION.version_compare('>=11.0')
  message('Found Video Codec SDK >= 11.0, requires NVIDIA Driver >= '+NVIDIA_DRIVER_11_0+' and mandatory CUDA Toolkit >= 11.0')
  if CUDA_VERSION != 'unknown' and CUDA_VERSION.version_compare('<11.0')
    warning('CUDA Toolkit @0@ too old!'.format(CUDA_VERSION))
  endif
  nvcuvid_cdefs += ['-DNVDECODE=11000']
elif NVDEC_VERSION.version_compare('>=10.0')
  message('Found Video Codec SDK >= 10.0, requires NVIDIA Driver >= '+NVIDIA_DRIVER_10_0+' and mandatory CUDA Toolkit >= 10.1')
  if CUDA_VERSION != 'unknown' and CUDA_VERSION.version_compare('<10.1')
    warning('CUDA Toolkit @0@ too old!'.format(CUDA_VERSION))
  endif
  nvcuvid_cdefs += ['-DNVDECODE=10000']
elif NVDEC_VERSION.version_compare('>=9.1')
  message('Found Video Codec SDK >= 9.1, requires NVIDIA Driver >= '+NVIDIA_DRIVER_9_1+' and mandatory CUDA Toolkit >= 10.0')
  if CUDA_VERSION != 'unknown' and CUDA_VERSION.version_compare('<10.0')
    warning('CUDA Toolkit @0@ too old!'.format(CUDA_VERSION))
  endif
  nvcuvid_cdefs += ['-DNVDECODE=9001']
elif NVDEC_VERSION.version_compare('>=9.0')
  message('Found Video Codec SDK >= 9.0, requires NVIDIA Driver >= '+NVIDIA_DRIVER_9_0+' and mandatory CUDA Toolkit >= 10.0')
  if CUDA_VERSION != 'unknown' and CUDA_VERSION.version_compare('<10.0')
    warning('CUDA Toolkit @0@ too old!'.format(CUDA_VERSION))
  endif
  nvcuvid_cdefs += ['-DNVDECODE=9000']
elif NVDEC_VERSION.version_compare('>=8.2')
  message('Found Video Codec SDK >= 8.2, requires NVIDIA Driver >= '+NVIDIA_DRIVER_8_2+' and mandatory CUDA Toolkit >= 8.0')
  if CUDA_VERSION != 'unknown' and CUDA_VERSION.version_compare('<8.0')
    warning('CUDA Toolkit @0@ too old!'.format(CUDA_VERSION))
  endif
  nvcuvid_cdefs += ['-DNVDECODE=8002']
elif NVDEC_VERSION.version_compare('>=8.1')
  message('Found Video Codec SDK == 8.1, requires NVIDIA Driver >= '+NVIDIA_DRIVER_8_1+' and mandatory CUDA Toolkit >= 8.0')
  if CUDA_VERSION != 'unknown' and CUDA_VERSION.version_compare('<8.0')
    warning('CUDA Toolkit @0@ too old!'.format(CUDA_VERSION))
  endif
  nvcuvid_cdefs += ['-DNVDECODE=8001']
elif NVDEC_VERSION.version_compare('>=8.0')
  message('Found Video Codec SDK == 8.0, requires NVIDIA Driver >= '+NVIDIA_DRIVER_8_0+' and optional  CUDA Toolkit >= 8.0')
  if CUDA_VERSION != 'unknown' and CUDA_VERSION.version_compare('<8.0')
    warning('CUDA Toolkit @0@ too old!'.format(CUDA_VERSION))
  endif
  nvcuvid_cdefs += ['-DNVDECODE=8000']
else
  warning('Found Video Codec SDK < 8.0, too old!')
endif

