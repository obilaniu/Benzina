#
# NVIDIA
#
# Detect CUDA Toolkit and NVIDIA driver. There are several possibilities:
#
# - This is a build in "Min"/"Old" driver mode. We build assuming the very
#   oldest NVIDIA driver version (and thus, Video Codec SDK) the CUDA Toolkit
#   supports.
#
# - This is a build in "Max"/"New" driver mode. We build assuming the very
#   newest NVIDIA driver version (and thus, Video Codec SDK) we support.
#
# - This is a build with a specified driver version number. We build assuming
#   the specific version indicated (and thus the implied Video Codec SDK).
#
# - This is a build in "Auto" mode. We have no information about the NVIDIA
#   driver version (and thus, Video Codec SDK) except that which we may infer
#   from local and remote sources. We can narrow down these bounds using the
#   CUDA Toolkit version, but may still need to perform link/run-tests to
#   verify the presence of certain symbols. There are two subcases:
#
#   - This is a cross-build in "Auto" driver mode. We are unable to test-run
#     any code, we can only test-link it to check the availability of specific
#     APIs. To do this we use the cross system libraries, and do not use the
#     stubs.
#
#   - This is a native build, in "Auto" driver mode. We do not know whether the
#     build machine is in fact the host machine, as the CUDA Toolkit need not
#     be installed on the host machine and the NVIDIA driver need not be
#     installed on the build machine. We will assume that they are the same,
#     and if that fails we fall back to test-link/test-run code snippets. To
#     do that we try to use the stubs, then the system libraries. If programs
#     link and run with specific APIs, they are assumed usable.
#
CUDA_ARCH         = get_option('nvidia_arch')
CUDA_HOME         = get_option('nvidia_home').split(':')[0]
CUDA_RUNTIME      = get_option('nvidia_runtime') == 'static' ? 'cudart_static' : 'cudart'
NVIDIA_DRIVER     = get_option('nvidia_driver').to_lower()
currSrcDir        = meson.current_source_dir()
nvccPath          = CUDA_HOME  / 'bin' / 'nvcc'
toolkitIncDir     = CUDA_HOME  / 'include'
toolkitLibDir     = CUDA_HOME  / 'lib64'
toolkitStubLibDir = CUDA_HOME  / 'lib64' / 'stubs'
stubLibDir        = currSrcDir / 'lib' / host_machine.cpu_family()
nvcuvidIncDir     = currSrcDir / 'include'
nvcuvidIncs       = include_directories('include', toolkitIncDir)
nvcuvidIFlags     = ['-I'+nvcuvidIncDir, '-I'+toolkitIncDir]
nvcuvidCDefs      = []
cudaIncs          = nvcuvidIncs
nvccGenCodeFlags  = []
nvccCudaRtFlags   = []
nvccOptFlags      = []
nvcc              = find_program(nvccPath, 'nvcc', required: false, disabler: true)
nvidiasmi         = find_program('nvidia-smi',     required: false, disabler: true)
NVCC_VERSION      = 'unknown'
if nvcc.found()
  NVCC_RUN = run_command(nvcc, ['-V'])
  if NVCC_RUN.returncode() == 0
    NVCC_VERSION = NVCC_RUN.stdout().split('V')[-1].strip()
  else
    nvcc = disabler()
  endif
endif
if   NVCC_VERSION == 'unknown'
  NVIDIA_DRIVER_MIN = IS_HOST_WINDOWS ? '378.66' : '378.13'
elif NVCC_VERSION.version_compare('>=10.1.105')
  NVIDIA_DRIVER_MIN = IS_HOST_WINDOWS ? '418.96' : '418.39'
elif NVCC_VERSION.version_compare('>=10.0.130')
  NVIDIA_DRIVER_MIN = IS_HOST_WINDOWS ? '411.31' : '410.48'
elif NVCC_VERSION.version_compare('>=9.2.148')
  NVIDIA_DRIVER_MIN = IS_HOST_WINDOWS ? '398.26' : '396.37'
elif NVCC_VERSION.version_compare('>=9.2.88')
  NVIDIA_DRIVER_MIN = IS_HOST_WINDOWS ? '397.44' : '396.26'
elif NVCC_VERSION.version_compare('>=9.1.85')
  NVIDIA_DRIVER_MIN = IS_HOST_WINDOWS ? '391.29' : '390.46'
elif NVCC_VERSION.version_compare('>=9.0.76')
  NVIDIA_DRIVER_MIN = IS_HOST_WINDOWS ? '385.54' : '384.81'
elif NVCC_VERSION.version_compare('>=8.0.61')
  NVIDIA_DRIVER_MIN = IS_HOST_WINDOWS ? '376.51' : '375.26'
elif NVCC_VERSION.version_compare('>=8.0.44')
  NVIDIA_DRIVER_MIN = IS_HOST_WINDOWS ? '369.30' : '367.48'
elif NVCC_VERSION.version_compare('>=7.5.16')
  NVIDIA_DRIVER_MIN = IS_HOST_WINDOWS ? '353.66' : '352.31'
elif NVCC_VERSION.version_compare('>=7.0.28')
  NVIDIA_DRIVER_MIN = IS_HOST_WINDOWS ? '347.62' : '346.46'
else
  NVIDIA_DRIVER_MIN = IS_HOST_WINDOWS ? '378.66' : '378.13'
endif
NVIDIA_DRIVER_MAX = IS_HOST_WINDOWS ? '419.17' : '418.43'
NVIDIA_DRIVER_9_0 = IS_HOST_WINDOWS ? '418.81' : '418.30'
NVIDIA_DRIVER_8_2 = IS_HOST_WINDOWS ? '397.93' : '396.24'
NVIDIA_DRIVER_8_1 = IS_HOST_WINDOWS ? '390.77' : '390.25'
NVIDIA_DRIVER_8_0 = IS_HOST_WINDOWS ? '378.66' : '378.13'
NVIDIA_DRIVER     = NVIDIA_DRIVER == 'old' ? NVIDIA_DRIVER_MIN : NVIDIA_DRIVER
NVIDIA_DRIVER     = NVIDIA_DRIVER == 'min' ? NVIDIA_DRIVER_MIN : NVIDIA_DRIVER
NVIDIA_DRIVER     = NVIDIA_DRIVER == 'new' ? NVIDIA_DRIVER_MAX : NVIDIA_DRIVER
NVIDIA_DRIVER     = NVIDIA_DRIVER == 'max' ? NVIDIA_DRIVER_MAX : NVIDIA_DRIVER
driverLibDirs     = NVIDIA_DRIVER == 'auto' and IS_CROSS ? [] : [stubLibDir, toolkitStubLibDir]
libcuda           = cc.find_library('cuda',        required: false, disabler: true, dirs: driverLibDirs)
libcudart         = cc.find_library(CUDA_RUNTIME,  required: false, disabler: true, dirs: toolkitLibDir)
libcudadevrt      = cc.find_library('cudadevrt',   required: false, disabler: true, dirs: toolkitLibDir)
libnvcuvid        = cc.find_library('nvcuvid',     required: false, disabler: true, dirs: driverLibDirs)
if NVCC_VERSION != 'unknown'
  message('Found    CUDA Toolkit:  '+NVCC_VERSION+', supported NVIDIA Driver >= '+NVIDIA_DRIVER_MIN)
endif
if NVIDIA_DRIVER != 'auto'
  message('Selected NVIDIA Driver: '+NVIDIA_DRIVER+', manually')
endif


#
# NVIDIA driver version autodetect, if required.
#
# First attempt is with nvidia-smi on native builds.
if NVIDIA_DRIVER == 'auto' and IS_NATIVE
  # This is a native build (build architecture matches host architecture), and
  # the build machine could even be the very same one as the host machine. We
  # query this machine's NVIDIA driver version if possible.
  if nvidiasmi.found()
    NVIDIA_SMI_RUN = run_command(nvidiasmi, ['--query-gpu=driver_version',
                                             '--format=csv,noheader,nounits'])
    if NVIDIA_SMI_RUN.returncode() == 0
      NVIDIA_DRIVER = NVIDIA_SMI_RUN.stdout().strip()
      message('Detected NVIDIA Driver: '+NVIDIA_DRIVER+', nvidia-smi')
    elif NVIDIA_SMI_RUN.stdout().to_lower().contains('mismatch')
      warning(NVIDIA_SMI_RUN.stdout().strip()) # Important API mismatch warning.
    endif
  endif
endif
#
# Second attempt involves manual NVDECODE/Video Codec SDK feature detection,
# due to the total lack of version-query functions in libnvcuvid's API.
#
if NVIDIA_DRIVER == 'auto' and libnvcuvid.found()
  codeCheck = '#include "nvcuvid.h"\r\nvolatile void* p=(void*)@0@;int main(){return !p;}'
  codeSDK82 = codeCheck.format('cuvidReconfigureDecoder')
  codeSDK80 = codeCheck.format('cuvidGetDecoderCaps')
  codeDeps  = libnvcuvid
  codeIncs  = [nvcuvidIncs, cudaIncs]
  testSDK82 = cc.run(codeSDK82, dependencies: codeDeps, include_directories: codeIncs)
  testSDK80 = cc.run(codeSDK80, dependencies: codeDeps, include_directories: codeIncs)
  if   testSDK82.compiled() and (IS_CROSS or testSDK82.returncode() == 0)
    NVIDIA_DRIVER_MIN = (NVIDIA_DRIVER_MIN.version_compare('<'+NVIDIA_DRIVER_8_2) ?
                         NVIDIA_DRIVER_8_2 : NVIDIA_DRIVER_MIN)
    NVIDIA_DRIVER_MAX =  NVIDIA_DRIVER_MAX
  elif testSDK80.compiled() and (IS_CROSS or testSDK80.returncode() == 0)
    NVIDIA_DRIVER_MIN = (NVIDIA_DRIVER_MIN.version_compare('<'+NVIDIA_DRIVER_8_0) ?
                         NVIDIA_DRIVER_8_0 : NVIDIA_DRIVER_MIN)
    NVIDIA_DRIVER_MAX =  '396.23'
  endif
endif
#
# If NVIDIA driver version is still unknown, default it to the minimum (oldest)
# possible version.
#
if NVIDIA_DRIVER == 'auto'
  NVIDIA_DRIVER = NVIDIA_DRIVER_MIN
  message('Guessed  NVIDIA Driver: '+NVIDIA_DRIVER+', lower-bounded')
endif


#
# Emit warning if CUDA Toolkit version is incompatible with NVIDIA driver version.
#
if NVCC_VERSION != 'unknown' and not NVIDIA_DRIVER.version_compare('>='+NVIDIA_DRIVER_MIN)
  warning('CUDA Toolkit '+NVCC_VERSION+' requires driver >= '+NVIDIA_DRIVER_MIN+
          ', but found '+NVIDIA_DRIVER+'!')
endif


#
# Infer Video Codec SDK version from NVIDIA driver version.
#
if   not libnvcuvid.found()
  warning('No Video Codec SDK found.')
elif NVIDIA_DRIVER.version_compare('>='+NVIDIA_DRIVER_9_0)
  message('Found Video Codec SDK >= 9.0, requires NVIDIA Driver >= '+NVIDIA_DRIVER_9_0+' and mandatory CUDA Toolkit >= 10.0')
  if NVCC_VERSION != 'unknown' and NVCC_VERSION.version_compare('<10.0')
    warning('Warning: CUDA Toolkit @0@ too old!'.format(NVCC_VERSION))
  endif
  nvcuvidCDefs += ['-DNVDECODE=9000']
elif NVIDIA_DRIVER.version_compare('>='+NVIDIA_DRIVER_8_2)
  message('Found Video Codec SDK >= 8.2, requires NVIDIA Driver >= '+NVIDIA_DRIVER_8_2+' and mandatory CUDA Toolkit >= 8.0')
  if NVCC_VERSION != 'unknown' and NVCC_VERSION.version_compare('<8.0')
    warning('Warning: CUDA Toolkit @0@ too old!'.format(NVCC_VERSION))
  endif
  nvcuvidCDefs += ['-DNVDECODE=8002']
elif NVIDIA_DRIVER.version_compare('>='+NVIDIA_DRIVER_8_1)
  message('Found Video Codec SDK == 8.1, requires NVIDIA Driver >= '+NVIDIA_DRIVER_8_1+' and mandatory CUDA Toolkit >= 8.0')
  if NVCC_VERSION != 'unknown' and NVCC_VERSION.version_compare('<8.0')
    warning('Warning: CUDA Toolkit @0@ too old!'.format(NVCC_VERSION))
  endif
  nvcuvidCDefs += ['-DNVDECODE=8001']
elif NVIDIA_DRIVER.version_compare('>='+NVIDIA_DRIVER_8_0)
  message('Found Video Codec SDK == 8.0, requires NVIDIA Driver >= '+NVIDIA_DRIVER_8_0+' and optional  CUDA Toolkit >= 8.0')
  if NVCC_VERSION != 'unknown' and NVCC_VERSION.version_compare('<8.0')
    warning('Warning: CUDA Toolkit @0@ too old!'.format(NVCC_VERSION))
  endif
  nvcuvidCDefs += ['-DNVDECODE=8000']
else
  warning('Found Video Codec SDK < 8.0, too old! Disabling!')
  libnvcuvid = disabler()
endif



#
# GPU presence detection and architecture selection
#
if nvcc.found()
  detectGPU = run_command(python, [detectGPUpy, nvcc, CUDA_ARCH])
  if detectGPU.returncode() != 0
    warning('GPU detection failed:')
    warning(detectGPU.stdout())
    warning(detectGPU.stderr())
  else
    message(detectGPU.stderr())
    nvccGenCodeFlags += detectGPU.stdout().split()
  endif
  nvccCudaRtFlags += ['-cudart', get_option('nvidia_runtime')]
  if(get_option('buildtype') == 'debugoptimized' or
     get_option('buildtype') == 'debug')
    nvccOptFlags += ['-g', '-lineinfo']
  else
    nvccOptFlags += ['-O3']
  endif
  nvccCcBinFlag = ['-ccbin', cxx.cmd_array()[0]]
endif


#
# Compile generator and linker args.
#
nvccCompGen   = generator(nvcc,
    arguments: nvccCcBinFlag + nvccGenCodeFlags + nvccCudaRtFlags + nvccOptFlags + nvcuvidIFlags +
               ['-x', 'cu', '@EXTRA_ARGS@', '@INPUT@', '-c', '-o', '@OUTPUT@'],
    output:    ['@BASENAME@.o'],
)
nvccLinkArgs  = ['-dlink'] + nvccCcBinFlag + nvccGenCodeFlags


#
# Construct convenient build-internal dependencies for use by remaining build
# artifacts. Reduces the need to carry around include directories and custom
# compile flags.
#
libcuda       = declare_dependency(include_directories: cudaIncs,
                                   dependencies       : libcuda)
libcudart     = declare_dependency(include_directories: cudaIncs,
                                   dependencies       : [libcudart, librt])
libcudadevrt  = declare_dependency(include_directories: cudaIncs,
                                   dependencies       : libcudart)
libnvcuvid    = declare_dependency(include_directories: nvcuvidIncs,
                                   compile_args       : nvcuvidCDefs,
                                   dependencies       : libnvcuvid)

