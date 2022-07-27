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
CUDA_VERSION        = 'unknown'
CUDA_ARCH           = get_option('nvidia_arch')
CUDA_HOME           = get_option('nvidia_home').split(':')[0]
CUDA_RUNTIME        = get_option('nvidia_runtime') == 'static' ? 'cudart_static' : 'cudart'
NVIDIA_DRIVER       = get_option('nvidia_driver').to_lower()
NVDEC_VERSION       = get_option('nvdec_version').to_lower()
nvcc_path           = CUDA_HOME  / 'bin' / 'nvcc'
toolkit_incdir      = CUDA_HOME  / 'include'
toolkit_libdir      = CUDA_HOME  / 'lib64'
toolkit_stub_libdir = CUDA_HOME  / 'lib64' / 'stubs'
cuda_incs           = []
nvcc_gencode_flags  = []
nvcc_cudart_flags   = []
nvcc_opt_flags      = []
nvcc_ccbin_flag     = []
nvcc                = find_program(nvcc_path, 'nvcc', required: false, disabler: true)
if nvcc.found()
  NVCC_RUN = run_command(nvcc, ['-V'], check: false)
  if NVCC_RUN.returncode() == 0
    CUDA_VERSION = NVCC_RUN.stdout().split('V')[-1].strip()
    #
    # Do this here to avoid configure failure when CUDA Toolkit is absent.
    #
    cuda_incs += [include_directories(toolkit_incdir)]
  else
    nvcc = disabler()
  endif
endif


#
# The driver version determines the NVIDIA Video Codec SDK version available.
# Note down some useful constants.
#
NVIDIA_DRIVER_11_1 = IS_HOST_WINDOWS ? '471.41' : '470.57.02'
NVIDIA_DRIVER_11_0 = IS_HOST_WINDOWS ? '456.71' : '455.28'
NVIDIA_DRIVER_10_0 = IS_HOST_WINDOWS ? '445.87' : '450.51'
NVIDIA_DRIVER_9_1  = IS_HOST_WINDOWS ? '436.15' : '435.21'
NVIDIA_DRIVER_9_0  = IS_HOST_WINDOWS ? '418.81' : '418.30'
NVIDIA_DRIVER_8_2  = IS_HOST_WINDOWS ? '397.93' : '396.24'
NVIDIA_DRIVER_8_1  = IS_HOST_WINDOWS ? '390.77' : '390.25'
NVIDIA_DRIVER_8_0  = IS_HOST_WINDOWS ? '378.66' : '378.13'


#
# At this point, we've either found and identified the CUDA Toolkit and version,
# or the CUDA Toolkit is missing. Either way, decide which NVIDIA driver version
# we will build for by progressively bracketing it down.
#
NVIDIA_DRIVER_MAX = IS_HOST_WINDOWS ? '516.59' : '515.57'
NVIDIA_DRIVER_MIN = IS_HOST_WINDOWS ? '378.66' : '378.13'
if CUDA_VERSION != 'unknown'
  NVIDIA_DRIVER_MIN = CUDA.min_driver_version(CUDA_VERSION)
endif


if   NVIDIA_DRIVER == 'min' or NVIDIA_DRIVER == 'old'
  NVIDIA_DRIVER     = NVIDIA_DRIVER_MIN
  NVIDIA_DRIVER_MAX = NVIDIA_DRIVER_MIN
  message('Selected NVIDIA Driver: '+NVIDIA_DRIVER+', minimum')
elif NVIDIA_DRIVER == 'max' or NVIDIA_DRIVER == 'new'
  NVIDIA_DRIVER     = NVIDIA_DRIVER_MAX
  NVIDIA_DRIVER_MIN = NVIDIA_DRIVER_MAX
  message('Selected NVIDIA Driver: '+NVIDIA_DRIVER+', maximum')
elif NVIDIA_DRIVER != 'auto'
  #
  # Probably a manually-specified driver version number, but verify anyways
  # ensure that the version number is made of dotted integers.
  #
  foreach d : NVIDIA_DRIVER.strip().split('.')
    d.to_int()
  endforeach
  NVIDIA_DRIVER_MIN = NVIDIA_DRIVER
  NVIDIA_DRIVER_MAX = NVIDIA_DRIVER
  message('Selected NVIDIA Driver: '+NVIDIA_DRIVER+', manually')
else
  #
  # The interesting case: 'auto' version detection of the driver.
  # 
  # We deploy a number of strategies to infer the driver version, with more
  # options available for native builds.
  #
  if IS_NATIVE
    #
    # First, most machine-portable attempt:
    #     nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits
    # May fail because of NVIDIA kernel driver not being loaded, or mismatching.
    #
    if NVIDIA_DRIVER == 'auto'
      nvidiasmi = find_program('nvidia-smi', required: false, disabler: true)
      if nvidiasmi.found()
        NVIDIA_SMI_RUN = run_command(nvidiasmi, ['--query-gpu=driver_version',
                                                 '--format=csv,noheader,nounits'], check: false)
        if NVIDIA_SMI_RUN.returncode() == 0
          NVIDIA_DRIVER     = NVIDIA_SMI_RUN.stdout().strip()
          NVIDIA_DRIVER_MIN = NVIDIA_DRIVER
          NVIDIA_DRIVER_MAX = NVIDIA_DRIVER
          message('Detected NVIDIA Driver: '+NVIDIA_DRIVER+', nvidia-smi')
        elif NVIDIA_SMI_RUN.stdout().to_lower().contains('mismatch')
          warning(NVIDIA_SMI_RUN.stdout().strip()) # Important API mismatch warning.
        endif
      endif
    endif
    #
    # Second, Linux-specific attempt:
    #     cat /sys/module/nvidia/version
    # May fail because of NVIDIA kernel driver not being loaded, but cannot
    # fail because of a userland-kernel mismatch.
    #
    if NVIDIA_DRIVER == 'auto' and IS_HOST_LINUX and fs.exists('/sys/module/nvidia/version')
      NVIDIA_DRIVER     = fs.read('/sys/module/nvidia/version').strip()
      NVIDIA_DRIVER_MIN = NVIDIA_DRIVER
      NVIDIA_DRIVER_MAX = NVIDIA_DRIVER
      message('Detected NVIDIA Driver: '+NVIDIA_DRIVER+', /sys/module/nvidia/version')
    endif
    #
    # Third, Linux-specific attempt:
    #     modinfo -F version nvidia
    # Can only fail if kernel module is not installed in the first place.
    # This can happen if, for instance, the build machine has the CUDA Toolkit
    # installed but not the NVIDIA driver, and the host machine has the NVIDIA
    # driver installed but not the CUDA Toolkit, and can happen even for native
    # builds (build and host machines of identical architecture)
    #
    if NVIDIA_DRIVER == 'auto' and IS_HOST_LINUX
      modinfo = find_program('modinfo', dirs: ['/sbin', '/usr/sbin'],
                             required: false, disabler: true)
      if modinfo.found()
        MODINFO_RUN = run_command(modinfo, ['-F', 'version', 'nvidia'], check: false)
        if MODINFO_RUN.returncode() == 0
          NVIDIA_DRIVER     = MODINFO_RUN.stdout().strip()
          NVIDIA_DRIVER_MIN = NVIDIA_DRIVER
          NVIDIA_DRIVER_MAX = NVIDIA_DRIVER
          message('Detected NVIDIA Driver: '+NVIDIA_DRIVER+', modinfo')
        endif
      endif
    endif
  endif
  #
  # Fourth attempt: Attempt to compile & link test code. This manual NVDECODE/
  # Video Codec SDK feature detection is necessary because of the total lack of
  # version-query functions or #defines in libnvcuvid's API.
  # 
  # On Linux, this will fail if libnvcuvid.so.1 cannot be found.
  # On Windows, this will fail if nvcuvid.dll cannot be found.
  #
  if NVIDIA_DRIVER == 'auto'
    testCodePrefix = '''
      #ifdef _WIN32
      #define CUDAAPI __stdcall
      #else
      #define CUDAAPI
      #endif
    '''
    testCodeSDK82 = '''
      extern int CUDAAPI cuvidReconfigureDecoder(void*, void*);
      volatile void* p=(void*)cuvidReconfigureDecoder;
      int main(){return !p;}
    '''
    testCodeSDK80 = '''
      extern int CUDAAPI cuvidGetDecoderCaps(void*);
      volatile void* p=(void*)cuvidReconfigureDecoder;
      int main(){return !p;}
    '''
    testCodeArgs = []
    if CC_SYNTAX == 'gcc' and not IS_HOST_WINDOWS
      testCodeArgs += ['-l:libnvcuvid.so.1']
    else
      testCodeArgs += ['-lnvcuvid']
    endif
    if   cc.links(testCodePrefix+testCodeSDK82, args: testCodeArgs)
      NVIDIA_DRIVER_MIN = (NVIDIA_DRIVER_MIN.version_compare('<'+NVIDIA_DRIVER_8_2) ?
                           NVIDIA_DRIVER_8_2 : NVIDIA_DRIVER_MIN)
    elif cc.links(testCodePrefix+testCodeSDK80, args: testCodeArgs)
      NVIDIA_DRIVER_MIN = (NVIDIA_DRIVER_MIN.version_compare('<'+NVIDIA_DRIVER_8_0) ?
                           NVIDIA_DRIVER_8_0 : NVIDIA_DRIVER_MIN)
      NVIDIA_DRIVER_MAX = (NVIDIA_DRIVER_MAX.version_compare('>396.23') ?
                           '396.23'          : NVIDIA_DRIVER_MAX)
    endif
  endif
  #
  # Before-last attempt:
  #     If NVIDIA_DRIVER_MIN == NVIDIA_DRIVER_MAX, set unconditionally
  #
  if NVIDIA_DRIVER == 'auto' and NVIDIA_DRIVER_MIN == NVIDIA_DRIVER_MAX
    NVIDIA_DRIVER = NVIDIA_DRIVER_MIN
    message('Inferred NVIDIA Driver: '+NVIDIA_DRIVER+', bracketing')
  endif
  #
  # Last attempt: If we are still undecided about the driver version, simply
  # unconditionally guess that it is the oldest possible driver.
  #
  if NVIDIA_DRIVER == 'auto'
    NVIDIA_DRIVER     = NVIDIA_DRIVER_MIN
    NVIDIA_DRIVER_MAX = NVIDIA_DRIVER_MIN
    message('Guessed  NVIDIA Driver: '+NVIDIA_DRIVER+', lower-bounded by autodetect')
  endif
endif


#
# Emit warning if CUDA Toolkit version is incompatible with NVIDIA driver version.
#
if CUDA_VERSION != 'unknown'
  NVIDIA_DRIVER_CUDA_MIN = CUDA.min_driver_version(CUDA_VERSION)
  message('Found    CUDA Toolkit:  '+CUDA_VERSION+', supported NVIDIA Driver >= '+NVIDIA_DRIVER_CUDA_MIN)
  if NVIDIA_DRIVER.version_compare('<'+NVIDIA_DRIVER_CUDA_MIN)
    warning('CUDA Toolkit '+CUDA_VERSION+' requires driver >= '+NVIDIA_DRIVER_CUDA_MIN+
            ', but found '+NVIDIA_DRIVER+'!')
  endif
endif


#
# Find CUDA libraries, but build linvcuvid stub ourselves.
#
libcuda      = cc.find_library('cuda',       required: false, disabler: true, dirs: toolkit_stub_libdir)
libcudart    = cc.find_library(CUDA_RUNTIME, required: false, disabler: true, dirs: toolkit_libdir)
libcudadevrt = cc.find_library('cudadevrt',  required: false, disabler: true, dirs: toolkit_libdir)
subdir('libnvcuvid')
cuda_incs   += nvcuvid_incs



#
# GPU architecture detection and compile flags selection.
#
if nvcc.found()
  gpuarchs = []
  if CUDA_ARCH == 'Auto'
    detectGPUargs   = ['-w', '-cudart', 'static', detectgpus_c, '--run']
    detectGPU       = run_command(nvcc, detectGPUargs, check: false)
    detectGPUclean  = run_command(nvcc, detectGPUargs+['--clean-targets'], check: false)
    detectGPUstdout = detectGPU.stdout().strip()
    detectGPUstderr = detectGPU.stderr().strip()
    if detectGPU.returncode() == 0
      gpuarchs += detectGPUstdout.split()
      if detectGPUstderr != ''
        message(detectGPUstderr)
      endif
    else
      warning('GPU detection failed:')
      if detectGPUstdout != ''
        warning(detectGPUstdout)
      endif
      if detectGPUstderr != ''
        warning(detectGPUstderr)
      endif
    endif
  endif
  nvcc_gencode_readable = CUDA.nvcc_arch_readable(CUDA_VERSION, CUDA_ARCH, detected: gpuarchs)
  nvcc_gencode_flags   += CUDA.nvcc_arch_flags   (CUDA_VERSION, CUDA_ARCH, detected: gpuarchs)
  nvcc_cudart_flags    += ['-cudart', get_option('nvidia_runtime')]
  if get_option('debug') == true
    nvcc_opt_flags += ['-g', '-lineinfo']
  endif
  if   '0123'.contains(get_option('optimization'))
    nvcc_opt_flags += ['-O'+get_option('optimization')]
  elif 'g'.contains(get_option('optimization'))
    nvcc_opt_flags += ['-O0']
  elif 's'.contains(get_option('optimization'))
    nvcc_opt_flags += ['-O2']
  endif
  if((get_option('b_ndebug') == 'if-release' and '23s'.contains(get_option('optimization'))) or
     (get_option('b_ndebug') == 'true'))
    nvcc_opt_flags += ['-DNDEBUG']
  endif
  if cxx.cmd_array().length() == 1
    nvcc_ccbin_flag += ['-ccbin'] + cxx.cmd_array()
  else
    warning('\nNVCC\'s -ccbin argument does not provide any way of supplying '+
            'multiple compiler arguments, multiple search directories or '+
            'compiler wrappers (e.g. ccache).\n'+
            'Ignoring [\''+'\', \''.join(cxx.cmd_array())+'\'] and using NVCC compiler defaults.')
  endif
  
  if   CUDA_ARCH != 'Auto'
    message('Building for selected GPUs [@0@]'.format(' '.join(nvcc_gencode_readable)))
  elif gpuarchs != []
    message('Building for detected GPUs [@0@]'.format(' '.join(nvcc_gencode_readable)))
  else
    message('Building for common GPUs [@0@]'  .format(' '.join(nvcc_gencode_readable)))
  endif
endif


#
# Compile generator and linker args.
#
nvcc_compiler_generator = generator(nvcc,
    arguments: nvcc_ccbin_flag + nvcc_gencode_flags + nvcc_cudart_flags + nvcc_opt_flags + nvcuvid_iflags +
               ['-x', 'cu', '@EXTRA_ARGS@', '@INPUT@', '-c', '-o', '@OUTPUT@'],
    output:    ['@BASENAME@.o'],
)
nvcc_link_args = ['-dlink'] + nvcc_ccbin_flag + nvcc_cudart_flags + nvcc_gencode_flags


#
# Construct convenient build-internal dependencies for use by remaining build
# artifacts. Reduces the need to carry around include directories and custom
# compile flags.
#
libcuda       = declare_dependency(include_directories: cuda_incs,
                                   dependencies       : libcuda)
libcudart     = declare_dependency(include_directories: cuda_incs,
                                   dependencies       : [libcudart, librt])
libcudadevrt  = declare_dependency(include_directories: cuda_incs,
                                   dependencies       : libcudart)
libnvcuvid    = declare_dependency(include_directories: nvcuvid_incs,
                                   dependencies       : libnvcuvid)

