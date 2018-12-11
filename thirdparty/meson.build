#
# NVIDIA
#
# Detect CUDA Toolkit
#
CUDA_ARCH         = get_option('cuda_arch')
CUDA_HOME         = get_option('cuda_home')
CUDA_RUNTIME      = get_option('cuda_runtime') == 'static' ? 'cudart_static' : 'cudart'
currSrcDir        = meson.current_source_dir()
nvccPath          = join_paths(CUDA_HOME,  'bin',   'nvcc')
toolkitIncDir     = join_paths(CUDA_HOME,  'include')
toolkitLibDir     = join_paths(CUDA_HOME,  'lib64')
toolkitStubLibDir = join_paths(CUDA_HOME,  'lib64', 'stubs')
nvcuvidIncDir     = join_paths(currSrcDir, 'include')
stubLibDir        = join_paths(currSrcDir, 'lib',   host_machine.cpu_family())
driverLibDirs     = ['/usr/lib64/nvidia',            # Fedora alternate
                     '/usr/lib64/x86_64-linux-gnu',  # Debian multi-arch
                     toolkitStubLibDir]              # CUDA stub libraries
stubLibDirs       = driverLibDirs + [stubLibDir]
nvcuvidIncs       = include_directories('include')
cudaIncs          = include_directories(toolkitIncDir)
nvcuvidCDefs      = []
cudaGenCodeFlags  = []
nvcc              = find_program(nvccPath, 'nvcc', required: false)
libcuda           = cc.find_library('cuda',        required: false, dirs: driverLibDirs)
libcudart         = cc.find_library(CUDA_RUNTIME,  required: false, dirs: toolkitLibDir)
libcudadevrt      = cc.find_library('cudadevrt',   required: false, dirs: toolkitLibDir)
libnvcuvid        = cc.find_library('nvcuvid',     required: false, dirs: stubLibDirs)
nvcc              = nvcc.found()         ? nvcc         : disabler()
libcuda           = libcuda.found()      ? libcuda      : disabler()
libcudart         = libcudart.found()    ? libcudart    : disabler()
libcudadevrt      = libcudadevrt.found() ? libcudadevrt : disabler()
libnvcuvid        = libnvcuvid.found()   ? libnvcuvid   : disabler()



#
# NVDECODE/Video Codec SDK feature detection
#
if libnvcuvid.found()
  if   cc.has_function('cuvidReconfigureDecoder', dependencies: libnvcuvid)
    message('Found Video Codec SDK >= 8.2')
    message('Info: Video Codec SDK 8.2 requires NVIDIA Driver >= 396.24 and mandatory CUDA Toolkit >= 8.0')
    nvcuvidCDefs += ['-DNVDECODE=8002']
  elif cc.has_function('cuvidGetDecoderCaps',     dependencies: libnvcuvid)
    message('Found Video Codec SDK == 8.0 or 8.1')
    message('Info: Video Codec SDK 8.1 requires NVIDIA Driver >= 390.25 and mandatory CUDA Toolkit >= 8.0')
    message('Info: Video Codec SDK 8.0 requires NVIDIA Driver >= 378.13 and optional  CUDA Toolkit >= 8.0')
    nvcuvidCDefs += ['-DNVDECODE=8000']
  else
    message('Found Video Codec SDK <  8.0')
    message('Info: Video Codec SDK 7.1 requires NVIDIA Driver >= 375.20 and optional CUDA Toolkit >= 7.5')
    message('Info: Video Codec SDK 7.0 requires NVIDIA Driver >= 367.35 and optional CUDA Toolkit >= 7.5')
    nvcuvidCDefs += ['-DNVDECODE=7000']
  endif
else
  warning('No Video Codec SDK found.')
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
    cudaGenCodeFlags += detectGPU.stdout().split()
  endif
  cudaGenCodeFlags += ['-cudart', get_option('cuda_runtime')]
  if(get_option('buildtype') == 'debugoptimized' or
     get_option('buildtype') == 'debug')
    cudaGenCodeFlags += ['-g', '-lineinfo']
  else
    cudaGenCodeFlags += ['-O3']
  endif
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

