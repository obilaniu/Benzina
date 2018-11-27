#
# NVIDIA
#
# CUDA Toolkit
#
CUDA_HOME    = get_option('cuda_home')
if get_option('cuda_runtime') == 'static'
  CUDA_RUNTIME = 'cudart_static'
else
  CUDA_RUNTIME = 'cudart'
endif
currSrcDir    = meson.current_source_dir()
nvccPath      = join_paths(CUDA_HOME,  'bin',    'nvcc')
toolkitLibDir = join_paths(CUDA_HOME,  'lib64')
toolkitIncDir = join_paths(CUDA_HOME,  'include')
nvcuvidIncDir = join_paths(currSrcDir, 'include')
driverLibDirs = [toolkitLibDir, '/usr/lib64', '/usr/lib64/nvidia']
stubLibDirs   = [join_paths(currSrcDir, 'lib', host_machine.cpu_family())] + driverLibDirs
cudaIncs      = include_directories('include', toolkitIncDir)
libcuda       = cc.find_library('cuda',          required: true, dirs: driverLibDirs)
libcudart     = cc.find_library(CUDA_RUNTIME,    required: true, dirs: toolkitLibDir)
libcudadevrt  = cc.find_library('cudadevrt',     required: true, dirs: toolkitLibDir)
libnvcuvid    = cc.find_library('nvcuvid',       required: true, dirs: stubLibDirs)
nvcc          = find_program(nvccPath, 'nvcc', required: true)


#
# NVDECODE/Video Codec SDK feature detection
#
if   cc.has_function('cuvidReconfigureDecoder', dependencies: [libnvcuvid])
  message('Found Video Codec SDK >= 8.2')
  message('Info: Video Codec SDK 8.2 requires NVIDIA Driver >= 396.24 and mandatory CUDA Toolkit >= 8.0')
  nvcuvidCDefs = ['-DNVDECODE=8002']
elif cc.has_function('cuvidGetDecoderCaps',     dependencies: [libnvcuvid])
  message('Found Video Codec SDK == 8.0 or 8.1')
  message('Info: Video Codec SDK 8.1 requires NVIDIA Driver >= 390.25 and mandatory CUDA Toolkit >= 8.0')
  message('Info: Video Codec SDK 8.0 requires NVIDIA Driver >= 378.13 and optional  CUDA Toolkit >= 8.0')
  nvcuvidCDefs = ['-DNVDECODE=8000']
else
  message('Found Video Codec SDK <  8.0')
  message('Info: Video Codec SDK 7.1 requires NVIDIA Driver >= 375.20 and optional CUDA Toolkit >= 7.5')
  message('Info: Video Codec SDK 7.0 requires NVIDIA Driver >= 367.35 and optional CUDA Toolkit >= 7.5')
  nvcuvidCDefs = ['-DNVDECODE=7000']
endif


