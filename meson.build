# ILSVRC
project('ILSVRC', 'c', 'cpp',
        version:                 '0.0.0',
        meson_version:           '>=0.38.1',
        default_options:         [])


# Library search paths
cudaLibs = ['/usr/local/cuda/lib64']


# Dependencies
cpp               = meson.get_compiler('cpp')
m                 = cpp.find_library('m',                 required : true)
#   OpenMP
omp               = cpp.find_library('gomp',              required : true)
#   HDF5
hdf5              = cpp.find_library('hdf5',              required : true)
#   ZMQ
zmq               = cpp.find_library('zmq',               required : true)
#   FFmpeg
avutil            = cpp.find_library('avutil',            required : true)
avcodec           = cpp.find_library('avcodec',           required : true)
#   CUDA
cuda              = cpp.find_library('cuda',              required : true, dirs : cudaLibs)
cudart            = cpp.find_library('cudart',            required : true, dirs : cudaLibs)
#   NVDEC
nvcuvid           = cpp.find_library('nvcuvid',           required : true, dirs : cudaLibs)
#   OpenCV
opencv_core       = cpp.find_library('opencv_core',       required : true)
opencv_imgcodecs  = cpp.find_library('opencv_imgcodecs',  required : true)
opencv_imgproc    = cpp.find_library('opencv_imgproc',    required : true)
opencv_highgui    = cpp.find_library('opencv_highgui',    required : true)

#   Which software needs what?
ilsvrcserver_deps = [m, zmq, hdf5,
                     cuda, cudart, nvcuvid,
                     opencv_core, opencv_imgcodecs, opencv_imgproc, opencv_highgui]
ilsvrcclient_deps = [m, zmq,
                     opencv_core, opencv_imgcodecs, opencv_imgproc, opencv_highgui]
ilsvrcgen_deps    = [m, hdf5, omp,
                     cuda, cudart, nvcuvid,
                     avcodec, avutil,
                     opencv_core, opencv_imgcodecs, opencv_imgproc, opencv_highgui]


# Includes
incs = include_directories('/usr/include/ffmpeg', '/usr/local/cuda/include')



# Executables
#   Server
ilsvrcserver_srcs = files('src/ilsvrcserver.cpp')
ilsvrcserver      = executable('ilsvrcserver', ilsvrcserver_srcs,
                               include_directories : incs,
                               dependencies        : ilsvrcserver_deps)


#   Benchmark Client
ilsvrcclient_srcs = files('src/ilsvrcclient.cpp')
ilsvrcclient      = executable('ilsvrcclient', ilsvrcclient_srcs,
                               include_directories : incs,
                               dependencies        : ilsvrcclient_deps)


#   Dataset Generator
ilsvrcgen_srcs      = files('src/ilsvrcgen.cpp')
ilsvrcgen_c_args    = ['-fopenmp']
ilsvrcgen_cpp_args  = ilsvrcgen_c_args
ilsvrcgen           = executable('ilsvrcgen',    ilsvrcgen_srcs,
                                 c_args              : ilsvrcgen_c_args,
                                 cpp_args            : ilsvrcgen_c_args,
                                 include_directories : incs,
                                 dependencies        : ilsvrcgen_deps)
