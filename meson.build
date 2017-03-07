# ILSVRC
project('ILSVRC', 'c', 'cpp',
        version:                 '0.0.0',
        meson_version:           '>=0.38.1',
        default_options:         [])


# Dependencies
cpp               = meson.get_compiler('cpp')
m                 = cpp.find_library('m',                 required : true)
hdf5              = cpp.find_library('hdf5',              required : true)
zmq               = cpp.find_library('zmq',               required : true)
turbojpeg         = cpp.find_library('turbojpeg',         required : true)
png               = cpp.find_library('png',               required : true)
opencv_core       = cpp.find_library('opencv_core',       required : true)
opencv_imgcodecs  = cpp.find_library('opencv_imgcodecs',  required : true)
opencv_imgproc    = cpp.find_library('opencv_imgproc',    required : true)
opencv_highgui    = cpp.find_library('opencv_highgui',    required : true)

deps = [m, hdf5, zmq, turbojpeg, png, opencv_core, opencv_imgcodecs,
        opencv_imgproc, opencv_highgui]


# Server
ilsvrcserver_srcs = files('src/ilsvrcserver.cpp')
ilsvrcserver      = executable('ilsvrcserver', ilsvrcserver_srcs,
                               dependencies : deps)


# Benchmark Client
ilsvrcclient_srcs = files('src/ilsvrcclient.cpp')
ilsvrcclient      = executable('ilsvrcclient', ilsvrcclient_srcs,
                               dependencies : deps)


# Dataset Generator
#ilsvrcgen_srcs    = files('src/ilsvrcgen.cpp')
#ilsvrcgen         = executable('ilsvrcgen', ilsvrcgen_srcs,
#                               dependencies : deps)
