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
avutil            = cpp.find_library('avutil',            required : true)
avcodec           = cpp.find_library('avcodec',           required : true)
opencv_core       = cpp.find_library('opencv_core',       required : true)
opencv_imgcodecs  = cpp.find_library('opencv_imgcodecs',  required : true)
opencv_imgproc    = cpp.find_library('opencv_imgproc',    required : true)
opencv_highgui    = cpp.find_library('opencv_highgui',    required : true)

ilsvrcserver_deps = [m, zmq, hdf5,
                     opencv_core, opencv_imgcodecs, opencv_imgproc, opencv_highgui]
ilsvrcclient_deps = [m, zmq,
                     opencv_core, opencv_imgcodecs, opencv_imgproc, opencv_highgui]
ilsvrcgen_deps    = [m, hdf5,
                     avcodec, avutil,
                     opencv_core, opencv_imgcodecs, opencv_imgproc, opencv_highgui]


# Includes
incs = include_directories('/usr/include/ffmpeg')



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
ilsvrcgen_srcs    = files('src/ilsvrcgen.cpp')
ilsvrcgen         = executable('ilsvrcgen',    ilsvrcgen_srcs,
                               include_directories : incs,
                               dependencies        : ilsvrcgen_deps)
