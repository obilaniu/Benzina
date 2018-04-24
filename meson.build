project('Benzina', 'c', 'cpp',
        version:                 '0.0.0',
        meson_version:           '>=0.45',
        default_options:         [
            'c_std=gnu99',
            'warning_level=2',
        ],
)


#
# Dependencies & Libraries
#
# Library and Include Search Paths
# More configurable defaults could surely be provided.
#
cc                = meson.get_compiler('c')
cudaLibDirs       = ['/usr/local/cuda/lib64']
cudaIncs          = include_directories('/usr/local/cuda/include')

threadsDep        = dependency('threads')
libm              = cc.find_library('m',                 required: true)
libdl             = cc.find_library('dl',                required: true)
libcuda           = cc.find_library('cuda',              required: true, dirs: cudaLibDirs)




#
# Subdirectories.
#
subdir('include')
subdir('src')
