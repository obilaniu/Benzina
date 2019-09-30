#
# Benzina Includes
#
# We generate a benzina/config.h header, which must then be installed.
#

config_h   = configure_file(configuration: configdata,
                            input:         'config.h',
                            output:        'config.h',
                            encoding:      'utf-8')
version_h  = configure_file(configuration: configdata,
                            input:         'version.h',
                            output:        'version.h',
                            encoding:      'utf-8')

install_headers('attribute.h',
                'benzina.h',
                'benzina-old.h',
                'benzina-cuda.h',
                'bits.h',
                'bitops.h',
                 config_h,
                'endian.h',
                'inline.h',
                'iso.h',
                 version_h,
                'visibility.h',
                install_dir: includedir/'benzina')

subdir('itu')
subdir('plugins')
