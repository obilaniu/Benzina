#
# Benzina Includes
#
# We generate a benzina/config.h header, which must then be installed.
#

configh    = configure_file(configuration: configdata,
                            output:        'config.h',
                            output_format: 'c',
                            encoding:      'utf-8')

install_headers('benzina.h',
                'benzina-cuda.h',
                'bits.h',
                'visibility.h',
                configh,
                install_dir: includedir/'benzina')

subdir('plugins')
