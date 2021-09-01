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

install_headers(
    'attribute.h',
    'benzina.h',
    'benzina-old.h',
    'bits.h',
     config_h,
    'endian.h',
    'inline.h',
    'intops.h',
    'ptrops.h',
     version_h,
    'visibility.h',
    install_dir: includedir/'benzina'
)
install_headers(
    'bcachefs/bcachefs.h',
    'bcachefs/bcachefs-types.h',
    install_dir: includedir/'benzina'/'bcachefs'
)
install_headers(
    'iso/bmff.h',
    'iso/bmff-box.h',
    'iso/bmff-intops.h',
    'iso/bmff-types.h',
    'iso/iso.h',
    install_dir: includedir/'benzina'/'iso'
)
install_headers(
    'itu/h264.h',
    'itu/h265.h',
    'itu/h26x.h',
    install_dir: includedir/'benzina'/'itu'
)
install_headers(
    'plugins/nvdecode.h',
    install_dir: includedir/'benzina'/'plugins'
)
