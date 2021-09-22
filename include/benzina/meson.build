#
# Benzina Generated Includes
#
# We generate the headers
#
#     benzina/config.h and
#     benzina/version.h,
#
# which must then be installed.
#
# Because in Meson the output: path cannot have a '/' in it, the configuration
# of these files *must* take place in this meson.build file and none other in
# order for it to "mask" the original config.h in the source tree.
#

install_headers(
    configure_file(configuration: configdata,
                   input:         'config.h',
                   output:        'config.h',
                   encoding:      'utf-8'),
    configure_file(configuration: configdata,
                   input:         'version.h',
                   output:        'version.h',
                   encoding:      'utf-8'),
    install_dir: includedir/'benzina',
)
