#
# Benzina Includes
#

subdir('benzina')

benzina_incs   = include_directories('.')
benzina_incdir = meson.current_source_dir()
benzina_iflags = ['-I'+benzina_incdir]
benzina_headers = [
    'attribute.h',
    'benzina.h',
    'benzina-old.h',
    'bits.h',
    'endian.h',
    'inline.h',
    'intops.h',
    'ptrops.h',
    'visibility.h',
    'bcachefs/bcachefs.h',
    'bcachefs/bcachefs-ops.h',
    'bcachefs/bcachefs-types.h',
    'iso/bmff.h',
    'iso/bmff-box.h',
    'iso/bmff-intops.h',
    'iso/bmff-types.h',
    'iso/iso.h',
    'itu/h264.h',
    'itu/h265.h',
    'itu/h26x.h',
    'plugins/nvdecode.h',
]
foreach h: benzina_headers
  install_headers('benzina'/h, install_dir: includedir/fs.parent('benzina'/h))
endforeach
