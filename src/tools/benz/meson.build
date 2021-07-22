#
# A thin wrapper for calling into libbenzina, on operating systems such as
# Darwin where it is not possible to link a PIE that is also a dynamically-
# linkable and -loadable shared library.
#

benzina_benz = executable(
    'benz', files('main.c'),
    dependencies:  libbenzina,
    build_rpath:   toolrpath,
    install_rpath: toolrpath,
    install_dir:   tooldir,
    install:       true,
)
