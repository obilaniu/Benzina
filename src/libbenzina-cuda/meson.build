#
# Library: libbenzina-cuda
#
# Contains CUDA kernels for image resizing, colorspace conversion and sampling.
#


### Dependency Check
if not nvcc.found()
  libbenzinacuda = disabler()
  subdir_done()
endif


### CUDA code compilation
libbenzinacuda_cdefs    = ['-DBENZINA_CUDA_IS_SHARED=1']
libbenzinacudaCuda_srcs = files('kernels.cu')
libbenzinacudaCuda_objs = nvcc_compiler_generator.process(libbenzinacudaCuda_srcs,
    extra_args: [
        '-Xcompiler', '-fPIC',
        '-Xptxas', '--warn-on-double-precision-use,-O3',
        '-DBENZINA_CUDA_IS_BUILDING=1',
        '-DBENZINA_IS_SHARED=1',
    ] + libbenzinacuda_cdefs + benzina_iflags
)
libbenzinacudaCuda_sta  = static_library('pre-dlink', libbenzinacudaCuda_objs)
libbenzinacudaCuda_dep  = declare_dependency(link_whole: libbenzinacudaCuda_sta)
libbenzinacudaCuda_objs = custom_target ('libbenzinacudaCuda-dlink',
    command         : [nvcc, '-shared'] + nvcc_link_args + ['@INPUT@', '-o', '@OUTPUT@'],
    input           : libbenzinacudaCuda_sta,
    output          : ['@BASENAME@-dlink.o'],
    build_by_default: true,
    install         : false
)
libbenzinacudaCuda_sta = static_library('dlink', libbenzinacudaCuda_objs)
libbenzinacudaCuda_dep = declare_dependency(dependencies: libbenzinacudaCuda_dep,
                                            link_whole:   libbenzinacudaCuda_sta)


### C code compilation
libbenzinacuda_srcs = []
libbenzinacuda_deps = [threads, libm, libdl, libcuda, libcudart, libcudadevrt,
                       libbenzina, libbenzinacudaCuda_dep]
libbenzinacuda_sha  = shared_library('benzina-cuda',        libbenzinacuda_srcs,
                                     dependencies         : libbenzinacuda_deps,
                                     c_args               : libbenzinacuda_cdefs + ['-DBENZINA_CUDA_IS_BUILDING=1'],
                                     gnu_symbol_visibility: 'hidden',
                                     build_rpath          : librpath,
                                     install_rpath        : librpath,
                                     install              : true,
                                     install_dir          : libdir)
libbenzinacuda      = declare_dependency(include_directories: benzina_incs,
                                         compile_args       : libbenzinacuda_cdefs,
                                         link_with          : libbenzinacuda_sha,
                                         dependencies       : libbenzina.partial_dependency(compile_args: true))

