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
libbenzinacudaCDefs    = ['-DBENZINA_CUDA_IS_SHARED=1']
libbenzinacudaCudaSrcs = files('kernels.cu')
libbenzinacudaCudaObjs = nvccCompGen.process(libbenzinacudaCudaSrcs, extra_args: [
    '-Xcompiler', '-fPIC',
    '-Xptxas', '--warn-on-double-precision-use,-O3',
    '-DBENZINA_CUDA_IS_BUILDING=1',
] + libbenzinacudaCDefs + libbenzinaCDefs + benzinaIFlags)
libbenzinacudaCudaSta  = static_library('pre-dlink', libbenzinacudaCudaObjs)
libbenzinacudaCudaDep  = declare_dependency(link_whole: libbenzinacudaCudaSta)
libbenzinacudaCudaObjs = custom_target ('libbenzinacudaCuda-dlink',
    command         : [nvcc, '-shared'] + nvccLinkArgs + ['@INPUT@', '-o', '@OUTPUT@'],
    input           : libbenzinacudaCudaSta,
    output          : ['@BASENAME@-dlink.o'],
    build_by_default: true,
    install         : false
)
libbenzinacudaCudaSta  = static_library('dlink', libbenzinacudaCudaObjs)
libbenzinacudaCudaDep  = declare_dependency(dependencies: libbenzinacudaCudaDep,
                                            link_whole:   libbenzinacudaCudaSta)


### C code compilation
libbenzinacudaSrcs  = []
libbenzinacudaDeps  = [threadsDep, libm, libdl,
                       libcuda, libcudart, libcudadevrt,
                       libbenzina, libbenzinacudaCudaDep,]
libbenzinacudaSha   = shared_library('benzina-cuda',        libbenzinacudaSrcs,
                                     dependencies         : libbenzinacudaDeps,
                                     c_args               : libbenzinacudaCDefs + ['-DBENZINA_CUDA_IS_BUILDING=1'],
                                     gnu_symbol_visibility: 'hidden',
                                     build_rpath          : librpath,
                                     install_rpath        : librpath,
                                     install              : true,
                                     install_dir          : libdir)
libbenzinacuda      = declare_dependency(include_directories: benzinaIncs,
                                         compile_args       : libbenzinacudaCDefs,
                                         link_with          : libbenzinacudaSha,
                                         dependencies       : libbenzina.partial_dependency(compile_args: true))

