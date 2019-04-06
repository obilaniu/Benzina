# libnvcuvid stub

## Introduction

The video codec library, `libnvcuvid`, nominally ships with the NVIDIA driver,
and _not_ the CUDA Toolkit.

However, sometimes this library cannot be found, or is not usable. There are
several reasons this can happen:

1) We may want to build Benzina on a machine with CUDA Toolkit but no driver,
   for instance because it has no NVIDIA GPU.
2) The `libnvcuvid` on the build machine may be older than that of the target
   machine.
3) On Linux especially, package managers may install `libnvcuvid.so.1` but fail
   to create a symlink `libnvcuvid.so`. This makes GCC unable to find the
   library using the standard `-lnvcuvid` flags.
4) The Video Codec SDK does ship with a "stub" `libnvcuvid` library, but not
   for all systems and architectures.

We work around this by creating our *own* stub `libnvcuvid`, but there are a few
details to observe.

## Headers & License

Two headers are important for the NVDECODE API:

- `cuviddec.h`
- `nvcuvid.h`

These ship with the Video Codec SDK, and contain an MIT license. But these
headers directly `#include` another header - `cuda.h`, which ships with the
CUDA Toolkit, and is not MIT-licensed as well.

To compile software against libnvcuvid without relying at all on the CUDA
Toolkit or Video Codec SDK being installed, we must nevertheless supply a
`cuda.h` from somewhere, with enough "stuff" to support it being built.
There is, however, in NVIDIA's `nv_codec_headers` project an MIT-licensed
`dynlink_cuda.h` header which, although it does not define the functions
that `cuda.h` has, does define all the enums and types needed.

At build time, we rename this `include/dynlink_cuda.h` to `src/cuda.h` using
`configure_file()`, and use it exclusively to build the stub. Because
`include/` is used as an include directory, the dynamic-link variant
`dynlink_cuda.h` **must never** be present under the name `cuda.h` in
`include/`. This is why the renamed `cuda.h` is placed in the `src/`
subdirectory of the stub, and is also why the `src/` **must not** ever be
included by any target except the stub.

As a result, those portions of the project are MIT-licensed by NVIDIA, with the
exact terms in the respective headers.

## Library Stub

The library source file itself, `src/libnvcuvid.c`, is of my own authorship,
although of course requires at a bare minimum copying the function
declarations from the NVIDIA headers and turning them into (invalid)
definitions.

On Linux, the library is further build with the same `SONAME` as the true
`libnvcuvid.so`, namely `libnvcuvid.so.1`.

In any case, the stub is and must **not** be installed, since by definition
stubs are empty and non-functional.

