## Linux

All Linux distributions are meant to be fully supported (please [report](https://github.com/hfp/libxsmm/issues/new) any compatibility issue). A shared library (`STATIC=0`) necessarily implies some performance hit when accessing thread-local memory (contended multicore execution). The GNU Compiler Collection prior to v5.1 may imply performance hits in some CPUID-dispatched code paths (non-JIT).

> In case of outdated Binutils, compilation can fail to assemble code that originates from code sections using Intrinsics (see issue [#170](https://github.com/hfp/libxsmm/issues/170) and [#212](https://github.com/hfp/libxsmm/issues/212#issuecomment-394620082)). To resolve the problem, please use `INTRINSICS=1` along with the desired target e.g., `AVX=3 MIC=0`, or `AVX=2`.

## CRAY

In addition to the regular Linux support, The CRAY Compiling Environment (CCE) is supported: Intel Compiler as well as the GNU Compiler Collection are detected even when invoked per CCE, and the CRAY compiler is likely configured to build for the architecture of the compute nodes and hence the compiler is sufficiently treated without specific build flags (`COMPATIBLE=1` is implicitly set). The CCE may suppress to build a shared library (`STATIC=0`), which also affects the TRACE facility (requires dynamic linkage even for static archives).

```bash
make CXX=CC CC=cc FC=ftn
```

The compatibility settings imply minor issues when using the CRAY compiler: full control and [customization](http://libxsmm.readthedocs.io/libxsmm_tune/) is not implemented, enabling symbols (`SYM=1`) appears to imply an unoptimized debug-build (due to the `-g` flag being present). Some sample codes/benchmarks enable symbols but are meant to not enable debug-code. The LIBXSMM library however is built without symbols by default.

## Windows

### Microsoft Windows

Microsoft Windows is [supported](https://github.com/hfp/libxsmm/wiki/Q&A#what-operating-systems-are-covered-by-libxsmm-and-what-about-microsoft-windows) using the Microsoft Visual Studio environment (no `make`). It is advised to review the build settings. However, the following configurations are available: `debug`, `release`, and release mode with `symbols`. JIT-code generation is enabled but limited to the MM domain (GEMM kernels and matcopy kernels; no transpose kernels). GEMM kernels with prefetch signature remain as non-prefetch kernels i.e., prefetch locations are ignored due to the effort of fully supporting the Windows calling convention. As a workaround and to properly preserve caller-state, each JIT-kernel call may be wrapped by an own function.

### Cygwin

Cygwin (non-MinGW) is fully supported. Please note, that all limitations of Microsoft Windows apply.

```bash
make
```

LIBXSMM can be built as a static library as well as a dynamic link library (STATIC=0).

### MinGW/Cygwin

This is about the Cygwin-hosted bits of MinGW. The `-fno-asynchronous-unwind-tables` compiler flag is automatically applied. Please note, that all limitations of Microsoft Windows apply.

```bash
make \
  CXX=x86_64-w64-mingw32-g++ \
  CC=x86_64-w64-mingw32-gcc \
  FC=x86_64-w64-mingw32-gfortran
```

To run tests, `BLAS=0` may be supplied (since Cygwin does not seem to provide BLAS-bits for the MinGW part). However, this may be different for "native" MinGW, or can be fixed by supplying a BLAS library somehow else.

### MinGW

This is about the "native" MinGW environment. Please note, there is the original [MinGW](www.mingw.org/) as well as a [fork](https://mingw-w64.org/) (made in 2007). Both of which can target Windows 64-bit. Here, the [MSYS2 installer](http://www.msys2.org/) (scroll down on that page to see the full installation instructions) has been used (see the [details](https://github.com/msys2/msys2/wiki/MSYS2-installation) on how to install missing packages).

```bash
pacman -S msys/make msys/python msys/diffutils \
  mingw64/mingw-w64-x86_64-gcc mingw64/mingw-w64-x86_64-gcc-fortran \
  mingw64/mingw-w64-x86_64-openblas
```

Similar to Cygwin/MinGW, the `-fno-asynchronous-unwind-tables` flag is automatically applied.

```bash
make
```

LIBXSMM can be built as a static library as well as a dynamic link library (`STATIC=0`).

## Apple macOS

LIBXSMM for macOS (OSX) is fully supported (i.e., it qualifies a release). The default is to rely on Apple's Clang based (platform-)compiler ("gcc"). However, the actual GCC as well as the Intel Compiler for macOS can be used.

## FreeBSD

LIBXSMM is occasionally tested under FreeBSD. For libxsmmext, it is necessary to install OpenMP (`sudo pkg install openmp`).

```bash
bash
gmake
```
An attempt to run the [tests](https://github.com/hfp/libxsmm/wiki/Validation) may ask for a LAPACK/BLAS installation (unless `BLAS=0` is given). Both, Netlib BLAS (reference) and OpenBLAS are available (in case of linker error due to the GNU Fortran runtime library, one can try `gmake CXX=g++7 CC=gcc7 FC=gfortran7` i.e., select a consistent tool chain and adjust `LD_LIBRARY_PATH` accordingly e.g., `/usr/local/lib/gcc7`).

## PGI Compiler

The PGI Compiler&#160;2019 (and later) is supported. Earlier versions were only occasionally tested and automatically enabled the `COMPATIBLE=1` and `INTRINSIC=0` settings. Still, atomic builtins seem incomplete (at least with `pgcc`) hence LIBXSMM built with PGI Compiler is not fully thread-safe (tests/threadsafety can fail). Support for GNU's libatomic has been incorporated mainly for PGI but is also missing built-in compiler support hence supposedly atomic operations are mapped to normal (non-atomic) code sequences (`LIBXSMM_SYNC_SYSTEM`).

```bash
make CXX=pgc++ CC=pgcc FC=pgfortran
```

### ARM AArch64

This section is not strictly about compiler compatibility but rather about AArch64 (v8.1) being supported, which practically covers the baseline ARM 64-bit architecture from embedded and mobile to supercomputers. The build and installation process of LIBXSMM is the same as for Intel Architecture (IA) and the library can be natively compiled or cross-compiled. The latter for instance looks like:

```bash
make PLATFORM=1 AR=aarch64-linux-gnu-ar \
  FC=aarch64-linux-gnu-gfortran \
  CXX=aarch64-linux-gnu-g++ \
  CC=aarch64-linux-gnu-gcc
```

**Note**: Apple&#160;M1 is supported but JIT code generation may fail due to macOS&#160;11 ("Big Sur"). LIBXSMM does not currently support macOS&#160;11.x (regardless of ARM or Intel Architecture).
