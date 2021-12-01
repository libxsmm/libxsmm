# LIBXSMM

[![License](https://img.shields.io/badge/license-BSD3-blue.svg "BSD 3-Clause License")](LICENSE.md) [![Build status](https://badge.buildkite.com/2e962d4cfc7ddb10a6cd6c27b0d8033edf179a799e156cb363.svg "Buildkite Status")](https://github.com/hfp/libxsmm/wiki/Status) [![Coverity](https://scan.coverity.com/projects/7405/badge.svg "Coverity Analysis Status")](https://scan.coverity.com/projects/hfp-libxsmm) [![ReadtheDocs](https://readthedocs.org/projects/libxsmm/badge/?version=latest "Read the Docs")](https://libxsmm.readthedocs.io/)

LIBXSMM is a library for specialized dense and sparse matrix operations as well as for deep learning primitives such as small convolutions. The library is targeting Intel Architecture with <span>Intel&#160;SSE</span>, <span>Intel&#160;AVX</span>, <span>Intel&#160;AVX2</span>, <span>Intel&#160;AVX&#8209;512</span> (with VNNI and Bfloat16), and <span>Intel&#160;AMX</span> (Advanced Matrix Extensions) supported by future Intel processor code-named Sapphire Rapids. Code generation is mainly based on <span>Just&#8209;In&#8209;Time (JIT)</span> code specialization for compiler-independent performance (matrix multiplications, matrix transpose/copy, sparse functionality, and deep learning). LIBXSMM is suitable for "build once and deploy everywhere", i.e., no special target flags are needed to exploit the available performance. Supported GEMM datatypes are: `FP64`, `FP32`, `bfloat16`, `int16`, and `int8`.

For a list questions and answers, please also have a look at [https://github.com/hfp/libxsmm/wiki/Q&A](https://github.com/hfp/libxsmm/wiki/Q&A).

**Where to go for documentation?**

* **ReadtheDocs**: [main](https://libxsmm.readthedocs.io/) and [sample](https://libxsmm.readthedocs.io/libxsmm_samples/) documentation with full text search.
* **PDF**: [main](https://github.com/hfp/libxsmm/raw/master/documentation/libxsmm.pdf) documentation file, and separate [sample](https://github.com/hfp/libxsmm/raw/master/documentation/libxsmm_samples.pdf) documentation.
* **Articles**: [magazine article](https://software.intel.com/sites/default/files/parallel-universe-issue-34.pdf) incl. [sample code](https://github.com/hfp/libxsmm/tree/master/samples/magazine) (full list of [Articles](#articles)).

<a name="getting-started"></a><a name="hello-libxsmm"></a>**Getting Started**: The following C++ code is focused on a specific functionality but may be considered as [Hello LIBXSMM](https://github.com/hfp/libxsmm/tree/master/samples/hello). Build the example with `cd /path/to/libxsmm; make STATIC=0` (shared library), save the code under `hello.cpp` (below) and compile with `g++ -I/path/to/libxsmm/include hello.cpp -L/path/to/libxsmm/lib -lxsmm -lblas -o hello` (GNU CCC), and finally execute with `LD_LIBRARY_PATH=/path/to/libxsmm/lib LIBXSMM_VERBOSE=2 ./hello`.

```cpp
#include <libxsmm.h>
#include <vector>
int main(/*int argc, char* argv[]*/) {
  typedef double T;
  int batchsize = 1000, m = 13, n = 5, k = 7;
  std::vector<T> a(batchsize * m * k), b(batchsize * k * n), c(m * n, 0);
  /* C/C++ and Fortran interfaces are available */
  typedef libxsmm_mmfunction<T> kernel_type;
  /* generates and dispatches a matrix multiplication kernel (C++ functor) */
  kernel_type kernel(LIBXSMM_GEMM_FLAG_NONE, m, n, k, 1.0 /*alpha*/, 1.0 /*beta*/);
  assert(kernel);
  for (int i = 0; i < batchsize; ++i) { /* initialize input */
    for (int ki = 0; ki < k; ++ki) {
      for (int j = 0; j < m; ++j) a[i * j * ki] = static_cast<T>(1) / ((i + j + ki) % 25);
      for (int j = 0; j < n; ++j) b[i * j * ki] = static_cast<T>(7) / ((i + j + ki) % 75);
    }
  }
  /* kernel multiplies and accumulates matrices: C += Ai * Bi */
  for (int i = 0; i < batchsize; ++i) kernel(&a[i * m * k], &b[i * k * n], &c[0]);
}
```

Plain [C code](https://github.com/hfp/libxsmm/blob/master/samples/hello/hello.c) as well as [Fortran code](https://github.com/hfp/libxsmm/blob/master/samples/hello/hello.f) resemble the same [example](https://github.com/hfp/libxsmm/tree/master/samples/hello).

<a name="what-is-a-small-matrix-multiplication"></a>**What is a small matrix multiplication?** When characterizing the problem-size by using the M, N, and K parameters, a problem-size suitable for LIBXSMM falls approximately within <i>(M&#160;N&#160;K)<sup>1/3</sup>&#160;&lt;=&#160;64</i> (which illustrates that non-square matrices or even "tall and skinny" shapes are covered as well). The library is typically used to generate code up to the specified [threshold](documentation/libxsmm_tune.md#auto-dispatch). Raising the threshold may not only generate excessive amounts of code (due to unrolling in M or K dimension), but also miss to implement a tiling scheme to effectively utilize the cache hierarchy. For auto-dispatched problem-sizes above the configurable threshold (explicitly JIT'ted code is **not** subject to the threshold), LIBXSMM is falling back to BLAS. In terms of GEMM, the supported kernels are limited to *Alpha := 1*, *Beta := \{ 1, 0 \}*, and *TransA := 'N'*.

<a name="what-is-a-small-convolution"></a>**What is a small convolution?** In the last years, new workloads such as deep learning and more specifically convolutional neural networks (CNN) emerged and are pushing the limits of today's hardware. One of the expensive kernels is a small convolution with certain kernel sizes such that calculations in the frequency space is not the most efficient method when compared with direct convolutions. LIBXSMM's current support for convolutions aims for an easy to use invocation of small (direct) convolutions, which are intended for CNN training and classification.

## Interfaces and Domains<a name="interfaces"></a>

### Overview<a name="general-interface"></a>

Please have a look at [https://github.com/hfp/libxsmm/tree/master/include](https://github.com/hfp/libxsmm/tree/master/include) for all published functions. Get started with the following list of available domains and documented functionality:

* MM: [Matrix Multiplication](#matrix-multiplication)
* DNN: [Deep Neural Networks](#deep-neural-networks)
* AUX: [Service Functions](#service-functions)
* PERF: [Performance](#performance)
* BE: [Backend](#jit-backend)

To initialize library internal resources, an explicit initialization routine helps to avoid lazy initialization overhead when calling LIBXSMM for the first time. The library deallocates internal resources at program exit, but also provides a companion of the afore mentioned initialization (finalize).

```C
/** Initialize the library; pay for setup cost at a specific point. */
void libxsmm_init(void);
/** De-initialize the library and free internal memory (optional). */
void libxsmm_finalize(void);
```

### Matrix Multiplication<a name="interface-for-matrix-multiplication"></a>

This domain (MM) supports Small Matrix Multiplications (SMM), batches of multiple multiplications as well as the industry-standard interface for GEneral Matrix Matrix multiplication (GEMM).

The [Matrix Multiplication domain (MM)](documentation/libxsmm_mm.md) contains routines for:

* [Small, tiled, and parallelized matrix multiplications](documentation/libxsmm_mm.md#overview)
* [Manual code dispatch (customized matrix batches)](documentation/libxsmm_mm.md#manual-code-dispatch)
* [Batched multiplication (explicit interface)](documentation/libxsmm_mm.md#batched-multiplication)
* [Call wrapper (static and dynamic linkage)](documentation/libxsmm_mm.md#call-wrapper)

### Deep Learning<a name="interface-for-convolutions"></a>

This domain (DL) is detailed by a separate [document](documentation/libxsmm_dl.md). It may be inspiring to have a look at the lightweight GxM framework, which uses LIBXSMM for end-to-end Deep Learning.

### Service Functions

For convenient operation of the library and to ease integration, some service routines are available. These routines may not belong to the core functionality of LIBXSMM (SMM or DNN domain), but users are encouraged to use this domain (AUX). There are two categories: <span>(1)&#160;routines</span> which are available for C and FORTRAN, and <span>(2)&#160;routines</span> that are only available per C interface.

The [service function domain (AUX)](documentation/libxsmm_aux.md) contains routines for:

* [Getting and setting the target architecture](documentation/libxsmm_aux.md#getting-and-setting-the-target-architecture)
* [Getting and setting the verbosity](documentation/libxsmm_aux.md#getting-and-setting-the-verbosity)
* [Measuring time durations (timer)](documentation/libxsmm_aux.md#timer-facility)
* [Dispatching user-data and multiple kernels](documentation/libxsmm_aux.md#user-data-dispatch)
* [Loading and storing data (I/O)](documentation/libxsmm_aux.md#meta-image-file-io)
* [Allocating memory](documentation/libxsmm_aux.md#memory-allocation)

### Backend<a name="jit-backend"></a>

More information about the JIT-backend and the code generator can be found in a separate [document](documentation/libxsmm_be.md). The [encoder sample collection](https://github.com/hfp/libxsmm/tree/master/samples/encoder) can help to get started writing a kernel using LIBXSMM. Please note, LIBXSMM's stand-alone <a name="generator-driver"></a>[generator-driver](documentation/libxsmm_be.md#generator-driver) is considered legacy (deprecated).

## Build Instructions

### Overview

The main interface file is *generated*, and it is therefore **not** stored in the code repository. Instead, one may have a look at the code generation template files for [C/C++](https://github.com/hfp/libxsmm/blob/master/src/template/libxsmm.h#L36) and [FORTRAN](https://github.com/hfp/libxsmm/blob/master/src/template/libxsmm.f#L32). There are two ways prepared to build and use LIBXSMM:

* [Classic Library (ABI)](#classic-library-abi) and [Link Instructions](#link-instructions) (C/C++ and FORTRAN)
* [Header-Only](#header-only) (C and C++)

**Note**: LIBXSMM is available as prebuilt package for Fedora/RedHat/CentOS, Ubuntu, and FreeBSD. Further, LIBXSMM can be installed with the [Spack Package Manager](http://computation.llnl.gov/projects/spack-hpc-package-manager) or per [EasyBuild+EasyConfig](https://github.com/easybuilders).

### Classic Library (ABI)

The build system relies on <span>GNU&#160;Make</span> (typically associated with the `make` command, but e.g. FreeBSD is calling it `gmake`). The build can be customized by using <span>key&#8209;value</span> pairs. <span>Key&#8209;value</span> pairs can be supplied in two ways: <span>(1)&#160;after</span> the "make" command, or <span>(2)&#160;prior</span> to the "make" command (`env`) which is effectively the same as exporting the <span>key&#8209;value</span> pair as an environment variable (`export`, or `setenv`). Both methods can be mixed (the second method may require make's `-e` flag).

<a name="zero-config-abi"></a>In contrast to [header-only](#zero-config) which does not require configuration by default, 3rd-party build systems can compile and link LIBXSMM's sources but still avoid configuring the library (per `libxsmm_config.py`). The prerequisite to omit configuration is to opt-in by defining LIBXSMM_DEFAULT_CONFIG (`-D`). The zero-config feature is not available for LIBXSMM's Fortran interface.

**Note**: By default, C/C++ and FORTRAN compilers are needed (some sample code is written in C++). Beside of specifying the compilers (`make CXX=g++ CC=gcc FC=gfortran` and maybe `AR=ar`), the need for a FORTRAN compiler can be relaxed (`make FC=` or `make FORTRAN=0`). The latter affects the availability of the MODule file and the corresponding `libxsmm.f` library (the interface `libxsmm.f` is still generated).

The build system considers a set of given key-value pairs as a single unique build and triggers a rebuild for a distinct set of flags. For more advanced builds or additional background, please consult the section about [Customization](documentation/libxsmm_tune.md). To generate the interface of the library inside of the `include` directory and to build the static library (by default, STATIC=1 is activated). Run any (or both) of the following command(s):

```bash
make STATIC=0
make
```

On CRAY systems, the CRAY Compiling Environment (CCE) should be used regardless of utilizing the CRAY compiler, the Intel Compiler, or the <span>GNU&#160;Compiler Collection (GCC)</span>. The CCE is eventually suppressing to build shared libraries (STATIC=0). In any case, <span>(1)&#160;switch</span> to the desired compiler (module load/switch), and <span>(2)&#160;rely</span> on:

```bash
make CXX=CC CC=cc FC=ftn
```

A variety of build environments is out-of-the-box compatible, see [https://github.com/hfp/libxsmm/wiki/Compatibility](https://github.com/hfp/libxsmm/wiki/Compatibility). If the build process is not successful, it may help to avoid advanced GCC flags. This is useful with a tool chain, which pretends to be GCC-compatible (and is treated as such) but fails to consume the afore mentioned flags:

```bash
make COMPATIBLE=1
```

<a name="outdated-binutils"></a>In case of outdated Binutils, compilation can fail to assemble code when building the library (this has nothing to do with JIT-generated code and it does not affect how JIT-code is targeting the system). LIBXSMM implements some functionality using compiler-intrinsics and multiple code-paths which are scheduled according to CPUID. In contrast to `INTRINSICS=2` (default), `INTRINSICS=1` enables a fully static code path according to the desired target. If no target is given (e.g., `AVX=3`, or `AVX=2`), instruction set extensions cannot be leveraged for such code-paths. Try to fix failing compilation by building the latest GNU Binutils (and `export PATH=/path/to/binutils/bin:${PATH}`). Binutils are versioned independently of <span>GNU&#160;GCC</span> and other compilers. If one cannot update Binutils, work around with a CPUID-value as tabulated in [libxsmm_cpuid.h](https://github.com/hfp/libxsmm/blob/master/include/libxsmm_cpuid.h): start at the upper end (less than 1999) and decrement until compilation passes (make INTRINSICS=_CPUID_, e.g., `make INTRINSICS=1021`). As a last resort, rely on a fully static code path:

```bash
make INTRINSICS=1
```

To test and validate a build, please consult [https://github.com/hfp/libxsmm/wiki/Validation](https://github.com/hfp/libxsmm/wiki/Validation). To run some basic sanity checks, remember that each set of given key-value pairs represents a different build (and test):

```bash
make STATIC=0 tests
```

To remove intermediate files, or to remove all generated files and folders (including the interface and the library archives), run one of the make-targets below. An additional distclean-target recursively cleans the entire tree (after <span>version&#160;1.9</span>).

```bash
make clean
make realclean
```

<a name="fortran"></a>FORTRAN code can make use of LIBXSMM:

* By using the module and linking with `libxsmmf`, `libxsmm`, and (optionally) `libxsmmext`,
* <a name="header-only-fortran"></a>By including `libxsmm.f` and linking with `libxsmm`, and (optionally) `libxsmmext`, or
* By (implicitly) calling a SUBROUTINE and linking with `libxsmm`, and (optionally) `libxsmmext`.

**Note**: Using the Fortran module or including the interface, requires at least a <span>Fortran&#160;2003</span> compiler (F2K3). <span>FORTRAN&#160;77</span> compatibility is only implicitly available (no interface), and the available subset of routines is documented in `libxsmm.f` and marked with [comments](https://github.com/hfp/libxsmm/search?q=implementation+provided+for+Fortran+77+compatibility) (part of the implementation).

### Header-Only

<span>Version&#160;1.4.4</span> introduced support for "header-only" usage in C and C++. By only including `libxsmm_source.h` allows to get around building the library. However, this gives up on a clearly defined application binary interface (ABI). An ABI may allow for hot-fixes after deploying an application (when relying on the shared library form), and it may also ensure to only rely on the public interface of LIBXSMM. In contrast, the header-only form not only exposes the internal implementation of LIBXSMM but can also increase the turnaround time during development of an application (due to longer compilation times). The header file is intentionally named "libxsmm_**source**.h" since this header file relies on the [src](https://github.com/hfp/libxsmm/tree/master/src) directory (with the implications as noted earlier).

<a name="zero-config"></a>The header-only form depends on `libxsmm_source.h` which is *generated* according to the content of the source folder (`src`). <span>LIBXSMM&#160;1.16</span> (and later) provides header-only support without invoking a make-target (zero configuration) for any given checkout of LIBXSMM. To use configured header-only (non-default), LIBXSMM_CONFIGURED must be defined (`-D`). Previously, it was necessary to invoke `make header-only` (v1.6.2 or later), `make cheader` (prior to v1.6.2), or any target building the library (`make`). The zero-config feature allows 3rd-party build systems an easier integration of LIBXSMM, which also holds true if the system builds LIBXSMM from source (see [classic ABI](#zero-config-abi)). Fortran code may [include](#header-only-fortran) `libxsmm.f` but still requires that interface to be generated.

**Note**: building an application applies the same build settings to LIBXSMM! For instance, to omit debug code inside of LIBXSMM `NDEBUG` must be defined (`-DNDEBUG`).

## Link Instructions

Using the [classic ABI](#classic-library-abi) (including [Fortran](#fortran) code), requires linking LIBXSMM against the application. The library is agnostic with respect to the threading-runtime, and therefore an application is free to use any threading runtime (e.g., OpenMP). The library is also thread-safe, and multiple application threads can call LIBXSMM's routines concurrently. Enabling OpenMP for LIBXSMM's main library is supported as well (OMP=1), and mostly affects the synchronization primitives used inside of the library. All of the "omp" functionality (function postfix) is served by the `libxsmmext` library, which is automatically built with OpenMP enabled. When using this "omp" functionality, `libxsmmext` needs to be present at the link line.

<a name="table-of-libraries"></a>Library | Purpose
:-------------|---------
libxsmm       | Thread-safe core functions (same routine can be called concurrently). Contains routines that can take a thread-ID and the number of library-external threads.
libxsmmf      | Necessary when using the Fortran MODule but not when including `libxsmm.f` or relying on implicit interfaces ([Fortran 77](https://github.com/hfp/libxsmm/search?q=implementation+provided+for+Fortran+77+compatibility)).
libxsmmext    | Provides library-internal OpenMP-threaded functions carrying the `omp` postfix when compared to function name names of the core library.
libxsmmnoblas | Supplies faked symbols for `dgemm` (and others) and thereby removes the need to link against a LAPACK/BLAS library.

<a name="pkg-config"></a>To ease linking with LIBXSMM, `pkg-config` can be used. For example:

```bash
export PKG_CONFIG_PATH=/path/to/libxsmm/lib
pkg-config libxsmm --libs
```

Similarly, an application is free to choose any BLAS or LAPACK library (if the link model available on the OS supports this), and therefore linking GEMM routines when linking LIBXSMM itself (by supplying BLAS=1&#124;2) may prevent a user from making this decision at the time of linking the actual application. To use LIBXSMM without GEMM-related functionality, any BLAS-dependency can be removed in two ways: <span>(1)&#160;building</span> a special library with `make BLAS=0`, or <span>(2)&#160;linking</span> the application against the `libxsmmnoblas` library. If an application however uses BLAS already, the [Call Wrapper](documentation/libxsmm_mm.md#call-wrapper) can be used to intercept existing BLAS calls (and to rely on LIBXSMM instead).

**Note**: LIBXSMM does not support to dynamically link `libxsmm` or `libxsmmext` ("so"), when BLAS is linked statically ("a"). If BLAS is linked statically, the static version of LIBXSMM must be used!

### Installation

There are two main mechanisms to install LIBXSMM (both mechanisms can be combined): <span>(1)&#160;building</span> the library in an <span>out&#8209;of&#8209;tree</span> fashion, and <span>(2)&#160;installing</span> into a certain location. <a name="install-build"></a>Building in an <span>out&#8209;of&#8209;tree</span> fashion looks like:

```bash
cd libxsmm-install
make -f /path/to/libxsmm/Makefile
```

<a name="install-prefix"></a>Installation into a specific location looks like (`PREFIX` or `DESTDIR`):

```bash
make MNK="1 2 3 4 5" PREFIX=/path/to/libxsmm-install install
```

<a name="install-destdir"></a>Both `PREFIX` and `DESTDIR` are equivalent and can be relative or absolute paths. An installation can be repeated for different locations without triggering a rebuild. The prefix directory *inside* of each of the [package configuration files](#pkg-config) is set to where LIBXSMM is built (staging folder) unless `PREFIX` or `DESTDIR` is specified. The effect of `PREFIX` (or `DESTDIR`) with respect to the pkg-config files is independent of whether the install-target is invoked or not (make).

Further, performing `make install-minimal` omits the documentation (default: `PREFIX/share/libxsmm`). Moreover, PINCDIR, POUTDIR, PBINDIR, and PDOCDIR allow to customize the locations underneath of the PREFIX location. To build a general package for an unpredictable audience (Linux distribution, or similar), it is advised to not over-specify or customize the build step, i.e., JIT, SSE, AVX, OMP, BLAS, etc. should not be used. The following is building and installing a complete set of libraries where the generated interface matches both the static and the shared libraries:

```bash
make PREFIX=/path/to/libxsmm-install STATIC=0 install
make PREFIX=/path/to/libxsmm-install install
```

## Runtime Control<a name="running"></a>

### Handling Errors

The library handles errors with mechanisms available to the C programming language (no exceptions). The backend uses result codes passed by an argument rather than an actual return value. Such an argument is often a descriptor (struct) guiding and covering the state of the code generation. The frontend however may not hand-out any error state, which can be a big relief on the call-side. Instead, the frontend implements a [verbose mode](#verbose-mode) to inform about unexpected input or an error captured from the backend. Guiding principles of LIBXSMM are muted operation by default (non-verbose) and no unexpected exit from execution.

### Verbose Mode

The [verbose mode](documentation/libxsmm_aux.md#getting-and-setting-the-verbosity) (level of verbosity) allows for an insight into the code dispatch mechanism by receiving a small tabulated statistic as soon as the library terminates. The design point for this functionality is to not impact the performance of any critical code path, i.e., verbose mode is always enabled and does not require symbols (SYM=1) or debug code (DBG=1). The statistics appears (`stderr`) when the environment variable LIBXSMM_VERBOSE is set to a non-zero value. For example:

```bash
LIBXSMM_VERBOSE=1 ./myapplication
[... application output]

HSW/SP      TRY    JIT    STA    COL
   0..13      0      0      0      0
  14..23      0      0      0      0
 24..128      3      3      0      0
```

The tables are distinct between single-precision and double-precision, but either table is pruned if all counters are zero. If both tables are pruned, the library shows the code path which would have been used for JIT'ting the code: `LIBXSMM_TARGET=hsw` (otherwise the code path is shown in the table's header). The actual counters are collected for three buckets: small kernels (<span>MNK<sup>1/3</sup>&#160;&lt;=&#160;13</span>), medium-sized kernels (<span>13&#160;&lt;&#160;MNK<sup>1/3</sup>&#160;&lt;=&#160;23</span>), and larger kernels (<span>23&#160;&lt;&#160;MNK<sup>1/3</sup>&#160;&lt;=&#160;64</span>; the actual upper bound depends on LIBXSMM_MAX_MNK as selected at compile-time). Keep in mind, that "larger" is supposedly still small in terms of arithmetic intensity (which grows linearly with the kernel size). Unfortunately, the arithmetic intensity depends on the way a kernel is used (which operands are loaded/stored into main memory) and it is not performance-neutral to collect this information.

The TRY counter represents all attempts to register statically generated kernels, and all attempts to dynamically generate and register kernels. The TRY counter includes rejected JIT requests due to unsupported GEMM arguments. The JIT and STA counters distinct the successful cases of the afore mentioned event (TRY) into dynamically (JIT) and statically (STA) generated code. In case the capacity (<span>O(*n*)&#160;=&#160;10<sup>5</sup></span>) of the code registry is exhausted, no more kernels can be registered although further attempts are not prevented. Registering many kernels (<span>O(*n*)&#160;=&#160;10<sup>3</sup></span>) may ramp the number of hash key collisions (COL), which can degrade performance. The latter is prevented if the small thread-local cache is utilized effectively.

Since explicitly JIT-generated code (`libxsmm_?mmdispatch`) does not fall under the THRESHOLD criterion, the above table is extended by one line if large kernels have been requested. This indicates a missing threshold-criterion (customized dispatch), or asks for cache-blocking the matrix multiplication. The latter is already implemented by LIBXSMM's "medium-sized" GEMM routines (`libxsmm_?gemm_omp`), which perform a tiled multiplication. Setting a verbosity level of at least two summarizes the number of registered JIT-generated kernels, which includes the total size and counters for GEMM, MCOPY (matrix copy), and TCOPY (matrix transpose) kernels.

```bash
Registry: 20 MB (gemm=0 mcopy=14 tcopy=0)
```

If the call-wrapper is used, an additional runtime statistic becomes available (see [Call Wrapper](documentation/libxsmm_mm.md#call-wrapper)).

<a name="objdump"></a>**Note**: Setting LIBXSMM_VERBOSE to a negative value will binary-dump each generated JIT kernel to a file with each file being named like the function name shown in [Intel VTune](documentation/libxsmm_prof.md#intelvtuneamplifier). Disassembly of the raw binary files can be accomplished by:

```bash
objdump -D -b binary -m i386 -M x86-64 [JIT-dump-file]
```

### Call Trace

During the initial steps of employing the LIBXSMM API, one may rely on a debug version of the library (`make DBG=1`). The latter also implies console output (`stderr`) in case of an error/warning condition inside of the library. It is also possible to print the execution flow (call trace) inside of LIBXSMM (can be combined with DBG=1 or OPT=0):

```bash
make TRACE=1
```

Building an application which traces calls (inside of the library) requires the shared library of LIBXSMM, alternatively the application is required to link the static library of LIBXSMM in a dynamic fashion (GNU tool chain: `-rdynamic`). Tracing calls (without debugger) can be then accomplished by an environment variable called LIBXSMM_TRACE.

```bash
LIBXSMM_TRACE=1 ./myapplication
```

Syntactically up to three arguments separated by commas (which allows to omit arguments) are taken (*tid*,*i*,*n*): *tid* signifies the ID of the thread to be traced with 1...NTHREADS being valid and where LIBXSMM_TRACE=1 is filtering for the "main thread" (in fact the first thread running into the trace facility); grabbing all threads (no filter) can be achieved by supplying a negative id (which is also the default when omitted). The second argument is pruning higher levels of the call-tree with *i=1* being the default (level zero is the highest at the same level as the main function). The last argument is taking the number of inclusive call levels with *n=-1* being the default (signifying no filter).

Although the `ltrace` (Linux utility) provides similar insight, the trace facility might be useful due to the afore mentioned filtering expressions. Please note that the trace facility is severely impacting the performance (even with LIBXSMM_TRACE=0), and this is not just because of console output but rather since inlining (internal) functions might be prevented along with additional call overhead on each function entry and exit. Therefore, debug symbols can be also enabled separately (`make SYM=1`; implied by TRACE=1 or DBG=1) which might be useful when profiling an application.

## Performance

<a name="profiling"></a>Profiling an application, which uses LIBXSMM's JIT-code is well-supported. The library supports <span>Intel&#160;VTune&#160;Amplifier</span> and <span>Linux&#160;perf</span>. Details are given on how to include profiler support, and how to run the application.

* [Profiling using Intel VTune Amplifier](documentation/libxsmm_prof.md#intelvtuneamplifier)
* [Profiling using Linux perf](documentation/libxsmm_prof.md#linuxperf)

<a name="tuning"></a>At build time, a variety of options exist to customize LIBXSMM. The library is setup for a broad range of use cases, which include sophisticated defaults for typical use.

* [Customizing performance](documentation/libxsmm_tune.md#tuning)
* <a name="auto-dispatch"></a>[Tuning auto-dispatch](documentation/libxsmm_tune.md#auto-dispatch)

<a name="results"></a>To find performance results of applications or performance reproducers, the repository provides an orphaned branch called "results" which collects collateral material such as measured performance results along with explanatory figures. The results can be found at [https://github.com/hfp/libxsmm/tree/results#libxsmm-results](https://github.com/hfp/libxsmm/tree/results#libxsmm-results), or the results can be cloned as shown below.

```bash
git clone --branch results \
  https://github.com/hfp/libxsmm.git \
  libxsmm-results
```

Please note that comparing performance results depends on whether the operands of the matrix multiplication are streamed or not. For example, multiplying with all matrices covered by the L1 cache may have an emphasis towards an implementation which perhaps performs worse for the real workload (if this real workload needs to stream some or all matrices from the main memory). Most of the [code samples](https://github.com/hfp/libxsmm/tree/master/samples) are aimed to reproduce performance results, and it is encouraged to model the exact case or to look at real [applications](#applications).

## Applications

### High Performance Computing (HPC)

<b>[1]&#160;</b>[https://cp2k.org/](https://cp2k.org/): Open Source Molecular Dynamics and the [DBCSR library](https://github.com/cp2k/dbcsr), which processes batches of small matrix multiplications. The batches originate from a distributed block-sparse matrix with problem-specific small matrices. Starting with [CP2K&#160;3.0](https://www.cp2k.org/version_history), LIBXSMM can substitute CP2K's `libsmm` library.

<b>[2]&#160;</b>[https://github.com/SeisSol/SeisSol/](https://github.com/SeisSol/SeisSol/): SeisSol is one of the leading codes for earthquake scenarios, for simulating dynamic rupture processes. LIBXSMM provides highly optimized assembly kernels which form the computational back-bone of SeisSol (see [https://github.com/TUM-I5/seissol_kernels/](https://github.com/TUM-I5/seissol_kernels/).

<b>[3]&#160;</b>[https://github.com/NekBox/NekBox](https://github.com/NekBox/NekBox): NekBox is a highly scalable and portable spectral element code, which is inspired by the [Nek5000](https://nek5000.mcs.anl.gov/) code. NekBox is specialized for box geometries and intended to prototype new methods as well as to leverage FORTRAN beyond the FORTRAN&#160;77 standard. LIBXSMM can be used to substitute the [MXM_STD](https://github.com/Nek5000/NekBox/blob/box/mxm_std.F90) code. Please also note LIBXSMM's [NekBox reproducer](https://github.com/hfp/libxsmm/tree/master/samples/nek#nek-sample-collection).

<b>[4]&#160;</b>[https://github.com/Nek5000/Nek5000](https://github.com/Nek5000/Nek5000): Nek5000 is the open-source, highly-scalable, always-portable spectral element code from [https://nek5000.mcs.anl.gov/](https://nek5000.mcs.anl.gov/). The development branch of the Nek5000 code [incorporates](https://github.com/Nek5000/Nek5000/blob/master/core/mxm_wrapper.f) LIBXSMM.

<b>[5]&#160;</b>[http://pyfr.org/](http://pyfr.org/): PyFR is an open-source Python based framework for solving advection-diffusion type problems on streaming architectures by using the flux reconstruction approach. PyFR&#160;1.6.0 optionally [incorporates LIBXSMM](http://pyfr.org/user_guide.php) as a matrix multiplication provider for the OpenMP backend. Please also note LIBXSMM's [PyFR-related code sample](https://github.com/hfp/libxsmm/tree/master/samples/pyfr).

<b>[6]&#160;</b>[http://dial3343.org/about/](http://dial3343.org/about/): The Extreme-scale Discontinuous Galerkin Environment (EDGE) is a solver for hyperbolic partial differential equations with emphasis on seismic simulations. The EDGE [source code](https://github.com/3343/edge) optionally relies on LIBXSMM, but for high performance LIBXSMM's kernels are highly recommended.

<b>[7]&#160;</b>[https://sxs-collaboration.github.io/spectre/](https://sxs-collaboration.github.io/spectre/): SpECTRE is an open-source code for multi-scale, multi-physics problems in astrophysics and gravitational physics which runs at Petascale and is designed for Exascale computers. In the future, SpECTRE may be applied to problems across discipline boundaries in fluid dynamics, geoscience, plasma physics, nuclear physics, and engineering.

<b>[8]&#160;</b>[https://ceed.exascaleproject.org/ceed-code/](https://ceed.exascaleproject.org/ceed-code/): The Center for Efficient Exascale Discretizations (CEED) is building on the efforts of the Nek5000, MFEM, MAGMA, OCCA and PETSc projects to develop application program interfaces (APIs), both at high-level and at low-level to enable applications to take advantage of high-order methods. The CEED low-level API, [libCEED](https://ceed.exascaleproject.org/libceed/) uses LIBXSMM as a [backend](https://github.com/CEED/libCEED#backends) for high performance on CPUs.

<b>[9]&#160;</b>[https://github.com/romeric/Fastor](https://github.com/romeric/Fastor): Fastor is a lightweight high performance tensor algebra framework for modern C++ and can optionally use LIBXSMM as [JIT-backend](https://github.com/romeric/Fastor/wiki/9.-Using-the-LIBXSMM-MKL-JIT-backend).

### Machine Learning (ML)

<b>[10]&#160;</b>[https://github.com/plaidml/plaidml](https://github.com/plaidml/plaidml): PlaidML is an open source tensor compiler aiming for performance portability across a wide range of CPUs, GPUs and other accelerators. Combined with Intel’s nGraph compiler, PlaidML is targeting popular deep learning frameworks such as PyTorch, Keras (TensorFlow), and OpenVino. [PlaidML/v1](https://github.com/plaidml/plaidml/tree/plaidml-v1) (development branch) adopted [MLIR](https://mlir.llvm.org/), an extensible compiler infrastructure gaining industry-wide adoption. PlaidML/v1 started using LIBXSMM as backend for targeting CPUs.

<b>[11]&#160;</b>[https://github.com/intel/intel-extension-for-pytorch](https://github.com/intel/intel-extension-for-pytorch): Intel Extension for PyTorch aims for a smooth user experience of PyTorch on CPUs by the means of good performance. The extension pack started to rely on [LIBXSMM for achieving high performance on CPUs](https://arxiv.org/abs/2005.04680).

<b>[12]&#160;</b>[https://www.tensorflow.org/](https://tensorflow.org/): <span>TensorFlow&trade;<span> is an open source software library for numerical computation using data flow graphs. TensorFlow was originally developed by researchers and engineers working on the Google Brain Team for the purposes of conducting machine learning and deep neural networks research. LIBXSMM was once [used](documentation/tensorflow.md) to increase the performance of TensorFlow on Intel hardware.

<b>[13]&#160;</b>[https://github.com/IntelLabs/SkimCaffe](https://github.com/IntelLabs/SkimCaffe#skimcaffe-specific-description): SkimCaffe from Intel Labs is a Caffe branch for training of sparse CNNs, which provide 80-95% sparsity in convolutions and fully-connected layers. LIBXSMM's SPMDM domain (SParseMatrix-DenseMatrix multiplication) evolved from SkimCaffe, and since then LIBXSMM implements the sparse operations in SkimCaffe.

<b>[14]&#160;</b>[https://github.com/baidu-research/DeepBench](https://github.com/baidu-research/DeepBench#deepbench): The primary purpose of DeepBench is to benchmark operations that are important to deep learning on different hardware platforms. LIBXSMM's DNN primitives have been [incorporated into DeepBench](https://github.com/baidu-research/DeepBench/tree/master/code/intel/convolution/libxsmm_conv) to demonstrate an increased performance of deep learning on Intel hardware.

### Automated Driving (AD)

<b>[15]&#160;</b>[https://software.seek.intel.com/accelerating-eigen-math-library](https://software.seek.intel.com/accelerating-eigen-math-library): Accelerating The Eigen Math Library for Automated Driving Workloads: The Need for Speed in Kalman Filtering. An article in [Issue&#160;31](https://software.intel.com/content/www/us/en/develop/download/parallel-universe-magazine-issue-31-january-2018.html) of The Parallel Universe magazine ([pdf](https://software.intel.com/content/dam/develop/public/us/en/documents/parallel-universe-issue-31.pdf)).

## References

<b>[1]&#160;</b>[https://sc19.supercomputing.org/proceedings/tech_poster/tech_poster_pages/rpost244.html](https://sc19.supercomputing.org/proceedings/tech_poster/tech_poster_pages/rpost244.html): High-Performance Deep Learning via a Single Building Block ([poster](https://sc19.supercomputing.org/proceedings/tech_poster/poster_files/rpost244s2-file2.pdf) and [abstract](https://sc19.supercomputing.org/proceedings/tech_poster/poster_files/rpost244s2-file3.pdf)), SC’19: The International Conference for High Performance Computing, Networking, Storage, and Analysis, Denver (Colorado).

<b>[2]&#160;</b>[https://dl.acm.org/doi/10.1109/SC.2018.00069](https://dl.acm.org/doi/10.1109/SC.2018.00069): Anatomy of High-Performance Deep Learning Convolutions on SIMD Architectures ([paper](https://arxiv.org/pdf/1808.05567.pdf)). SC'18: The International Conference for High Performance Computing, Networking, Storage, and Analysis, Dallas (Texas).

<b>[3]&#160;</b>[https://pasc17.pasc-conference.org/fileadmin/user_upload/pasc17/program/post116s2.pdf](https://pasc17.pasc-conference.org/fileadmin/user_upload/pasc17/program/post116s2.pdf): DBCSR: A Sparse Matrix Multiplication Library for Electronic Structure Codes (poster), PASC’17: The PASC17 Conference, Lugano (Switzerland).

<b>[4]&#160;</b>[https://sc17.supercomputing.org/SC17%20Archive/tech_poster/tech_poster_pages/post190.html](https://sc17.supercomputing.org/SC17%20Archive/tech_poster/tech_poster_pages/post190.html): Understanding the Performance of Small Convolution Operations for CNN on Intel Architecture ([poster](https://sc17.supercomputing.org/SC17%20Archive/tech_poster/poster_files/post190s2-file2.pdf) and [abstract](https://sc17.supercomputing.org/SC17%20Archive/tech_poster/poster_files/post190s2-file3.pdf)), SC’17: The International Conference for High Performance Computing, Networking, Storage, and Analysis, Denver (Colorado).

<b>[5]&#160;</b>[https://www.computer.org/csdl/proceedings-article/sc/2016/8815a981/12OmNCeaQ1D](https://www.computer.org/csdl/proceedings-article/sc/2016/8815a981/12OmNCeaQ1D): LIBXSMM: Accelerating Small Matrix Multiplications by Runtime Code Generation. SC'16: The International Conference for High Performance Computing, Networking, Storage and Analysis, Salt Lake City (Utah).

<b>[6]&#160;</b>[http://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/tech_poster_pages/post137.html](http://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/tech_poster_pages/post137.html): LIBXSMM: A High Performance Library for Small Matrix Multiplications ([poster](http://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/poster_files/post137s2-file2.pdf) and [abstract](http://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/poster_files/post137s2-file3.pdf)). SC'15: The International Conference for High Performance Computing, Networking, Storage and Analysis, Austin (Texas).

## Articles

<b>[1]&#160;</b>[https://www.nextplatform.com/2019/10/09/cloudy-supercomputers-join-the-hpc-petascale-club/](https://www.nextplatform.com/2019/10/09/cloudy-supercomputers-join-the-hpc-petascale-club/): Cloudy Supercomputers Join the HPC Petascale Club. An article written by Rob Farber, 2019. The article covers LIBXSMM in a separate section.

<b>[2]&#160;</b>[https://www.nextplatform.com/2019/06/26/counting-the-cost-of-scaling-hpc-applications/](https://www.nextplatform.com/2019/06/26/counting-the-cost-of-scaling-hpc-applications/): Counting The Cost Of Scaling HPC Applications. An article written by Timothy Prickett Morgan, 2019. This article is about CP2K Open Source Molecular Dynamics and not about LIBXSMM. However, LIBXSMM was key for application performance.

<b>[3]&#160;</b>[https://www.nextplatform.com/2019/06/26/counting-the-cost-of-scaling-hpc-applications/](https://www.nextplatform.com/2019/06/26/counting-the-cost-of-scaling-hpc-applications/): Azure Benchmarks HC-series Across Twenty-thousand Cores for HPC. An article written by John Russell, 2019. This article is about CP2K Open Source Molecular Dynamics and not about LIBXSMM. However, LIBXSMM was key for application performance.

<b>[4]&#160;</b>[https://software.intel.com/sites/default/files/parallel-universe-issue-34.pdf](https://software.intel.com/content/www/us/en/develop/download/parallel-universe-magazine-issue-34-october-2018.html): LIBXSMM: An Open Source-Based Inspiration for Hardware and Software Development at Intel ([pdf](https://software.intel.com/content/dam/develop/public/us/en/documents/parallel-universe-issue-34.pdf)). An article written by Hans Pabst, Greg Henry, and Alexander Heinecke, 2018.

<b>[5]&#160;</b>[https://medium.com/@rmfarber/libxsmm-brings-deep-learning-lessons-learned-to-many-hpc-applications-9143c6c93125](https://medium.com/@rmfarber/libxsmm-brings-deep-learning-lessons-learned-to-many-hpc-applications-9143c6c93125): LIBXSMM Brings Deep-learning "Lessons Learned" to Many HPC Applications. An article written by Rob Farber, 2018.

<b>[6]&#160;</b>[https://www.rdworldonline.com/largest-supercomputer-simulation-of-sumatra-andaman-earthquake/](https://www.rdworldonline.com/largest-supercomputer-simulation-of-sumatra-andaman-earthquake/): Largest Supercomputer Simulation of Sumatra-Andaman Earthquake. An article written by Linda Barney, 2018.

