# LIBXSMM

[![License](https://img.shields.io/badge/license-BSD3-blue.svg)](LICENSE.md) [![Travis CI](https://travis-ci.org/hfp/libxsmm.svg?branch=master "Master branch build status")](https://github.com/hfp/libxsmm/wiki/Status) [![Coverity](https://scan.coverity.com/projects/7405/badge.svg "Coverity Scan Build Status")](https://scan.coverity.com/projects/hfp-libxsmm) [![ReadtheDocs](https://readthedocs.org/projects/libxsmm/badge/?version=latest "Read the Docs")](https://libxsmm.readthedocs.io/)

LIBXSMM is a library for specialized dense and sparse matrix operations as well as for deep learning primitives such as small convolutions. The library is targeting Intel Architecture with Intel&#160;SSE, Intel&#160;AVX, Intel&#160;AVX2, Intel&#160;AVX&#8209;512 (with VNNI and Bfloat16). Code generation is mainly based on Just&#8209;In&#8209;Time (JIT) code specialization for compiler-independent performance (matrix multiplications, matrix transpose/copy, sparse functionality, and deep learning). LIBXSMM is suitable for "build once and deploy everywhere", i.e., no special target flags are needed to exploit the available performance. Supported GEMM datatypes are: `FP64`, `FP32`, `bfloat16`, `int16`, and `int8`.

For a list questions and answers, please also have a look at [https://github.com/hfp/libxsmm/wiki/Q&A](https://github.com/hfp/libxsmm/wiki/Q&A).

**Where to go for documentation?**

* **ReadtheDocs**: [main](https://libxsmm.readthedocs.io/) and [sample](https://libxsmm.readthedocs.io/libxsmm_samples/) documentation with full text search.
* **PDF**: [main](https://github.com/hfp/libxsmm/raw/master/documentation/libxsmm.pdf) documentation file, and separate [sample](https://github.com/hfp/libxsmm/raw/master/documentation/libxsmm_samples.pdf) documentation.
* **Articles**: [magazine article](https://software.intel.com/sites/default/files/parallel-universe-issue-34.pdf) incl. [sample code](https://github.com/hfp/libxsmm/tree/master/samples/magazine) (full list of [Articles](#articles)).

**<a name="what-is-a-small-matrix-multiplication"></a>What is a small matrix multiplication?** When characterizing the problem-size by using the M, N, and K parameters, a problem-size suitable for LIBXSMM falls approximately within *(M&#160;N&#160;K)<sup>1/3</sup>&#160;&lt;=&#160;64* (which illustrates that non-square matrices or even "tall and skinny" shapes are covered as well). The library is typically used to generate code up to the specified [threshold](documentation/libxsmm_tune.md#auto-dispatch). Raising the threshold may not only generate excessive amounts of code (due to unrolling in M or K dimension), but also miss to implement a tiling scheme to effectively utilize the cache hierarchy. For auto-dispatched problem-sizes above the configurable threshold (explicitly JIT'ted code is **not** subject to the threshold), LIBXSMM is falling back to BLAS. In terms of GEMM, the supported kernels are limited to *Alpha := 1*, *Beta := \{ 1, 0 \}*, and *TransA := 'N'*.

**<a name="what-is-a-small-convolution"></a>What is a small convolution?** In the last years, new workloads such as deep learning and more specifically convolutional neural networks (CNN) emerged and are pushing the limits of today's hardware. One of the expensive kernels is a small convolution with certain kernel sizes such that calculations in the frequency space is not the most efficient method when compared with direct convolutions. LIBXSMM's current support for convolutions aims for an easy to use invocation of small (direct) convolutions, which are intended for CNN training and classification.

**<a name="getting-started"></a><a name="hello-libxsmm"></a>Getting Started**: The following code is focused on a specific functionality but may be considered as "Hello LIBXSMM". Build the example with `cd /path/to/libxsmm; make STATIC=0` (shared library), save the code under `hello.cpp` (below) and compile with `g++ -I/path/to/libxsmm/include hello.cpp -L/path/to/libxsmm/lib -lxsmm -lblas -o hello` (GNU CCC), and finally execute with `LD_LIBRARY_PATH=/path/to/libxsmm/lib LIBXSMM_VERBOSE=2 ./hello`.

```cpp
#include <libxsmm.h>
#include <vector>
int main(int argc, char* argv[])
{
  typedef double value_type;
  int batchsize = 1000, m = 13, n = 5, k = 7;
  std::vector<value_type> a(batchsize*m*k), b(batchsize*k*n), c(m*n, 0);
  /* C/C++ and Fortran interfaces are available */
  typedef libxsmm_mmfunction<value_type> kernel_type;
  /* generates and dispatches a matrix multiplication kernel (C++ functor) */
  kernel_type kernel(LIBXSMM_GEMM_FLAG_NONE, m,n,k, 1.0/*alpha*/, 1.0/*beta*/);
  assert(kernel);
  for (int i = 0; i < batchsize; ++i) { /* initialize input */
    a[i*m*k] = static_cast<value_type>(1) / (i % 25);
    b[i*k*n] = static_cast<value_type>(7) / (i % 75);
  }
  /* kernel multiplies and accumulates matrix products: C += Ai * Bi */
  for (int i = 0; i < batchsize; ++i) kernel(&a[i*m*k], &b[i*k*n], &c[0]);
}
```

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

This domain (DL) is detailed by a separate [document](documentation/libxsmm_dl.md). It may be inspiring to have a look at the light-weight GxM framework, which uses LIBXSMM for end-to-end Deep Learning.

### Service Functions

For convenient operation of the library and to ease integration, some service routines are available. These routines may not belong to the core functionality of LIBXSMM (SMM or DNN domain), but users are encouraged to use this domain (AUX). There are two categories: (1)&#160;routines which are available for C and FORTRAN, and (2)&#160;routines that are only available per C interface.

The [service function domain (AUX)](documentation/libxsmm_aux.md) contains routines for:

* [Getting and setting the target architecture](documentation/libxsmm_aux.md#getting-and-setting-the-target-architecture)
* [Getting and setting the verbosity](documentation/libxsmm_aux.md#getting-and-setting-the-verbosity)
* [Measuring time durations (timer)](documentation/libxsmm_aux.md#timer-facility)
* [Loading and storing data (I/O)](documentation/libxsmm_aux.md#meta-image-file-io)
* [Allocating memory](documentation/libxsmm_aux.md#memory-allocation)

### Backend<a name="jit-backend"></a>

More information about the JIT-backend and the code generator can be found in a separate [document](documentation/libxsmm_be.md), which also includes information about LIBXSMM's stand-alone <a name="generator-driver"></a>[generator-driver](documentation/libxsmm_be.md#generator-driver) programs.

## Build Instructions

### Overview

The main interface file is *generated*, and it is therefore **not** stored in the code repository. Instead, one may have a look at the code generation template files for [C/C++](https://github.com/hfp/libxsmm/blob/master/src/template/libxsmm.h#L36) and [FORTRAN](https://github.com/hfp/libxsmm/blob/master/src/template/libxsmm.f#L32). There are two ways prepared to build and use LIBXSMM:

* [Classic Library (ABI)](#classic-library-abi) and [Link Instructions](#link-instructions) (C/C++ and FORTRAN)
* [Header-Only](#header-only) (C and C++)

**NOTE**: LIBXSMM is available as prebuilt package for Fedora/RedHat/CentOS, Ubuntu, and FreeBSD. Further, LIBXSMM can be installed with the [Spack Package Manager](http://computation.llnl.gov/projects/spack-hpc-package-manager) or per [EasyBuild+EasyConfig](https://github.com/easybuilders).

### Classic Library (ABI)

The build system relies on GNU&#160;Make (typically associated with the `make` command, but e.g. FreeBSD is calling it `gmake`). The build can be customized by using key&#8209;value pairs. Key&#8209;value pairs can be supplied in two ways: (1)&#160;after the "make" command, or (2)&#160;prior to the "make" command (`env`) which is effectively the same as exporting the key&#8209;value pair as an environment variable (`export`, or `setenv`). Both methods can be mixed (the second method may require make's `-e` flag).

**NOTE**: By default, C/C++ and FORTRAN compilers are needed (some sample code is written in C++). Beside of specifying the compilers (`make CXX=g++ CC=gcc FC=gfortran` and maybe `AR=ar`), the need for a FORTRAN compiler can be relaxed (`make FC=` or `make FORTRAN=0`). The latter affects the availability of the MODule file and the corresponding 'libxsmmf' library (the interface 'libxsmm.f' is still generated).

The build system considers a set of given key-value pairs as a single unique build and triggers a rebuild for a distinct set of flags. For more advanced builds or additional background, please consult the section about [Customization](documentation/libxsmm_tune.md). To generate the interface of the library inside of the 'include' directory and to build the static library (by default, STATIC=1 is activated). Run any (or both) of the following command(s):

```bash
make STATIC=0
make
```

On CRAY systems, the CRAY Compiling Environment (CCE) should be used regardless of utilizing the CRAY compiler, the Intel Compiler, or the GNU&#160;Compiler Collection (GCC). The CCE is eventually suppressing to build shared libraries (STATIC=0). In any case, (1)&#160;switch to the desired compiler (module load/switch), and (2)&#160;rely on:

```bash
make CXX=CC CC=cc FC=ftn
```

A variety of build environments is out-of-the-box compatible, see [https://github.com/hfp/libxsmm/wiki/Compatibility](https://github.com/hfp/libxsmm/wiki/Compatibility). If the build process is not successful, it may help to avoid advanced GCC flags. This is useful with a tool chain, which pretends to be GCC-compatible (and is treated as such) but fails to consume the afore mentioned flags:

```bash
make COMPATIBLE=1
```

<a name="outdated-binutils"></a>In case of outdated Binutils, compilation can fail to assemble code when building the library (this has nothing to do with JIT-generated code and it does not affect how JIT-code is targeting the system). In contrast to the default (`INTRINSICS=2`), `INTRINSICS=1` enables to statically depend on the desired target, e.g., `AVX=3 MIC=0`, or `AVX=2` (if the target is omitted, the default target is used). Try to work around failing compilation with:

```bash
make INTRINSICS=1
```

To test and validate a build, please consult [https://github.com/hfp/libxsmm/wiki/Validation](https://github.com/hfp/libxsmm/wiki/Validation). To run some basic sanity checks, remember that each set of given key-value pairs represents a different build (and test):

```bash
make STATIC=0 tests
```

To remove intermediate files, or to remove all generated files and folders (including the interface and the library archives), run one of the make-targets below. An additional distclean-target recursively cleans the entire tree (after version&#160;1.9).

```bash
make clean
make realclean
```

<a name="fortran"></a>FORTRAN code can make use of LIBXSMM:

* By using the module and linking with 'libxsmmf', 'libxsmm', and (optionally) 'libxsmmext',
* By including 'libxsmm.f' and linking with 'libxsmm', and (optionally) 'libxsmmext', or
* By (implicitly) calling a SUBROUTINE and linking with 'libxsmm', and (optionally) 'libxsmmext'.

**Note**: Using the Fortran module or including the interface, requires at least a Fortran&#160;2003 compiler (F2K3). FORTRAN&#160;77 compatibility is only implicitly available (no interface), and the available subset of routines is documented in 'libxsmm.f' and marked with [comments](https://github.com/hfp/libxsmm/search?q=implementation+provided+for+Fortran+77+compatibility) (part of the implementation).

### Header-Only

Version&#160;1.4.4 introduced support for "header-only" usage in C and C++. By only including 'libxsmm_source.h' allows to get around building the library. However, this gives up on a clearly defined application binary interface (ABI). An ABI may allow for hot-fixes after deploying an application (when relying on the shared library form), and it may also ensure to only rely on the public interface of LIBXSMM. In contrast, the header-only form not only exposes the internal implementation of LIBXSMM but can also increase the turnaround time during development of an application (due to longer compilation times). The header file is intentionally named "libxsmm_**source**.h" since this header file relies on the [src](https://github.com/hfp/libxsmm/tree/master/src) directory (with the implications as noted earlier).

To use the header-only form, 'libxsmm_source.h' needs to be *generated*. The build target shown below ('header-only') has been introduced in LIBXSMM&#160;1.6.2 but `make cheader` can be used alternatively (or must be used instead in case of earlier versions). Generating the C interface is necessary since the library must be configured (see [configuration](https://github.com/hfp/libxsmm/blob/master/src/template/libxsmm_config.h) template).

```bash
make header-only
```

**NOTE**: building an application applies the same build settings to LIBXSMM! For instance, to omit debug code inside of LIBXSMM `NDEBUG` must be defined (`-DNDEBUG`).

## Link Instructions

Using the [classic ABI](#classic-library-abi) (including [Fortran](#fortran) code), requires linking LIBXSMM against the application. The library is agnostic with respect to the threading-runtime, and therefore an application is free to use any threading runtime (e.g., OpenMP). The library is also thread-safe, and multiple application threads can call LIBXSMM's routines concurrently. Enabling OpenMP for LIBXSMM's main library is supported as well (OMP=1), and mostly affects the synchronization primitives used inside of the library. All of the "omp" functionality (function postfix) is served by the 'libxsmmext' library, which is automatically built with OpenMP enabled. When using this "omp" functionality, 'libxsmmext' needs to be present at the link line.

<a name="table-of-libraries"></a>Library | Purpose
:-------------|---------
libxsmm       | Thread-safe core functions (same routine can be called concurrently). Contains routines that can take a thread-ID and the number of library-external threads.
libxsmmf      | Necessary when using the Fortran MODule but not when including `libxsmm.f` or relying on implicit interfaces ([Fortran&#160;77](https://github.com/hfp/libxsmm/search?q=implementation+provided+for+Fortran+77+compatibility)).
libxsmmext    | Provides library-internal OpenMP-threaded functions carrying the `omp` postfix when compared to function name names of the core library.
libxsmmnoblas | Supplies faked symbols for `dgemm` (and others) and thereby removes the need to link against a LAPACK/BLAS library.

<a name="pkg-config"></a>To ease linking with LIBXSMM, `pkg-config` can be used. For example:

```bash
export PKG_CONFIG_PATH=/path/to/libxsmm/lib
pkg-config libxsmm --libs
```

Similarly, an application is free to choose any BLAS or LAPACK library (if the link model available on the OS supports this), and therefore linking GEMM routines when linking LIBXSMM itself (by supplying BLAS=1&#124;2) may prevent a user from making this decision at the time of linking the actual application. To use LIBXSMM without GEMM-related functionality, any BLAS-dependency can be removed in two ways: (1)&#160;building a special library with `make BLAS=0`, or (2)&#160;linking the application against the 'libxsmmnoblas' library. If an application however uses BLAS already, the [Call Wrapper](documentation/libxsmm_mm.md#call-wrapper) can be used to intercept existing BLAS calls (and to rely on LIBXSMM instead).

**NOTE**: LIBXSMM does not support to dynamically link 'libxsmm' or 'libxsmmext' ("so"), when BLAS is linked statically ("a"). If BLAS is linked statically, the static version of LIBXSMM must be used!

### Installation

There are two main mechanisms to install LIBXSMM (both mechanisms can be combined): (1)&#160;building the library in an out&#8209;of&#8209;tree fashion, and (2)&#160;installing into a certain location. <a name="install-build"></a>Building in an out&#8209;of&#8209;tree fashion looks like:

```bash
cd libxsmm-install
make -f /path/to/libxsmm/Makefile
```

<a name="install-prefix"></a>Installation into a specific location looks like (`PREFIX` or `DESTDIR`):

```bash
make MNK="1 2 3 4 5" PREFIX=/path/to/libxsmm-install install
```

<a name="install-destdir"></a>Both `PREFIX` and `DESTDIR` are equivalent and can be relative or absolute paths. An installation can be repeated for different locations without triggering a rebuild. The prefix directory *inside* of each of the [package configuration files](#pkg-config) is set to where LIBXSMM is built (staging folder) unless `PREFIX` or `DESTDIR` is specified. The effect of `PREFIX` (or `DESTDIR`) with respect to the pkg-config files is independent of whether the install-target is invoked or not (make).

Further, performing `make install-minimal` omits the documentation (default: 'PREFIX/share/libxsmm'). Moreover, PINCDIR, POUTDIR, PBINDIR, and PDOCDIR allow to customize the locations underneath of the PREFIX location. To build a general package for an unpredictable audience (Linux distribution, or similar), it is advised to not over-specify or customize the build step, i.e., JIT, SSE, AVX, OMP, BLAS, etc. should not be used. The following is building and installing a complete set of libraries where the generated interface matches both the static and the shared libraries:

```bash
make PREFIX=/path/to/libxsmm-install STATIC=0 install
make PREFIX=/path/to/libxsmm-install install
```

## Runtime Control<a name="running"></a>

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

The tables are distinct between single-precision and double-precision, but either table is pruned if all counters are zero. If both tables are pruned, the library shows the code path which would have been used for JIT'ting the code: `LIBXSMM_TARGET=hsw` (otherwise the code path is shown in the table's header). The actual counters are collected for three buckets: small kernels (MNK<sup>1/3</sup>&#160;&lt;=&#160;13), medium-sized kernels (13&#160;&lt;&#160;MNK<sup>1/3</sup>&#160;&lt;=&#160;23), and larger kernels (23&#160;&lt;&#160;MNK<sup>1/3</sup>&#160;&lt;=&#160;64; the actual upper bound depends on LIBXSMM_MAX_MNK as selected at compile-time). Keep in mind, that "larger" is supposedly still small in terms of arithmetic intensity (which grows linearly with the kernel size). Unfortunately, the arithmetic intensity depends on the way a kernel is used (which operands are loaded/stored into main memory) and it is not performance-neutral to collect this information.

The TRY counter represents all attempts to register statically generated kernels, and all attempts to dynamically generate and register kernels. The TRY counter includes rejected JIT requests due to unsupported GEMM arguments. The JIT and STA counters distinct the successful cases of the afore mentioned event (TRY) into dynamically (JIT) and statically (STA) generated code. In case the capacity (O(*n*)&#160;=&#160;10<sup>5</sup>) of the code registry is exhausted, no more kernels can be registered although further attempts are not prevented. Registering many kernels (O(*n*)&#160;=&#160;10<sup>3</sup>) may ramp the number of hash key collisions (COL), which can degrade performance. The latter is prevented if the small thread-local cache is utilized effectively.

Since explicitly JIT-generated code (`libxsmm_?mmdispatch`) does not fall under the THRESHOLD criterion, the above table is extended by one line if large kernels have been requested. This indicates a missing threshold-criterion (customized dispatch), or asks for cache-blocking the matrix multiplication. The latter is already implemented by LIBXSMM's "medium-sized" GEMM routines (`libxsmm_?gemm_omp`), which perform a tiled multiplication. Setting a verbosity level of at least two summarizes the number of registered JIT-generated kernels, which includes the total size and counters for GEMM, MCOPY (matrix copy), and TCOPY (matrix transpose) kernels.

```bash
Registry: 20 MB (gemm=0 mcopy=14 tcopy=0)
```

If the call-wrapper is used, an additional runtime statistic becomes available (see [Call Wrapper](documentation/libxsmm_mm.md#call-wrapper)).

<a name="objdump"></a>**NOTE**: Setting LIBXSMM_VERBOSE to a negative value will binary-dump each generated JIT kernel to a file with each file being named like the function name shown in [Intel&#160;VTune](documentation/libxsmm_prof.md#intelvtuneamplifier). Disassembly of the raw binary files can be accomplished by:

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

<a name="profiling"></a>Profiling an application, which uses LIBXSMM's JIT-code is well-supported. The library supports Intel&#160;VTune&#160;Amplifier and Linux&#160;perf. Details are given on how to include profiler support, and how to run the application.

* [Profiling using Intel&#160;VTune&#160;Amplifier](documentation/libxsmm_prof.md#intelvtuneamplifier)
* [Profiling using Linux&#160;perf](documentation/libxsmm_prof.md#linuxperf)

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

**\[1]&#160;[https://cp2k.org/](https://cp2k.org/)**: Open Source Molecular Dynamics and the [DBCSR library](https://github.com/cp2k/dbcsr), which processes batches of small matrix multiplications. The batches originate from a distributed block-sparse matrix with problem-specific small matrices. Starting with [CP2K&#160;3.0](https://www.cp2k.org/version_history), LIBXSMM can substitute CP2K's 'libsmm' library.

**\[2]&#160;[https://github.com/SeisSol/SeisSol/](https://github.com/SeisSol/SeisSol/)**: SeisSol is one of the leading codes for earthquake scenarios, for simulating dynamic rupture processes. LIBXSMM provides highly optimized assembly kernels which form the computational back-bone of SeisSol (see [https://github.com/TUM-I5/seissol_kernels/](https://github.com/TUM-I5/seissol_kernels/).

**\[3]&#160;[https://github.com/NekBox/NekBox](https://github.com/NekBox/NekBox)**: NekBox is a highly scalable and portable spectral element code, which is inspired by the [Nek5000](https://nek5000.mcs.anl.gov/) code. NekBox is specialized for box geometries and intended to prototype new methods as well as to leverage FORTRAN beyond the FORTRAN&#160;77 standard. LIBXSMM can be used to substitute the [MXM_STD](https://github.com/Nek5000/NekBox/blob/box/mxm_std.F90) code. Please also note LIBXSMM's [NekBox reproducer](https://github.com/hfp/libxsmm/tree/master/samples/nek#nek-sample-collection).

**\[4]&#160;[https://github.com/Nek5000/Nek5000](https://github.com/Nek5000/Nek5000)**: Nek5000 is the open-source, highly-scalable, always-portable spectral element code from [https://nek5000.mcs.anl.gov/](https://nek5000.mcs.anl.gov/). The development branch of the Nek5000 code [incorporates](https://github.com/Nek5000/Nek5000/blob/master/core/mxm_wrapper.f) LIBXSMM.

**\[5]&#160;[http://pyfr.org/](http://pyfr.org/)**: PyFR is an open-source Python based framework for solving advection-diffusion type problems on streaming architectures by using the flux reconstruction approach. PyFR&#160;1.6.0 optionally [incorporates LIBXSMM](http://pyfr.org/user_guide.php) as a matrix multiplication provider for the OpenMP backend. Please also note LIBXSMM's [PyFR-related code sample](https://github.com/hfp/libxsmm/tree/master/samples/pyfr).

**\[6]&#160;[http://dial3343.org/about/](http://dial3343.org/about/)**: The Extreme-scale Discontinuous Galerkin Environment (EDGE) is a solver for hyperbolic partial differential equations with emphasis on seismic simulations. The EDGE [source code](https://github.com/3343/edge) optionally relies on LIBXSMM, but for high performance LIBXSMM's kernels are highly recommended.

**\[7]&#160;[https://sxs-collaboration.github.io/spectre/](https://sxs-collaboration.github.io/spectre/)**: SpECTRE is an open-source code for multi-scale, multi-physics problems in astrophysics and gravitational physics which runs at petascale and is designed for exascale computers. In the future, SpECTRE may be applied to problems across discipline boundaries in fluid dynamics, geoscience, plasma physics, nuclear physics, and engineering.

**\[8]&#160;[https://ceed.exascaleproject.org/ceed-code/](https://ceed.exascaleproject.org/ceed-code/)**: The Center for Efficient Exascale Discretizations (CEED) is building on the efforts of the Nek5000, MFEM, MAGMA, OCCA and PETSc projects to develop application program interfaces (APIs), both at high-level and at low-level to enable applications to take advantage of high-order methods. The CEED low-level API, [libCEED](https://ceed.exascaleproject.org/libceed/) uses LIBXSMM as a [backend](https://github.com/CEED/libCEED#backends) for high performance on CPUs.

### Machine Learning (ML)

**\[9]&#160;[https://github.com/baidu-research/DeepBench](https://github.com/baidu-research/DeepBench#deepbench)**: The primary purpose of DeepBench is to benchmark operations that are important to deep learning on different hardware platforms. LIBXSMM's DNN primitives have been [incorporated into DeepBench](https://github.com/baidu-research/DeepBench/tree/master/code/intel/convolution/libxsmm_conv) to demonstrate an increased performance of deep learning on Intel hardware.

**\[10]&#160;[https://www.tensorflow.org/](https://tensorflow.org/)**: TensorFlow&trade; is an open source software library for numerical computation using data flow graphs. TensorFlow was originally developed by researchers and engineers working on the Google Brain Team for the purposes of conducting machine learning and deep neural networks research. LIBXSMM can be [used](documentation/tensorflow.md#tensorflow-with-libxsmm) to increase the performance of TensorFlow on Intel hardware.

**\[11]&#160;[https://github.com/IntelLabs/SkimCaffe](https://github.com/IntelLabs/SkimCaffe#skimcaffe-specific-description)**: SkimCaffe from Intel Labs is a Caffe branch for training of sparse CNNs, which provide 80-95% sparsity in convolutions and fully-connected layers. LIBXSMM's SPMDM domain (SParseMatrix-DenseMatrix multiplication) evolved from SkimCaffe, and since then LIBXSMM implements the sparse operations in SkimCaffe.

### Automated Driving (AD)

**\[12]&#160;[https://software.seek.intel.com/accelerating-eigen-math-library](https://software.seek.intel.com/accelerating-eigen-math-library)**: Accelerating The Eigen Math Library for Automated Driving Workloads: The Need for Speed in Kalman Filtering. An article in [Issue&#160;31](https://software.intel.com/sites/default/files/managed/4f/73/parallel-universe-issue-31.pdf) of The Parallel Universe magazine.

## References

**\[1]&#160;[https://sc19.supercomputing.org/proceedings/tech_poster/tech_poster_pages/rpost244.html](https://sc19.supercomputing.org/proceedings/tech_poster/tech_poster_pages/rpost244.html)**: High-Performance Deep Learning via a Single Building Block ([poster](https://sc19.supercomputing.org/proceedings/tech_poster/poster_files/rpost244s2-file2.pdf) and [abstract](https://sc19.supercomputing.org/proceedings/tech_poster/poster_files/rpost244s2-file3.pdf)), SC’19: The International Conference for High Performance Computing, Networking, Storage, and Analysis, Denver (Colorado).

**\[2]&#160;[https://sc18.supercomputing.org/presentation/?id=pap322&sess=sess190](https://sc18.supercomputing.org/presentation/?id=pap322&sess=sess190)**: Anatomy of High-Performance Deep Learning Convolutions on SIMD Architectures ([arXiv:1808.05567v2 paper](https://arxiv.org/pdf/1808.05567v2)). SC'18: The International Conference for High Performance Computing, Networking, Storage, and Analysis, Dallas (Texas).

**\[3]&#160;[http://sc17.supercomputing.org/SC17%20Archive/tech_poster/tech_poster_pages/post190.html](http://sc17.supercomputing.org/SC17%20Archive/tech_poster/tech_poster_pages/post190.html)**: Understanding the Performance of Small Convolution Operations for CNN on Intel Architecture ([poster](http://sc17.supercomputing.org/SC17%20Archive/tech_poster/poster_files/post190s2-file2.pdf) and [abstract](http://sc17.supercomputing.org/SC17%20Archive/tech_poster/poster_files/post190s2-file3.pdf)), SC’17: The International Conference for High Performance Computing, Networking, Storage, and Analysis, Denver (Colorado).

**\[4]&#160;[http://sc16.supercomputing.org/presentation/?id=pap364&sess=sess153](http://sc16.supercomputing.org/presentation/?id=pap364&sess=sess153)**: LIBXSMM: Accelerating Small Matrix Multiplications by Runtime Code Generation ([paper](http://www.computer.org/csdl/proceedings/sc/2016/8815/00/8815a981.pdf)). SC'16: The International Conference for High Performance Computing, Networking, Storage and Analysis, Salt Lake City (Utah).

**\[5]&#160;[http://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/tech_poster_pages/post137.html](http://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/tech_poster_pages/post137.html)**: LIBXSMM: A High Performance Library for Small Matrix Multiplications ([poster](http://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/poster_files/post137s2-file2.pdf) and [abstract](http://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/poster_files/post137s2-file3.pdf)). SC'15: The International Conference for High Performance Computing, Networking, Storage and Analysis, Austin (Texas).

## Articles

**\[1]&#160;[https://www.nextplatform.com/2019/10/09/cloudy-supercomputers-join-the-hpc-petascale-club/](https://www.nextplatform.com/2019/10/09/cloudy-supercomputers-join-the-hpc-petascale-club/)**: Cloudy Supercomputers Join the HPC Petascale Club. An article written by Rob Farber, 2019. The article covers LIBXSMM in a separate section.

**\[2]&#160;[https://www.nextplatform.com/2019/06/26/counting-the-cost-of-scaling-hpc-applications/](https://www.nextplatform.com/2019/06/26/counting-the-cost-of-scaling-hpc-applications/)**: Counting The Cost Of Scaling HPC Applications. An article written by Timothy Prickett Morgan, 2019. This article is about CP2K Open Source Molecular Dynamics and not about LIBXSMM in particular. However, LIBXSMM was key for application performance.

**\[3]&#160;[https://www.nextplatform.com/2019/06/26/counting-the-cost-of-scaling-hpc-applications/](https://www.nextplatform.com/2019/06/26/counting-the-cost-of-scaling-hpc-applications/)**: Azure Benchmarks HC-series Across Twentythousand Cores for HPC. An article written by John Russell, 2019. This article is about CP2K Open Source Molecular Dynamics and not about LIBXSMM in particular. However, LIBXSMM was key for application performance.

**\[4]&#160;[https://software.intel.com/sites/default/files/parallel-universe-issue-34.pdf](https://software.intel.com/en-us/download/parallel-universe-magazine-issue-34-october-2018)**: LIBXSMM: An Open Source-Based Inspiration for Hardware and Software Development at Intel. An article written by Hans Pabst, Greg Henry, and Alexander Heinecke, 2018.

**\[5]&#160;[https://medium.com/@rmfarber/libxsmm-brings-deep-learning-lessons-learned-to-many-hpc-applications-9143c6c93125](https://medium.com/@rmfarber/libxsmm-brings-deep-learning-lessons-learned-to-many-hpc-applications-9143c6c93125)**: LIBXSMM Brings Deep-learning "Lessons Learned" to Many HPC Applications. An article written by Rob Farber, 2018.

**\[6]&#160;[https://www.rdmag.com/article/2018/02/largest-supercomputer-simulation-sumatra-andaman-earthquake](https://www.rdmag.com/article/2018/02/largest-supercomputer-simulation-sumatra-andaman-earthquake)**: Largest Supercomputer Simulation of Sumatra-Andaman Earthquake. An article written by Linda Barney, 2018.

**\[7]&#160;[https://software.intel.com/en-us/articles/intel-xeon-phi-delivers-competitive-performance-for-deep-learning-and-getting-better-fast](https://software.intel.com/en-us/articles/intel-xeon-phi-delivers-competitive-performance-for-deep-learning-and-getting-better-fast)**: Intel Xeon&#160;Phi Delivers Competitive Performance For Deep Learning - And Getting Better Fast. Article mentioning LIBXSMM's performance of convolution kernels with [DeepBench](https://github.com/baidu-research/DeepBench/tree/master/code/intel/convolution/libxsmm_conv). Intel Corporation, 2016.

