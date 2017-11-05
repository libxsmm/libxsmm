# LIBXSMM



LIBXSMM is a library for small dense and small sparse matrix-matrix multiplications as well as for deep learning primitives such as small convolutions targeting Intel Architecture. Small matrix multiplication kernels are generated for the following instruction set extensions: Intel&#160;SSE, Intel&#160;AVX, Intel&#160;AVX2, IMCI (KNCni) for Intel&#160;Xeon&#160;Phi coprocessors ("KNC"), and Intel&#160;AVX&#8209;512 as found in the Intel&#160;Xeon&#160;Phi processor family&#160;(Knights Landing "KNL", Knights Mill "KNM") and Intel&#160;Xeon processors (Skylake-SP "SKX"). Historically small matrix multiplications were only optimized for the Intel&#160;Many Integrated Core Architecture "MIC") using intrinsic functions, meanwhile optimized assembly code is targeting all afore mentioned instruction set extensions (static code generation), and Just&#8209;In&#8209;Time (JIT) code generation is targeting Intel&#160;AVX and beyond. Optimized code for small convolutions is JIT-generated for Intel&#160;AVX2 and Intel&#160;AVX&#8209;512.

**Where to go for documentation?**

* **ReadtheDocs**: [main](http://libxsmm.readthedocs.io/) and [sample](http://libxsmm.readthedocs.io/libxsmm_samples/) documentation with full text search.
* **PDF**: [main](https://github.com/hfp/libxsmm/raw/master/documentation/libxsmm.pdf) documentation file, and separate [sample](https://github.com/hfp/libxsmm/raw/master/documentation/libxsmm_samples.pdf) documentation.

**<a name="what-is-a-small-matrix-multiplication"></a>What is a small matrix multiplication?** When characterizing the problem-size using the M, N, and K parameters, a problem-size suitable for LIBXSMM falls approximately within *(M&#160;N&#160;K)<sup>1/3</sup>&#160;&lt;=&#160;128* (which illustrates that non-square matrices or even "tall and skinny" shapes are covered as well). The library is typically used to generate code up to the specified [threshold](libxsmm_tune.md#auto-dispatch). Raising the threshold may not only generate excessive amounts of code (due to unrolling in M or K dimension), but also miss to implement a tiling scheme to effectively utilize the cache hierarchy. For auto-dispatched problem-sizes above the configurable threshold (explicitly JIT'ted code is **not** subject to the threshold), LIBXSMM is falling back to BLAS. In terms of GEMM, the supported kernels are limited to *Alpha := 1*, *Beta := \{ 1, 0 \}*, *TransA := 'N'*, and *TransB = 'N'*.

**<a name="what-is-a-small-convolution"></a>What is a small convolution?** In the last years, new workloads such as deep learning and more specifically convolutional neural networks (CNN) emerged, and are pushing the limits of today's hardware. One of the expensive kernels is a small convolution with certain kernel sizes (3, 5, or 7) such that calculations in the frequency space is not the most efficient method when compared with direct convolutions. LIBXSMM's current support for convolutions aims for an easy to use invocation of small (direct) convolutions, which are intended for CNN training and classification.

For more questions and answers, please have a look at [https://github.com/hfp/libxsmm/wiki/Q&A](https://github.com/hfp/libxsmm/wiki/Q&A).

Documented functionality and available domains:

* MM: [Matrix Multiplication](#matrix-multiplication)
* DNN: [Deep Neural Networks](#deep-neural-networks)
* AUX: [Service Functions](#service-functions)
* PERF: [Performance](#performance)
* BE: [Backend](#jit-backend)

For additional functionality, please have a look at [https://github.com/hfp/libxsmm/tree/master/include](https://github.com/hfp/libxsmm/tree/master/include).

## Build Instructions

### Overview

The main interface file is *generated*, and it is therefore **not** stored in the code repository. Instead, one may have a look at the code generation template files for [C/C++](https://github.com/hfp/libxsmm/blob/master/src/template/libxsmm.h#L36) and [FORTRAN](https://github.com/hfp/libxsmm/blob/master/src/template/libxsmm.f#L32). The main interface consists of the [general interface](#general-interface) as well as the [matrix multiplication](#matrix-multiplication) interface.

There are two ways to incorporate LIBXSMM into an application:

* [Classic Library (ABI)](#classic-library-abi) and [Link Instructions](#link-instructions)
* [Header-Only](#header-only)

### Classic Library (ABI)

The build system relies on GNU Make (typically associated with the `make` command, but e.g. FreeBSD is calling it `gmake`). The build can be customized by using key&#8209;value pairs. Key&#8209;value pairs can be supplied in two ways: (1)&#160;after the "make" command, or (2)&#160;prior to the "make" command (`env`) which is effectively the same as exporting the key&#8209;value pair as an environment variable (`export`, or `setenv`). Both methods can be mixed, however the second method may require the `-e` flag. Please note that the CXX, CC, and FC keys are considered in any case.

To generate the interface of the library inside of the 'include' directory and to build the static library (by default, STATIC=1 is activated), simply run the following command:

```bash
make
```

On CRAY systems, the CRAY Compiling Environment (CCE) should be used regardless of using the CRAY compiler, the Intel Compiler, or the GNU Compiler Collection (GCC). The latter is achieved by switching the programming environment to the desired compiler, but always relying on:

```bash
make CXX=CC CC=cc FC=ftn
```

If the build process is not successful, it may help to avoid more advanced GCC flags. This is useful with a tool chain, which pretends to be GCC-compatible (or is treated as such) but fails to consume the afore mentioned flags. In such a case, one may raise the compatibility:

```bash
make COMPATIBLE=1
```

By default, only the non-coprocessor targets are built (OFFLOAD=0 and KNC=0). In general, the subfolders of the 'lib' directory are separating the build targets where the 'mic' folder is containing the native library (KNC=1) targeting the Intel&#160;Xeon&#160;Phi coprocessor ("KNC"), and the 'intel64' folder is storing either the hybrid archive made of CPU and coprocessor code (OFFLOAD=1), or an archive which is only containing the CPU code. By default, an OFFLOAD=1 implies KNC=1.

To remove intermediate files, or to remove all generated files and folders (including the interface and the library archives), run one of the following commands:

```bash
make clean
make realclean
```

By default, LIBXSMM uses the [JIT backend](#jit-backend) which is automatically building optimized code. Matrix multiplication kernels can be also statically specialized at compile-time of the library (M, N, and K values), which is documented in the [Customization](libxsmm_tune.md) section. Functionality of LIBXSMM, which is unrelated to GEMM can be used without introducing a dependency to BLAS. This can be achieved in two ways: (1)&#160;building a special library with `make BLAS=0`, or (2)&#160;linking the application against the 'libxsmmnoblas' library.

**NOTE**: By default, C/C++ and FORTRAN compilers are needed (some sample code is written in C++). Beside of specifying the compilers (`make CXX=g++ CC=gcc FC=gfortran` and maybe `AR=ar`), the need for a FORTRAN compiler can be relaxed (`make FC=` or `make FORTRAN=0`). The latter affects the availability of the MODule file and the corresponding 'libxsmmf' library (the interface 'libxsmm.f' is still generated). FORTRAN code can make use of LIBXSMM in three ways:

* By relying on the module file, and by linking against 'libxsmmf', 'libxsmm', and (optionally) 'libxsmmext',
* By including the interface 'libxsmm.f' and linking against 'libxsmm', and (optionally) 'libxsmmext', or
* By optionally declaring a SUBROUTINE, or by simply calling a SUBROUTINE (FORTRAN&#160;77).

A FORTRAN&#160;77 interface is implicitly available and documented in 'libxsmm.f' (comments). It can be useful to have a look at the [implementation of the FORTRAN&#160;77 interface](https://github.com/hfp/libxsmm/search?q=implementation+provided+for+Fortran+77+compatibility).

### Link Instructions

The library is agnostic with respect to the threading-runtime, and therefore an application is free to use any threading runtime (e.g., OpenMP). The library is also thread-safe, and multiple application threads can call LIBXSMM's routines concurrently. Enabling OpenMP for LIBXSMM's main library is supported as well (OMP=1), and mostly affects the synchronization primitives used inside of the library. All of the "omp" functionality (function postfix) is served by the 'libxsmmext' library, which is automatically built with OpenMP enabled. When using this "omp" functionality, 'libxsmmext' needs to be present at the link line.

Similarly, an application is free to choose any BLAS or LAPACK library (if the link model available on the OS supports this), and therefore linking GEMM routines when linking LIBXSMM itself (by supplying BLAS=1&#124;2) may prevent a user from making this decision at the time of linking the actual application. For no application code changes, the [Call Wrapper](libxsmm_mm.md#call-wrapper) can intercept existing BLAS calls and use LIBXSMM instead.

**NOTE**: LIBXSMM does not support to dynamically link 'libxsmm' or 'libxsmmext' ("so"), when BLAS is linked statically ("a"). If BLAS is linked statically, the static version of LIBXSMM must be used!

### Header-Only

Version&#160;1.4.4 introduced support for "header-only" usage in C and C++. By only including 'libxsmm_source.h' allows to get around building the library. However, this gives up on a clearly defined application binary interface (ABI). An ABI may allow for hot-fixes after deploying an application (when relying on the shared library form), and it may also ensure to only rely on the public interface of LIBXSMM. In contrast, the header-only form not only exposes the internal implementation of LIBXSMM but it can also reduce the turnaround time during development of an application (due to longer compilation times). The header file is intentionally named "libxsmm_**source**.h" since this header file relies on the [src](https://github.com/hfp/libxsmm/tree/master/src) directory (with the implications as noted earlier).

To use the header-only form, 'libxsmm_source.h' needs to be *generated*. The build target shown below ('header-only') has been introduced in LIBXSMM&#160;1.6.2, but `make cheader` can be used alternatively (or must be used instead in case of earlier versions). Generating the C interface is necessary since the library must be configured (see [configuration](https://github.com/hfp/libxsmm/blob/master/src/template/libxsmm_config.h) template).

```bash
make header-only
```

**NOTE**: Differences between C and C++ makes a header-only implementation (which is portable between both languages) considerably "techy". Mixing C and C++ translation units (which rely on the header-only form of the library) is **not** supported. Also remember: to build an application now shares the same build settings with LIBXSMM! This is important not only to omit debug code inside of LIBXSMM (use `-DNDEBUG`).

### Installation

Installing LIBXSMM makes possibly the most sense when combining the JIT backend ([enabled by default](libxsmm_be.md)) with a collection of statically generated SSE kernels (by specifying M, N, K, or MNK). If the JIT backend is not disabled, statically generated kernels are only registered for dispatch if the CPUID flags at runtime are not supporting a more specific instruction set extension (code path). Since the JIT backend does not support or generate SSE code by itself, the library is compiled by selecting SSE code generation if not specified otherwise (AVX=1&#124;2&#124;3, or with SSE=0 falling back to an "arch-native" approach). Limiting the static code path to SSE4.2 allows to practically target any deployed system, however using SSE=0 and AVX=0 together is falling back to generic code, and any static kernels are not specialized using the assembly code generator.

There are two main mechanisms to install LIBXSMM (both mechanisms can be combined): (1)&#160;building the library in an out&#8209;of&#8209;tree fashion, and (2)&#160;installing into a certain location. Building in an out&#8209;of&#8209;tree fashion looks like:

```bash
cd libxsmm-install
make -f /path/to/libxsmm/Makefile
```

For example, installing into a specific location (incl. a selection of statically generated Intel&#160;SSE kernels) looks like:

```bash
make MNK="1 2 3 4 5" PREFIX=/path/to/libxsmm-install install
```

Performing `make install-minimal` omits the documentation (default: 'PREFIX/share/libxsmm'). Moreover, PINCDIR, POUTDIR, PBINDIR, and PDOCDIR allow to customize the locations underneath of the PREFIX location. To build a general package for an unpredictable audience (Linux distribution, or similar), it is advised to not over-specify or customize the build step i.e., JIT, SSE, AVX, OMP, BLAS, etc. should not be used. The following is building and installing a complete set of libraries where the generated interface matches both the static and the shared libraries:

```bash
make PREFIX=/path/to/libxsmm-install STATIC=0 install
make PREFIX=/path/to/libxsmm-install install
```

## Interfaces

### General Interface

To initialize the dispatch-table or other internal resources, an explicit initialization routine helps to avoid lazy initialization overhead when calling LIBXSMM for the first time. The library deallocates internal resources at program exit, but also provides a companion to the afore mentioned initialization (finalize).

```C
/** Initialize the library; pay for setup cost at a specific point. */
void libxsmm_init(void);
/** De-initialize the library and free internal memory (optional). */
void libxsmm_finalize(void);
```

### Matrix Multiplication<a name="interface-for-matrix-multiplication"></a>

This domain (MM) supports Small Matrix Multiplications (SMM), batches of multiple multiplications as well as the industry-standard interface for GEneral Matrix Matrix multiplication (GEMM). The details are covered in a separate [document](libxsmm_mm.md).

### Deep Neural Networks<a name="interface-for-convolutions"></a>

This domain (DNN) is detailed by a separate [document](libxsmm_dnn.md). Please also note on how to [Get Started with TensorFlow&trade; using LIBXSMM](tensorflow.md).

### Service Functions

For convenient operation of the library and to ease integration, some service routines are available. These routines may not belong to the core functionality of LIBXSMM (SMM or DNN domain), but users are encouraged to use this domain (AUX). There are two categories: (1)&#160;routines which are available for C and Fortran, and (2)&#160;routines that are only available per C interface.

The [service function domain (AUX)](libxsmm_aux.md) contains routines for:

* [Getting and setting the target architecture](libxsmm_aux.md#getting-and-setting-the-target-architecture)
* [Getting and setting the verbosity](libxsmm_aux.md#getting-and-setting-the-verbosity)
* [Measuring time durations (timer)](libxsmm_aux.md#timer-facility)
* [Loading and storing data (I/O)](libxsmm_aux.md#meta-image-file-io)
* [Allocating memory](libxsmm_aux.md#memory-allocation)

### Backend<a name="jit-backend"></a>

More information about the JIT-backend and the code generator can be found in a separate [document](libxsmm_be.md), which also includes information about LIBXSMM's stand-alone <a name="generator-driver"></a>[generator-driver](libxsmm_be.md#generator-driver) programs.

## Runtime Control<a name="running"></a>

### Verbose Mode

The [verbose mode](libxsmm_aux.md#getting-and-setting-the-verbosity) (level of verbosity) allows for an insight into the code dispatch mechanism by receiving a small tabulated statistic as soon as the library terminates. The design point for this functionality is to not impact the performance of any critical code path i.e., verbose mode is always enabled and does not require symbols (SYM=1) or debug code (DBG=1). The statistics appears (`stderr`) when the environment variable LIBXSMM_VERBOSE is set to a non-zero value. For example:

```bash
LIBXSMM_VERBOSE=1 ./myapplication
[... application output]

HSW/SP      TRY    JIT    STA    COL
   0..13      0      0      0      0
  14..23      0      0      0      0
 24..128      3      3      0      0
```

The tables are distinct between single-precision and double-precision, but either table is pruned if all counters are zero. If both tables are pruned, the library shows the code path which would have been used for JIT'ting the code: `LIBXSMM_TARGET=hsw` (otherwise the code path is shown in the table's header). The actual counters are collected for three buckets: small kernels (MNK<sup>1/3</sup>&#160;&lt;=&#160;13), medium-sized kernels (13&#160;&lt;&#160;MNK<sup>1/3</sup>&#160;&lt;=&#160;23), and larger kernels (23&#160;&lt;&#160;MNK<sup>1/3</sup>&#160;&lt;=&#160;128; the actual upper bound depends on LIBXSMM_MAX_MNK as selected at compile-time). Keep in mind, that "larger" is supposedly still small in terms of arithmetic intensity (which grows linearly with the kernel size). Unfortunately, the arithmetic intensity depends on the way a kernel is used (which operands are loaded/stored into main memory) and it is not performance-neutral to collect this information.

The TRY counter represents all attempts to register statically generated kernels, and all attempts to dynamically generate and register kernels. The TRY counter includes rejected JIT requests due to unsupported GEMM arguments. The JIT and STA counters distinct the successful cases of the afore mentioned event (TRY) into dynamically (JIT) and statically (STA) generated code. In case the capacity (O(*n*)&#160;=&#160;10<sup>5</sup>) of the code registry is exhausted, no more kernels can be registered although further attempts are not prevented. Registering many kernels (O(*n*)&#160;=&#160;10<sup>3</sup>) may ramp the number of hash key collisions (COL), which can degrade performance. The latter is prevented if the small thread-local cache is utilized effectively.

Since explicitly JIT-generated code (`libxsmm_?mmdispatch`) does not fall under the THRESHOLD criterion, the above table is extended by one line if large kernels have been requested. This indicates a missing threshold-criterion (customized dispatch), or asks for cache-blocking the matrix multiplication. The latter is already implemented by LIBXSMM's "medium-sized" GEMM routines (`libxsmm_?gemm_omp`), which perform a tiled multiplication. Setting a verbosity level of at least two summarizes the number of registered JIT-generated kernels, which includes the total size and counters for GEMM, MCOPY (matrix copy), and TCOPY (matrix transpose) kernels.

```bash
Registry: 20 MB (gemm=0 mcopy=14 tcopy=0)
```

If the call-wrapper is used, an additional runtime statistic becomes available (see [Call Wrapper](libxsmm_mm.md#call-wrapper)).

**NOTE**: Setting LIBXSMM_VERBOSE to a negative value will binary-dump each generated JIT kernel to a file with each file being named like the function name shown in [Intel&#160;VTune](libxsmm_prof.md#intelvtuneamplifier). Disassembly of the raw binary files can be accomplished by:

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

* [Profiling using Intel&#160;VTune&#160;Amplifier](libxsmm_prof.md#intelvtuneamplifier)
* [Profiling using Linux&#160;perf](libxsmm_prof.md#linuxperf)

<a name="tuning"></a>At build time, a variety of options exist to customize LIBXSMM. The library is setup for a broad range of use cases, which include sophisticated defaults for general use.

* [Customizing performance](libxsmm_tune.md#tuning)
* <a name="auto-dispatch"></a>[Tuning auto-dispatch](libxsmm_tune.md#auto-dispatch)

<a name="results"></a>To find performance results of applications or performance reproducers, the repository provides an orphaned branch called "results" which collects collateral material such as measured performance results along with explanatory figures. The results can be found at [https://github.com/hfp/libxsmm/tree/results#libxsmm-results](https://github.com/hfp/libxsmm/tree/results#libxsmm-results), or the results can be cloned as shown below.

```bash
git clone --branch results https://github.com/hfp/libxsmm.git libxsmm-results
```

Please note that comparing performance results depends on whether the operands of the matrix multiplication are streamed or not. For example, multiplying with all matrices covered by the L1 cache may have an emphasis towards an implementation which perhaps performs worse for the real workload (if this real workload needs to stream some or all matrices from the main memory). Most of the [code samples](https://github.com/hfp/libxsmm/tree/master/samples) are aimed to reproduce performance results, and it is encouraged to model the exact case or to look at real [applications](#applications).

## Applications

### High Performance Computing (HPC)

**\[1]&#160;[https://cp2k.org/](https://cp2k.org/)**: Open Source Molecular Dynamics with its DBCSR component processing batches of small matrix multiplications ("matrix stacks") out of a problem-specific distributed block-sparse matrix. Starting with [CP2K 3.0](https://www.cp2k.org/version_history), LIBXSMM can be used to substitute CP2K's 'libsmm' library. Prior to CP2K 3.0, only the [Intel-branch of CP2K](https://github.com/cp2k/cp2k/tree/intel) integrated LIBXSMM (see [https://github.com/hfp/libxsmm/raw/master/documentation/cp2k.pdf](https://github.com/hfp/libxsmm/raw/master/documentation/cp2k.pdf)).

**\[2]&#160;[https://github.com/SeisSol/SeisSol/](https://github.com/SeisSol/SeisSol/)**: SeisSol is one of the leading codes for earthquake scenarios, for simulating dynamic rupture processes. LIBXSMM provides highly optimized assembly kernels which form the computational back-bone of SeisSol (see [https://github.com/TUM-I5/seissol_kernels/](https://github.com/TUM-I5/seissol_kernels/).

**\[3]&#160;[https://github.com/NekBox/NekBox](https://github.com/NekBox/NekBox)**: NekBox is a highly scalable and portable spectral element code, which is inspired by the [Nek5000](https://nek5000.mcs.anl.gov/) code. NekBox is specialized for box geometries, and intended to prototype new methods as well as to leverage FORTRAN beyond the FORTRAN&#160;77 standard. LIBXSMM can be used to substitute the [MXM_STD](https://github.com/Nek5000/NekBox/blob/box/mxm_std.F90) code. Please also note LIBXSMM's [NekBox reproducer](https://github.com/hfp/libxsmm/tree/master/samples/nek#nek-sample-collection).

**\[4]&#160;[https://github.com/Nek5000/Nek5000](https://github.com/Nek5000/Nek5000)**: Nek5000 is the open-source, highly-scalable, always-portable spectral element code from [https://nek5000.mcs.anl.gov/](https://nek5000.mcs.anl.gov/). The development branch of the Nek5000 code [incorporates](https://github.com/Nek5000/Nek5000/blob/develop/core/mxm_wrapper.f) LIBXSMM.

**\[5]&#160;[http://pyfr.org/](http://pyfr.org/)**: PyFR is an open-source Python based framework for solving advection-diffusion type problems on streaming architectures using the flux reconstruction approach. PyFR&#160;1.6.0 optionally [incorporates LIBXSMM](http://pyfr.org/user_guide.php) as a matrix multiplication provider for the OpenMP backend. Please also note LIBXSMM's [PyFR-related code sample](https://github.com/hfp/libxsmm/tree/master/samples/pyfr).

### Machine Learning (ML)

**\[6]&#160;[https://github.com/baidu-research/DeepBench](https://github.com/baidu-research/DeepBench#deepbench)**: The primary purpose of DeepBench is to benchmark operations that are important to deep learning on different hardware platforms. LIBXSMM's DNN primitives have been [incorporated into DeepBench](https://github.com/baidu-research/DeepBench/tree/master/code/intel/convolution/libxsmm_conv) to demonstrate an increased performance of deep learning on Intel hardware. In addition, LIBXSMM's [DNN sample folder](https://github.com/hfp/libxsmm/tree/master/samples/dnn) contains scripts to run convolutions extracted from popular benchmarks in a stand-alone fashion.

**\[7]&#160;[https://www.tensorflow.org/](https://tensorflow.org/)**: TensorFlow&trade; is an open source software library for numerical computation using data flow graphs. TensorFlow was originally developed by researchers and engineers working on the Google Brain Team for the purposes of conducting machine learning and deep neural networks research. LIBXSMM can be [used](tensorflow.md#tensorflow-with-libxsmm) to increase the performance of TensorFlow on Intel hardware.

**\[8]&#160;[https://github.com/IntelLabs/SkimCaffe](https://github.com/IntelLabs/SkimCaffe#skimcaffe-specific-description)**: SkimCaffe from Intel Labs is a Caffe branch for training of sparse CNNs, which provide 80-95% sparsity in convolutions and fully-connected layers. LIBXSMM's SPMDM domain (SParseMatrix-DenseMatrix multiplication) evolved from SkimCaffe, and since then LIBXSMM implements the sparse operations in SkimCaffe.

## References

**\[1]&#160;[http://sc16.supercomputing.org/presentation/?id=pap364&sess=sess153](http://sc16.supercomputing.org/presentation/?id=pap364&sess=sess153)**: LIBXSMM: Accelerating Small Matrix Multiplications by Runtime Code Generation ([paper](http://www.computer.org/csdl/proceedings/sc/2016/8815/00/8815a981.pdf)). SC'16: The International Conference for High Performance Computing, Networking, Storage and Analysis, Salt Lake City (Utah).

**\[2]&#160;[http://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/tech_poster_pages/post137.html](http://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/tech_poster_pages/post137.html)**: LIBXSMM: A High Performance Library for Small Matrix Multiplications ([poster](http://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/poster_files/post137s2-file2.pdf) and [abstract](http://sc15.supercomputing.org/sites/all/themes/SC15images/tech_poster/poster_files/post137s2-file3.pdf)). SC'15: The International Conference for High Performance Computing, Networking, Storage and Analysis, Austin (Texas).

**\[3]&#160;[https://software.intel.com/en-us/articles/intel-xeon-phi-delivers-competitive-performance-for-deep-learning-and-getting-better-fast](https://software.intel.com/en-us/articles/intel-xeon-phi-delivers-competitive-performance-for-deep-learning-and-getting-better-fast)**: Intel Xeon Phi Delivers Competitive Performance For Deep Learning - And Getting Better Fast. Article mentioning LIBXSMM's performance of convolution kernels with [DeepBench](https://github.com/baidu-research/DeepBench/tree/master/code/intel/convolution/libxsmm_conv). Intel Corporation, 2016.

