## What is the background of the name "LIBXSMM"?
The "MM" stands for Matrix Multiplication, and the "S" clarifies the working domain i.e., Small Matrix Multiplication. The latter also means the name is neither a variation of "MXM" nor an eXtreme Small Matrix Multiplication but rather about Intel Architecture (x86) - and no, the library is [64&#8209;bit only](https://github.com/hfp/libxsmm/issues/103#issuecomment-256887962). The spelling of the name might follow the syllables of libx\\/smm, libx'smm, or libx&#8209;smm.
> **NOTE**: the library does [not](https://github.com/hfp/libxsmm/issues/103#issuecomment-256887962) support 32-bit architecture (64&#8209;bit only)

## What is a small matrix multiplication?
When characterizing the problem-size using the M, N, and K parameters, a problem-size suitable for LIBXSMM falls approximately within *(M&#160;N&#160;K)<sup>1/3</sup>&#160;\<=&#160;128* (which illustrates that non-square matrices or even "tall and skinny" shapes are covered as well). The library is typically used to generate code up to the specified [threshold](#auto-dispatch). Raising the threshold may not only generate excessive amounts of code (due to unrolling in M or K dimension), but also miss to implement a tiling scheme to effectively utilize the cache hierarchy. For auto-dispatched problem-sizes above the configurable threshold (explicitly JIT'ted code is **not** subject to the threshold), LIBXSMM is falling back to BLAS. In terms of GEMM, the supported kernels are limited to *Alpha := 1*, *Beta := \{ 1, 0 \}*, and *TransA := 'N'*.
> **NOTE**: *Alpha*, *Beta*, and *TransA* are limited to `1`, `{ 1, 0 }`, and `'N'` respectively.

## What is a small convolution?
In the last years, new workloads such as deep learning and more specifically convolutional neural networks (CNN) emerged, and are pushing the limits of today's hardware. One of the expensive kernels is a small convolution with certain kernel sizes (3, 5, or 7) such that calculations in the frequency space is not the most efficient method when compared with direct convolutions. LIBXSMM's current support for convolutions aims for an easy to use invocation of small (direct) convolutions, which are intended for CNN training and classification. The [Interface](#interface-for-convolutions) is currently ramping up, and the functionality increases quickly towards a broader set of use cases.

## What about "medium-sized" and big(ger) matrix multiplications?
A more recent addition are GEMM routines, which are parallelized using OpenMP (`libxsmm_?gemm_omp`). These routines leverage the same specialized kernel routines as the small matrix multiplications, in-memory code generation (JIT), and automatic code/parameter dispatch but they implement a tile-based multiplication scheme i.e., a scheme that is suitable for larger problem-sizes. For *Alpha*, *Beta*, *TransA*, and *TransB*, the limitations of the small matrix multiplication kernels apply. More details can be found in the [description of the xgemm sample code](https://github.com/hfp/libxsmm/tree/master/samples/xgemm#xgemm-tiled-gemm-routines).

## How to determine whether an application can benefit from using LIBXSMM or not?
Given the application uses BLAS to carry out matrix multiplications, one may use the [Call Wrapper](#call-wrapper), and measure the application performance e.g., time to solution. However, the latter can significantly improve when using LIBXSMM's API directly. To check whether there are applicable GEMM-calls, the [Verbose Mode](#verbose-mode) can help to collect an insight. Further, when an application uses [Intel&#160;MKL&#160;11.2](https://registrationcenter.intel.com/en/forms/?productid=2558) (or higher), then running the application with the environment variable MKL_VERBOSE=1 (`env MKL_VERBOSE=1 ./workload > verbose.txt`) can collect a similar insight (`grep -a "MKL_VERBOSE DGEMM(N,N" verbose.txt | cut -d'(' -f2 | cut -d, -f3-5"`).

## Is LIBXSMM compatible from version-to-version, or what is the ABI commitment?
One may have a look at issue [#120](https://github.com/hfp/libxsmm/issues/120#issuecomment-264498939) or [#282](https://github.com/hfp/libxsmm/issues/282#issuecomment-485390494), but in summary:
* Binary compatibility is not continuously tested (only manually for a subset of the API namely SMM domain).
* Major versions are likely breaking binary compatibility with existing integrations (that is typical).
* Minor versions may break binary compatibility of recently introduced features (may not be typical).
* Update and patch versions are binary compatible but may only be released on request (issue).

LIBXSMM's API for Small Matrix Multiplications (SMMs) is considered stable, and all major known applications (e.g., CP2K, EDGE, NEK5K, and SeisSol) either rely on SMMs or are able (and want) to benefit from an improved API of any of the other domains (e.g., DL). Until at least v2.0, LIBXSMM is not able to track or even maintain binary compatibility and hence the SONAME also goes with the semantic version. A [list of public functions](https://github.com/hfp/libxsmm/blob/master/.abi.txt) is maintained (but there is no distinction for a small subset of them that are only meant for communication between LIBXSMM and LIBXSMM/ext).

## I am relying on a prebuilt version of CP2K (or another application), is LIBXSMM incorporated and which version is it?
This can be determined using the environment variable `LIBXSMM_VERBOSE=2` (or higher verbosity). It is not even required to use an input or workload since the information in question is presented when the program terminates. For example:

```
LIBXSMM_VERBOSE=1 exe/Linux-x86-64-intelx/cp2k.psmp
[...]
LIBXSMM_VERSION: release-1.11
LIBXSMM_TARGET: clx
```

## I am relying on a prebuilt version of an application, and I am concerned about optimal compiler flags.
LIBXSMM uses JIT-generated code according to the CPUID of the system. This is independent of the compiler flags used to build the library. If LIBXSMM was incorporated per [classic ABI](https://libxsmm.readthedocs.io/#classic-library-abi), `LIBXSMM_DUMP_BUILD=1` environment variable allows to print build flags used for LIBXSMM at termination of the application. This output of `LIBXSMM_DUMP_BUILD=1` can yield hints about the flags used to build the application (if similar).

For concerns regarding the code of an application that cannot benefit from LIBXSMM, one may have a look at the build recipes of the [XCONFIGURE](http://xconfigure.readthedocs.io/) project.

## What Operating Systems are covered by LIBXSMM, and what about Microsoft Windows?
The answer here focuses on the actual runtime support rather than the supported compiler tool chains used to build the library. All flavors of Linux are supported (if the library was successfully built), which includes installations running a security-hardened Linux kernel (SELinux). The Apple OS (OSX) is supported, which also includes more recent SIP-enabled versions (System Integrity Protection). The BSD OS is likely supported, but building the library is only occasionally validated. Microsoft Windows is supported for non-JIT operation, and for most (e.g., GEMM and MATCOPY) of the JIT-kernels (prefetch signature is not supported). There is currently no support for JIT in the DNN domain (no further check is performed i.e., crash at runtime). See also [issue #71](https://github.com/hfp/libxsmm/issues/71).

## Does LIBXSMM has some support for GEMV?
The library generates acceptable code when using `M=1` or `N=1`. For example, building with `make M=16 N=1 K=16 AVX=2` and inspecting the assembly (build directory) or dumping/disassembling the JIT code (see reference documentation) shows the minimum number of load/store instructions. Given that GEMV is a memory bound operation, this suggests reasonable code quality. LIBXSMM selects from multiple microkernels (specific for each ISA extension) by using a fixed scheme/heuristic, which should be acceptable for GEMV. The sample code under [samples/smm](https://github.com/hfp/libxsmm/blob/master/samples/smm) provides ready-to-use benchmark drivers that can help to compare the performance with LAPACK/BLAS. Afore mentioned benchmarks exercise streaming all possible combinations of operands.

## What about complex and mixed types?
This question refers to the following kind of element type of the GEMM interface of LIBXSMM:
* Complex types: complex numbers in single and double-precision,
* Mixed types: e.g. real double-precision and complex double-precision
There are no (immediate) plans to support more types for the GEMM part. Please note, that LIBXSMM indeed supports lower precision GEMM (wgemm).

## What about voting for features?
All feedback and [issue reports](https://github.com/hfp/libxsmm/issues) are handled openly, are welcome and considered ([answered](https://github.com/hfp/libxsmm/issues?q=is%3Aissue+is%3Aclosed), and [collected](https://github.com/hfp/libxsmm/wiki/Development#longer-term-issues)). However, we do not seek for "feature votes" since the development of the library is not a democratic process.

## \<DEPRECATED\> What is the purpose of ROW_MAJOR vs. COL_MAJOR?
This build configuration is deprecated ([issue 85](https://github.com/hfp/libxsmm/issues/85)), otherwise there is nothing one cannot achieve with row-major as opposed to column-major storage order. In particular the choice is not about whether a program is written in C/C++ or in FORTRAN. The ROW_MAJOR setting is just offered for existing code, which calls into function(s) that assume row-major storage order and where these calls are to be replaced by LIBXSMM in a "1:1 fashion". It is encouraged to avoid the ROW_MAJOR setting since BLAS implies COL_MAJOR (and LIBXSMM is supposed to be compatible with BLAS). [More...](https://github.com/hfp/libxsmm/issues/80)
