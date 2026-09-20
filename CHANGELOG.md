# Changelog

## 2.1.0

Highlight: a new **PowerPC (ppc64le, POWER8–POWER10) JIT backend** with cross-architecture reference-kernel fallbacks.

### New Architecture Support
- **PowerPC backend (ppc64le, POWER8–POWER10)**: JIT GEMM/BRGEMM microkernels for VSX and MMA, kernel-independent dynamic k-blocking, CTR-based k-loops, sparse-kernel support, PPC CPUID/feature detection, and a POWER10 CI workflow. Adds cross-architecture reference-kernel fallbacks (trampolines) for GEMM, eltwise, equations, and packed GEMM, extended across ppc64le and RV64 (#1080).

### Enhancements
- **ARM SME**: Added Transpose-A and strided BRGEMM support, plus adjusted transpose code generation and CI coverage (#1068).
- **Reference fallback**: Fall back to the reference implementation for the built-for target when CPUID detection fails; hardened previously uninitialized structs (#1076).

### Bug Fixes
- Fixed ordering bug in A and B scale loads (#1073).
- Updated to the latest BSRINIT encoding and fixed the encoder (#1070).
- Fixed a TBAA dead-store elimination risk via `memcpy` into `unsigned long long l_imm_vals[4]`; verified clean under `-std=c89 -Wall -Werror` (#1078).

### Build & Packaging
- CMake: install headers into the `include/libxsmm` subdirectory (#1077).
- CMake: adjust Fortran module and header installation (#1079).
- Added an AArch64 CMake test with correct `libxsmm.so` detection and `-march=armv8.1-a` (#1075).

### CI / Docs / Dependencies
- Changed Slurm partition for CI (NFC).
- Added PETSc to the README applications list (#1072).
- Bumped `mistune` 3.2.1 → 3.3.0 in the docs theme (#1067).

**Full Changelog**: https://github.com/libxsmm/libxsmm/compare/2.0.0...2.1.0

## 2.0.0 (2026-06-29)

Major release (~4300 commits since 1.17). Highlights:

- **New architecture backends**: full AArch64 (NEON, SVE, SME with MMLA), RISC-V RV64 (RVV, software-pipelining, transpose, `gelu_minimax`), and Apple M4 support; AArch64/RISC-V static code generation.
- **x86**: Intel AMX GEMMs across bf16, bf8/hf8, fp8, int8 and bf32, with a correct AMX ABI for safe interop; AVX10.2 (BF16, MOVRS) instructions; ACE (AI Compute Extensions) v1; SPR/GNR/SRF tuning and CI.
- **Low/mixed precision**: MXFP4, int4/int8 (UU/SS), 1/2-bit GEMMs, VNNI8 layout, and quant/dequant TPPs.
- **Datatype refactor**: operand signedness encoded in datatypes (removed `LIBXSMM_GEMM_FLAG_{A,B,C}_UNSIGNED`); 6-bit datatype fields extended to eltwise and equation kernels.
- **TPPs / eltwise**: new binary-compare and ternary-select TPPs, absmax reduce, exp/gelu approximations, and SSE eltwise support.
- **Kernels**: packed GEMM on x86/AArch64 (FP32/FP64), bitmask-based sparse×dense GEMM, reference JIT kernels, and static code generation via binary export.
- **Build**: CMake support (package config, Fortran bindings, unit tests, install layout); Windows support finalized; minimized malloc API.
- **Removed**: in-tree deep-learning examples (moved to `libxsmm-dnn`).

**Full Changelog**: https://github.com/libxsmm/libxsmm/compare/1.17...2.0.0

## 1.17 (2021-12-02)

- Removed dependency on performance counters/markers (#562); back-ported additional changes from main.

## 1.16.3 (2021-10-13)

- Minor maintenance fixes.

## 1.16.2 (2021-08-31)

- Minor maintenance fixes.

## 1.16.1 (2020-06-26)

- Completed AVX-512 encoder support for all 48 FMA instruction variants across xmm/ymm registers; Cray Classic (non-Clang) compiler adjustments.

## 1.16 (2020-06-20)

- Reworked GEMM fused-eltwise infrastructure; column-bias + activation fusion and compressed eltwise flags.
- Large AVX-512 encoder expansion (48 FMA variants, `VMOVDQU64`, `vcvtne2ps2bf16` fix); added sigmoid and tanh fixes.
- Thread-safe random-number generation API; randomized GEMM testing for faster CI.

## 1.15 (2020-03-13)

- BF16 MLP drivers and a simple SGD; user-data dispatch via `libxsmm_xregister`/`libxsmm_xdispatch`/`libxsmm_xrelease`.
- Variable-length SOA sparse kernels (CSR/CSC); `LIBXSMM_CACHE` control; out-of-tree build fix (#371).

## 1.14 (2019-10-25)

- Introduced `libxsmm_cpuid_vlen32` and `libxsmm_xgemm` (sequential form); improved transpose tests (`otrans`/`itrans`).
- matcopy dispatch/descriptor fix-ups for low-precision CNNs; `libatomic` support; MKL_DIRECT improvements.

## 1.13 (2019-07-14)

- Split out `libxsmm_memory.h`; enabled bfloat16 intrinsic/compiler support.
- Encoder additions (`leaq`, `tzcnt`, `popcnt`); Slurm job-script runner and `cpuinfo` tool.

## 1.12.1 (2019-05-22)

- `libxsmm_realloc` stub; convolution test scripts; Clang `_mmask16` fix.

## 1.12 (2019-05-10)

- Public `libxsmm_?gemm_batch`/`omp` interface and `?gemm_batch` call interception; multiple tanh implementations (AVX512F).
- BF16 LSTM in `nc_kcck`/VNNI format; pip wheel packaging; TensorFlow LSTM Op wrapper; reworked WRAP control.

## 1.11 (2019-04-29)

- Packed GEMM and GETRF (compact format) via the code-registry.
- `LIBXSMM_PLATFORM_SUPPORTED` (ARM builds compile but issue a runtime error); encoder additions (`VPMOVDW`/`VPMOVSXWD`); `libxsmm_atomic_kind`; Cray/PGI portability.

## 1.10 (2018-11-12)

- BF16 GEMM groundwork and end-to-end BF16 training; combined RNN/LSTM code with a TensorFlow-conforming LSTM cell.
- KNM support for pooling, fused batchnorm and fully-connected layers; quantization macros; `LIBXSMM_DUMP_BUILD`.

## 1.9 (2018-03-15)

- Separate input/output types for `libxsmm_mmfunction`; new `libxsmm_math.h` (`sexp2`, integer `sqrt`/`cbrt`).
- I16I32/I16F32 small GEMM for Skylake-X (AVX512_VNNIW); JIT enabled starting from SSE3; BGEMM low-precision.

## 1.8.3 (2018-02-02)

- Extensive SSE3 encoder additions (`movaps`/`mulpd`/…); improved PGI compiler support; header-only C/C++ mixing supported; VTune on Intel-2018 toolchain.

## 1.8.2 (2017-12-24)

- Large release (~1350 commits): own mutex/rwlock (`LIBXSMM_LOCK_*`) and RW-lock API.
- AVX-512 encoder additions (`VPERMD`, `VPERMW`); KNM SGEMM for K not divisible by 4; qfma fill-ins; EDGE sample VS projects; cache blocking/unrolling.

## 1.8.1 (2017-05-12)

- Winograd forward-path AVX-512 optimizations; `LIBXSMM_GEMM_WRAP` env var; `KMOV` (b/d/q) encodings; low-precision backprop; headerless VTune profiling.

## 1.8 (2017-03-30)

- Scratch-memory pools (`LIBXSMM_SCRATCH_POOLS`/`_SCALE`); internalized TRACE/backtrace; scratch microbenchmark; NHWC backward padding fixes.

## 1.7.1 (2017-01-27)

- `libxsmm_release_scratch` (stub); documented default vs scratch memory domains.

## 1.7 (2017-01-26)

- Allocator overhaul: `libxsmm_set_allocator`, distinct default/scratch allocators, and `libxsmm_tf_allocator`/`libxsmm_scoped_allocator` for TensorFlow; thread-safe init/finalize.
- New DNN buffer/filter bind API; NEK-streaming GEMM cases (A/C and B/C streamed).

## 1.6.6 (2017-01-19)

- Code-quality fixes across the gemm/conv/spgemm generators (with assertions); `LIBXSMM_SE` control.

## 1.6.5 (2017-01-17)

- JIT for sparse-A-in-registers (PyFR); `libxsmm_create_gemm_descriptor` with explicit precision; constant-vector loads through the code segment; `LIBXSMM_INTRINSICS_NONE`.

## 1.6.4 (2017-01-09)

- Raw descriptor form; Clang/ICC AVX2/AVX-512 support workarounds; padding-API support for i8/i16/i32 convolutions; host/config environment sourcing.

## 1.6.3 (2016-12-23)

- Shared tiled-GEMM prefetch strategy; `INIT=0` build flag; exposed try-lock in the C/Fortran interfaces.

## 1.6.2 (2016-12-21)

- SPMDM transpose-B and sparse-A transpose; `LIBXSMM_MALLOC_NOCRC`/`_FALLBACK` options; Clang AVX-512 ICE workarounds; warning-as-error preprocessing.

## 1.6.1 (2016-12-05)

- SPMDM with bfloat16 (bf×bf→f32); 1D block-id API; JIT prefetch alignment; `PEDANTIC=2` for non-Fortran; reduced memory consumption.

## 1.6 (2016-11-30)

- AVX2 fallbacks for AVX-512 NHWC/RSCK backward and weight-update convolutions; per-layer error handling; `BL2_via_C` prefetch for sparse SOA matmul; header-only C89 conformance.

## 1.5.2 (2016-11-03)

- Build stand-alone generator binaries by default; ifort 1.5.1 `libxsmm.f` segfault workaround (#104).

## 1.5.1 (2016-10-26)

- SPMDM (#101); partial in-place transpose fallback; 8/16-bit integer instructions; merged AVX-512 convolution generators; Fortran interface fixes across GNU/PGI.

## 1.5 (2016-10-05)

- Header-only library support (including mixed C/C++); `libxsmmnoblas` to remove the BLAS dependency.
- FORTRAN 77 OpenMP out-of-place transpose; DNN convolution frontend; VS projects for libxsmmext/convolution.

## 1.4.4 (2016-08-01)

- Header-only library (#86); `BLAS=0` build and libxsmmext link-order control; JIT-code dump without a debug build (#88); transpose CPUID dispatch.

## 1.4.3 (2016-05-20)

- New sparse CSR generator (asparse) with row-major support; 16-byte descriptor comparison (SSE/AVX/AVX2).

## 1.4.2 (2016-05-17)

- Verbose mode with statistics tables (#78); get/set target-arch API rework (#75); distinct `AVX512_MIC` vs `AVX512_CORE`; VTune JIT-profiling fix.

## 1.4.1 (2016-05-04)

- Reinitialization counter for the thread-local code cache; 16-byte descriptor (`LIBXSMM_GENERATOR_BIGDESC`); batched MKL GEMM in the blas sample.

## 1.4 (2016-04-04)

- Prefetch-strategy selection (`LIBXSMM_PREFETCH`/`_AUTO`, per-CPUID, #69); `libxsmm_omp_?gemm` routines; hash-collision handling for static code; moved collateral data to an orphaned branch (#70).

## 1.3 (2016-03-22)

- Dispatch cache (`libxsmm_gemm_diffn`, AVX-512 path, #62); "medium-sized" matmul routines (#65); `libxsmm_get_target_arch()` for C/Fortran (#68).

## 1.2 (2016-02-19)

- Static link-time GEMM wrapper for MKL/BLAS (`--wrap`); `LIBXSMM_JIT` env var (`0|1|snb|hsw|knl|skx`); Makefile build-state tracking; Mersenne-prime registry sizing.

## 1.1.1 (2015-12-26)

- Allow manual code-path selection (fix for feature-bit suppression).

## 1.1 (2015-12-23)

- Row-major/col-major build toggle (`ROW_MAJOR`); `LIBXSMM_JIT` env var; CPUID gathered at init; Travis CI and code coverage.

## 1.0.2 (2015-12-03)

- Version-stamp interface (#31: `LIBXSMM_BRANCH`, `LIBXSMM_VERSION_*`); Python 2/3 compatibility (#46); ThreadSanitizer support; JIT race-condition and initialization fixes.

## 1.0.1 (2015-10-17)

- Fixed JIT level implementation and lazy initialization; BLAS warm-up in the smm sample.

## 1.0 (2015-10-16)

- Baseline release for this changelog. Earlier 0.8.x/0.9.x tags predate it and are omitted.
