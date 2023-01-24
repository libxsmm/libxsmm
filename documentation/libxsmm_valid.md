## Basic Tests

To run basic tests:

```bash
make tests
```

Remember: a set of key-value pairs represents a single unique (re-)build (and test):

```bash
make STATIC=0 tests
```

## Test Suites

It is possible to run whole test suites or collections of tests like for LIBXSMM's Continuous Integration (CI). The script `tool_test.sh` is included in archives and releases, i.e., it also works for non-repository folders. To run an entire collection (aka `scripts/tool_test.sh 0`).

```bash
scripts/tool_test.sh
```

It is also possible to select a single test (out of the whole collection):

```bash
scripts/tool_test.sh 1
```

In general, key-value pairs which are valid for LIBXSMM's `make` can be specified:

```bash
AVX=3 DBG=1 scripts/tool_test.sh
```

There are several collections of tests covering specific domains:

* `samples/utilities/wrap/wrap-test.sh`: test substituting standard symbols at link/run-time (gemm, gemv, etc).
* `samples/xgemm/kernel_test.sh`: test SMM kernels in an almost exhaustive fashion (brute-force).
* `samples/eltwise/run_test.sh`: test all kinds of element-wise kernels and variants.
* `samples/pyfr/test.sh`: test Sparse Matrix times Dense Matrix (FsSpMDM).

## Reproduce Failures

LIBXSMM's [verbose mode](https://libxsmm.readthedocs.io/#verbose-mode) can print the invocation arguments when launching a test driver (`LIBXSMM_VERBOSE=4` and beyond). For example (`LIBXSMM_VERBOSE=4 ./run_test_avx2.sh`), the termination message of a failing test may look like:

```text
[...]
LIBXSMM_TARGET: hsw
Registry and code: 13 MB + 8 KB (meltw=1)
Command: ./eltwise_binary_simple 1 0 F32 F32 F32 F32 10 10 10 10 
[...]
```

**Note**: scripts such `scripts/tool_pexec.sh` suppress error output (console) by default and capture error output in individual files, i.e., verbose output may not be immediately visible.

## Portability

It is desirable to exercise portability and reliability of LIBXSMM's source code even on Non-Intel Architecture by the means of compilation, linkage, and generic tests. This section is *not* about Intel Architecture (or compatible). Successful compilation (or even running some of the tests successfully) does not mean LIBXSMM is valuable on that platform.

Make sure to rely on `PLATFORM=1`, otherwise a compilation error should occur _Intel Architecture or compatible CPU required!_ This error avoids (automated) attempts to upstream LIBXSMM to an unsupported platform. LIBXSMM is upstreamed for Intel Architecture on all major Linux distributions, FreeBSD, and others. If compilation fails with _LIBXSMM is only supported on a 64-bit platform!_, `make PLATFORM=1 DBG=1` can be used to exercise compilation.

If platform support is forced (`PLATFORM=1`), runtime code generation is disabled at compile-time (`JIT=0`). Runtime code generation can be also enabled (`PLATFORM=1 JIT=1`) but code-dispatch will still return NULL-kernels. However, some tests will start failing as missing JIT-support it is not signaled at compile-time as with `JIT=0`.

**Note**: JIT-support normally guarantees a non-NULL code pointer ("kernel") if the request is according to the [limitations](https://github.com/libxsmm/libxsmm/wiki/Q&A#what-is-a-small-matrix-multiplication) (user-code is not asked to check for a NULL-kernel), which does not hold true if JIT is enabled on a platform that does not implement it.

### TinyCC

The Tiny C Compiler (TinyCC) supports Intel Architecture, but lacks at least support for thread-local storage (TLS).

```bash
make CC=tcc THREADS=0 INTRINSICS=0 VLA=0 ASNEEDED=0 BLAS=0 FORCE_CXX=0
```

### IBM XL Compiler for Linux (POWER)

The POWER platform requires aforementioned `PLATFORM=1` to unlock compilation.

```bash
make PLATFORM=1 CC=xlc CXX=xlc++ FC=xlf
```

### Cross-compilation for ARM

ARM AArch64 is regularly [supported](https://github.com/libxsmm/libxsmm/wiki/Compatibility#arm-aarch64). However, 32-bit ARM requires aforementioned `PLATFORM=1` to unlock compilation (similar to 32-bit Intel Architecture). Unlocking compilation for 32-bit ARM is not be confused with supporting 32-bit ARM architectures.

```bash
make PLATFORM=1 AR=arm-linux-gnueabi-ar \
  FC=arm-linux-gnueabi-gfortran \
  CXX=arm-linux-gnueabi-g++ \
  CC=arm-linux-gnueabi-gcc
```
