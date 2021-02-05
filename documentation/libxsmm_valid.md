## Basic Tests

To run basic [tests](http://libxsmm.readthedocs.io/#classic-library-abi):

```bash
make tests
```

Remember: a set of key-value pairs represents a single unique (re-)build (and test):

```bash
make STATIC=0 tests
```

There is a whole collection of test targets available (`test-cp2k`, `test-cpp`, `test-nek`). However, it is then better to rely on test-suites.

## Test Suites

It is possible to run tests like LIBXSMM's continuous integration ([https://travis-ci.org/hfp/libxsmm](https://travis-ci.org/hfp/libxsmm)):

```bash
scripts/tool_test.sh
```

The above command runs the entire collection ("scripts/tool_test.sh 0"). However, one test (of currently 11 tests) can be selected by number (1-11):

```bash
scripts/tool_test.sh 1
```

The suite itself can be also selected. For example, some DNN tests are described in `.test-dnn.yml`:

```bash
TESTSET=test-dnn scripts/tool_test.sh
```

In general, all key-value pairs valid for LIBXSMM's `make` can be given as part of the environment:

```bash
AVX=3 MIC=0 TESTSET=test-dnn scripts/tool_test.sh
```

Please note, the suite/test itself may be comprised of key-value pairs that take precedence.

## CI Tests

The `tool_test.sh` script is included in repository archives and releases i.e., it works for non-repository folders. In contrast, the Continuous Integration (CI) use case relies on the Git command being present and the folder being a Git-clone.

Functionality

* `[skip ci]` as part of a commit message will not trigger the CI agents, and tests are skipped for such a commit.
* `[full ci]` as part of a commit message will trigger a full test even if the setup uses the "Fast CI" option.

The "Fast CI" option is enabled per filename given as 2nd command line argument:

```bash
scripts/tool_test.sh 1 .fullci
```

In the above example, a file named `.fullci` may contain path/file patterns (wildcard format) triggering a full test if the files changed by the commit match any of the patterns.

## Portability

It is desirable to exercise portability and reliability of LIBXSMM's source code even on Non-Intel Architecture by the means of compilation, linkage, and generic tests. This section is *not* about Intel Architecture (or compatible). Successful compilation (or even running some of the tests successfully) does not mean LIBXSMM is valuable on that platform.

Make sure to rely on `PLATFORM=1`, otherwise a compilation error should occur _Intel Architecture or compatible CPU required!_ This error avoids (automated) attempts to upstream LIBXSMM to an unsupported platform. LIBXSMM is upstreamed for Intel Architecture on all major Linux distributions, FreeBSD, and others. If compilation fails with _LIBXSMM is only supported on a 64-bit platform!_, `make PLATFORM=1 DBG=1` can be used to exercise compilation.

If platform support is forced (`PLATFORM=1`), runtime code generation is disabled at compile-time (`JIT=0`). Runtime code generation can be also enabled (`PLATFORM=1 JIT=1`) but code-dispatch will still return NULL-kernels. However, some tests will start failing as missing JIT-support it is not signaled at compile-time as with `JIT=0`.

**Note**: JIT-support normally guarantees a non-NULL code pointer ("kernel") if the request is according to the [limitations](https://github.com/hfp/libxsmm/wiki/Q&A#what-is-a-small-matrix-multiplication) (user-code is not asked to check for a NULL-kernel), which does not hold true if JIT is enabled on a platform that does not implement it.

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

The ARM platform requires aforementioned `PLATFORM=1` to unlock compilation. Please note, `PLATFORM=1` also unlocks compilation for 32-bit targets.

```bash
make PLATFORM=1 AR=arm-linux-gnueabi-ar \
  FC=arm-linux-gnueabi-gfortran \
  CXX=arm-linux-gnueabi-g++ \
  CC=arm-linux-gnueabi-gcc
```

```bash
make PLATFORM=1 AR=aarch64-linux-gnu-ar \
  FC=aarch64-linux-gnu-gfortran \
  CXX=aarch64-linux-gnu-g++ \
  CC=aarch64-linux-gnu-gcc
```
