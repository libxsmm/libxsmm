# Validation

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
