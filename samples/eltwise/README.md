# Elementwise correctness- and performance tests

This folder contains tests for kernels, which work on each element of a given input separately. Examples for these operations are adding two matrices or vectors, or applying the square root to all elements individually.

*Disclosure: Performance tests haven't been implemented for all types yet.*

## Build

```bash
cd /path/to/libxsmm
make
or
make BLAS=0 LIBXSMM_NO_BLAS=1 STATIC=0 -j 64

cd /path/to/libxsmm/samples/eltwise
make -j 16
```

## Test specific kernels

To run a specific kernel, call one of the executable with its arguments. The arguments will be listed, when you call it without any.

## Test collections

In this directory, there are multiple bash files, which will execute multiple random tests for a specific type of operation.
These collections call bash scripts from the subdirectory "kernel_test", which will in turn call the executables in this directory.

## Compare performance between different architectures

If your machine supports multiple architectures like ARM ASIMD and ARM SVE, you can set the environment variable **ARCH1** to a second architecture. The performance tests will then run the kernels on both architectures, and compare them.

## Useful environment variables

When you want to test another architecture, specify **LIBXSMM_TARGET**.

If you want more debugging information, set **LIBXSMM_VERBOSE**. Setting it to -1 will print all debug information, and write the JIT-ed kernels into local files.