# Dispatch (Microbenchmark)

This code sample attempts to benchmark the performance of the dispatch mechanism. This mechanism is relevant when replacing GEMM calls (see [Call Wrapper](https://github.com/hfp/libxsmm#call-wrapper) section of the reference documentation), or generally when calling LIBXSMM's `libxsmm_?gemm` functions.

**Command Line Interface (CLI)**

* Optionally takes the number of dispatches to be performed
* Measures the duration needed to find the requested kernel
* Excludes the time needed to generate the kernel
* Shows time needed in relation to an empty function call

