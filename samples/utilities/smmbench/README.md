# SMM Benchmark

This collection of code samples exercises different memory streaming cases when performing the matrix multiplication *C<sub>m&#8239;x&#8239;n</sub> = alpha &middot; A<sub>m&#8239;x&#8239;k</sub> &middot; B<sub>k&#8239;x&#8239;n</sub> + beta &middot; C<sub>m&#8239;x&#8239;n</sub>*: (1)&#160;streaming the matrices A, B, and C which is usually referred as batched matrix multiplication, (2)&#160;streaming the inputs A and B but accumulating C within cache, (3)&#160;streaming the A and C matrices while B is kept in cache, (4)&#160;streaming the B and C matrices while A is kept in cache, and (4)&#160;not streaming any of the operands but repeating the very same multiplication until the requested number of matrix multiplications has been completed.

Beside of measuring the duration of a test case, the performance is presented in GFLOPS/s. As an alternative metric, the memory bandwidth is given (the artificial "cached" case omits to present the cache-memory bandwidth). The "pseudo-performance" given in FLOPS/cycle is an artificial scoring, it not only uses a non-standard formula for calculating the FLOPS (*2 \* M \* N \* K - M \* N* rather than *2 \* M \* N \* K*) but also relies on (pseudo-)clock cycles:

```
$ ./specialized.sh 0
m=32 n=32 k=32 size=87381 memory=2048.0 MB (DP)

Batched (A,B,C)...
        pseudo-perf.: 10.7 FLOPS/cycle
        performance: 23.9 GFLOPS/s
        bandwidth: 11.1 GB/s
        duration: 239 ms
Finished
```

There are two sub collections of samples codes: (1)&#160;a collection of C++ code samples showing either BLAS, Compiler-generated code (inlined code), LIBXSMM/dispatched, LIBXSMM/specialized functions to carry out the multiplication, and (2)&#160;a Fortran sample code showing BLAS versus LIBXSMM including some result validation.

**C/C++ Code Samples: Command Line Interface (CLI)**

* Takes an optional number (1st arg.) to select the streaming-case (0...8)
* Optionally takes the M, N, and K parameter of the GEMM in this order
* If only M is supplied, the N and K "inherit" the M-value
* Example I  (A,B,C): ./specialized.sh 0 16 8 9
* Example II   (A,B): ./specialized.sh 6 16

**Fortran Code Sample: Command Line Interface (CLI)**

* Optionally takes the M, N, and K parameter of the GEMM in this order
* Optional problem size (in MB) of the workload; M/N/K must have been supplied
* Optional total problem size (in MB) implying the number of repeated run
* If only M is supplied, the N and K are "inheriting" the M-value
* Shows the performance of each of the streaming cases
* Example I: ./smm.sh 16 8 9 1024 16384
* Example II: ./smm.sh 16

