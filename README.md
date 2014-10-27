libxsmm
=======
Library for small matrix-matrix multiplications targeting Intel Architecture (x86). This initial version of the library is targeting the Intel Xeon Phi coprocessor (an instance of the Intel Many Integrated Core Architecture "MIC") particularly using KNC intrinsic functions.

The code can be compiled to native code which is also usable in an offloaded code section (via a FORTRAN directive or a C/C++ pragma). The prerequisite for offloading the code is to compile it to position-independent (PIC) code even when building a static library.

Performance: the presented code is by no means "optimal" or "best-performing"; it just uses Intrinsic functions. In fact, a well-optimizing compiler may produce better performing code.

INSTRUCTIONS
============
To compile the library just use:
```
make
```
The static library is produced inside the directory lib. To remove generated files use:
```
make clean
```
or to remove generated files and the built library use:
```
make realclean
```
The interface (see include/xsmm_knc.h) to the library is:
```
dc_smm_dnn_function_type dc_smm_dnn_function(int M, int N, int K);
void xsmm_dnn(int M, int N, int K, const double* a, const double* b, double* c)
```
where C(M,N) = C(M,N) + A(M,K) * B(K,N). The function shown first helps to amortize
the cost of the dispatch when multiple calls with the same M, N, and K are needed.

The library can be configured to accept row-major (default) or column-major order matrices;
change the variable ROW_MAJOR inside the Makefile file (1 for row-major,
and column-major otherwise); or one can run:
```
make ROW_MAJOR=0
```
The values of the matrix sizes (M,N,K values) can be set by changing the 
variables inside the Makefile file or by running for example:
```
make INDICES_M="2 4" INDICES_N="1" INDICES_K="$(echo $(seq 2 5))"
```
which generates the (M,N,K) values:
```
(2,1,2), (2,1,3), (2,1,4), (2,1,5)
(4,1,2), (4,1,3), (4,1,4), (4,1,5)
```
The fallback of the library is DGEMM if it is called for values other than specified
by INDICES_M, INDICES_N, or INDICES_K.
