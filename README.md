libxsmm
=======

Library for small matrix-matrix multiplications targeting Intel Architecture (x86).

==============
 INSTRUCTIONS
==============

To compile the library just use

make

The static library is produced inside the directory lib.

To clean the library use

make clean (remove generated files) 

or 

make realclean (remove generated files and the library).

The interface to the library is:

void xsmm_dnn(int M, int N, int K, const double* a, const double* b, double* c)

where C(M,N) = C(M,N) + A(M,K) * B(K,N).

The library can be configured to accept Row-major or Column-major order matrices.
Change the variable ROW_MAJOR inside the Makefile file (1 for Row-major, 
Column-major otherwise).

The values of the matrix sizes (M,N,K values) can be set by changing the 
variables inside the Makefile file:

INDICES_M
INDICES_N
INDICES_K

For example:

INDICES_M := 2 4
INDICES_N := 1
INDICES_K := 2 4 5

it generates the (M,N,K) values:
(2,1,2), (2,1,4), (2,1,5)
(4,1,2), (4,1,4), (4,1,5)

The fallback for the library is DGEMM if it is called for not 
generated values of M, N, K.
