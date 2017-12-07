## Matrix Multiplication

### Overview<a name="small-matrix-multiplication-smm"></a>

To perform the dense matrix-matrix multiplication *C<sub>m&#8239;x&#8239;n</sub> = alpha &middot; A<sub>m&#8239;x&#8239;k</sub> &middot; B<sub>k&#8239;x&#8239;n</sub> + beta &middot; C<sub>m&#8239;x&#8239;n</sub>*, the full-blown GEMM interface can be treated with "default arguments" (which is deviating from the BLAS standard, however without compromising the binary compatibility).

```C
/** Automatically dispatched dense matrix multiplication (single/double-precision, C code). */
libxsmm_?gemm(NULL/*transa*/, NULL/*transb*/, &m/*required*/, &n/*required*/, &k/*required*/,
  NULL/*alpha*/, a/*required*/, NULL/*lda*/, b/*required*/, NULL/*ldb*/,
  NULL/*beta*/, c/*required*/, NULL/*ldc*/);
/** Automatically dispatched dense matrix multiplication (C++ code). */
libxsmm_gemm(NULL/*transa*/, NULL/*transb*/, m/*required*/, n/*required*/, k/*required*/,
  NULL/*alpha*/, a/*required*/, NULL/*lda*/, b/*required*/, NULL/*ldb*/,
  NULL/*beta*/, c/*required*/, NULL/*ldc*/);
```

For the C interface (with type prefix 's' or 'd'), all arguments including m, n, and k are passed by pointer. This is needed for binary compatibility with the original GEMM/BLAS interface. The C++ interface is also supplying overloaded versions where m, n, and k can be passed by&#8209;value (making it clearer that m, n, and k are non-optional arguments).

The FORTRAN interface supports optional arguments (without affecting the binary compatibility with the original BLAS interface) by allowing to omit arguments where the C/C++ interface allows for NULL to be passed.

```FORTRAN
! Automatically dispatched dense matrix multiplication (single/double-precision).
CALL libxsmm_?gemm(m=m, n=n, k=k, a=a, b=b, c=c)
! Automatically dispatched dense matrix multiplication (generic interface).
CALL libxsmm_gemm(m=m, n=n, k=k, a=a, b=b, c=c)
```

For convenience, a BLAS-based dense matrix multiplication (`libxsmm_blas_gemm`) is provided for all supported languages which is simply re-exposing the underlying GEMM/BLAS implementation. The BLAS-based GEMM might be useful for validation/benchmark purposes, and more important as a fallback when building an application-specific dispatch mechanism.

```C
/** Automatically dispatched dense matrix multiplication (single/double-precision). */
libxsmm_blas_?gemm(NULL/*transa*/, NULL/*transb*/, &m/*required*/, &n/*required*/, &k/*required*/,
  NULL/*alpha*/, a/*required*/, NULL/*lda*/, b/*required*/, NULL/*ldb*/,
  NULL/*beta*/, c/*required*/, NULL/*ldc*/);
```

A more recently added variant of matrix multiplication is parallelized based on the OpenMP standard. These routines will open an internal parallel region and rely on "classic" thread-based OpenMP. If these routines are called from inside of a parallel region, the parallelism will be based on tasks (OpenMP&#160;3.0). Please note that all OpenMP-based routines are hosted by the extension library (libxsmmext), which keeps the main library agnostic with respect to a threading runtime.

```C
/** OpenMP parallelized dense matrix multiplication (single/double-precision). */
libxsmm_?gemm_omp(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
```

### Manual Code Dispatch

Successively calling a kernel (i.e., multiple times) allows for amortizing the cost of the code dispatch. Moreover, to customize the dispatch mechanism, one can rely on the following interface.

```C
/** If non-zero function pointer is returned, call (*function_ptr)(a, b, c [, pa, pb, pc]). */
libxsmm_dmmfunction libxsmm_dmmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const double* alpha, const double* beta, const int* flags, const int* prefetch);
/** If non-zero function pointer is returned, call (*function_ptr)(a, b, c [, pa, pb, pc]). */
libxsmm_smmfunction libxsmm_smmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const float* alpha, const float* beta, const int* flags, const int* prefetch);
/** If non-zero function pointer is returned, call (*function_ptr)(a, b, c [, pa, pb, pc]). */
libxsmm_smmfunction libxsmm_wmmdispatch(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const int* alpha, const int* beta, const int* flags, const int* prefetch);
```

Overloaded function signatures are provided and allow to omit arguments (C++ and FORTRAN), which are then derived from the [configurable defaults](https://github.com/hfp/libxsmm/blob/master/src/template/libxsmm_config.h). In C++, `libxsmm_mmfunction<type>` can be used to instantiate a functor rather than making a distinction between numeric types per type-prefix.

```C
libxsmm_mmfunction<T> xmm(m, n, k); /* generates or dispatches the code specialization */
if (xmm) { /* JIT'ted code */
  for (int i = 0; i < n; ++i) { /* perhaps OpenMP parallelized */
    xmm(a+i*asize, b+i*bsize, c+i*csize); /* already dispatched */
  }
}
```

Similarly in FORTRAN (see [samples/smm/smm.f](https://github.com/hfp/libxsmm/blob/master/samples/smm/smm.f)), a generic interface (`libxsmm_mmdispatch`) can be used to dispatch a `LIBXSMM_?MMFUNCTION`. The handle encapsulated such a `LIBXSMM_?MMFUNCTION` can be called per `libxsmm_call`. Beside of dispatching code, one can also call statically generated kernels (e.g., `libxsmm_dmm_4_4_4`) by using the prototype functions included with the FORTRAN and C/C++ interface.

```FORTRAN
TYPE(LIBXSMM_DMMFUNCTION) :: xmm
CALL libxsmm_dispatch(xmm, m, n, k)
IF (libxsmm_available(xmm)) THEN
  DO i = LBOUND(c, 3), UBOUND(c, 3) ! perhaps OpenMP parallelized
    CALL libxsmm_dmmcall(xmm, a(:,:,i), b(:,:,i), c(:,:,i))
  END DO
END IF
```

### Batched Multiplication

In case of batched SMMs, it can be beneficial to supply "next locations" such that the upcoming operands are prefetched ahead of time. Such a location would be the address of the next matrix to be multiplied (and not any of the floating-point elements within the "current" matrix-operand). The "prefetch strategy" is requested at dispatch-time of a kernel. A [strategy](libxsmm_be.md#prefetch-strategy) other than `LIBXSMM_PREFETCH_NONE` turns the signature of a JIT'ted kernel into a function with six arguments (`a,b,c, pa,pb,pc` instead of `a,b,c`). To defer the decision about the strategy to a CPUID-based mechanism, one can choose `LIBXSMM_PREFETCH_AUTO`.

```C
int prefetch = LIBXSMM_PREFETCH_AUTO;
int flags = 0; /* LIBXSMM_FLAGS */
libxsmm_dmmfunction xmm = NULL;
double alpha = 1, beta = 0;
xmm = libxsmm_dmmdispatch(23/*m*/, 23/*n*/, 23/*k*/,
  NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/,
  &alpha, &beta, &flags, &prefetch);
```

Above, pointer-arguments of `libxsmm_dmmdispatch` can be NULL (or OPTIONAL in FORTRAN): for LDx this means a "tight" leading dimension, alpha, beta, and flags are given by a [default value](https://github.com/hfp/libxsmm/blob/master/src/template/libxsmm_config.h) (which is selected at compile-time), and for the prefetch strategy a NULL-argument refers to "no prefetch" (which is equivalent to an explicit `LIBXSMM_PREFETCH_NONE`). By design, the prefetch strategy can be changed at runtime (as soon as valid next-locations are used) without changing the call-site (kernel-signature with six arguments).

```C
if (0 < n) { /* check that n is at least 1 */
# pragma parallel omp private(i)
  for (i = 0; i < (n - 1); ++i) {
    const double *const ai = a + i * asize;
    const double *const bi = b + i * bsize;
    double *const ci = c + i * csize;
    xmm(ai, bi, ci, ai + asize, bi + bsize, ci + csize);
  }
  xmm(a + (n - 1) * asize, b + (n - 1) * bsize, c + (n - 1) * csize,
  /* pseudo prefetch for last element of batch (avoids page fault) */
      a + (n - 1) * asize, b + (n - 1) * bsize, c + (n - 1) * csize);
}
```

To process a batch of matrix multiplications and to prefetch the operands of the next multiplication ahead of time, the code presented in the [Overview](#overview) section may be modified as shown above. The last multiplication is peeled off from the batch to avoid prefetching out-of-bounds (OOB). Prefetching from an invalid address does not trap an exception, but an (unnecessary) page fault can be avoided as shown above.

```C
/** Process a series of matrix multiplications (explicit data representation). */
int libxsmm_mmbatch(libxsmm_xmmfunction kernel, libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const void* a, const void* b, void* c, libxsmm_blasint batchsize, int tid, int nthreads);
```

To further simplify the multiplication of matrices in a batch, the above interface can help if an explicit data representation is available. This low-level form is also able to employ a user-defined threading runtime. In case of OpenMP, `libxsmm_mmbatch_omp` is ready to use and hosted by the extension library (libxsmmext). An even higher-level set of procedures (and potentially more convenient functions) are available with `libxsmm_gemm_batch` and `libxsmm_gemm_batch_omp`.

```C
void libxsmm_gemm_batch(libxsmm_gemm_precision precision, const char* transa, const char* transb,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
                     const void* b, const libxsmm_blasint* ldb,
   const void* beta,       void* c, const libxsmm_blasint* ldc,
  libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize);
```

Please note that an explicit data representation is not actually necessary to process a series of matrix multiplications. A "chain" of multiplications can be programmatically described without the need for arrays of operands or indexes.

### Call Wrapper

#### Overview

Since the library is binary compatible with existing GEMM calls (BLAS), such calls can be replaced at link-time or intercepted at runtime of an application such that LIBXSMM is used instead of the original BLAS library. There are two cases to consider: (1)&#160;static linkage, and (2)&#160;dynamic linkage of the application against the original BLAS library. When calls are intercepted, one can select which problem sizes are handled (ignored). By default, only small GEMMs are handled by LIBXSMM (LIBXSMM_GEMM_WRAP=1). Of course, one can handle all problem sizes (LIBXSMM_GEMM_WRAP=2, which denotes "||" for "parallel"), and the default behavior can be also adjusted at compile-time e.g., `make GEMM=2` to handle larger problem-sizes in an OpenMP-parallel fashion.

```bash
LIBXSMM STATISTIC: 1000 multiplications
dgemm(trans=NN mnk=32,32,21 ldx=32,21,32 a,b=1,0): 8% [main$omp$1]
dgemm(trans=NN mnk=32,21,32 ldx=32,32,32 a,b=1,0): 8% [main$omp$1]
dgemm(trans=NN mnk=10,21,32 ldx=10,32,10 a,b=1,0): 5% [main$omp$1]
dgemm(trans=NN mnk=32,10,32 ldx=32,32,32 a,b=1,0): 5% [main$omp$1]
dgemm(trans=NN mnk=32,32,10 ldx=32,10,32 a,b=1,0): 5% [main$omp$1]
```

Intercepted GEMMs can also build a sophisticated statistic (histogram) with LIBXSMM_VERBOSE=3 (or higher). The histogram displays the call sites (debug symbol name) of all intercepted GEMMs ([example](https://github.com/hfp/libxsmm/blob/master/samples/wrap/autobatch.c) above depicts an OpenMP region hosted by the main function). With level&#160;4 (or higher), the histogram yields the entire content, and eventually less relevant entries are not pruned. An application must be built with symbols (`-g`) and export symbols similar to shared libraries (`-Wl,--export-dynamic` even when linked statically) in order to display the symbol names of where the GEMMs originated (call site).

**NOTE**: Intercepting GEMM calls is low effort but implies overhead, which can be relatively high for small problem sizes. LIBXSMM's native programming interface has lower overhead and allows to amortize this overhead when using the same multiplication kernel in a consecutive fashion along with sophisticated data prefetch.

#### Static Linkage

An application which is linked statically against BLAS requires to wrap the 'sgemm_' and the 'dgemm_' symbol (an alternative is to wrap only 'dgemm_'), and a special build of the libxsmm(ext) library is required (`make WRAP=1` to wrap SGEMM and DGEMM, or `make WRAP=2` to wrap only DGEMM). To relink the application (without editing the build system) can often be accomplished by copying and pasting the linker command as it appeared in the console output of the build system, and then re-invoking a modified link step:

```bash
gcc [...] -Wl,--wrap=sgemm_,--wrap=dgemm_ \
          /path/to/libxsmmext.a /path/to/libxsmm.a \
          /path/to/your_regular_blas.a
```

**NOTE**: The static link-time wrapper technique may only work with a GCC tool chain (GNU Binutils: `ld`, or `ld` via compiler-driver), and it has been tested with GNU GCC, Intel&#160;Compiler, and Clang. However, this does not work under Microsoft Windows (even when using the GNU tool chain or Cygwin), and it may not work under OS&#160;X (Compiler&#160;6.1 or earlier, later versions have not been tested).

#### Dynamic Linkage

An application that is dynamically linked against BLAS allows to intercept the GEMM calls at startup time (runtime) of the unmodified executable by using the LD_PRELOAD mechanism. The shared library of LIBXSMMext (`make STATIC=0`) can be used to intercept GEMM calls:

```bash
LD_PRELOAD=/path/to/libxsmm/lib/libxsmmext.so \
LD_LIBRARY_PATH=/path/to/libxsmm/lib:${LD_LIBRARY_PATH} \
   ./myapplication
```

