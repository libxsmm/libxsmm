## Matrix Multiplication

### Overview<a name="small-matrix-multiplication-smm"></a>

To perform the dense matrix-matrix multiplication <i>C<sub>m&#8239;x&#8239;n</sub> = alpha &middot; A<sub>m&#8239;x&#8239;k</sub> &middot; B<sub>k&#8239;x&#8239;n</sub> + beta &middot; C<sub>m&#8239;x&#8239;n</sub></i>, the full-blown GEMM interface can be treated with "default arguments" (which is deviating from the BLAS standard, however without compromising the binary compatibility). Default arguments are derived from compile-time constants (configurable) for historic reasons (LIBXSMM's "pre-JIT era").

```C
libxsmm_?gemm(NULL/*transa*/, NULL/*transb*/,
  &m/*required*/, &n/*required*/, &k/*required*/,
  NULL/*alpha*/, a/*required*/, NULL/*lda*/,
                 b/*required*/, NULL/*ldb*/,
   NULL/*beta*/, c/*required*/, NULL/*ldc*/);
```

For the C interface (with type prefix `s` or `d`), all arguments including m, n, and k are passed by pointer. This is needed for binary compatibility with the original GEMM/BLAS interface.

```C
libxsmm_gemm(NULL/*transa*/, NULL/*transb*/,
  m/*required*/, n/*required*/, k/*required*/,
  NULL/*alpha*/, a/*required*/, NULL/*lda*/,
                 b/*required*/, NULL/*ldb*/,
   NULL/*beta*/, c/*required*/, NULL/*ldc*/);
```

The C++ interface is also supplying overloaded versions where m, n, and k can be passed <span>by&#8209;value</span> (making it clearer that m, n, and k are non-optional arguments).

```FORTRAN
! Dense matrix multiplication (single/double-precision).
CALL libxsmm_?gemm(m=m, n=n, k=k, a=a, b=b, c=c)
! Dense matrix multiplication (generic interface).
CALL libxsmm_gemm(m=m, n=n, k=k, a=a, b=b, c=c)
```

The FORTRAN interface supports optional arguments (without affecting the binary compatibility with the original BLAS interface) by allowing to omit arguments where the C/C++ interface allows for NULL to be passed.

```C
/** Dense matrix multiplication (single/double-precision). */
libxsmm_blas_?gemm(NULL/*transa*/, NULL/*transb*/,
  &m/*required*/, &n/*required*/, &k/*required*/,
  NULL/*alpha*/, a/*required*/, NULL/*lda*/,
                 b/*required*/, NULL/*ldb*/,
   NULL/*beta*/, c/*required*/, NULL/*ldc*/);
```

For convenience, a BLAS-based dense matrix multiplication (`libxsmm_blas_gemm`) is provided for all supported languages. This only re-exposes the underlying GEMM/BLAS implementation, but the interface accepts optional arguments (or NULL pointers in C) where the regular GEMM expects a value. To remove any BLAS-dependency, please follow the [Link Instructions](index.md#link-instructions). A BLAS-based GEMM can be useful for validation/benchmark purposes, and more important as a fallback when building an application-specific dispatch mechanism.

### Manual Code Dispatch

Successively calling a kernel (i.e., multiple times) allows for amortizing the cost of the code dispatch. Moreover, to customize the dispatch mechanism, one can rely on the following interface.

```C
/** Call dispatched (*function_ptr)(a, b, c [, pa, pb, pc]). */
libxsmm_[s|d]mmfunction libxsmm_[type-prefix]mmdispatch(
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  /** NULL: tight fit (m) */ const libxsmm_blasint* lda,
  /** NULL: tight fit (k) */ const libxsmm_blasint* ldb,
  /** NULL: tight fit (m) */ const libxsmm_blasint* ldc,
  /** NULL: LIBXSMM_ALPHA */ const type* alpha,
  /** NULL: LIBXSMM_BETA  */ const type* beta,
  /** NULL: LIBXSMM_FLAGS */ const int* flags,
  /** NULL: LIBXSMM_PREFETCH_NONE (not LIBXSMM_PREFETCH!) */
  const int* prefetch);
```

Overloaded function signatures are provided and allow to omit arguments (C++ and FORTRAN), which are then derived from the [configurable defaults](https://github.com/libxsmm/libxsmm/blob/main/include/libxsmm_config.h). In C++, `libxsmm_mmfunction<type>` can be used to instantiate a functor rather than making a distinction between numeric types per type-prefix. For lower precision GEMMs, `libxsmm_mmfunction<itype,otype=itype>` optionally takes a second type (output type).

```C
/* generates or dispatches the code specialization */
libxsmm_mmfunction<T> xmm(m, n, k);
if (xmm) { /* JIT'ted code */
  /* can be parallelized per, e.g., OpenMP */
  for (int i = 0; i < n; ++i) {
    xmm(a+i*asize, b+i*bsize, c+i*csize);
  }
}
```

Similarly in FORTRAN (see [samples/utilities/smmbench/smm.f](https://github.com/libxsmm/libxsmm/blob/main/samples/utilities/smmbench/smm.f)), a generic interface (`libxsmm_mmdispatch`) can be used to dispatch a `LIBXSMM_?MMFUNCTION`. The handle encapsulated by such a `LIBXSMM_?MMFUNCTION` can be called per `libxsmm_call`. Beside of dispatching code, one can also call statically generated kernels (e.g., `libxsmm_dmm_4_4_4`) by using the prototype functions included with the FORTRAN and C/C++ interface. Prototypes are present whenever static code was requested at compile-time of the library (e.g. per `make MNK="1 2 3 4 5"`).

```FORTRAN
TYPE(LIBXSMM_DMMFUNCTION) :: xmm
CALL libxsmm_dispatch(xmm, m, n, k)
IF (libxsmm_available(xmm)) THEN
  DO i = LBOUND(c, 3), UBOUND(c, 3) ! consider OpenMP
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

Above, pointer-arguments of `libxsmm_dmmdispatch` can be NULL (or OPTIONAL in FORTRAN): for LDx this means a "tight" leading dimension, alpha, beta, and flags are given by a [default value](https://github.com/libxsmm/libxsmm/blob/main/include/libxsmm_config.h) (which is selected at compile-time), and for the prefetch strategy a NULL-argument refers to "no prefetch" (which is equivalent to an explicit `LIBXSMM_PREFETCH_NONE`). By design, the prefetch strategy can be changed at runtime (as soon as valid next-locations are used) without changing the call-site (kernel-signature with six arguments).

<a name="implicit-batches"></a>

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

To process a batch of matrix multiplications and to prefetch the operands of the next multiplication ahead of time, the code presented in the [Overview](#overview) section may be modified as shown above. The last multiplication is peeled from the main batch to avoid prefetching out-of-bounds (OOB). Prefetching from an invalid address does not trap an exception, but an (unnecessary) page fault can be avoided.

<a name="explicit-batch-interface"></a>

```C
/** Batched matrix multiplications (explicit data representation). */
int libxsmm_gemm_batch_task(libxsmm_datatype iprec, libxsmm_datatype oprec,
  const char* transa, const char* transb,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
                     const void* b, const libxsmm_blasint* ldb,
   const void* beta,       void* c, const libxsmm_blasint* ldc,
  libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[],
  const libxsmm_blasint stride_b[],
  const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize,
  int tid, int ntasks);
```

To further simplify the multiplication of matrices in a batch, LIBXSMM's batch interface can help to extract the necessary input from a variety of existing structures (integer indexes, array of pointers both with Byte sized strides). An expert interface (see above) can employ a user-defined threading runtime (`tid` and `ntasks`). In case of OpenMP, `libxsmm_gemm_batch_omp` is ready-to-use and hosted by the extension library (libxsmmext). Of course, `libxsmm_gemm_batch_omp` does not take `tid` and `ntasks` since both arguments are given by OpenMP. Similarly, a sequential version (shown below) is available per `libxsmm_gemm_batch` (libxsmm).

Please note that an explicit data representation should exist and reused rather than created only to call the explicit batch-interface. Creating such a data structure only for this matter can introduce an overhead which is hard to amortize (speedup). If no explicit data structure exists, a "chain" of multiplications can be often algorithmically described (see [self-hosted batch loop](#implicit-batches)).

```C
void libxsmm_gemm_batch(libxsmm_datatype iprec, libxsmm_datatype oprec,
  const char* transa, const char* transb,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
                     const void* b, const libxsmm_blasint* ldb,
   const void* beta,       void* c, const libxsmm_blasint* ldc,
  libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[],
  const libxsmm_blasint stride_b[],
  const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize);
```

<a name="blas-batch-interface"></a>In recent BLAS library implementations, `dgemm_batch` and `sgemm_batch` have been introduced. This BLAS(-like) interface allows for groups of homogeneous batches, which is like an additional loop around the interface as introduced above. On the other hand, the BLAS(-like) interface only supports arrays of pointers for the matrices. In contrast, above interface supports arrays of pointers as well as arrays of indexes plus a flexible way to extract data from arrays of structures (AoS). LIBXSMM also supports this (new) BLAS(-like) interface with `libxsmm_?gemm_batch` and `libxsmm_?gemm_batch_omp` (the latter of which relies on LIBXSMM/ext). Further, existing calls to `dgemm_batch` and `sgemm_batch` can be intercepted and replaced with [LIBXSMM's call wrapper](#call-wrapper). The signatures of `libxsmm_dgemm_batch` and `libxsmm_sgemm_batch` are equal except for the element type (`double` and `float` respectively).

```C
void libxsmm_dgemm_batch(const char transa_array[], const char transb_array[],
  const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const double alpha_array[], const double* a_array[], const libxsmm_blasint lda_array[],
                              const double* b_array[], const libxsmm_blasint ldb_array[],
  const double  beta_array[],       double* c_array[], const libxsmm_blasint ldc_array[],
  const libxsmm_blasint* group_count, const libxsmm_blasint group_size[]);
```

<a name="batch-sync"></a>**Note**: the multi-threaded implementation (`ntasks > 1` or "omp" form of the functions) avoids data races if indexes or pointers for the destination (C-)matrix are duplicated. This synchronization occurs automatically (`beta != 0`), but can be avoided by passing a negative `batchsize`, `group_size` and/or a negative `group_count`.

### User-Data Dispatch

It can be desired to dispatch user-defined data, i.e., to query a value based on a key. This functionality can be used to, e.g., dispatch multiple kernels in one step if a code location relies on multiple kernels. This way, one can pay the cost of dispatch one time per task rather than according to the number of JIT-kernels used by this task. This functionality is detailed in the section about [Service Functions](libxsmm_aux.md#user-data-dispatch).

### Call Wrapper

#### Overview

Since the library is binary compatible with existing GEMM calls (BLAS), such calls can be replaced at link-time or intercepted at runtime of an application such that LIBXSMM is used instead of the original BLAS library. There are two cases to consider: <span>(1)&#160;static</span> linkage, and <span>(2)&#160;dynamic</span> linkage of the application against the original BLAS library. When calls are intercepted, one can select a sequential (default) or an OpenMP-parallelized implementation (`make WRAP=2`).

**Note**: Intercepting GEMM calls is low effort but implies overhead, which can be relatively high for small-sized problems. LIBXSMM's native programming interface has lower overhead and can amortize overhead when using the same multiplication kernel in a consecutive fashion (and employ sophisticated data prefetch on top).

#### Static Linkage

An application which is linked statically against BLAS requires to wrap the `sgemm_` and the `dgemm_` symbol (an alternative is to wrap only `dgemm_`). To relink the application (without editing the build system) can often be accomplished by copying and pasting the linker command as it appeared in the console output of the build system, and then re-invoking a modified link step (please also consider `-Wl,--export-dynamic`).

```bash
gcc [...] -Wl,--wrap=dgemm_,--wrap=sgemm_ \
          /path/to/libxsmmext.a /path/to/libxsmm.a \
          /path/to/your_regular_blas.a
```

In addition, existing [BLAS(-like) batch-calls](#blas-batch-interface) can be intercepted as well:

```bash
gcc [...] -Wl,--wrap=dgemm_batch_,--wrap=sgemm_batch_ \
          -Wl,--wrap=dgemm_batch,--wrap=sgemm_batch \
          -Wl,--wrap=dgemm_,--wrap=sgemm_ \
          /path/to/libxsmmext.a /path/to/libxsmm.a \
          /path/to/your_regular_blas.a
```

Above, GEMM and GEMM_BATCH are intercepted both, however this can be chosen independently. For GEMM_BATCH the Fortran and C-form of the symbol may be intercepted both (regular GEMM can always be intercepted per `?gemm_` even when `?gemm` is used in C-code).

**Note**: The static link-time wrapper technique may only work with a GCC tool chain (<span>GNU&#160;Binutils</span>: `ld`, or `ld` via compiler-driver), and it has been tested with <span>GNU&#160;GCC</span>, <span>Intel&#160;Compiler</span>, and Clang. However, this does not work under Microsoft Windows (even when using the GNU tool chain or Cygwin).

#### Dynamic Linkage

An application that is dynamically linked against BLAS allows to intercept the GEMM calls at startup time (runtime) of the unmodified executable by using the LD_PRELOAD mechanism. The shared library of LIBXSMMext (`make STATIC=0`) can be used to intercept GEMM calls:

```bash
LD_LIBRARY_PATH=/path/to/libxsmm/lib:${LD_LIBRARY_PATH} \
LD_PRELOAD=libxsmmext.so \
   ./myapplication
```

