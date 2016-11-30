/******************************************************************************
** Copyright (c) 2014-2016, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_H
#define LIBXSMM_H

/** Name of the version (stringized set of version numbers). */
#define LIBXSMM_VERSION "$VERSION"
/** Name of the branch of which the version is derived from. */
#define LIBXSMM_BRANCH  "$BRANCH"
/** Major version based on the last reachable tag under RCS. */
#define LIBXSMM_VERSION_MAJOR $MAJOR
/** Minor version based on the last reachable tag of the RCS. */
#define LIBXSMM_VERSION_MINOR $MINOR
/** Update number based on the last reachable tag under RCS. */
#define LIBXSMM_VERSION_UPDATE $UPDATE
/** Patch number counting commits since the last version stamp. */
#define LIBXSMM_VERSION_PATCH $PATCH

#include "libxsmm_macros.h"
#include "libxsmm_typedefs.h"
#include "libxsmm_generator.h"
#include "libxsmm_frontend.h"
#include "libxsmm_malloc.h"
#include "libxsmm_spmdm.h"
#include "libxsmm_cpuid.h"
#include "libxsmm_timer.h"
#include "libxsmm_sync.h"
#include "libxsmm_dnn.h"

/** Integer type for LAPACK/BLAS (LP64: 32-bit, and ILP64: 64-bit). */
#if (0 != LIBXSMM_ILP64)
typedef long long libxsmm_blasint;
#else
typedef int libxsmm_blasint;
#endif

/** Initialize the library; pay for setup cost at a specific point. */
LIBXSMM_API void libxsmm_init(void);
/** De-initialize the library and free internal memory (optional). */
LIBXSMM_API void libxsmm_finalize(void);

/**
 * Returns the architecture and instruction set extension as determined by the CPUID flags, as set
 * by the libxsmm_get_target_arch* functions, or as set by the LIBXSMM_TARGET environment variable.
 */
LIBXSMM_API int libxsmm_get_target_archid(void);
/** Set target architecture (id: see libxsmm_typedefs.h) for subsequent code generation (JIT). */
LIBXSMM_API void libxsmm_set_target_archid(int id);

/**
 * Returns the name of the target architecture as determined by the CPUID flags, as set by the
 * libxsmm_get_target_arch* functions, or as set by the LIBXSMM_TARGET environment variable.
 */
LIBXSMM_API const char* libxsmm_get_target_arch(void);
/** Set target architecture (arch="0|sse|snb|hsw|knl|skx", NULL/"0": CPUID) for subsequent code generation (JIT). */
LIBXSMM_API void libxsmm_set_target_arch(const char* arch);

/** Get the level of verbosity. */
LIBXSMM_API int libxsmm_get_verbosity(void);
/**
 * Set the level of verbosity (0: off, positive value: verbosity level,
 * negative value: maximum verbosity, which also dumps JIT-code)
 */
LIBXSMM_API void libxsmm_set_verbosity(int level);

/** Get the default prefetch strategy. */
LIBXSMM_API libxsmm_gemm_prefetch_type libxsmm_get_gemm_auto_prefetch(void);
/** Set the default prefetch strategy. */
LIBXSMM_API void libxsmm_set_gemm_auto_prefetch(libxsmm_gemm_prefetch_type strategy);

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (descriptor form). */
LIBXSMM_API libxsmm_xmmfunction libxsmm_xmmdispatch(const libxsmm_gemm_descriptor* descriptor);

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (single-precision). */
LIBXSMM_API libxsmm_smmfunction libxsmm_smmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const float* alpha, const float* beta,
  const int* flags, const int* prefetch);
/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (double-precision). */
LIBXSMM_API libxsmm_dmmfunction libxsmm_dmmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const double* alpha, const double* beta,
  const int* flags, const int* prefetch);

/**
 * Code generation routine for the CSR format which multiplies a dense SOA matrix (each element holds a SIMD-width
 * wide vector) and a sparse matrix or a sparse matrix with a dense SOA matrix.
 * The result is always a SOA matrix. There is no code cache, and user code has to manage the code pointers.
 * Call libxsmm_release_kernel in order to deallocate the JIT'ted code.
 * @TODO: This is not great, probably need to declare values as void pointer
 */
LIBXSMM_API libxsmm_xmmfunction libxsmm_create_dcsr_soa(const libxsmm_gemm_descriptor* descriptor,
   const unsigned int* row_ptr, const unsigned int* column_idx, const double* values);

/** Deallocates the JIT'ted code as returned by libxsmm_create_* function. TODO: this is a no-op at the moment. */
LIBXSMM_API void libxsmm_release_kernel(const void* jit_code);

/** Matrix transposition (out-of-place form). */
LIBXSMM_API int libxsmm_otrans(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo);

/** Matrix transposition (out-of-place form, single-precision). */
LIBXSMM_API_INLINE int libxsmm_sotrans(float* out, const float* in,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
#if defined(LIBXSMM_BUILD)
;
#else
{ return libxsmm_otrans(out, in, sizeof(float), m, n, ldi, ldo); }
#endif

/** Matrix transposition (out-of-place form, double-precision). */
LIBXSMM_API_INLINE int libxsmm_dotrans(double* out, const double* in,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
#if defined(LIBXSMM_BUILD)
;
#else
{ return libxsmm_otrans(out, in, sizeof(double), m, n, ldi, ldo); }
#endif

/** Matrix transposition, which is multi-threadable using libxsmmext (out-of-place form). */
LIBXSMM_API int libxsmm_otrans_omp(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo);

/** Matrix transposition, which is multi-threadable (out-of-place form, single-precision). */
LIBXSMM_API int libxsmm_sotrans_omp(float* out, const float* in,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo);

/** Matrix transposition, which is multi-threadable (out-of-place form, double-precision). */
LIBXSMM_API int libxsmm_dotrans_omp(double* out, const double* in,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo);

/** Matrix transposition (in-place form). */
LIBXSMM_API int libxsmm_itrans(void* inout, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld);

/** Matrix transposition (in-place form, single-precision). */
LIBXSMM_API_INLINE int libxsmm_sitrans(float* inout,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld)
#if defined(LIBXSMM_BUILD)
;
#else
{ return libxsmm_itrans(inout, sizeof(float), m, n, ld); }
#endif

/** Matrix transposition (in-place form, double-precision). */
LIBXSMM_API_INLINE int libxsmm_ditrans(double* inout,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld)
#if defined(LIBXSMM_BUILD)
;
#else
{ return libxsmm_itrans(inout, sizeof(double), m, n, ld); }
#endif

/**
 * Utility function, which either prints information about the GEMM call
 * or dumps (FILE/ostream=0) all input and output data into MHD files.
 * The Meta Image Format (MHD) is suitable for visual inspection using e.g.,
 * ITK-SNAP or ParaView.
 */
LIBXSMM_API void libxsmm_gemm_print(void* ostream,
  libxsmm_gemm_xflags precision, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
  const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc);

/** Dispatched general dense matrix multiplication (single-precision); can be called from F77 code. */
LIBXSMM_API_INLINE void libxsmm_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
#if defined(LIBXSMM_BUILD)
;
#else
{ LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb);
  LIBXSMM_SGEMM(flags, *m, *n, *k,
    0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
    a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
    0 != beta ? *beta : ((float)LIBXSMM_BETA),
    c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
}
#endif

/** Dispatched general dense matrix multiplication (double-precision); can be called from F77 code. */
LIBXSMM_API_INLINE void libxsmm_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
#if defined(LIBXSMM_BUILD)
;
#else
{ LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb);
  LIBXSMM_DGEMM(flags, *m, *n, *k,
    0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
    a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
    0 != beta ? *beta : ((double)LIBXSMM_BETA),
    c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
}
#endif

/** Multi-threadable general dense matrix multiplication; requires linking libxsmmext (single-precision). */
LIBXSMM_API void libxsmm_sgemm_omp(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc);

/** Multi-threadable general dense matrix multiplication; requires linking libxsmmext (double-precision). */
LIBXSMM_API void libxsmm_dgemm_omp(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc);

/** General dense matrix multiplication based on LAPACK/BLAS (single-precision). */
LIBXSMM_API void libxsmm_blas_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc);

/** General dense matrix multiplication based on LAPACK/BLAS (double-precision). */
LIBXSMM_API void libxsmm_blas_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc);
$MNK_INTERFACE_LIST
#if defined(__cplusplus)

/** Construct and execute a specialized function. */
template<typename T> class LIBXSMM_RETARGETABLE libxsmm_mmfunction {};

/** Construct and execute a specialized function (single-precision). */
template<> class LIBXSMM_RETARGETABLE libxsmm_mmfunction<float> {
  mutable/*retargetable*/ libxsmm_smmfunction m_function;
public:
  libxsmm_mmfunction(): m_function(0) {}
  libxsmm_mmfunction(int m, int n, int k, int flags = LIBXSMM_FLAGS)
    : m_function(libxsmm_smmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, &flags, 0/*prefetch*/))
  {}
  libxsmm_mmfunction(int m, int n, int k, int lda, int ldb, int ldc, int flags = LIBXSMM_FLAGS)
    : m_function(libxsmm_smmdispatch(m, n, k, &lda, &ldb, &ldc, 0/*alpha*/, 0/*beta*/, &flags, 0/*prefetch*/))
  {}
  libxsmm_mmfunction(int flags, int m, int n, int k, int prefetch)
    : m_function(libxsmm_smmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, &flags, &prefetch))
  {}
  libxsmm_mmfunction(int flags, int m, int n, int k, int lda, int ldb, int ldc, int prefetch = LIBXSMM_PREFETCH)
    : m_function(libxsmm_smmdispatch(m, n, k, &lda, &ldb, &ldc, 0/*alpha*/, 0/*beta*/, &flags, &prefetch))
  {}
  libxsmm_mmfunction(int m, int n, int k, float alpha, float beta, int flags = LIBXSMM_FLAGS, int prefetch = LIBXSMM_PREFETCH)
    : m_function(libxsmm_smmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, &flags, &prefetch))
  {}
  libxsmm_mmfunction(int flags, int m, int n, int k, float alpha, float beta, int prefetch = LIBXSMM_PREFETCH)
    : m_function(libxsmm_smmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, &flags, &prefetch))
  {}
  libxsmm_mmfunction(int flags, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int prefetch = LIBXSMM_PREFETCH)
    : m_function(libxsmm_smmdispatch(m, n, k, &lda, &ldb, &ldc, &alpha, &beta, &flags, &prefetch))
  {}
  libxsmm_mmfunction(int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int flags = LIBXSMM_FLAGS, int prefetch = LIBXSMM_PREFETCH)
    : m_function(libxsmm_smmdispatch(m, n, k, &lda, &ldb, &ldc, &alpha, &beta, &flags, &prefetch))
  {}
public:
  operator libxsmm_smmfunction() const {
    return m_function;
  }
  void operator()(const float* a, const float* b, float* c) const {
    m_function(LIBXSMM_LD(a, b), LIBXSMM_LD(b, a), c);
  }
  void operator()(const float* a, const float* b, float* c,
    const float* pa, const float* pb, const float* pc) const
  {
    m_function(LIBXSMM_LD(a, b), LIBXSMM_LD(b, a), c,
      LIBXSMM_LD(pa, pb), LIBXSMM_LD(pb, pa), pc);
  }
};

/** Construct and execute a specialized function (double-precision). */
template<> class LIBXSMM_RETARGETABLE libxsmm_mmfunction<double> {
  mutable/*retargetable*/ libxsmm_dmmfunction m_function;
public:
  libxsmm_mmfunction(): m_function(0) {}
  libxsmm_mmfunction(int m, int n, int k, int flags = LIBXSMM_FLAGS)
    : m_function(libxsmm_dmmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, &flags, 0/*prefetch*/))
  {}
  libxsmm_mmfunction(int m, int n, int k, int lda, int ldb, int ldc, int flags = LIBXSMM_FLAGS)
    : m_function(libxsmm_dmmdispatch(m, n, k, &lda, &ldb, &ldc, 0/*alpha*/, 0/*beta*/, &flags, 0/*prefetch*/))
  {}
  libxsmm_mmfunction(int flags, int m, int n, int k, int prefetch)
    : m_function(libxsmm_dmmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, &flags, &prefetch))
  {}
  libxsmm_mmfunction(int flags, int m, int n, int k, int lda, int ldb, int ldc, int prefetch = LIBXSMM_PREFETCH)
    : m_function(libxsmm_dmmdispatch(m, n, k, &lda, &ldb, &ldc, 0/*alpha*/, 0/*beta*/, &flags, &prefetch))
  {}
  libxsmm_mmfunction(int m, int n, int k, double alpha, double beta, int flags = LIBXSMM_FLAGS, int prefetch = LIBXSMM_PREFETCH)
    : m_function(libxsmm_dmmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, &flags, &prefetch))
  {}
  libxsmm_mmfunction(int flags, int m, int n, int k, double alpha, double beta, int prefetch = LIBXSMM_PREFETCH)
    : m_function(libxsmm_dmmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, &flags, &prefetch))
  {}
  libxsmm_mmfunction(int flags, int m, int n, int k, int lda, int ldb, int ldc, double alpha, double beta, int prefetch = LIBXSMM_PREFETCH)
    : m_function(libxsmm_dmmdispatch(m, n, k, &lda, &ldb, &ldc, &alpha, &beta, &flags, &prefetch))
  {}
  libxsmm_mmfunction(int m, int n, int k, int lda, int ldb, int ldc, double alpha, double beta, int flags = LIBXSMM_FLAGS, int prefetch = LIBXSMM_PREFETCH)
    : m_function(libxsmm_dmmdispatch(m, n, k, &lda, &ldb, &ldc, &alpha, &beta, &flags, &prefetch))
  {}
public:
  operator libxsmm_dmmfunction() const {
    return m_function;
  }
  void operator()(const double* a, const double* b, double* c) const {
    LIBXSMM_MMCALL_ABC(m_function, a, b, c);
  }
  void operator()(const double* a, const double* b, double* c,
    const double* pa, const double* pb, const double* pc) const
  {
    LIBXSMM_MMCALL_PRF(m_function, a, b, c, pa, pb, pc);
  }
};

/** Matrix transposition (out-of-place form). */
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans(T* out, const T* in, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo) {
  return libxsmm_otrans(out, in, sizeof(T), m, n, ldi, ldo);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans(T* out, const T* in, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi) {
  return libxsmm_trans(out, in, m, n, ldi, ldi);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans(T* out, const T* in, libxsmm_blasint m, libxsmm_blasint n) {
  return libxsmm_trans(out, in, m, n, m);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans(T* out, const T* in, libxsmm_blasint n) {
  return libxsmm_trans(out, in, n, n);
}

/** Matrix transposition, which is multi-threadable (out-of-place form). */
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans_omp(T* out, const T* in, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo) {
  return libxsmm_otrans(out, in, sizeof(T), m, n, ldi, ldo);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans_omp(T* out, const T* in, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi) {
  return libxsmm_trans_omp(out, in, m, n, ldi, ldi);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans_omp(T* out, const T* in, libxsmm_blasint m, libxsmm_blasint n) {
  return libxsmm_trans_omp(out, in, m, n, m);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans_omp(T* out, const T* in, libxsmm_blasint n) {
  return libxsmm_trans_omp(out, in, n, n);
}

/** Matrix transposition (in-place form). */
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans(T* inout, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi) {
  return libxsmm_itrans(inout, sizeof(T), m, n, ldi);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans(T* inout, libxsmm_blasint m, libxsmm_blasint n) {
  return libxsmm_trans(inout, m, n, m);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans(T* inout, libxsmm_blasint n) {
  return libxsmm_trans(inout, n, n);
}

/** Dispatched general dense matrix multiplication (single-precision). */
inline LIBXSMM_RETARGETABLE void libxsmm_gemm(const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda, const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** Dispatched general dense matrix multiplication (double-precision). */
inline LIBXSMM_RETARGETABLE void libxsmm_gemm(const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda, const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** Dispatched general dense matrix multiplication (single-precision). */
inline LIBXSMM_RETARGETABLE void libxsmm_sgemm(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const float* alpha, const float* a, const libxsmm_blasint* lda, const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_sgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** Dispatched general dense matrix multiplication (double-precision). */
inline LIBXSMM_RETARGETABLE void libxsmm_dgemm(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const double* alpha, const double* a, const libxsmm_blasint* lda, const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  libxsmm_dgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** Dispatched general dense matrix multiplication (single-precision). */
inline LIBXSMM_RETARGETABLE void libxsmm_gemm(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const float* alpha, const float* a, const libxsmm_blasint* lda, const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** Dispatched general dense matrix multiplication (double-precision). */
inline LIBXSMM_RETARGETABLE void libxsmm_gemm(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const double* alpha, const double* a, const libxsmm_blasint* lda, const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (single-precision). */
inline LIBXSMM_RETARGETABLE void libxsmm_blas_gemm(const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda, const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (double-precision). */
inline LIBXSMM_RETARGETABLE void libxsmm_blas_gemm(const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda, const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (single-precision). */
inline LIBXSMM_RETARGETABLE void libxsmm_blas_sgemm(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const float* alpha, const float* a, const libxsmm_blasint* lda, const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_sgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (double-precision). */
inline LIBXSMM_RETARGETABLE void libxsmm_blas_dgemm(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const double* alpha, const double* a, const libxsmm_blasint* lda, const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_dgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (single-precision). */
inline LIBXSMM_RETARGETABLE void libxsmm_blas_gemm(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const float* alpha, const float* a, const libxsmm_blasint* lda, const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (double-precision). */
inline LIBXSMM_RETARGETABLE void libxsmm_blas_gemm(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const double* alpha, const double* a, const libxsmm_blasint* lda, const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#endif /*__cplusplus*/
#endif /*LIBXSMM_H*/

