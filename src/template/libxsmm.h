/******************************************************************************
** Copyright (c) 2014-2017, Intel Corporation                                **
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

#include "libxsmm_config.h"

/**
 * Strings to denote the version of LIBXSMM (libxsmm_config.h).
 * LIBXSMM_VERSION: Name of the version (stringized version numbers).
 * LIBXSMM_BRANCH:  Name of the branch this version is derived from.
 */
#define LIBXSMM_VERSION LIBXSMM_CONFIG_VERSION
#define LIBXSMM_BRANCH  LIBXSMM_CONFIG_BRANCH

/**
 * Numbers to denote the version of LIBXSMM (libxsmm_config.h).
 * LIBXSMM_VERSION_MAJOR:  Major version derived from the most recent RCS-tag.
 * LIBXSMM_VERSION_MINOR:  Minor version derived from the most recent RCS-tag.
 * LIBXSMM_VERSION_UPDATE: Update number derived from the most recent RCS-tag.
 * LIBXSMM_VERSION_PATCH:  Patch number based on distance to most recent RCS-tag.
 */
#define LIBXSMM_VERSION_MAJOR  LIBXSMM_CONFIG_VERSION_MAJOR
#define LIBXSMM_VERSION_MINOR  LIBXSMM_CONFIG_VERSION_MINOR
#define LIBXSMM_VERSION_UPDATE LIBXSMM_CONFIG_VERSION_UPDATE
#define LIBXSMM_VERSION_PATCH  LIBXSMM_CONFIG_VERSION_PATCH

#include "libxsmm_macros.h"
#include "libxsmm_typedefs.h"
#include "libxsmm_generator.h"
#include "libxsmm_frontend.h"
#include "libxsmm_bgemm.h"
#include "libxsmm_fsspmdm.h"
#include "libxsmm_malloc.h"
#include "libxsmm_spmdm.h"
#include "libxsmm_cpuid.h"
#include "libxsmm_timer.h"
#include "libxsmm_sync.h"
#include "libxsmm_dnn.h"


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

/** Query the try-lock property of the code registry. */
LIBXSMM_API int libxsmm_get_dispatch_trylock(void);
/** Set the try-lock property of the code registry. */
LIBXSMM_API void libxsmm_set_dispatch_trylock(int trylock);

/** Get the default prefetch strategy. */
LIBXSMM_API libxsmm_gemm_prefetch_type libxsmm_get_gemm_auto_prefetch(void);
/** Set the default prefetch strategy. */
LIBXSMM_API void libxsmm_set_gemm_auto_prefetch(libxsmm_gemm_prefetch_type strategy);

/** Get information about the code registry. */
LIBXSMM_API int libxsmm_get_registry_info(libxsmm_registry_info* info);

/** Initialize GEMM descriptor (similar to the LIBXSMM_GEMM_DESCRIPTOR macros). */
LIBXSMM_API int libxsmm_gemm_descriptor_init(libxsmm_gemm_descriptor* descriptor,
  libxsmm_gemm_precision precision, int m, int n, int k, const int* lda, const int* ldb, const int* ldc,
  const void* alpha, const void* beta, const int* flags, const int* prefetch);

/** DEPRECATED: use libxsmm_gemm_descriptor_init instead
 * Create a GEMM descriptor object (dynamically allocates memory). Use this function
 * for binding a language where the libxsmm_gemm_descriptor type is not convenient.
 */
LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_create_dgemm_descriptor(char transa, char transb,
  int m, int n, int k, int lda, int ldb, int ldc, double alpha, double beta,
  libxsmm_gemm_prefetch_type/*int*/ strategy);

/** DEPRECATED: see libxsmm_create_dgemm_descriptor
 * Release a GEMM descriptor object.
 */
LIBXSMM_API void libxsmm_release_gemm_descriptor(const libxsmm_gemm_descriptor* descriptor);

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
/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (low/short-precision). */
LIBXSMM_API libxsmm_wmmfunction libxsmm_wmmdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const int* alpha, const int* beta,
  const int* flags, const int* prefetch);

/**
 * This function is a no-op unless LIBXSMM is built to intercept GEMM calls.
 * Pointer arguments are used to prefilter intercepted GEMM calls such that
 * non-NULL values match. Otherwise (NULL) the respective argument is
 * considered a "free value" i.e., every value can match.
 */
LIBXSMM_API void libxsmm_mmbatch_begin(libxsmm_gemm_precision precision, const int* flags,
  const int* m, const int* n, const int* k, const int* lda, const int* ldb, const int* ldc,
  const void* alpha, const void* beta);

/** Processes the batch of previously recorded matrix multiplications (libxsmm_mmbatch_begin). */
LIBXSMM_API int libxsmm_mmbatch_end(/*TODO: signature*/);

/** Code generation routine for JIT matcopy using a descriptor. */
LIBXSMM_API libxsmm_xmatcopyfunction libxsmm_xmatcopydispatch(const libxsmm_matcopy_descriptor* descriptor);

/** Code generation routine for JIT transposes using a descriptor */
LIBXSMM_API libxsmm_xtransfunction libxsmm_xtransdispatch(const libxsmm_transpose_descriptor* descriptor);

/**
 * Code generation routine for the CSR format which multiplies a dense SOA matrix (each element holds a SIMD-width
 * wide vector) and a sparse matrix or a sparse matrix with a dense SOA matrix.
 * The result is always a SOA matrix. There is no code cache, and user code has to manage the code pointers.
 * Call libxsmm_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXSMM_API libxsmm_xmmfunction libxsmm_create_xcsr_soa(const libxsmm_gemm_descriptor* descriptor,
   const unsigned int* row_ptr, const unsigned int* column_idx, const void* values);

/**
 * Code generation routine for the CSR format which multiplies a dense matrix B into a dense matrix C.
 * The sparse matrix a is kept in registers.
 * Call libxsmm_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXSMM_API libxsmm_dmmfunction libxsmm_create_dcsr_reg(const libxsmm_gemm_descriptor* descriptor,
   const unsigned int* row_ptr, const unsigned int* column_idx, const double* values);

/**
 * Code generation routine for the CSR format which multiplies a dense matrix B into a dense matrix C.
 * The sparse matrix a is kept in registers.
 * Call libxsmm_release_kernel in order to deallocate the JIT'ted code.
 */
LIBXSMM_API libxsmm_smmfunction libxsmm_create_scsr_reg(const libxsmm_gemm_descriptor* descriptor,
   const unsigned int* row_ptr, const unsigned int* column_idx, const float* values);

/** Deallocates the JIT'ted code as returned by libxsmm_create_* function. TODO: this is a no-op at the moment. */
LIBXSMM_API void libxsmm_release_kernel(const void* jit_code);

/** Matrix copy function ("in" can be NULL to zero the destination). */
LIBXSMM_API int libxsmm_matcopy(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  const int* prefetch);

/** Matrix copy function ("in" can be NULL to zero the destination, per-thread form). */
LIBXSMM_API int libxsmm_matcopy_thread(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  const int* prefetch, int tid, int nthreads);

/** Matrix copy function ("in" can be NULL to zero the destination); MT via libxsmmext. */
LIBXSMM_API int libxsmm_matcopy_omp(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  const int* prefetch);

/** Matrix transposition (out-of-place form). */
LIBXSMM_API int libxsmm_otrans(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo);

/** Matrix transposition (out-of-place form, per-thread form). */
LIBXSMM_API int libxsmm_otrans_thread(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo,
  int tid, int nthreads);

  /** Matrix transposition; MT via libxsmmext (out-of-place form). */
LIBXSMM_API int libxsmm_otrans_omp(void* out, const void* in, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo);

/** Matrix transposition (in-place form). */
LIBXSMM_API int libxsmm_itrans(void* inout, unsigned int typesize,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld);

/** Dispatched general dense matrix multiplication (single-precision); can be called from F77 code. */
LIBXSMM_API void libxsmm_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc);

/** Dispatched general dense matrix multiplication (double-precision); can be called from F77 code. */
LIBXSMM_API void libxsmm_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc);

/** General dense matrix multiplication; MT via libxsmmext (single-precision). */
LIBXSMM_API void libxsmm_sgemm_omp(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc);

/** General dense matrix multiplication; MT via libxsmmext (double-precision). */
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

template<typename T> struct libxsmm_gemm_precision_enum {};
template<> struct libxsmm_gemm_precision_enum<double>         { static libxsmm_gemm_precision value() { return LIBXSMM_GEMM_PRECISION_F64; } };
template<> struct libxsmm_gemm_precision_enum<float>          { static libxsmm_gemm_precision value() { return LIBXSMM_GEMM_PRECISION_F32; } };
template<> struct libxsmm_gemm_precision_enum<signed short>   { static libxsmm_gemm_precision value() { return LIBXSMM_GEMM_PRECISION_I16; } };
template<> struct libxsmm_gemm_precision_enum<unsigned short> { static libxsmm_gemm_precision value() { return LIBXSMM_GEMM_PRECISION_I16; } };

/** Construct and execute a specialized function. */
template<typename T> class LIBXSMM_RETARGETABLE libxsmm_mmfunction {
  mutable/*retargetable*/ libxsmm_xmmfunction m_function;
public:
  libxsmm_mmfunction() { m_function.smm = 0; }
  libxsmm_mmfunction(int m, int n, int k, int flags = LIBXSMM_FLAGS) {
    libxsmm_gemm_descriptor descriptor;
    if (EXIT_SUCCESS == libxsmm_gemm_descriptor_init(&descriptor, libxsmm_gemm_precision_enum<T>::value(),
      m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, &flags, 0/*prefetch*/))
    {
      m_function = libxsmm_xmmdispatch(&descriptor);
    }
    else {
      m_function.smm = 0;
    }
  }
  libxsmm_mmfunction(int flags, int m, int n, int k, int prefetch) {
    libxsmm_gemm_descriptor descriptor;
    if (EXIT_SUCCESS == libxsmm_gemm_descriptor_init(&descriptor, libxsmm_gemm_precision_enum<T>::value(),
      m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, &flags, &prefetch))
    {
      m_function = libxsmm_xmmdispatch(&descriptor);
    }
    else {
      m_function.smm = 0;
    }
  }
  libxsmm_mmfunction(int flags, int m, int n, int k, float alpha, float beta) {
    libxsmm_gemm_descriptor descriptor;
    if (EXIT_SUCCESS == libxsmm_gemm_descriptor_init(&descriptor, libxsmm_gemm_precision_enum<T>::value(),
      m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, &flags, 0/*prefetch*/))
    {
      m_function = libxsmm_xmmdispatch(&descriptor);
    }
    else {
      m_function.smm = 0;
    }
  }
  libxsmm_mmfunction(int flags, int m, int n, int k, float alpha, float beta, int prefetch) {
    libxsmm_gemm_descriptor descriptor;
    if (EXIT_SUCCESS == libxsmm_gemm_descriptor_init(&descriptor, libxsmm_gemm_precision_enum<T>::value(),
      m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, &flags, &prefetch))
    {
      m_function = libxsmm_xmmdispatch(&descriptor);
    }
    else {
      m_function.smm = 0;
    }
  }
  libxsmm_mmfunction(int flags, int m, int n, int k, int lda, int ldb, int ldc, int prefetch) {
    libxsmm_gemm_descriptor descriptor;
    if (EXIT_SUCCESS == libxsmm_gemm_descriptor_init(&descriptor, libxsmm_gemm_precision_enum<T>::value(),
      m, n, k, &lda, &ldb, &ldc, 0/*alpha*/, 0/*beta*/, &flags, &prefetch))
    {
      m_function = libxsmm_xmmdispatch(&descriptor);
    }
    else {
      m_function.smm = 0;
    }
  }
  libxsmm_mmfunction(int flags, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta) {
    libxsmm_gemm_descriptor descriptor;
    if (EXIT_SUCCESS == libxsmm_gemm_descriptor_init(&descriptor, libxsmm_gemm_precision_enum<T>::value(),
      m, n, k, &lda, &ldb, &ldc, &alpha, &beta, &flags, 0/*prefetch*/))
    {
      m_function = libxsmm_xmmdispatch(&descriptor);
    }
    else {
      m_function.smm = 0;
    }
  }
  libxsmm_mmfunction(int flags, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int prefetch) {
    libxsmm_gemm_descriptor descriptor;
    if (EXIT_SUCCESS == libxsmm_gemm_descriptor_init(&descriptor, libxsmm_gemm_precision_enum<T>::value(),
      m, n, k, &lda, &ldb, &ldc, &alpha, &beta, &flags, &prefetch))
    {
      m_function = libxsmm_xmmdispatch(&descriptor);
    }
    else {
      m_function.smm = 0;
    }
  }
public:
  operator libxsmm_xmmfunction() const {
    return m_function;
  }
  void operator()(const T* a, const T* b, void* c) const {
    LIBXSMM_MMCALL_ABC(m_function, a, b, c);
  }
  void operator()(const T* a, const T* b, void* c, const T* pa, const T* pb, const void* pc) const {
    LIBXSMM_MMCALL_PRF(m_function, a, b, c, pa, pb, pc);
  }
};

/** Construct and execute a specialized function (single-precision). */
template<> class LIBXSMM_RETARGETABLE libxsmm_mmfunction<float> {
  mutable/*retargetable*/ libxsmm_smmfunction m_function;
#if defined(LIBXSMM_FALLBACK_SMMFUNCTION)
  int m_flags, m_m, m_n, m_k, m_lda, m_ldb, m_ldc;
  float m_alpha, m_beta;
#endif
public:
  libxsmm_mmfunction(): m_function(0)
#if defined(LIBXSMM_FALLBACK_SMMFUNCTION)
    , m_flags(LIBXSMM_FLAGS), m_m(0), m_n(0), m_k(0), m_lda(0), m_ldb(0), m_ldc(0), m_alpha(LIBXSMM_ALPHA), m_beta(LIBXSMM_BETA)
#endif
  {}
  libxsmm_mmfunction(int m, int n, int k, int flags = LIBXSMM_FLAGS)
    : m_function(libxsmm_smmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, &flags, 0/*prefetch*/))
#if defined(LIBXSMM_FALLBACK_SMMFUNCTION)
    , m_flags(flags), m_m(m), m_n(n), m_k(k), m_lda(m), m_ldb(k), m_ldc(m), m_alpha(LIBXSMM_ALPHA), m_beta(LIBXSMM_BETA)
#endif
  {}
  libxsmm_mmfunction(int flags, int m, int n, int k, int prefetch)
    : m_function(libxsmm_smmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, &flags, &prefetch))
#if defined(LIBXSMM_FALLBACK_SMMFUNCTION)
    , m_flags(flags), m_m(m), m_n(n), m_k(k), m_lda(m), m_ldb(k), m_ldc(m), m_alpha(LIBXSMM_ALPHA), m_beta(LIBXSMM_BETA)
#endif
  {}
  libxsmm_mmfunction(int flags, int m, int n, int k, float alpha, float beta)
    : m_function(libxsmm_smmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, &flags, 0/*prefetch*/))
#if defined(LIBXSMM_FALLBACK_SMMFUNCTION)
    , m_flags(flags), m_m(m), m_n(n), m_k(k), m_lda(m), m_ldb(k), m_ldc(m), m_alpha(alpha), m_beta(beta)
#endif
  {}
  libxsmm_mmfunction(int flags, int m, int n, int k, float alpha, float beta, int prefetch)
    : m_function(libxsmm_smmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, &flags, &prefetch))
#if defined(LIBXSMM_FALLBACK_SMMFUNCTION)
    , m_flags(flags), m_m(m), m_n(n), m_k(k), m_lda(m), m_ldb(k), m_ldc(m), m_alpha(alpha), m_beta(beta)
#endif
  {}
  libxsmm_mmfunction(int flags, int m, int n, int k, int lda, int ldb, int ldc, int prefetch)
    : m_function(libxsmm_smmdispatch(m, n, k, &lda, &ldb, &ldc, 0/*alpha*/, 0/*beta*/, &flags, &prefetch))
#if defined(LIBXSMM_FALLBACK_SMMFUNCTION)
    , m_flags(flags), m_m(m), m_n(n), m_k(k), m_lda(lda), m_ldb(ldb), m_ldc(ldc), m_alpha(LIBXSMM_ALPHA), m_beta(LIBXSMM_BETA)
#endif
  {}
  libxsmm_mmfunction(int flags, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta)
    : m_function(libxsmm_smmdispatch(m, n, k, &lda, &ldb, &ldc, &alpha, &beta, &flags, 0/*prefetch*/))
#if defined(LIBXSMM_FALLBACK_SMMFUNCTION)
    , m_flags(flags), m_m(m), m_n(n), m_k(k), m_lda(m), m_ldb(k), m_ldc(m), m_alpha(alpha), m_beta(beta)
#endif
  {}
  libxsmm_mmfunction(int flags, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int prefetch)
    : m_function(libxsmm_smmdispatch(m, n, k, &lda, &ldb, &ldc, &alpha, &beta, &flags, &prefetch))
#if defined(LIBXSMM_FALLBACK_SMMFUNCTION)
    , m_flags(flags), m_m(m), m_n(n), m_k(k), m_lda(m), m_ldb(k), m_ldc(m), m_alpha(alpha), m_beta(beta)
#endif
  {}
public:
  operator libxsmm_smmfunction() const {
    return m_function;
  }
  void operator()(const float* a, const float* b, float* c) const {
#if defined(LIBXSMM_FALLBACK_SMMFUNCTION)
    if (0 == m_function) {
      LIBXSMM_BLAS_XGEMM(float, m_flags, m_m, m_n, m_k, m_alpha, a, m_lda, b, m_ldb, m_beta, c, m_ldc);
    }
    else
#endif
    {
      LIBXSMM_MMCALL_ABC(m_function, a, b, c);
    }
  }
  void operator()(const float* a, const float* b, float* c, const float* pa, const float* pb, const float* pc) const {
#if defined(LIBXSMM_FALLBACK_SMMFUNCTION)
    if (0 == m_function) {
      LIBXSMM_BLAS_XGEMM(float, m_flags, m_m, m_n, m_k, m_alpha, a, m_lda, b, m_ldb, m_beta, c, m_ldc);
    }
    else
#endif
    {
      LIBXSMM_MMCALL_PRF(m_function, a, b, c, pa, pb, pc);
    }
  }
};

/** Construct and execute a specialized function (double-precision). */
template<> class LIBXSMM_RETARGETABLE libxsmm_mmfunction<double> {
  mutable/*retargetable*/ libxsmm_dmmfunction m_function;
#if defined(LIBXSMM_FALLBACK_DMMFUNCTION)
  int m_flags, m_m, m_n, m_k, m_lda, m_ldb, m_ldc;
  double m_alpha, m_beta;
#endif
public:
  libxsmm_mmfunction(): m_function(0)
#if defined(LIBXSMM_FALLBACK_SMMFUNCTION)
    , m_flags(LIBXSMM_FLAGS), m_m(0), m_n(0), m_k(0), m_lda(0), m_ldb(0), m_ldc(0), m_alpha(LIBXSMM_ALPHA), m_beta(LIBXSMM_BETA)
#endif
  {}
  libxsmm_mmfunction(int m, int n, int k, int flags = LIBXSMM_FLAGS)
    : m_function(libxsmm_dmmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, &flags, 0/*prefetch*/))
#if defined(LIBXSMM_FALLBACK_SMMFUNCTION)
    , m_flags(flags), m_m(m), m_n(n), m_k(k), m_lda(m), m_ldb(k), m_ldc(m), m_alpha(LIBXSMM_ALPHA), m_beta(LIBXSMM_BETA)
#endif
  {}
  libxsmm_mmfunction(int flags, int m, int n, int k, int prefetch)
    : m_function(libxsmm_dmmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, &flags, &prefetch))
#if defined(LIBXSMM_FALLBACK_SMMFUNCTION)
    , m_flags(flags), m_m(m), m_n(n), m_k(k), m_lda(m), m_ldb(k), m_ldc(m), m_alpha(LIBXSMM_ALPHA), m_beta(LIBXSMM_BETA)
#endif
  {}
  libxsmm_mmfunction(int flags, int m, int n, int k, double alpha, double beta)
    : m_function(libxsmm_dmmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, &flags, 0/*prefetch*/))
#if defined(LIBXSMM_FALLBACK_SMMFUNCTION)
    , m_flags(flags), m_m(m), m_n(n), m_k(k), m_lda(m), m_ldb(k), m_ldc(m), m_alpha(alpha), m_beta(beta)
#endif
  {}
  libxsmm_mmfunction(int flags, int m, int n, int k, double alpha, double beta, int prefetch)
    : m_function(libxsmm_dmmdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, &flags, &prefetch))
#if defined(LIBXSMM_FALLBACK_SMMFUNCTION)
    , m_flags(flags), m_m(m), m_n(n), m_k(k), m_lda(m), m_ldb(k), m_ldc(m), m_alpha(alpha), m_beta(beta)
#endif
  {}
  libxsmm_mmfunction(int flags, int m, int n, int k, int lda, int ldb, int ldc, int prefetch)
    : m_function(libxsmm_dmmdispatch(m, n, k, &lda, &ldb, &ldc, 0/*alpha*/, 0/*beta*/, &flags, &prefetch))
#if defined(LIBXSMM_FALLBACK_SMMFUNCTION)
    , m_flags(flags), m_m(m), m_n(n), m_k(k), m_lda(lda), m_ldb(ldb), m_ldc(ldc), m_alpha(LIBXSMM_ALPHA), m_beta(LIBXSMM_BETA)
#endif
  {}
  libxsmm_mmfunction(int flags, int m, int n, int k, int lda, int ldb, int ldc, double alpha, double beta)
    : m_function(libxsmm_dmmdispatch(m, n, k, &lda, &ldb, &ldc, &alpha, &beta, &flags, 0/*prefetch*/))
#if defined(LIBXSMM_FALLBACK_SMMFUNCTION)
    , m_flags(flags), m_m(m), m_n(n), m_k(k), m_lda(m), m_ldb(k), m_ldc(m), m_alpha(alpha), m_beta(beta)
#endif
  {}
  libxsmm_mmfunction(int flags, int m, int n, int k, int lda, int ldb, int ldc, double alpha, double beta, int prefetch)
    : m_function(libxsmm_dmmdispatch(m, n, k, &lda, &ldb, &ldc, &alpha, &beta, &flags, &prefetch))
#if defined(LIBXSMM_FALLBACK_SMMFUNCTION)
    , m_flags(flags), m_m(m), m_n(n), m_k(k), m_lda(m), m_ldb(k), m_ldc(m), m_alpha(alpha), m_beta(beta)
#endif
  {}
public:
  operator libxsmm_dmmfunction() const {
    return m_function;
  }
  void operator()(const double* a, const double* b, double* c) const {
#if defined(LIBXSMM_FALLBACK_DMMFUNCTION)
    if (0 == m_function) {
      LIBXSMM_BLAS_XGEMM(double, m_flags, m_m, m_n, m_k, m_alpha, a, m_lda, b, m_ldb, m_beta, c, m_ldc);
    }
    else
#endif
    {
      LIBXSMM_MMCALL_ABC(m_function, a, b, c);
    }
  }
  void operator()(const double* a, const double* b, double* c, const double* pa, const double* pb, const double* pc) const {
#if defined(LIBXSMM_FALLBACK_DMMFUNCTION)
    if (0 == m_function) {
      LIBXSMM_BLAS_XGEMM(double, m_flags, m_m, m_n, m_k, m_alpha, a, m_lda, b, m_ldb, m_beta, c, m_ldc);
    }
    else
#endif
    {
      LIBXSMM_MMCALL_PRF(m_function, a, b, c, pa, pb, pc);
    }
  }
};

/** Matrix copy function ("in" can be NULL to zero the destination). */
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_matcopy(T* out, const T* in, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo) {
  return libxsmm_matcopy(out, in, sizeof(T), m, n, ldi, ldo);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_matcopy(T* out, const T* in, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi) {
  return libxsmm_matcopy(out, in, m, n, ldi, ldi);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_matcopy(T* out, const T* in, libxsmm_blasint m, libxsmm_blasint n) {
  return libxsmm_matcopy(out, in, m, n, m);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_matcopy(T* out, const T* in, libxsmm_blasint n) {
  return libxsmm_matcopy(out, in, n, n);
}

/** Matrix copy function ("in" can be NULL to zero the destination); MT via libxsmmext. */
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_matcopy_omp(T* out, const T* in, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo) {
  return libxsmm_matcopy_omp(out, in, sizeof(T), m, n, ldi, ldo);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_matcopy_omp(T* out, const T* in, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi) {
  return libxsmm_matcopy_omp(out, in, m, n, ldi, ldi);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_matcopy_omp(T* out, const T* in, libxsmm_blasint m, libxsmm_blasint n) {
  return libxsmm_matcopy_omp(out, in, m, n, m);
}
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_matcopy_omp(T* out, const T* in, libxsmm_blasint n) {
  return libxsmm_matcopy_omp(out, in, n, n);
}

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

/** Matrix transposition; MT via libxsmmext (out-of-place form). */
template<typename T> inline/*superfluous*/ LIBXSMM_RETARGETABLE int libxsmm_trans_omp(T* out, const T* in, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo) {
  return libxsmm_otrans_omp(out, in, sizeof(T), m, n, ldi, ldo);
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

