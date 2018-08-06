/******************************************************************************
** Copyright (c) 2015-2018, Intel Corporation                                **
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
#ifndef LIBXSMM_GEMM_H
#define LIBXSMM_GEMM_H

#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if !defined(LIBXSMM_GEMM_WRAP_DYNAMIC) && defined(LIBXSMM_BUILD) && \
  (!defined(__BLAS) || (0 != __BLAS)) && defined(__GNUC__) && \
  !(defined(__APPLE__) && defined(__MACH__) && LIBXSMM_VERSION3(6, 1, 0) >= \
    LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__)) && \
  !defined(_WIN32) && !defined(__CYGWIN__)
# include <dlfcn.h>
# define LIBXSMM_GEMM_WRAP_DYNAMIC
#endif
#include <limits.h>
#include <stdio.h>
#include <math.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXSMM_GEMM_CHECK) && !defined(NDEBUG)
# define LIBXSMM_GEMM_CHECK
#endif
#if !defined(LIBXSMM_GEMM_LOCK)
# define LIBXSMM_GEMM_LOCK LIBXSMM_LOCK_DEFAULT
#endif
#if !defined(LIBXSMM_GEMM_TASKSCALE)
# define LIBXSMM_GEMM_TASKSCALE 2
#endif
#if !defined(LIBXSMM_GEMM_BATCHSCALE)
# define LIBXSMM_GEMM_BATCHSCALE 1.5
#endif

#if !defined(LIBXSMM_GEMM_MMBATCH) && defined(LIBXSMM_BUILD) && \
    (defined(LIBXSMM_CONFIG_WRAP) && 0 != (LIBXSMM_CONFIG_WRAP)) && \
    (defined(LIBXSMM_GEMM_WRAP_STATIC) || defined(LIBXSMM_GEMM_WRAP_DYNAMIC) || \
    !defined(NDEBUG) || defined(_WIN32)) /* debug purpose */
# define LIBXSMM_GEMM_MMBATCH
#endif

/** Undefine (disarm) MKL's DIRECT_CALL macros. */
#if defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# if defined(sgemm_)
#   undef sgemm_
# endif
# if defined(dgemm_)
#   undef dgemm_
# endif
#endif

#if (!defined(__BLAS) || (0 != __BLAS))
# define LIBXSMM_GEMM_WRAPPER_BLAS(TYPE, ORIGINAL, SYMBOL) if (0 == (ORIGINAL)) { \
    union { LIBXSMM_GEMMFUNCTION_TYPE(TYPE) pf; \
      void (*sf)(LIBXSMM_GEMM_CONST char*, LIBXSMM_GEMM_CONST char*, \
        LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, \
        LIBXSMM_GEMM_CONST float*, LIBXSMM_GEMM_CONST float*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, LIBXSMM_GEMM_CONST float*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, \
        LIBXSMM_GEMM_CONST float*, float*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*); \
      void (*df)(LIBXSMM_GEMM_CONST char*, LIBXSMM_GEMM_CONST char*, \
        LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, \
        LIBXSMM_GEMM_CONST double*, LIBXSMM_GEMM_CONST double*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, LIBXSMM_GEMM_CONST double*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, \
        LIBXSMM_GEMM_CONST double*, double*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*); \
    } libxsmm_gemm_wrapper_blas_; \
    libxsmm_gemm_wrapper_blas_.LIBXSMM_TPREFIX(TYPE,f) = (SYMBOL); \
    /*LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_STORE, LIBXSMM_BITS)(&(ORIGINAL), libxsmm_gemm_wrapper_blas_.pf, LIBXSMM_ATOMIC_RELAXED);*/ \
    ORIGINAL = libxsmm_gemm_wrapper_blas_.pf; \
  }
# define LIBXSMM_GEMV_WRAPPER_BLAS(TYPE, ORIGINAL, SYMBOL) if (0 == (ORIGINAL)) { \
    union { LIBXSMM_GEMVFUNCTION_TYPE(TYPE) pf; \
      void (*sf)(LIBXSMM_GEMM_CONST char*, \
        LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, \
        LIBXSMM_GEMM_CONST float*, LIBXSMM_GEMM_CONST float*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, LIBXSMM_GEMM_CONST float*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, \
        LIBXSMM_GEMM_CONST float*, float*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*); \
      void (*df)(LIBXSMM_GEMM_CONST char*, \
        LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, \
        LIBXSMM_GEMM_CONST double*, LIBXSMM_GEMM_CONST double*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, LIBXSMM_GEMM_CONST double*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*, \
        LIBXSMM_GEMM_CONST double*, double*, LIBXSMM_GEMM_CONST LIBXSMM_BLASINT*); \
    } libxsmm_gemv_wrapper_blas_; \
    libxsmm_gemv_wrapper_blas_.LIBXSMM_TPREFIX(TYPE,f) = (SYMBOL); \
    /*LIBXSMM_ATOMIC(LIBXSMM_ATOMIC_STORE, LIBXSMM_BITS)(&(ORIGINAL), libxsmm_gemv_wrapper_blas_.pf, LIBXSMM_ATOMIC_RELAXED);*/ \
    ORIGINAL = libxsmm_gemv_wrapper_blas_.pf; \
  }
#else
# define LIBXSMM_GEMM_WRAPPER_BLAS(TYPE, ORIGINAL, SYMBOL)
# define LIBXSMM_GEMV_WRAPPER_BLAS(TYPE, ORIGINAL, SYMBOL)
#endif

#if defined(LIBXSMM_GEMM_WRAP) && defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT) && \
  !(defined(__APPLE__) && defined(__MACH__) /*&& defined(__clang__)*/) && !defined(__CYGWIN__)
# if (2 != (LIBXSMM_GEMM_WRAP)) /* SGEMM and DGEMM */
#   define LIBXSMM_GEMM_WRAPPER_STATIC(TYPE, ORIGINAL) LIBXSMM_GEMM_WRAPPER_BLAS(TYPE, ORIGINAL, \
      LIBXSMM_FSYMBOL(LIBXSMM_CONCATENATE(__real_, LIBXSMM_TPREFIX(TYPE, gemm))))
#   define LIBXSMM_GEMV_WRAPPER_STATIC(TYPE, ORIGINAL) LIBXSMM_GEMM_WRAPPER_BLAS(TYPE, ORIGINAL, \
      LIBXSMM_FSYMBOL(LIBXSMM_CONCATENATE(__real_, LIBXSMM_TPREFIX(TYPE, gemv))))
# else /* DGEMM only */
#   define LIBXSMM_GEMM_WRAPPER_STATIC(TYPE, ORIGINAL) if (0 != LIBXSMM_EQUAL(TYPE, double)) { \
      LIBXSMM_GEMM_WRAPPER_BLAS(TYPE, ORIGINAL, LIBXSMM_FSYMBOL(__real_dgemm)) \
    }
#   define LIBXSMM_GEMV_WRAPPER_STATIC(TYPE, ORIGINAL) if (0 != LIBXSMM_EQUAL(TYPE, double)) { \
      LIBXSMM_GEMM_WRAPPER_BLAS(TYPE, ORIGINAL, LIBXSMM_FSYMBOL(__real_dgemv)) \
    }
# endif
# define LIBXSMM_GEMM_WRAP_STATIC
#else
# define LIBXSMM_GEMM_WRAPPER_STATIC(TYPE, ORIGINAL)
# define LIBXSMM_GEMV_WRAPPER_STATIC(TYPE, ORIGINAL)
#endif

#if defined(LIBXSMM_GEMM_WRAP_DYNAMIC)
# define LIBXSMM_GEMM_WRAPPER_DYNAMIC(TYPE, ORIGINAL) \
    if (0 == (ORIGINAL)) { \
      union { const void* pv; LIBXSMM_GEMMFUNCTION_TYPE(TYPE) pf; } libxsmm_gemm_wrapper_dynamic_ = { 0 }; \
      dlerror(); /* clear an eventual error status */ \
      libxsmm_gemm_wrapper_dynamic_.pv = dlsym(RTLD_NEXT, LIBXSMM_STRINGIFY(LIBXSMM_GEMM_SYMBOL(TYPE))); \
      /*LIBXSMM_ATOMIC_STORE(&(ORIGINAL), libxsmm_gemm_wrapper_dynamic_.pf, LIBXSMM_ATOMIC_RELAXED);*/ \
      ORIGINAL = libxsmm_gemm_wrapper_dynamic_.pf; \
      LIBXSMM_GEMM_WRAPPER_BLAS(TYPE, ORIGINAL, LIBXSMM_GEMM_SYMBOL(TYPE)); \
    }
# define LIBXSMM_GEMV_WRAPPER_DYNAMIC(TYPE, ORIGINAL) \
    if (0 == (ORIGINAL)) { \
      union { const void* pv; LIBXSMM_GEMVFUNCTION_TYPE(TYPE) pf; } libxsmm_gemv_wrapper_dynamic_ = { 0 }; \
      dlerror(); /* clear an eventual error status */ \
      libxsmm_gemv_wrapper_dynamic_.pv = dlsym(RTLD_NEXT, LIBXSMM_STRINGIFY(LIBXSMM_GEMV_SYMBOL(TYPE))); \
      /*LIBXSMM_ATOMIC_STORE(&(ORIGINAL), libxsmm_gemv_wrapper_dynamic_.pf, LIBXSMM_ATOMIC_RELAXED);*/ \
      ORIGINAL = libxsmm_gemv_wrapper_dynamic_.pf; \
      LIBXSMM_GEMV_WRAPPER_BLAS(TYPE, ORIGINAL, LIBXSMM_GEMV_SYMBOL(TYPE)); \
    }
#else
# define LIBXSMM_GEMM_WRAPPER_DYNAMIC(TYPE, ORIGINAL) LIBXSMM_GEMM_WRAPPER_BLAS( \
    TYPE, ORIGINAL, LIBXSMM_GEMM_SYMBOL(TYPE))
# define LIBXSMM_GEMV_WRAPPER_DYNAMIC(TYPE, ORIGINAL) LIBXSMM_GEMV_WRAPPER_BLAS( \
    TYPE, ORIGINAL, LIBXSMM_GEMV_SYMBOL(TYPE))
#endif

#if defined(NDEBUG) /* library code is expected to be mute */
# define LIBXSMM_GEMM_WRAPPER(TYPE, ORIGINAL) if (0 == (ORIGINAL)) { \
    LIBXSMM_GEMM_WRAPPER_STATIC(TYPE, ORIGINAL); \
    LIBXSMM_GEMM_WRAPPER_DYNAMIC(TYPE, ORIGINAL); \
  }
#else
# define LIBXSMM_GEMM_WRAPPER(TYPE, ORIGINAL) if (0 == (ORIGINAL)) { \
    LIBXSMM_GEMM_WRAPPER_STATIC(TYPE, ORIGINAL); \
    LIBXSMM_GEMM_WRAPPER_DYNAMIC(TYPE, ORIGINAL); \
    if (0 == (ORIGINAL)) { \
      static int libxsmm_gemm_wrapper_error_once_ = 0; \
      if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&libxsmm_gemm_wrapper_error_once_, 1, LIBXSMM_ATOMIC_RELAXED)) { \
        fprintf(stderr, "LIBXSMM ERROR: application must be linked against LAPACK/BLAS!\n"); \
      } \
    } \
  }
#endif


/** Provides GEMM functions available via BLAS; NOT thread-safe. */
LIBXSMM_API_INTERN void libxsmm_gemm_init(int archid);

/** Finalizes the GEMM facility; NOT thread-safe. */
LIBXSMM_API_INTERN void libxsmm_gemm_finalize(void);

/** Determines the size of the element-type given by precision. */
LIBXSMM_API_INTERN unsigned char libxsmm_gemm_typesize(libxsmm_gemm_precision precision);

/** Determines the given value in double-precision based on the given precision. */
LIBXSMM_API_INTERN int libxsmm_gemm_dvalue(libxsmm_gemm_precision precision, const void* value, double* dvalue);
/** Determines the value given in double-precision. */
LIBXSMM_API_INTERN int libxsmm_gemm_cast(libxsmm_gemm_precision precision, double dvalue, void* value);

LIBXSMM_API_INTERN int libxsmm_gemm_prefetch2uid(libxsmm_gemm_prefetch_type prefetch);
LIBXSMM_API_INTERN libxsmm_gemm_prefetch_type libxsmm_gemm_uid2prefetch(int uid);

#if defined(LIBXSMM_GEMM_WRAP_STATIC)
LIBXSMM_API void LIBXSMM_FSYMBOL(__real_sgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*, const float* b, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(__real_dgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*, const double* b, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);
#endif /*defined(LIBXSMM_GEMM_WRAP_STATIC)*/

#if defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(__wrap_sgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*, const float* b, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(__wrap_dgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*, const double* b, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);
#endif

LIBXSMM_GEMM_SYMBOL_DECL(LIBXSMM_GEMM_CONST, float);
LIBXSMM_GEMM_SYMBOL_DECL(LIBXSMM_GEMM_CONST, double);

LIBXSMM_EXTERN_C struct LIBXSMM_RETARGETABLE libxsmm_gemm_handle {
  libxsmm_code_pointer copy_a, copy_b, copy_i, copy_o;
  libxsmm_xmmfunction kernel[2];
  unsigned int m, n, k, lda, ldb, ldc;
  unsigned int tm, tn, tk, dm, dn, dk;
  unsigned int itypesize, otypesize;
  unsigned int nthreads, mt, nt, kt;
  int flags_gemm, flags_copy, prf_copy;
};

LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE libxsmm_gemm_batchitem {
  struct {
    const void *a, *b;
    void *c;
  } value;
  struct {
    libxsmm_gemm_descriptor desc;
    unsigned int count;
    const char* symbol;
  } stat;
  /* TODO: consider padding */
} libxsmm_gemm_batchitem;

LIBXSMM_API int libxsmm_mmbatch_internal(libxsmm_xmmfunction kernel, libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const void* a, const void* b, void* c, libxsmm_blasint batchsize, int tid, int nthreads,
  const libxsmm_gemm_descriptor* info);

LIBXSMM_API int libxsmm_dmmbatch_blas(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const double* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb, const double* beta, void* c, const libxsmm_blasint* ldc,
  libxsmm_blasint index_base, libxsmm_blasint index_stride, const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize);
LIBXSMM_API int libxsmm_smmbatch_blas(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const float* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb, const float* beta, void* c, const libxsmm_blasint* ldc,
  libxsmm_blasint index_base, libxsmm_blasint index_stride, const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize);

LIBXSMM_EXTERN_C typedef void (*libxsmm_mmbatch_flush_function)(void);

/** Table of M-extents per type-size (tile shape). */
LIBXSMM_APIVAR_PUBLIC(unsigned int* libxsmm_gemm_mtile);
/** Table of M-extents per type-size (tile shape). */
LIBXSMM_APIVAR_PUBLIC(unsigned int* libxsmm_gemm_ntile);
/** Table of M-extents per type-size (tile shape). */
LIBXSMM_APIVAR_PUBLIC(unsigned int* libxsmm_gemm_ktile);

/** auto-batch descriptor (filter). */
LIBXSMM_APIVAR_PUBLIC(libxsmm_gemm_descriptor libxsmm_gemm_batchdesc);
/** Records a batch of SMMs. */
LIBXSMM_APIVAR_PUBLIC(libxsmm_gemm_batchitem* libxsmm_gemm_batcharray);
/** Lock: libxsmm_mmbatch_begin, libxsmm_mmbatch_end, internal_mmbatch_flush. */
LIBXSMM_APIVAR_PUBLIC(LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK) libxsmm_gemm_batchlock);
/** Maximum size of the recorded batch. */
LIBXSMM_APIVAR_PUBLIC(unsigned int libxsmm_gemm_batchsize);
/** Minimum batchsize per thread/task. */
LIBXSMM_APIVAR_PUBLIC(unsigned int libxsmm_gemm_batchgrain);
/** Determines if OpenMP tasks are used, and scales beyond the number of threads. */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_gemm_taskscale);
/**
 * Intercepted GEMM
 * - odd: sequential and non-tiled (small problem sizes only)
 * - even (or negative): parallelized and tiled (all problem sizes)
 * - 3: GEMV is intercepted; small problem sizes
 * - 4: GEMV is intercepted; all problem sizes
 */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_gemm_wrap);
/** Prefetch strategy for tiled GEMM. */
LIBXSMM_APIVAR_PUBLIC(libxsmm_gemm_prefetch_type libxsmm_gemm_tiled_prefetch);

/** Determines the default prefetch strategy, which is used in case of LIBXSMM_PREFETCH_AUTO. */
LIBXSMM_APIVAR(libxsmm_gemm_prefetch_type libxsmm_gemm_auto_prefetch_default);
/** Determines the prefetch strategy, which is used in case of LIBXSMM_PREFETCH_AUTO. */
LIBXSMM_APIVAR(libxsmm_gemm_prefetch_type libxsmm_gemm_auto_prefetch);

#endif /*LIBXSMM_GEMM_H*/

