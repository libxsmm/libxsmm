/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_GEMM_H
#define LIBXSMM_GEMM_H

#include "libxsmm_main.h"

#if !defined(LIBXSMM_GEMM_LOCK)
# define LIBXSMM_GEMM_LOCK LIBXSMM_LOCK_DEFAULT
#endif

#if !defined(LIBXSMM_BLAS_WRAP_DYNAMIC) && defined(LIBXSMM_INTERCEPT_DYNAMIC) && (!defined(__BLAS) || (0 != __BLAS))
# define LIBXSMM_BLAS_WRAP_DYNAMIC
#endif
#if !defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD) && \
    (defined(LIBXSMM_CONFIG_WRAP) && 0 != (LIBXSMM_CONFIG_WRAP)) && \
    (defined(LIBXSMM_BLAS_WRAP_DYNAMIC) || !defined(NDEBUG) || defined(_WIN32)) /* debug */
# define LIBXSMM_WRAP LIBXSMM_CONFIG_WRAP
#endif

/** Undefine (disarm) MKL's DIRECT_CALL macros. */
#if (defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL))
# if defined(sgemm_)
#   undef sgemm_
# endif
# if defined(dgemm_)
#   undef dgemm_
# endif
#endif

/** Construct symbol name from a given real type name (float, double, etc.). */
#define LIBXSMM_BLAS_FNTYPE(TYPE, KIND) LIBXSMM_CONCATENATE3(libxsmm_, LIBXSMM_TPREFIX(TYPE, KIND), _function)

#if !defined(LIBXSMM_BLAS_ERROR_MSG)
# define LIBXSMM_BLAS_ERROR_MSG(SYMBOL) fprintf(stderr, "LIBXSMM WARNING: application shall be linked against LAPACK/BLAS %s!\n", SYMBOL)
#endif

#if !defined(LIBXSMM_BLAS_ERROR)
#define LIBXSMM_BLAS_ERROR(SYMBOL, PCOUNTER) do { \
    /*const*/ int hash = (int)libxsmm_hash32(libxsmm_hash_string(SYMBOL)), old = *(PCOUNTER); \
    if (LIBXSMM_ATOMIC_CMPSWP(PCOUNTER, old, hash, LIBXSMM_ATOMIC_RELAXED) && old != hash) { \
      LIBXSMM_BLAS_ERROR_MSG(SYMBOL); \
    } \
  } while(0)
#endif

#if defined(LIBXSMM_BUILD)
# define LIBXSMM_BLAS_WRAPPER_STATIC1(TYPE, KIND, ORIGINAL) if (NULL == (ORIGINAL)) do { \
    ORIGINAL = LIBXSMM_FSYMBOL(LIBXSMM_CONCATENATE(__real_, LIBXSMM_TPREFIX(TYPE, KIND))); \
  } while(0)
# define LIBXSMM_BLAS_WRAPPER_STATIC0 LIBXSMM_BLAS_WRAPPER_STATIC1
#else
# define LIBXSMM_BLAS_WRAPPER_STATIC1(TYPE, KIND, ORIGINAL) if (NULL == (ORIGINAL)) do { \
    ORIGINAL = (LIBXSMM_BLAS_FNTYPE(TYPE, KIND))LIBXSMM_BLAS_SYMBOL(TYPE, KIND); \
  } while(0)
# define LIBXSMM_BLAS_WRAPPER_STATIC0(TYPE, KIND, ORIGINAL)
#endif
#define LIBXSMM_BLAS_WRAPPER_STATIC(CONDITION, TYPE, KIND, ORIGINAL) \
  LIBXSMM_CONCATENATE(LIBXSMM_BLAS_WRAPPER_STATIC, CONDITION)(TYPE, KIND, ORIGINAL)

#if defined(LIBXSMM_BLAS_WRAP_DYNAMIC)
# define LIBXSMM_BLAS_WRAPPER_DYNAMIC(TYPE, KIND, ORIGINAL, NEXT) do { \
    union { const void* pfin; \
      LIBXSMM_BLAS_FNTYPE(TYPE, KIND) (*chain)(void); /* chain */ \
      LIBXSMM_BLAS_FNTYPE(TYPE, KIND) pfout; \
    } libxsmm_blas_wrapper_dynamic_ /*= { 0 }*/; \
    dlerror(); /* clear an eventual error status */ \
    libxsmm_blas_wrapper_dynamic_.pfin = dlsym(RTLD_DEFAULT, "mkl_blas." LIBXSMM_STRINGIFY(LIBXSMM_CBLAS_SYMBOL(TYPE, KIND))); \
    if (NULL == dlerror() && NULL != libxsmm_blas_wrapper_dynamic_.pfout) { \
      ORIGINAL = libxsmm_blas_wrapper_dynamic_.pfout; /* LIBXSMM_ATOMIC_STORE */ \
    } \
    else { \
      /*const*/ LIBXSMM_BLAS_FNTYPE(TYPE, KIND)(*libxsmm_blas_wrapper_dynamic_next_)(void) = NEXT; \
      libxsmm_blas_wrapper_dynamic_.pfin = (NULL == libxsmm_blas_wrapper_dynamic_next_ ? \
        dlsym(LIBXSMM_RTLD_NEXT, "libxsmm_original_" LIBXSMM_STRINGIFY(LIBXSMM_TPREFIX(TYPE, KIND))) : NULL); \
      if  (NULL != dlerror() || NULL == libxsmm_blas_wrapper_dynamic_.chain \
        || libxsmm_blas_wrapper_dynamic_next_ == libxsmm_blas_wrapper_dynamic_.chain \
        || NULL == libxsmm_blas_wrapper_dynamic_.chain()) \
      { \
        libxsmm_blas_wrapper_dynamic_.pfin = dlsym(LIBXSMM_RTLD_NEXT, LIBXSMM_STRINGIFY(LIBXSMM_BLAS_SYMBOL(TYPE, KIND))); \
        if (NULL != libxsmm_blas_wrapper_dynamic_.pfout) { \
          ORIGINAL = (NULL == dlerror() ? libxsmm_blas_wrapper_dynamic_.pfout : NULL); /* LIBXSMM_ATOMIC_STORE */ \
        } \
      } \
    } \
  } while(0)
#else
# define LIBXSMM_BLAS_WRAPPER_DYNAMIC(TYPE, KIND, ORIGINAL, NEXT)
#endif

#define LIBXSMM_BLAS_WRAPPER(CONDITION, TYPE, KIND, ORIGINAL, NEXT) if (NULL == (ORIGINAL)) do { \
  LIBXSMM_BLAS_WRAPPER_DYNAMIC(TYPE, KIND, ORIGINAL, NEXT); \
  LIBXSMM_BLAS_WRAPPER_STATIC(CONDITION, TYPE, KIND, ORIGINAL); \
} while(0)

/** Fall-back code paths: LIBXSMM_XGEMM_FALLBACK0, and LIBXSMM_XGEMM_FALLBACK1 (macro template). */
#if !defined(LIBXSMM_XGEMM_FALLBACK0)
# define LIBXSMM_XGEMM_FALLBACK0(ITYPE, OTYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
     LIBXSMM_BLAS_FUNCTION(ITYPE, OTYPE, gemm)(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#endif
#if !defined(LIBXSMM_XGEMM_FALLBACK1)
# define LIBXSMM_XGEMM_FALLBACK1(ITYPE, OTYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
     LIBXSMM_BLAS_FUNCTION(ITYPE, OTYPE, gemm)(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#endif

/** Default-initialize libxsmm_gemm_param structure for the given prefetch-strategy. */
#if (LIBXSMM_PREFETCH_NONE != LIBXSMM_PREFETCH) /* LIBXSMM_GEMM_PREFETCH_NONE is an enumerator! */
# define LIBXSMM_XGEMM_PREFETCH(ITYPE, OTYPE, M, N, K, ARGS) do { \
    (ARGS).a.quaternary = ((char*)(ARGS).a.primary) + sizeof(ITYPE) * (M) * (K); \
    (ARGS).b.quaternary = ((char*)(ARGS).b.primary) + sizeof(ITYPE) * (K) * (N); \
    (ARGS).c.quaternary = ((char*)(ARGS).c.primary) + sizeof(OTYPE) * (M) * (N); \
  } while(0)
#elif !defined(NDEBUG)
# define LIBXSMM_XGEMM_PREFETCH(ITYPE, OTYPE, M, N, K, ARGS) \
    (ARGS).a.quaternary = (ARGS).b.quaternary = (ARGS).c.quaternary = NULL
#else
# define LIBXSMM_XGEMM_PREFETCH(ITYPE, OTYPE, M, N, K, ARGS);
#endif

/**
 * Execute a specialized function, or use a fallback code path depending on threshold (macro template).
 * LIBXSMM_XGEMM_FALLBACK0 or specialized function: below LIBXSMM_MAX_MNK
 * LIBXSMM_XGEMM_FALLBACK1: above LIBXSMM_MAX_MNK
 */
#define LIBXSMM_XGEMM(ITYPE, OTYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) do { \
  const OTYPE libxsmm_xgemm_beta_ = (NULL != ((void*)(BETA)) ? (*(const OTYPE*)(BETA)) : ((OTYPE)LIBXSMM_BETA)); \
  const int libxsmm_xgemm_flags_ = LIBXSMM_GEMM_PFLAGS(TRANSA, TRANSB, LIBXSMM_FLAGS) | \
    (LIBXSMM_NEQ(0, libxsmm_xgemm_beta_) ? 0 : LIBXSMM_GEMM_FLAG_BETA_0); \
  const libxsmm_blasint *const libxsmm_xgemm_k_ = (NULL != (K) ? (K) : (M)); \
  const libxsmm_blasint *const libxsmm_xgemm_n_ = (NULL != (N) ? (N) : libxsmm_xgemm_k_); \
  const libxsmm_blasint libxsmm_xgemm_lda_ = LIBXSMM_MAX(NULL != ((void*)(LDA)) ? *(LDA) : \
    *(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & libxsmm_xgemm_flags_) ? (M) : libxsmm_xgemm_k_), 1); \
  const libxsmm_blasint libxsmm_xgemm_ldb_ = LIBXSMM_MAX(NULL != ((void*)(LDB)) ? *(LDB) : \
    *(0 == (LIBXSMM_GEMM_FLAG_TRANS_B & libxsmm_xgemm_flags_) ? libxsmm_xgemm_k_ : libxsmm_xgemm_n_), 1); \
  const libxsmm_blasint libxsmm_xgemm_ldc_ = LIBXSMM_MAX(NULL != (LDC) ? *(LDC) : *(M), 1); \
  if (LIBXSMM_SMM(*(M), *libxsmm_xgemm_n_, *libxsmm_xgemm_k_, 2/*RFO*/, sizeof(OTYPE))) { \
    const libxsmm_gemm_shape libxsmm_xgemm_shape_ = libxsmm_create_gemm_shape( \
      *(M), *libxsmm_xgemm_n_, *libxsmm_xgemm_k_, libxsmm_xgemm_lda_, libxsmm_xgemm_ldb_, libxsmm_xgemm_ldc_, \
      LIBXSMM_DATATYPE(ITYPE), LIBXSMM_DATATYPE(ITYPE), LIBXSMM_DATATYPE(OTYPE), LIBXSMM_DATATYPE(OTYPE)); \
    const libxsmm_gemmfunction libxsmm_xgemm_function_ = libxsmm_dispatch_gemm(libxsmm_xgemm_shape_, \
      (libxsmm_bitfield)libxsmm_xgemm_flags_, (libxsmm_bitfield)(LIBXSMM_PREFETCH)); \
    if (NULL != libxsmm_xgemm_function_) { \
      libxsmm_gemm_param libxsmm_xgemm_param_; libxsmm_xgemm_param_.a.primary = (ITYPE*)(A); \
      libxsmm_xgemm_param_.b.primary = (ITYPE*)(B); libxsmm_xgemm_param_.c.primary = (OTYPE*)(C); \
      LIBXSMM_XGEMM_PREFETCH(ITYPE, OTYPE, *(M), *libxsmm_xgemm_n_, *libxsmm_xgemm_k_, libxsmm_xgemm_param_); \
      libxsmm_xgemm_function_(&libxsmm_xgemm_param_); \
    } \
    else { \
      const char libxsmm_xgemm_transa_ = (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & libxsmm_xgemm_flags_) ? 'n' : 't'); \
      const char libxsmm_xgemm_transb_ = (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_B & libxsmm_xgemm_flags_) ? 'n' : 't'); \
      const OTYPE libxsmm_xgemm_alpha_ = (NULL != ((void*)(ALPHA)) ? (*(const OTYPE*)(ALPHA)) : ((OTYPE)LIBXSMM_ALPHA)); \
      LIBXSMM_XGEMM_FALLBACK0(ITYPE, OTYPE, &libxsmm_xgemm_transa_, &libxsmm_xgemm_transb_, \
        M, libxsmm_xgemm_n_, libxsmm_xgemm_k_, \
        &libxsmm_xgemm_alpha_, A, &libxsmm_xgemm_lda_, \
                               B, &libxsmm_xgemm_ldb_, \
         &libxsmm_xgemm_beta_, C, &libxsmm_xgemm_ldc_); \
    } \
  } \
  else { \
    const char libxsmm_xgemm_transa_ = (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & libxsmm_xgemm_flags_) ? 'n' : 't'); \
    const char libxsmm_xgemm_transb_ = (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_B & libxsmm_xgemm_flags_) ? 'n' : 't'); \
    const OTYPE libxsmm_xgemm_alpha_ = (NULL != ((void*)(ALPHA)) ? (*(const OTYPE*)(ALPHA)) : ((OTYPE)LIBXSMM_ALPHA)); \
    LIBXSMM_XGEMM_FALLBACK1(ITYPE, OTYPE, &libxsmm_xgemm_transa_, &libxsmm_xgemm_transb_, \
      M, libxsmm_xgemm_n_, libxsmm_xgemm_k_, \
      &libxsmm_xgemm_alpha_, A, &libxsmm_xgemm_lda_, \
                             B, &libxsmm_xgemm_ldb_, \
       &libxsmm_xgemm_beta_, C, &libxsmm_xgemm_ldc_); \
  } \
} while(0)

/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (macro template). */
#define LIBXSMM_BLAS_XGEMM(ITYPE, OTYPE, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) do { \
  /* Use 'n' (instead of 'N') avoids warning about "no macro replacement within a character constant". */ \
  const char libxsmm_blas_xgemm_transa_ = (char)(NULL != ((void*)(TRANSA)) ? (*(const char*)(TRANSA)) : \
    (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & LIBXSMM_FLAGS) ? 'n' : 't')); \
  const char libxsmm_blas_xgemm_transb_ = (char)(NULL != ((void*)(TRANSB)) ? (*(const char*)(TRANSB)) : \
    (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & LIBXSMM_FLAGS) ? 'n' : 't')); \
  const libxsmm_blasint *const libxsmm_blas_xgemm_k_ = (NULL != ((void*)(K)) ? (K) : (M)); \
  const libxsmm_blasint *const libxsmm_blas_xgemm_n_ = (NULL != ((void*)(N)) ? (N) : libxsmm_blas_xgemm_k_); \
  const libxsmm_blasint libxsmm_blas_xgemm_lda_ = LIBXSMM_MAX(NULL != ((void*)(LDA)) ? *(LDA) : \
    *(('n' == libxsmm_blas_xgemm_transa_ || *"N" == libxsmm_blas_xgemm_transa_) ? (M) : libxsmm_blas_xgemm_k_), 1); \
  const libxsmm_blasint libxsmm_blas_xgemm_ldb_ = LIBXSMM_MAX(NULL != ((void*)(LDB)) ? *(LDB) : \
    *(('n' == libxsmm_blas_xgemm_transb_ || *"N" == libxsmm_blas_xgemm_transb_) ? libxsmm_blas_xgemm_k_ : libxsmm_blas_xgemm_n_), 1); \
  const libxsmm_blasint libxsmm_blas_xgemm_ldc_ = LIBXSMM_MAX(NULL != ((void*)(LDC)) ? *(LDC) : *(M), 1); \
  const OTYPE libxsmm_blas_xgemm_alpha_ = (NULL != ((void*)(ALPHA)) ? (*(const OTYPE*)(ALPHA)) : ((OTYPE)LIBXSMM_ALPHA)); \
  const OTYPE libxsmm_blas_xgemm_beta_  = (NULL != ((void*)(BETA))  ? (*(const OTYPE*)(BETA))  : ((OTYPE)LIBXSMM_BETA)); \
  LIBXSMM_BLAS_FUNCTION(ITYPE, OTYPE, gemm)(&libxsmm_blas_xgemm_transa_, &libxsmm_blas_xgemm_transb_, \
    M, libxsmm_blas_xgemm_n_, libxsmm_blas_xgemm_k_, \
    &libxsmm_blas_xgemm_alpha_, (const ITYPE*)(A), &libxsmm_blas_xgemm_lda_, \
                                (const ITYPE*)(B), &libxsmm_blas_xgemm_ldb_, \
     &libxsmm_blas_xgemm_beta_,       (ITYPE*)(C), &libxsmm_blas_xgemm_ldc_); \
} while(0)


/** Provides GEMM functions available via BLAS; NOT thread-safe. */
LIBXSMM_API_INTERN void libxsmm_gemm_init(void);

/** Finalizes the GEMM facility; NOT thread-safe. */
LIBXSMM_API_INTERN void libxsmm_gemm_finalize(void);

/** Translates GEMM-prefetch requests incl. LIBXSMM_PREFETCH_AUTO. */
LIBXSMM_API libxsmm_gemm_prefetch_type libxsmm_get_gemm_prefetch(int prefetch);

#if defined(LIBXSMM_BUILD)
#if defined(LIBXSMM_BUILD_EXT)
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(__wrap_dgemm_batch_strided)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, double, gemm_batch_strided));
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(__wrap_sgemm_batch_strided)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, float, gemm_batch_strided));
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(__wrap_dgemm_batch)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, double, gemm_batch));
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(__wrap_sgemm_batch)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, float, gemm_batch));
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(__wrap_dgemm)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, double, gemm));
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(__wrap_sgemm)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, float, gemm));
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(__wrap_dgemv)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, double, gemv));
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(__wrap_sgemv)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, float, gemv));
LIBXSMM_APIEXT void __wrap_dgemm_batch_strided(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, double, gemm_batch_strided));
LIBXSMM_APIEXT void __wrap_sgemm_batch_strided(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, float, gemm_batch_strided));
LIBXSMM_APIEXT void __wrap_dgemm_batch(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, double, gemm_batch));
LIBXSMM_APIEXT void __wrap_sgemm_batch(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, float, gemm_batch));
#endif
LIBXSMM_API void LIBXSMM_FSYMBOL(__real_dgemm_batch_strided)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, double, gemm_batch_strided));
LIBXSMM_API void LIBXSMM_FSYMBOL(__real_sgemm_batch_strided)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, float, gemm_batch_strided));
LIBXSMM_API void LIBXSMM_FSYMBOL(__real_dgemm_batch)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, double, gemm_batch));
LIBXSMM_API void LIBXSMM_FSYMBOL(__real_sgemm_batch)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, float, gemm_batch));
LIBXSMM_API void LIBXSMM_FSYMBOL(__real_dgemm)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, double, gemm));
LIBXSMM_API void LIBXSMM_FSYMBOL(__real_sgemm)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, float, gemm));
LIBXSMM_API void LIBXSMM_FSYMBOL(__real_dgemv)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, double, gemv));
LIBXSMM_API void LIBXSMM_FSYMBOL(__real_sgemv)(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, float, gemv));
LIBXSMM_API void __real_dgemm_batch_strided(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, double, gemm_batch_strided));
LIBXSMM_API void __real_sgemm_batch_strided(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, float, gemm_batch_strided));
LIBXSMM_API void __real_dgemm_batch(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, double, gemm_batch));
LIBXSMM_API void __real_sgemm_batch(LIBXSMM_BLAS_SYMBOL_SIGNATURE(const*, *, float, gemm_batch));
#endif

LIBXSMM_BLAS_SYMBOL_FDECL(LIBXSMM_BLAS_CONST*, *, double, gemm_batch_strided);
LIBXSMM_BLAS_SYMBOL_CDECL(LIBXSMM_BLAS_CONST*, *, double, gemm_batch_strided);
LIBXSMM_BLAS_SYMBOL_FDECL(LIBXSMM_BLAS_CONST*, *, float, gemm_batch_strided);
LIBXSMM_BLAS_SYMBOL_CDECL(LIBXSMM_BLAS_CONST*, *, float, gemm_batch_strided);
LIBXSMM_BLAS_SYMBOL_FDECL(LIBXSMM_BLAS_CONST*, *, double, gemm_batch);
LIBXSMM_BLAS_SYMBOL_CDECL(LIBXSMM_BLAS_CONST*, *, double, gemm_batch);
LIBXSMM_BLAS_SYMBOL_FDECL(LIBXSMM_BLAS_CONST*, *, float, gemm_batch);
LIBXSMM_BLAS_SYMBOL_CDECL(LIBXSMM_BLAS_CONST*, *, float, gemm_batch);
LIBXSMM_BLAS_SYMBOL_FDECL(LIBXSMM_BLAS_CONST*, *, double, gemm);
LIBXSMM_BLAS_SYMBOL_FDECL(LIBXSMM_BLAS_CONST*, *, float, gemm);
LIBXSMM_BLAS_SYMBOL_FDECL(LIBXSMM_BLAS_CONST*, *, double, gemv);
LIBXSMM_BLAS_SYMBOL_FDECL(LIBXSMM_BLAS_CONST*, *, float, gemv);

/** Helper for tuning the given gemm_flags for batches of SMMs (NTS-hint). */
LIBXSMM_API libxsmm_bitfield libxsmm_gemm_batch_flags(
  int gemm_flags, const libxsmm_gemm_shape* gemm_shape, const void* c,
  /**
   * If the returned value is larger than zero, the vector-length (in Bytes)
   * is larger than C's element-width and it can be used to check against a
   * stride of subsequent C-addresses, i.e., there is sufficient alignment
   * if 0 == LIBXSMM_MOD2(stride_in_byte, *vlen) and the tuned flag
   * can be adopted.
   * The vlen argument can be NULL if no further check is desired.
   */
  int* vlen);

LIBXSMM_API int libxsmm_gemm_batch_kernel(libxsmm_gemmfunction kernel, libxsmm_blasint index_base,
  libxsmm_blasint index_stride, const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const void* a, const void* b, void* c, libxsmm_blasint batchsize, /*unsigned*/int tid, /*unsigned*/int ntasks,
  unsigned char itypesize, unsigned char otypesize, int flags);

LIBXSMM_API void libxsmm_gemm_batch_blas(libxsmm_datatype iprec, libxsmm_datatype oprec,
  const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const libxsmm_blasint stride_a[],
  const void* b, const libxsmm_blasint* ldb, const libxsmm_blasint stride_b[],
  const void* beta, void* c, const libxsmm_blasint* ldc, const libxsmm_blasint stride_c[],
  libxsmm_blasint index_stride, libxsmm_blasint index_base, libxsmm_blasint batchsize);

/**
 * General dense matrix multiplication, which re-exposes LAPACK/BLAS
 * but allows to rely on LIBXSMM's defaults (libxsmm_config.h)
 * when supplying NULL-arguments in certain places.
 */
LIBXSMM_API void libxsmm_blas_gemm(
  libxsmm_datatype iprec, libxsmm_datatype oprec, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
                     const void* b, const libxsmm_blasint* ldb,
  const void* beta,        void* c, const libxsmm_blasint* ldc);

#define libxsmm_blas_dgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  libxsmm_blas_gemm(LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, \
    TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#define libxsmm_blas_sgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  libxsmm_blas_gemm(LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, \
    TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)

LIBXSMM_API void libxsmm_sink(const void* arg, ...);

/** Minimum batchsize per thread/task. */
LIBXSMM_APIVAR_PUBLIC(unsigned int libxsmm_gemm_taskgrain);
/** Determines if OpenMP tasks are used. */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_gemm_tasks);
/**
 * Intercepted GEMM
 * - [>=1 and  odd]: sequential and non-tiled (small problem sizes only)
 * - [>=2 and even]: parallelized and tiled (all problem sizes)
 * - [>=3 and  odd]: GEMV is intercepted; small problem sizes
 * - [>=4 and even]: GEMV is intercepted; all problem sizes
 * - [0]: disabled
 */
LIBXSMM_APIVAR_PUBLIC(int libxsmm_gemm_wrap);

#endif /*LIBXSMM_GEMM_H*/
