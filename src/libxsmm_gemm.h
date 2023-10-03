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
