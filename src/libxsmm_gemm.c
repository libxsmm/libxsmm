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
#include "libxsmm_gemm.h"
#include "libxsmm_xcopy.h"
#include "libxsmm_hash.h"
#include <libxsmm_mhd.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if !defined(LIBXSMM_NO_LIBM)
# include <math.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if defined(LIBXSMM_GEMM_BATCHREDUCE) && 0
# define LIBXSMM_GEMM_BATCHREDUCE
#endif
#if defined(LIBXSMM_BUILD)
# define LIBXSMM_GEMM_WEAK LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK
#else
# define LIBXSMM_GEMM_WEAK LIBXSMM_API
#endif

#if (0 != LIBXSMM_SYNC) /** Locks for the batch interface (duplicated C indexes). */
# define LIBXSMM_GEMM_LOCKIDX(IDX, NPOT) LIBXSMM_MOD2(LIBXSMM_CRC32U(LIBXSMM_BLASINT_NBITS)(2507/*seed*/, &(IDX)), NPOT)
# define LIBXSMM_GEMM_LOCKPTR(PTR, NPOT) LIBXSMM_MOD2(LIBXSMM_CRCPTR(1975/*seed*/, PTR), NPOT)
# if !defined(LIBXSMM_GEMM_MAXNLOCKS)
#   define LIBXSMM_GEMM_MAXNLOCKS 1024
# endif
# if !defined(LIBXSMM_GEMM_LOCKFWD)
#   define LIBXSMM_GEMM_LOCKFWD
# endif
# if LIBXSMM_LOCK_TYPE_ISPOD(LIBXSMM_GEMM_LOCK)
LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE internal_gemm_locktype {
  char pad[LIBXSMM_CACHELINE];
  LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK) state;
} internal_gemm_locktype;
# else
LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE internal_gemm_locktype {
  LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK) state;
} internal_gemm_locktype;
# endif
LIBXSMM_APIVAR_DEFINE(internal_gemm_locktype internal_gemm_lock[LIBXSMM_GEMM_MAXNLOCKS]);
LIBXSMM_APIVAR_DEFINE(unsigned int internal_gemm_nlocks); /* populated number of locks */
#endif

/* definition of corresponding variables */
LIBXSMM_APIVAR_PUBLIC_DEF(/*volatile*/libxsmm_dgemm_batch_function libxsmm_original_dgemm_batch_function);
LIBXSMM_APIVAR_PUBLIC_DEF(/*volatile*/libxsmm_sgemm_batch_function libxsmm_original_sgemm_batch_function);
LIBXSMM_APIVAR_PUBLIC_DEF(/*volatile*/libxsmm_dgemm_function libxsmm_original_dgemm_function);
LIBXSMM_APIVAR_PUBLIC_DEF(/*volatile*/libxsmm_sgemm_function libxsmm_original_sgemm_function);
LIBXSMM_APIVAR_PUBLIC_DEF(/*volatile*/libxsmm_dgemv_function libxsmm_original_dgemv_function);
LIBXSMM_APIVAR_PUBLIC_DEF(/*volatile*/libxsmm_sgemv_function libxsmm_original_sgemv_function);
/* definition of corresponding variables */
LIBXSMM_APIVAR_PUBLIC_DEF(libxsmm_gemm_descriptor libxsmm_mmbatch_desc);
LIBXSMM_APIVAR_PUBLIC_DEF(void* libxsmm_mmbatch_array);
LIBXSMM_APIVAR_PUBLIC_DEF(LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK) libxsmm_mmbatch_lock);
LIBXSMM_APIVAR_PUBLIC_DEF(unsigned int libxsmm_mmbatch_size);
LIBXSMM_APIVAR_PUBLIC_DEF(unsigned int libxsmm_gemm_npargroups);
LIBXSMM_APIVAR_PUBLIC_DEF(unsigned int libxsmm_gemm_taskgrain);
LIBXSMM_APIVAR_PUBLIC_DEF(int libxsmm_gemm_wrap);

LIBXSMM_APIVAR_PRIVATE_DEF(libxsmm_gemm_prefetch_type libxsmm_gemm_auto_prefetch_default);
/** Determines the prefetch strategy, which is used in case of LIBXSMM_PREFETCH_AUTO. */
LIBXSMM_APIVAR_PRIVATE_DEF(libxsmm_gemm_prefetch_type libxsmm_gemm_auto_prefetch);

/** Determines if batch-reduce is enabled */
LIBXSMM_APIVAR_DEFINE(int internal_gemm_batchreduce);


#if defined(LIBXSMM_BUILD)
LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK void LIBXSMM_FSYMBOL(__real_dgemm_batch)(
  const char transa_array[], const char transb_array[], const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const double alpha_array[], const double* a_array[], const libxsmm_blasint lda_array[], const double* b_array[], const libxsmm_blasint ldb_array[],
  const double beta_array[], double* c_array[], const libxsmm_blasint ldc_array[], const libxsmm_blasint* group_count, const libxsmm_blasint group_size[])
{
#if (0 != LIBXSMM_BLAS)
# if defined(LIBXSMM_WRAP) && (0 > LIBXSMM_WRAP)
  if (0 > libxsmm_gemm_wrap) {
    LIBXSMM_FSYMBOL(dgemm_batch)(transa_array, transb_array, m_array, n_array, k_array,
      alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
      group_count, group_size);
  }
  else
# endif
  {
    const libxsmm_blasint ptrsize = sizeof(void*);
    libxsmm_blasint i, j = 0;
    LIBXSMM_ASSERT(NULL != transa_array && NULL != transb_array && NULL != group_count && NULL != group_size);
    LIBXSMM_ASSERT(NULL != m_array && NULL != n_array && NULL != k_array && NULL != lda_array && NULL != ldb_array && NULL != ldc_array);
    LIBXSMM_ASSERT(NULL != a_array && NULL != b_array && NULL != c_array && NULL != alpha_array && NULL != beta_array);
    for (i = 0; i < *group_count; ++i) {
      const libxsmm_blasint size = group_size[i];
      libxsmm_dmmbatch_blas(transa_array + i, transb_array + i, m_array[i], n_array[i], k_array[i], alpha_array + i,
        a_array + j, lda_array + i, b_array + j, ldb_array + i, beta_array + i,
        c_array + j, ldc_array + i, 0/*index_base*/, 0/*index_stride*/, &ptrsize, &ptrsize, &ptrsize, size);
      j += size;
    }
  }
#else
  libxsmm_blas_error("dgemm_batch")(transa_array, transb_array, m_array, n_array, k_array,
    alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
    group_count, group_size);
#endif
}


LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK void LIBXSMM_FSYMBOL(__real_sgemm_batch)(
  const char transa_array[], const char transb_array[], const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const float alpha_array[], const float* a_array[], const libxsmm_blasint lda_array[], const float* b_array[], const libxsmm_blasint ldb_array[],
  const float beta_array[], float* c_array[], const libxsmm_blasint ldc_array[], const libxsmm_blasint* group_count, const libxsmm_blasint group_size[])
{
#if (0 != LIBXSMM_BLAS)
# if defined(LIBXSMM_WRAP) && (0 > LIBXSMM_WRAP)
  if (0 > libxsmm_gemm_wrap) {
    LIBXSMM_FSYMBOL(sgemm_batch)(transa_array, transb_array, m_array, n_array, k_array,
      alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
      group_count, group_size);
  }
  else
# endif
  {
    const libxsmm_blasint ptrsize = sizeof(void*);
    libxsmm_blasint i;
    LIBXSMM_ASSERT(NULL != transa_array && NULL != transb_array && NULL != group_count && NULL != group_size);
    LIBXSMM_ASSERT(NULL != m_array && NULL != n_array && NULL != k_array && NULL != lda_array && NULL != ldb_array && NULL != ldc_array);
    LIBXSMM_ASSERT(NULL != a_array && NULL != b_array && NULL != c_array && NULL != alpha_array && NULL != beta_array);
    for (i = 0; i < *group_count; ++i) {
      const libxsmm_blasint size = group_size[i];
      libxsmm_smmbatch_blas(transa_array + i, transb_array + i, m_array[i], n_array[i], k_array[i], alpha_array + i,
        a_array + i, lda_array + i, b_array + i, ldb_array + i, beta_array + i,
        c_array + i, ldc_array + i, 0/*index_base*/, 0/*index_stride*/, &ptrsize, &ptrsize, &ptrsize, size);
    }
  }
#else
  libxsmm_blas_error("sgemm_batch")(transa_array, transb_array, m_array, n_array, k_array,
    alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
    group_count, group_size);
#endif
}


LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK void LIBXSMM_FSYMBOL(__real_dgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
#if (0 != LIBXSMM_BLAS)
  LIBXSMM_FSYMBOL(dgemm)((LIBXSMM_BLAS_CONST char*)transa, (LIBXSMM_BLAS_CONST char*)transb,
    (LIBXSMM_BLAS_CONST libxsmm_blasint*)m, (LIBXSMM_BLAS_CONST libxsmm_blasint*)n, (LIBXSMM_BLAS_CONST libxsmm_blasint*)k,
    (LIBXSMM_BLAS_CONST double*)alpha, (LIBXSMM_BLAS_CONST double*)a, (LIBXSMM_BLAS_CONST libxsmm_blasint*)lda,
                                       (LIBXSMM_BLAS_CONST double*)b, (LIBXSMM_BLAS_CONST libxsmm_blasint*)ldb,
    (LIBXSMM_BLAS_CONST double*) beta,                             c, (LIBXSMM_BLAS_CONST libxsmm_blasint*)ldc);
#else
  libxsmm_blas_error("dgemm")(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  LIBXSMM_INLINE_XGEMM(double, double, /* try producing a result even if LIBXSMM_INLINE_XGEMM is limited */
    transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#endif
}


LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK void LIBXSMM_FSYMBOL(__real_sgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
#if (0 != LIBXSMM_BLAS)
  LIBXSMM_FSYMBOL(sgemm)((LIBXSMM_BLAS_CONST char*)transa, (LIBXSMM_BLAS_CONST char*)transb,
    (LIBXSMM_BLAS_CONST libxsmm_blasint*)m, (LIBXSMM_BLAS_CONST libxsmm_blasint*)n, (LIBXSMM_BLAS_CONST libxsmm_blasint*)k,
    (LIBXSMM_BLAS_CONST float*)alpha, (LIBXSMM_BLAS_CONST float*)a, (LIBXSMM_BLAS_CONST libxsmm_blasint*)lda,
                                      (LIBXSMM_BLAS_CONST float*)b, (LIBXSMM_BLAS_CONST libxsmm_blasint*)ldb,
    (LIBXSMM_BLAS_CONST float*) beta,                            c, (LIBXSMM_BLAS_CONST libxsmm_blasint*)ldc);
#else
  libxsmm_blas_error("sgemm")(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  LIBXSMM_INLINE_XGEMM(float, float, /* try producing a result even if LIBXSMM_INLINE_XGEMM is limited */
    transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#endif
}


LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK void LIBXSMM_FSYMBOL(__real_dgemv)(const char* trans, const libxsmm_blasint* m, const libxsmm_blasint* n,
  const double* alpha, const double* a, const libxsmm_blasint* lda, const double* x, const libxsmm_blasint* incx,
  const double* beta, double* y, const libxsmm_blasint* incy)
{
#if (0 != LIBXSMM_BLAS)
  LIBXSMM_FSYMBOL(dgemv)((LIBXSMM_BLAS_CONST char*)trans, (LIBXSMM_BLAS_CONST libxsmm_blasint*)m, (LIBXSMM_BLAS_CONST libxsmm_blasint*)n,
    (LIBXSMM_BLAS_CONST double*)alpha, (LIBXSMM_BLAS_CONST double*)a, (LIBXSMM_BLAS_CONST libxsmm_blasint*)lda,
                                       (LIBXSMM_BLAS_CONST double*)x, (LIBXSMM_BLAS_CONST libxsmm_blasint*)incx,
    (LIBXSMM_BLAS_CONST double*) beta,                             y, (LIBXSMM_BLAS_CONST libxsmm_blasint*)incy);
#else
  libxsmm_blas_error("dgemv")(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
#endif
}


LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK void LIBXSMM_FSYMBOL(__real_sgemv)(const char* trans, const libxsmm_blasint* m, const libxsmm_blasint* n,
  const float* alpha, const float* a, const libxsmm_blasint* lda, const float* x, const libxsmm_blasint* incx,
  const float* beta, float* y, const libxsmm_blasint* incy)
{
#if (0 != LIBXSMM_BLAS)
  LIBXSMM_FSYMBOL(sgemv)((LIBXSMM_BLAS_CONST char*)trans, (LIBXSMM_BLAS_CONST libxsmm_blasint*)m, (LIBXSMM_BLAS_CONST libxsmm_blasint*)n,
    (LIBXSMM_BLAS_CONST float*)alpha, (LIBXSMM_BLAS_CONST float*)a, (LIBXSMM_BLAS_CONST libxsmm_blasint*)lda,
                                      (LIBXSMM_BLAS_CONST float*)x, (LIBXSMM_BLAS_CONST libxsmm_blasint*)incx,
    (LIBXSMM_BLAS_CONST float*) beta,                            y, (LIBXSMM_BLAS_CONST libxsmm_blasint*)incy);
#else
  libxsmm_blas_error("sgemv")(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
#endif
}


LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK void __real_dgemm_batch(
  const char transa_array[], const char transb_array[], const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const double alpha_array[], const double* a_array[], const libxsmm_blasint lda_array[], const double* b_array[], const libxsmm_blasint ldb_array[],
  const double beta_array[], double* c_array[], const libxsmm_blasint ldc_array[], const libxsmm_blasint* group_count, const libxsmm_blasint group_size[])
{
  LIBXSMM_FSYMBOL(__real_dgemm_batch)(transa_array, transb_array, m_array, n_array, k_array,
    alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
    group_count, group_size);
}


LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK void __real_sgemm_batch(
  const char transa_array[], const char transb_array[], const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const float alpha_array[], const float* a_array[], const libxsmm_blasint lda_array[], const float* b_array[], const libxsmm_blasint ldb_array[],
  const float beta_array[], float* c_array[], const libxsmm_blasint ldc_array[], const libxsmm_blasint* group_count, const libxsmm_blasint group_size[])
{
  LIBXSMM_FSYMBOL(__real_sgemm_batch)(transa_array, transb_array, m_array, n_array, k_array,
    alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
    group_count, group_size);
}
#endif /*defined(LIBXSMM_BUILD)*/


LIBXSMM_GEMM_WEAK libxsmm_dgemm_batch_function libxsmm_original_dgemm_batch(void)
{
#if (0 != LIBXSMM_BLAS) && defined(LIBXSMM_WRAP) && (0 > LIBXSMM_WRAP)
  LIBXSMM_BLAS_WRAPPER(1, double, gemm_batch, libxsmm_original_dgemm_batch_function, NULL/*unknown*/);
  /*LIBXSMM_ASSERT(NULL != libxsmm_original_dgemm_batch_function);*/
#else
  LIBXSMM_BLAS_WRAPPER(0, double, gemm_batch, libxsmm_original_dgemm_batch_function, NULL/*unknown*/);
#endif
  return libxsmm_original_dgemm_batch_function;
}


LIBXSMM_GEMM_WEAK libxsmm_sgemm_batch_function libxsmm_original_sgemm_batch(void)
{
#if (0 != LIBXSMM_BLAS) && defined(LIBXSMM_WRAP) && (0 > LIBXSMM_WRAP)
  LIBXSMM_BLAS_WRAPPER(1, float, gemm_batch, libxsmm_original_sgemm_batch_function, NULL/*unknown*/);
  /*LIBXSMM_ASSERT(NULL != libxsmm_original_sgemm_batch_function);*/
#else
  LIBXSMM_BLAS_WRAPPER(0, float, gemm_batch, libxsmm_original_sgemm_batch_function, NULL/*unknown*/);
#endif
  return libxsmm_original_sgemm_batch_function;
}


LIBXSMM_GEMM_WEAK libxsmm_dgemm_function libxsmm_original_dgemm(void)
{
#if (0 != LIBXSMM_BLAS)
  LIBXSMM_BLAS_WRAPPER(1, double, gemm, libxsmm_original_dgemm_function, NULL/*unknown*/);
  LIBXSMM_ASSERT(NULL != libxsmm_original_dgemm_function);
#else
  LIBXSMM_BLAS_WRAPPER(0, double, gemm, libxsmm_original_dgemm_function, NULL/*unknown*/);
#endif
  return libxsmm_original_dgemm_function;
}


LIBXSMM_GEMM_WEAK libxsmm_sgemm_function libxsmm_original_sgemm(void)
{
#if (0 != LIBXSMM_BLAS)
  LIBXSMM_BLAS_WRAPPER(1, float, gemm, libxsmm_original_sgemm_function, NULL/*unknown*/);
  LIBXSMM_ASSERT(NULL != libxsmm_original_sgemm_function);
#else
  LIBXSMM_BLAS_WRAPPER(0, float, gemm, libxsmm_original_sgemm_function, NULL/*unknown*/);
#endif
  return libxsmm_original_sgemm_function;
}


LIBXSMM_GEMM_WEAK libxsmm_dgemv_function libxsmm_original_dgemv(void)
{
#if (0 != LIBXSMM_BLAS)
  LIBXSMM_BLAS_WRAPPER(1, double, gemv, libxsmm_original_dgemv_function, NULL/*unknown*/);
  LIBXSMM_ASSERT(NULL != libxsmm_original_dgemv_function);
#else
  LIBXSMM_BLAS_WRAPPER(0, double, gemv, libxsmm_original_dgemv_function, NULL/*unknown*/);
#endif
  return libxsmm_original_dgemv_function;
}


LIBXSMM_GEMM_WEAK libxsmm_sgemv_function libxsmm_original_sgemv(void)
{
#if (0 != LIBXSMM_BLAS)
  LIBXSMM_BLAS_WRAPPER(1, float, gemv, libxsmm_original_sgemv_function, NULL/*unknown*/);
  LIBXSMM_ASSERT(NULL != libxsmm_original_sgemv_function);
#else
  LIBXSMM_BLAS_WRAPPER(0, float, gemv, libxsmm_original_sgemv_function, NULL/*unknown*/);
#endif
  return libxsmm_original_sgemv_function;
}


LIBXSMM_API libxsmm_sink_function libxsmm_blas_error(const char* symbol)
{
  static int error_once = 0;
  LIBXSMM_BLAS_ERROR(symbol, &error_once);
  return libxsmm_sink;
}


LIBXSMM_API_INTERN void libxsmm_gemm_init()
{
  LIBXSMM_LOCK_ATTR_TYPE(LIBXSMM_GEMM_LOCK) attr = { 0 };
  LIBXSMM_LOCK_ATTR_INIT(LIBXSMM_GEMM_LOCK, &attr);
#if defined(LIBXSMM_WRAP) /* determines if wrap is considered */
  { /* intercepted GEMMs (1: sequential and non-tiled, 2: parallelized and tiled) */
    const char *const env_wrap = getenv("LIBXSMM_GEMM_WRAP");
# if defined(__STATIC) /* with static library the user controls interceptor already */
    libxsmm_gemm_wrap = ((NULL == env_wrap || 0 == *env_wrap) /* LIBXSMM_WRAP=0: no promotion */
      ? (0 < (LIBXSMM_WRAP) ? (LIBXSMM_WRAP + 2) : (LIBXSMM_WRAP - 2)) : atoi(env_wrap));
# else
    libxsmm_gemm_wrap = ((NULL == env_wrap || 0 == *env_wrap) ? (LIBXSMM_WRAP) : atoi(env_wrap));
# endif
  }
#endif
#if (0 != LIBXSMM_SYNC)
  { /* initialize locks for the batch interface */
    const char *const env_nlocks = getenv("LIBXSMM_GEMM_NLOCKS");
    const int nlocks = ((NULL == env_nlocks || 0 == *env_nlocks) ? -1/*default*/ : atoi(env_nlocks));
    unsigned int i;
    internal_gemm_nlocks = LIBXSMM_UP2POT(0 > nlocks ? (LIBXSMM_GEMM_MAXNLOCKS) : LIBXSMM_MIN(nlocks, LIBXSMM_GEMM_MAXNLOCKS));
    for (i = 0; i < internal_gemm_nlocks; ++i) LIBXSMM_LOCK_INIT(LIBXSMM_GEMM_LOCK, &internal_gemm_lock[i].state, &attr);
  }
#endif
  { /* determines grain-size of tasks (when available) */
    const char *const env_npargroups = getenv("LIBXSMM_GEMM_NPARGROUPS");
    libxsmm_gemm_npargroups = ((NULL == env_npargroups || 0 == *env_npargroups || 0 >= atoi(env_npargroups))
      ? (LIBXSMM_GEMM_NPARGROUPS) : atoi(env_npargroups));
  }
  LIBXSMM_LOCK_ATTR_DESTROY(LIBXSMM_GEMM_LOCK, &attr);
  /* determine BLAS function-pointers */
  libxsmm_original_dgemm_batch();
  libxsmm_original_sgemm_batch();
  libxsmm_original_dgemm();
  libxsmm_original_sgemm();
  libxsmm_original_dgemv();
  libxsmm_original_sgemv();
}


LIBXSMM_API_INTERN void libxsmm_gemm_finalize(void)
{
#if (0 != LIBXSMM_SYNC)
  unsigned int i; for (i = 0; i < internal_gemm_nlocks; ++i) LIBXSMM_LOCK_DESTROY(LIBXSMM_GEMM_LOCK, &internal_gemm_lock[i].state);
#endif
}


LIBXSMM_API libxsmm_gemm_prefetch_type libxsmm_get_gemm_xprefetch(const int* prefetch)
{
  LIBXSMM_INIT /* load configuration */
  return libxsmm_get_gemm_prefetch(NULL == prefetch ? ((int)libxsmm_gemm_auto_prefetch) : *prefetch);
}


LIBXSMM_API libxsmm_gemm_prefetch_type libxsmm_get_gemm_prefetch(int prefetch)
{
  libxsmm_gemm_prefetch_type result;
#if !defined(_WIN32) && !defined(__CYGWIN__) && !defined(__MINGW32__)
  if (0 > prefetch) {
    LIBXSMM_INIT /* load configuration */
    result = libxsmm_gemm_auto_prefetch_default;
  }
  else {
    result = (libxsmm_gemm_prefetch_type)prefetch;
  }
#else /* TODO: full support for Windows calling convention */
  result = LIBXSMM_GEMM_PREFETCH_NONE;
  LIBXSMM_UNUSED(prefetch);
#endif
  return result;
}


LIBXSMM_API_INTERN int libxsmm_gemm_prefetch2uid(libxsmm_gemm_prefetch_type prefetch)
{
  switch ((int)prefetch) {
    case LIBXSMM_GEMM_PREFETCH_SIGONLY:            return 2;
    case LIBXSMM_GEMM_PREFETCH_BL2_VIA_C:          return 3;
    case LIBXSMM_GEMM_PREFETCH_AL2_AHEAD:          return 4;
    case LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD: return 5;
    case LIBXSMM_GEMM_PREFETCH_AL2:                return 6;
    case LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C:       return 7;
    case LIBXSMM_GEMM_PREFETCH_BRGEMM_OOB:         return 8;
    default: {
      LIBXSMM_ASSERT(LIBXSMM_GEMM_PREFETCH_NONE == prefetch);
      return 0;
    }
  }
}


LIBXSMM_API_INTERN libxsmm_gemm_prefetch_type libxsmm_gemm_uid2prefetch(int uid)
{
  switch (uid) {
    case 1: return LIBXSMM_GEMM_PREFETCH_NONE;               /* nopf */
    case 2: return LIBXSMM_GEMM_PREFETCH_SIGONLY;            /* pfsigonly */
    case 3: return LIBXSMM_GEMM_PREFETCH_BL2_VIA_C;          /* BL2viaC */
    case 4: return LIBXSMM_GEMM_PREFETCH_AL2_AHEAD;          /* curAL2 */
    case 5: return LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD; /* curAL2_BL2viaC */
    case 6: return LIBXSMM_GEMM_PREFETCH_AL2;                /* AL2 */
    case 7: return LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C;       /* AL2_BL2viaC */
    case 8: return LIBXSMM_GEMM_PREFETCH_BRGEMM_OOB;
    default: {
      if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
        static int error_once = 0;
        if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
          fprintf(stderr, "LIBXSMM WARNING: invalid prefetch strategy requested!\n");
        }
      }
      return LIBXSMM_GEMM_PREFETCH_NONE;
    }
  }
}


LIBXSMM_API void libxsmm_gemm_print(void* ostream,
  libxsmm_datatype precision, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
  const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc)
{
  libxsmm_gemm_print2(ostream, precision, precision, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_API void libxsmm_gemm_print2(void* ostream,
  libxsmm_datatype iprec, libxsmm_datatype oprec, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
  const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc)
{
  const libxsmm_blasint nn = *(n ? n : m), kk = *(k ? k : m);
  const char ctransa = (char)(NULL != transa ? (*transa) : (0 == (LIBXSMM_FLAGS & LIBXSMM_GEMM_FLAG_TRANS_A) ? 'n' : 't'));
  const char ctransb = (char)(NULL != transb ? (*transb) : (0 == (LIBXSMM_FLAGS & LIBXSMM_GEMM_FLAG_TRANS_B) ? 'n' : 't'));
  const libxsmm_blasint ilda = (NULL != lda ? *lda : (('n' == ctransa || 'N' == ctransa) ? *m : kk));
  const libxsmm_blasint ildb = (NULL != ldb ? *ldb : (('n' == ctransb || 'N' == ctransb) ? kk : nn));
  const libxsmm_blasint ildc = *(NULL != ldc ? ldc : m);
  libxsmm_mhd_elemtype mhd_elemtype = LIBXSMM_MHD_ELEMTYPE_UNKNOWN;
  char string_a[128] = "", string_b[128] = "", typeprefix = 0;

  switch (iprec | oprec) {
    case LIBXSMM_DATATYPE_F64: {
      LIBXSMM_ASSERT(iprec == oprec);
      LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "%g", NULL != alpha ? *((const double*)alpha) : LIBXSMM_ALPHA);
      LIBXSMM_SNPRINTF(string_b, sizeof(string_b), "%g", NULL != beta  ? *((const double*)beta)  : LIBXSMM_BETA);
      mhd_elemtype = LIBXSMM_MHD_ELEMTYPE_F64;
      typeprefix = 'd';
    } break;
    case LIBXSMM_DATATYPE_F32: {
      LIBXSMM_ASSERT(iprec == oprec);
      LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "%g", NULL != alpha ? *((const float*)alpha) : LIBXSMM_ALPHA);
      LIBXSMM_SNPRINTF(string_b, sizeof(string_b), "%g", NULL != beta  ? *((const float*)beta)  : LIBXSMM_BETA);
      mhd_elemtype = LIBXSMM_MHD_ELEMTYPE_F32;
      typeprefix = 's';
    } break;
    default: if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
      static int error_once = 0;
      if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) { /* TODO: support I16, etc. */
        fprintf(stderr, "LIBXSMM ERROR: unsupported data-type requested!\n");
      }
    }
  }

  if (0 != typeprefix) {
    if (NULL != ostream) { /* print information about GEMM call */
      if (NULL != a && NULL != b && NULL != c) {
        fprintf((FILE*)ostream, "%cgemm('%c', '%c', %" PRIuPTR "/*m*/, %" PRIuPTR "/*n*/, %" PRIuPTR "/*k*/,\n"
                                "  %s/*alpha*/, %p/*a*/, %" PRIuPTR "/*lda*/,\n"
                                "              %p/*b*/, %" PRIuPTR "/*ldb*/,\n"
                                "   %s/*beta*/, %p/*c*/, %" PRIuPTR "/*ldc*/)",
          typeprefix, ctransa, ctransb, (uintptr_t)*m, (uintptr_t)nn, (uintptr_t)kk,
          string_a, a, (uintptr_t)ilda, b, (uintptr_t)ildb, string_b, c, (uintptr_t)ildc);
      }
      else {
        fprintf((FILE*)ostream, "%cgemm(trans=%c%c mnk=%" PRIuPTR ",%" PRIuPTR ",%" PRIuPTR
                                                 " ldx=%" PRIuPTR ",%" PRIuPTR ",%" PRIuPTR " a,b=%s,%s)",
          typeprefix, ctransa, ctransb, (uintptr_t)*m, (uintptr_t)nn, (uintptr_t)kk,
          (uintptr_t)ilda, (uintptr_t)ildb, (uintptr_t)ildc, string_a, string_b);
      }
    }
    else { /* dump A, B, and C matrices into MHD files */
      char extension_header[256] = "";
      size_t data_size[2] = { 0 }, size[2] = { 0 };

      if (NULL != a) {
        LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "TRANS = %c\nALPHA = %s", ctransa, string_a);
        LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "libxsmm_a_%p.mhd", a);
        data_size[0] = (size_t)ilda; data_size[1] = (size_t)kk; size[0] = (size_t)(*m); size[1] = (size_t)kk;
        libxsmm_mhd_write(string_a, NULL/*offset*/, size, data_size, 2/*ndims*/, 1/*ncomponents*/, mhd_elemtype,
          NULL/*conversion*/, a, NULL/*header_size*/, extension_header, NULL/*extension*/, 0/*extension_size*/);
      }
      if (NULL != b) {
        LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "\nTRANS = %c", ctransb);
        LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "libxsmm_b_%p.mhd", b);
        data_size[0] = (size_t)ildb; data_size[1] = (size_t)nn; size[0] = (size_t)kk; size[1] = (size_t)nn;
        libxsmm_mhd_write(string_a, NULL/*offset*/, size, data_size, 2/*ndims*/, 1/*ncomponents*/, mhd_elemtype,
          NULL/*conversion*/, b, NULL/*header_size*/, extension_header, NULL/*extension*/, 0/*extension_size*/);
      }
      if (NULL != c) {
        LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "BETA = %s", string_b);
        LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "libxsmm_c_%p.mhd", c);
        data_size[0] = (size_t)ildc; data_size[1] = (size_t)nn; size[0] = (size_t)(*m); size[1] = (size_t)nn;
        libxsmm_mhd_write(string_a, NULL/*offset*/, size, data_size, 2/*ndims*/, 1/*ncomponents*/, mhd_elemtype,
          NULL/*conversion*/, c, NULL/*header_size*/, extension_header, NULL/*extension*/, 0/*extension_size*/);
      }
    }
  }
}


LIBXSMM_API void libxsmm_gemm_dprint(
  void* ostream, libxsmm_datatype precision, char transa, char transb,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, double dalpha, const void* a, libxsmm_blasint lda,
  const void* b, libxsmm_blasint ldb, double dbeta, void* c, libxsmm_blasint ldc)
{
  libxsmm_gemm_dprint2(ostream, precision, precision, transa, transb, m, n, k, dalpha, a, lda, b, ldb, dbeta, c, ldc);
}


LIBXSMM_API void libxsmm_gemm_dprint2(
  void* ostream, libxsmm_datatype iprec, libxsmm_datatype oprec, char transa, char transb,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, double dalpha, const void* a, libxsmm_blasint lda,
  const void* b, libxsmm_blasint ldb, double dbeta, void* c, libxsmm_blasint ldc)
{
  switch ((int)iprec) {
    case LIBXSMM_DATATYPE_F64: {
      libxsmm_gemm_print2(ostream, LIBXSMM_DATATYPE_F64, oprec, &transa, &transb,
        &m, &n, &k, &dalpha, a, &lda, b, &ldb, &dbeta, c, &ldc);
    } break;
    case LIBXSMM_DATATYPE_F32: {
      const float alpha = (float)dalpha, beta = (float)dbeta;
      libxsmm_gemm_print2(ostream, LIBXSMM_DATATYPE_F32, oprec, &transa, &transb,
        &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    } break;
    default: {
      libxsmm_gemm_print2(ostream, iprec, oprec, &transa, &transb,
        &m, &n, &k, &dalpha, a, &lda, b, &ldb, &dbeta, c, &ldc);
    }
  }
}


LIBXSMM_API void libxsmm_gemm_xprint(void* ostream,
  libxsmm_xmmfunction kernel, const void* a, const void* b, void* c)
{
  const libxsmm_descriptor* desc;
  libxsmm_code_pointer code = { NULL };
  size_t code_size;
  code.xgemm = kernel;
  if (NULL != libxsmm_get_kernel_xinfo(code, &desc, &code_size) &&
      NULL != desc && LIBXSMM_KERNEL_KIND_MATMUL == LIBXSMM_DESCRIPTOR_KIND(desc->kind))
  {
    libxsmm_gemm_dprint2(ostream,
      (libxsmm_datatype)LIBXSMM_GETENUM_INP(desc->gemm.desc.datatype),
      (libxsmm_datatype)LIBXSMM_GETENUM_OUT(desc->gemm.desc.datatype),
      (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & desc->gemm.desc.flags) ? 'N' : 'T'),
      (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_B & desc->gemm.desc.flags) ? 'N' : 'T'),
      (libxsmm_blasint)desc->gemm.desc.m, (libxsmm_blasint)desc->gemm.desc.n, (libxsmm_blasint)desc->gemm.desc.k,
      1, a, (libxsmm_blasint)desc->gemm.desc.lda, b, (libxsmm_blasint)desc->gemm.desc.ldb,
      0 != (LIBXSMM_GEMM_FLAG_BETA_0 & libxsmm_mmbatch_desc.flags) ? 0 : 1, c, (libxsmm_blasint)desc->gemm.desc.ldc);
    fprintf((FILE*)ostream, " = %p+%u", code.ptr_const, (unsigned int)code_size);
  }
}


LIBXSMM_API void libxsmm_blas_xgemm(libxsmm_datatype iprec, libxsmm_datatype oprec,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_INIT
  switch ((int)iprec) {
    case LIBXSMM_DATATYPE_F64: {
      LIBXSMM_ASSERT(iprec == oprec);
      LIBXSMM_BLAS_XGEMM(double, double, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } break;
    case LIBXSMM_DATATYPE_F32: {
      LIBXSMM_ASSERT(iprec == oprec);
      LIBXSMM_BLAS_XGEMM(float, float, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } break;
    default: if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
      static int error_once = 0;
      LIBXSMM_UNUSED(oprec);
      if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) { /* TODO: support I16, etc. */
        fprintf(stderr, "LIBXSMM ERROR: unsupported data-type requested!\n");
      }
    }
  }
}


LIBXSMM_API void libxsmm_dgemm_batch(
  const char transa_array[], const char transb_array[], const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const double alpha_array[], const double* a_array[], const libxsmm_blasint lda_array[], const double* b_array[], const libxsmm_blasint ldb_array[],
  const double beta_array[], double* c_array[], const libxsmm_blasint ldc_array[], const libxsmm_blasint* group_count, const libxsmm_blasint group_size[])
{
  const libxsmm_blasint ngroups = LIBXSMM_ABS(*group_count), ptrsize = sizeof(void*);
  libxsmm_blasint i, j = 0;
  for (i = 0; i < ngroups; ++i) {
    const libxsmm_blasint size = group_size[i];
    libxsmm_gemm_batch(LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, transa_array + i, transb_array + i,
      m_array[i], n_array[i], k_array[i], alpha_array + i, a_array + j, lda_array + i, b_array + j, ldb_array + i, beta_array + i, c_array + j, ldc_array + i,
      0/*index_base*/, 0/*index_stride*/, &ptrsize, &ptrsize, &ptrsize, size);
    j += LIBXSMM_ABS(size);
  }
}


LIBXSMM_API void libxsmm_sgemm_batch(
  const char transa_array[], const char transb_array[], const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const float alpha_array[], const float* a_array[], const libxsmm_blasint lda_array[], const float* b_array[], const libxsmm_blasint ldb_array[],
  const float beta_array[], float* c_array[], const libxsmm_blasint ldc_array[], const libxsmm_blasint* group_count, const libxsmm_blasint group_size[])
{
  const libxsmm_blasint ngroups = LIBXSMM_ABS(*group_count), ptrsize = sizeof(void*);
  libxsmm_blasint i, j = 0;
  for (i = 0; i < ngroups; ++i) {
    const libxsmm_blasint size = group_size[i];
    libxsmm_gemm_batch(LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, transa_array + i, transb_array + i,
      m_array[i], n_array[i], k_array[i], alpha_array + i, a_array + j, lda_array + i, b_array + j, ldb_array + i, beta_array + i, c_array + j, ldc_array + i,
      0/*index_base*/, 0/*index_stride*/, &ptrsize, &ptrsize, &ptrsize, size);
    j += LIBXSMM_ABS(size);
  }
}


LIBXSMM_API void libxsmm_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_XGEMM(double, double, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_API void libxsmm_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_XGEMM(float, float, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_API int libxsmm_mmbatch_kernel(libxsmm_xmmfunction kernel, libxsmm_blasint index_base,
  libxsmm_blasint index_stride, const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const void* a, const void* b, void* c, libxsmm_blasint batchsize, /*unsigned*/int tid, /*unsigned*/int ntasks,
  unsigned char itypesize, unsigned char otypesize, int flags)
{
  int result = EXIT_SUCCESS;
  const libxsmm_blasint size = LIBXSMM_ABS(batchsize);
  const libxsmm_blasint tasksize = LIBXSMM_UPDIV(size, ntasks);
  const libxsmm_blasint begin = tid * tasksize, span = begin + tasksize;
  const libxsmm_blasint end = LIBXSMM_MIN(span, size);

  LIBXSMM_ASSERT(NULL != a && NULL != b && NULL != c && NULL != kernel.xmm);
  if (begin < end) {
    const char *const a0 = (const char*)a, *const b0 = (const char*)b;
    char *const c0 = (char*)c;
    LIBXSMM_ASSERT(0 < itypesize && 0 < otypesize);
    if (0 == (LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS & flags)) {
      if (0 != index_stride) { /* stride arrays contain indexes */
        libxsmm_blasint i = begin * index_stride, ic = (NULL != stride_c ? (LIBXSMM_VALUE1(const libxsmm_blasint, stride_c, i) - index_base) : 0);
        const char* ai = &a0[NULL != stride_a ? ((LIBXSMM_VALUE1(const libxsmm_blasint, stride_a, i) - index_base) * itypesize) : 0];
        const char* bi = &b0[NULL != stride_b ? ((LIBXSMM_VALUE1(const libxsmm_blasint, stride_b, i) - index_base) * itypesize) : 0];
        char*       ci = &c0[ic * otypesize];
        const libxsmm_blasint end1 = (end != size ? end : (end - 1)) * index_stride;
#if (0 != LIBXSMM_SYNC)
        if (1 == ntasks || 0 == internal_gemm_nlocks || 0 > batchsize || 0 != (LIBXSMM_GEMM_FLAG_BETA_0 & flags))
#endif
        { /* no locking */
          if (NULL != stride_a && NULL != stride_b && NULL != stride_c) {
            const unsigned char ibits = (unsigned char)LIBXSMM_INTRINSICS_BITSCANBWD32(itypesize);
            const unsigned char obits = (unsigned char)LIBXSMM_INTRINSICS_BITSCANBWD32(otypesize);

            if (itypesize == (1 << ibits) && otypesize == (1 << obits)) {
              for (i += index_stride; i <= end1; i += index_stride) {
                const char *const an = &a0[(LIBXSMM_VALUE1(const libxsmm_blasint, stride_a, i) - index_base) << ibits];
                const char *const bn = &b0[(LIBXSMM_VALUE1(const libxsmm_blasint, stride_b, i) - index_base) << ibits];
                char       *const cn = &c0[(LIBXSMM_VALUE1(const libxsmm_blasint, stride_c, i) - index_base) << obits];
                kernel.xmm(ai, bi, ci/*, an, bn, cn*/); /* @TODO fix prefetch */
                ai = an; bi = bn; ci = cn; /* next */
              }
            }
            else { /* non-pot type sizes */
              for (i += index_stride; i <= end1; i += index_stride) {
                const char *const an = &a0[(LIBXSMM_VALUE1(const libxsmm_blasint, stride_a, i) - index_base) * itypesize];
                const char *const bn = &b0[(LIBXSMM_VALUE1(const libxsmm_blasint, stride_b, i) - index_base) * itypesize];
                char       *const cn = &c0[(LIBXSMM_VALUE1(const libxsmm_blasint, stride_c, i) - index_base) * otypesize];
                kernel.xmm(ai, bi, ci/*, an, bn, cn*/); /* @TODO fix prefetch */
                ai = an; bi = bn; ci = cn; /* next */
              }
            }
          }
          else { /* mixed specification of strides */
            for (i += index_stride; i <= end1; i += index_stride) {
              const char *const an = &a0[NULL != stride_a ? ((LIBXSMM_VALUE1(const libxsmm_blasint, stride_a, i) - index_base) * itypesize) : 0];
              const char *const bn = &b0[NULL != stride_b ? ((LIBXSMM_VALUE1(const libxsmm_blasint, stride_b, i) - index_base) * itypesize) : 0];
              char       *const cn = &c0[NULL != stride_c ? ((LIBXSMM_VALUE1(const libxsmm_blasint, stride_c, i) - index_base) * otypesize) : 0];
              kernel.xmm(ai, bi, ci/*, an, bn, cn*/); /* @TODO fix prefetch */
              ai = an; bi = bn; ci = cn; /* next */
            }
          }
          if (end == size) { /* remainder multiplication */
            kernel.xmm(ai, bi, ci/*, ai, bi, ci*/); /* @TODO fix prefetch */
          }
        }
#if (0 != LIBXSMM_SYNC)
        else { /* synchronize among C-indexes */
          LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK)* lock = &internal_gemm_lock[LIBXSMM_GEMM_LOCKIDX(ic, internal_gemm_nlocks)].state;
# if defined(LIBXSMM_GEMM_LOCKFWD)
          LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK)* lock0 = NULL;
# endif
          LIBXSMM_ASSERT(NULL != lock);
          if (NULL != stride_a && NULL != stride_b && NULL != stride_c) {
            for (i += index_stride; i <= end1; i += index_stride) {
              ic = LIBXSMM_VALUE1(const libxsmm_blasint, stride_c, i) - index_base;
              {
                const char *const an = &a0[(LIBXSMM_VALUE1(const libxsmm_blasint, stride_a, i) - index_base) * itypesize];
                const char *const bn = &b0[(LIBXSMM_VALUE1(const libxsmm_blasint, stride_b, i) - index_base) * itypesize];
                char       *const cn = &c0[ic * otypesize];
                LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK) *const lock1 = &internal_gemm_lock[LIBXSMM_GEMM_LOCKIDX(ic, internal_gemm_nlocks)].state;
# if defined(LIBXSMM_GEMM_LOCKFWD)
                if (lock != lock0) { lock0 = lock; LIBXSMM_LOCK_ACQUIRE(LIBXSMM_GEMM_LOCK, lock); }
# else
                LIBXSMM_LOCK_ACQUIRE(LIBXSMM_GEMM_LOCK, lock);
# endif
                kernel.xmm(ai, bi, ci/*, an, bn, cn*/); /* @TODO fix prefetch */
# if defined(LIBXSMM_GEMM_LOCKFWD)
                if (lock != lock1 || i == end1) { LIBXSMM_LOCK_RELEASE(LIBXSMM_GEMM_LOCK, lock); lock = lock1; }
# else
                LIBXSMM_LOCK_RELEASE(LIBXSMM_GEMM_LOCK, lock); lock = lock1;
# endif
                ai = an; bi = bn; ci = cn; /* next */
              }
            }
          }
          else {
            for (i += index_stride; i <= end1; i += index_stride) {
              ic = (NULL != stride_c ? (LIBXSMM_VALUE1(const libxsmm_blasint, stride_c, i) - index_base) : 0);
              {
                const char *const an = &a0[NULL != stride_a ? ((LIBXSMM_VALUE1(const libxsmm_blasint, stride_a, i) - index_base) * itypesize) : 0];
                const char *const bn = &b0[NULL != stride_b ? ((LIBXSMM_VALUE1(const libxsmm_blasint, stride_b, i) - index_base) * itypesize) : 0];
                char       *const cn = &c0[ic * otypesize];
                LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK) *const lock1 = &internal_gemm_lock[LIBXSMM_GEMM_LOCKIDX(ic, internal_gemm_nlocks)].state;
# if defined(LIBXSMM_GEMM_LOCKFWD)
                if (lock != lock0) { lock0 = lock; LIBXSMM_LOCK_ACQUIRE(LIBXSMM_GEMM_LOCK, lock); }
# else
                LIBXSMM_LOCK_ACQUIRE(LIBXSMM_GEMM_LOCK, lock);
# endif
                kernel.xmm(ai, bi, ci/*, an, bn, cn*/); /* @TODO fix prefetch */
# if defined(LIBXSMM_GEMM_LOCKFWD)
                if (lock != lock1 || i == end1) { LIBXSMM_LOCK_RELEASE(LIBXSMM_GEMM_LOCK, lock); lock = lock1; }
# else
                LIBXSMM_LOCK_RELEASE(LIBXSMM_GEMM_LOCK, lock); lock = lock1;
# endif
                ai = an; bi = bn; ci = cn; /* next */
              }
            }
          }
          if (end == size) { /* remainder multiplication */
            LIBXSMM_LOCK_ACQUIRE(LIBXSMM_GEMM_LOCK, lock);
            kernel.xmm(ai, bi, ci/*, ai, bi, ci*/); /* @TODO fix prefetch */
            LIBXSMM_LOCK_RELEASE(LIBXSMM_GEMM_LOCK, lock);
          }
        }
#endif /*(0 != LIBXSMM_SYNC)*/
      }
      else { /* array of pointers to matrices (singular strides are measured in Bytes) */
        const libxsmm_blasint da = (NULL != stride_a ? (*stride_a - index_base * sizeof(void*)) : 0);
        const libxsmm_blasint db = (NULL != stride_b ? (*stride_b - index_base * sizeof(void*)) : 0);
        const libxsmm_blasint dc = (NULL != stride_c ? (*stride_c - index_base * sizeof(void*)) : 0);
        const libxsmm_blasint end1 = (end != size ? end : (end - 1));
        const char *ai = a0 + (size_t)da * begin, *bi = b0 + (size_t)db * begin;
        char* ci = c0 + (size_t)dc * begin;
        libxsmm_blasint i;
#if (0 != LIBXSMM_SYNC)
        if (1 == ntasks || 0 == internal_gemm_nlocks || 0 > batchsize || 0 != (LIBXSMM_GEMM_FLAG_BETA_0 & flags))
#endif
        { /* no locking */
          for (i = begin; i < end1; ++i) {
            const char *const an = ai + da, *const bn = bi + db;
            char *const cn = ci + dc;
#if defined(LIBXSMM_BATCH_CHECK)
            if (NULL != *((const void**)ai) && NULL != *((const void**)bi) && NULL != *((const void**)ci))
#endif
            {
              kernel.xmm( /* with prefetch */
                *((const void**)ai), *((const void**)bi), *((void**)ci)/*,
                *((const void**)an), *((const void**)bn), *((const void**)cn)*/); /* @TODO fix prefetch */
            }
            ai = an; bi = bn; ci = cn; /* next */
          }
          if ( /* remainder multiplication */
#if defined(LIBXSMM_BATCH_CHECK)
            NULL != *((const void**)ai) && NULL != *((const void**)bi) && NULL != *((const void**)ci) &&
#endif
            end == size)
          {
            kernel.xmm( /* pseudo-prefetch */
              *((const void**)ai), *((const void**)bi), *((void**)ci)/*,
              *((const void**)ai), *((const void**)bi), *((const void**)ci)*/); /* @TODO fix prefetch */
          }
        }
#if (0 != LIBXSMM_SYNC)
        else { /* synchronize among C-indexes */
          void* cc = *((void**)ci);
          LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK)* lock = &internal_gemm_lock[LIBXSMM_GEMM_LOCKPTR(cc, internal_gemm_nlocks)].state;
# if defined(LIBXSMM_GEMM_LOCKFWD)
          LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK)* lock0 = NULL;
# endif
          LIBXSMM_ASSERT(NULL != lock);
          for (i = begin + 1; i <= end1; ++i) {
            const char *const an = ai + da, *const bn = bi + db;
            char *const cn = ci + dc;
            void *const nc = *((void**)cn);
# if defined(LIBXSMM_BATCH_CHECK)
            if (NULL != *((const void**)ai) && NULL != *((const void**)bi) && NULL != cc)
# endif
            {
              LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK) *const lock1 = &internal_gemm_lock[LIBXSMM_GEMM_LOCKPTR(nc, internal_gemm_nlocks)].state;
# if defined(LIBXSMM_GEMM_LOCKFWD)
              if (lock != lock0) { lock0 = lock; LIBXSMM_LOCK_ACQUIRE(LIBXSMM_GEMM_LOCK, lock); }
# else
              LIBXSMM_LOCK_ACQUIRE(LIBXSMM_GEMM_LOCK, lock);
# endif
              kernel.xmm( /* with prefetch */
                *((const void**)ai), *((const void**)bi), cc/*,
                *((const void**)an), *((const void**)bn), *((const void**)cn)*/); /* @TODO fix prefetch */
# if defined(LIBXSMM_GEMM_LOCKFWD)
              if (lock != lock1 || i == end1) { LIBXSMM_LOCK_RELEASE(LIBXSMM_GEMM_LOCK, lock); lock = lock1; }
# else
              LIBXSMM_LOCK_RELEASE(LIBXSMM_GEMM_LOCK, lock); lock = lock1;
# endif
            }
            ai = an; bi = bn; ci = cn; cc = nc; /* next */
          }
          if ( /* remainder multiplication */
# if defined(LIBXSMM_BATCH_CHECK)
            NULL != *((const void**)ai) && NULL != *((const void**)bi) && NULL != cc &&
# endif
            end == size)
          {
            LIBXSMM_LOCK_ACQUIRE(LIBXSMM_GEMM_LOCK, lock);
            kernel.xmm( /* pseudo-prefetch */
              *((const void**)ai), *((const void**)bi), cc/*,
              *((const void**)ai), *((const void**)bi), cc*/); /* @TODO fix prefetch */
            LIBXSMM_LOCK_RELEASE(LIBXSMM_GEMM_LOCK, lock);
          }
        }
#endif /*(0 != LIBXSMM_SYNC)*/
      }
    }
#if defined(LIBXSMM_GEMM_BATCHREDUCE)
    else /* LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS */
# if defined(LIBXSMM_BATCH_CHECK)
    if (
#   if (0 != LIBXSMM_SYNC)
      (1 == ntasks || 0 == internal_gemm_nlocks || 0 > batchsize) &&
#   endif
      (0 == (LIBXSMM_GEMM_FLAG_BETA_0 & flags)) &&
      (0 != internal_gemm_batchreduce))
# endif
    {
      const unsigned int n = libxsmm_mmbatch_size * (LIBXSMM_GEMM_BATCHSCALE) / ((unsigned int)sizeof(void*));
      LIBXSMM_ASSERT(NULL != libxsmm_mmbatch_array && 0 != libxsmm_mmbatch_size);
      if ((2U/*A and B matrices*/ * tasksize) <= n) {
        const void **ai = (const void**)libxsmm_mmbatch_array + begin, **bi = ai + size;
        unsigned long long count;
        if (0 != index_stride) { /* stride arrays contain indexes */
          const size_t end_stride = (size_t)end * index_stride;
          size_t i = (size_t)begin * index_stride;
          char *ci = &c0[NULL != stride_c ? ((LIBXSMM_VALUE1(const libxsmm_blasint, stride_c, i) - index_base) * otypesize) : 0], *cn = ci;
          do {
            for (count = 0; i < end_stride && ci == cn; ++count) {
              const size_t j = i + index_stride;
              *ai++ = &a0[NULL != stride_a ? ((LIBXSMM_VALUE1(const libxsmm_blasint, stride_a, i) - index_base) * itypesize) : 0];
              *bi++ = &b0[NULL != stride_b ? ((LIBXSMM_VALUE1(const libxsmm_blasint, stride_b, i) - index_base) * itypesize) : 0];
                 cn = &c0[NULL != stride_c ? ((LIBXSMM_VALUE1(const libxsmm_blasint, stride_c, j) - index_base) * otypesize) : 0];
              i = j;
            }
            ai = (const void**)libxsmm_mmbatch_array + begin; bi = ai + size;
            kernel.xbm(ai, bi, ci, &count);
            ci = cn;
          } while (i < end_stride);
        }
        else { /* array of pointers to matrices (singular strides are measured in Bytes) */
          const libxsmm_blasint da = (NULL != stride_a ? (*stride_a - index_base * sizeof(void*)) : 0);
          const libxsmm_blasint db = (NULL != stride_b ? (*stride_b - index_base * sizeof(void*)) : 0);
          const libxsmm_blasint dc = (NULL != stride_c ? (*stride_c - index_base * sizeof(void*)) : 0);
          const char *ia = a0 + (size_t)da * begin, *ib = b0 + (size_t)db * begin;
          char* ic = c0 + (size_t)dc * begin;
          if (
# if defined(LIBXSMM_BATCH_CHECK)
            NULL != *((const void**)ia) && NULL != *((const void**)ib) && NULL != *((const void**)ic) &&
# endif
            sizeof(void*) == da && sizeof(void*) == db) /* fast path */
          {
            if (0 != dc) {
              libxsmm_blasint i = begin;
              char* jc = ic;
              do {
                for (count = 0; i < end && *((const void**)ic) == *((const void**)jc); ++i) {
# if defined(LIBXSMM_BATCH_CHECK)
                  if (NULL != *((const void**)jc))
# endif
                  ++count;
                  jc += dc; /* next */
                }
                memcpy((void*)ai, ia, count * sizeof(void*));
                memcpy((void*)bi, ib, count * sizeof(void*));
                kernel.xbm(ai, bi, *((void**)ic), &count);
                ic = jc;
              } while (i < end);
            }
            else { /* fastest path */
              count = (unsigned long long)end - begin;
              memcpy((void*)ai, ia, count * sizeof(void*));
              memcpy((void*)bi, ib, count * sizeof(void*));
              kernel.xbm(ai, bi, *((void**)ic), &count);
            }
          }
          else { /* custom-copy required */
            libxsmm_blasint i = begin;
            char* jc = ic;
            do {
              for (count = 0; i < end && *((const void**)ic) == *((const void**)jc); ++i) {
# if defined(LIBXSMM_BATCH_CHECK)
                if (NULL != *((const void**)ia) && NULL != *((const void**)ib) && NULL != *((const void**)jc))
# endif
                {
                  *ai++ = *((const void**)ia); *bi++ = *((const void**)ib);
                  ++count;
                }
                ia += da; ib += db; jc += dc; /* next */
              }
              ai = (const void**)libxsmm_mmbatch_array + begin; bi = ai + size;
              kernel.xbm(ai, bi, *((void**)ic), &count);
              ic = jc;
            } while (i < end);
          }
        }
      }
      else { /* fallback */
        result = EXIT_FAILURE;
      }
    }
#endif /*defined(LIBXSMM_GEMM_BATCHREDUCE)*/
  }
  /* coverity[missing_unlock] */
  return result;
}


LIBXSMM_API void libxsmm_gemm_internal_set_batchflag(libxsmm_gemm_descriptor* descriptor, void* c, libxsmm_blasint index_stride,
  libxsmm_blasint batchsize, int multithreaded)
{
  LIBXSMM_ASSERT(NULL != descriptor);
  if (0 != (LIBXSMM_GEMM_FLAG_BETA_0 & descriptor->flags)) {
    const uintptr_t vw = (LIBXSMM_X86_AVX512 <= libxsmm_target_archid ? 64 : 32);
    /* assume that all C-matrices are aligned eventually */
    if (0 == LIBXSMM_MOD2((uintptr_t)c, vw)
#if 0 /* should fallback in BE */
      && LIBXSMM_X86_AVX <= libxsmm_target_archid
#endif
      && 0 != index_stride)
    {
      descriptor->flags |= LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT;
    }
  }
#if defined(LIBXSMM_GEMM_BATCHREDUCE)
  else if (0 != internal_gemm_batchreduce) { /* check if reduce-batch kernel can be used */
    static int error_once = 0;
    LIBXSMM_ASSERT(NULL != libxsmm_mmbatch_array);
# if (0 != LIBXSMM_SYNC)
    if (0 == multithreaded || 0 == internal_gemm_nlocks || 0 > batchsize)
# endif
    {
      int result = EXIT_FAILURE;
      switch (LIBXSMM_GETENUM_INP(descriptor->datatype)) {
        case LIBXSMM_DATATYPE_F64: {
          if (LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_OUT(descriptor->datatype)) {
            result = EXIT_SUCCESS;
          }
        } break;
        case LIBXSMM_DATATYPE_F32: {
          if (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT(descriptor->datatype)) {
            result = EXIT_SUCCESS;
          }
        } break;
      }
      if (EXIT_SUCCESS == result) {
        descriptor->flags |= LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS;
        descriptor->prefetch = 0; /* omit decision */
      }
      else {
        if ((LIBXSMM_VERBOSITY_WARN <= libxsmm_verbosity || 0 > libxsmm_verbosity) && /* library code is expected to be mute */
          1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM WARNING: data type not supported in batch-reduce!\n");
        }
      }
    }
# if (0 != LIBXSMM_SYNC)
    else if ((LIBXSMM_VERBOSITY_WARN <= libxsmm_verbosity || 0 > libxsmm_verbosity) && /* library code is expected to be mute */
      1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM: potential data races prevent batch-reduce.\n");
    }
# endif
  }
#endif /*defined(LIBXSMM_GEMM_BATCHREDUCE)*/
#if !defined(LIBXSMM_GEMM_BATCHREDUCE) || (0 == LIBXSMM_SYNC)
  LIBXSMM_UNUSED(batchsize); LIBXSMM_UNUSED(multithreaded);
#endif
}


LIBXSMM_API_INTERN void libxsmm_dmmbatch_blas(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const double* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb, const double* beta, void* c, const libxsmm_blasint* ldc,
  libxsmm_blasint index_base, libxsmm_blasint index_stride, const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize)
{
#if defined(LIBXSMM_BATCH_CHECK)
  if (NULL != a && NULL != b && NULL != c)
#endif
  {
    const libxsmm_blasint end = LIBXSMM_ABS(batchsize);
    libxsmm_blasint i;
    if (0 != index_stride) { /* stride arrays contain indexes */
      const libxsmm_blasint da = (NULL != stride_a ? (*stride_a - index_base) : 0);
      const libxsmm_blasint db = (NULL != stride_b ? (*stride_b - index_base) : 0);
      const libxsmm_blasint dc = (NULL != stride_c ? (*stride_c - index_base) : 0);
      const libxsmm_blasint end1 = end * index_stride;
      const double *const a0 = (const double*)a, *const b0 = (const double*)b, *ai = a0 + da, *bi = b0 + db;
      double *const c0 = (double*)c, *ci = c0 + dc;
      for (i = index_stride; i <= end1; i += index_stride) {
        const double *const an = &a0[NULL != stride_a ? (LIBXSMM_VALUE1(const libxsmm_blasint, stride_a, i) - index_base) : 0];
        const double *const bn = &b0[NULL != stride_b ? (LIBXSMM_VALUE1(const libxsmm_blasint, stride_b, i) - index_base) : 0];
        double       *const cn = &c0[NULL != stride_c ? (LIBXSMM_VALUE1(const libxsmm_blasint, stride_c, i) - index_base) : 0];
        libxsmm_blas_dgemm(transa, transb, &m, &n, &k, alpha, ai, lda, bi, ldb, beta, ci, ldc);
        ai = an; bi = bn; ci = cn; /* next */
      }
    }
    else { /* array of pointers to matrices (singular strides are measured in Bytes) */
      const libxsmm_blasint da = (NULL != stride_a ? (*stride_a - index_base * sizeof(void*)) : 0);
      const libxsmm_blasint db = (NULL != stride_b ? (*stride_b - index_base * sizeof(void*)) : 0);
      const libxsmm_blasint dc = (NULL != stride_c ? (*stride_c - index_base * sizeof(void*)) : 0);
      const char *const a0 = (const char*)a, *const b0 = (const char*)b, *ai = a0, *bi = b0;
      char *const c0 = (char*)c, *ci = c0;
      for (i = 0; i < end; ++i) {
        const char *const an = ai + da, *const bn = bi + db;
        char *const cn = ci + dc;
#if defined(LIBXSMM_BATCH_CHECK)
        if (NULL != *((const double**)ai) && NULL != *((const double**)bi) && NULL != *((const double**)ci))
#endif
        {
          libxsmm_blas_dgemm(transa, transb, &m, &n, &k, alpha, *((const double**)ai), lda, *((const double**)bi), ldb, beta, *((double**)ci), ldc);
        }
        ai = an; bi = bn; ci = cn; /* next */
      }
    }
  }
}


LIBXSMM_API_INTERN void libxsmm_smmbatch_blas(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const float* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb, const float* beta, void* c, const libxsmm_blasint* ldc,
  libxsmm_blasint index_base, libxsmm_blasint index_stride, const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize)
{
#if defined(LIBXSMM_BATCH_CHECK)
  if (NULL != a && NULL != b && NULL != c)
#endif
  {
    const libxsmm_blasint end = LIBXSMM_ABS(batchsize);
    libxsmm_blasint i;
    if (0 != index_stride) { /* stride arrays contain indexes */
      const libxsmm_blasint da = (NULL != stride_a ? (*stride_a - index_base) : 0);
      const libxsmm_blasint db = (NULL != stride_b ? (*stride_b - index_base) : 0);
      const libxsmm_blasint dc = (NULL != stride_c ? (*stride_c - index_base) : 0);
      const libxsmm_blasint end1 = end * index_stride;
      const float *a0 = (const float*)a, *b0 = (const float*)b, *ai = a0 + da, *bi = b0 + db;
      float *c0 = (float*)c, *ci = c0 + dc;
      for (i = index_stride; i <= end1; i += index_stride) {
        const float *const an = &a0[NULL != stride_a ? (LIBXSMM_VALUE1(const libxsmm_blasint, stride_a, i) - index_base) : 0];
        const float *const bn = &b0[NULL != stride_b ? (LIBXSMM_VALUE1(const libxsmm_blasint, stride_b, i) - index_base) : 0];
        float       *const cn = &c0[NULL != stride_c ? (LIBXSMM_VALUE1(const libxsmm_blasint, stride_c, i) - index_base) : 0];
        libxsmm_blas_sgemm(transa, transb, &m, &n, &k, alpha, ai, lda, bi, ldb, beta, ci, ldc);
        ai = an; bi = bn; ci = cn; /* next */
      }
    }
    else { /* array of pointers to matrices (singular strides are measured in Bytes) */
      const libxsmm_blasint da = (NULL != stride_a ? (*stride_a - index_base * sizeof(void*)) : 0);
      const libxsmm_blasint db = (NULL != stride_b ? (*stride_b - index_base * sizeof(void*)) : 0);
      const libxsmm_blasint dc = (NULL != stride_c ? (*stride_c - index_base * sizeof(void*)) : 0);
      const char *a0 = (const char*)a, *b0 = (const char*)b, *ai = a0, *bi = b0;
      char *c0 = (char*)c, *ci = c0;
      for (i = 0; i < end; ++i) {
        const char *const an = ai + da;
        const char *const bn = bi + db;
        char *const cn = ci + dc;
#if defined(LIBXSMM_BATCH_CHECK)
        if (NULL != *((const float**)ai) && NULL != *((const float**)bi) && NULL != *((const float**)ci))
#endif
        {
          libxsmm_blas_sgemm(transa, transb, &m, &n, &k, alpha, *((const float**)ai), lda, *((const float**)bi), ldb, beta, *((float**)ci), ldc);
        }
        ai = an; bi = bn; ci = cn; /* next */
      }
    }
  }
}


LIBXSMM_API int libxsmm_mmbatch_blas(
  libxsmm_datatype iprec, libxsmm_datatype oprec, const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb, const void* beta, void* c, const libxsmm_blasint* ldc,
  libxsmm_blasint index_base, libxsmm_blasint index_stride, const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize)
{
  int result;
  if (NULL != a && NULL != b && NULL != c) {
    switch (LIBXSMM_GETENUM(iprec, oprec)) {
      case LIBXSMM_DATATYPE_F64: {
        libxsmm_dmmbatch_blas(transa, transb, m, n, k,
          (const double*)alpha, a, lda, b, ldb, (const double*)beta, c, ldc,
          index_base, index_stride, stride_a, stride_b, stride_c, batchsize);
        result = EXIT_SUCCESS;
      } break;
      case LIBXSMM_DATATYPE_F32: {
        libxsmm_smmbatch_blas(transa, transb, m, n, k,
          (const float*)alpha, a, lda, b, ldb, (const float*)beta, c, ldc,
          index_base, index_stride, stride_a, stride_b, stride_c, batchsize);
        result = EXIT_SUCCESS;
      } break;
      default: result = EXIT_FAILURE;
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API void libxsmm_mmbatch(libxsmm_datatype iprec, libxsmm_datatype oprec,
  const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc, libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize, /*unsigned*/int tid, /*unsigned*/int ntasks)
{
  static int error_once = 0;
#if defined(LIBXSMM_BATCH_CHECK)
  if (NULL != a && NULL != b && NULL != c && 0 <= tid && tid < ntasks)
#endif
  {
    const unsigned char otypesize = libxsmm_typesize((libxsmm_datatype)oprec);
    int result = EXIT_FAILURE;
    LIBXSMM_INIT
    if (LIBXSMM_SMM_AI(m, n, k, 2/*RFO*/, otypesize)) { /* check if an SMM is suitable */
      const int gemm_flags = LIBXSMM_GEMM_PFLAGS(transa, transb, LIBXSMM_FLAGS);
      libxsmm_descriptor_blob blob;
      libxsmm_gemm_descriptor *const desc = libxsmm_gemm_descriptor_init2(&blob, iprec, oprec, m, n, k,
        NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
        NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
        NULL != ldc ? *ldc : m, alpha, beta, gemm_flags, libxsmm_get_gemm_prefetch(LIBXSMM_PREFETCH_AUTO));
      if (NULL != desc) {
        libxsmm_xmmfunction kernel;
        libxsmm_gemm_internal_set_batchflag(desc, c, index_stride, batchsize, 0/*multi-threaded*/);
        kernel = libxsmm_xmmdispatch(desc);
        if (NULL != kernel.xmm) {
          result = libxsmm_mmbatch_kernel(kernel, index_base, index_stride,
            stride_a, stride_b, stride_c, a, b, c, batchsize, tid, ntasks,
            libxsmm_typesize((libxsmm_datatype)iprec), otypesize, desc->flags);
        }
      }
    }
    if (EXIT_SUCCESS != result) { /* quiet fallback */
      if (EXIT_SUCCESS == libxsmm_mmbatch_blas(iprec, oprec,
        transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        index_base, index_stride, stride_a, stride_b, stride_c, batchsize))
      {
        if (LIBXSMM_VERBOSITY_WARN <= libxsmm_verbosity || 0 > libxsmm_verbosity) {
          const size_t threshold = LIBXSMM_MNK_SIZE(m, n, m);
          static size_t threshold_max = 0;
          if (threshold_max < threshold) {
            LIBXSMM_STDIO_ACQUIRE();
            fprintf(stderr, "LIBXSMM WARNING: ");
            libxsmm_gemm_print2(stderr, iprec, oprec, transa, transb, &m, &n, &k,
              alpha, NULL/*a*/, lda, NULL/*b*/, ldb, beta, NULL/*c*/, ldc);
            fprintf(stderr, " => batched GEMM was falling back!\n");
            LIBXSMM_STDIO_RELEASE();
            threshold_max = threshold;
          }
        }
      }
      else if (0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM ERROR: libxsmm_mmbatch failed!\n");
      }
    }
  }
#if defined(LIBXSMM_BATCH_CHECK)
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: incorrect arguments (libxsmm_mmbatch)!\n");
  }
#endif
}


LIBXSMM_API void libxsmm_gemm_batch(libxsmm_datatype iprec, libxsmm_datatype oprec,
  const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc, libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize)
{
  libxsmm_mmbatch(iprec, oprec, transa, transb, m, n, k,
    alpha,a, lda, b, ldb, beta, c, ldc, index_base, index_stride,
    stride_a, stride_b, stride_c, batchsize, 0/*tid*/, 1/*ntasks*/);
}


#if defined(LIBXSMM_BUILD) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_dgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*,
  const double*, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_dgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_sgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*,
  const float*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_sgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_blas_xgemm)(const libxsmm_datatype*, const libxsmm_datatype*,
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*,
  const float*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_blas_xgemm)(const libxsmm_datatype* iprec, const libxsmm_datatype* oprec,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_ASSERT(NULL != iprec && NULL != oprec);
  libxsmm_blas_xgemm(*iprec, *oprec, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_blas_dgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*,
  const double*, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_blas_dgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_blas_sgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*,
  const float*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_blas_sgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_mmbatch)(const libxsmm_datatype*, const libxsmm_datatype*,
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const void*, const void*, const libxsmm_blasint*, const void*, const libxsmm_blasint*,
  const void*, void*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const libxsmm_blasint[], const libxsmm_blasint[], const libxsmm_blasint[],
  const libxsmm_blasint*, const /*unsigned*/int*, const /*unsigned*/int*);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_mmbatch)(const libxsmm_datatype* iprec, const libxsmm_datatype* oprec,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc, const libxsmm_blasint* index_base, const libxsmm_blasint* index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const libxsmm_blasint* batchsize, const /*unsigned*/int* tid, const /*unsigned*/int* ntasks)
{
  LIBXSMM_ASSERT(NULL != iprec && NULL != oprec && NULL != m && NULL != n && NULL != k);
  LIBXSMM_ASSERT(NULL != index_base && NULL != index_stride && NULL != batchsize);
  LIBXSMM_ASSERT(NULL != tid && NULL != ntasks);
  libxsmm_mmbatch(*iprec, *oprec, transa, transb, *m, *n, *k, alpha, a, lda, b, ldb, beta, c, ldc,
    *index_base, *index_stride, stride_a, stride_b, stride_c, *batchsize, *tid, *ntasks);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_gemm_batch)(const libxsmm_datatype*, const libxsmm_datatype*,
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const void*, const void*, const libxsmm_blasint*, const void*, const libxsmm_blasint*,
  const void*, void*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const libxsmm_blasint[], const libxsmm_blasint[], const libxsmm_blasint[],
  const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_gemm_batch)(const libxsmm_datatype* iprec, const libxsmm_datatype* oprec,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc, const libxsmm_blasint* index_base, const libxsmm_blasint* index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const libxsmm_blasint* batchsize)
{
  LIBXSMM_ASSERT(NULL != iprec && NULL != oprec && NULL != m && NULL != n && NULL != k);
  LIBXSMM_ASSERT(NULL != index_base && NULL != index_stride && NULL != batchsize);
  libxsmm_gemm_batch(*iprec, *oprec, transa, transb, *m, *n, *k, alpha, a, lda, b, ldb, beta, c, ldc,
    *index_base, *index_stride, stride_a, stride_b, stride_c, *batchsize);
}

#endif /*defined(LIBXSMM_BUILD) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))*/

