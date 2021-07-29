/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
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

#if !defined(LIBXSMM_GEMM_XCOPY_JIT) && defined(LIBXSMM_XCOPY_JIT) && (0 != LIBXSMM_XCOPY_JIT)
# define LIBXSMM_GEMM_XCOPY_JIT
#endif
#if !defined(LIBXSMM_GEMM_KPARALLEL) && 0
# define LIBXSMM_GEMM_KPARALLEL
#endif
#if !defined(LIBXSMM_GEMM_BATCHSIZE)
# define LIBXSMM_GEMM_BATCHSIZE 1024
#endif
#if !defined(LIBXSMM_GEMM_TASKGRAIN)
# define LIBXSMM_GEMM_TASKGRAIN 128
#endif
#if !defined(LIBXSMM_GEMM_BATCHREDUCE) && !defined(_WIN32) && !defined(__CYGWIN__) /* not supported */
# define LIBXSMM_GEMM_BATCHREDUCE
#endif
#if !defined(LIBXSMM_GEMM_BATCHSCALE) && (defined(LIBXSMM_GEMM_BATCHREDUCE) || defined(LIBXSMM_WRAP))
#define LIBXSMM_GEMM_BATCHSCALE ((unsigned int)LIBXSMM_ROUND(sizeof(libxsmm_mmbatch_item) * (LIBXSMM_GEMM_MMBATCH_SCALE)))
#endif
#if defined(LIBXSMM_BUILD)
# define LIBXSMM_GEMM_WEAK LIBXSMM_API LIBXSMM_ATTRIBUTE_WEAK
#else
# define LIBXSMM_GEMM_WEAK LIBXSMM_API
#endif

#if (0 != LIBXSMM_SYNC) /** Locks for the batch interface (duplicated C indexes). */
# define LIBXSMM_GEMM_LOCKIDX(IDX, NPOT) LIBXSMM_MOD2(LIBXSMM_CRC32U(LIBXSMM_BLASINT_NBITS)(2507/*seed*/, &(IDX)), NPOT)
# define LIBXSMM_GEMM_LOCKPTR(PTR, NPOT) LIBXSMM_MOD2(LIBXSMM_CRC32U(LIBXSMM_BITS)(1975/*seed*/, &(PTR)), NPOT)
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
LIBXSMM_APIVAR_PUBLIC_DEF(int libxsmm_gemm_tasks);
LIBXSMM_APIVAR_PUBLIC_DEF(int libxsmm_gemm_wrap);

LIBXSMM_APIVAR_PRIVATE_DEF(libxsmm_gemm_prefetch_type libxsmm_gemm_auto_prefetch_default);
/** Determines the prefetch strategy, which is used in case of LIBXSMM_PREFETCH_AUTO. */
LIBXSMM_APIVAR_PRIVATE_DEF(libxsmm_gemm_prefetch_type libxsmm_gemm_auto_prefetch);

/** Prefetch strategy for tiled GEMM. */
LIBXSMM_APIVAR_DEFINE(libxsmm_gemm_prefetch_type internal_gemm_tiled_prefetch);
/** Vector width used for GEMM. */
LIBXSMM_APIVAR_DEFINE(unsigned int internal_gemm_vwidth);
/** Limit the M-extent of the tile. */
LIBXSMM_APIVAR_DEFINE(unsigned int internal_gemm_mlimit);
/** Table of M-extents per type-size (tile shape). */
LIBXSMM_APIVAR_DEFINE(float internal_gemm_nstretch);
/** Table of M-extents per type-size (tile shape). */
LIBXSMM_APIVAR_DEFINE(float internal_gemm_kstretch);
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
    libxsmm_blasint i, j = 0;
    LIBXSMM_ASSERT(NULL != transa_array && NULL != transb_array && NULL != group_count && NULL != group_size);
    LIBXSMM_ASSERT(NULL != m_array && NULL != n_array && NULL != k_array && NULL != lda_array && NULL != ldb_array && NULL != ldc_array);
    LIBXSMM_ASSERT(NULL != a_array && NULL != b_array && NULL != c_array && NULL != alpha_array && NULL != beta_array);
    for (i = 0; i < *group_count; ++i) {
      const libxsmm_blasint size = group_size[i];
      libxsmm_smmbatch_blas(transa_array + i, transb_array + i, m_array[i], n_array[i], k_array[i], alpha_array + i,
        a_array + i, lda_array + i, b_array + i, ldb_array + i, beta_array + i,
        c_array + i, ldc_array + i, 0/*index_base*/, 0/*index_stride*/, &ptrsize, &ptrsize, &ptrsize, size);
      j += size;
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


LIBXSMM_API_INTERN void libxsmm_gemm_init(int archid)
{
  const char* env_w = getenv("LIBXSMM_GEMM_WRAP");
  LIBXSMM_LOCK_ATTR_TYPE(LIBXSMM_GEMM_LOCK) attr;
  LIBXSMM_LOCK_ATTR_INIT(LIBXSMM_GEMM_LOCK, &attr);
#if defined(LIBXSMM_WRAP) /* determines if wrap is considered */
  { /* intercepted GEMMs (1: sequential and non-tiled, 2: parallelized and tiled) */
# if defined(__STATIC) /* with static library the user controls interceptor already */
    libxsmm_gemm_wrap = ((NULL == env_w || 0 == *env_w) /* LIBXSMM_WRAP=0: no promotion */
      ? (0 < (LIBXSMM_WRAP) ? (LIBXSMM_WRAP + 2) : (LIBXSMM_WRAP - 2)) : atoi(env_w));
# else
    libxsmm_gemm_wrap = ((NULL == env_w || 0 == *env_w) ? (LIBXSMM_WRAP) : atoi(env_w));
# endif
  }
#endif
  { /* setup prefetch strategy for tiled GEMMs */
    const char *const env_p = getenv("LIBXSMM_TGEMM_PREFETCH");
    const libxsmm_gemm_prefetch_type tiled_prefetch_default = LIBXSMM_GEMM_PREFETCH_AL2_AHEAD;
    const int uid = ((NULL == env_p || 0 == *env_p) ? LIBXSMM_PREFETCH_AUTO/*default*/ : atoi(env_p));
    internal_gemm_tiled_prefetch = (0 <= uid ? libxsmm_gemm_uid2prefetch(uid) : tiled_prefetch_default);
  }
#if (0 != LIBXSMM_SYNC)
  { /* initialize locks for the batch interface */
    const char *const env_locks = getenv("LIBXSMM_GEMM_NLOCKS");
    const int nlocks = ((NULL == env_locks || 0 == *env_locks) ? -1/*default*/ : atoi(env_locks));
    unsigned int i;
    internal_gemm_nlocks = LIBXSMM_UP2POT(0 > nlocks ? (LIBXSMM_GEMM_MAXNLOCKS) : LIBXSMM_MIN(nlocks, LIBXSMM_GEMM_MAXNLOCKS));
    for (i = 0; i < internal_gemm_nlocks; ++i) LIBXSMM_LOCK_INIT(LIBXSMM_GEMM_LOCK, &internal_gemm_lock[i].state, &attr);
  }
#endif
#if defined(LIBXSMM_GEMM_BATCHREDUCE) || defined(LIBXSMM_WRAP)
  { /* determines if batch-reduce kernel or batch-wrap is considered */
    const char *const env_r = getenv("LIBXSMM_GEMM_BATCHREDUCE");
    internal_gemm_batchreduce = (NULL == env_r || 0 == *env_r) ? 0 : atoi(env_r);
    if ((NULL == env_w || 0 == *env_w) && ((LIBXSMM_GEMM_MMBATCH_VERBOSITY <= libxsmm_verbosity && INT_MAX != libxsmm_verbosity) || 0 > libxsmm_verbosity)) {
      libxsmm_mmbatch_desc.flags = LIBXSMM_MMBATCH_FLAG_STATISTIC; /* enable auto-batch statistic */
      internal_gemm_batchreduce = 0;
    }
    if (0 != internal_gemm_batchreduce || 0 != libxsmm_gemm_wrap) {
      const char *const env_b = getenv("LIBXSMM_GEMM_BATCHSIZE");
      const int env_bi = (NULL == env_b || 0 == *env_b) ? -1/*auto*/ : atoi(env_b);
      const unsigned int env_bu = (unsigned int)(0 >= env_bi ? (LIBXSMM_GEMM_BATCHSIZE) : env_bi);
      const unsigned int batchscale = LIBXSMM_ABS(internal_gemm_batchreduce) * 2048/*arbitrary*/ * 2/*A and B-matrices*/ * sizeof(void*);
      const unsigned int minsize = LIBXSMM_UPDIV(batchscale * env_bu, LIBXSMM_GEMM_BATCHSCALE);
      const unsigned int batchsize = LIBXSMM_MAX(env_bu, minsize);
      const void *const extra = NULL;
      LIBXSMM_ASSERT(1 < (LIBXSMM_GEMM_MMBATCH_SCALE) && NULL == libxsmm_mmbatch_array);
      if (EXIT_SUCCESS == libxsmm_xmalloc(&libxsmm_mmbatch_array, (size_t)batchsize * (LIBXSMM_GEMM_BATCHSCALE), 0/*auto-alignment*/,
        LIBXSMM_MALLOC_FLAG_PRIVATE /*| LIBXSMM_MALLOC_FLAG_SCRATCH*/, &extra, sizeof(extra)))
      {
        LIBXSMM_LOCK_INIT(LIBXSMM_GEMM_LOCK, &libxsmm_mmbatch_lock, &attr);
        LIBXSMM_ASSERT(NULL != libxsmm_mmbatch_array);
        libxsmm_mmbatch_size = batchsize;
      }
    }
  }
#else
  LIBXSMM_UNUSED(env_w);
#endif
  { /* determines grain-size of tasks (when available) */
    const char *const env_s = getenv("LIBXSMM_GEMM_NPARGROUPS");
    libxsmm_gemm_npargroups = ((NULL == env_s || 0 == *env_s || 0 >= atoi(env_s))
      ? (LIBXSMM_GEMM_NPARGROUPS) : atoi(env_s));
  }
  if (LIBXSMM_X86_AVX512_CORE <= archid) {
    internal_gemm_vwidth = 64;
    internal_gemm_mlimit = 48;
    internal_gemm_nstretch = 3.0f;
    internal_gemm_kstretch = 2.0f;
  }
  else if (LIBXSMM_X86_AVX512_MIC <= archid) {
    internal_gemm_vwidth = 64;
    internal_gemm_mlimit = 64;
    internal_gemm_nstretch = 1.0f;
    internal_gemm_kstretch = 1.0f;
  }
  else if (LIBXSMM_X86_AVX2 <= archid) {
    internal_gemm_vwidth = 32;
    internal_gemm_mlimit = 48;
    internal_gemm_nstretch = 3.0f;
    internal_gemm_kstretch = 2.0f;
  }
  else if (LIBXSMM_X86_AVX <= archid) {
    internal_gemm_vwidth = 32;
    internal_gemm_mlimit = 48;
    internal_gemm_nstretch = 5.0f;
    internal_gemm_kstretch = 1.0f;
  }
  else {
    internal_gemm_vwidth = 16;
    internal_gemm_mlimit = 48;
    internal_gemm_nstretch = 7.0f;
    internal_gemm_kstretch = 5.0f;
  }
  { /* setup tile sizes according to environment (LIBXSMM_TGEMM_M, LIBXSMM_TGEMM_N, LIBXSMM_TGEMM_K) */
    const char *const env_m = getenv("LIBXSMM_TGEMM_M"), *const env_n = getenv("LIBXSMM_TGEMM_N"), *const env_k = getenv("LIBXSMM_TGEMM_K");
    const int m = ((NULL == env_m || 0 == *env_m) ? 0 : atoi(env_m));
    const int n = ((NULL == env_n || 0 == *env_n) ? 0 : atoi(env_n));
    const int k = ((NULL == env_k || 0 == *env_k) ? 0 : atoi(env_k));
    if (0 < m) {
      if (0 < n) internal_gemm_nstretch = ((float)n) / m;
      if (0 < k) internal_gemm_kstretch = ((float)k) / m;
    }
  }
  { /* setup tile sizes according to environment (LIBXSMM_TGEMM_NS, LIBXSMM_TGEMM_KS) */
    const char *const env_ns = getenv("LIBXSMM_TGEMM_NS"), *const env_ks = getenv("LIBXSMM_TGEMM_KS");
    const double ns = ((NULL == env_ns || 0 == *env_ns) ? 0 : atof(env_ns));
    const double ks = ((NULL == env_ks || 0 == *env_ks) ? 0 : atof(env_ks));
    if (0 < ns) internal_gemm_nstretch = (float)LIBXSMM_MIN(24, ns);
    if (0 < ks) internal_gemm_kstretch = (float)LIBXSMM_MIN(24, ks);
  }
  { /* determines if OpenMP tasks are used (when available) */
    const char *const env_t = getenv("LIBXSMM_GEMM_TASKS");
    const int gemm_tasks = ((NULL == env_t || 0 == *env_t) ? 0/*disabled*/ : atoi(env_t));
    libxsmm_gemm_tasks = (0 <= gemm_tasks ? LIBXSMM_ABS(gemm_tasks) : 1/*enabled*/);
  }
  { /* determines grain-size of tasks (when available) */
    const char *const env_g = getenv("LIBXSMM_GEMM_TASKGRAIN");
    const int gemm_taskgrain = ((NULL == env_g || 0 == *env_g || 0 >= atoi(env_g))
      ? (LIBXSMM_GEMM_TASKGRAIN) : atoi(env_g));
    /* adjust grain-size or scale beyond the number of threads */
    libxsmm_gemm_taskgrain = LIBXSMM_MAX(0 < libxsmm_gemm_tasks ? (gemm_taskgrain / libxsmm_gemm_tasks) : gemm_taskgrain, 1);
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
#if defined(LIBXSMM_GEMM_BATCHREDUCE) || defined(LIBXSMM_WRAP)
  if (NULL != libxsmm_mmbatch_array) {
    void *extra = NULL, *const mmbatch_array = libxsmm_mmbatch_array;
    if (EXIT_SUCCESS == libxsmm_get_malloc_xinfo(mmbatch_array, NULL/*size*/, NULL/*flags*/, &extra) && NULL != extra) {
      const libxsmm_mmbatch_flush_function flush = *(libxsmm_mmbatch_flush_function*)extra;
      if (NULL != flush) flush();
    }
#if !defined(NDEBUG)
    libxsmm_mmbatch_array = NULL;
#endif
    libxsmm_xfree(mmbatch_array, 0/*no check*/);
    LIBXSMM_LOCK_DESTROY(LIBXSMM_GEMM_LOCK, &libxsmm_mmbatch_lock);
  }
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
  switch (prefetch) {
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
  libxsmm_gemm_precision precision, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
  const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc)
{
  libxsmm_gemm_print2(ostream, precision, precision, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_API void libxsmm_gemm_print2(void* ostream,
  libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, const char* transa, const char* transb,
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
  char string_a[128], string_b[128], typeprefix = 0;

  switch (iprec | oprec) {
    case LIBXSMM_GEMM_PRECISION_F64: {
      LIBXSMM_ASSERT(iprec == oprec);
      LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "%g", NULL != alpha ? *((const double*)alpha) : LIBXSMM_ALPHA);
      LIBXSMM_SNPRINTF(string_b, sizeof(string_b), "%g", NULL != beta  ? *((const double*)beta)  : LIBXSMM_BETA);
      mhd_elemtype = LIBXSMM_MHD_ELEMTYPE_F64;
      typeprefix = 'd';
    } break;
    case LIBXSMM_GEMM_PRECISION_F32: {
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
      char extension_header[256];
      size_t data_size[2], size[2];

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
  void* ostream, libxsmm_gemm_precision precision, char transa, char transb,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, double dalpha, const void* a, libxsmm_blasint lda,
  const void* b, libxsmm_blasint ldb, double dbeta, void* c, libxsmm_blasint ldc)
{
  libxsmm_gemm_dprint2(ostream, precision, precision, transa, transb, m, n, k, dalpha, a, lda, b, ldb, dbeta, c, ldc);
}


LIBXSMM_API void libxsmm_gemm_dprint2(
  void* ostream, libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, char transa, char transb,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, double dalpha, const void* a, libxsmm_blasint lda,
  const void* b, libxsmm_blasint ldb, double dbeta, void* c, libxsmm_blasint ldc)
{
  switch (iprec) {
    case LIBXSMM_GEMM_PRECISION_F64: {
      libxsmm_gemm_print2(ostream, LIBXSMM_GEMM_PRECISION_F64, oprec, &transa, &transb,
        &m, &n, &k, &dalpha, a, &lda, b, &ldb, &dbeta, c, &ldc);
    } break;
    case LIBXSMM_GEMM_PRECISION_F32: {
      const float alpha = (float)dalpha, beta = (float)dbeta;
      libxsmm_gemm_print2(ostream, LIBXSMM_GEMM_PRECISION_F32, oprec, &transa, &transb,
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
  libxsmm_code_pointer code;
  size_t code_size;
  code.xgemm = kernel;
  if (NULL != libxsmm_get_kernel_xinfo(code, &desc, &code_size) &&
      NULL != desc && LIBXSMM_KERNEL_KIND_MATMUL == LIBXSMM_DESCRIPTOR_KIND(desc->kind))
  {
    libxsmm_gemm_dprint2(ostream,
      (libxsmm_gemm_precision)LIBXSMM_GETENUM_INP(desc->gemm.desc.datatype),
      (libxsmm_gemm_precision)LIBXSMM_GETENUM_OUT(desc->gemm.desc.datatype),
      (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & desc->gemm.desc.flags) ? 'N' : 'T'),
      (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_B & desc->gemm.desc.flags) ? 'N' : 'T'),
      (libxsmm_blasint)desc->gemm.desc.m, (libxsmm_blasint)desc->gemm.desc.n, (libxsmm_blasint)desc->gemm.desc.k,
      1, a, (libxsmm_blasint)desc->gemm.desc.lda, b, (libxsmm_blasint)desc->gemm.desc.ldb,
      0 != (LIBXSMM_GEMM_FLAG_BETA_0 & libxsmm_mmbatch_desc.flags) ? 0 : 1, c, (libxsmm_blasint)desc->gemm.desc.ldc);
    fprintf((FILE*)ostream, " = %p+%u", code.ptr_const, (unsigned int)code_size);
  }
}


LIBXSMM_API void libxsmm_blas_xgemm(libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_INIT
  switch (iprec) {
    case LIBXSMM_GEMM_PRECISION_F64: {
      LIBXSMM_ASSERT(iprec == oprec);
      LIBXSMM_BLAS_XGEMM(double, double, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } break;
    case LIBXSMM_GEMM_PRECISION_F32: {
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


LIBXSMM_API_INLINE int libxsmm_gemm_plan_internal(unsigned int ntasks,
  unsigned int m, unsigned int n, unsigned int k,           /* whole problem size */
  unsigned int tm, unsigned int tn, unsigned int tk,        /* tile size (kernel) */
  unsigned int* nmt, unsigned int* nnt, unsigned int* nkt,  /* number of tiles */
  unsigned int* mt, unsigned int* nt, unsigned int* kt)     /* number of tasks */
{
  unsigned int result = EXIT_SUCCESS, replan = 0;
  LIBXSMM_ASSERT(NULL != nmt && NULL != nnt && NULL != nkt);
  LIBXSMM_ASSERT(NULL != mt && NULL != nt && NULL != kt);
  LIBXSMM_ASSERT(0 < ntasks);
  *nmt = (m + tm - 1) / LIBXSMM_MAX(tm, 1);
  *nnt = (n + tn - 1) / LIBXSMM_MAX(tn, 1);
  *nkt = (k + tk - 1) / LIBXSMM_MAX(tk, 1);
#if !defined(NDEBUG)
  *mt = *nt = *kt = 0;
#endif
  do {
    if (1 >= replan) *mt = libxsmm_product_limit(*nmt, ntasks, 0);
    if (1 == replan || ntasks <= *mt) { /* M-parallelism */
      *nt = 1;
      *kt = 1;
      replan = 0;
    }
    else {
      const unsigned int mntasks = libxsmm_product_limit((*nmt) * (*nnt), ntasks, 0);
      if (0 == replan && *mt >= mntasks) replan = 1;
      if (2 == replan || (0 == replan && ntasks <= mntasks)) { /* MN-parallelism */
        *nt = mntasks / *mt;
        *kt = 1;
        replan = 0;
      }
      else { /* MNK-parallelism */
        const unsigned int mnktasks = libxsmm_product_limit((*nmt) * (*nnt) * (*nkt), ntasks, 0);
        if (mntasks < mnktasks) {
#if defined(LIBXSMM_GEMM_KPARALLEL)
          *nt = mntasks / *mt;
          *kt = mnktasks / mntasks;
          replan = 0;
#else
          static int error_once = 0;
          if ((LIBXSMM_VERBOSITY_HIGH <= libxsmm_verbosity || 0 > libxsmm_verbosity) /* library code is expected to be mute */
            && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
          {
            fprintf(stderr, "LIBXSMM WARNING: XGEMM K-parallelism triggered!\n");
          }
#endif
        }
#if defined(LIBXSMM_GEMM_KPARALLEL)
        else
#endif
        if (0 == replan) replan = 2;
      }
    }
  } while (0 != replan);
  if (0 == *mt || 0 == *nt || 0 == *kt) {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API libxsmm_gemm_handle* libxsmm_gemm_handle_init(libxsmm_gemm_blob* blob,
  libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const void* alpha, const void* beta, int flags, /*unsigned*/int ntasks)
{
  unsigned int ulda, uldb, um, un, uk, tm = 0, tn = 0, tk = 0, max_ntasks = 0;
  libxsmm_descriptor_blob desc_blob;
  union {
    libxsmm_gemm_handle* ptr;
    libxsmm_gemm_blob* blob;
  } result;
  LIBXSMM_ASSERT(sizeof(libxsmm_gemm_handle) <= sizeof(libxsmm_gemm_blob));
  if (NULL != blob && NULL != m && 0 < ntasks) {
    unsigned int ntm = 0, ntn = 0, ntk = 0, mt = 1, nt = 1, kt = 1;
    const char *const env_tm = getenv("LIBXSMM_TGEMM_M");
    libxsmm_blasint klda, kldb, kldc, km, kn;
    libxsmm_gemm_descriptor* desc;
    double dbeta;
    LIBXSMM_INIT
    result.blob = blob;
#if defined(NDEBUG)
    result.ptr->copy_a.ptr = result.ptr->copy_b.ptr = result.ptr->copy_i.ptr = result.ptr->copy_o.ptr = NULL;
#else
    memset(blob, 0, sizeof(libxsmm_gemm_blob));
#endif
    if (EXIT_SUCCESS != libxsmm_dvalue((libxsmm_datatype)oprec, beta, &dbeta)) dbeta = LIBXSMM_BETA; /* fuse beta into flags */
    result.ptr->gemm_flags = LIBXSMM_GEMM_PFLAGS(transa, transb, LIBXSMM_FLAGS) | (LIBXSMM_NEQ(0, dbeta) ? 0 : LIBXSMM_GEMM_FLAG_BETA_0);
    /* TODO: check that arguments fit into handle (unsigned int vs. libxsmm_blasint) */
    um = (unsigned int)(*m); uk = (NULL != k ? ((unsigned int)(*k)) : um); un = (NULL != n ? ((unsigned int)(*n)) : uk);
    result.ptr->otypesize = libxsmm_typesize((libxsmm_datatype)oprec);
    if (NULL == env_tm || 0 >= atoi(env_tm)) {
      const unsigned int vwidth = LIBXSMM_MAX(internal_gemm_vwidth / result.ptr->otypesize, 1);
      const double s2 = (double)internal_gemm_nstretch * internal_gemm_kstretch; /* LIBXSMM_INIT! */
      unsigned int tmi = libxsmm_product_limit(um, internal_gemm_mlimit, 0); /* LIBXSMM_INIT! */
      for (; vwidth <= tmi; tmi = libxsmm_product_limit(um, tmi - 1, 0)) {
        const double si = (double)(LIBXSMM_CONFIG_MAX_MNK) / ((double)tmi * tmi * tmi), s = (s2 <= si ? 1 : (s2 / si));
        unsigned int tni = libxsmm_product_limit(un, LIBXSMM_MAX((unsigned int)(tmi * (s * internal_gemm_nstretch)), 1), 0);
        unsigned int tki = libxsmm_product_limit(uk, LIBXSMM_MAX((unsigned int)(tmi * (s * internal_gemm_kstretch)), 1), 0);
        unsigned int ntmi, ntni, ntki, mti = 1, nti = 1, kti = 1;
        LIBXSMM_ASSERT(tmi <= um && tni <= un && tki <= uk);
        if (LIBXSMM_GEMM_FLAG_TRANS_AB == (LIBXSMM_GEMM_FLAG_TRANS_AB & result.ptr->gemm_flags)) {
          const unsigned int ttm = (unsigned int)libxsmm_product_limit(tmi, (unsigned int)ntasks, 0);
          const unsigned int ttn = (unsigned int)libxsmm_product_limit(tni, (unsigned int)ntasks, 0);
          tmi = tni = LIBXSMM_MIN(ttm, ttn); /* prefer threads over larger tile */
        }
        if (EXIT_SUCCESS == libxsmm_gemm_plan_internal((unsigned int)ntasks, um, un, uk, tmi, tni, tki,
          &ntmi, &ntni, &ntki, &mti, &nti, &kti))
        {
          const int exit_plan = ((tmi < um && tni < un && tki < uk && (tm != tmi || tn != tni || tk != tki)) ? 0 : 1);
          const unsigned itasks = mti * nti * kti;
          LIBXSMM_ASSERT(1 <= itasks);
          if (max_ntasks < itasks) {
            ntm = ntmi; ntn = ntni; ntk = ntki;
            mt = mti; nt = nti; kt = kti;
            tm = tmi; tn = tni; tk = tki;
            max_ntasks = itasks;
          }
          if (itasks == (unsigned int)ntasks || 0 != exit_plan) break;
        }
      }
    }
    else {
      const unsigned int tmi = atoi(env_tm);
      const double s2 = (double)internal_gemm_nstretch * internal_gemm_kstretch; /* LIBXSMM_INIT! */
      double si, s;
      tm = libxsmm_product_limit(um, LIBXSMM_MIN(tmi, internal_gemm_mlimit), 0); /* LIBXSMM_INIT! */
      si = (double)(LIBXSMM_CONFIG_MAX_MNK) / ((double)tm * tm * tm); s = (s2 <= si ? 1 : (s2 / si));
      tn = libxsmm_product_limit(un, LIBXSMM_MAX((unsigned int)(tm * (s * internal_gemm_nstretch)), 1), 0);
      tk = libxsmm_product_limit(uk, LIBXSMM_MAX((unsigned int)(tm * (s * internal_gemm_kstretch)), 1), 0);
      if (LIBXSMM_GEMM_FLAG_TRANS_AB == (LIBXSMM_GEMM_FLAG_TRANS_AB & result.ptr->gemm_flags)) {
        const unsigned int ttm = (unsigned int)libxsmm_product_limit(tm, (unsigned int)ntasks, 0);
        const unsigned int ttn = (unsigned int)libxsmm_product_limit(tn, (unsigned int)ntasks, 0);
        tm = tn = LIBXSMM_MIN(ttm, ttn); /* prefer threads over larger tile */
      }
      if (EXIT_SUCCESS == libxsmm_gemm_plan_internal((unsigned int)ntasks, um, un, uk, tm, tn, tk,
        &ntm, &ntn, &ntk, &mt, &nt, &kt))
      {
#if defined(NDEBUG)
        max_ntasks = 2; /* only need something unequal to zero to pass below condition */
#else
        max_ntasks = mt * nt * kt;
#endif
      }
    }
    LIBXSMM_ASSERT(LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & result.ptr->gemm_flags) || tm == tn);
    /* check for non-conforming GEMM parameters (error), and conforming GEMM parameters (fast-path, fallback) */
    if (0 == max_ntasks || 0 == tm || 0 == tn || 0 == tk || 0 != (um % tm) || 0 != (un % tn) || 0 != (uk % tk)) {
      return NULL;
    }
    result.ptr->flags = flags;
    if (LIBXSMM_GEMM_HANDLE_FLAG_AUTO == flags && 0 == LIBXSMM_SMM_AI(um, un, uk,
      0 == (result.ptr->gemm_flags & LIBXSMM_GEMM_FLAG_BETA_0) ? 1 : 2/*RFO*/, result.ptr->otypesize))
    {
      if (um == LIBXSMM_UP2POT(um) || un == LIBXSMM_UP2POT(un)) { /* power-of-two (POT) extent(s) */
        result.ptr->flags |= LIBXSMM_GEMM_HANDLE_FLAG_COPY_C;
        if (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & result.ptr->gemm_flags)) {
          result.ptr->flags |= LIBXSMM_GEMM_HANDLE_FLAG_COPY_A;
        }
      }
    }
    result.ptr->itypesize = libxsmm_typesize((libxsmm_datatype)iprec);
    result.ptr->ldc = (unsigned int)(NULL != ldc ? *ldc : *m);
    ulda = (NULL != lda ? ((unsigned int)(*lda)) : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & result.ptr->gemm_flags) ? ((unsigned int)(*m)) : uk));
    uldb = (NULL != ldb ? ((unsigned int)(*ldb)) : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & result.ptr->gemm_flags) ? uk : un));
    if (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & result.ptr->gemm_flags)) { /* NN, NT, or TN */
      const libxsmm_blasint itm = (libxsmm_blasint)tm, itk = (libxsmm_blasint)tk;
#if defined(LIBXSMM_GEMM_XCOPY_JIT)
      const libxsmm_blasint itn = (libxsmm_blasint)tn;
#endif
      kldc = (libxsmm_blasint)result.ptr->ldc;
      klda = (libxsmm_blasint)ulda;
      kldb = (libxsmm_blasint)uldb;
      if (0 != (LIBXSMM_GEMM_FLAG_TRANS_A & result.ptr->gemm_flags)) { /* TN */
#if defined(LIBXSMM_GEMM_XCOPY_JIT)
        result.ptr->copy_a.function = libxsmm_dispatch_meltw_unary(itk, itm, &klda, &itm,
          (libxsmm_datatype)iprec, (libxsmm_datatype)iprec, (libxsmm_datatype)iprec,
          LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
#endif
        klda = itm;
      }
      else if (0 != (LIBXSMM_GEMM_HANDLE_FLAG_COPY_A & result.ptr->flags)) {
#if defined(LIBXSMM_GEMM_XCOPY_JIT)
        result.ptr->copy_a.function = libxsmm_dispatch_meltw_unary(itm, itk, &klda, &itm,
          (libxsmm_datatype)iprec, (libxsmm_datatype)iprec, (libxsmm_datatype)iprec,
          LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
#endif
        klda = (libxsmm_blasint)tm;
      }
      if (0 != (LIBXSMM_GEMM_FLAG_TRANS_B & result.ptr->gemm_flags)) { /* NT */
#if defined(LIBXSMM_GEMM_XCOPY_JIT)
        result.ptr->copy_b.function = libxsmm_dispatch_meltw_unary(itn, itk, &kldb, &itk,
          (libxsmm_datatype)iprec, (libxsmm_datatype)iprec, (libxsmm_datatype)iprec,
          LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
#endif
        kldb = itk;
      }
      else if (0 != (LIBXSMM_GEMM_HANDLE_FLAG_COPY_B & result.ptr->flags)) {
#if defined(LIBXSMM_GEMM_XCOPY_JIT)
        result.ptr->copy_b.function = libxsmm_dispatch_meltw_unary(itk, itn, &kldb, &itk,
          (libxsmm_datatype)iprec, (libxsmm_datatype)iprec, (libxsmm_datatype)iprec,
          LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
#endif
        kldb = (libxsmm_blasint)tk;
      }
      if (0 != (LIBXSMM_GEMM_HANDLE_FLAG_COPY_C & result.ptr->flags)) {
#if defined(LIBXSMM_GEMM_XCOPY_JIT)
        result.ptr->copy_o.function = libxsmm_dispatch_meltw_unary(itm, itn, &itm, &kldc,
          (libxsmm_datatype)oprec, (libxsmm_datatype)oprec, (libxsmm_datatype)oprec,
          LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
        if (0 == (result.ptr->gemm_flags & LIBXSMM_GEMM_FLAG_BETA_0)) { /* copy-in only if beta!=0 */
          result.ptr->copy_i.function = libxsmm_dispatch_meltw_unary(itm, itn, &kldc, &itm,
            (libxsmm_datatype)oprec, (libxsmm_datatype)oprec, (libxsmm_datatype)oprec,
            LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
        }
#endif
        kldc = (libxsmm_blasint)tm;
      }
      result.ptr->lda = ulda; result.ptr->ldb = uldb;
      result.ptr->km = tm; result.ptr->kn = tn;
      result.ptr->mt = mt; result.ptr->nt = nt;
      result.ptr->m = um; result.ptr->n = un;
      result.ptr->dm = LIBXSMM_UPDIV(ntm, mt) * tm;
      result.ptr->dn = LIBXSMM_UPDIV(ntn, nt) * tn;
      km = tm; kn = tn;
    }
    else { /* TT */
      const unsigned int tt = tm;
      const libxsmm_blasint itt = (libxsmm_blasint)tt;
#if defined(LIBXSMM_GEMM_XCOPY_JIT)
      const libxsmm_blasint ildc = (libxsmm_blasint)result.ptr->ldc;
      result.ptr->copy_o.function = libxsmm_dispatch_meltw_unary(itt, itt, &itt, &ildc,
        (libxsmm_datatype)oprec, (libxsmm_datatype)oprec, (libxsmm_datatype)oprec,
        LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
      if (0 == (result.ptr->gemm_flags & LIBXSMM_GEMM_FLAG_BETA_0)) { /* copy-in only if beta!=0 */
        result.ptr->copy_i.function = libxsmm_dispatch_meltw_unary(itt, itt, &ildc, &itt,
          (libxsmm_datatype)oprec, (libxsmm_datatype)oprec, (libxsmm_datatype)oprec,
          LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
      }
#endif
      klda = (libxsmm_blasint)uldb;
      kldb = (libxsmm_blasint)ulda;
      kldc = itt;
      LIBXSMM_ASSERT(tt == tn);
      if (0 != (LIBXSMM_GEMM_HANDLE_FLAG_COPY_A & result.ptr->flags)) {
#if defined(LIBXSMM_GEMM_XCOPY_JIT)
        const libxsmm_blasint itk = (libxsmm_blasint)tk;
        result.ptr->copy_a.function = libxsmm_dispatch_meltw_unary(itt, itk, &kldb, &itk,
          (libxsmm_datatype)iprec, (libxsmm_datatype)iprec, (libxsmm_datatype)iprec,
          LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
#endif
        klda = itt;
      }
      if (0 != (LIBXSMM_GEMM_HANDLE_FLAG_COPY_B & result.ptr->flags)) {
#if defined(LIBXSMM_GEMM_XCOPY_JIT)
        const libxsmm_blasint itn = (libxsmm_blasint)tn, itk = (libxsmm_blasint)tk;
        result.ptr->copy_b.function = libxsmm_dispatch_meltw_unary(itk, itn, &klda, &itk,
          (libxsmm_datatype)iprec, (libxsmm_datatype)iprec, (libxsmm_datatype)iprec,
          LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
#endif
        kldb = (libxsmm_blasint)tk;
      }
      result.ptr->lda = uldb; result.ptr->ldb = ulda;
      result.ptr->km = tn; result.ptr->kn = tm;
      result.ptr->mt = nt; result.ptr->nt = mt;
      result.ptr->m = un; result.ptr->n = um;
      result.ptr->dm = LIBXSMM_UPDIV(ntn, nt) * tn;
      result.ptr->dn = LIBXSMM_UPDIV(ntm, mt) * tm;
      km = kn = tt;
    }
    result.ptr->dk = ntk / kt * tk;
    result.ptr->kk = tk;
    result.ptr->kt = kt;
    result.ptr->k = uk;
    desc = libxsmm_gemm_descriptor_init2( /* remove transpose flags from kernel request */
      &desc_blob, iprec, oprec, km, kn, result.ptr->kk, klda, kldb, kldc,
      alpha, beta, result.ptr->gemm_flags & ~LIBXSMM_GEMM_FLAG_TRANS_AB, internal_gemm_tiled_prefetch);
    result.ptr->kernel[0] = libxsmm_xmmdispatch(desc);
    if (NULL != result.ptr->kernel[0].xmm) {
      if (0 == (desc->flags & LIBXSMM_GEMM_FLAG_BETA_0)) { /* beta!=0 */
        result.ptr->kernel[1] = result.ptr->kernel[0];
      }
      else { /* generate kernel with beta=1 */
        desc->flags &= ~LIBXSMM_GEMM_FLAG_BETA_0;
        result.ptr->kernel[1] = libxsmm_xmmdispatch(desc);
        if (NULL == result.ptr->kernel[1].xmm) result.ptr = NULL;
      }
    }
    else result.ptr = NULL;
  }
  else {
    result.ptr = NULL;
  }
  return result.ptr;
}


LIBXSMM_API_INLINE size_t libxsmm_gemm_handle_get_scratch_size_a(const libxsmm_gemm_handle* handle)
{
  size_t result;
  if (NULL == handle || (0 == (handle->flags & LIBXSMM_GEMM_HANDLE_FLAG_COPY_A)
    && (LIBXSMM_GEMM_FLAG_TRANS_AB == (LIBXSMM_GEMM_FLAG_TRANS_AB & handle->gemm_flags) ||
       (LIBXSMM_GEMM_FLAG_TRANS_A & handle->gemm_flags) == 0)))
  {
    result = 0;
  }
  else {
    const size_t size = (size_t)handle->km * handle->kk * handle->itypesize;
    result = LIBXSMM_UP2(size, LIBXSMM_CACHELINE);
  }
  return result;
}


LIBXSMM_API_INLINE size_t libxsmm_gemm_handle_get_scratch_size_b(const libxsmm_gemm_handle* handle)
{
  size_t result;
  if (NULL == handle || (0 == (handle->flags & LIBXSMM_GEMM_HANDLE_FLAG_COPY_B)
    && (LIBXSMM_GEMM_FLAG_TRANS_AB == (LIBXSMM_GEMM_FLAG_TRANS_AB & handle->gemm_flags) ||
       (LIBXSMM_GEMM_FLAG_TRANS_B & handle->gemm_flags) == 0)))
  {
    result = 0;
  }
  else {
    const size_t size = (size_t)handle->kk * handle->kn * handle->itypesize;
    result = LIBXSMM_UP2(size, LIBXSMM_CACHELINE);
  }
  return result;
}


LIBXSMM_API_INLINE size_t libxsmm_gemm_handle_get_scratch_size_c(const libxsmm_gemm_handle* handle)
{
  size_t result;
  if (NULL == handle || (0 == (handle->flags & LIBXSMM_GEMM_HANDLE_FLAG_COPY_C)
    && LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & handle->gemm_flags)))
  {
    result = 0;
  }
  else {
    const size_t size = (size_t)handle->km * handle->kn * handle->otypesize;
    result = LIBXSMM_UP2(size, LIBXSMM_CACHELINE);
  }
  return result;
}


LIBXSMM_API size_t libxsmm_gemm_handle_get_scratch_size(const libxsmm_gemm_handle* handle)
{
  size_t result;
  if (NULL != handle) { /* thread-local scratch buffer for GEMM */
    const size_t size_a = libxsmm_gemm_handle_get_scratch_size_a(handle);
    const size_t size_b = libxsmm_gemm_handle_get_scratch_size_b(handle);
    const size_t size_c = libxsmm_gemm_handle_get_scratch_size_c(handle);
    result = (size_a + size_b + size_c) * handle->mt * handle->nt * handle->kt;
  }
  else {
    result = 0;
  }
  return result;
}


LIBXSMM_API void libxsmm_gemm_task(const libxsmm_gemm_handle* handle, void* scratch,
  const void* a, const void* b, void* c, /*unsigned*/int tid, /*unsigned*/int ntasks)
{
#if !defined(NDEBUG)
  if (NULL != handle && 0 <= tid && tid < ntasks)
#endif
  {
    const unsigned int utasks = (unsigned int)ntasks;
    const unsigned int wksize = handle->mt * handle->nt * handle->kt;
    const unsigned int spread = (wksize <= utasks ? (utasks / wksize) : 1);
    const unsigned int utid = (unsigned int)tid, vtid = utid / spread;
    if (utid < (spread * wksize) && 0 == (utid - vtid * spread)) {
      const int excess = (utasks << 1) <= (vtid + wksize);
      const unsigned int rtid = vtid / handle->mt, mtid = vtid - rtid * handle->mt, ntid = rtid % handle->nt, ktid = vtid / (handle->mt * handle->nt);
      const unsigned int m0 = mtid * handle->dm, m1 = (0 == excess ? LIBXSMM_MIN(m0 + handle->dm, handle->m) : handle->m);
      const unsigned int n0 = ntid * handle->dn, n1 = (0 == excess ? LIBXSMM_MIN(n0 + handle->dn, handle->n) : handle->n);
      const unsigned int k0 = ktid * handle->dk, k1 = (0 == excess ? LIBXSMM_MIN(k0 + handle->dk, handle->k) : handle->k);
      const unsigned int ldo = (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & handle->gemm_flags) ? handle->km : handle->kk);
      /* calculate increments to simplify address calculations */
      const unsigned int dom = handle->km * handle->otypesize;
      const unsigned int don = handle->kn * handle->otypesize;
      const unsigned int dik = handle->kk * handle->itypesize;
      const unsigned int on = handle->otypesize * n0;
      /* calculate base address of thread-local storage */
      const size_t size_a = libxsmm_gemm_handle_get_scratch_size_a(handle);
      const size_t size_b = libxsmm_gemm_handle_get_scratch_size_b(handle);
      const size_t size_c = libxsmm_gemm_handle_get_scratch_size_c(handle);
      char *const at = (char*)scratch + (size_a + size_b + size_c) * vtid;
      char *const bt = at + size_a, *const ct = bt + size_b;
      const libxsmm_xcopykernel kernel = { NULL };
      /* loop induction variables and other variables */
      unsigned int om = handle->otypesize * m0, im = m0, in = n0, ik = k0, im1, in1, ik1;
      LIBXSMM_ASSERT_MSG(mtid < handle->mt && ntid < handle->nt && ktid < handle->kt, "Invalid task-ID");
      LIBXSMM_ASSERT_MSG(m1 <= handle->m && n1 <= handle->n && k1 <= handle->k, "Invalid task size");
      for (im1 = im + handle->km; (im1 - 1) < m1; im = im1, im1 += handle->km, om += dom) {
        unsigned int dn = don, dka = dik, dkb = dik;
        char *c0 = (char*)c, *ci;
        const char *aa;
        if (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & handle->gemm_flags)) {
          if (0 != (LIBXSMM_GEMM_FLAG_TRANS_A & handle->gemm_flags)) { /* TN */
            aa = (const char*)a + ((size_t)im * handle->lda + k0) * handle->itypesize;
          }
          else if (0 != (LIBXSMM_GEMM_FLAG_TRANS_B & handle->gemm_flags)) { /* NT */
            aa = (const char*)a + ((size_t)k0 * handle->lda + im) * handle->itypesize;
            dka *= handle->lda; dkb *= handle->ldb;
          }
          else { /* NN */
            aa = (const char*)a + ((size_t)k0 * handle->lda + im) * handle->itypesize;
            dka *= handle->lda;
          }
          c0 += (size_t)on * handle->ldc + om;
          dn *= handle->ldc;
        }
        else { /* TT */
          aa = (const char*)b + ((size_t)k0 * handle->lda + im) * handle->itypesize;
          c0 += (size_t)on + handle->ldc * (size_t)om;
          dka *= handle->lda;
        }
        for (in = n0, in1 = in + handle->kn; (in1 - 1) < n1; in = in1, in1 += handle->kn, c0 += dn) {
          const char *a0 = aa, *b0 = (const char*)b;
          if (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & handle->gemm_flags)) {
            if (0 != (LIBXSMM_GEMM_FLAG_TRANS_B & handle->gemm_flags)) { /* NT */
              b0 += ((size_t)k0 * handle->ldb + in) * handle->itypesize;
            }
            else { /* NN or TN */
              b0 += ((size_t)in * handle->ldb + k0) * handle->itypesize;
            }
          }
          else { /* TT */
            b0 = (const char*)a + ((size_t)in * handle->ldb + k0) * handle->itypesize;
          }
#if defined(LIBXSMM_GEMM_XCOPY_JIT)
          if (NULL == handle->copy_i.ptr)
#endif
          {
            ci = (NULL == handle->copy_o.ptr ? c0 : ct);
            if (LIBXSMM_GEMM_FLAG_TRANS_AB == (LIBXSMM_GEMM_FLAG_TRANS_AB & handle->gemm_flags)) {
              const unsigned int km = handle->kn, kn = handle->km;
              libxsmm_otrans_internal(ct/*out*/, c0/*in*/, handle->otypesize, handle->ldc/*ldi*/, kn/*ldo*/,
                0, km, 0, kn, km/*tile*/, kn/*tile*/, kernel);
              ci = ct;
            }
            else if (0 != (LIBXSMM_GEMM_HANDLE_FLAG_COPY_C & handle->flags)) {
              if (0 == (handle->gemm_flags & LIBXSMM_GEMM_FLAG_BETA_0)) { /* copy-in only if beta!=0 */
                libxsmm_matcopy_internal(ct/*out*/, c0/*in*/, handle->otypesize, handle->ldc/*ldi*/, handle->km/*ldo*/,
                  0, handle->km, 0, handle->kn, handle->km/*tile*/, handle->kn/*tile*/, kernel);
              }
              ci = ct;
            }
          }
#if defined(LIBXSMM_GEMM_XCOPY_JIT)
          else { /* MCOPY/TCOPY kernel */
            LIBXSMM_MCOPY_CALL(handle->copy_i, handle->otypesize, c0, &handle->ldc, ct, &handle->km);
            ci = ct;
          }
#endif
          for (ik = k0, ik1 = ik + handle->kk; (ik1 - 1) < k1; ik = ik1, ik1 += handle->kk) {
            const char *const a1 = a0 + dka, *const b1 = b0 + dkb, *ai = a0, *bi = b0;
#if defined(LIBXSMM_GEMM_XCOPY_JIT)
            if (NULL == handle->copy_a.ptr)
#endif
            {
              if (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & handle->gemm_flags) &&
                 (LIBXSMM_GEMM_FLAG_TRANS_A & handle->gemm_flags) != 0) /* pure A-transpose */
              {
                LIBXSMM_ASSERT(ldo == handle->km);
                libxsmm_otrans_internal(at/*out*/, a0/*in*/, handle->itypesize, handle->lda/*ldi*/, ldo,
                  0, handle->kk, 0, handle->km, handle->kk/*tile*/, handle->km/*tile*/, kernel);
                ai = at;
              }
              else if (0 != (LIBXSMM_GEMM_HANDLE_FLAG_COPY_A & handle->flags)) {
                libxsmm_matcopy_internal(at/*out*/, a0/*in*/, handle->itypesize, handle->lda/*ldi*/, ldo,
                  0, handle->km, 0, handle->kk, handle->km/*tile*/, handle->kk/*tile*/, kernel);
                ai = at;
              }
            }
#if defined(LIBXSMM_GEMM_XCOPY_JIT)
            else { /* MCOPY/TCOPY kernel */
              LIBXSMM_MCOPY_CALL(handle->copy_a, handle->itypesize, a0, &handle->lda, at, &ldo);
              ai = at;
            }
#endif
#if defined(LIBXSMM_GEMM_XCOPY_JIT)
            if (NULL == handle->copy_b.ptr)
#endif
            {
              if (LIBXSMM_GEMM_FLAG_TRANS_AB != (LIBXSMM_GEMM_FLAG_TRANS_AB & handle->gemm_flags) &&
                 (LIBXSMM_GEMM_FLAG_TRANS_B & handle->gemm_flags) != 0) /* pure B-transpose */
              {
                libxsmm_otrans_internal(bt/*out*/, b0/*in*/, handle->itypesize, handle->ldb/*ldi*/, handle->kk/*ldo*/,
                  0, handle->kn, 0, handle->kk, handle->kn/*tile*/, handle->kk/*tile*/, kernel);
                bi = bt;
              }
              else if (0 != (LIBXSMM_GEMM_HANDLE_FLAG_COPY_B & handle->flags)) {
                libxsmm_matcopy_internal(bt/*out*/, b0/*in*/, handle->itypesize, handle->ldb/*ldi*/, handle->kk/*ldo*/,
                  0, handle->kk, 0, handle->kn, handle->kk/*tile*/, handle->kn/*tile*/, kernel);
                bi = bt;
              }
            }
#if defined(LIBXSMM_GEMM_XCOPY_JIT)
            else { /* MCOPY/TCOPY kernel */
              LIBXSMM_MCOPY_CALL(handle->copy_b, handle->itypesize, b0, &handle->ldb, bt, &handle->kk);
              bi = bt;
            }
#endif
            /* beta0-kernel on first-touch, beta1-kernel otherwise (beta0/beta1 are identical if beta=1) */
            LIBXSMM_MMCALL_PRF(handle->kernel[k0!=ik?1:0].xmm, ai, bi, ci, a1, b1, c0);
            a0 = a1;
            b0 = b1;
          }
          /* TODO: synchronize */
#if defined(LIBXSMM_GEMM_XCOPY_JIT)
          if (NULL == handle->copy_o.ptr)
#endif
          {
            if (LIBXSMM_GEMM_FLAG_TRANS_AB == (LIBXSMM_GEMM_FLAG_TRANS_AB & handle->gemm_flags)) {
              libxsmm_otrans_internal(c0/*out*/, ct/*in*/, handle->otypesize, handle->km/*ldi*/, handle->ldc/*ldo*/,
                0, handle->km, 0, handle->kn, handle->km/*tile*/, handle->kn/*tile*/, kernel);
            }
            else if (0 != (LIBXSMM_GEMM_HANDLE_FLAG_COPY_C & handle->flags)) {
              libxsmm_matcopy_internal(c0/*out*/, ct/*in*/, handle->otypesize, handle->km/*ldi*/, handle->ldc/*ldo*/,
                0, handle->km, 0, handle->kn, handle->km/*tile*/, handle->kn/*tile*/, kernel);
            }
          }
#if defined(LIBXSMM_GEMM_XCOPY_JIT)
          else { /* MCOPY/TCOPY kernel */
            LIBXSMM_MCOPY_CALL(handle->copy_o, handle->otypesize, ct, &handle->km, c0, &handle->ldc);
          }
#endif
        }
      }
    }
  }
#if !defined(NDEBUG)
  else if (/*implies LIBXSMM_INIT*/0 != libxsmm_get_verbosity()) { /* library code is expected to be mute */
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM ERROR: libxsmm_gemm_task - invalid handle!\n");
    }
  }
#endif
}


LIBXSMM_API void libxsmm_xgemm(libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc)
{
  libxsmm_gemm_blob blob;
  const libxsmm_gemm_handle *const handle = libxsmm_gemm_handle_init(&blob, iprec, oprec, transa, transb,
    m, n, k, lda, ldb, ldc, alpha, beta, LIBXSMM_GEMM_HANDLE_FLAG_AUTO, 1/*ntasks*/);
  const size_t scratch_size = libxsmm_gemm_handle_get_scratch_size(handle);
  void* scratch = NULL;
  if (NULL != handle && (0 == scratch_size ||
      NULL != (scratch = libxsmm_scratch_malloc(scratch_size, LIBXSMM_CACHELINE, LIBXSMM_MALLOC_INTERNAL_CALLER))))
  {
    libxsmm_gemm_task(handle, scratch, a, b, c, 0/*tid*/, 1/*ntasks*/);
    libxsmm_free(scratch);
  }
  else { /* fallback or error */
    static int error_once = 0;
    if (NULL == handle) { /* fallback */
      if ((LIBXSMM_VERBOSITY_HIGH <= libxsmm_verbosity || 0 > libxsmm_verbosity) /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM WARNING: XGEMM fallback code path triggered!\n");
      }
    }
    else if (0 != libxsmm_verbosity && /* library code is expected to be mute */
      1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: failed to allocate GEMM-scratch memory!\n");
    }
    libxsmm_blas_xgemm(iprec, oprec, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
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
    libxsmm_gemm_batch(LIBXSMM_GEMM_PRECISION_F64, LIBXSMM_GEMM_PRECISION_F64, transa_array + i, transb_array + i,
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
    libxsmm_gemm_batch(LIBXSMM_GEMM_PRECISION_F32, LIBXSMM_GEMM_PRECISION_F32, transa_array + i, transb_array + i,
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


LIBXSMM_API void libxsmm_wigemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const int* alpha, const short* a, const libxsmm_blasint* lda,
  const short* b, const libxsmm_blasint* ldb,
  const int* beta, int* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_XGEMM(short, int, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_API void libxsmm_bsgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const libxsmm_bfloat16* a, const libxsmm_blasint* lda,
  const libxsmm_bfloat16* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_XGEMM(libxsmm_bfloat16, float, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
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
        libxsmm_blasint i = begin * index_stride, ic = (NULL != stride_c ? (LIBXSMM_ACCESS(const libxsmm_blasint, stride_c, i) - index_base) : 0);
        const char* ai = &a0[NULL != stride_a ? ((LIBXSMM_ACCESS(const libxsmm_blasint, stride_a, i) - index_base) * itypesize) : 0];
        const char* bi = &b0[NULL != stride_b ? ((LIBXSMM_ACCESS(const libxsmm_blasint, stride_b, i) - index_base) * itypesize) : 0];
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
                const char *const an = &a0[(LIBXSMM_ACCESS(const libxsmm_blasint, stride_a, i) - index_base) << ibits];
                const char *const bn = &b0[(LIBXSMM_ACCESS(const libxsmm_blasint, stride_b, i) - index_base) << ibits];
                char       *const cn = &c0[(LIBXSMM_ACCESS(const libxsmm_blasint, stride_c, i) - index_base) << obits];
                kernel.xmm(ai, bi, ci, an, bn, cn); /* with prefetch */
                ai = an; bi = bn; ci = cn; /* next */
              }
            }
            else { /* non-pot type sizes */
              for (i += index_stride; i <= end1; i += index_stride) {
                const char *const an = &a0[(LIBXSMM_ACCESS(const libxsmm_blasint, stride_a, i) - index_base) * itypesize];
                const char *const bn = &b0[(LIBXSMM_ACCESS(const libxsmm_blasint, stride_b, i) - index_base) * itypesize];
                char       *const cn = &c0[(LIBXSMM_ACCESS(const libxsmm_blasint, stride_c, i) - index_base) * otypesize];
                kernel.xmm(ai, bi, ci, an, bn, cn); /* with prefetch */
                ai = an; bi = bn; ci = cn; /* next */
              }
            }
          }
          else { /* mixed specification of strides */
            for (i += index_stride; i <= end1; i += index_stride) {
              const char *const an = &a0[NULL != stride_a ? ((LIBXSMM_ACCESS(const libxsmm_blasint, stride_a, i) - index_base) * itypesize) : 0];
              const char *const bn = &b0[NULL != stride_b ? ((LIBXSMM_ACCESS(const libxsmm_blasint, stride_b, i) - index_base) * itypesize) : 0];
              char       *const cn = &c0[NULL != stride_c ? ((LIBXSMM_ACCESS(const libxsmm_blasint, stride_c, i) - index_base) * otypesize) : 0];
              kernel.xmm(ai, bi, ci, an, bn, cn); /* with prefetch */
              ai = an; bi = bn; ci = cn; /* next */
            }
          }
          if (end == size) { /* remainder multiplication */
            kernel.xmm(ai, bi, ci, ai, bi, ci); /* pseudo-prefetch */
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
              ic = LIBXSMM_ACCESS(const libxsmm_blasint, stride_c, i) - index_base;
              {
                const char *const an = &a0[(LIBXSMM_ACCESS(const libxsmm_blasint, stride_a, i) - index_base) * itypesize];
                const char *const bn = &b0[(LIBXSMM_ACCESS(const libxsmm_blasint, stride_b, i) - index_base) * itypesize];
                char       *const cn = &c0[ic * otypesize];
                LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK) *const lock1 = &internal_gemm_lock[LIBXSMM_GEMM_LOCKIDX(ic, internal_gemm_nlocks)].state;
# if defined(LIBXSMM_GEMM_LOCKFWD)
                if (lock != lock0) { lock0 = lock; LIBXSMM_LOCK_ACQUIRE(LIBXSMM_GEMM_LOCK, lock); }
# else
                LIBXSMM_LOCK_ACQUIRE(LIBXSMM_GEMM_LOCK, lock);
# endif
                kernel.xmm(ai, bi, ci, an, bn, cn); /* with prefetch */
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
              ic = (NULL != stride_c ? (LIBXSMM_ACCESS(const libxsmm_blasint, stride_c, i) - index_base) : 0);
              {
                const char *const an = &a0[NULL != stride_a ? ((LIBXSMM_ACCESS(const libxsmm_blasint, stride_a, i) - index_base) * itypesize) : 0];
                const char *const bn = &b0[NULL != stride_b ? ((LIBXSMM_ACCESS(const libxsmm_blasint, stride_b, i) - index_base) * itypesize) : 0];
                char       *const cn = &c0[ic * otypesize];
                LIBXSMM_LOCK_TYPE(LIBXSMM_GEMM_LOCK) *const lock1 = &internal_gemm_lock[LIBXSMM_GEMM_LOCKIDX(ic, internal_gemm_nlocks)].state;
# if defined(LIBXSMM_GEMM_LOCKFWD)
                if (lock != lock0) { lock0 = lock; LIBXSMM_LOCK_ACQUIRE(LIBXSMM_GEMM_LOCK, lock); }
# else
                LIBXSMM_LOCK_ACQUIRE(LIBXSMM_GEMM_LOCK, lock);
# endif
                kernel.xmm(ai, bi, ci, an, bn, cn); /* with prefetch */
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
            kernel.xmm(ai, bi, ci, ai, bi, ci); /* pseudo-prefetch */
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
                *((const void**)ai), *((const void**)bi), *((void**)ci),
                *((const void**)an), *((const void**)bn), *((const void**)cn));
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
              *((const void**)ai), *((const void**)bi), *((void**)ci),
              *((const void**)ai), *((const void**)bi), *((const void**)ci));
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
                *((const void**)ai), *((const void**)bi), cc,
                *((const void**)an), *((const void**)bn), *((const void**)cn));
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
              *((const void**)ai), *((const void**)bi), cc,
              *((const void**)ai), *((const void**)bi), cc);
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
          char *ci = &c0[NULL != stride_c ? ((LIBXSMM_ACCESS(const libxsmm_blasint, stride_c, i) - index_base) * otypesize) : 0], *cn = ci;
          do {
            for (count = 0; i < end_stride && ci == cn; ++count) {
              const size_t j = i + index_stride;
              *ai++ = &a0[NULL != stride_a ? ((LIBXSMM_ACCESS(const libxsmm_blasint, stride_a, i) - index_base) * itypesize) : 0];
              *bi++ = &b0[NULL != stride_b ? ((LIBXSMM_ACCESS(const libxsmm_blasint, stride_b, i) - index_base) * itypesize) : 0];
                 cn = &c0[NULL != stride_c ? ((LIBXSMM_ACCESS(const libxsmm_blasint, stride_c, j) - index_base) * otypesize) : 0];
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
      const int oprec = LIBXSMM_GETENUM_OUT(descriptor->datatype);
      const libxsmm_blasint typesize = LIBXSMM_TYPESIZE(oprec);
      const libxsmm_blasint csize = (libxsmm_blasint)descriptor->ldc * descriptor->n * typesize;
      /* finalize assumption if matrix-size is a multiple of the vector-width */
      descriptor->flags |= (unsigned short)(0 == LIBXSMM_MOD2(csize, vw) ? LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT : 0);
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
        case LIBXSMM_GEMM_PRECISION_F64: {
          if (LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_OUT(descriptor->datatype)) {
            result = EXIT_SUCCESS;
          }
        } break;
        case LIBXSMM_GEMM_PRECISION_F32: {
          if (LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_OUT(descriptor->datatype)) {
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
        const double *const an = &a0[NULL != stride_a ? (LIBXSMM_ACCESS(const libxsmm_blasint, stride_a, i) - index_base) : 0];
        const double *const bn = &b0[NULL != stride_b ? (LIBXSMM_ACCESS(const libxsmm_blasint, stride_b, i) - index_base) : 0];
        double       *const cn = &c0[NULL != stride_c ? (LIBXSMM_ACCESS(const libxsmm_blasint, stride_c, i) - index_base) : 0];
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
        const float *const an = &a0[NULL != stride_a ? (LIBXSMM_ACCESS(const libxsmm_blasint, stride_a, i) - index_base) : 0];
        const float *const bn = &b0[NULL != stride_b ? (LIBXSMM_ACCESS(const libxsmm_blasint, stride_b, i) - index_base) : 0];
        float       *const cn = &c0[NULL != stride_c ? (LIBXSMM_ACCESS(const libxsmm_blasint, stride_c, i) - index_base) : 0];
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
  libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb, const void* beta, void* c, const libxsmm_blasint* ldc,
  libxsmm_blasint index_base, libxsmm_blasint index_stride, const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize)
{
  int result;
  if (NULL != a && NULL != b && NULL != c) {
    switch (LIBXSMM_GETENUM(iprec, oprec)) {
      case LIBXSMM_GEMM_PRECISION_F64: {
        libxsmm_dmmbatch_blas(transa, transb, m, n, k,
          (const double*)alpha, a, lda, b, ldb, (const double*)beta, c, ldc,
          index_base, index_stride, stride_a, stride_b, stride_c, batchsize);
        result = EXIT_SUCCESS;
      } break;
      case LIBXSMM_GEMM_PRECISION_F32: {
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


LIBXSMM_API void libxsmm_mmbatch(libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec,
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


LIBXSMM_API void libxsmm_gemm_batch(libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec,
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
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_wigemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const int*, const short*, const libxsmm_blasint*,
  const short*, const libxsmm_blasint*,
  const int*, int*, const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_wigemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const int* alpha, const short* a, const libxsmm_blasint* lda,
  const short* b, const libxsmm_blasint* ldb,
  const int* beta, int* c, const libxsmm_blasint* ldc)
{
  libxsmm_wigemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_bsgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const libxsmm_bfloat16*, const libxsmm_blasint*,
  const libxsmm_bfloat16*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_bsgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const libxsmm_bfloat16* a, const libxsmm_blasint* lda,
  const libxsmm_bfloat16* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_bsgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_blas_xgemm)(const libxsmm_gemm_precision*, const libxsmm_gemm_precision*,
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*,
  const float*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_blas_xgemm)(const libxsmm_gemm_precision* iprec, const libxsmm_gemm_precision* oprec,
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
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_mmbatch)(const libxsmm_gemm_precision*, const libxsmm_gemm_precision*,
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const void*, const void*, const libxsmm_blasint*, const void*, const libxsmm_blasint*,
  const void*, void*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const libxsmm_blasint[], const libxsmm_blasint[], const libxsmm_blasint[],
  const libxsmm_blasint*, const /*unsigned*/int*, const /*unsigned*/int*);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_mmbatch)(const libxsmm_gemm_precision* iprec, const libxsmm_gemm_precision* oprec,
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
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_gemm_batch)(const libxsmm_gemm_precision*, const libxsmm_gemm_precision*,
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const void*, const void*, const libxsmm_blasint*, const void*, const libxsmm_blasint*,
  const void*, void*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const libxsmm_blasint[], const libxsmm_blasint[], const libxsmm_blasint[],
  const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_gemm_batch)(const libxsmm_gemm_precision* iprec, const libxsmm_gemm_precision* oprec,
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

