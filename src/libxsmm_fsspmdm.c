/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Hans Pabst (Intel Corp.), Freddie Witherden
******************************************************************************/
#include <libxsmm_fsspmdm.h>
#include <libxsmm_memory.h>
#include "generator_spgemm_csr_asparse_reg.h"
#include "generator_common.h"

#if !defined(LIBXSMM_FSSPMDM_DENSE_BIAS)
# define LIBXSMM_FSSPMDM_DENSE_BIAS(VALUE, BIAS) LIBXSMM_UPF(VALUE, (BIAS) * -1, 100)
#endif
#if !defined(LIBXSMM_FSSPMDM_NTUNE)
# define LIBXSMM_FSSPMDM_NTUNE 250
#endif


LIBXSMM_API libxsmm_fsspmdm* libxsmm_fsspmdm_create(libxsmm_datatype datatype,
  libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint K, libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  const void* alpha, const void* beta, const void* a_dense, int c_is_nt, libxsmm_timer_tickint (*timer_tick)(void))
{
  libxsmm_bitfield flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  libxsmm_gemmfunction k_sparse1 = NULL;
  libxsmm_gemmfunction k_sparse2 = NULL;
  libxsmm_gemmfunction k_sparse4 = NULL;
  libxsmm_gemmfunction k_dense = NULL;
  libxsmm_fsspmdm* new_handle = NULL;
  const char *const env_fsspmdm_hint = getenv("LIBXSMM_FSSPMDM_HINT");
  const int fsspmdm_hint = (NULL == env_fsspmdm_hint ? 0 : atoi(env_fsspmdm_hint));
  int N_sparse1, N_sparse2, N_sparse4, N_dense, i, j, n, nkerns, a_nnz = 0;
  unsigned char typesize = 0;
  static int error_once = 0;
  unsigned int* a_csr_rowptr = NULL;
  unsigned int* a_csr_colidx = NULL;
  double* a_csr_values = NULL;
  void* aa_dense = NULL;

  /* count number of attempts to create handle */
  LIBXSMM_EXPECT(0 < LIBXSMM_ATOMIC_ADD_FETCH(&libxsmm_statistic_num_spmdm, 1, LIBXSMM_ATOMIC_RELAXED));

  if (NULL == a_dense) { /* basic checks */
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR (libxsmm_fsspmdm_create): invalid input!\n");
    }
    return NULL;
  }

  LIBXSMM_INIT
  typesize = libxsmm_typesize(datatype);
  { /* Compute the vector/chunk sizes */
    const int vlen = libxsmm_cpuid_vlen(libxsmm_target_archid);
    const int vl = LIBXSMM_UPDIV(vlen, typesize);
    LIBXSMM_ASSERT(0 < vl);
    N_sparse1 = N_dense = vl;
    /* Dense NEON benefits from larger sizes */
    if (libxsmm_target_archid >= LIBXSMM_AARCH64_V81 &&
      libxsmm_target_archid <= LIBXSMM_AARCH64_ALLFEAT &&
      libxsmm_target_archid != LIBXSMM_AARCH64_A64FX)
    {
      if (0 == (N % (2 * vl))) {
        N_dense = 2 * vl;
      }
      if (0 == (N % (4 * vl))) {
        N_dense = 4 * vl;
      }
    }
  }
  N_sparse2 = 2 * N_sparse1;
  N_sparse4 = 4 * N_sparse1;

  switch ((int)datatype) {
    case LIBXSMM_DATATYPE_F64: {
      const double fbeta = (NULL != beta ? (*(const double*)beta) : LIBXSMM_BETA);
      if (0 == (N % N_sparse1)
        && (LIBXSMM_FEQ(fbeta, 1) || LIBXSMM_FEQ(fbeta, 0))
        && lda >= K && ldc >= N && ldb >= N)
      {
        /* Get the number of non-zeros */
        for (i = 0; i < M; ++i) for (j = 0; j < K; ++j) {
          if (LIBXSMM_NEQ(((const double*)a_dense)[i * lda + j], 0)) {
            ++a_nnz;
          }
        }
        if (LIBXSMM_FEQ(0, fbeta)) { /* update flags */
          flags |= (0 != c_is_nt ? LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT : 0);
          flags |= LIBXSMM_GEMM_FLAG_BETA_0;
        }
      }
      else typesize = 0;
    } break;
    case LIBXSMM_DATATYPE_F32: {
      const float fbeta = (NULL != beta ? (*(const float*)beta) : LIBXSMM_BETA);
      if (0 == (N % N_sparse1)
        && (LIBXSMM_FEQ(fbeta, 1) || LIBXSMM_FEQ(fbeta, 0))
        && lda >= K && ldc >= N && ldb >= N)
      {
        /* Get the number of non-zeros */
        for (i = 0; i < M; ++i) for (j = 0; j < K; ++j) {
          if (LIBXSMM_NEQ(((const float*)a_dense)[i * lda + j], 0)) {
            ++a_nnz;
          }
        }
        if (LIBXSMM_FEQ(0, fbeta)) { /* update flags */
          flags |= (0 != c_is_nt ? LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT : 0);
          flags |= LIBXSMM_GEMM_FLAG_BETA_0;
        }
      }
      else typesize = 0;
    } break;
    default: typesize = 0;
  }

  if (0 == typesize || 0 == LIBXSMM_IS_INT(ldb) || 0 == LIBXSMM_IS_INT(ldc) ||
      0 == LIBXSMM_IS_INT(M) || 0 == LIBXSMM_IS_INT(N) || 0 == LIBXSMM_IS_INT(K))
  { /* unsupported */
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR (libxsmm_fsspmdm_create): unsupported input!\n");
    }
    return NULL;
  }

  if (0 == a_nnz) { /* empty matrix */
    if ((LIBXSMM_VERBOSITY_WARN <= libxsmm_verbosity || 0 > libxsmm_verbosity)
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    { /* library code is expected to be mute */
      fprintf(stderr, "LIBXSMM WARNING (libxsmm_fsspmdm_create): discovered an empty matrix!\n");
    }
    return NULL;
  }

  /* Allocate handle */
  new_handle = (libxsmm_fsspmdm*)malloc(sizeof(libxsmm_fsspmdm));
  if (NULL == new_handle) {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR (libxsmm_fsspmdm_create): failed to allocate handle!\n");
    }
    return NULL;
  }

  /* Initialize the handle */
  LIBXSMM_MEMZERO127(new_handle);
  new_handle->datatype = datatype;
  new_handle->N = (int)N;
  new_handle->M = (int)M;
  new_handle->K = (int)K;
  new_handle->ldb = (int)ldb;
  new_handle->ldc = (int)ldc;

  /* Allocate CSR structure */
  a_csr_values = (double*)malloc(sizeof(double) * a_nnz);
  a_csr_rowptr = (unsigned int*)malloc(sizeof(unsigned int) * ((size_t)M + 1) );
  a_csr_colidx = (unsigned int*)malloc(sizeof(unsigned int) * a_nnz);

  /* Consider dense case */
  if (N_dense <= N) {
    aa_dense = libxsmm_aligned_malloc((size_t)M * K * typesize, LIBXSMM_ALIGNMENT);
  }

  if (NULL == a_csr_values || NULL == a_csr_rowptr || NULL == a_csr_colidx) {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR (libxsmm_fsspmdm_create): failed to allocate temporary buffers!\n");
    }
    free(a_csr_values); free(a_csr_rowptr); free(a_csr_colidx); free(new_handle);
    libxsmm_free(aa_dense);
    return NULL;
  }

  /* Also generate a dense kernel */
  if (NULL != aa_dense) {
    const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(
      N_dense, M, K, ldb, K, ldc, datatype, datatype, datatype, datatype);
    k_dense = libxsmm_dispatch_gemm(gemm_shape, flags, prefetch_flags);
  }

  switch ((int)datatype) {
    case LIBXSMM_DATATYPE_F64: {
      const double falpha = (NULL != alpha ? (*(const double*)alpha) : LIBXSMM_ALPHA);
      const double *const A = (const double*)a_dense;
      LIBXSMM_ASSERT(NULL == k_dense || NULL != aa_dense);
      /* Populate CSR structure, and copy A-matrix */
      for (i = 0, n = 0; i < M; ++i) {
        a_csr_rowptr[i] = n;
        for (j = 0; j < K; ++j) {
          const double aij_alpha = falpha * A[i*lda+j];
          if (LIBXSMM_NEQ(aij_alpha, 0)) {
            LIBXSMM_ASSERT(n < a_nnz);
            a_csr_values[n] = aij_alpha;
            a_csr_colidx[n] = j;
            ++n;
          }
          if (NULL != k_dense) { /* copy A-matrix */
            ((double*)aa_dense)[i * K + j] = aij_alpha;
          }
        }
      }
      LIBXSMM_ASSERT(n <= a_nnz);
      a_csr_rowptr[M] = n;
    } break;
    case LIBXSMM_DATATYPE_F32: {
      const float falpha = (NULL != alpha ? (*(const float*)alpha) : LIBXSMM_ALPHA);
      const float *const A = (const float*)a_dense;
      LIBXSMM_ASSERT(NULL == k_dense || NULL != aa_dense);
      /* Populate CSR structure, and copy A-matrix */
      for (i = 0, n = 0; i < M; ++i) {
        a_csr_rowptr[i] = n;
        for (j = 0; j < K; ++j) {
          const float aij_alpha = falpha * A[i * lda + j];
          if (LIBXSMM_NEQ(aij_alpha, 0)) {
            LIBXSMM_ASSERT(n < a_nnz);
            a_csr_values[n] = aij_alpha;
            a_csr_colidx[n] = j;
            ++n;
          }
          if (NULL != k_dense) { /* copy A-matrix */
            ((float*)aa_dense)[i * K + j] = aij_alpha;
          }
        }
      }
      LIBXSMM_ASSERT(n <= a_nnz);
      a_csr_rowptr[M] = n;
    } break;
    default: LIBXSMM_ASSERT_MSG(0, "Should not happen");
  }

  LIBXSMM_HANDLE_ERROR_OFF_BEGIN();
  {
    /* Attempt to JIT a sparse kernel */
    if (N_sparse1 <= N && (0 == fsspmdm_hint || 1 == fsspmdm_hint)) {
      const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(
        M, N_sparse1, K, 0, ldb, ldc, datatype, datatype, datatype, datatype);
      k_sparse1 = libxsmm_create_spgemm_csr_areg(gemm_shape, flags, prefetch_flags, N,
        a_csr_rowptr, a_csr_colidx, a_csr_values);
    }
    /* Try to JIT a second (wider) sparse kernel */
    if (N_sparse1 <= N && 0 == (N % N_sparse2) && (0 == fsspmdm_hint || 2 == fsspmdm_hint)) {
      const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(
        M, N_sparse2, K, 0, ldb, ldc, datatype, datatype, datatype, datatype);
      k_sparse2 = libxsmm_create_spgemm_csr_areg(gemm_shape, flags, prefetch_flags, N,
        a_csr_rowptr, a_csr_colidx, a_csr_values);
    }
    /* Try to JIT an even wider kernel */
    if (N_sparse1 <= N && 0 == (N % N_sparse4) && (0 == fsspmdm_hint || 3 == fsspmdm_hint)) {
      const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(
        M, N_sparse4, K, 0, ldb, ldc, datatype, datatype, datatype, datatype);
      k_sparse4 = libxsmm_create_spgemm_csr_areg(gemm_shape, flags, prefetch_flags, N,
        a_csr_rowptr, a_csr_colidx, a_csr_values);
    }
  }
  LIBXSMM_HANDLE_ERROR_OFF_END();

  /* Free CSR */
  free(a_csr_values);
  free(a_csr_rowptr);
  free(a_csr_colidx);

  /* Tally up how many kernels we got */
  nkerns = !!k_dense + !!k_sparse1 + !!k_sparse2 + !!k_sparse4;

  /* We have at least one kernel */
  if (0 < nkerns) {
    void *B = NULL, *C = NULL;
    libxsmm_timer_tickint dt_sparse1 = (libxsmm_timer_tickint)-1;
    libxsmm_timer_tickint dt_sparse2 = (libxsmm_timer_tickint)-1;
    libxsmm_timer_tickint dt_sparse4 = (libxsmm_timer_tickint)-1;
    libxsmm_timer_tickint dt_dense = (libxsmm_timer_tickint)-1;
    libxsmm_gemm_param gemm_param;
    libxsmm_timer_tickint s, t;

    /* Run benchmark if there are at least two kernels and a timer routine */
    if (2 <= nkerns && NULL != timer_tick) {
      B = libxsmm_aligned_malloc((size_t)K * ldb * typesize, LIBXSMM_ALIGNMENT);
      C = libxsmm_aligned_malloc((size_t)M * ldc * typesize, LIBXSMM_ALIGNMENT);
      if (NULL != B && NULL != C) {
        switch ((int)datatype) {
          case LIBXSMM_DATATYPE_F64: {
            const double fbeta = (NULL != beta ? (*(const double*)beta) : LIBXSMM_BETA);
            LIBXSMM_MATINIT(double, 0/*seed*/, B, N, K, ldb, 1/*scale*/);
            if (LIBXSMM_NEQ(fbeta, 0)) {
              LIBXSMM_MATINIT(double, 0/*seed*/, C, N, M, ldc, 1/*scale*/);
            }
          } break;
          case LIBXSMM_DATATYPE_F32: {
            const float fbeta = (NULL != beta ? (*(const float*)beta) : LIBXSMM_BETA);
            LIBXSMM_MATINIT(float, 0/*seed*/, B, N, K, ldb, 1/*scale*/);
            if (LIBXSMM_NEQ(fbeta, 0)) {
              LIBXSMM_MATINIT(float, 0/*seed*/, C, N, M, ldc, 1/*scale*/);
            }
          } break;
          default: LIBXSMM_ASSERT_MSG(0, "Should not happen");
        }
      }
    }

    /* Benchmark dense */
    if (NULL != k_dense && NULL != B && NULL != C) {
      const char *const env_fsspmdm_dense_bias = getenv("LIBXSMM_FSSPMDM_DENSE_BIAS");
      const int dense_bias_env = (NULL == env_fsspmdm_dense_bias ? 0 : atoi(env_fsspmdm_dense_bias));
      const int dense_bias = (0 == dense_bias_env ? 10/*default*/ : LIBXSMM_CLMP(dense_bias_env, 0, 100));
#if defined(_DEBUG)
      memset(&gemm_param, 0, sizeof(libxsmm_gemm_param));
#endif
      LIBXSMM_ASSERT(NULL != timer_tick);
      LIBXSMM_ASSERT(NULL != aa_dense);
      gemm_param.b.primary = aa_dense;
      s = timer_tick();
      for (i = 0; i < LIBXSMM_FSSPMDM_NTUNE; ++i) {
        for (j = 0; j < N; j += N_dense) {
          gemm_param.a.primary = (char*)B + j * typesize;
          gemm_param.c.primary = (char*)C + j * typesize;
          k_dense(&gemm_param);
        }
      }
      /* Bias to prefer dense kernels */
      t = timer_tick();
      dt_dense = LIBXSMM_FSSPMDM_DENSE_BIAS(LIBXSMM_DELTA(s, t), dense_bias);
    }

    /* Benchmark sparse (regular) */
    if (NULL != k_sparse1 && NULL != B && NULL != C) {
#if defined(_DEBUG)
      memset(&gemm_param, 0, sizeof(libxsmm_gemm_param));
#endif
      LIBXSMM_ASSERT(NULL != timer_tick);
      gemm_param.a.primary = NULL;
      gemm_param.b.primary = B;
      gemm_param.c.primary = C;
      s = timer_tick();
      for (i = 0; i < LIBXSMM_FSSPMDM_NTUNE; ++i) {
        k_sparse1(&gemm_param);
      }
      t = timer_tick();
      dt_sparse1 = LIBXSMM_DELTA(s, t);
    }

    /* Benchmark sparse (wide) */
    if (NULL != k_sparse2 && NULL != B && NULL != C) {
#if defined(_DEBUG)
      memset(&gemm_param, 0, sizeof(libxsmm_gemm_param));
#endif
      LIBXSMM_ASSERT(NULL != timer_tick);
      gemm_param.a.primary = NULL;
      gemm_param.b.primary = B;
      gemm_param.c.primary = C;
      s = timer_tick();
      for (i = 0; i < LIBXSMM_FSSPMDM_NTUNE; ++i) {
        k_sparse2(&gemm_param);
      }
      t = timer_tick();
      dt_sparse2 = LIBXSMM_DELTA(s, t);
    }

    /* Benchmark sparse (widest) */
    if (NULL != k_sparse4 && NULL != B && NULL != C) {
#if defined(_DEBUG)
      memset(&gemm_param, 0, sizeof(libxsmm_gemm_param));
#endif
      LIBXSMM_ASSERT(NULL != timer_tick);
      gemm_param.a.primary = NULL;
      gemm_param.b.primary = B;
      gemm_param.c.primary = C;
      s = timer_tick();
      for (i = 0; i < LIBXSMM_FSSPMDM_NTUNE; ++i) {
        k_sparse4(&gemm_param);
      }
      t = timer_tick();
      dt_sparse4 = LIBXSMM_DELTA(s, t);
    }

    /* Dense fastest (or within LIBXSMM_FSSPMDM_DENSE_BIAS) */
    if (NULL != k_dense && ((0 == fsspmdm_hint
        && dt_dense <= dt_sparse1
        && dt_dense <= dt_sparse2
        && dt_dense <= dt_sparse4)
      || (4 <= fsspmdm_hint || 0 > fsspmdm_hint)))
    {
      LIBXSMM_ASSERT(NULL != aa_dense);
      new_handle->N_chunksize = N_dense;
      new_handle->kernel = k_dense;
      new_handle->a_dense = aa_dense;
    }

    /* Sparse (regular) fastest */
    if (NULL != k_sparse1 && ((0 == fsspmdm_hint
        && dt_sparse1 < dt_dense
        && dt_sparse1 <= dt_sparse2
        && dt_sparse1 <= dt_sparse4)
      || 1 == fsspmdm_hint))
    {
      LIBXSMM_ASSERT(NULL != k_sparse1);
      new_handle->kernel = k_sparse1;
    }
    else if (NULL != k_sparse1) {
      void* fp = NULL;
      LIBXSMM_ASSIGN127(&fp, &k_sparse1);
      libxsmm_free(fp);
    }

    /* Sparse (wide) fastest */
    if (NULL != k_sparse2 && ((0 == fsspmdm_hint
        && dt_sparse2 < dt_dense
        && dt_sparse2 < dt_sparse1
        && dt_sparse2 <= dt_sparse4)
      || 2 == fsspmdm_hint))
    {
      LIBXSMM_ASSERT(NULL != k_sparse2);
      new_handle->kernel = k_sparse2;
    }
    else if (NULL != k_sparse2) {
      void* fp = NULL;
      LIBXSMM_ASSIGN127(&fp, &k_sparse2);
      libxsmm_free(fp);
    }

    /* Sparse (widest) fastest */
    if (NULL != k_sparse4 && ((0 == fsspmdm_hint
        && dt_sparse4 < dt_dense
        && dt_sparse4 < dt_sparse1
        && dt_sparse4 < dt_sparse2)
      || 3 == fsspmdm_hint))
    {
      LIBXSMM_ASSERT(NULL != k_sparse4);
      new_handle->kernel = k_sparse4;
    }
    else if (NULL != k_sparse4) {
      void* fp = NULL;
      LIBXSMM_ASSIGN127(&fp, &k_sparse4);
      libxsmm_free(fp);
    }

    if (k_dense != new_handle->kernel) {
      if (NULL != new_handle->kernel) {
        libxsmm_free(aa_dense);
      }
      else { /* fallback */
        LIBXSMM_ASSERT(NULL != k_dense && NULL != aa_dense);
        new_handle->N_chunksize = N_dense;
        new_handle->kernel = k_dense;
        new_handle->a_dense = aa_dense;
      }
    }

    libxsmm_free(B);
    libxsmm_free(C);
  }
  else {
    if ((LIBXSMM_VERBOSITY_WARN <= libxsmm_verbosity || 0 > libxsmm_verbosity)
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    { /* library code is expected to be mute */
      fprintf(stderr, "LIBXSMM WARNING (libxsmm_fsspmdm_create): failed to provide a kernel!\n");
    }
    libxsmm_free(aa_dense);
    free(new_handle);
    new_handle = NULL;
  }

  return new_handle;
}


LIBXSMM_API libxsmm_dfsspmdm* libxsmm_dfsspmdm_create(
  libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint K, libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  double alpha, double beta, const double* a_dense, int c_is_nt, libxsmm_timer_tickint (*timer_tick)(void))
{
  return libxsmm_fsspmdm_create(LIBXSMM_DATATYPE_F64, M, N, K, lda, ldb, ldc, &alpha, &beta, a_dense, c_is_nt, timer_tick);
}


LIBXSMM_API libxsmm_sfsspmdm* libxsmm_sfsspmdm_create(
  libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint K, libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  float alpha, float beta, const float* a_dense, int c_is_nt, libxsmm_timer_tickint (*timer_tick)(void))
{
  return libxsmm_fsspmdm_create(LIBXSMM_DATATYPE_F32, M, N, K, lda, ldb, ldc, &alpha, &beta, a_dense, c_is_nt, timer_tick);
}


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
LIBXSMM_API void libxsmm_fsspmdm_execute(const libxsmm_fsspmdm* handle, const void* B, void* C)
{
  libxsmm_gemm_param gemm_param;
  LIBXSMM_ASSERT(NULL != handle);
#if defined(_DEBUG)
  memset(&gemm_param, 0, sizeof(libxsmm_gemm_param));
#endif
  if (NULL == handle->a_dense) {
    gemm_param.a.primary = NULL;
    gemm_param.b.primary = (void*)B;
    gemm_param.c.primary = (void*)C;
    handle->kernel(&gemm_param);
  }
  else {
    const unsigned char typesize = libxsmm_typesize(handle->datatype);
    int i;
    gemm_param.b.primary = handle->a_dense;
    for (i = 0; i < handle->N; i += handle->N_chunksize) {
      gemm_param.a.primary = (char*)B + i * typesize;
      gemm_param.c.primary = (char*)C + i * typesize;
      handle->kernel(&gemm_param);
    }
  }
}
#pragma GCC diagnostic pop


LIBXSMM_API void libxsmm_dfsspmdm_execute(const libxsmm_dfsspmdm* handle, const double* B, double* C)
{
  LIBXSMM_ASSERT(NULL != handle && LIBXSMM_DATATYPE_F64 == handle->datatype);
  libxsmm_fsspmdm_execute(handle, B, C);
}


LIBXSMM_API void libxsmm_sfsspmdm_execute(const libxsmm_sfsspmdm* handle, const float* B, float* C)
{
  LIBXSMM_ASSERT(NULL != handle && LIBXSMM_DATATYPE_F32 == handle->datatype);
  libxsmm_fsspmdm_execute(handle, B, C);
}


LIBXSMM_API void libxsmm_fsspmdm_destroy(libxsmm_fsspmdm* handle)
{
  if (NULL != handle) {
    if (handle->a_dense != NULL) {
      libxsmm_free(handle->a_dense);
    }
    else {
      /* deallocate code known to be not registered; no index attached
         do not use libxsmm_release_kernel here! We also need to work
         around pointer-to-function to pointer-to-object conversion */
      void* fp = NULL;
      LIBXSMM_ASSIGN127(&fp, &handle->kernel);
      libxsmm_free(fp);
    }
    free(handle);
  }
}


LIBXSMM_API void libxsmm_dfsspmdm_destroy(libxsmm_dfsspmdm* handle)
{
  libxsmm_fsspmdm_destroy(handle);
}


LIBXSMM_API void libxsmm_sfsspmdm_destroy(libxsmm_sfsspmdm* handle)
{
  libxsmm_fsspmdm_destroy(handle);
}
