/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_spgemm_csr_asparse_reg.h"
#include <libxsmm_fsspmdm.h>
#include "generator_common.h"


LIBXSMM_API_INTERN
void libxsmm_fsspmdm_base_vlen( libxsmm_blasint N,
                                int i_fp64,
                                int* o_sparse,
                                int* o_dense);
LIBXSMM_API_INTERN
void libxsmm_fsspmdm_base_vlen( libxsmm_blasint N,
                                int i_fp64,
                                int* o_sparse,
                                int* o_dense) {
  int vl = libxsmm_cpuid_vlen32( libxsmm_target_archid );
  if ( i_fp64 ) {
    vl = LIBXSMM_UPDIV( vl, 2 );
  }

  *o_sparse = vl;
  *o_dense = vl;

  /* Dense NEON benefits from larger sizes */
  if ( libxsmm_target_archid >= LIBXSMM_AARCH64_V81 &&
       libxsmm_target_archid <= LIBXSMM_AARCH64_ALLFEAT &&
       libxsmm_target_archid != LIBXSMM_AARCH64_A64FX ) {
    if ( 0 == N % (2*vl) ) {
      *o_dense = 2*vl;
    }
    if ( 0 == N % (4*vl) ) {
      *o_dense = 4*vl;
    }
  }
}


LIBXSMM_API libxsmm_dfsspmdm* libxsmm_dfsspmdm_create(
  libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint K,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  const double alpha, const double beta, libxsmm_blasint c_is_nt,
  const double* a_dense)
{
  double* a_csr_values = NULL;
  unsigned int* a_csr_rowptr = NULL;
  unsigned int* a_csr_colidx = NULL;
  double* aa_dense = NULL;
  libxsmm_bitfield flags = LIBXSMM_GEMM_FLAGS('N', 'N') | ( ( beta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0 );
  libxsmm_bitfield prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  libxsmm_dfsspmdm* new_handle = NULL;
  libxsmm_gemmfunction k_sparse1 = NULL;
  libxsmm_gemmfunction k_sparse2 = NULL;
  libxsmm_gemmfunction k_sparse4 = NULL;
  libxsmm_gemmfunction k_dense = NULL;
  int i, j, n, nkerns, a_nnz = 0;
  int N_sparse1, N_sparse2, N_sparse4, N_dense;
  static int error_once = 0;

  /* Compute the vector/chunk sizes */
  libxsmm_fsspmdm_base_vlen( N, 1, &N_sparse1, &N_dense );
  N_sparse2 = 2*N_sparse1;
  N_sparse4 = 4*N_sparse1;

  /* some checks */
  if (0 != (N % N_sparse1)
    || (LIBXSMM_NEQ(beta, 1.0) && LIBXSMM_NEQ(beta, 0.0))
    || lda < K || ldc < N || ldb < N)
  {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR (libxsmm_dfsspmdm_create): cannot handle the given input!\n");
    }
    return NULL;
  }

  /* Get the number of non-zeros */
  for (i = 0; i < M; ++i) {
    for (j = 0; j < K; j++) {
      if (LIBXSMM_NEQ(a_dense[(i*lda) + j], 0.0)) {
        ++a_nnz;
      }
    }
  }

  /* Empty matrix */
  if (0 == a_nnz) {
    if ((LIBXSMM_VERBOSITY_WARN <= libxsmm_verbosity || 0 > libxsmm_verbosity)
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    { /* library code is expected to be mute */
      fprintf(stderr, "LIBXSMM WARNING (libxsmm_dfsspmdm_create): discovered an empty matrix!\n");
    }
    return NULL;
  }

  /* Allocate handle */
  new_handle = (libxsmm_dfsspmdm*)malloc(sizeof(libxsmm_dfsspmdm));
  if (NULL == new_handle) {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR (libxsmm_dfsspmdm_create): failed to allocate handle!\n");
    }
    return NULL;
  }

  /* Initialize the handle */
  LIBXSMM_MEMZERO127(new_handle);
  /* TODO: in case of ILP64, check value ranges */
  new_handle->N = (int)N;
  new_handle->M = (int)M;
  new_handle->K = (int)K;
  new_handle->ldb = (int)ldb;
  new_handle->ldc = (int)ldc;

  /* update flags */
  if ( beta == 0.0 && c_is_nt != 0 ) {
    flags |= LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT;
  }

  /* Allocate CSR structure */
  a_csr_values = (double*)malloc((size_t)a_nnz * sizeof(double));
  a_csr_rowptr = (unsigned int*)malloc(((size_t)M + 1) * sizeof(unsigned int));
  a_csr_colidx = (unsigned int*)malloc((size_t)a_nnz * sizeof(unsigned int));

  /* Consider dense case */
  if ( N_dense <= N ) {
    aa_dense = (double*)libxsmm_aligned_malloc((size_t)M * (size_t)K * sizeof(double), LIBXSMM_ALIGNMENT);
  }

  if ( NULL == a_csr_values || NULL == a_csr_rowptr || NULL == a_csr_colidx ) {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR (libxsmm_dfsspmdm_create): failed to allocate temporary buffers!\n");
    }
    free( a_csr_values ); free( a_csr_rowptr ); free( a_csr_colidx );
    free( new_handle );
    libxsmm_free( aa_dense );
    return NULL;
  }

  /* Populate CSR structure */
  for (i = 0, n = 0; i < M; ++i) {
    a_csr_rowptr[i] = n;
    for (j = 0; j < K; j++) {
      if (LIBXSMM_NEQ(a_dense[(i*lda) + j], 0.0)) {
        assert(n < a_nnz);
        a_csr_values[n] = alpha*a_dense[(i*lda) + j];
        a_csr_colidx[n] = j;
        ++n;
      }
    }
  }
  a_csr_rowptr[M] = a_nnz;

  LIBXSMM_HANDLE_ERROR_OFF_BEGIN();
  {
    /* Attempt to JIT a sparse kernel */
    if ( N_sparse1 <= N ) {
      const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(
        M, N_sparse1, K, 0, ldb, ldc, LIBXSMM_DATATYPE_F64,
        LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64 );
      k_sparse1 = libxsmm_create_spgemm_csr_areg_v2( gemm_shape, flags, prefetch_flags, N,
        a_csr_rowptr, a_csr_colidx, a_csr_values);
    }
    /* If that worked try to JIT a second (wider) sparse kernel */
    if ( NULL != k_sparse1 && 0 == (N % N_sparse2) ) {
      const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(
        M, N_sparse2, K, 0, ldb, ldc, LIBXSMM_DATATYPE_F64,
        LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64 );
      k_sparse2 = libxsmm_create_spgemm_csr_areg_v2( gemm_shape, flags, prefetch_flags, N,
        a_csr_rowptr, a_csr_colidx, a_csr_values);
    }
    /* And if that worked try going even wider still */
    if ( NULL != k_sparse2 && 0 == (N % N_sparse4) ) {
      const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(
        M, N_sparse4, K, 0, ldb, ldc, LIBXSMM_DATATYPE_F64,
        LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64 );
      k_sparse4 = libxsmm_create_spgemm_csr_areg_v2( gemm_shape, flags, prefetch_flags, N,
        a_csr_rowptr, a_csr_colidx, a_csr_values);
    }
  }
  LIBXSMM_HANDLE_ERROR_OFF_END();

  /* Free CSR */
  free( a_csr_values );
  free( a_csr_rowptr );
  free( a_csr_colidx );

  /* Also generate a dense kernel */
  if ( NULL != aa_dense ) {
    const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(
      N_dense, M, K, ldb, K, ldc, LIBXSMM_DATATYPE_F64,
      LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64 );
    k_dense = libxsmm_dispatch_gemm_v2( gemm_shape, flags, prefetch_flags );
  }

  if ( NULL != k_dense ) {
    assert(NULL != aa_dense);
    /* copy A over */
    for ( i = 0; i < M; ++i ) {
      for ( j = 0; j < K; ++j ) {
        aa_dense[i*K + j] = alpha*a_dense[i*lda + j];
      }
    }
  }

  /* Tally up how many kernels we got */
  nkerns = !!k_dense + !!k_sparse1 + !!k_sparse2 + !!k_sparse4;

  /* We have at least one kernel */
  if (0 < nkerns) {
    libxsmm_timer_tickint t;
    double *B = NULL, *C = NULL;
    double dt_dense = ( NULL != k_dense ) ? 1e5 : 1e6;
    double dt_sparse1 = ( NULL != k_sparse1 ) ? 1e5 : 1e6;
    double dt_sparse2 = ( NULL != k_sparse2 ) ? 1e5 : 1e6;
    double dt_sparse4 = ( NULL != k_sparse4 ) ? 1e5 : 1e6;
    libxsmm_gemm_param gemm_param;
    memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );

    /* If we have two or more kernels then try to benchmark them */
    if (2 <= nkerns) {
      B = (double*)libxsmm_aligned_malloc((size_t)K * (size_t)ldb * sizeof(double), LIBXSMM_ALIGNMENT);
      C = (double*)libxsmm_aligned_malloc((size_t)M * (size_t)ldc * sizeof(double), LIBXSMM_ALIGNMENT);

      if ( NULL != B && NULL != C ) {
        for ( i = 0; i < K; ++i ) {
          for ( j = 0; j < N; ++j ) {
            B[i*ldb + j] = 1;
          }
        }
        for ( i = 0; i < M; ++i ) {
          for ( j = 0; j < N; ++j ) {
            C[i*ldc + j] = 1;
          }
        }
      }
    }

    /* Benchmark dense */
    memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );
    if ( NULL != k_dense && NULL != B && NULL != C ) {
      t = libxsmm_timer_tick();
      for ( i = 0; i < 250; ++i ) {
        gemm_param.b.primary = (void*)aa_dense;
        for ( j = 0; j < N; j += N_dense ) {
          gemm_param.a.primary = (void*)(B+j);
          gemm_param.c.primary = (void*)(C+j);
          k_dense( &gemm_param );
        }
      }
      /* Bias to prefer dense kernels */
      dt_dense = libxsmm_timer_duration( t, libxsmm_timer_tick() ) / 1.1;
    }

    /* Benchmark sparse (regular) */
    memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );
    if ( NULL != k_sparse1 && NULL != B && NULL != C ) {
      t = libxsmm_timer_tick();
      gemm_param.b.primary = (void*)B;
      gemm_param.c.primary = (void*)C;
      for ( i = 0; i < 250; ++i ) {
        k_sparse1( &gemm_param );
      }
      dt_sparse1 = libxsmm_timer_duration( t, libxsmm_timer_tick() );
    }

    /* Benchmark sparse (wide) */
    memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );
    if ( NULL != k_sparse2 && NULL != B && NULL != C ) {
      t = libxsmm_timer_tick();
      gemm_param.b.primary = (void*)B;
      gemm_param.c.primary = (void*)C;
       for ( i = 0; i < 250; ++i ) {
        k_sparse2( &gemm_param );
      }
      dt_sparse2 = libxsmm_timer_duration( t, libxsmm_timer_tick() );
    }

    /* Benchmark sparse (widest) */
    memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );
    if ( NULL != k_sparse4 && NULL != B && NULL != C ) {
      t = libxsmm_timer_tick();
      gemm_param.b.primary = (void*)B;
      gemm_param.c.primary = (void*)C;
      for ( i = 0; i < 250; ++i ) {
        k_sparse4( &gemm_param );
      }
      dt_sparse4 = libxsmm_timer_duration( t, libxsmm_timer_tick() );
    }

    /* Dense fastest (or within 10%) */
    if ( dt_dense <= dt_sparse1 && dt_dense <= dt_sparse2 && dt_dense <= dt_sparse4 ) {
      assert(NULL != k_dense && NULL != aa_dense);
      new_handle->N_chunksize = N_dense;
      new_handle->kernel = k_dense;
      new_handle->a_dense = aa_dense;
    } else {
      libxsmm_free( aa_dense );
    }

    /* Sparse (regular) fastest */
    if ( dt_sparse1 < dt_dense && dt_sparse1 <= dt_sparse2 && dt_sparse1 <= dt_sparse4 ) {
      assert(NULL != k_sparse1);
      new_handle->kernel = k_sparse1;
    } else if ( NULL != k_sparse1 ) {
#if !defined(__APPLE__) && !defined(__arm64__)
      void* fp = NULL;
      LIBXSMM_ASSIGN127( &fp, &k_sparse1 );
      libxsmm_free( fp );
#endif
    }

    /* Sparse (wide) fastest */
    if ( dt_sparse2 < dt_dense && dt_sparse2 < dt_sparse1 && dt_sparse2 <= dt_sparse4 ) {
      assert(NULL != k_sparse2);
      new_handle->kernel = k_sparse2;
    } else if ( NULL != k_sparse2 ) {
#if !defined(__APPLE__) && !defined(__arm64__)
      void* fp = NULL;
      LIBXSMM_ASSIGN127( &fp, &k_sparse2 );
      libxsmm_free( fp );
#endif
    }

    /* Sparse (widest) fastest */
    if ( dt_sparse4 < dt_dense && dt_sparse4 < dt_sparse1 && dt_sparse4 < dt_sparse2 ) {
      assert(NULL != k_sparse4);
      new_handle->kernel = k_sparse4;
    } else if ( NULL != k_sparse4 ) {
#if !defined(__APPLE__) && !defined(__arm64__)
      void* fp = NULL;
      LIBXSMM_ASSIGN127( &fp, &k_sparse4 );
      libxsmm_free( fp );
#endif
    }

    libxsmm_free( B );
    libxsmm_free( C );
  }
  else {
    if ((LIBXSMM_VERBOSITY_WARN <= libxsmm_verbosity || 0 > libxsmm_verbosity)
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    { /* library code is expected to be mute */
      fprintf(stderr, "LIBXSMM WARNING (libxsmm_dfsspmdm_create): failed to provide a kernel!\n");
    }
    libxsmm_free( aa_dense );
    free( new_handle );
    new_handle = NULL;
  }

  return new_handle;
}


LIBXSMM_API libxsmm_sfsspmdm* libxsmm_sfsspmdm_create(
  libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint K,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  const float alpha, const float beta, libxsmm_blasint c_is_nt,
  const float* a_dense)
{
  double* a_csr_values = NULL;
  unsigned int* a_csr_rowptr = NULL;
  unsigned int* a_csr_colidx = NULL;
  float* aa_dense = NULL;
  libxsmm_bitfield flags = LIBXSMM_GEMM_FLAGS('N', 'N') | ( ( beta == 0 ) ? LIBXSMM_GEMM_FLAG_BETA_0 : 0 );
  libxsmm_bitfield prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  libxsmm_sfsspmdm* new_handle = NULL;
  libxsmm_gemmfunction k_sparse1 = NULL;
  libxsmm_gemmfunction k_sparse2 = NULL;
  libxsmm_gemmfunction k_sparse4 = NULL;
  libxsmm_gemmfunction k_dense = NULL;
  int i, j, n, nkerns, a_nnz = 0;
  int N_sparse1, N_sparse2, N_sparse4, N_dense;
  static int error_once = 0;

  /* Compute the vector/chunk sizes */
  libxsmm_fsspmdm_base_vlen(N, 0, &N_sparse1, &N_dense);
  N_sparse2 = 2*N_sparse1;
  N_sparse4 = 4*N_sparse1;

  /* some checks */
  if (0 != (N % N_sparse1)
    || (LIBXSMM_NEQ(beta, 1.0) && LIBXSMM_NEQ(beta, 0.0))
    || lda < K || ldc < N || ldb < N)
  {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR (libxsmm_dfsspmdm_create): cannot handle the given input!\n");
    }
    return NULL;
  }

  /* Get the number of non-zeros */
  for (i = 0; i < M; ++i) {
    for (j = 0; j < K; j++) {
      if (LIBXSMM_NEQ(a_dense[(i*lda) + j], 0.0)) {
        ++a_nnz;
      }
    }
  }

  /* Empty matrix */
  if (0 == a_nnz) {
    if ((LIBXSMM_VERBOSITY_WARN <= libxsmm_verbosity || 0 > libxsmm_verbosity)
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    { /* library code is expected to be mute */
      fprintf(stderr, "LIBXSMM WARNING (libxsmm_sfsspmdm_create): discovered an empty matrix!\n");
    }
    return NULL;
  }

  /* Allocate handle */
  new_handle = (libxsmm_sfsspmdm*)malloc(sizeof(libxsmm_sfsspmdm));
  if (NULL == new_handle) {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR (libxsmm_sfsspmdm_create): failed to allocate handle!\n");
    }
    return NULL;
  }

  /* Initialize the handle */
  LIBXSMM_MEMZERO127(new_handle);
  /* TODO: in case of ILP64, check value ranges */
  new_handle->N = (int)N;
  new_handle->M = (int)M;
  new_handle->K = (int)K;
  new_handle->ldb = (int)ldb;
  new_handle->ldc = (int)ldc;

  /* update flags */
  if ( beta == 0.0 && c_is_nt != 0 ) {
    flags |= LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT;
  }

  /* Allocate CSR structure */
  a_csr_values = (double*)malloc((size_t)a_nnz * sizeof(double));
  a_csr_rowptr = (unsigned int*)malloc(((size_t)M + 1) * sizeof(unsigned int));
  a_csr_colidx = (unsigned int*)malloc((size_t)a_nnz * sizeof(unsigned int));

  /* Consider dense case */
  if ( N_dense <= N ) {
    aa_dense = (float*)libxsmm_aligned_malloc((size_t)M * (size_t)K * sizeof(float), LIBXSMM_ALIGNMENT);
  }

  if ( NULL == a_csr_values || NULL == a_csr_rowptr || NULL == a_csr_colidx ) {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR (libxsmm_sfsspmdm_create): failed to allocate temporary buffers!\n");
    }
    free( a_csr_values ); free( a_csr_rowptr ); free( a_csr_colidx );
    free( new_handle );
    libxsmm_free( aa_dense );
    return NULL;
  }

  /* Populate CSR structure */
  for (i = 0, n = 0; i < M; ++i) {
    a_csr_rowptr[i] = n;
    for (j = 0; j < K; j++) {
      if (LIBXSMM_NEQ(a_dense[(i*lda) + j], 0.0f)) {
        assert(n < a_nnz);
        a_csr_values[n] = alpha*a_dense[(i*lda) + j];
        a_csr_colidx[n] = j;
        ++n;
      }
    }
  }
  a_csr_rowptr[M] = a_nnz;

  LIBXSMM_HANDLE_ERROR_OFF_BEGIN();
  {
    /* Attempt to JIT a sparse kernel */
    if ( N_sparse1 <= N ) {
      const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(
        M, N_sparse1, K, 0, ldb, ldc, LIBXSMM_DATATYPE_F32,
        LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
      k_sparse1 = libxsmm_create_spgemm_csr_areg_v2( gemm_shape, flags, prefetch_flags, N,
        a_csr_rowptr, a_csr_colidx, a_csr_values);
    }
    /* If that worked try to JIT a second (wider) sparse kernel */
    if ( NULL != k_sparse1 && 0 == (N % N_sparse2) ) {
      const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(
        M, N_sparse2, K, 0, ldb, ldc, LIBXSMM_DATATYPE_F32,
        LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
      k_sparse2 = libxsmm_create_spgemm_csr_areg_v2( gemm_shape, flags, prefetch_flags, N,
        a_csr_rowptr, a_csr_colidx, a_csr_values);
    }
    /* And if that worked try going even wider still */
    if ( NULL != k_sparse2 && 0 == (N % N_sparse4) ) {
      const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(
        M, N_sparse4, K, 0, ldb, ldc, LIBXSMM_DATATYPE_F32,
        LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
      k_sparse4 = libxsmm_create_spgemm_csr_areg_v2( gemm_shape, flags, prefetch_flags, N,
        a_csr_rowptr, a_csr_colidx, a_csr_values);
    }
  }
  LIBXSMM_HANDLE_ERROR_OFF_END();

  /* Free CSR */
  free( a_csr_values );
  free( a_csr_rowptr );
  free( a_csr_colidx );

  /* Also generate a dense kernel */
  if ( NULL != aa_dense ) {
    const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(
      N_dense, M, K, ldb, K, ldc, LIBXSMM_DATATYPE_F32,
      LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    k_dense = libxsmm_dispatch_gemm_v2( gemm_shape, flags, prefetch_flags );
  }

  if ( NULL != k_dense ) {
    assert(NULL != aa_dense);
    /* copy A over */
    for ( i = 0; i < M; ++i ) {
      for ( j = 0; j < K; ++j ) {
        aa_dense[i*K + j] = alpha*a_dense[i*lda + j];
      }
    }
  }

  /* Tally up how many kernels we got */
  nkerns = !!k_dense + !!k_sparse1 + !!k_sparse2 + !!k_sparse4;

  /* We have at least one kernel */
  if (0 < nkerns) {
    libxsmm_timer_tickint t;
    float *B = NULL, *C = NULL;
    double dt_dense = ( NULL != k_dense ) ? 1e5 : 1e6;
    double dt_sparse1 = ( NULL != k_sparse1 ) ? 1e5 : 1e6;
    double dt_sparse2 = ( NULL != k_sparse2 ) ? 1e5 : 1e6;
    double dt_sparse4 = ( NULL != k_sparse4 ) ? 1e5 : 1e6;
    libxsmm_gemm_param gemm_param;
    memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );

    /* If we have two or more kernels then try to benchmark them */
    if (2 <= nkerns) {
      B = (float*)libxsmm_aligned_malloc((size_t)K * (size_t)ldb * sizeof(float), LIBXSMM_ALIGNMENT);
      C = (float*)libxsmm_aligned_malloc((size_t)M * (size_t)ldc * sizeof(float), LIBXSMM_ALIGNMENT);

      if ( NULL != B && NULL != C ) {
        for ( i = 0; i < K; ++i ) {
          for ( j = 0; j < N; ++j ) {
            B[i*ldb + j] = 1;
          }
        }
        for ( i = 0; i < M; ++i ) {
          for ( j = 0; j < N; ++j ) {
            C[i*ldc + j] = 1;
          }
        }
      }
    }

    /* Benchmark dense */
    memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );
    if ( NULL != k_dense && NULL != B && NULL != C ) {
      t = libxsmm_timer_tick();
      for ( i = 0; i < 250; ++i ) {
        gemm_param.b.primary = (void*)aa_dense;
        for ( j = 0; j < N; j += N_dense ) {
          gemm_param.a.primary = (void*)(B+j);
          gemm_param.c.primary = (void*)(C+j);
          k_dense( &gemm_param );
        }
      }
      /* Bias to prefer dense kernels */
      dt_dense = libxsmm_timer_duration( t, libxsmm_timer_tick() ) / 1.1;
    }

    /* Benchmark sparse (regular) */
    memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );
    if ( NULL != k_sparse1 && NULL != B && NULL != C ) {
      t = libxsmm_timer_tick();
      gemm_param.b.primary = (void*)B;
      gemm_param.c.primary = (void*)C;
      for ( i = 0; i < 250; ++i ) {
        k_sparse1( &gemm_param );
      }
      dt_sparse1 = libxsmm_timer_duration( t, libxsmm_timer_tick() );
    }

    /* Benchmark sparse (wide) */
    memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );
    if ( NULL != k_sparse2 && NULL != B && NULL != C ) {
      t = libxsmm_timer_tick();
      gemm_param.b.primary = (void*)B;
      gemm_param.c.primary = (void*)C;
      for ( i = 0; i < 250; ++i ) {
        k_sparse2( &gemm_param );
      }
      dt_sparse2 = libxsmm_timer_duration( t, libxsmm_timer_tick() );
    }

    /* Benchmark sparse (widest) */
    memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );
    if ( NULL != k_sparse4 && NULL != B && NULL != C ) {
      t = libxsmm_timer_tick();
      gemm_param.b.primary = (void*)B;
      gemm_param.c.primary = (void*)C;
      for ( i = 0; i < 250; ++i ) {
        k_sparse4( &gemm_param );
      }
      dt_sparse4 = libxsmm_timer_duration( t, libxsmm_timer_tick() );
    }

    /* Dense fastest (or within 10%) */
    if ( dt_dense <= dt_sparse1 && dt_dense <= dt_sparse2 && dt_dense <= dt_sparse4 ) {
      assert(NULL != k_dense && NULL != aa_dense);
      new_handle->N_chunksize = N_dense;
      new_handle->kernel = k_dense;
      new_handle->a_dense = aa_dense;
    } else {
      libxsmm_free( aa_dense );
    }

    /* Sparse (regular) fastest */
    if ( dt_sparse1 < dt_dense && dt_sparse1 <= dt_sparse2 && dt_sparse1 <= dt_sparse4 ) {
      assert(NULL != k_sparse1);
      new_handle->kernel = k_sparse1;
    } else if ( NULL != k_sparse1 ) {
#if !defined(__APPLE__) && !defined(__arm64__)
      void* fp = NULL;
      LIBXSMM_ASSIGN127( &fp, &k_sparse1 );
      libxsmm_free( fp );
#endif
    }

    /* Sparse (wide) fastest */
    if ( dt_sparse2 < dt_dense && dt_sparse2 < dt_sparse1 && dt_sparse2 <= dt_sparse4 ) {
      assert(NULL != k_sparse2);
      new_handle->kernel = k_sparse2;
    } else if ( NULL != k_sparse2 ) {
#if !defined(__APPLE__) && !defined(__arm64__)
      void* fp = NULL;
      LIBXSMM_ASSIGN127( &fp, &k_sparse2 );
      libxsmm_free( fp );
#endif
    }

    /* Sparse (widest) fastest */
    if ( dt_sparse4 < dt_dense && dt_sparse4 < dt_sparse1 && dt_sparse4 < dt_sparse2 ) {
      assert(NULL != k_sparse4);
      new_handle->kernel = k_sparse4;
    } else if ( NULL != k_sparse4 ) {
#if !defined(__APPLE__) && !defined(__arm64__)
      void* fp = NULL;
      LIBXSMM_ASSIGN127( &fp, &k_sparse4 );
      libxsmm_free( fp );
#endif
    }

    libxsmm_free( B );
    libxsmm_free( C );
  }
  else {
    if ((LIBXSMM_VERBOSITY_WARN <= libxsmm_verbosity || 0 > libxsmm_verbosity)
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    { /* library code is expected to be mute */
      fprintf(stderr, "LIBXSMM WARNING (libxsmm_sfsspmdm_create): failed to provide a kernel!\n");
    }
    libxsmm_free( aa_dense );
    free( new_handle );
    new_handle = NULL;
  }

  return new_handle;
}


LIBXSMM_API void libxsmm_dfsspmdm_execute( const libxsmm_dfsspmdm* handle, const double* B, double* C )
{
  int i;
  libxsmm_gemm_param gemm_param;

  assert( handle != NULL );

  memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );
  if ( handle->a_dense == NULL ) {
    gemm_param.b.primary = (void*)B;
    gemm_param.c.primary = (void*)C;
    handle->kernel( &gemm_param );
  } else {
    gemm_param.b.primary = (void*)handle->a_dense;
    for ( i = 0; i < handle->N; i += handle->N_chunksize ) {
      gemm_param.a.primary = (void*)(B+i);
      gemm_param.c.primary = (void*)(C+i);
      handle->kernel( &gemm_param );
    }
  }
}


LIBXSMM_API void libxsmm_sfsspmdm_execute( const libxsmm_sfsspmdm* handle, const float* B, float* C )
{
  int i;
  libxsmm_gemm_param gemm_param;

  assert( handle != NULL );

  memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );
  if ( handle->a_dense == NULL ) {
    gemm_param.b.primary = (void*)B;
    gemm_param.c.primary = (void*)C;
    handle->kernel( &gemm_param );
  } else {
    gemm_param.b.primary = (void*)handle->a_dense;
    for ( i = 0; i < handle->N; i += handle->N_chunksize ) {
      gemm_param.a.primary = (void*)(B+i);
      gemm_param.c.primary = (void*)(C+i);
      handle->kernel( &gemm_param );
    }
  }
}


LIBXSMM_API void libxsmm_dfsspmdm_destroy( libxsmm_dfsspmdm* handle )
{
  assert( handle != NULL );

  if ( handle->a_dense != NULL ) {
    libxsmm_free( handle->a_dense );
  } else {
#if !defined(__APPLE__) && !defined(__arm64__)
    /* deallocate code known to be not registered; no index attached
       do not use libxsmm_release_kernel here! We also need to work
       around pointer-to-function to pointer-to-object conversion */
    void* fp = NULL;
    LIBXSMM_ASSIGN127( &fp, &handle->kernel );
    libxsmm_free( fp );
#endif
  }

  free( handle );
}


LIBXSMM_API void libxsmm_sfsspmdm_destroy( libxsmm_sfsspmdm* handle )
{
  assert( handle != NULL );

  if ( handle->a_dense != NULL ) {
    libxsmm_free(handle->a_dense);
  } else {
#if !defined(__APPLE__) && !defined(__arm64__)
    /* deallocate code known to be not registered; no index attached
       do not use libxsmm_release_kernel here! We also need to work
       around pointer-to-function to pointer-to-object conversion */
    void* fp = NULL;
    LIBXSMM_ASSIGN127( &fp, &handle->kernel );
    libxsmm_free( fp );
#endif
  }

  free( handle );
}
