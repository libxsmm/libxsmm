/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_spgemm_csr_asparse_reg.h"
#include <libxsmm_fsspmdm.h>
#include "generator_common.h"


LIBXSMM_API_INTERN const double* internal_dfsspmdm_init(void);
LIBXSMM_API_INTERN const double* internal_dfsspmdm_init(void)
{
  /* Double precision AVX-512 lane broadcasts */
  LIBXSMM_ALIGNED(static const unsigned int perm[], LIBXSMM_ALIGNMENT) = {
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
    4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
    6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
    8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
    10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
    12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
    14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15
  };
  static const double* result = NULL;
  if (NULL == result) { /* internal lazy initialization */
    result = (const double*)((const void*)perm);
    LIBXSMM_INIT
  }
  return result;
}


LIBXSMM_API_INTERN const float* internal_sfsspmdm_init(void);
LIBXSMM_API_INTERN const float* internal_sfsspmdm_init(void)
{
  /* Single precision AVX-512 lane broadcasts */
  LIBXSMM_ALIGNED(static const unsigned int perm[], LIBXSMM_ALIGNMENT) = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
    14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15
  };
  static const float* result = NULL;
  if (NULL == result) { /* internal lazy initialization */
    result = (const float*)((const void*)perm);
    LIBXSMM_INIT
  }
  return result;
}


LIBXSMM_API libxsmm_dfsspmdm* libxsmm_dfsspmdm_create(
  libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint K,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  const double alpha, const double beta, libxsmm_blasint c_is_nt,
  const double* a_dense)
{
  const double *const perm = internal_dfsspmdm_init(); /* shall be first */
  const double one = 1.0;
  double* a_csr_values = NULL;
  unsigned int* a_csr_rowptr = NULL;
  unsigned int* a_csr_colidx = NULL;
  double* aa_dense = NULL;
  int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  const int prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  const libxsmm_gemm_descriptor* xgemm_desc;
  libxsmm_descriptor_blob xgemm_blob;
  libxsmm_dfsspmdm* new_handle = NULL;
  libxsmm_dmmfunction k_sparse1 = NULL;
  libxsmm_dmmfunction k_sparse2 = NULL;
  libxsmm_dmmfunction k_dense = NULL;
  int i, j, n, nkerns, a_nnz = 0;
  const int vlen = LIBXSMM_UPDIV(libxsmm_cpuid_vlen32(libxsmm_target_archid), 2);
  const int N_sparse1 = vlen;
  const int N_sparse2 = vlen * 2;
  const int N_dense = vlen;
  static int error_once = 0;

  /* TODO: new vars */
  unsigned int a_unique;
  double* a_unique_values = NULL;
  unsigned int a_unique_reg;
  int archid;
  int unlimited_unique;
  const double* a_sparse;

  /* some checks */
  if (0 != (N % vlen)
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

  /* Let's figure out how many unique values we have */
  a_unique_values = (double*)(0 != a_nnz ? malloc(sizeof(double) * a_nnz) : NULL);

  if(NULL == a_unique_values) {
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR (libxsmm_dfsspmdm_create): failed to packed A matrix!\n");
    }
    return NULL;
  }

  a_unique = 1;
  a_unique_values[0] = fabs(a_csr_values[0]);

  for (n = 1; n < a_nnz; n++) {
    unsigned int l_hit = 0;
    /* search for the value */
    for (i = 0; i < a_unique; i++) {
      if (!(a_unique_values[i] < fabs(a_csr_values[n])) &&
          !(a_unique_values[i] > fabs(a_csr_values[n]))) /*a_unique_values[i] == a_csr_values[n]*/ {
        l_hit = 1;
        break;
      }
    }
    /* value was not found */
    if (l_hit == 0) {
      a_unique_values[a_unique] = fabs(a_csr_values[n]);
      a_unique++;
    }
  }

  /* check if A contains too many unique non-zeros for storing in registers */
  archid = libxsmm_get_target_archid();

  if (LIBXSMM_X86_AVX2 == archid) {
    a_unique_reg = 15;

  } else if (LIBXSMM_X86_AVX512_MIC == archid ||
            LIBXSMM_X86_AVX512_KNM == archid ||
            LIBXSMM_X86_AVX512_CORE == archid ||
            LIBXSMM_X86_AVX512_CLX == archid ||
            LIBXSMM_X86_AVX512_CPX == archid ||
            LIBXSMM_X86_AVX512_SPR == archid) {
    a_unique_reg = 240;

  } else {
    free( a_csr_values ); free( a_csr_rowptr ); free( a_csr_colidx );
    free( new_handle );
    free( a_unique_values );
    libxsmm_free( aa_dense );
    if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR (libxsmm_dfsspmdm_create): unsupported architecture!\n");
    }
    return NULL;
  }

  unlimited_unique = a_unique > a_unique_reg ? 1 : 0;

  if (unlimited_unique) {
    a_sparse = a_unique_values;
  } else {
    a_sparse = perm;
  }

  LIBXSMM_HANDLE_ERROR_OFF_BEGIN();
  {
    /* Attempt to JIT a sparse kernel */
    if ( N_sparse1 <= N ) {
      xgemm_desc = libxsmm_dgemm_descriptor_init(&xgemm_blob, M, N_sparse1, K,
                                                 0, ldb, ldc, one, beta, flags, prefetch);
      if ( NULL != xgemm_desc ) {
        k_sparse1 = libxsmm_create_dcsr_reg(xgemm_desc, a_csr_rowptr, a_csr_colidx, a_csr_values);
      }
    }
#if 0
    /* If that worked try to JIT a second (wider) sparse kernel */
    if ( NULL != k_sparse1 && N_sparse2 <= N ) {
      xgemm_desc = libxsmm_dgemm_descriptor_init(&xgemm_blob, M, N_sparse2, K,
                                                 0, ldb, ldc, one, beta, flags, prefetch);
      if ( NULL != xgemm_desc ) {
        k_sparse2 = libxsmm_create_dcsr_reg(xgemm_desc, a_csr_rowptr, a_csr_colidx, a_csr_values);
      }
    }
#endif
  }
  LIBXSMM_HANDLE_ERROR_OFF_END();

  /* Free CSR */
  free( a_csr_values );
  free( a_csr_rowptr );
  free( a_csr_colidx );

  /* Also generate a dense kernel */
  if ( NULL != aa_dense ) {
    k_dense = libxsmm_dmmdispatch(N_dense, M, K, &ldb, &K, &ldc,
      &one, &beta, &flags, NULL/*auto-prefetch*/);
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
  nkerns = !!k_dense + !!k_sparse1 + !!k_sparse2;

  /* We have at least one kernel */
  if (0 < nkerns) {
    libxsmm_timer_tickint t;
    double *B = NULL, *C = NULL;
    double dt_dense = ( NULL != k_dense ) ? 1e5 : 1e6;
    double dt_sparse1 = ( NULL != k_sparse1 ) ? 1e5 : 1e6;
    double dt_sparse2 = ( NULL != k_sparse2 ) ? 1e5 : 1e6;
    void* fp;

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
    if ( NULL != k_dense && NULL != B && NULL != C ) {
      t = libxsmm_timer_tick();
      for ( i = 0; i < 250; ++i ) {
        for ( j = 0; j < N; j += N_dense ) {
          k_dense( B + j, aa_dense, C + j );
        }
      }
      dt_dense = libxsmm_timer_duration( t, libxsmm_timer_tick() );
    }

    /* Benchmark sparse (regular) */
    if ( NULL != k_sparse1 && NULL != B && NULL != C ) {
      t = libxsmm_timer_tick();
      for ( i = 0; i < 250; ++i ) {
        for ( j = 0; j < N; j += N_sparse1 ) {
          k_sparse1( a_sparse, B + j, C + j );
        }
      }
      dt_sparse1 = libxsmm_timer_duration( t, libxsmm_timer_tick() );
    }

    /* Benchmark sparse (wide) */
    if ( NULL != k_sparse2 && NULL != B && NULL != C ) {
      t = libxsmm_timer_tick();
      for ( i = 0; i < 250; ++i ) {
        for ( j = 0; j < N; j += N_sparse2 ) {
          k_sparse2( a_sparse, B + j, C + j );
        }
      }
      dt_sparse2 = libxsmm_timer_duration( t, libxsmm_timer_tick() );
    }

    /* Dense fastest */
    if ( dt_dense <= dt_sparse1 && dt_dense <= dt_sparse2 ) {
      assert(NULL != k_dense && NULL != aa_dense);
      new_handle->N_chunksize = N_dense;
      new_handle->kernel = k_dense;
      new_handle->a_dense = aa_dense;
      free(a_unique_values);
    } else {
      libxsmm_free( aa_dense );
    }

    /* Sparse (regular) fastest */
    if ( dt_sparse1 < dt_dense && dt_sparse1 <= dt_sparse2 ) {
      assert(NULL != k_sparse1);
      new_handle->N_chunksize = N_sparse1;
      new_handle->kernel = k_sparse1;
      if (unlimited_unique) {
        new_handle->a_packed = a_unique_values;
      }
    } else if ( NULL != k_sparse1 ) {
      LIBXSMM_ASSIGN127( &fp, &k_sparse1 );
      libxsmm_free( fp );
    }

    /* Sparse (wide) fastest */
    if ( dt_sparse2 < dt_dense && dt_sparse2 < dt_sparse1 ) {
      assert(NULL != k_sparse2);
      new_handle->N_chunksize = N_sparse2;
      new_handle->kernel = k_sparse2;
      if (unlimited_unique) {
        new_handle->a_packed = a_unique_values;
      }
    } else if ( NULL != k_sparse2 ) {
      LIBXSMM_ASSIGN127( &fp, &k_sparse2 );
      libxsmm_free( fp );
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
    free(a_unique_values);
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
  const float *const perm = internal_sfsspmdm_init(); /* shall be first */
  const float one = 1.0f;
  float* a_csr_values = NULL;
  unsigned int* a_csr_rowptr = NULL;
  unsigned int* a_csr_colidx = NULL;
  float* aa_dense = NULL;
  int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  const int prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  const libxsmm_gemm_descriptor* xgemm_desc;
  libxsmm_descriptor_blob xgemm_blob;
  libxsmm_sfsspmdm* new_handle = NULL;
  libxsmm_smmfunction k_sparse1 = NULL;
  libxsmm_smmfunction k_sparse2 = NULL;
  libxsmm_smmfunction k_dense = NULL;
  int i, j, n, nkerns, a_nnz = 0;
  const int vlen = libxsmm_cpuid_vlen32(libxsmm_target_archid);
  const int N_sparse1 = vlen;
  const int N_sparse2 = vlen * 2;
  const int N_dense = vlen;
  static int error_once = 0;

  /* some checks */
  if (0 != (N % vlen)
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
  a_csr_values = (float*)malloc((size_t)a_nnz * sizeof(float));
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
      xgemm_desc = libxsmm_sgemm_descriptor_init(&xgemm_blob, M, N_sparse1, K,
                                                 0, ldb, ldc, one, beta, flags, prefetch);
      if ( NULL != xgemm_desc ) {
        k_sparse1 = libxsmm_create_scsr_reg(xgemm_desc, a_csr_rowptr, a_csr_colidx, a_csr_values);
      }
    }
#if 0
    /* If that worked try to JIT a second (wider) sparse kernel */
    if ( NULL != k_sparse1 && N_sparse2 <= N ) {
      xgemm_desc = libxsmm_sgemm_descriptor_init(&xgemm_blob, M, N_sparse2, K,
                                                 0, ldb, ldc, one, beta, flags, prefetch);
      if ( NULL != xgemm_desc ) {
        k_sparse2 = libxsmm_create_scsr_reg(xgemm_desc, a_csr_rowptr, a_csr_colidx, a_csr_values);
      }
    }
#endif
  } LIBXSMM_HANDLE_ERROR_OFF_END();

  /* Free CSR */
  free( a_csr_values );
  free( a_csr_rowptr );
  free( a_csr_colidx );

  /* Also generate a dense kernel */
  if ( NULL != aa_dense ) {
    k_dense = libxsmm_smmdispatch(N_dense, M, K, &ldb, &K, &ldc,
      &one, &beta, &flags, NULL/*auto-prefetch*/);
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
  nkerns = !!k_dense + !!k_sparse1 + !!k_sparse2;

  /* We have at least one kernel */
  if (0 < nkerns) {
    libxsmm_timer_tickint t;
    float *B = NULL, *C = NULL;
    double dt_dense = ( NULL != k_dense ) ? 1e5 : 1e6;
    double dt_sparse1 = ( NULL != k_sparse1 ) ? 1e5 : 1e6;
    double dt_sparse2 = ( NULL != k_sparse2 ) ? 1e5 : 1e6;
    void* fp;

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
    if ( NULL != k_dense && NULL != B && NULL != C ) {
      t = libxsmm_timer_tick();
      for ( i = 0; i < 250; ++i ) {
        for ( j = 0; j < N; j += N_dense ) {
          k_dense( B + j, aa_dense, C + j );
        }
      }
      dt_dense = libxsmm_timer_duration( t, libxsmm_timer_tick() );
    }

    /* Benchmark sparse (regular) */
    if ( NULL != k_sparse1 && NULL != B && NULL != C ) {
      t = libxsmm_timer_tick();
      for ( i = 0; i < 250; ++i ) {
        for ( j = 0; j < N; j += N_sparse1 ) {
          k_sparse1( perm, B + j, C + j );
        }
      }
      dt_sparse1 = libxsmm_timer_duration( t, libxsmm_timer_tick() );
    }

    /* Benchmark sparse (wide) */
    if ( NULL != k_sparse2 && NULL != B && NULL != C ) {
      t = libxsmm_timer_tick();
      for ( i = 0; i < 250; ++i ) {
        for ( j = 0; j < N; j += N_sparse2 ) {
          k_sparse2( perm, B + j, C + j );
        }
      }
      dt_sparse2 = libxsmm_timer_duration( t, libxsmm_timer_tick() );
    }

    /* Dense fastest */
    if ( dt_dense <= dt_sparse1 && dt_dense <= dt_sparse2 ) {
      assert(NULL != k_dense && NULL != aa_dense);
      new_handle->N_chunksize = N_dense;
      new_handle->kernel = k_dense;
      new_handle->a_dense = aa_dense;
    } else {
      libxsmm_free( aa_dense );
    }

    /* Sparse (regular) fastest */
    if ( dt_sparse1 < dt_dense && dt_sparse1 <= dt_sparse2 ) {
      assert(NULL != k_sparse1);
      new_handle->N_chunksize = N_sparse1;
      new_handle->kernel = k_sparse1;
    } else if ( NULL != k_sparse1 ) {
      LIBXSMM_ASSIGN127( &fp, &k_sparse1 );
      libxsmm_free( fp );
    }

    /* Sparse (wide) fastest */
    if ( dt_sparse2 < dt_dense && dt_sparse2 < dt_sparse1 ) {
      assert(NULL != k_sparse2);
      new_handle->N_chunksize = N_sparse2;
      new_handle->kernel = k_sparse2;
    } else if ( NULL != k_sparse2 ) {
      LIBXSMM_ASSIGN127( &fp, &k_sparse2 );
      libxsmm_free( fp );
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
  assert( handle != NULL );

  if (handle->a_dense != NULL) {
    for ( i = 0; i < handle->N; i += handle->N_chunksize ) {
      handle->kernel( B+i, handle->a_dense, C+i );
    }
  } else if (handle->a_packed != NULL) {
    for ( i = 0; i < handle->N; i += handle->N_chunksize ) {
      handle->kernel( handle->a_packed, B+i, C+i );
    }
  } else {
    const double *const perm = internal_dfsspmdm_init();
    for ( i = 0; i < handle->N; i += handle->N_chunksize ) {
      handle->kernel( perm, B+i, C+i );
    }
  }
}


LIBXSMM_API void libxsmm_sfsspmdm_execute( const libxsmm_sfsspmdm* handle, const float* B, float* C )
{
  int i;
  assert( handle != NULL );

  if ( handle->a_dense == NULL ) {
    const float *const perm = internal_sfsspmdm_init();
    for ( i = 0; i < handle->N; i += handle->N_chunksize ) {
      handle->kernel( perm, B+i, C+i );
    }
  } else {
    for ( i = 0; i < handle->N; i += handle->N_chunksize ) {
      handle->kernel( B+i, handle->a_dense, C+i );
    }
  }
}


LIBXSMM_API void libxsmm_dfsspmdm_destroy( libxsmm_dfsspmdm* handle )
{
  assert( handle != NULL );

  if ( handle->a_dense != NULL ) {
    libxsmm_free(handle->a_dense);
  }
  else {
    /* deallocate code known to be not registered; no index attached
       do not use libxsmm_release_kernel here! We also need to work
       around pointer-to-function to pointer-to-object conversion */
    void* fp;
    LIBXSMM_ASSIGN127(&fp, &handle->kernel);
    libxsmm_free(fp);
  }

  free(handle);
}


LIBXSMM_API void libxsmm_sfsspmdm_destroy( libxsmm_sfsspmdm* handle )
{
  assert( handle != NULL );

  if ( handle->a_dense != NULL ) {
    libxsmm_free(handle->a_dense);
  }
  else {
    /* deallocate code known to be not registered; no index attached
       do not use libxsmm_release_kernel here! We also need to work
       around pointer-to-function to pointer-to-object conversion */
    void* fp;
    LIBXSMM_ASSIGN127(&fp, &handle->kernel);
    libxsmm_free(fp);
  }

  free(handle);
}

