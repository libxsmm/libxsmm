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
#include "libxsmm_main.h"


/* Double precision AVX-512 lane broadcasts */
LIBXSMM_APIVAR_DEFINE(const unsigned int dperm[]) = {
  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
  2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
  4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
  6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
  8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
  10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
  12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
  14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15
};

/* Single precision AVX-512 lane broadcasts */
LIBXSMM_APIVAR_DEFINE(const unsigned int sperm[]) = {
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


LIBXSMM_API libxsmm_dfsspmdm* libxsmm_dfsspmdm_create(
  libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint K,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  const double alpha, const double beta, libxsmm_blasint c_is_nt,
  const double* a_dense)
{
  double one = 1.0;
  double* a_csr_values = NULL;
  unsigned int* a_csr_rowptr = NULL;
  unsigned int* a_csr_colidx = NULL;
  double* aa_dense = NULL;
  int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  const libxsmm_gemm_prefetch_type prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  const libxsmm_gemm_descriptor* xgemm_desc;
  libxsmm_descriptor_blob xgemm_blob;
  libxsmm_dfsspmdm* new_handle = NULL;
  libxsmm_dmmfunction k_sparse = NULL;
  libxsmm_dmmfunction k_dense = NULL;
  int i, j, n, a_nnz, N_sparse, N_dense;

  /* some checks... */
  assert(N % 8 == 0);
  assert(N >= 8);
  assert(LIBXSMM_FEQ(beta, 1.0) || LIBXSMM_FEQ(beta, 0.0));
  assert(K <= lda);
  assert(N <= ldc);
  assert(N <= ldb);

  /* Get the number of non-zeros */
  a_nnz = 0;
  for (i = 0; i < M; i++) {
    for (j = 0; j < K; j++) {
      if (LIBXSMM_NEQ(a_dense[(i*lda) + j], 0.0)) {
        a_nnz++;
      }
    }
  }

  /* Null matrix */
  if ( 0 == a_nnz ) return NULL;

  /* Allocate handle */
  new_handle = (libxsmm_dfsspmdm*)malloc(sizeof(libxsmm_dfsspmdm));
  if ( NULL == new_handle ) return NULL;

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

  /* Allocate dense storage */
  aa_dense = (double*)libxsmm_aligned_malloc((size_t)M * (size_t)K * sizeof(double), 64);

  if ( NULL == a_csr_values || NULL == a_csr_rowptr || NULL == a_csr_colidx || NULL == aa_dense ) {
    free( a_csr_values ); free( a_csr_rowptr ); free( a_csr_colidx );
    free( new_handle );
    libxsmm_free( aa_dense );
    return NULL;
  }

  /* Populate CSR structure */
  for (i = 0, n = 0; i < M; i++) {
    a_csr_rowptr[i] = n;
    for (j = 0; j < K; j++) {
      if (LIBXSMM_NEQ(a_dense[(i*lda) + j], 0.0)) {
        a_csr_values[n] = alpha*a_dense[(i*lda) + j];
        a_csr_colidx[n] = j;
        n++;
      }
    }
  }
  a_csr_rowptr[M] = a_nnz;

  /* Attempt to JIT a sparse kernel */
  N_sparse = libxsmm_cpuid_vlen32(libxsmm_cpuid()) / 2;
  xgemm_desc = libxsmm_dgemm_descriptor_init(&xgemm_blob, M, N_sparse, K,
                                              0, ldb, ldc, one, beta, flags, prefetch);
  if ( NULL != xgemm_desc ) {
    k_sparse = libxsmm_create_dcsr_reg(xgemm_desc, a_csr_rowptr, a_csr_colidx, a_csr_values);
  }

  /* Free CSR */
  free( a_csr_values );
  free( a_csr_rowptr );
  free( a_csr_colidx );

  /* Also generate a dense kernel */
  N_dense = 8;
  k_dense = libxsmm_dmmdispatch(N_dense, M, K, &ldb, &K, &ldc, &one, &beta, &flags, (const int*)LIBXSMM_GEMM_PREFETCH_NONE);

  if ( NULL != k_dense ) {
    /* copy A over */
    for ( i = 0; i < M; ++i ) {
      for ( j = 0; j < K; ++j ) {
        aa_dense[i*K + j] = alpha*a_dense[i*lda + j];
      }
    }
  }

  /* Sparse only */
  if ( NULL != k_sparse && NULL == k_dense ) {
    new_handle->N_chunksize = N_sparse;
    new_handle->kernel = k_sparse;
    libxsmm_free( aa_dense );
  }
  /* Dense only */
  else if ( NULL != k_dense && NULL == k_sparse ) {
    new_handle->N_chunksize = N_dense;
    new_handle->kernel = k_dense;
    new_handle->a_dense = aa_dense;
  }
  /* Both; benchmark */
  else if ( NULL != k_dense && NULL != k_sparse ) {
    int use_sparse = 1;
    libxsmm_timer_tickint t1, t2, t3;
    double* B = (double*)libxsmm_aligned_malloc((size_t)K * (size_t)ldb * sizeof(double), 64);
    double* C = (double*)libxsmm_aligned_malloc((size_t)M * (size_t)ldc * sizeof(double), 64);
    void* fp;

    if ( NULL != B && NULL != C ) {
      for ( i = 0; i < K; i++ ) {
        for ( j = 0; j < N; j++ ) {
          B[i*ldb + j] = 1;
        }
      }
      for ( i = 0; i < M; i++ ) {
        for ( j = 0; j < N; j++ ) {
          C[i*ldc + j] = 1;
        }
      }

      /* Benchmark sparse */
      t1 = libxsmm_timer_tick();
      for ( i = 0; i < 25; i++ ) {
        for ( j = 0; j < N; j += N_sparse ) {
          k_sparse( (const double*)dperm, B + j, C + j );
        }
      }
      t2 = libxsmm_timer_tick();

      /* Benchmark dense */
      for ( i = 0; i < 25; i++ ) {
        for ( j = 0; j < N; j += N_dense ) {
          k_dense( B + j, aa_dense, C + j );
        }
      }
      t3 = libxsmm_timer_tick();

      /* Compare */
      use_sparse = libxsmm_timer_duration( t1, t2 ) < libxsmm_timer_duration( t2, t3 );
    }

    /* Sparse faster */
    if ( use_sparse ) {
      new_handle->N_chunksize = N_sparse;
      new_handle->kernel = k_sparse;

      libxsmm_free( aa_dense );
    }
    /* Dense faster */
    else {
      new_handle->N_chunksize = N_dense;
      new_handle->kernel = k_dense;
      new_handle->a_dense = aa_dense;

      LIBXSMM_ASSIGN127( &fp, &k_sparse );
      libxsmm_free( fp );
    }

    libxsmm_free( B );
    libxsmm_free( C );
  }
  /* Neither */
  else {
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
  float one = 1.0f;
  float* a_csr_values = NULL;
  unsigned int* a_csr_rowptr = NULL;
  unsigned int* a_csr_colidx = NULL;
  float* aa_dense = NULL;
  int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  const libxsmm_gemm_prefetch_type prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  const libxsmm_gemm_descriptor* xgemm_desc;
  libxsmm_descriptor_blob xgemm_blob;
  libxsmm_sfsspmdm* new_handle = NULL;
  libxsmm_smmfunction k_sparse = NULL;
  libxsmm_smmfunction k_dense = NULL;
  int i, j, n, a_nnz, N_sparse, N_dense;

  /* some checks... */
  assert(N % 16 == 0);
  assert(N >= 16);
  assert(LIBXSMM_FEQ(beta, 1.0f) || LIBXSMM_FEQ(beta, 0.0f));
  assert(K <= lda);
  assert(N <= ldc);
  assert(N <= ldb);

  /* Get the number of non-zeros */
  a_nnz = 0;
  for (i = 0; i < M; i++) {
    for (j = 0; j < K; j++) {
      if (LIBXSMM_NEQ(a_dense[(i*lda) + j], 0.0)) {
        a_nnz++;
      }
    }
  }

  /* Null matrix */
  if ( 0 == a_nnz ) return 0;

  /* Allocate handle */
  new_handle = (libxsmm_sfsspmdm*)malloc(sizeof(libxsmm_sfsspmdm));
  if ( NULL == new_handle ) return NULL;

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

  /* Allocate dense storage */
  aa_dense = (float*)libxsmm_aligned_malloc((size_t)M * (size_t)K * sizeof(float), 64);

  if ( NULL == a_csr_values || NULL == a_csr_rowptr || NULL == a_csr_colidx || NULL == aa_dense ) {
    free( a_csr_values ); free( a_csr_rowptr ); free( a_csr_colidx );
    free( new_handle );
    libxsmm_free( aa_dense );
    return NULL;
  }

  /* Populate CSR structure */
  for (i = 0, n = 0; i < M; i++) {
    a_csr_rowptr[i] = n;
    for (j = 0; j < K; j++) {
      if (LIBXSMM_NEQ(a_dense[(i*lda) + j], 0.0f)) {
        a_csr_values[n] = alpha*a_dense[(i*lda) + j];
        a_csr_colidx[n] = j;
        n++;
      }
    }
  }
  a_csr_rowptr[M] = a_nnz;

  /* Attempt to JIT a sparse kernel */
  N_sparse = libxsmm_cpuid_vlen32(libxsmm_cpuid());
  xgemm_desc = libxsmm_sgemm_descriptor_init(&xgemm_blob, M, N_sparse, K,
                                             0, ldb, ldc, one, beta, flags, prefetch);
  if ( NULL != xgemm_desc ) {
    k_sparse = libxsmm_create_scsr_reg(xgemm_desc, a_csr_rowptr, a_csr_colidx, a_csr_values);
  }

  /* Free CSR */
  free( a_csr_values );
  free( a_csr_rowptr );
  free( a_csr_colidx );

  /* Also generate a dense kernel */
  N_dense = 16;
  k_dense = libxsmm_smmdispatch(N_dense, M, K, &ldb, &K, &ldc, &one, &beta, &flags, (const int*)LIBXSMM_GEMM_PREFETCH_NONE);

  if ( NULL != k_dense ) {
    /* copy A over */
    for ( i = 0; i < M; ++i ) {
      for ( j = 0; j < K; ++j ) {
        aa_dense[i*K + j] = alpha*a_dense[i*lda + j];
      }
    }
  }

  /* Sparse only */
  if ( NULL != k_sparse && NULL == k_dense ) {
    new_handle->N_chunksize = N_sparse;
    new_handle->kernel = k_sparse;
    libxsmm_free( aa_dense );
  }
  /* Dense only */
  else if ( NULL != k_dense && NULL == k_sparse ) {
    new_handle->N_chunksize = N_dense;
    new_handle->kernel = k_dense;
    new_handle->a_dense = aa_dense;
  }
  /* Both; benchmark */
  else if ( NULL != k_dense && NULL != k_sparse ) {
    int use_sparse = 1;
    libxsmm_timer_tickint t1, t2, t3;
    float* B = (float*)libxsmm_aligned_malloc((size_t)K * (size_t)ldb * sizeof(float), 64);
    float* C = (float*)libxsmm_aligned_malloc((size_t)M * (size_t)ldc * sizeof(float), 64);
    void* fp;

    if ( NULL != B && NULL != C ) {
      for ( i = 0; i < K; i++ ) {
        for ( j = 0; j < N; j++ ) {
          B[i*ldb + j] = 1;
        }
      }
      for ( i = 0; i < M; i++ ) {
        for ( j = 0; j < N; j++ ) {
          C[i*ldc + j] = 1;
        }
      }

      /* Benchmark sparse */
      t1 = libxsmm_timer_tick();
      for ( i = 0; i < 25; i++ ) {
        for ( j = 0; j < N; j += N_sparse ) {
          k_sparse( (const float*)sperm, B + j, C + j );
        }
      }
      t2 = libxsmm_timer_tick();

      /* Benchmark dense */
      for ( i = 0; i < 25; i++ ) {
        for ( j = 0; j < N; j += N_dense ) {
          k_dense( B + j, aa_dense, C + j );
        }
      }
      t3 = libxsmm_timer_tick();

      /* Compare */
      use_sparse = libxsmm_timer_duration( t1, t2 ) < libxsmm_timer_duration( t2, t3 );
    }

    /* Sparse faster */
    if ( use_sparse ) {
      new_handle->N_chunksize = N_sparse;
      new_handle->kernel = k_sparse;

      libxsmm_free( aa_dense );
    }
    /* Dense faster */
    else {
      new_handle->N_chunksize = N_dense;
      new_handle->kernel = k_dense;
      new_handle->a_dense = aa_dense;

      LIBXSMM_ASSIGN127( &fp, &k_sparse );
      libxsmm_free( fp );
    }

    libxsmm_free( B );
    libxsmm_free( C );
  }
  /* Neither */
  else {
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

  if ( handle->a_dense == NULL ) {
    for ( i = 0; i < handle->N; i+=handle->N_chunksize ) {
      handle->kernel( (const double*)dperm, B+i, C+i );
    }
  } else {
    for ( i = 0; i < handle->N; i+=handle->N_chunksize ) {
      handle->kernel( B+i, handle->a_dense, C+i );
    }
  }
}


LIBXSMM_API void libxsmm_sfsspmdm_execute( const libxsmm_sfsspmdm* handle, const float* B, float* C )
{
  int i;
  assert( handle != NULL );

  if ( handle->a_dense == NULL ) {
    for ( i = 0; i < handle->N; i+=handle->N_chunksize ) {
      handle->kernel( (const float*)sperm, B+i, C+i );
    }
  } else {
    for ( i = 0; i < handle->N; i+=handle->N_chunksize ) {
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


