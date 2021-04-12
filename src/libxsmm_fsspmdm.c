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


LIBXSMM_API libxsmm_dfsspmdm* libxsmm_dfsspmdm_create(
  libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint K,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  const double alpha, const double beta, libxsmm_blasint c_is_nt,
  const double* a_dense)
{
  double one = 1.0;
  double* a_csr_values = 0;
  unsigned int* a_csr_rowptr = 0;
  unsigned int* a_csr_colidx = 0;
  int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  const libxsmm_gemm_prefetch_type prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  const libxsmm_gemm_descriptor* xgemm_desc;
  libxsmm_descriptor_blob xgemm_blob;
  libxsmm_dfsspmdm* new_handle = 0;
  int i, j, a_nnz;

  /* some checks... */
  assert(N % 8 == 0);
  assert(N >= 8);
  assert(LIBXSMM_FEQ(beta, 1.0) || LIBXSMM_FEQ(beta, 0.0));
  assert(K <= lda);
  assert(N <= ldc);
  assert(N <= ldb);

  /* allocate handle */
  new_handle = (libxsmm_dfsspmdm*)malloc(sizeof(libxsmm_dfsspmdm));
  if (0 == new_handle) return 0;

  /* initialize the handle */
  LIBXSMM_MEMZERO127(new_handle);
  /* TODO: in case of ILP64, check value ranges */
  new_handle->N = (int)N;
  new_handle->M = (int)M;
  new_handle->K = (int)K;
  new_handle->ldb = (int)ldb;
  new_handle->ldc = (int)ldc;

  /* get number of non-zeros */
  a_nnz = 0;
  for (i = 0; i < M; ++i) {
    for (j = 0; j < K; j++) {
      if (LIBXSMM_NEQ(a_dense[(i*lda) + j], 0.0)) {
        a_nnz++;
      }
    }
  }

  if (0 < a_nnz) {
    /* allocate CSR structure */
    a_csr_values = (double*)malloc((size_t)a_nnz * sizeof(double));
    a_csr_rowptr = (unsigned int*)malloc(((size_t)M + 1) * sizeof(unsigned int));
    a_csr_colidx = (unsigned int*)malloc((size_t)a_nnz * sizeof(unsigned int));
  }

  /* update flags */
  if ( (beta == 0.0f) && (c_is_nt != 0) ) {
    flags |= LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT;
  }

  if (0 != a_csr_values && 0 != a_csr_rowptr && 0 != a_csr_colidx) {
    int n = 0;
    /* populate CSR structure */
    for (i = 0; i < M; i++) {
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

    /* attempt to JIT a sparse_reg */
    new_handle->N_chunksize = libxsmm_cpuid_vlen32(libxsmm_cpuid()) / 2;

    xgemm_desc = libxsmm_dgemm_descriptor_init(&xgemm_blob, M, new_handle->N_chunksize, K,
      0, ldb, ldc, one, beta, flags, prefetch);

    if (0 != xgemm_desc) {
      new_handle->kernel = libxsmm_create_dcsr_reg(xgemm_desc, a_csr_rowptr, a_csr_colidx, a_csr_values);
    }
  }

  /* continue with sparse A */
  if (new_handle->kernel != 0) {
    /* allocate 8 * 512-bit permute operands if not stored in registers */
    new_handle->permute_operands = (unsigned int*)libxsmm_aligned_malloc(8*16*sizeof(unsigned int), 64);
    /* store permute operands */
    for (i = 0; i < 8; i++) {
      j = 0;
      /* repeat pattern to select 64-bits using vpermd */
      while (j < 16) {
        new_handle->permute_operands[i*16+(j)] = i*2;
        j++;
        new_handle->permute_operands[i*16+(j)] = i*2 + 1;
        j++;
      }
    }
  /* attempt to JIT dense kernel as sparse_reg failed */
  } else {
    new_handle->N_chunksize = 8;
    new_handle->kernel = libxsmm_dmmdispatch(new_handle->N_chunksize, M, K, &ldb, &K, &ldc, &one, &beta, &flags, (const int*)LIBXSMM_GEMM_PREFETCH_NONE);
    /* copy A over */
    new_handle->a_dense = (double*)libxsmm_aligned_malloc((size_t)M * (size_t)K * sizeof(double), 64);
    for ( i = 0; i < M; ++i ) {
      for ( j = 0; j < K; ++j ) {
        new_handle->a_dense[(i*K)+j] = alpha*a_dense[(i*lda)+j];
      }
    }
  }

  /* free CSR */
  free( a_csr_values );
  free( a_csr_rowptr );
  free( a_csr_colidx );

  return new_handle;
}


LIBXSMM_API libxsmm_sfsspmdm* libxsmm_sfsspmdm_create(
  libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint K,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  const float alpha, const float beta, libxsmm_blasint c_is_nt,
  const float* a_dense)
{
  float one = 1.0f;
  float* a_csr_values = 0;
  unsigned int* a_csr_rowptr = 0;
  unsigned int* a_csr_colidx = 0;
  int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  const libxsmm_gemm_prefetch_type prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  const libxsmm_gemm_descriptor* xgemm_desc;
  libxsmm_descriptor_blob xgemm_blob;
  libxsmm_sfsspmdm* new_handle = 0;
  int i, j, a_nnz;

  /* some checks... */
  assert(N % 16 == 0);
  assert(N >= 16);
  assert(LIBXSMM_FEQ(beta, 1.0f) || LIBXSMM_FEQ(beta, 0.0f));
  assert(K <= lda);
  assert(N <= ldc);
  assert(N <= ldb);

  /* allocate handle */
  new_handle = (libxsmm_sfsspmdm*)malloc(sizeof(libxsmm_sfsspmdm));
  if (0 == new_handle) return 0;

  /* initialize the handle */
  LIBXSMM_MEMZERO127(new_handle);
  /* TODO: in case of ILP64, check value ranges */
  new_handle->N = (int)N;
  new_handle->M = (int)M;
  new_handle->K = (int)K;
  new_handle->ldb = (int)ldb;
  new_handle->ldc = (int)ldc;

  /* get number of non-zeros */
  a_nnz = 0;
  for (i = 0; i < M; ++i) {
    for (j = 0; j < K; j++) {
      if (LIBXSMM_NEQ(a_dense[(i*lda) + j], 0.0f)) {
        a_nnz++;
      }
    }
  }

  if (0 < a_nnz) {
    /* allocate CSR structure */
    a_csr_values = (float*)malloc((size_t)a_nnz * sizeof(float));
    a_csr_rowptr = (unsigned int*)malloc(((size_t)M + 1) * sizeof(unsigned int));
    a_csr_colidx = (unsigned int*)malloc((size_t)a_nnz * sizeof(unsigned int));
  }

  /* update flags */
  if ( (beta == 0.0f) && (c_is_nt != 0) ) {
    flags |= LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT;
  }

  if (0 != a_csr_values && 0 != a_csr_rowptr && 0 != a_csr_colidx) {
    int n = 0;
    /* populate CSR structure */
    for (i = 0; i < M; i++) {
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

    /* attempt to JIT a sparse_reg */
    new_handle->N_chunksize = libxsmm_cpuid_vlen32(libxsmm_cpuid());

    xgemm_desc = libxsmm_sgemm_descriptor_init(&xgemm_blob, M, new_handle->N_chunksize, K,
      0, ldb, ldc, one, beta, flags, prefetch);

    if (0 != xgemm_desc) {
      new_handle->kernel = libxsmm_create_scsr_reg(xgemm_desc, a_csr_rowptr, a_csr_colidx, a_csr_values);
    }
  }

  /* continue with sparse A */
  if (new_handle->kernel != 0) {
    /* allocate 16 * 512-bit permute operands if not stored in registers */
    new_handle->permute_operands = (unsigned int*)libxsmm_aligned_malloc(16*16*sizeof(unsigned int), 64);
    /* store permute operands */
    for (i = 0; i < 16; i++) {
      j = 0;
      /* repeat pattern to select 32-bits using vpermd */
      while (j < 16) {
        new_handle->permute_operands[i*16+j] = i;
        j++;
      }
    }
  /* attempt to JIT dense kernel as sparse_reg failed */
  } else {
    new_handle->N_chunksize = 16;
    new_handle->kernel = libxsmm_smmdispatch(new_handle->N_chunksize, M, K, &ldb, &K, &ldc, &one, &beta, &flags, (const int*)LIBXSMM_GEMM_PREFETCH_NONE);
    /* copy A over */
    new_handle->a_dense = (float*)libxsmm_aligned_malloc((size_t)M * (size_t)K * sizeof(float), 64);
    for ( i = 0; i < M; ++i ) {
      for ( j = 0; j < K; ++j ) {
        new_handle->a_dense[(i*K)+j] = alpha*a_dense[(i*lda)+j];
      }
    }
  }

  /* free CSR */
  free( a_csr_values );
  free( a_csr_rowptr );
  free( a_csr_colidx );

  return new_handle;
}


LIBXSMM_API void libxsmm_dfsspmdm_execute( const libxsmm_dfsspmdm* handle, const double* B, double* C )
{
  int i;
  assert( handle != 0 );

  if ( handle->a_dense == 0 ) {
    for ( i = 0; i < handle->N; i+=handle->N_chunksize ) {
      handle->kernel( (double*)handle->permute_operands, B+i, C+i );
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
  assert( handle != 0 );

  if ( handle->a_dense == 0 ) {
    for ( i = 0; i < handle->N; i+=handle->N_chunksize ) {
      handle->kernel( (float*)handle->permute_operands, B+i, C+i );
    }
  } else {
    for ( i = 0; i < handle->N; i+=handle->N_chunksize ) {
      handle->kernel( B+i, handle->a_dense, C+i );
    }
  }
}


LIBXSMM_API void libxsmm_dfsspmdm_destroy( libxsmm_dfsspmdm* handle )
{
  assert( handle != 0 );

  if (handle->a_dense != 0) {
    libxsmm_free(handle->a_dense);
  } else {
    /* deallocate code known to be not registered; no index attached
       do not use libxsmm_release_kernel here! We also need to work
       around pointer-to-function to pointer-to-object conversion */
    void* fp;
    if (handle->permute_operands != 0) {
      libxsmm_free(handle->permute_operands);
    }
    LIBXSMM_ASSIGN127(&fp, &handle->kernel);
    libxsmm_free(fp);
  }

  free(handle);
}


LIBXSMM_API void libxsmm_sfsspmdm_destroy( libxsmm_sfsspmdm* handle )
{
  assert( handle != 0 );

  if (handle->a_dense != 0) {
    libxsmm_free(handle->a_dense);
  } else {
    /* deallocate code known to be not registered; no index attached
       do not use libxsmm_release_kernel here! We also need to work
       around pointer-to-function to pointer-to-object conversion */
    void* fp;
    if (handle->permute_operands != 0) {
      libxsmm_free(handle->permute_operands);
    }
    LIBXSMM_ASSIGN127(&fp, &handle->kernel);
    libxsmm_free(fp);
  }

  free(handle);
}


