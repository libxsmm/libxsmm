/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm_fsspmdm.h>
#include <libxsmm.h>

#include "libxsmm_main.h"
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#if !defined(NDEBUG)
# include <stdio.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API libxsmm_dfsspmdm* libxsmm_dfsspmdm_create(
  libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint K,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  const double alpha, const double beta,
  const double* a_dense)
{
  double* a_csr_values = 0;
  unsigned int* a_csr_rowptr = 0;
  unsigned int* a_csr_colidx = 0;
  const int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  const libxsmm_gemm_prefetch_type prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  const libxsmm_gemm_descriptor* xgemm_desc;
  libxsmm_descriptor_blob xgemm_blob;
  libxsmm_dfsspmdm* new_handle = 0;
  int i, j, a_nnz;

  /* some checks... */
  assert(N % 16 == 0);
  assert(N >= 16);
  assert(LIBXSMM_FEQ(alpha, 1.0));
  assert(LIBXSMM_FEQ(beta, 1.0) || LIBXSMM_FEQ(beta, 0.0));
  assert(K <= lda);
  assert(N <= ldc);
  assert(N <= ldb);

  /* allocate handle */
  new_handle = (libxsmm_dfsspmdm*)malloc(sizeof(libxsmm_dfsspmdm));
  if (0 == new_handle) return 0;

  /* initialize the handle */
  memset((void*)new_handle, 0, sizeof(libxsmm_dfsspmdm));
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

  if (0 != a_csr_values && 0 != a_csr_rowptr && 0 != a_csr_colidx) {
    int n = 0;
    /* populate CSR structure */
    for (i = 0; i < M; i++) {
      a_csr_rowptr[i] = n;
      for (j = 0; j < K; j++) {
        if (LIBXSMM_NEQ(a_dense[(i*lda) + j], 0.0)) {
          a_csr_values[n] = a_dense[(i*lda) + j];
          a_csr_colidx[n] = j;
          n++;
        }
      }
    }
    a_csr_rowptr[M] = a_nnz;

    /* attempt to JIT a sparse_reg */
    new_handle->N_chunksize = 8;

    xgemm_desc = libxsmm_dgemm_descriptor_init(&xgemm_blob, M, new_handle->N_chunksize, K,
      0, ldb, ldc, alpha, beta, flags, prefetch);

    if (0 != xgemm_desc) {
      new_handle->kernel = libxsmm_create_dcsr_reg(xgemm_desc, a_csr_rowptr, a_csr_colidx, a_csr_values);
    }
  }

  /* continue with sparse A */
  if (new_handle->kernel != 0) {
  /* nothing to do */
  /* attempt to JIT dense kernel as sparse_reg failed */
  } else {
    new_handle->N_chunksize = 16;
    new_handle->kernel = libxsmm_dmmdispatch(new_handle->N_chunksize, M, K, &ldb, &K, &ldc, &alpha, &beta, 0, (const int*)LIBXSMM_GEMM_PREFETCH_NONE);
    /* copy A over */
    new_handle->a_dense = (double*)libxsmm_aligned_malloc((size_t)M * K * sizeof(double), 64);
    for ( i = 0; i < M; ++i ) {
      for ( j = 0; j < K; ++j ) {
        new_handle->a_dense[(i*K)+j] = a_dense[(i*lda)+j];
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
  const float alpha, const float beta,
  const float* a_dense)
{
  float* a_csr_values = 0;
  unsigned int* a_csr_rowptr = 0;
  unsigned int* a_csr_colidx = 0;
  const int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  const libxsmm_gemm_prefetch_type prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  const libxsmm_gemm_descriptor* xgemm_desc;
  libxsmm_descriptor_blob xgemm_blob;
  libxsmm_sfsspmdm* new_handle = 0;
  int i, j, a_nnz;

  /* some checks... */
  assert(N % 16 == 0);
  assert(N >= 16);
  assert(LIBXSMM_FEQ(alpha, 1.0f));
  assert(LIBXSMM_FEQ(beta, 1.0f) || LIBXSMM_FEQ(beta, 0.0f));
  assert(K <= lda);
  assert(N <= ldc);
  assert(N <= ldb);

  /* allocate handle */
  new_handle = (libxsmm_sfsspmdm*)malloc(sizeof(libxsmm_sfsspmdm));
  if (0 == new_handle) return 0;

  /* initialize the handle */
  memset((void*)new_handle, 0, sizeof(libxsmm_sfsspmdm));
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

  if (0 != a_csr_values && 0 != a_csr_rowptr && 0 != a_csr_colidx) {
    int n = 0;
    /* populate CSR structure */
    for (i = 0; i < M; i++) {
      a_csr_rowptr[i] = n;
      for (j = 0; j < K; j++) {
        if (LIBXSMM_NEQ(a_dense[(i*lda) + j], 0.0f)) {
          a_csr_values[n] = a_dense[(i*lda) + j];
          a_csr_colidx[n] = j;
          n++;
        }
      }
    }
    a_csr_rowptr[M] = a_nnz;

    /* attempt to JIT a sparse_reg */
    new_handle->N_chunksize = 16;

    xgemm_desc = libxsmm_sgemm_descriptor_init(&xgemm_blob, M, new_handle->N_chunksize, K,
      0, ldb, ldc, alpha, beta, flags, prefetch);

    if (0 != xgemm_desc) {
      new_handle->kernel = libxsmm_create_scsr_reg(xgemm_desc, a_csr_rowptr, a_csr_colidx, a_csr_values);
    }
  }

  /* continue with sparse A */
  if (new_handle->kernel != 0) {
  /* nothing to do */
  /* attempt to JIT dense kernel as sparse_reg failed */
  } else {
    new_handle->N_chunksize = 16;
    new_handle->kernel = libxsmm_smmdispatch(new_handle->N_chunksize, M, K, &ldb, &K, &ldc, &alpha, &beta, 0, (const int*)LIBXSMM_GEMM_PREFETCH_NONE);
    /* copy A over */
    new_handle->a_dense = (float*)libxsmm_aligned_malloc((size_t)M * K * sizeof(float), 64);
    for ( i = 0; i < M; ++i ) {
      for ( j = 0; j < K; ++j ) {
        new_handle->a_dense[(i*K)+j] = a_dense[(i*lda)+j];
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
      handle->kernel( handle->a_dense, B+i, C+i );
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
      handle->kernel( handle->a_dense, B+i, C+i );
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
    memcpy(&fp, &(handle->kernel), sizeof(libxsmm_dmmfunction));
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
    memcpy(&fp, &(handle->kernel), sizeof(libxsmm_smmfunction));
    libxsmm_free(fp);
  }

  free(handle);
}


