/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Nadathur Satish (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_SPMDM_H
#define LIBXSMM_SPMDM_H

#include "libxsmm_typedefs.h"


typedef enum libxsmm_spmdm_datatype {
  LIBXSMM_SPMDM_DATATYPE_F32,
  LIBXSMM_SPMDM_DATATYPE_BFLOAT16
} libxsmm_spmdm_datatype;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_spmdm_handle {
  /* The following are the matrix multiply dimensions: A (sparse): m X k, B (dense): k X n, Output C (dense): m X n */
  int m;
  int n;
  int k;
  /* The block sizes for A, B and C. */
  /* Here we fix A to be divided into 128 X 128 blocks, B/C to be 128 X 48 for HSW/BDW and 128 X 96 for SKX */
  int bm;
  int bn;
  int bk;
  /* The number of blocks for the m, n and k dimensions */
  int mb;
  int nb;
  int kb;
  libxsmm_spmdm_datatype datatype;
  char* base_ptr_scratch_A;
  char* base_ptr_scratch_B_scratch_C;
  int memory_for_scratch_per_thread;
} libxsmm_spmdm_handle;

/**
 * This stores a single sparse splice (or block) of sparse matrix A using a CSR representation (rowidx, colidx, and values
 * Each splice corresponds to a bm X bk region of A, and stores local indexes
 */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_CSR_sparseslice {
  /* Since bm and bk are assumed to be <=256, a 16-bit integer is enough to store the local rowidx, colidx */
  uint16_t* rowidx;
  uint16_t* colidx;
  float*    values;
} libxsmm_CSR_sparseslice;


LIBXSMM_API void libxsmm_spmdm_init(
  int M, int N, int K,
  int max_threads,
  libxsmm_spmdm_handle* handle,
  libxsmm_CSR_sparseslice** libxsmm_output_csr);

LIBXSMM_API void libxsmm_spmdm_destroy(
  libxsmm_spmdm_handle* handle);

LIBXSMM_API int libxsmm_spmdm_get_num_createSparseSlice_blocks(
  const libxsmm_spmdm_handle* handle);

LIBXSMM_API int libxsmm_spmdm_get_num_compute_blocks(
  const libxsmm_spmdm_handle* handle);

/** This converts a dense representation of the sparse matrix to 2D array of sparse slices. */
LIBXSMM_API void libxsmm_spmdm_createSparseSlice_fp32_thread(
  const libxsmm_spmdm_handle* handle,
  char transa,
  const float* a,
  libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
  int block_id,
  int tid, int nthreads);

LIBXSMM_API void libxsmm_spmdm_createSparseSlice_bfloat16_thread(
  const libxsmm_spmdm_handle* handle,
  char transa,
  const libxsmm_bfloat16* a,
  libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
  int block_id,
  int tid, int nthreads);

/** NOTE: This code currently ignores alpha input to the matrix multiply */
LIBXSMM_API void libxsmm_spmdm_compute_fp32_thread(
  const libxsmm_spmdm_handle* handle,
  char transa,
  char transb,
  const float* alpha,
  libxsmm_CSR_sparseslice* a_sparse,
  const float* b,
  char transc,
  const float* beta,
  float* c,
  int block_id,
  int tid, int nthreads);

/** NOTE: This code currently ignores alpha input to the matrix multiply */
LIBXSMM_API void libxsmm_spmdm_compute_bfloat16_thread(
  const libxsmm_spmdm_handle* handle,
  char transa,
  char transb,
  const libxsmm_bfloat16* alpha,
  libxsmm_CSR_sparseslice* a_sparse,
  const libxsmm_bfloat16* b,
  char transc,
  const libxsmm_bfloat16* beta,
  float* c,
  int block_id,
  int tid, int nthreads);

#endif /*LIBXSMM_SPMDM_H*/

