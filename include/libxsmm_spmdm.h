/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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
/* Nadathur Satish (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_SPMDM_H
#define LIBXSMM_SPMDM_H

#include <libxsmm_macros.h>


typedef enum libxsmm_spmdm_datatype {
  LIBXSMM_SPMDM_DATATYPE_F32,
  LIBXSMM_SPMDM_DATATYPE_BFLOAT16
} libxsmm_spmdm_datatype;

typedef struct libxsmm_spmdm_handle {
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
} libxsmm_spmdm_handle;

/**
 * This stores a single sparse splice (or block) of sparse matrix A using a CSR representation (rowidx, colidx, and values
 * Each splice corresponds to a bm X bk region of A, and stores local indices
 */
typedef struct libxsmm_CSR_sparseslice {
  /* Since bm and bk are assumed to be <=256, a 16-bit integer is enough to store the local rowidx, colidx */
  uint16_t * rowidx;
  uint16_t * colidx;
  float*     values;
} libxsmm_CSR_sparseslice;


LIBXSMM_API void libxsmm_spmdm_init(
  int M, int N, int K,
  libxsmm_spmdm_handle* handle,
  libxsmm_CSR_sparseslice** libxsmm_output_csr);

LIBXSMM_API void libxsmm_spmdm_createSparseSlice_fp32_notrans_thread(
  const libxsmm_spmdm_handle* handle,
  char transA,
  const float * A,
  libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
  int mb, int kb,
  int tid, int nthreads);

LIBXSMM_API void libxsmm_spmdm_createSparseSlice_bfloat16_notrans_thread(
  const libxsmm_spmdm_handle* handle,
  char transA,
  const uint16_t * A,
  libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
  int mb, int kb,
  int tid, int nthreads);

LIBXSMM_API void libxsmm_spmdm_compute_fp32_thread(
  const libxsmm_spmdm_handle* handle,
  char transA,
  char transB,
  const float *alpha,
  libxsmm_CSR_sparseslice* A_sparse,
  const float *B,
  const float *beta,
  float* C,
  int mb, int num_m_blocks, int nb,
  int tid, int nthreads);

LIBXSMM_API void libxsmm_spmdm_compute_bfloat16_thread(
  const libxsmm_spmdm_handle* handle,
  char transA,
  char transB,
  const uint16_t *alpha,
  libxsmm_CSR_sparseslice* A_sparse,
  const uint16_t *B,
  const uint16_t *beta,
  uint16_t* C,
  int mb, int num_m_blocks, int nb,
  int tid, int nthreads);

#endif /*LIBXSMM_SPMDM_H*/
