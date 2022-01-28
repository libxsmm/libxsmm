/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke, Kunal Banerjee (Intel Corp.)
******************************************************************************/

/* helper variables */
libxsmm_blasint i, ik, in, ic, inik, BF, CB, CB_BLOCKS, KB_BLOCKS;
/* input sizes */
const libxsmm_blasint K =  handle->desc.K;
const libxsmm_blasint N =  handle->desc.N;
const libxsmm_blasint C =  handle->desc.C;
const libxsmm_blasint t =  handle->T;
const libxsmm_blasint bk = handle->bk;
const libxsmm_blasint bn = handle->bn;
const libxsmm_blasint bc = handle->bc;
/* define tensors */
element_input_type  *xt = (element_input_type* )handle->xt->data;
element_input_type  *hpD= (element_input_type* )handle->hp->data;
element_filter_type *wD = (element_filter_type*)handle->w->data;
element_filter_type *rD = (element_filter_type*)handle->r->data;
element_output_type *b  = (element_output_type*)handle->b->data;
element_output_type *ht = (element_output_type*)handle->ht->data;
element_output_type *zt = (element_output_type*)handle->internal_z;
/*libxsmm_blasint nBlocks = N/bn;*/
libxsmm_blasint cBlocks = C/bc;
libxsmm_blasint kBlocks = K/bk;
unsigned long long blocks;
LIBXSMM_VLA_DECL(3, element_input_type,  x, xt, N, C);
LIBXSMM_VLA_DECL(2, element_input_type,  hp, hpD, K);
LIBXSMM_VLA_DECL(4, element_filter_type, w, wD, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, r, rD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(3, element_output_type, h, ht, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, z, zt, N, K);
int prefetch_mode = LIBXSMM_GEMM_PREFETCH_NONE/*LIBXSMM_GEMM_PREFETCH_AL1_BL1*/;
/* define gemm kernels */
const libxsmm_smmfunction_reducebatch_addr batchreduce_kernela = libxsmm_smmdispatch_reducebatch_addr( bk, bn, bc, &bk, &C, &K, NULL, NULL, NULL, &prefetch_mode );
const libxsmm_smmfunction_reducebatch_addr batchreduce_kernelb = libxsmm_smmdispatch_reducebatch_addr( bk, bn, bk, &bk, &K, &K, NULL, NULL, NULL, &prefetch_mode );

/* Auxiliary arrays for batch-reduce gemms */
const element_input_type *A_array[1024];
const element_input_type *B_array[1024];
const element_input_type *A_array2[1024];
const element_input_type *B_array2[1024];

/* computing first logical thread */
const libxsmm_blasint ltid = (libxsmm_blasint)tid - (libxsmm_blasint)start_thread;
/* number of tasks that could be run in parallel */
const libxsmm_blasint work = (N/bn) * (K/bk);
/* compute chunk size */
const libxsmm_blasint chunksize = (work % (libxsmm_blasint)handle->desc.threads == 0) ? (work / (libxsmm_blasint)handle->desc.threads) : ((work / (libxsmm_blasint)handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* Blocking reduction domain if it is too large */
BF = 1;
if (C >= 2048 && K >= 2048 && C%2 == 0 && K%2 == 0) {
  BF = 2;
}
CB_BLOCKS = cBlocks/BF;
KB_BLOCKS = kBlocks/BF;
assert(CB_BLOCKS <= 1024);
assert(KB_BLOCKS <= 1024);

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, (int)ltid);

/* All data is in column-major format */
for (i = 0; i < t; ++i) {
  /* let's run the cell in blocks for good locality */
  for (CB = 0; CB < BF; CB++) {
    for (inik = thr_begin; inik < thr_end; ++inik ) {
      if (C >= 2048 && K >= 2048) {
        in = inik % (N/bn);
        ik = inik / (N/bn);
      } else {
        in = inik / (K/bk);
        ik = inik % (K/bk);
      }

      /* z = per_col(b) */
      if (0 == CB) {
        libxsmm_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, i, in*bn, ik*bk, N, K), &b[ik*bk] );
      }

      /* z += W.x */
      /* Prepare arrays for the call */
      for (ic = 0; ic < CB_BLOCKS; ic++) {
        /* this is a small matmul */
        A_array[ic] = &LIBXSMM_VLA_ACCESS(4, w, ik, ic + CB*CB_BLOCKS, 0, 0, cBlocks, bc, bk);
        B_array[ic] = &LIBXSMM_VLA_ACCESS(3, x, i, in*bn, (ic + CB*CB_BLOCKS)*bc, N, C);
      }
      /* Reduce batch gemm call  */
      blocks = CB_BLOCKS;
      batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, z, i, in*bn, ik*bk, N, K), &blocks);

      /* z += U.h */
      if (0 == i) {
        /* Prepare arrays for the call */
        for (ic = 0; ic < KB_BLOCKS; ic++) {
          A_array2[ic] = &LIBXSMM_VLA_ACCESS(4, r, ik, ic + CB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
          B_array2[ic] = &LIBXSMM_VLA_ACCESS(2, hp, in*bn, (ic + CB*KB_BLOCKS)*bk, K);
        }
        /* Reduce batch gemm call  */
        blocks = KB_BLOCKS;
        batchreduce_kernelb(A_array2, B_array2, &LIBXSMM_VLA_ACCESS(3, z, i, in*bn, ik*bk, N, K), &blocks);
      } else {
        /* Prepare arrays for the call */
        for (ic = 0; ic < KB_BLOCKS; ic++) {
          A_array2[ic] = &LIBXSMM_VLA_ACCESS(4, r, ik, ic + CB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
          B_array2[ic] = &LIBXSMM_VLA_ACCESS(3, h, i-1, in*bn, (ic + CB*KB_BLOCKS)*bk, N, K);
        }
        /* Reduce batch gemm call  */
        blocks = KB_BLOCKS;
        batchreduce_kernelb(A_array2, B_array2, &LIBXSMM_VLA_ACCESS(3, z, i, in*bn, ik*bk, N, K), &blocks);
      }
#if defined(LIBXSMM_DNN_RNN_RELU_FWD)
      libxsmm_internal_matrix_relu_ld(    bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, i, in*bn, ik*bk, N, K), &LIBXSMM_VLA_ACCESS(3, h, i, in*bn, ik*bk, N, K) );
#endif
#if defined(LIBXSMM_DNN_RNN_SIGMOID_FWD)
      libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, i, in*bn, ik*bk, N, K), &LIBXSMM_VLA_ACCESS(3, h, i, in*bn, ik*bk, N, K) );
#endif
#if defined(LIBXSMM_DNN_RNN_TANH_FWD)
      libxsmm_internal_matrix_tanh_ld(    bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, i, in*bn, ik*bk, N, K), &LIBXSMM_VLA_ACCESS(3, h, i, in*bn, ik*bk, N, K) );
#endif
    }
  }
  libxsmm_barrier_wait(handle->barrier, (int)ltid);
}

