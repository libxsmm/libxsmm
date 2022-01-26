/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Kunal Banerjee (Intel Corp.)
******************************************************************************/

/* helper variables */
libxsmm_blasint i, ik, in, ic, inik;
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
LIBXSMM_VLA_DECL(3, element_input_type,  x, xt, N, C);
LIBXSMM_VLA_DECL(2, element_input_type,  hp, hpD, K);
LIBXSMM_VLA_DECL(2, element_filter_type, w, wD, K);
LIBXSMM_VLA_DECL(2, element_filter_type, r, rD, K);
LIBXSMM_VLA_DECL(3, element_output_type, h, ht, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, z, zt, N, K);
/* define gemm kernels */
libxsmm_smmfunction gemmkernela = libxsmm_smmdispatch( bk, bn, bc, &K, &C, &K, NULL, NULL, NULL, NULL );
libxsmm_smmfunction gemmkernelb = libxsmm_smmdispatch( bk, bn, bk, &K, &K, &K, NULL, NULL, NULL, NULL );
/* parallelize over C-blocks */
/* computing first logical thread */
const libxsmm_blasint ltid = (libxsmm_blasint)tid - (libxsmm_blasint)start_thread;
/* number of tasks that could be run in parallel */
const libxsmm_blasint work = (N/bn) * (K/bk);
/* compute chunk size */
const libxsmm_blasint chunksize = (work % (libxsmm_blasint)handle->desc.threads == 0) ? (work / (libxsmm_blasint)handle->desc.threads) : ((work / (libxsmm_blasint)handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, (int)ltid);

/* All data is in column-major format */
for (i = 0; i < t; ++i) {
  /* let's run the cell in blocks for good locality */
  for (inik = thr_begin; inik < thr_end; ++inik ) {
    in = (inik / (K/bk))*bn;
    ik = (inik % (K/bk))*bk;

    /* z = per_col(b) */
    libxsmm_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, i, in, ik, N, K), &b[ik] );

    /* z += W.x */
    for (ic = 0; ic < C; ic += bc) {
      /* this is a small matmul */
      gemmkernela( &LIBXSMM_VLA_ACCESS(2, w, ic, ik, K), &LIBXSMM_VLA_ACCESS(3, x, i, in, ic, N, C), &LIBXSMM_VLA_ACCESS(3, z, i, in, ik, N, K) );
    }
    /* z += U.h */
    if (0 == i) {
      for (ic = 0; ic < K; ic += bk) {
        /* this is a small matmul */
        gemmkernelb( &LIBXSMM_VLA_ACCESS(2, r, ic, ik, K), &LIBXSMM_VLA_ACCESS(2, hp, in, ic, K), &LIBXSMM_VLA_ACCESS(3, z, i, in, ik, N, K) );
      }
    } else {
      for (ic = 0; ic < K; ic += bk) {
        /* this is a small matmul */
        gemmkernelb( &LIBXSMM_VLA_ACCESS(2, r, ic, ik, K), &LIBXSMM_VLA_ACCESS(3, h, i-1, in, ic, N, K), &LIBXSMM_VLA_ACCESS(3, z, i, in, ik, N, K) );
      }
    }
#if defined(LIBXSMM_DNN_RNN_RELU_FWD)
    libxsmm_internal_matrix_relu_ld(    bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, i, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, h, i, in, ik, N, K) );
#endif
#if defined(LIBXSMM_DNN_RNN_SIGMOID_FWD)
    libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, i, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, h, i, in, ik, N, K) );
#endif
#if defined(LIBXSMM_DNN_RNN_TANH_FWD)
    libxsmm_internal_matrix_tanh_ld(    bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, i, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, h, i, in, ik, N, K) );
#endif
  }

  libxsmm_barrier_wait(handle->barrier, (int)ltid);
}
