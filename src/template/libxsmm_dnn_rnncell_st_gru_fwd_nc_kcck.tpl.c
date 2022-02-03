/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Kunal Banerjee (Intel Corp.)
******************************************************************************/

/* helper variables */
libxsmm_blasint j, ik, ikb, in, ic, icb, inik, BF, CB, CB_BLOCKS, KB_BLOCKS;
/* input sizes */
const libxsmm_blasint K =  handle->desc.K;
const libxsmm_blasint N =  handle->desc.N;
const libxsmm_blasint C =  handle->desc.C;
const libxsmm_blasint t =  handle->T;
const libxsmm_blasint bk = handle->bk;
const libxsmm_blasint bn = handle->bn;
const libxsmm_blasint bc = handle->bc;
const libxsmm_blasint cBlocks = C/bc;
const libxsmm_blasint kBlocks = K/bk;
unsigned long long blocks;

/* define tensors */
element_input_type  *xt  = (element_input_type* )handle->xt->data;
element_input_type  *hpD = (element_input_type* )handle->hp->data;
element_filter_type *w   = (element_filter_type*)handle->w->data;
element_filter_type *r   = (element_filter_type*)handle->r->data;
element_output_type *b   = (element_output_type*)handle->b->data;
element_output_type *ht  = (element_output_type*)handle->ht->data;
element_output_type *it  = (element_output_type*)handle->it->data;
element_output_type *ct  = (element_output_type*)handle->cit->data;
element_output_type *ft  = (element_output_type*)handle->ft->data;
element_output_type *ot  = (element_output_type*)handle->ot->data;
element_filter_type *wiD = &(w[0]);
element_filter_type *wcD = &(w[C*K]);
element_filter_type *wfD = &(w[2*C*K]);
element_filter_type *riD = &(r[0]);
element_filter_type *rcD = &(r[K*K]);
element_filter_type *rfD = &(r[2*K*K]);
element_output_type *bi  = &(b[0]);
element_output_type *bd  = &(b[K]);
element_output_type *bf  = &(b[2*K]);
LIBXSMM_VLA_DECL(3, element_input_type,  x, xt, N, C);
LIBXSMM_VLA_DECL(2, element_input_type,  hp, hpD, K);
LIBXSMM_VLA_DECL(4, element_filter_type, wi, wiD, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, wc, wcD, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, wf, wfD, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, ri, riD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, rc, rcD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, rf, rfD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(3, element_output_type, h, ht, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, i, it, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, c, ct, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, f, ft, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, o, ot, N, K);
/* define batch-reduce gemm kernels */
const libxsmm_smmfunction_reducebatch_addr batchreduce_kernela = libxsmm_smmdispatch_reducebatch_addr( bk, bn, bc, &bk, &C, &K, NULL, NULL, NULL, NULL );
const libxsmm_smmfunction_reducebatch_addr batchreduce_kernelb = libxsmm_smmdispatch_reducebatch_addr( bk, bn, bk, &bk, &K, &K, NULL, NULL, NULL, NULL );
/* define gemm kernels */
/* Auxiliary arrays for batch-reduce gemms */
const element_filter_type *A_array[1024];
const element_input_type  *B_array[1024];

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

#if 0
const int use_fused_implementation = (C == 2048 && K == 2048) ? 1 : 0;
#endif
BF = 1;
if ((C > 1024 && C <= 2048) || (K > 1024 && K <= 2048)) {
  BF = 8;
  while ( (cBlocks % BF != 0) || (kBlocks % BF != 0) ) {
    BF--;
  }
}
if (C > 2048 || K > 2048) {
  BF = 16;
  while ( (cBlocks % BF != 0) || (kBlocks % BF != 0) ) {
    BF--;
  }
}

if (C == 2048 && K == 1024) {
  BF = 2;
}

CB_BLOCKS = cBlocks/BF;
KB_BLOCKS = kBlocks/BF;

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, (int)ltid);

/* All data is in column-major format */
for (j = 0; j < t; ++j) {
  /* let's run the cell in blocks for good locality */
  /* Block reduction loop if requested */
  for (CB = 0; CB < BF; CB++) {
    for (inik = thr_begin; inik < thr_end; ++inik ) {
      in = (inik % (N/bn))*bn;
      ikb = inik / (N/bn);
      ik = ikb*bk;
      /* initialize i with bi */
      if (CB == 0) libxsmm_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &bi[ik] );
      /* i += W.x */
      for (icb = 0, ic = 0; icb < CB_BLOCKS; ic += bc, icb++) {
        A_array[icb] = &LIBXSMM_VLA_ACCESS(4, wi, ikb, icb + CB*CB_BLOCKS, 0, 0, cBlocks, bc, bk);
        B_array[icb] = &LIBXSMM_VLA_ACCESS(3, x, j, in, ic + CB*CB_BLOCKS*bc, N, C);
      }
      /* Reduce batch gemm call  */
      blocks = CB_BLOCKS;
      batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &blocks);
      /* i += R.hp */
      if (0 == j) {
        for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
          A_array[icb] = &LIBXSMM_VLA_ACCESS(4, ri, ikb, icb + CB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
          B_array[icb] = &LIBXSMM_VLA_ACCESS(2, hp, in, ic + CB*KB_BLOCKS*bk, K);
        }
      } else {
        for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
          A_array[icb] = &LIBXSMM_VLA_ACCESS(4, ri, ikb, icb + CB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
          B_array[icb] = &LIBXSMM_VLA_ACCESS(3, h, j-1, in, ic + CB*KB_BLOCKS*bk, N, K);
        }
      }
      /* Reduce batch gemm call  */
      blocks = KB_BLOCKS;
      batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &blocks);
      /* initialize c with bd */
      if (CB == 0) libxsmm_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, c, j, in, ik, N, K), &bd[ik] );
      /* c += W.x */
      for (icb = 0, ic = 0; icb < CB_BLOCKS; ic += bc, icb++) {
        A_array[icb] = &LIBXSMM_VLA_ACCESS(4, wc, ikb, icb + CB*CB_BLOCKS, 0, 0, cBlocks, bc, bk);
        B_array[icb] = &LIBXSMM_VLA_ACCESS(3, x, j, in, ic + CB*CB_BLOCKS*bc, N, C);
      }
      /* Reduce batch gemm call  */
      blocks = CB_BLOCKS;
      batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, c, j, in, ik, N, K), &blocks);
      /* c += R.hp */
      if (0 == j) {
        for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
          A_array[icb] = &LIBXSMM_VLA_ACCESS(4, rc, ikb, icb + CB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
          B_array[icb] = &LIBXSMM_VLA_ACCESS(2, hp, in, ic + CB*KB_BLOCKS*bk, K);
        }
      } else {
        for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
          A_array[icb] = &LIBXSMM_VLA_ACCESS(4, rc, ikb, icb + CB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
          B_array[icb] = &LIBXSMM_VLA_ACCESS(3, h, j-1, in, ic + CB*KB_BLOCKS*bk, N, K);
        }
      }
      /* Reduce batch gemm call  */
      blocks = KB_BLOCKS;
      batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, c, j, in, ik, N, K), &blocks);

      if (CB == BF-1) {
        /* i = sigmoid(i) */
        libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K) );
        /* o = hp . i */
        if (0 == j) {
          libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, hp, in, ik, K),        &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K) );
        } else {
          libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, h, j-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K) );
        }
      }
    }
  }
  libxsmm_barrier_wait(handle->barrier, (int)ltid);
  /* We need a barrier here to ensure all elements of o are computed before f can be computed */
  for (CB = 0; CB < BF; CB++) {
    for (inik = thr_begin; inik < thr_end; ++inik ) {
      in = (inik % (N/bn))*bn;
      ikb = inik / (N/bn);
      ik = ikb*bk;
      /* initialize f with bf */
      if (CB == 0) libxsmm_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &bf[ik] );
      /* f += W.x */
      for (icb = 0, ic = 0; icb < CB_BLOCKS; ic += bc, icb++) {
        A_array[icb] = &LIBXSMM_VLA_ACCESS(4, wf, ikb, icb + CB*CB_BLOCKS, 0, 0, cBlocks, bc, bk);
        B_array[icb] = &LIBXSMM_VLA_ACCESS(3, x, j, in, ic + CB*CB_BLOCKS*bc, N, C);
      }
      /* Reduce batch gemm call  */
      blocks = CB_BLOCKS;
      batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &blocks);
      /* f += R.o */
      for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
        A_array[icb] = &LIBXSMM_VLA_ACCESS(4, rf, ikb, icb + CB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
        B_array[icb] = &LIBXSMM_VLA_ACCESS(3, o, j, in, ic + CB*KB_BLOCKS*bk, N, K);
      }
      /* Reduce batch gemm call  */
      blocks = KB_BLOCKS;
      batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &blocks);

      if (CB == BF-1) {
        /* f = tanh(f) */
        libxsmm_internal_matrix_tanh_ld         ( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K) );
        /* c = sigmoid(c) */
        libxsmm_internal_matrix_sigmoid_ld      ( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, c, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, c, j, in, ik, N, K) );
        /* h = (1 - c) . f */
        libxsmm_internal_matrix_complement_ld   ( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, c, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, h, j, in, ik, N, K) );
        libxsmm_internal_matrix_eltwise_mult_ld ( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, h, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, h, j, in, ik, N, K) );
        /* h += c . hp */
        if (0 == j) {
          libxsmm_internal_matrix_eltwise_fma_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, c, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, hp, in, ik, K),        &LIBXSMM_VLA_ACCESS(3, h, j, in, ik, N, K) );
        } else {
          libxsmm_internal_matrix_eltwise_fma_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, c, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, h, j-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, h, j, in, ik, N, K) );
        }
      }
    }
  }
  libxsmm_barrier_wait(handle->barrier, (int)ltid);
}

