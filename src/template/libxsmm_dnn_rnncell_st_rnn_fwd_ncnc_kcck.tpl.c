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
libxsmm_blasint nBlocks = N/bn;
libxsmm_blasint cBlocks = C/bc;
libxsmm_blasint kBlocks = K/bk;
unsigned long long blocks;
LIBXSMM_VLA_DECL(5, element_input_type,  x, xt, nBlocks, cBlocks, bn, bc);
LIBXSMM_VLA_DECL(4, element_input_type,  hp, hpD, kBlocks, bn, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, w, wD, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, r, rD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(5, element_output_type, h, ht, nBlocks, kBlocks, bn, bk);
LIBXSMM_VLA_DECL(5, element_output_type, z, zt, nBlocks, kBlocks, bn, bk);
int prefetch_mode = LIBXSMM_GEMM_PREFETCH_NONE/*LIBXSMM_GEMM_PREFETCH_AL1_BL1*/;
/* define gemm kernels */
const libxsmm_smmfunction_reducebatch_addr batchreduce_kernela = libxsmm_smmdispatch_reducebatch_addr( bk, bn, bc, &bk, &bc, &bk, NULL, NULL, NULL, &prefetch_mode );
const libxsmm_smmfunction_reducebatch_addr batchreduce_kernelb = libxsmm_smmdispatch_reducebatch_addr( bk, bn, bk, &bk, &bk, &bk, NULL, NULL, NULL, &prefetch_mode );

/* computing first logical thread */
const libxsmm_blasint ltid = (libxsmm_blasint)tid - (libxsmm_blasint)start_thread;
/* number of tasks that could be run in parallel */
const libxsmm_blasint work = (N/bn) * (K/bk);
/* compute chunk size */
const libxsmm_blasint chunksize = (work % (libxsmm_blasint)handle->desc.threads == 0) ? (work / (libxsmm_blasint)handle->desc.threads) : ((work / (libxsmm_blasint)handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* The snippet below does a 2D domain decomposition of output IF the number of threads and the number of work items are compatible */
/* TODO: For now 2D decomposition targets single socket SKX */
int row_teams = 7;
int column_teams = 4;
libxsmm_blasint my_col_id = ltid % column_teams;
libxsmm_blasint my_row_id = ltid / column_teams;
int in_tasks = (int)(N/bn);
int ik_tasks = (int)(K/bk);
int in_tasks_per_thread = in_tasks/row_teams;
int ik_tasks_per_thread = ik_tasks/column_teams;
libxsmm_blasint my_in_start = my_row_id * in_tasks_per_thread;
libxsmm_blasint my_in_end = (my_row_id+1) * in_tasks_per_thread;
libxsmm_blasint my_ik_start = my_col_id * ik_tasks_per_thread;
libxsmm_blasint my_ik_end = (my_col_id+1) * ik_tasks_per_thread;
int perform_2d_decomp = (in_tasks % row_teams == 0 && ik_tasks % column_teams == 0 && row_teams*column_teams == handle->desc.threads && cBlocks <= 32 && kBlocks <= 32 && ik_tasks_per_thread <= 16 && in_tasks_per_thread <= 2 ) ? 1 : 0;

if (perform_2d_decomp) {
  /* Auxiliary arrays for batch-reduce gemms and potential prefetch */
  const element_input_type *A_array[16][2][32];
  const element_input_type *B_array[16][2][32];
  const element_input_type *A_array2[16][2][32];
  const element_input_type *B_array2[16][2][32];
  const element_input_type *A_array_pf[16][2][32];
  const element_input_type *B_array_pf[16][2][32];
  const element_input_type *A_array2_pf[16][2][32];
  const element_input_type *B_array2_pf[16][2][32];
  int ii, jj;

  /* lazy barrier init */
  libxsmm_barrier_init(handle->barrier, (int)ltid);

  /* All data is in column-major format */
  for (i = 0; i < t; ++i) {
    /* Prepare arrays for the batch-reduce calls */
    for (ik = my_ik_start, ii = 0; ik < my_ik_end; ++ik, ii++ ) {
      for (in = my_in_start, jj = 0; in < my_in_end; ++in, jj++ ) {
        /* Prepare arrays for the call */
        for (ic = 0; ic < cBlocks; ic++) {
          /* this is a small matmul */
          A_array[ii][jj][ic] = &LIBXSMM_VLA_ACCESS(4, w, ik, ic, 0, 0, cBlocks, bc, bk);
          B_array[ii][jj][ic] = &LIBXSMM_VLA_ACCESS(5, x, i, in, ic, 0, 0, nBlocks, cBlocks, bn, bc);
        }
        /* z += U.h */
        if (0 == i) {
          /* Prepare arrays for the call */
          for (ic = 0; ic < kBlocks; ic++) {
            A_array2[ii][jj][ic] = &LIBXSMM_VLA_ACCESS(4, r, ik, ic, 0, 0, kBlocks, bk, bk);
            B_array2[ii][jj][ic] = &LIBXSMM_VLA_ACCESS(4, hp, in, ic, 0, 0, kBlocks, bn, bk);
          }
        } else {
          /* Prepare arrays for the call */
          for (ic = 0; ic < kBlocks; ic++) {
            A_array2[ii][jj][ic] = &LIBXSMM_VLA_ACCESS(4, r, ik, ic, 0, 0, kBlocks, bk, bk);
            B_array2[ii][jj][ic] = &LIBXSMM_VLA_ACCESS(5, h, i-1, in, ic, 0, 0, nBlocks, kBlocks, bn, bk);
          }
        }
      }
    }

    if (prefetch_mode != LIBXSMM_GEMM_PREFETCH_NONE) { /* coverity[dead_error_begin] */
      /* Prepare additional prefetch arrays that are shifted images of regular ones when external prefetching is requested  */
      int pf_dist_A = 2;
      int pf_dist_B = 4;
      libxsmm_blasint total_blocks = in_tasks_per_thread*ik_tasks_per_thread*cBlocks;
      const element_input_type **src_ptr = &A_array[0][0][0];
      const element_input_type **dst_ptr = &A_array_pf[0][0][0];
      for (ii = 0; ii < total_blocks - pf_dist_A; ii++) {
        dst_ptr[ii] = src_ptr[ii+pf_dist_A];
      }
      src_ptr = &B_array[0][0][0];
      dst_ptr = &B_array_pf[0][0][0];
      for (ii = 0; ii < total_blocks - pf_dist_B; ii++) {
        dst_ptr[ii] = src_ptr[ii+pf_dist_B];
      }
      total_blocks = in_tasks_per_thread*ik_tasks_per_thread*kBlocks;
      src_ptr = &A_array2[0][0][0];
      dst_ptr = &A_array2_pf[0][0][0];
      for (ii = 0; ii < total_blocks - pf_dist_A; ii++) {
        dst_ptr[ii] = src_ptr[ii+pf_dist_A];
      }
      src_ptr = &B_array2[0][0][0];
      dst_ptr = &B_array2_pf[0][0][0];
      for (ii = 0; ii < total_blocks - pf_dist_B; ii++) {
        dst_ptr[ii] = src_ptr[ii+pf_dist_B];
      }
    }

    /* let's run the cell in blocks for good locality */
    for (ik = my_ik_start, ii = 0; ik < my_ik_end; ++ik, ii++ ) {
      for (in = my_in_start, jj = 0; in < my_in_end; ++in, jj++ ) {
        /* z = per_col(b) */
        libxsmm_internal_matrix_bcst_colvector_ld( bk, bn, bk, &LIBXSMM_VLA_ACCESS(5, z, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk), &b[ik*bk]);
        /* z += W.x */
        blocks = cBlocks;
        batchreduce_kernela(&A_array[ii][jj][0], &B_array[ii][jj][0], &LIBXSMM_VLA_ACCESS(5, z, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk), &blocks, &A_array_pf[ii][jj][0], &B_array_pf[ii][jj][0]);
        /* z += U.h */
        blocks = kBlocks;
        batchreduce_kernelb(&A_array2[ii][jj][0], &B_array2[ii][jj][0], &LIBXSMM_VLA_ACCESS(5, z, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk), &blocks, &A_array2_pf[ii][jj][0], &B_array2_pf[ii][jj][0]);

#if defined(LIBXSMM_DNN_RNN_RELU_FWD)
        libxsmm_internal_matrix_relu_ld(    bk, bn, bk, &LIBXSMM_VLA_ACCESS(5, z, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk), &LIBXSMM_VLA_ACCESS(5, h, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk));
#endif
#if defined(LIBXSMM_DNN_RNN_SIGMOID_FWD)
        libxsmm_internal_matrix_sigmoid_ld( bk, bn, bk, &LIBXSMM_VLA_ACCESS(5, z, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk), &LIBXSMM_VLA_ACCESS(5, h, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk));
#endif
#if defined(LIBXSMM_DNN_RNN_TANH_FWD)
        libxsmm_internal_matrix_tanh_ld(    bk, bn, bk, &LIBXSMM_VLA_ACCESS(5, z, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk), &LIBXSMM_VLA_ACCESS(5, h, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk));
#endif
      }
    }
    libxsmm_barrier_wait(handle->barrier, (int)ltid);
  }
} else {
  /* Auxiliary arrays for batch-reduce gemms */
  const element_input_type *A_array[1024];
  const element_input_type *B_array[1024];
  const element_input_type *A_array2[1024];
  const element_input_type *B_array2[1024];
  assert(kBlocks <= 1024);
  assert(cBlocks <= 1024);

  /* lazy barrier init */
  libxsmm_barrier_init(handle->barrier, (int)ltid);

  /* All data is in column-major format */
  for (i = 0; i < t; ++i) {
    /* let's run the cell in blocks for good locality */
    for (inik = thr_begin; inik < thr_end; ++inik ) {
      in = inik / (K/bk);
      ik = inik % (K/bk);

      /* z = per_col(b) */
      libxsmm_internal_matrix_bcst_colvector_ld( bk, bn, bk, &LIBXSMM_VLA_ACCESS(5, z, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk), &b[ik*bk]);

      /* z += W.x */
      /* Prepare arrays for the call */
      for (ic = 0; ic < cBlocks; ic++) {
        /* this is a small matmul */
        A_array[ic] = &LIBXSMM_VLA_ACCESS(4, w, ik, ic, 0, 0, cBlocks, bc, bk);
        B_array[ic] = &LIBXSMM_VLA_ACCESS(5, x, i, in, ic, 0, 0, nBlocks, cBlocks, bn, bc);
      }
      /* Reduce batch gemm call  */
      blocks = cBlocks;
      batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(5, z, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk), &blocks);

      /* z += U.h */
      if (0 == i) {
        /* Prepare arrays for the call */
        for (ic = 0; ic < kBlocks; ic++) {
          A_array2[ic] = &LIBXSMM_VLA_ACCESS(4, r, ik, ic, 0, 0, kBlocks, bk, bk);
          B_array2[ic] = &LIBXSMM_VLA_ACCESS(4, hp, in, ic, 0, 0, kBlocks, bn, bk);
        }
        /* Reduce batch gemm call  */
        blocks = kBlocks;
        batchreduce_kernelb(A_array2, B_array2, &LIBXSMM_VLA_ACCESS(5, z, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk), &blocks);
      } else {
        /* Prepare arrays for the call */
        for (ic = 0; ic < kBlocks; ic++) {
          A_array2[ic] = &LIBXSMM_VLA_ACCESS(4, r, ik, ic, 0, 0, kBlocks, bk, bk);
          B_array2[ic] = &LIBXSMM_VLA_ACCESS(5, h, i-1, in, ic, 0, 0, nBlocks, kBlocks, bn, bk);
        }
        /* Reduce batch gemm call  */
        blocks = kBlocks;
        batchreduce_kernelb(A_array2, B_array2, &LIBXSMM_VLA_ACCESS(5, z, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk), &blocks);
      }

#if defined(LIBXSMM_DNN_RNN_RELU_FWD)
      libxsmm_internal_matrix_relu_ld(    bk, bn, bk, &LIBXSMM_VLA_ACCESS(5, z, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk), &LIBXSMM_VLA_ACCESS(5, h, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk));
#endif
#if defined(LIBXSMM_DNN_RNN_SIGMOID_FWD)
      libxsmm_internal_matrix_sigmoid_ld( bk, bn, bk, &LIBXSMM_VLA_ACCESS(5, z, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk), &LIBXSMM_VLA_ACCESS(5, h, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk));
#endif
#if defined(LIBXSMM_DNN_RNN_TANH_FWD)
      libxsmm_internal_matrix_tanh_ld(    bk, bn, bk, &LIBXSMM_VLA_ACCESS(5, z, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk), &LIBXSMM_VLA_ACCESS(5, h, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk));
#endif
    }
    libxsmm_barrier_wait(handle->barrier, (int)ltid);
  }
}

