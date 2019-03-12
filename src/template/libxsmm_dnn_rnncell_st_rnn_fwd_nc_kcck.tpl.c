/******************************************************************************
 ** Copyright (c) 2017-2019, Intel Corporation                                **
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
/*int nBlocks = N/bn;*/
int cBlocks = C/bc;
int kBlocks = K/bk;
unsigned long long blocks;
LIBXSMM_VLA_DECL(3, element_input_type,  x, xt, N, C);
LIBXSMM_VLA_DECL(2, element_input_type,  hp, hpD, K);
LIBXSMM_VLA_DECL(4, element_filter_type, w, wD, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, r, rD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(3, element_output_type, h, ht, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, z, zt, N, K);
int prefetch_mode = LIBXSMM_GEMM_PREFETCH_NONE/*LIBXSMM_GEMM_PREFETCH_AL1_BL1*/;
/* define gemm kernels */
const libxsmm_smmfunction_reducebatch batchreduce_kernela = libxsmm_smmdispatch_reducebatch( bk, bn, bc, &bk, &C, &K, NULL, NULL, NULL, &prefetch_mode );
const libxsmm_smmfunction_reducebatch batchreduce_kernelb = libxsmm_smmdispatch_reducebatch( bk, bn, bk, &bk, &K, &K, NULL, NULL, NULL, &prefetch_mode );

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
int my_col_id = ltid % column_teams;
int my_row_id = ltid / column_teams;
int in_tasks = N/bn;
int ik_tasks = K/bk;
int in_tasks_per_thread = in_tasks/row_teams;
int ik_tasks_per_thread = ik_tasks/column_teams;
int my_in_start = my_row_id * in_tasks_per_thread;
int my_in_end = (my_row_id+1) * in_tasks_per_thread;
int my_ik_start = my_col_id * ik_tasks_per_thread;
int my_ik_end = (my_col_id+1) * ik_tasks_per_thread;
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
          A_array[ii][jj][ic] = (element_input_type*) &LIBXSMM_VLA_ACCESS(4, w, ik, ic, 0, 0, cBlocks, bc, bk);
          B_array[ii][jj][ic] = (element_input_type*) &LIBXSMM_VLA_ACCESS(3, x, i, in*bn, ic*bc, N, C);
        }
        /* z += U.h */
        if (0 == i) {
          /* Prepare arrays for the call */
          for (ic = 0; ic < kBlocks; ic++) {
            A_array2[ii][jj][ic] = (element_input_type*) &LIBXSMM_VLA_ACCESS(4, r, ik, ic, 0, 0, kBlocks, bk, bk);
            B_array2[ii][jj][ic] = (element_input_type*) &LIBXSMM_VLA_ACCESS(2, hp, in*bn, ic*bk, K);
          }
        } else {
          /* Prepare arrays for the call */
          for (ic = 0; ic < kBlocks; ic++) {
            A_array2[ii][jj][ic] = (element_input_type*) &LIBXSMM_VLA_ACCESS(4, r, ik, ic, 0, 0, kBlocks, bk, bk);
            B_array2[ii][jj][ic] = (element_input_type*) &LIBXSMM_VLA_ACCESS(3, h, i-1, in*bn, ic*bk, N, K);
          }
        }
      }
    }

    if (prefetch_mode != LIBXSMM_GEMM_PREFETCH_NONE) {
      /* Prepare addition prefetch arrays that are shifted images of regular ones when external prefetching is requested  */
      int pf_dist_A = 2;
      int pf_dist_B = 4;
      int total_blocks = in_tasks_per_thread*ik_tasks_per_thread*cBlocks;
      element_input_type *src_ptr = (element_input_type*) &A_array[0][0][0];
      element_input_type *dst_ptr = (element_input_type*) &A_array_pf[0][0][0];
      for (ii = 0 ; ii < total_blocks - pf_dist_A; ii++) {
        dst_ptr[ii] = src_ptr[ii+pf_dist_A];
      }
      src_ptr = (element_input_type*) &B_array[0][0][0];
      dst_ptr = (element_input_type*) &B_array_pf[0][0][0];
      for (ii = 0 ; ii < total_blocks - pf_dist_B; ii++) {
        dst_ptr[ii] = src_ptr[ii+pf_dist_B];
      }
      total_blocks = in_tasks_per_thread*ik_tasks_per_thread*kBlocks;
      src_ptr = (element_input_type*) &A_array2[0][0][0];
      dst_ptr = (element_input_type*) &A_array2_pf[0][0][0];
      for (ii = 0 ; ii < total_blocks - pf_dist_A; ii++) {
        dst_ptr[ii] = src_ptr[ii+pf_dist_A];
      }
      src_ptr = (element_input_type*) &B_array2[0][0][0];
      dst_ptr = (element_input_type*) &B_array2_pf[0][0][0];
      for (ii = 0 ; ii < total_blocks - pf_dist_B; ii++) {
        dst_ptr[ii] = src_ptr[ii+pf_dist_B];
      }
    }

    /* let's run the cell in blocks for good locality */
    for (ik = my_ik_start, ii = 0; ik < my_ik_end; ++ik, ii++ ) {
      for (in = my_in_start, jj = 0; in < my_in_end; ++in, jj++ ) {
        /* z = per_col(b) */
        libxsmm_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, i, in*bn, ik*bk, N, K), &b[ik*bk] );
        /* z += W.x */
        blocks = cBlocks;
        batchreduce_kernela(&A_array[ii][jj][0], &B_array[ii][jj][0], &LIBXSMM_VLA_ACCESS(3, z, i, in*bn, ik*bk, N, K), &blocks, &A_array_pf[ii][jj][0], &B_array_pf[ii][jj][0]);
        /* z += U.h */
        blocks = kBlocks;
        batchreduce_kernelb(&A_array2[ii][jj][0], &B_array2[ii][jj][0], &LIBXSMM_VLA_ACCESS(3, z, i, in*bn, ik*bk, N, K), &blocks, &A_array2_pf[ii][jj][0], &B_array2_pf[ii][jj][0]);
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
} else {
  /* Auxiliary arrays for batch-reduce gemms  */
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
      libxsmm_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, i, in*bn, ik*bk, N, K), &b[ik*bk] );

      /* z += W.x */
      /* Prepare arrays for the call */
      for (ic = 0; ic < cBlocks; ic++) {
        /* this is a small matmul */
        A_array[ic] = (element_input_type*) &LIBXSMM_VLA_ACCESS(4, w, ik, ic, 0, 0, cBlocks, bc, bk);
        B_array[ic] = (element_input_type*) &LIBXSMM_VLA_ACCESS(3, x, i, in*bn, ic*bc, N, C);
      }
      /* Reduce batch gemm call  */
      blocks = cBlocks;
      batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, z, i, in*bn, ik*bk, N, K), &blocks);

      /* z += U.h */
      if (0 == i) {
        /* Prepare arrays for the call */
        for (ic = 0; ic < kBlocks; ic++) {
          A_array2[ic] = (element_input_type*) &LIBXSMM_VLA_ACCESS(4, r, ik, ic, 0, 0, kBlocks, bk, bk);
          B_array2[ic] = (element_input_type*) &LIBXSMM_VLA_ACCESS(2, hp, in*bn, ic*bk, K);
        }
        /* Reduce batch gemm call  */
        blocks = kBlocks;
        batchreduce_kernelb(A_array2, B_array2, &LIBXSMM_VLA_ACCESS(3, z, i, in*bn, ik*bk, N, K), &blocks);
      } else {
        /* Prepare arrays for the call */
        for (ic = 0; ic < kBlocks; ic++) {
          A_array2[ic] = (element_input_type*) &LIBXSMM_VLA_ACCESS(4, r, ik, ic, 0, 0, kBlocks, bk, bk);
          B_array2[ic] = (element_input_type*) &LIBXSMM_VLA_ACCESS(3, h, i-1, in*bn, ic*bk, N, K);
        }
        /* Reduce batch gemm call  */
        blocks = kBlocks;
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
    libxsmm_barrier_wait(handle->barrier, (int)ltid);
  }
}

