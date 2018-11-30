/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
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
libxsmm_blasint i, ik, in, ic, inik, j;
/* input sizes */
const libxsmm_blasint K =  handle->desc.K;
const libxsmm_blasint N =  handle->desc.N;
const libxsmm_blasint C =  handle->desc.C;
const libxsmm_blasint t =  handle->desc.t;
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

int nBlocks = N/bn;
int cBlocks = C/bc;
int kBlocks = K/bk;

LIBXSMM_VLA_DECL(5, element_input_type,  x, xt, nBlocks, cBlocks, bn, bc);
LIBXSMM_VLA_DECL(4, element_input_type,  hp, hpD, kBlocks, bn, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, w, wD, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, r, rD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(5, element_output_type, h, ht, nBlocks, kBlocks, bn, bk);
LIBXSMM_VLA_DECL(5, element_output_type, z, zt, nBlocks, kBlocks, bn, bk);
/* define gemm kernels */

libxsmm_smmfunction_reducebatch batchreduce_kernela =  libxsmm_smmdispatch_reducebatch( bk, bn, bc, &bk, &bk, &bk, NULL, NULL );
libxsmm_smmfunction_reducebatch batchreduce_kernelb =  libxsmm_smmdispatch_reducebatch( bk, bn, bk, &bk, &bk, &bk, NULL, NULL );

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

unsigned long long c_blocks = C/bc;
const element_input_type *A_array[c_blocks];
const element_input_type *B_array[c_blocks];

unsigned long long k_blocks = K/bk;
const element_input_type *A_array2[k_blocks];
const element_input_type *B_array2[k_blocks];

/* All data is in column-major format */
for (i = 0; i < t; ++i) {
  /* let's run the cell in blocks for good locality */
  for (inik = thr_begin; inik < thr_end; ++inik ) {
    in = (inik / (K/bk))*bn;
    ik = (inik % (K/bk))*bk;

    /* z = per_col(b) */
    libxsmm_internal_matrix_bcst_colvector_ld( bk, bn, bk, &LIBXSMM_VLA_ACCESS(5, z, i, in/bn, ik/bk, 0, 0, nBlocks, kBlocks, bn, bk), &b[ik]);

    /* z += W.x */
    /* Prepare arrays for the call */
    for (ic = 0, j = 0 ; ic < C; ic += bc, j++) {
      /* this is a small matmul */
      A_array[j] = (element_input_type*) &LIBXSMM_VLA_ACCESS(4, w, ik/bk, ic/bc, 0, 0, cBlocks, bc, bk);
      B_array[j] = (element_input_type*) &LIBXSMM_VLA_ACCESS(5, x, i, in/bn, ic/bc, 0, 0, nBlocks, cBlocks, bn, bc);
    }
    /* Reduce batch gemm call  */
    batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(5, z, i, in/bn, ik/bk, 0, 0, nBlocks, kBlocks, bn, bk), &c_blocks);

    /* z += U.h */
    if (0 == i) {
      /* Prepare arrays for the call */
      for (ic = 0, j=0; ic < K; ic += bk, j++) {
        A_array2[j] = (element_input_type*) &LIBXSMM_VLA_ACCESS(4, r, ik/bk, ic/bk, 0, 0, kBlocks, bk, bk);
        B_array2[j] = (element_input_type*) &LIBXSMM_VLA_ACCESS(4, hp, in/bn, ic/bk, 0, 0, kBlocks, bn, bk);
      }
      /* Reduce batch gemm call  */
      batchreduce_kernelb(A_array2, B_array2, &LIBXSMM_VLA_ACCESS(5, z, i, in/bn, ik/bk, 0, 0, nBlocks, kBlocks, bn, bk), &k_blocks);
    } else {
      /* Prepare arrays for the call */
      for (ic = 0, j= 0; ic < K; ic += bk, j++) {
        A_array2[j] = (element_input_type*) &LIBXSMM_VLA_ACCESS(4, r, ik/bk, ic/bk, 0, 0, kBlocks, bk, bk);
        B_array2[j] = (element_input_type*) &LIBXSMM_VLA_ACCESS(5, h, i-1, in/bn, ic/bk, 0, 0, nBlocks, kBlocks, bn, bk);
      }
      /* Reduce batch gemm call  */
      batchreduce_kernelb(A_array2, B_array2, &LIBXSMM_VLA_ACCESS(5, z, i, in/bn, ik/bk, 0, 0, nBlocks, kBlocks, bn, bk), &k_blocks);
    }

#if defined(LIBXSMM_DNN_RNN_RELU_FWD)
    libxsmm_internal_matrix_relu_ld(    bk, bn, bk, &LIBXSMM_VLA_ACCESS(5, z, i, in/bn, ik/bk, 0, 0, nBlocks, kBlocks, bn, bk), &LIBXSMM_VLA_ACCESS(5, h, i, in/bn, ik/bk, 0, 0, nBlocks, kBlocks, bn, bk));
#endif
#if defined(LIBXSMM_DNN_RNN_SIGMOID_FWD)
    libxsmm_internal_matrix_sigmoid_ld( bk, bn, bk, &LIBXSMM_VLA_ACCESS(5, z, i, in/bn, ik/bk, 0, 0, nBlocks, kBlocks, bn, bk), &LIBXSMM_VLA_ACCESS(5, h, i, in/bn, ik/bk, 0, 0, nBlocks, kBlocks, bn, bk));
#endif
#if defined(LIBXSMM_DNN_RNN_TANH_FWD)
    libxsmm_internal_matrix_tanh_ld(    bk, bn, bk, &LIBXSMM_VLA_ACCESS(5, z, i, in/bn, ik/bk, 0, 0, nBlocks, kBlocks, bn, bk), &LIBXSMM_VLA_ACCESS(5, h, i, in/bn, ik/bk, 0, 0, nBlocks, kBlocks, bn, bk));
#endif
  }

  libxsmm_barrier_wait(handle->barrier, (int)ltid);
}

