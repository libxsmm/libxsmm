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
libxsmm_blasint i, ik, in, ic, inik;
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
unsigned long long nBlocks = N/bn;
unsigned long long cBlocks = C/bc;
unsigned long long kBlocks = K/bk;
LIBXSMM_VLA_DECL(5, element_input_type,  x, xt, nBlocks, cBlocks, bn, bc);
LIBXSMM_VLA_DECL(4, element_input_type,  hp, hpD, kBlocks, bn, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, w, wD, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, r, rD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(5, element_output_type, h, ht, nBlocks, kBlocks, bn, bk);
LIBXSMM_VLA_DECL(5, element_output_type, z, zt, nBlocks, kBlocks, bn, bk);
/* define gemm kernels */
libxsmm_smmfunction_reducebatch batchreduce_kernela =  libxsmm_smmdispatch_reducebatch( bk, bn, bc, &bk, &bk, &bk, NULL, NULL );
libxsmm_smmfunction_reducebatch batchreduce_kernelb =  libxsmm_smmdispatch_reducebatch( bk, bn, bk, &bk, &bk, &bk, NULL, NULL );

/* computing first logical thread */
const libxsmm_blasint ltid = (libxsmm_blasint)tid - (libxsmm_blasint)start_thread;
/* number of tasks that could be run in parallel */
const libxsmm_blasint work = (N/bn) * (K/bk);
/* compute chunk size */
const libxsmm_blasint chunksize = (work % (libxsmm_blasint)handle->desc.threads == 0) ? (work / (libxsmm_blasint)handle->desc.threads) : ((work / (libxsmm_blasint)handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* Auxuliary arrays for batch-reduce gemms  */
const element_input_type *A_array[cBlocks];
const element_input_type *B_array[cBlocks];
const element_input_type *A_array2[kBlocks];
const element_input_type *B_array2[kBlocks];

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
      A_array[ic] = (element_input_type*) &LIBXSMM_VLA_ACCESS(4, w, ik, ic, 0, 0, cBlocks, bc, bk);
      B_array[ic] = (element_input_type*) &LIBXSMM_VLA_ACCESS(5, x, i, in, ic, 0, 0, nBlocks, cBlocks, bn, bc);
    }
    /* Reduce batch gemm call  */
    batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(5, z, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk), &cBlocks);

    /* z += U.h */
    if (0 == i) {
      /* Prepare arrays for the call */
      for (ic = 0; ic < kBlocks; ic++) {
        A_array2[ic] = (element_input_type*) &LIBXSMM_VLA_ACCESS(4, r, ik, ic, 0, 0, kBlocks, bk, bk);
        B_array2[ic] = (element_input_type*) &LIBXSMM_VLA_ACCESS(4, hp, in, ic, 0, 0, kBlocks, bn, bk);
      }
      /* Reduce batch gemm call  */
      batchreduce_kernelb(A_array2, B_array2, &LIBXSMM_VLA_ACCESS(5, z, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk), &kBlocks);
    } else {
      /* Prepare arrays for the call */
      for (ic = 0; ic < kBlocks; ic++) {
        A_array2[ic] = (element_input_type*) &LIBXSMM_VLA_ACCESS(4, r, ik, ic, 0, 0, kBlocks, bk, bk);
        B_array2[ic] = (element_input_type*) &LIBXSMM_VLA_ACCESS(5, h, i-1, in, ic, 0, 0, nBlocks, kBlocks, bn, bk);
      }
      /* Reduce batch gemm call  */
      batchreduce_kernelb(A_array2, B_array2, &LIBXSMM_VLA_ACCESS(5, z, i, in, ik, 0, 0, nBlocks, kBlocks, bn, bk), &kBlocks);
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

