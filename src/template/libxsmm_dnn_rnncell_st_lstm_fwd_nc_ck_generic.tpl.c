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
/* Kunal Banerjee (Intel Corp.)
******************************************************************************/

/* helper variables */
libxsmm_blasint j, ik, in, ic, inik;
/* input sizes */
const libxsmm_blasint K =  handle->desc.K;
const libxsmm_blasint N =  handle->desc.N;
const libxsmm_blasint C =  handle->desc.C;
const libxsmm_blasint t =  handle->desc.t;
const libxsmm_blasint bk = handle->bk;
const libxsmm_blasint bn = handle->bn;
const libxsmm_blasint bc = handle->bc;
const libxsmm_blasint K4 = K * 4;
/* define tensors */
element_input_type  *xt  = (element_input_type* )handle->xt->data;
element_input_type  *csp = (element_input_type* )handle->csp->data;
element_input_type  *hpD = (element_input_type* )handle->hp->data;
element_filter_type *w   = (element_filter_type*)handle->w->data;
element_filter_type *r   = (element_filter_type*)handle->r->data;
element_output_type *b   = (element_output_type*)handle->b->data;
element_output_type *cst = (element_output_type*)handle->cst->data;
element_output_type *ht  = (element_output_type*)handle->ht->data;
element_output_type *it  = (element_output_type*)handle->it->data;
element_output_type *ft  = (element_output_type*)handle->ft->data;
element_output_type *ot  = (element_output_type*)handle->ot->data;
element_output_type *cit = (element_output_type*)handle->cit->data;
element_output_type *cot = (element_output_type*)handle->cot->data;
element_filter_type *wiD = &(w[0]);
element_filter_type *wcD = &(w[K]);
element_filter_type *wfD = &(w[2*K]);
element_filter_type *woD = &(w[3*K]);
element_filter_type *riD = &(r[0]);
element_filter_type *rcD = &(r[K]);
element_filter_type *rfD = &(r[2*K]);
element_filter_type *roD = &(r[3*K]);
element_output_type *bi  = &(b[0]);
element_output_type *bd  = &(b[K]);
element_output_type *bf  = &(b[2*K]);
element_output_type *bo  = &(b[3*K]);
LIBXSMM_VLA_DECL(3, element_input_type,  x, xt, N, C);
LIBXSMM_VLA_DECL(2, element_input_type,  cp, csp, K);
LIBXSMM_VLA_DECL(2, element_input_type,  hp, hpD, K);
LIBXSMM_VLA_DECL(2, element_filter_type, wi, wiD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, wf, wfD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, wo, woD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, wc, wcD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, ri, riD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, rf, rfD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, ro, roD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, rc, rcD, 4*K);
LIBXSMM_VLA_DECL(3, element_output_type, cs, cst, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, h, ht, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, i, it, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, f, ft, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, o, ot, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, ci, cit, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, co, cot, N, K);
/* define gemm kernels */
libxsmm_smmfunction gemmkernela = libxsmm_smmdispatch( bk, bn, bc, &K4, &C, &K, NULL, NULL, NULL, NULL );
libxsmm_smmfunction gemmkernelb = libxsmm_smmdispatch( bk, bn, bk, &K4, &K, &K, NULL, NULL, NULL, NULL );
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
for (j = 0; j < t; ++j) {
  /* let's run the cell in blocks for good locality */
  for (inik = thr_begin; inik < thr_end; ++inik ) {
    in = (inik / (K/bk))*bn;
    ik = (inik % (K/bk))*bk;
    /* initialize i with bi */
    libxsmm_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &bi[ik] );
    /* i += W.x */
    for (ic = 0; ic < C; ic += bc) {
      gemmkernela( &LIBXSMM_VLA_ACCESS(2, wi, ic, ik, 4*K), &LIBXSMM_VLA_ACCESS(3, x, j, in, ic, N, C), &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K) );
    }
    /* i += R.h */
    if (0 == j) {
      for (ic = 0; ic < K; ic += bk) {
        gemmkernelb( &LIBXSMM_VLA_ACCESS(2, ri, ic, ik, 4*K), &LIBXSMM_VLA_ACCESS(2, hp, in, ic, K), &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K) );
      }
    } else {
      for (ic = 0; ic < K; ic += bk) {
        gemmkernelb( &LIBXSMM_VLA_ACCESS(2, ri, ic, ik, 4*K), &LIBXSMM_VLA_ACCESS(3, h, j-1, in, ic, N, K), &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K) );
      }
    }
    /* i = sigmoid(i) */
    libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K) );

    /* initialize f with (bf + forget_bias) */
    libxsmm_internal_matrix_bcst_colvector_const_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &bf[ik], handle->forget_bias );
    /* f += W.x */
    for (ic = 0; ic < C; ic += bc) {
      gemmkernela( &LIBXSMM_VLA_ACCESS(2, wf, ic, ik, 4*K), &LIBXSMM_VLA_ACCESS(3, x, j, in, ic, N, C), &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K) );
    }
    /* f += R.h */
    if (0 == j) {
      for (ic = 0; ic < K; ic += bk) {
        gemmkernelb( &LIBXSMM_VLA_ACCESS(2, rf, ic, ik, 4*K), &LIBXSMM_VLA_ACCESS(2, hp, in, ic, K), &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K) );
      }
    } else {
      for (ic = 0; ic < K; ic += bk) {
        gemmkernelb( &LIBXSMM_VLA_ACCESS(2, rf, ic, ik, 4*K), &LIBXSMM_VLA_ACCESS(3, h, j-1, in, ic, N, K), &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K) );
      }
    }
    /* f = sigmoid(f) */
    libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K) );

    /* initialize o with bo */
    libxsmm_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &bo[ik] );
    /* o += W.x */
    for (ic = 0; ic < C; ic += bc) {
      gemmkernela( &LIBXSMM_VLA_ACCESS(2, wo, ic, ik, 4*K), &LIBXSMM_VLA_ACCESS(3, x, j, in, ic, N, C), &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K) );
    }
    /* o += R.h */
    if (0 == j) {
      for (ic = 0; ic < K; ic += bk) {
        gemmkernelb( &LIBXSMM_VLA_ACCESS(2, ro, ic, ik, 4*K), &LIBXSMM_VLA_ACCESS(2, hp, in, ic, K), &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K) );
      }
    } else {
      for (ic = 0; ic < K; ic += bk) {
        gemmkernelb( &LIBXSMM_VLA_ACCESS(2, ro, ic, ik, 4*K), &LIBXSMM_VLA_ACCESS(3, h, j-1, in, ic, N, K), &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K) );
      }
    }
    /* o = sigmoid(o) */
    libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K) );

    /* initialize ci with bd */
    libxsmm_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &bd[ik] );
    /* ci += W.x */
    for (ic = 0; ic < C; ic += bc) {
      gemmkernela( &LIBXSMM_VLA_ACCESS(2, wc, ic, ik, 4*K), &LIBXSMM_VLA_ACCESS(3, x, j, in, ic, N, C), &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K) );
    }
    /* ci += R.h */
    if (0 == j) {
      for (ic = 0; ic < K; ic += bk) {
        gemmkernelb( &LIBXSMM_VLA_ACCESS(2, rc, ic, ik, 4*K), &LIBXSMM_VLA_ACCESS(2, hp, in, ic, K), &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K) );
      }
    } else {
      for (ic = 0; ic < K; ic += bk) {
        gemmkernelb( &LIBXSMM_VLA_ACCESS(2, rc, ic, ik, 4*K), &LIBXSMM_VLA_ACCESS(3, h, j-1, in, ic, N, K), &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K) );
      }
    }
    /* ci = tanh(ci) */
    libxsmm_internal_matrix_tanh_ld(    bk, bn, K, &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K) );

    /* cs = f.cs */
    if (0 == j) {
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, cp, in, ik, K), &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K) );
    } else {
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, cs, j-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K) );
    }
    /* cs += i.ci */
    libxsmm_internal_matrix_eltwise_fma_ld(  bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K) );
    /* co = tanh(cs) */
    libxsmm_internal_matrix_tanh_ld(         bk, bn, K, &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K) );
    /* h = o.co */
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K),  &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, h, j, in, ik, N, K) );
  }

  libxsmm_barrier_wait(handle->barrier, (int)ltid);
}
