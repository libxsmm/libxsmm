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
/* Evangelos Georganas, Kunal Banerjee (Intel Corp.)
******************************************************************************/
//#define PROFILE

/* helper variables */
libxsmm_blasint j, ik, ikb, in, ic, icb, inik, BF, CB, CB_BLOCKS, KB_BLOCKS, ikic, jk, jc;
/* input sizes */
const libxsmm_blasint K =  handle->desc.K;
const libxsmm_blasint N =  handle->desc.N;
const libxsmm_blasint C =  handle->desc.C;
const libxsmm_blasint t =  handle->desc.t;
const libxsmm_blasint bk = handle->bk;
const libxsmm_blasint bn = handle->bn;
const libxsmm_blasint bc = handle->bc;
const int cBlocks = C/bc;
const int kBlocks = K/bk;
unsigned long long blocks;

/* define tensors */
element_input_type  *xt  = (element_input_type* )handle->xt->data;
element_input_type  *csp = (element_input_type* )handle->csp->data;
element_input_type  *hpD = (element_input_type* )handle->hp->data;
element_filter_type *w   = (element_filter_type*)handle->w->data;
element_filter_type *r   = (element_filter_type*)handle->r->data;
element_filter_type *w_scratch   = (element_filter_type*)handle->scratch_w;
element_filter_type *r_scratch   = (element_filter_type*)handle->scratch_r;
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
element_filter_type *wiD_scratch = &(w_scratch[0]);
element_filter_type *wcD_scratch = &(w_scratch[C*K]);
element_filter_type *wfD_scratch = &(w_scratch[2*C*K]);
element_filter_type *woD_scratch = &(w_scratch[3*C*K]);
element_filter_type *riD_scratch = &(r_scratch[0]);
element_filter_type *rcD_scratch = &(r_scratch[K*K]);
element_filter_type *rfD_scratch = &(r_scratch[2*K*K]);
element_filter_type *roD_scratch = &(r_scratch[3*K*K]);
element_output_type *bi  = &(b[0]);
element_output_type *bd  = &(b[K]);
element_output_type *bf  = &(b[2*K]);
element_output_type *bo  = &(b[3*K]);
LIBXSMM_VLA_DECL(3, element_input_type,  x, xt, N, C);
LIBXSMM_VLA_DECL(2, element_input_type,  cp, csp, K);
LIBXSMM_VLA_DECL(2, element_input_type,  hp, hpD, K);
LIBXSMM_VLA_DECL(4, element_filter_type, wi, wiD_scratch, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, wf, wfD_scratch, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, wo, woD_scratch, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, wc, wcD_scratch, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, ri, riD_scratch, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, rf, rfD_scratch, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, ro, roD_scratch, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, rc, rcD_scratch, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(2, element_filter_type, wi_ck, wiD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, wf_ck, wfD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, wo_ck, woD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, wc_ck, wcD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, ri_ck, riD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, rf_ck, rfD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, ro_ck, roD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, rc_ck, rcD, 4*K);
LIBXSMM_VLA_DECL(3, element_output_type, cs, cst, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, h, ht, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, i, it, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, f, ft, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, o, ot, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, ci, cit, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, co, cot, N, K);
/* define batch-reduce gemm kernels */
const libxsmm_smmfunction_reducebatch batchreduce_kernela = libxsmm_smmdispatch_reducebatch( bk, bn, bc, &bk, &C, &K, NULL, NULL, NULL );
const libxsmm_smmfunction_reducebatch batchreduce_kernelb = libxsmm_smmdispatch_reducebatch( bk, bn, bk, &bk, &K, &K, NULL, NULL, NULL );
/* Auxiliary arrays for batch-reduce gemms  */
const element_filter_type *A_array[1024];
const element_input_type  *B_array[1024];
element_output_type *cps_ptr = NULL;

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

/* number of tasks that could be run in parallel for C and K blocks*/
const libxsmm_blasint work_ck = (C/bc) * (K/bk);
/* compute chunk size */
const libxsmm_blasint chunksize_ck = (work_ck % (libxsmm_blasint)handle->desc.threads == 0) ? (work_ck / (libxsmm_blasint)handle->desc.threads) : ((work_ck / (libxsmm_blasint)handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const libxsmm_blasint thr_begin_ck = (ltid * chunksize_ck < work_ck) ? (ltid * chunksize_ck) : work_ck;
const libxsmm_blasint thr_end_ck = ((ltid + 1) * chunksize_ck < work_ck) ? ((ltid + 1) * chunksize_ck) : work_ck;

/* number of tasks that could be run in parallel for K and K blocks*/
const libxsmm_blasint work_kk = (K/bk) * (K/bk);
/* compute chunk size */
const libxsmm_blasint chunksize_kk = (work_kk % (libxsmm_blasint)handle->desc.threads == 0) ? (work_kk / (libxsmm_blasint)handle->desc.threads) : ((work_kk / (libxsmm_blasint)handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const libxsmm_blasint thr_begin_kk = (ltid * chunksize_kk < work_kk) ? (ltid * chunksize_kk) : work_kk;
const libxsmm_blasint thr_end_kk = ((ltid + 1) * chunksize_kk < work_kk) ? ((ltid + 1) * chunksize_kk) : work_kk;

#ifdef PROFILE
__int64_t eltwise_start, eltwise_end, eltwise_cycles = 0, gemm_start, gemm_end, gemm_cycles = 0, reformat_start, reformat_end, reformat_cycles = 0;
float total_time = 0.0;
#endif

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, (int)ltid);

/* Blocking reduction domain if it is too large */
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
CB_BLOCKS = cBlocks/BF;
KB_BLOCKS = kBlocks/BF;

/* Upfront reformating of W and R */
/* reformat W */
#ifdef PROFILE
if (ltid == 0) reformat_start = _rdtsc();
#endif
for (ikic = thr_begin_ck; ikic < thr_end_ck; ++ikic ) {
  ic = (ikic / (K/bk));
  ik = (ikic % (K/bk));
  for (jk = 0; jk < bk; ++jk) {
    for (jc = 0; jc < bc; ++jc) {
      LIBXSMM_VLA_ACCESS(4, wi, ik, ic, jc, jk, cBlocks, bc, bk) =  LIBXSMM_VLA_ACCESS(2, wi_ck, ic*bc+jc, ik*bk+jk, 4*K);
      LIBXSMM_VLA_ACCESS(4, wc, ik, ic, jc, jk, cBlocks, bc, bk) =  LIBXSMM_VLA_ACCESS(2, wc_ck, ic*bc+jc, ik*bk+jk, 4*K);
      LIBXSMM_VLA_ACCESS(4, wf, ik, ic, jc, jk, cBlocks, bc, bk) =  LIBXSMM_VLA_ACCESS(2, wf_ck, ic*bc+jc, ik*bk+jk, 4*K);
      LIBXSMM_VLA_ACCESS(4, wo, ik, ic, jc, jk, cBlocks, bc, bk) =  LIBXSMM_VLA_ACCESS(2, wo_ck, ic*bc+jc, ik*bk+jk, 4*K);
    }
  }
}

/* reformat R */
for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
  ik = (ikic / (K/bk));
  ic = (ikic % (K/bk));
  for (jk = 0; jk < bk; ++jk) {
    for (jc = 0; jc < bk; ++jc) {
      LIBXSMM_VLA_ACCESS(4, ri, ik, ic, jc, jk, kBlocks, bk, bk) =  LIBXSMM_VLA_ACCESS(2, ri_ck, ic*bk+jc, ik*bk+jk, 4*K);
      LIBXSMM_VLA_ACCESS(4, rc, ik, ic, jc, jk, kBlocks, bk, bk) =  LIBXSMM_VLA_ACCESS(2, rc_ck, ic*bk+jc, ik*bk+jk, 4*K);
      LIBXSMM_VLA_ACCESS(4, rf, ik, ic, jc, jk, kBlocks, bk, bk) =  LIBXSMM_VLA_ACCESS(2, rf_ck, ic*bk+jc, ik*bk+jk, 4*K);
      LIBXSMM_VLA_ACCESS(4, ro, ik, ic, jc, jk, kBlocks, bk, bk) =  LIBXSMM_VLA_ACCESS(2, ro_ck, ic*bk+jc, ik*bk+jk, 4*K);
    }
  }
}

libxsmm_barrier_wait(handle->barrier, (int)ltid);
#ifdef PROFILE
if (ltid == 0) {
  reformat_end = _rdtsc();
  reformat_cycles = reformat_end - reformat_start;
}
#endif

/* All data is in column-major format */
for (j = 0; j < t; ++j) {
  /* let's run the cell in blocks for good locality */
  /* Block reduction loop if requested */
#ifdef PROFILE
  if (ltid == 0) gemm_start = _rdtsc();
#endif
  for (CB = 0; CB < BF; CB++) {
    for (inik = thr_begin; inik < thr_end; ++inik ) {
      in = (inik % (N/bn))*bn;
      ikb = inik / (N/bn);
      ik = ikb*bk;
      /* initialize i with bi */
      if (CB == 0) libxsmm_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &bi[ik] );
      /* i += W.x */
      for (icb = 0, ic = 0; icb < CB_BLOCKS; ic += bc, icb++) {
        A_array[icb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, wi, ikb, icb + CB*CB_BLOCKS, 0, 0, cBlocks, bc, bk);
        B_array[icb] = (element_input_type*)  &LIBXSMM_VLA_ACCESS(3, x, j, in, ic + CB*CB_BLOCKS*bc, N, C);
      }
      /* Reduce batch gemm call  */
      blocks = CB_BLOCKS;
      batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &blocks);

      /* i += R.h */
      if (0 == j) {
        for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
          A_array[icb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, ri, ikb, icb + CB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
          B_array[icb] = (element_input_type*)  &LIBXSMM_VLA_ACCESS(2, hp, in, ic + CB*KB_BLOCKS*bk, K);
        }
      } else {
        for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
          A_array[icb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, ri, ikb, icb + CB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
          B_array[icb] = (element_input_type*)  &LIBXSMM_VLA_ACCESS(3, h, j-1, in, ic + CB*KB_BLOCKS*bk, N, K);
        }
      }
      /* Reduce batch gemm call  */
      blocks = KB_BLOCKS;
      batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &blocks);

      /* initialize ci with bd */
      if (CB == 0) libxsmm_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &bd[ik] );
      /* ci += W.x */
      for (icb = 0, ic = 0; icb < CB_BLOCKS; ic += bc, icb++) {
        A_array[icb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, wc, ikb, icb + CB*CB_BLOCKS, 0, 0, cBlocks, bc, bk);
        B_array[icb] = (element_input_type*)  &LIBXSMM_VLA_ACCESS(3, x, j, in, ic + CB*CB_BLOCKS*bc, N, C);
      }
      /* Reduce batch gemm call  */
      blocks = CB_BLOCKS;
      batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &blocks);

      /* ci += R.h */
      if (0 == j) {
        for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
          A_array[icb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, rc, ikb, icb + CB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
          B_array[icb] = (element_input_type*)  &LIBXSMM_VLA_ACCESS(2, hp, in, ic + CB*KB_BLOCKS*bk, K);
        }
      } else {
        for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
          A_array[icb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, rc, ikb, icb + CB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
          B_array[icb] = (element_input_type*)  &LIBXSMM_VLA_ACCESS(3, h, j-1, in, ic + CB*KB_BLOCKS*bk, N, K);
        }
      }
      /* Reduce batch gemm call  */
      blocks = KB_BLOCKS;
      batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &blocks);

      /* initialize f with (bf + forget_bias) */
      if (CB == 0)  libxsmm_internal_matrix_bcst_colvector_const_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &bf[ik], handle->forget_bias );
      /* f += W.x */
      for (icb = 0, ic = 0; icb < CB_BLOCKS; ic += bc, icb++) {
        A_array[icb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, wf, ikb, icb + CB*CB_BLOCKS, 0, 0, cBlocks, bc, bk);
        B_array[icb] = (element_input_type*)  &LIBXSMM_VLA_ACCESS(3, x, j, in, ic + CB*CB_BLOCKS*bc, N, C);
      }
      /* Reduce batch gemm call  */
      blocks = CB_BLOCKS;
      batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &blocks);

      /* f += R.h */
      if (0 == j) {
        for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
          A_array[icb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, rf, ikb, icb + CB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
          B_array[icb] = (element_input_type*)  &LIBXSMM_VLA_ACCESS(2, hp, in, ic + CB*KB_BLOCKS*bk, K);
        }
      } else {
        for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
          A_array[icb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, rf, ikb, icb + CB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
          B_array[icb] = (element_input_type*)  &LIBXSMM_VLA_ACCESS(3, h, j-1, in, ic + CB*KB_BLOCKS*bk, N, K);
        }
      }
      /* Reduce batch gemm call  */
      blocks = KB_BLOCKS;
      batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &blocks);

      /* initialize o with bo */
      if (CB == 0) libxsmm_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &bo[ik] );
      /* o += W.x */
      for (icb = 0, ic = 0; icb < CB_BLOCKS; ic += bc, icb++) {
        A_array[icb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, wo, ikb, icb + CB*CB_BLOCKS, 0, 0, cBlocks, bc, bk);
        B_array[icb] = (element_input_type*)  &LIBXSMM_VLA_ACCESS(3, x, j, in, ic + CB*CB_BLOCKS*bc, N, C);
      }
      /* Reduce batch gemm call  */
      blocks = CB_BLOCKS;
      batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &blocks);

      /* o += R.h */
      if (0 == j) {
        for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
          A_array[icb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, ro, ikb, icb + CB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
          B_array[icb] = (element_input_type*)  &LIBXSMM_VLA_ACCESS(2, hp, in, ic + CB*KB_BLOCKS*bk, K);
        }
      } else {
        for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
          A_array[icb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, ro, ikb, icb + CB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
          B_array[icb] = (element_input_type*)  &LIBXSMM_VLA_ACCESS(3, h, j-1, in, ic + CB*KB_BLOCKS*bk, N, K);
        }
      }
      /* Reduce batch gemm call  */
      blocks = KB_BLOCKS;
      batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &blocks);
    }
  }
#ifdef PROFILE
  if (ltid == 0) {
    gemm_end = _rdtsc();
    gemm_cycles += gemm_end-gemm_start;
    eltwise_start = gemm_end;
  }
#endif
  for (inik = thr_begin; inik < thr_end; ++inik ) {
    in = (inik % (N/bn))*bn;
    ikb = inik / (N/bn);
    ik = ikb*bk;
    cps_ptr = (j == 0) ? (element_output_type*) &LIBXSMM_VLA_ACCESS(2, cp, in, ik, K) : (element_output_type*) &LIBXSMM_VLA_ACCESS(3, cs, j-1, in, ik, N, K) ;
    /* Compute i, ci, f, o, cs, co and h */
#if defined(LIBXSMM_INTRINSICS_AVX512) 
    if (bk % 16 == 0 && bc % 16 == 0) {
      libxsmm_internal_compute_o_i_f_ci_cs_co_h_ld(bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), cps_ptr, &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, h, j, in, ik, N, K));
    } else {
      libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K) );
      libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K) );
      libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K) );
      libxsmm_internal_matrix_tanh_ld(    bk, bn, K, &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K) );
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), cps_ptr, &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K) );
      libxsmm_internal_matrix_eltwise_fma_ld(  bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K) );
      libxsmm_internal_matrix_tanh_ld(         bk, bn, K, &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K) );
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K),  &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, h, j, in, ik, N, K) );
    }
#else
    libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K) );
    libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K) );
    libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K) );
    libxsmm_internal_matrix_tanh_ld(    bk, bn, K, &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K) );
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), cps_ptr, &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K) );
    libxsmm_internal_matrix_eltwise_fma_ld(  bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K) );
    libxsmm_internal_matrix_tanh_ld(         bk, bn, K, &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K) );
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K),  &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, h, j, in, ik, N, K) );
#endif
  }
#ifdef PROFILE
  if (ltid == 0) {
    eltwise_end = _rdtsc();
    eltwise_cycles += eltwise_end-eltwise_start;
  }
#endif

  libxsmm_barrier_wait(handle->barrier, (int)ltid);
}
#ifdef PROFILE
if (ltid == 0) {
  printf("----- PROFILING LSTM FWD (N = %d, C = %d, K = %d, bn = %d. bc = %d, bk = %d)----\n", N, C, K, bn, bc, bk );
  total_time = (gemm_cycles+eltwise_cycles+reformat_cycles)/(2.5 * 1e9)*1000.0f;
  printf("Elementwise time is %f ms (%.2f%%)\n", eltwise_cycles/(2.5 * 1e9)*1000.0f, eltwise_cycles/(2.5 * 1e9)*1000.0f*100.0/total_time );
  printf("Reformat weights time is %f ms (%.2f%%)\n", reformat_cycles/(2.5 * 1e9)*1000.0f, reformat_cycles/(2.5 * 1e9)*1000.0f*100.0/total_time );
  printf("GEMM time is %f ms (%.2f%%) at %f GFLOPS\n\n", gemm_cycles/(2.5 * 1e9)*1000.0f, gemm_cycles/(2.5 * 1e9)*1000.0f*100.0/total_time, t*2.0*N*K*K*4*2.0/1e9/(gemm_cycles/(2.5 * 1e9)));
}
#undef PROFILE
#endif
