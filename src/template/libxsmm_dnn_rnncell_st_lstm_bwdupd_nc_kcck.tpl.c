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

/* helper variables */
libxsmm_blasint j, ik, ikb, in, inb, ic, icb, jk, jb/*jn shadows global variable*/, jc, ek, en, ec;
/* tensor dimensions */
libxsmm_blasint K = handle->desc.K;
libxsmm_blasint N = handle->desc.N;
libxsmm_blasint C = handle->desc.C;
libxsmm_blasint t = handle->desc.t;
libxsmm_blasint bk = handle->bk;
libxsmm_blasint bn = handle->bn;
libxsmm_blasint bc = handle->bc;
libxsmm_blasint K4 = K * 4;
const int cBlocks = C/bc;
const int kBlocks = K/bk;
const int nBlocks = N/bn;
unsigned long long blocks;
/* tensor raw pointers */
element_input_type  *xt    = (element_input_type* )handle->xt->data;
element_input_type *csp   = (element_input_type* )handle->csp->data;
element_input_type *hpD   = (element_input_type* )handle->hp->data;
element_filter_type *w     = (element_filter_type*)handle->w->data;
element_filter_type *r     = (element_filter_type*)handle->r->data;
element_output_type *cst   = (element_output_type*)handle->cst->data;
element_output_type *ht    = handle->ht ? (element_output_type*)handle->ht->data : (element_output_type*)NULL;
element_output_type *it    = (element_output_type*)handle->it->data;
element_output_type *ft    = (element_output_type*)handle->ft->data;
element_output_type *ot    = (element_output_type*)handle->ot->data;
element_output_type *cit   = (element_output_type*)handle->cit->data;
element_output_type *cot   = (element_output_type*)handle->cot->data;
element_input_type  *dxt   = (element_input_type*)handle->dxt->data;
element_input_type  *dcsp  = (element_input_type* )handle->dcsp->data;
element_input_type  *dhpD  = (element_input_type* )handle->dhp->data;
element_filter_type *dw    = (element_filter_type*)handle->dw->data;
element_filter_type *dr    = (element_filter_type*)handle->dr->data;
element_output_type *db    = (element_output_type*)handle->db->data;
element_output_type *dcsD  = (element_output_type*)handle->dcs->data;
element_output_type *dht   = (element_output_type*)handle->dht->data;
element_output_type *diD   = (element_output_type*)handle->scratch_di;
element_output_type *dfD   = (element_output_type*)handle->scratch_df;
element_output_type *doD   = (element_output_type*)handle->scratch_do;
element_output_type *dciD  = (element_output_type*)handle->scratch_dci;
element_output_type *doutD = (element_output_type*)handle->scratch_deltat;
element_output_type *t1D   = (element_output_type*)handle->scratch_t1;
element_output_type *t2D   = (element_output_type*)handle->scratch_t2;
element_input_type  *scratch_xT  = (element_input_type* )handle->scratch_xT;
element_filter_type *scratch_wT  = (element_filter_type*)handle->scratch_wT;
element_filter_type *scratch_rT  = (element_filter_type*)handle->scratch_rT;
element_output_type *scratch_hT  = (element_output_type*)handle->scratch_hT;
element_filter_type *wiD   = &(w[0]);
element_filter_type *wcD   = &(w[C*K]);
element_filter_type *wfD   = &(w[2*C*K]);
element_filter_type *woD   = &(w[3*C*K]);
element_filter_type *riD   = &(r[0]);
element_filter_type *rcD   = &(r[K*K]);
element_filter_type *rfD   = &(r[2*K*K]);
element_filter_type *roD   = &(r[3*K*K]);
element_filter_type *dwiD  = &(dw[0]);
element_filter_type *dwcD  = &(dw[C*K]);
element_filter_type *dwfD  = &(dw[2*C*K]);
element_filter_type *dwoD  = &(dw[3*C*K]);
element_filter_type *driD  = &(dr[0]);
element_filter_type *drcD  = &(dr[K*K]);
element_filter_type *drfD  = &(dr[2*K*K]);
element_filter_type *droD  = &(dr[3*K*K]);
element_output_type *dbi   = &(db[0]);
element_output_type *dbc   = &(db[K]);
element_output_type *dbf   = &(db[2*K]);
element_output_type *dbo   = &(db[3*K]);
element_filter_type *scratch_wiT = &(scratch_wT[0]);
element_filter_type *scratch_wcT = &(scratch_wT[C*K]);
element_filter_type *scratch_wfT = &(scratch_wT[2*C*K]);
element_filter_type *scratch_woT = &(scratch_wT[3*C*K]);
element_filter_type *scratch_riT = &(scratch_rT[0]);
element_filter_type *scratch_rcT = &(scratch_rT[K*K]);
element_filter_type *scratch_rfT = &(scratch_rT[2*K*K]);
element_filter_type *scratch_roT = &(scratch_rT[3*K*K]);
/* multidimensional arrays */
LIBXSMM_VLA_DECL(3, element_input_type,  x, xt, N, C);
LIBXSMM_VLA_DECL(2, element_input_type,  cp, csp, K);
LIBXSMM_VLA_DECL(2, element_input_type,  hp, hpD, K);
LIBXSMM_VLA_DECL(4, element_filter_type, wi, wiD, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, wf, wfD, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, wo, woD, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, wc, wcD, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, ri, riD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, rf, rfD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, ro, roD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, rc, rcD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(3, element_output_type, cs, cst, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, h, ht, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, i, it, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, f, ft, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, o, ot, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, ci, cit, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, co, cot, N, K);
LIBXSMM_VLA_DECL(3, element_input_type,  dx, dxt, N, C);
LIBXSMM_VLA_DECL(2, element_input_type,  dcp, dcsp, K);
LIBXSMM_VLA_DECL(2, element_input_type,  dhp, dhpD, K);
LIBXSMM_VLA_DECL(4, element_filter_type, dwi, dwiD, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, dwf, dwfD, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, dwo, dwoD, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, dwc, dwcD, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, dri, driD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, drf, drfD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, dro, droD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, drc, drcD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(2, element_output_type, dcs, dcsD, K);
LIBXSMM_VLA_DECL(3, element_output_type, dh, dht, N, K);
LIBXSMM_VLA_DECL(2, element_output_type, di, diD, K);
LIBXSMM_VLA_DECL(2, element_output_type, df, dfD, K);
LIBXSMM_VLA_DECL(2, element_output_type, dp, doD, K);
LIBXSMM_VLA_DECL(2, element_output_type, dci, dciD, K);
LIBXSMM_VLA_DECL(2, element_output_type, dout, doutD, K);
LIBXSMM_VLA_DECL(2, element_output_type, t1, t1D, K);
LIBXSMM_VLA_DECL(2, element_output_type, t2, t2D, K);
LIBXSMM_VLA_DECL(2, element_input_type,  xT, scratch_xT, N);
LIBXSMM_VLA_DECL(4, element_filter_type, wiT, scratch_wiT, kBlocks, bk, bc);
LIBXSMM_VLA_DECL(4, element_filter_type, wcT, scratch_wcT, kBlocks, bk, bc);
LIBXSMM_VLA_DECL(4, element_filter_type, wfT, scratch_wfT, kBlocks, bk, bc);
LIBXSMM_VLA_DECL(4, element_filter_type, woT, scratch_woT, kBlocks, bk, bc);
LIBXSMM_VLA_DECL(4, element_filter_type, riT, scratch_riT, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, rcT, scratch_rcT, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, rfT, scratch_rfT, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, roT, scratch_roT, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(2, element_output_type, hT, scratch_hT, N);
/* define batch-reduce gemm kernels */
const libxsmm_smmfunction_reducebatch batchreduce_kernela = libxsmm_smmdispatch_reducebatch( bc, bn, bk, &bc, &K, &C, NULL, NULL, NULL);
const libxsmm_smmfunction_reducebatch batchreduce_kernelb = libxsmm_smmdispatch_reducebatch( bk, bk, bn, &K, &N, &bk, NULL, NULL, NULL);
const libxsmm_smmfunction_reducebatch batchreduce_kernelc = libxsmm_smmdispatch_reducebatch( bk, bc, bn, &K, &N, &bk, NULL, NULL, NULL);
const libxsmm_smmfunction_reducebatch batchreduce_kerneld = libxsmm_smmdispatch_reducebatch( bk, bn, bk, &bk, &K, &K, NULL, NULL, NULL);

/* Auxiliary arrays for batch-reduce gemm calls  */
const element_filter_type *A_array[1024];
const element_output_type *B_array[1024];

/* computing first logical thread */
const libxsmm_blasint ltid = (libxsmm_blasint)tid - (libxsmm_blasint)start_thread;

/* number of tasks that could be run in parallel for N and K blocks*/
const libxsmm_blasint work_nk = (N/bn) * (K/bk);
/* compute chunk size */
const libxsmm_blasint chunksize_nk = (work_nk % (libxsmm_blasint)handle->desc.threads == 0) ? (work_nk / (libxsmm_blasint)handle->desc.threads) : ((work_nk / (libxsmm_blasint)handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const libxsmm_blasint thr_begin_nk = (ltid * chunksize_nk < work_nk) ? (ltid * chunksize_nk) : work_nk;
const libxsmm_blasint thr_end_nk = ((ltid + 1) * chunksize_nk < work_nk) ? ((ltid + 1) * chunksize_nk) : work_nk;

/* number of tasks that could be run in parallel for N and C blocks*/
const libxsmm_blasint work_nc = (N/bn) * (C/bc);
/* compute chunk size */
const libxsmm_blasint chunksize_nc = (work_nc % (libxsmm_blasint)handle->desc.threads == 0) ? (work_nc / (libxsmm_blasint)handle->desc.threads) : ((work_nc / (libxsmm_blasint)handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const libxsmm_blasint thr_begin_nc = (ltid * chunksize_nc < work_nc) ? (ltid * chunksize_nc) : work_nc;
const libxsmm_blasint thr_end_nc = ((ltid + 1) * chunksize_nc < work_nc) ? ((ltid + 1) * chunksize_nc) : work_nc;

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

/* number of tasks that could be run in parallel for K blocks*/
/* compute chunk size */
const libxsmm_blasint chunksize_k = (K % (libxsmm_blasint)handle->desc.threads == 0) ? (K / (libxsmm_blasint)handle->desc.threads) : ((K / (libxsmm_blasint)handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const libxsmm_blasint thr_begin_k = (ltid * chunksize_k < K) ? (ltid * chunksize_k) : K;
const libxsmm_blasint thr_end_k = ((ltid + 1) * chunksize_k < K) ? ((ltid + 1) * chunksize_k) : K;

int k_tasks = K/16;
int k_chunksize = (k_tasks % (libxsmm_blasint)handle->desc.threads == 0) ? (k_tasks / (libxsmm_blasint)handle->desc.threads) : ((k_tasks / (libxsmm_blasint)handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const libxsmm_blasint k_thr_begin = (ltid * k_chunksize * 16 < K) ? (ltid * k_chunksize * 16) : K;
const libxsmm_blasint k_thr_end = ((ltid + 1) * k_chunksize * 16 < K) ? ((ltid + 1) * k_chunksize * 16) : K;
__m512 dbi_sum, dbf_sum, dbo_sum, dbc_sum;
libxsmm_blasint ikic, inic, inik, icin, ikin;

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, (int)ltid);

/* initialization is done at the beginning */
if ( (LIBXSMM_DNN_COMPUTE_KIND_BWD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
  libxsmm_internal_matrix_zero(N*C*t, dxt, start_thread, tid, handle->desc.threads);
}
if ( (LIBXSMM_DNN_COMPUTE_KIND_UPD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
  libxsmm_internal_matrix_zero(C*K*4, dw,  start_thread, tid, handle->desc.threads);
  libxsmm_internal_matrix_zero(K*K*4, dr,  start_thread, tid, handle->desc.threads);
  libxsmm_internal_matrix_zero(K*4,   db,  start_thread, tid, handle->desc.threads);
}

/* transpose W */
for (ikic = thr_begin_ck; ikic < thr_end_ck; ++ikic ) {
  ic = (ikic / (K/bk));
  ik = (ikic % (K/bk));
  for (jk = 0; jk < bk; ++jk) {
    for (jc = 0; jc < bc; ++jc) {
      LIBXSMM_VLA_ACCESS(4, wiT, ic, ik, jk, jc, kBlocks, bk, bc) =  LIBXSMM_VLA_ACCESS(4, wi, ik, ic, jc, jk, cBlocks, bc, bk);
      LIBXSMM_VLA_ACCESS(4, wcT, ic, ik, jk, jc, kBlocks, bk, bc) =  LIBXSMM_VLA_ACCESS(4, wc, ik, ic, jc, jk, cBlocks, bc, bk);
      LIBXSMM_VLA_ACCESS(4, wfT, ic, ik, jk, jc, kBlocks, bk, bc) =  LIBXSMM_VLA_ACCESS(4, wf, ik, ic, jc, jk, cBlocks, bc, bk);
      LIBXSMM_VLA_ACCESS(4, woT, ic, ik, jk, jc, kBlocks, bk, bc) =  LIBXSMM_VLA_ACCESS(4, wo, ik, ic, jc, jk, cBlocks, bc, bk);
    }
  }
}

/* transpose R */
for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
  ik = (ikic / (K/bk));
  ic = (ikic % (K/bk));
  for (jk = 0; jk < bk; ++jk) {
    for (jc = 0; jc < bk; ++jc) {
      LIBXSMM_VLA_ACCESS(4, riT, ic, ik, jk, jc, kBlocks, bk, bk) =  LIBXSMM_VLA_ACCESS(4, ri, ik, ic, jc, jk, kBlocks, bk, bk);
      LIBXSMM_VLA_ACCESS(4, rcT, ic, ik, jk, jc, kBlocks, bk, bk) =  LIBXSMM_VLA_ACCESS(4, rc, ik, ic, jc, jk, kBlocks, bk, bk);
      LIBXSMM_VLA_ACCESS(4, rfT, ic, ik, jk, jc, kBlocks, bk, bk) =  LIBXSMM_VLA_ACCESS(4, rf, ik, ic, jc, jk, kBlocks, bk, bk);
      LIBXSMM_VLA_ACCESS(4, roT, ic, ik, jk, jc, kBlocks, bk, bk) =  LIBXSMM_VLA_ACCESS(4, ro, ik, ic, jc, jk, kBlocks, bk, bk);
    }
  }
}

for (j = t-1; j >= 0; --j) {
  /* let's run the cell in blocks for good locality */
  for (inik = thr_begin_nk; inik < thr_end_nk; ++inik ) {
    in = (inik % (N/bn))*bn;
    ik = (inik / (N/bn))*bk;

    /* compute dhp */
    if (j == t-1) {
      libxsmm_internal_matrix_copy_ld(  bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dh, t-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K) );
    } else {
      libxsmm_internal_matrix_add_ld(   bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K), &LIBXSMM_VLA_ACCESS(3, dh, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K) );
    }
    /* compute dcp */
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K), &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    libxsmm_internal_matrix_complement_square_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    if (j == t-1) {
      libxsmm_internal_matrix_add_ld(        bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dcs, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K) );
    } else {
      libxsmm_internal_matrix_add_ld(        bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K) );
    }
    /* compute dci */
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K), &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    libxsmm_internal_matrix_complement_square_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, dci, in, ik, K) );
    /* compute di */
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K), &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    libxsmm_internal_matrix_complement_ld(        bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, di, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, di, in, ik, K), &LIBXSMM_VLA_ACCESS(2, di, in, ik, K) );
    /* compute df */
    if (j == 0) {
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K), &LIBXSMM_VLA_ACCESS(2, cp, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    } else {
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K), &LIBXSMM_VLA_ACCESS(3, cs, j-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    }
    libxsmm_internal_matrix_complement_ld(   bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, df, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, df, in, ik, K), &LIBXSMM_VLA_ACCESS(2, df, in, ik, K) );
    /* compute dp */
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K), &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    libxsmm_internal_matrix_complement_ld(   bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, dp, in, ik, K) );
    /* update dcp */
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K), &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K) );
  }

  if ( (LIBXSMM_DNN_COMPUTE_KIND_UPD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
    /* transpose xt for current timestep */
    for (icin = thr_begin_nc; icin < thr_end_nc; ++icin ) {
      in = (icin / (C/bc))*bn;
      ic = (icin % (C/bc))*bc;

      for (jc = 0; jc < bc; ++jc) {
        for (jb = 0; jb < bn; ++jb) {
          en = in + jb;
          ec = ic + jc;
          LIBXSMM_VLA_ACCESS(2, xT, ec, en, N) =  LIBXSMM_VLA_ACCESS(3, x, j, en, ec, N, C);
        }
      }
    }

    /* transpose ht for current timestep */
    if (j == 0) {
      for (ikin = thr_begin_nk; ikin < thr_end_nk; ++ikin ) {
        in = (ikin / (K/bk))*bn;
        ik = (ikin % (K/bk))*bk;

        for (jk = 0; jk < bk; ++jk) {
          for (jb = 0; jb < bn; ++jb) {
            en = in + jb;
            ek = ik + jk;
            LIBXSMM_VLA_ACCESS(2, hT, ek, en, N) =  LIBXSMM_VLA_ACCESS(2, hp, en, ek, K);
          }
        }
      }
    } else {
      for (ikin = thr_begin_nk; ikin < thr_end_nk; ++ikin ) {
        in = (ikin / (K/bk))*bn;
        ik = (ikin % (K/bk))*bk;

        for (jk = 0; jk < bk; ++jk) {
          for (jb = 0; jb < bn; ++jb) {
            en = in + jb;
            ek = ik + jk;
            LIBXSMM_VLA_ACCESS(2, hT, ek, en, N) =  LIBXSMM_VLA_ACCESS(3, h, j-1, en, ek, N, K);
          }
        }
      }
    }
  }

  libxsmm_barrier_wait(handle->barrier, (int)ltid);

  for (inik = thr_begin_nk; inik < thr_end_nk; ++inik ) {
    in = (inik % (N/bn))*bn;
    ikb = inik / (N/bn);
    ik = ikb*bk;

    if (j > 0) {
      /* dout += R^T * difoc */
      libxsmm_internal_matrix_zero_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K) );
      for (ic = 0, icb = 0; ic < K; ic += bk, icb++) {
        A_array[icb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, riT, ikb, icb, 0, 0, kBlocks, bk, bk);
        B_array[icb] = (element_output_type*) &LIBXSMM_VLA_ACCESS(2, di,  in, ic, K);
      }
      /* Reduce batch gemm call  */
      blocks = kBlocks;
      batchreduce_kerneld(A_array, B_array, &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K) , &blocks);

      for (ic = 0, icb = 0; ic < K; ic += bk, icb++) {
        A_array[icb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, rfT, ikb, icb, 0, 0, kBlocks, bk, bk);
        B_array[icb] = (element_output_type*) &LIBXSMM_VLA_ACCESS(2, df,  in, ic, K);
      }
      /* Reduce batch gemm call  */
      batchreduce_kerneld(A_array, B_array, &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K) , &blocks);

      for (ic = 0, icb = 0; ic < K; ic += bk, icb++) {
        A_array[icb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, roT, ikb, icb, 0, 0, kBlocks, bk, bk);
        B_array[icb] = (element_output_type*) &LIBXSMM_VLA_ACCESS(2, dp,  in, ic, K);
      }
      /* Reduce batch gemm call  */
      batchreduce_kerneld(A_array, B_array, &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K) , &blocks);

      for (ic = 0, icb = 0; ic < K; ic += bk, icb++) {
        A_array[icb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, rcT, ikb, icb, 0, 0, kBlocks, bk, bk);
        B_array[icb] = (element_output_type*) &LIBXSMM_VLA_ACCESS(2, dci,  in, ic, K);
      }
      /* Reduce batch gemm call  */
      batchreduce_kerneld(A_array, B_array, &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K) , &blocks);
    } else {
      libxsmm_internal_matrix_zero_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dhp, in, ik, K) );
      for (ic = 0, icb = 0; ic < K; ic += bk, icb++) {
        A_array[icb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, riT, ikb, icb, 0, 0, kBlocks, bk, bk);
        B_array[icb] = (element_output_type*) &LIBXSMM_VLA_ACCESS(2, di,  in, ic, K);
      }
      /* Reduce batch gemm call  */
      blocks = kBlocks;
      batchreduce_kerneld(A_array, B_array, &LIBXSMM_VLA_ACCESS(2, dhp, in, ik, K) , &blocks);

      for (ic = 0, icb = 0; ic < K; ic += bk, icb++) {
        A_array[icb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, rfT, ikb, icb, 0, 0, kBlocks, bk, bk);
        B_array[icb] = (element_output_type*) &LIBXSMM_VLA_ACCESS(2, df,  in, ic, K);
      }
      /* Reduce batch gemm call  */
      batchreduce_kerneld(A_array, B_array, &LIBXSMM_VLA_ACCESS(2, dhp, in, ik, K) , &blocks);

      for (ic = 0, icb = 0; ic < K; ic += bk, icb++) {
        A_array[icb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, roT, ikb, icb, 0, 0, kBlocks, bk, bk);
        B_array[icb] = (element_output_type*) &LIBXSMM_VLA_ACCESS(2, dp,  in, ic, K);
      }
      /* Reduce batch gemm call  */
      batchreduce_kerneld(A_array, B_array, &LIBXSMM_VLA_ACCESS(2, dhp, in, ik, K) , &blocks);

      for (ic = 0, icb = 0; ic < K; ic += bk, icb++) {
        A_array[icb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, rcT, ikb, icb, 0, 0, kBlocks, bk, bk);
        B_array[icb] = (element_output_type*) &LIBXSMM_VLA_ACCESS(2, dci,  in, ic, K);
      }
      /* Reduce batch gemm call  */
      batchreduce_kerneld(A_array, B_array, &LIBXSMM_VLA_ACCESS(2, dhp, in, ik, K) , &blocks);
    }
  }

  if ( (LIBXSMM_DNN_COMPUTE_KIND_BWD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
    /* dx = W^T * difoc */
    for (inic = thr_begin_nc; inic < thr_end_nc; ++inic ) {
      in = (inic % (N/bn))*bn;
      icb = inic / (N/bn);
      ic = icb*bc;

      for (ik = 0, ikb = 0; ik < K; ik += bk, ikb++) {
        A_array[ikb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, wiT, icb, ikb, 0, 0, kBlocks, bk, bc);
        B_array[ikb] = (element_output_type*) &LIBXSMM_VLA_ACCESS(2, di,  in, ik, K);
      }
      /* Reduce batch gemm call  */
      blocks = kBlocks;
      batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, dx, j, in, ic, N, C)  , &blocks);

      for (ik = 0, ikb = 0; ik < K; ik += bk, ikb++) {
        A_array[ikb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, wfT, icb, ikb, 0, 0, kBlocks, bk, bc);
        B_array[ikb] = (element_output_type*) &LIBXSMM_VLA_ACCESS(2, df,  in, ik, K);
      }
      /* Reduce batch gemm call  */
      batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, dx, j, in, ic, N, C)  , &blocks);

      for (ik = 0, ikb = 0; ik < K; ik += bk, ikb++) {
        A_array[ikb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, woT, icb, ikb, 0, 0, kBlocks, bk, bc);
        B_array[ikb] = (element_output_type*) &LIBXSMM_VLA_ACCESS(2, dp,  in, ik, K);
      }
      /* Reduce batch gemm call  */
      batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, dx, j, in, ic, N, C)  , &blocks);

      for (ik = 0, ikb = 0; ik < K; ik += bk, ikb++) {
        A_array[ikb] = (element_filter_type*) &LIBXSMM_VLA_ACCESS(4, wcT, icb, ikb, 0, 0, kBlocks, bk, bc);
        B_array[ikb] = (element_output_type*) &LIBXSMM_VLA_ACCESS(2, dci,  in, ik, K);
      }
      /* Reduce batch gemm call  */
      batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, dx, j, in, ic, N, C)  , &blocks);
    }
  }

  if ( (LIBXSMM_DNN_COMPUTE_KIND_UPD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
    /* gradient bias */
    for (ik = k_thr_begin; ik < k_thr_end; ik += 16) {
      dbi_sum = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&dbi[ik]);
      dbf_sum = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&dbf[ik]);
      dbo_sum = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&dbo[ik]);
      dbc_sum = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&dbc[ik]);
      for (in = 0; in < N; in++) {
        dbi_sum = _mm512_add_ps(dbi_sum, LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(2, di,  in, ik, K)));
        dbf_sum = _mm512_add_ps(dbf_sum, LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(2, df,  in, ik, K)));
        dbo_sum = _mm512_add_ps(dbo_sum, LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(2, dp,  in, ik, K)));
        dbc_sum = _mm512_add_ps(dbc_sum, LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(2, dci,  in, ik, K)));
      }
      _mm512_store_ps(&dbi[ik], dbi_sum);
      _mm512_store_ps(&dbf[ik], dbf_sum);
      _mm512_store_ps(&dbo[ik], dbo_sum);
      _mm512_store_ps(&dbc[ik], dbc_sum);
    }

    /* dr = difoc * h^T */
    for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
      icb = ikic / (K/bk);
      ic = icb*bk;
      ikb = ikic % (K/bk);
      ik = ikb*bk;

      for (in = 0, inb = 0; in < N; in += bn, inb++) {
        A_array[inb] = &LIBXSMM_VLA_ACCESS(2, di,  in, ik, K);
        B_array[inb] = &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N);
      }
      blocks = nBlocks;
      batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dri, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

      for (in = 0, inb = 0; in < N; in += bn, inb++) {
        A_array[inb] = &LIBXSMM_VLA_ACCESS(2, df,  in, ik, K);
        B_array[inb] = &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N);
      }
      batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, drf, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

      for (in = 0, inb = 0; in < N; in += bn, inb++) {
        A_array[inb] = &LIBXSMM_VLA_ACCESS(2, dp,  in, ik, K);
        B_array[inb] = &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N);
      }
      batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dro, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

      for (in = 0, inb = 0; in < N; in += bn, inb++) {
        A_array[inb] = &LIBXSMM_VLA_ACCESS(2, dci, in, ik, K);
        B_array[inb] = &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N);
      }
      batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, drc, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);
    }

    /* dw = difoc * x^T */
    for (ikic = thr_begin_ck; ikic < thr_end_ck; ++ikic ) {
      icb = ikic / (K/bk);
      ic = icb*bc;
      ikb = ikic % (K/bk);
      ik = ikb*bk;

      for (in = 0, inb = 0; in < N; in += bn, inb++) {
        A_array[inb] = &LIBXSMM_VLA_ACCESS(2, di,  in, ik, K);
        B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
      }
      blocks = nBlocks;
      batchreduce_kernelc(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwi, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

      for (in = 0, inb = 0; in < N; in += bn, inb++) {
        A_array[inb] = &LIBXSMM_VLA_ACCESS(2, df,  in, ik, K);
        B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
      }
      batchreduce_kernelc(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwf, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

      for (in = 0, inb = 0; in < N; in += bn, inb++) {
        A_array[inb] = &LIBXSMM_VLA_ACCESS(2, dp,  in, ik, K);
        B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
      }
      batchreduce_kernelc(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwo, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

      for (in = 0, inb = 0; in < N; in += bn, inb++) {
        A_array[inb] = &LIBXSMM_VLA_ACCESS(2, dci,  in, ik, K);
        B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
      }
      batchreduce_kernelc(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwc, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);
    }
  }

  libxsmm_barrier_wait(handle->barrier, (int)ltid);
}
