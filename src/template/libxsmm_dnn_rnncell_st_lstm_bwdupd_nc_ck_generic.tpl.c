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
libxsmm_blasint j, ik, in, ic, jk, jb/*jn shadows global variable*/, jc, ek, en, ec;
/* tensor dimensions */
libxsmm_blasint K = handle->desc.K;
libxsmm_blasint N = handle->desc.N;
libxsmm_blasint C = handle->desc.C;
libxsmm_blasint t = handle->desc.t;
libxsmm_blasint bk = handle->bk;
libxsmm_blasint bn = handle->bn;
libxsmm_blasint bc = handle->bc;
libxsmm_blasint K4 = K * 4;
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
element_input_type  *dcspt = (element_input_type* )handle->dcspt->data;
element_input_type  *dhpt  = (element_input_type* )handle->dhpt->data;
element_filter_type *dw    = (element_filter_type*)handle->dw->data;
element_filter_type *dr    = (element_filter_type*)handle->dr->data;
element_output_type *db    = (element_output_type*)handle->db->data;
/*element_output_type *dcsD  = handle->dcs ? (element_output_type*)handle->dcs->data : (element_output_type*)NULL;*/
element_output_type *dcsD  = (element_output_type*)handle->dcs->data;
element_output_type *dht   = (element_output_type*)handle->dht->data;
element_output_type *dit   = (element_output_type*)handle->scratch_dit;
element_output_type *dft   = (element_output_type*)handle->scratch_dft;
element_output_type *dot   = (element_output_type*)handle->scratch_dot;
element_output_type *dcit  = (element_output_type*)handle->scratch_dcit;
element_output_type *doutt = (element_output_type*)handle->scratch_deltat;
element_output_type *t1D   = (element_output_type*)handle->scratch_t1;
element_output_type *t2D   = (element_output_type*)handle->scratch_t2;
element_input_type  *scratch_xT  = (element_input_type* )handle->scratch_xT;
element_filter_type *scratch_wT  = (element_filter_type*)handle->scratch_wT;
element_filter_type *scratch_rT  = (element_filter_type*)handle->scratch_rT;
element_output_type *scratch_hT  = (element_output_type*)handle->scratch_hT;
#if 0
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
#endif
element_filter_type *wiD   = &(w[0]);
element_filter_type *wcD   = &(w[K]);
element_filter_type *wfD   = &(w[2*K]);
element_filter_type *woD   = &(w[3*K]);
element_filter_type *riD   = &(r[0]);
element_filter_type *rcD   = &(r[K]);
element_filter_type *rfD   = &(r[2*K]);
element_filter_type *roD   = &(r[3*K]);
element_filter_type *dwiD  = &(dw[0]);
element_filter_type *dwcD  = &(dw[K]);
element_filter_type *dwfD  = &(dw[2*K]);
element_filter_type *dwoD  = &(dw[3*K]);
element_filter_type *driD  = &(dr[0]);
element_filter_type *drcD  = &(dr[K]);
element_filter_type *drfD  = &(dr[2*K]);
element_filter_type *droD  = &(dr[3*K]);
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
LIBXSMM_VLA_DECL(3, element_input_type,  dx, dxt, N, C);
LIBXSMM_VLA_DECL(3, element_input_type,  dcp, dcspt, N, K);
LIBXSMM_VLA_DECL(3, element_input_type,  dhp, dhpt, N, K);
LIBXSMM_VLA_DECL(2, element_filter_type, dwi, dwiD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, dwf, dwfD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, dwo, dwoD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, dwc, dwcD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, dri, driD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, drf, drfD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, dro, droD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, drc, drcD, 4*K);
LIBXSMM_VLA_DECL(2, element_output_type, dcs, dcsD, K);
LIBXSMM_VLA_DECL(3, element_output_type, dh, dht, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, di, dit, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, df, dft, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, dp, dot, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, dci, dcit, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, dout, doutt, N, K);
LIBXSMM_VLA_DECL(2, element_output_type, t1, t1D, K);
LIBXSMM_VLA_DECL(2, element_output_type, t2, t2D, K);
LIBXSMM_VLA_DECL(2, element_input_type,  xT, scratch_xT, N);
LIBXSMM_VLA_DECL(2, element_filter_type, wiT, scratch_wiT, C);
LIBXSMM_VLA_DECL(2, element_filter_type, wcT, scratch_wcT, C);
LIBXSMM_VLA_DECL(2, element_filter_type, wfT, scratch_wfT, C);
LIBXSMM_VLA_DECL(2, element_filter_type, woT, scratch_woT, C);
LIBXSMM_VLA_DECL(2, element_filter_type, riT, scratch_riT, K);
LIBXSMM_VLA_DECL(2, element_filter_type, rcT, scratch_rcT, K);
LIBXSMM_VLA_DECL(2, element_filter_type, rfT, scratch_rfT, K);
LIBXSMM_VLA_DECL(2, element_filter_type, roT, scratch_roT, K);
LIBXSMM_VLA_DECL(2, element_output_type, hT, scratch_hT, N);
/* define gemm kernels */
libxsmm_smmfunction gemmkernela = libxsmm_smmdispatch( bc, bn, bk, &C, &K, &C, NULL, NULL, NULL, NULL );
#if 0
libxsmm_smmfunction gemmkernelb = libxsmm_smmdispatch( bk, bk, bn, &K, &N, &K, NULL, NULL, NULL, NULL );
libxsmm_smmfunction gemmkernelc = libxsmm_smmdispatch( bk, bc, bn, &K, &N, &K, NULL, NULL, NULL, NULL );
#endif
libxsmm_smmfunction gemmkernelb = libxsmm_smmdispatch( bk, bk, bn, &K, &N, &K4, NULL, NULL, NULL, NULL );
libxsmm_smmfunction gemmkernelc = libxsmm_smmdispatch( bk, bc, bn, &K, &N, &K4, NULL, NULL, NULL, NULL );
libxsmm_smmfunction gemmkerneld = libxsmm_smmdispatch( bk, bn, bk, &K, &K, &K, NULL, NULL, NULL, NULL );

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
  ik = (ikic / (C/bc))*bk;
  ic = (ikic % (C/bc))*bc;

  for (jk = 0; jk < bk; ++jk) {
    for (jc = 0; jc < bc; ++jc) {
      ek = ik + jk;
      ec = ic + jc;
#if 0
      LIBXSMM_VLA_ACCESS(2, wiT, ek, ec, C) =  LIBXSMM_VLA_ACCESS(2, wi, ec, ek, K);
      LIBXSMM_VLA_ACCESS(2, wcT, ek, ec, C) =  LIBXSMM_VLA_ACCESS(2, wc, ec, ek, K);
      LIBXSMM_VLA_ACCESS(2, wfT, ek, ec, C) =  LIBXSMM_VLA_ACCESS(2, wf, ec, ek, K);
      LIBXSMM_VLA_ACCESS(2, woT, ek, ec, C) =  LIBXSMM_VLA_ACCESS(2, wo, ec, ek, K);
#endif
      LIBXSMM_VLA_ACCESS(2, wiT, ek, ec, C) =  LIBXSMM_VLA_ACCESS(2, wi, ec, ek, 4*K);
      LIBXSMM_VLA_ACCESS(2, wcT, ek, ec, C) =  LIBXSMM_VLA_ACCESS(2, wc, ec, ek, 4*K);
      LIBXSMM_VLA_ACCESS(2, wfT, ek, ec, C) =  LIBXSMM_VLA_ACCESS(2, wf, ec, ek, 4*K);
      LIBXSMM_VLA_ACCESS(2, woT, ek, ec, C) =  LIBXSMM_VLA_ACCESS(2, wo, ec, ek, 4*K);
    }
  }
}

/* transpose R */
for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
  ik = (ikic / (K/bk))*bk;
  ic = (ikic % (K/bk))*bk;

  for (jk = 0; jk < bk; ++jk) {
    for (jc = 0; jc < bk; ++jc) {
      ek = ik + jk;
      ec = ic + jc;
#if 0
      LIBXSMM_VLA_ACCESS(2, riT, ek, ec, K) =  LIBXSMM_VLA_ACCESS(2, ri, ec, ek, K);
      LIBXSMM_VLA_ACCESS(2, rcT, ek, ec, K) =  LIBXSMM_VLA_ACCESS(2, rc, ec, ek, K);
      LIBXSMM_VLA_ACCESS(2, rfT, ek, ec, K) =  LIBXSMM_VLA_ACCESS(2, rf, ec, ek, K);
      LIBXSMM_VLA_ACCESS(2, roT, ek, ec, K) =  LIBXSMM_VLA_ACCESS(2, ro, ec, ek, K);
#endif
      LIBXSMM_VLA_ACCESS(2, riT, ek, ec, K) =  LIBXSMM_VLA_ACCESS(2, ri, ec, ek, 4*K);
      LIBXSMM_VLA_ACCESS(2, rcT, ek, ec, K) =  LIBXSMM_VLA_ACCESS(2, rc, ec, ek, 4*K);
      LIBXSMM_VLA_ACCESS(2, rfT, ek, ec, K) =  LIBXSMM_VLA_ACCESS(2, rf, ec, ek, 4*K);
      LIBXSMM_VLA_ACCESS(2, roT, ek, ec, K) =  LIBXSMM_VLA_ACCESS(2, ro, ec, ek, 4*K);
    }
  }
}

libxsmm_barrier_wait(handle->barrier, (int)ltid);

for (j = t-1; j >= 0; --j) {
  /* transpose xt for current timestep */
  for (icin = thr_begin_nc; icin < thr_end_nc; ++icin ) {
    ic = (icin / (N/bn))*bc;
    in = (icin % (N/bn))*bn;

    for (jc = 0; jc < bc; ++jc) {
      for (jb = 0; jb < bn; ++jb) {
        en = in + jb;
        ec = ic + jc;
        LIBXSMM_VLA_ACCESS(2, xT, ec, en, N) =  LIBXSMM_VLA_ACCESS(3, x, j, en, ec, N, C);
      }
    }
  }

  /* transpose ht for current timestep */
  if (j == 1) {
    for (ikin = thr_begin_nk; ikin < thr_end_nk; ++ikin ) {
      ik = (ikin / (N/bn))*bk;
      in = (ikin % (N/bn))*bn;

      for (jk = 0; jk < bk; ++jk) {
        for (jb = 0; jb < bn; ++jb) {
          en = in + jb;
          ek = ik + jk;
          LIBXSMM_VLA_ACCESS(2, hT, ek, en, N) =  LIBXSMM_VLA_ACCESS(2, hp, en, ek, K);
        }
      }
    }
  } else if (j >= 2) {
    for (ikin = thr_begin_nk; ikin < thr_end_nk; ++ikin ) {
      ik = (ikin / (N/bn))*bk;
      in = (ikin % (N/bn))*bn;

      for (jk = 0; jk < bk; ++jk) {
        for (jb = 0; jb < bn; ++jb) {
          en = in + jb;
          ek = ik + jk;
          LIBXSMM_VLA_ACCESS(2, hT, ek, en, N) =  LIBXSMM_VLA_ACCESS(3, h, j-2, en, ek, N, K);
        }
      }
    }
  }

  libxsmm_barrier_wait(handle->barrier, (int)ltid);

  /* let's run the cell in blocks for good locality */
  for (inik = thr_begin_nk; inik < thr_end_nk; ++inik ) {
    in = (inik / (K/bk))*bn;
    ik = (inik % (K/bk))*bk;

    /* compute dhp */
    if (j == t-1) {
      libxsmm_internal_matrix_copy_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dh, t-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, dhp, t-1, in, ik, N, K) );
    } else {
      libxsmm_internal_matrix_add_ld(  bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dout, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, dh, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, dhp, j, in, ik, N, K) );
    }
    /* compute dcp */
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dhp, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    libxsmm_internal_matrix_complement_square_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    if (j == t-1) {
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(3, dcp, j, in, ik, N, K) );
      libxsmm_internal_matrix_add_ld(          bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dcs, in, ik, K), &LIBXSMM_VLA_ACCESS(3, dcp, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, dcp, j, in, ik, N, K) );
      if (dcsD) {
        /*libxsmm_internal_matrix_add_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dcs, in, ik, K), &LIBXSMM_VLA_ACCESS(3, dcp, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, dcp, j, in, ik, N, K) );*/
      }
    } else {
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(3, dcp, j, in, ik, N, K) );
      libxsmm_internal_matrix_eltwise_fma_ld(  bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dcp, j+1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, f, j+1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, dcp, j, in, ik, N, K) );
    }
    /* compute dci */
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dcp, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    libxsmm_internal_matrix_complement_square_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(3, dci, j, in, ik, N, K) );
    /* compute di */
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dcp, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    libxsmm_internal_matrix_complement_ld(        bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(3, di, j, in, ik, N, K) );
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(3, di, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, di, j, in, ik, N, K) );
    /* compute df */
    if (j == 0) {
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dcp, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, cp, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
      libxsmm_internal_matrix_complement_ld(   bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(3, df, j, in, ik, N, K) );
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(3, df, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, df, j, in, ik, N, K) );
    } else {
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dcp, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, cs, j-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
      libxsmm_internal_matrix_complement_ld(   bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(3, df, j, in, ik, N, K) );
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(3, df, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, df, j, in, ik, N, K) );
    }
    /* compute dp */
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dhp, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    libxsmm_internal_matrix_complement_ld(   bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(3, dp, j, in, ik, N, K) );

    if (j > 0) {
      /* dout-1 += R^T * difoc */
      for (ic = 0; ic < K; ic += bk) {
        gemmkerneld( &LIBXSMM_VLA_ACCESS(2, riT, ic, ik, K), &LIBXSMM_VLA_ACCESS(3, di,  j, in, ic, N, K), &LIBXSMM_VLA_ACCESS(3, dout, j-1, in, ik, N, K) );
        gemmkerneld( &LIBXSMM_VLA_ACCESS(2, rfT, ic, ik, K), &LIBXSMM_VLA_ACCESS(3, df,  j, in, ic, N, K), &LIBXSMM_VLA_ACCESS(3, dout, j-1, in, ik, N, K) );
        gemmkerneld( &LIBXSMM_VLA_ACCESS(2, roT, ic, ik, K), &LIBXSMM_VLA_ACCESS(3, dp,  j, in, ic, N, K), &LIBXSMM_VLA_ACCESS(3, dout, j-1, in, ik, N, K) );
        gemmkerneld( &LIBXSMM_VLA_ACCESS(2, rcT, ic, ik, K), &LIBXSMM_VLA_ACCESS(3, dci, j, in, ic, N, K), &LIBXSMM_VLA_ACCESS(3, dout, j-1, in, ik, N, K) );
      }
      /* libxsmm_internal_matrix_copy_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dout, j-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, dhp, j, in, ik, N, K) ); */
    }
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, dcp, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, dcp, j, in, ik, N, K) );
  }

  libxsmm_barrier_wait(handle->barrier, (int)ltid);

  if ( (LIBXSMM_DNN_COMPUTE_KIND_BWD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
    /* dx = W^T * difoc */
    for (inic = thr_begin_nc; inic < thr_end_nc; ++inic ) {
      in = (inic / (C/bc))*bn;
      ic = (inic % (C/bc))*bc;

      for (ik = 0; ik < K; ik += bk) {
        gemmkernela( &LIBXSMM_VLA_ACCESS(2, wiT, ik, ic, C), &LIBXSMM_VLA_ACCESS(3, di,  j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, dx, j, in, ic, N, C) );
        gemmkernela( &LIBXSMM_VLA_ACCESS(2, wfT, ik, ic, C), &LIBXSMM_VLA_ACCESS(3, df,  j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, dx, j, in, ic, N, C) );
        gemmkernela( &LIBXSMM_VLA_ACCESS(2, woT, ik, ic, C), &LIBXSMM_VLA_ACCESS(3, dp,  j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, dx, j, in, ic, N, C) );
        gemmkernela( &LIBXSMM_VLA_ACCESS(2, wcT, ik, ic, C), &LIBXSMM_VLA_ACCESS(3, dci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, dx, j, in, ic, N, C) );
      }
    }
  }

  if ( (LIBXSMM_DNN_COMPUTE_KIND_UPD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
      /* gradient bias */
      for (ik = thr_begin_k; ik < thr_end_k; ik++) {
        for (in = 0; in < N; in++) {
          dbi[ik] += LIBXSMM_VLA_ACCESS(3, di,  j, in, ik, N, K);
          dbf[ik] += LIBXSMM_VLA_ACCESS(3, df,  j, in, ik, N, K);
          dbo[ik] += LIBXSMM_VLA_ACCESS(3, dp,  j, in, ik, N, K);
          dbc[ik] += LIBXSMM_VLA_ACCESS(3, dci, j, in, ik, N, K);
        }
      }

    if (j > 0) {
      /* dr = difoc * h^T */
      for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
        ic = (ikic / (K/bk))*bk;
        ik = (ikic % (K/bk))*bk;

        for (in = 0; in < N; in += bn) {
          gemmkernelb( &LIBXSMM_VLA_ACCESS(3, di,  j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N), &LIBXSMM_VLA_ACCESS(2, dri, ic, ik, 4*K) );
          gemmkernelb( &LIBXSMM_VLA_ACCESS(3, df,  j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N), &LIBXSMM_VLA_ACCESS(2, drf, ic, ik, 4*K) );
          gemmkernelb( &LIBXSMM_VLA_ACCESS(3, dp,  j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N), &LIBXSMM_VLA_ACCESS(2, dro, ic, ik, 4*K) );
          gemmkernelb( &LIBXSMM_VLA_ACCESS(3, dci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N), &LIBXSMM_VLA_ACCESS(2, drc, ic, ik, 4*K) );
        }
      }
    }

    /* dw = difoc * x^T */
    for (ikic = thr_begin_ck; ikic < thr_end_ck; ++ikic ) {
      ic = (ikic / (K/bk))*bc;
      ik = (ikic % (K/bk))*bk;

      for (in = 0; in < N; in += bn ) {
        gemmkernelc( &LIBXSMM_VLA_ACCESS(3, di,  j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N), &LIBXSMM_VLA_ACCESS(2, dwi, ic, ik, 4*K) );
        gemmkernelc( &LIBXSMM_VLA_ACCESS(3, df,  j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N), &LIBXSMM_VLA_ACCESS(2, dwf, ic, ik, 4*K) );
        gemmkernelc( &LIBXSMM_VLA_ACCESS(3, dp,  j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N), &LIBXSMM_VLA_ACCESS(2, dwo, ic, ik, 4*K) );
        gemmkernelc( &LIBXSMM_VLA_ACCESS(3, dci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N), &LIBXSMM_VLA_ACCESS(2, dwc, ic, ik, 4*K) );
      }
    }
  }

  libxsmm_barrier_wait(handle->barrier, (int)ltid);
}
  for (inik = thr_begin_nk; inik < thr_end_nk; ++inik ) {
    in = (inik / (K/bk))*bn;
    ik = (inik % (K/bk))*bk;

    libxsmm_internal_matrix_zero_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dhp, 0, in, ik, N, K) );
    for (ic = 0; ic < K; ic += bk) {
      gemmkerneld( &LIBXSMM_VLA_ACCESS(2, riT, ic, ik, K), &LIBXSMM_VLA_ACCESS(3, di,  0, in, ic, N, K), &LIBXSMM_VLA_ACCESS(3, dhp, 0, in, ik, N, K) );
      gemmkerneld( &LIBXSMM_VLA_ACCESS(2, rfT, ic, ik, K), &LIBXSMM_VLA_ACCESS(3, df,  0, in, ic, N, K), &LIBXSMM_VLA_ACCESS(3, dhp, 0, in, ik, N, K) );
      gemmkerneld( &LIBXSMM_VLA_ACCESS(2, roT, ic, ik, K), &LIBXSMM_VLA_ACCESS(3, dp,  0, in, ic, N, K), &LIBXSMM_VLA_ACCESS(3, dhp, 0, in, ik, N, K) );
      gemmkerneld( &LIBXSMM_VLA_ACCESS(2, rcT, ic, ik, K), &LIBXSMM_VLA_ACCESS(3, dci, 0, in, ic, N, K), &LIBXSMM_VLA_ACCESS(3, dhp, 0, in, ik, N, K) );
    }
  }

  libxsmm_barrier_wait(handle->barrier, (int)ltid);

