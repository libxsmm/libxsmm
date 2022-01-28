/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Kunal Banerjee (Intel Corp.)
******************************************************************************/
#if 0
#define PROFILE
#endif

/* helper variables */
libxsmm_blasint j, ik, ikb, in, inb, ic, icb, jk, jb/*jn shadows global variable*/, jc, ek, en, ec, BF, KB_BLOCKS, KB;
/* tensor dimensions */
libxsmm_blasint K = handle->desc.K;
libxsmm_blasint N = handle->desc.N;
libxsmm_blasint C = handle->desc.C;
libxsmm_blasint t = handle->T;
libxsmm_blasint bk = handle->bk;
libxsmm_blasint bn = handle->bn;
libxsmm_blasint bc = handle->bc;
libxsmm_blasint K4 = K * 4;
const libxsmm_blasint cBlocks = C/bc;
const libxsmm_blasint kBlocks = K/bk;
const libxsmm_blasint nBlocks = N/bn;
const int lpb = handle->lpb;
/*const int bc_lp = bc/lpb;*/
const int bk_lp = bk/lpb;
const int bn_lp = bn/lpb;
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
element_output_type *db_bf16    = (element_output_type*)handle->db->data;
element_output_type *dcsD  = (element_output_type*)handle->dcs->data;
element_output_type *dht   = (element_output_type*)handle->dht->data;
element_output_type *diD   = (element_output_type*)handle->scratch_di;
element_output_type *dfD   = (element_output_type*)handle->scratch_df;
element_output_type *doD   = (element_output_type*)handle->scratch_do;
element_output_type *dciD  = (element_output_type*)handle->scratch_dci;
float *dxD   = (float*)handle->scratch_dx;
float *doutD = (float*)handle->scratch_deltat;
float *dhpD_f32 = (float*)handle->scratch_dhp;
float *db    = (float*)handle->scratch_db;
element_input_type  *scratch_xT  = (element_input_type* )handle->scratch_xT;
element_filter_type *scratch_wT  = (element_filter_type*)handle->scratch_wT;
element_filter_type *scratch_rT  = (element_filter_type*)handle->scratch_rT;
element_output_type *scratch_hT  = (element_output_type*)handle->scratch_hT;
float *w_scratch   = (float*)handle->scratch_w;
float *r_scratch   = (float*)handle->scratch_r;
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
float *dwiD_scratch  = &(w_scratch[0]);
float *dwcD_scratch  = &(w_scratch[C*K]);
float *dwfD_scratch  = &(w_scratch[2*C*K]);
float *dwoD_scratch  = &(w_scratch[3*C*K]);
float *driD_scratch  = &(r_scratch[0]);
float *drcD_scratch  = &(r_scratch[K*K]);
float *drfD_scratch  = &(r_scratch[2*K*K]);
float *droD_scratch  = &(r_scratch[3*K*K]);
float *dbi   = &(db[0]);
float *dbc   = &(db[K]);
float *dbf   = &(db[2*K]);
float *dbo   = &(db[3*K]);
element_output_type *dbi_bf16   = &(db_bf16[0]);
element_output_type *dbc_bf16   = &(db_bf16[K]);
element_output_type *dbf_bf16   = &(db_bf16[2*K]);
element_output_type *dbo_bf16   = &(db_bf16[3*K]);
element_filter_type *scratch_wiT = &(scratch_wT[0]);
element_filter_type *scratch_wcT = &(scratch_wT[C*K]);
element_filter_type *scratch_wfT = &(scratch_wT[2*C*K]);
element_filter_type *scratch_woT = &(scratch_wT[3*C*K]);
element_filter_type *scratch_riT = &(scratch_rT[0]);
element_filter_type *scratch_rcT = &(scratch_rT[K*K]);
element_filter_type *scratch_rfT = &(scratch_rT[2*K*K]);
element_filter_type *scratch_roT = &(scratch_rT[3*K*K]);
/*element_output_type *t1D   = (element_output_type*)handle->scratch_t1;*/
/*element_output_type *t2D   = (element_output_type*)handle->scratch_t2;*/
/* multidimensional arrays */
/*LIBXSMM_VLA_DECL(2, element_output_type, t1, t1D, K);*/
/*LIBXSMM_VLA_DECL(2, element_output_type, t2, t2D, K);*/
LIBXSMM_VLA_DECL(3, element_input_type,  x, xt, N, C);
LIBXSMM_VLA_DECL(2, element_input_type,  cp, csp, K);
LIBXSMM_VLA_DECL(2, element_input_type,  hp, hpD, K);
LIBXSMM_VLA_DECL(2, element_filter_type, wi, wiD, K4);
LIBXSMM_VLA_DECL(2, element_filter_type, wf, wfD, K4);
LIBXSMM_VLA_DECL(2, element_filter_type, wo, woD, K4);
LIBXSMM_VLA_DECL(2, element_filter_type, wc, wcD, K4);
LIBXSMM_VLA_DECL(2, element_filter_type, ri, riD, K4);
LIBXSMM_VLA_DECL(2, element_filter_type, rf, rfD, K4);
LIBXSMM_VLA_DECL(2, element_filter_type, ro, roD, K4);
LIBXSMM_VLA_DECL(2, element_filter_type, rc, rcD, K4);
LIBXSMM_VLA_DECL(3, element_output_type, cs, cst, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, h, ht, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, i, it, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, f, ft, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, o, ot, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, ci, cit, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, co, cot, N, K);
LIBXSMM_VLA_DECL(3, float,  dx, dxD, N, C);
LIBXSMM_VLA_DECL(3, element_input_type,  dx_bf16, dxt, N, C);
LIBXSMM_VLA_DECL(2, element_input_type,  dcp, dcsp, K);
LIBXSMM_VLA_DECL(2, element_input_type,  dhp, dhpD, K);
LIBXSMM_VLA_DECL(2, float,  dhp_f32, dhpD_f32, K);
LIBXSMM_VLA_DECL(4, float, dwi, dwiD_scratch, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, float, dwf, dwfD_scratch, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, float, dwo, dwoD_scratch, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, float, dwc, dwcD_scratch, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, float, dri, driD_scratch, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, float, drf, drfD_scratch, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, float, dro, droD_scratch, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, float, drc, drcD_scratch, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(2, element_filter_type, dwi_ck, dwiD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, dwf_ck, dwfD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, dwo_ck, dwoD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, dwc_ck, dwcD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, dri_ck, driD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, drf_ck, drfD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, dro_ck, droD, 4*K);
LIBXSMM_VLA_DECL(2, element_filter_type, drc_ck, drcD, 4*K);
LIBXSMM_VLA_DECL(2, element_output_type, dcs, dcsD, K);
LIBXSMM_VLA_DECL(3, element_output_type, dh, dht, N, K);
LIBXSMM_VLA_DECL(2, element_output_type, di, diD, K);
LIBXSMM_VLA_DECL(2, element_output_type, df, dfD, K);
LIBXSMM_VLA_DECL(2, element_output_type, dp, doD, K);
LIBXSMM_VLA_DECL(2, element_output_type, dci, dciD, K);
LIBXSMM_VLA_DECL(5, element_output_type, diB, (element_output_type*)handle->scratch_diB, nBlocks, bn_lp, bk, lpb);
LIBXSMM_VLA_DECL(5, element_output_type, dfB, (element_output_type*)handle->scratch_dfB, nBlocks, bn_lp, bk, lpb);
LIBXSMM_VLA_DECL(5, element_output_type, dpB, (element_output_type*)handle->scratch_dpB, nBlocks, bn_lp, bk, lpb);
LIBXSMM_VLA_DECL(5, element_output_type, dciB, (element_output_type*)handle->scratch_dciB, nBlocks, bn_lp, bk, lpb);
LIBXSMM_VLA_DECL(2, float, dout, doutD, K);
LIBXSMM_VLA_DECL(2, element_input_type,  xT, scratch_xT, N);
LIBXSMM_VLA_DECL(5, element_filter_type, wiT, scratch_wiT, kBlocks, bk_lp, bc, lpb);
LIBXSMM_VLA_DECL(5, element_filter_type, wcT, scratch_wcT, kBlocks, bk_lp, bc, lpb);
LIBXSMM_VLA_DECL(5, element_filter_type, wfT, scratch_wfT, kBlocks, bk_lp, bc, lpb);
LIBXSMM_VLA_DECL(5, element_filter_type, woT, scratch_woT, kBlocks, bk_lp, bc, lpb);
LIBXSMM_VLA_DECL(5, element_filter_type, riT, scratch_riT, kBlocks, bk_lp, bk, lpb);
LIBXSMM_VLA_DECL(5, element_filter_type, rcT, scratch_rcT, kBlocks, bk_lp, bk, lpb);
LIBXSMM_VLA_DECL(5, element_filter_type, rfT, scratch_rfT, kBlocks, bk_lp, bk, lpb);
LIBXSMM_VLA_DECL(5, element_filter_type, roT, scratch_roT, kBlocks, bk_lp, bk, lpb);
LIBXSMM_VLA_DECL(2, element_output_type, hT, scratch_hT, N);
float *dout_ptr = NULL;
/* define batch-reduce gemm kernels */
const libxsmm_bsmmfunction_reducebatch_strd batchreduce_kernela = handle->bwdupd_kernela;
const libxsmm_bsmmfunction_reducebatch_strd batchreduce_kernelb = handle->bwdupd_kernelb;
const libxsmm_bsmmfunction_reducebatch_strd batchreduce_kernelc = handle->bwdupd_kernelc;
const libxsmm_bsmmfunction_reducebatch_strd batchreduce_kerneld = handle->bwdupd_kerneld;
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

element_output_type *cps_ptr = NULL;
int k_tasks = K/16;
int k_chunksize = (k_tasks % (libxsmm_blasint)handle->desc.threads == 0) ? (k_tasks / (libxsmm_blasint)handle->desc.threads) : ((k_tasks / (libxsmm_blasint)handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const libxsmm_blasint k_thr_begin = (ltid * k_chunksize * 16 < K) ? (ltid * k_chunksize * 16) : K;
const libxsmm_blasint k_thr_end = ((ltid + 1) * k_chunksize * 16 < K) ? ((ltid + 1) * k_chunksize * 16) : K;
__m512 dbi_sum, dbf_sum, dbo_sum, dbc_sum;
#ifdef PROFILE
__int64_t _start, _end, eltwise_cycles = 0, dout_cycles = 0, weight_trans_cycles = 0, act_trans_cycles = 0, dx_cycles = 0, dwdr_cycles = 0, gradient_cycles = 0, reformat_cycles = 0;
float total_time = 0.0;
#endif
int bcbk_multiples_of_16 = ((bc % 16 == 0) && (bk % 16 == 0)) ? 1 : 0;

libxsmm_blasint ikic, inic, inik, icin, ikin;

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, (int)ltid);

/* Blocking reduction domain if it is too large */
BF = 1;
if (K > 1024 && K <= 2048) {
  BF = 8;
  while (kBlocks % BF != 0) {
    BF--;
  }
}

if (K > 2048) {
  BF = 16;
  while (kBlocks % BF != 0) {
    BF--;
  }
}
KB_BLOCKS = kBlocks/BF;

/* initialization is done at the beginning */
if ( (LIBXSMM_DNN_COMPUTE_KIND_BWD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
  libxsmm_internal_matrix_zero(N*C*t, dxD, start_thread, tid, handle->desc.threads);
}

/* initialization is done at the beginning */
if ( (LIBXSMM_DNN_COMPUTE_KIND_UPD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
  libxsmm_internal_matrix_zero(C*K*4, w_scratch,  start_thread, tid, handle->desc.threads);
  libxsmm_internal_matrix_zero(K*K*4, r_scratch,  start_thread, tid, handle->desc.threads);
  libxsmm_internal_matrix_zero(K*4,   db,  start_thread, tid, handle->desc.threads);
}

#ifdef PROFILE
if (ltid == 0) _start = _rdtsc();
#endif
/* transpose W */
for (ikic = thr_begin_ck; ikic < thr_end_ck; ++ikic ) {
  ic = (ikic / (K/bk));
  ik = (ikic % (K/bk));
  for (jk = 0; jk < bk; ++jk) {
    for (jc = 0; jc < bc; ++jc) {
      LIBXSMM_VLA_ACCESS(5, wiT, ic, ik, jk/lpb, jc, jk%lpb, kBlocks, bk_lp, bc, lpb) =  LIBXSMM_VLA_ACCESS(2, wi, ic*bc+jc, ik*bk+jk, 4*K);
      LIBXSMM_VLA_ACCESS(5, wcT, ic, ik, jk/lpb, jc, jk%lpb, kBlocks, bk_lp, bc, lpb) =  LIBXSMM_VLA_ACCESS(2, wc, ic*bc+jc, ik*bk+jk, 4*K);
      LIBXSMM_VLA_ACCESS(5, wfT, ic, ik, jk/lpb, jc, jk%lpb, kBlocks, bk_lp, bc, lpb) =  LIBXSMM_VLA_ACCESS(2, wf, ic*bc+jc, ik*bk+jk, 4*K);
      LIBXSMM_VLA_ACCESS(5, woT, ic, ik, jk/lpb, jc, jk%lpb, kBlocks, bk_lp, bc, lpb) =  LIBXSMM_VLA_ACCESS(2, wo, ic*bc+jc, ik*bk+jk, 4*K);
    }
  }
}

/* transpose R */
for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
  ik = (ikic / (K/bk));
  ic = (ikic % (K/bk));
  for (jk = 0; jk < bk; ++jk) {
    for (jc = 0; jc < bk; ++jc) {
      LIBXSMM_VLA_ACCESS(5, riT, ic, ik, jk/lpb, jc, jk%lpb, kBlocks, bk_lp, bk, lpb) =  LIBXSMM_VLA_ACCESS(2, ri, ic*bk+jc, ik*bk+jk, 4*K);
      LIBXSMM_VLA_ACCESS(5, rcT, ic, ik, jk/lpb, jc, jk%lpb, kBlocks, bk_lp, bk, lpb) =  LIBXSMM_VLA_ACCESS(2, rc, ic*bk+jc, ik*bk+jk, 4*K);
      LIBXSMM_VLA_ACCESS(5, rfT, ic, ik, jk/lpb, jc, jk%lpb, kBlocks, bk_lp, bk, lpb) =  LIBXSMM_VLA_ACCESS(2, rf, ic*bk+jc, ik*bk+jk, 4*K);
      LIBXSMM_VLA_ACCESS(5, roT, ic, ik, jk/lpb, jc, jk%lpb, kBlocks, bk_lp, bk, lpb) =  LIBXSMM_VLA_ACCESS(2, ro, ic*bk+jc, ik*bk+jk, 4*K);
    }
  }
}
#ifdef PROFILE
if (ltid == 0) {
  _end = _rdtsc();
  weight_trans_cycles += _end - _start;
}
#endif

#include "libxsmm_dnn_rnncell_st_lstm_bwdupd_nc_kcck_core_bf16.tpl.c"

if ( (LIBXSMM_DNN_COMPUTE_KIND_UPD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
#ifdef PROFILE
  if (ltid == 0) _start = _rdtsc();
#endif
  /* Store result weight matrices in CK format and downcovert to bf16 */
#if defined(LIBXSMM_RNN_CELL_AVX512)
  for (ikic = thr_begin_ck; ikic < thr_end_ck; ++ikic ) {
    icb = ikic / (K/bk);
    ic = icb*bc;
    ikb = ikic % (K/bk);
    ik = ikb*bk;
    for (jc = 0; jc < bc; ++jc) {
      for (jk = 0; jk < bk; jk += 16) {
        _mm256_storeu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(2, dwi_ck, ic+jc, ik+jk , K4), LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwi, ikb, icb, jc, jk, cBlocks, bc, bk))));
        _mm256_storeu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(2, dwc_ck, ic+jc, ik+jk , K4), LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwc, ikb, icb, jc, jk, cBlocks, bc, bk))));
        _mm256_storeu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(2, dwf_ck, ic+jc, ik+jk , K4), LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwf, ikb, icb, jc, jk, cBlocks, bc, bk))));
        _mm256_storeu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(2, dwo_ck, ic+jc, ik+jk , K4), LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwo, ikb, icb, jc, jk, cBlocks, bc, bk))));
      }
    }
  }

  for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
    icb = ikic / (K/bk);
    ic = icb*bk;
    ikb = ikic % (K/bk);
    ik = ikb*bk;
    for (jc = 0; jc < bk; ++jc) {
      for (jk = 0; jk < bk; jk += 16) {
        _mm256_storeu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(2, dri_ck, ic+jc, ik+jk , K4), LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dri, ikb, icb, jc, jk, kBlocks, bk, bk))));
        _mm256_storeu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(2, drc_ck, ic+jc, ik+jk , K4), LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, drc, ikb, icb, jc, jk, kBlocks, bk, bk))));
        _mm256_storeu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(2, drf_ck, ic+jc, ik+jk , K4), LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, drf, ikb, icb, jc, jk, kBlocks, bk, bk))));
        _mm256_storeu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(2, dro_ck, ic+jc, ik+jk , K4), LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dro, ikb, icb, jc, jk, kBlocks, bk, bk))));
      }
    }
  }
#else
  /* TODO: Add here non AVX512 replacement code  */
#endif
  libxsmm_barrier_wait(handle->barrier, (int)ltid);
#ifdef PROFILE
  if (ltid == 0) {
    _end = _rdtsc();
    reformat_cycles += _end - _start;
  }
#endif
}

#ifdef PROFILE
if (ltid == 0) {
  printf("----- PROFILING LSTM BWD/UPD (N = %d, C = %d, K = %d, bn = %d. bc = %d, bk = %d)----\n", N, C, K, bn, bc, bk );
  total_time = (gradient_cycles+dwdr_cycles+dx_cycles+act_trans_cycles+weight_trans_cycles+dout_cycles+eltwise_cycles+reformat_cycles)/(2.5 * 1e9)*1000.0f;
  printf("Transpose weights time is %f ms (%.2f%%)\n", weight_trans_cycles/(2.5 * 1e9)*1000.0f, weight_trans_cycles/(2.5 * 1e9)*1000.0f*100.0/total_time );
  printf("Elementwise time is %f ms (%.2f%%)\n", eltwise_cycles/(2.5 * 1e9)*1000.0f, eltwise_cycles/(2.5 * 1e9)*1000.0f*100.0/total_time );
  printf("Dx GEMM time is %f ms (%.2f%%) at %f GFLOPS\n", dx_cycles/(2.5 * 1e9)*1000.0f, dx_cycles/(2.5 * 1e9)*1000.0f*100.0/total_time, t*2.0*N*C*K*4/1e9/(dx_cycles/(2.5 * 1e9)));
  printf("Dh GEMM time is %f ms (%.2f%%) at %f GFLOPS\n", dout_cycles/(2.5 * 1e9)*1000.0f, dout_cycles/(2.5 * 1e9)*1000.0f*100.0/total_time, t*2.0*N*K*K*4/1e9/(dout_cycles/(2.5 * 1e9)));
  printf("Transpose input activations time is %f ms (%.2f%%)\n", act_trans_cycles/(2.5 * 1e9)*1000.0f, act_trans_cycles/(2.5 * 1e9)*1000.0f*100.0/total_time );
  printf("Dwdr GEMM time is %f ms (%.2f%%) at %f GFLOPS\n", dwdr_cycles/(2.5 * 1e9)*1000.0f, dwdr_cycles/(2.5 * 1e9)*1000.0f*100.0/total_time, t*2.0*(N*K*K*2.0+N*C*K*2.0)*2.0/1e9/(dwdr_cycles/(2.5 * 1e9)));
  printf("Gradient bias calculation time is %f ms (%.2f%%)\n", gradient_cycles/(2.5 * 1e9)*1000.0f, gradient_cycles/(2.5 * 1e9)*1000.0f*100.0/total_time );
  printf("Reformat dwdr time is %f ms (%.2f%%)\n\n", reformat_cycles/(2.5 * 1e9)*1000.0f, reformat_cycles/(2.5 * 1e9)*1000.0f*100.0/total_time );
}
#undef PROFILE
#endif

