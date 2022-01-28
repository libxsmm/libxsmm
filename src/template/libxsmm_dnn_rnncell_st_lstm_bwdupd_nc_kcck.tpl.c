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
const libxsmm_blasint cBlocks = C/bc;
const libxsmm_blasint kBlocks = K/bk;
const libxsmm_blasint nBlocks = N/bn;
unsigned long long blocks;
/* tensor raw pointers */
element_input_type  *xt    = (element_input_type* )handle->xt->data;
element_input_type *csp    = (element_input_type* )handle->csp->data;
element_input_type *hpD    = (element_input_type* )handle->hp->data;
element_filter_type *wt    = (element_filter_type*)handle->wt->data;
element_filter_type *rt    = (element_filter_type*)handle->rt->data;
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
element_input_type  *scratch_xT  = (element_input_type* )handle->scratch_xT;
#if 0
element_filter_type *scratch_wT  = (element_filter_type*)handle->scratch_wT;
element_filter_type *scratch_rT  = (element_filter_type*)handle->scratch_rT;
#endif
element_output_type *scratch_hT  = (element_output_type*)handle->scratch_hT;
element_filter_type *witD  = &(wt[0]);
element_filter_type *wctD  = &(wt[C*K]);
element_filter_type *wftD  = &(wt[2*C*K]);
element_filter_type *wotD  = &(wt[3*C*K]);
element_filter_type *ritD  = &(rt[0]);
element_filter_type *rctD  = &(rt[K*K]);
element_filter_type *rftD  = &(rt[2*K*K]);
element_filter_type *rotD  = &(rt[3*K*K]);
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
#if 0
element_filter_type *scratch_wiT = &(scratch_wT[0]);
element_filter_type *scratch_wcT = &(scratch_wT[C*K]);
element_filter_type *scratch_wfT = &(scratch_wT[2*C*K]);
element_filter_type *scratch_woT = &(scratch_wT[3*C*K]);
element_filter_type *scratch_riT = &(scratch_rT[0]);
element_filter_type *scratch_rcT = &(scratch_rT[K*K]);
element_filter_type *scratch_rfT = &(scratch_rT[2*K*K]);
element_filter_type *scratch_roT = &(scratch_rT[3*K*K]);
#endif
element_output_type *t1D   = (element_output_type*)handle->scratch_t1;
element_output_type *t2D   = (element_output_type*)handle->scratch_t2;
/* multidimensional arrays */
LIBXSMM_VLA_DECL(2, element_output_type, t1, t1D, K);
LIBXSMM_VLA_DECL(2, element_output_type, t2, t2D, K);
LIBXSMM_VLA_DECL(3, element_input_type,  x, xt, N, C);
LIBXSMM_VLA_DECL(2, element_input_type,  cp, csp, K);
LIBXSMM_VLA_DECL(2, element_input_type,  hp, hpD, K);
#if 0
LIBXSMM_VLA_DECL(4, element_filter_type, wi, wiD, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, wf, wfD, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, wo, woD, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, wc, wcD, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, ri, riD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, rf, rfD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, ro, roD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, rc, rcD, kBlocks, bk, bk);
#endif
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
LIBXSMM_VLA_DECL(2, element_input_type,  xT, scratch_xT, N);
LIBXSMM_VLA_DECL(4, element_filter_type, wiT, witD, kBlocks, bk, bc);
LIBXSMM_VLA_DECL(4, element_filter_type, wcT, wctD, kBlocks, bk, bc);
LIBXSMM_VLA_DECL(4, element_filter_type, wfT, wftD, kBlocks, bk, bc);
LIBXSMM_VLA_DECL(4, element_filter_type, woT, wotD, kBlocks, bk, bc);
LIBXSMM_VLA_DECL(4, element_filter_type, riT, ritD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, rcT, rctD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, rfT, rftD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, roT, rotD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(2, element_output_type, hT, scratch_hT, N);
element_output_type *dout_ptr = NULL;
/* define batch-reduce gemm kernels */
const libxsmm_smmfunction_reducebatch_addr batchreduce_kernela = libxsmm_smmdispatch_reducebatch_addr( bc, bn, bk, &bc, &K, &C, NULL, NULL, NULL, NULL);
const libxsmm_smmfunction_reducebatch_addr batchreduce_kernelb = libxsmm_smmdispatch_reducebatch_addr( bk, bk, bn, &bk, &N, &bk, NULL, NULL, NULL, NULL);
const libxsmm_smmfunction_reducebatch_addr batchreduce_kernelc = libxsmm_smmdispatch_reducebatch_addr( bk, bc, bn, &bk, &N, &bk, NULL, NULL, NULL, NULL);
const libxsmm_smmfunction_reducebatch_addr batchreduce_kernelb1 = libxsmm_smmdispatch_reducebatch_addr( bk, bk, bn, &K, &N, &bk, NULL, NULL, NULL, NULL);
const libxsmm_smmfunction_reducebatch_addr batchreduce_kernelc1 = libxsmm_smmdispatch_reducebatch_addr( bk, bc, bn, &K, &N, &bk, NULL, NULL, NULL, NULL);
const libxsmm_smmfunction_reducebatch_addr batchreduce_kerneld = libxsmm_smmdispatch_reducebatch_addr( bk, bn, bk, &bk, &K, &K, NULL, NULL, NULL, NULL);

/* Auxiliary arrays for batch-reduce gemm calls */
const element_filter_type *A_array[1024];
const element_output_type *B_array[1024];

LIBXSMM_VLA_DECL(4, element_output_type, diB, (element_output_type*)handle->scratch_diB, kBlocks, bn, bk);
LIBXSMM_VLA_DECL(4, element_output_type, dfB, (element_output_type*)handle->scratch_dfB, kBlocks, bn, bk);
LIBXSMM_VLA_DECL(4, element_output_type, dpB, (element_output_type*)handle->scratch_dpB, kBlocks, bn, bk);
LIBXSMM_VLA_DECL(4, element_output_type, dciB, (element_output_type*)handle->scratch_dciB, kBlocks, bn, bk);

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

#if defined(LIBXSMM_RNN_CELL_AVX512)
element_output_type *cps_ptr = NULL;
int k_tasks = K/16;
int k_chunksize = (k_tasks % (libxsmm_blasint)handle->desc.threads == 0) ? (k_tasks / (libxsmm_blasint)handle->desc.threads) : ((k_tasks / (libxsmm_blasint)handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const libxsmm_blasint k_thr_begin = (ltid * k_chunksize * 16 < K) ? (ltid * k_chunksize * 16) : K;
const libxsmm_blasint k_thr_end = ((ltid + 1) * k_chunksize * 16 < K) ? ((ltid + 1) * k_chunksize * 16) : K;__m512 dbi_sum, dbf_sum, dbo_sum, dbc_sum;
#endif
/* number of tasks that could be run in parallel for K blocks*/
/* compute chunk size */
const libxsmm_blasint chunksize_k = (K % (libxsmm_blasint)handle->desc.threads == 0) ? (K / (libxsmm_blasint)handle->desc.threads) : ((K / (libxsmm_blasint)handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const libxsmm_blasint thr_begin_k = (ltid * chunksize_k < K) ? (ltid * chunksize_k) : K;
const libxsmm_blasint thr_end_k = ((ltid + 1) * chunksize_k < K) ? ((ltid + 1) * chunksize_k) : K;
#ifdef PROFILE
__int64_t _start, _end, eltwise_cycles = 0, dout_cycles = 0, weight_trans_cycles = 0, act_trans_cycles = 0, dx_cycles = 0, dwdr_cycles = 0, gradient_cycles = 0;
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
  libxsmm_internal_matrix_zero(N*C*t, dxt, start_thread, tid, handle->desc.threads);
}

/* initialization is done at the beginning */
if ( (LIBXSMM_DNN_COMPUTE_KIND_UPD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
  libxsmm_internal_matrix_zero(C*K*4, dw,  start_thread, tid, handle->desc.threads);
  libxsmm_internal_matrix_zero(K*K*4, dr,  start_thread, tid, handle->desc.threads);
  libxsmm_internal_matrix_zero(K*4,   db,  start_thread, tid, handle->desc.threads);
}

/* Here we assume that the weight tensors come in transposed from framework */
#if 0
#ifdef PROFILE
if (ltid == 0) _start = _rdtsc();
#endif
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
#ifdef PROFILE
if (ltid == 0) {
  _end = _rdtsc();
  weight_trans_cycles += _end - _start;
}
#endif
#endif

#include "libxsmm_dnn_rnncell_st_lstm_bwdupd_nc_kcck_core.tpl.c"

#ifdef PROFILE
if (ltid == 0) {
  printf("----- PROFILING LSTM BWD/UPD (N = %d, C = %d, K = %d, bn = %d. bc = %d, bk = %d)----\n", N, C, K, bn, bc, bk );
  total_time = (gradient_cycles+dwdr_cycles+dx_cycles+act_trans_cycles+weight_trans_cycles+dout_cycles+eltwise_cycles)/(2.5 * 1e9)*1000.0f;
  printf("Transpose weights time is %f ms (%.2f%%)\n", weight_trans_cycles/(2.5 * 1e9)*1000.0f, weight_trans_cycles/(2.5 * 1e9)*1000.0f*100.0/total_time );
  printf("Elementwise time is %f ms (%.2f%%)\n", eltwise_cycles/(2.5 * 1e9)*1000.0f, eltwise_cycles/(2.5 * 1e9)*1000.0f*100.0/total_time );
  printf("Dx GEMM time is %f ms (%.2f%%) at %f GFLOPS\n", dx_cycles/(2.5 * 1e9)*1000.0f, dx_cycles/(2.5 * 1e9)*1000.0f*100.0/total_time, t*2.0*N*C*K*4/1e9/(dx_cycles/(2.5 * 1e9)));
  printf("Dh GEMM time is %f ms (%.2f%%) at %f GFLOPS\n", dout_cycles/(2.5 * 1e9)*1000.0f, dout_cycles/(2.5 * 1e9)*1000.0f*100.0/total_time, t*2.0*N*K*K*4/1e9/(dout_cycles/(2.5 * 1e9)));
  printf("Transpose input activations time is %f ms (%.2f%%)\n", act_trans_cycles/(2.5 * 1e9)*1000.0f, act_trans_cycles/(2.5 * 1e9)*1000.0f*100.0/total_time );
  printf("Dwdr GEMM time is %f ms (%.2f%%) at %f GFLOPS\n", dwdr_cycles/(2.5 * 1e9)*1000.0f, dwdr_cycles/(2.5 * 1e9)*1000.0f*100.0/total_time, t*2.0*(N*K*K*2.0+N*C*K*2.0)*2.0/1e9/(dwdr_cycles/(2.5 * 1e9)));
  printf("Gradient bias calculation time is %f ms (%.2f%%)\n", gradient_cycles/(2.5 * 1e9)*1000.0f, gradient_cycles/(2.5 * 1e9)*1000.0f*100.0/total_time );
}
#undef PROFILE
#endif
