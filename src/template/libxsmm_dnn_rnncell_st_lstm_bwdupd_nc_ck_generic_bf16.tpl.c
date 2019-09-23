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
#if 0
#define PROFILE
#endif

#define _mm512_roundbf16rne(A) LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(A)
#define _mm512_loadcvt_bf16_fp32(A)   _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(A))),16))
#define _mm512_storecvt_fp32_bf16(A,B)  _mm256_storeu_si256((__m256i*)(A),_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_roundbf16rne((B)),16)))

/* helper variables */
libxsmm_blasint ik, ikb, ic, icb, jk, jc, BF;
/* tensor dimensions */
libxsmm_blasint K = handle->desc.K;
libxsmm_blasint N = handle->desc.N;
libxsmm_blasint C = handle->desc.C;
libxsmm_blasint t = handle->T;
libxsmm_blasint bk = handle->bk;
libxsmm_blasint bc = handle->bc;
libxsmm_blasint K4 = K * 4;
const libxsmm_blasint cBlocks = C/bc;
const libxsmm_blasint kBlocks = K/bk;
const int lpb = handle->lpb;
const int bk_lp = bk/lpb;
/* tensor raw pointers */
element_filter_type *w     = (element_filter_type*)handle->w->data;
element_filter_type *r     = (element_filter_type*)handle->r->data;
element_filter_type *dw    = (element_filter_type*)handle->dw->data;
element_filter_type *dr    = (element_filter_type*)handle->dr->data;
float *dxD   = (float*)handle->scratch_dx;
float *db    = (float*)handle->scratch_db;
element_filter_type *scratch_wT  = (element_filter_type*)handle->scratch_wT;
element_filter_type *scratch_rT  = (element_filter_type*)handle->scratch_rT;
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
element_filter_type *scratch_wiT = &(scratch_wT[0]);
element_filter_type *scratch_wcT = &(scratch_wT[C*K]);
element_filter_type *scratch_wfT = &(scratch_wT[2*C*K]);
element_filter_type *scratch_woT = &(scratch_wT[3*C*K]);
element_filter_type *scratch_riT = &(scratch_rT[0]);
element_filter_type *scratch_rcT = &(scratch_rT[K*K]);
element_filter_type *scratch_rfT = &(scratch_rT[2*K*K]);
element_filter_type *scratch_roT = &(scratch_rT[3*K*K]);
/* multidimensional arrays */
LIBXSMM_VLA_DECL(2, element_filter_type, wi, wiD, K4);
LIBXSMM_VLA_DECL(2, element_filter_type, wf, wfD, K4);
LIBXSMM_VLA_DECL(2, element_filter_type, wo, woD, K4);
LIBXSMM_VLA_DECL(2, element_filter_type, wc, wcD, K4);
LIBXSMM_VLA_DECL(2, element_filter_type, ri, riD, K4);
LIBXSMM_VLA_DECL(2, element_filter_type, rf, rfD, K4);
LIBXSMM_VLA_DECL(2, element_filter_type, ro, roD, K4);
LIBXSMM_VLA_DECL(2, element_filter_type, rc, rcD, K4);
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
LIBXSMM_VLA_DECL(5, element_filter_type, wiT, scratch_wiT, kBlocks, bk_lp, bc, lpb);
LIBXSMM_VLA_DECL(5, element_filter_type, wcT, scratch_wcT, kBlocks, bk_lp, bc, lpb);
LIBXSMM_VLA_DECL(5, element_filter_type, wfT, scratch_wfT, kBlocks, bk_lp, bc, lpb);
LIBXSMM_VLA_DECL(5, element_filter_type, woT, scratch_woT, kBlocks, bk_lp, bc, lpb);
LIBXSMM_VLA_DECL(5, element_filter_type, riT, scratch_riT, kBlocks, bk_lp, bk, lpb);
LIBXSMM_VLA_DECL(5, element_filter_type, rcT, scratch_rcT, kBlocks, bk_lp, bk, lpb);
LIBXSMM_VLA_DECL(5, element_filter_type, rfT, scratch_rfT, kBlocks, bk_lp, bk, lpb);
LIBXSMM_VLA_DECL(5, element_filter_type, roT, scratch_roT, kBlocks, bk_lp, bk, lpb);

/* computing first logical thread */
const libxsmm_blasint ltid = (libxsmm_blasint)tid - (libxsmm_blasint)start_thread;

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
#if 0
const libxsmm_blasint chunksize_k = (K % (libxsmm_blasint)handle->desc.threads == 0) ? (K / (libxsmm_blasint)handle->desc.threads) : ((K / (libxsmm_blasint)handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const libxsmm_blasint thr_begin_k = (ltid * chunksize_k < K) ? (ltid * chunksize_k) : K;
const libxsmm_blasint thr_end_k = ((ltid + 1) * chunksize_k < K) ? ((ltid + 1) * chunksize_k) : K;
#endif
#ifdef PROFILE
__int64_t _start, _end, eltwise_cycles = 0, dout_cycles = 0, weight_trans_cycles = 0, act_trans_cycles = 0, dx_cycles = 0, dwdr_cycles = 0, gradient_cycles = 0, reformat_cycles = 0;
float total_time = 0.0;
#endif

libxsmm_blasint ikic;

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

/*#include "libxsmm_dnn_rnncell_st_lstm_bwdupd_nc_kcck_core_bf16.tpl.c"*/

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
        _mm512_storecvt_fp32_bf16(&LIBXSMM_VLA_ACCESS(2, dwi_ck, ic+jc, ik+jk , K4), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwi, ikb, icb, jc, jk, cBlocks, bc, bk)));
        _mm512_storecvt_fp32_bf16(&LIBXSMM_VLA_ACCESS(2, dwc_ck, ic+jc, ik+jk , K4), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwc, ikb, icb, jc, jk, cBlocks, bc, bk)));
        _mm512_storecvt_fp32_bf16(&LIBXSMM_VLA_ACCESS(2, dwf_ck, ic+jc, ik+jk , K4), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwf, ikb, icb, jc, jk, cBlocks, bc, bk)));
        _mm512_storecvt_fp32_bf16(&LIBXSMM_VLA_ACCESS(2, dwo_ck, ic+jc, ik+jk , K4), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwo, ikb, icb, jc, jk, cBlocks, bc, bk)));
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
        _mm512_storecvt_fp32_bf16(&LIBXSMM_VLA_ACCESS(2, dri_ck, ic+jc, ik+jk , K4), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dri, ikb, icb, jc, jk, kBlocks, bk, bk)));
        _mm512_storecvt_fp32_bf16(&LIBXSMM_VLA_ACCESS(2, drc_ck, ic+jc, ik+jk , K4), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, drc, ikb, icb, jc, jk, kBlocks, bk, bk)));
        _mm512_storecvt_fp32_bf16(&LIBXSMM_VLA_ACCESS(2, drf_ck, ic+jc, ik+jk , K4), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, drf, ikb, icb, jc, jk, kBlocks, bk, bk)));
        _mm512_storecvt_fp32_bf16(&LIBXSMM_VLA_ACCESS(2, dro_ck, ic+jc, ik+jk , K4), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dro, ikb, icb, jc, jk, kBlocks, bk, bk)));
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

#undef _mm512_roundbf16rne
#undef _mm512_loadcvt_bf16_fp32
#undef _mm512_storecvt_fp32_bf16

