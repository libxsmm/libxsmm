/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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

int ltid;
int work;
int chunksize;
int thr_begin;
int thr_end;
int job;
unsigned int img;
int ifm1;
int ofm1;
int oj;
int oi;
int k, l;
int a1, a2, t;
unsigned int i;
LIBXSMM_VLA_DECL(5, float, input,  (float*)handle->reg_input->data, handle->ifhp, handle->ifwp, handle->blocksifm, TDVLEN);
LIBXSMM_VLA_DECL(5, float, output, (float*)handle->reg_output->data, handle->ofhp, handle->ofwp, handle->blocksofm, TDVLEN);
LIBXSMM_VLA_DECL(6, float, weight, (float*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, TDVLEN, TDVLEN);

LIBXSMM_VLA_DECL(6, float, U,   (float*)handle->scratch1, ALPHA, handle->blocksofm, handle->blocksifm, TDVLEN, TDVLEN);
LIBXSMM_VLA_DECL(8, float, V,   (float*)handle->scratch3, ALPHA, ALPHA, handle->blocksifm, handle->cwino_upd.bimg, handle->cwino_upd.jtiles, handle->cwino_upd.itiles, TDVLEN);
LIBXSMM_VLA_DECL(8, float, M,   (float*)handle->scratch4, ALPHA, ALPHA, handle->blocksofm, handle->cwino_upd.bimg, handle->cwino_upd.jtiles, handle->cwino_upd.itiles, TDVLEN);
LIBXSMM_VLA_DECL(5, float, Iwp, (float*)handle->scratchIw, handle->cwino_upd.itiles*handle->cwino_upd.jtiles, ALPHA, ALPHA, TDVLEN);
LIBXSMM_VLA_DECL(5, float, Owp, (float*)handle->scratchOw, handle->cwino_upd.itiles*handle->cwino_upd.jtiles, ALPHA, ALPHA, TDVLEN);
LIBXSMM_VLA_DECL(7, float, Vk, (float*)handle->scratchVk, ALPHA, ALPHA, handle->blocksifm, handle->cwino_upd.bimg*handle->cwino_upd.jtiles*handle->cwino_upd.itiles/4, TDVLEN, 4);
#if 1
typedef libxsmm_sconvfunction libxsmm_convfunction;
libxsmm_convfunction jitted_conv_wu = (libxsmm_convfunction)handle->code_upd[1].xconv.sconv;
#endif
LIBXSMM_ASSUME_ALIGNED(handle->reg_input->data,  64);
LIBXSMM_ASSUME_ALIGNED(handle->reg_output->data, 64);
LIBXSMM_ASSUME_ALIGNED(handle->reg_filter->data, 64);
LIBXSMM_ASSUME_ALIGNED(handle->scratch1, 64);
LIBXSMM_ASSUME_ALIGNED(handle->scratch3, 64);
LIBXSMM_ASSUME_ALIGNED(handle->scratch4, 64);
LIBXSMM_ASSUME_ALIGNED(handle->scratchIw, 64);
LIBXSMM_ASSUME_ALIGNED(handle->scratchOw, 64);

/* computing first logical thread */
ltid = tid - start_thread;
libxsmm_barrier_init((libxsmm_barrier*)handle->barrier, ltid);

/* #define WTIME */
#ifdef WTIME
unsigned long long t_input  = 0;
unsigned long long t_wt     = 0;
unsigned long long t_output = 0;
unsigned long long t_gemm   = 0;
unsigned long long t_start  = 0;
#endif

/* number of tasks that could be run in parallel */
work = handle->desc.N*handle->blocksifm;
/* compute chunck size */
chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

#ifdef WTIME
t_start = __rdtsc();
#endif
for (job = thr_begin; job < thr_end; job++) {
  img  = job / handle->blocksifm;
  ifm1 = job % handle->blocksifm;
  if (handle->flag_reuseInput != 1 || handle->cwino_upd.alpha != 6 || handle->cwino_fwd.bimg != handle->cwino_upd.bimg) {
    internal_upd_input_transform_nhwc_custom(
      &LIBXSMM_VLA_ACCESS(5, input, img, 0, 0, ifm1, 0, handle->ifhp, handle->ifwp, handle->blocksifm, TDVLEN),
      &LIBXSMM_VLA_ACCESS(8, V, img/handle->cwino_upd.bimg, 0, 0, ifm1, img%handle->cwino_upd.bimg, 0, 0, 0, ALPHA, ALPHA, handle->blocksifm, handle->cwino_upd.bimg, handle->cwino_upd.jtiles, handle->cwino_upd.itiles, TDVLEN),
      &LIBXSMM_VLA_ACCESS(5, Iwp, tid, 0, 0, 0, 0, handle->cwino_upd.itiles*handle->cwino_upd.jtiles, ALPHA, ALPHA, TDVLEN), handle);
  } /*end flag_reuseInput*/

  for (a1 = 0; a1 < ALPHA; a1++) {
    for (a2 = 0; a2 < ALPHA; a2++) {
      for (t = 0; t < TDVLEN; t++) {
        for (i = 0; i < handle->cwino_upd.jtiles*handle->cwino_upd.itiles; i++) {
          LIBXSMM_VLA_ACCESS(7, Vk, img/handle->cwino_upd.bimg, a1, a2, ifm1, ((img%handle->cwino_upd.bimg)*handle->cwino_upd.jtiles*handle->cwino_upd.itiles + i)/4, t, ((img%handle->cwino_upd.bimg)*handle->cwino_upd.jtiles*handle->cwino_upd.itiles + i)%4, ALPHA, ALPHA, handle->blocksifm, handle->cwino_upd.bimg*handle->cwino_upd.jtiles*handle->cwino_upd.itiles/4, TDVLEN, 4) =
            LIBXSMM_VLA_ACCESS(8, V, img/handle->cwino_upd.bimg, a1, a2, ifm1, img%handle->cwino_upd.bimg, i/handle->cwino_upd.itiles, i%handle->cwino_upd.itiles, t, ALPHA, ALPHA, handle->blocksifm, handle->cwino_upd.bimg, handle->cwino_upd.jtiles, handle->cwino_upd.itiles, TDVLEN);
        }
      }
    }
  }
}
#ifdef WTIME
libxsmm_barrier_wait((libxsmm_barrier*)handle->barrier, ltid);
t_input = __rdtsc() - t_start;
#endif

/* number of tasks that could be run in parallel */
work = handle->desc.N*handle->blocksofm;
/* compute chunck size */
chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

#ifdef WTIME
t_start = __rdtsc();
#endif
for (job = thr_begin; job < thr_end; job++) {
  img  = job / handle->blocksofm;
  ofm1 = job % handle->blocksofm;
  internal_upd_deloutput_transform_nhwc_custom(
    &LIBXSMM_VLA_ACCESS(5, output, img, 0, 0, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, TDVLEN),
    &LIBXSMM_VLA_ACCESS(8, M, img/handle->cwino_upd.bimg, 0, 0, ofm1, img%handle->cwino_upd.bimg, 0, 0, 0, ALPHA, ALPHA, handle->blocksofm, handle->cwino_upd.bimg, handle->cwino_upd.jtiles, handle->cwino_upd.itiles, TDVLEN),
    &LIBXSMM_VLA_ACCESS(5, Owp, tid, 0, 0, 0, 0, handle->cwino_upd.itiles*handle->cwino_upd.jtiles, ALPHA, ALPHA, TDVLEN), handle);
}
libxsmm_barrier_wait((libxsmm_barrier*)handle->barrier, ltid);
#ifdef WTIME
t_output = __rdtsc() - t_start;
#endif

/* number of tasks that could be run in parallel */
work = ALPHA * ALPHA * handle->blocksofm;
/* compute chunck size */
chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

#ifdef WTIME
t_start = __rdtsc();
#endif
for (job = thr_begin; job < thr_end; job++) {
  oj = job / (ALPHA * handle->blocksofm);
  oi = (job % (ALPHA * handle->blocksofm)) / handle->blocksofm;
  ofm1 = (job % (ALPHA * handle->blocksofm)) % handle->blocksofm;
  for (img = 0; img < handle->desc.N/handle->cwino_upd.bimg; img++) {
    for (ifm1 = 0; ifm1 < handle->blocksifm; ifm1++) {
      if (img == 0) {
        for (k = 0; k < TDVLEN; k++) {
          LIBXSMM_PRAGMA_SIMD
          for (l = 0; l < TDVLEN; l++) {
            LIBXSMM_VLA_ACCESS(6, U, oj, oi, ofm1, ifm1, k, l, ALPHA, handle->blocksofm, handle->blocksifm, TDVLEN, TDVLEN) = 0.0f;
          }
        }
      }
#if 1
      jitted_conv_wu(
        &LIBXSMM_VLA_ACCESS(8, M, img, oj, oi, ofm1, 0, 0, 0, 0, ALPHA, ALPHA, handle->blocksofm, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles, TDVLEN),
        &LIBXSMM_VLA_ACCESS(7, Vk, img, oj, oi, ifm1, 0, 0, 0, ALPHA, ALPHA, handle->blocksifm, handle->cwino_fwd.bimg*handle->cwino_fwd.jtiles*handle->cwino_fwd.itiles/4, TDVLEN, 4),
        &LIBXSMM_VLA_ACCESS(6, U, oj, oi, ofm1, ifm1, 0, 0, ALPHA, handle->blocksofm, handle->blocksifm, TDVLEN, TDVLEN),
        0, 0, 0);
#else
      unsigned int ti, tj;
      unsigned int img1;
      int ifm2, ofm2;
      for (img1 = 0; img1 < handle->cwino_upd.bimg; img1++) {
        for (tj = 0; tj < handle->cwino_upd.jtiles; tj++) {
          for (ti = 0; ti < handle->cwino_upd.itiles; ti++) {
            for (ifm2 = 0; ifm2 < TDVLEN; ifm2++) {
              for (ofm2 = 0; ofm2 < TDVLEN; ofm2++) {
                LIBXSMM_VLA_ACCESS  (6, U, oj, oi, ofm1, ifm1, ifm2, ofm2, ALPHA, handle->blocksofm, handle->blocksifm, TDVLEN, TDVLEN) +=
                  LIBXSMM_VLA_ACCESS(8, M, img, oj, oi, ofm1, img1, tj, ti, ofm2, ALPHA, ALPHA, handle->blocksofm, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles, TDVLEN)
                * LIBXSMM_VLA_ACCESS(8, V, img, oj, oi, ifm1, img1, tj, ti, ifm2, ALPHA, ALPHA, handle->blocksifm, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles, TDVLEN);
              }
            }
          }
        }
      }
#endif
    }
  }
}
libxsmm_barrier_wait((libxsmm_barrier*)handle->barrier, ltid);
#ifdef WTIME
t_gemm = __rdtsc() - t_start;
#endif

/* number of tasks that could be run in parallel */
work = handle->blocksofm*handle->blocksifm;
/* compute chunck size */
chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

#ifdef WTIME
t_start = __rdtsc();
#endif
for (job = thr_begin; job < thr_end; job++) {
  ofm1 = job / handle->blocksifm;
  ifm1 = job % handle->blocksifm;
  internal_upd_delweight_transform(
    &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, TDVLEN, TDVLEN),
    &LIBXSMM_VLA_ACCESS(6, U, 0, 0, ofm1, ifm1, 0, 0, ALPHA, handle->blocksofm, handle->blocksifm, TDVLEN, TDVLEN), handle);
}
libxsmm_barrier_wait((libxsmm_barrier*)handle->barrier, ltid);
#ifdef WTIME
t_wt = __rdtsc() - t_start;
#endif

#ifdef WTIME
if (tid == 0) {
  int nOfm = handle->blocksofm*TDVLEN;
  int nIfm = handle->blocksifm*TDVLEN;
  double b_input = 1.0*handle->desc.N*nIfm*(handle->ifhp*handle->ifwp + handle->cwino_upd.jtiles*handle->cwino_upd.itiles*ALPHA*ALPHA) * sizeof(float);
  double b_wt    = 1.0*nOfm*nIfm*(handle->desc.R*handle->desc.S + ALPHA*ALPHA) * sizeof(float);
  double b_output= 1.0*handle->desc.N*nOfm*(handle->ofhp*handle->ofwp + handle->cwino_upd.jtiles*handle->cwino_upd.itiles*ALPHA*ALPHA) * sizeof(float);
  double f_gemm = 2.0*handle->desc.N*nOfm*nIfm*handle->cwino_upd.jtiles*handle->cwino_upd.itiles*ALPHA*ALPHA;
  printf("Time: i=%8.3f  w=%8.3f  o=%8.3f         g=%8.3f\n", t_input/1000.0, t_wt/1000.0, t_output/1000.0, t_gemm/1000.0);
  printf("BW:   i=%8.3f  w=%8.3f  o=%8.3f (b/c)   g=%8.3f (f/c)\n\n", b_input/t_input, b_wt/t_wt, b_output/t_output, f_gemm/t_gemm);
}
#endif

