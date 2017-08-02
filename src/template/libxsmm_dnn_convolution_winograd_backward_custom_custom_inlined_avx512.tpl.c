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
/* Kunal Banerjee, Alexander Heinecke, Jongsoo Park (Intel Corp.)
******************************************************************************/

int ltid;
int work;
int chunksize;
int thr_begin;
int thr_end;
int job;
int img, img1, img2;
int ifm1;
int ofm1;
int oj;
int oi;
unsigned int i, j, k, l;
typedef libxsmm_sconvfunction libxsmm_convfunction;
libxsmm_convfunction jitted_conv_bp;

LIBXSMM_VLA_DECL(5, float, input,  (float*)handle->grad_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, TDVLEN);
LIBXSMM_VLA_DECL(5, const float, output, (const float*)handle->grad_output->data, handle->blocksofm, handle->ofhp, handle->ofwp, TDVLEN);
LIBXSMM_VLA_DECL(6, float, weight, (float*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, TDVLEN, TDVLEN);

LIBXSMM_VLA_DECL(6, float, U,   (float*)handle->scratch1, ALPHA, handle->blocksifm, handle->blocksofm, TDVLEN, TDVLEN);
LIBXSMM_VLA_DECL(8, float, V,   (float*)handle->scratch3, handle->blocksifm, ALPHA, ALPHA, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles, TDVLEN);
LIBXSMM_VLA_DECL(8, float, M,   (float*)handle->scratch4, ALPHA, ALPHA, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles, handle->blocksofm, TDVLEN);
LIBXSMM_VLA_DECL(5, float, Iwp, (float*)handle->scratchIw, handle->cwino_bwd.itiles*handle->cwino_bwd.jtiles, ALPHA, ALPHA, TDVLEN);
LIBXSMM_VLA_DECL(5, float, Owp, (float*)handle->scratchOw, handle->cwino_bwd.itiles*handle->cwino_bwd.jtiles, ALPHA, ALPHA, TDVLEN);

LIBXSMM_ASSUME_ALIGNED(handle->grad_input->data, 64);
LIBXSMM_ASSUME_ALIGNED(handle->grad_output->data, 64);
LIBXSMM_ASSUME_ALIGNED(handle->reg_filter->data, 64);
LIBXSMM_ASSUME_ALIGNED(handle->scratch1, 64);
LIBXSMM_ASSUME_ALIGNED(handle->scratch3, 64);
LIBXSMM_ASSUME_ALIGNED(handle->scratch4, 64);
LIBXSMM_ASSUME_ALIGNED(handle->scratchIw, 64);
LIBXSMM_ASSUME_ALIGNED(handle->scratchOw, 64);

/* computing first logical thread */
ltid = tid - start_thread;
libxsmm_barrier_init((libxsmm_barrier*)handle->barrier, ltid);

/* #define BTIME */
#ifdef BTIME
#define BTIME_REPEAT (64)
#define BTIME_WARMUP (4)
static int btime_cnt = 0;
static unsigned long long t_input  = 0;
static unsigned long long t_wt     = 0;
static unsigned long long t_output = 0;
static unsigned long long t_gemm   = 0;
unsigned long long t_start  = 0;
if (0 == tid) {
  if (BTIME_WARMUP == btime_cnt) {
    t_input = 0;
    t_wt = 0;
    t_output = 0;
    t_gemm = 0;
  }
  ++btime_cnt;
}
#endif

/* number of tasks that could be run in parallel */
work = handle->desc.N*handle->blocksofm;
/* compute chunck size */
chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

#ifdef BTIME
t_start = __rdtsc();
#endif
for (job = thr_begin; job < thr_end; job++) {
  img  = job / handle->blocksofm;
  img1 = img / handle->cwino_bwd.bimg;
  img2 = img % handle->cwino_bwd.bimg;
  ofm1 = job % handle->blocksofm;
  internal_bwd_input_transform_custom_custom(
    &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, 0, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, TDVLEN),
    &LIBXSMM_VLA_ACCESS(8, M, img1, 0, 0, img2, 0, 0, ofm1, 0, ALPHA, ALPHA, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles, handle->blocksofm, TDVLEN),
    &LIBXSMM_VLA_ACCESS(5, Owp, tid, 0, 0, 0, 0, handle->cwino_bwd.itiles*handle->cwino_bwd.jtiles, ALPHA, ALPHA, TDVLEN), handle);
}
#ifdef BTIME
libxsmm_barrier_wait((libxsmm_barrier*)handle->barrier, ltid);
if (0 == tid) t_input += __rdtsc() - t_start;
#endif

/* number of tasks that could be run in parallel */
work = handle->blocksofm*handle->blocksifm;
/* compute chunck size */
chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

#ifdef BTIME
t_start = __rdtsc();
#endif
for (job = thr_begin; job < thr_end; job++) {
  ofm1 = job / handle->blocksifm;
  ifm1 = job % handle->blocksifm;
  internal_bwd_weight_transform(
    &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, TDVLEN, TDVLEN),
    &LIBXSMM_VLA_ACCESS(6, U, 0, 0, ifm1, ofm1, 0, 0, ALPHA, handle->blocksifm, handle->blocksofm, TDVLEN, TDVLEN), handle);
}
libxsmm_barrier_wait((libxsmm_barrier*)handle->barrier, ltid);
#ifdef BTIME
if (0 == tid) t_wt += __rdtsc() - t_start;
#endif

/* number of tasks that could be run in parallel */
work = ALPHA * ALPHA * handle->blocksifm;
/* compute chunck size */
chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

#ifdef BTIME
t_start = __rdtsc();
#endif
for (img1 = 0; img1 < (int)(handle->desc.N/handle->cwino_bwd.bimg); img1++) {
  for (job = thr_begin; job < thr_end; job++) {
    /* JSP: please see comments in libxsmm_dnn_convolution_winograd_forward_custom_custom_inlined.tpl.c
     * to see why we're using this parallelization and looping order.
     */
    oj = job / (ALPHA * handle->blocksifm);
    oi = (job % (ALPHA * handle->blocksifm)) / handle->blocksifm;
    ifm1 = job % handle->blocksifm;

#if 1
    jitted_conv_bp = (libxsmm_convfunction)handle->code_bwd[job == thr_end - 1 ? 2 : 1].xconv.sconv;
#endif

    if ((int)handle->cwino_bwd.ur_ifm != handle->blocksofm) {
      for (i = 0; i < handle->cwino_bwd.bimg; i++) {
        for (j = 0; j < handle->cwino_bwd.jtiles; j++) {
          for (k = 0; k < handle->cwino_bwd.itiles; k++) {
            LIBXSMM_PRAGMA_SIMD
            for (l = 0; l < TDVLEN; l++) {
              LIBXSMM_VLA_ACCESS(8, V, img1, ifm1, oj, oi, i, j, k, l, handle->blocksifm, ALPHA, ALPHA, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles, TDVLEN) = 0.0f;
            }
          }
        }
      }
      for (ofm1 = 0; ofm1 < handle->blocksofm; ofm1+=handle->cwino_bwd.ur_ifm) {
#if 1
        jitted_conv_bp(
          &LIBXSMM_VLA_ACCESS(6, U, oj, oi, ifm1, ofm1, 0, 0, ALPHA, handle->blocksifm, handle->blocksofm, TDVLEN, TDVLEN),
          &LIBXSMM_VLA_ACCESS(8, M, img1, oj, oi, 0, 0, 0, ofm1, 0, ALPHA, ALPHA, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles, handle->blocksofm, TDVLEN),
          &LIBXSMM_VLA_ACCESS(8, V, img1, ifm1, oj, oi, 0, 0, 0, 0, handle->blocksifm, ALPHA, ALPHA, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles, TDVLEN),
          0, 0, 0);
#else
        int ti, tj;
        int img2;
        int ifm2, ofm2;
        for (img2 = 0; img2 < handle->cwino_bwd.bimg; img2++) {
          for (tj = 0; tj < handle->cwino_bwd.jtiles; tj++) {
            for (ti = 0; ti < handle->cwino_bwd.itiles; ti++) {
              for (ofm2 = 0; ofm2 < TDVLEN; ofm2++) {
                for (ifm2 = 0; ifm2 < TDVLEN; ifm2++) {
                  LIBXSMM_VLA_ACCESS  (8, V, img1, ifm1, oj, oi, img2, tj, ti, ifm2, handle->blocksifm, ALPHA, ALPHA, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles, TDVLEN) +=
                    LIBXSMM_VLA_ACCESS(8, M, img1, oj, oi, img2, tj, ti, ofm1, ofm2, ALPHA, ALPHA, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles, handle->blocksofm, TDVLEN)
                  * LIBXSMM_VLA_ACCESS(6, U, oj, oi, ifm1, ofm1, ofm2, ifm2, ALPHA, handle->blocksifm, handle->blocksofm, TDVLEN, TDVLEN);
                }
              }
            }
          }
        }
#endif
      }
    }
    else {
#if 1
      jitted_conv_bp(
        &LIBXSMM_VLA_ACCESS(6, U, oj, oi, ifm1, 0, 0, 0, ALPHA, handle->blocksifm, handle->blocksofm, TDVLEN, TDVLEN),
        &LIBXSMM_VLA_ACCESS(8, M, img1, oj, oi, 0, 0, 0, 0, 0, ALPHA, ALPHA, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles, handle->blocksofm, TDVLEN),
        &LIBXSMM_VLA_ACCESS(8, V, img1, ifm1, oj, oi, 0, 0, 0, 0, handle->blocksifm, ALPHA, ALPHA, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles, TDVLEN),
        0, 0, 0);
#else
      int ti, tj;
      int img2;
      int ifm2, ofm2;
      for (ofm1 = 0; ofm1 < handle->blocksofm; ofm1++) {
        for (img2 = 0; img2 < handle->cwino_bwd.bimg; img2++) {
          for (tj = 0; tj < handle->cwino_bwd.jtiles; tj++) {
            for (ti = 0; ti < handle->cwino_bwd.itiles; ti++) {
              for (ofm2 = 0; ofm2 < TDVLEN; ofm2++) {
                for (ifm2 = 0; ifm2 < TDVLEN; ifm2++) {
                  LIBXSMM_VLA_ACCESS  (8, V, img1, ifm1, oj, oi, img2, tj, ti, ifm2, handle->blocksifm, ALPHA, ALPHA, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles, TDVLEN) +=
                    LIBXSMM_VLA_ACCESS(8, M, img1, oj, oi, img2, tj, ti, ofm1, ofm2, ALPHA, ALPHA, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles, handle->blocksofm, TDVLEN)
                  * LIBXSMM_VLA_ACCESS(6, U, oj, oi, ifm1, ofm1, ofm2, ifm2, ALPHA, handle->blocksifm, handle->blocksofm, TDVLEN, TDVLEN);
                }
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
#ifdef BTIME
if (0 == tid) t_gemm += __rdtsc() - t_start;
#endif

/* number of tasks that could be run in parallel */
work = handle->desc.N*handle->blocksifm;
/* compute chunck size */
chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

#ifdef BTIME
t_start = __rdtsc();
#endif
for (job = thr_begin; job < thr_end; job++) {
  img1 = job / (handle->blocksifm * handle->cwino_bwd.bimg);
  ifm1 = (job % (handle->blocksifm * handle->cwino_bwd.bimg)) / handle->cwino_bwd.bimg;
  img2 = job % handle->cwino_bwd.bimg;
  img  = img1*handle->cwino_bwd.bimg + img2;
  internal_bwd_output_transform_custom_custom(
    &LIBXSMM_VLA_ACCESS(8, V, img1, ifm1, 0, 0, img2, 0, 0, 0, handle->blocksifm, ALPHA, ALPHA, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles, TDVLEN),
    &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, TDVLEN),
    &LIBXSMM_VLA_ACCESS(5, Iwp, tid, 0, 0, 0, 0, handle->cwino_bwd.itiles*handle->cwino_bwd.jtiles, ALPHA, ALPHA, TDVLEN), handle);
}
libxsmm_barrier_wait((libxsmm_barrier*)handle->barrier, ltid);
#ifdef BTIME
if (0 == tid) {
  t_output += __rdtsc() - t_start;

  if (BTIME_REPEAT + BTIME_WARMUP == btime_cnt) {
    int nOfm = handle->blocksofm*TDVLEN;
    int nIfm = handle->blocksifm*TDVLEN;
    double b_input = 1.0*handle->desc.N*nIfm*(handle->ifhp*handle->ifwp + handle->cwino_bwd.jtiles*handle->cwino_bwd.itiles*ALPHA*ALPHA) * sizeof(float);
    double b_wt    = 1.0*nOfm*nIfm*(handle->desc.R*handle->desc.S + ALPHA*ALPHA) * sizeof(float);
    double b_output= 1.0*handle->desc.N*nOfm*(handle->ofhp*handle->ofwp + handle->cwino_bwd.jtiles*handle->cwino_bwd.itiles*ALPHA*ALPHA) * sizeof(float);
    double f_gemm = 2.0*handle->desc.N*nOfm*nIfm*handle->cwino_bwd.jtiles*handle->cwino_bwd.itiles*ALPHA*ALPHA;
    printf("Time: i=%8.3f  w=%8.3f  o=%8.3f         g=%8.3f\n", t_input/1000.0/BTIME_REPEAT, t_wt/1000.0/BTIME_REPEAT, t_output/1000.0/BTIME_REPEAT, t_gemm/1000.0/BTIME_REPEAT);
    printf("BW:   i=%8.3f  w=%8.3f  o=%8.3f (b/c)   g=%8.3f (f/c)\n\n", b_output/t_input*BTIME_REPEAT, b_wt/t_wt*BTIME_REPEAT, b_input/t_output*BTIME_REPEAT, f_gemm/t_gemm*BTIME_REPEAT);
  }
}
#endif
