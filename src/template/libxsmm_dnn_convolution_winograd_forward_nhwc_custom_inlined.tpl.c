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
/* Kunal Banerjee (Intel Corp.), Alexander Heinecke (Intel Corp.)
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
libxsmm_convfunction jitted_conv_fp;

LIBXSMM_VLA_DECL(5, const float, input, (const float*)handle->reg_input->data, handle->ifhp, handle->ifwp, handle->blocksifm, TDVLEN);
LIBXSMM_VLA_DECL(5, float, output, (float*)handle->reg_output->data, handle->ofhp, handle->ofwp, handle->blocksofm, TDVLEN);
LIBXSMM_VLA_DECL(6, float, weight, (float*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, TDVLEN, TDVLEN);
/*LIBXSMM_VLA_DECL(2, float, bias, handle->bias->data, TDVLEN);*/

LIBXSMM_VLA_DECL(6, float, U,   (float*)handle->scratch1, ALPHA, handle->blocksofm, handle->blocksifm, TDVLEN, TDVLEN);
LIBXSMM_VLA_DECL(8, float, V,   (float*)handle->scratch3, ALPHA, ALPHA, handle->cwino_fwd.bimg, handle->cwino_fwd.jtiles, handle->cwino_fwd.itiles, handle->blocksifm, TDVLEN);
LIBXSMM_VLA_DECL(8, float, M,   (float*)handle->scratch4, handle->blocksofm, ALPHA, ALPHA, handle->cwino_fwd.bimg, handle->cwino_fwd.jtiles, handle->cwino_fwd.itiles, TDVLEN);

LIBXSMM_ASSUME_ALIGNED(handle->reg_input->data,  64);
LIBXSMM_ASSUME_ALIGNED(handle->reg_output->data, 64);
LIBXSMM_ASSUME_ALIGNED(handle->reg_filter->data, 64);
LIBXSMM_ASSUME_ALIGNED(handle->scratch1, 64);
LIBXSMM_ASSUME_ALIGNED(handle->scratch3, 64);
LIBXSMM_ASSUME_ALIGNED(handle->scratch4, 64);

/* computing first logical thread */
ltid = tid - start_thread;
libxsmm_barrier_init((libxsmm_barrier*)handle->barrier, ltid);

/* #define FTIME */
#ifdef FTIME
#define FTIME_REPEAT (64)
#define FTIME_WARMUP (4)
static int ftime_cnt = 0;
static unsigned long long t_input  = 0;
static unsigned long long t_wt     = 0;
static unsigned long long t_output = 0;
static unsigned long long t_gemm   = 0;
unsigned long long t_start  = 0;
if (0 == tid) {
  if (FTIME_WARMUP == ftime_cnt) {
    t_input = 0;
    t_wt = 0;
    t_output = 0;
    t_gemm = 0;
  }
  ++ftime_cnt;
}
#endif

/* number of tasks that could be run in parallel */
work = handle->desc.N*handle->blocksifm;
/* compute chunck size */
chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

#ifdef FTIME
t_start = __rdtsc();
#endif
for (job = thr_begin; job < thr_end; job++) {
  img  = job / handle->blocksifm;
  img1 = img / handle->cwino_fwd.bimg;
  img2 = img % handle->cwino_fwd.bimg;
  ifm1 = job % handle->blocksifm;
  internal_fwd_input_transform_nhwc_custom(
    &LIBXSMM_VLA_ACCESS(5, input, img, 0, 0, ifm1, 0, handle->ifhp, handle->ifwp, handle->blocksifm, TDVLEN),
    &LIBXSMM_VLA_ACCESS(8, V, img1, 0, 0, img2, 0, 0, ifm1, 0, ALPHA, ALPHA, handle->cwino_fwd.bimg, handle->cwino_fwd.jtiles, handle->cwino_fwd.itiles, handle->blocksifm, TDVLEN), handle);
}
#ifdef FTIME
libxsmm_barrier_wait((libxsmm_barrier*)handle->barrier, ltid);
if (0 == tid) t_input = __rdtsc() - t_start;
#endif

/* number of tasks that could be run in parallel */
work = handle->blocksofm*handle->blocksifm;
/* compute chunck size */
chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

#ifdef FTIME
t_start = __rdtsc();
#endif
for (job = thr_begin; job < thr_end; job++) {
  ofm1 = job / handle->blocksifm;
  ifm1 = job % handle->blocksifm;
  internal_fwd_weight_transform(
    &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, TDVLEN, TDVLEN),
    &LIBXSMM_VLA_ACCESS(6, U, 0, 0, ofm1, ifm1, 0, 0, ALPHA, handle->blocksofm, handle->blocksifm, TDVLEN, TDVLEN), handle);
}
libxsmm_barrier_wait((libxsmm_barrier*)handle->barrier, ltid);
#ifdef FTIME
if (0 == tid) t_wt = __rdtsc() - t_start;
#endif

/* number of tasks that could be run in parallel */
work = ALPHA * ALPHA * handle->blocksofm;
/* compute chunck size */
chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

#ifdef FTIME
t_start = __rdtsc();
#endif
for (img1 = 0; img1 < (int)(handle->desc.N/handle->cwino_fwd.bimg); img1++) {
  for (job = thr_begin; job < thr_end; job++) {
    oj = job / (ALPHA * handle->blocksofm);
    oi = (job % (ALPHA * handle->blocksofm)) / handle->blocksofm;
    ofm1 = job % handle->blocksofm;

#if 1
    jitted_conv_fp = (libxsmm_convfunction)handle->code_fwd[job == thr_end - 1 ? 2 : 1].xconv.sconv;
#endif

    if ((int)handle->cwino_fwd.ur_ifm != handle->blocksifm) {
      for (i = 0; i < handle->cwino_fwd.bimg; i++) {
        for (j = 0; j < handle->cwino_fwd.jtiles; j++) {
          for (k = 0; k < handle->cwino_fwd.itiles; k++) {
            LIBXSMM_PRAGMA_SIMD
            for (l = 0; l < TDVLEN; l++) {
              LIBXSMM_VLA_ACCESS(8, M, img1, ofm1, oj, oi, i, j, k, l, handle->blocksofm, ALPHA, ALPHA, handle->cwino_fwd.bimg, handle->cwino_fwd.jtiles, handle->cwino_fwd.itiles, TDVLEN) = 0.0f;
            }
          }
        }
      }
      for (ifm1 = 0; ifm1 < handle->blocksifm; ifm1+=handle->cwino_fwd.ur_ifm) {
#if 1
        jitted_conv_fp(
          &LIBXSMM_VLA_ACCESS(6, U, oj, oi, ofm1, ifm1, 0, 0, ALPHA, handle->blocksofm, handle->blocksifm, TDVLEN, TDVLEN),
          &LIBXSMM_VLA_ACCESS(8, V, img1, oj, oi, 0, 0, 0, ifm1, 0, ALPHA, ALPHA, handle->cwino_fwd.bimg, handle->cwino_fwd.jtiles, handle->cwino_fwd.itiles, handle->blocksifm, TDVLEN),
          &LIBXSMM_VLA_ACCESS(8, M, img1, ofm1, oj, oi, 0, 0, 0, 0, handle->blocksofm, ALPHA, ALPHA, handle->cwino_fwd.bimg, handle->cwino_fwd.jtiles, handle->cwino_fwd.itiles, TDVLEN),
          0, 0, 0);
#else
        int ti, tj;
        int img2;
        int ifm2, ofm2;
        for (img2 = 0; img2 < handle->cwino_fwd.bimg; img2++) {
          for (tj = 0; tj < handle->cwino_fwd.jtiles; tj++) {
            for (ti = 0; ti < handle->cwino_fwd.itiles; ti++) {
              for (ifm2 = 0; ifm2 < TDVLEN; ifm2++) {
                for (ofm2 = 0; ofm2 < TDVLEN; ofm2++) {
                  LIBXSMM_VLA_ACCESS  (8, M, img1, ofm1, oj, oi, img2, tj, ti, ofm2, handle->blocksofm, ALPHA, ALPHA, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles, TDVLEN) +=
                    LIBXSMM_VLA_ACCESS(8, V, img1, oj, oi, img2, tj, ti, ifm1, ifm2, ALPHA, ALPHA, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles, handle->blocksifm, TDVLEN)
                  * LIBXSMM_VLA_ACCESS(6, U, oj, oi, ofm1, ifm1, ifm2, ofm2, ALPHA, handle->blocksofm, handle->blocksifm, TDVLEN, TDVLEN);
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
      /* when ur_ifm == blocksifm, we don't need to initialize M. Instead, we use streaming store to save read BW */
      jitted_conv_fp(
        &LIBXSMM_VLA_ACCESS(6, U, oj, oi, ofm1, 0, 0, 0, ALPHA, handle->blocksofm, handle->blocksifm, TDVLEN, TDVLEN),
        &LIBXSMM_VLA_ACCESS(8, V, img1, oj, oi, 0, 0, 0, 0, 0, ALPHA, ALPHA, handle->cwino_fwd.bimg, handle->cwino_fwd.jtiles, handle->cwino_fwd.itiles, handle->blocksifm, TDVLEN),
        &LIBXSMM_VLA_ACCESS(8, M, img1, ofm1, oj, oi, 0, 0, 0, 0, handle->blocksofm, ALPHA, ALPHA, handle->cwino_fwd.bimg, handle->cwino_fwd.jtiles, handle->cwino_fwd.itiles, TDVLEN),
        0, 0, 0);
#else
      int ti, tj;
      int img2;
      int ifm2, ofm2;
      for (ifm1 = 0; ifm1 < handle->blocksifm; ifm1++) {
        for (img2 = 0; img2 < handle->cwino_fwd.bimg; img2++) {
          for (tj = 0; tj < handle->cwino_fwd.jtiles; tj++) {
            for (ti = 0; ti < handle->cwino_fwd.itiles; ti++) {
              for (ifm2 = 0; ifm2 < TDVLEN; ifm2++) {
                for (ofm2 = 0; ofm2 < TDVLEN; ofm2++) {
                  LIBXSMM_VLA_ACCESS  (8, M, img1, ofm1, oj, oi, img2, tj, ti, ofm2, handle->blocksofm, ALPHA, ALPHA, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles, TDVLEN) +=
                    LIBXSMM_VLA_ACCESS(8, V, img1, oj, oi, img2, tj, ti, ifm1, ifm2, ALPHA, ALPHA, handle->cwino_bwd.bimg, handle->cwino_bwd.jtiles, handle->cwino_bwd.itiles, handle->blocksifm, TDVLEN)
                  * LIBXSMM_VLA_ACCESS(6, U, oj, oi, ofm1, ifm1, ifm2, ofm2, ALPHA, handle->blocksofm, handle->blocksifm, TDVLEN, TDVLEN);
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
#ifdef FTIME
if (0 == tid) t_gemm += __rdtsc() - t_start;
#endif

/* number of tasks that could be run in parallel */
work = handle->desc.N*handle->blocksofm;
/* compute chunck size */
chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

#ifdef FTIME
t_start = __rdtsc();
#endif
for (job = thr_begin; job < thr_end; job++) {
  img1 = job / (handle->blocksofm * handle->cwino_fwd.bimg);
  ofm1 = (job % (handle->blocksofm * handle->cwino_fwd.bimg)) / handle->cwino_fwd.bimg;
  img2 = job % handle->cwino_fwd.bimg;
  img  = img1*handle->cwino_fwd.bimg + img2;
  internal_fwd_output_transform_nhwc_custom(
    &LIBXSMM_VLA_ACCESS(8, M, img1, ofm1, 0, 0, img2, 0, 0, 0, handle->blocksofm, ALPHA, ALPHA, handle->cwino_fwd.bimg, handle->cwino_fwd.jtiles, handle->cwino_fwd.itiles, TDVLEN),
    &LIBXSMM_VLA_ACCESS(5, output, img, 0, 0, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, TDVLEN), 0 /*&bias[ofm1]*/, handle);
}
libxsmm_barrier_wait((libxsmm_barrier*)handle->barrier, ltid);
#ifdef FTIME
if (0 == tid) {
  t_output += __rdtsc() - t_start;

  if (FTIME_REPEAT + FTIME_WARMUP == ftime_cnt) {
    int nOfm = handle->blocksofm*TDVLEN;
    int nIfm = handle->blocksifm*TDVLEN;
    double b_input = 1.0*handle->desc.N*nIfm*(handle->ifhp*handle->ifwp + handle->cwino_fwd.jtiles*handle->cwino_fwd.itiles*ALPHA*ALPHA) * sizeof(float);
    double b_wt    = 1.0*nOfm*nIfm*(handle->desc.R*handle->desc.S + ALPHA*ALPHA) * sizeof(float);
    double b_output= 1.0*handle->desc.N*nOfm*(handle->ofhp*handle->ofwp + handle->cwino_fwd.jtiles*handle->cwino_fwd.itiles*ALPHA*ALPHA) * sizeof(float);
    double f_gemm = 2.0*handle->desc.N*nOfm*nIfm*handle->cwino_fwd.jtiles*handle->cwino_fwd.itiles*ALPHA*ALPHA;
    printf("Time: i=%8.3f  w=%8.3f  o=%8.3f         g=%8.3f\n", t_input/1000.0/FTIME_REPEAT, t_wt/1000.0/FTIME_REPEAT, t_output/1000.0/FTIME_REPEAT, t_gemm/1000.0/FTIME_REPEAT);
    printf("BW:   i=%8.3f  w=%8.3f  o=%8.3f (b/c)   g=%8.3f (f/c)\n\n", b_input/t_input*FTIME_REPEAT, b_wt/t_wt*FTIME_REPEAT, b_output/t_output*FTIME_REPEAT, f_gemm/t_gemm*FTIME_REPEAT);
  }
}
#endif
