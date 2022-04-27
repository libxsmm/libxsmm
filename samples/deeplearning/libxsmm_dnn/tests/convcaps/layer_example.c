/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
#include <omp.h>
#endif
#if defined(USE_BLAS) || defined(USE_IM2COL)
#include <mkl.h>
#endif

#define CHANNEL_BLOCKING 64
#define LP_BLOCKING 2

/* function-pointer to LIBXSMM kernel */
libxsmm_gemmfunction fwd_brgemmz;
libxsmm_gemmfunction fwd_brgemma;
libxsmm_blasint prec_bf16 = 0;

typedef struct {
  int nImg;
  int nIfm;
  int nOfm;
  int ifhp;
  int ifwp;
  int ifh;
  int ifw;
  int ofhp;
  int ofwp;
  int ofh;
  int ofw;
  int pad_h;
  int pad_w;
  int pad_h_in;
  int pad_w_in;
  int pad_h_out;
  int pad_w_out;
  int kh;
  int kw;
  int stride_h;
  int stride_w;
  int RK;
  int Mh;
  int Mw;
} naive_conv_t;

typedef struct {
  int nImg;
  int nBIfm;
  int nbIfm;
  int nBOfm;
  int nbOfm;
  int nlpb;
  int ifhp;
  int ifwp;
  int ifh;
  int ifw;
  int ofhp;
  int ofwp;
  int ofh;
  int ofw;
  int pad_h;
  int pad_w;
  int pad_h_in;
  int pad_w_in;
  int pad_h_out;
  int pad_w_out;
  int kh;
  int kw;
  int stride_h;
  int stride_w;
  int RK;
  int Mh;
  int Mw;
  unsigned long long brcount;
} gemm_conv_t;

typedef struct {
  double max_rel_err;
  double max_abs_err;
  double l2_rel_err;
  double one_norm_ref;
  double one_norm_test;
} correctness_t;

LIBXSMM_INLINE void zero_buf(float* buf, long size) {
  int i;
#if defined(_OPENMP)
  #pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; ++i) {
    buf[i] = 0.0f;
  }
}

LIBXSMM_INLINE void zero_buf_bf16(libxsmm_bfloat16* buf, size_t size) {
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = 0;
  }
}

LIBXSMM_INLINE void copy_buf(float* src, float* dst, long size) {
  int i;
#if defined(_OPENMP)
  #pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; ++i) {
    dst[i] = src[i];
  }
}

LIBXSMM_INLINE void init_buf(float* buf, long size, int initPos, int initOne)
{
  int i;
  zero_buf(buf, size);
  for (i = 0; i < size; ++i) {
    buf[i] = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? libxsmm_rng_f64() : (0.05 - libxsmm_rng_f64()/10.0)));
  }
}

LIBXSMM_INLINE void set_zeropad_nchw(float* nchw, int N, int C, int H, int W, int Mh, int RK, int pad_h, int pad_w)
{
  LIBXSMM_VLA_DECL(6, float, input, nchw, C, H, W, Mh, RK);
  int n, h, w, c, m, rk;

  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          for ( m = 0; m < Mh; m++ ) {
            for ( rk = 0; rk < RK; rk++ ) {
              if(h < pad_h || h >= H-pad_h || w < pad_w || w >= W-pad_w)
                LIBXSMM_VLA_ACCESS(6, input, n, c, h, w, m, rk, C, H, W, Mh, RK) = 0.0;
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void compare_buf(float* ref, float* test, long size, correctness_t* norms)
{
  int i;
  double diff, rel_err;

  norms->max_rel_err = 0.;
  norms->max_abs_err = 0.;
  norms->l2_rel_err = 0.;
  norms->one_norm_ref = 0.;
  norms->one_norm_test = 0.;

  for (i = 0; i < size; ++i) {
    norms->one_norm_ref += (double)ref[i];
    norms->one_norm_test += (double)test[i];
    diff = fabs((double)ref[i] - (double)test[i]);
    norms->l2_rel_err += (diff*diff);
    rel_err = 0.0;
    if (diff > 0.0 ) {
      rel_err = diff/fabs((double)ref[i]);
    }
    if (rel_err > norms->max_rel_err) {
      norms->max_rel_err = rel_err;
#if 0
      printf("MISMATCH@ %3d: A=%12.8g  B=%12.8g (E:%12.4e) (R:%12.4e)\n", i, ref[i], test[i], diff, rel_err);
#endif
    }
    if (diff > norms->max_abs_err) {
      norms->max_abs_err = diff;
    }
#if 0
    if (diff > 1.0) {
      printf("MISMATCH@ %3d: A=%12.8g  B=%12.8g (E:%12.4e)\n", i, ref[i], test[i], diff);
    }
#endif

  }
  norms->l2_rel_err = sqrt(norms->l2_rel_err);
}


LIBXSMM_INLINE void copy_naiveP_to_GEMM(const float* nchw, float* gemm, int N, int H, int W, int C, int Mh, int RK)
{
  LIBXSMM_VLA_DECL(7,       float, output, gemm, C/CHANNEL_BLOCKING, Mh, RK, H, W, CHANNEL_BLOCKING);
  LIBXSMM_VLA_DECL(6, const float,  input, nchw, H, W, C, Mh, RK);
  int n, h, w, c1, c2, m, rk;

  for ( n = 0; n < N; n++ ) {
    for ( c1 = 0; c1 < C/CHANNEL_BLOCKING; c1++ ) {
      for ( m = 0; m < Mh; m++ ) {
        for ( rk = 0; rk < RK; rk++ ) {
          for ( h = 0; h < H; h++ ) {
            for ( w = 0; w < W; w++ ) {
              for ( c2 = 0; c2 < CHANNEL_BLOCKING; c2++ ) {
                 LIBXSMM_VLA_ACCESS(7, output, n, c1, m, rk, h, w, c2, C/CHANNEL_BLOCKING, Mh, RK, H, W, CHANNEL_BLOCKING) =
                   LIBXSMM_VLA_ACCESS(6,  input, n, h, w, (c1*CHANNEL_BLOCKING)+c2, m, rk, H, W, C, Mh, RK);
              }
            }
          }
        }
      }
    }
  }
}


LIBXSMM_INLINE void copy_GEMM_to_naiveV(const float* gemm, float* nchw, int N, int H, int W, int C, int Mh, int Mw)
{
  LIBXSMM_VLA_DECL(7, const float,  input, gemm, C/CHANNEL_BLOCKING, Mh, Mw, H, W, CHANNEL_BLOCKING);
  LIBXSMM_VLA_DECL(6,       float, output, nchw, H, W, C, Mh, Mw);
  int n, h, w, c1, c2, mi, mj;

  for ( n = 0; n < N; n++ ) {
    for ( c1 = 0; c1 < C/CHANNEL_BLOCKING; c1++ ) {
      for ( mj = 0; mj < Mh; mj++) {
        for ( mi = 0; mi < Mw; mi++) {
          for ( h = 0; h < H; h++ ) {
            for ( w = 0; w < W; w++ ) {
              for ( c2 = 0; c2 < CHANNEL_BLOCKING; c2++ ) {
                LIBXSMM_VLA_ACCESS(6,  output, n, h, w, (c1*CHANNEL_BLOCKING)+c2, mj, mi, H, W, C, Mh, Mw) =
                  LIBXSMM_VLA_ACCESS(7, input, n, c1, mj, mi, h, w, c2, C/CHANNEL_BLOCKING, Mh, Mw, H, W, CHANNEL_BLOCKING);
              }
            }
          }
        }
      }
    }
  }
}


LIBXSMM_INLINE void copy_naiveF_to_GEMM(const float* kcrs, float* gemm, int R, int S, int C, int K, int RK, int Mw)
{
  LIBXSMM_VLA_DECL(8,       float,  output, gemm, C/CHANNEL_BLOCKING, Mw, RK, R, S, CHANNEL_BLOCKING, CHANNEL_BLOCKING);
  LIBXSMM_VLA_DECL(6, const float,   input, kcrs, K, R, S, RK, Mw);
  int r, s, c1, c2, k1, k2, rk, m;

  for ( k1 = 0; k1 < K/CHANNEL_BLOCKING; k1++ ) {
    for ( c1 = 0; c1 < C/CHANNEL_BLOCKING; c1++ ) {
      for ( m = 0; m < Mw; m++ ) {
        for ( rk = 0; rk < RK; rk++ ) {
          for ( r = 0; r < R; r++ ) {
            for ( s = 0; s < S; s++ ) {
              for ( c2 = 0; c2 < CHANNEL_BLOCKING; c2++ ) {
                for ( k2 = 0; k2 < CHANNEL_BLOCKING; k2++ ) {
                  LIBXSMM_VLA_ACCESS(8, output, k1, c1, m, rk, r, s, c2, k2, C/CHANNEL_BLOCKING, Mw, RK, R, S, CHANNEL_BLOCKING, CHANNEL_BLOCKING) =
                    LIBXSMM_VLA_ACCESS(6,  input, (c1*CHANNEL_BLOCKING)+c2, (k1*CHANNEL_BLOCKING)+k2, r, s, rk, m, C, R, S, RK, Mw);
                }
              }
            }
          }
        }
      }
    }
  }
}


LIBXSMM_INLINE void copy_naiveP_to_GEMM_bf16(const libxsmm_bfloat16* nchw, libxsmm_bfloat16* gemm, int N, int H, int W, int C, int Mh, int RK)
{
  LIBXSMM_VLA_DECL(7,       libxsmm_bfloat16, output, gemm, C/CHANNEL_BLOCKING, Mh, RK, H, W, CHANNEL_BLOCKING);
  LIBXSMM_VLA_DECL(6, const libxsmm_bfloat16,  input, nchw, H, W, C, Mh, RK);
  int n, h, w, c1, c2, m, rk;

  for ( n = 0; n < N; n++ ) {
    for ( c1 = 0; c1 < C/CHANNEL_BLOCKING; c1++ ) {
      for ( m = 0; m < Mh; m++ ) {
        for ( rk = 0; rk < RK; rk++ ) {
          for ( h = 0; h < H; h++ ) {
            for ( w = 0; w < W; w++ ) {
              for ( c2 = 0; c2 < CHANNEL_BLOCKING; c2++ ) {
                 LIBXSMM_VLA_ACCESS(7, output, n, c1, m, rk, h, w, c2, C/CHANNEL_BLOCKING, Mh, RK, H, W, CHANNEL_BLOCKING) =
                   LIBXSMM_VLA_ACCESS(6,  input, n, h, w, (c1*CHANNEL_BLOCKING)+c2, m, rk, H, W, C, Mh, RK);
              }
            }
          }
        }
      }
    }
  }
}


LIBXSMM_INLINE void copy_GEMM_to_naiveV_bf16(const libxsmm_bfloat16* gemm, libxsmm_bfloat16* nchw, int N, int H, int W, int C, int Mh, int Mw)
{
  LIBXSMM_VLA_DECL(7, const libxsmm_bfloat16,  input, gemm, C/CHANNEL_BLOCKING, Mh, Mw, H, W, CHANNEL_BLOCKING);
  LIBXSMM_VLA_DECL(6,       libxsmm_bfloat16, output, nchw, H, W, C, Mh, Mw);
  int n, h, w, c1, c2, mi, mj;

  for ( n = 0; n < N; n++ ) {
    for ( c1 = 0; c1 < C/CHANNEL_BLOCKING; c1++ ) {
      for ( mj = 0; mj < Mh; mj++) {
        for ( mi = 0; mi < Mw; mi++) {
          for ( h = 0; h < H; h++ ) {
            for ( w = 0; w < W; w++ ) {
              for ( c2 = 0; c2 < CHANNEL_BLOCKING; c2++ ) {
                LIBXSMM_VLA_ACCESS(6,  output, n, h, w, (c1*CHANNEL_BLOCKING)+c2, mj, mi, H, W, C, Mh, Mw) =
                  LIBXSMM_VLA_ACCESS(7, input, n, c1, mj, mi, h, w, c2, C/CHANNEL_BLOCKING, Mh, Mw, H, W, CHANNEL_BLOCKING);
              }
            }
          }
        }
      }
    }
  }
}


LIBXSMM_INLINE void copy_naiveF_to_GEMM_bf16(const libxsmm_bfloat16* kcrs, libxsmm_bfloat16* gemm, int R, int S, int C, int K, int RK, int Mw)
{
  LIBXSMM_VLA_DECL(9,       libxsmm_bfloat16,  output, gemm, C/CHANNEL_BLOCKING, Mw, RK, R, S, CHANNEL_BLOCKING/LP_BLOCKING, CHANNEL_BLOCKING, LP_BLOCKING);
  LIBXSMM_VLA_DECL(6, const libxsmm_bfloat16,   input, kcrs, K, R, S, RK, Mw);
  int r, s, c1, c2, c3, k1, k2, rk, m;

  for ( k1 = 0; k1 < K/CHANNEL_BLOCKING; k1++ ) {
    for ( c1 = 0; c1 < C/CHANNEL_BLOCKING; c1++ ) {
      for ( m = 0; m < Mw; m++ ) {
        for ( rk = 0; rk < RK; rk++ ) {
          for ( r = 0; r < R; r++ ) {
            for ( s = 0; s < S; s++ ) {
              for ( c2 = 0; c2 < CHANNEL_BLOCKING/LP_BLOCKING; c2++ ) {
                for ( k2 = 0; k2 < CHANNEL_BLOCKING; k2++ ) {
                  for ( c3 = 0; c3 < LP_BLOCKING; c3++ ) {
                    LIBXSMM_VLA_ACCESS(9, output, k1, c1, m, rk, r, s, c2, k2, c3, C/CHANNEL_BLOCKING, Mw, RK, R, S, CHANNEL_BLOCKING/LP_BLOCKING, CHANNEL_BLOCKING, LP_BLOCKING) =
                      LIBXSMM_VLA_ACCESS(6,  input, (c1*CHANNEL_BLOCKING)+(c2*LP_BLOCKING)+c3, (k1*CHANNEL_BLOCKING)+k2, r, s, rk, m, C, R, S, RK, Mw);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE int is_a_ge_zero_and_a_lt_b(int a, int b) {
  return (unsigned int)a < (unsigned int)(b);
}

LIBXSMM_INLINE void naive_convcaps_fp(naive_conv_t* param, const float* input, float* output, const float* filter)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  int pad_h     = param->pad_h;
  int pad_w     = param->pad_w;
  int pad_h_in  = param->pad_h_in;
  int pad_w_in  = param->pad_w_in;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  int RK        = param->RK;
  int Mh        = param->Mh;
  int Mw        = param->Mw;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki, rk, mj, mi;

  LIBXSMM_VLA_DECL(6,       float,  votes_t, output + (pad_w_out * ofwp + pad_h_out), ofhp, ofwp, nOfm, Mh, Mw);
  LIBXSMM_VLA_DECL(6, const float,  poses_t,  input + (pad_w_in * ifwp + pad_h_in), ifhp, ifwp, nIfm, Mh, RK);
  LIBXSMM_VLA_DECL(6, const float, filter_t, filter, nOfm, kh, kw, RK, Mw);

#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki, rk, mj, mi)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (oj = 0; oj < ofh; ++oj) {
        ij = oj * stride_h - pad_h;
        for (oi = 0; oi < ofw; ++oi) {
          ii = oi * stride_w - pad_w;
          for (mj = 0; mj < Mh; ++mj ) {
            for (mi = 0; mi < Mw; ++mi ) {
              LIBXSMM_VLA_ACCESS( 6, votes_t, img, oj, oi, ofm, mj, mi, ofhp, ofwp, nOfm, Mh, Mw) = 0.0f;
              for (ifm = 0; ifm < nIfm; ++ifm) {
                for (kj = 0; kj < kh; ++kj) {
                  /*if(ij+kj < 0 || ij+kj >= ifh) continue;*/
                  for (ki = 0; ki < kw; ++ki) {
                    /*if(ii+ki < 0 || ii+ki >= ifw) continue;*/
                    for (rk = 0; rk < RK; ++rk ) {
                      LIBXSMM_VLA_ACCESS( 6, votes_t, img, oj, oi, ofm, mj, mi, ofhp, ofwp, nOfm, Mh, Mw) +=
                        LIBXSMM_VLA_ACCESS( 6, poses_t, img, ij+kj, ii+ki, ifm, mj, rk, ifhp, ifwp, nIfm, Mh, RK) *
                        LIBXSMM_VLA_ACCESS( 6, filter_t, ifm, ofm, kj, ki, rk, mi, nOfm, kh, kw, RK, Mw);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void gemm_convcaps_fp(gemm_conv_t* param, const float* input, float* output, const float* filter, unsigned long long* aoff, unsigned long long* boff)
{
  int nImg     = param->nImg;
  int nBIfm     = param->nBIfm;
  int nbIfm     = param->nbIfm;
  int nBOfm     = param->nBOfm;
  int nbOfm     = param->nbOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ofh       = param->ofh;
  int pad_h     = param->pad_h;
  int pad_h_in  = param->pad_h_in;
  int pad_w_in  = param->pad_w_in;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int RK        = param->RK;
  int Mh        = param->Mh;
  int Mw        = param->Mw;
  unsigned long long  brcount   = param->brcount;
  /* loop counters */
  int img, ofm1, ifm1, oj, ij, rk, mj, mi;

  LIBXSMM_VLA_DECL(7,       float,  votes_t, output + (pad_w_out * ofwp + pad_h_out), nBOfm, Mh, Mw, ofhp, ofwp, nbOfm);
  LIBXSMM_VLA_DECL(7, const float,  poses_t,  input + (pad_w_in * ifwp + pad_h_in), nBIfm, Mh, RK, ifhp, ifwp, nbIfm);
  LIBXSMM_VLA_DECL(8, const float, filter_t, filter, nBIfm, Mw, RK, kh, kw, nbIfm, nbOfm);

#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm1, ifm1, oj, ij, mj, mi, rk)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm1 = 0; ofm1 < nBOfm; ++ofm1) {
      for (mj = 0; mj < Mh; ++mj ) {
        for (mi = 0; mi < Mw; ++mi ) {
          for (ifm1 = 0; ifm1 < nBIfm; ++ifm1) {
            for (rk = 0; rk < RK; ++rk ) {
              for (oj = 0; oj < ofh; ++oj) {
                ij = oj * stride_h - pad_h;
                {
                  libxsmm_gemm_param gemm_param;
                  gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(8, filter_t, ofm1, ifm1, mi, rk, 0,  0, 0, 0, nBIfm, Mw, RK, kh, kw, nbIfm, nbOfm);
                  gemm_param.a.secondary = aoff;
                  gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(7,  poses_t,  img, ifm1, mj, rk, ij, 0, 0, nBIfm, Mh, RK, ifhp, ifwp, nbIfm);
                  gemm_param.b.secondary = boff;
                  gemm_param.c.primary = &LIBXSMM_VLA_ACCESS(7,  votes_t,  img, ofm1, mj, mi, oj, 0, 0, nBOfm, Mh, Mw, ofhp, ofwp, nbOfm);
                  gemm_param.op.tertiary = &brcount;

                  if ( rk == 0 && ifm1 == 0 ) {
                    fwd_brgemmz( &gemm_param );
                  } else {
                    fwd_brgemma( &gemm_param );
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void gemm_convcaps_fp_bf16(gemm_conv_t* param, const libxsmm_bfloat16* input, libxsmm_bfloat16* output, const libxsmm_bfloat16* filter, unsigned long long* aoff, unsigned long long* boff)
{
  int nImg     = param->nImg;
  int nBIfm     = param->nBIfm;
  int nbIfm     = param->nbIfm;
  int nBOfm     = param->nBOfm;
  int nbOfm     = param->nbOfm;
  int nlpb      = param->nlpb;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ofh       = param->ofh;
  int pad_h     = param->pad_h;
  int pad_h_in  = param->pad_h_in;
  int pad_w_in  = param->pad_w_in;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int RK        = param->RK;
  int Mh        = param->Mh;
  int Mw        = param->Mw;
  unsigned long long  brcount   = param->brcount;
  /* loop counters */
  int img, ofm1, ifm1, oj, ij, rk, mj, mi;

  LIBXSMM_VLA_DECL(7,       libxsmm_bfloat16,  votes_t, output + (pad_w_out * ofwp + pad_h_out), nBOfm, Mh, Mw, ofhp, ofwp, nbOfm);
  LIBXSMM_VLA_DECL(7, const libxsmm_bfloat16,  poses_t,  input + (pad_w_in * ifwp + pad_h_in), nBIfm, Mh, RK, ifhp, ifwp, nbIfm);
  LIBXSMM_VLA_DECL(9, const libxsmm_bfloat16, filter_t, filter, nBIfm, Mw, RK, kh, kw, nbIfm/nlpb, nbOfm, nlpb);

#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm1, ifm1, oj, ij, mj, mi, rk)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm1 = 0; ofm1 < nBOfm; ++ofm1) {
      for (mj = 0; mj < Mh; ++mj ) {
        for (mi = 0; mi < Mw; ++mi ) {
          for (ifm1 = 0; ifm1 < nBIfm; ++ifm1) {
            for (rk = 0; rk < RK; ++rk ) {
              for (oj = 0; oj < ofh; ++oj) {
                ij = oj * stride_h - pad_h;
                {
                  libxsmm_gemm_param gemm_param;
                  gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(9, filter_t, ofm1, ifm1, mi, rk, 0,  0, 0, 0, 0, nBIfm, Mw, RK, kh, kw, nbIfm/nlpb, nbOfm, nlpb);
                  gemm_param.a.secondary = aoff;
                  gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(7,  poses_t,  img, ifm1, mj, rk, ij, 0, 0, nBIfm, Mh, RK, ifhp, ifwp, nbIfm);
                  gemm_param.b.secondary = boff;
                  gemm_param.c.primary = &LIBXSMM_VLA_ACCESS(7,  votes_t,  img, ofm1, mj, mi, oj, 0, 0, nBOfm, Mh, Mw, ofhp, ofwp, nbOfm);
                  gemm_param.op.tertiary = &brcount;

                  if ( rk == 0 && ifm1 == 0 ) {
                    fwd_brgemmz( &gemm_param );
                  } else {
                    fwd_brgemma( &gemm_param );
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void compute_broff(gemm_conv_t* param, unsigned long long* aoff, unsigned long long* boff) {
  int nbIfm     = param->nbIfm;
  int nbOfm     = param->nbOfm;
  int ifwp      = param->ifwp;
  int kh        = param->kh;
  int kw        = param->kw;
  /* loop counters */
  int kj, ki, i;

  i = 0;
  for (kj = 0; kj < kh; ++kj) {
    for (ki = 0; ki < kw; ++ki) {
      aoff[i] = (kj*(kw*nbIfm*nbOfm) + ki*(nbIfm*nbOfm))*sizeof(float);
      boff[i] = (kj*(ifwp*nbIfm) + ki*(nbIfm))*sizeof(float);
      i++;
    }
  }
}

LIBXSMM_INLINE void compute_broff_bf16(gemm_conv_t* param, unsigned long long* aoff, unsigned long long* boff) {
  int nbIfm     = param->nbIfm;
  int nbOfm     = param->nbOfm;
  int ifwp      = param->ifwp;
  int kh        = param->kh;
  int kw        = param->kw;
  /* loop counters */
  int kj, ki, i;

  i = 0;
  for (kj = 0; kj < kh; ++kj) {
    for (ki = 0; ki < kw; ++ki) {
      aoff[i] = (kj*(kw*nbIfm*nbOfm) + ki*(nbIfm*nbOfm))*sizeof(libxsmm_bfloat16);
      boff[i] = (kj*(ifwp*nbIfm) + ki*(nbIfm))*sizeof(libxsmm_bfloat16);
      i++;
    }
  }
}

int main(int argc, char* argv[])
{
  float *naive_input, *naive_output, *naive_filter;
  libxsmm_bfloat16 *naive_input_bf16, *naive_output_bf16, *naive_filter_bf16;
  float *gemm_input, *gemm_output, *gemm_filter;
  libxsmm_bfloat16 *gemm_input_bf16, *gemm_output_bf16, *gemm_filter_bf16;
  float *check_output;
  libxsmm_bfloat16 *check_output_bf16;
  unsigned long long *aoff, *boff;

  int ifhp, ifwp, ofhp, ofwp, ofh, ofw;
  int stride_h, stride_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out;
  int ldx;
  int brcount;

  libxsmm_gemm_shape l_shape;
  libxsmm_gemm_batch_reduce_config l_brconfig;
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags = LIBXSMM_PREFETCH_NONE;

  naive_conv_t naive_param;
  gemm_conv_t gemm_param;

  correctness_t norms_fwd;

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int iters = 100;         /* repetitions of benchmark */
  int ifw = 16;           /* input width, "W" */
  int ifh = 16;           /* input height, "H" */
  int nImg = 128;          /* mini-batch size, "N" */
  int nIfm = 128;         /* number of input feature maps, "C" */
  int nOfm = 256;         /* number of output feature maps, "K" */
  int kh = 3;             /* filter height, "R" */
  int kw = 3;             /* filter width, "S" */
  int pad_h = 0;            /* padding in output */
  int pad_w = 0;            /* padding in output */
  int stride = 2;         /* stride when accessing inputs */
  int Mh = 4;
  int Mw = 4;
  int RK = 4;
  char type = 'F';        /* 'A': ALL, 'F': FP, 'B': BP, 'U', WU */
#if defined(_OPENMP)
  int nThreads = omp_get_max_threads();      /* number of threads */
#else
  int nThreads = 1;       /* number of threads */
#endif

  unsigned long long l_start, l_end;
  double l_total = 0.0;
  double flops = 0.0;
  int i;

  memset(&norms_fwd, 0, sizeof(norms_fwd));

  naive_input = NULL;
  naive_output = NULL;
  naive_filter = NULL;
  naive_input_bf16 = NULL;
  naive_output_bf16 = NULL;
  naive_filter_bf16 = NULL;
  gemm_input = NULL;
  gemm_output = NULL;
  gemm_filter = NULL;
  gemm_input_bf16 = NULL;
  gemm_output_bf16 = NULL;
  gemm_filter_bf16 = NULL;
  check_output = NULL;
  check_output_bf16 = NULL;

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("\n\n\nUsage: %s iters H W N C K R S pad stride type(F,B,U,A)\n\n\n", argv[0]);
    return -1;
  }
  libxsmm_rng_set_seed(1);

  /* reading new values from cli */
  i = 1;
  if (argc > i) iters      = atoi(argv[i++]);
  if (argc > i) ifw        = atoi(argv[i++]);
  if (argc > i) ifh        = atoi(argv[i++]);
  if (argc > i) nImg       = atoi(argv[i++]);
  if (argc > i) nIfm       = atoi(argv[i++]);
  if (argc > i) nOfm       = atoi(argv[i++]);
  if (argc > i) kw         = atoi(argv[i++]);
  if (argc > i) kh         = atoi(argv[i++]);
  if (argc > i) pad_w      = atoi(argv[i++]);
  if (argc > i) pad_h      = atoi(argv[i++]);
  if (argc > i) stride     = atoi(argv[i++]);
  if (argc > i) RK         = atoi(argv[i++]);
  if (argc > i) Mw         = atoi(argv[i++]);
  if (argc > i) Mh         = atoi(argv[i++]);
  if (argc > i) type       = *(argv[i++]);

  /* apply stride in both dimensions */
  stride_w = stride;
  stride_h = stride;

  /* handle physical padding */
#ifdef USE_PHYSICAL_PADDING
#error "physical padding is not supported right now!"
  pad_h_in = pad_h;
  pad_w_in = pad_w;
  pad_h_out = 0;
  pad_w_out = 0;
#else
  pad_h_in = 0;
  pad_w_in = 0;
  pad_h_out = 0;
  pad_w_out = 0;
#endif

  /* deriving some values image size */
  ofh = (ifh + 2 * pad_h - kh) / stride_h + 1;
  ofw = (ifw + 2 * pad_w - kw) / stride_w + 1;
  ifhp = ifh + 2 * pad_h_in;
  ifwp = ifw + 2 * pad_w_in;
  ofhp = ofh + 2 * pad_h_out;
  ofwp = ofw + 2 * pad_w_out;

  /* set struct for naive convolution */
  naive_param.nImg = nImg;
  naive_param.nIfm = nIfm;
  naive_param.nOfm = nOfm;
  naive_param.ifhp = ifhp;
  naive_param.ifwp = ifwp;
  naive_param.ofhp = ofhp;
  naive_param.ofwp = ofwp;
  naive_param.ifh = ifh;
  naive_param.ifw = ifw;
  naive_param.ofh = ofh;
  naive_param.ofw = ofw;
  naive_param.pad_h = pad_h;
  naive_param.pad_w = pad_w;
  naive_param.pad_h_in = pad_h_in;
  naive_param.pad_w_in = pad_w_in;
  naive_param.pad_h_out = pad_h_out;
  naive_param.pad_w_out = pad_w_out;
  naive_param.kh = kh;
  naive_param.kw = kw;
  naive_param.stride_h = stride_h;
  naive_param.stride_w = stride_w;
  naive_param.RK = RK;
  naive_param.Mh = Mh;
  naive_param.Mw = Mw;

  /* set struct for naive convolution */
  gemm_param.nImg = nImg;
  gemm_param.nBIfm = nIfm/CHANNEL_BLOCKING;
  gemm_param.nbIfm = CHANNEL_BLOCKING;
  gemm_param.nBOfm = nOfm/CHANNEL_BLOCKING;
  gemm_param.nbOfm = CHANNEL_BLOCKING;
  if (prec_bf16 == 0 ) {
    gemm_param.nlpb = 1;
  } else {
    gemm_param.nlpb = LP_BLOCKING;
  }
  gemm_param.ifhp = ifhp;
  gemm_param.ifwp = ifwp;
  gemm_param.ofhp = ofhp;
  gemm_param.ofwp = ofwp;
  gemm_param.ifh = ifh;
  gemm_param.ifw = ifw;
  gemm_param.ofh = ofh;
  gemm_param.ofw = ofw;
  gemm_param.pad_h = pad_h;
  gemm_param.pad_w = pad_w;
  gemm_param.pad_h_in = pad_h_in;
  gemm_param.pad_w_in = pad_w_in;
  gemm_param.pad_h_out = pad_h_out;
  gemm_param.pad_w_out = pad_w_out;
  gemm_param.kh = kh;
  gemm_param.kw = kw;
  gemm_param.stride_h = stride_h;
  gemm_param.stride_w = stride_w;
  gemm_param.RK = RK;
  gemm_param.Mh = Mh;
  gemm_param.Mw = Mw;

  /* compute brcount */
  brcount = kh*kw;
  gemm_param.brcount = brcount;

  /* some empty lines at the beginning */
  printf("\n\n\n");

  /* print some summary */
  printf("##########################################\n");
  printf("#                Setting Up              #\n");
  printf("##########################################\n");
  printf("PARAMS: W:%d  H:%d  N:%d  C:%d  K:%d  R:%d  S:%d  P:%d Q:%d STRIDE: %d RK: %d Mh: %d Mw: %d\n", ifw, ifh, nImg, nIfm, nOfm, kw, kh, ofh, ofw, stride, RK, Mh, Mw);
  printf("PARAMS: ITERS:%d  Threads:%d\n", iters, nThreads);
  printf(" InImg %dx%d Padded (%dx%d)\n", ifh, ifw, ifhp, ifwp);
  printf("OutImg %dx%d Padded (%dx%d)\n", ofh, ofw, ofhp, ofwp);
  printf("SIZE Poses  (MB): %10.2f MiB\n", (double)(nImg*nIfm*ifhp*ifwp*Mh*RK*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Votes (MB): %10.2f MiB\n", (double)(nImg*nOfm*ofhp*ofwp*Mh*Mw*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Poses   (1): %10.2f MiB\n", (double)(1*nIfm*ifhp*ifwp*Mh*RK*   sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Votes  (1): %10.2f MiB\n", (double)(1*nOfm*ofhp*ofwp*Mh*Mw*   sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Weight     : %10.2f MiB\n", (double)(nIfm*nOfm*kw*kh*Mw*RK*    sizeof(float))/(1024.0*1024.0) );

  /* check for pass to run */
  if (type != 'A' && type != 'F' && type != 'B' && type != 'U') {
    printf("\ntype needs to be 'A' (All), 'F' (FP only), 'B' (BP only), 'U' (WU only)\n\n\n");
    return -1;
  }

  if ((nIfm % CHANNEL_BLOCKING != 0) || (nOfm % CHANNEL_BLOCKING != 0) ) {
    printf("\nThis code only works for ofm/ifm mod %i = 0!\n\n\n", CHANNEL_BLOCKING);
    return -1;
  }

  if (pad_w !=0 || pad_h !=0 || pad_h_in != 0 || pad_w_in != 0 || pad_h_out !=0 || pad_w_out != 0) {
    printf("\nThis code doesn't support padding right now\n!");
    return -1;
  }

  /* apply stride in both dimensions */
  /* JIT GEMM kernel */
  ldx = stride_w*CHANNEL_BLOCKING;
  if ( prec_bf16 == 0 ) {
    l_shape = libxsmm_create_gemm_shape( CHANNEL_BLOCKING, ofwp, CHANNEL_BLOCKING,
                                         CHANNEL_BLOCKING, ldx, CHANNEL_BLOCKING,
                                         LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
  } else {
    l_flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
    l_shape = libxsmm_create_gemm_shape( CHANNEL_BLOCKING, ofwp, CHANNEL_BLOCKING,
                                         CHANNEL_BLOCKING, ldx, CHANNEL_BLOCKING,
                                         LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32 );
  }
  l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_OFFSET, 0, 0, brcount );
  fwd_brgemma = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
  l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
  fwd_brgemmz = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );

  printf("BRGEMM FWD col-major: m=%d, n=%d, k=%d, lda=%d, ldb=%d, ldc=%d, transa='n', transb='n', alpha=1.0, beta=1.0, brcount=%d\n", CHANNEL_BLOCKING, ofwp, CHANNEL_BLOCKING, CHANNEL_BLOCKING, stride_w*CHANNEL_BLOCKING, CHANNEL_BLOCKING, brcount);

  /* allocate data */
  naive_input           = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*Mh*RK*sizeof(float), 2097152);
  naive_output          = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*Mh*Mw*sizeof(float), 2097152);
  naive_filter          = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*Mw*RK*    sizeof(float), 2097152);
  if (prec_bf16 == 0) {
    gemm_input            = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*Mh*RK*sizeof(float), 2097152);
    gemm_output           = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*Mh*Mw*sizeof(float), 2097152);
    gemm_filter           = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*Mw*RK*    sizeof(float), 2097152);
  } else {
    naive_input_bf16      = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*Mh*RK*sizeof(libxsmm_bfloat16), 2097152);
    naive_output_bf16     = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*Mh*Mw*sizeof(libxsmm_bfloat16), 2097152);
    naive_filter_bf16     = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*Mw*RK*    sizeof(libxsmm_bfloat16), 2097152);
    gemm_input_bf16       = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*Mh*RK*sizeof(libxsmm_bfloat16), 2097152);
    gemm_output_bf16      = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*Mh*Mw*sizeof(libxsmm_bfloat16), 2097152);
    gemm_filter_bf16      = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*Mw*RK*    sizeof(libxsmm_bfloat16), 2097152);
    check_output_bf16     = (libxsmm_bfloat16*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*Mh*Mw*sizeof(libxsmm_bfloat16), 2097152);
  }
  check_output          = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*Mh*Mw*sizeof(float), 2097152);
  aoff                  = (unsigned long long*)libxsmm_aligned_malloc( brcount*sizeof(unsigned long long), 2097152);
  boff                  = (unsigned long long*)libxsmm_aligned_malloc( brcount*sizeof(unsigned long long), 2097152);

  /* initialize data */
  init_buf(naive_input,                     nImg*nIfm*ifhp*ifwp*Mh*RK, 0, 0);
  set_zeropad_nchw(naive_input, nImg, nIfm, ifhp, ifwp, Mh, RK, pad_h_in, pad_w_in);
  init_buf(naive_filter,                    nOfm*nIfm*kh*kw*Mw*RK, 0, 0);
  zero_buf(naive_output,                    nImg*nOfm*ofhp*ofwp*Mw*Mh);

  if (prec_bf16 == 0) {
    /* copy data into GEMM optimized format */
    copy_naiveP_to_GEMM(naive_input,  gemm_input,  nImg, ifhp, ifwp, nIfm, Mh, RK);
    copy_naiveF_to_GEMM(naive_filter, gemm_filter, kh, kw, nIfm, nOfm, RK, Mw);
    zero_buf(gemm_output,                              nImg*nOfm*ofhp*ofwp*Mw*Mh);

    /* compute BRGEMM offsets */
    compute_broff( &gemm_param, aoff, boff );
  } else {
    /* copy data to bf16 */
    libxsmm_rne_convert_fp32_bf16( naive_input,     naive_input_bf16,     nImg*nIfm*ifhp*ifwp*Mh*RK );
    libxsmm_rne_convert_fp32_bf16( naive_filter,    naive_filter_bf16,    nOfm*nIfm*kh*kw*Mw*RK );

    /* copy data into GEMM optimized format */
    copy_naiveP_to_GEMM_bf16(naive_input_bf16,  gemm_input_bf16,  nImg, ifhp, ifwp, nIfm, Mh, RK);
    copy_naiveF_to_GEMM_bf16(naive_filter_bf16, gemm_filter_bf16, kh, kw, nIfm, nOfm, RK, Mw);
    zero_buf_bf16(gemm_output_bf16,                              nImg*nOfm*ofhp*ofwp*Mw*Mh);

    /* compute BRGEMM offsets */
    compute_broff_bf16( &gemm_param, aoff, boff );
  }

  /* check correctness forward */
  if (type == 'A' || type == 'F') {
    printf("##########################################\n");
    printf("#   Correctness - FWD (custom-Storage)   #\n");
    printf("##########################################\n");
    /* run naive convolution */
    naive_convcaps_fp(&naive_param, naive_input, naive_output, naive_filter);
    if (prec_bf16 == 0) {
      gemm_convcaps_fp(&gemm_param, gemm_input, gemm_output, gemm_filter, aoff, boff);
      copy_GEMM_to_naiveV(gemm_output, check_output,  nImg, ofhp, ofwp, nOfm, Mh, Mw);
    } else {
      gemm_convcaps_fp_bf16(&gemm_param, gemm_input_bf16, gemm_output_bf16, gemm_filter_bf16, aoff, boff);
      copy_GEMM_to_naiveV_bf16(gemm_output_bf16, check_output_bf16,  nImg, ofhp, ofwp, nOfm, Mh, Mw);
      /* copy data to FP32 */
      libxsmm_convert_bf16_f32( check_output_bf16, check_output, nImg*nOfm*ofhp*ofwp*Mh*Mw );
    }
    /* compare */
    compare_buf(naive_output, check_output, nImg*nOfm*ofhp*ofwp*Mh*Mw, &norms_fwd);
    printf("             1-norm of reference: %f\n", norms_fwd.one_norm_ref);
    printf("             1-norm of GEMM-code: %f\n", norms_fwd.one_norm_test);
    printf("      L2-error-norm of GEMM-code: %f\n", norms_fwd.l2_rel_err);
    printf("    inf-norm of comp. rel. error: %f\n", norms_fwd.max_rel_err);
    printf("    inf-norm of comp. abs. error: %f\n", norms_fwd.max_abs_err);
  }

  /* benchmark forward */
  if (type == 'A' || type == 'F') {
    printf("##########################################\n");
    printf("#   Performance - FWD (custom-Storage)   #\n");
    printf("##########################################\n");
    /* run LIBXSMM convolution for performance */
    l_start = libxsmm_timer_tick();
    for (i = 0; i < iters; ++i) {
      if (prec_bf16 == 0) {
        gemm_convcaps_fp(&gemm_param, gemm_input, gemm_output, gemm_filter, aoff, boff);
      } else {
        gemm_convcaps_fp_bf16(&gemm_param, gemm_input_bf16, gemm_output_bf16, gemm_filter_bf16, aoff, boff);
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)RK * (double)Mh * (double)Mw * (double)iters;

    printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

    printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
       ifw, ifh, kw, kh, stride, pad_h, pad_w, RK, Mh, Mw, ((double)(l_total/iters)), (flops*1e-9)/l_total,
       norms_fwd.max_rel_err, norms_fwd.max_abs_err, norms_fwd.l2_rel_err, norms_fwd.one_norm_ref, norms_fwd.one_norm_test );
  }

  /* deallocate data */
  libxsmm_free(naive_input);
  libxsmm_free(naive_output);
  libxsmm_free(naive_filter);
  if (prec_bf16 == 0) {
    libxsmm_free(gemm_input);
    libxsmm_free(gemm_output);
    libxsmm_free(gemm_filter);
  } else {
    libxsmm_free(naive_input_bf16);
    libxsmm_free(naive_output_bf16);
    libxsmm_free(naive_filter_bf16);
    libxsmm_free(gemm_input_bf16);
    libxsmm_free(gemm_output_bf16);
    libxsmm_free(gemm_filter_bf16);
    libxsmm_free(check_output_bf16);
  }
  libxsmm_free(check_output);
  libxsmm_free(aoff);
  libxsmm_free(boff);

  /* some empty lines at the end */
  printf("\n\n\n");

  return 0;
}

