/******************************************************************************
** Copyright (c) 2016-2019, Intel Corporation                                **
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
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

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
} naive_conv_t;

typedef struct {
  int N;
  int C;
  int H;
  int W;
  int stride_h;
  int stride_w;
  int norm_type;  /* 0: full batchnorm, 1: batch scaling only */
  int fuse_type;  /* 0: nothing fused, 1: relu fused, 2: elementwise fused, 3: relu and elementwise fused */
} naive_fusedbatchnorm_t;

typedef struct {
  int N;
  int C;
  int K;
  int fuse_type;  /* 0: nothing fused */
} naive_fullyconnected_t;

typedef struct {
  int N;
  int C;
  int H;
  int W;
  int R;
  int S;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int type;
} naive_pooling_t;

/* it's fine to alias in and out */
LIBXSMM_INLINE void truncate_mask_fp32_bfp16(float* in, float* out, unsigned int len) {
  unsigned int i = 0;

  /* truncate buffer to bfp16 */
  for ( i = 0; i < len; ++i ) {
    union libxsmm_bfloat16_hp t;

    t.f = in[i];
    t.i[0] = 0;
    out[i] = t.f;
  }
}

/* it's fine to alias in and out */
LIBXSMM_INLINE void rnaz_mask_fp32_bfp16(float* in, float* out, unsigned int len) {
  unsigned int i = 0;

  /* rnaz buffer to bfp16 */
  for ( i = 0; i < len; ++i ) {
    unsigned int int_round = 0;
    unsigned int do_round = 1;
    const void *const ptr = &int_round;

    int_round = *((unsigned int*)&(in[i]));

    /* we don't round NaN and inf */
    if ( (int_round & 0x7f800000) == 0x7f800000 ) {
      do_round = 0;
    }

    /* perform round nearest tie away from zero */
    if ( do_round != 0 ) {
      int_round = int_round + 0x00008000;
    }

    /* chop bits to create BFP16 in FP32 */
    int_round = int_round & 0xffff0000;

    out[i] = *((float*)ptr);
  }
}

/* it's fine to alias in and out */
LIBXSMM_INLINE void rne_mask_fp32_bfp16(float* in, float* out, unsigned int len) {
  unsigned int i = 0;

  /* rnaz buffer to bfp16 */
  for ( i = 0; i < len; ++i ) {
    unsigned int int_round = 0;
    unsigned int do_round = 1;
    const void *const ptr = &int_round;

    int_round = *((unsigned int*)&(in[i]));

    /* we don't round NaN and inf */
    if ( (int_round & 0x7f800000) == 0x7f800000 ) {
      do_round = 0;
    }

    /* perform round nearest tie even */
    if ( do_round != 0 ) {
      unsigned int fixup = (int_round >> 16) & 1;
      int_round = int_round + 0x00007fff + fixup;
    }

    /* chop bits to create BFP16 in FP32 */
    int_round = int_round & 0xffff0000;

    out[i] = *((float*)ptr);
  }
}

LIBXSMM_INLINE void zero_buf(float* buf, size_t size) {
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = 0.0f;
  }
}

LIBXSMM_INLINE void zero_buf_int16(short* buf, size_t size) {
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = 0;
  }
}

LIBXSMM_INLINE void zero_buf_int32(int* buf, size_t size) {
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = 0;
  }
}

LIBXSMM_INLINE void zero_buf_int8(char* buf, size_t size) {
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = 0;
  }
}

LIBXSMM_INLINE void zero_buf_uint8(unsigned char* buf, size_t size) {
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = 0;
  }
}

LIBXSMM_INLINE void copy_buf(float* src, float* dst, size_t size) {
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    dst[i] = src[i];
  }
}

LIBXSMM_INLINE void copy_buf_int16(short* src, short* dst, size_t size) {
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    dst[i] = src[i];
  }
}

LIBXSMM_INLINE void copy_buf_int8(char* src, char* dst, size_t size) {
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    dst[i] = src[i];
  }
}

LIBXSMM_INLINE void copy_buf_uint8(unsigned char* src, unsigned char* dst, size_t size) {
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    dst[i] = src[i];
  }
}

LIBXSMM_INLINE void init_buf(float* buf, size_t size, int initPos, int initOne)
{
  int i;
  zero_buf(buf, size);
  for (i = 0; i < (int)size; ++i) {
    buf[i] = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? libxsmm_rng_f64() : (0.05 - libxsmm_rng_f64()/10.0)));
  }
}

LIBXSMM_INLINE void init_buf_int16(short* buf, size_t size, int initPos, int initOne)
{
  int i;
  zero_buf_int16(buf, size);
  for (i = 0; i < (int)size; ++i) {
    buf[i] = (short)((initOne != 0) ? 1 : ((initPos != 0) ? (rand()%7) : (rand()%7-3)));
  }
}

LIBXSMM_INLINE void init_buf_int32(int* buf, size_t size, int initPos, int initOne)
{
  int i;
  zero_buf_int32(buf, size);
  for (i = 0; i < (int)size; ++i) {
    buf[i] = (int)((initOne != 0) ? 1 : ((initPos != 0) ? (rand()%7) : (rand()%7-3)));
  }
}

LIBXSMM_INLINE void init_buf_int8(char* buf, size_t size, int initPos, int initOne)
{
  int i;
  zero_buf_int8(buf, size);
  for (i = 0; i < (int)size; ++i) {
    buf[i] = (char)((initOne != 0) ? 1 : ((initPos != 0) ? (rand()%3) : (rand()%3)-1));
  }
}

LIBXSMM_INLINE void init_buf_uint8(unsigned char* buf, size_t size, int initPos, int initOne)
{
  int i;
  LIBXSMM_UNUSED(initPos);
  zero_buf_uint8(buf, size);
  for (i = 0; i < (int)size; ++i) {
    buf[i] = (unsigned char)((initOne != 0) ? 1 : (rand()%3));
  }
}

LIBXSMM_INLINE void set_zeropad_nchw(float* nchw, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXSMM_VLA_DECL(4, float, input, nchw, C, H, W);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          if (h < pad_h || h >= H-pad_h || w < pad_w || w >= W-pad_w)
            LIBXSMM_VLA_ACCESS(4,  input, n, c, h, w, C, H, W) = 0.0;
        }
      }
    }
  }
}

LIBXSMM_INLINE void set_zeropad_nchw_int16(short* nchw, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXSMM_VLA_DECL(4, short, input, nchw, C, H, W);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          if (h < pad_h || h >= H-pad_h || w < pad_w || w >= W-pad_w)
            LIBXSMM_VLA_ACCESS(4,  input, n, c, h, w, C, H, W) = 0;
        }
      }
    }
  }
}

LIBXSMM_INLINE void set_zeropad_nchw_int32(int* nchw, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXSMM_VLA_DECL(4, int, input, nchw, C, H, W);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          if (h < pad_h || h >= H-pad_h || w < pad_w || w >= W-pad_w)
            LIBXSMM_VLA_ACCESS(4,  input, n, c, h, w, C, H, W) = 0;
        }
      }
    }
  }
}

LIBXSMM_INLINE void set_zeropad_nchw_uint8(unsigned char* nchw, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXSMM_VLA_DECL(4, unsigned char, input, nchw, C, H, W);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          if (h < pad_h || h >= H-pad_h || w < pad_w || w >= W-pad_w)
            LIBXSMM_VLA_ACCESS(4,  input, n, c, h, w, C, H, W) = 0;
        }
      }
    }
  }
}

LIBXSMM_INLINE void copy_internal_nchw(float* dst , float* src, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXSMM_VLA_DECL(4, float, input, src, C, H, W);
  LIBXSMM_VLA_DECL(4, float, new_input, dst, C, H+2*pad_h, W+2*pad_w);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          LIBXSMM_VLA_ACCESS(4, new_input, n, c, h+pad_h, w+pad_w, C, H+2*pad_h, W+2*pad_w) =  LIBXSMM_VLA_ACCESS(4,  input, n, c, h, w, C, H, W);
        }
      }
    }
  }
}

LIBXSMM_INLINE void copy_internal_nchw_int16(short* dst , short* src, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXSMM_VLA_DECL(4, short, input, src, C, H, W);
  LIBXSMM_VLA_DECL(4, short, new_input, dst, C, H+2*pad_h, W+2*pad_w);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          LIBXSMM_VLA_ACCESS(4, new_input, n, c, h+pad_h, w+pad_w, C, H+2*pad_h, W+2*pad_w) =  LIBXSMM_VLA_ACCESS(4,  input, n, c, h, w, C, H, W);
        }
      }
    }
  }
}

LIBXSMM_INLINE void copy_internal_nchw_uint8(unsigned char* dst , unsigned char* src, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXSMM_VLA_DECL(4, unsigned char, input, src, C, H, W);
  LIBXSMM_VLA_DECL(4, unsigned char, new_input, dst, C, H+2*pad_h, W+2*pad_w);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          LIBXSMM_VLA_ACCESS(4, new_input, n, c, h+pad_h, w+pad_w, C, H+2*pad_h, W+2*pad_w) =  LIBXSMM_VLA_ACCESS(4,  input, n, c, h, w, C, H, W);
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_copy_NCHW_to_NHWC(const float* nchw, float* nhwc, int N, int H, int W, int C)
{
  LIBXSMM_VLA_DECL(4,       float, output, nhwc, H, W, C);
  LIBXSMM_VLA_DECL(4, const float,  input, nchw, C, H, W);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( h = 0; h < H; h++ ) {
      for ( w = 0; w < W; w++ ) {
        for ( c = 0; c < C; c++ ) {
          LIBXSMM_VLA_ACCESS(4, output, n, h, w, c, H, W, C) =
          LIBXSMM_VLA_ACCESS(4,  input, n, c, h, w, C, H, W);
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_copy_NHWC_to_NCHW(const float* nhwc, float* nchw, int N, int H, int W, int C)
{
  LIBXSMM_VLA_DECL(4,       float, output, nchw, C, H, W);
  LIBXSMM_VLA_DECL(4, const float,  input, nhwc, H, W, C);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( h = 0; h < H; h++ ) {
      for ( w = 0; w < W; w++ ) {
        for ( c = 0; c < C; c++ ) {
          LIBXSMM_VLA_ACCESS(4, output, n, c, h, w, C, H, W) =
          LIBXSMM_VLA_ACCESS(4,  input, n, h, w, c, H, W, C);
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_copy_KCRS_to_RSCK(const float* kcrs, float* rsck, int R, int S, int C, int K)
{
  LIBXSMM_VLA_DECL(4,       float, output, rsck, S, C, K);
  LIBXSMM_VLA_DECL(4, const float,  input, kcrs, C, R, S);
  int r, s, c, k;

  for ( r = 0; r < R; r++ ) {
    for ( s = 0; s < S; s++ ) {
      for ( c = 0; c < C; c++ ) {
        for ( k = 0; k < K; k++ ) {
          LIBXSMM_VLA_ACCESS(4, output, r, s, c, k, S, C, K) =
          LIBXSMM_VLA_ACCESS(4,  input, k, c, r, s, C, R, S);
        }
      }
    }
  }
}


LIBXSMM_INLINE void naive_copy_RSCK_to_KCRS(const float* rsck, float* kcrs, int R, int S, int C, int K)
{
  LIBXSMM_VLA_DECL(4, const float,  input, rsck, S, C, K);
  LIBXSMM_VLA_DECL(4,       float, output, kcrs, C, R, S);
  int r, s, c, k;

  for ( r = 0; r < R; r++ ) {
    for ( s = 0; s < S; s++ ) {
      for ( c = 0; c < C; c++ ) {
        for ( k = 0; k < K; k++ ) {
          LIBXSMM_VLA_ACCESS(4, output, k, c, r, s, C, R, S) =
            LIBXSMM_VLA_ACCESS(4,  input, r, s, c, k, S, C, K);
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_NC_to_NCNC(float *src, float *dst, int T, int N, int C, int bn, int bc)
{
  int t, n1, n2, c1, c2;
  int nBlocks = N/bn;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(3, float, real_src, src, N, C);
  LIBXSMM_VLA_DECL(5, float, real_dst, dst, nBlocks, cBlocks, bn, bc);

  for (t = 0; t < T; t++) {
    for (n1 = 0; n1 < nBlocks; n1++) {
      for (c1 = 0; c1 < cBlocks; c1++) {
        for (n2 = 0; n2 < bn; n2++) {
          for (c2 = 0; c2 < bc; c2++) {
            LIBXSMM_VLA_ACCESS(5, real_dst, t, n1, c1, n2, c2, nBlocks, cBlocks, bn, bc) =
              LIBXSMM_VLA_ACCESS(3, real_src, t, n1*bn+n2, c1*bc+c2, N, C);
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_NCNC_to_NC(float *src, float *dst, int T, int N, int C, int bn, int bc)
{
  int t, n1, n2, c1, c2;
  int nBlocks = N/bn;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(3, float, real_dst, dst, N, C);
  LIBXSMM_VLA_DECL(5, float, real_src, src, nBlocks, cBlocks, bn, bc);

  for (t = 0; t < T; t++) {
    for (n1 = 0; n1 < nBlocks; n1++) {
      for (c1 = 0; c1 < cBlocks; c1++) {
        for (n2 = 0; n2 < bn; n2++) {
          for (c2 = 0; c2 < bc; c2++) {
            LIBXSMM_VLA_ACCESS(3, real_dst, t, n1*bn+n2, c1*bc+c2, N, C) =
              LIBXSMM_VLA_ACCESS(5, real_src, t, n1, c1, n2, c2, nBlocks, cBlocks, bn, bc);
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_CK_to_KCCK(float *src, float *dst, int C, int K, int bc, int bk)
{
  int k1, k2, c1, c2;
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(2, float, real_src, src, K);
  LIBXSMM_VLA_DECL(4, float, real_dst, dst, cBlocks, bc, bk);

  for (k1 = 0; k1 < kBlocks; k1++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (k2 = 0; k2 < bk; k2++) {
          LIBXSMM_VLA_ACCESS(4, real_dst, k1, c1, c2, k2, cBlocks, bc, bk) =
            LIBXSMM_VLA_ACCESS(2, real_src, c1*bc+c2, k1*bk+k2, K);
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_CK_to_CKKC(float *src, float *dst, int C, int K, int bc, int bk)
{
  int k1, k2, c1, c2;
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(2, float, real_src, src, K);
  LIBXSMM_VLA_DECL(4, float, real_dst, dst, kBlocks, bk, bc);

  for (c1 = 0; c1 < cBlocks; c1++) {
    for (k1 = 0; k1 < kBlocks; k1++) {
      for (k2 = 0; k2 < bk; k2++) {
        for (c2 = 0; c2 < bc; c2++) {
          LIBXSMM_VLA_ACCESS(4, real_dst, c1, k1, k2, c2, kBlocks, bk, bc) =
            LIBXSMM_VLA_ACCESS(2, real_src, c1*bc+c2, k1*bk+k2, K);
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_KC_to_KCCK(float *src, float *dst, int C, int K, int bc, int bk)
{
  int k1, k2, c1, c2;
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(2, float, real_src, src, C);
  LIBXSMM_VLA_DECL(4, float, real_dst, dst, cBlocks, bc, bk);

  for (k1 = 0; k1 < kBlocks; k1++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (k2 = 0; k2 < bk; k2++) {
          LIBXSMM_VLA_ACCESS(4, real_dst, k1, c1, c2, k2, cBlocks, bc, bk) =
            LIBXSMM_VLA_ACCESS(2, real_src, k1*bk+k2, c1*bc+c2, C);
        }
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_KCCK_to_KC(float *src, float *dst, int C, int K, int bc, int bk)
{
  int k1, k2, c1, c2;
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(2, float, real_dst, dst, C);
  LIBXSMM_VLA_DECL(4, float, real_src, src, cBlocks, bc, bk);

  for (k1 = 0; k1 < kBlocks; k1++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (k2 = 0; k2 < bk; k2++) {
          LIBXSMM_VLA_ACCESS(2, real_dst, k1*bk+k2, c1*bc+c2, C) =
            LIBXSMM_VLA_ACCESS(4, real_src, k1, c1, c2, k2, cBlocks, bc, bk);
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_conv_fp(naive_conv_t* param, const float* input, float* output, const float* filter, const float* bias)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
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
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXSMM_VLA_DECL(4,       float, output_t, output + (pad_h_out * ofwp + pad_w_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4, const float,  input_t,  input + (pad_h_in * ifwp + pad_w_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const float, filter_t, filter, nIfm, kh, kw);

#if defined(USE_FUSED_BIAS) || defined(USE_FUSED_BIAS_RELU)
#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (oj = 0; oj < ofh; ++oj) {
        for (oi = 0; oi < ofw; ++oi) {
          LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) = bias[ofm];
        }
      }
    }
  }
#else
  LIBXSMM_UNUSED(bias);
#endif

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ofm); LIBXSMM_OMP_VAR(oj);  LIBXSMM_OMP_VAR(oi);
  LIBXSMM_OMP_VAR(ifm); LIBXSMM_OMP_VAR(ij);  LIBXSMM_OMP_VAR(ii);  LIBXSMM_OMP_VAR(kj);  LIBXSMM_OMP_VAR(ki);
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (ifm = 0; ifm < nIfm; ++ifm) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij+kj < 0 || ij+kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii+ki < 0 || ii+ki >= ifw) continue;
                LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) +=
                  LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp)
                  * LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw);
              }
            }
          }
        }
      }
#if defined(USE_FUSED_RELU) || defined(USE_FUSED_BIAS_RELU)
      for (oj = 0; oj < ofh; ++oj) {
        for (oi = 0; oi < ofw; ++oi) {
          LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) =
            (LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) < 0.0f) ? 0.0f : LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp);
        }
      }
#endif
    }
  }
}

LIBXSMM_INLINE void naive_conv_bp(naive_conv_t* param, float* input, const float* output, const float* filter, const float* naive_input_save)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
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
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXSMM_VLA_DECL(4, const float, output_t, output + (pad_h_out * ofwp + pad_w_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4,       float,  input_t,  input + (pad_h_in * ifwp + pad_w_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const float, filter_t, filter, nIfm, kh, kw);
#if (defined(USE_FUSED_RELU_BWD) || defined(USE_FUSED_BATCH_STATS_BWD))
  LIBXSMM_VLA_DECL(4, const float, naive_input_t, naive_input_save + (pad_h_in * ifwp + pad_w_in), nIfm, ifhp, ifwp);
#else
  LIBXSMM_UNUSED(naive_input_save);
#endif

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ofm); LIBXSMM_OMP_VAR(oj);  LIBXSMM_OMP_VAR(oi);
  LIBXSMM_OMP_VAR(ifm); LIBXSMM_OMP_VAR(ij);  LIBXSMM_OMP_VAR(ii);  LIBXSMM_OMP_VAR(kj);  LIBXSMM_OMP_VAR(ki);
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ifm = 0; ifm < nIfm; ++ifm) {
      for (ofm = 0; ofm < nOfm; ++ofm) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij+kj < 0 || ij+kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii+ki < 0 || ii+ki >= ifw) continue;
                LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp) +=
                  LIBXSMM_VLA_ACCESS(4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp)
                  * LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw);
              }
            }
          }
        }
      }
#if (defined(USE_FUSED_RELU_BWD) || defined(USE_FUSED_BATCH_STATS_BWD))
      for (ij = 0; ij < ifh; ij++) {
        for (ii = 0; ii < ifw; ii++) {
          if ( LIBXSMM_VLA_ACCESS(4,  naive_input_t, img, ifm, ij, ii , nIfm, ifhp, ifwp) == 0.0 ) {
            LIBXSMM_VLA_ACCESS(4, input_t, img, ifm, ij, ii , nIfm, ifhp, ifwp) = 0.0;
          }
        }
      }
#endif
    }
  }
}

LIBXSMM_INLINE void naive_conv_wu(naive_conv_t* param, const float* input, const float* output, float* filter)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
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
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXSMM_VLA_DECL(4, const float, output_t, output + (pad_h_out * ofwp + pad_w_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4, const float,  input_t,  input + (pad_h_in * ifwp + pad_w_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4,       float, filter_t, filter, nIfm, kh, kw);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ofm); LIBXSMM_OMP_VAR(oj);  LIBXSMM_OMP_VAR(oi);
  LIBXSMM_OMP_VAR(ifm); LIBXSMM_OMP_VAR(ij);  LIBXSMM_OMP_VAR(ii);  LIBXSMM_OMP_VAR(kj);  LIBXSMM_OMP_VAR(ki);
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (ofm = 0; ofm < nOfm; ++ofm) {
    for (ifm = 0; ifm < nIfm; ++ifm) {
      for (img = 0; img < nImg; ++img) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij+kj < 0 || ij+kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii+ki < 0 || ii+ki >= ifw) continue;
                LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw) +=
                  LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp)
                  * LIBXSMM_VLA_ACCESS(4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp);
              }
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_conv_fp_int16fp32(naive_conv_t* param, const short* input, float* output, const short* filter)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
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
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXSMM_VLA_DECL(4,       float,     output_t, output + (pad_w_out * ofwp + pad_h_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4, const short,      input_t,  input + (pad_w_in * ifwp + pad_h_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const short,     filter_t, filter, nIfm, kh, kw);


#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ofm); LIBXSMM_OMP_VAR(oj);  LIBXSMM_OMP_VAR(oi);
  LIBXSMM_OMP_VAR(ifm); LIBXSMM_OMP_VAR(ij);  LIBXSMM_OMP_VAR(ii);  LIBXSMM_OMP_VAR(kj);  LIBXSMM_OMP_VAR(ki);
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (ifm = 0; ifm < nIfm; ++ifm) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij+kj < 0 || ij+kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii+ki < 0 || ii+ki >= ifw) continue;
                LIBXSMM_VLA_ACCESS(4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) +=
                  (1.f * LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp))
                * (1.f * LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw));
              }
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_conv_fp_int16int32(naive_conv_t* param, const short* input, int* output, const short* filter)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
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
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXSMM_VLA_DECL(4,         int,     output_t, output + (pad_w_out * ofwp + pad_h_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4, const short,      input_t,  input + (pad_w_in * ifwp + pad_h_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const short,     filter_t, filter, nIfm, kh, kw);


#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ofm); LIBXSMM_OMP_VAR(oj);  LIBXSMM_OMP_VAR(oi);
  LIBXSMM_OMP_VAR(ifm); LIBXSMM_OMP_VAR(ij);  LIBXSMM_OMP_VAR(ii);  LIBXSMM_OMP_VAR(kj);  LIBXSMM_OMP_VAR(ki);
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (ifm = 0; ifm < nIfm; ++ifm) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij+kj < 0 || ij+kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii+ki < 0 || ii+ki >= ifw) continue;
                LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) += (int)
                 ( (int)LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp))
                * ( (int)  LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw));
              }
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_conv_fp_int8int32(naive_conv_t* param, const unsigned char* input, int* output, const char* filter)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
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
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXSMM_VLA_DECL(4,         int,     output_t, output + (pad_w_out * ofwp + pad_h_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4, const unsigned char,      input_t,  input + (pad_w_in * ifwp + pad_h_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const char,     filter_t, filter, nIfm, kh, kw);


#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ofm); LIBXSMM_OMP_VAR(oj);  LIBXSMM_OMP_VAR(oi);
  LIBXSMM_OMP_VAR(ifm); LIBXSMM_OMP_VAR(ij);  LIBXSMM_OMP_VAR(ii);  LIBXSMM_OMP_VAR(kj);  LIBXSMM_OMP_VAR(ki);
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (ifm = 0; ifm < nIfm; ++ifm) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij+kj < 0 || ij+kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii+ki < 0 || ii+ki >= ifw) continue;
                LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) += (int)
                 LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp)
                * LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw);
              }
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_fullyconnected_fp(naive_fullyconnected_t* param, const float* input_ptr, float* output_ptr, const float* filter_ptr)
{
  const int nImg = param->N;
  const int nIFm = param->C;
  const int nOFm = param->K;

  int img, ifm, ofm;

  LIBXSMM_VLA_DECL(2, const float, input,  input_ptr,  nIFm);
  LIBXSMM_VLA_DECL(2, const float, filter, filter_ptr, nIFm);
  LIBXSMM_VLA_DECL(2,       float, output, output_ptr, nOFm);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ifm); LIBXSMM_OMP_VAR(ofm);
# pragma omp parallel for private(img, ofm, ifm)
#endif
  for (ofm = 0; ofm < nOFm; ++ofm) {
    for(img = 0; img < nImg; ++img) {
      LIBXSMM_VLA_ACCESS(2, output, img, ofm, nOFm) = (float)0;
      for (ifm = 0; ifm < nIFm; ++ifm) {
        LIBXSMM_VLA_ACCESS(2, output, img, ofm, nOFm) +=
          LIBXSMM_VLA_ACCESS(2, filter, ofm, ifm, nIFm) * LIBXSMM_VLA_ACCESS(2, input, img, ifm, nIFm);
      }
    }
  }
}

LIBXSMM_INLINE void naive_fullyconnected_bp(naive_fullyconnected_t* param, float* delinput_ptr, const float* deloutput_ptr, const float* filter_ptr)
{
  const int nImg = param->N;
  const int nIFm = param->C;
  const int nOFm = param->K;

  int img, ifm, ofm;

  LIBXSMM_VLA_DECL(2,       float,  dinput,  delinput_ptr, nIFm);
  LIBXSMM_VLA_DECL(2, const float,  filter,    filter_ptr, nIFm);
  LIBXSMM_VLA_DECL(2, const float, doutput, deloutput_ptr, nOFm);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ofm); LIBXSMM_OMP_VAR(ifm);
# pragma omp parallel for private(img, ofm, ifm)
#endif
  for (ifm = 0; ifm < nIFm; ++ifm) {
    for(img = 0; img < nImg; ++img) {
      LIBXSMM_VLA_ACCESS(2, dinput, img, ifm, nIFm) = (float)0;
      for (ofm = 0; ofm < nOFm; ++ofm) {
        LIBXSMM_VLA_ACCESS(2, dinput, img, ifm, nIFm) +=
          LIBXSMM_VLA_ACCESS(2, filter, ofm, ifm, nIFm) * LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm);
      }
    }
  }
}

LIBXSMM_INLINE void naive_fullyconnected_wu(naive_fullyconnected_t* param, const float* input_ptr, const float* deloutput_ptr, float* delfilter_ptr)
{
  const int nImg = param->N;
  const int nIFm = param->C;
  const int nOFm = param->K;

  int img, ifm, ofm;

  LIBXSMM_VLA_DECL(2, const float,   input,     input_ptr, nIFm);
  LIBXSMM_VLA_DECL(2,       float, dfilter, delfilter_ptr, nIFm);
  LIBXSMM_VLA_DECL(2, const float, doutput, deloutput_ptr, nOFm);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(ofm); LIBXSMM_OMP_VAR(ifm);
# pragma omp parallel for private(img, ofm, ifm)
#endif
  for (ofm = 0; ofm < nOFm; ++ofm) {
    for (ifm = 0; ifm < nIFm; ++ifm) {
      LIBXSMM_VLA_ACCESS(2, dfilter, ofm, ifm, nIFm) = (float)0;
      for(img = 0; img < nImg; ++img) {
        LIBXSMM_VLA_ACCESS(2, dfilter, ofm, ifm, nIFm) +=
          LIBXSMM_VLA_ACCESS(2, doutput, img, ofm, nOFm) * LIBXSMM_VLA_ACCESS(2, input, img, ifm, nIFm);
      }
    }
  }
}

LIBXSMM_INLINE void naive_pooling_fp(naive_pooling_t* param, const float* input_ptr, float* output_ptr, int* mask_ptr)
{
  const int nImg = param->N;
  const int nFm = param->C;
  const int ifh = param->H;
  const int ifw = param->W;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  const int r = param->R;
  const int s = param->S;
  const int pad_h = param->pad_h;
  const int pad_w = param->pad_w;
  const int ofh = (ifh + 2*pad_h - r)/sh + 1;
  const int ofw = (ifw + 2*pad_w - s)/sw + 1;


  int img, fm;

  LIBXSMM_VLA_DECL(4, const float, input,   input_ptr, nFm, ifh, ifw);
  LIBXSMM_VLA_DECL(4,       int,   mask,     mask_ptr, nFm, ofh, ofw);
  LIBXSMM_VLA_DECL(4,       float, output, output_ptr, nFm, ofh, ofw);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(fm);
# pragma omp parallel for private(img, fm)
#endif
  for (img = 0; img < nImg; img++) {
    for (fm = 0; fm < nFm; fm++) {
      float* lcl_buffer_ptr = (float*)malloc(sizeof(float)*ofh*ofw);
      LIBXSMM_VLA_DECL(2, float, lcl_buffer, lcl_buffer_ptr, ofw);
      int i, ho, wo, hi, wi, kh, kw;

      if (param->type == 0 ) {
        for ( i = 0; i < ofh*ofw; i++ ) {
          lcl_buffer_ptr[i] = -FLT_MAX;
        }
      } else if (param->type == 1) {
        for ( i = 0; i < ofh*ofw; i++ ) {
          lcl_buffer_ptr[i] = 0.0;
        }
      } else {
        /* shouldn't happen */
      }

      for( ho = 0; ho < ofh; ho++ ) {
        hi = (ho * sh) - pad_h;
        for( wo = 0; wo < ofw; wo++ ) {
          wi = (wo * sw) - pad_w;
          for( kh = 0; kh < r; kh++ ) {
            if(hi+kh < 0 || hi+kh >= ifh) continue;
            for( kw = 0; kw < s; kw++ ) {
              if(wi+kw < 0 || wi+kw >= ifw) continue;
              if ( param->type == 0 ) {
                const int index = (hi+kh)*ifw + wi+kw;
                if ( LIBXSMM_VLA_ACCESS(4, input, img, fm, hi+kh, wi+kw, nFm, ifh, ifw) > LIBXSMM_VLA_ACCESS(2, lcl_buffer, ho, wo, ofw) ) {
                  LIBXSMM_VLA_ACCESS(2, lcl_buffer, ho, wo, ofw) = LIBXSMM_VLA_ACCESS(4, input, img, fm, hi+kh, wi+kw, nFm, ifh, ifw);
                  LIBXSMM_VLA_ACCESS(4, mask, img, fm, ho, wo, nFm, ofh, ofw) = index;
                }
              } else if ( param->type == 1 ) {
                LIBXSMM_VLA_ACCESS(2, lcl_buffer, ho, wo, ofw) += LIBXSMM_VLA_ACCESS(4, input, img, fm, hi+kh, wi+kw, nFm, ifh, ifw);
              } else {
                /* shouldn't happen */
              }
            }
          }
        }
      }

      if (param->type == 0 ) {
        for( ho = 0; ho < ofh; ho++ ) {
          for( wo = 0; wo < ofw; wo++ ) {
            LIBXSMM_VLA_ACCESS(4, output, img, fm, ho, wo, nFm, ofh, ofw) = LIBXSMM_VLA_ACCESS(2, lcl_buffer, ho, wo, ofw);
          }
        }
      } else if (param->type == 1) {
        for( ho = 0; ho < ofh; ho++ ) {
          for( wo = 0; wo < ofw; wo++ ) {
            LIBXSMM_VLA_ACCESS(4, output, img, fm, ho, wo, nFm, ofh, ofw) = LIBXSMM_VLA_ACCESS(2, lcl_buffer, ho, wo, ofw) * (1.0f/(((float)r) * ((float)s)));
          }
        }
      } else {
        /* shouldn't happen */
      }

      free( lcl_buffer_ptr );
    }
  }
}

LIBXSMM_INLINE void naive_pooling_bp(naive_pooling_t* param, float* dinput_ptr, const float* doutput_ptr, const int* mask_ptr)
{
  const int nImg = param->N;
  const int nFm = param->C;
  const int ifh = param->H;
  const int ifw = param->W;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  const int r = param->R;
  const int s = param->S;
  const int pad_h = param->pad_h;
  const int pad_w = param->pad_w;
  const int ofh = (ifh + 2*pad_h - r)/sh + 1;
  const int ofw = (ifw + 2*pad_w - s)/sw + 1;

  int img, fm;

  LIBXSMM_VLA_DECL(4,       float, dinput,   dinput_ptr, nFm, ifh, ifw);
  LIBXSMM_VLA_DECL(4, const int  ,  mask,      mask_ptr, nFm, ofh, ofw);
  LIBXSMM_VLA_DECL(4, const float, doutput, doutput_ptr, nFm, ofh, ofw);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(img); LIBXSMM_OMP_VAR(fm);
# pragma omp parallel for private(img, fm)
#endif
  for (img = 0; img < nImg; img++) {
    for (fm = 0; fm < nFm; fm++) {
      float* lcl_buffer_ptr = (float*)malloc(sizeof(float)*ifh*ifw);
      LIBXSMM_VLA_DECL(2, float, lcl_buffer, lcl_buffer_ptr, ifw);
      int i, ho, wo, hi, wi, kh, kw;

      for ( i = 0; i < ifh*ifw; i++ ) {
        lcl_buffer_ptr[i] = 0.0;
      }

      if (param->type == 0 ) {
        for( ho = 0; ho < ofh; ho++ ) {
          for( wo = 0; wo < ofw; wo++ ) {
            lcl_buffer_ptr[LIBXSMM_VLA_ACCESS(4, mask, img, fm, ho, wo, nFm, ofh, ofw)] += LIBXSMM_VLA_ACCESS(4, doutput, img, fm, ho, wo, nFm, ofh, ofw);
          }
        }
      } else if ( param->type == 1 ) {
        for( ho = 0; ho < ofh; ho++ ) {
          hi = (ho * sh) - pad_h;
          for( wo = 0; wo < ofw; wo++ ) {
            wi = (wo * sw) - pad_w;
            for( kh = 0; kh < r; kh++ ) {
              if(hi+kh < 0 || hi+kh >= ifh) continue;
              for( kw = 0; kw < s; kw++ ) {
                if(wi+kw < 0 || wi+kw >= ifw) continue;
                LIBXSMM_VLA_ACCESS(2, lcl_buffer, hi+kh, wi+kw, ifw) += ( LIBXSMM_VLA_ACCESS(4, doutput, img, fm, ho, wo, nFm, ofh, ofw) * (1.0f/(((float)r) * ((float)s))) );
              }
            }
          }
        }
      } else {
        /* shouldn't happen */
      }

      for( hi = 0; hi < ifh; hi++ ) {
        for( wi = 0; wi < ifw; wi++ ) {
          LIBXSMM_VLA_ACCESS(4, dinput, img, fm, hi, wi, nFm, ifh, ifw) = LIBXSMM_VLA_ACCESS(2, lcl_buffer, hi, wi, ifw);
        }
      }

      free( lcl_buffer_ptr );
    }
  }
}

LIBXSMM_INLINE void naive_fusedbatchnorm_fp(naive_fusedbatchnorm_t* param, const float* input_ptr, float* output_ptr, const float* input_add_ptr,
                                     const float* beta_ptr, const float* gamma_ptr, float* expectval_ptr, float* rcpstddev_ptr, float* variance_ptr)
{
  const int nImg = param->N;
  const int nFm = param->C;
  const int ifh = param->H;
  const int ifw = param->W;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  const int ofh = ifh/sh;
  const int ofw = ifw/sw;
  const float nhw = (float)(nImg * ifh * ifw);
  const float recp_nhw = 1.0f/nhw;
  const float sqrt_eps = 1e-7f;

  int img, fm, hi, wi, ho, wo;

  LIBXSMM_VLA_DECL(4, const float, input,     input_ptr,     nFm, ifh, ifw);
  LIBXSMM_VLA_DECL(4, const float, input_add, input_add_ptr, nFm, ifh, ifw);
  LIBXSMM_VLA_DECL(4,       float, output,    output_ptr,    nFm, ofh, ofw);

  if ( param->norm_type == 0 ) {
#if defined(_OPENMP)
    LIBXSMM_OMP_VAR(wi); LIBXSMM_OMP_VAR(hi);
#   pragma omp parallel for private(img, fm, hi, wi)
#endif
    for (fm = 0; fm < nFm; fm++) {
      float ch_sum = 0.0f;
      float ch_sumsq = 0.0f;
      float tbmean = 0.0f;
      float tbmeansq = 0.0f;
      float tsqbmean = 0.0f;
      float tbrstd = 0.0f;
      float tvariance = 0.0f;

      for ( img = 0; img < nImg; img++ ) {
        for ( hi = 0; hi < ifh; hi++ ) {
          for ( wi = 0; wi < ifw; wi++ ) {
            const float input_val = LIBXSMM_VLA_ACCESS(4, input, img, fm, hi, wi, nFm, ifh, ifw);
            ch_sum   += input_val;
            ch_sumsq += (input_val * input_val);
          }
        }
      }

      tbmean = (recp_nhw * ch_sum) ;
      tbmeansq  = tbmean * tbmean;
      tsqbmean = recp_nhw * ch_sumsq;
      tvariance = tsqbmean - tbmeansq;
      tbrstd = (float)(1.0/sqrt(tvariance + sqrt_eps));
      expectval_ptr[fm] = tbmean;
      rcpstddev_ptr[fm] = tbrstd;
      variance_ptr[fm] = tvariance;
    }
  }

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(ho); LIBXSMM_OMP_VAR(wo);
# pragma omp parallel for private(img, fm, hi, wi, ho, wo)
#endif
  for ( img = 0; img < nImg; img++ ) {
    for ( fm = 0; fm < nFm; fm++ ) {
      for ( hi = 0, ho = 0; hi < ifh; hi += sh, ho++ ) {
        for ( wi = 0, wo = 0; wi < ifw; wi += sw, wo++ ) {
          const float  input_val     =  LIBXSMM_VLA_ACCESS(4, input,     img, fm, hi, wi, nFm, ifh, ifw);
          const float  input_add_val =  LIBXSMM_VLA_ACCESS(4, input_add, img, fm, hi, wi, nFm, ifh, ifw);
                float* output_ptr2   = &LIBXSMM_VLA_ACCESS(4, output,    img, fm, ho, wo, nFm, ofh, ofw);

          /* BN + scale (gamma, beta) */
          float o = gamma_ptr[fm]*(input_val - expectval_ptr[fm])*rcpstddev_ptr[fm] + beta_ptr[fm];
          /* Eltwise */
          if ( (param->fuse_type == 2) || (param->fuse_type == 3) ) {
            o += input_add_val;
          }
          /* ReLU */
          if ( (param->fuse_type == 1) || (param->fuse_type == 3) ) {
            o = ( o < 0.0f ) ? 0.0f : o;
          }
          *output_ptr2 = o;
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_fusedbatchnorm_bp(naive_fusedbatchnorm_t* param, const float* input_ptr, float* dinput_ptr, const float* output_ptr, float* doutput_ptr, float* dinput_add_ptr,
                                     const float* beta_ptr, float* del_beta_ptr, const float* gamma_ptr, float* del_gamma_ptr,
                                     const float* expectval_ptr, const float* rcpstddev_ptr)
{
  const int nImg = param->N;
  const int nFm = param->C;
  const int ifh = param->H;
  const int ifw = param->W;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  const int ofh = ifh/sh;
  const int ofw = ifw/sw;
  const float nhw = (float)(nImg * ifh * ifw);
  const float recp_nhw = 1.0f/nhw;

  int img, fm, hi, wi, ho, wo;

  LIBXSMM_VLA_DECL(4, const float, input,      input_ptr,      nFm, ifh, ifw);
  LIBXSMM_VLA_DECL(4,       float, dinput,     dinput_ptr,     nFm, ifh, ifw);
  LIBXSMM_VLA_DECL(4,       float, dinput_add, dinput_add_ptr, nFm, ifh, ifw);
  LIBXSMM_VLA_DECL(4, const float, output,     output_ptr,     nFm, ofh, ofw);
  LIBXSMM_VLA_DECL(4,       float, doutput,    doutput_ptr,    nFm, ofh, ofw);
  LIBXSMM_UNUSED(beta_ptr);

  if ( param->norm_type == 0 ) {
#if defined(_OPENMP)
    LIBXSMM_OMP_VAR(hi); LIBXSMM_OMP_VAR(wi); LIBXSMM_OMP_VAR(ho); LIBXSMM_OMP_VAR(wo);
#   pragma omp parallel for private(img, fm, hi, wi, ho, wo)
#endif
    for ( fm = 0; fm < nFm; fm++ ) {
      del_gamma_ptr[fm] = 0.0f;
      del_beta_ptr[fm] = 0.0f;

      for ( img = 0; img < nImg; img++ ) {
        for ( hi = 0, ho = 0; hi < ifh; hi += sh, ho++ ) {
          for ( wi = 0, wo = 0; wi < ifw; wi += sw, wo++ ) {
                  float* del_input_add_ptr = &LIBXSMM_VLA_ACCESS(4, dinput_add, img, fm, hi, wi, fm, ifh, ifw);
            const float  output_val        =  LIBXSMM_VLA_ACCESS(4,     output, img, fm, ho, wo, fm, ofh, ofw);
            const float  input_val         =  LIBXSMM_VLA_ACCESS(4,      input, img, fm, hi, wi, fm, ifh, ifw);
                  float* del_output_ptr    = &LIBXSMM_VLA_ACCESS(4,    doutput, img, fm, ho, wo, fm, ofh, ofw);

            /* ReLU */
            if ( (param->fuse_type == 1) || (param->fuse_type == 3) ) {
              *del_output_ptr    = (output_val == 0) ? 0 : *del_output_ptr;
            }
            /* elementwise */
            if ( (param->fuse_type == 2) || (param->fuse_type == 3) ) {
              *del_input_add_ptr = *del_output_ptr;
            }
            del_gamma_ptr[fm] += (input_val - expectval_ptr[fm]) * (*del_output_ptr) * rcpstddev_ptr[fm];
            del_beta_ptr[fm]  += *del_output_ptr;
          }
        }
      }
    }
  }

#if defined(_OPENMP)
# pragma omp parallel for private(img, fm, hi, wi, ho, wo)
#endif
  for ( img = 0; img < nImg; img++ ) {
    for ( fm = 0; fm < nFm; fm++ ) {
      for ( hi = 0, ho = 0; hi < ifh; hi += sh, ho++ ) {
        for ( wi = 0, wo = 0; wi < ifw; wi += sw, wo++) {
                float* del_input_ptr  = &LIBXSMM_VLA_ACCESS(4,     dinput, img, fm, hi, wi, fm, ifh, ifw);
          const float  input_val      =  LIBXSMM_VLA_ACCESS(4,      input, img, fm, hi, wi, fm, ifh, ifw);
          const float  del_output_val =  LIBXSMM_VLA_ACCESS(4,    doutput, img, fm, ho, wo, fm, ofh, ofw);

          *del_input_ptr = gamma_ptr[fm] * rcpstddev_ptr[fm] * recp_nhw * (nhw * del_output_val -
                    (del_beta_ptr[fm] + (input_val - expectval_ptr[fm]) * del_gamma_ptr[fm] * rcpstddev_ptr[fm]));
        }
      }
    }
  }
}

