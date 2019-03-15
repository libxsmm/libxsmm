/******************************************************************************
** Copyright (c) 2018-2019, Intel Corporation                                **
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
#include <libxsmm.h>
#include <libxsmm_intrinsics_x86.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#define CHKERR_LIBXSMM_DNN(A) if ( A != LIBXSMM_DNN_SUCCESS ) fprintf(stderr, "%s\n", libxsmm_dnn_get_error(A) );

LIBXSMM_INLINE void zero_buf(float* buf, size_t size) {
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = 0.0f;
  }
}


LIBXSMM_INLINE void matrix_add(int size, float *a, float *b, float *c)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}


LIBXSMM_INLINE void matrix_eltwise_fma(int size, float *a, float *b, float *c)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    c[i] += a[i] * b[i];
  }
}


LIBXSMM_INLINE void matrix_eltwise_mult(int size, float *a, float *b, float *c)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    c[i] = a[i] * b[i];
  }
}


LIBXSMM_INLINE void matrix_sigmoid(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    const float exp_value = (float)exp((double) -src[i]);
    dst[i] = 1 / (1 + exp_value);
  }
}


LIBXSMM_INLINE void matrix_tanh(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (float)tanh((double)src[i]);
  }
}


LIBXSMM_INLINE void matrix_relu(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (src[i] >= 0) ? src[i] : 0;
  }
}


LIBXSMM_INLINE void matrix_sigmoid_inverse(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    const float exp_value = (float)exp((double) -src[i]);
    const float sig_exp = 1 / (1 + exp_value);
    dst[i] = (1 - sig_exp)*sig_exp;
  }
}


LIBXSMM_INLINE void matrix_tanh_inverse(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    const float tanh_value = (float)tanh((double)src[i]);
    dst[i] = 1 - (tanh_value * tanh_value);
  }
}


LIBXSMM_INLINE void matrix_relu_inverse(int size, float *src, float *dst, float *input)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (input[i] >= 0) ? src[i] : 0;
  }
}


LIBXSMM_INLINE void matrix_transpose(int rows, int cols, float *src, float *dst)
{
  libxsmm_otrans_omp(dst, src, sizeof(float), cols, rows, cols/*ldi*/, rows/*ldo*/);
}


LIBXSMM_INLINE void matrix_copy(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = src[i];
  }
}


LIBXSMM_INLINE void matrix_copy_bias(int m, int n, int ld, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < m*n; i++) {
    int row = i / m;
    int col = i % m;
    dst[row*ld + col] = src[col];
  }
}


LIBXSMM_INLINE void matrix_complement(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = 1 - src[i];
  }
}


LIBXSMM_INLINE void matrix_complement_square(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = 1 - (src[i] * src[i]);
  }
}


LIBXSMM_INLINE void matrix_inverse(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = -src[i];
  }
}


LIBXSMM_INLINE void convert_ck_c3k(int C, int K, float *src, float *dst)
{
  int x, y;
#if defined(_OPENMP)
# pragma omp parallel for private(x, y)
#endif
  for (y = 0; y < C; y++) {
    for (x = 0; x < K; x++) {
      dst[y*3*K + x] = src[y*K + x];
    }
  }
}


void gru_ref_fwd( int N, int C, int K, int t,
                  float *wi, float *wc, float *wf,
                  float *ri, float *rc, float *rf,
                  float *bi, float *bc, float *bf,
                  float *xt, float *hp, float *ht,
                  float *it, float *ct, float *ft, float *ot )
{
  const char transa = 'N', transb = 'N';   /* no transposes */
  const float alpha = 1, beta = 1;
  int j;
  LIBXSMM_VLA_DECL(2, float, x, xt, N * C);
  LIBXSMM_VLA_DECL(2, float, h, ht, K * N);
  LIBXSMM_VLA_DECL(2, float, i, it, K * N);
  LIBXSMM_VLA_DECL(2, float, c, ct, K * N);
  LIBXSMM_VLA_DECL(2, float, f, ft, K * N);
  LIBXSMM_VLA_DECL(2, float, o, ot, K * N);
  for (j = 0; j < t; ++j) {
    /* i_t = b_i */
    matrix_copy_bias(K, N, K, bi, &LIBXSMM_VLA_ACCESS(2, i, j, 0, K * N));
    /* i_t += W_i * x_t */
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &C, &alpha, wi, &K, &LIBXSMM_VLA_ACCESS(2, x, j, 0, N * C), &C, &beta, &LIBXSMM_VLA_ACCESS(2, i, j, 0, K * N), &K);
    /* i_t += R_i * h_{t-1} */
    if (0 == j) {
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, ri, &K, hp,                                       &K, &beta, &LIBXSMM_VLA_ACCESS(2, i, j, 0, K * N), &K);
    } else {
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, ri, &K, &LIBXSMM_VLA_ACCESS(2, h, j-1, 0, K * N), &K, &beta, &LIBXSMM_VLA_ACCESS(2, i, j, 0, K * N), &K);
    }
    /* i_t = sigmoid(i_t) */
    matrix_sigmoid(N*K, &LIBXSMM_VLA_ACCESS(2, i, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, i, j, 0, K * N));
    /* c_t = b_c */
    matrix_copy_bias(K, N, K, bc, &LIBXSMM_VLA_ACCESS(2, c, j, 0, K * N));
    /* c_t += W_c * x_t */
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &C, &alpha, wc, &K, &LIBXSMM_VLA_ACCESS(2, x, j, 0, N * C), &C, &beta, &LIBXSMM_VLA_ACCESS(2, c, j, 0, K * N), &K);
    /* c_t += R_c * h_{t-1} */
    if (0 == j) {
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rc, &K, hp,                                       &K, &beta, &LIBXSMM_VLA_ACCESS(2, c, j, 0, K * N), &K);
    } else {
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rc, &K, &LIBXSMM_VLA_ACCESS(2, h, j-1, 0, K * N), &K, &beta, &LIBXSMM_VLA_ACCESS(2, c, j, 0, K * N), &K);
    }
    /* c_t = sigmoid(c_t) */
    matrix_sigmoid(N*K, &LIBXSMM_VLA_ACCESS(2, c, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, c, j, 0, K * N));
    /* o_t = h_{t-1} . i_t */
    if (0 == j) {
      matrix_eltwise_mult(N*K, hp,                                       &LIBXSMM_VLA_ACCESS(2, i, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, o, j, 0, K * N));
    } else {
      matrix_eltwise_mult(N*K, &LIBXSMM_VLA_ACCESS(2, h, j-1, 0, K * N), &LIBXSMM_VLA_ACCESS(2, i, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, o, j, 0, K * N));
    }
    /* f_t = b_f */
    matrix_copy_bias(K, N, K, bf, &LIBXSMM_VLA_ACCESS(2, f, j, 0, K * N));
    /* f_t += W_f * x_t */
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &C, &alpha, wf, &K, &LIBXSMM_VLA_ACCESS(2, x, j, 0, N * C), &C, &beta, &LIBXSMM_VLA_ACCESS(2, f, j, 0, K * N), &K);
    /* f_t += R_f * o_t */
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rf, &K, &LIBXSMM_VLA_ACCESS(2, o, j, 0, K * N), &K, &beta, &LIBXSMM_VLA_ACCESS(2, f, j, 0, K * N), &K);
    /* f_t = tanh(f_t) */
    matrix_tanh(N*K, &LIBXSMM_VLA_ACCESS(2, f, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, f, j, 0, K * N));
    /* h_t = (1 - c_t) . f_t */
    matrix_complement  (N*K, &LIBXSMM_VLA_ACCESS(2, c, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, h, j, 0, K * N));
    matrix_eltwise_mult(N*K, &LIBXSMM_VLA_ACCESS(2, h, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, f, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, h, j, 0, K * N));
    /* h_t += c_t . h_{t-1} */
    if (0 == j) {
      matrix_eltwise_fma(N*K, &LIBXSMM_VLA_ACCESS(2, c, j, 0, K * N), hp,                                       &LIBXSMM_VLA_ACCESS(2, h, j, 0, K * N));
    } else {
      matrix_eltwise_fma(N*K, &LIBXSMM_VLA_ACCESS(2, c, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, h, j-1, 0, K * N), &LIBXSMM_VLA_ACCESS(2, h, j, 0, K * N));
    }
  }
}


void gru_ref_bwd_upd( int N, int C, int K, int t,
                      float *xt,  float *hpD,  float *ht,
                      float *it,  float *ct,   float *ft, float *ot,
                      float *wi,  float *wc,   float *wf,
                      float *ri,  float *rc,   float *rf,
                      float *dht, float *dw,   float *dr, float *db,
                      float *dxt, float *dhpD, float *scratch )
{
  const char transa = 'N', transb = 'N';   /* no transposes */
  const char transaT = 'T', transbT = 'T'; /* transposes */
  const float alpha = 1, beta = 1, beta0 = 0;
  int j, l, p;
  float *dwi = dw;
  float *dwc = &(dw[C*K]);
  float *dwf = &(dw[2*C*K]);
  float *dri = dr;
  float *drc = &(dr[K*K]);
  float *drf = &(dr[2*K*K]);
  float *dbi = db;
  float *dbc = &(db[K]);
  float *dbf = &(db[2*K]);
  float *deltaD = scratch;
  float *doutD  = &(scratch[N*K]);
  float *diD    = &(scratch[2*N*K]);
  float *dcD    = &(scratch[3*N*K]);
  float *dfD    = &(scratch[4*N*K]);
  float *doD    = &(scratch[5*N*K]);
  LIBXSMM_VLA_DECL(3, float, x,     xt,     N, C);
  LIBXSMM_VLA_DECL(2, float, hp,    hpD,    K);
  LIBXSMM_VLA_DECL(3, float, h,     ht,     N, K);
  LIBXSMM_VLA_DECL(3, float, i,     it,     N, K);
  LIBXSMM_VLA_DECL(3, float, c,     ct,     N, K);
  LIBXSMM_VLA_DECL(3, float, f,     ft,     N, K);
  LIBXSMM_VLA_DECL(3, float, o,     ot,     N, K);
  LIBXSMM_VLA_DECL(3, float, dx,    dxt,    N, C);
  LIBXSMM_VLA_DECL(2, float, dhp,   dhpD,   K);
  LIBXSMM_VLA_DECL(3, float, dh,    dht,    N, K);
  LIBXSMM_VLA_DECL(2, float, di,    diD,    K);
  LIBXSMM_VLA_DECL(2, float, dc,    dcD,    K);
  LIBXSMM_VLA_DECL(2, float, df,    dfD,    K);
  LIBXSMM_VLA_DECL(2, float, dp,    doD,    K);
  LIBXSMM_VLA_DECL(2, float, dout,  doutD,  K);
  LIBXSMM_VLA_DECL(2, float, delta, deltaD, K);
  for (j = t-1; j >= 0; j--) {
#if defined(_OPENMP)
#   pragma omp parallel for private(l, p) LIBXSMM_OPENMP_COLLAPSE(2)
#endif
    for (l = 0; l < N; l++) {
      for (p = 0; p < K; p++) {
        if (t-1 == j) {
          LIBXSMM_VLA_ACCESS(2, delta, l, p, K) = LIBXSMM_VLA_ACCESS(3, dh, t-1, l, p, N, K);
        } else {
          LIBXSMM_VLA_ACCESS(2, delta, l, p, K) = LIBXSMM_VLA_ACCESS(3, dh, j,   l, p, N, K) + LIBXSMM_VLA_ACCESS(2, dout, l, p, K);
        }
        /* df = delta . (1 - c_t) . (1 - (f_t . f_t)) */
        LIBXSMM_VLA_ACCESS(2, df, l, p, K) = LIBXSMM_VLA_ACCESS(2, delta, l, p, K) * (1.0f - LIBXSMM_VLA_ACCESS(3, c, j, l, p, N, K)) * (1.0f - (LIBXSMM_VLA_ACCESS(3, f, j, l, p, N, K) * LIBXSMM_VLA_ACCESS(3, f, j, l, p, N, K)));
        /* dc = delta . (h_{t-1} - f_t) . c_t . (1 - c_t) */
        if (0 == j) {
          LIBXSMM_VLA_ACCESS(2, dc, l, p, K) = LIBXSMM_VLA_ACCESS(2, delta, l, p, K) * (LIBXSMM_VLA_ACCESS(2, hp, l, p, K) -        LIBXSMM_VLA_ACCESS(3, f, j, l, p, N, K)) * LIBXSMM_VLA_ACCESS(3, c, j, l, p, N, K) * (1.0f - LIBXSMM_VLA_ACCESS(3, c, j, l, p, N, K));
        } else {
          LIBXSMM_VLA_ACCESS(2, dc, l, p, K) = LIBXSMM_VLA_ACCESS(2, delta, l, p, K) * (LIBXSMM_VLA_ACCESS(3, h, j-1, l, p, N, K) - LIBXSMM_VLA_ACCESS(3, f, j, l, p, N, K)) * LIBXSMM_VLA_ACCESS(3, c, j, l, p, N, K) * (1.0f - LIBXSMM_VLA_ACCESS(3, c, j, l, p, N, K));
        }
      }
    }
    /* do = {R_f}^T * df */
    LIBXSMM_XBLAS_SYMBOL(float)(&transaT, &transb, &K, &N, &K, &alpha, rf, &K, dfD, &K, &beta0, doD, &K);
    /* di = do . h_{t-1} . i_t . (1 - i_t) */
    if (0 == j) {
#if defined(_OPENMP)
#     pragma omp parallel for private(l, p) LIBXSMM_OPENMP_COLLAPSE(2)
#endif
      for (l = 0; l < N; l++) {
        for (p = 0; p < K; p++) {
          LIBXSMM_VLA_ACCESS(2, di, l, p, K) = LIBXSMM_VLA_ACCESS(2, dp, l, p, K) * LIBXSMM_VLA_ACCESS(2, hp, l, p, K)        * LIBXSMM_VLA_ACCESS(3, i, 0, l, p, N, K) * (1.0f - LIBXSMM_VLA_ACCESS(3, i, 0, l, p, N, K));
        }
      }
    } else {
#if defined(_OPENMP)
#     pragma omp parallel for private(l, p) LIBXSMM_OPENMP_COLLAPSE(2)
#endif
      for (l = 0; l < N; l++) {
        for (p = 0; p < K; p++) {
          LIBXSMM_VLA_ACCESS(2, di, l, p, K) = LIBXSMM_VLA_ACCESS(2, dp, l, p, K) * LIBXSMM_VLA_ACCESS(3, h, j-1, l, p, N, K) * LIBXSMM_VLA_ACCESS(3, i, j, l, p, N, K) * (1.0f - LIBXSMM_VLA_ACCESS(3, i, j, l, p, N, K));
        }
      }
    }
    /* dx_t  = {W_i}^T * di */
    LIBXSMM_XBLAS_SYMBOL(float)(&transaT, &transb, &C, &N, &K, &alpha, wi, &K, diD, &K, &beta0, &LIBXSMM_VLA_ACCESS(3, dx, j, 0, 0, N, C), &C);
    /* dx_t += {W_c}^T * dc */
    LIBXSMM_XBLAS_SYMBOL(float)(&transaT, &transb, &C, &N, &K, &alpha, wc, &K, dcD, &K, &beta,  &LIBXSMM_VLA_ACCESS(3, dx, j, 0, 0, N, C), &C);
    /* dx_t += {W_f}^T * df */
    LIBXSMM_XBLAS_SYMBOL(float)(&transaT, &transb, &C, &N, &K, &alpha, wf, &K, dfD, &K, &beta,  &LIBXSMM_VLA_ACCESS(3, dx, j, 0, 0, N, C), &C);
    /* dh_{t-1}  = {R_i}^T * di */
    /* dh_{t-1} += {R_c}^T * dc */
    if (0 == j) {
      LIBXSMM_XBLAS_SYMBOL(float)(&transaT, &transb, &K, &N, &K, &alpha, ri, &K, diD, &K, &beta0, dhpD, &K);
      LIBXSMM_XBLAS_SYMBOL(float)(&transaT, &transb, &K, &N, &K, &alpha, rc, &K, dcD, &K, &beta,  dhpD, &K);
    } else {
      LIBXSMM_XBLAS_SYMBOL(float)(&transaT, &transb, &K, &N, &K, &alpha, ri, &K, diD, &K, &beta0, doutD, &K);
      LIBXSMM_XBLAS_SYMBOL(float)(&transaT, &transb, &K, &N, &K, &alpha, rc, &K, dcD, &K, &beta,  doutD, &K);
    }
    /* dh_{t-1} += do * i_t + delta * c_t */
    if (0 == j) {
#if defined(_OPENMP)
#     pragma omp parallel for private(l, p) LIBXSMM_OPENMP_COLLAPSE(2)
#endif
      for (l = 0; l < N; l++) {
        for (p = 0; p < K; p++) {
          LIBXSMM_VLA_ACCESS(2, dhp,  l, p, K) += LIBXSMM_VLA_ACCESS(2, dp, l, p, K) * LIBXSMM_VLA_ACCESS(3, i, j, l, p, N, K) + LIBXSMM_VLA_ACCESS(2, delta, l, p, K) * LIBXSMM_VLA_ACCESS(3, c, j, l, p, N, K);
        }
      }
    } else {
#if defined(_OPENMP)
#     pragma omp parallel for private(l, p) LIBXSMM_OPENMP_COLLAPSE(2)
#endif
      for (l = 0; l < N; l++) {
        for (p = 0; p < K; p++) {
          LIBXSMM_VLA_ACCESS(2, dout, l, p, K) += LIBXSMM_VLA_ACCESS(2, dp, l, p, K) * LIBXSMM_VLA_ACCESS(3, i, j, l, p, N, K) + LIBXSMM_VLA_ACCESS(2, delta, l, p, K) * LIBXSMM_VLA_ACCESS(3, c, j, l, p, N, K);
        }
      }
    }
    /* dw_i += di * {x_t}^T */
    /* dw_c += dc * {x_t}^T */
    /* dw_f += df * {x_t}^T */
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transbT, &K, &C, &N, &alpha, diD, &K, &LIBXSMM_VLA_ACCESS(3, x, j, 0, 0, N, C), &C, &beta, dwi, &K);
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transbT, &K, &C, &N, &alpha, dcD, &K, &LIBXSMM_VLA_ACCESS(3, x, j, 0, 0, N, C), &C, &beta, dwc, &K);
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transbT, &K, &C, &N, &alpha, dfD, &K, &LIBXSMM_VLA_ACCESS(3, x, j, 0, 0, N, C), &C, &beta, dwf, &K);
    /* dr_i += di * {o_t}^T */
    /* dr_c += dc * {o_t}^T */
    /* dr_f += df * {h_{t-1}}^T */
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transbT, &K, &K, &N, &alpha, diD, &K, &LIBXSMM_VLA_ACCESS(3, o, j, 0, 0, N, K), &K, &beta, dri, &K);
    LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transbT, &K, &K, &N, &alpha, dcD, &K, &LIBXSMM_VLA_ACCESS(3, o, j, 0, 0, N, K), &K, &beta, drc, &K);
    if (0 == j) {
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transbT, &K, &K, &N, &alpha, dfD, &K, &LIBXSMM_VLA_ACCESS(2, hp, 0, 0, K),        &K, &beta, drf, &K);
    } else {
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transbT, &K, &K, &N, &alpha, dfD, &K, &LIBXSMM_VLA_ACCESS(3, h, j-1, 0, 0, N, K), &K, &beta, drf, &K);
    }
    /* compute db */
#if defined(_OPENMP)
#   pragma omp parallel for private(l, p)
#endif
    for (l = 0; l < K; l++) {
      for (p = 0; p < N; p++) {
        dbi[l] += LIBXSMM_VLA_ACCESS(2, di, p, l, K);
        dbc[l] += LIBXSMM_VLA_ACCESS(2, dc, p, l, K);
        dbf[l] += LIBXSMM_VLA_ACCESS(2, df, p, l, K);
      }
    }
  }
}


int main(int argc, char* argv[])
{
  float *wigold, *wcgold, *wfgold, *rigold, *rcgold, *rfgold, *bigold, *bcgold, *bfgold;
  float *xgoldt, *hpgold, *hgoldt;
  float *dwgold, *drgold, *dbgold;
  float *dxgoldt, *dhpgold, *dhgoldt;
  float *igoldt, *cgoldt, *fgoldt, *ogoldt;
  float *xt, *hp, *w, *r, *b, *ht;
  float *it, *ct, *ft, *ot;
  float *dxt, *dhp, *dw, *dr, *db, *dht;
  float *scratch_bu, *dwtest, *drtest;

  void *scratch, *internalstate;
  size_t scratch_size = 0, internalstate_size = 0;

  int iters = 10;   /* repetitions of benchmark */
  int pass = 0;     /* pass: 0--FWD, 1--BWD, 2--UPD, 3--BWD+UPD */
  int N = 168;      /* size of mini-batch */
  int C = 512;      /* number of inputs */
  int K = 256;      /* number of outputs */
  int t = 50;       /* number of time steps (>= 1) */
  int bn = 24;
  int bc = 64;
  int bk = 64;

  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 0/*disabled by default*/ : atof(env_check));

#if defined(_OPENMP)
  int nThreads = omp_get_max_threads(); /* number of threads */
#else
  int nThreads = 1; /* number of threads */
#endif

  unsigned long long l_start, l_end;
  double l_total = 0.0;
  double flops = 0.0;
  const double tflops = 12; /* transcendental flops */
  int j, l, p;

  libxsmm_dnn_rnncell_desc grucell_desc;
  libxsmm_dnn_rnncell* libxsmm_handle;
  libxsmm_dnn_tensor* libxsmm_input;
  libxsmm_dnn_tensor* libxsmm_hidden_state_prev;
  libxsmm_dnn_tensor* libxsmm_weight;
  libxsmm_dnn_tensor* libxsmm_recur_weight;
  libxsmm_dnn_tensor* libxsmm_bias;
  libxsmm_dnn_tensor* libxsmm_hidden_state;
  libxsmm_dnn_tensor* libxsmm_i;
  libxsmm_dnn_tensor* libxsmm_c;
  libxsmm_dnn_tensor* libxsmm_f;
  libxsmm_dnn_tensor* libxsmm_o;
  libxsmm_dnn_tensor* libxsmm_dinput;
  libxsmm_dnn_tensor* libxsmm_dhidden_state_prev;
  libxsmm_dnn_tensor* libxsmm_dweight;
  libxsmm_dnn_tensor* libxsmm_drecur_weight;
  libxsmm_dnn_tensor* libxsmm_dbias;
  libxsmm_dnn_tensor* libxsmm_dhidden_state;

  libxsmm_dnn_tensor_datalayout* libxsmm_layout;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status = LIBXSMM_DNN_SUCCESS;

  libxsmm_matdiff_info norms_fwd, norms_bwd, norms_upd_w, norms_upd_r, norms_upd_b, diff;
  libxsmm_matdiff_clear(&norms_fwd);
  libxsmm_matdiff_clear(&norms_bwd);
  libxsmm_matdiff_clear(&norms_upd_w);
  libxsmm_matdiff_clear(&norms_upd_r);
  libxsmm_matdiff_clear(&norms_upd_b);
  libxsmm_matdiff_clear(&diff);

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("\nUsage: ./grudriver [reps] [pass: 0--FWD, 1--BWD, 2--UPD, 3--BWD+UPD] [N] [C] [K] [time_steps > 0]\n\n");
    return 0;
  }
  libxsmm_rng_set_seed(1);

  /* reading new values from cli */
  j = 1;
  if (argc > j) iters = atoi(argv[j++]);
  if (argc > j) pass  = atoi(argv[j++]);
  if (argc > j) N     = atoi(argv[j++]);
  if (argc > j) C     = atoi(argv[j++]);
  if (argc > j) K     = atoi(argv[j++]);
  if (argc > j) t     = atoi(argv[j++]);
  if (argc > j) bn    = atoi(argv[j++]);
  if (argc > j) bc    = atoi(argv[j++]);
  if (argc > j) bk    = atoi(argv[j++]);

  if (t <= 0) {
    printf("time_steps %d should be greater than or equal to 1\n\n", t);
    return 0;
  }
  if (!(pass == 0 || pass == 1 || pass == 2 || pass == 3)) {
    printf("Unknown pass: %d, valid arguments for pass = {0(FWD), 1(BWD), 2(UPD), 3(BWD+UPD)\n\n", pass);
    return 0;
  }

#if defined(__SSE3__)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

  /* print some summary */
  printf("##########################################\n");
  printf("#          Setting Up (Common)           #\n");
  printf("##########################################\n");
  printf("PARAMS: N:%d  C:%d  K:%d  T:%d\n", N, C, K, t);
  printf("PARAMS: ITERS:%d", iters); if (LIBXSMM_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  printf("SIZE Weight (MB): %10.2f MiB\n", (double)(C*K*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Input (MB): %10.2f MiB\n", (double)(N*C*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Hidden State: %10.2f MiB\n", (double)(K*N*sizeof(float))/(1024.0*1024.0) );

  /* allocate data */
  xgoldt     = (float*)libxsmm_aligned_malloc(N*C*t*sizeof(float), 2097152);
  hpgold     = (float*)libxsmm_aligned_malloc(K*N*sizeof(float),   2097152);
  wigold     = (float*)libxsmm_aligned_malloc(C*K*sizeof(float),   2097152);
  wcgold     = (float*)libxsmm_aligned_malloc(C*K*sizeof(float),   2097152);
  wfgold     = (float*)libxsmm_aligned_malloc(C*K*sizeof(float),   2097152);
  rigold     = (float*)libxsmm_aligned_malloc(K*K*sizeof(float),   2097152);
  rcgold     = (float*)libxsmm_aligned_malloc(K*K*sizeof(float),   2097152);
  rfgold     = (float*)libxsmm_aligned_malloc(K*K*sizeof(float),   2097152);
  bigold     = (float*)libxsmm_aligned_malloc(K*sizeof(float),     2097152);
  bcgold     = (float*)libxsmm_aligned_malloc(K*sizeof(float),     2097152);
  bfgold     = (float*)libxsmm_aligned_malloc(K*sizeof(float),     2097152);
  hgoldt     = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  igoldt     = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  cgoldt     = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  fgoldt     = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  ogoldt     = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  dxgoldt    = (float*)libxsmm_aligned_malloc(N*C*t*sizeof(float), 2097152);
  dhpgold    = (float*)libxsmm_aligned_malloc(K*N*sizeof(float),   2097152);
  dwgold     = (float*)libxsmm_aligned_malloc(C*K*3*sizeof(float), 2097152);
  drgold     = (float*)libxsmm_aligned_malloc(K*K*3*sizeof(float), 2097152);
  dbgold     = (float*)libxsmm_aligned_malloc(K*3*sizeof(float),   2097152);
  dhgoldt    = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  scratch_bu = (float*)libxsmm_aligned_malloc(K*N*6*sizeof(float), 2097152);
  xt         = (float*)libxsmm_aligned_malloc(N*C*t*sizeof(float), 2097152);
  hp         = (float*)libxsmm_aligned_malloc(K*N*sizeof(float),   2097152);
  w          = (float*)libxsmm_aligned_malloc(C*K*3*sizeof(float), 2097152);
  r          = (float*)libxsmm_aligned_malloc(K*K*3*sizeof(float), 2097152);
  b          = (float*)libxsmm_aligned_malloc(K*3*sizeof(float),   2097152);
  ht         = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  it         = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  ct         = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  ft         = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  ot         = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  dxt        = (float*)libxsmm_aligned_malloc(N*C*t*sizeof(float), 2097152);
  dhp        = (float*)libxsmm_aligned_malloc(K*N*sizeof(float),   2097152);
  dw         = (float*)libxsmm_aligned_malloc(C*K*3*sizeof(float), 2097152);
  dr         = (float*)libxsmm_aligned_malloc(K*K*3*sizeof(float), 2097152);
  db         = (float*)libxsmm_aligned_malloc(K*3*sizeof(float),   2097152);
  dht        = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  dwtest     = (float*)libxsmm_aligned_malloc(C*K*3*sizeof(float), 2097152);
  drtest     = (float*)libxsmm_aligned_malloc(K*K*3*sizeof(float), 2097152);
  LIBXSMM_VLA_DECL(2, float, xgold, xgoldt, N * C);
  LIBXSMM_VLA_DECL(2, float, hgold, hgoldt, N * K);
  LIBXSMM_VLA_DECL(2, float, igold, igoldt, N * K);
  LIBXSMM_VLA_DECL(2, float, cgold, cgoldt, N * K);
  LIBXSMM_VLA_DECL(2, float, fgold, fgoldt, N * K);
  LIBXSMM_VLA_DECL(2, float, ogold, ogoldt, N * K);
  LIBXSMM_VLA_DECL(2, float, dhgold, dhgoldt, N * K);
  LIBXSMM_VLA_DECL(2, float, dxgold, dxgoldt, N * C);
  LIBXSMM_VLA_DECL(2, float, h, ht, N * K);

  /* initialize data */
  /* FWD */
  for (j = 0; j < t; ++j) {
    LIBXSMM_MATINIT_OMP(float, 24, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, N * C), N, C, N, 1.0);
  }
  LIBXSMM_MATINIT_OMP(float, 24, hpgold, N, K, N, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, wigold, C, K, C, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, wcgold, C, K, C, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, wfgold, C, K, C, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, rigold, K, K, K, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, rcgold, K, K, K, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, rfgold, K, K, K, 1.0);
  LIBXSMM_MATINIT_OMP(float, 24, bigold, 1, K, 1, 1.0);
  LIBXSMM_MATINIT_OMP(float, 24, bcgold, 1, K, 1, 1.0);
  LIBXSMM_MATINIT_OMP(float, 24, bfgold, 1, K, 1, 1.0);
  zero_buf(hgoldt, N*K*t);
  /* BWD/UPD */
  for (j = 0; j < t; ++j) {
    LIBXSMM_MATINIT_OMP(float, 24, &LIBXSMM_VLA_ACCESS(2, dhgold, j, 0, K * N), N, K, N, 1.0);
  }
  zero_buf(dxgoldt, N*C*t);
  zero_buf(dhpgold, K*N);
  zero_buf(dwgold,  C*K*3);
  zero_buf(drgold,  K*K*3);
  zero_buf(dbgold,  K*3);

  /* first touch LIBXSMM */
  zero_buf(xt,  N*C*t);
  zero_buf(hp,  K*N);
  zero_buf(w,   C*K*3);
  zero_buf(r,   K*K*3);
  zero_buf(b,   K*3);
  zero_buf(ht,  N*K*t);
  zero_buf(it,  K*N*t);
  zero_buf(ct,  K*N*t);
  zero_buf(ft,  K*N*t);
  zero_buf(ot,  K*N*t);
  zero_buf(dxt, N*C*t);
  zero_buf(dhp, K*N);
  zero_buf(dw,  C*K*3);
  zero_buf(dr,  K*K*3);
  zero_buf(db,  K*3);
  zero_buf(dht, K*N*t);

  if (LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#         Computing Reference ...        #\n");
    printf("##########################################\n");

    gru_ref_fwd( N, C, K, t,
                 wigold, wcgold, wfgold,
                 rigold, rcgold, rfgold,
                 bigold, bcgold, bfgold,
                 xgoldt, hpgold, hgoldt,
                 igoldt, cgoldt, fgoldt, ogoldt );

    gru_ref_bwd_upd( N, C, K, t,
                     xgoldt, hpgold, hgoldt,
                     igoldt, cgoldt, fgoldt, ogoldt,
                     wigold, wcgold, wfgold,
                     rigold, rcgold, rfgold,
                     dhgoldt, dwgold, drgold, dbgold,
                     dxgoldt, dhpgold, scratch_bu );

    printf("##########################################\n");
    printf("#      Computing Reference ... done      #\n");
    printf("##########################################\n");
  }

  if (1 /* format == 'A' || format == 'L' */) {
    printf("\n");
    printf("##########################################\n");
    printf("#      Setting Up  (custom-Storage)      #\n");
    printf("##########################################\n");

    /* setup LIBXSMM handle */
    grucell_desc.threads = nThreads;
    grucell_desc.N = N;
    grucell_desc.C = C;
    grucell_desc.K = K;
    grucell_desc.max_T = t;
    grucell_desc.bn = bn;
    grucell_desc.bc = bc;
    grucell_desc.bk = bk;
    grucell_desc.cell_type = LIBXSMM_DNN_RNNCELL_GRU;
    grucell_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    grucell_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
    grucell_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_NC;
    grucell_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_CK;

    libxsmm_handle = libxsmm_dnn_create_rnncell( grucell_desc, &status );
    CHKERR_LIBXSMM_DNN( status );

    /* setup LIBXSMM buffers and filter */
    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_input = libxsmm_dnn_link_tensor( libxsmm_layout, xt, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_hidden_state_prev = libxsmm_dnn_link_tensor( libxsmm_layout, hp, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_WEIGHT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_weight = libxsmm_dnn_link_tensor( libxsmm_layout, w, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_recur_weight = libxsmm_dnn_link_tensor( libxsmm_layout, r, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_BIAS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_bias = libxsmm_dnn_link_tensor( libxsmm_layout, b, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_hidden_state = libxsmm_dnn_link_tensor( libxsmm_layout, ht, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_I, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_i = libxsmm_dnn_link_tensor( libxsmm_layout, it, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_CI, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_c = libxsmm_dnn_link_tensor( libxsmm_layout, ct, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_F, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_f = libxsmm_dnn_link_tensor( libxsmm_layout, ft, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_O, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_o = libxsmm_dnn_link_tensor( libxsmm_layout, ot, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dinput = libxsmm_dnn_link_tensor( libxsmm_layout, dxt, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE_PREV, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dhidden_state_prev = libxsmm_dnn_link_tensor( libxsmm_layout, dhp, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_WEIGHT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dweight = libxsmm_dnn_link_tensor( libxsmm_layout, dw, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_drecur_weight = libxsmm_dnn_link_tensor( libxsmm_layout, dr, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_BIAS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dbias = libxsmm_dnn_link_tensor( libxsmm_layout, db, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dhidden_state = libxsmm_dnn_link_tensor( libxsmm_layout, dht, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    /* copy in data to LIBXSMM format */
    matrix_copy(N*C*t, xgoldt, xt);
    matrix_copy(K*N, hpgold, hp);
    convert_ck_c3k(C, K, wigold, w);
    convert_ck_c3k(C, K, wcgold, &(w[K]));
    convert_ck_c3k(C, K, wfgold, &(w[2*K]));
    convert_ck_c3k(K, K, rigold, r);
    convert_ck_c3k(K, K, rcgold, &(r[K]));
    convert_ck_c3k(K, K, rfgold, &(r[2*K]));
    matrix_copy(K, bigold, b);
    matrix_copy(K, bcgold, &(b[K]));
    matrix_copy(K, bfgold, &(b[2*K]));
    matrix_copy(K*N*t, dhgoldt, dht);

    /* bind buffers and filter to handle */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_input, LIBXSMM_DNN_RNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_hidden_state_prev, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_weight, LIBXSMM_DNN_RNN_REGULAR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_recur_weight, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_bias, LIBXSMM_DNN_RNN_REGULAR_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_hidden_state, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_i, LIBXSMM_DNN_RNN_INTERNAL_I ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_c, LIBXSMM_DNN_RNN_INTERNAL_CI ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_f, LIBXSMM_DNN_RNN_INTERNAL_F ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_o, LIBXSMM_DNN_RNN_INTERNAL_O ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dinput, LIBXSMM_DNN_RNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dhidden_state_prev, LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dweight, LIBXSMM_DNN_RNN_GRADIENT_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_drecur_weight, LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dbias, LIBXSMM_DNN_RNN_GRADIENT_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dhidden_state, LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE ) );

    /* let's allocate and bind scratch */
    if (pass == 0) {
      scratch_size = libxsmm_dnn_rnncell_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, &status );
      CHKERR_LIBXSMM_DNN( status );
      scratch = libxsmm_aligned_malloc( scratch_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, scratch ) );
    } else {
      scratch_size = libxsmm_dnn_rnncell_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
      CHKERR_LIBXSMM_DNN( status );
      scratch = libxsmm_aligned_malloc( scratch_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
    }
    zero_buf( (float*)scratch, scratch_size/4 );

    /* let's allocate and bind internalstate */
    if (pass == 0) {
      internalstate_size = libxsmm_dnn_rnncell_get_internalstate_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, &status );
      CHKERR_LIBXSMM_DNN( status );
      internalstate = libxsmm_aligned_malloc( internalstate_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, internalstate ) );
    } else {
      internalstate_size = libxsmm_dnn_rnncell_get_internalstate_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
      CHKERR_LIBXSMM_DNN( status );
      internalstate = libxsmm_aligned_malloc( internalstate_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, internalstate ) );
    }
    zero_buf( (float*)internalstate, internalstate_size/4 );

    if ((pass == 0) && LIBXSMM_NEQ(0, check)) {
      printf("##########################################\n");
      printf("#   Correctness - FWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM GRU */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
      }

      /* compare */
      libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, K*N, 1, &LIBXSMM_VLA_ACCESS(2, hgold, t-1, 0, K * N), &LIBXSMM_VLA_ACCESS(2, h, t-1, 0, K * N), 0, 0);
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_fwd);
    } else {
      /* We need to always run FWD pass once to populate i, c, f, o, h */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
      }
    }

    if ( (pass == 1) && LIBXSMM_NEQ(0, check) ) {
      printf("##########################################\n");
      printf("#   Correctness - BWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM GRU */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
      }

      /* compare */
      libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, N*C*t, 1, dxgoldt, dxt, 0, 0);
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);
    }

    if ( (pass == 2) && LIBXSMM_NEQ(0, check) ) {
      printf("##########################################\n");
      printf("#   Correctness - UPD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM GRU */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid ) );
      }

      convert_ck_c3k(C, K, &(dwgold[0]),     &(dwtest[0]));
      convert_ck_c3k(C, K, &(dwgold[C*K]),   &(dwtest[K]));
      convert_ck_c3k(C, K, &(dwgold[2*C*K]), &(dwtest[2*K]));
      convert_ck_c3k(K, K, &(drgold[0]),     &(drtest[0]));
      convert_ck_c3k(K, K, &(drgold[K*K]),   &(drtest[K]));
      convert_ck_c3k(K, K, &(drgold[2*K*K]), &(drtest[2*K]));
      /* compare */
      libxsmm_matdiff(&norms_upd_w, LIBXSMM_DATATYPE_F32, C*K*3, 1, dwtest, dw, 0, 0);
      printf("Delta weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_w.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_w.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_w.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_w.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_w.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_w.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_w.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_w);

      libxsmm_matdiff(&norms_upd_r, LIBXSMM_DATATYPE_F32, K*K*3, 1, drtest, dr, 0, 0);
      printf("Delta recurrent weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_r.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_r.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_r.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_r.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_r.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_r.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_r.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_r);

      libxsmm_matdiff(&norms_upd_b, LIBXSMM_DATATYPE_F32, K*3, 1, dbgold, db, 0, 0);
      printf("Delta bias\n");
      printf("L1 reference  : %.25g\n", norms_upd_b.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_b.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_b.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_b.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_b.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_b.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_b.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_b);
    }

    if ( (pass == 3) && LIBXSMM_NEQ(0, check) ) {
      printf("##########################################\n");
      printf("# Correctness - BWD+UPD (custom-Storage) #\n");
      printf("##########################################\n");
      /* run LIBXSMM GRU */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWDUPD, 0, tid ) );
      }

      convert_ck_c3k(C, K, &(dwgold[0]),     &(dwtest[0]));
      convert_ck_c3k(C, K, &(dwgold[C*K]),   &(dwtest[K]));
      convert_ck_c3k(C, K, &(dwgold[2*C*K]), &(dwtest[2*K]));
      convert_ck_c3k(K, K, &(drgold[0]),     &(drtest[0]));
      convert_ck_c3k(K, K, &(drgold[K*K]),   &(drtest[K]));
      convert_ck_c3k(K, K, &(drgold[2*K*K]), &(drtest[2*K]));
      /* compare */
      libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, N*C*t, 1, dxgoldt, dxt, 0, 0);
      printf("Delta input\n");
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);

      libxsmm_matdiff(&norms_upd_w, LIBXSMM_DATATYPE_F32, C*K*3, 1, dwtest, dw, 0, 0);
      printf("Delta weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_w.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_w.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_w.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_w.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_w.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_w.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_w.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_w);

      libxsmm_matdiff(&norms_upd_r, LIBXSMM_DATATYPE_F32, K*K*3, 1, drtest, dr, 0, 0);
      printf("Delta recurrent weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_r.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_r.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_r.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_r.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_r.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_r.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_r.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_r);

      libxsmm_matdiff(&norms_upd_b, LIBXSMM_DATATYPE_F32, K*3, 1, dbgold, db, 0, 0);
      printf("Delta bias\n");
      printf("L1 reference  : %.25g\n", norms_upd_b.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_b.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_b.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_b.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_b.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_b.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_b.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_b);
    }

    if ( pass == 0 ) {
      printf("##########################################\n");
      printf("#   Performance - FWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM GRU for performance */
      l_start = libxsmm_timer_tick();

#if defined(_OPENMP)
#     pragma omp parallel private(j)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (j = 0; j < iters; ++j) {
          libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = (((2.0 * K * N * C) + (2.0 * K * N * K) + (2.0 * K * N) + (tflops * K * N)) * 2.0 + (K * N) + (2.0 * K * N * C) + (2.0 * K * N * K) + (tflops * K * N) + 4.0 * (K * N)) * (double)t * (double)iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, N, C, K, t, bn, bc, bk, ((double)(l_total/iters)), (flops*1e-9)/l_total);
    }

    if ( pass == 1 ) {
      printf("##########################################\n");
      printf("#   Performance - BWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM GRU for performance */
      l_start = libxsmm_timer_tick();

#if defined(_OPENMP)
#     pragma omp parallel private(j)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (j = 0; j < iters; ++j) {
          libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = K * N; /* d3 = djdh + d23 (delta) */
      flops += 2.0 * K * N; /* d4 = (1 - z).d3 */
      flops += K * N; /* d5 = d3.h */
      flops += K * N; /* d6 = -d5 */
      flops += K * N; /* d7 = d3.g */
      flops += K * N; /* d8 = d3.z */
      flops += K * N; /* d9 = d7 + d8 */
      flops += 3.0 * K * N; /* d10 = d8.tanh'(g) */
      flops += 3.0 * K * N; /* d11 = d9.sig'(z) */
      flops += (2.0 * K * K * N + K * K) ; /* d13 = Wg^T * d10 (including transpose) */
      flops += (2.0 * K * K * N + K * K) ; /* d15 = Wz^T * d11 (including transpose) */
      flops += K * N; /* d16 = d13.z */
      flops += K * N; /* d17 = d13.r */
      flops += 3.0 * K * N; /* d18 = d16.sig'(r) */
      flops += K * N; /* d19 = d17 + d4 */
      flops += (2.0 * K * K * N + K * K) ; /* d21 = Wr^T * d18 (including transpose) */
      flops += K * N; /* d22 = d21 + d15 */
      flops += K * N; /* d23 = d19 + d22 */
      flops += (2.0 * K * C * N + K * C) ; /* d12 = Ug^T * d10 (including transpose) */
      flops += (2.0 * K * C * N + K * C) ; /* d14 = Uz^T * d11 (including transpose) */
      flops += (2.0 * K * C * N + K * C) ; /* d20 = Ur^T * d18 (including transpose) */
      flops += 2.0 * K * N; /* djdx = d12 + d14 + d20 */
      flops *= t; /* for t time steps */
      flops *= iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("bp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,BP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, N, C, K, t, bn, bc, bk, ((double)(l_total/iters)), (flops*1e-9)/l_total);
    }

    if ( pass == 2 ) {
      printf("##########################################\n");
      printf("#   Performance - UPD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM GRU for performance */
      l_start = libxsmm_timer_tick();

#if defined(_OPENMP)
#     pragma omp parallel private(j)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (j = 0; j < iters; ++j) {
          libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = K * N; /* d3 = djdh + d23 (delta) */
      flops += 2.0 * K * N; /* d4 = (1 - z).d3 */
      flops += K * N; /* d5 = d3.h */
      flops += K * N; /* d6 = -d5 */
      flops += K * N; /* d7 = d3.g */
      flops += K * N; /* d8 = d3.z */
      flops += K * N; /* d9 = d7 + d8 */
      flops += 3.0 * K * N; /* d10 = d8.tanh'(g) */
      flops += 3.0 * K * N; /* d11 = d9.sig'(z) */
      flops += (2.0 * K * K * N + K * K) ; /* d13 = Wg^T * d10 (including transpose) */
      flops += (2.0 * K * K * N + K * K) ; /* d15 = Wz^T * d11 (including transpose) */
      flops += K * N; /* d16 = d13.z */
      flops += K * N; /* d17 = d13.r */
      flops += 3.0 * K * N; /* d18 = d16.sig'(r) */
      flops += K * N; /* d19 = d17 + d4 */
      flops += (2.0 * K * K * N + K * K) ; /* d21 = Wr^T * d18 (including transpose) */
      flops += K * N; /* d22 = d21 + d15 */
      flops += K * N; /* d23 = d19 + d22 */
      flops += (2.0 * K * N * K + K * N + K * K) ; /* djdwr = djdwr + d18 * h^T */
      flops += (2.0 * K * N * K + K * N + K * K) ; /* djdwz = djdwz + d11 * h^T */
      flops += (2.0 * K * N * K + 2.0 * K * N + K * K) ; /* djdwg = djdwg + d10 * (h.r)^T */
      flops += (2.0 * K * N * C + C * N + K * C) ; /* djdur = djdur + d18 * x^T */
      flops += (2.0 * K * N * C + C * N + K * C) ; /* djduz = djduz + d11 * x^T */
      flops += (2.0 * K * N * C + C * N + K * C) ; /* djdug = djdug + d10 * x^T */
      flops += K * N; /* djdbr = djdbr + d18 */
      flops += K * N; /* djdbz = djdbz + d11 */
      flops += K * N; /* djdbg = djdbg + d10 */
      flops *= t; /* for t time steps */
      flops *= iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("wu time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,WU,%s,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, N, C, K, t, bn, bc, bk, ((double)(l_total/iters)), (flops*1e-9)/l_total);
    }

    if ( pass == 3 ) {
      printf("##########################################\n");
      printf("# Performance - BWD+UPD (custom-Storage) #\n");
      printf("##########################################\n");
      /* run LIBXSMM GRU for performance */
      l_start = libxsmm_timer_tick();

#if defined(_OPENMP)
#     pragma omp parallel private(j)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (j = 0; j < iters; ++j) {
          libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWDUPD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = K * N; /* d3 = djdh + d23 (delta) */
      flops += 2.0 * K * N; /* d4 = (1 - z).d3 */
      flops += K * N; /* d5 = d3.h */
      flops += K * N; /* d6 = -d5 */
      flops += K * N; /* d7 = d3.g */
      flops += K * N; /* d8 = d3.z */
      flops += K * N; /* d9 = d7 + d8 */
      flops += 3.0 * K * N; /* d10 = d8.tanh'(g) */
      flops += 3.0 * K * N; /* d11 = d9.sig'(z) */
      flops += (2.0 * K * K * N + K * K) ; /* d13 = Wg^T * d10 (including transpose) */
      flops += (2.0 * K * K * N + K * K) ; /* d15 = Wz^T * d11 (including transpose) */
      flops += K * N; /* d16 = d13.z */
      flops += K * N; /* d17 = d13.r */
      flops += 3.0 * K * N; /* d18 = d16.sig'(r) */
      flops += K * N; /* d19 = d17 + d4 */
      flops += (2.0 * K * K * N + K * K) ; /* d21 = Wr^T * d18 (including transpose) */
      flops += K * N; /* d22 = d21 + d15 */
      flops += K * N; /* d23 = d19 + d22 */
      flops += (2.0 * K * C * N + K * C) ; /* d12 = Ug^T * d10 (including transpose) */
      flops += (2.0 * K * C * N + K * C) ; /* d14 = Uz^T * d11 (including transpose) */
      flops += (2.0 * K * C * N + K * C) ; /* d20 = Ur^T * d18 (including transpose) */
      flops += 2.0 * K * N; /* djdx = d12 + d14 + d20 */
      flops += (2.0 * K * N * K + K * N + K * K) ; /* djdwr = djdwr + d18 * h^T */
      flops += (2.0 * K * N * K + K * N + K * K) ; /* djdwz = djdwz + d11 * h^T */
      flops += (2.0 * K * N * K + 2.0 * K * N + K * K) ; /* djdwg = djdwg + d10 * (h.r)^T */
      flops += (2.0 * K * N * C + C * N + K * C) ; /* djdur = djdur + d18 * x^T */
      flops += (2.0 * K * N * C + C * N + K * C) ; /* djduz = djduz + d11 * x^T */
      flops += (2.0 * K * N * C + C * N + K * C) ; /* djdug = djdug + d10 * x^T */
      flops += K * N; /* djdbr = djdbr + d18 */
      flops += K * N; /* djdbz = djdbz + d11 */
      flops += K * N; /* djdbg = djdbg + d10 */
      flops *= t; /* for t time steps */
      flops *= iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("bp+wu time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,BP+WU,%s,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, N, C, K, t, bn, bc, bk, ((double)(l_total/iters)), (flops*1e-9)/l_total);
    }

    /* clean-up */
    if (pass == 0) {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD ) );
    } else {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL ) );
    }
    libxsmm_free(scratch);
    libxsmm_free(internalstate);
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_I ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_CI ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_F ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_O ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_input ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_hidden_state_prev ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_weight ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_recur_weight ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_bias ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_hidden_state ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_i ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_c ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_f ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_o ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dinput ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dhidden_state_prev ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dweight ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_drecur_weight ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dbias ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dhidden_state ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_rnncell( libxsmm_handle ) );
  }

  /* deallocate data */
  libxsmm_free(xgoldt);
  libxsmm_free(hpgold);
  libxsmm_free(wigold);
  libxsmm_free(wcgold);
  libxsmm_free(wfgold);
  libxsmm_free(rigold);
  libxsmm_free(rcgold);
  libxsmm_free(rfgold);
  libxsmm_free(bigold);
  libxsmm_free(bcgold);
  libxsmm_free(bfgold);
  libxsmm_free(hgoldt);
  libxsmm_free(igoldt);
  libxsmm_free(cgoldt);
  libxsmm_free(fgoldt);
  libxsmm_free(ogoldt);
  libxsmm_free(dxgoldt);
  libxsmm_free(dhpgold);
  libxsmm_free(dwgold);
  libxsmm_free(drgold);
  libxsmm_free(dbgold);
  libxsmm_free(dhgoldt);
  libxsmm_free(xt);
  libxsmm_free(hp);
  libxsmm_free(w);
  libxsmm_free(r);
  libxsmm_free(b);
  libxsmm_free(ht);
  libxsmm_free(it);
  libxsmm_free(ct);
  libxsmm_free(ft);
  libxsmm_free(ot);
  libxsmm_free(dxt);
  libxsmm_free(dhp);
  libxsmm_free(dw);
  libxsmm_free(dr);
  libxsmm_free(db);
  libxsmm_free(dht);
  libxsmm_free(dwtest);
  libxsmm_free(drtest);

  { const char *const env_check_scale = getenv("CHECK_SCALE");
    const double check_scale = LIBXSMM_ABS(0 == env_check_scale ? 1.0 : atof(env_check_scale));
    if (LIBXSMM_NEQ(0, check) && (check < 100.0 * check_scale * diff.normf_rel) && (global_status == LIBXSMM_DNN_SUCCESS)) {
      fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
      exit(EXIT_FAILURE);
    }
  }

  /* some empty lines at the end */
  printf("\n\n\n");

  return EXIT_SUCCESS;
}

