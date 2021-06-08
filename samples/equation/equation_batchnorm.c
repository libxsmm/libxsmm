/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include "../../include/libxsmm_intrinsics_x86.h"

#define ALIGNDOWN(N, A) ((N) & ~((A)-1))
#define USE_VECTORIZED_PATH 1

inline __m512 _mm512_convert_bf_ps(__m256i a) { return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(a),16)); }
inline __m256i _mm256_convert_ps_bf(__m512 a) { return _mm512_cvtepi32_epi16(_mm512_srai_epi32(LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(a),16)); }
inline __m512 _mm512_loadu_ps_auto (libxsmm_bfloat16 const* mem_addr) { return _mm512_convert_bf_ps(_mm256_loadu_si256((__m256i*)mem_addr));}
inline __m512 _mm512_maskz_loadu_ps_auto (__mmask16 k, libxsmm_bfloat16 const* mem_addr) { return _mm512_convert_bf_ps(_mm256_maskz_loadu_epi16(k, (__m256i*)mem_addr));}
inline void _mm512_storeu_ps_auto (libxsmm_bfloat16* mem_addr, __m512 a) { _mm256_storeu_si256 ((__m256i*)mem_addr, _mm256_convert_ps_bf(a)); }
inline void _mm512_mask_storeu_ps_auto (libxsmm_bfloat16* mem_addr, __mmask16 k, __m512 a) { _mm256_mask_storeu_epi16 ((__m256i*)mem_addr, k, _mm256_convert_ps_bf(a)); }

float upconvert_bf16(libxsmm_bfloat16 x) {
  union libxsmm_bfloat16_hp bf16_hp;
  bf16_hp.i[1] = x;
  bf16_hp.i[0] = 0;
  return bf16_hp.f;
}

void tpp_batchnorm_fwd_bf16(long N, long CP, long HW, long CB, libxsmm_bfloat16 *pinp, libxsmm_bfloat16 *pgamma, libxsmm_bfloat16 *pbeta, float *mean, float *var, libxsmm_bfloat16 *pout, float eps, libxsmm_matrix_eqn_function func10, libxsmm_meltwfunction_unary reduce_HW_kernel) {


  LIBXSMM_ALIGNED(float sum_X_X2[2*CP*CB], 64);
  LIBXSMM_ALIGNED(float s[CP*CB], 64);
  LIBXSMM_ALIGNED(float b[CP*CB], 64);

  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, inp, pinp, CP, HW, CB);            /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, out, pout, CP, HW, CB);            /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, gamma, pgamma, CB);                /* [CP, CB] */
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, beta, pbeta, CB);                  /* [CP, CB] */

  #pragma omp parallel for
  for(int j = 0; j < CP*CB; j++){                               /* Initialize sum and sum_square array */
      sum_X_X2[j] = 0.0f;
      sum_X_X2[CP*CB + j] = 0.0f;
  }

  #pragma omp parallel for reduction(+: sum_X_X2[:2*CP*CB])                   /* Parallelize over batches with multiple threads reducing to sum_X_X2 array */
  for(int n = 0; n < N; n++){
    libxsmm_meltw_unary_param reduce_HW_params;       /*Private params and tmp array */
    LIBXSMM_ALIGNED(float tmp[2*CB], 64);
    reduce_HW_params.out.primary   = tmp;                                                         /* [2*CB]  */

    for(int cp = 0; cp < CP; cp++){
      reduce_HW_params.in.primary    = &LIBXSMM_VLA_ACCESS(4, inp, n, cp, 0, 0, CP, HW, CB);
      reduce_HW_kernel(&reduce_HW_params);                                                       /* [HW, CB] -----> [2 * CB] */
      for(int cb = 0; cb < CB; cb++){                                                            /* Update tmp array */
        sum_X_X2[cp*CB + cb] += tmp[cb];
        sum_X_X2[CP*CB + (cp*CB + cb)] += tmp[CB + cb];
      }
    }
  }


  #pragma omp parallel for
  for(int j = 0; j < CP*CB; j++){
    mean[j] = sum_X_X2[j] / ((float)N * HW);                                           /* E[X] */
    var[j] = (sum_X_X2[CP*CB + j] / ((float)N * HW)) - (mean[j]*mean[j]);              /* var(X) = E[X^2] - (E[X])^2 */
    s[j] = 1.0f / (sqrt(var[j] + eps));                                                /* s = 1/sqrt(var(X) + eps)     [CP, CB] */
    b[j] = -1 * mean[j] * s[j];                                                        /* b = -E[X]/sqrt(var(X) + eps) [CP, CB] */
  }

  #pragma omp parallel for
  for(int n = 0; n < N; n++){                                                                /* Parallelize over batches */
    libxsmm_matrix_arg arg_array[5];                                                         /* private eqn args and params*/
    libxsmm_matrix_eqn_param eqn_param;
    for (int cp = 0; cp < CP; cp++){
      arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, n, cp, 0, 0, CP, HW, CB);           /* [HW, CB] */
      arg_array[1].primary = &s[cp*CB];                                                      /* [CB] */
      arg_array[2].primary = &b[cp*CB];                                                      /* [CB] */
      arg_array[3].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);                       /* [CB] */
      arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, beta, cp, 0, CB);                        /* [CB] */
      eqn_param.inputs = arg_array;
      eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, out, n, cp, 0, 0, CP, HW, CB);       /* [HW,CB] */
      func10(&eqn_param);                                                                    /* Normalization equation -> y = ((s*x + b)*gamma + beta) */
    }
  }
}

void tpp_batchnorm_bwd_bf16(long N, long CP, long HW, long CB, libxsmm_bfloat16 *pdout, libxsmm_bfloat16 *pinp, float *mean, float *var, libxsmm_bfloat16 *pgamma, libxsmm_bfloat16 *pdin, float *pdgamma, float *pdbeta,
    libxsmm_matrix_eqn_function dgamma_func, libxsmm_matrix_eqn_function dbeta_func, libxsmm_matrix_eqn_function db_func, libxsmm_matrix_eqn_function ds_func, libxsmm_matrix_eqn_function din_func) {


  const float scale = 1.0f / ((float)N*HW);                   /* Scaling parameter*/

  LIBXSMM_ALIGNED(float a[CP*CB], 64);
  LIBXSMM_ALIGNED(float b[CP*CB], 64);
  LIBXSMM_ALIGNED(float c[CP*CB], 64);
  LIBXSMM_ALIGNED(float d_array[2*CP*CB], 64);                /* For reduction array of dgaama and dbeta*/

  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, din, pdin, CP, HW, CB);          /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, inp, pinp, CP, HW, CB);          /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, dout, pdout, CP, HW, CB);        /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, gamma, pgamma, CB);              /* [CP, CB] */
  LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, CB);            /* [CP, CB] */
  LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, CB);              /* [CP, CB] */

  #pragma omp parallel for
  for(int j = 0; j < CP*CB; j++){                             /* Initialize the arrays */
    a[j] = var[j];
    b[j] = -a[j]*mean[j];
    d_array[j] = 0;
    d_array[CP*CB + j] = 0;
  }


  double final_ds = 0.0f;                                     /* Double needed because reducing too many values */
  double final_db = 0.0f;

  #pragma omp parallel for reduction(+: final_ds, final_db) reduction(+: d_array[:2*CP*CB])     /* Parallelize over batches and reduce the values into d_array, final_ds, final_db */
  for (int n = 0; n < N; n++) {
    float ds = 0.0f;
    float db = 0.0f;
    libxsmm_matrix_arg arg_array[8];                                                           /* Private values of args and params */
    libxsmm_matrix_eqn_param eqn_param;

    for (int cp = 0; cp < CP; cp++) {
      arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, n, cp, 0, 0, CP, HW, CB);
      arg_array[1].primary = &a[cp*CB];
      arg_array[2].primary = &b[cp*CB];
      arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, n, cp, 0, 0, CP, HW, CB);
      /* arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, CB);  */
      /* arg_array[5].primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, CB);   */
      arg_array[4].primary = &d_array[cp*CB];
      arg_array[5].primary = &d_array[CP*CB + cp*CB];
      arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
      /* arg_array[7].primary = &c[cp*CB]; */
      eqn_param.inputs = arg_array;

      eqn_param.output.primary = &ds;
      ds_func(&eqn_param);                                                                  /* ds += dout * gamma * inp */

      eqn_param.output.primary = &db;
      db_func(&eqn_param);                                                                  /* db += dout * gamma */

      /* eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, CB); */
      eqn_param.output.primary = &d_array[cp*CB];
      dgamma_func(&eqn_param);                                                             /* dgamma += (a * inp + b) * dout */

      /* eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, CB); */
      eqn_param.output.primary = &d_array[CP*CB + cp*CB];
      dbeta_func(&eqn_param);                                                              /* dbeta += dout */

      final_ds += ds;
      final_db += db;
    }
  }

  /*
  #pragma omp parallel for
  for(int j = 0; j < CP*CB; j++){
    b[j] = (final_db * mean[j] - final_ds) * a[j] * a[j] * a[j] * scale;
    c[j] = -b[j] * mean[j] - final_db * a[j] * scale;
  }*/

  #pragma omp parallel for collapse(2)
  for (int cp = 0; cp < CP; cp++) {
    for (int cb = 0; cb < CB; cb++){
      LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, CB) = d_array[cp*CB + cb];                      /* Copy d_array data into dgamma and dbeta */
      LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, CB) = d_array[CP*CB + cp*CB + cb];
      b[cp*CB + cb] = (final_db * mean[cp*CB + cb] - final_ds) * a[cp*CB + cb] * a[cp*CB + cb] * a[cp*CB + cb] * scale;
      c[cp*CB + cb] = -b[cp*CB + cb] * mean[cp*CB + cb] - final_db * a[cp*CB + cb] * scale;
    }
  }

  #pragma omp parallel for                                                                  /* Parallelize over batches */
  for(int n = 0; n < N; n++){
    libxsmm_matrix_arg arg_array[8];                                                        /* Private eqn args and params */
    libxsmm_matrix_eqn_param eqn_param;
    for (int cp = 0; cp < CP; cp++) {
      arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, n, cp, 0, 0, CP, HW, CB);
      arg_array[1].primary = &a[cp*CB];
      arg_array[2].primary = &b[cp*CB];
      arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, n, cp, 0, 0, CP, HW, CB);
      /* arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, CB);  */
      /* arg_array[5].primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, CB);   */
      arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
      arg_array[7].primary = &c[cp*CB];
      eqn_param.inputs = arg_array;
      eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, din, n, cp, 0, 0, CP, HW, CB);
      din_func(&eqn_param);                                                                 /* din = dout * a * gamma + b * inp + c */
    }
  }
}

void tpp_batchnorm_fwd_fp32(long N, long CP, long HW, long CB, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout, float eps, libxsmm_matrix_eqn_function func10, libxsmm_meltwfunction_unary reduce_HW_kernel) {


  LIBXSMM_ALIGNED(float sum_X_X2[2*CP*CB], 64);
  LIBXSMM_ALIGNED(float s[CP*CB], 64);
  LIBXSMM_ALIGNED(float b[CP*CB], 64);

  LIBXSMM_VLA_DECL(4, float, inp, pinp, CP, HW, CB);            /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, float, out, pout, CP, HW, CB);            /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);                /* [CP, CB] */
  LIBXSMM_VLA_DECL(2, float, beta, pbeta, CB);                  /* [CP, CB] */

  #pragma omp parallel for
  for(int j = 0; j < CP*CB; j++){                               /* Initialize sum and sum_square array */
      sum_X_X2[j] = 0.0f;
      sum_X_X2[CP*CB + j] = 0.0f;
  }

  #pragma omp parallel for reduction(+: sum_X_X2[:2*CP*CB])                   /* Parallelize over batches with multiple threads reducing to sum_X_X2 array */
  for(int n = 0; n < N; n++){
    libxsmm_meltw_unary_param reduce_HW_params;       /*Private params and tmp array */
    LIBXSMM_ALIGNED(float tmp[2*CB], 64);
    reduce_HW_params.out.primary   = tmp;                                                         /* [2*CB]  */

    for(int cp = 0; cp < CP; cp++){
      reduce_HW_params.in.primary    = &LIBXSMM_VLA_ACCESS(4, inp, n, cp, 0, 0, CP, HW, CB);
      reduce_HW_kernel(&reduce_HW_params);                                                       /* [HW, CB] -----> [2 * CB] */
      for(int cb = 0; cb < CB; cb++){                                                            /* Update tmp array */
        sum_X_X2[cp*CB + cb] += tmp[cb];
        sum_X_X2[CP*CB + (cp*CB + cb)] += tmp[CB + cb];
      }
    }
  }


  #pragma omp parallel for
  for(int j = 0; j < CP*CB; j++){
    mean[j] = sum_X_X2[j] / ((float)N * HW);                                           /* E[X] */
    var[j] = (sum_X_X2[CP*CB + j] / ((float)N * HW)) - (mean[j]*mean[j]);              /* var(X) = E[X^2] - (E[X])^2 */
    s[j] = 1.0f / (sqrt(var[j] + eps));                                                /* s = 1/sqrt(var(X) + eps)     [CP, CB] */
    b[j] = -1 * mean[j] * s[j];                                                        /* b = -E[X]/sqrt(var(X) + eps) [CP, CB] */
  }

  #pragma omp parallel for
  for(int n = 0; n < N; n++){                                                                /* Parallelize over batches */
    libxsmm_matrix_arg arg_array[5];                                                         /* private eqn args and params*/
    libxsmm_matrix_eqn_param eqn_param;
    for (int cp = 0; cp < CP; cp++){
      arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, n, cp, 0, 0, CP, HW, CB);           /* [HW, CB] */
      arg_array[1].primary = &s[cp*CB];                                                      /* [CB] */
      arg_array[2].primary = &b[cp*CB];                                                      /* [CB] */
      arg_array[3].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);                       /* [CB] */
      arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, beta, cp, 0, CB);                        /* [CB] */
      eqn_param.inputs = arg_array;
      eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, out, n, cp, 0, 0, CP, HW, CB);       /* [HW,CB] */
      func10(&eqn_param);                                                                    /* Normalization equation -> y = ((s*x + b)*gamma + beta) */
    }
  }
}

void tpp_batchnorm_bwd_fp32(long N, long CP, long HW, long CB, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta,
    libxsmm_matrix_eqn_function dgamma_func, libxsmm_matrix_eqn_function dbeta_func, libxsmm_matrix_eqn_function db_func, libxsmm_matrix_eqn_function ds_func, libxsmm_matrix_eqn_function din_func) {


  const float scale = 1.0f / ((float)N*HW);                   /* Scaling parameter*/

  LIBXSMM_ALIGNED(float a[CP*CB], 64);
  LIBXSMM_ALIGNED(float b[CP*CB], 64);
  LIBXSMM_ALIGNED(float c[CP*CB], 64);
  LIBXSMM_ALIGNED(float d_array[2*CP*CB], 64);                /* For reduction array of dgaama and dbeta*/

  LIBXSMM_VLA_DECL(4, float, din, pdin, CP, HW, CB);          /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, float, inp, pinp, CP, HW, CB);          /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, float, dout, pdout, CP, HW, CB);        /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);              /* [CP, CB] */
  LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, CB);            /* [CP, CB] */
  LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, CB);              /* [CP, CB] */

  #pragma omp parallel for
  for(int j = 0; j < CP*CB; j++){                             /* Initialize the arrays */
    a[j] = var[j];
    b[j] = -a[j]*mean[j];
    d_array[j] = 0;
    d_array[CP*CB + j] = 0;
  }


  double final_ds = 0.0f;                                     /* Double needed because reducing too many values */
  double final_db = 0.0f;

  #pragma omp parallel for reduction(+: final_ds, final_db) reduction(+: d_array[:2*CP*CB])     /* Parallelize over batches and reduce the values into d_array, final_ds, final_db */
  for (int n = 0; n < N; n++) {
    float ds = 0.0f;
    float db = 0.0f;
    libxsmm_matrix_arg arg_array[8];                                                           /* Private values of args and params */
    libxsmm_matrix_eqn_param eqn_param;

    for (int cp = 0; cp < CP; cp++) {
      arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, n, cp, 0, 0, CP, HW, CB);
      arg_array[1].primary = &a[cp*CB];
      arg_array[2].primary = &b[cp*CB];
      arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, n, cp, 0, 0, CP, HW, CB);
      /* arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, CB);  */
      /* arg_array[5].primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, CB);   */
      arg_array[4].primary = &d_array[cp*CB];
      arg_array[5].primary = &d_array[CP*CB + cp*CB];
      arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
      /* arg_array[7].primary = &c[cp*CB]; */
      eqn_param.inputs = arg_array;

      eqn_param.output.primary = &ds;
      ds_func(&eqn_param);                                                                  /* ds += dout * gamma * inp */

      eqn_param.output.primary = &db;
      db_func(&eqn_param);                                                                  /* db += dout * gamma */

      /* eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, CB); */
      eqn_param.output.primary = &d_array[cp*CB];
      dgamma_func(&eqn_param);                                                             /* dgamma += (a * inp + b) * dout */

      /* eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, CB); */
      eqn_param.output.primary = &d_array[CP*CB + cp*CB];
      dbeta_func(&eqn_param);                                                              /* dbeta += dout */

      final_ds += ds;
      final_db += db;
    }
  }

  /*
  #pragma omp parallel for
  for(int j = 0; j < CP*CB; j++){
    b[j] = (final_db * mean[j] - final_ds) * a[j] * a[j] * a[j] * scale;
    c[j] = -b[j] * mean[j] - final_db * a[j] * scale;
  }*/

  #pragma omp parallel for collapse(2)
  for (int cp = 0; cp < CP; cp++) {
    for (int cb = 0; cb < CB; cb++){
      LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, CB) = d_array[cp*CB + cb];                      /* Copy d_array data into dgamma and dbeta */
      LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, CB) = d_array[CP*CB + cp*CB + cb];
      b[cp*CB + cb] = (final_db * mean[cp*CB + cb] - final_ds) * a[cp*CB + cb] * a[cp*CB + cb] * a[cp*CB + cb] * scale;
      c[cp*CB + cb] = -b[cp*CB + cb] * mean[cp*CB + cb] - final_db * a[cp*CB + cb] * scale;
    }
  }

  #pragma omp parallel for                                                                  /* Parallelize over batches */
  for(int n = 0; n < N; n++){
    libxsmm_matrix_arg arg_array[8];                                                        /* Private eqn args and params */
    libxsmm_matrix_eqn_param eqn_param;
    for (int cp = 0; cp < CP; cp++) {
      arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, n, cp, 0, 0, CP, HW, CB);
      arg_array[1].primary = &a[cp*CB];
      arg_array[2].primary = &b[cp*CB];
      arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, n, cp, 0, 0, CP, HW, CB);
      /* arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, CB);  */
      /* arg_array[5].primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, CB);   */
      arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
      arg_array[7].primary = &c[cp*CB];
      eqn_param.inputs = arg_array;
      eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, din, n, cp, 0, 0, CP, HW, CB);
      din_func(&eqn_param);                                                                 /* din = dout * a * gamma + b * inp + c */
    }
  }
}


void scaler_batchnorm_fwd_fp32(long N, long CP, long HW, long CB, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout, float eps){

  LIBXSMM_ALIGNED(float sum_X[CP*CB], 64);
  LIBXSMM_ALIGNED(float sum_X2[CP*CB], 64);
  LIBXSMM_ALIGNED(float s[CP*CB], 64);
  LIBXSMM_ALIGNED(float b[CP*CB], 64);

  LIBXSMM_VLA_DECL(4, float, inp, pinp, CP, HW, CB);            /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, float, out, pout, CP, HW, CB);            /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2, float, beta, pbeta, CB);

  #pragma omp parallel for collapse(2)
  for(int cp = 0; cp < CP; cp++){
    for(int cb = 0; cb < CB; cb++){
      sum_X[cp*CB + cb] = 0.0f;
      sum_X2[cp*CB + cb] = 0.0f;
    }
  }

  #pragma omp parallel for collapse(2)                          /* Parallelize over all channels */
  for(int cp = 0; cp < CP; cp++){
    for(int cb = 0; cb < CB; cb++){
      float value;
      for(int n = 0; n < N; n++){
        for(int hw = 0; hw < HW; hw++){
          sum_X[cp*CB + cb] += LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB);
          sum_X2[cp*CB + cb] += (LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB)*LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB));
        }
      }
      mean[cp*CB + cb] = sum_X[cp*CB + cb] / ((float)N * HW);                                           /* E[X] */
      var[cp*CB + cb] = (sum_X2[cp*CB + cb] / ((float)N * HW)) - (mean[cp*CB + cb]*mean[cp*CB + cb]);   /* var(X) = E[X^2] - (E[X])^2 */
      s[cp*CB + cb] = 1.0f / (sqrt(var[cp*CB + cb] + eps));                                             /* s = 1/sqrt(var(X) + eps)     [CP, CB] */
      b[cp*CB + cb] = -1 * mean[cp*CB + cb] * s[cp*CB + cb];                                            /* b = -E[X]/sqrt(var(X) + eps) [CP, CB] */

      for(int n = 0; n < N; n++){
        for(int hw = 0; hw < HW; hw++){
          value = LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB);
          value = ((value * s[cp*CB + cb]) + b[cp*CB + cb]) * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB) + LIBXSMM_VLA_ACCESS(2, beta, cp, cb, CB);        /* Normalization equation -> y = ((s*x + b)*gamma + beta) */
          LIBXSMM_VLA_ACCESS(4, out, n, cp, hw, cb, CP, HW, CB) = value;
        }
      }
    }
  }
}

void scaler_batchnorm_bwd_fp32(long N, long CP, long HW, long CB, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta) {

  LIBXSMM_ALIGNED(float a[CP*CB], 64);
  LIBXSMM_ALIGNED(float b[CP*CB], 64);
  LIBXSMM_ALIGNED(float c[CP*CB], 64);
  LIBXSMM_ALIGNED(float d_array[2*CP*CB], 64);

  LIBXSMM_VLA_DECL(4, float, din, pdin, CP, HW, CB);
  LIBXSMM_VLA_DECL(4, float, inp, pinp, CP, HW, CB);
  LIBXSMM_VLA_DECL(4, float, dout, pdout, CP, HW, CB);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, CB);

  #pragma omp parallel for collapse(2)
  for (int cp = 0; cp < CP; cp++) {                               /* Initialize all values */
    for (int cb = 0; cb < CB; cb++) {
      a[cp*CB + cb] = var[cp*CB + cb];
      b[cp*CB + cb] = -a[cp*CB + cb]*mean[cp*CB + cb];
      d_array[cp*CB + cb] = 0.0f;
      d_array[CP*CB + cp*CB + cb] = 0.0f;
    }
  }


  double ds = 0.0f;                                               /* double needed because reducing too many values */
  double db = 0.0f;
  const float scale = 1.0f / ((float)N*HW);

  #pragma omp parallel for reduction(+: ds, db) reduction(+: d_array[:2*CP*CB])           /* Parallelize over batches and reduce to d_array, ds, and db */
  for(int n = 0; n < N; n++){
    for (int cp = 0; cp < CP; cp++) {                    /* dgamma += (a * inp + b) * dout , dbeta += dout, ds += dout * gamma * inp, db += dout * gamma */
      for (int cb = 0; cb < CB; cb++) {
        for (int hw = 0; hw < HW; hw++){
          d_array[cp*CB + cb] += (a[cp*CB + cb] * LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB) + b[cp*CB + cb]) * LIBXSMM_VLA_ACCESS(4, dout, n, cp, hw, cb, CP, HW, CB);
          d_array[CP*CB + cp*CB +cb] += LIBXSMM_VLA_ACCESS(4, dout, n, cp, hw, cb, CP, HW, CB);
          /* LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, CB) += (a[cp*CB + cb] * LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB) + b[cp*CB + cb]) * LIBXSMM_VLA_ACCESS(4, dout, n, cp, hw, cb, CP, HW, CB);  */
          /* LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, CB) += LIBXSMM_VLA_ACCESS(4, dout, n, cp, hw, cb, CP, HW, CB);  */
          ds += LIBXSMM_VLA_ACCESS(4, dout, n, cp, hw, cb, CP, HW, CB) * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB) * LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB);
          db += LIBXSMM_VLA_ACCESS(4, dout, n, cp, hw, cb, CP, HW, CB) * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB);
        }
      }
    }
  }

  /* #pragma omp parallel for
  for(int j = 0; j < CP*CB; j++){
    b[j] = (db * mean[j] - ds) * a[j] * a[j] * a[j] * scale;
    c[j] = -b[j] * mean[j] - db * a[j] * scale;
  } */

  #pragma omp parallel for collapse(2)
  for (int cp = 0; cp < CP; cp++) {
    for (int cb = 0; cb < CB; cb++){
      LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, CB) = d_array[cp*CB + cb];                    /* Copy d_array data into dgamma and dbeta */
      LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, CB) = d_array[CP*CB + cp*CB + cb];
      b[cp*CB + cb] = (db * mean[cp*CB + cb] - ds) * a[cp*CB + cb] * a[cp*CB + cb] * a[cp*CB + cb] * scale;
      c[cp*CB + cb] = -b[cp*CB + cb] * mean[cp*CB + cb] - db * a[cp*CB + cb] * scale;
    }
  }

  #pragma omp parallel for
  for(int n = 0; n < N; n++){                                                             /* Parallelize over batches */
    for (int cp = 0; cp < CP; cp++) {                                                     /* din = dout * a * gamma + b * inp + c */
      for (int cb = 0; cb < CB; cb++) {
        for (int hw = 0; hw < HW; hw++){
          LIBXSMM_VLA_ACCESS(4, din, n, cp, hw, cb, CP, HW, CB) = LIBXSMM_VLA_ACCESS(4, dout, n, cp, hw, cb, CP, HW, CB)  * a[cp*CB + cb] * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB) + b[cp*CB + cb] * LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB) + c[cp*CB + cb];
        }
      }
    }
  }
}

int main( int argc, char* argv[] ) {
  libxsmm_blasint my_eqn10, my_eqn11, my_eqn12, my_eqn13, my_eqn14, my_eqn15;
  libxsmm_matrix_eqn_function func10, func11, func12, func13, func14, func15;
  libxsmm_meltw_unary_flags jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type  unary_type;
  libxsmm_meltwfunction_unary reduce_HW_kernel;

  const float eps = FLT_EPSILON;
  libxsmm_blasint i, it, ld, tmp_ld, tmp_ld2;
  unsigned long long l_start, l_end;
  double l_total = 0, l_total2 = 0;
  double t_vec = 0, t_tpp = 0;
  libxsmm_matdiff_info norms_out;
  float *inp, *out, *dinp, *dout, *eqn_dinp, *eqn_dout, *dbeta, *eqn_dbeta, *dgamma, *eqn_dgamma, *eqn_out, *gamma, *beta, *cache_fl, *mean, *var, sum = 0.0;
  libxsmm_bfloat16 *bf16_inp, *bf16_out, *bf16_dinp, *bf16_dout, *bf16_eqn_dinp, *bf16_eqn_dout, *bf16_gamma, *bf16_beta, *bf16_eqn_out;
  int N = 28;
  int CP = 32;
  int HW = 784;
  int CB = 16;
  int iters = 100;
  int datatype_mode = 0;
  libxsmm_datatype  in_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype  out_dt = LIBXSMM_DATATYPE_F32;

  if ( argc > 1 ) N = atoi(argv[1]);
  if ( argc > 2 ) CP = atoi(argv[2]);
  if ( argc > 3 ) HW = atoi(argv[3]);
  if ( argc > 4 ) CB = atoi(argv[4]);
  if ( argc > 5 ) datatype_mode = atoi(argv[5]);
  if ( argc > 6 ) iters = atoi(argv[6]);

  if (datatype_mode == 0) {
    in_dt = LIBXSMM_DATATYPE_F32;
    out_dt = LIBXSMM_DATATYPE_F32;
  } else if (datatype_mode == 1) {
    in_dt = LIBXSMM_DATATYPE_BF16;
    out_dt = LIBXSMM_DATATYPE_BF16;
  } else {
    printf("ERROR: Supporting only FP32 and BF16 precisions...\n");
  }

  inp = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*CB,   2097152);
  out = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*CB,   2097152);
  dinp = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*CB,   2097152);
  dout = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*CB,   2097152);
  dgamma = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  dbeta = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  eqn_dinp = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*CB,   2097152);
  eqn_dout = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*CB,   2097152);
  eqn_dgamma = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  eqn_dbeta = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  gamma = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  beta = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  mean = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  var = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  eqn_out  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*CB,   2097152);
  cache_fl  = (float*) libxsmm_aligned_malloc( sizeof(float)*1024*1024,   2097152);

  bf16_inp = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*CP*HW*CB,   2097152);
  bf16_out = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*CP*HW*CB,   2097152);
  bf16_dinp = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*CP*HW*CB,   2097152);
  bf16_dout = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*CP*HW*CB,   2097152);
  bf16_eqn_dinp = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*CP*HW*CB,   2097152);
  bf16_eqn_dout = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*CP*HW*CB,   2097152);
  bf16_gamma = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*CP*CB,   2097152);
  bf16_beta = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*CP*CB,   2097152);
  bf16_eqn_out  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*CP*HW*CB,   2097152);

  libxsmm_init();
  libxsmm_matdiff_clear(&norms_out);

  /* Initializing arrays */
  for ( i = 0; i < N*CP*HW*CB; ++i ) {
    inp[i] = (float)libxsmm_rng_f64();
    out[i] = (float)libxsmm_rng_f64();
    eqn_out[i] = out[i];
    dinp[i] = (float)libxsmm_rng_f64();
    dout[i] = (float)libxsmm_rng_f64();
    eqn_dinp[i] = dinp[i];
    eqn_dout[i] = dout[i];
    libxsmm_rne_convert_fp32_bf16( &inp[i], &bf16_inp[i], 1 );
    libxsmm_rne_convert_fp32_bf16( &out[i], &bf16_out[i], 1 );
    libxsmm_rne_convert_fp32_bf16( &eqn_out[i], &bf16_eqn_out[i], 1 );
    libxsmm_rne_convert_fp32_bf16( &dout[i], &bf16_dout[i], 1 );
    libxsmm_rne_convert_fp32_bf16( &eqn_dout[i], &bf16_eqn_dout[i], 1 );
    libxsmm_rne_convert_fp32_bf16( &dinp[i], &bf16_dinp[i], 1 );
    libxsmm_rne_convert_fp32_bf16( &eqn_dinp[i], &bf16_eqn_dinp[i], 1 );
  }

  for ( i = 0; i < CP*CB; ++i ) {
    gamma[i] = (float)libxsmm_rng_f64();
    beta[i] = (float)libxsmm_rng_f64();
    dbeta[i] = (float)libxsmm_rng_f64();
    dgamma[i] = (float)libxsmm_rng_f64();
    eqn_dbeta[i] = dbeta[i];
    eqn_dgamma[i] = dgamma[i];
    libxsmm_rne_convert_fp32_bf16( &gamma[i], &bf16_gamma[i], 1 );
    libxsmm_rne_convert_fp32_bf16( &beta[i], &bf16_beta[i], 1 );
  }

  for (i = 0; i < 1024 * 1024; i++ ) {
    cache_fl[i] = (float)libxsmm_rng_f64();
  }


  /* TPPs for reducing X and X2 in HW*/
  ld = CB;
  tmp_ld = CB;

  unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD;
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  reduce_HW_kernel = libxsmm_dispatch_meltw_unary(CB, HW, &ld, &tmp_ld, in_dt, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);

  /* TPP for scaling */
  ld = CB;
  tmp_ld = 1;
  tmp_ld2 = 1;
  my_eqn10 = libxsmm_matrix_eqn_create();                                                        /* y = (s*x + b)*gamma + beta */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn10, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn10, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, CB, HW, ld, 0, 0, in_dt );                         /* x = [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, CB, 1, tmp_ld, 1, 0, LIBXSMM_DATATYPE_F32 );       /* s = [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, CB, 1, tmp_ld, 2, 0, LIBXSMM_DATATYPE_F32 );       /* b = [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, CB, 1, tmp_ld2, 3, 0, in_dt );                     /* gamma = [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, CB, 1, tmp_ld2, 4, 0, in_dt );                     /* beta = [CB] */
  func10 = libxsmm_dispatch_matrix_eqn( CB, HW, &ld, out_dt, my_eqn10 );                         /* y = [HW, CB] */


  /* Check correctness */
  if (datatype_mode == 0) {
    scaler_batchnorm_fwd_fp32(N, CP, HW, CB, inp, gamma, beta, mean, var, out, eps);
    tpp_batchnorm_fwd_fp32(N, CP, HW, CB, inp, gamma, beta, mean, var, eqn_out, eps, func10, reduce_HW_kernel);
  } else if (datatype_mode == 1) {
    scaler_batchnorm_fwd_fp32(N, CP, HW, CB, inp, gamma, beta, mean, var, out, eps);
    tpp_batchnorm_fwd_bf16(N, CP, HW, CB, bf16_inp, bf16_gamma, bf16_beta, mean, var, bf16_eqn_out, eps, func10, reduce_HW_kernel);
    for ( i = 0; i < N*CP*HW*CB; ++i ) {
      /* out[i] = upconvert_bf16(bf16_out[i]); */
      eqn_out[i] = upconvert_bf16(bf16_eqn_out[i]);
    }
  }

  /* compare */
  printf("############################################\n");
  if (datatype_mode == 0) {
    printf("# Correctness FP32 FWD Batchnorm - Output  #\n");
  } else {
    printf("# Correctness BF16 FWD Batchnorm - Output  #\n");
  }
  printf("############################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, N*CP*HW*CB, 1, out, eqn_out, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  if (datatype_mode == 0) {
    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i];
    }
    scaler_batchnorm_fwd_fp32(N, CP, HW, CB, inp, gamma, beta, mean, var, out, eps);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      scaler_batchnorm_fwd_fp32(N, CP, HW, CB, inp, gamma, beta, mean, var, out, eps);
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("TPP batchnorm time FWD  = %.5g\n", ((double)(l_total)));
    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i] + (float)l_total;
    }
    tpp_batchnorm_fwd_fp32(N, CP, HW, CB, inp, gamma, beta, mean, var, eqn_out, eps, func10, reduce_HW_kernel);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      tpp_batchnorm_fwd_fp32(N, CP, HW, CB, inp, gamma, beta, mean, var, eqn_out, eps, func10, reduce_HW_kernel);
    }
    l_end = libxsmm_timer_tick();
    l_total2 = libxsmm_timer_duration(l_start, l_end);
    printf("TPP batchnorm time FWD  = %.5g\n", ((double)(l_total2)));
    printf("Speedup FWD is %.5g\n", l_total/l_total2);
  } else if (datatype_mode == 1) {
    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i];
    }
    scaler_batchnorm_fwd_fp32(N, CP, HW, CB, inp, gamma, beta, mean, var, out, eps);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      scaler_batchnorm_fwd_fp32(N, CP, HW, CB, inp, gamma, beta, mean, var, out, eps);
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("Scaler batchnorm (FP32) time FWD  = %.5g\n", ((double)(l_total)));
    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i] + (float)l_total;
    }
    tpp_batchnorm_fwd_bf16(N, CP, HW, CB, bf16_inp, bf16_gamma, bf16_beta, mean, var, bf16_eqn_out, eps, func10, reduce_HW_kernel);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      tpp_batchnorm_fwd_bf16(N, CP, HW, CB, bf16_inp, bf16_gamma, bf16_beta, mean, var, bf16_eqn_out, eps, func10, reduce_HW_kernel);
    }
    l_end = libxsmm_timer_tick();
    l_total2 = libxsmm_timer_duration(l_start, l_end);
    printf("TPP batchnorm (BF16) time FWD  = %.5g\n", ((double)(l_total2)));
    printf("Speedup FWD is %.5g\n", l_total/l_total2);
  }


  /* Group norm equations */
  /* Create MatEq for bwd layernorm */

  ld = CB;
  tmp_ld2 = 1;

  /* dgamma function  */
  my_eqn11 = libxsmm_matrix_eqn_create();                                                       /* dgamma = ((inp *a + b) * dout) + dgamma */
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn11, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32);                   /* dgamma = ((inp *a + b) * dout) + dgamma */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn11, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);   /* [HW, CB] -> [CB] */
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn11, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32);                   /* ((inp *a + b) * dout) */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn11, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, CB, HW, ld, 0, 0, in_dt );                        /* inp [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, CB, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 );           /* a [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, CB, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );           /* b [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, CB, HW, ld, 3, 0, in_dt );                        /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, CB, 1, 1, 4, 0, LIBXSMM_DATATYPE_F32 );           /* dgamma [CB] */
  func11 = libxsmm_dispatch_matrix_eqn( CB, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn11 );      /* dgamma [CB] */

  /* dbeta function  */
  my_eqn12 = libxsmm_matrix_eqn_create();                                                       /* dbeta [CB] = dout [HW, CB] + dbeta [CB] */
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn12, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );      /* dbeta_tmp [HW, CB] */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn12, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);  /* [HW, CB] -> [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn12, CB, HW, ld, 3, 0, in_dt );                        /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn12, CB, 1, 1, 5, 0, LIBXSMM_DATATYPE_F32 );           /* dbeta [CB] */
  func12 = libxsmm_dispatch_matrix_eqn( CB, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn12 );      /* dbeta [CB] */

  /* db equation */
  my_eqn13 = libxsmm_matrix_eqn_create();                                                       /* db = (dout * gamma) */
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn13, LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn13, CB, HW, ld, 3, 0, in_dt );                        /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn13, CB, 1, 1, 6, 0, in_dt );                          /* gamma [CB] */
  func13 = libxsmm_dispatch_matrix_eqn( 1, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn13 );       /* db [1] */

  /* ds equation */
  my_eqn14 = libxsmm_matrix_eqn_create();                                                       /* ds = ((dout * gamma) * inp) */
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn14, LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn14, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn14, CB, HW, ld, 3, 0, in_dt );                        /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn14, CB, 1, 1, 6, 0, in_dt );                          /* gamma [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn14, CB, HW, ld, 0, 0, in_dt );                        /* inp [HW, CB] */
  func14 = libxsmm_dispatch_matrix_eqn( 1, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn14 );       /* ds [1] */

  /* din equation */
  my_eqn15 = libxsmm_matrix_eqn_create();                                                       /* din = ((gamma * a) * dout) + (inp * b + c) */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn15, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn15, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, CB, 1, 1, 6, 0, in_dt );                          /* gamma [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, CB, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 );           /* a [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, CB, HW, ld, 3, 0, in_dt );                        /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn15, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, CB, HW, ld, 0, 0, in_dt );                        /* inp [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, CB, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );           /* b [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, CB, 1, 1, 7, 0, LIBXSMM_DATATYPE_F32 );           /* c [CB] */
  func15 = libxsmm_dispatch_matrix_eqn( CB, HW, &ld, in_dt, my_eqn15 );                         /* din [HW, CB] */

  if (datatype_mode == 0) {
    scaler_batchnorm_bwd_fp32(N, CP, HW, CB, dout, inp, mean, var, gamma, dinp, dgamma, dbeta);
    tpp_batchnorm_bwd_fp32(N, CP, HW, CB, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func13, func14, func15);
  } else if (datatype_mode == 1) {
    scaler_batchnorm_bwd_fp32(N, CP, HW, CB, dout, inp, mean, var, gamma, dinp, dgamma, dbeta);
    tpp_batchnorm_bwd_bf16(N, CP, HW, CB, bf16_eqn_dout, bf16_inp, mean, var, bf16_gamma, bf16_eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func13, func14, func15);
    for ( i = 0; i < N*CP*HW*CB; ++i ) {
      /* dinp[i] = upconvert_bf16(bf16_dinp[i]); */
      eqn_dinp[i] = upconvert_bf16(bf16_eqn_dinp[i]);
    }
  }

  /* compare */
  printf("############################################\n");
  if (datatype_mode == 0) {
    printf("# Correctness FP32 BWD Batchnorm - Dinput  #\n");
  } else {
    printf("# Correctness BF16 BWD Batchnorm - Dinput  #\n");
  }
  printf("############################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, N*CP*HW*CB, 1, dinp, eqn_dinp, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  printf("###########################################\n");
  if (datatype_mode == 0) {
    printf("# Correctness FP32 BWD Batchnorm - Dbeta  #\n");
  } else {
    printf("# Correctness BF16 BWD Batchnorm - Dbeta  #\n");
  }
  printf("###########################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, CP*CB, 1, dbeta, eqn_dbeta, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  printf("############################################\n");
  if (datatype_mode == 0) {
    printf("# Correctness FP32 BWD Batchnorm - Dgamma  #\n");
  } else {
    printf("# Correctness BF16 BWD Batchnorm - Dgamma #\n");
  }
  printf("############################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, CP*CB, 1, dgamma, eqn_dgamma, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  if (datatype_mode == 0) {
    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i];
    }
    scaler_batchnorm_bwd_fp32(N, CP, HW, CB, dout, inp, mean, var, gamma, dinp, dgamma, dbeta);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      scaler_batchnorm_bwd_fp32(N, CP, HW, CB, dout, inp, mean, var, gamma, dinp, dgamma, dbeta);
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("Scaler batchnorm time BWD = %.5g\n", ((double)(l_total)));
    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i] + (float)l_total;
    }
    tpp_batchnorm_bwd_fp32(N, CP, HW, CB, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func13, func14, func15);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      tpp_batchnorm_bwd_fp32(N, CP, HW, CB, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func13, func14, func15);
    }
    l_end = libxsmm_timer_tick();
    l_total2 = libxsmm_timer_duration(l_start, l_end);
    printf("TPP batchnorm time BWD = %.5g\n", ((double)(l_total2)));
    printf("Speedup BWD is %.5g\n", l_total/l_total2);
  } else if (datatype_mode == 1) {
    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i];
    }
    scaler_batchnorm_bwd_fp32(N, CP, HW, CB, dout, inp, mean, var, gamma, dinp, dgamma, dbeta);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      scaler_batchnorm_bwd_fp32(N, CP, HW, CB, dout, inp, mean, var, gamma, dinp, dgamma, dbeta);
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("Scaler batchnorm (FP32) time BWD  = %.5g\n", ((double)(l_total)));
    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i] + (float)l_total;
    }
    tpp_batchnorm_bwd_bf16(N, CP, HW, CB, bf16_eqn_dout, bf16_inp, mean, var, bf16_gamma, bf16_eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func13, func14, func15);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      tpp_batchnorm_bwd_bf16(N, CP, HW, CB, bf16_eqn_dout, bf16_inp, mean, var, bf16_gamma, bf16_eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func13, func14, func15);
    }
    l_end = libxsmm_timer_tick();
    l_total2 = libxsmm_timer_duration(l_start, l_end);
    printf("TPP batchnorm (BF16) time BWD = %.5g\n", ((double)(l_total2)));
    printf("Speedup BWD is %.5g\n", l_total/l_total2);
  }
  /* printf("Running sum is %.5f\n", sum); */

  t_tpp += l_total2;
  t_vec += l_total;

  printf("\n\n=================================\n");
  printf("Total Speedup via TPP Matrix equation is %.5g\n", t_vec/t_tpp);
  printf("=================================\n");

  libxsmm_free(inp);
  libxsmm_free(out);
  libxsmm_free(dinp);
  libxsmm_free(dout);
  libxsmm_free(eqn_dinp);
  libxsmm_free(eqn_dout);
  libxsmm_free(bf16_dinp);
  libxsmm_free(bf16_dout);
  libxsmm_free(bf16_eqn_dinp);
  libxsmm_free(bf16_eqn_dout);
  libxsmm_free(dgamma);
  libxsmm_free(dbeta);
  libxsmm_free(eqn_dgamma);
  libxsmm_free(eqn_dbeta);
  libxsmm_free(mean);
  libxsmm_free(var);
  libxsmm_free(gamma);
  libxsmm_free(beta);
  libxsmm_free(eqn_out);
  libxsmm_free(bf16_inp);
  libxsmm_free(bf16_out);
  libxsmm_free(bf16_gamma);
  libxsmm_free(bf16_beta);
  libxsmm_free(bf16_eqn_out);
  libxsmm_free(cache_fl);

  return 0;
}
