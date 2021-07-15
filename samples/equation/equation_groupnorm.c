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
#include <libxsmm_intrinsics_x86.h>

#define ALIGNDOWN(N, A) ((N) & ~((A)-1))
#define USE_VECTORIZED_PATH 1


float upconvert_bf16(libxsmm_bfloat16 x) {
  union libxsmm_bfloat16_hp bf16_hp;
  bf16_hp.i[1] = x;
  bf16_hp.i[0] = 0;
  return bf16_hp.f;
}

void tpp_layernorm_fwd_fp32(long S1, long S2, long S3, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout, float eps, libxsmm_matrix_eqn_function func0, libxsmm_meltwfunction_unary reduce_rows_kernel, libxsmm_meltwfunction_unary reduce_cols_kernel) {
  int s2;
  libxsmm_matrix_eqn_param eqn_param;
  libxsmm_meltw_unary_param m_reduce_rows_params, v_reduce_rows_params, reduce_cols_params;
  LIBXSMM_ALIGNED(float tmp[2*S3], 64);
  const float c = 1.0/((float)S1*S3);
  float m, v, s, b;
  libxsmm_matrix_arg  arg_array[5];
  LIBXSMM_VLA_DECL(3, float, inp, pinp, S2, S3);
  LIBXSMM_VLA_DECL(3, float, out, pout, S2, S3);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, S3);
  LIBXSMM_VLA_DECL(2, float, beta, pbeta, S3);

  eqn_param.inputs = arg_array;
  reduce_cols_params.out.primary   = tmp;
  arg_array[1].primary = &s;
  arg_array[2].primary = &b;
  arg_array[3].primary = &LIBXSMM_VLA_ACCESS(2, gamma, 0, 0, S3);
  arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, beta, 0, 0, S3);
  m_reduce_rows_params.in.primary    = tmp;
  m_reduce_rows_params.out.primary   = &m;
  v_reduce_rows_params.in.primary    = &tmp[S3];
  v_reduce_rows_params.out.primary   = &v;

  for (s2 = 0; s2 < S2; s2++) {
    reduce_cols_params.in.primary    = &LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3);
    reduce_cols_kernel(&reduce_cols_params);
    reduce_rows_kernel(&m_reduce_rows_params);
    reduce_rows_kernel(&v_reduce_rows_params);
    m = m * c;
    v = v * c;
    v = LIBXSMM_MAX(v - m * m, 0.0f);
    v = 1.0f / ((float)sqrt(v+eps));
    mean[s2] = m;
    var[s2] = v;
    s = v;
    b = -1.0 * v * m;
    arg_array[0].primary = &LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3);
    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, out, 0, s2, 0, S2, S3);
    func0(&eqn_param);
  }
}

void tpp_layernorm_fwd_bf16(long S1, long S2, long S3, libxsmm_bfloat16 *pinp, libxsmm_bfloat16 *pgamma, libxsmm_bfloat16 *pbeta, float *mean, float *var, libxsmm_bfloat16 *pout, float eps, libxsmm_matrix_eqn_function func0, libxsmm_meltwfunction_unary reduce_rows_kernel, libxsmm_meltwfunction_unary reduce_cols_kernel) {
  int s2;
  libxsmm_matrix_eqn_param eqn_param;
  libxsmm_meltw_unary_param m_reduce_rows_params, v_reduce_rows_params, reduce_cols_params;
  LIBXSMM_ALIGNED(float tmp[2*S3], 64);
  const float c = 1.0/((float)S1*S3);
  float m, v, s, b;
  libxsmm_matrix_arg  arg_array[5];
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, inp, pinp, S2, S3);
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, out, pout, S2, S3);
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, gamma, pgamma, S3);
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, beta, pbeta, S3);

  eqn_param.inputs = arg_array;
  reduce_cols_params.out.primary   = tmp;
  arg_array[1].primary = &s;
  arg_array[2].primary = &b;
  arg_array[3].primary = &LIBXSMM_VLA_ACCESS(2, gamma, 0, 0, S3);
  arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, beta, 0, 0, S3);
  m_reduce_rows_params.in.primary    = tmp;
  m_reduce_rows_params.out.primary   = &m;
  v_reduce_rows_params.in.primary    = &tmp[S3];
  v_reduce_rows_params.out.primary   = &v;

  for (s2 = 0; s2 < S2; s2++) {
    reduce_cols_params.in.primary    = &LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3);
    reduce_cols_kernel(&reduce_cols_params);
    reduce_rows_kernel(&m_reduce_rows_params);
    reduce_rows_kernel(&v_reduce_rows_params);
    m = m * c;
    v = v * c;
    v = LIBXSMM_MAX(v - m * m, 0.0f);
    v = 1.0f / ((float)sqrt(v+eps));
    mean[s2] = m;
    var[s2] = v;
    s = v;
    b = -1.0 * v * m;
    arg_array[0].primary = &LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3);
    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, out, 0, s2, 0, S2, S3);
    func0(&eqn_param);
  }
}


void tpp_layernorm_bwd_fp32(long S1, long S2, long S3, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta,
    libxsmm_matrix_eqn_function dgamma_func, libxsmm_matrix_eqn_function dbeta_func, libxsmm_matrix_eqn_function db_func, libxsmm_matrix_eqn_function ds_func, libxsmm_matrix_eqn_function din_func) {
  int s2;
  float a, b, c, db, ds;
  const float scale = 1.0f / ((float)S1*S3);
  LIBXSMM_VLA_DECL(3, float, din, pdin, S2, S3);
  LIBXSMM_VLA_DECL(3, float, inp, pinp, S2, S3);
  LIBXSMM_VLA_DECL(3, float, dout, pdout, S2, S3);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, S3);
  LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, S3);
  LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, S3);

  libxsmm_matrix_eqn_param eqn_param;
  libxsmm_matrix_arg arg_array[8];
  eqn_param.inputs = arg_array;

  arg_array[1].primary = &a;
  arg_array[2].primary = &b;
  arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, dgamma, 0, 0, S3);
  arg_array[5].primary = &LIBXSMM_VLA_ACCESS(2, dbeta, 0, 0, S3);
  arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, 0, 0, S3);
  arg_array[7].primary = &c;

  for (s2 = 0; s2 < S2; s2++) {
    a = var[s2];
    b = -a*mean[s2];
    arg_array[0].primary = &LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3);
    arg_array[3].primary = &LIBXSMM_VLA_ACCESS(3, dout, 0, s2, 0, S2, S3);

    eqn_param.output.primary = &ds;
    ds_func(&eqn_param);

    eqn_param.output.primary = &db;
    db_func(&eqn_param);

    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, 0, 0, S3);
    dgamma_func(&eqn_param);

    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, 0, 0, S3);
    dbeta_func(&eqn_param);

    b = (db * mean[s2] - ds) * a * a * a * scale;
    c = -b * mean[s2] - db * a * scale;

    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, din, 0, s2, 0, S2, S3);
    din_func(&eqn_param);
  }
}


void tpp_layernorm_bwd_bf16(long S1, long S2, long S3, libxsmm_bfloat16 *pdout, libxsmm_bfloat16 *pinp, float *mean, float *var, libxsmm_bfloat16 *pgamma, libxsmm_bfloat16 *pdin, float *pdgamma, float *pdbeta,
    libxsmm_matrix_eqn_function dgamma_func, libxsmm_matrix_eqn_function dbeta_func, libxsmm_matrix_eqn_function db_func, libxsmm_matrix_eqn_function ds_func, libxsmm_matrix_eqn_function din_func) {
  int s2;
  float a, b, c, db, ds;
  const float scale = 1.0f / ((float)S1*S3);
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, din, pdin, S2, S3);
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, inp, pinp, S2, S3);
  LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, dout, pdout, S2, S3);
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, gamma, pgamma, S3);
  LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, S3);
  LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, S3);

  libxsmm_matrix_eqn_param eqn_param;
  libxsmm_matrix_arg arg_array[8];
  eqn_param.inputs = arg_array;

  arg_array[1].primary = &a;
  arg_array[2].primary = &b;
  arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, dgamma, 0, 0, S3);
  arg_array[5].primary = &LIBXSMM_VLA_ACCESS(2, dbeta, 0, 0, S3);
  arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, 0, 0, S3);
  arg_array[7].primary = &c;

  for (s2 = 0; s2 < S2; s2++) {
    a = var[s2];
    b = -a*mean[s2];
    arg_array[0].primary = &LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3);
    arg_array[3].primary = &LIBXSMM_VLA_ACCESS(3, dout, 0, s2, 0, S2, S3);

    eqn_param.output.primary = &ds;
    ds_func(&eqn_param);

    eqn_param.output.primary = &db;
    db_func(&eqn_param);

    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, 0, 0, S3);
    dgamma_func(&eqn_param);

    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, 0, 0, S3);
    dbeta_func(&eqn_param);

    b = (db * mean[s2] - ds) * a * a * a * scale;
    c = -b * mean[s2] - db * a * scale;

    eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, din, 0, s2, 0, S2, S3);
    din_func(&eqn_param);
  }
}

void tpp_groupnorm_fwd_fp32(long CP, long NB, long HW, long CB, long G, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout, float eps, libxsmm_matrix_eqn_function func10, libxsmm_meltwfunction_unary reduce_HW_kernel, libxsmm_meltwfunction_unary reduce_rows_kernel, libxsmm_meltwfunction_unary reduce_groups_kernel) {


  libxsmm_matrix_eqn_param eqn_param;
  libxsmm_meltw_unary_param m_reduce_rows_params, m_reduce_groups_params, v_reduce_rows_params, v_reduce_groups_params, reduce_HW_params;

  LIBXSMM_ALIGNED(float tmp[2*CB], 64);
  LIBXSMM_ALIGNED(float sum_X[G], 64);
  LIBXSMM_ALIGNED(float sum_X2[G], 64);
  LIBXSMM_ALIGNED(float s[CP*CB], 64);
  LIBXSMM_ALIGNED(float b[CP*CB], 64);

  LIBXSMM_VLA_DECL(4, float, inp, pinp, NB, HW, CB);            /* [Cp, HW, NB, CB] */
  LIBXSMM_VLA_DECL(4, float, out, pout, NB, HW, CB);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2, float, beta, pbeta, CB);

  libxsmm_matrix_arg  arg_array[5];

  int group_size, g;
  float m, v;
  group_size = (CP*CB)/G;

  libxsmm_blasint ldo = G;
  libxsmm_meltwfunction_unary all_zero_kernel = libxsmm_dispatch_meltw_unary(G, 1, NULL, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  if ( all_zero_kernel == NULL) {
      fprintf( stderr, "JIT for initialization by unary all zero copy kernel failed. Bailing...!\n");
      exit(-1);
  }
  libxsmm_meltw_unary_param all_zero_param;

  for (int nb = 0; nb < NB; nb++) {                       /* [CP, nb, HW, CB] */
    all_zero_param.out.primary = sum_X;
    all_zero_kernel(&all_zero_param);
    all_zero_param.out.primary = sum_X2;
    all_zero_kernel(&all_zero_param);

    for (int cp = 0; cp < CP; cp++){                      /* [cp, nb, HW, CB] */

      reduce_HW_params.in.primary    = &LIBXSMM_VLA_ACCESS(4, inp, cp, nb, 0, 0, NB, HW, CB);      /* [HW, CB] -----> [2 * CB] */
      reduce_HW_params.out.primary   = tmp;                  /* [2*CB] */
      reduce_HW_kernel(&reduce_HW_params);

      if (group_size >= CB){                                 /* Group size >= block size  (Ex.- CP = 4, CB = 16, G = 2, group_size = 32) */
        g = (cp*CB)/group_size;                              /* determine current group */
        m_reduce_rows_params.in.primary    = tmp;
        m_reduce_rows_params.out.primary   = &m;
        v_reduce_rows_params.in.primary    = &tmp[CB];
        v_reduce_rows_params.out.primary   = &v;
        reduce_rows_kernel(&m_reduce_rows_params);
        reduce_rows_kernel(&v_reduce_rows_params);
        sum_X[g] += m;
        sum_X2[g] += v;
      }
      else{                                                 /* Group size < block size  (Ex.- CP = 4, CB = 16, G = 32, group_size = 2) */
        for(int i=0; i < CB; i += group_size){
          m_reduce_groups_params.in.primary    = &tmp[i];
          m_reduce_groups_params.out.primary   = &sum_X[cp*(CB/group_size) + (i/group_size)];
          v_reduce_groups_params.in.primary    = &tmp[CB + i];
          v_reduce_groups_params.out.primary   = &sum_X2[cp*(CB/group_size) + (i/group_size)];
          reduce_groups_kernel(&m_reduce_groups_params);
          reduce_groups_kernel(&v_reduce_groups_params);
        }
      }
    }

    for(g = 0; g < G; g++){                                                  /* mean and variance calculation */
      mean[nb*G + g] = sum_X[g] / ((float)group_size * HW);
      var[nb*G + g] = (sum_X2[g] / ((float)group_size * HW)) - (mean[nb*G + g]*mean[nb*G + g]);        /* var = E[X^2] - (E[X])^2 */

      for(int j = 0; j < group_size; j++){
        s[g*group_size + j] = 1.0f / ((float)sqrt(var[nb*G + g] + eps));                               /* 1/sqrt(var(X) + eps) */
        b[g*group_size + j] = -1 * mean[nb*G + g] * s[g*group_size + j];                               /* -E[X]/sqrt(var(X) + eps) */
      }
    }

    for (int cp = 0; cp < CP; cp++){
      arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, cp, nb, 0, 0, NB, HW, CB);                       /* [HW, CB] */
      arg_array[1].primary = &s[cp*CB];                                                                   /* [CB] */
      arg_array[2].primary = &b[cp*CB];                                                                   /* [CB] */
      arg_array[3].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);                                    /* [CB] */
      arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, beta, cp, 0, CB);                                     /* [CB] */
      eqn_param.inputs = arg_array;
      eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, out, cp, nb, 0, 0, NB, HW, CB);                   /* [HW,CB] */
      func10(&eqn_param);                                                                                 /* Normalization equation -> y = ((s*x + b)*gamma + beta) */
    }
  }
}

void tpp_groupnorm_fwd_bf16(long CP, long NB, long HW, long CB, long G, libxsmm_bfloat16 *pinp, libxsmm_bfloat16 *pgamma, libxsmm_bfloat16 *pbeta, float *mean, float *var, libxsmm_bfloat16 *pout, float eps, libxsmm_matrix_eqn_function func10, libxsmm_meltwfunction_unary reduce_HW_kernel, libxsmm_meltwfunction_unary reduce_rows_kernel, libxsmm_meltwfunction_unary reduce_groups_kernel) {


  libxsmm_matrix_eqn_param eqn_param;
  libxsmm_meltw_unary_param m_reduce_rows_params, m_reduce_groups_params, v_reduce_rows_params, v_reduce_groups_params, reduce_HW_params;

  LIBXSMM_ALIGNED(float tmp[2*CB], 64);
  LIBXSMM_ALIGNED(float sum_X[G], 64);
  LIBXSMM_ALIGNED(float sum_X2[G], 64);
  LIBXSMM_ALIGNED(float s[CP*CB], 64);
  LIBXSMM_ALIGNED(float b[CP*CB], 64);

  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, inp, pinp, NB, HW, CB);            /* [CP, HW, NB, CB] */
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, out, pout, NB, HW, CB);
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, beta, pbeta, CB);

  libxsmm_matrix_arg  arg_array[5];

  int group_size, g;
  float m, v;
  group_size = (CP*CB)/G;

  libxsmm_blasint ldo = G;
  libxsmm_meltwfunction_unary all_zero_kernel = libxsmm_dispatch_meltw_unary(G, 1, NULL, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  if ( all_zero_kernel == NULL) {
      fprintf( stderr, "JIT for initialization by unary all zero copy kernel failed. Bailing...!\n");
      exit(-1);
  }
  libxsmm_meltw_unary_param all_zero_param;

  for (int nb = 0; nb < NB; nb++) {                       /* [CP, nb, HW, CB] */
    all_zero_param.out.primary = sum_X;
    all_zero_kernel(&all_zero_param);
    all_zero_param.out.primary = sum_X2;
    all_zero_kernel(&all_zero_param);

    for (int cp = 0; cp < CP; cp++){                      /* [cp, nb, HW, CB] */

      reduce_HW_params.in.primary    = &LIBXSMM_VLA_ACCESS(4, inp, cp, nb, 0, 0, NB, HW, CB);      /* [HW, CB] -----> [2 * CB] */
      reduce_HW_params.out.primary   = tmp;                  /* [2*CB] */
      reduce_HW_kernel(&reduce_HW_params);

      if (group_size >= CB){                                 /* Group size >= block size  (Ex.- CP = 4, CB = 16, G = 2, group_size = 32) */
        g = (cp*CB)/group_size;                              /* determine current group */
        m_reduce_rows_params.in.primary    = tmp;
        m_reduce_rows_params.out.primary   = &m;
        v_reduce_rows_params.in.primary    = &tmp[CB];
        v_reduce_rows_params.out.primary   = &v;
        reduce_rows_kernel(&m_reduce_rows_params);
        reduce_rows_kernel(&v_reduce_rows_params);
        sum_X[g] += m;
        sum_X2[g] += v;
      }
      else{                                                 /* Group size < block size  (Ex.- CP = 4, CB = 16, G = 32, group_size = 2) */
        for(int i=0; i < CB; i += group_size){
          m_reduce_groups_params.in.primary    = &tmp[i];
          m_reduce_groups_params.out.primary   = &sum_X[cp*(CB/group_size) + (i/group_size)];
          v_reduce_groups_params.in.primary    = &tmp[CB + i];
          v_reduce_groups_params.out.primary   = &sum_X2[cp*(CB/group_size) + (i/group_size)];
          reduce_groups_kernel(&m_reduce_groups_params);
          reduce_groups_kernel(&v_reduce_groups_params);
        }
      }
    }

    for(g = 0; g < G; g++){                                                  /* mean and variance calculation */
      mean[nb*G + g] = sum_X[g] / ((float)group_size * HW);
      var[nb*G + g] = (sum_X2[g] / ((float)group_size * HW)) - (mean[nb*G + g]*mean[nb*G + g]);        /* var = E[X^2] - (E[X])^2 */

      for(int j = 0; j < group_size; j++){
        s[g*group_size + j] = 1.0f / ((float)sqrt(var[nb*G + g] + eps));                               /* 1/sqrt(var(X) + eps) */
        b[g*group_size + j] = -1 * mean[nb*G + g] * s[g*group_size + j];                               /* -E[X]/sqrt(var(X) + eps) */
      }
    }

    for (int cp = 0; cp < CP; cp++){
      arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, cp, nb, 0, 0, NB, HW, CB);                       /* [HW, CB] */
      arg_array[1].primary = &s[cp*CB];                                                                   /* [CB] */
      arg_array[2].primary = &b[cp*CB];                                                                   /* [CB] */
      arg_array[3].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);                                    /* [CB] */
      arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, beta, cp, 0, CB);                                     /* [CB] */
      eqn_param.inputs = arg_array;
      eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, out, cp, nb, 0, 0, NB, HW, CB);                   /* [HW,CB] */
      func10(&eqn_param);                                                                                 /* Normalization equation -> y = ((s*x + b)*gamma + beta) */
    }
  }
}

void tpp_groupnorm_bwd_fp32(long CP, long NB, long HW, long CB, long G, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta,
    libxsmm_matrix_eqn_function dgamma_func, libxsmm_matrix_eqn_function dbeta_func, libxsmm_matrix_eqn_function db_func, libxsmm_matrix_eqn_function ds_func, libxsmm_matrix_eqn_function din_func, float eps) {

  int cp, nb, hw, cb;
  int group_size, g;
  group_size = (CP*CB)/G;

  const float scale = 1.0f / ((float)CP*HW*CB);

  LIBXSMM_ALIGNED(float a[CP*CB], 64);
  LIBXSMM_ALIGNED(float b[CP*CB], 64);
  LIBXSMM_ALIGNED(float c[CP*CB], 64);
  LIBXSMM_ALIGNED(float ds[CP*CB], 64);
  LIBXSMM_ALIGNED(float db[CP*CB], 64);

  LIBXSMM_VLA_DECL(4, float, din, pdin, NB, HW, CB);
  LIBXSMM_VLA_DECL(4, float, inp, pinp, NB, HW, CB);
  LIBXSMM_VLA_DECL(4, float, dout, pdout, NB, HW, CB);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, CB);

  libxsmm_matrix_eqn_param eqn_param;
  libxsmm_matrix_arg arg_array[10];
  eqn_param.inputs = arg_array;

  for (nb = 0; nb < NB; nb++) {
    for(g = 0; g < G; g++){                                                  /* compute a and b for each channel from group means and variance */
      for(int j = 0; j < group_size; j++){
        a[g*group_size + j] = 1.0f / ((float)sqrt(var[nb*G + g] + eps));
        b[g*group_size + j] = -a[g*group_size + j]*mean[nb*G + g];
        ds[g*group_size + j] = 0.0f;
        db[g*group_size + j] = 0.0f;
      }
    }
    for (cp = 0; cp < CP; cp++) {
      arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, cp, nb, 0, 0, NB, HW, CB);
      arg_array[1].primary = &a[cp*CB];
      arg_array[2].primary = &b[cp*CB];
      arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, cp, nb, 0, 0, NB, HW, CB);
      arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, CB);
      arg_array[5].primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, CB);
      arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
      /* arg_array[7].primary = &c[cp*CB]; */
      arg_array[8].primary = &ds[cp*CB];
      arg_array[9].primary = &db[cp*CB];

      eqn_param.output.primary = &ds[cp*CB];
      ds_func(&eqn_param);

      eqn_param.output.primary = &db[cp*CB];
      db_func(&eqn_param);

      eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, CB);
      dgamma_func(&eqn_param);

      eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, CB);
      dbeta_func(&eqn_param);

    }

    /* b = (db * mean[nb] - ds) * a * a * a * scale; */
    /* c = -b * mean[nb] - db * a * scale; */

    for(g = 0; g < G; g++){                                                  /* compute b and c for each channel from group means and variance */
      float gds = 0.0f;
      float gdb = 0.0f;
      for(int j = 0; j < group_size; j++){
        gds += ds[g*group_size + j];                                        /* Group ds and db calculation */
        gdb += db[g*group_size + j];
      }
      for(int j = 0; j < group_size; j++){
        b[g*group_size + j] = (gdb * mean[nb*G + g] - gds) * a[g*group_size + j] * a[g*group_size + j] * a[g*group_size + j] * scale;
        c[g*group_size + j] = -b[g*group_size + j] * mean[nb*G + g] - gdb * a[g*group_size + j] * scale;
      }
    }

    for (cp = 0; cp < CP; cp++) {
      arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, cp, nb, 0, 0, NB, HW, CB);
      arg_array[1].primary = &a[cp*CB];
      arg_array[2].primary = &b[cp*CB];
      arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, cp, nb, 0, 0, NB, HW, CB);
      /* arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, CB); */
      /* arg_array[5].primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, CB); */
      arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
      arg_array[7].primary = &c[cp*CB];
      /* arg_array[8].primary = &ds[cp*CB]; */
      /* arg_array[9].primary = &db[cp*CB]; */
      eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, din, cp, nb, 0, 0, NB, HW, CB);
      din_func(&eqn_param);
    }
  }
}

void tpp_groupnorm_bwd_bf16(long CP, long NB, long HW, long CB, long G, libxsmm_bfloat16 *pdout, libxsmm_bfloat16 *pinp, float *mean, float *var, libxsmm_bfloat16 *pgamma, libxsmm_bfloat16 *pdin, float *pdgamma, float *pdbeta,
    libxsmm_matrix_eqn_function dgamma_func, libxsmm_matrix_eqn_function dbeta_func, libxsmm_matrix_eqn_function db_func, libxsmm_matrix_eqn_function ds_func, libxsmm_matrix_eqn_function din_func, float eps) {

  int cp, nb, hw, cb;
  int group_size, g;
  group_size = (CP*CB)/G;

  const float scale = 1.0f / ((float)CP*HW*CB);

  LIBXSMM_ALIGNED(float a[CP*CB], 64);
  LIBXSMM_ALIGNED(float b[CP*CB], 64);
  LIBXSMM_ALIGNED(float c[CP*CB], 64);
  LIBXSMM_ALIGNED(float ds[CP*CB], 64);
  LIBXSMM_ALIGNED(float db[CP*CB], 64);

  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, din, pdin, NB, HW, CB);
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, inp, pinp, NB, HW, CB);
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, dout, pdout, NB, HW, CB);
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, CB);

  libxsmm_matrix_eqn_param eqn_param;
  libxsmm_matrix_arg arg_array[10];
  eqn_param.inputs = arg_array;

  for (nb = 0; nb < NB; nb++) {
    for(g = 0; g < G; g++){                                                  /* compute a and b for each channel from group means and variance */
      for(int j = 0; j < group_size; j++){
        a[g*group_size + j] = 1.0f / ((float)sqrt(var[nb*G + g] + eps));
        b[g*group_size + j] = -a[g*group_size + j]*mean[nb*G + g];
        ds[g*group_size + j] = 0.0f;
        db[g*group_size + j] = 0.0f;
      }
    }
    for (cp = 0; cp < CP; cp++) {
      arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, cp, nb, 0, 0, NB, HW, CB);
      arg_array[1].primary = &a[cp*CB];
      arg_array[2].primary = &b[cp*CB];
      arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, cp, nb, 0, 0, NB, HW, CB);
      arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, CB);
      arg_array[5].primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, CB);
      arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
      /* arg_array[7].primary = &c[cp*CB]; */
      arg_array[8].primary = &ds[cp*CB];
      arg_array[9].primary = &db[cp*CB];

      eqn_param.output.primary = &ds[cp*CB];
      ds_func(&eqn_param);

      eqn_param.output.primary = &db[cp*CB];
      db_func(&eqn_param);

      eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, CB);
      dgamma_func(&eqn_param);

      eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, CB);
      dbeta_func(&eqn_param);

    }

    /* b = (db * mean[nb] - ds) * a * a * a * scale; */
    /* c = -b * mean[nb] - db * a * scale; */

    for(g = 0; g < G; g++){                                                  /* compute b and c for each channel from group means and variance */
      float gds = 0.0f;
      float gdb = 0.0f;
      for(int j = 0; j < group_size; j++){
        gds += ds[g*group_size + j];                                        /* Group ds and db calculation */
        gdb += db[g*group_size + j];
      }
      for(int j = 0; j < group_size; j++){
        b[g*group_size + j] = (gdb * mean[nb*G + g] - gds) * a[g*group_size + j] * a[g*group_size + j] * a[g*group_size + j] * scale;
        c[g*group_size + j] = -b[g*group_size + j] * mean[nb*G + g] - gdb * a[g*group_size + j] * scale;
      }
    }

    for (cp = 0; cp < CP; cp++) {
      arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, cp, nb, 0, 0, NB, HW, CB);
      arg_array[1].primary = &a[cp*CB];
      arg_array[2].primary = &b[cp*CB];
      arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, cp, nb, 0, 0, NB, HW, CB);
      /* arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, CB); */
      /* arg_array[5].primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, CB); */
      arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
      arg_array[7].primary = &c[cp*CB];
      /* arg_array[8].primary = &ds[cp*CB]; */
      /* arg_array[9].primary = &db[cp*CB]; */
      eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, din, cp, nb, 0, 0, NB, HW, CB);
      din_func(&eqn_param);
    }
  }
}

void scaler_groupnorm_fwd_fp32(long CP, long NB, long HW, long CB, long G, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout, float eps){

  LIBXSMM_ALIGNED(float sum_X[G], 64);
  LIBXSMM_ALIGNED(float sum_X2[G], 64);
  LIBXSMM_ALIGNED(float s[CP*CB], 64);
  LIBXSMM_ALIGNED(float b[CP*CB], 64);

  LIBXSMM_VLA_DECL(4, float, inp, pinp, NB, HW, CB);            /* [Cp, NB, HW, CB] */
  LIBXSMM_VLA_DECL(4, float, out, pout, NB, HW, CB);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2, float, beta, pbeta, CB);

  float m, v, value;

  int group_size, g;
  group_size = (CP*CB)/G;

  for(int nb = 0; nb < NB; nb ++){
    for(g = 0; g < G; g++){
      sum_X[g] = 0.0f;
      sum_X2[g] = 0.0f;
    }
    for(int cp = 0; cp < CP; cp++){                           /* Size = CP*HW*CB*4 */
      m = 0.0f;
      v = 0.0f;
      if (group_size >= CB){                                 /* Group size >= block size  (Ex.- CP = 4, CB = 16, G = 2, group_size = 32) */
        for(int cb = 0; cb < CB; cb++){
          for(int hw = 0; hw < HW; hw++){
            value = LIBXSMM_VLA_ACCESS(4, inp, cp, nb, hw, cb, NB, HW, CB);
            m += value;
            v += (value*value);
          }
        }
        g = (cp*CB)/group_size;                              /* determine current group */
        sum_X[g] += m;
        sum_X2[g] += v;
      }
      else{
        for(int i=0; i < CB; i += group_size){              /* Group size < block size  (Ex.- CP = 4, CB = 16, G = 32, group_size = 2) */
          for(int j = 0; j < group_size; j++){
            for(int hw = 0; hw < HW; hw++){
              value = LIBXSMM_VLA_ACCESS(4, inp, cp, nb, hw, (i + j), NB, HW, CB);
              sum_X[cp*(CB/group_size) + (i/group_size)] += value;
              sum_X2[cp*(CB/group_size) + (i/group_size)] += (value*value);
            }
          }
        }
      }
    }

    for(g = 0; g < G; g++){                                                  /* mean and variance calculation */           /* Size = 2*CP*CB*4 */
      mean[nb*G + g] = sum_X[g] / ((float)group_size * HW);
      var[nb*G + g] = (sum_X2[g] / ((float)group_size * HW)) - (mean[nb*G + g]*mean[nb*G + g]);      /* var = E[X^2] - (E[X])^2        [G] */

      for(int j = 0; j < group_size; j++){
        s[g*group_size + j] = 1.0f / ((float)sqrt(var[nb*G + g] + eps));                               /* s = 1/sqrt(var(X) + eps)     [CP, CB] */
        b[g*group_size + j] = -1 * mean[nb*G + g] * s[g*group_size + j];                               /* b = -E[X]/sqrt(var(X) + eps) [CP, CB] */
      }
    }

    for(int cp = 0; cp < CP; cp++){                                                     /* Size = 2*CP*HW*CB*4 + 2*CP*CB*4 */
      for(int cb = 0; cb < CB; cb++){
        for(int hw = 0; hw < HW; hw++){
          value = LIBXSMM_VLA_ACCESS(4, inp, cp, nb, hw, cb, NB, HW, CB);
          value = ((value * s[cp*CB + cb]) + b[cp*CB + cb]) * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB) + LIBXSMM_VLA_ACCESS(2, beta, cp, cb, CB);        /* Normalization equation -> y = ((s*x + b)*gamma + beta) */
          LIBXSMM_VLA_ACCESS(4, out, cp, nb, hw, cb, NB, HW, CB) = value;
        }
      }
    }
  }                                       /* end loops */
}

void scaler_groupnorm_bwd_fp32(long CP, long NB, long HW, long CB, long G, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta, float eps) {
  int cp, nb, hw, cb;

  int group_size, g;
  group_size = (CP*CB)/G;

  LIBXSMM_ALIGNED(float a[CP*CB], 64);
  LIBXSMM_ALIGNED(float b[CP*CB], 64);
  LIBXSMM_ALIGNED(float c[CP*CB], 64);
  LIBXSMM_ALIGNED(float ds[CP*CB], 64);
  LIBXSMM_ALIGNED(float db[CP*CB], 64);

  LIBXSMM_VLA_DECL(4, float, din, pdin, NB, HW, CB);
  LIBXSMM_VLA_DECL(4, float, inp, pinp, NB, HW, CB);
  LIBXSMM_VLA_DECL(4, float, dout, pdout, NB, HW, CB);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, CB);


  for (nb = 0; nb < NB; nb++) {
    for(g = 0; g < G; g++){                                                  /* compute a and b for each channel from group means and variance */
      for(int j = 0; j < group_size; j++){
        a[g*group_size + j] = 1.0f / ((float)sqrt(var[nb*G + g] + eps));
        b[g*group_size + j] = -a[g*group_size + j]*mean[nb*G + g];
        ds[g*group_size + j] = 0.0f;
        db[g*group_size + j] = 0.0f;
      }
    }

    float scale = 1.0f / (CP * HW* CB);
    for (cp = 0; cp < CP; cp++) {                    /* dgamma += (a * inp + b) * dout , dbeta += dout, ds += dout * gamma * inp, db += dout * gamma */    /* Size = 2*CP*HW*CB*4 */
      for (cb = 0; cb < CB; cb++) {
        for (hw = 0; hw < HW; hw++){
          LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, CB) += (a[cp*CB + cb] * LIBXSMM_VLA_ACCESS(4, inp, cp, nb, hw, cb, NB, HW, CB) + b[cp*CB + cb]) * LIBXSMM_VLA_ACCESS(4, dout, cp, nb, hw, cb, NB, HW, CB);
          LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, CB) += LIBXSMM_VLA_ACCESS(4, dout, cp, nb, hw, cb, NB, HW, CB);
          ds[cp*CB + cb] += LIBXSMM_VLA_ACCESS(4, dout, cp, nb, hw, cb, NB, HW, CB) * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB) * LIBXSMM_VLA_ACCESS(4, inp, cp, nb, hw, cb, NB, HW, CB);
          db[cp*CB + cb] += LIBXSMM_VLA_ACCESS(4, dout, cp, nb, hw, cb, NB, HW, CB) * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB);
        }
      }
    }
    /* b = (db * mean[nb] - ds) * a * a * a * scale; */
    /* c = -b * mean[nb] - db * a * scale; */
    for(g = 0; g < G; g++){                                                  /* compute b and c for each channel from group means and variance */
      float gds = 0.0f;
      float gdb = 0.0f;
      for(int j = 0; j < group_size; j++){
        gds += ds[g*group_size + j];                                        /* Group ds and db calculation */
        gdb += db[g*group_size + j];
      }
      for(int j = 0; j < group_size; j++){
        b[g*group_size + j] = (gdb * mean[nb*G + g] - gds) * a[g*group_size + j] * a[g*group_size + j] * a[g*group_size + j] * scale;
        c[g*group_size + j] = -b[g*group_size + j] * mean[nb*G + g] - gdb * a[g*group_size + j] * scale;
      }
    }

    for (cp = 0; cp < CP; cp++) {                                                     /* din = dout * a * gamma + b * inp + c */  /* Size = 3*CP*HW*CB*4 */
      for (cb = 0; cb < CB; cb++) {
        for (hw = 0; hw < HW; hw++){
          LIBXSMM_VLA_ACCESS(4, din, cp, nb, hw, cb, NB, HW, CB) = LIBXSMM_VLA_ACCESS(4, dout, cp, nb, hw, cb, NB, HW, CB)  * a[cp*CB + cb] * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB) + b[cp*CB + cb] * LIBXSMM_VLA_ACCESS(4, inp, cp, nb, hw, cb, NB, HW, CB) + c[cp*CB + cb];
        }
      }
    }
  }
}


int main( int argc, char* argv[] ) {
  libxsmm_blasint my_eqn0, my_eqn1, my_eqn2, my_eqn3, my_eqn4, my_eqn5, my_eqn10, my_eqn11, my_eqn12, my_eqn13, my_eqn14, my_eqn15;
  libxsmm_matrix_eqn_function func0, func1, func2, func3, func4, func5, func10, func11, func12, func13, func14, func15;
  libxsmm_meltw_unary_flags jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type  unary_type;
  libxsmm_meltwfunction_unary reduce_rows_kernel, reduce_cols_kernel, reduce_HW_kernel, reduce_groups_kernel;

  const float eps = FLT_EPSILON;
  libxsmm_blasint i, it, ld, tmp_ld, tmp_ld2;
  unsigned long long l_start, l_end;
  double l_total = 0, l_total2 = 0;
  double t_vec = 0, t_tpp = 0;
  libxsmm_matdiff_info norms_out;
  float *inp, *out, *dinp, *dout, *eqn_dinp, *eqn_dout, *dbeta, *eqn_dbeta, *dgamma, *eqn_dgamma, *eqn_out, *gamma, *beta, *cache_fl, *mean, *var, sum = 0.0;
  libxsmm_bfloat16 *bf16_inp, *bf16_out, *bf16_dinp, *bf16_dout, *bf16_eqn_dinp, *bf16_eqn_dout, *bf16_gamma, *bf16_beta, *bf16_eqn_out;
  int CP = 64;
  int NB = 64;
  int HW = 1;
  int CB = 64;
  int G = 1;
  int iters = 100;
  int datatype_mode = 0;
  libxsmm_datatype  in_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype  out_dt = LIBXSMM_DATATYPE_F32;

  if ( argc > 1 ) CP = atoi(argv[1]);
  if ( argc > 2 ) NB = atoi(argv[2]);
  if ( argc > 3 ) HW = atoi(argv[3]);
  if ( argc > 4 ) CB = atoi(argv[4]);
  if ( argc > 5 ) G = atoi(argv[5]);
  if ( argc > 6 ) datatype_mode = atoi(argv[6]);
  if ( argc > 7 ) iters = atoi(argv[7]);

  if (datatype_mode == 0) {
    in_dt = LIBXSMM_DATATYPE_F32;
    out_dt = LIBXSMM_DATATYPE_F32;
  } else if (datatype_mode == 1) {
    in_dt = LIBXSMM_DATATYPE_BF16;
    out_dt = LIBXSMM_DATATYPE_BF16;
  } else {
    printf("ERROR: Supporting only FP32 and BF16 precisions...\n");
  }

  inp = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*NB*HW*CB,   2097152);
  out = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*NB*HW*CB,   2097152);
  dinp = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*NB*HW*CB,   2097152);
  dout = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*NB*HW*CB,   2097152);
  dgamma = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  dbeta = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  eqn_dinp = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*NB*HW*CB,   2097152);
  eqn_dout = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*NB*HW*CB,   2097152);
  eqn_dgamma = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  eqn_dbeta = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  gamma = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  beta = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  mean = (float*) libxsmm_aligned_malloc( sizeof(float)*NB*G,   2097152);
  var = (float*) libxsmm_aligned_malloc( sizeof(float)*NB*G,   2097152);
  eqn_out  = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*NB*HW*CB,   2097152);
  cache_fl  = (float*) libxsmm_aligned_malloc( sizeof(float)*1024*1024,   2097152);

  bf16_inp = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*CP*NB*HW*CB,   2097152);
  bf16_out = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*CP*NB*HW*CB,   2097152);
  bf16_dinp = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*CP*NB*HW*CB,   2097152);
  bf16_dout = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*CP*NB*HW*CB,   2097152);
  bf16_eqn_dinp = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*CP*NB*HW*CB,   2097152);
  bf16_eqn_dout = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*CP*NB*HW*CB,   2097152);
  bf16_gamma = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*CP*CB,   2097152);
  bf16_beta = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*CP*CB,   2097152);
  bf16_eqn_out  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*CP*NB*HW*CB,   2097152);

  libxsmm_init();
  libxsmm_matdiff_clear(&norms_out);

  /* Initializing arrays */
  for ( i = 0; i < CP*NB*HW*CB; ++i ) {
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

  /* TPP for reducing groups */
  libxsmm_blasint group_size = (CP*CB)/G;
  ld = group_size;                /* group_size = (CP*CB)/G */
  tmp_ld = 1;
  unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD;
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS;
  reduce_groups_kernel = libxsmm_dispatch_meltw_unary(group_size, 1, &ld, &tmp_ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);


  /* TPPs for reducing X and X2 */
  ld = NB*CB;
  tmp_ld = CB;
  unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD;
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  reduce_cols_kernel = libxsmm_dispatch_meltw_unary(CB, CP, &ld, &tmp_ld, in_dt, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);

  ld = CB;
  tmp_ld = 1;
  unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD;
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS;
  reduce_rows_kernel = libxsmm_dispatch_meltw_unary(CB, 1, &ld, &tmp_ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);

  /* TPP for scaling */
  ld = NB*CB;
  tmp_ld = 1;
  tmp_ld2 = CB;
  my_eqn0 = libxsmm_matrix_eqn_create();
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn0, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn0, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, CB, CP, ld, 0, 0, in_dt );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, 1, 1, tmp_ld, 1, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, 1, 1, tmp_ld, 2, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, CB, CP, tmp_ld2, 3, 0, in_dt );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, CB, CP, tmp_ld2, 4, 0, in_dt );
  func0 = libxsmm_dispatch_matrix_eqn( CB, CP, &ld, out_dt, my_eqn0 );


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
    scaler_groupnorm_fwd_fp32(CP, NB, HW, CB, G, inp, gamma, beta, mean, var, out, eps);
    tpp_groupnorm_fwd_fp32(CP, NB, HW, CB, G, inp, gamma, beta, mean, var, eqn_out, eps, func10, reduce_HW_kernel, reduce_rows_kernel, reduce_groups_kernel);
    /* tpp_layernorm_fwd_fp32(CP, NB, CB, inp, gamma, beta, mean, var, eqn_out, eps, func0, reduce_rows_kernel, reduce_cols_kernel); */
  } else if (datatype_mode == 1) {
    scaler_groupnorm_fwd_fp32(CP, NB, HW, CB, G, inp, gamma, beta, mean, var, out, eps);
    tpp_groupnorm_fwd_bf16(CP, NB, HW, CB, G, bf16_inp, bf16_gamma, bf16_beta, mean, var, bf16_eqn_out, eps, func10, reduce_HW_kernel, reduce_rows_kernel, reduce_groups_kernel);
    /* tpp_layernorm_fwd_bf16(CP, NB, CB, bf16_inp, bf16_gamma, bf16_beta, mean, var, bf16_eqn_out, eps, func0, reduce_rows_kernel, reduce_cols_kernel); */

    for ( i = 0; i < CP*NB*HW*CB; ++i ) {
      /* out[i] = upconvert_bf16(bf16_out[i]); */
      eqn_out[i] = upconvert_bf16(bf16_eqn_out[i]);
    }
  }

  // printf("\n mean \n");
  // for ( i = 0; i < NB*G; ++i )
  //   printf("%f \t", mean[i]);

  // printf("\n variance \n");
  // for ( i = 0; i < NB*G; ++i )
  //   printf("%f \t", var[i]);

  /* compare */
  printf("############################################\n");
  if (datatype_mode == 0) {
    printf("# Correctness FP32 FWD Groupnorm - Output  #\n");
  } else {
    printf("# Correctness BF16 FWD Groupnorm - Output  #\n");
  }
  printf("############################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, CP*NB*HW*CB, 1, out, eqn_out, 0, 0);
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
    scaler_groupnorm_fwd_fp32(CP, NB, HW, CB, G, inp, gamma, beta, mean, var, out, eps);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      scaler_groupnorm_fwd_fp32(CP, NB, HW, CB, G, inp, gamma, beta, mean, var, out, eps);
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("Scaler time FWD  = %.5g\n", ((double)(l_total)));
    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i] + (float)l_total;
    }
    tpp_groupnorm_fwd_fp32(CP, NB, HW, CB, G, inp, gamma, beta, mean, var, eqn_out, eps, func10, reduce_HW_kernel, reduce_rows_kernel, reduce_groups_kernel);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      tpp_groupnorm_fwd_fp32(CP, NB, HW, CB, G, inp, gamma, beta, mean, var, eqn_out, eps, func10, reduce_HW_kernel, reduce_rows_kernel, reduce_groups_kernel);
    }
    l_end = libxsmm_timer_tick();
    l_total2 = libxsmm_timer_duration(l_start, l_end);
    printf("TPP groupnorm time FWD  = %.5g\n", ((double)(l_total2)));
    printf("Speedup FWD is %.5g\n", l_total/l_total2);
  } else if (datatype_mode == 1) {
    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i];
    }
    /* tpp_groupnorm_fwd_bf16(CP, NB, HW, CB, G, inp, gamma, beta, mean, var, eqn_out, eps, func10, reduce_HW_kernel, reduce_rows_kernel, reduce_groups_kernel); */
    scaler_groupnorm_fwd_fp32(CP, NB, HW, CB, G, inp, gamma, beta, mean, var, out, eps);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      /* tpp_groupnorm_fwd_bf16(CP, NB, HW, CB, G, inp, gamma, beta, mean, var, eqn_out, eps, func10, reduce_HW_kernel, reduce_rows_kernel, reduce_groups_kernel); */
      scaler_groupnorm_fwd_fp32(CP, NB, HW, CB, G, inp, gamma, beta, mean, var, out, eps);
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("Scaler FP32 groupnorm time FWD  = %.5g\n", ((double)(l_total)));

    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i] + (float)l_total;
    }
    /* tpp_layernorm_fwd_bf16(CP, NB, CB, bf16_inp, bf16_gamma, bf16_beta, mean, var, bf16_eqn_out, eps, func0, reduce_rows_kernel, reduce_cols_kernel); */
    tpp_groupnorm_fwd_bf16(CP, NB, HW, CB, G, inp, gamma, beta, mean, var, eqn_out, eps, func10, reduce_HW_kernel, reduce_rows_kernel, reduce_groups_kernel);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      /* tpp_layernorm_fwd_bf16(CP, NB, CB, bf16_inp, bf16_gamma, bf16_beta, mean, var, bf16_eqn_out, eps, func0, reduce_rows_kernel, reduce_cols_kernel); */
      tpp_groupnorm_fwd_bf16(CP, NB, HW, CB, G, inp, gamma, beta, mean, var, eqn_out, eps, func10, reduce_HW_kernel, reduce_rows_kernel, reduce_groups_kernel);
    }
    l_end = libxsmm_timer_tick();
    l_total2 = libxsmm_timer_duration(l_start, l_end);
    printf("TPP BF16 groupnorm time FWD  = %.5g\n", ((double)(l_total2)));
    printf("Speedup FWD is %.5g\n", l_total/l_total2);
  }

  t_tpp = l_total2;
  t_vec = l_total;

  /* Create MatEq for bwd layernorm */
  tmp_ld = CB;
  ld = NB*CB;
  tmp_ld2 = 1;

  /* dgamma function  */                                    /* dgamma += (a * inp + b) * dout */
  my_eqn1 = libxsmm_matrix_eqn_create();
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn1, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn1, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, CB, CP, ld, 0, 0, in_dt );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, CB, CP, ld, 3, 0, in_dt );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, CB, CP, tmp_ld, 4, 0, LIBXSMM_DATATYPE_F32 );
  func1 = libxsmm_dispatch_matrix_eqn( CB, CP, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn1 );

  /* dbeta function  */                                   /* dbeta += dout */
  my_eqn2 = libxsmm_matrix_eqn_create();
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn2, CB, CP, ld, 3, 0, in_dt );
  libxsmm_matrix_eqn_push_back_arg( my_eqn2, CB, CP, tmp_ld, 5, 0, LIBXSMM_DATATYPE_F32 );
  func2 = libxsmm_dispatch_matrix_eqn( CB, CP, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn2 );

  /* db equation */
  my_eqn3 = libxsmm_matrix_eqn_create();                  /* db += dout * gamma */
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn3, LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn3, CB, CP, ld, 3, 0, in_dt );
  libxsmm_matrix_eqn_push_back_arg( my_eqn3, CB, CP, tmp_ld, 6, 0, in_dt );
  func3 = libxsmm_dispatch_matrix_eqn( 1, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn3 );

  /* ds equation */
  my_eqn4 = libxsmm_matrix_eqn_create();                  /* ds += dout * gamma * inp */
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn4, LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn4, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn4, CB, CP, ld, 3, 0, in_dt );
  libxsmm_matrix_eqn_push_back_arg( my_eqn4, CB, CP, tmp_ld, 6, 0, in_dt );
  libxsmm_matrix_eqn_push_back_arg( my_eqn4, CB, CP, ld, 0, 0, in_dt );
  func4 = libxsmm_dispatch_matrix_eqn( 1, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn4 );

  /* din equation */
  my_eqn5 = libxsmm_matrix_eqn_create();                  /* din = dout * a * gamma + b * inp + c */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn5, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn5, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn5, CB, CP, tmp_ld, 6, 0, in_dt );
  libxsmm_matrix_eqn_push_back_arg( my_eqn5, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn5, CB, CP, ld, 3, 0, in_dt );
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn5, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn5, CB, CP, ld, 0, 0, in_dt );
  libxsmm_matrix_eqn_push_back_arg( my_eqn5, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn5, 1, 1, 1, 7, 0, LIBXSMM_DATATYPE_F32 );
  func5 = libxsmm_dispatch_matrix_eqn( CB, CP, &ld, in_dt, my_eqn5 );


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

  /* db new equation */
  my_eqn13 = libxsmm_matrix_eqn_create();                                                       /* db [CB] = (dout * gamma) [HW, CB] + db [CB]*/
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn13, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );                  /* db [CB] */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn13, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);   /* [HW, CB] -> [CB] */
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn13, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn13, CB, HW, ld, 3, 0, in_dt );                        /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn13, CB, 1, 1, 6, 0, in_dt );                          /* gamma [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn13, CB, 1, 1, 9, 0, LIBXSMM_DATATYPE_F32 );           /* db [CB] */
  func13 = libxsmm_dispatch_matrix_eqn( CB, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn13 );      /* db [CB] */

  /* ds new equation */
  my_eqn14 = libxsmm_matrix_eqn_create();                                                       /* ds [CB] = ((dout * gamma) * inp) [HW, CB] + ds [CB] */
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn14, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );                  /* ds [CB] */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn14, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);   /* [HW, CB] -> [CB] */
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn14, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn14, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1, LIBXSMM_DATATYPE_F32 );       /*(dout * gamma)*/
  libxsmm_matrix_eqn_push_back_arg( my_eqn14, CB, HW, ld, 3, 0, in_dt );                        /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn14, CB, 1, 1, 6, 0, in_dt );                          /* gamma [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn14, CB, HW, ld, 0, 0, in_dt );                        /* inp [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn14, CB, 1, 1, 8, 0, LIBXSMM_DATATYPE_F32 );           /* ds [CB] */
  func14 = libxsmm_dispatch_matrix_eqn( CB, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn14 );      /* ds [CB] */

  /* /* db old equation */
  /* my_eqn13 = libxsmm_matrix_eqn_create();                                                       /* db += (dout * gamma) */
  /* libxsmm_matrix_eqn_push_back_binary_op( my_eqn13, LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1, LIBXSMM_DATATYPE_F32 ); */
  /* libxsmm_matrix_eqn_push_back_arg( my_eqn13, CB, HW, ld, 3, 0, in_dt );                        /* dout [HW, CB] */
  /* libxsmm_matrix_eqn_push_back_arg( my_eqn13, CB, 1, 1, 6, 0, in_dt );                          /* gamma [CB] */
  /* func13 = libxsmm_dispatch_matrix_eqn( 1, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn13 );       /* db [1] */

  /* /* ds old equation */
  /* my_eqn14 = libxsmm_matrix_eqn_create();                                                       /* ds += ((dout * gamma) * inp) */
  /* libxsmm_matrix_eqn_push_back_binary_op( my_eqn14, LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 ); */
  /* libxsmm_matrix_eqn_push_back_binary_op( my_eqn14, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1, LIBXSMM_DATATYPE_F32 ); */
  /* libxsmm_matrix_eqn_push_back_arg( my_eqn14, CB, HW, ld, 3, 0, in_dt );                        /* dout [HW, CB] */
  /* libxsmm_matrix_eqn_push_back_arg( my_eqn14, CB, 1, 1, 6, 0, in_dt );                          /* gamma [CB] */
  /* libxsmm_matrix_eqn_push_back_arg( my_eqn14, CB, HW, ld, 0, 0, in_dt );                        /* inp [HW, CB] */
  /* func14 = libxsmm_dispatch_matrix_eqn( 1, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn14 );       /* ds [1] */

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
    scaler_groupnorm_bwd_fp32(CP, NB, HW, CB, G, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
    tpp_groupnorm_bwd_fp32(CP, NB, HW, CB, G, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func13, func14, func15, eps);
  } else if (datatype_mode == 1) {
    scaler_groupnorm_bwd_fp32(CP, NB, HW, CB, G, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
    tpp_groupnorm_bwd_bf16(CP, NB, HW, CB, G, bf16_dout, bf16_inp, mean, var, bf16_gamma, bf16_eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func13, func14, func15, eps);
    /* tpp_layernorm_bwd_bf16(CP, NB, CB, bf16_eqn_dout, bf16_inp, mean, var, bf16_gamma, bf16_eqn_dinp, eqn_dgamma, eqn_dbeta, func1, func2, func3, func4, func5); */
    for ( i = 0; i < CP*NB*HW*CB; ++i ) {
      /* dinp[i] = upconvert_bf16(bf16_dinp[i]); */
      eqn_dinp[i] = upconvert_bf16(bf16_eqn_dinp[i]);
    }
  }

  /* compare */
  printf("############################################\n");
  if (datatype_mode == 0) {
    printf("# Correctness FP32 BWD Groupnorm - Dinput  #\n");
  } else {
    printf("# Correctness BF16 BWD Groupnorm - Dinput  #\n");
  }
  printf("############################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, CP*NB*HW*CB, 1, dinp, eqn_dinp, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  printf("###########################################\n");
  if (datatype_mode == 0) {
    printf("# Correctness FP32 BWD Groupnorm - Dbeta  #\n");
  } else {
    printf("# Correctness BF16 BWD Groupnorm - Dbeta  #\n");
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
    printf("# Correctness FP32 BWD Groupnorm - Dgamma  #\n");
  } else {
    printf("# Correctness BF16 BWD Groupnorm - Dgamma #\n");
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
    scaler_groupnorm_bwd_fp32(CP, NB, HW, CB, G, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      scaler_groupnorm_bwd_fp32(CP, NB, HW, CB, G, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("Scaler groupnorm time BWD = %.5g\n", ((double)(l_total)));
    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i] + (float)l_total;
    }
    tpp_groupnorm_bwd_fp32(CP, NB, HW, CB, G, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func13, func14, func15, eps);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      tpp_groupnorm_bwd_fp32(CP, NB, HW, CB, G, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func13, func14, func15, eps);
    }
    l_end = libxsmm_timer_tick();
    l_total2 = libxsmm_timer_duration(l_start, l_end);
    printf("TPP groupnorm time BWD = %.5g\n", ((double)(l_total2)));
    printf("Speedup BWD is %.5g\n", l_total/l_total2);
  } else if (datatype_mode == 1) {
    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i];
    }
    /* tpp_groupnorm_bwd_bf16(CP, NB, HW, CB, G, bf16_dout, bf16_inp, mean, var, bf16_gamma, bf16_dinp, dgamma, dbeta, func11, func12, func13, func14, func15, eps); */
    scaler_groupnorm_bwd_fp32(CP, NB, HW, CB, G, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
    l_start = libxsmm_timer_tick();

    for (it = 0; it < iters; it++) {
      /* tpp_groupnorm_bwd_bf16(CP, NB, HW, CB, G, bf16_dout, bf16_inp, mean, var, bf16_gamma, bf16_dinp, dgamma, dbeta, func11, func12, func13, func14, func15, eps); */
      scaler_groupnorm_bwd_fp32(CP, NB, HW, CB, G, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("Scaler FP32 groupnorm time BWD  = %.5g\n", ((double)(l_total)));

    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i] + (float)l_total;
    }

    tpp_groupnorm_bwd_bf16(CP, NB, HW, CB, G, bf16_dout, bf16_inp, mean, var, bf16_gamma, bf16_dinp, dgamma, dbeta, func11, func12, func13, func14, func15, eps);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      tpp_groupnorm_bwd_bf16(CP, NB, HW, CB, G, bf16_dout, bf16_inp, mean, var, bf16_gamma, bf16_dinp, dgamma, dbeta, func11, func12, func13, func14, func15, eps);
    }
    l_end = libxsmm_timer_tick();
    l_total2 = libxsmm_timer_duration(l_start, l_end);
    printf("TPP BF16 groupnorm time BWD = %.5g\n", ((double)(l_total2)));
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
