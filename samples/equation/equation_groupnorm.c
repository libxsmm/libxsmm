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
#include <omp.h>

#define ALIGNDOWN(N, A) ((N) & ~((A)-1))
#define USE_VECTORIZED_PATH 1


float upconvert_bf16(libxsmm_bfloat16 x) {
  union libxsmm_bfloat16_hp bf16_hp;
  bf16_hp.i[1] = x;
  bf16_hp.i[0] = 0;
  return bf16_hp.f;
}

void tpp_groupnorm_fwd_fp32(long NP, long CP, long HW, long CB, long G, long num_HW_blocks, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout, float eps,
                            libxsmm_matrix_eqn_function func10, libxsmm_matrix_eqn_function func00, libxsmm_meltwfunction_unary reduce_HW_kernel, libxsmm_meltwfunction_unary reduce_G_HW_kernel, libxsmm_meltwfunction_unary reduce_rows_kernel,
                            libxsmm_meltwfunction_unary reduce_groups_kernel, libxsmm_meltwfunction_unary all_zero_G_kernel) {


  LIBXSMM_VLA_DECL(4, float, inp, pinp, CP, HW, CB);            /* [NP, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, float, out, pout, CP, HW, CB);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2, float, beta, pbeta, CB);

  int np, group_size;
  group_size = (CP*CB)/G;

  #pragma omp parallel for
  for(np = 0; np < NP; np++){

    LIBXSMM_ALIGNED(float tmp[2*CB], 64);
    LIBXSMM_ALIGNED(float sum_X[G], 64);
    LIBXSMM_ALIGNED(float sum_X2[G], 64);
    LIBXSMM_ALIGNED(float s[CP*CB], 64);
    LIBXSMM_ALIGNED(float b[CP*CB], 64);

    int i, j, cp, cb, hwb, g;
    float m, v;
    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_meltw_unary_param m_reduce_rows_params, m_reduce_groups_params, v_reduce_rows_params, v_reduce_groups_params, reduce_HW_params, reduce_G_HW_params;
    libxsmm_meltw_unary_param all_zero_param;
    libxsmm_matrix_arg arg_array[5];

    all_zero_param.out.primary = sum_X;
    all_zero_G_kernel(&all_zero_param);
    all_zero_param.out.primary = sum_X2;
    all_zero_G_kernel(&all_zero_param);

    if (group_size > CB){                                 /* Group size >= block size  (Ex.- CP = 4, CB = 16, G = 2, group_size = 32) */
      LIBXSMM_ALIGNED(float new_tmp[2*CB], 64);
      for (cp = 0; cp < CP; cp++){                      /* [cp, HW, CB] */
        for (cb = 0; cb < 2*CB; cb++) {
          tmp[cb] = 0.0f;
        }

        reduce_HW_params.out.primary   = new_tmp;                  /* [2*CB] */
        for(hwb=0; hwb < num_HW_blocks; hwb++){
          reduce_HW_params.in.primary    = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);      /* [HW, CB] -----> [2 * CB] */
          reduce_HW_kernel(&reduce_HW_params);

          #pragma omp simd
          for (cb = 0; cb < 2*CB; cb++) {
            tmp[cb] += new_tmp[cb];
          }
        }

        // if (group_size >= CB){                                 /* Group size >= block size  (Ex.- CP = 4, CB = 16, G = 2, group_size = 32) */
          g = (cp*CB)/group_size;                              /* determine current group */
          m_reduce_rows_params.in.primary    = tmp;
          m_reduce_rows_params.out.primary   = &m;
          v_reduce_rows_params.in.primary    = &tmp[CB];
          v_reduce_rows_params.out.primary   = &v;
          reduce_rows_kernel(&m_reduce_rows_params);
          reduce_rows_kernel(&v_reduce_rows_params);
          sum_X[g] += m;
          sum_X2[g] += v;
        // }
        // else{                                                 /* Group size < block size  (Ex.- CP = 4, CB = 16, G = 32, group_size = 2) */
        //   for(i=0; i < CB; i += group_size){
        //     m_reduce_groups_params.in.primary    = &tmp[i];
        //     m_reduce_groups_params.out.primary   = &sum_X[cp*(CB/group_size) + (i/group_size)];
        //     v_reduce_groups_params.in.primary    = &tmp[CB + i];
        //     v_reduce_groups_params.out.primary   = &sum_X2[cp*(CB/group_size) + (i/group_size)];
        //     reduce_groups_kernel(&m_reduce_groups_params);
        //     reduce_groups_kernel(&v_reduce_groups_params);
        //   }
        // }
      }

      for(g = 0; g < G; g++){                                                  /* mean and variance calculation */
        mean[np*G + g] = sum_X[g] / ((float)group_size * HW);
        var[np*G + g] = (sum_X2[g] / ((float)group_size * HW)) - (mean[np*G + g]*mean[np*G + g]);        /* var = E[X^2] - (E[X])^2 */

        for(j = 0; j < group_size; j++){
          s[g*group_size + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));                               /* 1/sqrt(var(X) + eps) */
          b[g*group_size + j] = -1 * mean[np*G + g] * s[g*group_size + j];                               /* -E[X]/sqrt(var(X) + eps) */
        }
      }

      for (cp = 0; cp < CP; cp++){

        arg_array[1].primary = &s[cp*CB];                                                                   /* [CB] */
        arg_array[2].primary = &b[cp*CB];                                                                   /* [CB] */
        arg_array[3].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);                                    /* [CB] */
        arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, beta, cp, 0, CB);                                     /* [CB] */

        for(hwb=0; hwb < num_HW_blocks; hwb++){
          arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);                       /* [HW, CB] */
          eqn_param.inputs = arg_array;
          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, out, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);                   /* [HW,CB] */
          func10(&eqn_param);                                                                                 /* Normalization equation -> y = ((s*x + b)*gamma + beta) */
        }
      }
    }

    else{                                 /* Group size < block size  (Ex.- CP = 4, CB = 16, G = 32, group_size = 2) */
      LIBXSMM_ALIGNED(float new_tmp[2*CB], 64);
      for (cp = 0; cp < CP; cp++){                      /* [cp, HW, CB] */
        // for(i=0; i < CB; i += group_size){              /* group loop */
          for (cb = 0; cb < 2*CB; cb++) {
            tmp[cb] = 0.0f;
          }

          reduce_HW_params.out.primary   = new_tmp;                  /* [2*group_size] */
          for(hwb=0; hwb < num_HW_blocks; hwb++){
            reduce_HW_params.in.primary    = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);      /* [HW_block, group_size] -----> [2 * group_size] */
            reduce_HW_kernel(&reduce_HW_params);

            for (cb = 0; cb < 2*CB; cb++) {
              tmp[cb] += new_tmp[cb];
            }
          }

          for(i=0; i < CB; i += group_size){
            g = (cp*CB + i)/group_size;                              /* determine current group */
            for(j = 0; j < group_size; j++){
              sum_X[g] += tmp[i + j];
              sum_X2[g] += tmp[CB + i + j];
            }
            mean[np*G + g] = sum_X[g] / ((float)group_size * HW);
            var[np*G + g] = (sum_X2[g] / ((float)group_size * HW)) - (mean[np*G + g]*mean[np*G + g]);        /* var = E[X^2] - (E[X])^2 */

            for(j = 0; j < group_size; j++){
              s[g*group_size + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));                               /* 1/sqrt(var(X) + eps) */
              b[g*group_size + j] = -1 * mean[np*G + g] * s[g*group_size + j];                               /* -E[X]/sqrt(var(X) + eps) */
            }
          }

          arg_array[1].primary = &s[cp*CB];                                                                   /* [group_size] */
          arg_array[2].primary = &b[cp*CB];                                                                   /* [group_size] */
          arg_array[3].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);                                    /* [group_size] */
          arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, beta, cp, 0, CB);                                     /* [group_size] */

          for(hwb=0; hwb < num_HW_blocks; hwb++){
            arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);                       /* [HW, group_size] */
            eqn_param.inputs = arg_array;
            eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, out, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);                   /* [HW, group_size] */
            func10(&eqn_param);                                                                                 /* Normalization equation -> y = ((s*x + b)*gamma + beta) */
          }
        // }
      }
    }
  }
}

void tpp_groupnorm_fwd_bf16(long NP, long CP, long HW, long CB, long G, long num_HW_blocks, libxsmm_bfloat16 *pinp, libxsmm_bfloat16 *pgamma, libxsmm_bfloat16 *pbeta, float *mean, float *var,
                            libxsmm_bfloat16 *pout, float eps, libxsmm_matrix_eqn_function func10, libxsmm_meltwfunction_unary reduce_HW_kernel, libxsmm_meltwfunction_unary reduce_rows_kernel,
                            libxsmm_meltwfunction_unary reduce_groups_kernel, libxsmm_meltwfunction_unary all_zero_G_kernel) {


  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, inp, pinp, CP, HW, CB);            /* [NP, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, out, pout, CP, HW, CB);
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, beta, pbeta, CB);

  int np, group_size;
  group_size = (CP*CB)/G;

  #pragma omp parallel for
  for(np = 0; np < NP; np++){

    LIBXSMM_ALIGNED(float tmp[2*CB], 64);
    LIBXSMM_ALIGNED(float sum_X[G], 64);
    LIBXSMM_ALIGNED(float sum_X2[G], 64);
    LIBXSMM_ALIGNED(float s[CP*CB], 64);
    LIBXSMM_ALIGNED(float b[CP*CB], 64);

    int i, j, cp, cb, g, hwb;
    float m, v;

    libxsmm_matrix_eqn_param eqn_param;
    libxsmm_meltw_unary_param m_reduce_rows_params, m_reduce_groups_params, v_reduce_rows_params, v_reduce_groups_params, reduce_HW_params;
    libxsmm_meltw_unary_param all_zero_param;
    libxsmm_matrix_arg  arg_array[5];

    all_zero_param.out.primary = sum_X;
    all_zero_G_kernel(&all_zero_param);
    all_zero_param.out.primary = sum_X2;
    all_zero_G_kernel(&all_zero_param);

    LIBXSMM_ALIGNED(float new_tmp[2*CB], 64);
    for (cp = 0; cp < CP; cp++){                      /* [cp, HW, CB] */
      #pragma omp simd
      for (cb = 0; cb < 2*CB; cb++) {
        tmp[cb] = 0.0f;
      }

      reduce_HW_params.out.primary   = new_tmp;                  /* [2*CB] */
      for(hwb=0; hwb < num_HW_blocks; hwb++){
        reduce_HW_params.in.primary    = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);      /* [HW, CB] -----> [2 * CB] */
        reduce_HW_kernel(&reduce_HW_params);

        #pragma omp simd
        for (cb = 0; cb < 2*CB; cb++) {
          tmp[cb] += new_tmp[cb];
        }
      }

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
        for(i=0; i < CB; i += group_size){
          m_reduce_groups_params.in.primary    = &tmp[i];
          m_reduce_groups_params.out.primary   = &sum_X[cp*(CB/group_size) + (i/group_size)];
          v_reduce_groups_params.in.primary    = &tmp[CB + i];
          v_reduce_groups_params.out.primary   = &sum_X2[cp*(CB/group_size) + (i/group_size)];
          reduce_groups_kernel(&m_reduce_groups_params);
          reduce_groups_kernel(&v_reduce_groups_params);
        }

        for(g = 0; g < G; g++){                                                  /* mean and variance calculation */
          mean[np*G + g] = sum_X[g] / ((float)group_size * HW);
          var[np*G + g] = (sum_X2[g] / ((float)group_size * HW)) - (mean[np*G + g]*mean[np*G + g]);        /* var = E[X^2] - (E[X])^2 */

          for(j = 0; j < group_size; j++){
            s[g*group_size + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));                               /* 1/sqrt(var(X) + eps) */
            b[g*group_size + j] = -1 * mean[np*G + g] * s[g*group_size + j];                               /* -E[X]/sqrt(var(X) + eps) */
          }
        }

        arg_array[1].primary = &s[cp*CB];                                                                   /* [CB] */
        arg_array[2].primary = &b[cp*CB];                                                                   /* [CB] */
        arg_array[3].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);                                    /* [CB] */
        arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, beta, cp, 0, CB);                                     /* [CB] */

        for(hwb=0; hwb < num_HW_blocks; hwb++){
          arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);                       /* [HW, CB] */
          eqn_param.inputs = arg_array;
          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, out, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);                   /* [HW,CB] */
          func10(&eqn_param);                                                                                 /* Normalization equation -> y = ((s*x + b)*gamma + beta) */
        }
      }
    }

    if (group_size >= CB){                                 /* Group size >= block size  (Ex.- CP = 4, CB = 16, G = 2, group_size = 32) */
      for(g = 0; g < G; g++){                                                  /* mean and variance calculation */
        mean[np*G + g] = sum_X[g] / ((float)group_size * HW);
        var[np*G + g] = (sum_X2[g] / ((float)group_size * HW)) - (mean[np*G + g]*mean[np*G + g]);        /* var = E[X^2] - (E[X])^2 */

        for(j = 0; j < group_size; j++){
          s[g*group_size + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));                               /* 1/sqrt(var(X) + eps) */
          b[g*group_size + j] = -1 * mean[np*G + g] * s[g*group_size + j];                               /* -E[X]/sqrt(var(X) + eps) */
        }
      }

      for (cp = 0; cp < CP; cp++){

        arg_array[1].primary = &s[cp*CB];                                                                   /* [CB] */
        arg_array[2].primary = &b[cp*CB];                                                                   /* [CB] */
        arg_array[3].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);                                    /* [CB] */
        arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, beta, cp, 0, CB);                                     /* [CB] */

        for(hwb=0; hwb < num_HW_blocks; hwb++){
          arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);                       /* [HW, CB] */
          eqn_param.inputs = arg_array;
          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, out, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);                   /* [HW,CB] */
          func10(&eqn_param);                                                                                 /* Normalization equation -> y = ((s*x + b)*gamma + beta) */
        }
      }
    }
  }
}

void tpp_groupnorm_bwd_fp32(long NP, long CP, long HW, long CB, long G, long num_HW_blocks, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta,
    libxsmm_matrix_eqn_function dgamma_func, libxsmm_matrix_eqn_function dbeta_func, libxsmm_matrix_eqn_function db_func, libxsmm_matrix_eqn_function ds_func, libxsmm_matrix_eqn_function din_func, float eps) {

  int group_size;
  group_size = (CP*CB)/G;

  const float scale = 1.0f / ((float)CP*HW*CB);

  LIBXSMM_VLA_DECL(4, float, din, pdin, CP, HW, CB);
  LIBXSMM_VLA_DECL(4, float, inp, pinp, CP, HW, CB);
  LIBXSMM_VLA_DECL(4, float, dout, pdout, CP, HW, CB);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, CB);

  LIBXSMM_ALIGNED(float dgamma_NP[NP*CP*CB], 64);
  LIBXSMM_ALIGNED(float dbeta_NP[NP*CP*CB], 64);

  #pragma omp parallel
  {
    LIBXSMM_ALIGNED(float a[CP*CB], 64);
    LIBXSMM_ALIGNED(float b[CP*CB], 64);
    LIBXSMM_ALIGNED(float c[CP*CB], 64);
    LIBXSMM_ALIGNED(float ds[CP*CB], 64);
    LIBXSMM_ALIGNED(float db[CP*CB], 64);
    int np;

    #pragma omp for
    for (np = 0; np < NP; np++) {
      int j, g, cp, hwb;

      for(j = 0; j < CP*CB; j++){
        dgamma_NP[np*CP*CB + j] = 0.0f;
        dbeta_NP[np*CP*CB + j] = 0.0f;
      }

      libxsmm_matrix_eqn_param eqn_param;
      libxsmm_matrix_arg arg_array[10];
      eqn_param.inputs = arg_array;

      for(g = 0; g < G; g++){                                                  /* compute a and b for each channel from group means and variance */
        for(j = 0; j < group_size; j++){
          a[g*group_size + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));
          b[g*group_size + j] = -a[g*group_size + j]*mean[np*G + + g];
          ds[g*group_size + j] = 0.0f;
          db[g*group_size + j] = 0.0f;
        }
      }
      for (cp = 0; cp < CP; cp++) {
        arg_array[1].primary = &a[cp*CB];
        arg_array[2].primary = &b[cp*CB];
        arg_array[4].primary = &dgamma_NP[np*CP*CB + cp*CB];
        arg_array[5].primary = &dbeta_NP[np*CP*CB + cp*CB];
        /* arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, CB); */
        /* arg_array[5].primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, CB); */
        arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
        /* arg_array[7].primary = &c[cp*CB]; */
        arg_array[8].primary = &ds[cp*CB];
        arg_array[9].primary = &db[cp*CB];

        for(hwb=0; hwb < num_HW_blocks; hwb++){

          arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);

          eqn_param.output.primary = &ds[cp*CB];
          ds_func(&eqn_param);

          eqn_param.output.primary = &db[cp*CB];
          db_func(&eqn_param);

          // eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, CB);
          eqn_param.output.primary = &dgamma_NP[np*CP*CB + cp*CB];
          dgamma_func(&eqn_param);

          // eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, CB);
          eqn_param.output.primary = &dbeta_NP[np*CP*CB + cp*CB];
          dbeta_func(&eqn_param);
        }
      }

      /* b = (db * mean[nb] - ds) * a * a * a * scale; */
      /* c = -b * mean[nb] - db * a * scale; */

      for(g = 0; g < G; g++){                                                  /* compute b and c for each channel from group means and variance */
        float gds = 0.0f;
        float gdb = 0.0f;
        for(j = 0; j < group_size; j++){
          gds += ds[g*group_size + j];                                        /* Group ds and db calculation */
          gdb += db[g*group_size + j];
        }
        for(j = 0; j < group_size; j++){
          b[g*group_size + j] = (gdb * mean[np*G + g] - gds) * a[g*group_size + j] * a[g*group_size + j] * a[g*group_size + j] * scale;
          c[g*group_size + j] = -b[g*group_size + j] * mean[np*G + g] - gdb * a[g*group_size + j] * scale;
        }
      }

      for (cp = 0; cp < CP; cp++) {

        arg_array[1].primary = &a[cp*CB];
        arg_array[2].primary = &b[cp*CB];

        /* arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, CB); */
        /* arg_array[5].primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, CB); */
        arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
        arg_array[7].primary = &c[cp*CB];
        /* arg_array[8].primary = &ds[cp*CB]; */
        /* arg_array[9].primary = &db[cp*CB]; */
        for(hwb=0; hwb < num_HW_blocks; hwb++){
          arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, din, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          din_func(&eqn_param);
        }
      }
    }

    int cp;
    #pragma omp for
    for (cp = 0; cp < CP; cp++) {
      for (np=0; np < NP; np++ ) {
        int cb;
        for(cb = 0; cb < CB; cb++){
          LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, CB) += dgamma_NP[np*CP*CB + cp*CB + cb];
          LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, CB) += dbeta_NP[np*CP*CB + cp*CB + cb];
        }
      }
    }
  }
}

void tpp_groupnorm_bwd_bf16(long NP, long CP, long HW, long CB, long G, long num_HW_blocks, libxsmm_bfloat16 *pdout, libxsmm_bfloat16 *pinp, float *mean, float *var, libxsmm_bfloat16 *pgamma, libxsmm_bfloat16 *pdin, float *pdgamma, float *pdbeta,
    libxsmm_matrix_eqn_function dgamma_func, libxsmm_matrix_eqn_function dbeta_func, libxsmm_matrix_eqn_function db_func, libxsmm_matrix_eqn_function ds_func, libxsmm_matrix_eqn_function din_func, float eps) {

  int group_size;
  group_size = (CP*CB)/G;

  const float scale = 1.0f / ((float)CP*HW*CB);

  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, din, pdin, CP, HW, CB);
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, inp, pinp, CP, HW, CB);
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, dout, pdout, CP, HW, CB);
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, CB);

  LIBXSMM_ALIGNED(float dgamma_NP[NP*CP*CB], 64);
  LIBXSMM_ALIGNED(float dbeta_NP[NP*CP*CB], 64);

  #pragma omp parallel
  {
    LIBXSMM_ALIGNED(float a[CP*CB], 64);
    LIBXSMM_ALIGNED(float b[CP*CB], 64);
    LIBXSMM_ALIGNED(float c[CP*CB], 64);
    LIBXSMM_ALIGNED(float ds[CP*CB], 64);
    LIBXSMM_ALIGNED(float db[CP*CB], 64);
    int np;

    #pragma omp for
    for (np = 0; np < NP; np++) {
      int j, g, cp, hwb;

      for(j = 0; j < CP*CB; j++){
        dgamma_NP[np*CP*CB + j] = 0.0f;
        dbeta_NP[np*CP*CB + j] = 0.0f;
      }

      libxsmm_matrix_eqn_param eqn_param;
      libxsmm_matrix_arg arg_array[10];
      eqn_param.inputs = arg_array;

      for(g = 0; g < G; g++){                                                  /* compute a and b for each channel from group means and variance */
        for(j = 0; j < group_size; j++){
          a[g*group_size + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));
          b[g*group_size + j] = -a[g*group_size + j]*mean[np*G + g];
          ds[g*group_size + j] = 0.0f;
          db[g*group_size + j] = 0.0f;
        }
      }
      for (cp = 0; cp < CP; cp++) {
        arg_array[1].primary = &a[cp*CB];
        arg_array[2].primary = &b[cp*CB];
        arg_array[4].primary = &dgamma_NP[np*CP*CB + cp*CB];
        arg_array[5].primary = &dbeta_NP[np*CP*CB + cp*CB];
        arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
        arg_array[8].primary = &ds[cp*CB];
        arg_array[9].primary = &db[cp*CB];

        for(hwb=0; hwb < num_HW_blocks; hwb++){
          arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          eqn_param.output.primary = &ds[cp*CB];
          ds_func(&eqn_param);

          eqn_param.output.primary = &db[cp*CB];
          db_func(&eqn_param);

          eqn_param.output.primary = &dgamma_NP[np*CP*CB + cp*CB];
          dgamma_func(&eqn_param);

          eqn_param.output.primary = &dbeta_NP[np*CP*CB + cp*CB];
          dbeta_func(&eqn_param);
        }
      }

      /* b = (db * mean[nb] - ds) * a * a * a * scale; */
      /* c = -b * mean[nb] - db * a * scale; */

      for(g = 0; g < G; g++){                                                  /* compute b and c for each channel from group means and variance */
        float gds = 0.0f;
        float gdb = 0.0f;
        for(j = 0; j < group_size; j++){
          gds += ds[g*group_size + j];                                        /* Group ds and db calculation */
          gdb += db[g*group_size + j];
        }
        for(j = 0; j < group_size; j++){
          b[g*group_size + j] = (gdb * mean[np*G + g] - gds) * a[g*group_size + j] * a[g*group_size + j] * a[g*group_size + j] * scale;
          c[g*group_size + j] = -b[g*group_size + j] * mean[np*G + g] - gdb * a[g*group_size + j] * scale;
        }
      }

      for (cp = 0; cp < CP; cp++) {

        arg_array[1].primary = &a[cp*CB];
        arg_array[2].primary = &b[cp*CB];
        arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
        arg_array[7].primary = &c[cp*CB];

        for(hwb=0; hwb < num_HW_blocks; hwb++){
          arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, din, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          din_func(&eqn_param);
        }
      }
    }

    int cp;
    #pragma omp for
    for (cp = 0; cp < CP; cp++) {
      for (np=0; np < NP; np++ ) {
        int cb;
        for(cb = 0; cb < CB; cb++){
          LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, CB) += dgamma_NP[np*CP*CB + cp*CB + cb];
          LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, CB) += dbeta_NP[np*CP*CB + cp*CB + cb];
        }
      }
    }
  }
}

void scaler_groupnorm_fwd_fp32(long NP, long CP, long HW, long CB, long G, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout, float eps){

  LIBXSMM_VLA_DECL(4, float, inp, pinp, CP, HW, CB);            /* [NP, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, float, out, pout, CP, HW, CB);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2, float, beta, pbeta, CB);

  int np, group_size;
  group_size = (CP*CB)/G;

  #pragma omp parallel for
  for(np = 0; np < NP; np++){

    LIBXSMM_ALIGNED(float sum_X[G], 64);
    LIBXSMM_ALIGNED(float sum_X2[G], 64);
    LIBXSMM_ALIGNED(float s[CP*CB], 64);
    LIBXSMM_ALIGNED(float b[CP*CB], 64);

    int i, j, cp, cb, hw, g;
    float m, v, value;

    for(g = 0; g < G; g++){
      sum_X[g] = 0.0f;
      sum_X2[g] = 0.0f;
    }
    for(cp = 0; cp < CP; cp++){                           /* Size = CP*HW*CB*4 */
      m = 0.0f;
      v = 0.0f;
      if (group_size >= CB){                                 /* Group size >= block size  (Ex.- CP = 4, CB = 16, G = 2, group_size = 32) */
        for(cb = 0; cb < CB; cb++){
          for(hw = 0; hw < HW; hw++){
            value = LIBXSMM_VLA_ACCESS(4, inp, np, cp, hw, cb, CP, HW, CB);
            m += value;
            v += (value*value);
          }
        }
        g = (cp*CB)/group_size;                              /* determine current group */
        sum_X[g] += m;
        sum_X2[g] += v;
      }
      else{
        for(i=0; i < CB; i += group_size){              /* Group size < block size  (Ex.- CP = 4, CB = 16, G = 32, group_size = 2) */
          for(j = 0; j < group_size; j++){
            for(hw = 0; hw < HW; hw++){
              value = LIBXSMM_VLA_ACCESS(4, inp, np, cp, hw, (i + j), CP, HW, CB);
              sum_X[cp*(CB/group_size) + (i/group_size)] += value;
              sum_X2[cp*(CB/group_size) + (i/group_size)] += (value*value);
            }
          }
        }
      }
    }

    for(g = 0; g < G; g++){                                                  /* mean and variance calculation */           /* Size = 2*CP*CB*4 */
      mean[np*G + g] = sum_X[g] / ((float)group_size * HW);
      var[np*G + g] = (sum_X2[g] / ((float)group_size * HW)) - (mean[np*G + g]*mean[np*G + g]);      /* var = E[X^2] - (E[X])^2        [G] */

      for(j = 0; j < group_size; j++){
        s[g*group_size + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));                               /* s = 1/sqrt(var(X) + eps)     [CP, CB] */
        b[g*group_size + j] = -1 * mean[np*G + g] * s[g*group_size + j];                               /* b = -E[X]/sqrt(var(X) + eps) [CP, CB] */
      }
    }

    for(cp = 0; cp < CP; cp++){                                                     /* Size = 2*CP*HW*CB*4 + 2*CP*CB*4 */
      for(cb = 0; cb < CB; cb++){
        for(hw = 0; hw < HW; hw++){
          value = LIBXSMM_VLA_ACCESS(4, inp, np, cp, hw, cb, CP, HW, CB);
          value = ((value * s[cp*CB + cb]) + b[cp*CB + cb]) * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB) + LIBXSMM_VLA_ACCESS(2, beta, cp, cb, CB);        /* Normalization equation -> y = ((s*x + b)*gamma + beta) */
          LIBXSMM_VLA_ACCESS(4, out, np, cp, hw, cb, CP, HW, CB) = value;
        }
      }
    }
  }                                         /*End multithreading loop*/
}

void scaler_groupnorm_bwd_fp32(long NP, long CP, long HW, long CB, long G, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta, float eps) {

  int np, group_size;
  group_size = (CP*CB)/G;
  float scale = 1.0f / (CP * HW* CB);

  LIBXSMM_VLA_DECL(4, float, din, pdin, CP, HW, CB);
  LIBXSMM_VLA_DECL(4, float, inp, pinp, CP, HW, CB);
  LIBXSMM_VLA_DECL(4, float, dout, pdout, CP, HW, CB);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, CB);

  LIBXSMM_ALIGNED(float dgamma_NP[NP*CP*CB], 64);
  LIBXSMM_ALIGNED(float dbeta_NP[NP*CP*CB], 64);

  #pragma omp parallel for
  for(np = 0; np < NP; np++){

    int j, cp, cb, hw, g;
    LIBXSMM_ALIGNED(float a[CP*CB], 64);
    LIBXSMM_ALIGNED(float b[CP*CB], 64);
    LIBXSMM_ALIGNED(float c[CP*CB], 64);
    LIBXSMM_ALIGNED(float ds[CP*CB], 64);
    LIBXSMM_ALIGNED(float db[CP*CB], 64);

    for(j = 0; j < CP*CB; j++){
      dgamma_NP[np*CP*CB + j] = 0.0f;
      dbeta_NP[np*CP*CB + j] = 0.0f;
    }

    for(g = 0; g < G; g++){                                                  /* compute a and b for each channel from group means and variance */
      for(j = 0; j < group_size; j++){
        a[g*group_size + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));
        b[g*group_size + j] = -a[g*group_size + j]*mean[np*G + g];
        ds[g*group_size + j] = 0.0f;
        db[g*group_size + j] = 0.0f;
      }
    }

    for (cp = 0; cp < CP; cp++) {                    /* dgamma += (a * inp + b) * dout , dbeta += dout, ds += dout * gamma * inp, db += dout * gamma */    /* Size = 2*CP*HW*CB*4 */
      for (cb = 0; cb < CB; cb++) {
        for (hw = 0; hw < HW; hw++){
          dgamma_NP[np*CP*CB + cp*CB + cb] += (a[cp*CB + cb] * LIBXSMM_VLA_ACCESS(4, inp, np, cp, hw, cb, CP, HW, CB) + b[cp*CB + cb]) * LIBXSMM_VLA_ACCESS(4, dout, np, cp, hw, cb, CP, HW, CB);
          dbeta_NP[np*CP*CB + cp*CB + cb] += LIBXSMM_VLA_ACCESS(4, dout, np, cp, hw, cb, CP, HW, CB);
          ds[cp*CB + cb] += LIBXSMM_VLA_ACCESS(4, dout, np, cp, hw, cb, CP, HW, CB) * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB) * LIBXSMM_VLA_ACCESS(4, inp, np, cp, hw, cb, CP, HW, CB);
          db[cp*CB + cb] += LIBXSMM_VLA_ACCESS(4, dout, np, cp, hw, cb, CP, HW, CB) * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB);
        }
      }
    }
    /* b = (db * mean[nb] - ds) * a * a * a * scale; */
    /* c = -b * mean[nb] - db * a * scale; */
    for(g = 0; g < G; g++){                                                  /* compute b and c for each channel from group means and variance */
      float gds = 0.0f;
      float gdb = 0.0f;
      for(j = 0; j < group_size; j++){
        gds += ds[g*group_size + j];                                        /* Group ds and db calculation */
        gdb += db[g*group_size + j];
      }
      for(j = 0; j < group_size; j++){
        b[g*group_size + j] = (gdb * mean[np*G + g] - gds) * a[g*group_size + j] * a[g*group_size + j] * a[g*group_size + j] * scale;
        c[g*group_size + j] = -b[g*group_size + j] * mean[np*G + g] - gdb * a[g*group_size + j] * scale;
      }
    }

    for (cp = 0; cp < CP; cp++) {                                                     /* din = dout * a * gamma + b * inp + c */  /* Size = 3*CP*HW*CB*4 */
      for (cb = 0; cb < CB; cb++) {
        for (hw = 0; hw < HW; hw++){
          LIBXSMM_VLA_ACCESS(4, din, np, cp, hw, cb, CP, HW, CB) = LIBXSMM_VLA_ACCESS(4, dout, np, cp, hw, cb, CP, HW, CB)  * a[cp*CB + cb] * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB) + b[cp*CB + cb] * LIBXSMM_VLA_ACCESS(4, inp, np, cp, hw, cb, CP, HW, CB) + c[cp*CB + cb];
        }
      }
    }
  }

  int cp;
  #pragma omp parallel for
  for (cp = 0; cp < CP; cp++) {
    for (np=0; np < NP; np++ ) {
      int cb;
      for(cb = 0; cb < CB; cb++){
        LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, CB) += dgamma_NP[np*CP*CB + cp*CB + cb];
        LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, CB) += dbeta_NP[np*CP*CB + cp*CB + cb];
      }
    }
  }
}


int main( int argc, char* argv[] ) {
  libxsmm_blasint my_eqn00, my_eqn10, my_eqn11, my_eqn12, my_eqn13, my_eqn14, my_eqn15;
  libxsmm_matrix_eqn_function func00, func10, func11, func12, func13, func14, func15;
  libxsmm_meltw_unary_flags jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type  unary_type;
  libxsmm_meltwfunction_unary reduce_rows_kernel, reduce_HW_kernel, reduce_G_HW_kernel, reduce_groups_kernel;

  const float eps = FLT_EPSILON;
  libxsmm_blasint i, it, ld, tmp_ld, tmp_ld2;
  unsigned long long l_start, l_end;
  double l_total = 0, l_total2 = 0;
  double t_vec = 0, t_tpp = 0;
  libxsmm_matdiff_info norms_out;
  float *inp, *out, *dinp, *dout, *eqn_dinp, *eqn_dout, *dbeta, *eqn_dbeta, *dgamma, *eqn_dgamma, *eqn_out, *gamma, *beta, *cache_fl, *mean, *var, sum = 0.0;
  libxsmm_bfloat16 *bf16_inp, *bf16_out, *bf16_dinp, *bf16_dout, *bf16_eqn_dinp, *bf16_eqn_dout, *bf16_gamma, *bf16_beta, *bf16_eqn_out;
  int NP = 28;
  int CP = 2;
  int HW = 784;
  int CB = 64;
  int G = 1;
  long num_HW_blocks = 16;
  int datatype_mode = 0;
  int iters = 100;
  libxsmm_datatype  in_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype  out_dt = LIBXSMM_DATATYPE_F32;

  if ( argc > 1 ) NP = atoi(argv[1]);
  if ( argc > 2 ) CP = atoi(argv[2]);
  if ( argc > 4 ) HW = atoi(argv[3]);
  if ( argc > 5 ) CB = atoi(argv[4]);
  if ( argc > 6 ) G = atoi(argv[5]);
  if ( argc > 7 ) num_HW_blocks = atoi(argv[6]);
  if ( argc > 8 ) datatype_mode = atoi(argv[7]);
  if ( argc > 9 ) iters = atoi(argv[8]);

  if (datatype_mode == 0) {
    in_dt = LIBXSMM_DATATYPE_F32;
    out_dt = LIBXSMM_DATATYPE_F32;
  } else if (datatype_mode == 1) {
    in_dt = LIBXSMM_DATATYPE_BF16;
    out_dt = LIBXSMM_DATATYPE_BF16;
  } else {
    printf("ERROR: Supporting only FP32 and BF16 precisions...\n");
  }

  inp = (float*) libxsmm_aligned_malloc( sizeof(float)*NP*CP*HW*CB,   2097152);
  out = (float*) libxsmm_aligned_malloc( sizeof(float)*NP*CP*HW*CB,   2097152);
  dinp = (float*) libxsmm_aligned_malloc( sizeof(float)*NP*CP*HW*CB,   2097152);
  dout = (float*) libxsmm_aligned_malloc( sizeof(float)*NP*CP*HW*CB,   2097152);
  dgamma = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  dbeta = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  eqn_dinp = (float*) libxsmm_aligned_malloc( sizeof(float)*NP*CP*HW*CB,   2097152);
  eqn_dout = (float*) libxsmm_aligned_malloc( sizeof(float)*NP*CP*HW*CB,   2097152);
  eqn_dgamma = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  eqn_dbeta = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  gamma = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  beta = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  mean = (float*) libxsmm_aligned_malloc( sizeof(float)*NP*G,   2097152);
  var = (float*) libxsmm_aligned_malloc( sizeof(float)*NP*G,   2097152);
  eqn_out  = (float*) libxsmm_aligned_malloc( sizeof(float)*NP*CP*HW*CB,   2097152);
  cache_fl  = (float*) libxsmm_aligned_malloc( sizeof(float)*1024*1024,   2097152);

  bf16_inp = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*NP*CP*HW*CB,   2097152);
  bf16_out = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*NP*CP*HW*CB,   2097152);
  bf16_dinp = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*NP*CP*HW*CB,   2097152);
  bf16_dout = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*NP*CP*HW*CB,   2097152);
  bf16_eqn_dinp = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*NP*CP*HW*CB,   2097152);
  bf16_eqn_dout = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*NP*CP*HW*CB,   2097152);
  bf16_gamma = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*CP*CB,   2097152);
  bf16_beta = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*CP*CB,   2097152);
  bf16_eqn_out  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*NP*CP*HW*CB,   2097152);

  libxsmm_init();
  libxsmm_matdiff_clear(&norms_out);

  /* Initializing arrays */
  for ( i = 0; i < NP*CP*HW*CB; ++i ) {
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


  libxsmm_blasint ldo = G;
  libxsmm_meltwfunction_unary all_zero_G_kernel = libxsmm_dispatch_meltw_unary(G, 1, NULL, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  if ( all_zero_G_kernel == NULL) {
    fprintf( stderr, "JIT for initialization by unary all zero group copy kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* TPPs for reducing X and X2 in HW*/
  ld = CB;
  tmp_ld = CB;

  unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD;
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  reduce_HW_kernel = libxsmm_dispatch_meltw_unary(CB, HW/num_HW_blocks, &ld, &tmp_ld, in_dt, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);

  /* TPPs for reducing X and X2 in groups HW*/
  ld = CB;
  tmp_ld = (CP*CB)/G;

  unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD;
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  reduce_G_HW_kernel = libxsmm_dispatch_meltw_unary((CP*CB)/G, HW/num_HW_blocks, &ld, &tmp_ld, in_dt, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);

  /* TPP for reducing groups */
  libxsmm_blasint group_size = (CP*CB)/G;
  ld = group_size;                /* group_size = (CP*CB)/G */
  tmp_ld = 1;
  unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD;
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS;
  reduce_groups_kernel = libxsmm_dispatch_meltw_unary(group_size, 1, &ld, &tmp_ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);

  ld = CB;
  tmp_ld = 1;
  unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD;
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS;
  reduce_rows_kernel = libxsmm_dispatch_meltw_unary(CB, 1, &ld, &tmp_ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);

  /* TPP for foward */
  ld = CB;
  tmp_ld = 1;
  tmp_ld2 = 1;
  my_eqn10 = libxsmm_matrix_eqn_create();                                                        /* y = (s*x + b)*gamma + beta */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn10, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn10, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, CB, HW/num_HW_blocks, ld, 0, 0, in_dt );                         /* x = [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, CB, 1, tmp_ld, 1, 0, LIBXSMM_DATATYPE_F32 );       /* s = [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, CB, 1, tmp_ld, 2, 0, LIBXSMM_DATATYPE_F32 );       /* b = [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, CB, 1, tmp_ld2, 3, 0, in_dt );                     /* gamma = [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, CB, 1, tmp_ld2, 4, 0, in_dt );                     /* beta = [CB] */
  func10 = libxsmm_dispatch_matrix_eqn( CB, HW/num_HW_blocks, &ld, out_dt, my_eqn10 );                         /* y = [HW, CB] */


  /* TPP for scaling */
  ld = CB;         /* group_size */
  tmp_ld = 1;
  tmp_ld2 = 1;
  my_eqn00 = libxsmm_matrix_eqn_create();                                                        /* y = (s*x + b)*gamma + beta */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn00, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn00, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn00, group_size, HW/num_HW_blocks, ld, 0, 0, in_dt );                         /* x = [HW, group_size] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn00, group_size, 1, tmp_ld, 1, 0, LIBXSMM_DATATYPE_F32 );       /* s = [group_size] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn00, group_size, 1, tmp_ld, 2, 0, LIBXSMM_DATATYPE_F32 );       /* b = [group_size] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn00, group_size, 1, tmp_ld2, 3, 0, in_dt );                     /* gamma = [group_size] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn00, group_size, 1, tmp_ld2, 4, 0, in_dt );                     /* beta = [group_size] */
  func00 = libxsmm_dispatch_matrix_eqn( group_size, HW/num_HW_blocks, &ld, out_dt, my_eqn00 );                         /* y = [HW, group_size] */


  /* Check correctness */
  if (datatype_mode == 0) {
    scaler_groupnorm_fwd_fp32(NP, CP, HW, CB, G, inp, gamma, beta, mean, var, out, eps);
    tpp_groupnorm_fwd_fp32(NP, CP, HW, CB, G, num_HW_blocks, inp, gamma, beta, mean, var, eqn_out, eps, func10, func00, reduce_HW_kernel, reduce_G_HW_kernel, reduce_rows_kernel, reduce_groups_kernel, all_zero_G_kernel);
  } else if (datatype_mode == 1) {
    scaler_groupnorm_fwd_fp32(NP, CP, HW, CB, G, inp, gamma, beta, mean, var, out, eps);
    tpp_groupnorm_fwd_bf16(NP, CP, HW, CB, G, num_HW_blocks, bf16_inp, bf16_gamma, bf16_beta, mean, var, bf16_eqn_out, eps, func10, reduce_HW_kernel, reduce_rows_kernel, reduce_groups_kernel, all_zero_G_kernel);

    for ( i = 0; i < NP*CP*HW*CB; ++i ) {
      /* out[i] = upconvert_bf16(bf16_out[i]); */
      eqn_out[i] = upconvert_bf16(bf16_eqn_out[i]);
    }
  }

  /* compare */
  printf("############################################\n");
  if (datatype_mode == 0) {
    printf("# Correctness FP32 FWD Groupnorm - Output  #\n");
  } else {
    printf("# Correctness BF16 FWD Groupnorm - Output  #\n");
  }
  printf("############################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, NP*CP*HW*CB, 1, out, eqn_out, 0, 0);
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
    scaler_groupnorm_fwd_fp32(NP, CP, HW, CB, G, inp, gamma, beta, mean, var, out, eps);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      scaler_groupnorm_fwd_fp32(NP, CP, HW, CB, G, inp, gamma, beta, mean, var, out, eps);
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("Scaler time FWD  = %.5g\n", ((double)(l_total)));
    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i] + (float)l_total;
    }
    tpp_groupnorm_fwd_fp32(NP, CP, HW, CB, G, num_HW_blocks, inp, gamma, beta, mean, var, eqn_out, eps, func10, func00, reduce_HW_kernel, reduce_G_HW_kernel, reduce_rows_kernel, reduce_groups_kernel, all_zero_G_kernel);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      tpp_groupnorm_fwd_fp32(NP, CP, HW, CB, G, num_HW_blocks, inp, gamma, beta, mean, var, eqn_out, eps, func10, func00, reduce_HW_kernel, reduce_G_HW_kernel, reduce_rows_kernel, reduce_groups_kernel, all_zero_G_kernel);
    }
    l_end = libxsmm_timer_tick();
    l_total2 = libxsmm_timer_duration(l_start, l_end);
    printf("TPP groupnorm time FWD  = %.5g\n", ((double)(l_total2)));
    printf("Speedup FWD is %.5g\n", l_total/l_total2);
  } else if (datatype_mode == 1) {
    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i];
    }

    scaler_groupnorm_fwd_fp32(NP, CP, HW, CB, G, inp, gamma, beta, mean, var, out, eps);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      scaler_groupnorm_fwd_fp32(NP, CP, HW, CB, G, inp, gamma, beta, mean, var, out, eps);
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("Scaler FP32 groupnorm time FWD  = %.5g\n", ((double)(l_total)));

    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i] + (float)l_total;
    }

    tpp_groupnorm_fwd_bf16(NP, CP, HW, CB, G, num_HW_blocks, bf16_inp, bf16_gamma, bf16_beta, mean, var, bf16_eqn_out, eps, func10, reduce_HW_kernel, reduce_rows_kernel, reduce_groups_kernel, all_zero_G_kernel);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      tpp_groupnorm_fwd_bf16(NP, CP, HW, CB, G, num_HW_blocks, bf16_inp, bf16_gamma, bf16_beta, mean, var, bf16_eqn_out, eps, func10, reduce_HW_kernel, reduce_rows_kernel, reduce_groups_kernel, all_zero_G_kernel);
    }
    l_end = libxsmm_timer_tick();
    l_total2 = libxsmm_timer_duration(l_start, l_end);
    printf("TPP BF16 groupnorm time FWD  = %.5g\n", ((double)(l_total2)));
    printf("Speedup FWD is %.5g\n", l_total/l_total2);
  }

  t_tpp = l_total2;
  t_vec = l_total;

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
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, CB, HW/num_HW_blocks, ld, 0, 0, in_dt );                        /* inp [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, CB, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 );           /* a [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, CB, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );           /* b [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, CB, HW/num_HW_blocks, ld, 3, 0, in_dt );                        /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, CB, 1, 1, 4, 0, LIBXSMM_DATATYPE_F32 );           /* dgamma [CB] */
  func11 = libxsmm_dispatch_matrix_eqn( CB, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn11 );      /* dgamma [CB] */

  /* dbeta function  */
  my_eqn12 = libxsmm_matrix_eqn_create();                                                       /* dbeta [CB] = dout [HW, CB] + dbeta [CB] */
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn12, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );      /* dbeta_tmp [HW, CB] */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn12, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);  /* [HW, CB] -> [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn12, CB, HW/num_HW_blocks, ld, 3, 0, in_dt );                        /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn12, CB, 1, 1, 5, 0, LIBXSMM_DATATYPE_F32 );           /* dbeta [CB] */
  func12 = libxsmm_dispatch_matrix_eqn( CB, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn12 );      /* dbeta [CB] */

  /* db new equation */
  my_eqn13 = libxsmm_matrix_eqn_create();                                                       /* db [CB] = (dout * gamma) [HW, CB] + db [CB]*/
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn13, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );                  /* db [CB] */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn13, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);   /* [HW, CB] -> [CB] */
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn13, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn13, CB, HW/num_HW_blocks, ld, 3, 0, in_dt );                        /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn13, CB, 1, 1, 6, 0, in_dt );                          /* gamma [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn13, CB, 1, 1, 9, 0, LIBXSMM_DATATYPE_F32 );           /* db [CB] */
  func13 = libxsmm_dispatch_matrix_eqn( CB, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn13 );      /* db [CB] */

  /* ds new equation */
  my_eqn14 = libxsmm_matrix_eqn_create();                                                       /* ds [CB] = ((dout * gamma) * inp) [HW, CB] + ds [CB] */
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn14, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );                  /* ds [CB] */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn14, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);   /* [HW, CB] -> [CB] */
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn14, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn14, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1, LIBXSMM_DATATYPE_F32 );       /*(dout * gamma)*/
  libxsmm_matrix_eqn_push_back_arg( my_eqn14, CB, HW/num_HW_blocks, ld, 3, 0, in_dt );                        /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn14, CB, 1, 1, 6, 0, in_dt );                          /* gamma [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn14, CB, HW/num_HW_blocks, ld, 0, 0, in_dt );                        /* inp [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn14, CB, 1, 1, 8, 0, LIBXSMM_DATATYPE_F32 );           /* ds [CB] */
  func14 = libxsmm_dispatch_matrix_eqn( CB, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn14 );      /* ds [CB] */

  /* din equation */
  my_eqn15 = libxsmm_matrix_eqn_create();                                                       /* din = ((gamma * a) * dout) + (inp * b + c) */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn15, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn15, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, CB, 1, 1, 6, 0, in_dt );                          /* gamma [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, CB, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 );           /* a [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, CB, HW/num_HW_blocks, ld, 3, 0, in_dt );                        /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn15, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, CB, HW/num_HW_blocks, ld, 0, 0, in_dt );                        /* inp [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, CB, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );           /* b [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, CB, 1, 1, 7, 0, LIBXSMM_DATATYPE_F32 );           /* c [CB] */
  func15 = libxsmm_dispatch_matrix_eqn( CB, HW/num_HW_blocks, &ld, in_dt, my_eqn15 );                         /* din [HW, CB] */

  if (datatype_mode == 0) {
    scaler_groupnorm_bwd_fp32(NP, CP, HW, CB, G, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
    tpp_groupnorm_bwd_fp32(NP, CP, HW, CB, G, num_HW_blocks, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func13, func14, func15, eps);
  } else if (datatype_mode == 1) {
    scaler_groupnorm_bwd_fp32(NP, CP, HW, CB, G, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
    tpp_groupnorm_bwd_bf16(NP, CP, HW, CB, G, num_HW_blocks, bf16_dout, bf16_inp, mean, var, bf16_gamma, bf16_eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func13, func14, func15, eps);

    for ( i = 0; i < NP*CP*HW*CB; ++i ) {
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
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, NP*CP*HW*CB, 1, dinp, eqn_dinp, 0, 0);
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
    scaler_groupnorm_bwd_fp32(NP, CP, HW, CB, G, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      scaler_groupnorm_bwd_fp32(NP, CP, HW, CB, G, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("Scaler groupnorm time BWD = %.5g\n", ((double)(l_total)));
    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i] + (float)l_total;
    }
    tpp_groupnorm_bwd_fp32(NP, CP, HW, CB, G, num_HW_blocks, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func13, func14, func15, eps);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      tpp_groupnorm_bwd_fp32(NP, CP, HW, CB, G, num_HW_blocks, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func13, func14, func15, eps);
    }
    l_end = libxsmm_timer_tick();
    l_total2 = libxsmm_timer_duration(l_start, l_end);
    printf("TPP groupnorm time BWD = %.5g\n", ((double)(l_total2)));
    printf("Speedup BWD is %.5g\n", l_total/l_total2);
  } else if (datatype_mode == 1) {
    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i];
    }

    scaler_groupnorm_bwd_fp32(NP, CP, HW, CB, G, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
    l_start = libxsmm_timer_tick();

    for (it = 0; it < iters; it++) {

      scaler_groupnorm_bwd_fp32(NP, CP, HW, CB, G, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("Scaler FP32 groupnorm time BWD  = %.5g\n", ((double)(l_total)));

    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i] + (float)l_total;
    }

    tpp_groupnorm_bwd_bf16(NP, CP, HW, CB, G, num_HW_blocks, bf16_dout, bf16_inp, mean, var, bf16_gamma, bf16_dinp, dgamma, dbeta, func11, func12, func13, func14, func15, eps);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      tpp_groupnorm_bwd_bf16(NP, CP, HW, CB, G, num_HW_blocks, bf16_dout, bf16_inp, mean, var, bf16_gamma, bf16_dinp, dgamma, dbeta, func11, func12, func13, func14, func15, eps);
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
