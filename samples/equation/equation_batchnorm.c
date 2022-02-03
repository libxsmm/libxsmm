/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <libxsmm_sync.h>
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

void tpp_batchnorm_fwd_bf16(long N, long CP, long HW, long CB, long num_HW_blocks, libxsmm_bfloat16 *pinp, libxsmm_bfloat16 *pgamma, libxsmm_bfloat16 *pbeta, float *mean, float *var,
    libxsmm_bfloat16 *pout, float eps, libxsmm_matrix_eqn_function func10, libxsmm_meltwfunction_unary reduce_HW_kernel, libxsmm_meltwfunction_unary all_zero_kernel, libxsmm_meltwfunction_binary add_kernel, libxsmm_meltwfunction_unary copy_kernel) {

  const float scale = 1.0f / ((float)N*HW);
  LIBXSMM_ALIGNED(float sum_X_X2[2*CP*CB], 64);

  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, inp, pinp, CP, HW, CB);            /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, out, pout, CP, HW, CB);            /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, gamma, pgamma, CB);                /* [CP, CB] */
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, beta, pbeta, CB);                  /* [CP, CB] */

  LIBXSMM_ALIGNED(float sum_N[CP*N*CB], 64);
  LIBXSMM_ALIGNED(float sumsq_N[CP*N*CB], 64);

  #pragma omp parallel
  {

    LIBXSMM_ALIGNED(float s[CB], 64);
    LIBXSMM_ALIGNED(float b[CB], 64);
    int n, cp;

    #pragma omp for nowait collapse(2)
    for (cp = 0; cp < CP; cp++) {
      for(n = 0; n < N; n++){

        int hwb = 0;
        float *sum_ncp_ptr = &sum_N[cp*N*CB + n*CB];
        float *sumsq_ncp_ptr = &sumsq_N[cp*N*CB + n*CB];

        libxsmm_meltw_unary_param all_zero_param;
        all_zero_param.out.primary = sum_ncp_ptr;
        all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = sumsq_ncp_ptr;
        all_zero_kernel(&all_zero_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < CB; cb++) { */
        /*   sum_ncp_ptr[cb] = 0.0f; */
        /*   sumsq_ncp_ptr[cb] = 0.0f; */
        /* } */

        libxsmm_meltw_binary_param add_param;

        libxsmm_meltw_unary_param reduce_HW_params;                           /*Private params and tmp array */
        LIBXSMM_ALIGNED(float lcl_sum_X_X2[2*CB], 64);
        reduce_HW_params.out.primary   = lcl_sum_X_X2;                        /* [2*CB]  */

        for(hwb=0; hwb < num_HW_blocks; hwb++){

          reduce_HW_params.in.primary    = &LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          reduce_HW_kernel(&reduce_HW_params);                                 /* [HW, CB] -----> [2 * CB] */

          add_param.in0.primary = sum_ncp_ptr;
          add_param.in1.primary = lcl_sum_X_X2;
          add_param.out.primary = sum_ncp_ptr;
          add_kernel(&add_param);

          add_param.in0.primary = sumsq_ncp_ptr;
          add_param.in1.primary = &lcl_sum_X_X2[CB];
          add_param.out.primary = sumsq_ncp_ptr;
          add_kernel(&add_param);

          /* for (int cb = 0; cb < CB; cb++) { */
          /*   sum_ncp_ptr[cb] += lcl_sum_X_X2[cb]; */
          /*   sumsq_ncp_ptr[cb] += lcl_sum_X_X2[CB + cb]; */
          /* } */
        }
      }
    }

    #pragma omp barrier

    #pragma omp for
    for (cp = 0; cp < CP; cp++) {
      libxsmm_meltw_unary_param all_zero_param;
      all_zero_param.out.primary = &sum_X_X2[cp*CB];
      all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &sum_X_X2[CP*CB + cp*CB];
      all_zero_kernel(&all_zero_param);

      /* #pragma omp simd */
      /* for (int cb = 0; cb < CB; cb++) { */
      /*   sum_X_X2[cp*CB + cb] = 0.0f; */
      /*   sum_X_X2[CP*CB + (cp*CB + cb)] = 0.0f; */
      /* } */

      int ni, cb;
      libxsmm_meltw_binary_param add_param;
      for(ni = 0; ni < N; ni++){
        add_param.in0.primary = &sum_X_X2[cp*CB];
        add_param.in1.primary = &sum_N[cp*N*CB + ni*CB];
        add_param.out.primary = &sum_X_X2[cp*CB];
        add_kernel(&add_param);

        add_param.in0.primary = &sum_X_X2[CP*CB + cp*CB];
        add_param.in1.primary = &sumsq_N[cp*N*CB + ni*CB];
        add_param.out.primary = &sum_X_X2[CP*CB + cp*CB];
        add_kernel(&add_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < CB; cb++) { */
        /*  sum_X_X2[cp*CB + cb] += sum_N[cp*N*CB + n*CB + cb]; */
        /*  sum_X_X2[CP*CB + (cp*CB + cb)] += sumsq_N[cp*N*CB + n*CB + cb]; */
        /*  } */
      }

      for(cb = 0; cb < CB; cb++){
        mean[cp*CB + cb] = sum_X_X2[cp*CB + cb] * scale;                                                             /* E[X] */
        var[cp*CB + cb] = (sum_X_X2[CP*CB + cp*CB + cb] * scale) - (mean[cp*CB + cb]*mean[cp*CB + cb]);              /* var(X) = E[X^2] - (E[X])^2 */
      }
    }

    #pragma omp for nowait collapse(2)
    for (cp = 0; cp < CP; cp++){
      for(n = 0; n < N; n++){                                                                                    /* Parallelize over batches and CP */
        libxsmm_matrix_arg arg_array[5];                                                                             /* private eqn args and params*/
        libxsmm_matrix_eqn_param eqn_param;
        int hwb, cb;
        for(cb = 0; cb < CB; cb++){
          s[cb] = 1.0f / ((float)sqrt(var[cp*CB + cb] + eps));                                                       /* s = 1/sqrt(var(X) + eps)     [CB] */
          b[cb] = -1 * mean[cp*CB + cb] * s[cb];                                                                     /* b = -E[X]/sqrt(var(X) + eps) [CB] */
        }
        arg_array[1].primary = s;                                                              /* [CB] */
        arg_array[2].primary = b;                                                              /* [CB] */
        arg_array[3].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);                       /* [CB] */
        arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, beta, cp, 0, CB);                        /* [CB] */

        for(hwb=0; hwb < num_HW_blocks; hwb++){
          arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);           /* [HW, CB] */
          eqn_param.inputs = arg_array;
          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, out, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);       /* [HW,CB] */
          func10(&eqn_param);                                                                                         /* Normalization equation -> y = ((s*x + b)*gamma + beta) */
        }
      }
    }
  }
}

void tpp_batchnorm_bwd_bf16(long N, long CP, long HW, long CB, long num_HW_blocks, libxsmm_bfloat16 *pdout, libxsmm_bfloat16 *pinp, float *mean, float *var, libxsmm_bfloat16 *pgamma, libxsmm_bfloat16 *pdin, float *pdgamma, float *pdbeta,
    libxsmm_matrix_eqn_function dgamma_func, libxsmm_matrix_eqn_function dbeta_func, libxsmm_matrix_eqn_function din_func, float eps,
    libxsmm_meltwfunction_unary all_zero_kernel, libxsmm_meltwfunction_binary add_kernel, libxsmm_meltwfunction_unary copy_kernel) {


  const float scale = 1.0f / ((float)N*HW);                              /* Scaling parameter*/

  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, din, pdin, CP, HW, CB);          /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, inp, pinp, CP, HW, CB);          /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, dout, pdout, CP, HW, CB);        /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, gamma, pgamma, CB);              /* [CP, CB] */
  /* LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, CB); */
  /* LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, CB); */

  LIBXSMM_ALIGNED(float dgamma_N[CP*N*CB], 64);
  LIBXSMM_ALIGNED(float dbeta_N[CP*N*CB], 64);

  #pragma omp parallel
  {
    LIBXSMM_ALIGNED(float a[CB], 64);
    LIBXSMM_ALIGNED(float b[CB], 64);
    LIBXSMM_ALIGNED(float c[CB], 64);
    int n, cp;

    #pragma omp for nowait collapse(2)
    for (cp = 0; cp < CP; cp++) {
      for (n = 0; n < N; n++) {

        int hwb, cb;
        libxsmm_matrix_arg arg_array[10];                                /* Private values of args and params */
        libxsmm_matrix_eqn_param eqn_param;

        LIBXSMM_ALIGNED(float lcl_dgamma_ptr[CB], 64);
        LIBXSMM_ALIGNED(float lcl_dbeta_ptr[CB], 64);

        float *dgamma_ncp_ptr = &dgamma_N[cp*N*CB + n*CB];
        float *dbeta_ncp_ptr = &dbeta_N[cp*N*CB + n*CB];

        libxsmm_meltw_unary_param all_zero_param;
        all_zero_param.out.primary = lcl_dgamma_ptr;
        all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = lcl_dbeta_ptr;
        all_zero_kernel(&all_zero_param);

        /* #pragma omp simd */
        /*  for (int cb = 0; cb < CB; cb++) { */
        /*  lcl_dgamma_ptr[cb] = 0.0f; */
        /*  lcl_dbeta_ptr[cb] = 0.0f; */
        /* } */

        for(cb = 0; cb < CB; cb++){
          a[cb] = 1.0f / ((float)sqrt(var[cp*CB + cb] + eps));
          b[cb] = -a[cb]*mean[cp*CB + cb];
        }

        arg_array[1].primary = a;
        arg_array[2].primary = b;
        arg_array[4].primary = lcl_dgamma_ptr;
        arg_array[5].primary = lcl_dbeta_ptr;
        arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);

        for(hwb=0; hwb < num_HW_blocks; hwb++){
          arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          eqn_param.inputs = arg_array;

          eqn_param.output.primary = lcl_dgamma_ptr;
          dgamma_func(&eqn_param);                                                             /* dgamma += (a * inp + b) * dout */

          eqn_param.output.primary = lcl_dbeta_ptr;
          dbeta_func(&eqn_param);                                                              /* dbeta += dout */
        }

        libxsmm_meltw_unary_param copy_param;
        copy_param.in.primary = lcl_dgamma_ptr;
        copy_param.out.primary = dgamma_ncp_ptr;
        copy_kernel(&copy_param);

        copy_param.in.primary = lcl_dbeta_ptr;
        copy_param.out.primary = dbeta_ncp_ptr;
        copy_kernel(&copy_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < CB; cb++) { */
        /*  dgamma_ncp_ptr[cb] = lcl_dgamma_ptr[cb]; */
        /*  dbeta_ncp_ptr[cb] = lcl_dbeta_ptr[cb]; */
        /* } */
      }
    }

    #pragma omp barrier

    #pragma omp for
    for (cp = 0; cp < CP; cp++) {
      libxsmm_meltw_unary_param all_zero_param;
      all_zero_param.out.primary = &pdgamma[cp*CB];
      all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &pdbeta[cp*CB];
      all_zero_kernel(&all_zero_param);

      /* #pragma omp simd */
      /* for (int cb = 0; cb < CB; cb++) { */
      /*  pdgamma[cp*CB + cb] = 0.0f; */
      /*  pdbeta[cp*CB + cb] = 0.0f; */
      /* } */

      libxsmm_meltw_binary_param add_param;
      int ni;
      for(ni = 0; ni < N; ni++){
        add_param.in0.primary = &pdgamma[cp*CB];
        add_param.in1.primary = &dgamma_N[cp*N*CB + ni*CB];
        add_param.out.primary = &pdgamma[cp*CB];
        add_kernel(&add_param);

        add_param.in0.primary = &pdbeta[cp*CB];
        add_param.in1.primary = &dbeta_N[cp*N*CB + ni*CB];
        add_param.out.primary = &pdbeta[cp*CB];
        add_kernel(&add_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < CB; cb++) { */
        /*  pdgamma[cp*CB + cb] += dgamma_N[cp*N*CB + n*CB + cb]; */
        /*  pdbeta[cp*CB + cb] += dbeta_N[cp*N*CB + n*CB + cb]; */
        /* } */
      }
    }


    #pragma omp for nowait collapse(2)                                                              /* Parallelize over batches */
    for (cp = 0; cp < CP; cp++) {
      for(n = 0; n < N; n++){
        libxsmm_matrix_arg arg_array[8];                                                           /* Private eqn args and params */
        libxsmm_matrix_eqn_param eqn_param;
        int hwb, cb;

        for(cb = 0; cb < CB; cb++){
          a[cb] = upconvert_bf16(pgamma[cp*CB + cb]) / ((float)sqrt(var[cp*CB + cb] + eps));        /* a = gamma_ptr[CB] * brstd_ptr[CB] */
          b[cb] = -a[cb] * scale * pdgamma[cp*CB + cb] / ((float)sqrt(var[cp*CB + cb] + eps));      /* b = gamma_ptr[CB] * brstd_ptr[CB] * del_gamma_ptr[v] * brstd_ptr[CB] * recp_nhw */
          c[cb] = -b[cb] * mean[cp*CB + cb] - a[cb] * scale * pdbeta[cp*CB + cb] ;                  /* c = -gamma_ptr[CB] * brstd_ptr[CB] * recp_nhw * del_beta_ptr[CB] + gamma_ptr[CB] * brstd_ptr[CB] * recp_nhw * bmean_ptr[CB] * del_gamma_ptr[CB] * brstd_ptr[CB]) */
        }

        arg_array[1].primary = a;
        arg_array[2].primary = b;
        arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
        arg_array[7].primary = c;

        for(hwb=0; hwb < num_HW_blocks; hwb++){
          arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);

          eqn_param.inputs = arg_array;
          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, din, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          din_func(&eqn_param);                                                                     /* din = dout * a * gamma + b * inp + c */
        }
      }
    }
  }
}

void tpp_batchnorm_fwd_fp32(long N, long CP, long HW, long CB, long num_HW_blocks, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout, float eps,
                          libxsmm_matrix_eqn_function func10, libxsmm_meltwfunction_unary reduce_HW_kernel, libxsmm_meltwfunction_unary all_zero_kernel, libxsmm_meltwfunction_binary add_kernel, libxsmm_meltwfunction_unary copy_kernel) {

  const float scale = 1.0f /((float)N * HW);
  LIBXSMM_ALIGNED(float sum_X_X2[CP*2*CB], 64);

  LIBXSMM_VLA_DECL(4, float, inp, pinp, CP, HW, CB);            /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, float, out, pout, CP, HW, CB);            /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);                /* [CP, CB] */
  LIBXSMM_VLA_DECL(2, float, beta, pbeta, CB);                  /* [CP, CB] */

  LIBXSMM_ALIGNED(float sum_N[CP*N*CB], 64);
  LIBXSMM_ALIGNED(float sumsq_N[CP*N*CB], 64);

  #pragma omp parallel
  {
    LIBXSMM_ALIGNED(float s[CB], 64);
    LIBXSMM_ALIGNED(float b[CB], 64);
    int n, cp;

    #pragma omp for nowait collapse(2)
    for (cp = 0; cp < CP; cp++) {
      for(n = 0; n < N; n++){

        int hwb;
        float *sum_ncp_ptr = &sum_N[cp*N*CB + n*CB];
        float *sumsq_ncp_ptr = &sumsq_N[cp*N*CB + n*CB];

        libxsmm_meltw_unary_param all_zero_param;
        all_zero_param.out.primary = sum_ncp_ptr;
        all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = sumsq_ncp_ptr;
        all_zero_kernel(&all_zero_param);

        /* #pragma omp simd  */
        /* for (int cb = 0; cb < CB; cb++) {  */
        /*   sum_ncp_ptr[cb] = 0.0f;    */
        /*   sumsq_ncp_ptr[cb] = 0.0f;  */
        /* } */

        libxsmm_meltw_binary_param add_param;

        libxsmm_meltw_unary_param reduce_HW_params;       /*Private params and tmp array */
        LIBXSMM_ALIGNED(float lcl_sum_X_X2[2*CB], 64);
        reduce_HW_params.out.primary   = lcl_sum_X_X2;                                                         /* [2*CB]  */
        for(hwb=0; hwb < num_HW_blocks; hwb++){

          reduce_HW_params.in.primary    = &LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          reduce_HW_kernel(&reduce_HW_params);                                                       /* [HW, CB] -----> [2 * CB] */

          add_param.in0.primary = sum_ncp_ptr;
          add_param.in1.primary = lcl_sum_X_X2;
          add_param.out.primary = sum_ncp_ptr;
          add_kernel(&add_param);

          add_param.in0.primary = sumsq_ncp_ptr;
          add_param.in1.primary = &lcl_sum_X_X2[CB];
          add_param.out.primary = sumsq_ncp_ptr;
          add_kernel(&add_param);

          /* #pragma omp simd */
          /* for (int cb = 0; cb < CB; cb++) {  */
          /*   sum_ncp_ptr[cb] += lcl_sum_X_X2[cb];  */
          /*   sumsq_ncp_ptr[cb] += lcl_sum_X_X2[CB + cb];  */
          /* }  */
        }
      }
    }

    #pragma omp barrier

    #pragma omp for
    for (cp = 0; cp < CP; cp++) {

      libxsmm_meltw_unary_param all_zero_param;
      all_zero_param.out.primary = &sum_X_X2[cp*CB];
      all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &sum_X_X2[CP*CB + cp*CB];
      all_zero_kernel(&all_zero_param);

      /* #pragma omp simd */
      /* for (int cb = 0; cb < CB; cb++) {  */
      /*   sum_X_X2[cp*CB + cb] = 0.0f;   */
      /*   sum_X_X2[CP*CB + (cp*CB + cb)] = 0.0f;  */
      /* } */

      libxsmm_meltw_binary_param add_param;
      int cb, ni;
      for(ni = 0; ni < N; ni++){

        add_param.in0.primary = &sum_X_X2[cp*CB];
        add_param.in1.primary = &sum_N[cp*N*CB + ni*CB];
        add_param.out.primary = &sum_X_X2[cp*CB];
        add_kernel(&add_param);

        add_param.in0.primary = &sum_X_X2[CP*CB + cp*CB];
        add_param.in1.primary = &sumsq_N[cp*N*CB + ni*CB];
        add_param.out.primary = &sum_X_X2[CP*CB + cp*CB];
        add_kernel(&add_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < CB; cb++) { */
        /*   sum_X_X2[cp*CB + cb] += sum_N[cp*N*CB + n*CB + cb]; */
        /*   sum_X_X2[CP*CB + (cp*CB + cb)] += sumsq_N[cp*N*CB + n*CB + cb]; */
        /* } */
      }

      for(cb = 0; cb < CB; cb++){
        mean[cp*CB + cb] = sum_X_X2[cp*CB + cb] * scale;                                                  /* E[X] */
        var[cp*CB + cb] = (sum_X_X2[CP*CB + cp*CB + cb] * scale) - (mean[cp*CB + cb]*mean[cp*CB + cb]);
      }
    }

    #pragma omp for nowait collapse(2)
    for (cp = 0; cp < CP; cp++){
      for(n = 0; n < N; n++){                                                             /* Parallelize over batches and CP*/

        libxsmm_matrix_arg arg_array[5];                                                         /* private eqn args and params*/
        libxsmm_matrix_eqn_param eqn_param;
        int hwb, cb;

        for(cb = 0; cb < CB; cb++){
          s[cb] = 1.0f / ((float)sqrt(var[cp*CB + cb] + eps));                                 /* s = 1/sqrt(var(X) + eps)     [CB] */
          b[cb] = -1 * mean[cp*CB + cb] * s[cb];                                               /* b = -E[X]/sqrt(var(X) + eps) [CB] */
        }
        arg_array[1].primary = s;                                                              /* [CB] */
        arg_array[2].primary = b;                                                              /* [CB] */
        arg_array[3].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);                       /* [CB] */
        arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, beta, cp, 0, CB);                        /* [CB] */

        for(hwb=0; hwb < num_HW_blocks; hwb++){
          arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);           /* [HW, CB] */
          eqn_param.inputs = arg_array;
          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, out, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);       /* [HW,CB] */
          func10(&eqn_param);                                                                    /* Normalization equation -> y = ((s*x + b)*gamma + beta) */
        }
      }
    }
  }
}

void tpp_batchnorm_bwd_fp32(long N, long CP, long HW, long CB, long num_HW_blocks, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta,
    libxsmm_matrix_eqn_function dgamma_func, libxsmm_matrix_eqn_function dbeta_func, libxsmm_matrix_eqn_function din_func, float eps,
    libxsmm_meltwfunction_unary all_zero_kernel, libxsmm_meltwfunction_binary add_kernel, libxsmm_meltwfunction_unary copy_kernel) {


  const float scale = 1.0f / ((float)N*HW);                   /* Scaling parameter*/

  LIBXSMM_VLA_DECL(4, float, din, pdin, CP, HW, CB);          /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, float, inp, pinp, CP, HW, CB);          /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, float, dout, pdout, CP, HW, CB);        /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);              /* [CP, CB] */
  /* LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, CB); */
  /* LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, CB); */

  LIBXSMM_ALIGNED(float dgamma_N[CP*N*CB], 64);
  LIBXSMM_ALIGNED(float dbeta_N[CP*N*CB], 64);


  #pragma omp parallel
  {
    LIBXSMM_ALIGNED(float a[CB], 64);
    LIBXSMM_ALIGNED(float b[CB], 64);
    LIBXSMM_ALIGNED(float c[CB], 64);
    int n, cp;

    #pragma omp for nowait collapse(2)
    for (cp = 0; cp < CP; cp++) {
      for (n = 0; n < N; n++) {

        int hwb, cb;
        libxsmm_matrix_arg arg_array[10];                                                           /* Private values of args and params */
        libxsmm_matrix_eqn_param eqn_param;

        LIBXSMM_ALIGNED(float lcl_dgamma_ptr[CB], 64);
        LIBXSMM_ALIGNED(float lcl_dbeta_ptr[CB], 64);

        float *dgamma_ncp_ptr = &dgamma_N[cp*N*CB + n*CB];
        float *dbeta_ncp_ptr = &dbeta_N[cp*N*CB + n*CB];

        libxsmm_meltw_unary_param all_zero_param;
        all_zero_param.out.primary = lcl_dgamma_ptr;
        all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = lcl_dbeta_ptr;
        all_zero_kernel(&all_zero_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < CB; cb++) { */
        /*   lcl_dgamma_ptr[cb] = 0.0f; */
        /*   lcl_dbeta_ptr[cb] = 0.0f; */
        /* } */

        for(cb = 0; cb < CB; cb++){
          a[cb] = 1.0f / ((float)sqrt(var[cp*CB + cb] + eps));
          b[cb] = -a[cb]*mean[cp*CB + cb];
        }

        arg_array[1].primary = a;
        arg_array[2].primary = b;
        arg_array[4].primary = lcl_dgamma_ptr;
        arg_array[5].primary = lcl_dbeta_ptr;
        arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);

        for(hwb=0; hwb < num_HW_blocks; hwb++){

          arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);

          eqn_param.inputs = arg_array;
          eqn_param.output.primary = lcl_dgamma_ptr;
          dgamma_func(&eqn_param);                                                             /* dgamma += (a * inp + b) * dout */

          eqn_param.output.primary = lcl_dbeta_ptr;
          dbeta_func(&eqn_param);                                                              /* dbeta += dout */
        }

        libxsmm_meltw_unary_param copy_param;
        copy_param.in.primary = lcl_dgamma_ptr;
        copy_param.out.primary = dgamma_ncp_ptr;
        copy_kernel(&copy_param);

        copy_param.in.primary = lcl_dbeta_ptr;
        copy_param.out.primary = dbeta_ncp_ptr;
        copy_kernel(&copy_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < CB; cb++) { */
        /*   dgamma_ncp_ptr[cb] = lcl_dgamma_ptr[cb]; */
        /*   dbeta_ncp_ptr[cb] = lcl_dbeta_ptr[cb]; */
        /* } */
      }
    }

    #pragma omp barrier

    #pragma omp for
    for (cp = 0; cp < CP; cp++) {
      libxsmm_meltw_unary_param all_zero_param;
      all_zero_param.out.primary = &pdgamma[cp*CB];
      all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &pdbeta[cp*CB];
      all_zero_kernel(&all_zero_param);

      /* #pragma omp simd */
      /* for (int cb = 0; cb < CB; cb++) { */
      /*   pdgamma[cp*CB + cb] = 0.0f; */
      /*   pdbeta[cp*CB + cb] = 0.0f; */
      /* } */

      libxsmm_meltw_binary_param add_param;
      int ni;
      for(ni = 0; ni < N; ni++){

        add_param.in0.primary = &pdgamma[cp*CB];
        add_param.in1.primary = &dgamma_N[cp*N*CB + ni*CB];
        add_param.out.primary = &pdgamma[cp*CB];
        add_kernel(&add_param);

        add_param.in0.primary = &pdbeta[cp*CB];
        add_param.in1.primary = &dbeta_N[cp*N*CB + ni*CB];
        add_param.out.primary = &pdbeta[cp*CB];
        add_kernel(&add_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < CB; cb++) { */
        /*   pdgamma[cp*CB + cb] += dgamma_N[cp*N*CB + n*CB + cb];  */
        /*   pdbeta[cp*CB + cb] += dbeta_N[cp*N*CB + n*CB + cb];  */
        /* } */
      }
    }

    #pragma omp for nowait collapse(2)                                                                  /* Parallelize over batches and CP*/
    for (cp = 0; cp < CP; cp++) {
      for(n = 0; n < N; n++){

        libxsmm_matrix_arg arg_array[8];                                                               /* Private eqn args and params */
        libxsmm_matrix_eqn_param eqn_param;
        int hwb, cb;

        for(cb = 0; cb < CB; cb++){
          a[cb] = pgamma[cp*CB + cb] / ((float)sqrt(var[cp*CB + cb] + eps));                            /* a = gamma_ptr[CB] * brstd_ptr[CB] */
          b[cb] = -a[cb] * scale * pdgamma[cp*CB + cb] / ((float)sqrt(var[cp*CB + cb] + eps));          /* b = gamma_ptr[CB] * brstd_ptr[CB] * del_gamma_ptr[v] * brstd_ptr[CB] * recp_nhw */
          c[cb] = -b[cb] * mean[cp*CB + cb] - a[cb] * scale * pdbeta[cp*CB + cb] ;                      /* c = -gamma_ptr[CB] * brstd_ptr[CB] * recp_nhw * del_beta_ptr[CB] + gamma_ptr[CB] * brstd_ptr[CB] * recp_nhw * bmean_ptr[CB] * del_gamma_ptr[CB] * brstd_ptr[CB]) */
        }

        arg_array[1].primary = a;
        arg_array[2].primary = b;
        arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
        arg_array[7].primary = c;

        for(hwb=0; hwb < num_HW_blocks; hwb++){
          arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);

          eqn_param.inputs = arg_array;
          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, din, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          din_func(&eqn_param);                                                                        /* din = dout * a + b * inp + c */
        }
      }
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

  LIBXSMM_ALIGNED(float sum_N[CP*N*CB], 64);
  LIBXSMM_ALIGNED(float sumsq_N[CP*N*CB], 64);

  /* #pragma omp parallel for collapse(2) reduction(+: sum_X[:2*CP*CB]) reduction(+: sum_X2[:2*CP*CB])    */
  /* for(int n = 0; n < N; n++){ */
  /*   for(int cp = 0; cp < CP; cp++){ */
  /*     for(int hw = 0; hw < HW; hw++){ */
  /*       for(int cb = 0; cb < CB; cb++){ */
  /*         sum_X[cp*CB + cb] += LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB); */
  /*         sum_X2[cp*CB + cb] += (LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB)*LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB)); */
  /*       } */
  /*     } */
  /*   } */
  /* } */

  int n, cp, j;

  #pragma omp parallel
  {

    #pragma omp for collapse(2)
    for(n = 0; n < N; n++){
      for (cp = 0; cp < CP; cp++) {

        int hw, cb;
        LIBXSMM_ALIGNED(float lcl_sum_ptr[CB], 64);
        LIBXSMM_ALIGNED(float lcl_sumsq_ptr[CB], 64);

        float *sum_ncp_ptr = &sum_N[cp*N*CB + n*CB];
        float *sumsq_ncp_ptr = &sumsq_N[cp*N*CB + n*CB];

        for (cb = 0; cb < CB; cb++) {
          lcl_sum_ptr[cb] = 0.0f;
          lcl_sumsq_ptr[cb] = 0.0f;
        }

        for(hw = 0; hw < HW; hw++){
          for(cb = 0; cb < CB; cb++){
            lcl_sum_ptr[cb] += LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB);
            lcl_sumsq_ptr[cb] += (LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB)*LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB));
          }
        }

        for (cb = 0; cb < CB; cb++) {
          sum_ncp_ptr[cb] = lcl_sum_ptr[cb];
          sumsq_ncp_ptr[cb] = lcl_sumsq_ptr[cb];
        }
      }
    }

    #pragma omp barrier

    #pragma omp for
    for (cp = 0; cp < CP; cp++) {
      int ni, cb;
      for (cb = 0; cb < CB; cb++) {
        sum_X[cp*CB + cb] = 0.0f;
        sum_X2[cp*CB + cb] = 0.0f;
      }

      for(ni = 0; ni < N; ni++){
        for (cb = 0; cb < CB; cb++) {
          sum_X[cp*CB + cb] += sum_N[cp*N*CB + ni*CB + cb];
          sum_X2[cp*CB + cb] += sumsq_N[cp*N*CB + ni*CB + cb];
        }
      }
    }
  }


  for(j = 0; j < CP*CB; j++){
    mean[j] = sum_X[j] / ((float)N * HW);                                           /* E[X] */
    var[j] = (sum_X2[j] / ((float)N * HW)) - (mean[j]*mean[j]);                     /* var(X) = E[X^2] - (E[X])^2 */
    s[j] = 1.0f / ((float)sqrt(var[j] + eps));                                      /* s = 1/sqrt(var(X) + eps)     [CP, CB] */
    b[j] = -1 * mean[j] * s[j];                                                     /* b = -E[X]/sqrt(var(X) + eps) [CP, CB] */
  }

  #pragma omp parallel for collapse(2)
  for(n = 0; n < N; n++){                                                       /* Data movement 2*N*CP*HW*CB */
    for(cp = 0; cp < CP; cp++){
      int cb, hw;
      for(hw = 0; hw < HW; hw++){
        for(cb = 0; cb < CB; cb++){
          LIBXSMM_VLA_ACCESS(4, out, n, cp, hw, cb, CP, HW, CB) = ((LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB) * s[cp*CB + cb]) + b[cp*CB + cb]) * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB) + LIBXSMM_VLA_ACCESS(2, beta, cp, cb, CB);        /* Normalization equation -> y = ((s*x + b)*gamma + beta) */
        }
      }
    }
  }
}

void scaler_batchnorm_bwd_fp32(long N, long CP, long HW, long CB, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta, float eps) {

  const float scale = 1.0f / ((float)N*HW);

  LIBXSMM_ALIGNED(float a[CP*CB], 64);
  LIBXSMM_ALIGNED(float b[CP*CB], 64);
  LIBXSMM_ALIGNED(float c[CP*CB], 64);
  LIBXSMM_ALIGNED(float ds[CP*CB], 64);
  LIBXSMM_ALIGNED(float db[CP*CB], 64);

  LIBXSMM_VLA_DECL(4, float, din, pdin, CP, HW, CB);
  LIBXSMM_VLA_DECL(4, float, inp, pinp, CP, HW, CB);
  LIBXSMM_VLA_DECL(4, float, dout, pdout, CP, HW, CB);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);
  /* LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, CB); */
  /* LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, CB); */

  LIBXSMM_ALIGNED(float dgamma_N[CP*N*CB], 64);
  LIBXSMM_ALIGNED(float dbeta_N[CP*N*CB], 64);
  LIBXSMM_ALIGNED(float ds_N[CP*N*CB], 64);
  LIBXSMM_ALIGNED(float db_N[CP*N*CB], 64);

  int n, cp, j;

  for(j = 0; j < CP*CB; j++){                             /* Initialize the arrays */
    a[j] = 1.0f / ((float)sqrt(var[j] + eps));
    b[j] = -a[j]*mean[j];
  }

  /* #pragma omp parallel for collapse(2) reduction(+: pdgamma[:CP*CB]) reduction(+: pdbeta[:CP*CB]) reduction(+: ds[:CP*CB]) reduction(+: db[:CP*CB]) */
  /* for(int n = 0; n < N; n++){ */
  /*   for (int cp = 0; cp < CP; cp++) {               */
  /*     for (int hw = 0; hw < HW; hw++){ */
  /*       for (int cb = 0; cb < CB; cb++) { */
  /*         pdgamma[cp*CB + cb] += (a[cp*CB + cb] * LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB) + b[cp*CB + cb]) * LIBXSMM_VLA_ACCESS(4, dout, n, cp, hw, cb, CP, HW, CB); */
  /*         pdbeta[cp*CB + cb] += LIBXSMM_VLA_ACCESS(4, dout, n, cp, hw, cb, CP, HW, CB); */
  /*         ds[cp*CB + cb] += LIBXSMM_VLA_ACCESS(4, dout, n, cp, hw, cb, CP, HW, CB) * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB) * LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB); */
  /*         db[cp*CB + cb] += LIBXSMM_VLA_ACCESS(4, dout, n, cp, hw, cb, CP, HW, CB) * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB); */
  /*       } */
  /*     } */
  /*   } */
  /* } */


  #pragma omp parallel
  {
    #pragma omp for collapse(2)
    for(n = 0; n < N; n++){
      for (cp = 0; cp < CP; cp++) {                    /* dgamma += (a * inp + b) * dout , dbeta += dout, ds += dout * gamma * inp, db += dout * gamma */

        int cb, hw;
        LIBXSMM_ALIGNED(float lcl_dgamma_ptr[CB], 64);
        LIBXSMM_ALIGNED(float lcl_dbeta_ptr[CB], 64);
        LIBXSMM_ALIGNED(float lcl_ds_ptr[CB], 64);
        LIBXSMM_ALIGNED(float lcl_db_ptr[CB], 64);
        float *dgamma_ncp_ptr = &dgamma_N[cp*N*CB + n*CB];
        float *dbeta_ncp_ptr = &dbeta_N[cp*N*CB + n*CB];
        float *ds_ncp_ptr = &ds_N[cp*N*CB + n*CB];
        float *db_ncp_ptr = &db_N[cp*N*CB + n*CB];

        for (cb = 0; cb < CB; cb++) {
          lcl_dgamma_ptr[cb] = 0.0f;
          lcl_dbeta_ptr[cb] = 0.0f;
          lcl_ds_ptr[cb] = 0.0f;
          lcl_db_ptr[cb] = 0.0f;
        }

        for (hw = 0; hw < HW; hw++){
          for (cb = 0; cb < CB; cb++) {
            lcl_dgamma_ptr[cb] += (a[cp*CB + cb] * LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB) + b[cp*CB + cb]) * LIBXSMM_VLA_ACCESS(4, dout, n, cp, hw, cb, CP, HW, CB);
            lcl_dbeta_ptr[cb] += LIBXSMM_VLA_ACCESS(4, dout, n, cp, hw, cb, CP, HW, CB);
            lcl_ds_ptr[cb] += LIBXSMM_VLA_ACCESS(4, dout, n, cp, hw, cb, CP, HW, CB) * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB) * LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB);
            lcl_db_ptr[cb] += LIBXSMM_VLA_ACCESS(4, dout, n, cp, hw, cb, CP, HW, CB) * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB);
          }
        }

        for (cb = 0; cb < CB; cb++) {
          dgamma_ncp_ptr[cb] = lcl_dgamma_ptr[cb];
          dbeta_ncp_ptr[cb] = lcl_dbeta_ptr[cb];
          ds_ncp_ptr[cb] = lcl_ds_ptr[cb];
          db_ncp_ptr[cb] = lcl_db_ptr[cb];
        }
      }
    }

    #pragma omp barrier

    #pragma omp for
    for (cp = 0; cp < CP; cp++) {

      int cb, ni;
      for (cb = 0; cb < CB; cb++) {
        pdgamma[cp*CB + cb] = 0.0f;
        pdbeta[cp*CB + cb] = 0.0f;
        ds[cp*CB + cb] = 0.0f;
        db[cp*CB + cb] = 0.0f;
      }

      for(ni = 0; ni < N; ni++){
        for (cb = 0; cb < CB; cb++) {
          pdgamma[cp*CB + cb] += dgamma_N[cp*N*CB + ni*CB + cb];
          pdbeta[cp*CB + cb] += dbeta_N[cp*N*CB + ni*CB + cb];
          ds[cp*CB + cb] += ds_N[cp*N*CB + ni*CB + cb];
          db[cp*CB + cb] += db_N[cp*N*CB + ni*CB + cb];
        }
      }
    }
  }

  /* b = (db * mean[nb] - ds) * a * a * a * scale; */
  /* c = -b * mean[nb] - db * a * scale; */

  for(j = 0; j < CP*CB; j++){
    b[j] = (db[j] * mean[j] - ds[j]) * a[j] * a[j] * a[j] * scale;
    c[j] = -b[j] * mean[j] - db[j] * a[j] * scale;
  }

  #pragma omp parallel for collapse(2)
  for(n = 0; n < N; n++){                                                             /* Parallelize over batches,      Data movement 3*N*CP*HW*CB */
    for (cp = 0; cp < CP; cp++) {                                                     /* din = dout * a * gamma + b * inp + c */
      int cb, hw;
      for (hw = 0; hw < HW; hw++){
        for (cb = 0; cb < CB; cb++) {
          LIBXSMM_VLA_ACCESS(4, din, n, cp, hw, cb, CP, HW, CB) = LIBXSMM_VLA_ACCESS(4, dout, n, cp, hw, cb, CP, HW, CB)  * a[cp*CB + cb] * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB) + b[cp*CB + cb] * LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB) + c[cp*CB + cb];
          /* LIBXSMM_VLA_ACCESS(4, din, n, cp, hw, cb, CP, HW, CB) = LIBXSMM_VLA_ACCESS(4, dout, n, cp, hw, cb, CP, HW, CB)  * a[cp*CB + cb] + b[cp*CB + cb] * LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB) + c[cp*CB + cb]; */
        }
      }
    }
  }
}

void reference_batchnorm_fwd_fp32(long N, long CP, long HW, long CB, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout, float eps){

  const float recp_nhw = 1.0f/((float)N*HW);

  LIBXSMM_ALIGNED(float expectval_ptr[CP*CB], 64);
  LIBXSMM_ALIGNED(float rcpstddev_ptr[CP*CB], 64);

  LIBXSMM_VLA_DECL(4, float, inp, pinp, CP, HW, CB);            /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, float, out, pout, CP, HW, CB);            /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2, float, beta, pbeta, CB);

  int n, cp, hw, cb = 0;                                                   /* Since no blocking on channels */
  for (cp = 0; cp < CP; cp++) {
    float ch_sum = 0.0f;
    float ch_sumsq = 0.0f;
    float tbmean = 0.0f;
    float tbmeansq = 0.0f;
    float tsqbmean = 0.0f;
    float tbrstd = 0.0f;
    float tvariance = 0.0f;

    for (n = 0; n < N; n++ ) {
      for (hw = 0; hw < HW; hw++){
        const float input_val = LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB);
        ch_sum   += input_val;
        ch_sumsq += (input_val * input_val);
      }
    }

    tbmean = recp_nhw * ch_sum;
    tbmeansq  = tbmean * tbmean;
    tsqbmean = recp_nhw * ch_sumsq;
    tvariance = tsqbmean - tbmeansq;
    tbrstd = (float)(1.0/sqrt(tvariance + eps));
    expectval_ptr[cp] = tbmean;
    rcpstddev_ptr[cp] = tbrstd;
  }

  for (n = 0; n < N; n++ ) {
    for (cp = 0; cp < CP; cp++ ) {
      for(hw = 0; hw < HW; hw++){
          const float  input_val     =  LIBXSMM_VLA_ACCESS(4, inp, n, cp, hw, cb, CP, HW, CB);

          /* BN + scale (gamma, beta) */
          LIBXSMM_VLA_ACCESS(4, out, n, cp, hw, cb, CP, HW, CB) = LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB)*(input_val - expectval_ptr[cp])*rcpstddev_ptr[cp] + LIBXSMM_VLA_ACCESS(2, beta, cp, cb, CB);
      }
    }
  }
}

void reference_batchnorm_bwd_fp32(long N, long CP, long HW, long CB, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta, float eps){

  const float nhw = (float)N * HW;
  const float recp_nhw = 1.0f/((float)N*HW);
  LIBXSMM_ALIGNED(float expectval_ptr[CP*CB], 64);
  LIBXSMM_ALIGNED(float rcpstddev_ptr[CP*CB], 64);

  LIBXSMM_VLA_DECL(4, float, din, pdin, CP, HW, CB);
  LIBXSMM_VLA_DECL(4, float, inp, pinp, CP, HW, CB);
  LIBXSMM_VLA_DECL(4, float, dout, pdout, CP, HW, CB);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, CB);

  printf("\n Using reference implementation \n");
  int n, cp, hw, cb = 0;                     /* Since no blocking on channels */
  for (cp = 0; cp < CP; cp++ ) {
    LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, CB) = 0.0f;
    LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, CB) = 0.0f;
    expectval_ptr[cp] = mean[cp];
    rcpstddev_ptr[cp] = (float)(1.0 / (sqrt(var[cp] + eps)));

    for (n = 0; n < N; n++ ) {
      for (hw = 0; hw < HW; hw++){
        const float  input_val         =  LIBXSMM_VLA_ACCESS(4,      inp, n, cp, hw, cb, CP, HW, CB);
        float* del_output_ptr    = &LIBXSMM_VLA_ACCESS(4,    dout, n, cp, hw, cb, CP, HW, CB);

        LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, CB) += (input_val - expectval_ptr[cp]) * (*del_output_ptr) * rcpstddev_ptr[cp];
        LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, CB)  += *del_output_ptr;
      }
    }
  }

  for (n = 0; n < N; n++ ) {
    for (cp = 0; cp < CP; cp++ ) {
      for (hw = 0; hw < HW; hw++){
        float* del_input_ptr  = &LIBXSMM_VLA_ACCESS(4,     din, n, cp, hw, cb, CP, HW, CB);
        const float  input_val      =  LIBXSMM_VLA_ACCESS(4,    inp, n, cp, hw, cb, CP, HW, CB);
        const float  del_output_val =  LIBXSMM_VLA_ACCESS(4,    dout, n, cp, hw, cb, CP, HW, CB);

        *del_input_ptr = LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB) * rcpstddev_ptr[cp] * recp_nhw * (nhw * del_output_val -
                  (LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, CB) + (input_val - expectval_ptr[cp]) * LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, CB) * rcpstddev_ptr[cp]));
      }
    }
  }
}


int main( int argc, char* argv[] ) {
  libxsmm_blasint my_eqn10, my_eqn11, my_eqn12, my_eqn16;
  libxsmm_matrix_eqn_function func10, func11, func12, func16;
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
  long N = 28;
  long CP = 2;
  long HW = 784;
  long CB = 64;
  long num_HW_blocks = 16;
  int iters = 100;
  int datatype_mode = 0;
  libxsmm_datatype  in_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype  out_dt = LIBXSMM_DATATYPE_F32;

  if ( argc > 1 ) N = atoi(argv[1]);
  if ( argc > 2 ) CP = atoi(argv[2]);
  if ( argc > 3 ) HW = atoi(argv[3]);
  if ( argc > 4 ) CB = atoi(argv[4]);
  if ( argc > 5 ) num_HW_blocks = atoi(argv[5]);
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

  libxsmm_blasint ldo = CB;
  libxsmm_meltwfunction_unary all_zero_kernel = libxsmm_dispatch_meltw_unary(CB, 1, NULL, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  if ( all_zero_kernel == NULL) {
      fprintf( stderr, "JIT for initialization by unary all zero copy kernel failed. Bailing...!\n");
      exit(-1);
  }

  libxsmm_meltwfunction_binary add_kernel = libxsmm_dispatch_meltw_binary(CB, 1, &ldo, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_MELTW_TYPE_BINARY_ADD);
  if ( add_kernel == NULL) {
      fprintf( stderr, "JIT for initialization of add kernel failed. Bailing...!\n");
      exit(-1);
  }

  libxsmm_meltwfunction_unary copy_kernel = libxsmm_dispatch_meltw_unary(CB, 1, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
  if ( copy_kernel == NULL) {
      fprintf( stderr, "JIT for initialization by copy kernel failed. Bailing...!\n");
      exit(-1);
  }

  /* TPPs for reducing X and X2 in HW*/
  ld = CB;
  tmp_ld = CB;

  unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD;
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  reduce_HW_kernel = libxsmm_dispatch_meltw_unary(CB, HW/num_HW_blocks, &ld, &tmp_ld, in_dt, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);

  /* TPP for scaling */
  ld = CB;
  tmp_ld = 1;
  tmp_ld2 = 1;

  my_eqn10 = libxsmm_matrix_eqn_create();                                                        /* y = (s*x + b)*gamma + beta */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn10, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32);
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn10, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32);
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, CB, HW/num_HW_blocks, ld, 0, 0, in_dt );                         /* x = [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, CB, 1, tmp_ld, 1, 0, LIBXSMM_DATATYPE_F32 );       /* s = [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, CB, 1, tmp_ld, 2, 0, LIBXSMM_DATATYPE_F32 );       /* b = [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, CB, 1, tmp_ld2, 3, 0, in_dt );                     /* gamma = [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, CB, 1, tmp_ld2, 4, 0, in_dt );                     /* beta = [CB] */
  /* libxsmm_matrix_eqn_tree_print( my_eqn10 ); */
  /* libxsmm_matrix_eqn_rpn_print( my_eqn10 ); */
  func10 = libxsmm_dispatch_matrix_eqn( CB, HW/num_HW_blocks, &ld, out_dt, my_eqn10 );                         /* y = [HW, CB] */


  /* Check correctness */
  if (datatype_mode == 0) {
    tpp_batchnorm_fwd_fp32(N, CP, HW, CB, num_HW_blocks, inp, gamma, beta, mean, var, eqn_out, eps, func10, reduce_HW_kernel, all_zero_kernel, add_kernel, copy_kernel);
    if(CB == 1)
      reference_batchnorm_fwd_fp32(N, CP, HW, CB, inp, gamma, beta, mean, var, out, eps);
    else
      scaler_batchnorm_fwd_fp32(N, CP, HW, CB, inp, gamma, beta, mean, var, out, eps);
  } else if (datatype_mode == 1) {
    tpp_batchnorm_fwd_bf16(N, CP, HW, CB, num_HW_blocks, bf16_inp, bf16_gamma, bf16_beta, mean, var, bf16_eqn_out, eps, func10, reduce_HW_kernel, all_zero_kernel, add_kernel, copy_kernel);
    scaler_batchnorm_fwd_fp32(N, CP, HW, CB, inp, gamma, beta, mean, var, out, eps);
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
    printf("Scaler batchnorm time FWD  = %.5g\n", ((double)(l_total)));
    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i] + (float)l_total;
    }
    tpp_batchnorm_fwd_fp32(N, CP, HW, CB, num_HW_blocks, inp, gamma, beta, mean, var, eqn_out, eps, func10, reduce_HW_kernel, all_zero_kernel, add_kernel, copy_kernel);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      tpp_batchnorm_fwd_fp32(N, CP, HW, CB, num_HW_blocks, inp, gamma, beta, mean, var, eqn_out, eps, func10, reduce_HW_kernel, all_zero_kernel, add_kernel, copy_kernel);
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
    tpp_batchnorm_fwd_bf16(N, CP, HW, CB, num_HW_blocks, bf16_inp, bf16_gamma, bf16_beta, mean, var, bf16_eqn_out, eps, func10, reduce_HW_kernel, all_zero_kernel, add_kernel, copy_kernel);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      tpp_batchnorm_fwd_bf16(N, CP, HW, CB, num_HW_blocks, bf16_inp, bf16_gamma, bf16_beta, mean, var, bf16_eqn_out, eps, func10, reduce_HW_kernel, all_zero_kernel, add_kernel, copy_kernel);
    }
    l_end = libxsmm_timer_tick();
    l_total2 = libxsmm_timer_duration(l_start, l_end);
    printf("TPP batchnorm (BF16) time FWD  = %.5g\n", ((double)(l_total2)));
    printf("Speedup FWD is %.5g\n", l_total/l_total2);
  }


  /* Create MatEq for bwd layernorm */

  ld = CB;
  tmp_ld2 = 1;

  /* dgamma function  */
  my_eqn11 = libxsmm_matrix_eqn_create();                                                       /* dgamma = ((inp *a + b) * dout) + dgamma */
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn11, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32);                   /* dgamma = ((inp *a + b) * dout) + dgamma */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn11, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);   /* [HW, CB] -> [CB] */
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn11, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32);                   /* ((inp *a + b) * dout) */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn11, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32);
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, CB, HW/num_HW_blocks, ld, 0, 0, in_dt );          /* inp [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, CB, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 );           /* a [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, CB, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );           /* b [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, CB, HW/num_HW_blocks, ld, 3, 0, in_dt );          /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, CB, 1, 1, 4, 0, LIBXSMM_DATATYPE_F32 );           /* dgamma [CB] */
  /* libxsmm_matrix_eqn_tree_print( my_eqn11 ); */
  /* libxsmm_matrix_eqn_rpn_print( my_eqn11 ); */
  func11 = libxsmm_dispatch_matrix_eqn( CB, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn11 );      /* dgamma [CB] */

  /* dbeta function  */
  my_eqn12 = libxsmm_matrix_eqn_create();                                                       /* dbeta [CB] = dout [HW, CB] + dbeta [CB] */
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn12, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );                /* dbeta_tmp [HW, CB] */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn12, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);  /* [HW, CB] -> [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn12, CB, HW/num_HW_blocks, ld, 3, 0, in_dt );          /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn12, CB, 1, 1, 5, 0, LIBXSMM_DATATYPE_F32 );           /* dbeta [CB] */
  /* libxsmm_matrix_eqn_tree_print( my_eqn12 ); */
  /* libxsmm_matrix_eqn_rpn_print( my_eqn12 ); */
  func12 = libxsmm_dispatch_matrix_eqn( CB, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn12 );      /* dbeta [CB] */

  /* din = gamma_ptr[v] * brstd_ptr[v] * recp_nhw * (nhw*del_output_ptr[v] - (del_beta_ptr[v] + (input_ptr[v] - bmean_ptr[v]) * del_gamma_ptr[v] * brstd_ptr[v])) */
  /* din = gamma_ptr[v] * brstd_ptr[v] *del_output_ptr[v] - gamma_ptr[v] * brstd_ptr[v] * recp_nhw * (del_beta_ptr[v] + (input_ptr[v] - bmean_ptr[v]) * del_gamma_ptr[v] * brstd_ptr[v])) */
  /* din = gamma_ptr[v] * brstd_ptr[v] *del_output_ptr[v] - gamma_ptr[v] * brstd_ptr[v] * recp_nhw * del_beta_ptr[v] + gamma_ptr[v] * brstd_ptr[v] * recp_nhw * (input_ptr[v] - bmean_ptr[v]) * del_gamma_ptr[v] * brstd_ptr[v]) */
  /* din = a * del_output_ptr[v] + b * input_ptr[v] + c */
  /* a = gamma_ptr[CB] * brstd_ptr[CB] */
  /* b = gamma_ptr[CB] *  del_gamma_ptr[v] * brstd_ptr[CB] * brstd_ptr[CB] * recp_nhw */
  /* c = -gamma_ptr[CB] * brstd_ptr[CB] * recp_nhw * del_beta_ptr[CB] + gamma_ptr[CB] * brstd_ptr[CB] * recp_nhw * bmean_ptr[CB] * del_gamma_ptr[CB] * brstd_ptr[CB]) */

  /* din long equation */
  my_eqn16 = libxsmm_matrix_eqn_create();                                                       /* din = a * dout + (b * inp + c) */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn16, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32);
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, CB, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 );           /* a [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, CB, HW/num_HW_blocks, ld, 3, 0, in_dt );          /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn16, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32);
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, CB, HW/num_HW_blocks, ld, 0, 0, in_dt );          /* inp [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, CB, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );           /* b [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, CB, 1, 1, 7, 0, LIBXSMM_DATATYPE_F32 );           /* c [CB] */
  /* libxsmm_matrix_eqn_tree_print( my_eqn16 ); */
  /* libxsmm_matrix_eqn_rpn_print( my_eqn16 ); */
  func16 = libxsmm_dispatch_matrix_eqn( CB, HW/num_HW_blocks, &ld, in_dt, my_eqn16 );           /* din [HW, CB] */


  if (datatype_mode == 0) {
    tpp_batchnorm_bwd_fp32(N, CP, HW, CB, num_HW_blocks, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func16, eps, all_zero_kernel, add_kernel, copy_kernel);
    if (CB == 1)
      reference_batchnorm_bwd_fp32(N, CP, HW, CB, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
    else
      scaler_batchnorm_bwd_fp32(N, CP, HW, CB, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);

  } else if (datatype_mode == 1) {
    tpp_batchnorm_bwd_bf16(N, CP, HW, CB, num_HW_blocks, bf16_eqn_dout, bf16_inp, mean, var, bf16_gamma, bf16_eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func16, eps, all_zero_kernel, add_kernel, copy_kernel);
    scaler_batchnorm_bwd_fp32(N, CP, HW, CB, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
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
    scaler_batchnorm_bwd_fp32(N, CP, HW, CB, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      scaler_batchnorm_bwd_fp32(N, CP, HW, CB, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("Scaler batchnorm time BWD = %.5g\n", ((double)(l_total)));
    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i] + (float)l_total;
    }
    tpp_batchnorm_bwd_fp32(N, CP, HW, CB, num_HW_blocks, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func16, eps, all_zero_kernel, add_kernel, copy_kernel);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      tpp_batchnorm_bwd_fp32(N, CP, HW, CB, num_HW_blocks, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func16, eps, all_zero_kernel, add_kernel, copy_kernel);
    }
    l_end = libxsmm_timer_tick();
    l_total2 = libxsmm_timer_duration(l_start, l_end);
    printf("TPP batchnorm time BWD = %.5g\n", ((double)(l_total2)));
    printf("Speedup BWD is %.5g\n", l_total/l_total2);
  } else if (datatype_mode == 1) {
    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i];
    }
    scaler_batchnorm_bwd_fp32(N, CP, HW, CB, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      scaler_batchnorm_bwd_fp32(N, CP, HW, CB, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("Scaler batchnorm (FP32) time BWD  = %.5g\n", ((double)(l_total)));
    for (i = 0; i < 1024 * 1024; i++ ) {
      sum += cache_fl[i] + (float)l_total;
    }
    tpp_batchnorm_bwd_bf16(N, CP, HW, CB, num_HW_blocks, bf16_eqn_dout, bf16_inp, mean, var, bf16_gamma, bf16_eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func16, eps, all_zero_kernel, add_kernel, copy_kernel);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      tpp_batchnorm_bwd_bf16(N, CP, HW, CB, num_HW_blocks, bf16_eqn_dout, bf16_inp, mean, var, bf16_gamma, bf16_eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func16, eps, all_zero_kernel, add_kernel, copy_kernel);
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
