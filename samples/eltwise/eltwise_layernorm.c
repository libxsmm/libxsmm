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
/* include c-based dnn library */
#include "../deeplearning/common/dnn_common.h"
#if defined(_OPENMP)
# include <omp.h>
#endif
#define EPS 1e-9

LIBXSMM_INLINE
void sfill_matrix ( float *matrix, unsigned int ld, unsigned int m, unsigned int n )
{
  unsigned int i, j;
  double dtmp;

  if ( ld < m )
  {
     fprintf(stderr,"Error is sfill_matrix: ld=%u m=%u mismatched!\n",ld,m);
     exit(EXIT_FAILURE);
  }
  for ( j = 1; j <= n; j++ )
  {
     /* Fill through the leading dimension */
     for ( i = 1; i <= ld; i++ )
     {
        dtmp = 1.0 - 2.0*libxsmm_rng_f64();
        matrix [ (j-1)*ld + (i-1) ] = (float) dtmp;
     }
  }
}

LIBXSMM_INLINE
void naive_layernorm(int m, int n, int ld_in, float *sinp, float *gamma, float *beta, float *sout_ref, float *mean_data_ref, float *rstd_data_ref)
{
  int i, j;

  for (j = 0; j < n; j++) {
    float mean_val_ref = 0.0, rstd_val_ref = 0.0, scale_ref = 0.0, bias_ref = 0.0, gamma_val_ref = 0.0, beta_val_ref = 0.0;
    mean_data_ref[j] = 0;
    rstd_data_ref[j] = 0;
    for (i = 0; i < m; i++) {
      mean_data_ref[j] += sinp[j*ld_in + i];
      rstd_data_ref[j] += sinp[j*ld_in + i] * sinp[j*ld_in + i];
    }
    mean_val_ref = mean_data_ref[j]/m;
    rstd_val_ref = (rstd_data_ref[j]/m)-mean_val_ref*mean_val_ref;
    rstd_val_ref = 1/((float)sqrt(rstd_val_ref));
    mean_data_ref[j] = mean_val_ref;
    rstd_data_ref[j] = rstd_val_ref;
    scale_ref = rstd_val_ref;
    bias_ref = -1.0 * rstd_val_ref * mean_val_ref;
    for (i = 0; i < m; i++) {
      gamma_val_ref = gamma[i];
      beta_val_ref = beta[i];
      sout_ref[j*ld_in+i] += (sinp[j*ld_in+i] * scale_ref + bias_ref) * gamma_val_ref + beta_val_ref;
    }
  }
}


  LIBXSMM_INLINE
void naive_layernorm_bwd(int m, int n, int ld_in, float *dY, float *X, float *mean, float *rstd, float *gamma, float *dX, float *dgamma, float *dbeta)
{
  float a, b, c, ds, db, scale = 1.0/ (1.0 * m);
  int i, j;

  for (j = 0; j < n; j++) {
    a = rstd[j];
    b = -1.0 * a * mean[j];
    ds = 0.0;
    db = 0.0;
    for (i = 0; i < m; i++) {
      dgamma[i]     += dY[j*ld_in+i] * (a * X[j*ld_in+i] + b);
      dbeta[i]      += dY[j*ld_in+i];
      ds            += dY[j*ld_in+i] * X[j*ld_in+i] * gamma[i];
      db            += dY[j*ld_in+i] * gamma[i];
    }

    b = (db * mean[j] - ds) * a * a * a * scale;
    c = -1.0 * b * mean[j] - db * a * scale;
    for (i = 0; i < m; i++) {
      dX[j*ld_in+i] = a * dY[j*ld_in+i] * gamma[i] + b * X[j*ld_in+i] + c;
    }
  }
}

  LIBXSMM_INLINE
void optimized_layernorm(int m, int n, int ld_in, float *sinp, float *gamma, float *beta, float *sout, float *mean_data, float *rstd_data, libxsmm_meltwfunction_reduce reduce_kernel, libxsmm_meltwfunction_scale scalemean_kernel, libxsmm_meltwfunction_scale scaleout_kernel, float * bias_aux)
{
  int i;
  float reverse_m = 1.0/(1.0*m);
#if defined(__AVX512F__)
  __m512 minus_ones = _mm512_set1_ps(-1.0);
#endif

  libxsmm_meltw_reduce_param reduce_params;
  libxsmm_meltw_scale_param scalemean_params;
  libxsmm_meltw_scale_param scaleout_params;

  reduce_params.in_ptr    = sinp;
  reduce_params.out_ptr_0 = mean_data;
  reduce_params.out_ptr_1 = rstd_data;
  reduce_kernel(&reduce_params);

  scalemean_params.in_ptr         = mean_data;
  scalemean_params.out_ptr        = mean_data;
  scalemean_params.scale_vals_ptr = &reverse_m;
  scalemean_kernel(&scalemean_params);

  scalemean_params.in_ptr         = rstd_data;
  scalemean_params.out_ptr        = rstd_data;
  scalemean_kernel(&scalemean_params);

  /* Calculate rstd and auxiliary bias vectors*/
#if defined(__AVX512F__)
  for (i = 0; i < n-15; i+= 16) {
    __m512 vrstd = _mm512_loadu_ps(rstd_data+i);
    __m512 vmean = _mm512_loadu_ps(mean_data+i);
    vrstd = _mm512_rsqrt14_ps(_mm512_sub_ps(vrstd, _mm512_mul_ps(vmean, vmean)));
    _mm512_storeu_ps(rstd_data+i, vrstd);
    _mm512_storeu_ps(bias_aux+i, _mm512_mul_ps(minus_ones, _mm512_mul_ps(vmean, vrstd)));
  }

  if (i < n) {
    int rem = n - i;
    __mmask16 mask = (1 << rem) - 1;
    __m512 vrstd = _mm512_maskz_loadu_ps(mask, rstd_data+i);
    __m512 vmean = _mm512_maskz_loadu_ps(mask, mean_data+i);
    vrstd = _mm512_maskz_rsqrt14_ps(mask, _mm512_sub_ps(vrstd, _mm512_mul_ps(vmean, vmean)));
    _mm512_mask_storeu_ps(rstd_data+i, mask, vrstd );
    _mm512_mask_storeu_ps(bias_aux+i, mask, _mm512_mul_ps(minus_ones, _mm512_mul_ps(vmean, vrstd)));
  }
#else
  for (i = 0; i < n; i++) {
    rstd_data[i]  = 1.0/((float)sqrt(rstd_data[i] - mean_data[i] * mean_data[i]));
    bias_aux[i]   =-1.0 * mean_data[i] * rstd_data[i];
  }
#endif

  scaleout_params.in_ptr          = sinp;
  scaleout_params.out_ptr         = sout;
  scaleout_params.scale_vals_ptr  = rstd_data;
  scaleout_params.bias_vals_ptr   = bias_aux;
  scaleout_params.scale_vals_ptr2 = gamma;
  scaleout_params.bias_vals_ptr2  = beta;
  scaleout_kernel(&scaleout_params);
}


  LIBXSMM_INLINE
void optimized_blocked_layernorm(int m, int n, int bm, int bn, float *data_in, float *gamma_data, float *beta_data, float *mean_data, float *rstd_data)
{
  int ld = bm, ld_vector = bn;
  libxsmm_meltw_redu_flags jit_reduce_flags = LIBXSMM_MELTW_FLAG_REDUCE_NONE;
  libxsmm_meltwfunction_reduce reduce_rows_kernel, reduce_cols_kernel;
  libxsmm_meltw_scal_flags jit_scale_flags = 0;
  libxsmm_meltwfunction_scale scale_kernel;
  libxsmm_meltw_scal_flags jit_scaleout_flags = 0;
  libxsmm_meltwfunction_scale scaleout_kernel;

  int nBlocks   = n/bn;
  int mBlocks   = m/bm;
  float scratch[2*n*mBlocks+n];
  float *sums_ptr     = (float*) scratch;
  float *sums_sq_ptr  = (float*) scratch + n * mBlocks;
  float *aux_bias_ptr = (float*) scratch + 2 * n * mBlocks;

  LIBXSMM_VLA_DECL(3, float, sums,        sums_ptr, mBlocks, bn);
  LIBXSMM_VLA_DECL(3, float, sums_sq,     sums_sq_ptr, mBlocks, bn);
  LIBXSMM_VLA_DECL(2, float, mean,        mean_data, bn);
  LIBXSMM_VLA_DECL(2, float, rstd,        rstd_data, bn);
  LIBXSMM_VLA_DECL(2, float, gamma,       gamma_data, bm);
  LIBXSMM_VLA_DECL(2, float, beta,        beta_data, bm);
  LIBXSMM_VLA_DECL(2, float, aux_bias,    aux_bias_ptr, bn);
  LIBXSMM_VLA_DECL(4, float, X,           data_in, mBlocks, bn, bm);

  /*libxsmm_barrier *barrier;*/

  libxsmm_init();

  /* Generate JITED kernels for optimized code */
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_REDUCE_ROWS | LIBXSMM_MELTW_FLAG_REDUCE_OP_ADD | LIBXSMM_MELTW_FLAG_REDUCE_ELTS | LIBXSMM_MELTW_FLAG_REDUCE_ELTS_SQUARED;
  reduce_rows_kernel = libxsmm_dispatch_meltw_reduce(bm, bn, &ld, &ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags);
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_REDUCE_COLS | LIBXSMM_MELTW_FLAG_REDUCE_OP_ADD | LIBXSMM_MELTW_FLAG_REDUCE_ELTS;
  reduce_cols_kernel = libxsmm_dispatch_meltw_reduce(bn, mBlocks, &ld, &ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags);
  jit_scale_flags = LIBXSMM_MELTW_FLAG_SCALE_ROWS | LIBXSMM_MELTW_FLAG_SCALE_MULT;
  scale_kernel = libxsmm_dispatch_meltw_scale(bn, 1, &ld_vector, &ld_vector, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_scale_flags);
  jit_scaleout_flags = LIBXSMM_MELTW_FLAG_SCALE_ROWS_COLS | LIBXSMM_MELTW_FLAG_SCALE_MULT | LIBXSMM_MELTW_FLAG_SCALE_ADD_BIAS;
  scaleout_kernel = libxsmm_dispatch_meltw_scale(bm, bn, &ld, &ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_scaleout_flags);

#if defined(_OPENMP)
#     pragma omp parallel
#endif
  {
    int i, imin, im, in;
    float reverse_m = 1.0/(1.0*m);
#if defined(__AVX512F__)
    __m512 minus_ones = _mm512_set1_ps(-1.0);
#endif
#if defined(_OPENMP)
    const int ltid = omp_get_thread_num();
#else
    const int ltid = 0;
#endif
#if defined(_OPENMP)
    int threads = omp_get_max_threads(); /* number of threads */
#else
    int threads = 1; /* number of threads */
#endif

    const int work_mn = nBlocks * mBlocks;
    const int chunksize_mn = (work_mn % threads == 0) ? (work_mn /threads) : ((work_mn / threads) + 1);
    const int thr_begin_mn = (ltid * chunksize_mn < work_mn) ? (ltid * chunksize_mn) : work_mn;
    const int thr_end_mn = ((ltid + 1) * chunksize_mn < work_mn) ? ((ltid + 1) * chunksize_mn) : work_mn;

    const int work_n = nBlocks;
    const int chunksize_n = (work_n % threads == 0) ? (work_n /threads) : ((work_n / threads) + 1);
    const int thr_begin_n = (ltid * chunksize_n < work_n) ? (ltid * chunksize_n) : work_n;
    const int thr_end_n = ((ltid + 1) * chunksize_n < work_n) ? ((ltid + 1) * chunksize_n) : work_n;

    libxsmm_meltw_reduce_param reduce_rows_params, reduce_cols_params;;
    libxsmm_meltw_scale_param scale_params;
    libxsmm_meltw_scale_param scaleout_params;

    /*libxsmm_barrier_init(barrier, ltid);*/

    for (imin = thr_begin_mn; imin < thr_end_mn; imin++) {
      in = imin / mBlocks;
      im = imin % mBlocks;
      reduce_rows_params.in_ptr    = &LIBXSMM_VLA_ACCESS(4, X, in, im, 0, 0, mBlocks, bn, bm);
      reduce_rows_params.out_ptr_0 = &LIBXSMM_VLA_ACCESS(3, sums,    in, im, 0, mBlocks, bn);
      reduce_rows_params.out_ptr_1 = &LIBXSMM_VLA_ACCESS(3, sums_sq, in, im, 0, mBlocks, bn);
      reduce_rows_kernel(&reduce_rows_params);
    }

#pragma omp barrier
    /*libxsmm_barrier_wait(barrier, ltid);*/

    scale_params.scale_vals_ptr = &reverse_m;
    for (in = thr_begin_n; in < thr_end_n; in++) {
      reduce_cols_params.in_ptr    = &LIBXSMM_VLA_ACCESS(3, sums,    in, 0, 0, mBlocks, bn);
      reduce_cols_params.out_ptr_0 = &LIBXSMM_VLA_ACCESS(2, mean,    in, 0, bn);
      reduce_cols_kernel(&reduce_cols_params);
      scale_params.in_ptr         = &LIBXSMM_VLA_ACCESS(2, mean,    in, 0, bn);
      scale_params.out_ptr        = &LIBXSMM_VLA_ACCESS(2, mean,    in, 0, bn);
      scale_kernel(&scale_params);
      reduce_cols_params.in_ptr    = &LIBXSMM_VLA_ACCESS(3, sums_sq, in, 0, 0, mBlocks, bn);
      reduce_cols_params.out_ptr_0 = &LIBXSMM_VLA_ACCESS(2, rstd,    in, 0, bn);
      reduce_cols_kernel(&reduce_cols_params);
      scale_params.in_ptr         = &LIBXSMM_VLA_ACCESS(2, rstd,    in, 0, bn);
      scale_params.out_ptr        = &LIBXSMM_VLA_ACCESS(2, rstd,    in, 0, bn);
      scale_kernel(&scale_params);
    }

#pragma omp barrier
    /*libxsmm_barrier_wait(barrier, ltid);*/

    /* Calculate rstd and auxiliary bias vectors*/
    for (in = thr_begin_n; in < thr_end_n; in++) {
      float *rstd_ptr = &LIBXSMM_VLA_ACCESS(2, rstd,    in, 0, bn);
      float *mean_ptr = &LIBXSMM_VLA_ACCESS(2, mean,    in, 0, bn);
      float *bias_ptr = &LIBXSMM_VLA_ACCESS(2, aux_bias, in, 0, bn);
#if defined(__AVX512F__)
      for (i = 0; i < bn-15; i+= 16) {
        __m512 vrstd = _mm512_loadu_ps(rstd_ptr+i);
        __m512 vmean = _mm512_loadu_ps(mean_ptr+i);
        vrstd = _mm512_rsqrt14_ps(_mm512_sub_ps(vrstd, _mm512_mul_ps(vmean, vmean)));
        _mm512_storeu_ps(rstd_ptr+i, vrstd);
        _mm512_storeu_ps(bias_ptr+i, _mm512_mul_ps(minus_ones, _mm512_mul_ps(vmean, vrstd)));
      }

      if (i < bn) {
        int rem = bn - i;
        __mmask16 mask = (1 << rem) - 1;
        __m512 vrstd = _mm512_maskz_loadu_ps(mask, rstd_ptr+i);
        __m512 vmean = _mm512_maskz_loadu_ps(mask, mean_ptr+i);
        vrstd = _mm512_maskz_rsqrt14_ps(mask, _mm512_sub_ps(vrstd, _mm512_mul_ps(vmean, vmean)));
        _mm512_mask_storeu_ps(rstd_ptr+i, mask, vrstd );
        _mm512_mask_storeu_ps(bias_ptr+i, mask, _mm512_mul_ps(minus_ones, _mm512_mul_ps(vmean, vrstd)));
      }
#else
      for (i = 0; i < bn; i++) {
        rstd_ptr[i]  = 1.0/((float)sqrt(rstd_ptr[i] - mean_ptr[i] * mean_ptr[i]));
        bias_ptr[i]   =-1.0 * mean_ptr[i] * mean_ptr[i];
      }
#endif
    }

#pragma omp barrier
    /*libxsmm_barrier_wait(barrier, ltid);*/

    for (imin = thr_begin_mn; imin < thr_end_mn; imin++) {
      in = imin / mBlocks;
      im = imin % mBlocks;
      scaleout_params.in_ptr          = &LIBXSMM_VLA_ACCESS(4, X, in, im, 0, 0, mBlocks, bn, bm);
      scaleout_params.out_ptr         = &LIBXSMM_VLA_ACCESS(4, X, in, im, 0, 0, mBlocks, bn, bm);
      scaleout_params.scale_vals_ptr  = &LIBXSMM_VLA_ACCESS(2, rstd,    in, 0, bn);
      scaleout_params.bias_vals_ptr   = &LIBXSMM_VLA_ACCESS(2, aux_bias, in, 0, bn);
      scaleout_params.scale_vals_ptr2 = &LIBXSMM_VLA_ACCESS(2, gamma,    im, 0, bm);
      scaleout_params.bias_vals_ptr2  = &LIBXSMM_VLA_ACCESS(2, beta,    im, 0, bm);
      scaleout_kernel(&scaleout_params);
    }
#pragma omp barrier
    /*libxsmm_barrier_wait(barrier, ltid);*/
  }
}

int main(int argc, char* argv[])
{
  unsigned int m = 64, n = 64, iters = 10000, k = 0;
  libxsmm_blasint ld_in = 64, ld_vector = 64;

  float  *sinp, *gamma, *beta, *sout, *sout_nc, *mean_data, *rstd_data, *sout_ref, *mean_data_ref, *rstd_data_ref, *bias_aux;
  libxsmm_matdiff_info norms_out, norms_mean, norms_rstd;
  unsigned long long l_start, l_end;
  double l_total = 0.0, l_total2 = 0.0;

  libxsmm_meltw_redu_flags jit_reduce_flags = LIBXSMM_MELTW_FLAG_REDUCE_NONE;
  libxsmm_meltwfunction_reduce reduce_kernel;
  libxsmm_meltw_scal_flags jit_scalemean_flags = 0;
  libxsmm_meltwfunction_scale scalemean_kernel;
  libxsmm_meltw_scal_flags jit_scaleout_flags = 0;
  libxsmm_meltwfunction_scale scaleout_kernel;
  libxsmm_init();

  libxsmm_matdiff_clear(&norms_out);
  libxsmm_matdiff_clear(&norms_mean);
  libxsmm_matdiff_clear(&norms_rstd);

  if ( argc > 1 ) m = atoi(argv[1]);
  if ( argc > 2 ) n = atoi(argv[2]);
  if ( argc > 3 ) ld_in = atoi(argv[3]);
  if ( argc > 4 ) iters = atoi(argv[7]);

  m = LIBXSMM_MAX(m,1);
  n = LIBXSMM_MAX(n,1);
  ld_vector = n;
  ld_in = LIBXSMM_MAX(ld_in,(libxsmm_blasint)m);

  /* Allocate arrays  */
  sinp      = (float*) malloc(ld_in*n*sizeof(float));
  gamma     = (float*) malloc(m*sizeof(float) );
  beta      = (float*) malloc(m*sizeof(float) );
  sout      = (float*) malloc(ld_in*n*sizeof(float) );
  sout_nc   = (float*) malloc(ld_in*n*sizeof(float) );
  mean_data = (float*) malloc(n*sizeof(float) );
  rstd_data = (float*) malloc(n*sizeof(float) );

  /* Allocate reference arrays  */
  mean_data_ref = (float*) malloc(n*sizeof(float) );
  rstd_data_ref = (float*) malloc(n*sizeof(float) );
  sout_ref      = (float*) malloc(ld_in*n*sizeof(float) );

  /* Allocate auxiliary arrays for optimized version */
  bias_aux      = (float*) malloc(n*sizeof(float) );

  /* Fill matrices with random data */
  sfill_matrix ( sinp, ld_in, m, n );
  sfill_matrix ( gamma, ld_in, m, 1 );
  sfill_matrix ( beta, ld_in, m, 1 );

  /* Calculate reference results... */
  naive_layernorm(m, n, ld_in, sinp, gamma, beta, sout_ref, mean_data_ref, rstd_data_ref);

  /* Generate JITED kernels for optimized code */
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_REDUCE_ROWS | LIBXSMM_MELTW_FLAG_REDUCE_OP_ADD | LIBXSMM_MELTW_FLAG_REDUCE_ELTS | LIBXSMM_MELTW_FLAG_REDUCE_ELTS_SQUARED;
  printf("JITing reduce kernel... \n");
  reduce_kernel = libxsmm_dispatch_meltw_reduce(m, n, &ld_in, &ld_in, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags);

  jit_scalemean_flags = LIBXSMM_MELTW_FLAG_SCALE_ROWS | LIBXSMM_MELTW_FLAG_SCALE_MULT;
  printf("JITing mean-scale kernel... \n");
  scalemean_kernel = libxsmm_dispatch_meltw_scale(n, 1, &ld_vector, &ld_vector, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_scalemean_flags);

  jit_scaleout_flags = LIBXSMM_MELTW_FLAG_SCALE_ROWS_COLS | LIBXSMM_MELTW_FLAG_SCALE_MULT | LIBXSMM_MELTW_FLAG_SCALE_ADD_BIAS;
  printf("JITing scaling kernel for output... \n");
  scaleout_kernel = libxsmm_dispatch_meltw_scale(m, n, &ld_in, &ld_in, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_scaleout_flags);

  /* Calculate blockde results... */
#if 0
  optimized_layernorm(m, n, ld_in, sinp, gamma, beta, sout, mean_data, rstd_data, reduce_kernel, scalemean_kernel, scaleout_kernel, bias_aux);
#else
  matrix_copy_NC_to_NCNC( sinp,  sout, 1, n, m, 64, 64 );

  optimized_blocked_layernorm(m, n, 64, 64, sout, gamma, beta, mean_data, rstd_data);

  matrix_copy_NCNC_to_NC( sout, sout_nc, 1, n, m, 64, 64 );
#endif

  /* compare */
  printf("##########################################\n");
  printf("#   Correctness - Output                 #\n");
  printf("##########################################\n");
#if 0
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ld_in*n, 1, sout_ref, sout, 0, 0);
#else
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ld_in*n, 1, sout_ref, sout_nc, 0, 0);
#endif
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  /* compare */
  printf("##########################################\n");
  printf("#   Correctness - Mean                   #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_mean, LIBXSMM_DATATYPE_F32, n, 1, mean_data_ref, mean_data, 0, 0);
  printf("L1 reference  : %.25g\n", norms_mean.l1_ref);
  printf("L1 test       : %.25g\n", norms_mean.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_mean.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_mean.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_mean.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_mean.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_mean.normf_rel);

  /* compare */
  printf("##########################################\n");
  printf("#   Correctness - Rstd                   #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_rstd, LIBXSMM_DATATYPE_F32, n, 1, rstd_data_ref, rstd_data, 0, 0);
  printf("L1 reference  : %.25g\n", norms_rstd.l1_ref);
  printf("L1 test       : %.25g\n", norms_rstd.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_rstd.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_rstd.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_rstd.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_rstd.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_rstd.normf_rel);


  l_start = libxsmm_timer_tick();
  /* Calculate reference results...  */
  for (k = 0; k < iters; k++) {
    naive_layernorm(m, n, ld_in, sinp, gamma, beta, sout_ref, mean_data_ref, rstd_data_ref);
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Reference time = %.5g\n", ((double)(l_total)));

  l_start = libxsmm_timer_tick();
  for (k = 0; k < iters; k++) {
#if 1
    optimized_blocked_layernorm(m, n, 64, 64, sout, gamma, beta, mean_data, rstd_data);
#else
    optimized_layernorm(m, n, ld_in, sinp, gamma, beta, sout, mean_data, rstd_data, reduce_kernel, scalemean_kernel, scaleout_kernel, bias_aux);
#endif
  }
  l_end = libxsmm_timer_tick();
  l_total2 = libxsmm_timer_duration(l_start, l_end);
  printf("Optimized time = %.5g\n", ((double)(l_total2)));
  printf("Speedup is = %.5g\n", ((double)(l_total/l_total2)));

  /* Free allocated arrays */
  free(sinp);
  free(gamma);
  free(beta);
  free(sout);
  free(mean_data);
  free(rstd_data);
  free(mean_data_ref);
  free(rstd_data_ref);
  free(sout_ref);
  free(bias_aux);

  return EXIT_SUCCESS;
}

