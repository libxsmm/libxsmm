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

#if defined(_OPENMP)
#pragma omp parallel for private(j)
#endif
  for (j = 0; j < n; j++) {
    float mean_val_ref = 0, rstd_val_ref = 0, scale_ref = 0, bias_ref = 0, gamma_val_ref = 0, beta_val_ref = 0;
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
    bias_ref = -1.f * rstd_val_ref * mean_val_ref;
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
  float a, b, c, ds, db, scale = (float)(1.0 / m);
  int i, j;

  for (i = 0; i < m; i++) {
    dgamma[i] = 0;
    dbeta[i]  = 0;
  }

  for (j = 0; j < n; j++) {
    a = rstd[j];
    b = -1.f * a * mean[j];
    ds = 0;
    db = 0;
    for (i = 0; i < m; i++) {
      dgamma[i]     += dY[j*ld_in+i] * (a * X[j*ld_in+i] + b);
      dbeta[i]      += dY[j*ld_in+i];
      ds            += dY[j*ld_in+i] * X[j*ld_in+i] * gamma[i];
      db            += dY[j*ld_in+i] * gamma[i];
    }

    b = (db * mean[j] - ds) * a * a * a * scale;
    c = -1.f * b * mean[j] - db * a * scale;
    for (i = 0; i < m; i++) {
      dX[j*ld_in+i] = a * dY[j*ld_in+i] * gamma[i] + b * X[j*ld_in+i] + c;
    }
  }
}

#if 0
LIBXSMM_INLINE
void optimized_layernorm(int m, int n, int ld_in, float *sinp, float *gamma, float *beta, float *sout, float *mean_data, float *rstd_data, libxsmm_meltwfunction_reduce reduce_kernel, libxsmm_meltwfunction_scale scalemean_kernel, libxsmm_meltwfunction_scale scaleout_kernel, float * bias_aux)
{
  int i;
  float reverse_m = (float)(1.0 / m);
#if defined(__AVX512F__)
  __m512 minus_ones = _mm512_set1_ps(-1.f);
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
    rstd_data[i] = (float)(1.0 / sqrt(rstd_data[i] - mean_data[i] * mean_data[i]));
    bias_aux[i]  = -1.f * mean_data[i] * rstd_data[i];
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

#else
LIBXSMM_INLINE
void optimized_blocked_layernorm(int m, int n, int bm, int bn, float *data_in, float *gamma_data, float *beta_data, float *mean_data, float *rstd_data)
{
  int ld = bm, ld_vector = bn, _ld;
  libxsmm_meltw_unary_flags jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type  unary_type;
  libxsmm_meltwfunction_unary reduce_rows_kernel, reduce_cols_kernel;
  libxsmm_meltw_scal_flags jit_scale_flags = 0;
  libxsmm_meltwfunction_scale scale_kernel;
  libxsmm_meltw_scal_flags jit_scaleout_flags = 0;
  libxsmm_meltwfunction_scale scaleout_kernel;
#if defined(_OPENMP)
  int threads = omp_get_max_threads(); /* number of threads */
#else
  int threads = 1; /* number of threads */
#endif

  int nBlocks   = n/bn;
  int mBlocks   = m/bm;
  float *const scratch = (float*)libxsmm_aligned_scratch((2 * n * mBlocks + n) * sizeof(float), 0/*auto-alignment*/);
  float *sums_sums_sq_ptr     = scratch;
  float *aux_bias_ptr         = scratch + 2 * n * mBlocks;

  LIBXSMM_VLA_DECL(3, float, sums_sums_sq,sums_sums_sq_ptr, mBlocks, 2*bn);
  LIBXSMM_VLA_DECL(2, float, mean,        mean_data, bn);
  LIBXSMM_VLA_DECL(2, float, rstd,        rstd_data, bn);
  LIBXSMM_VLA_DECL(2, float, gamma,       gamma_data, bm);
  LIBXSMM_VLA_DECL(2, float, beta,        beta_data, bm);
  LIBXSMM_VLA_DECL(2, float, aux_bias,    aux_bias_ptr, bn);
  LIBXSMM_VLA_DECL(4, float, X,           data_in, mBlocks, bn, bm);

  /*libxsmm_barrier *barrier;*/

  /* Generate JITED kernels for optimized code */
  unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD;
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS;
  reduce_rows_kernel = libxsmm_dispatch_meltw_unary(bm, bn, &ld, &ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);

  unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD;
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  _ld = 2*bn;
  reduce_cols_kernel = libxsmm_dispatch_meltw_unary(bn, mBlocks, &_ld, &ld_vector, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);

  jit_scale_flags = LIBXSMM_MELTW_FLAG_SCALE_ROWS | LIBXSMM_MELTW_FLAG_SCALE_MULT;
  scale_kernel = libxsmm_dispatch_meltw_scale(bn, 1, &ld_vector, &ld_vector, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_scale_flags, 0);
  jit_scaleout_flags = LIBXSMM_MELTW_FLAG_SCALE_ROWS_COLS | LIBXSMM_MELTW_FLAG_SCALE_MULT | LIBXSMM_MELTW_FLAG_SCALE_ADD_BIAS;
  scaleout_kernel = libxsmm_dispatch_meltw_scale(bm, bn, &ld, &ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_scaleout_flags, 0);

#if defined(_OPENMP)
#     pragma omp parallel
#endif
  {
    int i, imin, im, in;
    float reverse_m = (float)(1.0 / m);
#if defined(__AVX512F__)
    __m512 minus_ones = _mm512_set1_ps(-1.f);
#endif
#if defined(_OPENMP)
    const int ltid = omp_get_thread_num();
#else
    const int ltid = 0;
#endif

    const int work_mn = nBlocks * mBlocks;
    const int chunksize_mn = (work_mn % threads == 0) ? (work_mn /threads) : ((work_mn / threads) + 1);
    const int thr_begin_mn = (ltid * chunksize_mn < work_mn) ? (ltid * chunksize_mn) : work_mn;
    const int thr_end_mn = ((ltid + 1) * chunksize_mn < work_mn) ? ((ltid + 1) * chunksize_mn) : work_mn;

    const int work_n = nBlocks;
    const int chunksize_n = (work_n % threads == 0) ? (work_n /threads) : ((work_n / threads) + 1);
    const int thr_begin_n = (ltid * chunksize_n < work_n) ? (ltid * chunksize_n) : work_n;
    const int thr_end_n = ((ltid + 1) * chunksize_n < work_n) ? ((ltid + 1) * chunksize_n) : work_n;

    libxsmm_meltw_unary_param reduce_rows_params, reduce_cols_params;
    libxsmm_meltw_scale_param scale_params;
    libxsmm_meltw_scale_param scaleout_params;

    /*libxsmm_barrier_init(barrier, ltid);*/

    for (imin = thr_begin_mn; imin < thr_end_mn; imin++) {
      in = imin / mBlocks;
      im = imin % mBlocks;
      reduce_rows_params.in.primary    = &LIBXSMM_VLA_ACCESS(4, X, in, im, 0, 0, mBlocks, bn, bm);
      reduce_rows_params.out.primary   = &LIBXSMM_VLA_ACCESS(3, sums_sums_sq, in, im, 0, mBlocks, 2*bn);
      reduce_rows_kernel(&reduce_rows_params);
    }

#pragma omp barrier
    /*libxsmm_barrier_wait(barrier, ltid);*/

    scale_params.scale_vals_ptr = &reverse_m;
    for (in = thr_begin_n; in < thr_end_n; in++) {

      reduce_cols_params.in.primary    = &LIBXSMM_VLA_ACCESS(3, sums_sums_sq, in, 0, 0, mBlocks, 2*bn);
      reduce_cols_params.out.primary   = &LIBXSMM_VLA_ACCESS(2, mean,    in, 0, bn);
      reduce_cols_kernel(&reduce_cols_params);

      scale_params.in_ptr         = &LIBXSMM_VLA_ACCESS(2, mean,    in, 0, bn);
      scale_params.out_ptr        = &LIBXSMM_VLA_ACCESS(2, mean,    in, 0, bn);
      scale_kernel(&scale_params);

      reduce_cols_params.in.primary    = &LIBXSMM_VLA_ACCESS(3, sums_sums_sq, in, 0, bn, mBlocks, 2*bn);
      reduce_cols_params.out.primary   = &LIBXSMM_VLA_ACCESS(2, rstd,    in, 0, bn);
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
        rstd_ptr[i] = (float)(1.0 / sqrt(rstd_ptr[i] - mean_ptr[i] * mean_ptr[i]));
        bias_ptr[i] = -1.f * mean_ptr[i] * mean_ptr[i];
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

  libxsmm_free(scratch);
}
#endif

LIBXSMM_INLINE
void optimized_blocked_layernorm_bwd(int m, int n, int bm, int bn, float *_dY, float *_X, float *_mean, float *_rstd, float *_gamma, float *_dX, float *_dgamma, float *_dbeta)
{
  int ld = bm, ld_vector = bn;
  libxsmm_meltw_unary_flags jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type  unary_type;
  libxsmm_meltwfunction_unary reduce_rows_kernel, reduce_cols_kernel, reduce_cols_kernel2, reduce_cols_kernel3;
  int nBlocks   = n/bn;
  int mBlocks   = m/bm;
  float *const scratch = (float*)libxsmm_aligned_scratch((2 * n * mBlocks + 2 * m * nBlocks + 2 * n) * sizeof(float), 0/*auto-alignment*/);
  float *dgamma_aux_ptr     = scratch;
  float *dbeta_aux_ptr      = scratch + m * nBlocks;
  float *ds_aux_ptr         = scratch + 2 * m * nBlocks;
  float *db_aux_ptr         = scratch + 2 * m * nBlocks + n * mBlocks;
  float *db_ptr             = scratch + 2 * m * nBlocks + 2 * n * mBlocks;
  float *ds_ptr             = scratch + 2 * m * nBlocks + 2 * n * mBlocks + n;
  LIBXSMM_VLA_DECL(3, float, ds_aux,      ds_aux_ptr, mBlocks, bn);
  LIBXSMM_VLA_DECL(3, float, db_aux,      db_aux_ptr, mBlocks, bn);
  LIBXSMM_VLA_DECL(3, float, dgamma_aux,  dgamma_aux_ptr, nBlocks, bm);
  LIBXSMM_VLA_DECL(3, float, dbeta_aux,   dbeta_aux_ptr, nBlocks, bm);
  LIBXSMM_VLA_DECL(4, float, dY,          _dY, mBlocks, bn, bm);
  LIBXSMM_VLA_DECL(4, float, X,           _X, mBlocks, bn, bm);
  LIBXSMM_VLA_DECL(4, float, dX,          _dX, mBlocks, bn, bm);
  LIBXSMM_VLA_DECL(2, float, mean,        _mean, bn);
  LIBXSMM_VLA_DECL(2, float, rstd,        _rstd, bn);
  LIBXSMM_VLA_DECL(2, float, gamma,       _gamma, bm);
  LIBXSMM_VLA_DECL(2, float, dgamma,      _dgamma, bm);
  LIBXSMM_VLA_DECL(2, float, dbeta,       _dbeta, bm);
  LIBXSMM_VLA_DECL(2, float, ds,          ds_ptr, bn);
  LIBXSMM_VLA_DECL(2, float, db,          db_ptr, bn);

#if defined(_OPENMP)
  int threads = omp_get_max_threads(); /* number of threads */
#else
  int threads = 1; /* number of threads */
#endif

  /* Generate JITED kernels for optimized code */
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS;
  unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD;
  reduce_rows_kernel = libxsmm_dispatch_meltw_unary(bm, bn, &ld, &ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);

  jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  reduce_cols_kernel = libxsmm_dispatch_meltw_unary(bm, bn, &ld, &ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);
  reduce_cols_kernel2 = libxsmm_dispatch_meltw_unary(bm, nBlocks, &ld, &ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);
  reduce_cols_kernel3 = libxsmm_dispatch_meltw_unary(bn, mBlocks, &ld_vector, &ld_vector, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);

#if !defined(_OPENMP)
  float *const aux = (float*)libxsmm_aligned_scratch((3 * bm * bn) * sizeof(float), 0/*auto-alignment*/);
#else
  float *const aux = (float*)libxsmm_aligned_scratch((3 * bm * bn) * sizeof(float) * omp_get_max_threads(), 0/*auto-alignment*/);
# pragma omp parallel
#endif
  {
    int imin, im, in, ii, jj;
    float reverse_m = (float)(1.0 / m);
#if defined(__AVX512F__)
    __m512 minus_ones = _mm512_set1_ps(-1.f);
    __m512 scale      = _mm512_set1_ps(reverse_m);
#endif
#if defined(_OPENMP)
    const int ltid = omp_get_thread_num();
#else
    const int ltid = 0;
#endif
    const int work_mn = nBlocks * mBlocks;
    const int chunksize_mn = (work_mn % threads == 0) ? (work_mn /threads) : ((work_mn / threads) + 1);
    const int thr_begin_mn = (ltid * chunksize_mn < work_mn) ? (ltid * chunksize_mn) : work_mn;
    const int thr_end_mn = ((ltid + 1) * chunksize_mn < work_mn) ? ((ltid + 1) * chunksize_mn) : work_mn;

    const int work_n = nBlocks;
    const int chunksize_n = (work_n % threads == 0) ? (work_n /threads) : ((work_n / threads) + 1);
    const int thr_begin_n = (ltid * chunksize_n < work_n) ? (ltid * chunksize_n) : work_n;
    const int thr_end_n = ((ltid + 1) * chunksize_n < work_n) ? ((ltid + 1) * chunksize_n) : work_n;

    const int work_m = mBlocks;
    const int chunksize_m = (work_m % threads == 0) ? (work_m /threads) : ((work_m / threads) + 1);
    const int thr_begin_m = (ltid * chunksize_m < work_m) ? (ltid * chunksize_m) : work_m;
    const int thr_end_m = ((ltid + 1) * chunksize_m < work_m) ? ((ltid + 1) * chunksize_m) : work_m;

    libxsmm_meltw_unary_param reduce_rows_params, reduce_cols_params;

    for (imin = thr_begin_mn; imin < thr_end_mn; imin++) {
      float *const tmp  = aux + bm*bn * (ltid*3 + 0); /* aux block for db */
      float *const tmp2 = aux + bm*bn * (ltid*3 + 1); /* aux block for ds */
      float *const tmp3 = aux + bm*bn * (ltid*3 + 2); /* aux block for dgamma */
      in = imin / mBlocks;
      im = imin % mBlocks;

#if defined(__AVX512F__)
      /* Prepare blocks for reductions */
      for (jj = 0; jj < bn; jj++) {
        __m512 vrstd   = _mm512_set1_ps(LIBXSMM_VLA_ACCESS(2, rstd, in, jj, bn));
        __m512 vmean   = _mm512_set1_ps(LIBXSMM_VLA_ACCESS(2, mean, in, jj, bn));
        __m512 vb      = _mm512_mul_ps(vrstd, _mm512_mul_ps(minus_ones, vmean));
        for (ii = 0; ii < bm-15; ii+=16) {
          __m512 vgamma = _mm512_loadu_ps((float*)&LIBXSMM_VLA_ACCESS(2, gamma, im, ii, bm));
          __m512 vdY    = _mm512_loadu_ps((float*)&LIBXSMM_VLA_ACCESS(4, dY, in, im, jj, ii, mBlocks, bn, bm));
          __m512 vX     = _mm512_loadu_ps((float*)&LIBXSMM_VLA_ACCESS(4,  X, in, im, jj, ii, mBlocks, bn, bm));
          __m512 vaux   = _mm512_fmadd_ps(vrstd, vX, vb);
          __m512 vtmp   = _mm512_mul_ps(vgamma, vdY);
          _mm512_storeu_ps((float*)tmp+jj*bm+ii, vtmp);
          _mm512_storeu_ps((float*)tmp2+jj*bm+ii, _mm512_mul_ps(vtmp, vX));
          _mm512_storeu_ps((float*)tmp3+jj*bm+ii, _mm512_mul_ps(vdY, vaux));
        }
        if (ii < bm) {
          int rem = bm - ii;
          __mmask16 mask = (1 << rem) - 1;
          __m512 vgamma = _mm512_maskz_loadu_ps(mask, (float*)&LIBXSMM_VLA_ACCESS(2, gamma, im, ii, bm));
          __m512 vdY    = _mm512_maskz_loadu_ps(mask, (float*)&LIBXSMM_VLA_ACCESS(4, dY, in, im, jj, ii, mBlocks, bn, bm));
          __m512 vX     = _mm512_maskz_loadu_ps(mask, (float*)&LIBXSMM_VLA_ACCESS(4,  X, in, im, jj, ii, mBlocks, bn, bm));
          __m512 vaux   = _mm512_fmadd_ps(vrstd, vX, vb);
          __m512 vtmp   = _mm512_mul_ps(vgamma, vdY);
          _mm512_mask_storeu_ps((float*)tmp+jj*bm+ii, mask, vtmp);
          _mm512_mask_storeu_ps((float*)tmp2+jj*bm+ii, mask, _mm512_mul_ps(vtmp, vX));
          _mm512_mask_storeu_ps((float*)tmp3+jj*bm+ii, mask, _mm512_mul_ps(vdY, vaux));
        }
      }
#endif

      /* Now perform reductions */
      reduce_rows_params.in.primary    = tmp;
      reduce_rows_params.out.primary = &LIBXSMM_VLA_ACCESS(3, db_aux, in, im, 0, mBlocks, bn);
      reduce_rows_kernel(&reduce_rows_params);

      reduce_rows_params.in.primary    = tmp2;
      reduce_rows_params.out.primary = &LIBXSMM_VLA_ACCESS(3, ds_aux, in, im, 0, mBlocks, bn);
      reduce_rows_kernel(&reduce_rows_params);

      reduce_cols_params.in.primary    = (float*)&LIBXSMM_VLA_ACCESS(4, dY, in, im, 0, 0, mBlocks, bn, bm);
      reduce_cols_params.out.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_aux, im, in, 0, nBlocks, bm);
      reduce_cols_kernel(&reduce_cols_params);

      reduce_cols_params.in.primary    = tmp3;
      reduce_cols_params.out.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_aux, im, in, 0, nBlocks, bm);
      reduce_cols_kernel(&reduce_cols_params);
    }

#pragma omp barrier

    /* Second level of reductions */
    for (in = thr_begin_n; in < thr_end_n; in++) {
      reduce_cols_params.in.primary    = &LIBXSMM_VLA_ACCESS(3, db_aux, in, 0, 0, mBlocks, bn);
      reduce_cols_params.out.primary = &LIBXSMM_VLA_ACCESS(2, db, in, 0, bn);
      reduce_cols_kernel3(&reduce_cols_params);
      reduce_cols_params.in.primary    = &LIBXSMM_VLA_ACCESS(3, ds_aux, in, 0, 0, mBlocks, bn);
      reduce_cols_params.out.primary = &LIBXSMM_VLA_ACCESS(2, ds, in, 0, bn);
      reduce_cols_kernel3(&reduce_cols_params);
    }

    for (im = thr_begin_m; im < thr_end_m; im++) {
      reduce_cols_params.in.primary    = &LIBXSMM_VLA_ACCESS(3, dbeta_aux, im, 0, 0, nBlocks, bm);
      reduce_cols_params.out.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, im, 0, bm);
      reduce_cols_kernel2(&reduce_cols_params);
      reduce_cols_params.in.primary    = &LIBXSMM_VLA_ACCESS(3, dgamma_aux, im, 0, 0, nBlocks, bm);
      reduce_cols_params.out.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, im, 0, bm);
      reduce_cols_kernel2(&reduce_cols_params);
    }

#pragma omp barrier

    /* Calculate auxiliary b/c vectors -- overwritten on db/ds */
    for (in = thr_begin_n; in < thr_end_n; in++) {
#if defined(__AVX512F__)
      for (ii = 0; ii < bn-15; ii+=16) {
        __m512 vmean  = _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(2, mean, in, ii, bn));
        __m512 vrstd  = _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(2, rstd, in, ii, bn));
        __m512 vdb    = _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(2, db, in, ii, bn));
        __m512 vds    = _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(2, ds, in, ii, bn));
        __m512 ascale = _mm512_mul_ps(vrstd, scale);
        __m512 vrstd3 = _mm512_mul_ps(_mm512_mul_ps(vrstd, vrstd), ascale);
        __m512 vb     = _mm512_mul_ps(_mm512_fmsub_ps(vdb, vmean, vds), vrstd3);
        __m512 vc     = _mm512_sub_ps(_mm512_mul_ps(_mm512_mul_ps(minus_ones, vb), vmean), _mm512_mul_ps(vdb, ascale));
        _mm512_storeu_ps((float*)&LIBXSMM_VLA_ACCESS(2, db, in, ii, bn), vb);
        _mm512_storeu_ps((float*)&LIBXSMM_VLA_ACCESS(2, ds, in, ii, bn), vc);
      }
      if (ii < bn) {
        int rem = bn - ii;
        __mmask16 mask = (1 << rem) - 1;
        __m512 vmean  = _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(2, mean, in, ii, bn));
        __m512 vrstd  = _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(2, rstd, in, ii, bn));
        __m512 vdb    = _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(2, db, in, ii, bn));
        __m512 vds    = _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(2, ds, in, ii, bn));
        __m512 ascale = _mm512_mul_ps(vrstd, scale);
        __m512 vrstd3 = _mm512_mul_ps(_mm512_mul_ps(vrstd, vrstd), ascale);
        __m512 vb     = _mm512_mul_ps(_mm512_fmsub_ps(vdb, vmean, vds), vrstd3);
        __m512 vc     = _mm512_sub_ps(_mm512_mul_ps(_mm512_mul_ps(minus_ones, vb), vmean), _mm512_mul_ps(vdb, ascale));
        _mm512_mask_storeu_ps((float*)&LIBXSMM_VLA_ACCESS(2, db, in, ii, bn), mask, vb);
        _mm512_mask_storeu_ps((float*)&LIBXSMM_VLA_ACCESS(2, ds, in, ii, bn), mask, vc);
      }
#endif
    }

#pragma omp barrier

    /* Final computation of dX  */
    for (imin = thr_begin_mn; imin < thr_end_mn; imin++) {
      in = imin / mBlocks;
      im = imin % mBlocks;
#if defined(__AVX512F__)
      for (jj = 0; jj < bn; jj++) {
        __m512 va   = _mm512_set1_ps(LIBXSMM_VLA_ACCESS(2, rstd, in, jj, bn));
        __m512 vb   = _mm512_set1_ps(LIBXSMM_VLA_ACCESS(2, db, in, jj, bn));
        __m512 vc   = _mm512_set1_ps(LIBXSMM_VLA_ACCESS(2, ds, in, jj, bn));
        for (ii = 0; ii < bm-15; ii+=16) {
          __m512 vgamma = _mm512_loadu_ps((float*)&LIBXSMM_VLA_ACCESS(2, gamma, im, ii, bm));
          __m512 vdY    = _mm512_loadu_ps((float*)&LIBXSMM_VLA_ACCESS(4, dY, in, im, jj, ii, mBlocks, bn, bm));
          __m512 vX     = _mm512_loadu_ps((float*)&LIBXSMM_VLA_ACCESS(4,  X, in, im, jj, ii, mBlocks, bn, bm));
          __m512 vaux1  = _mm512_fmadd_ps(vb, vX, vc);
          __m512 vaux2  = _mm512_mul_ps(va, _mm512_mul_ps(vdY, vgamma));
          _mm512_storeu_ps((float*)&LIBXSMM_VLA_ACCESS(4, dX, in, im, jj, ii, mBlocks, bn, bm), _mm512_add_ps(vaux1, vaux2));
        }
        if (ii < bm) {
          int rem = bm - ii;
          __mmask16 mask = (1 << rem) - 1;
          __m512 vgamma = _mm512_maskz_loadu_ps(mask, (float*)&LIBXSMM_VLA_ACCESS(2, gamma, im, ii, bm));
          __m512 vdY    = _mm512_maskz_loadu_ps(mask, (float*)&LIBXSMM_VLA_ACCESS(4, dY, in, im, jj, ii, mBlocks, bn, bm));
          __m512 vX     = _mm512_maskz_loadu_ps(mask, (float*)&LIBXSMM_VLA_ACCESS(4,  X, in, im, jj, ii, mBlocks, bn, bm));
          __m512 vaux1  = _mm512_fmadd_ps(vb, vX, vc);
          __m512 vaux2  = _mm512_mul_ps(va, _mm512_mul_ps(vdY, vgamma));
          _mm512_mask_storeu_ps((float*)&LIBXSMM_VLA_ACCESS(4, dX, in, im, jj, ii, mBlocks, bn, bm), mask, _mm512_add_ps(vaux1, vaux2));
        }
      }
#endif
    }

#pragma omp barrier
  }

  libxsmm_free(scratch);
  libxsmm_free(aux);
}


int main(int argc, char* argv[])
{

  unsigned int m = 64, n = 64, iters = 10000, k = 0;
  libxsmm_blasint ld_in = 64, ld_vector = 64, block_size = 64;

  float  *sinp, *gamma, *beta, *sout, *sout_nc, *mean_data, *rstd_data, *sout_ref, *mean_data_ref, *rstd_data_ref, *bias_aux, *mean_rstd_data;
  float  *dY_ref, *X_ref, *mean_ref, *rstd_ref, *gamma_ref, *dX_ref, *dgamma_ref, *dbeta_ref;
  float  *dY_bwd, *X_bwd, *dX_bwd, *dgamma_bwd, *dbeta_bwd, *dX_bwd_nc;

  libxsmm_matdiff_info norms_out, norms_mean, norms_rstd, norms_dx, norms_dbeta, norms_dgamma;
  unsigned long long l_start, l_end;
  double l_total = 0, l_total2 = 0;

#if 0
  libxsmm_meltw_redu_flags jit_reduce_flags = LIBXSMM_MELTW_FLAG_REDUCE_NONE;
  libxsmm_meltwfunction_reduce reduce_kernel;
#endif
  libxsmm_meltw_scal_flags jit_scalemean_flags = 0;
  libxsmm_meltwfunction_scale scalemean_kernel;
  libxsmm_meltw_scal_flags jit_scaleout_flags = 0;
  libxsmm_meltwfunction_scale scaleout_kernel;

  libxsmm_init();

  libxsmm_matdiff_clear(&norms_out);
  libxsmm_matdiff_clear(&norms_mean);
  libxsmm_matdiff_clear(&norms_rstd);
  libxsmm_matdiff_clear(&norms_dx);
  libxsmm_matdiff_clear(&norms_dbeta);
  libxsmm_matdiff_clear(&norms_dgamma);


  if ( argc > 1 ) m = atoi(argv[1]);
  if ( argc > 2 ) n = atoi(argv[2]);
  if ( argc > 3 ) iters = atoi(argv[3]);
  if ( argc > 4 ) block_size = atoi(argv[4]);

  libxsmm_init();

  ld_in = m;
  n = LIBXSMM_MAX(n,1);
  ld_vector = n;
  ld_in = LIBXSMM_MAX(ld_in,(libxsmm_blasint)m);

  /* Allocate arrays  */
  sinp      = (float*) malloc(ld_in*n*sizeof(float));
  gamma     = (float*) malloc(m*sizeof(float) );
  beta      = (float*) malloc(m*sizeof(float) );
  sout      = (float*) malloc(ld_in*n*sizeof(float) );
  sout_nc   = (float*) malloc(ld_in*n*sizeof(float) );
  mean_rstd_data = (float*) malloc(2*n*sizeof(float) );
  mean_data = (float*) mean_rstd_data;
  rstd_data = (float*) mean_rstd_data + n;

  dY_ref    = (float*) malloc(m*n*sizeof(float));
  dY_bwd    = (float*) malloc(m*n*sizeof(float));
  X_ref     = (float*) malloc(m*n*sizeof(float));
  X_bwd     = (float*) malloc(m*n*sizeof(float));
  mean_ref  = (float*) malloc(n*sizeof(float));
  rstd_ref  = (float*) malloc(n*sizeof(float));
  gamma_ref = (float*) malloc(m*sizeof(float));
  dX_ref    = (float*) malloc(m*n*sizeof(float));
  dX_bwd    = (float*) malloc(m*n*sizeof(float));
  dX_bwd_nc = (float*) malloc(m*n*sizeof(float));
  dgamma_ref= (float*) malloc(m*sizeof(float));
  dgamma_bwd= (float*) malloc(m*sizeof(float));
  dbeta_ref = (float*) malloc(m*sizeof(float));
  dbeta_bwd = (float*) malloc(m*sizeof(float));

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

  sfill_matrix ( dY_ref, ld_in, m, n );
  matrix_copy_NC_to_NCNC( dY_ref, dY_bwd, 1, n, m, block_size, block_size );
  sfill_matrix ( X_ref, ld_in, m, n );
  matrix_copy_NC_to_NCNC( X_ref, X_bwd, 1, n, m, block_size, block_size );
  sfill_matrix ( mean_ref, n, n, 1 );
  sfill_matrix ( rstd_ref, n, n, 1 );
  sfill_matrix ( gamma_ref, m, m, 1 );

  /* Calculate reference results... */
  naive_layernorm(m, n, ld_in, sinp, gamma, beta, sout_ref, mean_data_ref, rstd_data_ref);
  naive_layernorm_bwd(m, n, ld_in, dY_ref, X_ref, mean_ref, rstd_ref, gamma_ref, dX_ref, dgamma_ref, dbeta_ref);

#if 0
  /* Generate JITED kernels for optimized code */
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_REDUCE_ROWS | LIBXSMM_MELTW_FLAG_REDUCE_OP_ADD | LIBXSMM_MELTW_FLAG_REDUCE_ELTS | LIBXSMM_MELTW_FLAG_REDUCE_ELTS_SQUARED;
  printf("JITing reduce kernel... \n");
  reduce_kernel = libxsmm_dispatch_meltw_reduce(m, n, &ld_in, &ld_in, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, 0);

  jit_scalemean_flags = LIBXSMM_MELTW_FLAG_SCALE_ROWS | LIBXSMM_MELTW_FLAG_SCALE_MULT;
  printf("JITing mean-scale kernel... \n");
  scalemean_kernel = libxsmm_dispatch_meltw_scale(n, 1, &ld_vector, &ld_vector, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_scalemean_flags, 0);

  jit_scaleout_flags = LIBXSMM_MELTW_FLAG_SCALE_ROWS_COLS | LIBXSMM_MELTW_FLAG_SCALE_MULT | LIBXSMM_MELTW_FLAG_SCALE_ADD_BIAS;
  printf("JITing scaling kernel for output... \n");
  scaleout_kernel = libxsmm_dispatch_meltw_scale(m, n, &ld_in, &ld_in, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_scaleout_flags, 0);
#endif

  /* Calculate blocked results... */
#if 0
  optimized_layernorm(m, n, ld_in, sinp, gamma, beta, sout, mean_data, rstd_data, reduce_kernel, scalemean_kernel, scaleout_kernel, bias_aux);
#else
  matrix_copy_NC_to_NCNC( sinp,  sout, 1, n, m, block_size, block_size );
  optimized_blocked_layernorm(m, n, block_size, block_size, sout, gamma, beta, mean_data, rstd_data);
  matrix_copy_NCNC_to_NC( sout, sout_nc, 1, n, m, block_size, block_size );

  optimized_blocked_layernorm_bwd(m, n, block_size, block_size, dY_bwd, X_bwd, mean_ref, rstd_ref, gamma_ref, dX_bwd, dgamma_bwd, dbeta_bwd);
  matrix_copy_NCNC_to_NC( dX_bwd, dX_bwd_nc, 1, n, m, block_size, block_size );
#endif

  /* compare */
  printf("##########################################\n");
  printf("#   Correctness FWD - Output             #\n");
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
  printf("#   Correctness FWD - Mean               #\n");
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
  printf("#   Correctness FWD - Rstd               #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_rstd, LIBXSMM_DATATYPE_F32, n, 1, rstd_data_ref, rstd_data, 0, 0);
  printf("L1 reference  : %.25g\n", norms_rstd.l1_ref);
  printf("L1 test       : %.25g\n", norms_rstd.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_rstd.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_rstd.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_rstd.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_rstd.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_rstd.normf_rel);

  /* compare */
  printf("##########################################\n");
  printf("#   Correctness BWD - dX                  #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_dx, LIBXSMM_DATATYPE_F32, ld_in*n, 1, dX_ref, dX_bwd_nc, 0, 0);
  printf("L1 reference  : %.25g\n", norms_dx.l1_ref);
  printf("L1 test       : %.25g\n", norms_dx.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_dx.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_dx.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_dx.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_dx.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_dx.normf_rel);

  /* compare */
  printf("##########################################\n");
  printf("#   Correctness BWD - dbeta              #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_dbeta, LIBXSMM_DATATYPE_F32, m, 1, dbeta_ref, dbeta_bwd, 0, 0);
  printf("L1 reference  : %.25g\n", norms_dbeta.l1_ref);
  printf("L1 test       : %.25g\n", norms_dbeta.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_dbeta.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_dbeta.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_dbeta.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_dbeta.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_dbeta.normf_rel);

  /* compare */
  printf("##########################################\n");
  printf("#   Correctness BWD - dgamma             #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_dgamma, LIBXSMM_DATATYPE_F32, m, 1, dgamma_ref, dgamma_bwd, 0, 0);
  printf("L1 reference  : %.25g\n", norms_dgamma.l1_ref);
  printf("L1 test       : %.25g\n", norms_dgamma.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_dgamma.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_dgamma.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_dgamma.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_dgamma.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_dgamma.normf_rel);


  l_start = libxsmm_timer_tick();
  /* Calculate reference results...  */
  for (k = 0; k < iters; k++) {
    naive_layernorm(m, n, ld_in, sinp, gamma, beta, sout_ref, mean_data_ref, rstd_data_ref);
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Reference fwd time = %.5g\n", ((double)(l_total)));

  l_start = libxsmm_timer_tick();
  for (k = 0; k < iters; k++) {
#if 1
    optimized_blocked_layernorm(m, n, block_size, block_size, sout, gamma, beta, mean_data, rstd_data);
#else
    optimized_layernorm(m, n, ld_in, sinp, gamma, beta, sout, mean_data, rstd_data, reduce_kernel, scalemean_kernel, scaleout_kernel, bias_aux);
#endif
  }
  l_end = libxsmm_timer_tick();
  l_total2 = libxsmm_timer_duration(l_start, l_end);
  printf("Optimized fwd time = %.5g\n", ((double)(l_total2)));
  printf("Speedup fwd is = %.5g\n", ((double)(l_total/l_total2)));

  l_start = libxsmm_timer_tick();
  /* Calculate reference results...  */
  for (k = 0; k < iters; k++) {
    naive_layernorm_bwd(m, n, ld_in, dY_ref, X_ref, mean_ref, rstd_ref, gamma_ref, dX_ref, dgamma_ref, dbeta_ref);
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Reference bwd time = %.5g\n", ((double)(l_total)));

  l_start = libxsmm_timer_tick();
  for (k = 0; k < iters; k++) {
    optimized_blocked_layernorm_bwd(m, n, block_size, block_size, dY_bwd, X_bwd, mean_ref, rstd_ref, gamma_ref, dX_bwd, dgamma_bwd, dbeta_bwd);
  }
  l_end = libxsmm_timer_tick();
  l_total2 = libxsmm_timer_duration(l_start, l_end);
  printf("Optimized bwd time = %.5g\n", ((double)(l_total2)));
  printf("Speedup bwd is = %.5g\n", ((double)(l_total/l_total2)));
  /* Free allocated arrays */
  free(sinp);
  free(gamma);
  free(beta);
  free(sout);
  free(mean_rstd_data);
  free(mean_data_ref);
  free(rstd_data_ref);
  free(sout_ref);
  free(bias_aux);
  free(dY_ref);
  free(X_ref);
  free(mean_ref);
  free(rstd_ref);
  free(gamma_ref);
  free(dX_ref);
  free(dgamma_ref);
  free(dbeta_ref);
  free(dY_bwd);
  free(X_bwd);
  free(dX_bwd);
  free(dgamma_bwd);
  free(dbeta_bwd);
  free(dX_bwd_nc);

  return EXIT_SUCCESS;
}

