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
#include <libxsmm_sync.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define REFACTORED_FWD
#define REFACTORED_BWD
//#define ONLY_FWD

#define COMPUTE_FP64_REFERENCE

#define TRUE_PARALLEL_BWD
#define TRUE_PARALLEL_FWD

//#define DEBUGGING

#if defined(REFACTORED_BWD) || defined(REFACTORED_FWD)

/* include c-based dnn library */
#include "../common/dnn_common.h"

#ifdef COMPUTE_FP64_REFERENCE
inline void buf_extend_fp32_to_fp64 (const float *src, double *dst, int length)
{
  int i;
  for (i = 0; i < length; i++)
    dst[i] = (double)(src[i]);
}

inline void buf_truncate_fp64_to_fp32 (const double *src, float *dst, int length)
{
  int i;
  for (i = 0; i < length; i++)
    dst[i] = (float)(src[i]);
}


LIBXSMM_INLINE void naive_fusedbatchnorm_fp_fp64(naive_fusedbatchnorm_t* param, const double* input_ptr, double* output_ptr, const double* input_add_ptr,
                                     const double* beta_ptr, const double* gamma_ptr, double eps, double* expectval_ptr, double* rcpstddev_ptr, double* variance_ptr)
{
  const int nImg = param->N;
  const int nFm = param->C;
  const int ifh = param->H;
  const int ifw = param->W;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  const int ofh = ifh/sh;
  const int ofw = ifw/sw;
  const double nhw = (double)(nImg * ifh * ifw);
  const double recp_nhw = 1.0f/nhw;
  //const float sqrt_eps = 1e-7f;

  int img, fm, hi, wi, ho, wo;

  LIBXSMM_VLA_DECL(4, const double, input,     input_ptr,     nFm, ifh, ifw);
  LIBXSMM_VLA_DECL(4, const double, input_add, input_add_ptr, nFm, ifh, ifw);
  LIBXSMM_VLA_DECL(4,       double, output,    output_ptr,    nFm, ofh, ofw);

  if ( param->norm_type == 0 ) {
#if defined(_OPENMP)
    LIBXSMM_OMP_VAR(wi); LIBXSMM_OMP_VAR(hi);
#   pragma omp parallel for private(img, fm, hi, wi)
#endif
    for (fm = 0; fm < nFm; fm++) {
      double ch_sum = 0.0f;
      double ch_sumsq = 0.0f;
      double tbmean = 0.0f;
      double tbmeansq = 0.0f;
      double tsqbmean = 0.0f;
      double tbrstd = 0.0f;
      double tvariance = 0.0f;

      for ( img = 0; img < nImg; img++ ) {
        for ( hi = 0; hi < ifh; hi++ ) {
          for ( wi = 0; wi < ifw; wi++ ) {
            const double input_val = LIBXSMM_VLA_ACCESS(4, input, img, fm, hi, wi, nFm, ifh, ifw);
            ch_sum   += input_val;
            ch_sumsq += (input_val * input_val);
          }
        }
      }

      tbmean = recp_nhw * ch_sum;
      tbmeansq  = tbmean * tbmean;
      tsqbmean = recp_nhw * ch_sumsq;
      tvariance = tsqbmean - tbmeansq;
      tbrstd = (double)(1.0/sqrt(tvariance + eps));//sqrt_eps));
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
          const double  input_val     =  LIBXSMM_VLA_ACCESS(4, input,     img, fm, hi, wi, nFm, ifh, ifw);
          const double  input_add_val =  LIBXSMM_VLA_ACCESS(4, input_add, img, fm, hi, wi, nFm, ifh, ifw);
                double* output_ptr2   = &LIBXSMM_VLA_ACCESS(4, output,    img, fm, ho, wo, nFm, ofh, ofw);

          /* BN + scale (gamma, beta) */
          double o = gamma_ptr[fm]*(input_val - expectval_ptr[fm])*rcpstddev_ptr[fm] + beta_ptr[fm];
          /* Eltwise */
          if ( (param->fuse_type == 2) || (param->fuse_type == 3) || (param->fuse_type == 5) ) {
            o += input_add_val;
          }
          /* ReLU */
          if ( (param->fuse_type == 1) || (param->fuse_type == 3) || (param->fuse_type == 4) || (param->fuse_type == 5) ) {
            o = ( o < 0.0f ) ? 0.0f : o;
          }
          *output_ptr2 = o;
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_fusedbatchnorm_bp_fp64(naive_fusedbatchnorm_t* param, const double* input_ptr, double* dinput_ptr, const double* output_ptr, double* doutput_ptr, double* dinput_add_ptr,
                                     const double* beta_ptr, double* del_beta_ptr, const double* gamma_ptr, double* del_gamma_ptr,
                                     const double* expectval_ptr, const double* rcpstddev_ptr)
{
  const int nImg = param->N;
  const int nFm = param->C;
  const int ifh = param->H;
  const int ifw = param->W;
  const int sh = param->stride_h;
  const int sw = param->stride_w;
  const int ofh = ifh/sh;
  const int ofw = ifw/sw;
  const double nhw = (double)(nImg * ifh * ifw);
  const double recp_nhw = 1.0f/nhw;

  int img, fm, hi, wi, ho, wo;

  LIBXSMM_VLA_DECL(4, const double, input,      input_ptr,      nFm, ifh, ifw);
  LIBXSMM_VLA_DECL(4,       double, dinput,     dinput_ptr,     nFm, ifh, ifw);
  LIBXSMM_VLA_DECL(4,       double, dinput_add, dinput_add_ptr, nFm, ifh, ifw);
  LIBXSMM_VLA_DECL(4, const double, output,     output_ptr,     nFm, ofh, ofw);
  LIBXSMM_VLA_DECL(4,       double, doutput,    doutput_ptr,    nFm, ofh, ofw);
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
                  double* del_input_add_ptr = &LIBXSMM_VLA_ACCESS(4, dinput_add, img, fm, hi, wi, fm, ifh, ifw);
            const double  output_val        =  LIBXSMM_VLA_ACCESS(4,     output, img, fm, ho, wo, fm, ofh, ofw);
            const double  input_val         =  LIBXSMM_VLA_ACCESS(4,      input, img, fm, hi, wi, fm, ifh, ifw);
                  double* del_output_ptr    = &LIBXSMM_VLA_ACCESS(4,    doutput, img, fm, ho, wo, fm, ofh, ofw);

            /* ReLU */
            if ( (param->fuse_type == 1) || (param->fuse_type == 3) || (param->fuse_type == 4) || (param->fuse_type == 5) ) {
              *del_output_ptr    = (output_val == 0) ? 0 : *del_output_ptr;
            }
            /* elementwise */
            if ( (param->fuse_type == 2) || (param->fuse_type == 3) || (param->fuse_type == 5) ) {
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
                double* del_input_ptr  = &LIBXSMM_VLA_ACCESS(4,     dinput, img, fm, hi, wi, fm, ifh, ifw);
          const double  input_val      =  LIBXSMM_VLA_ACCESS(4,      input, img, fm, hi, wi, fm, ifh, ifw);
          const double  del_output_val =  LIBXSMM_VLA_ACCESS(4,    doutput, img, fm, ho, wo, fm, ofh, ofw);

          *del_input_ptr = gamma_ptr[fm] * rcpstddev_ptr[fm] * recp_nhw * (nhw * del_output_val -
                    (del_beta_ptr[fm] + (input_val - expectval_ptr[fm]) * del_gamma_ptr[fm] * rcpstddev_ptr[fm]));
        }
      }
    }
  }
}


#endif // for #ifdef COMPUTE_FP64_REFERENCE
#endif // for either REFACTORED_FWD or REFACTORED_BWD


#ifdef REFACTORED_FWD

//void tpp_batchnorm_fwd_fp32(long N, long CP, long HW, long CB, long num_HW_blocks, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout, float eps,
//                          libxsmm_matrix_eqn_function func10, libxsmm_meltwfunction_unary reduce_HW_kernel, libxsmm_meltwfunction_unary all_zero_kernel, libxsmm_meltwfunction_binary add_kernel, libxsmm_meltwfunction_unary copy_kernel) {

typedef struct my_bn_fwd_config {
  libxsmm_blasint N;
  libxsmm_blasint CP;
  libxsmm_blasint HW;
  libxsmm_blasint CB;
  libxsmm_blasint num_HW_blocks;
  libxsmm_blasint threads;

  libxsmm_barrier* barrier;

  libxsmm_matrix_eqn_function func10;
  libxsmm_meltwfunction_unary reduce_HW_kernel;
  libxsmm_meltwfunction_unary all_zero_kernel;
  libxsmm_meltwfunction_binary add_kernel;
  libxsmm_meltwfunction_unary copy_kernel;
  //my_eltwise_fuse fuse_type;
} my_bn_fwd_config;

#endif

#ifndef ONLY_FWD
#ifdef REFACTORED_BWD
//void tpp_batchnorm_bwd_fp32(int N, int CP, int HW, int CB, int num_HW_blocks, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta,
//    libxsmm_matrix_eqn_function dgamma_func, libxsmm_matrix_eqn_function dbeta_func, libxsmm_matrix_eqn_function din_func, float eps,
//    libxsmm_meltwfunction_unary all_zero_kernel, libxsmm_meltwfunction_binary add_kernel, libxsmm_meltwfunction_unary copy_kernel) {

typedef struct my_bn_bwd_config {
  libxsmm_blasint N;
  libxsmm_blasint CP;
  libxsmm_blasint HW;
  libxsmm_blasint CB;
  libxsmm_blasint num_HW_blocks;
  libxsmm_blasint threads;

  libxsmm_barrier* barrier;

  libxsmm_matrix_eqn_function dgamma_func;
  libxsmm_matrix_eqn_function dbeta_func;
  libxsmm_matrix_eqn_function din_func;
  libxsmm_meltwfunction_unary all_zero_kernel;
  libxsmm_meltwfunction_binary add_kernel;
  libxsmm_meltwfunction_unary copy_kernel;
  //my_eltwise_fuse fuse_type;
} my_bn_bwd_config;
#endif
#endif

#ifdef REFACTORED_FWD

//void tpp_batchnorm_fwd_fp32(long N, long CP, long HW, long CB, long num_HW_blocks, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout, float eps,
//                          libxsmm_matrix_eqn_function func10, libxsmm_meltwfunction_unary reduce_HW_kernel, libxsmm_meltwfunction_unary all_zero_kernel, libxsmm_meltwfunction_binary add_kernel, libxsmm_meltwfunction_unary copy_kernel) {

my_bn_fwd_config setup_my_bn_fwd(libxsmm_blasint N, libxsmm_blasint CP, libxsmm_blasint HW, libxsmm_blasint CB,
                                 libxsmm_blasint num_HW_blocks, libxsmm_blasint threads /*, my_eltwise_fuse fuse_type*/ ) {
  my_bn_fwd_config res;
  libxsmm_blasint ldo = CB;
  libxsmm_blasint ld  = CB;
  libxsmm_blasint tmp_ld, tmp_ld2;
  libxsmm_blasint my_eqn10;

  libxsmm_meltw_unary_flags jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type  unary_type;

  libxsmm_datatype  in_dt  = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype  out_dt = LIBXSMM_DATATYPE_F32;

  /* setting up some handle values */
  res.N  = N;
  res.CP = CP;
  res.HW = HW;
  res.CB = CB;
  res.num_HW_blocks = num_HW_blocks;
  res.threads = threads;
  //res.fuse_type = fuse_type;

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* TPP creation */

  /* Eltwise TPPs  */
  res.all_zero_kernel = libxsmm_dispatch_meltw_unary(res.CB, 1, NULL, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  if ( res.all_zero_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd all_zero_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.copy_kernel = libxsmm_dispatch_meltw_unary(CB, 1, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
  if ( res.copy_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd copy_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.add_kernel = libxsmm_dispatch_meltw_binary(res.CB, 1, &ldo, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_MELTW_TYPE_BINARY_ADD);
  if ( res.add_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd add_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* TPPs for reducing X and X2 in HW*/
  tmp_ld = CB;

  unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD;
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  res.reduce_HW_kernel = libxsmm_dispatch_meltw_unary(res.CB, res.HW/res.num_HW_blocks, &ld, &tmp_ld, in_dt, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);
  if ( res.reduce_HW_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd reduce_HW_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* TPP for scaling */
  ld = CB;
  tmp_ld = 1;
  tmp_ld2 = 1;

  my_eqn10 = libxsmm_matrix_eqn_create();                                                            /* y = (s*x + b)*gamma + beta */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn10, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn10, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, res.CB, res.HW/res.num_HW_blocks, ld, 0, 0, in_dt );   /* x = [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, res.CB, 1, tmp_ld, 1, 0, LIBXSMM_DATATYPE_F32 );       /* s = [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, res.CB, 1, tmp_ld, 2, 0, LIBXSMM_DATATYPE_F32 );       /* b = [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, res.CB, 1, tmp_ld2, 3, 0, in_dt );                     /* gamma = [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, res.CB, 1, tmp_ld2, 4, 0, in_dt );                     /* beta = [CB] */
  /* libxsmm_matrix_eqn_tree_print( my_eqn10 ); */
  /* libxsmm_matrix_eqn_rpn_print( my_eqn10 ); */

  res.func10 = libxsmm_dispatch_matrix_eqn( res.CB, res.HW/res.num_HW_blocks, &ld, out_dt, my_eqn10 );   /* y = [HW, CB] */
  if ( res.func10 == NULL) {
    fprintf( stderr, "JIT for TPP fwd func10 (eqn10) failed. Bailing...!\n");
    exit(-1);
  }

  return res;
}

//void tpp_batchnorm_fwd_fp32(long N, long CP, long HW, long CB, long num_HW_blocks, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout, float eps,
//                          libxsmm_matrix_eqn_function func10, libxsmm_meltwfunction_unary reduce_HW_kernel, libxsmm_meltwfunction_unary all_zero_kernel, libxsmm_meltwfunction_binary add_kernel, libxsmm_meltwfunction_unary copy_kernel) {


// FIXME: Remove omp pragmas from the copy-pasted code, need to use tid to have a reasonable kernel
// FIXME: Set const modifiers properly? Cannot put const as then reduce_HW_params.in.primary    = &LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB); complains
void my_bn_fwd_exec( my_bn_fwd_config cfg, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout, float eps, int start_tid, int my_tid, void *scratch ) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint HW = cfg.HW;
  const libxsmm_blasint CB = cfg.CB;
  const libxsmm_blasint num_HW_blocks = cfg.num_HW_blocks;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;
#ifdef TRUE_PARALLEL_FWD
  /* number of tasks that could be run in parallel for 1d blocking */
  // Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here?
  const libxsmm_blasint work_dN = CP * N;
  /* compute chunk size */
  const libxsmm_blasint chunksize_dN = (work_dN % cfg.threads == 0) ?
    (work_dN / cfg.threads) : ((work_dN / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_dN = (ltid * chunksize_dN < work_dN) ? (ltid * chunksize_dN) : work_dN;
  const libxsmm_blasint thr_end_dN = ((ltid + 1) * chunksize_dN < work_dN) ? ((ltid + 1) * chunksize_dN) : work_dN;

  /* number of tasks that could be run in parallel for 1d blocking */
  // Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here?
  const libxsmm_blasint work_C = CP;
  /* compute chunk size */
  const libxsmm_blasint chunksize_C = (work_C % cfg.threads == 0) ?
    (work_C / cfg.threads) : ((work_C / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_C = (ltid * chunksize_C < work_C) ? (ltid * chunksize_C) : work_C;
  const libxsmm_blasint thr_end_C = ((ltid + 1) * chunksize_C < work_C) ? ((ltid + 1) * chunksize_C) : work_C;
#endif

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  LIBXSMM_VLA_DECL(4, float, inp, pinp, CP, HW, CB);            /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, float, out, pout, CP, HW, CB);            /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);                /* [CP, CB] */
  LIBXSMM_VLA_DECL(2, float, beta, pbeta, CB);                  /* [CP, CB] */

  const float scale = 1.0f /((float)N * HW);

#ifdef TRUE_PARALLEL_FWD
  LIBXSMM_VLA_DECL(3, float, sum_X_X2, ((float*)scratch), CP, CB);  /* [2, CP, CB] */
  LIBXSMM_ASSUME_ALIGNED(sum_X_X2_, 64);
  const libxsmm_blasint sum_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + CP * 2 * CB), 64) - ((uintptr_t)(scratch))) / sizeof(float);
  LIBXSMM_VLA_DECL(3, float, sum_N,  ((float*)scratch) + sum_N_offset, N, CB);  /* [CP, N, CB] */
  LIBXSMM_ASSUME_ALIGNED(sum_N_, 64);
  const libxsmm_blasint sumsq_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + sum_N_offset + CP * N * CB), 64) - ((uintptr_t)(scratch))) / sizeof(float);
  LIBXSMM_VLA_DECL(3, float, sumsq_N,  ((float*)scratch) + sumsq_N_offset, N, CB);  /* [CP, N, CB] */
  LIBXSMM_ASSUME_ALIGNED(sumsq_N_, 64);
#else
  LIBXSMM_ALIGNED(float sum_X_X2[CP*2*CB], 64); // should get replaced with scratch
  LIBXSMM_ALIGNED(float sum_N[CP*N*CB], 64);
  LIBXSMM_ALIGNED(float sumsq_N[CP*N*CB], 64);
#endif

//#if 0


#ifdef TRUE_PARALLEL_FWD
  {
#else
  #pragma omp parallel
  {
#endif
    LIBXSMM_ALIGNED(float s[CB], 64);
    LIBXSMM_ALIGNED(float b[CB], 64);
    int n, cp;

#ifdef TRUE_PARALLEL_FWD
    int cpxnt;
    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      { // stupid block to keep indentation
        n  = cpxnt%N;
        cp = cpxnt/N;
#else
    #pragma omp for nowait collapse(2)
    for (cp = 0; cp < CP; cp++) {
      for(n = 0; n < N; n++){
#endif

        int hwb;

#ifdef TRUE_PARALLEL_FWD
        float *sum_ncp_ptr   = &LIBXSMM_VLA_ACCESS(3, sum_N, cp, n, 0, N, CB);
        float *sumsq_ncp_ptr = &LIBXSMM_VLA_ACCESS(3, sumsq_N, cp, n, 0, N, CB);
#else
        float *sum_ncp_ptr   = &sum_N[cp*N*CB + n*CB];
        float *sumsq_ncp_ptr = &sumsq_N[cp*N*CB + n*CB];
#endif

        libxsmm_meltw_unary_param all_zero_param;
        all_zero_param.out.primary = sum_ncp_ptr;
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = sumsq_ncp_ptr;
        cfg.all_zero_kernel(&all_zero_param);

        /* #pragma omp simd  */
        /* for (int cb = 0; cb < CB; cb++) {  */
        /*   sum_ncp_ptr[cb] = 0.0f;    */
        /*   sumsq_ncp_ptr[cb] = 0.0f;  */
        /* } */

        libxsmm_meltw_binary_param add_param;

        libxsmm_meltw_unary_param reduce_HW_params;       /*Private params and tmp array */
        LIBXSMM_ALIGNED(float lcl_sum_X_X2[2*CB], 64); // FIXME: needs to be moved to scratch memory!
        reduce_HW_params.out.primary   = lcl_sum_X_X2;                                                         /* [2*CB]  */
        for(hwb=0; hwb < num_HW_blocks; hwb++){

          reduce_HW_params.in.primary = &LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          cfg.reduce_HW_kernel(&reduce_HW_params);                                                       /* [HW, CB] -----> [2 * CB] */

          add_param.in0.primary = sum_ncp_ptr;
          add_param.in1.primary = lcl_sum_X_X2;
          add_param.out.primary = sum_ncp_ptr;
          cfg.add_kernel(&add_param);

          add_param.in0.primary = sumsq_ncp_ptr;
          add_param.in1.primary = &lcl_sum_X_X2[CB];
          add_param.out.primary = sumsq_ncp_ptr;
          cfg.add_kernel(&add_param);

          /* #pragma omp simd */
          /* for (int cb = 0; cb < CB; cb++) {  */
          /*   sum_ncp_ptr[cb] += lcl_sum_X_X2[cb];  */
          /*   sumsq_ncp_ptr[cb] += lcl_sum_X_X2[CB + cb];  */
          /* }  */
        }
      }
    }

#ifdef TRUE_PARALLEL_FWD
    libxsmm_barrier_wait(cfg.barrier, ltid); // valid replacement? what exactly ltid argument does?
#else
    #pragma omp barrier
#endif

#ifdef TRUE_PARALLEL_FWD
    for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {
#else
    #pragma omp for
    for (cp = 0; cp < CP; cp++) {
#endif

#ifdef TRUE_PARALLEL_FWD
      libxsmm_meltw_unary_param all_zero_param;
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, 0, CP, CB);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, 0, CP, CB);
      cfg.all_zero_kernel(&all_zero_param);
#else
      libxsmm_meltw_unary_param all_zero_param;
      all_zero_param.out.primary = &sum_X_X2[cp*CB];
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &sum_X_X2[CP*CB + cp*CB];
      cfg.all_zero_kernel(&all_zero_param);
#endif

      /* #pragma omp simd */
      /* for (int cb = 0; cb < CB; cb++) {  */
      /*   sum_X_X2[cp*CB + cb] = 0.0f;   */
      /*   sum_X_X2[CP*CB + (cp*CB + cb)] = 0.0f;  */
      /* } */

      libxsmm_meltw_binary_param add_param;
      int cb, ni;
      for(ni = 0; ni < N; ni++){

#ifdef TRUE_PARALLEL_FWD
        add_param.in0.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, 0, CP, CB);//sum_X_X2_[cp*CB];
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, sum_N, cp, ni, 0, N, CB);
        add_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, 0, CP, CB);//sum_X_X2_[cp*CB];
        cfg.add_kernel(&add_param);

        add_param.in0.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, 0, CP, CB);//sum_X_X2_[CP*CB + cp*CB];
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, sumsq_N, cp, ni, 0, N, CB);
        add_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, 0, CP, CB);//sum_X_X2_[CP*CB + cp*CB];
        cfg.add_kernel(&add_param);
#else
        add_param.in0.primary = &sum_X_X2[cp*CB];
        add_param.in1.primary = &sum_N[cp*N*CB + ni*CB];
        add_param.out.primary = &sum_X_X2[cp*CB];
        cfg.add_kernel(&add_param);

        add_param.in0.primary = &sum_X_X2[CP*CB + cp*CB];
        add_param.in1.primary = &sumsq_N[cp*N*CB + ni*CB];
        add_param.out.primary = &sum_X_X2[CP*CB + cp*CB];
        cfg.add_kernel(&add_param);
#endif
        /* #pragma omp simd */
        /* for (int cb = 0; cb < CB; cb++) { */
        /*   sum_X_X2[cp*CB + cb] += sum_N[cp*N*CB + n*CB + cb]; */
        /*   sum_X_X2[CP*CB + (cp*CB + cb)] += sumsq_N[cp*N*CB + n*CB + cb]; */
        /* } */
      }

      for(cb = 0; cb < CB; cb++){
#ifdef TRUE_PARALLEL_FWD
        mean[cp*CB + cb] = (LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, cb, CP, CB)) * scale;                 /* E[X] */
        var[cp*CB + cb] = ((LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, cb, CP, CB)) * scale) - (mean[cp*CB + cb]*mean[cp*CB + cb]);
#else
        mean[cp*CB + cb] = sum_X_X2[cp*CB + cb] * scale;                                                  /* E[X] */
        var[cp*CB + cb] = (sum_X_X2[CP*CB + cp*CB + cb] * scale) - (mean[cp*CB + cb]*mean[cp*CB + cb]);
#endif
      }
    }

    // FIXME: Why there was no barrier in the older implementation here?
#ifdef TRUE_PARALLEL_FWD
    libxsmm_barrier_wait(cfg.barrier, ltid); // valid replacement? what exactly ltid argument does?
#endif

#ifdef TRUE_PARALLEL_FWD
    //int cpxnt;
    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      { // stupid block to keep indentation
        n  = cpxnt%N;
        cp = cpxnt/N;
#else
    #pragma omp for nowait collapse(2)
    for (cp = 0; cp < CP; cp++){
      for(n = 0; n < N; n++){                                                             /* Parallelize over batches and CP*/
#endif

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
          cfg.func10(&eqn_param);                                                                    /* Normalization equation -> y = ((s*x + b)*gamma + beta) */
        }
      }
    }
  }
//#endif // for #if 0

  libxsmm_barrier_wait(cfg.barrier, ltid);

}

#endif // REFACTORED_FWD

#define ALIGNDOWN(N, A) ((N) & ~((A)-1))
#define USE_VECTORIZED_PATH 1

float upconvert_bf16(libxsmm_bfloat16 x) {
  union libxsmm_bfloat16_hp bf16_hp;
  bf16_hp.i[1] = x;
  bf16_hp.i[0] = 0;
  return bf16_hp.f;
}

void tpp_batchnorm_fwd_fp32(int N, int CP, int HW, int CB, int num_HW_blocks, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout, float eps,
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

#ifndef ONLY_FWD
#ifdef REFACTORED_BWD

//void tpp_batchnorm_bwd_fp32(int N, int CP, int HW, int CB, int num_HW_blocks, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta,
//    libxsmm_matrix_eqn_function dgamma_func, libxsmm_matrix_eqn_function dbeta_func, libxsmm_matrix_eqn_function din_func, float eps,
//    libxsmm_meltwfunction_unary all_zero_kernel, libxsmm_meltwfunction_binary add_kernel, libxsmm_meltwfunction_unary copy_kernel) {

my_bn_bwd_config setup_my_bn_bwd(libxsmm_blasint N, libxsmm_blasint CP, libxsmm_blasint HW, libxsmm_blasint CB,
                                 libxsmm_blasint num_HW_blocks, libxsmm_blasint threads /*, my_eltwise_fuse fuse_type*/ ) {
  my_bn_bwd_config res;
  libxsmm_blasint ldo = CB;
  libxsmm_blasint ld  = CB;
  libxsmm_blasint tmp_ld2;
  libxsmm_blasint my_eqn11, my_eqn12, my_eqn16;

  libxsmm_datatype  in_dt  = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype  out_dt = LIBXSMM_DATATYPE_F32;

  /* setting up some handle values */
  res.N  = N;
  res.CP = CP;
  res.HW = HW;
  res.CB = CB;
  res.num_HW_blocks = num_HW_blocks;
  res.threads = threads;
  //res.fuse_type = fuse_type;

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* TPP creation */

  /* Eltwise TPPs  */
  res.all_zero_kernel = libxsmm_dispatch_meltw_unary(res.CB, 1, NULL, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  if ( res.all_zero_kernel == NULL) {
    fprintf( stderr, "JIT for TPP bwd all_zero_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.copy_kernel = libxsmm_dispatch_meltw_unary(CB, 1, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
  if ( res.copy_kernel == NULL) {
    fprintf( stderr, "JIT for TPP bwd copy_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.add_kernel = libxsmm_dispatch_meltw_binary(res.CB, 1, &ldo, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_MELTW_TYPE_BINARY_ADD);
  if ( res.add_kernel == NULL) {
    fprintf( stderr, "JIT for TPP bwd add_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* TPP equations for dgamma, dbeta and din */

  ld = CB;
  tmp_ld2 = 1;

  /* dgamma function  */
  my_eqn11 = libxsmm_matrix_eqn_create();                                                       /* dgamma = ((inp *a + b) * dout) + dgamma */
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn11, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32);                   /* dgamma = ((inp *a + b) * dout) + dgamma */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn11, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);   /* [HW, CB] -> [CB] */
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn11, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32);                   /* ((inp *a + b) * dout) */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn11, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, res.CB, res.HW/res.num_HW_blocks, ld, 0, 0, in_dt );          /* inp [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, res.CB, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 );           /* a [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, res.CB, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );           /* b [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, res.CB, res.HW/res.num_HW_blocks, ld, 3, 0, in_dt );          /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, res.CB, 1, 1, 4, 0, LIBXSMM_DATATYPE_F32 );           /* dgamma [CB] */
  /* libxsmm_matrix_eqn_tree_print( my_eqn11 ); */
  /* libxsmm_matrix_eqn_rpn_print( my_eqn11 ); */
  res.dgamma_func = libxsmm_dispatch_matrix_eqn( res.CB, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn11 );      /* dgamma [CB] */
  if ( res.dgamma_func == NULL) {
    fprintf( stderr, "JIT for TPP bwd dgamma_func (eqn11) failed. Bailing...!\n");
    exit(-1);
  }

  /* dbeta function  */
  my_eqn12 = libxsmm_matrix_eqn_create();                                                       /* dbeta [CB] = dout [HW, CB] + dbeta [CB] */
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn12, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );                /* dbeta_tmp [HW, CB] */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn12, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);  /* [HW, CB] -> [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn12, res.CB, res.HW/res.num_HW_blocks, ld, 3, 0, in_dt );          /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn12, res.CB, 1, 1, 5, 0, LIBXSMM_DATATYPE_F32 );           /* dbeta [CB] */
  /* libxsmm_matrix_eqn_tree_print( my_eqn12 ); */
  /* libxsmm_matrix_eqn_rpn_print( my_eqn12 ); */
  res.dbeta_func = libxsmm_dispatch_matrix_eqn( res.CB, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn12 );      /* dbeta [CB] */
  if ( res.dbeta_func == NULL) {
    fprintf( stderr, "JIT for TPP bwd dbeta_func (eqn12) failed. Bailing...!\n");
    exit(-1);
  }

  /* din = gamma_ptr[v] * brstd_ptr[v] * recp_nhw * (nhw*del_output_ptr[v] - (del_beta_ptr[v] + (input_ptr[v] - bmean_ptr[v]) * del_gamma_ptr[v] * brstd_ptr[v])) */
  /* din = gamma_ptr[v] * brstd_ptr[v] *del_output_ptr[v] - gamma_ptr[v] * brstd_ptr[v] * recp_nhw * (del_beta_ptr[v] + (input_ptr[v] - bmean_ptr[v]) * del_gamma_ptr[v] * brstd_ptr[v])) */
  /* din = gamma_ptr[v] * brstd_ptr[v] *del_output_ptr[v] - gamma_ptr[v] * brstd_ptr[v] * recp_nhw * del_beta_ptr[v] + gamma_ptr[v] * brstd_ptr[v] * recp_nhw * (input_ptr[v] - bmean_ptr[v]) * del_gamma_ptr[v] * brstd_ptr[v]) */
  /* din = a * del_output_ptr[v] + b * input_ptr[v] + c */
  /* a = gamma_ptr[CB] * brstd_ptr[CB] */
  /* b = gamma_ptr[CB] *  del_gamma_ptr[v] * brstd_ptr[CB] * brstd_ptr[CB] * recp_nhw */
  /* c = -gamma_ptr[CB] * brstd_ptr[CB] * recp_nhw * del_beta_ptr[CB] + gamma_ptr[CB] * brstd_ptr[CB] * recp_nhw * bmean_ptr[CB] * del_gamma_ptr[CB] * brstd_ptr[CB]) */

  /* din long equation */
  my_eqn16 = libxsmm_matrix_eqn_create();                                                       /* din = a * dout + (b * inp + c) */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn16, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, res.CB, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 );           /* a [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, res.CB, res.HW/res.num_HW_blocks, ld, 3, 0, in_dt );          /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn16, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, res.CB, res.HW/res.num_HW_blocks, ld, 0, 0, in_dt );          /* inp [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, res.CB, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );           /* b [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, res.CB, 1, 1, 7, 0, LIBXSMM_DATATYPE_F32 );           /* c [CB] */
  /* libxsmm_matrix_eqn_tree_print( my_eqn16 ); */
  /* libxsmm_matrix_eqn_rpn_print( my_eqn16 ); */
  res.din_func = libxsmm_dispatch_matrix_eqn( res.CB, res.HW/res.num_HW_blocks, &ld, in_dt, my_eqn16 );           /* din [HW, CB] */
  if ( res.din_func == NULL) {
    fprintf( stderr, "JIT for TPP bwd din_func (eqn16) failed. Bailing...!\n");
    exit(-1);
  }

  return res;
}

//void tpp_batchnorm_bwd_fp32(int N, int CP, int HW, int CB, int num_HW_blocks,
// float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta,
//    libxsmm_matrix_eqn_function dgamma_func, libxsmm_matrix_eqn_function dbeta_func, libxsmm_matrix_eqn_function din_func, float eps,
//    libxsmm_meltwfunction_unary all_zero_kernel, libxsmm_meltwfunction_binary add_kernel, libxsmm_meltwfunction_unary copy_kernel) {


// FIXME: Remove omp pragmas from the copy-pasted code, need to use tid to have a reasonable kernel
// FIXME: Set const modifiers properly?
// FIXME: Add a "my_pass" type of input argument to distinguish between backward over weights vs backward over data
void my_bn_bwd_exec( my_bn_bwd_config cfg, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta, float eps,
                     int start_tid, int my_tid, void *scratch) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint HW = cfg.HW;
  const libxsmm_blasint CB = cfg.CB;
  const libxsmm_blasint num_HW_blocks = cfg.num_HW_blocks;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;
#ifdef TRUE_PARALLEL_BWD
  /* number of tasks that could be run in parallel for 1d blocking */
  // Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here?
  const libxsmm_blasint work_dN = N * CP;
  /* compute chunk size */
  const libxsmm_blasint chunksize_dN = (work_dN % cfg.threads == 0) ?
    (work_dN / cfg.threads) : ((work_dN / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_dN = (ltid * chunksize_dN < work_dN) ? (ltid * chunksize_dN) : work_dN;
  const libxsmm_blasint thr_end_dN = ((ltid + 1) * chunksize_dN < work_dN) ? ((ltid + 1) * chunksize_dN) : work_dN;

  /* number of tasks that could be run in parallel for 1d blocking */
  // Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here?
  const libxsmm_blasint work_C = CP;
  /* compute chunk size */
  const libxsmm_blasint chunksize_C = (work_C % cfg.threads == 0) ?
    (work_C / cfg.threads) : ((work_C / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_C = (ltid * chunksize_C < work_C) ? (ltid * chunksize_C) : work_C;
  const libxsmm_blasint thr_end_C = ((ltid + 1) * chunksize_C < work_C) ? ((ltid + 1) * chunksize_C) : work_C;
#endif

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

//#if 0

  const float scale = 1.0f / ((float)N*HW);                   /* Scaling parameter*/

  LIBXSMM_VLA_DECL(4, float, din, pdin, CP, HW, CB);          /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, float, inp, pinp, CP, HW, CB);          /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, float, dout, pdout, CP, HW, CB);        /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);              /* [CP, CB] */
  /* LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, CB); */
  /* LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, CB); */

#ifdef TRUE_PARALLEL_BWD
  const libxsmm_blasint dbeta_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + CP * N * CB), 64) - ((uintptr_t)(scratch))) / sizeof(float);// ((CP*N*CB) + 64 - 1) / 64;
  LIBXSMM_VLA_DECL(3, float, dgamma_N, ((float*)scratch),                  N, CB);  /* [CP, N, CB] */
  LIBXSMM_ASSUME_ALIGNED(dgamma_N_, 64);
  LIBXSMM_VLA_DECL(3, float, dbeta_N,  ((float*)scratch) + dbeta_N_offset, N, CB);  /* [CP, N, CB] */
  LIBXSMM_ASSUME_ALIGNED(dbeta_N_, 64);
#else
  LIBXSMM_ALIGNED(float dgamma_N[CP*N*CB], 64); // should get replaced with scratch
  LIBXSMM_ALIGNED(float dbeta_N[CP*N*CB], 64);
#endif

#ifdef TRUE_PARALLEL_BWD
  {
#else
  #pragma omp parallel
  {
#endif
    LIBXSMM_ALIGNED(float a[CB], 64); // should also get replaced with scratch I guess
    LIBXSMM_ALIGNED(float b[CB], 64);
    LIBXSMM_ALIGNED(float c[CB], 64);
    int n, cp;

#ifdef TRUE_PARALLEL_BWD
    int cpxnt;
    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      { // stupid block to keep indentation
        n  = cpxnt%N;
        cp = cpxnt/N;
#else
    #pragma omp for nowait collapse(2)
    for (cp = 0; cp < CP; cp++) {
      for (n = 0; n < N; n++) {
#endif

        int hwb, cb;
        libxsmm_matrix_arg arg_array[10];                                                           /* Private values of args and params */
        libxsmm_matrix_eqn_param eqn_param;

        LIBXSMM_ALIGNED(float lcl_dgamma_ptr[CB], 64);
        LIBXSMM_ALIGNED(float lcl_dbeta_ptr[CB], 64);

#ifdef TRUE_PARALLEL_BWD
        float *dgamma_ncp_ptr = &LIBXSMM_VLA_ACCESS(3, dgamma_N, cp, n, 0, N, CB);
        float *dbeta_ncp_ptr  = &LIBXSMM_VLA_ACCESS(3, dbeta_N, cp, n, 0, N, CB);
#else
        float *dgamma_ncp_ptr = &dgamma_N[cp*N*CB + n*CB];
        float *dbeta_ncp_ptr = &dbeta_N[cp*N*CB + n*CB];
#endif

        libxsmm_meltw_unary_param all_zero_param;
        all_zero_param.out.primary = lcl_dgamma_ptr;
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = lcl_dbeta_ptr;
        cfg.all_zero_kernel(&all_zero_param);

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
          cfg.dgamma_func(&eqn_param);                                                             /* dgamma += (a * inp + b) * dout */

          eqn_param.output.primary = lcl_dbeta_ptr;
          cfg.dbeta_func(&eqn_param);                                                              /* dbeta += dout */
        }

        libxsmm_meltw_unary_param copy_param;
        copy_param.in.primary = lcl_dgamma_ptr;
        copy_param.out.primary = dgamma_ncp_ptr;
        cfg.copy_kernel(&copy_param);

        copy_param.in.primary = lcl_dbeta_ptr;
        copy_param.out.primary = dbeta_ncp_ptr;
        cfg.copy_kernel(&copy_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < CB; cb++) { */
        /*   dgamma_ncp_ptr[cb] = lcl_dgamma_ptr[cb]; */
        /*   dbeta_ncp_ptr[cb] = lcl_dbeta_ptr[cb]; */
        /* } */
      }
    }

#ifdef TRUE_PARALLEL_BWD
    libxsmm_barrier_wait(cfg.barrier, ltid); // valid replacement? what exactly ltid argument does?
#else
    #pragma omp barrier
#endif

#ifdef TRUE_PARALLEL_BWD
    for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {
#else
    #pragma omp for
    for (cp = 0; cp < CP; cp++) {
#endif
      libxsmm_meltw_unary_param all_zero_param;
      all_zero_param.out.primary = &pdgamma[cp*CB];
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &pdbeta[cp*CB];
      cfg.all_zero_kernel(&all_zero_param);

      /* #pragma omp simd */
      /* for (int cb = 0; cb < CB; cb++) { */
      /*   pdgamma[cp*CB + cb] = 0.0f; */
      /*   pdbeta[cp*CB + cb] = 0.0f; */
      /* } */

      libxsmm_meltw_binary_param add_param;
      int ni;
      for(ni = 0; ni < N; ni++){

        add_param.in0.primary = &pdgamma[cp*CB];
#ifdef TRUE_PARALLEL_BWD
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, cp, ni, 0, N, CB);
#else
        add_param.in1.primary = &dgamma_N[cp*N*CB + ni*CB];
#endif
        add_param.out.primary = &pdgamma[cp*CB];
        cfg.add_kernel(&add_param);

        add_param.in0.primary = &pdbeta[cp*CB];
#ifdef TRUE_PARALLEL_BWD
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, cp, n, 0, N, CB);
#else
        add_param.in1.primary = &dbeta_N[cp*N*CB + ni*CB];
#endif
        add_param.out.primary = &pdbeta[cp*CB];
        cfg.add_kernel(&add_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < CB; cb++) { */
        /*   pdgamma[cp*CB + cb] += dgamma_N[cp*N*CB + n*CB + cb];  */
        /*   pdbeta[cp*CB + cb] += dbeta_N[cp*N*CB + n*CB + cb];  */
        /* } */
      }
    }

//#if 0

#ifdef TRUE_PARALLEL_BWD
    libxsmm_barrier_wait(cfg.barrier, ltid); // valid replacement? what exactly ltid argument does?
#endif

#ifdef TRUE_PARALLEL_BWD
    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      { // stupid block to keep indentation
        n  = cpxnt%N;
        cp = cpxnt/N;
#else
    #pragma omp for nowait collapse(2)                                                                  /* Parallelize over batches and CP*/
    for (cp = 0; cp < CP; cp++) {
      for(n = 0; n < N; n++){
#endif
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
          cfg.din_func(&eqn_param);                                                                        /* din = dout * a + b * inp + c */
        }
      }
    }
//#endif // for #if 0

  }

//#endif // for #if 0

  libxsmm_barrier_wait(cfg.barrier, ltid);


}

//void tpp_batchnorm_bwd_fp32(int N, int CP, int HW, int CB, int num_HW_blocks, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta,
//    libxsmm_matrix_eqn_function dgamma_func, libxsmm_matrix_eqn_function dbeta_func, libxsmm_matrix_eqn_function din_func, float eps,
//    libxsmm_meltwfunction_unary all_zero_kernel, libxsmm_meltwfunction_binary add_kernel, libxsmm_meltwfunction_unary copy_kernel) {

#ifdef DEBUGGING
void my_bn_bwd_exec_dbg( my_bn_bwd_config cfg, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta, float eps,
                     int start_tid, int my_tid,
    libxsmm_matrix_eqn_function dgamma_func, libxsmm_matrix_eqn_function dbeta_func, libxsmm_matrix_eqn_function din_func,
    libxsmm_meltwfunction_unary all_zero_kernel, libxsmm_meltwfunction_binary add_kernel, libxsmm_meltwfunction_unary copy_kernel) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint HW = cfg.HW;
  const libxsmm_blasint CB = cfg.CB;
  const libxsmm_blasint num_HW_blocks = cfg.num_HW_blocks;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

//#if 0

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

//#if 0

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
//#endif // for #if 0

  }

//#endif // for #if 0

  libxsmm_barrier_wait(cfg.barrier, ltid);

}

#endif // for #ifdef DEBUGGING

#endif // for #ifdef REFACTORED_BWD


void tpp_batchnorm_bwd_fp32(int N, int CP, int HW, int CB, int num_HW_blocks, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta,
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

//#if 0

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
//#endif // for #if 0

  }
}

#endif // for #ifndef ONLY_FWD

void scaler_batchnorm_fwd_fp32(int N, int CP, int HW, int CB, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout, float eps){

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

#ifndef ONLY_FWD

void scaler_batchnorm_bwd_fp32(int N, int CP, int HW, int CB, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta, float eps) {

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

#endif // for #ifndef ONLY_FWD

void reference_batchnorm_fwd_fp32(int N, int CP, int HW, int CB, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout, float eps){

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

#ifndef ONLY_FWD

void reference_batchnorm_bwd_fp32(int N, int CP, int HW, int CB, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta, float eps){

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

#endif // for #ifndef ONLY_FWD

int main( int argc, char* argv[] ) {

#ifdef REFACTORED_FWD
  my_bn_fwd_config my_bn_fwd;
#endif

#ifdef REFACTORED_BWD
#ifndef ONLY_FWD
  my_bn_bwd_config my_bn_bwd;
#endif
#endif

#if defined(REFACTORED_BWD) || defined(REFACTORED_FWD)
  void *scratch;

  naive_fusedbatchnorm_t naive_param;
//LIBXSMM_INLINE void naive_fusedbatchnorm_fp(naive_fusedbatchnorm_t* param, const float* input_ptr, float* output_ptr, const float* input_add_ptr,
//                                     const float* beta_ptr, const float* gamma_ptr, float* expectval_ptr, float* rcpstddev_ptr, float* variance_ptr)

#endif

  // Some are unused if either FWD or BWD is defined
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
#if defined(REFACTORED_BWD) || defined(REFACTORED_FWD)
  float *naive_inp, *naive_out, *naive_rcpstdev, *naive_zeros, *naive_dinp, *naive_dout, *naive_dbeta, *naive_dgamma;

#ifdef COMPUTE_FP64_REFERENCE
  double *naive_inp_fp64, *naive_out_fp64, *naive_rcpstdev_fp64, *naive_zeros_fp64, *naive_dinp_fp64, *naive_dout_fp64, *naive_dbeta_fp64, *naive_dgamma_fp64;
  double *beta_fp64, *gamma_fp64, *mean_fp64, *var_fp64;
  double *dbeta_fp64, *dgamma_fp64;
  float *naive_out_fp64_downscaled_to_fp32, *out_fp64_downscaled_to_fp32;
  float *naive_dinp_fp64_downscaled_to_fp32, *dinp_fp64_downscaled_to_fp32;
  float *dgamma_fp64_downscaled_to_fp32;
  float *dbeta_fp64_downscaled_to_fp32;
#endif

//  naive_fusedbatchnorm_fp(&naive_param, naive_inp, naive_out, naive_zeros /*cannot pass NULL or &dummy due to VLA_ACCESS but should be unused when fuse = 0 const float* input_add_ptr*/,
//                                        beta, gamma, eps, mean, naive_rcpstdev, var);

#endif

#ifdef DEBUGGING
  float *dbg_eqn_dinp, *dbg_eqn_dout;//, *dbg_mean, *dbg_var, *dbg_gamma;
  float *dbg_eqn_dgamma, *dbg_eqn_dbeta;
  //my_bn_bwd_exec( my_bn_bwd, dbg_eqn_dout, inp, dbg_mean, dbg_var, dbg_gamma, dbg_eqn_dinp, dbg_eqn_dgamma, dbg_eqn_dbeta, eps, 0, tid );
#endif

  libxsmm_bfloat16 *bf16_inp, *bf16_out, *bf16_dinp, *bf16_dout, *bf16_eqn_dinp, *bf16_eqn_dout, *bf16_gamma, *bf16_beta, *bf16_eqn_out;
  int N = 28;
  int CP = 2;
  int HW = 784;
  int CB = 64;
  int num_HW_blocks = 16;
  int iters = 100;
  int datatype_mode = 0;
  libxsmm_datatype  in_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype  out_dt = LIBXSMM_DATATYPE_F32;

  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 1 : atof(env_check));

#if defined(_OPENMP)
  int nThreads = omp_get_max_threads(); /* number of threads */
#else
  int nThreads = 1; /* number of threads */
#endif

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

#if defined(REFACTORED_BWD) || defined(REFACTORED_FWD)
  naive_param.N = N;
  naive_param.C = CP*CB;
  naive_param.H = HW;
  naive_param.W = 1;
  naive_param.stride_h = 1;
  naive_param.stride_w = 1;
  naive_param.norm_type = 0; /* full batchnorm */
  naive_param.fuse_type = 0; /* nothing fused */

  /* let's allocate and bind scratch */
  //if ( my_fc_fwd.scratch_size > 0 || my_fc_bwd.scratch_size > 0 ) {
    //size_t alloc_size = LIBXSMM_MAX( my_fc_fwd.scratch_size, my_fc_bwd.scratch_size);
    size_t alloc_size = ( 100 * N * CP * CB + 1000) * sizeof(float); // need to be fixed later
    scratch = libxsmm_aligned_malloc( alloc_size, 2097152 );
    init_buf( (float*)(scratch), (alloc_size)/4, 0, 0 );
  //}

#endif

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

#ifdef DEBUGGING
  dbg_eqn_dinp = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*CB,   2097152);
  dbg_eqn_dout = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*CB,   2097152);
  //dbg_mean = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  //dbg_var = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  //dbg_gamma = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  dbg_eqn_dgamma = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  dbg_eqn_dbeta = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  //my_bn_bwd_exec( my_bn_bwd, dbg_eqn_dout, inp, dbg_mean, dbg_var, dbg_gamma, dbg_eqn_dinp, dbg_eqn_dgamma, dbg_eqn_dbeta, eps, 0, tid );
#endif

  bf16_inp = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*CP*HW*CB,   2097152);
  bf16_out = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*CP*HW*CB,   2097152);
  bf16_dinp = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*CP*HW*CB,   2097152);
  bf16_dout = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*CP*HW*CB,   2097152);
  bf16_eqn_dinp = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*CP*HW*CB,   2097152);
  bf16_eqn_dout = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*CP*HW*CB,   2097152);
  bf16_gamma = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*CP*CB,   2097152);
  bf16_beta = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*CP*CB,   2097152);
  bf16_eqn_out  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*CP*HW*CB,   2097152);

#if defined(REFACTORED_BWD) || defined(REFACTORED_FWD)
  naive_inp = (float*) libxsmm_aligned_malloc( sizeof(float)*N*(CP*CB)*HW*1, 2097152);
  naive_out = (float*) libxsmm_aligned_malloc( sizeof(float)*N*(CP*CB)*HW*1, 2097152);
  naive_dinp = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*CB,   2097152);
  naive_dout = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*CB,   2097152);
  naive_dgamma = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  naive_dbeta = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
  naive_rcpstdev = (float*) libxsmm_aligned_malloc( sizeof(float)*(CP*CB),   2097152);
  naive_zeros = (float*) libxsmm_aligned_malloc( sizeof(float)*N*(CP*CB)*HW*1, 2097152);

#ifdef COMPUTE_FP64_REFERENCE
  naive_inp_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*N*(CP*CB)*HW*1, 2097152);
  naive_out_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*N*(CP*CB)*HW*1, 2097152);
  naive_dinp_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*N*CP*HW*CB,   2097152);
  naive_dout_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*N*CP*HW*CB,   2097152);
  naive_dgamma_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*CB,   2097152);
  naive_dbeta_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*CB,   2097152);
  naive_rcpstdev_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*(CP*CB),   2097152);
  naive_zeros_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*N*(CP*CB)*HW*1, 2097152);

  gamma_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*CB,   2097152);
  beta_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*CB,   2097152);
  mean_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*CB,   2097152);
  var_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*CB,   2097152);

  dgamma_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*CB,   2097152);
  dbeta_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*CB,   2097152);

  naive_out_fp64_downscaled_to_fp32 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*(CP*CB)*HW*1, 2097152);
  out_fp64_downscaled_to_fp32 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*(CP*CB)*HW*1, 2097152);

  naive_dinp_fp64_downscaled_to_fp32 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*(CP*CB)*HW*1, 2097152);
  dinp_fp64_downscaled_to_fp32 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*(CP*CB)*HW*1, 2097152);

  dgamma_fp64_downscaled_to_fp32 = (float*) libxsmm_aligned_malloc( sizeof(float)*(CP*CB)*1, 2097152);
  dbeta_fp64_downscaled_to_fp32 = (float*) libxsmm_aligned_malloc( sizeof(float)*(CP*CB)*1, 2097152);
#endif

#endif

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
#if defined(REFACTORED_BWD) || defined(REFACTORED_FWD)
    naive_zeros[i] = 0.0f;
    naive_zeros_fp64[i] = 0.0;
#endif

#ifdef DEBUGGING
    dbg_eqn_dinp[i] = eqn_dinp[i];
    dbg_eqn_dout[i] = eqn_dout[i];
#endif

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

#ifdef DEBUGGING
    //dbg_mean = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
    //dbg_var = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
    //dbg_gamma[i] = gamma[i];
    dbg_eqn_dgamma[i] = eqn_dgamma[i];
    dbg_eqn_dbeta[i]  = eqn_dbeta[i];
#endif

#ifdef COMPUTE_FP64_REFERENCE
    gamma_fp64[i] = gamma[i];
    beta_fp64 [i] = beta[i];
#endif
  }

  for (i = 0; i < 1024 * 1024; i++ ) {
    cache_fl[i] = (float)libxsmm_rng_f64();
  }

#ifdef REFACTORED_FWD
  my_bn_fwd = setup_my_bn_fwd(N, CP, HW, CB, num_HW_blocks, nThreads /*, my_eltwise_fuse fuse_type*/ );
#endif

#if !defined(REFACTORED_BWD) || !defined(REFACTORED_FWD)

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

#endif // for #ifdef-else !FWD or !BWD

#ifdef REFACTORED_FWD
#else
  unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD;
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  reduce_HW_kernel = libxsmm_dispatch_meltw_unary(CB, HW/num_HW_blocks, &ld, &tmp_ld, in_dt, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);

  /* TPP for scaling */
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
  /* libxsmm_matrix_eqn_tree_print( my_eqn10 ); */
  /* libxsmm_matrix_eqn_rpn_print( my_eqn10 ); */
  func10 = libxsmm_dispatch_matrix_eqn( CB, HW/num_HW_blocks, &ld, out_dt, my_eqn10 );                         /* y = [HW, CB] */
#endif

  /* Check correctness */
#ifdef REFACTORED_FWD
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_bn_fwd_exec( my_bn_fwd, inp, gamma, beta, mean, var, eqn_out, eps, 0, tid, scratch);
      //my_fc_fwd_exec( my_fc_fwd, filter_libxsmm, input_libxsmm, output_libxsmm,
      //    bias_libxsmm, relumask_libxsmm, 0, tid, scratch );
    }
#else
  tpp_batchnorm_fwd_fp32(N, CP, HW, CB, num_HW_blocks, inp, gamma, beta, mean, var, eqn_out, eps, func10, reduce_HW_kernel, all_zero_kernel, add_kernel, copy_kernel);
#endif // for #ifdef-else REFACTORED_FWD


#ifdef REFACTORED_FWD
  tensor_copy_NCHWc_to_NCHW (inp, naive_inp, N, CP*CB, HW, 1, CB);
  //LIBXSMM_INLINE void naive_fusedbatchnorm_fp(naive_fusedbatchnorm_t* param, const float* input_ptr, float* output_ptr, const float* input_add_ptr,
  //                                   const float* beta_ptr, const float* gamma_ptr, float eps, float* expectval_ptr, float* rcpstddev_ptr, float* variance_ptr)

  naive_fusedbatchnorm_fp(&naive_param, naive_inp, naive_out, naive_zeros /*cannot pass NULL or &dummy due to VLA_ACCESS but should be unused when fuse = 0 const float* input_add_ptr*/,
                                        beta, gamma, eps, mean, naive_rcpstdev, var);

  tensor_copy_NCHW_to_NCHWc (naive_out, out,  N, CP*CB, HW, 1, CB);

#ifdef COMPUTE_FP64_REFERENCE
  buf_extend_fp32_to_fp64 (naive_inp, naive_inp_fp64, N*CP*CB*HW);

  naive_fusedbatchnorm_fp_fp64(&naive_param, naive_inp_fp64, naive_out_fp64, naive_zeros_fp64 /*cannot pass NULL or &dummy due to VLA_ACCESS but should be unused when fuse = 0 const float* input_add_ptr*/,
                                        beta_fp64, gamma_fp64, eps, mean_fp64, naive_rcpstdev_fp64, var_fp64);

  buf_truncate_fp64_to_fp32 (naive_out_fp64, naive_out_fp64_downscaled_to_fp32, N*CP*CB*HW);

  tensor_copy_NCHW_to_NCHWc (naive_out_fp64_downscaled_to_fp32, out_fp64_downscaled_to_fp32,  N, CP*CB, HW, 1, CB);
#endif


//  /* while debugging */
//  float *debug_naive_out = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*CB,   2097152);
//  tensor_copy_NCHW_to_NCHWc (naive_out, debug_naive_out,  N, CP*CB, HW, 1, CB);

#else
  if(CB == 1)
    reference_batchnorm_fwd_fp32(N, CP, HW, CB, inp, gamma, beta, mean, var, out, eps);
  else
    scaler_batchnorm_fwd_fp32(N, CP, HW, CB, inp, gamma, beta, mean, var, out, eps);
#endif // for #ifdef REFACTORED_FWD

  /* compare */
  printf("############################################\n");
  printf("# Correctness FP32 FWD Batchnorm - Output  #\n");
  printf("############################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, N*CP*HW*CB, 1, out, eqn_out, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, N*CP*HW*CB, 1, out_fp64_downscaled_to_fp32, eqn_out, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  //return 0;

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
#ifdef REFACTORED_FWD
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_bn_fwd_exec( my_bn_fwd, inp, gamma, beta, mean, var, eqn_out, eps, 0, tid, scratch);
    }
#else
  tpp_batchnorm_fwd_fp32(N, CP, HW, CB, num_HW_blocks, inp, gamma, beta, mean, var, eqn_out, eps, func10, reduce_HW_kernel, all_zero_kernel, add_kernel, copy_kernel);
#endif
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
#ifdef REFACTORED_FWD
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_bn_fwd_exec( my_bn_fwd, inp, gamma, beta, mean, var, eqn_out, eps, 0, tid, scratch );
    }
#else
    tpp_batchnorm_fwd_fp32(N, CP, HW, CB, num_HW_blocks, inp, gamma, beta, mean, var, eqn_out, eps, func10, reduce_HW_kernel, all_zero_kernel, add_kernel, copy_kernel);
#endif
  }
  l_end = libxsmm_timer_tick();
  l_total2 = libxsmm_timer_duration(l_start, l_end);
  printf("TPP batchnorm time FWD  = %.5g\n", ((double)(l_total2)));
  printf("Speedup FWD is %.5g\n", l_total/l_total2);

#ifndef ONLY_FWD

#ifdef REFACTORED_BWD
  my_bn_bwd = setup_my_bn_bwd(N, CP, HW, CB, num_HW_blocks, nThreads /*, my_eltwise_fuse fuse_type*/ );
#else // for #ifdef REFACTORED_BWD

  /* Create MatEq for bwd layernorm */

  ld = CB;
  tmp_ld2 = 1;

  /* dgamma function  */
  my_eqn11 = libxsmm_matrix_eqn_create();                                                       /* dgamma = ((inp *a + b) * dout) + dgamma */
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn11, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32);                   /* dgamma = ((inp *a + b) * dout) + dgamma */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn11, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);   /* [HW, CB] -> [CB] */
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn11, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32);                   /* ((inp *a + b) * dout) */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn11, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
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
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn16, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, CB, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 );           /* a [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, CB, HW/num_HW_blocks, ld, 3, 0, in_dt );          /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn16, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, CB, HW/num_HW_blocks, ld, 0, 0, in_dt );          /* inp [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, CB, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );           /* b [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, CB, 1, 1, 7, 0, LIBXSMM_DATATYPE_F32 );           /* c [CB] */
  /* libxsmm_matrix_eqn_tree_print( my_eqn16 ); */
  /* libxsmm_matrix_eqn_rpn_print( my_eqn16 ); */
  func16 = libxsmm_dispatch_matrix_eqn( CB, HW/num_HW_blocks, &ld, in_dt, my_eqn16 );           /* din [HW, CB] */

#endif // for #ifdef REFACTORED_BWD

//void tpp_batchnorm_bwd_fp32(int N, int CP, int HW, int CB, int num_HW_blocks, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta,
//    libxsmm_matrix_eqn_function dgamma_func, libxsmm_matrix_eqn_function dbeta_func, libxsmm_matrix_eqn_function din_func, float eps,
//    libxsmm_meltwfunction_unary all_zero_kernel, libxsmm_meltwfunction_binary add_kernel, libxsmm_meltwfunction_unary copy_kernel) {

#ifdef REFACTORED_BWD
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif

#ifdef DEBUGGING
      //my_bn_bwd_exec( my_bn_bwd, dbg_eqn_dout, inp, mean, var, gamma, dbg_eqn_dinp, dbg_eqn_dgamma, dbg_eqn_dbeta, eps, 0, tid );
      my_bn_bwd_exec_dbg( my_bn_bwd, dbg_eqn_dout, inp, mean, var, gamma, dbg_eqn_dinp, dbg_eqn_dgamma, dbg_eqn_dbeta, eps, 0, tid, func11, func12, func16, all_zero_kernel, add_kernel, copy_kernel );
#else
      my_bn_bwd_exec( my_bn_bwd, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, eps, 0, tid, scratch );
#endif
    }
#else
  tpp_batchnorm_bwd_fp32(N, CP, HW, CB, num_HW_blocks, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func16, eps, all_zero_kernel, add_kernel, copy_kernel);
#endif

#ifdef REFACTORED_BWD
  tensor_copy_NCHWc_to_NCHW (inp,   naive_inp   ,  N, CP*CB, HW, 1, CB);
  tensor_copy_NCHWc_to_NCHW (out,   naive_out   ,  N, CP*CB, HW, 1, CB);
  tensor_copy_NCHWc_to_NCHW (dout,  naive_dout  ,  N, CP*CB, HW, 1, CB);
  //tensor_copy_NCHWc_to_NCHW (beta,  naive_beta  ,  1, CP*CB,  1, 1, CB); // done earlier = for fwd
  //tensor_copy_NCHWc_to_NCHW (gamma, naive_gamma ,  1, CP*CB,  1, 1, CB); // done earlier = for fwd
  //LIBXSMM_INLINE void naive_fusedbatchnorm_bp(naive_fusedbatchnorm_t* param, const float* input_ptr, float* dinput_ptr, const float* output_ptr, float* doutput_ptr, float* dinput_add_ptr,
  //                                   const float* beta_ptr, float* del_beta_ptr, const float* gamma_ptr, float* del_gamma_ptr,
  //                                   const float* expectval_ptr, const float* rcpstddev_ptr)

  naive_fusedbatchnorm_bp(&naive_param, naive_inp, naive_dinp, naive_out, naive_dout, naive_zeros /*cannot pass NULL or &dummy due to VLA_ACCESS but should be unsued when fuse = 0 const float* dinput_add_ptr*/,
                                       beta, dbeta, gamma, dgamma, mean, naive_rcpstdev);

  /* when not debugging */
  tensor_copy_NCHW_to_NCHWc (naive_dinp  , dinp  ,  N, CP*CB, HW, 1, CB);
//  tensor_copy_NCHW_to_NCHWc (naive_dbeta , dbeta ,  1, CP*CB, 1, 1, CB); // not needed, can use naive_dbeta and naive_dgamma immediately as beta and gamma?
//  tensor_copy_NCHW_to_NCHWc (naive_dgamma, dgamma,  1, CP*CB, 1, 1, CB);


#ifdef COMPUTE_FP64_REFERENCE
  buf_extend_fp32_to_fp64 (naive_inp,  naive_inp_fp64,  N*CP*CB*HW);
  buf_extend_fp32_to_fp64 (naive_out,  naive_out_fp64,  N*CP*CB*HW);
  buf_extend_fp32_to_fp64 (naive_dout, naive_dout_fp64, N*CP*CB*HW);

  naive_fusedbatchnorm_bp_fp64(&naive_param, naive_inp_fp64, naive_dinp_fp64, naive_out_fp64, naive_dout_fp64, naive_zeros_fp64 /*cannot pass NULL or &dummy due to VLA_ACCESS but should be unsued when fuse = 0 const float* dinput_add_ptr*/,
                                       beta_fp64, dbeta_fp64, gamma_fp64, dgamma_fp64, mean_fp64, naive_rcpstdev_fp64);

  buf_truncate_fp64_to_fp32 (naive_dinp_fp64,   naive_dinp_fp64_downscaled_to_fp32, N*CP*CB*HW);
  buf_truncate_fp64_to_fp32 (dgamma_fp64, dgamma_fp64_downscaled_to_fp32, CP*CB);
  buf_truncate_fp64_to_fp32 (dbeta_fp64,  dbeta_fp64_downscaled_to_fp32, CP*CB);

  tensor_copy_NCHW_to_NCHWc (naive_dinp_fp64_downscaled_to_fp32, dinp_fp64_downscaled_to_fp32,  N, CP*CB, HW, 1, CB);
#endif


//  /* while debugging */
//  float *debug_naive_dinp = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*CB,   2097152);
//  float *debug_naive_dbeta = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
//  float *debug_naive_dgamma = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*CB,   2097152);
//  tensor_copy_NCHW_to_NCHWc (naive_dinp, debug_naive_dinp,  N, CP*CB, HW, 1, CB);
//  tensor_copy_NCHW_to_NCHWc (naive_dbeta, debug_naive_dbeta,  1, CP*CB, 1, 1, CB);
//  tensor_copy_NCHW_to_NCHWc (naive_dgamma, debug_naive_dgamma,  1, CP*CB, 1, 1, CB);

#else
  if (CB == 1)
    reference_batchnorm_bwd_fp32(N, CP, HW, CB, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
  else
    scaler_batchnorm_bwd_fp32(N, CP, HW, CB, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
#endif

  /* compare */
  printf("############################################\n");
  printf("# Correctness FP32 BWD Batchnorm - Dinput  #\n");
  printf("############################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, N*CP*HW*CB, 1, dinp, eqn_dinp, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, N*CP*HW*CB, 1, dinp_fp64_downscaled_to_fp32, eqn_dinp, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);


  printf("###########################################\n");
  printf("# Correctness FP32 BWD Batchnorm - Dbeta  #\n");
  printf("###########################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, CP*CB, 1, dbeta, eqn_dbeta, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, CP*CB, 1, dbeta_fp64_downscaled_to_fp32, eqn_dbeta, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  printf("############################################\n");
  printf("# Correctness FP32 BWD Batchnorm - Dgamma  #\n");
  printf("############################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, CP*CB, 1, dgamma, eqn_dgamma, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, CP*CB, 1, dgamma_fp64_downscaled_to_fp32, eqn_dgamma, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

//  return 0;

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
#ifdef REFACTORED_BWD
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_bn_bwd_exec( my_bn_bwd, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, eps, 0, tid, scratch );
    }
#else
  tpp_batchnorm_bwd_fp32(N, CP, HW, CB, num_HW_blocks, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func16, eps, all_zero_kernel, add_kernel, copy_kernel);
#endif
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
#ifdef REFACTORED_BWD
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_bn_bwd_exec( my_bn_bwd, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, eps, 0, tid, scratch );
    }
#else
    tpp_batchnorm_bwd_fp32(N, CP, HW, CB, num_HW_blocks, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func16, eps, all_zero_kernel, add_kernel, copy_kernel);
#endif
  }
  l_end = libxsmm_timer_tick();
  l_total2 = libxsmm_timer_duration(l_start, l_end);
  printf("TPP batchnorm time BWD = %.5g\n", ((double)(l_total2)));
  printf("Speedup BWD is %.5g\n", l_total/l_total2);
  /* printf("Running sum is %.5f\n", sum); */

  t_tpp += l_total2;
  t_vec += l_total;

  printf("\n\n=================================\n");
  printf("Total Speedup via TPP Matrix equation is %.5g\n", t_vec/t_tpp);
  printf("=================================\n");

#endif // for #ifdef ONLY_FWD

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
#if defined(REFACTORED_BWD) || defined(REFACTORED_FWD)
  libxsmm_free(naive_inp);
  libxsmm_free(naive_out);
  libxsmm_free(naive_dinp);
  libxsmm_free(naive_dout);
  libxsmm_free(naive_dgamma);
  libxsmm_free(naive_dbeta);
  libxsmm_free(naive_rcpstdev);
  libxsmm_free(naive_zeros);
#endif

#ifdef DEBUGGING
  libxsmm_free(dbg_eqn_dinp);
  libxsmm_free(dbg_eqn_dout);
  //libxsmm_free(dbg_mean);
  //libxsmm_free(dbg_var);
  //libxsmm_free(dbg_gamma);
  libxsmm_free(dbg_eqn_dgamma);
  libxsmm_free(dbg_eqn_dbeta);
  //my_bn_bwd_exec( my_bn_bwd, dbg_eqn_dout, inp, dbg_mean, dbg_var, dbg_gamma, dbg_eqn_dinp, dbg_eqn_dgamma, dbg_eqn_dbeta, eps, 0, tid );
#endif


  return 0;
}
