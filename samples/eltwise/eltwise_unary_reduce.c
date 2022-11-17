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
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
/*#define FP16_REDUCE_COLSIDX*/
#ifdef FP16_REDUCE_COLSIDX
#include <immintrin.h>
#endif

#include "eltwise_common.h"

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
void reference_reduce_kernel_f64( libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld_in, libxsmm_blasint n_cols_idx,
                              double *sinp, double *ref_result_reduce_elts, double *ref_result_reduce_elts_squared, unsigned long long *cols_ind_array,
                              unsigned int reduce_op, unsigned int reduce_rows,
                              unsigned int record_idx, unsigned long long *ref_argop_off, libxsmm_datatype dtype, unsigned int reduce_on_output ) {
  libxsmm_blasint i = 0, j = 0, jj = 0;
  LIBXSMM_UNUSED(dtype);
  if (reduce_op == 0) {
    /* Calculate reference results... */
    if (reduce_rows == 1) {
      for (j = 0; j < n; j++) {
        if ( reduce_on_output == 0 ) {
          ref_result_reduce_elts[j] = 0;
          ref_result_reduce_elts_squared[j] = 0;
        }
        for (i = 0; i < m; i++) {
          ref_result_reduce_elts[j] += sinp[j*ld_in + i];
          ref_result_reduce_elts_squared[j] += sinp[j*ld_in + i] * sinp[j*ld_in + i];
        }
      }
    } else {
      if (n_cols_idx == 0) {
        /* In this case we reduce columns */
        for (i = 0; i < m; i++) {
          if ( reduce_on_output == 0 ) {
            ref_result_reduce_elts[i] = 0;
            ref_result_reduce_elts_squared[i] = 0;
          }
          for (j = 0; j < n; j++) {
            ref_result_reduce_elts[i] += sinp[j*ld_in + i];
            ref_result_reduce_elts_squared[i] += sinp[j*ld_in + i] * sinp[j*ld_in + i];
          }
        }
      } else {
        for (i = 0; i < m; i++) {
          if ( reduce_on_output == 0 ) {
            ref_result_reduce_elts[i] = 0;
          }
          for (jj = 0; jj < n_cols_idx; jj++) {
            j = (libxsmm_blasint)cols_ind_array[jj];
            ref_result_reduce_elts[i] += sinp[j*ld_in + i];
          }
        }
      }
    }
  } else {
    if (reduce_rows == 1) {
      for (j = 0; j < n; j++) {
        ref_result_reduce_elts[j] = sinp[j*ld_in];
        for (i = 0; i < m; i++) {
          ref_result_reduce_elts[j] = LIBXSMM_MAX( ref_result_reduce_elts[j], sinp[j*ld_in + i] );
        }
      }
    } else {
      if (n_cols_idx == 0) {
        /* In this case we reduce columns */
        for (i = 0; i < m; i++) {
          ref_result_reduce_elts[i] = sinp[i];
          for (j = 0; j < n; j++) {
            ref_result_reduce_elts[i] = LIBXSMM_MAX( sinp[j*ld_in + i], ref_result_reduce_elts[i]);
          }
        }
      } else {
        for (i = 0; i < m; i++) {
          ref_result_reduce_elts[i] = -FLT_MAX;
          for (jj = 0; jj < n_cols_idx; jj++) {
            j = LIBXSMM_CAST_BLASINT(cols_ind_array[jj]);
            if (record_idx > 0) {
              if (sinp[j*ld_in + i] >= ref_result_reduce_elts[i] ) {
                ref_result_reduce_elts[i] = sinp[j*ld_in + i];
                ref_argop_off[i] = j;
              }
            } else {
              ref_result_reduce_elts[i] = LIBXSMM_MAX( sinp[j*ld_in + i], ref_result_reduce_elts[i]);
            }
          }
        }
      }
    }
  }
}

LIBXSMM_INLINE
void reference_reduce_kernel( libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld_in, libxsmm_blasint n_cols_idx,
                              float *sinp, float *ref_result_reduce_elts, float *ref_result_reduce_elts_squared, unsigned long long *cols_ind_array,
                              unsigned int reduce_op, unsigned int reduce_rows,
                              unsigned int record_idx, unsigned long long *ref_argop_off, libxsmm_datatype dtype, unsigned int reduce_on_output ) {
  libxsmm_blasint i = 0, j = 0, jj = 0;
  char *tmp_sinp_lp = NULL, *tmp_ref_result_reduce_elts_lp = NULL, *tmp_ref_result_reduce_elts_squared_lp = NULL;
  int result_size = (reduce_rows == 1) ? n : ld_in;
  if (dtype == LIBXSMM_DATATYPE_BF16) {
    tmp_sinp_lp  = (char*) malloc( sizeof(libxsmm_bfloat16)*ld_in*n );
    tmp_ref_result_reduce_elts_lp = (char*) malloc( sizeof(libxsmm_bfloat16)*result_size );
    tmp_ref_result_reduce_elts_squared_lp = (char*) malloc( sizeof(libxsmm_bfloat16)*result_size );
    if (tmp_sinp_lp == NULL || tmp_ref_result_reduce_elts_lp == NULL || tmp_ref_result_reduce_elts_squared_lp == NULL ) {
      fprintf(stderr,"Error : reference_reduce_kernel allocation failed\n");
      exit(-1);
    }
    libxsmm_rne_convert_fp32_bf16( sinp, (libxsmm_bfloat16*)tmp_sinp_lp, ld_in*n );
    libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)tmp_sinp_lp, sinp, ld_in*n );
  } else if (dtype == LIBXSMM_DATATYPE_F16) {
    tmp_sinp_lp  = (char*) malloc( sizeof(libxsmm_float16)*ld_in*n );
    tmp_ref_result_reduce_elts_lp = (char*) malloc( sizeof(libxsmm_float16)*result_size );
    tmp_ref_result_reduce_elts_squared_lp = (char*) malloc( sizeof(libxsmm_float16)*result_size );
    if (tmp_sinp_lp == NULL || tmp_ref_result_reduce_elts_lp == NULL || tmp_ref_result_reduce_elts_squared_lp == NULL ) {
      fprintf(stderr,"Error : reference_reduce_kernel allocation failed\n");
      exit(-1);
    }
    libxsmm_rne_convert_fp32_f16( sinp, (libxsmm_float16*)tmp_sinp_lp, ld_in*n );
    libxsmm_convert_f16_f32( (libxsmm_float16*)tmp_sinp_lp, sinp, ld_in*n );
  } else if (dtype == LIBXSMM_DATATYPE_BF8) {
    tmp_sinp_lp  = (char*) malloc( sizeof(libxsmm_bfloat8)*ld_in*n );
    tmp_ref_result_reduce_elts_lp = (char*) malloc( sizeof(libxsmm_bfloat8)*result_size );
    tmp_ref_result_reduce_elts_squared_lp = (char*) malloc( sizeof(libxsmm_bfloat8)*result_size );
    if (tmp_sinp_lp == NULL || tmp_ref_result_reduce_elts_lp == NULL || tmp_ref_result_reduce_elts_squared_lp == NULL ) {
      fprintf(stderr,"Error : reference_reduce_kernel allocation failed\n");
      exit(-1);
    }
    libxsmm_rne_convert_fp32_bf8( sinp, (libxsmm_bfloat8*)tmp_sinp_lp, ld_in*n );
    libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)tmp_sinp_lp, sinp, ld_in*n );
  } else if (dtype == LIBXSMM_DATATYPE_HF8) {
    tmp_sinp_lp  = (char*) malloc( sizeof(libxsmm_hfloat8)*ld_in*n );
    tmp_ref_result_reduce_elts_lp = (char*) malloc( sizeof(libxsmm_hfloat8)*result_size );
    tmp_ref_result_reduce_elts_squared_lp = (char*) malloc( sizeof(libxsmm_hfloat8)*result_size );
    if (tmp_sinp_lp == NULL || tmp_ref_result_reduce_elts_lp == NULL || tmp_ref_result_reduce_elts_squared_lp == NULL ) {
      fprintf(stderr,"Error : reference_reduce_kernel allocation failed\n");
      exit(-1);
    }
    libxsmm_rne_convert_fp32_hf8( sinp, (libxsmm_hfloat8*)tmp_sinp_lp, ld_in*n );
    libxsmm_convert_hf8_f32( (libxsmm_hfloat8*)tmp_sinp_lp, sinp, ld_in*n );
  }

  if (reduce_op == 0) {
    /* Calculate reference results... */
    if (reduce_rows == 1) {
      for (j = 0; j < n; j++) {
        if ( reduce_on_output == 0 ) {
          ref_result_reduce_elts[j] = 0;
          ref_result_reduce_elts_squared[j] = 0;
        }
        for (i = 0; i < m; i++) {
          ref_result_reduce_elts[j] += sinp[j*ld_in + i];
          ref_result_reduce_elts_squared[j] += sinp[j*ld_in + i] * sinp[j*ld_in + i];
        }
      }
    } else {
      if (n_cols_idx == 0) {
        /* In this case we reduce columns */
        for (i = 0; i < m; i++) {
          if ( reduce_on_output == 0 ) {
            ref_result_reduce_elts[i] = 0;
            ref_result_reduce_elts_squared[i] = 0;
          }
          for (j = 0; j < n; j++) {
            ref_result_reduce_elts[i] += sinp[j*ld_in + i];
            ref_result_reduce_elts_squared[i] += sinp[j*ld_in + i] * sinp[j*ld_in + i];
          }
        }
      } else {
        for (i = 0; i < m; i++) {
          if ( reduce_on_output == 0 ) {
            ref_result_reduce_elts[i] = 0;
          }
          for (jj = 0; jj < n_cols_idx; jj++) {
            j = (libxsmm_blasint)cols_ind_array[jj];
            ref_result_reduce_elts[i] += sinp[j*ld_in + i];
          }
        }
      }
    }
  } else {
    if (reduce_rows == 1) {
      for (j = 0; j < n; j++) {
        ref_result_reduce_elts[j] = sinp[j*ld_in];
        for (i = 0; i < m; i++) {
          ref_result_reduce_elts[j] = LIBXSMM_MAX( ref_result_reduce_elts[j], sinp[j*ld_in + i] );
        }
      }
    } else {
      if (n_cols_idx == 0) {
        /* In this case we reduce columns */
        for (i = 0; i < m; i++) {
          ref_result_reduce_elts[i] = sinp[i];
          for (j = 0; j < n; j++) {
            ref_result_reduce_elts[i] = LIBXSMM_MAX( sinp[j*ld_in + i], ref_result_reduce_elts[i]);
          }
        }
      } else {
        for (i = 0; i < m; i++) {
          ref_result_reduce_elts[i] = -FLT_MAX;
          for (jj = 0; jj < n_cols_idx; jj++) {
            j = LIBXSMM_CAST_BLASINT(cols_ind_array[jj]);
            if (record_idx > 0) {
              if (sinp[j*ld_in + i] >= ref_result_reduce_elts[i] ) {
                ref_result_reduce_elts[i] = sinp[j*ld_in + i];
                ref_argop_off[i] = j;
              }
            } else {
              ref_result_reduce_elts[i] = LIBXSMM_MAX( sinp[j*ld_in + i], ref_result_reduce_elts[i]);
            }
          }
        }
      }
    }
  }
  if (dtype == LIBXSMM_DATATYPE_BF16) {
    libxsmm_rne_convert_fp32_bf16( ref_result_reduce_elts, (libxsmm_bfloat16*)tmp_ref_result_reduce_elts_lp, result_size );
    libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)tmp_ref_result_reduce_elts_lp, ref_result_reduce_elts, result_size );
    libxsmm_rne_convert_fp32_bf16( ref_result_reduce_elts_squared, (libxsmm_bfloat16*)tmp_ref_result_reduce_elts_squared_lp, result_size );
    libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)tmp_ref_result_reduce_elts_squared_lp, ref_result_reduce_elts_squared, result_size );
  } else if (dtype == LIBXSMM_DATATYPE_F16) {
    libxsmm_rne_convert_fp32_f16( ref_result_reduce_elts, (libxsmm_float16*)tmp_ref_result_reduce_elts_lp, result_size );
    libxsmm_convert_f16_f32( (libxsmm_float16*)tmp_ref_result_reduce_elts_lp, ref_result_reduce_elts, result_size );
    libxsmm_rne_convert_fp32_f16( ref_result_reduce_elts_squared, (libxsmm_float16*)tmp_ref_result_reduce_elts_squared_lp, result_size );
    libxsmm_convert_f16_f32( (libxsmm_float16*)tmp_ref_result_reduce_elts_squared_lp, ref_result_reduce_elts_squared, result_size );
  } else if (dtype == LIBXSMM_DATATYPE_BF8) {
    libxsmm_rne_convert_fp32_bf8( ref_result_reduce_elts, (libxsmm_bfloat8*)tmp_ref_result_reduce_elts_lp, result_size );
    libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)tmp_ref_result_reduce_elts_lp, ref_result_reduce_elts, result_size );
    libxsmm_rne_convert_fp32_bf8( ref_result_reduce_elts_squared, (libxsmm_bfloat8*)tmp_ref_result_reduce_elts_squared_lp, result_size );
    libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)tmp_ref_result_reduce_elts_squared_lp, ref_result_reduce_elts_squared, result_size );
  } else if (dtype == LIBXSMM_DATATYPE_HF8) {
    libxsmm_rne_convert_fp32_hf8( ref_result_reduce_elts, (libxsmm_hfloat8*)tmp_ref_result_reduce_elts_lp, result_size );
    libxsmm_convert_hf8_f32( (libxsmm_hfloat8*)tmp_ref_result_reduce_elts_lp, ref_result_reduce_elts, result_size );
    libxsmm_rne_convert_fp32_hf8( ref_result_reduce_elts_squared, (libxsmm_hfloat8*)tmp_ref_result_reduce_elts_squared_lp, result_size );
    libxsmm_convert_hf8_f32( (libxsmm_hfloat8*)tmp_ref_result_reduce_elts_squared_lp, ref_result_reduce_elts_squared, result_size );
  }
  if (dtype != LIBXSMM_DATATYPE_F32) {
    free (tmp_sinp_lp);
    free (tmp_ref_result_reduce_elts_lp);
    free (tmp_ref_result_reduce_elts_squared_lp);
  }
}

LIBXSMM_INLINE
void setup_tpp_kernel_and_param_struct( libxsmm_meltwfunction_unary *res_kernel, libxsmm_meltwfunction_unary *res_kernel2, libxsmm_meltw_unary_param *res_unary_param, libxsmm_meltw_unary_param *res_unary_param2,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint _ld_in , libxsmm_blasint n_cols_idx,
  unsigned int reduce_rows, unsigned int reduce_op, unsigned int reduce_elts, unsigned int reduce_elts_squared, libxsmm_datatype dtype, unsigned int idx_type,
  float *sinp, float *result_reduce_elts,
  char *sinp_lp, char *result_reduce_elts_lp,
#ifdef FP16_REDUCE_COLSIDX
  unsigned short *sinp_hp, unsigned short *result_reduce_elts_hp,
#endif
  unsigned long long *cols_ind_array, unsigned int *cols_ind_array_32bit,
  unsigned int record_idx, unsigned long long *argop_off, unsigned int *argop_off_i32, unsigned int reduce_on_outputs ) {
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type  unary_type = LIBXSMM_MELTW_TYPE_UNARY_NONE;
  libxsmm_meltw_unary_shape unary_shape /*= { 0 }*/;
  libxsmm_blasint ld_in = _ld_in;
  libxsmm_meltwfunction_unary kernel = NULL;
  libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
  libxsmm_meltwfunction_unary kernel2 = NULL;
  libxsmm_meltw_unary_param params2 /*= { 0 }*/;

  memset(&unary_param, 0, sizeof(unary_param));
  memset(&params2, 0, sizeof(params2));

  if (reduce_rows == 1) {
    unary_flags = LIBXSMM_EOR(libxsmm_meltw_unary_flags, unary_flags, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS);
  } else {
    unary_flags = LIBXSMM_EOR(libxsmm_meltw_unary_flags, unary_flags, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS);
  }

  if (reduce_on_outputs > 0) {
    unary_flags = LIBXSMM_EOR(libxsmm_meltw_unary_flags, unary_flags, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INIT_ACC);
  }

  if (reduce_op == 0) {
    if ((reduce_elts == 1) && (reduce_elts_squared == 1)) {
      unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD;
    }
    if ((reduce_elts == 0) && (reduce_elts_squared == 1)) {
      unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD;
    }
    if ((reduce_elts == 1) && (reduce_elts_squared == 0)) {
      unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD;
    }
  } else {
    if ((reduce_elts == 1) && (reduce_elts_squared == 0)) {
      unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX;
    }
  }

  unary_shape.m = m;
  unary_shape.n = n;
  unary_shape.ldi = ld_in;
  unary_shape.ldo = ld_in;
#ifdef FP16_REDUCE_COLSIDX
  unary_shape.in_type = LIBXSMM_DATATYPE_F16;
  unary_shape.out_type = LIBXSMM_DATATYPE_F16;
#else
  unary_shape.in0_type = dtype;
  unary_shape.out_type = dtype;
#endif
  if (dtype == LIBXSMM_DATATYPE_F64) {
    unary_shape.comp_type = LIBXSMM_DATATYPE_F64;
  } else {
    unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
  }

  /* JIT kernel */
  if (n_cols_idx == 0) {
    kernel = libxsmm_dispatch_meltw_unary_v2( unary_type, unary_shape, unary_flags );
  } else {
    if (idx_type == 0) {
      unary_flags = LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES;
    } else {
      unary_flags = LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES;
    }
    if (reduce_on_outputs > 0) {
      unary_flags |= LIBXSMM_EOR(libxsmm_meltw_unary_flags, unary_flags, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INIT_ACC);
    }
    unary_shape.n = 0;
    if (reduce_op == 0) {
      kernel2 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_ADD, unary_shape, unary_flags );
    } else {
      unary_flags = LIBXSMM_EOR(libxsmm_meltw_unary_flags, unary_flags, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_NEG_INF_ACC);
      if (record_idx > 0) {
        unary_flags = LIBXSMM_EOR(libxsmm_meltw_unary_flags, unary_flags, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_RECORD_ARGOP);
        if (idx_type == 0) {
          params2.out.secondary = argop_off;
        } else {
          params2.out.secondary = argop_off_i32;
        }
      }
      kernel2 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MAX, unary_shape, unary_flags );
    }
  }

  /* Setup param struct */
  if (dtype == LIBXSMM_DATATYPE_F32 || dtype == LIBXSMM_DATATYPE_F64) {
    unary_param.in.primary = sinp;
    unary_param.out.primary = result_reduce_elts;
  } else {
    unary_param.in.primary = sinp_lp;
    unary_param.out.primary = result_reduce_elts_lp;
  }
#ifdef FP16_REDUCE_COLSIDX
  params2.in.primary  = sinp_hp;
  params2.out.primary = result_reduce_elts_hp;
#else
  if (dtype == LIBXSMM_DATATYPE_F32) {
    params2.in.primary  = sinp;
    params2.out.primary  = result_reduce_elts;
  } else {
    params2.in.primary  = sinp_lp;
    params2.out.primary  = result_reduce_elts_lp;
  }
#endif
  if (idx_type == 0) {
    params2.in.secondary = cols_ind_array;
  } else {
    params2.in.secondary = cols_ind_array_32bit;
  }

  *res_kernel = kernel;
  *res_kernel2 = kernel2;
  *res_unary_param = unary_param;
  *res_unary_param2 = params2;
}

int main(int argc, char* argv[])
{
  unsigned int m = 64, n = 64, reduce_elts = 1, reduce_elts_squared = 1, reduce_rows = 1, result_size, result_size_check, j, k, iters = 1000, reduce_op = 0;
  unsigned long long n_cols_idx = 0;
  unsigned int idx_type = 0;
  unsigned int record_idx = 0;
  libxsmm_blasint ld_in = 64/*, ld_out = 64*/;
  float  *sinp = NULL, *result_reduce_elts = NULL, *result_reduce_elts_squared = NULL, *ref_result_reduce_elts = NULL, *ref_result_reduce_elts_squared = NULL;
  double *dinp = NULL, *d_result_reduce_elts = NULL, *d_result_reduce_elts_squared = NULL, *d_ref_result_reduce_elts = NULL, *d_ref_result_reduce_elts_squared = NULL;
  unsigned long long *ref_argop_off = NULL, *argop_off = NULL;
  unsigned int *ref_argop_off_i32 = NULL, *argop_off_i32 = NULL;
  char *sinp_lp = NULL;
  char *result_reduce_elts_lp = NULL;
  char *result_reduce_elts_squared_lp = NULL;
#ifdef FP16_REDUCE_COLSIDX
  unsigned short *sinp_hp = NULL, *result_reduce_elts_hp = NULL;
#endif
  unsigned long long *cols_ind_array = NULL;
  unsigned int *cols_ind_array_32bit = NULL;
  libxsmm_meltwfunction_unary kernel = NULL;
  libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
  libxsmm_meltwfunction_unary kernel2 = NULL;
  libxsmm_meltw_unary_param params2;
  libxsmm_matdiff_info norms_elts, norms_elts_squared, diff;
  unsigned long long l_start, l_end;
  double l_total = 0.0, l_total2 = 0.0;
  unsigned int reduce_on_outputs = 0;
  char* dt = NULL;
  libxsmm_datatype dtype = LIBXSMM_DATATYPE_UNSUPPORTED;

  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 1 : atof(env_check));

  libxsmm_init();

  libxsmm_matdiff_clear(&norms_elts);
  libxsmm_matdiff_clear(&norms_elts_squared);
  libxsmm_matdiff_clear(&diff);

  if ( argc > 1 ) m = atoi(argv[1]);
  if ( argc > 2 ) n = atoi(argv[2]);
  if ( argc > 3 ) ld_in = atoi(argv[3]);
  if ( argc > 4 ) reduce_elts = atoi(argv[4]);
  if ( argc > 5 ) reduce_elts_squared = atoi(argv[5]);
  if ( argc > 6 ) reduce_rows = atoi(argv[6]);
  if ( argc > 7 ) reduce_op = atoi(argv[7]);
  if ( argc > 8 ) dt = argv[8];
  if ( argc > 9 ) n_cols_idx = atoi(argv[9]);
  if ( argc > 10 ) iters = atoi(argv[10]);
  if ( argc > 11 ) idx_type = atoi(argv[11]);
  if ( argc > 12 ) record_idx = atoi(argv[12]);
  if ( argc > 13 ) reduce_on_outputs = atoi(argv[13]);

  printf("CL is: %u %u %i %u %u %u %u %s %llu %u %u %u %u\n", m, n, ld_in, reduce_elts, reduce_elts_squared, reduce_rows, reduce_op, dt, n_cols_idx, iters, idx_type, record_idx, reduce_on_outputs);

  m = LIBXSMM_MAX(m,1);
  n = LIBXSMM_MAX(n,1);
  ld_in = LIBXSMM_MAX(ld_in,(libxsmm_blasint)m);
  result_size = (reduce_rows == 1) ? n : (unsigned int)ld_in;
  result_size_check = (reduce_rows == 1) ? n : m;

  dtype = char_to_libxsmm_datatype( dt );

  if ( (dtype != LIBXSMM_DATATYPE_F32)  &&
       (dtype != LIBXSMM_DATATYPE_F64)  &&
       (dtype != LIBXSMM_DATATYPE_F16)  &&
       (dtype != LIBXSMM_DATATYPE_BF16) &&
       (dtype != LIBXSMM_DATATYPE_BF8)  &&
       (dtype != LIBXSMM_DATATYPE_HF8) ) {
    printf(" Only F32,F64,BF16,F16,BF8,HF8 are supported datatypes \n");
    exit(EXIT_FAILURE);
  }

  /* Allocate arrays */
  sinp  = (float*) malloc( sizeof(float)*ld_in*n );
  dinp  = (double*) malloc( sizeof(double)*ld_in*n );
  result_reduce_elts = (float*) malloc( sizeof(float)*result_size*2 );
  d_result_reduce_elts = (double*) malloc( sizeof(double)*result_size*2 );
  /* Fill matrices with random data */
  sfill_matrix ( sinp, ld_in, m, n );
  if (reduce_on_outputs > 0) {
    sfill_matrix ( result_reduce_elts, result_size*2, result_size*2, 1 );
  }
  for ( k = 0; k < ld_in * n; k++) {
    dinp[k] = (double) sinp[k];
  }
  for ( k = 0; k < result_size*2; k++) {
    d_result_reduce_elts[k] = (double) result_reduce_elts[k];
  }
  ref_result_reduce_elts = (float*) malloc(result_size*sizeof(float) );
  ref_result_reduce_elts_squared = (float*) malloc(result_size*sizeof(float) );
  d_ref_result_reduce_elts = (double*) malloc(result_size*sizeof(double) );
  d_ref_result_reduce_elts_squared = (double*) malloc(result_size*sizeof(double) );

  if (dtype == LIBXSMM_DATATYPE_BF16) {
    sinp_lp  = (char*) malloc( sizeof(libxsmm_bfloat16)*ld_in*n );
    libxsmm_rne_convert_fp32_bf16( sinp, (libxsmm_bfloat16*)sinp_lp, ld_in*n );
  } else if (dtype == LIBXSMM_DATATYPE_F16) {
    sinp_lp  = (char*) malloc( sizeof(libxsmm_float16)*ld_in*n );
    libxsmm_rne_convert_fp32_f16( sinp, (libxsmm_float16*)sinp_lp, ld_in*n );
  } else if (dtype == LIBXSMM_DATATYPE_BF8) {
    sinp_lp  = (char*) malloc( sizeof(libxsmm_bfloat8)*ld_in*n );
    libxsmm_rne_convert_fp32_bf8( sinp, (libxsmm_bfloat8*)sinp_lp, ld_in*n );
  } else if (dtype == LIBXSMM_DATATYPE_HF8) {
    sinp_lp  = (char*) malloc( sizeof(libxsmm_hfloat8)*ld_in*n );
    libxsmm_rne_convert_fp32_hf8( sinp, (libxsmm_hfloat8*)sinp_lp, ld_in*n );
  }

  if (dtype == LIBXSMM_DATATYPE_BF16) {
    result_reduce_elts_lp = (char*) malloc( sizeof(libxsmm_bfloat16)*result_size*2 );
    memset(result_reduce_elts_lp, 0, sizeof(libxsmm_bfloat16)*result_size*2 );
    result_reduce_elts_squared_lp = NULL;
    if (reduce_on_outputs > 0) {
      libxsmm_rne_convert_fp32_bf16( result_reduce_elts, (libxsmm_bfloat16*)result_reduce_elts_lp, result_size*2 );
    }
  } else if (dtype == LIBXSMM_DATATYPE_F16) {
    result_reduce_elts_lp = (char*) malloc( sizeof(libxsmm_float16)*result_size*2 );
    memset(result_reduce_elts_lp, 0, sizeof(libxsmm_float16)*result_size*2 );
    result_reduce_elts_squared_lp = NULL;
    if (reduce_on_outputs > 0) {
      libxsmm_rne_convert_fp32_f16( result_reduce_elts, (libxsmm_float16*)result_reduce_elts_lp, result_size*2 );
    }
  } else if (dtype == LIBXSMM_DATATYPE_BF8) {
    result_reduce_elts_lp = (char*) malloc( sizeof(libxsmm_bfloat8)*result_size*2 );
    memset(result_reduce_elts_lp, 0, sizeof(libxsmm_bfloat8)*result_size*2 );
    result_reduce_elts_squared_lp = NULL;
    if (reduce_on_outputs > 0) {
      libxsmm_rne_convert_fp32_bf8( result_reduce_elts, (libxsmm_bfloat8*)result_reduce_elts_lp, result_size*2 );
    }
  } else if (dtype == LIBXSMM_DATATYPE_HF8) {
    result_reduce_elts_lp = (char*) malloc( sizeof(libxsmm_hfloat8)*result_size*2 );
    memset(result_reduce_elts_lp, 0, sizeof(libxsmm_hfloat8)*result_size*2 );
    result_reduce_elts_squared_lp = NULL;
    if (reduce_on_outputs > 0) {
      libxsmm_rne_convert_fp32_hf8( result_reduce_elts, (libxsmm_hfloat8*)result_reduce_elts_lp, result_size*2 );
    }
  }

  cols_ind_array = (unsigned long long*) malloc(n_cols_idx*sizeof(unsigned long long));
  cols_ind_array_32bit = (unsigned int*) malloc(n_cols_idx*sizeof(unsigned int));
  ref_argop_off        = (unsigned long long*) malloc(ld_in*sizeof(unsigned long long));
  ref_argop_off_i32    = (unsigned int*) malloc(ld_in*sizeof(unsigned int));
  argop_off            = (unsigned long long*) malloc(ld_in*sizeof(unsigned long long));
  argop_off_i32        = (unsigned int*) malloc(ld_in*sizeof(unsigned int));

  result_reduce_elts_squared = NULL;

  if (reduce_op == 0) {
    if (reduce_elts == 1) {
      result_reduce_elts_squared = (float*) result_reduce_elts + result_size;
      d_result_reduce_elts_squared = (double*) d_result_reduce_elts + result_size;
      if (dtype == LIBXSMM_DATATYPE_BF16) {
        result_reduce_elts_squared_lp = (char*)((libxsmm_bfloat16*) result_reduce_elts_lp + result_size);
        /* TODO: this needs clean-up */
        libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)result_reduce_elts_lp, ref_result_reduce_elts, result_size );
        libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)result_reduce_elts_squared_lp, ref_result_reduce_elts_squared, result_size );
      } else if (dtype == LIBXSMM_DATATYPE_F16) {
        result_reduce_elts_squared_lp = (char*)((libxsmm_float16*) result_reduce_elts_lp + result_size);
        /* TODO: this needs clean-up */
        libxsmm_convert_f16_f32( (libxsmm_float16*)result_reduce_elts_lp, ref_result_reduce_elts, result_size );
        libxsmm_convert_f16_f32( (libxsmm_float16*)result_reduce_elts_squared_lp, ref_result_reduce_elts_squared, result_size );
      } else if (dtype == LIBXSMM_DATATYPE_BF8) {
        result_reduce_elts_squared_lp = (char*)((libxsmm_bfloat8*) result_reduce_elts_lp + result_size);
        /* TODO: this needs clean-up */
        libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)result_reduce_elts_lp, ref_result_reduce_elts, result_size );
        libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)result_reduce_elts_squared_lp, ref_result_reduce_elts_squared, result_size );
      } else if (dtype == LIBXSMM_DATATYPE_HF8) {
        result_reduce_elts_squared_lp = (char*)((libxsmm_hfloat8*) result_reduce_elts_lp + result_size);
        /* TODO: this needs clean-up */
        libxsmm_convert_hf8_f32( (libxsmm_hfloat8*)result_reduce_elts_lp, ref_result_reduce_elts, result_size );
        libxsmm_convert_hf8_f32( (libxsmm_hfloat8*)result_reduce_elts_squared_lp, ref_result_reduce_elts_squared, result_size );
      } else if (dtype == LIBXSMM_DATATYPE_F32) {
        /* TODO: this needs clean-up */
        memcpy( (void*)ref_result_reduce_elts, (void*)result_reduce_elts, sizeof(float)*result_size );
        memcpy( (void*)ref_result_reduce_elts_squared, (void*)result_reduce_elts_squared, sizeof(float)*result_size );
      } else if (dtype == LIBXSMM_DATATYPE_F64) {
        /* TODO: this needs clean-up */
        memcpy( (void*)d_ref_result_reduce_elts, (void*)d_result_reduce_elts, sizeof(double)*result_size );
        memcpy( (void*)d_ref_result_reduce_elts_squared, (void*)d_result_reduce_elts_squared, sizeof(double)*result_size );
      }
    }
    if ((reduce_elts == 0) && (reduce_elts_squared == 1)) {
      result_reduce_elts_squared = (float*) result_reduce_elts;
      d_result_reduce_elts_squared = (double*) d_result_reduce_elts;
      if (dtype == LIBXSMM_DATATYPE_BF16) {
        result_reduce_elts_squared_lp = result_reduce_elts_lp;
        /* TODO: this needs clean-up */
        libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)result_reduce_elts_squared_lp, ref_result_reduce_elts_squared, result_size );
      } else if (dtype == LIBXSMM_DATATYPE_F16) {
        result_reduce_elts_squared_lp = result_reduce_elts_lp;
        /* TODO: this needs clean-up */
        libxsmm_convert_f16_f32( (libxsmm_float16*)result_reduce_elts_squared_lp, ref_result_reduce_elts_squared, result_size );
      } else if (dtype == LIBXSMM_DATATYPE_BF8) {
        result_reduce_elts_squared_lp = result_reduce_elts_lp;
        /* TODO: this needs clean-up */
        libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)result_reduce_elts_squared_lp, ref_result_reduce_elts_squared, result_size );
      } else if (dtype == LIBXSMM_DATATYPE_HF8) {
        result_reduce_elts_squared_lp = result_reduce_elts_lp;
        /* TODO: this needs clean-up */
        libxsmm_convert_hf8_f32( (libxsmm_hfloat8*)result_reduce_elts_squared_lp, ref_result_reduce_elts_squared, result_size );
      } else if (dtype == LIBXSMM_DATATYPE_F32) {
        memcpy( (void*)ref_result_reduce_elts_squared, (void*)result_reduce_elts_squared, sizeof(float)*result_size );
      } else if (dtype == LIBXSMM_DATATYPE_F64) {
        memcpy( (void*)d_ref_result_reduce_elts_squared, (void*)d_result_reduce_elts_squared, sizeof(double)*result_size );
      }
    }
  }

  /* Initialize cold_ind array */
  for (j = 0; j < n_cols_idx; j++) {
    cols_ind_array[j] = rand() % n;
    cols_ind_array_32bit[j] = (unsigned int) cols_ind_array[j];
  }

  printf("Running reference reduce kernel... \n");
  if (dtype == LIBXSMM_DATATYPE_F64) {
    reference_reduce_kernel_f64( m, n, ld_in, (libxsmm_blasint)n_cols_idx, dinp, d_ref_result_reduce_elts, d_ref_result_reduce_elts_squared, cols_ind_array, reduce_op, reduce_rows, record_idx, ref_argop_off, dtype, reduce_on_outputs );
  } else {
    reference_reduce_kernel( m, n, ld_in, (libxsmm_blasint)n_cols_idx, sinp, ref_result_reduce_elts, ref_result_reduce_elts_squared, cols_ind_array, reduce_op, reduce_rows, record_idx, ref_argop_off, dtype, reduce_on_outputs );
  }

  printf("JITing reduce kernel... \n");
  setup_tpp_kernel_and_param_struct( &kernel, &kernel2, &unary_param, &params2, m, n, ld_in, (libxsmm_blasint)n_cols_idx, reduce_rows, reduce_op, reduce_elts, reduce_elts_squared, dtype, idx_type,
      (dtype == LIBXSMM_DATATYPE_F64) ? (float*)dinp : sinp, (dtype == LIBXSMM_DATATYPE_F64) ? (float*)d_result_reduce_elts : result_reduce_elts,
      sinp_lp, result_reduce_elts_lp,
      cols_ind_array, cols_ind_array_32bit,
      record_idx, argop_off, argop_off_i32, reduce_on_outputs );

  if (n_cols_idx == 0) {
    /* Call JITed kernel and compare results */
    printf("Calling JITed reduce kernel... \n");
    kernel( &unary_param );
  } else {
    printf("Calling JITed reduce cols idx kernel... \n");
    params2.in.tertiary = &n_cols_idx;
    kernel2( &params2 );
  }

  /* compare */
  printf("##########################################\n");
  if (n_cols_idx == 0) {
    if (dtype == LIBXSMM_DATATYPE_F64) {
      printf("#   FP64 Correctness - Eltwise reduce    #\n");
    } else if (dtype == LIBXSMM_DATATYPE_F32) {
      printf("#   FP32 Correctness - Eltwise reduce    #\n");
    } else if (dtype == LIBXSMM_DATATYPE_BF8) {
      printf("#   BF8  Correctness - Eltwise reduce    #\n");
    } else if (dtype == LIBXSMM_DATATYPE_HF8) {
      printf("#   HF8  Correctness - Eltwise reduce    #\n");
    } else if (dtype == LIBXSMM_DATATYPE_F16) {
      printf("#   F16  Correctness - Eltwise reduce    #\n");
    } else {
      printf("#   BF16 Correctness - Eltwise reduce    #\n");
    }
  } else {
    if (dtype == LIBXSMM_DATATYPE_F32) {
      printf("# FP32 Correctness - Eltwise red. colsidx#\n");
    } else if (dtype == LIBXSMM_DATATYPE_BF8) {
      printf("# BF8  Correctness - Eltwise red. colsidx#\n");
    } else if (dtype == LIBXSMM_DATATYPE_HF8) {
      printf("# HF8  Correctness - Eltwise red. colsidx#\n");
    } else if (dtype == LIBXSMM_DATATYPE_F16) {
      printf("# F16  Correctness - Eltwise red. colsidx#\n");
    } else {
      printf("# BF16 Correctness - Eltwise red. colsidx#\n");
    }
  }

  if (reduce_elts > 0) {
    printf("##########################################\n");
    if (dtype == LIBXSMM_DATATYPE_BF16) {
      libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)result_reduce_elts_lp, result_reduce_elts, result_size );
    } else if (dtype == LIBXSMM_DATATYPE_F16) {
      libxsmm_convert_f16_f32( (libxsmm_float16*)result_reduce_elts_lp, result_reduce_elts, result_size );
    } else if (dtype == LIBXSMM_DATATYPE_BF8) {
      libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)result_reduce_elts_lp, result_reduce_elts, result_size );
    } else if (dtype == LIBXSMM_DATATYPE_HF8) {
      libxsmm_convert_hf8_f32( (libxsmm_hfloat8*)result_reduce_elts_lp, result_reduce_elts, result_size );
    }
    if (dtype == LIBXSMM_DATATYPE_F64) {
      libxsmm_matdiff(&norms_elts, LIBXSMM_DATATYPE_F64, result_size_check, 1, d_ref_result_reduce_elts, d_result_reduce_elts, 0, 0);
    } else {
      libxsmm_matdiff(&norms_elts, LIBXSMM_DATATYPE_F32, result_size_check, 1, ref_result_reduce_elts, result_reduce_elts, 0, 0);
    }
    printf("L1 reference  : %.25g\n", norms_elts.l1_ref);
    printf("L1 test       : %.25g\n", norms_elts.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_elts.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_elts.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_elts.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_elts.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_elts.normf_rel);
    libxsmm_matdiff_reduce(&diff, &norms_elts);
  }

  /* compare */
  if (reduce_elts_squared > 0) {
    if (n_cols_idx == 0) {
      if (dtype == LIBXSMM_DATATYPE_BF16) {
        libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)result_reduce_elts_squared_lp, result_reduce_elts_squared, result_size );
      } else if (dtype == LIBXSMM_DATATYPE_F16) {
        libxsmm_convert_f16_f32( (libxsmm_float16*)result_reduce_elts_squared_lp, result_reduce_elts_squared, result_size );
      } else if (dtype == LIBXSMM_DATATYPE_BF8) {
        libxsmm_convert_bf8_f32( (libxsmm_bfloat8*)result_reduce_elts_squared_lp, result_reduce_elts_squared, result_size );
      } else if (dtype == LIBXSMM_DATATYPE_HF8) {
        libxsmm_convert_hf8_f32( (libxsmm_hfloat8*)result_reduce_elts_squared_lp, result_reduce_elts_squared, result_size );
      }

      printf("##########################################\n");
      if (dtype == LIBXSMM_DATATYPE_F64) {
        printf("# FP64 Correctness - Eltwise-square reduce  #\n");
      } else if (dtype == LIBXSMM_DATATYPE_F32) {
        printf("# FP32 Correctness - Eltwise-square reduce  #\n");
      } else if (dtype == LIBXSMM_DATATYPE_BF8) {
        printf("# BF8  Correctness - Eltwise-square reduce  #\n");
      } else if (dtype == LIBXSMM_DATATYPE_HF8) {
        printf("# HF8  Correctness - Eltwise-square reduce  #\n");
      } else if (dtype == LIBXSMM_DATATYPE_F16) {
        printf("# F16  Correctness - Eltwise-square reduce  #\n");
      } else {
        printf("# BF16 Correctness - Eltwise-square reduce  #\n");
      }
      printf("##########################################\n");
      if (dtype == LIBXSMM_DATATYPE_F64) {
        libxsmm_matdiff(&norms_elts_squared, LIBXSMM_DATATYPE_F64, result_size_check, 1, d_ref_result_reduce_elts_squared, d_result_reduce_elts_squared, 0, 0);
      } else {
        libxsmm_matdiff(&norms_elts_squared, LIBXSMM_DATATYPE_F32, result_size_check, 1, ref_result_reduce_elts_squared, result_reduce_elts_squared, 0, 0);
      }
      printf("L1 reference  : %.25g\n", norms_elts_squared.l1_ref);
      printf("L1 test       : %.25g\n", norms_elts_squared.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_elts_squared.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_elts_squared.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_elts_squared.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_elts_squared.linf_rel);
      printf("Check-norm    : %.24f\n\n", norms_elts_squared.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_elts_squared);
    }
  }

  if (record_idx > 0) {
    for (k = 0; k < m; k++) {
      ref_argop_off_i32[k] = LIBXSMM_CAST_UINT(ref_argop_off[k]);
    }
    if (idx_type == 0) {
      for (k = 0; k < m; k++) {
        argop_off_i32[k] = LIBXSMM_CAST_UINT(argop_off[k]);
      }
    }
    printf("##########################################\n");
    printf("# Arg idx correctness  #\n");
    printf("##########################################\n");
    libxsmm_matdiff(&norms_elts, LIBXSMM_DATATYPE_I32, m, 1, ref_argop_off_i32, argop_off_i32, 0, 0);
    printf("L1 reference  : %.25g\n", norms_elts.l1_ref);
    printf("L1 test       : %.25g\n", norms_elts.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_elts.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_elts.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_elts.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_elts.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_elts.normf_rel);
    libxsmm_matdiff_reduce(&diff, &norms_elts);
  }

  l_start = libxsmm_timer_tick();
  /* Calculate reference results... */
  for (k = 0; k < iters; k++) {
    if (dtype == LIBXSMM_DATATYPE_F64) {
      reference_reduce_kernel_f64( m, n, ld_in, (libxsmm_blasint)n_cols_idx, dinp, d_ref_result_reduce_elts, d_ref_result_reduce_elts_squared, cols_ind_array, reduce_op, reduce_rows, record_idx, ref_argop_off, dtype, reduce_on_outputs );
    } else {
      reference_reduce_kernel( m, n, ld_in, (libxsmm_blasint)n_cols_idx, sinp, ref_result_reduce_elts, ref_result_reduce_elts_squared, cols_ind_array, reduce_op, reduce_rows, record_idx, ref_argop_off, dtype, reduce_on_outputs );
    }
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Reference time = %.5g\n", ((double)(l_total)));

  l_start = libxsmm_timer_tick();
  if (n_cols_idx == 0) {
    for (k = 0; k < iters; k++) {
      kernel( &unary_param );
    }
  } else {
    params2.in.tertiary = &n_cols_idx;
    for (k = 0; k < iters; k++) {
      kernel2( &params2 );
    }
  }
  l_end = libxsmm_timer_tick();
  l_total2 = libxsmm_timer_duration(l_start, l_end);
  printf("Optimized time = %.5g\n", ((double)(l_total2)));
  printf("Speedup is = %.5g\n", ((double)(l_total/l_total2)));

  free(sinp);
  free(result_reduce_elts);
  free(dinp);
  free(d_result_reduce_elts);
  free(sinp_lp);
  free(result_reduce_elts_lp);
  free(ref_result_reduce_elts);
  free(ref_result_reduce_elts_squared);
  free(d_ref_result_reduce_elts);
  free(d_ref_result_reduce_elts_squared);

  {
    const char *const env_check_scale = getenv("CHECK_SCALE");
    const double check_scale = LIBXSMM_ABS(0 == env_check_scale ? 1.0 : atof(env_check_scale));
    if (LIBXSMM_NEQ(0, check) && (check < 100.0 * check_scale * diff.normf_rel)) {
      fprintf(stdout, "FAILED unary reduce with an error of %f%%!\n", 100.0 * diff.normf_rel);
      exit(EXIT_FAILURE);
    }
  }

  fprintf(stdout, "SUCCESS unnary reduce\n" );
  return EXIT_SUCCESS;
}
