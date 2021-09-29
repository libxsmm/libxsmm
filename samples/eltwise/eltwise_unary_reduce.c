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
/*#define FP16_REDUCE_COLSIDX*/
#ifdef FP16_REDUCE_COLSIDX
#include <immintrin.h>
#endif

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

int main(int argc, char* argv[])
{
  unsigned int m = 64, n = 64, reduce_elts = 1, reduce_elts_squared = 1, reduce_rows = 1, result_size, result_size_check, i, j, jj, k, iters = 10000, n_cols_idx = 0, reduce_op = 0, use_bf16 = 0;
  libxsmm_blasint ld_in = 64/*, ld_out = 64*/;
  float  *sinp, *result_reduce_elts, *result_reduce_elts_squared, *ref_result_reduce_elts, *ref_result_reduce_elts_squared;
  libxsmm_bfloat16 *sinp_bf16 = NULL;
  libxsmm_bfloat16 *result_reduce_elts_bf16 = NULL;
  libxsmm_bfloat16 *result_reduce_elts_squared_bf16 = NULL;
#ifdef FP16_REDUCE_COLSIDX
  unsigned short *sinp_hp, *result_reduce_elts_hp;
#endif
  unsigned long long *cols_ind_array;
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type  unary_type = LIBXSMM_MELTW_TYPE_UNARY_NONE;
  libxsmm_meltwfunction_unary kernel = NULL;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltwfunction_reduce_cols_idx kernel2 = NULL;
  libxsmm_meltw_reduce_cols_idx_param params2;
  libxsmm_matdiff_info norms_elts, norms_elts_squared, diff;
  unsigned long long l_start, l_end;
  double l_total = 0.0, l_total2 = 0.0;

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
  if ( argc > 8 ) use_bf16 = atoi(argv[8]);
  if ( argc > 9 ) n_cols_idx = atoi(argv[9]);
  if ( argc > 10 ) iters = atoi(argv[10]);

  printf("CL is: %d %d %d %d %d %d %d %d %d %d\n", m, n, ld_in, reduce_elts, reduce_elts_squared, reduce_rows, reduce_op, use_bf16, n_cols_idx, iters);

#if 0
  libxsmm_meltw_opreduce_vecs_flags opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIDX_VECIN | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_MUL | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_SUM | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_SCALE_OP_RESULT;
  libxsmm_meltwfunction_opreduce_vecs_idx new_kernel = libxsmm_dispatch_meltw_opreduce_vecs_idx(m, &ld_in, &ld_in, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_I64, opredop_flags);
#endif

  m = LIBXSMM_MAX(m,1);
  n = LIBXSMM_MAX(n,1);
  ld_in = LIBXSMM_MAX(ld_in,(libxsmm_blasint)m);
  result_size = (reduce_rows == 1) ? n : ld_in;
  result_size_check = (reduce_rows == 1) ? n : m;

  /* Allocate arrays  */
  sinp  = (float*) malloc( ld_in*n*sizeof(float) );
  result_reduce_elts = (float*) malloc(2 * result_size*sizeof(float) );

  if (use_bf16 == 1) {
    sinp_bf16  = (libxsmm_bfloat16*) malloc( ld_in*n*sizeof(libxsmm_bfloat16) );
    result_reduce_elts_bf16 = (libxsmm_bfloat16*) malloc(2 * result_size*sizeof(libxsmm_bfloat16) );
    memset(result_reduce_elts_bf16, 0, 2 * result_size * sizeof(libxsmm_bfloat16) );
    result_reduce_elts_squared_bf16 = NULL;
  }

  ref_result_reduce_elts = (float*) malloc(result_size*sizeof(float) );
  ref_result_reduce_elts_squared = (float*) malloc(result_size*sizeof(float) );
  cols_ind_array = (unsigned long long*) malloc(n_cols_idx*sizeof(unsigned long long));
  result_reduce_elts_squared = NULL;

  /* Fill matrices with random data */
  sfill_matrix ( sinp, ld_in, m, n );

  if (use_bf16 == 1) {
    libxsmm_rne_convert_fp32_bf16( sinp, sinp_bf16, ld_in*n );
    libxsmm_convert_bf16_f32( sinp_bf16, sinp, ld_in*n );
  }

#ifdef FP16_REDUCE_COLSIDX
  sinp_hp  = (unsigned short*) malloc( ld_in*n*sizeof(unsigned short) );
  result_reduce_elts_hp = (unsigned short*) malloc(result_size*sizeof(unsigned short) );
  for (i = 0; i < m; i++) {
    ref_result_reduce_elts[i] = 0;
    result_reduce_elts_hp[i] = _cvtss_sh(ref_result_reduce_elts[i], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  }
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      sinp_hp[j*ld_in + i] = _cvtss_sh(sinp[j*ld_in + i], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
  }
#endif

  if (reduce_op == 0) {
    /* Calculate reference results...  */
    if (reduce_rows == 1) {
      for (j = 0; j < n; j++) {
        ref_result_reduce_elts[j] = 0;
        ref_result_reduce_elts_squared[j] = 0;
        for (i = 0; i < m; i++) {
          ref_result_reduce_elts[j] += sinp[j*ld_in + i];
          ref_result_reduce_elts_squared[j] += sinp[j*ld_in + i] * sinp[j*ld_in + i];
        }
      }
    } else {
      if (n_cols_idx == 0) {
        /* In this case we reduce columns */
        for (i = 0; i < m; i++) {
          ref_result_reduce_elts[i] = 0;
          ref_result_reduce_elts_squared[i] = 0;
          for (j = 0; j < n; j++) {
            ref_result_reduce_elts[i] += sinp[j*ld_in + i];
            ref_result_reduce_elts_squared[i] += sinp[j*ld_in + i] * sinp[j*ld_in + i];
          }
        }
      } else {
        /* In this case we reduce columns */
        for (j = 0; j < n_cols_idx; j++) {
          cols_ind_array[j] = rand() % n;
        }
        for (i = 0; i < m; i++) {
          ref_result_reduce_elts[i] = 0;
          result_reduce_elts[i] = 0;
          for (jj = 0; jj < n_cols_idx; jj++) {
            j = cols_ind_array[jj];
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
      /* In this case we reduce columns */
      for (i = 0; i < m; i++) {
        ref_result_reduce_elts[i] = sinp[i];
        for (j = 0; j < n; j++) {
          ref_result_reduce_elts[i] = LIBXSMM_MAX( sinp[j*ld_in + i], ref_result_reduce_elts[i]);
        }
      }
    }
  }

  if (reduce_rows == 1) {
    unary_flags |= LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS;
  } else {
    unary_flags |= LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  }

  if (reduce_op == 0) {
    if ((reduce_elts == 1) && (reduce_elts_squared == 1)) {
      result_reduce_elts_squared = (float*) result_reduce_elts + result_size;
      if (use_bf16 == 1) {
        result_reduce_elts_squared_bf16 = (libxsmm_bfloat16*) result_reduce_elts_bf16 + result_size;
      }
      unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD;
    }
    if ((reduce_elts == 0) && (reduce_elts_squared == 1)) {
      result_reduce_elts_squared = (float*) result_reduce_elts;
      if (use_bf16 == 1) {
        result_reduce_elts_squared_bf16 = (libxsmm_bfloat16*) result_reduce_elts_bf16;
      }
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

  printf("JITing reduce kernel... \n");
  if (n_cols_idx == 0) {
    if (use_bf16 == 0) {
      kernel = libxsmm_dispatch_meltw_unary(m, n, &ld_in, &ld_in, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, unary_type);
    } else {
      kernel = libxsmm_dispatch_meltw_unary(m, n, &ld_in, &ld_in, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, unary_flags, unary_type);
    }

    /* Call JITed kernel and compare result  */
    printf("Calling JITed reduce kernel... \n");
    if (use_bf16 == 0) {
      unary_param.in.primary = sinp;
      unary_param.out.primary = result_reduce_elts;
    } else {
      unary_param.in.primary = sinp_bf16;
      unary_param.out.primary = result_reduce_elts_bf16;
    }
    kernel( &unary_param );
  } else {
#ifdef FP16_REDUCE_COLSIDX
    kernel2 = libxsmm_dispatch_meltw_reduce_cols_idx(m, &ld_in, &ld_in, LIBXSMM_DATATYPE_F16, LIBXSMM_DATATYPE_F16, LIBXSMM_DATATYPE_I64);
#else
    if (use_bf16 == 0) {
      kernel2 = libxsmm_dispatch_meltw_reduce_cols_idx(m, &ld_in, &ld_in, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_I64);
    } else {
      kernel2 = libxsmm_dispatch_meltw_reduce_cols_idx(m, &ld_in, &ld_in, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_I64);
    }
#endif
    /* Call JITed kernel and compare result  */
    printf("Calling JITed reduce cols idx kernel... \n");
    params2.n = n_cols_idx;
    params2.ind_ptr = cols_ind_array;
#ifdef FP16_REDUCE_COLSIDX
    params2.inp_ptr = sinp_hp;
    params2.out_ptr = result_reduce_elts_hp;
#else
    if (use_bf16 == 0) {
      params2.inp_ptr = sinp;
      params2.out_ptr = result_reduce_elts;
    } else {
      params2.inp_ptr = sinp_bf16;
      params2.out_ptr = result_reduce_elts_bf16;
    }
#endif
#ifdef FP16_REDUCE_COLSIDX
    for (i = 0; i < m; i++) {
      result_reduce_elts_hp[i] = _cvtss_sh(result_reduce_elts[i], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }
#endif
    kernel2( &params2 );
#ifdef FP16_REDUCE_COLSIDX
    for (i = 0; i < m; i++) {
      result_reduce_elts[i] = _cvtsh_ss(result_reduce_elts_hp[i]);
    }
#endif
  }

  /* compare */
  printf("##########################################\n");
  if (use_bf16 == 0) {
    if (n_cols_idx == 0) {
      printf("#   FP32 Correctness - Eltwise reduce         #\n");
    } else {
      printf("#   FP32 Correctness - Eltwise reduce colsidx #\n");
    }
  } else {
    if (n_cols_idx == 0) {
      printf("#   BF16 Correctness - Eltwise reduce         #\n");
    } else {
      printf("#   BF16 Correctness - Eltwise reduce colsidx #\n");
    }
  }

  if (reduce_elts > 0) {
    printf("##########################################\n");
    if (use_bf16 == 1) {
      libxsmm_convert_bf16_f32( result_reduce_elts_bf16, result_reduce_elts, result_size );
    }
    libxsmm_matdiff(&norms_elts, LIBXSMM_DATATYPE_F32, result_size_check, 1, ref_result_reduce_elts, result_reduce_elts, 0, 0);
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
      if (use_bf16 == 1) {
        libxsmm_convert_bf16_f32( result_reduce_elts_squared_bf16, result_reduce_elts_squared, result_size );
      }
      printf("##########################################\n");
      if (use_bf16 == 0) {
        printf("# FP32 Correctness - Eltwise-square reduce  #\n");
      } else {
        printf("# BF16 Correctness - Eltwise-square reduce  #\n");
      }
      printf("##########################################\n");
      libxsmm_matdiff(&norms_elts_squared, LIBXSMM_DATATYPE_F32, result_size_check, 1, ref_result_reduce_elts_squared, result_reduce_elts_squared, 0, 0);
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

  l_start = libxsmm_timer_tick();
  /* Calculate reference results...  */
  for (k = 0; k < iters; k++) {
    if (reduce_op == 0) {

      if (reduce_rows == 1) {
        for (j = 0; j < n; j++) {
          ref_result_reduce_elts[j] = 0;
          ref_result_reduce_elts_squared[j] = 0;
          for (i = 0; i < m; i++) {
            ref_result_reduce_elts[j] += sinp[j*ld_in + i];
            ref_result_reduce_elts_squared[j] += sinp[j*ld_in + i] * sinp[j*ld_in + i];
          }
        }
      } else {
        if (n_cols_idx == 0) {
          /* In this case we reduce columns */
          for (i = 0; i < m; i++) {
            ref_result_reduce_elts[i] = 0;
            ref_result_reduce_elts_squared[i] = 0;
            for (j = 0; j < n; j++) {
              ref_result_reduce_elts[i] += sinp[j*ld_in + i];
              ref_result_reduce_elts_squared[i] += sinp[j*ld_in + i] * sinp[j*ld_in + i];
            }
          }
        } else {
          /* In this case we reduce columns */
          for (j = 0; j < n_cols_idx; j++) {
            cols_ind_array[j] = rand() % n_cols_idx;
          }
          for (i = 0; i < m; i++) {
            ref_result_reduce_elts[i] = 0;
            for (jj = 0; jj < n_cols_idx; jj++) {
              j = cols_ind_array[jj];
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
        /* In this case we reduce columns */
        for (i = 0; i < m; i++) {
          ref_result_reduce_elts[i] = sinp[i];
          for (j = 0; j < n; j++) {
            ref_result_reduce_elts[i] = LIBXSMM_MAX( sinp[j*ld_in + i], ref_result_reduce_elts[i]);
          }
        }
      }
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
  if (use_bf16 == 1) {
    free(sinp_bf16);
    free(result_reduce_elts_bf16);
  }
  free(ref_result_reduce_elts);
  free(ref_result_reduce_elts_squared);

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

