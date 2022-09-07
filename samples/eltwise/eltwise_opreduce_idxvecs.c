/*********************************************************************************************
* Copyright (c) Intel Corporation, Friedrich Schiller University Jena - All rights reserved. *
* This file is part of the LIBXSMM library.                                                  *
*                                                                                            *
* For information on the license, see the LICENSE file.                                      *
* Further information: https://github.com/libxsmm/libxsmm/                                   *
* SPDX-License-Identifier: BSD-3-Clause                                                      *
*********************************************************************************************/
/* Evangelos Georganas (Intel Corp.), Antonio Noack (FSU Jena)
*********************************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define OP_COPY 0
#define OP_ADD  1
#define OP_MUL  2
#define OP_SUB  3
#define OP_DIV  4
#define OP_DOT  5
#define OPORDER_VECIN_VECIDX 0
#define OPORDER_VECIDX_VECIN 1
#define NO_SCALE_OP_RESULT 0
#define SCALE_OP_RESULT 1
#define REDOP_NONE  0
#define REDOP_SUM   1
#define REDOP_MAX   2
#define REDOP_MIN   3
#define OFFSET 1.0

LIBXSMM_INLINE
void sfill_matrix ( float *matrix, unsigned int ld, unsigned int m, unsigned int n, unsigned int avoid_small_vals )
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
        if (avoid_small_vals > 0) {
          if (dtmp < 0.0 && dtmp > -1.0 * OFFSET) {
             dtmp = dtmp - OFFSET;
          }
          if (dtmp > 0.0 && dtmp <  OFFSET) {
             dtmp = dtmp + OFFSET;
          }
        }
        matrix [ (j-1)*ld + (i-1) ] = (float) dtmp;
     }
  }
}

LIBXSMM_INLINE
void shuffle_array(unsigned long long *array, int n) {
  if (n > 1)
  {
    int i;
    for (i = 0; i < n - 1; i++)
    {
      int j = i + rand() / (RAND_MAX / (n - i) + 1);
      unsigned long long t = array[j];
      array[j] = array[i];
      array[i] = t;
    }
  }
}

int main(int argc, char* argv[])
{
  unsigned int m = 64, n = 64, i, j, jj, k, iters = 10000, n_cols_idx = 32, op = 0, op_order = 0, scale_op_res = 0, redop = 0, use_implicit_idx = 0, _j = 0, _i = 0;
  libxsmm_blasint ld_in = 64;
  float  *inp_matrix, *result, *ref_result, *inp_matrix2, *scale_vals, *vec_in;
  libxsmm_bfloat16 *inp_matrix_bf16, *result_bf16, *inp_matrix_bf162, *scale_vals_bf16;
  unsigned long long *cols_ind_array, *cols_ind_array2, *all_ns;
  unsigned long long *argop_off_vec_0, *argop_off_vec_1, *ref_argop_off_vec_0, *ref_argop_off_vec_1;
  unsigned int  *argop_off_vec_0_i32, *argop_off_vec_1_i32, *ref_argop_off_vec_0_i32, *ref_argop_off_vec_1_i32;
  unsigned int  *cols_ind_array_i32, *cols_ind_array2_i32;
  libxsmm_meltw_opreduce_vecs_idx_param params /*= { 0 }*/;
  libxsmm_meltw_opreduce_vecs_flags opredop_flags;
  libxsmm_meltwfunction_opreduce_vecs_idx kernel;
  libxsmm_matdiff_info norms_elts, diff;
  unsigned long long l_start, l_end;
  double l_total = 0.0, l_total2 = 0.0;
  char opname[50];
  char opordername[50];
  char scaleopresname[50];
  char redopname[50];
  unsigned int use_bf16 = 0;
  unsigned short bcast_factor = 0;
  unsigned int argop_mode = 0;
  unsigned int idx_mode = 0;
  unsigned int use_regular_vecin = 0;
  unsigned int _n = 0;
  unsigned int argop_vec_0;
  unsigned int argop_vec_1;
  libxsmm_datatype idx_dtype;
  unsigned int avoid_small_vals = 0;

  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 1 : atof(env_check));

  libxsmm_init();
  libxsmm_matdiff_clear(&norms_elts);
  libxsmm_matdiff_clear(&diff);

  if(argc == 1){
    /* probably help is wanted */
    printf(" Error! Usage: %s [M=64] [N=64] [N_COLS_IDX=32] [LD_IN=64] [OP=0] [OP_ORDER=0] [SCALE_OP_RES=0] [REDOP=0] [REG_VECIN=0] [IMPLICIT_IDX=0] [ARGOP_MODE=0] [IDX_MODE=0] [ITERS=10000] [BF16=0] [BCAST_FACTOR=0]\n", argv[0] );
    exit(-1);
  }

  if ( argc > 1 ) m           = atoi(argv[1]);
  if ( argc > 2 ) n           = atoi(argv[2]);
  if ( argc > 3 ) n_cols_idx  = atoi(argv[3]);
  if ( argc > 4 ) ld_in       = atoi(argv[4]);
  if ( argc > 5 ) op          = atoi(argv[5]);
  if ( argc > 6 ) op_order    = atoi(argv[6]);
  if ( argc > 7 ) scale_op_res= atoi(argv[7]);
  if ( argc > 8 ) redop       = atoi(argv[8]);
  if ( argc > 9 ) use_regular_vecin = atoi(argv[9]);
  if ( argc > 10 ) use_implicit_idx = atoi(argv[10]);
  if ( argc > 11 ) argop_mode      = atoi(argv[11]);
  if ( argc > 12 ) idx_mode        = atoi(argv[12]);
  if ( argc > 13 ) iters       = atoi(argv[13]);
  if ( argc > 14 ) use_bf16    = atoi(argv[14]);
  if ( argc > 15 ) bcast_factor = (unsigned short)atoi(argv[15]);

  if (op == OP_DIV)  avoid_small_vals = 1;

  /* Some basic arg checking... */
  if ((use_regular_vecin > 0) && (argop_mode > 0)) {
    fprintf(stderr, "When using regular vec_in (i.e. non-indexed) using argop params is meaningless...\n");
    exit(EXIT_FAILURE);
  }

  if ((op == 0) && (op_order == 0) && (use_regular_vecin > 0)) {
    fprintf(stderr, "When using COPY OP and regular vec_in, only supported OP order is VECIDX_VECIN...\n");
    exit(EXIT_FAILURE);
  }

  if ((op == 0) && (argop_mode > 0)) {
    if ((op_order == 0) && (argop_mode == 1 || argop_mode == 3)) {
      fprintf(stderr, "When using COPY OP and and order VECIN_VECIDX the only argop_modes that make sense are 2 or 0...\n");
      exit(EXIT_FAILURE);
    }
    if ((op_order == 1) && (argop_mode == 2 || argop_mode == 3)) {
      fprintf(stderr, "When using COPY OP and and order VECIDX_VECIN the only argop_modes that make sense are 1 or 0...\n");
      exit(EXIT_FAILURE);
    }
  }

  if ((redop == 0) && (use_regular_vecin == 0)) {
    fprintf(stderr, "If redop is NONE only regular_vec_in makes sense...\n");
    exit(EXIT_FAILURE);
  }

  argop_vec_0 = (argop_mode & 1);
  argop_vec_1 = (argop_mode & 2) >> 1;

  if (idx_mode == 0) {
    idx_dtype = LIBXSMM_DATATYPE_I32;
  } else {
    idx_dtype = LIBXSMM_DATATYPE_I64;
  }

  if (op == OP_COPY) {
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_COPY;
    sprintf(opname, "COPY");
  } else if (op == OP_ADD) {
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_ADD;
    sprintf(opname, "ADD");
  } else if (op == OP_MUL) {
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_MUL;
    sprintf(opname, "MUL");
  } else if (op == OP_SUB) {
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_SUB;
    sprintf(opname, "SUB");
  } else if (op == OP_DIV) {
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_DIV;
    sprintf(opname, "DIV");
  } else if (op == OP_DOT) {
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_DOT;
    sprintf(opname, "DOT");
    printf("ERROR: DOT OP requested, and is not supported yet!!!\n");
    return EXIT_SUCCESS;
  } else {
    printf("ERROR: Invalid OP requested!!!\n");
    return EXIT_SUCCESS;
  }

  if (op_order == OPORDER_VECIN_VECIDX) {
    opredop_flags = LIBXSMM_EOR(libxsmm_meltw_opreduce_vecs_flags, opredop_flags, LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIN_VECIDX);
    sprintf(opordername, "VECIN_VECIDX");
  } else if (op_order == OPORDER_VECIDX_VECIN) {
    opredop_flags = LIBXSMM_EOR(libxsmm_meltw_opreduce_vecs_flags, opredop_flags, LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIDX_VECIN);
    sprintf(opordername, "VECIDX_VECIN");
  } else {
    printf("ERROR: Invalid OP_ORDER requested!!!\n");
    return EXIT_SUCCESS;
  }

  if (scale_op_res == NO_SCALE_OP_RESULT) {
    sprintf(scaleopresname, "NO");
  } else if (scale_op_res == SCALE_OP_RESULT) {
    opredop_flags = LIBXSMM_EOR(libxsmm_meltw_opreduce_vecs_flags, opredop_flags, LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_SCALE_OP_RESULT);
    sprintf(scaleopresname, "YES");
  } else {
    printf("ERROR: Scale OP result should be 0 or 1!!!\n");
    return EXIT_SUCCESS;
  }

  if (redop == REDOP_NONE) {
    opredop_flags = LIBXSMM_EOR(libxsmm_meltw_opreduce_vecs_flags, opredop_flags, LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_NONE);
    sprintf(redopname, "NONE");
  } else if (redop == REDOP_SUM) {
    opredop_flags = LIBXSMM_EOR(libxsmm_meltw_opreduce_vecs_flags, opredop_flags, LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_SUM);
    sprintf(redopname, "SUM");
  } else if (redop == REDOP_MAX) {
    opredop_flags = LIBXSMM_EOR(libxsmm_meltw_opreduce_vecs_flags, opredop_flags, LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MAX);
    sprintf(redopname, "MAX");
  } else if (redop == REDOP_MIN) {
    opredop_flags = LIBXSMM_EOR(libxsmm_meltw_opreduce_vecs_flags, opredop_flags, LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MIN);
    sprintf(redopname, "MIN");
  } else {
    printf("ERROR: Invalid REDOP requested!!!\n");
    return EXIT_SUCCESS;
  }

  if (op != OP_COPY) {
    if (use_regular_vecin == 0) {
      if (use_implicit_idx > 0) {
        opredop_flags = LIBXSMM_EOR(libxsmm_meltw_opreduce_vecs_flags, opredop_flags, LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_IMPLICIT_INDEXED_VEC);
      } else {
        opredop_flags = LIBXSMM_EOR(libxsmm_meltw_opreduce_vecs_flags, opredop_flags, LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_INDEXED_VEC);
      }
    }
  }

  if (argop_vec_0 == 1) {
    opredop_flags = LIBXSMM_EOR(libxsmm_meltw_opreduce_vecs_flags, opredop_flags, LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_RECORD_ARGOP_OFF_VEC_0);
  }
  if (argop_vec_1 == 1) {
    opredop_flags = LIBXSMM_EOR(libxsmm_meltw_opreduce_vecs_flags, opredop_flags, LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_RECORD_ARGOP_OFF_VEC_1);
  }

  if ((op == OP_COPY) && (use_implicit_idx > 0)) {
    opredop_flags = LIBXSMM_EOR(libxsmm_meltw_opreduce_vecs_flags, opredop_flags, LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_IMPLICIT_INDEXED_VECIDX);
  }

  if (use_bf16 == 0) {
    kernel = libxsmm_dispatch_meltw_opreduce_vecs_idx(m, &ld_in, &ld_in, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, idx_dtype, opredop_flags, bcast_factor);
  } else {
    kernel = libxsmm_dispatch_meltw_opreduce_vecs_idx(m, &ld_in, &ld_in, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, idx_dtype, opredop_flags, bcast_factor);
  }
  m = LIBXSMM_MAX(m,1);
  n = LIBXSMM_MAX(n,1);
  _n = (use_regular_vecin > 0) ? 1 : n;
  ld_in = LIBXSMM_MAX(ld_in,(libxsmm_blasint)m);

  /* Allocate arrays  */
  inp_matrix              = (float*) malloc(sizeof(float)*ld_in*n);
  result                  = (float*) malloc(sizeof(float)*ld_in);
  ref_result              = (float*) malloc(sizeof(float)*ld_in);
  inp_matrix2             = (float*) malloc(sizeof(float)*ld_in*_n);
  cols_ind_array          = (unsigned long long*) malloc(sizeof(unsigned long long)*n_cols_idx);
  cols_ind_array2         = (unsigned long long*) malloc(sizeof(unsigned long long)*n_cols_idx);
  cols_ind_array_i32      = (unsigned int*) malloc(sizeof(unsigned int)*n_cols_idx);
  cols_ind_array2_i32     = (unsigned int*) malloc(sizeof(unsigned int)*n_cols_idx);
  all_ns                  = (unsigned long long*) malloc(sizeof(unsigned long long)*n);
  scale_vals              = (float*) malloc(sizeof(float)*n_cols_idx);

  ref_argop_off_vec_0     = (unsigned long long*) malloc(sizeof(unsigned long long)*ld_in);
  ref_argop_off_vec_1     = (unsigned long long*) malloc(sizeof(unsigned long long)*ld_in);
  ref_argop_off_vec_0_i32 = (unsigned int*) malloc(sizeof(unsigned int)*ld_in);
  ref_argop_off_vec_1_i32 = (unsigned int*) malloc(sizeof(unsigned int)*ld_in);
  argop_off_vec_0         = (unsigned long long*) malloc(sizeof(unsigned long long)*ld_in);
  argop_off_vec_1         = (unsigned long long*) malloc(sizeof(unsigned long long)*ld_in);
  argop_off_vec_0_i32     = (unsigned int*) malloc(sizeof(unsigned int)*ld_in);
  argop_off_vec_1_i32     = (unsigned int*) malloc(sizeof(unsigned int)*ld_in);

  if (use_bf16 == 1) {
    inp_matrix_bf16  = (libxsmm_bfloat16*) malloc(sizeof(libxsmm_bfloat16)*ld_in*n);
    result_bf16      = (libxsmm_bfloat16*) malloc(sizeof(libxsmm_bfloat16)*ld_in);
    inp_matrix_bf162 = (libxsmm_bfloat16*) malloc(sizeof(libxsmm_bfloat16)*ld_in*n);
    scale_vals_bf16  = (libxsmm_bfloat16*) malloc(sizeof(libxsmm_bfloat16)*n_cols_idx);
  } else {
    inp_matrix_bf16  = NULL;
    result_bf16      = NULL;
    inp_matrix_bf162 = NULL;
    scale_vals_bf16  = NULL;
  }

  /* Fill matrices with random data */
  for (i = 0; i < n; i++) {
    all_ns[i] = i;
  }
  shuffle_array(all_ns, n);
  for (i = 0; i < n_cols_idx; i++) {
    if ((op == OP_COPY) && (use_implicit_idx > 0)) {
      cols_ind_array[i] = i;
    } else {
      cols_ind_array[i] = all_ns[i];
    }
  }
  if ((op != OP_COPY) && (use_implicit_idx > 0)) {
    for (i = 0; i < n_cols_idx; i++) {
      cols_ind_array2[i] = i;
    }
  } else {
    shuffle_array(all_ns, n);
    if ((op_order == OPORDER_VECIN_VECIDX) && (use_implicit_idx > 0) ) {
      for (i = 0; i < n_cols_idx; i++) {
        cols_ind_array2[i] = i;
      }
    } else {
      for (i = 0; i < n_cols_idx; i++) {
        cols_ind_array2[i] = all_ns[i];
      }
    }
  }

  for (i = 0; i < n_cols_idx; i++) {
    cols_ind_array_i32[i] = (unsigned int) cols_ind_array[i];
    cols_ind_array2_i32[i] = (unsigned int) cols_ind_array2[i];
  }

  sfill_matrix ( inp_matrix, ld_in, m, n, avoid_small_vals );
  sfill_matrix ( inp_matrix2, ld_in, m, _n, avoid_small_vals );
  sfill_matrix ( ref_result, ld_in, m, 1, avoid_small_vals );
#if 0
  for (jj = 0; jj < ld_in; jj++) {
    ref_result[jj] = -FLT_MAX;
  }
#endif
  memcpy(result, ref_result, ld_in * sizeof(float));
  sfill_matrix ( scale_vals, n_cols_idx, n_cols_idx, 1, avoid_small_vals );

  if (use_bf16 == 1) {
    libxsmm_rne_convert_fp32_bf16( inp_matrix, inp_matrix_bf16, ld_in*n );
    libxsmm_convert_bf16_f32( inp_matrix_bf16, inp_matrix, ld_in*n );
    libxsmm_rne_convert_fp32_bf16( inp_matrix2, inp_matrix_bf162, ld_in*_n );
    libxsmm_convert_bf16_f32( inp_matrix_bf162, inp_matrix2, ld_in*_n );
    libxsmm_rne_convert_fp32_bf16( scale_vals, scale_vals_bf16, n_cols_idx );
    libxsmm_convert_bf16_f32( scale_vals_bf16, scale_vals, n_cols_idx);
    libxsmm_rne_convert_fp32_bf16( result, result_bf16, ld_in );
    libxsmm_convert_bf16_f32( result_bf16, result, ld_in);
    memcpy(ref_result, result, ld_in * sizeof(float));
  }

  for (i = 0; i < m; i++) {
    ref_argop_off_vec_0[i] = 0;
    ref_argop_off_vec_0_i32[i] = 0;
    ref_argop_off_vec_1[i] = 0;
    ref_argop_off_vec_1_i32[i] = 0;
    argop_off_vec_0[i] = 0;
    argop_off_vec_0_i32[i] = 0;
    argop_off_vec_1[i] = 0;
    argop_off_vec_1_i32[i] = 0;
  }

  /* Calculate reference results...  */
  for (jj = 0; jj < n_cols_idx; jj++) {
    float op_res = 0.0f;
    j = (unsigned int)cols_ind_array[jj];
    if (use_regular_vecin == 1) {
      _j = 0;
    } else {
      _j = (unsigned int)cols_ind_array2[jj];
    }
    for (_i = 0; _i < m; _i++) {
      i = _i;
      if (bcast_factor > 1) {
        i = _i/bcast_factor;
      }
      if (redop == REDOP_NONE) {
        vec_in = ref_result;
      } else {
        vec_in = inp_matrix2;
      }

      if (op == OP_COPY) {
        if (op_order == OPORDER_VECIDX_VECIN) {
         op_res = inp_matrix[j * ld_in + i];
        } else {/* if (op_order == OPORDER_VECIDX_VECIN) */
         op_res = vec_in[_j * ld_in + i];
        }
      } else if (op == OP_ADD) {
        op_res = inp_matrix[j * ld_in + i] + vec_in[_j * ld_in + i];
      } else if (op == OP_MUL) {
        op_res = inp_matrix[j * ld_in + i] * vec_in[_j * ld_in + i];
      } else if (op == OP_SUB) {
        if (op_order == OPORDER_VECIN_VECIDX) {
          op_res = vec_in[_j * ld_in + i] - inp_matrix[j * ld_in + i];
        } else {/* if (op_order == OPORDER_VECIDX_VECIN) */
          op_res = inp_matrix[j * ld_in + i] - vec_in[_j * ld_in + i];
        }
      } else if (op == OP_DIV) {
        if (op_order == OPORDER_VECIN_VECIDX) {
          op_res = vec_in[_j * ld_in + i] / inp_matrix[j * ld_in + i];
        } else {/* if (op_order == OPORDER_VECIDX_VECIN) */
          op_res = inp_matrix[j * ld_in + i] / vec_in[_j * ld_in + i];
        }/* else should not happen */
      }/* else should not happen */

      if (scale_op_res == SCALE_OP_RESULT) {
        op_res = op_res * scale_vals[jj];
      }

      if (redop == REDOP_NONE) {
        ref_result[_i] = op_res;
      } else if (redop == REDOP_SUM) {
        ref_result[_i] += op_res;
      } else if (redop == REDOP_MIN) {
        if (argop_vec_0 > 0) {
          if (op_res <= ref_result[_i]) {
            ref_argop_off_vec_0[_i] = j;
          }
        }
        if (argop_vec_1 > 0) {
          if (op_res <= ref_result[_i]) {
            ref_argop_off_vec_1[_i] = _j;
          }
        }
        ref_result[_i] = LIBXSMM_MIN(ref_result[_i], op_res);
      } else if (redop == REDOP_MAX) {
        if (argop_vec_0 > 0) {
          if (op_res >= ref_result[_i]) {
            ref_argop_off_vec_0[_i] = j;
          }
        }
        if (argop_vec_1 > 0) {
          if (op_res >= ref_result[_i]) {
            ref_argop_off_vec_1[_i] = _j;
          }
        }
        ref_result[_i] = LIBXSMM_MAX(ref_result[_i], op_res);
      }
    }
  }

  for (i = 0; i < m; i++) {
    ref_argop_off_vec_0_i32[i] = (unsigned int) ref_argop_off_vec_0[i];
    ref_argop_off_vec_1_i32[i] = (unsigned int) ref_argop_off_vec_1[i];
  }

  /* Call JITed kernel */
  params.n            = n_cols_idx;
  if (idx_mode == 0) {
    params.indices    = cols_ind_array_i32;
    if (use_implicit_idx == 0) {
      params.indices2 = cols_ind_array2_i32;
    }
  } else {
    params.indices    = cols_ind_array;
    if (use_implicit_idx == 0) {
      params.indices2 = cols_ind_array2;
    }
  }

  if (use_bf16 == 0) {
    params.in_matrix    = inp_matrix;
    params.out_vec      = result;
    params.scale_vals   = scale_vals;
    if (use_regular_vecin == 1) {
      params.in_vec     = (redop == REDOP_NONE) ? result : inp_matrix2;
    } else {
      params.in_matrix2 = inp_matrix2;
    }
  } else {
    params.in_matrix    = inp_matrix_bf16;
    params.out_vec      = result_bf16;
    params.scale_vals   = scale_vals_bf16;
    if (use_regular_vecin == 1) {
      params.in_vec     = (redop == REDOP_NONE) ? result_bf16 : inp_matrix_bf162;
    } else {
      params.in_matrix2 = inp_matrix_bf162;
    }
  }

  if ( idx_mode == 0 ) {
    params.argop_off_vec_0 = argop_off_vec_0_i32;
    params.argop_off_vec_1 = argop_off_vec_1_i32;
  } else {
    params.argop_off_vec_0 = argop_off_vec_0;
    params.argop_off_vec_1 = argop_off_vec_1;
  }

  kernel(&params);

  if (idx_mode > 0) {
    for (i = 0; i < m; i++) {
      argop_off_vec_0_i32[i] = (unsigned int) argop_off_vec_0[i];
      argop_off_vec_1_i32[i] = (unsigned int) argop_off_vec_1[i];
    }
  }

  /* compare */
  printf("#   Correctness  #\n");
  printf("OP=%s, OPORDER=%s, SCALE_OP_RES=%s, REDOP=%s\n", opname, opordername, scaleopresname, redopname);
  printf("IDX_VEC_IN=%i, IMPLICIT_IDX_VEC_IN=%u, ARGOP_MODE=%u, IDX_SIZE=%i\n", (use_regular_vecin > 0) ? 0 : 1,
    use_implicit_idx, argop_mode, (idx_mode == 0) ? 4 : 8);
  printf("##########################################\n");
  if (use_bf16 == 1) {
    libxsmm_convert_bf16_f32( result_bf16, result, ld_in);
  }
  libxsmm_matdiff(&norms_elts, LIBXSMM_DATATYPE_F32, m, 1, ref_result, result, 0, 0);
  printf("L1 reference  : %.25g\n", norms_elts.l1_ref);
  printf("L1 test       : %.25g\n", norms_elts.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_elts.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_elts.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_elts.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_elts.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_elts.normf_rel);
  libxsmm_matdiff_reduce(&diff, &norms_elts);

  if (argop_vec_0 > 0) {
    printf("###### Arg idx 0 #######\n");
    libxsmm_matdiff(&norms_elts, LIBXSMM_DATATYPE_I32, m, 1, ref_argop_off_vec_0_i32, argop_off_vec_0_i32, 0, 0);

    printf("L1 reference  : %.25g\n", norms_elts.l1_ref);
    printf("L1 test       : %.25g\n", norms_elts.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_elts.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_elts.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_elts.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_elts.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_elts.normf_rel);
    libxsmm_matdiff_reduce(&diff, &norms_elts);
  }

  if (argop_vec_1 > 0) {
    printf("###### Arg idx 1 #######\n");
    libxsmm_matdiff(&norms_elts, LIBXSMM_DATATYPE_I32, m, 1, ref_argop_off_vec_1_i32, argop_off_vec_1_i32, 0, 0);

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
  /* Calculate reference results...  */
  for (k = 0; k < iters; k++) {
    for (jj = 0; jj < n_cols_idx; jj++) {
      float op_res = 0.0f;
      j = (unsigned int)cols_ind_array[jj];
      if (use_regular_vecin == 1) {
        _j = 0;
      } else {
        _j = (unsigned int)cols_ind_array2[jj];
      }
      for (_i = 0; _i < m; _i++) {
        i = _i;
        if (bcast_factor > 0) {
          i = _i/bcast_factor;
        }
        if (redop == REDOP_NONE) {
          vec_in = ref_result;
        } else {
          vec_in = inp_matrix2;
        }

        if (op != OP_COPY) {
          if (op == OP_ADD) {
            op_res = inp_matrix[j * ld_in + i] + vec_in[_j * ld_in + i];
          }
          if (op == OP_MUL) {
            op_res = inp_matrix[j * ld_in + i] * vec_in[_j * ld_in + i];
          }
          if (op == OP_SUB) {
            if (op_order == OPORDER_VECIN_VECIDX) {
              op_res = vec_in[_j * ld_in + i] - inp_matrix[j * ld_in + i];
            }
            if (op_order == OPORDER_VECIDX_VECIN) {
              op_res = inp_matrix[j * ld_in + i] - vec_in[_j * ld_in + i];
            }
          }
          if (op == OP_DIV) {
            if (op_order == OPORDER_VECIN_VECIDX) {
              op_res = vec_in[_j * ld_in + i] / inp_matrix[j * ld_in + i];
            }
            if (op_order == OPORDER_VECIDX_VECIN) {
              op_res = inp_matrix[j * ld_in + i] / vec_in[_j * ld_in + i];
            }
          }
        } else {
          if (op_order == OPORDER_VECIDX_VECIN) {
           op_res = inp_matrix[j * ld_in + i];
          } else {
           op_res = vec_in[_j * ld_in + i];
          }
        }

        if (scale_op_res == SCALE_OP_RESULT) {
          op_res = op_res * scale_vals[jj];
        }
        if (redop != REDOP_NONE) {
          if (redop == REDOP_SUM) {
            ref_result[_i] += op_res;
          }
          if (redop == REDOP_MIN) {
            if (argop_vec_0 > 0) {
              if (op_res < ref_result[_i]) {
                ref_argop_off_vec_0[_i] = j;
              }
            }
            if (argop_vec_1 > 0) {
              if (op_res < ref_result[_i]) {
                ref_argop_off_vec_1[_i] = _j;
              }
            }
            ref_result[_i] = LIBXSMM_MIN(ref_result[_i], op_res);
          }
          if (redop == REDOP_MAX) {
            if (argop_vec_0 > 0) {
              if (op_res > ref_result[_i]) {
                ref_argop_off_vec_0[_i] = j;
              }
            }
            if (argop_vec_1 > 0) {
              if (op_res > ref_result[_i]) {
                ref_argop_off_vec_1[_i] = _j;
              }
            }
            ref_result[_i] = LIBXSMM_MAX(ref_result[_i], op_res);
          }
        } else {
          ref_result[_i] = op_res;
        }
      }
    }
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Reference time = %.5g\n", ((double)(l_total)));

  l_start = libxsmm_timer_tick();
  for (k = 0; k < iters; k++) {
    kernel( &params );
  }
  l_end = libxsmm_timer_tick();
  l_total2 = libxsmm_timer_duration(l_start, l_end);
  printf("Optimized time = %.5g\n", ((double)(l_total2)));
  printf("Speedup is = %.5g\n", ((double)(l_total/l_total2)));

  free(inp_matrix);
  free(result);
  free(ref_result);
  free(inp_matrix2);
  free(cols_ind_array);
  free(cols_ind_array2);
  free(cols_ind_array_i32);
  free(cols_ind_array2_i32);
  free(all_ns);
  free(scale_vals);
  free(ref_argop_off_vec_0);
  free(ref_argop_off_vec_0_i32);
  free(ref_argop_off_vec_1);
  free(ref_argop_off_vec_1_i32);
  free(argop_off_vec_0);
  free(argop_off_vec_0_i32);
  free(argop_off_vec_1);
  free(argop_off_vec_1_i32);
  if (use_bf16 == 1) {
    free(inp_matrix_bf16);
    free(result_bf16);
    free(inp_matrix_bf162);
    free(scale_vals_bf16);
  }

  {
    const char *const env_check_scale = getenv("CHECK_SCALE");
    const double check_scale = LIBXSMM_ABS(0 == env_check_scale ? 1.0 : atof(env_check_scale));
    if (LIBXSMM_NEQ(0, check) && (check < 100.0 * check_scale * diff.normf_rel)) {
      fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
      exit(EXIT_FAILURE);
    }
  }

  return EXIT_SUCCESS;
}
