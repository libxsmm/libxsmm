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
  unsigned int m = 64, n = 64, i, j, jj, k, iters = 10000, n_cols_idx = 32, op = 0, op_order = 0, scale_op_res = 0, redop = 0, use_implicit_idx = 0, _j = 0;
  libxsmm_blasint ld_in = 64;
  float  *inp_matrix, *result, *ref_result, *inp_matrix2, *scale_vals;
  libxsmm_bfloat16 *inp_matrix_bf16, *result_bf16, *inp_matrix_bf162, *scale_vals_bf16;
  unsigned long long *cols_ind_array, *cols_ind_array2, *all_ns;
  libxsmm_meltw_opreduce_vecs_idx_param     params;
  libxsmm_meltw_opreduce_vecs_flags         opredop_flags;
  libxsmm_meltwfunction_opreduce_vecs_idx   kernel;
  libxsmm_matdiff_info                      norms_elts, diff;
  unsigned long long l_start, l_end;
  double l_total = 0.0, l_total2 = 0.0;
  char opname[50];
  char opordername[50];
  char scaleopresname[50];
  char redopname[50];
  unsigned int use_bf16 = 0;

  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 1 : atof(env_check));

  libxsmm_init();
  libxsmm_matdiff_clear(&norms_elts);
  libxsmm_matdiff_clear(&diff);

  if ( argc > 1 ) m           = atoi(argv[1]);
  if ( argc > 2 ) n           = atoi(argv[2]);
  if ( argc > 3 ) n_cols_idx  = atoi(argv[3]);
  if ( argc > 4 ) ld_in       = atoi(argv[4]);
  if ( argc > 5 ) op          = atoi(argv[5]);
  if ( argc > 6 ) op_order    = atoi(argv[6]);
  if ( argc > 7 ) scale_op_res= atoi(argv[7]);
  if ( argc > 8 ) redop       = atoi(argv[8]);
  if ( argc > 9 ) use_implicit_idx = atoi(argv[9]);
  if ( argc > 10 ) iters       = atoi(argv[10]);
  if ( argc > 11 ) use_bf16    = atoi(argv[11]);

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
    opredop_flags = opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIN_VECIDX;
    sprintf(opordername, "VECIN_VECIDX");
  } else if (op_order == OPORDER_VECIDX_VECIN) {
    opredop_flags = opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIDX_VECIN;
    sprintf(opordername, "VECIDX_VECIN");
  } else {
    printf("ERROR: Invalid OP_ORDER requested!!!\n");
    return EXIT_SUCCESS;
  }

  if (scale_op_res == NO_SCALE_OP_RESULT) {
    sprintf(scaleopresname, "NO");
  } else if (scale_op_res == SCALE_OP_RESULT) {
    opredop_flags = opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_SCALE_OP_RESULT;
    sprintf(scaleopresname, "YES");
  } else {
    printf("ERROR: Scale OP result should be 0 or 1!!!\n");
    return EXIT_SUCCESS;
  }

  if (redop == REDOP_NONE) {
    opredop_flags = opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_NONE;
    sprintf(redopname, "NONE");
  } else if (redop == REDOP_SUM) {
    opredop_flags = opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_SUM;
    sprintf(redopname, "SUM");
  } else if (redop == REDOP_MAX) {
    opredop_flags = opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MAX;
    sprintf(redopname, "MAX");
  } else if (redop == REDOP_MIN) {
    opredop_flags = opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MIN;
    sprintf(redopname, "MIN");
  } else {
    printf("ERROR: Invalid REDOP requested!!!\n");
    return EXIT_SUCCESS;
  }

  if (op != OP_COPY) {
    opredop_flags = opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_INDEXED_VEC;
    if (use_implicit_idx > 0) {
      opredop_flags = opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_IMPLICIT_INDEXED_VEC;
    }
  }

  if ((op == OP_COPY) && (use_implicit_idx > 0)) {
    opredop_flags = opredop_flags | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_IMPLICIT_INDEXED_VECIDX;
  }

  if (use_bf16 == 0) {
    kernel = libxsmm_dispatch_meltw_opreduce_vecs_idx(m, &ld_in, &ld_in, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_I64, opredop_flags);
  } else {
    kernel = libxsmm_dispatch_meltw_opreduce_vecs_idx(m, &ld_in, &ld_in, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_I64, opredop_flags);
  }
  m = LIBXSMM_MAX(m,1);
  n = LIBXSMM_MAX(n,1);
  ld_in = LIBXSMM_MAX(ld_in,(libxsmm_blasint)m);

  /* Allocate arrays  */
  inp_matrix              = (float*) malloc(ld_in*n*sizeof(float) );
  result                  = (float*) malloc(ld_in*sizeof(float) );
  ref_result              = (float*) malloc(ld_in*sizeof(float) );
  inp_matrix2             = (float*) malloc(ld_in*n*sizeof(float) );
  cols_ind_array          = (unsigned long long*) malloc(n_cols_idx*sizeof(unsigned long long));
  cols_ind_array2         = (unsigned long long*) malloc(n_cols_idx*sizeof(unsigned long long));
  all_ns                  = (unsigned long long*) malloc(n*sizeof(unsigned long long));
  scale_vals              = (float*) malloc(n_cols_idx*sizeof(float));

  if (use_bf16 == 1) {
    inp_matrix_bf16              = (libxsmm_bfloat16*) malloc(ld_in*n*sizeof(libxsmm_bfloat16) );
    result_bf16                  = (libxsmm_bfloat16*) malloc(ld_in*sizeof(libxsmm_bfloat16) );
    inp_matrix_bf162              = (libxsmm_bfloat16*) malloc(ld_in*n*sizeof(libxsmm_bfloat16) );
    scale_vals_bf16              = (libxsmm_bfloat16*) malloc(n_cols_idx*sizeof(libxsmm_bfloat16));
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
  sfill_matrix ( inp_matrix, ld_in, m, n );
  sfill_matrix ( inp_matrix2, ld_in, m, n );
  sfill_matrix ( ref_result, ld_in, m, 1 );
  memcpy(result, ref_result, ld_in * sizeof(float));
  sfill_matrix ( scale_vals, n_cols_idx, n_cols_idx, 1 );

  if (use_bf16 == 1) {
    libxsmm_rne_convert_fp32_bf16( inp_matrix, inp_matrix_bf16, ld_in*n );
    libxsmm_rne_convert_fp32_bf16( inp_matrix2, inp_matrix_bf162, ld_in*n );
    libxsmm_rne_convert_fp32_bf16( scale_vals, scale_vals_bf16, n_cols_idx );
    libxsmm_rne_convert_fp32_bf16( result, result_bf16, ld_in );
  }

  /* Calculate reference results...  */
  for (jj = 0; jj < n_cols_idx; jj++) {
    float op_res;
    j = cols_ind_array[jj];
    _j = cols_ind_array2[jj];
    for (i = 0; i < m; i++) {
      if (op != OP_COPY) {
        if (op == OP_ADD) {
          op_res = inp_matrix[j * ld_in + i] + inp_matrix2[_j * ld_in + i];
        }
        if (op == OP_MUL) {
          op_res = inp_matrix[j * ld_in + i] * inp_matrix2[_j * ld_in + i];
        }
        if (op == OP_SUB) {
          if (op_order == OPORDER_VECIN_VECIDX) {
            op_res = inp_matrix2[_j * ld_in + i] - inp_matrix[j * ld_in + i];
          }
          if (op_order == OPORDER_VECIDX_VECIN) {
            op_res = inp_matrix[j * ld_in + i] - inp_matrix2[_j * ld_in + i];
          }
        }
        if (op == OP_DIV) {
          if (op_order == OPORDER_VECIN_VECIDX) {
            op_res = inp_matrix2[_j * ld_in + i] / inp_matrix[j * ld_in + i];
          }
          if (op_order == OPORDER_VECIDX_VECIN) {
            op_res = inp_matrix[j * ld_in + i] / inp_matrix2[_j * ld_in + i];
          }
        }
      } else {
        if (op_order == OPORDER_VECIDX_VECIN) {
         op_res = inp_matrix[j * ld_in + i];
        } else {
         op_res = inp_matrix2[_j * ld_in + i];
        }
      }

      if (scale_op_res == SCALE_OP_RESULT) {
        op_res = op_res * scale_vals[jj];
      }
      if (redop != REDOP_NONE) {
        if (redop == REDOP_SUM) {
          ref_result[i] += op_res;
        }
        if (redop == REDOP_MIN) {
          ref_result[i] = LIBXSMM_MIN(ref_result[i], op_res);
        }
        if (redop == REDOP_MAX) {
          ref_result[i] = LIBXSMM_MAX(ref_result[i], op_res);
        }
      } else {
        ref_result[i] = op_res;
      }
    }
  }

  /* Call JITed kernel */
  params.n            = n_cols_idx;
  params.indices      = cols_ind_array;
  if (use_implicit_idx == 0) {
    params.indices2      = cols_ind_array2;
  }

  if (use_bf16 == 0) {
    params.in_matrix    = inp_matrix;
    params.out_vec      = result;
    params.scale_vals   = scale_vals;
    params.in_matrix2    = inp_matrix2;
  } else {
    params.in_matrix    = inp_matrix_bf16;
    params.out_vec      = result_bf16;
    params.scale_vals   = scale_vals_bf16;
    params.in_matrix2    = inp_matrix_bf162;
  }
  kernel(&params);

  /* compare */
  printf("#   Correctness  #\n");
  printf("OP=%s, OPORDER=%s, SCALE_OP_RES=%s, REDOP=%s\n", opname, opordername, scaleopresname, redopname);
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

  l_start = libxsmm_timer_tick();
  /* Calculate reference results...  */
  for (k = 0; k < iters; k++) {
    for (jj = 0; jj < n_cols_idx; jj++) {
      float op_res;
      j = cols_ind_array[jj];
      _j = cols_ind_array2[jj];
      for (i = 0; i < m; i++) {
        if (op != OP_COPY) {
          if (op == OP_ADD) {
            op_res = inp_matrix[j * ld_in + i] + inp_matrix2[_j * ld_in + i];
          }
          if (op == OP_MUL) {
            op_res = inp_matrix[j * ld_in + i] * inp_matrix2[_j * ld_in + i];
          }
          if (op == OP_SUB) {
            if (op_order == OPORDER_VECIN_VECIDX) {
              op_res = inp_matrix2[_j * ld_in + i] - inp_matrix[j * ld_in + i];
            }
            if (op_order == OPORDER_VECIDX_VECIN) {
              op_res = inp_matrix[j * ld_in + i] - inp_matrix2[_j * ld_in + i];
            }
          }
          if (op == OP_DIV) {
            if (op_order == OPORDER_VECIN_VECIDX) {
              op_res = inp_matrix2[_j * ld_in + i] / inp_matrix[j * ld_in + i];
            }
            if (op_order == OPORDER_VECIDX_VECIN) {
              op_res = inp_matrix[j * ld_in + i] / inp_matrix2[_j * ld_in + i];
            }
          }
        } else {
          if (op_order == OPORDER_VECIDX_VECIN) {
           op_res = inp_matrix[j * ld_in + i];
          } else {
           op_res = inp_matrix2[_j * ld_in + i];
          }
        }

        if (scale_op_res == SCALE_OP_RESULT) {
          op_res = op_res * scale_vals[jj];
        }
        if (redop != REDOP_NONE) {
          if (redop == REDOP_SUM) {
            ref_result[i] += op_res;
          }
          if (redop == REDOP_MIN) {
            ref_result[i] = LIBXSMM_MIN(ref_result[i], op_res);
          }
          if (redop == REDOP_MAX) {
            ref_result[i] = LIBXSMM_MAX(ref_result[i], op_res);
          }
        } else {
          ref_result[i] = op_res;
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
  free(all_ns);
  free(scale_vals);
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

