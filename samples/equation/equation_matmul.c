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
#if defined(_OPENMP)
# include <omp.h>
#endif

float fsigmoid(float x) {
  return (LIBXSMM_TANHF(x/2.0f) + 1.0f)/2.0f;
}

float upconvert_bf16(libxsmm_bfloat16 x) {
  union libxsmm_bfloat16_hp bf16_hp;
  bf16_hp.i[1] = x;
  bf16_hp.i[0] = 0;
  return bf16_hp.f;
}

float gelu(float x) {
  return (LIBXSMM_ERFF(x/LIBXSMM_SQRTF(2.0f)) + 1.0f)*0.5f*x;
}

void gemm_fp32(float *A, float *B, float *C, float beta,
              libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
              libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
              libxsmm_blasint brgemm_count, libxsmm_blasint str_a, libxsmm_blasint str_b) {

  libxsmm_blasint i, j, l, br, br_count;

  if (brgemm_count <= 0) {
    br_count = 1;
  } else {
    br_count = brgemm_count;
  }

  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      if (beta == 0) {
        C[i + j * ldc] = 0.0;
      }
      for (br = 0; br < br_count; br++) {
        for (l = 0; l < k; l++) {
          C[i + j * ldc] += A[i + l * lda + br * str_a] * B[l + j * ldb + br * str_b];
        }
      }
    }
  }
}

void gemm_bf16(libxsmm_bfloat16 *A, libxsmm_bfloat16 *B, libxsmm_bfloat16 *C, libxsmm_bfloat16 beta,
              libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
              libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
              libxsmm_blasint brgemm_count, libxsmm_blasint str_a, libxsmm_blasint str_b) {

  libxsmm_blasint i, j, l, br, br_count;

  if (brgemm_count <= 0) {
    br_count = 1;
  } else {
    br_count = brgemm_count;
  }

  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      float acc = 0.0;
      if (beta == 0) {
        C[i + j * ldc] = 0;
      }
      acc = upconvert_bf16(C[i + j * ldc]);
      for (br = 0; br < br_count; br++) {
        for (l = 0; l < k; l++) {
          acc +=  upconvert_bf16(A[i + l * lda + br * str_a]) * upconvert_bf16(B[l + j * ldb + br * str_b]);
        }
      }
      libxsmm_rne_convert_fp32_bf16( &acc, &C[i + j * ldc], 1 );
    }
  }
}

void eqn0_f32(float *Out, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld,
              float *A, libxsmm_blasint m_A, libxsmm_blasint n_A, libxsmm_blasint lda,
              float *B, libxsmm_blasint m_B, libxsmm_blasint n_B, libxsmm_blasint ldb,
              float *C, libxsmm_blasint m_C, libxsmm_blasint n_C, libxsmm_blasint ldc,
              float *D, libxsmm_blasint m_D, libxsmm_blasint n_D, libxsmm_blasint ldd ) {
 /*
  *
  * Result = gelu(A+B) * tanh(C x D)
  *
  */

  libxsmm_blasint i, j;
  float tmp[m_C * n_D];

  gemm_fp32(C, D, tmp, 0,
            m_C, n_D, n_C,
            ldc, ldd, m_C,
            0, 0, 0);
  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      Out[i + j * ld] = gelu(A[i + j * lda] + B[i + j *ldb]) * LIBXSMM_TANHF(tmp[i + j * m_C]);
    }
  }
}

void eqn1_f32(float *Out, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld,
              float *A, libxsmm_blasint m_A, libxsmm_blasint n_A, libxsmm_blasint lda,
              float *B, libxsmm_blasint m_B, libxsmm_blasint n_B, libxsmm_blasint ldb, libxsmm_blasint brgemm_count,
              float *C, libxsmm_blasint m_C, libxsmm_blasint n_C, libxsmm_blasint ldc, libxsmm_blasint stride_a,
              float *D, libxsmm_blasint m_D, libxsmm_blasint n_D, libxsmm_blasint ldd, libxsmm_blasint stride_b ) {
 /*
  *
  * Result = gelu(A) * tanh( B + Sum Ci x Di )
  *
  */

  libxsmm_blasint i, j;

  gemm_fp32(C, D, B, 1.0,
            m_C, n_D, n_C,
            ldc, ldd, ldb,
            brgemm_count, stride_a, stride_b);

  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      Out[i + j * ld] = gelu(A[i + j * lda]) * LIBXSMM_TANHF(B[i + j * ldb]);
    }
  }
}

void eqn2_f32(float *Out, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld,
              float *A, libxsmm_blasint m_A, libxsmm_blasint n_A, libxsmm_blasint lda,
              float *B, libxsmm_blasint m_B, libxsmm_blasint n_B, libxsmm_blasint ldb, libxsmm_blasint brgemm_count,
              float *C, libxsmm_blasint m_C, libxsmm_blasint n_C, libxsmm_blasint ldc, libxsmm_blasint stride_a,
              float *D, libxsmm_blasint m_D, libxsmm_blasint n_D, libxsmm_blasint ldd, libxsmm_blasint stride_b ) {
 /*
  *
  * Result = gelu(A) * tanh( Sum Ci x Di )
  *
  */

  libxsmm_blasint i, j;
  float tmp[m_C * n_D];

  gemm_fp32(C, D, tmp, 0,
            m_C, n_D, n_C,
            ldc, ldd, m_C,
            brgemm_count, stride_a, stride_b);

  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      Out[i + j * ld] = gelu(A[i + j * lda]) + LIBXSMM_TANHF(tmp[i + j * m_C]);
    }
  }
}

void eqn3_f32(float *Out, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld,
              float *A, libxsmm_blasint m_A, libxsmm_blasint n_A, libxsmm_blasint lda,
              float *B, libxsmm_blasint m_B, libxsmm_blasint n_B, libxsmm_blasint ldb, libxsmm_blasint brgemm_count,
              float *C, libxsmm_blasint m_C, libxsmm_blasint n_C, libxsmm_blasint ldc, libxsmm_blasint stride_a,
              float *D, libxsmm_blasint m_D, libxsmm_blasint n_D, libxsmm_blasint ldd, libxsmm_blasint stride_b ) {

  /* Result = Sum(Ci x Di) + gelu(A) */

  libxsmm_blasint i, j;
  float tmp[m_C * n_D];

  gemm_fp32(C, D, tmp, 0,
            m_C, n_D, n_C,
            ldc, ldd, m_C,
            brgemm_count, stride_a, stride_b);

  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      Out[i + j * ld] = gelu(A[i + j * lda]) + tmp[i + j * m_C];
    }
  }
}

void eqn4_f32(float *Out, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld,
              float *A, libxsmm_blasint m_A, libxsmm_blasint n_A, libxsmm_blasint lda,
              float *B, libxsmm_blasint m_B, libxsmm_blasint n_B, libxsmm_blasint ldb, libxsmm_blasint brgemm_count,
              float *C, libxsmm_blasint m_C, libxsmm_blasint n_C, libxsmm_blasint ldc, libxsmm_blasint stride_a,
              float *D, libxsmm_blasint m_D, libxsmm_blasint n_D, libxsmm_blasint ldd, libxsmm_blasint stride_b,
              float *colbias, int relu_sigmoid_fusion_mode ) {

  /* Result =  sigmoid( Sum(Ci x Di) + colbias + 1.0) */

  libxsmm_blasint i, j;
  float tmp[m_C * n_D];

  gemm_fp32(C, D, tmp, 0,
            m_C, n_D, n_C,
            ldc, ldd, m_C,
            brgemm_count, stride_a, stride_b);

  if (relu_sigmoid_fusion_mode == 0) {
    for (j = 0; j < n; j++) {
      for (i = 0; i < m; i++) {
        Out[i + j * ld] = tmp[j *m_C + i] + colbias[i] + 1.0;
      }
    }
  } else if (relu_sigmoid_fusion_mode == 1) {
    for (j = 0; j < n; j++) {
      for (i = 0; i < m; i++) {
        Out[i + j * ld] = LIBXSMM_MAX(0.0, tmp[j *m_C + i] + colbias[i] + 1.0);
      }
    }
  } else if (relu_sigmoid_fusion_mode == 2) {
    for (j = 0; j < n; j++) {
      for (i = 0; i < m; i++) {
        Out[i + j * ld] = fsigmoid(tmp[j *m_C + i] + colbias[i] + 1.0);
      }
    }
  }
}

int main( int argc, char* argv[] ) {
  libxsmm_blasint my_eqn0, my_eqn1, my_eqn2, my_eqn3, my_eqn4, my_eqn5;
  libxsmm_matrix_eqn_function func0, func1, func2, func3, func4, func5;
  libxsmm_blasint i, j, k, l, it;
  libxsmm_matrix_eqn_param eqn_param;
  unsigned long long l_start, l_end;
  double l_total = 0, l_total2 = 0;
  libxsmm_matdiff_info norms_out;
  int iters = 100;
  int datatype_mode = 0;
  libxsmm_datatype  in_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype  out_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_blasint m_i[128], n_i[128], ld_i[128], blocks_i[128];
  libxsmm_meqn_arg_shape  arg_shape[128];
  libxsmm_matrix_arg_attributes arg_set_attr0, arg_set_attr1;
  libxsmm_matrix_arg_attributes arg_singular_attr;
  float *arg[128];
  libxsmm_matrix_arg arg_array[128];
  libxsmm_bfloat16 *bf16_arg[128];
  libxsmm_matrix_arg bf16_arg_array[128];
  libxsmm_matrix_eqn_arg_metadata arg_metadata[128];
  libxsmm_matrix_eqn_op_metadata  op_metadata[128];
  libxsmm_blasint n_tensors, ref_id;
  libxsmm_matrix_op_arg op_arg_arr[9];
  int relu_sigmoid_fusion_mode = 0;
  unsigned long long  brcount;
  i = 1;
  n_tensors = atoi(argv[i++]);
  ref_id = n_tensors;
  for (j = 0; j < n_tensors; j++) {
    m_i[j] = atoi(argv[i++]);
    n_i[j] = atoi(argv[i++]);
    ld_i[j] = atoi(argv[i++]);
    blocks_i[j] = atoi(argv[i++]);
  }
  datatype_mode = atoi(argv[i++]);
  relu_sigmoid_fusion_mode = atoi(argv[i++]);
  iters = atoi(argv[i]);

#if defined(__SSE3__)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

  if (datatype_mode == 0) {
    in_dt = LIBXSMM_DATATYPE_F32;
    out_dt = LIBXSMM_DATATYPE_F32;
  } else if (datatype_mode == 1) {
    in_dt = LIBXSMM_DATATYPE_BF16;
    out_dt = LIBXSMM_DATATYPE_BF16;
  } else if (datatype_mode == 2) {
    in_dt = LIBXSMM_DATATYPE_F32;
    out_dt = LIBXSMM_DATATYPE_BF16;
  } else if (datatype_mode == 3) {
    in_dt = LIBXSMM_DATATYPE_BF16;
    out_dt = LIBXSMM_DATATYPE_F32;
  }

  for (j = 0; j < n_tensors; j++) {
    arg_shape[j].m = m_i[j];
    arg_shape[j].n = n_i[j];
    arg_shape[j].ld = ld_i[j];
    if (j == n_tensors-1) {
      arg_shape[j].type = out_dt;
    } else {
      arg_shape[j].type = in_dt;
    }
  }

  for (i = 0; i < n_tensors; i++) {
    arg[i] = (float*) libxsmm_aligned_malloc( sizeof(float)*n_i[i]*ld_i[i]*blocks_i[i],   64);
  }
  arg[ref_id] = (float*) libxsmm_aligned_malloc( sizeof(float)*n_i[n_tensors-1]*ld_i[n_tensors-1]*blocks_i[n_tensors-1],   64);

  for (i = 0; i < n_tensors; i++) {
    bf16_arg[i] = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*n_i[i]*ld_i[i]*blocks_i[i],   64);
  }
  bf16_arg[ref_id] = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*n_i[n_tensors-1]*ld_i[n_tensors-1]*blocks_i[n_tensors-1],   64);

  libxsmm_init();
  libxsmm_matdiff_clear(&norms_out);

  for (k = 0; k < n_tensors; k++) {
    float *cur_arr = arg[k];
    libxsmm_bfloat16 *bf16_cur_arr = bf16_arg[k];
    libxsmm_blasint block_size = ld_i[k]*n_i[k];
    for ( i = 0; i < n_i[k]; i++ ) {
      for ( j = 0; j < ld_i[k]; j++ ) {
        for ( l = 0; l < blocks_i[k]; l++) {
          float val = (float)libxsmm_rng_f64();
          cur_arr[j + i *ld_i[k] + l * block_size] = val;
          libxsmm_rne_convert_fp32_bf16( &cur_arr[j + i *ld_i[k] + l * block_size], &bf16_cur_arr[j + i *ld_i[k] + l * block_size], 1 );
          if ( datatype_mode == 1) {
            libxsmm_convert_bf16_f32(&bf16_cur_arr[j + i *ld_i[k] + l * block_size],&cur_arr[j + i *ld_i[k] + l * block_size], 1 );
          }
        }
      }
    }
  }

  float *ref_arr = arg[ref_id];
  float *out_arr = arg[n_tensors-1];
  libxsmm_bfloat16 *bf16_ref_arr = bf16_arg[ref_id];
  libxsmm_bfloat16 *bf16_out_arr = bf16_arg[n_tensors-1];
  libxsmm_blasint block_size = ld_i[n_tensors-1]*n_i[n_tensors-1];
  for ( i = 0; i < n_i[n_tensors-1]; i++ ) {
    for ( j = 0; j < ld_i[n_tensors-1]; j++ ) {
      for ( l = 0; l < blocks_i[n_tensors-1]; l++) {
        ref_arr[j + i * ld_i[n_tensors-1] + l * block_size] = out_arr[j + i * ld_i[n_tensors-1] + l * block_size];
        bf16_ref_arr[j + i * ld_i[n_tensors-1] + l * block_size] = bf16_out_arr[j + i * ld_i[n_tensors-1] + l * block_size];
      }
    }
  }

  for (k = 0; k < n_tensors-1; k++) {
    arg_array[k].primary = arg[k];
    bf16_arg_array[k].primary = bf16_arg[k];
  }

  /* Result = gelu(A+B) * tanh(C x D)  */
  arg_singular_attr.type = LIBXSMM_MATRIX_ARG_TYPE_SINGULAR;

  my_eqn0 = libxsmm_matrix_eqn_create();
  arg_metadata[0].eqn_idx     = my_eqn0;
  arg_metadata[0].in_arg_pos  = 0;
  arg_metadata[1].eqn_idx     = my_eqn0;
  arg_metadata[1].in_arg_pos  = 1;
  arg_metadata[2].eqn_idx     = my_eqn0;
  arg_metadata[2].in_arg_pos  = 2;
  arg_metadata[3].eqn_idx     = my_eqn0;
  arg_metadata[3].in_arg_pos  = 3;
  op_metadata[0].eqn_idx      = my_eqn0;
  op_metadata[0].op_arg_pos   = -1;

  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE);
  libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_UNARY_GELU, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE);
  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE);
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[0], arg_shape[0], arg_singular_attr);
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[1], arg_shape[1], arg_singular_attr);
  libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_UNARY_TANH, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE);
  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_BINARY_MATMUL, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE);
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[2], arg_shape[2], arg_singular_attr);
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[3], arg_shape[3], arg_singular_attr);
  libxsmm_matrix_eqn_tree_print( my_eqn0 );
  func0 = libxsmm_dispatch_matrix_eqn_v2( my_eqn0, arg_shape[n_tensors-1] );

  if ( in_dt == LIBXSMM_DATATYPE_F32 ) {
    eqn_param.inputs = arg_array;
  } else {
    eqn_param.inputs = bf16_arg_array;
  }
  if ( out_dt == LIBXSMM_DATATYPE_F32 ) {
    eqn_param.output.primary = arg[n_tensors-1];
  } else {
   eqn_param.output.primary  = bf16_arg[n_tensors-1];
  }

  func0(&eqn_param);

  if (datatype_mode == 0 || datatype_mode == 1) {
    eqn0_f32( arg[ref_id], m_i[n_tensors-1], n_i[n_tensors-1], ld_i[n_tensors-1],
              arg[0], m_i[0], n_i[0], ld_i[0],
              arg[1], m_i[1], n_i[1], ld_i[1],
              arg[2], m_i[2], n_i[2], ld_i[2],
              arg[3], m_i[3], n_i[3], ld_i[3] );
  } else if (datatype_mode == 2) {
  } else if (datatype_mode == 3) {
  }

  if (datatype_mode == 0) {
    printf("Equation IN: F32, OUT: F32 \n");
  } else if (datatype_mode == 1) {
    printf("Equation IN: BF16, OUT: BF16 \n");
  } else if (datatype_mode == 2) {
    printf("Equation IN: F32, OUT: BF16 \n");
  } else if (datatype_mode == 3) {
    printf("Equation IN: BF16, OUT: F32 \n");
  }

  printf("\n\nNow testing equation 0...\n\n");
  if (datatype_mode == 1) {
    libxsmm_convert_bf16_f32( bf16_arg[n_tensors-1], arg[n_tensors-1], ld_i[n_tensors-1] * n_i[n_tensors-1] * blocks_i[n_tensors-1] );
  }
  /* compare */
  printf("##########################################\n");
  printf("#   Correctness  - Output                #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ld_i[n_tensors-1] * n_i[n_tensors-1] * blocks_i[n_tensors-1], 1, arg[ref_id], arg[n_tensors-1], 0, 0);

  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  /* Now benchmarking the equations */
  if (datatype_mode == 0 || datatype_mode == 1) {
    eqn0_f32( arg[ref_id], m_i[n_tensors-1], n_i[n_tensors-1], ld_i[n_tensors-1],
              arg[0], m_i[0], n_i[0], ld_i[0],
              arg[1], m_i[1], n_i[1], ld_i[1],
              arg[2], m_i[2], n_i[2], ld_i[2],
              arg[3], m_i[3], n_i[3], ld_i[3] );
  } else if (datatype_mode == 2) {
  } else if (datatype_mode == 3) {
  }
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
    if (datatype_mode == 0 || datatype_mode == 1) {
      eqn0_f32( arg[ref_id], m_i[n_tensors-1], n_i[n_tensors-1], ld_i[n_tensors-1],
                arg[0], m_i[0], n_i[0], ld_i[0],
                arg[1], m_i[1], n_i[1], ld_i[1],
                arg[2], m_i[2], n_i[2], ld_i[2],
                arg[3], m_i[3], n_i[3], ld_i[3] );
    } else if (datatype_mode == 2) {
    } else if (datatype_mode == 3) {
    }
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Compiler equation time  = %.5g\n", ((double)(l_total)));

  func0(&eqn_param);
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
    func0(&eqn_param);
  }
  l_end = libxsmm_timer_tick();
  l_total2 = libxsmm_timer_duration(l_start, l_end);
  printf("JITed TPP equation time = %.5g\n", ((double)(l_total2)));
  printf("Speedup (%d iters) is %.5g\n", iters, l_total/l_total2);

  printf("\n\nNow testing equation 1...\n\n");

  /* Create copy of B since this is now Inout  */
  float *copy_B = (float*) libxsmm_aligned_malloc( sizeof(float)*n_i[1]*ld_i[1]*blocks_i[1],   64);
  float *orig_B = arg[1];
  for ( i = 0; i < n_i[1]; i++ ) {
    for ( j = 0; j < ld_i[1]; j++ ) {
      for ( l = 0; l < blocks_i[1]; l++) {
        copy_B[j + i * ld_i[1] + l * 0] = orig_B[j + i * ld_i[1] + l * 0];
      }
    }
  }
  arg_array[1].primary = copy_B;
  bf16_arg_array[1].primary = copy_B;
  arg_shape[1].type = LIBXSMM_DATATYPE_F32;
  /* Result = gelu(A) * tanh( B + Sum Ci x Di ) */

  arg_set_attr0.type = LIBXSMM_MATRIX_ARG_TYPE_SET;
  arg_set_attr0.set_type = LIBXSMM_MATRIX_ARG_SET_TYPE_STRIDE_BASE;
  arg_set_attr0.set_cardinality_hint = blocks_i[2];
  if (in_dt == LIBXSMM_DATATYPE_F32) {
    arg_set_attr0.set_stride_hint = ld_i[2] * n_i[2] * sizeof(float);
  } else {
    arg_set_attr0.set_stride_hint = ld_i[2] * n_i[2] * sizeof(libxsmm_bfloat16);
  }

  arg_set_attr1.type = LIBXSMM_MATRIX_ARG_TYPE_SET;
  arg_set_attr1.set_type = LIBXSMM_MATRIX_ARG_SET_TYPE_STRIDE_BASE;
  arg_set_attr1.set_cardinality_hint = blocks_i[3];
  if (in_dt == LIBXSMM_DATATYPE_F32) {
    arg_set_attr1.set_stride_hint = ld_i[3] * n_i[3] * sizeof(float);
  } else {
    arg_set_attr1.set_stride_hint = ld_i[3] * n_i[3] * sizeof(libxsmm_bfloat16);
  }

  brcount = blocks_i[2];
  op_arg_arr[7].tertiary = (void*)&brcount;
  eqn_param.ops_args = op_arg_arr;

  my_eqn1 = libxsmm_matrix_eqn_create();
  arg_metadata[0].eqn_idx     = my_eqn1;
  arg_metadata[1].eqn_idx     = my_eqn1;
  arg_metadata[2].eqn_idx     = my_eqn1;
  arg_metadata[3].eqn_idx     = my_eqn1;
  op_metadata[0].eqn_idx      = my_eqn1;
  op_metadata[1].eqn_idx      = my_eqn1;
  op_metadata[1].op_arg_pos   = 7;

  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE);
  libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_UNARY_GELU, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE);
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[0], arg_shape[0], arg_singular_attr);
  libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_UNARY_TANH, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE);
  libxsmm_matrix_eqn_push_back_ternary_op_v2(op_metadata[1], LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[2], arg_shape[2], arg_set_attr0);
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[3], arg_shape[3], arg_set_attr1);
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[1], arg_shape[1], arg_singular_attr);
  libxsmm_matrix_eqn_tree_print( my_eqn1 );
  func1 = libxsmm_dispatch_matrix_eqn_v2( my_eqn1, arg_shape[n_tensors-1] );
  func1(&eqn_param);
  /* Recover type of arg1  */
  arg_shape[1].type = in_dt;

  if (datatype_mode == 0 || datatype_mode == 1) {
    eqn1_f32( arg[ref_id], m_i[n_tensors-1], n_i[n_tensors-1], ld_i[n_tensors-1],
              arg[0], m_i[0], n_i[0], ld_i[0],
              arg[1], m_i[1], n_i[1], ld_i[1], blocks_i[2],
              arg[2], m_i[2], n_i[2], ld_i[2], ld_i[2] * n_i[2],
              arg[3], m_i[3], n_i[3], ld_i[3], ld_i[3] * n_i[3]);
  } else if (datatype_mode == 2) {
  } else if (datatype_mode == 3) {
  }

  if (datatype_mode == 1) {
    libxsmm_convert_bf16_f32( bf16_arg[n_tensors-1], arg[n_tensors-1], ld_i[n_tensors-1] * n_i[n_tensors-1] * blocks_i[n_tensors-1] );
  }
  /* compare */
  printf("##########################################\n");
  printf("#   Correctness  - Output                #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ld_i[n_tensors-1] * n_i[n_tensors-1] * blocks_i[n_tensors-1], 1, arg[ref_id], arg[n_tensors-1], 0, 0);

  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  /* Now benchmarking the equations */
  if (datatype_mode == 0 || datatype_mode == 1) {
    eqn1_f32( arg[ref_id], m_i[n_tensors-1], n_i[n_tensors-1], ld_i[n_tensors-1],
              arg[0], m_i[0], n_i[0], ld_i[0],
              arg[1], m_i[1], n_i[1], ld_i[1], blocks_i[2],
              arg[2], m_i[2], n_i[2], ld_i[2], ld_i[2] * n_i[2],
              arg[3], m_i[3], n_i[3], ld_i[3], ld_i[3] * n_i[3]);
  } else if (datatype_mode == 2) {
  } else if (datatype_mode == 3) {
  }
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
    if (datatype_mode == 0 || datatype_mode == 1) {
      eqn1_f32( arg[ref_id], m_i[n_tensors-1], n_i[n_tensors-1], ld_i[n_tensors-1],
                arg[0], m_i[0], n_i[0], ld_i[0],
                arg[1], m_i[1], n_i[1], ld_i[1], blocks_i[2],
                arg[2], m_i[2], n_i[2], ld_i[2], ld_i[2] * n_i[2],
                arg[3], m_i[3], n_i[3], ld_i[3], ld_i[3] * n_i[3]);
    } else if (datatype_mode == 2) {
    } else if (datatype_mode == 3) {
    }
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Compiler equation time  = %.5g\n", ((double)(l_total)));

  func1(&eqn_param);
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
    func1(&eqn_param);
  }
  l_end = libxsmm_timer_tick();
  l_total2 = libxsmm_timer_duration(l_start, l_end);
  printf("JITed TPP equation time = %.5g\n", ((double)(l_total2)));
  printf("Speedup (%d iters) is %.5g\n", iters, l_total/l_total2);

  if (datatype_mode == 0) {
    printf("\n\nNow testing equation with fused avx512 GEMM...\n\n");
    /* sigmoid( Sum(Ci x Di) + colbias + 1.0) */
    float *colbias_f32_fused = (float*) libxsmm_aligned_malloc( sizeof(float)*m_i[2],   64);
    memcpy(colbias_f32_fused, arg[2], m_i[2] * sizeof(float));

    libxsmm_meqn_arg_shape  colbias_shape_fused;
    colbias_shape_fused.m = m_i[2];
    colbias_shape_fused.n = 1;
    colbias_shape_fused.ld = m_i[2];
    colbias_shape_fused.type = in_dt;

    my_eqn5 = libxsmm_matrix_eqn_create();
    arg_metadata[2].eqn_idx     = my_eqn5;
    arg_metadata[3].eqn_idx     = my_eqn5;
    op_metadata[0].eqn_idx      = my_eqn5;
    op_metadata[1].eqn_idx      = my_eqn5;
    arg_metadata[42].eqn_idx    = my_eqn5;
    arg_metadata[42].in_arg_pos = 42;

    if (relu_sigmoid_fusion_mode == 0) {
      /* Do nothing  */
    } else if (relu_sigmoid_fusion_mode == 1) {
      libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_UNARY_RELU, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    } else if (relu_sigmoid_fusion_mode == 2) {
      libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_UNARY_SIGMOID, LIBXSMM_DATATYPE_F32 , LIBXSMM_MELTW_FLAG_UNARY_NONE);
    }
    libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0);
    libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_UNARY_INC, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[42], colbias_shape_fused, arg_singular_attr);
    libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[1], LIBXSMM_MELTW_TYPE_BINARY_BRGEMM, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE);
    libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[2], arg_shape[2], arg_set_attr0);
    libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[3], arg_shape[3], arg_set_attr1);
    libxsmm_matrix_eqn_tree_print( my_eqn5 );
    func5 = libxsmm_dispatch_matrix_eqn_v2( my_eqn5, arg_shape[n_tensors-1] );

    arg_array[42].primary = colbias_f32_fused;
    func5(&eqn_param);

    eqn4_f32( arg[ref_id], m_i[n_tensors-1], n_i[n_tensors-1], ld_i[n_tensors-1],
        arg[0], m_i[0], n_i[0], ld_i[0],
        arg[1], m_i[1], n_i[1], ld_i[1], blocks_i[2],
        arg[2], m_i[2], n_i[2], ld_i[2], ld_i[2] * n_i[2],
        arg[3], m_i[3], n_i[3], ld_i[3], ld_i[3] * n_i[3], colbias_f32_fused, relu_sigmoid_fusion_mode);

    /* compare */
    printf("##########################################\n");
    printf("#   Correctness  - Output                #\n");
    printf("##########################################\n");
    libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ld_i[n_tensors-1] * n_i[n_tensors-1] * blocks_i[n_tensors-1], 1, arg[ref_id], arg[n_tensors-1], 0, 0);

    printf("L1 reference  : %.25g\n", norms_out.l1_ref);
    printf("L1 test       : %.25g\n", norms_out.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

    /* Now benchmarking the equations */
    if (datatype_mode == 0 || datatype_mode == 1) {
      eqn4_f32( arg[ref_id], m_i[n_tensors-1], n_i[n_tensors-1], ld_i[n_tensors-1],
          arg[0], m_i[0], n_i[0], ld_i[0],
          arg[1], m_i[1], n_i[1], ld_i[1], blocks_i[2],
          arg[2], m_i[2], n_i[2], ld_i[2], ld_i[2] * n_i[2],
          arg[3], m_i[3], n_i[3], ld_i[3], ld_i[3] * n_i[3], colbias_f32_fused, relu_sigmoid_fusion_mode);
    } else if (datatype_mode == 2) {
    } else if (datatype_mode == 3) {
    }
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      if (datatype_mode == 0 || datatype_mode == 1) {
        eqn4_f32( arg[ref_id], m_i[n_tensors-1], n_i[n_tensors-1], ld_i[n_tensors-1],
            arg[0], m_i[0], n_i[0], ld_i[0],
            arg[1], m_i[1], n_i[1], ld_i[1], blocks_i[2],
            arg[2], m_i[2], n_i[2], ld_i[2], ld_i[2] * n_i[2],
            arg[3], m_i[3], n_i[3], ld_i[3], ld_i[3] * n_i[3], colbias_f32_fused, relu_sigmoid_fusion_mode);
      } else if (datatype_mode == 2) {
      } else if (datatype_mode == 3) {
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("Compiler equation time  = %.5g\n", ((double)(l_total)));

    func5(&eqn_param);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      func5(&eqn_param);
    }
    l_end = libxsmm_timer_tick();
    l_total2 = libxsmm_timer_duration(l_start, l_end);
    printf("JITed TPP equation time = %.5g\n", ((double)(l_total2)));
    printf("Speedup (%d iters) is %.5g\n", iters, l_total/l_total2);
  }

  if ((datatype_mode == 1) && (n_i[2] % 2 == 0)) {
    printf("\n\nNow testing equation 2...\n\n");

    /* Result = gelu(A) * tanh( Sum Ci x Di ) */
    arg_set_attr0.type = LIBXSMM_MATRIX_ARG_TYPE_SET;
    arg_set_attr0.set_type = LIBXSMM_MATRIX_ARG_SET_TYPE_STRIDE_BASE;
    arg_set_attr0.set_cardinality_hint = blocks_i[2];
    arg_set_attr0.set_stride_hint = ld_i[2] * n_i[2] * sizeof(libxsmm_bfloat16);

    arg_set_attr1.type = LIBXSMM_MATRIX_ARG_TYPE_SET;
    arg_set_attr1.set_type = LIBXSMM_MATRIX_ARG_SET_TYPE_STRIDE_BASE;
    arg_set_attr1.set_cardinality_hint = blocks_i[3];
    arg_set_attr1.set_stride_hint = ld_i[3] * n_i[3] * sizeof(libxsmm_bfloat16);

    brcount = blocks_i[2];
    op_arg_arr[7].tertiary = (void*)&brcount;
    eqn_param.ops_args = op_arg_arr;

    /* Create copy of C in VNNI format */
    libxsmm_bfloat16 *copy_C = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*n_i[2]*ld_i[2]*blocks_i[2],   64);
    libxsmm_bfloat16 *orig_C = bf16_arg[2];
    int ___i = 0;
    for ( l = 0; l < blocks_i[2]; l++) {
      for ( i = 0; i < n_i[2]/2; i++ ) {
        for ( j = 0; j < m_i[2]; j++ ) {
          for ( ___i = 0; ___i < 2; ___i++ ) {
            copy_C[___i + j*2 + i * ld_i[2] * 2 + l * ld_i[2] * n_i[2]] = orig_C[j + (i*2+___i) * ld_i[2] + l * ld_i[2] * n_i[2]];
          }
        }
      }
    }
    bf16_arg_array[2].primary = copy_C;

    my_eqn2 = libxsmm_matrix_eqn_create();
    arg_metadata[0].eqn_idx     = my_eqn2;
    arg_metadata[2].eqn_idx     = my_eqn2;
    arg_metadata[3].eqn_idx     = my_eqn2;
    op_metadata[0].eqn_idx      = my_eqn2;
    op_metadata[1].eqn_idx      = my_eqn2;

    libxsmm_matrix_eqn_push_back_binary_op_v2( op_metadata[0], LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE);
    libxsmm_matrix_eqn_push_back_unary_op_v2( op_metadata[0], LIBXSMM_MELTW_TYPE_UNARY_GELU, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    libxsmm_matrix_eqn_push_back_arg_v2( arg_metadata[0], arg_shape[0], arg_singular_attr);
    libxsmm_matrix_eqn_push_back_unary_op_v2( op_metadata[0], LIBXSMM_MELTW_TYPE_UNARY_TANH, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    libxsmm_matrix_eqn_push_back_binary_op_v2( op_metadata[1], LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_VNNI, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE);
    libxsmm_matrix_eqn_push_back_arg_v2( arg_metadata[2], arg_shape[2], arg_set_attr0);
    libxsmm_matrix_eqn_push_back_arg_v2( arg_metadata[3], arg_shape[3], arg_set_attr1);
    libxsmm_matrix_eqn_tree_print( my_eqn2 );
    func2 = libxsmm_dispatch_matrix_eqn_v2( my_eqn2, arg_shape[n_tensors-1] );
    func2(&eqn_param);

    eqn2_f32( arg[ref_id], m_i[n_tensors-1], n_i[n_tensors-1], ld_i[n_tensors-1],
        arg[0], m_i[0], n_i[0], ld_i[0],
        arg[1], m_i[1], n_i[1], ld_i[1], blocks_i[2],
        arg[2], m_i[2], n_i[2], ld_i[2], ld_i[2] * n_i[2],
        arg[3], m_i[3], n_i[3], ld_i[3], ld_i[3] * n_i[3]);

    libxsmm_convert_bf16_f32( bf16_arg[n_tensors-1], arg[n_tensors-1], ld_i[n_tensors-1] * n_i[n_tensors-1] * blocks_i[n_tensors-1] );

    /* compare */
    printf("##########################################\n");
    printf("#   Correctness  - Output                #\n");
    printf("##########################################\n");
    libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ld_i[n_tensors-1] * n_i[n_tensors-1] * blocks_i[n_tensors-1], 1, arg[ref_id], arg[n_tensors-1], 0, 0);

    printf("L1 reference  : %.25g\n", norms_out.l1_ref);
    printf("L1 test       : %.25g\n", norms_out.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

    /* Now benchmarking the equations */
    if (datatype_mode == 0 || datatype_mode == 1) {
      eqn2_f32( arg[ref_id], m_i[n_tensors-1], n_i[n_tensors-1], ld_i[n_tensors-1],
          arg[0], m_i[0], n_i[0], ld_i[0],
          arg[1], m_i[1], n_i[1], ld_i[1], blocks_i[2],
          arg[2], m_i[2], n_i[2], ld_i[2], ld_i[2] * n_i[2],
          arg[3], m_i[3], n_i[3], ld_i[3], ld_i[3] * n_i[3]);
    } else if (datatype_mode == 2) {
    } else if (datatype_mode == 3) {
    }
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      if (datatype_mode == 0 || datatype_mode == 1) {
        eqn2_f32( arg[ref_id], m_i[n_tensors-1], n_i[n_tensors-1], ld_i[n_tensors-1],
            arg[0], m_i[0], n_i[0], ld_i[0],
            arg[1], m_i[1], n_i[1], ld_i[1], blocks_i[2],
            arg[2], m_i[2], n_i[2], ld_i[2], ld_i[2] * n_i[2],
            arg[3], m_i[3], n_i[3], ld_i[3], ld_i[3] * n_i[3]);
      } else if (datatype_mode == 2) {
      } else if (datatype_mode == 3) {
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("Compiler equation time  = %.5g\n", ((double)(l_total)));

    func2(&eqn_param);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      func2(&eqn_param);
    }
    l_end = libxsmm_timer_tick();
    l_total2 = libxsmm_timer_duration(l_start, l_end);
    printf("JITed TPP equation time = %.5g\n", ((double)(l_total2)));
    printf("Speedup (%d iters) is %.5g\n", iters, l_total/l_total2);


    printf("\n\nNow testing equation 3...\n\n");

    /* Result = Sum(Ci x Di) + gelu(A) */
    arg_set_attr0.type = LIBXSMM_MATRIX_ARG_TYPE_SET;
    arg_set_attr0.set_type = LIBXSMM_MATRIX_ARG_SET_TYPE_STRIDE_BASE;
    arg_set_attr0.set_cardinality_hint = blocks_i[2];
    arg_set_attr0.set_stride_hint = ld_i[2] * n_i[2] * sizeof(libxsmm_bfloat16);

    arg_set_attr1.type = LIBXSMM_MATRIX_ARG_TYPE_SET;
    arg_set_attr1.set_type = LIBXSMM_MATRIX_ARG_SET_TYPE_STRIDE_BASE;
    arg_set_attr1.set_cardinality_hint = blocks_i[3];
    arg_set_attr1.set_stride_hint = ld_i[3] * n_i[3] * sizeof(libxsmm_bfloat16);

    brcount = blocks_i[2];
    op_arg_arr[7].tertiary = (void*)&brcount;
    eqn_param.ops_args = op_arg_arr;

    my_eqn3 = libxsmm_matrix_eqn_create();
    arg_metadata[0].eqn_idx     = my_eqn3;
    arg_metadata[2].eqn_idx     = my_eqn3;
    arg_metadata[3].eqn_idx     = my_eqn3;
    op_metadata[0].eqn_idx      = my_eqn3;
    op_metadata[1].eqn_idx      = my_eqn3;

    libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    libxsmm_matrix_eqn_push_back_ternary_op_v2(op_metadata[1], LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM_A_VNNI, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
    libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[2], arg_shape[2], arg_set_attr0);
    libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[3], arg_shape[3], arg_set_attr1);
    libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_UNARY_GELU, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[0], arg_shape[0], arg_singular_attr);

    libxsmm_matrix_eqn_tree_print( my_eqn3 );
    func3 = libxsmm_dispatch_matrix_eqn_v2( my_eqn3, arg_shape[n_tensors-1] );
    func3(&eqn_param);

    eqn3_f32( arg[ref_id], m_i[n_tensors-1], n_i[n_tensors-1], ld_i[n_tensors-1],
        arg[0], m_i[0], n_i[0], ld_i[0],
        arg[1], m_i[1], n_i[1], ld_i[1], blocks_i[2],
        arg[2], m_i[2], n_i[2], ld_i[2], ld_i[2] * n_i[2],
        arg[3], m_i[3], n_i[3], ld_i[3], ld_i[3] * n_i[3]);

    libxsmm_convert_bf16_f32( bf16_arg[n_tensors-1], arg[n_tensors-1], ld_i[n_tensors-1] * n_i[n_tensors-1] * blocks_i[n_tensors-1] );

    /* compare */
    printf("##########################################\n");
    printf("#   Correctness  - Output                #\n");
    printf("##########################################\n");
    libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ld_i[n_tensors-1] * n_i[n_tensors-1] * blocks_i[n_tensors-1], 1, arg[ref_id], arg[n_tensors-1], 0, 0);

    printf("L1 reference  : %.25g\n", norms_out.l1_ref);
    printf("L1 test       : %.25g\n", norms_out.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

    /* Now benchmarking the equations */
    if (datatype_mode == 0 || datatype_mode == 1) {
      eqn3_f32( arg[ref_id], m_i[n_tensors-1], n_i[n_tensors-1], ld_i[n_tensors-1],
          arg[0], m_i[0], n_i[0], ld_i[0],
          arg[1], m_i[1], n_i[1], ld_i[1], blocks_i[2],
          arg[2], m_i[2], n_i[2], ld_i[2], ld_i[2] * n_i[2],
          arg[3], m_i[3], n_i[3], ld_i[3], ld_i[3] * n_i[3]);
    } else if (datatype_mode == 2) {
    } else if (datatype_mode == 3) {
    }
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      if (datatype_mode == 0 || datatype_mode == 1) {
        eqn3_f32( arg[ref_id], m_i[n_tensors-1], n_i[n_tensors-1], ld_i[n_tensors-1],
            arg[0], m_i[0], n_i[0], ld_i[0],
            arg[1], m_i[1], n_i[1], ld_i[1], blocks_i[2],
            arg[2], m_i[2], n_i[2], ld_i[2], ld_i[2] * n_i[2],
            arg[3], m_i[3], n_i[3], ld_i[3], ld_i[3] * n_i[3]);
      } else if (datatype_mode == 2) {
      } else if (datatype_mode == 3) {
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("Compiler equation time  = %.5g\n", ((double)(l_total)));

    func3(&eqn_param);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      func3(&eqn_param);
    }
    l_end = libxsmm_timer_tick();
    l_total2 = libxsmm_timer_duration(l_start, l_end);
    printf("JITed TPP equation time = %.5g\n", ((double)(l_total2)));
    printf("Speedup (%d iters) is %.5g\n", iters, l_total/l_total2);


    printf("\n\nNow testing equation 4...\n\n");

    /* sigmoid( Sum(Ci x Di) + colbias + 1.0) */

    libxsmm_bfloat16 *colbias = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*m_i[2],   64);
    float *colbias_f32 = (float*) libxsmm_aligned_malloc( sizeof(float)*m_i[2],   64);
    libxsmm_rne_convert_fp32_bf16( arg[2], colbias, m_i[2] );
    libxsmm_convert_bf16_f32(colbias, colbias_f32, m_i[2] );

    libxsmm_meqn_arg_shape  colbias_shape_bf16;
    colbias_shape_bf16.m = m_i[2];
    colbias_shape_bf16.n = 1;
    colbias_shape_bf16.ld = m_i[2];
    colbias_shape_bf16.type = in_dt;

    my_eqn4 = libxsmm_matrix_eqn_create();
    arg_metadata[2].eqn_idx     = my_eqn4;
    arg_metadata[3].eqn_idx     = my_eqn4;
    arg_metadata[42].eqn_idx    = my_eqn4;
    arg_metadata[42].in_arg_pos = 42;
    op_metadata[0].eqn_idx      = my_eqn4;
    op_metadata[1].eqn_idx      = my_eqn4;

    if (relu_sigmoid_fusion_mode == 0) {
      /* Do nothing  */
    } else if (relu_sigmoid_fusion_mode == 1) {
      libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_UNARY_RELU, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    } else if (relu_sigmoid_fusion_mode == 2) {
      libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_UNARY_SIGMOID, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    }
    libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0);
    libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_UNARY_INC, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[42], colbias_shape_bf16, arg_singular_attr );
    libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[1], LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_VNNI, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_BINARY_NONE);
    libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[2], arg_shape[2], arg_set_attr0);
    libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[3], arg_shape[3], arg_set_attr1);
    libxsmm_matrix_eqn_tree_print( my_eqn4 );
    func4 = libxsmm_dispatch_matrix_eqn_v2( my_eqn4, arg_shape[n_tensors-1] );
    bf16_arg_array[42].primary = colbias;

    func4(&eqn_param);

    eqn4_f32( arg[ref_id], m_i[n_tensors-1], n_i[n_tensors-1], ld_i[n_tensors-1],
        arg[0], m_i[0], n_i[0], ld_i[0],
        arg[1], m_i[1], n_i[1], ld_i[1], blocks_i[2],
        arg[2], m_i[2], n_i[2], ld_i[2], ld_i[2] * n_i[2],
        arg[3], m_i[3], n_i[3], ld_i[3], ld_i[3] * n_i[3], colbias_f32, relu_sigmoid_fusion_mode);

    libxsmm_convert_bf16_f32( bf16_arg[n_tensors-1], arg[n_tensors-1], ld_i[n_tensors-1] * n_i[n_tensors-1] * blocks_i[n_tensors-1] );

    /* compare */
    printf("##########################################\n");
    printf("#   Correctness  - Output                #\n");
    printf("##########################################\n");
    libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ld_i[n_tensors-1] * n_i[n_tensors-1] * blocks_i[n_tensors-1], 1, arg[ref_id], arg[n_tensors-1], 0, 0);

    printf("L1 reference  : %.25g\n", norms_out.l1_ref);
    printf("L1 test       : %.25g\n", norms_out.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

    /* Now benchmarking the equations */
    if (datatype_mode == 0 || datatype_mode == 1) {
      eqn4_f32( arg[ref_id], m_i[n_tensors-1], n_i[n_tensors-1], ld_i[n_tensors-1],
          arg[0], m_i[0], n_i[0], ld_i[0],
          arg[1], m_i[1], n_i[1], ld_i[1], blocks_i[2],
          arg[2], m_i[2], n_i[2], ld_i[2], ld_i[2] * n_i[2],
          arg[3], m_i[3], n_i[3], ld_i[3], ld_i[3] * n_i[3], colbias_f32, relu_sigmoid_fusion_mode);
    } else if (datatype_mode == 2) {
    } else if (datatype_mode == 3) {
    }
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      if (datatype_mode == 0 || datatype_mode == 1) {
        eqn4_f32( arg[ref_id], m_i[n_tensors-1], n_i[n_tensors-1], ld_i[n_tensors-1],
            arg[0], m_i[0], n_i[0], ld_i[0],
            arg[1], m_i[1], n_i[1], ld_i[1], blocks_i[2],
            arg[2], m_i[2], n_i[2], ld_i[2], ld_i[2] * n_i[2],
            arg[3], m_i[3], n_i[3], ld_i[3], ld_i[3] * n_i[3], colbias_f32, relu_sigmoid_fusion_mode);
      } else if (datatype_mode == 2) {
      } else if (datatype_mode == 3) {
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);
    printf("Compiler equation time  = %.5g\n", ((double)(l_total)));

    func4(&eqn_param);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      func4(&eqn_param);
    }
    l_end = libxsmm_timer_tick();
    l_total2 = libxsmm_timer_duration(l_start, l_end);
    printf("JITed TPP equation time = %.5g\n", ((double)(l_total2)));
    printf("Speedup (%d iters) is %.5g\n", iters, l_total/l_total2);
  }

  return 0;
}
