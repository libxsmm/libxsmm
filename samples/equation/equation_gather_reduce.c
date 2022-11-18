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
#include <equation_common.h>

#define EXPANSION_FACTOR 8

LIBXSMM_INLINE
void eqn1_f32f32(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, float *arg0, unsigned int *cols_ind_array, float *out) {
  libxsmm_blasint i, j, ind;
  for ( j = 0; j < M; ++j ) {
    out[j] = 0.0;
    for ( i = 0; i < N; ++i ) {
      ind = cols_ind_array[i];
      out[j] += arg0[ind * ld + j];
    }
  }
}

int main( int argc, char* argv[] ) {
  int ret = EXIT_SUCCESS;
  double error_bound = 0.00001;
  libxsmm_blasint my_eqn0;
  libxsmm_matrix_eqn_function func0;
  libxsmm_blasint i, j,it;
  libxsmm_matrix_eqn_param eqn_param;
  unsigned long long l_start, l_end;
  double l_total = 0, l_total2 = 0;
  libxsmm_matdiff_info norms_out;
  float *out, *eqn_out;
  libxsmm_bfloat16 *bf16_out, *bf16_eqn_out;
  libxsmm_float16 *f16_out, *f16_eqn_out;
  libxsmm_bfloat8 *bf8_out, *bf8_eqn_out;
  libxsmm_hfloat8 *hf8_out, *hf8_eqn_out;
  libxsmm_matrix_arg arg_array[1];
  libxsmm_matrix_arg bf16_arg_array[1];
  libxsmm_matrix_arg f16_arg_array[1];
  libxsmm_matrix_arg bf8_arg_array[1];
  libxsmm_matrix_arg hf8_arg_array[1];
  libxsmm_matrix_eqn_arg_metadata arg_metadata;
  libxsmm_matrix_eqn_op_metadata  op_metadata;
  libxsmm_meqn_arg_shape          arg_shape_in, arg_shape_out;
  libxsmm_matrix_arg_attributes   arg_singular_attr = libxsmm_create_matrix_arg_attributes( LIBXSMM_MATRIX_ARG_TYPE_SINGULAR, LIBXSMM_MATRIX_ARG_SET_TYPE_NONE, 0, 0);
  unsigned int       *cols_ind_array;
  unsigned long long *cols_ind_array_64b;
  unsigned long long *unique_random_array;
  float              *large_input;
  libxsmm_bfloat16   *large_input_bf16;
  libxsmm_float16   *large_input_f16;
  libxsmm_bfloat8   *large_input_bf8;
  libxsmm_hfloat8   *large_input_hf8;

  int M = 64;
  int N = 64;
  int large_N = EXPANSION_FACTOR * N;
  int ld = 64;
  int iters = 100;
  int datatype_mode = 0;
  int idx_type = 0;
  libxsmm_datatype  in_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype  out_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype  compute_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;

  if ( argc > 1 ) M = atoi(argv[1]);
  if ( argc > 2 ) N = atoi(argv[2]);
  if ( argc > 3 ) ld = atoi(argv[3]);
  if ( argc > 4 ) datatype_mode = atoi(argv[4]);
  if ( argc > 5 ) idx_type = atoi(argv[5]);
  if ( argc > 6 ) iters = atoi(argv[6]);

  large_N = EXPANSION_FACTOR * N;

  set_in_out_compute_dt(datatype_mode, &in_dt, &out_dt, &compute_dt);

  out  = (float*) libxsmm_aligned_malloc( sizeof(float)*1*ld,   64);
  eqn_out  = (float*) libxsmm_aligned_malloc( sizeof(float)*1*ld,   64);

  cols_ind_array      = (unsigned int*) libxsmm_aligned_malloc( sizeof(unsigned int)*N,   64);
  cols_ind_array_64b  = (unsigned long long*) libxsmm_aligned_malloc( sizeof(unsigned long long)*N,   64);
  unique_random_array = (unsigned long long*) libxsmm_aligned_malloc( sizeof(unsigned long long)*large_N,   64);

  large_input = (float*) libxsmm_aligned_malloc( sizeof(float)*large_N*ld,   64);
  large_input_bf16 = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*large_N*ld,   64);
  large_input_f16 = (libxsmm_float16*) libxsmm_aligned_malloc( sizeof(libxsmm_float16)*large_N*ld,   64);
  large_input_bf8 = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*large_N*ld,   64);
  large_input_hf8 = (libxsmm_hfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_hfloat8)*large_N*ld,   64);

  bf16_out  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*1*ld,   64);
  bf16_eqn_out  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*1*ld,   64);

  f16_out  = (libxsmm_float16*) libxsmm_aligned_malloc( sizeof(libxsmm_float16)*1*ld,   64);
  f16_eqn_out  = (libxsmm_float16*) libxsmm_aligned_malloc( sizeof(libxsmm_float16)*1*ld,   64);

  bf8_out  = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*1*ld,   64);
  bf8_eqn_out  = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*1*ld,   64);

  hf8_out  = (libxsmm_hfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_hfloat8)*1*ld,   64);
  hf8_eqn_out  = (libxsmm_hfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_hfloat8)*1*ld,   64);

  libxsmm_init();
  libxsmm_matdiff_clear(&norms_out);

  for ( i = 0; i < large_N; ++i ) {
    for ( j = 0; j < ld; ++j ) {
      large_input[(i*ld)+j] = (float)libxsmm_rng_f64();
      libxsmm_rne_convert_fp32_bf16( &large_input[(i*ld)+j], &large_input_bf16[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_f16( &large_input[(i*ld)+j], &large_input_f16[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf8( &large_input[(i*ld)+j], &large_input_bf8[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_hf8( &large_input[(i*ld)+j], &large_input_hf8[(i*ld)+j], 1 );
      if (datatype_mode == 1) {
        large_input[(i*ld)+j] = upconvert_bf16(large_input_bf16[(i*ld)+j]);
      } else if (datatype_mode == 4) {
        large_input[(i*ld)+j] = upconvert_bf8(large_input_bf8[(i*ld)+j]);
      } else if (datatype_mode == 7) {
        large_input[(i*ld)+j] = upconvert_f16(large_input_f16[(i*ld)+j]);
      } else if (datatype_mode == 10) {
        large_input[(i*ld)+j] = upconvert_hf8(large_input_hf8[(i*ld)+j]);
      }
    }
  }

  print_dt_info(datatype_mode);

  /* Now we test a gather-reduce equation */
  create_unique_random_array(unique_random_array, large_N);
  for (i = 0; i < N; i++) {
    cols_ind_array_64b[i] = (unsigned long long) unique_random_array[i];
    cols_ind_array[i] = (unsigned int) cols_ind_array_64b[i];
  }
  my_eqn0       = libxsmm_matrix_eqn_create();
  arg_metadata  = libxsmm_create_matrix_eqn_arg_metadata(my_eqn0, 0);
  op_metadata   = libxsmm_create_matrix_eqn_op_metadata(my_eqn0, -1);
  arg_shape_in  = libxsmm_create_meqn_arg_shape( M, N, ld, in_dt );
  arg_shape_out = libxsmm_create_meqn_arg_shape( M, 1, ld, out_dt);
  unary_flags   = (idx_type == 0) ? LIBXSMM_MELTW_FLAG_UNARY_GS_COLS | LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES : LIBXSMM_MELTW_FLAG_UNARY_GS_COLS | LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES;

  libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, in_dt, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS);
  libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata, LIBXSMM_MELTW_TYPE_UNARY_GATHER, in_dt, unary_flags);
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata, arg_shape_in, arg_singular_attr);
  func0 = libxsmm_dispatch_matrix_eqn_v2( my_eqn0, arg_shape_out );

  if (datatype_mode == 0) {
    arg_array[0].primary = large_input;
    if (idx_type == 0) {
      arg_array[0].secondary = cols_ind_array;
    } else {
      arg_array[0].secondary = cols_ind_array_64b;
    }
  } else if (datatype_mode == 1) {
    bf16_arg_array[0].primary = large_input_bf16;
    if (idx_type == 0) {
      bf16_arg_array[0].secondary = cols_ind_array;
    } else {
      bf16_arg_array[0].secondary = cols_ind_array_64b;
    }
  } else if(datatype_mode == 4) {
    bf8_arg_array[0].primary = large_input_bf8;
    if (idx_type == 0) {
      bf8_arg_array[0].secondary = cols_ind_array;
    } else {
      bf8_arg_array[0].secondary = cols_ind_array_64b;
    }
  } else if (datatype_mode == 7) {
    f16_arg_array[0].primary = large_input_f16;
    if (idx_type == 0) {
      f16_arg_array[0].secondary = cols_ind_array;
    } else {
      f16_arg_array[0].secondary = cols_ind_array_64b;
    }
  } else if (datatype_mode == 10) {
    hf8_arg_array[0].primary = large_input_hf8;
    if (idx_type == 0) {
      hf8_arg_array[0].secondary = cols_ind_array;
    } else {
      hf8_arg_array[0].secondary = cols_ind_array_64b;
    }
  }

  if ( in_dt == LIBXSMM_DATATYPE_F32 ) {
    eqn_param.inputs = arg_array;
  } else if ( in_dt == LIBXSMM_DATATYPE_BF16  ) {
    eqn_param.inputs = bf16_arg_array;
  } else if ( in_dt == LIBXSMM_DATATYPE_F16  ) {
    eqn_param.inputs = f16_arg_array;
  } else if ( in_dt == LIBXSMM_DATATYPE_BF8  ) {
    eqn_param.inputs = bf8_arg_array;
  } else if ( in_dt == LIBXSMM_DATATYPE_HF8  ) {
    eqn_param.inputs = hf8_arg_array;
  }
  if ( out_dt == LIBXSMM_DATATYPE_F32 ) {
    eqn_param.output.primary = eqn_out;
  } else if ( out_dt == LIBXSMM_DATATYPE_BF16  ) {
    eqn_param.output.primary  = bf16_eqn_out;
  } else if ( out_dt == LIBXSMM_DATATYPE_F16  ) {
    eqn_param.output.primary  = f16_eqn_out;
  } else if ( out_dt == LIBXSMM_DATATYPE_BF8  ) {
    eqn_param.output.primary  = bf8_eqn_out;
  } else if ( out_dt == LIBXSMM_DATATYPE_HF8  ) {
    eqn_param.output.primary  = hf8_eqn_out;
  }

  func0(&eqn_param);
  eqn1_f32f32(M, N, ld, large_input, cols_ind_array, out);
  if (datatype_mode == 1) {
    for (i = 0; i < M; i++) {
      libxsmm_bfloat16 _eqn_out;
      libxsmm_rne_convert_fp32_bf16( &out[i], &_eqn_out, 1 );
      out[i] = upconvert_bf16(_eqn_out);
    }
  } else if (datatype_mode == 4) {
    for (i = 0; i < M; i++) {
      libxsmm_bfloat8 _eqn_out;
      libxsmm_rne_convert_fp32_bf8( &out[i], &_eqn_out, 1 );
      out[i] = upconvert_bf8(_eqn_out);
    }
  } else if (datatype_mode == 7) {
    for (i = 0; i < M; i++) {
      libxsmm_float16 _eqn_out;
      libxsmm_rne_convert_fp32_f16( &out[i], &_eqn_out, 1 );
      out[i] = upconvert_f16(_eqn_out);
    }
  } else if (datatype_mode == 10) {
    for (i = 0; i < M; i++) {
      libxsmm_hfloat8 _eqn_out;
      libxsmm_rne_convert_fp32_hf8( &out[i], &_eqn_out, 1 );
      out[i] = upconvert_hf8(_eqn_out);
    }
  }

  /* compare */
  printf("\n\n##########################################\n");
  printf("#   Correctness  GATHER-REDUCE- Output   #\n");
  printf("##########################################\n");
  if (datatype_mode == 1) {
    for (i = 0; i < M; i++) {
      eqn_out[i] = upconvert_bf16(bf16_eqn_out[i]);
    }
  } else if (datatype_mode == 4) {
    for (i = 0; i < M; i++) {
      eqn_out[i] = upconvert_bf8(bf8_eqn_out[i]);
    }
  } else if (datatype_mode == 7) {
    for (i = 0; i < M; i++) {
      eqn_out[i] = upconvert_f16(f16_eqn_out[i]);
    }
  } else if (datatype_mode == 10) {
    for (i = 0; i < M; i++) {
      eqn_out[i] = upconvert_hf8(hf8_eqn_out[i]);
    }
  }

  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, M, 1, out, eqn_out, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  if ( norms_out.normf_rel > error_bound ) {
    ret = EXIT_FAILURE;
  }

  if (iters > 0) {
    eqn1_f32f32(M, N, ld, large_input, cols_ind_array, out);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      eqn1_f32f32(M, N, ld, large_input, cols_ind_array, out);
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

    printf("Speedup is %.5g\n", l_total/l_total2);
  }

  libxsmm_free(out);
  libxsmm_free(eqn_out);

  libxsmm_free(cols_ind_array);
  libxsmm_free(cols_ind_array_64b);
  libxsmm_free(unique_random_array);
  libxsmm_free(large_input);
  libxsmm_free(large_input_bf16);
  libxsmm_free(large_input_f16);
  libxsmm_free(large_input_bf8);
  libxsmm_free(large_input_hf8);

  libxsmm_free(bf16_out);
  libxsmm_free(bf16_eqn_out);

  libxsmm_free(f16_out);
  libxsmm_free(f16_eqn_out);

  libxsmm_free(bf8_out);
  libxsmm_free(bf8_eqn_out);

  libxsmm_free(hf8_out);
  libxsmm_free(hf8_eqn_out);

  return ret;
}
