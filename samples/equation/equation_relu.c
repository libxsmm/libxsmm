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

LIBXSMM_INLINE
void eqn0_f32f32(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, float *arg0, float *arg1, float *arg2, unsigned char* relu_mask, float *out) {
  libxsmm_blasint i, j;
  libxsmm_blasint mask_ld = ((ld+15)-((ld+15)%16))/8;

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      float Arg0, Arg1, Arg2, res;
      Arg0 = arg0[(i*ld)+j];
      Arg1 = arg1[(i*ld)+j];
      Arg2 = arg2[(i*ld)+j];
      res = gelu(Arg0 - Arg1) + Arg2;
      /* Set relu mask */
      relu_mask[(i*mask_ld) + j/8] |= (unsigned char)(( res < 0.0f ) ? 0x0 : (1 << (j%8)) );
      /* Applu relu  */
      res = (res < 0.0f) ? 0.0f : res;
      out[(i*ld)+j] = res;
    }
  }
}

int main( int argc, char* argv[] ) {
  int ret = EXIT_SUCCESS;
  double error_bound = 0.00001;
  libxsmm_blasint my_eqn0;
  libxsmm_matrix_eqn_function func0;
  libxsmm_blasint i, j, s;
  libxsmm_matrix_eqn_param eqn_param;
  libxsmm_matdiff_info norms_out;
  float *arg0, *arg1, *arg2, *out, *eqn_out;
  libxsmm_matrix_arg arg_array[4];
  libxsmm_bfloat16 *bf16_arg0, *bf16_arg1, *bf16_arg2,  *bf16_out, *bf16_eqn_out;
  libxsmm_float16 *f16_arg0, *f16_arg1, *f16_arg2, *f16_out, *f16_eqn_out;
  libxsmm_bfloat8 *bf8_arg0, *bf8_arg1, *bf8_arg2, *bf8_out, *bf8_eqn_out;
  libxsmm_hfloat8 *hf8_arg0, *hf8_arg1, *hf8_arg2, *hf8_out, *hf8_eqn_out;
  libxsmm_matrix_arg bf16_arg_array[3];
  libxsmm_matrix_arg f16_arg_array[3];
  libxsmm_matrix_arg bf8_arg_array[3];
  libxsmm_matrix_arg hf8_arg_array[3];
  libxsmm_matrix_eqn_arg_metadata arg_metadata;
  libxsmm_matrix_eqn_op_metadata  op_metadata;
  libxsmm_meqn_arg_shape          arg_shape_in, arg_shape_out;
  libxsmm_matrix_arg_attributes   arg_singular_attr = libxsmm_create_matrix_arg_attributes( LIBXSMM_MATRIX_ARG_TYPE_SINGULAR, LIBXSMM_MATRIX_ARG_SET_TYPE_NONE, 0, 0);
  libxsmm_blasint mask_ld;
  unsigned char *mask_ref;
  unsigned char *mask_eqn;

  int M = 64;
  int N = 64;
  int ld = 64;
  int datatype_mode = 0;
  libxsmm_datatype  in_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype  out_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype  compute_dt = LIBXSMM_DATATYPE_F32;

  if ( argc > 1 ) M = atoi(argv[1]);
  if ( argc > 2 ) N = atoi(argv[2]);
  if ( argc > 3 ) ld = atoi(argv[3]);
  if ( argc > 4 ) datatype_mode = atoi(argv[4]);

  mask_ld = ((ld+15)-((ld+15)%16))/8;
  mask_ref = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld,   64);
  mask_eqn = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld,   64);
  memset(mask_ref, 0, sizeof(unsigned char) * N * mask_ld);
  memset(mask_eqn, 0, sizeof(unsigned char) * N * mask_ld);

  arg0 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  arg1 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  arg2 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  out  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  eqn_out  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);

  bf16_arg0 = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);
  bf16_arg1 = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);
  bf16_arg2 = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);
  bf16_out  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);
  bf16_eqn_out  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);

  f16_arg0 = (libxsmm_float16*) libxsmm_aligned_malloc( sizeof(libxsmm_float16)*N*ld,   64);
  f16_arg1 = (libxsmm_float16*) libxsmm_aligned_malloc( sizeof(libxsmm_float16)*N*ld,   64);
  f16_arg2 = (libxsmm_float16*) libxsmm_aligned_malloc( sizeof(libxsmm_float16)*N*ld,   64);
  f16_out  = (libxsmm_float16*) libxsmm_aligned_malloc( sizeof(libxsmm_float16)*N*ld,   64);
  f16_eqn_out  = (libxsmm_float16*) libxsmm_aligned_malloc( sizeof(libxsmm_float16)*N*ld,   64);

  bf8_arg0 = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*N*ld,   64);
  bf8_arg1 = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*N*ld,   64);
  bf8_arg2 = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*N*ld,   64);
  bf8_out  = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*N*ld,   64);
  bf8_eqn_out  = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*N*ld,   64);

  hf8_arg0 = (libxsmm_hfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_hfloat8)*N*ld,   64);
  hf8_arg1 = (libxsmm_hfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_hfloat8)*N*ld,   64);
  hf8_arg2 = (libxsmm_hfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_hfloat8)*N*ld,   64);
  hf8_out  = (libxsmm_hfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_hfloat8)*N*ld,   64);
  hf8_eqn_out  = (libxsmm_hfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_hfloat8)*N*ld,   64);

  libxsmm_init();
  libxsmm_matdiff_clear(&norms_out);
  set_in_out_compute_dt(datatype_mode, &in_dt, &out_dt, &compute_dt);

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ld; ++j ) {
      arg0[(i*ld)+j] = (float)libxsmm_rng_f64();
      arg1[(i*ld)+j] = (float)libxsmm_rng_f64();
      arg2[(i*ld)+j] = (float)libxsmm_rng_f64();
      out[(i*ld)+j]  = (float)libxsmm_rng_f64();
      eqn_out[(i*ld)+j] = out[(i*ld)+j];
      libxsmm_rne_convert_fp32_bf16( &arg0[(i*ld)+j], &bf16_arg0[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf16( &arg1[(i*ld)+j], &bf16_arg1[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf16( &arg2[(i*ld)+j], &bf16_arg2[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf16( &out[(i*ld)+j], &bf16_out[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf16( &eqn_out[(i*ld)+j], &bf16_eqn_out[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_f16( &arg0[(i*ld)+j], &f16_arg0[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_f16( &arg1[(i*ld)+j], &f16_arg1[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_f16( &arg2[(i*ld)+j], &f16_arg2[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_f16( &out[(i*ld)+j], &f16_out[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_f16( &eqn_out[(i*ld)+j], &f16_eqn_out[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf8( &arg0[(i*ld)+j], &bf8_arg0[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf8( &arg1[(i*ld)+j], &bf8_arg1[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf8( &arg2[(i*ld)+j], &bf8_arg2[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf8( &out[(i*ld)+j], &bf8_out[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf8( &eqn_out[(i*ld)+j], &bf8_eqn_out[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_hf8( &arg0[(i*ld)+j], &hf8_arg0[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_hf8( &arg1[(i*ld)+j], &hf8_arg1[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_hf8( &arg2[(i*ld)+j], &hf8_arg2[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_hf8( &out[(i*ld)+j], &hf8_out[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_hf8( &eqn_out[(i*ld)+j], &hf8_eqn_out[(i*ld)+j], 1 );
      if (in_dt == LIBXSMM_DATATYPE_BF16) {
        arg0[(i*ld)+j] = upconvert_bf16(bf16_arg0[(i*ld)+j]);
        arg1[(i*ld)+j] = upconvert_bf16(bf16_arg1[(i*ld)+j]);
        arg2[(i*ld)+j] = upconvert_bf16(bf16_arg2[(i*ld)+j]);
        out[(i*ld)+j] = upconvert_bf16(bf16_out[(i*ld)+j]);
        eqn_out[(i*ld)+j] = upconvert_bf16(bf16_eqn_out[(i*ld)+j]);
      }
      if (in_dt == LIBXSMM_DATATYPE_F16) {
        arg0[(i*ld)+j] = upconvert_f16(f16_arg0[(i*ld)+j]);
        arg1[(i*ld)+j] = upconvert_f16(f16_arg1[(i*ld)+j]);
        arg2[(i*ld)+j] = upconvert_f16(f16_arg2[(i*ld)+j]);
        out[(i*ld)+j] = upconvert_f16(f16_out[(i*ld)+j]);
        eqn_out[(i*ld)+j] = upconvert_f16(f16_eqn_out[(i*ld)+j]);
      }
      if (in_dt == LIBXSMM_DATATYPE_BF8) {
        arg0[(i*ld)+j] = upconvert_bf8(bf8_arg0[(i*ld)+j]);
        arg1[(i*ld)+j] = upconvert_bf8(bf8_arg1[(i*ld)+j]);
        arg2[(i*ld)+j] = upconvert_bf8(bf8_arg2[(i*ld)+j]);
        out[(i*ld)+j] = upconvert_bf8(bf8_out[(i*ld)+j]);
        eqn_out[(i*ld)+j] = upconvert_bf8(bf8_eqn_out[(i*ld)+j]);
      }
      if (in_dt == LIBXSMM_DATATYPE_HF8) {
        arg0[(i*ld)+j] = upconvert_hf8(hf8_arg0[(i*ld)+j]);
        arg1[(i*ld)+j] = upconvert_hf8(hf8_arg1[(i*ld)+j]);
        arg2[(i*ld)+j] = upconvert_hf8(hf8_arg2[(i*ld)+j]);
        out[(i*ld)+j] = upconvert_hf8(hf8_out[(i*ld)+j]);
        eqn_out[(i*ld)+j] = upconvert_hf8(hf8_eqn_out[(i*ld)+j]);
      }
    }
  }

  arg_array[0].primary = arg0;
  arg_array[1].primary = arg1;
  arg_array[2].primary = arg2;

  bf16_arg_array[0].primary = bf16_arg0;
  bf16_arg_array[1].primary = bf16_arg1;
  bf16_arg_array[2].primary = bf16_arg2;

  f16_arg_array[0].primary = f16_arg0;
  f16_arg_array[1].primary = f16_arg1;
  f16_arg_array[2].primary = f16_arg2;

  bf8_arg_array[0].primary = bf8_arg0;
  bf8_arg_array[1].primary = bf8_arg1;
  bf8_arg_array[2].primary = bf8_arg2;

  hf8_arg_array[0].primary = hf8_arg0;
  hf8_arg_array[1].primary = hf8_arg1;
  hf8_arg_array[2].primary = hf8_arg2;

  my_eqn0      = libxsmm_matrix_eqn_create();
  op_metadata   = libxsmm_create_matrix_eqn_op_metadata(my_eqn0, -1);
  arg_shape_in  = libxsmm_create_meqn_arg_shape( M, N, ld, in_dt );
  arg_shape_out = libxsmm_create_meqn_arg_shape( M, N, ld, out_dt);
  libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata, LIBXSMM_MELTW_TYPE_UNARY_RELU, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT);
  if (datatype_mode != 0) {
    libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, out_dt, LIBXSMM_MELTW_FLAG_UNARY_NONE);
  }
  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE);
  libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata, LIBXSMM_MELTW_TYPE_UNARY_GELU, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE);
  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata, LIBXSMM_MELTW_TYPE_BINARY_SUB, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE);
  arg_metadata  = libxsmm_create_matrix_eqn_arg_metadata(my_eqn0, 0);
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata, arg_shape_in, arg_singular_attr);
  arg_metadata  = libxsmm_create_matrix_eqn_arg_metadata(my_eqn0, 1);
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata, arg_shape_in, arg_singular_attr);
  arg_metadata  = libxsmm_create_matrix_eqn_arg_metadata(my_eqn0, 2);
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata, arg_shape_in, arg_singular_attr);
  func0 = libxsmm_dispatch_matrix_eqn_v2( my_eqn0, arg_shape_out );

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
  eqn_param.output.secondary = mask_eqn;

  func0(&eqn_param);
  eqn0_f32f32(M, N, ld, arg0, arg1, arg2, mask_ref, out);
  if (out_dt == LIBXSMM_DATATYPE_BF16) {
    for (i = 0; i < N*ld; i++) {
      libxsmm_bfloat16 _eqn_out;
      libxsmm_rne_convert_fp32_bf16( &out[i], &_eqn_out, 1 );
      out[i] = upconvert_bf16(_eqn_out);
    }
  } else if (out_dt == LIBXSMM_DATATYPE_BF8) {
    for (i = 0; i < N*ld; i++) {
      libxsmm_bfloat8 _eqn_out;
      libxsmm_rne_convert_fp32_bf8( &out[i], &_eqn_out, 1 );
      out[i] = upconvert_bf8(_eqn_out);
    }
  } else if (out_dt == LIBXSMM_DATATYPE_F16) {
    for (i = 0; i < N*ld; i++) {
      libxsmm_float16 _eqn_out;
      libxsmm_rne_convert_fp32_f16( &out[i], &_eqn_out, 1 );
      out[i] = upconvert_f16(_eqn_out);
    }
  } else if (out_dt == LIBXSMM_DATATYPE_HF8) {
    for (i = 0; i < N*ld; i++) {
      libxsmm_hfloat8 _eqn_out;
      libxsmm_rne_convert_fp32_hf8( &out[i], &_eqn_out, 1 );
      out[i] = upconvert_hf8(_eqn_out);
    }
  }

  print_dt_info(datatype_mode);

  /* compare */
  printf("\n\n##########################################\n");
  printf("#   Correctness RELU Equation - Output   #\n");
  printf("##########################################\n");
  if ( out_dt == LIBXSMM_DATATYPE_BF16  ) {
    for (i = 0; i < N*ld; i++) {
      eqn_out[i] = upconvert_bf16(bf16_eqn_out[i]);
    }
  } else if ( out_dt == LIBXSMM_DATATYPE_F16  ) {
    for (i = 0; i < N*ld; i++) {
      eqn_out[i] = upconvert_f16(f16_eqn_out[i]);
    }
  } else if ( out_dt == LIBXSMM_DATATYPE_BF8  ) {
    for (i = 0; i < N*ld; i++) {
      eqn_out[i] = upconvert_bf8(bf8_eqn_out[i]);
    }
  } else if ( out_dt == LIBXSMM_DATATYPE_HF8  ) {
    for (i = 0; i < N*ld; i++) {
      eqn_out[i] = upconvert_hf8(hf8_eqn_out[i]);
    }
  }

  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ld*N, 1, out, eqn_out, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M/8; ++j ) {
      if ( mask_ref[(i*mask_ld)+j] != mask_eqn[(i*mask_ld)+j] ) {
        printf("error at possition i=%i, j=%i, %u, %u\n", i, j*8, mask_ref[(i*mask_ld)+j], mask_eqn[(i*mask_ld)+j]);
        s = 1;
      }
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS mask\n");
  } else {
    printf("FAILURE mask\n");
    ret = EXIT_FAILURE;
  }
  printf("##########################################\n");
  printf("#   Correctness RELU Equation - MASK     #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_I32, (mask_ld*N)/4, 1, mask_ref, mask_eqn, 0, 0);
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

  libxsmm_free(arg0);
  libxsmm_free(arg1);
  libxsmm_free(arg2);
  libxsmm_free(out);
  libxsmm_free(eqn_out);

  libxsmm_free(bf16_arg0);
  libxsmm_free(bf16_arg1);
  libxsmm_free(bf16_arg2);
  libxsmm_free(bf16_out);
  libxsmm_free(bf16_eqn_out);

  libxsmm_free(f16_arg0);
  libxsmm_free(f16_arg1);
  libxsmm_free(f16_arg2);
  libxsmm_free(f16_out);
  libxsmm_free(f16_eqn_out);

  libxsmm_free(bf8_arg0);
  libxsmm_free(bf8_arg1);
  libxsmm_free(bf8_arg2);
  libxsmm_free(bf8_out);
  libxsmm_free(bf8_eqn_out);

  libxsmm_free(hf8_arg0);
  libxsmm_free(hf8_arg1);
  libxsmm_free(hf8_arg2);
  libxsmm_free(hf8_out);
  libxsmm_free(hf8_eqn_out);

  return ret;
}
