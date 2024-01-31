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
#include "../eltwise/eltwise_common.h"

#define EPS 1.19209290e-03F


int main( int argc, char* argv[] ) {
  libxsmm_blasint my_eqn0;
  libxsmm_meqn_function func0;

  libxsmm_blasint S1, S2, S3, s1, s3;
  libxsmm_blasint ld, tmp_ld, tmp_ld2;
  libxsmm_datatype bg_dt, in_dt, out_dt;
  libxsmm_meqn_arg_metadata arg_metadata;
  libxsmm_meqn_op_metadata  op_metadata;
  libxsmm_meqn_arg_shape          arg_shape_in, arg_shape_out;
  libxsmm_matrix_arg_attributes   arg_singular_attr = libxsmm_create_matrix_arg_attributes( LIBXSMM_MATRIX_ARG_TYPE_SINGULAR, LIBXSMM_MATRIX_ARG_SET_TYPE_NONE, 0, 0);
  float* arg[5] = { NULL }, *out = NULL, *eqn_out = NULL;
  libxsmm_bfloat8* bf8_arg[5] = { NULL }, *bf8_out = NULL, *bf8_eqn_out = NULL;
  libxsmm_meqn_param eqn_param;
  libxsmm_matrix_arg arg_array[5];
  unsigned int all_correct = 1;
  int result = EXIT_SUCCESS;
  LIBXSMM_UNUSED(argc);
  LIBXSMM_UNUSED(argv);

  S1 = 16;
  S2 = 64;
  S3 = 64;
  bg_dt = LIBXSMM_DATATYPE_F32;
  in_dt = LIBXSMM_DATATYPE_BF8;
  out_dt = LIBXSMM_DATATYPE_BF8;
  tmp_ld = 1;
  tmp_ld2 = S3;
  ld = S2 * S3;

  arg[0] = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*ld, 64);
  arg[1] = (float*) libxsmm_aligned_malloc( sizeof(float)*1*tmp_ld, 64);
  arg[2] = (float*) libxsmm_aligned_malloc( sizeof(float)*1*tmp_ld, 64);
  arg[3] = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*tmp_ld2, 64);
  arg[4] = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*tmp_ld2, 64);
  out  = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*ld, 64);
  eqn_out  = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*ld, 64);

  init_random_matrix( LIBXSMM_DATATYPE_F32, arg[0], 1, ld, S1, 0);
  init_random_matrix( LIBXSMM_DATATYPE_F32, arg[1], 1, tmp_ld, 1, 0);
  init_random_matrix( LIBXSMM_DATATYPE_F32, arg[2], 1, tmp_ld, 1, 0);
  init_random_matrix( LIBXSMM_DATATYPE_F32, arg[3], 1, tmp_ld2, S1, 0);
  init_random_matrix( LIBXSMM_DATATYPE_F32, arg[4], 1, tmp_ld2, S1, 0);
  init_random_matrix( LIBXSMM_DATATYPE_F32, out, 1, ld, S1, 0);

  bf8_arg[0] = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*S1*ld, 64);
  bf8_arg[1] = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*1*tmp_ld, 64);
  bf8_arg[2] = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*1*tmp_ld, 64);
  bf8_arg[3] = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*S1*tmp_ld2, 64);
  bf8_arg[4] = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*S1*tmp_ld2, 64);
  bf8_out  = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*S1*ld, 64);
  bf8_eqn_out  = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*S1*ld, 64);

  libxsmm_rne_convert_fp32_bf8( arg[0], bf8_arg[0], S1*ld );
  libxsmm_rne_convert_fp32_bf8( arg[1], bf8_arg[1], 1*tmp_ld );
  libxsmm_rne_convert_fp32_bf8( arg[2], bf8_arg[2], 1*tmp_ld );
  libxsmm_rne_convert_fp32_bf8( arg[3], bf8_arg[3], S1*tmp_ld2 );
  libxsmm_rne_convert_fp32_bf8( arg[4], bf8_arg[4], S1*tmp_ld2 );
  libxsmm_rne_convert_fp32_bf8( out, bf8_out, S1*ld );
  libxsmm_rne_convert_fp32_bf8( out, bf8_eqn_out, S1*ld );

  libxsmm_convert_bf8_f32(bf8_arg[0], arg[0], S1*ld);
  libxsmm_convert_bf8_f32(bf8_arg[1], arg[1], 1*tmp_ld);
  libxsmm_convert_bf8_f32(bf8_arg[2], arg[2], 1*tmp_ld);
  libxsmm_convert_bf8_f32(bf8_arg[3], arg[3], S1*tmp_ld2);
  libxsmm_convert_bf8_f32(bf8_arg[4], arg[4], S1*tmp_ld2);
  libxsmm_convert_bf8_f32(bf8_out, out, S1*ld);
  libxsmm_convert_bf8_f32(bf8_eqn_out, eqn_out, S1*ld);

  my_eqn0 = libxsmm_meqn_create();
  op_metadata   = libxsmm_create_meqn_op_metadata(my_eqn0, -1);
  libxsmm_meqn_push_back_ternary_op( op_metadata, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT );
  libxsmm_meqn_push_back_ternary_op( op_metadata, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_DATATYPE_F32,
    (libxsmm_meltw_ternary_flags)(LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT) );
  arg_shape_in  = libxsmm_create_meqn_arg_shape( S3, S1, ld, in_dt );
  arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn0, 0);
  libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
  arg_shape_in  = libxsmm_create_meqn_arg_shape( 1, 1, tmp_ld, LIBXSMM_DATATYPE_F32 );
  arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn0, 1);
  libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
  arg_shape_in  = libxsmm_create_meqn_arg_shape( 1, 1, tmp_ld, LIBXSMM_DATATYPE_F32 );
  arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn0, 2);
  libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
  arg_shape_in  = libxsmm_create_meqn_arg_shape( S3, S1, tmp_ld2, bg_dt );
  arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn0, 3);
  libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
  arg_shape_in  = libxsmm_create_meqn_arg_shape( S3, S1, tmp_ld2, bg_dt );
  arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn0, 4);
  libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
  arg_shape_out = libxsmm_create_meqn_arg_shape( S3, S1, ld, out_dt );
  func0 = libxsmm_dispatch_meqn( my_eqn0, arg_shape_out );

  arg_array[0].primary = (void*)bf8_arg[0];
  arg_array[1].primary = (void*)arg[1];
  arg_array[2].primary = (void*)arg[2];
  arg_array[3].primary = (void*)arg[3];
  arg_array[4].primary = (void*)arg[4];
  eqn_param.inputs = arg_array;
  eqn_param.output.primary = (void*)bf8_eqn_out;

  if (NULL != func0) {
    func0(&eqn_param);

    /* Run reference */
    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < S3; s3++) {
        out[s1*ld+s3] = (arg[0][s1*ld+s3] * arg[1][0] + arg[2][0]) * arg[3][s1*tmp_ld2+s3] + arg[4][s1*tmp_ld2+s3];
      }
    }

    libxsmm_rne_convert_fp32_bf8( out, bf8_out, S1*ld );


    for (s1 = 0; s1 < S1; s1++) {
      for (s3 = 0; s3 < ld; s3++) {
        if (bf8_out[s1*ld+s3] != bf8_eqn_out[s1*ld+s3]) {
          all_correct = 0;
        }
      }
    }

    if (all_correct == 1) {
      printf("CORRECT equation!!!\n");
    } else {
      printf("FAILED equation!!!\n");
    }
  }
  else {
    fprintf(stderr, "JIT for func0 failed. Bailing...!\n");
    result = -1;
  }

  libxsmm_free(arg[0]);
  libxsmm_free(arg[1]);
  libxsmm_free(arg[2]);
  libxsmm_free(arg[3]);
  libxsmm_free(arg[4]);
  libxsmm_free(out);
  libxsmm_free(eqn_out);

  libxsmm_free(bf8_arg[0]);
  libxsmm_free(bf8_arg[1]);
  libxsmm_free(bf8_arg[2]);
  libxsmm_free(bf8_arg[3]);
  libxsmm_free(bf8_arg[4]);
  libxsmm_free(bf8_out);
  libxsmm_free(bf8_eqn_out);

  return result;
}
