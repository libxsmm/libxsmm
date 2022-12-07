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
#include "../eltwise/eltwise_common.h"

#define EPS 1.19209290e-03F

int main( int argc, char* argv[] ) {
  libxsmm_blasint my_eqn0;
  libxsmm_matrix_eqn_function func0;

  libxsmm_blasint S1, S2, S3, s1, s3;
  libxsmm_blasint ld, tmp_ld, tmp_ld2;
  libxsmm_datatype bg_dt, in_dt, out_dt;
  libxsmm_meqn_arg_shape arg_shape_out;
  float *arg0, *arg1, *arg2, *arg3, *arg4, *out, *eqn_out;
  libxsmm_bfloat8 *bf8_arg0, *bf8_arg1, *bf8_arg2, *bf8_arg3, *bf8_arg4, *bf8_out, *bf8_eqn_out;
  libxsmm_matrix_eqn_param eqn_param;
  libxsmm_matrix_arg arg_array[5];
  unsigned int all_correct = 1;
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

  arg0 = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*ld,   64);
  arg1 = (float*) libxsmm_aligned_malloc( sizeof(float)*1*tmp_ld,   64);
  arg2 = (float*) libxsmm_aligned_malloc( sizeof(float)*1*tmp_ld,   64);
  arg3 = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*tmp_ld2,   64);
  arg4 = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*tmp_ld2,   64);
  out  = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*ld,   64);
  eqn_out  = (float*) libxsmm_aligned_malloc( sizeof(float)*S1*ld,   64);

  init_random_matrix( LIBXSMM_DATATYPE_F32, arg0, 1, ld, S1, 0);
  init_random_matrix( LIBXSMM_DATATYPE_F32, arg1, 1, tmp_ld, 1, 0);
  init_random_matrix( LIBXSMM_DATATYPE_F32, arg2, 1, tmp_ld, 1, 0);
  init_random_matrix( LIBXSMM_DATATYPE_F32, arg3, 1, tmp_ld2, S1, 0);
  init_random_matrix( LIBXSMM_DATATYPE_F32, arg4, 1, tmp_ld2, S1, 0);
  init_random_matrix( LIBXSMM_DATATYPE_F32, out, 1, ld, S1, 0);

  bf8_arg0 = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*S1*ld,   64);
  bf8_arg1 = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*1*tmp_ld,   64);
  bf8_arg2 = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*1*tmp_ld,   64);
  bf8_arg3 = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*S1*tmp_ld2,   64);
  bf8_arg4 = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*S1*tmp_ld2,   64);
  bf8_out  = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*S1*ld,   64);
  bf8_eqn_out  = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*S1*ld,   64);

  libxsmm_rne_convert_fp32_bf8( arg0, bf8_arg0, S1*ld );
  libxsmm_rne_convert_fp32_bf8( arg1, bf8_arg1, 1*tmp_ld );
  libxsmm_rne_convert_fp32_bf8( arg2, bf8_arg2, 1*tmp_ld );
  libxsmm_rne_convert_fp32_bf8( arg3, bf8_arg3, S1*tmp_ld2 );
  libxsmm_rne_convert_fp32_bf8( arg4, bf8_arg4, S1*tmp_ld2 );
  libxsmm_rne_convert_fp32_bf8( out, bf8_out, S1*ld );
  libxsmm_rne_convert_fp32_bf8( out, bf8_eqn_out, S1*ld );

  libxsmm_convert_bf8_f32(bf8_arg0, arg0, S1*ld);
  libxsmm_convert_bf8_f32(bf8_arg1, arg1, 1*tmp_ld);
  libxsmm_convert_bf8_f32(bf8_arg2, arg2, 1*tmp_ld);
  libxsmm_convert_bf8_f32(bf8_arg3, arg3, S1*tmp_ld2);
  libxsmm_convert_bf8_f32(bf8_arg4, arg4, S1*tmp_ld2);
  libxsmm_convert_bf8_f32(bf8_out, out, S1*ld);
  libxsmm_convert_bf8_f32(bf8_eqn_out, eqn_out, S1*ld);

  my_eqn0 = libxsmm_matrix_eqn_create();
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn0, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn0, LIBXSMM_MELTW_TYPE_TERNARY_MULADD,
    (libxsmm_meltw_ternary_flags)(LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT),
    LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, S3, S1, ld, 0, 0, in_dt );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, 1, 1, tmp_ld, 1, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, 1, 1, tmp_ld, 2, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, S3, S1, tmp_ld2, 3, 0, bg_dt );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, S3, S1, tmp_ld2, 4, 0, bg_dt );
  arg_shape_out = libxsmm_create_meqn_arg_shape( S3, S1, ld, out_dt );
  func0 = libxsmm_dispatch_matrix_eqn_v2( my_eqn0, arg_shape_out );

  arg_array[0].primary = (void*)bf8_arg0;
  arg_array[1].primary = (void*)arg1;
  arg_array[2].primary = (void*)arg2;
  arg_array[3].primary = (void*)arg3;
  arg_array[4].primary = (void*)arg4;
  eqn_param.inputs = arg_array;
  eqn_param.output.primary = (void*)bf8_eqn_out;

  func0(&eqn_param);

  /* Run reference */
  for (s1 = 0; s1 < S1; s1++) {
    for (s3 = 0; s3 < S3; s3++) {
      out[s1*ld+s3] = (arg0[s1*ld+s3] * arg1[0] + arg2[0]) * arg3[s1*tmp_ld2+s3] + arg4[s1*tmp_ld2+s3];
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

  return 0;
}

