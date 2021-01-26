/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define EPS 1.19209290e-03F

int unequal_fp32_vals(float a, float b) {
  if (fabs(a-b) < EPS) {
    return 0;
  } else {
    return 1;
  }
}

float gelu(float x) {
  return (erf(x/sqrtf(2.0)) + 1.0)*0.5*x;
}

int main( int argc, char* argv[] ) {
  libxsmm_blasint my_eqn0, my_eqn1;
  libxsmm_matrix_eqn_function func0, func1;
  libxsmm_blasint i, j, s;
  libxsmm_matrix_eqn_param eqn_param;

  int M = 32;
  int N = 32;
  int ld = 32;
  float *arg0 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  float *arg1 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  float *arg2 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  float *arg3 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  float *out  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  float *eqn_out  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  float *arg_array[4];

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ld; ++j ) {
      arg0[(i*ld)+j] = (float)libxsmm_rng_f64();
      arg1[(i*ld)+j] = (float)libxsmm_rng_f64();
      arg2[(i*ld)+j] = (float)libxsmm_rng_f64();
      arg3[(i*ld)+j] = (float)libxsmm_rng_f64();
      out[(i*ld)+j]  = (float)libxsmm_rng_f64();
      eqn_out[(i*ld)+j] = out[(i*ld)+j];
    }
  }

  arg_array[0] = arg0;
  arg_array[1] = arg1;
  arg_array[2] = arg2;
  arg_array[3] = arg3;

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      out[(i*ld)+j] = ((float) (arg0[(i*ld)+j] + tanhf(arg1[(i*ld)+j]))) / ((float) ( gelu((float)exp(arg2[(i*ld)+j])) + arg3[(i*ld)+j]));
    }
  }


  my_eqn0 = libxsmm_matrix_eqn_create();
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_DIV, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, 32, 32, 32, 0, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_TANH, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, 32, 32, 32, 1, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_GELU, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_EXP, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, 32, 32, 32, 2, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, 32, 32, 32, 3, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_tree_print( my_eqn0 );
  libxsmm_matrix_eqn_rpn_print( my_eqn0 );
  func0 = libxsmm_dispatch_matrix_eqn( 32, 32, NULL, LIBXSMM_DATATYPE_F32, my_eqn0 );

  eqn_param.in_ptrs = (const void**)arg_array;
  eqn_param.out_ptr = eqn_out;
  func0(&eqn_param);

  /* compare result */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( unequal_fp32_vals(out[(i*ld)+j], eqn_out[(i*ld)+j])  ) {
        printf("error at possition i=%i, j=%i, %f, %f\n", i, j, out[(i*ld)+j], eqn_out[(i*ld)+j]);
        s = 1;
      }
#if 0
      else {
        printf("correct at possition i=%i, j=%i, %f, %f\n", i, j, out[(i*ldo)+j], out_gold[(i*ldo)+j]);
      }
#endif
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS output of eqn 0\n");
  } else {
    printf("FAILURE output of eqn 0\n");
  }

  my_eqn1 = libxsmm_matrix_eqn_create();
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_DIV, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_SUB, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 32, 32, 32, 0, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 32, 32, 32, 1, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 32, 32, 32, 2, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 32, 32, 32, 3, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_SUB, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_DIV, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 32, 32, 32, 4, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 32, 32, 32, 5, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 32, 32, 32, 6, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 32, 32, 32, 7, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_DIV, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 32, 32, 32, 8, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 32, 32, 32, 9, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 32, 32, 32, 10, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_tree_print( my_eqn1 );
  libxsmm_matrix_eqn_rpn_print( my_eqn1 );
  func1 = libxsmm_dispatch_matrix_eqn( 32, 32, NULL, LIBXSMM_DATATYPE_F32, my_eqn1 );

  return 0;
}
