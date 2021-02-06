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

float upconvert_bf16(libxsmm_bfloat16 x) {
  union libxsmm_bfloat16_hp bf16_hp;
  bf16_hp.i[1] = x;
  bf16_hp.i[0] = 0;
  return bf16_hp.f;
}

int unequal_bf16_vals(libxsmm_bfloat16 a, libxsmm_bfloat16 b) {
  union libxsmm_bfloat16_hp bf16_hp, bf16_hp2;
  bf16_hp.i[1] = a;
  bf16_hp.i[0] = 0;
  bf16_hp2.i[1] = b;
  bf16_hp2.i[0] = 0;
  if (fabs(bf16_hp.f - bf16_hp2.f) < EPS) {
    return 0;
  } else {
    return 1;
  }
}

float gelu(float x) {
  return (erf(x/sqrtf(2.0)) + 1.0)*0.5*x;
}

void eqn0_f32f32(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, float *arg0, float *arg1, float *arg2, float*arg3, float *out) {
  libxsmm_blasint i, j;

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      float Arg0, Arg1, Arg2, Arg3;
      Arg0 = arg0[(i*ld)+j];
      Arg1 = arg1[(i*ld)+j];
      Arg2 = arg2[(i*ld)+j];
      Arg3 = arg3[(i*ld)+j];

      out[(i*ld)+j]  = (float) ((float)Arg0 + (float)((float)1.0 + Arg1)) * (float) ((float)((float)tanhf((float)1.0/((float)Arg2))) + (float)Arg3);
    }
  }
}

void eqn0_bf16bf16(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, libxsmm_bfloat16 *bf16_arg0, libxsmm_bfloat16 *bf16_arg1, libxsmm_bfloat16 *bf16_arg2, libxsmm_bfloat16* bf16_arg3, libxsmm_bfloat16 *bf16_out) {
  libxsmm_blasint i, j;

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      float Arg0, Arg1, Arg2, Arg3, res;
      union libxsmm_bfloat16_hp bf16_hp;
      bf16_hp.i[0] = 0;
      bf16_hp.i[1] = bf16_arg0[(i*ld)+j];
      Arg0 = bf16_hp.f;
      bf16_hp.i[1] = bf16_arg1[(i*ld)+j];
      Arg1 = bf16_hp.f;
      bf16_hp.i[1] = bf16_arg2[(i*ld)+j];
      Arg2 = bf16_hp.f;
      bf16_hp.i[1] = bf16_arg3[(i*ld)+j];
      Arg3 = bf16_hp.f;

      res = (Arg0 + (1.0 + Arg1)) * (tanhf(1.0/Arg2) + Arg3);
      libxsmm_rne_convert_fp32_bf16( &res, &bf16_out[(i*ld)+j], 1 );
    }
  }
}

void eqn0_bf16f32(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, libxsmm_bfloat16 *bf16_arg0, libxsmm_bfloat16 *bf16_arg1, libxsmm_bfloat16 *bf16_arg2, libxsmm_bfloat16* bf16_arg3, float *out) {
  libxsmm_blasint i, j;

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      float Arg0, Arg1, Arg2, Arg3;
      union libxsmm_bfloat16_hp bf16_hp;
      bf16_hp.i[0] = 0;
      bf16_hp.i[1] = bf16_arg0[(i*ld)+j];
      Arg0 = bf16_hp.f;
      bf16_hp.i[1] = bf16_arg1[(i*ld)+j];
      Arg1 = bf16_hp.f;
      bf16_hp.i[1] = bf16_arg2[(i*ld)+j];
      Arg2 = bf16_hp.f;
      bf16_hp.i[1] = bf16_arg3[(i*ld)+j];
      Arg3 = bf16_hp.f;
      out[(i*ld)+j] = (Arg0 + (1.0 + Arg1)) * (tanhf(1.0/Arg2) + Arg3);
    }
  }
}

void eqn0_f32bf16(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, float *arg0, float *arg1, float *arg2, float*arg3, libxsmm_bfloat16 *bf16_out) {
  libxsmm_blasint i, j;

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      float Arg0, Arg1, Arg2, Arg3, res;
      Arg0 = arg0[(i*ld)+j];
      Arg1 = arg1[(i*ld)+j];
      Arg2 = arg2[(i*ld)+j];
      Arg3 = arg3[(i*ld)+j];
      res = (Arg0 + (1.0 + Arg1)) * (tanhf(1.0/Arg2) + Arg3);
      libxsmm_rne_convert_fp32_bf16( &res, &bf16_out[(i*ld)+j], 1 );
    }
  }
}

int main( int argc, char* argv[] ) {
  libxsmm_blasint my_eqn0; /*, my_eqn1*/;
  libxsmm_matrix_eqn_function func0; /*, func1;*/
  libxsmm_blasint i, j, s, it;
  libxsmm_matrix_eqn_param eqn_param;
  unsigned long long l_start, l_end;
  double l_total = 0, l_total2 = 0;
  libxsmm_matdiff_info norms_out;
  float *arg0, *arg1, *arg2, *arg3, *out, *eqn_out, *arg_array[4];
  libxsmm_bfloat16 *bf16_arg0, *bf16_arg1, *bf16_arg2, *bf16_arg3, *bf16_out, *bf16_eqn_out, *bf16_arg_array[4];
  int M = 64;
  int N = 64;
  int ld = 64;
  int iters = 100;
  int datatype_mode = 0;
  libxsmm_datatype  in_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype  out_dt = LIBXSMM_DATATYPE_F32;

  if ( argc > 1 ) M = atoi(argv[1]);
  if ( argc > 2 ) N = atoi(argv[2]);
  if ( argc > 3 ) ld = atoi(argv[3]);
  if ( argc > 4 ) datatype_mode = atoi(argv[4]);
  if ( argc > 5 ) iters = atoi(argv[5]);

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
    in_dt = LIBXSMM_DATATYPE_BF16;;
    out_dt = LIBXSMM_DATATYPE_F32;
  }

  arg0 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  arg1 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  arg2 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  arg3 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  out  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  eqn_out  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);

  bf16_arg0 = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);
  bf16_arg1 = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);
  bf16_arg2 = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);
  bf16_arg3 = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);
  bf16_out  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);
  bf16_eqn_out  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);

  libxsmm_init();
  libxsmm_matdiff_clear(&norms_out);

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ld; ++j ) {
      arg0[(i*ld)+j] = (float)libxsmm_rng_f64();
      arg1[(i*ld)+j] = (float)libxsmm_rng_f64();
      arg2[(i*ld)+j] = (float)libxsmm_rng_f64();
      arg3[(i*ld)+j] = (float)libxsmm_rng_f64();
      out[(i*ld)+j]  = (float)libxsmm_rng_f64();
      eqn_out[(i*ld)+j] = out[(i*ld)+j];
      libxsmm_rne_convert_fp32_bf16( &arg0[(i*ld)+j], &bf16_arg0[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf16( &arg1[(i*ld)+j], &bf16_arg1[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf16( &arg2[(i*ld)+j], &bf16_arg2[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf16( &arg3[(i*ld)+j], &bf16_arg3[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf16( &out[(i*ld)+j], &bf16_out[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf16( &eqn_out[(i*ld)+j], &bf16_eqn_out[(i*ld)+j], 1 );
    }
  }

  arg_array[0] = arg0;
  arg_array[1] = arg1;
  arg_array[2] = arg2;
  arg_array[3] = arg3;

  bf16_arg_array[0] = bf16_arg0;
  bf16_arg_array[1] = bf16_arg1;
  bf16_arg_array[2] = bf16_arg2;
  bf16_arg_array[3] = bf16_arg3;


#if 0
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      out[(i*ld)+j] = ((float) (arg0[(i*ld)+j] + tanhf(arg1[(i*ld)+j]))) / ((float) ( gelu((float)exp(arg2[(i*ld)+j])) + arg3[(i*ld)+j]));
    }
  }

  my_eqn0 = libxsmm_matrix_eqn_create();
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_DIV, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, 64, 64, 64, 0, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_TANH, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, 64, 64, 64, 1, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_GELU, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_EXP, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, 64, 64, 64, 2, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, 64, 64, 64, 3, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_tree_print( my_eqn0 );
  libxsmm_matrix_eqn_rpn_print( my_eqn0 );
  func0 = libxsmm_dispatch_matrix_eqn( 64, 64, NULL, LIBXSMM_DATATYPE_F32, my_eqn0 );

  eqn_param.in_ptrs = (const void**)arg_array;
  eqn_param.out_ptr = eqn_out;
  func0(&eqn_param);
#else

  if (datatype_mode == 0) {
    eqn0_f32f32(M, N, ld, arg0, arg1, arg2, arg3, out);
  } else if (datatype_mode == 1) {
    eqn0_bf16bf16(M, N, ld, bf16_arg0, bf16_arg1, bf16_arg2, bf16_arg3, bf16_out);
  } else if (datatype_mode == 2) {
    eqn0_f32bf16(M, N, ld, arg0, arg1, arg2, arg3, bf16_out);
  } else if (datatype_mode == 3) {
    eqn0_bf16f32(M, N, ld, bf16_arg0, bf16_arg1, bf16_arg2, bf16_arg3, out);
  }

  my_eqn0 = libxsmm_matrix_eqn_create();
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, M, N, ld, 0, 0, in_dt );
  libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_INC, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, M, N, ld, 1, 0, in_dt );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_TANH, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, M, N, ld, 2, 0, in_dt );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, M, N, ld, 3, 0, in_dt );
  libxsmm_matrix_eqn_tree_print( my_eqn0 );
  libxsmm_matrix_eqn_rpn_print( my_eqn0 );
  func0 = libxsmm_dispatch_matrix_eqn( M, N, &ld, out_dt, my_eqn0 );

  if ( in_dt == LIBXSMM_DATATYPE_F32 ) {
    eqn_param.in_ptrs = (const void**)arg_array;
  } else {
    eqn_param.in_ptrs = (const void**)bf16_arg_array;
  }
  if ( out_dt == LIBXSMM_DATATYPE_F32 ) {
    eqn_param.out_ptr = eqn_out;
  } else {
   eqn_param.out_ptr = bf16_eqn_out;
  }
  func0(&eqn_param);
#endif

  /* compare result */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ld; ++j ) {
      if (out_dt == LIBXSMM_DATATYPE_F32) {
        if ( unequal_fp32_vals(out[(i*ld)+j], eqn_out[(i*ld)+j])  ) {
          /*printf("error at possition i=%i, j=%i, %f, %f\n", i, j, out[(i*ld)+j], eqn_out[(i*ld)+j]);*/
          s = 1;
        }
      } else if (out_dt == LIBXSMM_DATATYPE_BF16) {
        out[(i*ld)+j] = upconvert_bf16(bf16_out[(i*ld)+j]);
        eqn_out[(i*ld)+j] = upconvert_bf16(bf16_eqn_out[(i*ld)+j]);
        if ( unequal_bf16_vals(bf16_out[(i*ld)+j], bf16_eqn_out[(i*ld)+j])  ) {
          /*printf("error at possition i=%i, j=%i, %f, %f\n", i, j, upconvert_bf16(bf16_out[(i*ld)+j]), upconvert_bf16(bf16_eqn_out[(i*ld)+j]));*/
          s = 1;
        }
      }
#if 0
      else {
        printf("correct at possition i=%i, j=%i, %f, %f\n", i, j, out[(i*ldo)+j], out_gold[(i*ldo)+j]);
      }
#endif
    }
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
  if ( s == 0 ) {
    /*printf("SUCCESS\n");*/
  } else {
    /*printf("FAILURE\n");*/
  }

  /* compare */
  printf("##########################################\n");
  printf("#   Correctness  - Output                #\n");
  printf("##########################################\n");
#if 0
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ld_in*n, 1, sout_ref, sout, 0, 0);
#else
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ld*N, 1, out, eqn_out, 0, 0);
#endif
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  /* Now benchmarking the equations */

  if (datatype_mode == 0) {
    eqn0_f32f32(M, N, ld, arg0, arg1, arg2, arg3, out);
  } else if (datatype_mode == 1) {
    eqn0_bf16bf16(M, N, ld, bf16_arg0, bf16_arg1, bf16_arg2, bf16_arg3, bf16_out);
  } else if (datatype_mode == 2) {
    eqn0_f32bf16(M, N, ld, arg0, arg1, arg2, arg3, bf16_out);
  } else if (datatype_mode == 3) {
    eqn0_bf16f32(M, N, ld, bf16_arg0, bf16_arg1, bf16_arg2, bf16_arg3, out);
  }


  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
    if (datatype_mode == 0) {
      eqn0_f32f32(M, N, ld, arg0, arg1, arg2, arg3, out);
    } else if (datatype_mode == 1) {
      eqn0_bf16bf16(M, N, ld, bf16_arg0, bf16_arg1, bf16_arg2, bf16_arg3, bf16_out);
    } else if (datatype_mode == 2) {
      eqn0_f32bf16(M, N, ld, arg0, arg1, arg2, arg3, bf16_out);
    } else if (datatype_mode == 3) {
      eqn0_bf16f32(M, N, ld, bf16_arg0, bf16_arg1, bf16_arg2, bf16_arg3, out);
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

  printf("Speedup is %.5g\n", l_total/l_total2);

#if 0
  my_eqn1 = libxsmm_matrix_eqn_create();
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_DIV, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_SUB, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 64, 64, 64, 0, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 64, 64, 64, 1, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 64, 64, 64, 2, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 64, 64, 64, 3, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_SUB, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_DIV, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 64, 64, 64, 4, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 64, 64, 64, 5, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 64, 64, 64, 6, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 64, 64, 64, 7, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_DIV, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 64, 64, 64, 8, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 64, 64, 64, 9, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn1, 64, 64, 64, 10, 0, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_tree_print( my_eqn1 );
  libxsmm_matrix_eqn_rpn_print( my_eqn1 );
  func1 = libxsmm_dispatch_matrix_eqn( 64, 64, NULL, LIBXSMM_DATATYPE_F32, my_eqn1 );
#endif

  return 0;
}
