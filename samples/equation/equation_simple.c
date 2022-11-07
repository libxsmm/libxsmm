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
void eqn0_f32f32(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, float *arg0, float *arg1, float *arg2, float*arg3, float *out) {
  libxsmm_blasint i, j;

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      float Arg0, Arg1, Arg2, Arg3;
      Arg0 = arg0[(i*ld)+j];
      Arg1 = arg1[(i*ld)+j];
      Arg2 = arg2[(i*ld)+j];
      Arg3 = arg3[(i*ld)+j];
#if 0
      out[(i*ld)+j] = (Arg0 + 1.0f + Arg1) * (LIBXSMM_TANHF(1.0f/Arg2) + Arg3);
#else
      out[(i*ld)+j] = (Arg0 + 1.0f + Arg1) * ((Arg2*Arg2) + Arg3);
#endif
    }
  }
}

LIBXSMM_INLINE
void eqn0_f64f64(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, double *arg0, double *arg1, double *arg2, double*arg3, double *out) {
  libxsmm_blasint i, j;

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      double Arg0, Arg1, Arg2, Arg3;
      Arg0 = arg0[(i*ld)+j];
      Arg1 = arg1[(i*ld)+j];
      Arg2 = arg2[(i*ld)+j];
      Arg3 = arg3[(i*ld)+j];

#if 0
      out[(i*ld)+j] = (Arg0 + 1.0f + Arg1) * (LIBXSMM_TANHF(1.0f/Arg2) + Arg3);
#else
      out[(i*ld)+j] = (Arg0 + 1.0f + Arg1) * ((Arg2*Arg2) + Arg3);
#endif
    }
  }
}

LIBXSMM_INLINE
void eqn0_bf16bf16(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, libxsmm_bfloat16 *bf16_arg0, libxsmm_bfloat16 *bf16_arg1, libxsmm_bfloat16 *bf16_arg2, libxsmm_bfloat16* bf16_arg3, libxsmm_bfloat16 *bf16_out) {
  libxsmm_blasint i, j;

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      float Arg0, Arg1, Arg2, Arg3, res;
      libxsmm_bfloat16_f32 bf16_hp;
      bf16_hp.i[0] = 0;
      bf16_hp.i[1] = bf16_arg0[(i*ld)+j];
      Arg0 = bf16_hp.f;
      bf16_hp.i[1] = bf16_arg1[(i*ld)+j];
      Arg1 = bf16_hp.f;
      bf16_hp.i[1] = bf16_arg2[(i*ld)+j];
      Arg2 = bf16_hp.f;
      bf16_hp.i[1] = bf16_arg3[(i*ld)+j];
      Arg3 = bf16_hp.f;
#if 0
      res = (Arg0 + 1.0f + Arg1) * (LIBXSMM_TANHF(1.0f/Arg2) + Arg3);
#else
      res = (Arg0 + 1.0f + Arg1) * ((Arg2*Arg2) + Arg3);
#endif
      libxsmm_rne_convert_fp32_bf16( &res, &bf16_out[(i*ld)+j], 1 );
    }
  }
}

LIBXSMM_INLINE
void eqn0_bf16f32(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, libxsmm_bfloat16 *bf16_arg0, libxsmm_bfloat16 *bf16_arg1, libxsmm_bfloat16 *bf16_arg2, libxsmm_bfloat16* bf16_arg3, float *out) {
  libxsmm_blasint i, j;

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      float Arg0, Arg1, Arg2, Arg3;
      libxsmm_bfloat16_f32 bf16_hp;
      bf16_hp.i[0] = 0;
      bf16_hp.i[1] = bf16_arg0[(i*ld)+j];
      Arg0 = bf16_hp.f;
      bf16_hp.i[1] = bf16_arg1[(i*ld)+j];
      Arg1 = bf16_hp.f;
      bf16_hp.i[1] = bf16_arg2[(i*ld)+j];
      Arg2 = bf16_hp.f;
      bf16_hp.i[1] = bf16_arg3[(i*ld)+j];
      Arg3 = bf16_hp.f;
#if 0
      out[(i*ld)+j] = (Arg0 + 1.0f + Arg1) * (LIBXSMM_TANHF(1.0f/Arg2) + Arg3);
#else
      out[(i*ld)+j] = (Arg0 + 1.0f + Arg1) * ((Arg2*Arg2) + Arg3);
#endif
    }
  }
}

LIBXSMM_INLINE
void eqn0_f32bf16(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, float *arg0, float *arg1, float *arg2, float*arg3, libxsmm_bfloat16 *bf16_out) {
  libxsmm_blasint i, j;

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      float Arg0, Arg1, Arg2, Arg3, res;
      Arg0 = arg0[(i*ld)+j];
      Arg1 = arg1[(i*ld)+j];
      Arg2 = arg2[(i*ld)+j];
      Arg3 = arg3[(i*ld)+j];
#if 0
      res = (Arg0 + 1.0f + Arg1) * (LIBXSMM_TANHF(1.0f/Arg2) + Arg3);
#else
      res = (Arg0 + 1.0f + Arg1) * ((Arg2*Arg2) + Arg3);
#endif
      libxsmm_rne_convert_fp32_bf16( &res, &bf16_out[(i*ld)+j], 1 );
    }
  }
}

LIBXSMM_INLINE
void eqn0_f16f16(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, libxsmm_float16 *f16_arg0, libxsmm_float16 *f16_arg1, libxsmm_float16 *f16_arg2, libxsmm_float16* f16_arg3, libxsmm_float16 *f16_out) {
  libxsmm_blasint i, j;

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      float Arg0, Arg1, Arg2, Arg3, res;
      libxsmm_convert_f16_f32( &(f16_arg0[(i*ld)+j]), &Arg0, 1);
      libxsmm_convert_f16_f32( &(f16_arg1[(i*ld)+j]), &Arg1, 1);
      libxsmm_convert_f16_f32( &(f16_arg2[(i*ld)+j]), &Arg2, 1);
      libxsmm_convert_f16_f32( &(f16_arg3[(i*ld)+j]), &Arg3, 1);
#if 0
      res = (Arg0 + 1.0f + Arg1) * (LIBXSMM_TANHF(1.0f/Arg2) + Arg3);
#else
      res = (Arg0 + 1.0f + Arg1) * ((Arg2*Arg2) + Arg3);
#endif
      libxsmm_rne_convert_fp32_f16( &res, &f16_out[(i*ld)+j], 1 );
    }
  }
}

LIBXSMM_INLINE
void eqn0_f16f32(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, libxsmm_float16 *f16_arg0, libxsmm_float16 *f16_arg1, libxsmm_float16 *f16_arg2, libxsmm_float16* f16_arg3, float *out) {
  libxsmm_blasint i, j;

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      float Arg0, Arg1, Arg2, Arg3;
      libxsmm_convert_f16_f32( &(f16_arg0[(i*ld)+j]), &Arg0, 1);
      libxsmm_convert_f16_f32( &(f16_arg1[(i*ld)+j]), &Arg1, 1);
      libxsmm_convert_f16_f32( &(f16_arg2[(i*ld)+j]), &Arg2, 1);
      libxsmm_convert_f16_f32( &(f16_arg3[(i*ld)+j]), &Arg3, 1);
#if 0
      out[(i*ld)+j] = (Arg0 + 1.0f + Arg1) * (LIBXSMM_TANHF(1.0f/Arg2) + Arg3);
#else
      out[(i*ld)+j] = (Arg0 + 1.0f + Arg1) * ((Arg2*Arg2) + Arg3);
#endif
    }
  }
}

LIBXSMM_INLINE
void eqn0_f32f16(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, float *arg0, float *arg1, float *arg2, float*arg3, libxsmm_float16 *f16_out) {
  libxsmm_blasint i, j;

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      float Arg0, Arg1, Arg2, Arg3, res;
      Arg0 = arg0[(i*ld)+j];
      Arg1 = arg1[(i*ld)+j];
      Arg2 = arg2[(i*ld)+j];
      Arg3 = arg3[(i*ld)+j];
#if 0
      res = (Arg0 + 1.0f + Arg1) * (LIBXSMM_TANHF(1.0f/Arg2) + Arg3);
#else
      res = (Arg0 + 1.0f + Arg1) * ((Arg2*Arg2) + Arg3);
#endif
      libxsmm_rne_convert_fp32_f16( &res, &f16_out[(i*ld)+j], 1 );
    }
  }
}

LIBXSMM_INLINE
void eqn0_bf8bf8(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, libxsmm_bfloat8 *bf8_arg0, libxsmm_bfloat8 *bf8_arg1, libxsmm_bfloat8 *bf8_arg2, libxsmm_bfloat8* bf8_arg3, libxsmm_bfloat8 *bf8_out) {
  libxsmm_blasint i, j;

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      float Arg0, Arg1, Arg2, Arg3, res;
      libxsmm_convert_bf8_f32( &(bf8_arg0[(i*ld)+j]), &Arg0, 1);
      libxsmm_convert_bf8_f32( &(bf8_arg1[(i*ld)+j]), &Arg1, 1);
      libxsmm_convert_bf8_f32( &(bf8_arg2[(i*ld)+j]), &Arg2, 1);
      libxsmm_convert_bf8_f32( &(bf8_arg3[(i*ld)+j]), &Arg3, 1);
#if 0
      res = (Arg0 + 1.0f + Arg1) * (LIBXSMM_TANHF(1.0f/Arg2) + Arg3);
#else
      res = (Arg0 + 1.0f + Arg1) * ((Arg2*Arg2) + Arg3);
#endif
      libxsmm_rne_convert_fp32_bf8( &res, &bf8_out[(i*ld)+j], 1 );
    }
  }
}

LIBXSMM_INLINE
void eqn0_bf8f32(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, libxsmm_bfloat8 *bf8_arg0, libxsmm_bfloat8 *bf8_arg1, libxsmm_bfloat8 *bf8_arg2, libxsmm_bfloat8* bf8_arg3, float *out) {
  libxsmm_blasint i, j;

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      float Arg0, Arg1, Arg2, Arg3;
      libxsmm_convert_bf8_f32( &(bf8_arg0[(i*ld)+j]), &Arg0, 1);
      libxsmm_convert_bf8_f32( &(bf8_arg1[(i*ld)+j]), &Arg1, 1);
      libxsmm_convert_bf8_f32( &(bf8_arg2[(i*ld)+j]), &Arg2, 1);
      libxsmm_convert_bf8_f32( &(bf8_arg3[(i*ld)+j]), &Arg3, 1);
#if 0
      out[(i*ld)+j] = (Arg0 + 1.0f + Arg1) * (LIBXSMM_TANHF(1.0f/Arg2) + Arg3);
#else
      out[(i*ld)+j] = (Arg0 + 1.0f + Arg1) * ((Arg2*Arg2) + Arg3);
#endif
    }
  }
}

LIBXSMM_INLINE
void eqn0_f32bf8(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, float *arg0, float *arg1, float *arg2, float*arg3, libxsmm_bfloat8 *bf8_out) {
  libxsmm_blasint i, j;

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      float Arg0, Arg1, Arg2, Arg3, res;
      Arg0 = arg0[(i*ld)+j];
      Arg1 = arg1[(i*ld)+j];
      Arg2 = arg2[(i*ld)+j];
      Arg3 = arg3[(i*ld)+j];
#if 0
      res = (Arg0 + 1.0f + Arg1) * (LIBXSMM_TANHF(1.0f/Arg2) + Arg3);
#else
      res = (Arg0 + 1.0f + Arg1) * ((Arg2*Arg2) + Arg3);
#endif
      libxsmm_rne_convert_fp32_bf8( &res, &bf8_out[(i*ld)+j], 1 );
    }
  }
}

LIBXSMM_INLINE
void eqn0_hf8hf8(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, libxsmm_hfloat8 *hf8_arg0, libxsmm_hfloat8 *hf8_arg1, libxsmm_hfloat8 *hf8_arg2, libxsmm_hfloat8* hf8_arg3, libxsmm_hfloat8 *hf8_out) {
  libxsmm_blasint i, j;

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      float Arg0, Arg1, Arg2, Arg3, res;
      libxsmm_convert_hf8_f32( &(hf8_arg0[(i*ld)+j]), &Arg0, 1);
      libxsmm_convert_hf8_f32( &(hf8_arg1[(i*ld)+j]), &Arg1, 1);
      libxsmm_convert_hf8_f32( &(hf8_arg2[(i*ld)+j]), &Arg2, 1);
      libxsmm_convert_hf8_f32( &(hf8_arg3[(i*ld)+j]), &Arg3, 1);
#if 0
      res = (Arg0 + 1.0f + Arg1) * (LIBXSMM_TANHF(1.0f/Arg2) + Arg3);
#else
      res = (Arg0 + 1.0f + Arg1) * ((Arg2*Arg2) + Arg3);
#endif
      libxsmm_rne_convert_fp32_hf8( &res, &hf8_out[(i*ld)+j], 1 );
    }
  }
}

LIBXSMM_INLINE
void eqn0_hf8f32(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, libxsmm_hfloat8 *hf8_arg0, libxsmm_hfloat8 *hf8_arg1, libxsmm_hfloat8 *hf8_arg2, libxsmm_hfloat8* hf8_arg3, float *out) {
  libxsmm_blasint i, j;

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      float Arg0, Arg1, Arg2, Arg3;
      libxsmm_convert_hf8_f32( &(hf8_arg0[(i*ld)+j]), &Arg0, 1);
      libxsmm_convert_hf8_f32( &(hf8_arg1[(i*ld)+j]), &Arg1, 1);
      libxsmm_convert_hf8_f32( &(hf8_arg2[(i*ld)+j]), &Arg2, 1);
      libxsmm_convert_hf8_f32( &(hf8_arg3[(i*ld)+j]), &Arg3, 1);
#if 0
      out[(i*ld)+j] = (Arg0 + 1.0f + Arg1) * (LIBXSMM_TANHF(1.0f/Arg2) + Arg3);
#else
      out[(i*ld)+j] = (Arg0 + 1.0f + Arg1) * ((Arg2*Arg2) + Arg3);
#endif
    }
  }
}

LIBXSMM_INLINE
void eqn0_f32hf8(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, float *arg0, float *arg1, float *arg2, float*arg3, libxsmm_hfloat8 *hf8_out) {
  libxsmm_blasint i, j;

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      float Arg0, Arg1, Arg2, Arg3, res;
      Arg0 = arg0[(i*ld)+j];
      Arg1 = arg1[(i*ld)+j];
      Arg2 = arg2[(i*ld)+j];
      Arg3 = arg3[(i*ld)+j];
#if 0
      res = (Arg0 + 1.0f + Arg1) * (LIBXSMM_TANHF(1.0f/Arg2) + Arg3);
#else
      res = (Arg0 + 1.0f + Arg1) * ((Arg2*Arg2) + Arg3);
#endif
      libxsmm_rne_convert_fp32_hf8( &res, &hf8_out[(i*ld)+j], 1 );
    }
  }
}

int main( int argc, char* argv[] ) {
  int ret = EXIT_SUCCESS;
  double error_bound = 0.00001;
  libxsmm_blasint my_eqn0;
  libxsmm_matrix_eqn_function func0;
  libxsmm_blasint i, j, s, it;
  libxsmm_matrix_eqn_param eqn_param;
  unsigned long long l_start, l_end;
  double l_total = 0, l_total2 = 0;
  libxsmm_matdiff_info norms_out;
  float *arg0, *arg1, *arg2, *arg3, *out, *eqn_out;
  libxsmm_matrix_arg arg_array[4];
  double *f64_arg0, *f64_arg1, *f64_arg2, *f64_arg3, *f64_out, *f64_eqn_out;
  libxsmm_bfloat16 *bf16_arg0, *bf16_arg1, *bf16_arg2, *bf16_arg3, *bf16_out, *bf16_eqn_out;
  libxsmm_float16 *f16_arg0, *f16_arg1, *f16_arg2, *f16_arg3, *f16_out, *f16_eqn_out;
  libxsmm_bfloat8 *bf8_arg0, *bf8_arg1, *bf8_arg2, *bf8_arg3, *bf8_out, *bf8_eqn_out;
  libxsmm_hfloat8 *hf8_arg0, *hf8_arg1, *hf8_arg2, *hf8_arg3, *hf8_out, *hf8_eqn_out;
  libxsmm_matrix_arg f64_arg_array[4];
  libxsmm_matrix_arg bf16_arg_array[4];
  libxsmm_matrix_arg f16_arg_array[4];
  libxsmm_matrix_arg bf8_arg_array[4];
  libxsmm_matrix_arg hf8_arg_array[4];
  libxsmm_meqn_arg_shape arg_shape_out;

  int M = 64;
  int N = 64;
  int ld = 64;
  int iters = 100;
  int datatype_mode = 0;
  libxsmm_datatype  in_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype  out_dt = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype  compute_dt = LIBXSMM_DATATYPE_F32;

  if ( argc > 1 ) M = atoi(argv[1]);
  if ( argc > 2 ) N = atoi(argv[2]);
  if ( argc > 3 ) ld = atoi(argv[3]);
  if ( argc > 4 ) datatype_mode = atoi(argv[4]);
  if ( argc > 5 ) iters = atoi(argv[5]);

  arg0 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  arg1 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  arg2 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  arg3 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  out  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  eqn_out  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);

  f64_arg0 = (double*) libxsmm_aligned_malloc( sizeof(double)*N*ld,   64);
  f64_arg1 = (double*) libxsmm_aligned_malloc( sizeof(double)*N*ld,   64);
  f64_arg2 = (double*) libxsmm_aligned_malloc( sizeof(double)*N*ld,   64);
  f64_arg3 = (double*) libxsmm_aligned_malloc( sizeof(double)*N*ld,   64);
  f64_out  = (double*) libxsmm_aligned_malloc( sizeof(double)*N*ld,   64);
  f64_eqn_out  = (double*) libxsmm_aligned_malloc( sizeof(double)*N*ld,   64);

  bf16_arg0 = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);
  bf16_arg1 = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);
  bf16_arg2 = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);
  bf16_arg3 = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);
  bf16_out  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);
  bf16_eqn_out  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld,   64);

  f16_arg0 = (libxsmm_float16*) libxsmm_aligned_malloc( sizeof(libxsmm_float16)*N*ld,   64);
  f16_arg1 = (libxsmm_float16*) libxsmm_aligned_malloc( sizeof(libxsmm_float16)*N*ld,   64);
  f16_arg2 = (libxsmm_float16*) libxsmm_aligned_malloc( sizeof(libxsmm_float16)*N*ld,   64);
  f16_arg3 = (libxsmm_float16*) libxsmm_aligned_malloc( sizeof(libxsmm_float16)*N*ld,   64);
  f16_out  = (libxsmm_float16*) libxsmm_aligned_malloc( sizeof(libxsmm_float16)*N*ld,   64);
  f16_eqn_out  = (libxsmm_float16*) libxsmm_aligned_malloc( sizeof(libxsmm_float16)*N*ld,   64);

  bf8_arg0 = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*N*ld,   64);
  bf8_arg1 = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*N*ld,   64);
  bf8_arg2 = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*N*ld,   64);
  bf8_arg3 = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*N*ld,   64);
  bf8_out  = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*N*ld,   64);
  bf8_eqn_out  = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*N*ld,   64);

  hf8_arg0 = (libxsmm_hfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_hfloat8)*N*ld,   64);
  hf8_arg1 = (libxsmm_hfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_hfloat8)*N*ld,   64);
  hf8_arg2 = (libxsmm_hfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_hfloat8)*N*ld,   64);
  hf8_arg3 = (libxsmm_hfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_hfloat8)*N*ld,   64);
  hf8_out  = (libxsmm_hfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_hfloat8)*N*ld,   64);
  hf8_eqn_out  = (libxsmm_hfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_hfloat8)*N*ld,   64);

  libxsmm_init();
  libxsmm_matdiff_clear(&norms_out);
  set_in_out_compute_dt(datatype_mode, &in_dt, &out_dt, &compute_dt);

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ld; ++j ) {
      f64_arg0[(i*ld)+j] = libxsmm_rng_f64();
      f64_arg1[(i*ld)+j] = libxsmm_rng_f64();
      f64_arg2[(i*ld)+j] = libxsmm_rng_f64();
      f64_arg3[(i*ld)+j] = libxsmm_rng_f64();
      f64_out[(i*ld)+j]  = libxsmm_rng_f64();
      f64_eqn_out[(i*ld)+j] = f64_out[(i*ld)+j];
      arg0[(i*ld)+j] = (float)f64_arg0[(i*ld)+j];
      arg1[(i*ld)+j] = (float)f64_arg1[(i*ld)+j];
      arg2[(i*ld)+j] = (float)f64_arg2[(i*ld)+j];
      arg3[(i*ld)+j] = (float)f64_arg3[(i*ld)+j];
      out[(i*ld)+j]  = (float)f64_out[(i*ld)+j];
      eqn_out[(i*ld)+j] = out[(i*ld)+j];
      libxsmm_rne_convert_fp32_bf16( &arg0[(i*ld)+j], &bf16_arg0[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf16( &arg1[(i*ld)+j], &bf16_arg1[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf16( &arg2[(i*ld)+j], &bf16_arg2[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf16( &arg3[(i*ld)+j], &bf16_arg3[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf16( &out[(i*ld)+j], &bf16_out[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf16( &eqn_out[(i*ld)+j], &bf16_eqn_out[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_f16( &arg0[(i*ld)+j], &f16_arg0[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_f16( &arg1[(i*ld)+j], &f16_arg1[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_f16( &arg2[(i*ld)+j], &f16_arg2[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_f16( &arg3[(i*ld)+j], &f16_arg3[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_f16( &out[(i*ld)+j], &f16_out[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_f16( &eqn_out[(i*ld)+j], &f16_eqn_out[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf8( &arg0[(i*ld)+j], &bf8_arg0[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf8( &arg1[(i*ld)+j], &bf8_arg1[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf8( &arg2[(i*ld)+j], &bf8_arg2[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf8( &arg3[(i*ld)+j], &bf8_arg3[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf8( &out[(i*ld)+j], &bf8_out[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf8( &eqn_out[(i*ld)+j], &bf8_eqn_out[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_hf8( &arg0[(i*ld)+j], &hf8_arg0[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_hf8( &arg1[(i*ld)+j], &hf8_arg1[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_hf8( &arg2[(i*ld)+j], &hf8_arg2[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_hf8( &arg3[(i*ld)+j], &hf8_arg3[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_hf8( &out[(i*ld)+j], &hf8_out[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_hf8( &eqn_out[(i*ld)+j], &hf8_eqn_out[(i*ld)+j], 1 );
    }
  }

  arg_array[0].primary = arg0;
  arg_array[1].primary = arg1;
  arg_array[2].primary = arg2;
  arg_array[3].primary = arg3;

  f64_arg_array[0].primary = f64_arg0;
  f64_arg_array[1].primary = f64_arg1;
  f64_arg_array[2].primary = f64_arg2;
  f64_arg_array[3].primary = f64_arg3;

  bf16_arg_array[0].primary = bf16_arg0;
  bf16_arg_array[1].primary = bf16_arg1;
  bf16_arg_array[2].primary = bf16_arg2;
  bf16_arg_array[3].primary = bf16_arg3;

  f16_arg_array[0].primary = f16_arg0;
  f16_arg_array[1].primary = f16_arg1;
  f16_arg_array[2].primary = f16_arg2;
  f16_arg_array[3].primary = f16_arg3;

  bf8_arg_array[0].primary = bf8_arg0;
  bf8_arg_array[1].primary = bf8_arg1;
  bf8_arg_array[2].primary = bf8_arg2;
  bf8_arg_array[3].primary = bf8_arg3;

  hf8_arg_array[0].primary = hf8_arg0;
  hf8_arg_array[1].primary = hf8_arg1;
  hf8_arg_array[2].primary = hf8_arg2;
  hf8_arg_array[3].primary = hf8_arg3;

  if (datatype_mode == 0) {
    eqn0_f32f32(M, N, ld, arg0, arg1, arg2, arg3, out);
  } else if (datatype_mode == 1) {
    eqn0_bf16bf16(M, N, ld, bf16_arg0, bf16_arg1, bf16_arg2, bf16_arg3, bf16_out);
  } else if (datatype_mode == 2) {
    eqn0_f32bf16(M, N, ld, arg0, arg1, arg2, arg3, bf16_out);
  } else if (datatype_mode == 3) {
    eqn0_bf16f32(M, N, ld, bf16_arg0, bf16_arg1, bf16_arg2, bf16_arg3, out);
  } else if (datatype_mode == 4) {
    eqn0_bf8bf8(M, N, ld, bf8_arg0, bf8_arg1, bf8_arg2, bf8_arg3, bf8_out);
  } else if (datatype_mode == 5) {
    eqn0_f32bf8(M, N, ld, arg0, arg1, arg2, arg3, bf8_out);
  } else if (datatype_mode == 6) {
    eqn0_bf8f32(M, N, ld, bf8_arg0, bf8_arg1, bf8_arg2, bf8_arg3, out);
  } else if (datatype_mode == 7) {
    eqn0_f16f16(M, N, ld, f16_arg0, f16_arg1, f16_arg2, f16_arg3, f16_out);
  } else if (datatype_mode == 8) {
    eqn0_f32f16(M, N, ld, arg0, arg1, arg2, arg3, f16_out);
  } else if (datatype_mode == 9) {
    eqn0_f16f32(M, N, ld, f16_arg0, f16_arg1, f16_arg2, f16_arg3, out);
  } else if (datatype_mode == 10) {
    eqn0_hf8hf8(M, N, ld, hf8_arg0, hf8_arg1, hf8_arg2, hf8_arg3, hf8_out);
  } else if (datatype_mode == 11) {
    eqn0_f32hf8(M, N, ld, arg0, arg1, arg2, arg3, hf8_out);
  } else if (datatype_mode == 12) {
    eqn0_hf8f32(M, N, ld, hf8_arg0, hf8_arg1, hf8_arg2, hf8_arg3, out);
  } else if (datatype_mode == 13) {
    eqn0_f64f64(M, N, ld, f64_arg0, f64_arg1, f64_arg2, f64_arg3, f64_out);
  }

  my_eqn0 = libxsmm_matrix_eqn_create();
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, compute_dt );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, compute_dt );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, M, N, ld, 0, 0, in_dt );
  libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_INC, LIBXSMM_MELTW_FLAG_UNARY_NONE, compute_dt );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, M, N, ld, 1, 0, in_dt );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, compute_dt );
  libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_X2, LIBXSMM_MELTW_FLAG_UNARY_NONE, compute_dt );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, M, N, ld, 2, 0, in_dt );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, M, N, ld, 3, 0, in_dt );
  libxsmm_matrix_eqn_tree_print( my_eqn0 );
  libxsmm_matrix_eqn_rpn_print( my_eqn0 );
  arg_shape_out = libxsmm_create_meqn_arg_shape( M, N, ld, out_dt );
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
  } else if ( in_dt == LIBXSMM_DATATYPE_F64  ) {
    eqn_param.inputs = f64_arg_array;
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
  } else if ( out_dt == LIBXSMM_DATATYPE_F64  ) {
    eqn_param.output.primary  = f64_eqn_out;
  }
  func0(&eqn_param);

  /* compare result */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ld; ++j ) {
      if (out_dt == LIBXSMM_DATATYPE_F32) {
        if ( unequal_fp32_vals(out[(i*ld)+j], eqn_out[(i*ld)+j])  ) {
          /*printf("error at possition i=%i, j=%i, %f, %f\n", i, j, out[(i*ld)+j], eqn_out[(i*ld)+j]);*/
          s = 1;
        }
      } else if (out_dt == LIBXSMM_DATATYPE_F64) {
        if ( unequal_fp64_vals(f64_out[(i*ld)+j], f64_eqn_out[(i*ld)+j])  ) {
          /*printf("error at possition i=%i, j=%i, %f, %f\n", i, j, f64_out[(i*ld)+j], f64_eqn_out[(i*ld)+j]);*/
          s = 1;
        }
      } else if (out_dt == LIBXSMM_DATATYPE_BF16) {
        out[(i*ld)+j] = upconvert_bf16(bf16_out[(i*ld)+j]);
        eqn_out[(i*ld)+j] = upconvert_bf16(bf16_eqn_out[(i*ld)+j]);
        if ( unequal_bf16_vals(bf16_out[(i*ld)+j], bf16_eqn_out[(i*ld)+j])  ) {
          /*printf("error at possition i=%i, j=%i, %f, %f\n", i, j, upconvert_bf16(bf16_out[(i*ld)+j]), upconvert_bf16(bf16_eqn_out[(i*ld)+j]));*/
          s = 1;
        }
      } else if (out_dt == LIBXSMM_DATATYPE_F16) {
        libxsmm_convert_f16_f32(&(f16_out[(i*ld)+j]), &(out[(i*ld)+j]), 1);
        libxsmm_convert_f16_f32(&(f16_eqn_out[(i*ld)+j]), &(eqn_out[(i*ld)+j]), 1);
        if ( unequal_f16_vals(f16_out[(i*ld)+j], f16_eqn_out[(i*ld)+j])  ) {
          /*printf("error at possition i=%i, j=%i, %f, %f\n", i, j, upconvert_bf16(bf16_out[(i*ld)+j]), upconvert_bf16(bf16_eqn_out[(i*ld)+j]));*/
          s = 1;
        }
      } else if (out_dt == LIBXSMM_DATATYPE_BF8) {
        libxsmm_convert_bf8_f32(&(bf8_out[(i*ld)+j]), &(out[(i*ld)+j]), 1);
        libxsmm_convert_bf8_f32(&(bf8_eqn_out[(i*ld)+j]), &(eqn_out[(i*ld)+j]), 1);
        if ( unequal_bf8_vals(bf8_out[(i*ld)+j], bf8_eqn_out[(i*ld)+j])  ) {
          /*printf("error at possition i=%i, j=%i, %f, %f\n", i, j, upconvert_bf16(bf16_out[(i*ld)+j]), upconvert_bf16(bf16_eqn_out[(i*ld)+j]));*/
          s = 1;
        }
      } else if (out_dt == LIBXSMM_DATATYPE_HF8) {
        libxsmm_convert_hf8_f32(&(hf8_out[(i*ld)+j]), &(out[(i*ld)+j]), 1);
        libxsmm_convert_hf8_f32(&(hf8_eqn_out[(i*ld)+j]), &(eqn_out[(i*ld)+j]), 1);
        if ( unequal_hf8_vals(hf8_out[(i*ld)+j], hf8_eqn_out[(i*ld)+j])  ) {
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

  print_dt_info(datatype_mode);

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
  if (datatype_mode == 13)  {
    libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F64, ld*N, 1, f64_out, f64_eqn_out, 0, 0);
  } else {
    libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ld*N, 1, out, eqn_out, 0, 0);
  }
#endif
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

  /* Now benchmarking the equations */
  if (iters > 0) {
    if (datatype_mode == 0) {
      eqn0_f32f32(M, N, ld, arg0, arg1, arg2, arg3, out);
    } else if (datatype_mode == 1) {
      eqn0_bf16bf16(M, N, ld, bf16_arg0, bf16_arg1, bf16_arg2, bf16_arg3, bf16_out);
    } else if (datatype_mode == 2) {
      eqn0_f32bf16(M, N, ld, arg0, arg1, arg2, arg3, bf16_out);
    } else if (datatype_mode == 3) {
      eqn0_bf16f32(M, N, ld, bf16_arg0, bf16_arg1, bf16_arg2, bf16_arg3, out);
    } else if (datatype_mode == 4) {
      eqn0_bf8bf8(M, N, ld, bf8_arg0, bf8_arg1, bf8_arg2, bf8_arg3, bf8_out);
    } else if (datatype_mode == 5) {
      eqn0_f32bf8(M, N, ld, arg0, arg1, arg2, arg3, bf8_out);
    } else if (datatype_mode == 6) {
      eqn0_bf8f32(M, N, ld, bf8_arg0, bf8_arg1, bf8_arg2, bf8_arg3, out);
    } else if (datatype_mode == 7) {
      eqn0_f16f16(M, N, ld, f16_arg0, f16_arg1, f16_arg2, f16_arg3, f16_out);
    } else if (datatype_mode == 8) {
      eqn0_f32f16(M, N, ld, arg0, arg1, arg2, arg3, f16_out);
    } else if (datatype_mode == 9) {
      eqn0_f16f32(M, N, ld, f16_arg0, f16_arg1, f16_arg2, f16_arg3, out);
    } else if (datatype_mode == 10) {
      eqn0_hf8hf8(M, N, ld, hf8_arg0, hf8_arg1, hf8_arg2, hf8_arg3, hf8_out);
    } else if (datatype_mode == 11) {
      eqn0_f32hf8(M, N, ld, arg0, arg1, arg2, arg3, hf8_out);
    } else if (datatype_mode == 12) {
      eqn0_hf8f32(M, N, ld, hf8_arg0, hf8_arg1, hf8_arg2, hf8_arg3, out);
    } else if (datatype_mode == 13) {
      eqn0_f64f64(M, N, ld, f64_arg0, f64_arg1, f64_arg2, f64_arg3, f64_out);
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
      } else if (datatype_mode == 4) {
        eqn0_bf8bf8(M, N, ld, bf8_arg0, bf8_arg1, bf8_arg2, bf8_arg3, bf8_out);
      } else if (datatype_mode == 5) {
        eqn0_f32bf8(M, N, ld, arg0, arg1, arg2, arg3, bf8_out);
      } else if (datatype_mode == 6) {
        eqn0_bf8f32(M, N, ld, bf8_arg0, bf8_arg1, bf8_arg2, bf8_arg3, out);
      } else if (datatype_mode == 7) {
        eqn0_f16f16(M, N, ld, f16_arg0, f16_arg1, f16_arg2, f16_arg3, f16_out);
      } else if (datatype_mode == 8) {
        eqn0_f32f16(M, N, ld, arg0, arg1, arg2, arg3, f16_out);
      } else if (datatype_mode == 9) {
        eqn0_f16f32(M, N, ld, f16_arg0, f16_arg1, f16_arg2, f16_arg3, out);
      } else if (datatype_mode == 10) {
        eqn0_hf8hf8(M, N, ld, hf8_arg0, hf8_arg1, hf8_arg2, hf8_arg3, hf8_out);
      } else if (datatype_mode == 11) {
        eqn0_f32hf8(M, N, ld, arg0, arg1, arg2, arg3, hf8_out);
      } else if (datatype_mode == 12) {
        eqn0_hf8f32(M, N, ld, hf8_arg0, hf8_arg1, hf8_arg2, hf8_arg3, out);
      } else if (datatype_mode == 13) {
        eqn0_f64f64(M, N, ld, f64_arg0, f64_arg1, f64_arg2, f64_arg3, f64_out);
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
  }

  libxsmm_free(arg0);
  libxsmm_free(arg1);
  libxsmm_free(arg2);
  libxsmm_free(arg3);
  libxsmm_free(out);
  libxsmm_free(eqn_out);

  libxsmm_free(bf16_arg0);
  libxsmm_free(bf16_arg1);
  libxsmm_free(bf16_arg2);
  libxsmm_free(bf16_arg3);
  libxsmm_free(bf16_out);
  libxsmm_free(bf16_eqn_out);

  libxsmm_free(f16_arg0);
  libxsmm_free(f16_arg1);
  libxsmm_free(f16_arg2);
  libxsmm_free(f16_arg3);
  libxsmm_free(f16_out);
  libxsmm_free(f16_eqn_out);

  libxsmm_free(bf8_arg0);
  libxsmm_free(bf8_arg1);
  libxsmm_free(bf8_arg2);
  libxsmm_free(bf8_arg3);
  libxsmm_free(bf8_out);
  libxsmm_free(bf8_eqn_out);

  libxsmm_free(hf8_arg0);
  libxsmm_free(hf8_arg1);
  libxsmm_free(hf8_arg2);
  libxsmm_free(hf8_arg3);
  libxsmm_free(hf8_out);
  libxsmm_free(hf8_eqn_out);

  return ret;
}
