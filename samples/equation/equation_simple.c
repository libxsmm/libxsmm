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
#define EXPANSION_FACTOR 8

#define EPS 1.19209290e-03F

void create_unique_random_array(unsigned long long *inout_array, int n) {
  if (n > 1)
  {
    int i;
    for (i = 0; i < n; i++) {
      inout_array[i] = i;
    }
    for (i = 0; i < n - 1; i++) {
      int j = i + rand() / (RAND_MAX / (n - i) + 1);
      unsigned long long t = inout_array[j];
      inout_array[j] = inout_array[i];
      inout_array[i] = t;
    }
  }
}

int unequal_fp32_vals(float a, float b) {
  if (fabs(a-b) < EPS) {
    return 0;
  } else {
    return 1;
  }
}

float upconvert_bf16(libxsmm_bfloat16 x) {
  libxsmm_bfloat16_f32 bf16_hp;
  bf16_hp.i[1] = x;
  bf16_hp.i[0] = 0;
  return bf16_hp.f;
}

int unequal_bf16_vals(libxsmm_bfloat16 a, libxsmm_bfloat16 b) {
  libxsmm_bfloat16_f32 bf16_hp, bf16_hp2;
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

int unequal_f16_vals(libxsmm_float16 a, libxsmm_float16 b) {
  float af, bf;
  libxsmm_convert_f16_f32( &a, &af, 1);
  libxsmm_convert_f16_f32( &b, &bf, 1);

  if (fabs(af - bf) < EPS) {
    return 0;
  } else {
    return 1;
  }
}

int unequal_bf8_vals(libxsmm_bfloat8 a, libxsmm_bfloat8 b) {
  float af, bf;
  libxsmm_convert_bf8_f32( &a, &af, 1);
  libxsmm_convert_bf8_f32( &b, &bf, 1);

  if (fabs(af - bf) < EPS) {
    return 0;
  } else {
    return 1;
  }
}

int unequal_hf8_vals(libxsmm_hfloat8 a, libxsmm_hfloat8 b) {
  float af, bf;
  libxsmm_convert_hf8_f32( &a, &af, 1);
  libxsmm_convert_hf8_f32( &b, &bf, 1);

  if (fabs(af - bf) < EPS) {
    return 0;
  } else {
    return 1;
  }
}

float gelu(float x) {
  return (LIBXSMM_ERFF(x/LIBXSMM_SQRTF(2.0f)) + 1.0f)*0.5f*x;
}

void eqn2_f32f32(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ld, float *arg0, float *arg1, float *arg2, unsigned char* relu_mask, float *out) {
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
  libxsmm_blasint my_eqn0, my_eqn1, my_eqn2;
  libxsmm_matrix_eqn_function func0, func1, func2;
  libxsmm_blasint i, j, s, it;
  libxsmm_matrix_eqn_param eqn_param;
  unsigned long long l_start, l_end;
  double l_total = 0, l_total2 = 0;
  libxsmm_matdiff_info norms_out;
  float *arg0, *arg1, *arg2, *arg3, *out, *eqn_out;
  libxsmm_matrix_arg arg_array[4];
  libxsmm_bfloat16 *bf16_arg0, *bf16_arg1, *bf16_arg2, *bf16_arg3, *bf16_out, *bf16_eqn_out;
  libxsmm_float16 *f16_arg0, *f16_arg1, *f16_arg2, *f16_arg3, *f16_out, *f16_eqn_out;
  libxsmm_bfloat8 *bf8_arg0, *bf8_arg1, *bf8_arg2, *bf8_arg3, *bf8_out, *bf8_eqn_out;
  libxsmm_hfloat8 *hf8_arg0, *hf8_arg1, *hf8_arg2, *hf8_arg3, *hf8_out, *hf8_eqn_out;
  libxsmm_matrix_arg bf16_arg_array[4];
  libxsmm_matrix_arg f16_arg_array[4];
  libxsmm_matrix_arg bf8_arg_array[4];
  libxsmm_matrix_arg hf8_arg_array[4];
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
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;

  int test_relu_eq = 0;

  if ( argc > 1 ) M = atoi(argv[1]);
  if ( argc > 2 ) N = atoi(argv[2]);
  if ( argc > 3 ) ld = atoi(argv[3]);
  if ( argc > 4 ) datatype_mode = atoi(argv[4]);
  if ( argc > 5 ) iters = atoi(argv[5]);
  if ( argc > 6 ) idx_type = atoi(argv[6]);
  if ( argc > 7 ) test_relu_eq = atoi(argv[7]);

  large_N = EXPANSION_FACTOR * N;

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
  } else if (datatype_mode == 4) {
    in_dt = LIBXSMM_DATATYPE_BF8;
    out_dt = LIBXSMM_DATATYPE_BF8;
  } else if (datatype_mode == 5) {
    in_dt = LIBXSMM_DATATYPE_F32;
    out_dt = LIBXSMM_DATATYPE_BF8;
  } else if (datatype_mode == 6) {
    in_dt = LIBXSMM_DATATYPE_BF8;;
    out_dt = LIBXSMM_DATATYPE_F32;
  } else if (datatype_mode == 7) {
    in_dt = LIBXSMM_DATATYPE_F16;
    out_dt = LIBXSMM_DATATYPE_F16;
  } else if (datatype_mode == 8) {
    in_dt = LIBXSMM_DATATYPE_F32;
    out_dt = LIBXSMM_DATATYPE_F16;
  } else if (datatype_mode == 9) {
    in_dt = LIBXSMM_DATATYPE_F16;
    out_dt = LIBXSMM_DATATYPE_F32;
  } else if (datatype_mode == 10) {
    in_dt = LIBXSMM_DATATYPE_HF8;
    out_dt = LIBXSMM_DATATYPE_HF8;
  } else if (datatype_mode == 11) {
    in_dt = LIBXSMM_DATATYPE_F32;
    out_dt = LIBXSMM_DATATYPE_HF8;
  } else if (datatype_mode == 12) {
    in_dt = LIBXSMM_DATATYPE_HF8;
    out_dt = LIBXSMM_DATATYPE_F32;
  }

  arg0 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  arg1 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  arg2 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  arg3 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  out  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  eqn_out  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ld,   64);
  cols_ind_array      = (unsigned int*) libxsmm_aligned_malloc( sizeof(unsigned int)*N,   64);
  cols_ind_array_64b  = (unsigned long long*) libxsmm_aligned_malloc( sizeof(unsigned long long)*N,   64);
  unique_random_array = (unsigned long long*) libxsmm_aligned_malloc( sizeof(unsigned long long)*large_N,   64);
  large_input = (float*) libxsmm_aligned_malloc( sizeof(float)*large_N*ld,   64);
  large_input_bf16 = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*large_N*ld,   64);
  large_input_f16 = (libxsmm_float16*) libxsmm_aligned_malloc( sizeof(libxsmm_float16)*large_N*ld,   64);
  large_input_bf8 = (libxsmm_bfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat8)*large_N*ld,   64);
  large_input_hf8 = (libxsmm_hfloat8*) libxsmm_aligned_malloc( sizeof(libxsmm_hfloat8)*large_N*ld,   64);
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

  for ( i = 0; i < large_N; ++i ) {
    for ( j = 0; j < ld; ++j ) {
      large_input[(i*ld)+j] = (float)libxsmm_rng_f64();
      libxsmm_rne_convert_fp32_bf16( &large_input[(i*ld)+j], &large_input_bf16[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_f16( &large_input[(i*ld)+j], &large_input_f16[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_bf8( &large_input[(i*ld)+j], &large_input_bf8[(i*ld)+j], 1 );
      libxsmm_rne_convert_fp32_hf8( &large_input[(i*ld)+j], &large_input_hf8[(i*ld)+j], 1 );
    }
  }

  arg_array[0].primary = arg0;
  arg_array[1].primary = arg1;
  arg_array[2].primary = arg2;
  arg_array[3].primary = arg3;

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
#if 0
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      out[(i*ld)+j] = ((float) (arg0[(i*ld)+j] + LIBXSMM_TANHF(arg1[(i*ld)+j]))) / ((float) ( gelu((float)exp(arg2[(i*ld)+j])) + arg3[(i*ld)+j]));
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
  }

  my_eqn0 = libxsmm_matrix_eqn_create();
#if 0
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
#else
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, M, N, ld, 0, 0, in_dt );
  libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_INC, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, M, N, ld, 1, 0, in_dt );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_X2, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, M, N, ld, 2, 0, in_dt );
  libxsmm_matrix_eqn_push_back_arg( my_eqn0, M, N, ld, 3, 0, in_dt );
#endif
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

  if (datatype_mode == 0) {
    printf("Equation IN: F32, OUT: F32 \n");
  } else if (datatype_mode == 1) {
    printf("Equation IN: BF16, OUT: BF16 \n");
  } else if (datatype_mode == 2) {
    printf("Equation IN: F32, OUT: BF16 \n");
  } else if (datatype_mode == 3) {
    printf("Equation IN: BF16, OUT: F32 \n");
  } else if (datatype_mode == 4) {
    printf("Equation IN: BF8, OUT: BF8 \n");
  } else if (datatype_mode == 5) {
    printf("Equation IN: F32, OUT: BF8 \n");
  } else if (datatype_mode == 6) {
    printf("Equation IN: BF8, OUT: F32 \n");
  } else if (datatype_mode == 7) {
    printf("Equation IN: F16, OUT: F16 \n");
  } else if (datatype_mode == 8) {
    printf("Equation IN: F32, OUT: F16 \n");
  } else if (datatype_mode == 9) {
    printf("Equation IN: F16, OUT: F32 \n");
  } else if (datatype_mode == 10) {
    printf("Equation IN: HF8, OUT: HF8 \n");
  } else if (datatype_mode == 11) {
    printf("Equation IN: F32, OUT: HF8 \n");
  } else if (datatype_mode == 12) {
    printf("Equation IN: HF8, OUT: F32 \n");
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

  if (datatype_mode == 0 || datatype_mode == 1) {
    /* Now we test a gather-reduce equation */
    create_unique_random_array(unique_random_array, large_N);
    for (i = 0; i < N; i++) {
      cols_ind_array_64b[i] = (unsigned long long) unique_random_array[i];
      cols_ind_array[i] = (unsigned int) cols_ind_array_64b[i];
    }
    my_eqn1       = libxsmm_matrix_eqn_create();
    arg_metadata  = libxsmm_create_matrix_eqn_arg_metadata(my_eqn1, 0);
    op_metadata   = libxsmm_create_matrix_eqn_op_metadata(my_eqn1, -1);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( M, N, ld, in_dt );
    arg_shape_out = libxsmm_create_meqn_arg_shape( M, 1, ld, out_dt);
    unary_flags   = (idx_type == 0) ? LIBXSMM_MELTW_FLAG_UNARY_GS_COLS | LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES : LIBXSMM_MELTW_FLAG_UNARY_GS_COLS | LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES;

    libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, in_dt, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS);
    libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata, LIBXSMM_MELTW_TYPE_UNARY_GATHER, in_dt, unary_flags);
    libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata, arg_shape_in, arg_singular_attr);
    func1 = libxsmm_dispatch_matrix_eqn_v2( my_eqn1, arg_shape_out );

    if (datatype_mode == 0) {
      arg_array[0].primary   = large_input;
      if (idx_type == 0) {
        arg_array[0].secondary = cols_ind_array;
      } else {
        arg_array[0].secondary = cols_ind_array_64b;
      }
      eqn_param.output.primary = eqn_out;
    } else if (datatype_mode == 1) {
      bf16_arg_array[0].primary   = large_input_bf16;
      if (idx_type == 0) {
        bf16_arg_array[0].secondary = cols_ind_array;
      } else {
        bf16_arg_array[0].secondary = cols_ind_array_64b;
      }
      eqn_param.output.primary = bf16_eqn_out;
    }

    func1(&eqn_param);
    eqn1_f32f32(M, N, ld, large_input, cols_ind_array, out);

    /* compare */
    printf("\n\n##########################################\n");
    printf("#   Correctness  GATHER-REDUCE- Output   #\n");
    printf("##########################################\n");
    if (datatype_mode == 1) {
      for (i = 0; i < M; i++) {
        eqn_out[i] = upconvert_bf16(bf16_eqn_out[i]);
      }
    }
    libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ld, 1, out, eqn_out, 0, 0);
    printf("L1 reference  : %.25g\n", norms_out.l1_ref);
    printf("L1 test       : %.25g\n", norms_out.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

    eqn1_f32f32(M, N, ld, large_input, cols_ind_array, out);
    l_start = libxsmm_timer_tick();
    for (it = 0; it < iters; it++) {
      eqn1_f32f32(M, N, ld, large_input, cols_ind_array, out);
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

    printf("Speedup is %.5g\n", l_total/l_total2);
  }

  if (test_relu_eq > 0) {
    libxsmm_blasint mask_ld = ((ld+15)-((ld+15)%16))/8;
    unsigned char *mask_ref = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld,   64);
    unsigned char *mask_eqn = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld,   64);
    memset(mask_ref, 0, N * mask_ld * sizeof(unsigned char));
    memset(mask_eqn, 0, N * mask_ld * sizeof(unsigned char));
    s = 0;

    my_eqn2       = libxsmm_matrix_eqn_create();
    op_metadata   = libxsmm_create_matrix_eqn_op_metadata(my_eqn2, -1);
    arg_shape_in  = libxsmm_create_meqn_arg_shape( M, N, ld, in_dt );
    arg_shape_out = libxsmm_create_meqn_arg_shape( M, N, ld, out_dt);
    libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata, LIBXSMM_MELTW_TYPE_UNARY_RELU, out_dt, LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT);
    if (datatype_mode == 1) {
      libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, out_dt, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    }
    libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE);
    libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata, LIBXSMM_MELTW_TYPE_UNARY_GELU, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata, LIBXSMM_MELTW_TYPE_BINARY_SUB, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE);
    arg_metadata  = libxsmm_create_matrix_eqn_arg_metadata(my_eqn2, 0);
    libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_metadata  = libxsmm_create_matrix_eqn_arg_metadata(my_eqn2, 1);
    libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata, arg_shape_in, arg_singular_attr);
    arg_metadata  = libxsmm_create_matrix_eqn_arg_metadata(my_eqn2, 2);
    libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata, arg_shape_in, arg_singular_attr);
    func2 = libxsmm_dispatch_matrix_eqn_v2( my_eqn2, arg_shape_out );

    if (datatype_mode == 0) {
      arg_array[0].primary   = arg0;
      arg_array[1].primary   = arg1;
      arg_array[2].primary   = arg2;
      eqn_param.output.primary   = eqn_out;
      eqn_param.output.secondary = mask_eqn;
    } else if (datatype_mode == 1) {
      bf16_arg_array[0].primary   = bf16_arg0;
      bf16_arg_array[1].primary   = bf16_arg1;
      bf16_arg_array[2].primary   = bf16_arg2;
      eqn_param.output.primary   = bf16_eqn_out;
      eqn_param.output.secondary = mask_eqn;
    }
    func2(&eqn_param);
    eqn2_f32f32(M, N, ld, arg0, arg1, arg2, mask_ref, out);

    /* compare */
    printf("\n\n##########################################\n");
    printf("#   Correctness RELU Equation - Output   #\n");
    printf("##########################################\n");
    if (datatype_mode == 1) {
      for (i = 0; i < N*ld; i++) {
        eqn_out[i] = upconvert_bf16(bf16_eqn_out[i]);
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
    }
    printf("##########################################\n");
    printf("#   Correctness RELU Equation - MASK   #\n");
    printf("##########################################\n");
    libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_I32, (mask_ld*N)/4, 1, mask_ref, mask_eqn, 0, 0);
    printf("L1 reference  : %.25g\n", norms_out.l1_ref);
    printf("L1 test       : %.25g\n", norms_out.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);
  }

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
