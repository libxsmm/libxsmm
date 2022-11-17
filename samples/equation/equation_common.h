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

#define EPS 1.19209290e-03F
#define DEPS 1.19209290e-06F

LIBXSMM_INLINE
void print_dt_info(int datatype_mode) {
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
  } else if (datatype_mode == 13) {
    printf("Equation IN: F64, OUT: F64 \n");
  }
}

LIBXSMM_INLINE
void set_in_out_compute_dt(int datatype_mode, libxsmm_datatype *res_in_dt, libxsmm_datatype *res_out_dt, libxsmm_datatype *res_compute_dt) {
  libxsmm_datatype in_dt = LIBXSMM_DATATYPE_F32, out_dt = LIBXSMM_DATATYPE_F32, compute_dt = LIBXSMM_DATATYPE_F32;
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
  } else if (datatype_mode == 4) {
    in_dt = LIBXSMM_DATATYPE_BF8;
    out_dt = LIBXSMM_DATATYPE_BF8;
  } else if (datatype_mode == 5) {
    in_dt = LIBXSMM_DATATYPE_F32;
    out_dt = LIBXSMM_DATATYPE_BF8;
  } else if (datatype_mode == 6) {
    in_dt = LIBXSMM_DATATYPE_BF8;
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
  } else if (datatype_mode == 13) {
    in_dt = LIBXSMM_DATATYPE_F64;
    out_dt = LIBXSMM_DATATYPE_F64;
    compute_dt = LIBXSMM_DATATYPE_F64;
  }
  *res_in_dt = in_dt;
  *res_out_dt = out_dt;
  *res_compute_dt = compute_dt;
}

LIBXSMM_INLINE
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

LIBXSMM_INLINE
int unequal_fp64_vals(double a, double b) {
  if (fabs(a-b) < DEPS) {
    return 0;
  } else {
    return 1;
  }
}

LIBXSMM_INLINE
int unequal_fp32_vals(float a, float b) {
  if (LIBXSMM_FABSF(a-b) < EPS) {
    return 0;
  } else {
    return 1;
  }
}

LIBXSMM_INLINE
float upconvert_bf16(libxsmm_bfloat16 x) {
  libxsmm_bfloat16_f32 bf16_hp;
  bf16_hp.i[1] = x;
  bf16_hp.i[0] = 0;
  return bf16_hp.f;
}

LIBXSMM_INLINE
float upconvert_bf8(libxsmm_bfloat8 x) {
  libxsmm_bfloat8 _x = x;
  float res;
  libxsmm_convert_bf8_f32( &_x, &res, 1 );
  return res;
}

LIBXSMM_INLINE
float upconvert_hf8(libxsmm_hfloat8 x) {
  libxsmm_hfloat8 _x = x;
  float res;
  libxsmm_convert_hf8_f32( &_x, &res, 1 );
  return res;
}

LIBXSMM_INLINE
float upconvert_f16(libxsmm_float16 x) {
  libxsmm_float16 _x = x;
  float res;
  libxsmm_convert_f16_f32( &_x, &res, 1 );
  return res;
}

LIBXSMM_INLINE
int unequal_bf16_vals(libxsmm_bfloat16 a, libxsmm_bfloat16 b) {
  libxsmm_bfloat16_f32 bf16_hp, bf16_hp2;
  bf16_hp.i[1] = a;
  bf16_hp.i[0] = 0;
  bf16_hp2.i[1] = b;
  bf16_hp2.i[0] = 0;
  if (LIBXSMM_FABSF(bf16_hp.f - bf16_hp2.f) < EPS) {
    return 0;
  } else {
    return 1;
  }
}

LIBXSMM_INLINE
int unequal_f16_vals(libxsmm_float16 a, libxsmm_float16 b) {
  float af, bf;
  libxsmm_convert_f16_f32( &a, &af, 1);
  libxsmm_convert_f16_f32( &b, &bf, 1);

  if (LIBXSMM_FABSF(af - bf) < EPS) {
    return 0;
  } else {
    return 1;
  }
}

LIBXSMM_INLINE
int unequal_bf8_vals(libxsmm_bfloat8 a, libxsmm_bfloat8 b) {
  float af, bf;
  libxsmm_convert_bf8_f32( &a, &af, 1);
  libxsmm_convert_bf8_f32( &b, &bf, 1);

  if (LIBXSMM_FABSF(af - bf) < EPS) {
    return 0;
  } else {
    return 1;
  }
}

LIBXSMM_INLINE
int unequal_hf8_vals(libxsmm_hfloat8 a, libxsmm_hfloat8 b) {
  float af, bf;
  libxsmm_convert_hf8_f32( &a, &af, 1);
  libxsmm_convert_hf8_f32( &b, &bf, 1);

  if (LIBXSMM_FABSF(af - bf) < EPS) {
    return 0;
  } else {
    return 1;
  }
}

LIBXSMM_INLINE
float gelu(float x) {
  return (LIBXSMM_ERFF(x/LIBXSMM_SQRTF(2.0f)) + 1.0f)*0.5f*x;
}


