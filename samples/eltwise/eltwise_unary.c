/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define PI 3.14159265358979323846
#define EPS 1.19209290e-04F

#define COPY_OP 0
#define X2_OP 2
#define XOR_OP 3
#define TANH_OP 4
#define SIGMOID_OP 5
#define GELU_OP 6
#define GELU_INV_OP 7
#define TANH_INV_OP 8
#define SIGMOID_INV_OP 9

int unequal_fp32_vals(float a, float b) {
  if (fabs(a-b) < EPS) {
    return 0;
  } else {
    return 1;
  }
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

float fsigmoid(float x) {
  return (tanhf(x/2.0) + 1.0)/2.0;
}

float fsigmoid_inv(float x) {
  return fsigmoid(x) * (1.0-fsigmoid(x));
}

float tanh_inv(float x) {
  return 1.0-tanhf(x)*tanhf(x);
}

float gelu(float x) {
  return (erf(x/sqrtf(2.0)) + 1.0)*0.5*x;
}

float gelu_inv(float x) {
  return (0.5 + 0.5 * erf(x/sqrtf(2.0)) + x/(sqrtf(2.0*PI))*exp(-0.5*x*x) );
}

void unary_op_f32_f32_gold(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, float *in, float *out, unsigned int op) {
  libxsmm_blasint i, j;
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M; ++i ) {
      if ( op == COPY_OP) {
        out[(j*ldo) + i] = in[(j*ldi) + i];
      }
      if (op == X2_OP) {
        out[(j*ldo) + i] = in[(j*ldi) + i] *  in[(j*ldi) + i];
      }
      if (op == XOR_OP) {
        out[(j*ldo) + i] = 0;
      }
      if (op == TANH_OP) {
        out[(j*ldo) + i] = tanhf(in[(j*ldi) + i]);
      }
      if (op == SIGMOID_OP) {
        out[(j*ldo) + i] = fsigmoid(in[(j*ldi) + i]);
      }
      if (op == GELU_OP) {
        out[(j*ldo) + i] = gelu(in[(j*ldi) + i]);
      }
      if (op == GELU_INV_OP) {
        out[(j*ldo) + i] = gelu_inv(in[(j*ldi) + i]);
      }
      if (op == TANH_INV_OP) {
        out[(j*ldo) + i] = tanh_inv(in[(j*ldi) + i]);
      }
      if (op == SIGMOID_INV_OP) {
        out[(j*ldo) + i] = fsigmoid_inv(in[(j*ldi) + i]);
      }
    }
  }
}

void unary_op_bf16_bf16_gold(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, libxsmm_bfloat16 *in, libxsmm_bfloat16 *out, unsigned int op) {
  libxsmm_blasint i, j;
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M; ++i ) {
      union libxsmm_bfloat16_hp bf16_hp;
      float res;
      bf16_hp.i[1] = in[(j*ldi) + i];
      bf16_hp.i[0] = 0;

      if (op == COPY_OP) {
        out[(j*ldo) + i] = in[(j*ldi) + i];
      } else {
        if (op == X2_OP) {
          res = bf16_hp.f * bf16_hp.f;
        }
        if (op == XOR_OP) {
          res = 0;
        }
        if (op == TANH_OP) {
          res = tanhf(bf16_hp.f);
        }
        if (op == SIGMOID_OP) {
          res = fsigmoid(bf16_hp.f);
        }
        if (op == GELU_OP) {
          res = gelu(bf16_hp.f);
        }
        if (op == GELU_INV_OP) {
          res = gelu_inv(bf16_hp.f);
        }
        if (op == TANH_INV_OP) {
          res = tanh_inv(bf16_hp.f);
        }
        if (op == SIGMOID_INV_OP) {
          res = fsigmoid_inv(bf16_hp.f);
        }
        libxsmm_rne_convert_fp32_bf16( &res, &out[(j*ldo) + i], 1 );
      }
    }
  }
}

void unary_op_f32_bf16_gold(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, float *in, libxsmm_bfloat16 *out, unsigned int op) {
  libxsmm_blasint i, j;
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M; ++i ) {
      float res;

      if (op == COPY_OP) {
        res = in[(j*ldi) + i];
      }
      if (op == X2_OP) {
        res = in[(j*ldi) + i] * in[(j*ldi) + i];
      }
      if (op == XOR_OP) {
        res = 0;
      }
      if (op == TANH_OP) {
        res = tanhf(in[(j*ldi) + i]);
      }
      if (op == SIGMOID_OP) {
        res = fsigmoid(in[(j*ldi) + i]);
      }
      if (op == GELU_OP) {
        res = gelu(in[(j*ldi) + i]);
      }
      if (op == GELU_INV_OP) {
        res = gelu_inv(in[(j*ldi) + i]);
      }
      if (op == TANH_INV_OP) {
        res = tanh_inv(in[(j*ldi) + i]);
      }
      if (op == SIGMOID_INV_OP) {
        res = fsigmoid_inv(in[(j*ldi) + i]);
      }
      libxsmm_rne_convert_fp32_bf16( &res, &out[(j*ldo) + i], 1 );
    }
  }
}

void unary_op_bf16_f32_gold(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, libxsmm_bfloat16 *in, float *out, unsigned int op) {
  libxsmm_blasint i, j;
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M; ++i ) {
      union libxsmm_bfloat16_hp bf16_hp;
      float res;
      bf16_hp.i[1] = in[(j*ldi) + i];
      bf16_hp.i[0] = 0;

      if (op == COPY_OP) {
        res = bf16_hp.f;
      }
      if (op == X2_OP) {
        res = bf16_hp.f * bf16_hp.f;
      }
      if (op == XOR_OP) {
        res = 0;
      }
      if (op == TANH_OP) {
        res = tanhf(bf16_hp.f);
      }
      if (op == SIGMOID_OP) {
        res = fsigmoid(bf16_hp.f);
      }
      if (op == GELU_OP) {
        res = gelu(bf16_hp.f);
      }
      if (op == GELU_INV_OP) {
        res = gelu_inv(bf16_hp.f);
      }
      if (op == TANH_INV_OP) {
        res = tanh_inv(bf16_hp.f);
      }
      if (op == SIGMOID_INV_OP) {
        res = fsigmoid_inv(bf16_hp.f);
      }
      out[(j*ldo) + i] = res;
    }
  }
}

void test_unary_op_f32_f32( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned int op ) {
  float *in;
  float *out, *out_gold;
  unsigned int i, j;
  unsigned int s;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_meltw_unary_type  unary_type;
  char opname[256];

  if ( op == COPY_OP ) {
    sprintf(opname, "copy");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
  }
  if ( op == X2_OP ) {
    sprintf(opname, "x2");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_X2;
  }
  if ( op == XOR_OP ) {
    sprintf(opname, "xor");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_XOR;
  }
  if ( op == TANH_OP ) {
    sprintf(opname, "tanh");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_TANH;
  }
  if ( op == SIGMOID_OP ) {
    sprintf(opname, "sigmoid");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_SIGMOID;
  }
  if ( op == GELU_OP ) {
    sprintf(opname, "gelu");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_GELU;
  }
  if ( op == GELU_INV_OP ) {
    sprintf(opname, "gelu_inv");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_GELU_INV;
  }
  if ( op == TANH_INV_OP ) {
    sprintf(opname, "tanh_inv");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_TANH_INV;
  }
  if ( op == SIGMOID_INV_OP ) {
    sprintf(opname, "sigmoid_inv");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV;
  }

  if ( M > ldi ) {
    fprintf( stderr, "test_unary_%s_f32_f32: ldi needs to be equal to or bigger than M\n", opname);
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_unary_%s_f32_f32: ldo needs to be equal to or bigger than N\n", opname);
    exit(-1);
  }

  libxsmm_rng_set_seed(1);

  in        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi,   64);
  out       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  out_gold  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      in[(i*ldi)+j] = (float)libxsmm_rng_f64();
    }
  }

  /* init out */
  for ( i = 0; i < N*ldo; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < N*ldo; ++i ) {
    out_gold[i] = 0;
  }

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    unary_op_f32_f32_gold( M, 1, ldi, ldo, &in[(i*ldi)], &out_gold[(i*ldo)], op );
  }

  /* use jited tranpose */
  unary_param.in_ptr  = (void*)in;
  unary_param.out_ptr = (void*)out;
  unary_param.mask_ptr = NULL;
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;

  libxsmm_meltwfunction_unary unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, unary_type);
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel( &unary_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( unequal_fp32_vals(out_gold[(i*ldo)+j], out[(i*ldo)+j])  ) {
        printf("error at possition i=%i, j=%i, %f, %f\n", i, j, out[(i*ldo)+j], out_gold[(i*ldo)+j]);
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
    printf("SUCCESS output\n");
  } else {
    printf("FAILURE output\n");
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
}

void test_unary_op_bf16_bf16( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned int op ) {
  libxsmm_bfloat16 *in;
  libxsmm_bfloat16 *out, *out_gold;
  unsigned int i, j;
  unsigned int s;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_meltw_unary_type  unary_type;
  libxsmm_dnn_datatype compute_dtype = LIBXSMM_DATATYPE_F32;
  char opname[256];

  if ( op == COPY_OP ) {
    sprintf(opname, "copy");
    compute_dtype = LIBXSMM_DATATYPE_BF16;
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
  }
  if ( op == X2_OP ) {
    sprintf(opname, "x2");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_X2;
  }
  if ( op == XOR_OP ) {
    sprintf(opname, "xor");
    compute_dtype = LIBXSMM_DATATYPE_BF16;
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_XOR;
  }
  if ( op == TANH_OP ) {
    sprintf(opname, "tanh");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_TANH;
  }
  if ( op == SIGMOID_OP ) {
    sprintf(opname, "sigmoid");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_SIGMOID;
  }
  if ( op == GELU_OP ) {
    sprintf(opname, "gelu");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_GELU;
  }
  if ( op == GELU_INV_OP ) {
    sprintf(opname, "gelu_inv");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_GELU_INV;
  }
  if ( op == TANH_INV_OP ) {
    sprintf(opname, "tanh_inv");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_TANH_INV;
  }
  if ( op == SIGMOID_INV_OP ) {
    sprintf(opname, "sigmoid_inv");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV;
  }

  if ( M > ldi ) {
    fprintf( stderr, "test_unary_%s_bf16_bf16: ldi needs to be equal to or bigger than M\n", opname);
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_unary_%s_bf16_bf16: ldo needs to be equal to or bigger than N\n", opname);
    exit(-1);
  }

  libxsmm_rng_set_seed(1);

  in        = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldi,   64);
  out       = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,   64);
  out_gold  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,   64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      union libxsmm_bfloat16_hp bf16_hp;
      bf16_hp.f = (float)libxsmm_rng_f64();
      in[(i*ldi)+j] = bf16_hp.i[1];
    }
  }

  /* init out */
  for ( i = 0; i < N*ldo; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < N*ldo; ++i ) {
    out_gold[i] = 0;
  }

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    unary_op_bf16_bf16_gold( M, 1, ldi, ldo, &in[(i*ldi)], &out_gold[(i*ldo)], op );
  }

  /* use jited tranpose */
  unary_param.in_ptr  = (void*)in;
  unary_param.out_ptr = (void*)out;
  unary_param.mask_ptr = NULL;
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltwfunction_unary unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, compute_dtype, LIBXSMM_DATATYPE_BF16, unary_flags, unary_type);
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel( &unary_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( unequal_bf16_vals(out_gold[(i*ldo)+j], out[(i*ldo)+j]) ) {
        printf("error at possition i=%i, j=%i, %f, %f\n", i, j, (float) out[(i*ldo)+j], (float) out_gold[(i*ldo)+j]);
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
    printf("SUCCESS output\n");
  } else {
    printf("FAILURE output\n");
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
}

void test_unary_op_f32_bf16( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned int op ) {
  float *in;
  libxsmm_bfloat16 *out, *out_gold;
  unsigned int i, j;
  unsigned int s;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_meltw_unary_type  unary_type;
  char opname[256];

  if ( op == COPY_OP ) {
    sprintf(opname, "copy");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
  }
  if ( op == X2_OP ) {
    sprintf(opname, "x2");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_X2;
  }
  if ( op == XOR_OP ) {
    sprintf(opname, "xor");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_XOR;
  }
  if ( op == TANH_OP ) {
    sprintf(opname, "tanh");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_TANH;
  }
  if ( op == SIGMOID_OP ) {
    sprintf(opname, "sigmoid");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_SIGMOID;
  }
  if ( op == GELU_OP ) {
    sprintf(opname, "gelu");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_GELU;
  }
  if ( op == GELU_INV_OP ) {
    sprintf(opname, "gelu_inv");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_GELU_INV;
  }
  if ( op == TANH_INV_OP ) {
    sprintf(opname, "tanh_inv");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_TANH_INV;
  }
  if ( op == SIGMOID_INV_OP ) {
    sprintf(opname, "sigmoid_inv");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV;
  }

  if ( M > ldi ) {
    fprintf( stderr, "test_unary_%s_f32_bf16: ldi needs to be equal to or bigger than M\n", opname);
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_unary_%s_f32_bf16: ldo needs to be equal to or bigger than N\n", opname);
    exit(-1);
  }

  libxsmm_rng_set_seed(1);

  in        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi,   64);
  out       = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,   64);
  out_gold  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,   64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      in[(i*ldi)+j] = (float)libxsmm_rng_f64();
    }
  }

  /* init out */
  for ( i = 0; i < N*ldo; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < N*ldo; ++i ) {
    out_gold[i] = 0;
  }

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    unary_op_f32_bf16_gold( M, 1, ldi, ldo, &in[(i*ldi)], &out_gold[(i*ldo)], op );
  }

  unary_param.in_ptr  = (void*)in;
  unary_param.out_ptr = (void*)out;
  unary_param.mask_ptr = NULL;
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltwfunction_unary unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, unary_flags, unary_type);
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel( &unary_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( unequal_bf16_vals(out_gold[(i*ldo)+j], out[(i*ldo)+j]) ) {
        printf("error at possition i=%i, j=%i, %f, %f\n", i, j, (float)out[(i*ldo)+j], (float)out_gold[(i*ldo)+j]);
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
    printf("SUCCESS output\n");
  } else {
    printf("FAILURE output\n");
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
}

void test_unary_op_bf16_f32( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned int op ) {
  libxsmm_bfloat16 *in;
  float *out, *out_gold;
  unsigned int i, j;
  unsigned int s;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_meltw_unary_type  unary_type;
  char opname[256];

  if ( op == COPY_OP ) {
    sprintf(opname, "copy");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
  }
  if ( op == X2_OP ) {
    sprintf(opname, "x2");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_X2;
  }
  if ( op == XOR_OP ) {
    sprintf(opname, "xor");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_XOR;
  }
  if ( op == TANH_OP ) {
    sprintf(opname, "tanh");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_TANH;
  }
  if ( op == SIGMOID_OP ) {
    sprintf(opname, "sigmoid");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_SIGMOID;
  }
  if ( op == GELU_OP ) {
    sprintf(opname, "gelu");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_GELU;
  }
  if ( op == GELU_INV_OP ) {
    sprintf(opname, "gelu_inv");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_GELU_INV;
  }
  if ( op == TANH_INV_OP ) {
    sprintf(opname, "tanh_inv");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_TANH_INV;
  }
  if ( op == SIGMOID_INV_OP ) {
    sprintf(opname, "sigmoid_inv");
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV;
  }

  if ( M > ldi ) {
    fprintf( stderr, "test_unary_%s_bf16_f32: ldi needs to be equal to or bigger than M\n", opname);
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_unary_%s_bf16_f32: ldo needs to be equal to or bigger than N\n", opname);
    exit(-1);
  }

  libxsmm_rng_set_seed(1);

  in        = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldi,   64);
  out       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  out_gold  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      union libxsmm_bfloat16_hp bf16_hp;
      bf16_hp.f = (float)libxsmm_rng_f64();
      in[(i*ldi)+j] = bf16_hp.i[1];
    }
  }

  /* init out */
  for ( i = 0; i < N*ldo; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < N*ldo; ++i ) {
    out_gold[i] = 0;
  }

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    unary_op_bf16_f32_gold( M, 1, ldi, ldo, &in[(i*ldi)], &out_gold[(i*ldo)], op );
  }

  unary_param.in_ptr  = (void*)in;
  unary_param.out_ptr = (void*)out;
  unary_param.mask_ptr = NULL;
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltwfunction_unary unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, unary_type);
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel( &unary_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( unequal_fp32_vals(out_gold[(i*ldo)+j], out[(i*ldo)+j])  ) {
        printf("error at possition i=%i, j=%i, %f, %f\n", i, j, out[(i*ldo)+j], out_gold[(i*ldo)+j]);
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
    printf("SUCCESS output\n");
  } else {
    printf("FAILURE output\n");
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
}

int main( int argc, char* argv[] ) {
  libxsmm_blasint dtype_in;
  libxsmm_blasint dtype_out;
  libxsmm_blasint dtype_comp;
  unsigned char op;
  libxsmm_blasint bitm;
  libxsmm_blasint M;
  libxsmm_blasint N;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;
  libxsmm_blasint valid_op;
  char opname[256];

  if ( argc != 10 ) {
    printf(" Error! Usage: %s [type] [bitmask: 0/1] [prec_in: 4/2] [compute_prec: 4/2] [prec_out: 4/2] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  op         = atoi(argv[1]);
  bitm       = atoi(argv[2]);
  dtype_in   = atoi(argv[3]);
  dtype_comp = atoi(argv[4]);
  dtype_out  = atoi(argv[5]);
  M          = atoi(argv[6]);
  N          = atoi(argv[7]);
  ldi        = atoi(argv[8]);
  ldo        = atoi(argv[9]);

  if ( op == COPY_OP ) {
    sprintf(opname, "copy");
  }
  if ( op == X2_OP ) {
    sprintf(opname, "x2");
  }
  if ( op == XOR_OP ) {
    sprintf(opname, "xor");
  }
  if ( op == TANH_OP ) {
    sprintf(opname, "tanh");
  }
  if ( op == SIGMOID_OP ) {
    sprintf(opname, "sigmoid");
  }
  if ( op == GELU_OP ) {
    sprintf(opname, "gelu");
  }
  if ( op == GELU_INV_OP ) {
    sprintf(opname, "gelu_inv");
  }
  if ( op == TANH_INV_OP ) {
    sprintf(opname, "tanh_inv");
  }
  if ( op == SIGMOID_INV_OP ) {
    sprintf(opname, "sigmoid_inv");
  }

  valid_op = ( op == COPY_OP || op == X2_OP || op == XOR_OP || op == TANH_OP || op == SIGMOID_OP || op == GELU_OP || op == GELU_INV_OP || op == TANH_INV_OP || op == SIGMOID_INV_OP) ? 1 : 0;

  if ( op == COPY_OP && dtype_in == 4 && dtype_out == 4 && dtype_comp == 4 ) {
    printf("Testing F32 F32 copy\n");
    test_unary_op_f32_f32( M, N, ldi, ldo, op);
  } else if ( op == COPY_OP && dtype_in == 2 && dtype_out == 2 && (dtype_comp == 4 || dtype_comp == 2) ) {
    printf("Testing BF16 BF16 copy\n");
    test_unary_op_bf16_bf16( M, N, ldi, ldo, op);
  } else if ( op == COPY_OP && dtype_in == 4 && dtype_out == 2 && (dtype_comp == 4 || dtype_comp == 2) ) {
    printf("Testing F32 BF16 copy\n");
    test_unary_op_f32_bf16( M, N, ldi, ldo, op);
  } else if ( op == COPY_OP && dtype_in == 2 && dtype_out == 4 && (dtype_comp == 4 || dtype_comp == 2) ) {
    printf("Testing BF16 F32 copy\n");
    test_unary_op_bf16_f32( M, N, ldi, ldo, op);
  } else if ( valid_op > 0 && dtype_in == 4 && dtype_out == 4 && dtype_comp == 4 ) {
    printf("Testing F32 F32 %s\n", opname);
    test_unary_op_f32_f32( M, N, ldi, ldo, op);
  } else if ( valid_op > 0 && dtype_in == 2 && dtype_out == 2 && (dtype_comp == 4 || (dtype_comp == 2 && op == XOR_OP)) ) {
    printf("Testing BF16 BF16 %s\n", opname);
    test_unary_op_bf16_bf16( M, N, ldi, ldo, op);
  } else if ( valid_op > 0 && dtype_in == 4 && dtype_out == 2 && dtype_comp == 4 ) {
    printf("Testing F32 BF16 %s\n", opname);
    test_unary_op_f32_bf16( M, N, ldi, ldo, op);
  } else if ( valid_op > 0 && dtype_in == 2 && dtype_out == 4 && dtype_comp == 4 ) {
    printf("Testing BF16 F32 %s\n", opname);
    test_unary_op_bf16_f32( M, N, ldi, ldo, op);
  } else {
    printf(" Error! Usage: %s [type] [bitmask: 0/1] [prec_in: 4/2] [compute_prec: 4/2] [prec_out: 4/2] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }
}
