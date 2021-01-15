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
#define EPS 1.19209290e-03F

#define NO_BCAST 0
#define ROW_BCAST 1
#define COL_BCAST 2
#define SCALAR_BCAST 3

#define COPY_OP 0
#define X2_OP 2
#define XOR_OP 3
#define TANH_OP 4
#define SIGMOID_OP 5
#define GELU_OP 6
#define GELU_INV_OP 7
#define TANH_INV_OP 8
#define SIGMOID_INV_OP 9
#define SQRT_OP 10
#define NEGATE_OP 11
#define INC_OP 12
#define RCP_OP 13
#define RCP_SQRT_OP 14

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

float fp32_unary_compute(float in, unsigned int op) {
  float res;
  if ( op == COPY_OP) {
    res = in;
  }
  if ( op == NEGATE_OP) {
    res = -1.0 * in;
  }
  if (op == X2_OP) {
    res = in * in;
  }
  if (op == XOR_OP) {
    res = 0;
  }
  if (op == TANH_OP) {
    res = tanhf(in);
  }
  if (op == SIGMOID_OP) {
    res = fsigmoid(in);
  }
  if (op == GELU_OP) {
    res = gelu(in);
  }
  if (op == GELU_INV_OP) {
    res = gelu_inv(in);
  }
  if (op == TANH_INV_OP) {
    res = tanh_inv(in);
  }
  if (op == SIGMOID_INV_OP) {
    res = fsigmoid_inv(in);
  }
  if (op == SQRT_OP) {
    res = sqrtf(in);
  }
  if (op == INC_OP) {
    res = in + 1.0;
  }
  if (op == RCP_OP) {
    res = 1.0/in;
  }
  if (op == RCP_SQRT_OP) {
    res = 1.0/sqrtf(in);
  }
  return res;
}

void set_opname(unsigned int op, char *opname) {
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
  if ( op == SQRT_OP ) {
    sprintf(opname, "sqrt");
  }
  if ( op == NEGATE_OP ) {
    sprintf(opname, "negate");
  }
  if (op == INC_OP) {
    sprintf(opname, "inc");
  }
  if (op == RCP_OP) {
    sprintf(opname, "reciprocal");
  }
  if (op == RCP_SQRT_OP) {
    sprintf(opname, "reciprocal sqrt");
  }
}

void set_unarytype(unsigned int op, libxsmm_meltw_unary_type *type) {
  libxsmm_meltw_unary_type  unary_type;

  if ( op == COPY_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
  }
  if ( op == X2_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_X2;
  }
  if ( op == XOR_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_XOR;
  }
  if ( op == TANH_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_TANH;
  }
  if ( op == SIGMOID_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_SIGMOID;
  }
  if ( op == GELU_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_GELU;
  }
  if ( op == GELU_INV_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_GELU_INV;
  }
  if ( op == TANH_INV_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_TANH_INV;
  }
  if ( op == SIGMOID_INV_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV;
  }
  if ( op == SQRT_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_SQRT;
  }
  if ( op == NEGATE_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_NEGATE;
  }
  if (op == INC_OP) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_INC;
  }
  if (op == RCP_OP) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL;
   }
  if (op == RCP_SQRT_OP) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT;
  }
  *type = unary_type;
}

void unary_op_f32_f32_gold(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, float *in, float *out, unsigned int op) {
  libxsmm_blasint i, j;
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M; ++i ) {
      out[(j*ldo) + i] = fp32_unary_compute(in[(j*ldi) + i], op);
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
        res = fp32_unary_compute(bf16_hp.f, op);
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
      res = fp32_unary_compute(in[(j*ldi) + i], op);
      libxsmm_rne_convert_fp32_bf16( &res, &out[(j*ldo) + i], 1 );
    }
  }
}

void unary_op_bf16_f32_gold(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, libxsmm_bfloat16 *in, float *out, unsigned int op) {
  libxsmm_blasint i, j;
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M; ++i ) {
      union libxsmm_bfloat16_hp bf16_hp;
      bf16_hp.i[1] = in[(j*ldi) + i];
      bf16_hp.i[0] = 0;
      out[(j*ldo) + i] = fp32_unary_compute(bf16_hp.f, op);
    }
  }
}

void test_unary_op_f32_f32( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned int op, unsigned int use_bcast) {
  float *in, *in_vector, *_in;
  float *out, *out_gold;
  unsigned int i, j;
  unsigned int s;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_meltw_unary_type  unary_type;
  char opname[256];

  set_opname(op, opname);
  set_unarytype(op, &unary_type);

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
  _in       = in;

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      in[(i*ldi)+j] = (float)libxsmm_rng_f64();
    }
  }

  if (use_bcast != NO_BCAST) {
    in_vector =  (float*) libxsmm_aligned_malloc( sizeof(float)*LIBXSMM_MAX(ldi, N),   64);
    if (use_bcast == ROW_BCAST) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < ldi; ++j ) {
          in[(i*ldi)+j] = in[i*ldi];
        }
      }
      for ( i = 0; i < N; ++i ) {
        in_vector[i] = in[i*ldi];
      }
    }
    if (use_bcast == COL_BCAST) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < ldi; ++j ) {
          in[(i*ldi)+j] = in[j];
        }
      }
      for ( j = 0; j < ldi; ++j ) {
        in_vector[j] = in[j];
      }
    }
    if (use_bcast == SCALAR_BCAST) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < ldi; ++j ) {
          in[(i*ldi)+j] = in[0];
        }
      }
     in_vector[0] = in[0];
    }
    _in = in_vector;
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
  unary_param.in_ptr  = (void*)_in;
  unary_param.out_ptr = (void*)out;
  unary_param.mask_ptr = NULL;
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  if (use_bcast != NO_BCAST) {
    if (use_bcast == ROW_BCAST) {
      unary_flags = LIBXSMM_MELTW_TYPE_UNARY_BCAST_ROW;
    }
    if (use_bcast == COL_BCAST) {
      unary_flags = LIBXSMM_MELTW_TYPE_UNARY_BCAST_COL;
    }
    if (use_bcast == SCALAR_BCAST) {
      unary_flags = LIBXSMM_MELTW_TYPE_UNARY_BCAST_SCALAR;
    }
  }
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
  if (use_bcast != NO_BCAST) {
    libxsmm_free( in_vector );
  }
}

void test_unary_op_bf16_bf16( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned int op, unsigned int use_bcast ) {
  libxsmm_bfloat16 *in, *in_vector, *_in;
  libxsmm_bfloat16 *out, *out_gold;
  unsigned int i, j;
  unsigned int s;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_meltw_unary_type  unary_type;
  libxsmm_dnn_datatype compute_dtype = LIBXSMM_DATATYPE_F32;
  char opname[256];

  set_opname(op, opname);
  set_unarytype(op, &unary_type);

  if ( op == COPY_OP ) {
    compute_dtype = LIBXSMM_DATATYPE_BF16;
  }
  if ( op == XOR_OP ) {
    compute_dtype = LIBXSMM_DATATYPE_BF16;
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
  _in       = in;

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      union libxsmm_bfloat16_hp bf16_hp;
      bf16_hp.f = (float)libxsmm_rng_f64();
      in[(i*ldi)+j] = bf16_hp.i[1];
    }
  }

  if (use_bcast != NO_BCAST) {
    in_vector =  (float*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*LIBXSMM_MAX(ldi, N),   64);
    if (use_bcast == ROW_BCAST) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < ldi; ++j ) {
          in[(i*ldi)+j] = in[i*ldi];
        }
      }
      for ( i = 0; i < N; ++i ) {
        in_vector[i] = in[i*ldi];
      }
    }
    if (use_bcast == COL_BCAST) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < ldi; ++j ) {
          in[(i*ldi)+j] = in[j];
        }
      }
      for ( j = 0; j < ldi; ++j ) {
        in_vector[j] = in[j];
      }
    }
    if (use_bcast == SCALAR_BCAST) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < ldi; ++j ) {
          in[(i*ldi)+j] = in[0];
        }
      }
     in_vector[0] = in[0];
    }
    _in = in_vector;
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
  unary_param.in_ptr  = (void*)_in;
  unary_param.out_ptr = (void*)out;
  unary_param.mask_ptr = NULL;
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  if (use_bcast != NO_BCAST) {
    if (use_bcast == ROW_BCAST) {
      unary_flags = LIBXSMM_MELTW_TYPE_UNARY_BCAST_ROW;
    }
    if (use_bcast == COL_BCAST) {
      unary_flags = LIBXSMM_MELTW_TYPE_UNARY_BCAST_COL;
    }
    if (use_bcast == SCALAR_BCAST) {
      unary_flags = LIBXSMM_MELTW_TYPE_UNARY_BCAST_SCALAR;
    }
  }
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
  if (use_bcast != NO_BCAST) {
    libxsmm_free( in_vector );
  }
}

void test_unary_op_f32_bf16( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned int op, unsigned int use_bcast ) {
  float *in, *in_vector, *_in;
  libxsmm_bfloat16 *out, *out_gold;
  unsigned int i, j;
  unsigned int s;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_meltw_unary_type  unary_type;
  char opname[256];

  set_opname(op, opname);
  set_unarytype(op, &unary_type);

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
  _in       = in;

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      in[(i*ldi)+j] = (float)libxsmm_rng_f64();
    }
  }

  if (use_bcast != NO_BCAST) {
    in_vector =  (float*) libxsmm_aligned_malloc( sizeof(float)*LIBXSMM_MAX(ldi, N),   64);
    if (use_bcast == ROW_BCAST) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < ldi; ++j ) {
          in[(i*ldi)+j] = in[i*ldi];
        }
      }
      for ( i = 0; i < N; ++i ) {
        in_vector[i] = in[i*ldi];
      }
    }
    if (use_bcast == COL_BCAST) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < ldi; ++j ) {
          in[(i*ldi)+j] = in[j];
        }
      }
      for ( j = 0; j < ldi; ++j ) {
       in_vector[j] = in[j];
      }
    }
    if (use_bcast == SCALAR_BCAST) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < ldi; ++j ) {
          in[(i*ldi)+j] = in[0];
        }
      }
     in_vector[0] = in[0];
    }
    _in = in_vector;
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

  unary_param.in_ptr  = (void*)_in;
  unary_param.out_ptr = (void*)out;
  unary_param.mask_ptr = NULL;
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  if (use_bcast != NO_BCAST) {
    if (use_bcast == ROW_BCAST) {
      unary_flags = LIBXSMM_MELTW_TYPE_UNARY_BCAST_ROW;
    }
    if (use_bcast == COL_BCAST) {
      unary_flags = LIBXSMM_MELTW_TYPE_UNARY_BCAST_COL;
    }
    if (use_bcast == SCALAR_BCAST) {
      unary_flags = LIBXSMM_MELTW_TYPE_UNARY_BCAST_SCALAR;
    }
  }
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
  if (use_bcast != NO_BCAST) {
    libxsmm_free( in_vector );
  }
}

void test_unary_op_bf16_f32( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned int op, unsigned int use_bcast ) {
  libxsmm_bfloat16 *in, *in_vector, *_in;
  float *out, *out_gold;
  unsigned int i, j;
  unsigned int s;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_meltw_unary_type  unary_type;
  char opname[256];

  set_opname(op, opname);
  set_unarytype(op, &unary_type);

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
  _in       = in;

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      union libxsmm_bfloat16_hp bf16_hp;
      bf16_hp.f = (float)libxsmm_rng_f64();
      in[(i*ldi)+j] = bf16_hp.i[1];
    }
  }

  if (use_bcast != NO_BCAST) {
    in_vector =  (float*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*LIBXSMM_MAX(ldi, N),   64);
    if (use_bcast == ROW_BCAST) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < ldi; ++j ) {
          in[(i*ldi)+j] = in[i*ldi];
        }
      }
      for ( i = 0; i < N; ++i ) {
        in_vector[i] = in[i*ldi];
      }
    }
    if (use_bcast == COL_BCAST) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < ldi; ++j ) {
          in[(i*ldi)+j] = in[j];
        }
      }
      for ( j = 0; j < ldi; ++j ) {
        in_vector[j] = in[j];
      }
    }
    if (use_bcast == SCALAR_BCAST) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < ldi; ++j ) {
          in[(i*ldi)+j] = in[0];
        }
      }
     in_vector[0] = in[0];
    }
    _in = in_vector;
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

  unary_param.in_ptr  = (void*)_in;
  unary_param.out_ptr = (void*)out;
  unary_param.mask_ptr = NULL;
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  if (use_bcast != NO_BCAST) {
    if (use_bcast == ROW_BCAST) {
      unary_flags = LIBXSMM_MELTW_TYPE_UNARY_BCAST_ROW;
    }
    if (use_bcast == COL_BCAST) {
      unary_flags = LIBXSMM_MELTW_TYPE_UNARY_BCAST_COL;
    }
    if (use_bcast == SCALAR_BCAST) {
      unary_flags = LIBXSMM_MELTW_TYPE_UNARY_BCAST_SCALAR;
    }
  }
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
  if (use_bcast != NO_BCAST) {
    libxsmm_free( in_vector );
  }
}

int main( int argc, char* argv[] ) {
  libxsmm_blasint dtype_in;
  libxsmm_blasint dtype_out;
  libxsmm_blasint dtype_comp;
  unsigned char op;
  libxsmm_blasint use_bcast;
  libxsmm_blasint M;
  libxsmm_blasint N;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;
  libxsmm_blasint valid_op;
  char opname[256];

  if ( argc != 10 ) {
    printf(" Error! Usage: %s [type] [use_bcast: 0/1/2/3] [prec_in: 4/2] [compute_prec: 4/2] [prec_out: 4/2] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  op         = atoi(argv[1]);
  use_bcast  = atoi(argv[2]);
  dtype_in   = atoi(argv[3]);
  dtype_comp = atoi(argv[4]);
  dtype_out  = atoi(argv[5]);
  M          = atoi(argv[6]);
  N          = atoi(argv[7]);
  ldi        = atoi(argv[8]);
  ldo        = atoi(argv[9]);

  set_opname(op, opname);

  valid_op = ( op == COPY_OP || op == X2_OP || op == XOR_OP || op == TANH_OP || op == SIGMOID_OP || op == GELU_OP ||
               op == GELU_INV_OP || op == TANH_INV_OP || op == SIGMOID_INV_OP || op == SQRT_OP || op == NEGATE_OP ||
               op == INC_OP || op == RCP_OP || op == RCP_SQRT_OP) ? 1 : 0;

  if (use_bcast != NO_BCAST) {
    if (use_bcast == ROW_BCAST) {
      printf("Using row broadcast for the input row-vector ...\n");
    }
    if (use_bcast == COL_BCAST) {
      printf("Using column broadcast for the input column-vector...\n");
    }
    if (use_bcast == SCALAR_BCAST) {
      printf("Using scalar broadcast for the input value...\n");
    }
  }
  if ( op == COPY_OP && dtype_in == 4 && dtype_out == 4 && dtype_comp == 4 ) {
    printf("Testing F32 F32 copy\n");
    test_unary_op_f32_f32( M, N, ldi, ldo, op, use_bcast);
  } else if ( op == COPY_OP && dtype_in == 2 && dtype_out == 2 && (dtype_comp == 4 || dtype_comp == 2) ) {
    printf("Testing BF16 BF16 copy\n");
    test_unary_op_bf16_bf16( M, N, ldi, ldo, op, use_bcast);
  } else if ( op == COPY_OP && dtype_in == 4 && dtype_out == 2 && (dtype_comp == 4 || dtype_comp == 2) ) {
    printf("Testing F32 BF16 copy\n");
    test_unary_op_f32_bf16( M, N, ldi, ldo, op, use_bcast);
  } else if ( op == COPY_OP && dtype_in == 2 && dtype_out == 4 && (dtype_comp == 4 || dtype_comp == 2) ) {
    printf("Testing BF16 F32 copy\n");
    test_unary_op_bf16_f32( M, N, ldi, ldo, op, use_bcast);
  } else if ( valid_op > 0 && dtype_in == 4 && dtype_out == 4 && dtype_comp == 4 ) {
    printf("Testing F32 F32 %s\n", opname);
    test_unary_op_f32_f32( M, N, ldi, ldo, op, use_bcast);
  } else if ( valid_op > 0 && dtype_in == 2 && dtype_out == 2 && (dtype_comp == 4 || (dtype_comp == 2 && op == XOR_OP)) ) {
    printf("Testing BF16 BF16 %s\n", opname);
    test_unary_op_bf16_bf16( M, N, ldi, ldo, op, use_bcast);
  } else if ( valid_op > 0 && dtype_in == 4 && dtype_out == 2 && dtype_comp == 4 ) {
    printf("Testing F32 BF16 %s\n", opname);
    test_unary_op_f32_bf16( M, N, ldi, ldo, op, use_bcast);
  } else if ( valid_op > 0 && dtype_in == 2 && dtype_out == 4 && dtype_comp == 4 ) {
    printf("Testing BF16 F32 %s\n", opname);
    test_unary_op_bf16_f32( M, N, ldi, ldo, op, use_bcast);
  } else {
    printf(" Error! Usage: %s [type] [use_bcast: 0/1/2/3] [prec_in: 4/2] [compute_prec: 4/2] [prec_out: 4/2] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  return 0;
}
