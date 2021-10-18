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

#define COPY_OP 1
#define XOR_OP 2
#define X2_OP 3
#define SQRT_OP 4
#define TANH_OP 7
#define TANH_INV_OP 8
#define SIGMOID_OP 9
#define SIGMOID_INV_OP 10
#define GELU_OP 11
#define GELU_INV_OP 12
#define NEGATE_OP 13
#define INC_OP 14
#define RCP_OP 15
#define RCP_SQRT_OP 16
#define EXP_OP 17
#define REPLICATE_COL_VAR 27

float upconvert_bf16(libxsmm_bfloat16 x) {
  union libxsmm_bfloat16_hp bf16_hp;
  bf16_hp.i[1] = x;
  bf16_hp.i[0] = 0;
  return bf16_hp.f;
}

int unequal_fp32_vals(float a, float b) {
  if (fabs(a-b) < EPS) {
    return 0;
  } else {
    return 1;
  }
}


float fsigmoid(float x) {
  return (LIBXSMM_TANHF(x/2.0f) + 1.0f)/2.0f;
}

float fsigmoid_inv(float x) {
  return fsigmoid(x) * (1.0f-fsigmoid(x));
}

float tanh_inv(float x) {
  return 1.0f-LIBXSMM_TANHF(x)*LIBXSMM_TANHF(x);
}

float gelu(float x) {
  return (LIBXSMM_ERFF(x/LIBXSMM_SQRTF(2.0f)) + 1.0f)*0.5f*x;
}

float gelu_inv(float x) {
  return (0.5f + 0.5f * LIBXSMM_ERFF(x/LIBXSMM_SQRTF(2.0f)) + x/(LIBXSMM_SQRTF(2.0f*PI))* LIBXSMM_EXPF(-0.5f*x*x) );
}

float fp32_unary_compute(float in, unsigned int op) {
  float res = 0;

  if ( op == COPY_OP || op == REPLICATE_COL_VAR) {
    res = in;
  } else if ( op == NEGATE_OP) {
    res = -1.0f * in;
  } else if (op == X2_OP) {
    res = in * in;
  } else if (op == XOR_OP) {
    res = 0;
  } else if (op == TANH_OP) {
    res = LIBXSMM_TANHF(in);
  } else if (op == SIGMOID_OP) {
    res = fsigmoid(in);
  } else if (op == GELU_OP) {
    res = gelu(in);
  } else if (op == GELU_INV_OP) {
    res = gelu_inv(in);
  } else if (op == TANH_INV_OP) {
    res = tanh_inv(in);
  } else if (op == SIGMOID_INV_OP) {
    res = fsigmoid_inv(in);
  } else if (op == SQRT_OP) {
    res = LIBXSMM_SQRTF(in);
  } else if (op == INC_OP) {
    res = in + 1.0f;
  } else if (op == RCP_OP) {
    res = 1.0f/in;
  } else if (op == RCP_SQRT_OP) {
    res = 1.0f/LIBXSMM_SQRTF(in);
  } else if (op == EXP_OP) {
    res = LIBXSMM_EXPF(in);
  } else {
    printf("Invalid OP\n");
    exit(-1);
  }

  return res;
}

void set_opname(unsigned int op, char *opname) {
  if ( op == COPY_OP ) {
    sprintf(opname, "copy");
  } else if ( op == REPLICATE_COL_VAR ) {
    sprintf(opname, "replicate_col_var");
  } else if ( op == X2_OP ) {
    sprintf(opname, "x2");
  } else if ( op == XOR_OP ) {
    sprintf(opname, "xor");
  } else if ( op == TANH_OP ) {
    sprintf(opname, "tanh");
  } else if ( op == SIGMOID_OP ) {
    sprintf(opname, "sigmoid");
  } else if ( op == GELU_OP ) {
    sprintf(opname, "gelu");
  } else if ( op == GELU_INV_OP ) {
    sprintf(opname, "gelu_inv");
  } else if ( op == TANH_INV_OP ) {
    sprintf(opname, "tanh_inv");
  } else if ( op == SIGMOID_INV_OP ) {
    sprintf(opname, "sigmoid_inv");
  } else if ( op == SQRT_OP ) {
    sprintf(opname, "sqrt");
  } else if ( op == NEGATE_OP ) {
    sprintf(opname, "negate");
  } else if (op == INC_OP) {
    sprintf(opname, "inc");
  } else if (op == RCP_OP) {
    sprintf(opname, "reciprocal");
  } else if (op == RCP_SQRT_OP) {
    sprintf(opname, "reciprocal sqrt");
  } else if (op == EXP_OP) {
    sprintf(opname, "exp");
  } else {
    printf("Invalid OP\n");
    exit(-1);
  }
}

void set_unarytype(unsigned int op, libxsmm_meltw_unary_type *type) {
  libxsmm_meltw_unary_type  unary_type;

  if ( op == COPY_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_IDENTITY;
  } else if ( op == REPLICATE_COL_VAR ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR;
  } else if ( op == X2_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_X2;
  } else if ( op == XOR_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_XOR;
  } else if ( op == TANH_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_TANH;
  } else if ( op == SIGMOID_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_SIGMOID;
  } else if ( op == GELU_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_GELU;
  } else if ( op == GELU_INV_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_GELU_INV;
  } else if ( op == TANH_INV_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_TANH_INV;
  } else if ( op == SIGMOID_INV_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV;
  } else if ( op == SQRT_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_SQRT;
  } else if ( op == NEGATE_OP ) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_NEGATE;
  } else if (op == INC_OP) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_INC;
  } else if (op == RCP_OP) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL;
  } else if (op == EXP_OP) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_EXP;
  } else if (op == RCP_SQRT_OP) {
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT;
  } else {
    printf("Invalid OP\n");
    exit(-1);
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

int test_unary_op_f32_f32( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned int op, unsigned int use_bcast) {
  float *in, *in_vector, *_in;
  float *out, *out_gold;
  libxsmm_blasint i, j;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  libxsmm_meltw_unary_type unary_type;
  char opname[256];
  unsigned long long _N = N;

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

  in        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi, 64);
  out       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo, 64);
  out_gold  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo, 64);
  _in       = in;
  in_vector = NULL;

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
//      in[(i*ldi)+j] = (float)libxsmm_rng_f64();
      in[(i*ldi)+j] = (float)libxsmm_rng_f64() - 5.0;
    }
  }

  if (use_bcast != NO_BCAST) {
    in_vector =  (float*) libxsmm_aligned_malloc( sizeof(float)*LIBXSMM_MAX(ldi, N), 64);
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
  unary_param.in.primary  = (void*)_in;
  unary_param.out.primary = (void*)out;
  if (unary_type == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) {
    unary_param.op.primary = (void*) &_N;
  }
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  if (use_bcast != NO_BCAST) {
    if (use_bcast == ROW_BCAST) {
      unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW;
    }
    if (use_bcast == COL_BCAST) {
      unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL;
    }
    if (use_bcast == SCALAR_BCAST) {
      unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR;
    }
  }

  libxsmm_meltwfunction_unary unary_kernel;
  if (unary_type == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) {
    unary_kernel = libxsmm_dispatch_meltw_unary(M, 0, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, unary_type);
  } else {
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, unary_type);
  }
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel( &unary_param );

#if 1
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      // if ( unequal_fp32_vals(out_gold[(i*ldo)+j], out[(i*ldo)+j])  ) {
        printf("error at possition i=%i, j=%i, %f, %f\n", i, j, out[(i*ldo)+j], out_gold[(i*ldo)+j]);
      // }
    }
  }
#endif

  /* compare result */
  libxsmm_matdiff_clear(&norms_out);
  printf("##########################################\n");
  printf("#   Correctness  - Output                #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ldo*N, 1, out_gold, out, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

double error_bound =0.0;
if(RCP_OP || RCP_SQRT_OP){
  error_bound = 0.0027;
}else{
  error_bound = 0.0007;
}

if ( norms_out.normf_rel > error_bound ) {
  ret = EXIT_FAILURE;
}



  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  if (use_bcast != NO_BCAST) {
    libxsmm_free( in_vector );
  }

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary simple fp32 fp32\n");
  } else {
    printf("FAILURE unary simple fp32 fp32\n");
  }

  return ret;
}

int test_unary_op_bf16_bf16( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned int op, unsigned int use_bcast ) {
  libxsmm_bfloat16 *in, *in_vector, *_in;
  libxsmm_bfloat16 *out, *out_gold;
  float *f32out, *f32out_gold;
  libxsmm_blasint i, j;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  libxsmm_meltw_unary_type  unary_type;
  libxsmm_dnn_datatype compute_dtype = LIBXSMM_DATATYPE_F32;
  char opname[256];
  unsigned long long _N = N;

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

  in          = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldi, 64);
  out         = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo, 64);
  out_gold    = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo, 64);
  f32out      = (float*)            libxsmm_aligned_malloc( sizeof(float)*N*ldo,            64);
  f32out_gold = (float*)            libxsmm_aligned_malloc( sizeof(float)*N*ldo,            64);
  _in       = in;
  in_vector = NULL;

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      union libxsmm_bfloat16_hp bf16_hp;
      bf16_hp.f = (float)libxsmm_rng_f64();
      in[(i*ldi)+j] = bf16_hp.i[1];
    }
  }

  if (use_bcast != NO_BCAST) {
    in_vector = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*LIBXSMM_MAX(ldi, N), 64);
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
  unary_param.in.primary  = (void*)_in;
  unary_param.out.primary = (void*)out;
  if (unary_type == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) {
    unary_param.op.primary = (void*) &_N;
  }
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  if (use_bcast != NO_BCAST) {
    if (use_bcast == ROW_BCAST) {
      unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW;
    }
    if (use_bcast == COL_BCAST) {
      unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL;
    }
    if (use_bcast == SCALAR_BCAST) {
      unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR;
    }
  }
  libxsmm_meltwfunction_unary unary_kernel;
  if (unary_type == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) {
    unary_kernel = libxsmm_dispatch_meltw_unary(M, 0, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, compute_dtype, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, unary_type);
  } else {
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, compute_dtype, LIBXSMM_DATATYPE_BF16, unary_flags, unary_type);
  }

  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel( &unary_param );

  /* compare result */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldo; ++j ) {
      f32out_gold[(i*ldo)+j] = upconvert_bf16(out_gold[(i*ldo)+j]);
      f32out[(i*ldo)+j] = upconvert_bf16(out[(i*ldo)+j]);
    }
  }

  libxsmm_matdiff_clear(&norms_out);
  printf("##########################################\n");
  printf("#   Correctness  - Output                #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ldo*N, 1, f32out_gold, f32out, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  if ( norms_out.normf_rel > 0.007 ) {
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( f32out_gold );
  libxsmm_free( f32out );
  libxsmm_free( in );
  if (use_bcast != NO_BCAST) {
    libxsmm_free( in_vector );
  }

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary simple bf16 bf16\n");
  } else {
    printf("FAILURE unary simple bf16 bf16\n");
  }

  return ret;
}

int test_unary_op_f32_bf16( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned int op, unsigned int use_bcast ) {
  float *in, *in_vector, *_in;
  libxsmm_bfloat16 *out, *out_gold;
  float *f32out, *f32out_gold;
  libxsmm_blasint i, j;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  libxsmm_meltw_unary_type  unary_type;
  char opname[256];
  unsigned long long _N = N;

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

  in          = (float*)            libxsmm_aligned_malloc( sizeof(float)           *N*ldi, 64);
  out         = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo, 64);
  out_gold    = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo, 64);
  f32out      = (float*)            libxsmm_aligned_malloc( sizeof(float)           *N*ldo, 64);
  f32out_gold = (float*)            libxsmm_aligned_malloc( sizeof(float)           *N*ldo, 64);
  _in       = in;
  in_vector = NULL;

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

  unary_param.in.primary  = (void*)_in;
  unary_param.out.primary = (void*)out;
  if (unary_type == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) {
    unary_param.op.primary = (void*) &_N;
  }
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  if (use_bcast != NO_BCAST) {
    if (use_bcast == ROW_BCAST) {
      unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW;
    }
    if (use_bcast == COL_BCAST) {
      unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL;
    }
    if (use_bcast == SCALAR_BCAST) {
      unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR;
    }
  }
  libxsmm_meltwfunction_unary unary_kernel;
  if (unary_type == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) {
    unary_kernel = libxsmm_dispatch_meltw_unary(M, 0, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, unary_type);
  } else {
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, unary_flags, unary_type);
  }

  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel( &unary_param );

  /* compare result */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldo; ++j ) {
      f32out_gold[(i*ldo)+j] = upconvert_bf16(out_gold[(i*ldo)+j]);
      f32out[(i*ldo)+j] = upconvert_bf16(out[(i*ldo)+j]);
    }
  }

  libxsmm_matdiff_clear(&norms_out);
  printf("##########################################\n");
  printf("#   Correctness  - Output                #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ldo*N, 1, f32out_gold, f32out, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  if ( norms_out.normf_rel > 0.007 ) {
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( f32out_gold );
  libxsmm_free( f32out );
  libxsmm_free( in );
  if (use_bcast != NO_BCAST) {
    libxsmm_free( in_vector );
  }

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary simple fp32 bf16\n");
  } else {
    printf("FAILURE unary simple fp32 bf16\n");
  }

  return ret;
}

int test_unary_op_bf16_f32( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned int op, unsigned int use_bcast ) {
  libxsmm_bfloat16 *in, *in_vector, *_in;
  float *out, *out_gold;
  libxsmm_blasint i, j;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  libxsmm_meltw_unary_type  unary_type;
  char opname[256];
  unsigned long long _N = N;

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

  in        = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldi, 64);
  out       = (float*)            libxsmm_aligned_malloc( sizeof(float)*N*ldo,            64);
  out_gold  = (float*)            libxsmm_aligned_malloc( sizeof(float)*N*ldo,            64);
  _in       = in;
  in_vector = NULL;

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      union libxsmm_bfloat16_hp bf16_hp;
      bf16_hp.f = (float)libxsmm_rng_f64();
      in[(i*ldi)+j] = bf16_hp.i[1];
    }
  }

  if (use_bcast != NO_BCAST) {
    in_vector = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*LIBXSMM_MAX(ldi, N), 64);
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

  unary_param.in.primary  = (void*)_in;
  unary_param.out.primary = (void*)out;
  if (unary_type == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) {
    unary_param.op.primary = (void*) &_N;
  }
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  if (use_bcast != NO_BCAST) {
    if (use_bcast == ROW_BCAST) {
      unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW;
    }
    if (use_bcast == COL_BCAST) {
      unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL;
    }
    if (use_bcast == SCALAR_BCAST) {
      unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR;
    }
  }

  libxsmm_meltwfunction_unary unary_kernel;
  if (unary_type == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) {
    unary_kernel = libxsmm_dispatch_meltw_unary(M, 0, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, unary_type);
  } else {
    unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, unary_type);
  }

  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel( &unary_param );

  /* compare result */
  libxsmm_matdiff_clear(&norms_out);
  printf("##########################################\n");
  printf("#   Correctness  - Output                #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, ldo*N, 1, out_gold, out, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  if ( norms_out.normf_rel > 0.007 ) {
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  if (use_bcast != NO_BCAST) {
    libxsmm_free( in_vector );
  }

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary simple bf16 fp32\n");
  } else {
    printf("FAILURE unary simple bf16 fp32\n");
  }

  return ret;
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
  int ret = EXIT_FAILURE;

  if ( argc != 10 ) {
    printf(" Error! Usage: %s [type] [use_bcast: 0/1/2/3] [prec_in: 4/2] [compute_prec: 4/2] [prec_out: 4/2] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  op         = (unsigned char)atoi(argv[1]);
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
               op == INC_OP || op == RCP_OP || op == RCP_SQRT_OP || op == EXP_OP || op == REPLICATE_COL_VAR) ? 1 : 0;

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

  if (op == REPLICATE_COL_VAR) {
    use_bcast = COL_BCAST;
  }

  if ( op == COPY_OP && dtype_in == 4 && dtype_out == 4 && dtype_comp == 4 ) {
    printf("Testing unary F32 F32 copy - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_unary_op_f32_f32( M, N, ldi, ldo, op, use_bcast);
  } else if ( op == COPY_OP && dtype_in == 2 && dtype_out == 2 && (dtype_comp == 4 || dtype_comp == 2) ) {
    printf("Testing unary BF16 BF16 copy - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_unary_op_bf16_bf16( M, N, ldi, ldo, op, use_bcast);
  } else if ( op == COPY_OP && dtype_in == 4 && dtype_out == 2 && (dtype_comp == 4 || dtype_comp == 2) ) {
    printf("Testing unary F32 BF16 copy - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_unary_op_f32_bf16( M, N, ldi, ldo, op, use_bcast);
  } else if ( op == COPY_OP && dtype_in == 2 && dtype_out == 4 && (dtype_comp == 4 || dtype_comp == 2) ) {
    printf("Testing unary BF16 F32 copy - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_unary_op_bf16_f32( M, N, ldi, ldo, op, use_bcast);
  } else if ( valid_op > 0 && dtype_in == 4 && dtype_out == 4 && dtype_comp == 4 ) {
    printf("Testing unary F32 F32 %s - M=%i, N=%i, LDI=%i, LDO=%i\n", opname, M, N, ldi, ldo);
    ret = test_unary_op_f32_f32( M, N, ldi, ldo, op, use_bcast);
  } else if ( valid_op > 0 && dtype_in == 2 && dtype_out == 2 && (dtype_comp == 4 || (dtype_comp == 2 && op == XOR_OP)) ) {
    printf("Testing unary BF16 BF16 %s - M=%i, N=%i, LDI=%i, LDO=%i\n", opname, M, N, ldi, ldo);
    ret = test_unary_op_bf16_bf16( M, N, ldi, ldo, op, use_bcast);
  } else if ( valid_op > 0 && dtype_in == 4 && dtype_out == 2 && dtype_comp == 4 ) {
    printf("Testing unary F32 BF16 %s - M=%i, N=%i, LDI=%i, LDO=%i\n", opname, M, N, ldi, ldo);
    ret = test_unary_op_f32_bf16( M, N, ldi, ldo, op, use_bcast);
  } else if ( valid_op > 0 && dtype_in == 2 && dtype_out == 4 && dtype_comp == 4 ) {
    printf("Testing unary BF16 F32 %s - M=%i, N=%i, LDI=%i, LDO=%i\n", opname, M, N, ldi, ldo);
    ret = test_unary_op_bf16_f32( M, N, ldi, ldo, op, use_bcast);
  } else {
    printf(" Error! Usage: %s [type] [use_bcast: 0/1/2/3] [prec_in: 4/2] [compute_prec: 4/2] [prec_out: 4/2] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  return ret;
}
