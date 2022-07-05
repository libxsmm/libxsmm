/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
*               Friedrich Schiller University Jena - All rights reserved.     *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.), Antonio Noack (FSU Jena)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "eltwise_common.h"

#define PI 3.14159265358979323846
#define EPS 1.19209290e-03F

#define RND_RNE 0
#define RND_STOCHASTIC 1

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
#if 0
#define USE_ZERO_RNG_STATE_UNITTEST
#endif

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
  return (0.5f + 0.5f * LIBXSMM_ERFF(x/LIBXSMM_SQRTF(2.0f)) + x/(LIBXSMM_SQRTF(2.0f*PI)) * LIBXSMM_EXPF(-0.5f*x*x) );
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

void unary_op_gold(const libxsmm_blasint M, const libxsmm_blasint N, const libxsmm_blasint ldi, const libxsmm_blasint ldo, const void *in, void *out,
                   const unsigned int op, const libxsmm_datatype dtype_in, const libxsmm_datatype dtype_out, const libxsmm_datatype dtype_comp, libxsmm_meltw_unary_flags flags) {
  size_t i,j;

  if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_F32) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    const float* f_in = (const float*)in;
    float* f_out = (float*)out;
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        f_out[(j*ldo) + i] = fp32_unary_compute(f_in[(j*ldi) + i], op);
      }
    }
  } else if ( (dtype_in == LIBXSMM_DATATYPE_BF16) && (dtype_out == LIBXSMM_DATATYPE_BF16) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        const libxsmm_bfloat16* bf_in = (const libxsmm_bfloat16*)in;
        libxsmm_bfloat16* bf_out = (libxsmm_bfloat16*)out;
        float in_value, out_value;
        libxsmm_convert_bf16_f32( &(bf_in[(j*ldi) + i]), &in_value, 1 );
        out_value = fp32_unary_compute(in_value, op);
        libxsmm_rne_convert_fp32_bf16(&out_value, &(bf_out[(j*ldo) + i]), 1);
      }
    }
  } else if ( (dtype_in == LIBXSMM_DATATYPE_BF8) && (dtype_out == LIBXSMM_DATATYPE_BF8) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        const libxsmm_bfloat8* bf_in = (const libxsmm_bfloat8*)in;
        libxsmm_bfloat8* bf_out = (libxsmm_bfloat8*)out;
        float in_value, out_value;
        libxsmm_convert_bf8_f32( &(bf_in[(j*ldi) + i]), &in_value, 1 );
        out_value = fp32_unary_compute(in_value, op);
        if ((flags & LIBXSMM_MELTW_FLAG_UNARY_STOCHASTIC_ROUND) > 0 ) {
          libxsmm_stochastic_convert_fp32_bf8(&out_value, &(bf_out[(j*ldo) + i]), 1);
        } else {
          libxsmm_rne_convert_fp32_bf8(&out_value, &(bf_out[(j*ldo) + i]), 1);
        }
      }
    }
  } else if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_BF16) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        const float* f_in = (const float*)in;
        libxsmm_bfloat16* bf_out = (libxsmm_bfloat16*)out;
        float out_value;
        out_value = fp32_unary_compute(f_in[(j*ldi) + i], op);
        libxsmm_rne_convert_fp32_bf16(&out_value, &(bf_out[(j*ldo) + i]), 1);
      }
    }
  } else if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_BF8) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        const float* f_in = (const float*)in;
        libxsmm_bfloat8* bf_out = (libxsmm_bfloat8*)out;
        float out_value;
        out_value = fp32_unary_compute(f_in[(j*ldi) + i], op);
        if ((flags & LIBXSMM_MELTW_FLAG_UNARY_STOCHASTIC_ROUND) > 0 ) {
          libxsmm_stochastic_convert_fp32_bf8(&out_value, &(bf_out[(j*ldo) + i]), 1);
        } else {
          libxsmm_rne_convert_fp32_bf8(&out_value, &(bf_out[(j*ldo) + i]), 1);
        }
      }
    }
  } else if ( (dtype_in == LIBXSMM_DATATYPE_BF16) && (dtype_out == LIBXSMM_DATATYPE_F32) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        const libxsmm_bfloat16* bf_in = (const libxsmm_bfloat16*)in;
        float* f_out = (float*)out;
        float in_value;
        libxsmm_convert_bf16_f32( &(bf_in[(j*ldi) + i]), &in_value, 1 );
        f_out[(j*ldo) + i] = fp32_unary_compute(in_value, op);
      }
    }
  } else if ( (dtype_in == LIBXSMM_DATATYPE_BF8) && (dtype_out == LIBXSMM_DATATYPE_F32) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        const libxsmm_bfloat8* bf_in = (const libxsmm_bfloat8*)in;
        float* f_out = (float*)out;
        float in_value;
        libxsmm_convert_bf8_f32( &(bf_in[(j*ldi) + i]), &in_value, 1 );
        f_out[(j*ldo) + i] = fp32_unary_compute(in_value, op);
      }
    }
  } else {
    /* should happen */
  }
}

int test_unary_op( const libxsmm_blasint M, const libxsmm_blasint N, const libxsmm_blasint ldi, const libxsmm_blasint ldo, const unsigned int op, const unsigned int use_bcast, const libxsmm_datatype dtype_in, const libxsmm_datatype dtype_out, const libxsmm_datatype dtype_comp, const unsigned int rnd_mode ) {
  char *in, *_in;
  char *out, *out_gold;
  unsigned int *rng_state;

  int ret = EXIT_SUCCESS;
  libxsmm_matdiff_info norms_out;
  libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape( M, N, ldi, ldo, dtype_in, dtype_out, dtype_comp );
  libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_meltw_unary_type unary_type;
  libxsmm_meltwfunction_unary unary_kernel;
  char opname[256];
  unsigned long long _N = N;

  set_opname(op, opname);
  set_unarytype(op, &unary_type);

  if ( M > ldi && !(use_bcast == ROW_BCAST || use_bcast == SCALAR_BCAST) ) {
    fprintf( stderr, "test_unary_%s %i %i %i: ldi needs to be equal to or bigger than M\n", opname, dtype_in, dtype_out, dtype_comp);
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_unary_%s %i %i %i: ldo needs to be equal to or bigger than N\n", opname, dtype_in, dtype_out, dtype_comp);
    exit(-1);
  }

  libxsmm_rng_set_seed(1);

  in        = (char*) libxsmm_aligned_malloc( LIBXSMM_TYPESIZE(dtype_in) *N*LIBXSMM_MAX(M,ldi), 64 );
  out       = (char*) libxsmm_aligned_malloc( LIBXSMM_TYPESIZE(dtype_out)*N*ldo,                64 );
  out_gold  = (char*) libxsmm_aligned_malloc( LIBXSMM_TYPESIZE(dtype_out)*N*ldo,                64 );
  _in       = in;

  /* init in */
  init_random_matrix( dtype_in,  in,       1, ldi, N, 0 );
  init_zero_matrix(   dtype_out, out,      1, ldo, N );
  init_zero_matrix(   dtype_out, out_gold, 1, ldo, N );

  if (use_bcast != NO_BCAST) {
    if (use_bcast == ROW_BCAST) {
      apply_row_bcast_matrix( dtype_in, in, ldi, M, N );
    }
    if (use_bcast == COL_BCAST) {
      apply_col_bcast_matrix( dtype_in, in, ldi, M, N );
    }
    if (use_bcast == SCALAR_BCAST) {
      apply_scalar_bcast_matrix( dtype_in, in, ldi, M, N );
    }
  }

  if (rnd_mode == RND_STOCHASTIC) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_STOCHASTIC_ROUND;
    rng_state = libxsmm_rng_create_extstate( 555 );
#ifdef USE_ZERO_RNG_STATE_UNITTEST
    memset( (void*)rng_state, 0, libxsmm_rng_get_extstate_size() );
#endif
    unary_param.op.secondary = (void*)rng_state;
  } else {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  }

  /* compute out_gold */
  unary_op_gold( M, N, ldi, ldo, in, out_gold, op, dtype_in, dtype_out, dtype_comp, unary_flags );

  /* use jited tranpose */
  unary_param.in.primary  = (void*)_in;
  unary_param.out.primary = (void*)out;
  if (unary_type == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) {
    unary_param.op.primary = (void*) &_N;
  }
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


  if (unary_type == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) {
    unary_shape.n = 0;
    unary_kernel = libxsmm_dispatch_meltw_unary_v2( unary_type, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  } else {
    unary_kernel = libxsmm_dispatch_meltw_unary_v2( unary_type, unary_shape, unary_flags );
  }
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for UNARY TPP. Bailing...!\n");
    exit(-1);
  }
  unary_kernel( &unary_param );

  /* compare result */
  norms_out = check_matrix( dtype_out, out_gold, out, ldo, M, N );
  printf("##########################################\n");
  printf("#   Correctness  - Output                #\n");
  printf("##########################################\n");
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  double error_bound =0.0;
  if ( op == RCP_OP || op == RCP_SQRT_OP ) {
    error_bound = 0.0027;
  } else if ( op == SQRT_OP || op == EXP_OP || op == TANH_OP || op == TANH_INV_OP ||
              op == SIGMOID_OP || op == SIGMOID_INV_OP || op == GELU_OP || op == GELU_INV_OP ) {
    if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_F32) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
      error_bound = 0.0007;
    } else if ( dtype_out == LIBXSMM_DATATYPE_BF16 ) {
      error_bound = 0.007;
    } else {
      error_bound = 0.007;
    }
  } else {
    error_bound = 0.00001;
  }

  if ( norms_out.normf_rel > error_bound ) {
    ret = EXIT_FAILURE;
  }

  benchmark_unary(unary_type, unary_shape, unary_flags, unary_param);

  if (rnd_mode == RND_STOCHASTIC) {
    libxsmm_rng_destroy_extstate( rng_state );
  }
  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary simple %i %i %i\n", dtype_in, dtype_out, dtype_comp);
  } else {
    printf("FAILURE unary simple %i %i %i\n", dtype_in, dtype_out, dtype_comp);
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
  unsigned int rnd_mode = RND_RNE;
  char opname[256];
  int ret = EXIT_FAILURE;

  if ( argc != 11 && argc != 10 ) {
    printf(" Error! Usage: %s [type] [use_bcast: 0/1/2/3] [prec_in: 4/2/1] [compute_prec: 4/2] [prec_out: 4/2/1] [M] [N] [ldi] [ldo] [Opt: rnd_mode: 0/1]\n", argv[0] );
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

  if (argc > 10 ) {
    rnd_mode   = atoi(argv[10]);
  }
  if ((rnd_mode == RND_STOCHASTIC && dtype_out != 1) || (rnd_mode > RND_STOCHASTIC)) {
    printf(" Error! rnd_mode = %d is not supported with the selected output precision, prec_out : %d\n", rnd_mode, dtype_out );
    exit(-1);
  }

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
    ret = test_unary_op( M, N, ldi, ldo, op, use_bcast, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, rnd_mode );
  } else if ( op == COPY_OP && dtype_in == 2 && dtype_out == 2 && (dtype_comp == 4 || dtype_comp == 2) ) {
    printf("Testing unary BF16 BF16 copy - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_unary_op( M, N, ldi, ldo, op, use_bcast, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, rnd_mode );
  } else if ( op == COPY_OP && dtype_in == 1 && dtype_out == 1 && (dtype_comp == 4 || dtype_comp == 2) ) {
    printf("Testing unary BF8 BF8 copy - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_unary_op( M, N, ldi, ldo, op, use_bcast, LIBXSMM_DATATYPE_BF8, LIBXSMM_DATATYPE_BF8, LIBXSMM_DATATYPE_F32, rnd_mode );
  } else if ( op == COPY_OP && dtype_in == 4 && dtype_out == 2 && (dtype_comp == 4 || dtype_comp == 2) ) {
    printf("Testing unary F32 BF16 copy - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_unary_op( M, N, ldi, ldo, op, use_bcast, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, rnd_mode );
  } else if ( op == COPY_OP && dtype_in == 4 && dtype_out == 1 && dtype_comp == 4 ) {
    printf("Testing unary F32 BF8 copy - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_unary_op( M, N, ldi, ldo, op, use_bcast, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF8, LIBXSMM_DATATYPE_F32, rnd_mode );
  } else if ( op == COPY_OP && dtype_in == 2 && dtype_out == 4 && (dtype_comp == 4 || dtype_comp == 2) ) {
    printf("Testing unary BF16 F32 copy - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_unary_op( M, N, ldi, ldo, op, use_bcast, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, rnd_mode );
  } else if ( valid_op > 0 && dtype_in == 4 && dtype_out == 4 && dtype_comp == 4 ) {
    printf("Testing unary F32 F32 %s - M=%i, N=%i, LDI=%i, LDO=%i\n", opname, M, N, ldi, ldo);
    ret = test_unary_op( M, N, ldi, ldo, op, use_bcast, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, rnd_mode );
  } else if ( valid_op > 0 && dtype_in == 2 && dtype_out == 2 && (dtype_comp == 4 || (dtype_comp == 2 && op == XOR_OP)) ) {
    printf("Testing unary BF16 BF16 %s - M=%i, N=%i, LDI=%i, LDO=%i\n", opname, M, N, ldi, ldo);
    ret = test_unary_op( M, N, ldi, ldo, op, use_bcast, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, rnd_mode );
  } else if ( valid_op > 0 && dtype_in == 1 && dtype_out == 1 && dtype_comp == 4 ) {
    printf("Testing unary BF8 BF8 %s - M=%i, N=%i, LDI=%i, LDO=%i\n", opname, M, N, ldi, ldo);
    ret = test_unary_op( M, N, ldi, ldo, op, use_bcast, LIBXSMM_DATATYPE_BF8, LIBXSMM_DATATYPE_BF8, LIBXSMM_DATATYPE_F32, rnd_mode );
  } else if ( valid_op > 0 && dtype_in == 4 && dtype_out == 2 && dtype_comp == 4 ) {
    printf("Testing unary F32 BF16 %s - M=%i, N=%i, LDI=%i, LDO=%i\n", opname, M, N, ldi, ldo);
    ret = test_unary_op( M, N, ldi, ldo, op, use_bcast, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, rnd_mode );
  } else if ( valid_op > 0 && dtype_in == 4 && dtype_out == 1 && dtype_comp == 4 ) {
    printf("Testing unary F32 BF8 %s - M=%i, N=%i, LDI=%i, LDO=%i\n", opname, M, N, ldi, ldo);
    ret = test_unary_op( M, N, ldi, ldo, op, use_bcast, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF8, LIBXSMM_DATATYPE_F32, rnd_mode );
  } else if ( valid_op > 0 && dtype_in == 2 && dtype_out == 4 && dtype_comp == 4 ) {
    printf("Testing unary BF16 F32 %s - M=%i, N=%i, LDI=%i, LDO=%i\n", opname, M, N, ldi, ldo);
    ret = test_unary_op( M, N, ldi, ldo, op, use_bcast, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, rnd_mode );
  } else if ( valid_op > 0 && dtype_in == 1 && dtype_out == 4 && dtype_comp == 4 ) {
    printf("Testing unary BF8 F32 %s - M=%i, N=%i, LDI=%i, LDO=%i\n", opname, M, N, ldi, ldo);
    ret = test_unary_op( M, N, ldi, ldo, op, use_bcast, LIBXSMM_DATATYPE_BF8, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, rnd_mode );
  } else {
    printf("  %s, Op: %d, prec_in: %d, compute_prec: %d, prec_out: %d, rnd_mode : %d\n", argv[0], valid_op, dtype_in, dtype_comp, dtype_out, rnd_mode);
    printf(" Error! Usage: %s [type] [use_bcast: 0/1/2/3] [prec_in: 4/2/1] [compute_prec: 4/2] [prec_out: 4/2/1] [M] [N] [ldi] [ldo] [rnd_mode : 0/1]\n", argv[0] );
    exit(-1);
  }

  return ret;
}
