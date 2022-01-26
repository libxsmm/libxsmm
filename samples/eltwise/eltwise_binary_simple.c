/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define NO_BCAST 0
#define ROW_BCAST_IN0 1
#define COL_BCAST_IN0 2
#define SCALAR_BCAST_IN0 3
#define ROW_BCAST_IN1 4
#define COL_BCAST_IN1 5
#define SCALAR_BCAST_IN1 6

#define ADD_OP 1
#define MUL_OP 2
#define SUB_OP 3
#define DIV_OP 4
#define MULADD_OP 5

float upconvert_bf16(libxsmm_bfloat16 x) {
  union libxsmm_bfloat16_hp bf16_hp;
  bf16_hp.i[1] = x;
  bf16_hp.i[0] = 0;
  return bf16_hp.f;
}

float fp32_binary_compute(float in0, float in1, float out, unsigned int op) {
  float res = out;

  if ( op == ADD_OP) {
    res = in0 + in1;
  } else  if ( op == SUB_OP) {
    res = in0 - in1;
  } else if ( op == MUL_OP) {
    res = in0 * in1;
  } else if ( op == DIV_OP) {
    res = in0 / in1;
  } else if ( op == MULADD_OP) {
    res += in0 * in1;
  } else {
    printf("Invalid OP\n");
    exit(-1);
  }

  return res;
}

void set_opname(unsigned int op, char *opname) {
  if ( op == ADD_OP ) {
    sprintf(opname, "add");
  } else if ( op == SUB_OP ) {
    sprintf(opname, "sub");
  } else if ( op == MUL_OP ) {
    sprintf(opname, "mul");
  } else if ( op == DIV_OP ) {
    sprintf(opname, "div");
  } else if ( op == MULADD_OP ) {
    sprintf(opname, "muladd");
  } else {
    printf("Invalid OP\n");
    exit(-1);
  }
}

void set_binarytype(unsigned int op, libxsmm_meltw_binary_type *type) {
  libxsmm_meltw_binary_type  binary_type;

  if ( op == ADD_OP ) {
    binary_type = LIBXSMM_MELTW_TYPE_BINARY_ADD;
  } else if ( op == SUB_OP ) {
    binary_type = LIBXSMM_MELTW_TYPE_BINARY_SUB;
  } else if ( op == MUL_OP ) {
    binary_type = LIBXSMM_MELTW_TYPE_BINARY_MUL;
  } else if ( op == DIV_OP ) {
    binary_type = LIBXSMM_MELTW_TYPE_BINARY_DIV;
  } else if ( op == MULADD_OP ) {
    binary_type = LIBXSMM_MELTW_TYPE_BINARY_MULADD;
  } else {
    printf("Invalid OP\n");
    exit(-1);
  }

  *type = binary_type;
}

void binary_op_f32_f32_gold(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, float *in, float *in2, float *out, unsigned int op) {
  libxsmm_blasint i, j;
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M; ++i ) {
      out[(j*ldo) + i] = fp32_binary_compute(in[(j*ldi) + i], in2[(j*ldi) + i], out[(j*ldo) + i], op);
    }
  }
}

void binary_op_bf16_bf16_gold(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, libxsmm_bfloat16 *in, libxsmm_bfloat16 *in2, libxsmm_bfloat16 *out, unsigned int op) {
  libxsmm_blasint i, j;
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M; ++i ) {
      union libxsmm_bfloat16_hp bf16_hp, bf16_hp2, bf16_hp3;
      float res;
      bf16_hp.i[1] = in[(j*ldi) + i];
      bf16_hp.i[0] = 0;
      bf16_hp2.i[1] = in2[(j*ldi) + i];
      bf16_hp2.i[0] = 0;
      bf16_hp3.i[1] = out[(j*ldo) + i];
      bf16_hp3.i[0] = 0;
      res = bf16_hp3.f;
      res = fp32_binary_compute(bf16_hp.f, bf16_hp2.f, res, op);
      libxsmm_rne_convert_fp32_bf16( &res, &out[(j*ldo) + i], 1 );
    }
  }
}

void binary_op_f32_bf16_gold(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, float *in, float *in2, libxsmm_bfloat16 *out, unsigned int op) {
  libxsmm_blasint i, j;
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M; ++i ) {
      union libxsmm_bfloat16_hp bf16_hp;
      float res;
      bf16_hp.i[1] = out[(j*ldo) + i];
      bf16_hp.i[0] = 0;
      res = bf16_hp.f;
      res = fp32_binary_compute(in[(j*ldi) + i], in2[(j*ldi) + i], res, op);
      libxsmm_rne_convert_fp32_bf16( &res, &out[(j*ldo) + i], 1 );
    }
  }
}

void binary_op_bf16_f32_gold(libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, libxsmm_bfloat16 *in, libxsmm_bfloat16 *in2, float *out, unsigned int op) {
  libxsmm_blasint i, j;
  for ( j = 0; j < N; ++j ) {
    for ( i = 0; i < M; ++i ) {
      union libxsmm_bfloat16_hp bf16_hp, bf16_hp2;
      bf16_hp.i[1] = in[(j*ldi) + i];
      bf16_hp.i[0] = 0;
      bf16_hp2.i[1] = in2[(j*ldi) + i];
      bf16_hp2.i[0] = 0;
      out[(j*ldo) + i] = fp32_binary_compute(bf16_hp.f, bf16_hp2.f, out[(j*ldo) + i], op);
    }
  }
}

int test_binary_op_f32_f32( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned int op, unsigned int use_bcast ) {
  float *in, *_in, *in2, *_in2;
  float *out, *out_gold;
  unsigned int i, j;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_binary_param binary_param;
  libxsmm_meltw_binary_flags binary_flags;
  libxsmm_meltw_binary_shape binary_shape;
  libxsmm_matdiff_info norms_out;
  libxsmm_meltw_binary_type  binary_type;
  char opname[256];

  set_opname(op, opname);
  set_binarytype(op, &binary_type);

  if ( M > ldi && !(use_bcast == ROW_BCAST_IN0 || use_bcast == SCALAR_BCAST_IN0 || use_bcast == ROW_BCAST_IN1 || use_bcast == SCALAR_BCAST_IN1) ) {
    fprintf( stderr, "test_binary_%s_f32_f32: ldi needs to be equal to or bigger than M\n", opname);
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_binary_%s_f32_f32: ldo needs to be equal to or bigger than N\n", opname);
    exit(-1);
  }

  libxsmm_rng_set_seed(1);

  in        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*LIBXSMM_MAX(M,ldi),   64);
  in2       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*LIBXSMM_MAX(M,ldi),   64);
  out       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  out_gold  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  _in       = in;
  _in2      = in2;

   /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      in[(i*ldi)+j] = (float)libxsmm_rng_f64();
    }
  }

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      in2[(i*ldi)+j] = (float)libxsmm_rng_f64();
    }
  }

  if (use_bcast != NO_BCAST) {
    if (use_bcast == ROW_BCAST_IN0) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in[(i*ldi)+j] = in[i*ldi];
        }
      }
    }
    if (use_bcast == COL_BCAST_IN0) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in[(i*ldi)+j] = in[j];
        }
      }
    }
    if (use_bcast == SCALAR_BCAST_IN0) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in[(i*ldi)+j] = in[0];
        }
      }
    }
    if (use_bcast == ROW_BCAST_IN1) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in2[(i*ldi)+j] = in2[i*ldi];
        }
      }
    }
    if (use_bcast == COL_BCAST_IN1) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in2[(i*ldi)+j] = in2[j];
        }
      }
    }
    if (use_bcast == SCALAR_BCAST_IN1) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in2[(i*ldi)+j] = in2[0];
        }
      }
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
    binary_op_f32_f32_gold( M, 1, ldi, ldo, &in[(i*ldi)], &in2[(i*ldi)], &out_gold[(i*ldo)], op );
  }

  /* use jited tranpose */
  binary_param.in0.primary  = (void*)_in;
  binary_param.in1.primary  = (void*)_in2;
  binary_param.out.primary  = (void*)out;
  binary_flags = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  if (use_bcast != NO_BCAST) {
    if (use_bcast == ROW_BCAST_IN0) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0;
    }
    if (use_bcast == COL_BCAST_IN0) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
    }
    if (use_bcast == SCALAR_BCAST_IN0) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0;
    }
    if (use_bcast == ROW_BCAST_IN1) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1;
    }
    if (use_bcast == COL_BCAST_IN1) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1;
    }
    if (use_bcast == SCALAR_BCAST_IN1) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1;
    }
  }

  binary_shape.m = M;
  binary_shape.n = N;
  binary_shape.ldi = &ldi;
  binary_shape.ldi2 = &ldi;
  binary_shape.ldo = &ldo;
  binary_shape.in_type = LIBXSMM_DATATYPE_F32;
  binary_shape.out_type = LIBXSMM_DATATYPE_F32;
  binary_shape.comp_type = LIBXSMM_DATATYPE_F32;

  libxsmm_meltwfunction_binary binary_kernel = libxsmm_dispatch_meltw_binary_v2( binary_type, binary_shape, binary_flags );
  if ( binary_kernel == NULL ) {
    fprintf( stderr, "JIT for BINARY TPP. Bailing...!\n");
    exit(-1);
  }
  binary_kernel( &binary_param );

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

  if ( norms_out.normf_rel > 0.00001 ) {
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( in2 );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS binary simple fp32 fp32\n");
  } else {
    printf("FAILURE binary simple fp32 fp32\n");
  }

  return ret;
}

int test_binary_op_bf16_bf16( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned int op, unsigned int use_bcast ) {
  libxsmm_bfloat16 *in, *_in, *in2, *_in2;
  libxsmm_bfloat16 *out, *out_gold;
  float *f32out, *f32out_gold;
  unsigned int i, j;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_binary_param binary_param;
  libxsmm_meltw_binary_flags binary_flags;
  libxsmm_meltw_binary_shape binary_shape;
  libxsmm_matdiff_info norms_out;
  libxsmm_meltw_binary_type  binary_type;
  char opname[256];

  set_opname(op, opname);
  set_binarytype(op, &binary_type);

  if ( M > ldi && !(use_bcast == ROW_BCAST_IN0 || use_bcast == SCALAR_BCAST_IN0 || use_bcast == ROW_BCAST_IN1 || use_bcast == SCALAR_BCAST_IN1) ) {
    fprintf( stderr, "test_binary_%s_bf16_bf16: ldi needs to be equal to or bigger than M\n", opname);
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_binary_%s_bf16_bf16: ldo needs to be equal to or bigger than N\n", opname);
    exit(-1);
  }

  libxsmm_rng_set_seed(1);

  in          = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*LIBXSMM_MAX(M,ldi), 64);
  in2         = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*LIBXSMM_MAX(M,ldi), 64);
  out         = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo, 64);
  out_gold    = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo, 64);
  f32out      = (float*)            libxsmm_aligned_malloc( sizeof(float)*N*ldo,            64);
  f32out_gold = (float*)            libxsmm_aligned_malloc( sizeof(float)*N*ldo,            64);
   _in        = in;
  _in2        = in2;

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      union libxsmm_bfloat16_hp bf16_hp;
      bf16_hp.f = (float)libxsmm_rng_f64();
      in[(i*ldi)+j] = bf16_hp.i[1];
    }
  }

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      union libxsmm_bfloat16_hp bf16_hp;
      bf16_hp.f = (float)libxsmm_rng_f64();
      in2[(i*ldi)+j] = bf16_hp.i[1];
    }
  }

  if (use_bcast != NO_BCAST) {
    if (use_bcast == ROW_BCAST_IN0) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in[(i*ldi)+j] = in[i*ldi];
        }
      }
    }
    if (use_bcast == COL_BCAST_IN0) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in[(i*ldi)+j] = in[j];
        }
      }
    }
    if (use_bcast == SCALAR_BCAST_IN0) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in[(i*ldi)+j] = in[0];
        }
      }
    }
    if (use_bcast == ROW_BCAST_IN1) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in2[(i*ldi)+j] = in2[i*ldi];
        }
      }
    }
    if (use_bcast == COL_BCAST_IN1) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in2[(i*ldi)+j] = in2[j];
        }
      }
    }
    if (use_bcast == SCALAR_BCAST_IN1) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in2[(i*ldi)+j] = in2[0];
        }
      }
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
    binary_op_bf16_bf16_gold( M, 1, ldi, ldo, &in[(i*ldi)], &in2[(i*ldi)], &out_gold[(i*ldo)], op );
  }

  /* use jited tranpose */
  binary_param.in0.primary  = (void*)_in;
  binary_param.in1.primary  = (void*)_in2;
  binary_param.out.primary  = (void*)out;
  binary_flags = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  if (use_bcast != NO_BCAST) {
    if (use_bcast == ROW_BCAST_IN0) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0;
    }
    if (use_bcast == COL_BCAST_IN0) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
    }
    if (use_bcast == SCALAR_BCAST_IN0) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0;
    }
    if (use_bcast == ROW_BCAST_IN1) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1;
    }
    if (use_bcast == COL_BCAST_IN1) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1;
    }
    if (use_bcast == SCALAR_BCAST_IN1) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1;
    }
  }

  binary_shape.m = M;
  binary_shape.n = N;
  binary_shape.ldi = &ldi;
  binary_shape.ldi2 = &ldi;
  binary_shape.ldo = &ldo;
  binary_shape.in_type = LIBXSMM_DATATYPE_BF16;
  binary_shape.out_type = LIBXSMM_DATATYPE_BF16;
  binary_shape.comp_type = LIBXSMM_DATATYPE_F32;

  libxsmm_meltwfunction_binary binary_kernel = libxsmm_dispatch_meltw_binary_v2( binary_type, binary_shape, binary_flags );
  if ( binary_kernel == NULL ) {
    fprintf( stderr, "JIT for BINARY TPP. Bailing...!\n");
    exit(-1);
  }
  binary_kernel( &binary_param );

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

  if ( norms_out.normf_rel > 0.005 ) {
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( f32out_gold );
  libxsmm_free( f32out );
  libxsmm_free( in );
  libxsmm_free( in2 );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS binary simple bf16 bf16\n");
  } else {
    printf("FAILURE binary simple bf16 bf16\n");
  }

  return ret;
}

int test_binary_op_f32_bf16( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned int op, unsigned int use_bcast) {
  float *in,*_in, *in2, *_in2;
  libxsmm_bfloat16 *out, *out_gold;
  float *f32out, *f32out_gold;
  unsigned int i, j;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_binary_param binary_param;
  libxsmm_meltw_binary_flags binary_flags;
  libxsmm_meltw_binary_shape binary_shape;
  libxsmm_matdiff_info norms_out;
  libxsmm_meltw_binary_type  binary_type;
  char opname[256];

  set_opname(op, opname);
  set_binarytype(op, &binary_type);

  if ( M > ldi && !(use_bcast == ROW_BCAST_IN0 || use_bcast == SCALAR_BCAST_IN0 || use_bcast == ROW_BCAST_IN1 || use_bcast == SCALAR_BCAST_IN1) ) {
    fprintf( stderr, "test_binary_%s_f32_bf16: ldi needs to be equal to or bigger than M\n", opname);
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_binary_%s_f32_bf16: ldo needs to be equal to or bigger than N\n", opname);
    exit(-1);
  }

  libxsmm_rng_set_seed(1);

  in          = (float*) libxsmm_aligned_malloc( sizeof(float)*N*LIBXSMM_MAX(M,ldi),                       64);
  in2         = (float*) libxsmm_aligned_malloc( sizeof(float)*N*LIBXSMM_MAX(M,ldi),                       64);
  out         = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo, 64);
  out_gold    = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo, 64);
  f32out      = (float*)            libxsmm_aligned_malloc( sizeof(float)*N*ldo,            64);
  f32out_gold = (float*)            libxsmm_aligned_malloc( sizeof(float)*N*ldo,            64);
  _in         = in;
  _in2        = in2;

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      in[(i*ldi)+j] = (float)libxsmm_rng_f64();
    }
  }

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      in2[(i*ldi)+j] = (float)libxsmm_rng_f64();
    }
  }

  if (use_bcast != NO_BCAST) {
    if (use_bcast == ROW_BCAST_IN0) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in[(i*ldi)+j] = in[i*ldi];
        }
      }
    }
    if (use_bcast == COL_BCAST_IN0) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in[(i*ldi)+j] = in[j];
        }
      }
    }
    if (use_bcast == SCALAR_BCAST_IN0) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in[(i*ldi)+j] = in[0];
        }
      }
    }
    if (use_bcast == ROW_BCAST_IN1) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in2[(i*ldi)+j] = in2[i*ldi];
        }
      }
    }
    if (use_bcast == COL_BCAST_IN1) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in2[(i*ldi)+j] = in2[j];
        }
      }
    }
    if (use_bcast == SCALAR_BCAST_IN1) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in2[(i*ldi)+j] = in2[0];
        }
      }
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
    binary_op_f32_bf16_gold( M, 1, ldi, ldo, &in[(i*ldi)], &in2[(i*ldi)], &out_gold[(i*ldo)], op );
  }

  binary_param.in0.primary  = (void*)_in;
  binary_param.in1.primary  = (void*)_in2;
  binary_param.out.primary  = (void*)out;
  binary_flags = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  if (use_bcast != NO_BCAST) {
    if (use_bcast == ROW_BCAST_IN0) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0;
    }
    if (use_bcast == COL_BCAST_IN0) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
    }
    if (use_bcast == SCALAR_BCAST_IN0) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0;
    }
    if (use_bcast == ROW_BCAST_IN1) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1;
    }
    if (use_bcast == COL_BCAST_IN1) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1;
    }
    if (use_bcast == SCALAR_BCAST_IN1) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1;
    }
  }

  binary_shape.m = M;
  binary_shape.n = N;
  binary_shape.ldi = &ldi;
  binary_shape.ldi2 = &ldi;
  binary_shape.ldo = &ldo;
  binary_shape.in_type = LIBXSMM_DATATYPE_F32;
  binary_shape.out_type = LIBXSMM_DATATYPE_BF16;
  binary_shape.comp_type = LIBXSMM_DATATYPE_F32;

  libxsmm_meltwfunction_binary binary_kernel = libxsmm_dispatch_meltw_binary_v2( binary_type, binary_shape, binary_flags );
  if ( binary_kernel == NULL ) {
    fprintf( stderr, "JIT for BINARY TPP. Bailing...!\n");
    exit(-1);
  }
  binary_kernel( &binary_param );

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

  if ( norms_out.normf_rel > 0.005 ) {
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( f32out_gold );
  libxsmm_free( f32out );
  libxsmm_free( in );
  libxsmm_free( in2 );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS binary simple fp32 bf16\n");
  } else {
    printf("FAILURE binary simple fp32 bf16\n");
  }

  return ret;
}

int test_binary_op_bf16_f32( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo, unsigned int op, unsigned int use_bcast ) {
  libxsmm_bfloat16 *in, *_in, *in2, *_in2;
  float *out, *out_gold;
  unsigned int i, j;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_binary_param binary_param;
  libxsmm_meltw_binary_flags binary_flags;
  libxsmm_meltw_binary_shape binary_shape;
  libxsmm_matdiff_info norms_out;
  libxsmm_meltw_binary_type  binary_type;
  char opname[256];

  set_opname(op, opname);
  set_binarytype(op, &binary_type);

  if ( M > ldi && !(use_bcast == ROW_BCAST_IN0 || use_bcast == SCALAR_BCAST_IN0 || use_bcast == ROW_BCAST_IN1 || use_bcast == SCALAR_BCAST_IN1) ) {
    fprintf( stderr, "test_binary_%s_bf16_f32: ldi needs to be equal to or bigger than M\n", opname);
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_binary_%s_bf16_f32: ldo needs to be equal to or bigger than N\n", opname);
    exit(-1);
  }

  libxsmm_rng_set_seed(1);

  in        = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*LIBXSMM_MAX(M,ldi),   64);
  in2       = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*LIBXSMM_MAX(M,ldi),   64);
  out       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  out_gold  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  _in       = in;
  _in2      = in2;

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      union libxsmm_bfloat16_hp bf16_hp;
      bf16_hp.f = (float)libxsmm_rng_f64();
      in[(i*ldi)+j] = bf16_hp.i[1];
    }
  }

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ldi; ++j ) {
      union libxsmm_bfloat16_hp bf16_hp;
      bf16_hp.f = (float)libxsmm_rng_f64();
      in2[(i*ldi)+j] = bf16_hp.i[1];
    }
  }

  if (use_bcast != NO_BCAST) {
    if (use_bcast == ROW_BCAST_IN0) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in[(i*ldi)+j] = in[i*ldi];
        }
      }
    }
    if (use_bcast == COL_BCAST_IN0) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in[(i*ldi)+j] = in[j];
        }
      }
    }
    if (use_bcast == SCALAR_BCAST_IN0) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in[(i*ldi)+j] = in[0];
        }
      }
    }
    if (use_bcast == ROW_BCAST_IN1) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in2[(i*ldi)+j] = in2[i*ldi];
        }
      }
    }
    if (use_bcast == COL_BCAST_IN1) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in2[(i*ldi)+j] = in2[j];
        }
      }
    }
    if (use_bcast == SCALAR_BCAST_IN1) {
      for ( i = 0; i < N; ++i ) {
        for ( j = 0; j < LIBXSMM_MAX(M,ldi); ++j ) {
          in2[(i*ldi)+j] = in2[0];
        }
      }
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
    binary_op_bf16_f32_gold( M, 1, ldi, ldo, &in[(i*ldi)], &in2[(i*ldi)], &out_gold[(i*ldo)], op );
  }

  binary_param.in0.primary  = (void*)_in;
  binary_param.in1.primary  = (void*)_in2;
  binary_param.out.primary  = (void*)out;
  binary_flags = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  if (use_bcast != NO_BCAST) {
    if (use_bcast == ROW_BCAST_IN0) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0;
    }
    if (use_bcast == COL_BCAST_IN0) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
    }
    if (use_bcast == SCALAR_BCAST_IN0) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0;
    }
    if (use_bcast == ROW_BCAST_IN1) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1;
    }
    if (use_bcast == COL_BCAST_IN1) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1;
    }
    if (use_bcast == SCALAR_BCAST_IN1) {
      binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1;
    }
  }

  binary_shape.m = M;
  binary_shape.n = N;
  binary_shape.ldi = &ldi;
  binary_shape.ldi2 = &ldi;
  binary_shape.ldo = &ldo;
  binary_shape.in_type = LIBXSMM_DATATYPE_BF16;
  binary_shape.out_type = LIBXSMM_DATATYPE_F32;
  binary_shape.comp_type = LIBXSMM_DATATYPE_F32;

  libxsmm_meltwfunction_binary binary_kernel = libxsmm_dispatch_meltw_binary_v2( binary_type, binary_shape, binary_flags );
  if ( binary_kernel == NULL ) {
    fprintf( stderr, "JIT for BINARY TPP. Bailing...!\n");
    exit(-1);
  }
  binary_kernel( &binary_param );

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

  if ( norms_out.normf_rel > 0.005 ) {
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( in2 );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS binary simple bf16 fp32\n");
  } else {
    printf("FAILURE binary simple bf16 fp32\n");
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
  int res = EXIT_FAILURE;

  if ( argc != 10 ) {
    printf(" Error! Usage: %s [type] [use_bcast: 0/1/2/3/4/5/6] [prec_in: 4/2] [compute_prec: 4/2] [prec_out: 4/2] [M] [N] [ldi] [ldo]\n", argv[0] );
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

  valid_op = ( op == ADD_OP || op == SUB_OP || op == MUL_OP || op == DIV_OP || op == MULADD_OP ) ? 1 : 0;

  if (use_bcast != NO_BCAST) {
    if (use_bcast == ROW_BCAST_IN0) {
      printf("Using row broadcast for the input0 row-vector ...\n");
    }
    if (use_bcast == COL_BCAST_IN0) {
      printf("Using column broadcast for the input0 column-vector...\n");
    }
    if (use_bcast == SCALAR_BCAST_IN0) {
      printf("Using scalar broadcast for the input0 value...\n");
    }
    if (use_bcast == ROW_BCAST_IN1) {
      printf("Using row broadcast for the input1 row-vector ...\n");
    }
    if (use_bcast == COL_BCAST_IN1) {
      printf("Using column broadcast for the input1 column-vector...\n");
    }
    if (use_bcast == SCALAR_BCAST_IN1) {
      printf("Using scalar broadcast for the input1 value...\n");
    }
  }

  if ( valid_op > 0 && dtype_in == 4 && dtype_out == 4 && dtype_comp == 4 ) {
    printf("Testing binary F32 F32 %s - M=%i, N=%i, LDI=%i, LDO=%i\n", opname, M, N, ldi, ldo);
    res = test_binary_op_f32_f32( M, N, ldi, ldo, op, use_bcast);
  } else if ( valid_op > 0 && dtype_in == 2 && dtype_out == 2 && dtype_comp == 4 ) {
    printf("Testing binary BF16 BF16 %s - M=%i, N=%i, LDI=%i, LDO=%i\n", opname, M, N, ldi, ldo);
    res = test_binary_op_bf16_bf16( M, N, ldi, ldo, op, use_bcast);
  } else if ( valid_op > 0 && dtype_in == 4 && dtype_out == 2 && dtype_comp == 4 ) {
    printf("Testing binary F32 BF16 %s - M=%i, N=%i, LDI=%i, LDO=%i\n", opname, M, N, ldi, ldo);
    res = test_binary_op_f32_bf16( M, N, ldi, ldo, op, use_bcast);
  } else if ( valid_op > 0 && dtype_in == 2 && dtype_out == 4 && dtype_comp == 4 ) {
    printf("Testing binry BF16 F32 %s - M=%i, N=%i, LDI=%i, LDO=%i\n", opname, M, N, ldi, ldo);
    res = test_binary_op_bf16_f32( M, N, ldi, ldo, op, use_bcast);
  } else {
    printf(" Error! Usage: %s [type] [use_bcast: 0/1/2/3/4/5/6]] [prec_in: 4/2] compute_prec: 4 [prec_out: 4/2] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  return res;
}
