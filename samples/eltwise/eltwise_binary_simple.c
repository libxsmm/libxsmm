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


LIBXSMM_INLINE
void adjust_inputs_for_hf8_div( libxsmm_datatype dtype_in, void *in, libxsmm_datatype dtype_in1,  void* in2, libxsmm_blasint ldi, libxsmm_blasint N, unsigned int use_bcast ) {
  float *in_f  = (float*) libxsmm_aligned_malloc(sizeof(float)*N*ldi, 64);
  float *in2_f  = (float*) libxsmm_aligned_malloc(sizeof(float)*N*ldi, 64);
  float *in_use;
  float *in2_use;
  libxsmm_blasint i, j;

  if (dtype_in == LIBXSMM_DATATYPE_HF8) {
    libxsmm_convert_hf8_f32( (libxsmm_hfloat8*)in, in_f, N*ldi );
    in_use = in_f;
  } else {
    in_use = (float*)in;
  }

  if (dtype_in1 == LIBXSMM_DATATYPE_HF8) {
    libxsmm_convert_hf8_f32( (libxsmm_hfloat8*)in2, in2_f, N*ldi );
    in2_use = in2_f;
  } else {
    in2_use = (float*)in2;
  }

  for (j = 0; j < N; j++) {
    for (i = 0; i < ldi; i++) {
      if (in2_use[j*ldi+i] == 0.0f) {
        in2_use[j*ldi+i] = 1.0;
      }
      in_use[j*ldi+i] = 2.f * in2_use[j*ldi+i];
      if (LIBXSMM_ABS(in_use[j*ldi+i]) > 400.f) {
        in_use[j*ldi+i] = 400.f;
        in2_use[j*ldi+i] = 200.f;
      }
      if (use_bcast > 0) {
        in2_use[j*ldi+i] = 1.f;
      }
    }
  }

  if (dtype_in == LIBXSMM_DATATYPE_HF8) {
    libxsmm_rne_convert_fp32_hf8( in_use, (libxsmm_hfloat8*)in, N*ldi );
  }

  if (dtype_in1 == LIBXSMM_DATATYPE_HF8) {
    libxsmm_rne_convert_fp32_hf8( in2_use, (libxsmm_hfloat8*)in2, N*ldi );
  }

  libxsmm_free(in_f);
  libxsmm_free(in2_f);
}

LIBXSMM_INLINE
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

LIBXSMM_INLINE
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

LIBXSMM_INLINE
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

LIBXSMM_INLINE
void binary_op_gold(const libxsmm_blasint M, const libxsmm_blasint N, const libxsmm_blasint ldi0, const libxsmm_blasint ldi1, const libxsmm_blasint ldo,
                    const void *in0, const void *in1, char *out, const unsigned int op,
                    const libxsmm_datatype dtype_in0, const libxsmm_datatype dtype_in1, const libxsmm_datatype dtype_out, const libxsmm_datatype dtype_comp) {
  libxsmm_blasint i,j;
  LIBXSMM_UNUSED(ldi1);

  if ( dtype_comp == LIBXSMM_DATATYPE_F32 ) {
    float in1_value = 0;
    float in0_value = 0;
    float out_value = 0;
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        if ( dtype_in0 == LIBXSMM_DATATYPE_F32 ) {
          const float* f_in0 = (const float*)in0;
          in0_value = f_in0[(j*ldi0) + i];
        } else if ( dtype_in0 == LIBXSMM_DATATYPE_BF16 ) {
          const libxsmm_bfloat16* bf16_in0 = (const libxsmm_bfloat16*)in0;
          libxsmm_convert_bf16_f32( &(bf16_in0[(j*ldi0) + i]), &in0_value, 1 );
        } else if ( dtype_in0 == LIBXSMM_DATATYPE_F16 ) {
          const libxsmm_float16* f16_in0 = (const libxsmm_float16*)in0;
          libxsmm_convert_f16_f32( &(f16_in0[(j*ldi0) + i]), &in0_value, 1 );
        } else if ( dtype_in0 == LIBXSMM_DATATYPE_BF8 ) {
          const libxsmm_bfloat8* bf8_in0 = (const libxsmm_bfloat8*)in0;
          libxsmm_convert_bf8_f32( &(bf8_in0[(j*ldi0) + i]), &in0_value, 1 );
        } else if ( dtype_in0 == LIBXSMM_DATATYPE_HF8 ) {
          const libxsmm_hfloat8* hf8_in0 = (const libxsmm_hfloat8*)in0;
          libxsmm_convert_hf8_f32( &(hf8_in0[(j*ldi0) + i]), &in0_value, 1 );
        } else {
          /* shouldn't happen */
        }

        if ( dtype_in1 == LIBXSMM_DATATYPE_F32 ) {
          const float* f_in1 = (const float*)in1;
          in1_value = f_in1[(j*ldi1) + i];
        } else if ( dtype_in1 == LIBXSMM_DATATYPE_BF16 ) {
          const libxsmm_bfloat16* bf16_in1 = (const libxsmm_bfloat16*)in1;
          libxsmm_convert_bf16_f32( &(bf16_in1[(j*ldi1) + i]), &in1_value, 1 );
        } else if ( dtype_in1 == LIBXSMM_DATATYPE_F16 ) {
          const libxsmm_float16* f16_in1 = (const libxsmm_float16*)in1;
          libxsmm_convert_f16_f32( &(f16_in1[(j*ldi1) + i]), &in1_value, 1 );
        } else if ( dtype_in1 == LIBXSMM_DATATYPE_BF8 ) {
          const libxsmm_bfloat8* bf8_in1 = (const libxsmm_bfloat8*)in1;
          libxsmm_convert_bf8_f32( &(bf8_in1[(j*ldi1) + i]), &in1_value, 1 );
        } else if ( dtype_in1 == LIBXSMM_DATATYPE_HF8 ) {
          const libxsmm_hfloat8* hf8_in1 = (const libxsmm_hfloat8*)in1;
          libxsmm_convert_hf8_f32( &(hf8_in1[(j*ldi1) + i]), &in1_value, 1 );
        } else {
          /* shouldn't happen */
        }

        if ( dtype_out == LIBXSMM_DATATYPE_F32 ) {
          const float* f_out = (const float*)out;
          out_value = f_out[(j*ldo) + i];
        } else if ( dtype_out == LIBXSMM_DATATYPE_BF16 ) {
          const libxsmm_bfloat16* bf16_out = (const libxsmm_bfloat16*)out;
          libxsmm_convert_bf16_f32( &(bf16_out[(j*ldo) + i]), &out_value, 1 );
        } else if ( dtype_out == LIBXSMM_DATATYPE_F16 ) {
          const libxsmm_float16* f16_out = (const libxsmm_float16*)out;
          libxsmm_convert_f16_f32( &(f16_out[(j*ldo) + i]), &out_value, 1 );
        } else if ( dtype_out == LIBXSMM_DATATYPE_BF8 ) {
          const libxsmm_bfloat8* bf8_out = (const libxsmm_bfloat8*)out;
          libxsmm_convert_bf8_f32( &(bf8_out[(j*ldo) + i]), &out_value, 1 );
        } else if ( dtype_out == LIBXSMM_DATATYPE_HF8 ) {
          const libxsmm_hfloat8* hf8_out = (const libxsmm_hfloat8*)out;
          libxsmm_convert_hf8_f32( &(hf8_out[(j*ldo) + i]), &out_value, 1 );
        } else {
          /* shouldn't happen */
        }

        out_value = fp32_binary_compute(in0_value, in1_value, out_value, op);

        if ( dtype_out == LIBXSMM_DATATYPE_F32 ) {
          float* f_out = (float*)out;
          f_out[(j*ldo) + i] = out_value;
        } else if ( dtype_out == LIBXSMM_DATATYPE_BF16 ) {
          libxsmm_bfloat16* bf16_out = (libxsmm_bfloat16*)out;
          libxsmm_rne_convert_fp32_bf16(&out_value, &(bf16_out[(j*ldo) + i]), 1);
        } else if ( dtype_out == LIBXSMM_DATATYPE_F16 ) {
          libxsmm_float16* f16_out = (libxsmm_float16*)out;
          libxsmm_rne_convert_fp32_f16(&out_value, &(f16_out[(j*ldo) + i]), 1);
        } else if ( dtype_out == LIBXSMM_DATATYPE_BF8 ) {
          libxsmm_bfloat8* bf8_out = (libxsmm_bfloat8*)out;
          libxsmm_rne_convert_fp32_bf8(&out_value, &(bf8_out[(j*ldo) + i]), 1);
        } else if ( dtype_out == LIBXSMM_DATATYPE_HF8 ) {
          libxsmm_hfloat8* hf8_out = (libxsmm_hfloat8*)out;
          libxsmm_rne_convert_fp32_hf8(&out_value, &(hf8_out[(j*ldo) + i]), 1);
        } else {
          /* shouldn't happen */
        }
      }
    }
  } else {
    /* shouodn't happen */
  }
}

LIBXSMM_INLINE
int test_binary_op( const libxsmm_blasint M, const libxsmm_blasint N, const libxsmm_blasint ldi, const libxsmm_blasint ldo, const unsigned int op, const unsigned int use_bcast,
                    const libxsmm_datatype dtype_in, const libxsmm_datatype dtype_in1, const libxsmm_datatype dtype_out, const libxsmm_datatype dtype_comp ) {
  char *in, *_in, *in2, *_in2;
  char *out, *out_gold;
  int ret = EXIT_SUCCESS;
  libxsmm_meltwfunction_binary binary_kernel;
  libxsmm_meltw_binary_param binary_param /*= { 0 }*/;
  libxsmm_meltw_binary_flags binary_flags;
  libxsmm_meltw_binary_shape binary_shape = libxsmm_create_meltw_binary_shape( M, N, ldi, ldi, ldo, dtype_in, dtype_in1, dtype_out, dtype_comp );
  libxsmm_matdiff_info norms_out;
  libxsmm_meltw_binary_type  binary_type;
  char opname[256];

  set_opname(op, opname);
  set_binarytype(op, &binary_type);

  if ( M > ldi && !(use_bcast == ROW_BCAST_IN0 || use_bcast == SCALAR_BCAST_IN0 || use_bcast == ROW_BCAST_IN1 || use_bcast == SCALAR_BCAST_IN1) ) {
    fprintf( stderr, "test_binary_%s %i %i %i %i: ldi needs to be equal to or bigger than M\n", opname, (int)dtype_in, (int)dtype_in1, (int)dtype_out, (int)dtype_comp);
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_binary_%s %i %i %i %i: ldo needs to be equal to or bigger than N\n", opname, (int)dtype_in, (int)dtype_in1, (int)dtype_out, (int)dtype_comp);
    exit(-1);
  }

  libxsmm_rng_set_seed(1);

  in        = (char*) libxsmm_aligned_malloc((size_t)LIBXSMM_TYPESIZE(dtype_in)*N*LIBXSMM_MAX(M,ldi), 64);
  in2       = (char*) libxsmm_aligned_malloc((size_t)LIBXSMM_TYPESIZE(dtype_in1)*N*LIBXSMM_MAX(M,ldi), 64);
  out       = (char*) libxsmm_aligned_malloc((size_t)LIBXSMM_TYPESIZE(dtype_out)*N*ldo, 64);
  out_gold  = (char*) libxsmm_aligned_malloc((size_t)LIBXSMM_TYPESIZE(dtype_out)*N*ldo, 64);
  _in       = in;
  _in2      = in2;

  /* init in */
  init_random_matrix( dtype_in,  in,       1, ldi, N, 0 );
  init_random_matrix( dtype_in1,  in2,      1, ldi, N, 0 );
  init_zero_matrix(   dtype_out, out,      1, ldo, N );
  init_zero_matrix(   dtype_out, out_gold, 1, ldo, N );

  if ((op == DIV_OP) && ((dtype_in == LIBXSMM_DATATYPE_HF8) || (dtype_in1 == LIBXSMM_DATATYPE_HF8) || (dtype_out == LIBXSMM_DATATYPE_HF8))) {
    adjust_inputs_for_hf8_div( dtype_in, in, dtype_in1,  in2, ldi, N, use_bcast  );
  }

  if (use_bcast != NO_BCAST) {
    if (use_bcast == ROW_BCAST_IN0) {
      apply_row_bcast_matrix( dtype_in, in, ldi, M, N );
    }
    if (use_bcast == COL_BCAST_IN0) {
      apply_col_bcast_matrix( dtype_in, in, ldi, M, N );
    }
    if (use_bcast == SCALAR_BCAST_IN0) {
      apply_scalar_bcast_matrix( dtype_in, in, ldi, M, N );
    }
    if (use_bcast == ROW_BCAST_IN1) {
      apply_row_bcast_matrix( dtype_in1, in2, ldi, M, N );
    }
    if (use_bcast == COL_BCAST_IN1) {
      apply_col_bcast_matrix( dtype_in1, in2, ldi, M, N );
    }
    if (use_bcast == SCALAR_BCAST_IN1) {
      apply_scalar_bcast_matrix( dtype_in1, in2, ldi, M, N );
    }
  }

  /* compute out_gold */
  binary_op_gold( M, N, ldi, ldi, ldo, in, in2, out_gold, op, dtype_in, dtype_in1, dtype_out, dtype_comp );

  /* use jited transpose */
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

  binary_kernel = libxsmm_dispatch_meltw_binary_v2( binary_type, binary_shape, binary_flags );
  if ( binary_kernel == NULL ) {
    fprintf( stderr, "JIT for BINARY TPP. Bailing...!\n");
    exit(-1);
  }
  binary_kernel( &binary_param );

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

  if ( norms_out.normf_rel > 0.00001 ) {
    ret = EXIT_FAILURE;
  }

  benchmark_binary(binary_type, binary_shape, binary_flags, binary_param);

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( in2 );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS binary simple %i %i %i %i\n", (int)dtype_in, (int)dtype_in1, (int)dtype_out, (int)dtype_comp);
  } else {
    printf("FAILURE binary simple %i %i %i %i\n", (int)dtype_in, (int)dtype_in1, (int)dtype_out, (int)dtype_comp);
  }

  return ret;
}

int main( int argc, char* argv[] ) {
  char* dt_in0;
  char* dt_in1;
  char* dt_out;
  char* dt_comp;
  libxsmm_datatype dtype_in0;
  libxsmm_datatype dtype_in1;
  libxsmm_datatype dtype_out;
  libxsmm_datatype dtype_comp;
  libxsmm_blasint op;
  libxsmm_blasint use_bcast;
  libxsmm_blasint M;
  libxsmm_blasint N;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;
  libxsmm_blasint valid_op;
  char opname[256];
  int res = EXIT_FAILURE;

  if ( argc != 11 ) {
    printf(" Error! Usage: %s [type] [use_bcast: 0/1/2/3/4/5/6] [prec_in0: F32/BF16/F16/BF8/HF8] [prec_in1: F32/BF16/F16/BF8/HF8] [compute_prec: F32] [prec_out: F32/BF16/F16/BF8/HF8] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  op         = atoi(argv[1]);
  use_bcast  = atoi(argv[2]);
  dt_in0     = argv[3];
  dt_in1     = argv[4];
  dt_comp    = argv[5];
  dt_out     = argv[6];
  M          = atoi(argv[7]);
  N          = atoi(argv[8]);
  ldi        = atoi(argv[9]);
  ldo        = atoi(argv[10]);

  dtype_in0  = char_to_libxsmm_datatype( dt_in0 );
  dtype_in1  = char_to_libxsmm_datatype( dt_in1 );
  dtype_out  = char_to_libxsmm_datatype( dt_out );
  dtype_comp = char_to_libxsmm_datatype( dt_comp );

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

  if ( valid_op > 0 ) {
    if ( ( (dtype_in0 == LIBXSMM_DATATYPE_F32 ) && (dtype_in1 == LIBXSMM_DATATYPE_F32 ) && (dtype_out == LIBXSMM_DATATYPE_F32 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         /* BF16 */
         ( (dtype_in0 == LIBXSMM_DATATYPE_F32 ) && (dtype_in1 == LIBXSMM_DATATYPE_F32 ) && (dtype_out == LIBXSMM_DATATYPE_BF16) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_F32 ) && (dtype_in1 == LIBXSMM_DATATYPE_BF16) && (dtype_out == LIBXSMM_DATATYPE_F32 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_F32 ) && (dtype_in1 == LIBXSMM_DATATYPE_BF16) && (dtype_out == LIBXSMM_DATATYPE_BF16) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_BF16) && (dtype_in1 == LIBXSMM_DATATYPE_F32 ) && (dtype_out == LIBXSMM_DATATYPE_F32 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_BF16) && (dtype_in1 == LIBXSMM_DATATYPE_F32 ) && (dtype_out == LIBXSMM_DATATYPE_BF16) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_BF16) && (dtype_in1 == LIBXSMM_DATATYPE_BF16) && (dtype_out == LIBXSMM_DATATYPE_F32 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_BF16) && (dtype_in1 == LIBXSMM_DATATYPE_BF16) && (dtype_out == LIBXSMM_DATATYPE_BF16) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         /* F16 */
         ( (dtype_in0 == LIBXSMM_DATATYPE_F32 ) && (dtype_in1 == LIBXSMM_DATATYPE_F32 ) && (dtype_out == LIBXSMM_DATATYPE_F16 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_F32 ) && (dtype_in1 == LIBXSMM_DATATYPE_F16 ) && (dtype_out == LIBXSMM_DATATYPE_F32 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_F32 ) && (dtype_in1 == LIBXSMM_DATATYPE_F16 ) && (dtype_out == LIBXSMM_DATATYPE_F16 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_F16 ) && (dtype_in1 == LIBXSMM_DATATYPE_F32 ) && (dtype_out == LIBXSMM_DATATYPE_F32 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_F16 ) && (dtype_in1 == LIBXSMM_DATATYPE_F32 ) && (dtype_out == LIBXSMM_DATATYPE_F16 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_F16 ) && (dtype_in1 == LIBXSMM_DATATYPE_F16 ) && (dtype_out == LIBXSMM_DATATYPE_F32 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_F16 ) && (dtype_in1 == LIBXSMM_DATATYPE_F16 ) && (dtype_out == LIBXSMM_DATATYPE_F16 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         /* BF8 */
         ( (dtype_in0 == LIBXSMM_DATATYPE_F32 ) && (dtype_in1 == LIBXSMM_DATATYPE_F32 ) && (dtype_out == LIBXSMM_DATATYPE_BF8 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_F32 ) && (dtype_in1 == LIBXSMM_DATATYPE_BF8 ) && (dtype_out == LIBXSMM_DATATYPE_F32 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_F32 ) && (dtype_in1 == LIBXSMM_DATATYPE_BF8 ) && (dtype_out == LIBXSMM_DATATYPE_BF8 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_BF8 ) && (dtype_in1 == LIBXSMM_DATATYPE_F32 ) && (dtype_out == LIBXSMM_DATATYPE_F32 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_BF8 ) && (dtype_in1 == LIBXSMM_DATATYPE_F32 ) && (dtype_out == LIBXSMM_DATATYPE_BF8 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_BF8 ) && (dtype_in1 == LIBXSMM_DATATYPE_BF8 ) && (dtype_out == LIBXSMM_DATATYPE_F32 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_BF8 ) && (dtype_in1 == LIBXSMM_DATATYPE_BF8 ) && (dtype_out == LIBXSMM_DATATYPE_BF8 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         /* HF8 */
         ( (dtype_in0 == LIBXSMM_DATATYPE_F32 ) && (dtype_in1 == LIBXSMM_DATATYPE_F32 ) && (dtype_out == LIBXSMM_DATATYPE_HF8 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_F32 ) && (dtype_in1 == LIBXSMM_DATATYPE_HF8 ) && (dtype_out == LIBXSMM_DATATYPE_F32 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_F32 ) && (dtype_in1 == LIBXSMM_DATATYPE_HF8 ) && (dtype_out == LIBXSMM_DATATYPE_HF8 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_HF8 ) && (dtype_in1 == LIBXSMM_DATATYPE_F32 ) && (dtype_out == LIBXSMM_DATATYPE_F32 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_HF8 ) && (dtype_in1 == LIBXSMM_DATATYPE_F32 ) && (dtype_out == LIBXSMM_DATATYPE_HF8 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_HF8 ) && (dtype_in1 == LIBXSMM_DATATYPE_HF8 ) && (dtype_out == LIBXSMM_DATATYPE_F32 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_HF8 ) && (dtype_in1 == LIBXSMM_DATATYPE_HF8 ) && (dtype_out == LIBXSMM_DATATYPE_HF8 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ) {
      printf("Testing binary (in0:%s in1:%s out:%s comp:%s) %s - M=%i, N=%i, LDI=%i, LDO=%i\n",
        libxsmm_get_typename(dtype_in0), libxsmm_get_typename(dtype_in1), libxsmm_get_typename(dtype_out), libxsmm_get_typename(dtype_comp), opname, M, N, ldi, ldo);
      res = test_binary_op( M, N, ldi, ldo, op, use_bcast, dtype_in0, dtype_in1, dtype_out, dtype_comp);
    } else {
      printf(" Error! Usage: %s [type] [use_bcast: 0/1/2/3/4/5/6] [prec_in0: F32/BF16/F16/BF8/HF8] [prec_in1: F32/BF16/F16/BF8/HF8] [compute_prec: F32] [prec_out: F32/BF16/F16/BF8/HF8] [M] [N] [ldi] [ldo]\n", argv[0] );
      exit(-1);
    }
  } else {
    printf(" Error! Usage: %s [type] [use_bcast: 0/1/2/3/4/5/6] [prec_in0: F32/BF16/F16/BF8/HF8] [prec_in1: F32/BF16/F16/BF8/HF8] [compute_prec: F32] [prec_out: F32/BF16/F16/BF8/HF8] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  return res;
}
