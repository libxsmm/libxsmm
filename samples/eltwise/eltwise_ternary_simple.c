/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
*               Friedrich Schiller University Jena - All rights reserved.     *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas
******************************************************************************/
#include "eltwise_common.h"

#define RND_RNE 0
#define RND_STOCHASTIC 1

#define NO_BCAST 0
#define ROW_BCAST_IN0 1
#define COL_BCAST_IN0 2
#define SCALAR_BCAST_IN0 3
#define ROW_BCAST_IN1 4
#define COL_BCAST_IN1 5
#define SCALAR_BCAST_IN1 6
#define ROW_BCAST_IN2 7
#define COL_BCAST_IN2 8
#define SCALAR_BCAST_IN2 9

#define SELECT_OP     1

#if 1
#define USE_ZERO_RNG_STATE_UNITTEST
#endif

LIBXSMM_INLINE
void set_opname(unsigned int op, char *opname) {
  if ( op == SELECT_OP ) {
    sprintf(opname, "select");
  } else {
     printf("Invalid OP\n");
    exit(-1);
  }
}

LIBXSMM_INLINE
void set_ternarytype(unsigned int op, libxsmm_meltw_ternary_type *type) {
  libxsmm_meltw_ternary_type  ternary_type;

  if ( op == SELECT_OP ) {
    ternary_type = LIBXSMM_MELTW_TYPE_TERNARY_SELECT;
  } else {
    printf("Invalid OP\n");
    exit(-1);
  }

  *type = ternary_type;
}

LIBXSMM_INLINE
unsigned char extract_bit(const char *bit_matrix, libxsmm_blasint i, libxsmm_blasint j, libxsmm_blasint ld) {
  unsigned char result = 0;
  libxsmm_blasint byte_load = i/8;
  libxsmm_blasint pos_in_byte = i%8;
  char byte_loaded = bit_matrix[byte_load + j * (ld/8)];
  result = ((unsigned char)(byte_loaded << (7-pos_in_byte))) >> 7;
  result = (result == 0) ? 0 : 1;
  return result;
}

LIBXSMM_INLINE
void ternary_op_gold(const libxsmm_blasint M, const libxsmm_blasint N, const libxsmm_blasint ldi0, const libxsmm_blasint ldi1, const libxsmm_blasint ldi2, const libxsmm_blasint ldo,
                    const void *in0, const void *in1, const void *in2, char *out, const unsigned int op,
                    const libxsmm_datatype dtype_in0, const libxsmm_datatype dtype_in1, const libxsmm_datatype dtype_in2, const libxsmm_datatype dtype_out, const libxsmm_datatype dtype_comp, libxsmm_bitfield flags, void *rng_state) {
  libxsmm_blasint i,j;
  LIBXSMM_UNUSED(ldi1);

  if ( dtype_comp == LIBXSMM_DATATYPE_F32 ) {
    float in1_value = 0;
    float in0_value = 0;
    float out_value = 0;
    unsigned char bit_val = 0;
    unsigned int seed_idx = 0;
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
          /* should not happen */
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
          /* should not happen */
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
          /* should not happen */
        }

        if (op == SELECT_OP) {
          bit_val = extract_bit(in2, i, j, ldi2);
          out_value = (bit_val == 0) ? in0_value : in1_value;
        } else {

        }

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
          if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_STOCHASTIC_ROUND) > 0 ) {
            libxsmm_stochastic_convert_fp32_bf8(&out_value, &(bf8_out[(j*ldo) + i]), 1, rng_state, seed_idx);
            seed_idx++;
          } else {
            libxsmm_rne_convert_fp32_bf8(&out_value, &(bf8_out[(j*ldo) + i]), 1);
          }
        } else if ( dtype_out == LIBXSMM_DATATYPE_HF8 ) {
          libxsmm_hfloat8* hf8_out = (libxsmm_hfloat8*)out;
          libxsmm_rne_convert_fp32_hf8(&out_value, &(hf8_out[(j*ldo) + i]), 1);
        } else {
          /* should not happen */
        }
      }
    }
  } else if ( dtype_comp == LIBXSMM_DATATYPE_F64 ) {
    double in1_value = 0;
    double in0_value = 0;
    double out_value = 0;
    unsigned char bit_val = 0;
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        const double* d_in0 = (const double*)in0;
        const double* d_in1 = (const double*)in1;
        double* d_out = (double*)out;
        in0_value = d_in0[(j*ldi0) + i];
        in1_value = d_in1[(j*ldi1) + i];
        out_value = d_out[(j*ldo) + i];
        if (op == SELECT_OP) {
          bit_val = extract_bit(in2, i, j, ldi2);
          out_value = (bit_val == 0) ? in0_value : in1_value;
        } else {

        }
        d_out[(j*ldo) + i] = out_value;
      }
    }
  } else {
    /* should not happen */
  }
}

LIBXSMM_INLINE
int test_ternary_op( const libxsmm_blasint M, const libxsmm_blasint N, const libxsmm_blasint ldi, const libxsmm_blasint ldo, const unsigned int op, const unsigned int use_bcast,
                    const libxsmm_datatype dtype_in, const libxsmm_datatype dtype_in1, const libxsmm_datatype dtype_in2, const libxsmm_datatype dtype_out, const libxsmm_datatype dtype_comp, const unsigned int rnd_mode ) {
  char *in, *_in, *in2, *_in2, *in3, *_in3;
  char *out, *out_gold;
  unsigned int *rng_state = NULL;
  unsigned int *rng_state_gold = NULL;

  int ret = EXIT_SUCCESS;
  libxsmm_meltwfunction_ternary ternary_kernel;
  libxsmm_meltw_ternary_param ternary_param /*= { 0 }*/;
  libxsmm_bitfield ternary_flags;
  libxsmm_meltw_ternary_shape ternary_shape = libxsmm_create_meltw_ternary_shape( M, N, ldi, ldi, ldi, ldo, dtype_in, dtype_in1, dtype_in2, dtype_out, dtype_comp );
  libxsmm_matdiff_info norms_out;
  libxsmm_meltw_ternary_type  ternary_type;
  libxsmm_blasint l_ld2 = (op == SELECT_OP) ? LIBXSMM_UPDIV(ldi, 16)*16 : ldi;

  char opname[256];

  set_opname(op, opname);
  set_ternarytype(op, &ternary_type);

  if ( M > ldi && !(use_bcast == ROW_BCAST_IN0 || use_bcast == SCALAR_BCAST_IN0 || use_bcast == ROW_BCAST_IN1 || use_bcast == SCALAR_BCAST_IN1) ) {
    fprintf( stderr, "test_ternary_%s %i %i %i %i %i: ldi needs to be equal to or bigger than M\n", opname, (int)dtype_in, (int)dtype_in1, (int)dtype_in2, (int)dtype_out, (int)dtype_comp);
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_ternry_%s %i %i %i %i %i: ldo needs to be equal to or bigger than N\n", opname, (int)dtype_in, (int)dtype_in1, (int)dtype_in2, (int)dtype_out, (int)dtype_comp);
    exit(-1);
  }

  libxsmm_rng_set_seed(1);

  in        = (char*) libxsmm_aligned_malloc((size_t)LIBXSMM_TYPESIZE(dtype_in)*N*LIBXSMM_MAX(M,ldi), 64);
  in2       = (char*) libxsmm_aligned_malloc((size_t)LIBXSMM_TYPESIZE(dtype_in1)*N*LIBXSMM_MAX(M,ldi), 64);
  if (op == SELECT_OP) {
    in3       = (char*) libxsmm_aligned_malloc((size_t)N*LIBXSMM_MAX(LIBXSMM_UPDIV(M,16)*16,l_ld2), 64);
  } else {
    in3       = (char*) libxsmm_aligned_malloc((size_t)LIBXSMM_TYPESIZE(dtype_in2)*N*LIBXSMM_MAX(M,ldi), 64);
  }
  out       = (char*) libxsmm_aligned_malloc((size_t)LIBXSMM_TYPESIZE(dtype_out)*N*ldo, 64);
  out_gold  = (char*) libxsmm_aligned_malloc((size_t)LIBXSMM_TYPESIZE(dtype_out)*N*ldo, 64);
  _in       = in;
  _in2      = in2;
  _in3      = in3;


  /* init in */
  init_random_matrix( dtype_in,  in,       1, ldi, N, 0 );
  init_random_matrix( dtype_in1, in2,      1, ldi, N, 0 );
  init_random_matrix( (op == SELECT_OP) ? LIBXSMM_DATATYPE_BF8 : dtype_in2, in3,  1, l_ld2, N, 0 );
  init_zero_matrix(   dtype_out, out,      1, ldo, N );
  init_zero_matrix(   dtype_out, out_gold, 1, ldo, N );

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

  if (rnd_mode == RND_STOCHASTIC) {
    ternary_flags = LIBXSMM_MELTW_FLAG_TERNARY_STOCHASTIC_ROUND;
    rng_state = libxsmm_rng_create_extstate( 555 );
    rng_state_gold = libxsmm_rng_create_extstate( 555 );
#ifdef USE_ZERO_RNG_STATE_UNITTEST
    memset( (void*)rng_state, 0, libxsmm_rng_get_extstate_size() );
    memset( (void*)rng_state_gold, 0, libxsmm_rng_get_extstate_size() );
#endif
    ternary_param.op.secondary = (void*)rng_state;
  } else {
    ternary_flags = LIBXSMM_MELTW_FLAG_TERNARY_NONE;
  }

  if (op == SELECT_OP) {
    ternary_flags |= LIBXSMM_MELTW_FLAG_TERNARY_BITMASK_2BYTEMULT;
  }

  /* compute out_gold */
  ternary_op_gold( M, N, ldi, ldi, l_ld2, ldo, in, in2, in3, out_gold, op, dtype_in, dtype_in1, dtype_in2, dtype_out, dtype_comp, ternary_flags, rng_state_gold );

  ternary_param.in0.primary  = (void*)_in;
  ternary_param.in1.primary  = (void*)_in2;
  ternary_param.in2.primary  = (void*)_in3;
  ternary_param.out.primary  = (void*)out;

  if (use_bcast != NO_BCAST) {
    if (use_bcast == ROW_BCAST_IN0) {
      ternary_flags |= LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_0;
    }
    if (use_bcast == COL_BCAST_IN0) {
      ternary_flags |= LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0;
    }
    if (use_bcast == SCALAR_BCAST_IN0) {
      ternary_flags |= LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_0;
    }
    if (use_bcast == ROW_BCAST_IN1) {
      ternary_flags |= LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_1;
    }
    if (use_bcast == COL_BCAST_IN1) {
      ternary_flags |= LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1;
    }
    if (use_bcast == SCALAR_BCAST_IN1) {
      ternary_flags |= LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1;
    }
    if (use_bcast == ROW_BCAST_IN2) {
      ternary_flags |= LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_2;
    }
    if (use_bcast == COL_BCAST_IN2) {
      ternary_flags |= LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2;
    }
    if (use_bcast == SCALAR_BCAST_IN2) {
      ternary_flags |= LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2;
    }
  }

  ternary_kernel = libxsmm_dispatch_meltw_ternary( ternary_type, ternary_shape, ternary_flags );
  if ( ternary_kernel == NULL ) {
    fprintf( stderr, "JIT for TERNARY TPP. Bailing...!\n");
    exit(-1);
  }
  ternary_kernel( &ternary_param );

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

  if (rnd_mode == RND_STOCHASTIC) {
    libxsmm_rng_destroy_extstate( rng_state );
    libxsmm_rng_destroy_extstate( rng_state_gold );
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( in2 );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS ternary simple %i %i %i %i %i\n", (int)dtype_in, (int)dtype_in1, (int)dtype_in2, (int)dtype_out, (int)dtype_comp);
  } else {
    printf("FAILURE ternary simple %i %i %i %i %i\n", (int)dtype_in, (int)dtype_in1, (int)dtype_in2, (int)dtype_out, (int)dtype_comp);
  }

  return ret;
}

int main( int argc, char* argv[] ) {
  char* dt_in0;
  char* dt_in1;
  char* dt_in2;
  char* dt_out;
  char* dt_comp;
  libxsmm_datatype dtype_in0;
  libxsmm_datatype dtype_in1;
  libxsmm_datatype dtype_in2;
  libxsmm_datatype dtype_out;
  libxsmm_datatype dtype_comp;
  libxsmm_blasint op;
  libxsmm_blasint use_bcast;
  libxsmm_blasint M;
  libxsmm_blasint N;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;
  libxsmm_blasint valid_op;
  unsigned int rnd_mode = RND_RNE;
  char opname[256];
  int res = EXIT_FAILURE;

  if ( argc != 12 && argc != 13 ) {
    printf(" Error! Usage: %s [type] [use_bcast: 0/1/2/3/4/5/6] [prec_in0: F32/BF16/F16/BF8/HF8] [prec_in1: F32/BF16/F16/BF8/HF8] [prec_in2: F32/BF16/F16/BF8/HF8] [compute_prec: F32] [prec_out: F32/BF16/F16/BF8/HF8] [M] [N] [ldi] [ldo] [Opt: rnd_mode: 0/1]\n", argv[0] );
    exit(-1);
  }

  op         = atoi(argv[1]);
  use_bcast  = atoi(argv[2]);
  dt_in0     = argv[3];
  dt_in1     = argv[4];
  dt_in2     = argv[5];
  dt_comp    = argv[6];
  dt_out     = argv[7];
  M          = atoi(argv[8]);
  N          = atoi(argv[9]);
  ldi        = atoi(argv[10]);
  ldo        = atoi(argv[11]);

  if (argc > 12 ) {
    rnd_mode   = atoi(argv[12]);
  }

  dtype_in0  = char_to_libxsmm_datatype( dt_in0 );
  dtype_in1  = char_to_libxsmm_datatype( dt_in1 );
  dtype_in2  = char_to_libxsmm_datatype( dt_in2 );
  dtype_out  = char_to_libxsmm_datatype( dt_out );
  dtype_comp = char_to_libxsmm_datatype( dt_comp );

  set_opname(op, opname);

  valid_op = ( op == SELECT_OP ) ? 1 : 0;

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
    if (use_bcast == ROW_BCAST_IN2) {
      printf("Using row broadcast for the input2 row-vector ...\n");
    }
    if (use_bcast == COL_BCAST_IN2) {
      printf("Using column broadcast for the input2 column-vector...\n");
    }
    if (use_bcast == SCALAR_BCAST_IN2) {
      printf("Using scalar broadcast for the input2 value...\n");
    }
  }

  if ( valid_op > 0 ) {
    if ( op == SELECT_OP && (dtype_in2 == LIBXSMM_DATATYPE_IMPLICIT) && (
         ( (dtype_in0 == LIBXSMM_DATATYPE_F32 ) && (dtype_in1 == LIBXSMM_DATATYPE_F32 ) && (dtype_out == LIBXSMM_DATATYPE_F32 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ||
         ( (dtype_in0 == LIBXSMM_DATATYPE_F64 ) && (dtype_in1 == LIBXSMM_DATATYPE_F64 ) && (dtype_out == LIBXSMM_DATATYPE_F64 ) && (dtype_comp == LIBXSMM_DATATYPE_F64 ) ) ||
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
         ( (dtype_in0 == LIBXSMM_DATATYPE_HF8 ) && (dtype_in1 == LIBXSMM_DATATYPE_HF8 ) && (dtype_out == LIBXSMM_DATATYPE_HF8 ) && (dtype_comp == LIBXSMM_DATATYPE_F32 ) ) ) ){
      printf("Testing ternary (in0:%s in1:%s in2:%s out:%s comp:%s) %s - M=%i, N=%i, LDI=%i, LDO=%i\n",
        libxsmm_get_typename(dtype_in0), libxsmm_get_typename(dtype_in1), libxsmm_get_typename(dtype_in2), libxsmm_get_typename(dtype_out), libxsmm_get_typename(dtype_comp), opname, M, N, ldi, ldo);
      res = test_ternary_op( M, N, ldi, ldo, op, use_bcast, dtype_in0, dtype_in1, dtype_in2, dtype_out, dtype_comp, rnd_mode);
    } else {
      printf(" Error! Usage: %s [type] [use_bcast: 0/1/2/3/4/5/6/7/8/9] [prec_in0: F32/BF16/F16/BF8/HF8] [prec_in1: F32/BF16/F16/BF8/HF8] [prec_in2: F32/BF16/F16/BF8/HF8] [compute_prec: F32] [prec_out: F32/BF16/F16/BF8/HF8] [M] [N] [ldi] [ldo]\n", argv[0] );
      exit(-1);
    }
  } else {
    printf(" Error! Usage: %s [type] [use_bcast: 0/1/2/3/4/5/6/7/8/9] [prec_in0: F32/BF16/F16/BF8/HF8] [prec_in1: F32/BF16/F16/BF8/HF8] [prec_in2: F32/BF16/F16/BF8/HF8]  [compute_prec: F32] [prec_out: F32/BF16/F16/BF8/HF8] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  return res;
}
