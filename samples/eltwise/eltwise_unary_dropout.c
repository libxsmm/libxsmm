/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "eltwise_common.h"

#if 0
#define USE_ZERO_RNG_STATE_UNITTEST
#endif

#define LIBXSMM_ALIGNDOWN(N, A) ((N) & ~((A)-1))

LIBXSMM_INLINE
float upconvert_bf16(libxsmm_bfloat16 x) {
  libxsmm_bfloat16_f32 bf16_hp /* = { 0 }*/;
  bf16_hp.i[1] = x;
  bf16_hp.i[0] = 0;
  return bf16_hp.f;
}

LIBXSMM_INLINE
void lsfr_Xwide( unsigned int* rng_state, float* prng_out, const unsigned int width ) {
  union { unsigned int i; float f; } rng_num = { 0 };
  const unsigned int state_ld = 16;
  const float one = 1.0f;
  unsigned int w;

  for ( w = 0 ; w < width; ++w ) {
    unsigned int state_0 = rng_state[w + (0 * state_ld)];
    unsigned int state_1 = rng_state[w + (1 * state_ld)];
    unsigned int state_2 = rng_state[w + (2 * state_ld)];
    unsigned int state_3 = rng_state[w + (3 * state_ld)];
    unsigned int tmp_0, tmp_1;
    rng_num.i = state_3 + state_0;
    rng_num.i = rng_num.i >> 9;
    rng_num.i = 0x3f800000 | rng_num.i;
    prng_out[w] = rng_num.f - one;
    tmp_0 = state_1 << 9;
    state_2 = state_2 ^ state_0;
    state_3 = state_3 ^ state_1;
    state_1 = state_1 ^ state_2;
    state_0 = state_0 ^ state_3;
    state_2 = state_2 ^ tmp_0;
    tmp_0 = state_3 << 11;
    tmp_1 = state_3 >> 21;
    state_3 = tmp_0 | tmp_1;
    rng_state[w + (0 * state_ld)] = state_0;
    rng_state[w + (1 * state_ld)] = state_1;
    rng_state[w + (2 * state_ld)] = state_2;
    rng_state[w + (3 * state_ld)] = state_3;
  }
}

LIBXSMM_INLINE
void dropout_fwd_f32_f32_gold(const unsigned int M, const float *in, float *out, unsigned char *dropout_mask, void* rng_state, const float p) {
  float vrng[16];
  unsigned int i;
  unsigned int j;
  float pn = 1 - p;
  float pi = 1/pn;
  unsigned int w = libxsmm_cpuid_vlen32(libxsmm_get_target_archid());

  for (i = 0; i < LIBXSMM_ALIGNDOWN(M, w); i+=w) {
    lsfr_Xwide( (unsigned int*)rng_state, vrng, w );
    for ( j = 0; j < w; ++j ) {
      out[i+j] = ( vrng[j] < pn ) ? pi * in[i+j] : 0.0f;
      dropout_mask[(i+j)/8] |= (unsigned char)(( vrng[j] < pn ) ? (1 << ((i+j)%8)) : 0x0 );
    }
  }
  if (i < M) {
    lsfr_Xwide( (unsigned int*)rng_state, vrng, w );
    j = 0;
    for ( ; i < M; ++i ) {
      out[i] = ( vrng[j] < pn ) ? pi * in[i] : 0.0f;
      dropout_mask[i/8] |= (unsigned char)(( vrng[j] < pn ) ? (1 << (i%8)) : 0x0 );
      j++;
    }
  }
}

LIBXSMM_INLINE
void dropout_fwd_gold(const libxsmm_blasint M, const libxsmm_blasint N, const libxsmm_blasint ldi, const libxsmm_blasint ldo, const libxsmm_blasint mask_ld,
                      const void *in, void *out, unsigned char *mask, void* rng_state, float p, const libxsmm_datatype dtype_in, const libxsmm_datatype dtype_out, const libxsmm_datatype dtype_comp) {
  libxsmm_blasint j;

  if ( dtype_comp == LIBXSMM_DATATYPE_F32 ) {
    float *in_values  = (float*)libxsmm_aligned_malloc( M*sizeof(float), 4096 );
    float *out_values = (float*)libxsmm_aligned_malloc( M*sizeof(float), 4096 );
    for ( j = 0; j < N; ++j ) {
      if ( dtype_in == LIBXSMM_DATATYPE_F32 ) {
        const float* f_in = (const float*)in;
        memcpy( in_values, (void*)&(f_in[(j*ldi)]), M*sizeof(float) );
      } else if ( dtype_in == LIBXSMM_DATATYPE_BF16 ) {
        const libxsmm_bfloat16* bf16_in = (const libxsmm_bfloat16*)in;
        libxsmm_convert_bf16_f32( &(bf16_in[(j*ldi)]), in_values, M );
      } else if ( dtype_in == LIBXSMM_DATATYPE_F16 ) {
        const libxsmm_float16* f16_in = (const libxsmm_float16*)in;
        libxsmm_convert_f16_f32( &(f16_in[(j*ldi)]), in_values, M );
      } else if ( dtype_in == LIBXSMM_DATATYPE_BF8 ) {
        const libxsmm_bfloat8* bf8_in = (const libxsmm_bfloat8*)in;
        libxsmm_convert_bf8_f32( &(bf8_in[(j*ldi)]), in_values, M );
      } else if ( dtype_in == LIBXSMM_DATATYPE_HF8 ) {
        const libxsmm_hfloat8* hf8_in = (const libxsmm_hfloat8*)in;
        libxsmm_convert_hf8_f32( &(hf8_in[(j*ldi)]), in_values, M );
      } else {
        /* shouldn't happen */
      }

      dropout_fwd_f32_f32_gold( M, in_values, out_values, &(mask[(j*mask_ld)]), rng_state, p );

      if ( dtype_out == LIBXSMM_DATATYPE_F32 ) {
        float* f_out = (float*)out;
        memcpy( (void*)&(f_out[(j*ldo)]), out_values, M*sizeof(float) );
      } else if ( dtype_out == LIBXSMM_DATATYPE_BF16 ) {
        libxsmm_bfloat16* bf16_out = (libxsmm_bfloat16*)out;
        libxsmm_rne_convert_fp32_bf16( out_values, &(bf16_out[(j*ldo)]), M );
      } else if ( dtype_out == LIBXSMM_DATATYPE_F16 ) {
        libxsmm_float16* f16_out = (libxsmm_float16*)out;
        libxsmm_rne_convert_fp32_f16( out_values, &(f16_out[(j*ldo)]),  M );
      } else if ( dtype_out == LIBXSMM_DATATYPE_BF8 ) {
        libxsmm_bfloat8* bf8_out = (libxsmm_bfloat8*)out;
        libxsmm_rne_convert_fp32_bf8( out_values, &(bf8_out[(j*ldo)]), M );
      } else if ( dtype_out == LIBXSMM_DATATYPE_HF8 ) {
        libxsmm_hfloat8* hf8_out = (libxsmm_hfloat8*)out;
        libxsmm_rne_convert_fp32_hf8( out_values, &(hf8_out[(j*ldo)]), M );
      } else {
        /* shouldn't happen */
      }
    }
    libxsmm_free( in_values );
    libxsmm_free( out_values );
  } else {
    /* shouldn't happen */
  }
}

LIBXSMM_INLINE
void dropout_bwd_gold(const libxsmm_blasint M, const libxsmm_blasint N, const libxsmm_blasint ldi, const libxsmm_blasint ldo, const libxsmm_blasint mask_ld,
                      const void *in, void *out, unsigned char *mask, float p, const libxsmm_datatype dtype_in, const libxsmm_datatype dtype_out, const libxsmm_datatype dtype_comp) {
  libxsmm_blasint i, j;
  float pn = 1.0f - p;
  float pi = 1.0f/pn;

  if ( dtype_comp == LIBXSMM_DATATYPE_F32 ) {
    float in_value = 0, out_value;
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        if ( dtype_in == LIBXSMM_DATATYPE_F32 ) {
          const float* f_in = (const float*)in;
          in_value = f_in[(j*ldi) + i];
        } else if ( dtype_in == LIBXSMM_DATATYPE_BF16 ) {
          const libxsmm_bfloat16* bf16_in = (const libxsmm_bfloat16*)in;
          libxsmm_convert_bf16_f32( &(bf16_in[(j*ldi) + i]), &in_value, 1 );
        } else if ( dtype_in == LIBXSMM_DATATYPE_F16 ) {
          const libxsmm_float16* f16_in = (const libxsmm_float16*)in;
          libxsmm_convert_f16_f32( &(f16_in[(j*ldi) + i]), &in_value, 1 );
        } else if ( dtype_in == LIBXSMM_DATATYPE_BF8 ) {
          const libxsmm_bfloat8* bf8_in = (const libxsmm_bfloat8*)in;
          libxsmm_convert_bf8_f32( &(bf8_in[(j*ldi) + i]), &in_value, 1 );
        } else if ( dtype_in == LIBXSMM_DATATYPE_HF8 ) {
          const libxsmm_hfloat8* hf8_in = (const libxsmm_hfloat8*)in;
          libxsmm_convert_hf8_f32( &(hf8_in[(j*ldi) + i]), &in_value, 1 );
        } else {
          /* shouldn't happen */
        }

        out_value = ( ( mask[(j*mask_ld) + (i/8)] & (1 << (i%8)) ) != 0 ) ? in_value * pi : 0.0f;

        if ( dtype_out == LIBXSMM_DATATYPE_F32 ) {
          float* f_out = (float*)out;
          f_out[(j*ldo) + i] = out_value;
        } else if ( dtype_out == LIBXSMM_DATATYPE_BF16 ) {
          libxsmm_bfloat16* bf16_out = (libxsmm_bfloat16*)out;
          libxsmm_rne_convert_fp32_bf16(&out_value, &(bf16_out[(j*ldo) + i]), 1 );
        } else if ( dtype_out == LIBXSMM_DATATYPE_F16 ) {
          libxsmm_float16* f16_out = (libxsmm_float16*)out;
          libxsmm_rne_convert_fp32_f16(&out_value, &(f16_out[(j*ldo) + i]), 1 );
        } else if ( dtype_out == LIBXSMM_DATATYPE_BF8 ) {
          libxsmm_bfloat8* bf8_out = (libxsmm_bfloat8*)out;
          libxsmm_rne_convert_fp32_bf8(&out_value, &(bf8_out[(j*ldo) + i]), 1 );
        } else if ( dtype_out == LIBXSMM_DATATYPE_HF8 ) {
          libxsmm_hfloat8* hf8_out = (libxsmm_hfloat8*)out;
          libxsmm_rne_convert_fp32_hf8(&out_value, &(hf8_out[(j*ldo) + i]), 1 );
        } else {
          /* shouldn't happen */
        }
      }
    }
  } else {
    /* shouldn't happen */
  }
}

LIBXSMM_INLINE
int test_dropout_fwd( const libxsmm_blasint bitm, const libxsmm_blasint M, const libxsmm_blasint N, const libxsmm_blasint ldi, const libxsmm_blasint ldo,
                      const libxsmm_datatype dtype_in, const libxsmm_datatype dtype_out, const libxsmm_datatype dtype_comp ) {
  char *in;
  char *out, *out_gold;
  unsigned char *mask, *mask_gold;
  unsigned int *rng_state, *rng_state_gold;
  libxsmm_blasint i, j;
  unsigned int s;
  float p = 0.3f;
  int ret = EXIT_SUCCESS;
  libxsmm_meltwfunction_unary unary_kernel;
  libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape( M, N, ldi, ldo, dtype_in, dtype_out, dtype_comp );
  libxsmm_matdiff_info norms_out;
  libxsmm_blasint mask_ld = (bitm == 0) ? ldo : ((ldo+15)-((ldo+15)%16))/8;

  if ( M > ldi ) {
    fprintf( stderr, "test_dropout_fwd %i %i %i: ldi needs to be equal to or bigger than M\n", (int)dtype_in, (int)dtype_out, (int)dtype_comp);
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_dropout_fwd %i %i %i: ldo needs to be equal to or bigger than N\n", (int)dtype_in, (int)dtype_out, (int)dtype_comp);
    exit(-1);
  }

  in        = (char*) libxsmm_aligned_malloc((size_t)LIBXSMM_TYPESIZE(dtype_in) *N*ldi, 64);
  out       = (char*) libxsmm_aligned_malloc((size_t)LIBXSMM_TYPESIZE(dtype_out)*N*ldo, 64);
  out_gold  = (char*) libxsmm_aligned_malloc((size_t)LIBXSMM_TYPESIZE(dtype_out)*N*ldo, 64);
  mask      = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);

  /* init in/out */
  init_random_matrix( dtype_in,  in,       1, ldi, N, 0 );
  init_zero_matrix(   dtype_out, out,      1, ldo, N );
  init_zero_matrix(   dtype_out, out_gold, 1, ldo, N );
  init_zero_matrix(   LIBXSMM_DATATYPE_I8, mask,      1, mask_ld, N );
  init_zero_matrix(   LIBXSMM_DATATYPE_I8, mask_gold, 1, mask_ld, N );

  rng_state = libxsmm_rng_create_extstate( 555 );
  rng_state_gold = libxsmm_rng_create_extstate( 555 );

#ifdef USE_ZERO_RNG_STATE_UNITTEST
  memset( (void*)rng_state, 0, libxsmm_rng_get_extstate_size() );
  memset( (void*)rng_state_gold, 0, libxsmm_rng_get_extstate_size() );
#endif

  /* compute out_gold */
  dropout_fwd_gold( M, N, ldi, ldo, mask_ld, in, out_gold, mask_gold, rng_state_gold, p, dtype_in, dtype_out, dtype_comp );

  /* use jited transpose */
  unary_param.op.primary = (void*)&p;
  unary_param.op.secondary = (void*)rng_state;
  unary_param.in.primary  = (void*)in;
  unary_param.out.primary = (void*)out;
  unary_param.out.secondary = (bitm == 0) ? NULL : (void*)mask;
  unary_flags = (bitm == 0) ? LIBXSMM_MELTW_FLAG_UNARY_NONE : LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  unary_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_DROPOUT, unary_shape, unary_flags );
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
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

  if ( norms_out.normf_rel > 0.00001 ) {
    ret = EXIT_FAILURE;
  }

  if ( bitm != 0 ) {
    s = 0;
    for ( i = 0; i < N; ++i ) {
      for ( j = 0; j < M/8; ++j ) {
        if ( mask_gold[(i*mask_ld)+j] != mask[(i*mask_ld)+j] ) {
          printf("error at possition i=%i, j=%i, %i, %i\n", i, j, mask[(i*mask_ld)+j], mask_gold[(i*mask_ld)+j]);
          s = 1;
        }
#if 0
        else {
          printf("correct at possition i=%i, j=%i, %i, %i\n", i, j, mask[(i*mask_ld)+j], mask_gold[(i*mask_ld)+j]);
        }
#endif
      }
    }
    if ( s == 0 ) {
      printf("SUCCESS mask\n");
    } else {
      printf("FAILURE mask\n");
      ret = EXIT_FAILURE;
    }
  }

  libxsmm_rng_destroy_extstate( rng_state );
  libxsmm_rng_destroy_extstate( rng_state_gold );

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( mask );
  libxsmm_free( mask_gold );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary dropout fwd %i %i %i\n", (int)dtype_in, (int)dtype_out, (int)dtype_comp);
  } else {
    printf("FAILURE unary dropout fwd %i %i %i\n", (int)dtype_in, (int)dtype_out, (int)dtype_comp);
  }

  return ret;
}

LIBXSMM_INLINE
int test_dropout_bwd( const libxsmm_blasint M, const libxsmm_blasint N, const libxsmm_blasint ldi, const libxsmm_blasint ldo,
                      const libxsmm_datatype dtype_in, const libxsmm_datatype dtype_out, const libxsmm_datatype dtype_comp ) {
  char *in;
  char *out, *out_gold;
  unsigned char *mask;
  unsigned char *mask_gold;
  libxsmm_blasint i;
  float p = 0.3f;
  int ret = EXIT_SUCCESS;
  libxsmm_meltwfunction_unary unary_kernel;
  libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape( M, N, ldi, ldo, dtype_in, dtype_out, dtype_comp );
  libxsmm_matdiff_info norms_out;
  libxsmm_blasint mask_ld = ((ldi+15)-((ldi+15)%16))/8;

  if ( M > ldi ) {
    fprintf( stderr, "test_dropout_fwd %i %i %i: ldi needs to be equal to or bigger than M\n", (int)dtype_in, (int)dtype_out, (int)dtype_comp);
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_dropout_fwd %i %i %i: ldo needs to be equal to or bigger than N\n", (int)dtype_in, (int)dtype_out, (int)dtype_comp);
    exit(-1);
  }

  in        = (char*) libxsmm_aligned_malloc((size_t)LIBXSMM_TYPESIZE(dtype_in) *N*ldi, 64);
  out       = (char*) libxsmm_aligned_malloc((size_t)LIBXSMM_TYPESIZE(dtype_out)*N*ldo, 64);
  out_gold  = (char*) libxsmm_aligned_malloc((size_t)LIBXSMM_TYPESIZE(dtype_out)*N*ldo, 64);
  mask      = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);

  /* init in,out */
  init_random_matrix( dtype_in,  in,       1, ldi, N, 0 );
  init_zero_matrix(   dtype_out, out,      1, ldo, N );
  init_zero_matrix(   dtype_out, out_gold, 1, ldo, N );
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask[i] = (unsigned char)(0xaa ^ (i%256));
  }
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask_gold[i] = (unsigned char)(0xaa ^ (i%256));
  }

  /* compute out_gold */
  dropout_bwd_gold( M, N, ldi, ldo, mask_ld, in, out_gold, mask_gold, p, dtype_in, dtype_out, dtype_comp );

  /* use jited transpose */
  unary_param.op.primary = (void*)&p;
  unary_param.in.primary  = (void*)in;
  unary_param.in.secondary = (void*)mask;
  unary_param.out.primary = (void*)out;

  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  unary_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV, unary_shape, unary_flags );
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
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

  if ( norms_out.normf_rel > 0.00001 ) {
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( mask );
  libxsmm_free( mask_gold );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary dropout bwd %i %i %i\n", (int)dtype_in, (int)dtype_out, (int)dtype_comp);
  } else {
    printf("FAILURE unary dropout bwd %i %i %i\n", (int)dtype_in, (int)dtype_out, (int)dtype_comp);
  }

  return ret;
}

int main( int argc, char* argv[] ) {
  char* dt_in = NULL;
  char* dt_out = NULL;
  libxsmm_datatype dtype_in;
  libxsmm_datatype dtype_out;
  libxsmm_datatype dtype_comp = LIBXSMM_DATATYPE_F32;
  char op;
  libxsmm_blasint bitm;
  libxsmm_blasint M;
  libxsmm_blasint N;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;
  int ret = EXIT_FAILURE;

  if ( argc != 9 ) {
    printf(" Error! Usage: %s [F/B] [bitmask: 0/1] [prec_in: F32/BF16/F16/BF8/HF8] [prec_out: F32/BF16/F16/BF8/HF8] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  op        = *(argv[1]);
  bitm      = atoi(argv[2]);
  dt_in     = argv[3];
  dt_out    = argv[4];
  M         = atoi(argv[5]);
  N         = atoi(argv[6]);
  ldi       = atoi(argv[7]);
  ldo       = atoi(argv[8]);

  if (  op == 'B' && bitm == 0 ) {
    printf("Backward needs masks!\n");
    return ret;
  }

  dtype_in  = char_to_libxsmm_datatype( dt_in );
  dtype_out = char_to_libxsmm_datatype( dt_out );

  if ( ( (dtype_in == LIBXSMM_DATATYPE_F32 ) && (dtype_out == LIBXSMM_DATATYPE_F32 ) ) ||
       ( (dtype_in == LIBXSMM_DATATYPE_BF16) && (dtype_out == LIBXSMM_DATATYPE_BF16) ) ||
       ( (dtype_in == LIBXSMM_DATATYPE_BF16) && (dtype_out == LIBXSMM_DATATYPE_F32 ) ) ||
       ( (dtype_in == LIBXSMM_DATATYPE_F32 ) && (dtype_out == LIBXSMM_DATATYPE_BF16) ) ||
       ( (dtype_in == LIBXSMM_DATATYPE_F16 ) && (dtype_out == LIBXSMM_DATATYPE_F16 ) ) ||
       ( (dtype_in == LIBXSMM_DATATYPE_F16 ) && (dtype_out == LIBXSMM_DATATYPE_F32 ) ) ||
       ( (dtype_in == LIBXSMM_DATATYPE_F32 ) && (dtype_out == LIBXSMM_DATATYPE_F16 ) ) ||
       ( (dtype_in == LIBXSMM_DATATYPE_BF8 ) && (dtype_out == LIBXSMM_DATATYPE_BF8 ) ) ||
       ( (dtype_in == LIBXSMM_DATATYPE_BF8 ) && (dtype_out == LIBXSMM_DATATYPE_F32 ) ) ||
       ( (dtype_in == LIBXSMM_DATATYPE_F32 ) && (dtype_out == LIBXSMM_DATATYPE_BF8 ) ) ||
       ( (dtype_in == LIBXSMM_DATATYPE_HF8 ) && (dtype_out == LIBXSMM_DATATYPE_HF8 ) ) ||
       ( (dtype_in == LIBXSMM_DATATYPE_HF8 ) && (dtype_out == LIBXSMM_DATATYPE_F32 ) ) ||
       ( (dtype_in == LIBXSMM_DATATYPE_F32 ) && (dtype_out == LIBXSMM_DATATYPE_HF8 ) ) ) {
    if (  op == 'F' ) {
      printf("in: %s out: %s comp: %s forward dropout - M=%i, N=%i, LDI=%i, LDO=%i\n", libxsmm_get_typename(dtype_in), libxsmm_get_typename(dtype_out), libxsmm_get_typename(dtype_comp), M, N, ldi, ldo );
      ret = test_dropout_fwd( bitm, M, N, ldi, ldo, dtype_in, dtype_out, dtype_comp );
    } else if (  op == 'B' ) {
      printf("in: %s out: %s comp: %s backward dropout - M=%i, N=%i, LDI=%i, LDO=%i\n", libxsmm_get_typename(dtype_in), libxsmm_get_typename(dtype_out), libxsmm_get_typename(dtype_comp), M, N, ldi, ldo );
      ret = test_dropout_bwd( M, N, ldi, ldo, dtype_in, dtype_out, dtype_comp );
    } else {
      printf(" Not implemented case! Usage: %s [F/B] [bitmask: 0/1] [prec_in: F32/BF16/F16/BF8/HF8] [prec_out: F32/BF16/F16/BF8/HF8] [M] [N] [ldi] [ldo]\n", argv[0] );
      exit(-1);
    }
  } else {
    printf(" Not implemented case! Usage: %s [F/B] [bitmask: 0/1] [prec_in: F32/BF16/F16/BF8/HF8] [prec_out: F32/BF16/F16/BF8/HF8] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  return ret;
}
