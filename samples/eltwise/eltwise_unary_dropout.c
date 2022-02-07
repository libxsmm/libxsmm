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

float upconvert_bf16(libxsmm_bfloat16 x) {
  union libxsmm_bfloat16_hp bf16_hp;
  bf16_hp.i[1] = x;
  bf16_hp.i[0] = 0;
  return bf16_hp.f;
}

void lsfr_Xwide( unsigned int* rng_state, float* prng_out, const unsigned int width ) {
  const unsigned int state_ld = 16;
  const float one = 1.0f;
  unsigned int w;
  union { unsigned int i; float f; } rng_num;

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

void dropout_fwd_f32_f32_gold(const unsigned int M, const float *in, float *out, unsigned char *dropout_mask, void* rng_state, const float p) {
  float vrng[16];
  unsigned int w;
  unsigned int i;
  unsigned int j;
  float pn = 1 - p;
  float pi = 1/pn;
  unsigned int cpuid = libxsmm_cpuid();
  const char *env_cpuid = getenv("LIBXSMM_TARGET");
  int is_env_cpuid_avx512 = 0;
  int is_env_cpuid_avx2 = 0;

  if ( env_cpuid != NULL ) {
    is_env_cpuid_avx512 = ( env_cpuid == libxsmm_stristr(env_cpuid, "cpx") ||
                            env_cpuid == libxsmm_stristr(env_cpuid, "clx") ||
                            env_cpuid == libxsmm_stristr(env_cpuid, "skx") ||
                            env_cpuid == libxsmm_stristr(env_cpuid, "skl") ||
                            env_cpuid == libxsmm_stristr(env_cpuid, "avx3") ||
                            env_cpuid == libxsmm_stristr(env_cpuid, "avx512") ||
                            env_cpuid == libxsmm_stristr(env_cpuid, "knm") ||
                            env_cpuid == libxsmm_stristr(env_cpuid, "knl") ||
                            env_cpuid == libxsmm_stristr(env_cpuid, "mic") ||
                            env_cpuid == libxsmm_stristr(env_cpuid, "spr") ||
                            env_cpuid == libxsmm_stristr(env_cpuid, "amx") );
    is_env_cpuid_avx2 = ( env_cpuid == libxsmm_stristr(env_cpuid, "hsw") ||
                          env_cpuid == libxsmm_stristr(env_cpuid, "avx2") );
  }

  if ( ((cpuid >= LIBXSMM_X86_AVX512_MIC) && (cpuid <= LIBXSMM_X86_ALLFEAT)) || ( is_env_cpuid_avx512 != 0 ) ) {
    w = 16;
  } else if ( (cpuid == LIBXSMM_X86_AVX2) || ( is_env_cpuid_avx2 != 0  ) ) {
    w = 8;
  } else {
    w = 4;
  }

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

void dropout_fwd_gold(const libxsmm_blasint M, const libxsmm_blasint N, const libxsmm_blasint ldi, const libxsmm_blasint ldo, const libxsmm_blasint mask_ld,
                      const void *in, void *out, unsigned char *mask, void* rng_state, float p, const libxsmm_datatype dtype_in, const libxsmm_datatype dtype_out, const libxsmm_datatype dtype_comp) {
  size_t j;

  if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_F32) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    const float* f_in = (const float*)in;
    float* f_out = (float*)out;
    for ( j = 0; j < N; ++j ) {
      dropout_fwd_f32_f32_gold( M, &(f_in[(j*ldi)]), &(f_out[(j*ldo)]), &(mask[(j*mask_ld)]), rng_state, p );
    }
  } else if ( (dtype_in == LIBXSMM_DATATYPE_BF16) && (dtype_out == LIBXSMM_DATATYPE_BF16) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    float* flt_in  = (float*)libxsmm_aligned_malloc( M*sizeof(float), 4096 );
    float* flt_out = (float*)libxsmm_aligned_malloc( M*sizeof(float), 4096 );
    const libxsmm_bfloat16* bf_in = (const libxsmm_bfloat16*)in;
    libxsmm_bfloat16* bf_out = (libxsmm_bfloat16*)out;
    for ( j = 0; j < N; ++j ) {
      libxsmm_convert_bf16_f32( &(bf_in[(j*ldi)]), flt_in, M );
      dropout_fwd_f32_f32_gold( M, flt_in, flt_out, &(mask[(j*mask_ld)]), rng_state, p );
      libxsmm_rne_convert_fp32_bf16( flt_out, &(bf_out[(j*ldo)]), M );
    }
    libxsmm_free( flt_in );
    libxsmm_free( flt_out );
  } else if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_BF16) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    float* flt_out = (float*)libxsmm_aligned_malloc( M*sizeof(float), 4096 );
    libxsmm_bfloat16* bf_out = (libxsmm_bfloat16*)out;
    const float* f_in = (const float*)in;
    for ( j = 0; j < N; ++j ) {
      dropout_fwd_f32_f32_gold( M, &(f_in[(j*ldi)]), flt_out, &(mask[(j*mask_ld)]), rng_state, p );
      libxsmm_rne_convert_fp32_bf16( flt_out, &(bf_out[(j*ldo)]), M );
    }
    libxsmm_free( flt_out );
  } else if ( (dtype_in == LIBXSMM_DATATYPE_BF16) && (dtype_out == LIBXSMM_DATATYPE_F32) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    float* flt_in  = (float*)libxsmm_aligned_malloc( M*sizeof(float), 4096 );
    const libxsmm_bfloat16* bf_in = (const libxsmm_bfloat16*)in;
    float* f_out = (float*)out;
    for ( j = 0; j < N; ++j ) {
      libxsmm_convert_bf16_f32( &(bf_in[(j*ldi)]), flt_in, M );
      dropout_fwd_f32_f32_gold( M, flt_in, &(f_out[(j*ldo)]), &(mask[(j*mask_ld)]), rng_state, p );
    }
    libxsmm_free( flt_in );
  } else {
    /* shouldn't happen */
  }
}

void dropout_bwd_gold(const libxsmm_blasint M, const libxsmm_blasint N, const libxsmm_blasint ldi, const libxsmm_blasint ldo, const libxsmm_blasint mask_ld,
                      const void *in, void *out, unsigned char *mask, float p, const libxsmm_datatype dtype_in, const libxsmm_datatype dtype_out, const libxsmm_datatype dtype_comp) {
  size_t i, j;
  float pn = 1.0f - p;
  float pi = 1.0f/pn;

  if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_F32) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    const float* f_in = (const float*)in;
    float* f_out = (float*)out;
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        f_out[(j*ldo) + i] = ( ( mask[(j*mask_ld) + (i/8)] & (1 << (i%8)) ) != 0 ) ? f_in[(j*ldi) + i] * pi : 0.0f;
      }
    }
  } else if ( (dtype_in == LIBXSMM_DATATYPE_BF16) && (dtype_out == LIBXSMM_DATATYPE_BF16) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    const libxsmm_bfloat16* bf_in = (const libxsmm_bfloat16*)in;
    libxsmm_bfloat16* bf_out = (libxsmm_bfloat16*)out;
    float in_value, out_value;
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        libxsmm_convert_bf16_f32( &(bf_in[(j*ldi) + i]), &in_value, 1 );
        out_value = ( ( mask[(j*mask_ld) + (i/8)] & (1 << (i%8)) ) != 0 ) ? in_value * pi : 0.0f;
        libxsmm_rne_convert_fp32_bf16(&out_value, &(bf_out[(j*ldo) + i]), 1);
      }
    }
  } else if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_BF16) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    const float* f_in = (const float*)in;
    libxsmm_bfloat16* bf_out = (libxsmm_bfloat16*)out;
    float out_value;
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        out_value = ( ( mask[(j*mask_ld) + (i/8)] & (1 << (i%8)) ) != 0 ) ? f_in[(j*ldi) + i] * pi : 0.0f;
        libxsmm_rne_convert_fp32_bf16(&out_value, &(bf_out[(j*ldo) + i]), 1);
      }
    }
  } else if ( (dtype_in == LIBXSMM_DATATYPE_BF16) && (dtype_out == LIBXSMM_DATATYPE_F32) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    const libxsmm_bfloat16* bf_in = (const libxsmm_bfloat16*)in;
    float* f_out = (float*)out;
    float in_value;
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        libxsmm_convert_bf16_f32( &(bf_in[(j*ldi) + i]), &in_value, 1 );
        f_out[(j*ldo) + i] = ( ( mask[(j*mask_ld) + (i/8)] & (1 << (i%8)) ) != 0 ) ? in_value * pi : 0.0f;
      }
    }
  } else {
    /* shouldn't happen */
  }
}

int test_dropout_fwd( const libxsmm_blasint bitm, const libxsmm_blasint M, const libxsmm_blasint N, const libxsmm_blasint ldi, const libxsmm_blasint ldo,
                      const libxsmm_datatype dtype_in, const libxsmm_datatype dtype_out, const libxsmm_datatype dtype_comp ) {
  char *in;
  char *out, *out_gold;
  unsigned char *mask, *mask_gold;
  unsigned int *rng_state, *rng_state_gold;
  unsigned int i, j;
  unsigned int s;
  float p = 0.3f;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape( M, N, &ldi, &ldo, dtype_in, dtype_out, dtype_comp );
  libxsmm_matdiff_info norms_out;
  libxsmm_blasint mask_ld = (bitm == 0) ? ldo : ((ldo+15)-((ldo+15)%16))/8;

  if ( M > ldi ) {
    fprintf( stderr, "test_dropout_fwd %i %i %i: ldi needs to be equal to or bigger than M\n", dtype_in, dtype_out, dtype_comp);
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_dropout_fwd %i %i %i: ldo needs to be equal to or bigger than N\n", dtype_in, dtype_out, dtype_comp);
    exit(-1);
  }

  in        = (char*) libxsmm_aligned_malloc( LIBXSMM_TYPESIZE(dtype_in) *N*ldi,   64);
  out       = (char*) libxsmm_aligned_malloc( LIBXSMM_TYPESIZE(dtype_out)*N*ldo,   64);
  out_gold  = (char*) libxsmm_aligned_malloc( LIBXSMM_TYPESIZE(dtype_out)*N*ldo,   64);
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

  /* use jited tranpose */
  unary_param.op.primary = (void*)&p;
  unary_param.op.secondary = (void*)rng_state;
  unary_param.in.primary  = (void*)in;
  unary_param.out.primary = (void*)out;
  unary_param.out.secondary = (bitm == 0) ? NULL : (void*)mask;
  unary_flags = (bitm == 0) ? LIBXSMM_MELTW_FLAG_UNARY_NONE : LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  libxsmm_meltwfunction_unary unary_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_DROPOUT, unary_shape, unary_flags );
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

  if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_F32) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    if ( norms_out.normf_rel > 0.00001 ) {
      ret = EXIT_FAILURE;
    }
  } else {
    if ( norms_out.normf_rel > 0.00001 ) {
      ret = EXIT_FAILURE;
    }
  }

  if ( bitm != 0 ) {
    s = 0;
    for ( i = 0; i < N; ++i ) {
      for ( j = 0; j < M/8; ++j ) {
        if ( mask_gold[(i*mask_ld)+j] != mask[(i*mask_ld)+j] ) {
          printf("error at possition i=%i, j=%i, %u, %u\n", i, j, mask[(i*mask_ld)+j], mask_gold[(i*mask_ld)+j]);
          s = 1;
        }
#if 0
        else {
          printf("correct at possition i=%i, j=%i, %u, %u\n", i, j, mask[(i*mask_ld)+j], mask_gold[(i*mask_ld)+j]);
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
    printf("SUCCESS unary dropout fwd %i %i %i\n", dtype_in, dtype_out, dtype_comp);
  } else {
    printf("FAILURE unary dropout fwd %i %i %i\n", dtype_in, dtype_out, dtype_comp);
  }

  return ret;
}

int test_dropout_bwd( const libxsmm_blasint M, const libxsmm_blasint N, const libxsmm_blasint ldi, const libxsmm_blasint ldo,
                      const libxsmm_datatype dtype_in, const libxsmm_datatype dtype_out, const libxsmm_datatype dtype_comp ) {
  char *in;
  char *out, *out_gold;
  unsigned char *mask;
  unsigned char *mask_gold;
  size_t i;
  float p = 0.3f;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape( M, N, &ldi, &ldo, dtype_in, dtype_out, dtype_comp );
  libxsmm_matdiff_info norms_out;
  libxsmm_blasint mask_ld = ((ldi+15)-((ldi+15)%16))/8;

  if ( M > ldi ) {
    fprintf( stderr, "test_dropout_fwd %i %i %i: ldi needs to be equal to or bigger than M\n", dtype_in, dtype_out, dtype_comp);
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_dropout_fwd %i %i %i: ldo needs to be equal to or bigger than N\n", dtype_in, dtype_out, dtype_comp);
    exit(-1);
  }

  in        = (char*) libxsmm_aligned_malloc( LIBXSMM_TYPESIZE(dtype_in)  *N*ldi,   64);
  out       = (char*) libxsmm_aligned_malloc( LIBXSMM_TYPESIZE(dtype_out)*N*ldo,   64);
  out_gold  = (char*) libxsmm_aligned_malloc( LIBXSMM_TYPESIZE(dtype_out)*N*ldo,   64);
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

  /* use jited tranpose */
  unary_param.op.primary = (void*)&p;
  unary_param.in.primary  = (void*)in;
  unary_param.in.secondary = (void*)mask;
  unary_param.out.primary = (void*)out;
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  libxsmm_meltwfunction_unary unary_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV, unary_shape, unary_flags );
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

  if ( (dtype_in == LIBXSMM_DATATYPE_F32) && (dtype_out == LIBXSMM_DATATYPE_F32) && (dtype_comp == LIBXSMM_DATATYPE_F32) ) {
    if ( norms_out.normf_rel > 0.00001 ) {
      ret = EXIT_FAILURE;
    }
  } else {
    if ( norms_out.normf_rel > 0.007 ) {
      ret = EXIT_FAILURE;
    }
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( mask );
  libxsmm_free( mask_gold );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary dropout bwd fp32 fp32\n");
  } else {
    printf("FAILURE unary dropout bwd fp32 fp32\n");
  }

  return ret;
}

int main( int argc, char* argv[] ) {
  libxsmm_blasint dtype_in;
  libxsmm_blasint dtype_out;
  char op;
  libxsmm_blasint bitm;
  libxsmm_blasint M;
  libxsmm_blasint N;
  libxsmm_blasint ldi;
  libxsmm_blasint ldo;
  int ret = EXIT_FAILURE;

  if ( argc != 9 ) {
    printf(" Error! Usage: %s [F/B] [bitmask: 0/1] [prec_in: 4/2] [prec_out: 4/2] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  op        = *(argv[1]);
  bitm      = atoi(argv[2]);
  dtype_in  = atoi(argv[3]);
  dtype_out = atoi(argv[4]);
  M         = atoi(argv[5]);
  N         = atoi(argv[6]);
  ldi       = atoi(argv[7]);
  ldo       = atoi(argv[8]);

  if (  op == 'B' && bitm == 0 ) {
    printf("Backward needs masks!\n");
    return ret;
  }

  if ( op == 'F' && dtype_in == 4 && dtype_out == 4  ) {
    printf("Testing F32 F32 forward dropout - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_dropout_fwd( bitm, M, N, ldi, ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
  } else if ( op == 'F' && dtype_in == 2  && dtype_out == 2 ) {
    printf("Testing BF16 BF16 forward dropout - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_dropout_fwd( bitm, M, N, ldi, ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32 );
  } else if ( op == 'F' && dtype_in == 4  && dtype_out == 2 ) {
    printf("Testing F32 BF16 forward dropout - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_dropout_fwd( bitm, M, N, ldi, ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32 );
  } else if ( op == 'F' && dtype_in == 2  && dtype_out == 4 ) {
    printf("Testing BF16 F32 forward dropout - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_dropout_fwd( bitm, M, N, ldi, ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
  } else if ( op == 'B' && dtype_in == 4 && dtype_out == 4 ) {
    printf("Testing F32 F32 backward dropout - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_dropout_bwd( M, N, ldi, ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
  } else if ( op == 'B' && dtype_in == 2 && dtype_out == 2 ) {
    printf("Testing BF16 BF16 backward dropout - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_dropout_bwd( M, N, ldi, ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32 );
  } else if ( op == 'B' && dtype_in == 4 && dtype_out == 2 ) {
    printf("Testing F32 BF16 backward dropout - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_dropout_bwd( M, N, ldi, ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32 );
  } else if ( op == 'B' && dtype_in == 2 && dtype_out == 4 ) {
    printf("Testing BF16 F32 backward dropout - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_dropout_bwd( M, N, ldi, ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
  } else {
    printf(" Not implemented case! Usage: %s [F/B] [bitmask: 0/1] [prec_in: 4/2] [prec_out: 4/2] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  return ret;
}
