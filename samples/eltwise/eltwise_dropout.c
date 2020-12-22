/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef __AVX512BW__
#include <libxsmm_intrinsics_x86.h>
#endif

#ifdef __AVX512BW__
void dropout_fwd_f32_f32_gold(unsigned int M, float *in, float *out, unsigned short *dropout_mask, void* rng_state, float p) {
  unsigned int i;
  float pn = 1 - p;
  __m512 vp = _mm512_set1_ps(pn);
  __m512 vpi = _mm512_set1_ps(1.0/pn);
  for (i = 0; i < M - 15; i+=16) {
    __m512 rnd = LIBXSMM_INTRINSICS_MM512_RNG_EXTSTATE_PS(rng_state);
    __m512 vin = _mm512_loadu_ps(in+i);
    __mmask16 dmsk = _mm512_cmplt_ps_mask(rnd, vp);
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm512_storeu_ps(out+i, vout);
    dropout_mask[i/16] = dmsk;
  }
  if (i < M) {
    int rem = M - i;
    __mmask16 mask = (1 << rem) - 1;
    __m512 rnd = LIBXSMM_INTRINSICS_MM512_RNG_EXTSTATE_PS(rng_state);
    __m512 vin = _mm512_maskz_loadu_ps(mask, in+i);
    __mmask16 dmsk = _mm512_cmplt_ps_mask(rnd, vp);
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm512_mask_storeu_ps(out+i, mask, vout);
    dropout_mask[i/16] = dmsk & mask;
  }
}

void dropout_fwd_bf16_bf16_gold(unsigned int M, libxsmm_bfloat16 *in, libxsmm_bfloat16 *out, unsigned short *dropout_mask, void* rng_state, float p) {
  unsigned int i;
  float pn = 1 - p;
  __m512 vp = _mm512_set1_ps(pn);
  __m512 vpi = _mm512_set1_ps(1.0/pn);
  for (i = 0; i < M - 15; i+=16) {
    __m512 rnd = LIBXSMM_INTRINSICS_MM512_RNG_EXTSTATE_PS(rng_state);
    __m512 vin = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(in+i))),16));
    __mmask16 dmsk = _mm512_cmplt_ps_mask(rnd, vp);
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm256_storeu_si256((__m256i*)(out+i),_mm512_cvtepi32_epi16(_mm512_srai_epi32(LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(vout),16)));
    dropout_mask[i/16] = dmsk;
  }
  if (i < M) {
    int rem = M - i;
    __mmask16 mask = (1 << rem) - 1;
    __m512 rnd = LIBXSMM_INTRINSICS_MM512_RNG_EXTSTATE_PS(rng_state);
    __m512 vin = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(_mm256_maskz_loadu_epi16(mask, (__m256i*)(in+i))),16));
    __mmask16 dmsk = _mm512_cmplt_ps_mask(rnd, vp);
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm256_mask_storeu_epi16((__m256i*)(out+i), mask, _mm512_cvtepi32_epi16(_mm512_srai_epi32(LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(vout),16)));
    dropout_mask[i/16] = dmsk & mask;
  }
}

void dropout_fwd_f32_bf16_gold(unsigned int M, float *in, libxsmm_bfloat16 *out, unsigned short *dropout_mask, void* rng_state, float p) {
  unsigned int i;
  float pn = 1 - p;
  __m512 vp = _mm512_set1_ps(pn);
  __m512 vpi = _mm512_set1_ps(1.0/pn);
  for (i = 0; i < M - 15; i+=16) {
    __m512 rnd = LIBXSMM_INTRINSICS_MM512_RNG_EXTSTATE_PS(rng_state);
    __m512 vin = _mm512_loadu_ps(in+i);
    __mmask16 dmsk = _mm512_cmplt_ps_mask(rnd, vp);
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm256_storeu_si256((__m256i*)(out+i),_mm512_cvtepi32_epi16(_mm512_srai_epi32(LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(vout),16)));
    dropout_mask[i/16] = dmsk;
  }
  if (i < M) {
    int rem = M - i;
    __mmask16 mask = (1 << rem) - 1;
    __m512 rnd = LIBXSMM_INTRINSICS_MM512_RNG_EXTSTATE_PS(rng_state);
    __m512 vin = _mm512_maskz_loadu_ps(mask, in+i);
    __mmask16 dmsk = _mm512_cmplt_ps_mask(rnd, vp);
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm256_mask_storeu_epi16((__m256i*)(out+i), mask, _mm512_cvtepi32_epi16(_mm512_srai_epi32(LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(vout),16)));
    dropout_mask[i/16] = dmsk & mask;
  }
}

void dropout_fwd_bf16_f32_gold(unsigned int M, libxsmm_bfloat16 *in, float *out, unsigned short *dropout_mask, void* rng_state, float p) {
  unsigned int i;
  float pn = 1 - p;
  __m512 vp = _mm512_set1_ps(pn);
  __m512 vpi = _mm512_set1_ps(1.0/pn);
  for (i = 0; i < M - 15; i+=16) {
    __m512 rnd = LIBXSMM_INTRINSICS_MM512_RNG_EXTSTATE_PS(rng_state);
    __m512 vin = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(in+i))),16));
    __mmask16 dmsk = _mm512_cmplt_ps_mask(rnd, vp);
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm512_storeu_ps(out+i, vout);
    dropout_mask[i/16] = dmsk;
  }
  if (i < M) {
    int rem = M - i;
    __mmask16 mask = (1 << rem) - 1;
    __m512 rnd = LIBXSMM_INTRINSICS_MM512_RNG_EXTSTATE_PS(rng_state);
    __m512 vin = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(_mm256_maskz_loadu_epi16(mask, (__m256i*)(in+i))),16));
    __mmask16 dmsk = _mm512_cmplt_ps_mask(rnd, vp);
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm512_mask_storeu_ps(out+i, mask, vout);
    dropout_mask[i/16] = dmsk & mask;
  }
}

void dropout_bwd_f32_f32_gold(unsigned int M, float *in, float *out, unsigned short *dropout_mask, float p) {
  unsigned int i = 0;
  float pn = 1 - p;
  __m512 vpi = _mm512_set1_ps(1.0/pn);
  for (i = 0; i < M - 15; i+=16) {
    __m512 vin = _mm512_loadu_ps(in+i);
    __mmask16 dmsk = dropout_mask[i/16];
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm512_storeu_ps(out+i, vout);
  }
  if (i < M) {
    int rem = M - i;
    __mmask16 mask = (1 << rem) - 1;
    __m512 vin = _mm512_maskz_loadu_ps(mask, in+i);
    __mmask16 dmsk = dropout_mask[i/16];
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm512_mask_storeu_ps(out+i, mask, vout);
  }
}

void dropout_bwd_bf16_bf16_gold(unsigned int M, libxsmm_bfloat16 *in, libxsmm_bfloat16 *out, unsigned short *dropout_mask, float p) {
  unsigned int i = 0;
  float pn = 1 - p;
  __m512 vpi = _mm512_set1_ps(1.0/pn);
  for (i = 0; i < M - 15; i+=16) {
    __m512 vin = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(in+i))),16));
    __mmask16 dmsk = dropout_mask[i/16];
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm256_storeu_si256((__m256i*)(out+i),_mm512_cvtepi32_epi16(_mm512_srai_epi32(LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(vout),16)));
  }
  if (i < M) {
    int rem = M - i;
    __mmask16 mask = (1 << rem) - 1;
    __m512 vin = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(_mm256_maskz_loadu_epi16(mask, (__m256i*)(in+i))),16));
    __mmask16 dmsk = dropout_mask[i/16];
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm256_mask_storeu_epi16((__m256i*)(out+i), mask, _mm512_cvtepi32_epi16(_mm512_srai_epi32(LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(vout),16)));
  }
}

void dropout_bwd_f32_bf16_gold(unsigned int M, float *in, libxsmm_bfloat16 *out, unsigned short *dropout_mask, float p) {
  unsigned int i = 0;
  float pn = 1 - p;
  __m512 vpi = _mm512_set1_ps(1.0/pn);
  for (i = 0; i < M - 15; i+=16) {
    __m512 vin = _mm512_loadu_ps(in+i);
    __mmask16 dmsk = dropout_mask[i/16];
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm256_storeu_si256((__m256i*)(out+i),_mm512_cvtepi32_epi16(_mm512_srai_epi32(LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(vout),16)));
  }
  if (i < M) {
    int rem = M - i;
    __mmask16 mask = (1 << rem) - 1;
    __m512 vin = _mm512_maskz_loadu_ps(mask, in+i);
    __mmask16 dmsk = dropout_mask[i/16];
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm256_mask_storeu_epi16((__m256i*)(out+i), mask, _mm512_cvtepi32_epi16(_mm512_srai_epi32(LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(vout),16)));
  }
}

void dropout_bwd_bf16_f32_gold(unsigned int M, libxsmm_bfloat16 *in, float *out, unsigned short *dropout_mask, float p) {
  unsigned int i = 0;
  float pn = 1 - p;
  __m512 vpi = _mm512_set1_ps(1.0/pn);
  for (i = 0; i < M - 15; i+=16) {
    __m512 vin = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(in+i))),16));
    __mmask16 dmsk = dropout_mask[i/16];
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm512_storeu_ps(out+i, vout);
  }
  if (i < M) {
    int rem = M - i;
    __mmask16 mask = (1 << rem) - 1;
    __m512 vin = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(_mm256_maskz_loadu_epi16(mask, (__m256i*)(in+i))),16));
    __mmask16 dmsk = dropout_mask[i/16];
    __m512 vout = _mm512_maskz_mul_ps(dmsk, vin, vpi);
    _mm512_mask_storeu_ps(out+i, mask, vout);
  }
}
#else
void dropout_fwd_f32_f32_gold(unsigned int M, float *in, float *out, unsigned short *dropout_mask, void* rng_state, float p) {
  LIBXSMM_UNUSED( M );
  LIBXSMM_UNUSED( in );
  LIBXSMM_UNUSED( out );
  LIBXSMM_UNUSED( dropout_mask );
  LIBXSMM_UNUSED( rng_state );
  LIBXSMM_UNUSED( p );
  fprintf( stderr, "In order to run the dropout test you have to compile with AVX512BW support!\n" );
}

void dropout_fwd_bf16_bf16_gold(unsigned int M, libxsmm_bfloat16 *in, libxsmm_bfloat16 *out, unsigned short *dropout_mask, void* rng_state, float p) {
  LIBXSMM_UNUSED( M );
  LIBXSMM_UNUSED( in );
  LIBXSMM_UNUSED( out );
  LIBXSMM_UNUSED( dropout_mask );
  LIBXSMM_UNUSED( rng_state );
  LIBXSMM_UNUSED( p );
  fprintf( stderr, "In order to run the dropout test you have to compile with AVX512BW support!\n" );
}

void dropout_fwd_f32_bf16_gold(unsigned int M, float *in, libxsmm_bfloat16 *out, unsigned short *dropout_mask, void* rng_state, float p) {
  LIBXSMM_UNUSED( M );
  LIBXSMM_UNUSED( in );
  LIBXSMM_UNUSED( out );
  LIBXSMM_UNUSED( dropout_mask );
  LIBXSMM_UNUSED( rng_state );
  LIBXSMM_UNUSED( p );
  fprintf( stderr, "In order to run the dropout test you have to compile with AVX512BW support!\n" );
}

void dropout_fwd_bf16_f32_gold(unsigned int M, libxsmm_bfloat16 *in, float *out, unsigned short *dropout_mask, void* rng_state, float p) {
  LIBXSMM_UNUSED( M );
  LIBXSMM_UNUSED( in );
  LIBXSMM_UNUSED( out );
  LIBXSMM_UNUSED( dropout_mask );
  LIBXSMM_UNUSED( rng_state );
  LIBXSMM_UNUSED( p );
  fprintf( stderr, "In order to run the dropout test you have to compile with AVX512BW support!\n" );
}

void dropout_bwd_f32_f32_gold(unsigned int M, float *in, float *out, unsigned short *dropout_mask, float p) {
  LIBXSMM_UNUSED( M );
  LIBXSMM_UNUSED( in );
  LIBXSMM_UNUSED( out );
  LIBXSMM_UNUSED( dropout_mask );
  LIBXSMM_UNUSED( p );
  fprintf( stderr, "In order to run the dropout test you have to compile with AVX512BW support!\n" );
}

void dropout_bwd_bf16_bf16_gold(unsigned int M, libxsmm_bfloat16 *in, libxsmm_bfloat16 *out, unsigned short *dropout_mask, float p) {
  LIBXSMM_UNUSED( M );
  LIBXSMM_UNUSED( in );
  LIBXSMM_UNUSED( out );
  LIBXSMM_UNUSED( dropout_mask );
  LIBXSMM_UNUSED( p );
  fprintf( stderr, "In order to run the dropout test you have to compile with AVX512BW support!\n" );
}

void dropout_bwd_f32_bf16_gold(unsigned int M, float *in, libxsmm_bfloat16 *out, unsigned short *dropout_mask, float p) {
  LIBXSMM_UNUSED( M );
  LIBXSMM_UNUSED( in );
  LIBXSMM_UNUSED( out );
  LIBXSMM_UNUSED( dropout_mask );
  LIBXSMM_UNUSED( p );
  fprintf( stderr, "In order to run the dropout test you have to compile with AVX512BW support!\n" );
}

void dropout_bwd_bf16_f32_gold(unsigned int M, libxsmm_bfloat16 *in, float *out, unsigned short *dropout_mask, float p) {
  LIBXSMM_UNUSED( M );
  LIBXSMM_UNUSED( in );
  LIBXSMM_UNUSED( out );
  LIBXSMM_UNUSED( dropout_mask );
  LIBXSMM_UNUSED( p );
  fprintf( stderr, "In order to run the dropout test you have to compile with AVX512BW support!\n" );
}
#endif

void test_dropout_f32_f32_fwd( libxsmm_blasint bitm, libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  float *in;
  float *out, *out_gold;
  unsigned char *mask, *mask_gold;
  unsigned int *rng_state, *rng_state_gold;
  unsigned int i, j;
  unsigned int s;
  float p = 0.3f;
  libxsmm_meltw_dropout_param dropout_param;
  libxsmm_meltw_dropout_flags dropout_flags;
  libxsmm_blasint mask_ld = (bitm == 0) ? ldo : ldo/8;

  if ( M > ldi ) {
    fprintf( stderr, "test_dropout_f32_f32_fwd: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_dropout_f32_f32_fwd: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi,   64);
  out       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  out_gold  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  mask      = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*(mask_ld+1), 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = (double)(((i*ldi)+j)%4096);
    }
  }

  /* init out */
  for ( i = 0; i < N*ldo; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < N*ldo; ++i ) {
    out_gold[i] = 0;
  }
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask[i] = 0;
  }
  for ( i = 0; i < N*(mask_ld+1); ++i ) {
    mask_gold[i] = 0;
  }

  rng_state = libxsmm_rng_create_avx512_extstate( 555 );
  rng_state_gold = libxsmm_rng_create_avx512_extstate( 555 );

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    dropout_fwd_f32_f32_gold( M, &in[(i*ldi)], &out_gold[(i*ldo)], (unsigned short*)&mask_gold[(i*ldo)/8], rng_state_gold, p );
  }

  /* use jited tranpose */
  dropout_param.in_ptr  = (void*)in;
  dropout_param.out_ptr = (void*)out;
  dropout_param.mask_ptr = (bitm == 0) ? NULL : (void*)mask;
  dropout_param.prob_ptr = (void*)&p;
  dropout_param.rng_state = (void*)rng_state;
  dropout_flags = (bitm == 0) ? LIBXSMM_MELTW_FLAG_DROPOUT_FWD : LIBXSMM_MELTW_FLAG_DROPOUT_FWD_BITMASK;
  libxsmm_meltwfunction_dropout dropout_kernel = libxsmm_dispatch_meltw_dropout(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, dropout_flags);
  if ( dropout_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
    exit(-1);
  }
  dropout_kernel( &dropout_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( out_gold[(i*ldo)+j] != out[(i*ldo)+j] ) {
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
  if ( bitm != 0 ) {
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
    }
  }

  libxsmm_rng_destroy_avx512_extstate( rng_state );
  libxsmm_rng_destroy_avx512_extstate( rng_state_gold );

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( mask );
  libxsmm_free( mask_gold );
}

void test_dropout_bf16_bf16_fwd( libxsmm_blasint bitm, libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  libxsmm_bfloat16 *in;
  libxsmm_bfloat16 *out, *out_gold;
  unsigned char *mask, *mask_gold;
  unsigned int *rng_state, *rng_state_gold;
  unsigned int i, j;
  unsigned int s;
  float p = 0.3f;
  libxsmm_meltw_dropout_param dropout_param;
  libxsmm_meltw_dropout_flags dropout_flags;
  union libxsmm_bfloat16_hp bf16_hp;
  union libxsmm_bfloat16_hp bf16_hp_2;
  libxsmm_blasint mask_ld = (bitm == 0) ? ldo : ldo/8;

  if ( M > ldi ) {
    fprintf( stderr, "test_dropout_bf16_bf16_fwd: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_dropout_bf16_bf16_fwd: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in        = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldi,   64);
  out       = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,   64);
  out_gold  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,   64);
  mask      = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*(mask_ld+1), 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      bf16_hp.f = (float)(((i*ldi)+j)%4096);
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
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask[i] = 0;
  }
  for ( i = 0; i < N*(mask_ld+1); ++i ) {
    mask_gold[i] = 0;
  }

  rng_state = libxsmm_rng_create_avx512_extstate( 555 );
  rng_state_gold = libxsmm_rng_create_avx512_extstate( 555 );

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    dropout_fwd_bf16_bf16_gold( M, &in[(i*ldi)], &out_gold[(i*ldo)], (unsigned short*)&mask_gold[(i*ldo)/8], rng_state_gold, p );
  }

  /* use jited tranpose */
  dropout_param.in_ptr  = (void*)in;
  dropout_param.out_ptr = (void*)out;
  dropout_param.mask_ptr = (bitm == 0) ? NULL : (void*)mask;
  dropout_param.prob_ptr = (void*)&p;
  dropout_param.rng_state = (void*)rng_state;
  dropout_flags = (bitm == 0) ? LIBXSMM_MELTW_FLAG_DROPOUT_FWD : LIBXSMM_MELTW_FLAG_DROPOUT_FWD_BITMASK;
  libxsmm_meltwfunction_dropout dropout_kernel = libxsmm_dispatch_meltw_dropout(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, dropout_flags);
  if ( dropout_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
    exit(-1);
  }
  dropout_kernel( &dropout_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( out_gold[(i*ldo)+j] != out[(i*ldo)+j] ) {
        bf16_hp.i[1] = out[(i*ldo)+j];
        bf16_hp_2.i[1] = out_gold[(i*ldo)+j];
        bf16_hp.i[0] = 0;
        bf16_hp_2.i[0] = 0;
        printf("error at possition i=%i, j=%i, %f, %f\n", i, j, bf16_hp.f, bf16_hp_2.f);
        s = 1;
      }
#if 0
      else {
        printf("correct at possition i=%i, j=%i, %u, %u\n", i, j, out[(i*ldo)+j], out_gold[(i*ldo)+j]);
      }
#endif
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS output\n");
  } else {
    printf("FAILURE output\n");
  }
  if ( bitm != 0 ) {
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
    }
  }

  libxsmm_rng_destroy_avx512_extstate( rng_state );
  libxsmm_rng_destroy_avx512_extstate( rng_state_gold );

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( mask );
  libxsmm_free( mask_gold );
}

void test_dropout_f32_bf16_fwd( libxsmm_blasint bitm, libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  float *in;
  libxsmm_bfloat16 *out, *out_gold;
  unsigned char *mask, *mask_gold;
  unsigned int *rng_state, *rng_state_gold;
  unsigned int i, j;
  unsigned int s;
  float p = 0.3f;
  libxsmm_meltw_dropout_param dropout_param;
  libxsmm_meltw_dropout_flags dropout_flags;
  union libxsmm_bfloat16_hp bf16_hp;
  union libxsmm_bfloat16_hp bf16_hp_2;
  libxsmm_blasint mask_ld = (bitm == 0) ? ldo : ldo/8;

  if ( M > ldi ) {
    fprintf( stderr, "test_dropout_f32_bf16_fwd: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_dropout_f32_bf16_fwd: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi,   64);
  out       = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,   64);
  out_gold  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,   64);
  mask      = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*(mask_ld+1), 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = (float)(((i*ldi)+j)%4096);
    }
  }

  /* init out */
  for ( i = 0; i < N*ldo; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < N*ldo; ++i ) {
    out_gold[i] = 0;
  }
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask[i] = 0;
  }
  for ( i = 0; i < N*(mask_ld+1); ++i ) {
    mask_gold[i] = 0;
  }

  rng_state = libxsmm_rng_create_avx512_extstate( 555 );
  rng_state_gold = libxsmm_rng_create_avx512_extstate( 555 );

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    dropout_fwd_f32_bf16_gold( M, &in[(i*ldi)], &out_gold[(i*ldo)], (unsigned short*)&mask_gold[(i*ldo)/8], rng_state_gold, p );
  }

  /* use jited tranpose */
  dropout_param.in_ptr  = (void*)in;
  dropout_param.out_ptr = (void*)out;
  dropout_param.mask_ptr = (bitm == 0) ? NULL : (void*)mask;
  dropout_param.prob_ptr = (void*)&p;
  dropout_param.rng_state = (void*)rng_state;
  dropout_flags = (bitm == 0) ? LIBXSMM_MELTW_FLAG_DROPOUT_FWD : LIBXSMM_MELTW_FLAG_DROPOUT_FWD_BITMASK;
  libxsmm_meltwfunction_dropout dropout_kernel = libxsmm_dispatch_meltw_dropout(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, dropout_flags);
  if ( dropout_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
    exit(-1);
  }
  dropout_kernel( &dropout_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( out_gold[(i*ldo)+j] != out[(i*ldo)+j] ) {
        bf16_hp.i[1] = out[(i*ldo)+j];
        bf16_hp_2.i[1] = out_gold[(i*ldo)+j];
        bf16_hp.i[0] = 0;
        bf16_hp_2.i[0] = 0;
        printf("error at possition i=%i, j=%i, %f, %f\n", i, j, bf16_hp.f, bf16_hp_2.f);
        s = 1;
      }
#if 0
      else {
        printf("correct at possition i=%i, j=%i, %u, %u\n", i, j, out[(i*ldo)+j], out_gold[(i*ldo)+j]);
      }
#endif
    }
  }
  if ( s == 0 ) {
    printf("SUCCESS output\n");
  } else {
    printf("FAILURE output\n");
  }
  if ( bitm != 0 ) {
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
    }
  }

  libxsmm_rng_destroy_avx512_extstate( rng_state );
  libxsmm_rng_destroy_avx512_extstate( rng_state_gold );

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( mask );
  libxsmm_free( mask_gold );
}

void test_dropout_bf16_f32_fwd( libxsmm_blasint bitm, libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  libxsmm_bfloat16 *in;
  float *out, *out_gold;
  unsigned char *mask, *mask_gold;
  unsigned int *rng_state, *rng_state_gold;
  unsigned int i, j;
  unsigned int s;
  float p = 0.3f;
  libxsmm_meltw_dropout_param dropout_param;
  libxsmm_meltw_dropout_flags dropout_flags;
  union libxsmm_bfloat16_hp bf16_hp;
  libxsmm_blasint mask_ld = (bitm == 0) ? ldo : ldo/8;

  if ( M > ldi ) {
    fprintf( stderr, "test_dropout_bf16_f32_fwd: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_dropout_bf16_f32_fwd: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in        = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldi,   64);
  out       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  out_gold  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  mask      = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*(mask_ld+1), 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      bf16_hp.f = (float)(((i*ldi)+j)%4096);
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
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask[i] = 0;
  }
  for ( i = 0; i < N*(mask_ld+1); ++i ) {
    mask_gold[i] = 0;
  }

  rng_state = libxsmm_rng_create_avx512_extstate( 555 );
  rng_state_gold = libxsmm_rng_create_avx512_extstate( 555 );

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    dropout_fwd_bf16_f32_gold( M, &in[(i*ldi)], &out_gold[(i*ldo)], (unsigned short*)&mask_gold[(i*ldo)/8], rng_state_gold, p );
  }

  /* use jited tranpose */
  dropout_param.in_ptr  = (void*)in;
  dropout_param.out_ptr = (void*)out;
  dropout_param.mask_ptr = (bitm == 0) ? NULL : (void*)mask;
  dropout_param.prob_ptr = (void*)&p;
  dropout_param.rng_state = (void*)rng_state;
  dropout_flags = (bitm == 0) ? LIBXSMM_MELTW_FLAG_DROPOUT_FWD : LIBXSMM_MELTW_FLAG_DROPOUT_FWD_BITMASK;
  libxsmm_meltwfunction_dropout dropout_kernel = libxsmm_dispatch_meltw_dropout(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, dropout_flags);
  if ( dropout_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
    exit(-1);
  }
  dropout_kernel( &dropout_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( out_gold[(i*ldo)+j] != out[(i*ldo)+j] ) {
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
  if ( bitm != 0 ) {
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
    }
  }

  libxsmm_rng_destroy_avx512_extstate( rng_state );
  libxsmm_rng_destroy_avx512_extstate( rng_state_gold );

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( mask );
  libxsmm_free( mask_gold );
}

void test_dropout_f32_f32_bwd( libxsmm_blasint bitm, libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  float *in;
  float *out, *out_gold;
  unsigned int *mask;
  unsigned char *mask_gold;
  unsigned int i, j;
  unsigned int s;
  float p = 0.3f;
  libxsmm_meltw_dropout_param dropout_param;
  libxsmm_meltw_dropout_flags dropout_flags;
  libxsmm_blasint mask_ld = (bitm == 0) ? ldi : ldi/8;

  if ( M > ldi ) {
    fprintf( stderr, "test_dropout_f32_f32_fwd: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_dropout_f32_f32_fwd: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi,   64);
  out       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  out_gold  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  mask      = (unsigned int*) libxsmm_aligned_malloc( sizeof(unsigned int)*N*mask_ld, 64);
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*(mask_ld+1), 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = (double)(((i*ldi)+j)%4096);
    }
  }

  /* init out */
  for ( i = 0; i < N*ldo; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < N*ldo; ++i ) {
    out_gold[i] = 0;
  }
  for ( i = 0; i < N*mask_ld; ++i ) {
    if ( bitm == 0 ) {
      mask[i] = ( i % 2 == 1) ? 0x3f800000 : 0x0;
    } else {
      mask[i] = 0xaaaaaaaa;
    }
  }
  for ( i = 0; i < N*(mask_ld+1); ++i ) {
    mask_gold[i] = 0xaa;
  }

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    dropout_bwd_f32_f32_gold( M, &in[(i*ldi)], &out_gold[(i*ldo)], (unsigned short*)&mask_gold[(i*ldo)/8], p );
  }

  /* use jited tranpose */
  dropout_param.in_ptr  = (void*)in;
  dropout_param.out_ptr = (void*)out;
  dropout_param.mask_ptr = (void*)mask;
  dropout_param.prob_ptr = (void*)&p;
  dropout_param.rng_state = NULL;
  dropout_flags = (bitm == 0) ? LIBXSMM_MELTW_FLAG_DROPOUT_BWD : LIBXSMM_MELTW_FLAG_DROPOUT_BWD_BITMASK;
  libxsmm_meltwfunction_dropout dropout_kernel = libxsmm_dispatch_meltw_dropout(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, dropout_flags);
  if ( dropout_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
    exit(-1);
  }
  dropout_kernel( &dropout_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( out_gold[(i*ldo)+j] != out[(i*ldo)+j] ) {
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
  libxsmm_free( mask );
  libxsmm_free( mask_gold );
}

void test_dropout_bf16_bf16_bwd( libxsmm_blasint bitm, libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  libxsmm_bfloat16 *in;
  libxsmm_bfloat16 *out, *out_gold;
  libxsmm_bfloat16 *mask;
  unsigned char *mask_gold;
  unsigned int i, j;
  unsigned int s;
  float p = 0.3f;
  libxsmm_meltw_dropout_param dropout_param;
  libxsmm_meltw_dropout_flags dropout_flags;
  union libxsmm_bfloat16_hp bf16_hp;
  union libxsmm_bfloat16_hp bf16_hp_2;
  libxsmm_blasint mask_ld = (bitm == 0) ? ldi : ldi/8;

  if ( M > ldi ) {
    fprintf( stderr, "test_dropout_bf16_bf16_bwd: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_dropout_bf16_bf16_bwd: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in        = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldi,   64);
  out       = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,   64);
  out_gold  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,   64);
  mask      = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*mask_ld, 64);
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*(mask_ld+1), 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      bf16_hp.f = (float)(((i*ldi)+j)%4096);
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
  for ( i = 0; i < N*mask_ld; ++i ) {
    if ( bitm == 0 ) {
      bf16_hp.f = 1.0f;
      mask[i] = ( i % 2 == 1) ? bf16_hp.i[1] : 0;
    } else {
      mask[i] = 0xaaaa;
    }
  }
  for ( i = 0; i < N*(mask_ld+1); ++i ) {
    mask_gold[i] = 0xaa;
  }

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    dropout_bwd_bf16_bf16_gold( M, &in[(i*ldi)], &out_gold[(i*ldo)], (unsigned short*)&mask_gold[(i*ldo)/8], p );
  }

  /* use jited tranpose */
  dropout_param.in_ptr  = (void*)in;
  dropout_param.out_ptr = (void*)out;
  dropout_param.mask_ptr = (void*)mask;
  dropout_param.prob_ptr = (void*)&p;
  dropout_param.rng_state = NULL;
  dropout_flags = (bitm == 0) ? LIBXSMM_MELTW_FLAG_DROPOUT_BWD : LIBXSMM_MELTW_FLAG_DROPOUT_BWD_BITMASK;
  libxsmm_meltwfunction_dropout dropout_kernel = libxsmm_dispatch_meltw_dropout(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, dropout_flags);
  if ( dropout_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
    exit(-1);
  }
  dropout_kernel( &dropout_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( out_gold[(i*ldo)+j] != out[(i*ldo)+j] ) {
        bf16_hp.i[1] = out[(i*ldo)+j];
        bf16_hp_2.i[1] = out_gold[(i*ldo)+j];
        bf16_hp.i[0] = 0;
        bf16_hp_2.i[0] = 0;
        printf("error at possition i=%i, j=%i, %f, %f\n", i, j, bf16_hp.f, bf16_hp_2.f);
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
  libxsmm_free( mask );
  libxsmm_free( mask_gold );
}

void test_dropout_f32_bf16_bwd( libxsmm_blasint bitm, libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  float *in;
  libxsmm_bfloat16 *out, *out_gold;
  unsigned int *mask;
  unsigned char *mask_gold;
  unsigned int i, j;
  unsigned int s;
  float p = 0.3f;
  libxsmm_meltw_dropout_param dropout_param;
  libxsmm_meltw_dropout_flags dropout_flags;
  union libxsmm_bfloat16_hp bf16_hp;
  union libxsmm_bfloat16_hp bf16_hp_2;
  libxsmm_blasint mask_ld = (bitm == 0) ? ldi : ldi/8;

  if ( M > ldi ) {
    fprintf( stderr, "test_dropout_f32_bf16_bwd: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_dropout_f32_bf16_bwd: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi,   64);
  out       = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,   64);
  out_gold  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,   64);
  mask      = (unsigned int*) libxsmm_aligned_malloc( sizeof(unsigned int)*N*mask_ld, 64);
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*(mask_ld+1), 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      in[(i*ldi)+j] = (float)(((i*ldi)+j)%4096);
    }
  }

  /* init out */
  for ( i = 0; i < N*ldo; ++i ) {
    out[i] = 0;
  }
  for ( i = 0; i < N*ldo; ++i ) {
    out_gold[i] = 0;
  }
  for ( i = 0; i < N*mask_ld; ++i ) {
    if ( bitm == 0 ) {
      mask[i] = ( i % 2 == 1) ? 0x3f800000 : 0x0;
    } else {
      mask[i] = 0xaaaaaaaa;
    }
  }
  for ( i = 0; i < N*(mask_ld+1); ++i ) {
    mask_gold[i] = 0xaa;
  }

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    dropout_bwd_f32_bf16_gold( M, &in[(i*ldi)], &out_gold[(i*ldo)], (unsigned short*)&mask_gold[(i*ldo)/8], p );
  }

  /* use jited tranpose */
  dropout_param.in_ptr  = (void*)in;
  dropout_param.out_ptr = (void*)out;
  dropout_param.mask_ptr = (void*)mask;
  dropout_param.prob_ptr = (void*)&p;
  dropout_param.rng_state = NULL;
  dropout_flags = (bitm == 0) ? LIBXSMM_MELTW_FLAG_DROPOUT_BWD : LIBXSMM_MELTW_FLAG_DROPOUT_BWD_BITMASK;
  libxsmm_meltwfunction_dropout dropout_kernel = libxsmm_dispatch_meltw_dropout(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, dropout_flags);
  if ( dropout_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
    exit(-1);
  }
  dropout_kernel( &dropout_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( out_gold[(i*ldo)+j] != out[(i*ldo)+j] ) {
        bf16_hp.i[1] = out[(i*ldo)+j];
        bf16_hp_2.i[1] = out_gold[(i*ldo)+j];
        bf16_hp.i[0] = 0;
        bf16_hp_2.i[0] = 0;
        printf("error at possition i=%i, j=%i, %f, %f\n", i, j, bf16_hp.f, bf16_hp_2.f);
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
  libxsmm_free( mask );
  libxsmm_free( mask_gold );
}

void test_dropout_bf16_f32_bwd( libxsmm_blasint bitm, libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  libxsmm_bfloat16 *in;
  float *out, *out_gold;
  libxsmm_bfloat16 *mask;
  unsigned char*mask_gold;
  unsigned int i, j;
  unsigned int s;
  float p = 0.3f;
  libxsmm_meltw_dropout_param dropout_param;
  libxsmm_meltw_dropout_flags dropout_flags;
  union libxsmm_bfloat16_hp bf16_hp;
  libxsmm_blasint mask_ld = (bitm == 0) ? ldo : ldo/8;

  if ( M > ldi ) {
    fprintf( stderr, "test_dropout_bf16_f32_bwd: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_dropout_bf16_f32_bwd: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in        = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldi,   64);
  out       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  out_gold  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldo,   64);
  mask      = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*mask_ld, 64);
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*(mask_ld+1), 64);

  /* init in */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      bf16_hp.f = (float)(((i*ldi)+j)%4096);
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
  for ( i = 0; i < N*mask_ld; ++i ) {
    if ( bitm == 0 ) {
      bf16_hp.f = 1.0f;
      mask[i] = ( i % 2 == 1) ? bf16_hp.i[1] : 0;
    } else {
      mask[i] = 0xaaaa;
    }
  }
  for ( i = 0; i < N*(mask_ld+1); ++i ) {
    mask_gold[i] = 0xaa;
  }

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    dropout_bwd_bf16_f32_gold( M, &in[(i*ldi)], &out_gold[(i*ldo)], (unsigned short*)&mask_gold[(i*ldo)/8], p );
  }

  /* use jited tranpose */
  dropout_param.in_ptr  = (void*)in;
  dropout_param.out_ptr = (void*)out;
  dropout_param.mask_ptr = (void*)mask;
  dropout_param.prob_ptr = (void*)&p;
  dropout_param.rng_state = NULL;
  dropout_flags = (bitm == 0) ? LIBXSMM_MELTW_FLAG_DROPOUT_BWD : LIBXSMM_MELTW_FLAG_DROPOUT_BWD_BITMASK;
  libxsmm_meltwfunction_dropout dropout_kernel = libxsmm_dispatch_meltw_dropout(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, dropout_flags);
  if ( dropout_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
    exit(-1);
  }
  dropout_kernel( &dropout_param );

  /* compare result */
  s = 0;
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      if ( out_gold[(i*ldo)+j] != out[(i*ldo)+j] ) {
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
  libxsmm_free( mask );
  libxsmm_free( mask_gold );
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

  if ( op == 'F' && dtype_in == 4 && dtype_out == 4  ) {
    printf("Testing F32 F32 forward dropout\n");
    test_dropout_f32_f32_fwd( bitm, M, N, ldi, ldo );
  } else if ( op == 'F' && dtype_in == 2  && dtype_out == 2 ) {
    printf("Testing BF16 BF16 forward dropout\n");
    test_dropout_bf16_bf16_fwd( bitm, M, N, ldi, ldo );
  } else if ( op == 'F' && dtype_in == 4  && dtype_out == 2 ) {
    printf("Testing F32 BF16 forward dropout\n");
    test_dropout_f32_bf16_fwd( bitm, M, N, ldi, ldo );
  } else if ( op == 'F' && dtype_in == 2  && dtype_out == 4 ) {
    printf("Testing BF16 F32 forward dropout\n");
    test_dropout_bf16_f32_fwd( bitm, M, N, ldi, ldo );
  } else if ( op == 'B' && dtype_in == 4 && dtype_out == 4 ) {
    printf("Testing F32 F32 backward dropout\n");
    test_dropout_f32_f32_bwd( bitm, M, N, ldi, ldo );
  } else if ( op == 'B' && dtype_in == 2 && dtype_out == 2 ) {
    printf("Testing BF16 BF16 backward dropout\n");
    test_dropout_bf16_bf16_bwd( bitm, M, N, ldi, ldo );
  } else if ( op == 'B' && dtype_in == 4 && dtype_out == 2 ) {
    printf("Testing F32 BF16 backward dropout\n");
    test_dropout_f32_bf16_bwd( bitm, M, N, ldi, ldo );
  } else if ( op == 'B' && dtype_in == 2 && dtype_out == 4 ) {
    printf("Testing BF16 F32 backward dropout\n");
    test_dropout_bf16_f32_bwd( bitm, M, N, ldi, ldo );
  } else {
    printf(" Not implemented case! Usage: %s [F/B] [bitmask: 0/1] [prec_in: 4/2] [prec_out: 4/2] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }
}
