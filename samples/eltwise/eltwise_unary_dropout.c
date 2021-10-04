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
#else
#ifdef __AVX2__
#include <libxsmm_intrinsics_x86.h>
#endif
#endif

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

#ifdef __AVX512BW__
void dropout_fwd_f32_f32_gold(unsigned int M, float *in, float *out, unsigned short *dropout_mask, void* rng_state, float p) {
  unsigned int i;
  float pn = 1 - p;
  __m512 vp = _mm512_set1_ps(pn);
  __m512 vpi = _mm512_set1_ps(1.0/pn);
  for (i = 0; i < LIBXSMM_ALIGNDOWN(M, 16); i+=16) {
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
  for (i = 0; i < LIBXSMM_ALIGNDOWN(M, 16); i+=16) {
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
  for (i = 0; i < LIBXSMM_ALIGNDOWN(M, 16); i+=16) {
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
  for (i = 0; i < LIBXSMM_ALIGNDOWN(M, 16); i+=16) {
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
  for (i = 0; i < LIBXSMM_ALIGNDOWN(M, 16); i+=16) {
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
  for (i = 0; i < LIBXSMM_ALIGNDOWN(M, 16); i+=16) {
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
  for (i = 0; i < LIBXSMM_ALIGNDOWN(M, 16); i+=16) {
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
  for (i = 0; i < LIBXSMM_ALIGNDOWN(M, 16); i+=16) {
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
#ifdef __AVX2__
void dropout_fwd_f32_f32_gold(unsigned int M, float *in, float *out, unsigned short *dropout_mask, void* rng_state, float p) {
  unsigned int i;
  float pn = 1 - p;
  unsigned char* out_mask = (unsigned char*)dropout_mask;
  __m256 vp = _mm256_set1_ps(pn);
  __m256 vpi = _mm256_set1_ps(1.0/pn);
  __m256 vzero = _mm256_setzero_ps();
  for (i = 0; i < LIBXSMM_ALIGNDOWN(M, 8); i+=8) {
    __m256 rnd = LIBXSMM_INTRINSICS_MM256_RNG_EXTSTATE_PS(rng_state);
    __m256 vin = _mm256_loadu_ps(in+i);
    __m256 vmsk = _mm256_cmp_ps(vp, rnd, 0x06);
    __m256 vout = _mm256_mul_ps(vin, vpi);
    vout = _mm256_blendv_ps( vzero, vout, vmsk );
    _mm256_storeu_ps(out+i, vout);
    out_mask[i/8] = _mm256_movemask_ps( vmsk );
  }
  if (i < M) {
    int rem = M - i;
    __m256 rnd = LIBXSMM_INTRINSICS_MM256_RNG_EXTSTATE_PS(rng_state);
    __m256i memmsk = _mm256_set_epi32( (rem == 8) ? 0xffffffff : 0x00000000, (rem >= 7) ? 0xffffffff : 0x00000000, (rem >= 6) ? 0xffffffff : 0x00000000, (rem >= 5) ? 0xffffffff : 0x00000000,
                                       (rem >= 4) ? 0xffffffff : 0x00000000, (rem >= 3) ? 0xffffffff : 0x00000000, (rem >= 2) ? 0xffffffff : 0x00000000, (rem >= 1) ? 0xffffffff : 0x00000000  );
    __m256 vin = _mm256_maskload_ps(in+i, memmsk);
    __m256 vmsk = _mm256_cmp_ps(vp, rnd, 0x06);
    __m256 vout = _mm256_mul_ps(vin, vpi);
    vout = _mm256_blendv_ps( vzero, vout, vmsk );
    _mm256_maskstore_ps(out+i, memmsk, vout);
    out_mask[i/8] = _mm256_movemask_ps( vmsk );
  }
}

void dropout_bwd_f32_f32_gold(unsigned int M, float *in, float *out, unsigned short *dropout_mask, float p) {
  unsigned int i = 0;
  float pn = 1 - p;
  unsigned char* out_mask = (unsigned char*)dropout_mask;
  __m256 vpi = _mm256_set1_ps(1.0/pn);
  __m256 vzero = _mm256_setzero_ps();
  __m256i vand = _mm256_set_epi32( 0x00000080, 0x00000040, 0x00000020, 0x00000010, 0x00000008, 0x00000004, 0x00000002, 0x00000001 );
  for (i = 0; i < LIBXSMM_ALIGNDOWN(M, 8); i+=8) {
    __m256 vin = _mm256_loadu_ps(in+i);
    __m256i vmask = _mm256_set1_epi8( out_mask[i/8] );
    vmask = _mm256_and_si256( vmask, vand );
    vmask = _mm256_cmpeq_epi32( vmask, vand );
    __m256 vout = _mm256_mul_ps(vin, vpi);
    vout = _mm256_blendv_ps( vzero, vout, _mm256_castsi256_ps(vmask) );
    _mm256_storeu_ps(out+i, vout);
  }
  if (i < M) {
    int rem = M - i;
    __m256i memmsk = _mm256_set_epi32( (rem == 8) ? 0xffffffff : 0x00000000, (rem >= 7) ? 0xffffffff : 0x00000000, (rem >= 6) ? 0xffffffff : 0x00000000, (rem >= 5) ? 0xffffffff : 0x00000000,
                                       (rem >= 4) ? 0xffffffff : 0x00000000, (rem >= 3) ? 0xffffffff : 0x00000000, (rem >= 2) ? 0xffffffff : 0x00000000, (rem >= 1) ? 0xffffffff : 0x00000000  );
    __m256 vin = _mm256_maskload_ps(in+i, memmsk);
    __m256i vmask = _mm256_set1_epi8( out_mask[i/8] );
    vmask = _mm256_and_si256( vmask, vand );
    vmask = _mm256_cmpeq_epi32( vmask, vand );
    __m256 vout = _mm256_mul_ps(vin, vpi);
    vout = _mm256_blendv_ps( vzero, vout, _mm256_castsi256_ps(vmask) );
    _mm256_maskstore_ps(out+i, memmsk, vout);
  }
}
#else
void lsfr_Xwide( unsigned int* rng_state, float* prng_out, unsigned int width ) {
  const unsigned int state_ld = 16;
  const float one = 1.0f;
  unsigned int w;
  union { unsigned int i; float f; } rng_num;

  for ( w = 0 ; w < width; ++w ) {
    /* setup */
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

void dropout_fwd_f32_f32_gold(unsigned int M, float *in, float *out, unsigned char *dropout_mask, void* rng_state, float p) {
  float vrng[16];
  unsigned int w = 4;
  unsigned int i;
  unsigned int j;
  float pn = 1 - p;
  float pi = 1/pn;

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

void dropout_bwd_f32_f32_gold(unsigned int M, float *in, float *out, unsigned short *dropout_mask, float p) {
  LIBXSMM_UNUSED( M );
  LIBXSMM_UNUSED( in );
  LIBXSMM_UNUSED( out );
  LIBXSMM_UNUSED( dropout_mask );
  LIBXSMM_UNUSED( p );
  fprintf( stderr, "In order to run the dropout test you have to compile with AVX512BW support!\n" );
}
#endif

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

int test_dropout_f32_f32_fwd( libxsmm_blasint bitm, libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  float *in;
  float *out, *out_gold;
  unsigned char *mask, *mask_gold;
  unsigned int *rng_state, *rng_state_gold;
  unsigned int i, j;
  unsigned int s;
  float p = 0.3f;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  libxsmm_blasint mask_ld = (bitm == 0) ? ldo : ((ldo+15)-((ldo+15)%16))/8;

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
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);

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
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask_gold[i] = 0;
  }

  rng_state = libxsmm_rng_create_extstate( 555 );
  rng_state_gold = libxsmm_rng_create_extstate( 555 );

#ifdef USE_ZERO_RNG_STATE_UNITTEST
  memset( (void*)rng_state, 0, libxsmm_rng_get_extstate_size() );
  memset( (void*)rng_state_gold, 0, libxsmm_rng_get_extstate_size() );
#endif

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    dropout_fwd_f32_f32_gold( M, &in[(i*ldi)], &out_gold[(i*ldo)], (unsigned char*)&mask_gold[(i*mask_ld)], rng_state_gold, p );
  }

  /* use jited tranpose */
  unary_param.op.primary = (void*)&p;
  unary_param.op.secondary = (void*)rng_state;
  unary_param.in.primary  = (void*)in;
  unary_param.out.primary = (void*)out;
  unary_param.out.secondary = (bitm == 0) ? NULL : (void*)mask;
  unary_flags = (bitm == 0) ? LIBXSMM_MELTW_FLAG_UNARY_NONE : LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  libxsmm_meltwfunction_unary unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_DROPOUT);
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
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

  if ( norms_out.normf_rel > 0.00001 ) {
    ret = EXIT_FAILURE;
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
    printf("SUCCESS unary dropout fwd fp32 fp32\n");
  } else {
    printf("FAILURE unary dropout fwd fp32 fp32\n");
  }

  return ret;
}

int test_dropout_bf16_bf16_fwd( libxsmm_blasint bitm, libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  libxsmm_bfloat16 *in;
  libxsmm_bfloat16 *out, *out_gold;
  float *f32out, *f32out_gold;
  unsigned char *mask, *mask_gold;
  unsigned int *rng_state, *rng_state_gold;
  unsigned int i, j;
  unsigned int s;
  float p = 0.3f;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  union libxsmm_bfloat16_hp bf16_hp;
  libxsmm_blasint mask_ld = (bitm == 0) ? ldo : ((ldo+15)-((ldo+15)%16))/8;

  if ( M > ldi ) {
    fprintf( stderr, "test_dropout_bf16_bf16_fwd: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_dropout_bf16_bf16_fwd: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in          = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldi,   64);
  out         = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,   64);
  out_gold    = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,   64);
  f32out      = (float*)            libxsmm_aligned_malloc( sizeof(float)*N*ldo,              64);
  f32out_gold = (float*)            libxsmm_aligned_malloc( sizeof(float)*N*ldo,              64);
  mask        = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld,     64);
  mask_gold   = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld,     64);

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
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask_gold[i] = 0;
  }

  rng_state = libxsmm_rng_create_extstate( 555 );
  rng_state_gold = libxsmm_rng_create_extstate( 555 );

#ifdef USE_ZERO_RNG_STATE_UNITTEST
  memset( (void*)rng_state, 0, libxsmm_rng_get_extstate_size() );
  memset( (void*)rng_state_gold, 0, libxsmm_rng_get_extstate_size() );
#endif

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    dropout_fwd_bf16_bf16_gold( M, &in[(i*ldi)], &out_gold[(i*ldo)], (unsigned short*)&mask_gold[(i*mask_ld)], rng_state_gold, p );
  }

  /* use jited tranpose */
  unary_param.op.primary = (void*)&p;
  unary_param.op.secondary = (void*)rng_state;
  unary_param.in.primary  = (void*)in;
  unary_param.out.primary = (void*)out;
  unary_param.out.secondary = (bitm == 0) ? NULL : (void*)mask;
  unary_flags = (bitm == 0) ? LIBXSMM_MELTW_FLAG_UNARY_NONE : LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  libxsmm_meltwfunction_unary unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_DROPOUT);
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
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

  if ( norms_out.normf_rel > 0.005 ) {
    ret = EXIT_FAILURE;
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
  libxsmm_free( f32out_gold );
  libxsmm_free( f32out );
  libxsmm_free( in );
  libxsmm_free( mask );
  libxsmm_free( mask_gold );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary dropout fwd bf16 bf16\n");
  } else {
    printf("FAILURE unary dropout fwd bf16 bf16\n");
  }

  return ret;
}

int test_dropout_f32_bf16_fwd( libxsmm_blasint bitm, libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  float *in;
  libxsmm_bfloat16 *out, *out_gold;
  float *f32out, *f32out_gold;
  unsigned char *mask, *mask_gold;
  unsigned int *rng_state, *rng_state_gold;
  unsigned int i, j;
  unsigned int s;
  float p = 0.3f;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  libxsmm_blasint mask_ld = (bitm == 0) ? ldo : ((ldo+15)-((ldo+15)%16))/8;

  if ( M > ldi ) {
    fprintf( stderr, "test_dropout_f32_bf16_fwd: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_dropout_f32_bf16_fwd: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in          = (float*)            libxsmm_aligned_malloc( sizeof(float)*N*ldi,              64);
  out         = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,   64);
  out_gold    = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,   64);
  f32out      = (float*)            libxsmm_aligned_malloc( sizeof(float)*N*ldo,              64);
  f32out_gold = (float*)            libxsmm_aligned_malloc( sizeof(float)*N*ldo,              64);
  mask        = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld,     64);
  mask_gold   = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld,     64);

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
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask_gold[i] = 0;
  }

  rng_state = libxsmm_rng_create_extstate( 555 );
  rng_state_gold = libxsmm_rng_create_extstate( 555 );

#ifdef USE_ZERO_RNG_STATE_UNITTEST
  memset( (void*)rng_state, 0, libxsmm_rng_get_extstate_size() );
  memset( (void*)rng_state_gold, 0, libxsmm_rng_get_extstate_size() );
#endif

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    dropout_fwd_f32_bf16_gold( M, &in[(i*ldi)], &out_gold[(i*ldo)], (unsigned short*)&mask_gold[(i*mask_ld)], rng_state_gold, p );
  }

  /* use jited tranpose */
  unary_param.op.primary = (void*)&p;
  unary_param.op.secondary = (void*)rng_state;
  unary_param.in.primary  = (void*)in;
  unary_param.out.primary = (void*)out;
  unary_param.out.secondary = (bitm == 0) ? NULL : (void*)mask;
  unary_flags = (bitm == 0) ? LIBXSMM_MELTW_FLAG_UNARY_NONE : LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  libxsmm_meltwfunction_unary unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_DROPOUT);
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
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

  if ( norms_out.normf_rel > 0.005 ) {
    ret = EXIT_FAILURE;
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
  libxsmm_free( f32out_gold );
  libxsmm_free( f32out );
  libxsmm_free( in );
  libxsmm_free( mask );
  libxsmm_free( mask_gold );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary dropout fwd fp32 bf16\n");
  } else {
    printf("FAILURE unary dropout fwd fp32 bf16\n");
  }

  return ret;
}

int test_dropout_bf16_f32_fwd( libxsmm_blasint bitm, libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  libxsmm_bfloat16 *in;
  float *out, *out_gold;
  unsigned char *mask, *mask_gold;
  unsigned int *rng_state, *rng_state_gold;
  unsigned int i, j;
  unsigned int s;
  float p = 0.3f;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  union libxsmm_bfloat16_hp bf16_hp;
  libxsmm_blasint mask_ld = (bitm == 0) ? ldo : ((ldo+15)-((ldo+15)%16))/8;

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
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);

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
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask_gold[i] = 0;
  }

  rng_state = libxsmm_rng_create_extstate( 555 );
  rng_state_gold = libxsmm_rng_create_extstate( 555 );

#ifdef USE_ZERO_RNG_STATE_UNITTEST
  memset( (void*)rng_state, 0, libxsmm_rng_get_extstate_size() );
  memset( (void*)rng_state_gold, 0, libxsmm_rng_get_extstate_size() );
#endif

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    dropout_fwd_bf16_f32_gold( M, &in[(i*ldi)], &out_gold[(i*ldo)], (unsigned short*)&mask_gold[(i*mask_ld)], rng_state_gold, p );
  }

  /* use jited tranpose */
  unary_param.op.primary = (void*)&p;
  unary_param.op.secondary = (void*)rng_state;
  unary_param.in.primary  = (void*)in;
  unary_param.out.primary = (void*)out;
  unary_param.out.secondary = (bitm == 0) ? NULL : (void*)mask;
  unary_flags = (bitm == 0) ? LIBXSMM_MELTW_FLAG_UNARY_NONE : LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  libxsmm_meltwfunction_unary unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_DROPOUT);
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
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

  if ( norms_out.normf_rel > 0.005 ) {
    ret = EXIT_FAILURE;
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
    printf("SUCCESS unary dropout fwd bf16 fp32\n");
  } else {
    printf("FAILURE unary dropout fwd bf16 fp32\n");
  }

  return ret;
}

int test_dropout_f32_f32_bwd( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  float *in;
  float *out, *out_gold;
  unsigned char *mask;
  unsigned char *mask_gold;
  unsigned int i, j;
  float p = 0.3f;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  libxsmm_blasint mask_ld = ((ldi+15)-((ldi+15)%16))/8;

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
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);

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
    mask[i] = 0xaa;
  }
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask_gold[i] = 0xaa;
  }

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    dropout_bwd_f32_f32_gold( M, &in[(i*ldi)], &out_gold[(i*ldo)], (unsigned short*)&mask_gold[(i*mask_ld)], p );
  }

  /* use jited tranpose */
  unary_param.op.primary = (void*)&p;
  unary_param.in.primary  = (void*)in;
  unary_param.in.secondary = (void*)mask;
  unary_param.out.primary = (void*)out;
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  libxsmm_meltwfunction_unary unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV);
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
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

  if ( norms_out.normf_rel > 0.00001 ) {
    ret = EXIT_FAILURE;
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

int test_dropout_bf16_bf16_bwd( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  libxsmm_bfloat16 *in;
  libxsmm_bfloat16 *out, *out_gold;
  float *f32out, *f32out_gold;
  unsigned char *mask;
  unsigned char *mask_gold;
  unsigned int i, j;
  float p = 0.3f;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  union libxsmm_bfloat16_hp bf16_hp;
  libxsmm_blasint mask_ld = ((ldi+15)-((ldi+15)%16))/8;

  if ( M > ldi ) {
    fprintf( stderr, "test_dropout_bf16_bf16_bwd: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_dropout_bf16_bf16_bwd: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in          = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldi,     64);
  out         = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,     64);
  out_gold    = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo,     64);
  f32out      = (float*)            libxsmm_aligned_malloc( sizeof(float)*N*ldo,                64);
  f32out_gold = (float*)            libxsmm_aligned_malloc( sizeof(float)*N*ldo,                64);
  mask        = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld,       64);
  mask_gold   = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld,       64);

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
    mask[i] = 0xaa;
  }
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask_gold[i] = 0xaa;
  }

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    dropout_bwd_bf16_bf16_gold( M, &in[(i*ldi)], &out_gold[(i*ldo)], (unsigned short*)&mask_gold[(i*mask_ld)], p );
  }

  /* use jited tranpose */
  unary_param.op.primary = (void*)&p;
  unary_param.in.primary  = (void*)in;
  unary_param.in.secondary = (void*)mask;
  unary_param.out.primary = (void*)out;
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  libxsmm_meltwfunction_unary unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV);
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
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

  if ( norms_out.normf_rel > 0.005 ) {
    ret = EXIT_FAILURE;
  }


  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( f32out_gold );
  libxsmm_free(f32out );
  libxsmm_free( in );
  libxsmm_free( mask );
  libxsmm_free( mask_gold );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary dropout bwd bf16 bf16\n");
  } else {
    printf("FAILURE unary dropout bwd bf16 bf16\n");
  }

  return ret;
}

int test_dropout_f32_bf16_bwd( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  float *in;
  libxsmm_bfloat16 *out, *out_gold;
  float *f32out, *f32out_gold;
  unsigned char *mask;
  unsigned char *mask_gold;
  unsigned int i, j;
  float p = 0.3f;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  libxsmm_blasint mask_ld = ((ldi+15)-((ldi+15)%16))/8;

  if ( M > ldi ) {
    fprintf( stderr, "test_dropout_f32_bf16_bwd: ldi needs to be equal to or bigger than M\n");
    exit(-1);
  }
  if (M > ldo ) {
    fprintf( stderr, "test_dropout_f32_bf16_bwd: ldo needs to be equal to or bigger than N\n");
    exit(-1);
  }

  in          = (float*) libxsmm_aligned_malloc( sizeof(float)*N*ldi,                       64);
  out         = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo, 64);
  out_gold    = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ldo, 64);
  f32out      = (float*)            libxsmm_aligned_malloc( sizeof(float)*N*ldo,            64);
  f32out_gold = (float*)            libxsmm_aligned_malloc( sizeof(float)*N*ldo,            64);
  mask        = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld,   64);
  mask_gold   = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld,   64);

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
    mask[i] = 0xaa;
  }
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask_gold[i] = 0xaa;
  }

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    dropout_bwd_f32_bf16_gold( M, &in[(i*ldi)], &out_gold[(i*ldo)], (unsigned short*)&mask_gold[(i*mask_ld)], p );
  }

  /* use jited tranpose */
  unary_param.op.primary = (void*)&p;
  unary_param.in.primary  = (void*)in;
  unary_param.in.secondary = (void*)mask;
  unary_param.out.primary = (void*)out;
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  libxsmm_meltwfunction_unary unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV);
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
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

  if ( norms_out.normf_rel > 0.005 ) {
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( f32out_gold );
  libxsmm_free( f32out );
  libxsmm_free( in );
  libxsmm_free( mask );
  libxsmm_free( mask_gold );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary dropout bwd fp32 bf16\n");
  } else {
    printf("FAILURE unary dropout bwd fp32 bf16\n");
  }

  return ret;
}

int test_dropout_bf16_f32_bwd( libxsmm_blasint M, libxsmm_blasint N, libxsmm_blasint ldi, libxsmm_blasint ldo ) {
  libxsmm_bfloat16 *in;
  float *out, *out_gold;
  unsigned char *mask;
  unsigned char *mask_gold;
  unsigned int i, j;
  float p = 0.3f;
  int ret = EXIT_SUCCESS;
  libxsmm_meltw_unary_param unary_param;
  libxsmm_meltw_unary_flags unary_flags;
  libxsmm_matdiff_info norms_out;
  union libxsmm_bfloat16_hp bf16_hp;
  libxsmm_blasint mask_ld = ((ldi+15)-((ldi+15)%16))/8;

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
  mask      = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);
  mask_gold = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*mask_ld, 64);

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
    mask[i] = 0xaa;
  }
  for ( i = 0; i < N*mask_ld; ++i ) {
    mask_gold[i] = 0xaa;
  }

  /* compute out_gold */
  for ( i = 0; i < N; ++i ) {
    dropout_bwd_bf16_f32_gold( M, &in[(i*ldi)], &out_gold[(i*ldo)], (unsigned short*)&mask_gold[(i*mask_ld)], p );
  }

  /* use jited tranpose */
  unary_param.op.primary = (void*)&p;
  unary_param.in.primary  = (void*)in;
  unary_param.in.secondary = (void*)mask;
  unary_param.out.primary = (void*)out;
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  libxsmm_meltwfunction_unary unary_kernel = libxsmm_dispatch_meltw_unary(M, N, &ldi, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV);
  if ( unary_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
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

  if ( norms_out.normf_rel > 0.005 ) {
    ret = EXIT_FAILURE;
  }

  libxsmm_free( out_gold );
  libxsmm_free( out );
  libxsmm_free( in );
  libxsmm_free( mask );
  libxsmm_free( mask_gold );

  if ( ret == EXIT_SUCCESS ) {
    printf("SUCCESS unary dropout bwd bf16 fp32\n");
  } else {
    printf("FAILURE unary dropout bwd bf16 fp32\n");
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
    ret = test_dropout_f32_f32_fwd( bitm, M, N, ldi, ldo );
  } else if ( op == 'F' && dtype_in == 2  && dtype_out == 2 ) {
    printf("Testing BF16 BF16 forward dropout - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_dropout_bf16_bf16_fwd( bitm, M, N, ldi, ldo );
  } else if ( op == 'F' && dtype_in == 4  && dtype_out == 2 ) {
    printf("Testing F32 BF16 forward dropout - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_dropout_f32_bf16_fwd( bitm, M, N, ldi, ldo );
  } else if ( op == 'F' && dtype_in == 2  && dtype_out == 4 ) {
    printf("Testing BF16 F32 forward dropout - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_dropout_bf16_f32_fwd( bitm, M, N, ldi, ldo );
  } else if ( op == 'B' && dtype_in == 4 && dtype_out == 4 ) {
    printf("Testing F32 F32 backward dropout - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_dropout_f32_f32_bwd( M, N, ldi, ldo );
  } else if ( op == 'B' && dtype_in == 2 && dtype_out == 2 ) {
    printf("Testing BF16 BF16 backward dropout - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_dropout_bf16_bf16_bwd( M, N, ldi, ldo );
  } else if ( op == 'B' && dtype_in == 4 && dtype_out == 2 ) {
    printf("Testing F32 BF16 backward dropout - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_dropout_f32_bf16_bwd( M, N, ldi, ldo );
  } else if ( op == 'B' && dtype_in == 2 && dtype_out == 4 ) {
    printf("Testing BF16 F32 backward dropout - M=%i, N=%i, LDI=%i, LDO=%i\n", M, N, ldi, ldo);
    ret = test_dropout_bf16_f32_bwd( M, N, ldi, ldo );
  } else {
    printf(" Not implemented case! Usage: %s [F/B] [bitmask: 0/1] [prec_in: 4/2] [prec_out: 4/2] [M] [N] [ldi] [ldo]\n", argv[0] );
    exit(-1);
  }

  return ret;
}
