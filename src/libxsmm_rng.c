/******************************************************************************
** Copyright (c) 2016-2019, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include "libxsmm_rng.h"
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

/* 128 bit state for scalar rng */
static uint32_t libxsmm_rng_scalar_state[4];

static inline void  libxsmm_rng_float_jump( uint32_t* state0, uint32_t* state1, uint32_t* state2, uint32_t* state3 ) {
  static const uint32_t JUMP[] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };

  uint32_t s0 = 0;
  uint32_t s1 = 0;
  uint32_t s2 = 0;
  uint32_t s3 = 0;
  uint32_t t;
  size_t i, b;

  for(i = 0; i < sizeof(JUMP) / sizeof(*JUMP); i++) {
    for(b = 0; b < 32; b++) {
      if (JUMP[i] & UINT32_C(1) << b) {
        s0 ^= *state0;
        s1 ^= *state1;
        s2 ^= *state2;
        s3 ^= *state3;
      }
      /* draw one more integer */
      t = *state1 << 9;
      *state2 ^= *state0;
      *state3 ^= *state1;
      *state1 ^= *state2;
      *state0 ^= *state3;
      *state2 ^= t;
      *state3 = ( (*state3 << 11) | (*state3 >> (32 - 11)) );
    }
  }
  *state0 = s0;
  *state1 = s1;
  *state2 = s2;
  *state3 = s3;
}


static inline float libxsmm_rng_scalar_float_next() {
  union Rng_local {
    uint32_t i;
    float    f;
  } rng;
  const uint32_t rng_mantissa = ( libxsmm_rng_scalar_state[0] + libxsmm_rng_scalar_state[3] ) >> 9;
  const uint32_t t = libxsmm_rng_scalar_state[1] << 9;

  libxsmm_rng_scalar_state[2] ^= libxsmm_rng_scalar_state[0];
  libxsmm_rng_scalar_state[3] ^= libxsmm_rng_scalar_state[1];
  libxsmm_rng_scalar_state[1] ^= libxsmm_rng_scalar_state[2];
  libxsmm_rng_scalar_state[0] ^= libxsmm_rng_scalar_state[3];
  libxsmm_rng_scalar_state[2] ^= t;
  libxsmm_rng_scalar_state[3] = ( (libxsmm_rng_scalar_state[3] << 11) | (libxsmm_rng_scalar_state[3] >> (32 - 11)) );

  rng.i = 0x3f800000 | rng_mantissa;
  return (rng.f - 1.0f);
}


#if defined(__AVX512F__)
/* 2048 bit state for AVX512 rng */
static __m512i libxsmm_rng_avx512_state_0;
static __m512i libxsmm_rng_avx512_state_1;
static __m512i libxsmm_rng_avx512_state_2;
static __m512i libxsmm_rng_avx512_state_3;


LIBXSMM_API void libxsmm_rng_float_set_seed_avx512( const uint32_t seed ) {
  libxsmm_blasint i;
  uint32_t temp_state[] = { seed+ 31, seed+ 30, seed+ 29, seed+ 28, seed+ 27, seed+ 26, seed+ 25, seed+ 24,
                              seed+ 23, seed+ 22, seed+ 21, seed+ 20, seed+ 19, seed+ 18, seed+ 17, seed+ 16,
                              seed+131, seed+130, seed+129, seed+128, seed+127, seed+126, seed+125, seed+124,
                              seed+123, seed+122, seed+121, seed+120, seed+119, seed+118, seed+117, seed+116,
                              seed+231, seed+230, seed+229, seed+228, seed+227, seed+226, seed+225, seed+224,
                              seed+223, seed+222, seed+221, seed+220, seed+219, seed+218, seed+217, seed+216,
                              seed+331, seed+330, seed+329, seed+328, seed+327, seed+326, seed+325, seed+324,
                              seed+323, seed+322, seed+321, seed+320, seed+319, seed+318, seed+317, seed+316  };

  /* progress each sequence by 2^64 */
  for ( i = 0; i < 16; ++i ) {
    libxsmm_rng_float_jump( temp_state+i, temp_state+16+i, temp_state+32+i, temp_state+48+i );
  }

  /* load state */
  libxsmm_rng_avx512_state_0 = _mm512_loadu_epi32( temp_state    );
  libxsmm_rng_avx512_state_1 = _mm512_loadu_epi32( temp_state+16 );
  libxsmm_rng_avx512_state_2 = _mm512_loadu_epi32( temp_state+32 );
  libxsmm_rng_avx512_state_3 = _mm512_loadu_epi32( temp_state+48 );
}


static inline __m512 _mm512_libxmm_rng_ps() {
  __m512i rng_mantissa = _mm512_srli_epi32( _mm512_add_epi32( libxsmm_rng_avx512_state_0, libxsmm_rng_avx512_state_3 ), 9 );
  __m512i t = _mm512_slli_epi32( libxsmm_rng_avx512_state_1, 9) ;

  libxsmm_rng_avx512_state_2 = _mm512_xor_epi32( libxsmm_rng_avx512_state_2, libxsmm_rng_avx512_state_0 );
  libxsmm_rng_avx512_state_3 = _mm512_xor_epi32( libxsmm_rng_avx512_state_3, libxsmm_rng_avx512_state_1 );
  libxsmm_rng_avx512_state_1 = _mm512_xor_epi32( libxsmm_rng_avx512_state_1, libxsmm_rng_avx512_state_2 );
  libxsmm_rng_avx512_state_0 = _mm512_xor_epi32( libxsmm_rng_avx512_state_0, libxsmm_rng_avx512_state_3 );
  libxsmm_rng_avx512_state_2 = _mm512_xor_epi32( libxsmm_rng_avx512_state_2, t );
  libxsmm_rng_avx512_state_3 = _mm512_or_epi32( _mm512_slli_epi32( libxsmm_rng_avx512_state_3, 11 ),
                                                _mm512_srli_epi32( libxsmm_rng_avx512_state_3, 32-11 ) );
  
  return _mm512_sub_ps( _mm512_castsi512_ps( _mm512_or_epi32( _mm512_set1_epi32( 0x3f800000 ), rng_mantissa ) ),
                        _mm512_set1_ps( 1.0f ) );
}
#endif


LIBXSMM_API void libxsmm_rng_float_set_seed( const uint32_t seed ) {
  libxsmm_rng_scalar_state[0] = seed;
  libxsmm_rng_scalar_state[1] = seed+100;
  libxsmm_rng_scalar_state[2] = seed+200;
  libxsmm_rng_scalar_state[3] = seed+300;

  libxsmm_rng_float_jump( libxsmm_rng_scalar_state, libxsmm_rng_scalar_state+1, libxsmm_rng_scalar_state+2, libxsmm_rng_scalar_state+3 );
#if defined(__AVX512F__)
  libxsmm_rng_float_set_seed_avx512( seed );
#endif
}


LIBXSMM_API void libxsmm_rng_float_seq( float* rngs, const libxsmm_blasint count ) {
  libxsmm_blasint i = 0;

#if defined(__AVX512F__)
  for (    ; i < (count/16)*16; i+=16 ) {
    _mm512_storeu_ps( rngs+i, _mm512_libxmm_rng_ps() );
  }
#endif
  for (    ; i < count; ++i ) {
    rngs[i] = libxsmm_rng_scalar_float_next();
  }
}
