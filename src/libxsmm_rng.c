/******************************************************************************
** Copyright (c) 2019, Intel Corporation                                     **
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
/* Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/

#include "libxsmm_rng.h"
#include <libxsmm_intrinsics_x86.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


/* 128 bit state for scalar RNG */
LIBXSMM_APIVAR(uint32_t libxsmm_rng_scalar_state[4]);


LIBXSMM_API_INLINE void libxsmm_rng_float_jump(uint32_t* state0, uint32_t* state1, uint32_t* state2, uint32_t* state3)
{
  static const uint32_t jump_table[] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };

  uint32_t s0 = 0;
  uint32_t s1 = 0;
  uint32_t s2 = 0;
  uint32_t s3 = 0;
  uint32_t t;
  size_t i, b;

  for (i = 0; i < sizeof(jump_table) / sizeof(*jump_table); ++i) {
    for (b = 0; b < 32; ++b) {
      if (jump_table[i] & UINT32_C(1) << b) {
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
      *state3 = ((*state3 << 11) | (*state3 >> (32 - 11)));
    }
  }
  *state0 = s0;
  *state1 = s1;
  *state2 = s2;
  *state3 = s3;
}


LIBXSMM_API_INLINE float libxsmm_rng_scalar_float_next(void)
{
  union {
    uint32_t i;
    float f;
  } rng;
  const uint32_t rng_mantissa = (libxsmm_rng_scalar_state[0] + libxsmm_rng_scalar_state[3]) >> 9;
  const uint32_t t = libxsmm_rng_scalar_state[1] << 9;

  libxsmm_rng_scalar_state[2] ^= libxsmm_rng_scalar_state[0];
  libxsmm_rng_scalar_state[3] ^= libxsmm_rng_scalar_state[1];
  libxsmm_rng_scalar_state[1] ^= libxsmm_rng_scalar_state[2];
  libxsmm_rng_scalar_state[0] ^= libxsmm_rng_scalar_state[3];
  libxsmm_rng_scalar_state[2] ^= t;
  libxsmm_rng_scalar_state[3] = ((libxsmm_rng_scalar_state[3] << 11) | (libxsmm_rng_scalar_state[3] >> (32 - 11)));

  rng.i = 0x3f800000 | rng_mantissa;
  return (rng.f - 1.0f);
}


#if (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH) /* __AVX512F__ */
/* 2048 bit state for AVX512 RNG */
LIBXSMM_APIVAR(__m512i libxsmm_rng_avx512_state_0);
LIBXSMM_APIVAR(__m512i libxsmm_rng_avx512_state_1);
LIBXSMM_APIVAR(__m512i libxsmm_rng_avx512_state_2);
LIBXSMM_APIVAR(__m512i libxsmm_rng_avx512_state_3);


LIBXSMM_API void libxsmm_rng_set_seed_avx512(unsigned int/*uint32_t*/ seed)
{
  uint32_t temp_state[] = {
     31,  30,  29,  28,  27,  26,  25,  24,  23,  22,  21,  20,  19,  18,  17,  16,
    131, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116,
    231, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 218, 217, 216,
    331, 330, 329, 328, 327, 326, 325, 324, 323, 322, 321, 320, 319, 318, 317, 316
  };
  libxsmm_blasint i;

  /* finish initializing the state */
  for (i = 0; i < sizeof(temp_state) / sizeof(*temp_state); ++i) temp_state[i] += seed;

  /* progress each sequence by 2^64 */
  for (i = 0; i < 16; ++i) {
    libxsmm_rng_float_jump(temp_state+i, temp_state+16+i, temp_state+32+i, temp_state+48+i);
  }

  /* load state */
  libxsmm_rng_avx512_state_0 = _mm512_loadu_si512(temp_state);
  libxsmm_rng_avx512_state_1 = _mm512_loadu_si512(temp_state+16);
  libxsmm_rng_avx512_state_2 = _mm512_loadu_si512(temp_state+32);
  libxsmm_rng_avx512_state_3 = _mm512_loadu_si512(temp_state+48);
}


LIBXSMM_API_INLINE __m512 _mm512_libxmm_rng_ps(void) {
  const __m512i rng_mantissa = _mm512_srli_epi32(_mm512_add_epi32(libxsmm_rng_avx512_state_0, libxsmm_rng_avx512_state_3), 9);
  const __m512i t = _mm512_slli_epi32(libxsmm_rng_avx512_state_1, 9);

  libxsmm_rng_avx512_state_2 = _mm512_xor_epi32(libxsmm_rng_avx512_state_2, libxsmm_rng_avx512_state_0);
  libxsmm_rng_avx512_state_3 = _mm512_xor_epi32(libxsmm_rng_avx512_state_3, libxsmm_rng_avx512_state_1);
  libxsmm_rng_avx512_state_1 = _mm512_xor_epi32(libxsmm_rng_avx512_state_1, libxsmm_rng_avx512_state_2);
  libxsmm_rng_avx512_state_0 = _mm512_xor_epi32(libxsmm_rng_avx512_state_0, libxsmm_rng_avx512_state_3);
  libxsmm_rng_avx512_state_2 = _mm512_xor_epi32(libxsmm_rng_avx512_state_2, t);
  libxsmm_rng_avx512_state_3 = _mm512_or_epi32(_mm512_slli_epi32(libxsmm_rng_avx512_state_3, 11),
                                               _mm512_srli_epi32(libxsmm_rng_avx512_state_3, 32-11));

  return _mm512_sub_ps(_mm512_castsi512_ps(_mm512_or_epi32(_mm512_set1_epi32(0x3f800000), rng_mantissa)),
                       _mm512_set1_ps(1.0f));
}
#endif


LIBXSMM_API void libxsmm_rng_set_seed(unsigned int/*uint32_t*/ seed)
{
  libxsmm_rng_scalar_state[0] = seed;
  libxsmm_rng_scalar_state[1] = seed + 100;
  libxsmm_rng_scalar_state[2] = seed + 200;
  libxsmm_rng_scalar_state[3] = seed + 300;
  libxsmm_rng_float_jump(libxsmm_rng_scalar_state, libxsmm_rng_scalar_state+1, libxsmm_rng_scalar_state+2, libxsmm_rng_scalar_state+3);
#if (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH) /* __AVX512F__ */
  libxsmm_rng_set_seed_avx512(seed);
#endif
#if !defined(_WIN32) && !defined(__CYGWIN__) && (defined(_SVID_SOURCE) || defined(_XOPEN_SOURCE))
  srand48(seed);
#endif
  srand(seed);
}


LIBXSMM_API void libxsmm_rng_f32_seq(float* rngs, libxsmm_blasint count)
{
  libxsmm_blasint i = 0;
#if (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH) /* __AVX512F__ */
  const libxsmm_blasint n = (count / 16) * 16;
  for (; i < n; i += 16) {
    _mm512_storeu_ps(rngs+i, _mm512_libxmm_rng_ps());
  }
#endif
  for (; i < count; ++i) {
    rngs[i] = libxsmm_rng_scalar_float_next();
  }
}


LIBXSMM_API unsigned int libxsmm_rng_u32(unsigned int n)
{
#if defined(_WIN32) || defined(__CYGWIN__) || !(defined(_SVID_SOURCE) || defined(_XOPEN_SOURCE))
  const unsigned int rand_max1 = (unsigned int)(RAND_MAX)+1U;
  const unsigned int q = (rand_max1 / n) * n;
  unsigned int r = (unsigned int)rand();
  if (q != rand_max1)
#else
  const unsigned int q = ((1U << 31) / n) * n;
  unsigned int r = (unsigned int)lrand48();
  if (q != (1U << 31))
#endif
  {
#if defined(_WIN32) || defined(__CYGWIN__) || !(defined(_SVID_SOURCE) || defined(_XOPEN_SOURCE))
    while (q <= r) r = (unsigned int)rand();
#else
    while (q <= r) r = (unsigned int)lrand48();
#endif
  }
  return r % n;
}


LIBXSMM_API double libxsmm_rng_f64(void)
{
#if defined(_WIN32) || defined(__CYGWIN__) || !(defined(_SVID_SOURCE) || defined(_XOPEN_SOURCE))
  static const double scale = 1.0 / (RAND_MAX);
  return scale * (double)rand();
#else
  return drand48();
#endif
}

