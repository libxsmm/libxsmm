/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>

#if !defined(LIBXSMM_RNG_AVX512) && 1
# define LIBXSMM_RNG_AVX512
#endif
#if !defined(LIBXSMM_RNG_SIMD_MIN)
# define LIBXSMM_RNG_SIMD_MIN 8
#endif

/* dispatched RNG functions (separate typedef for legacy Cray C++ needed) */
typedef void (*internal_rng_f32_seq_fn)(float*, libxsmm_blasint);
LIBXSMM_APIVAR_DEFINE(internal_rng_f32_seq_fn internal_rng_f32_seq);
/* 2048-bit state for RNG */
LIBXSMM_APIVAR_DEFINE(unsigned int internal_rng_state0[16]);
LIBXSMM_APIVAR_DEFINE(unsigned int internal_rng_state1[16]);
LIBXSMM_APIVAR_DEFINE(unsigned int internal_rng_state2[16]);
LIBXSMM_APIVAR_DEFINE(unsigned int internal_rng_state3[16]);


LIBXSMM_API_INLINE void internal_rng_float_jump(uint32_t* state0, uint32_t* state1, uint32_t* state2, uint32_t* state3)
{
  static const uint32_t jump_table[] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };
  uint32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  int i, b;

  LIBXSMM_ASSERT(4 == sizeof(jump_table) / sizeof(*jump_table));
  for (i = 0; i < 4; ++i) {
    for (b = 0; b < 32; ++b) {
      if (jump_table[i] & (1U << b)) {
        s0 ^= *state0;
        s1 ^= *state1;
        s2 ^= *state2;
        s3 ^= *state3;
      }
      { /* draw one more integer */
        const uint32_t t = *state1 << 9;
        *state2 ^= *state0;
        *state3 ^= *state1;
        *state1 ^= *state2;
        *state0 ^= *state3;
        *state2 ^= t;
        *state3 = ((*state3 << 11) | (*state3 >> (32 - 11)));
      }
    }
  }
  *state0 = s0;
  *state1 = s1;
  *state2 = s2;
  *state3 = s3;
}


LIBXSMM_API_INLINE float internal_rng_scalar_float_next(int i)
{
  const uint32_t rng_mantissa = (internal_rng_state0[i] + internal_rng_state3[i]) >> 9;
  const uint32_t t = internal_rng_state1[i] << 9;
  union { uint32_t i; float f; } rng = { 0 };

  internal_rng_state2[i] ^= internal_rng_state0[i];
  internal_rng_state3[i] ^= internal_rng_state1[i];
  internal_rng_state1[i] ^= internal_rng_state2[i];
  internal_rng_state0[i] ^= internal_rng_state3[i];
  internal_rng_state2[i] ^= t;
  internal_rng_state3[i] = ((internal_rng_state3[i] << 11) | (internal_rng_state3[i] >> (32 - 11)));

  rng.i = 0x3f800000 | rng_mantissa;
  return rng.f - 1.0f;
}


LIBXSMM_API_INTERN void internal_rng_set_seed_sw(uint32_t seed);
LIBXSMM_API_INTERN void internal_rng_set_seed_sw(uint32_t seed)
{
  static const uint32_t temp_state[] = {
     31,  30,  29,  28,  27,  26,  25,  24,  23,  22,  21,  20,  19,  18,  17,  16,
    131, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116,
    231, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 218, 217, 216,
    331, 330, 329, 328, 327, 326, 325, 324, 323, 322, 321, 320, 319, 318, 317, 316
  };
  libxsmm_blasint i;

  /* finish initializing the state */
  LIBXSMM_ASSERT((16 * 4) == sizeof(temp_state) / sizeof(*temp_state));
  for (i = 0; i < 16; ++i) {
    internal_rng_state0[i] = seed + temp_state[i];
    internal_rng_state1[i] = seed + temp_state[i+16];
    internal_rng_state2[i] = seed + temp_state[i+32];
    internal_rng_state3[i] = seed + temp_state[i+48];
  }
  for (i = 0; i < 16; ++i) {
    internal_rng_float_jump( /* progress each sequence by 2^64 */
      internal_rng_state0 + i, internal_rng_state1 + i,
      internal_rng_state2 + i, internal_rng_state3 + i);
  }
  /* for consistency, other RNGs are seeded as well */
#if !defined(_WIN32) && !defined(__CYGWIN__) && (defined(_SVID_SOURCE) || defined(_XOPEN_SOURCE))
  srand48(seed);
#endif
  srand(seed);
}


LIBXSMM_API_INLINE void internal_rng_f32_seq_sw(float* rngs, libxsmm_blasint count)
{
  libxsmm_blasint i = 0;
  for (; i < count; ++i) {
    rngs[i] = internal_rng_scalar_float_next(LIBXSMM_MOD2(i, 16));
  }
}


#if defined(LIBXSMM_INTRINSICS_AVX512_SKX) /* __AVX512F__ */
LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX)
void internal_rng_set_seed_avx512(uint32_t seed)
{
  internal_rng_set_seed_sw(seed);
  /* bring scalar state to AVX-512 */
  LIBXSMM_INTRINSICS_MM512_RNG_STATE(0) = _mm512_loadu_si512(internal_rng_state0);
  LIBXSMM_INTRINSICS_MM512_RNG_STATE(1) = _mm512_loadu_si512(internal_rng_state1);
  LIBXSMM_INTRINSICS_MM512_RNG_STATE(2) = _mm512_loadu_si512(internal_rng_state2);
  LIBXSMM_INTRINSICS_MM512_RNG_STATE(3) = _mm512_loadu_si512(internal_rng_state3);
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX)
void internal_rng_f32_seq_avx512(float* rngs, libxsmm_blasint count)
{
  if ((LIBXSMM_RNG_SIMD_MIN << 4) <= count) { /* SIMD code path */
    const libxsmm_blasint n = (count >> 4) << 4; /* multiple of vector-length */
    libxsmm_blasint i = 0;
    for (; i < n; i += 16) {
      _mm512_storeu_ps(rngs + i, LIBXSMM_INTRINSICS_MM512_RNG_PS());
    }
    if (i < count) { /* remainder */
#if 0 /* assert(0 < n) */
      if (0 < n)
#endif
      { /* bring AVX-512 state to scalar */
        _mm512_storeu_si512(internal_rng_state0, LIBXSMM_INTRINSICS_MM512_RNG_STATE(0));
        _mm512_storeu_si512(internal_rng_state1, LIBXSMM_INTRINSICS_MM512_RNG_STATE(1));
        _mm512_storeu_si512(internal_rng_state2, LIBXSMM_INTRINSICS_MM512_RNG_STATE(2));
        _mm512_storeu_si512(internal_rng_state3, LIBXSMM_INTRINSICS_MM512_RNG_STATE(3));
      }
      LIBXSMM_ASSERT(count < i + 16);
      do { /* scalar remainder */
        rngs[i] = internal_rng_scalar_float_next(LIBXSMM_MOD2(i, 16));
        ++i;
      } while (i < count);
      /* bring scalar state to AVX-512 */
      LIBXSMM_INTRINSICS_MM512_RNG_STATE(0) = _mm512_loadu_si512(internal_rng_state0);
      LIBXSMM_INTRINSICS_MM512_RNG_STATE(1) = _mm512_loadu_si512(internal_rng_state1);
      LIBXSMM_INTRINSICS_MM512_RNG_STATE(2) = _mm512_loadu_si512(internal_rng_state2);
      LIBXSMM_INTRINSICS_MM512_RNG_STATE(3) = _mm512_loadu_si512(internal_rng_state3);
    }
  }
  else { /* scalar code path */
    internal_rng_f32_seq_sw(rngs, count);
  }
}
#endif /*defined(LIBXSMM_INTRINSICS_AVX512_SKX)*/


LIBXSMM_API unsigned int* libxsmm_rng_create_extstate(unsigned int/*uint32_t*/ seed)
{
  unsigned int* state = (unsigned int*) libxsmm_aligned_malloc( 64*sizeof(unsigned int), 64 );
  static const uint32_t temp_state[] = {
     31,  30,  29,  28,  27,  26,  25,  24,  23,  22,  21,  20,  19,  18,  17,  16,
    131, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116,
    231, 230, 229, 228, 227, 226, 225, 224, 223, 222, 221, 220, 219, 218, 217, 216,
    331, 330, 329, 328, 327, 326, 325, 324, 323, 322, 321, 320, 319, 318, 317, 316
  };
  libxsmm_blasint i;

  /* finish initializing the state */
  LIBXSMM_ASSERT((16 * 4) == sizeof(temp_state) / sizeof(*temp_state));
  for (i = 0; i < 16; ++i) {
    state[i   ] = seed + temp_state[i];
    state[i+16] = seed + temp_state[i+16];
    state[i+32] = seed + temp_state[i+32];
    state[i+48] = seed + temp_state[i+48];
  }
  for (i = 0; i < 16; ++i) {
    internal_rng_float_jump( /* progress each sequence by 2^64 */
      state +      i, state + 16 + i,
      state + 32 + i, state + 48 + i);
  }

  return state;
}


LIBXSMM_API unsigned int libxsmm_rng_get_extstate_size(void)
{
  return (unsigned int)(sizeof(unsigned int)*64);
}


LIBXSMM_API void libxsmm_rng_destroy_extstate(unsigned int* stateptr)
{
  libxsmm_free(stateptr);
}


LIBXSMM_API void libxsmm_rng_set_seed(unsigned int/*uint32_t*/ seed)
{
  LIBXSMM_INIT
#if (LIBXSMM_X86_AVX512_SKX <= LIBXSMM_STATIC_TARGET_ARCH) && defined(LIBXSMM_RNG_AVX512)
# if !defined(NDEBUG) /* used to track if seed is initialized */
  internal_rng_f32_seq = internal_rng_f32_seq_avx512;
# endif
  internal_rng_set_seed_avx512(seed);
#elif defined(LIBXSMM_INTRINSICS_AVX512_SKX) && defined(LIBXSMM_RNG_AVX512) /* __AVX512F__ */
  if (LIBXSMM_X86_AVX512_SKX <= libxsmm_target_archid) {
    internal_rng_f32_seq = internal_rng_f32_seq_avx512;
    internal_rng_set_seed_avx512(seed);
  }
  else {
    internal_rng_f32_seq = internal_rng_f32_seq_sw;
    internal_rng_set_seed_sw(seed);
  }
#else
# if !defined(NDEBUG) /* used to track if seed is initialized */
  internal_rng_f32_seq = internal_rng_f32_seq_sw;
# endif
  internal_rng_set_seed_sw(seed);
#endif
}


LIBXSMM_API void libxsmm_rng_f32_seq(float* rngs, libxsmm_blasint count)
{
  LIBXSMM_ASSERT_MSG(NULL != internal_rng_f32_seq, "RNG must be initialized");
#if (LIBXSMM_X86_AVX512_SKX <= LIBXSMM_STATIC_TARGET_ARCH) && defined(LIBXSMM_RNG_AVX512)
  internal_rng_f32_seq_avx512(rngs, count);
#else
# if defined(LIBXSMM_INTRINSICS_AVX512_SKX) && defined(LIBXSMM_RNG_AVX512) /* __AVX512F__ */
  if ((LIBXSMM_RNG_SIMD_MIN << 4) <= count) { /* SIMD code path */
    internal_rng_f32_seq(rngs, count); /* pointer based function call */
  }
  else /* scalar code path */
# endif
  internal_rng_f32_seq_sw(rngs, count);
#endif
}
