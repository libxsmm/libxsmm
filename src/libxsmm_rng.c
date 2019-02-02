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

/* 128 bit state for scalar rng */
static uint32_t libxsmm_rng_scalar_state[4];

LIBXSMM_API_INTERN inline uint32_t libxsmm_rng_float_next_uint32() {
  const uint32_t new = libxsmm_rng_scalar_state[0] + libxsmm_rng_scalar_state[3];
  const uint32_t t = libxsmm_rng_scalar_state[1] << 9;

  libxsmm_rng_scalar_state[2] ^= libxsmm_rng_scalar_state[0];
  libxsmm_rng_scalar_state[3] ^= libxsmm_rng_scalar_state[1];
  libxsmm_rng_scalar_state[1] ^= libxsmm_rng_scalar_state[2];
  libxsmm_rng_scalar_state[0] ^= libxsmm_rng_scalar_state[3];
  libxsmm_rng_scalar_state[2] ^= t;
  libxsmm_rng_scalar_state[3] = ( (libxsmm_rng_scalar_state[3] << 11) | (libxsmm_rng_scalar_state[3] >> (32 - 11)) );

  return new;
}


LIBXSMM_API void libxsmm_rng_float_set_seed( const uint32_t seed ) {
  libxsmm_rng_scalar_state[0] = seed;
  libxsmm_rng_scalar_state[1] = seed+10;
  libxsmm_rng_scalar_state[2] = seed+20;
  libxsmm_rng_scalar_state[3] = seed+30;
}


LIBXSMM_API float libxsmm_rng_float_next() {
  union Rng_local {
    uint32_t i;
    float    f;
  } rng;
  uint32_t rng_mantissa = libxsmm_rng_float_next_uint32();

  rng_mantissa = rng_mantissa >> 9;
  rng.f = 1.0f;
  rng.i = rng.i | rng_mantissa;

  return (rng.f - 1.0f);
}


LIBXSMM_API void  libxsmm_rng_float_jump() {
  static const uint32_t JUMP[] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };

  uint32_t s0 = 0;
  uint32_t s1 = 0;
  uint32_t s2 = 0;
  uint32_t s3 = 0;
  int i,b;

  for(i = 0; i < sizeof(JUMP) / sizeof(*JUMP); i++) {
    for(b = 0; b < 32; b++) {
      if (JUMP[i] & UINT32_C(1) << b) {
        s0 ^= libxsmm_rng_scalar_state[0];
        s1 ^= libxsmm_rng_scalar_state[1];
        s2 ^= libxsmm_rng_scalar_state[2];
        s3 ^= libxsmm_rng_scalar_state[3];
      }
      /* draw one more integer */
      libxsmm_rng_float_next_uint32();
    }
  }
  libxsmm_rng_scalar_state[0] = s0;
  libxsmm_rng_scalar_state[1] = s1;
  libxsmm_rng_scalar_state[2] = s2;
  libxsmm_rng_scalar_state[3] = s3;
}
