/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_UTILS_MATH_H
#define LIBXSMM_UTILS_MATH_H

#include "libxsmm_typedefs.h"


/**
 * Returns a (pseudo-)random value based on rand/rand48 in the interval [0, n),
 * i.e., the generated range of values 0..n-1 excludes n.
 * This function compensates for an n, which is not a factor of RAND_MAX.
 * Note: libxsmm_rng_set_seed must be used if one wishes to seed the generator.
 */
LIBXSMM_API unsigned int libxsmm_rng_u32(unsigned int n);

/** Sequence of random data based on libxsmm_rng_u32. */
LIBXSMM_API void libxsmm_rng_seq(void* data, size_t nbytes);

/**
 * Similar to libxsmm_rng_u32, but returns a DP-value in the interval [0, 1).
 * Note: libxsmm_rng_set_seed must be used if one wishes to seed the generator.
 */
LIBXSMM_API double libxsmm_rng_f64(void);

/** CBRT with Newton's method using integer arithmetic. */
LIBXSMM_API unsigned int libxsmm_icbrt_u64(unsigned long long x);
/** CBRT with Newton's method using integer arithmetic. */
LIBXSMM_API unsigned int libxsmm_icbrt_u32(unsigned int x);

/** Single-precision approximation of exponential function (base 2). */
LIBXSMM_API float libxsmm_sexp2(float x);

/**
 * Exponential function (base 2), which is limited to unsigned 8-bit input values.
 * This function reproduces bit-accurate results (single-precision).
 */
LIBXSMM_API float libxsmm_sexp2_u8(unsigned char x);

/**
* Exponential function (base 2), which is limited to signed 8-bit input values.
* This function reproduces bit-accurate results (single-precision).
*/
LIBXSMM_API float libxsmm_sexp2_i8(signed char x);

/** Similar to libxsmm_sexp2_i8, but takes an integer as signed 8-bit value (check). */
LIBXSMM_API float libxsmm_sexp2_i8i(int x);

/** Inlineable fast tanh, such that a the compiler can potentially vectorize. */
LIBXSMM_API_INLINE float libxsmm_stanh_pade78(float i_x) {
  const float l_c0 = 2027025.0f;
  const float l_c1 = 270270.0f;
  const float l_c2 = 6930.0f;
  const float l_c3 = 36.0f;
  const float l_c1_d = 945945.0f;
  const float l_c2_d = 51975.0f;
  const float l_c3_d = 630.0f;
  const float l_hi_bound = 4.97f;
  const float l_lo_bound = -4.97f;
  const float l_ones = 1.0f;
  const float l_neg_ones = -1.0f;
  const float x2 = i_x * i_x;
  const float t1_nom = (l_c3 * x2) + l_c2;
  const float t2_nom = (t1_nom * x2) + l_c1;
  const float t3_nom = (t2_nom * x2) + l_c0;
  const float nom = t3_nom * i_x;
  const float t1_denom = x2 + l_c3_d;
  const float t2_denom = (t1_denom * x2) + l_c2_d;
  const float t3_denom = (t2_denom * x2) + l_c1_d;
  const float denom = (t3_denom * x2) + l_c0;
  float result = nom / denom;
  result = (result > l_hi_bound) ? l_ones : result;
  result = (result < l_lo_bound) ? l_neg_ones : result;
  return result;
}

#endif /*LIBXSMM_UTILS_MATH_H*/
