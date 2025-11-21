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
#include <libxsmm_utils.h>
#include <libxsmm.h>

/** Denote quality of scalar random number generator. */
#if !defined(LIBXSMM_RNG_DRAND48) && !defined(_WIN32) && !defined(__CYGWIN__) && \
    (defined(_SVID_SOURCE) || defined(_XOPEN_SOURCE))
# define LIBXSMM_RNG_DRAND48
#endif

LIBXSMM_API unsigned int libxsmm_rng_u32(unsigned int n)
{
  unsigned int result;
  if (1 < n) {
#if defined(LIBXSMM_RNG_DRAND48)
    const unsigned int rmax = (1U << 31);
    unsigned int r = (unsigned int)lrand48();
#else
    const unsigned int rmax = (unsigned int)(RAND_MAX + 1U);
    unsigned int r = (unsigned int)rand();
#endif
    const unsigned int nmax = LIBXSMM_MIN(n, rmax);
    const unsigned int q = (rmax / nmax) * nmax;
#if defined(LIBXSMM_RNG_DRAND48)
    /* coverity[dont_call] */
    while (q <= r) r = (unsigned int)lrand48();
#else
    while (q <= r) r = (unsigned int)rand();
#endif
    if (n <= nmax) result = r % nmax;
    else { /* input range exhausts RNG-state (precision) */
      const double s = ((double)n / nmax) * r + 0.5;
      result = (unsigned int)s;
    }
  }
  else result = 0;
  return result;
}


LIBXSMM_API void libxsmm_rng_seq(void* data, size_t nbytes)
{
  unsigned char* dst = (unsigned char*)data;
  unsigned char* end = dst + (nbytes & 0xFFFFFFFFFFFFFFFC);
  unsigned int r;
  for (; dst < end; dst += 4) {
#if defined(LIBXSMM_RNG_DRAND48)
    /* coverity[dont_call] */
    r = (unsigned int)lrand48();
#else
    r = (unsigned int)rand();
#endif
    LIBXSMM_MEMCPY127(dst, &r, 4);
  }
  end = (unsigned char*)data + nbytes;
  if (dst < end) {
#if defined(LIBXSMM_RNG_DRAND48)
    r = (unsigned int)lrand48();
#else
    r = (unsigned int)rand();
#endif
    LIBXSMM_MEMCPY127(dst, &r, end - dst);
  }
}


LIBXSMM_API double libxsmm_rng_f64(void)
{
#if defined(LIBXSMM_RNG_DRAND48)
  /* coverity[dont_call] */
  return drand48();
#else
  static const double scale = 1.0 / (RAND_MAX);
  return scale * (double)rand();
#endif
}


LIBXSMM_API unsigned int libxsmm_icbrt_u64(unsigned long long x)
{
  unsigned long long b; unsigned int y = 0; int s;
  for (s = 63; 0 <= s; s -= 3) {
    y += y; b = ((unsigned long long)y + 1) * 3 * y + 1ULL;
    if (b <= (x >> s)) { x -= b << s; ++y; }
  }
  return y;
}


LIBXSMM_API unsigned int libxsmm_icbrt_u32(unsigned int x)
{
  unsigned int b; unsigned int y = 0; int s;
  for (s = 30; 0 <= s; s -= 3) {
    y += y; b = 3 * y * (y + 1) + 1;
    if (b <= (x >> s)) { x -= b << s; ++y; }
  }
  return y;
}


#if defined(LIBXSMM_NO_LIBM)
/* Implementation based on Claude Baumann's product (http://www.convict.lu/Jeunes/ultimate_stuff/exp_ln_2.htm).
 * Exponential function, which exposes the number of iterations taken in the main case (1...22).
 */
LIBXSMM_API_INLINE float internal_math_sexp2(float x, int maxiter)
{
  static const float lut[] = { /* tabulated powf(2.f, powf(2.f, -index)) */
    2.00000000f, 1.41421354f, 1.18920708f, 1.09050775f, 1.04427373f, 1.02189720f, 1.01088929f, 1.00542986f,
    1.00271130f, 1.00135469f, 1.00067711f, 1.00033855f, 1.00016928f, 1.00008464f, 1.00004232f, 1.00002110f,
    1.00001061f, 1.00000525f, 1.00000262f, 1.00000131f, 1.00000072f, 1.00000036f, 1.00000012f
  };
  const int lut_size = sizeof(lut) / sizeof(*lut), lut_size1 = lut_size - 1;
  int sign, temp, unbiased, exponent, mantissa;
  union { int i; float s; } result;

  result.s = x;
  sign = (0 == (result.i & 0x80000000) ? 0 : 1);
  temp = result.i & 0x7FFFFFFF; /* clear sign */
  unbiased = (temp >> 23) - 127; /* exponent */
  exponent = -unbiased;
  mantissa = (temp << 8) | 0x80000000;

  if (lut_size1 >= exponent) {
    if (lut_size1 != exponent) { /* multiple lookups needed */
      if (7 >= unbiased) { /* not a degenerated case */
        const int n = (0 >= maxiter || lut_size1 <= maxiter) ? lut_size1 : maxiter;
        int i = 1;
        if (0 > unbiased) { /* regular/main case */
          LIBXSMM_ASSERT(0 <= exponent && exponent < lut_size);
          result.s = lut[exponent]; /* initial value */
          i = exponent + 1; /* next LUT offset */
        }
        else {
          result.s = 2.f; /* lut[0] */
          i = 1; /* next LUT offset */
        }
        for (; i <= n && 0 != mantissa; ++i) {
          mantissa <<= 1;
          if (0 != (mantissa & 0x80000000)) { /* check MSB */
            LIBXSMM_ASSERT(0 <= i && i < lut_size);
            result.s *= lut[i]; /* TODO: normalized multiply */
          }
        }
        for (i = 0; i < unbiased; ++i) { /* compute squares */
          result.s *= result.s;
        }
        if (0 != sign) { /* negative value, so reciprocal */
          result.s = 1.f / result.s;
        }
      }
      else { /* out of range */
#if defined(INFINITY) && /*overflow warning*/!defined(_CRAYC)
        result.s = (0 == sign ? ((float)(INFINITY)) : 0.f);
#else
        result.i = (0 == sign ? 0x7F800000 : 0);
#endif
      }
    }
    else if (0 == sign) {
      result.s = lut[lut_size1];
    }
    else { /* reciprocal */
      result.s = 1.f / lut[lut_size1];
    }
  }
  else {
    result.s = 1.f; /* case 2^0 */
  }
  return result.s;
}
#endif


LIBXSMM_API float libxsmm_sexp2(float x)
{
#if !defined(LIBXSMM_NO_LIBM)
  return LIBXSMM_EXP2F(x);
#else /* fallback */
  return internal_math_sexp2(x, 20/*compromise*/);
#endif
}


LIBXSMM_API float libxsmm_sexp2_u8(unsigned char x)
{
  union { int i; float s; } result = { 0 };
  if (128 > x) {
    if (31 < x) {
      static const float r32 = 2.f * ((float)(1U << 31)); /* 2^32 */
      const float r33 = r32 * r32, r34 = (float)(1U << LIBXSMM_MOD2(x, 32));
      result.s = r32 * r34;
      if (95 < x) result.s *= r33;
      else if (63 < x) result.s *= r32;
    }
    else {
      result.s = (float)(1U << x);
    }
  }
  else {
#if defined(INFINITY) && /*overflow warning*/!defined(_CRAYC)
    result.s = (float)(INFINITY);
#else
    result.i = 0x7F800000;
#endif
  }
  return result.s;
}


LIBXSMM_API float libxsmm_sexp2_i8(signed char x)
{
  return libxsmm_sexp2((float)x);
#if 0
  union { int i; float s; } result = { 0 };
  if (-128 != x) {
    const signed char ux = (signed char)LIBXSMM_ABS(x);
    if (31 < ux) {
      static const float r32 = 2.f * ((float)(1U << 31)); /* 2^32 */
      signed char n = ux >> 5, r = ux - (signed char)(n << 5), i;
      result.s = r32;
      for (i = 1; i < n; ++i) result.s *= r32;
      result.s *= (float)(1U << r);
    }
    else {
      result.s = (float)(1U << ux);
    }
    if (ux != x) { /* signed */
      result.s = 1.f / result.s;
    }
  }
  else {
    result.i = 0x200000;
  }
  return result.s;
#endif
}


LIBXSMM_API float libxsmm_sexp2_i8i(int x)
{
  LIBXSMM_ASSERT(-128 <= x && x <= 127);
  return libxsmm_sexp2_i8((signed char)x);
}

