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
}


LIBXSMM_API float libxsmm_sexp2_i8i(int x)
{
  LIBXSMM_ASSERT(-128 <= x && x <= 127);
  return libxsmm_sexp2_i8((signed char)x);
}


LIBXSMM_API void libxsmm_gemm_print(void* ostream,
  libxsmm_datatype precision, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
  const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc)
{
  libxsmm_gemm_print2(ostream, precision, precision, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_API void libxsmm_gemm_print2(void* ostream,
  libxsmm_datatype iprec, libxsmm_datatype oprec, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
  const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc)
{
  const libxsmm_blasint nn = *(n ? n : m), kk = *(k ? k : m);
  const int itransa = LIBXSMM_FLAGS & LIBXSMM_GEMM_FLAG_TRANS_A, itransb = LIBXSMM_FLAGS & LIBXSMM_GEMM_FLAG_TRANS_B;
  const char ctransa = (char)(NULL != transa ? (*transa) : (0 == itransa ? 'n' : 't'));
  const char ctransb = (char)(NULL != transb ? (*transb) : (0 == itransb ? 'n' : 't'));
  const libxsmm_blasint ilda = (NULL != lda ? *lda : (('n' == ctransa || 'N' == ctransa) ? *m : kk));
  const libxsmm_blasint ildb = (NULL != ldb ? *ldb : (('n' == ctransb || 'N' == ctransb) ? kk : nn));
  const libxsmm_blasint ildc = *(NULL != ldc ? ldc : m);
  libxsmm_mhd_elemtype mhd_elemtype = LIBXSMM_MHD_ELEMTYPE_UNKNOWN;
  char string_a[128] = "", string_b[128] = "", typeprefix = 0;

  switch (iprec | oprec) {
    case LIBXSMM_DATATYPE_F64: {
      LIBXSMM_ASSERT(iprec == oprec);
      LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "%g", NULL != alpha ? *((const double*)alpha) : LIBXSMM_ALPHA);
      LIBXSMM_SNPRINTF(string_b, sizeof(string_b), "%g", NULL != beta  ? *((const double*)beta)  : LIBXSMM_BETA);
      mhd_elemtype = LIBXSMM_MHD_ELEMTYPE_F64;
      typeprefix = 'd';
    } break;
    case LIBXSMM_DATATYPE_F32: {
      LIBXSMM_ASSERT(iprec == oprec);
      LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "%g", NULL != alpha ? *((const float*)alpha) : LIBXSMM_ALPHA);
      LIBXSMM_SNPRINTF(string_b, sizeof(string_b), "%g", NULL != beta  ? *((const float*)beta)  : LIBXSMM_BETA);
      mhd_elemtype = LIBXSMM_MHD_ELEMTYPE_F32;
      typeprefix = 's';
    } break;
    default: if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
      static int error_once = 0;
      if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) { /* TODO: support I16, etc. */
        fprintf(stderr, "LIBXSMM ERROR: unsupported data-type requested!\n");
      }
    }
  }

  if (0 != typeprefix) {
    if (NULL != ostream) { /* print information about GEMM call */
      if (NULL != a && NULL != b && NULL != c) {
        fprintf((FILE*)ostream, "%cgemm('%c', '%c', %" PRIuPTR "/*m*/, %" PRIuPTR "/*n*/, %" PRIuPTR "/*k*/,\n"
                                "  %s/*alpha*/, %p/*a*/, %" PRIuPTR "/*lda*/,\n"
                                "              %p/*b*/, %" PRIuPTR "/*ldb*/,\n"
                                "   %s/*beta*/, %p/*c*/, %" PRIuPTR "/*ldc*/)",
          typeprefix, ctransa, ctransb, (uintptr_t)*m, (uintptr_t)nn, (uintptr_t)kk,
          string_a, a, (uintptr_t)ilda, b, (uintptr_t)ildb, string_b, c, (uintptr_t)ildc);
      }
      else {
        fprintf((FILE*)ostream, "%cgemm(trans=%c%c mnk=%" PRIuPTR ",%" PRIuPTR ",%" PRIuPTR
                                                 " ldx=%" PRIuPTR ",%" PRIuPTR ",%" PRIuPTR " a,b=%s,%s)",
          typeprefix, ctransa, ctransb, (uintptr_t)*m, (uintptr_t)nn, (uintptr_t)kk,
          (uintptr_t)ilda, (uintptr_t)ildb, (uintptr_t)ildc, string_a, string_b);
      }
    }
    else { /* dump A, B, and C matrices into MHD files */
      char extension_header[256] = "";
      size_t data_size[2] = { 0 }, size[2] = { 0 };

      if (NULL != a) {
        LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "TRANS = %c\nALPHA = %s", ctransa, string_a);
        LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "libxsmm_a_%p.mhd", a);
        data_size[0] = (size_t)ilda; data_size[1] = (size_t)kk; size[0] = (size_t)(*m); size[1] = (size_t)kk;
        libxsmm_mhd_write(string_a, NULL/*offset*/, size, data_size, 2/*ndims*/, 1/*ncomponents*/, mhd_elemtype,
          NULL/*conversion*/, a, NULL/*header_size*/, extension_header, NULL/*extension*/, 0/*extension_size*/);
      }
      if (NULL != b) {
        LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "\nTRANS = %c", ctransb);
        LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "libxsmm_b_%p.mhd", b);
        data_size[0] = (size_t)ildb; data_size[1] = (size_t)nn; size[0] = (size_t)kk; size[1] = (size_t)nn;
        libxsmm_mhd_write(string_a, NULL/*offset*/, size, data_size, 2/*ndims*/, 1/*ncomponents*/, mhd_elemtype,
          NULL/*conversion*/, b, NULL/*header_size*/, extension_header, NULL/*extension*/, 0/*extension_size*/);
      }
      if (NULL != c) {
        LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "BETA = %s", string_b);
        LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "libxsmm_c_%p.mhd", c);
        data_size[0] = (size_t)ildc; data_size[1] = (size_t)nn; size[0] = (size_t)(*m); size[1] = (size_t)nn;
        libxsmm_mhd_write(string_a, NULL/*offset*/, size, data_size, 2/*ndims*/, 1/*ncomponents*/, mhd_elemtype,
          NULL/*conversion*/, c, NULL/*header_size*/, extension_header, NULL/*extension*/, 0/*extension_size*/);
      }
    }
  }
}
