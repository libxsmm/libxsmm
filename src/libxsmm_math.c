/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
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
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm_math.h>
#include <libxsmm_mhd.h>
#include "libxsmm_hash.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if !defined(LIBXSMM_NO_LIBM)
# include <math.h>
#endif
#include <string.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXSMM_MATH_DISPATCH1) && defined(__INTEL_COMPILER)
# define LIBXSMM_MATH_DISPATCH1
#endif
#if !defined(LIBXSMM_MATH_MAXPRODUCT)
# define LIBXSMM_MATH_MAXPRODUCT 1024
#endif
#if !defined(LIBXSMM_MATH_MEMCMP) && 0
# define LIBXSMM_MATH_MEMCMP
#endif

#define LIBXSMM_MATH_DIFF(DIFF, MOD, A, BN, ELEMSIZE, STRIDE, HINT, N) { \
  const char *const libxsmm_diff_b_ = (const char*)(BN); \
  const unsigned int libxsmm_diff_end_ = (HINT) + (N); \
  unsigned int libxsmm_diff_i_; \
  LIBXSMM_PRAGMA_LOOP_COUNT(4, 1024, 4) \
  for (libxsmm_diff_i_ = HINT; libxsmm_diff_i_ != libxsmm_diff_end_; ++libxsmm_diff_i_) { \
    const unsigned int libxsmm_diff_j_ = MOD(libxsmm_diff_i_, N); /* wrap around index */ \
    const unsigned int libxsmm_diff_k_ = libxsmm_diff_j_ * (STRIDE); \
    if (0 == (DIFF)(A, libxsmm_diff_b_ + libxsmm_diff_k_, ELEMSIZE)) return libxsmm_diff_j_; \
  } \
  return N; \
}


LIBXSMM_API int libxsmm_matdiff(libxsmm_datatype datatype, libxsmm_blasint m, libxsmm_blasint n,
  const void* ref, const void* tst, const libxsmm_blasint* ldref, const libxsmm_blasint* ldtst,
  libxsmm_matdiff_info* info)
{
  int result = EXIT_SUCCESS, result_swap = 0;
  if (0 == ref && 0 != tst) { ref = tst; tst = NULL; result_swap = 1; }
  if (0 != ref && 0 != info) {
    libxsmm_blasint mm = m, nn = n, ldr = (0 == ldref ? m : *ldref), ldt = (0 == ldtst ? m : *ldtst);
    union { int i; float s; } inf;
#if defined(INFINITY)
    inf.s = INFINITY;
#else
    inf.i = 0x7F800000;
#endif
    if (1 == n) { mm = ldr = ldt = 1; nn = m; } /* ensure row-vector shape to standardize results */
    memset(info, 0, sizeof(*info)); /* nullify */
    switch (datatype) {
      case LIBXSMM_DATATYPE_F64: {
#       define LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE double
#       include "template/libxsmm_matdiff.tpl.c"
#       undef  LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE
      } break;
      case LIBXSMM_DATATYPE_F32: {
#       define LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE float
#       include "template/libxsmm_matdiff.tpl.c"
#       undef  LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE
      } break;
      case LIBXSMM_DATATYPE_I32: {
#       define LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE int
#       include "template/libxsmm_matdiff.tpl.c"
#       undef  LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE
      } break;
      case LIBXSMM_DATATYPE_I16: {
#       define LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE short
#       include "template/libxsmm_matdiff.tpl.c"
#       undef  LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE
      } break;
      case LIBXSMM_DATATYPE_I8: {
#       define LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE signed char
#       include "template/libxsmm_matdiff.tpl.c"
#       undef  LIBXSMM_MATDIFF_TEMPLATE_ELEM_TYPE
      } break;
      default: {
        static int error_once = 0;
        if (0 != libxsmm_verbosity /* library code is expected to be mute */
          && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
        {
          fprintf(stderr, "LIBXSMM ERROR: unsupported data-type requested!\n");
        }
        result = EXIT_FAILURE;
      }
    }
    if (EXIT_SUCCESS == result) {
      const char *const env = getenv("LIBXSMM_DUMP");
      if (0 != env && 0 != *env && (('0' < *env && '9' >= *env) || '0' != *env)) {
        const char *const defaultname = ('0' < *env && '9' >= *env) ? "libxsmm_dump" : env;
        const libxsmm_mhd_elemtype type_src = (libxsmm_mhd_elemtype)datatype;
        const libxsmm_mhd_elemtype type_dst = LIBXSMM_MAX(LIBXSMM_MHD_ELEMTYPE_U8, type_src);
        char filename[256];
        size_t size[2], pr[2]; size[0] = mm; size[1] = nn; pr[0] = ldr; pr[1] = nn;
        LIBXSMM_SNPRINTF(filename, sizeof(filename), "%s-ref%p.mhd", defaultname, ref);
        libxsmm_mhd_write(filename, NULL/*offset*/, size, pr, 2/*ndims*/, 1/*ncomponents*/,
          type_src, &type_dst, ref, NULL/*header_size*/, NULL/*extension_header*/,
          NULL/*extension*/, 0/*extension_size*/);
        if (NULL != tst) {
          size_t pt[2]; pt[0] = ldt; pt[1] = nn;
          LIBXSMM_SNPRINTF(filename, sizeof(filename), "%s-tst%p.mhd", defaultname, tst);
          libxsmm_mhd_write(filename, NULL/*offset*/, size, pt, 2/*ndims*/, 1/*ncomponents*/,
            type_src, &type_dst, tst, NULL/*header_size*/, NULL/*extension_header*/,
            NULL/*extension*/, 0/*extension_size*/);
        }
      }
      info->normf_rel = libxsmm_dsqrt(info->normf_rel);
      info->l2_abs = libxsmm_dsqrt(info->l2_abs);
      info->l2_rel = libxsmm_dsqrt(info->l2_rel);
      if (1 == n) {
        const libxsmm_blasint tmp = info->linf_abs_m;
        info->linf_abs_m = info->linf_abs_n;
        info->linf_abs_n = tmp;
      }
      if (0 != result_swap) {
        info->l1_tst = info->l1_ref;
        info->l1_ref = 0;
      }
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API void libxsmm_matdiff_reduce(libxsmm_matdiff_info* output, const libxsmm_matdiff_info* input)
{
  LIBXSMM_ASSERT(0 != output && 0 != input);
  if (output->normf_rel < input->normf_rel) {
    output->linf_abs_m = input->linf_abs_m;
    output->linf_abs_n = input->linf_abs_n;
    output->norm1_abs = input->norm1_abs;
    output->norm1_rel = input->norm1_rel;
    output->normi_abs = input->normi_abs;
    output->normi_rel = input->normi_rel;
    output->normf_rel = input->normf_rel;
    output->linf_abs = input->linf_abs;
    output->linf_rel = input->linf_rel;
    output->l2_abs = input->l2_abs;
    output->l2_rel = input->l2_rel;
    output->l1_ref = input->l1_ref;
    output->l1_tst = input->l1_tst;
  }
}


LIBXSMM_API_INTERN unsigned int libxsmm_diff_sw(const void* a, const void* b, unsigned char size);
LIBXSMM_API_INTERN unsigned int libxsmm_diff_sw(const void* a, const void* b, unsigned char size)
{
  unsigned int result;
  unsigned char i;
#if (LIBXSMM_X86_SSE3 <= LIBXSMM_STATIC_TARGET_ARCH)
  const __m128i *const a128 = (const __m128i*)a;
  const __m128i *const b128 = (const __m128i*)b;
  const unsigned char n = (unsigned char)(size >> 4);
  result = 0;
  for (i = 0; i < n /*&& 0 == result*/; ++i) {
    const __m128i ai = _mm_loadu_si128(a128 + i), bi = _mm_loadu_si128(b128 + i);
    result |= (0xFFFF != _mm_movemask_epi8(_mm_cmpeq_epi8(ai, bi)));
  }
  i <<= 4;
#else
  const unsigned long long *const a2 = (const unsigned long long*)a;
  const unsigned long long *const b2 = (const unsigned long long*)b;
  union { unsigned long long u; unsigned int v[2]; } result8 = { 0 };
  const unsigned char n = (unsigned char)(size >> 3);
  for (i = 0; i < n /*&& 0 == result8.u*/; ++i) result8.u |= a2[i] ^ b2[i];
  result = result8.v[0] | result8.v[1]; i <<= 3;
#endif
  for (; i < size /*&& 0 == result*/; ++i) {
    result |= *((const unsigned char*)a + i) ^ *((const unsigned char*)b + i);
  }
  return result;
}


#if !defined(LIBXSMM_MATH_MEMCMP)
LIBXSMM_API_INTERN unsigned int libxsmm_diff_avx2(const void* a, const void* b, unsigned char size);
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX2)
unsigned int libxsmm_diff_avx2(const void* a, const void* b, unsigned char size)
{
  unsigned int result;
#if defined(LIBXSMM_INTRINSICS_AVX2)
  const __m256i *const a256 = (const __m256i*)a;
  const __m256i *const b256 = (const __m256i*)b;
  const unsigned char n = (unsigned char)(size >> 5);
  unsigned char i;
  result = 0;
  for (i = 0; i < n /*&& 0 == result*/; ++i) {
    const __m256i ai = _mm256_loadu_si256(a256 + i), bi = _mm256_loadu_si256(b256 + i);
    result |= _mm256_movemask_epi8(_mm256_cmpeq_epi8(ai, bi)) + 1;
  }
  for (i <<= 5; i < size /*&& 0 == result*/; ++i) {
    result |= *((const unsigned char*)a + i) ^ *((const unsigned char*)b + i);
  }
#else
  result = libxsmm_diff_sw(a, b, size);
#endif
  return result;
}
#endif


LIBXSMM_API unsigned int libxsmm_diff(const void* a, const void* b, unsigned char size)
{
  unsigned int result;
#if defined(LIBXSMM_MATH_MEMCMP)
  const int diff = memcmp(a, b, size);
  result = LIBXSMM_ABS(diff);
#elif (LIBXSMM_X86_AVX2 <= LIBXSMM_STATIC_TARGET_ARCH)
  result = libxsmm_diff_avx2(a, b, size);
#else
# if defined(LIBXSMM_MATH_DISPATCH1)
  if (LIBXSMM_X86_AVX2 <= libxsmm_target_archid) {
    result = libxsmm_diff_avx2(a, b, size);
  }
  else
# endif
  {
    result = libxsmm_diff_sw(a, b, size);
  }
#endif
  return result;
}


LIBXSMM_API unsigned int libxsmm_diff_n(const void* a, const void* bn, unsigned char size,
  unsigned char stride, unsigned int hint, unsigned int n)
{
  LIBXSMM_ASSERT(size <= stride);
#if (LIBXSMM_X86_AVX2 <= LIBXSMM_STATIC_TARGET_ARCH)
  LIBXSMM_MATH_DIFF(libxsmm_diff_avx2, LIBXSMM_MOD, a, bn, size, stride, hint, n);
#else
  if (LIBXSMM_X86_AVX2 <= libxsmm_target_archid) {
    LIBXSMM_MATH_DIFF(libxsmm_diff_avx2, LIBXSMM_MOD, a, bn, size, stride, hint, n);
  }
  else {
    LIBXSMM_MATH_DIFF(libxsmm_diff_sw, LIBXSMM_MOD, a, bn, size, stride, hint, n);
  }
#endif
}


LIBXSMM_API unsigned int libxsmm_diff_npot(const void* a, const void* bn, unsigned char size,
  unsigned char stride, unsigned int hint, unsigned int n)
{
#if !defined(NDEBUG)
  const unsigned int npot = LIBXSMM_UP2POT(n);
  assert(size <= stride && n == npot); /* !LIBXSMM_ASSERT */
#endif
#if (LIBXSMM_X86_AVX2 <= LIBXSMM_STATIC_TARGET_ARCH)
  LIBXSMM_MATH_DIFF(libxsmm_diff_avx2, LIBXSMM_MOD2, a, bn, size, stride, hint, n);
#else
  if (LIBXSMM_X86_AVX2 <= libxsmm_target_archid) {
    LIBXSMM_MATH_DIFF(libxsmm_diff_avx2, LIBXSMM_MOD2, a, bn, size, stride, hint, n);
  }
  else {
    LIBXSMM_MATH_DIFF(libxsmm_diff_sw, LIBXSMM_MOD2, a, bn, size, stride, hint, n);
  }
#endif
}


LIBXSMM_API unsigned int libxsmm_hash(const void* data, unsigned int size, unsigned int seed)
{
  LIBXSMM_INIT
  return libxsmm_crc32(data, size, seed);
}


LIBXSMM_API size_t libxsmm_gcd(size_t a, size_t b)
{
  while (0 != b) {
    const size_t r = a % b;
    a = b;
    b = r;
  }
  return a;
}


LIBXSMM_API size_t libxsmm_lcm(size_t a, size_t b)
{
  return (a * b) / libxsmm_gcd(a, b);
}


LIBXSMM_API int libxsmm_primes_u32(unsigned int num, unsigned int num_factors_n32[])
{
  unsigned int c = num, i;
  int n = 0;
  if (0 < c && 0 == (c & 1)) { /* non-zero even */
    unsigned int j = c / 2;
    while (c == (2 * j)) {
      num_factors_n32[n++] = 2;
      c = j; j /= 2;
    }
  }
  for (i = 3; i <= c; i += 2) {
    unsigned int j = c / i;
    while (c == (i * j)) {
      num_factors_n32[n++] = i;
      c = j; j /= i;
    }
    if ((i * i) > num) {
      break;
    }
  }
  if (1 < c && 0 != n) {
    num_factors_n32[n++] = c;
  }
  return n;
}


LIBXSMM_API size_t libxsmm_shuffle(unsigned int n)
{
  const unsigned int s = (0 != (n & 1) ? ((n / 2 - 1) | 1) : ((n / 2) & ~1));
  const unsigned int d = (0 != (n & 1) ? 1 : 2);
  unsigned int result = (1 < n ? 1 : 0), i;
  for (i = (d < n ? (n - 1) : 0); d < i; i -= d) {
    unsigned int c = (s <= i ? (i - s) : (s - i));
    unsigned int a = n, b = c;
    do {
      const unsigned int r = a % b;
      a = b;
      b = r;
    } while (0 != b);
    if (1 == a) {
      result = c;
      if (2 * c <= n) {
        i = d; /* break */
      }
    }
  }
  assert((0 == result && 1 >= n) || (result < n && 1 == libxsmm_gcd(result, n)));
  return result;
}


LIBXSMM_API_INLINE unsigned int internal_product_limit(unsigned int product, unsigned int limit)
{
  unsigned int fact[32], maxp = limit, result = 1;
  int i, n;
  /* attempt to lower the memory requirement for DP; can miss best solution */
  if (LIBXSMM_MATH_MAXPRODUCT < limit) {
    const unsigned int minfct = (limit + limit - 1) / LIBXSMM_MATH_MAXPRODUCT;
    const unsigned int maxfct = (unsigned int)libxsmm_gcd(product, limit);
    result = maxfct;
    if (minfct < maxfct) {
      n = libxsmm_primes_u32(result, fact);
      for (i = 0; i < n; ++i) {
        if (minfct < fact[i]) {
          result = fact[i];
          i = n; /* break */
        }
      }
    }
    maxp /= result;
  }
  if (LIBXSMM_MATH_MAXPRODUCT >= maxp) {
    unsigned int k[2][LIBXSMM_MATH_MAXPRODUCT], *k0 = k[0], *k1 = k[1], *kt, p;
    n = libxsmm_primes_u32(product / result, fact);
    /* initialize table with trivial factor */
    for (p = 0; p <= maxp; ++p) k[0][p] = 1;
    k[0][0] = k[1][0] = 1;
    for (i = 1; i <= n; ++i) {
      for (p = 1; p <= maxp; ++p) {
        const unsigned int f = fact[i - 1], h = k0[p];
        if (p < f) {
          k1[p] = h;
        }
        else {
          const unsigned int g = f * k0[p / f];
          k1[p] = LIBXSMM_MAX(g, h);
        }
      }
      kt = k0; k0 = k1; k1 = kt;
    }
    result *= k0[maxp];
  }
  else { /* trivial approximation */
    n = libxsmm_primes_u32(product, fact);
    for (i = 0; i < n; ++i) {
      const unsigned int f = result * fact[i];
      if (f <= limit) {
        result = f;
      }
      else i = n; /* break */
    }
  }
  return result;
}


LIBXSMM_API unsigned int libxsmm_product_limit(unsigned int product, unsigned int limit, int is_lower)
{
  unsigned int result;
  if (1 < limit) { /* check for fast-path */
    result = internal_product_limit(product, limit);
  }
  else {
    result = limit;
  }
  if (0 != is_lower && limit < product) {
    if (result < limit) {
      result = internal_product_limit(product, 2 * limit - 1);
    }
    if (result < limit) {
      result = product;
    }
    LIBXSMM_ASSERT(limit <= result);
  }
  if (product < result) {
    result = product;
  }
  LIBXSMM_ASSERT(result <= product);
  return result;
}


LIBXSMM_API unsigned int libxsmm_isqrt_u64(unsigned long long x)
{
  unsigned long long b; unsigned int y = 0, s;
  for (s = 0x80000000/*2^31*/; 0 < s; s >>= 1) {
    b = y | s; y |= (b * b <= x ? s : 0);
  }
  return y;
}

LIBXSMM_API unsigned int libxsmm_isqrt_u32(unsigned int x)
{
  unsigned int b; unsigned int y = 0; int s;
  for (s = 0x40000000/*2^30*/; 0 < s; s >>= 2) {
    b = y | s; y >>= 1;
    if (b <= x) { x -= b; y |= s; }
  }
  return y;
}


LIBXSMM_API LIBXSMM_INTRINSICS(LIBXSMM_X86_GENERIC) double libxsmm_dsqrt(double x)
{
#if defined(LIBXSMM_INTRINSICS_X86)
  const __m128d a = LIBXSMM_INTRINSICS_MM_UNDEFINED_PD();
  const double result = _mm_cvtsd_f64(_mm_sqrt_sd(a, _mm_set_sd(x)));
#else
  double result, y = x;
  if (LIBXSMM_NEQ(0, x)) {
    do {
      result = y;
      y = 0.5 * (y + x / y);
    } while (LIBXSMM_NEQ(result, y));
  }
  result = y;
#endif
  return result;
}


LIBXSMM_API LIBXSMM_INTRINSICS(LIBXSMM_X86_GENERIC) float libxsmm_ssqrt(float x)
{
#if defined(LIBXSMM_INTRINSICS_X86)
  const float result = _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)));
#else
  float result, y = x;
  if (LIBXSMM_NEQ(0, x)) {
    do {
      result = y;
      y = 0.5f * (y + x / y);
    } while (LIBXSMM_NEQ(result, y));
  }
  result = y;
#endif
  return result;
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

/* Implementation based on Claude Baumann's product (http://www.convict.lu/Jeunes/ultimate_stuff/exp_ln_2.htm). */
LIBXSMM_API float libxsmm_sexp2_fast(float x, int maxiter)
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
#if defined(INFINITY)
        result.s = (0 == sign ? (INFINITY) : 0.f);
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


LIBXSMM_API float libxsmm_sexp2(float x)
{
#if defined(LIBXSMM_NO_LIBM)
  return libxsmm_sexp2_fast(x, 20/*compromise*/);
#else
  return powf(2.f, x);
#endif
}


LIBXSMM_API float libxsmm_sexp2_u8(unsigned char x)
{
  union { int i; float s; } result;
  if (128 > x) {
    if (31 < x) {
      const float r32 = 2.f * ((float)(1U << 31)); /* 2^32 */
      const int n = x >> 5;
      int i;
      result.s = r32;
      for (i = 1; i < n; ++i) result.s *= r32;
      result.s *= (1U << (x - (n << 5)));
    }
    else {
      result.s = (float)(1U << x);
    }
  }
  else {
#if defined(INFINITY)
    result.s = INFINITY;
#else
    result.i = 0x7F800000;
#endif
  }
  return result.s;
}


LIBXSMM_API float libxsmm_sexp2_i8(signed char x)
{
  union { int i; float s; } result;
  if (-128 != x) {
    const signed char ux = (signed char)LIBXSMM_ABS(x);
    if (31 < ux) {
      const float r32 = 2.f * ((float)(1U << 31)); /* 2^32 */
      const int n = ux >> 5;
      int i;
      result.s = r32;
      for (i = 1; i < n; ++i) result.s *= r32;
      result.s *= (1U << (ux - (n << 5)));
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


LIBXSMM_API void libxsmm_srand(unsigned int seed)
{
#if defined(_WIN32) || defined(__CYGWIN__) || !(defined(_SVID_SOURCE) || defined(_XOPEN_SOURCE))
  srand(seed);
#else
  srand48(seed);
#endif
}


LIBXSMM_API unsigned int libxsmm_rand_u32(unsigned int n)
{
#if defined(_WIN32) || defined(__CYGWIN__) || !(defined(_SVID_SOURCE) || defined(_XOPEN_SOURCE))
  const unsigned int rand_max1 = (unsigned int)(RAND_MAX) + 1U;
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


LIBXSMM_API double libxsmm_rand_f64(void)
{
#if defined(_WIN32) || defined(__CYGWIN__) || !(defined(_SVID_SOURCE) || defined(_XOPEN_SOURCE))
  static const double scale = 1.0 / (RAND_MAX);
  return scale * (double)rand();
#else
  return drand48();
#endif
}


#if defined(LIBXSMM_BUILD)

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_hash)(int* hash, const void* /*data*/, const int* /*size*/, const int* /*seed*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_hash)(int* hash, const void* data, const int* size, const int* seed)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != hash && NULL != data && NULL != size && NULL != seed)
#endif
  {
    *hash = (libxsmm_hash(data, *size, *seed) & 0x7FFFFFFF);
  }
#if !defined(NDEBUG)
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_hash specified!\n");
  }
#endif
}

#endif /*defined(LIBXSMM_BUILD)*/

