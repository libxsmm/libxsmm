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
#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if !defined(LIBXSMM_NO_LIBM)
# include <math.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API int libxsmm_matdiff(libxsmm_datatype datatype, libxsmm_blasint m, libxsmm_blasint n,
  const void* ref, const void* tst, const libxsmm_blasint* ldref, const libxsmm_blasint* ldtst,
  libxsmm_matdiff_info* info)
{
  int result = EXIT_SUCCESS;
  if (0 != ref && 0 != tst && 0 != info) {
    libxsmm_blasint mm = m, nn = n, ldr = (0 == ldref ? m : *ldref), ldt = (0 == ldtst ? m : *ldtst);
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
        fprintf(stderr, "LIBXSMM ERROR: unsupported data-type requested for libxsmm_matdiff!\n");
      }
      result = EXIT_FAILURE;
    }
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) { /* square-root without libm dependency */
    int i;
    if (0 < info->l2_abs) {
      const double squared = info->l2_abs; info->l2_abs *= 0.5;
      for (i = 0; i < 16; ++i) info->l2_abs = 0.5 * (info->l2_abs + squared / info->l2_abs);
    }
    if (0 < info->l2_rel) {
      const double squared = info->l2_rel; info->l2_rel *= 0.5;
      for (i = 0; i < 16; ++i) info->l2_rel = 0.5 * (info->l2_rel + squared / info->l2_rel);
    }
    if (0 < info->normf_rel) {
      const double squared = info->normf_rel; info->normf_rel *= 0.5;
      for (i = 0; i < 16; ++i) info->normf_rel = 0.5 * (info->normf_rel + squared / info->normf_rel);
    }
    if (1 == n) {
      const libxsmm_blasint tmp = info->linf_abs_m;
      info->linf_abs_m = info->linf_abs_n;
      info->linf_abs_n = tmp;
    }
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


LIBXSMM_API unsigned int libxsmm_sqrt_u64(unsigned long long n)
{
  unsigned long long b; unsigned int y = 0, s;
  for (s = 0x80000000/*2^31*/; 0 < s; s >>= 1) {
    b = y | s; y |= (b * b <= n ? s : 0);
  }
  return y;
}

LIBXSMM_API unsigned int libxsmm_sqrt_u32(unsigned int n)
{
  unsigned int b; unsigned int y = 0; int s;
  for (s = 0x40000000/*2^30*/; 0 < s; s >>= 2) {
    b = y | s; y >>= 1;
    if (b <= n) { n -= b; y |= s; }
  }
  return y;
}


LIBXSMM_API unsigned int libxsmm_cbrt_u64(unsigned long long n)
{
  unsigned long long b; unsigned int y = 0; int s;
  for (s = 63; 0 <= s; s -= 3) {
    y += y; b = 3 * y * ((unsigned long long)y + 1) + 1;
    if (b <= (n >> s)) { n -= b << s; ++y; }
  }
  return y;
}


LIBXSMM_API unsigned int libxsmm_cbrt_u32(unsigned int n)
{
  unsigned int b; unsigned int y = 0; int s;
  for (s = 30; 0 <= s; s -= 3) {
    y += y; b = 3 * y * (y + 1) + 1;
    if (b <= (n >> s)) { n -= b << s; ++y; }
  }
  return y;
}

LIBXSMM_API float libxsmm_sexp2_fast(float x, int maxiter)
{
  static const float lut[] = { /* tabulated powf(2.f, powf(2.f, -index)) */
    2.00000000f, 1.41421354f, 1.18920708f, 1.09050775f, 1.04427373f, 1.02189720f, 1.01088929f, 1.00542986f,
    1.00271130f, 1.00135469f, 1.00067711f, 1.00033855f, 1.00016928f, 1.00008464f, 1.00004232f, 1.00002110f,
    1.00001061f, 1.00000525f, 1.00000262f, 1.00000131f, 1.00000072f, 1.00000036f, 1.00000012f
  };
  const int lut_size = sizeof(lut) / sizeof(*lut), lut_size1 = lut_size - 1;
  const int *const raw = (const int*)&x;
  const int sign = (0 == (*raw & 0x80000000) ? 0 : 1);
  const int temp = *raw & 0x7FFFFFFF; /* clear sign */
  const int unbiased = (temp >> 23) - 127; /* exponent */
  const int exponent = -unbiased;
  int mantissa = (temp << 8) | 0x80000000;
  float result;
  if (lut_size1 >= exponent) {
    if (lut_size1 != exponent) { /* multiple lookups needed */
      if (7 >= unbiased) { /* not a degenerated case */
        const int n = (0 >= maxiter || lut_size1 <= maxiter) ? lut_size1 : maxiter;
        int i = 1;
        if (0 > unbiased) { /* regular/main case */
          LIBXSMM_ASSERT(0 <= exponent && exponent < lut_size);
          result = lut[exponent]; /* initial value */
          i = exponent + 1; /* next LUT offset */
        }
        else {
          result = 2.f; /* lut[0] */
          i = 1; /* next LUT offset */
        }
        for (; i <= n && 0 != mantissa; ++i) {
          mantissa <<= 1;
          if (0 != (mantissa & 0x80000000)) { /* check MSB */
            LIBXSMM_ASSERT(0 <= i && i < lut_size);
            result *= lut[i]; /* TODO: normalized multiply */
          }
        }
        for (i = 0; i < unbiased; ++i) { /* compute squares */
          result *= result;
        }
        if (0 != sign) { /* negative value, so reciprocal */
          result = 1.f / result;
        }
      }
      else { /* out of range */
#if defined(INFINITY)
        result = (0 == sign ? (INFINITY) : +0.f);
#else
        static const union { int i; float s; } infinity = { 0x7F800000 };
        result = (0 == sign ? infinity.s : +0.f);
#endif
      }
    }
    else if (0 == sign) {
      result = lut[lut_size1];
    }
    else { /* reciprocal */
      result = 1.f / lut[lut_size1];
    }
  }
  else {
    result = 1.f; /* case 2^0 */
  }
  return result;
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
  float result;
  if (128 > x) {
    if (31 < x) {
      const float r32 = 2.f * ((float)(1U << 31)); /* 2^32 */
      const int n = x >> 5;
      int i;
      result = r32;
      for (i = 1; i < n; ++i) result *= r32;
      result *= (1U << (x - (n << 5)));
    }
    else {
      result = (float)(1U << x);
    }
  }
  else {
#if defined(INFINITY)
    result = INFINITY;
#else
    static const union { int i; float s; } infinity = { 0x7F800000 };
    result = infinity.s;
#endif
  }
  return result;
}

