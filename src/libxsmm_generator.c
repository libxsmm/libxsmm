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
#include <libxsmm_generator.h>
#include "libxsmm_main.h"
#include "generator_common.h"
#if !defined(_WIN32)
# include <sys/file.h>
#endif

#if !defined(LIBXSMM_PRODUCT_LIMIT)
# define LIBXSMM_PRODUCT_LIMIT 1024
#endif


LIBXSMM_APIVAR_PUBLIC_DEF(unsigned int libxsmm_intrinsics_mm512_rng_state0[16]);
LIBXSMM_APIVAR_PUBLIC_DEF(unsigned int libxsmm_intrinsics_mm512_rng_state1[16]);
LIBXSMM_APIVAR_PUBLIC_DEF(unsigned int libxsmm_intrinsics_mm512_rng_state2[16]);
LIBXSMM_APIVAR_PUBLIC_DEF(unsigned int libxsmm_intrinsics_mm512_rng_state3[16]);

/* definition of corresponding variables */
LIBXSMM_APIVAR_PUBLIC_DEF(int libxsmm_stdio_handle);
LIBXSMM_APIVAR_PUBLIC_DEF(unsigned int libxsmm_ninit);
LIBXSMM_APIVAR_PUBLIC_DEF(int libxsmm_target_archid);
LIBXSMM_APIVAR_PUBLIC_DEF(int libxsmm_verbosity);
LIBXSMM_APIVAR_PUBLIC_DEF(int libxsmm_se);


LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_gemm_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_datatype a_type, libxsmm_datatype b_type, libxsmm_datatype comp_type, libxsmm_datatype c_type,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc, int flags, int prefetch)
{
  /*const*/int no_bypass = LIBXSMM_GEMM_NO_BYPASS_DIMS(lda, ldb, ldc) && LIBXSMM_GEMM_NO_BYPASS_DIMS(m, n, k);
  union {
    libxsmm_gemm_descriptor* ptr;
    libxsmm_descriptor_blob* blob;
  } result = { 0 };
  LIBXSMM_ASSERT(NULL == result.ptr);
  if (no_bypass) {
    unsigned char datatype_field[3];
    result.blob = blob;
    LIBXSMM_ASSERT(NULL != result.ptr);
    LIBXSMM_GEMM_SET_DESC_DATATYPE(a_type, b_type, c_type,
                                   comp_type, datatype_field);
    /* Note: iprec/oprec combination is not checked to omit type-switch (invalid combination may result in BE-error) */
    LIBXSMM_GEMM_DESCRIPTOR(*result.ptr, datatype_field[0], datatype_field[1], datatype_field[2],
                             flags, m, n, k, lda, ldb, ldc, prefetch);
  }
  return result.ptr;
}


LIBXSMM_API libxsmm_meltw_descriptor* libxsmm_meltw_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_datatype in_type, libxsmm_datatype out_type,
  libxsmm_blasint m, libxsmm_blasint n,
  libxsmm_blasint ldi, libxsmm_blasint ldo,
  unsigned short flags, unsigned short param, unsigned char operation)
{
  union {
    libxsmm_meltw_descriptor* ptr;
    libxsmm_descriptor_blob* blob;
  } result = { 0 };
  LIBXSMM_DESCRIPTOR_CLEAR(blob);
  result.blob = blob;
  LIBXSMM_ASSERT(NULL != result.ptr);
  result.ptr->datatype = (unsigned char)LIBXSMM_GETENUM(in_type, out_type);
  result.ptr->datatype2 = 0;
  result.ptr->flags = (unsigned short)flags;
  result.ptr->operation = (unsigned char)operation;
  result.ptr->param = (unsigned short)param;
  result.ptr->ldi = ldi;
  result.ptr->ldo = ldo;
  result.ptr->ldi2 = 0;
  result.ptr->ldi3 = 0;
  result.ptr->m = m;
  result.ptr->n = n;
  return result.ptr;
}


LIBXSMM_API libxsmm_meltw_descriptor* libxsmm_meltw_descriptor_init2(libxsmm_descriptor_blob* blob,
  libxsmm_datatype in0_type, libxsmm_datatype in1_type, libxsmm_datatype in2_type, libxsmm_datatype comp_type, libxsmm_datatype out_type,
  libxsmm_blasint m, libxsmm_blasint n,
  libxsmm_blasint ldi, libxsmm_blasint ldo, libxsmm_blasint ldi2, libxsmm_blasint ldi3,
  unsigned short flags, unsigned short param, unsigned char operation)
{
  union {
    libxsmm_meltw_descriptor* ptr;
    libxsmm_descriptor_blob* blob;
  } result = { 0 };
  LIBXSMM_DESCRIPTOR_CLEAR(blob);
  result.blob = blob;
  LIBXSMM_ASSERT(NULL != result.ptr);
  result.ptr->datatype = (unsigned char)LIBXSMM_GETENUM(in0_type, out_type);
  result.ptr->datatype1 = (unsigned char)LIBXSMM_GETENUM(in1_type, in2_type);
  result.ptr->datatype2 = (unsigned char)LIBXSMM_GETENUM(comp_type, out_type);
  result.ptr->flags = (unsigned short)flags;
  result.ptr->operation = (unsigned char)operation;
  result.ptr->param = (unsigned short)param;
  result.ptr->ldi = ldi;
  result.ptr->ldo = ldo;
  result.ptr->ldi2 = ldi2;
  result.ptr->ldi3 = ldi3;
  result.ptr->m = m;
  result.ptr->n = n;
  return result.ptr;
}


LIBXSMM_API libxsmm_meqn_descriptor* libxsmm_meqn_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_datatype out_type, libxsmm_blasint m, libxsmm_blasint n,
  libxsmm_blasint ldo, unsigned int eqn_idx)
{
  union {
    libxsmm_meqn_descriptor* ptr;
    libxsmm_descriptor_blob* blob;
  } result = { 0 };
  LIBXSMM_DESCRIPTOR_CLEAR(blob);
  result.blob = blob;
  LIBXSMM_ASSERT(NULL != result.ptr);
  result.ptr->datatype = (unsigned char)LIBXSMM_GETENUM( out_type, out_type);
  result.ptr->eqn_idx = eqn_idx;
  result.ptr->ldo = ldo;
  result.ptr->m = m;
  result.ptr->n = n;
  return result.ptr;
}


LIBXSMM_API size_t libxsmm_gcd(size_t a, size_t b)
{
  while (0 != b) {
    const size_t r = a % b;
    a = b; b = r;
  }
  return 0 != a ? a : 1;
}


LIBXSMM_API size_t libxsmm_lcm(size_t a, size_t b)
{
  const size_t gcd = libxsmm_gcd(a, b);
  return 0 != gcd ? ((a / gcd) * b) : 0;
}


LIBXSMM_API unsigned int libxsmm_remainder(unsigned int a, unsigned int b,
  const unsigned int* limit, const unsigned int* remainder)
{
  /* normalize such that a <= b */
  unsigned int ci = ((b < a && 0 != b) ? LIBXSMM_UP(a, b) : b), c = a * ci;
  /* sanitize limit argument */
  if (NULL != limit && (0 == b || ((*limit / b) * b) < a)) limit = NULL;
  if (1 <= a) {
    unsigned int r = a - 1;
    for (; ((NULL != remainder ? *remainder : 0) < r)
        &&  (NULL == limit || ci <= *limit); ci += b)
    {
      const unsigned int ri = ci % a;
      if (ri < r) {
        c = ci;
        r = ri;
      }
    }
  }
  return c;
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


LIBXSMM_API_INLINE unsigned int internal_product_limit(unsigned int product, unsigned int limit)
{
  unsigned int fact[32], maxp = limit, result = 1;
  int i, n;
  /* attempt to lower the memory requirement for DP; can miss best solution */
  if (LIBXSMM_PRODUCT_LIMIT < limit) {
    const unsigned int minfct = (limit + limit - 1) / LIBXSMM_PRODUCT_LIMIT;
    const unsigned int maxfct = (unsigned int)libxsmm_gcd(product, limit);
    result = maxfct;
    if (minfct < maxfct) {
      n = libxsmm_primes_u32(result, fact);
      for (i = 0; i < n; ++i) {
        if (minfct < fact[i]) {
          result = fact[i];
          break;
        }
      }
    }
    maxp /= result;
  }
  if (LIBXSMM_PRODUCT_LIMIT >= maxp) {
    unsigned int k[2][LIBXSMM_PRODUCT_LIMIT] = { {0} }, *k0 = k[0], *k1 = k[1], *kt, p;
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
      else break;
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
  if (0 != is_lower) {
    if (limit < product) {
      if (result < limit) {
        result = internal_product_limit(product, 2 * limit - 1);
      }
      if (result < limit) {
        result = product;
      }
      LIBXSMM_ASSERT(limit <= result);
    }
    else if (0 != product) {
      result = LIBXSMM_UP(limit, product);
    }
    else result = 0;
  }
  else if (product < result) {
    result = product;
  }
  LIBXSMM_ASSERT(0 != is_lower || result <= product);
  return result;
}


LIBXSMM_API void libxsmm_stdio_acquire(void)
{
#if !defined(_WIN32)
  if (0 < libxsmm_stdio_handle) {
    flock(libxsmm_stdio_handle - 1, LOCK_EX);
  }
  else
#endif
  {
    LIBXSMM_FLOCK(stdout);
    LIBXSMM_FLOCK(stderr);
  }
}


LIBXSMM_API void libxsmm_stdio_release(void)
{
#if !defined(_WIN32)
  if (0 < libxsmm_stdio_handle) {
    flock(libxsmm_stdio_handle - 1, LOCK_UN);
  }
  else
#endif
  {
    LIBXSMM_FUNLOCK(stderr);
    LIBXSMM_FUNLOCK(stdout);
  }
}
