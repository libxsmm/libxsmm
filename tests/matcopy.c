/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>

#if !defined(ELEM_TYPE)
# define ELEM_TYPE float
#endif
#if !defined(TEST_MZERO)
# define TEST_MZERO
#endif
#if !defined(TEST_MCOPY)
# define TEST_MCOPY
#endif
#if !defined(TEST_JIT)
# define TEST_JIT
#endif

#if LIBXSMM_EQUAL(ELEM_TYPE, float) || LIBXSMM_EQUAL(ELEM_TYPE, double)
# if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
#   include <mkl_trans.h>
#   define MATCOPY_GOLD(M, N, A, LDI, B, LDO) \
      LIBXSMM_CONCATENATE(mkl_, LIBXSMM_TPREFIX(ELEM_TYPE, omatcopy))('C', 'n', \
        (size_t)(*(M)), (size_t)(*(N)), (ELEM_TYPE)1, A, (size_t)(*(LDI)), B, (size_t)(*(LDO)))
# elif defined(__OPENBLAS77) && 0/* issue #390 */
#   include <f77blas.h>
#   define MATCOPY_GOLD(M, N, A, LDI, B, LDO) { \
      /*const*/char matcopy_gold_tc_ = 'C', matcopy_gold_tt_ = 'n'; \
      /*const*/ELEM_TYPE matcopy_gold_alpha_ = 1; \
      LIBXSMM_FSYMBOL(LIBXSMM_TPREFIX(ELEM_TYPE, omatcopy))(&matcopy_gold_tc_, &matcopy_gold_tt_, \
        (libxsmm_blasint*)(M), (libxsmm_blasint*)(N), &matcopy_gold_alpha_, \
        A, (libxsmm_blasint*)(LDI), B, (libxsmm_blasint*)(LDO)); \
    }
# endif
#endif


int main(void)
{
#if defined(TEST_MZERO) || defined(TEST_MCOPY) || defined(TEST_JIT)
  /* test#:                       1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18  19   20  21  22   23   24    25 */
  /* index:                       0  1  2  3  4  5  6  7  8   9 10 11 12 13 14 15 16 17  18   19  20  21   22   23    24 */
  const libxsmm_blasint m[]   = { 0, 0, 0, 1, 1, 1, 1, 1, 2,  2, 3, 4, 6, 6, 6, 6, 9, 9,  9,   8, 16, 63,  16,  16, 2507 };
  const libxsmm_blasint n[]   = { 0, 0, 1, 0, 1, 6, 7, 7, 2,  4, 3, 4, 1, 1, 1, 1, 5, 9, 23, 250, 16, 31, 500, 448, 1975 };
  const libxsmm_blasint ldi[] = { 0, 1, 1, 1, 1, 1, 2, 2, 2, 17, 3, 6, 6, 8, 6, 7, 9, 9,  9, 512, 16, 63,  16, 512, 3000 };
  const libxsmm_blasint ldo[] = { 0, 1, 1, 1, 1, 1, 1, 8, 2,  2, 3, 4, 6, 6, 8, 8, 9, 9,  9,  16, 16, 64, 512,  16, 3072 };
  const int start = 0, ntests = sizeof(m) / sizeof(*m);
  libxsmm_blasint max_size_a = 0, max_size_b = 0, i, j;
  unsigned int nerrors = 0;
  ELEM_TYPE *a = 0, *b = 0;
# if defined(MATCOPY_GOLD)
  ELEM_TYPE *c = 0;
# endif
  void (*matcopy[])(void*, const void*, unsigned int, libxsmm_blasint, libxsmm_blasint, libxsmm_blasint, libxsmm_blasint) = {
    libxsmm_matcopy, libxsmm_matcopy_omp
  };
  int test, fun;

  for (test = start; test < ntests; ++test) {
    const libxsmm_blasint size_a = ldi[test] * n[test], size_b = ldo[test] * n[test];
    assert(m[test] <= ldi[test] && m[test] <= ldo[test]);
    max_size_a = LIBXSMM_MAX(max_size_a, size_a);
    max_size_b = LIBXSMM_MAX(max_size_b, size_b);
  }
  a = (ELEM_TYPE*)libxsmm_malloc((size_t)(max_size_a * sizeof(ELEM_TYPE)));
  b = (ELEM_TYPE*)libxsmm_malloc((size_t)(max_size_b * sizeof(ELEM_TYPE)));
  assert(NULL != a && NULL != b);

  LIBXSMM_MATINIT_OMP(ELEM_TYPE, 42, a, max_size_a, 1, max_size_a, 1.0);
# if defined(MATCOPY_GOLD)
  c = (ELEM_TYPE*)libxsmm_malloc((size_t)(max_size_b * sizeof(ELEM_TYPE)));
  assert(NULL != c);
# endif

  for (fun = 0; fun < 2; ++fun) {
    for (test = start; test < ntests; ++test) {
      ELEM_TYPE pattern;
      memset(b, -1, (size_t)(max_size_b * sizeof(ELEM_TYPE)));
      pattern = b[0]; /* -NaN */
# if defined(TEST_MZERO)
      matcopy[fun](b, NULL, sizeof(ELEM_TYPE), m[test], n[test], ldi[test], ldo[test]);
      for (i = 0; i < n[test]; ++i) {
        for (j = 0; j < m[test]; ++j) {
          const ELEM_TYPE u = 0;
          const ELEM_TYPE v = b[i*ldo[test]+j];
          if (LIBXSMM_NEQ(u, v)) {
            ++nerrors;
          }
        }
        for (j = m[test]; j < ldo[test]; ++j) {
          if (0 != memcmp(&pattern, b + (size_t)i * ldo[test] + j, sizeof(ELEM_TYPE))) {
            ++nerrors;
          }
        }
      }
      if (0 != nerrors) { /* break-out */
        fun = test = INT_MAX; break;
      }
# endif
# if defined(TEST_MCOPY)
      matcopy[fun](b, a, sizeof(ELEM_TYPE), m[test], n[test], ldi[test], ldo[test]);
      for (i = 0; i < n[test]; ++i) {
        for (j = 0; j < m[test]; ++j) {
          const ELEM_TYPE u = a[i*ldi[test]+j];
          const ELEM_TYPE v = b[i*ldo[test]+j];
          if (LIBXSMM_NEQ(u, v)) {
            ++nerrors;
          }
        }
        for (j = m[test]; j < ldo[test]; ++j) {
          if (0 != memcmp(&pattern, b + (size_t)i * ldo[test] + j, sizeof(ELEM_TYPE))) {
            ++nerrors;
          }
        }
      }
      if (0 != nerrors) { /* break-out */
        fun = test = INT_MAX; break;
      }
# endif
# if defined(MATCOPY_GOLD)
      if (0 == fun) {
        MATCOPY_GOLD(m + test, n + test, a, ldi + test, c, ldo + test);
        for (i = 0; i < n[test]; ++i) {
          for (j = 0; j < m[test]; ++j) {
            const ELEM_TYPE u = b[i*ldo[test]+j];
            const ELEM_TYPE v = c[i*ldo[test]+j];
            if (LIBXSMM_NEQ(u, v)) {
              ++nerrors;
            }
          }
          for (j = m[test]; j < ldo[test]; ++j) {
            if (0 != memcmp(&pattern, b + (size_t)i * ldo[test] + j, sizeof(ELEM_TYPE))) {
              ++nerrors;
            }
          }
        }
        if (0 != nerrors) { /* break-out */
          fun = test = INT_MAX; break;
        }
      }
# endif
# if defined(TEST_JIT) && (0 != LIBXSMM_JIT) /* dispatch and check kernel */
      if (0 == fun
        && (LIBXSMM_X86_AVX2 <= libxsmm_get_target_archid() &&
            LIBXSMM_AARCH64_ALLFEAT >= libxsmm_get_target_archid())
        && (/*LIBXSMM_DATATYPE_F64 == LIBXSMM_DATATYPE(ELEM_TYPE) ||*/
            LIBXSMM_DATATYPE_F32 == LIBXSMM_DATATYPE(ELEM_TYPE)))
      {
        const libxsmm_datatype type = LIBXSMM_DATATYPE(ELEM_TYPE);
        const libxsmm_meltwfunction_unary kernel = libxsmm_dispatch_meltw_unary(
          m[test], n[test], ldi + test, ldo + test, type, type, type,
          LIBXSMM_MELTW_FLAG_UNARY_NONE,
#   if 1
          LIBXSMM_MELTW_TYPE_UNARY_IDENTITY/*mcopy*/
#   else
          LIBXSMM_MELTW_TYPE_UNARY_XOR/*mzero*/
#   endif
        );
        if (NULL == kernel) {
#   if defined(_DEBUG)
          fprintf(stderr, "\nERROR: kernel %i.%i not generated!\n", fun + 1, test + 1);
#   endif
          ++nerrors; fun = test = INT_MAX; break; /* break-out */
        }
      }
# endif
    }
  }

  libxsmm_free(a);
  libxsmm_free(b);
# if defined(MATCOPY_GOLD)
  libxsmm_free(c);
# endif

  if (0 != nerrors) {
# if defined(_DEBUG)
    fprintf(stderr, "errors=%u\n", nerrors);
# endif
    return EXIT_FAILURE;
  }
#endif
  return EXIT_SUCCESS;
}

