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
# define ELEM_TYPE double
#endif


unsigned int validate(const ELEM_TYPE* a, const ELEM_TYPE* b, const ELEM_TYPE* c, libxsmm_blasint max_size_b,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo);

int main(void)
{
  /* test #:                      1  2  3  4  5  6  7  8  9 10 11 12  13  14  15  16  17  18  19  20  21   22   23   24    25 */
  /* index:                       0  1  2  3  4  5  6  7  8  9 10 11  12  13  14  15  16  17  18  19  20   21   22   23    24 */
  const libxsmm_blasint m[]   = { 0, 1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 5,  5, 13, 13, 16, 22, 23, 23, 63, 64,  16,  16,  75, 2507 };
  const libxsmm_blasint n[]   = { 0, 1, 7, 7, 7, 2, 3, 1, 1, 1, 1, 5, 13,  5, 13, 16, 22, 17, 29, 31, 64, 500,  32, 130, 1975 };
  const libxsmm_blasint ldi[] = { 0, 1, 1, 1, 9, 2, 3, 4, 5, 8, 8, 5,  5, 13, 13, 16, 22, 24, 32, 64, 64,  16, 512,  87, 3000 };
  const libxsmm_blasint ldo[] = { 1, 1, 7, 8, 8, 2, 3, 1, 1, 1, 4, 5, 13,  5, 13, 16, 22, 24, 32, 32, 64, 512,  64, 136, 3072 };
  const libxsmm_blasint batchsize = 13;
  const int start = 0, ntests = sizeof(m) / sizeof(*m);
  libxsmm_blasint max_size_a = 0, max_size_b = 0, i;
  ELEM_TYPE *a = NULL, *b = NULL, *c = NULL;
  const size_t typesize = sizeof(ELEM_TYPE);
  unsigned int nerrors = 0;
  int* batchidx = NULL;
  int test, fun;

  void (*otrans[])(void*, const void*, unsigned int, libxsmm_blasint,
    libxsmm_blasint, libxsmm_blasint, libxsmm_blasint) = {
    libxsmm_otrans, libxsmm_otrans_omp
  };
  void (*itrans[])(void*, unsigned int, libxsmm_blasint,
    libxsmm_blasint, libxsmm_blasint, libxsmm_blasint) = {
    libxsmm_itrans, libxsmm_itrans/*_omp*/
  };

  for (test = start; test < ntests; ++test) {
    const libxsmm_blasint size_a = ldi[test] * n[test], size_b = ldo[test] * m[test];
    LIBXSMM_ASSERT(m[test] <= ldi[test] && n[test] <= ldo[test]);
    max_size_a = LIBXSMM_MAX(max_size_a, size_a);
    max_size_b = LIBXSMM_MAX(max_size_b, size_b);
  }
  a = (ELEM_TYPE*)libxsmm_malloc(typesize * max_size_a);
  b = (ELEM_TYPE*)libxsmm_malloc(typesize * max_size_b * batchsize);
  c = (ELEM_TYPE*)libxsmm_malloc(typesize * max_size_b * batchsize);
  batchidx = (int*)libxsmm_malloc(sizeof(int) * batchsize);
  LIBXSMM_ASSERT(NULL != a && NULL != b && NULL != c);

  /* initialize data */
  LIBXSMM_MATINIT(ELEM_TYPE, 42/*seed*/, a, max_size_a, 1, max_size_a, 1.0/*scale*/);
  for (i = 0; i < batchsize; ++i) {
    LIBXSMM_MATINIT(ELEM_TYPE, 24 + i/*seed*/, b + (size_t)i * max_size_b,
      max_size_b, 1, max_size_b, 1.0/*scale*/);
    batchidx[i] = i * max_size_b;
  }

  for (fun = 0; fun < 2; ++fun) {
    for (test = start; test < ntests; ++test) {
      memcpy(c, b, typesize * max_size_b);
      otrans[fun](b, a, (unsigned int)typesize, m[test], n[test], ldi[test], ldo[test]);
      nerrors += validate(a, b, c, max_size_b, m[test], n[test], ldi[test], ldo[test]);
#if (0 != LIBXSMM_JIT) /* dispatch kernel and check that it is available */
      if (LIBXSMM_X86_AVX2 <= libxsmm_get_target_archid() &&
          LIBXSMM_X86_ALLFEAT >= libxsmm_get_target_archid()
        && (LIBXSMM_DATATYPE_F64 == LIBXSMM_DATATYPE(ELEM_TYPE) ||
            LIBXSMM_DATATYPE_F32 == LIBXSMM_DATATYPE(ELEM_TYPE)))
      {
        const libxsmm_meltwfunction_unary kernel = libxsmm_dispatch_meltw_unary(
          m[test], n[test], ldi + test, ldo + test,
          LIBXSMM_DATATYPE(ELEM_TYPE), LIBXSMM_DATATYPE(ELEM_TYPE), LIBXSMM_DATATYPE(ELEM_TYPE),
          LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
        if (NULL == kernel) {
# if defined(_DEBUG)
          fprintf(stderr, "\nERROR: kernel %i.%i not generated!\n", fun + 1, test + 1);
# endif
          ++nerrors;
        }
      }
#endif
      if (0 == fun) {
        memcpy(c, b, typesize * max_size_b);
        itrans[fun](b, (unsigned int)typesize, m[test], n[test], ldi[test], ldo[test]);
        nerrors += validate(c, b, c, max_size_b, m[test], n[test], ldi[test], ldo[test]);
      }
    }
  }

  for (test = start; test < ntests; ++test) {
    memcpy(c, b, typesize * max_size_b * batchsize);
    libxsmm_itrans_batch(b, (unsigned int)typesize, m[test], n[test], ldi[test], ldo[test],
      0/*index_base*/, sizeof(int)/*index_stride*/, batchidx, batchsize,
      0/*tid*/, 1/*ntasks*/);
    for (i = 0; i < batchsize; ++i) {
      const size_t stride = (size_t)i * max_size_b;
      nerrors += validate(c + stride, b + stride, c + stride,
        max_size_b, m[test], n[test], ldi[test], ldo[test]);
    }
  }

  libxsmm_free(batchidx);
  libxsmm_free(a);
  libxsmm_free(b);
  libxsmm_free(c);

  if (0 == nerrors) {
    return EXIT_SUCCESS;
  }
  else {
# if defined(_DEBUG)
    fprintf(stderr, "errors=%u\n", nerrors);
# endif
    return EXIT_FAILURE;
  }
}


unsigned int validate(const ELEM_TYPE* a, const ELEM_TYPE* b, const ELEM_TYPE* c, libxsmm_blasint max_size_b,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ldi, libxsmm_blasint ldo)
{
  unsigned int result = 0;
  libxsmm_blasint i, j;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < m; ++j) {
      const libxsmm_blasint u = i * ldi + j;
      const libxsmm_blasint v = j * ldo + i;
      if (LIBXSMM_NEQ(a[u], b[v])) {
        ++result; i = n; break;
      }
    }
    for (j = m; j < ldi && 0 == result; ++j) {
      const libxsmm_blasint v = j * ldo + i;
      if (v < max_size_b && LIBXSMM_NEQ(b[v], c[v])) {
        ++result;
      }
    }
  }
  for (i = n; i < ldo && 0 == result; ++i) {
    for (j = 0; j < m; ++j) {
      const libxsmm_blasint v = j * ldo + i;
      if ((v < max_size_b && LIBXSMM_NEQ(b[v], c[v])) || v >= max_size_b) {
        ++result; break;
      }
    }
    for (j = m; j < ldi && 0 == result; ++j) {
      const libxsmm_blasint v = j * ldo + i;
      if (v < max_size_b && LIBXSMM_NEQ(b[v], c[v])) {
        ++result;
      }
    }
  }
  return result;
}
