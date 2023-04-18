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
#include <utils/libxsmm_utils.h>
#include <libxsmm.h>

#if !defined(ELEMTYPE)
# define ELEMTYPE float
#endif
#if !defined(ITRANS) && 1
# define ITRANS
#endif

#if !defined(PRINT) && (defined(_DEBUG) || 0)
# define PRINT
#endif
#if defined(PRINT)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif


unsigned int validate(const ELEMTYPE* a, const ELEMTYPE* b, const ELEMTYPE* c, libxsmm_blasint max_size_b,
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
  unsigned int nerrors = 0, before = 0;
#if defined(PRINT)
  unsigned int ntotal = 0;
#endif
#if (0 != LIBXSMM_JIT)
# if defined(PRINT)
  unsigned int njit = 0;
# endif
  /*const*/ int elemtype = LIBXSMM_DATATYPE(ELEMTYPE);
#endif
  libxsmm_blasint max_size_a = 0, max_size_b = 0, i;
  ELEMTYPE *a = NULL, *b = NULL, *c = NULL;
  const size_t typesize = sizeof(ELEMTYPE);
  int* batchidx = NULL;
  int test, fun;

  void (*otrans[])(void*, const void*, unsigned int, libxsmm_blasint,
    libxsmm_blasint, libxsmm_blasint, libxsmm_blasint) = {
    libxsmm_otrans, libxsmm_otrans_omp
  };
  void (*itrans[])(void*, unsigned int, libxsmm_blasint,
    libxsmm_blasint, libxsmm_blasint, libxsmm_blasint) = {
    libxsmm_itrans/*, libxsmm_itrans_omp*/
  };
  const int nfun = sizeof(otrans) / sizeof(*otrans);

  for (test = start; test < ntests; ++test) {
    const libxsmm_blasint size_a = ldi[test] * n[test], size_b = ldo[test] * m[test];
    LIBXSMM_ASSERT(m[test] <= ldi[test] && n[test] <= ldo[test]);
    max_size_a = LIBXSMM_MAX(max_size_a, size_a);
    max_size_b = LIBXSMM_MAX(max_size_b, size_b);
  }
  a = (ELEMTYPE*)libxsmm_malloc(typesize * max_size_a);
  b = (ELEMTYPE*)libxsmm_malloc(typesize * max_size_b * batchsize);
  c = (ELEMTYPE*)libxsmm_malloc(typesize * max_size_b * batchsize);
  batchidx = (int*)libxsmm_malloc(sizeof(int) * batchsize);
  LIBXSMM_ASSERT(NULL != a && NULL != b && NULL != c);

  /* initialize data */
  LIBXSMM_MATINIT(ELEMTYPE, 42/*seed*/, a, max_size_a, 1, max_size_a, 1.0/*scale*/);
  for (i = 0; i < batchsize; ++i) {
    LIBXSMM_MATINIT(ELEMTYPE, 24 + i/*seed*/, b + (size_t)i * max_size_b,
      max_size_b, 1, max_size_b, 1.0/*scale*/);
    batchidx[i] = i * max_size_b;
  }

  for (fun = 0; fun < nfun; ++fun) {
    for (test = start; test < ntests; ++test) {
      memcpy(c, b, typesize * max_size_b); /* prepare */
      otrans[fun](b, a, (unsigned int)typesize, m[test], n[test], ldi[test], ldo[test]);
      nerrors += validate(a, b, c, max_size_b, m[test], n[test], ldi[test], ldo[test]);
      memcpy(b, c, typesize * max_size_b); /* restore */
      if (before != nerrors) {
        FPRINTF(stderr, "ERROR (libxsmm_otrans): %ix%i-kernel with ldi=%i ldo=%i failed!\n",
          m[test], n[test], ldi[test], ldo[test]);
        before = nerrors;
      }
#if defined(PRINT)
      ++ntotal;
#endif
#if (0 != LIBXSMM_JIT) /* dispatch kernel and check that it is available */
      {
        const libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape(
          m[test], n[test], ldi[test], ldo[test], elemtype, elemtype, elemtype);
        const libxsmm_meltwfunction_unary kernel = libxsmm_dispatch_meltw_unary_v2(
          LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape,
          LIBXSMM_MELTW_FLAG_UNARY_NONE);
        if (NULL != kernel) {
          libxsmm_meltw_unary_param unary_param /*= { 0 }*/;
          unary_param.in.primary = (void*)a;
          unary_param.out.primary = (void*)b;
          memcpy(c, b, typesize * max_size_b); /* prepare */
          kernel(&unary_param);
          nerrors += validate(a, b, c, max_size_b, m[test], n[test], ldi[test], ldo[test]);
          memcpy(b, c, typesize * max_size_b); /* restore */
          if (before != nerrors) {
            FPRINTF(stderr, "ERROR (meltw_unary): %ix%i-kernel with ldi=%i ldo=%i failed!\n",
              m[test], n[test], ldi[test], ldo[test]);
            before = nerrors;
          }
# if defined(PRINT)
          ++njit;
# endif
        }
        else {
          FPRINTF(stderr, "ERROR (meltw_unary): %ix%i-kernel with ldi=%i ldo=%i not generated!\n",
            m[test], n[test], ldi[test], ldo[test]);
          ++nerrors;
        }
# if defined(PRINT)
        ++ntotal;
# endif
      }
#endif
      if (0 == fun) {
        memcpy(c, b, typesize * max_size_b); /* prepare */
        itrans[fun](b, (unsigned int)typesize, m[test], n[test], ldi[test], ldo[test]);
        nerrors += validate(c, b, c, max_size_b, m[test], n[test], ldi[test], ldo[test]);
        memcpy(b, c, typesize * max_size_b); /* restore */
        if (before != nerrors) {
          FPRINTF(stderr, "ERROR (libxsmm_itrans): %ix%i-kernel with ldi=%i ldo=%i failed!\n",
            m[test], n[test], ldi[test], ldo[test]);
          before = nerrors;
        }
#if defined(PRINT)
        ++ntotal;
#endif
      }
    }
  }

#if defined(ITRANS)
  for (test = start; test < ntests; ++test) {
    memcpy(c, b, typesize * max_size_b * batchsize); /* prepare */
    libxsmm_itrans_batch(b, (unsigned int)typesize, m[test], n[test], ldi[test], ldo[test],
      0/*index_base*/, sizeof(int)/*index_stride*/, batchidx, batchsize,
      0/*tid*/, 1/*ntasks*/);
    for (i = 0; i < batchsize; ++i) {
      const size_t stride = (size_t)i * max_size_b;
      nerrors += validate(c + stride, b + stride, c + stride,
        max_size_b, m[test], n[test], ldi[test], ldo[test]);
      if (before != nerrors) {
        FPRINTF(stderr, "ERROR (libxsmm_itrans_batch): %ix%i-kernel with ldi=%i ldo=%i failed!\n",
          m[test], n[test], ldi[test], ldo[test]);
        before = nerrors;
      }
# if defined(PRINT)
      ++ntotal;
# endif
    }
    memcpy(b, c, typesize* max_size_b* batchsize); /* restore */
  }
#endif

  libxsmm_free(batchidx);
  libxsmm_free(a);
  libxsmm_free(b);
  libxsmm_free(c);

  if (0 == nerrors) {
    return EXIT_SUCCESS;
  }
  else {
    FPRINTF(stderr, "total=%u jitted=%u errors=%u (%i%%)\n",
      ntotal, njit, nerrors, (int)LIBXSMM_ROUND(100.0 * nerrors / ntotal));
    return EXIT_FAILURE;
  }
}


unsigned int validate(const ELEMTYPE* a, const ELEMTYPE* b, const ELEMTYPE* c, libxsmm_blasint max_size_b,
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
