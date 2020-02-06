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


int main(void)
{
  /* test #:                      1  2  3  4  5  6  7  8  9 10 11 12  13  14  15  16  17  18  19   20   21   22    23 */
  /* index:                       0  1  2  3  4  5  6  7  8  9 10 11  12  13  14  15  16  17  18   19   20   21    22 */
  const libxsmm_blasint m[]   = { 0, 1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 5,  5, 13, 13, 16, 22, 63, 64,  16,  16,  75, 2507 };
  const libxsmm_blasint n[]   = { 0, 1, 7, 7, 7, 2, 3, 1, 1, 1, 1, 5, 13,  5, 13, 16, 22, 31, 64, 500,  32, 130, 1975 };
  const libxsmm_blasint ldi[] = { 0, 1, 1, 1, 9, 2, 3, 4, 5, 8, 8, 5,  5, 13, 13, 16, 22, 64, 64,  16, 512,  87, 3000 };
  const libxsmm_blasint ldo[] = { 1, 1, 7, 8, 8, 2, 3, 1, 1, 1, 4, 5, 13,  5, 13, 16, 22, 32, 64, 512,  64, 136, 3072 };
  const int start = 0, ntests = sizeof(m) / sizeof(*m);
  libxsmm_blasint max_size_a = 0, max_size_b = 0;
  ELEM_TYPE *a = NULL, *b = NULL, *c = NULL;
  const size_t typesize = sizeof(ELEM_TYPE);
  unsigned int nerrors = 0;
  int test, fun;

  void (*otrans[])(void*, const void*, unsigned int, libxsmm_blasint,
    libxsmm_blasint, libxsmm_blasint, libxsmm_blasint) = {
      libxsmm_otrans, libxsmm_otrans_omp
    };
  void (*itrans[])(void*, unsigned int, libxsmm_blasint,
    libxsmm_blasint, libxsmm_blasint) = {
      libxsmm_itrans, libxsmm_itrans/*_omp*/
    };

  for (test = start; test < ntests; ++test) {
    const libxsmm_blasint size_a = ldi[test] * n[test], size_b = ldo[test] * m[test];
    LIBXSMM_ASSERT(m[test] <= ldi[test] && n[test] <= ldo[test]);
    max_size_a = LIBXSMM_MAX(max_size_a, size_a);
    max_size_b = LIBXSMM_MAX(max_size_b, size_b);
  }
  a = (ELEM_TYPE*)libxsmm_malloc(typesize * max_size_a);
  b = (ELEM_TYPE*)libxsmm_malloc(typesize * max_size_b);
  c = (ELEM_TYPE*)libxsmm_malloc(typesize * max_size_b);
  LIBXSMM_ASSERT(NULL != a && NULL != b && NULL != c);

  /* initialize data */
  LIBXSMM_MATINIT(ELEM_TYPE, 42, a, max_size_a, 1, max_size_a, 1.0);
  LIBXSMM_MATINIT(ELEM_TYPE, 24, b, max_size_b, 1, max_size_b, 1.0);

  for (fun = 0; fun < 2; ++fun) {
    for (test = start; test < ntests; ++test) {
      memcpy(c, b, typesize * max_size_b);
      otrans[fun](b, a, (unsigned int)typesize, m[test], n[test], ldi[test], ldo[test]);
      { libxsmm_blasint testerrors = 0, i, j; /* validation */
        for (i = 0; i < n[test]; ++i) {
          for (j = 0; j < m[test]; ++j) {
            const libxsmm_blasint u = i * ldi[test] + j;
            const libxsmm_blasint v = j * ldo[test] + i;
            if (LIBXSMM_NEQ(a[u], b[v])) {
              ++testerrors; i = n[test]; break;
            }
          }
          for (j = m[test]; j < ldi[test] && 0 == testerrors; ++j) {
            const libxsmm_blasint v = j * ldo[test] + i;
            if (v < max_size_b && LIBXSMM_NEQ(b[v], c[v])) {
              ++testerrors;
            }
          }
        }
        for (i = n[test]; i < ldo[test] && 0 == testerrors; ++i) {
          for (j = 0; j < m[test]; ++j) {
            const libxsmm_blasint v = j * ldo[test] + i;
            if ((v < max_size_b && LIBXSMM_NEQ(b[v], c[v])) || v >= max_size_b) {
              ++testerrors; break;
            }
          }
          for (j = m[test]; j < ldi[test] && 0 == testerrors; ++j) {
            const libxsmm_blasint v = j * ldo[test] + i;
            if (v < max_size_b && LIBXSMM_NEQ(b[v], c[v])) {
              ++testerrors;
            }
          }
        }
#if (0 != LIBXSMM_JIT) /* dispatch kernel and check that it is available */
        if (LIBXSMM_X86_AVX <= libxsmm_get_target_archid() && (4 == typesize || 8 == typesize)) {
          libxsmm_descriptor_blob blob;
          const libxsmm_trans_descriptor *const desc = libxsmm_trans_descriptor_init(
            &blob, (unsigned int)typesize, m[test], n[test], ldo[test]);
          const libxsmm_xtransfunction kernel = libxsmm_dispatch_trans(desc);
          if (NULL == kernel) {
# if defined(_DEBUG)
            fprintf(stderr, "\nERROR: kernel %i.%i not generated!\n", fun + 1, test + 1);
# endif
            ++testerrors;
          }
        }
#endif
        nerrors += testerrors;
      }

      if (LIBXSMM_MAX(n[test], 1) > ldi[test] || 0 != fun) continue;
#if 1 /* TODO */
      if (m[test] != ldi[test] || n[test] != ldi[test]) continue;
#endif
      memcpy(c, b, typesize * max_size_b);
      itrans[fun](b, (unsigned int)typesize, m[test], n[test], ldi[test]);
      { libxsmm_blasint testerrors = 0, i, j; /* validation */
        for (i = 0; i < n[test]; ++i) {
          for (j = 0; j < m[test]; ++j) {
            const libxsmm_blasint u = i * ldi[test] + j;
            if (LIBXSMM_NEQ(a[u], b[u])) {
              ++testerrors; i = n[test]; break;
            }
          }
        }
        nerrors += testerrors;
      }
    }
  }

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

