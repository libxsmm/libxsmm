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
#ifndef MAGAZINE_H
#define MAGAZINE_H

#include <stdio.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(_WIN32)
# include <Windows.h>
#else
# include <sys/time.h>
#endif

#if !defined(TYPE)
# define TYPE double
#endif

#if !defined(ALPHA)
# define ALPHA 1
#endif
#if !defined(BETA)
# define BETA 1
#endif

#if 1
# define STREAM_A(EXPR) (EXPR)
#else
# define STREAM_A(EXPR) 0
#endif
#if 1
# define STREAM_B(EXPR) (EXPR)
#else
# define STREAM_B(EXPR) 0
#endif
#if 1
# define STREAM_C(EXPR) (EXPR)
#else
# define STREAM_C(EXPR) 0
#endif

/**
 * Permuting the data introduces a dependency to LIBXSMM
 * even for the Eigen/Blaze/Blas based sample code.
 */
#if 0 /* process batch of A, B, and C in "random" order */
# define SHUFFLE
#endif

#if 0 /* PAD (alignment) must be power of two */
# define PAD 64
#else
# define PAD 1
#endif

#if defined(__cplusplus)
# define INLINE_KEYWORD inline
# define INLINE INLINE_KEYWORD
#else /* C */
# if defined(__STDC_VERSION__) && (199901L <= __STDC_VERSION__) /*C99*/
#   define INLINE_KEYWORD inline
# elif defined(_MSC_VER)
#   define INLINE_KEYWORD __inline
#   define INLINE_FIXUP
# endif
# if !defined(INLINE_KEYWORD)
#   define INLINE_KEYWORD
#   define INLINE_FIXUP
# endif
# define INLINE static INLINE_KEYWORD
#endif
#if defined(INLINE_FIXUP) && !defined(inline)
# define inline INLINE_KEYWORD
#endif

#if defined(SHUFFLE)
# include <libxsmm_source.h>
#endif


INLINE void init(int seed, TYPE* dst, int nrows, int ncols, int ld, double scale) {
  const double seed1 = scale * seed + scale;
  int i, j;
  for (i = 0; i < ncols; ++i) {
    for (j = 0; j < nrows; ++j) {
      const int k = i * ld + j;
      dst[k] = (TYPE)(seed1 * ((unsigned long long)i * nrows + j + 1));
    }
    for (; j < ld; ++j) {
      const int k = i * ld + j;
      dst[k] = (TYPE)(seed);
    }
  }
}


INLINE double norm(const TYPE* src, int nrows, int ncols, int ld) {
  int i, j;
  double result = 0, comp = 0;
  for (i = 0; i < ncols; ++i) {
    for (j = 0; j < nrows; ++j) {
      const int k = i * ld + j;
      const double v = src[k], a = (0 <= v ? v : -v) - comp, b = result + a;
      comp = (b - result) - a;
      result = b;
    }
  }
  return result;
}


INLINE double seconds(void) {
#if defined(_OPENMP)
  return omp_get_wtime();
#elif defined(_WIN32)
  LARGE_INTEGER t, f;
  QueryPerformanceCounter(&t);
  QueryPerformanceFrequency(&f);
  return (double)t.QuadPart / f.QuadPart;
#else
  struct timeval t;
  gettimeofday(&t, 0);
  return 1E-6 * (1000000ULL * t.tv_sec + t.tv_usec);
#endif
}

#endif /*MAGAZINE_H*/
