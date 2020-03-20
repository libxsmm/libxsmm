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
#if 0
# define STREAM_C(EXPR) (EXPR)
# define SYNC(IDX, INC, END) ((IDX) * (INC))
#elif defined(_OPENMP)
# define STREAM_C(EXPR) (EXPR)
# define SYNC(IDX, INC, END) (((1048573 * omp_get_thread_num()) % (END)) * (INC))
#else
# define STREAM_C(EXPR) 0
/* synchronization among C matrices */
# define SYNC(IDX, INC, END) 0
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

#if defined(SHUFFLE)
# include <libxsmm_source.h>
#endif


static void init(int seed, TYPE* dst, int nrows, int ncols, int ld, double scale) {
  const double seed1 = scale * seed + scale;
  int i;
  for (i = 0; i < ncols; ++i) {
    int j = 0;
    for (; j < nrows; ++j) {
      const int k = i * ld + j;
      dst[k] = (TYPE)(seed1 / (1.0 + k));
    }
    for (; j < ld; ++j) {
      const int k = i * ld + j;
      dst[k] = (TYPE)(seed);
    }
  }
}


static double norm(const TYPE* src, int nrows, int ncols, int ld) {
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


static double seconds(void) {
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

