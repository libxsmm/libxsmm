/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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
#ifndef LIBXSMM_TRANS_H
#define LIBXSMM_TRANS_H

#include <libxsmm.h>

#if !defined(LIBXSMM_TRANS_CHUNKSIZE)
# if defined(__MIC__)
#   define LIBXSMM_TRANS_CHUNKSIZE 8
# else
#   define LIBXSMM_TRANS_CHUNKSIZE 32
# endif
#endif
#if !defined(LIBXSMM_TRANS_TYPEOPT)
# define LIBXSMM_TRANS_TYPEOPT
#endif

#define LIBXSMM_OTRANS_GENERIC(TYPESIZE, OUT, IN, M0, M1, N0, N1, N, LD, LDO) { \
  const char *const a = (const char*)(IN); \
  char *const b = (char*)(OUT); \
  libxsmm_blasint i, j; \
  unsigned int k; \
  for (i = M0; i < (M1); ++i) { \
    LIBXSMM_PRAGMA_NONTEMPORAL \
    for (j = N0; j < (N1); ++j) { \
      const char *const aji = a + (TYPESIZE) * (j * (LD) + i); \
      char *const bij = b + (TYPESIZE) * (i * (LDO) + j); \
      for (k = 0; k < (TYPESIZE); ++k) { \
        bij[k] = aji[k]; \
      } \
    } \
  } \
}

#define LIBXSMM_OTRANS(TYPE, CHUNKSIZE, OUT, IN, M0, M1, N0, N1, N, LD, LDO) { \
  if (CHUNKSIZE == (N) && 0 == LIBXSMM_MOD2((uintptr_t)(IN), LIBXSMM_ALIGNMENT)) { \
    const TYPE *const a = (const TYPE*)(IN); \
    TYPE *const b = (TYPE*)(OUT); \
    libxsmm_blasint i, j; \
    for (i = M0; i < (M1); ++i) { \
      LIBXSMM_PRAGMA_VALIGNED_VARS(b) \
      LIBXSMM_PRAGMA_NONTEMPORAL \
      for (j = N0; j < (N0) + (CHUNKSIZE); ++j) { \
        /* use consecutive stores and strided loads */ \
        b[i*(LDO)+j] = a[j*(LD)+i]; \
      } \
    } \
  } \
  else { \
    LIBXSMM_OTRANS_GENERIC(sizeof(TYPE), OUT, IN, M0, M1, N0, N1, N, LD, LDO); \
  } \
}

#if defined(LIBXSMM_TRANS_TYPEOPT)
# define LIBXSMM_OTRANS_TYPEOPT_BEGIN(OUT, IN, TYPESIZE, CHUNKSIZE, M0, M1, N0, N1, N, LD, LDO) \
    switch(TYPESIZE) { \
      case 1: { \
        LIBXSMM_OTRANS(char, CHUNKSIZE, OUT, IN, M0, M1, N0, N1, n, LD, LDO); \
      } break; \
      case 2: { \
        LIBXSMM_OTRANS(short, CHUNKSIZE, OUT, IN, M0, M1, N0, N1, n, LD, LDO); \
      } break; \
      case 4: { \
        LIBXSMM_OTRANS(float, CHUNKSIZE, OUT, IN, M0, M1, N0, N1, n, LD, LDO); \
      } break; \
      case 8: { \
        LIBXSMM_OTRANS(double, CHUNKSIZE, OUT, IN, M0, M1, N0, N1, n, LD, LDO); \
      } break; \
      case 16: { \
        typedef struct dvec2_t { double value[2]; } dvec2_t; \
        LIBXSMM_OTRANS(dvec2_t, CHUNKSIZE, OUT, IN, M0, M1, N0, N1, n, LD, LDO); \
      } break; \
      default:
#else
# define LIBXSMM_OTRANS_TYPEOPT_BEGIN(OUT, IN, TYPESIZE, CHUNKSIZE, M0, M1, N0, N1, N, LD, LDO) {
#endif
#define LIBXSMM_OTRANS_TYPEOPT_END }

/**
 * Based on the cache-oblivious transpose by Frigo et.al. with some additional
 * optimization such as using a loop with bounds which are known at compile-time
 * due to splitting up tiles with one fixed-size extent (chunk).
 */
#define LIBXSMM_OTRANS_MAIN(KERNEL_START, SYNC, FN, OUT, IN, TYPESIZE, CHUNKSIZE, M0, M1, N0, N1, LD, LDO) { \
  const libxsmm_blasint m = (M1) - (M0), n = (N1) - (N0); \
  if (m * n * (TYPESIZE) <= ((LIBXSMM_CPU_DCACHESIZE) / 2)) { \
    LIBXSMM_OTRANS_TYPEOPT_BEGIN(OUT, IN, TYPESIZE, CHUNKSIZE, M0, M1, N0, N1, n, LD, LDO) \
    /* fall-back code path which is generic with respect to the typesize */ \
    LIBXSMM_OTRANS_GENERIC(TYPESIZE, OUT, IN, M0, M1, N0, N1, n, LD, LDO); \
    LIBXSMM_OTRANS_TYPEOPT_END \
  } \
  else if (m >= n) { \
    const libxsmm_blasint mi = ((M0) + (M1)) / 2; \
    KERNEL_START \
    (FN)(OUT, IN, TYPESIZE, M0, mi, N0, N1, LD, LDO); \
    KERNEL_START \
    (FN)(OUT, IN, TYPESIZE, mi, M1, N0, N1, LD, LDO); \
  } \
  else { \
    if ((CHUNKSIZE) < n) { \
      const libxsmm_blasint ni = (N0) + (CHUNKSIZE); \
      KERNEL_START \
      (FN)(OUT, IN, TYPESIZE, M0, M1, N0, ni, LD, LDO); \
      KERNEL_START \
      (FN)(OUT, IN, TYPESIZE, M0, M1, ni, N1, LD, LDO); \
    } \
    else \
    { \
      const libxsmm_blasint ni = ((N0) + (N1)) / 2; \
      KERNEL_START \
      (FN)(OUT, IN, TYPESIZE, M0, M1, N0, ni, LD, LDO); \
      KERNEL_START \
      (FN)(OUT, IN, TYPESIZE, M0, M1, ni, N1, LD, LDO); \
    } \
  } \
  SYNC \
}

#endif /*LIBXSMM_TRANS_H*/

