/******************************************************************************
** Copyright (c) 2018, Intel Corporation                                     **
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
#ifndef MAGAZINE_H
#define MAGAZINE_H

#include <stdio.h>

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
#if 1
# define STREAM_C(EXPR) (EXPR)
#else
# define STREAM_C(EXPR) 0
#endif


void init(int seed, TYPE* dst, int nrows, int ncols, int ld, double scale) {
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


double norm(const TYPE* src, int nrows, int ncols, int ld) {
  int i, j;
  double result = 0, comp = 0;
  for (i = 0; i < ncols; ++i) {
    for (j = 0; j < nrows; ++j) {
      const int k = i * ld + j;
      const TYPE v = src[k], a = (0 <= v ? v : -v) - comp, b = result + a;
      comp = (b - result) - a;
      result = b;
    }
  }
  return result;
}

#endif /*MAGAZINE_H*/

