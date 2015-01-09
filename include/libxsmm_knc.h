/******************************************************************************
** Copyright (c) 2013-2015, Intel Corporation                                **
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
/* Christopher Dahnken (Intel Corp.), Hans Pabst (Intel Corp.),
 * Alfio Lazzaro (CRAY Inc.), and Gilles Fourestey (CSCS)
******************************************************************************/
#ifndef LIBXSMM_KNC_H
#define LIBXSMM_KNC_H

#include "libxsmm.h"
#include <immintrin.h>


#ifdef __MIC__

LIBXSMM_INLINE __m512d MM512_LOADU_PD(const double* a)
{
  __m512d va;
  va = _mm512_extloadunpacklo_pd(va, &a[0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  va = _mm512_extloadunpackhi_pd(va, &a[8], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  return va;
}


LIBXSMM_INLINE __m512d MM512_LOADNTU_PD(const double* a)
{
  __m512d va;
  va = _mm512_extloadunpacklo_pd(va, &a[0], _MM_UPCONV_PD_NONE, _MM_HINT_NT);
  va = _mm512_extloadunpackhi_pd(va, &a[8], _MM_UPCONV_PD_NONE, _MM_HINT_NT);
  return va;
}


LIBXSMM_INLINE __m512d MM512_MASK_LOADU_PD(const double* a, char mask)
{
  __m512d va = _mm512_setzero_pd();
  va = _mm512_mask_extloadunpacklo_pd(va, mask, &a[0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  va = _mm512_mask_extloadunpackhi_pd(va, mask, &a[8], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
  return va;
}


LIBXSMM_INLINE __m512d MM512_MASK_LOADNTU_PD(const double* a, char mask)
{
  __m512d va = _mm512_setzero_pd();
  va = _mm512_mask_extloadunpacklo_pd(va, mask, &a[0], _MM_UPCONV_PD_NONE, _MM_HINT_NT);
  va = _mm512_mask_extloadunpackhi_pd(va, mask, &a[8], _MM_UPCONV_PD_NONE, _MM_HINT_NT);
  return va;
}


LIBXSMM_INLINE void MM512_STOREU_PD(double* a, __m512d v)
{
  _mm512_extpackstorelo_pd(&a[0], v, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
  _mm512_extpackstorehi_pd(&a[8], v, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
}


LIBXSMM_INLINE void MM512_STORENTU_PD(double* a, __m512d v)
{
  _mm512_extpackstorelo_pd(&a[0], v, _MM_DOWNCONV_PD_NONE, _MM_HINT_NT);
  _mm512_extpackstorehi_pd(&a[8], v, _MM_DOWNCONV_PD_NONE, _MM_HINT_NT);
}


LIBXSMM_INLINE void MM512_MASK_STOREU_PD(double* a, __m512d v, char mask)
{
  _mm512_mask_extpackstorelo_pd(&a[0], mask, v, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
  _mm512_mask_extpackstorehi_pd(&a[8], mask, v, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
}


LIBXSMM_INLINE void MM512_MASK_STORENTU_PD(double* a, __m512d v, char mask)
{
  _mm512_mask_extpackstorelo_pd(&a[0], mask, v, _MM_DOWNCONV_PD_NONE, _MM_HINT_NT);
  _mm512_mask_extpackstorehi_pd(&a[8], mask, v, _MM_DOWNCONV_PD_NONE, _MM_HINT_NT);
}

#endif // __MIC__
#endif // LIBXSMM_KNC_H
