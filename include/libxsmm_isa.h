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
#ifndef LIBXSMM_ISA_H
#define LIBXSMM_ISA_H

#include "libxsmm.h"
#include <immintrin.h>


#if defined(__AVX512F__)

#if !defined(_MM_HINT_NONE)
# define _MM_HINT_NONE 0
#endif
#if !defined(_MM_HINT_NT)
# define _MM_HINT_NT 1
#endif

LIBXSMM_INLINE __m512d MM512_SET1_PD(double value)
{
  return _mm512_set1_pd(value);
}

LIBXSMM_INLINE __m512d MM512_FMADD_PD(__m512d u, __m512d v, __m512d w)
{
  return _mm512_fmadd_pd(u, v, w);
}

LIBXSMM_INLINE __m512d MM512_FMADD_MASK_PD(__m512d u, __m512d v, __m512d w, __mmask8 mask)
{
  return _mm512_mask3_fmadd_pd(u, v, w, mask);
}

LIBXSMM_INLINE __m512d MM512_LOAD_PD(const double* a, int hint)
{
  return _mm512_load_pd(a); // no hint
}

LIBXSMM_INLINE __m512d MM512_LOAD_MASK_PD(const double* a, __mmask8 mask, int hint)
{
  return _mm512_maskz_load_pd(mask, a); // no hint
}

LIBXSMM_INLINE __m512d MM512_LOADU_PD(const double* a, int hint)
{
  return _mm512_loadu_pd(a); // no hint
}

LIBXSMM_INLINE __m512d MM512_LOADU_MASK_PD(const double* a, __mmask8 mask, int hint)
{
  return _mm512_maskz_loadu_pd(mask, a); // no hint
}

LIBXSMM_INLINE void MM512_STORE_PD(double* a, __m512d v, int hint)
{
  _mm512_store_pd(a, v); // no hint
}

LIBXSMM_INLINE void MM512_STORENRNGO_PD(double* a, __m512d v)
{
  _mm512_stream_pd(a, v);
}

LIBXSMM_INLINE void MM512_STORE_MASK_PD(double* a, __m512d v, __mmask8 mask, int hint)
{
  _mm512_mask_store_pd(a, mask, v); // no hint
}

LIBXSMM_INLINE void MM512_STOREU_PD(double* a, __m512d v, int hint)
{
  _mm512_storeu_pd(a, v); // no hint
}

LIBXSMM_INLINE void MM512_STOREU_MASK_PD(double* a, __m512d v, __mmask8 mask, int hint)
{
  _mm512_mask_storeu_pd(a, mask, v); // no hint
}

#elif defined(__MIC__)

LIBXSMM_INLINE __m512d MM512_SET1_PD(double value)
{
  return _mm512_set1_pd(value);
}

LIBXSMM_INLINE __m512d MM512_FMADD_PD(__m512d u, __m512d v, __m512d w)
{
  return _mm512_fmadd_pd(u, v, w);
}

LIBXSMM_INLINE __m512d MM512_FMADD_MASK_PD(__m512d u, __m512d v, __m512d w, __mmask8 mask)
{
  return _mm512_mask3_fmadd_pd(u, v, w, mask);
}

LIBXSMM_INLINE __m512d MM512_LOAD_PD(const double* a, int hint)
{
  return _mm512_extload_pd(a, _MM_UPCONV_PD_NONE, _MM_BROADCAST64_NONE, hint);
}

LIBXSMM_INLINE __m512d MM512_LOAD_MASK_PD(const double* a, __mmask8 mask, int hint)
{
  __m512d va = _mm512_setzero_pd();
  va = _mm512_mask_extload_pd(va, mask, a, _MM_UPCONV_PD_NONE, _MM_BROADCAST64_NONE, hint);
  return va;
}

LIBXSMM_INLINE __m512d MM512_LOADU_PD(const double* a, int hint)
{
  __m512d va;
  va = _mm512_extloadunpacklo_pd(va, &a[0], _MM_UPCONV_PD_NONE, hint);
  va = _mm512_extloadunpackhi_pd(va, &a[8], _MM_UPCONV_PD_NONE, hint);
  return va;
}

LIBXSMM_INLINE __m512d MM512_LOADU_MASK_PD(const double* a, __mmask8 mask, int hint)
{
  __m512d va = _mm512_setzero_pd();
  va = _mm512_mask_extloadunpacklo_pd(va, mask, &a[0], _MM_UPCONV_PD_NONE, hint);
  va = _mm512_mask_extloadunpackhi_pd(va, mask, &a[8], _MM_UPCONV_PD_NONE, hint);
  return va;
}

LIBXSMM_INLINE void MM512_STORE_PD(double* a, __m512d v, int hint)
{
  _mm512_extstore_pd(a, v, _MM_DOWNCONV_PD_NONE, hint);
}

LIBXSMM_INLINE void MM512_STORENRNGO_PD(double* a, __m512d v)
{
  _mm512_storenrngo_pd(a, v);
}

LIBXSMM_INLINE void MM512_STORE_MASK_PD(double* a, __m512d v, __mmask8 mask, int hint)
{
  _mm512_mask_extstore_pd(a, mask, v, _MM_DOWNCONV_PD_NONE, hint);
}

LIBXSMM_INLINE void MM512_STOREU_PD(double* a, __m512d v, int hint)
{
  _mm512_extpackstorelo_pd(&a[0], v, _MM_DOWNCONV_PD_NONE, hint);
  _mm512_extpackstorehi_pd(&a[8], v, _MM_DOWNCONV_PD_NONE, hint);
}

LIBXSMM_INLINE void MM512_STOREU_MASK_PD(double* a, __m512d v, __mmask8 mask, int hint)
{
  _mm512_mask_extpackstorelo_pd(&a[0], mask, v, _MM_DOWNCONV_PD_NONE, hint);
  _mm512_mask_extpackstorehi_pd(&a[8], mask, v, _MM_DOWNCONV_PD_NONE, hint);
}

#endif // __MIC__
#endif // LIBXSMM_ISA_H
