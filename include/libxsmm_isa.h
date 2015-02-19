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

#define MM_PREFETCH_L1(A) \
  _mm_prefetch(A, _MM_HINT_T0)
#define MM_PREFETCH_L2(A) \
  _mm_prefetch(A, _MM_HINT_T0)
#define MM_PREFETCH_L3(A) \
  _mm_prefetch(A, _MM_HINT_T0)
#define MM_PREFETCH_NT(A) \
  MM_PREFETCH_L2(A)

#define MM512_SET1_PD(V) \
  _mm512_set1_pd(V)
#define MM512_FMADD_PD(U, V, W) \
  _mm512_fmadd_pd(U, V, W)
#define MM512_FMADD_MASK_PD(U, V, W, MASK) \
  _mm512_mask3_fmadd_pd(U, V, W, MASK)

#define MM512_LOAD_PD(A, HINT) \
  _mm512_load_pd(A)
#define MM512_LOAD_MASK_PD(A, MASK, HINT) \
  _mm512_maskz_load_pd(MASK, A)
#define MM512_LOADU_PD(A, HINT) \
  _mm512_loadu_pd(A)
#define MM512_LOADU_MASK_PD(A, MASK, HINT) \
  _mm512_maskz_loadu_pd(MASK, A)

#define MM512_STORE_PD(A, V, HINT) \
  _mm512_store_pd(A, V)
#define MM512_STORENRNGO_PD(A, V) \
  _mm512_stream_pd(A, V)
#define MM512_STORE_MASK_PD(A, V, MASK, HINT) \
  _mm512_mask_store_pd(A, MASK, V)
#define MM512_STOREU_PD(A, V, HINT) \
  _mm512_storeu_pd(A, V)
#define MM512_STOREU_MASK_PD(A, V, MASK, HINT) \
  _mm512_mask_storeu_pd(A, MASK, V)

#elif defined(__MIC__)

#define MM_PREFETCH_L1(A) \
  _mm_prefetch(A, _MM_HINT_T0)
#define MM_PREFETCH_L2(A) \
  _mm_prefetch(A, _MM_HINT_T0)
#define MM_PREFETCH_L3(A) \
  _mm_prefetch(A, _MM_HINT_T0)
#define MM_PREFETCH_NT(A) \
  MM_PREFETCH_L2(A)

LIBXSMM_INLINE __m512d MM512_GET_PD() {
  __m512d value; return value;
}
#define MM512_SET1_PD(V) \
  _mm512_set1_pd(V)
#define MM512_FMADD_PD(U, V, W) \
  _mm512_fmadd_pd(U, V, W)
#define MM512_FMADD_MASK_PD(U, V, W, MASK) \
  _mm512_mask3_fmadd_pd(U, V, W, MASK)

#define MM512_LOAD_PD(A, HINT) \
  _mm512_extload_pd(A, _MM_UPCONV_PD_NONE, _MM_BROADCAST64_NONE, HINT)
#define MM512_LOAD_MASK_PD(A, MASK, HINT) \
  _mm512_mask_extload_pd(_mm512_setzero_pd(), MASK, A, _MM_UPCONV_PD_NONE, _MM_BROADCAST64_NONE, HINT)
#define MM512_LOADU_PD(A, HINT) \
  _mm512_extloadunpackhi_pd( \
    _mm512_extloadunpacklo_pd(MM512_GET_PD(), A, _MM_UPCONV_PD_NONE, HINT), \
    (A) + 8, _MM_UPCONV_PD_NONE, HINT)
#define MM512_LOADU_MASK_PD(A, MASK, HINT) \
  _mm512_mask_extloadunpackhi_pd( \
    _mm512_mask_extloadunpacklo_pd(_mm512_setzero_pd(), MASK, A, _MM_UPCONV_PD_NONE, HINT), \
    MASK, (A) + 8, _MM_UPCONV_PD_NONE, HINT)

#define MM512_STORE_PD(A, V, HINT) \
  _mm512_extstore_pd(A, V, _MM_DOWNCONV_PD_NONE, HINT)
#define MM512_STORENRNGO_PD(A, V) \
  _mm512_storenrngo_pd(A, V)
#define MM512_STORE_MASK_PD(A, V, MASK, HINT) \
  _mm512_mask_extstore_pd(A, MASK, V, _MM_DOWNCONV_PD_NONE, HINT)
#define MM512_STOREU_PD(A, V, HINT) \
  _mm512_extpackstorelo_pd(A, V, _MM_DOWNCONV_PD_NONE, HINT); \
  _mm512_extpackstorehi_pd((A) + 8, V, _MM_DOWNCONV_PD_NONE, HINT)
#define MM512_STOREU_MASK_PD(A, V, MASK, HINT) \
  _mm512_mask_extpackstorelo_pd(A, MASK, V, _MM_DOWNCONV_PD_NONE, HINT); \
  _mm512_mask_extpackstorehi_pd((A) + 8, MASK, V, _MM_DOWNCONV_PD_NONE, HINT)

#endif // __MIC__
#endif // LIBXSMM_ISA_H
