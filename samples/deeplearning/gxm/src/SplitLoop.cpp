/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
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
/* Sasikanth Avancha, Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/


#include <stdio.h>
#include <omp.h>
#include <immintrin.h>
#include "SplitLoop.hpp"

# define _mm512_load_act(A)     _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(A))),16))
#if 1
__m512i vnaninf = _mm512_set1_epi32( 0x7f800000 );
__m512i vrneadd = _mm512_set1_epi32( 0x00007fff );
__m512i vfixup = _mm512_set1_epi32( 0x00000001 );
__m512i vfixupmask = _mm512_set1_epi32( 0x00010000 );
# define _mm512_roundbf16rne(A) _mm512_mask_add_epi32( _mm512_castps_si512( A ), _mm512_cmp_epi32_mask( _mm512_and_epi32( _mm512_castps_si512( A ), vnaninf ), vnaninf, _MM_CMPINT_NE ), _mm512_castps_si512( A ), _mm512_mask_add_epi32( vrneadd , _mm512_cmp_epi32_mask( _mm512_and_epi32( _mm512_castps_si512( A ), vfixupmask ), vfixupmask, _MM_CMPINT_EQ ), vrneadd, vfixup ) )
# define _mm512_stream_act(A,B) _mm256_stream_si256((__m256i*)A,_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_roundbf16rne((B)),16)))
# define _mm512_store_act(A,B)  _mm256_storeu_si256((__m256i*)A,_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_roundbf16rne((B)),16)))
#else
# define _mm512_stream_act(A,B) _mm256_stream_si256((__m256i*)A,_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_castps_si512((B)),16)))
# define _mm512_store_act(A,B)  _mm256_storeu_si256((__m256i*)A,_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_castps_si512((B)),16)))
#endif

#define VLEN 16

void SplitLoop::forwardPropagate(TensorBuf *inpb, vector<TensorBuf*>& outpb, int tid)
{
  for(int i=0; i<outpb.size(); i++)
  {
    outpb[i]->setBuffer(inpb->getBuffer());
    outpb[i]->setBufferSize(inpb->getBufferSize());
    outpb[i]->setLayoutType(inpb->getLayoutType());
  }
}

void SplitLoop::backPropagate(vector<TensorBuf *>& deloutpb, TensorBuf *delinpb, int tid)
{
  assert(gp->bdims == gp->tdims);

  int nImg = gp->batch_size;
  int nIfm = gp->nInput;
  int ifh = gp->iHeight;
  int ifw = gp->iWidth;

  int in_dtype = delinpb->getDataType();
  int out_dtype = deloutpb[0]->getDataType();

  void* delinp = delinpb->getBuffer();

  void *deloutp[deloutpb.size()];
  int num_outp = 1;
  int size = nImg*nIfm*ifh*ifw;

  deloutp[0] = deloutpb[0]->getBuffer();

  for(int i=1; i<deloutpb.size(); i++)
  {
    if(deloutpb[i] == NULL) continue;

    deloutp[num_outp] = deloutpb[i]->getBuffer();
    num_outp++;
  }

  if(in_dtype == DT_FLOAT && out_dtype == DT_FLOAT)
  {
#ifdef __AVX512F__
    if (size % 16 == 0) {
      if ( num_outp == 2 ) {
        float* out1 = (float*)deloutp[0];
        float* out2 = (float*)deloutp[1];
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int j=0; j<size; j+=16) {
          __m512 vo = _mm512_load_ps( out1+j );
          vo = _mm512_add_ps( vo, _mm512_load_ps( out2+j ) );
#ifdef USE_NTS_SPLIT
          _mm512_stream_ps( &(((float*)delinp)[j]), vo );
#else
          _mm512_store_ps( &(((float*)delinp)[j]), vo );
#endif
        }
      } else if ( num_outp == 1 ) {
        float* out1 = (float*)deloutp[0];
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int j=0; j<size; j+=16) {
          __m512 vo = _mm512_load_ps( out1+j );
#ifdef USE_NTS_SPLIT
          _mm512_stream_ps( &(((float*)delinp)[j]), vo );
#else
          _mm512_store_ps( &(((float*)delinp)[j]), vo );
#endif
        }
      } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int j=0; j<size; j+=16) {
          __m512 vo = _mm512_load_ps( &(((float*)deloutp[0])[j]) );
          for(int i=1; i<num_outp; i++) {
            vo = _mm512_add_ps( vo, _mm512_load_ps( &(((float*)deloutp[i])[j]) ) );
          }
#ifdef USE_NTS_SPLIT
          _mm512_stream_ps( &(((float*)delinp)[j]), vo );
#else
          _mm512_store_ps( &(((float*)delinp)[j]), vo );
#endif
        }
      }
    } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int j=0; j<size; j++) {
        float o = ((float*)deloutp[0])[j];
        for(int i=1; i<num_outp; i++) {
          o += ((float*)deloutp[i])[j];
        }
        ((float*)delinp)[j] = o;
      }
    }
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int j=0; j<size; j++) {
      float o = ((float*)deloutp[0])[j];
      for(int i=1; i<num_outp; i++) {
        o += ((float*)deloutp[i])[j];
      }
      delinp[j] = o;
    }
#endif
  }
  else if(in_dtype == DT_BF16 && out_dtype == DT_BF16)
  {
#ifdef __AVX512F__
    if (size % 16 == 0) {
      if ( num_outp == 2 ) {
        libxsmm_bfloat16* out1 = (libxsmm_bfloat16*)deloutp[0];
        libxsmm_bfloat16* out2 = (libxsmm_bfloat16*)deloutp[1];
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int j=0; j<size; j+=16) {
          __m512 vo = _mm512_load_act( out1+j );
          vo = _mm512_add_ps( vo, _mm512_load_act( out2+j ) );
#ifdef USE_NTS_SPLIT
          _mm512_stream_act( &(((libxsmm_bfloat16*)delinp)[j]), vo );
#else
          _mm512_store_act( &(((libxsmm_bfloat16*)delinp)[j]), vo );
#endif
        }
      } else if ( num_outp == 1 ) {
        libxsmm_bfloat16* out1 = (libxsmm_bfloat16*)deloutp[0];
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int j=0; j<size; j+=16) {
          __m512 vo = _mm512_load_act( out1+j );
#ifdef USE_NTS_SPLIT
          _mm512_stream_act( &(((libxsmm_bfloat16*)delinp)[j]), vo );
#else
          _mm512_store_act( &(((libxsmm_bfloat16*)delinp)[j]), vo );
#endif
        }
      } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int j=0; j<size; j+=16) {
          __m512 vo = _mm512_load_act( &(((libxsmm_bfloat16*)deloutp[0])[j]) );
          for(int i=1; i<num_outp; i++) {
            vo = _mm512_add_ps( vo, _mm512_load_act( &(((libxsmm_bfloat16*)deloutp[i])[j]) ) );
          }
#ifdef USE_NTS_SPLIT
          _mm512_stream_act( &(((libxsmm_bfloat16*)delinp)[j]), vo );
#else
          _mm512_store_act( &(((libxsmm_bfloat16*)delinp)[j]), vo );
#endif
        }
      }
    } else {
#if defined(_OPENMP)
#pragma omp parallel
#endif
      {
        union libxsmm_bfloat16_hp deloutput_32_0, deloutput_32_1;

        deloutput_32_0.i[0] = 0;
        deloutput_32_0.i[1] = 0;
        deloutput_32_1.i[0] = 0;
        deloutput_32_1.i[1] = 0;

#if defined(_OPENMP)
#pragma omp for
#endif
        for(int j=0; j<size; j++) {
          deloutput_32_0.i[1] = ((libxsmm_bfloat16*)deloutp[0])[j];
          for(int i=1; i<num_outp; i++) {
            deloutput_32_1.i[1] = ((libxsmm_bfloat16*)deloutp[i])[j];
            deloutput_32_0.f += deloutput_32_1.f;
          }
          ((libxsmm_bfloat16*)delinp)[j] = deloutput_32_0.i[1];
          deloutput_32_0.i[0] = 0;
          deloutput_32_0.i[1] = 0;
        }
      }
    }
#else
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
      union libxsmm_bfloat16_hp deloutput_32_0, deloutput_32_1;

      deloutput_32_0.i[0] = 0;
      deloutput_32_0.i[1] = 0;
      deloutput_32_1.i[0] = 0;
      deloutput_32_1.i[1] = 0;

#if defined(_OPENMP)
#pragma omp for
#endif
      for(int j=0; j<size; j++) {
        deloutput_32_0.i[1] = ((libxsmm_bfloat16*)deloutp[0])[j];
        for(int i=1; i<num_outp; i++) {
          deloutput_32_1.i[1] = ((libxsmm_bfloat16*)deloutp[i])[j];
          deloutput_32_0.f += deloutput_32_1.f;
        }
        ((libxsmm_bfloat16*)delinp)[j] = deloutput_32_0.i[1];
        deloutput_32_0.i[0] = 0;
        deloutput_32_0.i[1] = 0;
      }
    }
#endif
  }

  delinpb->setLayoutType(deloutpb[0]->getLayoutType());
}
