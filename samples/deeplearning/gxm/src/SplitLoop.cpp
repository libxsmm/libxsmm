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

#define VLEN 16

void SplitLoop::forwardPropagate(TensorBuf *inpb, vector<TensorBuf*>& outpb, int tid)
{
  int in_dtype = inpb->getDataType();

  for(int i=0; i<outpb.size(); i++)
  {
    outpb[i]->setBuffer(inpb->getBuffer());
    if(in_dtype == DT_DFP16)
    {
      outpb[i]->setLPBuffer(inpb->getLPBuffer());
      outpb[i]->setLPSF(inpb->getLPSF());
    }
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

  // TODO: Handle DFP16 type if/when required.
  int dtype = delinpb->getDataType();

  float* delinp = (float*)delinpb->getBuffer();

  float *deloutp[deloutpb.size()];
  int num_outp = 1;
  int size = nImg*nIfm*ifh*ifw;

  deloutp[0] = (float*)(deloutpb[0]->getBuffer());

  for(int i=1; i<deloutpb.size(); i++)
  {
    if(deloutpb[i] == NULL) continue;

    deloutp[num_outp] = (float*)(deloutpb[i]->getBuffer());
    num_outp++;
  }

#ifdef __AVX512F__
  if (size % 16 == 0) {
    if ( num_outp == 2 ) {
      float* out1 = deloutp[0];
      float* out2 = deloutp[1];
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int j=0; j<size; j+=16) {
        __m512 vo = _mm512_load_ps( out1+j );
        vo = _mm512_add_ps( vo, _mm512_load_ps( out2+j ) );
#ifdef USE_NTS_SPLIT
        _mm512_stream_ps( &(delinp[j]), vo );
#else
        _mm512_store_ps( &(delinp[j]), vo );
#endif
      }
    } else if ( num_outp == 1 ) {
      float* out1 = deloutp[0];
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int j=0; j<size; j+=16) {
        __m512 vo = _mm512_load_ps( out1+j );
#ifdef USE_NTS_SPLIT
        _mm512_stream_ps( &(delinp[j]), vo );
#else
        _mm512_store_ps( &(delinp[j]), vo );
#endif
      }
    } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int j=0; j<size; j+=16) {
        __m512 vo = _mm512_load_ps( &((deloutp[0])[j]) );
        for(int i=1; i<num_outp; i++) {
          vo = _mm512_add_ps( vo, _mm512_load_ps( &((deloutp[i])[j]) ) );
        }
#ifdef USE_NTS_SPLIT
        _mm512_stream_ps( &(delinp[j]), vo );
#else
        _mm512_store_ps( &(delinp[j]), vo );
#endif
      }
    }
  } else {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int j=0; j<size; j++) {
      float o = (deloutp[0])[j];
      for(int i=1; i<num_outp; i++) {
        o += (deloutp[i])[j];
      }
      delinp[j] = o;
    }
  }
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int j=0; j<size; j++) {
    float o = (deloutp[0])[j];
    for(int i=1; i<num_outp; i++) {
      o += (deloutp[i])[j];
    }
    delinp[j] = o;
  }
#endif

  delinpb->setLayoutType(deloutpb[0]->getLayoutType());
}
