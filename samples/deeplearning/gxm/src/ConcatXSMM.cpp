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
#include "ConcatXSMM.hpp"

#define VLEN 16

void ConcatXSMM::forwardPropagate(vector<TensorBuf*>& inpb, TensorBuf *outpb, int tid)
{
  float *outp = (float*)outpb->getBuffer();

  int nImg = gp->batch_size;
  int nOfm = gp->nOutput;
  int nBOfm = gp->nOutput/VLEN;
  int rem = 0;
  int ifh = gp->iHeight;
  int ifw = gp->iWidth;
  int ofh = gp->oHeight;
  int ofw = gp->oWidth;
  bool needs_conversion = false;
  int threads = gp->num_threads;

  __assume_aligned(outp, 64);

  float (* __restrict output)[nBOfm][ofh][ofw][VLEN] = (float (*)[*][*][*][VLEN])outp;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int img=0; img < nImg; img++) {
    int ofm = 0;
    for(int b=0; b<inpb.size(); b++) {
      int nBIfm = gp->nInput[b]/VLEN;
      float *inp __attribute__((aligned(64)));
      inp = (float*)inpb[b]->getBuffer();
      float (* __restrict input )[nBIfm][ifh][ifw][VLEN] = (float (*)[*][*][*][VLEN])inp;
      for(int ifm=0; ifm < nBIfm; ifm++) {
        for(int h=0; h < ifh; h++) {
          for(int w=0; w < ifw; w++) {
#pragma simd
#pragma vector aligned
#pragma vector nontemporal
            for(int v=0; v < VLEN; v++) {
              output[img][ofm][h][w][v] = input[img][ifm][h][w][v];
            }
          }
        }
        ofm++;
      }
    }
  }
  outpb->setLayoutType(LIBXSMM_CUSTOM_LAYOUT);
}

void ConcatXSMM::backPropagate(TensorBuf *deloutpb, vector<TensorBuf*>& delinpb, int tid)
{
  float *deloutp = (float*)deloutpb->getBuffer();

  int nImg = gp->batch_size;
  int nOfm = gp->nOutput;
  int nBOfm = gp->nOutput/VLEN;
  int ifh = gp->iHeight;
  int ifw = gp->iWidth;
  int ofh = gp->oHeight;
  int ofw = gp->oWidth;
  int rem = 0;

  int threads = gp->num_threads;

  __assume_aligned(deloutp, 64);

  float (* __restrict del_output)[nBOfm][ofh][ofw][VLEN] = (float (*)[*][*][*][VLEN])deloutp;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int img=0; img < nImg; img++) {
    int ofm = 0;
    for(int b=0; b<delinpb.size(); b++) {
      int nBIfm = gp->nInput[b]/VLEN;
      float *delinp __attribute__((aligned(64)));
      delinp = (float*)delinpb[b]->getBuffer();
      float (* __restrict del_input)[nBIfm][ifh][ifw][VLEN] = (float (*)[*][*][*][VLEN])delinp;
      for(int ifm=0; ifm < nBIfm; ifm++) {
        for(int h=0; h < ifh; h++) {
          for(int w=0; w < ifw; w++) {
#pragma simd
#pragma vector aligned
#pragma vector nontemporal
            for(int v=0; v < VLEN; v++) {
              del_input[img][ifm][h][w][v] = del_output[img][ofm][h][w][v];
            }
          }
        }
        ofm++;
      }
    }
  }

  for(int b=0; b<delinpb.size(); b++)
    delinpb[b]->setLayoutType(LIBXSMM_CUSTOM_LAYOUT);
}
