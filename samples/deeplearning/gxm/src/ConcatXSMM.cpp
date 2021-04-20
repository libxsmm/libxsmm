/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
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
      LIBXSMM_ALIGNED(float *inp, 64);
      inp = (float*)inpb[b]->getBuffer();
      float (* __restrict input )[nBIfm][ifh][ifw][VLEN] = (float (*)[*][*][*][VLEN])inp;
      for(int ifm=0; ifm < nBIfm; ifm++) {
        for(int h=0; h < ifh; h++) {
          for(int w=0; w < ifw; w++) {
#pragma omp simd
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
      LIBXSMM_ALIGNED(float *delinp, 64);
      delinp = (float*)delinpb[b]->getBuffer();
      float (* __restrict del_input)[nBIfm][ifh][ifw][VLEN] = (float (*)[*][*][*][VLEN])delinp;
      for(int ifm=0; ifm < nBIfm; ifm++) {
        for(int h=0; h < ifh; h++) {
          for(int w=0; w < ifw; w++) {
#pragma omp simd
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
