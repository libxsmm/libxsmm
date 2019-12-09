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
#include <stdlib.h>
#include <omp.h>
#include <assert.h>
#include "ReLUXSMM.hpp"

void ReLUXSMM::forwardPropagate(TensorBuf *inpb, TensorBuf *outpb, int tid)
{
  float *inp = (float*)inpb->getBuffer();
  float *outp = (float*)outpb->getBuffer();

  int nImg  = gp->batch_size;
  int nOfm = gp->nOutput;
  int ofh = gp->oHeight;
  int ofw = gp->oWidth;

  __assume_aligned(inp,64);
  __assume_aligned(outp,64);

  int size = nImg * nOfm * ofh * ofw;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0; i<size; i++) {
    if(inp[i] < 0.0)
      outp[i] = 0.0;
    else
      outp[i] = inp[i];
  }

  outpb->setLayoutType(inpb->getLayoutType());
  outpb->setLayout(inpb->getLayout());
}

void ReLUXSMM::backPropagate(TensorBuf *inpb, TensorBuf *deloutpb, TensorBuf *delinpb, int tid)
{
  float *inp = (float*)inpb->getBuffer();
  float *deloutp = (float*)deloutpb->getBuffer();
  float *delinp = (float*)delinpb->getBuffer();

  int nImg  = gp->batch_size;
  int nOfm = gp->nOutput;
  int ofh = gp->oHeight;
  int ofw = gp->oWidth;
  int nIfm = gp->nInput;
  int ifh = gp->iHeight;
  int ifw = gp->iWidth;

  int threads = gp->num_threads;

  __assume_aligned(inp,64);
  __assume_aligned(delinp,64);
  __assume_aligned(deloutp,64);

  int size = nImg * nOfm * ofh * ofw;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0; i<size; i++) {
    if(inp[i] > 0.0)
      delinp[i] = deloutp[i];
    else
      delinp[i] = 0.0;
  }

  delinpb->setLayoutType(deloutpb->getLayoutType());
  delinpb->setLayout(deloutpb->getLayout());
}
