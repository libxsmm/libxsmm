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


#include <omp.h>
#include "SoftmaxLossLoop.hpp"
#include "common.hpp"
#include <math.h>
#include <stdio.h>

void SMaxLossLoop::forwardPropagate(TensorBuf* inpb, TensorBuf* labelb, TensorBuf* outpb)
{
  int nImg  = gp->batch_size;
  int nFM = gp->nInput;

  float* inp, *outp;
  int *label;

  label = (int*)labelb->getBuffer();
  inp = (float*)inpb->getBuffer();
  outp = (float*)outpb->getBuffer();

  float (* __restrict input)[nFM] = (float (*)[*])inp;
  float (* __restrict output)[nFM] = (float (*)[*])outp;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0; i<nImg; i++)
  {
    float max = FLT_MIN;

#pragma omp simd
    for(int fm = 0; fm < nFM; fm++)
    {
      output[i][fm] = input[i][fm];
      if(input[i][fm] > max)
        max = input[i][fm];
    }

    float sum_of_exp = 0.0;
#pragma omp simd reduction(+: sum_of_exp)
    for(int fm = 0; fm < nFM; fm++)
    {
      output[i][fm] = output[i][fm] - max;
      output[i][fm] = exp(output[i][fm]);
      sum_of_exp += output[i][fm];
    }

    float recp_soe = 1.0/sum_of_exp;

    //Normalize each value by sum_of_exp
#pragma omp simd
    for(int fm = 0; fm < nFM; fm++)
      output[i][fm] = output[i][fm]*recp_soe;
  }

  float loss = 0.0;

#pragma omp parallel for reduction(+: loss)
  for(int img = 0; img < nImg; img++)
  {
    float val = output[img][label[img]] > FLT_MIN ? output[img][label[img]] : FLT_MIN;
    loss += log(val);
  }

  gp->loss = -loss/nImg;
}

void SMaxLossLoop::backPropagate(TensorBuf *outpb, TensorBuf* labelb, TensorBuf *delinpb)
{
  int nImg  = gp->batch_size;
  int nFM = gp->nInput;

  float *outp, *delinp;
  int* label;

  label = (int*)labelb->getBuffer();
  delinp = (float*)delinpb->getBuffer();
  outp = (float*)outpb->getBuffer();

#ifdef USE_MLSL
  float recp_mb = 1.0/(nImg * num_nodes);
#else
  float recp_mb = 1.0/nImg;
#endif

  float (* __restrict output )[nFM] = (float (*)[*])outp;
  float (* __restrict del_input )[nFM] = (float (*)[*])delinp;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0; i<nImg; i++)
  {
#pragma omp simd
    for(int fm = 0; fm < nFM; fm++)
    {
      if(fm == label[i])
        del_input[i][fm] = (output[i][fm] - 1) * recp_mb * gp->loss_weight;
      else
        del_input[i][fm] = output[i][fm] * recp_mb * gp->loss_weight;
    }
  }
}

