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


#include <omp.h>
#include "SoftmaxLossLoop.hpp"
#include "common.hpp"
#include <math.h>
#include <stdio.h>

void SMaxLossLoop::forwardPropagate(float *inp, int* label, float *outp)
{
  int nImg  = gp->batch_size;
  int nIfm = gp->nInput;
  int nOfm = gp->nOutput;

  int threads = gp->num_threads;

  __assume_aligned(inp,64);
  __assume_aligned(outp,64);

  float (* __restrict input )[nIfm] = (float (*)[*])inp;
  float (* __restrict output)[nOfm] = (float (*)[*])outp;

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads)
#endif
  for(int img = 0; img < nImg; img++) {

    float max = input[img][0];
    output[img][0] = input[img][0];

    for(int ofm = 1; ofm < nOfm; ofm++) {
      output[img][ofm] = input[img][ofm];
      if(input[img][ofm] > max)
        max = input[img][ofm];
    }

    float sum_of_exp = 0.0;
    for(int ofm = 0; ofm < nOfm; ofm++) {
      output[img][ofm] = output[img][ofm] - max;
      output[img][ofm] = exp(output[img][ofm]);
      sum_of_exp += output[img][ofm];
    }

    float recp_soe = 1.0/sum_of_exp;

    //Normalize each value by sum_of_exp
    for(int ofm = 0; ofm < nOfm; ofm++)
      output[img][ofm] = output[img][ofm]*recp_soe;
  }

  float loss = 0.0;

  for(int img = 0; img < nImg; img++) {
    float val = output[img][label[img]] > FLT_MIN ? output[img][label[img]] : FLT_MIN;
    loss += log(val);
  }

  gp->loss = -loss/nImg;
}

void SMaxLossLoop::backPropagate(float *outp, int* label, float *delinp)
{
  int nImg  = gp->batch_size;
  int nIfm = gp->nInput;

  int threads = gp->num_threads;

  __assume_aligned(delinp,64);

  float (* __restrict del_input )[nIfm] = (float (*)[*])delinp;
  float (* __restrict output )[nIfm] = (float (*)[*])outp;

#ifdef USE_MLSL
  float recp_mb = 1.0/(nImg * num_nodes);
#else
  float recp_mb = 1.0/nImg;
#endif

#ifdef _OPENMP
#pragma omp parallel for collapse(2) num_threads(threads)
#endif
  for(int img = 0; img < nImg; img++) {
    for(int fm = 0; fm < nIfm; fm++) {
      if(fm == label[img])
        del_input[img][fm] = (output[img][fm] - 1) * recp_mb * gp->loss_weight;
      else
        del_input[img][fm] = output[img][fm] * recp_mb * gp->loss_weight;
    }
  }
}

