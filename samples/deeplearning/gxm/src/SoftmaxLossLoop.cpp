/******************************************************************************
** Copyright (c) 2017-2019, Intel Corporation                                **
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

void SMaxLossLoop::forwardPropagate(TensorBuf* inpb, TensorBuf* labelb, TensorBuf* outpb)
{
  int nImg  = gp->batch_size/NUM_NUMA_NODES;
  int nIfm = gp->nInput;
  int nOfm = gp->nOutput;

  float* inp[NUM_NUMA_NODES], *outp[NUM_NUMA_NODES];
  int *label[NUM_NUMA_NODES];

  label[0] = (int*)labelb->getBuffer();
  for(int n=1; n<NUM_NUMA_NODES; n++)
    label[n] = label[n-1] + nImg;


  inp[0] = (float*)inpb->getBuffer();
  for(int n=1; n<NUM_NUMA_NODES; n++)
    inp[n] = inp[n-1] + nImg*nIfm;

  outp[0] = (float*)outpb->getBuffer();
  for(int n=1; n<NUM_NUMA_NODES; n++)
    outp[n] = outp[n-1] + nImg*nOfm;

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = omp_get_thread_num();
    int ntps = gp->num_threads/NUM_NUMA_NODES;
    int n = tid/ntps;
    int ltid = tid - n*ntps;

    float (* __restrict input )[nIfm] = (float (*)[*])inp[n];
    float (* __restrict output)[nOfm] = (float (*)[*])outp[n];

    int jobs = (nImg % ntps == 0) ? nImg/ntps : (nImg/ntps) + 1;
    int tb = (ltid*jobs < nImg) ? ltid*jobs : nImg;
    int te = ((ltid+1)*jobs < nImg) ? (ltid+1)*jobs : nImg;

#pragma omp simd
    for(int i=tb; i<te; i++)
    {
      float max = input[i][0];
      output[i][0] = input[i][0];

      for(int ofm = 1; ofm < nOfm; ofm++) {
        output[i][ofm] = input[i][ofm];
        if(input[i][ofm] > max)
          max = input[i][ofm];
      }

      float sum_of_exp = 0.0;
      for(int ofm = 0; ofm < nOfm; ofm++) {
        output[i][ofm] = output[i][ofm] - max;
        output[i][ofm] = exp(output[i][ofm]);
        sum_of_exp += output[i][ofm];
      }

      float recp_soe = 1.0/sum_of_exp;

      //Normalize each value by sum_of_exp
      for(int ofm = 0; ofm < nOfm; ofm++)
        output[i][ofm] = output[i][ofm]*recp_soe;
    }
  }

  for(int n=0; n<NUM_NUMA_NODES; n++)
  {
    float (* __restrict output)[nOfm] = (float (*)[*])outp[n];
    int* lab = label[n];

    float loss = 0.0;

    for(int img = 0; img < nImg; img++)
    {
      float val = output[img][lab[img]] > FLT_MIN ? output[img][lab[img]] : FLT_MIN;
      loss += log(val);
    }

    gp->loss[n] = -loss/nImg;
  }
}

void SMaxLossLoop::backPropagate(TensorBuf *outpb, TensorBuf* labelb, TensorBuf *delinpb)
{
  int nImg  = gp->batch_size/NUM_NUMA_NODES;
  int nIfm = gp->nInput;

  float *outp[NUM_NUMA_NODES], *delinp[NUM_NUMA_NODES];
  int* label[NUM_NUMA_NODES];

  label[0] = (int*)labelb->getBuffer();
  for(int n=1; n<NUM_NUMA_NODES; n++)
    label[n] = label[n-1] + nImg;


  delinp[0] = (float*)delinpb->getBuffer();
  for(int n=1; n<NUM_NUMA_NODES; n++)
    delinp[n] = delinp[n-1] + nImg*nIfm;

  outp[0] = (float*)outpb->getBuffer();
  for(int n=1; n<NUM_NUMA_NODES; n++)
    outp[n] = outp[n-1] + nImg*nIfm;

#ifdef USE_MLSL
  float recp_mb = 1.0/(nImg * num_nodes);
#else
  float recp_mb = 1.0/nImg;
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = omp_get_thread_num();
    int ntps = gp->num_threads/NUM_NUMA_NODES;
    int n = tid/ntps;
    int ltid = tid - n*ntps;

    float (* __restrict output )[nIfm] = (float (*)[*])outp[n];
    float (* __restrict del_input )[nIfm] = (float (*)[*])delinp[n];
    int* lab = label[n];

    int jobs = (nImg % ntps == 0) ? nImg/ntps : (nImg/ntps) + 1;
    int tb = (ltid*jobs < nImg) ? ltid*jobs : nImg;
    int te = ((ltid+1)*jobs < nImg) ? (ltid+1)*jobs : nImg;

#pragma omp simd
    for(int i=tb; i<te; i++)
    {
      for(int fm = 0; fm < nIfm; fm++)
      {
        if(fm == lab[i])
          del_input[i][fm] = (output[i][fm] - 1) * recp_mb * gp->loss_weight;
        else
          del_input[i][fm] = output[i][fm] * recp_mb * gp->loss_weight;
      }
    }
  }
}

