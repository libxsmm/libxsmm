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
#pragma simd
  for(int i=0; i<size; i++) {
    if(inp[i] < 0.0)
      outp[i] = 0.0;
    else
      outp[i] = inp[i];
  }

#ifdef DUMP_DATA
  string fname = gp->node_name + "_fp_in";
  FILE *f = fopen(fname.c_str(), "w");
  for(int i=0; i<size; i++)
    fprintf(f, "%g\n", inp[i]);
  fclose(f);

  fname = gp->node_name + "_fp_out";
  f = fopen(fname.c_str(), "w");
  for(int i=0; i<size; i++)
    fprintf(f, "%g\n", outp[i]);
  fclose(f);
#endif

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
#pragma simd
  for(int i=0; i<size; i++) {
    if(inp[i] > 0.0)
      delinp[i] = deloutp[i];
    else
      delinp[i] = 0.0;
    }

#ifdef DUMP_DATA
  string fname = gp->node_name + "_bp_delin";
  FILE *f = fopen(fname.c_str(), "w");
  for(int i=0; i<size; i++)
    fprintf(f, "%g\n", delinp[i]);
  fclose(f);

  fname = gp->node_name + "_bp_delout";
  f = fopen(fname.c_str(), "w");
  for(int i=0; i<size; i++)
    fprintf(f, "%g\n", deloutp[i]);
  fclose(f);
#endif

  delinpb->setLayoutType(deloutpb->getLayoutType());
  delinpb->setLayout(deloutpb->getLayout());
}
