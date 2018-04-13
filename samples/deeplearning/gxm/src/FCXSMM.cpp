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



#include "FCXSMM.hpp"
#include "cblas.h"

extern int iter;

void FCXSMM::forwardPropagate(TensorBuf *inpb, TensorBuf* weightpb, TensorBuf* biaspb, TensorBuf *outpb, int tid)
{
#ifdef RETURNALL
  return;
#endif


  assert(top_compute_engine != -1);
  assert(bot_compute_engine != -1);

  float *inp = (float*)inpb->getBuffer();

  float *weightp = (float*)weightpb->getBuffer();
  float *biasp;
  if(gp->bias_term)
    biasp = (float*)biaspb->getBuffer();
  float *outp = (float*)outpb->getBuffer();

  __assume_aligned(inp,64);
  __assume_aligned(weightp,64);
  __assume_aligned(biasp, 64);
  __assume_aligned(outp,64);

  int M = gp->batch_size;
  int K = gp->nInput;
  int N = gp->nOutput;

  int IH = gp->iHeight;
  int IW = gp->iWidth;
  int OH = gp->oHeight;
  int OW = gp->oWidth;

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N*OH*OW, K*IH*IW, (float)1., inp, K*IH*IW, weightp, K*IH*IW, (float)0., outp, N*OH*OW);

  if(gp->bias_term)
  {
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for(int img=0; img<M; img++)
      for(int ofm=0; ofm<N; ofm++)
        outp[img*M+ofm] += biasp[ofm];
  }

#ifdef DUMP_ACT_DATA
  string fname = gp->node_name + "_fp_out_" + to_string(iter);
  FILE *f = fopen(fname.c_str(), "w");
  for(int i=0; i<M*N*OH*OW; i++)
    fprintf(f, "%g\n", outp[i]);
  fclose(f);

  fname = gp->node_name + "_fp_in_" + to_string(iter);
  f = fopen(fname.c_str(), "w");
  for(int i=0; i<M*K*IH*IW; i++)
    fprintf(f, "%g\n", inp[i]);
  fclose(f);

  fname = gp->node_name + "_fp_wt_" + to_string(iter);
  f = fopen(fname.c_str(), "w");
  for(int i=0; i<N*OH*OW*K*IH*IW; i++)
    fprintf(f, "%g\n", weightp[i]);
  fclose(f);

  fname = gp->node_name + "_fp_bias_" + to_string(iter);
  f = fopen(fname.c_str(), "w");
  for(int i=0; i<N; i++)
    fprintf(f, "%g\n", biasp[i]);
  fclose(f);

#endif

}

void FCXSMM::backPropagate(TensorBuf *deloutpb, TensorBuf *weightpb, TensorBuf *delinpb, int tid)
{
#ifdef RETURNALL
  return;
#endif

  assert(top_compute_engine != -1);
  assert(bot_compute_engine != -1);

  float *deloutp = (float*)deloutpb->getBuffer();
  float *delinp = (float*)delinpb->getBuffer();
  float *weightp = (float*)weightpb->getBuffer();

  __assume_aligned(deloutp,64);
  __assume_aligned(weightp,64);
  __assume_aligned(delinp,64);

  int M = gp->batch_size;
  int K = gp->nInput;
  int N = gp->nOutput;

  int IH = gp->iHeight;
  int IW = gp->iWidth;
  int OH = gp->oHeight;
  int OW = gp->oWidth;

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, K*IH*IW, N*OH*OW, (float)1., deloutp, N*OH*OW, weightp, K*IH*IW, (float)0., delinp, K*IH*IW);

#ifdef DUMP_ACT_DATA
  string fname = gp->node_name + "_bp_delout_" + to_string(iter);
  FILE *f = fopen(fname.c_str(), "w");
  for(int i=0; i<M*N*OH*OW; i++)
    fprintf(f, "%10g\n", deloutp[i]);
  fclose(f);

  fname = gp->node_name + "_bp_wt_" + to_string(iter);
  f = fopen(fname.c_str(), "w");
  for(int i=0; i<N*OH*OW*K*IH*IW; i++)
    fprintf(f, "%g\n", weightp[i]);
  fclose(f);

  fname = gp->node_name + "_bp_delin_" + to_string(iter);
  f = fopen(fname.c_str(), "w");
  for(int i=0; i<M*K*IH*IW; i++)
    fprintf(f, "%g\n", delinp[i]);
  fclose(f);

#endif
}

void FCXSMM::weightUpdate(TensorBuf *deloutpb, TensorBuf *inpb, TensorBuf *delweightpb, TensorBuf *delbiaspb, int tid)
{
#ifdef RETURNALL
  return;
#endif

  assert(top_compute_engine != -1);
  assert(bot_compute_engine != -1);

  float *inp = (float*)inpb->getBuffer();
  float *deloutp = (float*)deloutpb->getBuffer();
  float *delweightp = (float*)delweightpb->getBuffer();
  float *delbiasp;
  if(gp->bias_term)
    delbiasp = (float*)delbiaspb->getBuffer();

  __assume_aligned(deloutp,64);
  __assume_aligned(inp,64);
  __assume_aligned(delweightp,64);
  __assume_aligned(delbiasp, 64);

  int M = gp->batch_size;
  int K = gp->nInput;
  int N = gp->nOutput;

  int IH = gp->iHeight;
  int IW = gp->iWidth;
  int OH = gp->oHeight;
  int OW = gp->oWidth;

  if(gp->bias_term)
  {
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for(int ofm=0; ofm<N; ofm++) {
      for(int img=0; img<M; img++)
        delbiasp[ofm] += deloutp[img*N+ofm];
    }
  }

  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N*OH*OW, K*IH*IW, M, (float)1., deloutp, N*OH*OW, inp, K*IH*IW, (float)0., delweightp, K*IH*IW);

#ifdef DUMP_ACT_DATA
  string fname = gp->node_name + "_wu_delout_" + to_string(iter);
  FILE *f = fopen(fname.c_str(), "w");
  for(int i=0; i<M*N*OH*OW; i++)
    fprintf(f, "%g\n", deloutp[i]);
  fclose(f);

  fname = gp->node_name + "_wu_in_" + to_string(iter);
  f = fopen(fname.c_str(), "w");
  for(int i=0; i<M*K*IH*IW; i++)
    fprintf(f, "%g\n", inp[i]);
  fclose(f);
#endif

#ifdef DUMP_WT_DATA
  string fname = gp->node_name + "_wu_delwt_" + to_string(iter);
  FILE* f = fopen(fname.c_str(), "w");
  for(int i=0; i<N*OH*OW*K*IH*IW; i++)
    fprintf(f, "%g\n", delweightp[i]);
  fclose(f);

  fname = gp->node_name + "_wu_delbias_" + to_string(iter);
  f = fopen(fname.c_str(), "w");
  for(int i=0; i<N; i++)
    fprintf(f, "%g\n", delbiasp[i]);
  fclose(f);
#endif
}
