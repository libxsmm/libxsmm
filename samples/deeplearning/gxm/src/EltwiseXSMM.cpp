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
#include "EltwiseXSMM.hpp"

#define VLEN 16

void EltwiseXSMM::convert_NCHW_to_NCHWV(float *inp, int n, int c, int h, int w, float *outp)
{
  __assume_aligned(inp,64);
  __assume_aligned(outp,64);

  int index=0;
  int cv = c/VLEN;

  for(int img=0; img < n; img++)
    for(int fm=0; fm < cv; fm++)
      for(int fh=0; fh < h; fh++)
        for(int fw=0; fw < w; fw++)
          for(int v=0; v<VLEN; v++)
          {
            int ii = img*c*h*w + fm*h*w*VLEN + v*h*w + fh*w + fw;
            outp[index++] = inp[ii];
          }
}

void EltwiseXSMM::convert_NCHWV_to_NCHW(float *inp, int n, int c, int h, int w, float *outp)
{
  __assume_aligned(inp,64);
  __assume_aligned(outp,64);

  int index=0;
  int cv = c/VLEN;

  for(int img=0; img < n; img++)
    for(int fm=0; fm < cv; fm++)
      for(int fh=0; fh < h; fh++)
        for(int fw=0; fw < w; fw++)
          for(int v=0; v<VLEN; v++)
          {
            int oi = img*c*h*w + fm*h*w*VLEN + v*h*w + fh*w + fw;
            outp[oi] = inp[index++];
          }
}

void EltwiseXSMM::forwardPropagate(vector<TensorBuf*>& inpb, TensorBuf *outpb, int tid)
{
  float *outp = (float*)outpb->getBuffer();
  float *outpp = (float*)outpb->getPrivBuffer();

  float *inp_r = (float*)inpb[0]->getBuffer();
  float *inp_nr = (float*)inpb[1]->getBuffer();

  int nImg = gp->batch_size;
  int nOfm = gp->nOutput;
  int rem = 0;
  int ifh = gp->iHeight;
  int ifw = gp->iWidth;
  int ofh = gp->oHeight;
  int ofw = gp->oWidth;
  bool needs_conversion = false;
  int threads = gp->num_threads;

  int op = gp->op;

  __assume_aligned(outp, 64);
  __assume_aligned(outpp, 64);

  if(top_compute_engine != engine)
    needs_conversion = true;

  if(needs_conversion)
  {
    if(outpp == NULL) outpp = (float*)libxsmm_aligned_malloc(nImg*nOfm*ofh*ofw*sizeof(float), 64);
    assert(outpp != NULL);
    outpb->setPrivBuffer(outpp);
  }

  float* out = needs_conversion ? outpp : outp;

  for(int b=0; b<inpb.size(); b++)
    assert(bot_compute_engine[b] == engine);

  int size = nImg * nOfm * ofh *ofw;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0; i<size; i++)
    out[i] = 0;

  switch(op)
  {
    case ELSUM:
      {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i=0; i < size; i++)
          out[i] = inp_r[i] + inp_nr[i];
      }
      break;

    case ELPROD:
      break;

    case ELMAX:
      break;
  }

  if(needs_conversion)
  {
#if 0
#ifdef USE_MLSL
    if(MLSL::GetNodeId() == 0)
#endif
      printf("%s converting output buffer in forward prop\n",nname.c_str());
#endif

    convert_NCHWV_to_NCHW(outpp, nImg, nOfm, ofh, ofw, outp);
    outpb->setLayoutType(NCHW);
  }
  else
    outpb->setLayoutType(LIBXSMM_CUSTOM_LAYOUT);
}

#if 0
void EltwiseXSMM::backPropagate(TensorBuf *deloutpb, vector<TensorBuf*>& delinpb, int tid)
{
#if !defined(USE_OPTBP)
  float *deloutp = (float*)deloutpb->getBuffer();
  float *deloutpp = (float*)deloutpb->getPrivBuffer();
  float *delinp_r = (float*)delinpb[0]->getBuffer();
  float *delinp_nr = (float*)delinpb[1]->getBuffer();
#else
  float *deloutp = (float*)deloutpb->getGradBuffer();
  float *deloutpp = (float*)deloutpb->getGradPrivBuffer();
  float *delinp_r = (float*)delinpb[0]->getGradBuffer();
  float *delinp_nr = (float*)delinpb[1]->getGradBuffer();
  #ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
    printf("ELT delinp_r %p, delinp_nr %p\n",delinp_r, delinp_nr);
  #endif
#endif

  int nImg = gp->batch_size;
  int nOfm = gp->nOutput;
  int nIfm = gp->nOutput;
  int ifh = gp->iHeight;
  int ifw = gp->iWidth;
  int ofh = gp->oHeight;
  int ofw = gp->oWidth;

  int op = gp->op;

  int rem = 0;

  int threads = gp->num_threads;

  __assume_aligned(deloutp, 64);
  __assume_aligned(deloutpp, 64);

  if(top_compute_engine != engine)
  {
#if 0
#ifdef USE_MLSL
    if(MLSL::GetNodeId() == 0)
#endif
      printf("%s converting output buffer in forward prop\n",nname.c_str());
#endif

    if(deloutpp == NULL) deloutpp = (float*)libxsmm_aligned_malloc(nImg*nOfm*ofh*ofw*sizeof(float), 64);
    assert(deloutpp != NULL);
    deloutpb->setPrivBuffer(deloutpp);
    convert_NCHW_to_NCHWV(deloutpp, nImg, nOfm, ofh, ofw, deloutp);
  }

#ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
  {
    if((deloutp == delinp_r) || (deloutp == delinp_nr))
    {
      printf("node delout %p and delin %p are equal!!\n",deloutp, delinp_r, delinp_nr);
      fflush(stdout);
      exit(1);
    }
  }
#endif

  float *delout = deloutpp != NULL ? deloutpp : deloutp;

  for(int b=0; b<delinpb.size(); b++)
    assert(bot_compute_engine[b] == engine);

  int size = nImg * nIfm * ifh *ifw;

  switch(op)
  {
    case ELSUM:
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i < size; i++) {
        delinp_r[i] = delout[i];
        delinp_nr[i] = delout[i];
      }
    }
    break;

    case ELPROD:
      break;

    case ELMAX:
      break;
  }

  for(int b=0; b<delinpb.size(); b++)
    delinpb[b]->setLayoutType(LIBXSMM_CUSTOM_LAYOUT);
}
#endif

#if 1
void EltwiseXSMM::backPropagate(TensorBuf *deloutpb, vector<TensorBuf*>& delinpb, int tid)
{
  float *deloutp = (float*)deloutpb->getBuffer();

  int op = gp->op;

  switch(op)
  {
    case ELSUM:
    {
      for(int i=0; i<delinpb.size(); i++)
        delinpb[i]->setBuffer(deloutp);
    }
    break;

    case ELPROD:
      break;

    case ELMAX:
      break;
  }

  for(int b=0; b<delinpb.size(); b++)
    delinpb[b]->setLayoutType(LIBXSMM_CUSTOM_LAYOUT);
}
#endif
