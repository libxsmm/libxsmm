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


#pragma once
#include "FusedBNormImpl.hpp"
#include "check.hpp"
#include "libxsmm.h"

class FusedBNormXSMM : public FusedBNormImpl
{
  protected:
    float *bmeanp=NULL, *brstdp=NULL;
#if 0
    float *bstats_op=NULL;
    float *bstats2_op=NULL;
#endif
    float *del_gamma_imgp = NULL, *del_beta_imgp = NULL;
    float scaling_factor_=0;
    short *i16_outp=NULL, *i16_delinp_r=NULL, *i16_delinp_l=NULL;
  public:
    FusedBNormXSMM(FusedBNormImplParams* gp, int engine) : FusedBNormImpl(gp, engine)
    {
      top_layout_type = LIBXSMM_CUSTOM_LAYOUT;
      top_layout = NULL;
      gbot_layout_type = LIBXSMM_CUSTOM_LAYOUT;
      gbot_layout = NULL;
    }

    // Assume external threading, e.g., #pragma omp
    void forwardPropagate(vector<TensorBuf*> inp, TensorBuf* gammap, TensorBuf* betap, float *gmeanp, float *grstdp, TensorBuf *outp, int tid);
    void backPropagate(vector<TensorBuf*> inp, TensorBuf* outp, TensorBuf* gammap, TensorBuf *deloutp, TensorBuf *delgammap, TensorBuf *delbetap, vector<TensorBuf *> delinp, int tid);
};
