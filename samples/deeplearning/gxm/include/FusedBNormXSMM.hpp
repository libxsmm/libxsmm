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

#define CHKERR_LIBXSMM_DNN(A) if ( A != LIBXSMM_DNN_SUCCESS )\
{\
  fprintf(stdout, "%s, %s\n", gp->node_name.c_str(), libxsmm_dnn_get_error(A) );\
  fflush(stdout);\
}
class FusedBNormXSMM : public FusedBNormImpl
{
  protected:
    FusedBNormImpl *gp_;
    libxsmm_dnn_fusedbn_desc fusedbn_desc;
    libxsmm_dnn_fusedbn* libxsmm_handle = NULL;
    libxsmm_dnn_tensor* libxsmm_input = NULL;
    libxsmm_dnn_tensor* libxsmm_input_add = NULL;
    libxsmm_dnn_tensor* libxsmm_output = NULL;
    libxsmm_dnn_tensor* libxsmm_expectval = NULL;
    libxsmm_dnn_tensor* libxsmm_stddev = NULL;
    libxsmm_dnn_tensor* libxsmm_gamma = NULL;
    libxsmm_dnn_tensor* libxsmm_beta = NULL;
    libxsmm_dnn_tensor* libxsmm_delinput = NULL;
    libxsmm_dnn_tensor* libxsmm_delinput_add = NULL;
    libxsmm_dnn_tensor* libxsmm_deloutput = NULL;
    libxsmm_dnn_tensor* libxsmm_delgamma = NULL;
    libxsmm_dnn_tensor* libxsmm_delbeta = NULL;
    libxsmm_dnn_tensor_datalayout* libxsmm_layout;
    libxsmm_dnn_err_t status;

    void *bexpect=NULL, *bstddev=NULL;
    float scaling_factor_=0;
    void *scratch=NULL;
    bool updated_scratch=false;

  public:
    FusedBNormXSMM(FusedBNormImplParams* gp, int engine);
    virtual ~FusedBNormXSMM(void) {}

    // Assume external threading, e.g., #pragma omp
    void forwardPropagate(vector<TensorBuf*> inp, TensorBuf* gammap, TensorBuf* betap, float *gmeanp, float *grstdp, TensorBuf *outp, int tid);
    void backPropagate(vector<TensorBuf*> inp, TensorBuf* outp, TensorBuf* gammap, TensorBuf *deloutp, TensorBuf *delgammap, TensorBuf *delbetap, vector<TensorBuf *> delinp, int tid);
};
