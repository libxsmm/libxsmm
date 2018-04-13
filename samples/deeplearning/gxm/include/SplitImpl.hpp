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

#include <omp.h>
#include <assert.h>
#include <sys/time.h>
#include <string.h>
#include "common.hpp"
#include "check.hpp"
#include "Tensor.hpp"

typedef struct {
  int bdims, tdims;
  int nInput;
  vector<int> nOutput;
  int batch_size;
  int iHeight, iWidth, iDepth;
  int oHeight, oWidth, oDepth;
  int stride_h, stride_w, stride_d;
  int num_threads;
} SplitImplParams;

class SplitImpl
{
  protected:
    SplitImplParams *gp;
    int engine;
    TensorLayoutType top_layout_type, gbot_layout_type;
    void *top_layout, *gbot_layout;
    int bot_compute_engine=-1;
    vector<int> top_compute_engine;

  public:
    SplitImpl(SplitImplParams* gp_, int engine_) : gp(gp_), engine(engine_) {}

    void set_top_compute_engine(int e) { top_compute_engine.push_back(e);}
    void set_bot_compute_engine(int e) { bot_compute_engine = e;}

    virtual void forwardPropagate(TensorBuf *inp, vector<TensorBuf *>& outp, int tid) = 0;
    virtual void backPropagate(vector<TensorBuf *>& deloutp, TensorBuf *delinp, int tid) = 0;

    virtual void forwardPropagate(TensorBuf *inp, vector<TensorBuf *>& outp)
    {
      switch(engine)
      {
        case XSMM:
          forwardPropagate(inp, outp, 0);
          break;
      }
    }

    virtual void backPropagate(vector<TensorBuf*>& deloutp, TensorBuf* delinp)
    {
      switch(engine)
      {
        case XSMM:
          backPropagate(deloutp, delinp, 0);
          break;
      }
    }
};
