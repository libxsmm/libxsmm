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
#include "common.hpp"
#include "check.hpp"
#include "Tensor.hpp"

typedef struct {
  string node_name;
  int bdims, tdims;
  int nInput, nOutput;
  int iDepth, iHeight, iWidth;
  int oDepth, oHeight, oWidth;
  int batch_size;
  float negative_slope;
  int data_type;
  int algType;
  int num_threads;
}ReLUImplParams;

class ReLUImpl
{
  protected:
    ReLUImplParams *gp;
    int engine;
    TensorLayoutType bot_layout_type, top_layout_type, gbot_layout_type;
    void *bot_layout, *top_layout, *gbot_layout;
    int top_compute_engine=-1;
    int bot_compute_engine=-1;

  public:
    ReLUImpl(ReLUImplParams* gp_, int engine_): gp(gp_), engine(engine_) {}

    void set_top_compute_engine(int e) { top_compute_engine = e;}
    void set_bot_compute_engine(int e) { bot_compute_engine = e;}

    // Assume external threading, e.g., #pragma omp
    virtual void forwardPropagate(TensorBuf *inp, TensorBuf *outp, int tid) = 0;
    virtual void backPropagate(TensorBuf* inp, TensorBuf *deloutp, TensorBuf *delinp, int tid) = 0;

    virtual void forwardPropagate(TensorBuf *inp, TensorBuf *outp)
    {
      switch(engine)
      {
        case XSMM:
          forwardPropagate(inp, outp, 0);
          break;
      }
    }

    virtual void backPropagate(TensorBuf* inp, TensorBuf *deloutp, TensorBuf *delinp)
    {
      switch(engine)
      {
        case XSMM:
          backPropagate(inp, deloutp, delinp, 0);
          break;
      }
    }
};
