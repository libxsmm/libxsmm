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
  int nOutput;
  vector<int> nInput;
  int bdims;
  int tdims;
  int iHeight;
  int iWidth;
  int oHeight;
  int oWidth;
  int batch_size;
  int axis;
  int algType;
  int data_type;
  int num_threads;
} ConcatImplParams;


class ConcatImpl
{
  protected:
    ConcatImplParams *gp;
    int engine;
    TensorLayoutType top_layout_type;
    vector<TensorLayoutType> gbot_layout_type;
    void *top_layout;
    vector<void*> gbot_layout;
    int top_compute_engine=-1;
    vector<int> bot_compute_engine;
    string next_ntype, nname;

  public:
    ConcatImpl(ConcatImplParams* gp_, int engine_): gp(gp_), engine(engine_) {}

    void set_top_compute_engine(int e) { top_compute_engine = e;}
    void set_bot_compute_engine(int e) { bot_compute_engine.push_back(e);}
    void set_next_node_type(string s) { next_ntype = s; }
    void set_node_name(string s) { nname = s; }

    virtual void forwardPropagate(vector<TensorBuf *>& inp, TensorBuf *outp, int tid) = 0;
    virtual void backPropagate(TensorBuf* deloutp, vector<TensorBuf*>& delinp, int tid) = 0;

    virtual void forwardPropagate(vector<TensorBuf*>& inp, TensorBuf* outp)
    {
      switch(engine)
      {
        case XSMM:
          forwardPropagate(inp, outp, 0);
          break;
      }
    }

    virtual void backPropagate(TensorBuf* deloutp, vector<TensorBuf*>& delinp)
    {
      switch(engine)
      {
        case XSMM:
          backPropagate(deloutp, delinp, 0);
          break;
      }
    }
};
