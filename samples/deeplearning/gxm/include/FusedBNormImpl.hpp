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
  vector<int> nInput;
  int nOutput;
  int pad_h, pad_w;
  int ipad_h, ipad_w;
  int stride_h, stride_w;
  int iHeight, iWidth;
  int oHeight, oWidth;
  int batch_size;
  float eps, mmf;
  bool relu, bwd_relu;
  bool eltwise, use_global_stats;
  int algType;
  int in_data_type, out_data_type;
  int num_threads;
}FusedBNormImplParams;

class FusedBNormImpl
{
  protected:
    FusedBNormImplParams *gp;
    int engine;
    TensorLayoutType bot_layout_type, top_layout_type, gbot_layout_type;
    void *bot_layout, *top_layout, *gbot_layout;
    int top_compute_engine=-1;
    int bot_compute_engine=-1;
    bool bpdone=false;
    string nname;

  public:
    FusedBNormImpl(FusedBNormImplParams* gp_, int engine_): gp(gp_), engine(engine_) {}

    void set_top_compute_engine(int e) { top_compute_engine = e;}
    void set_bot_compute_engine(int e) { bot_compute_engine = e;}
    void set_bpdone(int b) { bpdone = b; }
    bool get_bpdone() { return bpdone; }
    void set_node_name(string s) { nname = s; }

    // Assume external threading, e.g., #pragma omp
   virtual void forwardPropagate(vector<TensorBuf *> inp, TensorBuf* gammap, TensorBuf* betap, float* gmeanp, float* grstdp, TensorBuf *outp, int tid)
    {
      switch(engine)
      {
        case XSMM:
          forwardPropagate(inp, gammap, betap, gmeanp, grstdp, outp, tid);
          break;
      }
    }

    virtual void backPropagate(vector<TensorBuf*> inp, TensorBuf* outp, TensorBuf* gammap, TensorBuf *deloutp, TensorBuf *delgammap, TensorBuf *delbetap, vector<TensorBuf*> delinp, int tid)
    {
      switch(engine)
      {
        case XSMM:
          backPropagate(inp, outp, gammap, deloutp, delgammap, delbetap, delinp, tid);
          break;
      }
    }
};
