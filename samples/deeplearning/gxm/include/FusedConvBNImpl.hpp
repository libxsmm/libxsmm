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
  string node_name, node_type;
  vector<int> nInput;
  int nOutput;
  int batch_size;
  int iHeight, iWidth, mHeight, mWidth, oHeight, oWidth;
  int ipad_h, ipad_w, mpad_h, mpad_w, opad_h, opad_w;
  int c_stride_h, c_stride_w, bn_stride_h, bn_stride_w;
  int kh, kw, kd;
  int group;
  float eps, mmf;
  bool use_global_stats, eltwise, split, bprop;
  bool relu_fwd, relu_bwd;
  bool physical_padding;
  int algType;
  int bdims, mdims, tdims, wdims;
  int in_data_type, out_data_type;
  int num_numa_nodes;
  int num_threads;
  void **prev_bn_train_handle_ptr, **prev_bn_test_handle_ptr;
  void **my_bn_train_handle_ptr, **my_bn_test_handle_ptr;
  string exec_mode;
} FusedConvBNImplParams;

class FusedConvBNImpl
{
  protected:
    FusedConvBNImplParams *gp;
    int engine;
    TensorLayoutType top_layout_type;
    TensorLayoutType gbot_layout_type;
    void *top_layout, *gbot_layout;
    vector<int> top_compute_engine, bot_compute_engine;
    bool use_global_stats;
    string nname;
    TensorBuf* scratchp;
    float scaling_factor_;

  public:
    FusedConvBNImpl(FusedConvBNImplParams* gp_, int engine_): gp(gp_), engine(engine_) {}

    void set_top_compute_engine(int e) { top_compute_engine.push_back(e);}
    void set_bot_compute_engine(int e) { bot_compute_engine.push_back(e);}
    void set_node_name(string s) { nname = s; }
    void set_scratch_buffer(TensorBuf* sb) { scratchp = sb; }
    void set_global_stats(bool s) { use_global_stats = s; }
    void set_scaling_factor(float s) { scaling_factor_ = s; }

    virtual void forwardPropagate(vector<TensorBuf *>& inp, TensorBuf* weightp, TensorBuf *hweightp, TensorBuf* midp, TensorBuf* gammap, TensorBuf* betap, TensorBuf *gmeanp, TensorBuf *gvarp, TensorBuf *outp, int tid) = 0;
    virtual void backPropagate(TensorBuf *delmidp, TensorBuf* weightp, TensorBuf *delinp, int tid) = 0;
    virtual void weightUpdate(TensorBuf *inp, TensorBuf *deloutp, TensorBuf *delmidp, TensorBuf *delinpl, TensorBuf *delweightp, TensorBuf *delgammap, TensorBuf *delbetap, int tid) = 0;

    virtual void dumpBuffer(TensorBuf*, void*) {}

    virtual void forwardPropagate(vector<TensorBuf *>& inp, TensorBuf* weightp, TensorBuf* hweightp, TensorBuf* midp, TensorBuf* gammap, TensorBuf* betap, TensorBuf *gmeanp, TensorBuf *gvarp, TensorBuf *outp)
    {
      switch(engine)
      {
        case XSMM:
          forwardPropagate(inp, weightp, hweightp, midp, gammap, betap, gmeanp, gvarp, outp, 0);
          break;
      }
    }

    virtual void backPropagate(TensorBuf *delmidp, TensorBuf* weightp, TensorBuf *delinp)
    {
      switch(engine)
      {
        case XSMM:
          backPropagate(delmidp, weightp, delinp, 0);
          break;
      }
    }

    virtual void weightUpdate(TensorBuf *inp, TensorBuf *deloutp, TensorBuf *delmidp, TensorBuf *delinpl, TensorBuf *delweightp, TensorBuf *delgammap, TensorBuf *delbetap)
    {
      switch(engine)
      {
        case XSMM:
          weightUpdate(inp, deloutp, delmidp, delinpl, delweightp, delgammap, delbetap, 0);
          break;
      }
    }
};


