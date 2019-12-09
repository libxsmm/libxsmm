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
    virtual void backPropagate(TensorBuf *deloutp, TensorBuf* weightp, TensorBuf* delgammap, TensorBuf* delbetap, TensorBuf *delmidp, vector<TensorBuf *>& delinp, int tid) = 0;

    virtual void weightUpdate(TensorBuf *inp, TensorBuf *deloutp, TensorBuf *delmidp, TensorBuf *delweightp, TensorBuf *delgammap, TensorBuf *delbetap, int tid) = 0;

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

    virtual void backPropagate(TensorBuf *deloutp, TensorBuf* weightp, TensorBuf* delgammap, TensorBuf* delbetap, TensorBuf *delmidp, vector<TensorBuf *>& delinp)
    {
      switch(engine)
      {
        case XSMM:
          backPropagate(deloutp, weightp, delgammap, delbetap, delmidp, delinp, 0);
          break;
      }
    }

    virtual void weightUpdate(TensorBuf *inp, TensorBuf *deloutp, TensorBuf *delmidp, TensorBuf *delweightp, TensorBuf *delgammap, TensorBuf *delbetap)
    {
      switch(engine)
      {
        case XSMM:
          weightUpdate(inp, delmidp, deloutp, delweightp, delgammap, delbetap, 0);
          break;
      }
    }
};


