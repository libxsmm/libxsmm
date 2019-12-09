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
  string exec_mode;
  int algType;
  int in_data_type, out_data_type;
  int num_threads;
  int num_numa_nodes;
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
    bool use_global_stats;
    string nname;
    TensorBuf* scratchp;
    float scaling_factor_;

  public:
    FusedBNormImpl(FusedBNormImplParams* gp_, int engine_): gp(gp_), engine(engine_) {}

    void set_top_compute_engine(int e) { top_compute_engine = e;}
    void set_bot_compute_engine(int e) { bot_compute_engine = e;}
    void set_node_name(string s) { nname = s; }
    void set_scratch_buffer(TensorBuf* sb) { scratchp = sb; }
    void set_global_stats(bool s) { use_global_stats = s; }
    void set_scaling_factor(float s) { scaling_factor_ = s; }

    // Assume external threading, e.g., #pragma omp
   virtual void forwardPropagate(vector<TensorBuf *> inp, TensorBuf* gammap, TensorBuf* betap, TensorBuf* gmeanp, TensorBuf* gvarp, TensorBuf *outp, int tid)
    {
      switch(engine)
      {
        case XSMM:
          forwardPropagate(inp, gammap, betap, gmeanp, gvarp, outp, tid);
          break;
      }
    }

    virtual void backPropagate(TensorBuf *deloutp, TensorBuf *delgammap, TensorBuf *delbetap, vector<TensorBuf*> delinp, int tid)
    {
      switch(engine)
      {
        case XSMM:
          backPropagate(deloutp, delgammap, delbetap, delinp, tid);
          break;
      }
    }
};
