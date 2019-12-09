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
  string node_name;
  int nInput, nOutput;
  int batch_size;
  int iHeight, iWidth, iDepth;
  int oHeight, oWidth, oDepth;
  int ipad_h, ipad_w, ipad_d;
  int opad_h, opad_w, opad_d;
  int pad_h, pad_w, pad_d;
  int stride_h, stride_w, stride_d;
  int kh, kw, kd;
  int group;
  bool bias_term, compute_stats;
  bool relu, bwd_relu, physical_padding;
  int algType;
  int bdims, tdims, wdims, bidims;
  int in_data_type, out_data_type;
  int num_threads;
  int num_numa_nodes;
} ConvImplParams;

class ConvImpl
{
  protected:
    ConvImplParams *gp;
    int engine;
    TensorLayoutType top_layout_type, gbot_layout_type;
    void *top_layout, *gbot_layout;
    int top_compute_engine=-1;
    int bot_compute_engine=-1;
    string nname;
    TensorBuf* scratchp;

  public:
    ConvImpl(ConvImplParams* gp_, int engine_): gp(gp_), engine(engine_) {}

    void set_top_compute_engine(int e) { top_compute_engine = e;}
    void set_bot_compute_engine(int e) { bot_compute_engine = e;}
    void set_node_name(string s) { nname = s; }
    void set_scratch_buffer(TensorBuf* sb) { scratchp = sb; }

    virtual void forwardPropagate(TensorBuf *inp, TensorBuf *weightp, TensorBuf* hweightp, TensorBuf *biasp, TensorBuf *outp, int tid) = 0;
    virtual void backPropagate(TensorBuf* inp, TensorBuf *deloutp, TensorBuf* weightp, TensorBuf *delinp, int tid) = 0;

    virtual void weightUpdate(TensorBuf *inp, TensorBuf *deloutp, TensorBuf *delweightp, TensorBuf *delbiasp, int tid) = 0;
    virtual void dumpBuffer(TensorBuf*, void*) {}

    virtual void forwardPropagate(TensorBuf *inp, TensorBuf* weightp, TensorBuf* hweightp, TensorBuf* biasp, TensorBuf *outp)
    {
      switch(engine)
      {
        case XSMM:
          forwardPropagate(inp, weightp, hweightp, biasp, outp, 0);
          break;
      }
    }

    virtual void backPropagate(TensorBuf* inp, TensorBuf* weightp, TensorBuf *deloutp, TensorBuf *delinp)
    {
      switch(engine)
      {
        case XSMM:
          backPropagate(inp, weightp, deloutp, delinp, 0);
          break;
      }
    }

    virtual void weightUpdate(TensorBuf *inp, TensorBuf *deloutp, TensorBuf *delweightp, TensorBuf *delbiasp)
    {
      switch(engine)
      {
        case XSMM:
          weightUpdate(inp, deloutp, delweightp, delbiasp, 0);
          break;
      }
    }
};


