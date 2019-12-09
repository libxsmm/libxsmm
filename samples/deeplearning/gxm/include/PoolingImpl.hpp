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
#include <assert.h>
#include "common.hpp"
#include "check.hpp"
#include "Tensor.hpp"

typedef struct {
  string node_name;
  int bdims, tdims;
  int nInput, nOutput;
  int batch_size;
  int in_data_type, out_data_type;
  int iHeight, iWidth, iDepth;
  int oHeight, oWidth, oDepth;
  int ipad_h, ipad_w, ipad_d;
  int opad_h, opad_w, opad_d;
  int pad_h, pad_w, pad_d;
  int stride_h, stride_w, stride_d;
  int kh, kw, kd;
  int pool_mode, data_type;
  int algType;
  int num_threads;
} PoolImplParams;

enum PoolFuncType {MAX, AVE};

class PoolImpl
{
  protected:
    PoolImplParams *gp;
    int engine;
    TensorLayoutType bot_layout_type, top_layout_type, gbot_layout_type;
    void *bot_layout=NULL, *top_layout=NULL, *gbot_layout=NULL;
    int top_compute_engine=-1;
    int bot_compute_engine=-1;
    string next_ntype, nname;
    TensorBuf* scratchp;

  public:
    PoolImpl(PoolImplParams* gp_, int engine_) : gp(gp_), engine(engine_) {}

    void set_top_compute_engine(int e) { top_compute_engine = e;}
    void set_bot_compute_engine(int e) { bot_compute_engine = e;}
    void set_next_node_type(string s) { next_ntype = s; }
    void set_node_name(string s) { nname = s; }
    void set_scratch_buffer(TensorBuf* sb) { scratchp = sb; }

    // Assume external threading, e.g., #pragma omp
    virtual void forwardPropagate(TensorBuf *inp, TensorBuf *outp, int *maskp, int tid) = 0;
    virtual void backPropagate(TensorBuf *deloutp, int *maskp, TensorBuf *delinp, int tid) = 0;

    virtual void forwardPropagate(TensorBuf *inp, TensorBuf *outp, int *maskp)
    {
      switch(engine)
      {
        case XSMM:
          forwardPropagate(inp, outp, maskp, 0);
          break;
      }
    }

    virtual void backPropagate(TensorBuf *deloutp, int *maskp, TensorBuf *delinp)
    {
      switch(engine)
      {
        case XSMM:
          backPropagate(deloutp, maskp, delinp, 0);
          break;
      }
    }
};
