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
  int bdims, tdims;
  int nInput;
  vector<int> nOutput;
  int batch_size;
  int iHeight, iWidth, iDepth;
  int oHeight, oWidth, oDepth;
  int stride_h, stride_w, stride_d;
  int in_data_type, out_data_type;
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
