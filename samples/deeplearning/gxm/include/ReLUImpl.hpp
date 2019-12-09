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
