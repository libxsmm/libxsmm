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
