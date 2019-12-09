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
  int nInput, nOutput;
  int batch_size;
  int iHeight, iWidth;
  int oHeight, oWidth;
  int kh, kw;
  bool bias_term;
  int in_data_type, out_data_type;
  int algType;
  int num_numa_nodes;
  int num_threads;
} FCImplParams;

class FCImpl
{
  protected:
    FCImplParams* gp;
    int engine;
    TensorLayoutType bot_layout_type, top_layout_type, gbot_layout_type;
    void *bot_layout=NULL, *top_layout=NULL, *gbot_layout=NULL;
    int top_compute_engine=-1;
    int bot_compute_engine=-1;
    string nname;
    TensorBuf* scratchp;

  public:
    FCImpl(FCImplParams* gp_, int engine_): gp(gp_), engine(engine_) {}

    void set_top_compute_engine(int e) { top_compute_engine = e;}
    void set_bot_compute_engine(int e) { bot_compute_engine = e;}
    void set_node_name(string s) { nname = s; }
    void set_scratch_buffer(TensorBuf* sb) { scratchp = sb; }

    virtual void forwardPropagate(TensorBuf *inp, TensorBuf* weightp, TensorBuf *hweightp, TensorBuf* biasp, TensorBuf *outp, int tid) = 0;
    virtual void backPropagate(TensorBuf *deloutp, TensorBuf* weightp, TensorBuf *delinp, int tid) = 0;
    virtual void weightUpdate(TensorBuf *deloutp, TensorBuf *inp, TensorBuf *delweightp, TensorBuf *delbiasp, int tid) = 0;

    virtual void forwardPropagate(TensorBuf *inp, TensorBuf* weightp, TensorBuf *hweightp, TensorBuf* biasp, TensorBuf *outp)
    {
      switch(engine)
      {
        case XSMM:
          forwardPropagate(inp, weightp, hweightp, biasp, outp, 0);
          break;
      }
    }

    virtual void backPropagate(TensorBuf *deloutp, TensorBuf *weightp, TensorBuf *delinp)
    {
      switch(engine)
      {
        case XSMM:
          backPropagate(deloutp, weightp, delinp, 0);
          break;
      }
    }

    virtual void weightUpdate(TensorBuf *deloutp, TensorBuf *inp, TensorBuf *delweightp, TensorBuf *delbiasp)
    {
      switch(engine)
      {
        case XSMM:
          weightUpdate(deloutp, inp, delweightp, delbiasp, 0);
          break;
      }
    }
};
