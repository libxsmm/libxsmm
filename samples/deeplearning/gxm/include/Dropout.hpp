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
#include <string>
#include <stdio.h>

#include "Node.hpp"
#include "Engine.hpp"
#include "Params.hpp"
#include "Tensor.hpp"

#include "proto/gxm.pb.h"
#include "check.hpp"
#include "io.hpp"

using namespace std;
using namespace gxm;

typedef struct {
  int nInput, nOutput;
  int iHeight, iWidth;
  int oHeight, oWidth;
  int batch_size;
  int algType, data_type;
  int num_threads;
}DropoutImplParams;

class DropoutParams : public NNParams
{
  public:
    DropoutParams(void) {}
    virtual ~DropoutParams(void) {}

    void set_dropout_ratio(float r) { dropout_ratio_ = r; }
    float get_dropout_ratio() { return dropout_ratio_; }

    void set_data_type(int t) { data_type_ = t; }
    int get_data_type() { return data_type_; }

    void set_compute_engine(int ce) { compute_engine_ = ce; }
    int get_compute_engine() { return compute_engine_; }

    void set_algo_type(int at) { algotype_ = at; }
    int get_algo_type() { return algotype_; }

  protected:
    float dropout_ratio_;
    int compute_engine_, algotype_, data_type_;
};

static MLParams* parseDropoutParams(NodeParameter* np)
{
  DropoutParams* dp = new DropoutParams();

  // Set name of node
  string str = np->name();
  assert(!str.empty());
  dp->set_node_name(str);

  //Set node type (Dropout)
  str = np->type();
  assert(!str.empty());
  dp->set_node_type(str);

  //Set tensor names
  assert(np->bottom_size() == 1);
  assert(!np->bottom(0).empty());
  dp->set_bottom_names(np->bottom(0));

  assert(np->top_size() == 1);
  assert(!np->top(0).empty());
  dp->set_top_names(np->top(0));

  //Set Mode for the node
  assert((np->mode() == TRAIN) || (np->mode() == TEST));
  dp->set_mode(np->mode());

  //Set backprop needed/not needed flag for this node
  dp->set_bprop_flag(np->propagate_down());

  DropoutParameter p = np->dropout_param();

  dp->set_dropout_ratio(p.dropout_ratio());
  dp->set_data_type(p.data_type());
  dp->set_compute_engine(p.engine());
  dp->set_algo_type(p.algotype());

  return dp;
}

class DropoutNode : public NNNode
{
  public:
    DropoutNode(DropoutParams* p, MLEngine* e);
    virtual ~DropoutNode(void) {}

  protected:
    void forwardPropagate();
    void backPropagate();

    void configure(int engine);

    void shape_setzero(Shape* s)
    {
      for(int i=0; i<MAX_DIMS; i++)
        s->dims[i] = 0;
    }

    Tensor *tenTop_; // Output tensor pointer
    Tensor *tenBot_; // Input tensor pointer
    void *tenMask_;
    TensorBuf *tenBotDiff_, *tenBotData_; // Data & Gradients with respect to input
    TensorBuf *tenTopData_; // Output buffer
    unsigned int *seeds; // Mask and seeds buffers
    Shape ts_;
    float threshold_, scale_;
    unsigned int uint_threshold_;
    MLEngine* eptr_;

    DropoutImplParams gparams_;
};
