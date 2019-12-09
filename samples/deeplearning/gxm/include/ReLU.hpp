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
#include "ReLUImpl.hpp"
#include "ReLUXSMM.hpp"

using namespace std;
using namespace gxm;

class ReLUParams : public NNParams
{
  public:
    ReLUParams(void) {}

    virtual ~ReLUParams(void) {}

    void set_negative_slope(float s) { neg_slope_ = s; }
    float get_negative_slope() { return neg_slope_; }

    void set_data_type(int t) { data_type_ = t; }
    int get_data_type() { return data_type_; }

    void set_compute_engine(int ce) { compute_engine_ = ce; }
    int get_compute_engine() { return compute_engine_; }

    void set_algo_type(int at) { algotype_ = at; }
    int get_algo_type() { return algotype_; }

  protected:
    float neg_slope_;
    int compute_engine_, algotype_, data_type_;
};

static MLParams* parseReLUParams(NodeParameter* np)
{
  ReLUParams* rp = new ReLUParams();

  // Set name of node
  string str = np->name();
  assert(!str.empty());
  rp->set_node_name(str);

  //Set node type (ReLU)
  str = np->type();
  assert(!str.empty());
  rp->set_node_type(str);

  //Set tensor names
  assert(np->bottom_size() == 1);
  assert(!np->bottom(0).empty());
  rp->set_bottom_names(np->bottom(0));

  assert(np->top_size() == 1);
  assert(!np->top(0).empty());
  rp->set_top_names(np->top(0));

  //Set Mode for the node
  assert((np->mode() == TRAIN) || (np->mode() == TEST));
  rp->set_mode(np->mode());

  //Set backprop needed/not needed flag for this node
  rp->set_bprop_flag(np->propagate_down());

  ReLUParameter p = np->relu_param();

  rp->set_negative_slope(p.negative_slope());

  rp->set_data_type(p.data_type());
  rp->set_compute_engine(p.engine());
  rp->set_algo_type(p.algotype());

  return rp;
}

class ReLUNode : public NNNode
{
  public:
    ReLUNode(ReLUParams* p, MLEngine* e);

    virtual ~ReLUNode(void) {}

  protected:
    void forwardPropagate();
    void backPropagate();
    void configure(int engine);

    void shape_setzero(Shape* s)
    {
      for(int i=0; i<MAX_DIMS; i++)
        s->dims[i] = 0;
    }

    Tensor* tenTop_; // Output tensor pointer
    Tensor* tenBot_; // Input tensor pointer
    ReLUImplParams gparams_;
    TensorBuf *tenBotDiff_, *tenBotData_; // Data & Gradients with respect to input
    TensorBuf *tenTopData_, *tenTopDiff_; // Output data

    int count_;

    int bot_cengine_;
    Shape ts_;
    ReLUImpl *impl;
    MLEngine* eptr_;
};
