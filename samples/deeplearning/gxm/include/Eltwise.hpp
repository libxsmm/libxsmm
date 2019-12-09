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
#include "assert.h"
#include "Node.hpp"
#include "Engine.hpp"
#include "Params.hpp"
#include "Tensor.hpp"
#include "proto/gxm.pb.h"
#include "EltwiseImpl.hpp"
#include "EltwiseXSMM.hpp"

using namespace std;
using namespace gxm;

class EltwiseParams : public NNParams
{
  public:
    EltwiseParams(void) {}
    virtual ~EltwiseParams(void) {}

    void set_op(int op) {op_ = op; }
    int get_op() { return op_; }

    void set_data_type(int t) { data_type_ = t; }
    int get_data_type() { return data_type_; }

    void set_compute_engine(int ce) { compute_engine_ = ce; }
    int get_compute_engine() { return compute_engine_; }

    void set_algo_type(int at) { algotype_ = at; }
    int get_algo_type() { return algotype_; }

  protected:
    int op_, compute_engine_, algotype_, data_type_;
};

static MLParams* parseEltwiseParams(NodeParameter* np)
{
  EltwiseParams *ep = new EltwiseParams();


  // Set name of node
  string str = np->name();
  assert(!str.empty());
  ep->set_node_name(str);

  //Set node type (Convolution, FullyConnected, etc)
  str = np->type();
  assert(!str.empty());
  ep->set_node_type(str);

  //Set tensor names
  for(int i=0; i<np->bottom_size(); i++)
    ep->set_bottom_names(np->bottom(i));
  assert(np->top_size() == 1);
  assert(!np->top(0).empty());
  ep->set_top_names(np->top(0));

  //Set Mode for the node
  assert((np->mode() == TRAIN) || (np->mode() == TEST));
  ep->set_mode(np->mode());

  //Set backprop needed/not needed flag for this node
  ep->set_bprop_flag(np->propagate_down());

  EltwiseParameter pep = np->eltwise_param();

  ep->set_op(pep.op());
  ep->set_data_type(pep.data_type());
  ep->set_compute_engine(pep.engine());
  ep->set_algo_type(pep.algotype());

  return ep;
}

class EltwiseNode : public NNNode
{
  public:
    EltwiseNode(EltwiseParams *p, MLEngine* e);
    virtual ~EltwiseNode(void) {}

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
    vector<Tensor*> tenBot_; // Input tensor pointer
    EltwiseImplParams gparams_;
    vector<TensorBuf*> tenBotDiff_, tenBotData_; // Data & Gradients with respect to input
    TensorBuf *tenTopData_, *tenTopDiff_; // Output data and gradients with respect to output
    Shape ts_;
    vector<int> bot_cengine_;
    int count_ = 0;

    EltwiseImpl *impl=NULL;
};
