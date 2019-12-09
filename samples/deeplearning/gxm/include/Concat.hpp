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
#include "ConcatImpl.hpp"
#include "ConcatXSMM.hpp"

using namespace std;
using namespace gxm;

class ConcatParams : public NNParams
{
  public:
    ConcatParams(void) {}
    virtual ~ConcatParams(void) {}

    void set_concat_axis(int axis) {axis_ = axis; }
    int get_concat_axis() { return axis_; }

    void set_data_type(int t) { data_type_ = t; }
    int get_data_type() { return data_type_; }

    void set_compute_engine(int ce) { compute_engine_ = ce; }
    int get_compute_engine() { return compute_engine_; }

    void set_algo_type(int at) { algotype_ = at; }
    int get_algo_type() { return algotype_; }

  protected:
    int axis_, data_type_, compute_engine_, algotype_;
};

static MLParams* parseConcatParams(NodeParameter* np)
{
  ConcatParams *cp = new ConcatParams();


  // Set name of node
  string str = np->name();
  assert(!str.empty());
  cp->set_node_name(str);

  //Set node type (Convolution, FullyConnected, etc)
  str = np->type();
  assert(!str.empty());
  cp->set_node_type(str);

  //Set tensor names
  for(int i=0; i<np->bottom_size(); i++)
    cp->set_bottom_names(np->bottom(i));
  assert(np->top_size() == 1);
  assert(!np->top(0).empty());
  cp->set_top_names(np->top(0));

  //Set Mode for the node
  assert((np->mode() == TRAIN) || (np->mode() == TEST));
  cp->set_mode(np->mode());

  //Set backprop needed/not needed flag for this node
  cp->set_bprop_flag(np->propagate_down());

  ConcatParameter pcp = np->concat_param();

  cp->set_concat_axis(pcp.axis());
  cp->set_data_type(pcp.data_type());
  cp->set_compute_engine(pcp.engine());
  cp->set_algo_type(pcp.algotype());

  return cp;
}

class ConcatNode : public NNNode
{
  public:
    ConcatNode(ConcatParams *p, MLEngine* e);
    virtual ~ConcatNode(void) {}

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
    ConcatImplParams gparams_;
    vector<TensorBuf*> tenBotDiff_, tenBotData_; // Data & Gradients with respect to input
    TensorBuf *tenTopData_, *tenTopDiff_; // Output data and gradients with respect to output
    Shape ts_;
    vector<int> bot_cengine_;
    int count_ = 0;

    ConcatImpl *impl=NULL;
};
