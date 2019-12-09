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
#include "Node.hpp"
#include "Engine.hpp"
#include "Params.hpp"
#include "Tensor.hpp"
#include "proto/gxm.pb.h"

using namespace std;
using namespace gxm;

class AccuracyParams : public NNParams
{
  public:
    AccuracyParams(void) {}
    virtual ~AccuracyParams(void) {}

    void set_axis(int axis) { axis_ = axis; }
    int get_axis() { return axis_; }

    void set_top_k(int top_k) { top_k_ = top_k; }
    int get_top_k() { return top_k_; }

  protected:
    int axis_, top_k_;
};

static MLParams* parseAccuracyParams(NodeParameter* np)
{
  AccuracyParams *p = new AccuracyParams();
  AccuracyParameter ap = np->accuracy_param();

  // Set name of node
  assert(!np->name().empty());
  p->set_node_name(np->name());

  //Set node type (Convolution, FullyConnected, etc)
  assert(!np->type().empty());
  p->set_node_type(np->type());

  //Set tensor names
  //Set tensor names
  for(int i=0; i<np->bottom_size(); i++)
  {
    assert(!np->bottom(i).empty());
    p->set_bottom_names(np->bottom(i));
  }

  //Set Mode for the node
  assert((np->mode() == TRAIN) || (np->mode() == TEST));
  p->set_mode(np->mode());

  int axis = ap.axis();
  p->set_axis(axis);

  int top_k = ap.top_k();
  p->set_top_k(top_k);

  return p;
}

class AccuracyNode : public NNNode
{
  public:

    AccuracyNode(AccuracyParams* p, MLEngine* e);

    virtual ~AccuracyNode(void) {}

  protected:
    void forwardPropagate();

    vector<Tensor*> tenBot_;
    vector<TensorBuf*> tenBotData_;
    string node_name_, node_type_;
    Shape ts_;
    int top_k_, train_batch_count_, test_batch_count_;
    double avg_train_acc_, avg_test_acc_;
    MLEngine *eptr_;
#if 1
    vector<float> max_val;
    vector<int> max_id;
    vector<std::pair<float, int> > bot_data_vec;
#endif
    void shape_setzero(Shape* s)
    {
      for(int i=0; i<MAX_DIMS; i++)
        s->dims[i] = 0;
    }
};
