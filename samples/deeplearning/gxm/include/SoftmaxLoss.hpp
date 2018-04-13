/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
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
#include "SoftmaxLossImpl.hpp"
#include "SoftmaxLossLoop.hpp"

using namespace std;
using namespace gxm;


class SoftmaxLossParams : public NNParams
{
  public:
    SoftmaxLossParams(void) {}
    virtual ~SoftmaxLossParams(void) {}

    void set_axis(int axis) { axis_ = axis; }
    int get_axis() { return axis_; }

    void set_data_type(int t) { data_type_ = t; }
    int get_data_type() { return data_type_; }

    void set_loss_weight(vector<float> l)
    {
      for(int i=0; i<l.size(); i++)
        loss_weight_.push_back(l[i]);
    }
    const vector<float>& get_loss_weight() { return loss_weight_; }

  protected:
    int axis_, data_type_;
    vector<float> loss_weight_;
};

static MLParams* parseSoftmaxParams(NodeParameter* np)
{
  SoftmaxLossParams *p = new SoftmaxLossParams();
  SoftmaxParameter sp = np->softmax_param();

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

  assert(np->top_size() == 1);
  assert(!np->top(0).empty());
  p->set_top_names(np->top(0));

  //Set Mode for the node
  assert((np->mode() == TRAIN) || (np->mode() == TEST));
  p->set_mode(np->mode());

  p->set_bprop_flag(np->propagate_down());

  int axis = sp.axis();
  p->set_axis(axis);

  p->set_data_type(sp.data_type());

  vector<float> lw;
  for(int i=0; i<np->loss_weight_size(); i++)
    lw.push_back(np->loss_weight(i));
  p->set_loss_weight(lw);

  return p;
}

class SoftmaxLossNode : public NNNode
{
  public:

    SoftmaxLossNode(SoftmaxLossParams* p, MLEngine* e);

    virtual ~SoftmaxLossNode(void) {}

    void configure(int smaxtype);

  protected:
    vector<Tensor*> tenBot_;
    Tensor *tenTop_;
    TensorBuf *tenTopData_, *tenBotDiff_;
    vector<TensorBuf*> tenBotData_;
    string node_name_, node_type_;
    SMaxLossImplParams gparams_;
    Shape ts_;
    vector<float> loss_weight_;
    float test_loss_;
    size_t node_id_, num_nodes_;

    void shape_setzero(Shape* s)
    {
      for(int i=0; i<MAX_DIMS; i++)
        s->dims[i] = 0;
    }

    void forwardPropagate();
    void backPropagate();

    SMaxLossImpl* impl;
    MLEngine* eptr_;
};


