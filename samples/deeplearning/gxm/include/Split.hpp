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
#include <stdio.h>
#include "assert.h"
#include "Node.hpp"
#include "Engine.hpp"
#include "Params.hpp"
#include "Tensor.hpp"
#include "proto/gxm.pb.h"
#include "SplitImpl.hpp"
#include "SplitLoop.hpp"

using namespace std;
using namespace gxm;

class SplitParams : public NNParams
{
    public:
      SplitParams(void) {}
      virtual ~SplitParams(void) {}

      void set_data_type(int t) { data_type_ = t; }
      int get_data_type() { return data_type_; }

      void set_compute_engine(int ce) { compute_engine_ = ce; }
      int get_compute_engine() { return compute_engine_; }

    protected:
      int compute_engine_, data_type_;
};

static MLParams* parseSplitParams(NodeParameter* np)
{
    SplitParams* sp = new SplitParams();

    // Set name of node
    string str = np->name();
    assert(!str.empty());
    sp->set_node_name(str);

    //Set node type (Convolution, FullyConnected, etc)
    str = np->type();
    assert(!str.empty());
    sp->set_node_type(str);

    //Set tensor names
    assert(np->bottom_size() == 1);
    assert(!np->bottom(0).empty());
    sp->set_bottom_names(np->bottom(0));

    for(int i=0; i<np->top_size(); i++)
      sp->set_top_names(np->top(i));

    //Set Mode for the node
    assert((np->mode() == TRAIN) || (np->mode() == TEST));
    sp->set_mode(np->mode());

    //Set backprop needed/not needed flag for this node
    sp->set_bprop_flag(np->propagate_down());

    SplitParameter psp = np->split_param();

    sp->set_data_type(psp.data_type());
    sp->set_compute_engine(psp.engine());

    return sp;
}

class SplitNode : public NNNode
{
    public:
      SplitNode(SplitParams* p, MLEngine* e);
      virtual ~SplitNode(void) {}

    protected:
      void forwardPropagate();
      void backPropagate();
      void configure(int engine);

      void shape_setzero(Shape* s)
      {
        for(int i=0; i<MAX_DIMS; i++)
          s->dims[i] = 0;
      }

      vector<Tensor *>tenTop_;
      Tensor *tenBot_;
      vector<TensorBuf *> tenTopData_, tenTopDiff_;
      TensorBuf *tenBotData_, *tenBotDiff_;
      int bot_cengine_;
      int count_;

      SplitImplParams gparams_;
      SplitImpl *impl=NULL;
      MLEngine* eptr_;
};
