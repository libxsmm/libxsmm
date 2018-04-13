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

using namespace std;
using namespace gxm;

class SoftmaxLossNode : public NNNode
{
  public:

    SoftmaxLossNode(SoftmaxLossParams* p, MLEngine* e) : NNNode(p, e) {
      NNNode::nname_ = p->get_node_name();
      NNNode::ntype_ = p->get_node_type();
      NNNode::mode_ = p->get_mode();
      NNNode::top_ = p->get_top_name();
      NNNode::bottom_ = p->get_bottom_name();

      NNNode::has_weights_ = false;
      NNNode::bp_flag_ = true;

      //Create output tensor
      this->tenTop_ = new Tensor(NNNode::top_);
      assert(this->tenTop_ != NULL);
      this->tenTop_->setOwner(this);
      tenTopData_ = tenTop_->getBuf(DATA);

      this->tenBot_ = e->get_tensor(NNNode::bottom_);
      assert(this->tenBot_ != NULL);
      this->setPrevNode((NNNode*)this->tenBot_->getOwner());
      tenBotData_ = tenBot_->getBuf(DATA);

      //Output tensor data type = input tensor data type
      int dtype = this->tenBot_->getBufDataType(DATA);
      this->tenTop_->setBufDataType(DATA, dtype);

      Shape* bs = this->tenBot_->getShape();
      assert(bs->ndims <= MAX_DIMS);

      shape_setzero(&ts_);

      ts_.ndims = 1;
      ts_.dims[0] = 1;
      tenTop_->setShape(&ts_);

      long long int size = 1;
      for(int i=0; i<ts_.ndims; i++)
        size *= ts_.dims[i];

      if(dtype == DT_FLOAT)
        size = size*sizeof(float);
      else if(dtype == DT_INT)
        size = size*sizeof(int);

      // Set the logical size of the tensor buffer for bufId=0 (forward data buffer).
      // Note: we have no knowledge of the machine parameters here, so effectively this is single-machine config
      this->tenTop_->setDataBufferSize(DATA, size);

      // Register output tensor in tensorMap
      bool inserted = e->register_tensor(NNNode::top_, this->tenTop_);
      if(!inserted)
        printf("Warning: Tensor %s already registered\n",NNNode::top_.c_str());

      if(!e->is_inference_only())
      {

        if(NNNode::bp_flag_)
        {
          tenBotDiff_ = tenBot_->addBuf();
          tenBotDiff_->setDataType(dtype);

          size = 1;
          for(int i=0; i<bs->ndims; i++)
            size = size*bs->dims[i];
          if(dtype == DT_FLOAT)
            size = size*sizeof(float);
          else if(dtype == DT_INT)
            size = size*sizeof(int);

          // Set the size of the input-gradient buffer
          tenBotDiff_->setBufferSize(size);
        }
      }
    }

    virtual ~SoftmaxLossNode(void) {}

    void createTasks(list<Task*>, int);
    void createPersistentTask();

    void createStrategy(int);
    void enqueTask(int pos);
    void createCheckPoint();
    void restoreCheckPoint();

  protected:
    Tensor *tenBot_, *tenTop_;
    TensorBuf *tenTopData_, *tenBotData_, *tenBotDiff_;
    string node_name_, node_type_;
    Shape ts_;

    void shape_setzero(Shape* s)
    {
      for(int i=0; i<MAX_DIMS; i++)
        s->dims[i] = 0;
    }

    void forwardPropagate();
    void backPropagate();
    void weightUpdate();
    void solverStep();

};

