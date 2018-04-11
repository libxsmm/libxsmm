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


class DummyDataParams: public NNParams
{
  public:
    DummyDataParams(void) {}
    virtual ~DummyDataParams(void) {}

    void set_lookahead(int l)
    {
      this->lookahead_ = l;
    }

    void set_chunk(int chunk)
    {
      this->chunk_ = chunk;
    }

    void set_shape_zero()
    {
      shape_.ndims = 0;
      for(int i=0; i<MAX_DIMS; i++)
        shape_.dims[i] = 0;
    }

    void set_shape(int batch)
    {
      shape_.ndims = 1;
      shape_.dims[0] = batch;
      shape_.dims[1] = 1;
      shape_.dims[2] = 1;
      shape_.dims[3] = 1;
    }

    void set_shape(int batch, int channels)
    {
      shape_.ndims = 2;
      shape_.dims[0] = batch;
      shape_.dims[1] = channels;
      shape_.dims[2] = 1;
      shape_.dims[3] = 1;
    }

    void set_shape(int batch, int channel, int height, int width)
    {
      shape_.ndims = 4;
      shape_.dims[0] = batch;
      shape_.dims[1] = channel;
      shape_.dims[2] = height;
      shape_.dims[3] = width;
    }

    void set_num_train_files(int t) {ntrain_ = t; }
    void set_num_test_files(int t) {ntest_ = t;}

    void set_filler_type(string type)
    {
      filler_type_ = type;
    }

    void set_filler_val(float v)
    {
      filler_val_ = v;
    }

    void set_data_type(int t) {data_type_ = t; }
    int get_data_type() {return data_type_; }

    void set_compute_engine(int e) {compute_engine_ = e; }
    int get_compute_engine() {return compute_engine_; }

    int get_lookahead() { return lookahead_; }
    int get_chunk_size() { return chunk_; }
    Shape* get_shape() { return &shape_; }
    string get_filler_type() { return filler_type_; }
    float get_filler_val() { return filler_val_; }
    int get_num_train_files() {return ntrain_;}
    int get_num_test_files() { return ntest_; }

  protected:
    int lookahead_, compute_engine_;
    int chunk_, ntrain_=0, ntest_=0;
    int data_type_;
    Shape shape_;
    string filler_type_;
    float filler_val_;
};

static MLParams* parseDummyDataParams(NodeParameter* np)
{
  DummyDataParams* p = new DummyDataParams();
  const DummyDataParameter* ddp = &(np->dummy_data_param());

  // Set name of node
  assert(!np->name().empty());
  p->set_node_name(np->name());

  //Set node type (Convolution, FullyConnected, etc)
  assert(!np->type().empty());
  p->set_node_type(np->type());

  //Set tensor names
  assert(np->bottom_size() == 0);

  for(int i=0; i<np->top_size(); i++)
  {
    assert(!np->top(i).empty());
    p->set_top_names(np->top(i));
  }

  //Set Mode for the node
  assert((np->mode() == TRAIN) || (np->mode() == TEST));
  p->set_mode(np->mode());
  p->set_bprop_flag(np->propagate_down());

  if(ddp != NULL)
  {
    TensorShape s = ddp->shape(0);
    int ndims = s.dim_size();

    for(int i=0; i<ndims; i++)
      assert(s.dim(i) > 0);

    p->set_shape_zero();

    if(ndims == 1)
      p->set_shape(s.dim(0));
    else if(ndims == 2)
    {
      if(s.dim(1) > 3)
      {
        p->set_shape(s.dim(1));
        p->set_num_train_files(s.dim(0));
        p->set_num_test_files(s.dim(0));
      }
      else
        p->set_shape(s.dim(0), s.dim(1));
    }
    else if(ndims == 4)
      p->set_shape(s.dim(0), s.dim(1), s.dim(2), s.dim(3));
    else if(ndims == 5)
    {
      p->set_shape(s.dim(1), s.dim(2), s.dim(3), s.dim(4));
      p->set_num_train_files(s.dim(0));
      p->set_num_test_files(s.dim(0));
    }

    FillerParameter dfp = ddp->data_filler(0);
    p->set_filler_type(dfp.type());
    if(dfp.value())
      p->set_filler_val(dfp.value());
    else
      p->set_filler_val(0.0f);

    p->set_data_type(ddp->data_type());
    p->set_compute_engine(ddp->engine());
  }

  return p;
}

class DummyDataNode : public NNNode
{
  public:

    DummyDataNode(DummyDataParams* p, MLEngine* e);

    virtual ~DummyDataNode(void) {}

    void fillData(float* ptr, long long int size);
    void fillData(int* ptr, long long int size);
    void fillData(short int* ptr, long long int size);

  protected:
    Tensor *tenBot_;
    vector<Tensor*> tenTop_;
    vector<TensorBuf*> tenTopData_;
    string node_name_, node_type_;
    string filler_type_;
    float filler_val_;
    int global_batch_size_, num_machines_;

    void forwardPropagate();
};

