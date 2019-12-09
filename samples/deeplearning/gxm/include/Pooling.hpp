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
#include "proto/gxm.pb.h"
#include "common.hpp"
#include "PoolingImpl.hpp"
#include "PoolingXSMM.hpp"

using namespace std;
using namespace gxm;

class PoolingParams : public NNParams
{
  public:
    PoolingParams(void)  {}
    ~PoolingParams(void) {}

    void set_kernel_dims(int kdims, int ksize)
    {
      for(int i=0; i<kdims; i++)
        kernel_dim_.push_back(ksize);
    }

    void set_kernel_dims(int kh, int kw, int kd)
    {
      kernel_dim_.push_back(kh);
      kernel_dim_.push_back(kw);
      kernel_dim_.push_back(kd);
    }

    vector<int>& get_kernel_dims() { return kernel_dim_; }

    void set_strides(int sdims, int stride)
    {
      for(int i=0; i<sdims; i++)
        strides_.push_back(stride);
    }

    void set_strides(int sh, int sw, int sd)
    {
      strides_.push_back(sh);
      strides_.push_back(sw);
      strides_.push_back(sd);
    }

    vector<int>& get_strides() { return strides_; }

    void set_pads(int pdims, int pad)
    {
      for(int i=0; i<pdims; i++)
        pads_.push_back(pad);
    }

    void set_pads(int ph, int pw, int pd)
    {
      pads_.push_back(ph);
      pads_.push_back(pw);
      pads_.push_back(pd);
    }

    vector<int>& get_pads() { return pads_; }

    void set_pool_mode(int m) { pool_mode_ = m; }
    int get_pool_mode() { return pool_mode_; }

    void set_compute_engine(int ce) { compute_engine_ = ce; }
    int get_compute_engine() { return compute_engine_; }

    void set_data_type(int t) { data_type_ = t; }
    int get_data_type() { return data_type_; }

    void set_algo_type(int at) { algotype_ = at; }
    int get_algo_type() { return algotype_; }

  protected:
    vector<int> kernel_dim_; // Order of dimensions is Height, Width, Depth (for 3D Pooling)
    vector<int> strides_;    // Order follows kernel dimension
    vector<int> pads_;       // Order follows kernel dimension
    int pool_mode_, compute_engine_, algotype_, data_type_;
};

static MLParams* parsePoolingParams(NodeParameter* np)
{

  PoolingParams* pp = new PoolingParams();

  // Set name of node
  assert(!np->name().empty());
  pp->set_node_name(np->name());

  //Set node type (Convolution, FullyConnected, etc)
  assert(!np->type().empty());
  pp->set_node_type(np->type());

  //Set tensor names
  assert(np->bottom_size() == 1);
  assert(!np->bottom(0).empty());
  pp->set_bottom_names(np->bottom(0));


  assert(np->top_size() == 1);
  assert(!np->top(0).empty());
  pp->set_top_names(np->top(0));

  //Set Mode for the node
  assert((np->mode() == TRAIN) || (np->mode() == TEST));
  pp->set_mode(np->mode());

  //Set backprop needed/not needed flag for this node
  pp->set_bprop_flag(np->propagate_down());

  // kernel dimensions
  PoolingParameter ppp = np->pooling_param();
  int kdims = ppp.kernel_size_size();

  switch(kdims)
  {
    int kh, kw, kd;
    case 0:
      kh = ppp.kernel_h();
      kw = ppp.kernel_w();
      if(ppp.ndims() == 3)
        kd = ppp.kernel_d();
      else
        kd = 0;

      assert((kh > 0) && (kw > 0));
      pp->set_kernel_dims(kh, kw, kd);
      break;

    case 1:
      kh = ppp.kernel_size(0);
      if(ppp.ndims() == 2)
        pp->set_kernel_dims(kh, kh, 0);
      else if(ppp.ndims() == 3)
        pp->set_kernel_dims(kh, kh, kh);
      break;

    case 2:
      kh = ppp.kernel_size(0);
      kw = ppp.kernel_size(1);
      assert(ppp.ndims() == 2);
      pp->set_kernel_dims(kh, kw, 0);
      break;

    case 3:
      kh = ppp.kernel_size(0);
      kw = ppp.kernel_size(1);
      kd = ppp.kernel_size(2);
      assert(ppp.ndims() == 3);
      pp->set_kernel_dims(kh, kw, kd);
      break;
  }

  // strides
  int sdims = ppp.stride_size();
  switch(sdims)
  {
    int sh, sw, sd;

    case 0:
      sh = ppp.stride_h();
      sw = ppp.stride_w();
      if(ppp.ndims() == 3)
        sd = ppp.stride_d();
      else
        sd = 0;

      assert((sh > 0) && (sw > 0));
      pp->set_strides(sh, sw, sd);
      break;

    case 1:
      sh = ppp.stride(0);
      if(ppp.ndims() == 2)
      pp->set_strides(sh, sh, 0);
      else if(ppp.ndims() == 3)
      pp->set_strides(sh, sh, sh);
      break;

    case 2:
      sh = ppp.stride(0);
      sw = ppp.stride(1);
      assert(ppp.ndims() == 2);
      pp->set_strides(sh, sw, 0);
      break;

    case 3:
      sh = ppp.stride(0);
      sw = ppp.stride(1);
      sd = ppp.stride(2);
      assert(ppp.ndims() == 3);
      pp->set_strides(sh, sw, sd);
      break;
  }

  // pads
  int pdims = ppp.pad_size();
  switch(pdims)
  {
    int ph, pw, pd;
    case 0:
      ph = ppp.pad_h();
      pw = ppp.pad_w();
      if(ppp.ndims() == 3)
        pd = ppp.pad_d();
      else
        pd = 0;

      pp->set_pads(ph, pw, pd);
      break;

    case 1:
      ph = ppp.pad(0);
      if(ppp.ndims() == 2)
        pp->set_pads(ph, ph, 0);
      else if(ppp.ndims() == 3)
        pp->set_pads(ph, ph, ph);
      break;

    case 2:
      ph = ppp.pad(0);
      pw = ppp.pad(1);
      assert(ppp.ndims() == 2);
      pp->set_pads(ph, pw, 0);
      break;

    case 3:
      ph = ppp.pad(0);
      pw = ppp.pad(1);
      pd = ppp.pad(2);
      assert(ppp.ndims() == 3);
      pp->set_pads(ph, pw, pd);
      break;
  }

  pp->set_pool_mode(ppp.pool());

  pp->set_data_type(ppp.data_type());
  pp->set_compute_engine(ppp.engine());
  pp->set_algo_type(ppp.algotype());

  return pp;
}

class PoolingNode : public NNNode
{
  public:
    PoolingNode(PoolingParams* p, MLEngine* e);
    virtual ~PoolingNode(void) {}

  protected:
    void forwardPropagate();
    void backPropagate();
    void shape_setzero(Shape* s)
    {
      for(int i=0; i<MAX_DIMS; i++)
        s->dims[i] = 0;
    }

    void configure(int engine);
    void convert_bf16_f32(libxsmm_bfloat16* in, float* out, int len);

    Tensor* tenTop_; // Output tensor pointer
    Tensor* tenBot_; // Input tensor pointer
    int* tenMask_;
    PoolImplParams gparams_;
    TensorBuf *tenBotDiff_, *tenBotData_;
    TensorBuf *tenTopData_, *tenTopDiff_;
    TensorBuf *tenScratchData_;
    Shape ts_;

    int count_, in_dtype, out_dtype;
    int bot_cengine_;
    bool first_fp=true;
    float *stptr=NULL, cbptr[16];

    PoolImpl* impl;
    MLEngine* eptr_;
};

