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
#include "Solver.hpp"
#include "proto/gxm.pb.h"
#include "ConvImpl.hpp"
#include "ConvXSMM.hpp"

using namespace std;
using namespace gxm;

class ConvParams : public NNParams
{
  public:
    ConvParams(void) {}

    virtual ~ConvParams(void) {}

    void set_kernel_dims(int kdims, int ksize)
    {
      for(int i=0; i<kdims; i++)
        this->kernel_dim_.push_back(ksize);
    }

    void set_kernel_dims(int kh, int kw, int kd)
    {
      this->kernel_dim_.push_back(kh);
      this->kernel_dim_.push_back(kw);
      this->kernel_dim_.push_back(kd);
    }

    vector<int>& get_kernel_dims() { return kernel_dim_; }

    void set_strides(int sdims, int stride)
    {
      for(int i=0; i<sdims; i++)
        this->strides_.push_back(stride);
    }

    void set_strides(int sh, int sw, int sd)
    {
      this->strides_.push_back(sh);
      this->strides_.push_back(sw);
      this->strides_.push_back(sd);
    }

    vector<int>& get_strides() { return strides_; }

    void set_pads(int pdims, int pad)
    {
      for(int i=0; i<pdims; i++)
        this->pads_.push_back(pad);
    }
    void set_pads(int ph, int pw, int pd)
    {
      this->pads_.push_back(ph);
      this->pads_.push_back(pw);
      this->pads_.push_back(pd);
    }
    vector<int>& get_pads() { return pads_; }

    void set_output_pads(int pdims, int pad)
    {
      for(int i=0; i<pdims; i++)
        this->opads_.push_back(pad);
    }

    void set_output_pads(int ph, int pw, int pd)
    {
      this->opads_.push_back(ph);
      this->opads_.push_back(pw);
      this->opads_.push_back(pd);
    }
    vector<int>& get_output_pads() { return opads_; }

    void set_group(int g) { this->group_ = g;}
    int get_group() { return this->group_; }

    void set_nOutput(int num_output) { this->nOutput_ = num_output; }
    int get_output() { return nOutput_; }

    void set_weight_filler_type(string ftype) { wfiller_type_ = ftype; }
    string get_weight_filler_type() { return wfiller_type_; }

    void set_std(float s) { std_ = s; }
    float get_std() { return std_; }

    void set_variance_norm(int v) { variance_norm_ = v; }
    int get_variance_norm() { return variance_norm_; }

    void set_bias_filler_type(string ftype) { bfiller_type_ = ftype; }
    string get_bias_filler_type() { return bfiller_type_; }

    void set_value(float v) { value_ = v; }
    float get_value() { return value_; }

    void set_fused_relu(bool relu) { relu_ = relu; }
    bool get_fused_relu() { return relu_; }

    void set_bwd_relu(bool br) { bwd_relu_ = br; }
    bool get_bwd_relu() { return bwd_relu_; }

    void set_bias_term(bool bias) { bias_term_ = bias; }
    bool get_bias_term() { return bias_term_; }

    void set_compute_stats(bool s) { compute_stats_ = s; }
    bool get_compute_stats() { return compute_stats_; }

    void set_physical_padding(bool p) { phys_pad_ = p; }
    bool get_physical_padding() { return phys_pad_; }

    void set_compute_engine(int ce) { compute_engine_ = ce; }
    int get_compute_engine() { return compute_engine_; }

    void set_algo_type(int at) { algotype_ = at; }
    int get_algo_type() { return algotype_; }

    void set_global_params(vector<ParamSpec> psv)
    {
      for(int i=0; i<psv.size(); i++)
      {
        lr_mult_.push_back(psv[i].lr_mult());
        decay_mult_.push_back(psv[i].decay_mult());
      }
    }
    const vector<float>& get_lr_mult() { return lr_mult_; }
    const vector<float>& get_decay_mult() { return decay_mult_; }

    void set_data_type(int t) { data_type_ = t; }
    int get_data_type() { return data_type_; }

  protected:
    vector<int> kernel_dim_; // Order of dimensions is Height, Width, Depth (for 3D Conv)
    vector<int> strides_;    // Order follows kernel dimension
    vector<int> pads_, opads_;       // Order follows kernel dimension
    int nOutput_;            // Number of output feature maps
    string wfiller_type_, bfiller_type_;
    float std_, value_;
    bool relu_, bwd_relu_, bias_term_, compute_stats_;
    bool phys_pad_;
    int group_, compute_engine_, algotype_;
    int variance_norm_, data_type_;
    vector<float> lr_mult_, decay_mult_;
};

static MLParams* parseConvParams(NodeParameter* np)
{

  ConvParams* cp = new ConvParams();

  // Set name of node
  string str = np->name();
  assert(!str.empty());
  cp->set_node_name(str);

  //Set node type (Convolution, FullyConnected, etc)
  str = np->type();
  assert(!str.empty());
  cp->set_node_type(str);

  //Set tensor names
  assert(np->bottom_size() == 1);
  assert(!np->bottom(0).empty());
  cp->set_bottom_names(np->bottom(0));

  for(int i=0; i<np->top_size(); i++)
  {
    assert(!np->top(i).empty());
    cp->set_top_names(np->top(i));
  }

  //Set Mode for the node
  assert((np->mode() == TRAIN) || (np->mode() == TEST));
  cp->set_mode(np->mode());

  //Set backprop needed/not needed flag for this node
  cp->set_bprop_flag(np->propagate_down());

  vector<ParamSpec> psv;
  for(int i=0; i<np->param_size(); i++)
    psv.push_back(np->param(i));
  cp->set_global_params(psv);

  ConvolutionParameter pcp = np->convolution_param();

  bool bias_term = pcp.bias_term();

  int kdims = pcp.kernel_size_size();

  switch(kdims)
  {
    int kh, kw, kd;
    case 0:
      kh = pcp.kernel_h();
      kw = pcp.kernel_w();
      if(pcp.ndims() == 3)
        kd = pcp.kernel_d();
      else
        kd = 0;

      assert((kh > 0) && (kw > 0));
      cp->set_kernel_dims(kh, kw, kd);
      break;

    case 1:
      kh = pcp.kernel_size(0);
      if(pcp.ndims() == 2)
        cp->set_kernel_dims(kh, kh, 0);
      else if(pcp.ndims() == 3)
        cp->set_kernel_dims(kh, kh, kh);
      break;

    case 2:
      kh = pcp.kernel_size(0);
      kw = pcp.kernel_size(1);
      assert(pcp.ndims() == 2);
      cp->set_kernel_dims(kh, kw, 0);
      break;

    case 3:
      kh = pcp.kernel_size(0);
      kw = pcp.kernel_size(1);
      kd = pcp.kernel_size(2);
      assert(pcp.ndims() == 3);
      cp->set_kernel_dims(kh, kw, kd);
      break;
  }

  // strides
  int sdims = pcp.stride_size();
  switch(sdims)
  {
    int sh, sw, sd;

    case 0:
      sh = pcp.stride_h();
      sw = pcp.stride_w();
      if(pcp.ndims() == 3)
        sd = pcp.stride_d();
      else
        sd = 0;

      assert((sh > 0) && (sw > 0));
      cp->set_strides(sh, sw, sd);
      break;

    case 1:
      sh = pcp.stride(0);
      if(pcp.ndims() == 2)
      cp->set_strides(sh, sh, 0);
      else if(pcp.ndims() == 3)
      cp->set_strides(sh, sh, sh);
      break;

    case 2:
      sh = pcp.stride(0);
      sw = pcp.stride(1);
      assert(pcp.ndims() == 2);
      cp->set_strides(sh, sw, 0);
      break;

    case 3:
      sh = pcp.stride(0);
      sw = pcp.stride(1);
      sd = pcp.stride(2);
      assert(pcp.ndims() == 3);
      cp->set_strides(sh, sw, sd);
      break;
  }

  // pads
  int pdims = pcp.pad_size();
  switch(pdims)
  {
    int ph, pw, pd;
    case 0:
      ph = pcp.pad_h();
      pw = pcp.pad_w();
      if(pcp.ndims() == 3)
        pd = pcp.pad_d();
      else
        pd = 0;

      cp->set_pads(ph, pw, pd);
      break;

    case 1:
      ph = pcp.pad(0);
      if(pcp.ndims() == 2)
        cp->set_pads(ph, ph, 0);
      else if(pcp.ndims() == 3)
        cp->set_pads(ph, ph, ph);
      break;

    case 2:
      ph = pcp.pad(0);
      pw = pcp.pad(1);
      assert(pcp.ndims() == 2);
      cp->set_pads(ph, pw, 0);
      break;

    case 3:
      ph = pcp.pad(0);
      pw = pcp.pad(1);
      pd = pcp.pad(2);
      assert(pcp.ndims() == 3);
      cp->set_pads(ph, pw, pd);
      break;
  }

  // output pads
  int opdims = pcp.opad_size();
  switch(opdims)
  {
    int oph, opw, opd;
    case 0:
      oph = pcp.opad_h();
      opw = pcp.opad_w();
      if(pcp.ndims() == 3)
        opd = pcp.opad_d();
      else
        opd = 0;

      cp->set_output_pads(oph, opw, opd);
      break;

    case 1:
      oph = pcp.opad(0);
      if(pcp.ndims() == 2)
        cp->set_output_pads(oph, oph, 0);
      else if(pcp.ndims() == 3)
        cp->set_output_pads(oph, oph, oph);
      break;

    case 2:
      oph = pcp.opad(0);
      opw = pcp.opad(1);
      assert(pcp.ndims() == 2);
      cp->set_output_pads(oph, opw, 0);
      break;

    case 3:
      oph = pcp.opad(0);
      opw = pcp.opad(1);
      opd = pcp.opad(2);
      assert(pcp.ndims() == 3);
      cp->set_output_pads(oph, opw, opd);
      break;
  }

  if(pcp.group() > 1)
    cp->set_group(pcp.group());
  else
    cp->set_group(1);

  int nOutput = pcp.num_output();
  cp->set_nOutput(nOutput);

  FillerParameter wp = pcp.weight_filler();
  cp->set_weight_filler_type(wp.type());
  cp->set_std(wp.std());
  cp->set_variance_norm(wp.variance_norm());

  cp->set_bias_term(bias_term);
  if(bias_term)
  {
    FillerParameter bp = pcp.bias_filler();
    cp->set_bias_filler_type(bp.type());
    cp->set_value(bp.value());
  }

  cp->set_fused_relu(pcp.fusedrelu());
  cp->set_bwd_relu(pcp.bwd_relu());
  cp->set_compute_stats(pcp.compute_stats());
  cp->set_physical_padding(pcp.physical_padding());

  cp->set_data_type(pcp.data_type());
  cp->set_compute_engine(pcp.engine());
  cp->set_algo_type(pcp.algotype());

  return cp;
}

class ConvNode : public NNNode
{
  public:
    ConvNode(ConvParams* p, MLEngine* e);

    virtual ~ConvNode(void) {}

    string get_weight_filler_type() { return wfiller_type_; }
    float get_std() { return std_; }

    string get_bias_filler_type() { return bfiller_type_; }
    float get_value() { return value_; }

    void fillWeightBuffers(TensorBuf* tBuf, int buftype, long long int size);
    void fillWeightMultipliers(float* lr_mult, float* decay_mult, long long int bytes);
    void fillBiasBuffers(TensorBuf* tBuf, int buftype, long long int size);
    void fillBiasMultipliers(float* lr_mult, float* decay_mult, long long int bytes);
    void Checkpoint(TensorBuf* tBuf, string name, string format);
    void convert_bf16_f32(libxsmm_bfloat16* in, float* out, int len);
    void convert_f32_bf16(float* in, libxsmm_bfloat16* out, int len);

  protected:
    void forwardPropagate();
    void backPropagate();
    void weightUpdate();
    void solverStep();

    void configure(int engine);

    void shape_setzero(Shape* s)
    {
      for(int i=0; i<MAX_DIMS; i++)
        s->dims[i] = 0;
    }

    Tensor *tenTop_, *tenBot_, *tenWeight_, *tenBias_;

    ConvImplParams gparams_;
    TensorBuf *tenBotDiff_, *tenBotData_; // Data & Gradients with respect to input
    TensorBuf *tenTopData_;
    TensorBuf *tenTopDiff_; // Output data
    TensorBuf *tenWeightDiff_, *tenWeightData_, *tenWeightInc_; // Weight gradients, data, increments
    TensorBuf *tenBiasData_, *tenBiasDiff_, *tenBiasInc_; // Bias data, gradients, increments
    TensorBuf *tenScratchData_;

    Shape ts_, ws_;
    string wfiller_type_, bfiller_type_;
    string weight_, bias_, mean_, mean2_;
    int variance_norm_;
    float std_, value_;
    int bot_cengine_;
    int count_, in_dtype, out_dtype;
    vector<float> lr_mult_, decay_mult_;
    bool first_fp = true, first_bp=true;
    bool compute_stats_;
    libxsmm_bfloat16* bf16_wt_ptr=NULL;
    float *cbptr, *stptr=NULL, *dwptr=NULL;
    ConvImpl *impl=NULL;

    SolverNode *solver_;
    MLEngine* eptr_;
};



