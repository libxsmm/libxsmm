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
#include "Solver.hpp"
#include "proto/gxm.pb.h"
#include "FusedConvBNImpl.hpp"
#include "FusedConvBNXSMM.hpp"

#define VLEN 16

using namespace std;
using namespace gxm;

class FusedConvBNParams : public NNParams
{
  public:
    FusedConvBNParams(void) {}

    virtual ~FusedConvBNParams(void) {}

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

    void set_bn_strides(int sdims, int stride)
    {
      for(int i=0; i<sdims; i++)
        bn_strides_.push_back(stride);
    }

    void set_bn_strides(int sh, int sw, int sd)
    {
      bn_strides_.push_back(sh);
      bn_strides_.push_back(sw);
      bn_strides_.push_back(sd);
    }

    vector<int>& get_bn_strides() { return bn_strides_; }

    void set_c_strides(int sdims, int stride)
    {
      for(int i=0; i<sdims; i++)
        c_strides_.push_back(stride);
    }

    void set_c_strides(int sh, int sw, int sd)
    {
      c_strides_.push_back(sh);
      c_strides_.push_back(sw);
      c_strides_.push_back(sd);
    }

    vector<int>& get_c_strides() { return c_strides_; }

    void set_bot_pads(int pdims, int pad)
    {
      for(int i=0; i<pdims; i++)
        bot_pads_.push_back(pad);
    }
    void set_bot_pads(int ph, int pw, int pd)
    {
      bot_pads_.push_back(ph);
      bot_pads_.push_back(pw);
      bot_pads_.push_back(pd);
    }
    vector<int>& get_bot_pads() { return bot_pads_; }

    void set_top_pads(int pdims, int pad)
    {
      for(int i=0; i<pdims; i++)
        top_pads_.push_back(pad);
    }

    void set_top_pads(int ph, int pw, int pd)
    {
      top_pads_.push_back(ph);
      top_pads_.push_back(pw);
      top_pads_.push_back(pd);
    }
    vector<int>& get_top_pads() { return top_pads_; }

    void set_mid_pads(int pdims, int pad)
    {
      for(int i=0; i<pdims; i++)
        mid_pads_.push_back(pad);
    }

    void set_mid_pads(int ph, int pw, int pd)
    {
      mid_pads_.push_back(ph);
      mid_pads_.push_back(pw);
      mid_pads_.push_back(pd);
    }
    vector<int>& get_mid_pads() { return mid_pads_; }

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

    void set_eps(float eps) { eps_ = eps; }
    float get_eps() { return eps_; }

    void set_mmf(float mmf) { mmf_ = mmf; }
    float get_mmf() { return mmf_; }

    void set_global_stats_flag(bool s) { use_global_stats_ = s; }
    bool get_global_stats_flag() { return use_global_stats_; }

    void set_eltwise(bool e) { eltwise_ = e; }
    bool get_eltwise() { return eltwise_; }

    void set_relu_fwd(bool relu_fwd) { relu_fwd_ = relu_fwd; }
    bool get_relu_fwd() { return relu_fwd_; }

    void set_relu_bwd(bool br) { relu_bwd_ = br; }
    bool get_relu_bwd() { return relu_bwd_; }

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
    vector<int> kernel_dim_;
    vector<int> c_strides_, bn_strides_;
    vector<int> bot_pads_, mid_pads_, top_pads_;
    int nOutput_;
    string wfiller_type_;
    float std_, eps_, mmf_;
    bool relu_fwd_, relu_bwd_;
    bool phys_pad_, use_global_stats_, eltwise_;
    int group_, compute_engine_, algotype_;
    int variance_norm_, data_type_;
    vector<float> lr_mult_, decay_mult_;
};

static MLParams* parseFusedConvBNParams(NodeParameter* np)
{

  FusedConvBNParams* fcbnp = new FusedConvBNParams();

  // Set name of node
  string str = np->name();
  assert(!str.empty());
  fcbnp->set_node_name(str);

  //Set node type (Convolution, FullyConnected, etc)
  str = np->type();
  assert(!str.empty());
  fcbnp->set_node_type(str);

  //Set tensor names
  for(int i=0; i<np->bottom_size(); i++)
  {
    assert(!np->bottom(i).empty());
    fcbnp->set_bottom_names(np->bottom(i));
  }

  for(int i=0; i<np->top_size(); i++)
  {
    assert(!np->top(i).empty());
    fcbnp->set_top_names(np->top(i));
  }

  //Set Mode for the node
  assert((np->mode() == TRAIN) || (np->mode() == TEST));
  fcbnp->set_mode(np->mode());

  //Set backprop needed/not needed flag for this node
  fcbnp->set_bprop_flag(np->propagate_down());

  vector<ParamSpec> psv;
  for(int i=0; i<np->param_size(); i++)
    psv.push_back(np->param(i));
  fcbnp->set_global_params(psv);

  FusedConvBNParameter pcp = np->fused_conv_bn_param();

  int kdims = pcp.kernel_size_size();

  switch(kdims)
  {
    int kh, kw;
    case 0:
      kh = pcp.kernel_h();
      kw = pcp.kernel_w();

      assert((kh > 0) && (kw > 0));
      fcbnp->set_kernel_dims(kh, kw, 0);
      break;

    case 1:
      kh = pcp.kernel_size(0);
      fcbnp->set_kernel_dims(kh, kh, 0);
      break;

    case 2:
      kh = pcp.kernel_size(0);
      kw = pcp.kernel_size(1);
      assert(pcp.ndims() == 2);
      fcbnp->set_kernel_dims(kh, kw, 0);
      break;

    default:
      printf("illegal kernel dimension size\n");
      break;
  }

  // conv strides
  int sdims = pcp.c_stride_size();
  switch(sdims)
  {
    int sh, sw, sd;

    case 0:
      sh = pcp.c_stride_h();
      sw = pcp.c_stride_w();
      assert((sh > 0) && (sw > 0));
      fcbnp->set_c_strides(sh, sw, 0);
      break;

    case 1:
      sh = pcp.c_stride(0);
      fcbnp->set_c_strides(sh, sh, 0);
      break;

    case 2:
      sh = pcp.c_stride(0);
      sw = pcp.c_stride(1);
      assert(pcp.ndims() == 2);
      fcbnp->set_c_strides(sh, sw, 0);
      break;
  }

  // bn strides
  sdims = pcp.bn_stride_size();
  switch(sdims)
  {
    int sh, sw, sd;

    case 0:
      sh = pcp.bn_stride_h();
      sw = pcp.bn_stride_w();

      assert((sh > 0) && (sw > 0));
      fcbnp->set_bn_strides(sh, sw, 0);
      break;

    case 1:
      sh = pcp.bn_stride(0);
      fcbnp->set_bn_strides(sh, sh, 0);
      break;

    case 2:
      sh = pcp.bn_stride(0);
      sw = pcp.bn_stride(1);
      assert(pcp.ndims() == 2);
      fcbnp->set_bn_strides(sh, sw, 0);
      break;
  }

  // input pads
  int pdims = pcp.ipad_size();
  switch(pdims)
  {
    int ph, pw, pd;
    case 0:
      ph = pcp.ipad_h();
      pw = pcp.ipad_w();

      fcbnp->set_bot_pads(ph, pw, 0);
      break;

    case 1:
      ph = pcp.ipad(0);
      fcbnp->set_bot_pads(ph, ph, 0);
      break;

    case 2:
      ph = pcp.ipad(0);
      pw = pcp.ipad(1);
      assert(pcp.ndims() == 2);
      fcbnp->set_bot_pads(ph, pw, 0);
      break;
  }

  // middle pads
  pdims = pcp.mpad_size();
  switch(pdims)
  {
    int ph, pw, pd;
    case 0:
      ph = pcp.mpad_h();
      pw = pcp.mpad_w();

      fcbnp->set_mid_pads(ph, pw, 0);
      break;

    case 1:
      ph = pcp.mpad(0);
      fcbnp->set_mid_pads(ph, ph, 0);
      break;

    case 2:
      ph = pcp.mpad(0);
      pw = pcp.mpad(1);
      assert(pcp.ndims() == 2);
      fcbnp->set_mid_pads(ph, pw, 0);
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

      fcbnp->set_top_pads(oph, opw, 0);
      break;

    case 1:
      oph = pcp.opad(0);
      fcbnp->set_top_pads(oph, oph, 0);
      break;

    case 2:
      oph = pcp.opad(0);
      opw = pcp.opad(1);
      assert(pcp.ndims() == 2);
      fcbnp->set_top_pads(oph, opw, 0);
      break;
  }

  if(pcp.group() > 1)
    fcbnp->set_group(pcp.group());
  else
    fcbnp->set_group(1);

  int nOutput = pcp.num_output();
  fcbnp->set_nOutput(nOutput);

  fcbnp->set_mmf(pcp.mmf());
  fcbnp->set_eps(pcp.eps());
  fcbnp->set_global_stats_flag(pcp.use_global_stats());
  fcbnp->set_relu_fwd(pcp.relu_fwd());
  fcbnp->set_relu_bwd(pcp.relu_bwd());

  FillerParameter wp = pcp.weight_filler();
  fcbnp->set_weight_filler_type(wp.type());
  fcbnp->set_std(wp.std());
  fcbnp->set_variance_norm(wp.variance_norm());

  fcbnp->set_eltwise(pcp.eltwise());

  fcbnp->set_physical_padding(pcp.physical_padding());

  fcbnp->set_data_type(pcp.data_type());
  fcbnp->set_compute_engine(pcp.engine());
  fcbnp->set_algo_type(pcp.algotype());

  return fcbnp;
}

class FusedConvBNNode : public NNNode
{
  public:
    FusedConvBNNode(FusedConvBNParams* p, MLEngine* e);

    virtual ~FusedConvBNNode(void) {}

    string get_weight_filler_type() { return wfiller_type_; }
    float get_std() { return std_; }

    void fillWeightBuffers(TensorBuf* tBuf, int buftype, long long int size);
    void fillBuffer(TensorBuf* tBuf, int buftype, long long int size);
    void fillWeightMultipliers(float* lr_mult, float* decay_mult, long long int bytes);
    void fillBiasMultipliers(float* lr_mult, float* decay_mult, long long int bytes);
    void Checkpoint(TensorBuf* tBuf, string name, string format);
    void convert_f32_bf16(float* in, libxsmm_bfloat16* out, int len);
    void convert_bf16_f32(libxsmm_bfloat16* in, float* out, int len);
    void** getBNTrainHandlePtr();
    void** getBNTestHandlePtr();

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

    vector<Tensor*> tenBot_;
    Tensor *tenMid_, *tenTop_, *tenWeight_, *tenScale_, *tenShift_, *tenMean_, *tenVar_;
    FusedConvBNImplParams gparams_;
    vector<TensorBuf *> tenBotDiff_, tenBotData_; // Data & Gradients with respect to input
    TensorBuf *tenMidData_, *tenTopData_;
    TensorBuf *tenMidDiff_=NULL, *tenTopDiff_;
    TensorBuf *tenWeightDiff_, *tenWeightData_, *tenWeightInc_; // Weight gradients, data, increments
    TensorBuf *tenScaleData_, *tenScaleDiff_, *tenScaleInc_; // Gamma data, gradients, increments
    TensorBuf *tenShiftData_, *tenShiftDiff_, *tenShiftInc_; // Beta data, gradients, increments
    TensorBuf *tenMeanData_, *tenVarData_; // Mean, variance data
    TensorBuf *tenScratchData_;

    int in_dtype, out_dtype;

    Shape ts_, ws_, ms_;
    string wfiller_type_;
    string weight_, scale_, shift_, mean_, var_;
    int variance_norm_;
    float std_, *cbptr=NULL, *stptr=NULL, scf_=0;
    float* dwptr=NULL;
    int bot_cengine_;
    int count_;
    vector<float> lr_mult_, decay_mult_;
    bool first_fp = true, first_bp=true, first_upd=true;

    FusedConvBNImpl *impl=NULL;

    SolverNode *solver_;
    MLEngine* eptr_;
};



