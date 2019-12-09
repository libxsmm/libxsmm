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
#include "Node.hpp"
#include "Engine.hpp"
#include "Params.hpp"
#include "Tensor.hpp"
#include "proto/gxm.pb.h"
#include "fillers.hpp"
#include "FusedBNormImpl.hpp"
#include "FusedBNormXSMM.hpp"

using namespace std;
using namespace gxm;

class FusedBNormParams : public NNParams
{
  public:
    FusedBNormParams(void) {}

    virtual ~FusedBNormParams(void) {}

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

    void set_ipads(int ipdims, int ipad)
    {
      for(int i=0; i<ipdims; i++)
        ipads_.push_back(ipad);
    }

    void set_ipads(int iph, int ipw, int ipd)
    {
      ipads_.push_back(iph);
      ipads_.push_back(ipw);
      ipads_.push_back(ipd);
    }

    vector<int>& get_ipads() { return ipads_; }

    void set_lr_mult(float lr) {lr_mult_ = lr;}
    float get_lr_mult() { return lr_mult_; }

    void set_decay_mult(float decay) { decay_mult_ = decay;}
    float get_decay_mult() { return decay_mult_; }

    void set_eps(float eps) { eps_ = eps; }
    float get_eps() { return eps_; }

    void set_mmf(float mmf) { mmf_ = mmf; }
    float get_mmf() { return mmf_; }

    void set_global_stats_flag(bool s) { use_global_stats_ = s; }
    bool get_global_stats_flag() { return use_global_stats_; }

    void set_relu(bool r) { relu_ = r; }
    bool get_relu() { return relu_; }

    void set_bwd_relu(bool br) { brelu_ = br; }
    bool get_bwd_relu() { return brelu_; }

    void set_eltwise(bool e) { eltwise_ = e; }
    bool get_eltwise() { return eltwise_; }

    void set_data_type(int t) { data_type_ = t; }
    int get_data_type() { return data_type_; }

    void set_compute_engine(int ce) { compute_engine_ = ce; }
    int get_compute_engine() { return compute_engine_; }

    void set_algo_type(int at) { algotype_ = at; }
    int get_algo_type() { return algotype_; }

  protected:
    vector<int> strides_;
    vector<int> pads_, ipads_;
    bool relu_, brelu_, eltwise_, use_global_stats_;
    float eps_, mmf_, lr_mult_, decay_mult_;
    int compute_engine_, algotype_, data_type_;
};

static MLParams* parseFusedBNormParams(NodeParameter* np)
{
  FusedBNormParams* fbnp = new FusedBNormParams();

  // Set name of node
  string str = np->name();
  assert(!str.empty());
  fbnp->set_node_name(str);

  //Set node type (FusedBNorm)
  str = np->type();
  assert(!str.empty());
  fbnp->set_node_type(str);

  //Set tensor names
  for(int i=0; i<np->bottom_size(); i++)
  {
    assert(!np->bottom(i).empty());
    fbnp->set_bottom_names(np->bottom(i));
  }

  assert(!np->top(0).empty());
  fbnp->set_top_names(np->top(0));

  //Set Mode for the node
  assert((np->mode() == TRAIN) || (np->mode() == TEST));
  fbnp->set_mode(np->mode());

  //Set backprop needed/not needed flag for this node
  fbnp->set_bprop_flag(np->propagate_down());

  FusedBNormParameter p = np->fused_bnorm_param();

  int sdims = p.stride_size();
  switch(sdims)
  {
    int sh, sw, sd=0;

    case 0:
      sh = p.stride_h();
      sw = p.stride_w();

      assert((sh > 0) && (sw > 0));
      fbnp->set_strides(sh, sw, sd);
      break;

    case 1:
      sh = p.stride(0);
      fbnp->set_strides(sh, sh, sd);
      break;

    case 2:
      sh = p.stride(0);
      sw = p.stride(1);
      fbnp->set_strides(sh, sw, sd);
      break;

    case 3:
      sh = p.stride(0);
      sw = p.stride(1);
      sd = p.stride(2);
      fbnp->set_strides(sh, sw, sd);
      break;
  }

  // pads
  int pdims = p.pad_size();
  switch(pdims)
  {
    int ph, pw, pd=0;
    case 0:
      ph = p.pad_h();
      pw = p.pad_w();

      fbnp->set_pads(ph, pw, pd);
      break;

    case 1:
      ph = p.pad(0);
        fbnp->set_pads(ph, ph, pd);
      break;

    case 2:
      ph = p.pad(0);
      pw = p.pad(1);
      fbnp->set_pads(ph, pw, pd);
      break;

    case 3:
      ph = p.pad(0);
      pw = p.pad(1);
      pd = p.pad(2);
      fbnp->set_pads(ph, pw, pd);
      break;
  }

  // input pads
  int ipdims = p.ipad_size();
  switch(ipdims)
  {
    int iph, ipw, ipd=0;
    case 0:
      iph = p.ipad_h();
      ipw = p.ipad_w();

      fbnp->set_ipads(iph, ipw, ipd);
      break;

    case 1:
      iph = p.ipad(0);
        fbnp->set_ipads(iph, iph, ipd);
      break;

    case 2:
      iph = p.ipad(0);
      ipw = p.ipad(1);
      fbnp->set_ipads(iph, ipw, ipd);
      break;

    case 3:
      iph = p.ipad(0);
      ipw = p.ipad(1);
      ipd = p.ipad(2);
      fbnp->set_ipads(iph, ipw, ipd);
      break;
  }

  fbnp->set_lr_mult(p.lr_mult());
  fbnp->set_decay_mult(p.decay_mult());
  fbnp->set_mmf(p.mmf());
  fbnp->set_eps(p.eps());
  fbnp->set_global_stats_flag(p.use_global_stats());
  fbnp->set_relu(p.relu());
  fbnp->set_bwd_relu(p.bwd_relu());
  fbnp->set_eltwise(p.eltwise());

  fbnp->set_data_type(p.data_type());
  fbnp->set_compute_engine(p.engine());
  fbnp->set_algo_type(p.algotype());

  return fbnp;
}

class FusedBNormNode : public NNNode
{
  public:
    FusedBNormNode(FusedBNormParams* p, MLEngine* e);
    void Checkpoint(TensorBuf *tBuf, string name, string format);
    void fillBuffer(TensorBuf *tBuf, int buftype, long long int bytes);
    void fillBiasMultipliers(float* lr_mult, float* decay_mult, long long int bytes);
    void convert_bf16_f32(libxsmm_bfloat16* in, float* out, int len);

    virtual ~FusedBNormNode(void) {}

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

    Tensor* tenTop_;
    vector<Tensor *> tenBot_;
    Tensor *tenScale_, *tenShift_;
    Tensor *tenMean_, *tenVar_;

    FusedBNormImplParams gparams_;
    vector<TensorBuf *> tenBotDiff_, tenBotData_;
    TensorBuf *tenTopData_, *tenTopDiff_; // Output data
    TensorBuf *tenScaleData_, *tenScaleDiff_;
    TensorBuf *tenShiftData_, *tenShiftDiff_;
    TensorBuf *tenScaleInc_, *tenShiftInc_;
    TensorBuf *tenMeanData_, *tenVarData_;
    TensorBuf *tenScratchData_;

    float *gmean_, *gvar_, eps, lr_mult_, decay_mult_;
    float *stptr=NULL,*cbptr;
    string scale_, shift_, mean_, var_;
    bool first_fp=true, first_bp=true;

    int count_, in_dtype, out_dtype;
    float scf_=0;

    vector<int> bot_cengine_;
    Shape ts_;
    FusedBNormImpl *impl=NULL;
    SolverNode *solver_;
    MLEngine* eptr_;
};
