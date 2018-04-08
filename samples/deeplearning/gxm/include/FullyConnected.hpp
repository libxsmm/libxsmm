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
#include "Solver.hpp"
#include "common.hpp"
#include "io.hpp"
#include "proto/gxm.pb.h"
#include "FCImpl.hpp"
#include "FCXSMM.hpp"

using namespace std;
using namespace gxm;

class FCParams : public NNParams
{
  public:
    FCParams(void) {}
    virtual ~FCParams(void) {}

    void set_nOutput(int num_output)  { this->nOutput_ = num_output;  }
    int get_output() { return nOutput_; }

    void set_activation_filler_type(string ftype) { afiller_type_ = ftype; }
    string get_activation_filler_type() { return afiller_type_; }

    void set_weight_filler_type(string ftype) { wfiller_type_ = ftype; }
    string get_weight_filler_type() { return wfiller_type_; }

    void set_std(float s) { std_ = s; }
    float get_std() { return std_; }

    void set_variance_norm(int v) { variance_norm_ = v; }
    int get_variance_norm() { return variance_norm_; }

    void set_bias_filler_type(string ftype) { bfiller_type_ = ftype; }
    string get_bias_filler_type() { return bfiller_type_; }

    void set_bias_term(bool bias) { bias_term_ = bias; }
    bool get_bias_term() { return bias_term_; }

    void set_value(float v) { value_ = v; }
    float get_value() { return value_; }

    void set_timeSteps(int nt) { this->timesteps_ = nt; }

    void set_transpose_flag(bool xpose) { transpose_ = xpose; }
    bool get_transpose_flag() { return transpose_; }

    void set_data_type(int t) { data_type_ = t; }
    int get_data_type() { return data_type_; }

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

  protected:
    int nOutput_, data_type_;
    int timesteps_, compute_engine_, algotype_;
    int variance_norm_;
    bool transpose_;
    string wfiller_type_, bfiller_type_, afiller_type_;
    float std_, value_;
    bool bias_term_;
    vector<float> lr_mult_, decay_mult_;
};

static MLParams* parseFCParams(NodeParameter* np)
{
  FCParams* fcp = new FCParams();

  // Set name of node
  assert(!np->name().empty());
  fcp->set_node_name(np->name());

  //Set node type (Convolution, FullyConnected, etc)
  assert(!np->type().empty());
  fcp->set_node_type(np->type());

  //Set tensor names
  assert(np->bottom_size() == 1);
  assert(!np->bottom(0).empty());
  fcp->set_bottom_names(np->bottom(0));

  assert(np->top_size() == 1);
  assert(!np->top(0).empty());
  fcp->set_top_names(np->top(0));

  //Set Mode for the node
  assert((np->mode() == TRAIN) || (np->mode() == TEST));
  fcp->set_mode(np->mode());

  //Set backprop needed/not needed flag for this node
  fcp->set_bprop_flag(np->propagate_down());

  // Set global parameters such as learning rate multiplier etc.
  vector<ParamSpec> psv;
  for(int i=0; i<np->param_size(); i++)
    psv.push_back(np->param(i));
  fcp->set_global_params(psv);

  FullyConnectedParameter pfcp = np->fc_param();

  int num_output = pfcp.num_output();
  fcp->set_nOutput(num_output);

  FillerParameter wp = pfcp.weight_filler();
  fcp->set_weight_filler_type(wp.type());
  fcp->set_std(wp.std());
  fcp->set_variance_norm(wp.variance_norm());

  bool bias_term = pfcp.bias_term();
  fcp->set_bias_term(bias_term);

  if(bias_term)
  {
    FillerParameter bp = pfcp.bias_filler();
    fcp->set_bias_filler_type(bp.type());
    fcp->set_value(bp.value());
  }

  bool xpose = pfcp.transpose();
  if(xpose)
    fcp->set_transpose_flag(xpose);

  bool activation_term = pfcp.activation_term();
  if(activation_term)
  {
    FillerParameter ap = pfcp.activation_filler();
    fcp->set_activation_filler_type(ap.type());
    fcp->set_value(ap.value());
  }

  int nt = pfcp.num_timesteps();
  fcp->set_timeSteps(nt);

  fcp->set_data_type(pfcp.data_type());
  fcp->set_compute_engine(pfcp.engine());
  fcp->set_algo_type(pfcp.algotype());

  return fcp;
}

class FCNode: public NNNode
{
  public:
    FCNode(FCParams *p, MLEngine* e);

    virtual ~FCNode(void) {}

    string get_weight_filler_type() { return wfiller_type_; }
    float get_std() { return std_; }

    string get_bias_filler_type() { return bfiller_type_; }
    float get_value() { return value_; }

    void fillWeightBuffers(TensorBuf* tBuf, int buftype, long long int size, int machines);
    void fillWeightMultipliers(float* lr_mult, float* decay_mult, long long int bytes);
    void fillBiasBuffers(TensorBuf* tBuf, int buftype, long long int size);
    void fillBiasMultipliers(float *lr_mult, float *decay_mult, long long int bytes);
    void Checkpoint(TensorBuf *ptr, string name, string format);

  protected:
    void forwardPropagate();
    void backPropagate();
    void weightUpdate();
    void solverStep();
    void shape_setzero(Shape* s)
    {
      for(int i=0; i<MAX_DIMS; i++)
        s->dims[i] = 0;
    }

    void configure(int engine);

    Tensor *tenTop_=NULL; // Output tensor pointer
    Tensor *tenBot_=NULL; // Input tensor pointer
    Tensor *tenWeight_=NULL; // Weight tensor pointer
    Tensor *tenBias_=NULL;
    FCImplParams gparams_;
    TensorBuf *tenBotDiff_=NULL, *tenBotData_=NULL;
    TensorBuf *tenTopData_=NULL, *tenTopDiff_=NULL;
    TensorBuf *tenWeightDiff_=NULL, *tenWeightData_=NULL, *tenWeightInc_=NULL;
    TensorBuf *tenBiasData_=NULL, *tenBiasDiff_=NULL, *tenBiasInc_=NULL;
    Shape bs_, ts_, ws_;

    int bot_cengine_;

    int count_;

    string wfiller_type_, bfiller_type_;
    string weight_, bias_;
    float std_, value_;
    int variance_norm_;

    vector<float> lr_mult_, decay_mult_;

    FCImpl* impl;
    SolverNode* solver_;
    MLEngine* eptr_;
};

