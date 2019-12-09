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
#include "assert.h"
#include "MLNode.hpp"
#include "Engine.hpp"
#include <math.h>
#include "libxsmm.h"
#include "check.hpp"

using namespace std;

class SolverParams : public MLParams
{
  public:
    SolverParams(void){}
    virtual ~SolverParams(void) {}

    void setLRPolicy(string p) {lr_policy_ = p;}
    string getLRPolicy() { return lr_policy_; }

    void setGamma(float g) { gamma_ = g; }
    float getGamma() { return gamma_; }

    void setPower(float p) { power_ = p; }
    float getPower() { return power_; }

    void setStepSize(int s) { step_size_ = s; }
    int getStepSize() { return step_size_; }

    void setMaxIter(int i) { max_iter_ = i; }
    int getMaxIter() { return max_iter_; }

    void setLearningRate(float lr) { lr_.push_back(lr); }
    float getLearningRate() { return lr_[0]; }

    void setLearningRates(vector<float> lr)
    {
      for(int i=0; i<lr.size(); i++)
        lr_.push_back(lr[i]);
    }

    const vector<float>& getLearningRates() const { return lr_; }

    void setWarmupLR(float lr) { warmup_lr_.push_back(lr); }
    float getWarmupLR() { return warmup_lr_[0]; }

    void setMomentum(float m) { momentum_.push_back(m); }
    float getMomentum() { return momentum_[0]; }

    void setMomentums(vector<float> m)
    {
      for(int i=0; i<m.size(); i++)
        momentum_.push_back(m[i]);
    }

    const vector<float>& getMomentums() const { return momentum_; }

    void setWeightDecay(float d) { decay_.push_back(d); }
    float getWeightDecay() { return decay_[0]; }

    void setWeightDecays(vector<float> d)
    {
      for(int i=0; i<d.size(); i++)
        decay_.push_back(d[i]);
    }

    const vector<float>& getWeightDecays() const { return decay_; }

    void setLRChangeEpochs(vector<int> e)
    {
      for(int i=0; i<e.size(); i++)
        lrcepochs_.push_back(e[i]);
    }

    const vector<int>& getLRChangeEpochs() const { return lrcepochs_; }

    void setStepValues(vector<int> s)
    {
      stepvalues_.resize(s.size());
      for(int i=0; i<s.size(); i++)
        stepvalues_[i] = s[i];
    }

    const vector<int>& getStepValues() const { return stepvalues_; }

    void setWarmupEpochs(int we) { warmup_epochs_ = we; }
    int getWarmupEpochs() { return warmup_epochs_; }

    void setEpochs(int e) { epochs_ = e; }
    int getEpochs() { return epochs_; }

    void setTestEpoch(int te) { test_epoch_ = te; }
    int getTestEpoch() { return test_epoch_; }

    void setSolverType(string s) { solver_type_ = s; }
    string getSolverType() { return solver_type_; }

    void setGlobalFlag(bool g) { global_ = g; }
    bool getGlobalFlag() { return global_; }

    void setDataType(int t) { data_type_ = t; }
    int getDataType() { return data_type_; }

  protected:
    vector<float> lr_, momentum_, decay_, warmup_lr_;
    vector<int> lrcepochs_, stepvalues_;
    int epochs_, test_epoch_, step_size_, max_iter_;
    string solver_type_, lr_policy_;
    float gamma_, power_;
    int warmup_epochs_, data_type_;
    bool global_;
};

static SolverParams* parseSolverParams(SolverParameter* p)
{
  SolverParams* sp = new SolverParams();

  vector<float> temp;
  vector<int> itemp;

  string policy = p->lr_policy();
  sp->setLRPolicy(policy);

  sp->setLearningRate(p->learning_rate(0));
  sp->setWarmupLR(p->warmup_lr(0));
  sp->setMomentum(p->momentum(0));
  sp->setWeightDecay(p->weight_decay(0));
  sp->setPower(p->power());
  sp->setGamma(p->gamma());
  sp->setStepSize(p->stepsize());
  sp->setMaxIter(p->max_iter());
  if(p->step_values_size() > 0)
  {
    itemp.resize(p->step_values_size());
    for(int i=0; i<itemp.size(); i++)
      itemp[i] = p->step_values(i);
    sp->setStepValues(itemp);
  }
  sp->setWarmupEpochs(p->warmup_epochs());

  assert(p->max_epochs() >= 1);
  sp->setEpochs(p->max_epochs());
  assert(p->test_epoch() >= 1);
  sp->setTestEpoch(p->test_epoch());

  sp->setSolverType(p->type());
  sp->setDataType(p->data_type());

  sp->setGlobalFlag(p->global());

  return sp;
}

class SolverNode : public MLNode
{
  public:

    SolverNode(SolverParams* p, MLEngine* e);
    virtual ~SolverNode(void) {}

    void applyUpdate(float**, float**, void**, int, float, float, string);
    void applyUpdate(float*, float*, void*, int, float, float, string);
    void applyUpdate(float*, float*, void*, int, float*, float*, string);
    void applyUpdate(float**, float**, void**, int, float**, float**, string);
    void convert_bf16_f32(libxsmm_bfloat16**, float**, int);
    void convert_bf16_f32(libxsmm_bfloat16*, float*, int);
    bool getGlobalFlag() { return global_; }

  protected:
    vector<float> lr_, momentum_, decay_;
    vector<int> lrcepochs_, stepvalues_;
    int epochs_, test_epoch_, step_size_, max_iter_;
    int stepidx_, warmup_max_epoch_;
    int data_type_;
    bool global_;
    string solver_type_, lr_policy_;
    map<int, vector<float>> hpmap_;
    float base_lr_, lrval_, mval_, decayval_;
    float gamma_, power_, warmup_lr_;
    float mc_, mc1_, mc2_, prev_lrval_=-1, prev_lrval_1_=-1;
    float *tmp_grad[NUM_NUMA_NODES]={NULL};
    MLEngine *eptr_;
};
