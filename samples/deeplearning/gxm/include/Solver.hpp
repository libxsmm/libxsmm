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
#include "assert.h"
#include "MLNode.hpp"
#include "Engine.hpp"
#include <math.h>

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

  protected:
    vector<float> lr_, momentum_, decay_, warmup_lr_;
    vector<int> lrcepochs_, stepvalues_;
    int epochs_, test_epoch_, step_size_, max_iter_;
    string solver_type_, lr_policy_;
    float gamma_, power_;
    int warmup_epochs_;
    bool global_;
};

static SolverParams* parseSolverParams(SolverParameter* p)
{
  SolverParams* sp = new SolverParams();

  vector<float> temp;
  vector<int> itemp;

  string policy = p->lr_policy();
  sp->setLRPolicy(policy);

  if(policy.compare("pcl_dnn") == 0)
  {
    assert(p->learning_rate_size() > 0);
    for(int i=0; i<p->learning_rate_size(); i++)
      temp.push_back(p->learning_rate(i));
    sp->setLearningRates(temp);

    temp.clear();
    assert(p->momentum_size() > 0);
    for(int i=0; i<p->momentum_size(); i++)
      temp.push_back(p->momentum(i));
    sp->setMomentums(temp);

    temp.clear();
    assert(p->weight_decay_size() > 0);
    for(int i=0; i<p->weight_decay_size(); i++)
      temp.push_back(p->weight_decay(i));
    sp->setWeightDecays(temp);

    assert(p->lr_change_epochs_size() > 0);
    for(int i=0; i<p->lr_change_epochs_size(); i++)
      itemp.push_back(p->lr_change_epochs(i));
    sp->setLRChangeEpochs(itemp);
  }
  else // all other policy types implemented via formula
  {
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
  }

  assert(p->max_epochs() >= 1);
  sp->setEpochs(p->max_epochs());
  assert(p->test_epoch() >= 1);
  sp->setTestEpoch(p->test_epoch());

  sp->setSolverType(p->type());

  sp->setGlobalFlag(p->global());

  return sp;
}

class SolverNode : public MLNode
{
  public:

    SolverNode(SolverParams* p, MLEngine* e);
    virtual ~SolverNode(void) {}

    void applyUpdate(float *blob, float *inc, float *grad, int size, float lr_mult, float decay_mult);
    void applyUpdate(float *blob, float *inc, float *grad, int size, float* lr_mult, float* decay_mult);
    bool getGlobalFlag() { return global_; }

  protected:
    vector<float> lr_, momentum_, decay_;
    vector<int> lrcepochs_, stepvalues_;
    int epochs_, test_epoch_, step_size_, max_iter_;
    int stepidx_, warmup_max_epoch_;
    bool global_;
    string solver_type_, lr_policy_;
    map<int, vector<float>> hpmap_;
    float base_lr_, lrval_, mval_, decayval_;
    float gamma_, power_, warmup_lr_;
    float mc_, mc1_, mc2_, prev_lrval_=-1, prev_lrval_1_=-1;
    MLEngine *eptr_;
};
