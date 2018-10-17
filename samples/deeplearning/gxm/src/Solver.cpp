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


#include <string>
#include "Solver.hpp"

#define VLEN 16

using namespace std;
using namespace gxm;

SolverNode::SolverNode(SolverParams* p, MLEngine* e): MLNode(p, e)
{
  lr_policy_ = p->getLRPolicy();
  base_lr_ = p->getLearningRate();
  warmup_lr_ = p->getWarmupLR();
  mval_ = p->getMomentum();
  decayval_ = p->getWeightDecay();
  power_ = p->getPower();
  gamma_ = p->getGamma();
  step_size_ = p->getStepSize();
  max_iter_ = p->getMaxIter();
  stepvalues_ = p->getStepValues();
  warmup_max_epoch_ = p->getWarmupEpochs();

  stepidx_ = 0;
  epochs_ = p->getEpochs();
  test_epoch_ = p->getTestEpoch();
  solver_type_ = p->getSolverType();
  global_ = p->getGlobalFlag();
  data_type_ = p->getDataType();

  eptr_ = e;
}

void SolverNode::convert_bf16_f32(libxsmm_bfloat16 *in, float* out, int len)
{
  int i;

#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
  for ( i = 0; i < len; i+=16 ) {
    __m256i vbfp16    = _mm256_loadu_si256( (const __m256i*)(in+i) );
    __m512  vfp32     = gxm_bfp16_to_fp32_avx512f( vbfp16 );
    _mm512_storeu_ps( out+i, vfp32 );
  }
}

void SolverNode::applyUpdate(float *blob, float *inc, void *grad, int s, float* lr_mult, float* decay_mult, string tensorType)
{
  int iter = eptr_->get_current_batch() + eptr_->get_num_train_batches() * eptr_->get_current_epoch();
  int warmup_max_iter = eptr_->get_num_train_batches() * warmup_max_epoch_; // Warm-up

  if(eptr_->get_current_epoch() < warmup_max_epoch_)
    lrval_ = (iter*base_lr_ + (warmup_max_iter - iter) * warmup_lr_)/warmup_max_iter;
  else if(lr_policy_.compare("fixed") == 0)
    lrval_ = base_lr_;
  else if(lr_policy_.compare("step") == 0)
    lrval_ = base_lr_ * pow(gamma_, floor((double)iter/(double)step_size_));
  else if(lr_policy_.compare("poly") == 0)
    lrval_ = base_lr_ * pow(((float)1. - ((float)iter/(float)max_iter_)), power_);
  else if(lr_policy_.compare("inv") == 0)
    lrval_ = base_lr_ * pow((1 + gamma_ * iter), (-power_));
  else if(lr_policy_.compare("multistep") == 0)
  {
    if(stepidx_ < stepvalues_.size() && iter > stepvalues_[stepidx_])
      stepidx_++;
    lrval_ = base_lr_ * pow(gamma_, (float)stepidx_);
  }

  eptr_->set_learning_rate(lrval_);

  float *wgrad_ptr;
  if(tensorType=="WEIGHT" && data_type_ == BF16)
  {
    if(tmp_grad == NULL)
      tmp_grad = (float*)libxsmm_aligned_malloc(s*sizeof(float), 2097152);
    convert_bf16_f32((libxsmm_bfloat16*)grad, tmp_grad, s);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<s/16; i++)
#pragma omp simd
      for(int j=0; j<16; j++)
        ((libxsmm_bfloat16*)grad)[i*16+j] = 0;

    wgrad_ptr = tmp_grad;
  }
  else if(tensorType=="WEIGHT" && data_type_ == FLOAT || tensorType=="BIAS")
    wgrad_ptr = (float*)grad;

  if(solver_type_.compare("SGD") == 0)
  {
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      int threads = omp_get_num_threads();
      int jobs = (s % threads == 0) ? (s / threads) : (s / threads) + 1;
      int tb = (tid * jobs < s) ? (tid * jobs) : s;
      int te = (tid + 1)*jobs < s ? (tid + 1)*jobs : s;

#pragma omp simd
      for(int i=tb; i<te; i++)
      {
        inc[i] = mval_*inc[i] + lrval_ * lr_mult[i] * (wgrad_ptr[i] + decayval_ * decay_mult[i] * blob[i]);
        blob[i] = blob[i] - inc[i];
        wgrad_ptr[i] = 0.0;
      }
    }
  }
  else if(solver_type_ == "SGD_MC")
  {
    mc_ = 1;
    if(eptr_->get_current_epoch() < warmup_max_epoch_)
    {
      if(prev_lrval_ != -1)
      {
        mc_ = lrval_/prev_lrval_;
        prev_lrval_ = lrval_;
      }
      else
        prev_lrval_ = lrval_;
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      int threads = omp_get_num_threads();
      int jobs = (s % threads == 0) ? (s / threads) : (s / threads) + 1;
      int tb = (tid * jobs < s) ? (tid * jobs) : s;
      int te = (tid + 1)*jobs < s ? (tid + 1)*jobs : s;

#pragma omp simd
      for(int i=tb; i<te; i++)
      {
        inc[i] = mval_*mc_*inc[i] + lrval_ * lr_mult[i] * (wgrad_ptr[i] + decayval_ * decay_mult[i] * blob[i]);
        blob[i] = blob[i] - inc[i];
        wgrad_ptr[i] = 0.0;
      }
    }
  }
  else if(solver_type_ == "NESTEROV")
  {
    mc1_ = 1;
    mc2_ = 1;
    if(eptr_->get_current_epoch() < warmup_max_epoch_)
    {
      if(prev_lrval_ != -1)
      {
        mc1_ = lrval_/prev_lrval_;
        if(prev_lrval_1_ != -1)
          mc2_ = prev_lrval_/prev_lrval_1_;
      }
      prev_lrval_1_ = prev_lrval_;
      prev_lrval_ = lrval_;
    }
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      int threads = omp_get_num_threads();
      int jobs = (s % threads == 0) ? (s / threads) : (s / threads) + 1;
      int tb = (tid * jobs < s) ? (tid * jobs) : s;
      int te = (tid + 1)*jobs < s ? (tid + 1)*jobs : s;

#pragma omp simd
      for(int i=tb; i<te; i++)
      {
        float tinc = inc[i];
        inc[i] = mval_*mc1_*inc[i] + lrval_ * lr_mult[i] * (wgrad_ptr[i] + decayval_ * decay_mult[i] * blob[i]);
        tinc = (1 + mval_*mc1_) * inc[i] - mval_*mc2_*tinc;
        blob[i] = blob[i] - tinc;
        wgrad_ptr[i] = 0.0;
      }
    }
  }
  else if(solver_type_.compare("ADAGRAD") == 0)
  {}
}
