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

void SolverNode::convert_bf16_f32(libxsmm_bfloat16 **in, float** out, int len)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = omp_get_thread_num();
    int ntps = eptr_->get_num_threads()/NUM_NUMA_NODES;
    int n = tid/ntps;
    int ltid = tid - n*ntps;

    libxsmm_bfloat16 *inp = in[n];
    float *outp = out[n];

    int jobs = (len % ntps == 0) ? len/ntps : len/ntps + 1;
    int tb = (ltid*jobs < len) ? ltid*jobs : len;
    int te = ((ltid+1)*jobs < len) ? (ltid+1)*jobs : len;

    for (int i = tb; i < te; i+=16 ) {
      __m256i vbfp16    = _mm256_loadu_si256( (const __m256i*)(inp+i) );
      __m512  vfp32     = gxm_bfp16_to_fp32_avx512f( vbfp16 );
      _mm512_storeu_ps( outp+i, vfp32 );
    }
  }
}

void SolverNode::convert_bf16_f32(libxsmm_bfloat16 *in, float* out, int len)
{
  int i;

#ifdef _OPENMP
#pragma omp parallel  for private(i)
#endif
  for (i = 0; i < len; i+=16 ) {
    __m256i vbfp16    = _mm256_loadu_si256( (const __m256i*)(in+i) );
    __m512  vfp32     = gxm_bfp16_to_fp32_avx512f( vbfp16 );
    _mm512_storeu_ps( out+i, vfp32 );
  }
}

void SolverNode::applyUpdate(float **blob, float **inc, void **grad, int s, float** lr_mult, float** decay_mult, string tensorType)
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

  if(tensorType=="WEIGHT" && data_type_ == BF16)
  {
    for(int n=0; n<NUM_NUMA_NODES; n++)
      if(tmp_grad[n] == NULL)
        tmp_grad[n] = (float*)libxsmm_aligned_malloc(s*sizeof(float), 2097152);

    convert_bf16_f32((libxsmm_bfloat16**)grad, tmp_grad, s);
  }

  int sn = s/NUM_NUMA_NODES;
  float **wgrad_ptr = (tensorType == "WEIGHT" && data_type_ == BF16) ? tmp_grad : (float**)grad;

#ifndef USE_MLSL

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = omp_get_thread_num();
    int ntps = eptr_->get_num_threads()/NUM_NUMA_NODES;
    int n = tid/ntps;
    int ltid = tid - n*ntps;

    int jobs = (sn % ntps == 0) ? (sn/ntps) : (sn/ntps) + 1;
    int tb = (ltid * jobs < sn) ? (ltid * jobs) : sn;
    int te = (ltid + 1)*jobs < sn ? (ltid + 1)*jobs : sn;

    float *wgp = (wgrad_ptr[n]+n*sn);

    for(int nn=0; nn<NUM_NUMA_NODES; nn++)
    {
      if(n == nn) continue;

      float *rgp = (wgrad_ptr[nn]+n*sn);

#pragma omp simd
      for(int i=tb; i<te; i++)
        wgp[i] += rgp[i];
    }
  }
#else

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = omp_get_thread_num();
    int ntps = eptr_->get_num_threads()/NUM_NUMA_NODES;
    int n = tid/ntps;
    if(n != 0)
    {
      int ltid = tid - n*ntps;

      int jobs = (sn % ntps == 0) ? (sn/ntps) : (sn/ntps) + 1;
      int tb = (ltid * jobs < sn) ? (ltid * jobs) : sn;
      int te = (ltid + 1)*jobs < sn ? (ltid + 1)*jobs : sn;

      float *wgp = wgrad_ptr[0]+n*sn;
      float *rgp = wgrad_ptr[n]+n*sn;

#pragma omp simd
      for(int i=tb; i<te; i++)
        rgp[i] = wgp[i];
    }
  }
#endif

  if(solver_type_.compare("SGD") == 0)
  {
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      int ntps = eptr_->get_num_threads()/NUM_NUMA_NODES;
      int n = tid/ntps;
      int ltid = tid - n*ntps;

      float *blobp = blob[n] + n*sn;
      float *incp = inc[n] + n*sn;
      float *lrp = lr_mult[n] + n*sn;
      float *dcp = decay_mult[n] + n*sn;

      int jobs = (sn % ntps == 0) ? (sn / ntps) : (sn / ntps) + 1;
      int tb = (ltid * jobs < sn) ? (ltid * jobs) : sn;
      int te = (ltid + 1)*jobs < sn ? (ltid + 1)*jobs : sn;

#pragma omp barrier
#pragma omp simd
      for(int i=tb; i<te; i++)
      {
        incp[i] = mval_*incp[i] + lrval_ * lrp[i] * ((wgrad_ptr[n]+n*sn)[i] + decayval_ * dcp[i] * blobp[i]);
        blobp[i] = blobp[i] - incp[i];
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
      int ntps = eptr_->get_num_threads()/NUM_NUMA_NODES;
      int n = tid/ntps;
      int ltid = tid - n*ntps;

      float *blobp = blob[n] + n*sn;
      float *incp = inc[n] + n*sn;
      float *lrp = lr_mult[n] + n*sn;
      float *dcp = decay_mult[n] + n*sn;

      int jobs = (sn % ntps == 0) ? (sn / ntps) : (sn / ntps) + 1;
      int tb = (ltid * jobs < sn) ? (ltid * jobs) : sn;
      int te = (ltid + 1)*jobs < sn ? (ltid + 1)*jobs : sn;

#pragma omp barrier
#pragma omp simd
      for(int i=tb; i<te; i++)
      {
        incp[i] = mval_*mc_*incp[i] + lrval_ * lrp[i] * ((wgrad_ptr[n]+n*sn)[i] + decayval_ * dcp[i] * blobp[i]);
        blobp[i] = blobp[i] - incp[i];
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
      int ntps = eptr_->get_num_threads()/NUM_NUMA_NODES;
      int n = tid/ntps;
      int ltid = tid - n*ntps;

      int jobs = (sn % ntps == 0) ? (sn / ntps) : (sn / ntps) + 1;
      int tb = (ltid * jobs < sn) ? (ltid * jobs) : sn;
      int te = (ltid + 1)*jobs < sn ? (ltid + 1)*jobs : sn;

      float *bp = blob[n] + n*sn;
      float *incp = inc[n] + n*sn;
      float *lrp = lr_mult[n] + n*sn;
      float *dcp = decay_mult[n] + n*sn;
      float *wgp = wgrad_ptr[n] + n*sn;

#pragma omp simd
      for(int i=tb; i<te; i++)
      {
        float tinc = incp[i];
        incp[i] = mval_*mc1_*tinc + lrval_ * lrp[i] * (wgp[i] + decayval_ * dcp[i] * bp[i]);
        tinc = (1 + mval_*mc1_) * incp[i] - mval_*mc2_*tinc;
        bp[i] = bp[i] - tinc;
      }
    }
  }
  else if(solver_type_.compare("ADAGRAD") == 0)
  {
  }

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = omp_get_thread_num();
    int ntps = eptr_->get_num_threads()/NUM_NUMA_NODES;
    int n = tid/ntps;
    int ltid = tid - n*ntps;

    int jobs = (sn % ntps == 0) ? (sn / ntps) : (sn / ntps) + 1;
    int tb = (ltid * jobs < sn) ? (ltid * jobs) : sn;
    int te = (ltid + 1)*jobs < sn ? (ltid + 1)*jobs : sn;

    for(int nn=0; nn<NUM_NUMA_NODES; nn++)
    {
      if(n == nn) continue;

      float *wgp = (wgrad_ptr[n]+nn*sn);
      float *bp = (blob[n]+nn*sn);
      float *rbp = (blob[nn]+nn*sn);

#pragma vector nontemporal
#pragma omp simd
      for(int i=tb; i<te; i++)
        bp[i] = rbp[i];
    }
  }
}

void SolverNode::applyUpdate(float **blob, float **inc, void **grad, int s, float lr_mult, float decay_mult, string tensorType)
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

#ifdef BF16_MLSL
  if(tensorType=="WEIGHT" && data_type_ == BF16)
  {
    for(int n=0; n<NUM_NUMA_NODES; n++)
      if(tmp_grad[n] == NULL)
        tmp_grad[n] = (float*)libxsmm_aligned_malloc(s*sizeof(float), 2097152);

    convert_bf16_f32((libxsmm_bfloat16**)grad, tmp_grad, s);
  }

  float **wgrad_ptr = (tensorType == "WEIGHT" && data_type_ == BF16) ? tmp_grad : (float**)grad;
#else
  float **wgrad_ptr = (float**)grad;
#endif

  int sn = s/NUM_NUMA_NODES;

#ifndef USE_MLSL

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = omp_get_thread_num();
    int ntps = eptr_->get_num_threads()/NUM_NUMA_NODES;
    int n = tid/ntps;
    int ltid = tid - n*ntps;

    int jobs = (sn % ntps == 0) ? (sn/ntps) : (sn/ntps) + 1;
    int tb = (ltid * jobs < sn) ? (ltid * jobs) : sn;
    int te = (ltid + 1)*jobs < sn ? (ltid + 1)*jobs : sn;

    float *wgp = (wgrad_ptr[n]+n*sn);

    for(int nn=0; nn<NUM_NUMA_NODES; nn++)
    {
      if(n == nn) continue;

      float *rgp = (wgrad_ptr[nn]+n*sn);

#pragma omp simd
      for(int i=tb; i<te; i++)
        wgp[i] += rgp[i];
    }
  }
#else

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = omp_get_thread_num();
    int ntps = eptr_->get_num_threads()/NUM_NUMA_NODES;
    int n = tid/ntps;
    if(n != 0)
    {
      int ltid = tid - n*ntps;

      int jobs = (sn % ntps == 0) ? (sn/ntps) : (sn/ntps) + 1;
      int tb = (ltid * jobs < sn) ? (ltid * jobs) : sn;
      int te = (ltid + 1)*jobs < sn ? (ltid + 1)*jobs : sn;

      float *wgp = wgrad_ptr[0]+n*sn;
      float *rgp = wgrad_ptr[n]+n*sn;

#pragma vector nontemporal
#pragma omp simd
      for(int i=tb; i<te; i++)
        rgp[i] = wgp[i];
    }
  }
#endif

  if(solver_type_.compare("SGD") == 0)
  {
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      int ntps = eptr_->get_num_threads()/NUM_NUMA_NODES;
      int n = tid/ntps;
      int ltid = tid - n*ntps;

      float *blobp = blob[n] + n*sn;
      float *incp = inc[n] + n*sn;

      int jobs = (sn % ntps == 0) ? (sn / ntps) : (sn / ntps) + 1;
      int tb = (ltid * jobs < sn) ? (ltid * jobs) : sn;
      int te = (ltid + 1)*jobs < sn ? (ltid + 1)*jobs : sn;

#pragma omp barrier
#pragma omp simd
      for(int i=tb; i<te; i++)
      {
        incp[i] = mval_*incp[i] + lrval_ * lr_mult * ((wgrad_ptr[n]+n*sn)[i] + decayval_ * decay_mult * blobp[i]);
        blobp[i] = blobp[i] - incp[i];
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
      int ntps = eptr_->get_num_threads()/NUM_NUMA_NODES;
      int n = tid/ntps;
      int ltid = tid - n*ntps;

      float *blobp = blob[n] + n*sn;
      float *incp = inc[n] + n*sn;

      int jobs = (sn % ntps == 0) ? (sn / ntps) : (sn / ntps) + 1;
      int tb = (ltid * jobs < sn) ? (ltid * jobs) : sn;
      int te = (ltid + 1)*jobs < sn ? (ltid + 1)*jobs : sn;

#pragma omp barrier
#pragma omp simd
      for(int i=tb; i<te; i++)
      {
        incp[i] = mval_*mc_*incp[i] + lrval_ * lr_mult * ((wgrad_ptr[n]+n*sn)[i] + decayval_ * decay_mult * blobp[i]);
        blobp[i] = blobp[i] - incp[i];
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
      int ntps = eptr_->get_num_threads()/NUM_NUMA_NODES;
      int n = tid/ntps;
      int ltid = tid - n*ntps;

      int jobs = (sn % ntps == 0) ? (sn / ntps) : (sn / ntps) + 1;
      int tb = (ltid * jobs < sn) ? (ltid * jobs) : sn;
      int te = (ltid + 1)*jobs < sn ? (ltid + 1)*jobs : sn;

      float *incp = (inc[n]+n*sn);
      float *wgp = (wgrad_ptr[n]+n*sn);
      float *bp = (blob[n]+n*sn);

#pragma omp simd
      for(int i=tb; i<te; i++)
      {
        float tinc = incp[i];
        incp[i] = mval_*mc1_*tinc + lrval_ * lr_mult * (wgp[i] + decayval_ * decay_mult * bp[i]);
        tinc = (1 + mval_*mc1_) * incp[i] - mval_*mc2_*tinc;
        bp[i] = bp[i] - tinc;
      }
    }
  }
  else if(solver_type_.compare("ADAGRAD") == 0)
  {
  }

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = omp_get_thread_num();
    int ntps = eptr_->get_num_threads()/NUM_NUMA_NODES;
    int n = tid/ntps;
    int ltid = tid - n*ntps;

    int jobs = (sn % ntps == 0) ? (sn / ntps) : (sn / ntps) + 1;
    int tb = (ltid * jobs < sn) ? (ltid * jobs) : sn;
    int te = (ltid + 1)*jobs < sn ? (ltid + 1)*jobs : sn;

    for(int nn=0; nn<NUM_NUMA_NODES; nn++)
    {
      if(n == nn) continue;

      float *bp = (blob[n]+nn*sn);
      float *rbp = (blob[nn]+nn*sn);

#pragma vector nontemporal
#pragma omp simd
      for(int i=tb; i<te; i++)
        bp[i] = rbp[i];
    }
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
    if(tmp_grad[0] == NULL)
      tmp_grad[0] = (float*)libxsmm_aligned_malloc(s*sizeof(float), 2097152);
    convert_bf16_f32((libxsmm_bfloat16*)grad, tmp_grad[0], s);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<s/16; i++)
#pragma omp simd
      for(int j=0; j<16; j++)
        ((libxsmm_bfloat16*)grad)[i*16+j] = 0;

    wgrad_ptr = tmp_grad[0];
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
        inc[i] = mval_*mc1_*tinc + lrval_ * lr_mult[i] * (wgrad_ptr[i] + decayval_ * decay_mult[i] * blob[i]);
        tinc = (1 + mval_*mc1_) * inc[i] - mval_*mc2_*tinc;
        blob[i] = blob[i] - tinc;
        wgrad_ptr[i] = 0.0;
      }
    }
  }
  else if(solver_type_.compare("ADAGRAD") == 0)
  {}
}

void SolverNode::applyUpdate(float *blob, float *inc, void *grad, int s, float lr_mult, float decay_mult, string tensorType)
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
    if(tmp_grad[0] == NULL)
      tmp_grad[0] = (float*)libxsmm_aligned_malloc(s*sizeof(float), 2097152);
    convert_bf16_f32((libxsmm_bfloat16*)grad, tmp_grad[0], s);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<s/16; i++)
#pragma omp simd
      for(int j=0; j<16; j++)
        ((libxsmm_bfloat16*)grad)[i*16+j] = 0;

    wgrad_ptr = tmp_grad[0];
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
        inc[i] = mval_*inc[i] + lrval_ * lr_mult * (wgrad_ptr[i] + decayval_ * decay_mult * blob[i]);
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
        inc[i] = mval_*mc_*inc[i] + lrval_ * lr_mult * (wgrad_ptr[i] + decayval_ * decay_mult * blob[i]);
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
        inc[i] = mval_*mc1_*tinc + lrval_ * lr_mult * (wgrad_ptr[i] + decayval_ * decay_mult * blob[i]);
        tinc = (1 + mval_*mc1_) * inc[i] - mval_*mc2_*tinc;
        blob[i] = blob[i] - tinc;
        wgrad_ptr[i] = 0.0;
      }
    }
  }
  else if(solver_type_.compare("ADAGRAD") == 0)
  {}
}
