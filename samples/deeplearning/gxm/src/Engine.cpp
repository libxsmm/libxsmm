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


#include <map>
#include "assert.h"
#include "proto/gxm.pb.h"
#include "Node.hpp"
#include "Engine.hpp"
#include "Conv.hpp"
#include "FullyConnected.hpp"
#include "FusedBNorm.hpp"
#include "DummyData.hpp"
#include "TypeList.hpp"

#define VLEN 16

using namespace std;
using namespace gxm;

int iter=0;

bool compare_task_bins(Task* first, Task* second)
{
  return (first->getMaxBin() < second->getMinBin());
}

void MLEngine::create_schedule(int mode)
{
  for(auto it=etg_[mode].begin(); it != etg_[mode].end(); it++)
  {
    Task* t = *it;
    vector<Task*> tp = t->getBackDepTasks();
    for(int i=0; i<tp.size(); i++) {
      string s = dynamic_cast<NNNode*>(tp[i]->getNode())->getNodeName();

      if(tp[i]->getBasicTaskId() == BASIC_TASK_FORW) {
        int maxbin = tp[i]->getMaxBin();
        if((maxbin == 0) || (maxbin > t->getMinBin()-1))
        {
          tp[i]->setMinBin(t->getMaxBin() - 1);
          tp[i]->setMaxBin(t->getMaxBin() - 1);
          etg_[mode].push_back(tp[i]);
#ifdef DEBUG
          printf("FP task %p (node %s), with bin %d pushed to etg_\n",tp[i], s.c_str(), tp[i]->getMaxBin());
#endif
        }
      }
    }
  }

  if(mode == TRAIN)
  {
    for(auto it=etg_[mode].begin(); it != etg_[mode].end(); it++)
    {
      Task* t = *it;
      vector<Task*> tp = t->getForwDepTasks();
      for(int i=0; i<tp.size(); i++)
      {
        string s = dynamic_cast<NNNode*>(tp[i]->getNode())->getNodeName();

        if(tp[i]->getBasicTaskId() != BASIC_TASK_FORW)
        {
          int maxbin = tp[i]->getMaxBin();
          if((maxbin == 0) || (maxbin < t->getMinBin()+1))
          {
            tp[i]->setMinBin(t->getMaxBin() + 1);
            tp[i]->setMaxBin(t->getMaxBin() + 1);
            etg_[mode].push_back(tp[i]);
#ifdef DEBUG
            if(tp[i]->getBasicTaskId() == BASIC_TASK_BACK)
              printf("BP task %p (node %s), with bin %d pushed to etg_\n",tp[i], s.c_str(), tp[i]->getMaxBin());
            else if(tp[i]->getBasicTaskId() == BASIC_TASK_WGRAD)
              printf("WU task %p (node %s), with bin %d pushed to etg_\n",tp[i], s.c_str(), tp[i]->getMaxBin());
            else if(tp[i]->getBasicTaskId() == BASIC_TASK_SOLVE)
              printf("SOLVE task %p (node %s), with bin %d pushed to etg_\n",tp[i], s.c_str(), tp[i]->getMaxBin());
#endif
          }
        }
      }
    }
  }
}

int MLEngine::find_in_nodeTypeList(string name)
{
  for(int i=0; i<numTypes; i++)
    if(nodeTypes[i].typeName.compare(name) == 0)
      return i;
  return -1;
}

bool MLEngine::register_tensor(string name, int type, Tensor* t)
{
  TensorPair tp;
  tp.name = name;
  tp.t = t;

  Iter it;

  switch(type)
  {
    case INPUT:
    case LABEL:
      it = inTList_.insert(inTList_.end(), tp);
      inTensorMap_[name] = it;
      break;

    case ACT:
      it = outTList_.insert(outTList_.end(), tp);
      outTensorMap_[name] = it;
      break;

    case CONVWEIGHT:
    case FCWEIGHT:
      it = wTList_.insert(wTList_.end(), tp);
      weightTensorMap_[name] = it;
      break;

    case CONVBIAS:
    case FCBIAS:
    case BNORMSCALE:
    case BNORMSHIFT:
      it = biasTList_.insert(biasTList_.end(), tp);
      biasTensorMap_[name] = it;
      break;

    case BNORMMEAN:
    case BNORMRSTDEV:
      it = statsTList_.insert(statsTList_.end(), tp);
      statsTensorMap_[name] = it;
      break;
  }
  return true;
}

Tensor* MLEngine::get_tensor(string name, int type)
{
  Iter it = defTList_.end();

  switch(type)
  {
    case INPUT:
    case LABEL:
      it = inTensorMap_[name];
      break;

    case ACT:
      it = outTensorMap_[name];
      break;

    case CONVWEIGHT:
    case FCWEIGHT:
      it = weightTensorMap_[name];
      break;

    case CONVBIAS:
    case FCBIAS:
    case BNORMSCALE:
    case BNORMSHIFT:
      it = biasTensorMap_[name];
      break;

    case BNORMMEAN:
    case BNORMRSTDEV:
      it = statsTensorMap_[name];
      break;
  }

  if(it == defTList_.end())
    return NULL;

  TensorPair tp = *it;
  return tp.t;
}

void MLEngine::optimize_schedule(int mode)
{
  etg_[mode].sort(compare_task_bins);
  etg_[mode].erase(std::stable_partition(etg_[mode].begin(), etg_[mode].end(), dupChecker_()), etg_[mode].end());
  etg_[mode].unique();
}

void MLEngine::clear_history(TensorList L)
{
  int buftype = HISTORY;

  for(Iter it=L.begin(); it != L.end(); it++)
  {
    Tensor* t = it->t;
    TensorBuf *tBuf;
    bool found = false;
    for(int index=0; index<t->getNumDataBuffers(); index++)
    {
      tBuf = t->getBuf(index);
      if(tBuf->getBufferType() == buftype)
      {
        found = true;
        break;
      }
    }
    if(!found) continue;

    long long int bytes = tBuf->getBufferSize();
    int dtype = tBuf->getDataType();

    if(dtype == DT_FLOAT)
    {
      float *fp = (float*)(tBuf->getBuffer());
      #ifdef _OPENMP
      #pragma omp parallel for
      #endif
      for(int i=0; i<bytes/sizeof(float); i++)
        fp[i] = 0.f;
    }
    else if(dtype == DT_DFP16)
    {
      short int* sp = (short int*)(tBuf->getBuffer());
      #ifdef _OPENMP
      #pragma omp parallel for
      #endif
      for(int i=0; i<bytes/sizeof(short int); i++)
        sp[i] = 0;
    }
  }
}

void MLEngine::checkpoint(TensorList L)
{
  int buftype = DATA;

  for(Iter it=L.begin(); it != L.end(); it++)
  {
    Tensor* t = it->t;
    TensorBuf *tBuf;
    bool found=false;

    for(int index=0; index<t->getNumDataBuffers(); index++)
    {
      tBuf = t->getBuf(index);
      if(tBuf->getBufferType() == buftype)
      {
        found = true;
        break;
      }
    }
    if(!found) continue;

    int tenType = t->getType();
    string n = checkpoint_dir_ + "/" + t->getTensorName();

    if((tenType == CONVWEIGHT) || (tenType == CONVBIAS))
    {
      ConvNode* cn = dynamic_cast<ConvNode*>(t->getOwner());
      cn->Checkpoint(tBuf, n, checkpoint_format_);
    }
    else if((tenType == FCWEIGHT) || (tenType == FCBIAS))
    {
      FCNode* fn = dynamic_cast<FCNode*>(t->getOwner());
      fn->Checkpoint(tBuf, n, checkpoint_format_);
    }
    else if((tenType == BNORMSCALE) || (tenType == BNORMSHIFT) || (tenType == BNORMMEAN) || (tenType == BNORMRSTDEV))
    {
      FusedBNormNode* bn = dynamic_cast<FusedBNormNode*>(t->getOwner());
      bn->Checkpoint(tBuf, n, checkpoint_format_);
    }
  }
}

void MLEngine::read_checkpoint_file(TensorBuf* tBuf, string filename, string format)
{
  long long int bytes = tBuf->getBufferSize();
  int dtype = tBuf->getBufferType();

  void* ptr = tBuf->getBuffer();

  FILE* f;
  if(format.compare("binary") == 0)
  {
    f = fopen(filename.c_str(), "rb");
    assert(f != NULL);
    size_t b = fread(ptr, 1, bytes, f);
    assert((long long int)b == bytes);
  }
  else
  {
    printf("Reading from %s\n",filename.c_str());
    f = fopen(filename.c_str(), "r");
    assert(f != NULL);
    if(dtype == DT_FLOAT)
    {
      float* p = (float*)ptr;
      for(int i=0; i < bytes/sizeof(float); i++)
        fscanf(f, "%f", &p[i]);
    }
    else if(dtype == DT_DFP16)
    {
      short int* p = (short int*)ptr;
      for(int i=0; i < bytes/sizeof(short int); i++)
        fscanf(f, "%d", &p[i]);
    }
  }
  fclose(f);
}

void MLEngine::load_checkpoint(TensorList L, string format)
{
  int buftype = DATA;
  TensorBuf* tBuf;

  for(Iter it=L.begin(); it != L.end(); it++)
  {
    Tensor* t = it->t;
    int tenType = t->getType();
    if((tenType != CONVWEIGHT) && (tenType != CONVBIAS) && (tenType != FCWEIGHT) && (tenType != FCBIAS))
      if((tenType != BNORMSCALE) && (tenType != BNORMSHIFT) && (tenType != BNORMMEAN) && (tenType != BNORMRSTDEV))
      continue;

    bool found = false;
    for(int index=0; index<t->getNumDataBuffers(); index++)
    {
      tBuf = t->getBuf(index);
      if(tBuf->getBufferType() == buftype)
      {
        found = true;
        break;
      }
    }
    if(!found) continue;

    string n = checkpoint_dir_ + "/" + t->getTensorName();
    size_t pos;
    while((pos = n.find("/", 10)) != n.npos)
      n.replace(pos, 1, 1, '_');
    read_checkpoint_file(tBuf, n, format);
  }
}

void MLEngine::canary_check(void* ptr, vector<int>& cp, int nc)
{
  if(ptr == NULL)
  {
    printf("FATAL: NULL pointer to buffer\n");
    //exit(1);
  }

  int *p = (int*)ptr;
  for(int i=0; i<START_GUARD_BAND/sizeof(int); i++)
  {
   // printf("p[%d] = %x\n",i, p[i]);
    if(p[i] != 0x7f7f7f7f)
    {
      printf("Fatal: canary value overwritten at %d in buffer at %p\n",i, ptr);
      //exit(1);
    }
  }

  void *vp = (void*)(ptr + START_GUARD_BAND);

  for(int i=0; i<nc; i++)
  {
    int next = cp[i];
    vp = (void*)(vp + next);
    int *pp = (int*)vp;
    for(int j=0; j<END_GUARD_BAND/sizeof(int); j++)
    {
     // printf("pp[%d] = %x\n",j, pp[j]);
      if(pp[j] != 0x7f7f7f7f)
      {
        printf("Fatal: canary value overwritten at %d in buffer at %p\n",j,pp);
        //exit(1);
      }
    }
    vp += END_GUARD_BAND;
  }
}

void MLEngine::quantize_and_transpose_weights(TensorList L)
{
  int buftype = DATA;

  for(Iter it=L.begin(); it != L.end(); it++)
  {
    Tensor* t = it->t;
    TensorBuf *tBuf;
    bool found=false;

    for(int index=0; index<t->getNumDataBuffers(); index++)
    {
      tBuf = t->getBuf(index);
      if(tBuf->getBufferType() == buftype)
      {
        found = true;
        break;
      }
    }
    if(!found) continue;

    if(tBuf->getLPBuffer() == NULL) continue;

    Shape* s = t->getShape();
    assert(s->ndims == 4);
    int welem = s->dims[0] * s->dims[1] * s->dims[2] * s->dims[3];

    float* fptr = (float*)tBuf->getBuffer();
    void* lp_weight = tBuf->getLPBuffer();
    unsigned char scf_filter;
    libxsmm_dnn_quantize_fil(fptr, (short*)lp_weight, s->dims[0], s->dims[1], s->dims[2], s->dims[3], 16, 8, 16, 16, 2, 2, &scf_filter, LIBXSMM_DNN_QUANT_FPHW_ROUND);
    tBuf->setLPSF(scf_filter);
  }
}

void MLEngine::run(int mode)
{
  if(mode == TRAIN)
  {
    if(load_from_checkpoint_)
    {
      FILE *f = fopen("checkpoint", "r");
      if(f != NULL)
      {
        fscanf(f, "%d %f\n",&current_epoch_, &lr_);
        fclose(f);
      }
      else
        printf("No checkpoint state file to read\n");

      current_epoch_++;
      load_checkpoint(wTList_, checkpoint_format_);
      load_checkpoint(biasTList_, checkpoint_format_);
      load_checkpoint(statsTList_, checkpoint_format_);
      load_from_checkpoint_ = false;
    }

    fflush(stdout);

#ifdef USE_MLSL
     data_parallelism->Barrier(MLSL::GT_DATA);
#endif

    // current_epoch_ is set in create() function or by checkpoint code above
    for(; current_epoch_ < num_epochs_; current_epoch_++)
    {
      // Tell data node that it should use training data
      exec_mode_ = TRAIN;
      if(global_node_id_ == 0)
      {
        printf("===========================================\n");
        printf("TRAIN mode, epoch %d, training batches %d\n", current_epoch_, num_train_batches_);
        printf("===========================================\n");
      }

      // Run training network for an epoch
      struct timeval tvs, tve, tvts, tvte, tvis, tvie;
      double fbtime, runtime = 0;

      for(; current_batch_<num_train_batches_; current_batch_++)
      {
        //iter = current_batch_;

        if(global_node_id_ == 0)
          printf("Executing batch number %d\n",current_batch_);

        gettimeofday(&tvs, NULL);

        for(auto it = etg_[TRAIN].begin(); it != etg_[TRAIN].end(); it++)
        {
#ifdef TIMING
          gettimeofday(&tvts, NULL);
#endif

          (*it)->invoke();

#ifdef TIMING
          gettimeofday(&tvte, NULL);
          double tasktime = (tvte.tv_sec*1e6 + tvte.tv_usec) - (tvts.tv_sec*1e6 + tvts.tv_usec);
          NNNode *nn = dynamic_cast<NNNode*>((*it)->getNode());
          if(global_node_id_ == 0)
            printf("Node %s (task %d) time = %f ms\n",nn->getNodeName().c_str(), (*it)->getBasicTaskId(), tasktime/1000);
#endif
        }

        if(solver_->getGlobalFlag())
        {
#ifdef TIMING
          gettimeofday(&tvis, NULL);
#endif

          solver_->applyUpdate((float*)weight_buf_, (float*)winc_buf_, (float*)wdiff_buf_, total_weights_, wt_lr_mult_, wt_decay_mult_);

          solver_->applyUpdate((float*)bias_buf_, (float*)biinc_buf_, (float*)bidiff_buf_, total_biases_, bias_lr_mult_, bias_decay_mult_);

#ifdef TIMING
          gettimeofday(&tvie, NULL);
          double sgdtime = (tvie.tv_sec + tvie.tv_usec*1e-6) - (tvis.tv_sec + tvis.tv_usec*1e-6);
          printf("global sgd time: %f ms\n",sgdtime*1000);
#endif
        }

        if(data_type_ == DT_DFP16)
          quantize_and_transpose_weights(wTList_);

        gettimeofday(&tve, NULL);
        fbtime = (tve.tv_sec + tve.tv_usec*1e-6) - (tvs.tv_sec + tvs.tv_usec*1e-6);
        if(global_node_id_ == 0)
          printf("Fwd-Bwd time: %f ms\n",fbtime*1000);
        runtime += fbtime;

#ifdef CANARY_CHECK
        canary_check(input_buf_, input_can_ptr, ic);
        canary_check(fact_buf_, fact_can_ptr, fac);
        canary_check(weight_buf_, wt_can_ptr, wtc);
        canary_check(bias_buf_, bias_can_ptr, bic);

        canary_check(bact_buf_, bact_can_ptr, bac);
        canary_check(wdiff_buf_, wdiff_can_ptr, wdc);
        canary_check(winc_buf_, winc_can_ptr, wic);
        canary_check(bidiff_buf_, bidiff_can_ptr, bidc);
        canary_check(biinc_buf_, biinc_can_ptr, biic);
#endif
      }

      current_batch_ = 0;

      printf("Average Training time = %f seconds\n",runtime/num_train_batches_);
      if(runtime > 0)
        printf("Training throughput = %f images/s\n",(float)(batch_size_*num_train_batches_)/runtime);

      // Checkpoint weights and biases
#if !defined(DUMP_ACT_DATA) && !defined(DUMP_WT_DATA)
      if(global_node_id_ == 0)
      {
        checkpoint(wTList_);
        checkpoint(biasTList_);
        checkpoint(statsTList_);
        FILE* f = fopen("checkpoint", "w");
        if(f != NULL)
        {
          fprintf(f, "%d %10g\n",current_epoch_, lr_);
          fclose(f);
        }
      }
#ifdef USE_MLSL
      data_parallelism->Barrier(MLSL::GT_DATA);
#endif

#endif

#if 1
      clear_history(wTList_);
      clear_history(biasTList_);
#endif

      // Tell data node that it should use test data

#if !defined(DUMP_ACT_DATA) && !defined(DUMP_WT_DATA)
#if 1
      exec_mode_ = TEST;

      if(global_node_id_ == 0)
      {
        printf("===========================================\n");
        printf("TEST mode, testing batches %d\n", num_test_batches_);
        printf("===========================================\n");
      }

      // Run validation network at end of each epoch
      for(; current_batch_<num_test_batches_; current_batch_++)
      {
        for(int v=0; v<num_test_views_; v++)
          for(auto it = etg_[TEST].begin(); it != etg_[TEST].end(); it++)
            (*it)->invoke();
      }
#endif

      current_batch_ = 0;

#ifdef CANARY_CHECK
        canary_check(input_buf_, input_can_ptr, ic);
        canary_check(fact_buf_, fact_can_ptr, fac);
        canary_check(weight_buf_, wt_can_ptr, wtc);
        canary_check(bias_buf_, bias_can_ptr, bic);
#endif
#endif
    }
  }
  else if(mode == TEST)
  {
    // Run validation or test network when command-line mode is set to "test"
    for(int b=0; b<num_test_batches_; b++)
    {
      for(auto it = etg_[TEST].begin(); it != etg_[TEST].end(); it++)
        (*it)->invoke();
    }
  }
}

void MLEngine::allocate_tensor_memory(Tensor* t, int buftype, void* buf_)
{
  bool found = false;
  TensorBuf* tBuf;
  for(int index=0; index<t->getNumDataBuffers(); index++)
  {
    tBuf = t->getBuf(index);
    if(tBuf->getBufferType() == buftype)
    {
      found = true;
      break;
    }
  }

  assert(found == true);

  int dtype = tBuf->getDataType();
  long long int size = tBuf->getBufferSize();
  if(dtype == DT_FLOAT)
  {
    buf_ = (float*)_mm_malloc(size, 64);
    tBuf->setBuffer((float*)buf_);
  }
  else if(dtype == DT_DFP16)
  {
    buf_ = (short int*)_mm_malloc(size, 64);
    tBuf->setBuffer((short int*)buf_);
  }
  else if(dtype == DT_INT)
  {
    buf_ = (int*)_mm_malloc(size, 64);
    tBuf->setBuffer((int*)buf_);
  }
}

void* MLEngine::allocate_memory(string tenType, TensorList L, int buftype, vector<int>& can_ptr, int* nc, long long int* bufsize, long long int* max, int machines)
{
  bool ttp = (tenType != "WEIGHT") & (tenType != "BIAS");

  long long int s = ttp ? START_GUARD_BAND : 0;
  TensorBuf* tBuf;
  int num_canaries = 0;

  float* lrptr, *decptr;

  // Get total buffer size required for tensors of type buftype
  for(Iter it=L.begin(); it != L.end(); it++)
  {
    Tensor* t = it->t;

    bool found = false;
    for(int i=0; i<t->getNumDataBuffers(); i++)
    {
      tBuf = t->getBuf(i);
      if(tBuf->getBufferType() == buftype)
      {
        found = true;
        break;
      }
    }
    if(!found) continue;

    long long int size = tBuf->getBufferSize();
    if(size > 0)
    {
      if(max != NULL)
        if(size > *max) *max = size;

      if(global_node_id_ == 0)
      {
        printf("Tensor %s needs %lld bytes for buffer %d\n", t->getTensorName().c_str(), size, buftype);
        fflush(stdout);
      }
      s += size;
      if(ttp)
        s += END_GUARD_BAND;

      if(ttp)
        num_canaries++;
    }
  }

  if(solver_->getGlobalFlag())
  {
    if(tenType == "WEIGHT")
    {
      total_weights_ = s / sizeof(float);
      int factor = num_threads_ * VLEN;
      int nwt = (total_weights_ + factor - 1)/factor;
      total_weights_ = nwt * factor;

      s = total_weights_ * sizeof(float);
    }
    else if(tenType == "BIAS")
    {
      total_biases_ = s / sizeof(float);
      int factor = num_threads_ * VLEN;
      int nwt = (total_biases_ + factor - 1)/factor;
      total_biases_ = nwt * factor;

      s = total_biases_ * sizeof(float);
    }
  }

  // Total buffer size, including guard bands before and after each buffer (currntly 64 bytes long)
  *bufsize = s;

  // Number of guard bands in tensor; used for canary checking
  *nc = num_canaries;

  // Allocate memory
#ifdef USE_MLSL
  void* buf_ = (void*)MLSL::Environment::GetEnv().Alloc(s, 2097152);
#else
  void* buf_ = (void*)libxsmm_aligned_malloc(s, 2097152);
#endif

  printf("Tensor with buffers %d @ %p with total size %lld\n",buftype, buf_, s);
  fflush(stdout);

  if(buf_ != NULL)
  {
#ifndef USE_NUMA
    memset(buf_, 0, s);
#endif
  }
  else {
    printf("could not allocate tensor memory.. exiting\n");
    exit(-1);
  }

#if 1
  if(solver_->getGlobalFlag())
  {
    if(tenType == "WEIGHT" && buftype == DIFF)
    {
      wt_lr_mult_ = (float*)_mm_malloc(s, 64);
      wt_decay_mult_ = (float*)_mm_malloc(s, 64);
      if(wt_lr_mult_ != NULL)
      {
        memset(wt_lr_mult_, 0, s);
        lrptr = wt_lr_mult_;
      }
      else {
        printf("could not allocate lr_wt memory.. exiting\n");
        exit(-1);
      }
      if(wt_decay_mult_ != NULL)
      {
        memset(wt_decay_mult_, 0, s);
        decptr = wt_decay_mult_;
      }
      else {
        printf("could not allocate decay_wt memory.. exiting\n");
        exit(-1);
      }
    }
    else if(tenType == "BIAS" && buftype == DIFF)
    {
      bias_lr_mult_ = (float*)_mm_malloc(s, 64);
      bias_decay_mult_ = (float*)_mm_malloc(s, 64);
      if(bias_lr_mult_ != NULL)
      {
        memset(bias_lr_mult_, 0, s);
        lrptr = bias_lr_mult_;
      }
      else {
        printf("could not allocate lr_bias memory.. exiting\n");
        exit(-1);
      }
      if(bias_decay_mult_ != NULL)
      {
        memset(bias_decay_mult_, 0, s);
        decptr = bias_decay_mult_;
      }
      else {
        printf("could not allocate decay_bias memory.. exiting\n");
        exit(-1);
      }
    }
  }
#endif

  if(ttp)
    memset(buf_, CANARY, START_GUARD_BAND);

  long long int bytes=0;

  //Set up tensor buffer pointers
  void* ptr = ttp ? (void*)buf_ + START_GUARD_BAND : (void*)buf_;

  for(Iter it=L.begin(); it != L.end(); it++)
  {
    Tensor* t = it->t;

    bool found = false;
    for(int i=0; i<t->getNumDataBuffers(); i++)
    {
      tBuf = t->getBuf(i);
      if(tBuf->getBufferType() == buftype)
      {
        found = true;
        break;
      }
    }
    if(!found) continue;

    // Don't process Split nodes further for forward activations
    string nntype = dynamic_cast<NNNode*>(t->getOwner())->getNodeType();
    if(nntype.find("Split") != nntype.npos && buftype == DATA)
      continue;

    // Scrub or initialize buffers appropriately
    //
    bytes = tBuf->getBufferSize();
    assert(ptr+bytes <= buf_+s);

#ifndef USE_NUMA
    if(t->getType() == INPUT || t->getType() == ACT)
    {
      if(bytes > 0)
        memset(ptr, 0, bytes);
    }
#endif

    int dtype = tBuf->getDataType();

    // Set each node's tensor buffer pointers to the appropritate location in the global buffer
    tBuf->setBuffer(ptr);

    // If weight or bias tensor, call corresponding intialization function (for training only)
    if(!is_inference_only())
    {
      int tType = t->getType();
      if(tType == CONVWEIGHT)
      {
        ConvNode* cn = dynamic_cast<ConvNode*>(t->getOwner());
        assert(bytes > 0);
        cn->fillWeightBuffers(tBuf, buftype, bytes);
        if(solver_->getGlobalFlag())
          if(buftype == DIFF)
            cn->fillWeightMultipliers(lrptr, decptr, bytes/sizeof(float));
      }
      else if(tType == CONVBIAS)
      {
        ConvNode* cn = dynamic_cast<ConvNode*>(t->getOwner());
        assert(bytes > 0);
        cn->fillBiasBuffers(tBuf, buftype, bytes);
        if(solver_->getGlobalFlag())
          if(buftype == DIFF)
            cn->fillBiasMultipliers(lrptr, decptr, bytes/sizeof(float));
      }
      else if(tType == FCWEIGHT)
      {
        FCNode* fn = dynamic_cast<FCNode*>(t->getOwner());
        assert(bytes > 0);
        fn->fillWeightBuffers(tBuf, buftype, bytes, machines);
        if(solver_->getGlobalFlag())
          if(buftype == DIFF)
            fn->fillWeightMultipliers(lrptr, decptr, bytes/sizeof(float));
      }
      else if(tType == FCBIAS)
      {
        FCNode* fn = dynamic_cast<FCNode*>(t->getOwner());
        assert(bytes > 0);
        fn->fillBiasBuffers(tBuf, buftype, bytes);
        if(solver_->getGlobalFlag())
          if(buftype == DIFF)
            fn->fillBiasMultipliers(lrptr, decptr, bytes/sizeof(float));
      }
      else if((tType == BNORMSCALE) || (tType == BNORMSHIFT))
      {
        FusedBNormNode* bn = dynamic_cast<FusedBNormNode*>(t->getOwner());
        assert(bytes > 0);
        bn->fillBuffer(tBuf, buftype, bytes);
        if(solver_->getGlobalFlag())
          if(buftype == DIFF)
            bn->fillBiasMultipliers(lrptr, decptr, bytes/sizeof(float));
      }
      else if((tType == BNORMMEAN) || (tType == BNORMRSTDEV))
      {
        FusedBNormNode* bn = dynamic_cast<FusedBNormNode*>(t->getOwner());
        assert(bytes > 0);
        bn->fillBuffer(tBuf, buftype, bytes);
      }
    }

    if(bytes > 0)
    {
      ptr += bytes;

      if(solver_->getGlobalFlag())
      {
        if((tenType == "WEIGHT" || tenType == "BIAS") && buftype == DIFF)
        {
          lrptr += bytes/sizeof(float);
          decptr += bytes/sizeof(float);
        }
      }

      assert(ptr <= buf_ + s);

      // For canary checking
      if(ttp)
      {
        memset(ptr, CANARY, END_GUARD_BAND);
        can_ptr.push_back(bytes);
        assert(can_ptr.size() <= num_canaries);
      }
    }
    if(ttp)
      ptr += END_GUARD_BAND;
    assert(ptr <= buf_ + s);
#if 0
    printf("ptr @ %p\n",ptr);
#endif
  }

  return buf_;
}

void* MLEngine::allocate_gradient_tensor(TensorList L, int buftype, int n, long long int size)
{
#ifdef USE_MLSL
  void *rw_buf = (void*)MLSL::Environment::GetEnv().Alloc(n * size, 2097152);
#else
  void *rw_buf = (void*)libxsmm_aligned_malloc(n * size, 2097152);
#endif

  int count = 0;
  for(Iter it = L.begin(); it != L.end(); it++)
  {

    Tensor *t = it->t;
    string owner = dynamic_cast<NNNode*>(t->getOwner())->getNodeName();
    for(int j=0; j<t->getNumDataBuffers(); j++)
    {
      TensorBuf *tBuf = t->getBuf(j);
      if(tBuf->getBufferType() == buftype)
      {
        long long int offset = (count % n) * size;
        tBuf->setBuffer(rw_buf + offset);
#ifdef DEBUG
#ifdef USE_MLSL
        if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
          printf("node %s, tensor %s, count %d, pointer %p\n",owner.c_str(), it->name.c_str(), count, rw_buf + offset);
#endif
#endif
        count++;
      }
    }
  }
  return rw_buf;
}

void MLEngine::insertSplitNodes(NTGParameter& p, NTGParameter* ps)
{
  ps->CopyFrom(p);
  ps->clear_node();

  vector< pair<string, string> > top_names;

  for(int i=0; i<p.node_size(); i++)
  {
    const NodeParameter& np = p.node(i);
    string nn = np.name();
    for(int j=0; j<np.top_size(); j++)
      top_names.push_back(make_pair(np.top(j), nn));
  }

  std::multimap<std::string, NodeParameter> top_as_bot;

  for(int i=0; i < top_names.size(); i++)
  {
    pair<string, string> tn = top_names[i];
    for(int j=0; j < p.node_size(); j++)
    {
      const NodeParameter& np = p.node(j);
      string nn = p.node(j).name();
      if(nn.compare(tn.second) == 0) continue;
      for(int k=0; k < np.bottom_size(); k++)
      {
        std::string t = tn.first;
        if(t.compare(p.node(j).bottom(k)) == 0)
          top_as_bot.insert(make_pair(t, p.node(j)));
      }
    }
  }

  std::multimap<std::string, std::string> old_bottom;
  std::multimap<std::string, std::string> new_bottom;

  for(int i=0; i<p.node_size(); i++)
  {
    NodeParameter* np = ps->add_node();
    np->CopyFrom(p.node(i));
    string onn = np->name();

    for(int j=0; j<np->top_size(); j++)
    {
      string t = np->top(j);
      int split_count = top_as_bot.count(t);
      if(split_count > 1)
      {
        NodeParameter *snp = ps->add_node();
        snp->Clear();
        snp->add_bottom(t);
        string snn = t + "_" + onn + "_" + std::to_string(j) + "_split";
        snp->set_name(snn);
        snp->set_type("Split");
        if(t.compare("label") == 0)
          snp->set_propagate_down(false);

        std::multimap<string, NodeParameter>::iterator it;
        int k = 0;
        for(it=top_as_bot.equal_range(t).first; it != top_as_bot.equal_range(t).second; it++)
        {
          NodeParameter onp = (*it).second;
          string nn = onp.name();

          string stn = t + "_" + nn + "_" + std::to_string(j) + "_split_" + std::to_string(k);
          snp->add_top(stn);
          k++;

          for(int l=0; l<onp.bottom_size(); l++)
          {
            if(onp.bottom(l) == t)
            {
              old_bottom.insert(make_pair(t, nn));
              new_bottom.insert(make_pair(nn, stn));
            }
          }
        }
      }
    }
  }

  std::multimap<std::string, std::string>::iterator it1;
  std::multimap<std::string, std::string>::iterator it2;
  for(int i=0; i<ps->node_size(); i++)
  {
    NodeParameter* mn = ps->mutable_node(i);
    if(mn->type().compare("Split") == 0) continue;
    for(int j=0; j<mn->bottom_size(); j++)
    {
      string t = mn->bottom(j);
      it1 = old_bottom.find(t);
      if(it1 == old_bottom.end()) continue;

      for(it1=old_bottom.equal_range(t).first; it1 != old_bottom.equal_range(t).second; it1++)
        if(mn->name() == (*it1).second) break;

      assert(it1 != old_bottom.end());
      string s = (*it1).second;
      for(it2=new_bottom.equal_range(s).first; it2 != new_bottom.equal_range(s).second; it2++)
      {
        string v = (*it2).second;
        if(v.find(mn->bottom(j)) != v.npos)
          mn->set_bottom(j, v);
      }
    }
  }
}

void MLEngine::create(int mode, string ntgConfig, string solverConfig)
{
  bool parsed = parseMLConfig(ntgConfig, &ntgparam_);
  if(!parsed) exit(-1);

  if(!solverConfig.empty())
  {
    parsed = parseSolverConfig(solverConfig, &sparam_);
    if(!parsed) exit(-1);

    num_epochs_ = sparam_.max_epochs();
    current_epoch_ = 0;
    current_batch_ = 0;
    load_from_checkpoint_ = sparam_.load_checkpoint();
    checkpoint_dir_ = sparam_.checkpoint_dir();
    checkpoint_format_ = sparam_.checkpoint_format();
    data_type_ = sparam_.data_type();
  }

#ifdef _OPENMP
  num_threads_ = omp_get_max_threads();
#else
  num_threads_ = 1;
#endif

  printf("Using %d threads\n",num_threads_);

#ifdef USE_MLSL
  global_node_id_ = MLSL::Environment::GetEnv().GetProcessIdx();
  num_machines_ = MLSL::Environment::GetEnv().GetProcessCount();
  data_parallelism = NULL;
  if(mode == TRAIN)
    session_ = MLSL::Environment::GetEnv().CreateSession(MLSL::PT_TRAIN);
  else
    session_ = MLSL::Environment::GetEnv().CreateSession(MLSL::PT_TEST);
#else
  global_node_id_ = 0;
  num_machines_ = 1;
#endif

  // if no training mode in config, then set inferenceOnly_ to true
  inferenceOnly_ = (mode == TEST);

  // Initialize solver node
  int ni = find_in_nodeTypeList("Solver");
  solverParams_ = parseSolverParams(&sparam_);
  solver_ = new SolverNode(solverParams_, this);

  /*************************************************************************************/
  /*** Create a global tensor to hold scratch memory needed by Conv layers (LIBXSMM) ***/
  /*************************************************************************************/
  tenScratch_ = new Tensor("scratch");

  NTGParameter split_ntgparam;

  insertSplitNodes(ntgparam_, &split_ntgparam);
  if(global_node_id_ == 0)
    split_ntgparam.PrintDebugString();

  int numNodes = split_ntgparam.node_size();

  for(int i=0; i<numNodes; i++)
  {
    // get name and type of each node
    // call parse and create node functions based on type
    // find member of TypeList
    NodeParameter np = split_ntgparam.node(i);
    string ntype = np.type();

#ifdef DEBUG
    printf("node type %s\n",ntype.c_str());
#endif
    int j = find_in_nodeTypeList(ntype);

    MLParams *p = nodeTypes[j].parse(&np);
    MLNode *node = nodeTypes[j].create(p, this);
    ntg_.push_back(node);
#ifdef USE_MLSL
    if(ntype.find("Data") != ntype.npos)
      data_parallelism = MLSL::Environment::GetEnv().CreateDistribution(num_machines_, 1);
#endif

  }

  // We assert that the first node in the topology be a data node. Graph creation starts from data node
  NNNode* dnode = dynamic_cast<NNNode*>(ntg_[0]);
  assert(dnode != NULL);

  string first = dnode->getNodeType();
#ifdef DEBUG
  printf("first node type %s\n",first.c_str());
#endif
  assert(first.find("Data") != first.npos);

  // Create the neural network graph for training or testing mode
  dnode->createNNGraph(mode);

  // Forward Pass Binning.
  // Look for tasks attached to nodes with no successors. Add them to the Executing Task Graph (etg) first.
  for(int i=numNodes-1; i>0; i--)
  {
    NNNode *nn = dynamic_cast<NNNode*>(ntg_[i]);
    Task* t = nn->getBasicTask(BASIC_TASK_FORW);

    if(nn->getNumNextNodes() == 0)
    {
      etg_[mode].push_back(t);
#ifdef DEBUG
      printf("FP task %p (node %s), bin %d pushed to etg_\n",t, nn->getNodeName().c_str(), t->getMaxBin());
#endif
    }
  }

  // Assign bins to tasks based on their dependencies. Tasks with lower bin number must
  // execute before those with higher bin number. Tasks with same bin number can execute in parallel
  // Ensure no duplicate tasks in etg
  create_schedule(mode);
  optimize_schedule(mode);

  if(mode == TRAIN)
  {
    for(auto it = etg_[mode].begin(); it != etg_[mode].end(); it++)
    {
      Task *t = *it;
      if(t->getBasicTaskId() == BASIC_TASK_FORW)
        etg_[TEST].push_back(t);
      else
        break;
    }
  }

#ifdef DEBUG
  for(auto it=etg_[mode].begin(); it != etg_[mode].end(); it++)
  {
    Task* t = (*it);
    string s = dynamic_cast<NNNode*>(t->getNode())->getNodeName();
    if(t->getBasicTaskId() == BASIC_TASK_FORW)
      printf("FP Task %p in node %s at bin %d\n",t, s.c_str(), t->getMaxBin());
    else if(t->getBasicTaskId() == BASIC_TASK_BACK)
      printf("BP  Task %p in node %s at bin %d\n",t, s.c_str(), t->getMaxBin());
    else if(t->getBasicTaskId() == BASIC_TASK_WGRAD)
      printf("WG Task %p in node %s at bin %d\n",t, s.c_str(), t->getMaxBin());
    else
      printf("SOLVER Task %p in node %s at bin %d\n",t, s.c_str(), t->getMaxBin());
  }
#endif

  if(mode == TRAIN)
    printf("Training schedule has %u tasks\n",(unsigned int)etg_[mode].size());
  else
    printf("Testing schedule has %u tasks\n",(unsigned int)etg_[mode].size());


  /*** Allocate memory and set pointers for INPUT and LABEL buffers ***/
  /**********************************************************************/
  long long int total_input_size;
  long long int max_fwd_buffer_size=0;

  input_buf_ = allocate_memory("INPUT", inTList_, DATA, input_can_ptr, &ic, &total_input_size, NULL, num_machines_);
  if(global_node_id_ == 0)
    printf("Total input memory allocated %lld bytes\n", total_input_size);

  /**********************************************************************/
  /*** Allocate memory and set pointers for FORWARD ACTIVATION buffer ***/
  /**********************************************************************/
  long long int total_fact_size;
  fact_buf_ = allocate_memory("FACT", outTList_, DATA, fact_can_ptr, &fac, &total_fact_size, &max_fwd_buffer_size, num_machines_);
  if(global_node_id_ == 0)
    printf("Total forward activation memory allocated %lld bytes\n", total_fact_size);

  /***********************************************************/
  /*** Allocate memory and set pointers for WEIGHTS buffer ***/
  /***********************************************************/
  long long int total_weight_size;
  weight_buf_ = allocate_memory("WEIGHT", wTList_, DATA, wt_can_ptr, &wtc, &total_weight_size, NULL, num_machines_);
  if(global_node_id_ == 0)
    printf("Total weights memory allocated %lld bytes\n", total_weight_size);

  /***********************************************************/
  /*** Allocate memory and set pointers for BIASES buffer ***/
  /***********************************************************/
  long long int total_bias_size;
  bias_buf_ = allocate_memory("BIAS", biasTList_, DATA, bias_can_ptr, &bic, &total_bias_size, NULL, num_machines_);
  if(global_node_id_ == 0)
  printf("Total bias memory allocated %lld bytes\n", total_bias_size);

  /***********************************************************/
  /*** Allocate memory and set pointers for STATS buffer ***/
  /***********************************************************/
  long long int total_stats_size;
  stats_buf_ = allocate_memory("STATS", statsTList_, DATA, stats_can_ptr, &sic, &total_stats_size, NULL, num_machines_);
  if(global_node_id_ == 0)
    printf("Total stats memory allocated %lld bytes\n", total_stats_size);

  // Required only for training
  long long int total_bp_size;
  if(!inferenceOnly_)
  {
    /***********************************************************************/
    /*** Allocate memory and set pointers for BACKWARD ACTIVATION buffer ***/
    /***********************************************************************/
#if !defined(USE_OPTBP_ALLOC)
    long long int total_bact_size;
    bact_buf_ = allocate_memory("ACT", outTList_, DIFF, bact_can_ptr, &bac, &total_bact_size, NULL, num_machines_);
    if(global_node_id_ == 0)
      printf("Total backward activation memory allocated %lld bytes\n", total_bact_size);
#else
    long long int total_bact_size = NDIFFS * max_fwd_buffer_size;
    bact_buf_ = allocate_gradient_tensor(outTList_, DIFF, NDIFFS, max_fwd_buffer_size);
    if(global_node_id_ == 0)
      printf("Total backward activation memory allocated %lld bytes\n", total_bact_size);
#endif

    /********************************************************************/
    /*** Allocate memory and set pointers for WEIGHT GRADIENTS buffer ***/
    /********************************************************************/
    long long int total_wdiff_size;
    wdiff_buf_ = allocate_memory("WEIGHT", wTList_, DIFF, wdiff_can_ptr, &wdc, &total_wdiff_size, NULL, num_machines_);
    if(global_node_id_ == 0)
      printf("Total weight gradient memory allocated %lld bytes\n", total_wdiff_size);

    /*********************************************************************/
    /*** Allocate memory and set pointers for WEIGHT INCREMENTS buffer ***/
    /*********************************************************************/
    long long int total_winc_size;
    winc_buf_ = allocate_memory("WEIGHT", wTList_, HISTORY, winc_can_ptr, &wic, &total_winc_size, NULL, num_machines_);
    if(global_node_id_ == 0)
      printf("Total weight increment memory allocated %lld bytes\n", total_winc_size);

    /********************************************************************/
    /*** Allocate memory and set pointers for BIAS GRADIENTS buffer ***/
    /********************************************************************/
    long long int total_bidiff_size;
    bidiff_buf_ = allocate_memory("BIAS", biasTList_, DIFF, bidiff_can_ptr, &bidc, &total_bidiff_size, NULL, num_machines_);
    if(global_node_id_ == 0)
      printf("Total bias gradient memory allocated %lld bytes\n", total_bidiff_size);

    /*********************************************************************/
    /*** Allocate memory and set pointers for BIAS INCREMENTS buffer ***/
    /*********************************************************************/
    long long int total_biinc_size;
    biinc_buf_ = allocate_memory("BIAS", biasTList_, HISTORY, biinc_can_ptr, &biic, &total_biinc_size, NULL, num_machines_);
    if(global_node_id_ == 0)
      printf("Total bias increment memory allocated %lld bytes\n", total_biinc_size);

    total_bp_size = total_bact_size + total_wdiff_size + total_winc_size + total_bidiff_size + total_biinc_size;
  }

  long long int total_memory = total_input_size + total_fact_size + total_weight_size + total_bias_size + total_bp_size;
  if(global_node_id_ == 0)
    printf("Total tensor memory = %lld\n",total_memory);
}
