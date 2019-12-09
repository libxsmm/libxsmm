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


#include <map>
#include "assert.h"
#include "proto/gxm.pb.h"
#include "Node.hpp"
#include "Engine.hpp"
#include "Conv.hpp"
#include "FullyConnected.hpp"
#include "FusedBNorm.hpp"
#include "FusedConvBN.hpp"
#include "DummyData.hpp"
#include "TypeList.hpp"

#include "unistd.h"
#include "limits.h"

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
    case BNORMVAR:
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
    case BNORMVAR:
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

    float *fp = (float*)(tBuf->getBuffer());
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<bytes/sizeof(float); i++)
      fp[i] = 0.f;
  }
}

void MLEngine::checkpoint(TensorList L, int buftype)
{
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
    string tn = t->getTensorName();
    string n = checkpoint_dir_ + "/" + tn;
    if(buftype == HISTORY)
      n = n + "_history";
    else if(buftype == DIFF)
      n = n + "_grad";

    string nntype = dynamic_cast<NNNode*>(t->getOwner())->getNodeType();

    if(current_epoch_ == 30 || current_epoch_ == 60 || current_epoch_ == 80)
    {
      if(tenType == ACT)
      {
        n = checkpoint_dir_ + to_string(current_epoch_) + "/" + tn;
        if(tn.find("bn") != tn.npos)
        {
          if(nntype == "FusedBatchNorm")
          {
            FusedBNormNode* bn = dynamic_cast<FusedBNormNode*>(t->getOwner());
            bn->Checkpoint(tBuf, n, checkpoint_format_);
          }
          else if(nntype == "FusedConvBN")
          {
            FusedConvBNNode* fcbn = dynamic_cast<FusedConvBNNode*>(t->getOwner());
            fcbn->Checkpoint(tBuf, n, checkpoint_format_);
          }
        }
      }
    }

    if((tenType == CONVWEIGHT) || (tenType == CONVBIAS))
    {
      if(nntype == "Convolution")
      {
        ConvNode* cn = dynamic_cast<ConvNode*>(t->getOwner());
        cn->Checkpoint(tBuf, n, checkpoint_format_);
        if(current_epoch_ == 30 || current_epoch_ == 60 || current_epoch_ == 80)
        {
          n = checkpoint_dir_ + to_string(current_epoch_) + "/" + tn;
          if(buftype == HISTORY)
            n = n + "_history";
          else if(buftype == DIFF)
            n = n + "_diff";
          cn->Checkpoint(tBuf, n, checkpoint_format_);
        }
      }
      else if(nntype == "FusedConvBN")
      {
        FusedConvBNNode* fcbn = dynamic_cast<FusedConvBNNode*>(t->getOwner());
        fcbn->Checkpoint(tBuf, n, checkpoint_format_);
        if(current_epoch_ == 30 || current_epoch_ == 60 || current_epoch_ == 80)
        {
          n = checkpoint_dir_ + to_string(current_epoch_) + "/" + tn;
          if(buftype == HISTORY)
            n = n + "_history";
          else if(buftype == DIFF)
            n = n + "_grad";
          fcbn->Checkpoint(tBuf, n, checkpoint_format_);
        }
      }
    }
    else if((tenType == FCWEIGHT) || (tenType == FCBIAS))
    {
      FCNode* fn = dynamic_cast<FCNode*>(t->getOwner());
      fn->Checkpoint(tBuf, n, checkpoint_format_);
      if(current_epoch_ == 30 || current_epoch_ == 60 || current_epoch_ == 80)
      {
        n = checkpoint_dir_ + to_string(current_epoch_) + "/" + tn;
        if(buftype == HISTORY)
          n = n + "_history";
        else if(buftype == DIFF)
          n = n + "_grad";
        fn->Checkpoint(tBuf, n, checkpoint_format_);
      }
    }
    else if((tenType == BNORMSCALE) || (tenType == BNORMSHIFT) || (tenType == BNORMMEAN) || (tenType == BNORMVAR))
    {
      if(nntype == "FusedBatchNorm")
      {
        FusedBNormNode* bn = dynamic_cast<FusedBNormNode*>(t->getOwner());
        bn->Checkpoint(tBuf, n, checkpoint_format_);
        if(current_epoch_ == 30 || current_epoch_ == 60 || current_epoch_ == 80)
        {
          n = checkpoint_dir_ + to_string(current_epoch_) + "/" + tn;
          if(buftype == HISTORY)
            n = n + "_history";
          else if(buftype == DIFF)
            n = n + "_grad";
          bn->Checkpoint(tBuf, n, checkpoint_format_);
        }
      }
      else if(nntype == "FusedConvBN")
      {
        FusedConvBNNode* fcbn = dynamic_cast<FusedConvBNNode*>(t->getOwner());
        fcbn->Checkpoint(tBuf, n, checkpoint_format_);
        if(current_epoch_ == 30 || current_epoch_ == 60 || current_epoch_ == 80)
        {
          n = checkpoint_dir_ + to_string(current_epoch_) + "/" + tn;
          if(buftype == HISTORY)
            n = n + "_history";
          else if(buftype == DIFF)
            n = n + "_grad";
          fcbn->Checkpoint(tBuf, n, checkpoint_format_);
        }
      }
    }
  }
}

void MLEngine::read_checkpoint_file(TensorBuf* tBuf, string filename, string format)
{
  long long int bytes = tBuf->getBufferSize();
  int dtype = tBuf->getDataType();

  void* ptr;
  ptr = tBuf->getBuffer();

  FILE* f;
  if(format == "binary")
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
  }
  fclose(f);

  if(data_type_ == BF16 && (filename.find("wt") != filename.npos))
    if(filename.find("history") == filename.npos)
      convert_f32_bf16((float*)ptr, (libxsmm_bfloat16*)tBuf->getLPBuffer(), bytes/sizeof(float), 0);

}

void MLEngine::load_checkpoint(TensorList L, int buftype, string format)
{
  TensorBuf* tBuf;

  for(Iter it=L.begin(); it != L.end(); it++)
  {
    Tensor* t = it->t;
    int tenType = t->getType();
    if((tenType != CONVWEIGHT) && (tenType != CONVBIAS) && (tenType != FCWEIGHT) && (tenType != FCBIAS))
      if((tenType != BNORMSCALE) && (tenType != BNORMSHIFT) && (tenType != BNORMMEAN) && (tenType != BNORMVAR))
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

    if(buftype == HISTORY)
      n = n + "_history";

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

void MLEngine:: waitForComms(string tenType)
{
#ifdef USE_MLSL
  if(tenType=="WEIGHT")
  {
    if(!wtgrad_comms_vec.empty())
    {
      for(int i=0; i<wtgrad_comms_vec.size(); i++)
        wtgrad_comms_vec[i]->GetParameterSet(0)->WaitGradientComm();
    }
  }
  else if(tenType=="BIAS")
  {
    if(!bias_grad_comms_vec.empty())
    {
      for(int i=0; i<bias_grad_comms_vec.size(); i++)
      {
        bias_grad_comms_vec[i]->GetParameterSet(0)->WaitGradientComm();
        bias_grad_comms_vec[i]->GetParameterSet(1)->WaitGradientComm();
        bias_grad_comms_vec[i]->GetParameterSet(2)->WaitGradientComm();
        bias_grad_comms_vec[i]->GetParameterSet(3)->WaitGradientComm();
      }
    }
  }
  else if(tenType=="COMBO")
  {
    if(!combo_grad_comms_vec.empty())
    {
      for(int i=0; i<combo_grad_comms_vec.size(); i++)
      {
        combo_grad_comms_vec[i]->GetParameterSet(0)->WaitGradientComm();
        combo_grad_comms_vec[i]->GetParameterSet(1)->WaitGradientComm();
        combo_grad_comms_vec[i]->GetParameterSet(2)->WaitGradientComm();
        combo_grad_comms_vec[i]->GetParameterSet(3)->WaitGradientComm();
        combo_grad_comms_vec[i]->GetParameterSet(4)->WaitGradientComm();
      }
    }
  }
#endif
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
        fscanf(f, "%d %f %f\n",&current_epoch_, &lr_, &scf_);
        fclose(f);
      }
      else
        printf("No checkpoint state file to read\n");

      if(current_epoch_ != num_epochs_ - 1)
        current_epoch_++;
      load_checkpoint(wTList_, DATA, checkpoint_format_);
      load_checkpoint(wTList_, HISTORY, checkpoint_format_);
      load_checkpoint(biasTList_, DATA, checkpoint_format_);
      load_checkpoint(biasTList_, HISTORY, checkpoint_format_);
      load_checkpoint(statsTList_, DATA, checkpoint_format_);

#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        int tid = omp_get_thread_num();
        int ntps = num_threads_/NUM_NUMA_NODES;
        int n = tid/ntps;
        int w = total_weights_;
        int b = total_biases_;

        if(n != 0 && tid % ntps == 0)
        {
          float *wptr = (float*)weight_buf_[n];
#if 1
          float *bptr = (float*)bias_buf_[n];
          float *sptr = (float*)stats_buf_[n];
#endif

#pragma omp simd
          for(int i=0; i<w; i++)
            wptr[i] = ((float*)weight_buf_[0])[i];

#if 1
#pragma omp simd
          for(int i=0; i<b; i++)
          {
            bptr[i] = ((float*)bias_buf_[0])[i];
            sptr[i] = ((float*)stats_buf_[0])[i];
          }
#endif
          if(lpweight_buf_[0] != NULL)
          {
            libxsmm_bfloat16 *lwptr = (libxsmm_bfloat16*)lpweight_buf_[n];
#pragma omp simd
            for(int i=0; i<w; i++)
              lwptr[i] = ((libxsmm_bfloat16*)lpweight_buf_[0])[i];
          }
        }
      }
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
        if(global_node_id_ == 0 && current_batch_ % 100 == 0)
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

#ifdef DUMP_WT
          if(global_node_id_ == 0)
            if(current_epoch_ == 30 || current_epoch_ == 60 || current_epoch_ == 80)
              if(current_batch_ == num_train_batches_-1)
                checkpoint(wTList_, DIFF);
#endif

#ifdef USE_MLSL
          waitForComms("WEIGHT");
          waitForComms("BIAS");
          waitForComms("COMBO");
#endif

#ifdef MLSL
          data_parallelism->Barrier(MLSL::GT_DATA);
#endif

#if 0
          solver_->applyUpdate((float**)weight_buf_, (float**)winc_buf_, wdiff_buf_, total_weights_, (float**)wt_lr_mult_, (float**)wt_decay_mult_, "WEIGHT");
#else
          solver_->applyUpdate((float**)weight_buf_, (float**)winc_buf_, wdiff_buf_, total_weights_, 1.0, 1.0, "WEIGHT");
#endif
          if(data_type_ == BF16)
            convert_f32_bf16((float**)weight_buf_, (libxsmm_bfloat16**)lpweight_buf_, total_weights_);

#if 0
          solver_->applyUpdate((float**)bias_buf_, (float**)biinc_buf_, bidiff_buf_, total_biases_, (float**)bias_lr_mult_, (float**)bias_decay_mult_, "BIAS");
#else
#if 1
          solver_->applyUpdate((float**)bias_buf_, (float**)biinc_buf_, bidiff_buf_, total_biases_, 1.0, 0.0, "BIAS");
#else
          solver_->applyUpdate((float*)bias_buf_, (float*)biinc_buf_, bidiff_buf_, total_biases_, 1.0, 0.0, "BIAS");
#endif
#endif

#ifdef TIMING
          gettimeofday(&tvie, NULL);
          double sgdtime = (tvie.tv_sec + tvie.tv_usec*1e-6) - (tvis.tv_sec + tvis.tv_usec*1e-6);
          printf("global sgd time: %f ms\n",sgdtime*1000);
#endif
        }

        gettimeofday(&tve, NULL);
        fbtime = (tve.tv_sec + tve.tv_usec*1e-6) - (tvs.tv_sec + tvs.tv_usec*1e-6);
        if(global_node_id_ == 0 && current_batch_ % 100 == 0)
          printf("Fwd-Bwd time: %f ms\n",fbtime*1000);

        if ( current_batch_ > 1 )
          runtime += fbtime;

#ifdef CANARY_CHECK
        canary_check(input_buf_, input_can_ptr, ic);
        canary_check(fact_buf_, fact_can_ptr, fac);
        canary_check(bact_buf_, bact_can_ptr, bac);
#endif
      }

      current_batch_ = 0;

      if ( num_train_batches_ > 1 ) {
        char hostname[HOST_NAME_MAX + 1];
        gethostname(hostname, HOST_NAME_MAX + 1);
        printf("%s; Average Training time = %f seconds", hostname, runtime/((double)(num_train_batches_-2)));
        if(runtime > 0) {
          printf("; Average Training throughput = %f images/s\n", ((double)(batch_size_*(num_train_batches_-2)))/runtime);
        } else {
          printf("\n");
        }
      }

      // Checkpoint weights and biases
      if(global_node_id_ == 0)
      {
        checkpoint(wTList_, DATA);
        checkpoint(wTList_, HISTORY);
        checkpoint(biasTList_, DATA);
        checkpoint(biasTList_, HISTORY);
        checkpoint(statsTList_, DATA);

#ifdef DUMP_ACT_DATA
        if(current_epoch_ == 30 || current_epoch_ == 60 || current_epoch_ == 80)
        {
          checkpoint(outTList_, DATA);
          checkpoint(outTList_, DIFF);
        }
#endif

        FILE* f = fopen("checkpoint", "w");
        if(f != NULL)
        {
          fprintf(f, "%d %10g %10g\n",current_epoch_, lr_, scf_);
          fclose(f);
        }
      }
#ifdef USE_MLSL
      data_parallelism->Barrier(MLSL::GT_DATA);
#endif

      // Tell data node that it should use test data
      exec_mode_ = VAL;

      if(global_node_id_ == 0)
      {
        printf("===========================================\n");
        printf("VAL mode, testing batches %d\n", num_test_batches_);
        printf("===========================================\n");
      }

      // Run validation network at end of each epoch
      for(; current_batch_<num_test_batches_; current_batch_++)
      {
        for(int v=0; v<num_test_views_; v++)
          for(auto it = etg_[VAL].begin(); it != etg_[VAL].end(); it++)
            (*it)->invoke();
      }

      current_batch_ = 0;

#ifdef CANARY_CHECK
      canary_check(input_buf_, input_can_ptr, ic);
      canary_check(fact_buf_, fact_can_ptr, fac);
      canary_check(weight_buf_, wt_can_ptr, wtc);
      canary_check(bias_buf_, bias_can_ptr, bic);
#endif
    }

#ifdef USE_MLSL
    MLSL::Environment::GetEnv().Free(input_buf_);
    MLSL::Environment::GetEnv().Free(fact_buf_);
    MLSL::Environment::GetEnv().Free(bact_buf_);
#else
    libxsmm_free(input_buf_);
    libxsmm_free(fact_buf_);
    libxsmm_free(bact_buf_);
#endif

    for(int n=0; n<NUM_NUMA_NODES; n++)
    {
#ifdef USE_MLSL
      MLSL::Environment::GetEnv().Free(weight_buf_[n]);
      if(lpweight_buf_[n] != NULL)
        MLSL::Environment::GetEnv().Free(lpweight_buf_[n]);
      MLSL::Environment::GetEnv().Free(wdiff_buf_[n]);
      if(lpwdiff_buf_[n] != NULL)
        MLSL::Environment::GetEnv().Free(lpwdiff_buf_[n]);
      MLSL::Environment::GetEnv().Free(winc_buf_[n]);
#if 1
      MLSL::Environment::GetEnv().Free(bias_buf_[n]);
      MLSL::Environment::GetEnv().Free(bidiff_buf_[n]);
      MLSL::Environment::GetEnv().Free(biinc_buf_[n]);
      MLSL::Environment::GetEnv().Free(stats_buf_[n]);
#else
      MLSL::Environment::GetEnv().Free(bias_buf_);
      MLSL::Environment::GetEnv().Free(bidiff_buf_);
      MLSL::Environment::GetEnv().Free(biinc_buf_);
      MLSL::Environment::GetEnv().Free(stats_buf_);
#endif
#else
      libxsmm_free(weight_buf_[n]);
      libxsmm_free(wdiff_buf_[n]);
      if(lpweight_buf_[n] != NULL)
        libxsmm_free(lpweight_buf_[n]);
      if(lpwdiff_buf_[n] != NULL)
        libxsmm_free(lpwdiff_buf_[n]);
      libxsmm_free(winc_buf_[n]);
#if 1
      libxsmm_free(bias_buf_[n]);
      libxsmm_free(bidiff_buf_[n]);
      libxsmm_free(biinc_buf_[n]);
      libxsmm_free(stats_buf_[n]);
#else
      libxsmm_free(bias_buf_);
      libxsmm_free(bidiff_buf_);
      libxsmm_free(biinc_buf_);
      libxsmm_free(stats_buf_);
#endif
#endif
    }
  }
  else if(mode == TEST)
  {
    exec_mode_ = TEST;

    FILE *f = fopen("checkpoint", "r");
    fscanf(f, "%d %f %f\n",&current_epoch_, &lr_, &scf_);
    fclose(f);

    printf("====================================================================\n");
    printf("TEST mode, testing batches %d, scaling factor %.10f\n", num_test_batches_, scf_);
    printf("====================================================================\n");

    load_checkpoint(wTList_, DATA, checkpoint_format_);
    load_checkpoint(biasTList_, DATA, checkpoint_format_);
    load_checkpoint(statsTList_, DATA, checkpoint_format_);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      int ntps = num_threads_/NUM_NUMA_NODES;
      int n = tid/ntps;
      int w = total_weights_;
      int b = total_biases_;

      if(n != 0 && tid % ntps == 0)
      {
        float *wptr = (float*)weight_buf_[n];
#if 1
        float *bptr = (float*)bias_buf_[n];
        float *sptr = (float*)stats_buf_[n];
#endif

#pragma omp simd
        for(int i=0; i<w; i++)
          wptr[i] = ((float*)weight_buf_[0])[i];

#if 1
#pragma omp simd
        for(int i=0; i<b; i++)
        {
          bptr[i] = ((float*)bias_buf_[0])[i];
          sptr[i] = ((float*)stats_buf_[0])[i];
        }
#endif
        if(lpweight_buf_[0] != NULL)
        {
          libxsmm_bfloat16 *lwptr = (libxsmm_bfloat16*)lpweight_buf_[n];
#pragma omp simd
          for(int i=0; i<w; i++)
            lwptr[i] = ((libxsmm_bfloat16*)lpweight_buf_[0])[i];
        }
      }
    }

    // Run test network when command-line mode is set to "test"
    for(int b=0; b<num_test_batches_; b++)
    {
      for(auto it = etg_[TEST].begin(); it != etg_[TEST].end(); it++)
        (*it)->invoke();
    }
  }
}

void MLEngine::convert_f32_bf16(float* in, libxsmm_bfloat16* out, int len, int numa_node)
{

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = omp_get_thread_num();
    int ntps = num_threads_/NUM_NUMA_NODES;
    int n = tid/ntps;
    int ltid = tid - numa_node*ntps;

    if(n == numa_node)
    {
      int jobs = (len % ntps == 0) ? len/ntps : len/ntps + 1;
      int tb = (ltid*jobs < len) ? ltid*jobs : len;
      int te = ((ltid+1)*jobs < len) ? (ltid+1)*jobs : len;

      for (int i = tb; i < te; i+=16 ) {
        __m512  vfp32  = gxm_fp32_to_bfp16_rne_adjustment_avx512f( _mm512_loadu_ps( in+i ) );
        __m256i vbfp16 = gxm_fp32_to_bfp16_truncate_avx512f( vfp32 );
        _mm256_storeu_si256( (__m256i*)(out+i), vbfp16 );
      }
    }
  }
}

void MLEngine::convert_f32_bf16(float** in, libxsmm_bfloat16** out, int len)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = omp_get_thread_num();
    int ntps = num_threads_/NUM_NUMA_NODES;
    int n = tid/ntps;
    int ltid = tid - n*ntps;

    float *inp = in[n];
    libxsmm_bfloat16 *outp = out[n];

    int jobs = (len % ntps == 0) ? len/ntps : len/ntps + 1;
    int tb = (ltid*jobs < len) ? ltid*jobs : len;
    int te = ((ltid+1)*jobs < len) ? (ltid+1)*jobs : len;

    for (int i = tb; i < te; i+=16 ) {
      __m512  vfp32  = gxm_fp32_to_bfp16_rne_adjustment_avx512f(_mm512_loadu_ps(inp + i));
      __m256i vbfp16 = gxm_fp32_to_bfp16_truncate_avx512f(vfp32);
      _mm256_storeu_si256( (__m256i*)(outp+i), vbfp16 );
    }
  }
}
void MLEngine::convert_bf16_f32(libxsmm_bfloat16* in, float* out, int len)
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

void MLEngine::allocate_memory(string tenType, TensorList L, int buftype, vector<int>& can_ptr, int* nc, long long int* bufsize)
{
  bool ttp = false; //(tenType != "WEIGHT") & (tenType != "BIAS");

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

  if(tenType == "WEIGHT")
    total_weights_ = s/sizeof(float);
  else if(tenType == "BIAS" || tenType == "STATS")
    total_biases_ = s/sizeof(float);

  if(solver_->getGlobalFlag())
  {
    if(tenType == "WEIGHT")
    {
#ifdef BF16_MLSL
      if(buftype == DIFF)
      {
        if(data_type_ == FLOAT)
          total_weights_ = s/sizeof(float);
        else if(data_type_ == BF16)
          total_weights_ = s/sizeof(libxsmm_bfloat16);
      }
      else
#endif
        total_weights_ = s/sizeof(float);

      int factor = num_threads_ * VLEN;
      int nwt = (total_weights_ + factor - 1)/factor;
      total_weights_ = nwt * factor;

#ifdef BF16_MLSL
      if(buftype == DIFF)
      {
        if(data_type_ == FLOAT)
          s = total_weights_ * sizeof(float);
        else if(data_type_ == BF16)
          s = total_weights_ * sizeof(libxsmm_bfloat16);
      }
      else
#endif
        s = total_weights_ * sizeof(float);
    }
    else if(tenType == "BIAS" || tenType == "STATS")
    {
      total_biases_ = s / sizeof(float);
      int factor = num_threads_ * VLEN;
      int nwt = (total_biases_ + factor - 1)/factor;
      total_biases_ = nwt * factor;

      s = total_biases_ * sizeof(float);
    }
  }

  // Number of guard bands in tensor; used for canary checking
  *nc = num_canaries;

  // Allocate memory
#ifdef BF16_MLSL
  bool lp = (data_type_ == BF16) && (tenType=="WEIGHT") && (buftype == DATA);
#else
  bool lp = (data_type_ == BF16) && (tenType=="WEIGHT");
#endif

  void *buf_;
  void **ptrptr, **lptrptr=NULL;

#if 0 //def USE_MLSL
  s = ALIGN_SIZE(s, 2097152);
#endif

  if(tenType=="INPUT")
  {
#ifdef USE_MLSL
    buf_ = (void*)MLSL::Environment::GetEnv().Alloc(s, 2097152);
#else
    buf_ = (void*)libxsmm_aligned_malloc(s, 2097152);
#endif
    input_buf_ = buf_;
  }
  else if(tenType == "FACT")
  {
#ifdef USE_MLSL
    buf_ = (void*)MLSL::Environment::GetEnv().Alloc(s, 2097152);
#else
    buf_ = (void*)libxsmm_aligned_malloc(s, 2097152);
#endif
    fact_buf_ = buf_;
  }
  else if(tenType == "WEIGHT")
  {
    if(buftype == DATA)
    {
      for(int n=0; n<NUM_NUMA_NODES; n++)
      {
#ifdef USE_MLSL
        weight_buf_[n] = (void*)MLSL::Environment::GetEnv().Alloc(s, 2097152);
#else
        weight_buf_[n] = (void*)libxsmm_aligned_malloc(s, 2097152);
#endif
        if(lp)
#ifdef USE_MLSL
          lpweight_buf_[n] = (void*)MLSL::Environment::GetEnv().Alloc(s/sizeof(libxsmm_bfloat16), 2097152);
#else
          lpweight_buf_[n] = (void*)libxsmm_aligned_malloc(s/sizeof(libxsmm_bfloat16), 2097152);
#endif
      }
      buf_ = weight_buf_[0];
      ptrptr = weight_buf_;
      if(lp)
        lptrptr = lpweight_buf_;
    }
    else if(buftype == DIFF)
    {
      for(int n=0; n<NUM_NUMA_NODES; n++)
      {
#ifdef USE_MLSL
        wdiff_buf_[n] = (void*)MLSL::Environment::GetEnv().Alloc(s, 2097152);
#else
        wdiff_buf_[n] = (void*)libxsmm_aligned_malloc(s, 2097152);
#endif
        if(lp)
#ifdef USE_MLSL
          lpwdiff_buf_[n] = (void*)MLSL::Environment::GetEnv().Alloc(s/sizeof(libxsmm_bfloat16), 2097152);
#else
          lpwdiff_buf_[n] = (void*)libxsmm_aligned_malloc(s/sizeof(libxsmm_bfloat16), 2097152);
#endif
      }
      buf_ = wdiff_buf_[0];
      ptrptr = wdiff_buf_;
      if(lp)
        lptrptr = lpwdiff_buf_;
    }
    else if(buftype == HISTORY)
    {
      for(int n=0; n<NUM_NUMA_NODES; n++)
#ifdef USE_MLSL
        winc_buf_[n] = (void*)MLSL::Environment::GetEnv().Alloc(s, 2097152);
#else
        winc_buf_[n] = (void*)libxsmm_aligned_malloc(s, 2097152);
#endif
      buf_ = winc_buf_[0];
      ptrptr = winc_buf_;
    }
  }
  else if(tenType == "BACT")
  {
#ifdef USE_MLSL
    buf_ = (void*)MLSL::Environment::GetEnv().Alloc(s, 2097152);
#else
    buf_ = (void*)libxsmm_aligned_malloc(s, 2097152);
#endif
    bact_buf_ = buf_;
  }
  else if(tenType == "STATS")
  {
#if 1
    for(int n=0; n<NUM_NUMA_NODES; n++)
#ifdef USE_MLSL
      stats_buf_[n] = (void*)MLSL::Environment::GetEnv().Alloc(s, 2097152);
#else
      stats_buf_[n] = (void*)libxsmm_aligned_malloc(s, 2097152);
#endif
    buf_ = stats_buf_[0];
    ptrptr = stats_buf_;
#else
#ifdef USE_MLSL
    stats_buf_ = (void*)MLSL::Environment::GetEnv().Alloc(s, 2097152);
#else
    stats_buf_ = (void*)libxsmm_aligned_malloc(s, 2097152);
#endif
    buf_ = stats_buf_;
#endif
  }
  else if(tenType == "BIAS")
  {
    if(buftype == DATA)
    {
#if 1
      for(int n=0; n<NUM_NUMA_NODES; n++)
#ifdef USE_MLSL
        bias_buf_[n] = (void*)MLSL::Environment::GetEnv().Alloc(s, 2097152);
#else
        bias_buf_[n] = (void*)libxsmm_aligned_malloc(s, 2097152);
#endif
      buf_ = bias_buf_[0];
      ptrptr = bias_buf_;
#else
#ifdef USE_MLSL
      bias_buf_ = (void*)MLSL::Environment::GetEnv().Alloc(s, 2097152);
#else
      bias_buf_ = (void*)libxsmm_aligned_malloc(s, 2097152);
#endif
      buf_ = bias_buf_;
#endif
    }
    else if(buftype == DIFF)
    {
#if 1
      for(int n=0; n<NUM_NUMA_NODES; n++)
#ifdef USE_MLSL
        bidiff_buf_[n] = (void*)MLSL::Environment::GetEnv().Alloc(s, 2097152);
#else
        bidiff_buf_[n] = (void*)libxsmm_aligned_malloc(s, 2097152);
#endif
      buf_ = bidiff_buf_[0];
      ptrptr = bidiff_buf_;
#else
#ifdef USE_MLSL
      bidiff_buf_ = (void*)MLSL::Environment::GetEnv().Alloc(s, 2097152);
#else
      bidiff_buf_ = (void*)libxsmm_aligned_malloc(s, 2097152);
#endif
      buf_ = bidiff_buf_;
#endif
    }
    else if(buftype == HISTORY)
    {
#if 1
      for(int n=0; n<NUM_NUMA_NODES; n++)
#ifdef USE_MLSL
        biinc_buf_[n] = (void*)MLSL::Environment::GetEnv().Alloc(s, 2097152);
#else
        biinc_buf_[n] = (void*)libxsmm_aligned_malloc(s, 2097152);
#endif
      buf_ = biinc_buf_[0];
      ptrptr = biinc_buf_;
#else
#ifdef USE_MLSL
      biinc_buf_ = (void*)MLSL::Environment::GetEnv().Alloc(s, 2097152);
#else
      biinc_buf_ = (void*)libxsmm_aligned_malloc(s, 2097152);
#endif
      buf_ = biinc_buf_;
#endif
    }
  }

  // Total buffer size, including guard bands before and after each buffer (currntly 64 bytes long)
  *bufsize = s + (lp ? s/sizeof(libxsmm_bfloat16) : 0);

#if 0
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

  if(lp && lpweight_buf_==NULL)
  {
    printf("could not allocate low precision weights memory.. exiting\n");
    exit(-1);
  }

  if(solver_->getGlobalFlag())
  {
    if(tenType == "WEIGHT" && buftype == DIFF)
    {
      for(int n=0; n<NUM_NUMA_NODES; n++)
      {
        wt_lr_mult_[n] = (float*)libxsmm_aligned_malloc(total_weights_*sizeof(float), 2097152);
        if(wt_lr_mult_[n] != NULL)
        {
          float *ptr = wt_lr_mult_[n];

#ifdef _OPENMP
#pragma omp parallel
#endif
          {
            int tid = omp_get_thread_num();
            int ntps = num_threads_/NUM_NUMA_NODES;
            int s = tid/ntps;
            if(s == n && tid % ntps == 0)
              for(int i=0; i<total_weights_; i++)
                ptr[i] = 0.0;
          }
        }

        wt_decay_mult_[n] = (float*)libxsmm_aligned_malloc(total_weights_*sizeof(float), 2097152);
        if(wt_decay_mult_[n] != NULL)
        {
          float *ptr = wt_decay_mult_[n];

#ifdef _OPENMP
#pragma omp parallel
#endif
          {
            int tid = omp_get_thread_num();
            int ntps = num_threads_/NUM_NUMA_NODES;
            int s = tid/ntps;
            if(s == n && tid % ntps == 0)
              for(int i=0; i<total_weights_; i++)
                ptr[i] = 0.0;
          }
        }
      }
      lrptr = wt_lr_mult_[0];
      decptr = wt_decay_mult_[0];
    }
    else if(tenType == "BIAS" && buftype == DIFF)
    {
      for(int n=0; n<NUM_NUMA_NODES; n++)
      {
        bias_lr_mult_[n] = (float*)_mm_malloc(total_biases_*sizeof(float), 64);
        if(bias_lr_mult_[n] != NULL)
        {
          float *ptr = bias_lr_mult_[n];

#ifdef _OPENMP
#pragma omp parallel
#endif
          {
            int tid = omp_get_thread_num();
            int ntps = num_threads_/NUM_NUMA_NODES;
            int s = tid/ntps;
            if(s == n && tid % ntps == 0)
              for(int i=0; i<total_biases_; i++)
                ptr[i] = 0.0;
          }
        }

        bias_decay_mult_[n] = (float*)_mm_malloc(total_biases_*sizeof(float), 64);
        if(bias_decay_mult_[n] != NULL)
        {
          float *ptr = bias_decay_mult_[n];

#ifdef _OPENMP
#pragma omp parallel
#endif
          {
            int tid = omp_get_thread_num();
            int ntps = num_threads_/NUM_NUMA_NODES;
            int s = tid/ntps;
            if(s == n && tid % ntps == 0)
              for(int i=0; i<total_biases_; i++)
                ptr[i] = 0.0;
          }
        }
      }
      lrptr = bias_lr_mult_[0];
      decptr = bias_decay_mult_[0];
    }
  }
#endif

  if(ttp)
    memset(buf_, CANARY, START_GUARD_BAND);

  long long int bytes=0, lpbytes=0;
  int offset=0, bias_offset=0;

  //Set up tensor buffer pointers
  void* ptr = ttp ? buf_ + START_GUARD_BAND : buf_;
  void* lptr = lp ? lpweight_buf_[0] : NULL;
  void* lgptr = lp ? lpwdiff_buf_[0] : NULL;

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
    bytes = tBuf->getBufferSize();
    assert(ptr+bytes <= buf_+s);

    lpbytes = lp ? bytes/sizeof(libxsmm_bfloat16) : 0;

#ifndef USE_NUMA
    if(t->getType() == INPUT || t->getType() == ACT)
    {
      if(bytes > 0)
        memset(ptr, 0, bytes);
    }
#endif

    int dtype = tBuf->getDataType();

    // Set each node's tensor buffer pointers to the appropritate location in the global buffer
    if(tenType == "WEIGHT" || tenType == "BIAS" || tenType == "STATS")
    {
      if(buftype == DATA || buftype == DIFF)
      {
        tBuf->setBufferPtr(ptrptr);
        tBuf->setOffset(offset);
      }
      tBuf->setBuffer(ptr);

      if(lp)
      {
        if(buftype == DATA)
          tBuf->setLPBuffer(lptr);
        else if(buftype == DIFF)
          tBuf->setLPBuffer(lgptr);
        tBuf->setLPBufferPtr(lptrptr);
      }
    }
    else
      tBuf->setBuffer(ptr);

    // If weight or bias tensor, call corresponding intialization function (for training only)
    if(!is_inference_only())
    {
      int tType = t->getType();
      if(tType == CONVWEIGHT)
      {
        if(nntype == "FusedConvBN")
        {
          FusedConvBNNode *fcbn = dynamic_cast<FusedConvBNNode*>(t->getOwner());
          assert(bytes > 0);
          if(!load_from_checkpoint_)
          {
            fcbn->fillWeightBuffers(tBuf, buftype, bytes);
#if 0
            if(lp)
              convert_f32_bf16((float*)ptr, (libxsmm_bfloat16*)lptr, lpbytes/sizeof(libxsmm_bfloat16), 0);
#endif
          }
#if 0
          if(solver_->getGlobalFlag())
            if(buftype == DIFF)
              if(data_type_ == FLOAT)
                fcbn->fillWeightMultipliers(lrptr, decptr, bytes/sizeof(float));
              else if(data_type_ == BF16)
                fcbn->fillWeightMultipliers(lrptr, decptr, bytes/sizeof(libxsmm_bfloat16));
#endif
        }
        else if(nntype == "Convolution")
        {
          ConvNode* cn = dynamic_cast<ConvNode*>(t->getOwner());
          assert(bytes > 0);
          if(!load_from_checkpoint_)
          {
            cn->fillWeightBuffers(tBuf, buftype, bytes);
#if 0
            if(lp)
              convert_f32_bf16((float*)ptr, (libxsmm_bfloat16*)lptr, lpbytes/sizeof(libxsmm_bfloat16), 0);
#endif
          }

#if 0
          if(solver_->getGlobalFlag())
            if(buftype == DIFF)
              if(data_type_ == FLOAT)
                cn->fillWeightMultipliers(lrptr, decptr, bytes/sizeof(float));
              else if(data_type_ == BF16)
                cn->fillWeightMultipliers(lrptr, decptr, bytes/sizeof(libxsmm_bfloat16));
#endif
        }
      }
      else if(tType == CONVBIAS)
      {
        ConvNode* cn = dynamic_cast<ConvNode*>(t->getOwner());
        assert(bytes > 0);
        if(!load_from_checkpoint_)
          cn->fillBiasBuffers(tBuf, buftype, bytes);
#if 0
        if(solver_->getGlobalFlag())
          if(buftype == DIFF)
            cn->fillBiasMultipliers(lrptr, decptr, bytes/sizeof(float));
#endif
      }
      else if(tType == FCWEIGHT)
      {
        FCNode* fn = dynamic_cast<FCNode*>(t->getOwner());
        assert(bytes > 0);
        if(!load_from_checkpoint_)
        {
          fn->fillWeightBuffers(tBuf, buftype, bytes);
#if 0
          if(lp)
            convert_f32_bf16((float*)ptr, (libxsmm_bfloat16*)lptr, lpbytes/sizeof(libxsmm_bfloat16), 0);
#endif
        }

#if 0
        if(solver_->getGlobalFlag())
          if(buftype == DIFF)
            if(data_type_ == FLOAT)
              fn->fillWeightMultipliers(lrptr, decptr, bytes/sizeof(float));
            else if(data_type_ == BF16)
              fn->fillWeightMultipliers(lrptr, decptr, bytes/sizeof(libxsmm_bfloat16));
#endif
      }
      else if(tType == FCBIAS)
      {
        FCNode* fn = dynamic_cast<FCNode*>(t->getOwner());
        assert(bytes > 0);
        if(!load_from_checkpoint_)
          fn->fillBiasBuffers(tBuf, buftype, bytes);
#if 0
        if(solver_->getGlobalFlag())
          if(buftype == DIFF)
            fn->fillBiasMultipliers(lrptr, decptr, bytes/sizeof(float));
#endif
      }
      else if((tType == BNORMSCALE) || (tType == BNORMSHIFT))
      {
        if(nntype == "FusedConvBN")
        {
          FusedConvBNNode *fcbn = dynamic_cast<FusedConvBNNode*>(t->getOwner());
          assert(bytes > 0);
          if(!load_from_checkpoint_)
            fcbn->fillBuffer(tBuf, buftype, bytes);
#if 0
          if(solver_->getGlobalFlag())
            if(buftype == DIFF)
              fcbn->fillBiasMultipliers(lrptr, decptr, bytes/sizeof(float));
#endif
        }
        else if(nntype == "FusedBatchNorm")
        {
          FusedBNormNode* bn = dynamic_cast<FusedBNormNode*>(t->getOwner());
          assert(bytes > 0);
          if(!load_from_checkpoint_)
            bn->fillBuffer(tBuf, buftype, bytes);
#if 0
          if(solver_->getGlobalFlag())
            if(buftype == DIFF)
              bn->fillBiasMultipliers(lrptr, decptr, bytes/sizeof(float));
#endif
        }
      }
      else if((tType == BNORMMEAN) || (tType == BNORMVAR))
      {
        if(nntype == "FusedConvBN")
        {
          FusedConvBNNode *fcbn = dynamic_cast<FusedConvBNNode*>(t->getOwner());
          assert(bytes > 0);
          if(!load_from_checkpoint_)
            fcbn->fillBuffer(tBuf, buftype, bytes);
        }
        else if(nntype == "FusedBatchNorm")
        {
          FusedBNormNode* bn = dynamic_cast<FusedBNormNode*>(t->getOwner());
          assert(bytes > 0);
          if(!load_from_checkpoint_)
            bn->fillBuffer(tBuf, buftype, bytes);
        }
      }
    }

    if(bytes > 0)
    {
      ptr += bytes;
      if(lp)
        if(buftype == DATA)
          lptr += lpbytes;
#ifndef BF16_MLSL
        else if(buftype == DIFF)
          lgptr += lpbytes;
#endif

#ifdef BF16_MLSL
      if(tenType == "WEIGHT" && buftype == DATA)
        offset += bytes/sizeof(float);
      else if(tenType == "WEIGHT" && buftype == DIFF)
      {
        if(data_type_ == FLOAT)
          offset += bytes/sizeof(float);
        else if(data_type_ == BF16)
          offset += bytes/sizeof(libxsmm_bfloat16);
      }
#else
      if(tenType == "WEIGHT")
        offset += bytes/sizeof(float);
#endif
      else if((tenType == "BIAS" && (buftype == DATA || buftype == DIFF)) || tenType == "STATS")
        offset += bytes/sizeof(float);

#if 0
      if(solver_->getGlobalFlag())
      {
        if(tenType == "WEIGHT" && buftype == DIFF)
        {
          if(data_type_ == FLOAT)
          {
            lrptr += bytes/sizeof(float);
            decptr += bytes/sizeof(float);
          }
          else if(data_type_ == BF16)
          {
            lrptr += bytes/sizeof(libxsmm_bfloat16);
            decptr += bytes/sizeof(libxsmm_bfloat16);
          }
        }
        else if(tenType == "BIAS" && buftype == DIFF)
        {
          lrptr += bytes/sizeof(float);
          decptr += bytes/sizeof(float);
        }
      }
#endif

      assert(ptr <= buf_ + s);

      // For canary checking
      if(ttp)
      {
        memset(ptr, CANARY, END_GUARD_BAND);
        can_ptr.push_back(bytes);
        assert(can_ptr.size() <= num_canaries);
      }
      if(ttp)
        ptr += END_GUARD_BAND;
    }
    assert(ptr <= buf_ + s);
#if 0
    printf("ptr @ %p\n",ptr);
#endif
  }

  if(tenType=="WEIGHT" && buftype==DATA)
  {
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      int ntps = num_threads_/NUM_NUMA_NODES;
      int n = tid/ntps;
      int w = total_weights_;
      if(n != 0 && tid % ntps == 0)
      {
        float *wtptr = (float*)weight_buf_[n];

#pragma omp simd
        for(int i=0; i<w; i++)
          wtptr[i] = ((float*)weight_buf_[0])[i];
      }
    }

    if(lp)
      convert_f32_bf16((float**)weight_buf_, (libxsmm_bfloat16**)lpweight_buf_, total_weights_);
  }

  if(tenType=="WEIGHT" && buftype==DIFF)
  {
    if(data_type_ == FLOAT)
    {
#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        int tid = omp_get_thread_num();
        int ntps = num_threads_/NUM_NUMA_NODES;
        int n = tid/ntps;
        int w = total_weights_;
        if(n != 0 && tid % ntps == 0)
        {
          float *wdiff = (float*)wdiff_buf_[n];

#pragma omp simd
          for(int i=0; i<w; i++)
            wdiff[i] = ((float*)wdiff_buf_[0])[i];
        }
      }
    }
    else if(data_type_ == BF16)
    {
#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        int tid = omp_get_thread_num();
        int ntps = num_threads_/NUM_NUMA_NODES;
        int n = tid/ntps;
        int w = total_weights_;
        if(n != 0 && tid % ntps == 0)
        {
          libxsmm_bfloat16 *lpwdiff = (libxsmm_bfloat16*)lpwdiff_buf_[n];
          float *wdiff = (float*)wdiff_buf_[n];

#pragma omp simd
          for(int i=0; i<w; i++)
          {
            lpwdiff[i] = ((libxsmm_bfloat16*)lpwdiff_buf_[0])[i];
            wdiff[i] = ((float*)wdiff_buf_[0])[i];
          }
        }
      }
    }

#if 0
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      int ntps = num_threads_/NUM_NUMA_NODES;
      int n = tid/ntps;
      int w = total_weights_;
      if(n != 0 && tid % ntps == 0)
      {
        float *lrp = (float*)wt_lr_mult_[n];
        float *dcp = (float*)wt_decay_mult_[n];

#pragma omp simd
        for(int i=0; i<w; i++)
        {
          lrp[i] = ((float*)wt_lr_mult_[0])[i];
          dcp[i] = ((float*)wt_decay_mult_[0])[i];
        }
      }
    }
#endif
  }

  if(tenType=="WEIGHT" && buftype==HISTORY)
  {
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      int ntps = num_threads_/NUM_NUMA_NODES;
      int n = tid/ntps;
      int w = total_weights_;
      if(n != 0 && tid % ntps == 0)
      {
        float *inc = (float*)winc_buf_[n];

#pragma omp simd
        for(int i=0; i<w; i++)
          inc[i] = ((float*)winc_buf_[0])[i];
      }
    }
  }

#if 1
  if(tenType == "BIAS" && buftype == DATA)
  {
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      int ntps = num_threads_/NUM_NUMA_NODES;
      int n = tid/ntps;
      int b = total_biases_;

      if(n != 0 && tid % ntps == 0)
      {
        float *bias = (float*)bias_buf_[n];

#pragma omp simd
        for(int i=0; i<b; i++)
          bias[i] = ((float*)bias_buf_[0])[i];
      }
    }
  }


  if(tenType == "BIAS" && buftype == DIFF)
  {
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      int ntps = num_threads_/NUM_NUMA_NODES;
      int n = tid/ntps;
      int b = total_biases_;

      if(n != 0 && tid % ntps == 0)
      {
        float *bidiff = (float*)bidiff_buf_[n];
#if 0
        float *lrp = (float*)bias_lr_mult_[n];
        float *dcp = (float*)bias_decay_mult_[n];
#endif

#pragma omp simd
        for(int i=0; i<b; i++)
        {
          bidiff[i] = ((float*)bidiff_buf_[0])[i];
#if 0
          lrp[i] = ((float*)bias_lr_mult_[0])[i];
          dcp[i] = ((float*)bias_decay_mult_[0])[i];
#endif
        }
      }
    }
  }

  if(tenType == "BIAS" && buftype == HISTORY)
  {
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      int ntps = num_threads_/NUM_NUMA_NODES;
      int n = tid/ntps;
      int b = total_biases_;

      if(n != 0 && tid % ntps == 0)
      {
        float *biinc = (float*)biinc_buf_[n];

#pragma omp simd
        for(int i=0; i<b; i++)
          biinc[i] = ((float*)biinc_buf_[0])[i];
      }
    }
  }

  if(tenType == "STATS")
  {
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      int ntps = num_threads_/NUM_NUMA_NODES;
      int n = tid/ntps;
      int b = total_biases_;

      if(n != 0 && tid % ntps == 0)
      {
        float *stats = (float*)stats_buf_[n];

#pragma omp simd
        for(int i=0; i<b; i++)
          stats[i] = ((float*)stats_buf_[0])[i];
      }
    }
  }
#endif
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
  if(mode == TRAIN || mode == VAL)
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
  tenScratchBuf_ = tenScratch_->getBuf(DATA);
  tenScratchBuf_->setBufferPtr(scratch);

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
  // Look for tasks attached to nodes with no successors. Add them to the Execution Task Graph (etg) first.
  for(int i=numNodes-1; i>0; i--)
  {
    NNNode *nn = dynamic_cast<NNNode*>(ntg_[i]);
    Task* t = nn->getBasicTask(BASIC_TASK_FORW);

    if(nn->getNumNextNodes() == 0)
    {
      etg_[mode].push_back(t);
#ifndef NDEBUG
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
        etg_[VAL].push_back(t);
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

  allocate_memory("INPUT", inTList_, DATA, input_can_ptr, &ic, &total_input_size);
  if(global_node_id_ == 0)
    printf("Total input memory allocated %lld bytes\n", total_input_size);

  /**********************************************************************/
  /*** Allocate memory and set pointers for FORWARD ACTIVATION buffer ***/
  /**********************************************************************/
  long long int total_fact_size;
  allocate_memory("FACT", outTList_, DATA, fact_can_ptr, &fac, &total_fact_size);
  if(global_node_id_ == 0)
    printf("Total forward activation memory allocated %lld bytes\n", total_fact_size);

  /***********************************************************/
  /*** Allocate memory and set pointers for WEIGHTS buffer ***/
  /***********************************************************/
  long long int total_weight_size;
  allocate_memory("WEIGHT", wTList_, DATA, wt_can_ptr, &wtc, &total_weight_size);
  if(global_node_id_ == 0)
    printf("Total weights memory allocated %lld bytes\n", total_weight_size);

  /***********************************************************/
  /*** Allocate memory and set pointers for BIASES buffer ***/
  /***********************************************************/
  long long int total_bias_size;
  allocate_memory("BIAS", biasTList_, DATA, bias_can_ptr, &bic, &total_bias_size);
  if(global_node_id_ == 0)
    printf("Total bias memory allocated %lld bytes\n", total_bias_size);

  /***********************************************************/
  /*** Allocate memory and set pointers for STATS buffer ***/
  /***********************************************************/
  long long int total_stats_size;
  allocate_memory("STATS", statsTList_, DATA, stats_can_ptr, &sic, &total_stats_size);
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
    allocate_memory("BACT", outTList_, DIFF, bact_can_ptr, &bac, &total_bact_size);
    if(global_node_id_ == 0)
      printf("Total backward activation memory allocated %lld bytes\n", total_bact_size);
#else
    long long int total_bact_size = NDIFFS * max_fwd_buffer_size;
    allocate_gradient_tensor(outTList_, DIFF, NDIFFS, max_fwd_buffer_size);
    if(global_node_id_ == 0)
      printf("Total backward activation memory allocated %lld bytes\n", total_bact_size);
#endif

    /********************************************************************/
    /*** Allocate memory and set pointers for WEIGHT GRADIENTS buffer ***/
    /********************************************************************/
    long long int total_wdiff_size;
    allocate_memory("WEIGHT", wTList_, DIFF, wdiff_can_ptr, &wdc, &total_wdiff_size);
    if(global_node_id_ == 0)
      printf("Total weight gradient memory allocated %lld bytes\n", total_wdiff_size);

    /*********************************************************************/
    /*** Allocate memory and set pointers for WEIGHT INCREMENTS buffer ***/
    /*********************************************************************/
    long long int total_winc_size;
    allocate_memory("WEIGHT", wTList_, HISTORY, winc_can_ptr, &wic, &total_winc_size);
    if(global_node_id_ == 0)
      printf("Total weight increment memory allocated %lld bytes\n", total_winc_size);

    /********************************************************************/
    /*** Allocate memory and set pointers for BIAS GRADIENTS buffer ***/
    /********************************************************************/
    long long int total_bidiff_size;
    allocate_memory("BIAS", biasTList_, DIFF, bidiff_can_ptr, &bidc, &total_bidiff_size);
    if(global_node_id_ == 0)
      printf("Total bias gradient memory allocated %lld bytes\n", total_bidiff_size);

    /*********************************************************************/
    /*** Allocate memory and set pointers for BIAS INCREMENTS buffer ***/
    /*********************************************************************/
    long long int total_biinc_size;
    allocate_memory("BIAS", biasTList_, HISTORY, biinc_can_ptr, &biic, &total_biinc_size);
    if(global_node_id_ == 0)
      printf("Total bias increment memory allocated %lld bytes\n", total_biinc_size);

    total_bp_size = total_bact_size + total_wdiff_size + total_winc_size + total_bidiff_size + total_biinc_size;
  }

  long long int total_memory = total_input_size + total_fact_size + total_weight_size + total_bias_size + total_bp_size;
  if(global_node_id_ == 0)
    printf("Total tensor memory = %lld\n",total_memory);
}
