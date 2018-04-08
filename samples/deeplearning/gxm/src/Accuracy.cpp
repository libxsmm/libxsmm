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


#include <omp.h>
#include "Accuracy.hpp"
#ifdef USE_MLSL
#include "mpi.h"
#endif

AccuracyNode::AccuracyNode(AccuracyParams* p, MLEngine* e) : NNNode(p, e)
{
  nname_ = p->get_node_name();
  ntype_ = p->get_node_type();
  mode_ = p->get_mode();
  top_ = p->get_top_names();
  bottom_ = p->get_bottom_names();

  has_weights_ = false;
  bp_flag_ = p->get_bprop_flag();

  tenBot_.resize(bottom_.size());
  tenBotData_.resize(bottom_.size());

  for(int i=0; i<NNNode::bottom_.size(); i++)
  {
    if((bottom_[i]).find("label") != bottom_[i].npos)
      tenBot_[i] = e->get_tensor(bottom_[i], LABEL);
    else
      tenBot_[i] = e->get_tensor(bottom_[i], ACT);
    assert(tenBot_[i] != NULL);
    setPrevNode((NNNode*)tenBot_[i]->getOwner());
    tenBotData_[i] = tenBot_[i]->getBuf(DATA);
  }

  // Get input tensor shape (bottom)
  Shape* bs = tenBot_[0]->getShape();
  assert(bs->ndims <= MAX_DIMS);

  shape_setzero(&ts_);

  ts_.ndims = 2;
  ts_.dims[0] = bs->dims[0]; // minibatch
  ts_.dims[1] = bs->dims[1]; // num output = num_input

  top_k_ = p->get_top_k();

  max_val.resize(top_k_ + 1);
  max_id.resize(top_k_ + 1);

  eptr_ = e;
  train_batch_count_ = 0;
  test_batch_count_ = 0;
  avg_train_acc_ = 0;
  avg_test_acc_ = 0;
}

void AccuracyNode::forwardPropagate()
{
#ifdef RETURNALL
  return;
#endif

  float* bot = (float*)(tenBotData_[0]->getBuffer());
  int* label = (int*)(tenBotData_[1]->getBuffer());

#ifdef DEBUG
  printf("Executing FP %s: input %p, label %p\n",NNNode::nname_.c_str(), bot, label);
#endif

  int accuracy = 0;
  int count = 0;
  for(int img=0; img<ts_.dims[0]; img++)
  {
    if(!bot_data_vec.empty()) bot_data_vec.clear();
    for(int k=0; k<ts_.dims[1]; k++)
      bot_data_vec.push_back(std::make_pair(bot[img*ts_.dims[1] + k], k));

    std::partial_sort(bot_data_vec.begin(), bot_data_vec.begin() + top_k_, bot_data_vec.end(), std::greater<std::pair<float, int> >());

    for(int k=0; k<top_k_; k++)
    {
      if(bot_data_vec[k].second == label[img])
      {
        accuracy++;
        break;
      }
    }
    count++;
  }

#ifdef USE_MLSL
  size_t num_nodes = MLSL::Environment::GetEnv().GetProcessCount();
  size_t node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
  size_t num_nodes = 1;
  size_t node_id = 0;
#endif

  if(eptr_->get_execution_mode() == TRAIN)
  {
    avg_train_acc_ += (double)accuracy/(double)count;
    train_batch_count_++;
    if(train_batch_count_ == eptr_->get_num_train_batches())
    {
      avg_train_acc_ = avg_train_acc_/(double)train_batch_count_;
#ifdef USE_MLSL
      MPI_Allreduce(MPI_IN_PLACE, &avg_train_acc_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      avg_train_acc_ = avg_train_acc_/num_nodes;
      if(node_id == 0)
        printf("Top-%d Minibatch training accuracy = %f\n", top_k_, avg_train_acc_);
#else
      printf("Top-%d Minibatch training accuracy = %f\n", top_k_, avg_train_acc_);
#endif
      train_batch_count_ = 0;
      avg_train_acc_ = 0;
    }
  }
  else if(eptr_->get_execution_mode() == TEST)
  {
    avg_test_acc_ += (double)accuracy/(double)count;
    test_batch_count_++;
    if(test_batch_count_ == eptr_->get_num_test_batches()*eptr_->get_num_test_views())
    {
      avg_test_acc_ = avg_test_acc_/(double)test_batch_count_;
#ifdef USE_MLSL
      MPI_Allreduce(MPI_IN_PLACE, &avg_test_acc_, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      avg_test_acc_ = avg_test_acc_/num_nodes;
      if(node_id == 0)
        printf("Top-%d Minibatch testing accuracy = %f\n", top_k_, avg_test_acc_);
#else
      printf("Top-%d Minibatch testing accuracy = %f\n", top_k_, avg_test_acc_);
#endif
      test_batch_count_ = 0;
      avg_test_acc_ = 0;
    }
  }
}

