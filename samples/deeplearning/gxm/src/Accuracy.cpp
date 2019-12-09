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
    int label_value = label[img];
    float prob_true_class = bot[img*ts_.dims[1] + label_value];
    int num_better_predictions = -1;

    for(int k=0; k < ts_.dims[1] && num_better_predictions < top_k_; k++)
      num_better_predictions += bot[img*ts_.dims[1]+k] >= prob_true_class;

    if(num_better_predictions < top_k_)
      accuracy++;

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
  else if(eptr_->get_execution_mode() == TEST || eptr_->get_execution_mode() == VAL)
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

