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
#include "SoftmaxLoss.hpp"
#ifdef USE_MLSL
#include "mpi.h"
#endif

#define SMAXLOSS_TYPE_DIRECT 0
#define LOSSFREQ 100

SoftmaxLossNode::SoftmaxLossNode(SoftmaxLossParams* p, MLEngine* e) : NNNode(p, e)
{
  nname_ = p->get_node_name();
  ntype_ = p->get_node_type();
  mode_ = p->get_mode();
  top_ = p->get_top_names();
  bottom_ = p->get_bottom_names();

  has_weights_ = false;
  bp_flag_ = true;

  //Create output tensor
  tenTop_ = new Tensor(top_[0]);
  assert(tenTop_ != NULL);
  tenTop_->setOwner(this);
  tenTop_->setType(ACT);
  tenTopData_ = tenTop_->getBuf(DATA);
  tenTopData_->setBufferType(DATA);
  int dtype = p->get_data_type();
  Shape *bs;

  tenBot_.resize(bottom_.size());
  tenBotData_.resize(bottom_.size());

  for(int i=0; i<bottom_.size(); i++)
  {
    if((bottom_[i]).find("label") != bottom_[i].npos)
      tenBot_[i] = e->get_tensor(bottom_[i], LABEL);
    else
      tenBot_[i] = e->get_tensor(bottom_[i], ACT);
    assert(this->tenBot_[i] != NULL);
    tenBotData_[i] = tenBot_[i]->getBuf(DATA);
    if((bottom_[i]).find("label") == bottom_[i].npos)
    {
      setPrevNode((NNNode*)tenBot_[i]->getOwner());
      // Get input tensor shape (bottom)
      bs = tenBot_[i]->getShape();
    }
  }

  //Output tensor data type = input tensor data type
  tenTopData_->setDataType(dtype);

  assert(bs->ndims <= MAX_DIMS);

  shape_setzero(&ts_);

  ts_.ndims = 2;
  ts_.dims[0] = bs->dims[0]; // minibatch
  ts_.dims[1] = bs->dims[1]; // num output = num_input

  tenTop_->setShape(&ts_);

  long long int size = 1;
  for(int i=0; i<ts_.ndims; i++)
    size *= ts_.dims[i];

  if(dtype == DT_FLOAT)
    size = size*sizeof(float);
  else if(dtype == DT_INT)
    size = size*sizeof(int);

  tenTopData_->setBufferSize(size);

  loss_weight_ = p->get_loss_weight();

  if(!e->is_inference_only())
  {
    if(bp_flag_)
    {
      for(int i=0; i<bottom_.size(); i++)
      {
        if((bottom_[i]).find("label") == bottom_[i].npos)
        {
          tenBotDiff_ = tenBot_[i]->addBuf();
          tenBotDiff_->setDataType(dtype);
          tenBotDiff_->setBufferType(DIFF);

          size = 1;
          for(int i=0; i<bs->ndims; i++)
            size = size*bs->dims[i];
          if(dtype == DT_FLOAT)
            size = size*sizeof(float);
          else if(dtype == DT_INT)
            size = size*sizeof(int);

          // Set the size of the input-gradient buffer
          tenBotDiff_->setBufferSize(size);

          break;
        }
      }
    }
  }

  // Register output tensor in tensorMap
  bool inserted = e->register_tensor(top_[0], ACT, this->tenTop_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",top_[0].c_str());

  gparams_.node_name = nname_;
  gparams_.batch_size = bs->dims[0];
  gparams_.nInput = bs->dims[1];
  gparams_.nOutput = ts_.dims[1];
  gparams_.loss_weight = loss_weight_[0];

  gparams_.num_threads = e->get_num_threads();

  eptr_ = e;
#ifdef USE_MLSL
  node_id_ = MLSL::Environment::GetEnv().GetProcessIdx();
  num_nodes_ = MLSL::Environment::GetEnv().GetProcessCount();
#else
  node_id_ = 0;
  num_nodes_ = 1;
#endif

  test_loss_ = 0;

  impl = new SMaxLossLoop(&gparams_);
}

void SoftmaxLossNode::forwardPropagate()
{
#ifdef RETURNALL
  return;
#endif

  struct timeval tvss, tvse, tvcs, tvce;
  float* bot = (float*)(tenBotData_[0]->getBuffer());
  int* label = (int*)(tenBotData_[1]->getBuffer());
  float* top = (float*)(tenTopData_->getBuffer());

#ifdef TIMING
  gettimeofday(&tvss, NULL);
#endif

  impl->forwardPropagate(tenBotData_[0], tenBotData_[1], tenTopData_);

#ifdef TIMING
  gettimeofday(&tvse, NULL);
  double smaxtime = (tvse.tv_sec + tvse.tv_usec*1e-6) - (tvss.tv_sec + tvss.tv_usec*1e-6);
  if(node_id_ == 0)
    printf("Softmax FP time: %f ms\n",smaxtime*1000);
#endif

#ifdef GETSTATS
  if(node_id_ == 0)
  {
    MeanOfLayer("SMFPIn", bot, gparams_.batch_size*gparams_.nInput);
    MeanOfLayer("SMFPOut", top, gparams_.batch_size*gparams_.nOutput);
    MeanOfLayer("SMFPLabel", label, gparams_.batch_size);
  }
#endif

#ifdef TIMING
  gettimeofday(&tvcs, NULL);
#endif
#ifdef USE_MLSL
  MPI_Allreduce(MPI_IN_PLACE, &gparams_.loss, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#endif

#ifdef TIMING
  gettimeofday(&tvce, NULL);
  double allrtime = (tvce.tv_sec + tvce.tv_usec*1e-6) - (tvcs.tv_sec + tvcs.tv_usec*1e-6);
  if(node_id_ == 0)
    printf("Softmax all-reduce time: %f ms\n",allrtime*1000);
#endif
  if(node_id_ == 0 && eptr_->get_current_batch() % LOSSFREQ == 0)
  {
    gparams_.loss = gparams_.loss/num_nodes_;
    printf("loss = %.15f (weighted loss = %.15f)\n", gparams_.loss, gparams_.loss*gparams_.loss_weight);
  }
}

void SoftmaxLossNode::backPropagate()
{
#ifdef RETURNALL
  return;
#endif

  float* gbot = (float*)(tenBotDiff_->getBuffer());

  int* label = (int*)(tenBotData_[1]->getBuffer());
  float* top = (float*)(tenTopData_->getBuffer());

#ifdef GETSTATS
  printf("Executing BP %s: Grad output %p, label %p Grad input %p\n",NNNode::nname_.c_str(), top, label, gbot);
  if(node_id_ == 0)
    MeanOfLayer("BPIn", top, gparams_.batch_size*gparams_.nOutput);
#endif

  impl->set_num_nodes(num_nodes_);
  impl->backPropagate(tenTopData_, tenBotData_[1], tenBotDiff_);

#ifdef GETSTATS
  if(node_id_ == 0)
    MeanOfLayer("BPOut", gbot, gparams_.batch_size*gparams_.nInput);
#endif
}
