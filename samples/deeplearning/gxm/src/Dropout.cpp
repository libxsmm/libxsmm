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
#include "Dropout.hpp"
#include "fillers.hpp"

#define PRIME_SEED 131

using namespace std;
using namespace gxm;

DropoutNode::DropoutNode(DropoutParams* p, MLEngine* e): NNNode(p, e)
{
  nname_ = p->get_node_name();
  ntype_ = p->get_node_type();
  mode_ = p->get_mode();
  bottom_ = p->get_bottom_names();
  top_ = p->get_top_names();
  bp_flag_ = p->get_bprop_flag();
  has_weights_ = false;

  assert((bottom_.size() == 1) && (top_.size() == 1));

  tenTop_ = new Tensor(top_[0]);
  assert(tenTop_ != NULL);
  tenTop_->setOwner(this);
  tenTop_->setType(ACT);
  tenTopData_ = tenTop_->getBuf(DATA);
  tenTopData_->setBufferType(DATA);

#ifdef DEBUG
  printf("bottom name %s\n",bottom_[0].c_str());
#endif

  tenBot_ = e->get_tensor(bottom_[0], ACT);
  assert(tenBot_ != NULL);
  setPrevNode((NNNode*)tenBot_->getOwner());
  tenBotData_ = tenBot_->getBuf(DATA);

  //Output tensor data type = input tensor data type
  int dtype = p->get_data_type();
  tenTopData_->setDataType(dtype);

  // Get input tensor shape (bottom)
  Shape* bs = tenBot_->getShape();
  assert(bs->ndims <= MAX_DIMS);

  Shape ts;
  shape_setzero(&ts);

  ts.ndims = bs->ndims;
  for(int i=0; i < bs->ndims; i++)
    ts.dims[i] = bs->dims[i];

  tenTop_->setShape(&ts);

  long long int tsize = 1;
  for(int i=0; i<ts.ndims; i++)
    tsize = tsize*ts.dims[i];

  // Mask to select neuron activations to be dropped out
  tenMask_ = new int[tsize];

  if(dtype == DT_FLOAT)
    tsize = tsize*sizeof(float);
  else if(dtype == DT_INT16)
    tsize = tsize*sizeof(short int);

  // Set the logical size of the tensor buffer for bufId=0 (forward data buffer).
  // Note: we have no knowledge of the machine parameters here, so effectively this is single-machine config
  tenTopData_->setBufferSize(tsize);

  if(!e->is_inference_only())
  {
    if(bp_flag_)
    {
      tenBotDiff_ = tenBot_->addBuf(); // DIFF type and index
      tenBotDiff_->setDataType(dtype);
      tenBotDiff_->setBufferType(DIFF);

      long long int bsize = 1;
      for(int i=0; i<bs->ndims; i++)
        bsize = bsize*bs->dims[i];
      if(dtype == DT_FLOAT)
        bsize = bsize*sizeof(float);
      else if(dtype == DT_INT)
        bsize = bsize*sizeof(int);

      // Set the size of the input-gradient buffer
      tenBotDiff_->setBufferSize(bsize);
    }
  }
  else
    tenBotDiff_ = NULL;

  // Compute scale via dropout_ratio
  threshold_ = p->get_dropout_ratio();
  if(threshold_ != 0.5)
  {
    printf("Support for threshold %f not implemented! Resetting to 0.5\n",threshold_);
    threshold_ = 0.5;
  }
  scale_ = 1./(1 - threshold_);

  // Register output tensor in tensor map
  bool inserted = e->register_tensor(top_[0], ACT, tenTop_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",NNNode::top_[0].c_str());

  gparams_.batch_size = bs->dims[0];
  gparams_.nInput = bs->dims[1];
  gparams_.nOutput = gparams_.nInput;
  gparams_.iHeight = bs->dims[2];
  gparams_.iWidth = bs->dims[3];
  gparams_.oHeight = ts.dims[2];
  gparams_.oWidth = ts.dims[3];
  gparams_.data_type = dtype;

  gparams_.num_threads = e->get_num_threads();

  seeds = new unsigned int[gparams_.num_threads];
  for(int i=0; i<gparams_.num_threads; i++)
    seeds[i] = PRIME_SEED + i;

  eptr_ = e;
};

void DropoutNode::forwardPropagate()
{
#ifdef RETURNALL
  return;
#endif

  float* bot = (float*)(tenBotData_->getBuffer());
  float* top = (float*)(tenTopData_->getBuffer());
  int *mask = (int *)tenMask_;
 // unsigned int *seeds = tenSeeds_;

#ifdef DEBUG
  printf("Executing FP %s: input %p, output %p\n",NNNode::nname_.c_str(), bot, top);
  printf("Inputs: %d\n",gparams_.nInput);
  printf("Outputs: %d\n",gparams_.nOutput);
#endif

  int M = gparams_.batch_size;
  int N = gparams_.nOutput;
  int H = gparams_.oHeight;
  int W = gparams_.oWidth;

  if(eptr_->get_execution_mode() == TRAIN)
  {

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < M*N*H*W; i++)
    {
      int r = rand_r(&seeds[omp_get_thread_num()]);
      if(r%2 == 0)
        top[i] = 0;
      else
        top[i] = bot[i] * scale_;
    }
  }
  else
  {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < M*N*H*W; i++)
      top[i] = bot[i];
  }

#ifdef DEBUG
  MeanOfLayer((char*)bottom_[0].c_str(), bot, M*N*H*W);
  MeanOfLayer((char*)top_[0].c_str(), top, M*N*H*W);
#endif
}

void DropoutNode::backPropagate()
{
#ifdef REUTRNALL
  return;
#endif

  int M = gparams_.batch_size;
  int N = gparams_.nOutput;
  int H = gparams_.oHeight;
  int W = gparams_.oWidth;

  TensorBuf *tenTopDiff = tenTop_->getBuf(DIFF);
  float *gtop = (float*)(tenTopDiff->getBuffer());
  assert(gtop != NULL);

  float* gbot = (float*)(tenBotDiff_->getBuffer());

  int *mask = (int *)tenMask_;

#ifdef DEBUG
  printf("Executing BP %s: grad_output %p, grad_input %p\n",NNNode::nname_.c_str(), gtop, gbot);
  printf("Grad Outputs: %d\n", N*H*W);
  printf("Grad Inputs: %d\n", N*H*W);
#endif

  assert(eptr_->get_execution_mode() == TRAIN);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < M*N*H*W; i++)
    gbot[i] = gtop[i] * mask[i] * scale_;

#ifdef DEBUG
  MeanOfLayer((char*)bottom_[0].c_str(), gtop, M*N*H*W);
  MeanOfLayer((char*)top_[0].c_str(), gbot, M*N*H*W);
#endif
}
