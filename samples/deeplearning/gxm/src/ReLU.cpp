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
#include "ReLU.hpp"

using namespace std;
using namespace gxm;

ReLUNode::ReLUNode(ReLUParams* p, MLEngine* e): NNNode(p, e)
{
  nname_ = p->get_node_name();
  ntype_ = p->get_node_type();
  mode_ = p->get_mode();
  bottom_ = p->get_bottom_names();
  top_ = p->get_top_names();
  bp_flag_ = p->get_bprop_flag();
  has_weights_ = false;
  bot_compute_engine_ = p->get_compute_engine();

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
  NNNode *pnn = (NNNode*)tenBot_->getOwner();
  setPrevNode(pnn);
  pnn->set_top_compute_engine(p->get_compute_engine());
  bot_cengine_ = pnn->get_bot_compute_engine();

  tenBotData_ = tenBot_->getBuf(DATA);

  //Output tensor data type = input tensor data type
  int dtype = p->get_data_type();
  tenTopData_->setDataType(dtype);

  // Get input tensor shape (bottom)
  Shape* bs = tenBot_->getShape();
  assert(bs->ndims <= MAX_DIMS);

  tenTop_->setShape(bs);

  long long int tsize = 1;
  for(int i=0; i<bs->ndims; i++)
    tsize = tsize*bs->dims[i];

  if(dtype == DT_FLOAT)
    tsize = tsize*sizeof(float);
  else if(dtype == DT_INT)
    tsize = tsize*sizeof(int);

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

  // Register output tensor in tensor map
  bool inserted = e->register_tensor(top_[0], ACT, tenTop_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",top_[0].c_str());

  gparams_.bdims = gparams_.tdims = bs->ndims;
  gparams_.batch_size = bs->dims[0];
  gparams_.node_name = nname_;
  gparams_.nInput = bs->dims[1];
  gparams_.nOutput = gparams_.nInput;
  if(bs->ndims == 5)
  {
    gparams_.iDepth = gparams_.iHeight = gparams_.iWidth = bs->dims[2];
    gparams_.oDepth = gparams_.oHeight = gparams_.oWidth = bs->dims[3];
  }
  else if(bs->ndims == 4)
  {
    gparams_.iDepth = gparams_.oDepth = 0;
    gparams_.iHeight = gparams_.oHeight = bs->dims[2];
    gparams_.iWidth = gparams_.oWidth = bs->dims[3];
  }

  gparams_.negative_slope = p->get_negative_slope();

  gparams_.data_type = dtype;
  gparams_.algType = p->get_algo_type();
  gparams_.num_threads = e->get_num_threads();

  configure(p->get_compute_engine());

  eptr_ = e;
};

void ReLUNode::configure(int engine)
{
  switch(engine)
  {
    case XSMM:
      impl = new ReLUXSMM(&gparams_, engine);
      break;
  }
}

void ReLUNode::forwardPropagate()
{
#ifdef DEBUG
  float* bot = (float*)(tenBotData_->getBuffer());
  float* top = (float*)(tenTopData_->getBuffer());

  printf("Executing FP %s: input %p, output %p\n",NNNode::nname_.c_str(), bot, top);
  if(gparams_.bdims > 4)
    printf("Inputs: %d x %d x %d x %d\n",gparams_.nInput, gparams_.iDepth, gparams_.iHeight, gparams_.iWidth);
  else if(gparams_.bdims > 3)
    printf("Inputs: %d x %d x %d\n",gparams_.nInput, gparams_.iHeight, gparams_.iWidth);

  if(gparams_.tdims > 4)
    printf("Outputs: %d x %d x %d x %d\n",gparams_.nOutput, gparams_.oDepth, gparams_.oHeight, gparams_.oWidth);
  else if(gparams_.tdims > 3)
    printf("Outputs: %d x %d x %d\n",gparams_.nOutput, gparams_.oHeight, gparams_.oWidth);
#endif

  impl->set_bot_compute_engine(bot_cengine_);
  impl->set_top_compute_engine(top_compute_engine_);
  impl->forwardPropagate(tenBotData_, tenTopData_);

#ifdef GETSTATS
#ifdef USE_MLSL
  size_t node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
  size_t node_id = 0;
#endif
  if(node_id == 0 && eptr_->get_current_batch() % STATFREQ == 0)
  {
    float *ptr = (float*)tenBotData_->getBuffer();
    float *pptr = (float*)tenBotData_->getPrivBuffer();
    float *p = (pptr == NULL) ? ptr : pptr;
    string s = nname_ + "_Inp";
    MeanOfLayer((char*)s.c_str(), p, gparams_.batch_size*gparams_.nInput* gparams_.iHeight*gparams_.iWidth);

    ptr = (float*)tenTopData_->getBuffer();
    pptr = (float*)tenTopData_->getPrivBuffer();
    p = (pptr == NULL) ? ptr : pptr;
    s = nname_ + "_Outp";
    MeanOfLayer((char*)s.c_str(), p, gparams_.batch_size*gparams_.nOutput* gparams_.oHeight*gparams_.oWidth);
  }
#endif
}

void ReLUNode::backPropagate()
{

  tenTopDiff_ = tenTop_->getBuf(DIFF);

#ifdef DEBUG
  float *gtop = (float*)(tenTopDiff_->getBuffer());
  assert(gtop != NULL);
  float* gbot = (float*)(tenBotDiff_->getBuffer());
  float* bot = (float*)(tenBotData_->getBuffer());

  printf("Executing BP %s: grad_output %p, grad_input %p\n",NNNode::nname_.c_str(), gtop, gbot);
  if(gparams_.bdims > 4)
  {
    printf("Inputs: %d x %d x %d x %d\n",gparams_.nInput, gparams_.iDepth, gparams_.iHeight, gparams_.iWidth);
    printf("Grad Inputs: %d x %d x %d x %d\n",gparams_.nInput, gparams_.iDepth, gparams_.iHeight, gparams_.iWidth);
  }
  else if(gparams_.bdims > 3)
  {
    printf("Inputs: %d x %d x %d\n",gparams_.nInput, gparams_.iHeight, gparams_.iWidth);
    printf("Grad Inputs: %d x %d x %d\n",gparams_.nInput, gparams_.iHeight, gparams_.iWidth);
  }

  if(gparams_.tdims > 4)
    printf("Grad Outputs: %d x %d x %d x %d\n",gparams_.nOutput, gparams_.oDepth, gparams_.oHeight, gparams_.oWidth);
  else if(gparams_.tdims > 3)
    printf("Grad Outputs: %d x %d x %d\n",gparams_.nOutput, gparams_.oHeight, gparams_.oWidth);
#endif

  impl->backPropagate(tenBotData_, tenTopDiff_, tenBotDiff_);

#ifdef GETSTATS
#ifdef USE_MLSL
  size_t node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
  size_t node_id = 0;
#endif
  if(node_id == 0 && eptr_->get_current_batch() % STATFREQ == 0)
  {
    float *ptr = (float*)tenTopDiff_->getBuffer();
    float *pptr = (float*)tenTopDiff_->getPrivBuffer();
    float *p = (pptr == NULL) ? ptr : pptr;

    string s = nname_ + "_delOutp";
    MeanOfLayer((char*)s.c_str(), p, gparams_.batch_size*gparams_.nOutput* gparams_.oHeight*gparams_.oWidth);

    ptr = (float*)tenBotDiff_->getBuffer();
    pptr = (float*)tenBotDiff_->getPrivBuffer();
    p = (pptr == NULL) ? ptr : pptr;

    s = nname_ + "_delInp";
    MeanOfLayer((char*)s.c_str(), p, gparams_.batch_size*gparams_.nInput* gparams_.iHeight*gparams_.iWidth);
  }
#endif
}

