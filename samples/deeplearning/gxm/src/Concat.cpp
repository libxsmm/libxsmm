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
#include "Concat.hpp"

using namespace std;
using namespace gxm;

ConcatNode::ConcatNode(ConcatParams* p, MLEngine* e) : NNNode(p, e)
{
  nname_ = p->get_node_name();
  ntype_ = p->get_node_type();
  mode_ = p->get_mode();
  bottom_ = p->get_bottom_names();
  top_ = p->get_top_names();
  bp_flag_ = p->get_bprop_flag();
  has_weights_ = false;
  bot_compute_engine_ = p->get_compute_engine();

  assert(top_.size() == 1);
  tenTop_ = new Tensor(top_[0]);
  assert(tenTop_ != NULL);
  tenTop_->setOwner(this);
  tenTop_->setType(ACT);
  tenTopData_ = tenTop_->getBuf(DATA);
  tenTopData_->setBufferType(DATA);

#ifdef DEBUG
  printf("bottom name %s\n",bottom_[0].c_str());
#endif

  Shape ts;
  shape_setzero(&ts);

  tenBot_.resize(bottom_.size());
  bot_cengine_.resize(bottom_.size());
  tenBotData_.resize(bottom_.size());

  for(int i=0; i<bottom_.size(); i++)
  {
    tenBot_[i] = e->get_tensor(bottom_[i], ACT);
    assert(tenBot_[i] != NULL);
    NNNode *pnn = (NNNode*)tenBot_[i]->getOwner();
    setPrevNode(pnn);
    pnn->set_top_compute_engine(p->get_compute_engine());

    bot_cengine_[i] = pnn->get_bot_compute_engine();
    tenBotData_[i] = tenBot_[i]->getBuf(DATA);
  }

  // number of concats
  gparams_.nInput.resize(bottom_.size());
  tenBotDiff_.resize(bottom_.size());

  int dtype = p->get_data_type();
  for(int i=0; i<bottom_.size(); i++)
  {
    // Get input tensor shape (bottom)
    Shape* bs = tenBot_[i]->getShape();
    assert(bs->ndims <= MAX_DIMS);

    ts.dims[1] += bs->dims[1];
    gparams_.nInput[i] = bs->dims[1];

    if(!e->is_inference_only())
    {
      if(NNNode::bp_flag_)
      {
        tenBotDiff_[i] = tenBot_[i]->addBuf(); // DIFF type and index
        tenBotDiff_[i]->setDataType(dtype);
        tenBotDiff_[i]->setBufferType(DIFF);

        long long int bsize = 1;
        for(int s=0; s<bs->ndims; s++)
          bsize = bsize*bs->dims[s];
        if(dtype == DT_FLOAT)
          bsize = bsize*sizeof(float);
        else if(dtype == DT_INT16)
          bsize = bsize*sizeof(short int);
        else if(dtype == DT_INT)
          bsize = bsize*sizeof(int);

        // Set the size of the input-gradient buffer
        tenBotDiff_[i]->setBufferSize(bsize);
      }
    }
    else
      tenBotDiff_[i] = NULL;
  }

  //Output tensor data type = input tensor data type
  tenTopData_->setDataType(dtype);

  Shape *bs = tenBot_[0]->getShape();
  ts.ndims = bs->ndims;
  ts.dims[0] = bs->dims[0];
  ts.dims[2] = bs->dims[2];
  ts.dims[3] = bs->dims[3];

  tenTop_->setShape(&ts);

  long long int tsize = 1;
  for(int s=0; s<ts.ndims; s++)
    tsize = tsize * ts.dims[s];

  if(dtype == DT_FLOAT)
    tsize = tsize*sizeof(float);
  else if(dtype == DT_INT)
    tsize = tsize*sizeof(int);

  tenTopData_->setBufferSize(tsize);

  // Register output tensor in tensor map
  bool inserted = e->register_tensor(NNNode::top_[0], ACT, tenTop_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",NNNode::top_[0].c_str());

  gparams_.bdims = bs->ndims;
  gparams_.tdims = ts.ndims;
  gparams_.batch_size = ts.dims[0];
  gparams_.nOutput = ts.dims[1];
  gparams_.iHeight = bs->dims[2];
  gparams_.iWidth = bs->dims[3];
  gparams_.oHeight = ts.dims[2];
  gparams_.oWidth = ts.dims[3];

  gparams_.data_type = dtype;
  gparams_.algType = p->get_algo_type();
  gparams_.num_threads = e->get_num_threads();

#ifdef GETSTATS
  count_ = 0;
#endif

  configure(p->get_compute_engine());
}

void ConcatNode::configure(int engine)
{
  switch(engine)
  {
    case XSMM:
      impl = new ConcatXSMM(&gparams_, engine);
      break;
  }
}

void ConcatNode::forwardPropagate()
{
#ifdef DEBUG
  float* bot;

  float* top = (float*)(tenTopData_->getBuffer());
  for(int i=0; i<tenBotData_.size(); i++)
  {
    bot = (float*)(tenBotData_[i]->getBuffer());
    printf("Executing FP %s: input %p, output %p\n",NNNode::nname_.c_str(), bot, top);
  }
#endif

  for(int i=0; i<tenBotData_.size(); i++)
    impl->set_bot_compute_engine(bot_cengine_[i]);
  impl->set_top_compute_engine(top_compute_engine_);
  impl->set_next_node_type(next_ntype_);
  impl->set_node_name(nname_);

  impl->forwardPropagate(tenBotData_, tenTopData_);

#ifdef CHECK_BLOWUP_FP32
  float* ptr = (float*)tenTopData_->getBuffer();
  for(int i=0; i<16; i++)
  {
    if(isnan(ptr[i]) || isinf(ptr[i]))
    {
      printf("Warning! %s layer FP activations are NaN or Inf\n", nname_.c_str());
      exit(-1);
    }
  }
#endif


#ifdef GETSTATS
#ifdef USE_MLSL
  size_t node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
  size_t node_id = 0;
#endif
  if(node_id==0 && count_ % STATFREQ == 0)
  {
    float* p, *pp, *ptr;
    int size;

    for(int i=0; i<tenBotData_.size(); i++)
    {
      p = (float*)tenBotData_[i]->getBuffer();
      pp = (float*)tenBotData_[i]->getPrivBuffer();
      ptr = (pp == NULL) ? p : pp;
      size = tenBotData_[i]->getBufferSize()/sizeof(float);
      MeanOfLayer((char*)bottom_[i].c_str(), ptr, size);
    }

    p = (float*)tenTopData_->getBuffer();
    pp = (float*)tenTopData_->getPrivBuffer();
    ptr = (pp == NULL) ? p : pp;
    size = tenTopData_->getBufferSize()/sizeof(float);
    MeanOfLayer((char*)top_[0].c_str(), ptr, size);
  }
#endif
}

void ConcatNode::backPropagate()
{
  tenTopDiff_ = tenTop_->getBuf(DIFF);

#ifdef DEBUG
  for(int i=0; i<tenBotDiff_.size(); i++)
    printf("Executing BP %s: deloutp %p, delinp %p\n",nname_.c_str(), tenTopDiff_->getBuffer(), tenBotDiff_[i]->getBuffer());
#endif

  impl->backPropagate(tenTopDiff_, tenBotDiff_);

#ifdef CHECK_BLOWUP_FP32
  float* ptr = (float*)tenTopDiff_->getBuffer();
  for(int i=0; i<16; i++)
  {
    if(isnan(ptr[i]) || isinf(ptr[i]))
    {
      printf("Warning! %s layer BP activations are NaN or Inf\n", nname_.c_str());
      exit(-1);
    }
  }
#endif

#ifdef GETSTATS
#ifdef USE_MLSL
  size_t node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
  size_t node_id = 0;
#endif

  if(node_id==0 && count_ % STATFREQ == 0)
  {
    float* p, *pp, *ptr;
    p = (float*)tenTopDiff_->getBuffer();
    pp = (float*)tenTopDiff_->getPrivBuffer();

    ptr = (pp == NULL) ? p : pp;
    int size = tenTopDiff_->getBufferSize()/sizeof(float);
    MeanOfLayer((char*)top_[0].c_str(), ptr, size);

    for(int i=0; i<tenBotDiff_.size(); i++)
    {
      p = (float*)tenBotDiff_[i]->getBuffer();
      pp = (float*)tenBotDiff_[i]->getPrivBuffer();
      ptr = (pp == NULL) ? p : pp;
      size = tenBotDiff_[i]->getBufferSize()/sizeof(float);
      MeanOfLayer((char*)bottom_[i].c_str(), ptr, size);
    }
    count_++;
  }
#endif
}
