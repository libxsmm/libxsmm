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
#include "Split.hpp"

using namespace std;
using namespace gxm;

SplitNode::SplitNode(SplitParams *p, MLEngine *e) : NNNode(p, e)
{
  nname_ = p->get_node_name();
  ntype_ = p->get_node_type();
  mode_ = p->get_mode();
  bottom_ = p->get_bottom_names();
  top_ = p->get_top_names();
  bp_flag_ = p->get_bprop_flag();
  has_weights_ = false;
  bot_compute_engine_ = p->get_compute_engine();

  if(nname_.find("label") != nname_.npos)
    tenBot_ = e->get_tensor(bottom_[0], LABEL);
  else
    tenBot_ = e->get_tensor(bottom_[0], ACT);

  assert(tenBot_ != NULL);
  NNNode *pnn = (NNNode*)tenBot_->getOwner();
  setPrevNode(pnn);
  bot_cengine_ = pnn->get_bot_compute_engine();
  pnn->set_top_compute_engine(p->get_compute_engine());
  pnn->set_next_node_type(ntype_);

  tenBotData_ = tenBot_->getBuf(DATA);
  int out_dtype = p->get_data_type();
  int in_dtype = tenBotData_->getDataType();

  Shape* bs = tenBot_->getShape();
  assert(bs->ndims <= MAX_DIMS);

  // number of splits
  gparams_.nOutput.resize(top_.size());
  tenTop_.resize(top_.size());
  tenTopData_.resize(top_.size());

  for(int i=0; i<top_.size(); i++)
  {
    tenTop_[i] = new Tensor(top_[i]);
    assert(tenTop_[i] != NULL);
    tenTop_[i]->setOwner(this);
    if(nname_.find("label") != nname_.npos)
      tenTop_[i]->setType(LABEL);
    else
      tenTop_[i]->setType(ACT);

    Shape ts;
    shape_setzero(&ts);

    ts.ndims = bs->ndims;
    for(int j=0; j<bs->ndims; j++)
      ts.dims[j] = bs->dims[j];

    tenTop_[i]->setShape(&ts);

    tenTopData_[i] = tenTop_[i]->getBuf(DATA);
    tenTopData_[i]->setBufferType(DATA);
    tenTopData_[i]->setDataType(in_dtype);

#if 0
    long long int tsize = 1;
    for(int d=0; d<ts.ndims; d++)
      tsize = tsize*ts.dims[d];

    if(dtype == DT_FLOAT)
      tsize = tsize*sizeof(float);
    else if(dtype == DT_INT)
      tsize = tsize*sizeof(int);

    // Set the logical size of the tensor buffer for bufId=0 (forward data buffer).
    // Note: we have no knowledge of the machine parameters here, so effectively this is single-machine config
    tenTopData_[i]->setBufferSize(tsize);
#endif

    bool inserted;

    if(nname_.find("label") != nname_.npos)
      inserted = e->register_tensor(NNNode::top_[i], LABEL, tenTop_[i]);
    else
      inserted = e->register_tensor(NNNode::top_[i], ACT, tenTop_[i]);

    if(!inserted)
      printf("Warning: Tensor %s already registered\n",NNNode::top_[i].c_str());
  }

  if(!e->is_inference_only())
  {
    if(bp_flag_)
    {
      tenBotDiff_ = tenBot_->addBuf();
      tenBotDiff_->setDataType(DT_FLOAT); //@TODO: This is a HACK. Must be fixed
      tenBotDiff_->setBufferType(DIFF);
      tenBotDiff_->setBufferSize(tenBotData_->getBufferSize());
    }
  }

  gparams_.bdims = bs->ndims;
  gparams_.tdims = bs->ndims;
  gparams_.batch_size = bs->dims[0];
  gparams_.nInput = bs->dims[1];
  for(int i=0; i<top_.size(); i++)
    gparams_.nOutput[i] = bs->dims[1];
  gparams_.iHeight = bs->dims[2];
  gparams_.iWidth = bs->dims[3];
  gparams_.oHeight = gparams_.iHeight;
  gparams_.oWidth = gparams_.iWidth;
  gparams_.num_threads = e->get_num_threads();

  eptr_ = e;
#if 0
#ifdef USE_MLSL
  if(nname_.find("label") == nname_.npos)
  {
    MLSL::DataType dt = MLSL::DT_FLOAT;
    MLSL::ComputeOpRegInfo *myRegInfo;
    myRegInfo = new MLSL::ComputeOpRegInfo(MLSL::COMP_OP_TYPE_BCAST);
    myRegInfo->SetName(nname_.c_str());

    if(gparams_.iWidth > 0 && gparams_.iHeight > 0)
      myRegInfo->AddInputFeatureMap(gparams_.nInput, gparams_.iWidth*gparams_.iHeight, dt);
    else
      myRegInfo->AddInputFeatureMap(gparams_.nInput, 1, dt);

    for(int i=0; i<gparams_.nOutput.size(); i++)
    {
      if(gparams_.oWidth > 0 && gparams_.oHeight > 0)
        myRegInfo->AddOutputFeatureMap(gparams_.nOutput[i], gparams_.oWidth*gparams_.oHeight, dt);
      else
        myRegInfo->AddOutputFeatureMap(gparams_.nOutput[i], 1, dt);
    }

    myRegInfo->Validate();
    this->op_ = new MLSL::ComputeOp(myRegInfo, e->get_distribution());
    delete myRegInfo;
  }
#endif
#endif

#if 0
  configure(p->get_compute_engine());
#else
  configure(XSMM);
#endif

}

void SplitNode::configure(int engine)
{
  switch(engine)
  {
    case XSMM:
      impl = new SplitLoop(&gparams_, engine);
      break;
  }
}

void SplitNode::forwardPropagate()
{
  impl->set_bot_compute_engine(bot_cengine_);
  for(int i=0; i<top_.size(); i++)
    impl->set_top_compute_engine(top_compute_engine_);

  impl->forwardPropagate(tenBotData_, tenTopData_);

#ifdef CHECK_BLOWUP_FP32
  for(int i=0; i<top_.size(); i++)
  {
    float* ptr = (float*)tenTopData_[i]->getBuffer();
    for(int j=0; j<16; j++)
    {
      if(isnan(ptr[j]) || isinf(ptr[j]))
      {
        printf("Warning! %s layer FP activations are NaN or Inf\n", nname_.c_str());
        exit(-1);
      }
    }
  }
#endif


#ifdef GETSTATS
#ifdef USE_MLSL
  size_t node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
  size_t node_id = 0;
#endif
  if(node_id == 0 && eptr_->get_current_batch() % STATFREQ == 0)
  {
    float* bot = (float*) tenBotData_->getBuffer();
#ifdef DEBUG
    for(int i=0; i<top_.size(); i++)
    {
      float* top = (float*)tenTopData_[i]->getBuffer();
      printf("Executing FP %s: bot %p, top %p\n",nname_.c_str(), bot, top);
    }
#endif
    Shape *bs = tenBot_->getShape();
    int size = bs->dims[0]*bs->dims[1]*bs->dims[2]*bs->dims[3];
    string s = nname_ + "_Inp";
    MeanOfLayer((char*)s.c_str(), bot, size);

    for(int i=0; i<top_.size(); i++)
    {
      Shape *ts = tenTop_[i]->getShape();
      int size = ts->dims[0]*ts->dims[1]*ts->dims[2]*ts->dims[3];
      float* top = (float*)tenTopData_[i]->getBuffer();
      s = nname_ + "_Outp_" + to_string(i);
      MeanOfLayer((char*)s.c_str(), top, size);
    }
  }
#endif
}

void SplitNode::backPropagate()
{
  int num_gtops=0;
  int nni;

  for(int i=0; i<tenTop_.size(); i++)
  {
    if(tenTop_[i]->getBuf(DIFF) != NULL)
    {
      nni = i;
      num_gtops++;
    }
  }

  tenTopDiff_.resize(num_gtops);
  if(num_gtops == 1)
    tenTopDiff_[0] = tenTop_[nni]->getBuf(DIFF);
  else
  {
    for(int i=0; i<num_gtops; i++)
      tenTopDiff_[i] = tenTop_[i]->getBuf(DIFF);
  }

#ifdef DEBUG
  float *p, *pp, *ptr;
  for(int i=0; i<tenTopDiff_.size(); i++)
  {
    if(tenTopDiff_[i] != NULL)
    {
      p = (float*)tenTopDiff_[i]->getBuffer();
      pp = (float*)tenTopDiff_[i]->getPrivBuffer();
      ptr = (pp == NULL) ? p : pp;
      printf("Executing BP %s: gtop %p, gbot %p\n",nname_.c_str(), ptr, tenBotDiff_->getBuffer());
    }
  }
#endif

  impl->backPropagate(tenTopDiff_, tenBotDiff_);

#ifdef CHECK_BLOWUP_FP32
  for(int i=0; i<num_gtops; i++)
  {
    float* ptr = (float*)tenTopDiff_[i]->getBuffer();
    for(int j=0; j<16; j++)
    {
      if(isnan(ptr[j]) || isinf(ptr[j]))
      {
        printf("Warning! %s layer BP activations are NaN or Inf\n", nname_.c_str());
        exit(-1);
      }
    }
  }
#endif
#ifdef GETSTATS
#ifdef USE_MLSL
  size_t node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
  size_t node_id = 0;
#endif
  if(node_id == 0 && eptr_->get_current_batch() % STATFREQ == 0)
  {
    int size;
#ifndef DEBUG
    float *p, *pp, *ptr;
#endif
    for(int i=0; i<tenTopDiff_.size(); i++)
    {
      if(tenTopDiff_[i] != NULL)
      {
        p = (float*)tenTopDiff_[i]->getBuffer();
        pp = (float*)tenTopDiff_[i]->getPrivBuffer();
        ptr = (pp == NULL) ? p : pp;

        Shape *ts = tenTop_[i]->getShape();
        size = ts->dims[0]*ts->dims[1]*ts->dims[2]*ts->dims[3];
        string s = nname_ + "_delOutp_" + to_string(i);
        MeanOfLayer((char*)s.c_str(), ptr, size);
      }
    }

    p = (float*)tenBotDiff_->getBuffer();
    pp = (float*)tenBotDiff_->getPrivBuffer();
    ptr = (pp == NULL) ? p : pp;
    Shape *bs = tenBot_->getShape();
    size = bs->dims[0]*bs->dims[1]*bs->dims[2]*bs->dims[3];

    string s = nname_ + "_delInp";
    MeanOfLayer((char*)s.c_str(), ptr, size);
  }
#endif
}
