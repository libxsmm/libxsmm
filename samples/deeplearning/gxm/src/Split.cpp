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
  in_dtype = tenBotData_->getDataType();
  out_dtype = in_dtype;

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
    if(nname_.find("label") == nname_.npos)
      tenTopData_[i]->setDataType(in_dtype);
    else
      tenTopData_[i]->setDataType(DT_INT);

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
      tenBotDiff_->setDataType(in_dtype); 
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
  gparams_.in_data_type = in_dtype;
  gparams_.out_data_type = out_dtype;
  gparams_.num_threads = e->get_num_threads();

  eptr_ = e;

  configure(XSMM);
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

  int nImg = gparams_.batch_size;
  int ifm = gparams_.nInput;
  int fh = gparams_.iHeight;
  int fw = gparams_.iWidth;

  impl->set_bot_compute_engine(bot_cengine_);
  for(int i=0; i<top_.size(); i++)
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
    if(gparams_.in_data_type == DT_FLOAT)
    {
      float* bot = (float*) tenBotData_->getBuffer();
      Shape *bs = tenBot_->getShape();
      int size = bs->dims[0]*bs->dims[1]*bs->dims[2]*bs->dims[3];
      string s = nname_ + "_Inp";
      MeanOfLayer((char*)s.c_str(), bot, size);
    }
    else if(gparams_.in_data_type == DT_BF16)
    {
      int size = nImg*ifm*fh*fw;
      if(stptr == NULL)
        stptr = (float*)libxsmm_aligned_malloc(size*sizeof(float), 2097152);
      libxsmm_bfloat16* bot = (libxsmm_bfloat16*) tenBotData_->getBuffer();
      Shape *bs = tenBot_->getShape();
      libxsmm_convert_bf16_f32(bot, stptr, size);
      string s = nname_ + "_Inp";
      MeanOfLayer((char*)s.c_str(), stptr, size);
    }

    for(int i=0; i<top_.size(); i++)
    {
      if(gparams_.out_data_type == DT_FLOAT)
      {
        Shape *ts = tenTop_[i]->getShape();
        int size = ts->dims[0]*ts->dims[1]*ts->dims[2]*ts->dims[3];
        float* top = (float*)tenTopData_[i]->getBuffer();
        string s = nname_ + "_Outp_" + to_string(i);
        MeanOfLayer((char*)s.c_str(), top, size);
      }
      else if(gparams_.out_data_type == DT_BF16)
      {
        Shape *ts = tenTop_[i]->getShape();
        int size = ts->dims[0]*ts->dims[1]*ts->dims[2]*ts->dims[3];
        libxsmm_bfloat16* top = (libxsmm_bfloat16*)tenTopData_[i]->getBuffer();
        libxsmm_convert_bf16_f32(top, stptr, size);
        string s = nname_ + "_Outp_" + to_string(i);
        MeanOfLayer((char*)s.c_str(), stptr, size);
      }
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

  impl->backPropagate(tenTopDiff_, tenBotDiff_);

#ifdef CHECK_BLOWUP_FP32
  if(in_dtype == DT_FLOAT)
  {
    for(int i=0; i<16; i++)
    {
      float v = ((float*)tenBotDiff_->getBuffer())[i];
      if(isnan(v) || isinf(v))
      {
        printf("Warning! %s layer BP activations are NaN or Inf\n", nname_.c_str());
        exit(-1);
      }
    }
  }
  else if(in_dtype == DT_BF16)
  {
    libxsmm_convert_bf16_f32((libxsmm_bfloat16*)tenBotDiff_->getBuffer(), cbptr, 16);
    for(int i=0; i<16; i++)
    {
      if(isnan(cbptr[i]) || isinf(cbptr[i]))
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
    for(int i=0; i<tenTopDiff_.size(); i++)
    {
      if(tenTopDiff_[i] != NULL)
      {
        if(gparams_.out_data_type == DT_FLOAT)
        {
          float *ptr = (float*)tenTopDiff_[i]->getBuffer();
          Shape *ts = tenTop_[i]->getShape();
          int size = ts->dims[0]*ts->dims[1]*ts->dims[2]*ts->dims[3];
          string s = nname_ + "_delOutp_" + to_string(i);
          MeanOfLayer((char*)s.c_str(), ptr, size);
        }
        else if(gparams_.out_data_type == DT_BF16)
        {
          libxsmm_bfloat16 *ptr = (libxsmm_bfloat16*)tenTopDiff_[i]->getBuffer();
          Shape *ts = tenTop_[i]->getShape();
          int size = ts->dims[0]*ts->dims[1]*ts->dims[2]*ts->dims[3];
          libxsmm_convert_bf16_f32(ptr, stptr, size);
          string s = nname_ + "_delOutp_" + to_string(i);
          MeanOfLayer((char*)s.c_str(), stptr, size);
        }
      }
    }

    if(gparams_.in_data_type == DT_FLOAT)
    {
      float *ptr = (float*)tenBotDiff_->getBuffer();
      Shape *bs = tenBot_->getShape();
      size = bs->dims[0]*bs->dims[1]*bs->dims[2]*bs->dims[3];
      string s = nname_ + "_delInp";
      MeanOfLayer((char*)s.c_str(), ptr, size);
    }
    else if(gparams_.in_data_type == DT_BF16)
    {
      libxsmm_bfloat16 *ptr = (libxsmm_bfloat16*)tenBotDiff_->getBuffer();
      Shape *bs = tenBot_->getShape();
      size = bs->dims[0]*bs->dims[1]*bs->dims[2]*bs->dims[3];
      libxsmm_convert_bf16_f32(ptr, stptr, size);
      string s = nname_ + "_delInp";
      MeanOfLayer((char*)s.c_str(), stptr, size);
    }
  }
#endif
}
