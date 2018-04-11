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
#include "FusedBNorm.hpp"

using namespace std;
using namespace gxm;

FusedBNormNode::FusedBNormNode(FusedBNormParams* p, MLEngine* e): NNNode(p, e)
{
  nname_ = p->get_node_name();
  ntype_ = p->get_node_type();
  mode_ = p->get_mode();
  bottom_ = p->get_bottom_names();
  top_ = p->get_top_names();
  bp_flag_ = p->get_bprop_flag();

  has_weights_ = true;
  bot_compute_engine_ = p->get_compute_engine();

  tenTop_ = new Tensor(top_[0]);
  assert(tenTop_ != NULL);
  tenTop_->setOwner(this);
  tenTop_->setType(ACT);
  tenTopData_ = tenTop_->getBuf(DATA);
  tenTopData_->setBufferType(DATA);

#ifdef DEBUG
  for(int i=0; i<bottom_.size(); i++)
    printf("bottom name %s\n",bottom_[i].c_str());
#endif

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

  int in_dtype = tenBotData_[0]->getDataType();
  int out_dtype = p->get_data_type();

  tenTopData_->setDataType(out_dtype);

  vector<int> vp = p->get_pads();
  vector<int> ivp = p->get_ipads();
  vector<int> vs = p->get_strides();

  // Get input tensor shape (bottom)
  // Even if there are two inputs for eltwise operation, their shapes are the same. So, pick the first one

  Shape* bs = tenBot_[0]->getShape();
  assert(bs->ndims <= MAX_DIMS);

  Shape ts;
  shape_setzero(&ts);
  ts.ndims = bs->ndims;
  ts.dims[0] = bs->dims[0];
  ts.dims[1] = bs->dims[1];
  ts.dims[2] = bs->dims[2]/vs[0];
  ts.dims[3] = bs->dims[3]/vs[1];

  tenTop_->setShape(&ts);

  int telem = ts.dims[0] * ts.dims[1] * (ts.dims[2] + 2*vp[0]) * (ts.dims[3] + 2*vp[1]);
  long long int tsize;

  if(in_dtype == DT_FLOAT && out_dtype == DT_FLOAT)
    tsize = telem*sizeof(float);
  else if(in_dtype == DT_FLOAT && out_dtype == DT_DFP16)
    tsize = telem*sizeof(float) + telem*sizeof(short);
  else if(in_dtype == DT_DFP16 && out_dtype == DT_DFP16)
    tsize = telem*sizeof(short);

  tenTopData_->setBufferSize(tsize);

  Shape sss;
  shape_setzero(&sss);
  sss.ndims = 1;
  sss.dims[0] = bs->dims[1];

  scale_ = top_[0] + "_scale";
  tenScale_ = new Tensor(scale_);
  assert(tenScale_ != NULL);
  tenScale_->setOwner(this);
  tenScale_->setType(BNORMSCALE);
  tenScale_->setShape(&sss);
  tenScaleData_ = tenScale_->getBuf(DATA);
  tenScaleData_->setDataType(DT_FLOAT); //TODO: Eventually move to dfp16 gamma. Currently it is FP32
  tenScaleData_->setBufferType(DATA);

  telem = sss.dims[0];
  tsize = telem*sizeof(float);
  tenScaleData_->setBufferSize(tsize);

  shift_ = top_[0] + "_shift";
  tenShift_ = new Tensor(shift_);
  assert(tenShift_ != NULL);
  tenShift_->setOwner(this);
  tenShift_->setType(BNORMSHIFT);
  tenShift_->setShape(&sss);
  tenShiftData_ = tenShift_->getBuf(DATA);
  tenShiftData_->setDataType(DT_FLOAT); //TODO: Eventually move to dfp16 beta. Currently it is FP32
  tenShiftData_->setBufferType(DATA);

  tenShiftData_->setBufferSize(tsize);

  // number of inputs
  gparams_.nInput.resize(bottom_.size());
  tenBotDiff_.resize(bottom_.size());

  mean_ = top_[0] + "_mean";
  tenMean_ = new Tensor(mean_);
  assert(tenMean_ != NULL);
  tenMean_->setOwner(this);
  tenMean_->setType(BNORMMEAN);
  tenMean_->setShape(&sss);
  tenMeanData_ = tenMean_->getBuf(DATA);
  tenMeanData_->setDataType(DT_FLOAT);
  tenMeanData_->setBufferType(DATA);
  tenMeanData_->setBufferSize(tsize);

  rstdev_ = top_[0] + "_rstdev";
  tenRstdev_ = new Tensor(rstdev_);
  assert(tenRstdev_ != NULL);
  tenRstdev_->setOwner(this);
  tenRstdev_->setType(BNORMRSTDEV);
  tenRstdev_->setShape(&sss);
  tenRstdevData_ = tenRstdev_->getBuf(DATA);
  tenRstdevData_->setDataType(DT_FLOAT);
  tenRstdevData_->setBufferType(DATA);
  tenRstdevData_->setBufferSize(tsize);

  for(int i=0; i<bottom_.size(); i++)
  {
    // Get input tensor shape (bottom)
    Shape* bs = tenBot_[i]->getShape();
    assert(bs->ndims <= MAX_DIMS);

    gparams_.nInput[i] = bs->dims[1];

    if(!e->is_inference_only())
    {
      if(bp_flag_)
      {
        tenBotDiff_[i] = tenBot_[i]->addBuf(); // DIFF type and index
        tenBotDiff_[i]->setDataType(out_dtype); // this is a hack; actually, it should be in_dtype..
        tenBotDiff_[i]->setBufferType(DIFF);

        int belem = bs->dims[0] * bs->dims[1] * (bs->dims[2] + 2*ivp[0]) * (bs->dims[3] + 2*ivp[1]);
        long long int bsize;

        if(in_dtype == DT_FLOAT && out_dtype == DT_FLOAT)
          bsize = belem*sizeof(float);
        else if(in_dtype == DT_FLOAT && out_dtype == DT_DFP16)
          bsize = belem*sizeof(float) + belem*sizeof(short);
        else if(in_dtype == DT_DFP16 && out_dtype == DT_DFP16)
          // BN computes delinput in FP32, then quantizes to DFP16; so need space for both buffers
          // DFP16 buffer needs extra element to hold scaling factor after quantization
          bsize = belem*sizeof(short);

        // Set the size of the input-gradient buffer
        tenBotDiff_[i]->setBufferSize(bsize);
      }
    }
    else
      tenBotDiff_[i] = NULL;
  }

  if(!e->is_inference_only())
  {
    if(has_weights_)
    {
      tenScaleDiff_ = tenScale_->addBuf();
      tenScaleDiff_->setDataType(DT_FLOAT);
      tenScaleDiff_->setBufferType(DIFF);
      tenScaleDiff_->setBufferSize(tsize);

      tenScaleInc_ = tenScale_->addBuf();
      tenScaleInc_->setDataType(DT_FLOAT);
      tenScaleInc_->setBufferType(HISTORY);
      tenScaleInc_->setBufferSize(tsize);

      tenShiftDiff_ = tenShift_->addBuf();
      tenShiftDiff_->setDataType(DT_FLOAT);
      tenShiftDiff_->setBufferType(DIFF);
      tenShiftDiff_->setBufferSize(tsize);

      tenShiftInc_ = tenShift_->addBuf();
      tenShiftInc_->setDataType(DT_FLOAT);
      tenShiftInc_->setBufferType(HISTORY);
      tenShiftInc_->setBufferSize(tsize);
    }
  }
  else
  {
    tenScaleDiff_ = NULL;
    tenShiftDiff_ = NULL;
    tenScaleInc_ = NULL;
    tenShiftInc_ = NULL;
  }

  // Register output tensor in tensor map
  bool inserted = e->register_tensor(top_[0], ACT, tenTop_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",top_[0].c_str());

  inserted = e->register_tensor(scale_, BNORMSCALE, tenScale_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",scale_.c_str());

  inserted = e->register_tensor(shift_, BNORMSHIFT, tenShift_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",shift_.c_str());

  inserted = e->register_tensor(mean_, BNORMMEAN, tenMean_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",mean_.c_str());

  inserted = e->register_tensor(rstdev_, BNORMRSTDEV, tenRstdev_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",rstdev_.c_str());

  gparams_.bdims = gparams_.tdims = bs->ndims;
  gparams_.batch_size = bs->dims[0];
  gparams_.node_name = nname_;
  gparams_.nOutput = bs->dims[1];
  gparams_.iHeight = gparams_.oHeight = bs->dims[2];
  gparams_.iWidth = gparams_.oWidth = bs->dims[3];
  gparams_.pad_h = vp[0];
  gparams_.pad_w = vp[1];
  gparams_.ipad_h = ivp[0];
  gparams_.ipad_w = ivp[1];
  gparams_.stride_h = vs[0];
  gparams_.stride_w = vs[1];

  gparams_.mmf = p->get_mmf();
  gparams_.eps = p->get_eps();
  gparams_.relu = p->get_relu();
  gparams_.bwd_relu = p->get_bwd_relu();
  gparams_.eltwise = p->get_eltwise();

  gparams_.in_data_type = in_dtype;
  gparams_.out_data_type = out_dtype;
  gparams_.algType = p->get_algo_type();
  gparams_.num_threads = e->get_num_threads();
  gparams_.use_global_stats = false;

  lr_mult_ = p->get_lr_mult();
  decay_mult_ = p->get_decay_mult();

  configure(p->get_compute_engine());

  solver_ = e->getSolver();
  eptr_ = e;

#ifdef USE_MLSL
  MLSL::DataType dt = MLSL::DT_FLOAT;
  MLSL::OperationRegInfo *myRegInfo;
  MLSL::Session *s = eptr_->get_session();
  myRegInfo = s->CreateOperationRegInfo(MLSL::OT_BIAS);
  myRegInfo->AddParameterSet(gparams_.nOutput, 1, dt, false);
  myRegInfo->AddParameterSet(gparams_.nOutput, 1, dt, false);

  myRegInfo->Validate();
  size_t opIdx = s->AddOperation(myRegInfo, e->get_distribution());
  this->op_ = s->GetOperation(opIdx);
  s->DeleteOperationRegInfo(myRegInfo);
#endif
};

void FusedBNormNode::configure(int engine)
{
  switch(engine)
  {
    case XSMM:
      impl = new FusedBNormXSMM(&gparams_, engine);
      break;
  }
}

void FusedBNormNode::fillBuffer(TensorBuf *tBuf, int buftype, long long int bytes)
{
  int ttype = tBuf->getTensor()->getType();
  int dtype = tBuf->getDataType();
  void *ptr = tBuf->getBuffer();
  float value;
  if(ttype==BNORMSCALE && buftype == DATA)
  {
    if(nname_.find("bn3") == nname_.npos)
      value = 1;
    else
      value = 0.;
  }
  else
    value = 0;
  initConstantBuffer(ptr, dtype, bytes, "CONSTANT", value);
}

void FusedBNormNode::fillBiasMultipliers(float* lr, float* decay, long long int size)
{
  for(int i=0; i < size; i++)
  {
    lr[i] = lr_mult_;
    decay[i] = decay_mult_;
  }
}

void FusedBNormNode::Checkpoint(TensorBuf *tBuf, string name, string format)
{
  long long int bytes = tBuf->getBufferSize();
  int dtype = tBuf->getDataType();

  FILE* f;
  void* ptr;
  size_t pos;

  while((pos = name.find("/", 10)) != name.npos)
    name.replace(pos, 1, 1, '_');

  float* p = (float*)tBuf->getBuffer();
  bool no_checkpt = false;
  for(int i=0; i<16; i++)
  {
    if(isnan(p[i]) || isinf(p[i]))
    {
      no_checkpt = true;
      printf("Warning! %s Did not checkpoint! Weights are NaNs or Inf\n", nname_.c_str());
      break;
    }
  }

  if(!no_checkpt)
  {
    if(format.compare("binary") == 0)
    {
      f = fopen(name.c_str(), "wb");
      if(f != NULL)
      {
        ptr = tBuf->getBuffer();

        size_t b = fwrite(ptr, 1, bytes, f);
        assert((long long int)b == bytes);
      }
      else
        printf("Warning: could not checkpoint to file %s\n",name.c_str());
    }
    else
    {
      f = fopen(name.c_str(), "w");
      if(f != NULL)
      {
        ptr = tBuf->getBuffer();
        if(dtype == DT_FLOAT)
        {
          for(int i=0; i<bytes/sizeof(float); i++)
            fprintf(f, "%f\n", *((float*)ptr + i));
        }
        else if(dtype == DT_INT16)
        {
          for(int i=0; i<bytes/sizeof(short int); i++)
            fprintf(f, "%d\n", *((short int*)ptr + i));
        }
        else
          printf("Warning: could not checkpoint to file %s\n",name.c_str());
      }
    }
    if(f != NULL)
    {
      fflush(f);
      fclose(f);
    }
  }
}
void FusedBNormNode::forwardPropagate()
{
#ifdef DEBUG
  {
    int offset = gparams_.batch_size * gparams_.nInput * gparams_.iHeight * gparams_.iWidth;
    float* bot_r = (float*)(tenBotData_[0]->getBuffer());
    float* mean = bot_r + offset;
    float* mean2 = mean + gparams_.nInput;

    float* bot_l = (float*)(tenBotData_[1]->getBuffer());

    float* top = (float*)(tenTopData_->getBuffer());

    printf("Executing FP %s: input_r %p, mean %p, mean2 %p, input_l %p, output %p\n",NNNode::nname_.c_str(), bot_r, mean, mean2, bot_l, top);
    printf("Inputs: %d x %d x %d\n",gparams_.nInput, gparams_.iHeight, gparams_.iWidth);
    printf("Outputs: %d x %d x %d\n",gparams_.nOutput, gparams_.oHeight, gparams_.oWidth);
  }
#endif

  impl->set_bot_compute_engine(bot_cengine_[0]);
  impl->set_top_compute_engine(top_compute_engine_);
  impl->set_node_name(nname_);

  gmean_ = (float*)tenMeanData_->getBuffer();
  grstd_ = (float*)tenRstdevData_->getBuffer();

  if(first_fp)
  {
    int size = gparams_.batch_size * gparams_.nOutput * (gparams_.oHeight/gparams_.stride_h + 2*gparams_.pad_h) * (gparams_.oWidth/gparams_.stride_w + 2*gparams_.pad_w);

    if((gparams_.in_data_type == DT_FLOAT && gparams_.out_data_type == DT_FLOAT)
      || (gparams_.in_data_type == DT_FLOAT && gparams_.out_data_type == DT_DFP16))
    {
      float* ptr = (float*)tenTopData_->getBuffer();

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<size; i++)
        ptr[i] = 0;

      if(gparams_.in_data_type == DT_FLOAT && gparams_.out_data_type == DT_DFP16)
        tenTopData_->setLPBuffer(ptr + size);
    }
    else if(gparams_.in_data_type == DT_DFP16 && gparams_.out_data_type == DT_DFP16)
    {
      short* ptr = (short*)tenTopData_->getBuffer();

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<size+1; i++)
        ptr[i] = 0;
    }
    first_fp = false;
  }

  impl->forwardPropagate(tenBotData_, tenScaleData_, tenShiftData_, gmean_, grstd_, tenTopData_, 0);

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
  unsigned int node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
  unsigned int node_id = 0;
#endif
  if(node_id == 0 && eptr_->get_current_batch() % STATFREQ == 0)
  {
    float *ptr = (float*)tenBotData_[0]->getBuffer();
    float *pptr = (float*)tenBotData_[0]->getPrivBuffer();
    float *p = (pptr == NULL) ? ptr : pptr;
    string s = nname_ + "_r_Inp";
    MeanOfLayer((char*)s.c_str(), p, gparams_.batch_size*gparams_.nInput[0]* (gparams_.iHeight + 2*gparams_.ipad_h) * (gparams_.iWidth + 2*gparams_.ipad_w) );

    if(gparams_.nInput.size() > 1)
    {
      ptr = (float*)tenBotData_[1]->getBuffer();
      pptr = (float*)tenBotData_[1]->getPrivBuffer();
      p = (pptr == NULL) ? ptr : pptr;
      s = nname_ + "_l_Inp";
      if(p != NULL)
        MeanOfLayer((char*)s.c_str(), p, gparams_.batch_size*gparams_.nInput[1]* gparams_.iHeight*gparams_.iWidth);
    }

#if 1
    s = nname_ + "_gammap";
    float* gamma = (float*)tenScaleData_->getBuffer();
    MeanOfLayer((char*)s.c_str(), gamma, gparams_.nInput[0]);

    s = nname_ + "_betap";
    float* beta = (float*)tenShiftData_->getBuffer();
    MeanOfLayer((char*)s.c_str(), beta, gparams_.nInput[0]);

#ifdef BNTEST
    s = nname_ + "_meanp";
    int offset = gparams_.batch_size*gparams_.nInput[0]* (gparams_.iHeight + 2*gparams_.ipad_h) * (gparams_.iWidth + 2*gparams_.ipad_2);
    float* m = ptr + offset;
    MeanOfLayer((char*)s.c_str(), m, gparams_.nInput[0]);

    s = nname_ + "_mean2p";
    float* m2 = m + gparams_.nInput[0];
    MeanOfLayer((char*)s.c_str(), m2, gparams_.nInput[0]);
#endif

#if 0
    s = nname_ + "_gmeanp";
    MeanOfLayer((char*)s.c_str(), gmean, gparams_.nInput[0]);

    s = nname_ + "_grstdp";
    MeanOfLayer((char*)s.c_str(), grstd, gparams_.nInput[0]);
#endif
#endif

#if 1
    ptr = (float*)tenTopData_->getBuffer();
    s = nname_ + "_Outp";
    int size = gparams_.batch_size * gparams_.nOutput * (gparams_.oHeight/gparams_.stride_h + 2*gparams_.pad_h) * (gparams_.oWidth/gparams_.stride_w + 2*gparams_.pad_w);
    MeanOfLayer((char*)s.c_str(), ptr, size);
#endif
  }
#endif
}

void FusedBNormNode::backPropagate()
{
  tenTopDiff_ = tenTop_->getBuf(DIFF);

#ifdef DEBUG
  int offset = gparams_.batch_size * gparams_.nInput[0] * gparams_.iHeight * gparams_.iWidth;
  float *gtop = (float*)(tenTopDiff_->getBuffer());
  assert(gtop != NULL);
  float* gbot = (float*)(tenBotDiff_[0]->getBuffer());

  float* bot = (float*)(tenBotData_[0]->getBuffer());
  float* mean = bot + offset;

  printf("Executing BP %s: grad_output %p, mean %p, grad_input %p\n",NNNode::nname_.c_str(), gtop, mean, gbot);

  printf("Inputs: %d x %d x %d\n",gparams_.nInput[0], gparams_.iHeight, gparams_.iWidth);
  printf("Grad Inputs: %d x %d x %d\n",gparams_.nInput[0], gparams_.iHeight, gparams_.iWidth);
  printf("Grad Outputs: %d x %d x %d\n",gparams_.nOutput, gparams_.oHeight, gparams_.oWidth);
#endif

  if(first_bp)
  {
    const int size = gparams_.batch_size * gparams_.nInput[0] * (gparams_.iHeight + 2*gparams_.ipad_h) *
      (gparams_.iWidth + 2*gparams_.ipad_w);

    int in_dtype = tenBotData_[0]->getDataType();
    int out_dtype = tenTopData_->getDataType();

    if((in_dtype == DT_FLOAT && out_dtype == DT_FLOAT) ||
        (in_dtype == DT_FLOAT && out_dtype == DT_DFP16))
    {
      float* gbot0 = (float*)(tenBotDiff_[0]->getBuffer());
      float* gbot1 = gparams_.eltwise ? (float*)(tenBotDiff_[1]->getBuffer()) : NULL;

#if 1
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<size; i++)
        gbot0[i] = 0.0f;

      if(in_dtype == DT_FLOAT && out_dtype == DT_DFP16)
        tenBotDiff_[0]->setLPBuffer(gbot0 + size);

      if(gbot1)
      {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i=0; i<size; i++)
          gbot1[i] = 0.0f;

        if(in_dtype == DT_FLOAT && out_dtype == DT_DFP16)
          tenBotDiff_[1]->setLPBuffer(gbot1 + size);
      }
#endif
    }
    else if(in_dtype == DT_DFP16 && out_dtype == DT_DFP16)
    {
      short* gbot0 = (short*)(tenBotDiff_[0]->getBuffer());
      short* gbot1 = gparams_.eltwise ? (short*)(tenBotDiff_[1]->getBuffer()) : NULL;

#if 1
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<size+1; i++)
        gbot0[i] = 0.0f;

      if(gbot1)
      {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i=0; i<size+1; i++)
          gbot1[i] = 0.0f;
      }
#endif
    }

    first_bp = false;
  }

  impl->backPropagate(tenBotData_, tenTopData_, tenScaleData_, tenTopDiff_, tenScaleDiff_, tenShiftDiff_, tenBotDiff_, 0);

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
  unsigned int node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
  unsigned int node_id = 0;
#endif
  string s;
  float *ptr, *pptr, *p;
  if(node_id == 0 && eptr_->get_current_batch() % STATFREQ == 0) //&& gparams_.pad_h && nname_ == "node_64_2_bn1")
  {
#if 1
    ptr = (float*)tenTopDiff_->getBuffer();
    pptr = (float*)tenTopDiff_->getPrivBuffer();
    int size = gparams_.batch_size * gparams_.nOutput * (gparams_.oHeight/gparams_.stride_h + 2*gparams_.pad_h) * (gparams_.oWidth/gparams_.stride_w + 2*gparams_.pad_w);
    p = (pptr == NULL) ? ptr : pptr;
    s = nname_ + "_delOutp";
    MeanOfLayer((char*)s.c_str(), p, size);
#endif

    s = nname_ + "_delgammap";
    float* delgamma = (float*)tenScaleDiff_->getBuffer();
    MeanOfLayer((char*)s.c_str(), delgamma, gparams_.nOutput);

    s = nname_ + "_delbetap";
    float* delbeta = (float*)tenShiftDiff_->getBuffer();
    MeanOfLayer((char*)s.c_str(), delbeta, gparams_.nOutput);

    ptr = (float*)tenBotDiff_[0]->getBuffer();
    s = nname_ + "_delInp";
    MeanOfLayer((char*)s.c_str(), ptr, gparams_.batch_size*gparams_.nInput[0]* (gparams_.iHeight + 2*gparams_.ipad_h) * (gparams_.iWidth + 2*gparams_.ipad_w));

  }
#endif
}

void FusedBNormNode::weightUpdate()
{
#if 1
#ifdef USE_MLSL
  this->op_->GetParameterSet(0)->StartGradientComm(tenScaleDiff_->getBuffer());
  this->op_->GetParameterSet(1)->StartGradientComm(tenShiftDiff_->getBuffer());
#endif
#endif
}

void FusedBNormNode::solverStep()
{
  if(!impl->get_bpdone())
    printf("Warning! Solver task before BP/WU task... old gradients\n");

  float *delgamma = (float*)tenScaleDiff_->getBuffer();
  float *delbeta = (float*)tenShiftDiff_->getBuffer();

#if 1
#ifdef USE_MLSL
  void *mptr = op_->GetParameterSet(0)->WaitGradientComm();
  if(mptr != NULL && mptr != delgamma)
    memcpy((void*)delgamma, mptr, gparams_.nOutput*sizeof(float));

  mptr = op_->GetParameterSet(1)->WaitGradientComm();
  if(mptr != NULL && mptr != delbeta)
    memcpy((void*)delbeta, mptr, gparams_.nOutput*sizeof(float));
#endif
#endif

#ifdef CHECK_BLOWUP_FP32
  for(int i=0; i<16; i++)
  {
    if(isnan(delgamma[i]) || isinf(delgamma[i]))
    {
      printf("Warning! %s layer Solver gamma gradients are NaN or Inf\n", nname_.c_str());
      exit(-1);
    }
  }
  for(int i=0; i<16; i++)
  {
    if(isnan(delbeta[i]) || isinf(delbeta[i]))
    {
      printf("Warning! %s layer Solver beta gradients are NaN or Inf\n", nname_.c_str());
      exit(-1);
    }
  }
#endif

#ifdef GETSTATS
#ifdef USE_MLSL
  unsigned int node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
  unsigned int node_id = 0;
#endif
  if(node_id == 0 && eptr_->get_current_batch() % STATFREQ == 0) // && gparams_.pad_h && nname_ == "node_64_2_bn1")
  {
    string s = nname_ + "_delgammap_bef";
    MeanOfLayer((char*)s.c_str(), delgamma, gparams_.nOutput);
    s = nname_ + "_delbetap_bef";
    MeanOfLayer((char*)s.c_str(), delbeta, gparams_.nOutput);
  }
#endif

  impl->set_bpdone(false);

  float* gamma = (float*)tenScaleData_->getBuffer();
  float* igamma =(float*)tenScaleInc_->getBuffer();
  float* beta = (float*)tenShiftData_->getBuffer();
  float* ibeta = (float*)tenShiftInc_->getBuffer();

  if(solver_->getGlobalFlag())
    return;

  solver_->applyUpdate(gamma, igamma, delgamma, gparams_.nOutput, 1.0, 0.0);
  solver_->applyUpdate(beta, ibeta, delbeta, gparams_.nOutput, 1.0, 0.0);

#ifdef GETSTATS
  //unsigned int node_id = MLSL::Environment::GetEnv().GetProcessIdx();
  if(node_id == 0 && eptr_->get_current_batch() % STATFREQ == 0)// && eptr_->get_current_epoch()==1)
  {
    string s = nname_ + "_gammap_aft";
    MeanOfLayer((char*)s.c_str(), gamma, gparams_.nInput[0]);

    s = nname_ + "_delgammap_aft";
    MeanOfLayer((char*)s.c_str(), delgamma, gparams_.nOutput);

    s = nname_ + "_igamma";
    MeanOfLayer((char*)s.c_str(), igamma, gparams_.nOutput);

    s = nname_ + "_betap_aft";
    MeanOfLayer((char*)s.c_str(), beta, gparams_.nInput[0]);

    s = nname_ + "_delbetap_aft";
    MeanOfLayer((char*)s.c_str(), delbeta, gparams_.nOutput);

    s = nname_ + "_ibeta";
    MeanOfLayer((char*)s.c_str(), ibeta, gparams_.nOutput);
  }
#endif
}

