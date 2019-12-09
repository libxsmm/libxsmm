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

  in_dtype = tenBotData_[0]->getDataType();
  out_dtype = p->get_data_type();

  tenTopData_->setDataType(out_dtype);

  vector<int> vp = p->get_pads();
  vector<int> ivp = p->get_ipads();
  vector<int> vs = p->get_strides();

  // Get input tensor shape (bottom)
  // Even if there are two inputs for eltwise operation, their shapes are the same. So, pick the first one

  Shape* bs = tenBot_[0]->getShape();
  assert(bs->ndims <= MAX_DIMS);

  shape_setzero(&ts_);
  ts_.ndims = bs->ndims;
  ts_.dims[0] = bs->dims[0];
  ts_.dims[1] = bs->dims[1];
  ts_.dims[2] = bs->dims[2]/vs[0];
  ts_.dims[3] = bs->dims[3]/vs[1];

  tenTop_->setShape(&ts_);

  int telem = ts_.dims[0] * ts_.dims[1] * (ts_.dims[2] + 2*vp[0]) * (ts_.dims[3] + 2*vp[1]);
  long long int tsize;

  if(in_dtype == DT_FLOAT && out_dtype == DT_FLOAT)
    tsize = telem*sizeof(float);
  else if(in_dtype == DT_FLOAT && out_dtype == DT_BF16)
    tsize = telem*sizeof(float) + telem*sizeof(libxsmm_bfloat16);
  else if(in_dtype == DT_BF16 && out_dtype == DT_BF16)
    tsize = telem*sizeof(libxsmm_bfloat16);

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
  tenScaleData_->setDataType(DT_FLOAT);
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
  tenShiftData_->setDataType(DT_FLOAT);
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

  var_ = top_[0] + "_var";
  tenVar_ = new Tensor(var_);
  assert(tenVar_ != NULL);
  tenVar_->setOwner(this);
  tenVar_->setType(BNORMVAR);
  tenVar_->setShape(&sss);
  tenVarData_ = tenVar_->getBuf(DATA);
  tenVarData_->setDataType(DT_FLOAT);
  tenVarData_->setBufferType(DATA);
  tenVarData_->setBufferSize(tsize);

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
        tenBotDiff_[i]->setDataType(in_dtype);
        tenBotDiff_[i]->setBufferType(DIFF);

        int belem = bs->dims[0] * bs->dims[1] * (bs->dims[2] + 2*ivp[0]) * (bs->dims[3] + 2*ivp[1]);
        long long int bsize;

        if(in_dtype == DT_FLOAT && out_dtype == DT_FLOAT)
          bsize = belem*sizeof(float);
        else if(in_dtype == DT_FLOAT && out_dtype == DT_BF16)
          bsize = belem*sizeof(float) + belem*sizeof(libxsmm_bfloat16);
        else if(in_dtype == DT_BF16 && out_dtype == DT_BF16)
          bsize = belem*sizeof(libxsmm_bfloat16);

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

  inserted = e->register_tensor(var_, BNORMVAR, tenVar_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",var_.c_str());

  gparams_.bdims = gparams_.tdims = bs->ndims;
  gparams_.batch_size = bs->dims[0];
  gparams_.node_name = nname_;
  gparams_.nOutput = bs->dims[1];
  gparams_.iHeight = bs->dims[2];
  gparams_.iWidth = bs->dims[3];
  gparams_.oHeight = ts_.dims[2];
  gparams_.oWidth = ts_.dims[3];
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
  gparams_.num_numa_nodes = NUM_NUMA_NODES;
  gparams_.use_global_stats = false;

  lr_mult_ = p->get_lr_mult();
  decay_mult_ = p->get_decay_mult();

  configure(p->get_compute_engine());

  solver_ = e->getSolver();
  eptr_ = e;

  //get global scratch tensor buffer
  tenScratchData_ = e->getScratchBuffer();
#ifdef USE_MLSL
  MLSL::DataType dt = MLSL::DT_FLOAT;
  MLSL::OperationRegInfo *myRegInfo;
  MLSL::Session *s = eptr_->get_session();
  myRegInfo = s->CreateOperationRegInfo(MLSL::OT_BIAS);
  myRegInfo->AddParameterSet(gparams_.nOutput, 1, dt, false);
  myRegInfo->AddParameterSet(gparams_.nOutput, 1, dt, false);
  myRegInfo->AddParameterSet(gparams_.nOutput, 1, dt, false);
  myRegInfo->AddParameterSet(gparams_.nOutput, 1, dt, false);
  myRegInfo->Validate();
  size_t opIdx = s->AddOperation(myRegInfo, e->get_distribution());
  this->op_ = s->GetOperation(opIdx);
  s->DeleteOperationRegInfo(myRegInfo);
  e->get_bias_grad_comms_vec().push_back(op_);
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

void FusedBNormNode::convert_bf16_f32(libxsmm_bfloat16* in, float* out, int len)
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

void FusedBNormNode::fillBuffer(TensorBuf *tBuf, int buftype, long long int bytes)
{
  int ttype = tBuf->getTensor()->getType();
  int dtype = tBuf->getDataType();
  void *ptr = tBuf->getBuffer();

  if(ttype==BNORMSCALE && buftype == DATA)
  {
    if(nname_.find("bn3") == nname_.npos)
      initConstantBuffer(ptr, bytes, "CONSTANT", 1.0f);
    else
      initConstantBuffer(ptr, bytes, "CONSTANT", 0.0f);
  }
  else
    initConstantBuffer(ptr, bytes, "CONSTANT", 0.0f);
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

  if((name.find("30") == name.npos) && (name.find("60") == name.npos) && (name.find("80") == name.npos))
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
    if(format == "binary")
    {
      f = fopen(name.c_str(), "wb");
      if(f != NULL)
      {
        if(name.find("mean") != name.npos || name.find("var") != name.npos)
          ptr = tBuf->getPrivBuffer();
        else
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
  int nImg = gparams_.batch_size;
  int ifm0 = gparams_.nInput[0];
  int ifm1 = gparams_.nInput[1];
  int ofm = gparams_.nOutput;
  int ifh = gparams_.iHeight;
  int ifhp = ifh + 2*gparams_.ipad_h;
  int ifw = gparams_.iWidth;
  int ifwp = ifw + 2*gparams_.ipad_w;
  int ofh = gparams_.oHeight;
  int ofw = gparams_.oWidth;
  int ofhp = ofh + 2*gparams_.pad_h;
  int ofwp = ofw + 2*gparams_.pad_w;
  int oph = gparams_.pad_h;
  int opw = gparams_.pad_w;
  int sh = gparams_.stride_h;
  int sw = gparams_.stride_w;

#ifdef DEBUG
  {
    printf("Executing FP %s\n");
    printf("Inputs: %d x %d x %d\n",gparams_.nInput[0], gparams_.iHeight, gparams_.iWidth);
    printf("Outputs: %d x %d x %d\n",gparams_.nOutput, gparams_.oHeight, gparams_.oWidth);
  }
#endif

  if(eptr_->get_execution_mode() == TRAIN) // || eptr_->get_execution_mode() == VAL)
  {
    impl->set_global_stats(false);
    gparams_.exec_mode = "TRAIN";
  }
  else if(eptr_->get_execution_mode() == TEST || eptr_->get_execution_mode() == VAL)
  {
    impl->set_global_stats(true);
    gparams_.exec_mode = "TEST";
  }

  if(first_fp)
  {
    impl->set_bot_compute_engine(bot_cengine_[0]);
    impl->set_top_compute_engine(top_compute_engine_);
    impl->set_node_name(nname_);
    impl->set_scratch_buffer(tenScratchData_);
#if 0
    if(eptr_->get_execution_mode() == TRAIN) // || eptr_->get_execution_mode() == VAL)
    {
      impl->set_global_stats(false);
      gparams_.exec_mode = "TRAIN";
    }
    else if(eptr_->get_execution_mode() == TEST || eptr_->get_execution_mode() == VAL)
    {
      impl->set_global_stats(true);
      gparams_.exec_mode = "TEST";
    }
#endif

    int size = nImg * ofm * (ofh + 2*oph) * (ofw + 2*opw);

    if((gparams_.in_data_type == DT_FLOAT && gparams_.out_data_type == DT_FLOAT)
        || (gparams_.in_data_type == DT_FLOAT && gparams_.out_data_type == DT_BF16))
    {
      float* ptr = (float*)tenTopData_->getBuffer();

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<size; i++)
        ptr[i] = 0;
    }
    else if(gparams_.in_data_type == DT_BF16 && gparams_.out_data_type == DT_BF16)
    {
      libxsmm_bfloat16* ptr = (libxsmm_bfloat16*)tenTopData_->getBuffer();

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<size; i++)
        ptr[i] = 0;
    }

#ifdef CHECK_BLOWUP_FP32
    cbptr = (float*)_mm_malloc(10240*4, 64);
#endif

    scf_ = eptr_->get_scaling_factor();
    impl->set_scaling_factor(scf_);

    void** meanp = tenMeanData_->getBufferPtr();
    void** varp = tenVarData_->getBufferPtr();

    for(int n=0; n<gparams_.num_numa_nodes; n++)
    {
      float *mean = (float*)meanp[n];
      float *var = (float*)varp[n];
      for(int i=0; i<ifm0; i++)
      {
        mean[i] = 0.0;
        var[i] = 0.0;
      }
    }
    first_fp = false;
  }

  impl->forwardPropagate(tenBotData_, tenScaleData_, tenShiftData_, tenMeanData_, tenVarData_, tenTopData_, 0);

  if(eptr_->get_execution_mode() != TEST && eptr_->get_execution_mode() != VAL)
  {
    scf_ *= gparams_.mmf;
    scf_ += 1.;

    eptr_->set_scaling_factor(scf_);
  }

#ifdef CHECK_BLOWUP_FP32
  if(out_dtype == DT_FLOAT)
  {
    for(int i=0; i<16; i++)
    {
      float v = ((float*)tenTopData_->getBuffer())[i];
      if(isnan(v) || isinf(v))
      {
        printf("Warning! %s layer FP activations are NaN or Inf\n", nname_.c_str());
        exit(-1);
      }
    }
  }
  else if(out_dtype == DT_BF16)
  {
    convert_bf16_f32((libxsmm_bfloat16*)tenTopData_->getBuffer(), cbptr, 10240);
    for(int i=0; i<10240; i++)
    {
      if(isnan(cbptr[i]) || isinf(cbptr[i]))
      {
        printf("Warning! %s layer FP activations are NaN or Inf\n", nname_.c_str());
        exit(-1);
      }
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
    if(in_dtype == DT_FLOAT)
    {
      float *ptr = (float*)tenBotData_[0]->getBuffer();
      string s = nname_ + "_r_Inp";
      MeanOfLayer((char*)s.c_str(), ptr, nImg*ifm0*ifhp*ifwp);

      if(gparams_.nInput.size() > 1)
      {
        ptr = (float*)tenBotData_[1]->getBuffer();
        s = nname_ + "_l_Inp";
        MeanOfLayer((char*)s.c_str(), ptr, nImg*ifm1*ifh*ifw);
      }
    }
    else if(in_dtype == DT_BF16)
    {
      if(stptr == NULL)
      {
        int s = nImg*ofm*ofhp*ofwp;
        int is = nImg*ofm*ifhp*ifwp;
        if(s > is)
          stptr = (float*)libxsmm_aligned_malloc(s*sizeof(float), 2097152);
        else
          stptr = (float*)libxsmm_aligned_malloc(is*sizeof(float), 2097152);
      }

      libxsmm_bfloat16 *ptr = (libxsmm_bfloat16*)tenBotData_[0]->getBuffer();
      string s = nname_ + "_r_Inp";
      convert_bf16_f32(ptr, stptr, nImg*ifm0*ifhp*ifwp);
      MeanOfLayer((char*)s.c_str(), stptr, nImg*ifm0*ifhp*ifwp);

      if(gparams_.nInput.size() > 1)
      {
        ptr = (libxsmm_bfloat16*)tenBotData_[1]->getBuffer();
        convert_bf16_f32(ptr, stptr, nImg*ifm1*ifhp*ifwp);
        s = nname_ + "_l_Inp";
        MeanOfLayer((char*)s.c_str(), stptr, nImg*ifm1*ifh*ifw);
      }
    }

    string s = nname_ + "_gammap0";
    float* gamma = (float*)tenScaleData_->getBuffer();
    MeanOfLayer((char*)s.c_str(), gamma, gparams_.nOutput);
#if 0
    void **g = tenScaleData_->getBufferPtr();
    float *g1 = (float*)g[1] + tenScaleData_->getOffset();
    s = nname_ + "_gammap1";
    MeanOfLayer((char*)s.c_str(), g1, gparams_.nOutput);
#endif

    s = nname_ + "_betap0";
    float* beta = (float*)tenShiftData_->getBuffer();
    MeanOfLayer((char*)s.c_str(), beta, gparams_.nOutput);
#if 0
    void **b = tenShiftData_->getBufferPtr();
    float *b1 = (float*)b[1] + tenShiftData_->getOffset();
    s = nname_ + "_betap1";
    MeanOfLayer((char*)s.c_str(), b1, gparams_.nOutput);
#endif

    if(gparams_.exec_mode == "TEST")
    {
      float meanp[2048], varp[2048], stdevp[2048];

      s = nname_ + "_meanp";
      float *mean = (float*)tenMeanData_->getBuffer();
      for(int i=0; i<gparams_.nOutput; i++)
        meanp[i] = mean[i]/scf_;
      MeanOfLayer((char*)s.c_str(), meanp, gparams_.nOutput);

      s = nname_ + "_varp";
      float *var = (float*)tenVarData_->getBuffer();
      for(int i=0; i<gparams_.nOutput; i++)
        varp[i] = var[i]/scf_;
      MeanOfLayer((char*)s.c_str(), varp, gparams_.nOutput);

      s = nname_ + "_stdevp";
      for(int i=0; i<gparams_.nOutput; i++)
        stdevp[i] = 1/sqrt(varp[i] + 1e-7);
      MeanOfLayer((char*)s.c_str(), stdevp, gparams_.nOutput);
    }

    if(out_dtype == DT_FLOAT)
    {
      void *ptr = tenTopData_->getBuffer();
      string s = nname_ + "_Outp";
      int size = nImg*ofm*(ofh + 2*gparams_.pad_h)*(ofw + 2*gparams_.pad_w);
      MeanOfLayer((char*)s.c_str(), (float*)ptr, size);
    }
    else if(out_dtype == DT_BF16)
    {
      libxsmm_bfloat16 *ptr = (libxsmm_bfloat16*)tenTopData_->getBuffer();
      s = nname_ + "_Outp";
      int size = nImg*ofm*(ofh + 2*gparams_.pad_h)*(ofw + 2*gparams_.pad_w);
      convert_bf16_f32(ptr, stptr, size);
      MeanOfLayer((char*)s.c_str(), stptr, size);
    }
  }
#endif
}

void FusedBNormNode::backPropagate()
{
  int nImg = gparams_.batch_size;
  int ifm0 = gparams_.nInput[0];
  int ifm1 = gparams_.nInput[1];
  int ofm = gparams_.nOutput;
  int ifh = gparams_.iHeight;
  int ifhp = ifh + 2*gparams_.ipad_h;
  int ifw = gparams_.iWidth;
  int ifwp = ifw + 2*gparams_.ipad_w;
  int ofh = gparams_.oHeight;
  int ofw = gparams_.oWidth;
  int oph = gparams_.pad_h;
  int opw = gparams_.pad_w;
  int ofhp = ofh + 2*oph;
  int ofwp = ofw + 2*opw;
  int sh = gparams_.stride_h;
  int sw = gparams_.stride_w;

  tenTopDiff_ = tenTop_->getBuf(DIFF);

#ifndef NDEBUG
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
    if(in_dtype == DT_FLOAT)
    {
      int size = nImg*ifm0*ifhp*ifwp;
      float* gbot0 = (float*)(tenBotDiff_[0]->getBuffer());
      float* gbot1 = gparams_.eltwise ? (float*)(tenBotDiff_[1]->getBuffer()) : NULL;

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<size; i++)
        gbot0[i] = 0.0f;

      if(gbot1)
      {
        size = nImg*gparams_.nInput[1]*ifhp*ifwp;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i=0; i<size; i++)
          gbot1[i] = 0.0f;
      }
    }
    else if(in_dtype == DT_BF16 && out_dtype == DT_BF16)
    {
      libxsmm_bfloat16* gbot0 = (libxsmm_bfloat16*)(tenBotDiff_[0]->getBuffer());
      libxsmm_bfloat16* gbot1 = gparams_.eltwise ? (libxsmm_bfloat16*)(tenBotDiff_[1]->getBuffer()) : NULL;

      int size = nImg*ifm0*ifhp*ifwp;
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<size; i++)
        gbot0[i] = 0;

      if(gbot1)
      {
        size = nImg*gparams_.nInput[1]*ifhp*ifwp;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i=0; i<size; i++)
          gbot1[i] = 0;
      }
    }

    first_bp = false;
  }

  impl->backPropagate(tenTopDiff_, tenScaleDiff_, tenShiftDiff_, tenBotDiff_, 0);

#ifdef CHECK_BLOWUP_FP32
  if(out_dtype == DT_FLOAT)
  {
    for(int i=0; i<10240; i++)
    {
      float v = ((float*)tenBotDiff_[0]->getBuffer())[i];
      if(isnan(v) || isinf(v))
      {
        printf("Warning! %s layer BP activations are NaN or Inf\n", nname_.c_str());
        exit(-1);
      }
    }
  }
  else if(out_dtype == DT_BF16)
  {
    convert_bf16_f32((libxsmm_bfloat16*)tenBotDiff_[0]->getBuffer(), cbptr, 10240);
#ifdef USE_MLSL
    int node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
    int node_id = 0;
#endif
    if(node_id == 0)
    {
      for(int i=0; i<10240; i++)
      {
        if(isnan(cbptr[i]) || isinf(cbptr[i]))
        {
          printf("Warning! %s layer BP activations are NaN or Inf\n", nname_.c_str());
          MeanOfLayer((char*)nname_.c_str(), (libxsmm_bfloat16*)tenBotDiff_[0]->getBuffer(), nImg*ifm0*ifhp*ifwp);
          if(gparams_.eltwise)
            MeanOfLayer((char*)nname_.c_str(), (libxsmm_bfloat16*)tenBotDiff_[1]->getBuffer(), nImg*ifm1*ifhp*ifwp);
#ifdef USE_MLSL
          MPI_Finalize();
#endif
          exit(-1);
        }
      }
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
    if(out_dtype == DT_FLOAT)
    {
      float *ptr = (float*)tenTopDiff_->getBuffer();
      int size = nImg*ofm*ofhp*ofwp;
      string s = nname_ + "_delOutp";
      MeanOfLayer((char*)s.c_str(), ptr, size);
    }
    else if(out_dtype == DT_BF16)
    {
      libxsmm_bfloat16 *ptr = (libxsmm_bfloat16*)tenTopDiff_->getBuffer();
      int size = nImg*ofm*ofhp*ofwp;
      convert_bf16_f32(ptr, stptr, size);
      string s = nname_ + "_delOutp";
      MeanOfLayer((char*)s.c_str(), stptr, size);
    }

    string s = nname_ + "_delgammap0";
    float* delgamma = (float*)tenScaleDiff_->getBuffer();
    MeanOfLayer((char*)s.c_str(), delgamma, gparams_.nOutput);
#if 0
    void **g = tenScaleDiff_->getBufferPtr();
    float *g1 = (float*)g[1] + tenScaleDiff_->getOffset();
    s = nname_ + "_delgammap1";
    MeanOfLayer((char*)s.c_str(), g1, gparams_.nOutput);
#endif

    s = nname_ + "_delbetap0";
    float* delbeta = (float*)tenShiftDiff_->getBuffer();
    MeanOfLayer((char*)s.c_str(), delbeta, gparams_.nOutput);
#if 0
    void **b = tenShiftDiff_->getBufferPtr();
    float *b1 = (float*)b[1] + tenShiftDiff_->getOffset();
    s = nname_ + "_delbetap1";
    MeanOfLayer((char*)s.c_str(), b1, gparams_.nOutput);
#endif

    if(in_dtype == DT_FLOAT)
    {
      float *ptr = (float*)tenBotDiff_[0]->getBuffer();
      string s = nname_ + "_delInp";
      int size = nImg*ifm0*ifhp*ifwp;
      MeanOfLayer((char*)s.c_str(), ptr, size);
    }
    else if(in_dtype == DT_BF16)
    {
      libxsmm_bfloat16 *ptr = (libxsmm_bfloat16*)tenBotDiff_[0]->getBuffer();
      s = nname_ + "_delInp";
      int size = nImg*ifm0*ifhp*ifwp;
      convert_bf16_f32(ptr, stptr, size);
      MeanOfLayer((char*)s.c_str(), stptr, size);
    }
  }
#endif
}

void FusedBNormNode::weightUpdate()
{
#ifdef USE_MLSL
  void* gexp_test = tenMeanData_->getPrivBuffer();
  void* gvar_test = tenVarData_->getPrivBuffer();

  float *gmean = (float*)tenMeanData_->getBuffer();
  float *gvar = (float*)tenVarData_->getBuffer();

  int num_nodes = eptr_->get_num_machines();
  for(int i=0; i<gparams_.nOutput; i++)
  {
    ((float*)gexp_test)[i] = gmean[i]/num_nodes;
    ((float*)gvar_test)[i] = gvar[i]/num_nodes;
  }

  op_->GetParameterSet(0)->StartGradientComm(tenScaleDiff_->getBuffer());
  op_->GetParameterSet(1)->StartGradientComm(tenShiftDiff_->getBuffer());
  op_->GetParameterSet(2)->StartGradientComm(gexp_test);
  op_->GetParameterSet(3)->StartGradientComm(gvar_test);
#endif
}

void FusedBNormNode::solverStep()
{
#if defined(USE_MLSL) || defined(CHECK_BLOWUP_FP32)
  float *delgamma = (float*)tenScaleDiff_->getBuffer();
  float *delbeta = (float*)tenShiftDiff_->getBuffer();
  void* gexp_test = tenMeanData_->getPrivBuffer();
  void* gvar_test = tenVarData_->getPrivBuffer();
#endif

#ifdef USE_MLSL
  void *mptr = op_->GetParameterSet(0)->WaitGradientComm();
  if(mptr != NULL && mptr != delgamma)
    memcpy((void*)delgamma, mptr, gparams_.nOutput*sizeof(float));

  mptr = op_->GetParameterSet(1)->WaitGradientComm();
  if(mptr != NULL && mptr != delbeta)
    memcpy((void*)delbeta, mptr, gparams_.nOutput*sizeof(float));

  mptr = op_->GetParameterSet(2)->WaitGradientComm();
  if(mptr != NULL && mptr != gexp_test)
    memcpy((void*)gexp_test, mptr, gparams_.nOutput*sizeof(float));

  mptr = op_->GetParameterSet(3)->WaitGradientComm();
  if(mptr != NULL && mptr != gvar_test)
    memcpy((void*)gvar_test, mptr, gparams_.nOutput*sizeof(float));
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
}

