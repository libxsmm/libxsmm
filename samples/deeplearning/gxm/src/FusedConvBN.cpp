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
/* Sasikanth Avancha, Dhiraj Kalamkar, Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include <string>
#include "FusedConvBN.hpp"
#include "fillers.hpp"

#ifdef USE_MLSL
#include "mpi.h"
#endif


using namespace std;
using namespace gxm;

FusedConvBNNode::FusedConvBNNode(FusedConvBNParams* p, MLEngine* e): NNNode(p, e)
{
  nname_ = p->get_node_name();
  ntype_ = p->get_node_type();
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

  tenMid_ = new Tensor("mid_"+top_[0]);
  assert(tenMid_ != NULL);
  tenMid_->setOwner(this);
  tenMid_->setType(ACT);
  tenMidData_ = tenMid_->getBuf(DATA);
  tenMidData_->setBufferType(DATA);

  tenBot_.resize(bottom_.size());
  tenBotData_.resize(bottom_.size());

  for(int i=0; i < bottom_.size(); i++)
  {
#ifndef NDEBUG
    printf("bottom%d name %s\n",i,bottom_[i].c_str());
#endif

    if(bottom_[i] == "data")
      tenBot_[i] = e->get_tensor(bottom_[i], INPUT);
    else
      tenBot_[i] = e->get_tensor(bottom_[i], ACT);

    assert(tenBot_[i] != NULL);
    NNNode *pnn = (NNNode*)tenBot_[i]->getOwner();
    setPrevNode(pnn);
    mode_ = pnn->getMode();
    pnn->set_top_compute_engine(p->get_compute_engine());
    bot_cengine_ = pnn->get_bot_compute_engine();

    tenBotData_[i] = tenBot_[i]->getBuf(DATA);
  }

  in_dtype = tenBotData_[0]->getDataType();
  out_dtype = p->get_data_type();
  tenTopData_->setDataType(out_dtype);

  // Get input tensor shape (bottom)
  Shape* bs = tenBot_[0]->getShape();
  assert(bs->ndims <= MAX_DIMS);

  // Create shape of output tensor (top)
  vector<int> vd = p->get_kernel_dims();
  vector<int> mvp = p->get_mid_pads();
  vector<int> ovp = p->get_top_pads();
  vector<int> ivp = p->get_bot_pads();
  vector<int> vcs = p->get_c_strides();
  vector<int> vbns = p->get_bn_strides();

  shape_setzero(&ms_);
  ms_.ndims = bs->ndims; // Number of dimensions
  ms_.dims[0] = bs->dims[0]; // Minibatch size
  ms_.dims[1] = p->get_output(); // Num output feature maps
  ms_.dims[2] = (bs->dims[2] - vd[0] + 2*ivp[0])/vcs[0] + 1; // Height
  ms_.dims[3] = (bs->dims[3] - vd[1] + 2*ivp[1])/vcs[1] + 1; // Width

  tenMid_->setShape(&ms_);

  shape_setzero(&ts_);
  ts_.ndims = bs->ndims; // Number of dimensions
  ts_.dims[0] = bs->dims[0]; // Minibatch size
  ts_.dims[1] = p->get_output(); // Num output feature maps
  ts_.dims[2] = ms_.dims[2]/vbns[0]; // Height
  ts_.dims[3] = ms_.dims[3]/vbns[1]; // Width

  tenTop_->setShape(&ts_);

  long long int tsize;
  int convelem = ms_.dims[0] * ms_.dims[1] * (ms_.dims[2] + 2*mvp[0]) * (ms_.dims[3] + 2*mvp[1]);
  int bnelem = ts_.dims[0] * ts_.dims[1] * (ts_.dims[2] + 2*ovp[0]) * (ts_.dims[3] + 2*ovp[1]);
  int telem = convelem + bnelem;

  if(out_dtype == DT_FLOAT)
    tsize = telem*sizeof(float);
  else if(out_dtype = DT_BF16)
    tsize = telem*sizeof(libxsmm_bfloat16);

  tenTopData_->setBufferSize(tsize);

  // Create FP weight tensor
  weight_ = top_[0] + "_wt";
  tenWeight_ = new Tensor(weight_);
  assert(tenWeight_ != NULL);
  tenWeight_->setOwner(this);
  tenWeight_->setType(CONVWEIGHT);

  shape_setzero(&ws_);

  ws_.ndims = ts_.ndims;      // Number of dimesions
  ws_.dims[0] = ms_.dims[1];  // Num output feature maps (from mid tensor)
  ws_.dims[1] = bs->dims[1];  // Num input feature maps (from bottom tensor)
  ws_.dims[2] = vd[0];        // Kernel height
  ws_.dims[3] = vd[1]; // Kernel width

  tenWeight_->setShape(&ws_);
  tenWeight_->setBufDataType(DATA, DT_FLOAT);
  tenWeightData_ = tenWeight_->getBuf(DATA);
  tenWeightData_->setBufferType(DATA);

  int welem = 1;
  long long int wsize;
  for(int i=0; i<ws_.ndims; i++)
    welem = welem*ws_.dims[i];

  // size of master weights -- FP32.
  wsize = welem*sizeof(float);

  gparams_.num_numa_nodes = NUM_NUMA_NODES;
  tenWeightData_->setBufferSize(wsize);

  wfiller_type_ = p->get_weight_filler_type();
  variance_norm_ = p->get_variance_norm();
  std_ = p->get_std();

  lr_mult_ = p->get_lr_mult();
  decay_mult_ = p->get_decay_mult();

  Shape sss;
  shape_setzero(&sss);
  sss.ndims = 1;
  sss.dims[0] = ts_.dims[1];

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

  if(!e->is_inference_only()) {
    tenBotDiff_.resize(bottom_.size());

    if(bp_flag_)
    {
      tenBotDiff_[0] = tenBot_[0]->addBuf(); // DIFF type and index
      tenBotDiff_[0]->setDataType(in_dtype);
      tenBotDiff_[0]->setBufferType(DIFF);

      // Set the size of the input-gradient buffer
      Shape *bs = tenBot_[0]->getShape();
      int botelem = bs->dims[0] * bs->dims[1] * (bs->dims[2] + 2*ivp[0]) * (bs->dims[3] + 2*ivp[1]);
      if(in_dtype == DT_FLOAT)
        tenBotDiff_[0]->setBufferSize((botelem + convelem)*sizeof(float));
      else if(in_dtype == DT_BF16)
        tenBotDiff_[0]->setBufferSize((botelem + convelem)*sizeof(libxsmm_bfloat16));
    }
    tenMidDiff_ = tenMid_->addBuf(); // DIFF type and index
    tenMidDiff_->setDataType(in_dtype);
    tenMidDiff_->setBufferType(DIFF);

    if(has_weights_)
    {
      for(int i=1; i<bottom_.size(); i++)
      {
        tenBotDiff_[i] = tenBot_[i]->addBuf(); // DIFF type and index
        tenBotDiff_[i]->setDataType(in_dtype);
        tenBotDiff_[i]->setBufferType(DIFF);

        // Set the size of the input-gradient buffer
        Shape *bs = tenBot_[i]->getShape();
        int botelem = bs->dims[0] * bs->dims[1] * (bs->dims[2] + 2*ivp[0]) * (bs->dims[3] + 2*ivp[1]);
        if(in_dtype == DT_FLOAT)
          tenBotDiff_[i]->setBufferSize((botelem + convelem)*sizeof(float));
        else if(in_dtype == DT_BF16)
          tenBotDiff_[i]->setBufferSize((botelem + convelem)*sizeof(libxsmm_bfloat16));
      }
      tenWeightDiff_ = tenWeight_->addBuf(); // DIFF type and index
      tenWeightDiff_->setBufferType(DIFF);

      tenWeightInc_ = tenWeight_->addBuf(); // SHARED type and index
      tenWeightInc_->setDataType(DT_FLOAT);
      tenWeightInc_->setBufferType(HISTORY);
      tenWeightInc_->setBufferSize(welem*sizeof(float));

      // Set the size of the weight-gradient buffer and the weight-increment buffer
      if(in_dtype == DT_FLOAT)
      {
        tenWeightDiff_->setDataType(DT_FLOAT);
        tenWeightDiff_->setBufferSize(welem*sizeof(float));
      }
      else if(in_dtype == DT_BF16)
      {
        tenWeightDiff_->setDataType(DT_BF16);
        tenWeightDiff_->setBufferSize(welem*sizeof(libxsmm_bfloat16));
      }

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
  else {
    tenMidDiff_ = NULL;
    tenWeightDiff_ = NULL;
    tenWeightInc_ = NULL;
    tenScaleDiff_ = NULL;
    tenShiftDiff_ = NULL;
    tenScaleInc_ = NULL;
    tenShiftInc_ = NULL;
  }

  // Register output tensor in tensor map
  bool inserted = e->register_tensor(top_[0], ACT, tenTop_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",top_[0].c_str());

  string m = "mid_"+top_[0];
  inserted = e->register_tensor(m, ACT, tenMid_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",m.c_str());

  // Register weight tensor in weight tensor map
  inserted = e->register_tensor(weight_, CONVWEIGHT, tenWeight_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",weight_.c_str());

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

  // Setup parameter structure for convolution computation in library
  gparams_.bdims = bs->ndims;
  gparams_.tdims = ts_.ndims;
  gparams_.mdims = ms_.ndims;
  gparams_.wdims = ws_.ndims;

  gparams_.node_name = nname_;
  gparams_.node_type = ntype_;
  gparams_.nInput.resize(bottom_.size());
  if(bottom_.size() > 1)
    gparams_.nInput.resize(bottom_.size());
  gparams_.nInput[0] = bs->dims[1];
  if(bottom_.size() > 1)
    gparams_.nInput[1] = tenBot_[1]->getShape()->dims[1];
  gparams_.nOutput = ts_.dims[1];
  gparams_.batch_size = bs->dims[0];
  gparams_.iHeight = bs->dims[2];
  gparams_.iWidth = bs->dims[3];
  gparams_.mHeight = ms_.dims[2];
  gparams_.mWidth = ms_.dims[3];
  gparams_.oHeight = ts_.dims[2];
  gparams_.oWidth = ts_.dims[3];
  gparams_.ipad_h = ivp[0];
  gparams_.ipad_w = ivp[1];
  gparams_.mpad_h = mvp[0];
  gparams_.mpad_w = mvp[1];
  gparams_.opad_h = ovp[0];
  gparams_.opad_w = ovp[1];
  gparams_.physical_padding = p->get_physical_padding();

  gparams_.group = p->get_group();
  gparams_.c_stride_h = vcs[0];
  gparams_.c_stride_w = vcs[1];
  gparams_.bn_stride_h = vbns[0];
  gparams_.bn_stride_w = vbns[1];
  gparams_.kh = ws_.dims[2];
  gparams_.kw = ws_.dims[3];

  gparams_.relu_fwd = p->get_relu_fwd();
  gparams_.relu_bwd = p->get_relu_bwd();

  gparams_.mmf = p->get_mmf();
  gparams_.eps = p->get_eps();
  gparams_.use_global_stats = (e->get_execution_mode() == TEST);
  gparams_.eltwise = p->get_eltwise();
  gparams_.bprop = bp_flag_;

  gparams_.in_data_type = in_dtype;
  gparams_.out_data_type = out_dtype;

  NNNode *pnn = (NNNode*)tenBot_[0]->getOwner();
  if(pnn->getNodeType() == "FusedConvBN")
  {
    gparams_.prev_bn_train_handle_ptr = dynamic_cast<FusedConvBNNode*>(pnn)->getBNTrainHandlePtr();
    gparams_.prev_bn_test_handle_ptr = dynamic_cast<FusedConvBNNode*>(pnn)->getBNTestHandlePtr();
  }
  else
  {
    gparams_.prev_bn_train_handle_ptr = NULL;
    gparams_.prev_bn_test_handle_ptr = NULL;
  }

  gparams_.algType = p->get_algo_type();
  gparams_.num_threads = e->get_num_threads();

  // get solver
  solver_ = e->getSolver();

  //get global scratch tensor buffer
  tenScratchData_ = e->getScratchBuffer();

  // get engine
  eptr_ = e;

#ifdef USE_MLSL
  MLSL::DataType dt = MLSL::DT_FLOAT;
  MLSL::OperationRegInfo *myRegInfo;
  MLSL::Session *s = eptr_->get_session();
  myRegInfo = s->CreateOperationRegInfo(MLSL::OT_CC);
  myRegInfo->SetName(nname_.c_str());
  myRegInfo->AddParameterSet(gparams_.nInput[0]*gparams_.nOutput/gparams_.group, gparams_.kw*gparams_.kh, dt, false);
  myRegInfo->AddParameterSet(gparams_.nOutput, 1, dt, false);
  myRegInfo->AddParameterSet(gparams_.nOutput, 1, dt, false);
  myRegInfo->AddParameterSet(gparams_.nOutput, 1, dt, false);
  myRegInfo->AddParameterSet(gparams_.nOutput, 1, dt, false);

  myRegInfo->Validate();
  size_t opIdx = s->AddOperation(myRegInfo, e->get_distribution());
  this->op_ = s->GetOperation(opIdx);
  s->DeleteOperationRegInfo(myRegInfo);
  e->get_combo_grad_comms_vec().push_back(op_);
#endif

  configure(p->get_compute_engine());
}

void FusedConvBNNode::configure(int engine)
{
  switch(engine)
  {
    case XSMM:
      impl = new FusedConvBNXSMM(&gparams_, engine);
      break;
  }
}

void** FusedConvBNNode::getBNTrainHandlePtr()
{
  return gparams_.my_bn_train_handle_ptr;
}

void** FusedConvBNNode::getBNTestHandlePtr()
{
  return gparams_.my_bn_test_handle_ptr;
}

void FusedConvBNNode::fillWeightBuffers(TensorBuf* tBuf, int buftype, long long int size)
{
  int dtype = DT_FLOAT;
  void *ptr = tBuf->getBuffer();

#ifdef USE_MLSL
  unsigned int node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
  unsigned int node_id = 0;
#endif

  int ic = gparams_.nInput[0];
  int oc = gparams_.nOutput;
  int kh = gparams_.kh;
  int kw = gparams_.kw;
  int g = gparams_.group;
  int fanin = (ic * kh * kw)/g;
  int fanout = (oc * kh * kw)/g;
  int welem = ic * oc * kh * kw;

  if(buftype == DATA)
  {
    initBuffer(ptr, dtype, variance_norm_, fanin, fanout, welem*sizeof(float), wfiller_type_, (unsigned int)(node_id+PRIME_SEED), std_);

#ifdef USE_MLSL
    if(dtype == DT_FLOAT)
      MPI_Bcast(ptr, welem, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
  }
  else if(buftype == HISTORY || buftype == DIFF)
    memset(ptr, 0, size);
}

void FusedConvBNNode::fillWeightMultipliers(float* lr, float* decay, long long int size)
{
  for(int i=0; i < size; i++)
  {
    lr[i] = lr_mult_[0];
    decay[i] = decay_mult_[0];
  }
}

void FusedConvBNNode::fillBiasMultipliers(float* lr, float* decay, long long int size)
{
  for(int i=0; i < size; i++)
  {
    lr[i] = lr_mult_[1];
    decay[i] = decay_mult_[1];
  }
}

void FusedConvBNNode::fillBuffer(TensorBuf* tBuf, int buftype, long long int size)
{
  int ttype = tBuf->getTensor()->getType();
  int dtype = DT_FLOAT;
  void *ptr = tBuf->getBuffer();

  if(ttype==BNORMSCALE && buftype == DATA)
  {
    if(nname_.find("bn3") == nname_.npos)
      initConstantBuffer(ptr, dtype, size, "CONSTANT", 1.0f);
    else
      initConstantBuffer(ptr, dtype, size, "CONSTANT", 0.0f);
  }
  else
      initConstantBuffer(ptr, dtype, size, "CONSTANT", 0.0f);
}

void FusedConvBNNode::Checkpoint(TensorBuf *tBuf, string name, string format)
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
    if(format.compare("binary") == 0)
    {
      f = fopen(name.c_str(), "wb");
      if(f != NULL)
      {
        if(name.find("wt") != name.npos)
        {
          ptr = _mm_malloc(bytes, 64);
          assert(ptr != NULL);
          impl->dumpBuffer(tBuf, ptr);
        }
        else if(name.find("mean") != name.npos || name.find("var") != name.npos)
          ptr = tBuf->getPrivBuffer();
        else
          ptr = tBuf->getBuffer();

        size_t b = fwrite(ptr, 1, bytes, f);
        assert((long long int)b == bytes);

        if(name.find("wt") != name.npos)
          _mm_free(ptr);
      }
      else
        printf("Warning: could not checkpoint to file %s\n",name.c_str());
    }
    else
    {
      f = fopen(name.c_str(), "w");
      if(f != NULL)
      {
        if(name.find("wt") != name.npos)
        {
          ptr = _mm_malloc(bytes, 64);
          assert(ptr != NULL);
          impl->dumpBuffer(tBuf, ptr);
        }
        else
          ptr = tBuf->getBuffer();

        for(int i=0; i<bytes/sizeof(float); i++)
          fprintf(f, "%f\n", *((float*)ptr + i));

        if(name.find("wt") != name.npos)
          _mm_free(ptr);
      }
      else
        printf("Warning: could not checkpoint to file %s\n",name.c_str());
    }
    if(f != NULL)
    {
      fflush(f);
      fclose(f);
    }
  }
}

void FusedConvBNNode::convert_f32_bf16(float* in, libxsmm_bfloat16* out, int len)
{
  int i;

#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
  for ( i = 0; i < len; i+=16 ) {
    __m512  vfp32  = gxm_fp32_to_bfp16_rne_adjustment_avx512f( _mm512_loadu_ps( in+i ) );
    __m256i vbfp16 = gxm_fp32_to_bfp16_truncate_avx512f( vfp32 );
    _mm256_storeu_si256( (__m256i*)(out+i), vbfp16 );
  }
}

void FusedConvBNNode::convert_bf16_f32(libxsmm_bfloat16* in, float* out, int len)
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

void FusedConvBNNode::forwardPropagate()
{
  int nImg = gparams_.batch_size;
  int ifm0 = gparams_.nInput[0];
  int ifm1 = gparams_.eltwise ? gparams_.nInput[1] : 0;
  int ofm = gparams_.nOutput;
  int ifh = gparams_.iHeight;
  int ifhp = ifh + 2*gparams_.ipad_h;
  int ifw = gparams_.iWidth;
  int ifwp = ifw + 2*gparams_.ipad_w;
  int mfh = gparams_.mHeight;
  int mfw = gparams_.mWidth;
  int mfhp = mfh + 2*gparams_.mpad_h;
  int mfwp = mfw + 2*gparams_.mpad_w;
  int ofh = gparams_.oHeight;
  int ofw = gparams_.oWidth;
  int oph = gparams_.opad_h;
  int opw = gparams_.opad_w;
  int ofhp = ofh + 2*oph;
  int ofwp = ofw + 2*opw;
  int bnsh = gparams_.bn_stride_h;
  int bnsw = gparams_.bn_stride_w;
  int kh = gparams_.kh;
  int kw = gparams_.kw;

#ifndef NDEBUG
  // printf("Executing FP %s: input %p, weights %p, output %p\n",NNNode::nname_.c_str(), bot, wt, top);
  printf("Executing FP %s\n",NNNode::nname_.c_str());
  printf("Inputs: %d x %d x %d\n",ifm0, ifh, ifw);
  printf("Outputs: %d x %d x %d\n",ofm, ofh, ofw);
  printf("Weights: %d x %d x %d x %d\n", ifm0, ofm, kh, kw);
  printf("Bias: %d\n", ofm);
#endif

  if(first_fp)
  {
    impl->set_top_compute_engine(top_compute_engine_);
    impl->set_bot_compute_engine(bot_cengine_);
    impl->set_node_name(nname_);
    impl->set_scratch_buffer(tenScratchData_);

    if(eptr_->get_execution_mode() == TRAIN || eptr_->get_execution_mode() == VAL)
    {
      impl->set_global_stats(false);
      gparams_.exec_mode = "TRAIN";
    }
    else if(eptr_->get_execution_mode() == TEST)
      impl->set_global_stats(true);

    tenMidData_->setBuffer(tenTopData_->getBuffer());

    if(out_dtype == DT_FLOAT)
    {
      float* ptr = (float*)tenMidData_->getBuffer();
      int size = nImg * ofm * mfhp * mfwp;
      tenMidData_->setBufferSize(size*sizeof(float));
      tenTopData_->setBuffer(tenTopData_->getBuffer() + size*sizeof(float));
      tenTopData_->setBufferSize(tenTopData_->getBufferSize() - size*sizeof(float));

      // NUMA initialize Conv output
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<size; i++)
        ptr[i] = 0;

      // NUMA initialize BN output
      size = nImg * ofm * (ofh/bnsh +2*oph) * (ofw/bnsw + 2*opw);
      ptr = (float*)tenTopData_->getBuffer();

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<size; i++)
        ptr[i] = 0;
    }
    else if(out_dtype == DT_BF16)
    {
      libxsmm_bfloat16* ptr = (libxsmm_bfloat16*)tenMidData_->getBuffer();
      int size = nImg * ofm * mfhp * mfwp;
      tenMidData_->setBufferSize(size*sizeof(libxsmm_bfloat16));
      tenTopData_->setBuffer(tenTopData_->getBuffer() + size*sizeof(libxsmm_bfloat16));
      tenTopData_->setBufferSize(tenTopData_->getBufferSize() - size*sizeof(libxsmm_bfloat16));

      // NUMA initialize Conv output
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<size; i++)
        ptr[i] = 0;

      // NUMA initialize BN output
      ptr = (libxsmm_bfloat16*)tenTopData_->getBuffer();
      size = nImg * ofm * (ofh/bnsh + 2*oph) * (ofw/bnsw + 2*opw);

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<size; i++)
        ptr[i] = 0;
    }

    cbptr = (float*)_mm_malloc(10240*4, 64);
    scf_ = eptr_->get_scaling_factor();
    impl->set_scaling_factor(scf_);

    first_fp = false;
  }

  impl->forwardPropagate(tenBotData_, tenWeightData_, tenWeightInc_, tenMidData_, tenScaleData_, tenShiftData_, tenMeanData_, tenVarData_, tenTopData_, 0);

  if(eptr_->get_execution_mode() != TEST && eptr_->get_execution_mode() != VAL)
  {
    scf_ *= gparams_.mmf;
    scf_ += 1.;

    eptr_->set_scaling_factor(scf_);
  }

#ifdef CHECK_BLOWUP_FP32
  if(out_dtype == DT_FLOAT)
  {
    for(int i=0; i<10240; i++)
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
    convert_bf16_f32((libxsmm_bfloat16*)tenMidData_->getBuffer(), cbptr, 10240);
    for(int i=0; i<10240; i++)
    {
      if(isnan(cbptr[i]) || isinf(cbptr[i]))
      {
        printf("Warning! %s layer FP mid activations are NaN or Inf\n", nname_.c_str());
        libxsmm_bfloat16 *ptr = (libxsmm_bfloat16*)tenMidData_->getBuffer();
        printf("cbptr[%d] = %d, cbptr[%d] = %f\n",i,ptr[i],i,cbptr[i]);
        exit(-1);
      }
    }
    convert_bf16_f32((libxsmm_bfloat16*)tenTopData_->getBuffer(), cbptr, 10240);
    for(int i=0; i<10240; i++)
    {
      if(isnan(cbptr[i]) || isinf(cbptr[i]))
      {
        printf("Warning! %s layer FP activations are NaN or Inf\n", nname_.c_str());
        libxsmm_bfloat16 *ptr = (libxsmm_bfloat16*)tenTopData_->getBuffer();
        printf("cbptr[%d] = %d, cbptr[%d] = %f\n",i,ptr[i],i,cbptr[i]);
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
  if(node_id == 0)
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
        MeanOfLayer((char*)s.c_str(), ptr, nImg*ifm1*ifhp*ifwp);
      }

      ptr = (float*)tenMidData_->getBuffer();
      s = nname_ + "_mid";
      MeanOfLayer((char*)s.c_str(), ptr, nImg*ofm*mfhp*mfwp);
    }
    else if(in_dtype == DT_BF16)
    {
      if(stptr == NULL)
      {
        int s = nImg*ofm*ofhp*ofwp;
        int ms = nImg*ofm*mfhp*mfwp;
        int is = nImg*ifm0*ifhp*ifwp;
        int is1=0;
        if(gparams_.nInput.size() > 1)
          is1 = nImg*ifm1*ifhp*ifwp;

        int size = s > ms ? s : ms;
        size = size > is ? size : is;
        size = size > is1 ? size : is1;

        stptr = (float*)libxsmm_aligned_malloc(size*sizeof(float), 2097152);
      }

      libxsmm_bfloat16 *ptr;
      if(tenBotData_[0]->getLPBuffer() != NULL)
        ptr = (libxsmm_bfloat16*)tenBotData_[0]->getLPBuffer();
      else
        ptr = (libxsmm_bfloat16*)tenBotData_[0]->getBuffer();

      string s = nname_ + "_r_Inp";
      convert_bf16_f32(ptr, stptr, nImg*ifm0*ifhp*ifwp);
      MeanOfLayer((char*)s.c_str(), stptr, nImg*ifm0*ifhp*ifwp);

      if(gparams_.nInput.size() > 1)
      {
        if(tenBotData_[1]->getLPBuffer() != NULL)
          ptr = (libxsmm_bfloat16*)tenBotData_[1]->getLPBuffer();
        else
          ptr = (libxsmm_bfloat16*)tenBotData_[1]->getBuffer();

        convert_bf16_f32(ptr, stptr, nImg*ifm1*ifhp*ifwp);
        s = nname_ + "_l_Inp";
        MeanOfLayer((char*)s.c_str(), stptr, nImg*ifm1*ifhp*ifwp);
      }

      ptr = (libxsmm_bfloat16*)tenMidData_->getBuffer();
      convert_bf16_f32(ptr, stptr, nImg*ofm*mfhp*mfwp);
      s = nname_ + "_mid";
      MeanOfLayer((char*)s.c_str(), stptr, nImg*ofm*mfhp*mfwp);
    }

    string s = nname_ + "_wt";
    float* wt = (float*)tenWeightData_->getBuffer();
    MeanOfLayer((char*)s.c_str(), wt, ifm0*ofm*kh*kw);

    s = nname_ + "_gammap";
    float* gamma = (float*)tenScaleData_->getBuffer();
    MeanOfLayer((char*)s.c_str(), gamma, gparams_.nOutput);

    s = nname_ + "_betap";
    float* beta = (float*)tenShiftData_->getBuffer();
    MeanOfLayer((char*)s.c_str(), beta, gparams_.nOutput);

    if(out_dtype == DT_FLOAT)
    {
      float *ptr = (float*)tenTopData_->getBuffer();
      string s = nname_ + "_Outp";
      int size = nImg*ofm*(ofh/bnsh + 2*oph)*(ofw/bnsw + 2*opw);
      MeanOfLayer((char*)s.c_str(), ptr, size);
    }
    else if(out_dtype == DT_BF16)
    {
      libxsmm_bfloat16 *ptr = (libxsmm_bfloat16*)tenTopData_->getBuffer();
      s = nname_ + "_Outp";
      int size = nImg*ofm*(ofh/bnsh + 2*oph)*(ofw/bnsw + 2*opw);
      convert_bf16_f32(ptr, stptr, size);
      MeanOfLayer((char*)s.c_str(), stptr, size);
    }
  }
#endif
}

void FusedConvBNNode::backPropagate()
{

  int nImg = gparams_.batch_size;
  int ifm0 = gparams_.nInput[0];
  int ofm = gparams_.nOutput;
  int ifh = gparams_.iHeight;
  int ifhp = ifh + 2*gparams_.ipad_h;
  int ifw = gparams_.iWidth;
  int ifwp = ifw + 2*gparams_.ipad_w;
  int mfh = gparams_.mHeight;
  int mfw = gparams_.mWidth;
  int mfhp = mfh + 2*gparams_.mpad_h;
  int mfwp = mfw + 2*gparams_.mpad_w;
  int kh = gparams_.kh;
  int kw = gparams_.kw;

#ifdef DEBUG
  printf("Executing BP %s\n",NNNode::nname_.c_str());
  printf("Grad Outputs: %d x %d x %d\n", ofm, ofh, ofw);
  printf("Grad Inputs: %d x %d x %d\n", ifm, ifh, ifw);
  printf("Weights: %d x %d x %d x %d\n", ofm, ifm, kh, kw);
#endif

  tenTopDiff_ = tenTop_->getBuf(DIFF);

  if(first_bp)
  {
    int bsize0 = nImg*ifm0*ifhp*ifwp;

    if(in_dtype == DT_FLOAT)
    {
      float* ptr = (float*)tenBotDiff_[0]->getBuffer();
      tenBotDiff_[0]->setBufferSize(bsize0*sizeof(float));

      // NUMA initialize Conv delinp
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<bsize0; i++)
        ptr[i] = 0;

    }
    else if(in_dtype == DT_BF16)
    {
      libxsmm_bfloat16* ptr = (libxsmm_bfloat16*)tenBotDiff_[0]->getBuffer();
      tenBotDiff_[0]->setBufferSize(bsize0*sizeof(libxsmm_bfloat16));

      // NUMA initialize Conv delinp
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<bsize0; i++)
        ptr[i] = 0;

    }
    first_bp = false;
  }

  impl->backPropagate(tenMidDiff_, tenWeightData_, tenBotDiff_[0], 0);

#ifdef CHECK_BLOWUP_FP32
  float* cbptr = (float*)tenTopDiff_->getBuffer();
  for(int i=0; i<10240; i++)
  {
    if(isnan(cbptr[i]) || isinf(cbptr[i]))
    {
      printf("Warning! %s layer BP activations are NaN or Inf\n", nname_.c_str());
      exit(-1);
    }
  }
#endif

#ifdef GETSTATS
  float *ptr, *pptr, *p, *bias;
#ifdef USE_MLSL
  unsigned int node_id_ = MLSL::Environment::GetEnv().GetProcessIdx();
#else
  unsigned int node_id_ = 0;
#endif
  if(node_id_ == 0)
  {

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

void FusedConvBNNode::weightUpdate()
{
  int nImg = gparams_.batch_size;
  int ifm0 = gparams_.nInput[0];
  int ifm1 = gparams_.eltwise ? gparams_.nInput[1] : 0;
  int ofm = gparams_.nOutput;
  int ifh = gparams_.iHeight;
  int ifw = gparams_.iWidth;
  int mfh = gparams_.mHeight;
  int mfw = gparams_.mWidth;
  int mfhp = mfh + 2*gparams_.mpad_h;
  int mfwp = mfw + 2*gparams_.mpad_w;
  int ifhp = ifh + 2*gparams_.ipad_h;
  int ifwp = ifw + 2*gparams_.ipad_w;
  int ofh = gparams_.oHeight;
  int ofw = gparams_.oWidth;
  int ofhp = ofh + 2*gparams_.opad_h;
  int ofwp = ofw + 2*gparams_.opad_w;
  int kh = gparams_.kh;
  int kw = gparams_.kw;

#ifdef DEBUG
  // printf("Executing WU %s: grad_output %p, grad_weights %p, input %p\n",NNNode::nname_.c_str(), gtop, gwt, bot);
  printf("Executing WU %s\n",NNNode::nname_.c_str());
  printf("Grad Outputs: %d x %d x %d\n",ofm, ofh,ofw);
  printf("Inputs: %d x %d x %d\n",ifm0, ifh, ifw);
  printf("del-Weights: %d x %d x %d x %d\n", ofm, ifm0, kh, kw);
  printf("del-Biases: %d\n", ofm);
#endif

#ifdef GETSTATS
#ifdef USE_MLSL
  int node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
  int node_id = 0;
#endif
  if(node_id == 0)
  {
    if(in_dtype == DT_FLOAT)
    {
      string s = nname_ + "_delWt_Bef";
      float *ptr = (float*)tenWeightDiff_->getBuffer();
      MeanOfLayer((char*)s.c_str(), ptr, ifm0*ofm*kh*kw);
    }
    else if(in_dtype == DT_BF16)
    {
      string s = nname_ + "_delWt_Bef";
      libxsmm_bfloat16 *ptr = (libxsmm_bfloat16*)tenWeightDiff_->getBuffer();
      memset(stptr, 0, ifm0*ofm*kh*kw);
      convert_bf16_f32(ptr, stptr, ifm0*ofm*kh*kw);
      MeanOfLayer((char*)s.c_str(), stptr, ifm0*ofm*kh*kw);
    }
  }
#endif

  int bsize0 = nImg*ifm0*ifhp*ifwp;
  int bsize1 = nImg*ifm1*ifhp*ifwp;
  int msize = nImg*ofm*mfhp*mfwp;
  int tsize = nImg*ofm*ofhp*ofwp;

  tenTopDiff_ = tenTop_->getBuf(DIFF);

  if(first_upd)
  {
    if(bp_flag_)
      tenMidDiff_->setBuffer(tenBotDiff_[0]->getBuffer() + bsize0*sizeof(float));
    else
      tenMidDiff_->setBuffer(tenTopDiff_->getBuffer() + tsize*sizeof(float));

    tenMidDiff_->setBufferSize(msize*sizeof(float));
    if(gparams_.eltwise)
      tenBotDiff_[1]->setBufferSize(bsize1*sizeof(float));

    if(in_dtype == DT_FLOAT)
    {
      float *ptr = (float*)tenMidDiff_->getBuffer();

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<msize; i++)
        ptr[i] = 0;

      ptr = gparams_.eltwise ? (float*)tenBotDiff_[1]->getBuffer() : NULL;
      if(ptr)
      {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i=0; i<bsize1; i++)
          ptr[i] = 0;
      }
    }
    else if(in_dtype == DT_BF16)
    {
      tenMidDiff_->setBuffer(tenBotDiff_[0]->getBuffer() + bsize0*sizeof(libxsmm_bfloat16));
      tenMidDiff_->setBufferSize(msize*sizeof(libxsmm_bfloat16));
      libxsmm_bfloat16* ptr = (libxsmm_bfloat16*)tenMidDiff_->getBuffer();

      // NUMA initialize = Conv delmidp
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<msize; i++)
        ptr[i] = 0;

      if(gparams_.eltwise)
        tenBotDiff_[1]->setBufferSize(bsize1*sizeof(libxsmm_bfloat16));

      ptr = gparams_.eltwise ? (libxsmm_bfloat16*)tenBotDiff_[1]->getBuffer() : NULL;
      if(ptr)
      {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int i=0; i<bsize1; i++)
          ptr[i] = 0;
      }
    }
    first_upd = false;
  }

  impl->weightUpdate(tenBotData_[0], tenTopDiff_, tenMidDiff_, tenBotDiff_[1], tenWeightDiff_, tenScaleDiff_, tenShiftDiff_, 0);

#ifdef CHECK_BLOWUP_FP32
  if(out_dtype == DT_FLOAT)
  {
    for(int i=0; i<16; i++)
    {
      float v = ((float*)tenWeightDiff_->getBuffer())[i];
      if(isnan(v) || isinf(v))
      {
        printf("Warning! %s layer BP activations are NaN or Inf\n", nname_.c_str());
        exit(-1);
      }
    }
  }
  else if(out_dtype == DT_BF16)
  {
    convert_bf16_f32((libxsmm_bfloat16*)tenWeightDiff_->getBuffer(), cbptr, 16);
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

  void* gexp[NUM_NUMA_NODES];
  void* gvar[NUM_NUMA_NODES];
  void* gexp_test = tenMeanData_->getPrivBuffer();
  void* gvar_test = tenVarData_->getPrivBuffer();

  void **mptrptr = tenMeanData_->getBufferPtr();
  void **vptrptr = tenVarData_->getBufferPtr();
  int offset = tenMeanData_->getOffset();

  for(int n=0; n<NUM_NUMA_NODES; n++)
    gexp[n] = mptrptr[n] + offset*sizeof(float);

  offset = tenVarData_->getOffset();
  for(int n=0; n<NUM_NUMA_NODES; n++)
    gvar[n] = vptrptr[n] + offset*sizeof(float);

#ifdef USE_MLSL
  void *mptr = tenWeightDiff_->getBuffer();

  if(in_dtype == DT_BF16)
  {
    if(dwptr == NULL)
    {
      int wsize = ALIGN_SIZE(ifm0*ofm*kh*kw*sizeof(float), 2097152);
      dwptr = (float*)MLSL::Environment::GetEnv().Alloc(wsize, 2097152);
    }
    convert_bf16_f32((libxsmm_bfloat16*)mptr, dwptr, ifm0*ofm*kh*kw);
    op_->GetParameterSet(0)->StartGradientComm(dwptr);
  }
  else if(in_dtype == DT_FLOAT)
    op_->GetParameterSet(0)->StartGradientComm(mptr);

  op_->GetParameterSet(1)->StartGradientComm(tenScaleDiff_->getBuffer());
  op_->GetParameterSet(2)->StartGradientComm(tenShiftDiff_->getBuffer());

  int num_nodes = eptr_->get_num_machines();
  for(int i=0; i<ofm; i++)
  {
    float mtmp = 0.0;
    float vtmp = 0.0;

    for(int n=0; n<NUM_NUMA_NODES; n++)
    {
      mtmp += ((float*)gexp[n])[i];
      vtmp += ((float*)gvar[n])[i];
    }

    mtmp = mtmp/NUM_NUMA_NODES;
    vtmp = vtmp/NUM_NUMA_NODES;

    ((float*)gexp_test)[i] = mtmp/num_nodes;
    ((float*)gvar_test)[i] = vtmp/num_nodes;
  }
  this->op_->GetParameterSet(3)->StartGradientComm(gexp_test);
  this->op_->GetParameterSet(4)->StartGradientComm(gvar_test);
#endif

#ifdef GETSTATS
#ifdef USE_MLSL
  node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
  node_id = 0;
#endif
  if(node_id == 0)
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
    if(in_dtype == DT_FLOAT)
    {
      string s = nname_ + "_Inp";
      float *ptr = (float*)tenBotData_[0]->getBuffer();
      MeanOfLayer((char*)s.c_str(), ptr, nImg*ifm0*ifhp*ifwp);

      s = nname_ + "_delMidp";
      ptr = (float*)tenMidDiff_->getBuffer();
      MeanOfLayer((char*)s.c_str(), ptr, nImg*ofm*mfhp*mfwp);

      s = nname_ + "_delWt_Aft";
      ptr = (float*)tenWeightDiff_->getBuffer();
      float *pptr = (float*)tenWeightDiff_->getPrivBuffer();
      float *p = (pptr == NULL) ? ptr : pptr;
      MeanOfLayer((char*)s.c_str(), p, ifm0*ofm*kh*kw);
    }
    else if(in_dtype == DT_BF16)
    {
      string s = nname_ + "_Inp";
      libxsmm_bfloat16 *ptr = (libxsmm_bfloat16*)tenBotData_[0]->getBuffer();
      memset(stptr, 0, nImg*ifm0*ifhp*ifwp);
      convert_bf16_f32(ptr, stptr, nImg*ifm0*ifhp*ifwp);
      MeanOfLayer((char*)s.c_str(), stptr, nImg*ifm0*ifhp*ifwp);

      s = nname_ + "_delMidp";
      ptr = (libxsmm_bfloat16*)tenMidDiff_->getBuffer();
      memset(stptr, 0, nImg*ofm*mfhp*mfwp);
      convert_bf16_f32(ptr, stptr, nImg*ofm*mfhp*mfwp);
      MeanOfLayer((char*)s.c_str(), stptr, nImg*ofm*mfhp*mfwp);

      s = nname_ + "_delgammap";
      float* delgamma = (float*)tenScaleDiff_->getBuffer();
      MeanOfLayer((char*)s.c_str(), delgamma, gparams_.nOutput);

      s = nname_ + "_delbetap";
      float* delbeta = (float*)tenShiftDiff_->getBuffer();
      MeanOfLayer((char*)s.c_str(), delbeta, gparams_.nOutput);

      s = nname_ + "_delWt_Aft";
#ifdef USE_MLSL
      MeanOfLayer((char*)s.c_str(), dwptr, ifm0*ofm*kh*kw);
#else
      ptr = (libxsmm_bfloat16*)tenWeightDiff_->getBuffer();
      memset(stptr, 0, ifm0*ofm*kh*kw);
      convert_bf16_f32(ptr, stptr, ifm0*ofm*kh*kw);
      MeanOfLayer((char*)s.c_str(), stptr, ifm0*ofm*kh*kw);
#endif
    }
  }
#endif
}

void FusedConvBNNode::solverStep()
{
#ifdef USE_MLSL
  int ifm = gparams_.nInput[0];
  int ofm = gparams_.nOutput;
  int kh = gparams_.kh;
  int kw = gparams_.kw;

  float *gwt = (float*)(tenWeightDiff_->getBuffer());
  float *delgamma = (float*)tenScaleDiff_->getBuffer();
  float *delbeta = (float*)tenShiftDiff_->getBuffer();
  void* gexp_test = tenMeanData_->getPrivBuffer();
  void* gvar_test = tenVarData_->getPrivBuffer();

  int wsize = ifm*ofm*kh*kw;

  void *mptr = op_->GetParameterSet(0)->WaitGradientComm();
  if(in_dtype == DT_FLOAT)
  {
    if(mptr != NULL && mptr != gwt)
      memcpy((void*)gwt, mptr, wsize*sizeof(float));
  }
  else if(in_dtype == DT_BF16)
  {
    if(mptr != NULL && mptr != dwptr)
      memcpy((void*)dwptr, mptr, wsize*sizeof(float));
    convert_f32_bf16(dwptr, (libxsmm_bfloat16*)gwt, wsize);
  }

  mptr = op_->GetParameterSet(1)->WaitGradientComm();
  if(mptr != NULL && mptr != delgamma)
      memcpy((void*)delgamma, mptr, ofm*sizeof(float));

  mptr = op_->GetParameterSet(2)->WaitGradientComm();
  if(mptr != NULL && mptr != delbeta)
      memcpy((void*)delbeta, mptr, ofm*sizeof(float));

  mptr = op_->GetParameterSet(3)->WaitGradientComm();
  if(mptr != NULL && mptr != gexp_test)
    memcpy((void*)gexp_test, mptr, ofm*sizeof(float));

  mptr = op_->GetParameterSet(4)->WaitGradientComm();
  if(mptr != NULL && mptr != gvar_test)
    memcpy((void*)gvar_test, mptr, ofm*sizeof(float));

#ifdef CHECK_BLOWUP_FP32
  float* ptr = (float*)tenWeightDiff_->getBuffer();
  for(int i=0; i<16; i++)
  {
    if(isnan(ptr[i]) || isinf(ptr[i]))
    {
      printf("Warning! %s layer Solver gradients are NaN or Inf\n", nname_.c_str());
      exit(-1);
    }
  }
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
#endif
}
