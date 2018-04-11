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
#include "Conv.hpp"
#include "fillers.hpp"

#ifdef USE_MLSL
#include "mpi.h"
#endif


using namespace std;
using namespace gxm;

ConvNode::ConvNode(ConvParams* p, MLEngine* e): NNNode(p, e)
{
  nname_ = p->get_node_name();
  ntype_ = p->get_node_type();
  bottom_ = p->get_bottom_names();
  top_ = p->get_top_names();
  bp_flag_ = p->get_bprop_flag();
  has_weights_ = true;
  compute_stats_ = p->get_compute_stats();
  bot_compute_engine_ = p->get_compute_engine();

  assert((bottom_.size() == 1) && (top_.size() == 1));
  bool bias_term = p->get_bias_term();

  tenTop_ = new Tensor(top_[0]);
  assert(tenTop_ != NULL);
  tenTop_->setOwner(this);
  tenTop_->setType(ACT);
  tenTopData_ = tenTop_->getBuf(DATA);
  tenTopData_->setBufferType(DATA);

#ifndef NDEBUG
  printf("bottom name %s\n",bottom_[0].c_str());
#endif

  if(bottom_[0] == "data")
    tenBot_ = e->get_tensor(bottom_[0], INPUT);
  else
    tenBot_ = e->get_tensor(bottom_[0], ACT);

  assert(tenBot_ != NULL);
  NNNode *pnn = (NNNode*)tenBot_->getOwner();
  setPrevNode(pnn);
  mode_ = pnn->getMode();
  pnn->set_top_compute_engine(p->get_compute_engine());
  bot_cengine_ = pnn->get_bot_compute_engine();

  tenBotData_ = tenBot_->getBuf(DATA);

  int out_dtype = p->get_data_type();
  int in_dtype = tenBotData_->getDataType();

  tenTopData_->setDataType(out_dtype);

  // Get input tensor shape (bottom)
  Shape* bs = tenBot_->getShape();
  assert(bs->ndims <= MAX_DIMS);

  // Create shape of output tensor (top)
  vector<int> vd = p->get_kernel_dims();
  vector<int> ovp = p->get_output_pads();
  vector<int> vp = p->get_pads();
  vector<int> vs = p->get_strides();

  assert((vd.size() == vp.size()) && (vd.size() == vs.size()) && (vs.size() == ovp.size()));

  shape_setzero(&ts_);

  ts_.ndims = bs->ndims; // Number of dimensions
  ts_.dims[0] = bs->dims[0]; // Minibatch size
  ts_.dims[1] = p->get_output(); // Num output feature maps

  ts_.dims[2] = (bs->dims[2] - vd[0] + 2*vp[0])/vs[0] + 1; // Height

  if(ts_.ndims == 4)
    ts_.dims[3] = (bs->dims[3] - vd[1] + 2*vp[1])/vs[1] + 1; // Width
  else if(ts_.ndims == 5)
  {
    ts_.dims[3] = (bs->dims[3] - vd[1] + 2*vp[1])/vs[1] + 1; // Width
    ts_.dims[4] = (bs->dims[4] - vd[2] + 2*vp[2])/vs[2] + 1; // Depth (for 3D)
  }

  tenTop_->setShape(&ts_);

  long long int tsize;
  int telem = ts_.dims[0] * ts_.dims[1] * (ts_.dims[2] + 2*ovp[0]) * (ts_.dims[3] + 2*ovp[1]);

  // Buffer space for sum and sum^2
  int tstats;
  if(compute_stats_)
    tstats = 2*ts_.dims[0]*ts_.dims[1];

  if(out_dtype == DT_FLOAT)
    tsize = telem*sizeof(float) + tstats*sizeof(double);
  else if(out_dtype == DT_DFP16)
    tsize = (telem + tstats)*sizeof(short);

  tenTopData_->setBufferSize(tsize);

  // Create FP weight tensor
  weight_ = top_[0] + "_wt";
  tenWeight_ = new Tensor(weight_);
  assert(tenWeight_ != NULL);
  tenWeight_->setOwner(this);
  tenWeight_->setType(CONVWEIGHT);

  shape_setzero(&ws_);

  ws_.ndims = ts_.ndims;      // Number of dimesions
  ws_.dims[0] = ts_.dims[1];  // Num output feature maps (from top tensor)
  ws_.dims[1] = bs->dims[1];  // Num input feature maps (from bottom tensor)
  ws_.dims[2] = vd[0];        // Kernel height

  if(ts_.ndims == 4)
  {
    ws_.dims[3] = vd[1]; // Kernel width
  }
  else if(ts_.ndims == 5)
  {
    ws_.dims[3] = vd[1];
    ws_.dims[4] = vd[2];
  }

  tenWeight_->setShape(&ws_);
  tenWeight_->setBufDataType(DATA, DT_FLOAT);
  tenWeightData_ = tenWeight_->getBuf(DATA);
  tenWeightData_->setBufferType(DATA);

  int welem = 1;
  long long int wsize;
  for(int i=0; i<ws_.ndims; i++)
    welem = welem*ws_.dims[i];

  // size of weights -- always in FP32.
  if((in_dtype == DT_FLOAT) && (out_dtype == DT_FLOAT))
    wsize = welem*sizeof(float);
  else if(in_dtype == DT_DFP16)
    wsize = welem*sizeof(float);

  tenWeightData_->setBufferSize(wsize);

  wfiller_type_ = p->get_weight_filler_type();
  variance_norm_ = p->get_variance_norm();
  std_ = p->get_std();

  lr_mult_ = p->get_lr_mult();
  decay_mult_ = p->get_decay_mult();

  // Create bias tensor
  long long int bisize;

  Shape bis;
  {
    if(bias_term)
    {
      bias_ = top_[0] + "_bias";
      tenBias_ = new Tensor(bias_);
      assert(tenBias_ != NULL);
      tenBias_->setOwner(this);
      tenBias_->setType(CONVBIAS);

      shape_setzero(&bis);

      bis.ndims = 1;
      bis.dims[0] = ts_.dims[1];
      tenBias_->setShape(&bis);
      tenBiasData_ = tenBias_->getBuf(DATA);
      tenBiasData_->setDataType(DT_FLOAT);
      tenBiasData_->setBufferType(DATA);

      bisize = bis.dims[0];
      bisize = bisize*sizeof(float); // Biases are always in FP32
      tenBiasData_->setBufferSize(bisize);

      bfiller_type_ = p->get_bias_filler_type();
      value_ = p->get_value();
    }
  }

  if(!e->is_inference_only()) {
    if(bp_flag_)
    {
      tenBotDiff_ = tenBot_->addBuf(); // DIFF type and index
      tenBotDiff_->setDataType(in_dtype);
      tenBotDiff_->setBufferType(DIFF);

      long long int bsize = bs->dims[0] * bs->dims[1] * (bs->dims[2] + 2*vp[0]) * (bs->dims[3] + 2*vp[1]);

      if((in_dtype == DT_FLOAT && out_dtype == DT_FLOAT) ||
          (in_dtype == DT_DFP16 && out_dtype == DT_FLOAT))
        bsize = bsize*sizeof(float);
      else if(in_dtype == DT_DFP16 && out_dtype == DT_DFP16)
        bsize = bsize*sizeof(short);

      // Set the size of the input-gradient buffer
      tenBotDiff_->setBufferSize(bsize);
    }

    if(has_weights_)
    {
      tenWeightDiff_ = tenWeight_->addBuf(); // DIFF type and index
      tenWeightDiff_->setDataType(DT_FLOAT);
      tenWeightDiff_->setBufferType(DIFF);

      tenWeightInc_ = tenWeight_->addBuf(); // SHARED type and index
      tenWeightInc_->setDataType(DT_FLOAT);
      tenWeightInc_->setBufferType(HISTORY);

      // Set the size of the weight-gradient buffer and the weight-increment buffer
      tenWeightDiff_->setBufferSize(welem*sizeof(float));
      tenWeightInc_->setBufferSize(welem*sizeof(float));

      if(bias_term)
      {
        tenBiasDiff_ = tenBias_->addBuf(); // DIFF type and index
        tenBiasDiff_->setDataType(DT_FLOAT);
        tenBiasDiff_->setBufferType(DIFF);

        tenBiasInc_ = tenBias_->addBuf(); // SHARED type and index
        tenBiasInc_->setDataType(DT_FLOAT);
        tenBiasInc_->setBufferType(HISTORY);

        // Set the size of the weight-gradient buffer and the weight-increment buffer
        tenBiasDiff_->setBufferSize(bisize);
        tenBiasInc_->setBufferSize(bisize);
      }
    }
  }
  else {
    tenBotDiff_ = NULL;
    tenWeightDiff_ = NULL;
    tenWeightInc_ = NULL;
    tenBiasDiff_ = NULL;
    tenBiasInc_ = NULL;
  }

  // Register output tensor in tensor map
  bool inserted = e->register_tensor(top_[0], ACT, tenTop_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",top_[0].c_str());

  // Register weight tensor in weight tensor map
  inserted = e->register_tensor(weight_, CONVWEIGHT, tenWeight_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",weight_.c_str());

  // Register bias tensor in bias tensor map
  if(bias_term)
  {
    inserted = e->register_tensor(bias_, CONVBIAS, tenBias_);
    if(!inserted)
      printf("Warning: Tensor %s already registered\n",bias_.c_str());
  }


  // Setup parameter structure for convolution computation in library
  gparams_.bdims = bs->ndims;
  gparams_.tdims = ts_.ndims;
  gparams_.wdims = ws_.ndims;
  gparams_.bidims = bis.ndims;

  gparams_.node_name = nname_;
  gparams_.nInput = bs->dims[1];
  gparams_.nOutput = ts_.dims[1];
  gparams_.batch_size = bs->dims[0];
  gparams_.iHeight = bs->dims[2];
  gparams_.iWidth = bs->dims[3];
  gparams_.iDepth = bs->dims[4];
  gparams_.oHeight = ts_.dims[2];
  gparams_.oWidth = ts_.dims[3];
  gparams_.oDepth = ts_.dims[4];
  gparams_.pad_h = vp[0];
  gparams_.pad_w = vp[1];
  gparams_.pad_d = vp[2];
  gparams_.physical_padding = p->get_physical_padding();
  gparams_.compute_stats = compute_stats_;

  if(gparams_.physical_padding)
  {
    gparams_.ipad_h = (nname_ == "conv1") ? 0 : vp[0];
    gparams_.ipad_w = (nname_ == "conv1") ? 0 : vp[1];
    gparams_.ipad_d = (nname_ == "conv1") ? 0 : vp[2];
  }
  else
  {
    gparams_.ipad_h = 0;
    gparams_.ipad_w = 0;
    gparams_.ipad_d = 0;
  }

  if(gparams_.physical_padding)
  {
    gparams_.opad_h = (nname_ == "conv1") ? 0 : ovp[0];
    gparams_.opad_w = (nname_ == "conv1") ? 0 : ovp[1];
  }
  else
  {
    gparams_.opad_h = 0;
    gparams_.opad_w = 0;
    gparams_.opad_d = 0;
  }

  gparams_.group = p->get_group();
  gparams_.stride_h = vs[0];
  gparams_.stride_w = vs[1];
  gparams_.stride_d = vs[2];
  gparams_.kh = ws_.dims[2];
  gparams_.kw = ws_.dims[3];
  gparams_.kd = ws_.dims[4];

  gparams_.bias_term = bias_term;
  gparams_.relu = p->get_fused_relu();
  gparams_.bwd_relu = p->get_bwd_relu();

  gparams_.in_data_type = in_dtype;
  gparams_.out_data_type = out_dtype;
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
  myRegInfo->AddInput(gparams_.nInput, gparams_.iWidth*gparams_.iHeight, dt);
  myRegInfo->AddOutput(gparams_.nOutput, gparams_.oWidth*gparams_.oHeight, dt);
  myRegInfo->AddParameterSet(gparams_.nInput*gparams_.nOutput/gparams_.group, gparams_.kw*gparams_.kh, dt, false);

  if(bias_term)
    myRegInfo->AddParameterSet(gparams_.nOutput, 1, dt, false);

  myRegInfo->Validate();
  size_t opIdx = s->AddOperation(myRegInfo, e->get_distribution());
  this->op_ = s->GetOperation(opIdx);
  s->DeleteOperationRegInfo(myRegInfo);
#endif

  configure(p->get_compute_engine());
}

void ConvNode::fillWeightBuffers(TensorBuf* tBuf, int buftype, long long int size)
{
  int dtype = DT_FLOAT;
  void *ptr = tBuf->getBuffer();

#ifdef USE_MLSL
    unsigned int node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
    unsigned int node_id = 0;
#endif

  if(buftype == DATA)
  {
    int n = gparams_.batch_size;
    int ic = gparams_.nInput;
    int oc = gparams_.nOutput;
    int kh = gparams_.kh;
    int kw = gparams_.kw;
    int g = gparams_.group;
    int fanin = (ic * kh * kw)/g;
    int fanout = (oc * kh * kw)/g;
    int welem = ic * oc * kh * kw;

    initBuffer(ptr, dtype, variance_norm_, fanin, fanout, welem*sizeof(float), wfiller_type_, (unsigned int)(node_id+PRIME_SEED), std_);

#ifdef USE_MLSL
    if(dtype == DT_FLOAT)
      MPI_Bcast(ptr, welem, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif

    int in_dtype = tenBotData_->getDataType();
    int out_dtype = tenTopData_->getDataType();

    // Quantization of weights
    if(in_dtype == DT_DFP16 || out_dtype == DT_DFP16)
    {
      if(i16_wt_ptr == NULL)
        i16_wt_ptr = (short*)libxsmm_aligned_malloc(welem*sizeof(short), 2097152);
      tBuf->setLPBuffer((void*)i16_wt_ptr);

      unsigned char scf_filter;
      libxsmm_dnn_quantize_fil((float*)ptr, i16_wt_ptr, oc, ic, kh, kw, 16, 8, 16, 16, 2, 2, &scf_filter, LIBXSMM_DNN_QUANT_FPHW_ROUND);
      tBuf->setLPSF(scf_filter);
    }
  }
  else
    memset(ptr, 0, size);
}

void ConvNode::fillWeightMultipliers(float* lr, float* decay, long long int size)
{
  for(int i=0; i < size; i++)
  {
    lr[i] = lr_mult_[0];
    decay[i] = decay_mult_[0];
  }
}

void ConvNode::fillBiasBuffers(TensorBuf* tBuf, int buftype, long long int size)
{
  int dtype = DT_FLOAT;
  void *ptr = tBuf->getBuffer();

  if(buftype == DATA)
  {
    initConstantBuffer(ptr, dtype, size, "CONSTANT", value_);
  }
  else
    memset(ptr, 0, size);
}

void ConvNode::fillBiasMultipliers(float* lr, float* decay, long long int size)
{
  if(gparams_.bias_term)
  {
    for(int i=0; i < size; i++)
    {
      lr[i] = lr_mult_[1];
      decay[i] = decay_mult_[1];
    }
  }
}

void ConvNode::Checkpoint(TensorBuf *tBuf, string name, string format)
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
        if(name.find("wt") != name.npos)
        {
          ptr = _mm_malloc(bytes, 64);
          assert(ptr != NULL);
          impl->dumpBuffer(tBuf, ptr);
        }
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

void ConvNode::configure(int engine)
{
  switch(engine)
  {
    case XSMM:
      impl = new ConvXSMM(&gparams_, engine);
  }
}

void ConvNode::forwardPropagate()
{
  int nImg = gparams_.batch_size;
  int ifm = gparams_.nInput;
  int ofm = gparams_.nOutput;
  int ifh = gparams_.iHeight;
  int ifhp = ifh + 2*gparams_.ipad_h;
  int ifw = gparams_.iWidth;
  int ifwp = ifw + 2*gparams_.ipad_w;
  int ofh = gparams_.oHeight;
  int ofw = gparams_.oWidth;
  int ofhp = ofh + 2*gparams_.opad_h;
  int ofwp = ofw + 2*gparams_.opad_w;
  int kh = gparams_.kh;
  int kw = gparams_.kw;

#ifndef NDEBUG
  // printf("Executing FP %s: input %p, weights %p, output %p\n",NNNode::nname_.c_str(), bot, wt, top);
  printf("Executing FP %s\n",NNNode::nname_.c_str());
  printf("Inputs: %d x %d x %d\n",ifm, ifh, ifw);
  printf("Outputs: %d x %d x %d\n",ofm, ofh, ofw);
  printf("Weights: %d x %d x %d x %d\n", ifm, ofm, kh, kw);
  printf("Bias: %d\n", ofm);

  if (gparams_.relu) printf("Fused relu\n");
#endif

  impl->set_top_compute_engine(top_compute_engine_);
  impl->set_bot_compute_engine(bot_cengine_);
  impl->set_node_name(nname_);
  impl->set_scratch_buffer(tenScratchData_);

  long long int size = nImg * ofm * ofhp * ofwp;

  if(first_fp)
  {
    if(tenTopData_->getDataType() == DT_FLOAT)
    {
      float* ptr = (float*)tenTopData_->getBuffer();

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<size; i++)
        ptr[i] = 0;
    }
    else if(tenTopData_->getDataType() == DT_DFP16)
    {
      short* ptr = (short*)tenTopData_->getLPBuffer();

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<size; i++)
        ptr[i] = 0;
    }

    first_fp = false;
  }

  if(tenTopData_->getDataType() == DT_FLOAT)
  {
    float* ptr = (float*)tenTopData_->getBuffer();
    float* sptr = ptr + size;

    /* @TODO move this into Batch Norm/LIBXSMM */
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<4*nImg*ofm; i++)
      sptr[i] = 0;
  }
  else if(tenTopData_->getDataType() == DT_DFP16)
  {
    short* ptr = (short*)tenTopData_->getLPBuffer();
    short* sptr = ptr + size;

    /* @TODO move this into Batch Norm/LIBXSMM */
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<4*nImg*ofm; i++)
      sptr[i] = 0;
  }

  impl->forwardPropagate(tenBotData_, tenWeightData_, tenBiasData_, tenTopData_);

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
  if(node_id == 0)
#endif
  {
    float *ptr, *pptr, *p;

    if(eptr_->get_current_batch() % STATFREQ == 0)
    {
      string s = nname_ + "_Inp";
      ptr = (float*)tenBotData_->getBuffer();
      pptr = (float*)tenBotData_->getPrivBuffer();
      p = (pptr == NULL) ? ptr : pptr;
      MeanOfLayer((char*)s.c_str(), p, nImg*ifm*ifhp*ifwp);

      s = nname_ + "_Wt";
      ptr = (float*)tenWeightData_->getBuffer();
      pptr = (float*)tenWeightData_->getPrivBuffer();
      p = (pptr == NULL) ? ptr : pptr;
      MeanOfLayer((char*)s.c_str(), p, ifm*ofm*kh*kw);

      if(gparams_.bias_term)
      {
        s = nname_ + "_Bias";
        p = (float*)tenBiasData_->getBuffer();
        MeanOfLayer((char*)s.c_str(), p, ofm);
      }

      s = nname_ + "_Outp";
      ptr = (float*)tenTopData_->getBuffer();
      pptr = (float*)tenTopData_->getPrivBuffer();
      p = (pptr == NULL) ? ptr : pptr;
      MeanOfLayer((char*)s.c_str(), p, nImg*ofm*ofhp*ofwp);

      s = nname_ + "_sump";
      int offset = nImg*ofm*ofhp*ofwp*sizeof(float);
      void* m = (void*)p + offset;
      MeanOfLayer((char*)s.c_str(), (double*)m, nImg*ofm);

      s = nname_ + "_sum2p";
      void* m2 = (void*)m + nImg*ofm*sizeof(double);
      MeanOfLayer((char*)s.c_str(), (double*)m2, nImg*ofm);
    }
  }
#endif
}

void ConvNode::backPropagate()
{

  int nImg = gparams_.batch_size;
  int ifm = gparams_.nInput;
  int ofm = gparams_.nOutput;
  int ifh = gparams_.iHeight;
  int ifhp = ifh + 2*gparams_.ipad_h;
  int ifw = gparams_.iWidth;
  int ifwp = ifw + 2*gparams_.ipad_w;
  int ofh = gparams_.oHeight;
  int ofw = gparams_.oWidth;
  int ofhp = ofh + 2*gparams_.opad_h;
  int ofwp = ofw + 2*gparams_.opad_w;
  int kh = gparams_.kh;
  int kw = gparams_.kw;

#ifdef DEBUG
  printf("Executing BP %s\n",NNNode::nname_.c_str());
  printf("Grad Outputs: %d x %d x %d\n", ofm, ofh, ofw);
  printf("Grad Inputs: %d x %d x %d\n", ifm, ifh, ifw);
  printf("Weights: %d x %d x %d x %d\n", ofm, ifm, kh, kw);
#endif

  tenTopDiff_ = tenTop_->getBuf(DIFF);

#if 1
  if(first_bp)
  {
    long long int size = nImg * ifm * ifhp *ifwp;

    int in_dtype = tenBotData_->getDataType();
    int out_dtype = tenTopData_->getDataType();

    if((in_dtype == DT_DFP16 && out_dtype == DT_FLOAT)
        || (in_dtype == DT_FLOAT && out_dtype == DT_FLOAT))
    {
      float* ptr = (float*)tenBotDiff_->getBuffer();
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<size; i++)
        ptr[i] = 0;
    }
    else if(in_dtype == DT_DFP16 && out_dtype == DT_DFP16)
    {
      short* ptr = (short*)tenBotDiff_->getBuffer();
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<size; i++)
        ptr[i] = 0;
    }

   first_bp = false;
  }
#endif

  impl->backPropagate(tenTopData_, tenWeightData_, tenTopDiff_, tenBotDiff_);

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
  float *ptr, *pptr, *p, *bias;
#ifdef USE_MLSL
  unsigned int node_id_ = MLSL::Environment::GetEnv().GetProcessIdx();
  if(node_id_ == 0)
#endif
  {
    if(eptr_->get_current_batch() % STATFREQ == 0)// && gparams_.ipad_h)
    {
      string s = nname_ + "_delOutp";

      ptr = (float*)tenTopDiff_->getBuffer();
      pptr = (float*)tenTopDiff_->getPrivBuffer();
      p = (pptr == NULL) ? ptr : pptr;
      printf("Conv deloutp %p\n",p);
      MeanOfLayer((char*)s.c_str(), p, nImg*ofm*ofhp*ofwp);

      s = nname_ + "_Wt";
      ptr = (float*)tenWeightData_->getBuffer();
      pptr = (float*)tenWeightData_->getPrivBuffer();
      p = (pptr == NULL) ? ptr : pptr;
      MeanOfLayer((char*)s.c_str(), p, ifm*ofm*kh*kw);

      s = nname_ + "_delInp";
      ptr = (float*)tenBotDiff_->getBuffer();
      pptr = (float*)tenBotDiff_->getPrivBuffer();
      p = (pptr == NULL) ? ptr : pptr;
      MeanOfLayer((char*)s.c_str(), p, nImg*ifm*ifhp*ifwp);
    }
  }
#endif
}

void ConvNode::weightUpdate()
{
  int nImg = gparams_.batch_size;
  int ifm = gparams_.nInput;
  int ofm = gparams_.nOutput;
  int ifh = gparams_.iHeight;
  int ifw = gparams_.iWidth;
  int ofh = gparams_.oHeight;
  int ofw = gparams_.oWidth;
  int ofhp = ofh + 2*gparams_.opad_h;
  int ofwp = ofw + 2*gparams_.opad_w;
  int ifhp = ifh + 2*gparams_.ipad_h;
  int ifwp = ifw + 2*gparams_.ipad_w;
  int kh = gparams_.kh;
  int kw = gparams_.kw;

#ifdef DEBUG
  // printf("Executing WU %s: grad_output %p, grad_weights %p, input %p\n",NNNode::nname_.c_str(), gtop, gwt, bot);
  printf("Executing WU %s\n",NNNode::nname_.c_str());
  printf("Grad Outputs: %d x %d x %d\n",ofm, ofh,ofw);
  printf("Inputs: %d x %d x %d\n",ifm, ifh, ifw);
  printf("del-Weights: %d x %d x %d x %d\n", ofm, ifm, kh, kw);
  printf("del-Biases: %d\n", ofm);
#endif

#ifdef GETSTATS
{
  float *ptr, *pptr, *p;

#ifdef USE_MLSL
  unsigned int node_id = MLSL::Environment::GetEnv().GetProcessIdx();
  if(node_id == 0 && eptr_->get_current_batch() % STATFREQ == 0)
#else
  if(eptr_->get_current_batch() % STATFREQ == 0)
#endif
  {
    string s = nname_ + "_delWt_Bef";
    ptr = (float*)tenWeightDiff_->getBuffer();
    pptr = (float*)tenWeightDiff_->getPrivBuffer();
    p = (pptr == NULL) ? ptr : pptr;
    MeanOfLayer((char*)s.c_str(), p, ifm*ofm*kh*kw);

    if(gparams_.bias_term)
    {
      s = nname_ + "_delBias_Bef";
      p = (float*)tenBiasDiff_->getBuffer();
      MeanOfLayer((char*)s.c_str(), p, ofm);
    }
  }
}
#endif

  tenTopDiff_ = tenTop_->getBuf(DIFF);

  impl->weightUpdate(tenBotData_, tenTopDiff_, tenWeightDiff_, tenBiasDiff_);

#ifdef CHECK_BLOWUP_FP32
  float* ptr = (float*)tenWeightDiff_->getBuffer();
  for(int i=0; i<16; i++)
  {
    if(isnan(ptr[i]) || isinf(ptr[i]))
    {
      printf("Warning! %s layer WU gradients are NaN or Inf\n", nname_.c_str());
      exit(-1);
    }
  }
#endif

#ifdef USE_MLSL
  void *mptr = tenWeightDiff_->getBuffer();
  void *mpptr = tenWeightDiff_->getPrivBuffer();
  void *mp = (mpptr == NULL) ? mptr : mpptr;

  op_->GetParameterSet(0)->StartGradientComm(mp);
  if(gparams_.bias_term)
    op_->GetParameterSet(1)->StartGradientComm(tenBiasDiff_->getBuffer());
#endif

#ifdef GETSTATS
  float *ptr, *pptr, *p;

#ifdef USE_MLSL
  unsigned int node_id = MLSL::Environment::GetEnv().GetProcessIdx();
  if(node_id == 0 && eptr_->get_current_batch() % STATFREQ == 0)
#else
  if(eptr_->get_current_batch() % STATFREQ == 0)
#endif
  {
    string s = nname_ + "_Inp";
    ptr = (float*)tenBotData_->getBuffer();
    pptr = (float*)tenBotData_->getPrivBuffer();
    p = (pptr == NULL) ? ptr : pptr;
    MeanOfLayer((char*)s.c_str(), p, nImg*ifm*ifhp*ifwp);

    s = nname_ + "_delOutp";

    ptr = (float*)tenTopDiff_->getBuffer();
    pptr = (float*)tenTopDiff_->getPrivBuffer();
    p = (pptr == NULL) ? ptr : pptr;
    MeanOfLayer((char*)s.c_str(), p, nImg*ofm*ofhp*ofwp);

    s = nname_ + "_delWt_Aft";
    ptr = (float*)tenWeightDiff_->getBuffer();
    pptr = (float*)tenWeightDiff_->getPrivBuffer();
    p = (pptr == NULL) ? ptr : pptr;
    MeanOfLayer((char*)s.c_str(), p, ifm*ofm*kh*kw);

    if(gparams_.bias_term)
    {
      s = nname_ + "_delBias_Aft";
      p = (float*)tenBiasDiff_->getBuffer();
      MeanOfLayer((char*)s.c_str(), p, ofm);
    }
  }
#endif
}

void ConvNode::solverStep()
{
#ifdef RETURNALL
  return;
#endif

  int nImg = gparams_.batch_size;
  int ifm = gparams_.nInput;
  int ofm = gparams_.nOutput;
  int ifh = gparams_.iHeight;
  int ifw = gparams_.iWidth;
  int ofh = gparams_.oHeight;
  int ofw = gparams_.oWidth;
  int kh = gparams_.kh;
  int kw = gparams_.kw;

  float *wt_prv_ptr = (float*)tenWeightData_->getPrivBuffer();
  float *wt_ptr = (float*)(tenWeightData_->getBuffer());

  float *gwt_prv_ptr = (float*)(tenWeightDiff_->getPrivBuffer());
  float *gwt_ptr = (float*)(tenWeightDiff_->getBuffer());

  float *wt = (wt_prv_ptr == NULL) ? wt_ptr : wt_prv_ptr;
  float *gwt = (gwt_prv_ptr == NULL) ? gwt_ptr : gwt_prv_ptr;

  float *iwt = (float*)(tenWeightInc_->getBuffer());

  float *bias_prv_ptr, *bias_ptr, *bias;
  float *gbias_prv_ptr, *gbias_ptr, *gbias, *ibias;

  if(gparams_.bias_term)
  {
    bias_prv_ptr = (float*)tenBiasData_->getPrivBuffer();
    bias_ptr = (float*)(tenBiasData_->getBuffer());
    bias = (bias_prv_ptr == NULL) ? bias_ptr : bias_prv_ptr;

    gbias_prv_ptr = (float*)tenBiasDiff_->getPrivBuffer();
    gbias_ptr = (float*)(tenBiasDiff_->getBuffer());
    gbias = (gbias_prv_ptr == NULL) ? gbias_ptr : gbias_prv_ptr;
    ibias = (float*)(tenBiasInc_->getBuffer());
  }

  int wsize = ifm*ofm*kh*kw;

#ifdef DEBUG
  printf("Executing Solver: weights %p, grad_weights %p, bias %p, grad_bias %p\n", wt, gwt, bias, gbias);
  printf("Grad Weights: %d x %d x %d x %d\n", ofm, ifm, kh, kw);
  printf("Grad Biases: %d\n",ofm);
#endif

#ifdef GETSTATS
#ifdef USE_MLSL
  unsigned int node_id = MLSL::Environment::GetEnv().GetProcessIdx();
  if(node_id == 0 && eptr_->get_current_batch() % STATFREQ == 0)
#else
  if(eptr_->get_current_batch() % STATFREQ == 0)
#endif
  {
    string s = nname_ + "_OldWt";
    MeanOfLayer((char*)s.c_str(), wt, ifm*ofm*kh*kw);
    if(gparams_.bias_term)
    {
      s = nname_ + "_OldBias";
      MeanOfLayer((char*)s.c_str(), bias, ofm);
    }
  }
#endif

  int num_nodes = 1;

#ifdef USE_MLSL
  void *mptr = op_->GetParameterSet(0)->WaitGradientComm();
  if(mptr != NULL && mptr != gwt)
    memcpy((void*)gwt, mptr, wsize*sizeof(float));

  if(gparams_.bias_term)
  {
    mptr = op_->GetParameterSet(1)->WaitGradientComm();
    if(mptr != NULL && mptr != gbias)
      memcpy((void*)gbias, mptr, ofm*sizeof(float));
  }
  num_nodes = MLSL::Environment::GetEnv().GetProcessCount();
#endif

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
#endif

  if(solver_->getGlobalFlag())
    return;

#ifdef DUMP_WT_DATA
  float *wtemp = (float*)_mm_malloc(wsize*sizeof(float), 64);
  string fname;
  FILE* f;
  int iter;

  {
    iter = eptr_->get_current_batch();

    impl->dumpBuffer(tenWeightDiff_, wtemp);
    fname = gparams_.node_name + "_solver_delwt_" + to_string(iter);
    f = fopen(fname.c_str(), "w");
    for(int i=0; i<wsize; i++)
      fprintf(f, "%g\n", wtemp[i]);
    fclose(f);

    if(gparams_.bias_term)
    {
      fname = gparams_.node_name + "_solver_delbias_" + to_string(iter);
      f = fopen(fname.c_str(), "w");
      for(int i=0; i<ofm; i++)
        fprintf(f, "%g\n", gbias[i]);
      fclose(f);
    }

    impl->dumpBuffer(tenWeightData_, wtemp);
    fname = gparams_.node_name + "_solver_wt_" + to_string(iter);
    f = fopen(fname.c_str(), "w");
    for(int i=0; i<wsize; i++)
      fprintf(f, "%g\n", wtemp[i]);
    fclose(f);

    if(gparams_.bias_term)
    {
      fname = gparams_.node_name + "_solver_bias_" + to_string(iter);
      f = fopen(fname.c_str(), "w");
      for(int i=0; i<ofm; i++)
        fprintf(f, "%g\n", bias[i]);
      fclose(f);
    }
  }
#endif

  solver_->applyUpdate(wt, iwt, gwt, wsize, lr_mult_[0], decay_mult_[0]);
#if 1
  if(gparams_.bias_term)
    solver_->applyUpdate(bias, ibias, gbias, ofm, lr_mult_[1], decay_mult_[1]);
#endif

#ifdef DUMP_WT_DATA
  {
    impl->dumpBuffer(tenWeightData_, wtemp);
    fname = gparams_.node_name + "_solver_newwt_" + to_string(iter);
    f = fopen(fname.c_str(), "w");
    for(int i=0; i<wsize; i++)
      fprintf(f, "%g\n", wtemp[i]);
    fclose(f);

    impl->dumpBuffer(tenWeightInc_, wtemp);
    fname = gparams_.node_name + "_solver_wtinc_" + to_string(iter);
    f = fopen(fname.c_str(), "w");
    for(int i=0; i<wsize; i++)
      fprintf(f, "%g\n", wtemp[i]);
    fclose(f);

    if(gparams_.bias_term)
    {
      fname = gparams_.node_name + "_solver_newbias_" + to_string(iter);
      f = fopen(fname.c_str(), "w");
      for(int i=0; i<ofm; i++)
        fprintf(f, "%g\n", bias[i]);
      fclose(f);

      fname = gparams_.node_name + "_solver_biasinc_" + to_string(iter);
      f = fopen(fname.c_str(), "w");
      for(int i=0; i<ofm; i++)
        fprintf(f, "%g\n", ibias[i]);
      fclose(f);
    }
  }
  _mm_free(wtemp);
#endif

#ifdef GETSTATS
#ifdef USE_MLSL
  if(node_id == 0 && eptr_->get_current_batch() % STATFREQ == 0)
#else
  if(eptr_->get_current_batch() % STATFREQ == 0)
#endif
  {
    string s = nname_ + "_WInc";
    MeanOfLayer((char*)s.c_str(), iwt, ifm*ofm*kh*kw);
    s = nname_ + "Wt";
    MeanOfLayer((char*)s.c_str(), wt, ifm*ofm*kh*kw);
    if(gparams_.bias_term)
    {
      s = nname_ + "BiasInc";
      MeanOfLayer((char*)s.c_str(), ibias, ofm);
      s = nname_ + "Bias";
      MeanOfLayer((char*)s.c_str(), bias, ofm);
    }
  }
#endif

}
