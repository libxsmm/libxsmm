/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
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

  out_dtype = p->get_data_type();
  in_dtype = tenBotData_->getDataType();

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
  ts_.dims[3] = (bs->dims[3] - vd[1] + 2*vp[1])/vs[1] + 1; // Width

  tenTop_->setShape(&ts_);

  long long int tsize;
  int telem = ts_.dims[0] * ts_.dims[1] * (ts_.dims[2] + 2*ovp[0]) * (ts_.dims[3] + 2*ovp[1]);

  // Buffer space for sum and sum^2
  int tstats=0;
  if(compute_stats_)
    tstats = 2*ts_.dims[0]*ts_.dims[1];

  if(out_dtype == DT_FLOAT)
    tsize = telem*sizeof(float) + tstats*sizeof(float);
  else if(out_dtype == DT_BF16)
    tsize = telem*sizeof(libxsmm_bfloat16) + tstats*sizeof(float);

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

  // size of master weights -- FP32
  wsize = welem*sizeof(float);

  gparams_.num_numa_nodes = NUM_NUMA_NODES;
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
          (in_dtype == DT_BF16 && out_dtype == DT_FLOAT))
        bsize = bsize*sizeof(float);
      else if(in_dtype == DT_BF16 && out_dtype == DT_BF16)
        bsize = bsize*sizeof(libxsmm_bfloat16);

      // Set the size of the input-gradient buffer
      tenBotDiff_->setBufferSize(bsize);
    }

    if(has_weights_)
    {
      tenWeightDiff_ = tenWeight_->addBuf(); // DIFF type and index
      tenWeightDiff_->setBufferType(DIFF);

      tenWeightInc_ = tenWeight_->addBuf(); // SHARED type and index
      tenWeightInc_->setBufferType(HISTORY);
      tenWeightInc_->setDataType(DT_FLOAT);
      tenWeightInc_->setBufferSize(welem*sizeof(float));

      if(in_dtype == DT_FLOAT)
      {
        tenWeightDiff_->setDataType(DT_FLOAT);
        tenWeightDiff_->setBufferSize(welem*sizeof(float));
      }
      else if(in_dtype == DT_BF16)
      {
        tenWeightDiff_->setDataType(DT_BF16);
#ifdef BF16_MLSL
        tenWeightDiff_->setBufferSize(welem*sizeof(libxsmm_bfloat16));
#else
        tenWeightDiff_->setBufferSize(welem*sizeof(float));
#endif
      }

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
  gparams_.oHeight = ts_.dims[2];
  gparams_.oWidth = ts_.dims[3];
  gparams_.pad_h = vp[0];
  gparams_.pad_w = vp[1];
  gparams_.physical_padding = p->get_physical_padding();
  gparams_.compute_stats = compute_stats_;

  if(gparams_.physical_padding)
  {
    gparams_.ipad_h = vp[0];
    gparams_.ipad_w = vp[1];
  }
  else
  {
    gparams_.ipad_h = 0;
    gparams_.ipad_w = 0;
  }

  if(gparams_.physical_padding)
  {
    gparams_.opad_h = ovp[0];
    gparams_.opad_w = ovp[1];
  }
  else
  {
    gparams_.opad_h = 0;
    gparams_.opad_w = 0;
  }

  gparams_.group = p->get_group();
  gparams_.stride_h = vs[0];
  gparams_.stride_w = vs[1];
  gparams_.kh = ws_.dims[2];
  gparams_.kw = ws_.dims[3];

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
  myRegInfo->AddParameterSet(gparams_.nInput*gparams_.nOutput/gparams_.group, gparams_.kw*gparams_.kh, dt, false);

  if(bias_term)
    myRegInfo->AddParameterSet(gparams_.nOutput, 1, dt, false);

  myRegInfo->Validate();
  size_t opIdx = s->AddOperation(myRegInfo, e->get_distribution());
  this->op_ = s->GetOperation(opIdx);
  s->DeleteOperationRegInfo(myRegInfo);
  e->get_wtgrad_comms_vec().push_back(op_);
#endif

  configure(p->get_compute_engine());
}

void ConvNode::fillWeightBuffers(TensorBuf* tBuf, int buftype, long long int size)
{
  void *ptr = tBuf->getBuffer();

#ifdef USE_MLSL
  unsigned int node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
  unsigned int node_id = 0;
#endif

  int ic = gparams_.nInput;
  int oc = gparams_.nOutput;
  int kh = gparams_.kh;
  int kw = gparams_.kw;
  int g = gparams_.group;
  int fanin = (ic * kh * kw)/g;
  int fanout = (oc * kh * kw)/g;
  int welem = ic * oc * kh * kw;

  if(buftype == DATA)
  {
    if(node_id == 0)
      initBuffer(ptr, variance_norm_, fanin, fanout, welem*sizeof(float), wfiller_type_, std_);

#ifdef USE_MLSL
    MPI_Bcast(ptr, welem, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
  }
  else if(buftype == HISTORY || buftype == DIFF)
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
  void *ptr = tBuf->getBuffer();

  if(buftype == DATA)
  {
    initConstantBuffer(ptr, size, "CONSTANT", value_);
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
  int buftype = tBuf->getBufferType();

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
#if 0
        if(name.find("wt") != name.npos)
        {
          ptr = _mm_malloc(bytes, 64);
          assert(ptr != NULL);
          impl->dumpBuffer(tBuf, ptr);
        }
        else
#endif
          ptr = tBuf->getBuffer();

        size_t b = fwrite(ptr, 1, bytes, f);
        assert((long long int)b == bytes);

#if 0
        if(name.find("wt") != name.npos)
          _mm_free(ptr);
#endif
      }
      else
        printf("Warning: could not checkpoint to file %s\n",name.c_str());
    }
    else
    {
      f = fopen(name.c_str(), "w");
      if(f != NULL)
      {
#if 0
        if(name.find("wt") != name.npos)
        {
          ptr = _mm_malloc(bytes, 64);
          assert(ptr != NULL);
          impl->dumpBuffer(tBuf, ptr);
        }
        else
#endif
          ptr = tBuf->getBuffer();

        for(int i=0; i<bytes/sizeof(float); i++)
          fprintf(f, "%f\n", *((float*)ptr + i));

#if 0
        if(name.find("wt") != name.npos)
          _mm_free(ptr);
#endif
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

void ConvNode::convert_f32_bf16(float* in, libxsmm_bfloat16* out, int len)
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

void ConvNode::convert_bf16_f32(libxsmm_bfloat16* in, float* out, int len)
{
#if 1

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = omp_get_thread_num();
    int ntps = gparams_.num_threads/gparams_.num_numa_nodes;
    int n = tid/ntps;
    if(n == 0)
    {
      int lenv = len/16;
      int rem = lenv % ntps;
      int jobs = (rem == 0) ? (lenv/ntps)*16 : ((lenv-rem)/ntps)*16;
      int tb = (tid*jobs < len) ? tid*jobs : len;
      int te = ((tid+1)*jobs < len) ? (tid+1)*jobs : len;

      for (int i = tb; i < te; i+=16 ) {
        __m256i vbfp16    = _mm256_loadu_si256( (const __m256i*)(in+i) );
        __m512  vfp32     = gxm_bfp16_to_fp32_avx512f( vbfp16 );
        _mm512_storeu_ps( out+i, vfp32 );
      }

      //Remainder processing
      if(tid == 0)
      {
        if(rem > 0)
        {
          for(int i=ntps*jobs; i<len; i+=16)
          {
            __m256i vbfp16    = _mm256_loadu_si256( (const __m256i*)(in+i) );
            __m512  vfp32     = gxm_bfp16_to_fp32_avx512f( vbfp16 );
            _mm512_storeu_ps( out+i, vfp32 );
          }
        }
      }
    }
  }
#else

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    int tid = omp_get_thread_num();
    int ntps = gparams_.num_threads/gparams_.num_numa_nodes;
    int n = tid/ntps;

    if(n == 0)
    {
      union libxsmm_bfloat16_hp delwt_32_0;
      delwt_32_0.i[0] = 0;

      int jobs = (len % ntps == 0) ? len/ntps : len/ntps + 1;
      int tb = (tid*jobs < len) ? tid*jobs : len;
      int te = ((tid+1)*jobs < len) ? (tid+1)*jobs : len;

      for(int j=tb; j<te; j++)
      {
        delwt_32_0.i[1] = in[j];
        out[j] = delwt_32_0.f;
      }
    }
  }
#endif
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
    else if(tenTopData_->getDataType() == DT_BF16)
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

    first_fp = false;
  }

  if(tenTopData_->getDataType() == DT_FLOAT)
  {
    float* ptr = (float*)tenTopData_->getBuffer();
    if(compute_stats_)
    {
      float* sptr = ptr + size;

      /* @TODO move this into Batch Norm/LIBXSMM */
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<2*nImg*ofm; i++)
        sptr[i] = 0;
    }
  }
  else if(tenTopData_->getDataType() == DT_BF16)
  {
    libxsmm_bfloat16* ptr = (libxsmm_bfloat16*)tenTopData_->getBuffer();
    if(compute_stats_)
    {
      libxsmm_bfloat16* sptr = ptr + size;

      /* @TODO move this into Batch Norm/LIBXSMM */
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<2*nImg*ofm; i++)
        sptr[i] = 0;
    }
  }

  impl->forwardPropagate(tenBotData_, tenWeightData_, tenWeightInc_, tenBiasData_, tenTopData_);

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
  if(node_id == 0)
#endif
  {
    if(out_dtype == DT_FLOAT)
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

        if(compute_stats_)
        {
          s = nname_ + "_sump";
          int offset = nImg*ofm*ofhp*ofwp*sizeof(float);
          void* m = (void*)p + offset;
          MeanOfLayer((char*)s.c_str(), (double*)m, nImg*ofm);

          s = nname_ + "_sum2p";
          void* m2 = (void*)m + nImg*ofm*sizeof(double);
          MeanOfLayer((char*)s.c_str(), (double*)m2, nImg*ofm);
        }
      }
    }
    else if(out_dtype == DT_BF16)
    {
      if(stptr == NULL)
      {
        int os = nImg*ofm*ofhp*ofwp;
        int is = nImg*ifm*ifhp*ifwp;
        int ws = ifm*ofm*kh*kw;
        int m = os < is ? is : os;
        int msize = m < ws ? ws : m;
        stptr = (float*)libxsmm_aligned_malloc(msize*sizeof(float), 2097152);
      }

      {
        string s = nname_ + "_Inp";
        libxsmm_bfloat16 *ptr;
        if(tenBotData_->getLPBuffer() != NULL)
          ptr = (libxsmm_bfloat16*)tenBotData_->getLPBuffer();
        else
          ptr = (libxsmm_bfloat16*)tenBotData_->getBuffer();
        convert_bf16_f32(ptr, stptr, nImg*ifm*ifhp*ifwp);
        MeanOfLayer((char*)s.c_str(), stptr, nImg*ifm*ifhp*ifwp);

        s = nname_ + "_Wt";
        float *fptr = (float*)tenWeightData_->getBuffer();
        int w = ifm*ofm*kh*kw;
        MeanOfLayer((char*)s.c_str(), fptr, w);

        if(gparams_.bias_term)
        {
          s = nname_ + "_Bias";
          float *p = (float*)tenBiasData_->getBuffer();
          MeanOfLayer((char*)s.c_str(), p, ofm);
        }

        s = nname_ + "_Outp";
        ptr = (libxsmm_bfloat16*)tenTopData_->getBuffer();
        memset(stptr, 0, nImg*ofm*ofhp*ofwp);
        convert_bf16_f32(ptr, stptr, nImg*ofm*ofhp*ofwp);
        MeanOfLayer((char*)s.c_str(), stptr, nImg*ofm*ofhp*ofwp);

        if(compute_stats_)
        {
          s = nname_ + "_sump";
          int offset = nImg*ofm*ofhp*ofwp*sizeof(float);
          void* m = (void*)ptr + offset;
          MeanOfLayer((char*)s.c_str(), (float*)m, nImg*ofm);

          s = nname_ + "_sum2p";
          void* m2 = (void*)m + nImg*ofm*sizeof(float);
          MeanOfLayer((char*)s.c_str(), (float*)m2, nImg*ofm);
        }
      }
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

  if(first_bp)
  {
    long long int size = nImg * ifm * ifhp *ifwp;

    if((in_dtype == DT_BF16 && out_dtype == DT_FLOAT)
        || (in_dtype == DT_FLOAT && out_dtype == DT_FLOAT))
    {
      float* ptr = (float*)tenBotDiff_->getBuffer();
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<size; i++)
        ptr[i] = 0;
    }
    else if(in_dtype == DT_BF16 && out_dtype == DT_BF16)
    {
      libxsmm_bfloat16* ptr = (libxsmm_bfloat16*)tenBotDiff_->getBuffer();

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<size; i++)
        ptr[i] = 0;
    }

   first_bp = false;
  }

  impl->backPropagate(tenTopData_, tenWeightData_, tenTopDiff_, tenBotDiff_);

#ifdef CHECK_BLOWUP_FP32
  if(out_dtype == DT_FLOAT)
  {
    for(int i=0; i<10240; i++)
    {
      float v = ((float*)tenBotDiff_->getBuffer())[i];
      if(isnan(v) || isinf(v))
      {
        printf("Warning! %s layer BP activations are NaN or Inf\n", nname_.c_str());
        exit(-1);
      }
    }
  }
  else if(out_dtype == DT_BF16)
  {
    convert_bf16_f32((libxsmm_bfloat16*)tenBotDiff_->getBuffer(), cbptr, 10240);
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
          MeanOfLayer((char*)((nname_+"_delin").c_str()), (libxsmm_bfloat16*)tenBotDiff_->getBuffer(), nImg*ifm*ifhp*ifwp);
          MeanOfLayer((char*)((nname_+"_delout").c_str()), (libxsmm_bfloat16*)tenTopDiff_->getBuffer(), nImg*ofm*ofhp*ofwp);
          MeanOfLayer((char*)((nname_+"_weight").c_str()), (libxsmm_bfloat16*)tenWeightData_->getLPBuffer(), ofm*ifm*kh*kw);
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
  unsigned int node_id_ = MLSL::Environment::GetEnv().GetProcessIdx();
  if(node_id_ == 0)
#endif
  {
    if(eptr_->get_current_batch() % STATFREQ == 0)
    {
      if(in_dtype == DT_FLOAT && out_dtype == DT_FLOAT)
      {
        string s = nname_ + "_delOutp";

        float *ptr = (float*)tenTopDiff_->getBuffer();
        MeanOfLayer((char*)s.c_str(), ptr, nImg*ofm*ofhp*ofwp);

        s = nname_ + "_Wt";
        ptr = (float*)tenWeightData_->getBuffer();
        MeanOfLayer((char*)s.c_str(), ptr, ifm*ofm*kh*kw);

        s = nname_ + "_delInp";
        ptr = (float*)tenBotDiff_->getBuffer();
        MeanOfLayer((char*)s.c_str(), ptr, nImg*ifm*ifhp*ifwp);
      }
      else if(in_dtype == DT_BF16 && out_dtype == DT_BF16)
      {
        string s = nname_ + "_delOutp";

        libxsmm_bfloat16 *ptr = (libxsmm_bfloat16*)tenTopDiff_->getBuffer();
        memset(stptr, 0, nImg*ofm*ofhp*ofwp);
        convert_bf16_f32(ptr, stptr, nImg*ofm*ofhp*ofwp);
        MeanOfLayer((char*)s.c_str(), stptr, nImg*ofm*ofhp*ofwp);

        s = nname_ + "_Wt";
        float *fptr = (float*)tenWeightData_->getBuffer();
        MeanOfLayer((char*)s.c_str(), fptr, ifm*ofm*kh*kw);

        s = nname_ + "_delInp";
        ptr = (libxsmm_bfloat16*)tenBotDiff_->getBuffer();
        memset(stptr, 0, nImg*ifm*ifhp*ifwp);
        convert_bf16_f32(ptr, stptr, nImg*ifm*ifhp*ifwp);
        MeanOfLayer((char*)s.c_str(), stptr, nImg*ifm*ifhp*ifwp);
      }
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
  int node_id = 0;
#ifdef USE_MLSL
  node_id = MLSL::Environment::GetEnv().GetProcessIdx();
  if(node_id == 0 && eptr_->get_current_batch() % STATFREQ == 0)
#else
  if(eptr_->get_current_batch() % STATFREQ == 0)
#endif
  {
    if(in_dtype == DT_FLOAT)
    {
      string s = nname_ + "_delWt_Bef";
      float *ptr = (float*)tenWeightDiff_->getBuffer();
      MeanOfLayer((char*)s.c_str(), ptr, ifm*ofm*kh*kw);
    }
    else if(in_dtype == DT_BF16)
    {
      string s = nname_ + "_delWt_Bef";
      libxsmm_bfloat16 *ptr = (libxsmm_bfloat16*)tenWeightDiff_->getBuffer();
      memset(stptr, 0, ifm*ofm*kh*kw);
      convert_bf16_f32(ptr, stptr, ifm*ofm*kh*kw);
      MeanOfLayer((char*)s.c_str(), stptr, ifm*ofm*kh*kw);
    }

    if(gparams_.bias_term)
    {
      string s = nname_ + "_delBias_Bef";
      float *p = (float*)tenBiasDiff_->getBuffer();
      MeanOfLayer((char*)s.c_str(), p, ofm);
    }
  }
#endif

  tenTopDiff_ = tenTop_->getBuf(DIFF);

  impl->weightUpdate(tenBotData_, tenTopDiff_, tenWeightDiff_, tenBiasDiff_);

#ifdef CHECK_BLOWUP_FP32
  if(out_dtype == DT_FLOAT)
  {
    for(int i=0; i<10240; i++)
    {
      float v = ((float*)tenWeightDiff_->getBuffer())[i];
      if(isnan(v) || isinf(v))
      {
        printf("Warning! %s layer weight-gradients are NaN or Inf\n", nname_.c_str());
        exit(-1);
      }
    }
  }
  else if(out_dtype == DT_BF16)
  {
#ifdef BF16_MLSL
    void **wptrptr = tenWeightDiff_->getBufferPtr();
#else
    void **wptrptr = tenWeightDiff_->getLPBufferPtr();
#endif
    int offset = tenWeightDiff_->getOffset();
    void* bf16_wtdiff = wptrptr[0] + offset*sizeof(libxsmm_bfloat16);

    convert_bf16_f32((libxsmm_bfloat16*)bf16_wtdiff, cbptr, 10240);
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
          printf("Warning! %s layer weight-gradients are NaN or Inf\n", nname_.c_str());
          MeanOfLayer((char*)nname_.c_str(), (libxsmm_bfloat16*)bf16_wtdiff, ofm*ifm*kw*kw);
          exit(-1);
        }
      }
    }
  }
#endif

#ifdef USE_MLSL
  void *mptr = tenWeightDiff_->getBuffer();

#ifndef BF16_MLSL
  void *lmptr = tenWeightDiff_->getLPBuffer();

  if(in_dtype == DT_BF16)
  {
    convert_bf16_f32((libxsmm_bfloat16*)lmptr, (float*)mptr, ifm*ofm*kh*kw);
    op_->GetParameterSet(0)->StartGradientComm(mptr);
  }
  else if(in_dtype == DT_FLOAT)
    op_->GetParameterSet(0)->StartGradientComm(mptr);
#else
  op_->GetParameterSet(0)->StartGradientComm(mptr);
#endif

  if(gparams_.bias_term)
    op_->GetParameterSet(1)->StartGradientComm(tenBiasDiff_->getBuffer());
#endif

#ifdef GETSTATS
#ifdef USE_MLSL
  node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
  node_id = 0;
#endif
  if(node_id == 0)
  {
    if(in_dtype == DT_FLOAT)
    {
      string s = nname_ + "_Inp";
      float *ptr = (float*)tenBotData_->getBuffer();
      MeanOfLayer((char*)s.c_str(), ptr, nImg*ifm*ifhp*ifwp);
      s = nname_ + "_delOutp";
      ptr = (float*)tenTopDiff_->getBuffer();
      MeanOfLayer((char*)s.c_str(), ptr, nImg*ofm*ofhp*ofwp);

      s = nname_ + "_delWt_Aft";
      ptr = (float*)tenWeightDiff_->getBuffer();
      float *pptr = (float*)tenWeightDiff_->getPrivBuffer();
      float *p = (pptr == NULL) ? ptr : pptr;
      MeanOfLayer((char*)s.c_str(), p, ifm*ofm*kh*kw);
    }
    else if(in_dtype == DT_BF16)
    {
      string s = nname_ + "_Inp";
      libxsmm_bfloat16 *ptr;
      if(tenBotData_->getLPBuffer() != NULL)
        ptr = (libxsmm_bfloat16*)tenBotData_->getLPBuffer();
      else
        ptr = (libxsmm_bfloat16*)tenBotData_->getBuffer();

      memset(stptr, 0, nImg*ifm*ifhp*ifwp);
      convert_bf16_f32(ptr, stptr, nImg*ifm*ifhp*ifwp);
      MeanOfLayer((char*)s.c_str(), stptr, nImg*ifm*ifhp*ifwp);

      s = nname_ + "_delOutp";
      ptr = (libxsmm_bfloat16*)tenTopDiff_->getBuffer();
      memset(stptr, 0, nImg*ofm*ofhp*ofwp);
      convert_bf16_f32(ptr, stptr, nImg*ofm*ofhp*ofwp);
      MeanOfLayer((char*)s.c_str(), stptr, nImg*ofm*ofhp*ofwp);

      s = nname_ + "_delWt_Aft";
      ptr = (libxsmm_bfloat16*)tenWeightDiff_->getBuffer();
      memset(stptr, 0, ifm*ofm*kh*kw);
      convert_bf16_f32(ptr, stptr, ifm*ofm*kh*kw);
      MeanOfLayer((char*)s.c_str(), stptr, ifm*ofm*kh*kw);
    }

    if(gparams_.bias_term)
    {
      string s = nname_ + "_delBias_Aft";
      float *p = (float*)tenBiasDiff_->getBuffer();
      MeanOfLayer((char*)s.c_str(), p, ofm);
    }
  }
#endif
}

void ConvNode::solverStep()
{
#ifdef USE_MLSL
  int ifm = gparams_.nInput;
  int ofm = gparams_.nOutput;
  int kh = gparams_.kh;
  int kw = gparams_.kw;

  void *gwt = tenWeightDiff_->getBuffer();

  float *gbias;
  if(gparams_.bias_term)
    gbias = (float*)(tenBiasDiff_->getBuffer());

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
  if(gparams_.bias_term)
  {
    mptr = op_->GetParameterSet(1)->WaitGradientComm();
    if(mptr != NULL && mptr != gbias)
      memcpy((void*)gbias, mptr, ofm*sizeof(float));
  }
#endif
}
