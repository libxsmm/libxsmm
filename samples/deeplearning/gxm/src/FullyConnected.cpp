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
#include "FullyConnected.hpp"
#include "fillers.hpp"
#ifdef USE_MLSL
#include "mpi.h"
#endif

FCNode::FCNode(FCParams *p, MLEngine* e) : NNNode(p, e)
{
  nname_ = p->get_node_name();
  ntype_ = p->get_node_type();
  mode_ = p->get_mode();
  bottom_ = p->get_bottom_names();
  top_ = p->get_top_names();
  bp_flag_ = p->get_bprop_flag();
  has_weights_ = true;

  tenTop_ = new Tensor(top_[0]);
  assert(tenTop_ != NULL);
  tenTop_->setOwner(this);
  tenTop_->setType(ACT);
  tenTopData_ = tenTop_->getBuf(DATA);
  tenTopData_->setBufferType(DATA);

#ifdef DEBUG
  printf("bottom name %s\n",bottom_[0].c_str());
#endif

  if((bottom_[0]).compare("data") == 0)
    tenBot_ = e->get_tensor(bottom_[0], INPUT);
  else
    tenBot_ = e->get_tensor(bottom_[0], ACT);
  assert(tenBot_ != NULL);

  NNNode *pnn = (NNNode*)tenBot_->getOwner();
  setPrevNode(pnn);
  mode_ = pnn->getMode();
  int cengine = p->get_compute_engine();
  pnn->set_top_compute_engine(cengine);
  bot_cengine_ = pnn->get_bot_compute_engine();

  tenBotData_ = tenBot_->getBuf(DATA);
  in_dtype = tenBotData_->getDataType();

  //Output tensor data type = input tensor data type
  out_dtype = p->get_data_type();
  tenTopData_->setDataType(out_dtype);
  tenTopData_->setBufferType(DATA);

  // Get input tensor shape (bottom)
  Shape* bs = tenBot_->getShape();
  assert(bs->ndims <= MAX_DIMS);

  shape_setzero(&ts_);

  ts_.ndims = 4;  // Number of dimensions
  ts_.dims[0] = bs->dims[0]; // Minibatch size
  ts_.dims[1] = p->get_output(); // Num output feature maps
  ts_.dims[2] = 1;
  ts_.dims[3] = 1;

  tenTop_->setShape(&ts_);

  long long int tsize = 1;
  for(int i=0; i<ts_.ndims; i++)
    tsize = tsize*ts_.dims[i];

  if(out_dtype == DT_FLOAT)
    tsize = tsize*sizeof(float);
  else if(out_dtype == DT_BF16)
    tsize = tsize*sizeof(libxsmm_bfloat16);

  tenTopData_->setBufferSize(tsize);

  gparams_.num_numa_nodes = NUM_NUMA_NODES;

  // Create weight tensor
  weight_ = top_[0] + "_fp_wt";
  tenWeight_ = new Tensor(weight_);
  assert(tenWeight_ != NULL);
  tenWeight_->setOwner(this);

  tenWeight_->setType(FCWEIGHT);

  shape_setzero(&ws_);

  ws_.ndims = ts_.ndims; // Number of dimensions
  if(p->get_transpose_flag())
  {
    ws_.dims[0] = bs->dims[1] * bs->dims[2] * bs->dims[3]; // Num input feature maps (from bottom tensor)
    ws_.dims[1] = ts_.dims[1]; // Num output feature maps (from top tensor)
    ws_.dims[2] = 1;
    ws_.dims[3] = 1;
  }
  else
  {
    ws_.dims[1] = bs->dims[1] * bs->dims[2] * bs->dims[3];   // Num input feature maps (from bottom tensor)
    ws_.dims[0] = ts_.dims[1];             // Num output feature maps (from top tensor)
    ws_.dims[2] = 1;
    ws_.dims[3] = 1;
  }

  tenWeight_->setShape(&ws_);
  tenWeight_->setBufDataType(DATA, DT_FLOAT);
  tenWeightData_ = tenWeight_->getBuf(DATA);
  tenWeightData_->setBufferType(DATA);

  long long int wsize = 1;
  for(int i=0; i<ws_.ndims; i++)
    wsize = wsize*ws_.dims[i];

  wsize = wsize*sizeof(float);
  tenWeightData_->setBufferSize(wsize);

  wfiller_type_ = p->get_weight_filler_type();
  std_ = p->get_std();

  lr_mult_ = p->get_lr_mult();
  decay_mult_ = p->get_decay_mult();

  // Create bias tensor
  Shape bis;
  if(p->get_bias_term())
  {
    bias_ = top_[0] + "_fp_bias";
    tenBias_ = new Tensor(bias_);
    assert(tenBias_ != NULL);
    tenBias_->setOwner(this);

    tenBias_->setType(FCBIAS);

    shape_setzero(&bis);

    bis.ndims = 1;
    bis.dims[0] = ts_.dims[1];
    tenBias_->setShape(&bis);
    tenBias_->setBufDataType(DATA, DT_FLOAT);
    tenBiasData_ = tenBias_->getBuf(DATA);
    tenBiasData_->setBufferType(DATA);

    long long int bisize = bis.dims[0];
    bisize = bisize*sizeof(float);
    tenBiasData_->setBufferSize(bisize);

    bfiller_type_ = p->get_bias_filler_type();
    value_ = p->get_value();
  }

  if(!e->is_inference_only()) {
    if(bp_flag_)
    {
      tenBotDiff_ = tenBot_->addBuf();
      tenBotDiff_->setDataType(in_dtype);
      tenBotDiff_->setBufferType(DIFF);

      long long int bsize = 1;
      for(int i=0; i<bs->ndims; i++)
        bsize = bsize*bs->dims[i];

      if(in_dtype == DT_FLOAT)
        bsize = bsize*sizeof(float);
      else if(in_dtype == DT_BF16)
        bsize = bsize*sizeof(libxsmm_bfloat16);

      // Set the size of the input-gradient buffer
      tenBotDiff_->setBufferSize(bsize);
    }

    if(has_weights_)
    {
      tenWeightDiff_ = tenWeight_->addBuf();
      if(in_dtype == DT_BF16 || out_dtype == DT_BF16)
      {
        tenWeightDiff_->setDataType(DT_BF16);
        int welem = ws_.dims[0]*ws_.dims[1];
#ifdef BF16_MLSL
        tenWeightDiff_->setBufferSize(welem*sizeof(libxsmm_bfloat16));
#else
        tenWeightDiff_->setBufferSize(welem*sizeof(float));
#endif
      }
      else
      {
        tenWeightDiff_->setDataType(DT_FLOAT);
        tenWeightDiff_->setBufferSize(wsize);
      }
      tenWeightDiff_->setBufferType(DIFF);

      tenWeightInc_ = tenWeight_->addBuf();
      tenWeightInc_->setDataType(DT_FLOAT);
      tenWeightInc_->setBufferType(HISTORY);

      // Set the size of weight-increment buffer
      tenWeightInc_->setBufferSize(wsize);

      if(p->get_bias_term())
      {
        tenBiasDiff_ = tenBias_->addBuf(); // DIFF type and index
        tenBiasDiff_->setDataType(DT_FLOAT);
        tenBiasDiff_->setBufferType(DIFF);

        tenBiasInc_ = tenBias_->addBuf(); // SHARED type and index
        tenBiasInc_->setDataType(DT_FLOAT);
        tenBiasInc_->setBufferType(HISTORY);

        // Set the size of the weight-gradient buffer and the weight-increment buffer
        long long int bisize = bis.dims[0];
        bisize = bisize*sizeof(float);

        tenBiasDiff_->setBufferSize(bisize);
        tenBiasInc_->setBufferSize(bisize);
      }
    }
  }
  else {
    tenBotDiff_ = NULL;
    tenWeightDiff_ = NULL;
    tenWeightInc_ = NULL;
  }

  // Register output tensor in tensor Map
  bool inserted = e->register_tensor(top_[0], ACT, tenTop_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",top_[0].c_str());

  // Register weight tensor in tensor Map
  inserted = e->register_tensor(weight_, FCWEIGHT, tenWeight_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",weight_.c_str());

  // Register bias tensor in tensor Map
  if(p->get_bias_term())
  {
    inserted = e->register_tensor(bias_, FCBIAS, tenBias_);
    if(!inserted)
      printf("Warning: Tensor %s already registered\n",bias_.c_str());
  }

  // Setup parameter structure for computation in library

  gparams_.node_name = nname_;
  gparams_.batch_size = bs->dims[0];
  gparams_.nInput = bs->dims[1];
  gparams_.nOutput = ts_.dims[1];
  gparams_.iHeight = bs->dims[2];
  gparams_.iWidth = bs->dims[3];
  gparams_.oHeight = ts_.dims[2];
  gparams_.oWidth = ts_.dims[3];
  gparams_.kh = 1;
  gparams_.kw = 1;

  gparams_.bias_term = p->get_bias_term();

  gparams_.in_data_type = in_dtype;
  gparams_.out_data_type = out_dtype;
  gparams_.algType = p->get_algo_type();
  gparams_.num_threads = e->get_num_threads();

  // get solver
  solver_ = e->getSolver();

  //get global scratch tensor buffer
  tenScratchData_ = e->getScratchBuffer();

  //get engine
  eptr_ = e;

#ifdef USE_MLSL
  MLSL::DataType dt = MLSL::DT_FLOAT;
  MLSL::OperationRegInfo *myRegInfo;
  MLSL::Session *s = eptr_->get_session();
  myRegInfo = s->CreateOperationRegInfo(MLSL::OT_CC);
  myRegInfo->SetName(nname_.c_str());
  myRegInfo->AddParameterSet(gparams_.nInput*gparams_.nOutput, gparams_.kh*gparams_.kw, dt, false);

  if (gparams_.bias_term) {
    myRegInfo->AddParameterSet(gparams_.nOutput, 1, dt, false);
  }

  myRegInfo->Validate();
  size_t opIdx = s->AddOperation(myRegInfo, e->get_distribution());
  this->op_ = s->GetOperation(opIdx);
  s->DeleteOperationRegInfo(myRegInfo);
  e->get_wtgrad_comms_vec().push_back(op_);
#endif
  configure(p->get_compute_engine());
}

void FCNode::fillWeightBuffers(TensorBuf* tBuf, int buftype, long long int size)
{
  int dtype = tBuf->getBufferType();
  void *ptr = tBuf->getBuffer();

#ifdef USE_MLSL
  unsigned int node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
  unsigned int node_id = 0;
#endif

  int ic = gparams_.nInput;
  int oc = gparams_.nOutput;
  int welem = ic * oc;
  int fanin = ic;
  int fanout = oc;

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

void FCNode::fillWeightMultipliers(float* lr, float* decay, long long int size)
{
  for(int i=0; i < size; i++)
  {
    lr[i] = lr_mult_[0];
    decay[i] = decay_mult_[0];
  }
}

void FCNode::fillBiasBuffers(TensorBuf* tBuf, int buftype, long long int size)
{
  int dtype = tBuf->getBufferType();
  void *ptr = tBuf->getBuffer();

  if(buftype == DATA)
  {
    assert(bfiller_type_.compare("constant") == 0);
    initConstantBuffer(ptr, size, "CONSTANT", value_);
  }
  else
    memset(ptr, 0, size);
}

void FCNode::fillBiasMultipliers(float* lr, float* decay, long long int size)
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

void FCNode::Checkpoint(TensorBuf *tBuf, string name, string format)
{
  long long int bytes = tBuf->getBufferSize();
  int dtype = tBuf->getBufferType();

  void* ptr = tBuf->getBuffer();

  FILE* f;
  size_t pos;

  if((name.find("30") == name.npos) && (name.find("60") == name.npos) && (name.find("80") == name.npos))
    while((pos = name.find("/", 10)) != name.npos)
      name.replace(pos, 1, 1, '_');

  float* p = (float*)ptr;
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
        if(dtype == DT_FLOAT)
        {
          for(int i=0; i<bytes/sizeof(float); i++)
            fprintf(f, "%f\n", *((float*)ptr + i));
        }
        else if(dtype == DT_BF16)
        {
          for(int i=0; i<bytes/sizeof(libxsmm_bfloat16); i++)
            fprintf(f, "%d\n", *((libxsmm_bfloat16*)ptr + i));
        }
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

void FCNode::configure(int engine)
{
  switch(engine)
  {
    case XSMM:
      impl = new FCXSMM(&gparams_, engine);
      break;
  }
}

void FCNode::convert_f32_bf16(float* in, libxsmm_bfloat16* out, int len)
{
  int i = 0;

#ifdef _OPENMP
#pragma omp parallel for private(i)
#endif
  for ( i = 0; i < len; i+=16 ) {
    __m512  vfp32  = gxm_fp32_to_bfp16_rne_adjustment_avx512f(_mm512_loadu_ps(in + i));
    __m256i vbfp16 = gxm_fp32_to_bfp16_truncate_avx512f(vfp32);
    _mm256_storeu_si256( (__m256i*)(out+i), vbfp16 );
  }
}

void FCNode::convert_bf16_f32(libxsmm_bfloat16* in, float* out, int len)
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

void FCNode::forwardPropagate()
{

  int nImg = gparams_.batch_size;
  int ifm = gparams_.nInput;
  int ofm = gparams_.nOutput;
  int kh = gparams_.kh;
  int kw = gparams_.kw;

#ifdef DEBUG
  void* bot = (void*)(tenBotData_->getBuffer());
  void* wt = (void*)(tenWeightData_->getBuffer());
  void* bias;
  if(gparams_.bias_term)
    bias = (void*)(tenBiasData_->getBuffer());
  void* top = (void*)(tenTopData_->getBuffer());

  printf("Executing FP %s: input %p, weights %p, output %p\n",NNNode::nname_.c_str(), bot, wt, top);
  fflush(NULL);
  printf("Inputs: %d x %d\n",gparams_.batch_size, gparams_.nInput*gparams_.iHeight*gparams_.iWidth);
  printf("Outputs: %d x %d\n",gparams_.batch_size, gparams_.nOutput*gparams_.oHeight*gparams_.oWidth);
  printf("Weights: %d x %d x %d x %d\n", gparams_.nInput, gparams_.nOutput, gparams_.kw, gparams_.kw);
#endif

  impl->set_bot_compute_engine(bot_cengine_);
  impl->set_top_compute_engine(top_compute_engine_);
  impl->set_node_name(nname_);
  impl->set_scratch_buffer(tenScratchData_);

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
    convert_bf16_f32((libxsmm_bfloat16*)tenTopData_->getBuffer(), cbptr, 16);
    for(int i=0; i<16; i++)
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
  if(node_id == 0 && eptr_->get_current_batch() % STATFREQ == 0)
#else
  if(eptr_->get_current_batch() % STATFREQ == 0)
#endif
  {
    if(in_dtype == DT_FLOAT)
    {
      string s = nname_ + "_Inp";
      float *ptr = (float*)tenBotData_->getBuffer();
      MeanOfLayer((char*)s.c_str(), ptr, gparams_.batch_size*gparams_.nInput);
    }
    else if(in_dtype == DT_BF16)
    {
      if(stptr == NULL)
      {
        int os = nImg*ofm;
        int is = nImg*ifm;
        int ws = ifm*ofm;
        int m = os < is ? is : os;
        int msize = m < ws ? ws : m;
        stptr = (float*)libxsmm_aligned_malloc(msize*sizeof(float), 2097152);
      }
      string s = nname_ + "_Inp";
      libxsmm_bfloat16 *ptr = (libxsmm_bfloat16*)tenBotData_->getBuffer();
      convert_bf16_f32(ptr, stptr, gparams_.batch_size*gparams_.nInput);
      MeanOfLayer((char*)s.c_str(), stptr, gparams_.batch_size*gparams_.nInput);
    }

    string  s = nname_ + "_Wt";
    float *ptr = (float*)tenWeightData_->getBuffer();
    MeanOfLayer((char*)s.c_str(), ptr, gparams_.nInput*gparams_.nOutput);

    if(out_dtype == DT_FLOAT)
    {
      string s = nname_ + "_Outp";
      float *ptr = (float*)tenTopData_->getBuffer();
      MeanOfLayer((char*)s.c_str(), ptr, gparams_.batch_size*gparams_.nOutput);
    }
    else if(out_dtype == DT_BF16)
    {
      string s = nname_ + "_Outp";
      libxsmm_bfloat16 *ptr = (libxsmm_bfloat16*)tenTopData_->getBuffer();
      convert_bf16_f32(ptr, stptr, gparams_.batch_size*gparams_.nOutput);
      MeanOfLayer((char*)s.c_str(), stptr, gparams_.batch_size*gparams_.nOutput);
    }

    if(gparams_.bias_term)
    {
      string s = nname_ + "_Bias";
      float *ptr = (float*)tenBiasData_->getBuffer();
      MeanOfLayer((char*)s.c_str(), ptr, gparams_.nOutput);
    }
  }
#endif
}

void FCNode::backPropagate()
{
  tenTopDiff_ = tenTop_->getBuf(DIFF);

#ifdef DEBUG
  void* top = (void*)(tenTopData_->getBuffer());
  void *gtop = (void*)(tenTopDiff_->getBuffer());
  assert(gtop != NULL);

  void* wt = (void*)(tenWeightData_->getBuffer());
  void* gbot = (void*)(tenBotDiff_->getBuffer());

  printf("Executing BP %s: grad_output %p, weights %p, grad_input %p\n",NNNode::nname_.c_str(), gtop, wt, gbot);
  printf("Grad Outputs: %d x %d\n", gparams_.batch_size, gparams_.nOutput);
  printf("Grad Inputs: %d x %d\n", gparams_.batch_size, gparams_.nInput);
  printf("Weights: %d x %d x %d x %d\n", gparams_.nOutput, gparams_.nInput, gparams_.kh, gparams_.kw);
#endif

  impl->backPropagate(tenTopDiff_, tenWeightData_, tenBotDiff_);

#ifdef CHECK_BLOWUP_FP32
  if(out_dtype == DT_FLOAT)
  {
    for(int i=0; i<16; i++)
    {
      float v = ((float*)tenBotDiff_->getBuffer())[i];
      if(isnan(v) || isinf(v))
      {
        printf("Warning! %s layer FP activations are NaN or Inf\n", nname_.c_str());
        exit(-1);
      }
    }
  }
  else if(out_dtype == DT_BF16)
  {
    convert_bf16_f32((libxsmm_bfloat16*)tenBotDiff_->getBuffer(), cbptr, 16);
    for(int i=0; i<16; i++)
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
  if(node_id == 0 && eptr_->get_current_batch() % STATFREQ == 0)
#else
  if(eptr_->get_current_batch() % STATFREQ == 0)
#endif
  {
    if(out_dtype == DT_FLOAT)
    {
      string s = nname_ + "_delOutp";
      float *ptr = (float*)tenTopDiff_->getBuffer();
      MeanOfLayer((char*)s.c_str(), ptr, gparams_.batch_size*gparams_.nOutput);
    }
    else if(out_dtype == DT_BF16)
    {
      string s = nname_ + "_delOutp";
      libxsmm_bfloat16 *ptr = (libxsmm_bfloat16*)tenTopDiff_->getBuffer();
      convert_bf16_f32(ptr, stptr, gparams_.batch_size*gparams_.nOutput);
      MeanOfLayer((char*)s.c_str(), stptr, gparams_.batch_size*gparams_.nOutput);
    }

    string s = nname_ + "_Wt";
    float *ptr = (float*)tenWeightData_->getBuffer();
    MeanOfLayer((char*)s.c_str(), ptr, gparams_.nInput*gparams_.nOutput);

    if(in_dtype == DT_FLOAT)
    {
      string s = nname_ + "_delInp";
      float *ptr = (float*)tenBotDiff_->getBuffer();
      MeanOfLayer((char*)s.c_str(), ptr, gparams_.batch_size*gparams_.nInput);
    }
    else if(in_dtype == DT_BF16)
    {
      string s = nname_ + "_delInp";
      libxsmm_bfloat16 *ptr = (libxsmm_bfloat16*)tenBotDiff_->getBuffer();
      convert_bf16_f32(ptr, stptr, gparams_.batch_size*gparams_.nInput);
      MeanOfLayer((char*)s.c_str(), stptr, gparams_.batch_size*gparams_.nInput);
    }
  }
#endif
}

void FCNode::weightUpdate()
{
  tenTopDiff_ = tenTop_->getBuf(DIFF);

#ifdef DEBUG
  void *gtop = (void*)(tenTopDiff_->getBuffer());

  assert(gtop != NULL);
  void* bot = (void*)(tenBotData_->getBuffer());
  void* gwt = (void*)(tenWeightDiff_->getBuffer());
  void* gbias;
  if(gparams_.bias_term)
    gbias = (void*)(tenBiasDiff_->getBuffer());

  printf("Executing WU %s: grad_output %p, grad_weights %p, grad_biases %p, input %p\n",NNNode::nname_.c_str(), gtop, gwt, gbias, bot);
  printf("Grad Outputs: %d x %d\n", gparams_.batch_size, gparams_.nOutput);
  printf("Inputs: %d x %d\n", gparams_.batch_size, gparams_.nInput);
  printf("Grad Weights: %d x %d x %d x %d\n", gparams_.nOutput, gparams_.nInput, gparams_.kh, gparams_.kw);
  printf("Grad Biases: %d\n", gparams_.nOutput);
#endif

  impl->weightUpdate(tenTopDiff_, tenBotData_, tenWeightDiff_, tenBiasDiff_);

#ifdef CHECK_BLOWUP_FP32
  if(out_dtype == DT_FLOAT)
  {
    for(int i=0; i<16; i++)
    {
      float v = ((float*)tenWeightDiff_->getBuffer())[i];
      if(isnan(v) || isinf(v))
      {
        printf("Warning! %s layer FP activations are NaN or Inf\n", nname_.c_str());
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
        printf("Warning! %s layer FP activations are NaN or Inf\n", nname_.c_str());
        exit(-1);
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
    convert_bf16_f32((libxsmm_bfloat16*)lmptr, (float*)mptr, gparams_.nInput*gparams_.nOutput);
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
}

void FCNode::solverStep()
{
#ifdef RETURNALL
  return;
#endif

  void *gwt = tenWeightDiff_->getBuffer();

  void *gbias;
  if(gparams_.bias_term)
    gbias = (void*)(tenBiasDiff_->getBuffer());

  int wsize = gparams_.nInput*gparams_.nOutput;

#ifdef USE_MLSL
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
      memcpy((void*)gbias, mptr, gparams_.nOutput*sizeof(float));
  }
#endif
}

