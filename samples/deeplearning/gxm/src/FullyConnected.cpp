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

  //Output tensor data type = input tensor data type
  int dtype = p->get_data_type();
  tenTopData_->setDataType(dtype);
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

  if(dtype == DT_FLOAT)
    tsize = tsize*sizeof(float);
  else if(dtype == DT_DFP16)
    tsize = tsize*sizeof(short int);

  tenTopData_->setBufferSize(tsize);

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
  tenWeight_->setBufDataType(DATA, dtype);
  tenWeightData_ = tenWeight_->getBuf(DATA);
  tenWeightData_->setBufferType(DATA);

  long long int wsize = 1;
  for(int i=0; i<ws_.ndims; i++)
    wsize = wsize*ws_.dims[i];

  if(dtype == DT_FLOAT)
    wsize = wsize*sizeof(float);
  else if(dtype == DT_DFP16)
    wsize = wsize*sizeof(short int);

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
    tenBias_->setBufDataType(DATA, dtype);
    tenBiasData_ = tenBias_->getBuf(DATA);
    tenBiasData_->setBufferType(DATA);

    long long int bisize = bis.dims[0];
    if(dtype == DT_FLOAT)
      bisize = bisize*sizeof(float);
    else if(dtype == DT_DFP16)
      bisize = bisize*sizeof(short int);
    tenBiasData_->setBufferSize(bisize);

    bfiller_type_ = p->get_bias_filler_type();
    value_ = p->get_value();
  }

  if(!e->is_inference_only()) {
    if(bp_flag_)
    {
      tenBotDiff_ = tenBot_->addBuf();
      tenBotDiff_->setDataType(dtype);
      tenBotDiff_->setBufferType(DIFF);

      long long int bsize = 1;
      for(int i=0; i<bs->ndims; i++)
        bsize = bsize*bs->dims[i];

      if(dtype == DT_FLOAT)
        bsize = bsize*sizeof(float);
      else if(dtype == DT_DFP16)
        bsize = bsize*sizeof(short int);

      // Set the size of the input-gradient buffer
      tenBotDiff_->setBufferSize(bsize);
    }

    if(has_weights_)
    {
      tenWeightDiff_ = tenWeight_->addBuf();
      tenWeightDiff_->setDataType(dtype);
      tenWeightDiff_->setBufferType(DIFF);

      tenWeightInc_ = tenWeight_->addBuf();
      tenWeightInc_->setDataType(dtype);
      tenWeightInc_->setBufferType(HISTORY);

      // Set the size of the weight-gradient buffer and the weight-increment buffer
      tenWeightDiff_->setBufferSize(wsize);
      tenWeightInc_->setBufferSize(wsize);

      if(p->get_bias_term())
      {
        tenBiasDiff_ = tenBias_->addBuf(); // DIFF type and index
        tenBiasDiff_->setDataType(dtype);
        tenBiasDiff_->setBufferType(DIFF);

        tenBiasInc_ = tenBias_->addBuf(); // SHARED type and index
        tenBiasInc_->setDataType(dtype);
        tenBiasInc_->setBufferType(HISTORY);

        // Set the size of the weight-gradient buffer and the weight-increment buffer
        long long int bisize = bis.dims[0];
        if(dtype == DT_FLOAT)
          bisize = bisize*sizeof(float);
        else if(dtype == DT_DFP16)
          bisize = bisize*sizeof(short int);

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

  gparams_.data_type = dtype;
  gparams_.algType = p->get_algo_type();
  gparams_.num_threads = e->get_num_threads();

  // get solver
  solver_ = e->getSolver();

  //get engine
  eptr_ = e;

#ifdef USE_MLSL
  MLSL::DataType dt = MLSL::DT_FLOAT;
  MLSL::OperationRegInfo *myRegInfo;
  MLSL::Session *s = eptr_->get_session();
  myRegInfo = s->CreateOperationRegInfo(MLSL::OT_CC);
  myRegInfo->SetName(nname_.c_str());
  myRegInfo->AddInput(gparams_.nInput, gparams_.iHeight*gparams_.iWidth, dt);
  myRegInfo->AddOutput(gparams_.nOutput, gparams_.oHeight*gparams_.oWidth, dt);
  myRegInfo->AddParameterSet(gparams_.nInput*gparams_.nOutput, gparams_.kh*gparams_.kw, dt, false);

  if (gparams_.bias_term) {
    myRegInfo->AddParameterSet(gparams_.nOutput, 1, dt, false);
  }

  myRegInfo->Validate();
  size_t opIdx = s->AddOperation(myRegInfo, e->get_distribution());
  this->op_ = s->GetOperation(opIdx);
  s->DeleteOperationRegInfo(myRegInfo);
#endif
  configure(p->get_compute_engine());
}

void FCNode::fillWeightBuffers(TensorBuf* tBuf, int buftype, long long int size, int machines)
{
  int dtype = tBuf->getBufferType();
  void *ptr = tBuf->getBuffer();

  int fanin = gparams_.nInput * gparams_.iHeight * gparams_.iWidth;
  int fanout = gparams_.batch_size;

#ifdef USE_MLSL
  unsigned int node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
  unsigned int node_id = 0;
#endif

  if(buftype == DATA)
  {
      initBuffer(ptr, dtype, variance_norm_, fanin, fanout, size, wfiller_type_, (unsigned int)node_id+PRIME_SEED, std_);

#ifdef USE_MLSL
    MPI_Bcast(ptr, size, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
  }
  else
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
    initConstantBuffer(ptr, dtype, size, "CONSTANT", value_);
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
        else if(dtype == DT_DFP16)
        {
          for(int i=0; i<bytes/sizeof(short int); i++)
            fprintf(f, "%d\n", *((short int*)ptr + i));
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

void FCNode::forwardPropagate()
{

#ifdef GETSTATS
  float* bot = (float*)(tenBotData_->getBuffer());
  float* wt = (float*)(tenWeightData_->getBuffer());
  float* bias;
  if(gparams_.bias_term)
    bias = (float*)(tenBiasData_->getBuffer());
  float* top = (float*)(tenTopData_->getBuffer());
#ifdef DEBUG

  printf("Executing FP %s: input %p, weights %p, output %p\n",NNNode::nname_.c_str(), bot, wt, top);
  fflush(NULL);
  printf("Inputs: %d x %d\n",gparams_.batch_size, gparams_.nInput*gparams_.iHeight*gparams_.iWidth);
  printf("Outputs: %d x %d\n",gparams_.batch_size, gparams_.nOutput*gparams_.oHeight*gparams_.oWidth);
  printf("Weights: %d x %d x %d x %d\n", gparams_.nInput*gparams_.iHeight*gparams_.iWidth, gparams_.nOutput, gparams_.kw, gparams_.kw);

#endif
#endif

  impl->set_bot_compute_engine(bot_cengine_);
  impl->set_top_compute_engine(top_compute_engine_);

  int size = gparams_.batch_size * gparams_.nOutput * gparams_.oHeight * gparams_.oWidth;
  float *out = (float*)(tenTopData_->getBuffer());

#pragma omp parallel for
  for(int i=0; i<size; i++)
    out[i] = 0.0;

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
  if(node_id == 0 && eptr_->get_current_batch() % STATFREQ == 0)
#else
  if(eptr_->get_current_batch() % STATFREQ == 0)
#endif
  {
    string s = nname_ + "_Inp";
    MeanOfLayer((char*)s.c_str(), bot, gparams_.batch_size*gparams_.nInput);
    s = nname_ + "_Wt";
    MeanOfLayer((char*)s.c_str(), wt, gparams_.nInput*gparams_.iHeight*gparams_.iWidth*gparams_.nOutput*gparams_.kh*gparams_.kw);
    s = nname_ + "_Outp";
    MeanOfLayer((char*)s.c_str(), top, gparams_.batch_size*gparams_.nOutput);

    if(gparams_.bias_term)
    {
      s = nname_ + "_Bias";
      MeanOfLayer((char*)s.c_str(), bias, gparams_.nOutput);
    }
  }
#endif
}

void FCNode::backPropagate()
{
  tenTopDiff_ = tenTop_->getBuf(DIFF);

#ifdef GETSTATS
  float* top = (float*)(tenTopData_->getBuffer());
  float *gtop = (float*)(tenTopDiff_->getBuffer());
  assert(gtop != NULL);

  float* wt = (float*)(tenWeightData_->getBuffer());
  float* gbot = (float*)(tenBotDiff_->getBuffer());

#ifdef DEBUG
  {
    printf("Executing BP %s: grad_output %p, weights %p, grad_input %p\n",NNNode::nname_.c_str(), gtop, wt, gbot);
    printf("Grad Outputs: %d x %d\n", gparams_.batch_size, gparams_.nOutput*gparams_.oHeight*gparams_.oWidth);
    printf("Grad Inputs: %d x %d\n", gparams_.batch_size, gparams_.nInput* gparams_.iHeight * gparams_.iWidth);
    printf("Weights: %d x %d x %d x %d\n", gparams_.nOutput, gparams_.nInput*gparams_.iHeight*gparams_.iWidth, gparams_.kh, gparams_.kw);
  }
#endif
#endif

  int size = gparams_.batch_size * gparams_.nInput * gparams_.iHeight * gparams_.iWidth;

  float *delinp = (float*)(tenBotDiff_->getBuffer());

#pragma omp parallel for
  for(int i=0; i<size; i++)
    delinp[i] = 0.0;

  impl->backPropagate(tenTopDiff_, tenWeightData_, tenBotDiff_);

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
  if(node_id == 0 && eptr_->get_current_batch() % STATFREQ == 0)
#else
  if(eptr_->get_current_batch() % STATFREQ == 0)
#endif
  {
    string s = nname_ + "_Inp";
    MeanOfLayer((char*)s.c_str(), top, gparams_.batch_size*gparams_.nOutput * gparams_.oHeight * gparams_.oWidth);
    s = nname_ + "_delOutp";
    MeanOfLayer((char*)s.c_str(), gtop, gparams_.batch_size*gparams_.nOutput* gparams_.oHeight * gparams_.oWidth);
    s = nname_ + "_Wt";
    MeanOfLayer((char*)s.c_str(), wt, gparams_.nInput*gparams_.iHeight*gparams_.iWidth*gparams_.nOutput*gparams_.kh*gparams_.kw);
    s = nname_ + "_delInp";
    MeanOfLayer((char*)s.c_str(), gbot, gparams_.batch_size*gparams_.nInput*gparams_.iHeight * gparams_.iWidth);
  }
#endif
}

void FCNode::weightUpdate()
{
  tenTopDiff_ = tenTop_->getBuf(DIFF);

#ifdef GETSTATS
  float *gtop = (float*)(tenTopDiff_->getBuffer());

  assert(gtop != NULL);
  float* bot = (float*)(tenBotData_->getBuffer());
  float* gwt = (float*)(tenWeightDiff_->getBuffer());
  float* gbias;
  if(gparams_.bias_term)
    gbias = (float*)(tenBiasDiff_->getBuffer());
#ifdef DEBUG

  printf("Executing WU %s: grad_output %p, grad_weights %p, grad_biases %p, input %p\n",NNNode::nname_.c_str(), gtop, gwt, gbias, bot);
  printf("Grad Outputs: %d x %d\n", gparams_.batch_size, gparams_.nOutput);
  printf("Inputs: %d x %d\n", gparams_.batch_size, gparams_.nInput*gparams_.iHeight*gparams_.iWidth);
  printf("Grad Weights: %d x %d x %d x %d\n", gparams_.nOutput, gparams_.nInput*gparams_.iHeight*gparams_.iWidth, gparams_.kh, gparams_.kw);
  printf("Grad Biases: %d\n", gparams_.nOutput);
#endif
#endif

  int size = gparams_.nOutput * gparams_.nInput * gparams_.iHeight*gparams_.iWidth*gparams_.kh * gparams_.kw;
  float* delwt_ptr = (float*)(tenWeightDiff_->getBuffer());
  float* delbias_ptr;
  if(gparams_.bias_term)
    delbias_ptr = (float*)(tenBiasDiff_->getBuffer());

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0; i<size; i++)
    delwt_ptr[i] = 0.0;

  if(gparams_.bias_term)
  {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<gparams_.nOutput; i++)
      delbias_ptr[i] = 0.0;
  }

  impl->weightUpdate(tenTopDiff_, tenBotData_, tenWeightDiff_, tenBiasDiff_);

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
#ifdef USE_MLSL
  unsigned int node_id = MLSL::Environment::GetEnv().GetProcessIdx();
  if(node_id == 0 && eptr_->get_current_batch() % STATFREQ == 0)
#else
  if(eptr_->get_current_batch() % STATFREQ == 0)
#endif
  {
    string s = nname_ + "_Inp";
    MeanOfLayer((char*)s.c_str(), bot, gparams_.batch_size*gparams_.nInput*gparams_.iHeight*gparams_.iWidth);
    s = nname_ + "_delOutp";
    MeanOfLayer((char*)s.c_str(), gtop, gparams_.batch_size*gparams_.nOutput);
    s = nname_ + "_Wt";
    MeanOfLayer((char*)s.c_str(), gwt, gparams_.nInput*gparams_.iHeight*gparams_.iWidth*gparams_.nOutput*gparams_.kh*gparams_.kw);
    if(gparams_.bias_term)
    {
      s = nname_ + "_delBias";
      MeanOfLayer((char*)s.c_str(), gbias, gparams_.nOutput);
    }
  }
#endif
}

void FCNode::solverStep()
{
#ifdef RETURNALL
  return;
#endif

  float *wt = (float*)(tenWeightData_->getBuffer());
  float *gwt = (float*)(tenWeightDiff_->getBuffer());
  float *iwt = (float*)(tenWeightInc_->getBuffer());

  float *bias, *gbias, *ibias;
  if(gparams_.bias_term)
  {
    bias = (float*)(tenBiasData_->getBuffer());
    gbias = (float*)(tenBiasDiff_->getBuffer());
    ibias = (float*)(tenBiasInc_->getBuffer());
  }
#ifdef DEBUG
  printf("Executing Solver: weights %p, grad_weights %p, bias %p, grad_bias %p\n", wt, gwt, bias, gbias);
  printf("Grad Weights: %d x %d\n", gparams_.nOutput, gparams_.nInput*gparams_.iHeight*gparams_.iWidth);
  printf("Grad Biases: %d\n",gparams_.nOutput);
#endif

  int wsize = gparams_.nInput*gparams_.iHeight*gparams_.iWidth*gparams_.kw*gparams_.kh*gparams_.nOutput;

  int num_nodes = 1;

#ifdef USE_MLSL
  void *mptr = op_->GetParameterSet(0)->WaitGradientComm();
  if(mptr != NULL && mptr != gwt)
    memcpy((void*)gwt, mptr, wsize*sizeof(float));

  if(gparams_.bias_term)
  {
    mptr = op_->GetParameterSet(1)->WaitGradientComm();
    if(mptr != NULL && mptr != gbias)
      memcpy((void*)gbias, mptr, gparams_.nOutput*sizeof(float));
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
  int iter = eptr->get_current_batch();
  FILE *f;
  string fname;

  {
    fname = gparams_.node_name + "_solver_delwt_" + to_string(iter);
    f = fopen(fname.c_str(), "w");
    for(int i=0; i<wsize; i++)
      fprintf(f, "%g\n", gwt[i]);
    fclose(f);

    if(gparams_.bias_term)
    {
      fname = gparams_.node_name + "_solver_delbias_" + to_string(iter);
      f = fopen(fname.c_str(), "w");
      for(int i=0; i<gparams_.nOutput; i++)
        fprintf(f, "%g\n", gbias[i]);
      fclose(f);
    }

    fname = gparams_.node_name + "_solver_wt_" + to_string(iter);
    f = fopen(fname.c_str(), "w");
    for(int i=0; i<wsize; i++)
      fprintf(f, "%g\n", wt[i]);
    fclose(f);

    fname = gparams_.node_name + "_solver_bias_" + to_string(iter);
    f = fopen(fname.c_str(), "w");
    for(int i=0; i<gparams_.nOutput; i++)
      fprintf(f, "%g\n", bias[i]);
    fclose(f);
  }
#endif

  solver_->applyUpdate(wt, iwt, gwt, wsize, lr_mult_[0], decay_mult_[0]);
  if(gparams_.bias_term)
    solver_->applyUpdate(bias, ibias, gbias, gparams_.nOutput, lr_mult_[1], decay_mult_[1]);

#ifdef DUMP_WT_DATA
  {
    fname = gparams_.node_name + "_solver_newwt_" + to_string(iter);
    f = fopen(fname.c_str(), "w");
    for(int i=0; i<wsize; i++)
      fprintf(f, "%g\n", wt[i]);
    fclose(f);


    fname = gparams_.node_name + "_solver_wtinc_" + to_string(iter);
    f = fopen(fname.c_str(), "w");
    for(int i=0; i<wsize; i++)
      fprintf(f, "%g\n", iwt[i]);
    fclose(f);

    if(gparams_.bias_term)
    {
      fname = gparams_.node_name + "_solver_newbias_" + to_string(iter);
      f = fopen(fname.c_str(), "w");
      for(int i=0; i<gparams_.nOutput; i++)
        fprintf(f, "%g\n", bias[i]);
      fclose(f);

      fname = gparams_.node_name + "_solver_biasinc_" + to_string(iter);
      f = fopen(fname.c_str(), "w");
      for(int i=0; i<gparams_.nOutput; i++)
        fprintf(f, "%g\n", ibias[i]);
      fclose(f);
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
    string s = nname_ + "_Winc";
    MeanOfLayer((char*)s.c_str(), iwt, wsize);
    s = nname_ + "Wt";
    MeanOfLayer((char*)s.c_str(), wt, wsize);

    if(gparams_.bias_term)
    {
      s = nname_ + "_BiasInc";
      MeanOfLayer((char*)s.c_str(), ibias, gparams_.nOutput);
      s = nname_ + "_Bias";
      MeanOfLayer((char*)s.c_str(), bias, gparams_.nOutput);
    }
  }
#endif
}

