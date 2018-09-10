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
#include "Pooling.hpp"

PoolingNode::PoolingNode(PoolingParams* p, MLEngine* e): NNNode(p, e)
{
  nname_ = p->get_node_name();
  ntype_ = p->get_node_type();
  mode_ = p->get_mode();
  bottom_ = p->get_bottom_names();
  top_ = p->get_top_names();
  bp_flag_ = p->get_bprop_flag();
  has_weights_ = false;
  bot_compute_engine_ = p->get_compute_engine();

  tenTop_ = new Tensor(top_[0]);
  assert(tenTop_ != NULL);
  tenTop_->setOwner(this);
  tenTop_->setType(ACT);
  tenTopData_ = tenTop_->getBuf(DATA);
  tenTopData_->setBufferType(DATA);

#ifdef DEBUG
  printf("bottom name %s\n",bottom_[0].c_str());
#endif

  tenBot_ = e->get_tensor(bottom_[0], ACT);
  assert(tenBot_ != NULL);
  NNNode *pnn = (NNNode*)tenBot_->getOwner();
  setPrevNode(pnn);
  NNNode::mode_ = pnn->getMode();
  pnn->set_top_compute_engine(p->get_compute_engine());
  bot_cengine_ = pnn->get_bot_compute_engine();

  tenBotData_ = tenBot_->getBuf(DATA);

  //Output tensor data type = input tensor data type
  int dtype = p->get_data_type();
  tenTopData_->setDataType(dtype);

  // Get input tensor shape (bottom)
  Shape* bs = tenBot_->getShape();
  assert(bs->ndims <= MAX_DIMS);

  // Create shape of output tensor (top)
  vector<int> vd = p->get_kernel_dims();
  vector<int> vp = p->get_pads();
  vector<int> vs = p->get_strides();

  assert((vd.size() == vp.size()) && (vd.size() == vs.size()));

  shape_setzero(&ts_);

  ts_.ndims = bs->ndims; // Number of dimensions
  ts_.dims[0] = bs->dims[0]; // Minibatch size
  ts_.dims[1] = bs->dims[1]; // Num output feature maps

  ts_.dims[2] = (bs->dims[2] - vd[0] + 2*vp[0])/vs[0] + 1; // Height

  if(ts_.ndims == 4)
    ts_.dims[3] = (bs->dims[3] - vd[1] + 2*vp[1])/vs[1] + 1; // Width
  else if(ts_.ndims == 5)
  {
    ts_.dims[3] = (bs->dims[3] - vd[1] + 2*vp[1])/vs[1] + 1; // Width
    ts_.dims[4] = (bs->dims[4] - vd[2] + 2*vp[2])/vs[2] + 1; // Depth (for 3D)
  }

  if(vp[0])
    if((ts_.dims[2] - 1) * vs[0] >= bs->dims[2] + vp[0])
      ts_.dims[2]--;

  if(vp[1])
    if((ts_.dims[3] - 1) * vs[1] >= bs->dims[3] + vp[1])
      ts_.dims[3]--;

  if(ts_.ndims == 5)
  {
    if(vp[2])
      if((ts_.dims[4] - 1) * vs[2] >= bs->dims[4] + vp[2])
        ts_.dims[4]--;
  }

  // Set output tensor shape
  tenTop_->setShape(&ts_);

  long long int tsize = 1;
  for(int i=0; i<ts_.ndims; i++)
    tsize = tsize*ts_.dims[i];

  // For now, we only support float
  if(dtype == DT_FLOAT)
    tsize = tsize*sizeof(float);
  else if(dtype == DT_DFP16)
    tsize = tsize*sizeof(float) + tsize*sizeof(short);
  else if(dtype == DT_INT)
    tsize = tsize*sizeof(int);

  // Set the logical size of the tensor buffer for bufId=0 (forward data buffer).
  // Note: we have no knowledge of the machine parameters here, so effectively this is single-machine config
  tenTopData_->setBufferSize(tsize);

  // Tensor representing mask of selected neurons. Shape is that of output tensor
  long long int size = 1;
  for(int i=0; i<ts_.ndims; i++)
    size = size*ts_.dims[i];
  size = size*sizeof(int);
  tenMask_ = new int[size];

  if(!e->is_inference_only())
  {
    if(NNNode::bp_flag_)
    {
      tenBotDiff_ = tenBot_->addBuf(); // DIFF type and index
      tenBotDiff_->setDataType(dtype);
      tenBotDiff_->setBufferType(DIFF);

      long long int bsize = 1;
      for(int i=0; i<bs->ndims; i++)
        bsize = bsize*bs->dims[i];
      if(dtype == DT_FLOAT)
        bsize = bsize*sizeof(float);
      else if(dtype == DT_DFP16)
        bsize = bsize*sizeof(float);

      // Set the size of the input-gradient buffer
      tenBotDiff_->setBufferSize(bsize);
    }
  }
  else
    tenBotDiff_ = NULL;

  // Register output tensor in tensorMap
  bool inserted = e->register_tensor(top_[0], ACT, this->tenTop_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",NNNode::top_[0].c_str());

  // Setup parameter structure for convolution computation in library
  gparams_.bdims = bs->ndims;
  gparams_.tdims = ts_.ndims;
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
  gparams_.stride_h = vs[0];
  gparams_.stride_w = vs[1];
  gparams_.stride_d = vs[2];
  gparams_.kh = vd[0];
  gparams_.kw = vd[1];
  gparams_.kd = vd[2];

  gparams_.ipad_w = 0;
  gparams_.ipad_h = 0;
  gparams_.opad_w = 0;
  gparams_.opad_h = 0;

  gparams_.pool_mode = p->get_pool_mode();

  gparams_.data_type = dtype;
  gparams_.algType = p->get_algo_type();
  gparams_.num_threads = e->get_num_threads();

  //get global scratch tensor buffer
  tenScratchData_ = e->getScratchBuffer();

  eptr_ = e;

  configure(p->get_compute_engine());
}

void PoolingNode::configure(int engine)
{
  switch(engine)
  {
    case XSMM:
      impl = new PoolXSMM(&gparams_, engine);
      break;
  }
}

void PoolingNode::forwardPropagate()
{
#ifdef DEBUG
  float* bot = (float*)(tenBotData_->getBuffer());
  float* top = (float*)(tenTopData_->getBuffer());

  printf("Executing FP %s: input %p, output %p mask %p\n",NNNode::nname_.c_str(), bot, top, tenMask_);
  printf("Inputs: %d x %d x %d\n",gparams_.nInput, gparams_.iHeight, gparams_.iWidth);
  printf("Outputs: %d x %d x %d\n",gparams_.nOutput, gparams_.oHeight, gparams_.oWidth);
#endif

  int nImg = gparams_.batch_size;
  int nOfm = gparams_.nOutput;
  int ofh = gparams_.oHeight;
  int ofw = gparams_.oWidth;

  impl->set_bot_compute_engine(bot_cengine_);
  impl->set_top_compute_engine(top_compute_engine_);
  impl->set_next_node_type(next_ntype_);
  impl->set_node_name(nname_);
  impl->set_scratch_buffer(tenScratchData_);

  if(first_fp && gparams_.data_type == DT_DFP16)
  {
    tenTopData_->setLPBuffer(tenTopData_->getBuffer() + nImg*nOfm*ofh*ofw*sizeof(float));
    first_fp = false;
  }

  impl->forwardPropagate(tenBotData_, tenTopData_, tenMask_);

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
  if(node_id==0 && eptr_->get_current_batch() % STATFREQ == 0)
  {
    float *ptr = (float*)tenBotData_->getBuffer();
    float *pptr = (float*)tenBotData_->getPrivBuffer();
    float *p = (pptr == NULL) ? ptr : pptr;
    string s = nname_ + "_Inp";
    MeanOfLayer((char*)s.c_str(), p, gparams_.batch_size*gparams_.nInput* gparams_.iHeight*gparams_.iWidth);

    ptr = (float*)tenTopData_->getBuffer();
    pptr = (float*)tenTopData_->getPrivBuffer();
    p = (pptr == NULL) ? ptr : pptr;
    s = nname_ + "_Outp";
    MeanOfLayer((char*)s.c_str(), p, gparams_.batch_size*gparams_.nOutput* gparams_.oHeight*gparams_.oWidth);
  }
#endif
}

void PoolingNode::backPropagate()
{
  tenTopDiff_ = tenTop_->getBuf(DIFF);

#ifdef DEBUG
  float *gtop = (float*)(tenTopDiff_->getBuffer());
  assert(gtop != NULL);

  float* gbot = (float*)(tenBotDiff_->getBuffer());

  printf("Executing BP %s: grad_output %p, grad_input %p\n",NNNode::nname_.c_str(), gtop, gbot);
  printf("Grad Outputs: %d x %d x %d\n", gparams_.nOutput, gparams_.oHeight, gparams_.oWidth);
  printf("Grad Inputs: %d x %d x %d\n", gparams_.nInput, gparams_.iHeight, gparams_.iWidth);
#endif

  impl->backPropagate(tenTopDiff_, tenMask_, tenBotDiff_);

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
  if(node_id==0 && eptr_->get_current_batch() % STATFREQ == 0)
  {
    float *ptr = (float*)tenTopDiff_->getBuffer();
    float *pptr = (float*)tenTopDiff_->getPrivBuffer();
    float *p = (pptr == NULL) ? ptr : pptr;
    string s = nname_ + "_delOutp";
    MeanOfLayer((char*)s.c_str(), p, gparams_.batch_size*gparams_.nOutput* gparams_.oHeight*gparams_.oWidth);

    ptr = (float*)tenBotDiff_->getBuffer();
    pptr = (float*)tenBotDiff_->getPrivBuffer();
    p = (pptr == NULL) ? ptr : pptr;
    s = nname_ + "_delInp";
    MeanOfLayer((char*)s.c_str(), p, gparams_.batch_size*gparams_.nInput* gparams_.iHeight*gparams_.iWidth);
  }
#endif
}
