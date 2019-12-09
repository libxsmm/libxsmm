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

  out_dtype = p->get_data_type();
  tenTopData_->setDataType(out_dtype);
  in_dtype = tenBotData_->getDataType();

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

  if(out_dtype == DT_FLOAT)
    tsize = tsize*sizeof(float);
  else if(out_dtype == DT_BF16)
    tsize = tsize*sizeof(libxsmm_bfloat16);

  tenTopData_->setBufferSize(tsize);

  // Tensor representing mask of selected neurons.
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

  gparams_.in_data_type = in_dtype;
  gparams_.out_data_type = out_dtype;
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

void PoolingNode::convert_bf16_f32(libxsmm_bfloat16* in, float* out, int len)
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

void PoolingNode::forwardPropagate()
{
#ifdef DEBUG
  void *bot = tenBotData_->getBuffer();
  void *top = tenTopData_->getBuffer();

  printf("Executing FP %s: input %p, output %p mask %p\n",NNNode::nname_.c_str(), bot, top, tenMask_);
  printf("Inputs: %d x %d x %d\n",gparams_.nInput, gparams_.iHeight, gparams_.iWidth);
  printf("Outputs: %d x %d x %d\n",gparams_.nOutput, gparams_.oHeight, gparams_.oWidth);
#endif

  int nImg = gparams_.batch_size;
  int ofm = gparams_.nOutput;
  int ifh = gparams_.iHeight;
  int ifw = gparams_.iWidth;
  int ofh = gparams_.oHeight;
  int ofw = gparams_.oWidth;

  impl->set_bot_compute_engine(bot_cengine_);
  impl->set_top_compute_engine(top_compute_engine_);
  impl->set_next_node_type(next_ntype_);
  impl->set_node_name(nname_);
  impl->set_scratch_buffer(tenScratchData_);

  impl->forwardPropagate(tenBotData_, tenTopData_, tenMask_);

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
  size_t node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
  size_t node_id = 0;
#endif
  if(node_id==0 && eptr_->get_current_batch() % STATFREQ == 0)
  {
    if(gparams_.in_data_type == DT_FLOAT && gparams_.out_data_type == DT_FLOAT)
    {
      float *ptr = (float*)tenBotData_->getBuffer();
      string s = nname_ + "_Inp";
      MeanOfLayer((char*)s.c_str(), ptr, nImg*ofm*ifh*ifw);

      ptr = (float*)tenTopData_->getBuffer();
      s = nname_ + "_Outp";
      MeanOfLayer((char*)s.c_str(), ptr, nImg*ofm*ofh*ofw);
    }
    else if(gparams_.in_data_type == DT_BF16 && gparams_.out_data_type == DT_BF16)
    {
      if(stptr == NULL)
      {
        int s = nImg*ofm*ofh*ofw;
        int is = nImg*ofm*ifh*ifw;
        if(s > is)
          stptr = (float*)libxsmm_aligned_malloc(s*sizeof(float), 2097152);
        else
          stptr = (float*)libxsmm_aligned_malloc(is*sizeof(float), 2097152);
      }

      libxsmm_bfloat16 *ptr = (libxsmm_bfloat16*)tenBotData_->getBuffer();
      convert_bf16_f32(ptr, stptr,  nImg*ofm*ifh*ifw);
      string s = nname_ + "_Inp";
      MeanOfLayer((char*)s.c_str(), stptr, nImg*ofm*ifh*ifw);

      ptr = (libxsmm_bfloat16*)tenTopData_->getBuffer();
      convert_bf16_f32(ptr, stptr,  nImg*ofm*ofh*ofw);
      s = nname_ + "_Outp";
      MeanOfLayer((char*)s.c_str(), stptr, nImg*ofm*ofh*ofw);
    }
  }
#endif
}

void PoolingNode::backPropagate()
{
  int nImg = gparams_.batch_size;
  int ofm = gparams_.nOutput;
  int ifh = gparams_.iHeight;
  int ifw = gparams_.iWidth;
  int ofh = gparams_.oHeight;
  int ofw = gparams_.oWidth;

  tenTopDiff_ = tenTop_->getBuf(DIFF);

#ifdef DEBUG
  void *gtop = tenTopDiff_->getBuffer();
  assert(gtop != NULL);

  void* gbot = tenBotDiff_->getBuffer();

  printf("Executing BP %s: grad_output %p, grad_input %p\n",NNNode::nname_.c_str(), gtop, gbot);
  printf("Grad Outputs: %d x %d x %d\n", gparams_.nOutput, gparams_.oHeight, gparams_.oWidth);
  printf("Grad Inputs: %d x %d x %d\n", gparams_.nInput, gparams_.iHeight, gparams_.iWidth);
#endif

  impl->backPropagate(tenTopDiff_, tenMask_, tenBotDiff_);

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
  size_t node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
  size_t node_id = 0;
#endif
  if(node_id==0 && eptr_->get_current_batch() % STATFREQ == 0)
  {
    if(gparams_.in_data_type == DT_FLOAT && gparams_.out_data_type == DT_FLOAT)
    {
      float *ptr = (float*)tenTopDiff_->getBuffer();
      string s = nname_ + "_delOutp";
      MeanOfLayer((char*)s.c_str(), ptr, nImg*ofm*ofh*ofw);

      ptr = (float*)tenBotDiff_->getBuffer();
      s = nname_ + "_delInp";
      MeanOfLayer((char*)s.c_str(), ptr, nImg*ofm*ifh*ifw);
    }
    else if(gparams_.in_data_type == DT_BF16 && gparams_.out_data_type == DT_BF16)
    {
      libxsmm_bfloat16 *ptr = (libxsmm_bfloat16*)tenTopDiff_->getBuffer();
      convert_bf16_f32(ptr, stptr, nImg*ofm*ofh*ofw);
      string s = nname_ + "_delOutp";
      MeanOfLayer((char*)s.c_str(), stptr, nImg*ofm*ofh*ofw);

      ptr = (libxsmm_bfloat16*)tenBotDiff_->getBuffer();
      convert_bf16_f32(ptr, stptr, nImg*ofm*ifh*ifw);
      s = nname_ + "_delInp";
      MeanOfLayer((char*)s.c_str(), stptr, nImg*ofm*ifh*ifw);
    }
  }
#endif
}
