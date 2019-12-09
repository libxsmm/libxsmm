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



#include "DummyData.hpp"

DummyDataNode::DummyDataNode(DummyDataParams* p, MLEngine* e) : NNNode(p, e)
{
  nname_ = p->get_node_name();
  ntype_ = p->get_node_type();
  mode_ = p->get_mode();
  top_ = p->get_top_names();
  has_weights_ = false;
  bot_compute_engine_ = p->get_compute_engine();

  int dtype;

  tenTop_.resize(top_.size());
  tenTopData_.resize(top_.size());
  for(int i=0; i<top_.size(); i++)
  {
    tenTop_[i] = new Tensor(top_[i]);
    assert(tenTop_[i] != NULL);
    tenTop_[i]->setOwner(this);
    tenTopData_[i] = tenTop_[i]->getBuf(DATA);

    if(top_[i].compare("data") == 0)
    {
      tenTop_[i]->setType(INPUT);

      tenTopData_[i]->setBufferType(DATA);
      // FIXME: the data type should be set elsewhere...
      dtype = p->get_data_type();
      tenTopData_[i]->setDataType(dtype);
      ts_ = p->get_shape();
      pad_h_ = p->get_pad_h();
      pad_w_ = p->get_pad_w();

      tenTop_[i]->setShape(ts_);

      long long int size = ts_->dims[0] * ts_->dims[1] * (ts_->dims[2] + 2*pad_h_) * (ts_->dims[3] + 2*pad_w_);

      if(dtype == DT_FLOAT)
        size = size*sizeof(float);
      else if(dtype == DT_BF16)
        size = size*sizeof(float);

      // Set the logical size of the tensor buffer for bufId=0 (forward data buffer).
      // Note: we have no knowledge of the machine parameters here, so effectively this is single-machine config
      tenTop_[i]->setDataBufferSize(DATA, size);

      num_machines_ = e->get_num_machines();
      global_batch_size_ = num_machines_ * ts_->dims[0];
#ifdef USE_MLSL
      MLSL::Session *s = e->get_session();
      s->SetGlobalMinibatchSize(global_batch_size_);
#endif

      if(p->get_num_train_files() != 0)
        e->set_num_train_batches(p->get_num_train_files()/ts_->dims[0]);

      if(p->get_num_test_files() != 0)
      {
        e->set_num_test_batches(p->get_num_test_files()/ts_->dims[0]);
        e->set_num_test_views(1);
      }

      e->set_batch_size(ts_->dims[0]);
      bool inserted = e->register_tensor(top_[i], INPUT, tenTop_[i]);
      if(!inserted)
        printf("Warning: Tensor %s already registered\n",top_[i].c_str());

      filler_type_ = p->get_filler_type();
      filler_val_ = p->get_filler_val();
    }
    else if(top_[i].compare("label") == 0)
    {
      tenTop_[i]->setType(LABEL);

      Shape *ts = p->get_shape();
      ts->ndims = 1;
      ts->dims[1] = 0;
      ts->dims[2] = 0;
      ts->dims[3] = 0;

      // FIXME: the data type should be set elsewhere...
      dtype = DT_INT;
      tenTopData_[i]->setDataType(dtype);
      tenTopData_[i]->setBufferType(DATA);
      tenTop_[i]->setDataBufferSize(DATA, ts->dims[0]*sizeof(int));
      tenTop_[i]->setShape(ts);

      bool inserted = e->register_tensor(top_[i], LABEL, tenTop_[i]);
      if(!inserted)
        printf("Warning: Tensor %s already registered\n",NNNode::top_[i].c_str());
    }
  }

  //No input tensor to this node
  this->tenBot_ = NULL;
}

void convert_f32_bf16(float* in, libxsmm_bfloat16* out, int len)
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

void DummyDataNode::fillData(float* ptr, long long int size)
{
  if(first_fp)
  {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<size; i++)
      ptr[i] = 0;

    first_fp = false;
  }

  Shape *ts = tenTop_[0]->getShape();

  int ifhp = ts->dims[2]+2*pad_h_;
  int ifwp = ts->dims[3]+2*pad_w_;
  int nFM = ts->dims[1];

  float (* __restrict input)[ifhp][ifwp][nFM] = (float (*)[*][*][*])ptr;

  if(filler_type_ == "rand")
  {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int img=0; img<ts->dims[0]; img++) {
      for(int h=pad_h_; h<ts->dims[2]+pad_h_; h++) {
        for(int w=pad_w_; w<ts->dims[3]+pad_w_; w++) {
          for(int fm=0; fm<ts->dims[1]; fm++) {
            input[img][h][w][fm] = (float)(rand()/RAND_MAX);
          }
        }
      }
    }
  }
  else if(filler_type_ == "constant")
  {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int img=0; img<ts->dims[0]; img++) {
      for(int h=pad_h_; h<ts->dims[2]+pad_h_; h++) {
        for(int w=pad_w_; w<ts->dims[3]+pad_w_; w++) {
          for(int fm=0; fm<ts->dims[1]; fm++) {
            input[img][h][w][fm] = filler_val_;
          }
        }
      }
    }
  }
}

void DummyDataNode::fillData(int* ptr, long long int size)
{
  if(filler_type_.compare("rand") == 0)
  {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(long long int i=0; i<size; i++)
      ptr[i] = rand()%1000;
  }
  else if(filler_type_.compare("constant") == 0)
  {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(long long int i=0; i<size; i++)
      ptr[i] = filler_val_;
  }
}

void DummyDataNode::fillData(float *inptr, libxsmm_bfloat16* outptr, long long int size)
{
  Shape *ts = tenTop_[0]->getShape();

  int ifhp = ts->dims[2]+2*pad_h_;
  int ifwp = ts->dims[3]+2*pad_w_;
  int nFM = ts->dims[1];

  if(first_fp)
  {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<size; i++)
      outptr[i] = 0;

    first_fp = false;
  }

  float (* __restrict input)[ifhp][ifwp][nFM] = (float (*)[*][*][*])inptr;

  if(filler_type_.compare("rand") == 0)
  {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int img=0; img<ts->dims[0]; img++) {
      for(int h=pad_h_; h<ts->dims[2]+pad_h_; h++) {
        for(int w=pad_w_; w<ts->dims[3]+pad_w_; w++) {
          for(int fm=0; fm<ts->dims[1]; fm++) {
            input[img][h][w][fm] = (float)rand()/(float)RAND_MAX;
          }
        }
      }
    }
  }
  else if(filler_type_.compare("constant") == 0)
  {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int img=0; img<ts->dims[0]; img++) {
      for(int h=pad_h_; h<ts->dims[2]+pad_h_; h++) {
        for(int w=pad_w_; w<ts->dims[3]+pad_w_; w++) {
          for(int fm=0; fm<ts->dims[1]; fm++) {
            input[img][h][w][fm] = (float)filler_val_;
          }
        }
      }
    }
  }
  convert_f32_bf16(inptr, outptr, size);
}


void DummyDataNode::forwardPropagate()
{
#ifdef RETURNALL
  return;
#endif

  for(int i=0; i<tenTopData_.size(); i++)
  {
    int dtype = tenTopData_[i]->getDataType();
    long long int bytes = tenTopData_[i]->getBufferSize();

    if(dtype == DT_FLOAT)
    {
      float* top = (float*)(tenTopData_[i]->getBuffer());
      fillData(top, bytes/sizeof(float));
#ifdef DEBUG
      printf("Executing FP %s: Data %p\n",node_name_.c_str(), top);
#endif
    }
    else if(dtype == DT_BF16)
    {
      libxsmm_bfloat16* top = (libxsmm_bfloat16*)(tenTopData_[i]->getLPBuffer());
      if(top == NULL)
        top = (libxsmm_bfloat16*)_mm_malloc(bytes/sizeof(libxsmm_bfloat16), 64);
      tenTopData_[i]->setLPBuffer(top);
      float *bot = (float*)tenTopData_[i]->getBuffer();
      fillData(bot, top, bytes/sizeof(float));

#ifdef DEBUG
      printf("Executing FP %s: Data %p\n",node_name_.c_str(), top);
#endif
    }
    else if(dtype == DT_INT)
    {
      int* top = (int*)(tenTopData_[i]->getBuffer());
      for(long long int i=0; i<bytes/sizeof(int); i++)
        top[i] = rand()%1000;
    }
  }
}

