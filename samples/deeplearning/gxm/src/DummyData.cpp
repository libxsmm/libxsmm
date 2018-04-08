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
      Shape* ts = p->get_shape();
      tenTop_[i]->setShape(ts);

      long long int size = 1;
      for(int j=0; j<ts->ndims; j++)
        size *= ts->dims[j];

      if(dtype == DT_FLOAT)
        size = size*sizeof(float);
      else if(dtype == DT_INT16)
        size = size*sizeof(short int);
      else if(dtype == DT_INT)
        size = size*sizeof(int);

      // Set the logical size of the tensor buffer for bufId=0 (forward data buffer).
      // Note: we have no knowledge of the machine parameters here, so effectively this is single-machine config
      tenTop_[i]->setDataBufferSize(DATA, size);

      num_machines_ = e->get_num_machines();
      global_batch_size_ = num_machines_ * ts->dims[0];
printf("gbs = %d\n", global_batch_size_);
#ifdef USE_MLSL
      MLSL::Session *s = e->get_session();
      s->SetGlobalMinibatchSize(global_batch_size_);
#endif

      if(p->get_num_train_files() != 0)
        e->set_num_train_batches(p->get_num_train_files()/ts->dims[0]);

      if(p->get_num_test_files() != 0)
      {
        e->set_num_test_batches(p->get_num_test_files()/ts->dims[0]);
        e->set_num_test_views(1);
      }

      e->set_batch_size(ts->dims[0]);
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

void DummyDataNode::fillData(float* ptr, long long int size)
{
  if(filler_type_.compare("rand") == 0)
  {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(long long int i=0; i<size; i++)
      ptr[i] = rand()/RAND_MAX;
#if 0
    for(int i=0; i<10; i++)
      printf("%f ",ptr[i]);
    printf("\n");
#endif
  }
  if(filler_type_.compare("constant") == 0)
  {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(long long int i=0; i<size; i++)
      ptr[i] = filler_val_;
#if 0
    for(int i=0; i<10; i++)
      printf("%f ",ptr[i]);
    printf("\n");
#endif
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
#if 0
    for(int i=0; i<10; i++)
      printf("%d ",ptr[i]);
    printf("\n");
#endif
  }
  else if(filler_type_.compare("constant") == 0)
  {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(long long int i=0; i<size; i++)
      ptr[i] = filler_val_;
#if 0
    for(int i=0; i<10; i++)
      printf("%d ",ptr[i]);
    printf("\n");
#endif

  }
}

void DummyDataNode::fillData(short int* ptr, long long int size)
{
  if(filler_type_.compare("rand") == 0)
  {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(long long int i=0; i<size; i++)
      ptr[i] = (short int)(rand()%size);
#if 0
    for(int i=0; i<10; i++)
      printf("%d ",ptr[i]);
    printf("\n");
#endif
  }
  else if(filler_type_.compare("constant") == 0)
  {
#ifdef _OEPNMP
#pragma omp parallel for
#endif
    for(long long int i=0; i<size; i++)
      ptr[i] = filler_val_;
#if 0
    for(int i=0; i<10; i++)
      printf("%d ",ptr[i]);
    printf("\n");
#endif

  }
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
    else if(dtype == DT_INT16)
    {
      short int* top = (short int*)(tenTopData_[i]->getBuffer());
      fillData(top, bytes/sizeof(short int));
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

