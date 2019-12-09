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
#include "string.h"
#include <ctime>
#include <omp.h>
#include <unistd.h>
#include "ImageData.hpp"
#include "check.hpp"

using namespace std;
using namespace gxm;

ImageDataNode::ImageDataNode(ImageDataParams* p, MLEngine* e) : NNNode(p, e)
{
  nname_ = p->get_node_name();
  ntype_ = p->get_node_type();
  mode_ = p->get_mode();
  top_ = p->get_top_names();
  bp_flag_ = p->get_bprop_flag();
  has_weights_ = false;
  bot_compute_engine_ = LOOP;

  //Create output tensor
  for(int i=0; i<NNNode::top_.size(); i++)
  {
    tenTop_.push_back(new Tensor(top_[i]));
    assert(tenTop_[i] != NULL);
    tenTop_[i]->setOwner(this);
    tenTopData_.push_back(tenTop_[i]->getBuf(DATA));

    if(NNNode::top_[i].compare("data") == 0)
    {
      tenTop_[i]->setType(INPUT);

      int dtype = p->get_data_type();
      tenTopData_[i]->setDataType(dtype);
      tenTopData_[i]->setBufferType(DATA);

      Shape tts;

      shape_setzero(&tts);

      tts.ndims = 4;

      tts.dims[0] = p->get_batch_size();
      tts.dims[1] = p->get_channels();

      vector<int> v = p->get_crop_sizes();

      tts.dims[2] = v[0];
      tts.dims[3] = v[1];

      tenTop_[i]->setShape(&tts);

      long long int size = 1;
      for(int j=0; j<tts.ndims; j++)
        size *= tts.dims[j];

      // Size of data tensor buffer = batch_size * channels * height * width * sizeof(float/short int)
      if(dtype == DT_FLOAT)
        size = size*sizeof(float);
      else if(dtype == DT_INT16)
        size = size*sizeof(short int);

      tenTopData_[i]->setBufferSize(size);

      // Register output tensor in tensorMap
      bool inserted = e->register_tensor(NNNode::top_[i], INPUT, tenTop_[i]);
      if(!inserted)
        printf("Warning: Tensor %s already registered\n",NNNode::top_[i].c_str());

    }
    else if(top_[i].compare("label") == 0)
    {
      tenTop_[i]->setType(LABEL);

      int dtype = p->get_label_data_type();
      tenTopData_[i]->setDataType(dtype);
      tenTopData_[i]->setBufferType(DATA);

      Shape tts;
      shape_setzero(&tts);

      tts.ndims = 1;
      tts.dims[0] = p->get_batch_size();

      tenTop_[i]->setShape(&tts);

      long long int size = 1;
      for(int j=0; j<tts.ndims; j++)
        size *= tts.dims[j];

      // Size of label tensor buffer = batch_size*sizeof(int)
      assert(dtype == DT_INT);
      size = size*sizeof(int);

      tenTopData_[i]->setBufferSize(size);

      // Register output tensor in tensorMap
      bool inserted = e->register_tensor(NNNode::top_[i], LABEL, tenTop_[i]);
      if(!inserted)
        printf("Warning: Tensor %s already registered\n",NNNode::top_[i].c_str());
    }

  }

  // If training mode, setup training and validation data files, else only latter
  int mode = p->get_mode();

  ap.mirror = p->get_mirror();
  ap.vignette = p->get_vignette();
  ap.color_bump = p->get_color_bump();

  train_source_path_ = p->get_train_source_path();
  test_source_path_ = p->get_test_source_path();
  num_machines_ = e->get_num_machines();
  num_epochs_ = e->get_num_epochs();

  batch_size_ = p->get_batch_size();
  global_batch_size_ = batch_size_ * num_machines_;

  e->set_batch_size(batch_size_);

  gparams_.channels = p->get_channels();
  gparams_.crop_sizes = p->get_crop_sizes();
  gparams_.orig_sizes = p->get_orig_sizes();
  gparams_.batch_size = batch_size_;
  gparams_.threads = e->get_num_threads();

  gparams_.lookahead = p->get_lookahead();
  gparams_.mean_values = p->get_mean_values();
  gparams_.scale_values = p->get_scale_values();
  gparams_.test_views = p->get_num_test_views();

  jitters_ = p->get_jitters();

  current_epoch_ = 0;
  ctrain_pf_mb_ = 0;
  ctest_pf_mb_ = 0;
  ctrain_proc_mb_ = 0;
  ctest_proc_mb_ = 0;
  curr_test_view_ = 0;
  full_train_prefetch_ = true;
  full_test_prefetch_ = true;

  eptr = e;

  global_node_id_ = e->get_global_node_id();

  if(mode == TRAIN)
  {
    num_train_files_ = p->get_num_train_files();
    createImageList(train_list_, p->get_train_img_info(), num_train_files_);
#ifdef DUMP_DATA
    train_batches_ = 1;
#else
    train_batches_ = num_train_files_ % global_batch_size_ > 0 ? (((int)(num_train_files_/global_batch_size_)) + 1) : num_train_files_/global_batch_size_;
#endif
    e->set_num_train_batches(train_batches_);

    setupTrainIndices();

    num_test_files_ = p->get_num_test_files();
    createImageList(test_list_, p->get_test_img_info(), num_test_files_);
    test_batches_ = num_test_files_ % global_batch_size_ > 0 ? (((int)(num_test_files_/global_batch_size_)) + 1) : num_test_files_/global_batch_size_;
    e->set_num_test_batches(test_batches_);
    e->set_num_test_views(gparams_.test_views);

    setupTestIndices();
  }
  else if(mode == TEST)
  {
    num_test_files_ = p->get_num_test_files();
    createImageList(test_list_, p->get_test_img_info(), num_test_files_);
    test_batches_ = num_test_files_/global_batch_size_;
    e->set_num_test_batches(test_batches_);
    e->set_num_test_views(gparams_.test_views);

    setupTestIndices();
  }

  labels_.resize(gparams_.lookahead * gparams_.batch_size);

  // Allocate temporary buffer to hold 1 image with maximal original size
  int len = gparams_.batch_size * gparams_.channels * gparams_.orig_sizes[0] * gparams_.orig_sizes[1];

  // Size of input buffer = batch_size * channels * height * width * lookahead * sizeof(char)
  len = len * gparams_.lookahead;

  tempbuf_ = (unsigned char*)_mm_malloc(len, 64);
  memset((unsigned char*)tempbuf_, 0, len);

#ifdef USE_MLSL
  MLSL::Session *s = e->get_session();
  s->SetGlobalMinibatchSize(global_batch_size_);
#endif

  configure(RGB_FLATFILE);
}

void ImageDataNode::configure(int dataType)
{
  switch(dataType)
  {
    case RGB_FLATFILE:
      if(num_machines_ == 1)
        srand(727);
      else
        srand(global_node_id_);
      impl = new ImageDataRGBFlat(&gparams_, &ap);
      break;
  }
}

void ImageDataNode::setupTrainIndices()
{
  int ntrain = num_train_files_;

  if(ntrain <= batch_size_)
  {
    ntrain = batch_size_;
    gparams_.lookahead = 1; // Override default lookahead, if any
  }

  t_files_ = ntrain % global_batch_size_ > 0 ? (((int)(ntrain/global_batch_size_)) + 1)*global_batch_size_ : ntrain;

  tfiles_per_mc_ = t_files_/num_machines_;

  train_imginfo_index_.resize(tfiles_per_mc_);

  vector<int> tfile_index(t_files_);

  for(int i=0; i<t_files_; i++)
  {
    if(i >= ntrain)
      tfile_index[i] = tfile_index[i-ntrain];
    else
      tfile_index[i] = i;
  }

  random_shuffle(tfile_index.begin(), tfile_index.end());

  int k=0;
  for(int n1=0; n1 < train_batches_; n1++)
    for(int n2=0; n2 < batch_size_; n2++)
      train_imginfo_index_[k++] = tfile_index[n1*global_batch_size_ + n2*num_machines_ + global_node_id_];
}

void ImageDataNode::setupTestIndices()
{
  int ntest = num_test_files_;

  if(ntest <= batch_size_)
  {
    ntest = batch_size_;
    gparams_.lookahead = 1; // Override default lookahead, if any
  }

  v_files_ = ntest % global_batch_size_ > 0 ? (((int)(ntest/global_batch_size_)) + 1)*global_batch_size_ : ntest;
  vfiles_per_mc_ = v_files_/num_machines_;

  vector<int> temp_index(v_files_);
  test_imginfo_index_.resize(vfiles_per_mc_);

  for(int i=0; i<v_files_; i++)
  {
    if(i >= ntest)
      temp_index[i] = temp_index[i-ntest];
    else
      temp_index[i] = i;
  }

  for(int n=0; n<vfiles_per_mc_; n++)
    test_imginfo_index_[n] = temp_index[global_node_id_*vfiles_per_mc_ + n];

}

void ImageDataNode::createImageList(vector<ImageInfo>& list, string infofile, int nfiles)
{
  FILE *f = fopen(infofile.c_str(), "r");
  ImageInfo ii;

  for(int i=0; i<nfiles; i++)
  {
    char s[32]={0};
    fscanf(f, "%s %d %d %d %d\n", s, &(ii.height), &(ii.width), &(ii.length), &(ii.label));
    string temp(s);
    ii.name = temp;

    list.push_back(ii);
  }

  fclose(f);
}

void ImageDataNode::forwardPropagate()
{
  float *topdata = (float*)(tenTopData_[0]->getBuffer());
  int* toplabel = (int*)(tenTopData_[1]->getBuffer());

#ifdef DEBUG
  printf("Executing FP %s: Data %p, Label %p\n", NNNode::nname_.c_str(),topdata, toplabel);
#endif

  int em = eptr->get_execution_mode();
  gparams_.exec_mode = em;
  current_epoch_ = eptr->get_current_epoch();

  if(em == TRAIN)
  {
    if(full_train_prefetch_)
    {
      for(int i=0; i<gparams_.lookahead; i++)
      {
        for(int img=0; img<gparams_.batch_size; img++)
        {
          int idx = i*gparams_.batch_size + img;
          ImageInfo ii = train_list_[train_imginfo_index_[idx]];
          labels_[i*gparams_.batch_size+img] = ii.label;

          // Read gparams_.batch_size*lookahead files into tempbuf_
          string fpath(train_source_path_ + ii.name);
          FILE *f = fopen(fpath.c_str(), "rb");
          int bytes = fread(&tempbuf_[idx*ii.length], sizeof(char), ii.length, f);
          assert(bytes == ii.length);
          fclose(f);
#if 0
          printf("idx %d, ii.name %s, ii.label %d\n",idx, index, ii.name.c_str(), ii.label);
#endif
        }
      }
      ctrain_pf_mb_ += gparams_.lookahead;
      full_train_prefetch_ = false;
    }
    else
    {
      if(ctrain_pf_mb_ < train_batches_)
      {
        for(int img=0; img<gparams_.batch_size; img++)
        {
          int idx = ctrain_pf_mb_ * gparams_.batch_size + img;
          int index = (ctrain_pf_mb_%gparams_.lookahead)*gparams_.batch_size + img;

          ImageInfo ii = train_list_[train_imginfo_index_[idx]];
          labels_[index] = ii.label;

          // Read batch_size files into tempbuf_
          string fpath(train_source_path_ + ii.name);
          FILE* f = fopen(fpath.c_str(), "rb");
          int bytes = fread(&tempbuf_[index*ii.length], sizeof(char), ii.length, f);
          assert(bytes == ii.length);
          fclose(f);
#if 0
          printf("idx %d, index %d, ii.name %s, ii.label %d\n",idx, index, ii.name.c_str(), ii.label);
#endif
        }
        ctrain_pf_mb_++;
      }
    }

    int mbslot = (ctrain_proc_mb_%gparams_.lookahead)*gparams_.batch_size;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<gparams_.batch_size; i++)
      toplabel[i] = labels_[mbslot+i];

    int orig_img_size = gparams_.orig_sizes[0]*gparams_.orig_sizes[1]*gparams_.channels;
    impl->forwardPropagate((unsigned char*)&tempbuf_[mbslot*orig_img_size], topdata);
#if 0
    printf("mblsot %d, orig_img_size %d\n",mbslot,orig_img_size);
#endif

#ifdef GETSTATS
#ifdef USE_MLSL
    size_t node_id = MLSL::Environment::GetEnv().GetProcessIdx();
    if(node_id == 0 && eptr->get_current_batch() % STATFREQ == 0)
#endif
    {
      int crop_img_size = gparams_.crop_sizes[0]*gparams_.crop_sizes[1]*gparams_.channels;
      MeanOfLayer("Data", topdata, gparams_.batch_size*crop_img_size);
    }
#endif

#ifdef DEBUG
    int crop_img_size = gparams_.crop_sizes[0]*gparams_.crop_sizes[1]*gparams_.channels;
    double sum=0.;
    for(int i=0; i<gparams_.batch_size*crop_img_size; i++)
      sum += topdata[i];
    printf("Data checksum: %.10f\n",sum);

    int isum=0;
    long long int min=INT_MAX, max=-INT_MAX;
    double mean, stdev=0.0;
    for(int i=0; i<gparams_.batch_size; i++)
    {
      isum += toplabel[i];
      if((long long int)toplabel[i] > max) max = toplabel[i];
      if((long long int)toplabel[i] < min) min = toplabel[i];
    }
    mean = (double)isum/(double)gparams_.batch_size;

    for(int i=0; i<gparams_.batch_size; i++)
      stdev += ((double)toplabel[i] - mean)*((double)toplabel[i] - mean);
    stdev = sqrt(stdev/gparams_.batch_size);
    printf("label sum %d, mean %f, stdev %f min %lld, max %lld\n",isum, mean, stdev, min, max);

#endif
    ctrain_proc_mb_++;
    if(ctrain_proc_mb_ == train_batches_)
    {
      ctrain_pf_mb_ = 0;
      ctrain_proc_mb_ = 0;
      full_train_prefetch_ = true;
    }
  }
  else if(em == TEST)
  {
    if(full_test_prefetch_)
    {
      for(int i=0; i<gparams_.lookahead; i++)
      {
        for(int img=0; img<gparams_.batch_size; img++)
        {
          int idx = i*gparams_.batch_size + img;
          ImageInfo ii = test_list_[test_imginfo_index_[idx]];
          labels_[idx] = ii.label;

          // Read gparams_.batch_size*lookahead files into tempbuf_
          int index = i*gparams_.batch_size + img;

          string fpath(test_source_path_ + ii.name);
          FILE *f = fopen(fpath.c_str(), "rb");
          int bytes = fread(&tempbuf_[index*ii.length], sizeof(char), ii.length, f);
          assert(bytes == ii.length);
          fclose(f);
        }
      }
      ctest_pf_mb_ += gparams_.lookahead;
      full_test_prefetch_ = false;
    }
    else
    {
      if(ctest_pf_mb_ < test_batches_)
      {
        for(int img=0; img<gparams_.batch_size; img++)
        {
          int idx = ctest_pf_mb_ * gparams_.batch_size + img;
          int index = (ctest_pf_mb_%gparams_.lookahead)*gparams_.batch_size + img;

          ImageInfo ii = test_list_[test_imginfo_index_[idx]];
          labels_[index] = ii.label;

          // Read batch_size files into tempbuf_
          string fpath(test_source_path_ + ii.name);
          FILE* f = fopen(fpath.c_str(), "rb");
          int bytes = fread(&tempbuf_[index*ii.length], sizeof(char), ii.length, f);
          assert(bytes == ii.length);
          fclose(f);
        }
        ctest_pf_mb_++;
      }
    }

    int mbslot = (ctest_proc_mb_%gparams_.lookahead)*gparams_.batch_size;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<gparams_.batch_size; i++)
      toplabel[i] = labels_[mbslot+i];

    int orig_img_size = gparams_.orig_sizes[0]*gparams_.orig_sizes[1]*gparams_.channels;
    impl->forwardPropagate((unsigned char*)&tempbuf_[mbslot*orig_img_size], curr_test_view_, topdata);

    curr_test_view_++;

#ifdef DEBUG
    printf("tv = %d\n",curr_test_view_);
#endif
    if(curr_test_view_ == gparams_.test_views)
    {
      curr_test_view_ = 0;

      ctest_proc_mb_++;
      if(ctest_proc_mb_ == test_batches_)
      {
        ctest_pf_mb_ = 0;
        ctest_proc_mb_ = 0;
        full_test_prefetch_ = true;
      }
    }
  }

#ifdef DUMP_DATA
  int crop_size = gparams_.batch_size * gparams_.crop_sizes[0] * gparams_.crop_sizes[1] * gparams_.channels;
  string fname = NNNode::nname_ + "_fp_out";
  FILE *f = fopen(fname.c_str(), "w");
  for(int i=0; i<crop_size; i++)
    fprintf(f, "%g\n", topdata[i]);
  fclose(f);

  fname = NNNode::nname_ + "_fp_label";
  f = fopen(fname.c_str(), "w");
  for(int i=0; i<gparams_.batch_size; i++)
    fprintf(f, "%d\n", toplabel[i]);
  fclose(f);
#endif
}
