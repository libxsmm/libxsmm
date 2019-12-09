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


#include "LMDBData.hpp"

using namespace std;
using namespace gxm;

LMDBDataNode::LMDBDataNode(LMDBDataParams* p, MLEngine* e) : NNNode(p, e)
{
  nname_ = p->get_node_name();
  ntype_ = p->get_node_type();
  mode_ = p->get_mode();
  top_ = p->get_top_names();
  bp_flag_ = p->get_bprop_flag();
  has_weights_ = false;
  bot_compute_engine_ = p->get_compute_engine();

  //Create output tensor

  tenTop_.resize(top_.size());
  tenTopData_.resize(top_.size());

  vector<int> vc = p->get_crop_sizes();
  vector<int> vo = p->get_orig_sizes();

  assert(vo.size() > 0);

  for(int i=0; i<top_.size(); i++)
  {
    tenTop_[i] = new Tensor(top_[i]);
    assert(tenTop_[i] != NULL);
    tenTop_[i]->setOwner(this);
    tenTopData_[i] = tenTop_[i]->getBuf(DATA);

    if(top_[i].compare("data") == 0)
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

      if(vc.size() > 0)
      {
        tts.dims[2] = vc[0];
        tts.dims[3] = vc[1];
      }
      else
      {
        tts.dims[2] = vo[0];
        tts.dims[3] = vo[1];
      }

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
      bool inserted = e->register_tensor(top_[i], INPUT, tenTop_[i]);
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
      bool inserted = e->register_tensor(top_[i], LABEL, tenTop_[i]);
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
  split_db_ = p->get_split_db_flag();
  num_machines_ = e->get_num_machines();
  num_epochs_ = e->get_num_epochs();

  batch_size_ = p->get_batch_size();
  global_batch_size_ = batch_size_ * num_machines_;

  e->set_batch_size(batch_size_);

  gparams_.channels = p->get_channels();
  gparams_.orig_sizes = vo;
  gparams_.crop_sizes = vc.size() > 0 ? vc : vo;
  gparams_.batch_size = batch_size_;
  gparams_.threads = e->get_num_threads();

  gparams_.lookahead = p->get_lookahead();
  if(p->get_mean_values().size() > 0)
    gparams_.mean_values = p->get_mean_values();
  else if(p->get_mean_file().size() > 0)
    gparams_.mean_file = p->get_mean_file();

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

  tempbuf_.resize(gparams_.lookahead);
  for(int i=0; i < gparams_.lookahead; i++)
    tempbuf_[i].resize(gparams_.batch_size);

  if(mode == TRAIN)
  {
    num_train_files_ = p->get_num_train_files();

    train_batches_ = num_train_files_ % global_batch_size_ > 0 ? (((int)(num_train_files_/global_batch_size_)) + 1) : num_train_files_/global_batch_size_;

    e->set_num_train_batches(train_batches_);

    num_test_files_ = p->get_num_test_files();
    test_batches_ = num_test_files_ % global_batch_size_ > 0 ? (((int)(num_test_files_/global_batch_size_)) + 1) : num_test_files_/global_batch_size_;
    e->set_num_test_batches(test_batches_);
    e->set_num_test_views(gparams_.test_views);

  }
  else if(mode == TEST)
  {
    num_test_files_ = p->get_num_test_files();
    test_batches_ = num_test_files_ % global_batch_size_ > 0 ? (((int)(num_test_files_/global_batch_size_)) + 1) : num_test_files_/global_batch_size_;
    e->set_num_test_batches(test_batches_);
    e->set_num_test_views(gparams_.test_views);
  }

#ifdef USE_MLSL
  MLSL::Session *s = e->get_session();
  s->SetGlobalMinibatchSize(global_batch_size_);
#endif

  tenSeeds_ = new unsigned int[gparams_.threads*16];
  initSeeds(tenSeeds_, gparams_.threads);

  r_offset = new int[gparams_.batch_size]();
  c_offset = new int[gparams_.batch_size]();
  augmentation = new int[gparams_.batch_size]();

  configure();
#ifdef USE_MLSL
  node_id_ = MLSL::Environment::GetEnv().GetProcessIdx();
  num_nodes_ = MLSL::Environment::GetEnv().GetProcessCount();
#else
  node_id_ = 0;
  num_nodes_ = 1;
#endif
}

void LMDBDataNode::configure()
{
  srand48(global_node_id_);

  train_lmdb_ = new LMDB();
  train_lmdb_->Open(train_source_path_);
  train_cursor_ = train_lmdb_->NewCursor();
#ifdef USE_MLSL
  if(node_id_ > 0 && !split_db_)
    train_cursor_->Next(global_node_id_ - 1); // Each node computes num images to skip.
#endif

  test_lmdb_ = new LMDB();
  test_lmdb_->Open(test_source_path_);
  test_cursor_ = test_lmdb_->NewCursor();
#ifdef USE_MLSL
  if(node_id_ > 0 && !split_db_)
    test_cursor_->Next(global_node_id_ - 1); // Each node computes num images to skip.
#endif
}

void LMDBDataNode::trainImageTransform(vector<Datum>& v, float* outp)
{
  int nImg = gparams_.batch_size;
  int nOfm = gparams_.channels;
  int ofh = gparams_.crop_sizes[0];
  int ofw = gparams_.crop_sizes[1];
//  vector<float>& mean = gparams_.mean_values;
//  vector<float>& scale = gparams_.scale_values;

  float (* __restrict output)[nOfm][ofh][ofw] = (float (*)[*][*][*])outp;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int img = 0; img < nImg; img++) {
    for(int ofm = 0; ofm < nOfm; ofm++) {
      for(int h = 0; h < ofh; h++) {
        for(int w = 0; w < ofw; w++) {
          int ifh = v[img].height();
          int ifw = v[img].width();

          assert(v[img].channels() == nOfm);
          const unsigned char (* __restrict input)[ifh][ifw] = (unsigned char (*)[*][*])v[img].data().c_str();

          int r_off = r_offset[img];
          int c_off = c_offset[img];

          float inp = (float)input[ofm][h+r_off][w+c_off];
          int fm = (gparams_.scale_values.size() == 1) ? 0 : ofm;

          if((augmentation[img] < 6) && (ap.mirror == true))
            output[img][ofm][h][ofw-w-1] = (inp - gparams_.mean_values[ofm]) * gparams_.scale_values[fm];
          else
            output[img][ofm][h][w] = (inp - gparams_.mean_values[ofm]) * gparams_.scale_values[fm];
        }
      }
    }
  }
}

void LMDBDataNode::testImageTransform(vector<Datum>& v, int tv, float* outp)
{
  int nImg = gparams_.batch_size;
  int nOfm = gparams_.channels;
  int ofh = gparams_.crop_sizes[0];
  int ofw = gparams_.crop_sizes[1];
//  vector<float>& mean = gparams_.mean_values;
//  vector<float>& scale = gparams_.scale_values;

  float (* __restrict output)[nOfm][ofh][ofw] = (float (*)[*][*][*])outp;

  int tv2 = tv/2;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0; i<nImg; i++)
  {
    if(tv % 2 == 0)
    {
      if(tv2 == 0) {
        r_offset[i] = (v[i].height() - ofh)/2;
        c_offset[i] = (v[i].width() - ofw)/2;
      }
      else if(tv2 == 1) {
        r_offset[i] = 0;
        c_offset[i] = 0;
      }
      else if(tv2 == 2) {
        r_offset[i] = 0;
        c_offset[i] = (v[i].width() - ofw);
      }
      else if(tv2 == 3) {
        r_offset[i] = (v[i].height() - ofh);
        c_offset[i] = 0;
      }
      else if(tv2 == 4) {
        r_offset[i] = v[i].height() - ofh;
        c_offset[i] = v[i].width() - ofw;
      }
      else if(tv2 == 5)
      {
        if(gparams_.crop_sizes[0] != gparams_.orig_sizes[0] && gparams_.crop_sizes[1] != gparams_.orig_sizes[1])
        {
          int r = v[i].height() - gparams_.crop_sizes[0] + 1;
          int c = v[i].width() - gparams_.crop_sizes[1] + 1;
          r_offset[i] = lrand48() % r;
          c_offset[i] = lrand48() % c;
        }
      }
    }
  }

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int img = 0; img < nImg; img++) {
    for(int ofm = 0; ofm < nOfm; ofm++) {
      for(int h = 0; h < ofh; h++) {
        for(int w = 0; w < ofw; w++) {
          int ifh = v[img].height();
          int ifw = v[img].width();

          assert(v[img].channels() == nOfm);
          const unsigned char (* __restrict input)[ifh][ifw] = (const unsigned char (*)[*][*])v[img].data().c_str();

          int r_off = r_offset[img];
          int c_off = c_offset[img];

          float inp = (float)input[ofm][h+r_off][w+c_off];
          int fm = (gparams_.scale_values.size() == 1) ? 0 : ofm;

          if(tv % 2 == 0)
            output[img][ofm][h][w] = (inp - gparams_.mean_values[ofm]) * gparams_.scale_values[fm];
          else
            output[img][ofm][ofh-h-1][ofw-w-1] = (inp - gparams_.mean_values[ofm]) * gparams_.scale_values[fm];
        }
      }
    }
  }
}

void LMDBDataNode::forwardPropagate()
{
  float *topdata = (float*)(tenTopData_[0]->getBuffer());
  int* toplabel = (int*)(tenTopData_[1]->getBuffer());

#if 0 //def DEBUG
  printf("Executing FP %s: Data %p, Label %p\n", NNNode::nname_.c_str(),topdata, toplabel);
#endif

  int em = eptr->get_execution_mode();
  gparams_.exec_mode = em;
  current_epoch_ = eptr->get_current_epoch();


  if(em == TRAIN) {
    if(full_train_prefetch_) {
      for(int i=0; i<gparams_.lookahead; i++) {
        for(int img=0; img<gparams_.batch_size; img++) {
          tempbuf_[i][img].Clear();
          tempbuf_[i][img].ParseFromString(train_cursor_->value());
          if(tempbuf_[i][img].channels() == 0)
            DecodeDatumNative(&(tempbuf_[i][img]));

#if 0 //def DEBUG
            printf("filename: %s label: %d\n",train_cursor_->key().c_str(), tempbuf_[i][img].label());
#endif

#ifdef USE_MLSL
          if(!split_db_)
            train_cursor_->Next(num_nodes_-1);
          else
            train_cursor_->Next();
#else
          train_cursor_->Next();
#endif
        }
      }
      ctrain_pf_mb_ += gparams_.lookahead;
      full_train_prefetch_ = false;
    }
    else {
      if(ctrain_pf_mb_ < train_batches_) {
        for(int img=0; img<gparams_.batch_size; img++) {
          int i = ctrain_pf_mb_ % gparams_.lookahead;

          tempbuf_[i][img].Clear();
          tempbuf_[i][img].ParseFromString(train_cursor_->value());
          if(tempbuf_[i][img].channels() == 0)
            DecodeDatumNative(&(tempbuf_[i][img]));

#if 0 //def DEBUG
            printf("filename: %s label: %d\n",train_cursor_->key().c_str(), tempbuf_[i][img].label());
#endif

#ifdef USE_MLSL
          if(!split_db_)
            train_cursor_->Next(num_nodes_-1);
          else
            train_cursor_->Next();
#else
          train_cursor_->Next();
#endif
        }
        ctrain_pf_mb_++;
      }
    }

#ifdef RETURNALL
    return;
#endif

    int mbslot = ctrain_proc_mb_ % gparams_.lookahead;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<gparams_.batch_size; i++)
      toplabel[i] = tempbuf_[mbslot][i].label();

    if(gparams_.crop_sizes[0] != gparams_.orig_sizes[0] && gparams_.crop_sizes[1] != gparams_.orig_sizes[1])
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<gparams_.batch_size; i++)
      {
        int r = tempbuf_[mbslot][i].height() - gparams_.crop_sizes[0] + 1;
        int c = tempbuf_[mbslot][i].width() - gparams_.crop_sizes[1] + 1;
        r_offset[i] = lrand48() % r;
        c_offset[i] = lrand48() % c;
        augmentation[i] = lrand48() % 12;
      }
    }

    trainImageTransform(tempbuf_[mbslot], topdata);

#ifdef GETSTATS
    int crop_img_size = gparams_.crop_sizes[0]*gparams_.crop_sizes[1]*gparams_.channels;
    MeanOfLayer("Data", topdata, gparams_.batch_size*crop_img_size);
#endif

    ctrain_proc_mb_++;
    if(ctrain_proc_mb_ == train_batches_)
    {
      ctrain_pf_mb_ = 0;
      ctrain_proc_mb_ = 0;
      full_train_prefetch_ = true;
    }
  }
  else if(em == TEST) {
    if(full_test_prefetch_) {
      for(int i=0; i<gparams_.lookahead; i++) {
        for(int img=0; img<gparams_.batch_size; img++) {
          tempbuf_[i][img].Clear();
          tempbuf_[i][img].ParseFromString(test_cursor_->value());
          if(tempbuf_[i][img].channels() == 0)
            DecodeDatumNative(&(tempbuf_[i][img]));

#if 0 //def DEBUG
          printf("filename: %s label: %d\n",test_cursor_->key().c_str(), tempbuf_[i][img].label());
#endif

#ifdef USE_MLSL
          if(!split_db_)
            test_cursor_->Next(num_nodes_-1);
          else
            test_cursor_->Next();
#else
          test_cursor_->Next();
#endif
        }
      }
      ctest_pf_mb_ += gparams_.lookahead;
      full_test_prefetch_ = false;
    }
    else
    {
      {
        if(ctest_pf_mb_ < test_batches_) {
          for(int img=0; img<gparams_.batch_size; img++) {
            int i = ctest_pf_mb_ % gparams_.lookahead;

            tempbuf_[i][img].Clear();
            tempbuf_[i][img].ParseFromString(test_cursor_->value());
            if(tempbuf_[i][img].channels() == 0)
              DecodeDatumNative(&(tempbuf_[i][img]));

#if 0 //def DEBUG
            printf("filename: %s label: %d\n",test_cursor_->key().c_str(), tempbuf_[i][img].label());
#endif

#ifdef USE_MLSL
            if(!split_db_)
              test_cursor_->Next(num_nodes_-1);
            else
              test_cursor_->Next();
#else
            test_cursor_->Next();
#endif
          }
          ctest_pf_mb_++;
        }
      }
    }

    int mbslot = ctest_proc_mb_ % gparams_.lookahead;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<gparams_.batch_size; i++)
      toplabel[i] = tempbuf_[mbslot][i].label();


    testImageTransform(tempbuf_[mbslot], curr_test_view_, topdata);

    curr_test_view_++;

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
}

