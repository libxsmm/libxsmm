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


#include "JitterData.hpp"

using namespace std;
using namespace gxm;

JitterDataNode::JitterDataNode(JitterDataParams* p, MLEngine* e) : NNNode(p, e)
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

      if(vc.size() > 0) {
        tts.dims[2] = vc[0];
        tts.dims[3] = vc[1];
      }
      else {
        tts.dims[2] = vo[0];
        tts.dims[3] = vo[1];
      }

      tenTop_[i]->setShape(&tts);

      long long int size = tts.dims[0] * tts.dims[1];

      bool phys_padding = p->get_physical_padding();
      if(phys_padding)
      {
        gparams_.pad_h = p->get_pad_h();
        gparams_.pad_w = p->get_pad_w();
        size = size * (tts.dims[2] + 2*gparams_.pad_h) * (tts.dims[3] + 2*gparams_.pad_w);
      }
      else {
        gparams_.pad_h = 0;
        gparams_.pad_w = 0;
        size = size * tts.dims[2] * tts.dims[3];
      }

      if(dtype == DT_FLOAT)
        size = size*sizeof(float);
      else if(dtype == DT_BF16)
        size = size*sizeof(float);

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
  train_list_path_ = p->get_train_list_path();
  test_list_path_ = p->get_test_list_path();
  numsplits_ = p->get_numsplits();
  num_machines_ = e->get_num_machines();
  num_epochs_ = e->get_num_epochs();
  duplicates_ = num_machines_ == 1 ? 1 : num_machines_/numsplits_;

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

  gparams_.scalejittering_min = p->get_jitter_min();
  gparams_.scalejittering_max = p->get_jitter_max();
  gparams_.min_percent_area = p->get_percent_min_area();
  gparams_.max_percent_area = p->get_percent_max_area();
  gparams_.min_aspect_ratio = p->get_min_aspect_ratio();
  gparams_.max_aspect_ratio = p->get_max_aspect_ratio();
  gparams_.test_smaller_side = p->get_test_smaller_side();

  gparams_.shuffle = p->get_shuffle_flag();

  current_epoch_ = 0;
  ctrain_pf_mb_ = 0;
  ctest_pf_mb_ = 0;
  ctrain_proc_mb_ = 0;
  ctest_proc_mb_ = 0;
  curr_test_view_ = 0;
  full_train_prefetch_ = true;
  full_test_prefetch_ = true;

  eptr_ = e;

  global_node_id_ = e->get_global_node_id();

  tempbuf_.resize(gparams_.lookahead);
  cropbuf_.resize(gparams_.lookahead);
  for(int i=0; i < gparams_.lookahead; i++) {
    tempbuf_[i].resize(gparams_.batch_size);
    cropbuf_[i].resize(gparams_.batch_size);
  }

  if(mode == TRAIN)
  {
    num_train_files_ = p->get_num_train_files();
    int factor = batch_size_ * duplicates_;
    train_batches_ = (num_train_files_ + factor - 1)/factor;
    train_files_ = train_batches_ * factor;
    train_files_per_mc_ = train_files_/duplicates_;
    train_list_per_mc_.resize(train_files_per_mc_);

    e->set_num_train_batches(train_batches_);

    num_test_files_ = p->get_num_test_files();
    test_batches_ = (num_test_files_ + factor - 1)/factor;
    test_files_ = test_batches_ * factor;
    test_files_per_mc_ = test_files_/duplicates_;
    test_list_per_mc_.resize(test_files_per_mc_);

    e->set_num_test_batches(test_batches_);
    e->set_num_test_views(gparams_.test_views);
  }
  else if(mode == TEST)
  {
    int factor = batch_size_ * duplicates_;
    num_test_files_ = p->get_num_test_files();
    test_batches_ = (num_test_files_ + factor - 1)/factor;
    test_files_ = test_batches_ * factor;
    test_files_per_mc_ = test_files_/duplicates_;
    test_list_per_mc_.resize(test_files_per_mc_);

    e->set_num_test_batches(test_batches_);
    e->set_num_test_views(gparams_.test_views);
  }

#ifdef USE_MLSL
  MLSL::Session *s = e->get_session();
  s->SetGlobalMinibatchSize(global_batch_size_);
#endif

  r_offset = new long long int[gparams_.batch_size*60];
  c_offset = new long long int[gparams_.batch_size*60];
  augmentation = new int[gparams_.batch_size];
  drand1 = new double[gparams_.batch_size*60];
  drand2 = new double[gparams_.batch_size*60];
  drand3 = new double[gparams_.batch_size*60];

  if(mode == TRAIN)
    setupTrainIndices();
  setupTestIndices();
  labels_.resize(gparams_.lookahead);
  for(int i=0; i < gparams_.lookahead; i++)
    labels_[i].resize(gparams_.batch_size);
}

void JitterDataNode::setupTrainIndices()
{
  std::ifstream infile(train_list_path_.c_str());
  string line;
  train_list_.resize(num_train_files_);
  int idx=0;
  while (std::getline(infile, line) && idx < num_train_files_) {
    size_t pos = line.find_last_of(' ');
    int label = atoi(line.substr(pos + 1).c_str());
    train_list_[idx] = std::make_pair(line.substr(0, pos), label);
    idx++;
  }
#ifdef USE_MLSL
  size_t node_id = MLSL::Environment::GetEnv().GetProcessIdx();
  if(node_id == 0)
#endif
    printf("Read %d training filenames from list\n",idx);

  train_file_index_.resize(train_files_);

  for(int i=0; i<train_files_; i++)
  {
    if(i >= num_train_files_)
      train_file_index_[i] = train_file_index_[i-num_train_files_];
    else
      train_file_index_[i] = i;
  }

  srand(727);
}

void JitterDataNode::setupTestIndices()
{
  std::ifstream infile(test_list_path_.c_str());
  string line;
  int idx=0;

  test_list_.resize(num_test_files_);

  while (std::getline(infile, line) && idx < num_test_files_) {
    size_t pos = line.find_last_of(' ');
    int label = atoi(line.substr(pos + 1).c_str());
    test_list_[idx] = std::make_pair(line.substr(0, pos), label);
    idx++;
  }

  test_file_index_.resize(test_files_);

  for(int i=0; i<test_files_; i++)
  {
    if(i >= num_test_files_)
      test_file_index_[i] = test_file_index_[i-num_test_files_];
    else
      test_file_index_[i] = i;
  }
  idx = global_node_id_%duplicates_;
  for(int n=0; n<test_files_per_mc_; n++)
    test_list_per_mc_[n] = test_file_index_[n*duplicates_ + idx];
}

void JitterDataNode::cropVGG(const cv::Mat& cv_img, cv::Mat& cv_cropped_img, int *h_off, int *w_off)
{
  int img_channels = cv_img.channels();
  int img_height = cv_img.rows;
  int img_width = cv_img.cols;
  cv::Mat jittered_cv_img;
#ifdef _OPENMP
  int ridx = omp_get_thread_num();
#else
  int ridx = 0;
#endif

  cv_cropped_img = cv_img;

  if(gparams_.crop_sizes[0] == img_height && gparams_.crop_sizes[1] == img_width)
    return;

  const int scalejittering_min = gparams_.scalejittering_min;
  const int scalejittering_max = gparams_.scalejittering_max;
  int curr_scalejittering = 0, jittered_img_width = 0, jittered_img_height = 0;

  assert(scalejittering_min >= gparams_.crop_sizes[0]);
  assert(scalejittering_max >= gparams_.crop_sizes[0]);

  if(eptr_->get_execution_mode() == TRAIN) {
    curr_scalejittering = r_offset[ridx] % (scalejittering_max-scalejittering_min+1) + scalejittering_min;
    ridx++;
  }
  else
    curr_scalejittering = (scalejittering_max+scalejittering_min)/2;

  if(img_height < img_width) {
    jittered_img_height = curr_scalejittering;
    jittered_img_width = (int)((float)jittered_img_height*(float)img_width/(float)img_height);
  }
  else {
    jittered_img_width = curr_scalejittering;
    jittered_img_height = (int)((float)jittered_img_width*(float)img_height/(float)img_width);
  }
  cv::resize( cv_img, jittered_cv_img, cv::Size(jittered_img_width, jittered_img_height), 0 , 0, cv::INTER_CUBIC  );
  img_height = jittered_cv_img.rows;
  img_width = jittered_cv_img.cols;

  //    /* We only do random crop when we do training.*/
  if (eptr_->get_execution_mode() == TRAIN) {
    *h_off = r_offset[ridx] % (img_height - gparams_.crop_sizes[0] + 1);
    *w_off = c_offset[ridx] % (img_width - gparams_.crop_sizes[1] + 1);
    ridx++;
  } else {
    *h_off = (img_height - gparams_.crop_sizes[0]) / 2;
    *w_off = (img_width - gparams_.crop_sizes[1]) / 2;
  }
  cv::Rect roi(*w_off, *h_off, gparams_.crop_sizes[1], gparams_.crop_sizes[0]);
  cv_cropped_img = jittered_cv_img(roi);
}

void JitterDataNode::cropTorch(const cv::Mat& cv_img, cv::Mat& cv_cropped_img, int *h_off, int *w_off)
{
  int img_channels = cv_img.channels();
  int img_height = cv_img.rows;
  int img_width = cv_img.cols;
  float min_percent_area = gparams_.min_percent_area;
  float max_percent_area = gparams_.max_percent_area;
  float min_aspect_ratio = gparams_.min_aspect_ratio;
  float max_aspect_ratio = gparams_.max_aspect_ratio;

#ifdef _OPENMP
  int ridx=omp_get_thread_num()*60;
  int didx=omp_get_thread_num()*60;
#else
  int ridx=0;
  int didx=0;
#endif

  cv_cropped_img = cv_img;

  if(gparams_.crop_sizes[0] == img_height && gparams_.crop_sizes[1] == img_width)
    return;

  if(eptr_->get_execution_mode() == TRAIN)
  {
    float area = img_height*img_width;

    for(int attempt = 0; attempt < 60; attempt++) {
      float target_area = ((max_percent_area-min_percent_area)*((float)drand1[didx]) + min_percent_area)*area;
      float aspect_ratio = ((max_aspect_ratio-min_aspect_ratio)*((float)drand2[didx]) + min_aspect_ratio);

      int tmp_w = 0, tmp_h = 0;
      tmp_w = round(sqrt(target_area * aspect_ratio));
      tmp_h = round(sqrt(target_area /aspect_ratio));

      if((float)drand3[didx] < 0.5) {
        tmp_w += tmp_h;
        tmp_h = tmp_w - tmp_h;
        tmp_w -= tmp_h;
      }
      didx++;

      if( tmp_w < img_width && tmp_h < img_height) {
        int rw = img_width - tmp_w + 1;
        int rh = img_height - tmp_h + 1;
        *w_off = c_offset[ridx] % rw;
        *h_off = r_offset[ridx] % rh;
        ridx++;

        cv::Rect roi(*w_off, *h_off, tmp_w, tmp_h);
        cv_cropped_img = cv_img(roi);
        cv::resize(cv_cropped_img, cv_cropped_img, cv::Size(gparams_.crop_sizes[0], gparams_.crop_sizes[1]), 0 , 0, cv::INTER_CUBIC );
        return;
      }
    }

    // Fall back
    printf("falling back to VGG jittering method\n");
    cropVGG(cv_img, cv_cropped_img, h_off, w_off);
  }
  else {
    int jittered_img_width = 0, jittered_img_height = 0;
    int curr_scalejittering = gparams_.test_smaller_side;
    cv::Mat jittered_cv_img;

    assert(curr_scalejittering >= gparams_.crop_sizes[0]);

    if(img_height < img_width) {
      jittered_img_height = curr_scalejittering;
      jittered_img_width = (int)((float)jittered_img_height*(float)img_width/(float)img_height);
    }
    else {
      jittered_img_width = curr_scalejittering;
      jittered_img_height = (int)((float)jittered_img_width*(float)img_height/(float)img_width);
    }
    cv::resize( cv_img, jittered_cv_img, cv::Size(jittered_img_width, jittered_img_height), 0 , 0, cv::INTER_CUBIC );
    *h_off = (jittered_img_height - gparams_.crop_sizes[0]) / 2;
    *w_off = (jittered_img_width - gparams_.crop_sizes[1]) / 2;
    cv::Rect roi(*w_off, *h_off, gparams_.crop_sizes[1], gparams_.crop_sizes[0]);
    cv_cropped_img = jittered_cv_img(roi);
  }
}

void JitterDataNode::imageTransform(vector<cv::Mat>& vcrop, float* outp)
{
  int nImg = gparams_.batch_size;
  int nOfm = gparams_.channels;
  int ofh = gparams_.crop_sizes[0];
  int ofw = gparams_.crop_sizes[1];
  int padh = gparams_.pad_h;
  int padw = gparams_.pad_w;
  int ofhp = ofh + 2*padh;
  int ofwp = ofw + 2*padw;
  vector<float>& mean = gparams_.mean_values;
  vector<float> rscale;
  rscale.resize(gparams_.scale_values.size());

  for(int i=0; i < nOfm; i++)
    rscale[i] = 1./gparams_.scale_values[i];

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int img = 0; img < nImg; img++) {
    for(int h=0; h < ofh; h++) {
      const unsigned char* ptr = vcrop[img].ptr<uchar>(h);
      int img_index = 0;
      for(int w = 0; w < ofw; w++) {
        for(int ofm = 0; ofm < nOfm; ofm++) {

          //assert(vcrop[img].channels() == nOfm);
          int out_idx;

          int oh = h+padh;
          int ow = w+padw;

          float inp;
          int fm;
          if(ofm < 3)
          {
            inp = static_cast<float>(ptr[img_index++]);
            fm = (gparams_.scale_values.size() == 1) ? 0 : ofm;
          }

          if(gparams_.exec_mode == TRAIN)
          {
            if((augmentation[img] < 6) && (ap.mirror == true))
              out_idx = img * ofhp * ofwp * nOfm + oh * ofwp * nOfm + (ofwp-ow-1) * nOfm + ofm;
            else
              out_idx = img * ofhp * ofwp * nOfm + oh * ofwp * nOfm + ow * nOfm + ofm;
          }
          else
            out_idx = img * ofhp * ofwp * nOfm + oh * ofwp * nOfm + ow * nOfm + ofm;

          if(ofm == 3)
            outp[out_idx] = 0.0;
          else
            outp[out_idx] = (inp - mean[ofm]) * rscale[fm];
        }
      }
    }
  }
}

int myrandom (int i) { return std::rand()%i;}

/* it's fine to alias in and out */
void JitterDataNode::convert_f32_bf16(float* in, libxsmm_bfloat16* out, unsigned int len) {

  unsigned int i = 0;

#pragma omp parallel for private(i)
  for ( i = 0; i < len; i+=16 ) {
    __m512  vfp32  = gxm_fp32_to_bfp16_rne_adjustment_avx512f(_mm512_loadu_ps(in + i));
    __m256i vbfp16 = gxm_fp32_to_bfp16_truncate_avx512f(vfp32);
    _mm256_storeu_si256( (__m256i*)(out+i), vbfp16 );
  }
}

void JitterDataNode::forwardPropagate()
{
  int nImg = gparams_.batch_size;
  int nOfm = gparams_.channels;
  int ofh = gparams_.crop_sizes[0];
  int ofw = gparams_.crop_sizes[1];
  int padh = gparams_.pad_h;
  int padw = gparams_.pad_w;
  int ofhp = ofh + 2*padh;
  int ofwp = ofw + 2*padw;
  int out_dtype = tenTopData_[0]->getDataType();

  float *topdata = (float*)(tenTopData_[0]->getBuffer());
  int* toplabel = (int*)(tenTopData_[1]->getBuffer());

  if(first_fp)
  {
    int size = nImg * nOfm * ofhp *ofwp;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<size; i++)
      topdata[i] = 0.0f;

    first_fp = false;
  }

  int em = eptr_->get_execution_mode();
  gparams_.exec_mode = em;
  current_epoch_ = eptr_->get_current_epoch();

  if(em == TRAIN) {
    if(full_train_prefetch_) {
      if(gparams_.shuffle)
        random_shuffle(train_file_index_.begin(), train_file_index_.end(), myrandom);

      int idx = global_node_id_ % duplicates_;
      for(int n=0; n<train_files_per_mc_; n++)
        train_list_per_mc_[n] = train_file_index_[n*duplicates_ + idx];

      for(int i=0; i<gparams_.lookahead; i++) {
        for(int img=0; img<gparams_.batch_size; img++) {
          int idx = i*gparams_.batch_size + img;
          int fileidx = train_list_per_mc_[idx];
          string path = train_source_path_ + "/" + train_list_[fileidx].first;
          tempbuf_[i][img] = cv::imread(path, true);
          if(tempbuf_[i][img].empty()) {
            printf("Null data read from %s.. exiting\n",path.c_str());
            exit(1);
          }

          labels_[i][img] = train_list_[fileidx].second;
        }
      }
      ctrain_pf_mb_ += gparams_.lookahead;
      full_train_prefetch_ = false;
    }
    else {
      if(ctrain_pf_mb_ < train_batches_) {
        int mbs = ctrain_pf_mb_ % gparams_.lookahead;

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int img=0; img<gparams_.batch_size; img++) {
          int idx = ctrain_pf_mb_ * gparams_.batch_size + img;
          int fileidx = train_list_per_mc_[idx];
          string path = train_source_path_ + "/" + train_list_[fileidx].first;
          tempbuf_[mbs][img] = cv::imread(path, true);
          if(!tempbuf_[mbs][img].data) {
            printf("Null data read from %s.. exiting\n",path.c_str());
            exit(1);
          }
          labels_[mbs][img] = train_list_[fileidx].second;
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
      toplabel[i] = labels_[mbslot][i];

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<gparams_.batch_size; i++)
    {
      for(int attempts=0; attempts<60; attempts++) {
        r_offset[i*60 + attempts] = lrand48();
        c_offset[i*60 + attempts] = lrand48();
        drand1[i*60 + attempts] = drand48();
        drand2[i*60 + attempts] = drand48();
        drand3[i*60 + attempts] = drand48();
      }
      augmentation[i] = lrand48() % 12;
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<gparams_.batch_size; i++)
    {
      int r_off, c_off;
      cropTorch(tempbuf_[mbslot][i], cropbuf_[mbslot][i], &r_off, &c_off);
    }

    imageTransform(cropbuf_[mbslot], topdata);

#ifndef NDEBUG
    if(gparams_.pad_h && gparams_.pad_w)
      check_physical_pad(nname_.c_str(), topdata, nImg, 1, ofh, ofw, nOfm, gparams_.pad_h, gparams_.pad_w);
#endif

    int crop_img_size = nImg * ofhp * ofwp * nOfm;

    if(out_dtype == DT_BF16)
    {
      if(bf16_img == NULL)
      {
        bf16_img = _mm_malloc(crop_img_size*sizeof(libxsmm_bfloat16), 64);
        tenTopData_[0]->setLPBuffer(bf16_img);
      }
      convert_f32_bf16(topdata, (libxsmm_bfloat16*)bf16_img, crop_img_size);
    }

#ifdef GETSTATS
    if(global_node_id_ == 0)
    {
      MeanOfLayer("Data", topdata, crop_img_size);
      MeanOfLayer("Labels", toplabel, gparams_.batch_size);
    }
#endif

    ctrain_proc_mb_++;
    if(ctrain_proc_mb_ == train_batches_)
    {
      ctrain_pf_mb_ = 0;
      ctrain_proc_mb_ = 0;
      full_train_prefetch_ = true;
    }
  }
  else if(em == TEST || em == VAL) {
    if(full_test_prefetch_) {
      for(int i=0; i<gparams_.lookahead; i++) {
        for(int img=0; img<gparams_.batch_size; img++) {
          int idx = i*gparams_.batch_size + img;
          int fileidx = test_list_per_mc_[idx];
          string path = test_source_path_ + "/" + test_list_[fileidx].first;

          tempbuf_[i][img] = cv::imread(path, true);
          if(tempbuf_[i][img].empty()) {
            printf("Null data read from %s.. exiting\n",path.c_str());
            exit(1);
          }
          labels_[i][img] = test_list_[fileidx].second;
        }
      }
      ctest_pf_mb_ += gparams_.lookahead;
      full_test_prefetch_ = false;
    }
    else
    {
      {
        if(ctest_pf_mb_ < test_batches_) {
          int i = ctest_pf_mb_ % gparams_.lookahead;

#ifdef _OPENMP
#pragma omp parallel for
#endif
          for(int img=0; img<gparams_.batch_size; img++) {
            int idx = ctest_pf_mb_ * gparams_.batch_size + img;
            int fileidx = test_list_per_mc_[idx];
            string path = test_source_path_ + "/" + test_list_[fileidx].first;

            tempbuf_[i][img] = cv::imread(path, true);
            if(tempbuf_[i][img].empty()) {
              printf("Null data read from %s... exiting\n",path.c_str());
              exit(1);
            }
            labels_[i][img] = test_list_[fileidx].second;
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
      toplabel[i] = labels_[mbslot][i];

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<gparams_.batch_size; i++)
    {
      int r_off, c_off;
      cropTorch(tempbuf_[mbslot][i], cropbuf_[mbslot][i], &r_off, &c_off);
    }

    imageTransform(cropbuf_[mbslot], topdata);

    int crop_img_size = nImg * ofhp * ofwp * nOfm;

    if(out_dtype == DT_BF16)
    {
      if(bf16_img == NULL)
      {
        bf16_img = _mm_malloc(crop_img_size*sizeof(libxsmm_bfloat16), 64);
        tenTopData_[0]->setLPBuffer(bf16_img);
      }
      convert_f32_bf16(topdata, (libxsmm_bfloat16*)bf16_img, crop_img_size);
    }

    ctest_proc_mb_++;
    if(ctest_proc_mb_ == test_batches_)
    {
      ctest_pf_mb_ = 0;
      ctest_proc_mb_ = 0;
      full_test_prefetch_ = true;
    }
  }
}

