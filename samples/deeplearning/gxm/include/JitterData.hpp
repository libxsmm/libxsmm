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


#pragma once
#include <string>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <limits.h>
#include <omp.h>
#include <unistd.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include "Node.hpp"
#include "Engine.hpp"
#include "Params.hpp"
#include "Tensor.hpp"
#include "proto/gxm.pb.h"
#include "Shape.h"
#include "io.hpp"
#include "check.hpp"
#include "common.hpp"

#define SQUARE 1
#define RECT 2
#define RGB 3
#define GRAY 1

using namespace std;
using namespace gxm;

typedef struct
{
  bool mirror;
  bool vignette;
  bool color_bump;
} JitterAugmentParams;

typedef struct
{
  int batch_size;
  int channels;
  vector<int> orig_sizes;
  vector<int> crop_sizes;
  int pad_w, pad_h;
  vector<float> mean_values;
  vector<float> scale_values;
  int scalejittering_min, scalejittering_max;
  int test_smaller_side;
  float min_percent_area, max_percent_area;
  float min_aspect_ratio, max_aspect_ratio;
  bool shuffle;
  string mean_file;
  int test_views;
  int lookahead;
  int threads;
  int exec_mode;
} JitterDataImplParams;

class JitterDataParams : public NNParams
{
  public:
    JitterDataParams(void) {}
    virtual ~JitterDataParams(void) {}

    void set_transform_params(bool mirror, bool vignette, bool color_bump)
    {
      mirror_ = mirror;
      vignette_ = vignette;
      color_bump_ = color_bump;
    }

    bool get_mirror() { return mirror_; }
    bool get_vignette() { return vignette_; }
    bool get_color_bump() { return color_bump_; }

    void set_train_source_path(string source_name) { train_source_ = source_name; }
    string get_train_source_path() { return train_source_; }

    void set_test_source_path(string source_name) { test_source_ = source_name; }
    string get_test_source_path() { return test_source_; }

    void set_train_list_path(string tr) { train_list_ = tr; }
    string get_train_list_path() { return train_list_; }

    void set_test_list_path(string te) { test_list_ = te; }
    string get_test_list_path() { return test_list_; }

    void set_numsplits(int s) {numsplits_ = s; }
    int get_numsplits() { return numsplits_; }

    void set_shuffle_flag(bool f) {shuffle_ = f; }
    bool get_shuffle_flag() { return shuffle_; }

    void set_data_type(int t) { data_type_ = t; }
    int get_data_type() { return data_type_; }

    void set_label_data_type(int t) { label_dtype_ = t; }
    int get_label_data_type() { return label_dtype_; }

    void set_batch_size(int batch) { batch_size_ = batch; }
    int get_batch_size() { return batch_size_; }

    void set_lookahead(int l) { lookahead_ = l; }
    int get_lookahead() { return lookahead_; }

    void set_num_train_files(int ntrain) { num_train_files_ = ntrain; }
    int get_num_train_files() { return num_train_files_; }

    void set_num_test_files(int ntest) { num_test_files_ = ntest; }
    int get_num_test_files() { return num_test_files_; }

    void set_mean_values(int channels, float mean_val)
    {
      for(int i=0; i<channels; i++)
        mean_values_.push_back(mean_val);
    }

    void set_mean_values(float m1, float m2, float m3)
    {
      mean_values_.push_back(m1);
      mean_values_.push_back(m2);
      mean_values_.push_back(m3);
    }

    vector<float>& get_mean_values() { return mean_values_; }

    void set_mean_file(string n) { mean_file_ = n; }
    string get_mean_file() { return mean_file_; }

    void set_scale_values(int channels, float std_val)
    {
      for(int i=0; i<channels; i++)
        scale_values_.push_back(std_val);
    }
    void set_scale_values(float s1, float s2, float s3)
    {
      scale_values_.push_back(s1);
      scale_values_.push_back(s2);
      scale_values_.push_back(s3);
    }

    vector<float>& get_scale_values() { return scale_values_; }

    void set_channels(int c) { channels_ = c; }
    int get_channels() { return channels_; }

    void set_crop_image(bool crop) { crop_image_ = crop; }
    bool get_crop_image() { return crop_image_; }

    void set_crop_sizes(int s, int v1, int v2)
    {
      if(s == SQUARE)
      {
        for(int i=0; i<v1; i++)
          crop_sizes_.push_back(v2);
      }
      else if(s == RECT)
      {
        crop_sizes_.push_back(v1);
        crop_sizes_.push_back(v2);
      }
    }

    vector<int>& get_crop_sizes() { return crop_sizes_; }

    void set_physical_padding(bool p) {phys_pad_ = p; }
    bool get_physical_padding() { return phys_pad_; }

    void set_pad_h(int h) { pad_h_ = h; }
    int get_pad_h() {return pad_h_; }

    void set_pad_w(int w) { pad_w_ = w; }
    int get_pad_w() {return pad_w_; }

    void set_orig_sizes(int s, int v1, int v2)
    {
      if(s == SQUARE)
      {
        for(int i=0; i<v1; i++)
          orig_sizes_.push_back(v2);
      }
      else if(s == RECT)
      {
        orig_sizes_.push_back(v1);
        orig_sizes_.push_back(v2);
      }
    }

    vector<int>& get_orig_sizes() { return orig_sizes_; }

    void set_num_test_views(int nt) { test_views_ = nt; }
    int get_num_test_views() { return test_views_; }

    void set_scale_jitters(int sjmin, int sjmax)
    {
      sjmin_ = sjmin;
      sjmax_ = sjmax;
    }
    int get_jitter_min() { return sjmin_; }
    int get_jitter_max() { return sjmax_; }

    void set_percent_areas(float amin, float amax)
    {
      pc_amin_ = amin;
      pc_amax_ = amax;
    }
    float get_percent_min_area() { return pc_amin_; }
    float get_percent_max_area() { return pc_amax_; }

    void set_aspect_ratios(float armin, float armax)
    {
      ar_min_ = armin;
      ar_max_ = armax;
    }
    float get_min_aspect_ratio() { return ar_min_; }
    float get_max_aspect_ratio() { return ar_max_; }

    void set_test_smaller_side(int s) { test_smaller_side_ = s; }
    int get_test_smaller_side() {return test_smaller_side_; }

    void set_compute_engine(int e) {compute_engine_ = e; }
    int get_compute_engine() {return compute_engine_; }

  protected:
    vector <int> crop_sizes_, orig_sizes_;
    int pad_h_, pad_w_;
    bool crop_image_, phys_pad_;
    vector <float> mean_values_, scale_values_;
    int batch_size_, channels_, lookahead_, numsplits_;
    int num_train_files_, num_test_files_;
    int sjmin_, sjmax_, test_smaller_side_;
    int data_type_, label_dtype_, test_views_;
    int compute_engine_;
    float mean_, std, pc_amin_, pc_amax_;
    float ar_min_, ar_max_;
    string train_source_, test_source_;
    string mean_file_, train_list_, test_list_;
    bool mirror_, vignette_, color_bump_, shuffle_;
};

static MLParams* parseJitterDataParams(NodeParameter* np)
{
  JitterDataParams* jp = new JitterDataParams();
  DataParameter dp = np->data_param();
  ImageTransformParameter itp = np->data_param().image_xform_param();

  // Set name of node
  assert(!np->name().empty());
  jp->set_node_name(np->name());

  //Set node type (Convolution, FullyConnected, etc)
  assert(!np->type().empty());
  jp->set_node_type(np->type());

  //Set tensor names
  assert(np->bottom_size() == 0);

  for(int i=0; i<np->top_size(); i++)
  {
    assert(!np->top(i).empty());
    jp->set_top_names(np->top(i));
  }
  //Set backprop needed/not needed flag for this node
  jp->set_bprop_flag(np->propagate_down());

  //Set Mode for the node
  int mode = np->mode();
  assert((mode == TRAIN) || (mode == TEST));
  jp->set_mode(mode);

  // Get data source path
  assert(!(dp.train_source()).empty());
  jp->set_train_source_path(dp.train_source());

  assert(!(dp.test_source()).empty());
  jp->set_test_source_path(dp.test_source());

  // Get data list path
  assert(!(dp.train_list()).empty());
  jp->set_train_list_path(dp.train_list());

  assert(!(dp.test_list()).empty());
  jp->set_test_list_path(dp.test_list());

  // Get number of splits
  jp->set_numsplits(dp.numsplits());

  // Get shuffle flag
  jp->set_shuffle_flag(dp.shuffle());

  // Get batch size
  assert(dp.batch_size() > 0);
  jp->set_batch_size(dp.batch_size());

  // Get lookahead
  assert(dp.lookahead() > 0);
  jp->set_lookahead(dp.lookahead());

  // Get data types
  jp->set_data_type(dp.data_type());
  jp->set_label_data_type(dp.label_data_type());

  // Get number of input files
  if((mode == TRAIN))
  {
    assert((dp.num_train_files() > 0) && (dp.num_test_files() > 0));
    jp->set_num_train_files(dp.num_train_files());
    jp->set_num_test_files(dp.num_test_files());
    jp->set_num_test_views(itp.test_views());
  }
  else if(mode == TEST)
  {
    assert(dp.num_test_files() > 0);
    jp->set_num_test_files(dp.num_test_files());
    jp->set_num_test_views(itp.test_views());
  }

  // If cropping is turned on, set the crop size
  if(itp.crop_image() == false)
    jp->set_crop_image(false);
  else
  {
    jp->set_crop_image(true);
    int cdims = itp.crop_size_size();
    if(cdims > 0)
      jp->set_crop_sizes(SQUARE, 2, itp.crop_size(0));
    else
    {
      int ch = itp.crop_h();
      int cw = itp.crop_w();
      assert((ch > 0) && (cw > 0));
      jp->set_crop_sizes(RECT, ch, cw);
    }
  }

  int odims = itp.orig_size_size();
  if(odims > 0)
    jp->set_orig_sizes(SQUARE, 2, itp.orig_size(0));
  else
  {
    int oh, ow;
    if(itp.orig_h() > 0)
      oh = itp.orig_h();
    if(itp.orig_w() > 0)
      ow = itp.orig_w();
    jp->set_orig_sizes(RECT, oh, ow);
  }

  jp->set_pad_h(itp.pad_h());
  jp->set_pad_w(itp.pad_w());
  jp->set_physical_padding(itp.physical_padding());

  int channels = itp.channels();
  bool force_color = itp.force_color();
  bool force_gray = itp.force_gray();

  if(force_color) jp->set_channels(RGB);
  else if(force_gray) jp->set_channels(GRAY);
  else jp->set_channels(channels);

  if(itp.mean_values_size() > 1)
    jp->set_mean_values(itp.mean_values(0), itp.mean_values(1), itp.mean_values(2));
  else if(itp.mean_values_size() > 0)
    jp->set_mean_values(channels, itp.mean_values(0));
  else if(itp.mean_file().size() > 0)
    jp->set_mean_file(itp.mean_file());
  else
    jp->set_mean_values(channels, 0);

  if(itp.scale_values_size() > 1)
    jp->set_scale_values(itp.scale_values(0), itp.scale_values(1), itp.scale_values(2));
  else if(itp.scale_values_size() > 0)
    jp->set_scale_values(channels, itp.scale_values(0));
  else
    jp->set_scale_values(channels, 1);

  bool mirror = itp.mirror();
  bool vignette = itp.vignette();
  bool color_bump = itp.color_bump();

  jp->set_transform_params(mirror, vignette, color_bump);

  jp->set_scale_jitters(itp.scalejittering_min(), itp.scalejittering_max());
  jp->set_percent_areas(itp.min_percent_area(), itp.max_percent_area());
  jp->set_aspect_ratios(itp.min_aspect_ratio(), itp.max_aspect_ratio());
  jp->set_test_smaller_side(itp.test_smaller_side());

  jp->set_compute_engine(dp.engine());

  return jp;
}

class JitterDataNode : public NNNode
{
  public:

    JitterDataNode(JitterDataParams* p, MLEngine* e);
    ~JitterDataNode() {}

  protected:
    vector <Tensor*> tenTop_;
    vector <TensorBuf*> tenTopData_;
    int t_files_, v_files_, n_files_;
    int current_epoch_, ctrain_pf_mb_, ctest_pf_mb_;
    int ctrain_proc_mb_, ctest_proc_mb_, curr_test_view_;
    int train_batches_, test_batches_, numsplits_, duplicates_;
    bool full_train_prefetch_, full_test_prefetch_;
    long long int *r_offset, *c_offset;
    double *drand1, *drand2, *drand3;
    int *augmentation;
    float* mean_data_;
    bool first_fp=true;

    vector < vector<cv::Mat> > tempbuf_, cropbuf_;
    vector < vector<int> > labels_;
    JitterDataImplParams gparams_;

    MLEngine* eptr_;

    string train_source_path_, test_source_path_, train_list_path_, test_list_path_;
    vector<std::pair<std::string, int> > train_list_, test_list_;
    vector<int> train_file_index_, test_file_index_;
    vector<int> train_list_per_mc_, test_list_per_mc_;
    int num_epochs_, batch_size_, global_batch_size_;
    int num_train_files_, num_test_files_, num_machines_;
    int train_files_, test_files_, train_files_per_mc_, test_files_per_mc_;
    int global_node_id_, ridx_;
    void* bf16_img=NULL;

    JitterAugmentParams ap;

    vector<int> jitters_;

    void shape_setzero(Shape* s)
    {
      for(int i=0; i<MAX_DIMS; i++)
        s->dims[i] = 0;
    }

    void forwardPropagate();
    void cropTorch(const cv::Mat&, cv::Mat&, int*, int*);
    void cropVGG(const cv::Mat&, cv::Mat&, int*, int*);
    void imageTransform(vector<cv::Mat>&, float*);
    void setupTrainIndices();
    void setupTestIndices();
    void convert_f32_bf16(float* in, libxsmm_bfloat16* out, unsigned int len);
};
