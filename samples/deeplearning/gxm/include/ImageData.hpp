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
#include "Node.hpp"
#include "Engine.hpp"
#include "Params.hpp"
#include "Tensor.hpp"
#include "proto/gxm.pb.h"
#include "Shape.h"
#include "ImageDataImpl.hpp"
#include "ImageDataRGBFlat.hpp"
#include "common.hpp"

#define SQUARE 1
#define RECT 2
#define RGB 3
#define GRAY 1

using namespace std;
using namespace gxm;

class ImageDataParams : public NNParams
{
  public:
    ImageDataParams(void) {}
    virtual ~ImageDataParams(void) {}

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

    void set_train_img_info(string s) { train_img_info_ = s; }
    string get_train_img_info() { return train_img_info_; }

    void set_num_test_files(int ntest) { num_test_files_ = ntest; }
    int get_num_test_files() { return num_test_files_; }

    void set_test_img_info(string s) { test_img_info_ = s; }
    string get_test_img_info() { return test_img_info_; }

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

    void set_jitters(int j) { jitters_.push_back(j); }

    vector<int>& get_jitters() { return jitters_; }

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

  protected:
    vector <int> jitters_, crop_sizes_, orig_sizes_;
    bool crop_image_;
    vector <float> mean_values_, scale_values_;
    int batch_size_, channels_, lookahead_;
    int num_train_files_, num_test_files_;
    int data_type_, label_dtype_, test_views_;
    float mean_, std;
    string train_source_, test_source_, train_img_info_, test_img_info_;
    bool mirror_, vignette_, color_bump_;
};

static MLParams* parseImageDataParams(NodeParameter* np)
{
  ImageDataParams* itp = new ImageDataParams();
  DataParameter dp = np->data_param();
  ImageTransformParameter pitp = np->data_param().image_xform_param();

  // Set name of node
  assert(!np->name().empty());
  itp->set_node_name(np->name());

  //Set node type (Convolution, FullyConnected, etc)
  assert(!np->type().empty());
  itp->set_node_type(np->type());

  //Set tensor names
  assert(np->bottom_size() == 0);

  for(int i=0; i<np->top_size(); i++)
  {
    assert(!np->top(i).empty());
    itp->set_top_names(np->top(i));
  }
  //Set backprop needed/not needed flag for this node
  itp->set_bprop_flag(np->propagate_down());

  //Set Mode for the node
  int mode = np->mode();
  assert((mode == TRAIN) || (mode == TEST));
  itp->set_mode(mode);

  // Get data source path
  assert(!(dp.train_source()).empty());
  itp->set_train_source_path(dp.train_source());

  assert(!(dp.test_source()).empty());
  itp->set_test_source_path(dp.test_source());

  // Get batch size
  assert(dp.batch_size() > 0);
  itp->set_batch_size(dp.batch_size());

  // Get lookahead
  assert(dp.lookahead() > 0);
  itp->set_lookahead(dp.lookahead());

  // Get data types
  itp->set_data_type(dp.data_type());
  itp->set_label_data_type(dp.label_data_type());

  // Get number of input files
  if((mode == TRAIN))
  {
    assert((dp.num_train_files() > 0) && (dp.num_test_files() > 0));
    itp->set_num_train_files(dp.num_train_files());
    itp->set_num_test_files(dp.num_test_files());
    itp->set_test_img_info(dp.test_data_info());
    itp->set_num_test_views(pitp.test_views());

    // Get data info file names
    itp->set_train_img_info(dp.train_data_info());
  }
  else if(mode == TEST)
  {
    assert(dp.num_test_files() > 0);
    itp->set_num_test_files(dp.num_test_files());
    itp->set_test_img_info(dp.test_data_info());
    itp->set_num_test_views(pitp.test_views());
  }

  // If cropping is turned on, set the crop size
  if(pitp.crop_image() == false)
    itp->set_crop_image(false);
  else
  {
    itp->set_crop_image(true);
    int cdims = pitp.crop_size_size();
    if(cdims > 0)
      itp->set_crop_sizes(SQUARE, 2, pitp.crop_size(0));
    else
    {
      int ch = pitp.crop_h();
      int cw = pitp.crop_w();
      assert((ch > 0) && (cw > 0));
      itp->set_crop_sizes(RECT, ch, cw);
    }
  }

  int odims = pitp.orig_size_size();
  if(odims > 0)
    itp->set_orig_sizes(SQUARE, 2, pitp.orig_size(0));
  else
  {
    int oh = pitp.orig_h();
    int ow = pitp.orig_w();
    assert((oh > 0) && (ow > 0));
    itp->set_orig_sizes(RECT, oh, ow);
  }

  int channels = pitp.channels();
  bool force_color = pitp.force_color();
  bool force_gray = pitp.force_gray();

  if(force_color) itp->set_channels(RGB);
  else if(force_gray) itp->set_channels(GRAY);
  else itp->set_channels(channels);

  if(pitp.mean_values_size() > 1)
    itp->set_mean_values(pitp.mean_values(0), pitp.mean_values(1), pitp.mean_values(2));
  else
    itp->set_mean_values(channels, pitp.mean_values(0));

  if(pitp.scale_values_size() > 1)
    itp->set_scale_values(pitp.scale_values(0), pitp.scale_values(1), pitp.scale_values(2));
  else
    itp->set_scale_values(channels, pitp.scale_values(0));

  bool mirror = pitp.mirror();
  bool vignette = pitp.vignette();
  bool color_bump = pitp.color_bump();

  itp->set_transform_params(mirror, vignette, color_bump);

  for(int i=0; i<pitp.jitters_size(); i++)
    itp->set_jitters(pitp.jitters(i));

  return itp;
}

class ImageDataNode : public NNNode
{
  public:

    ImageDataNode(ImageDataParams* p, MLEngine* e);
    ~ImageDataNode() {}

  protected:
    vector <Tensor*> tenTop_;
    vector <TensorBuf*> tenTopData_;
    int t_files_, v_files_, n_files_;
    int tfiles_per_mc_, vfiles_per_mc_;
    int current_epoch_, ctrain_pf_mb_, ctest_pf_mb_;
    int ctrain_proc_mb_, ctest_proc_mb_, curr_test_view_;
    int train_batches_, test_batches_;
    bool full_train_prefetch_, full_test_prefetch_;
    unsigned char *tempbuf_;
    DataImplParams gparams_;
    int* r_offset, *c_offset, *augmentation;

    MLEngine* eptr;

    string train_source_path_, test_source_path_;
    int num_epochs_, batch_size_, global_batch_size_;
    int num_train_files_, num_test_files_, num_machines_;
    int global_node_id_;
    int max_ep_, io_ep_, iopass_;
    vector<int> train_imginfo_index_, test_imginfo_index_;
    vector<int> labels_;

    AugmentParams ap;

    vector<int> jitters_;
    vector<ImageInfo> train_list_;
    vector<ImageInfo> test_list_;

    void shape_setzero(Shape* s)
    {
      for(int i=0; i<MAX_DIMS; i++)
        s->dims[i] = 0;
    }

    void configure(int dataType);
    void setupTrainIndices();
    void setupTestIndices();
    void createImageList(vector<ImageInfo>&, string, int);
#if 0
    MPI_Request *req_;
#endif
    ImageDataImpl *impl;
    void forwardPropagate();
};
