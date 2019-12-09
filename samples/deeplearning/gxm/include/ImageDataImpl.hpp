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
#include <vector>
#include <omp.h>
#include <math.h>
#include <limits.h>
#include <chrono>
#include "Engine.hpp"
#include "io.hpp"

#define RGB_FLATFILE 0
#define JPEG_FLATFILE 1
#define RGB_LMDB 2
#define JPEG_LMDB 3

using namespace std;

typedef struct
{
  bool mirror;
  bool vignette;
  bool color_bump;
} AugmentParams;

typedef struct
{
  string name;
  int height;
  int width;
  int length;
  int label;
} ImageInfo;

typedef struct
{
  int batch_size;
  int channels;
  vector<int> orig_sizes;
  vector<int> crop_sizes;
  vector<float> mean_values;
  vector<float> scale_values;
  int test_views;
  int lookahead;
  int threads;
  int exec_mode;
} DataImplParams;

class ImageDataImpl
{
  protected:
    DataImplParams *gp;
    AugmentParams *ap;
    unsigned int* tenSeeds_;
  public:
    ImageDataImpl(DataImplParams *gp_, AugmentParams *ap_): gp(gp_), ap(ap_)
    {
      tenSeeds_ = new unsigned int[gp->threads*16];
      initSeeds(tenSeeds_, gp->threads);
    }

    virtual void forwardPropagate(unsigned char *inp, float *outp) = 0;
    virtual void forwardPropagate(unsigned char *inp, int test_view, float *outp) = 0;
};

