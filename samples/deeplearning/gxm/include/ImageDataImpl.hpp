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

