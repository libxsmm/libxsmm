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


#include "ImageDataRGBFlat.hpp"
#include <stdlib.h>
#include <stdio.h>

void ImageDataRGBFlat::processTrainMinibatch(unsigned char *inp, float *outp)
{
  int nImg = gp->batch_size;
  int nOfm = gp->channels;
  int ofh = gp->crop_sizes[0];
  int ofw = gp->crop_sizes[1];
  int ifh = gp->orig_sizes[0];
  int ifw = gp->orig_sizes[1];

  unsigned char (* __restrict input)[ifh][ifw][nOfm] = (unsigned char (*)[*][*][*])inp;
  float (* __restrict output)[nOfm][ofh][ofw] = (float (*)[*][*][*])outp;

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
  for(int img = 0; img < nImg; img++) {
    for(int ofm = 0; ofm < nOfm; ofm++) {
      for(int h = 0; h < ofh; h++) {
        for(int w = 0; w < ofw; w++) {

          int r_off = r_offset[img];
          int c_off = c_offset[img];

          if((augmentation[img] < 6) && (ap->mirror == true))
            output[img][ofm][h][ofw-w-1] = ((float)input[img][h+r_off][w+c_off][ofm] - gp->mean_values[ofm])*gp->scale_values[0];
          else
            output[img][ofm][h][w] = ((float)input[img][h+r_off][w+c_off][ofm] - gp->mean_values[ofm])*gp->scale_values[0];
        }
      }
    }
  }
}

void ImageDataRGBFlat::processTestMinibatch(unsigned char *inp, int tv, float *outp)
{
  int nImg = gp->batch_size;
  int nOfm = gp->channels;
  int ofh = gp->crop_sizes[0];
  int ofw = gp->crop_sizes[1];
  int ifh = gp->orig_sizes[0];
  int ifw = gp->orig_sizes[1];

  unsigned char (* __restrict input)[ifh][ifw][nOfm] = (unsigned char (*)[*][*][*])inp;
  float (* __restrict output)[nOfm][ofh][ofw] = (float (*)[*][*][*])outp;

  int tv2 = tv/2;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0; i<nImg; i++)
  {
    if(tv2 == 0) {
      r_offset[i] = (ifh - ofh)/2;
      c_offset[i] = (ifw - ofw)/2;
    }
    else if(tv2 == 1) {
      r_offset[i] = 0;
      c_offset[i] = 0;
    }
    else if(tv2 == 2) {
      r_offset[i] = 0;
      c_offset[i] = (ifw - ofw);
    }
    else if(tv2 == 3) {
      r_offset[i] = (ifh - ofw);
      c_offset[i] = 0;
    }
    else if(tv2 == 4) {
      r_offset[i] = ifh - ofh;
      c_offset[i] = ifw - ofw;
    }
  }

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
  for(int img = 0; img < nImg; img++) {
    for(int ofm = 0; ofm < nOfm; ofm++) {
      for(int h = 0; h < ofh; h++) {
        for(int w = 0; w < ofw; w++) {

          int r_off = r_offset[img];
          int c_off = c_offset[img];

          output[img][ofm][h][w] = ((float)input[img][h+r_off][w+c_off][ofm] - gp->mean_values[ofm])*gp->scale_values[0];
        }
      }
    }
  }
}

void ImageDataRGBFlat::forwardPropagate(unsigned char *inp, float *outp)
{
  int em = gp->exec_mode;
  assert(em == TRAIN);

  for(int i=0; i<gp->batch_size; i++)
  {
    r_offset[i] = lrand48() % (gp->orig_sizes[0] - gp->crop_sizes[0] + 1);
    c_offset[i] = lrand48() % (gp->orig_sizes[1] - gp->crop_sizes[1] + 1);
    augmentation[i] = lrand48() % 12;
  }

  processTrainMinibatch(inp, outp);
}

void ImageDataRGBFlat::forwardPropagate(unsigned char *inp, int tv, float *outp)
{
  int em = gp->exec_mode;
  assert(em == TEST);

  for(int i=0; i<gp->batch_size; i++)
  {
    r_offset[i] = lrand48() % (gp->orig_sizes[0] - gp->crop_sizes[0] + 1);
    c_offset[i] = lrand48() % (gp->orig_sizes[1] - gp->crop_sizes[1] + 1);
  }

  processTestMinibatch(inp, tv, outp);
}
