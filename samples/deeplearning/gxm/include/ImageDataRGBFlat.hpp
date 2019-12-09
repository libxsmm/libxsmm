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
#include "ImageDataImpl.hpp"

class ImageDataRGBFlat : public ImageDataImpl
{
  protected:
    int *r_offset, *c_offset, *augmentation;

  public:
    ImageDataRGBFlat(DataImplParams *gp, AugmentParams *ap) : ImageDataImpl(gp, ap)
    {
      r_offset = new int[gp->batch_size];
      c_offset = new int[gp->batch_size];
      augmentation = new int[gp->batch_size];
    }

    void forwardPropagate(unsigned char *inp, float *outp);
    void forwardPropagate(unsigned char *inp, int test_view, float *outp);
    void processTrainMinibatch(unsigned char *inp, float *outp);
    void processTestMinibatch(unsigned char *inp, int tv, float *outp);
};
