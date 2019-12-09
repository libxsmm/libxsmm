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
#include "SoftmaxLossImpl.hpp"

class SMaxLossLoop : public SMaxLossImpl
{
  public:
    SMaxLossLoop(SMaxLossImplParams* gp) : SMaxLossImpl(gp) {}

    // Assume external threading, e.g., #pragma omp
    void forwardPropagate(TensorBuf *inp, TensorBuf* label, TensorBuf *outp);
    void backPropagate(TensorBuf *outp, TensorBuf* label, TensorBuf *delinp);
};
