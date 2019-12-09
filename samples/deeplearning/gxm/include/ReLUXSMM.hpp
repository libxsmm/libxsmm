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
#include "ReLUImpl.hpp"
#include "check.hpp"

class ReLUXSMM : public ReLUImpl
{
  protected:

  public:
    ReLUXSMM(ReLUImplParams* gp, int engine) : ReLUImpl(gp, engine)
    {
      top_layout_type = LIBXSMM_CUSTOM_LAYOUT;
      top_layout = NULL;
      gbot_layout_type = LIBXSMM_CUSTOM_LAYOUT;
      gbot_layout = NULL;
    }

    // Assume external threading, e.g., #pragma omp
    void forwardPropagate(TensorBuf *inp, TensorBuf *outp, int tid);
    void backPropagate(TensorBuf *inp, TensorBuf *deloutp, TensorBuf *delinp, int tid);
};
