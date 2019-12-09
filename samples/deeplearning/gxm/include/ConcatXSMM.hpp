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
#include "ConcatImpl.hpp"
#include "libxsmm.h"

class ConcatXSMM : public ConcatImpl
{
  public:
    ConcatXSMM(ConcatImplParams* gp, int engine) : ConcatImpl(gp, engine)
    {
      top_layout_type = LIBXSMM_CUSTOM_LAYOUT;
      top_layout = NULL;

      for(int n=0; n<gp->nInput.size(); n++)
      {
        gbot_layout_type.push_back(LIBXSMM_CUSTOM_LAYOUT);
        gbot_layout.push_back(NULL);
      }
    }

    void forwardPropagate(vector<TensorBuf*>& inp, TensorBuf* outp, int tid);
    void backPropagate(TensorBuf* deloutp, vector<TensorBuf*>& delinp, int tid);
    void convert_NCHW_to_NCHWV(float*, int, int, int, int, float*);
    void convert_NCHWV_to_NCHW(float*, int, int, int, int, float*);
};
