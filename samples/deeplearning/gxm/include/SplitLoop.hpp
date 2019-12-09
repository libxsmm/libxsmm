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
#include "SplitImpl.hpp"

class SplitLoop : public SplitImpl
{
    public:
      SplitLoop(SplitImplParams* gp, int engine) : SplitImpl(gp, engine)
      {
          top_layout_type = NCHWV;
          top_layout = NULL;

          gbot_layout_type = NCHWV;
          gbot_layout = NULL;
      }

      void forwardPropagate(TensorBuf *inpb, vector<TensorBuf*>& outpb, int tid);
      void backPropagate(vector<TensorBuf*>& deloutpb, TensorBuf *delinpb, int tid);
};
