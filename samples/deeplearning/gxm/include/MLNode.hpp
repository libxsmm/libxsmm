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
#include "Params.hpp"
#include "MLNode.fwd.hpp"
#include "Engine.fwd.hpp"

using namespace std;
using namespace gxm;

class MLNode
{
  protected:

  public:

    MLNode(MLParams* p, MLEngine* e) {}

    virtual ~MLNode(void) {}

    virtual void createStrategy(int) {}
    virtual int executeTask(int) {return 0;}
    virtual void enqueTask(int pos) {}
    virtual void createCheckPoint() {}
    virtual void restoreCheckPoint() {}
    virtual void createPersistentTask() {}
};


// Constructor should create Tensors for its output and internal buffers and assign type to it

template <typename NType, typename PType>
MLNode *CreateMLNode(MLParams *param, MLEngine *engine)
{
  NType *obj = new NType(dynamic_cast<PType*>(param), engine);
  return dynamic_cast<MLNode*>(obj);
}
