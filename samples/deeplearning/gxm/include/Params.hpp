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
#include <assert.h>
#include "Shape.h"
#include "proto/gxm.pb.h"

using namespace std;
using namespace gxm;

class MLParams
{
  protected:

  public:

    MLParams(void) {}

    virtual ~MLParams(void) {}
};

