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
#include <omp.h>
#include <sys/time.h>
#include <limits.h>
#include "check.hpp"
#include "Tensor.hpp"
#include "common.hpp"

using namespace std;

typedef struct {
  string node_name;
  int nInput, nOutput;
  int batch_size;
  int nBInput, nBOutput;
  int iBlock, oBlock;
  float loss;
  float loss_weight;
  int num_threads;
} SMaxLossImplParams;

class SMaxLossImpl
{
  protected:
    SMaxLossImplParams *gp;
    size_t num_nodes;
  public:
    SMaxLossImpl(SMaxLossImplParams* gp_): gp(gp_) {}
    void set_num_nodes(size_t n) { num_nodes = n; }
    size_t get_num_nodes() { return num_nodes; }

    virtual void forwardPropagate(TensorBuf *inp, TensorBuf *label, TensorBuf *outp) = 0;
    virtual void backPropagate(TensorBuf *outp, TensorBuf *label, TensorBuf *delinp) = 0;
};
