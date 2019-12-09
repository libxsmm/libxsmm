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

#include <string.h>
#include <omp.h>
#include "FCImpl.hpp"
#include "libxsmm.h"

#define VLEN 16

#define CHKERR_LIBXSMM_DNN(A) if ( A != LIBXSMM_DNN_SUCCESS )\
{\
  fprintf(stdout, "%s, %s\n", gp->node_name.c_str(), libxsmm_dnn_get_error(A) );\
  fflush(stdout);\
}

class FCXSMM : public FCImpl
{
  protected:
    FCImpl *gp_;
    libxsmm_dnn_fullyconnected_desc fullyconnected_desc;
    libxsmm_dnn_fullyconnected* libxsmm_handle[NUM_NUMA_NODES];
    libxsmm_dnn_tensor*  libxsmm_input[NUM_NUMA_NODES]={NULL};
    libxsmm_dnn_tensor*  libxsmm_delinput[NUM_NUMA_NODES]={NULL};
    libxsmm_dnn_tensor*  libxsmm_output[NUM_NUMA_NODES]={NULL};
    libxsmm_dnn_tensor*  libxsmm_deloutput[NUM_NUMA_NODES]={NULL};
    libxsmm_dnn_tensor*  libxsmm_filter[NUM_NUMA_NODES]={NULL};
    libxsmm_dnn_tensor*  libxsmm_checkpoint_filter=NULL;
    libxsmm_dnn_tensor*  libxsmm_checkpoint_history_filter=NULL;
    libxsmm_dnn_tensor*  libxsmm_delfilter[NUM_NUMA_NODES]={NULL};
    libxsmm_dnn_tensor_datalayout* libxsmm_layout;
    libxsmm_dnn_err_t status;
    libxsmm_dnn_err_t global_status = LIBXSMM_DNN_SUCCESS;
    bool updated_scratch_fwd=false, updated_scratch_bwd=false, updated_scratch_upd=false;
    void *scratch[NUM_NUMA_NODES]={NULL};
    int prev_scratch_size = 0;

  public:
    FCXSMM(FCImplParams* gp, int engine);
    virtual ~FCXSMM(void) {}

    bool firstTimeFwd=true, firstTimeBwd=true;

    void forwardPropagate(TensorBuf *inp, TensorBuf *weightp, TensorBuf *hweightp, TensorBuf *biasp, TensorBuf *outp, int tid);
    void backPropagate(TensorBuf *deloutp, TensorBuf* weightp, TensorBuf *delinp, int tid);
    void weightUpdate(TensorBuf *deloutp, TensorBuf *inp, TensorBuf *delweightp, TensorBuf *delbiasp, int tid);
};

