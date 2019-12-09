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
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "common.hpp"
#include "check.hpp"
#include "ConvImpl.hpp"
#include "libxsmm.h"

#define VLEN 16

#define CHKERR_LIBXSMM_DNN(A) if ( A != LIBXSMM_DNN_SUCCESS )\
{\
  fprintf(stdout, "%s, %s\n", gp->node_name.c_str(), libxsmm_dnn_get_error(A) );\
  fflush(stdout);\
}

class ConvXSMM : public ConvImpl
{
  protected:
    ConvImpl *gp_;
    libxsmm_dnn_conv_desc conv_desc;
    libxsmm_dnn_layer* libxsmm_handle[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_input[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_output[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_filter[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_checkpoint_filter = NULL;
    libxsmm_dnn_tensor* libxsmm_checkpoint_history_filter = NULL;
    libxsmm_dnn_tensor* libxsmm_delinput[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_deloutput[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_delfilter[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_temp = NULL;
    libxsmm_dnn_tensor_datalayout* libxsmm_layout;
    libxsmm_dnn_err_t status;

    ConvImplParams *cp;
    float *dinptr, *dwtptr;
    bool updated_scratch_fwd=false, updated_scratch_bwd=false, updated_scratch_upd=false;
    void *in_ptr[NUM_NUMA_NODES] = {NULL}, *wt_ptr[NUM_NUMA_NODES]={NULL}, *hwt_ptr=NULL;
    void *out_ptr[NUM_NUMA_NODES] = {NULL}, *f32_wt_ptr[NUM_NUMA_NODES]={NULL};
    void *din_ptr[NUM_NUMA_NODES] = {NULL}, *dout_ptr[NUM_NUMA_NODES] = {NULL};
    void *scratch[NUM_NUMA_NODES]={NULL};
    int prev_scratch_size = 0;

  public:
    ConvXSMM(ConvImplParams *gp, int engine);
    virtual ~ConvXSMM(void) {}
    void forwardPropagate(TensorBuf *inp, TensorBuf* weightp, TensorBuf* hweightp, TensorBuf* biasp, TensorBuf *outp, int tid);
    void backPropagate(TensorBuf *inp, TensorBuf* weightp, TensorBuf *deloutp, TensorBuf *delinp, int tid);
    void weightUpdate(TensorBuf *inp, TensorBuf *deloutp, TensorBuf *delweightp, TensorBuf *delbiasp, int tid);
    void dumpBuffer(TensorBuf *wt, void* temp);
};
