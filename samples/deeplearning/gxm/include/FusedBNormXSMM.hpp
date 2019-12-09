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
#include "FusedBNormImpl.hpp"
#include "check.hpp"
#include "libxsmm.h"

#define CHKERR_LIBXSMM_DNN(A) if ( A != LIBXSMM_DNN_SUCCESS )\
{\
  fprintf(stdout, "%s, %s\n", gp->node_name.c_str(), libxsmm_dnn_get_error(A) );\
  fflush(stdout);\
}
class FusedBNormXSMM : public FusedBNormImpl
{
  protected:
    FusedBNormImpl *gp_;
    libxsmm_dnn_fusedbatchnorm_desc fusedbn_desc_train[2];
    libxsmm_dnn_fusedbatchnorm_desc fusedbn_desc_test;
    libxsmm_dnn_fusedbatchnorm* libxsmm_handle_train[2][NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_fusedbatchnorm* libxsmm_handle_test[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_input_train[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_input_add_train[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_output_train[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_relumask_train[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_expectval_train[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_stddev_train[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_variance_train[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_gamma_train[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_beta_train[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_input_test[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_input_add_test[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_output_test[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_relumask_test[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_expectval_test[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_stddev_test[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_variance_test[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_gamma_test[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_beta_test[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_delinput[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_delinput_add[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_deloutput[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_delgamma[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor* libxsmm_delbeta[NUM_NUMA_NODES] = {NULL};
    libxsmm_dnn_tensor_datalayout* libxsmm_layout;
    libxsmm_dnn_err_t status;

    float *bexpect[NUM_NUMA_NODES]={NULL}, *bstddev[NUM_NUMA_NODES]={NULL}, *bvariance[NUM_NUMA_NODES]={NULL};
    void *relu_mask[NUM_NUMA_NODES]={NULL};
    void *scratch=NULL;
    bool updated_scratch_fwd=false, updated_scratch_bwd=false;
    int nBlocksFm, ofmblock;
    float *sumscratch=NULL;

  public:
    FusedBNormXSMM(FusedBNormImplParams* gp, int engine);
    virtual ~FusedBNormXSMM(void) {}

    // Assume external threading, e.g., #pragma omp
    void forwardPropagate(vector<TensorBuf*> inp, TensorBuf* gammap, TensorBuf* betap, TensorBuf *gmeanp, TensorBuf *gvarp, TensorBuf *outp, int tid);
    void backPropagate(TensorBuf *deloutp, TensorBuf *delgammap, TensorBuf *delbetap, vector<TensorBuf *> delinp, int tid);
};
