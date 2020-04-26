/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Sasikanth Avancha, Dhiraj Kalamkar, Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <assert.h>
#include <immintrin.h>
#include "FusedBNormXSMM.hpp"

#define VLEN 16

FusedBNormXSMM::FusedBNormXSMM(FusedBNormImplParams* gp, int engine) : FusedBNormImpl(gp, engine)
{
  fusedbn_desc_train[0].partN = gp->batch_size/gp->num_numa_nodes;
  fusedbn_desc_train[0].fullN = gp->batch_size;
  fusedbn_desc_train[0].C = gp->nInput[0];
  fusedbn_desc_train[0].H = gp->iHeight;
  fusedbn_desc_train[0].W = gp->iWidth;
  fusedbn_desc_train[0].u = gp->stride_h;
  fusedbn_desc_train[0].v = gp->stride_w;
  fusedbn_desc_train[0].pad_h_in = gp->ipad_h;
  fusedbn_desc_train[0].pad_w_in = gp->ipad_w;
  fusedbn_desc_train[0].pad_h_out = gp->pad_h;
  fusedbn_desc_train[0].pad_w_out = gp->pad_w;
  fusedbn_desc_train[0].threads = gp->num_threads/gp->num_numa_nodes;

  if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
  {
    fusedbn_desc_train[0].datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    fusedbn_desc_train[0].datatype_out = LIBXSMM_DNN_DATATYPE_F32;
  }
  else if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_BF16)
  {
    fusedbn_desc_train[0].datatype_in = LIBXSMM_DNN_DATATYPE_BF16;
    fusedbn_desc_train[0].datatype_out = LIBXSMM_DNN_DATATYPE_BF16;
  }

  fusedbn_desc_train[0].datatype_stats = LIBXSMM_DNN_DATATYPE_F32;
  fusedbn_desc_train[0].buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  fusedbn_desc_train[0].fuse_order = LIBXSMM_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU;
  fusedbn_desc_train[0].fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_NORED;

  if(gp->relu)
    fusedbn_desc_train[0].fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_NORED_RELU_WITH_MASK;

  if(gp->eltwise)
    fusedbn_desc_train[0].fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_NORED_ELTWISE;

  if(gp->relu && gp->eltwise)
    fusedbn_desc_train[0].fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_NORED_ELTWISE_RELU_WITH_MASK;

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    libxsmm_handle_train[0][n] = libxsmm_dnn_create_fusedbatchnorm( fusedbn_desc_train[0], &status );
    CHKERR_LIBXSMM_DNN( status );
  }

  fusedbn_desc_train[1].partN = gp->batch_size/gp->num_numa_nodes;
  fusedbn_desc_train[1].fullN = gp->batch_size;
  fusedbn_desc_train[1].C = gp->nInput[0];
  fusedbn_desc_train[1].H = gp->iHeight;
  fusedbn_desc_train[1].W = gp->iWidth;
  fusedbn_desc_train[1].u = gp->stride_h;
  fusedbn_desc_train[1].v = gp->stride_w;
  fusedbn_desc_train[1].pad_h_in = gp->ipad_h;
  fusedbn_desc_train[1].pad_w_in = gp->ipad_w;
  fusedbn_desc_train[1].pad_h_out = gp->pad_h;
  fusedbn_desc_train[1].pad_w_out = gp->pad_w;
  fusedbn_desc_train[1].threads = gp->num_threads/gp->num_numa_nodes;

  if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
  {
    fusedbn_desc_train[1].datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    fusedbn_desc_train[1].datatype_out = LIBXSMM_DNN_DATATYPE_F32;
  }
  else if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_BF16)
  {
    fusedbn_desc_train[1].datatype_in = LIBXSMM_DNN_DATATYPE_BF16;
    fusedbn_desc_train[1].datatype_out = LIBXSMM_DNN_DATATYPE_BF16;
  }

  fusedbn_desc_train[1].datatype_stats = LIBXSMM_DNN_DATATYPE_F32;
  fusedbn_desc_train[1].buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  fusedbn_desc_train[1].fuse_order = LIBXSMM_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU;
  fusedbn_desc_train[1].fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE;

  if(gp->relu)
    fusedbn_desc_train[1].fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_RELU_WITH_MASK;

  if(gp->eltwise)
    fusedbn_desc_train[1].fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE;

  if(gp->relu && gp->eltwise)
    fusedbn_desc_train[1].fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE_RELU_WITH_MASK;

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    libxsmm_handle_train[1][n] = libxsmm_dnn_create_fusedbatchnorm( fusedbn_desc_train[1], &status );
    CHKERR_LIBXSMM_DNN( status );
  }

  fusedbn_desc_test.partN = gp->batch_size/gp->num_numa_nodes;
  fusedbn_desc_test.fullN = gp->batch_size;
  fusedbn_desc_test.C = gp->nInput[0];
  fusedbn_desc_test.H = gp->iHeight;
  fusedbn_desc_test.W = gp->iWidth;
  fusedbn_desc_test.u = gp->stride_h;
  fusedbn_desc_test.v = gp->stride_w;
  fusedbn_desc_test.pad_h_in = gp->ipad_h;
  fusedbn_desc_test.pad_w_in = gp->ipad_w;
  fusedbn_desc_test.pad_h_out = gp->pad_h;
  fusedbn_desc_test.pad_w_out = gp->pad_w;
  fusedbn_desc_test.threads = gp->num_threads/gp->num_numa_nodes;

  if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
  {
    fusedbn_desc_test.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    fusedbn_desc_test.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
  }
  else if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_BF16)
  {
    fusedbn_desc_test.datatype_in = LIBXSMM_DNN_DATATYPE_BF16;
    fusedbn_desc_test.datatype_out = LIBXSMM_DNN_DATATYPE_BF16;
  }

  fusedbn_desc_test.datatype_stats = LIBXSMM_DNN_DATATYPE_F32;
  fusedbn_desc_test.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  fusedbn_desc_test.fuse_order = LIBXSMM_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU;
  fusedbn_desc_test.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE;
  if(gp->relu)
    fusedbn_desc_test.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_RELU_WITH_MASK;
  if(gp->eltwise)
    fusedbn_desc_test.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE;
  if(gp->relu && gp->eltwise)
    fusedbn_desc_test.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE_RELU_WITH_MASK;

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    libxsmm_handle_test[n] = libxsmm_dnn_create_fusedbatchnorm( fusedbn_desc_test, &status );
    CHKERR_LIBXSMM_DNN( status );
  }
}

void FusedBNormXSMM::forwardPropagate(vector<TensorBuf *> inpb, TensorBuf *gammapb, TensorBuf *betapb, TensorBuf *meanpb, TensorBuf *varpb, TensorBuf *outpb, int tid)
{
  int nImg = gp->batch_size/gp->num_numa_nodes;
  int nFM = gp->nInput[0];
  int nBfm = nFM/VLEN;
  int ifh = gp->iHeight;
  int ifw = gp->iWidth;
  int iph = gp->ipad_h;
  int ipw = gp->ipad_w;
  int ifhp = ifh +2*iph;
  int ifwp = ifw + 2*ipw;
  int ofh = gp->oHeight;
  int ofw = gp->oWidth;
  int oph = gp->pad_h;
  int opw = gp->pad_w;
  int ofhp = ofh + 2*oph;
  int ofwp = ofw + 2*opw;

  void *inp_r[NUM_NUMA_NODES];
  void *inp_l[NUM_NUMA_NODES];
  void *output[NUM_NUMA_NODES];
  float *gamma[NUM_NUMA_NODES];
  float *beta[NUM_NUMA_NODES];
  float *gexpect[NUM_NUMA_NODES];
  float *gvar[NUM_NUMA_NODES];
  float *gexp_test = (float*)meanpb->getPrivBuffer();
  float *gvar_test = (float*)varpb->getPrivBuffer();

  inp_r[0] = inpb[0]->getBuffer();
  int imoff = nImg*nFM*ifhp*ifwp;
  if(gp->in_data_type == DT_FLOAT)
    imoff = imoff*sizeof(float);
  else if(gp->in_data_type == DT_BF16)
    imoff = imoff*sizeof(libxsmm_bfloat16);
  for(int n=1; n<gp->num_numa_nodes; n++)
    inp_r[n] = inp_r[n-1] + imoff;

  inp_l[0] = gp->eltwise ? inpb[1]->getBuffer() : NULL;
  if(inp_l[0])
  {
    imoff = nImg*gp->nInput[1]*ifhp*ifwp;
    if(gp->in_data_type == DT_FLOAT)
      imoff = imoff*sizeof(float);
    else if(gp->in_data_type == DT_BF16)
      imoff = imoff*sizeof(libxsmm_bfloat16);
    for(int n=1; n<gp->num_numa_nodes; n++)
      inp_l[n] = inp_l[n-1] + imoff;
  }

  output[0] = outpb->getBuffer();
  imoff = nImg*gp->nOutput*ofhp*ofwp;
  if(gp->out_data_type == DT_FLOAT)
    imoff = imoff*sizeof(float);
  else if(gp->out_data_type == DT_BF16)
    imoff = imoff*sizeof(libxsmm_bfloat16);
  for(int n=1; n<gp->num_numa_nodes; n++)
    output[n] = output[n-1] + imoff;

  void **gptrptr = gammapb->getBufferPtr();
  int offset = gammapb->getOffset();
  for(int n=0; n<gp->num_numa_nodes; n++)
    gamma[n] = (float*)gptrptr[n] + offset;

  void **bptrptr = betapb->getBufferPtr();
  offset = betapb->getOffset();
  for(int n=0; n<gp->num_numa_nodes; n++)
    beta[n] = (float*)bptrptr[n] + offset;

  void **mptrptr = meanpb->getBufferPtr();
  offset = meanpb->getOffset();
  for(int n=0; n<gp->num_numa_nodes; n++)
    gexpect[n] = (float*)mptrptr[n] + offset;

  void **vptrptr = varpb->getBufferPtr();
  offset = varpb->getOffset();
  for(int n=0; n<gp->num_numa_nodes; n++)
    gvar[n] = (float*)vptrptr[n] + offset;

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(bexpect[n] == NULL)
    {
      bexpect[n] = (float*)libxsmm_aligned_malloc(nFM*sizeof(float), 2097152);

#ifndef NDEBUG
      printf("%s allocated %lu bytes for mean\n",nname.c_str(), nFM*sizeof(float));
#endif
    }

    if(bstddev[n] == NULL)
    {
      bstddev[n] = (float*)libxsmm_aligned_malloc(nFM*sizeof(float), 2097152);

#ifndef NDEBUG
      printf("%s allocated %lu bytes for stdev\n",nname.c_str(), nFM*sizeof(float));
#endif
    }

    if(bvariance[n] == NULL)
    {
      bvariance[n] = (float*)libxsmm_aligned_malloc(nFM*sizeof(float), 2097152);

#ifndef NDEBUG
      printf("%s allocated %lu bytes for variance\n",nname.c_str(), nFM*sizeof(float));
#endif
    }

    if(relu_mask[n] == NULL)
      relu_mask[n] = (void*)libxsmm_aligned_malloc(nImg*nFM*ofhp*ofwp*sizeof(unsigned char), 2097152);
  }

  if(gexp_test == NULL)
  {
    gexp_test = (float*)libxsmm_aligned_malloc(nFM*sizeof(float), 2097152);
    meanpb->setPrivBuffer((void*)gexp_test);

#ifndef NDEBUG
    printf("%s allocated %lu bytes for mean test\n",nname.c_str(), nFM*sizeof(float));
#endif
  }

  if(gvar_test == NULL)
  {
    gvar_test = (float*)libxsmm_aligned_malloc(nFM*sizeof(float), 2097152);
    varpb->setPrivBuffer((void*)gvar_test);

#ifndef NDEBUG
    printf("%s allocated %lu bytes for mean test\n",nname.c_str(), nFM*sizeof(float));
#endif
  }

  void **sptrptr = scratchp->getBufferPtr();

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_input_train[n] == NULL && libxsmm_input_add_train[n] == NULL && libxsmm_expectval_train[n] == NULL &&
        libxsmm_stddev_train[n] == NULL && libxsmm_variance_train[n] == NULL && libxsmm_gamma_train[n] == NULL &&
          libxsmm_beta_train[n] == NULL && libxsmm_output_train[n] == NULL)
    {
      for(int t=0; t < 2; t++)
      {
        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train[t][n], LIBXSMM_DNN_REGULAR_INPUT, &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_input_train[n] = libxsmm_dnn_link_tensor( libxsmm_layout, inp_r[n], &status ); CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN(libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_train[t][n], libxsmm_input_train[n], LIBXSMM_DNN_REGULAR_INPUT ) );

        if(gp->eltwise)
        {
          libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train[t][n], LIBXSMM_DNN_REGULAR_INPUT_ADD, &status );
          CHKERR_LIBXSMM_DNN( status );
          libxsmm_input_add_train[n] = libxsmm_dnn_link_tensor(libxsmm_layout, inp_l[n], &status);
          CHKERR_LIBXSMM_DNN( status );
          libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
          CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train[t][n], libxsmm_input_add_train[n], LIBXSMM_DNN_REGULAR_INPUT_ADD ) )
        }

        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train[t][n], LIBXSMM_DNN_CHANNEL_EXPECTVAL, &status);
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_expectval_train[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)bexpect[n], &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train[t][n], libxsmm_expectval_train[n], LIBXSMM_DNN_CHANNEL_EXPECTVAL ) );

        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train[t][n], LIBXSMM_DNN_CHANNEL_RCPSTDDEV, &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_stddev_train[n] = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)bstddev[n], &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train[t][n], libxsmm_stddev_train[n], LIBXSMM_DNN_CHANNEL_RCPSTDDEV ) );

        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train[t][n], LIBXSMM_DNN_CHANNEL_VARIANCE, &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_variance_train[n] = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)bvariance[n], &status);
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN(libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_train[t][n], libxsmm_variance_train[n], LIBXSMM_DNN_CHANNEL_VARIANCE ) );

        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train[t][n], LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA, &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_gamma_train[n] = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)gamma[n], &status ); CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_train[t][n], libxsmm_gamma_train[n], LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA ) );

        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_train[t][n], LIBXSMM_DNN_REGULAR_CHANNEL_BETA, &status);
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_beta_train[n] = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)beta[n], &status ); CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_train[t][n], libxsmm_beta_train[n], LIBXSMM_DNN_REGULAR_CHANNEL_BETA ) );

        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train[t][n], LIBXSMM_DNN_REGULAR_OUTPUT, &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_output_train[n] = libxsmm_dnn_link_tensor( libxsmm_layout, output[n], &status ); CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train[t][n], libxsmm_output_train[n], LIBXSMM_DNN_REGULAR_OUTPUT ) );

        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train[t][n], LIBXSMM_DNN_RELU_MASK, &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_relumask_train[n] = libxsmm_dnn_link_tensor( libxsmm_layout, relu_mask[n], &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_train[t][n], libxsmm_relumask_train[n], LIBXSMM_DNN_RELU_MASK) );
      }
    }
  }

  /* let's allocate (if required) and bind scratch */
  if(sptrptr == NULL)
  {
    sptrptr = (void**)libxsmm_aligned_malloc(gp->num_numa_nodes*sizeof(void*), 2097152);
    scratchp->setBufferPtr(sptrptr);
  }

  int max_size = 0;
  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(sptrptr[n] == NULL)
    {
      long long int mysize = libxsmm_dnn_fusedbatchnorm_get_scratch_size( libxsmm_handle_train[0][n], &status );
      CHKERR_LIBXSMM_DNN( status );
      sptrptr[n] = (void*)libxsmm_aligned_malloc(mysize , 2097152);
      max_size = mysize;

#ifdef USE_MLSL
      if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
        printf("%s allocated %lld bytes for scratch @ %p\n",nname.c_str(), mysize, sptrptr[n]);
    }
    else
    {
      long long int ssize = scratchp->getBufferSize();
      long long int mysize = libxsmm_dnn_fusedbatchnorm_get_scratch_size( libxsmm_handle_train[0][n], &status );

      CHKERR_LIBXSMM_DNN( status );

      if(ssize < mysize)
      {
        libxsmm_free(sptrptr[n]);
        sptrptr[n] = (void*)libxsmm_aligned_malloc(mysize, 2097152);
        scratchp->setBufferSize(mysize);
#ifdef USE_MLSL
        if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
          printf("%s allocated %lld bytes for scratch @ %p, prev size was %lld bytes\n",nname.c_str(), mysize, sptrptr[n], ssize);
      }
      else
        max_size = ssize;
    }
  }
  scratchp->setBufferSize(max_size);

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_input_test[n] == NULL && libxsmm_input_add_test[n] == NULL && libxsmm_expectval_test[n] == NULL &&
        libxsmm_stddev_test[n] == NULL && libxsmm_variance_test[n] == NULL && libxsmm_gamma_test[n] == NULL &&
        libxsmm_beta_test[n] == NULL && libxsmm_output_test[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_test[n], LIBXSMM_DNN_REGULAR_INPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input_test[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, inp_r[n], &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_test[n], libxsmm_input_test[n], LIBXSMM_DNN_REGULAR_INPUT ) );

      if(gp->eltwise)
      {
        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_test[n], LIBXSMM_DNN_REGULAR_INPUT_ADD, &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_input_add_test[n] = libxsmm_dnn_link_tensor( libxsmm_layout, inp_l[n], &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_test[n], libxsmm_input_add_test[n], LIBXSMM_DNN_REGULAR_INPUT_ADD ) )
      }

      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_test[n], LIBXSMM_DNN_CHANNEL_EXPECTVAL, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_expectval_test[n] = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)bexpect[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_test[n], libxsmm_expectval_test[n], LIBXSMM_DNN_CHANNEL_EXPECTVAL ) );

      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_test[n], LIBXSMM_DNN_CHANNEL_RCPSTDDEV, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_stddev_test[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)bstddev[n], &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_test[n], libxsmm_stddev_test[n], LIBXSMM_DNN_CHANNEL_RCPSTDDEV ) );

      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_test[n], LIBXSMM_DNN_CHANNEL_VARIANCE, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_variance_test[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)bvariance[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_test[n], libxsmm_variance_test[n], LIBXSMM_DNN_CHANNEL_VARIANCE ) );

      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_test[n], LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_gamma_test[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)gamma[n], &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_test[n], libxsmm_gamma_test[n], LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA ) );

      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_test[n], LIBXSMM_DNN_REGULAR_CHANNEL_BETA, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_beta_test[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)beta[n], &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_test[n], libxsmm_beta_test[n], LIBXSMM_DNN_REGULAR_CHANNEL_BETA ) );

      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_test[n], LIBXSMM_DNN_REGULAR_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_output_test[n] = libxsmm_dnn_link_tensor( libxsmm_layout, output[n], &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_test[n], libxsmm_output_test[n], LIBXSMM_DNN_REGULAR_OUTPUT ) );

      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_test[n], LIBXSMM_DNN_RELU_MASK, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_relumask_test[n] = libxsmm_dnn_link_tensor( libxsmm_layout, relu_mask[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_test[n], libxsmm_relumask_test[n], LIBXSMM_DNN_RELU_MASK) );
    }
  }

  if(!updated_scratch_fwd)
  {
    for(int n=0; n<gp->num_numa_nodes; n++)
    {
      for(int t=0; t < 2; t++)
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_scratch( libxsmm_handle_train[t][n], sptrptr[n] ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_scratch( libxsmm_handle_test[n], sptrptr[n] ) );
    }
    updated_scratch_fwd = true;
  }

#if 0
#ifndef NDEBUG
  if ( (oph > 0 || opw > 0) && (iph > 0 || ipw > 0) ) {
    printf("node %s: batchnorm forward input and output is padded which cannot be :-(\n", nname.c_str());
  }

  /* check rims */
  if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
  {
    check_physical_pad( nname.c_str(),    (float*)inp_r[0], nImg, nBfm, ifh,  ifw,  VLEN, iph, ipw );
    check_physical_pad( nname.c_str(),     (float*)output[0], nImg, nBfm, ofh, ofw, VLEN, oph,  opw );
  }
  else if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_BF16)
  {
    check_physical_pad( nname.c_str(),    (libxsmm_bfloat16*)inp_r[0], nImg, nBfm, ifh,  ifw,  VLEN, iph, ipw );
    check_physical_pad( nname.c_str(),     (libxsmm_bfloat16*)output[0], nImg, nBfm, ofh, ofw, VLEN, oph,  opw );
  }
#endif
#endif

  if(!use_global_stats)
  {
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      int ntps = gp->num_threads/gp->num_numa_nodes;
      int n = tid/ntps;
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_execute_st( libxsmm_handle_train[0][n], LIBXSMM_DNN_COMPUTE_KIND_FWD, n*ntps, tid ) );

#pragma omp barrier

      if(n == 0)
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_reduce_stats_st(&libxsmm_handle_train[0][0], gp->num_numa_nodes, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid) );

#pragma omp barrier

      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_execute_st( libxsmm_handle_train[1][n], LIBXSMM_DNN_COMPUTE_KIND_FWD, n*ntps, tid ) );
    }

#if 0
#ifndef NDEBUG
    if ( (oph > 0 || opw > 0) && (iph > 0 || ipw > 0) ) {
      printf("node %s: batchnorm forward input and output is padded which cannot be :-(\n", nname.c_str());
    }

    /* check rims */
    if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
    {
      check_physical_pad( nname.c_str(),    (float*)inp_r[0], nImg, nBfm, ifh,  ifw,  VLEN, iph, ipw );
      check_physical_pad( nname.c_str(),     (float*)output[0], nImg, nBfm, ofh, ofw, VLEN, oph,  opw );
    }
    else if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_BF16)
    {
      check_physical_pad( nname.c_str(),    (libxsmm_bfloat16*)inp_r[0], nImg, nBfm, ifh,  ifw,  VLEN, iph, ipw );
      check_physical_pad( nname.c_str(),     (libxsmm_bfloat16*)output[0], nImg, nBfm, ofh, ofw, VLEN, oph, opw );
    }
#endif
#endif

    if(gp->exec_mode == "TRAIN")
    {
      for(int n=0; n<gp->num_numa_nodes; n++)
      {
        float *gexp = (float*)gexpect[n];
        float *gv = (float*)gvar[n];

        float (* __restrict bmean)[VLEN] = (float (*)[VLEN])bexpect[n];
        float (* __restrict bvar)[VLEN] = (float (*)[VLEN])bvariance[n];
        float nhw_ratio = float(nImg*ifh*ifw)/float(nImg*ifh*ifw - 1);

#ifdef __AVX512F__
        __m512  vmmf       = _mm512_set1_ps(gp->mmf);
        __m512  vnhw_ratio = _mm512_set1_ps(nhw_ratio);

        for (int b = 0; b < nBfm; ++b) {
          __m512 vbm = _mm512_loadu_ps(&bmean[b][0]);
          __m512 vbvar = _mm512_loadu_ps(&bvar[b][0]);

          _mm512_storeu_ps( &(gexp[b*VLEN]), _mm512_add_ps(_mm512_mul_ps(_mm512_loadu_ps( &(gexp[b*VLEN]) ), vmmf), vbm));
          _mm512_storeu_ps( &(gv[b*VLEN]), _mm512_add_ps( _mm512_mul_ps( _mm512_loadu_ps( &(gv[b*VLEN]) ), vmmf), _mm512_mul_ps(vnhw_ratio, vbvar)));
        }
#else

        for (int b = 0; b < nBfm; ++b) {
#pragma omp simd
          for (int v = 0; v < 16; ++v) {
            gexp[(b*16)+v] = gexp[(b*16)+v] * gp->mmf + bmean[b][v];
            gv[(b*16)+v] = gv[(b*16)+v] * gp->mmf + nhw_ratio*bvar[b][v];
          }
        }
#endif
      }
      scaling_factor_ *= gp->mmf;
      scaling_factor_ += 1.;
    }
  }
  else
  {
    for(int n=0; n<gp->num_numa_nodes; n++)
    {
      float *gexp = (float*)gexpect[n];
      float *gv = (float*)gvar[n];

#pragma omp simd
      for(int i=0; i < nFM; i++)
      {
        ((float*)bexpect[n])[i] = gexp[i]/scaling_factor_;
        float tmp = gv[i]/scaling_factor_;
        ((float*)bstddev[n])[i] = 1./sqrt(tmp + gp->eps);
      }
    }

#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      int ntps = gp->num_threads/gp->num_numa_nodes;
      int n = tid/ntps;
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_execute_st( libxsmm_handle_test[n], LIBXSMM_DNN_COMPUTE_KIND_FWD, n*ntps, tid ) );
    }
  }
}

void FusedBNormXSMM::backPropagate(TensorBuf *deloutpb, TensorBuf *delgammapb, TensorBuf *delbetapb, vector<TensorBuf *> delinpb, int tid)
{
  int nImg  = gp->batch_size/gp->num_numa_nodes;
  int nFM = gp->nOutput;
  int nBfm = nFM/VLEN;
  int ofh = gp->oHeight;
  int ofw = gp->oWidth;
  int oph = gp->pad_h;
  int opw = gp->pad_w;
  int ofhp = ofh + 2*oph;
  int ofwp = ofw + 2*opw;
  int ifh = gp->iHeight;
  int ifw = gp->iWidth;
  int iph = gp->ipad_h;
  int ipw = gp->ipad_w;
  int ifhp = ifh + 2*iph;
  int ifwp = ifw + 2*ipw;
  int sh = gp->stride_h;
  int sw = gp->stride_w;

  int imoff;

  void *deloutput[NUM_NUMA_NODES];
  void *delinp_r[NUM_NUMA_NODES];
  void *delinp_l[NUM_NUMA_NODES];
  void *delgamma[NUM_NUMA_NODES];
  void *delbeta[NUM_NUMA_NODES];

  deloutput[0] = deloutpb->getBuffer();
  imoff = nImg * nFM * ofhp * ofwp;
  if(gp->out_data_type == DT_FLOAT)
    imoff = imoff * sizeof(float);
  else if(gp->out_data_type == DT_BF16)
    imoff = imoff * sizeof(libxsmm_bfloat16);
  for(int n=1; n<gp->num_numa_nodes; n++)
    deloutput[n] = deloutput[n-1] + imoff;

  delinp_r[0] = delinpb[0]->getBuffer();
  imoff = nImg * gp->nInput[0] * ifhp * ifwp;
  if(gp->in_data_type == DT_FLOAT)
    imoff = imoff * sizeof(float);
  else if(gp->in_data_type == DT_BF16)
    imoff = imoff * sizeof(libxsmm_bfloat16);
  for(int n=1; n<gp->num_numa_nodes; n++)
    delinp_r[n] = delinp_r[n-1] + imoff;

  delinp_l[0] = gp->eltwise ? delinpb[1]->getBuffer() : NULL;
  if(delinp_l[0])
  {
    imoff = nImg * gp->nInput[1] * ifhp * ifwp;
    if(gp->in_data_type == DT_FLOAT)
      imoff = imoff * sizeof(float);
    else if(gp->in_data_type == DT_BF16)
      imoff = imoff * sizeof(libxsmm_bfloat16);
    for(int n=1; n<gp->num_numa_nodes; n++)
      delinp_l[n] = delinp_l[n-1] + imoff;
  }

  void **gptrptr = delgammapb->getBufferPtr();
  int offset = delgammapb->getOffset() * sizeof(float);
  for(int n=0; n<gp->num_numa_nodes; n++)
    delgamma[n] = gptrptr[n] + offset;

  void **bptrptr = delbetapb->getBufferPtr();
  offset = delbetapb->getOffset() * sizeof(float);
  for(int n=0; n<gp->num_numa_nodes; n++)
    delbeta[n] = bptrptr[n] + offset;

#if 0
  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(gp->in_data_type == DT_FLOAT)
    {
      float (* __restrict del_input_r)[nBfm][ifhp][ifwp][64] = (float (*)[*][*][*][64])delinp_r[n];

      /* zero the rims in case of physical padding */
      /* @TODO, we need to do the same thing with del_input_l?! */
      if (iph > 0 || ipw > 0) {
#pragma omp parallel for
        for (int img = 0; img < nImg; img++) {
          for (int fm = 0; fm < nBfm; fm++) {
            for (int w = 0; w < ifwp; w++) {
              for (int ph = 0; ph < iph; ph++) {
#ifdef __AVX512F__
                for(int i=0; i<64; i+=16) {
                  _mm512_stream_ps( &(del_input_r[img][fm][ph      ][w][i]), _mm512_setzero_ps() );
                  _mm512_stream_ps( &(del_input_r[img][fm][ifhp-1-ph][w][i]), _mm512_setzero_ps() );
                }
#else
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
                for(int v=0; v < 64; v++) {
                  del_input_r[img][fm][ph][w][v] = 0.0f;
                  del_input_r[img][fm][ifhp-1-ph][w][v] = 0.0f;
                }
#endif
              }
              for (int h = iph; h < ifh+iph; h++) {
                for (int pw = 0; pw < ipw; pw++) {
#ifdef __AVX512F__
                  for(int i=0; i<64; i+=16) {
                    _mm512_stream_ps( &(del_input_r[img][fm][h][pw      ][i]), _mm512_setzero_ps() );
                    _mm512_stream_ps( &(del_input_r[img][fm][h][ifwp-1-pw][i]), _mm512_setzero_ps() );
                  }
#else
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
                  for(int v=0; v < 64; v++) {
                    del_input_r[img][fm][h][pw][v] = 0.0f;
                    del_input_r[img][fm][h][ifwp-1-pw][v] = 0.0f;
                  }
#endif
                }
              }
            }
          }
        }
      }
    }
    else if(gp->in_data_type == DT_BF16)
    {
      libxsmm_bfloat16 (* __restrict del_input_r)[nBlocksFm][ifhp][ifwp][64] = (libxsmm_bfloat16 (*)[*][*][*][64])delinp_r[n];

      /* zero the rims in case of physical padding */
      /* @TODO, we need to do the same thing with del_input_l?! */
      if (iph > 0 || ipw > 0) {
#pragma omp parallel for
        for (int img = 0; img < nImg; img++) {
          for (int fm = 0; fm < nBlocksFm; fm++) {
            for (int w = 0; w < ifwp; w++) {
              for (int ph = 0; ph < iph; ph++) {
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
                for(int v=0; v < 64; v++) {
                  del_input_r[img][fm][ph][w][v] = 0;
                  del_input_r[img][fm][ifhp-1-ph][w][v] = 0;
                }
              }
            }
            for (int h = iph; h < ifh+iph; h++) {
              for (int pw = 0; pw < ipw; pw++) {
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
                for(int v=0; v < 64; v++) {
                  del_input_r[img][fm][h][pw][v] = 0;
                  del_input_r[img][fm][h][ifwp-1-pw][v] = 0;
                }
              }
            }
          }
        }
      }
    }
  }
#endif
  /* Perform physical padding tests */
#if 0
#ifndef NDEBUG
  if ( (oph > 0 || opw > 0) && (iph > 0 || ipw > 0) ) {
    printf("node %s: batchnorm backward input and output is padded which cannot be :-(\n", nname.c_str());
  }

  /* check rims */
  if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
  {
    check_physical_pad( nname.c_str(), (float*)delinp_r[0], nImg, nBfm, ifh,  ifw,  VLEN, iph, ipw );
    check_physical_pad( nname.c_str(),  (float*)deloutput[0], nImg, nBfm, ofh, ofw, VLEN, oph,  opw );
  }
  else if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_BF16)
  {
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)delinp_r[0], nImg, nBfm, ifh,  ifw,  VLEN, iph, ipw );
    check_physical_pad( nname.c_str(),  (libxsmm_bfloat16*)deloutput[0], nImg, nBfm, ofh, ofw, VLEN, oph,  opw );
  }
#endif
#endif

  void **sptrptr = scratchp->getBufferPtr();
  if(!updated_scratch_bwd)
  {
    for(int n=0; n<gp->num_numa_nodes; n++)
      for(int t=0; t<2; t++)
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_scratch( libxsmm_handle_train[t][n], sptrptr[n] ) );
    updated_scratch_bwd = true;
  }

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_deloutput[n] == NULL && libxsmm_delinput[n] == NULL && libxsmm_delinput_add[n] == NULL &&
        libxsmm_delgamma[n] == NULL && libxsmm_delbeta[n] == NULL)
    {
      for(int t=0; t < 2; t++)
      {
        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train[t][n], LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_deloutput[n] = libxsmm_dnn_link_tensor( libxsmm_layout, deloutput[n], &status ); CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_train[t][n], libxsmm_deloutput[n], LIBXSMM_DNN_GRADIENT_OUTPUT ) );

        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train[t][n], LIBXSMM_DNN_GRADIENT_INPUT, &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_delinput[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, delinp_r[n], &status ); CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train[t][n], libxsmm_delinput[n], LIBXSMM_DNN_GRADIENT_INPUT ) );

        if(gp->eltwise)
        {
          libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_train[t][n], LIBXSMM_DNN_GRADIENT_INPUT_ADD, &status);
          CHKERR_LIBXSMM_DNN( status );
          libxsmm_delinput_add[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, delinp_l[n], &status );
          CHKERR_LIBXSMM_DNN( status );
          libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
          CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train[t][n], libxsmm_delinput_add[n], LIBXSMM_DNN_GRADIENT_INPUT_ADD ) );
        }

        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_train[t][n], LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA, &status);
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_delgamma[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, delgamma[n], &status ); CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train[t][n], libxsmm_delgamma[n], LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA ) );

        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_train[t][n], LIBXSMM_DNN_GRADIENT_CHANNEL_BETA, &status);
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_delbeta[n] = libxsmm_dnn_link_tensor( libxsmm_layout, delbeta[n], &status ); CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train[t][n], libxsmm_delbeta[n], LIBXSMM_DNN_GRADIENT_CHANNEL_BETA ) );
      }
    }
  }

#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    int ntps = gp->num_threads/gp->num_numa_nodes;
    int n = tid/ntps;
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_execute_st( libxsmm_handle_train[0][n], LIBXSMM_DNN_COMPUTE_KIND_BWD, n*ntps, tid ) );

#pragma omp barrier

    if(n == 0)
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_reduce_stats_st(&libxsmm_handle_train[0][0], gp->num_numa_nodes, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid));

#pragma omp barrier

    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_execute_st( libxsmm_handle_train[1][n], LIBXSMM_DNN_COMPUTE_KIND_BWD, n*ntps, tid ) );
  }

  /* Perform physical padding tests */
#if 0
#ifndef NDEBUG
  if ( (oph > 0 || opw > 0) && (iph > 0 || ipw > 0) ) {
    printf("node %s: batchnorm backward input and output is padded which cannot be :-(\n", nname.c_str());
  }

  /* check rims */
  if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
  {
    check_physical_pad( nname.c_str(), (float*)delinp_r[0], nImg, nBfm, ifh,  ifw,  VLEN, iph, ipw );
    check_physical_pad( nname.c_str(),  (float*)deloutput[0], nImg, nBfm, ofh, ofw, VLEN, oph,  opw );
  }
  else if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_BF16)
  {
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)delinp_r[0], nImg, nBfm, ifh,  ifw,  VLEN, iph, ipw );
    check_physical_pad( nname.c_str(),  (libxsmm_bfloat16*)deloutput[0], nImg, nBfm, ofh, ofw, VLEN, oph,  opw );
  }
#endif
#endif
}

