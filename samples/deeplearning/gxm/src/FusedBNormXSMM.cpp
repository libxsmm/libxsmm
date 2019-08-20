/******************************************************************************
** Copyright (c) 2017-2019, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
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
  fusedbn_desc_train.N = gp->batch_size/gp->num_numa_nodes;
  fusedbn_desc_train.C = gp->nInput[0];
  fusedbn_desc_train.H = gp->iHeight;
  fusedbn_desc_train.W = gp->iWidth;
  fusedbn_desc_train.u = gp->stride_h;
  fusedbn_desc_train.v = gp->stride_w;
  fusedbn_desc_train.pad_h_in = gp->ipad_h;
  fusedbn_desc_train.pad_w_in = gp->ipad_w;
  fusedbn_desc_train.pad_h_out = gp->pad_h;
  fusedbn_desc_train.pad_w_out = gp->pad_w;
  fusedbn_desc_train.threads = gp->num_threads/gp->num_numa_nodes;

  if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
  {
    fusedbn_desc_train.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    fusedbn_desc_train.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
  }
  else if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_BF16)
  {
    fusedbn_desc_train.datatype_in = LIBXSMM_DNN_DATATYPE_BF16;
    fusedbn_desc_train.datatype_out = LIBXSMM_DNN_DATATYPE_BF16;
  }

  fusedbn_desc_train.datatype_stats = LIBXSMM_DNN_DATATYPE_F32;
  fusedbn_desc_train.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  fusedbn_desc_train.fuse_order = LIBXSMM_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU;
  fusedbn_desc_train.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN;

  if(gp->relu)
    fusedbn_desc_train.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_RELU_WITH_MASK;

  if(gp->eltwise)
    fusedbn_desc_train.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_ELTWISE;

  if(gp->relu && gp->eltwise)
    fusedbn_desc_train.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_ELTWISE_RELU_WITH_MASK;

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    libxsmm_handle_train[n] = libxsmm_dnn_create_fusedbatchnorm( fusedbn_desc_train, &status );
    CHKERR_LIBXSMM_DNN( status );
  }

  fusedbn_desc_test.N = gp->batch_size/gp->num_numa_nodes;
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
      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train[n], LIBXSMM_DNN_REGULAR_INPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      ofmblock = libxsmm_layout->dim_size[0];
      nBlocksFm = libxsmm_layout->dim_size[3];
      libxsmm_input_train[n] = libxsmm_dnn_link_tensor( libxsmm_layout, inp_r[n], &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_train[n], libxsmm_input_train[n], LIBXSMM_DNN_REGULAR_INPUT ) );

      if(gp->eltwise)
      {
        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train[n], LIBXSMM_DNN_REGULAR_INPUT_ADD, &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_input_add_train[n] = libxsmm_dnn_link_tensor(libxsmm_layout, inp_l[n], &status);
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train[n], libxsmm_input_add_train[n], LIBXSMM_DNN_REGULAR_INPUT_ADD ) )
      }

      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train[n], LIBXSMM_DNN_CHANNEL_EXPECTVAL, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_expectval_train[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)bexpect[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train[n], libxsmm_expectval_train[n], LIBXSMM_DNN_CHANNEL_EXPECTVAL ) );

      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train[n], LIBXSMM_DNN_CHANNEL_RCPSTDDEV, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_stddev_train[n] = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)bstddev[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train[n], libxsmm_stddev_train[n], LIBXSMM_DNN_CHANNEL_RCPSTDDEV ) );

      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train[n], LIBXSMM_DNN_CHANNEL_VARIANCE, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_variance_train[n] = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)bvariance[n], &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_train[n], libxsmm_variance_train[n], LIBXSMM_DNN_CHANNEL_VARIANCE ) );

      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train[n], LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_gamma_train[n] = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)gamma[n], &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_train[n], libxsmm_gamma_train[n], LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA ) );

      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_train[n], LIBXSMM_DNN_REGULAR_CHANNEL_BETA, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_beta_train[n] = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)beta[n], &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_train[n], libxsmm_beta_train[n], LIBXSMM_DNN_REGULAR_CHANNEL_BETA ) );

      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train[n], LIBXSMM_DNN_REGULAR_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_output_train[n] = libxsmm_dnn_link_tensor( libxsmm_layout, output[n], &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train[n], libxsmm_output_train[n], LIBXSMM_DNN_REGULAR_OUTPUT ) );

      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train[n], LIBXSMM_DNN_RELU_MASK, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_relumask_train[n] = libxsmm_dnn_link_tensor( libxsmm_layout, relu_mask[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_train[n], libxsmm_relumask_train[n], LIBXSMM_DNN_RELU_MASK) );
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
      long long int mysize = libxsmm_dnn_fusedbatchnorm_get_scratch_size( libxsmm_handle_train[n], &status );
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
      long long int mysize = libxsmm_dnn_fusedbatchnorm_get_scratch_size( libxsmm_handle_train[n], &status );

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
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_scratch( libxsmm_handle_train[n], sptrptr[n] ) );
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
#if 1
    if(gp->in_data_type == DT_FLOAT && gp->num_numa_nodes > 1)
    {
      if(sumscratch == NULL)
        sumscratch = (float*)libxsmm_aligned_malloc(gp->batch_size * nFM * 2 * sizeof(float), 2097152);

#include "reduce_mean_var.c"
    }
    else if(gp->in_data_type == DT_BF16 && gp->num_numa_nodes > 1)
    {
      int fullmb = gp->batch_size;
      if(sumscratch == NULL)
        sumscratch = (float*)libxsmm_aligned_malloc(fullmb * nFM * 2 * sizeof(float), 2097152);

      libxsmm_bfloat16* in = (libxsmm_bfloat16*)inpb[0]->getBuffer();
      libxsmm_bfloat16 (* __restrict input)[nBlocksFm][ifhp][ifwp][64] = (libxsmm_bfloat16 (*)[*][*][*][64])in;
      float (* __restrict sum_img)[fullmb][64] = (float (*)[*][64])sumscratch;
      float (* __restrict sumsq_img)[fullmb][64] = (float (*)[*][64])(sumscratch + fullmb*nFM);

      const float sqrt_eps = 1e-7f;
      const float nhw = (float)(gp->batch_size * ifh * ifw);
      const float recp_nhw = 1.0f/nhw;

      // Mean, rstdev, variance
      float (* __restrict bmean)[64] = (float (*)[64])bexpect[0];
      float (* __restrict brstd)[64] = (float (*)[64])bstddev[0];
      float (* __restrict bvar)[64]  = (float (*)[64])bvariance[0];

        // Parallelize over full minibatch
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int img=0; img<fullmb; img++)
      {
        float tempv[64], tempsqv[64];
        float inpv[64];
        libxsmm_bfloat16 linpv[64];

        // For each FM block
        for(int fm=0; fm<nBlocksFm; fm++)
        {
#pragma omp simd simdlen(16)
          for(int v=0; v<64; v++)
          {
            tempv[v] = 0.;
            tempsqv[v] = 0.;
          }

          // Reduce over H, W
          for(int h=iph; h<(ifh + iph); h++) {
            for(int w=ipw; w<(ifw + ipw); w++) {
#pragma omp simd simdlen(16)
              for(int v=0; v<64; v++)
                linpv[v] = input[img][fm][h][w][v];

              for(int i=0; i<64; i+=16)
              {
                __m256i vbfp16 = _mm256_loadu_si256( (const __m256i*)(linpv+i) );
                __m512  vfp32  = gxm_bfp16_to_fp32_avx512f( vbfp16 );
                _mm512_storeu_ps(inpv+i, vfp32 );
              }

#pragma omp simd simdlen(16)
              for(int v=0; v<64; v++) {
                tempv[v] += inpv[v];
                tempsqv[v] += inpv[v] * inpv[v];
              }
            }
          }

          // Partial sum/sumsq in scratch
#pragma omp simd simdlen(16)
          for(int v=0; v<64; v++) {
            sum_img[fm][img][v] = tempv[v];
            sumsq_img[fm][img][v] = tempsqv[v];
          }
        }
      }

#if 1
      if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
      {
        bool noNan = true;
        for(int i=0; i<64; i++)
        {
          if(isnan(sumscratch[i]) || isinf(sumscratch[i]))
          {
            noNan = false;
            break;
          }
        }
        if(!noNan)
        {
          MeanOfLayer((char*)((gp->node_name + "_sum_img").c_str()), sumscratch, fullmb*nFM);
          MeanOfLayer((char*)((gp->node_name + "_sumsq_img").c_str()), (sumscratch+fullmb*nFM), fullmb*nFM);
          MeanOfLayer((char*)((gp->node_name + "_inp").c_str()), (libxsmm_bfloat16*)in, fullmb*nFM*ifhp*ifwp);
          printf("exiting\n");
          exit(-1);
        }
      }
#endif

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int fm=0; fm < nBlocksFm; fm++)
      {
        // Partial sum/sumsq from NUMA node 0
        float tempv[64], tempsqv[64];

#pragma omp simd simdlen(16)
        for(int v=0; v<64; v++)
        {
          tempv[v] = 0.;
          tempsqv[v] = 0.;
        }

        // ofmblocks of all images in NUMA node 1..N-1
        for(int img=0; img < fullmb; img++) {
#pragma omp simd simdlen(16)
          for(int v=0; v<64; v++) {
            tempv[v] += sum_img[fm][img][v];
            tempsqv[v] += sumsq_img[fm][img][v];
          }
        }

#pragma omp simd simdlen(16)
        for(int v=0; v<64; v++) {
          bmean[fm][v] = tempv[v] * recp_nhw;
          bvar[fm][v] = (tempsqv[v] * recp_nhw) - (bmean[fm][v] * bmean[fm][v]);
          brstd[fm][v] = 1/sqrt(bvar[fm][v] + sqrt_eps);
        }
      }

#if 1
      if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
      {
        for(int i=0; i<64; i++)
        {
          if(isnan(*((float*)bexpect[0]+i)) || isinf(*((float*)bexpect[0]+i)))
            printf("%s mean bexpect[%d]: %f\n",gp->node_name.c_str(), i, *((float*)bexpect[0]+i));
        }
        for(int i=0; i<64; i++)
        {
          if(isnan(*((float*)bexpect[0]+i)) || isinf(*((float*)bexpect[0]+i)))
          {
            printf("exiting\n");
            exit(-1);
          }
        }
      }
#endif
      //#include "reduce_mean_var_bf16.c"
    }

    for(int n=1; n<gp->num_numa_nodes; n++)
    {
      for(int i=0; i<nFM; i++)
      {
        (bexpect[n])[i] = (bexpect[0])[i];
        (bstddev[n])[i] = (bstddev[0])[i];
        (bvariance[n])[i] = (bvariance[0])[i];
      }
    }
#endif

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
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_execute_st( libxsmm_handle_train[n], LIBXSMM_DNN_COMPUTE_KIND_FWD, n*ntps, tid ) );
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

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
          int tid = omp_get_thread_num();
          int ntps = gp->num_threads/gp->num_numa_nodes;
          int s = tid/ntps;

          if(s==n && tid % ntps == 0)
          {
            for (int b = 0; b < nBfm; ++b) {
              __m512 vbm = _mm512_load_ps(&bmean[b][0]);
              __m512 vbvar = _mm512_load_ps(&bvar[b][0]);

              _mm512_store_ps( &(gexp[b*VLEN]), _mm512_add_ps(_mm512_mul_ps(_mm512_load_ps( &(gexp[b*VLEN]) ), vmmf), vbm));
              _mm512_store_ps( &(gv[b*VLEN]), _mm512_add_ps( _mm512_mul_ps( _mm512_load_ps( &(gv[b*VLEN]) ), vmmf), _mm512_mul_ps(vnhw_ratio, vbvar)));
            }
          }
        }
#else

#ifdef _OPENMP
#pragma omp parallel for
#endif
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
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      int ntps = gp->num_threads/gp->num_numa_nodes;
      int s = tid/ntps;
      int ltid = tid - s*ntps;

      if(ltid == 0)
      {
        float *gexp = (float*)gexpect[s];
        float *gv = (float*)gvar[s];

        for(int i=0; i < nFM; i++)
        {
          ((float*)bexpect[s])[i] = gexp[i]/scaling_factor_;
          float tmp = gv[i]/scaling_factor_;
          ((float*)bstddev[s])[i] = 1./sqrt(tmp + gp->eps);
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
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_execute_st( libxsmm_handle_test[n], LIBXSMM_DNN_COMPUTE_KIND_FWD, n*ntps, tid ) );
    }
  }
}

void FusedBNormXSMM::backPropagate(TensorBuf *deloutpb, TensorBuf *delgammapb, TensorBuf *delbetapb, vector<TensorBuf*> delinpb, int tid)
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

  void *deloutput[NUM_NUMA_NODES];
  void *delinp_r[NUM_NUMA_NODES];
  void *delinp_l[NUM_NUMA_NODES];
  void *delgamma[NUM_NUMA_NODES];
  void *delbeta[NUM_NUMA_NODES];

  deloutput[0] = deloutpb->getBuffer();
  int imoff = nImg * nFM * ofhp * ofwp;
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

#if 1
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
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_scratch( libxsmm_handle_train[n], sptrptr[n] ) );
    updated_scratch_bwd = true;
  }

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_deloutput[n] == NULL && libxsmm_delinput[n] == NULL && libxsmm_delinput_add[n] == NULL &&
        libxsmm_delgamma[n] == NULL && libxsmm_delbeta[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train[n], LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_deloutput[n] = libxsmm_dnn_link_tensor( libxsmm_layout, deloutput[n], &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_train[n], libxsmm_deloutput[n], LIBXSMM_DNN_GRADIENT_OUTPUT ) );

      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train[n], LIBXSMM_DNN_GRADIENT_INPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_delinput[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, delinp_r[n], &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train[n], libxsmm_delinput[n], LIBXSMM_DNN_GRADIENT_INPUT ) );

      if(gp->eltwise)
      {
        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_train[n], LIBXSMM_DNN_GRADIENT_INPUT_ADD, &status);
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_delinput_add[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, delinp_l[n], &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train[n], libxsmm_delinput_add[n], LIBXSMM_DNN_GRADIENT_INPUT_ADD ) );
      }

      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_train[n], LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_delgamma[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, delgamma[n], &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train[n], libxsmm_delgamma[n], LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA ) );

      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_train[n], LIBXSMM_DNN_GRADIENT_CHANNEL_BETA, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_delbeta[n] = libxsmm_dnn_link_tensor( libxsmm_layout, delbeta[n], &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train[n], libxsmm_delbeta[n], LIBXSMM_DNN_GRADIENT_CHANNEL_BETA ) );
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
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_execute_st( libxsmm_handle_train[n], LIBXSMM_DNN_COMPUTE_KIND_BWD, n*ntps, tid ) );

#ifdef USE_MLSL

#pragma omp barrier

    if(tid == 0)
    {
      float *dgp = (float*)delgamma[0];
      float *dbp = (float*)delbeta[0];

      for(int n=1; n<gp->num_numa_nodes; n++)
      {
        float *rdgp = (float*)delgamma[n];
        float *rdbp = (float*)delbeta[n];

#pragma omp simd
        for(int i=0; i<nFM; i++)
        {
          dgp[i] += rdgp[i];
          dbp[i] += rdbp[i];
        }
      }

      for(int n=1; n<gp->num_numa_nodes; n++)
      {
        float *rdgp = (float*)delgamma[n];
        float *rdbp = (float*)delbeta[n];

#pragma vector nontemporal
#pragma omp simd
        for(int i=0; i<nFM; i++)
        {
          rdgp[i] = dgp[i];
          rdbp[i] = dbp[i];
        }
      }
    }
#endif
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

