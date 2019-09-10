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
  fusedbn_desc_train.N = gp->batch_size;
  fusedbn_desc_train.C = gp->nInput[0];
  fusedbn_desc_train.H = gp->iHeight;
  fusedbn_desc_train.W = gp->iWidth;
  fusedbn_desc_train.u = gp->stride_h;
  fusedbn_desc_train.v = gp->stride_w;
  fusedbn_desc_train.pad_h_in = gp->ipad_h;
  fusedbn_desc_train.pad_w_in = gp->ipad_w;
  fusedbn_desc_train.pad_h_out = gp->pad_h;
  fusedbn_desc_train.pad_w_out = gp->pad_w;
  fusedbn_desc_train.threads = gp->num_threads;

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

  libxsmm_handle_train = libxsmm_dnn_create_fusedbatchnorm( fusedbn_desc_train, &status );
  CHKERR_LIBXSMM_DNN( status );

  fusedbn_desc_test.N = gp->batch_size;
  fusedbn_desc_test.C = gp->nInput[0];
  fusedbn_desc_test.H = gp->iHeight;
  fusedbn_desc_test.W = gp->iWidth;
  fusedbn_desc_test.u = gp->stride_h;
  fusedbn_desc_test.v = gp->stride_w;
  fusedbn_desc_test.pad_h_in = gp->ipad_h;
  fusedbn_desc_test.pad_w_in = gp->ipad_w;
  fusedbn_desc_test.pad_h_out = gp->pad_h;
  fusedbn_desc_test.pad_w_out = gp->pad_w;
  fusedbn_desc_test.threads = gp->num_threads;

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

  libxsmm_handle_test = libxsmm_dnn_create_fusedbatchnorm( fusedbn_desc_test, &status );
  CHKERR_LIBXSMM_DNN( status );
}

void FusedBNormXSMM::forwardPropagate(vector<TensorBuf *> inpb, TensorBuf *gammapb, TensorBuf *betapb, TensorBuf *meanpb, TensorBuf *varpb, TensorBuf *outpb, int tid)
{
  int nImg = gp->batch_size;
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

  float *gexp_test = (float*)meanpb->getPrivBuffer();
  float *gvar_test = (float*)varpb->getPrivBuffer();

  void *inp_r = inpb[0]->getBuffer();
  void *inp_l = gp->eltwise ? inpb[1]->getBuffer() : NULL;

  void *output = outpb->getBuffer();
  float *gamma = (float*)gammapb->getBuffer();
  float *beta = (float*)betapb->getBuffer();
  float *gmean = (float*)meanpb->getBuffer();
  float *gvar = (float*)varpb->getBuffer();

  if(bexpect == NULL)
  {
    bexpect = (float*)libxsmm_aligned_malloc(nFM*sizeof(float), 2097152);

#ifndef NDEBUG
    printf("%s allocated %lu bytes for mean\n",nname.c_str(), nFM*sizeof(float));
#endif
  }

  if(bstddev == NULL)
  {
    bstddev = (float*)libxsmm_aligned_malloc(nFM*sizeof(float), 2097152);

#ifndef NDEBUG
    printf("%s allocated %lu bytes for stdev\n",nname.c_str(), nFM*sizeof(float));
#endif
  }

  if(bvariance == NULL)
  {
    bvariance = (float*)libxsmm_aligned_malloc(nFM*sizeof(float), 2097152);

#ifndef NDEBUG
    printf("%s allocated %lu bytes for variance\n",nname.c_str(), nFM*sizeof(float));
#endif
  }

  if(relu_mask == NULL)
    relu_mask = (void*)libxsmm_aligned_malloc(nImg*nFM*ofhp*ofwp*sizeof(unsigned char), 2097152);

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

  if(libxsmm_input_train == NULL && libxsmm_input_add_train == NULL && libxsmm_expectval_train == NULL &&
      libxsmm_stddev_train == NULL && libxsmm_variance_train == NULL && libxsmm_gamma_train == NULL &&
      libxsmm_beta_train == NULL && libxsmm_output_train == NULL)
  {
    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train, LIBXSMM_DNN_REGULAR_INPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_input_train = libxsmm_dnn_link_tensor( libxsmm_layout, inp_r, &status ); CHKERR_LIBXSMM_DNN( status );
    nBlocksFm = libxsmm_layout->dim_size[3];
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_train, libxsmm_input_train, LIBXSMM_DNN_REGULAR_INPUT ) );

    if(gp->eltwise)
    {
      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train, LIBXSMM_DNN_REGULAR_INPUT_ADD, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input_add_train = libxsmm_dnn_link_tensor(libxsmm_layout, inp_l, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train, libxsmm_input_add_train, LIBXSMM_DNN_REGULAR_INPUT_ADD ) )
    }

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train, LIBXSMM_DNN_CHANNEL_EXPECTVAL, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_expectval_train  = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)bexpect, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train, libxsmm_expectval_train, LIBXSMM_DNN_CHANNEL_EXPECTVAL ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train, LIBXSMM_DNN_CHANNEL_RCPSTDDEV, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_stddev_train = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)bstddev, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train, libxsmm_stddev_train, LIBXSMM_DNN_CHANNEL_RCPSTDDEV ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train, LIBXSMM_DNN_CHANNEL_VARIANCE, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_variance_train = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)bvariance, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_train, libxsmm_variance_train, LIBXSMM_DNN_CHANNEL_VARIANCE ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_gamma_train = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)gamma, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_train, libxsmm_gamma_train, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_train, LIBXSMM_DNN_REGULAR_CHANNEL_BETA, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_beta_train = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)beta, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_train, libxsmm_beta_train, LIBXSMM_DNN_REGULAR_CHANNEL_BETA ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train, LIBXSMM_DNN_REGULAR_OUTPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_output_train = libxsmm_dnn_link_tensor( libxsmm_layout, output, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train, libxsmm_output_train, LIBXSMM_DNN_REGULAR_OUTPUT ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train, LIBXSMM_DNN_RELU_MASK, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_relumask_train = libxsmm_dnn_link_tensor( libxsmm_layout, relu_mask, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_train, libxsmm_relumask_train, LIBXSMM_DNN_RELU_MASK) );
  }

  /* let's allocate (if required) and bind scratch */
  int max_size = 0;
  if(sptrptr[0] == NULL)
  {
    long long int mysize = libxsmm_dnn_fusedbatchnorm_get_scratch_size( libxsmm_handle_train, &status );
    CHKERR_LIBXSMM_DNN( status );
    sptrptr[0] = (void*)libxsmm_aligned_malloc(mysize , 2097152);
    max_size = mysize;

#ifdef USE_MLSL
    if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
      printf("%s allocated %lld bytes for scratch @ %p\n",nname.c_str(), mysize, sptrptr[0]);
  }
  else
  {
    long long int ssize = scratchp->getBufferSize();
    long long int mysize = libxsmm_dnn_fusedbatchnorm_get_scratch_size( libxsmm_handle_train, &status );

    CHKERR_LIBXSMM_DNN( status );

    if(ssize < mysize)
    {
      libxsmm_free(sptrptr[0]);
      sptrptr[0] = (void*)libxsmm_aligned_malloc(mysize, 2097152);
      scratchp->setBufferSize(mysize);
#ifdef USE_MLSL
      if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
        printf("%s allocated %lld bytes for scratch @ %p, prev size was %lld bytes\n",nname.c_str(), mysize, sptrptr[0], ssize);
    }
    else
      max_size = ssize;
  }
  scratchp->setBufferSize(max_size);

  if(libxsmm_input_test == NULL && libxsmm_input_add_test == NULL && libxsmm_expectval_test == NULL &&
      libxsmm_stddev_test == NULL && libxsmm_variance_test == NULL && libxsmm_gamma_test == NULL &&
      libxsmm_beta_test == NULL && libxsmm_output_test == NULL)
  {
    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_test, LIBXSMM_DNN_REGULAR_INPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_input_test  = libxsmm_dnn_link_tensor( libxsmm_layout, inp_r, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_test, libxsmm_input_test, LIBXSMM_DNN_REGULAR_INPUT ) );

    if(gp->eltwise)
    {
      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_test, LIBXSMM_DNN_REGULAR_INPUT_ADD, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input_add_test = libxsmm_dnn_link_tensor( libxsmm_layout, inp_l, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_test, libxsmm_input_add_test, LIBXSMM_DNN_REGULAR_INPUT_ADD ) )
    }

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_test, LIBXSMM_DNN_CHANNEL_EXPECTVAL, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_expectval_test = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)bexpect, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_test, libxsmm_expectval_test, LIBXSMM_DNN_CHANNEL_EXPECTVAL ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_test, LIBXSMM_DNN_CHANNEL_RCPSTDDEV, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_stddev_test  = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)bstddev, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_test, libxsmm_stddev_test, LIBXSMM_DNN_CHANNEL_RCPSTDDEV ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_test, LIBXSMM_DNN_CHANNEL_VARIANCE, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_variance_test  = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)bvariance, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_test, libxsmm_variance_test, LIBXSMM_DNN_CHANNEL_VARIANCE ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_test, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_gamma_test  = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)gamma, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_test, libxsmm_gamma_test, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_test, LIBXSMM_DNN_REGULAR_CHANNEL_BETA, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_beta_test  = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)beta, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_test, libxsmm_beta_test, LIBXSMM_DNN_REGULAR_CHANNEL_BETA ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_test, LIBXSMM_DNN_REGULAR_OUTPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_output_test = libxsmm_dnn_link_tensor( libxsmm_layout, output, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_test, libxsmm_output_test, LIBXSMM_DNN_REGULAR_OUTPUT ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_test, LIBXSMM_DNN_RELU_MASK, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_relumask_test = libxsmm_dnn_link_tensor( libxsmm_layout, relu_mask, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_test, libxsmm_relumask_test, LIBXSMM_DNN_RELU_MASK) );
  }

  if(!updated_scratch_fwd)
  {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_scratch( libxsmm_handle_train, sptrptr[0] ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_scratch( libxsmm_handle_test, sptrptr[0] ) );
    updated_scratch_fwd = true;
  }

#ifndef NDEBUG
  if ( (oph > 0 || opw > 0) && (iph > 0 || ipw > 0) ) {
    printf("node %s: batchnorm forward input and output is padded which cannot be :-(\n", nname.c_str());
  }

  /* check rims */
  if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
  {
    check_physical_pad( nname.c_str(),    (float*)inp_r, nImg, nFM/64, ifh,  ifw,  64, iph, ipw );
    check_physical_pad( nname.c_str(),     (float*)output, nImg, nFM/64, ofh, ofw, 64, oph,  opw );
  }
  else if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_BF16)
  {
    check_physical_pad( nname.c_str(),    (libxsmm_bfloat16*)inp_r, nImg, nFM/64, ifh,  ifw,  64, iph, ipw );
    check_physical_pad( nname.c_str(),     (libxsmm_bfloat16*)output, nImg, nFM/64, ofh, ofw, 64, oph,  opw );
  }
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
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_execute_st( libxsmm_handle_train, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
    }

#ifndef NDEBUG
    if ( (oph > 0 || opw > 0) && (iph > 0 || ipw > 0) ) {
      printf("node %s: batchnorm forward input and output is padded which cannot be :-(\n", nname.c_str());
    }

    /* check rims */
    if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
    {
      check_physical_pad( nname.c_str(),    (float*)inp_r, nImg, nFM/64, ifh,  ifw,  64, iph, ipw );
      check_physical_pad( nname.c_str(),     (float*)output, nImg, nFM/64, ofh, ofw, 64, oph,  opw );
    }
    else if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_BF16)
    {
      check_physical_pad( nname.c_str(),    (libxsmm_bfloat16*)inp_r, nImg, nFM/64, ifh,  ifw,  64, iph, ipw );
      check_physical_pad( nname.c_str(),     (libxsmm_bfloat16*)output, nImg, nFM/64, ofh, ofw, 64, oph, opw );
    }
#endif

    if(gp->exec_mode == "TRAIN")
    {
      float nhw_ratio = float(nImg*ifh*ifw)/float(nImg*ifh*ifw - 1);
#if 0
      float (* __restrict bmean)[VLEN] = (float (*)[VLEN])bexpect;
      float (* __restrict bvar)[VLEN] = (float (*)[VLEN])bvariance;

#ifdef __AVX512F__
      __m512  vmmf       = _mm512_set1_ps(gp->mmf);
      __m512  vnhw_ratio = _mm512_set1_ps(nhw_ratio);

      for (int b = 0; b < nBfm; ++b) {
        __m512 vbm = _mm512_load_ps(&bmean[b][0]);
        __m512 vbvar = _mm512_load_ps(&bvar[b][0]);

        _mm512_store_ps( &(gmean[b*VLEN]), _mm512_add_ps(_mm512_mul_ps(_mm512_load_ps( &(gmean[b*VLEN]) ), vmmf), vbm));
        _mm512_store_ps( &(gvar[b*VLEN]), _mm512_add_ps( _mm512_mul_ps( _mm512_load_ps( &(gvar[b*VLEN]) ), vmmf), _mm512_mul_ps(vnhw_ratio, vbvar)));
      }
#else

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int b = 0; b < nBfm; ++b) {
#pragma omp simd
        for (int v = 0; v < 16; ++v) {
          gmean[(b*16)+v] = gmean[(b*16)+v] * gp->mmf + bmean[b][v];
          gvar[(b*16)+v] = gvar[(b*16)+v] * gp->mmf + nhw_ratio*bvar[b][v];
        }
      }
#endif
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<nFM; i++)
      {
        gmean[i] = gmean[i]*gp->mmf + bexpect[i];
        gvar[i] = gvar[i]*gp->mmf + nhw_ratio*bvariance[i];
      }
#endif
      scaling_factor_ *= gp->mmf;
      scaling_factor_ += 1.;
    }
  }
  else
  {
#pragma omp parallel for
    for(int i=0; i < nFM; i++)
    {
      bexpect[i] = gmean[i]/scaling_factor_;
      float tmp = gvar[i]/scaling_factor_;
      bstddev[i] = 1./sqrt(tmp + gp->eps);
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
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_execute_st( libxsmm_handle_test, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
    }
  }
}

void FusedBNormXSMM::backPropagate(TensorBuf *deloutpb, TensorBuf *delgammapb, TensorBuf *delbetapb, vector<TensorBuf*> delinpb, int tid)
{
  int nImg  = gp->batch_size;
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

  void *deloutput = deloutpb->getBuffer();
  void *delinp_r = delinpb[0]->getBuffer();
  void *delinp_l = gp->eltwise ? delinpb[1]->getBuffer() : NULL;
  float *delgamma = (float*)delgammapb->getBuffer();
  float *delbeta = (float*)delbetapb->getBuffer();

#if 0
  {
    if(gp->in_data_type == DT_FLOAT)
    {
      float (* __restrict del_input_r)[nBlocksFm][ifhp][ifwp][64] = (float (*)[*][*][*][64])delinp_r;

      /* zero the rims in case of physical padding */
      /* @TODO, we need to do the same thing with del_input_l?! */
      if (iph > 0 || ipw > 0) {
#pragma omp parallel for
        for (int img = 0; img < nImg; img++) {
          for (int fm = 0; fm < nBlocksFm; fm++) {
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
      libxsmm_bfloat16 (* __restrict del_input_r)[nBlocksFm][ifhp][ifwp][64] = (libxsmm_bfloat16 (*)[*][*][*][64])delinp_r;

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
#ifndef NDEBUG
  if ( (oph > 0 || opw > 0) && (iph > 0 || ipw > 0) ) {
    printf("node %s: batchnorm backward input and output is padded which cannot be :-(\n", nname.c_str());
  }

  /* check rims */
  if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
  {
    check_physical_pad( nname.c_str(), (float*)delinp_r, nImg, nFM/64, ifh,  ifw,  64, iph, ipw );
    check_physical_pad( nname.c_str(),  (float*)deloutput, nImg, nFM/64, ofh, ofw, 64, oph,  opw );
  }
  else if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_BF16)
  {
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)delinp_r, nImg, nFM/64, ifh,  ifw,  64, iph, ipw );
    check_physical_pad( nname.c_str(),  (libxsmm_bfloat16*)deloutput, nImg, nFM/64, ofh, ofw, 64, oph,  opw );
  }
#endif

  void **sptrptr = scratchp->getBufferPtr();
  if(!updated_scratch_bwd)
  {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_scratch( libxsmm_handle_train, sptrptr[0] ) );
    updated_scratch_bwd = true;
  }

  if(libxsmm_deloutput == NULL && libxsmm_delinput == NULL && libxsmm_delinput_add == NULL &&
      libxsmm_delgamma == NULL && libxsmm_delbeta == NULL)
  {
    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train, LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_deloutput = libxsmm_dnn_link_tensor( libxsmm_layout, deloutput, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_train, libxsmm_deloutput, LIBXSMM_DNN_GRADIENT_OUTPUT ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_train, LIBXSMM_DNN_GRADIENT_INPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_delinput = libxsmm_dnn_link_tensor( libxsmm_layout, delinp_r, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train, libxsmm_delinput, LIBXSMM_DNN_GRADIENT_INPUT ) );

    if(gp->eltwise)
    {
      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_train, LIBXSMM_DNN_GRADIENT_INPUT_ADD, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_delinput_add = libxsmm_dnn_link_tensor( libxsmm_layout, delinp_l, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train, libxsmm_delinput_add, LIBXSMM_DNN_GRADIENT_INPUT_ADD ) );
    }

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_train, LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_delgamma = libxsmm_dnn_link_tensor( libxsmm_layout, delgamma, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train, libxsmm_delgamma, LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_train, LIBXSMM_DNN_GRADIENT_CHANNEL_BETA, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_delbeta = libxsmm_dnn_link_tensor( libxsmm_layout, delbeta, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_train, libxsmm_delbeta, LIBXSMM_DNN_GRADIENT_CHANNEL_BETA ) );
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
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_execute_st( libxsmm_handle_train, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
  }

  /* Perform physical padding tests */
#ifndef NDEBUG
  if ( (oph > 0 || opw > 0) && (iph > 0 || ipw > 0) ) {
    printf("node %s: batchnorm backward input and output is padded which cannot be :-(\n", nname.c_str());
  }

  /* check rims */
  if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
  {
    check_physical_pad( nname.c_str(), (float*)delinp_r, nImg, nFM/64, ifh,  ifw,  64, iph, ipw );
    check_physical_pad( nname.c_str(),  (float*)deloutput, nImg, nFM/64, ofh, ofw, 64, oph,  opw );
  }
  else if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_BF16)
  {
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)delinp_r, nImg, nFM/64, ifh,  ifw,  64, iph, ipw );
    check_physical_pad( nname.c_str(),  (libxsmm_bfloat16*)deloutput, nImg, nFM/64, ofh, ofw, 64, oph,  opw );
  }
#endif
}

