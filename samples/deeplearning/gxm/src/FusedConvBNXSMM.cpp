/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
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

#include "FusedConvBNXSMM.hpp"

using namespace std;

FusedConvBNXSMM::FusedConvBNXSMM(FusedConvBNImplParams* gp, int engine) : FusedConvBNImpl(gp, engine)
{
  conv_desc.N = gp->batch_size;
  conv_desc.C = gp->nInput[0];
  conv_desc.H = gp->iHeight;
  conv_desc.W = gp->iWidth;
  conv_desc.K = gp->nOutput;
  conv_desc.R = gp->kh;
  conv_desc.S = gp->kw;
  conv_desc.u = gp->c_stride_h;
  conv_desc.v = gp->c_stride_w;

  if(gp->physical_padding)
  {
    conv_desc.pad_h_in = gp->ipad_h;
    conv_desc.pad_w_in = gp->ipad_w;
  }
  else
  {
    conv_desc.pad_h_in = 0;
    conv_desc.pad_w_in = 0;
  }

  conv_desc.pad_w = gp->ipad_w;
  conv_desc.pad_h = gp->ipad_h;

  if(gp->physical_padding)
  {
    conv_desc.pad_h_out = gp->mpad_h;
    conv_desc.pad_w_out = gp->mpad_w;
  }
  else
  {
    conv_desc.pad_h_out = 0;
    conv_desc.pad_w_out = 0;
  }

  conv_desc.threads = gp->num_threads;
  conv_desc.algo = LIBXSMM_DNN_CONV_ALGO_DIRECT;
  conv_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  conv_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
  if(gp->out_data_type == DT_FLOAT)
    conv_desc.options = LIBXSMM_DNN_CONV_OPTION_OVERWRITE;
  else if(gp->out_data_type == DT_BF16)
    conv_desc.options = LIBXSMM_DNN_CONV_OPTION_F32_BF16_CVT_RNE_OVERWRITE;

  if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_FLOAT)
  {
    conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_BF16;
    conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
  }
  else if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_BF16)
  {
    conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_BF16;
    conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_BF16;
  }
  else if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
  {
    conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
  }

  libxsmm_handle_conv = libxsmm_dnn_create_conv_layer( conv_desc, &status );
  CHKERR_LIBXSMM_DNN( status );

  fusedbn_desc_train.N = gp->batch_size;
  fusedbn_desc_train.C = gp->nOutput;
  fusedbn_desc_train.H = gp->mHeight;
  fusedbn_desc_train.W = gp->mWidth;
  fusedbn_desc_train.u = gp->bn_stride_h;
  fusedbn_desc_train.v = gp->bn_stride_w;
  fusedbn_desc_train.pad_h_in = gp->mpad_h;
  fusedbn_desc_train.pad_w_in = gp->mpad_w;
  fusedbn_desc_train.pad_h_out = gp->opad_h;
  fusedbn_desc_train.pad_w_out = gp->opad_w;
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
  if(gp->relu_fwd)
    fusedbn_desc_train.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_RELU;
  if(gp->eltwise)
    fusedbn_desc_train.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_ELTWISE;
  if(gp->relu_fwd && gp->eltwise)
    fusedbn_desc_train.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_ELTWISE_RELU;

  libxsmm_handle_bn_train = libxsmm_dnn_create_fusedbatchnorm( fusedbn_desc_train, &status );
  CHKERR_LIBXSMM_DNN( status );

  fusedbn_desc_test.N = gp->batch_size;
  fusedbn_desc_test.C = gp->nOutput;
  fusedbn_desc_test.H = gp->mHeight;
  fusedbn_desc_test.W = gp->mWidth;
  fusedbn_desc_test.u = gp->bn_stride_h;
  fusedbn_desc_test.v = gp->bn_stride_w;
  fusedbn_desc_test.pad_h_in = gp->mpad_h;
  fusedbn_desc_test.pad_w_in = gp->mpad_w;
  fusedbn_desc_test.pad_h_out = gp->opad_h;
  fusedbn_desc_test.pad_w_out = gp->opad_w;
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
  if(gp->relu_fwd)
    fusedbn_desc_test.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_RELU;
  if(gp->eltwise)
    fusedbn_desc_test.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE;
  if(gp->relu_fwd && gp->eltwise)
    fusedbn_desc_test.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE_RELU;

  libxsmm_handle_bn_test = libxsmm_dnn_create_fusedbatchnorm( fusedbn_desc_test, &status );
  CHKERR_LIBXSMM_DNN( status );
}

void FusedConvBNXSMM::forwardPropagate(vector<TensorBuf *>& inp, TensorBuf *weightp, TensorBuf *hweightp, TensorBuf *midp, TensorBuf *gammap, TensorBuf *betap, TensorBuf *meanp, TensorBuf *varp, TensorBuf *outp, int tid)
{
  int nImg = gp->batch_size;
  int nIFM = gp->nInput[0];
  int nOFM = gp->nOutput;
  int nBIfm = nIFM/VLEN;
  int nBOfm = nOFM/VLEN;
  int ifh = gp->iHeight;
  int ifw = gp->iWidth;
  int mfh = gp->mHeight;
  int mfw = gp->mWidth;
  int ofh = gp->oHeight;
  int ofw = gp->oWidth;
  int bsh = gp->bn_stride_h;
  int bsw = gp->bn_stride_w;
  int csh = gp->c_stride_h;
  int csw = gp->c_stride_w;
  int iph = gp->ipad_h;
  int ipw = gp->ipad_w;
  int mph = gp->mpad_h;
  int mpw = gp->mpad_w;
  int oph = gp->opad_h;
  int opw = gp->opad_w;
  int fbhs = ofh/bsh;
  int fbws = ofw/bsw;
  int fhm = mfh + 2*mph;
  int fwm = mfw + 2*mpw;

  assert(bot_compute_engine[0] != -1);
  assert(top_compute_engine[0] != -1);

  // Conv input. LPBuffer is non-NULL if data layer output is BF16
  void *inp_r, *inp_l, *wt_ptr, *hwt_ptr, *middle, *output;

  if(inp[0]->getLPBuffer() != NULL)
    inp_r = inp[0]->getLPBuffer();
  else
    inp_r = inp[0]->getBuffer();

  if(gp->eltwise)
    if(inp[1]->getLPBuffer() != NULL)
      inp_l = inp[1]->getLPBuffer();
    else
      inp_l = inp[1]->getBuffer();

  // Conv Weight
  if(weightp->getLPBuffer() != NULL)
    wt_ptr = weightp->getLPBuffer();
  else
    wt_ptr = weightp->getBuffer();
  void *wt_prv_ptr = NULL;

  // Conv weight history
  if(hweightp != NULL)
  {
    if(hweightp->getLPBuffer() != NULL)
      hwt_ptr = hweightp->getLPBuffer();
    else
      hwt_ptr = hweightp->getBuffer();
  }

  // Conv output
  middle = midp->getBuffer();
  output = outp->getBuffer();

  void *gamma = gammap->getBuffer();
  void *beta = betap->getBuffer();
  float *gexpect = (float*)meanp->getBuffer();
  float *gvar = (float*)varp->getBuffer();
  float *gexp_test = (float*)meanp->getPrivBuffer();
  float *gvar_test = (float*)varp->getPrivBuffer();

  if(bexpect == NULL)
  {
    bexpect = (void*)_mm_malloc(nOFM*sizeof(float), 64);

#ifndef NDEBUG
    printf("%s allocated %lu bytes for mean\n",nname.c_str(), nOFM*sizeof(float));
#endif
  }

  if(bstddev == NULL)
  {
    bstddev = (void*)_mm_malloc(nOFM*sizeof(float), 64);

#ifndef NDEBUG
    printf("%s allocated %lu bytes for stdev\n",nname.c_str(), nOFM*sizeof(float));
#endif
  }

  if(bvariance == NULL)
  {
    bvariance = (void*)_mm_malloc(nOFM*sizeof(float), 64);

#ifndef NDEBUG
    printf("%s allocated %lu bytes for variance\n",nname.c_str(), nOFM*sizeof(float));
#endif
  }

  if(gexp_test == NULL)
  {
    gexp_test = (float*)_mm_malloc(nOFM*sizeof(float), 64);
    meanp->setPrivBuffer((void*)gexp_test);

#ifndef NDEBUG
    printf("%s allocated %lu bytes for mean test\n",nname.c_str(), nOFM*sizeof(float));
#endif
  }

  if(gvar_test == NULL)
  {
    gvar_test = (float*)_mm_malloc(nOFM*sizeof(float), 64);
    varp->setPrivBuffer((void*)gvar_test);

#ifndef NDEBUG
    printf("%s allocated %lu bytes for mean test\n",nname.c_str(), nOFM*sizeof(float));
#endif
  }

  if(scratch != NULL)
  {
    if(updated_scratch && scratch != scratchp->getBuffer())
    {
      printf("Warning: updating scratch from %p to %p\n",scratch, scratchp->getBuffer());
      scratch = scratchp->getBuffer();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle_conv, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_scratch( libxsmm_handle_bn_train, scratch ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_scratch( libxsmm_handle_bn_test, scratch ) );
    }
  }
  else
    scratch = scratchp->getBuffer();

  if(libxsmm_input == NULL && libxsmm_filter == NULL && libxsmm_middle == NULL)
  {
    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle_conv, LIBXSMM_DNN_REGULAR_INPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_input = libxsmm_dnn_link_tensor( libxsmm_layout, inp_r, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle_conv, libxsmm_input, LIBXSMM_DNN_REGULAR_INPUT ) );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle_conv, LIBXSMM_DNN_REGULAR_FILTER, &status );
    CHKERR_LIBXSMM_DNN( status );

    int welem = gp->nInput[0] * gp->nOutput * gp->kw * gp->kh;
    if(gp->in_data_type == DT_FLOAT)
    {
      wt_prv_ptr = (void*)libxsmm_aligned_malloc(welem*sizeof(float), 2097152);

      // Transform weight layout
      libxsmm_filter = libxsmm_dnn_link_tensor( libxsmm_layout, wt_prv_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );

      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_filter, (void*)wt_ptr, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );
      memcpy(wt_ptr, wt_prv_ptr, welem*sizeof(float));

      libxsmm_checkpoint_filter = libxsmm_dnn_link_tensor(libxsmm_layout, wt_ptr, &status);
      CHKERR_LIBXSMM_DNN( status );

      libxsmm_filter = libxsmm_dnn_link_tensor( libxsmm_layout, wt_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );

      // Transform weight history layout
      if(hwt_ptr != NULL)
      {
        libxsmm_temp = libxsmm_dnn_link_tensor( libxsmm_layout, wt_prv_ptr, &status );
        CHKERR_LIBXSMM_DNN( status );

        CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_temp, (void*)hwt_ptr, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );
        memcpy(hwt_ptr, wt_prv_ptr, welem*sizeof(float));

        libxsmm_checkpoint_history_filter = libxsmm_dnn_link_tensor(libxsmm_layout, hwt_ptr, &status);
        CHKERR_LIBXSMM_DNN( status );
      }

      libxsmm_free(wt_prv_ptr);
      wt_prv_ptr = NULL;
      weightp->setPrivBuffer(NULL);
    }
    else if(gp->in_data_type == DT_BF16)
    {
      wt_prv_ptr = (void*)libxsmm_aligned_malloc(welem*sizeof(libxsmm_bfloat16), 2097152);

      // Transform BF16 weight layout
      libxsmm_filter = libxsmm_dnn_link_tensor( libxsmm_layout, wt_prv_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );

      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_filter, (void*)wt_ptr, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );
      memcpy(wt_ptr, wt_prv_ptr, welem*sizeof(libxsmm_bfloat16));
      libxsmm_filter = libxsmm_dnn_link_tensor( libxsmm_layout, wt_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_free(wt_prv_ptr);

      // Transform FP32 weight layout
      libxsmm_layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
      wt_prv_ptr = (void*)libxsmm_aligned_malloc(welem*sizeof(float), 2097152);
      libxsmm_checkpoint_filter = libxsmm_dnn_link_tensor( libxsmm_layout, wt_prv_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );
      void *fwt_ptr = weightp->getBuffer();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_checkpoint_filter, (void*)fwt_ptr, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );
      memcpy(fwt_ptr, wt_prv_ptr, welem*sizeof(float));

      libxsmm_checkpoint_filter = libxsmm_dnn_link_tensor( libxsmm_layout, fwt_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );

      // Transform FP32 weight history layout
      if(hwt_ptr != NULL)
      {
        libxsmm_checkpoint_history_filter = libxsmm_dnn_link_tensor( libxsmm_layout, wt_prv_ptr, &status );
        CHKERR_LIBXSMM_DNN( status );

        void *hfwt_ptr = hweightp->getBuffer();
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_checkpoint_history_filter, (void*)hfwt_ptr, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );
        memcpy(hfwt_ptr, wt_prv_ptr, welem*sizeof(float));

        libxsmm_checkpoint_history_filter = libxsmm_dnn_link_tensor(libxsmm_layout, hfwt_ptr, &status);
        CHKERR_LIBXSMM_DNN( status );
      }

      libxsmm_free(wt_prv_ptr);
      wt_prv_ptr = NULL;
      weightp->setPrivBuffer(NULL);
    }

    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle_conv, libxsmm_filter, LIBXSMM_DNN_REGULAR_FILTER ) );

    // Conv Output
    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle_conv, LIBXSMM_DNN_REGULAR_OUTPUT, &status );
    CHKERR_LIBXSMM_DNN(      status );
    libxsmm_middle = libxsmm_dnn_link_tensor( libxsmm_layout, middle, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle_conv, libxsmm_middle, LIBXSMM_DNN_REGULAR_OUTPUT ) );

    /* let's allocate (if required) and bind scratch */
    if(scratch == NULL)
    {
      long long int mysize = libxsmm_dnn_get_scratch_size( libxsmm_handle_conv, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
      CHKERR_LIBXSMM_DNN( status );
      scratch = (void*)libxsmm_aligned_malloc(mysize , 2097152);
      scratchp->setBuffer(scratch);
      scratchp->setBufferSize(mysize);

#ifdef USE_MLSL
      if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
        printf("%s allocated %lld bytes for scratch @ %p\n",nname.c_str(), mysize, scratch);
    }
    else
    {
      long long int ssize = scratchp->getBufferSize();
      long long int mysize = libxsmm_dnn_get_scratch_size( libxsmm_handle_conv, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );

      CHKERR_LIBXSMM_DNN( status );

      if(ssize < mysize)
      {
        libxsmm_free(scratch);
        scratch = (void*)libxsmm_aligned_malloc(mysize, 2097152);
        scratchp->setBuffer(scratch);
        scratchp->setBufferSize(mysize);
#ifdef USE_MLSL
        if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
          printf("%s allocated %lld bytes for scratch @ %p, prev size was %lld bytes\n",nname.c_str(), mysize, scratch, ssize);
      }
    }
  }
    
  if (libxsmm_input_bntrain==NULL && libxsmm_input_add_bntrain == NULL && libxsmm_output_bntrain == NULL && libxsmm_gamma_train == NULL && libxsmm_beta_train == NULL && libxsmm_expectval_train == NULL && libxsmm_stddev_train == NULL && libxsmm_variance_train == NULL)
  {
    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_train, LIBXSMM_DNN_REGULAR_INPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_input_bntrain  = libxsmm_dnn_link_tensor( libxsmm_layout, middle, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_train, libxsmm_input_bntrain, LIBXSMM_DNN_REGULAR_INPUT ) );

    if(gp->eltwise)
    {
      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_train, LIBXSMM_DNN_REGULAR_INPUT_ADD, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input_add_bntrain = libxsmm_dnn_link_tensor( libxsmm_layout, inp_l, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_train, libxsmm_input_add_bntrain, LIBXSMM_DNN_REGULAR_INPUT_ADD ) )
    }

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_train, LIBXSMM_DNN_CHANNEL_EXPECTVAL, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_expectval_train  = libxsmm_dnn_link_tensor( libxsmm_layout, bexpect, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_train, libxsmm_expectval_train, LIBXSMM_DNN_CHANNEL_EXPECTVAL ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_train, LIBXSMM_DNN_CHANNEL_RCPSTDDEV, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_stddev_train  = libxsmm_dnn_link_tensor( libxsmm_layout, bstddev, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_train, libxsmm_stddev_train, LIBXSMM_DNN_CHANNEL_RCPSTDDEV ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_train, LIBXSMM_DNN_CHANNEL_VARIANCE, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_variance_train  = libxsmm_dnn_link_tensor( libxsmm_layout, bvariance, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_train, libxsmm_variance_train, LIBXSMM_DNN_CHANNEL_VARIANCE ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_train, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_gamma_train  = libxsmm_dnn_link_tensor( libxsmm_layout, gamma, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_train, libxsmm_gamma_train, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_bn_train, LIBXSMM_DNN_REGULAR_CHANNEL_BETA, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_beta_train  = libxsmm_dnn_link_tensor( libxsmm_layout, beta, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_train, libxsmm_beta_train, LIBXSMM_DNN_REGULAR_CHANNEL_BETA ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_train, LIBXSMM_DNN_REGULAR_OUTPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_output_bntrain  = libxsmm_dnn_link_tensor( libxsmm_layout, output, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_train, libxsmm_output_bntrain, LIBXSMM_DNN_REGULAR_OUTPUT ) );

    /* let's allocate (if required) and bind scratch */
    if(scratch == NULL)
    {
      long long int mysize = libxsmm_dnn_fusedbatchnorm_get_scratch_size( libxsmm_handle_bn_train, &status );
      CHKERR_LIBXSMM_DNN( status );
      scratch = (void*)libxsmm_aligned_malloc(mysize , 2097152);
      scratchp->setBuffer(scratch);
      scratchp->setBufferSize(mysize);

#ifdef USE_MLSL
      if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
        printf("%s allocated %lld bytes for scratch @ %p\n",nname.c_str(), mysize, scratch);
    }
    else
    {
      long long int ssize = scratchp->getBufferSize();
      long long int mysize = libxsmm_dnn_fusedbatchnorm_get_scratch_size( libxsmm_handle_bn_train, &status );

      CHKERR_LIBXSMM_DNN( status );

      if(ssize < mysize)
      {
        libxsmm_free(scratch);
        scratch = (void*)libxsmm_aligned_malloc(mysize, 2097152);
        scratchp->setBuffer(scratch);
        scratchp->setBufferSize(mysize);
#ifdef USE_MLSL
        if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
          printf("%s allocated %lld bytes for scratch @ %p, prev size was %lld bytes\n",nname.c_str(), mysize, scratch, ssize);
      }
    }
  }

  if(libxsmm_input_bntest==NULL && libxsmm_input_add_bntest == NULL && libxsmm_output_bntest == NULL && libxsmm_gamma_test == NULL && libxsmm_beta_test == NULL && libxsmm_expectval_test == NULL && libxsmm_stddev_test == NULL && libxsmm_variance_test == NULL)
  {
    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_test, LIBXSMM_DNN_REGULAR_INPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_input_bntest  = libxsmm_dnn_link_tensor( libxsmm_layout, middle, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_test, libxsmm_input_bntest, LIBXSMM_DNN_REGULAR_INPUT ) );

    if(gp->eltwise)
    {
      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_test, LIBXSMM_DNN_REGULAR_INPUT_ADD, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input_add_bntest = libxsmm_dnn_link_tensor( libxsmm_layout, inp_l, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_test, libxsmm_input_add_bntest, LIBXSMM_DNN_REGULAR_INPUT_ADD ) )
    }

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_test, LIBXSMM_DNN_CHANNEL_EXPECTVAL, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_expectval_test  = libxsmm_dnn_link_tensor( libxsmm_layout, bexpect, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_test, libxsmm_expectval_test, LIBXSMM_DNN_CHANNEL_EXPECTVAL ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_test, LIBXSMM_DNN_CHANNEL_RCPSTDDEV, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_stddev_test  = libxsmm_dnn_link_tensor( libxsmm_layout, bstddev, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_test, libxsmm_stddev_test, LIBXSMM_DNN_CHANNEL_RCPSTDDEV ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_test, LIBXSMM_DNN_CHANNEL_VARIANCE, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_variance_test  = libxsmm_dnn_link_tensor( libxsmm_layout, bvariance, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_test, libxsmm_variance_test, LIBXSMM_DNN_CHANNEL_VARIANCE ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_test, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_gamma_test  = libxsmm_dnn_link_tensor( libxsmm_layout, gamma, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_test, libxsmm_gamma_test, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_bn_test, LIBXSMM_DNN_REGULAR_CHANNEL_BETA, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_beta_test  = libxsmm_dnn_link_tensor( libxsmm_layout, beta, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_test, libxsmm_beta_test, LIBXSMM_DNN_REGULAR_CHANNEL_BETA ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_test, LIBXSMM_DNN_REGULAR_OUTPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_output_bntest  = libxsmm_dnn_link_tensor( libxsmm_layout, output, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_test, libxsmm_output_bntest, LIBXSMM_DNN_REGULAR_OUTPUT ) );

    /* let's allocate (if required) and bind scratch */
    if(scratch == NULL)
    {
      long long int mysize = libxsmm_dnn_fusedbatchnorm_get_scratch_size( libxsmm_handle_bn_test, &status );
      CHKERR_LIBXSMM_DNN( status );
      scratch = (void*)libxsmm_aligned_malloc(mysize , 2097152);
      scratchp->setBuffer(scratch);
      scratchp->setBufferSize(mysize);

#ifdef USE_MLSL
      if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
        printf("%s allocated %lld bytes for scratch @ %p\n",nname.c_str(), mysize, scratch);
    }
    else
    {
      long long int ssize = scratchp->getBufferSize();
      long long int mysize = libxsmm_dnn_fusedbatchnorm_get_scratch_size( libxsmm_handle_bn_test, &status );

      CHKERR_LIBXSMM_DNN( status );

      if(ssize < mysize)
      {
        libxsmm_free(scratch);
        scratch = (void*)libxsmm_aligned_malloc(mysize, 2097152);
        scratchp->setBuffer(scratch);
        scratchp->setBufferSize(mysize);
#ifdef USE_MLSL
        if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
          printf("%s allocated %lld bytes for scratch @ %p, prev size was %lld bytes\n",nname.c_str(), mysize, scratch, ssize);
      }
    }
  }

  if(!updated_scratch)
  {
    scratch = scratchp->getBuffer();
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle_conv, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_scratch( libxsmm_handle_bn_train, scratch ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_scratch( libxsmm_handle_bn_test, scratch ) );
    updated_scratch = true;
  }

#ifndef NDEBUG
  /* check physical padding */
  if ( (iph > 0 || ipw > 0) && (mph > 0 || mpw > 0) ) {
  } else if ( (iph == 0 || ipw == 0) && (mph == 0 || mpw == 0) ) {
  } else {
    printf("node %s: conv xsmm forward is partially padded which cannot be :-(\n", nname.c_str());
  }

  if ( (oph > 0 || opw > 0) && (mph > 0 || mpw > 0) ) {
    printf("node %s: batchnorm forward input and output is padded which cannot be :-(\n", nname.c_str());
  }

  /* check rims */
  if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
  {
    if(nIFM > 3)
      check_physical_pad( nname.c_str(), (float*)inp_r, nImg, nBIfm, ifh, ifw, VLEN, iph, ipw );
    else
      check_physical_pad( nname.c_str(), (float*)inp_r, nImg, 1, ifh, ifw, 3, iph, ipw );
    check_physical_pad( nname.c_str(),    (float*)middle, nImg, nBOfm, mfh,  mfw,  VLEN, mph, mpw );
    check_physical_pad( nname.c_str(),     (float*)output, nImg, nBOfm, fbhs, fbws, VLEN, oph,  opw );
  }
  else if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_BF16)
  {
    if(nIFM > 3)
      check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)inp_r, nImg, nBIfm, ifh, ifw, VLEN, iph, ipw );
    else
      check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)inp_r, nImg, 1, ifh, ifw, 3, iph, ipw );
    check_physical_pad( nname.c_str(),    (libxsmm_bfloat16*)middle, nImg, nBOfm, mfh,  mfw,  VLEN, mph, mpw );
    check_physical_pad( nname.c_str(),     (libxsmm_bfloat16*)output, nImg, nBOfm, fbhs, fbws, VLEN, oph,  opw );
  }
#endif

  if(!use_global_stats)
  {
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif

      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle_conv, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_execute_st( libxsmm_handle_bn_train, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
    }

#ifndef NDEBUG
  /* check physical padding */
  if ( (iph > 0 || ipw > 0) && (mph > 0 || mpw > 0) ) {
  } else if ( (iph == 0 || ipw == 0) && (mph == 0 || mpw == 0) ) {
  } else {
    printf("node %s: conv xsmm forward is partially padded which cannot be :-(\n", nname.c_str());
  }

  if ( (oph > 0 || opw > 0) && (mph > 0 || mpw > 0) ) {
    printf("node %s: batchnorm forward input and output is padded which cannot be :-(\n", nname.c_str());
  }

  /* check rims */
  if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
  {
    if(nIFM > 3)
      check_physical_pad( nname.c_str(), (float*)inp_r, nImg, nBIfm, ifh, ifw, VLEN, iph, ipw );
    else
      check_physical_pad( nname.c_str(), (float*)inp_r, nImg, 1, ifh, ifw, 3, iph, ipw );
    check_physical_pad( nname.c_str(),    (float*)middle, nImg, nBOfm, mfh,  mfw,  VLEN, mph, mpw );
    check_physical_pad( nname.c_str(),     (float*)output, nImg, nBOfm, fbhs, fbws, VLEN, oph,  opw );
  }
  else if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_BF16)
  {
    if(nIFM > 3)
      check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)inp_r, nImg, nBIfm, ifh, ifw, VLEN, iph, ipw );
    else
      check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)inp_r, nImg, 1, ifh, ifw, 3, iph, ipw );
    check_physical_pad( nname.c_str(),    (libxsmm_bfloat16*)middle, nImg, nBOfm, mfh,  mfw,  VLEN, mph, mpw );
    check_physical_pad( nname.c_str(),     (libxsmm_bfloat16*)output, nImg, nBOfm, fbhs, fbws, VLEN, oph,  opw );
  }
#endif

    if(gp->exec_mode == "TRAIN")
    {
      float (* __restrict bmean)[VLEN] = (float (*)[VLEN])bexpect;
      float (* __restrict bvar)[VLEN] = (float (*)[VLEN])bvariance;
      float nhw_ratio = float(nImg*fbhs*fbws)/float(nImg*fbhs*fbws - 1);

#ifdef __AVX512F__
      __m512  vmmf       = _mm512_set1_ps(gp->mmf);
      __m512  vnhw_ratio = _mm512_set1_ps(nhw_ratio);

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int b = 0; b < nBOfm; ++b) {
        __m512 vbm = _mm512_load_ps(&bmean[b][0]);
        __m512 vbvar = _mm512_load_ps(&bvar[b][0]);

        _mm512_store_ps( &(gexpect[b*VLEN]), _mm512_add_ps(_mm512_mul_ps(_mm512_load_ps( &(gexpect[b*VLEN]) ), vmmf), vbm));
        _mm512_store_ps( &(gvar[b*VLEN]), _mm512_add_ps( _mm512_mul_ps( _mm512_load_ps( &(gvar[b*VLEN]) ), vmmf), _mm512_mul_ps(vnhw_ratio, vbvar)));
      }
#else

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int b = 0; b < nBOfm; ++b) {
#pragma omp simd
        for (int v = 0; v < 16; ++v) {
          gexpect[(b*16)+v] = gexpect[(b*16)+v] * gp->mmf + bmean[b][v];
          gvar[(b*16)+v] = gvar[(b*16)+v] * gp->mmf + nhw_ratio*bvar[b][v];
        }
      }
#endif

      scaling_factor_ *= gp->mmf;
      scaling_factor_ += 1.;
    }
  }
  else
  {
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for(int i=0; i < nOFM; i++)
    {
      ((float*)bexpect)[i] = gexpect[i]/scaling_factor_;
      float tmp = (float)gvar[i]/scaling_factor_;
      ((float*)bstddev)[i] = 1./sqrt(tmp + gp->eps);
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif

      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle_conv, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_execute_st( libxsmm_handle_bn_test, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
    }
  }
}

void FusedConvBNXSMM::backPropagate(TensorBuf *deloutp, TensorBuf* weightp, TensorBuf *delgammap, TensorBuf *delbetap, TensorBuf *delmidp, vector<TensorBuf*>& delinp, int tid)
{
  void *deloutput = deloutp->getBuffer();
  void *delmiddle = delmidp->getBuffer();
  void *delinp_r = delinp[0]->getBuffer();
  void *delinp_l = gp->eltwise ? delinp[1]->getBuffer() : NULL;
  void *delgamma = delgammap->getBuffer();
  void *delbeta = delbetap->getBuffer();

  int nImg  = gp->batch_size;
  int nIFM = gp->nInput[0];
  int nOFM = gp->nOutput;
  int nBIfm = nIFM/VLEN;
  int nBOfm = nOFM/VLEN;
  int ofh = gp->oHeight;
  int ofw = gp->oWidth;
  int mfh = gp->mHeight;
  int mfw = gp->mWidth;
  int ifh = gp->iHeight;
  int ifw = gp->iWidth;

  int iph = gp->ipad_h;
  int ipw = gp->ipad_w;
  int oph = gp->opad_h;
  int opw = gp->opad_w;
  int mph = gp->mpad_h;
  int mpw = gp->mpad_w;

  int bsh = gp->bn_stride_h;
  int bsw = gp->bn_stride_w;
  int csh = gp->c_stride_h;
  int csw = gp->c_stride_w;

  int fhbs = ofh/bsh;
  int fwbs = ofw/bsw;
  int fhm = mfh + 2*mph;
  int fwm = mfw + 2*mpw;
  int fhi = ifh + 2*iph;
  int fwi = ifw + 2*ipw;

  if(gp->in_data_type == DT_FLOAT)
  {
    float (* __restrict del_middle)[nBOfm][fhm][fwm][VLEN] = (float (*)[*][*][*][VLEN])delmiddle;

    /* zero the rims in case of physical padding */
    if (mph > 0 || mpw > 0) {
#pragma omp parallel for
      for (int img = 0; img < nImg; img++) {
        for (int fm = 0; fm < nBOfm; fm++) {
          for (int w = 0; w < fwm; w++) {
            for (int ph = 0; ph < mph; ph++) {
#ifdef __AVX512F__
              _mm512_stream_ps( &(del_middle[img][fm][ph      ][w][0]), _mm512_setzero_ps() );
              _mm512_stream_ps( &(del_middle[img][fm][fhm-1-ph][w][0]), _mm512_setzero_ps() );
#else
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
              for(int v=0; v < VLEN; v++) {
                del_middle[img][fm][ph][w][v] = 0.0f;
                del_middle[img][fm][fhm-1-ph][w][v] = 0.0f;
              }
#endif
            }
          }
          for (int h = mph; h < mfh+mph; h++) {
            for (int pw = 0; pw < mpw; pw++) {
#ifdef __AVX512F__
              _mm512_stream_ps( &(del_middle[img][fm][h][pw      ][0]), _mm512_setzero_ps() );
              _mm512_stream_ps( &(del_middle[img][fm][h][fwm-1-pw][0]), _mm512_setzero_ps() );
#else
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
              for(int v=0; v < VLEN; v++) {
                del_middle[img][fm][h][pw][v] = 0.0f;
                del_middle[img][fm][h][fwm-1-pw][v] = 0.0f;
              }
#endif
            }
          }
        }
      }
    }
  }
  else if(gp->in_data_type == DT_BF16)
  {
    libxsmm_bfloat16 (* __restrict del_middle)[nBOfm][fhm][fwm][VLEN] = (libxsmm_bfloat16 (*)[*][*][*][VLEN])delmiddle;

    /* zero the rims in case of physical padding */
    /* @TODO, we need to do the same thing with del_input_l?! */
    if (iph > 0 || iph > 0) {
#pragma omp parallel for
      for (int img = 0; img < nImg; img++) {
        for (int fm = 0; fm < nBOfm; fm++) {
          for (int w = 0; w < fwm; w++) {
            for (int ph = 0; ph < mph; ph++) {
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
              for(int v=0; v < VLEN; v++) {
                del_middle[img][fm][ph][w][v] = 0;
                del_middle[img][fm][fhm-1-ph][w][v] = 0;
              }
            }
          }
          for (int h = mph; h < mfh+mph; h++) {
            for (int pw = 0; pw < mpw; pw++) {
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
              for(int v=0; v < VLEN; v++) {
                del_middle[img][fm][h][pw][v] = 0;
                del_middle[img][fm][h][fwm-1-pw][v] = 0;
              }
            }
          }
        }
      }
    }
  }

  /* Perform physical padding tests */
#ifndef NDEBUG
  if ( (oph > 0 || opw > 0) && (mph > 0 || mpw > 0) ) {
    printf("node %s: batchnorm backward input and output is padded which cannot be :-(\n", nname.c_str());
  }

  /* check rims */
  if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
  {
    check_physical_pad( nname.c_str(), (float*)delmiddle, nImg, nBOfm, mfh,  mfw,  VLEN, mph, mpw );
    check_physical_pad( nname.c_str(),  (float*)deloutput, nImg, nBOfm, fhbs, fwbs, VLEN, oph,  opw );
  }
  else if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_BF16)
  {
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)delmiddle, nImg, nBOfm, mfh,  mfw,  VLEN, mph, mpw );
    check_physical_pad( nname.c_str(),  (libxsmm_bfloat16*)deloutput, nImg, nBOfm, fhbs, fwbs, VLEN, oph,  opw );
  }
#endif

  if(scratch != scratchp->getBuffer())
  {
    scratch = scratchp->getBuffer();
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_scratch( libxsmm_handle_bn_train, scratch ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle_conv, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
  }

  if(libxsmm_deloutput == NULL && libxsmm_delmiddle_bn == NULL && libxsmm_delinput_add == NULL &&
      libxsmm_delgamma == NULL && libxsmm_delbeta == NULL)
  {
    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_train, LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_deloutput = libxsmm_dnn_link_tensor( libxsmm_layout, deloutput, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_train, libxsmm_deloutput, LIBXSMM_DNN_GRADIENT_OUTPUT ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_train, LIBXSMM_DNN_GRADIENT_INPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_delmiddle_bn  = libxsmm_dnn_link_tensor( libxsmm_layout, delmiddle, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_train, libxsmm_delmiddle_bn, LIBXSMM_DNN_GRADIENT_INPUT ) );

    if(gp->eltwise)
    {
      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_bn_train, LIBXSMM_DNN_GRADIENT_INPUT_ADD, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_delinput_add  = libxsmm_dnn_link_tensor( libxsmm_layout, delinp_l, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_train, libxsmm_delinput_add, LIBXSMM_DNN_GRADIENT_INPUT_ADD ) );
    }

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_bn_train, LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_delgamma  = libxsmm_dnn_link_tensor( libxsmm_layout, delgamma, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_train, libxsmm_delgamma, LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA ) );

    libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_bn_train, LIBXSMM_DNN_GRADIENT_CHANNEL_BETA, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_delbeta  = libxsmm_dnn_link_tensor( libxsmm_layout, delbeta, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_train, libxsmm_delbeta, LIBXSMM_DNN_GRADIENT_CHANNEL_BETA ) );
  }

  /* Perform physical padding tests */
#ifndef NDEBUG
  if ( (oph > 0 || opw > 0) && (mph > 0 || mpw > 0) ) {
    printf("node %s: batchnorm backward input and output is padded which cannot be :-(\n", nname.c_str());
  }

  /* check rims */
  if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
  {
    check_physical_pad( nname.c_str(), (float*)delmiddle, nImg, nBOfm, mfh,  mfw,  VLEN, mph, mpw );
    check_physical_pad( nname.c_str(),  (float*)deloutput, nImg, nBOfm, fhbs, fwbs, VLEN, oph,  opw );
  }
  else if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_BF16)
  {
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)delmiddle, nImg, nBOfm, mfh,  mfw,  VLEN, mph, mpw );
    check_physical_pad( nname.c_str(),  (libxsmm_bfloat16*)deloutput, nImg, nBOfm, fhbs, fwbs, VLEN, oph,  opw );
  }
#endif

  if(libxsmm_delinput == NULL && libxsmm_delmiddle_conv == NULL)
  {
    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle_conv, LIBXSMM_DNN_GRADIENT_INPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_delinput = libxsmm_dnn_link_tensor(libxsmm_layout, delinp_r, &status );
    CHKERR_LIBXSMM_DNN(status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle_conv, libxsmm_delinput, LIBXSMM_DNN_GRADIENT_INPUT ) );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle_conv, LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
    CHKERR_LIBXSMM_DNN(status );
    libxsmm_delmiddle_conv = libxsmm_dnn_link_tensor( libxsmm_layout, delmiddle, &status );
    CHKERR_LIBXSMM_DNN(status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle_conv, libxsmm_delmiddle_conv, LIBXSMM_DNN_GRADIENT_OUTPUT ) );
  }

#ifndef NDEBUG
  /* check physical padding */
  if ( (gp->ipad_h > 0 || gp->ipad_w > 0) && (gp->mpad_h > 0 || gp->mpad_w > 0) ) {
  } else if ( (gp->ipad_h == 0 || gp->ipad_w == 0) && (gp->mpad_h == 0 || gp->mpad_w == 0) ) {
  } else {
    printf("node %s: conv xsmm backward is partially padded which cannot be :-(\n", nname.c_str());
  }
  if(gp->out_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)delinp_r, gp->batch_size, gp->nInput[0]/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );
  else if(gp->out_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)delinp_r, gp->batch_size, gp->nInput[0]/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );

  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)delmiddle, gp->batch_size, gp->nOutput/16, gp->mHeight, gp->mWidth, 16, gp->mpad_h, gp->mpad_w );
  else if(gp->in_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)delmiddle, gp->batch_size, gp->nOutput/16, gp->mHeight, gp->mWidth, 16, gp->mpad_h, gp->mpad_w );
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif

      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_execute_st( libxsmm_handle_bn_train, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle_conv, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
    }

#ifndef NDEBUG
  /* check physical padding */
  if ( (gp->ipad_h > 0 || gp->ipad_w > 0) && (gp->mpad_h > 0 || gp->mpad_w > 0) ) {
  } else if ( (gp->ipad_h == 0 || gp->ipad_w == 0) && (gp->mpad_h == 0 || gp->mpad_w == 0) ) {
  } else {
    printf("node %s: conv xsmm backward is partially padded which cannot be :-(\n", nname.c_str());
  }
  if(gp->out_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)delinp_r, gp->batch_size, gp->nInput[0]/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );
  else if(gp->out_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)delinp_r, gp->batch_size, gp->nInput[0]/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );

  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)delmiddle, gp->batch_size, gp->nOutput/16, gp->mHeight, gp->mWidth, 16, gp->mpad_h, gp->mpad_w );
  else if(gp->in_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)delmiddle, gp->batch_size, gp->nOutput/16, gp->mHeight, gp->mWidth, 16, gp->mpad_h, gp->mpad_w );
#endif
}

void FusedConvBNXSMM::weightUpdate(TensorBuf *inp, TensorBuf *delmidp, TensorBuf* delweightp, int tid)
{
  void *dwt_ptr = delweightp->getBuffer();
  void *delmiddle = delmidp->getBuffer();

  if(scratch != scratchp->getBuffer())
  {
    scratch = scratchp->getBuffer();
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle_conv, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
  }

  if(libxsmm_delfilter == NULL)
  {
    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle_conv, LIBXSMM_DNN_GRADIENT_FILTER, &status );
    CHKERR_LIBXSMM_DNN(status );
    libxsmm_delfilter = libxsmm_dnn_link_tensor( libxsmm_layout, dwt_ptr, &status );
    CHKERR_LIBXSMM_DNN(status);
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle_conv, libxsmm_delfilter, LIBXSMM_DNN_GRADIENT_FILTER ) );
  }

  if(libxsmm_delmiddle_conv == NULL)
  {
    delmiddle = delmidp->getBuffer();
    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle_conv, LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
    CHKERR_LIBXSMM_DNN(      status );
    libxsmm_delmiddle_conv = libxsmm_dnn_link_tensor(libxsmm_layout, delmiddle, &status );
    CHKERR_LIBXSMM_DNN(status);
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle_conv, libxsmm_delmiddle_conv, LIBXSMM_DNN_GRADIENT_OUTPUT ) );
  }

#ifndef NDEBUG
  /* check physical padding */
  if ( (gp->ipad_h > 0 || gp->ipad_w > 0) && (gp->mpad_h > 0 || gp->mpad_w > 0) ) {
  } else if ( (gp->ipad_h == 0 || gp->ipad_w == 0) && (gp->mpad_h == 0 || gp->mpad_w == 0) ) {
  } else {
    printf("node %s: conv xsmm backward is partially padded which cannot be :-(\n", nname.c_str());
  }
  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)delmiddle, gp->batch_size, gp->nOutput/16, gp->mHeight, gp->mWidth, 16, gp->mpad_h, gp->mpad_w );
  else if(gp->in_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)delmiddle, gp->batch_size, gp->nOutput/16, gp->mHeight, gp->mWidth, 16, gp->mpad_h, gp->mpad_w );
#endif

#ifdef USE_XSMM_TIMING
  struct timeval tvsc, tvec;
  gettimeofday(&tvsc, NULL);
#endif
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif

      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle_conv, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid ) );
    }

#ifdef USE_XSMM_TIMING
  gettimeofday(&tvec, NULL);
  double wu_time = (tvec.tv_sec + tvec.tv_usec*1e-6) - (tvsc.tv_sec + tvsc.tv_usec*1e-6);

#ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
  {
    double gf = (double)gp->batch_size * (double)gp->nInput * (double)gp->nOutput * (double)gp->mHeight * (double)gp->mWidth * (double)gp->kh * (double)gp->kw * 2;
    if(gp->stride_h == 1 && gp->pad_h == 0)
      printf("XSMM-CONV-WU mb%dic%dih%doc%doh%dkh%dn time = %g ms, GFLOPS = %.1f\n",gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->mHeight,gp->kh,wu_time*1000.0, gf/wu_time/1e9);
    else if(gp->stride_h == 2)
      printf("XSMM-CONV-WU mb%dic%dih%doc%doh%dkh%dsh%dn time = %g ms, GFLOPS = %.1f\n",gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->mHeight,gp->kh,gp->c_stride_h,wu_time*1000.0, gf/wu_time/1e9);
    else if(gp->pad_h == 1)
      printf("XSMM-CONV-WU mb%dic%dih%doc%doh%dkh%dph%dn time = %g ms, GFLOPS = %.1f\n",gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->mHeight,gp->kh,gp->mpad_h,wu_time*1000.0, gf/wu_time/1e9);
  }
#endif

#ifndef NDEBUG
  /* check physical padding */
  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)delmiddle, gp->batch_size, gp->nOutput/16, gp->mHeight, gp->mWidth, 16, gp->mpad_h, gp->mpad_w );
  else if(gp->in_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)delmiddle, gp->batch_size, gp->nOutput/16, gp->mHeight, gp->mWidth, 16, gp->mpad_h, gp->mpad_w );
#endif
}

void FusedConvBNXSMM::dumpBuffer(TensorBuf* tBuf, void* wtemp)
{
  int buftype = tBuf->getBufferType();

  if(buftype == DATA)
  {
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_copyout_tensor(libxsmm_checkpoint_filter, wtemp, LIBXSMM_DNN_TENSOR_FORMAT_KCRS));
  }
  else if(buftype == HISTORY)
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_copyout_tensor(libxsmm_checkpoint_history_filter, wtemp, LIBXSMM_DNN_TENSOR_FORMAT_KCRS));
}
