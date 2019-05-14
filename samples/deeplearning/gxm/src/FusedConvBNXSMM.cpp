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
  fusedbn_desc_train.N = gp->batch_size/gp->num_numa_nodes;
  fusedbn_desc_train.C = gp->nOutput;
  fusedbn_desc_train.H = gp->mHeight;
  fusedbn_desc_train.W = gp->mWidth;
  fusedbn_desc_train.u = gp->bn_stride_h;
  fusedbn_desc_train.v = gp->bn_stride_w;
  fusedbn_desc_train.pad_h_in = gp->mpad_h;
  fusedbn_desc_train.pad_w_in = gp->mpad_w;
  fusedbn_desc_train.pad_h_out = gp->opad_h;
  fusedbn_desc_train.pad_w_out = gp->opad_w;
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
  if(gp->relu_fwd)
    fusedbn_desc_train.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_RELU_WITH_MASK;
  if(gp->eltwise)
    fusedbn_desc_train.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_ELTWISE;
  if(gp->relu_fwd && gp->eltwise)
    fusedbn_desc_train.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_ELTWISE_RELU_WITH_MASK;

  for(int i=0; i<gp->num_numa_nodes; i++)
  {
    libxsmm_handle_bn_train[i] = libxsmm_dnn_create_fusedbatchnorm( fusedbn_desc_train, &status );
    CHKERR_LIBXSMM_DNN( status );
  }

  if(gp->use_global_stats)
  {
    fusedbn_desc_test.N = gp->batch_size/gp->num_numa_nodes;
    fusedbn_desc_test.C = gp->nOutput;
    fusedbn_desc_test.H = gp->mHeight;
    fusedbn_desc_test.W = gp->mWidth;
    fusedbn_desc_test.u = gp->bn_stride_h;
    fusedbn_desc_test.v = gp->bn_stride_w;
    fusedbn_desc_test.pad_h_in = gp->mpad_h;
    fusedbn_desc_test.pad_w_in = gp->mpad_w;
    fusedbn_desc_test.pad_h_out = gp->opad_h;
    fusedbn_desc_test.pad_w_out = gp->opad_w;
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
    if(gp->relu_fwd)
      fusedbn_desc_test.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_RELU_WITH_MASK;

    if(gp->eltwise)
      fusedbn_desc_test.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE;
    if(gp->relu_fwd && gp->eltwise)
      fusedbn_desc_test.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE_RELU_WITH_MASK;

    for(int i=0; i<gp->num_numa_nodes; i++)
    {
      libxsmm_handle_bn_test[i] = libxsmm_dnn_create_fusedbatchnorm( fusedbn_desc_test, &status );
      CHKERR_LIBXSMM_DNN( status );
    }
  }

  for(int i=0; i<gp->num_numa_nodes; i++)
  {
    conv_desc[i].N = gp->batch_size/gp->num_numa_nodes;
    conv_desc[i].C = gp->nInput[0];
    conv_desc[i].H = gp->iHeight;
    conv_desc[i].W = gp->iWidth;
    conv_desc[i].K = gp->nOutput;
    conv_desc[i].R = gp->kh;
    conv_desc[i].S = gp->kw;
    conv_desc[i].u = gp->c_stride_h;
    conv_desc[i].v = gp->c_stride_w;
    conv_desc[i].threads = gp->num_threads/gp->num_numa_nodes;

    if(gp->physical_padding)
    {
      conv_desc[i].pad_h_in = gp->ipad_h;
      conv_desc[i].pad_w_in = gp->ipad_w;
    }
    else
    {
      conv_desc[i].pad_h_in = 0;
      conv_desc[i].pad_w_in = 0;
    }

    conv_desc[i].pad_w = gp->ipad_w;
    conv_desc[i].pad_h = gp->ipad_h;

    if(gp->physical_padding)
    {
      conv_desc[i].pad_h_out = gp->mpad_h;
      conv_desc[i].pad_w_out = gp->mpad_w;
    }
    else
    {
      conv_desc[i].pad_h_out = 0;
      conv_desc[i].pad_w_out = 0;
    }

    conv_desc[i].algo = LIBXSMM_DNN_CONV_ALGO_DIRECT;
    conv_desc[i].buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
    conv_desc[i].filter_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
    conv_desc[i].fuse_ops = LIBXSMM_DNN_CONV_FUSE_BATCHNORM;
    if(gp->out_data_type == DT_FLOAT)
      conv_desc[i].options = LIBXSMM_DNN_CONV_OPTION_OVERWRITE;
    else if(gp->out_data_type == DT_BF16)
      conv_desc[i].options = LIBXSMM_DNN_CONV_OPTION_F32_BF16_CVT_RNE_OVERWRITE;

    if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_FLOAT)
    {
      conv_desc[i].datatype_in = LIBXSMM_DNN_DATATYPE_BF16;
      conv_desc[i].datatype_out = LIBXSMM_DNN_DATATYPE_F32;
    }
    else if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_BF16)
    {
      conv_desc[i].datatype_in = LIBXSMM_DNN_DATATYPE_BF16;
      conv_desc[i].datatype_out = LIBXSMM_DNN_DATATYPE_BF16;
    }
    else if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
    {
      conv_desc[i].datatype_in = LIBXSMM_DNN_DATATYPE_F32;
      conv_desc[i].datatype_out = LIBXSMM_DNN_DATATYPE_F32;
    }

    if(!gp->use_global_stats)
    {
      if(gp->prev_bn_train_handle_ptr != NULL)
        conv_desc[i].pre_bn = ((libxsmm_dnn_fusedbatchnorm**)(gp->prev_bn_train_handle_ptr))[i];
      else
        conv_desc[i].pre_bn = NULL;
      conv_desc[i].post_bn = libxsmm_handle_bn_train[i];
    }
    else
    {
      if(gp->prev_bn_test_handle_ptr != NULL)
        conv_desc[i].pre_bn = ((libxsmm_dnn_fusedbatchnorm**)(gp->prev_bn_test_handle_ptr))[i];
      else
        conv_desc[i].pre_bn = NULL;
      conv_desc[i].post_bn = libxsmm_handle_bn_test[i];
    }

    libxsmm_handle_conv[i] = libxsmm_dnn_create_conv_layer( conv_desc[i], &status );
    CHKERR_LIBXSMM_DNN( status );
  }
  gp->my_bn_train_handle_ptr = (void**)libxsmm_handle_bn_train;
  gp->my_bn_test_handle_ptr = (void**)libxsmm_handle_bn_test;
}

void FusedConvBNXSMM::forwardPropagate(vector<TensorBuf *>& inp, TensorBuf *weightp, TensorBuf *hweightp, TensorBuf *midp, TensorBuf *gammap, TensorBuf *betap, TensorBuf *meanp, TensorBuf *varp, TensorBuf *outp, int tid)
{
  int nImg = gp->batch_size/gp->num_numa_nodes;
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
  int fhm = mfh + 2*mph;
  int fwm = mfw + 2*mpw;
  int ifhp = ifh + 2*iph;
  int ifwp = ifw + 2*ipw;
  int ofhp = ofh + 2*oph;
  int ofwp = ofw + 2*opw;

  assert(bot_compute_engine[0] != -1);
  assert(top_compute_engine[0] != -1);

  // Conv input. LPBuffer is non-NULL if data layer output is BF16
  void *inp_r[NUM_NUMA_NODES], *inp_l[NUM_NUMA_NODES], *hwt_ptr, *middle[NUM_NUMA_NODES], *output[NUM_NUMA_NODES];
  void *wt_ptr[NUM_NUMA_NODES];

  int imoff = nImg * nIFM * ifhp * ifwp;
  if(gp->in_data_type == DT_BF16)
  {
    if(inp[0]->getLPBuffer() != NULL)
      inp_r[0] = inp[0]->getLPBuffer();
    else
      inp_r[0] = inp[0]->getBuffer();
    imoff = imoff * sizeof(libxsmm_bfloat16);
  }
  else if(gp->in_data_type == DT_FLOAT)
  {
    inp_r[0] = inp[0]->getBuffer();
    imoff = imoff * sizeof(float);
  }

  for(int n=1; n<gp->num_numa_nodes; n++)
    inp_r[n] = inp_r[n-1] + imoff;

  if(gp->eltwise)
  {
    imoff = fusedbn_desc_train.N * gp->nInput[1] * ifhp * ifwp;
    if(gp->out_data_type == DT_FLOAT)
      imoff = imoff * sizeof(float);
    else if(gp->out_data_type == DT_BF16)
      imoff = imoff * sizeof(libxsmm_bfloat16);

    if(inp[1]->getLPBuffer() != NULL)
      inp_l[0] = inp[1]->getLPBuffer();
    else
      inp_l[0] = inp[1]->getBuffer();

    for(int n=1; n<gp->num_numa_nodes; n++)
      inp_l[n] = inp_l[n-1] + imoff;
  }

  // Conv Weight
  void **lptrptr = weightp->getLPBufferPtr();
  void **ptrptr = weightp->getBufferPtr();
  int offset = weightp->getOffset();

  if(lptrptr != NULL)
    for(int n=0; n<gp->num_numa_nodes; n++)
      wt_ptr[n] = lptrptr[n] + offset*sizeof(libxsmm_bfloat16);
  else
    for(int n=0; n<gp->num_numa_nodes; n++)
      wt_ptr[n] = ptrptr[n] + offset*sizeof(float);

  void *wt_prv_ptr = NULL;

  // Conv weight history
  if(hweightp != NULL)
    hwt_ptr = hweightp->getBuffer();
  else
    hwt_ptr=NULL;

  // Conv output
  middle[0] = midp->getBuffer();
  imoff = nImg * nOFM * fhm * fwm;
  if(gp->out_data_type == DT_FLOAT)
    imoff = imoff * sizeof(float);
  else if(gp->out_data_type == DT_BF16)
    imoff = imoff * sizeof(libxsmm_bfloat16);

  for(int n=1; n<gp->num_numa_nodes; n++)
    middle[n] = middle[n-1] + imoff;

  output[0] = outp->getBuffer();
  imoff = fusedbn_desc_train.N * fusedbn_desc_train.C * ofhp * ofwp;
  if(gp->out_data_type == DT_FLOAT)
    imoff = imoff * sizeof(float);
  else if(gp->out_data_type == DT_BF16)
    imoff = imoff * sizeof(libxsmm_bfloat16);

  for(int n=1; n<gp->num_numa_nodes; n++)
    output[n] = output[n-1] + imoff;

  void *gamma[NUM_NUMA_NODES];
  void *beta[NUM_NUMA_NODES];
  float *gexpect[NUM_NUMA_NODES];
  float *gvar[NUM_NUMA_NODES];
  float *gexp_test = (float*)meanp->getPrivBuffer();
  float *gvar_test = (float*)varp->getPrivBuffer();

  void **gptrptr = gammap->getBufferPtr();
  offset = gammap->getOffset() * sizeof(float);
  for(int n=0; n<gp->num_numa_nodes; n++)
    gamma[n] = gptrptr[n] + offset;

  void **bptrptr = betap->getBufferPtr();
  offset = betap->getOffset() * sizeof(float);
  for(int n=0; n<gp->num_numa_nodes; n++)
    beta[n] = bptrptr[n] + offset;

  void **mptrptr = meanp->getBufferPtr();
  offset = meanp->getOffset();
  for(int n=0; n<gp->num_numa_nodes; n++)
    gexpect[n] = (float*)mptrptr[n] + offset;

  void **vptrptr = varp->getBufferPtr();
  offset = varp->getOffset();
  for(int n=0; n<gp->num_numa_nodes; n++)
    gvar[n] = (float*)vptrptr[n] + offset;

  void **sptrptr = scratchp->getBufferPtr();

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(bexpect[n] == NULL)
    {
      bexpect[n] = (void*)_mm_malloc(nOFM*sizeof(float), 64);

#ifndef NDEBUG
      printf("%s allocated %lu bytes for mean\n",nname.c_str(), nOFM*sizeof(float));
#endif
    }

    if(bstddev[n] == NULL)
    {
      bstddev[n] = (void*)_mm_malloc(nOFM*sizeof(float), 64);

#ifndef NDEBUG
      printf("%s allocated %lu bytes for stdev\n",nname.c_str(), nOFM*sizeof(float));
#endif
    }

    if(bvariance[n] == NULL)
    {
      bvariance[n] = (void*)_mm_malloc(nOFM*sizeof(float), 64);

#ifndef NDEBUG
      printf("%s allocated %lu bytes for variance\n",nname.c_str(), nOFM*sizeof(float));
#endif
    }

    if(relu_mask[n] == NULL)
      relu_mask[n] = (void*)libxsmm_aligned_malloc(nImg*nOFM*ofhp*ofwp*sizeof(unsigned char), 2097152);
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

  // Create Conv Handle
  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_input[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle_conv[n], LIBXSMM_DNN_REGULAR_INPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input[n] = libxsmm_dnn_link_tensor( libxsmm_layout, inp_r[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle_conv[n], libxsmm_input[n], LIBXSMM_DNN_REGULAR_INPUT ) );
    }
  }

  int welem = gp->nInput[0] * gp->nOutput * gp->kw * gp->kh;

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_filter[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle_conv[n], LIBXSMM_DNN_REGULAR_FILTER, &status );
      CHKERR_LIBXSMM_DNN( status );

      if(gp->in_data_type == DT_FLOAT)
      {
        int wsize = welem*sizeof(float);
        wt_prv_ptr = (void*)libxsmm_aligned_malloc(wsize, 2097152);

        // Transform weight layout
        libxsmm_filter[n] = libxsmm_dnn_link_tensor( libxsmm_layout, wt_prv_ptr, &status );
        CHKERR_LIBXSMM_DNN( status );

        CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor(libxsmm_filter[n], wt_ptr[n], LIBXSMM_DNN_TENSOR_FORMAT_KCRS) );
        memcpy(wt_ptr[n], wt_prv_ptr, wsize);

        if(n==0)
        {
          libxsmm_checkpoint_filter = libxsmm_dnn_link_tensor(libxsmm_layout, wt_ptr[n], &status);
          CHKERR_LIBXSMM_DNN( status );
        }

        libxsmm_filter[n] = libxsmm_dnn_link_tensor( libxsmm_layout, wt_ptr[n], &status );
        CHKERR_LIBXSMM_DNN( status );

        // Transform weight history layout
        if(n == 0)
        {
          if(hwt_ptr != NULL)
          {
            libxsmm_temp = libxsmm_dnn_link_tensor( libxsmm_layout, wt_prv_ptr, &status );
            CHKERR_LIBXSMM_DNN( status );

            CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_temp, (void*)hwt_ptr, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );
            memcpy(hwt_ptr, wt_prv_ptr, welem*sizeof(float));

            libxsmm_checkpoint_history_filter = libxsmm_dnn_link_tensor(libxsmm_layout, hwt_ptr, &status);
            CHKERR_LIBXSMM_DNN( status );
          }
        }
        libxsmm_free(wt_prv_ptr);
        wt_prv_ptr = NULL;
        weightp->setPrivBuffer(NULL);
      }
      else if(gp->in_data_type == DT_BF16)
      {
        int wsize = welem*sizeof(libxsmm_bfloat16);
        wt_prv_ptr = (void*)libxsmm_aligned_malloc(wsize, 2097152);

        // Transform BF16 weight layout
        libxsmm_filter[n] = libxsmm_dnn_link_tensor( libxsmm_layout, wt_prv_ptr, &status );
        CHKERR_LIBXSMM_DNN( status );

        CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor(libxsmm_filter[n], wt_ptr[n], LIBXSMM_DNN_TENSOR_FORMAT_KCRS) );
        memcpy(wt_ptr[n], wt_prv_ptr, wsize);
        libxsmm_filter[n] = libxsmm_dnn_link_tensor( libxsmm_layout, wt_ptr[n], &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_free(wt_prv_ptr);

        // Transform FP32 weight layout
        if(n == 0)
        {
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
      }

      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle_conv[n], libxsmm_filter[n], LIBXSMM_DNN_REGULAR_FILTER ) );
    }
  }

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_middle[n] == NULL)
    {
      // Conv Output
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle_conv[n], LIBXSMM_DNN_REGULAR_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN(      status );
      libxsmm_middle[n] = libxsmm_dnn_link_tensor( libxsmm_layout, middle[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor(libxsmm_handle_conv[n], libxsmm_middle[n], LIBXSMM_DNN_REGULAR_OUTPUT));
    }
  }

  // Create BN Train handle
  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_input_bntrain[n]==NULL)
    {
      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_bn_train[n], LIBXSMM_DNN_REGULAR_INPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input_bntrain[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, middle[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_train[n], libxsmm_input_bntrain[n], LIBXSMM_DNN_REGULAR_INPUT ) );
    }
  }

  if(gp->eltwise)
  {
    for(int n=0; n<gp->num_numa_nodes; n++)
    {
      if(libxsmm_input_add_bntrain[n] == NULL)
      {
        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_train[n], LIBXSMM_DNN_REGULAR_INPUT_ADD, &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_input_add_bntrain[n] = libxsmm_dnn_link_tensor( libxsmm_layout, inp_l[n], &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_train[n], libxsmm_input_add_bntrain[n], LIBXSMM_DNN_REGULAR_INPUT_ADD ) )
      }
    }
  }

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_expectval_train[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_bn_train[n], LIBXSMM_DNN_CHANNEL_EXPECTVAL, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_expectval_train[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, bexpect[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_bn_train[n], libxsmm_expectval_train[n], LIBXSMM_DNN_CHANNEL_EXPECTVAL ) );
    }

    if(libxsmm_stddev_train[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_train[n], LIBXSMM_DNN_CHANNEL_RCPSTDDEV, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_stddev_train[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, bstddev[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_bn_train[n], libxsmm_stddev_train[n], LIBXSMM_DNN_CHANNEL_RCPSTDDEV ) );
    }

    if(libxsmm_variance_train[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_bn_train[n], LIBXSMM_DNN_CHANNEL_VARIANCE, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_variance_train[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, bvariance[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_bn_train[n], libxsmm_variance_train[n], LIBXSMM_DNN_CHANNEL_VARIANCE ) );
    }

    if(libxsmm_gamma_train[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_train[n], LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_gamma_train[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, gamma[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_bn_train[n], libxsmm_gamma_train[n], LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA ) );
    }

    if(libxsmm_beta_train[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_bn_train[n], LIBXSMM_DNN_REGULAR_CHANNEL_BETA, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_beta_train[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, beta[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_bn_train[n], libxsmm_beta_train[n], LIBXSMM_DNN_REGULAR_CHANNEL_BETA ) );
    }

    if(libxsmm_output_bntrain[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_bn_train[n], LIBXSMM_DNN_REGULAR_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_output_bntrain[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, output[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_bn_train[n], libxsmm_output_bntrain[n], LIBXSMM_DNN_REGULAR_OUTPUT ) );
    }

    if(libxsmm_relumask_bntrain[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_bn_train[n], LIBXSMM_DNN_RELU_MASK, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_relumask_bntrain[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, relu_mask[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_bn_train[n], libxsmm_relumask_bntrain[n], LIBXSMM_DNN_RELU_MASK ) );
    }
  }

  if(use_global_stats)
  {
    // Create BN test handle
    for(int n=0; n<gp->num_numa_nodes; n++)
    {
      if(libxsmm_input_bntest[n]==NULL)
      {
        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_bn_test[n], LIBXSMM_DNN_REGULAR_INPUT, &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_input_bntest[n] = libxsmm_dnn_link_tensor( libxsmm_layout, middle[n], &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_test[n], libxsmm_input_bntest[n], LIBXSMM_DNN_REGULAR_INPUT ) );
      }
    }

    if(gp->eltwise)
    {
      for(int n=0; n<gp->num_numa_nodes; n++)
      {
        if(libxsmm_input_add_bntest[n] == NULL)
        {
          libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_test[n], LIBXSMM_DNN_REGULAR_INPUT_ADD, &status );
          CHKERR_LIBXSMM_DNN( status );
          libxsmm_input_add_bntest[n] = libxsmm_dnn_link_tensor( libxsmm_layout, inp_l[n], &status );
          CHKERR_LIBXSMM_DNN( status );
          libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
          CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_test[n], libxsmm_input_add_bntest[n], LIBXSMM_DNN_REGULAR_INPUT_ADD ) )
        }
      }
    }

    for(int n=0; n<gp->num_numa_nodes; n++)
    {
      if(libxsmm_expectval_test[n] == NULL)
      {
        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_bn_test[n], LIBXSMM_DNN_CHANNEL_EXPECTVAL, &status);
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_expectval_test[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, bexpect[n], &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_bn_test[n], libxsmm_expectval_test[n], LIBXSMM_DNN_CHANNEL_EXPECTVAL ) );
      }

      if(libxsmm_stddev_test[n] == NULL)
      {
        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_test[n], LIBXSMM_DNN_CHANNEL_RCPSTDDEV, &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_stddev_test[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, bstddev[n], &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_bn_test[n], libxsmm_stddev_test[n], LIBXSMM_DNN_CHANNEL_RCPSTDDEV ) );
      }

      if(libxsmm_variance_test[n] == NULL)
      {
        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_bn_test[n], LIBXSMM_DNN_CHANNEL_VARIANCE, &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_variance_test[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, bvariance[n], &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_bn_test[n], libxsmm_variance_test[n], LIBXSMM_DNN_CHANNEL_VARIANCE ) );
      }

      if(libxsmm_gamma_test[n] == NULL)
      {
        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_test[n], LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA, &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_gamma_test[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, gamma[n], &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_bn_test[n], libxsmm_gamma_test[n], LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA ) );
      }

      if(libxsmm_beta_test[n] == NULL)
      {
        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_bn_test[n], LIBXSMM_DNN_REGULAR_CHANNEL_BETA, &status);
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_beta_test[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, beta[n], &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_bn_test[n], libxsmm_beta_test[n], LIBXSMM_DNN_REGULAR_CHANNEL_BETA ) );
      }

      if(libxsmm_output_bntest[n] == NULL)
      {
        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_bn_test[n], LIBXSMM_DNN_REGULAR_OUTPUT, &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_output_bntest[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, output[n], &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_bn_test[n], libxsmm_output_bntest[n], LIBXSMM_DNN_REGULAR_OUTPUT ) );
      }

      if(libxsmm_relumask_bntest[n] == NULL)
      {
        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_bn_test[n], LIBXSMM_DNN_RELU_MASK, &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_relumask_bntest[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, relu_mask[n], &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_handle_bn_test[n], libxsmm_relumask_bntest[n], LIBXSMM_DNN_RELU_MASK ) );
      }
    }
  }

  /* let's allocate (if required) and bind scratch */
  int max_size =  0;
  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(sptrptr[n] == NULL)
    {
      int csize = libxsmm_dnn_get_scratch_size( libxsmm_handle_conv[n], LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
      CHKERR_LIBXSMM_DNN( status );
      int bnsize;
      if(!use_global_stats)
        bnsize = libxsmm_dnn_fusedbatchnorm_get_scratch_size( libxsmm_handle_bn_train[n], &status );
      else
        bnsize = libxsmm_dnn_fusedbatchnorm_get_scratch_size( libxsmm_handle_bn_test[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      int mysize = csize + bnsize;
      sptrptr[n] = (void*)libxsmm_aligned_malloc(mysize , 2097152);
      max_size = mysize;

#ifdef USE_MLSL
      if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
        printf("%s allocated %d bytes for scratch @ %p\n",nname.c_str(), mysize, sptrptr[n]);
    }
    else
    {
      int ssize = scratchp->getBufferSize();
      int csize = libxsmm_dnn_get_scratch_size( libxsmm_handle_conv[n], LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
      CHKERR_LIBXSMM_DNN( status );
      int bnsize;
      if(!use_global_stats)
        bnsize = libxsmm_dnn_fusedbatchnorm_get_scratch_size( libxsmm_handle_bn_train[n], &status );
      else
        bnsize = libxsmm_dnn_fusedbatchnorm_get_scratch_size( libxsmm_handle_bn_test[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      int mysize = csize + bnsize;

      if(ssize < mysize)
      {
        libxsmm_free(sptrptr[n]);
        sptrptr[n] = (void*)libxsmm_aligned_malloc(mysize, 2097152);
        max_size = mysize;

#ifdef USE_MLSL
        if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
          printf("%s allocated %d bytes for scratch @ %p, prev size was %d bytes\n",nname.c_str(), mysize, sptrptr[n], ssize);
      }
      else
        max_size = ssize;
    }
  }
  scratchp->setBufferSize(max_size);

  if(prev_scratch_size == 0)
    prev_scratch_size = scratchp->getBufferSize();

  if(!updated_scratch_fwd || prev_scratch_size != scratchp->getBufferSize())
  {
    for(int n=0; n<gp->num_numa_nodes; n++)
    {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle_conv[n], LIBXSMM_DNN_COMPUTE_KIND_ALL, sptrptr[n] ) );
      if(!use_global_stats)
      {
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_scratch( libxsmm_handle_bn_train[n], sptrptr[n] ) );
      }
      else
      {
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_scratch( libxsmm_handle_bn_test[n], sptrptr[n] ) );
      }
    }
    updated_scratch_fwd = true;
    prev_scratch_size = scratchp->getBufferSize();
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
      check_physical_pad( nname.c_str(), (float*)inp_r[0], nImg, nBIfm, ifh, ifw, VLEN, iph, ipw );
    else
      check_physical_pad( nname.c_str(), (float*)inp_r[0], nImg, 1, ifh, ifw, 3, iph, ipw );
    check_physical_pad( nname.c_str(),    (float*)middle[0], nImg, nBOfm, mfh,  mfw,  VLEN, mph, mpw );
    check_physical_pad( nname.c_str(),     (float*)output[0], nImg, nBOfm, ofh, ofw, VLEN, oph,  opw );
  }
  else if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_BF16)
  {
    if(nIFM > 3)
      check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)inp_r[0], nImg, nBIfm, ifh, ifw, VLEN, iph, ipw );
    else
      check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)inp_r[0], nImg, 1, ifh, ifw, 3, iph, ipw );
    check_physical_pad( nname.c_str(),    (libxsmm_bfloat16*)middle[0], nImg, nBOfm, mfh,  mfw,  VLEN, mph, mpw );
    check_physical_pad( nname.c_str(),     (libxsmm_bfloat16*)output[0], nImg, nBOfm, ofh, ofw, VLEN, oph,  opw );
  }
#endif

  if(!use_global_stats)
  {
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

      int ntps = gp->num_threads/gp->num_numa_nodes;
      int n = tid/ntps;
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_execute_st( libxsmm_handle_conv[n], LIBXSMM_DNN_COMPUTE_KIND_FWD, n*ntps, tid) );
    }

#ifdef USE_XSMM_TIMING
    gettimeofday(&tvec, NULL);
    double fp_time = (tvec.tv_sec + tvec.tv_usec*1e-6) - (tvsc.tv_sec + tvsc.tv_usec*1e-6);

#ifdef USE_MLSL
    if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
    {
      double gf = (double)gp->batch_size * (double)gp->nInput[0] * (double)gp->nOutput * (double)gp->mHeight * (double)gp->mWidth * (double)gp->kh * (double)gp->kw * 2;
      if(gp->c_stride_h == 1 && gp->mpad_h == 0)
        printf("%s XSMM-CONV-FP mb%dic%dih%doc%doh%dkh%dn time = %g ms, GFLOPS = %.1f\n",gp->node_name.c_str(),gp->batch_size,gp->nInput[0],gp->iHeight,gp->nOutput,gp->mHeight,gp->kh,fp_time*1000.0, gf/fp_time/1e9);
      else if(gp->c_stride_h == 2)
        printf("%s XSMM-CONV-FP mb%dic%dih%doc%doh%dkh%dsh%dn time = %g ms, GFLOPS = %.1f\n",gp->node_name.c_str(),gp->batch_size,gp->nInput[0],gp->iHeight,gp->nOutput,gp->mHeight,gp->kh,gp->c_stride_h,fp_time*1000.0, gf/fp_time/1e9);
      else if(gp->mpad_h == 1)
        printf("%s XSMM-CONV-FP mb%dic%dih%doc%doh%dkh%dph%dn time = %g ms, GFLOPS = %.1f\n",gp->node_name.c_str(),gp->batch_size,gp->nInput[0],gp->iHeight,gp->nOutput,gp->mHeight,gp->kh,gp->mpad_h,fp_time*1000.0, gf/fp_time/1e9);
    }
#endif

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
        check_physical_pad( nname.c_str(), (float*)inp_r[0], nImg, nBIfm, ifh, ifw, VLEN, iph, ipw );
      else
        check_physical_pad( nname.c_str(), (float*)inp_r[0], nImg, 1, ifh, ifw, 3, iph, ipw );
      check_physical_pad( nname.c_str(),    (float*)middle[0], nImg, nBOfm, mfh,  mfw,  VLEN, mph, mpw );
      check_physical_pad( nname.c_str(),     (float*)output[0], nImg, nBOfm, ofh, ofw, VLEN, oph,  opw );
    }
    else if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_BF16)
    {
      if(nIFM > 3)
        check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)inp_r[0], nImg, nBIfm, ifh, ifw, VLEN, iph, ipw );
      else
        check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)inp_r[0], nImg, 1, ifh, ifw, 3, iph, ipw );
      check_physical_pad( nname.c_str(),    (libxsmm_bfloat16*)middle[0], nImg, nBOfm, mfh,  mfw,  VLEN, mph, mpw );
      check_physical_pad( nname.c_str(),     (libxsmm_bfloat16*)output[0], nImg, nBOfm, ofh, ofw, VLEN, oph,  opw );
    }
#endif

    if(gp->exec_mode == "TRAIN")
    {
      for(int n=0; n<gp->num_numa_nodes; n++)
      {
        float *gexp = gexpect[n];
        float *gv = gvar[n];

        float (* __restrict bmean)[VLEN] = (float (*)[VLEN])bexpect[n];
        float (* __restrict bvar)[VLEN] = (float (*)[VLEN])bvariance[n];
        float nhw_ratio = float(fusedbn_desc_train.N*mfh*mfw)/float(fusedbn_desc_train.N*mfh*mfw - 1);

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
            for (int b = 0; b < nBOfm; ++b) {
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
        for (int b = 0; b < nBOfm; ++b) {
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
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      int ntps = gp->num_threads/gp->num_numa_nodes;
      int s = tid/ntps;
      int ltid = tid - s*ntps;

      int jobs = (nOFM % ntps == 0) ? nOFM/ntps : nOFM/ntps + 1;
      int tb = (ltid*jobs < nOFM) ? ltid*jobs : nOFM;
      int te = ((ltid+1)*jobs < nOFM) ? (ltid+1)*jobs : nOFM;

      for(int i=tb; i < te; i++)
      {
        ((float*)bexpect[s])[i] = ((float*)gexpect[s])[i]/scaling_factor_;
        float tmp = ((float*)gvar[s])[i]/scaling_factor_;
        ((float*)bstddev[s])[i] = 1./sqrt(tmp + gp->eps);
      }
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

      int ntps = gp->num_threads/gp->num_numa_nodes;
      int n = tid/ntps;
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_execute_st( libxsmm_handle_conv[n], LIBXSMM_DNN_COMPUTE_KIND_FWD, n*ntps, tid) );
    }
  }
}

void FusedConvBNXSMM::backPropagate(TensorBuf *delmidp, TensorBuf* weightp, TensorBuf* delinp, int tid)
{
  void *delmiddle[NUM_NUMA_NODES];
  void *delinp_r[NUM_NUMA_NODES];

  int nImg  = gp->batch_size/gp->num_numa_nodes;
  int nIFM = gp->nInput[0];
  int nOFM = gp->nOutput;
  int nBIfm = nIFM/VLEN;
  int nBOfm = nOFM/VLEN;
  int mfh = gp->mHeight;
  int mfw = gp->mWidth;
  int ifh = gp->iHeight;
  int ifw = gp->iWidth;

  int iph = gp->ipad_h;
  int ipw = gp->ipad_w;
  int mph = gp->mpad_h;
  int mpw = gp->mpad_w;

  int csh = gp->c_stride_h;
  int csw = gp->c_stride_w;

  int fhm = mfh + 2*mph;
  int fwm = mfw + 2*mpw;
  int fhi = ifh + 2*iph;
  int fwi = ifw + 2*ipw;

  delmiddle[0] = delmidp->getBuffer();
  delinp_r[0] = delinp->getBuffer();

  int imoff = nImg * nOFM * fhm * fwm;
  if(gp->out_data_type == DT_FLOAT)
    imoff = imoff * sizeof(float);
  else if(gp->out_data_type == DT_BF16)
    imoff = imoff * sizeof(libxsmm_bfloat16);

  for(int n=1; n<gp->num_numa_nodes; n++)
    delmiddle[n] = delmiddle[n-1] + imoff;

  imoff = nImg * nIFM * fhi * fwi;
  if(gp->in_data_type == DT_FLOAT)
    imoff = imoff * sizeof(float);
  else if(gp->in_data_type == DT_BF16)
    imoff = imoff * sizeof(libxsmm_bfloat16);

  for(int n=1; n<gp->num_numa_nodes; n++)
    delinp_r[n] = delinp_r[n-1] + imoff;

  void **sptrptr = scratchp->getBufferPtr();

  if(!updated_scratch_bwd)
  {
    for(int n=0; n<gp->num_numa_nodes; n++)
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle_conv[n], LIBXSMM_DNN_COMPUTE_KIND_ALL, sptrptr[n] ) );
    updated_scratch_bwd = true;
  }

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_delinput[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle_conv[n], LIBXSMM_DNN_GRADIENT_INPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_delinput[n] = libxsmm_dnn_link_tensor(libxsmm_layout, delinp_r[n], &status );
      CHKERR_LIBXSMM_DNN(status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor(libxsmm_handle_conv[n], libxsmm_delinput[n], LIBXSMM_DNN_GRADIENT_INPUT));
    }

    if(libxsmm_delmiddle_conv[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle_conv[n], LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN(status );
      libxsmm_delmiddle_conv[n] = libxsmm_dnn_link_tensor( libxsmm_layout, delmiddle[n], &status );
      CHKERR_LIBXSMM_DNN(status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle_conv[n], libxsmm_delmiddle_conv[n], LIBXSMM_DNN_GRADIENT_OUTPUT ) );
    }
  }

#ifndef NDEBUG
  /* check physical padding */
  if ( (iph > 0 || ipw > 0) && (mph > 0 || mpw > 0) ) {
  } else if ( (iph == 0 || ipw == 0) && (mph == 0 || mpw == 0) ) {
  } else {
    printf("node %s: conv xsmm backward is partially padded which cannot be :-(\n", nname.c_str());
  }
  if(gp->out_data_type == DT_FLOAT)
  {
    check_physical_pad( nname.c_str(), (float*)delinp_r[0], nImg, nBIfm, ifh, ifw, 16, iph, ipw );
    check_physical_pad( nname.c_str(), (float*)delmiddle[0], nImg, nBOfm, mfh, mfw, 16, mph, mpw );
  }
  else if(gp->out_data_type == DT_BF16)
  {
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)delinp_r[0], nImg, nBIfm, ifh, ifw, 16, iph, ipw );
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)delmiddle[0], nImg, nBOfm, mfh, mfw, 16, mph, mpw );
  }
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

      int ntps = gp->num_threads/gp->num_numa_nodes;
      int n = tid/ntps;
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle_conv[n], LIBXSMM_DNN_COMPUTE_KIND_BWD, n*ntps, tid ) );
    }

#ifdef USE_XSMM_TIMING
  gettimeofday(&tvec, NULL);
  double bp_time = (tvec.tv_sec + tvec.tv_usec*1e-6) - (tvsc.tv_sec + tvsc.tv_usec*1e-6);

#ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
  {
    double gf = (double)gp->batch_size * (double)gp->nInput[0] * (double)gp->nOutput * (double)gp->mHeight * (double)gp->mWidth * (double)gp->kh * (double)gp->kw * 2;
    if(gp->c_stride_h == 1 && gp->mpad_h == 0)
      printf("%s XSMM-CONV-BP mb%dic%dih%doc%doh%dkh%dn time = %g ms, GFLOPS = %.1f\n",gp->node_name.c_str(),gp->batch_size, gp->nInput[0], gp->iHeight,gp->nOutput,gp->mHeight,gp->kh,bp_time*1000.0, gf/bp_time/1e9);
    else if(gp->c_stride_h == 2)
      printf("%s XSMM-CONV-BP mb%dic%dih%doc%doh%dkh%dsh%dn time = %g ms, GFLOPS = %.1f\n",gp->node_name.c_str(),gp->batch_size,gp->nInput[0],gp->iHeight,gp->nOutput,gp->mHeight,gp->kh,gp->c_stride_h,bp_time*1000.0, gf/bp_time/1e9);
    else if(gp->mpad_h == 1)
      printf("%s XSMM-CONV-BP mb%dic%dih%doc%doh%dkh%dph%dn time = %g ms, GFLOPS = %.1f\n",gp->node_name.c_str(),gp->batch_size,gp->nInput[0],gp->iHeight,gp->nOutput,gp->mHeight,gp->kh,gp->mpad_h,bp_time*1000.0, gf/bp_time/1e9);
  }
#endif

#ifndef NDEBUG
  /* check physical padding */
  if ( (gp->ipad_h > 0 || gp->ipad_w > 0) && (gp->mpad_h > 0 || gp->mpad_w > 0) ) {
  } else if ( (gp->ipad_h == 0 || gp->ipad_w == 0) && (gp->mpad_h == 0 || gp->mpad_w == 0) ) {
  } else {
    printf("node %s: conv xsmm backward is partially padded which cannot be :-(\n", nname.c_str());
  }
  if(gp->out_data_type == DT_FLOAT)
  {
    check_physical_pad( nname.c_str(), (float*)delinp_r[0], nImg, nBIfm, ifh, ifw, 16, iph, ipw );
    check_physical_pad( nname.c_str(), (float*)delmiddle[0], nImg, nBOfm, mfh, mfw, 16, mph, mpw );
  }
  else if(gp->out_data_type == DT_BF16)
  {
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)delinp_r[0], nImg, nBIfm, ifh, ifw, 16, iph, ipw );
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)delmiddle[0], nImg, nBOfm, mfh, mfw, 16, mph, mpw );
  }
#endif
}

void FusedConvBNXSMM::weightUpdate(TensorBuf *inp, TensorBuf *deloutp, TensorBuf *delmidp, TensorBuf *delinpl, TensorBuf* delweightp, TensorBuf *delgammap, TensorBuf* delbetap, int tid)
{
  int nImg = gp->batch_size/gp->num_numa_nodes;
  int nOFM = gp->nOutput;
  int ofm = gp->nOutput;
  int ifm = gp->nInput[0];
  int kh = gp->kh;
  int kw = gp->kw;
  int nBOfm = nOFM/VLEN;
  int ofh = gp->oHeight;
  int ofw = gp->oWidth;
  int oph = gp->opad_h;
  int opw = gp->opad_w;
  int ofhp = ofh + 2*oph;
  int ofwp = ofw + 2*opw;
  int mfh = gp->mHeight;
  int mfw = gp->mWidth;
  int mph = gp->mpad_h;
  int mpw = gp->mpad_w;
  int fhm = mfh + 2*mph;
  int fwm = mfw + 2*mpw;
  int ifh = gp->iHeight;
  int ifw = gp->iWidth;
  int iph = gp->ipad_h;
  int ipw = gp->ipad_w;
  int ifhp = ifh + 2*iph;
  int ifwp = ifw + 2*ipw;

  void *deloutput[NUM_NUMA_NODES];
  void *delgamma[NUM_NUMA_NODES];
  void *delbeta[NUM_NUMA_NODES];
  void *dwt_ptr[NUM_NUMA_NODES];
  void *delmiddle[NUM_NUMA_NODES];
  void *delinp_l[NUM_NUMA_NODES];

  deloutput[0] = deloutp->getBuffer();

  int imoff = fusedbn_desc_train.N * fusedbn_desc_train.C * ofhp * ofwp;
  if(gp->out_data_type == DT_FLOAT)
    imoff = imoff * sizeof(float);
  else if(gp->out_data_type == DT_BF16)
    imoff = imoff * sizeof(libxsmm_bfloat16);

  for(int n=1; n<gp->num_numa_nodes; n++)
    deloutput[n] = deloutput[n-1] + imoff;

  delinp_l[0] = gp->eltwise ? delinpl->getBuffer() : NULL;
  if(gp->eltwise)
  {
    imoff = fusedbn_desc_train.N * gp->nInput[1] * ifhp * ifwp;
    if(gp->in_data_type == DT_FLOAT)
      imoff = imoff * sizeof(float);
    else if(gp->in_data_type == DT_BF16)
      imoff = imoff * sizeof(libxsmm_bfloat16);

    for(int n=1; n<gp->num_numa_nodes; n++)
      delinp_l[n] = delinp_l[n-1] + imoff;
  }

  void **ptrptr = delweightp->getBufferPtr();
  int offset = delweightp->getOffset();

  if(gp->in_data_type == DT_FLOAT)
    offset = offset*sizeof(float);
  else if(gp->in_data_type == DT_BF16)
    offset = offset*sizeof(libxsmm_bfloat16);

  for(int n=0; n<gp->num_numa_nodes; n++)
    dwt_ptr[n] = ptrptr[n] + offset;

  void **gptrptr = delgammap->getBufferPtr();
  void **bptrptr = delbetap->getBufferPtr();
  int goffset = delgammap->getOffset() * sizeof(float);
  int boffset = delbetap->getOffset() * sizeof(float);

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    delgamma[n] = gptrptr[n] + goffset;
    delbeta[n] = bptrptr[n] + boffset;
  }

  delmiddle[0] = delmidp->getBuffer();
  imoff = nImg * nOFM * fhm * fwm;
  if(gp->out_data_type == DT_FLOAT)
    imoff = imoff * sizeof(float);
  else if(gp->out_data_type == DT_BF16)
    imoff = imoff * sizeof(libxsmm_bfloat16);
  for(int n=1; n<gp->num_numa_nodes; n++)
    delmiddle[n] = delmiddle[n-1] + imoff;

  void **sptrptr = scratchp->getBufferPtr();

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(gp->in_data_type == DT_FLOAT)
    {
      float (* __restrict del_middle)[nBOfm][fhm][fwm][VLEN] = (float (*)[*][*][*][VLEN])delmiddle[n];

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
      libxsmm_bfloat16 (* __restrict del_middle)[nBOfm][fhm][fwm][VLEN] = (libxsmm_bfloat16 (*)[*][*][*][VLEN])delmiddle[n];

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
  }

  if(!updated_scratch_upd)
  {
    for(int n=0; n<gp->num_numa_nodes; n++)
    {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_scratch( libxsmm_handle_bn_train[n], sptrptr[n] ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle_conv[n], LIBXSMM_DNN_COMPUTE_KIND_ALL, sptrptr[n] ) );
    }
    updated_scratch_upd = true;
  }

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_delfilter[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle_conv[n], LIBXSMM_DNN_GRADIENT_FILTER, &status );
      CHKERR_LIBXSMM_DNN(status );
      libxsmm_delfilter[n] = libxsmm_dnn_link_tensor( libxsmm_layout, dwt_ptr[n], &status );
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle_conv[n], libxsmm_delfilter[n], LIBXSMM_DNN_GRADIENT_FILTER ) );
    }
  }

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_delmiddle_conv[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle_conv[n], LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN(      status );
      libxsmm_delmiddle_conv[n] = libxsmm_dnn_link_tensor(libxsmm_layout, delmiddle[n], &status );
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle_conv[n], libxsmm_delmiddle_conv[n], LIBXSMM_DNN_GRADIENT_OUTPUT ) );
    }

    if(gp->eltwise)
    {
      if(libxsmm_delinput_add[n] == NULL)
      {
        libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_bn_train[n], LIBXSMM_DNN_GRADIENT_INPUT_ADD, &status);
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_delinput_add[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, delinp_l[n], &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_train[n], libxsmm_delinput_add[n], LIBXSMM_DNN_GRADIENT_INPUT_ADD ) );
      }
    }
  }

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_deloutput[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_train[n], LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_deloutput[n] = libxsmm_dnn_link_tensor( libxsmm_layout, deloutput[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_train[n], libxsmm_deloutput[n], LIBXSMM_DNN_GRADIENT_OUTPUT ) );
    }

    if(libxsmm_delmiddle_bn[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout( libxsmm_handle_bn_train[n], LIBXSMM_DNN_GRADIENT_INPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_delmiddle_bn[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, delmiddle[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_train[n], libxsmm_delmiddle_bn[n], LIBXSMM_DNN_GRADIENT_INPUT ) );
    }

    if(libxsmm_delgamma[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_bn_train[n], LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_delgamma[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, delgamma[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_train[n], libxsmm_delgamma[n], LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA ) );
    }

    if(libxsmm_delbeta[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(libxsmm_handle_bn_train[n], LIBXSMM_DNN_GRADIENT_CHANNEL_BETA, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_delbeta[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, delbeta[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbatchnorm_bind_tensor( libxsmm_handle_bn_train[n], libxsmm_delbeta[n], LIBXSMM_DNN_GRADIENT_CHANNEL_BETA ) );
    }
  }

#ifndef NDEBUG
  /* check physical padding */
  if ( (gp->ipad_h > 0 || gp->ipad_w > 0) && (gp->mpad_h > 0 || gp->mpad_w > 0) ) {
  } else if ( (gp->ipad_h == 0 || gp->ipad_w == 0) && (gp->mpad_h == 0 || gp->mpad_w == 0) ) {
  } else {
    printf("node %s: conv xsmm backward is partially padded which cannot be :-(\n", nname.c_str());
  }
  if(gp->in_data_type == DT_FLOAT)
  {
    check_physical_pad( nname.c_str(), (float*)delmiddle[0], nImg, nBOfm, mfh, mfw, 16, mph, mpw);
    check_physical_pad( nname.c_str(), (float*)deloutput[0], nImg, nBOfm, ofh, ofw, 16, oph, opw);
  }
  else if(gp->in_data_type == DT_BF16)
  {
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)delmiddle[0], nImg, nBOfm, mfh, mfw, 16, mph, mpw);
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)deloutput[0], nImg, nBOfm, ofh, ofw, 16, oph, opw);
  }
#endif

#ifdef USE_XSMM_TIMING__
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

    int ntps = gp->num_threads/gp->num_numa_nodes;
    int n = tid/ntps;
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle_conv[n], LIBXSMM_DNN_COMPUTE_KIND_UPD, n*ntps, tid ) );

#ifdef USE_MLSL
#pragma omp barrier

    if(gp->in_data_type == DT_FLOAT)
    {
#include "reduce_weight_grads.c"
    }
    else if(gp->in_data_type == DT_BF16)
    {
#include "reduce_weight_grads_bf16.c"
    }

#pragma omp barrier
    if(tid == 0)
    {
      float *dgp = (float*)delgamma[0];
      float *dbp = (float*)delbeta[0];
      for(int nn=1; nn<gp->num_numa_nodes; nn++)
      {
        float *rdgp = (float*)delgamma[nn];
        float *rdbp = (float*)delbeta[nn];

#pragma omp simd
        for(int i=0; i<nOFM; i++)
        {
          dgp[i] += rdgp[i];
          dbp[i] += rdbp[i];
        }

#pragma vector nontemporal
#pragma omp simd
        for(int i=0; i<nOFM; i++)
        {
          rdgp[i] = dgp[i];
          rdbp[i] = dbp[i];
        }
      }
    }
#endif
  }

#ifdef USE_XSMM_TIMING__
  gettimeofday(&tvec, NULL);
  double wu_time = (tvec.tv_sec + tvec.tv_usec*1e-6) - (tvsc.tv_sec + tvsc.tv_usec*1e-6);

#ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
  {
    double gf = (double)gp->batch_size * (double)gp->nInput[0] * (double)gp->nOutput * (double)gp->mHeight * (double)gp->mWidth * (double)gp->kh * (double)gp->kw * 2;
    if(gp->c_stride_h == 1 && gp->mpad_h == 0)
      printf("%s XSMM-CONV-WU mb%dic%dih%doc%doh%dkh%dn time = %g ms, GFLOPS = %.1f\n",gp->node_name.c_str(),gp->batch_size,gp->nInput[0],gp->iHeight,gp->nOutput,gp->mHeight,gp->kh,wu_time*1000.0, gf/wu_time/1e9);
    else if(gp->c_stride_h == 2)
      printf("%s XSMM-CONV-WU mb%dic%dih%doc%doh%dkh%dsh%dn time = %g ms, GFLOPS = %.1f\n",gp->node_name.c_str(),gp->batch_size,gp->nInput[0],gp->iHeight,gp->nOutput,gp->mHeight,gp->kh,gp->c_stride_h,wu_time*1000.0, gf/wu_time/1e9);
    else if(gp->mpad_h == 1)
      printf("%s XSMM-CONV-WU mb%dic%dih%doc%doh%dkh%dph%dn time = %g ms, GFLOPS = %.1f\n",gp->node_name.c_str(),gp->batch_size,gp->nInput[0],gp->iHeight,gp->nOutput,gp->mHeight,gp->kh,gp->mpad_h,wu_time*1000.0, gf/wu_time/1e9);
  }
#endif

#ifndef NDEBUG
  /* check physical padding */
  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)delmiddle[0], nImg, nBOfm, mfh, mfw, 16, mph, mpw);
  else if(gp->in_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)delmiddle[0], nImg, nBOfm, mfh, mfw, 16, mph, mpw);
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
