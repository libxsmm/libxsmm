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

#include "ConvXSMM.hpp"

using namespace std;

ConvXSMM::ConvXSMM(ConvImplParams* gp, int engine) : ConvImpl(gp, engine)
{
  conv_desc.N = gp->batch_size;
  conv_desc.C = gp->nInput;
  conv_desc.H = gp->iHeight;
  conv_desc.W = gp->iWidth;
  conv_desc.K = gp->nOutput;
  conv_desc.R = gp->kh;
  conv_desc.S = gp->kw;
  conv_desc.u = gp->stride_h;
  conv_desc.v = gp->stride_w;

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

  conv_desc.pad_w = gp->pad_w;
  conv_desc.pad_h = gp->pad_h;

  if(gp->physical_padding)
  {
    conv_desc.pad_h_out = gp->opad_h;
    conv_desc.pad_w_out = gp->opad_w;
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
  {
    conv_desc.options = LIBXSMM_DNN_CONV_OPTION_OVERWRITE;
    conv_desc.options = LIBXSMM_DNN_CONV_OPTION_F32_BF16_CVT_RNE_OVERWRITE;
  }

  if(gp->bias_term)
    conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_BIAS;
  if(gp->relu)
    conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_RELU;
  if(gp->bias_term && gp->relu)
    conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_BIAS_RELU;
  if(gp->compute_stats)
    conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_BATCH_STATS_FWD;
  if(gp->compute_stats && gp->bwd_relu)
    conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_BATCH_STATS_FWD_RELU_BWD;

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

  libxsmm_handle = libxsmm_dnn_create_conv_layer( conv_desc, &status );
  CHKERR_LIBXSMM_DNN( status );

  top_layout_type = LIBXSMM_CUSTOM_LAYOUT;
  top_layout = libxsmm_handle;
  gbot_layout_type = LIBXSMM_CUSTOM_LAYOUT;
  gbot_layout = libxsmm_handle;
}

void ConvXSMM::forwardPropagate(TensorBuf *inp, TensorBuf *weightp, TensorBuf *biasp, TensorBuf *outp, int tid)
{
  // Conv input
  if(inp->getLPBuffer() != NULL)
    in_ptr = inp->getLPBuffer();
  else
    in_ptr = inp->getBuffer();

  // Conv Weight
  if(weightp->getLPBuffer() != NULL)
    wt_ptr = weightp->getLPBuffer();
  else
    wt_ptr = weightp->getBuffer();
  void *wt_prv_ptr = NULL;

  // Conv output
  out_ptr = outp->getBuffer();

  //Stats of output appended to output buffer
  int offset = conv_desc.N * conv_desc.K * (gp->oHeight + 2*conv_desc.pad_h_out) * (gp->oWidth + 2*conv_desc.pad_w_out);
  void *stats_ptr;

  if(gp->compute_stats)
  {
    if(gp->out_data_type == DT_FLOAT)
      stats_ptr = out_ptr + offset * sizeof(float);
    else if(gp->out_data_type == DT_BF16)
      stats_ptr = out_ptr + offset * sizeof(libxsmm_bfloat16);
  }

  void *scratch = scratchp->getBuffer();

  if(libxsmm_input == NULL && libxsmm_filter == NULL && libxsmm_output == NULL)
  {
    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_input = libxsmm_dnn_link_tensor( libxsmm_layout, in_ptr, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_input, LIBXSMM_DNN_REGULAR_INPUT ) );

    // Assume that weights are in KCRS format
    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER, &status );
    CHKERR_LIBXSMM_DNN( status );

    int welem = gp->nInput * gp->nOutput * gp->kw * gp->kh;
    if(gp->in_data_type == DT_FLOAT)
    {
      wt_prv_ptr = (void*)libxsmm_aligned_malloc(welem*sizeof(float), 2097152);
      libxsmm_filter = libxsmm_dnn_link_tensor( libxsmm_layout, wt_prv_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );

      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_filter, (void*)wt_ptr, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );
      memcpy(wt_ptr, wt_prv_ptr, welem*sizeof(float));
      
      libxsmm_checkpoint_filter = libxsmm_dnn_link_tensor(libxsmm_layout, wt_ptr, &status);
      CHKERR_LIBXSMM_DNN( status );

      libxsmm_filter = libxsmm_dnn_link_tensor( libxsmm_layout, wt_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );

      libxsmm_free(wt_prv_ptr);
      wt_prv_ptr = NULL;
      weightp->setPrivBuffer(NULL);
    }
    else if(gp->in_data_type == DT_BF16)
    {
      wt_prv_ptr = (void*)libxsmm_aligned_malloc(welem*sizeof(libxsmm_bfloat16), 2097152);
      libxsmm_filter = libxsmm_dnn_link_tensor( libxsmm_layout, wt_prv_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );

      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_filter, (void*)wt_ptr, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );
      memcpy(wt_ptr, wt_prv_ptr, welem*sizeof(libxsmm_bfloat16));
      libxsmm_filter = libxsmm_dnn_link_tensor( libxsmm_layout, wt_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_free(wt_prv_ptr);

      libxsmm_layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
      wt_prv_ptr = (void*)libxsmm_aligned_malloc(welem*sizeof(float), 2097152);
      libxsmm_checkpoint_filter = libxsmm_dnn_link_tensor( libxsmm_layout, wt_prv_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );
      void *fwt_ptr = weightp->getBuffer();
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_checkpoint_filter, (void*)fwt_ptr, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );
      memcpy(fwt_ptr, wt_prv_ptr, welem*sizeof(float));

      libxsmm_checkpoint_filter = libxsmm_dnn_link_tensor( libxsmm_layout, fwt_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );

      libxsmm_free(wt_prv_ptr);
      wt_prv_ptr = NULL;
      weightp->setPrivBuffer(NULL);
    }

    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_filter, LIBXSMM_DNN_REGULAR_FILTER ) );

    // Conv Output
    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, &status );
    CHKERR_LIBXSMM_DNN(      status );
    libxsmm_output = libxsmm_dnn_link_tensor( libxsmm_layout, out_ptr, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_output, LIBXSMM_DNN_REGULAR_OUTPUT ) );

    if(gp->compute_stats)
    {
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_BATCH_STATS, &status ); 
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_batchstats  = libxsmm_dnn_link_tensor( libxsmm_layout, stats_ptr, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_batchstats, LIBXSMM_DNN_BATCH_STATS ) );
    }

    /* let's allocate (if required) and bind scratch */
    if(scratch == NULL)
    {
      long long int mysize = libxsmm_dnn_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
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
      long long int mysize = libxsmm_dnn_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );

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
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
    updated_scratch = true;
  }

#ifndef NDEBUG
  /* check physical padding */
  if ( (gp->ipad_h > 0 || gp->ipad_w > 0) && (gp->opad_h > 0 || gp->opad_w > 0) ) {
  } else if ( (gp->ipad_h == 0 || gp->ipad_w == 0) && (gp->opad_h == 0 || gp->opad_w == 0) ) {
  } else {
    printf("node %s: conv xsmm forward is partially padded which cannot be :-(\n", nname.c_str());
  }

  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)in_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );
  else if(gp->in_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)in_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );

  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)out_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
  else if(gp->in_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)out_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );

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

      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
    }
#ifdef USE_XSMM_TIMING
  gettimeofday(&tvec, NULL);
  double fp_time = (tvec.tv_sec + tvec.tv_usec*1e-6) - (tvsc.tv_sec + tvsc.tv_usec*1e-6);

#ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
  {
    double gf = (double)gp->batch_size * (double)gp->nInput * (double)gp->nOutput * (double)gp->oHeight * (double)gp->oWidth * (double)gp->kh * (double)gp->kw * 2;
    if(gp->stride_h == 1 && gp->pad_h == 0)
      printf("XSMM-CONV-FP mb%dic%dih%doc%doh%dkh%dn time = %g ms, GFLOPS = %.1f\n",gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,fp_time*1000.0, gf/fp_time/1e9);
    else if(gp->stride_h == 2)
      printf("XSMM-CONV-FP mb%dic%dih%doc%doh%dkh%dsh%dn time = %g ms, GFLOPS = %.1f\n",gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,gp->stride_h,fp_time*1000.0, gf/fp_time/1e9);
    else if(gp->pad_h == 1)
      printf("XSMM-CONV-FP mb%dic%dih%doc%doh%dkh%dph%dn time = %g ms, GFLOPS = %.1f\n",gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,gp->pad_h,fp_time*1000.0, gf/fp_time/1e9);
  }
#endif

  top_layout_type = LIBXSMM_CUSTOM_LAYOUT;
  outp->setLayoutType(top_layout_type);
  outp->setLayout(libxsmm_handle);

#ifndef NDEBUG
  /* check physical padding */
  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)in_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );
  else if(gp->in_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)in_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );

  if(gp->out_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)out_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
  else if(gp->out_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)out_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
#endif
}

void ConvXSMM::backPropagate(TensorBuf* inp, TensorBuf* weightp, TensorBuf *deloutp, TensorBuf* delinp, int tid)
{
  dout_ptr = deloutp->getBuffer();

  din_ptr = delinp->getBuffer();

  if(scratch != scratchp->getBuffer())
  {
    scratch = scratchp->getBuffer();
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
  }

  if(libxsmm_delinput == NULL && libxsmm_deloutput == NULL)
  {
    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_delinput = libxsmm_dnn_link_tensor(libxsmm_layout, din_ptr, &status );
    CHKERR_LIBXSMM_DNN(status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_delinput, LIBXSMM_DNN_GRADIENT_INPUT ) );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
    CHKERR_LIBXSMM_DNN(status );
    libxsmm_deloutput = libxsmm_dnn_link_tensor( libxsmm_layout, dout_ptr, &status );
    CHKERR_LIBXSMM_DNN(status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_deloutput, LIBXSMM_DNN_GRADIENT_OUTPUT ) );
  }

#ifndef NDEBUG
  /* check physical padding */
  if ( (gp->ipad_h > 0 || gp->ipad_w > 0) && (gp->opad_h > 0 || gp->opad_w > 0) ) {
  } else if ( (gp->ipad_h == 0 || gp->ipad_w == 0) && (gp->opad_h == 0 || gp->opad_w == 0) ) {
  } else {
    printf("node %s: conv xsmm backward is partially padded which cannot be :-(\n", nname.c_str());
  }
  if(gp->out_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)din_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );
  else if(gp->out_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)din_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );

  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)dout_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
  else if(gp->in_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)dout_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
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

      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
    }

#ifdef USE_XSMM_TIMING
  gettimeofday(&tvec, NULL);
  double bp_time = (tvec.tv_sec + tvec.tv_usec*1e-6) - (tvsc.tv_sec + tvsc.tv_usec*1e-6);

#ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
  {
    double gf = (double)gp->batch_size * (double)gp->nInput * (double)gp->nOutput * (double)gp->oHeight * (double)gp->oWidth * (double)gp->kh * (double)gp->kw * 2;
    if(gp->stride_h == 1 && gp->pad_h == 0)
      printf("XSMM-CONV-BP mb%dic%dih%doc%doh%dkh%dn time = %g ms, GFLOPS = %.1f\n",gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,bp_time*1000.0, gf/bp_time/1e9);
    else if(gp->stride_h == 2)
      printf("XSMM-CONV-BP mb%dic%dih%doc%doh%dkh%dsh%dn time = %g ms, GFLOPS = %.1f\n",gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,gp->stride_h,bp_time*1000.0, gf/bp_time/1e9);
    else if(gp->pad_h == 1)
      printf("XSMM-CONV-BP mb%dic%dih%doc%doh%dkh%dph%dn time = %g ms, GFLOPS = %.1f\n",gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,gp->pad_h,bp_time*1000.0, gf/bp_time/1e9);
  }
#endif

  gbot_layout_type = LIBXSMM_CUSTOM_LAYOUT;
  delinp->setLayoutType(gbot_layout_type);
  delinp->setLayout(libxsmm_handle);

#ifndef NDEBUG
  /* check physical padding */
  if(gp->out_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)din_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );
  else if(gp->out_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)din_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );

  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)dout_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
  else if(gp->in_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)dout_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
#endif
}

void ConvXSMM::weightUpdate(TensorBuf *inp, TensorBuf *deloutp, TensorBuf* delweightp, TensorBuf* delbiasp, int tid)
{
  void *dwt_ptr = delweightp->getBuffer();

  if(scratch != scratchp->getBuffer())
  {
    scratch = scratchp->getBuffer();
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
  }

  if(libxsmm_delfilter == NULL)
  {
    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER, &status );
    CHKERR_LIBXSMM_DNN(status );
    libxsmm_delfilter = libxsmm_dnn_link_tensor( libxsmm_layout, dwt_ptr, &status );
    CHKERR_LIBXSMM_DNN(status);
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_delfilter, LIBXSMM_DNN_GRADIENT_FILTER ) );
  }

  if(libxsmm_deloutput == NULL)
  {
    dout_ptr = deloutp->getBuffer();
    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
    CHKERR_LIBXSMM_DNN(      status );
    libxsmm_deloutput = libxsmm_dnn_link_tensor(libxsmm_layout, dout_ptr, &status );
    CHKERR_LIBXSMM_DNN(status);
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_deloutput, LIBXSMM_DNN_GRADIENT_OUTPUT ) );
  }

#ifndef NDEBUG
  /* check physical padding */
  if ( (gp->ipad_h > 0 || gp->ipad_w > 0) && (gp->opad_h > 0 || gp->opad_w > 0) ) {
  } else if ( (gp->ipad_h == 0 || gp->ipad_w == 0) && (gp->opad_h == 0 || gp->opad_w == 0) ) {
  } else {
    printf("node %s: conv xsmm backward is partially padded which cannot be :-(\n", nname.c_str());
  }
  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)in_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );
  else if(gp->in_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)in_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );

  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)dout_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
  else if(gp->in_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)dout_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
#endif

  if(gp->bias_term)
  {
    /* Conv del-bias */
    float* delbias = (float*)delbiasp->getBuffer();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<gp->nOutput; i++)
      delbias[i] = 0.0;

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for(int ofm1=0; ofm1<gp->nOutput/16; ofm1++)
      for(int ofm2=0; ofm2<16; ofm2++)
        for(int img=0; img<gp->batch_size; img++)
          for(int ofh=0; ofh<gp->oHeight; ofh++)
            for(int ofw=0; ofw<gp->oWidth; ofw++)
            {
              float* delout= (float*)dout_ptr;
              int in_idx = img * gp->nOutput * gp->oHeight * gp->oWidth + ofm1 * gp->oHeight * gp->oWidth * 16 + ofh * gp->oWidth * 16 + ofw*16 + ofm2;
              int out_idx = ofm1*16 + ofm2;
              delbias[out_idx] += delout[in_idx];
            }
  }

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

      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid ) );
    }

#ifdef USE_XSMM_TIMING
  gettimeofday(&tvec, NULL);
  double wu_time = (tvec.tv_sec + tvec.tv_usec*1e-6) - (tvsc.tv_sec + tvsc.tv_usec*1e-6);

#ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
  {
    double gf = (double)gp->batch_size * (double)gp->nInput * (double)gp->nOutput * (double)gp->oHeight * (double)gp->oWidth * (double)gp->kh * (double)gp->kw * 2;
    if(gp->stride_h == 1 && gp->pad_h == 0)
      printf("XSMM-CONV-WU mb%dic%dih%doc%doh%dkh%dn time = %g ms, GFLOPS = %.1f\n",gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,wu_time*1000.0, gf/wu_time/1e9);
    else if(gp->stride_h == 2)
      printf("XSMM-CONV-WU mb%dic%dih%doc%doh%dkh%dsh%dn time = %g ms, GFLOPS = %.1f\n",gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,gp->stride_h,wu_time*1000.0, gf/wu_time/1e9);
    else if(gp->pad_h == 1)
      printf("XSMM-CONV-WU mb%dic%dih%doc%doh%dkh%dph%dn time = %g ms, GFLOPS = %.1f\n",gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,gp->pad_h,wu_time*1000.0, gf/wu_time/1e9);
  }
#endif

#ifndef NDEBUG
  /* check physical padding */
  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)in_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );
  else if(gp->in_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)in_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );

  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)dout_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
  else if(gp->in_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)dout_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
#endif
}

void ConvXSMM::dumpBuffer(TensorBuf* tBuf, void* wtemp)
{
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_checkpoint_filter, (void*)wtemp, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );
}
