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
  conv_desc.options = LIBXSMM_DNN_CONV_OPTION_OVERWRITE;

  if(gp->bias_term)
    conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_BIAS;
  if(gp->relu)
    conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_RELU;
  if(gp->bias_term && gp->relu)
    conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_BIAS_RELU;
  if(gp->compute_stats)
    conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_BATCH_STATS;
  if(gp->compute_stats && gp->bwd_relu)
    conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_BATCH_STATS_RELU_BWD;

  if(gp->in_data_type == DT_DFP16 && gp->out_data_type == DT_FLOAT)
  {
    conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_I16;
    conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
  }
  else if(gp->in_data_type == DT_DFP16 && gp->out_data_type == DT_DFP16)
  {
    conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_I16;
    conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_I16;
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
#ifdef TIMING_OV
  struct timeval tvs, tve;

  gettimeofday(&tvs, NULL);
#endif
  assert(bot_compute_engine != -1);
  assert(top_compute_engine != -1);

  // Conv input
  if(gp->in_data_type == DT_FLOAT)
    in_ptr = inp->getBuffer();
  else if(gp->in_data_type == DT_DFP16)
  {
    in_ptr = inp->getLPBuffer();
    assert(in_ptr != NULL);
    scf_input = inp->getLPSF();
  }
  if(gp->in_data_type == DT_FLOAT)
    in_prv_ptr = inp->getPrivBuffer();
  else if(gp->in_data_type == DT_DFP16)
  {
    in_prv_ptr = inp->getLPPrivBuffer();
    if(in_prv_ptr != NULL)
      scf_input = inp->getLPPrivSF();
  }

  // Conv output
  out_ptr = outp->getBuffer();
  out_prv_ptr = outp->getPrivBuffer();

  // Conv Weight
  if(gp->in_data_type == DT_FLOAT)
    wt_ptr = weightp->getBuffer();
  else if(gp->in_data_type == DT_DFP16)
  {
    wt_ptr = weightp->getLPBuffer();
    assert(wt_ptr != NULL);
    scf_filter = weightp->getLPSF();
  }
  void *wt_prv_ptr = NULL;

  //Stats of output appended to output buffer
  int offset = conv_desc.N * conv_desc.K * (gp->oHeight + 2*conv_desc.pad_h_out) * (gp->oWidth + 2*conv_desc.pad_w_out);
  void *stats_ptr = out_ptr + offset * sizeof(float);

  void *scratch = scratchp->getBuffer();

  if(libxsmm_input == NULL && libxsmm_filter == NULL && libxsmm_output == NULL)
  {
    if(bot_compute_engine != engine)
    {
      if(in_prv_ptr == NULL)
      {
        int size = gp->batch_size * gp->nInput * (gp->iHeight + 2*gp->ipad_h) * (gp->iWidth + 2*gp->ipad_w);
        if(gp->in_data_type == DT_DFP16)
          in_prv_ptr = (void*)libxsmm_aligned_malloc(size*sizeof(short), 2097152);
        else if(gp->in_data_type == DT_FLOAT)
          in_prv_ptr = (void*)libxsmm_aligned_malloc(size*sizeof(float), 2097152);

        inp->setPrivBuffer(in_prv_ptr);
      }

      /* setup LIBXSMM buffers and filter */
      in_buffer_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input = libxsmm_dnn_link_tensor( in_buffer_layout, in_prv_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );
    }
    else
    {
      in_buffer_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input = libxsmm_dnn_link_tensor( in_buffer_layout, in_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );
    }

    // Bind input buffer to handle
    CHKERR_LIBXSMM_DNN_BIND("input", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_input, LIBXSMM_DNN_REGULAR_INPUT ) );

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

      libxsmm_free(wt_prv_ptr);
      wt_prv_ptr = NULL;
      weightp->setPrivBuffer(NULL);
    }

    libxsmm_filter = libxsmm_dnn_link_tensor( libxsmm_layout, wt_ptr, &status );

    CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    CHKERR_LIBXSMM_DNN_BIND("wt", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_filter, LIBXSMM_DNN_REGULAR_FILTER ) );

    // Conv Output
    if(top_compute_engine != engine)
    {
      if(out_prv_ptr == NULL)
      {

        int size = gp->batch_size * gp->nOutput * (gp->oHeight + 2*conv_desc.pad_h_out) * (gp->oWidth + 2*conv_desc.pad_w_out);
        out_prv_ptr = (void*)libxsmm_aligned_malloc(size*sizeof(float), 2097152);
        outp->setPrivBuffer(out_prv_ptr);
      }

      out_buffer_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN(      status );
      libxsmm_output = libxsmm_dnn_link_tensor( out_buffer_layout, out_prv_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );
    }
    else
    {
      out_buffer_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN(      status );
      libxsmm_output = libxsmm_dnn_link_tensor( out_buffer_layout, out_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );
    }

    CHKERR_LIBXSMM_DNN_BIND("out", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_output, LIBXSMM_DNN_REGULAR_OUTPUT ) );

    if(gp->compute_stats)
    {
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_BATCH_STATS, &status ); CHKERR_LIBXSMM_DNN( status );
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

  if(bot_compute_engine != engine)
  {
    assert(gp->in_data_type == DT_FLOAT);
    assert(in_buffer_layout->num_dims == 5);
    assert(in_prv_ptr != NULL);

    /* copy input data to LIBXSMM format */
    int i1, i2, i3, i4, i5;
    int N = in_buffer_layout->dim_size[4];
    int fmb = in_buffer_layout->dim_size[3];
    int bfm = in_buffer_layout->dim_size[0];
    int H = in_buffer_layout->dim_size[2];
    int W = in_buffer_layout->dim_size[1];

    LIBXSMM_VLA_DECL(4, const float, user_data, (const float*)in_ptr, fmb * bfm, H, W);
    LIBXSMM_VLA_DECL(5, float, handle_data_1, (float*)in_prv_ptr, fmb, H, W, bfm);

#ifdef _OPENMP
#pragma omp parallel for collapse(4) private(i1, i2, i3, i4, i5)
#endif
    for (i1 = 0; i1 < N; ++i1) {
      for (i2 = 0; i2 < fmb; ++i2) {
        for (i3 = 0; i3 < H; ++i3) {
          for (i4 = 0; i4 < W; ++i4) {
            for (i5 = 0; i5 < bfm; ++i5) {
              LIBXSMM_VLA_ACCESS(5, handle_data_1, i1, i2, i3, i4, i5, fmb, H, W, bfm) =
                LIBXSMM_VLA_ACCESS(4, user_data, i1, (i2*bfm) + i5, i3, i4, fmb * bfm, H, W);
            }
          }
        }
      }
    }
  }
  else
  {
    if(gp->in_data_type == DT_DFP16)
      libxsmm_dnn_set_qtensor_scf(libxsmm_input, scf_input);

    if(!destroyed_in_)
    {
      libxsmm_dnn_destroy_tensor_datalayout( in_buffer_layout );
      destroyed_in_ = true;
    }
  }

  if(gp->in_data_type == DT_DFP16 || gp->out_data_type == DT_DFP16)
    libxsmm_dnn_set_qtensor_scf(libxsmm_filter, scf_filter);

#ifdef TIMING_OV
  gettimeofday(&tve, NULL);

  double fpo_time = (tve.tv_sec + tve.tv_usec*1e-6) - (tvs.tv_sec + tvs.tv_usec*1e-6);

#ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
  {
    printf("Conv FP Overhead time = %g s\n",fpo_time);
  }
#endif

  if(conv_desc.options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE == false)
  {
    int nImg = gp->batch_size;
    int nOfm = gp->nOutput;
    int ofh = gp->oHeight;
    int ofw = gp->oWidth;
    float* out = (out_prv_ptr != NULL) ? (float*)out_prv_ptr : (float*)out_ptr;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<nImg*nOfm*ofh*ofw; i++)
      out[i] = 0.0;
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
  else if(gp->in_data_type == DT_DFP16)
    check_physical_pad( nname.c_str(), (short*)in_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );

  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)out_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
  else if(gp->in_data_type == DT_DFP16)
    check_physical_pad( nname.c_str(), (short*)out_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );

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

#ifdef TIMING_PO
  struct timeval tvsp, tvep;

  gettimeofday(&tvsp, NULL);
#endif

  if(top_compute_engine != engine)
  {
    assert(out_buffer_layout->num_dims == 5);
    assert(out_prv_ptr != NULL);

    /* copy input data to LIBXSMM format */
    int o1, o2, o3, o4, o5;
    int N = out_buffer_layout->dim_size[4];
    int fmb = out_buffer_layout->dim_size[3];
    int bfm = out_buffer_layout->dim_size[0];
    int H = out_buffer_layout->dim_size[2];
    int W = out_buffer_layout->dim_size[1];

    LIBXSMM_VLA_DECL(4, float, out_user_data, (float*)out_ptr, fmb * bfm, H, W);
    LIBXSMM_VLA_DECL(5, const float, out_handle_data_1, (const float*)out_prv_ptr, fmb, H, W, bfm);

#ifdef _OPENMP
#pragma omp parallel for collapse(4) private(o1, o2, o3, o4, o5)
#endif
    for (o1 = 0; o1 < N; ++o1) {
      for (o2 = 0; o2 < fmb; ++o2) {
        for (o3 = 0; o3 < H; ++o3) {
          for (o4 = 0; o4 < W; ++o4) {
            for (o5 = 0; o5 < bfm; ++o5) {
              LIBXSMM_VLA_ACCESS(4, out_user_data, o1, (o2*bfm) + o5, o3, o4, fmb * bfm, H, W) =
                LIBXSMM_VLA_ACCESS(5, out_handle_data_1, o1, o2, o3, o4, o5, fmb, H, W, bfm);
            }
          }
        }
      }
    }
    top_layout_type = NCHW;
    outp->setLayoutType(top_layout_type);
    outp->setLayout(NULL);
  }
  else
  {
    if(!destroyed_out_)
    {
      libxsmm_dnn_destroy_tensor_datalayout(out_buffer_layout);
      destroyed_out_ = true;
    }

    top_layout_type = LIBXSMM_CUSTOM_LAYOUT;
    outp->setLayoutType(top_layout_type);
    outp->setLayout(libxsmm_handle);
  }
#ifdef TIMING_PO
  gettimeofday(&tvep, NULL);

  double fpp_time = (tvep.tv_sec + tvep.tv_usec*1e-6) - (tvsp.tv_sec + tvsp.tv_usec*1e-6);

#ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
  {
    printf("Conv FP post compute time = %g s\n",fpp_time);
  }
#endif

#ifndef NDEBUG
  /* check physical padding */
  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)in_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );
  else if(gp->in_data_type == DT_DFP16)
    check_physical_pad( nname.c_str(), (short*)in_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );

  if(gp->out_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)out_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
  else if(gp->out_data_type == DT_DFP16)
    check_physical_pad( nname.c_str(), (short*)out_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
#endif
}

void ConvXSMM::backPropagate(TensorBuf* inp, TensorBuf* weightp, TensorBuf *deloutp, TensorBuf* delinp, int tid)
{
#ifdef TIMING_OV
  struct timeval tvs, tve;

  gettimeofday(&tvs, NULL);
#endif

  assert(bot_compute_engine != -1);
  assert(top_compute_engine != -1);

  if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
    dout_ptr = deloutp->getBuffer();
  else if(gp->in_data_type == DT_DFP16 || gp->out_data_type == DT_DFP16)
  {
    dout_ptr = deloutp->getLPBuffer();
    assert(dout_ptr != NULL);
    scf_deloutput = deloutp->getLPSF();
  }

  if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
    dout_prv_ptr = deloutp->getPrivBuffer();
  else if(gp->in_data_type == DT_DFP16 || gp->out_data_type == DT_DFP16)
  {
    dout_prv_ptr = deloutp->getLPPrivBuffer();
    if(dout_prv_ptr != NULL)
      pscf_deloutput = deloutp->getLPPrivSF();
  }

  din_ptr = delinp->getBuffer();
  din_prv_ptr = delinp->getPrivBuffer();
  dout_converted_in_BP = false;

  if(scratch != scratchp->getBuffer())
  {
    scratch = scratchp->getBuffer();
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
  }

  if(libxsmm_delinput == NULL && libxsmm_deloutput == NULL)
  {
    if(bot_compute_engine != engine)
    {
      if(din_prv_ptr == NULL)
      {
        int size = gp->batch_size * gp->nInput * (gp->iHeight + 2*conv_desc.pad_h_in) * (gp->iWidth + 2*conv_desc.pad_w_in);
        din_prv_ptr = (void*)libxsmm_aligned_malloc(size*sizeof(float), 2097152);
        delinp->setPrivBuffer(din_prv_ptr);
      }

      /* setup LIBXSMM buffers and filter */
      din_buffer_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, &status );
      CHKERR_LIBXSMM_DNN(      status );
      libxsmm_delinput = libxsmm_dnn_link_tensor(din_buffer_layout, din_prv_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );
    }
    else
    {
      din_buffer_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, &status );
      CHKERR_LIBXSMM_DNN_CREATE("delin", status );
      libxsmm_delinput = libxsmm_dnn_link_tensor(din_buffer_layout, din_ptr, &status );
      CHKERR_LIBXSMM_DNN_LINK("link", status );
    }
    CHKERR_LIBXSMM_DNN_BIND( "delin", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_delinput, LIBXSMM_DNN_GRADIENT_INPUT ) );

    if(top_compute_engine != engine)
    {
      if(dout_prv_ptr == NULL)
      {
        int size = gp->batch_size * gp->nOutput * (gp->oHeight + 2*conv_desc.pad_h_out) * (gp->oWidth + 2*conv_desc.pad_w_out);
        if(gp->out_data_type == DT_DFP16) {
          dout_prv_ptr = (void*)libxsmm_aligned_malloc(size*sizeof(short), 2097152);
        } else {
          dout_prv_ptr = (void*)libxsmm_aligned_malloc(size*sizeof(float), 2097152);
        }
        deloutp->setPrivBuffer(dout_prv_ptr);
      }

      dout_buffer_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN_CREATE("delout", status );
      libxsmm_deloutput = libxsmm_dnn_link_tensor( dout_buffer_layout, dout_prv_ptr, &status );
      CHKERR_LIBXSMM_DNN_LINK("delout", status );
    }
    else
    {
      dout_buffer_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN_CREATE("delout", status );

      libxsmm_deloutput = libxsmm_dnn_link_tensor( dout_buffer_layout, dout_ptr, &status );

      CHKERR_LIBXSMM_DNN_LINK("delout", status );
    }

    CHKERR_LIBXSMM_DNN_BIND("delout", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_deloutput, LIBXSMM_DNN_GRADIENT_OUTPUT ) );
  }

  if(top_compute_engine != engine)
  {
    assert(dout_prv_ptr != NULL);

    if(gp->in_data_type == DT_DFP16 || gp->out_data_type == DT_DFP16)
    {
      libxsmm_dnn_set_qtensor_scf(libxsmm_deloutput, pscf_deloutput);
      dout_converted_in_BP = true;
    }
    else
    {
      /* copy input data to LIBXSMM format */
      int o1, o2, o3, o4, o5;
      int N = out_buffer_layout->dim_size[4];
      int fmb = out_buffer_layout->dim_size[3];
      int bfm = out_buffer_layout->dim_size[0];
      int H = out_buffer_layout->dim_size[2];
      int W = out_buffer_layout->dim_size[1];

      LIBXSMM_VLA_DECL(4, const float, dout_user_data, (const float*)dout_ptr, fmb * bfm, H, W);
      LIBXSMM_VLA_DECL(5, float, dout_handle_data_1, (float*)dout_prv_ptr, fmb, H, W, bfm);
#ifdef _OPENMP
#pragma omp parallel for collapse(4) private(o1, o2, o3, o4, o5)
#endif
      for (o1 = 0; o1 < N; ++o1) {
        for (o2 = 0; o2 < fmb; ++o2) {
          for (o3 = 0; o3 < H; ++o3) {
            for (o4 = 0; o4 < W; ++o4) {
              for (o5 = 0; o5 < bfm; ++o5) {
                LIBXSMM_VLA_ACCESS(5, dout_handle_data_1, o1, o2, o3, o4, o5, fmb, H, W, bfm) =
                  LIBXSMM_VLA_ACCESS(4, dout_user_data, o1, (o2*bfm) + o5, o3, o4, fmb * bfm, H, W);
              }
            }
          }
        }
      }
      dout_converted_in_BP = true;
    }
  }
  else
  {
    if(gp->in_data_type == DT_DFP16 || gp->out_data_type == DT_DFP16)
    {
      libxsmm_dnn_set_qtensor_scf(libxsmm_deloutput, scf_deloutput);

      dout_converted_in_BP = true;
    }

    if(!destroyed_dout_)
    {
      libxsmm_dnn_destroy_tensor_datalayout( dout_buffer_layout );
      destroyed_dout_ = true;
    }
  }

  float* dinp = (din_prv_ptr != NULL) ? (float*)din_prv_ptr : (float*)din_ptr;

  if(conv_desc.options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE == false)
  {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<gp->batch_size*gp->nInput*(gp->iHeight+2*gp->ipad_h)*(gp->iWidth+2*gp->ipad_w); i++)
        dinp[i] = 0.0;
  }

#ifdef TIMING_OV
  gettimeofday(&tve, NULL);

  double bpo_time = (tve.tv_sec + tve.tv_usec*1e-6) - (tvs.tv_sec + tvs.tv_usec*1e-6);

#ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
  {
    printf("Conv BP Overhead time = %g s\n",bpo_time);
  }
#endif

#ifndef NDEBUG
  /* check physical padding */
  if ( (gp->ipad_h > 0 || gp->ipad_w > 0) && (gp->opad_h > 0 || gp->opad_w > 0) ) {
  } else if ( (gp->ipad_h == 0 || gp->ipad_w == 0) && (gp->opad_h == 0 || gp->opad_w == 0) ) {
  } else {
    printf("node %s: conv xsmm backward is partially padded which cannot be :-(\n", nname.c_str());
  }
  if(gp->out_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)din_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );
  else if(gp->out_data_type == DT_DFP16)
    check_physical_pad( nname.c_str(), (short*)din_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );

  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)dout_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
  else if(gp->in_data_type == DT_DFP16)
    check_physical_pad( nname.c_str(), (short*)dout_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
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

#ifdef TIMING_PO
  struct timeval tvsp, tvep;

  gettimeofday(&tvsp, NULL);
#endif

  if(bot_compute_engine != engine)
  {
    assert(din_buffer_layout->num_dims == 5);
    assert(din_prv_ptr != NULL);

    /* copy input data to LIBXSMM format */
    int i1, i2, i3, i4, i5;
    int N = din_buffer_layout->dim_size[4];
    int fmb = din_buffer_layout->dim_size[3];
    int bfm = din_buffer_layout->dim_size[0];
    int H = din_buffer_layout->dim_size[2];
    int W = din_buffer_layout->dim_size[1];

    LIBXSMM_VLA_DECL(4, float, user_data, (float*)din_ptr, fmb * bfm, H, W);
    LIBXSMM_VLA_DECL(5, const float, handle_data_1, (const float*)din_prv_ptr, fmb, H, W, bfm);

#ifdef _OPENMP
#pragma omp parallel for collapse(4) private(i1, i2, i3, i4, i5)
#endif
    for (i1 = 0; i1 < N; ++i1) {
      for (i2 = 0; i2 < fmb; ++i2) {
        for (i3 = 0; i3 < H; ++i3) {
          for (i4 = 0; i4 < W; ++i4) {
            for (i5 = 0; i5 < bfm; ++i5) {
              LIBXSMM_VLA_ACCESS(4, user_data, i1, (i2*bfm) + i5, i3, i4, fmb * bfm, H, W) =
                LIBXSMM_VLA_ACCESS(5, handle_data_1, i1, i2, i3, i4, i5, fmb, H, W, bfm);
            }
          }
        }
      }
    }
    gbot_layout_type = NCHW;
    delinp->setLayoutType(gbot_layout_type);
    delinp->setLayout(NULL);
  }
  else
  {
    if(!destroyed_din_)
    {
      libxsmm_dnn_destroy_tensor_datalayout(din_buffer_layout);
      destroyed_din_ = true;
    }
    gbot_layout_type = LIBXSMM_CUSTOM_LAYOUT;
    delinp->setLayoutType(gbot_layout_type);
    delinp->setLayout(libxsmm_handle);
  }

#ifdef TIMING_PO
  gettimeofday(&tvep, NULL);

  double bpp_time = (tvep.tv_sec + tvep.tv_usec*1e-6) - (tvsp.tv_sec + tvsp.tv_usec*1e-6);

#ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
  {
    printf("Conv BP post compute time = %g s\n",bpp_time);
  }
#endif

#ifndef NDEBUG
  /* check physical padding */
  if(gp->out_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)din_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );
  else if(gp->out_data_type == DT_DFP16)
    check_physical_pad( nname.c_str(), (short*)din_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );

  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)dout_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
  else if(gp->in_data_type == DT_DFP16)
    check_physical_pad( nname.c_str(), (short*)dout_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
#endif
}

void ConvXSMM::weightUpdate(TensorBuf *inp, TensorBuf *deloutp, TensorBuf* delweightp, TensorBuf* delbiasp, int tid)
{
#ifdef TIMING_OV
  struct timeval tvs, tve;

  gettimeofday(&tvs, NULL);
#endif

  if(libxsmm_deloutput == NULL)
  {
    if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
      dout_ptr = deloutp->getBuffer();
    else if(gp->in_data_type == DT_DFP16 || gp->out_data_type == DT_DFP16)
    {
      dout_ptr = deloutp->getLPBuffer();
      assert(dout_ptr != NULL);
      scf_deloutput = deloutp->getLPSF();
#ifndef NDEBUG
    int size = gp->batch_size * gp->nOutput * (gp->oHeight + 2*conv_desc.pad_h_out) * (gp->oWidth + 2*conv_desc.pad_w_out);
    for(int i=0; i<size; i++)
      short k = *((short*)dout_ptr + i);
    printf("size %d, all ok\n", size);
#endif
    }

    if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
      dout_prv_ptr = deloutp->getPrivBuffer();
    else if(gp->in_data_type == DT_DFP16 || gp->out_data_type == DT_DFP16)
    {
      dout_prv_ptr = deloutp->getLPPrivBuffer();
      if(dout_prv_ptr != NULL)
        pscf_deloutput = deloutp->getLPPrivSF();
    }
  }

  void *dwt_ptr = delweightp->getBuffer();

  if(scratch != scratchp->getBuffer())
  {
    scratch = scratchp->getBuffer();
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
  }

  assert(bot_compute_engine != -1);
  assert(top_compute_engine != -1);

  if(libxsmm_delfilter == NULL)
  {
    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER, &status );
    CHKERR_LIBXSMM_DNN_CREATE("delwt",status );
    libxsmm_delfilter = libxsmm_dnn_link_tensor( libxsmm_layout, dwt_ptr, &status );
    CHKERR_LIBXSMM_DNN_LINK("delwt", status);
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    /* bind buffers and filter to handle */
    CHKERR_LIBXSMM_DNN_BIND("delwt", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_delfilter, LIBXSMM_DNN_GRADIENT_FILTER ) );
  }

  if(libxsmm_deloutput == NULL)
  {
    if((top_compute_engine != engine) && dout_converted_in_BP == false)
    {
      if(dout_prv_ptr == NULL)
      {
        int size = gp->batch_size * gp->nOutput * (gp->oHeight + 2*gp->opad_h) * (gp->oWidth + 2*gp->opad_w);
        if(gp->in_data_type == DT_DFP16 || gp->out_data_type == DT_DFP16)
          dout_prv_ptr = (void*)libxsmm_aligned_malloc(size*sizeof(short), 2097152);
        else
          dout_prv_ptr = (void*)libxsmm_aligned_malloc(size*sizeof(float), 2097152);
        deloutp->setPrivBuffer(dout_prv_ptr);
      }
      dout_buffer_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN_CREATE("delout",  status );
      libxsmm_deloutput = libxsmm_dnn_link_tensor( dout_buffer_layout, dout_prv_ptr, &status );
      CHKERR_LIBXSMM_DNN_LINK("delout", status);
    }
    else
    {
      dout_buffer_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN(      status );

      libxsmm_deloutput = libxsmm_dnn_link_tensor(dout_buffer_layout, dout_ptr, &status );

      CHKERR_LIBXSMM_DNN_LINK("delout", status);
    }
    CHKERR_LIBXSMM_DNN_BIND("delout", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_deloutput, LIBXSMM_DNN_GRADIENT_OUTPUT ) );
  }

  if(top_compute_engine != engine && dout_converted_in_BP == false)
  {
    assert(dout_prv_ptr != NULL);

    if(gp->in_data_type == DT_DFP16 || gp->out_data_type == DT_DFP16)
      libxsmm_dnn_set_qtensor_scf(libxsmm_deloutput, scf_deloutput);
    else
    {
      assert(dout_buffer_layout->num_dims == 5);

      /* copy input data to LIBXSMM format */
      int o1, o2, o3, o4, o5;
      int N = dout_buffer_layout->dim_size[4];
      int fmb = dout_buffer_layout->dim_size[3];
      int bfm = dout_buffer_layout->dim_size[0];
      int H = dout_buffer_layout->dim_size[2];
      int W = dout_buffer_layout->dim_size[1];
      LIBXSMM_VLA_DECL(4, const float, dout_user_data, (const float*)dout_ptr, fmb * bfm, H, W);
      LIBXSMM_VLA_DECL(5, float, dout_handle_data_1, (float*)dout_prv_ptr, fmb, H, W, bfm);

#ifdef _OPENMP
#pragma omp parallel for collapse(4) private(o1, o2, o3, o4, o5)
#endif
      for (o1 = 0; o1 < N; ++o1) {
        for (o2 = 0; o2 < fmb; ++o2) {
          for (o3 = 0; o3 < H; ++o3) {
            for (o4 = 0; o4 < W; ++o4) {
              for (o5 = 0; o5 < bfm; ++o5) {
                LIBXSMM_VLA_ACCESS(5, dout_handle_data_1, o1, o2, o3, o4, o5, fmb, H, W, bfm) =
                  LIBXSMM_VLA_ACCESS(4, dout_user_data, o1, (o2*bfm) + o5, o3, o4, fmb * bfm, H, W);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    if((gp->in_data_type == DT_DFP16 || gp->out_data_type == DT_DFP16) && dout_converted_in_BP==false)
      libxsmm_dnn_set_qtensor_scf(libxsmm_deloutput, scf_deloutput);

    if(!destroyed_dout_)
    {
      libxsmm_dnn_destroy_tensor_datalayout( dout_buffer_layout );
      destroyed_dout_ = true;
    }
  }
  int wsize = gp->nInput * gp->nOutput * gp->kh * gp->kw;
  float *dwt = (float*)dwt_ptr;

  if(conv_desc.options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE == false)
  {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<wsize; i++)
      dwt[i] = 0.0;
  }

#ifdef TIMING_OV
  gettimeofday(&tve, NULL);

  double wuo_time = (tve.tv_sec + tve.tv_usec*1e-6) - (tvs.tv_sec + tvs.tv_usec*1e-6);

#ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
  {
    printf("Conv WU Overhead time = %g s\n",wuo_time);
  }
#endif

#ifndef NDEBUG
  /* check physical padding */
  if ( (gp->ipad_h > 0 || gp->ipad_w > 0) && (gp->opad_h > 0 || gp->opad_w > 0) ) {
  } else if ( (gp->ipad_h == 0 || gp->ipad_w == 0) && (gp->opad_h == 0 || gp->opad_w == 0) ) {
  } else {
    printf("node %s: conv xsmm backward is partially padded which cannot be :-(\n", nname.c_str());
  }
  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)in_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );
  else if(gp->in_data_type == DT_DFP16)
    check_physical_pad( nname.c_str(), (short*)in_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );

  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)dout_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
  else if(gp->in_data_type == DT_DFP16)
    check_physical_pad( nname.c_str(), (short*)dout_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
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
  else if(gp->in_data_type == DT_DFP16)
    check_physical_pad( nname.c_str(), (short*)in_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );

  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)dout_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
  else if(gp->in_data_type == DT_DFP16)
    check_physical_pad( nname.c_str(), (short*)dout_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
#endif
}

void ConvXSMM::dumpBuffer(TensorBuf* tBuf, void* wtemp)
{
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_filter, (void*)wtemp, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );
}
