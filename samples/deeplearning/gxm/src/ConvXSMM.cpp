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

#include "ConvXSMM.hpp"

using namespace std;

ConvXSMM::ConvXSMM(ConvImplParams* gp, int engine) : ConvImpl(gp, engine)
{
  conv_desc.N = gp->batch_size/gp->num_numa_nodes;
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

  conv_desc.threads = gp->num_threads/gp->num_numa_nodes;
  conv_desc.algo = LIBXSMM_DNN_CONV_ALGO_DIRECT;
  conv_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  conv_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;
  if(gp->out_data_type == DT_FLOAT)
    conv_desc.options = LIBXSMM_DNN_CONV_OPTION_OVERWRITE;
  else if(gp->out_data_type == DT_BF16)
    conv_desc.options = LIBXSMM_DNN_CONV_OPTION_OVERWRITE;

  if(gp->bias_term)
    printf("GxM Error: Fusion for Bias into convolution is not available!\n");
  if(gp->relu)
    printf("GxM Error: Fusion for ReLU into convolution is not available!\n");
  if(gp->bias_term && gp->relu)
    printf("GxM Error: Fusion for ReLU ind Bias into convolution is not available!\n");
  if(gp->compute_stats)
    printf("GxM Error: Fusion for Batch stats into convolution is not available!\n");
  if(gp->compute_stats && gp->bwd_relu)
    printf("GxM Error: Fusion for Batch stats and ReLU into convolution is not available!\n");

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

  for(int i=0; i<gp->num_numa_nodes; i++)
  {
    libxsmm_handle[i] = libxsmm_dnn_create_conv_layer( conv_desc, &status );
    CHKERR_LIBXSMM_DNN( status );
  }

  top_layout_type = LIBXSMM_CUSTOM_LAYOUT;
  top_layout = libxsmm_handle;
  gbot_layout_type = LIBXSMM_CUSTOM_LAYOUT;
  gbot_layout = libxsmm_handle;
}

void ConvXSMM::forwardPropagate(TensorBuf *inp, TensorBuf *weightp, TensorBuf *hweightp, TensorBuf *biasp, TensorBuf *outp, int tid)
{
  int nIFM = gp->nInput;
  int nOFM = gp->nOutput;
  int nBIfm = nIFM/VLEN;
  int nBOfm = nOFM/VLEN;
  int ifh = gp->iHeight;
  int ifw = gp->iWidth;
  int ofh = gp->oHeight;
  int ofw = gp->oWidth;
  int iph = gp->ipad_h;
  int ipw = gp->ipad_w;
  int oph = gp->opad_h;
  int opw = gp->opad_w;
  int ifhp = ifh + 2*iph;
  int ifwp = ifw + 2*ipw;
  int ofhp = ofh + 2*oph;
  int ofwp = ofw + 2*opw;
  int kh = gp->kh;
  int kw = gp->kw;

  // Conv input. LPBuffer is non-NULL if data layer output is BF16
  int imoff = conv_desc.N * conv_desc.C * ifhp * ifwp;
  if(gp->in_data_type == DT_BF16)
  {
    if(inp->getLPBuffer() != NULL)
      in_ptr[0] = inp->getLPBuffer();
    else
      in_ptr[0] = inp->getBuffer();
    imoff = imoff * sizeof(libxsmm_bfloat16);
  }
  else if(gp->in_data_type == DT_FLOAT)
  {
    in_ptr[0] = inp->getBuffer();
    imoff = imoff * sizeof(float);
  }

  for(int n=1; n<gp->num_numa_nodes; n++)
    in_ptr[n] = in_ptr[n-1] + imoff;

  // Conv Weight
  void **lptrptr = weightp->getLPBufferPtr();
  void **ptrptr = weightp->getBufferPtr();
  int offset = weightp->getOffset();

  if(gp->in_data_type == DT_BF16)
  {
    if(lptrptr != NULL)
      for(int n=0; n<gp->num_numa_nodes; n++)
      {
        wt_ptr[n] = lptrptr[n] + offset*sizeof(libxsmm_bfloat16);
        f32_wt_ptr[n] = ptrptr[n] + offset*sizeof(float);
      }
  }
  else if(gp->in_data_type == DT_FLOAT)
    for(int n=0; n<gp->num_numa_nodes; n++)
      wt_ptr[n] = ptrptr[n] + offset*sizeof(float);

  void *wt_prv_ptr = NULL;

  // Conv weight history
  if(hweightp != NULL)
    hwt_ptr = hweightp->getBuffer();
  else
    hwt_ptr = NULL;

  // Conv output
  out_ptr[0] = outp->getBuffer();
  imoff = conv_desc.N * conv_desc.K * ofhp * ofwp;
  if(gp->out_data_type == DT_FLOAT)
    imoff = imoff * sizeof(float);
  else if(gp->out_data_type == DT_BF16)
    imoff = imoff * sizeof(libxsmm_bfloat16);

  for(int n=1; n<gp->num_numa_nodes; n++)
    out_ptr[n] = out_ptr[n-1] + imoff;

  void **sptrptr = scratchp->getBufferPtr();

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_input[n] == NULL && libxsmm_output[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle[n], LIBXSMM_DNN_REGULAR_INPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input[n] = libxsmm_dnn_link_tensor( libxsmm_layout, in_ptr[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle[n], libxsmm_input[n], LIBXSMM_DNN_REGULAR_INPUT ) );

      // Conv Output
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle[n], LIBXSMM_DNN_REGULAR_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN(      status );
      libxsmm_output[n] = libxsmm_dnn_link_tensor( libxsmm_layout, out_ptr[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle[n], libxsmm_output[n], LIBXSMM_DNN_REGULAR_OUTPUT ) );
    }
  }

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_filter[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle[n], LIBXSMM_DNN_REGULAR_FILTER, &status );
      CHKERR_LIBXSMM_DNN( status );

      int welem = gp->nInput * gp->nOutput * gp->kw * gp->kh;
      if(gp->in_data_type == DT_FLOAT)
      {
        libxsmm_filter[n] = libxsmm_dnn_link_tensor( libxsmm_layout, wt_ptr[n], &status );
        CHKERR_LIBXSMM_DNN( status );

        if(gp->node_name == "conv1" || gp->node_name == "convbn1")
        {
          libxsmm_dnn_tensor *tensor = libxsmm_filter[n];
          libxsmm_dnn_err_t status;
          libxsmm_dnn_tensor_datalayout *mylayout = libxsmm_dnn_get_tensor_datalayout (tensor, &status);
          int i1, i2, i3, i4, i5, i6;
          int bofm = 0;
          int bifm = 0;
          int S = 0;
          int R = 0;
          int ifmb = 0;
          int ofmb = 0;

          assert( mylayout->num_dims == 6 );

          bofm = mylayout->dim_size[0];
          bifm = mylayout->dim_size[1];
          S = mylayout->dim_size[2];
          R = mylayout->dim_size[3];
          ifmb = mylayout->dim_size[4];
          ofmb = mylayout->dim_size[5];
          LIBXSMM_VLA_DECL(6, float, handle_data, (float*)wt_ptr[n], ifmb, R, S, bifm, bofm);

          for (i1 = 0; i1 < ofmb; ++i1) {
            for (i2 = 0; i2 < ifmb; ++i2) {
              for (i3 = 0; i3 < R; ++i3) {
                for (i4 = 0; i4 < S; ++i4) {
                  for (i5 = 0; i5 < bifm; ++i5) {
                    for (i6 = 0; i6 < bofm; ++i6) {
                      /* set 4th input channel to 0 */
                      if ( (i6 == 1) && (i5 == 1) ) {
                        LIBXSMM_VLA_ACCESS(6, handle_data, i1, i2, i3, i4, i5, i6, ifmb, R, S, bifm, bofm) = (float)0;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      else if(gp->in_data_type == DT_BF16)
      {

        libxsmm_filter[n] = libxsmm_dnn_link_tensor( libxsmm_layout, wt_ptr[n], &status );
        CHKERR_LIBXSMM_DNN( status );

        if(gp->node_name == "conv1" || gp->node_name == "convbn1")
        {
          libxsmm_dnn_tensor *tensor = libxsmm_filter[n];
          libxsmm_dnn_err_t status;
          libxsmm_bfloat16* mydata = (libxsmm_bfloat16*)libxsmm_dnn_get_tensor_data_ptr(tensor, &status);
          libxsmm_dnn_tensor_datalayout * mylayout = libxsmm_dnn_get_tensor_datalayout (tensor, &status);
          int i1, i2, i3, i4, i5, i6, i7;
          int lpb = 0;
          int bofm = 0;
          int bifm = 0;
          int S = 0;
          int R = 0;
          int ifmb = 0;
          int ofmb = 0;

          /* check for VNNI weights */
          assert( mylayout->num_dims == 7 );

          lpb = mylayout->dim_size[0];
          bofm = mylayout->dim_size[1];
          bifm = mylayout->dim_size[2];
          S = mylayout->dim_size[3];
          R = mylayout->dim_size[4];
          ifmb = mylayout->dim_size[5];
          ofmb = mylayout->dim_size[6];

          LIBXSMM_VLA_DECL(7, libxsmm_bfloat16, handle_data_1, mydata, ifmb, R, S, bifm, bofm, lpb);

          for (i1 = 0; i1 < ofmb; ++i1) {
            for (i2 = 0; i2 < ifmb; ++i2) {
              for (i3 = 0; i3 < R; ++i3) {
                for (i4 = 0; i4 < S; ++i4) {
                  for (i5 = 0; i5 < bifm; ++i5) {
                    for (i6 = 0; i6 < bofm; ++i6) {
                      for (i7 = 0; i7 < lpb; ++i7) {
                        /* set 4th input channel to 0 */
                        if ( (i7 == 1) && (i5 == 1) ) {
                          LIBXSMM_VLA_ACCESS(7, handle_data_1, i1, i2, i3, i4, i5, i6, i7, ifmb, R, S, bifm, bofm, lpb) = (libxsmm_bfloat16)0;
                        }
                      }
                    }
                  }
                }
              }
            }
          }

          LIBXSMM_VLA_DECL(7, float, handle_data_2, (float*)f32_wt_ptr[n], ifmb, R, S, bifm, bofm, lpb);

          for (i1 = 0; i1 < ofmb; ++i1) {
            for (i2 = 0; i2 < ifmb; ++i2) {
              for (i3 = 0; i3 < R; ++i3) {
                for (i4 = 0; i4 < S; ++i4) {
                  for (i5 = 0; i5 < bifm; ++i5) {
                    for (i6 = 0; i6 < bofm; ++i6) {
                      for (i7 = 0; i7 < lpb; ++i7) {
                        /* set 4th input channel to 0 */
                        if ( (i7 == 1) && (i5 == 1) ) {
                          LIBXSMM_VLA_ACCESS(7, handle_data_2, i1, i2, i3, i4, i5, i6, i7, ifmb, R, S, bifm, bofm, lpb) = (float)0;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle[n], libxsmm_filter[n], LIBXSMM_DNN_REGULAR_FILTER ) );
    }
  }

  /* let's allocate (if required) and bind scratch */
  if(sptrptr == NULL)
  {
    sptrptr = (void**)libxsmm_aligned_malloc(gp->num_numa_nodes*sizeof(void*), 2097152);
    scratchp->setBufferPtr(sptrptr);
  }

  if(prev_scratch_size == 0)
    prev_scratch_size = scratchp->getBufferSize();

  if(!updated_scratch_fwd || prev_scratch_size != scratchp->getBufferSize())
  {
    int max_size = 0;
    for(int n=0; n<gp->num_numa_nodes; n++)
    {
      if(sptrptr[n] == NULL)
      {
        int mysize = libxsmm_dnn_get_scratch_size( libxsmm_handle[n], LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
        CHKERR_LIBXSMM_DNN( status );
        sptrptr[n] = (void*)libxsmm_aligned_malloc(mysize, 2097152);
        max_size = mysize;

#ifdef USE_MLSL
        if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
          printf("%s allocated %d bytes for scratch @ %p\n",nname.c_str(), mysize, sptrptr[n]);
      }
      else
      {
        int ssize = scratchp->getBufferSize();
        int mysize = libxsmm_dnn_get_scratch_size( libxsmm_handle[n], LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );

        CHKERR_LIBXSMM_DNN( status );

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

    for(int n=0; n<gp->num_numa_nodes; n++)
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle[n], LIBXSMM_DNN_COMPUTE_KIND_ALL, sptrptr[n] ) );
    updated_scratch_fwd = true;
    prev_scratch_size = scratchp->getBufferSize();
  }

#if 0
#ifndef NDEBUG
  /* check physical padding */
  if ( (gp->ipad_h > 0 || gp->ipad_w > 0) && (gp->opad_h > 0 || gp->opad_w > 0) ) {
  } else if ( (gp->ipad_h == 0 || gp->ipad_w == 0) && (gp->opad_h == 0 || gp->opad_w == 0) ) {
  } else {
    printf("node %s: conv xsmm forward is partially padded which cannot be :-(\n", nname.c_str());
  }

  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)in_ptr[0], conv_desc.N, nBIfm, ifh, ifw, VLEN, iph, ipw);
  else if(gp->in_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)in_ptr[0], conv_desc.N, nBIfm, ifh, ifw, VLEN, iph, ipw);

  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)out_ptr[0], conv_desc.N, nBOfm, ofh, ofw, VLEN, oph, opw);
  else if(gp->in_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)out_ptr[0], conv_desc.N, nBOfm, ofh, ofw, VLEN, oph, opw);
#endif
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
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle[n], LIBXSMM_DNN_COMPUTE_KIND_FWD, n*ntps, tid) );
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
      printf("%s XSMM-CONV-FP mb%dic%dih%doc%doh%dkh%dn time = %g ms, GFLOPS = %.1f\n",gp->node_name.c_str(),gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,fp_time*1000.0, gf/fp_time/1e9);
    else if(gp->stride_h == 2)
      printf("%s XSMM-CONV-FP mb%dic%dih%doc%doh%dkh%dsh%dn time = %g ms, GFLOPS = %.1f\n",gp->node_name.c_str(),gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,gp->stride_h,fp_time*1000.0, gf/fp_time/1e9);
    else if(gp->pad_h == 1)
      printf("%s XSMM-CONV-FP mb%dic%dih%doc%doh%dkh%dph%dn time = %g ms, GFLOPS = %.1f\n",gp->node_name.c_str(),gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,gp->pad_h,fp_time*1000.0, gf/fp_time/1e9);
  }
#endif

  top_layout_type = LIBXSMM_CUSTOM_LAYOUT;
  outp->setLayoutType(top_layout_type);
  outp->setLayout(libxsmm_handle);

#if 0
#ifndef NDEBUG
  /* check physical padding */
  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)in_ptr[0], conv_desc.N, nBIfm, ifh, ifw, VLEN, iph, ipw);
  else if(gp->in_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)in_ptr[0], conv_desc.N, nBIfm, ifh, ifw, VLEN, iph, ipw);

  if(gp->out_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)out_ptr[0], conv_desc.N, nBOfm, ofh, ofw, VLEN, oph, opw);
  else if(gp->out_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)out_ptr[0], conv_desc.N, nBOfm, ofh, ofw, VLEN, oph, opw);
#endif
#endif
}

void ConvXSMM::backPropagate(TensorBuf* inp, TensorBuf* weightp, TensorBuf *deloutp, TensorBuf* delinp, int tid)
{
  int nIFM = gp->nInput;
  int nOFM = gp->nOutput;
  int nBIfm = nIFM/VLEN;
  int nBOfm = nOFM/VLEN;
  int ifh = gp->iHeight;
  int ifw = gp->iWidth;
  int ofh = gp->oHeight;
  int ofw = gp->oWidth;
  int iph = gp->ipad_h;
  int ipw = gp->ipad_w;
  int oph = gp->opad_h;
  int opw = gp->opad_w;
  int ifhp = ifh + 2*iph;
  int ifwp = ifw + 2*ipw;
  int ofhp = ofh + 2*oph;
  int ofwp = ofw + 2*opw;

  int imoff = conv_desc.N * conv_desc.K * ofhp * ofwp;
  if(gp->out_data_type == DT_FLOAT)
    imoff = imoff * sizeof(float);
  else if(gp->out_data_type == DT_BF16)
    imoff = imoff * sizeof(libxsmm_bfloat16);

  dout_ptr[0] = deloutp->getBuffer();
  for(int n=1; n<gp->num_numa_nodes; n++)
    dout_ptr[n] = dout_ptr[n-1] + imoff;

  imoff = conv_desc.N * conv_desc.C * ifhp * ifwp;
  if(gp->in_data_type == DT_FLOAT)
    imoff = imoff * sizeof(float);
  else if(gp->in_data_type == DT_BF16)
    imoff = imoff * sizeof(libxsmm_bfloat16);
  din_ptr[0] = delinp->getBuffer();
  for(int n=1; n<gp->num_numa_nodes; n++)
    din_ptr[n] = din_ptr[n-1] + imoff;

  void **sptrptr = scratchp->getBufferPtr();

  if(!updated_scratch_bwd)
  {
    for(int n=0; n<gp->num_numa_nodes; n++)
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle[n], LIBXSMM_DNN_COMPUTE_KIND_ALL, sptrptr[n] ) );
    updated_scratch_bwd = true;
  }

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_delinput[n] == NULL && libxsmm_deloutput[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle[n], LIBXSMM_DNN_GRADIENT_INPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_delinput[n] = libxsmm_dnn_link_tensor(libxsmm_layout, din_ptr[n], &status );
      CHKERR_LIBXSMM_DNN(status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle[n], libxsmm_delinput[n], LIBXSMM_DNN_GRADIENT_INPUT ) );

      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle[n], LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN(status );
      libxsmm_deloutput[n] = libxsmm_dnn_link_tensor( libxsmm_layout, dout_ptr[n], &status );
      CHKERR_LIBXSMM_DNN(status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle[n], libxsmm_deloutput[n], LIBXSMM_DNN_GRADIENT_OUTPUT ) );
    }
  }

#if 0
#ifndef NDEBUG
  /* check physical padding */
  if ( (gp->ipad_h > 0 || gp->ipad_w > 0) && (gp->opad_h > 0 || gp->opad_w > 0) ) {
  } else if ( (gp->ipad_h == 0 || gp->ipad_w == 0) && (gp->opad_h == 0 || gp->opad_w == 0) ) {
  } else {
    printf("node %s: conv xsmm backward is partially padded which cannot be :-(\n", nname.c_str());
  }
  if(gp->out_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)din_ptr[0], conv_desc.N, nBIfm, ifh, ifw, VLEN, iph, ipw);
  else if(gp->out_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)din_ptr[0], conv_desc.N, nBIfm, ifh, ifw, VLEN, iph, ipw);

  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)dout_ptr[0], conv_desc.N, nBOfm, ofh, ofw, VLEN, oph, opw);
  else if(gp->in_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)dout_ptr[0], conv_desc.N, nBOfm, ofh, ofw, VLEN, oph, opw);
#endif
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
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle[n], LIBXSMM_DNN_COMPUTE_KIND_BWD, n*ntps, tid ) );
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
      printf("%s XSMM-CONV-BP mb%dic%dih%doc%doh%dkh%dn time = %g ms, GFLOPS = %.1f\n",gp->node_name.c_str(),gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,bp_time*1000.0, gf/bp_time/1e9);
    else if(gp->stride_h == 2)
      printf("%s XSMM-CONV-BP mb%dic%dih%doc%doh%dkh%dsh%dn time = %g ms, GFLOPS = %.1f\n",gp->node_name.c_str(),gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,gp->stride_h,bp_time*1000.0, gf/bp_time/1e9);
    else if(gp->pad_h == 1)
      printf("%s XSMM-CONV-BP mb%dic%dih%doc%doh%dkh%dph%dn time = %g ms, GFLOPS = %.1f\n",gp->node_name.c_str(),gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,gp->pad_h,bp_time*1000.0, gf/bp_time/1e9);
  }
#endif

  gbot_layout_type = LIBXSMM_CUSTOM_LAYOUT;
  delinp->setLayoutType(gbot_layout_type);
  delinp->setLayout(libxsmm_handle);

#if 0
#ifndef NDEBUG
  /* check physical padding */
  if(gp->out_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)din_ptr[0], conv_desc.N, nBIfm, ifh, ifw, VLEN, iph, ipw);
  else if(gp->out_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)din_ptr[0], conv_desc.N, nBIfm, ifh, ifw, VLEN, iph, ipw);

  if(gp->in_data_type == DT_FLOAT)
    check_physical_pad( nname.c_str(), (float*)dout_ptr[0], conv_desc.N, nBOfm, ofh, ofw, VLEN, oph, opw);
  else if(gp->in_data_type == DT_BF16)
    check_physical_pad( nname.c_str(), (libxsmm_bfloat16*)dout_ptr[0], conv_desc.N, nBOfm, ofh, ofw, VLEN, oph, opw);
#endif
#endif
}

void ConvXSMM::weightUpdate(TensorBuf *inp, TensorBuf *deloutp, TensorBuf* delweightp, TensorBuf* delbiasp, int tid)
{
  int ifm = gp->nInput;
  int ofm = gp->nOutput;
  int ofh = gp->oHeight;
  int ofw = gp->oWidth;
  int oph = gp->opad_h;
  int opw = gp->opad_w;
  int ofhp = ofh + 2*oph;
  int ofwp = ofw + 2*opw;
  int kh = gp->kh;
  int kw = gp->kw;

  void *dwt_ptr[NUM_NUMA_NODES];
  void **ptrptr;

  if(gp->in_data_type == DT_BF16)
  {
#ifdef BF16_MLSL
    ptrptr = delweightp->getBufferPtr();
#else
    ptrptr = delweightp->getLPBufferPtr();
#endif
  }
  else
    ptrptr = delweightp->getBufferPtr();

  int offset = delweightp->getOffset();

  if(gp->in_data_type == DT_FLOAT)
    offset = offset*sizeof(float);
  else if(gp->in_data_type == DT_BF16)
    offset = offset*sizeof(libxsmm_bfloat16);

  for(int n=0; n<gp->num_numa_nodes; n++)
    dwt_ptr[n] = ptrptr[n] + offset;

  int imoff = conv_desc.N * conv_desc.K * ofhp * ofwp;
  if(gp->out_data_type == DT_FLOAT)
    imoff = imoff * sizeof(float);
  else if(gp->out_data_type == DT_BF16)
    imoff = imoff * sizeof(libxsmm_bfloat16);
  dout_ptr[0] = deloutp->getBuffer();
  for(int n=1; n<gp->num_numa_nodes; n++)
    dout_ptr[n] = dout_ptr[n-1] + imoff;

  void **sptrptr = scratchp->getBufferPtr();
  if(!updated_scratch_upd)
  {
    for(int n=0; n<gp->num_numa_nodes; n++)
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle[n], LIBXSMM_DNN_COMPUTE_KIND_ALL, sptrptr[n] ) );
    updated_scratch_upd = true;
  }

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_delfilter[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle[n], LIBXSMM_DNN_GRADIENT_FILTER, &status );
      CHKERR_LIBXSMM_DNN(status );
      libxsmm_delfilter[n] = libxsmm_dnn_link_tensor( libxsmm_layout, dwt_ptr[n], &status );
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle[n], libxsmm_delfilter[n], LIBXSMM_DNN_GRADIENT_FILTER ) );
    }
  }

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_deloutput[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle[n], LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN(      status );
      libxsmm_deloutput[n] = libxsmm_dnn_link_tensor(libxsmm_layout, dout_ptr[n], &status );
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_bind_tensor( libxsmm_handle[n], libxsmm_deloutput[n], LIBXSMM_DNN_GRADIENT_OUTPUT ) );
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

    int ntps = gp->num_threads/gp->num_numa_nodes;
    int n = tid/ntps;
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle[n], LIBXSMM_DNN_COMPUTE_KIND_UPD, n*ntps, tid ) );

#ifdef USE_MLSL
#pragma omp barrier

    if(gp->num_numa_nodes > 1)
    {
      if(gp->in_data_type == DT_FLOAT)
      {
        int jobs = ofm * ifm * kh * kw;
        int jn = jobs/gp->num_numa_nodes;
        int jnv = jn/VLEN;
        int jpt = (jnv % ntps == 0) ? (jnv/ntps)*VLEN : ((jnv/ntps)+1)*VLEN;
        int ltid = tid - n*ntps;
        int tb = (ltid * jpt < jn) ? ltid*jpt : jn;
        int te = ((ltid+1)*jpt < jn) ? (ltid+1)*jpt : jn;

        float *wgp = (float*)dwt_ptr[n]+n*jn;

        for(int nn=0; nn<gp->num_numa_nodes; nn++)
        {
          if(n == nn) continue;

          float *rgp = (float*)dwt_ptr[nn]+n*jn;

#pragma omp simd
          for(int i=tb; i<te; i++)
            wgp[i] += rgp[i];
        }

#pragma omp barrier

        for(int nn=0; nn<gp->num_numa_nodes; nn++)
        {
          if(n == nn) continue;
          float *wgp = (float*)dwt_ptr[n]+nn*jn;
          float *rgp = (float*)dwt_ptr[nn]+nn*jn;

#pragma vector nontemporal
#pragma omp simd
          for(int i=tb; i<te; i++)
            wgp[i] = rgp[i];
        }
      }
      else if(gp->in_data_type == DT_BF16)
      {
        if(n == 0)
        {
          int jobs = ofm * ifm * kh * kw;
          assert(jobs % VLEN == 0);
          int jv = jobs/VLEN;
          int rem = jv % ntps;
          int jpt = (rem == 0) ? (jv/ntps)*VLEN : ((jv-rem)/ntps)*VLEN;
          int tb = (tid * jpt < jobs) ? tid*jpt : jobs;
          int te = ((tid+1)*jpt < jobs) ? (tid+1)*jpt : jobs;

          libxsmm_bfloat16 *my_ptr = (libxsmm_bfloat16*)dwt_ptr[n];

          for(int nn=1; nn<gp->num_numa_nodes; nn++)
          {
            libxsmm_bfloat16 *rem_ptr = (libxsmm_bfloat16*)dwt_ptr[nn];

            for(int i=tb; i<te; i+=VLEN)
            {
              __m512  vfp32_l  = gxm_bfp16_to_fp32_avx512f(_mm256_loadu_si256( (const __m256i*)(my_ptr + i)));
              __m512  vfp32_r  = gxm_bfp16_to_fp32_avx512f(_mm256_loadu_si256( (const __m256i*)(rem_ptr + i)));
              __m512  vfp32 = _mm512_add_ps(vfp32_l, vfp32_r);
              __m512  vfp32rne = gxm_fp32_to_bfp16_rne_adjustment_avx512f(vfp32);
              __m256i vbfp16 = gxm_fp32_to_bfp16_truncate_avx512f(vfp32rne);
              _mm256_storeu_si256( (__m256i*)(my_ptr + i), vbfp16);
            }

            //Remainder processing
            if(tid == 0)
            {
              if(rem > 0)
              {
                for(int i=ntps*jpt; i<jobs; i+=VLEN)
                {
                  __m512  vfp32_l  = gxm_bfp16_to_fp32_avx512f(_mm256_loadu_si256( (const __m256i*)(my_ptr + i)));
                  __m512  vfp32_r  = gxm_bfp16_to_fp32_avx512f(_mm256_loadu_si256( (const __m256i*)(rem_ptr + i)));
                  __m512  vfp32 = _mm512_add_ps(vfp32_l, vfp32_r);
                  __m512  vfp32rne = gxm_fp32_to_bfp16_rne_adjustment_avx512f(vfp32);
                  __m256i vbfp16 = gxm_fp32_to_bfp16_truncate_avx512f(vfp32rne);
                  _mm256_storeu_si256( (__m256i*)(my_ptr + i), vbfp16);
                }
              }
            }
          }
        }
      }
    }
#endif
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
      printf("%s XSMM-CONV-WU mb%dic%dih%doc%doh%dkh%dn time = %g ms, GFLOPS = %.1f\n",gp->node_name.c_str(),gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,wu_time*1000.0, gf/wu_time/1e9);
    else if(gp->stride_h == 2)
      printf("%s XSMM-CONV-WU mb%dic%dih%doc%doh%dkh%dsh%dn time = %g ms, GFLOPS = %.1f\n",gp->node_name.c_str(),gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,gp->stride_h,wu_time*1000.0, gf/wu_time/1e9);
    else if(gp->pad_h == 1)
      printf("%s XSMM-CONV-WU mb%dic%dih%doc%doh%dkh%dph%dn time = %g ms, GFLOPS = %.1f\n",gp->node_name.c_str(),gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,gp->pad_h,wu_time*1000.0, gf/wu_time/1e9);
  }
#endif

}

void ConvXSMM::dumpBuffer(TensorBuf* tBuf, void* wtemp)
{
  int buftype = tBuf->getBufferType();

  if(buftype == DATA)
  {
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_copyout_tensor(libxsmm_checkpoint_filter, wtemp, LIBXSMM_DNN_TENSOR_FORMAT_KCRS));
  }
  else if(buftype == HISTORY)
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_copyout_tensor(libxsmm_checkpoint_history_filter, wtemp, LIBXSMM_DNN_TENSOR_FORMAT_KCRS));
}
