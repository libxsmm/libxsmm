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
  fusedbn_desc_train.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
  fusedbn_desc_train.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
  fusedbn_desc_train.datatype_stats = LIBXSMM_DNN_DATATYPE_F32;
  fusedbn_desc_train.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  fusedbn_desc_train.fuse_order = LIBXSMM_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU;
  fusedbn_desc_train.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN;
  if(gp->relu)
    fusedbn_desc_train.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_RELU;
  if(gp->eltwise)
    fusedbn_desc_train.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_ELTWISE;
  if(gp->relu && gp->eltwise)
    fusedbn_desc_train.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_ELTWISE_RELU;

  libxsmm_handle_train = libxsmm_dnn_create_fusedbn( fusedbn_desc_train, &status );
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
  fusedbn_desc_test.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
  fusedbn_desc_test.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
  fusedbn_desc_test.datatype_stats = LIBXSMM_DNN_DATATYPE_F32;
  fusedbn_desc_test.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  fusedbn_desc_test.fuse_order = LIBXSMM_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU;
  fusedbn_desc_test.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE;
  if(gp->relu)
    fusedbn_desc_test.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_RELU;
  if(gp->eltwise)
    fusedbn_desc_test.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE;
  if(gp->relu && gp->eltwise)
    fusedbn_desc_test.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE_RELU;

  libxsmm_handle_test = libxsmm_dnn_create_fusedbn( fusedbn_desc_test, &status );
  CHKERR_LIBXSMM_DNN( status );
}

void FusedBNormXSMM::forwardPropagate(vector<TensorBuf *> inpb, TensorBuf *gammapb, TensorBuf *betapb, float *gexpect, float *gstddev, TensorBuf *outpb, int tid)
{
  void *inp_r = inpb[0]->getBuffer();
  void *inp_l = gp->eltwise ? inpb[1]->getBuffer() : NULL;
  void *output = outpb->getBuffer();
  void *gamma = gammapb->getBuffer();
  void *beta = betapb->getBuffer();

  int nImg = gp->batch_size;
  int nFM = gp->nInput[0];
  int nBfm = nFM/VLEN;
  int fh = gp->iHeight;
  int fw = gp->iWidth;

  if(bexpect == NULL)
  {
    bexpect = (void*)_mm_malloc(nFM*sizeof(float), 64);

#ifndef NDEBUG
    printf("%s allocated %lu bytes for mean\n",nname.c_str(), nFM*sizeof(float));
#endif
  }
  if(bstddev == NULL)
  {
    bstddev = (void*)_mm_malloc(nFM*sizeof(float), 64);

#ifndef NDEBUG
    printf("%s allocated %lu bytes for stdev\n",nname.c_str(), nFM*sizeof(float));
#endif
  }

  __assume_aligned(inp_r,64);
  if(inp_l)
    __assume_aligned(inp_l,64);

  __assume_aligned(gamma, 64);
  __assume_aligned(beta, 64);
  __assume_aligned(bexpect, 64);
  __assume_aligned(bstddev, 64);
  __assume_aligned(gexpect, 64);
  __assume_aligned(gstddev, 64);
  __assume_aligned(output,64);

  void *scratch = scratchp->getBuffer();

  if(libxsmm_input_train == NULL && libxsmm_input_add_train == NULL && libxsmm_expectval_train == NULL &&
      libxsmm_stddev_train == NULL && libxsmm_gamma_train == NULL && libxsmm_beta_train == NULL &&
      libxsmm_output_train == NULL)
  {
    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle_train, LIBXSMM_DNN_REGULAR_INPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_input_train  = libxsmm_dnn_link_tensor( libxsmm_layout, inp_r, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle_train, libxsmm_input_train, LIBXSMM_DNN_REGULAR_INPUT ) );

    if(gp->eltwise)
    {
      libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle_train, LIBXSMM_DNN_REGULAR_INPUT_ADD, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input_add_train = libxsmm_dnn_link_tensor( libxsmm_layout, inp_l, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle_train, libxsmm_input_add_train, LIBXSMM_DNN_REGULAR_INPUT_ADD ) )
    }

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle_train, LIBXSMM_DNN_CHANNEL_EXPECTVAL, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_expectval_train  = libxsmm_dnn_link_tensor( libxsmm_layout, bexpect, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle_train, libxsmm_expectval_train, LIBXSMM_DNN_CHANNEL_EXPECTVAL ) );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle_train, LIBXSMM_DNN_CHANNEL_STDDEV, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_stddev_train  = libxsmm_dnn_link_tensor( libxsmm_layout, bstddev, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle_train, libxsmm_stddev_train, LIBXSMM_DNN_CHANNEL_STDDEV ) );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle_train, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_gamma_train  = libxsmm_dnn_link_tensor( libxsmm_layout, gamma, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle_train, libxsmm_gamma_train, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA ) );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout(libxsmm_handle_train, LIBXSMM_DNN_REGULAR_CHANNEL_BETA, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_beta_train  = libxsmm_dnn_link_tensor( libxsmm_layout, beta, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle_train, libxsmm_beta_train, LIBXSMM_DNN_REGULAR_CHANNEL_BETA ) );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle_train, LIBXSMM_DNN_REGULAR_OUTPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_output_train  = libxsmm_dnn_link_tensor( libxsmm_layout, output, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle_train, libxsmm_output_train, LIBXSMM_DNN_REGULAR_OUTPUT ) );

    /* let's allocate (if required) and bind scratch */
    if(scratch == NULL)
    {
      long long int mysize = libxsmm_dnn_fusedbn_get_scratch_size( libxsmm_handle_train, &status );
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
      long long int mysize = libxsmm_dnn_fusedbn_get_scratch_size( libxsmm_handle_train, &status );

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

  if(libxsmm_input_test == NULL && libxsmm_input_add_test == NULL && libxsmm_expectval_test == NULL &&
      libxsmm_stddev_test == NULL && libxsmm_gamma_test == NULL && libxsmm_beta_test == NULL &&
      libxsmm_output_test == NULL)
  {
    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle_test, LIBXSMM_DNN_REGULAR_INPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_input_test  = libxsmm_dnn_link_tensor( libxsmm_layout, inp_r, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle_test, libxsmm_input_test, LIBXSMM_DNN_REGULAR_INPUT ) );

    if(gp->eltwise)
    {
      libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle_test, LIBXSMM_DNN_REGULAR_INPUT_ADD, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input_add_test = libxsmm_dnn_link_tensor( libxsmm_layout, inp_l, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle_test, libxsmm_input_add_test, LIBXSMM_DNN_REGULAR_INPUT_ADD ) )
    }

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle_test, LIBXSMM_DNN_CHANNEL_EXPECTVAL, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_expectval_test  = libxsmm_dnn_link_tensor( libxsmm_layout, bexpect, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle_test, libxsmm_expectval_test, LIBXSMM_DNN_CHANNEL_EXPECTVAL ) );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle_test, LIBXSMM_DNN_CHANNEL_STDDEV, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_stddev_test  = libxsmm_dnn_link_tensor( libxsmm_layout, bstddev, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle_test, libxsmm_stddev_test, LIBXSMM_DNN_CHANNEL_STDDEV ) );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle_test, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_gamma_test  = libxsmm_dnn_link_tensor( libxsmm_layout, gamma, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle_test, libxsmm_gamma_test, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA ) );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout(libxsmm_handle_test, LIBXSMM_DNN_REGULAR_CHANNEL_BETA, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_beta_test  = libxsmm_dnn_link_tensor( libxsmm_layout, beta, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle_test, libxsmm_beta_test, LIBXSMM_DNN_REGULAR_CHANNEL_BETA ) );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle_test, LIBXSMM_DNN_REGULAR_OUTPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_output_test  = libxsmm_dnn_link_tensor( libxsmm_layout, output, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle_test, libxsmm_output_test, LIBXSMM_DNN_REGULAR_OUTPUT ) );
  }

  if(!updated_scratch)
  {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_scratch( libxsmm_handle_train, scratch ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_scratch( libxsmm_handle_test, scratch ) );
    updated_scratch = true;
  }


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
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_execute_st( libxsmm_handle_train, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
    }

    float (* __restrict bmean)[VLEN] = (float (*)[VLEN])bexpect;
    float (* __restrict brstd)[VLEN] = (float (*)[VLEN])bstddev;
    float nhw_ratio = float(nImg*fh*fw)/float(nImg*fh*fw - 1);

#ifdef __AVX512F__
    __m512  vmmf       = _mm512_set1_ps(gp->mmf);
    __m512  vnhw_ratio = _mm512_set1_ps(nhw_ratio);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int b = 0; b < nBfm; ++b) {
      __m512 vbm = _mm512_load_ps(&bmean[b][0]);
      __m512 vbs = _mm512_load_ps(&brstd[b][0]);

      _mm512_store_ps( &(gexpect[b*VLEN]), _mm512_add_ps(_mm512_mul_ps(_mm512_load_ps( &(gexpect[b*VLEN]) ), vmmf), vbm));
      _mm512_store_ps( &(gstddev[b*VLEN]), _mm512_add_ps( _mm512_mul_ps( _mm512_load_ps( &(gstddev[b*VLEN]) ), vmmf), _mm512_mul_ps(vnhw_ratio, vbs)));
    }
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int b = 0; b < nBfm; ++b) {
#pragma omp simd
      for (int v = 0; v < 16; ++v) {
        gexpect[(b*16)+v] = gexpect[(b*16)+v] * gp->mmf + bmean[b][v];
        gstddev[(b*16)+v] = gstddev[(b*16)+v] * gp->mmf + nhw_ratio*brstd[b][v];
      }
    }
#endif

    scaling_factor_ = scaling_factor_ * gp->mmf + 1.;
  }
  else
  {
#if defined(_OPENMP)
#pragma omp parallel
#endif
    {
#pragma omp for
      for(int i=0; i < nFM; i++)
      {
        ((float*)bexpect)[i] = gexpect[i]/scaling_factor_;
        ((float*)bstddev)[i] = gstddev[i]/scaling_factor_;
      }

#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_execute_st( libxsmm_handle_test, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
    }
  }
}

void FusedBNormXSMM::backPropagate(vector<TensorBuf*> inpb, TensorBuf* outpb, TensorBuf *gammapb, TensorBuf *deloutpb, TensorBuf *delgammapb, TensorBuf *delbetapb, vector<TensorBuf*> delinpb, int tid)
{
  void *inp_r = inpb[0]->getBuffer();
  void *outp = outpb->getBuffer();
  void *deloutput = deloutpb->getBuffer();
  void *delinp_r = delinpb[0]->getBuffer();
  void *delinp_l = gp->eltwise ? delinpb[1]->getBuffer() : NULL;
  void *delgamma = delgammapb->getBuffer();
  void *delbeta = delbetapb->getBuffer();

  __assume_aligned(delinp_r,64);
  __assume_aligned(deloutput,64);

  int nImg  = gp->batch_size;
  int nFM = gp->nOutput;
  int nBfm = nFM/VLEN;
  int fh = gp->oHeight;
  int fw = gp->oWidth;
  int ph = gp->pad_h;
  int pw = gp->pad_w;
  int sh = gp->stride_h;
  int sw = gp->stride_w;
  int iph = gp->ipad_h;
  int ipw = gp->ipad_w;
  int fhs = fh/sh;
  int fws = fw/sw;
  int fhi = fh + 2*iph;
  int fwi = fw + 2*ipw;

  float (* __restrict del_input_r)[nBfm][fhi][fwi][VLEN] = (float (*)[*][*][*][VLEN])delinp_r;
  float (* __restrict del_input_l)[nBfm][fhi][fwi][VLEN] = gp->eltwise ? (float (*)[*][*][*][VLEN])delinp_l : NULL;

  /* zero the rims in case of physical padding */
  /* @TODO, we need to do the same thing with del_input_l?! */
  if (iph > 0 || iph > 0) {
#pragma omp parallel for
    for (int img = 0; img < nImg; img++) {
      for (int fm = 0; fm < nBfm; fm++) {
        for (int w = 0; w < fwi; w++) {
          for (int ph = 0; ph < iph; ph++) {
#ifdef __AVX512F__
            _mm512_stream_ps( &(del_input_r[img][fm][ph      ][w][0]), _mm512_setzero_ps() );
            _mm512_stream_ps( &(del_input_r[img][fm][fhi-1-ph][w][0]), _mm512_setzero_ps() );
#else
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
            for(int v=0; v < VLEN; v++) {
              del_input_r[img][fm][ph][w][v] = 0.0f;
              del_input_r[img][fm][fhi-1-ph][w][v] = 0.0f;
            }
#endif
          }
        }
        for (int h = iph; h < fh+iph; h++) {
          for (int pw = 0; pw < ipw; pw++) {
#ifdef __AVX512F__
            _mm512_stream_ps( &(del_input_r[img][fm][h][pw      ][0]), _mm512_setzero_ps() );
            _mm512_stream_ps( &(del_input_r[img][fm][h][fwi-1-pw][0]), _mm512_setzero_ps() );
#else
#pragma omp simd
#pragma vector aligned
#ifdef USE_NTS_BN
#pragma vector nontemporal
#endif
            for(int v=0; v < VLEN; v++) {
              del_input_r[img][fm][h][pw][v] = 0.0f;
              del_input_r[img][fm][h][fwi-1-pw][v] = 0.0f;
            }
#endif
          }
        }
      }
    }
  }

  /* Perform physical padding tests */
#ifndef NDEBUG
  if ( (ph > 0 || pw > 0) && (iph > 0 || ipw > 0) ) {
    printf("node %s: batchnorm backward input and output is padded which cannot be :-(\n", nname.c_str());
  }

  /* check rims */
  check_physical_pad( nname.c_str(), (float*)delinp_r, nImg, nBfm, fh,  fw,  VLEN, iph, ipw );
  check_physical_pad( nname.c_str(),    (float*)inp_r, nImg, nBfm, fh,  fw,  VLEN, iph, ipw );
  check_physical_pad( nname.c_str(),  (float*)deloutp, nImg, nBfm, fhs, fws, VLEN, ph,  pw );
  check_physical_pad( nname.c_str(),     (float*)outp, nImg, nBfm, fhs, fws, VLEN, ph,  pw );
#endif

  if(scratch != scratchp->getBuffer())
  {
    scratch = scratchp->getBuffer();
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_scratch( libxsmm_handle_train, scratch ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_scratch( libxsmm_handle_test, scratch ) );
  }

  if(libxsmm_deloutput == NULL && libxsmm_delinput == NULL && libxsmm_delinput_add == NULL &&
      libxsmm_delgamma == NULL && libxsmm_delbeta == NULL)
  {
    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle_train, LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_deloutput = libxsmm_dnn_link_tensor( libxsmm_layout, deloutput, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle_train, libxsmm_deloutput, LIBXSMM_DNN_GRADIENT_OUTPUT ) );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle_train, LIBXSMM_DNN_GRADIENT_INPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_delinput  = libxsmm_dnn_link_tensor( libxsmm_layout, delinp_r, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle_train, libxsmm_delinput, LIBXSMM_DNN_GRADIENT_INPUT ) );

    if(gp->eltwise)
    {
      libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout(libxsmm_handle_train, LIBXSMM_DNN_GRADIENT_INPUT_ADD, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_delinput_add  = libxsmm_dnn_link_tensor( libxsmm_layout, delinp_l, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle_train, libxsmm_delinput_add, LIBXSMM_DNN_GRADIENT_INPUT_ADD ) );
    }

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout(libxsmm_handle_train, LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_delgamma  = libxsmm_dnn_link_tensor( libxsmm_layout, delgamma, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle_train, libxsmm_delgamma, LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA ) );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout(libxsmm_handle_train, LIBXSMM_DNN_GRADIENT_CHANNEL_BETA, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_delbeta  = libxsmm_dnn_link_tensor( libxsmm_layout, delbeta, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle_train, libxsmm_delbeta, LIBXSMM_DNN_GRADIENT_CHANNEL_BETA ) );
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
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_execute_st( libxsmm_handle_train, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
  }

  /* Perform physical padding tests */
#ifndef NDEBUG
  if ( (ph > 0 || pw > 0) && (iph > 0 || ipw > 0) ) {
    printf("node %s: batchnorm backward input and output is padded which cannot be :-(\n", nname.c_str());
  }

  /* check rims */
  check_physical_pad( nname.c_str(), (float*)delinp_r, nImg, nBfm, fh,  fw,  VLEN, iph, ipw );
  check_physical_pad( nname.c_str(),    (float*)inp_r, nImg, nBfm, fh,  fw,  VLEN, iph, ipw );
  check_physical_pad( nname.c_str(),  (float*)deloutp, nImg, nBfm, fhs, fws, VLEN, ph,  pw );
  check_physical_pad( nname.c_str(),     (float*)outp, nImg, nBfm, fhs, fws, VLEN, ph,  pw );
#endif
  bpdone = true;
}

