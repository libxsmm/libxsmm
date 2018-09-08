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

FusedBNormXSMM::FusedBNormXSMM(FusedBNormImplParams* gp, int engine) : FusedBNormImpl(gp, engine)
{
  fusedbn_desc.N = gp->batch_size;
  fusedbn_desc.C = gp->nInput[0];
  fusedbn_desc.H = gp->iHeight;
  fusedbn_desc.W = gp->iWidth;
  fusedbn_desc.u = gp->stride_h;
  fusedbn_desc.v = gp->stride_w;
  fusedbn_desc.pad_h_in = gp->ipad_h;
  fusedbn_desc.pad_w_in = gp->ipad_w;
  fusedbn_desc.pad_h_out = gp->pad_h;
  fusedbn_desc.pad_w_out = gp->pad_w;
  fusedbn_desc.threads = gp->num_threads;
  fusedbn_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
  fusedbn_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
  fusedbn_desc.datatype_stats = LIBXSMM_DNN_DATATYPE_F32;
  fusedbn_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  fusedbn_desc.fuse_order = LIBXSMM_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU;
  fusedbn_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN;
  if(gp->relu)
    fusedbn_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_RELU;
  if(gp->eltwise)
    fusedbn_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_ELTWISE;
  if(gp->relu && gp->eltwise)
    fusedbn_desc.fuse_ops = LIBXSMM_DNN_FUSEDBN_OPS_BN_ELTWISE_RELU;

  libxsmm_handle = libxsmm_dnn_create_fusedbn( fusedbn_desc, &status );
  CHKERR_LIBXSMM_DNN( status );
}

void FusedBNormXSMM::forwardPropagate(vector<TensorBuf *> inpb, TensorBuf *gammapb, TensorBuf *betapb, float *gexpect, float *gstddev, TensorBuf *outpb, int tid)
{
  void *inp_r = inpb[0]->getBuffer();
  void *inp_l = gp->eltwise ? inpb[1]->getBuffer() : NULL;
  void *output = outpb->getBuffer();
  void *gamma = gammapb->getBuffer();
  void *beta = betapb->getBuffer();
  int nFM = gp->nInput[0];

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

  if(libxsmm_input == NULL && libxsmm_input_add == NULL && libxsmm_expectval == NULL && libxsmm_stddev == NULL
      && libxsmm_gamma == NULL && libxsmm_beta == NULL && libxsmm_output == NULL)
  {
    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, &status ); 
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_input  = libxsmm_dnn_link_tensor( libxsmm_layout, inp_r, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_input, LIBXSMM_DNN_REGULAR_INPUT ) );

    if(gp->eltwise)
    {
      libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT_ADD, &status ); 
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input_add = libxsmm_dnn_link_tensor( libxsmm_layout, inp_l, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_input_add, LIBXSMM_DNN_REGULAR_INPUT_ADD ) )
    }

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_CHANNEL_EXPECTVAL, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_expectval  = libxsmm_dnn_link_tensor( libxsmm_layout, bexpect, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_expectval, LIBXSMM_DNN_CHANNEL_EXPECTVAL ) );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_CHANNEL_STDDEV, &status ); 
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_stddev  = libxsmm_dnn_link_tensor( libxsmm_layout, bstddev, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_stddev, LIBXSMM_DNN_CHANNEL_STDDEV ) );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA, &status ); 
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_gamma  = libxsmm_dnn_link_tensor( libxsmm_layout, gamma, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_gamma, LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA ) );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout(libxsmm_handle, LIBXSMM_DNN_REGULAR_CHANNEL_BETA, &status); 
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_beta  = libxsmm_dnn_link_tensor( libxsmm_layout, beta, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_beta, LIBXSMM_DNN_REGULAR_CHANNEL_BETA ) );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, &status ); 
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_output  = libxsmm_dnn_link_tensor( libxsmm_layout, output, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_output, LIBXSMM_DNN_REGULAR_OUTPUT ) );

    /* let's allocate (if required) and bind scratch */
    if(scratch == NULL)
    {
      long long int mysize = libxsmm_dnn_fusedbn_get_scratch_size( libxsmm_handle, &status );
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
      long long int mysize = libxsmm_dnn_fusedbn_get_scratch_size( libxsmm_handle, &status );

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
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_scratch( libxsmm_handle, scratch ) );
    updated_scratch = true;
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
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
  }
}

void FusedBNormXSMM::backPropagate(vector<TensorBuf*> inpb, TensorBuf* outpb, TensorBuf *gammapb, TensorBuf *deloutpb, TensorBuf *delgammapb, TensorBuf *delbetapb, vector<TensorBuf*> delinpb, int tid)
{
  void *deloutput = deloutpb->getBuffer();
  void *delinp_r = delinpb[0]->getBuffer();
  void *delinp_l = gp->eltwise ? delinpb[1]->getBuffer() : NULL;
  void *delgamma = delgammapb->getBuffer();
  void *delbeta = delbetapb->getBuffer();

  __assume_aligned(delinp_r,64);
  __assume_aligned(deloutput,64);

  if(scratch != scratchp->getBuffer())
  {
    scratch = scratchp->getBuffer();
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_scratch( libxsmm_handle, scratch ) );
  }

  if(libxsmm_deloutput == NULL && libxsmm_delinput == NULL && libxsmm_delinput_add == NULL &&
      libxsmm_delgamma == NULL && libxsmm_delbeta == NULL)
  {
    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, &status ); 
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_deloutput = libxsmm_dnn_link_tensor( libxsmm_layout, deloutput, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_deloutput, LIBXSMM_DNN_GRADIENT_OUTPUT ) );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, &status ); 
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_delinput  = libxsmm_dnn_link_tensor( libxsmm_layout, delinp_r, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_delinput, LIBXSMM_DNN_GRADIENT_INPUT ) );

    if(gp->eltwise)
    {
      libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT_ADD, &status); 
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_delinput_add  = libxsmm_dnn_link_tensor( libxsmm_layout, delinp_l, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_delinput_add, LIBXSMM_DNN_GRADIENT_INPUT_ADD ) );
    }

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout(libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA, &status); 
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_delgamma  = libxsmm_dnn_link_tensor( libxsmm_layout, delgamma, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_delgamma, LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA ) );

    libxsmm_layout = libxsmm_dnn_fusedbn_create_tensor_datalayout(libxsmm_handle, LIBXSMM_DNN_GRADIENT_CHANNEL_BETA, &status); 
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_delbeta  = libxsmm_dnn_link_tensor( libxsmm_layout, delbeta, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_bind_tensor( libxsmm_handle, libxsmm_delbeta, LIBXSMM_DNN_GRADIENT_CHANNEL_BETA ) );
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
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fusedbn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
  }

  bpdone = true;
}

