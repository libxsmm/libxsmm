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
/* Sasikanth Avancha, Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/


#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "PoolingXSMM.hpp"

#define VLEN 16

PoolXSMM::PoolXSMM(PoolImplParams *gp, int engine) : PoolImpl(gp, engine)
{
  pooling_desc.N = gp->batch_size;
  pooling_desc.C = gp->nInput;
  pooling_desc.H = gp->iHeight;
  pooling_desc.W = gp->iWidth;
  pooling_desc.u = gp->stride_h;
  pooling_desc.v = gp->stride_w;
  pooling_desc.R = gp->kh;
  pooling_desc.S = gp->kw;
  pooling_desc.pad_h = gp->pad_h;
  pooling_desc.pad_w = gp->pad_w;
  pooling_desc.pad_h_in = gp->ipad_h;
  pooling_desc.pad_w_in = gp->ipad_w;
  pooling_desc.pad_h_out = gp->opad_h;
  pooling_desc.pad_w_out = gp->opad_w;
  pooling_desc.threads = gp->num_threads;

  if(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT)
  {
    pooling_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    pooling_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
  }
  else if(gp->in_data_type == DT_BF16 && gp->out_data_type == DT_BF16)
  {
    pooling_desc.datatype_in = LIBXSMM_DNN_DATATYPE_BF16;
    pooling_desc.datatype_out = LIBXSMM_DNN_DATATYPE_BF16;
  }

  pooling_desc.datatype_mask = LIBXSMM_DNN_DATATYPE_I32;
  pooling_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;

  if(gp->pool_mode == MAX)
    pooling_desc.pooling_type = LIBXSMM_DNN_POOLING_MAX;
  else if(gp->pool_mode == AVE)
    pooling_desc.pooling_type = LIBXSMM_DNN_POOLING_AVG;

  libxsmm_handle = libxsmm_dnn_create_pooling( pooling_desc, &status );
  CHKERR_LIBXSMM_DNN( status );
}

void PoolXSMM::forwardPropagate(TensorBuf *inpb, TensorBuf *outpb, int *mask, int tid)
{
  void *input = inpb->getBuffer();
  void *output = outpb->getBuffer();
  void *scratch = scratchp->getBuffer();

  if(libxsmm_input == NULL && libxsmm_mask == NULL && libxsmm_output == NULL)
  {
    libxsmm_layout = libxsmm_dnn_pooling_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_input  = libxsmm_dnn_link_tensor( libxsmm_layout, input, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_bind_tensor( libxsmm_handle, libxsmm_input,     LIBXSMM_DNN_REGULAR_INPUT ) );

    libxsmm_layout = libxsmm_dnn_pooling_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_output  = libxsmm_dnn_link_tensor( libxsmm_layout, output, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_bind_tensor( libxsmm_handle, libxsmm_output,    LIBXSMM_DNN_REGULAR_OUTPUT ) );

    libxsmm_layout = libxsmm_dnn_pooling_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_POOLING_MASK, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_mask  = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)mask, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_bind_tensor( libxsmm_handle, libxsmm_mask  ,    LIBXSMM_DNN_POOLING_MASK ) );

    if(scratch == NULL)
    {
      long long mysize = libxsmm_dnn_pooling_get_scratch_size( libxsmm_handle, &status );
      CHKERR_LIBXSMM_DNN( status );
      scratch = libxsmm_aligned_scratch( mysize, 2097152 );
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
      long long int mysize = libxsmm_dnn_pooling_get_scratch_size( libxsmm_handle, &status );

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
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_bind_scratch( libxsmm_handle, scratch ) );
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
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
  }
}

void PoolXSMM::backPropagate(TensorBuf *deloutpb, int *mask, TensorBuf *delinpb, int tid)
{
  void *deloutput = deloutpb->getBuffer();
  void *delinput = delinpb->getBuffer();

  if(scratch != scratchp->getBuffer())
  {
    scratch = scratchp->getBuffer();
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_bind_scratch( libxsmm_handle, scratch ) );
  }

  if(libxsmm_deloutput == NULL && libxsmm_delinput == NULL)
  {
    libxsmm_layout = libxsmm_dnn_pooling_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_deloutput  = libxsmm_dnn_link_tensor( libxsmm_layout, deloutput, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_bind_tensor( libxsmm_handle, libxsmm_deloutput, LIBXSMM_DNN_GRADIENT_OUTPUT ) );

    libxsmm_layout = libxsmm_dnn_pooling_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_delinput  = libxsmm_dnn_link_tensor( libxsmm_layout, delinput, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_bind_tensor( libxsmm_handle, libxsmm_delinput,  LIBXSMM_DNN_GRADIENT_INPUT ) );
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
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
  }
  delinpb->setLayoutType(LIBXSMM_CUSTOM_LAYOUT);
}
