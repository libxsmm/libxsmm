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

#include "FCXSMM.hpp"

extern int iter;

FCXSMM::FCXSMM(FCImplParams *gp, int engine) : FCImpl(gp, engine)
{
  /* setup LIBXSMM handle */
  fullyconnected_desc.N = gp->batch_size;
  fullyconnected_desc.C = gp->nInput;
  fullyconnected_desc.K = gp->nOutput;
  fullyconnected_desc.threads = gp->num_threads;
  fullyconnected_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
  fullyconnected_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
  fullyconnected_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  fullyconnected_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE;

  libxsmm_handle = libxsmm_dnn_create_fullyconnected( fullyconnected_desc, &status );
  CHKERR_LIBXSMM_DNN( status );
}

void FCXSMM::forwardPropagate(TensorBuf *inpb, TensorBuf* weightpb, TensorBuf* biaspb, TensorBuf *outpb, int tid)
{
#ifdef RETURNALL
  return;
#endif

  assert(top_compute_engine != -1);
  assert(bot_compute_engine != -1);

  void *input = inpb->getBuffer();
  void *weight = weightpb->getBuffer();
  void *bias;
  if(gp->bias_term)
    bias = biaspb->getBuffer();
  void *output = outpb->getBuffer();
  void *scratch = scratchp->getBuffer();

  __assume_aligned(input,64);
  __assume_aligned(weight,64);
  __assume_aligned(bias, 64);
  __assume_aligned(output,64);


  /* setup LIBXSMM buffers */
  if(libxsmm_input == NULL && libxsmm_filter == NULL && libxsmm_output == NULL)
  {
    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_input  = libxsmm_dnn_link_tensor( libxsmm_layout, input, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle, libxsmm_input, LIBXSMM_DNN_REGULAR_INPUT));

    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout(libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_filter  = libxsmm_dnn_link_tensor( libxsmm_layout, weight, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle, libxsmm_filter, LIBXSMM_DNN_REGULAR_FILTER ) );

    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout(libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, &status);
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_output  = libxsmm_dnn_link_tensor( libxsmm_layout, output, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle, libxsmm_output, LIBXSMM_DNN_REGULAR_OUTPUT ) );

    if(scratch == NULL)
    {
      long long mysize = libxsmm_dnn_fullyconnected_get_scratch_size( libxsmm_handle, &status );
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
      long long int mysize = libxsmm_dnn_fullyconnected_get_scratch_size( libxsmm_handle, &status );

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
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_scratch( libxsmm_handle, scratch ) );
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
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
  }

  if(gp->bias_term)
  {
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for(int img=0; img<gp->batch_size; img++)
      for(int ofm=0; ofm<gp->nOutput; ofm++)
        ((float*)output)[img*gp->batch_size+ofm] += ((float*)bias)[ofm];
  }
}

void FCXSMM::backPropagate(TensorBuf *deloutpb, TensorBuf *weightpb, TensorBuf *delinpb, int tid)
{
#ifdef RETURNALL
  return;
#endif

  assert(top_compute_engine != -1);
  assert(bot_compute_engine != -1);

  void *deloutput = deloutpb->getBuffer();
  void *delinput = delinpb->getBuffer();

  __assume_aligned(deloutput, 64);
  __assume_aligned(delinput, 64);

  if(scratch != scratchp->getBuffer())
  {
    scratch = scratchp->getBuffer();
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_scratch( libxsmm_handle, scratch ) );
  }

  if(libxsmm_deloutput == NULL && libxsmm_delinput == NULL)
  {
    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_deloutput  = libxsmm_dnn_link_tensor( libxsmm_layout, deloutput, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle, libxsmm_deloutput, LIBXSMM_DNN_GRADIENT_OUTPUT ) );

    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_delinput  = libxsmm_dnn_link_tensor( libxsmm_layout, delinput, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle, libxsmm_delinput, LIBXSMM_DNN_GRADIENT_INPUT ) );
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
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
  }
}

void FCXSMM::weightUpdate(TensorBuf *deloutpb, TensorBuf *inpb, TensorBuf *delweightpb, TensorBuf *delbiaspb, int tid)
{
#ifdef RETURNALL
  return;
#endif

  assert(top_compute_engine != -1);
  assert(bot_compute_engine != -1);

  void *deloutput = deloutpb->getBuffer();
  void *delweight = delweightpb->getBuffer();
  void *delbias;
  if(gp->bias_term)
    delbias = delbiaspb->getBuffer();

  __assume_aligned(delweight,64);
  __assume_aligned(delbias, 64);

  if(scratch != scratchp->getBuffer())
  {
    scratch = scratchp->getBuffer();
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_scratch( libxsmm_handle, scratch ) );
  }

  if(libxsmm_delfilter == NULL)
  {
    libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER, &status );
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_delfilter  = libxsmm_dnn_link_tensor( libxsmm_layout, delweight, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle, libxsmm_delfilter, LIBXSMM_DNN_GRADIENT_FILTER ) );
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
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid ) );
  }

  if(gp->bias_term)
  {
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for(int ofm=0; ofm<gp->nOutput; ofm++) {
      for(int img=0; img<gp->batch_size; img++)
        ((float*)delbias)[ofm] += ((float*)deloutput)[img*gp->nOutput+ofm];
    }
  }

}
