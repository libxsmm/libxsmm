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
/* Sasikanth Avancha, Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/

#include "FCXSMM.hpp"

extern int iter;

FCXSMM::FCXSMM(FCImplParams *gp, int engine) : FCImpl(gp, engine)
{
  /* setup LIBXSMM handle */
  fullyconnected_desc.N = gp->batch_size/gp->num_numa_nodes;
  fullyconnected_desc.C = gp->nInput;
  fullyconnected_desc.K = gp->nOutput;
  fullyconnected_desc.threads = gp->num_threads/gp->num_numa_nodes;

  if(gp->in_data_type == DT_FLOAT)
    fullyconnected_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
  else if(gp->in_data_type == DT_BF16)
    fullyconnected_desc.datatype_in = LIBXSMM_DNN_DATATYPE_BF16;

  if(gp->out_data_type == DT_FLOAT)
    fullyconnected_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
  else if(gp->out_data_type == DT_BF16)
    fullyconnected_desc.datatype_out = LIBXSMM_DNN_DATATYPE_BF16;

  fullyconnected_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  fullyconnected_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  fullyconnected_desc.fuse_ops = LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE;

  for(int i=0; i<gp->num_numa_nodes; i++)
  {
    libxsmm_handle[i] = libxsmm_dnn_create_fullyconnected( fullyconnected_desc, &status );
    CHKERR_LIBXSMM_DNN( status );
  }
}

void FCXSMM::forwardPropagate(TensorBuf *inpb, TensorBuf* weightpb, TensorBuf* hweightpb, TensorBuf* biaspb, TensorBuf *outpb, int tid)
{
#ifdef RETURNALL
  return;
#endif

  int nIFM = gp->nInput;
  int kh = gp->kh;
  int kw = gp->kw;
  assert(top_compute_engine != -1);
  assert(bot_compute_engine != -1);

  void *input[NUM_NUMA_NODES];
  input[0] = inpb->getBuffer();
  int imoff = fullyconnected_desc.N * fullyconnected_desc.C;
  if(gp->in_data_type == DT_FLOAT)
    imoff = imoff*sizeof(float);
  else if(gp->in_data_type == DT_BF16)
    imoff = imoff*sizeof(libxsmm_bfloat16);

  for(int n=1; n<gp->num_numa_nodes; n++)
    input[n] = input[n-1] + imoff;

  void *weight[NUM_NUMA_NODES], *f32_weight[NUM_NUMA_NODES];
  void *wt_prv_ptr;
  void **lptrptr = weightpb->getLPBufferPtr();
  void **ptrptr = weightpb->getBufferPtr();

  int offset = weightpb->getOffset();

  if(lptrptr != NULL)
  {
    for(int n=0; n<gp->num_numa_nodes; n++)
    {
      weight[n] = lptrptr[n] + offset*sizeof(libxsmm_bfloat16);
      f32_weight[n] = ptrptr[n] + offset*sizeof(float);
    }
  }
  else
    for(int n=0; n<gp->num_numa_nodes; n++)
      weight[n] = ptrptr[n] + offset*sizeof(float);

  void *hwt_ptr;
  if(hweightpb != NULL)
    hwt_ptr = hweightpb->getBuffer();
  else
    hwt_ptr = NULL;

  void *bias;
  if(gp->bias_term)
    bias = biaspb->getBuffer();

  void *output[NUM_NUMA_NODES];
  output[0] = outpb->getBuffer();
  imoff = fullyconnected_desc.N * fullyconnected_desc.K;
  if(gp->out_data_type == DT_FLOAT)
    imoff = imoff*sizeof(float);
  else if(gp->out_data_type == DT_BF16)
    imoff = imoff*sizeof(libxsmm_bfloat16);

  for(int n=1; n<gp->num_numa_nodes; n++)
    output[n] = output[n-1] + imoff;

  void **sptrptr = scratchp->getBufferPtr();

  /* setup LIBXSMM buffers */
  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_input[n] == NULL && libxsmm_output[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout(libxsmm_handle[n], LIBXSMM_DNN_REGULAR_INPUT, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, input[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle[n], libxsmm_input[n], LIBXSMM_DNN_REGULAR_INPUT));


      libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout(libxsmm_handle[n], LIBXSMM_DNN_REGULAR_OUTPUT, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_output[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, output[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle[n], libxsmm_output[n], LIBXSMM_DNN_REGULAR_OUTPUT ) );
    }
  }

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_filter[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout(libxsmm_handle[n], LIBXSMM_DNN_REGULAR_FILTER, &status);
      CHKERR_LIBXSMM_DNN( status );

      int welem = gp->nInput * gp->nOutput * gp->kw * gp->kh;
      if(gp->in_data_type == DT_FLOAT)
      {
        int wsize = welem*sizeof(float);
        wt_prv_ptr = (void*)libxsmm_aligned_malloc(welem*sizeof(float), 2097152);

        // Transform weight layout
        libxsmm_filter[n] = libxsmm_dnn_link_tensor( libxsmm_layout, wt_prv_ptr, &status );
        CHKERR_LIBXSMM_DNN( status );

        CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_filter[n], weight[n], LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );
        memcpy(weight[n], wt_prv_ptr, welem*sizeof(float));

        if(n == 0)
        {
          libxsmm_checkpoint_filter = libxsmm_dnn_link_tensor(libxsmm_layout, weight[n], &status);
          CHKERR_LIBXSMM_DNN( status );
        }
        libxsmm_filter[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, weight[n], &status );
        CHKERR_LIBXSMM_DNN( status );

        // Transform weight history layout
        if(n == 0)
        {
          if(hwt_ptr != NULL)
          {
            libxsmm_dnn_tensor *libxsmm_temp = libxsmm_dnn_link_tensor( libxsmm_layout, wt_prv_ptr, &status );
            CHKERR_LIBXSMM_DNN( status );

            CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_temp, (void*)hwt_ptr, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );
            memcpy(hwt_ptr, wt_prv_ptr, welem*sizeof(float));

            libxsmm_checkpoint_history_filter = libxsmm_dnn_link_tensor(libxsmm_layout, hwt_ptr, &status);
            CHKERR_LIBXSMM_DNN( status );
          }
        }
        libxsmm_free(wt_prv_ptr);
      }
      else if(gp->in_data_type == DT_BF16)
      {
        wt_prv_ptr = (void*)libxsmm_aligned_malloc(welem*sizeof(libxsmm_bfloat16), 2097152);

        // Transform BF16 weight layout
#if 0
        libxsmm_filter[n] = libxsmm_dnn_link_tensor( libxsmm_layout, wt_prv_ptr, &status );
        CHKERR_LIBXSMM_DNN( status );

        CHKERR_LIBXSMM_DNN(libxsmm_dnn_copyin_tensor(libxsmm_filter[n], weight[n], LIBXSMM_DNN_TENSOR_FORMAT_KCRS) );
        memcpy(weight[n], wt_prv_ptr, welem*sizeof(libxsmm_bfloat16));
#endif
        libxsmm_filter[n] = libxsmm_dnn_link_tensor( libxsmm_layout, weight[n], &status );
        CHKERR_LIBXSMM_DNN( status );
        libxsmm_free(wt_prv_ptr);

#if 0
        libxsmm_dnn_tensor_datalayout * mylayout = libxsmm_dnn_get_tensor_datalayout (libxsmm_filter[n], &status);
        /* check for VNNI weights */
        assert( mylayout->num_dims == 7 );

        int lpb = mylayout->dim_size[0];
        int bofm = mylayout->dim_size[1];
        int bifm = mylayout->dim_size[2];
        int S = mylayout->dim_size[3];
        int R = mylayout->dim_size[4];
        int ifmb = mylayout->dim_size[5];
        int ofmb = mylayout->dim_size[6];

        int welem = gp->nInput * gp->nOutput;
        void *wt_prv_ptr = (void*)libxsmm_aligned_malloc(welem*sizeof(float), 2097152);
        memcpy(wt_prv_ptr, f32_weight[n], welem*sizeof(float));
        float (* __restrict wt_prv)[nIFM][kh][kw] = (float (*)[*][*][*])wt_prv_ptr;
        float (* __restrict f32_wt)[ifmb][R][S][bifm][bofm][lpb] = (float (*)[*][*][*][*][*][*])f32_weight[n];
        for(int k=0; k<ofmb; k++)
          for(int c=0; c<ifmb; c++)
            for(int r=0; r<R; r++)
              for(int s=0; s<S; s++)
                for(int cc=0; cc<bifm; cc++)
                  for(int kk=0; kk<bofm; kk++)
                    for(int ccc=0; ccc<lpb; ccc++)
                      f32_wt[k][c][r][s][cc][kk][ccc] = wt_prv[k*bofm+kk][c*bifm*lpb+cc*lpb+ccc][r][s];

        libxsmm_free(wt_prv_ptr);
#endif
#if 0
        // Transform FP32 weight layout
        if(n == 0)
        {
          libxsmm_checkpoint_filter = libxsmm_dnn_link_tensor( libxsmm_layout, f32_weight[n], &status );
          CHKERR_LIBXSMM_DNN( status );

          // Transform FP32 weight history layout
          if(hwt_ptr != NULL)
          {
            libxsmm_checkpoint_history_filter = libxsmm_dnn_link_tensor( libxsmm_layout, wt_prv_ptr, &status );
            CHKERR_LIBXSMM_DNN( status );

            void *hfwt_ptr = hweightpb->getBuffer();
            CHKERR_LIBXSMM_DNN(libxsmm_dnn_copyin_tensor(libxsmm_checkpoint_history_filter, hfwt_ptr, LIBXSMM_DNN_TENSOR_FORMAT_KCRS));
            memcpy(hfwt_ptr, wt_prv_ptr, welem*sizeof(float));

            libxsmm_checkpoint_history_filter = libxsmm_dnn_link_tensor(libxsmm_layout, hfwt_ptr, &status);
            CHKERR_LIBXSMM_DNN( status );
          }
        }
        libxsmm_free(wt_prv_ptr);
        wt_prv_ptr = NULL;
#endif
      }
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle[n], libxsmm_filter[n], LIBXSMM_DNN_REGULAR_FILTER ) );
    }
  }

  if(sptrptr == NULL)
  {
    sptrptr = (void**)libxsmm_aligned_malloc(gp->num_numa_nodes*sizeof(void*), 2097152);
    scratchp->setBufferPtr(sptrptr);
  }

  int max_size = 0;
  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(sptrptr[n] == NULL)
    {
      int mysize = libxsmm_dnn_fullyconnected_get_scratch_size( libxsmm_handle[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      sptrptr[n] = libxsmm_aligned_scratch( mysize, 2097152 );
      max_size = mysize;

#ifdef USE_MLSL
      if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
        printf("%s allocated %d bytes for scratch @ %p\n",nname.c_str(), mysize, sptrptr[n]);
    }
    else
    {
      int ssize = scratchp->getBufferSize();
      int mysize = libxsmm_dnn_fullyconnected_get_scratch_size( libxsmm_handle[n], &status );

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
      max_size = ssize;
    }
  }
  scratchp->setBufferSize(max_size);

  if(prev_scratch_size == 0)
    prev_scratch_size = scratchp->getBufferSize();

  if(!updated_scratch_fwd || prev_scratch_size != scratchp->getBufferSize())
  {
    for(int n=0; n<gp->num_numa_nodes; n++)
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_scratch( libxsmm_handle[n], sptrptr[n] ) );
    updated_scratch_fwd = true;
    prev_scratch_size = scratchp->getBufferSize();
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
    int ntps = gp->num_threads/gp->num_numa_nodes;
    int n = tid/ntps;
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle[n], LIBXSMM_DNN_COMPUTE_KIND_FWD, n*ntps, tid ) );
  }
}

void FCXSMM::backPropagate(TensorBuf *deloutpb, TensorBuf *weightpb, TensorBuf *delinpb, int tid)
{

  assert(top_compute_engine != -1);
  assert(bot_compute_engine != -1);

  void *deloutput[NUM_NUMA_NODES];
  void *delinput[NUM_NUMA_NODES];
  deloutput[0] = deloutpb->getBuffer();
  delinput[0] = delinpb->getBuffer();

  int imoff = fullyconnected_desc.N * fullyconnected_desc.K;
  if(gp->out_data_type == DT_FLOAT)
    imoff = imoff*sizeof(float);
  else if(gp->out_data_type == DT_BF16)
    imoff = imoff*sizeof(libxsmm_bfloat16);

  for(int n=1; n<gp->num_numa_nodes; n++)
    deloutput[n] = deloutput[n-1] + imoff;

  imoff = fullyconnected_desc.N * fullyconnected_desc.C;
  if(gp->in_data_type == DT_FLOAT)
    imoff = imoff*sizeof(float);
  else if(gp->in_data_type == DT_BF16)
    imoff = imoff*sizeof(libxsmm_bfloat16);

  for(int n=1; n<gp->num_numa_nodes; n++)
    delinput[n] = delinput[n-1] + imoff;

  void **sptrptr = scratchp->getBufferPtr();
  if(!updated_scratch_bwd)
  {
    for(int n=0; n<gp->num_numa_nodes; n++)
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_scratch( libxsmm_handle[n], sptrptr[n] ) );
    updated_scratch_bwd = true;
  }

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_deloutput[n] == NULL && libxsmm_delinput[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_handle[n], LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_deloutput[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, deloutput[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle[n], libxsmm_deloutput[n], LIBXSMM_DNN_GRADIENT_OUTPUT ) );

      libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_handle[n], LIBXSMM_DNN_GRADIENT_INPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_delinput[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, delinput[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle[n], libxsmm_delinput[n], LIBXSMM_DNN_GRADIENT_INPUT ) );
    }
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
    int ntps = gp->num_threads/gp->num_numa_nodes;
    int n = tid/ntps;
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle[n], LIBXSMM_DNN_COMPUTE_KIND_BWD, n*ntps, tid ) );
  }
}

void FCXSMM::weightUpdate(TensorBuf *deloutpb, TensorBuf *inpb, TensorBuf *delweightpb, TensorBuf *delbiaspb, int tid)
{

  int ofm = fullyconnected_desc.K;
  int ifm = fullyconnected_desc.C;
  int kh = 1;
  int kw = 1;
  assert(top_compute_engine != -1);
  assert(bot_compute_engine != -1);

  void *dwt_ptr[NUM_NUMA_NODES];

#ifdef BF16_MLSL
  void **ptrptr = delweightpb->getBufferPtr();
#else
  void **ptrptr = delweightpb->getLPBufferPtr();
#endif

  int offset = delweightpb->getOffset();
  if(gp->in_data_type == DT_FLOAT)
    offset = offset*sizeof(float);
  else if(gp->in_data_type == DT_BF16)
    offset = offset*sizeof(libxsmm_bfloat16);

  for(int n=0; n<gp->num_numa_nodes; n++)
    dwt_ptr[n] = ptrptr[n] + offset;

  void **sptrptr = scratchp->getBufferPtr();
  if(!updated_scratch_upd)
  {
    for(int n=0; n<gp->num_numa_nodes; n++)
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_scratch( libxsmm_handle[n], sptrptr[n] ) );
    updated_scratch_upd = true;
  }

  for(int n=0; n<gp->num_numa_nodes; n++)
  {
    if(libxsmm_delfilter[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout( libxsmm_handle[n], LIBXSMM_DNN_GRADIENT_FILTER, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_delfilter[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, dwt_ptr[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle[n], libxsmm_delfilter[n], LIBXSMM_DNN_GRADIENT_FILTER ) );
    }
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
    int ntps = gp->num_threads/gp->num_numa_nodes;
    int n = tid/ntps;
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_execute_st( libxsmm_handle[n], LIBXSMM_DNN_COMPUTE_KIND_UPD, n*ntps, tid ) );

#ifdef USE_MLSL
#pragma omp barrier

    if(gp->num_numa_nodes > 1)
    {
      if(gp->in_data_type == DT_FLOAT)
      {
#include "reduce_weight_grads.c"
      }
      else if(gp->in_data_type == DT_BF16)
      {
#include "reduce_weight_grads_bf16.c"
      }
    }
#endif
  }
}
