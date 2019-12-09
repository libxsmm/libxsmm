/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
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
      weight[n] = lptrptr[n] + offset*sizeof(libxsmm_bfloat16);
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
    if(libxsmm_input[n] == NULL && libxsmm_output[n] == NULL && libxsmm_filter[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout(libxsmm_handle[n], LIBXSMM_DNN_REGULAR_INPUT, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, input[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle[n], libxsmm_input[n], LIBXSMM_DNN_REGULAR_INPUT));

      libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout(libxsmm_handle[n], LIBXSMM_DNN_REGULAR_FILTER, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_filter[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, weight[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle[n], libxsmm_filter[n], LIBXSMM_DNN_REGULAR_FILTER ) );

      libxsmm_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout(libxsmm_handle[n], LIBXSMM_DNN_REGULAR_OUTPUT, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_output[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, output[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_fullyconnected_bind_tensor( libxsmm_handle[n], libxsmm_output[n], LIBXSMM_DNN_REGULAR_OUTPUT ) );
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
  void **ptrptr;

  if(gp->in_data_type == DT_BF16)
  {
#ifdef BF16_MLSL
    ptrptr = delweightpb->getBufferPtr();
#else
    ptrptr = delweightpb->getLPBufferPtr();
#endif
  }
  else
    ptrptr = delweightpb->getBufferPtr();

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

        if(n == 0)
        {
          for(int nn=1; nn<gp->num_numa_nodes; nn++)
          {
            float *wgp = (float*)dwt_ptr[n]+nn*jn;
            float *rgp = (float*)dwt_ptr[nn]+nn*jn;

#pragma vector nontemporal
#pragma omp simd
            for(int i=tb; i<te; i++)
              wgp[i] = rgp[i];
          }
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
}
