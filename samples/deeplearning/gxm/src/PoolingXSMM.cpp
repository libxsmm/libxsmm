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


#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "PoolingXSMM.hpp"

#define VLEN 16

PoolXSMM::PoolXSMM(PoolImplParams *gp, int engine) : PoolImpl(gp, engine)
{
  pooling_desc.N = gp->batch_size/NUM_NUMA_NODES;
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
  pooling_desc.threads = gp->num_threads/NUM_NUMA_NODES;

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

  for(int n=0; n<NUM_NUMA_NODES; n++)
  {
    libxsmm_handle[n] = libxsmm_dnn_create_pooling( pooling_desc, &status );
    CHKERR_LIBXSMM_DNN( status );
  }
}

void PoolXSMM::forwardPropagate(TensorBuf *inpb, TensorBuf *outpb, int *mask, int tid)
{
  int ifh = gp->iHeight;
  int ifw = gp->iWidth;
  int iph = gp->ipad_h;
  int ipw = gp->ipad_w;
  int ifhp = ifh + 2*iph;
  int ifwp = ifw + 2*ipw;
  int ofh = gp->oHeight;
  int ofw = gp->oWidth;
  int oph = gp->opad_h;
  int opw = gp->opad_w;
  int ofhp = ofh + 2*oph;
  int ofwp = ofw + 2*opw;

  void *input[NUM_NUMA_NODES];
  void *output[NUM_NUMA_NODES];
  int *pool_mask[NUM_NUMA_NODES];

  int imoff = pooling_desc.N * pooling_desc.C * ifhp * ifwp;
  if(gp->in_data_type == DT_FLOAT)
    imoff *= sizeof(float);
  else if(gp->in_data_type == DT_BF16)
    imoff *= sizeof(libxsmm_bfloat16);
  input[0] = inpb->getBuffer();
  for(int n=1; n<NUM_NUMA_NODES; n++)
    input[n] = input[n-1] + imoff;

  imoff = pooling_desc.N * pooling_desc.C * ofhp * ofwp;
  if(gp->in_data_type == DT_FLOAT)
    imoff *= sizeof(float);
  else if(gp->in_data_type == DT_BF16)
    imoff *= sizeof(libxsmm_bfloat16);
  output[0] = outpb->getBuffer();
  for(int n=1; n<NUM_NUMA_NODES; n++)
    output[n] = output[n-1] + imoff;

  imoff = pooling_desc.N * pooling_desc.C * ofhp * ofwp;
  pool_mask[0] = mask;
  for(int n=1; n<NUM_NUMA_NODES; n++)
    pool_mask[n] = pool_mask[n-1] + imoff;

  void **sptrptr = scratchp->getBufferPtr();

  for(int n=0; n<NUM_NUMA_NODES; n++)
  {
    if(libxsmm_input[n] == NULL && libxsmm_mask[n] == NULL && libxsmm_output[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_pooling_create_tensor_datalayout( libxsmm_handle[n], LIBXSMM_DNN_REGULAR_INPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, input[n], &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_pooling_bind_tensor( libxsmm_handle[n], libxsmm_input[n], LIBXSMM_DNN_REGULAR_INPUT));

      libxsmm_layout = libxsmm_dnn_pooling_create_tensor_datalayout( libxsmm_handle[n], LIBXSMM_DNN_REGULAR_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_output[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, output[n], &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_pooling_bind_tensor(libxsmm_handle[n], libxsmm_output[n], LIBXSMM_DNN_REGULAR_OUTPUT));

      libxsmm_layout = libxsmm_dnn_pooling_create_tensor_datalayout( libxsmm_handle[n], LIBXSMM_DNN_POOLING_MASK, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_mask[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)pool_mask[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_bind_tensor(libxsmm_handle[n], libxsmm_mask[n], LIBXSMM_DNN_POOLING_MASK ) );
    }
  }

  if(sptrptr == NULL)
  {
    sptrptr = (void**)libxsmm_aligned_malloc(NUM_NUMA_NODES*sizeof(void*), 2097152);
    scratchp->setBufferPtr(sptrptr);
  }

  if(prev_scratch_size == 0)
    prev_scratch_size = scratchp->getBufferSize();

  if(!updated_scratch_fwd || prev_scratch_size != scratchp->getBufferSize())
  {
    int max_size=0;

    for(int n=0; n<NUM_NUMA_NODES; n++)
    {
      if(sptrptr[n] == NULL)
      {
        long long mysize = libxsmm_dnn_pooling_get_scratch_size( libxsmm_handle[n], &status );
        CHKERR_LIBXSMM_DNN( status );
        sptrptr[n] = libxsmm_aligned_scratch( mysize, 2097152 );
        max_size = mysize;

#ifdef USE_MLSL
        if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
          printf("%s allocated %lld bytes for scratch @ %p\n",nname.c_str(), mysize, sptrptr[n]);
      }
      else
      {
        long long int ssize = scratchp->getBufferSize();
        long long int mysize = libxsmm_dnn_pooling_get_scratch_size( libxsmm_handle[n], &status );

        CHKERR_LIBXSMM_DNN( status );

        if(ssize < mysize)
        {
          libxsmm_free(sptrptr[n]);
          sptrptr[n] = (void*)libxsmm_aligned_malloc(mysize, 2097152);
          max_size = mysize;

#ifdef USE_MLSL
          if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
            printf("%s allocated %lld bytes for scratch @ %p, prev size was %lld bytes\n",nname.c_str(), mysize, sptrptr[n], ssize);
        }
        else
          max_size = ssize;
      }
    }
    scratchp->setBufferSize(max_size);

    for(int n=0; n<NUM_NUMA_NODES; n++)
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_bind_scratch( libxsmm_handle[n], sptrptr[n] ) );
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
    int ntps = gp->num_threads/NUM_NUMA_NODES;
    int n = tid/ntps;
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_execute_st( libxsmm_handle[n], LIBXSMM_DNN_COMPUTE_KIND_FWD, n*ntps, tid ) );
  }
}

void PoolXSMM::backPropagate(TensorBuf *deloutpb, int *mask, TensorBuf *delinpb, int tid)
{
  int ifh = gp->iHeight;
  int ifw = gp->iWidth;
  int iph = gp->ipad_h;
  int ipw = gp->ipad_w;
  int ifhp = ifh + 2*iph;
  int ifwp = ifw + 2*ipw;
  int ofh = gp->oHeight;
  int ofw = gp->oWidth;
  int oph = gp->opad_h;
  int opw = gp->opad_w;
  int ofhp = ofh + 2*oph;
  int ofwp = ofw + 2*opw;

  void *deloutput[NUM_NUMA_NODES];
  void *delinput[NUM_NUMA_NODES];
  int* pool_mask[NUM_NUMA_NODES];

  int imoff = pooling_desc.N * pooling_desc.C * ifhp * ifwp;
  if(gp->in_data_type == DT_FLOAT)
    imoff *= sizeof(float);
  else if(gp->in_data_type == DT_BF16)
    imoff *= sizeof(libxsmm_bfloat16);
  delinput[0] = delinpb->getBuffer();
  for(int n=1; n<NUM_NUMA_NODES; n++)
    delinput[n] = delinput[n-1] + imoff;

  imoff = pooling_desc.N * pooling_desc.C * ofhp * ofwp;
  if(gp->in_data_type == DT_FLOAT)
    imoff *= sizeof(float);
  else if(gp->in_data_type == DT_BF16)
    imoff *= sizeof(libxsmm_bfloat16);
  deloutput[0] = deloutpb->getBuffer();
  for(int n=1; n<NUM_NUMA_NODES; n++)
    deloutput[n] = deloutput[n-1] + imoff;

  imoff = pooling_desc.N * pooling_desc.C * ofhp * ofwp;
  pool_mask[0] = mask;
  for(int n=1; n<NUM_NUMA_NODES; n++)
    pool_mask[n] = pool_mask[n-1] + imoff;

  void **sptrptr = scratchp->getBufferPtr();
  if(!updated_scratch_bwd)
  {
    for(int n=0; n<NUM_NUMA_NODES; n++)
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_pooling_bind_scratch( libxsmm_handle[n], sptrptr[n] ) );
    updated_scratch_bwd = true;
  }

  for(int n=0; n<NUM_NUMA_NODES; n++)
  {
    if(libxsmm_deloutput[n] == NULL && libxsmm_delinput[n] == NULL)
    {
      libxsmm_layout = libxsmm_dnn_pooling_create_tensor_datalayout(libxsmm_handle[n], LIBXSMM_DNN_GRADIENT_OUTPUT, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_deloutput[n] = libxsmm_dnn_link_tensor( libxsmm_layout, deloutput[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_pooling_bind_tensor(libxsmm_handle[n], libxsmm_deloutput[n], LIBXSMM_DNN_GRADIENT_OUTPUT));

      libxsmm_layout = libxsmm_dnn_pooling_create_tensor_datalayout(libxsmm_handle[n], LIBXSMM_DNN_GRADIENT_INPUT, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_delinput[n]  = libxsmm_dnn_link_tensor( libxsmm_layout, delinput[n], &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN(libxsmm_dnn_pooling_bind_tensor(libxsmm_handle[n], libxsmm_delinput[n], LIBXSMM_DNN_GRADIENT_INPUT));
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
    int ntps = gp->num_threads/NUM_NUMA_NODES;
    int n = tid/ntps;
    CHKERR_LIBXSMM_DNN(libxsmm_dnn_pooling_execute_st(libxsmm_handle[n], LIBXSMM_DNN_COMPUTE_KIND_BWD, n*ntps, tid ) );
  }
  delinpb->setLayoutType(LIBXSMM_CUSTOM_LAYOUT);
}
