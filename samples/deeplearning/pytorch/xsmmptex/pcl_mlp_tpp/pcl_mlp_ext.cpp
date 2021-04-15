/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/


#include <torch/extension.h>
#include <ATen/record_function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>

#include <vector>
#include <map>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#pragma message "Using OpenMP"
#else
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#endif
#include <libxsmm.h>

#include "mlp_impl_common.h"
#include "mlp_impl_f32.h"
#include "mlp_impl_bf16.h"
#include "mlp_impl_bf16_amx.h"

#define PCL_ASSERT(cond, x...) do { if(!(cond)) { printf(x); fflush(stdout); exit(1); } } while(0)

typedef struct layer_handle {
  my_fc_fwd_config_f32 fwd_f32;
  my_fc_bwd_config_f32 bwd_f32;
  my_fc_fwd_config_bf16 fwd_bf16;
  my_fc_bwd_config_bf16 bwd_bf16;
  my_fc_fwd_config_bf16_amx fwd_bf16_amx;
  my_fc_bwd_config_bf16_amx bwd_bf16_amx;
  my_arch arch;
  long scratch_size;
  void *scratch;
  unsigned char *relumask;
} layer_handle;

void *create_handle(int N, int C, int K, int bn, int bc, int bk, int dtype, int fuse_bias, int act_type)
{
  layer_handle* libxsmm_handle;
  auto nThreads = omp_get_max_threads();
  my_eltwise_fuse fuse_type = MY_ELTWISE_FUSE_NONE;
  my_arch arch = MY_ARCH_F32_COMMON;
  size_t scratch_size = 0;

  PCL_ASSERT(act_type == 0 || act_type == 1, "Unsupported activation type\n");

  if (fuse_bias == 1 && act_type == 1) fuse_type = MY_ELTWISE_FUSE_BIAS_RELU;
  else if (fuse_bias == 1) fuse_type = MY_ELTWISE_FUSE_BIAS;
  else if (act_type == 1) fuse_type = MY_ELTWISE_FUSE_RELU;

  if ( dtype == 1 ) {
    arch = MY_ARCH_F32_COMMON;
  } else if ( dtype == 2 ) {
    if ( (libxsmm_cpuid() >= LIBXSMM_X86_AVX512_SPR) && (libxsmm_cpuid() <= LIBXSMM_X86_ALLFEAT) ) {
      arch = MY_ARCH_BF16_AMX_SPR;
    } else {
      arch = MY_ARCH_BF16_COMMON;
    }
  } else {
    /* shouldn't happen */
    PCL_ASSERT(0 == 1, "Unsupported datatype/arch combination\n");
  }

  libxsmm_handle = (layer_handle*)malloc(sizeof(layer_handle));
  PCL_ASSERT(libxsmm_handle != 0, "Failed to allocate memory for handle\n");

  libxsmm_handle->arch = arch;
  if ( libxsmm_handle->arch == MY_ARCH_F32_COMMON ) {
    libxsmm_handle->fwd_f32 = setup_my_fc_fwd_f32(N, C, K, bn, bc, bk, nThreads, fuse_type);
    libxsmm_handle->bwd_f32 = setup_my_fc_bwd_f32(N, C, K, bn, bc, bk, nThreads, fuse_type);
    if (libxsmm_handle->fwd_f32.scratch_size > scratch_size) scratch_size = libxsmm_handle->fwd_f32.scratch_size;
    if (libxsmm_handle->bwd_f32.scratch_size > scratch_size) scratch_size = libxsmm_handle->bwd_f32.scratch_size;
  } else if ( libxsmm_handle->arch == MY_ARCH_BF16_COMMON ) {
    libxsmm_handle->fwd_bf16 = setup_my_fc_fwd_bf16(N, C, K, bn, bc, bk, nThreads, fuse_type);
    libxsmm_handle->bwd_bf16 = setup_my_fc_bwd_bf16(N, C, K, bn, bc, bk, nThreads, fuse_type);
    if (libxsmm_handle->fwd_bf16.scratch_size > scratch_size) scratch_size = libxsmm_handle->fwd_bf16.scratch_size;
    if (libxsmm_handle->bwd_bf16.scratch_size > scratch_size) scratch_size = libxsmm_handle->bwd_bf16.scratch_size;
  } else {
    libxsmm_handle->fwd_bf16_amx = setup_my_fc_fwd_bf16_amx(N, C, K, bn, bc, bk, nThreads, fuse_type);
    libxsmm_handle->bwd_bf16_amx = setup_my_fc_bwd_bf16_amx(N, C, K, bn, bc, bk, nThreads, fuse_type);
    if (libxsmm_handle->fwd_bf16_amx.scratch_size > scratch_size) scratch_size = libxsmm_handle->fwd_bf16_amx.scratch_size;
    if (libxsmm_handle->bwd_bf16_amx.scratch_size > scratch_size) scratch_size = libxsmm_handle->bwd_bf16_amx.scratch_size;
  }

  libxsmm_handle->scratch_size = scratch_size;
  if (scratch_size > 0) libxsmm_handle->scratch = libxsmm_aligned_scratch( scratch_size, 2097152 );
  libxsmm_handle->relumask = 0;
  if (act_type == 1) libxsmm_handle->relumask = (unsigned char*)libxsmm_aligned_malloc( N*K*sizeof(unsigned char), 2097152);

  return (void *)libxsmm_handle;
}

at::Tensor mlp_forward(void *libxsmm_handle_, torch::Tensor input, torch::Tensor weight, torch::Tensor bias)
{
  layer_handle* libxsmm_handle = (layer_handle*)libxsmm_handle_;
  auto nbn = input.size(0);
  auto bn = input.size(2);
  auto nbk = weight.size(0);
  auto bk = weight.size(3);
  auto output = at::empty({nbn, nbk, bn, bk}, input.options());
  //std::cout << "FWD Handle = " << libxsmm_handle_ << std::endl;
  {
    RECORD_FUNCTION("xsmm_mm_fwd", std::vector<c10::IValue>({input, weight}), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      if ( libxsmm_handle->arch == MY_ARCH_F32_COMMON ) {
        my_fc_fwd_exec_f32( libxsmm_handle->fwd_f32, (const float*)weight.data_ptr(), (const float*)input.data_ptr(), (float*)output.data_ptr(), (const float*)bias.data_ptr(), libxsmm_handle->relumask, 0, tid, libxsmm_handle->scratch);
      } else if ( libxsmm_handle->arch == MY_ARCH_BF16_COMMON ) {
        my_fc_fwd_exec_bf16( libxsmm_handle->fwd_bf16, (const libxsmm_bfloat16*)weight.data_ptr(), (const libxsmm_bfloat16*)input.data_ptr(), (libxsmm_bfloat16*)output.data_ptr(), (const libxsmm_bfloat16*)bias.data_ptr(), libxsmm_handle->relumask, 0, tid, libxsmm_handle->scratch);
      } else {
        my_fc_fwd_exec_bf16_amx( libxsmm_handle->fwd_bf16_amx, (const libxsmm_bfloat16*)weight.data_ptr(), (const libxsmm_bfloat16*)input.data_ptr(), (libxsmm_bfloat16*)output.data_ptr(), (const libxsmm_bfloat16*)bias.data_ptr(), libxsmm_handle->relumask, 0, tid, libxsmm_handle->scratch);
      }
    }
  }
  return output;

}

std::vector<at::Tensor> mlp_backward(void *libxsmm_handle_, torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight)
{
  auto nbk = weight.size(0);
  auto bk = weight.size(3);
  auto grad_input = at::empty(input.sizes(), input.options());
  auto grad_weight = at::empty(weight.sizes(), weight.options());
  auto grad_bias = at::empty({nbk * bk}, weight.options());

  layer_handle* libxsmm_handle = (layer_handle*)libxsmm_handle_;
  //std::cout << "BWD Handle = " << libxsmm_handle_ << std::endl;

  {
    RECORD_FUNCTION("xsmm_mm_bwdupd", std::vector<c10::IValue>({grad_output, weight}), -1 /*torch::autograd::Node::peek_at_next_sequence_nr()*/);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      if ( libxsmm_handle->arch == MY_ARCH_F32_COMMON ) {
        my_fc_bwd_exec_f32( libxsmm_handle->bwd_f32, (const float*)weight.data_ptr(), (float*)grad_input.data_ptr(), (float*)grad_output.data_ptr(), (float*)grad_weight.data_ptr(), (const float*)input.data_ptr(),
          (float*)grad_bias.data_ptr(), libxsmm_handle->relumask, MY_PASS_BWD, 0, tid, libxsmm_handle->scratch);
      } else if ( libxsmm_handle->arch == MY_ARCH_BF16_COMMON ) {
        my_fc_bwd_exec_bf16( libxsmm_handle->bwd_bf16, (const libxsmm_bfloat16*)weight.data_ptr(), (libxsmm_bfloat16*)grad_input.data_ptr(), (const libxsmm_bfloat16*)grad_output.data_ptr(), (libxsmm_bfloat16*)grad_weight.data_ptr(),
          (const libxsmm_bfloat16*)input.data_ptr(), (libxsmm_bfloat16*)grad_bias.data_ptr(), libxsmm_handle->relumask, MY_PASS_BWD, 0, tid, libxsmm_handle->scratch);
      } else {
        my_fc_bwd_exec_bf16_amx( libxsmm_handle->bwd_bf16_amx, (libxsmm_bfloat16*)weight.data_ptr(), (libxsmm_bfloat16*)grad_input.data_ptr(), (const libxsmm_bfloat16*)grad_output.data_ptr(), (libxsmm_bfloat16*)grad_weight.data_ptr(),
          (const libxsmm_bfloat16*)input.data_ptr(), (libxsmm_bfloat16*)grad_bias.data_ptr(), libxsmm_handle->relumask, MY_PASS_BWD, 0, tid, libxsmm_handle->scratch);
      }
    }
  }
  return {grad_input, grad_weight, grad_bias};
}

void destroy_handle( void* libxsmm_handle_ )
{
  layer_handle* libxsmm_handle = (layer_handle*)libxsmm_handle_;
  //std::cout << "Destroy Handle = " << libxsmm_handle << std::endl;

  if (libxsmm_handle->scratch) libxsmm_free(libxsmm_handle->scratch);
  if (libxsmm_handle->relumask) libxsmm_free(libxsmm_handle->relumask);

  free(libxsmm_handle);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mlp_forward, "Pcl libxsmm MLP forward");
  m.def("backward", &mlp_backward, "Pcl libxsmm MLP backward");
  m.def("create_handle", &create_handle, "Pcl libxsmm create MLP handle");
  m.def("destroy_handle", &destroy_handle, "Pcl libxsmm destroy MLP handle");
}
