/******************************************************************************
 * * Copyright (c) Intel Corporation - All rights reserved.                      *
 * * This file is part of the LIBXSMM library.                                   *
 * *                                                                             *
 * * For information on the license, see the LICENSE file.                       *
 * * Further information: https://github.com/hfp/libxsmm/                        *
 * * SPDX-License-Identifier: BSD-3-Clause                                       *
 * ******************************************************************************/
/* Sasikanth Avancha, Ramanarayan Mohanty (Intel Corp.)
 * ******************************************************************************/
#include "mc_funcs.h"

#include "mlpcell_fp32.h"
#include "mlpcell_bf16.h"

void *create_handle(int N, int in_features, int out_features, int bn, int bc, int bk, bool bias, bool skip, int act, bool norm, float p, bool train, std::string dtype)
{
   // std::cout << dtype << " train: " << train << std::endl;
   if (dtype == "f32"){
       // std::cout << "Create Handle F32 "  << std::endl;
        MLPCell_F32 *handle = new MLPCell_F32(N, in_features, out_features, bn, bc, bk, bias, skip, act, norm, p, train);
        return (void*)handle;
   }
   else if(dtype == "bf16"){
       // std::cout << "Create Handle BF16 "  << std::endl;
       MLPCell_BF16 *handle = new MLPCell_BF16(N, in_features, out_features, bn, bc, bk, bias, skip, act, norm, p, train);
       return (void*)handle;
   }
   else{
       fprintf( stderr, "Can't create handle Datatype not matching...!\n");
   }

}

std::vector<at::Tensor> mlpcell_forward(void *handle_, std::vector<at::Tensor> inputs, std::string dtype)
{

    // TO-DO: Implement Skip=False and norm = True
    if (dtype == "f32"){
        // std::cout << "MF F32 "  << std::endl;
        MLPCell_F32 *handle = (MLPCell_F32*)handle_;
        RECORD_FUNCTION("mlpcell_fwd", std::vector<c10::IValue>({inputs[0], inputs[1]}));

        bool bias = handle->has_bias();
        bool skip = handle->has_skip();
        bool norm = handle->has_norm();

        if (!norm)
          return handle->fwd(inputs);
    }

    else if(dtype == "bf16"){

        // std::cout << "MF BF16 "  << std::endl;
        MLPCell_BF16 *handle = (MLPCell_BF16*)handle_;
        RECORD_FUNCTION("mlpcell_fwd", std::vector<c10::IValue>({inputs[0], inputs[1]}));

        bool bias = handle->has_bias();
        bool skip = handle->has_skip();
        bool norm = handle->has_norm();

        if(!norm)
          // printf("!skip && !norm --------------\n");
          return handle->fwd(inputs);
    }
    else{
        fprintf( stderr, "mlpcell forward function failed ...!\n");
    }
}

std::vector<at::Tensor> mlpcell_backward(void *handle_, std::vector<at::Tensor> inputs, std::string dtype)
{
   if (dtype == "f32"){
      // std::cout << "MB F32 "  << std::endl;
      MLPCell_F32 *handle = (MLPCell_F32*)handle_;
      RECORD_FUNCTION("mlpcell_bwd", std::vector<c10::IValue>({inputs[0], inputs[1]}));

      bool bias = handle->has_bias();
      bool skip = handle->has_skip();
      bool norm = handle->has_norm();
      if(!norm){
        return handle->bwd(inputs);}
    }
    else if(dtype == "bf16"){
      // std::cout << "MB BF16 "  << std::endl;
      MLPCell_BF16 *handle = (MLPCell_BF16*)handle_;
      RECORD_FUNCTION("mlpcell_bwd", std::vector<c10::IValue>({inputs[0], inputs[1]}));

      bool bias = handle->has_bias();
      bool skip = handle->has_skip();
      bool norm = handle->has_norm();
      if(!norm){
        return handle->bwd(inputs);}
      //else if(!norm){
      //  return handle->bwd_bias_skip_act_dropout(inputs);}
    }
    else{
      fprintf( stderr, "mlpcell backward function failed ...!\n");
    }
}

void destroy_handle(void* handle_, std::string dtype)
{
    if (dtype == "f32"){
        // std::cout << "DH F32 "  << std::endl;
        MLPCell_F32 *handle = (MLPCell_F32*)handle_;
        delete handle;
    }
    else if (dtype == "bf16"){
        // std::cout << "DH BF16 "  << std::endl;
        MLPCell_BF16 *handle = (MLPCell_BF16*)handle_;
        delete handle;
    }
    else{
        fprintf( stderr, "No handle to destroy ...!\n");
    }
}

//=====================================================================================================================================

inline void Dropout_f32_fwd(long N, long M, libxsmm_meltw_unary_param *params)
{
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltwfunction_unary dropout_kernel = libxsmm_dispatch_meltw_unary(M, N, NULL, NULL, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_DROPOUT);
  if ( dropout_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
    exit(-1);
  }
  dropout_kernel( params );
}

inline void Dropout_f32_bwd(long N, long M, libxsmm_meltw_unary_param *params)
{
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE ;
  libxsmm_meltwfunction_unary dropout_kernel = libxsmm_dispatch_meltw_unary(M, N, NULL, NULL, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV);
  if ( dropout_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
    exit(-1);
  }
  dropout_kernel( params );
}

inline void Dropout_bf16_fwd(long N, long M, libxsmm_meltw_unary_param *params)
{
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE ;
  libxsmm_meltwfunction_unary dropout_kernel = libxsmm_dispatch_meltw_unary(M, N, NULL, NULL, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_DROPOUT);
  if ( dropout_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
    exit(-1);
  }
  dropout_kernel( params );
}

inline void Dropout_bf16_bwd(long N, long M, libxsmm_meltw_unary_param *params)
{
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE ;
  libxsmm_meltwfunction_unary dropout_kernel = libxsmm_dispatch_meltw_unary(M, N, NULL, NULL, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, unary_flags, LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV);
  if ( dropout_kernel == NULL ) {
    fprintf( stderr, "JIT for DROPOUT TPP. Bailing...!\n");
    exit(-1);
  }
  dropout_kernel( params );
}


std::vector<at::Tensor> dropout_forward(torch::Tensor input, float p, bool train, std::string dtype)
{
  if(!train || p == 0.0)
  {
    at::Tensor dropout_mask = at::empty(input.sizes(), torch::TensorOptions().dtype(torch::kInt32));
    at::Tensor output = at::empty(input.sizes(), input.options());

    dropout_mask.zero_();
    output = input;

    return std::vector<at::Tensor>({output, dropout_mask});
  }
  else
  {
    at::Tensor output = at::empty(input.sizes(), input.options());
    int N=1;
    for(int i=0; i<output.sizes().size(); i++)
      N = N * output.sizes()[i];

    at::Tensor dropout_mask;
    if(input.sizes().size() == 4)
    {
      int d = input.sizes()[3]; // % 16 == 0 ? input.sizes()[3]/16 : input.sizes()[3]/16 + 1;
      dropout_mask = at::empty({input.sizes()[0], input.sizes()[1], input.sizes()[2], d}, torch::TensorOptions().dtype(torch::kInt32));
    }
    else if(input.sizes().size() == 2)
    {
      int d = input.sizes()[1];// % 16 == 0 ? input.sizes()[1]/16 : input.sizes()[1]/16 + 1;
      dropout_mask = at::empty({input.sizes()[0], d}, torch::TensorOptions().dtype(torch::kInt32));
    }

    __mmask32 (* mask) = (train && p > 0) ? (__mmask32 *)(dropout_mask.data_ptr()) : NULL;



    if (dtype == "f32")
    {

      float *in = input.data_ptr<float>();
      float *out = output.data_ptr<float>();

      libxsmm_meltw_unary_param dropout_params;


#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        int tid = omp_get_thread_num();
        int threads = omp_get_max_threads();
        int jobs = (N % threads == 0) ? N/threads : N/threads + 1;
        int tb = (tid*jobs < N) ? tid*jobs : N;
        int te = ((tid+1)*jobs < N) ? (tid+1)*jobs : N;
        int loc_N = te - tb;

        libxsmm_meltw_unary_param dropout_params;

        dropout_params.in.primary = in+tb;
        dropout_params.in.secondary = rnd_state;
        dropout_params.in.tertiary = &p;
        dropout_params.out.primary = out+tb;
        dropout_params.out.secondary = &mask[tb];
        Dropout_f32_fwd(1, loc_N, &dropout_params);

      }

      // libxsmm_free((void*)scratch);
      return std::vector<at::Tensor>({output, dropout_mask});
    }
    else if (dtype == "bf16")
    {
      libxsmm_bfloat16 *in = (libxsmm_bfloat16*)(input.data_ptr<at::BFloat16>());
      libxsmm_bfloat16 *out = (libxsmm_bfloat16*)(output.data_ptr<at::BFloat16>());

      libxsmm_meltw_unary_param dropout_params;


  #ifdef _OPENMP
  #pragma omp parallel
  #endif
      {
        int tid = omp_get_thread_num();
        int threads = omp_get_max_threads();
        int jobs = (N % threads == 0) ? N/threads : N/threads + 1;
        int tb = (tid*jobs < N) ? tid*jobs : N;
        int te = ((tid+1)*jobs < N) ? (tid+1)*jobs : N;
        int loc_N = te - tb;

        libxsmm_meltw_unary_param dropout_params;

        dropout_params.in.primary = in+tb;
        dropout_params.in.secondary = rnd_state;
        dropout_params.in.tertiary = &p;
        dropout_params.out.primary = out+tb;
        dropout_params.out.secondary = &mask[tb];
        Dropout_bf16_fwd(1, loc_N, &dropout_params);

      }
      return std::vector<at::Tensor>({output, dropout_mask});
    }
    else
    {
      fprintf( stderr, "Dropout forward function failed ...!\n");
    }
  }
}

at::Tensor dropout_backward(torch::Tensor input, torch::Tensor dropout_mask, float p, std::string dtype)
{
  at::Tensor output = at::empty(input.sizes(), input.options());
  int N=1;
  for(int i=0; i<output.sizes().size(); i++)
    N = N * output.sizes()[i];

  __mmask32 (* mask) = (__mmask32 *)(dropout_mask.data_ptr()) ;//: NULL;


   if (dtype == "f32")
    {

      float *in = input.data_ptr<float>();
      float *out = output.data_ptr<float>();

      libxsmm_meltw_unary_param dropout_params;

#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        int tid = omp_get_thread_num();
        int threads = omp_get_max_threads();
        int jobs = (N % threads == 0) ? N/threads : N/threads + 1;
        int tb = (tid*jobs < N) ? tid*jobs : N;
        int te = ((tid+1)*jobs < N) ? (tid+1)*jobs : N;
        int loc_N = te - tb;


        libxsmm_meltw_unary_param dropout_params;

        dropout_params.in.primary = in+tb;
        dropout_params.in.secondary = &mask[tb];
        dropout_params.in.tertiary = &p;
        dropout_params.out.primary = out+tb;
        Dropout_f32_bwd(1, loc_N, &dropout_params);

      }

    return(output);
  }
  else if (dtype == "bf16")
  {
    libxsmm_bfloat16 *in = (libxsmm_bfloat16*)(input.data_ptr<at::BFloat16>());
    libxsmm_bfloat16 *out = (libxsmm_bfloat16*)(output.data_ptr<at::BFloat16>());

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      int threads = omp_get_max_threads();
      int jobs = (N % threads == 0) ? N/threads : N/threads + 1;
      int tb = (tid*jobs < N) ? tid*jobs : N;
      int te = ((tid+1)*jobs < N) ? (tid+1)*jobs : N;
      int loc_N = te - tb;

      libxsmm_meltw_unary_param dropout_params;

      dropout_params.in.primary = in+tb;
      dropout_params.in.secondary = &mask[tb];
      dropout_params.in.tertiary = &p;
      dropout_params.out.primary = out+tb;
      Dropout_bf16_bwd(1, loc_N, &dropout_params);

    }
    return(output);
  }
  else
  {
    fprintf( stderr, "Dropout backward function failed ...!\n");
  }
}

// void lRelu_fwd_f32_f32_gold(unsigned int M, float *in, float *out, unsigned short *relu_mask, float alpha)
// {
//   unsigned int i;
//   __m512 x = _mm512_set1_ps(alpha);
//   for (i = 0; i < LIBXSMM_ALIGNDOWN(M, 16); i+=16) {
//     __m512 vin = _mm512_loadu_ps(in+i);
//     __mmask16 rmask = _mm512_cmp_ps_mask(vin, _mm512_setzero_ps(), _CMP_LT_OQ); // when vin <= 0 rmask=1 a=vin, b=0, if a <= b? 1; 0
//     __m512 vout = _mm512_mask_blend_ps(rmask, vin, _mm512_mul_ps(x, vin));
//     _mm512_storeu_ps(out+i, vout);
//     relu_mask[i/16] = rmask;
//   }
//   if (i < M) {
//     int rem = M - i;
//     __mmask16 mask = (1 << rem) - 1;
//     __m512 vin = _mm512_maskz_loadu_ps(mask, (in+i));
//     __mmask16 rmask = _mm512_mask_cmp_ps_mask(mask, vin, _mm512_setzero_ps(), _CMP_LT_OQ);
//     __m512 vout = _mm512_mask_blend_ps(rmask, vin, _mm512_mul_ps(x, vin));
//     _mm512_mask_storeu_ps((out+i), mask, vout);
//     relu_mask[i/16] = rmask;
//   }
// }

// void lRelu_bwd_f32_f32_gold(unsigned int M, float *in, float *out, unsigned short *relu_mask, float alpha) {
//   unsigned int i;
//   __m512 x = _mm512_set1_ps(alpha);
//   for (i = 0; i < LIBXSMM_ALIGNDOWN(M, 16); i+=16) {
//     __m512 vin = _mm512_loadu_ps(in+i);
//     __mmask16 rmask = relu_mask[i/16];
//     __m512 vout = _mm512_mask_blend_ps(rmask, vin, _mm512_mul_ps(x, vin));
//     _mm512_storeu_ps((out+i), vout);
//   }
//   if (i < M) {
//     int rem = M - i;
//     __mmask16 mask = (1 << rem) - 1;
//     __m512 vin = _mm512_maskz_loadu_ps(mask, (in+i));
//     __mmask16 rmask = relu_mask[i/16];
//     __m512 vout = _mm512_mask_blend_ps(rmask, vin, _mm512_mul_ps(x, vin));
//     _mm512_mask_storeu_ps((out+i), mask, vout);
//   }
// }

// std::vector<at::Tensor> leakyrelu_forward(torch::Tensor input, float alpha, std::string dtype)
// {

//     at::Tensor lrelu_mask = at::empty(input.sizes(), torch::TensorOptions().dtype(torch::kInt16));
//     at::Tensor output = at::empty(input.sizes(), input.options());

//     int nthreads = omp_get_max_threads();

//     // at::Tensor lrelu_count = at::empty(nthreads, torch::TensorOptions().dtype(torch::kInt16));
//     // lrelu_count._zeros();
//     int * lrelu_count = (int *) calloc(sizeof(int), nthreads);
//     int N=1;
//     for(int i=0; i<output.sizes().size(); i++)
//       N = N * output.sizes()[i];

//     unsigned short *mask = (unsigned short *)(lrelu_mask.data_ptr<short>());

//     float *in = input.data_ptr<float>();
//     float *out = output.data_ptr<float>();

//     unsigned int i;

//       #ifdef _OPENMP
//     #pragma omp parallel
//     #endif
//         {
//         int tid = omp_get_thread_num();
//         int threads = omp_get_max_threads();
//         int jobs = (N % threads == 0) ? N/threads : N/threads + 1;
//         int mchunk = (jobs % 16 ==0) ? jobs/16 : (jobs/16) + 1;
//         lrelu_count[tid] = mchunk * tid;
//         int tb = (tid*jobs < N) ? tid*jobs : N;
//         int te = ((tid+1)*jobs < N) ? (tid+1)*jobs : N;
//         int loc_N = te - tb;

//         lRelu_fwd_f32_f32_gold(loc_N, in+tb, out+tb, &mask[lrelu_count[tid]++], alpha);
//         }
//     free(lrelu_count);
//     return std::vector<at::Tensor>({output, lrelu_mask});
// }

// at::Tensor leakyrelu_backward(torch::Tensor input, torch::Tensor lrelu_mask, float alpha, std::string dtype)
// {
//   at::Tensor output = at::empty(input.sizes(), input.options());
//   unsigned short *mask = (unsigned short *)(lrelu_mask.data_ptr<short>());

//   int nthreads = omp_get_max_threads();

//   // at::Tensor lrelu_count = at::empty(nthreads, torch::TensorOptions().dtype(torch::kInt16));
//   int * lrelu_count = (int *) calloc(sizeof(int), nthreads);

//   int N=1;
//   for(int i=0; i<output.sizes().size(); i++)
//     N = N * output.sizes()[i];

//   float *in = input.data_ptr<float>();
//   float *out = output.data_ptr<float>();

//   #ifdef _OPENMP
// #pragma omp parallel
// #endif
//   {
//     int tid = omp_get_thread_num();
//     int threads = omp_get_max_threads();
//     int jobs = (N % threads == 0) ? N/threads : N/threads + 1;
//     int mchunk = (jobs % 16 ==0) ? jobs/16 : (jobs/16) + 1;
//     lrelu_count[tid] = mchunk * tid;
//     int tb = (tid*jobs < N) ? tid*jobs : N;
//     int te = ((tid+1)*jobs < N) ? (tid+1)*jobs : N;
//     int loc_N = te - tb;

//     lRelu_bwd_f32_f32_gold(loc_N, in+tb, out+tb, &mask[lrelu_count[tid]++], alpha);
//   }
//   free (lrelu_count);

//   return(output);
// }



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_libxsmm", &init_libxsmm, "Initialize libxsmm");
  m.def("set_rnd_seed", &set_rnd_seed, "Set seed for Rnd generator");
  m.def("mlpcell_forward", &mlpcell_forward, "GraphSAGE MLP Cell forward");
  m.def("mlpcell_backward", &mlpcell_backward, "GraphSAGE MLP backward");
  m.def("dropout_forward", &dropout_forward, "PCL Dropout forward");
  m.def("dropout_backward", &dropout_backward, "PCL Dropout backward");
  // m.def("leakyrelu_forward", &leakyrelu_forward, "PCL LeakyReLU forward");
  // m.def("leakyrelu_backward", &leakyrelu_backward, "PCL LeakyReLU backward");
  m.def("create_handle", &create_handle, "GraphSAGE MLP Create Handle");
  m.def("destroy_handle", &destroy_handle, "GraphSAGE MLP Destroy Handle");
}

