/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Narendra Chaudhary, Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/


#include<immintrin.h>
// #include<x86intrin.h>
#include<iostream>
#include<stdio.h>
// #include<stdlib.h>
#include<torch/extension.h>
#include<tuple>
#include<omp.h>
#include<libxsmm.h>
#include<libxsmm_intrinsics_x86.h>

// #include <torch/csrc/autograd/record_function.h>
#include <ATen/record_function.h>

#define PCL_ASSERT(cond, x...) do { if(!(cond)) { printf(x); fflush(stdout); exit(1); } } while(0)

#define XS_TILE_FORWARD 64
#define XS_TILE_DBACKWARD 64
#define XS_TILE_WBACKWARD 64                // 256 for peak performance

#define USE_TPP                             // Flag for using the TPP kernels

#ifndef USE_TPP                             // If not usintg TPP kernels

// #include "Bfloat16.h"

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
void bf16_vnni_reformat(libxsmm_bfloat16 *_in, libxsmm_bfloat16 *_out, int M, int N, int ld_in, int ld_out) {
    /* Function to do VNNI transform */

#if defined(LIBXSMM_INTRINSICS_AVX512_CORE)
  int n_full_pairs = N/2, n_pair, m;
  int half_n_pair = N%2;
  libxsmm_bfloat16 *in = _in, *out = _out;
  const __m512i selector = LIBXSMM_INTRINSICS_MM512_SET_EPI16(32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0);
  const __m512i offsets_lo = LIBXSMM_INTRINSICS_MM512_SET_EPI16(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
  const __m512i offsets_hi = LIBXSMM_INTRINSICS_MM512_SET_EPI16(31, 31, 30, 30, 29, 29, 28, 28, 27, 27, 26, 26, 25, 25, 24, 24, 23, 23, 22, 22, 21, 21, 20, 20, 19, 19, 18, 18, 17, 17, 16, 16);
  const __m512i idx_lo =  _mm512_or_epi32(selector, offsets_lo);
  const __m512i idx_hi =  _mm512_or_epi32(selector, offsets_hi);
  const __m512i zero_reg = _mm512_setzero_si512();
  __m512i n0, n1, out_lo, out_hi;
  LIBXSMM_UNUSED(ld_out);
  for (n_pair = 0; n_pair < n_full_pairs; n_pair++) {
    for (m = 0; m < M; m+=32) {
      n0 = _mm512_loadu_si512((const libxsmm_bfloat16*)in+m);
      n1 = _mm512_loadu_si512((const libxsmm_bfloat16*)in+m+ld_in);
      out_lo = _mm512_permutex2var_epi16(n0, idx_lo, n1);
      out_hi = _mm512_permutex2var_epi16(n0, idx_hi, n1);
      _mm512_storeu_si512((libxsmm_bfloat16*)out+m*2, out_lo);
      _mm512_storeu_si512((libxsmm_bfloat16*)out+m*2+32, out_hi);
#ifdef USE_CLDEMOTE
      _mm_cldemote((libxsmm_bfloat16*)out+m*2);
      _mm_cldemote((libxsmm_bfloat16*)out+m*2+32);
#endif
    }
    in += 2*ld_in;
    // out += 2*ld_in;                      // I change this
    out += 2*ld_out;
  }
  if (half_n_pair == 1) {
    for (m = 0; m < M; m+=32) {
      n0 = _mm512_loadu_si512((const libxsmm_bfloat16*)in+m);
      n1 = zero_reg;
      out_lo = _mm512_permutex2var_epi16(n0, idx_lo, n1);
      out_hi = _mm512_permutex2var_epi16(n0, idx_lo, n1);
      _mm512_storeu_si512((libxsmm_bfloat16*)out+m*2, out_lo);
      _mm512_storeu_si512((libxsmm_bfloat16*)out+m*2+32, out_hi);
#ifdef USE_CLDEMOTE
      _mm_cldemote((libxsmm_bfloat16*)out+m*2);
      _mm_cldemote((libxsmm_bfloat16*)out+m*2+32);
#endif
    }
  }
#else
 LIBXSMM_UNUSED(_in); LIBXSMM_UNUSED(_out); LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ld_in); LIBXSMM_UNUSED(ld_out);
#endif
}

libxsmm_xtransfunction get_tr_kernel(int typesize, int M, int N, int LDO) {
    /* Function for fetching a transpose kernel */
    libxsmm_xtransfunction tr_kernel;
    libxsmm_descriptor_blob blob;
    libxsmm_trans_descriptor* tr_desc;
    tr_desc = libxsmm_trans_descriptor_init(&blob, typesize, M, N, LDO);
    tr_kernel = libxsmm_dispatch_trans(tr_desc);
    PCL_ASSERT(tr_kernel, "Null Transpose kernel");
    return tr_kernel;
}

// void convert_f32_bf16(float* in, bfloat16* out, int len)
// {
//     /* Function to convert a Float32 array into a BFloat16 array */

//     int i = 0;

//     for (i = 0; i < len-16 + 1; i+=16 ) {
//         __m512  vfp32  = fp32_to_bfp16_rne_adjustment_avx512f( _mm512_loadu_ps( in+i ) );
//         __m256i vbfp16 = fp32_to_bfp16_truncate_avx512f( vfp32 );
//         _mm256_storeu_si256( (__m256i*)(out+i), vbfp16 );
//     }
//     if (i < len){
//         for (i = 0; i < len-15; i+=16 ) {
//             __mmask16 Msk = 0xFFFF  >> (16 - (len % 16));
//             __m512  vfp32  = fp32_to_bfp16_rne_adjustment_avx512f( _mm512_maskz_loadu_ps( Msk, (in+i) ) );
//             __m256i vbfp16 = fp32_to_bfp16_truncate_avx512f( vfp32 );
//             // _mm256_storeu_si256( (__m256i*)(out+i), vbfp16 );
//             _mm256_mask_storeu_epi16( (__m256i*)(out+i), Msk, vbfp16 );
//         }
//     }
// }

#endif

at::Tensor Conv1dOpti_forward_bf16_libxsmm(at::Tensor& input, at::Tensor& weight, int dilation){

    // RECORD_FUNCTION("Conv1dOpti_forward_bf16", std::vector<c10::IValue>({input, weight}));        // For recording time

    int64_t N_t = input.size(0);                    // Batch
    int64_t C_t = input.size(1);                    // Channel
    int64_t Win_t = input.size(2);                  // input width

    int64_t F_t = weight.size(0);                   // Number of filters
    int64_t WW_t = weight.size(2);                  // filter width

    int64_t dial = dilation;                        // dilation parameter
    int64_t pad_size = ((WW_t- 1))*dial;            // Total padding size
    int64_t W_t = Win_t - pad_size;                 // output width

    auto Y = input.new_empty({N_t,F_t,W_t});        // New tensor for output

    libxsmm_bfloat16* input_a = (libxsmm_bfloat16*) input.data_ptr<at::BFloat16>();       // Get BFloat16 data pointers for accessing tensors
    libxsmm_bfloat16* weight_a = (libxsmm_bfloat16*) weight.data_ptr<at::BFloat16>();
    libxsmm_bfloat16* Y_a = (libxsmm_bfloat16*) Y.data_ptr<at::BFloat16>();

    auto flip_weight = weight.new_empty({WW_t,F_t,C_t});                                // Weight tensor with permuted dimension (width, filters, channels)
    libxsmm_bfloat16* flip_weight_a = (libxsmm_bfloat16*) flip_weight.data_ptr<at::BFloat16>();    // Get BFloat16 data pointers for accessing the tensor

    for(int kw = 0; kw < WW_t; kw++){                               // Loop to permute weight tensor dimensions
        for(int filter=0; filter < F_t; filter++){
            for (int channel=0; channel < C_t; channel++){
                // permute dimensions
                flip_weight_a[kw*F_t*C_t + filter*C_t + channel] = weight_a[filter*C_t*WW_t + channel*WW_t + kw];
            }
        }
    }

    int lda = C_t;                      // Input channels (16)
    int ldb = Win_t;                    // Input width    (60400)
    int ldc = W_t;                      // Output width   (60000)
    unsigned long long l_br = WW_t;     // Number of batches in brGEMM (= width of kernel = 51)

    int tile_multiple = (W_t/XS_TILE_FORWARD)*XS_TILE_FORWARD;              // Number of blocks/Tiles in the output width

    int short_width = ((XS_TILE_FORWARD + (WW_t-1)*dial)/XS_TILE_FORWARD + 1)*XS_TILE_FORWARD;          // width of short buffer
    auto input_shortvnni = input.new_empty({N_t,C_t,short_width});                                      // VNNI transformed array of the short buffer
    libxsmm_bfloat16* input_a_shortvnni = (libxsmm_bfloat16*) input_shortvnni.data_ptr<at::BFloat16>(); // Get pointer


    int edge_width = (((W_t - tile_multiple) + (WW_t-1)*dial)/XS_TILE_FORWARD + 1)*XS_TILE_FORWARD;     // width of buffer in the edge case (last block)
    auto input_edgevnni = input.new_empty({N_t,C_t,edge_width});                                        // VNNI VNNI transformed array of the edge buffer
    libxsmm_bfloat16* input_a_edgevnni = (libxsmm_bfloat16*) input_edgevnni.data_ptr<at::BFloat16>();   // Get pointer


    /* Dispatch brGEMM kernels for the normal case and the edge case*/
    libxsmm_bmmfunction_reducebatch_strd bmmshortkernel = libxsmm_bmmdispatch_reducebatch_strd(XS_TILE_FORWARD, F_t, C_t, dial*2*sizeof(libxsmm_bfloat16), F_t*C_t*sizeof(libxsmm_bfloat16), &short_width, &lda, &ldc, NULL, NULL, NULL, NULL);
    libxsmm_bmmfunction_reducebatch_strd bmmedgekernel2 = libxsmm_bmmdispatch_reducebatch_strd(W_t - tile_multiple, F_t, C_t, dial*2*sizeof(libxsmm_bfloat16), F_t*C_t*sizeof(libxsmm_bfloat16), &edge_width, &lda, &ldc, NULL, NULL, NULL, NULL);

#ifdef USE_TPP                                  // When using TPP kernel for initialization
    /* Also JIT eltwise TPPs... */
    libxsmm_blasint tpp_m = XS_TILE_FORWARD;                      // rows
    libxsmm_blasint tpp_n = F_t;      // columns
    libxsmm_blasint ld_zero = W_t;

    libxsmm_meltwfunction_copy copy_kernel = libxsmm_dispatch_meltw_copy(tpp_m, tpp_n, NULL, &ld_zero, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_COPY_ZERO);
    if ( copy_kernel == NULL ) {
        fprintf( stderr, "JIT for initialization by TPP copy kernel failed. Bailing...!\n");
        exit(-1);
    }

#endif

#ifdef USE_TPP                                  // When using TPP kernel for VNNI transform
    /* use jited VNNI */
    libxsmm_blasint ldi = Win_t;
    libxsmm_blasint ldo_short = short_width;
    libxsmm_blasint ldo_edge = edge_width;

    libxsmm_meltw_transform_flags trans_vnni_flags;
    if ( C_t % 2 == 1 ) {
        trans_vnni_flags = LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_VNNI_PAD;
    } else {
        trans_vnni_flags = LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_VNNI;
    }

    libxsmm_meltwfunction_transform trans_shortvnni_kernel = libxsmm_dispatch_meltw_transform((XS_TILE_FORWARD + dial*(WW_t-1)), C_t, &ldi, &ldo_short, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, trans_vnni_flags);
    libxsmm_meltwfunction_transform trans_edgevnni_kernel = libxsmm_dispatch_meltw_transform((W_t - tile_multiple + dial*(WW_t-1)), C_t, &ldi, &ldo_edge, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, trans_vnni_flags);
    if ( trans_shortvnni_kernel == NULL | trans_edgevnni_kernel == NULL) {
        fprintf( stderr, "JIT for NORM_TO_VNNI TPP. Bailing...!\n");
        exit(-1);
    }

#endif

    #pragma omp parallel for
    for(int n = 0; n < N_t; n++) {                               // Loop for batches
        int last_block = 0;
#ifdef USE_TPP
        libxsmm_meltw_copy_param copy_params;           // Copy parameter variable for holding the pointer
        libxsmm_meltw_transform_param trans_param_short;
        libxsmm_meltw_transform_param trans_param_edge;
#endif
        for(int wb = 0; wb < W_t - XS_TILE_FORWARD + 1; wb += XS_TILE_FORWARD) {    // width blocking loop (Normal case)

#ifndef USE_TPP                                     // Not using TPP kernel
            for(int filter = 0; filter < F_t; filter++){                            /* Loop for initialization of output array */
                for(int out_w = wb; out_w < (wb + XS_TILE_FORWARD); out_w++){
                    Y_a[n*F_t*W_t + filter*W_t + out_w] = 0;
                }
            }
            // VNNI transform
            bf16_vnni_reformat(&input_a[n*C_t*Win_t + 0*Win_t + wb], &input_a_shortvnni[n*C_t*short_width], (XS_TILE_FORWARD + dial*(WW_t-1)), C_t, Win_t, short_width);
#else
            copy_params.out_ptr = &Y_a[n*F_t*W_t + wb];                 /* Initialization of output array */
            copy_kernel(&copy_params);

            // VNNI transform
            trans_param_short.in_ptr  = &input_a[n*C_t*Win_t + 0*Win_t + wb];
            trans_param_short.out_ptr = &input_a_shortvnni[n*C_t*short_width];
            trans_shortvnni_kernel( &trans_param_short );
#endif
            // VNNI transform and brGEMM
            // bf16_vnni_reformat(&input_a[n*C_t*Win_t + 0*Win_t + wb], &input_a_shortvnni[n*C_t*short_width], (XS_TILE_FORWARD + dial*(WW_t-1)), C_t, Win_t, short_width);
            bmmshortkernel(&input_a_shortvnni[n*C_t*short_width], &flip_weight_a[0], &Y_a[n*F_t*W_t + 0*W_t + wb], &l_br);

            last_block = wb;        // Store value for last block
        }

        if (W_t % XS_TILE_FORWARD != 0){                       // Edge case
            for(int filter = 0; filter < F_t; filter++){       /* Loop for initialization of output array */
                for(int out_w = last_block + XS_TILE_FORWARD; out_w < W_t; out_w++){
                    Y_a[n*F_t*W_t + filter*W_t + out_w] = 0;
                }
            }
#ifndef USE_TPP
            // VNNI transform
            bf16_vnni_reformat(&input_a[n*C_t*Win_t + 0*Win_t + (last_block + XS_TILE_FORWARD)], &input_a_edgevnni[n*C_t*edge_width], (W_t - tile_multiple + dial*(WW_t-1)), C_t, Win_t, edge_width);
#else
            // VNNI transform
            trans_param_edge.in_ptr  = &input_a[n*C_t*Win_t + 0*Win_t + (last_block + XS_TILE_FORWARD)];
            trans_param_edge.out_ptr = &input_a_edgevnni[n*C_t*edge_width];
            trans_edgevnni_kernel( &trans_param_edge );
#endif
            // bf16_vnni_reformat(&input_a[n*C_t*Win_t + 0*Win_t + (last_block + XS_TILE_FORWARD)], &input_a_edgevnni[n*C_t*edge_width], (W_t - tile_multiple + dial*(WW_t-1)), C_t, Win_t, edge_width);
            bmmedgekernel2(&input_a_edgevnni[n*C_t*edge_width], &flip_weight_a[0], &Y_a[n*F_t*W_t + 0*W_t + (last_block + XS_TILE_FORWARD)], &l_br);
        }
    }

    return Y;              // Return output tensor
}

std::tuple<at::Tensor, at::Tensor> Conv1dOpti_backward_bf16_libxsmm(at::Tensor& grad, at::Tensor& input, at::Tensor& weight, int dilation){

    // RECORD_FUNCTION("Conv1dOpti_backward_bf16", std::vector<c10::IValue>({grad, input, weight}));        // For recording time

    int64_t N_t = input.size(0);                    // Batch
    int64_t C_t = input.size(1);                    // Channel
    int64_t Win_t = input.size(2);                  // input width

    int64_t F_t = weight.size(0);                   // Number of filters
    int64_t WW_t = weight.size(2);                  // filter width

    int64_t dial = dilation;                        // dilation parameter
    int64_t pad_size = ((WW_t- 1))*dial;            // Total padding size
    int64_t W_t = Win_t - pad_size;                 // output width

    auto d_input = input.new_empty({N_t,C_t,Win_t});            // declare data gradiant tensor
    auto d_weight = weight.new_empty({F_t,C_t,WW_t});           // declare weight gradiant tensor

    libxsmm_bfloat16* input_a = (libxsmm_bfloat16*) input.data_ptr<at::BFloat16>();         // Get BFloat16 data pointers for accessing tensors
    libxsmm_bfloat16* weight_a = (libxsmm_bfloat16*) weight.data_ptr<at::BFloat16>();
    libxsmm_bfloat16* grad_a = (libxsmm_bfloat16*) grad.data_ptr<at::BFloat16>();
    libxsmm_bfloat16* d_input_a = (libxsmm_bfloat16*) d_input.data_ptr<at::BFloat16>();
    libxsmm_bfloat16* d_weight_a = (libxsmm_bfloat16*) d_weight.data_ptr<at::BFloat16>();

    /* Backward Data part of the code */

    auto flip_weight_tensor = weight.new_empty({WW_t,C_t,F_t});                             // Weight tensor with permuted dimension (width, channels, filters)
    libxsmm_bfloat16* flip_weight_a = (libxsmm_bfloat16*) flip_weight_tensor.data_ptr<at::BFloat16>();   // Get pointer

    for(int kw = 0; kw < WW_t; kw++){                       // Loop to permute and flip weight tensor
        for (int channel=0; channel < C_t; channel++){
            for(int filter=0; filter < F_t; filter++){
                // permute and flip the kernel
                flip_weight_a[kw*F_t*C_t + channel*F_t + filter] = weight_a[filter*C_t*WW_t + channel*WW_t + WW_t - kw - 1];
            }
        }
    }

    int64_t Wpad_t = W_t + 2*(WW_t - 1)*dial;                             // For padding gradiant on both sides
    int64_t tile_multiple = (Win_t/XS_TILE_DBACKWARD)*XS_TILE_DBACKWARD;  // Number of blocks/tiles in Input

    int lda = F_t;                      // Number of Filters (16)
    int ldb_orig = W_t;                 // Output width (60000)
    int ldb = Wpad_t;                   // Extra padded grad input case 60800
    int ldc = Win_t;                    // Input width (60400)
    unsigned long long l_br = WW_t;     // Number of batches in brGEMM (= width of kernel = 51)

    int pad_tile_multiple = 2 * (((WW_t - 1)*dial)/XS_TILE_DBACKWARD + 1) * XS_TILE_DBACKWARD;       // Padded block/tile (896)
    int ldb_shortpad = 2*pad_tile_multiple;       // grad padded short buffer (1792)

    auto grad_shortpad_tensor = grad.new_empty({N_t,F_t,2*pad_tile_multiple});
    libxsmm_bfloat16* grad_a_shortpad = (libxsmm_bfloat16*) grad_shortpad_tensor.data_ptr<at::BFloat16>();   // short buffer for padded gradiant

    #pragma omp parallel for
    for(int n = 0; n < N_t; n++){                   // loop to store the edges for gradiant array into grad_a_shortpad buffer
        for(int filter=0; filter < F_t; filter++){
            for(int w = 0; w < pad_tile_multiple; w++){
                // initialize start of array
                if (w >= ((WW_t - 1)*dial) && w < (W_t + (WW_t - 1)*dial)){
                    grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w] = grad_a[n*F_t*W_t + filter*W_t + w - (WW_t - 1)*dial];
                }
                else{
                    grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w] = 0.0f;
                }
            }
            for(int w = Wpad_t - pad_tile_multiple; w < Wpad_t ; w++){
                // initialize end of array
                if (w >= ((WW_t - 1)*dial) && w < (W_t + (WW_t - 1)*dial)){
                    grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w - Wpad_t + 2*pad_tile_multiple] = grad_a[n*F_t*W_t + filter*W_t + w - (WW_t - 1)*dial];
                }
                else{
                    grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w - Wpad_t + 2*pad_tile_multiple] = 0.0f;
                }
            }
        }
    }

    int short_width = ((XS_TILE_DBACKWARD + (WW_t-1)*dial)/XS_TILE_DBACKWARD + 1)*XS_TILE_DBACKWARD;    // Width of buffer

    auto grad_shortvnni_tensor = grad.new_empty({N_t,F_t,short_width});                                 // Buffer for storing VNNI transform
    libxsmm_bfloat16* grad_a_shortvnni = (libxsmm_bfloat16*) grad_shortvnni_tensor.data_ptr<at::BFloat16>();

    /* Dispatch brGEMM kernels for the normal case and the edge case*/
    libxsmm_bmmfunction_reducebatch_strd bmmshortkernel = libxsmm_bmmdispatch_reducebatch_strd(XS_TILE_DBACKWARD, C_t, F_t, 2*dial*sizeof(libxsmm_bfloat16), C_t*F_t*sizeof(libxsmm_bfloat16), &short_width, &lda, &ldc, NULL, NULL, NULL, NULL);
    libxsmm_bmmfunction_reducebatch_strd bmmshortkernel2 = libxsmm_bmmdispatch_reducebatch_strd(Win_t - tile_multiple, C_t, F_t, 2*dial*sizeof(libxsmm_bfloat16), C_t*F_t*sizeof(libxsmm_bfloat16), &short_width, &lda, &ldc, NULL, NULL, NULL, NULL);

#ifdef USE_TPP                                  // When using TPP kernel for initialization
    /* Also JIT eltwise TPPs... */
    libxsmm_blasint tpp_m = XS_TILE_DBACKWARD;                      // rows
    libxsmm_blasint tpp_n = C_t;      // columns
    libxsmm_blasint ld_zero = Win_t;

    libxsmm_meltwfunction_copy copy_kernel = libxsmm_dispatch_meltw_copy(tpp_m, tpp_n, NULL, &ld_zero, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_COPY_ZERO);
    if ( copy_kernel == NULL ) {
        fprintf( stderr, "JIT for initialization by TPP copy kernel failed. Bailing...!\n");
        exit(-1);
    }

#endif


#ifdef USE_TPP                                  // When using TPP kernel for VNNI transform
    /* use jited VNNI */
    libxsmm_blasint ldi_1 = W_t;
    libxsmm_blasint ldi_2 = ldb_shortpad;
    libxsmm_blasint ldo = short_width;

    libxsmm_meltw_transform_flags trans_vnni_flags;
    if ( F_t % 2 == 1 ) {
        trans_vnni_flags = LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_VNNI_PAD;
    } else {
        trans_vnni_flags = LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_VNNI;
    }

    libxsmm_meltwfunction_transform trans_shortvnni_kernel_1 = libxsmm_dispatch_meltw_transform((XS_TILE_DBACKWARD + dial*(WW_t-1)), F_t, &ldi_1, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, trans_vnni_flags);
    libxsmm_meltwfunction_transform trans_shortvnni_kernel_2 = libxsmm_dispatch_meltw_transform((XS_TILE_DBACKWARD + dial*(WW_t-1)), F_t, &ldi_2, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, trans_vnni_flags);
    if ( trans_shortvnni_kernel_1 == NULL | trans_shortvnni_kernel_2 == NULL) {
        fprintf( stderr, "JIT for NORM_TO_VNNI TPP. Bailing...!\n");
        exit(-1);
    }

#endif

    #pragma omp parallel for
    for(int n = 0; n < N_t; n++) {
        int last_block=0;
#ifdef USE_TPP
        libxsmm_meltw_copy_param copy_params;                       // Copy parameter variable for holding the pointer
        libxsmm_meltw_transform_param trans_param_1;
        libxsmm_meltw_transform_param trans_param_2;
#endif
        for(int wb = 0; wb < Win_t - XS_TILE_DBACKWARD + 1; wb += XS_TILE_DBACKWARD) {

#ifndef USE_TPP
            for(int channel = 0; channel < C_t; channel++){                 /* Loop for initialization of grad array */
                for(int in_w = wb; in_w < wb + XS_TILE_DBACKWARD; in_w++){
                    d_input_a[n*C_t*Win_t + channel*Win_t + in_w] = 0;
                }
            }
#else
            copy_params.out_ptr = &d_input_a[n*C_t*Win_t + wb];
            copy_kernel(&copy_params);
#endif
            if (wb >= (WW_t-1)*dial && wb < Win_t - (WW_t-1)*dial - XS_TILE_DBACKWARD){
                // Normal case (Take VNNI transform of a portion of grad_a array )
#ifndef USE_TPP
                // VNNI transform
                bf16_vnni_reformat(&grad_a[n*F_t*W_t + 0*W_t + wb - (WW_t-1)*dial], &grad_a_shortvnni[n*F_t*short_width], (XS_TILE_DBACKWARD + dial*(WW_t-1)), F_t, W_t, short_width);
#else
                // VNNI transform
                trans_param_1.in_ptr  = &grad_a[n*F_t*W_t + 0*W_t + wb - (WW_t-1)*dial];
                trans_param_1.out_ptr = &grad_a_shortvnni[n*F_t*short_width];
                trans_shortvnni_kernel_1( &trans_param_1 );
#endif
                // bf16_vnni_reformat(&grad_a[n*F_t*W_t + 0*W_t + wb - (WW_t-1)*dial], &grad_a_shortvnni[n*F_t*short_width], (XS_TILE_DBACKWARD + dial*(WW_t-1)), F_t, W_t, short_width);
                bmmshortkernel(&grad_a_shortvnni[n*F_t*short_width], &flip_weight_a[0], &d_input_a[n*C_t*Win_t + 0*Win_t + wb], &l_br);
            }
            else if (wb < (WW_t-1)*dial){
                // Right side case (Take VNNI transform of grad_a_shortpad array)
#ifndef USE_TPP
                // VNNI transform
                bf16_vnni_reformat(&grad_a_shortpad[n*F_t*2*pad_tile_multiple + wb], &grad_a_shortvnni[n*F_t*short_width], (XS_TILE_DBACKWARD + dial*(WW_t-1)), F_t, ldb_shortpad, short_width);
#else
                // VNNI transform
                trans_param_2.in_ptr  = &grad_a_shortpad[n*F_t*2*pad_tile_multiple + wb];
                trans_param_2.out_ptr = &grad_a_shortvnni[n*F_t*short_width];
                trans_shortvnni_kernel_2( &trans_param_2 );
#endif
                // bf16_vnni_reformat(&grad_a_shortpad[n*F_t*2*pad_tile_multiple + wb], &grad_a_shortvnni[n*F_t*short_width], (XS_TILE_DBACKWARD + dial*(WW_t-1)), F_t, ldb_shortpad, short_width);
                bmmshortkernel(&grad_a_shortvnni[n*F_t*short_width], &flip_weight_a[0], &d_input_a[n*C_t*Win_t + 0*Win_t + wb], &l_br);
            }
            else{
                // Left side case (Take VNNI transform of grad_a_shortpad array)
#ifndef USE_TPP
                // VNNI transform
                bf16_vnni_reformat(&grad_a_shortpad[n*F_t*2*pad_tile_multiple + wb - Wpad_t + 2*pad_tile_multiple], &grad_a_shortvnni[n*F_t*short_width], (XS_TILE_DBACKWARD + dial*(WW_t-1)), F_t, ldb_shortpad, short_width);
#else
                // VNNI transform
                trans_param_2.in_ptr  = &grad_a_shortpad[n*F_t*2*pad_tile_multiple + wb - Wpad_t + 2*pad_tile_multiple];
                trans_param_2.out_ptr = &grad_a_shortvnni[n*F_t*short_width];
                trans_shortvnni_kernel_2( &trans_param_2 );
#endif
                // bf16_vnni_reformat(&grad_a_shortpad[n*F_t*2*pad_tile_multiple + wb - Wpad_t + 2*pad_tile_multiple], &grad_a_shortvnni[n*F_t*short_width], (XS_TILE_DBACKWARD + dial*(WW_t-1)), F_t, ldb_shortpad, short_width);
                bmmshortkernel(&grad_a_shortvnni[n*F_t*short_width], &flip_weight_a[0], &d_input_a[n*C_t*Win_t + 0*Win_t + wb], &l_br);
            }
            last_block = wb;
        }

        if (Win_t % XS_TILE_DBACKWARD != 0){                                // Edge case
            for(int channel = 0; channel < C_t; channel++){                 /* Loop for initialization of grad array */
                for(int in_w = last_block + XS_TILE_DBACKWARD; in_w < Win_t; in_w++){
                    d_input_a[n*C_t*Win_t + channel*Win_t + in_w] = 0;
                }
            }
            // Right side case (Take VNNI transform of grad_a_shortpad array)
#ifndef USE_TPP
            // VNNI transform
            bf16_vnni_reformat(&grad_a_shortpad[n*F_t*2*pad_tile_multiple + last_block + XS_TILE_DBACKWARD - Wpad_t + 2*pad_tile_multiple], &grad_a_shortvnni[n*F_t*short_width], (XS_TILE_DBACKWARD + dial*(WW_t-1)), F_t, ldb_shortpad, short_width);
#else
            // VNNI transform
            trans_param_2.in_ptr  = &grad_a_shortpad[n*F_t*2*pad_tile_multiple + last_block + XS_TILE_DBACKWARD - Wpad_t + 2*pad_tile_multiple];
            trans_param_2.out_ptr = &grad_a_shortvnni[n*F_t*short_width];
            trans_shortvnni_kernel_2( &trans_param_2 );
#endif
            // bf16_vnni_reformat(&grad_a_shortpad[n*F_t*2*pad_tile_multiple + last_block + XS_TILE_DBACKWARD - Wpad_t + 2*pad_tile_multiple], &grad_a_shortvnni[n*F_t*short_width], (XS_TILE_DBACKWARD + dial*(WW_t-1)), F_t, ldb_shortpad, short_width);
            bmmshortkernel2(&grad_a_shortvnni[n*F_t*short_width], &flip_weight_a[0], &d_input_a[n*C_t*Win_t + last_block + XS_TILE_DBACKWARD], &l_br);
        }
    }


    /* Backward Weight part of the code */


    float* flip_d_weight_a = (float*) libxsmm_aligned_malloc( F_t*C_t*WW_t*sizeof(float), 64 );             // Array for permuted weight gradiant

    for(int w = 0; w < F_t*C_t*WW_t; w++){          // Initialize array
        flip_d_weight_a[w] = 0.0f;
    }

    // lda = W_t;                   // Already defined variables
    // ldb = Win_t;
    // ldc = C_t;
    l_br = WW_t;                    // Number of batches in brGEMM (= width of kernel = 51)
    tile_multiple = (W_t/XS_TILE_WBACKWARD)*XS_TILE_WBACKWARD;


    // Blocking on grad_a
    int lda_g = Win_t;
    int ldb_trans_g = F_t;
    int ldc_g = F_t;

#ifndef USE_TPP
    unsigned int M_g = W_t/2;       // Output rows
    unsigned int N_g = F_t;         // Output columns
    int short_W_t = XS_TILE_WBACKWARD;
    int edge_W_t = W_t - tile_multiple;
#else
    libxsmm_blasint M_g = W_t/2;
    libxsmm_blasint N_g = F_t;
    libxsmm_blasint short_W_t = XS_TILE_WBACKWARD;
    libxsmm_blasint edge_W_t = W_t - tile_multiple;
#endif

    auto grad_shortvnni_tensor2 = grad.new_empty({N_t,F_t,short_W_t});                            // Short buffer for storing VNNI transform
    libxsmm_bfloat16* grad_shortvnni = (libxsmm_bfloat16*) grad_shortvnni_tensor2.data_ptr<at::BFloat16>();

    auto grad_edgevnni_tensor2 = grad.new_empty({N_t,F_t,edge_W_t});                              // Short buffer for storing VNNI transform in edge case
    libxsmm_bfloat16* grad_edgevnni = (libxsmm_bfloat16*) grad_edgevnni_tensor2.data_ptr<at::BFloat16>();

#ifndef USE_TPP                     // When Not using TPP kernels

    libxsmm_xtransfunction trans_shortkernel_grad = get_tr_kernel(sizeof(float), short_W_t/2, F_t, F_t);      // Dispatch transpose kernel
    libxsmm_xtransfunction trans_edgekernel_grad = get_tr_kernel(sizeof(float), edge_W_t/2, F_t, F_t);        // Dispatch transpose kernel
#else
    /* use jited tranpose */
    libxsmm_meltw_transform_flags trans_flags;
    trans_flags = LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_NORMT;
    libxsmm_meltwfunction_transform trans_shortkernel_grad = libxsmm_dispatch_meltw_transform(short_W_t/2, N_g, &M_g, &N_g, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, trans_flags);
    libxsmm_meltwfunction_transform trans_edgekernel_grad = libxsmm_dispatch_meltw_transform(edge_W_t/2, N_g, &M_g, &N_g, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, trans_flags);
    if ( trans_shortkernel_grad == NULL | trans_edgekernel_grad == NULL) {
        fprintf( stderr, "JIT for NORM_TO_NORMT TPP. Bailing...!\n");
        exit(-1);
    }

#endif
    /* Dispatch brGEMM kernels for the normal case and the edge case*/
    libxsmm_bsmmfunction bsmmkernel5 = libxsmm_bsmmdispatch(F_t, C_t, XS_TILE_WBACKWARD, &ldb_trans_g, &lda_g, &ldc_g, NULL, NULL, NULL, NULL);
    libxsmm_bsmmfunction bsmmkernel6 = libxsmm_bsmmdispatch(F_t, C_t, W_t - tile_multiple, &ldb_trans_g, &lda_g, &ldc_g, NULL, NULL, NULL, NULL);

    #pragma omp parallel for reduction(+: flip_d_weight_a[:F_t*C_t*WW_t])
    for(int n = 0; n < N_t; n++) {
        int last_block = 0;
#ifdef USE_TPP
        libxsmm_meltw_transform_param trans_param_short;
        libxsmm_meltw_transform_param trans_param_edge;
#endif
        for(int wb = 0; wb < W_t - XS_TILE_WBACKWARD + 1; wb += XS_TILE_WBACKWARD) {            // Normal Case

#ifndef USE_TPP
            /* Take transpose assumping FP32 (This will do both transpose and VNNI transform for BF16) */
            trans_shortkernel_grad(&grad_a[n*F_t*W_t + wb], &M_g, &grad_shortvnni[n*F_t*short_W_t], &N_g);
#else
            /* Take transpose assumping FP32 (This will do both transpose and VNNI transform for BF16) */
            trans_param_short.in_ptr  = &grad_a[n*F_t*W_t + wb];
            trans_param_short.out_ptr = &grad_shortvnni[n*F_t*short_W_t];
            trans_shortkernel_grad( &trans_param_short );
#endif

            for(int kw = 0; kw < WW_t; kw++) {
                // libxsmm_bsmmfunction bsmmkernel5 = libxsmm_bsmmdispatch(F_t, C_t, XS_TILE_WBACKWARD, &ldb_trans_g, &lda_g, &ldc_g, NULL, NULL, NULL, NULL);
                bsmmkernel5(&grad_shortvnni[n*F_t*short_W_t], &input_a[n*C_t*Win_t + wb + kw*dial], &flip_d_weight_a[kw*C_t*F_t]);
            }
            last_block = wb;
        }

        if (W_t % XS_TILE_WBACKWARD != 0){              // Edge Case

#ifndef USE_TPP
            trans_edgekernel_grad(&grad_a[n*F_t*W_t + (last_block + XS_TILE_WBACKWARD)], &M_g, &grad_edgevnni[n*F_t*edge_W_t], &N_g);
#else
            trans_param_edge.in_ptr  = &grad_a[n*F_t*W_t + last_block + XS_TILE_WBACKWARD];
            trans_param_edge.out_ptr = &grad_edgevnni[n*F_t*edge_W_t];
            trans_edgekernel_grad( &trans_param_edge );
#endif
            for(int kw = 0; kw < WW_t; kw++) {
                // libxsmm_bsmmfunction bsmmkernel6 = libxsmm_bsmmdispatch(F_t, C_t, W_t - tile_multiple, &ldb_trans_g, &lda_g, &ldc_g, NULL, NULL, NULL, NULL);
                bsmmkernel6(&grad_edgevnni[n*F_t*edge_W_t], &input_a[n*C_t*Win_t + (last_block + XS_TILE_WBACKWARD) + kw*dial], &flip_d_weight_a[kw*F_t*C_t]);
            }
        }
    }


    auto flip_d_weight_tensor = weight.new_empty({WW_t,C_t,F_t});
    libxsmm_bfloat16* flip_d_weight_bf16 = (libxsmm_bfloat16*) flip_d_weight_tensor.data_ptr<at::BFloat16>();

// #ifndef USE_TPP
    // convert_f32_bf16(flip_d_weight_a, flip_d_weight_bf16, F_t*C_t*WW_t);            // Convert FP32 weight gradiant array into BF16 weight gradiant (permuted) array
// #else
    /* Also JIT eltwise TPPs... */
    libxsmm_blasint cvt_m = 1;
    libxsmm_blasint cvt_n = F_t*C_t*WW_t;

    libxsmm_meltwfunction_cvtfp32bf16 eltwise_kernel = libxsmm_dispatch_meltw_cvtfp32bf16(cvt_m, cvt_n, NULL, NULL, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_CVT_NONE);
    if ( eltwise_kernel == NULL ) {
        fprintf( stderr, "JIT for TPP convert FP32 to BF16 failed. Bailing...!\n");
        exit(-1);
    }
    libxsmm_meltw_cvtfp32bf16_param eltwise_params;

    eltwise_params.in_ptr = &flip_d_weight_a[0];
    eltwise_params.out_ptr = &flip_d_weight_bf16[0];
    eltwise_kernel(&eltwise_params);

// #endif

    for(int kw = 0; kw < WW_t; kw++){                       // permute and write to weight gradiant array for return
        for(int filter=0; filter < F_t; filter++){
            for (int channel=0; channel < C_t; channel++){
                d_weight_a[filter*C_t*WW_t + channel*WW_t + kw] = flip_d_weight_bf16[kw*C_t*F_t + channel*F_t + filter];
            }
        }
    }

    libxsmm_free(flip_d_weight_a);

    return {d_input, d_weight};
}


at::Tensor Conv1dOpti_forward_libxsmm(at::Tensor& input, at::Tensor& weight, int dilation){

    // RECORD_FUNCTION("Conv1dOpti_forward_libxsmm", std::vector<c10::IValue>({input, weight}));    // For recording time

    int64_t N_t = input.size(0);                    // Batch
    int64_t C_t = input.size(1);                    // Channel
    int64_t Win_t = input.size(2);                  // input width

    int64_t F_t = weight.size(0);                   // Number of filters
    int64_t WW_t = weight.size(2);                  // filter width

    int64_t dial = dilation;                        // dilation parameter
    int64_t pad_size = ((WW_t- 1))*dial;            // Total padding size
    int64_t W_t = Win_t - pad_size;                 // output width

    auto Y = input.new_empty({N_t,F_t,W_t});        // New tensor for output

    float* input_a = input.data_ptr<float>();       // Get pointers for accessing the tensors
    float* weight_a = weight.data_ptr<float>();
    float* Y_a = Y.data_ptr<float>();

    auto flip_weight = weight.new_empty({WW_t,F_t,C_t});        // Array to store permuted weight tensor (width, filters, channels)
    float* flip_weight_a = flip_weight.data_ptr<float>();

    for(int kw = 0; kw < WW_t; kw++){                           // Loops to permute the array dimensions
        for(int filter=0; filter < F_t; filter++){
            for (int channel=0; channel < C_t; channel++){
                // permute dimensions
                flip_weight_a[kw*F_t*C_t + filter*C_t + channel] = weight_a[filter*C_t*WW_t + channel*WW_t + kw];
            }
        }
    }

    int lda = C_t;                      // Input channels (15)
    int ldb = Win_t;                    // Input width (60400)
    int ldc = W_t;                      // Output width (60000)
    unsigned long long l_br = WW_t;

    int tile_multiple = (W_t/XS_TILE_FORWARD)*XS_TILE_FORWARD;

    /* Dispatch brGEMM kernels for the normal case and the edge case*/
    libxsmm_smmfunction_reducebatch_strd kernel = libxsmm_smmdispatch_reducebatch_strd(XS_TILE_FORWARD, F_t, C_t, dial*sizeof(float), F_t*C_t*sizeof(float), &ldb, &lda, &ldc, NULL, NULL, NULL, NULL);
    libxsmm_smmfunction_reducebatch_strd kernel2 = libxsmm_smmdispatch_reducebatch_strd(W_t - tile_multiple, F_t, C_t, dial*sizeof(float), F_t*C_t*sizeof(float), &ldb, &lda, &ldc, NULL, NULL, NULL, NULL);

#ifdef USE_TPP                                  // When using TPP kernel for initialization
    /* Also JIT eltwise TPPs... */
    libxsmm_blasint tpp_m = XS_TILE_FORWARD;                      // columns
    libxsmm_blasint tpp_n = F_t;      // rows
    libxsmm_blasint ld_zero = W_t;

    libxsmm_meltwfunction_copy copy_kernel = libxsmm_dispatch_meltw_copy(tpp_m, tpp_n, NULL, &ld_zero, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_COPY_ZERO);
    if ( copy_kernel == NULL ) {
        fprintf( stderr, "JIT for initialization by TPP copy kernel failed. Bailing...!\n");
        exit(-1);
    }

#endif

    #pragma omp parallel for
    for(int n = 0; n < N_t; n++) {                               // Loop for batches
        int last_block = 0;
#ifdef USE_TPP
        libxsmm_meltw_copy_param copy_params;           // Copy parameter variable for holding the pointer
#endif
        for(int wb = 0; wb < W_t - XS_TILE_FORWARD + 1; wb += XS_TILE_FORWARD) {    // width blocking loop (Normal case)

#ifndef USE_TPP                                                 // If not using TPP
            for(int filter = 0; filter < F_t; filter++){       /* Loop for initialization of output array */
                for(int out_w = wb; out_w < (wb + XS_TILE_FORWARD); out_w++){
                    Y_a[n*F_t*W_t + filter*W_t + out_w] = 0.0f;
                }
            }
#else
            copy_params.out_ptr = &Y_a[n*F_t*W_t + wb];
            copy_kernel(&copy_params);
#endif
            kernel(&input_a[n*C_t*Win_t + 0*Win_t + wb], &flip_weight_a[0], &Y_a[n*F_t*W_t + 0*W_t + wb], &l_br);
            last_block = wb;
        }

        if (W_t % XS_TILE_FORWARD != 0){                        // Edge Case
            for(int filter = 0; filter < F_t; filter++){       /* Loop for initialization of output array */
                for(int out_w = last_block + XS_TILE_FORWARD; out_w < W_t; out_w++){
                    Y_a[n*F_t*W_t + filter*W_t + out_w] = 0.0f;
                }
            }
            kernel2(&input_a[n*C_t*Win_t + 0*Win_t + last_block + XS_TILE_FORWARD], &flip_weight_a[0], &Y_a[n*F_t*W_t + 0*W_t + last_block + XS_TILE_FORWARD], &l_br);
        }
    }

    return Y;           // Return output array
}

std::tuple<at::Tensor, at::Tensor>
Conv1dOpti_backward_libxsmm(at::Tensor& grad, at::Tensor& input, at::Tensor& weight, int dilation){

    // RECORD_FUNCTION("Conv1dOpti_backward_libxsmm", std::vector<c10::IValue>({grad, input, weight}));

    int64_t N_t = input.size(0);                    // Batch
    int64_t C_t = input.size(1);                    // Channel
    int64_t Win_t = input.size(2);                  // input width

    int64_t F_t = weight.size(0);                   // Number of filters
    int64_t WW_t = weight.size(2);                  // filter width

    int64_t dial = dilation;                        // dilation parameter
    int64_t pad_size = ((WW_t- 1))*dial;            // Total padding size
    int64_t W_t = Win_t - pad_size;                 // output width

    auto d_input = input.new_empty({N_t,C_t,Win_t});            // declare data gradiant tensor
    auto d_weight = weight.new_empty({F_t,C_t,WW_t});           // declare weight gradiant tensor


    float* input_a = input.data_ptr<float>();                   // Get data pointers for accessing tensors
    float* weight_a = weight.data_ptr<float>();
    float* grad_a = grad.data_ptr<float>();
    float* d_input_a = d_input.data_ptr<float>();
    float* d_weight_a = d_weight.data_ptr<float>();

    /*  Backward data part of the code */

    auto flip_weight = weight.new_empty({WW_t,C_t,F_t});                  // Tensor for permuted weights (width, channels, filters)
    float* flip_weight_a = flip_weight.data_ptr<float>();

    for(int kw = 0; kw < WW_t; kw++){                   // Loop to permute and flip weights
        for (int channel=0; channel < C_t; channel++){
            for(int filter=0; filter < F_t; filter++){
                // permute and flip the kernel
                flip_weight_a[kw*F_t*C_t + channel*F_t + filter] = weight_a[filter*C_t*WW_t + channel*WW_t + WW_t - kw - 1];
            }
        }
    }

    int64_t Wpad_t = W_t + 2*(WW_t - 1)*dial;
    int64_t tile_multiple = (Win_t/XS_TILE_DBACKWARD)*XS_TILE_DBACKWARD;

    int lda = F_t;                   // Filters (15)
    int ldb_orig = W_t;              // grad width 60000
    int ldb = Wpad_t;                // Extra padded grad input case 60800
    int ldc = Win_t;                 // Input width (60400)
    unsigned long long l_br = WW_t;  // Number of batches for brGEMM (51)

    libxsmm_smmfunction_reducebatch_strd kernel = libxsmm_smmdispatch_reducebatch_strd(XS_TILE_DBACKWARD, C_t, F_t, dial*sizeof(float), C_t*F_t*sizeof(float), &ldb_orig, &lda, &ldc, NULL, NULL, NULL, NULL);

    int pad_tile_multiple = 2 * (((WW_t - 1)*dial)/XS_TILE_DBACKWARD + 1) * XS_TILE_DBACKWARD;       // 896

    auto grad_shortpad_tensor = grad.new_empty({N_t,F_t,2*pad_tile_multiple});
    float* grad_a_shortpad = grad_shortpad_tensor.data_ptr<float>();


    int ldb_shortpad = 2*pad_tile_multiple;       // grad pad 1792

    /* Dispatch kernels for normal and edge cases*/
    libxsmm_smmfunction_reducebatch_strd kernel4 = libxsmm_smmdispatch_reducebatch_strd(XS_TILE_DBACKWARD, C_t, F_t, dial*sizeof(float), C_t*F_t*sizeof(float), &ldb_shortpad, &lda, &ldc, NULL, NULL, NULL, NULL);
    libxsmm_smmfunction_reducebatch_strd kernel5 = libxsmm_smmdispatch_reducebatch_strd(Win_t - tile_multiple, C_t, F_t, dial*sizeof(float), C_t*F_t*sizeof(float), &ldb_shortpad, &lda, &ldc, NULL, NULL, NULL, NULL);


    #pragma omp parallel for
    for(int n = 0; n < N_t; n++){                       // Loops for storing the edge portion of gradinant array into grad_a_shortpad
        for(int filter=0; filter < F_t; filter++){
            for(int w = 0; w < pad_tile_multiple; w++){
                // initialize start of array
                if (w >= ((WW_t - 1)*dial) && w < (W_t + (WW_t - 1)*dial)){
                    grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w] = grad_a[n*F_t*W_t + filter*W_t + w - (WW_t - 1)*dial];
                }
                else{
                    grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w] = 0.0f;
                }
            }
            for(int w = Wpad_t - pad_tile_multiple; w < Wpad_t ; w++){
                // initialize end of array
                if (w >= ((WW_t - 1)*dial) && w < (W_t + (WW_t - 1)*dial)){
                    grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w - Wpad_t + 2*pad_tile_multiple] = grad_a[n*F_t*W_t + filter*W_t + w - (WW_t - 1)*dial];
                }
                else{
                    grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w - Wpad_t + 2*pad_tile_multiple] = 0.0f;
                }
            }
        }
    }

#ifdef USE_TPP                                  // When using TPP kernel for initialization
    /* Also JIT eltwise TPPs... */
    libxsmm_blasint tpp_m = XS_TILE_DBACKWARD;                      // rows
    libxsmm_blasint tpp_n = C_t;      // columns
    libxsmm_blasint ld_zero = Win_t;

    libxsmm_meltwfunction_copy copy_kernel = libxsmm_dispatch_meltw_copy(tpp_m, tpp_n, NULL, &ld_zero, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_COPY_ZERO);
    if ( copy_kernel == NULL ) {
        fprintf( stderr, "JIT for initialization by TPP copy kernel failed. Bailing...!\n");
        exit(-1);
    }

#endif

    #pragma omp parallel for
    for(int n = 0; n < N_t; n++) {
        int last_block=0;
#ifdef USE_TPP
        libxsmm_meltw_copy_param copy_params;           // Copy parameter variable for holding the pointer
#endif
        for(int wb = 0; wb < Win_t - XS_TILE_DBACKWARD + 1; wb += XS_TILE_DBACKWARD) {

#ifndef USE_TPP                                                             // Not using TPP kernel
            for(int channel = 0; channel < C_t; channel++){                 /* Loop for initialization of grad array */
                for(int in_w = wb; in_w < wb + XS_TILE_DBACKWARD; in_w++){
                    d_input_a[n*C_t*Win_t + channel*Win_t + in_w] = 0.0f;
                }
            }
#else
            copy_params.out_ptr = &d_input_a[n*C_t*Win_t + wb];
            copy_kernel(&copy_params);
#endif
            if (wb >= (WW_t-1)*dial && wb < Win_t - (WW_t-1)*dial - XS_TILE_DBACKWARD)              // Normal case
                kernel(&grad_a[n*F_t*W_t + 0*W_t + wb - (WW_t-1)*dial], &flip_weight_a[0], &d_input_a[n*C_t*Win_t + 0*Win_t + wb], &l_br);
            else if (wb < (WW_t-1)*dial)                // Right side case
                kernel4(&grad_a_shortpad[n*F_t*2*pad_tile_multiple + wb], &flip_weight_a[0], &d_input_a[n*C_t*Win_t + wb], &l_br);
            else             // left side case
                kernel4(&grad_a_shortpad[n*F_t*2*pad_tile_multiple + wb - Wpad_t + 2*pad_tile_multiple], &flip_weight_a[0], &d_input_a[n*C_t*Win_t + wb], &l_br);

            last_block = wb;     // store position for last block
        }

        if (Win_t % XS_TILE_DBACKWARD != 0){                                // Edge case
            for(int channel = 0; channel < C_t; channel++){                 /* Loop for initialization of grad array */
                for(int in_w = last_block + XS_TILE_DBACKWARD; in_w < Win_t; in_w++){
                    d_input_a[n*C_t*Win_t + channel*Win_t + in_w] = 0.0f;
                }
            }
            kernel5(&grad_a_shortpad[n*F_t*2*pad_tile_multiple + last_block + XS_TILE_DBACKWARD - Wpad_t + 2*pad_tile_multiple], &flip_weight_a[0], &d_input_a[n*C_t*Win_t + last_block + XS_TILE_DBACKWARD], &l_br);
        }
    }

    /* Backward weight part of the code  */


    auto flip_d_weight = weight.new_empty({WW_t,C_t,F_t});                  // Tensor for storing permuted weight gradiant
    float* flip_d_weight_a = flip_d_weight.data_ptr<float>();

    for(int w = 0; w < F_t*C_t*WW_t; w++){
        flip_d_weight_a[w] = 0.0f;
    }

    // lda = W_t;
    // ldb = Win_t;
    // int ldb_trans = C_t;
    // ldc = C_t;
    l_br = WW_t;
    tile_multiple = (W_t/XS_TILE_WBACKWARD)*XS_TILE_WBACKWARD;


    // Blocking on grad_a
    int lda_g = Win_t;
    // int ldb_g = W_t;
    int ldb_trans_g = F_t;
    int ldc_g = F_t;

#ifndef USE_TPP                                 // If not usintg TPP kernels
    unsigned int M_g = W_t;      //Output rows
    unsigned int N_g = F_t;    // Output columns
    int short_W_t = XS_TILE_WBACKWARD;
    int edge_W_t = W_t - tile_multiple;
#else
    libxsmm_blasint short_W_t = XS_TILE_WBACKWARD;
    libxsmm_blasint edge_W_t = W_t - tile_multiple;
    libxsmm_blasint M_g = W_t;
    libxsmm_blasint N_g = F_t;
#endif

    auto grad_shorttrans_tensor = grad.new_empty({N_t,F_t,short_W_t});              // Tensor for storing transposed short buffer
    float* grad_shorttrans = grad_shorttrans_tensor.data_ptr<float>();

    auto grad_edgetrans_tensor = grad.new_empty({N_t,F_t,edge_W_t});                // Tensor for storing transposed short buffer in edge case
    float* grad_edgetrans = grad_edgetrans_tensor.data_ptr<float>();

#ifndef USE_TPP                                 // If not usintg TPP kernels

    libxsmm_xtransfunction trans_shortkernel_grad = get_tr_kernel(sizeof(float), short_W_t, F_t, F_t);          // Dispatch transpose kernel
    libxsmm_xtransfunction trans_edgekernel_grad = get_tr_kernel(sizeof(float), edge_W_t, F_t, F_t);            // Dispatch transpose kernel
#else
    /* use jited tranpose */
    libxsmm_meltw_transform_flags trans_flags;
    trans_flags = LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_NORMT;
    libxsmm_meltwfunction_transform trans_shortkernel_grad = libxsmm_dispatch_meltw_transform(short_W_t, N_g, &M_g, &N_g, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, trans_flags);
    libxsmm_meltwfunction_transform trans_edgekernel_grad = libxsmm_dispatch_meltw_transform(edge_W_t, N_g, &M_g, &N_g, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, trans_flags);
    if ( trans_shortkernel_grad == NULL | trans_edgekernel_grad == NULL) {
        fprintf( stderr, "JIT for NORM_TO_NORMT TPP. Bailing...!\n");
        exit(-1);
    }

#endif

    /* Dispatch brGEMM kernel for normal and edge cases*/
    libxsmm_smmfunction kernel_w5 = libxsmm_smmdispatch(F_t, C_t, XS_TILE_WBACKWARD, &ldb_trans_g, &lda_g, &ldc_g, NULL, NULL, NULL, NULL);
    libxsmm_smmfunction kernel_w6 = libxsmm_smmdispatch(F_t, C_t, W_t - tile_multiple, &ldb_trans_g, &lda_g, &ldc_g, NULL, NULL, NULL, NULL);

    #pragma omp parallel for reduction(+: flip_d_weight_a[:F_t*C_t*WW_t])                // Distribute the weight array
    for(int n = 0; n < N_t; n++) {
        int last_block = 0;
#ifdef USE_TPP
        libxsmm_meltw_transform_param trans_param_short;                    // Pointer to hold trans short
        libxsmm_meltw_transform_param trans_param_edge;                     // Pointer to hold trans edge
#endif
        for(int wb = 0; wb < W_t - XS_TILE_WBACKWARD + 1; wb += XS_TILE_WBACKWARD) {                // Normal case

#ifndef USE_TPP                                 // If not usintg TPP kernels

            trans_shortkernel_grad(&grad_a[n*F_t*W_t + wb], &M_g, &grad_shorttrans[n*F_t*short_W_t], &N_g);
#else
            trans_param_short.in_ptr  = &grad_a[n*F_t*W_t + wb];
            trans_param_short.out_ptr = &grad_shorttrans[n*F_t*short_W_t];
            trans_shortkernel_grad( &trans_param_short );
#endif
            for(int kw = 0; kw < WW_t; kw++) {
                kernel_w5(&grad_shorttrans[n*F_t*short_W_t], &input_a[n*C_t*Win_t + wb + kw*dial], &flip_d_weight_a[kw*C_t*F_t]);
            }
            last_block = wb;
        }

        if (W_t % XS_TILE_WBACKWARD != 0){
#ifndef USE_TPP                                 // If not usintg TPP kernels

            trans_edgekernel_grad(&grad_a[n*F_t*W_t + last_block + XS_TILE_WBACKWARD], &M_g, &grad_edgetrans[n*F_t*edge_W_t], &N_g);
#else
            trans_param_edge.in_ptr  = &grad_a[n*F_t*W_t + last_block + XS_TILE_WBACKWARD];
            trans_param_edge.out_ptr = &grad_edgetrans[n*F_t*edge_W_t];
            trans_edgekernel_grad( &trans_param_edge );
#endif
            for(int kw = 0; kw < WW_t; kw++) {

                kernel_w6(&grad_edgetrans[n*F_t*edge_W_t], &input_a[n*C_t*Win_t + (last_block + XS_TILE_WBACKWARD) + kw*dial], &flip_d_weight_a[kw*F_t*C_t]);
            }
        }
    }

    for(int kw = 0; kw < WW_t; kw++){                   // permute weight gradiant array for return
        for(int filter=0; filter < F_t; filter++){
            for (int channel=0; channel < C_t; channel++){

                d_weight_a[filter*C_t*WW_t + channel*WW_t + kw] = flip_d_weight_a[kw*C_t*F_t + channel*F_t + filter];
            }
        }
    }

    return {d_input, d_weight};         // return data gradiant and weight gradiant
}


at::Tensor relu_forward_bf16(at::Tensor& input){

    // RECORD_FUNCTION("ReLU_forward_bf16", std::vector<c10::IValue>({input}));           // For recording time

    int64_t N_t = input.size(0);                    // Batch
    int64_t C_t = input.size(1);                    // Channel
    int64_t W_t = input.size(2);                    // input width

    auto Y = input.new_empty({N_t,C_t,W_t});        // New tensor for output
    libxsmm_bfloat16* input_a = (libxsmm_bfloat16*) input.data_ptr<at::BFloat16>();
    libxsmm_bfloat16* Y_a = (libxsmm_bfloat16*) Y.data_ptr<at::BFloat16>();

// #ifndef USE_TPP

    #pragma omp parallel for
    for(unsigned long w = 0; w < N_t*C_t*W_t; w++) {    // width loop
        if(input_a[w] >= 32768)                         // sign bit indicates if value is negative
            Y_a[w] = 0;
        else
            Y_a[w] = input_a[w];
    }

// #else
    // libxsmm_blasint tpp_m = 1;                      // rows
    // libxsmm_blasint tpp_n = N_t*C_t*W_t;      // columns
    // libxsmm_meltwfunction_relu relu_fwd_kernel = libxsmm_dispatch_meltw_relu(tpp_m, tpp_n, NULL, NULL, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_RELU_FWD, 0);
    // if ( relu_fwd_kernel == NULL ) {
    //     fprintf( stderr, "JIT for TPP relu_fwd_kernel failed. Bailing...!\n");
    //     exit(-1);
    // }
    // libxsmm_meltw_relu_param relu_params;
    // relu_params.in_ptr   = &input_a[0];
    // relu_params.out_ptr  = &Y_a[0];
    // relu_fwd_kernel(&relu_params);
// #endif

    return Y;
}

at::Tensor relu_backward_bf16(at::Tensor& grad, at::Tensor& output){

    // RECORD_FUNCTION("ReLU_backward_bf16", std::vector<c10::IValue>({grad, output}));        // For recording time

    int64_t N_t = output.size(0);                    // Batch
    int64_t C_t = output.size(1);                    // Channel
    int64_t W_t = output.size(2);                    // input width

    libxsmm_bfloat16* output_a = (libxsmm_bfloat16*) output.data_ptr<at::BFloat16>();
    libxsmm_bfloat16* grad_a = (libxsmm_bfloat16*) grad.data_ptr<at::BFloat16>();

    auto d_input = output.new_empty({N_t,C_t,W_t});        // New tensor for input grad
    libxsmm_bfloat16* d_input_a = (libxsmm_bfloat16*) d_input.data_ptr<at::BFloat16>();

// #ifndef USE_TPP
    #pragma omp parallel for
    for(unsigned long w = 0; w < N_t*C_t*W_t; w++) {    // width blocking loop
        // if(output_a[w] == 0 || output_a[w] == 32768)
        if(output_a[w] == 0)                            // If output array value was zero
            d_input_a[w] = 0;
        else
            d_input_a[w] = grad_a[w];
    }

// #else
    // libxsmm_blasint tpp_m = 1;                      // rows
    // libxsmm_blasint tpp_n = N_t*C_t*W_t;      // columns
    // libxsmm_blasint ld = 1;
    // libxsmm_meltwfunction_relu relu_bwd_kernel = libxsmm_dispatch_meltw_relu(tpp_m, tpp_n, NULL, NULL, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_RELU_BWD, 0);
    // if ( relu_bwd_kernel == NULL ) {
    //     fprintf( stderr, "JIT for TPP relu_bwd_kernel failed. Bailing...!\n");
    //     exit(-1);
    // }
    // libxsmm_meltw_relu_param relu_params;
    // relu_params.in_ptr   = grad_a;
    // relu_params.out_ptr  = d_input_a;
    // relu_params.mask_ptr = output_a;
    // relu_bwd_kernel(&relu_params);
// #endif
    return d_input;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("forward", &Conv1dOpti_forward_libxsmm, "Conv1dOpti lib forward");
m.def("backward", &Conv1dOpti_backward_libxsmm, "Conv1dOpti lib backward");
m.def("forward_bf16", &Conv1dOpti_forward_bf16_libxsmm, "Conv1dOpti bf16 forward");
m.def("backward_bf16", &Conv1dOpti_backward_bf16_libxsmm, "Conv1dOpti bf16 backward");
m.def("relu_forward_bf16", &relu_forward_bf16, "ReLU bf16 forward");
m.def("relu_backward_bf16", &relu_backward_bf16, "ReLU bf16 backward");
}