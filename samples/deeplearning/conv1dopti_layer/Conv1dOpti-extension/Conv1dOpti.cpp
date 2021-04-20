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

    /* jited tranpose to permute the array dimensions
        Overall convert (F_t, C_t, WW_t) -----> (WW_t, F_t, C_t)*/
    libxsmm_blasint per_m = WW_t;
    libxsmm_blasint per_n = F_t*C_t;
    libxsmm_blasint per_ldi = WW_t;
    libxsmm_blasint per_ldo = F_t*C_t;

    libxsmm_meltwfunction_unary trans_permute_kernel = libxsmm_dispatch_meltw_unary(per_m, per_n, &per_ldi, &per_ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
    if ( trans_permute_kernel == NULL) {
        fprintf( stderr, "JIT unary TPP for NORM_TO_NORMT. Bailing...!\n");
        exit(-1);
    }
    libxsmm_meltw_unary_param trans_permute_param;
    trans_permute_param.in.primary  = weight_a;
    trans_permute_param.out.primary = flip_weight_a;
    trans_permute_kernel( &trans_permute_param);

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

    /* JIT eltwise TPPs for initialization ... */
    libxsmm_blasint tpp_m1 = XS_TILE_FORWARD;                      // columns
    libxsmm_blasint tpp_m2 = W_t - tile_multiple;                  // columns
    libxsmm_blasint tpp_n = F_t;                                   // rows
    libxsmm_blasint ld_zero = W_t;

    libxsmm_meltwfunction_unary copy_kernel_1 = libxsmm_dispatch_meltw_unary(tpp_m1, tpp_n, NULL, &ld_zero, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
    libxsmm_meltwfunction_unary copy_kernel_2 = libxsmm_dispatch_meltw_unary(tpp_m2, tpp_n, NULL, &ld_zero, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
    if ( copy_kernel_1 == NULL || copy_kernel_2 == NULL) {
        fprintf( stderr, "JIT for initialization by unary copy kernel failed. Bailing...!\n");
        exit(-1);
    }

    /* use jited VNNI */
    libxsmm_blasint ldi = Win_t;
    libxsmm_blasint ldo_short = short_width;
    libxsmm_blasint ldo_edge = edge_width;

    libxsmm_meltw_unary_type trans_vnni_type;
    if ( C_t % 2 == 1 ) {
        trans_vnni_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI_PAD;
    } else {
        trans_vnni_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI;
    }
    tpp_m1 = (XS_TILE_FORWARD + dial*(WW_t-1));
    tpp_m2 = (W_t - tile_multiple + dial*(WW_t-1));
    libxsmm_meltwfunction_unary trans_shortvnni_kernel = libxsmm_dispatch_meltw_unary(tpp_m1, C_t, &ldi, &ldo_short, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, trans_vnni_type);
    libxsmm_meltwfunction_unary trans_edgevnni_kernel = libxsmm_dispatch_meltw_unary(tpp_m2, C_t, &ldi, &ldo_edge, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, trans_vnni_type);
    if ( trans_shortvnni_kernel == NULL | trans_edgevnni_kernel == NULL) {
        fprintf( stderr, "JIT for NORM_TO_VNNI TPP. Bailing...!\n");
        exit(-1);
    }

    // Main compute loop
    #pragma omp parallel for
    for(int n = 0; n < N_t; n++) {                               // Loop for batches
        int last_block = 0;
        libxsmm_meltw_unary_param copy_params_1;           // Copy parameter variable for holding the pointer
        libxsmm_meltw_unary_param copy_params_2;
        libxsmm_meltw_unary_param trans_param_short;
        libxsmm_meltw_unary_param trans_param_edge;

        for(int wb = 0; wb < W_t - XS_TILE_FORWARD + 1; wb += XS_TILE_FORWARD) {    // width blocking loop (Normal case)

            copy_params_1.out.primary = &Y_a[n*F_t*W_t + wb];                 /* Initialization of output array */
            copy_kernel_1(&copy_params_1);

            // VNNI transform
            trans_param_short.in.primary  = &input_a[n*C_t*Win_t + 0*Win_t + wb];
            trans_param_short.out.primary = &input_a_shortvnni[n*C_t*short_width];
            trans_shortvnni_kernel( &trans_param_short );

            // brGEMM
            bmmshortkernel(&input_a_shortvnni[n*C_t*short_width], &flip_weight_a[0], &Y_a[n*F_t*W_t + 0*W_t + wb], &l_br);

            last_block = wb;        // Store value for last block
        }

        if (W_t % XS_TILE_FORWARD != 0){                       // Edge case

            copy_params_2.out.primary = &Y_a[n*F_t*W_t + last_block + XS_TILE_FORWARD];                 /* Initialization of output array */
            copy_kernel_2(&copy_params_2);

            // VNNI transform
            trans_param_edge.in.primary  = &input_a[n*C_t*Win_t + 0*Win_t + (last_block + XS_TILE_FORWARD)];
            trans_param_edge.out.primary = &input_a_edgevnni[n*C_t*edge_width];
            trans_edgevnni_kernel( &trans_param_edge );

            // brGEMM
            bmmedgekernel2(&input_a_edgevnni[n*C_t*edge_width], &flip_weight_a[0], &Y_a[n*F_t*W_t + 0*W_t + (last_block + XS_TILE_FORWARD)], &l_br);
        }
    }

    return Y;              // Return output tensorcd
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


    auto weight_buffer = weight.new_empty({F_t,C_t,WW_t});                  // Tensor weight buffer
    libxsmm_bfloat16* weight_buffer_a = (libxsmm_bfloat16*) weight_buffer.data_ptr<at::BFloat16>();

    #pragma omp parallel for
    for(int i = 0; i < F_t*C_t; i++){
        for(int kw = 0; kw < WW_t; kw++){                                   // reverse copy
            flip_weight_a[i*WW_t + kw] = weight_a[i*WW_t + WW_t - kw - 1];
        }
    }

    /* jited tranpose to permute the array dimensions
        Overall convert (F_t, C_t, WW_t) -----> (WW_t, C_t, F_t)*/
    libxsmm_blasint flip_m1 = WW_t;
    libxsmm_blasint flip_n1 = F_t*C_t;
    libxsmm_blasint flip_ldi_1 = WW_t;
    libxsmm_blasint flip_ldo_1 = F_t*C_t;

    libxsmm_meltwfunction_unary trans_flip_1 = libxsmm_dispatch_meltw_unary(flip_m1, flip_n1, &flip_ldi_1, &flip_ldo_1, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
    if ( trans_flip_1 == NULL) {
        fprintf( stderr, "JIT for unary NORM_TO_NORMT TPP. Bailing...!\n");
        exit(-1);
    }

    // Convert (F_t, C_t, WW_t) -----> (WW_t, F_t, C_t)
    libxsmm_meltw_unary_param trans_param_flip_1;
    trans_param_flip_1.in.primary  = flip_weight_a;
    trans_param_flip_1.out.primary = weight_buffer_a;
    trans_flip_1( &trans_param_flip_1 );

    libxsmm_blasint flip_m2 = C_t;
    libxsmm_blasint flip_n2 = F_t;
    libxsmm_blasint flip_ldi_2 = C_t;
    libxsmm_blasint flip_ldo_2 = F_t;

    libxsmm_meltwfunction_unary trans_flip_2 = libxsmm_dispatch_meltw_unary(flip_m2, flip_n2, &flip_ldi_2, &flip_ldo_2, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
    if ( trans_flip_2 == NULL) {
        fprintf( stderr, "JIT for unary NORM_TO_NORMT TPP. Bailing...!\n");
        exit(-1);
    }

    // Convert (WW_t, F_t, C_t) -----> (F_t, C_t, WW_t)
    #pragma omp parallel for
    for(int kw = 0; kw < WW_t; kw++){                   // permute last two dimensions
        libxsmm_meltw_unary_param trans_param_flip_2;
        trans_param_flip_2.in.primary  = &weight_buffer_a[kw*C_t*F_t];
        trans_param_flip_2.out.primary = &flip_weight_a[kw*C_t*F_t];
        trans_flip_2( &trans_param_flip_2 );
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

// #ifdef USE_TPP
    //    Virtual copy kernels
    libxsmm_blasint virtual_m1 = pad_tile_multiple - ((WW_t - 1)*dial);                      // columns
    libxsmm_blasint virtual_m2 = ((WW_t - 1)*dial);                      // columns
    libxsmm_blasint virtual_n = F_t;                                        // rows
    libxsmm_blasint ldi_virtual = W_t;
    libxsmm_blasint ldo_virtual = 2*pad_tile_multiple;

    if (ldi_virtual < virtual_m1){                      // corner case when width's are very small
        virtual_m1 = ldi_virtual;
        libxsmm_meltwfunction_unary all_zero = libxsmm_dispatch_meltw_unary(ldo_virtual, virtual_n, NULL, &ldo_virtual, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
        if ( all_zero == NULL) {
            fprintf( stderr, "JIT for initialization by unary virtual all zero kernel failed. Bailing...!\n");
            exit(-1);
        }
        #pragma omp parallel for
        for(int n = 0; n < N_t; n++){
            libxsmm_meltw_unary_param all_zero_params;
            all_zero_params.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual];                 // Initialize the entire array when widths are small
            all_zero(&all_zero_params);
        }
    }

    libxsmm_meltwfunction_unary virtual_copy = libxsmm_dispatch_meltw_unary(virtual_m1, virtual_n, &ldi_virtual, &ldo_virtual, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
    libxsmm_meltwfunction_unary virtual_copy_zero = libxsmm_dispatch_meltw_unary(virtual_m2, virtual_n, NULL, &ldo_virtual, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
    if ( virtual_copy == NULL || virtual_copy_zero == NULL) {
        fprintf( stderr, "JIT for initialization by unary virtual copy kernel failed. Bailing...!\n");
        exit(-1);
    }

    #pragma omp parallel for
    for(int n = 0; n < N_t; n++){                       // Loops for storing the edge portion of gradinant array into grad_a_shortpad

        libxsmm_meltw_unary_param vcopy_params;           // Copy parameter variable for holding the pointer
        libxsmm_meltw_unary_param vcopy_params_zero;

        vcopy_params_zero.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual];                                        // copy zeros
        virtual_copy_zero(&vcopy_params_zero);

        vcopy_params.in.primary = &grad_a[n*F_t*W_t];                                                             // copy after zeros from start of the grad array
        vcopy_params.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual + ((WW_t - 1)*dial)];
        virtual_copy(&vcopy_params);

        vcopy_params.in.primary = &grad_a[n*F_t*W_t + W_t - virtual_m1];              // copy from the end of the grad array
        vcopy_params.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual + ldo_virtual - virtual_m1 - ((WW_t - 1)*dial)];
        virtual_copy(&vcopy_params);

        vcopy_params_zero.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual + ldo_virtual - ((WW_t - 1)*dial)];     // copy zeros
        virtual_copy_zero(&vcopy_params_zero);
    }

// #else
//     #pragma omp parallel for
//     for(int n = 0; n < N_t; n++){                   // loop to store the edges for gradiant array into grad_a_shortpad buffer
//         for(int filter=0; filter < F_t; filter++){
//             for(int w = 0; w < pad_tile_multiple; w++){
//                 // initialize start of array
//                 if (w >= ((WW_t - 1)*dial) && w < (W_t + (WW_t - 1)*dial)){
//                     grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w] = grad_a[n*F_t*W_t + filter*W_t + w - (WW_t - 1)*dial];
//                 }
//                 else{
//                     grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w] = 0.0f;
//                 }
//             }
//             for(int w = Wpad_t - pad_tile_multiple; w < Wpad_t ; w++){
//                 // initialize end of array
//                 if (w >= ((WW_t - 1)*dial) && w < (W_t + (WW_t - 1)*dial)){
//                     grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w - Wpad_t + 2*pad_tile_multiple] = grad_a[n*F_t*W_t + filter*W_t + w - (WW_t - 1)*dial];
//                 }
//                 else{
//                     grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w - Wpad_t + 2*pad_tile_multiple] = 0.0f;
//                 }
//             }
//         }
//     }

// #endif

    int short_width = ((XS_TILE_DBACKWARD + (WW_t-1)*dial)/XS_TILE_DBACKWARD + 1)*XS_TILE_DBACKWARD;    // Width of buffer   (512)

    auto grad_shortvnni_tensor = grad.new_empty({N_t,F_t,short_width});                                 // Buffer for storing VNNI transform
    libxsmm_bfloat16* grad_a_shortvnni = (libxsmm_bfloat16*) grad_shortvnni_tensor.data_ptr<at::BFloat16>();

    /* Dispatch brGEMM kernels for the normal case and the edge case*/
    libxsmm_bmmfunction_reducebatch_strd bmmshortkernel = libxsmm_bmmdispatch_reducebatch_strd(XS_TILE_DBACKWARD, C_t, F_t, 2*dial*sizeof(libxsmm_bfloat16), C_t*F_t*sizeof(libxsmm_bfloat16), &short_width, &lda, &ldc, NULL, NULL, NULL, NULL);
    libxsmm_bmmfunction_reducebatch_strd bmmshortkernel2 = libxsmm_bmmdispatch_reducebatch_strd(Win_t - tile_multiple, C_t, F_t, 2*dial*sizeof(libxsmm_bfloat16), C_t*F_t*sizeof(libxsmm_bfloat16), &short_width, &lda, &ldc, NULL, NULL, NULL, NULL);

    if ( bmmshortkernel == NULL || bmmshortkernel2 == NULL) {
        fprintf( stderr, "JIT for bmm kernel failed. Bailing...!\n");
        exit(-1);
    }

    /* JIT eltwise TPPs for initialization ... */
    libxsmm_blasint tpp_m1 = XS_TILE_DBACKWARD;                      // columns
    libxsmm_blasint tpp_m2 = Win_t - tile_multiple;                      // columns
    libxsmm_blasint tpp_n = C_t;                                     // rows
    libxsmm_blasint ld_zero = Win_t;

    libxsmm_meltwfunction_unary copy_kernel_1 = libxsmm_dispatch_meltw_unary(tpp_m1, tpp_n, NULL, &ld_zero, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
    libxsmm_meltwfunction_unary copy_kernel_2 = libxsmm_dispatch_meltw_unary(tpp_m2, tpp_n, NULL, &ld_zero, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
    if ( copy_kernel_1 == NULL || copy_kernel_2 == NULL) {
        fprintf( stderr, "JIT for initialization by unary copy kernel failed. Bailing...!\n");
        exit(-1);
    }

    /* use jited VNNI */
    libxsmm_blasint ldi_1 = W_t;
    libxsmm_blasint ldi_2 = ldb_shortpad;                   // (1792)
    libxsmm_blasint ldo = short_width;                      // (512)

    libxsmm_meltw_unary_type trans_vnni_type;
    if ( F_t % 2 == 1 ) {
        trans_vnni_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI_PAD;
    } else {
        trans_vnni_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI;
    }

    tpp_m1 = (XS_TILE_DBACKWARD + dial*(WW_t-1));
    tpp_m2 = (XS_TILE_DBACKWARD + dial*(WW_t-1));

    libxsmm_meltwfunction_unary trans_shortvnni_kernel_1 = libxsmm_dispatch_meltw_unary(tpp_m1, F_t, &ldi_1, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, trans_vnni_type);
    libxsmm_meltwfunction_unary trans_shortvnni_kernel_2 = libxsmm_dispatch_meltw_unary(tpp_m2, F_t, &ldi_2, &ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, trans_vnni_type);
    if ( trans_shortvnni_kernel_1 == NULL | trans_shortvnni_kernel_2 == NULL) {
        fprintf( stderr, "JIT for unary NORM_TO_VNNI TPP. Bailing...!\n");
        exit(-1);
    }

    // Main computation loop
    #pragma omp parallel for
    for(int n = 0; n < N_t; n++) {
        int last_block=0;

        libxsmm_meltw_unary_param copy_params_1;                       // Copy parameter variable for holding the pointer
        libxsmm_meltw_unary_param copy_params_2;
        libxsmm_meltw_unary_param trans_param_1;
        libxsmm_meltw_unary_param trans_param_2;

        for(int wb = 0; wb < Win_t - XS_TILE_DBACKWARD + 1; wb += XS_TILE_DBACKWARD) {

            copy_params_1.out.primary = &d_input_a[n*C_t*Win_t + wb];             // Initialization
            copy_kernel_1(&copy_params_1);

            if (wb >= (WW_t-1)*dial && wb < Win_t - (WW_t-1)*dial - XS_TILE_DBACKWARD){
                // Normal case (Take VNNI transform of a portion of grad_a array )

                // VNNI transform
                trans_param_1.in.primary  = &grad_a[n*F_t*W_t + 0*W_t + wb - (WW_t-1)*dial];
                trans_param_1.out.primary = &grad_a_shortvnni[n*F_t*short_width];
                trans_shortvnni_kernel_1( &trans_param_1 );

                // brGEMM
                bmmshortkernel(&grad_a_shortvnni[n*F_t*short_width], &flip_weight_a[0], &d_input_a[n*C_t*Win_t + 0*Win_t + wb], &l_br);
            }
            else if (wb < (WW_t-1)*dial){
                // Right side case (Take VNNI transform of grad_a_shortpad array)

                // VNNI transform
                trans_param_2.in.primary  = &grad_a_shortpad[n*F_t*2*pad_tile_multiple + wb];
                trans_param_2.out.primary = &grad_a_shortvnni[n*F_t*short_width];
                trans_shortvnni_kernel_2( &trans_param_2 );

                // brGEMM
                bmmshortkernel(&grad_a_shortvnni[n*F_t*short_width], &flip_weight_a[0], &d_input_a[n*C_t*Win_t + 0*Win_t + wb], &l_br);
            }
            else{
                // Left side case (Take VNNI transform of grad_a_shortpad array)

                // VNNI transform
                trans_param_2.in.primary  = &grad_a_shortpad[n*F_t*2*pad_tile_multiple + wb - Wpad_t + 2*pad_tile_multiple];
                trans_param_2.out.primary = &grad_a_shortvnni[n*F_t*short_width];
                trans_shortvnni_kernel_2( &trans_param_2 );

                // brGEMM
                bmmshortkernel(&grad_a_shortvnni[n*F_t*short_width], &flip_weight_a[0], &d_input_a[n*C_t*Win_t + 0*Win_t + wb], &l_br);
            }
            last_block = wb;
        }

        if (Win_t % XS_TILE_DBACKWARD != 0){                                // Edge case

            // Right side case (Take VNNI transform of grad_a_shortpad array)

            copy_params_2.out.primary = &d_input_a[n*C_t*Win_t + last_block + XS_TILE_DBACKWARD];             // Initialization
            copy_kernel_2(&copy_params_2);

            // VNNI transform
            trans_param_2.in.primary  = &grad_a_shortpad[n*F_t*2*pad_tile_multiple + last_block + XS_TILE_DBACKWARD - Wpad_t + 2*pad_tile_multiple];
            trans_param_2.out.primary = &grad_a_shortvnni[n*F_t*short_width];
            trans_shortvnni_kernel_2( &trans_param_2 );

            // brGEMM
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

    libxsmm_blasint M_g = W_t/2;
    libxsmm_blasint N_g = F_t;
    libxsmm_blasint short_W_t = XS_TILE_WBACKWARD;
    libxsmm_blasint edge_W_t = W_t - tile_multiple;

    auto grad_shortvnni_tensor2 = grad.new_empty({N_t,F_t,short_W_t});                            // Short buffer for storing VNNI transform
    libxsmm_bfloat16* grad_shortvnni = (libxsmm_bfloat16*) grad_shortvnni_tensor2.data_ptr<at::BFloat16>();

    auto grad_edgevnni_tensor2 = grad.new_empty({N_t,F_t,edge_W_t});                              // Short buffer for storing VNNI transform in edge case
    libxsmm_bfloat16* grad_edgevnni = (libxsmm_bfloat16*) grad_edgevnni_tensor2.data_ptr<at::BFloat16>();

    /* use jited tranpose */
    libxsmm_meltwfunction_unary trans_shortkernel_grad = libxsmm_dispatch_meltw_unary(short_W_t/2, N_g, &M_g, &N_g, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
    libxsmm_meltwfunction_unary trans_edgekernel_grad = libxsmm_dispatch_meltw_unary(edge_W_t/2, N_g, &M_g, &N_g, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
    if ( trans_shortkernel_grad == NULL | trans_edgekernel_grad == NULL) {
        fprintf( stderr, "JIT for unary NORM_TO_NORMT TPP. Bailing...!\n");
        exit(-1);
    }

    /* Dispatch brGEMM kernels for the normal case and the edge case*/
    libxsmm_bsmmfunction bsmmkernel5 = libxsmm_bsmmdispatch(F_t, C_t, XS_TILE_WBACKWARD, &ldb_trans_g, &lda_g, &ldc_g, NULL, NULL, NULL, NULL);
    libxsmm_bsmmfunction bsmmkernel6 = libxsmm_bsmmdispatch(F_t, C_t, W_t - tile_multiple, &ldb_trans_g, &lda_g, &ldc_g, NULL, NULL, NULL, NULL);

    if ( bsmmkernel5 == NULL | bsmmkernel6 == NULL) {
        fprintf( stderr, "JIT for bsmm kernel. Bailing...!\n");
        exit(-1);
    }

    // Main compute loop
    #pragma omp parallel for reduction(+: flip_d_weight_a[:F_t*C_t*WW_t])
    for(int n = 0; n < N_t; n++) {
        int last_block = 0;
        libxsmm_meltw_unary_param trans_param_short;
        libxsmm_meltw_unary_param trans_param_edge;

        for(int wb = 0; wb < W_t - XS_TILE_WBACKWARD + 1; wb += XS_TILE_WBACKWARD) {            // Normal Case

            /* Take transpose assumping FP32 (This will do both transpose and VNNI transform for BF16) */
            trans_param_short.in.primary  = &grad_a[n*F_t*W_t + wb];
            trans_param_short.out.primary = &grad_shortvnni[n*F_t*short_W_t];
            trans_shortkernel_grad( &trans_param_short );

            for(int kw = 0; kw < WW_t; kw++) {
                // libxsmm_bsmmfunction bsmmkernel5 = libxsmm_bsmmdispatch(F_t, C_t, XS_TILE_WBACKWARD, &ldb_trans_g, &lda_g, &ldc_g, NULL, NULL, NULL, NULL);
                bsmmkernel5(&grad_shortvnni[n*F_t*short_W_t], &input_a[n*C_t*Win_t + wb + kw*dial], &flip_d_weight_a[kw*C_t*F_t]);
            }
            last_block = wb;
        }

        if (W_t % XS_TILE_WBACKWARD != 0){              // Edge Case

            trans_param_edge.in.primary  = &grad_a[n*F_t*W_t + last_block + XS_TILE_WBACKWARD];
            trans_param_edge.out.primary = &grad_edgevnni[n*F_t*edge_W_t];
            trans_edgekernel_grad( &trans_param_edge );

            for(int kw = 0; kw < WW_t; kw++) {
                // libxsmm_bsmmfunction bsmmkernel6 = libxsmm_bsmmdispatch(F_t, C_t, W_t - tile_multiple, &ldb_trans_g, &lda_g, &ldc_g, NULL, NULL, NULL, NULL);
                bsmmkernel6(&grad_edgevnni[n*F_t*edge_W_t], &input_a[n*C_t*Win_t + (last_block + XS_TILE_WBACKWARD) + kw*dial], &flip_d_weight_a[kw*F_t*C_t]);
            }
        }
    }


    auto flip_d_weight_tensor = weight.new_empty({WW_t,C_t,F_t});
    libxsmm_bfloat16* flip_d_weight_bf16 = (libxsmm_bfloat16*) flip_d_weight_tensor.data_ptr<at::BFloat16>();


    /* JIT eltwise TPPs for FP32 to BF16 conversion... */
    libxsmm_blasint cvt_m = 1;
    libxsmm_blasint cvt_n = F_t*C_t*WW_t;
    libxsmm_meltwfunction_unary eltwise_kernel = libxsmm_dispatch_meltw_unary(cvt_m, cvt_n, NULL, NULL, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
    if ( eltwise_kernel == NULL ) {
        fprintf( stderr, "JIT for TPP convert FP32 to BF16 failed. Bailing...!\n");
        exit(-1);
    }

    libxsmm_meltw_unary_param eltwise_params;
    eltwise_params.in.primary = flip_d_weight_a;
    eltwise_params.out.primary = flip_d_weight_bf16;
    eltwise_kernel(&eltwise_params);


    /* jited tranpose to permute the array dimensions
        Overall Convert (WW_t, C_t, F_t) -----> (F_t, C_t, WW_t)*/
    libxsmm_blasint per_m1 = F_t;
    libxsmm_blasint per_n1 = C_t;
    libxsmm_blasint ldi_per_1 = F_t;
    libxsmm_blasint ldo_per_1 = C_t;

    libxsmm_meltwfunction_unary trans_permute_1 = libxsmm_dispatch_meltw_unary(per_m1, per_n1, &ldi_per_1, &ldo_per_1, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
    if ( trans_permute_1 == NULL) {
        fprintf( stderr, "JIT for unary NORM_TO_NORMT TPP. Bailing...!\n");
        exit(-1);
    }
    // Convert (WW_t, C_t, F_t) -----> (WW_t, F_t, C_t)
    #pragma omp parallel for
    for(int kw = 0; kw < WW_t; kw++){                   // permute last two dimensions
        libxsmm_meltw_unary_param trans_param_permute_1;
        trans_param_permute_1.in.primary  = &flip_d_weight_bf16[kw*C_t*F_t];
        trans_param_permute_1.out.primary = &flip_weight_a[kw*C_t*F_t];
        trans_permute_1( &trans_param_permute_1 );
    }

    libxsmm_blasint per_m2 = F_t*C_t;
    libxsmm_blasint per_n2 = WW_t;
    libxsmm_blasint ldi_per_2 = F_t*C_t;
    libxsmm_blasint ldo_per_2 = WW_t;

    libxsmm_meltwfunction_unary trans_permute_2 = libxsmm_dispatch_meltw_unary(per_m2, per_n2, &ldi_per_2, &ldo_per_2, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
    if ( trans_permute_2 == NULL) {
        fprintf( stderr, "JIT for unary NORM_TO_NORMT TPP. Bailing...!\n");
        exit(-1);
    }

    // Convert (WW_t, F_t, C_t) -----> (F_t, C_t, WW_t)
    libxsmm_meltw_unary_param trans_param_permute_2;
    trans_param_permute_2.in.primary  = flip_weight_a;
    trans_param_permute_2.out.primary = d_weight_a;
    trans_permute_2( &trans_param_permute_2 );

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

    /* jited tranpose to permute the array dimensions
        Overall convert (F_t, C_t, WW_t) -----> (WW_t, F_t, C_t)*/

    libxsmm_blasint per_m = WW_t;
    libxsmm_blasint per_n = F_t*C_t;
    libxsmm_blasint per_ldi = WW_t;
    libxsmm_blasint per_ldo = F_t*C_t;

    libxsmm_meltwfunction_unary trans_permute_kernel = libxsmm_dispatch_meltw_unary(per_m, per_n, &per_ldi, &per_ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
    if ( trans_permute_kernel == NULL) {
        fprintf( stderr, "JIT unary TPP for NORM_TO_NORMT TPP. Bailing...!\n");
        exit(-1);
    }
    libxsmm_meltw_unary_param trans_permute_param;
    trans_permute_param.in.primary  = weight_a;
    trans_permute_param.out.primary = flip_weight_a;
    trans_permute_kernel( &trans_permute_param);

    int lda = C_t;                      // Input channels (15)
    int ldb = Win_t;                    // Input width (60400)
    int ldc = W_t;                      // Output width (60000)
    unsigned long long l_br = WW_t;

    int tile_multiple = (W_t/XS_TILE_FORWARD)*XS_TILE_FORWARD;

    /* Dispatch brGEMM kernels for the normal case and the edge case*/
    libxsmm_smmfunction_reducebatch_strd kernel = libxsmm_smmdispatch_reducebatch_strd(XS_TILE_FORWARD, F_t, C_t, dial*sizeof(float), F_t*C_t*sizeof(float), &ldb, &lda, &ldc, NULL, NULL, NULL, NULL);
    libxsmm_smmfunction_reducebatch_strd kernel2 = libxsmm_smmdispatch_reducebatch_strd(W_t - tile_multiple, F_t, C_t, dial*sizeof(float), F_t*C_t*sizeof(float), &ldb, &lda, &ldc, NULL, NULL, NULL, NULL);

    /* JIT eltwise TPPs for initialization... */
    libxsmm_blasint tpp_m1 = XS_TILE_FORWARD;                      // columns
    libxsmm_blasint tpp_m2 = W_t - tile_multiple;                      // columns
    libxsmm_blasint tpp_n = F_t;      // rows
    libxsmm_blasint ld_zero = W_t;

    libxsmm_meltw_unary_type unary_type;
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_XOR;
    libxsmm_meltw_unary_flags unary_flags;
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;

    libxsmm_meltwfunction_unary unary_kernel_1 = libxsmm_dispatch_meltw_unary(tpp_m1, tpp_n, NULL, &ld_zero, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, unary_type);
    libxsmm_meltwfunction_unary unary_kernel_2 = libxsmm_dispatch_meltw_unary(tpp_m2, tpp_n, NULL, &ld_zero, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, unary_flags, unary_type);
    if ( unary_kernel_1 == NULL || unary_kernel_2 == NULL) {
        fprintf( stderr, "JIT for copy UNARY kernel. Bailing...!\n");
        exit(-1);
    }

    // Main compute loop
    #pragma omp parallel for
    for(int n = 0; n < N_t; n++) {                               // Loop for batches
        int last_block = 0;
        libxsmm_meltw_unary_param unary_param_1;
        libxsmm_meltw_unary_param unary_param_2;

        for(int wb = 0; wb < W_t - XS_TILE_FORWARD + 1; wb += XS_TILE_FORWARD) {    // width blocking loop (Normal case)

            unary_param_1.out.primary = &Y_a[n*F_t*W_t + wb];       // Initialization
            unary_kernel_1( &unary_param_1 );

            kernel(&input_a[n*C_t*Win_t + 0*Win_t + wb], &flip_weight_a[0], &Y_a[n*F_t*W_t + 0*W_t + wb], &l_br);
            last_block = wb;
        }

        if (W_t % XS_TILE_FORWARD != 0){                        // Edge Case

            unary_param_2.out.primary = &Y_a[n*F_t*W_t + last_block + XS_TILE_FORWARD];     // Initialization
            unary_kernel_2( &unary_param_2 );

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


    auto weight_buffer = weight.new_empty({F_t,C_t,WW_t});                  // Tensor weight buffer
    float* weight_buffer_a = weight_buffer.data_ptr<float>();

    #pragma omp parallel for
    for(int i = 0; i < F_t*C_t; i++){
        for(int kw = 0; kw < WW_t; kw++){                                   // reverse copy
            flip_weight_a[i*WW_t + kw] = weight_a[i*WW_t + WW_t - kw - 1];
        }
    }

    /* jited tranpose to permute the array dimensions
        Overall convert (F_t, C_t, WW_t) -----> (WW_t, C_t, F_t)*/

    libxsmm_blasint flip_m1 = WW_t;
    libxsmm_blasint flip_n1 = F_t*C_t;
    libxsmm_blasint flip_ldi_1 = WW_t;
    libxsmm_blasint flip_ldo_1 = F_t*C_t;

    libxsmm_meltwfunction_unary trans_unary_flip_1 = libxsmm_dispatch_meltw_unary(flip_m1, flip_n1, &flip_ldi_1, &flip_ldo_1, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
    if ( trans_unary_flip_1 == NULL) {
        fprintf( stderr, "JIT for unary NORM_TO_NORMT TPP. Bailing...!\n");
        exit(-1);
    }
    // Convert (F_t, C_t, WW_t) -----> (WW_t, F_t, C_t)
    libxsmm_meltw_unary_param trans_unary_param_flip_1;
    trans_unary_param_flip_1.in.primary  = flip_weight_a;
    trans_unary_param_flip_1.out.primary = weight_buffer_a;
    trans_unary_flip_1( &trans_unary_param_flip_1 );

    libxsmm_blasint flip_m2 = C_t;
    libxsmm_blasint flip_n2 = F_t;
    libxsmm_blasint flip_ldi_2 = C_t;
    libxsmm_blasint flip_ldo_2 = F_t;

    libxsmm_meltwfunction_unary trans_unary_flip_2 = libxsmm_dispatch_meltw_unary(flip_m2, flip_n2, &flip_ldi_2, &flip_ldo_2, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
    if ( trans_unary_flip_2 == NULL) {
        fprintf( stderr, "JIT for unary NORM_TO_NORMT TPP. Bailing...!\n");
        exit(-1);
    }

    // Convert (WW_t, F_t, C_t) -----> (F_t, C_t, WW_t)
    #pragma omp parallel for
    for(int kw = 0; kw < WW_t; kw++){                   // permute last two dimensions
        libxsmm_meltw_unary_param trans_unary_param_flip_2;
        trans_unary_param_flip_2.in.primary  = &weight_buffer_a[kw*C_t*F_t];
        trans_unary_param_flip_2.out.primary = &flip_weight_a[kw*C_t*F_t];
        trans_unary_flip_2( &trans_unary_param_flip_2 );
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

// #ifdef USE_TPP
    // Virtual copy kernels
    libxsmm_blasint virtual_m1 = pad_tile_multiple - ((WW_t - 1)*dial);                      // columns
    libxsmm_blasint virtual_m2 = ((WW_t - 1)*dial);                      // columns
    libxsmm_blasint virtual_n = F_t;                                        // rows
    libxsmm_blasint ldi_virtual = W_t;
    libxsmm_blasint ldo_virtual = 2*pad_tile_multiple;

    if (ldi_virtual < virtual_m1){                      // corner case when width's are very small
        virtual_m1 = ldi_virtual;
        libxsmm_meltwfunction_unary all_zero = libxsmm_dispatch_meltw_unary(ldo_virtual, virtual_n, NULL, &ldo_virtual, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
        if ( all_zero == NULL) {
            fprintf( stderr, "JIT for initialization by unary all zero copy kernel failed. Bailing...!\n");
            exit(-1);
        }
        #pragma omp parallel for
        for(int n = 0; n < N_t; n++){
            libxsmm_meltw_unary_param all_zero_params;
            all_zero_params.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual];                 // Initialize the entire array when widths are small
            all_zero(&all_zero_params);
        }
    }

    libxsmm_meltwfunction_unary virtual_copy = libxsmm_dispatch_meltw_unary(virtual_m1, virtual_n, &ldi_virtual, &ldo_virtual, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
    libxsmm_meltwfunction_unary virtual_copy_zero = libxsmm_dispatch_meltw_unary(virtual_m2, virtual_n, NULL, &ldo_virtual, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
    if ( virtual_copy == NULL || virtual_copy_zero == NULL) {
        fprintf( stderr, "JIT for initialization by unary copy kernel failed. Bailing...!\n");
        exit(-1);
    }

    #pragma omp parallel for
    for(int n = 0; n < N_t; n++){                         // Loops for storing the edge portion of gradinant array into grad_a_shortpad

        libxsmm_meltw_unary_param vcopy_params;             // Copy parameter variable for holding the pointer
        libxsmm_meltw_unary_param vcopy_params_zero;

        vcopy_params_zero.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual];                                        // copy zeros
        virtual_copy_zero(&vcopy_params_zero);

        vcopy_params.in.primary = &grad_a[n*F_t*W_t];                                                              // copy after zeros from start of the grad array
        vcopy_params.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual + ((WW_t - 1)*dial)];
        virtual_copy(&vcopy_params);

        vcopy_params.in.primary = &grad_a[n*F_t*W_t + W_t - virtual_m1];              // copy from the end of the grad array
        vcopy_params.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual + ldo_virtual - virtual_m1 - ((WW_t - 1)*dial)];
        virtual_copy(&vcopy_params);

        vcopy_params_zero.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual + ldo_virtual - ((WW_t - 1)*dial)];     // copy zeros
        virtual_copy_zero(&vcopy_params_zero);
    }

// #else

//     #pragma omp parallel for
//     for(int n = 0; n < N_t; n++){                       // Loops for storing the edge portion of gradinant array into grad_a_shortpad
//         for(int filter=0; filter < F_t; filter++){
//             for(int w = 0; w < pad_tile_multiple; w++){
//                 // initialize start of array
//                 if (w >= ((WW_t - 1)*dial) && w < (W_t + (WW_t - 1)*dial)){
//                     grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w] = grad_a[n*F_t*W_t + filter*W_t + w - (WW_t - 1)*dial];
//                 }
//                 else{
//                     grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w] = 0.0f;
//                 }
//             }
//             for(int w = Wpad_t - pad_tile_multiple; w < Wpad_t ; w++){
//                 // initialize end of array
//                 if (w >= ((WW_t - 1)*dial) && w < (W_t + (WW_t - 1)*dial)){
//                     grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w - Wpad_t + 2*pad_tile_multiple] = grad_a[n*F_t*W_t + filter*W_t + w - (WW_t - 1)*dial];
//                 }
//                 else{
//                     grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w - Wpad_t + 2*pad_tile_multiple] = 0.0f;
//                 }
//             }
//         }
//     }

// #endif

    /* JIT eltwise TPPs for initialization... */
    libxsmm_blasint tpp_m1 = XS_TILE_DBACKWARD;                      // columns
    libxsmm_blasint tpp_m2 = Win_t - tile_multiple;                      // columns
    libxsmm_blasint tpp_n = C_t;      // rows
    libxsmm_blasint ld_zero = Win_t;

    libxsmm_meltwfunction_unary copy_kernel_1 = libxsmm_dispatch_meltw_unary(tpp_m1, tpp_n, NULL, &ld_zero, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
    libxsmm_meltwfunction_unary copy_kernel_2 = libxsmm_dispatch_meltw_unary(tpp_m2, tpp_n, NULL, &ld_zero, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
    if ( copy_kernel_1 == NULL || copy_kernel_2 == NULL) {
        fprintf( stderr, "JIT for initialization by TPP copy kernel failed. Bailing...!\n");
        exit(-1);
    }

    // Main compute kernel
    #pragma omp parallel for
    for(int n = 0; n < N_t; n++) {
        int last_block=0;

        libxsmm_meltw_unary_param copy_params_1;
        libxsmm_meltw_unary_param copy_params_2;
        for(int wb = 0; wb < Win_t - XS_TILE_DBACKWARD + 1; wb += XS_TILE_DBACKWARD) {

            copy_params_1.out.primary = &d_input_a[n*C_t*Win_t + wb];            // Initialization
            copy_kernel_1(&copy_params_1);

            if (wb >= (WW_t-1)*dial && wb < Win_t - (WW_t-1)*dial - XS_TILE_DBACKWARD)              // Normal case
                kernel(&grad_a[n*F_t*W_t + 0*W_t + wb - (WW_t-1)*dial], &flip_weight_a[0], &d_input_a[n*C_t*Win_t + 0*Win_t + wb], &l_br);
            else if (wb < (WW_t-1)*dial)                // Right side case
                kernel4(&grad_a_shortpad[n*F_t*2*pad_tile_multiple + wb], &flip_weight_a[0], &d_input_a[n*C_t*Win_t + wb], &l_br);
            else             // left side case
                kernel4(&grad_a_shortpad[n*F_t*2*pad_tile_multiple + wb - Wpad_t + 2*pad_tile_multiple], &flip_weight_a[0], &d_input_a[n*C_t*Win_t + wb], &l_br);

            last_block = wb;     // store position for last block
        }

        if (Win_t % XS_TILE_DBACKWARD != 0){                                // Edge case

            copy_params_2.out.primary = &d_input_a[n*C_t*Win_t + last_block + XS_TILE_DBACKWARD];            // Initialization
            copy_kernel_2(&copy_params_2);

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

    libxsmm_blasint short_W_t = XS_TILE_WBACKWARD;
    libxsmm_blasint edge_W_t = W_t - tile_multiple;
    libxsmm_blasint M_g = W_t;
    libxsmm_blasint N_g = F_t;


    auto grad_shorttrans_tensor = grad.new_empty({N_t,F_t,short_W_t});              // Tensor for storing transposed short buffer
    float* grad_shorttrans = grad_shorttrans_tensor.data_ptr<float>();

    auto grad_edgetrans_tensor = grad.new_empty({N_t,F_t,edge_W_t});                // Tensor for storing transposed short buffer in edge case
    float* grad_edgetrans = grad_edgetrans_tensor.data_ptr<float>();

    /* use jited tranpose */
    libxsmm_meltwfunction_unary trans_shortkernel_grad = libxsmm_dispatch_meltw_unary(short_W_t, N_g, &M_g, &N_g, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
    libxsmm_meltwfunction_unary trans_edgekernel_grad = libxsmm_dispatch_meltw_unary(edge_W_t, N_g, &M_g, &N_g, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
    if ( trans_shortkernel_grad == NULL | trans_edgekernel_grad == NULL) {
        fprintf( stderr, "JIT for unary NORM_TO_NORMT TPP. Bailing...!\n");
        exit(-1);
    }

    /* Dispatch brGEMM kernel for normal and edge cases*/
    libxsmm_smmfunction kernel_w5 = libxsmm_smmdispatch(F_t, C_t, XS_TILE_WBACKWARD, &ldb_trans_g, &lda_g, &ldc_g, NULL, NULL, NULL, NULL);
    libxsmm_smmfunction kernel_w6 = libxsmm_smmdispatch(F_t, C_t, W_t - tile_multiple, &ldb_trans_g, &lda_g, &ldc_g, NULL, NULL, NULL, NULL);

    // Main compute loop
    #pragma omp parallel for reduction(+: flip_d_weight_a[:F_t*C_t*WW_t])                // Distribute the weight array
    for(int n = 0; n < N_t; n++) {
        int last_block = 0;
        libxsmm_meltw_unary_param trans_param_short;                    // Pointer to hold trans short
        libxsmm_meltw_unary_param trans_param_edge;                     // Pointer to hold trans edge

        for(int wb = 0; wb < W_t - XS_TILE_WBACKWARD + 1; wb += XS_TILE_WBACKWARD) {                // Normal case

            trans_param_short.in.primary  = &grad_a[n*F_t*W_t + wb];
            trans_param_short.out.primary = &grad_shorttrans[n*F_t*short_W_t];
            trans_shortkernel_grad( &trans_param_short );

            for(int kw = 0; kw < WW_t; kw++) {
                kernel_w5(&grad_shorttrans[n*F_t*short_W_t], &input_a[n*C_t*Win_t + wb + kw*dial], &flip_d_weight_a[kw*C_t*F_t]);
            }
            last_block = wb;
        }

        if (W_t % XS_TILE_WBACKWARD != 0){

            trans_param_edge.in.primary  = &grad_a[n*F_t*W_t + last_block + XS_TILE_WBACKWARD];
            trans_param_edge.out.primary = &grad_edgetrans[n*F_t*edge_W_t];
            trans_edgekernel_grad( &trans_param_edge );

            for(int kw = 0; kw < WW_t; kw++) {

                kernel_w6(&grad_edgetrans[n*F_t*edge_W_t], &input_a[n*C_t*Win_t + (last_block + XS_TILE_WBACKWARD) + kw*dial], &flip_d_weight_a[kw*F_t*C_t]);
            }
        }
    }


    /* jited tranpose to permute the array dimensions
        Overall Convert (WW_t, C_t, F_t) -----> (F_t, C_t, WW_t)*/
    libxsmm_blasint per_m1 = F_t;
    libxsmm_blasint per_n1 = C_t;
    libxsmm_blasint ldi_1 = F_t;
    libxsmm_blasint ldo_1 = C_t;

    libxsmm_meltwfunction_unary trans_permute_1 = libxsmm_dispatch_meltw_unary(per_m1, per_n1, &ldi_1, &ldo_1, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
    if ( trans_permute_1 == NULL) {
        fprintf( stderr, "JIT for unary NORM_TO_NORMT TPP. Bailing...!\n");
        exit(-1);
    }

    // Convert (WW_t, C_t, F_t) -----> (WW_t, F_t, C_t)
    #pragma omp parallel for
    for(int kw = 0; kw < WW_t; kw++){                   // permute last two dimensions
        libxsmm_meltw_unary_param trans_param_permute_1;
        trans_param_permute_1.in.primary  = &flip_d_weight_a[kw*C_t*F_t];
        trans_param_permute_1.out.primary = &flip_weight_a[kw*C_t*F_t];
        trans_permute_1( &trans_param_permute_1 );
    }


    libxsmm_blasint per_m2 = F_t*C_t;
    libxsmm_blasint per_n2 = WW_t;
    libxsmm_blasint ldi_2 = F_t*C_t;
    libxsmm_blasint ldo_2 = WW_t;

    libxsmm_meltwfunction_unary trans_permute_2 = libxsmm_dispatch_meltw_unary(per_m2, per_n2, &ldi_2, &ldo_2, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
    if ( trans_permute_2 == NULL) {
        fprintf( stderr, "JIT for unary NORM_TO_NORMT TPP. Bailing...!\n");
        exit(-1);
    }

    // Convert (WW_t, F_t, C_t) -----> (F_t, C_t, WW_t)
    libxsmm_meltw_unary_param trans_param_permute_2;
    trans_param_permute_2.in.primary  = flip_weight_a;
    trans_param_permute_2.out.primary = d_weight_a;
    trans_permute_2( &trans_param_permute_2 );


    return {d_input, d_weight};         // return data gradiant and weight gradiant
}


at::Tensor relu_forward_bf16(at::Tensor& input){

    // RECORD_FUNCTION("ReLU_forward_bf16", std::vector<c10::IValue>({input}));           // For recording time

    int64_t N_t = input.size(0);                    // Batch
    int64_t C_t = input.size(1);                    // Channel
    int64_t W_t = input.size(2);                    // input width

    libxsmm_bfloat16* input_a = (libxsmm_bfloat16*) input.data_ptr<at::BFloat16>();

    libxsmm_blasint tpp_m = W_t;                      // columns
    libxsmm_blasint tpp_n = C_t;                  // rows
    libxsmm_blasint ld = W_t;

    libxsmm_meltw_unary_type unary_type;
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_RELU;
    libxsmm_meltw_unary_flags unary_flags;
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    libxsmm_meltwfunction_unary relu_fwd_kernel = libxsmm_dispatch_meltw_unary(tpp_m, tpp_n, &ld, &ld, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, unary_flags, unary_type);

    if ( relu_fwd_kernel == NULL ) {
        fprintf( stderr, "JIT for TPP relu_fwd_kernel failed. Bailing...!\n");
        exit(-1);
    }

    #pragma omp parallel for
    for(int n = 0; n < N_t; n++) {
        libxsmm_meltw_unary_param relu_params;
        relu_params.in.primary   = &input_a[n*C_t*W_t];
        relu_params.out.primary  = &input_a[n*C_t*W_t];
        relu_params.out.secondary = NULL;
        relu_fwd_kernel(&relu_params);
    }

    return input;
}

at::Tensor relu_backward_bf16(at::Tensor& grad, at::Tensor& output){

    // RECORD_FUNCTION("ReLU_backward_bf16", std::vector<c10::IValue>({grad, output}));        // For recording time

    int64_t N_t = grad.size(0);                    // Batch
    int64_t C_t = grad.size(1);                    // Channel
    int64_t W_t = grad.size(2);                    // input width

    unsigned short* output_a = (unsigned short*) output.data_ptr<at::BFloat16>();
    libxsmm_bfloat16* grad_a = (libxsmm_bfloat16*) grad.data_ptr<at::BFloat16>();

    libxsmm_blasint tpp_m = W_t;                      // columns
    libxsmm_blasint tpp_n = C_t;                                // rows
    libxsmm_blasint ld = W_t;

    libxsmm_meltw_unary_type unary_type;
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_RELU_INV;
    libxsmm_meltw_unary_flags unary_flags;
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    libxsmm_meltwfunction_unary relu_bwd_kernel = libxsmm_dispatch_meltw_unary(tpp_m, tpp_n, &ld, &ld, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, unary_flags, unary_type);

    if ( relu_bwd_kernel == NULL ) {
        fprintf( stderr, "JIT for TPP relu_bwd_kernel failed. Bailing...!\n");
        exit(-1);
    }

    #pragma omp parallel for
    for(int n = 0; n < N_t; n++) {
        libxsmm_meltw_unary_param relu_params;
        relu_params.in.primary   = &grad_a[n*C_t*W_t];
        relu_params.out.primary  = &grad_a[n*C_t*W_t];
        relu_params.in.secondary = &output_a[n*C_t*W_t];
        relu_bwd_kernel(&relu_params);
    }

    return grad;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("forward", &Conv1dOpti_forward_libxsmm, "Conv1dOpti lib forward");
m.def("backward", &Conv1dOpti_backward_libxsmm, "Conv1dOpti lib backward");
m.def("forward_bf16", &Conv1dOpti_forward_bf16_libxsmm, "Conv1dOpti bf16 forward");
m.def("backward_bf16", &Conv1dOpti_backward_bf16_libxsmm, "Conv1dOpti bf16 backward");
m.def("relu_forward_bf16", &relu_forward_bf16, "ReLU bf16 forward");
m.def("relu_backward_bf16", &relu_backward_bf16, "ReLU bf16 backward");
}
