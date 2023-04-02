/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Narendra Chaudhary, Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/


#include <immintrin.h>
#include <iostream>
#include <stdio.h>
#include <torch/extension.h>
#include <tuple>
#include <omp.h>
#include <libxsmm.h>
#include <utils/libxsmm_intrinsics_x86.h>

/* #include <torch/csrc/autograd/record_function.h> */
#include <ATen/record_function.h>

#define PCL_ASSERT(cond, x...) do { if (!(cond)) { printf(x); fflush(stdout); exit(1); } } while(0)

#define XS_TILE_FORWARD 64
#define XS_TILE_DBACKWARD 64
#define XS_TILE_WBACKWARD 64                /* 256 for peak performance */


at::Tensor Conv1dOpti_forward_bf16_libxsmm(at::Tensor& input, at::Tensor& weight, int dilation) {

    /* RECORD_FUNCTION("Conv1dOpti_forward_bf16", std::vector<c10::IValue>({input, weight}));        // For recording time */

    int64_t N_t = input.size(0);                    /* Batch */
    int64_t C_t = input.size(1);                    /* Channel */
    int64_t Win_t = input.size(2);                  /* input width */

    int64_t F_t = weight.size(0);                   /* Number of filters */
    int64_t WW_t = weight.size(2);                  /* filter width */

    int64_t dial = dilation;                        /* dilation parameter */
    int64_t pad_size = ((WW_t- 1))*dial;            /* Total padding size */
    int64_t W_t = Win_t - pad_size;                 /* output width */

    auto Y = input.new_empty({N_t,F_t,W_t});        /* New tensor for output */

    libxsmm_bfloat16* input_a = (libxsmm_bfloat16*) input.data_ptr<at::BFloat16>();                /* Get BFloat16 data pointers for accessing tensors */
    libxsmm_bfloat16* weight_a = (libxsmm_bfloat16*) weight.data_ptr<at::BFloat16>();
    libxsmm_bfloat16* Y_a = (libxsmm_bfloat16*) Y.data_ptr<at::BFloat16>();

    auto flip_weight = weight.new_empty({WW_t,F_t,C_t});                                           /* Weight tensor with permuted dimension (width, filters, channels) */
    libxsmm_bfloat16* flip_weight_a = (libxsmm_bfloat16*) flip_weight.data_ptr<at::BFloat16>();    /* Get BFloat16 data pointers for accessing the tensor */

    /* jited transpose to permute the array dimensions
        Overall convert (F_t, C_t, WW_t) -----> (WW_t, F_t, C_t)*/
    libxsmm_blasint per_m = WW_t;
    libxsmm_blasint per_n = F_t*C_t;
    libxsmm_blasint per_ldi = WW_t;
    libxsmm_blasint per_ldo = F_t*C_t;

    libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape( per_m, per_n, per_ldi, per_ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    libxsmm_meltwfunction_unary trans_permute_kernel_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( trans_permute_kernel_bf16 == NULL) {
        fprintf( stderr, "JIT unary TPP for trans_permute_kernel (NORM_TO_NORM transform) in forward pass failed. Bailing...!\n");
        exit(-1);
    }
    libxsmm_meltw_unary_param trans_permute_param;
    trans_permute_param.in.primary  = weight_a;
    trans_permute_param.out.primary = flip_weight_a;
    trans_permute_kernel_bf16( &trans_permute_param);

    int lda = C_t;                      /* Input channels (16) */
    /*int ldb = Win_t;                     Input width    (60400) */
    int ldc = W_t;                      /* Output width   (60000) */
    unsigned long long l_br = WW_t;     /* Number of batches in brGEMM (= width of kernel = 51) */

    int tile_multiple = (W_t/XS_TILE_FORWARD)*XS_TILE_FORWARD;                                          /* Number of blocks/Tiles in the output width */

    int main_width = ((XS_TILE_FORWARD + (WW_t-1)*dial)/XS_TILE_FORWARD + 1)*XS_TILE_FORWARD;          /* width of main buffer */
    auto input_mainvnni = input.new_empty({N_t,C_t,main_width});                                       /* VNNI transformed array of the main buffer */
    libxsmm_bfloat16* input_a_mainvnni = (libxsmm_bfloat16*) input_mainvnni.data_ptr<at::BFloat16>();  /* Get pointer */


    int edge_width = (((W_t - tile_multiple) + (WW_t-1)*dial)/XS_TILE_FORWARD + 1)*XS_TILE_FORWARD;     /* width of buffer in the edge case (last block) */
    auto input_edgevnni = input.new_empty({N_t,C_t,edge_width});                                        /* VNNI VNNI transformed array of the edge buffer */
    libxsmm_bfloat16* input_a_edgevnni = (libxsmm_bfloat16*) input_edgevnni.data_ptr<at::BFloat16>();   /* Get pointer */


    /* Dispatch brGEMM kernels for the normal case and the edge case*/
    libxsmm_gemm_flags l_flags;
    libxsmm_gemm_prefetch_type l_prefetch;
    libxsmm_gemm_shape l_shape;
    libxsmm_gemm_batch_reduce_config l_brconfig;

    l_flags = LIBXSMM_GEMM_FLAG_VNNI_A;
    l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;

    l_shape = libxsmm_create_gemm_shape(XS_TILE_FORWARD, F_t, C_t, main_width, lda, ldc, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16);
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
    l_brconfig.br_stride_a_hint = dial*2*sizeof(libxsmm_bfloat16);
    l_brconfig.br_stride_b_hint = F_t*C_t*sizeof(libxsmm_bfloat16);
    libxsmm_gemmfunction brgemm_kernel_main_bf16 = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch, l_brconfig);

    l_shape = libxsmm_create_gemm_shape(W_t - tile_multiple, F_t, C_t, edge_width, lda, ldc, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16);
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
    l_brconfig.br_stride_a_hint = dial*2*sizeof(libxsmm_bfloat16);
    l_brconfig.br_stride_b_hint = F_t*C_t*sizeof(libxsmm_bfloat16);
    libxsmm_gemmfunction brgemm_kernel_edge_bf16 = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch, l_brconfig);

    /* JIT eltwise TPPs for initialization ... */
    libxsmm_blasint tpp_m1 = XS_TILE_FORWARD;                      /* columns */
    libxsmm_blasint tpp_m2 = W_t - tile_multiple;                  /* columns */
    libxsmm_blasint tpp_n = F_t;                                   /* rows */
    libxsmm_blasint ld_zero = W_t;

    unary_shape = libxsmm_create_meltw_unary_shape( tpp_m1, tpp_n, tpp_m1, ld_zero, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    libxsmm_meltwfunction_unary copy_kernel_forward_main_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    unary_shape = libxsmm_create_meltw_unary_shape( tpp_m2, tpp_n, tpp_m2, ld_zero, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    libxsmm_meltwfunction_unary copy_kernel_forward_edge_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

    if ((copy_kernel_forward_main_bf16 == NULL) || (copy_kernel_forward_edge_bf16 == NULL)) {
        fprintf( stderr, "JIT unary TPP for copy_kernel_forward_main_bf16 in forward pass failed. Bailing...!\n");
        exit(-1);
    }

    /* use jited VNNI */
    libxsmm_blasint ldi = Win_t;
    libxsmm_blasint ldo_main = main_width;
    libxsmm_blasint ldo_edge = edge_width;

    libxsmm_meltw_unary_type trans_vnni_type;
    if ( C_t % 2 == 1 ) {
        trans_vnni_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI_PAD;
    } else {
        trans_vnni_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI;
    }
    tpp_m1 = (XS_TILE_FORWARD + dial*(WW_t-1));
    tpp_m2 = (W_t - tile_multiple + dial*(WW_t-1));

    unary_shape = libxsmm_create_meltw_unary_shape( tpp_m1, C_t, ldi, ldo_main, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    libxsmm_meltwfunction_unary trans_mainvnni_kernel = libxsmm_dispatch_meltw_unary_v2( trans_vnni_type, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    unary_shape = libxsmm_create_meltw_unary_shape( tpp_m2, C_t, ldi, ldo_edge, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    libxsmm_meltwfunction_unary trans_edgevnni_kernel = libxsmm_dispatch_meltw_unary_v2( trans_vnni_type, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

    if ( (trans_mainvnni_kernel == NULL) || (trans_edgevnni_kernel == NULL)) {
        fprintf( stderr, "JIT unary TPP for trans_mainvnni_kernel (NORM_TO_VNNI transform) in forward pass failed. Bailing...!\n");
        exit(-1);
    }

    /* Main compute loop */
    #pragma omp parallel for
    for (int n = 0; n < N_t; n++) {                                                      /* Loop for batches */
        int last_block = 0;
        libxsmm_meltw_unary_param copy_params_main, copy_params_edge;                  /* Copy parameter variable for holding the pointer */
        libxsmm_meltw_unary_param trans_param_main, trans_param_edge;
        libxsmm_gemm_param gemm_param_main, gemm_param_edge;

        for (int wb = 0; wb < W_t - XS_TILE_FORWARD + 1; wb += XS_TILE_FORWARD) {        /* width blocking loop (Main case) */

            copy_params_main.out.primary = &Y_a[n*F_t*W_t + wb];                       /* Initialization of output array */
            copy_kernel_forward_main_bf16(&copy_params_main);

            /* VNNI transform */
            trans_param_main.in.primary  = &input_a[n*C_t*Win_t + 0*Win_t + wb];
            trans_param_main.out.primary = &input_a_mainvnni[n*C_t*main_width];
            trans_mainvnni_kernel( &trans_param_main );

            /* brGEMM */
            gemm_param_main.a.primary = &input_a_mainvnni[n*C_t*main_width];
            gemm_param_main.b.primary = &flip_weight_a[0];
            gemm_param_main.c.primary = &Y_a[n*F_t*W_t + 0*W_t + wb];
            gemm_param_main.op.tertiary = &l_br;
            brgemm_kernel_main_bf16( &gemm_param_main );

            last_block = wb;                                                           /* Store value for last block */
        }

        if (W_t % XS_TILE_FORWARD != 0) {                                               /* Edge case */

            copy_params_edge.out.primary = &Y_a[n*F_t*W_t + last_block + XS_TILE_FORWARD];                 /* Initialization of output array */
            copy_kernel_forward_edge_bf16(&copy_params_edge);

            /* VNNI transform */
            trans_param_edge.in.primary  = &input_a[n*C_t*Win_t + 0*Win_t + (last_block + XS_TILE_FORWARD)];
            trans_param_edge.out.primary = &input_a_edgevnni[n*C_t*edge_width];
            trans_edgevnni_kernel( &trans_param_edge );

            /* brGEMM */
            gemm_param_edge.a.primary = &input_a_edgevnni[n*C_t*edge_width];
            gemm_param_edge.b.primary = &flip_weight_a[0];
            gemm_param_edge.c.primary = &Y_a[n*F_t*W_t + 0*W_t + (last_block + XS_TILE_FORWARD)];
            gemm_param_edge.op.tertiary = &l_br;
            brgemm_kernel_edge_bf16( &gemm_param_edge );
        }
    }

    return Y;              /* Return output tensor */
}

std::tuple<at::Tensor, at::Tensor> Conv1dOpti_backward_bf16_libxsmm(at::Tensor& grad, at::Tensor& input, at::Tensor& weight, int dilation) {

    /* RECORD_FUNCTION("Conv1dOpti_backward_bf16", std::vector<c10::IValue>({grad, input, weight}));        // For recording time */

    int64_t N_t = input.size(0);                    /* Batch */
    int64_t C_t = input.size(1);                    /* Channel */
    int64_t Win_t = input.size(2);                  /* input width */

    int64_t F_t = weight.size(0);                   /* Number of filters */
    int64_t WW_t = weight.size(2);                  /* filter width */

    int64_t dial = dilation;                        /* dilation parameter */
    int64_t pad_size = ((WW_t- 1))*dial;            /* Total padding size */
    int64_t W_t = Win_t - pad_size;                 /* output width */

    auto d_input = input.new_empty({N_t,C_t,Win_t});            /* declare data gradiant tensor */
    auto d_weight = weight.new_empty({F_t,C_t,WW_t});           /* declare weight gradiant tensor */

    libxsmm_bfloat16* input_a = (libxsmm_bfloat16*) input.data_ptr<at::BFloat16>();         /* Get BFloat16 data pointers for accessing tensors */
    libxsmm_bfloat16* weight_a = (libxsmm_bfloat16*) weight.data_ptr<at::BFloat16>();
    libxsmm_bfloat16* grad_a = (libxsmm_bfloat16*) grad.data_ptr<at::BFloat16>();
    libxsmm_bfloat16* d_input_a = (libxsmm_bfloat16*) d_input.data_ptr<at::BFloat16>();
    libxsmm_bfloat16* d_weight_a = (libxsmm_bfloat16*) d_weight.data_ptr<at::BFloat16>();

    /* Backward Data part of the code */

    auto flip_weight_tensor = weight.new_empty({WW_t,C_t,F_t});                             /* Weight tensor with permuted dimension (width, channels, filters) */
    libxsmm_bfloat16* flip_weight_a = (libxsmm_bfloat16*) flip_weight_tensor.data_ptr<at::BFloat16>();   /* Get pointer */


    auto weight_buffer = weight.new_empty({F_t,C_t,WW_t});                  /* Tensor weight buffer */
    libxsmm_bfloat16* weight_buffer_a = (libxsmm_bfloat16*) weight_buffer.data_ptr<at::BFloat16>();

    #pragma omp parallel for
    for (int i = 0; i < F_t*C_t; i++) {
        for (int kw = 0; kw < WW_t; kw++) {                                   /* reverse copy */
            flip_weight_a[i*WW_t + kw] = weight_a[i*WW_t + WW_t - kw - 1];
        }
    }

    /* jited transpose to permute the array dimensions
        Overall convert (F_t, C_t, WW_t) -----> (WW_t, C_t, F_t)*/
    libxsmm_blasint flip_m1 = WW_t;
    libxsmm_blasint flip_n1 = F_t*C_t;
    libxsmm_blasint flip_ldi_1 = WW_t;
    libxsmm_blasint flip_ldo_1 = F_t*C_t;

    libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape( flip_m1, flip_n1, flip_ldi_1, flip_ldo_1, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    libxsmm_meltwfunction_unary trans_flip_1 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( trans_flip_1 == NULL) {
        fprintf( stderr, "JIT unary TPP for trans_flip_1 (NORM_TO_NORM Transform) in backward data failed. Bailing...!\n");
        exit(-1);
    }

    /* Convert (F_t, C_t, WW_t) -----> (WW_t, F_t, C_t) */
    libxsmm_meltw_unary_param trans_param_flip_1;
    trans_param_flip_1.in.primary  = flip_weight_a;
    trans_param_flip_1.out.primary = weight_buffer_a;
    trans_flip_1( &trans_param_flip_1 );

    libxsmm_blasint flip_m2 = C_t;
    libxsmm_blasint flip_n2 = F_t;
    libxsmm_blasint flip_ldi_2 = C_t;
    libxsmm_blasint flip_ldo_2 = F_t;

    unary_shape = libxsmm_create_meltw_unary_shape( flip_m2, flip_n2, flip_ldi_2, flip_ldo_2, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    libxsmm_meltwfunction_unary trans_flip_2 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( trans_flip_2 == NULL) {
        fprintf( stderr, "JIT unary TPP for trans_flip_2 (NORM_TO_NORM Transform) in backward data failed. Bailing...!\n");
        exit(-1);
    }

    /* Convert (WW_t, F_t, C_t) -----> (F_t, C_t, WW_t) */
    #pragma omp parallel for
    for (int kw = 0; kw < WW_t; kw++) {                                     /* permute last two dimensions */
        libxsmm_meltw_unary_param trans_param_flip_2;
        trans_param_flip_2.in.primary  = &weight_buffer_a[kw*C_t*F_t];
        trans_param_flip_2.out.primary = &flip_weight_a[kw*C_t*F_t];
        trans_flip_2( &trans_param_flip_2 );
    }

    int64_t Wpad_t = W_t + 2*(WW_t - 1)*dial;                             /* For padding gradiant on both sides */
    int64_t tile_multiple = (Win_t/XS_TILE_DBACKWARD)*XS_TILE_DBACKWARD;  /* Number of blocks/tiles in Input */

    int lda = F_t;                                                        /* Number of Filters (16) */
    /* int ldb_orig = W_t;                                                //    Output width (60000) */
    /* int ldb = Wpad_t;                                                  //    Extra padded grad input case 60800 */
    int ldc = Win_t;                                                      /* Input width (60400) */
    unsigned long long l_br = WW_t;                                       /* Number of batches in brGEMM (= width of kernel = 51) */

    int pad_tile_multiple = 2 * (((WW_t - 1)*dial)/XS_TILE_DBACKWARD + 1) * XS_TILE_DBACKWARD;       /* Padded block/tile (896) */
    int ldb_shortpad = 2*pad_tile_multiple;                               /* grad padded short buffer (1792) */

    auto grad_shortpad_tensor = grad.new_empty({N_t,F_t,2*pad_tile_multiple});
    libxsmm_bfloat16* grad_a_shortpad = (libxsmm_bfloat16*) grad_shortpad_tensor.data_ptr<at::BFloat16>();   /* short buffer for padded gradiant */

    /* Virtual copy kernels */
    libxsmm_blasint virtual_m1 = pad_tile_multiple - ((WW_t - 1)*dial);     /* columns */
    libxsmm_blasint virtual_m2 = ((WW_t - 1)*dial);                         /* columns */
    libxsmm_blasint virtual_n = F_t;                                        /* rows */
    libxsmm_blasint ldi_virtual = W_t;
    libxsmm_blasint ldo_virtual = 2*pad_tile_multiple;

    if (ldi_virtual < virtual_m1) {                                          /* corner case when width's are very small */
        virtual_m1 = ldi_virtual;
        unary_shape = libxsmm_create_meltw_unary_shape( ldo_virtual, virtual_n, ldo_virtual, ldo_virtual, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
        libxsmm_meltwfunction_unary all_zero_backdata_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
        if ( all_zero_backdata_bf16 == NULL) {
            fprintf( stderr, "JIT unary TPP for all_zero_backdata_bf16 kernel in backward data pass failed. Bailing...!\n");
            exit(-1);
        }
        #pragma omp parallel for
        for (int n = 0; n < N_t; n++) {
            libxsmm_meltw_unary_param all_zero_params;
            all_zero_params.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual];     /* Initialize the entire array when widths are small */
            all_zero_backdata_bf16(&all_zero_params);
        }
    }

    unary_shape = libxsmm_create_meltw_unary_shape( virtual_m1, virtual_n, ldi_virtual, ldo_virtual, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    libxsmm_meltwfunction_unary virtual_copy_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    unary_shape = libxsmm_create_meltw_unary_shape( virtual_m2, virtual_n, virtual_m2, ldo_virtual, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    libxsmm_meltwfunction_unary virtual_copy_zero_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

    if ((virtual_copy_bf16 == NULL) || (virtual_copy_zero_bf16 == NULL)) {
        fprintf( stderr, "JIT unary TPP for virtual_copy_bf16 kernel in backward data pass failed. Bailing...!\n");
        exit(-1);
    }

    #pragma omp parallel for
    for (int n = 0; n < N_t; n++) {                         /* Loops for storing the edge portion of gradinant array into grad_a_shortpad */

        libxsmm_meltw_unary_param vcopy_params;           /* Copy parameter variable for holding the pointer */
        libxsmm_meltw_unary_param vcopy_params_zero;

        vcopy_params_zero.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual];                                        /* copy zeros */
        virtual_copy_zero_bf16(&vcopy_params_zero);

        vcopy_params.in.primary = &grad_a[n*F_t*W_t];                                                               /* copy after zeros from start of the grad array */
        vcopy_params.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual + ((WW_t - 1)*dial)];
        virtual_copy_bf16(&vcopy_params);

        vcopy_params.in.primary = &grad_a[n*F_t*W_t + W_t - virtual_m1];                                            /* copy from the end of the grad array */
        vcopy_params.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual + ldo_virtual - virtual_m1 - ((WW_t - 1)*dial)];
        virtual_copy_bf16(&vcopy_params);

        vcopy_params_zero.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual + ldo_virtual - ((WW_t - 1)*dial)];     /* copy zeros */
        virtual_copy_zero_bf16(&vcopy_params_zero);
    }

/*
#else
    #pragma omp parallel for
    for (int n = 0; n < N_t; n++) {                   // loop to store the edges for gradiant array into grad_a_shortpad buffer
        for (int filter=0; filter < F_t; filter++) {
            for (int w = 0; w < pad_tile_multiple; w++) {
                // initialize start of array
                if (w >= ((WW_t - 1)*dial) && w < (W_t + (WW_t - 1)*dial)) {
                    grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w] = grad_a[n*F_t*W_t + filter*W_t + w - (WW_t - 1)*dial];
                }
                else {
                    grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w] = 0.0f;
                }
            }
            for (int w = Wpad_t - pad_tile_multiple; w < Wpad_t ; w++) {
                // initialize end of array
                if (w >= ((WW_t - 1)*dial) && w < (W_t + (WW_t - 1)*dial)) {
                    grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w - Wpad_t + 2*pad_tile_multiple] = grad_a[n*F_t*W_t + filter*W_t + w - (WW_t - 1)*dial];
                }
                else {
                    grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w - Wpad_t + 2*pad_tile_multiple] = 0.0f;
                }
            }
        }
    }
#endif
*/

    int short_width = ((XS_TILE_DBACKWARD + (WW_t-1)*dial)/XS_TILE_DBACKWARD + 1)*XS_TILE_DBACKWARD;    /* Width of buffer   (512) */

    auto grad_shortvnni_backdata = grad.new_empty({N_t,F_t,short_width});                                 /* Buffer for storing VNNI transform */
    libxsmm_bfloat16* grad_a_shortvnni = (libxsmm_bfloat16*) grad_shortvnni_backdata.data_ptr<at::BFloat16>();

    /* Dispatch brGEMM kernels for the normal case and the edge case*/
    libxsmm_gemm_flags l_flags;
    libxsmm_gemm_prefetch_type l_prefetch;
    libxsmm_gemm_shape l_shape;
    libxsmm_gemm_batch_reduce_config l_brconfig;

    l_flags = LIBXSMM_GEMM_FLAG_VNNI_A;
    l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;

    l_shape = libxsmm_create_gemm_shape(XS_TILE_DBACKWARD, C_t, F_t, short_width, lda, ldc, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16);
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
    l_brconfig.br_stride_a_hint = 2*dial*sizeof(libxsmm_bfloat16);
    l_brconfig.br_stride_b_hint = C_t*F_t*sizeof(libxsmm_bfloat16);
    libxsmm_gemmfunction backdata_shortkernel_main = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch, l_brconfig);

    l_shape = libxsmm_create_gemm_shape(Win_t - tile_multiple, C_t, F_t, short_width, lda, ldc, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16);
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
    l_brconfig.br_stride_a_hint = 2*dial*sizeof(libxsmm_bfloat16);
    l_brconfig.br_stride_b_hint = C_t*F_t*sizeof(libxsmm_bfloat16);
    libxsmm_gemmfunction backdata_shortkernel_edge = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch, l_brconfig);

    if ((backdata_shortkernel_main == NULL) | (backdata_shortkernel_edge == NULL)) {
        fprintf( stderr, "JIT for backdata_shortkernel_main in backward data pass failed. Bailing...!\n");
        exit(-1);
    }

    /* JIT eltwise TPPs for initialization ... */
    libxsmm_blasint tpp_m1 = XS_TILE_DBACKWARD;                      /* columns */
    libxsmm_blasint tpp_m2 = Win_t - tile_multiple;                  /* columns */
    libxsmm_blasint tpp_n = C_t;                                     /* rows */
    libxsmm_blasint ld_zero = Win_t;

    unary_shape = libxsmm_create_meltw_unary_shape( tpp_m1, tpp_n, tpp_m1, ld_zero, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    libxsmm_meltwfunction_unary copy_kernel_main_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    unary_shape = libxsmm_create_meltw_unary_shape( tpp_m2, tpp_n, tpp_m2, ld_zero, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    libxsmm_meltwfunction_unary copy_kernel_edge_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

    if ((copy_kernel_main_bf16 == NULL) || (copy_kernel_edge_bf16  == NULL)) {
        fprintf( stderr, "JIT for copy_kernel_main_bf16 in backward data pass failed. Bailing...!\n");
        exit(-1);
    }

    /* use jited VNNI */
    libxsmm_blasint ldi_1 = W_t;
    libxsmm_blasint ldi_2 = ldb_shortpad;                            /* (1792) */
    libxsmm_blasint ldo = short_width;                               /* (512) */

    libxsmm_meltw_unary_type trans_vnni_type;
    if ( F_t % 2 == 1 ) {
        trans_vnni_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI_PAD;
    } else {
        trans_vnni_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI;
    }

    tpp_m1 = (XS_TILE_DBACKWARD + dial*(WW_t-1));
    tpp_m2 = (XS_TILE_DBACKWARD + dial*(WW_t-1));

    unary_shape = libxsmm_create_meltw_unary_shape( tpp_m1, F_t, ldi_1, ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    libxsmm_meltwfunction_unary trans_shortvnni_kernel_1 = libxsmm_dispatch_meltw_unary_v2( trans_vnni_type, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    unary_shape = libxsmm_create_meltw_unary_shape( tpp_m2, F_t, ldi_2, ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    libxsmm_meltwfunction_unary trans_shortvnni_kernel_2 = libxsmm_dispatch_meltw_unary_v2( trans_vnni_type, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

    if ((trans_shortvnni_kernel_1 == NULL) || (trans_shortvnni_kernel_2 == NULL)) {
        fprintf( stderr, "JIT unary TPP for trans_shortvnni_kernel_1 (NORM_TO_VNN transfor) in backward data pass failed. Bailing...!\n");
        exit(-1);
    }

    /* Main backward data pass loop */
    #pragma omp parallel for
    for (int n = 0; n < N_t; n++) {
        int last_block=0;

        libxsmm_meltw_unary_param copy_params_main, copy_params_edge;                 /* Copy parameter variable for holding the pointer */
        libxsmm_meltw_unary_param trans_param_1, trans_param_2;
        libxsmm_gemm_param gemm_param_main, gemm_param_edge;

        for (int wb = 0; wb < Win_t - XS_TILE_DBACKWARD + 1; wb += XS_TILE_DBACKWARD) {

            copy_params_main.out.primary = &d_input_a[n*C_t*Win_t + wb];             /* Initialization */
            copy_kernel_main_bf16(&copy_params_main);

            if (wb >= (WW_t-1)*dial && wb < Win_t - (WW_t-1)*dial - XS_TILE_DBACKWARD) {
                /* Normal case (Take VNNI transform of a portion of grad_a array ) */

                /* VNNI transform */
                trans_param_1.in.primary  = &grad_a[n*F_t*W_t + 0*W_t + wb - (WW_t-1)*dial];
                trans_param_1.out.primary = &grad_a_shortvnni[n*F_t*short_width];
                trans_shortvnni_kernel_1( &trans_param_1 );

                /* brGEMM */
                gemm_param_main.a.primary = &grad_a_shortvnni[n*F_t*short_width];
                gemm_param_main.b.primary = &flip_weight_a[0];
                gemm_param_main.c.primary = &d_input_a[n*C_t*Win_t + 0*Win_t + wb];
                gemm_param_main.op.tertiary = &l_br;
                backdata_shortkernel_main( &gemm_param_main );
            }
            else if (wb < (WW_t-1)*dial) {
                /* Right side case (Take VNNI transform of grad_a_shortpad array) */

                /* VNNI transform */
                trans_param_2.in.primary  = &grad_a_shortpad[n*F_t*2*pad_tile_multiple + wb];
                trans_param_2.out.primary = &grad_a_shortvnni[n*F_t*short_width];
                trans_shortvnni_kernel_2( &trans_param_2 );

                /* brGEMM */
                gemm_param_main.a.primary = &grad_a_shortvnni[n*F_t*short_width];
                gemm_param_main.b.primary = &flip_weight_a[0];
                gemm_param_main.c.primary = &d_input_a[n*C_t*Win_t + 0*Win_t + wb];
                gemm_param_main.op.tertiary = &l_br;
                backdata_shortkernel_main( &gemm_param_main );
            }
            else {
                /* Left side case (Take VNNI transform of grad_a_shortpad array) */

                /* VNNI transform */
                trans_param_2.in.primary  = &grad_a_shortpad[n*F_t*2*pad_tile_multiple + wb - Wpad_t + 2*pad_tile_multiple];
                trans_param_2.out.primary = &grad_a_shortvnni[n*F_t*short_width];
                trans_shortvnni_kernel_2( &trans_param_2 );

                /* brGEMM */
                gemm_param_main.a.primary = &grad_a_shortvnni[n*F_t*short_width];
                gemm_param_main.b.primary = &flip_weight_a[0];
                gemm_param_main.c.primary = &d_input_a[n*C_t*Win_t + 0*Win_t + wb];
                gemm_param_main.op.tertiary = &l_br;
                backdata_shortkernel_main( &gemm_param_main );
            }
            last_block = wb;
        }

        if (Win_t % XS_TILE_DBACKWARD != 0) {                                /* Edge case */

            /* Right side case (Take VNNI transform of grad_a_shortpad array) */

            copy_params_edge.out.primary = &d_input_a[n*C_t*Win_t + last_block + XS_TILE_DBACKWARD];             /* Initialization */
            copy_kernel_edge_bf16(&copy_params_edge);

            /* VNNI transform */
            trans_param_2.in.primary  = &grad_a_shortpad[n*F_t*2*pad_tile_multiple + last_block + XS_TILE_DBACKWARD - Wpad_t + 2*pad_tile_multiple];
            trans_param_2.out.primary = &grad_a_shortvnni[n*F_t*short_width];
            trans_shortvnni_kernel_2( &trans_param_2 );

            /* brGEMM */
            gemm_param_edge.a.primary = &grad_a_shortvnni[n*F_t*short_width];
            gemm_param_edge.b.primary = &flip_weight_a[0];
            gemm_param_edge.c.primary = &d_input_a[n*C_t*Win_t + last_block + XS_TILE_DBACKWARD];
            gemm_param_edge.op.tertiary = &l_br;
            backdata_shortkernel_edge( &gemm_param_edge );
        }
    }


    /* -------------------------------  Backward Weight part of the code ---------------------------------- */


    float* flip_d_weight_a = (float*) libxsmm_aligned_malloc( F_t*C_t*WW_t*sizeof(float), 64 );             /* Array for permuted weight gradiant */

    for (int w = 0; w < F_t*C_t*WW_t; w++) {          /* Initialize array */
        flip_d_weight_a[w] = 0.0f;
    }

    /*  lda = W_t;                                  // Already defined variables */
    /* ldb = Win_t; */
    /* ldc = C_t; */
    l_br = WW_t;                                    /* Number of batches in brGEMM (= width of kernel = 51) */
    tile_multiple = (W_t/XS_TILE_WBACKWARD)*XS_TILE_WBACKWARD;


    /* Blocking on grad_a */
    int lda_g = Win_t;
    int ldb_trans_g = F_t;
    int ldc_g = F_t;

    libxsmm_blasint M_g = W_t/2;
    libxsmm_blasint N_g = F_t;
    libxsmm_blasint short_W_t = XS_TILE_WBACKWARD;
    libxsmm_blasint edge_W_t = W_t - tile_multiple;

    auto grad_shortvnni_backweight = grad.new_empty({N_t,F_t,short_W_t});                            /* Short buffer for storing VNNI transform */
    libxsmm_bfloat16* grad_shortvnni = (libxsmm_bfloat16*) grad_shortvnni_backweight.data_ptr<at::BFloat16>();

    auto grad_edgevnni_backweight = grad.new_empty({N_t,F_t,edge_W_t});                              /* Short buffer for storing VNNI transform in edge case */
    libxsmm_bfloat16* grad_edgevnni = (libxsmm_bfloat16*) grad_edgevnni_backweight.data_ptr<at::BFloat16>();

    /* use jited transpose */
    unary_shape = libxsmm_create_meltw_unary_shape( short_W_t/2, N_g, M_g, N_g, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    libxsmm_meltwfunction_unary trans_shortkernel_grad = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    unary_shape = libxsmm_create_meltw_unary_shape( edge_W_t/2, N_g, M_g, N_g, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    libxsmm_meltwfunction_unary trans_edgekernel_grad = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

    if ((trans_shortkernel_grad == NULL) || (trans_edgekernel_grad == NULL)) {
        fprintf( stderr, "JIT unary TPP for trans_shortkernel_grad (NORM_TO_NORM Transform) in backward data failed. Bailing...!\n");
        exit(-1);
    }

    /* Dispatch brGEMM kernels for the normal case and the edge case*/
    l_flags = LIBXSMM_GEMM_FLAG_VNNI_A;
    l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;

    l_shape = libxsmm_create_gemm_shape(F_t, C_t, XS_TILE_WBACKWARD, ldb_trans_g, lda_g, ldc_g, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16);
    libxsmm_gemmfunction backweight_kernel_main = libxsmm_dispatch_gemm_v2(l_shape, l_flags, l_prefetch);

    l_shape = libxsmm_create_gemm_shape(F_t, C_t, W_t - tile_multiple, ldb_trans_g, lda_g, ldc_g, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16);
    libxsmm_gemmfunction backweight_kernel_edge = libxsmm_dispatch_gemm_v2(l_shape, l_flags, l_prefetch);

    if ((backweight_kernel_main == NULL) || (backweight_kernel_edge == NULL)) {
        fprintf( stderr, "JIT for backweight_kernel_main failed. Bailing...!\n");
        exit(-1);
    }

    /* Main compute loop for backward weight pass */
    #pragma omp parallel for reduction(+: flip_d_weight_a[:F_t*C_t*WW_t])
    for (int n = 0; n < N_t; n++) {
        int last_block = 0;
        libxsmm_meltw_unary_param trans_param_short, trans_param_edge;
        libxsmm_gemm_param gemm_param_main, gemm_param_edge;

        for (int wb = 0; wb < W_t - XS_TILE_WBACKWARD + 1; wb += XS_TILE_WBACKWARD) {            /* Main Case */

            /* Take transpose assumping FP32 (This will do both transpose and VNNI transform for BF16) */
            trans_param_short.in.primary  = &grad_a[n*F_t*W_t + wb];
            trans_param_short.out.primary = &grad_shortvnni[n*F_t*short_W_t];
            trans_shortkernel_grad( &trans_param_short );

            for (int kw = 0; kw < WW_t; kw++) {
                gemm_param_main.a.primary = &grad_shortvnni[n*F_t*short_W_t];
                gemm_param_main.b.primary = &input_a[n*C_t*Win_t + wb + kw*dial];
                gemm_param_main.c.primary = &flip_d_weight_a[kw*C_t*F_t];
                backweight_kernel_main( &gemm_param_main );
            }
            last_block = wb;
        }

        if (W_t % XS_TILE_WBACKWARD != 0) {              /* Edge Case */

            trans_param_edge.in.primary  = &grad_a[n*F_t*W_t + last_block + XS_TILE_WBACKWARD];
            trans_param_edge.out.primary = &grad_edgevnni[n*F_t*edge_W_t];
            trans_edgekernel_grad( &trans_param_edge );

            for (int kw = 0; kw < WW_t; kw++) {
                gemm_param_edge.a.primary = &grad_edgevnni[n*F_t*edge_W_t];
                gemm_param_edge.b.primary = &input_a[n*C_t*Win_t + (last_block + XS_TILE_WBACKWARD) + kw*dial];
                gemm_param_edge.c.primary = &flip_d_weight_a[kw*F_t*C_t];
                backweight_kernel_edge( &gemm_param_edge );
            }
        }
    }


    auto flip_d_weight_tensor = weight.new_empty({WW_t,C_t,F_t});
    libxsmm_bfloat16* flip_d_weight_bf16 = (libxsmm_bfloat16*) flip_d_weight_tensor.data_ptr<at::BFloat16>();


    /* JIT eltwise TPPs for FP32 to BF16 conversion... */
    libxsmm_blasint cvt_m = 1;
    libxsmm_blasint cvt_n = F_t*C_t*WW_t;

    unary_shape = libxsmm_create_meltw_unary_shape( cvt_m, cvt_n, cvt_m, cvt_m, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    libxsmm_meltwfunction_unary eltwise_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( eltwise_kernel == NULL ) {
        fprintf( stderr, "JIT unary TPP to convert FP32 to BF16 failed. Bailing...!\n");
        exit(-1);
    }

    libxsmm_meltw_unary_param eltwise_params;
    eltwise_params.in.primary = flip_d_weight_a;
    eltwise_params.out.primary = flip_d_weight_bf16;
    eltwise_kernel(&eltwise_params);


    /* jited transpose to permute the array dimensions
        Overall Convert (WW_t, C_t, F_t) -----> (F_t, C_t, WW_t)*/
    libxsmm_blasint per_m1 = F_t;
    libxsmm_blasint per_n1 = C_t;
    libxsmm_blasint ldi_per_1 = F_t;
    libxsmm_blasint ldo_per_1 = C_t;

    unary_shape = libxsmm_create_meltw_unary_shape( per_m1, per_n1, ldi_per_1, ldo_per_1, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    libxsmm_meltwfunction_unary trans_permute_1 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( trans_permute_1 == NULL) {
        fprintf( stderr, "JIT unary TPP trans_permute_1 (NORM_TO_NORM Transform) in backward weight failed. Bailing...!\n");
        exit(-1);
    }

    /* Convert (WW_t, C_t, F_t) -----> (WW_t, F_t, C_t) */
    #pragma omp parallel for
    for (int kw = 0; kw < WW_t; kw++) {                   /* permute last two dimensions */
        libxsmm_meltw_unary_param trans_param_permute_1;
        trans_param_permute_1.in.primary  = &flip_d_weight_bf16[kw*C_t*F_t];
        trans_param_permute_1.out.primary = &flip_weight_a[kw*C_t*F_t];
        trans_permute_1( &trans_param_permute_1 );
    }

    libxsmm_blasint per_m2 = F_t*C_t;
    libxsmm_blasint per_n2 = WW_t;
    libxsmm_blasint ldi_per_2 = F_t*C_t;
    libxsmm_blasint ldo_per_2 = WW_t;

    unary_shape = libxsmm_create_meltw_unary_shape( per_m2, per_n2, ldi_per_2, ldo_per_2, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    libxsmm_meltwfunction_unary trans_permute_2 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( trans_permute_2 == NULL) {
        fprintf( stderr, "JIT unary TPP trans_permute_2 (NORM_TO_NORM Transform) in backward weight failed. Bailing...!\n");
        exit(-1);
    }

    /* Convert (WW_t, F_t, C_t) -----> (F_t, C_t, WW_t) */
    libxsmm_meltw_unary_param trans_param_permute_2;
    trans_param_permute_2.in.primary  = flip_weight_a;
    trans_param_permute_2.out.primary = d_weight_a;
    trans_permute_2( &trans_param_permute_2 );

    libxsmm_free(flip_d_weight_a);

    return {d_input, d_weight};
}


at::Tensor Conv1dOpti_forward_libxsmm(at::Tensor& input, at::Tensor& weight, int dilation) {

    /* RECORD_FUNCTION("Conv1dOpti_forward_libxsmm", std::vector<c10::IValue>({input, weight}));    // For recording time   */

    int64_t N_t = input.size(0);                                 /* Batch */
    int64_t C_t = input.size(1);                                 /* Channel */
    int64_t Win_t = input.size(2);                               /* input width */

    int64_t F_t = weight.size(0);                                /* Number of filters */
    int64_t WW_t = weight.size(2);                               /* filter width */

    int64_t dial = dilation;                                     /* dilation parameter */
    int64_t pad_size = ((WW_t- 1))*dial;                         /* Total padding size */
    int64_t W_t = Win_t - pad_size;                              /* output width */

    auto Y = input.new_empty({N_t,F_t,W_t});                     /* New tensor for output */

    float* input_a = input.data_ptr<float>();                    /* Get pointers for accessing the tensors */
    float* weight_a = weight.data_ptr<float>();
    float* Y_a = Y.data_ptr<float>();

    auto flip_weight = weight.new_empty({WW_t,F_t,C_t});        /* Array to store permuted weight tensor (width, filters, channels) */
    float* flip_weight_a = flip_weight.data_ptr<float>();

    /* jited transpose to permute the array dimensions
        Overall convert (F_t, C_t, WW_t) -----> (WW_t, F_t, C_t)*/

    libxsmm_blasint per_m = WW_t;
    libxsmm_blasint per_n = F_t*C_t;
    libxsmm_blasint per_ldi = WW_t;
    libxsmm_blasint per_ldo = F_t*C_t;

    libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape( per_m, per_n, per_ldi, per_ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    libxsmm_meltwfunction_unary trans_permute_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( trans_permute_kernel == NULL) {
        fprintf( stderr, "JIT unary TPP for trans_permute_kernel (normal to normal transform) isn't working in the forward pass. Bailing...!\n");
        exit(-1);
    }
    libxsmm_meltw_unary_param trans_permute_param;
    trans_permute_param.in.primary  = weight_a;
    trans_permute_param.out.primary = flip_weight_a;
    trans_permute_kernel( &trans_permute_param);

    int lda = C_t;                                              /* Input channels (15) */
    int ldb = Win_t;                                            /* Input width (60400) */
    int ldc = W_t;                                              /* Output width (60000)*/
    unsigned long long l_br = WW_t;

    int tile_multiple = (W_t/XS_TILE_FORWARD)*XS_TILE_FORWARD;

    /* Dispatch SGEMM kernels for the normal case and the edge case*/
    /* setting update GEMM struct */
    libxsmm_gemm_flags l_flags;
    libxsmm_gemm_prefetch_type l_prefetch;
    libxsmm_gemm_shape l_shape;
    libxsmm_gemm_batch_reduce_config l_brconfig;

    l_flags = LIBXSMM_GEMM_FLAG_NONE;
    l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;

    l_shape = libxsmm_create_gemm_shape(XS_TILE_FORWARD, F_t, C_t, ldb, lda, ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
    l_brconfig.br_stride_a_hint = dial*sizeof(float);
    l_brconfig.br_stride_b_hint = F_t*C_t*sizeof(float);
    libxsmm_gemmfunction brgemm_kernel_main = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch, l_brconfig);

    l_shape = libxsmm_create_gemm_shape(W_t - tile_multiple, F_t, C_t, ldb, lda, ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
    l_brconfig.br_stride_a_hint = dial*sizeof(float);
    l_brconfig.br_stride_b_hint = F_t*C_t*sizeof(float);
    libxsmm_gemmfunction brgemm_kernel_edge = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch, l_brconfig);

    /* JIT eltwise TPPs for initialization... */
    libxsmm_blasint tpp_m1 = XS_TILE_FORWARD;                      /* columns */
    libxsmm_blasint tpp_m2 = W_t - tile_multiple;                  /* columns */
    libxsmm_blasint tpp_n = F_t;                                   /* rows */
    libxsmm_blasint ld_zero = W_t;

    libxsmm_meltw_unary_type unary_type;
    unary_type = LIBXSMM_MELTW_TYPE_UNARY_XOR;
    libxsmm_meltw_unary_flags unary_flags;
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;

    unary_shape = libxsmm_create_meltw_unary_shape( tpp_m1, tpp_n, tpp_m1, ld_zero, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    libxsmm_meltwfunction_unary zero_kernel_main = libxsmm_dispatch_meltw_unary_v2( unary_type, unary_shape, unary_flags );
    unary_shape = libxsmm_create_meltw_unary_shape( tpp_m2, tpp_n, tpp_m2, ld_zero, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    libxsmm_meltwfunction_unary zero_kernel_edge = libxsmm_dispatch_meltw_unary_v2( unary_type, unary_shape, unary_flags );

    if ((zero_kernel_main == NULL) || (zero_kernel_edge == NULL)) {
        fprintf( stderr, "JIT UNARY kernel for initilizing zeros failed in forward pass. Bailing...!\n");
        exit(-1);
    }

    /* Main compute loop */
    #pragma omp parallel for
    for (int n = 0; n < N_t; n++) {                               /* Loop for batches */
        int last_block = 0;
        libxsmm_meltw_unary_param zero_param_main, zero_param_edge;
        libxsmm_gemm_param gemm_param_main, gemm_param_edge;

        for (int wb = 0; wb < W_t - XS_TILE_FORWARD + 1; wb += XS_TILE_FORWARD) {    /* width blocking loop (Main case) */

            zero_param_main.out.primary = &Y_a[n*F_t*W_t + wb];       /* Initialization */
            zero_kernel_main( &zero_param_main );

            gemm_param_main.a.primary = &input_a[n*C_t*Win_t + 0*Win_t + wb];
            gemm_param_main.b.primary = &flip_weight_a[0];
            gemm_param_main.c.primary = &Y_a[n*F_t*W_t + 0*W_t + wb];
            gemm_param_main.op.tertiary = &l_br;
            brgemm_kernel_main( &gemm_param_main );

            last_block = wb;
        }

        if (W_t % XS_TILE_FORWARD != 0) {                        /* Edge Case */

            zero_param_edge.out.primary = &Y_a[n*F_t*W_t + last_block + XS_TILE_FORWARD];     /* Initialization */
            zero_kernel_edge( &zero_param_edge );

            gemm_param_edge.a.primary = &input_a[n*C_t*Win_t + 0*Win_t + last_block + XS_TILE_FORWARD];
            gemm_param_edge.b.primary = &flip_weight_a[0];
            gemm_param_edge.c.primary = &Y_a[n*F_t*W_t + 0*W_t + last_block + XS_TILE_FORWARD];
            gemm_param_edge.op.tertiary = &l_br;
            brgemm_kernel_edge( &gemm_param_edge );
        }
    }

    return Y;           /* Return output array */
}

std::tuple<at::Tensor, at::Tensor>
Conv1dOpti_backward_libxsmm(at::Tensor& grad, at::Tensor& input, at::Tensor& weight, int dilation) {

    /* RECORD_FUNCTION("Conv1dOpti_backward_libxsmm", std::vector<c10::IValue>({grad, input, weight})); */

    int64_t N_t = input.size(0);                                /* Batch */
    int64_t C_t = input.size(1);                                /* Channel */
    int64_t Win_t = input.size(2);                              /* input width */

    int64_t F_t = weight.size(0);                               /* Number of filters */
    int64_t WW_t = weight.size(2);                              /* filter width */

    int64_t dial = dilation;                                    /* dilation parameter */
    int64_t pad_size = ((WW_t- 1))*dial;                        /* Total padding size */
    int64_t W_t = Win_t - pad_size;                             /* output width */

    auto d_input = input.new_empty({N_t,C_t,Win_t});            /* declare data gradiant tensor */
    auto d_weight = weight.new_empty({F_t,C_t,WW_t});           /* declare weight gradiant tensor */

    float* input_a = input.data_ptr<float>();                   /* Get data pointers for accessing tensors */
    float* weight_a = weight.data_ptr<float>();
    float* grad_a = grad.data_ptr<float>();
    float* d_input_a = d_input.data_ptr<float>();
    float* d_weight_a = d_weight.data_ptr<float>();

    /*  Backward data part of the code */

    auto flip_weight = weight.new_empty({WW_t,C_t,F_t});                    /* Tensor for permuted weights (width, channels, filters) */
    float* flip_weight_a = flip_weight.data_ptr<float>();


    auto weight_buffer = weight.new_empty({F_t,C_t,WW_t});                  /* Tensor weight buffer */
    float* weight_buffer_a = weight_buffer.data_ptr<float>();

    #pragma omp parallel for
    for (int i = 0; i < F_t*C_t; i++) {
        for (int kw = 0; kw < WW_t; kw++) {                                   /* reverse copy */
            flip_weight_a[i*WW_t + kw] = weight_a[i*WW_t + WW_t - kw - 1];
        }
    }

    /* jited transpose to permute the array dimensions
        Overall convert (F_t, C_t, WW_t) -----> (WW_t, C_t, F_t)*/

    libxsmm_blasint flip_m1 = WW_t;
    libxsmm_blasint flip_n1 = F_t*C_t;
    libxsmm_blasint flip_ldi_1 = WW_t;
    libxsmm_blasint flip_ldo_1 = F_t*C_t;

    libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape( flip_m1, flip_n1, flip_ldi_1, flip_ldo_1, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    libxsmm_meltwfunction_unary trans_unary_flip_1 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( trans_unary_flip_1 == NULL) {
        fprintf( stderr, "JIT unary TPP for trans_unary_flip_1 (NORM_TO_NORMT transform) in backward data pass failed. Bailing...!\n");
        exit(-1);
    }

    /* Convert (F_t, C_t, WW_t) -----> (WW_t, F_t, C_t) */
    libxsmm_meltw_unary_param trans_unary_param_flip_1;
    trans_unary_param_flip_1.in.primary  = flip_weight_a;
    trans_unary_param_flip_1.out.primary = weight_buffer_a;
    trans_unary_flip_1( &trans_unary_param_flip_1 );

    libxsmm_blasint flip_m2 = C_t;
    libxsmm_blasint flip_n2 = F_t;
    libxsmm_blasint flip_ldi_2 = C_t;
    libxsmm_blasint flip_ldo_2 = F_t;

    unary_shape = libxsmm_create_meltw_unary_shape( flip_m2, flip_n2, flip_ldi_2, flip_ldo_2, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    libxsmm_meltwfunction_unary trans_unary_flip_2 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( trans_unary_flip_2 == NULL) {
        fprintf( stderr, "JIT unary TPP for trans_unary_flip_2 (NORM_TO_NORMT transform) in backward data pass failed. Bailing...!\n");
        exit(-1);
    }

    /* Convert (WW_t, F_t, C_t) -----> (F_t, C_t, WW_t) */
    #pragma omp parallel for
    for (int kw = 0; kw < WW_t; kw++) {                          /* permute last two dimensions */
        libxsmm_meltw_unary_param trans_unary_param_flip_2;
        trans_unary_param_flip_2.in.primary  = &weight_buffer_a[kw*C_t*F_t];
        trans_unary_param_flip_2.out.primary = &flip_weight_a[kw*C_t*F_t];
        trans_unary_flip_2( &trans_unary_param_flip_2 );
    }

    int64_t Wpad_t = W_t + 2*(WW_t - 1)*dial;
    int64_t tile_multiple = (Win_t/XS_TILE_DBACKWARD)*XS_TILE_DBACKWARD;

    int lda = F_t;                                              /* Filters (15) */
    int ldb_orig = W_t;                                         /* grad width 60000 */
    /* int ldb = Wpad_t;                                        //    Extra padded grad input case 60800 */
    int ldc = Win_t;                                            /* Input width (60400) */
    unsigned long long l_br = WW_t;                             /* Number of batches for brGEMM (51) */

    libxsmm_gemm_flags l_flags;
    libxsmm_gemm_prefetch_type l_prefetch;
    libxsmm_gemm_shape l_shape;
    libxsmm_gemm_batch_reduce_config l_brconfig;

    l_flags = LIBXSMM_GEMM_FLAG_NONE;
    l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;

    l_shape = libxsmm_create_gemm_shape(XS_TILE_DBACKWARD, C_t, F_t, ldb_orig, lda, ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
    l_brconfig.br_stride_a_hint = dial*sizeof(float);
    l_brconfig.br_stride_b_hint = C_t*F_t*sizeof(float);
    libxsmm_gemmfunction backdata_kernel_main = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch, l_brconfig);

    int pad_tile_multiple = 2 * (((WW_t - 1)*dial)/XS_TILE_DBACKWARD + 1) * XS_TILE_DBACKWARD;       /* 896 */

    auto grad_shortpad_tensor = grad.new_empty({N_t,F_t,2*pad_tile_multiple});
    float* grad_a_shortpad = grad_shortpad_tensor.data_ptr<float>();


    int ldb_shortpad = 2*pad_tile_multiple;                     /* grad pad 1792 */

    /* Dispatch kernels for normal and edge cases*/

    l_shape = libxsmm_create_gemm_shape(XS_TILE_DBACKWARD, C_t, F_t, ldb_shortpad, lda, ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
    l_brconfig.br_stride_a_hint = dial*sizeof(float);
    l_brconfig.br_stride_b_hint = C_t*F_t*sizeof(float);
    libxsmm_gemmfunction backdata_kernel_lr = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch, l_brconfig);

    l_shape = libxsmm_create_gemm_shape(Win_t - tile_multiple, C_t, F_t, ldb_shortpad, lda, ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
    l_brconfig.br_stride_a_hint = dial*sizeof(float);
    l_brconfig.br_stride_b_hint = C_t*F_t*sizeof(float);
    libxsmm_gemmfunction backdata_kernel_edge = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch, l_brconfig);

    /* Virtual copy kernels */
    libxsmm_blasint virtual_m1 = pad_tile_multiple - ((WW_t - 1)*dial);     /* columns */
    libxsmm_blasint virtual_m2 = ((WW_t - 1)*dial);                         /* columns */
    libxsmm_blasint virtual_n = F_t;                                        /* rows */
    libxsmm_blasint ldi_virtual = W_t;
    libxsmm_blasint ldo_virtual = 2*pad_tile_multiple;

    if (ldi_virtual < virtual_m1) {                                          /* corner case when width's are very small */
        virtual_m1 = ldi_virtual;
        unary_shape = libxsmm_create_meltw_unary_shape( ldo_virtual, virtual_n, ldo_virtual, ldo_virtual, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
        libxsmm_meltwfunction_unary all_zero_backdata = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
        if ( all_zero_backdata == NULL) {
            fprintf( stderr, "JIT unary all zero intilization kernel in backward data pass failed. Bailing...!\n");
            exit(-1);
        }
        #pragma omp parallel for
        for (int n = 0; n < N_t; n++) {
            libxsmm_meltw_unary_param all_zero_params;
            all_zero_params.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual];     /* Initialize the entire array when widths are small */
            all_zero_backdata(&all_zero_params);
        }
    }

    unary_shape = libxsmm_create_meltw_unary_shape( virtual_m1, virtual_n, ldi_virtual, ldo_virtual, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    libxsmm_meltwfunction_unary virtual_copy = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    unary_shape = libxsmm_create_meltw_unary_shape( virtual_m2, virtual_n, virtual_m2, ldo_virtual, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    libxsmm_meltwfunction_unary virtual_copy_zero = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

    if ((virtual_copy == NULL) || (virtual_copy_zero == NULL)) {
        fprintf( stderr, "JIT unary kernel of virtual_copy in backward data pass failed. Bailing...!\n");
        exit(-1);
    }

    /* Loops for storing the edge portion of gradinant array into grad_a_shortpad */
    #pragma omp parallel for
    for (int n = 0; n < N_t; n++) {
        libxsmm_meltw_unary_param vcopy_params, vcopy_params_zero;                                                  /* Copy parameter variable for holding the pointer */

        vcopy_params_zero.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual];                                        /* copy zeros */
        virtual_copy_zero(&vcopy_params_zero);

        vcopy_params.in.primary = &grad_a[n*F_t*W_t];                                                              /* copy after zeros from start of the grad array */
        vcopy_params.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual + ((WW_t - 1)*dial)];
        virtual_copy(&vcopy_params);

        vcopy_params.in.primary = &grad_a[n*F_t*W_t + W_t - virtual_m1];                                           /* copy from the end of the grad array */
        vcopy_params.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual + ldo_virtual - virtual_m1 - ((WW_t - 1)*dial)];
        virtual_copy(&vcopy_params);

        vcopy_params_zero.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual + ldo_virtual - ((WW_t - 1)*dial)];     /* copy zeros */
        virtual_copy_zero(&vcopy_params_zero);
    }

/*
#else

    #pragma omp parallel for
    for (int n = 0; n < N_t; n++) {                       // Loops for storing the edge portion of gradinant array into grad_a_shortpad
        for (int filter=0; filter < F_t; filter++) {
            for (int w = 0; w < pad_tile_multiple; w++) {
                // initialize start of array
                if (w >= ((WW_t - 1)*dial) && w < (W_t + (WW_t - 1)*dial)) {
                    grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w] = grad_a[n*F_t*W_t + filter*W_t + w - (WW_t - 1)*dial];
                }
                else {
                    grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w] = 0.0f;
                }
            }
            for (int w = Wpad_t - pad_tile_multiple; w < Wpad_t ; w++) {
                // initialize end of array
                if (w >= ((WW_t - 1)*dial) && w < (W_t + (WW_t - 1)*dial)) {
                    grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w - Wpad_t + 2*pad_tile_multiple] = grad_a[n*F_t*W_t + filter*W_t + w - (WW_t - 1)*dial];
                }
                else {
                    grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w - Wpad_t + 2*pad_tile_multiple] = 0.0f;
                }
            }
        }
    }

#endif
*/

    /* JIT eltwise TPPs for initialization... */
    libxsmm_blasint tpp_m1 = XS_TILE_DBACKWARD;                  /* columns */
    libxsmm_blasint tpp_m2 = Win_t - tile_multiple;              /* columns */
    libxsmm_blasint tpp_n = C_t;                                 /* rows */
    libxsmm_blasint ld_zero = Win_t;

    unary_shape = libxsmm_create_meltw_unary_shape( tpp_m1, tpp_n, tpp_m1, ld_zero, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    libxsmm_meltwfunction_unary copy_kernel_main = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    unary_shape = libxsmm_create_meltw_unary_shape( tpp_m2, tpp_n, tpp_m1, ld_zero, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    libxsmm_meltwfunction_unary copy_kernel_edge = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

    if ((copy_kernel_main == NULL) || (copy_kernel_edge == NULL)) {
        fprintf( stderr, "JIT unary kernel for copy_kernel_main in backward data pass failed. Bailing...!\n");
        exit(-1);
    }

    /* Main compute kernel */
    #pragma omp parallel for
    for (int n = 0; n < N_t; n++) {
        int last_block=0;

        libxsmm_meltw_unary_param copy_params_main, copy_params_edge;
        libxsmm_gemm_param gemm_param_main, gemm_param_lr, gemm_param_edge;

        for (int wb = 0; wb < Win_t - XS_TILE_DBACKWARD + 1; wb += XS_TILE_DBACKWARD) {

            copy_params_main.out.primary = &d_input_a[n*C_t*Win_t + wb];                      /* Initialization */
            copy_kernel_main(&copy_params_main);

            if (wb >= (WW_t-1)*dial && wb < Win_t - (WW_t-1)*dial - XS_TILE_DBACKWARD) {       /* Main case */
                gemm_param_main.a.primary = &grad_a[n*F_t*W_t + 0*W_t + wb - (WW_t-1)*dial];
                gemm_param_main.b.primary = &flip_weight_a[0];
                gemm_param_main.c.primary = &d_input_a[n*C_t*Win_t + wb];
                gemm_param_main.op.tertiary = &l_br;
                backdata_kernel_main( &gemm_param_main );
            }
            else if (wb < (WW_t-1)*dial) {                                                      /* Right side case */
                gemm_param_lr.a.primary = &grad_a_shortpad[n*F_t*2*pad_tile_multiple + wb];
                gemm_param_lr.b.primary = &flip_weight_a[0];
                gemm_param_lr.c.primary = &d_input_a[n*C_t*Win_t + wb];
                gemm_param_lr.op.tertiary = &l_br;
                backdata_kernel_lr( &gemm_param_lr );
            }
            else {                                                                              /* left side case */
                gemm_param_lr.a.primary = &grad_a_shortpad[n*F_t*2*pad_tile_multiple + wb - Wpad_t + 2*pad_tile_multiple];
                gemm_param_lr.b.primary = &flip_weight_a[0];
                gemm_param_lr.c.primary = &d_input_a[n*C_t*Win_t + wb];
                gemm_param_lr.op.tertiary = &l_br;
                backdata_kernel_lr( &gemm_param_lr );
            }

            last_block = wb;                                                                    /* store position for last block */
        }

        if (Win_t % XS_TILE_DBACKWARD != 0) {                                                    /* Edge case */

            copy_params_edge.out.primary = &d_input_a[n*C_t*Win_t + last_block + XS_TILE_DBACKWARD];            /* Initialization */
            copy_kernel_edge(&copy_params_edge);

            gemm_param_edge.a.primary = &grad_a_shortpad[n*F_t*2*pad_tile_multiple + last_block + XS_TILE_DBACKWARD - Wpad_t + 2*pad_tile_multiple];
            gemm_param_edge.b.primary = &flip_weight_a[0];
            gemm_param_edge.c.primary = &d_input_a[n*C_t*Win_t + last_block + XS_TILE_DBACKWARD];
            gemm_param_edge.op.tertiary = &l_br;
            backdata_kernel_edge( &gemm_param_edge );
        }
    }



    /* ------------------------------- Backward weight part of the code --------------------------------- */

    auto flip_d_weight = weight.new_empty({WW_t,C_t,F_t});                  /* Tensor for storing permuted weight gradiant */
    float* flip_d_weight_a = flip_d_weight.data_ptr<float>();

    for (int w = 0; w < F_t*C_t*WW_t; w++) {
        flip_d_weight_a[w] = 0.0f;
    }

    /* lda = W_t; */
    /* ldb = Win_t; */
    /* int ldb_trans = C_t; */
    /* ldc = C_t; */
    l_br = WW_t;
    tile_multiple = (W_t/XS_TILE_WBACKWARD)*XS_TILE_WBACKWARD;

    /* Blocking on grad_a */
    int lda_g = Win_t;
    /* int ldb_g = W_t; */
    int ldb_trans_g = F_t;
    int ldc_g = F_t;

    libxsmm_blasint short_W_t = XS_TILE_WBACKWARD;
    libxsmm_blasint edge_W_t = W_t - tile_multiple;
    libxsmm_blasint M_g = W_t;
    libxsmm_blasint N_g = F_t;


    auto grad_shorttrans_tensor = grad.new_empty({N_t,F_t,short_W_t});              /* Tensor for storing transposed short buffer */
    float* grad_shorttrans = grad_shorttrans_tensor.data_ptr<float>();

    auto grad_edgetrans_tensor = grad.new_empty({N_t,F_t,edge_W_t});                /* Tensor for storing transposed short buffer in edge case */
    float* grad_edgetrans = grad_edgetrans_tensor.data_ptr<float>();

    /* use jited transpose */
    unary_shape = libxsmm_create_meltw_unary_shape( short_W_t, N_g, M_g, N_g, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    libxsmm_meltwfunction_unary trans_shortkernel_grad = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    unary_shape = libxsmm_create_meltw_unary_shape( edge_W_t, N_g, M_g, N_g, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    libxsmm_meltwfunction_unary trans_edgekernel_grad = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

    if ((trans_shortkernel_grad == NULL) || (trans_edgekernel_grad == NULL)) {
        fprintf( stderr, "JIT unary TPP for trans_shortkernel_grad (NORM_TO_NORM transform) failed in backward weight pass. Bailing...!\n");
        exit(-1);
    }

    /* Dispatch brGEMM kernel for normal and edge cases*/
    l_flags = LIBXSMM_GEMM_FLAG_NONE;
    l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;

    l_shape = libxsmm_create_gemm_shape(F_t, C_t, XS_TILE_WBACKWARD, ldb_trans_g, lda_g, ldc_g, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
    libxsmm_gemmfunction backweight_kernel_main = libxsmm_dispatch_gemm_v2(l_shape, l_flags, l_prefetch);

    l_shape = libxsmm_create_gemm_shape(F_t, C_t, W_t - tile_multiple, ldb_trans_g, lda_g, ldc_g, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
    libxsmm_gemmfunction backweight_kernel_edge = libxsmm_dispatch_gemm_v2(l_shape, l_flags, l_prefetch);

    /* Main compute loop for backward weight */
    #pragma omp parallel for reduction(+: flip_d_weight_a[:F_t*C_t*WW_t])                /* Distribute the weight array */
    for (int n = 0; n < N_t; n++) {
        int last_block = 0;
        libxsmm_meltw_unary_param trans_param_short, trans_param_edge;                   /* Pointer to hold trans short and edge */
        libxsmm_gemm_param gemm_param_main, gemm_param_edge;

        for (int wb = 0; wb < W_t - XS_TILE_WBACKWARD + 1; wb += XS_TILE_WBACKWARD) {     /* Normal case */

            trans_param_short.in.primary  = &grad_a[n*F_t*W_t + wb];
            trans_param_short.out.primary = &grad_shorttrans[n*F_t*short_W_t];
            trans_shortkernel_grad( &trans_param_short );

            for (int kw = 0; kw < WW_t; kw++) {
                gemm_param_main.a.primary = &grad_shorttrans[n*F_t*short_W_t];
                gemm_param_main.b.primary = &input_a[n*C_t*Win_t + wb + kw*dial];
                gemm_param_main.c.primary = &flip_d_weight_a[kw*C_t*F_t];
                backweight_kernel_main( &gemm_param_main );
            }
            last_block = wb;
        }

        if (W_t % XS_TILE_WBACKWARD != 0) {

            trans_param_edge.in.primary  = &grad_a[n*F_t*W_t + last_block + XS_TILE_WBACKWARD];
            trans_param_edge.out.primary = &grad_edgetrans[n*F_t*edge_W_t];
            trans_edgekernel_grad( &trans_param_edge );

            for (int kw = 0; kw < WW_t; kw++) {
                gemm_param_edge.a.primary = &grad_edgetrans[n*F_t*edge_W_t];
                gemm_param_edge.b.primary = &input_a[n*C_t*Win_t + (last_block + XS_TILE_WBACKWARD) + kw*dial];
                gemm_param_edge.c.primary = &flip_d_weight_a[kw*F_t*C_t];
                backweight_kernel_edge( &gemm_param_edge );
            }
        }
    }


    /* jited transpose to permute the array dimensions
        Overall Convert (WW_t, C_t, F_t) -----> (F_t, C_t, WW_t)*/
    libxsmm_blasint per_m1 = F_t;
    libxsmm_blasint per_n1 = C_t;
    libxsmm_blasint ldi_1 = F_t;
    libxsmm_blasint ldo_1 = C_t;

    unary_shape = libxsmm_create_meltw_unary_shape( per_m1, per_n1, ldi_1, ldo_1, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    libxsmm_meltwfunction_unary trans_permute_1 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( trans_permute_1 == NULL) {
        fprintf( stderr, "JIT unary TPP for trans_permute_1 (NORM_TO_NORMT) in backward weight pass failed. Bailing...!\n");
        exit(-1);
    }

    /* Convert (WW_t, C_t, F_t) -----> (WW_t, F_t, C_t) */
    #pragma omp parallel for
    for (int kw = 0; kw < WW_t; kw++) {                           /* permute last two dimensions */
        libxsmm_meltw_unary_param trans_param_permute_1;
        trans_param_permute_1.in.primary  = &flip_d_weight_a[kw*C_t*F_t];
        trans_param_permute_1.out.primary = &flip_weight_a[kw*C_t*F_t];
        trans_permute_1( &trans_param_permute_1 );
    }


    libxsmm_blasint per_m2 = F_t*C_t;
    libxsmm_blasint per_n2 = WW_t;
    libxsmm_blasint ldi_2 = F_t*C_t;
    libxsmm_blasint ldo_2 = WW_t;

    unary_shape = libxsmm_create_meltw_unary_shape( per_m2, per_n2, ldi_2, ldo_2, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    libxsmm_meltwfunction_unary trans_permute_2 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( trans_permute_2 == NULL) {
        fprintf( stderr, "JIT unary TPP for trans_permute_2 (NORM_TO_NORMT) in backward weight pass failed. Bailing...!\n");
        exit(-1);
    }

    /* Convert (WW_t, F_t, C_t) -----> (F_t, C_t, WW_t) */
    libxsmm_meltw_unary_param trans_param_permute_2;
    trans_param_permute_2.in.primary  = flip_weight_a;
    trans_param_permute_2.out.primary = d_weight_a;
    trans_permute_2( &trans_param_permute_2 );


    return {d_input, d_weight};         /* return data gradiant and weight gradiant */
}


std::tuple<at::Tensor, at::Tensor> relu_forward_bf16(at::Tensor& input) {

    /* RECORD_FUNCTION("ReLU_forward_bf16", std::vector<c10::IValue>({input}));           // For recording time */

    int64_t N_t = input.size(0);                    /* Batch */
    int64_t C_t = input.size(1);                    /* Channel */
    int64_t W_t = input.size(2);                    /* input width */

    libxsmm_bfloat16* input_a = (libxsmm_bfloat16*) input.data_ptr<at::BFloat16>();

    libxsmm_blasint tpp_m = W_t;                    /* columns */
    libxsmm_blasint tpp_n = C_t;                    /* rows */
    libxsmm_blasint ldi = W_t;

    libxsmm_blasint mask_ld = ((ldi+15)-((ldi+15)%16))/16;
    auto mask = input.new_empty({N_t,C_t,mask_ld});
    unsigned short* mask_a = (unsigned short*) mask.data_ptr<at::BFloat16>();

    libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
    libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape( tpp_m, tpp_n, ldi, ldi, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    libxsmm_meltwfunction_unary relu_fwd_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_RELU, unary_shape, unary_flags );

    #pragma omp parallel for
    for (int n = 0; n < N_t; n++) {
        libxsmm_meltw_unary_param relu_params;
        relu_params.in.primary   = &input_a[n*C_t*W_t];
        relu_params.out.primary  = &input_a[n*C_t*W_t];
        relu_params.out.secondary = &mask_a[n*C_t*mask_ld];
        relu_fwd_kernel(&relu_params);
    }

    return {input, mask};
}

at::Tensor relu_backward_bf16(at::Tensor& grad, at::Tensor& mask) {

    /* RECORD_FUNCTION("ReLU_backward_bf16", std::vector<c10::IValue>({grad, output}));        // For recording time */

    int64_t N_t = grad.size(0);                    /* Batch */
    int64_t C_t = grad.size(1);                    /* Channel */
    int64_t W_t = grad.size(2);                    /* input width */

    libxsmm_bfloat16* grad_a = (libxsmm_bfloat16*) grad.data_ptr<at::BFloat16>();

    libxsmm_blasint tpp_m = W_t;                   /* columns */
    libxsmm_blasint tpp_n = C_t;                   /* rows */
    libxsmm_blasint ldi = W_t;

    libxsmm_blasint mask_ld = ((ldi+15)-((ldi+15)%16))/16;
    unsigned short* mask_a = (unsigned short*) mask.data_ptr<at::BFloat16>();

    libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
    libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape( tpp_m, tpp_n, ldi, ldi, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    libxsmm_meltwfunction_unary relu_bwd_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_RELU_INV, unary_shape, unary_flags );

    #pragma omp parallel for
    for (int n = 0; n < N_t; n++) {
        libxsmm_meltw_unary_param relu_params;
        relu_params.in.primary   = &grad_a[n*C_t*W_t];
        relu_params.out.primary  = &grad_a[n*C_t*W_t];
        relu_params.in.secondary = &mask_a[n*C_t*mask_ld];
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
