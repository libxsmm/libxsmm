/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Kirill Voronin, Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include <libxsmm_dnn_fc.h>

LIBXSMM_API libxsmm_dnn_fc_fwd_config setup_libxsmm_dnn_fc_fwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint K, libxsmm_blasint bn,
                                 libxsmm_blasint bc, libxsmm_blasint bk, libxsmm_blasint threads, libxsmm_dnn_fc_eltw_fuse fuse_type,
                                 libxsmm_datatype datatype_in, libxsmm_datatype datatype_out, libxsmm_datatype datatype_comp ) {
  libxsmm_dnn_fc_fwd_config res;
  libxsmm_blasint lda = bk;
  libxsmm_blasint ldb = bc;
  libxsmm_blasint ldc = bk;
  libxsmm_blasint ld_zero = bk*bn;
  libxsmm_blasint unroll_hint = 0;

  libxsmm_bitfield l_flags, l_tc_flags, l_tr_flags;
  libxsmm_bitfield l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;

  libxsmm_meltw_unary_shape  l_unary_shape;
  libxsmm_bitfield  l_unary_flags;

  libxsmm_gemm_shape l_shape;
  libxsmm_gemm_batch_reduce_config l_brconfig;
  libxsmm_gemm_ext_unary_argops   l_argops;
  libxsmm_gemm_ext_binary_postops l_postops;

  /* init libxsmm structs */
  memset( &l_shape, 0, sizeof(libxsmm_gemm_shape) );
  memset( &l_brconfig, 0, sizeof(libxsmm_gemm_batch_reduce_config) );
  memset( &l_argops, 0, sizeof(libxsmm_gemm_ext_unary_argops) );
  memset( &l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops) );
  memset( &l_unary_shape, 0, sizeof(libxsmm_meltw_unary_shape) );

  /* setting up some handle values */
  res.N = N;
  res.C = C;
  res.K = K;
  res.bn = bn;
  res.bc = bc;
  res.bk = bk;
  res.threads = threads;
  res.fuse_type = fuse_type;

  /* setup parallelization strategy */
  res.fwd_M_hyperpartitions = 1; /* TODO: enable hyperpartitions */
  res.fwd_N_hyperpartitions = 1;
  res.fwd_bf = 1;
  res.fwd_2d_blocking = 0;
  res.fwd_col_teams = 1;
  res.fwd_row_teams = 1;

  if (threads == 16) {
    res.fwd_bf = 1;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 2;
    res.fwd_row_teams = 8;
  } else if (threads == 14) {
    res.fwd_bf = 1;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 2;
    res.fwd_row_teams = 7;
  } else if (threads == 8) {
    res.fwd_bf = 1;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 2;
    res.fwd_row_teams = 4;
  } /*else if (threads == 28) {
    res.fwd_bf = 1;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 1;
    res.fwd_row_teams = 14;
    res.fwd_M_hyperpartitions = 1;
    res.fwd_N_hyperpartitions = 2;
  } */else if (threads == 56) {
    res.fwd_bf = 1;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 1;
    res.fwd_row_teams = 14;
    res.fwd_M_hyperpartitions = 1;
    res.fwd_N_hyperpartitions = 4;
  } else if (threads == 1) {
    res.fwd_bf = 1;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 1;
    res.fwd_row_teams = 1;
    res.fwd_M_hyperpartitions = 1;
    res.fwd_N_hyperpartitions = 1;
  } else {
    res.fwd_bf = 1;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 1;
    res.fwd_row_teams = 1;
  }

  if (res.C == 100 && res.K == 1024 && res.threads == 28) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 14;
    res.fwd_row_teams = 2;
  } else if (res.C == 1024 && res.K == 1024 && res.threads == 28) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 7;
    res.fwd_row_teams = 4;
  } else if (res.C == 100 && res.K == 1024 && res.threads == 40) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 10;
    res.fwd_row_teams = 4;
  } else if (res.C == 1024 && res.K == 1024 && res.threads == 40) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 10;
    res.fwd_row_teams = 4;
  } else if (res.C == 100 && res.K == 1024 && res.threads == 22) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 11;
    res.fwd_row_teams = 2;
  } else if (res.C == 1024 && res.K == 1024 && res.threads == 22) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 11;
    res.fwd_row_teams = 2;
  } else if (res.C == 100 && res.K == 1024 && res.threads == 64) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 8;
    res.fwd_row_teams = 8;
  } else if (res.C == 1024 && res.K == 1024 && res.threads == 64) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 8;
    res.fwd_row_teams = 8;
  } else if (res.C == 512 && res.K == 512 && res.threads == 28) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 1;
    res.fwd_row_teams = 1;
  } else if (res.C == 1024 && res.K == 1 && res.threads == 28) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 1;
    res.fwd_row_teams = 1;
  } else if (res.C == 512 && res.K == 512 && res.threads == 40) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 1;
    res.fwd_row_teams = 1;
  } else if (res.C == 1024 && res.K == 1 && res.threads == 40) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 1;
    res.fwd_row_teams = 1;
  } else if (res.C == 1024 && res.K == 1024 && res.threads == 20) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 5;
    res.fwd_row_teams = 4;
  } else if (res.C == 100 && res.K == 1024 && res.threads == 20) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 5;
    res.fwd_row_teams = 4;
  } else if (res.C == 1024 && res.K == 1024 && res.threads == 24) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 6;
    res.fwd_row_teams = 4;
  } else if (res.C == 100 && res.K == 1024 && res.threads == 24) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 5;
    res.fwd_row_teams = 4;
  } else if (res.C == 512 && res.K == 512 && res.threads == 24) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 5;
    res.fwd_row_teams = 4;
  } else if (res.C == 512 && res.K == 512 && res.threads == 20) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 5;
    res.fwd_row_teams = 4;
  } else if (res.C == 1024 && res.K == 1 && res.threads == 24) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 5;
    res.fwd_row_teams = 4;
  } else if (res.C == 1024 && res.K == 1 && res.threads == 20) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 0;
    res.fwd_col_teams = 6;
    res.fwd_row_teams = 4;
  } else if (res.C == 4096 && res.K == 4096 && res.threads == 8) {
    res.fwd_bf = 8/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 4;
    res.fwd_row_teams = 2;
  } else if (res.C == 1024 && res.K == 1024 && res.threads == 8) {
    res.fwd_bf = 1/*((res.C/res.bc) % 1 == 0) ? 1 : 1*/;
    res.fwd_2d_blocking = 1;
    res.fwd_col_teams = 4;
    res.fwd_row_teams = 2;
  }

  if ((res.C >= 512) && (res.threads == 22)) {
    res.fwd_bf = 4;
    while  ((res.C/res.bc) % res.fwd_bf != 0) {
      res.fwd_bf--;
    }
  }

#if 0
  res.fwd_bf = atoi(getenv("FWD_BF"));
  res.fwd_2d_blocking = atoi(getenv("FWD_2D_BLOCKING"));
  res.fwd_col_teams = atoi(getenv("FWD_COL_TEAMS"));
  res.fwd_row_teams = atoi(getenv("FWD_ROW_TEAMS"));
#endif

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* TPP creation */
  if ( (datatype_in == LIBXSMM_DATATYPE_F32) &&
       (datatype_out == LIBXSMM_DATATYPE_F32) &&
       (datatype_comp == LIBXSMM_DATATYPE_F32) ) {
    libxsmm_datatype dtype = LIBXSMM_DATATYPE_F32;
    libxsmm_blasint stride_a = res.bk*res.bc*sizeof(float);
    libxsmm_blasint stride_b = res.bc*res.bn*sizeof(float);

    l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');

    l_shape                         = libxsmm_create_gemm_shape(res.bk, res.bn, res.bc, lda, ldb, ldc, dtype, dtype, dtype, dtype);
    l_brconfig                      = libxsmm_create_gemm_batch_reduce_config(LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, stride_a, stride_b, 0 /*br_unroll_hint*/);
    res.fwd_compute_kernel_strd = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if ( res.fwd_compute_kernel_strd == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd failed. Bailing...!\n");
      exit(-1);
    }

    if (res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU || res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU ||
        res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK || res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK) {
      memset( &l_argops,  0, sizeof(libxsmm_gemm_ext_unary_argops  ) );
      memset( &l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops) );

      if (res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK || res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK)
        l_argops.cp_unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
      else
        l_argops.cp_unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
      l_argops.cp_unary_type    = LIBXSMM_MELTW_TYPE_UNARY_RELU;
      l_argops.ldcp             = ldc;

      res.fwd_compute_kernel_strd_fused_f32 = libxsmm_dispatch_brgemm_ext_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig,
          l_argops, l_postops );
      if (  res.fwd_compute_kernel_strd_fused_f32 == NULL ) {
        fprintf( stderr, "JIT for BRGEMM TPP fwd_compute_kernel_strd_fused_f32 failed. Bailing...!\n");
        exit(-1);
      }
    }

    l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
    res.fwd_compute_kernel2_strd = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if ( res.fwd_compute_kernel2_strd == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd2 failed. Bailing...!\n");
      exit(-1);
    }

    if (res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS || res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU ||
        res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK) {

      memset( &l_argops,  0, sizeof(libxsmm_gemm_ext_unary_argops  ) );
      memset( &l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops) );

      l_postops.d_in_type      = dtype;
      l_postops.d_binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
      l_postops.d_binary_type  = LIBXSMM_MELTW_TYPE_BINARY_ADD;
      l_postops.ldd            = ldc;

      res.fwd_compute_kernel2_strd_fused_f32 = libxsmm_dispatch_brgemm_ext_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig,
          l_argops, l_postops );
      if (  res.fwd_compute_kernel2_strd_fused_f32 == NULL ) {
        fprintf( stderr, "JIT for BRGEMM TPP fwd_compute_kernel2_strd_fused_f32 failed. Bailing...!\n");
        exit(-1);
      }
    }

    if (res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS || res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU || res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU ||
        res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK || res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK) {

      memset( &l_argops,  0, sizeof(libxsmm_gemm_ext_unary_argops  ) );
      memset( &l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops) );

      l_postops.d_in_type      = dtype;
      l_postops.d_binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
      l_postops.d_binary_type  = LIBXSMM_MELTW_TYPE_BINARY_ADD;
      l_postops.ldd            = ldc;

      if (res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK || res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK)
        l_argops.cp_unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
      else
        l_argops.cp_unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
      l_argops.cp_unary_type    = LIBXSMM_MELTW_TYPE_UNARY_RELU;
      l_argops.ldcp             = ldc;

      res.fwd_compute_kernel3_strd_fused_f32 = libxsmm_dispatch_brgemm_ext_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig,
          l_argops, l_postops );
      if (  res.fwd_compute_kernel3_strd_fused_f32 == NULL ) {
        fprintf( stderr, "JIT for BRGEMM TPP fwd_compute_kernel3_strd_fused_f32 failed. Bailing...!\n");
        exit(-1);
      }
    }

    if (res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU || res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU ||
        res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK || res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK) {
      memset( &l_argops,  0, sizeof(libxsmm_gemm_ext_unary_argops  ) );
      memset( &l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops) );

      if (res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK || res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK)
        l_argops.cp_unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
      else
        l_argops.cp_unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
      l_argops.cp_unary_type    = LIBXSMM_MELTW_TYPE_UNARY_RELU;
      l_argops.ldcp             = ldc;

      res.fwd_compute_kernel4_strd_fused_f32 = libxsmm_dispatch_brgemm_ext_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig,
          l_argops, l_postops );
      if (  res.fwd_compute_kernel4_strd_fused_f32 == NULL ) {
        fprintf( stderr, "JIT for BRGEMM TPP fwd_compute_kernel4_strd_fused_f32 failed. Bailing...!\n");
        exit(-1);
      }
    }

#if 0
    /* Eltwise TPPs  */
    l_unary_shape       = libxsmm_create_meltw_unary_shape(res.bn * res.bk, 1, ld_zero, ld_zero, dtype, dtype, dtype);
    res.fwd_zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    if ( res.fwd_zero_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP fwd_zero_kernel failed. Bailing...!\n");
      exit(-1);
    }

    l_unary_shape         = libxsmm_create_meltw_unary_shape(res.bk, res.bn, ldc, ldc, dtype, dtype, dtype);
    if ( res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK || res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK )
      l_unary_flags       = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
    else
      l_unary_flags       = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    res.fwd_act_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_RELU, l_unary_shape, l_unary_flags);
    if ( res.fwd_act_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP fwd_relu_kernel failed. Bailing...!\n");
      exit(-1);
    }

    l_unary_shape                  = libxsmm_create_meltw_unary_shape(res.bk, res.bn, ldc, ldc, dtype, dtype, dtype);
    l_unary_flags                  = LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL;
    res.fwd_colbcast_load_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, l_unary_shape, l_unary_flags);
    if ( res.fwd_colbcast_load_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP fwd_colbcast_fp32_copy_kernel failed. Bailing...!\n");
      exit(-1);
    }
#endif
    /* init scratch */
    res.scratch_size = 0;
  } else if ( (datatype_in == LIBXSMM_DATATYPE_BF16) &&
              (datatype_out == LIBXSMM_DATATYPE_BF16) &&
              (datatype_comp == LIBXSMM_DATATYPE_F32) ) {
    l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
    l_tr_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
    l_shape = libxsmm_create_gemm_shape( res.bk, res.bn, res.bc,
                                         lda, ldb, ldc,
                                         LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16,
                                         LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );

    res.fwd_tileconfig_kernel = libxsmm_dispatch_gemm_v2( l_shape, l_tc_flags, l_prefetch_flags );
    if ( res.fwd_tileconfig_kernel == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP fwd_config_kernel failed. Bailing...!\n");
      exit(-1);
    }
    res.fwd_tilerelease_kernel = libxsmm_dispatch_gemm_v2( l_shape, l_tr_flags, l_prefetch_flags );
    if ( res.fwd_tilerelease_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP tilerelease_kernel failed. Bailing...!\n");
      exit(-1);
    }

    l_flags = ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
    unroll_hint = (res.C/res.bc)/res.fwd_bf;
    l_shape = libxsmm_create_gemm_shape( res.bk, res.bn, res.bc,
                                         lda, ldb, ldc,
                                         LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16,
                                         LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE,
                                                          res.bk*res.bc*sizeof(libxsmm_bfloat16),
                                                          res.bc*res.bn*sizeof(libxsmm_bfloat16),
                                                          unroll_hint );
    /* strided BRGEMM, FP32 out, Beta = 1 */
    res.fwd_compute_kernel_strd = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if ( res.fwd_compute_kernel_strd == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd failed. Bailing...!\n");
      exit(-1);
    }

    l_flags = l_flags | LIBXSMM_GEMM_FLAG_BETA_0;
    l_shape.out_type = LIBXSMM_DATATYPE_BF16;
    /* strdied BRGEMM, BF16 out, Betat = 0 */
    res.fwd_compute_kernel2_strd = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if ( res.fwd_compute_kernel2_strd == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd3 failed. Bailing...!\n");
      exit(-1);
    }

    if ( (fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_BIAS) == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS ) {
      l_postops.d_in_type      = LIBXSMM_DATATYPE_BF16;
      l_postops.d_binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
      l_postops.d_binary_type  = LIBXSMM_MELTW_TYPE_BINARY_ADD;
      l_postops.ldd            = ldc;
    }
    if ( (fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_RELU) == LIBXSMM_DNN_FC_ELTW_FUSE_RELU ) {
      l_argops.cp_unary_type  = LIBXSMM_MELTW_TYPE_UNARY_RELU;
      l_argops.cp_unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
      l_argops.ldcp           = ldc;
    }
    if ( (fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK) == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK ) {
      l_argops.cp_unary_type  = LIBXSMM_MELTW_TYPE_UNARY_RELU;
      l_argops.cp_unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
      l_argops.ldcp           = ldc;
    }

    res.fwd_compute_kernel5_strd_fused = libxsmm_dispatch_brgemm_ext_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig, l_argops, l_postops );
    if ( res.fwd_compute_kernel5_strd_fused == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd4 failed. Bailing...!\n");
      exit(-1);
    }

    /* Also JIT eltwise TPPs... */
    l_unary_shape = libxsmm_create_meltw_unary_shape( res.bk, res.bn, ldc, ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32 );
    res.fwd_store_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( res.fwd_store_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP fwd_store_kernel failed. Bailing...!\n");
      exit(-1);
    }
    if ( (fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK) == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK )
      l_unary_flags       = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
    else
      l_unary_flags       = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    res.fwd_act_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_RELU, l_unary_shape, l_unary_flags );
    if ( res.fwd_act_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP fwd_act_kernel failed. Bailing...!\n");
      exit(-1);
    }

    l_unary_shape.in0_type = LIBXSMM_DATATYPE_BF16;
    l_unary_shape.out_type = LIBXSMM_DATATYPE_F32;
    res.fwd_colbcast_load_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL );
    if ( res.fwd_colbcast_load_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP fwd_colbcast_load_kernel failed. Bailing...!\n");
      exit(-1);
    }

    l_unary_shape = libxsmm_create_meltw_unary_shape( res.bk*res.bn, 1, ld_zero, ld_zero, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    res.fwd_zero_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( res.fwd_zero_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP fwd_zero_kernel failed. Bailing...!\n");
      exit(-1);
    }

    /* init scratch */
    res.scratch_size = sizeof(float) *  LIBXSMM_MAX(res.K * res.N, res.threads * LIBXSMM_MAX(res.bk * res.bn, res.K));
  } else {
    fprintf( stderr, "Unsupported precision for fwd pass Bailing...!\n");
    exit(-1);
  }

  return res;
}

LIBXSMM_API libxsmm_dnn_fc_bwd_config setup_libxsmm_dnn_fc_bwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint K, libxsmm_blasint bn,
    libxsmm_blasint bc, libxsmm_blasint bk, libxsmm_blasint threads, libxsmm_dnn_fc_eltw_fuse fuse_type,
    libxsmm_datatype datatype_in, libxsmm_datatype datatype_out, libxsmm_datatype datatype_comp ) {
  libxsmm_dnn_fc_bwd_config res;
  libxsmm_blasint lda = bc;
  libxsmm_blasint ldb = bk;
  libxsmm_blasint ldc = bc;
  libxsmm_blasint ld_zero_bwd = bc*bn;
  libxsmm_blasint ld_zero_upd = bk;
  libxsmm_blasint ld_relu_bwd = bk;
  libxsmm_blasint delbias_K = K;
  libxsmm_blasint delbias_N = N;
  libxsmm_blasint updM;
  libxsmm_blasint updN;
  libxsmm_blasint bbk;
  libxsmm_blasint bbc;
  /* @TODO egeor, kvoronin: can you please double check */
  libxsmm_blasint ldaT = bk;
  libxsmm_blasint ldb_orig= bc;

  libxsmm_meltw_unary_shape  l_unary_shape;
  libxsmm_bitfield  l_unary_flags;

  libxsmm_gemm_shape l_shape;
  libxsmm_gemm_batch_reduce_config l_brconfig;

  libxsmm_bitfield l_flags, l_tc_flags, l_tr_flags;
  libxsmm_bitfield l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  libxsmm_blasint unroll_hint;
  size_t size_bwd_scratch;
  size_t size_upd_scratch;

  /* setting up some handle values */
  res.N = N;
  res.C = C;
  res.K = K;
  res.bn = bn;
  res.bc = bc;
  res.bk = bk;
  res.threads = threads;
  res.fuse_type = fuse_type;

  /* setup parallelization strategy */
  res.bwd_M_hyperpartitions = 1; /* TODO: enable hyperpartitions */
  res.upd_M_hyperpartitions = 1;
  res.bwd_N_hyperpartitions = 1;
  res.upd_N_hyperpartitions = 1;
  res.bwd_bf = 1;
  res.bwd_2d_blocking = 0;
  res.bwd_col_teams = 1;
  res.bwd_row_teams = 1;
  res.upd_bf = 1;
  res.upd_2d_blocking = 0;
  res.upd_col_teams = 1;
  res.upd_row_teams = 1;
  res.ifm_subtasks = 1;
  res.ofm_subtasks = 1;

  if (threads == 16) {
    res.bwd_bf = 1;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 2;
    res.bwd_row_teams = 8;
    res.upd_bf = 1;
    res.upd_2d_blocking = 1;
    res.upd_col_teams = 2;
    res.upd_row_teams = 8;
    res.ifm_subtasks = 1;
    res.ofm_subtasks = 1;
  } else if (threads == 14) {
    res.bwd_bf = 1;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 2;
    res.bwd_row_teams = 7;
    res.upd_bf = 1;
    res.upd_2d_blocking = 1;
    res.upd_col_teams = 2;
    res.upd_row_teams = 7;
    res.ifm_subtasks = 1;
    res.ofm_subtasks = 1;
  } else if (threads == 56) {
    res.bwd_bf = 1;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 1;
    res.bwd_row_teams = 14;
    res.bwd_M_hyperpartitions = 1;
    res.bwd_N_hyperpartitions = 4;
    res.upd_bf = 1;
    res.upd_2d_blocking = 1;
    res.upd_col_teams = 1;
    res.upd_row_teams = 14;
    res.upd_M_hyperpartitions = 1;
    res.upd_N_hyperpartitions = 4;
    res.ifm_subtasks = 1;
    res.ofm_subtasks = 1;
  } else if (res.threads == 2) {
    res.bwd_bf = 1;
    res.upd_bf = 1;
    res.bwd_2d_blocking = 1;
    res.upd_2d_blocking = 0;
    res.bwd_col_teams = 2;
    res.bwd_row_teams = 1;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = 1;
    res.ofm_subtasks = 1;
  } else if (res.threads == 4) {
    res.bwd_bf = 1;
    res.upd_bf = 1;
    res.bwd_2d_blocking = 1;
    res.upd_2d_blocking = 0;
    res.bwd_col_teams = 2;
    res.bwd_row_teams = 2;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = 1;
    res.ofm_subtasks = 1;
  } else if (res.threads == 8) {
    res.bwd_bf = 1;
    res.upd_bf = 1;
    res.bwd_2d_blocking = 1;
    res.upd_2d_blocking = 0;
    res.bwd_col_teams = 2;
    res.bwd_row_teams = 4;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = 1;
    res.ofm_subtasks = 1;
  } else if (threads == 1) {
    res.bwd_bf = 1;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 1;
    res.bwd_row_teams = 1;
    res.bwd_M_hyperpartitions = 1;
    res.bwd_N_hyperpartitions = 1;
    res.upd_bf = 1;
    res.upd_2d_blocking = 1;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.upd_M_hyperpartitions = 1;
    res.upd_N_hyperpartitions = 1;
    res.ifm_subtasks = 1;
    res.ofm_subtasks = 1;
  } else {
    res.bwd_bf = 1;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 1;
    res.bwd_row_teams = 1;
    res.upd_bf = 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = 1;
    res.ofm_subtasks = 1;
  }

  if (res.C == 1024 && res.K == 1024 && res.threads == 22) {
    res.bwd_bf = ((res.K/res.bk) % 8 == 0) ? 8 : 1;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 11;
    res.bwd_row_teams = 2;
    res.upd_bf = ((res.N/res.bn) % 14 == 0) ? 14 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 7;
    res.upd_row_teams = 4;
    res.ifm_subtasks = 1/*((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1*/;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  } else if (res.C == 512 && res.K == 512 && res.threads == 22) {
    res.bwd_bf = ((res.K/res.bk) % 4 == 0) ? 4 : 1;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 11;
    res.bwd_row_teams = 2;
    res.upd_bf = ((res.N/res.bn) % 14 == 0) ? 14 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = 1/*((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1*/;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  } else if (res.C == 1024 && res.K == 1024 && res.threads == 64) {
    res.bwd_bf = ((res.K/res.bk) % 8 == 0) ? 8 : 1;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 8;
    res.bwd_row_teams = 8;
    res.upd_bf = ((res.N/res.bn) % 14 == 0) ? 14 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 7;
    res.upd_row_teams = 4;
    res.ifm_subtasks = ((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  } else if (res.C == 512 && res.K == 512 && res.threads == 64) {
    res.bwd_bf = ((res.K/res.bk) % 4 == 0) ? 4 : 1;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 8;
    res.bwd_row_teams = 8;
    res.upd_bf = ((res.N/res.bn) % 14 == 0) ? 14 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = ((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  } else if (res.C == 100 && res.K == 1024 && res.threads == 28) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 1;
    res.bwd_row_teams = 1;
    res.upd_bf = ((res.N/res.bn) % 14 == 0) ? 14 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = 1/*((res.bc % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  } else if (res.C == 1024 && res.K == 1024 && res.threads == 28) {
    res.bwd_bf = ((res.K/res.bk) % 8 == 0) ? 8 : 1;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 7;
    res.bwd_row_teams = 4;
    res.upd_bf = ((res.N/res.bn) % 14 == 0) ? 14 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 7;
    res.upd_row_teams = 4;
    res.ifm_subtasks = ((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  } else if (res.C == 512 && res.K == 512 && res.threads == 28) {
    res.bwd_bf = ((res.K/res.bk) % 4 == 0) ? 4 : 1;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 1;
    res.bwd_row_teams = 1;
    res.upd_bf = ((res.N/res.bn) % 14 == 0) ? 14 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = ((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  } else if (res.C == 1024 && res.K == 1 && res.threads == 28) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 14;
    res.bwd_row_teams = 2;
    res.upd_bf = ((res.N/res.bn) % 2 == 0) ? 2 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = ((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  } else if (res.C == 100 && res.K == 1024 && res.threads == 40) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 1;
    res.bwd_row_teams = 1;
    res.upd_bf = ((res.N/res.bn) % 10 == 0) ? 10 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = 1/*((res.bc % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  } else if (res.C == 1024 && res.K == 1024 && res.threads == 40) {
    res.bwd_bf = ((res.K/res.bk) % 8 == 0) ? 8 : 1;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 7;
    res.bwd_row_teams = 4;
    res.upd_bf = ((res.N/res.bn) % 10 == 0) ? 10 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 7;
    res.upd_row_teams = 4;
    res.ifm_subtasks = 1/*((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1*/;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  } else if (res.C == 512 && res.K == 512 && res.threads == 40) {
    res.bwd_bf = ((res.K/res.bk) % 4 == 0) ? 4 : 1;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 1;
    res.bwd_row_teams = 1;
    res.upd_bf = ((res.N/res.bn) % 10 == 0) ? 10 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = 1/*((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1*/;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  } else if (res.C == 1024 && res.K == 1 && res.threads == 40) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 10;
    res.bwd_row_teams = 4;
    res.upd_bf = ((res.N/res.bn) % 2 == 0) ? 2 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = 1/*((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1*/;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  } else if (res.C == 1024 && res.K == 1024 && res.threads == 20) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 5;
    res.bwd_row_teams = 4;
    res.upd_bf = ((res.N/res.bn) % 15 == 0) ? 15 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 5;
    res.upd_row_teams = 4;
    res.ifm_subtasks = 1/*((res.bc % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  } else if (res.C == 100 && res.K == 1024 && res.threads == 20) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 1;
    res.bwd_row_teams = 1;
    res.upd_bf = ((res.N/res.bn) % 9 == 0) ? 9 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = 1/*((res.bc % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
    res.ofm_subtasks = ((res.bk % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1;
  } else if (res.C == 1024 && res.K == 1024 && res.threads == 24) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 6;
    res.bwd_row_teams = 4;
    res.upd_bf = ((res.N/res.bn) % 15 == 0) ? 15 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 6;
    res.upd_row_teams = 4;
    res.ifm_subtasks = ((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  } else if (res.C == 100 && res.K == 1024 && res.threads == 24) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 12;
    res.bwd_row_teams = 2;
    res.upd_bf = ((res.N/res.bn) % 15 == 0) ? 15 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 5;
    res.upd_row_teams = 4;
    res.ifm_subtasks = 1/*((res.bc % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  } else if (res.C == 512 && res.K == 512 && res.threads == 24) {
    res.bwd_bf = ((res.K/res.bk) % 4 == 0) ? 4 : 1;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 5;
    res.bwd_row_teams = 4;
    res.upd_bf = ((res.N/res.bn) % 15 == 0) ? 15 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 5;
    res.upd_row_teams = 4;
    res.ifm_subtasks = ((res.bc % 2 == 0) && (res.upd_2d_blocking == 0)) ? 2 : 1;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  } else if (res.C == 512 && res.K == 512 && res.threads == 20) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 1;
    res.bwd_row_teams = 1;
    res.upd_bf = ((res.N/res.bn) % 15 == 0) ? 15 : 1;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 1;
    res.upd_row_teams = 1;
    res.ifm_subtasks = ((res.bc % 4 == 0) && (res.upd_2d_blocking == 0)) ? 4 : 1;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  } else if (res.C == 1024 && res.K == 1 && res.threads == 24) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 0;
    res.bwd_col_teams = 5;
    res.bwd_row_teams = 4;
    res.upd_bf = 1/*((res.N/res.bn) % 1 == 0) ? 1 : 1*/;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 5;
    res.upd_row_teams = 4;
    res.ifm_subtasks = ((res.bc % 4 == 0) && (res.upd_2d_blocking == 0)) ? 4 : 1;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  } else if (res.C == 1024 && res.K == 1 && res.threads == 20) {
    res.bwd_bf = 1/*((res.K/res.bk) % 1 == 0) ? 1 : 1*/;
    res.bwd_2d_blocking = 1;
    res.bwd_col_teams = 5;
    res.bwd_row_teams = 4;
    res.upd_bf = 1/*((res.N/res.bn) % 1 == 0) ? 1 : 1*/;
    res.upd_2d_blocking = 0;
    res.upd_col_teams = 6;
    res.upd_row_teams = 4;
    res.ifm_subtasks = 1/*((res.bc % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
    res.ofm_subtasks = 1/*((res.bk % 1 == 0) && (res.upd_2d_blocking == 0)) ? 1 : 1*/;
  }

  if ((res.N >= 896) && (res.C >= 2048) && (res.K >= 2048)) {
    res.upd_bf = 28;
    while  ((res.N/res.bn) % res.upd_bf != 0) {
      res.upd_bf--;
    }
  }

  if ((res.K >= 512) && (res.threads == 22)) {
    res.bwd_bf = 4;
    while  ((res.K/res.bk) % res.bwd_bf != 0) {
      res.bwd_bf--;
    }
  }

  if ((res.N >= 512) && (res.threads == 22) && (res.ifm_subtasks == 1) && (res.ofm_subtasks == 1)) {
    res.upd_bf = 8;
    while  ((res.N/res.bn) % res.upd_bf != 0) {
      res.upd_bf--;
    }
  }

  bbk = (res.upd_2d_blocking == 1) ? bk : bk/res.ofm_subtasks;
  bbc = (res.upd_2d_blocking == 1) ? bc : bc/res.ifm_subtasks;

#if 0
  res.bwd_bf = atoi(getenv("BWD_BF"));
  res.bwd_2d_blocking = atoi(getenv("BWD_2D_BLOCKING"));
  res.bwd_col_teams = atoi(getenv("BWD_COL_TEAMS"));
  res.bwd_row_teams = atoi(getenv("BWD_ROW_TEAMS"));
  res.upd_bf = atoi(getenv("UPD_BF"));
  res.upd_2d_blocking = atoi(getenv("UPD_2D_BLOCKING"));
  res.upd_col_teams = atoi(getenv("UPD_COL_TEAMS"));
  res.upd_row_teams = atoi(getenv("UPD_ROW_TEAMS"));
  res.ifm_subtasks = atoi(getenv("IFM_SUBTASKS"));
  res.ofm_subtasks = atoi(getenv("OFM_SUBTASKS"));
#endif

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* TPP creation */
  if ( (datatype_in == LIBXSMM_DATATYPE_F32) &&
       (datatype_out == LIBXSMM_DATATYPE_F32) &&
       (datatype_comp == LIBXSMM_DATATYPE_F32) ) {
    libxsmm_datatype dtype = LIBXSMM_DATATYPE_F32;
    libxsmm_blasint stride_a = res.bk*res.bc*sizeof(float);
    libxsmm_blasint stride_b = res.bk*res.bn*sizeof(float);
    /* @TODO: egeor, kvorinon: setting FP32 back to its old value */
    l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');

    l_shape                         = libxsmm_create_gemm_shape(res.bc, res.bn, res.bk, lda, ldb, ldc, dtype, dtype, dtype, dtype);
    l_brconfig                      = libxsmm_create_gemm_batch_reduce_config(LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, stride_a, stride_b, 0 /*br_unroll_hint*/);
    res.bwd_compute_kernel_strd = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if ( res.bwd_compute_kernel_strd == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP gemm_bwd failed. Bailing...!\n");
      exit(-1);
    }

    l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
    res.bwd_compute_kernel2_strd = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if ( res.bwd_compute_kernel2_strd == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP gemm_bwd2 failed. Bailing...!\n");
      exit(-1);
    }

    l_unary_shape            = libxsmm_create_meltw_unary_shape(res.bk, res.bc, ldaT, lda, dtype, dtype, dtype);
    l_unary_flags            = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    res.norm_to_normT_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, l_unary_shape, l_unary_flags);
    if ( res.norm_to_normT_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP norm_to_normT_kernel failed. Bailing...!\n");
      exit(-1);
    }

    if (res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU || res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU ||
        res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK || res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK) {
      l_unary_shape         = libxsmm_create_meltw_unary_shape(res.bk, res.bn, ld_relu_bwd, ld_relu_bwd, dtype, dtype, dtype);
      if ( res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK || res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK )
        l_unary_flags       = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
      else
        l_unary_flags       = LIBXSMM_MELTW_FLAG_UNARY_NONE;
      res.bwd_relu_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_RELU_INV, l_unary_shape, l_unary_flags);
      if ( res.bwd_relu_kernel == NULL ) {
        fprintf( stderr, "JIT for TPP bwd_relu_kernel failed. Bailing...!\n");
        exit(-1);
      }
    }

    l_unary_shape         = libxsmm_create_meltw_unary_shape(res.bn * res.bc, 1, ld_zero_bwd, ld_zero_bwd, dtype, dtype, dtype);
    l_unary_flags         = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    res.bwd_zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, l_unary_flags);
    if ( res.bwd_zero_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP bwd_zero_kernel failed. Bailing...!\n");
      exit(-1);
    }

    /* UPD GEMM */
    lda = res.bk;
    ldb = res.bc;
    ldc = res.bk;
    updM = res.bk/res.ofm_subtasks;
    updN = res.bc/res.ifm_subtasks;

    l_flags = LIBXSMM_GEMM_FLAGS('N', 'T');

    stride_a = res.K*res.bn*sizeof(float);
    stride_b = res.C*res.bn*sizeof(float);

    l_shape                         = libxsmm_create_gemm_shape(updM, updN, res.bn, lda, ldb, ldc, dtype, dtype, dtype, dtype);
    l_brconfig                      = libxsmm_create_gemm_batch_reduce_config(LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, stride_a, stride_b, 0 /*br_unroll_hint*/);
    res.upd_compute_kernel_strd = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if ( res.upd_compute_kernel_strd == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP gemm_upd failed. Bailing...!\n");
      exit(-1);
    }

    l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;

    res.upd_compute_kernel2_strd = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if ( res.upd_compute_kernel2_strd == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP gemm_upd2 failed. Bailing...!\n");
      exit(-1);
    }

    /* JIT TPP kernels */
    l_unary_shape         = libxsmm_create_meltw_unary_shape(res.bk, res.bc, ld_zero_upd, ld_zero_upd, dtype, dtype, dtype);
    l_unary_flags         = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    res.upd_zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, l_unary_flags);
    if ( res.upd_zero_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP upd_zero_kernel failed. Bailing...!\n");
      exit(-1);
    }

    if (res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS || res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU ||
        res.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK) {
      l_unary_shape               = libxsmm_create_meltw_unary_shape(res.bk, res.bn, delbias_K, delbias_N, dtype, dtype, dtype);
      l_unary_flags               = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
      res.delbias_reduce_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD_NCNC_FORMAT, l_unary_shape, l_unary_flags);
      if ( res.delbias_reduce_kernel == NULL ) {
        fprintf( stderr, "JIT for TPP delbias_reduce_kernel failed. Bailing...!\n");
        exit(-1);
      }
    }

    /* init scratch */
    res.scratch_size =  sizeof(float) * ( (((size_t)res.C + (size_t)res.K) * (size_t)res.N) + ((size_t)res.C * (size_t)res.K) );
  } else if ( (datatype_in == LIBXSMM_DATATYPE_BF16) &&
              (datatype_out == LIBXSMM_DATATYPE_BF16) &&
              (datatype_comp == LIBXSMM_DATATYPE_F32) ) {
    if (res.bwd_2d_blocking != 1) {
      printf("Requested private wt transposes, but for the current # of threads the bwd decomposition is not 2D. Will perform upfront/shared wt transposes...\n");
    }
    if (res.upd_2d_blocking != 1) {
      printf("Requested private act transposes, but for the current # of threads the upd decomposition is not 2D. Will perform upfront/shared act transposes...\n");
    }
    if (res.upd_2d_blocking != 1) {
      printf("Requested private dact transposes, but for the current # of threads the upd decomposition is not 2D. Will perform upfront/shared dact transposes...\n");
    }

    l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
    l_tr_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
    l_shape = libxsmm_create_gemm_shape( res.bc, res.bn, res.bk,
                                         lda, ldb, ldc,
                                         LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16,
                                         LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    res.bwd_tileconfig_kernel = libxsmm_dispatch_gemm_v2( l_shape, l_tc_flags, l_prefetch_flags );
    if ( res.bwd_tileconfig_kernel == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP bwd_config_kernel failed. Bailing...!\n");
      exit(-1);
    }
    res.bwd_tilerelease_kernel = libxsmm_dispatch_gemm_v2( l_shape, l_tr_flags, l_prefetch_flags );
    if ( res.bwd_tilerelease_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP tilerelease_kernel failed. Bailing...!\n");
      exit(-1);
    }

    l_flags = ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
    unroll_hint = (res.K/res.bk)/res.bwd_bf;
    l_shape = libxsmm_create_gemm_shape( res.bc, res.bn, res.bk,
                                         lda, ldb, ldc,
                                         LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16,
                                         LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE,
                                                          res.bk*res.bc*sizeof(libxsmm_bfloat16),
                                                          res.bk*res.bn*sizeof(libxsmm_bfloat16),
                                                          unroll_hint );
    res.bwd_compute_kernel_strd = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if ( res.bwd_compute_kernel_strd == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP gemm_bwd failed. Bailing...!\n");
      exit(-1);
    }

    l_flags = l_flags | LIBXSMM_GEMM_FLAG_BETA_0;
    l_shape.out_type = LIBXSMM_DATATYPE_BF16;
    /* strdied BRGEMM, BF16 out, Betat = 0 */
    res.bwd_compute_kernel2_strd = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if ( res.bwd_compute_kernel2_strd == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP gemm_bwd3 failed. Bailing...!\n");
      exit(-1);
    }

    l_unary_shape = libxsmm_create_meltw_unary_shape( res.bc, res.bn, ldc, ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32 );
    /* Also JIT eltwise TPPs... */
    res.bwd_store_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( res.bwd_store_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP bwd_cvtfp32bf16_kernel failed. Bailing...!\n");
      exit(-1);
    }

    l_unary_shape = libxsmm_create_meltw_unary_shape( res.bc, res.bn, ldc, ldc, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32 );
    res.bwd_relu_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_RELU_INV, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT );
    if ( res.bwd_relu_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP bwd_relu_kernel failed. Bailing...!\n");
      exit(-1);
    }

    l_unary_shape = libxsmm_create_meltw_unary_shape( bk, bn, delbias_K, delbias_N, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32 );
    res.delbias_reduce_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD_NCNC_FORMAT, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS );
    if( res.delbias_reduce_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP delbias_reduce_kernel failed. Bailing...!\n");
      exit(-1);
    }

    l_unary_shape = libxsmm_create_meltw_unary_shape( res.bc*res.bn, 1, ld_zero_bwd, ld_zero_bwd, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    res.bwd_zero_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( res.bwd_zero_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP bwd_zero_kernel failed. Bailing...!\n");
      exit(-1);
    }

    /* JITing the tranpose kernel */
    l_unary_shape = libxsmm_create_meltw_unary_shape( bk, bc, ldaT, lda, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    res.vnni_to_vnniT_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI_TO_VNNIT, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( res.vnni_to_vnniT_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP vnni_to_vnniT_kernel failed. Bailing...!\n");
      exit(-1);
    }

    /* UPD GEMM */
    lda = res.bk;
    ldb = res.bn;
    ldc = res.bk;
    updM = res.bk/res.ofm_subtasks;
    updN = res.bc/res.ifm_subtasks;

    l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
    l_tr_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
    l_shape = libxsmm_create_gemm_shape( updM, updN, res.bn,
                                         lda, ldb, ldc,
                                         LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16,
                                         LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    res.upd_tileconfig_kernel = libxsmm_dispatch_gemm_v2( l_shape, l_tc_flags, l_prefetch_flags );
    if ( res.upd_tileconfig_kernel == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP fwd_config_kernel failed. Bailing...!\n");
      exit(-1);
    }
    res.upd_tilerelease_kernel = libxsmm_dispatch_gemm_v2( l_shape, l_tr_flags, l_prefetch_flags );
    if ( res.upd_tilerelease_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP tilerelease_kernel failed. Bailing...!\n");
      exit(-1);
    }

    l_flags = ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
    unroll_hint = (res.N/res.bn)/res.upd_bf;
    l_shape = libxsmm_create_gemm_shape( updM, updN, res.bn,
                                         lda, ldb, ldc,
                                         LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16,
                                         LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE,
                                                          res.bk*res.bn*sizeof(libxsmm_bfloat16),
                                                          res.bc*res.bn*sizeof(libxsmm_bfloat16),
                                                          unroll_hint );
    /* strided BRGEMM, FP32 out, Beta = 1 */
    res.upd_compute_kernel_strd = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if ( res.upd_compute_kernel_strd == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP gemm_upd failed. Bailing...!\n");
      exit(-1);
    }

    l_shape.out_type = LIBXSMM_DATATYPE_BF16;
    l_flags = l_flags | LIBXSMM_GEMM_FLAG_BETA_0;
    l_flags = l_flags | LIBXSMM_GEMM_FLAG_VNNI_C;
    /* strdied BRGEMM, FP16 out, VNNI Layout, Beta= 0 */
    res.upd_compute_kernel2_strd = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    if ( res.upd_compute_kernel2_strd == NULL ) {
      fprintf( stderr, "JIT for BRGEMM TPP gemm_upd3 failed. Bailing...!\n");
      exit(-1);
    }

    /* Also JIT eltwise TPPs... */
    l_unary_shape = libxsmm_create_meltw_unary_shape( bbk, bbc, ldc, ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32 );
    res.upd_store_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( res.upd_store_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP upd_cvtfp32bf16_kernel failed. Bailing...!\n");
      exit(-1);
    }

    l_unary_shape = libxsmm_create_meltw_unary_shape( bbk, bbc, ld_zero_upd, ld_zero_upd, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
    res.upd_zero_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( res.upd_zero_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP upd_zero_kernel failed. Bailing...!\n");
      exit(-1);
    }

    /* JITing the tranpose kernels */
    l_unary_shape = libxsmm_create_meltw_unary_shape( bk, bn, lda, lda, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    res.norm_to_vnni_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( res.norm_to_vnni_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP norm_to_vnni_kernel failed. Bailing...!\n");
      exit(-1);
    }

    l_unary_shape = libxsmm_create_meltw_unary_shape( bbk, bbc, ldc, ldc, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    res.norm_to_vnni_kernel_wt = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( res.norm_to_vnni_kernel_wt == NULL ) {
      fprintf( stderr, "JIT for TPP norm_to_vnni_kernel failed. Bailing...!\n");
      exit(-1);
    }

    l_unary_shape = libxsmm_create_meltw_unary_shape( bc, bn, ldb_orig, ldb, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    res.norm_to_normT_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( res.norm_to_normT_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP norm_to_normT_kernel failed. Bailing...!\n");
      exit(-1);
    }

    /* init scratch */
    size_bwd_scratch = sizeof(float) * LIBXSMM_MAX(res.C * res.N, res.threads * res.bc * res.bn) + sizeof(libxsmm_bfloat16) * res.C * res.K;
    res.bwd_private_tr_wt_scratch_mark = size_bwd_scratch;
    size_bwd_scratch += res.threads * res.bc * res.K * sizeof(libxsmm_bfloat16);
    size_upd_scratch = sizeof(float) * LIBXSMM_MAX(res.C * res.K, res.threads * res.bc * res.bk) + sizeof(libxsmm_bfloat16) * res.threads * res.bk * res.bc + sizeof(libxsmm_bfloat16) * (res.N * (res.C + res.K));
    res.scratch_size = LIBXSMM_MAX(size_bwd_scratch, size_upd_scratch) + 2 * sizeof(libxsmm_bfloat16) * res.N * res.K;
    res.doutput_scratch_mark = LIBXSMM_MAX(size_bwd_scratch, size_upd_scratch) ;
    res.upd_private_tr_dact_scratch_mark = res.scratch_size;
    res.scratch_size += res.threads * res.bk * res.N * sizeof(libxsmm_bfloat16);
    res.upd_private_tr_act_scratch_mark = res.scratch_size;
    res.scratch_size += res.threads * res.bc * res.N * (((res.C/res.bc)+res.upd_col_teams-1)/res.upd_col_teams) * sizeof(libxsmm_bfloat16);
   } else {
    fprintf( stderr, "Unsupported precision for bwdupd pass Bailing...!\n");
    exit(-1);
  }

  return res;
}

LIBXSMM_API void libxsmm_dnn_fc_fwd_exec_f32( libxsmm_dnn_fc_fwd_config cfg, const float* wt_ptr, const float* in_act_ptr, float* out_act_ptr,
    const float* bias_ptr, unsigned char* relu_ptr, int start_tid, int my_tid, void* scratch ) {
  const libxsmm_blasint nBlocksIFm = cfg.C / cfg.bc;
  const libxsmm_blasint nBlocksOFm = cfg.K / cfg.bk;
  const libxsmm_blasint nBlocksMB  = cfg.N / cfg.bn;

  libxsmm_gemm_param gemm_param;

  gemm_param.a.secondary = NULL;
  gemm_param.b.secondary = NULL;

  libxsmm_gemm_ext_param gemm_param_ext;
  gemm_param_ext.a.secondary = NULL;
  gemm_param_ext.b.secondary = NULL;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;
  /* number of tasks that could be run in parallel */
  const libxsmm_blasint work = nBlocksOFm * nBlocksMB;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (work % cfg.threads == 0) ?
    (work / cfg.threads) : ((work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

  /* loop variables */
  libxsmm_blasint mb1ofm1 = 0, mb1 = 0, ofm1 = 0, ifm1 = 0;
  libxsmm_blasint N_tasks_per_thread = 0, M_tasks_per_thread = 0;
  libxsmm_blasint my_M_start = 0, my_M_end = 0, my_N_start = 0, my_N_end = 0;
  libxsmm_blasint my_col_id = 0, my_row_id = 0, col_teams = 0, row_teams = 0;

  LIBXSMM_VLA_DECL(4, float,           output, out_act_ptr, nBlocksOFm, cfg.bn, cfg.bk);
  LIBXSMM_VLA_DECL(4, const float,      input,  in_act_ptr, nBlocksIFm, cfg.bn, cfg.bc);
  LIBXSMM_VLA_DECL(4, const float,     filter,      wt_ptr, nBlocksIFm, cfg.bc, cfg.bk);
  LIBXSMM_VLA_DECL(2, const float,       bias,    bias_ptr,                     cfg.bk);
  LIBXSMM_VLA_DECL(4, unsigned char, relubitmask,    relu_ptr, nBlocksOFm, cfg.bn, cfg.bk/8);

  unsigned long long  blocks = nBlocksIFm;
  libxsmm_blasint CB_BLOCKS = nBlocksIFm, BF = 1;
  LIBXSMM_UNUSED( scratch );

  BF = cfg.fwd_bf;
  CB_BLOCKS = nBlocksIFm/BF;
  blocks = CB_BLOCKS;

  col_teams = cfg.fwd_col_teams;
  row_teams = cfg.fwd_row_teams;
  my_row_id = ltid % row_teams;
  my_col_id = ltid / row_teams;
  N_tasks_per_thread = LIBXSMM_UPDIV(nBlocksMB, col_teams);
  M_tasks_per_thread = LIBXSMM_UPDIV(nBlocksOFm, row_teams);
  my_N_start = LIBXSMM_MIN(my_col_id * N_tasks_per_thread, nBlocksMB);
  my_N_end = LIBXSMM_MIN((my_col_id+1) * N_tasks_per_thread, nBlocksMB);
  my_M_start = LIBXSMM_MIN(my_row_id * M_tasks_per_thread, nBlocksOFm);
  my_M_end = LIBXSMM_MIN((my_row_id+1) * M_tasks_per_thread, nBlocksOFm);

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  if (cfg.fwd_2d_blocking == 1) {
    if (BF > 1) {
      for (ifm1 = 0; ifm1 < BF; ++ifm1) {
        for (ofm1 = my_M_start; ofm1 < my_M_end; ++ofm1) {
          for (mb1 = my_N_start; mb1 < my_N_end; ++mb1) {
            if ( ifm1 == 0 ) {
              if ( cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS || cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU || cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK ) {
                  gemm_param_ext.op.tertiary = &blocks;
                  gemm_param_ext.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
                  gemm_param_ext.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
                  gemm_param_ext.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                  gemm_param_ext.d.primary = (void*)&LIBXSMM_VLA_ACCESS(2, bias, ofm1, 0, cfg.bk);
                  cfg.fwd_compute_kernel2_strd_fused_f32( &gemm_param_ext ); /* beta = 0.0 + bias */
              } else {
                gemm_param.op.tertiary = &blocks;
                gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
                gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
                gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                cfg.fwd_compute_kernel2_strd( &gemm_param ); /* beta = 0.0 */
              }
            } else { /* not ifm1 = 0 */
              if ( ( ifm1 == BF-1 ) &&
                   (cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU ||
                      cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK)
                 ) {
                gemm_param_ext.op.tertiary = &blocks;
                gemm_param_ext.a.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
                gemm_param_ext.b.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, input,   mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
                gemm_param_ext.c.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, output,      mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                gemm_param_ext.c.secondary = (void*)&LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/8);
                cfg.fwd_compute_kernel_strd_fused_f32( &gemm_param_ext ); /* beta = 1.0 + relu */
              } else {
                gemm_param.op.tertiary = &blocks;
                gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
                gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
                gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                cfg.fwd_compute_kernel_strd( &gemm_param ); /* beta = 1.0 */
              }
            }
          }
        }
      }
    } else {
      for (ofm1 = my_M_start; ofm1 < my_M_end; ++ofm1) {
        for (mb1 = my_N_start; mb1 < my_N_end; ++mb1) {
          if ( cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS ) { /* bias only */
            gemm_param_ext.op.tertiary = &blocks;
            gemm_param_ext.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
            gemm_param_ext.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
            gemm_param_ext.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
            gemm_param_ext.d.primary = (void*)&LIBXSMM_VLA_ACCESS(2, bias, ofm1, 0, cfg.bk);
            cfg.fwd_compute_kernel2_strd_fused_f32( &gemm_param_ext ); /* beta = 0.0 + bias */
          } else if ( cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU || cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK ) { /* bias + relu */
            gemm_param_ext.op.tertiary = &blocks;
            gemm_param_ext.a.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
            gemm_param_ext.b.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1,  0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
            gemm_param_ext.d.primary   = (void*)&LIBXSMM_VLA_ACCESS(2, bias,   ofm1, 0, cfg.bk);
            gemm_param_ext.c.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, output,      mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
            gemm_param_ext.c.secondary = (void*)&LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/8);
            cfg.fwd_compute_kernel3_strd_fused_f32( &gemm_param_ext ); /* beta = 0.0 + bias + relu */
          } else if ( cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK ) { /* only relu */
            gemm_param_ext.op.tertiary = &blocks;
            gemm_param_ext.a.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
            gemm_param_ext.b.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1,  0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
            gemm_param_ext.c.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, output,      mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
            gemm_param_ext.c.secondary = (void*)&LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/8);
            cfg.fwd_compute_kernel4_strd_fused_f32( &gemm_param_ext ); /* beta = 0.0 + relu */
          } else { /* no fusion */
            gemm_param.op.tertiary = &blocks;
            gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
            gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
            gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
            cfg.fwd_compute_kernel2_strd( &gemm_param ); /* beta = 0.0 */
          }
        }
      }
    }
  } else {
    if (BF > 1) {
      for ( ifm1 = 0; ifm1 < BF; ++ifm1 ) {
        for ( mb1ofm1 = thr_begin; mb1ofm1 < thr_end; ++mb1ofm1 ) {
          mb1  = mb1ofm1%nBlocksMB;
          ofm1 = mb1ofm1/nBlocksMB;
          if ( ifm1 == 0 ) {
            if ( cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS || cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU || cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK ) {
              gemm_param_ext.op.tertiary = &blocks;
              gemm_param_ext.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
              gemm_param_ext.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
              gemm_param_ext.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              gemm_param_ext.d.primary = (void*)&LIBXSMM_VLA_ACCESS(2, bias, ofm1, 0, cfg.bk);
              cfg.fwd_compute_kernel2_strd_fused_f32( &gemm_param_ext ); /* beta = 0.0 + bias */
            } else {
              gemm_param.op.tertiary = &blocks;
              gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
              gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
              gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              cfg.fwd_compute_kernel2_strd( &gemm_param ); /* beta = 0.0 */
            }
          } else { /* not ifm1 = 0 */
            if ( ( ifm1 == BF-1 ) &&
                 (cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU ||
                    cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK)
               ) {
              gemm_param_ext.op.tertiary = &blocks;
              gemm_param_ext.a.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
              gemm_param_ext.b.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1,  ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
              gemm_param_ext.c.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, output,      mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              gemm_param_ext.c.secondary = (void*)&LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/8);
              cfg.fwd_compute_kernel_strd_fused_f32( &gemm_param_ext ); /* beta = 1.0 + relu */
            } else {
              gemm_param.op.tertiary = &blocks;
              gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
              gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
              gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              cfg.fwd_compute_kernel_strd( &gemm_param ); /* beta = 1.0 */
            }
          }
        }
      }
    } else {
      for ( mb1ofm1 = thr_begin; mb1ofm1 < thr_end; ++mb1ofm1 ) {
        mb1  = mb1ofm1%nBlocksMB;
        ofm1 = mb1ofm1/nBlocksMB;
        if ( cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS ) { /* bias only */
          gemm_param_ext.op.tertiary = &blocks;
          gemm_param_ext.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
          gemm_param_ext.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
          gemm_param_ext.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
          gemm_param_ext.d.primary = (void*)&LIBXSMM_VLA_ACCESS(2, bias, ofm1, 0, cfg.bk);
          cfg.fwd_compute_kernel2_strd_fused_f32( &gemm_param_ext ); /* beta = 0.0 + bias */
        } else if ( cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU || cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK ) { /* bias and relu */
          gemm_param_ext.op.tertiary = &blocks;
          gemm_param_ext.a.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
          gemm_param_ext.b.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
          gemm_param_ext.d.primary   = (void*)&LIBXSMM_VLA_ACCESS(2, bias,   ofm1, 0, cfg.bk);
          gemm_param_ext.c.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, output,      mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
          gemm_param_ext.c.secondary = (void*)&LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/8);
          cfg.fwd_compute_kernel3_strd_fused_f32( &gemm_param_ext ); /* beta = 0.0 + bias + relu */
        } else if ( cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK ) { /* relu only */
          gemm_param_ext.op.tertiary = &blocks;
          gemm_param_ext.a.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
          gemm_param_ext.b.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1,  0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
          gemm_param_ext.c.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, output,      mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
          gemm_param_ext.c.secondary = (void*)&LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/8);
          cfg.fwd_compute_kernel4_strd_fused_f32( &gemm_param_ext ); /* beta = 0.0 + relu */
        } else { /* no fusion */
          gemm_param.op.tertiary = &blocks;
          gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, cfg.bc, cfg.bk);
          gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
          gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
          cfg.fwd_compute_kernel2_strd( &gemm_param ); /* beta = 0.0 */
        }
      }
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);
}

LIBXSMM_API void libxsmm_dnn_fc_fwd_exec_bf16_vnni_format( libxsmm_dnn_fc_fwd_config cfg, const libxsmm_bfloat16* wt_ptr, const libxsmm_bfloat16* in_act_ptr, libxsmm_bfloat16* out_act_ptr,
                                      const libxsmm_bfloat16* bias_ptr, unsigned char* relu_ptr, int start_tid, int my_tid, void* scratch )
{
  const libxsmm_blasint nBlocksIFm = cfg.C / cfg.bc;
  const libxsmm_blasint nBlocksOFm = cfg.K / cfg.bk;
  const libxsmm_blasint nBlocksMB  = cfg.N / cfg.bn;
  const libxsmm_blasint bn = cfg.bn;
  const libxsmm_blasint bk = cfg.bk;
  const libxsmm_blasint lpb = 2;
  const libxsmm_blasint bc_lp = (cfg.bc/lpb > 0) ? cfg.bc/lpb : 1;
  /* const libxsmm_blasint bc = cfg.bc;*/
  libxsmm_blasint use_2d_blocking = cfg.fwd_2d_blocking;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;
  /* number of tasks that could be run in parallel */
  const libxsmm_blasint work = nBlocksOFm * nBlocksMB;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

  /* loop variables */
  libxsmm_blasint mb1ofm1 = 0, mb1 = 0, ofm1 = 0, ifm1 = 0;
  libxsmm_blasint N_tasks_per_thread = 0, M_tasks_per_thread = 0, my_M_start = 0, my_M_end = 0, my_N_start = 0, my_N_end = 0, my_col_id = 0, my_row_id = 0, col_teams = 0, row_teams = 0;
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16,       output,  out_act_ptr, nBlocksOFm, cfg.bn, cfg.bk);
  LIBXSMM_VLA_DECL(4, const libxsmm_bfloat16,  input,   in_act_ptr,  nBlocksIFm, cfg.bn, cfg.bc);
  LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16, filter,       wt_ptr, nBlocksIFm, bc_lp, cfg.bk, lpb);
  LIBXSMM_VLA_DECL(4, float, output_f32, (float*)scratch, nBlocksOFm, bn, bk);
  LIBXSMM_VLA_DECL(2, const libxsmm_bfloat16, bias, ((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_BIAS) == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS) ? (libxsmm_bfloat16*) bias_ptr : NULL, cfg.bk);
  LIBXSMM_VLA_DECL(4, unsigned int,  relubitmask, ((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_RELU) == LIBXSMM_DNN_FC_ELTW_FUSE_RELU) ? (unsigned int*)relu_ptr : NULL, nBlocksOFm, cfg.bn, cfg.bk/32);

  libxsmm_meltw_unary_param unary_st_param;
  libxsmm_meltw_unary_param unary_ld_param;
  libxsmm_gemm_param gemm_param;
  libxsmm_gemm_ext_param gemm_ext_param;

  unsigned long long  blocks = nBlocksIFm;
  libxsmm_blasint CB_BLOCKS = nBlocksIFm, BF = 1;

  memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );
  memset( &gemm_ext_param, 0, sizeof(libxsmm_gemm_ext_param) );
  memset( &unary_st_param, 0, sizeof(libxsmm_meltw_unary_param) );
  memset( &unary_ld_param, 0, sizeof(libxsmm_meltw_unary_param) );

  BF = cfg.fwd_bf;
  CB_BLOCKS = nBlocksIFm/BF;
  blocks = CB_BLOCKS;

  if (use_2d_blocking == 1) {
    int _ltid, M_hyperpartition_id, N_hyperpartition_id, _nBlocksOFm, _nBlocksMB, hyperteam_id;
    col_teams    = cfg.fwd_col_teams;
    row_teams = cfg.fwd_row_teams;
    hyperteam_id = ltid/(col_teams*row_teams);
    _nBlocksOFm  = nBlocksOFm/cfg.fwd_M_hyperpartitions;
    _nBlocksMB   = nBlocksMB/cfg.fwd_N_hyperpartitions;
    _ltid = ltid % (col_teams * row_teams);
    M_hyperpartition_id = hyperteam_id % cfg.fwd_M_hyperpartitions;
    N_hyperpartition_id = hyperteam_id / cfg.fwd_M_hyperpartitions;
    my_row_id = _ltid % row_teams;
    my_col_id = _ltid / row_teams;
    N_tasks_per_thread = (_nBlocksMB + col_teams-1)/col_teams;
    M_tasks_per_thread = (_nBlocksOFm + row_teams-1)/row_teams;
    my_N_start = N_hyperpartition_id * _nBlocksMB + LIBXSMM_MIN( my_col_id * N_tasks_per_thread, _nBlocksMB);
    my_N_end   = N_hyperpartition_id * _nBlocksMB + LIBXSMM_MIN( (my_col_id+1) * N_tasks_per_thread, _nBlocksMB);
    my_M_start = M_hyperpartition_id * _nBlocksOFm + LIBXSMM_MIN( my_row_id * M_tasks_per_thread, _nBlocksOFm);
    my_M_end   = M_hyperpartition_id * _nBlocksOFm + LIBXSMM_MIN( (my_row_id+1) * M_tasks_per_thread, _nBlocksOFm);
  }

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  cfg.fwd_tileconfig_kernel( NULL );

  if (use_2d_blocking == 1) {
    if ((BF > 1) || (cfg.K % 32 != 0)) {
      for ( ifm1 = 0; ifm1 < BF; ++ifm1 ) {
        for (ofm1 = my_M_start; ofm1 < my_M_end; ++ofm1) {
          for (mb1 = my_N_start; mb1 < my_N_end; ++mb1) {
            if ( ifm1 == 0 ) {
              if ( (cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_BIAS) == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS ) {
                unary_ld_param.in.primary  = (void*) &LIBXSMM_VLA_ACCESS(2, bias, ofm1, 0,cfg.bk);
                unary_ld_param.out.primary = &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm,cfg.bn,cfg.bk);
                cfg.fwd_colbcast_load_kernel( &unary_ld_param );
              } else {
                unary_ld_param.out.primary = &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                cfg.fwd_zero_kernel( &unary_ld_param );
              }
            }
            gemm_param.op.tertiary = (void*)&blocks;
            gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(5, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, 0, nBlocksIFm, bc_lp, cfg.bk, lpb);
            gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
            gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
            cfg.fwd_compute_kernel_strd( &gemm_param );
            if ( ifm1 == BF-1  ) {
              if ( ((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_RELU) == LIBXSMM_DNN_FC_ELTW_FUSE_RELU ) ||
                   ((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK) == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK) ) {
                unary_st_param.in.primary = &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                unary_st_param.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                if ((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK) == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK) {
                  unary_st_param.out.secondary = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/32);
                }
                cfg.fwd_act_kernel( &unary_st_param );
              } else {
                unary_st_param.in.primary = &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                unary_st_param.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                cfg.fwd_store_kernel( &unary_st_param );
              }
            }
          }
        }
      }
    } else {
      for (ofm1 = my_M_start; ofm1 < my_M_end; ++ofm1) {
        for (mb1 = my_N_start; mb1 < my_N_end; ++mb1) {
          if ( ((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_BIAS) == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS) ||
               ((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_RELU) == LIBXSMM_DNN_FC_ELTW_FUSE_RELU) ||
               ((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK) == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK) ) {
            gemm_ext_param.op.tertiary = (void*)&blocks;
            gemm_ext_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(5, filter, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, cfg.bk, lpb);
            gemm_ext_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
            gemm_ext_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
            if ((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_BIAS) == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS) {
              gemm_ext_param.d.primary = (void*)&LIBXSMM_VLA_ACCESS(2, bias, ofm1, 0, cfg.bk);
            }
            if ((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK) == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK) {
              gemm_ext_param.c.secondary = (void*)&LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/32);
            }
            cfg.fwd_compute_kernel5_strd_fused( &gemm_ext_param );
          } else {
            gemm_param.op.tertiary = (void*)&blocks;
            gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(5, filter, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, cfg.bk, lpb);
            gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
            gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
            cfg.fwd_compute_kernel2_strd( &gemm_param );
          }
        }
      }
    }
  } else {
    if ((BF > 1) || (cfg.K % 32 != 0)) {
      for ( ifm1 = 0; ifm1 < BF; ++ifm1 ) {
        for ( mb1ofm1 = thr_begin; mb1ofm1 < thr_end; ++mb1ofm1 ) {
          mb1  = mb1ofm1%nBlocksMB;
          ofm1 = mb1ofm1/nBlocksMB;
          /* Initialize libxsmm_blasintermediate f32 tensor */
          if ( ifm1 == 0 ) {
            if ( (cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_BIAS) == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS ) {
              unary_ld_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(2, bias, ofm1, 0,cfg.bk);
              unary_ld_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm,cfg.bn,cfg.bk);
              cfg.fwd_colbcast_load_kernel( &unary_ld_param );
            } else {
              unary_ld_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              cfg.fwd_zero_kernel( &unary_ld_param );
            }
          }
          gemm_param.op.tertiary = (void*)&blocks;
          gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(5, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, 0, nBlocksIFm, bc_lp, cfg.bk, lpb);
          gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
          gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
          cfg.fwd_compute_kernel_strd( &gemm_param );
          if ( ifm1 == BF-1  ) {
            if ( ((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_RELU) == LIBXSMM_DNN_FC_ELTW_FUSE_RELU) ||
                 ((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK) == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK) ) {
              unary_st_param.in.primary = &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              unary_st_param.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              if ( (cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK) == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK ) {
                unary_st_param.out.secondary = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/32);
              }
              cfg.fwd_act_kernel( &unary_st_param );
            } else {
              unary_st_param.in.primary = &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              unary_st_param.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              cfg.fwd_store_kernel( &unary_st_param );
            }
          }
        }
      }
    } else {
      for ( mb1ofm1 = thr_begin; mb1ofm1 < thr_end; ++mb1ofm1 ) {
        mb1  = mb1ofm1%nBlocksMB;
        ofm1 = mb1ofm1/nBlocksMB;
        if ( ((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_BIAS) == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS) ||
             ((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_RELU) == LIBXSMM_DNN_FC_ELTW_FUSE_RELU) ||
             ((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK) == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK) ) {
          gemm_ext_param.op.tertiary = (void*)&blocks;
          gemm_ext_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(5, filter, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, cfg.bk, lpb);
          gemm_ext_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
          gemm_ext_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
          if ((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_BIAS) == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS) {
            gemm_ext_param.d.primary = (void*)&LIBXSMM_VLA_ACCESS(2, bias, ofm1, 0, cfg.bk);
          }
          if ((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK) == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK) {
            gemm_ext_param.c.secondary = (void*)&LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/32);
          }
          cfg.fwd_compute_kernel5_strd_fused( &gemm_ext_param );
        } else {
          gemm_param.op.tertiary = (void*)&blocks;
          gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(5, filter, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, cfg.bk, lpb);
          gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,  mb1, 0, 0, 0, nBlocksIFm, cfg.bn, cfg.bc);
          gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
          cfg.fwd_compute_kernel2_strd( &gemm_param );
        }
      }
    }
  }

  cfg.fwd_tilerelease_kernel( NULL );
  libxsmm_barrier_wait(cfg.barrier, ltid);
}

LIBXSMM_API void libxsmm_dnn_fc_fwd_exec_bf16( libxsmm_dnn_fc_fwd_config cfg, const libxsmm_bfloat16* wt_ptr, const libxsmm_bfloat16* in_act_ptr, libxsmm_bfloat16* out_act_ptr,
                          const libxsmm_bfloat16* bias_ptr, unsigned char* relu_ptr, int start_tid, int my_tid, void* scratch )
{
  libxsmm_dnn_fc_fwd_exec_bf16_vnni_format( cfg, wt_ptr, in_act_ptr, out_act_ptr, bias_ptr, relu_ptr, start_tid, my_tid, scratch );
}

LIBXSMM_API void libxsmm_dnn_fc_bwd_exec_f32( libxsmm_dnn_fc_bwd_config cfg, const float* wt_ptr, float* din_act_ptr,
    const float* dout_act_ptr, float* dwt_ptr, const float* in_act_ptr,
    float* dbias_ptr, const unsigned char* relu_ptr, libxsmm_dnn_fc_pass pass, int start_tid, int my_tid, void* scratch ) {
  /* here we assume that input and output blocking is similar */
  const libxsmm_blasint bn = cfg.bn;
  const libxsmm_blasint bk = cfg.bk;
  const libxsmm_blasint bc = cfg.bc;
  const libxsmm_blasint nBlocksIFm = cfg.C / bc;
  const libxsmm_blasint nBlocksOFm = cfg.K / bk;
  const libxsmm_blasint nBlocksMB  = cfg.N / bn;

  libxsmm_gemm_param gemm_param;

  gemm_param.a.secondary = NULL;
  gemm_param.b.secondary = NULL;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;

  /* number of tasks for transpose that could be run in parallel */
  const libxsmm_blasint eltwise_work = nBlocksOFm * nBlocksMB;
  /* compute chunk size */
  const libxsmm_blasint eltwise_chunksize = (eltwise_work % cfg.threads == 0) ? (eltwise_work / cfg.threads) : ((eltwise_work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint eltwise_thr_begin = (ltid * eltwise_chunksize < eltwise_work) ? (ltid * eltwise_chunksize) : eltwise_work;
  const libxsmm_blasint eltwise_thr_end = ((ltid + 1) * eltwise_chunksize < eltwise_work) ? ((ltid + 1) * eltwise_chunksize) : eltwise_work;
  libxsmm_blasint mb1ofm1;

  /* number of tasks for transpose that could be run in parallel */
  const libxsmm_blasint dbias_work = nBlocksOFm;
  /* compute chunk size */
  const libxsmm_blasint dbias_chunksize = (dbias_work % cfg.threads == 0) ? (dbias_work / cfg.threads) : ((dbias_work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint dbias_thr_begin = (ltid * dbias_chunksize < dbias_work) ? (ltid * dbias_chunksize) : dbias_work;
  const libxsmm_blasint dbias_thr_end = ((ltid + 1) * dbias_chunksize < dbias_work) ? ((ltid + 1) * dbias_chunksize) : dbias_work;

  /* loop variables */
  libxsmm_blasint ofm1 = 0, mb1 = 0, ofm2 = 0;

  float *grad_output_ptr = ( (cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU ||
                              cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK)
                              ? ((float*)scratch)+(cfg.C*cfg.K) : (float*)dout_act_ptr);
  LIBXSMM_VLA_DECL(4, const float, doutput_orig,    dout_act_ptr, nBlocksOFm, bn, bk);
  LIBXSMM_VLA_DECL(4,       float,      doutput, grad_output_ptr, nBlocksOFm, bn, bk);

  LIBXSMM_VLA_DECL(2,               float,    dbias, dbias_ptr,                     cfg.bk);
  LIBXSMM_VLA_DECL(4, const unsigned char, relubitmask,  relu_ptr, nBlocksOFm, cfg.bn, cfg.bk/8);
  libxsmm_meltw_unary_param eltwise_params;
  libxsmm_meltw_unary_param trans_param;

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  if (cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU ||
      cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK) {
    for ( mb1ofm1 = eltwise_thr_begin; mb1ofm1 < eltwise_thr_end; ++mb1ofm1 ) {
      mb1  = mb1ofm1%nBlocksMB;
      ofm1 = mb1ofm1/nBlocksMB;
      eltwise_params.in.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, doutput_orig, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
      eltwise_params.out.primary  = &LIBXSMM_VLA_ACCESS(4, doutput, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
      eltwise_params.in.secondary = ((cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK) ?
                                      (void*)&LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/8) : NULL);
      cfg.bwd_relu_kernel(&eltwise_params);
    }

    /* wait for eltwise to finish */
    libxsmm_barrier_wait(cfg.barrier, ltid);
  }

  if (cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS || cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU ||
      cfg.fuse_type == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK) {
    for ( ofm1 = dbias_thr_begin; ofm1 < dbias_thr_end; ++ofm1 ) {
      eltwise_params.in.primary    = &LIBXSMM_VLA_ACCESS(4,  doutput, 0, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
      eltwise_params.out.primary   = &LIBXSMM_VLA_ACCESS(2,  dbias, ofm1, 0, cfg.bk);
      cfg.delbias_reduce_kernel(&eltwise_params);
    }

    /* wait for eltwise to finish */
    libxsmm_barrier_wait(cfg.barrier, ltid);
  }

  if ( (pass & LIBXSMM_DNN_FC_PASS_BWD_D) == LIBXSMM_DNN_FC_PASS_BWD_D ) {
    const libxsmm_blasint use_2d_blocking = cfg.bwd_2d_blocking;

    /* number of tasks that could be run in parallel */
    const libxsmm_blasint work = nBlocksIFm * nBlocksMB;
    /* compute chunk size */
    const libxsmm_blasint chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
    /* compute thr_begin and thr_end */
    const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
    const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

    /* number of tasks for transpose that could be run in parallel */
    const libxsmm_blasint transpose_work = nBlocksIFm * nBlocksOFm;
    /* compute chunk size */
    const libxsmm_blasint transpose_chunksize = (transpose_work % cfg.threads == 0) ? (transpose_work / cfg.threads) : ((transpose_work / cfg.threads) + 1);
    /* compute thr_begin and thr_end */
    const libxsmm_blasint transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
    const libxsmm_blasint transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;

    /* loop variables */
    libxsmm_blasint ifm1 = 0, ifm1ofm1 = 0, mb1ifm1 = 0;
    libxsmm_blasint N_tasks_per_thread = 0, M_tasks_per_thread = 0, my_M_start = 0, my_M_end = 0, my_N_start = 0, my_N_end = 0, my_col_id = 0, my_row_id = 0, col_teams = 0, row_teams = 0;

    LIBXSMM_VLA_DECL(4, const float,    filter,          wt_ptr, nBlocksIFm, bc, bk);
    LIBXSMM_VLA_DECL(4,       float,    dinput,     din_act_ptr, nBlocksIFm, bn, bc);
    LIBXSMM_VLA_DECL(4,       float, filter_tr, (float*)scratch, nBlocksOFm, bk, bc);

    unsigned long long  blocks = nBlocksOFm;
    libxsmm_blasint KB_BLOCKS = nBlocksOFm, BF = 1;
    BF = cfg.bwd_bf;
    KB_BLOCKS = nBlocksOFm/BF;
    blocks = KB_BLOCKS;

    if (use_2d_blocking == 1) {
      col_teams = cfg.bwd_col_teams;
      row_teams = cfg.bwd_row_teams;
      my_row_id = ltid % row_teams;
      my_col_id = ltid / row_teams;
      N_tasks_per_thread = LIBXSMM_UPDIV(nBlocksMB, col_teams);
      M_tasks_per_thread = LIBXSMM_UPDIV(nBlocksIFm, row_teams);
      my_N_start = LIBXSMM_MIN(my_col_id * N_tasks_per_thread, nBlocksMB);
      my_N_end = LIBXSMM_MIN((my_col_id+1) * N_tasks_per_thread, nBlocksMB);
      my_M_start = LIBXSMM_MIN(my_row_id * M_tasks_per_thread, nBlocksIFm);
      my_M_end = LIBXSMM_MIN((my_row_id+1) * M_tasks_per_thread, nBlocksIFm);
    }

    /* transpose weight */
    for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
      ofm1 = ifm1ofm1 / nBlocksIFm;
      ifm1 = ifm1ofm1 % nBlocksIFm;
      trans_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(4,    filter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk);
      trans_param.out.primary = &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1, 0, 0, nBlocksOFm, bk, bc);
      cfg.norm_to_normT_kernel(&trans_param);
    }

    /* wait for transpose to finish */
    libxsmm_barrier_wait(cfg.barrier, ltid);

    if (use_2d_blocking == 1) {
      if (BF > 1) {
        for ( ofm1 = 0; ofm1 < BF; ++ofm1 ) {
          for (ifm1 = my_M_start; ifm1 < my_M_end; ++ifm1) {
            for (mb1 = my_N_start; mb1 < my_N_end; ++mb1) {
              /* Initialize intermediate f32 tensor */
              if ( ofm1 == 0 ) {
                eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, dinput, mb1, ifm1, 0, 0, nBlocksIFm, bn, bc);
                cfg.bwd_zero_kernel(&eltwise_params);
              }
              gemm_param.op.tertiary = &blocks;
              gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bk, bc );
              gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk);
              gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
              cfg.bwd_compute_kernel_strd( &gemm_param );
            }
          }
        }
      } else {
        for (ifm1 = my_M_start; ifm1 < my_M_end; ++ifm1) {
          for (mb1 = my_N_start; mb1 < my_N_end; ++mb1) {
              gemm_param.op.tertiary = &blocks;
              gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, 0, 0, 0, nBlocksOFm, bk, bc);
              gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  0, 0, 0, nBlocksOFm, bn, bk);
              gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
              cfg.bwd_compute_kernel2_strd( &gemm_param );
          }
        }
      }
    } else {
      if (BF > 1) {
        for ( ofm1 = 0; ofm1 < BF; ++ofm1 ) {
          for ( mb1ifm1 = thr_begin; mb1ifm1 < thr_end; ++mb1ifm1 ) {
            mb1  = mb1ifm1%nBlocksMB;
            ifm1 = mb1ifm1/nBlocksMB;
            /* Initialize intermediate f32 tensor */
            if ( ofm1 == 0 ) {
              eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, dinput, mb1, ifm1, 0, 0, nBlocksIFm, bn, bc);
              cfg.bwd_zero_kernel(&eltwise_params);
            }
            gemm_param.op.tertiary = &blocks;
            gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bk, bc );
            gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk);
            gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
            cfg.bwd_compute_kernel_strd( &gemm_param );
          }
        }
      } else {
        for ( mb1ifm1 = thr_begin; mb1ifm1 < thr_end; ++mb1ifm1 ) {
          mb1  = mb1ifm1%nBlocksMB;
          ifm1 = mb1ifm1/nBlocksMB;
          gemm_param.op.tertiary = &blocks;
          gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, 0, 0, 0, nBlocksOFm, bk, bc);
          gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  0, 0, 0, nBlocksOFm, bn, bk);
          gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
          cfg.bwd_compute_kernel2_strd( &gemm_param );
        }
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);
  }

  if ( (pass & LIBXSMM_DNN_FC_PASS_BWD_W) == LIBXSMM_DNN_FC_PASS_BWD_W ) {
    /* number of tasks that could be run in parallel */
    const libxsmm_blasint ofm_subtasks = (cfg.upd_2d_blocking == 1) ? 1 : cfg.ofm_subtasks;
    const libxsmm_blasint ifm_subtasks = (cfg.upd_2d_blocking == 1) ? 1 : cfg.ifm_subtasks;
    const libxsmm_blasint bbk = (cfg.upd_2d_blocking == 1) ? bk : bk/ofm_subtasks;
    const libxsmm_blasint bbc = (cfg.upd_2d_blocking == 1) ? bc : bc/ifm_subtasks;
    const libxsmm_blasint work = nBlocksIFm * ifm_subtasks * nBlocksOFm * ofm_subtasks;
    const libxsmm_blasint Cck_work = nBlocksIFm * ifm_subtasks * ofm_subtasks;
    const libxsmm_blasint Cc_work = nBlocksIFm * ifm_subtasks;

    /* 2D blocking parameters  */
    libxsmm_blasint use_2d_blocking = cfg.upd_2d_blocking;
    libxsmm_blasint N_tasks_per_thread = 0, M_tasks_per_thread = 0, my_M_start = 0, my_M_end = 0, my_N_start = 0, my_N_end = 0, my_col_id = 0, my_row_id = 0, col_teams = 0, row_teams = 0;

    /* compute chunk size */
    const libxsmm_blasint chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
    /* compute thr_begin and thr_end */
    const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
    const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
    libxsmm_blasint BF = cfg.upd_bf;

    /* loop variables */
    libxsmm_blasint ifm1ofm1 = 0, ifm1 = 0, ifm2 = 0, bfn = 0;

    /* Batch reduce related variables */
    unsigned long long  blocks = nBlocksMB/BF;

    LIBXSMM_VLA_DECL(4, const float,   input, in_act_ptr, nBlocksIFm, bn, bc);
    LIBXSMM_VLA_DECL(4,       float, dfilter,    dwt_ptr, nBlocksIFm, bc, bk);

    if (use_2d_blocking == 1) {
      col_teams = cfg.upd_col_teams;
      row_teams = cfg.upd_row_teams;
      my_row_id = ltid % row_teams;
      my_col_id = ltid / row_teams;
      N_tasks_per_thread = LIBXSMM_UPDIV(nBlocksIFm, col_teams);
      M_tasks_per_thread = LIBXSMM_UPDIV(nBlocksOFm, row_teams);
      my_N_start = LIBXSMM_MIN(my_col_id * N_tasks_per_thread, nBlocksIFm);
      my_N_end = LIBXSMM_MIN((my_col_id+1) * N_tasks_per_thread, nBlocksIFm);
      my_M_start = LIBXSMM_MIN(my_row_id * M_tasks_per_thread, nBlocksOFm);
      my_M_end = LIBXSMM_MIN((my_row_id+1) * M_tasks_per_thread, nBlocksOFm);
    }

    if (use_2d_blocking == 1) {
      if (BF == 1) {
        for (ofm1 = my_M_start; ofm1 < my_M_end; ++ofm1) {
          for (ifm1 = my_N_start; ifm1 < my_N_end; ++ifm1) {
            gemm_param.op.tertiary = &blocks;
            gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, doutput, 0, ofm1, 0, 0, nBlocksOFm, bn, bk);
            gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,   0, ifm1, 0, 0, nBlocksIFm, bn, bc);
            gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk);
            cfg.upd_compute_kernel2_strd( &gemm_param );
          }
        }
      } else {
        for (bfn = 0; bfn < BF; bfn++) {
          for (ofm1 = my_M_start; ofm1 < my_M_end; ++ofm1) {
            for (ifm1 = my_N_start; ifm1 < my_N_end; ++ifm1) {
              /* initialize current work task to zero */
              if (bfn == 0) {
                eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk);
                cfg.upd_zero_kernel(&eltwise_params);
              }
              gemm_param.op.tertiary = &blocks;
              gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, doutput, bfn*blocks, ofm1, 0, 0, nBlocksOFm, bn, bk);
              gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,   bfn*blocks, ifm1, 0, 0, nBlocksIFm, bn, bc);
              gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk);
              cfg.upd_compute_kernel_strd( &gemm_param );
            }
          }
        }
      }
    } else {
      if (BF == 1) {
        for ( ifm1ofm1 = thr_begin; ifm1ofm1 < thr_end; ++ifm1ofm1 ) {
          ofm1 = ifm1ofm1 / Cck_work;
          ofm2 = (ifm1ofm1 % Cck_work) / Cc_work;
          ifm1 = ((ifm1ofm1 % Cck_work) % Cc_work) / ifm_subtasks;
          ifm2 = ((ifm1ofm1 % Cck_work) % Cc_work) % ifm_subtasks;

          gemm_param.op.tertiary = &blocks;
          gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, doutput, 0, ofm1, 0, ofm2*bbk, nBlocksOFm, bn, bk);
          gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,   0, ifm1, 0, ifm2*bbc, nBlocksIFm, bn, bc);
          gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk);
          cfg.upd_compute_kernel2_strd( &gemm_param );
        }
      } else {
        for (bfn = 0; bfn < BF; bfn++) {
          for ( ifm1ofm1 = thr_begin; ifm1ofm1 < thr_end; ++ifm1ofm1 ) {
            ofm1 = ifm1ofm1 / Cck_work;
            ofm2 = (ifm1ofm1 % Cck_work) / Cc_work;
            ifm1 = ((ifm1ofm1 % Cck_work) % Cc_work) / ifm_subtasks;
            ifm2 = ((ifm1ofm1 % Cck_work) % Cc_work) % ifm_subtasks;

            /* initialize current work task to zero */
            if (bfn == 0) {
              eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk);
              cfg.upd_zero_kernel(&eltwise_params);
            }

            gemm_param.op.tertiary = &blocks;
            gemm_param.a.primary = (void*)&LIBXSMM_VLA_ACCESS(4, doutput, bfn*blocks, ofm1, 0, ofm2*bbk, nBlocksOFm, bn, bk);
            gemm_param.b.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input,   bfn*blocks, ifm1, 0, ifm2*bbc, nBlocksIFm, bn, bc);
            gemm_param.c.primary = (void*)&LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk);
            cfg.upd_compute_kernel_strd( &gemm_param );
          }
        }
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);
  }
}

LIBXSMM_API void libxsmm_dnn_fc_bwd_exec_bf16_vnni_format( libxsmm_dnn_fc_bwd_config cfg,  const libxsmm_bfloat16* wt_ptr, libxsmm_bfloat16* din_act_ptr,
                                      const libxsmm_bfloat16* dout_act_ptr, libxsmm_bfloat16* dwt_ptr, const libxsmm_bfloat16* in_act_ptr,
                                      libxsmm_bfloat16* dbias_ptr, const unsigned char* relu_ptr, libxsmm_dnn_fc_pass pass, int start_tid, int my_tid, void* scratch )
{
  /* size variables, all const */
  /* here we assume that input and output blocking is similar */
  const libxsmm_blasint bn = cfg.bn;
  const libxsmm_blasint bk = cfg.bk;
  const libxsmm_blasint bc = cfg.bc;
  libxsmm_blasint lpb = 2;
  const libxsmm_blasint bc_lp = (bc/lpb > 0) ? bc/lpb : 1;
  const libxsmm_blasint bk_lp = (bk/lpb > 0) ? bk/lpb : 1;
  const libxsmm_blasint bn_lp = (bn/lpb > 0) ? bn/lpb : 1;
  const libxsmm_blasint nBlocksIFm = cfg.C / cfg.bc;
  const libxsmm_blasint nBlocksOFm = cfg.K / cfg.bk;
  const libxsmm_blasint nBlocksMB  = cfg.N / cfg.bn;
  libxsmm_blasint mb1ofm1 = 0, mb1 = 0, ofm1 = 0, ofm2 = 0;
  libxsmm_blasint performed_doutput_transpose = 0;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;

  /* number of tasks for transpose that could be run in parallel */
  const libxsmm_blasint eltwise_work = nBlocksOFm * nBlocksMB;
  /* compute chunk size */
  const libxsmm_blasint eltwise_chunksize = (eltwise_work % cfg.threads == 0) ? (eltwise_work / cfg.threads) : ((eltwise_work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint eltwise_thr_begin = (ltid * eltwise_chunksize < eltwise_work) ? (ltid * eltwise_chunksize) : eltwise_work;
  const libxsmm_blasint eltwise_thr_end = ((ltid + 1) * eltwise_chunksize < eltwise_work) ? ((ltid + 1) * eltwise_chunksize) : eltwise_work;

  /* number of tasks for transpose that could be run in parallel */
  const libxsmm_blasint dbias_work = nBlocksOFm;
  /* compute chunk size */
  const libxsmm_blasint dbias_chunksize = (dbias_work % cfg.threads == 0) ? (dbias_work / cfg.threads) : ((dbias_work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint dbias_thr_begin = (ltid * dbias_chunksize < dbias_work) ? (ltid * dbias_chunksize) : dbias_work;
  const libxsmm_blasint dbias_thr_end = ((ltid + 1) * dbias_chunksize < dbias_work) ? ((ltid + 1) * dbias_chunksize) : dbias_work;

  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, dbias, ((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_BIAS) == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS) ? (libxsmm_bfloat16*) dbias_ptr : NULL, cfg.bk);
  LIBXSMM_VLA_DECL(4, unsigned int, relubitmask, ((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_RELU) == LIBXSMM_DNN_FC_ELTW_FUSE_RELU) ? (unsigned int*)relu_ptr : NULL, nBlocksOFm, cfg.bn, cfg.bk/32);

  libxsmm_bfloat16 *grad_output_ptr = (((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_RELU) == LIBXSMM_DNN_FC_ELTW_FUSE_RELU)) ? (libxsmm_bfloat16*)((char*)scratch + cfg.doutput_scratch_mark) : (libxsmm_bfloat16*)dout_act_ptr;
  libxsmm_bfloat16 *tr_doutput_ptr = (((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_RELU) == LIBXSMM_DNN_FC_ELTW_FUSE_RELU)) ? (libxsmm_bfloat16*)grad_output_ptr + cfg.N * cfg.K : (libxsmm_bfloat16*)scratch;
  LIBXSMM_VLA_DECL(4, const libxsmm_bfloat16,   doutput_orig, dout_act_ptr, nBlocksOFm, bn, bk);
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16,   doutput, grad_output_ptr, nBlocksOFm, bn, bk);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, doutput_tr, tr_doutput_ptr, nBlocksMB, bn_lp, bk, lpb);

  libxsmm_meltw_unary_param format_param;
  libxsmm_meltw_unary_param copy_param;
  libxsmm_meltw_unary_param delbias_param;
  libxsmm_meltw_unary_param act_param;
  libxsmm_gemm_param gemm_param;

  memset( &format_param, 0, sizeof(libxsmm_meltw_unary_param) );
  memset( &copy_param, 0, sizeof(libxsmm_meltw_unary_param) );
  memset( &delbias_param, 0, sizeof(libxsmm_meltw_unary_param) );
  memset( &act_param, 0, sizeof(libxsmm_meltw_unary_param) );
  memset( &gemm_param, 0, sizeof(libxsmm_gemm_param) );

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  /* Apply to doutput potential fusions */
  if (((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_RELU) == LIBXSMM_DNN_FC_ELTW_FUSE_RELU)) {
    for ( mb1ofm1 = eltwise_thr_begin; mb1ofm1 < eltwise_thr_end; ++mb1ofm1 ) {
      mb1  = mb1ofm1/nBlocksOFm;
      ofm1 = mb1ofm1%nBlocksOFm;

      act_param.in.primary   = (void*) &LIBXSMM_VLA_ACCESS(4, doutput_orig, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
      act_param.out.primary  = (void*)&LIBXSMM_VLA_ACCESS(4, doutput, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
      act_param.in.secondary = (void*)&LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/32);
      cfg.bwd_relu_kernel( &act_param );

      /* If in UPD pass, also perform transpose of doutput  */
      if ( ( cfg.upd_2d_blocking == 0 ) && ((pass & LIBXSMM_DNN_FC_PASS_BWD_W) == LIBXSMM_DNN_FC_PASS_BWD_W) ) {
        format_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(4, doutput,  mb1, ofm1, 0, 0, nBlocksOFm, bn, bk);
        format_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, mb1, 0, 0, 0, nBlocksMB, bn_lp, bk, lpb);
        cfg.norm_to_vnni_kernel(&format_param);
        performed_doutput_transpose = 1;
      }
    }
    libxsmm_barrier_wait(cfg.barrier, ltid);
  }
  /* Accumulation of bias happens in f32 */
  if (((cfg.fuse_type & LIBXSMM_DNN_FC_ELTW_FUSE_BIAS) == LIBXSMM_DNN_FC_ELTW_FUSE_BIAS)) {
    for ( ofm1 = dbias_thr_begin; ofm1 < dbias_thr_end; ++ofm1 ) {
      delbias_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(4,  doutput, 0, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
      delbias_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(2,  dbias, ofm1, 0, cfg.bk);
      cfg.delbias_reduce_kernel(&delbias_param);
    }
    /* wait for eltwise to finish */
    libxsmm_barrier_wait(cfg.barrier, ltid);
  }

  if ( (pass & LIBXSMM_DNN_FC_PASS_BWD_D) == LIBXSMM_DNN_FC_PASS_BWD_D ){
    libxsmm_blasint use_2d_blocking = cfg.bwd_2d_blocking;

    /* number of tasks that could be run in parallel */
    const libxsmm_blasint work = nBlocksIFm * nBlocksMB;
    /* compute chunk size */
    const libxsmm_blasint chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
    /* compute thr_begin and thr_end */
    const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
    const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

    /* number of tasks for transpose that could be run in parallel */
    const libxsmm_blasint transpose_work = nBlocksIFm * nBlocksOFm;
    /* compute chunk size */
    const libxsmm_blasint transpose_chunksize = (transpose_work % cfg.threads == 0) ? (transpose_work / cfg.threads) : ((transpose_work / cfg.threads) + 1);
    /* compute thr_begin and thr_end */
    const libxsmm_blasint transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
    const libxsmm_blasint transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;

    /* loop variables */
    libxsmm_blasint ifm1 = 0, ifm1ofm1 = 0, mb1ifm1 = 0;
    libxsmm_blasint N_tasks_per_thread = 0, M_tasks_per_thread = 0, my_M_start = 0, my_M_end = 0, my_N_start = 0, my_N_end = 0, my_col_id = 0, my_row_id = 0, col_teams = 0, row_teams = 0;

    LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16, filter,    wt_ptr, nBlocksIFm, bc_lp, bk, lpb);
    LIBXSMM_VLA_DECL(4,       libxsmm_bfloat16, dinput,    din_act_ptr, nBlocksIFm, bn, bc);
    LIBXSMM_VLA_DECL(5,       libxsmm_bfloat16, filter_tr, (libxsmm_bfloat16*)scratch, nBlocksOFm, bk_lp, bc, lpb);
    float* temp_output = (float*)scratch + (cfg.C * cfg.K)/2;
    LIBXSMM_VLA_DECL(4,        float,           dinput_f32, (float*) temp_output, nBlocksIFm, bn, bc);

    unsigned long long  blocks = nBlocksOFm;
    libxsmm_blasint KB_BLOCKS = nBlocksOFm, BF = 1;
    BF = cfg.bwd_bf;
    KB_BLOCKS = nBlocksOFm/BF;
    blocks = KB_BLOCKS;

    cfg.bwd_tileconfig_kernel( NULL );

    if (use_2d_blocking == 1) {
      int _ltid, M_hyperpartition_id, N_hyperpartition_id, _nBlocksIFm, _nBlocksMB, hyperteam_id;
      col_teams    = cfg.bwd_col_teams;
      row_teams = cfg.bwd_row_teams;
      hyperteam_id = ltid/(col_teams*row_teams);
      _nBlocksIFm  = nBlocksIFm/cfg.bwd_M_hyperpartitions;
      _nBlocksMB   = nBlocksMB/cfg.bwd_N_hyperpartitions;
      _ltid = ltid % (col_teams * row_teams);
      M_hyperpartition_id = hyperteam_id % cfg.bwd_M_hyperpartitions;
      N_hyperpartition_id = hyperteam_id / cfg.bwd_M_hyperpartitions;
      my_row_id = _ltid % row_teams;
      my_col_id = _ltid / row_teams;
      N_tasks_per_thread = (_nBlocksMB + col_teams-1)/col_teams;
      M_tasks_per_thread = (_nBlocksIFm + row_teams-1)/row_teams;
      my_N_start = N_hyperpartition_id * _nBlocksMB + LIBXSMM_MIN( my_col_id * N_tasks_per_thread, _nBlocksMB);
      my_N_end   = N_hyperpartition_id * _nBlocksMB + LIBXSMM_MIN( (my_col_id+1) * N_tasks_per_thread, _nBlocksMB);
      my_M_start = M_hyperpartition_id * _nBlocksIFm + LIBXSMM_MIN( my_row_id * M_tasks_per_thread, _nBlocksIFm);
      my_M_end   = M_hyperpartition_id * _nBlocksIFm + LIBXSMM_MIN( (my_row_id+1) * M_tasks_per_thread, _nBlocksIFm);
    }

    /* transpose weight */
    if (cfg.bwd_2d_blocking == 0) {
      for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
        ofm1 = ifm1ofm1 / nBlocksIFm;
        ifm1 = ifm1ofm1 % nBlocksIFm;
        format_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(5, filter,  ofm1, ifm1, 0, 0, 0, nBlocksIFm, bc_lp, bk, lpb);
        format_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, ofm1, 0, 0, 0, nBlocksOFm, bk_lp, bc, lpb);
        cfg.vnni_to_vnniT_kernel( &format_param );
      }
      /* wait for transpose to finish */
      libxsmm_barrier_wait(cfg.barrier, ltid);
    }

    if (use_2d_blocking == 1) {
      LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, tmp_filter_tr, ((libxsmm_bfloat16*)((char*)scratch + cfg.bwd_private_tr_wt_scratch_mark)) + ltid * bc * cfg.K, bk_lp, bc, lpb);
      if (BF > 1) {
        for ( ofm1 = 0; ofm1 < BF; ++ofm1 ) {
          for (ifm1 = my_M_start; ifm1 < my_M_end; ++ifm1) {
            for (ofm2 = ofm1*KB_BLOCKS; ofm2 < (ofm1+1)*KB_BLOCKS; ofm2++) {
              format_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(5, filter,  ofm2, ifm1, 0, 0, 0, nBlocksIFm, bc_lp, bk, lpb);
              format_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(4, tmp_filter_tr, ofm2, 0, 0, 0, bk_lp, bc, lpb);
              cfg.vnni_to_vnniT_kernel( &format_param );
            }
            for (mb1 = my_N_start; mb1 < my_N_end; ++mb1) {
              /* Initialize libxsmm_blasintermediate f32 tensor */
              if ( ofm1 == 0 ) {
                copy_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(4, dinput_f32, mb1, ifm1, 0, 0, nBlocksIFm, bn, bc);
                cfg.bwd_zero_kernel( &copy_param );
              }
              gemm_param.op.tertiary = (void*)&blocks;
              gemm_param.a.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, tmp_filter_tr, ofm1*KB_BLOCKS, 0, 0, 0, bk_lp, bc, lpb);
              gemm_param.b.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk);
              gemm_param.c.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
              cfg.bwd_compute_kernel_strd( &gemm_param );
              /* downconvert libxsmm_blasintermediate f32 tensor to bf 16 and store to final C */
              if ( ofm1 == BF-1  ) {
                copy_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
                copy_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
                cfg.bwd_store_kernel( &copy_param );
              }
            }
          }
        }
      } else {
        for (ifm1 = my_M_start; ifm1 < my_M_end; ++ifm1) {
          for (ofm2 = 0; ofm2 < nBlocksOFm; ofm2++) {
            format_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(5, filter,  ofm2, ifm1, 0, 0, 0, nBlocksIFm, bc_lp, bk, lpb);
            format_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(4, tmp_filter_tr, ofm2, 0, 0, 0, bk_lp, bc, lpb);
            cfg.vnni_to_vnniT_kernel( &format_param );
          }
          for (mb1 = my_N_start; mb1 < my_N_end; ++mb1) {
            gemm_param.op.tertiary = (void*)&blocks;
            gemm_param.a.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, tmp_filter_tr, 0, 0, 0, 0, bk_lp, bc, lpb);
            gemm_param.b.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  0, 0, 0, nBlocksOFm, bn, bk);
            gemm_param.c.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
            cfg.bwd_compute_kernel2_strd( &gemm_param );
          }
        }
      }
    } else {
      if (BF > 1) {
        for ( ofm1 = 0; ofm1 < BF; ++ofm1 ) {
          for ( mb1ifm1 = thr_begin; mb1ifm1 < thr_end; ++mb1ifm1 ) {
            mb1  = mb1ifm1%nBlocksMB;
            ifm1 = mb1ifm1/nBlocksMB;
            /* Initialize libxsmm_blasintermediate f32 tensor */
            if ( ofm1 == 0 ) {
              copy_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(4, dinput_f32, mb1, ifm1, 0, 0, nBlocksIFm, bn, bc);
              cfg.bwd_zero_kernel( &copy_param );
            }
            gemm_param.op.tertiary = (void*)&blocks;
            gemm_param.a.primary   = (void*)&LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, 0, nBlocksOFm, bk_lp, bc, lpb);
            gemm_param.b.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk);
            gemm_param.c.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
            cfg.bwd_compute_kernel_strd( &gemm_param );
            /* downconvert libxsmm_blasintermediate f32 tensor to bf 16 and store to final C */
            if ( ofm1 == BF-1  ) {
              copy_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
              copy_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
              cfg.bwd_store_kernel( &copy_param );
            }
          }
        }
      } else {
        for ( mb1ifm1 = thr_begin; mb1ifm1 < thr_end; ++mb1ifm1 ) {
          mb1  = mb1ifm1%nBlocksMB;
          ifm1 = mb1ifm1/nBlocksMB;
          gemm_param.op.tertiary = (void*)&blocks;
          gemm_param.a.primary   = (void*)&LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, 0, 0, 0, 0, nBlocksOFm, bk_lp, bc, lpb);
          gemm_param.b.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  0, 0, 0, nBlocksOFm, bn, bk);
          gemm_param.c.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
          cfg.bwd_compute_kernel2_strd( &gemm_param );
        }
      }
    }

    cfg.bwd_tilerelease_kernel( NULL );
    libxsmm_barrier_wait(cfg.barrier, ltid);
  }

  if ( (pass & LIBXSMM_DNN_FC_PASS_BWD_W) == LIBXSMM_DNN_FC_PASS_BWD_W ) {
    /* number of tasks that could be run in parallel */
    const libxsmm_blasint ofm_subtasks = (cfg.upd_2d_blocking == 1) ? 1 : cfg.ofm_subtasks;
    const libxsmm_blasint ifm_subtasks = (cfg.upd_2d_blocking == 1) ? 1 : cfg.ifm_subtasks;
    const libxsmm_blasint bbk = (cfg.upd_2d_blocking == 1) ? bk : bk/ofm_subtasks;
    const libxsmm_blasint bbc = (cfg.upd_2d_blocking == 1) ? bc : bc/ifm_subtasks;
    const libxsmm_blasint work = nBlocksIFm * ifm_subtasks * nBlocksOFm * ofm_subtasks;
    const libxsmm_blasint Cck_work = nBlocksIFm * ifm_subtasks * ofm_subtasks;
    const libxsmm_blasint Cc_work = nBlocksIFm * ifm_subtasks;

    /* 2D blocking parameters  */
    libxsmm_blasint use_2d_blocking = cfg.upd_2d_blocking;
    libxsmm_blasint N_tasks_per_thread = 0, M_tasks_per_thread = 0, my_M_start = 0, my_M_end = 0, my_N_start = 0, my_N_end = 0, my_col_id = 0, my_row_id = 0, col_teams = 0, row_teams = 0;

    /* compute chunk size */
    const libxsmm_blasint chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
    /* compute thr_begin and thr_end */
    const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
    const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
    libxsmm_blasint BF = cfg.upd_bf;

    /* loop variables */
    libxsmm_blasint ifm1ofm1 = 0, ifm1 = 0, ifm2 = 0, bfn = 0, mb1ifm1 = 0, mb3 = 0;

    /* Batch reduce related variables */
    unsigned long long  blocks = nBlocksMB/BF;

    LIBXSMM_VLA_DECL(4, const libxsmm_bfloat16,  input,    (libxsmm_bfloat16* )in_act_ptr, nBlocksIFm, bn, bc);
    LIBXSMM_VLA_DECL(5,       libxsmm_bfloat16, dfilter,  (libxsmm_bfloat16*)dwt_ptr, nBlocksIFm, bc_lp, bk, lpb);

    /* Set up tensors for transposing/scratch before vnni reformatting dfilter */
    libxsmm_bfloat16  *tr_inp_ptr = (libxsmm_bfloat16*) ((libxsmm_bfloat16*)scratch + cfg.N * cfg.K);
    float               *dfilter_f32_ptr = (float*) ((libxsmm_bfloat16*)tr_inp_ptr + cfg.N * cfg.C);

    LIBXSMM_VLA_DECL(4, libxsmm_bfloat16,  input_tr,    (libxsmm_bfloat16*)tr_inp_ptr, nBlocksMB, bc, bn);
    LIBXSMM_VLA_DECL(4,       float, dfilter_f32,  (float*)dfilter_f32_ptr, nBlocksIFm, bc, bk);

    const libxsmm_blasint tr_out_work = nBlocksMB * nBlocksOFm;
    const libxsmm_blasint tr_out_chunksize = (tr_out_work % cfg.threads == 0) ? (tr_out_work / cfg.threads) : ((tr_out_work / cfg.threads) + 1);
    const libxsmm_blasint tr_out_thr_begin = (ltid * tr_out_chunksize < tr_out_work) ? (ltid * tr_out_chunksize) : tr_out_work;
    const libxsmm_blasint tr_out_thr_end = ((ltid + 1) * tr_out_chunksize < tr_out_work) ? ((ltid + 1) * tr_out_chunksize) : tr_out_work;

    const libxsmm_blasint tr_inp_work = nBlocksMB * nBlocksIFm;
    const libxsmm_blasint tr_inp_chunksize = (tr_inp_work % cfg.threads == 0) ? (tr_inp_work / cfg.threads) : ((tr_inp_work / cfg.threads) + 1);
    const libxsmm_blasint tr_inp_thr_begin = (ltid * tr_inp_chunksize < tr_inp_work) ? (ltid * tr_inp_chunksize) : tr_inp_work;
    const libxsmm_blasint tr_inp_thr_end = ((ltid + 1) * tr_inp_chunksize < tr_inp_work) ? ((ltid + 1) * tr_inp_chunksize) : tr_inp_work;

    cfg.upd_tileconfig_kernel( NULL );

    if (use_2d_blocking == 1) {
      int _ltid, M_hyperpartition_id, N_hyperpartition_id, _nBlocksOFm,  _nBlocksIFm, hyperteam_id;
      col_teams = cfg.upd_col_teams;
      row_teams = cfg.upd_row_teams;
      hyperteam_id = ltid/(col_teams*row_teams);
      _nBlocksOFm  = nBlocksOFm/cfg.upd_M_hyperpartitions;
      _nBlocksIFm  = nBlocksIFm/cfg.upd_N_hyperpartitions;
      _ltid = ltid % (col_teams * row_teams);
      M_hyperpartition_id = hyperteam_id % cfg.upd_M_hyperpartitions;
      N_hyperpartition_id = hyperteam_id / cfg.upd_M_hyperpartitions;
      my_row_id = _ltid % row_teams;
      my_col_id = _ltid / row_teams;
      N_tasks_per_thread = (_nBlocksIFm + col_teams-1)/col_teams;
      M_tasks_per_thread = (_nBlocksOFm + row_teams-1)/row_teams;
      my_N_start = N_hyperpartition_id * _nBlocksIFm + LIBXSMM_MIN( my_col_id * N_tasks_per_thread, _nBlocksIFm);
      my_N_end   = N_hyperpartition_id * _nBlocksIFm + LIBXSMM_MIN( (my_col_id+1) * N_tasks_per_thread, _nBlocksIFm);
      my_M_start = M_hyperpartition_id * _nBlocksOFm + LIBXSMM_MIN( my_row_id * M_tasks_per_thread, _nBlocksOFm);
      my_M_end   = M_hyperpartition_id * _nBlocksOFm + LIBXSMM_MIN( (my_row_id+1) * M_tasks_per_thread, _nBlocksOFm);
    }

    if (cfg.upd_2d_blocking == 0) {
      /* Required upfront tranposes */
      for (mb1ifm1 = tr_inp_thr_begin; mb1ifm1 < tr_inp_thr_end; mb1ifm1++) {
        mb1 = mb1ifm1%nBlocksMB;
        ifm1 = mb1ifm1/nBlocksMB;
        format_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(4, input, mb1, ifm1, 0, 0, nBlocksIFm, bn, bc);
        format_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, mb1, 0, 0, nBlocksMB, bc, bn);
        cfg.norm_to_normT_kernel( &format_param );
      }

      if (performed_doutput_transpose == 0) {
        for (mb1ofm1 = tr_out_thr_begin; mb1ofm1 < tr_out_thr_end; mb1ofm1++) {
          mb1 = mb1ofm1%nBlocksMB;
          ofm1 = mb1ofm1/nBlocksMB;
          format_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(4, doutput,  mb1, ofm1, 0, 0, nBlocksOFm, bn, bk);
          format_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, mb1, 0, 0, 0, nBlocksMB, bn_lp, bk, lpb);
          cfg.norm_to_vnni_kernel( &format_param );
        }
      }
      libxsmm_barrier_wait(cfg.barrier, ltid);
    }

    if (use_2d_blocking == 1) {
      LIBXSMM_VLA_DECL(4, libxsmm_bfloat16,  tmp_input_tr, ((libxsmm_bfloat16*)((char*)scratch + cfg.upd_private_tr_act_scratch_mark)) + ltid * bc * cfg.N * N_tasks_per_thread, nBlocksMB, bc, bn);
      LIBXSMM_VLA_DECL(4, libxsmm_bfloat16, tmp_doutput_tr, ((libxsmm_bfloat16*)((char*)scratch + cfg.upd_private_tr_dact_scratch_mark)) + ltid * bk * cfg.N, bn_lp, bk, lpb);
      ifm2 = 0;
      ofm2 = 0;
      if (BF == 1) {
        for (ofm1 = my_M_start; ofm1 < my_M_end; ++ofm1) {
          /* Transpose output block  */
          for (mb3 = 0; mb3 < nBlocksMB; mb3++) {
            format_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(4, doutput,  mb3, ofm1, 0, 0, nBlocksOFm, bn, bk);
            format_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(4, tmp_doutput_tr, mb3, 0, 0, 0, bn_lp, bk, lpb);
            cfg.norm_to_vnni_kernel( &format_param );
          }
          for (ifm1 = my_N_start; ifm1 < my_N_end; ++ifm1) {
            /* Transpose input block */
            if (ofm1 == my_M_start) {
              for (mb3 = 0; mb3 < nBlocksMB; mb3++) {
                format_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(4, input, mb3, ifm1, 0, 0, nBlocksIFm, bn, bc);
                format_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(4, tmp_input_tr, ifm1-my_N_start, mb3, 0, 0, nBlocksMB, bc, bn);
                cfg.norm_to_normT_kernel( &format_param );
              }
            }
            gemm_param.op.tertiary = (void*)&blocks;
            gemm_param.a.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, tmp_doutput_tr, 0, 0, 0, 0, bn_lp, bk, lpb);
            gemm_param.b.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, tmp_input_tr, ifm1-my_N_start, 0, 0, 0, nBlocksMB, bc, bn);
            gemm_param.c.primary   = (void*)&LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, 0, 0, 0, nBlocksIFm, bc_lp, bk, lpb);
            cfg.upd_compute_kernel2_strd( &gemm_param );
          }
        }
      } else {
        for (bfn = 0; bfn < BF; bfn++) {
          for (ofm1 = my_M_start; ofm1 < my_M_end; ++ofm1) {
            /* Transpose output block  */
            for (mb3 = bfn*(libxsmm_blasint)blocks; mb3 < (bfn+1)*(libxsmm_blasint)blocks; mb3++) {
              format_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(4, doutput,  mb3, ofm1, 0, 0, nBlocksOFm, bn, bk);
              format_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(4, tmp_doutput_tr, mb3, 0, 0, 0, bn_lp, bk, lpb);
              cfg.norm_to_vnni_kernel( &format_param );
            }
            for (ifm1 = my_N_start; ifm1 < my_N_end; ++ifm1) {
              /* Transpose input block */
              if (ofm1 == my_M_start) {
                for (mb3 = bfn*(libxsmm_blasint)blocks; mb3 < (bfn+1)*(libxsmm_blasint)blocks; mb3++) {
                  format_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(4, input, mb3, ifm1, 0, 0, nBlocksIFm, bn, bc);
                  format_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(4, tmp_input_tr, ifm1-my_N_start, mb3, 0, 0, nBlocksMB, bc, bn);
                  cfg.norm_to_normT_kernel( &format_param );
                }
              }
              /* initialize current work task to zero */
              if (bfn == 0) {
                copy_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk);
                cfg.upd_zero_kernel( &copy_param );
              }
              gemm_param.op.tertiary = (void*)&blocks;
              gemm_param.a.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, tmp_doutput_tr, bfn*blocks, 0, 0, 0, bn_lp, bk, lpb);
              gemm_param.b.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, tmp_input_tr, ifm1-my_N_start, bfn*blocks, ifm2*bbc, 0, nBlocksMB, bc, bn);
              gemm_param.c.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk);
              cfg.upd_compute_kernel_strd( &gemm_param );
              /* Downconvert result to BF16 and vnni format */
              if (bfn == BF-1) {
                LIBXSMM_ALIGNED(libxsmm_bfloat16 tmp_buf[bc][bk], 64);
                copy_param.in.primary    = (void*)&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk);
                copy_param.out.primary   = (void*)tmp_buf;
                format_param.in.primary  = (void*)tmp_buf;
                format_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, 0, 0, 0, nBlocksIFm, bc_lp, bk, lpb);
                cfg.upd_store_kernel( &copy_param );
                cfg.norm_to_vnni_kernel_wt( &format_param );
              }
            }
          }
        }
      }
    } else {
      if (BF == 1) {
        for ( ifm1ofm1 = thr_begin; ifm1ofm1 < thr_end; ++ifm1ofm1 ) {
          ofm1 = ifm1ofm1 / Cck_work;
          ofm2 = (ifm1ofm1 % Cck_work) / Cc_work;
          ifm1 = ((ifm1ofm1 % Cck_work) % Cc_work) / ifm_subtasks;
          ifm2 = ((ifm1ofm1 % Cck_work) % Cc_work) % ifm_subtasks;
          gemm_param.op.tertiary = (void*)&blocks;
          gemm_param.a.primary   = (void*)&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, 0, 0, ofm2*bbk, 0, nBlocksMB, bn_lp, bk, lpb);
          gemm_param.b.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, 0, ifm2*bbc, 0, nBlocksMB, bc, bn);
          gemm_param.c.primary   = (void*) &LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc)/lpb, ofm2*bbk, 0, nBlocksIFm, bc_lp, bk, lpb);
          cfg.upd_compute_kernel2_strd( &gemm_param );
        }
      } else {
        for (bfn = 0; bfn < BF; bfn++) {
          for ( ifm1ofm1 = thr_begin; ifm1ofm1 < thr_end; ++ifm1ofm1 ) {
            ofm1 = ifm1ofm1 / Cck_work;
            ofm2 = (ifm1ofm1 % Cck_work) / Cc_work;
            ifm1 = ((ifm1ofm1 % Cck_work) % Cc_work) / ifm_subtasks;
            ifm2 = ((ifm1ofm1 % Cck_work) % Cc_work) % ifm_subtasks;
            /* initialize current work task to zero */
            if (bfn == 0) {
              copy_param.out.primary = &LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk);
              cfg.upd_zero_kernel( &copy_param );
            }
            gemm_param.op.tertiary = (void*)&blocks;
            gemm_param.a.primary   = (void*)&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, bfn*blocks, 0, ofm2*bbk, 0, nBlocksMB, bn_lp, bk, lpb);
            gemm_param.b.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, bfn*blocks, ifm2*bbc, 0, nBlocksMB, bc, bn);
            gemm_param.c.primary   = (void*) &LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk);
            cfg.upd_compute_kernel_strd( &gemm_param );
            /* Downconvert result to BF16 and vnni format */
            if (bfn == BF-1) {
              LIBXSMM_ALIGNED(libxsmm_bfloat16 tmp_buf[bc][bk], 64);
              copy_param.in.primary   = (void*)&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk);
              copy_param.out.primary  = (void*)tmp_buf;
              format_param.in.primary  = (void*)tmp_buf;
              format_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc)/lpb, ofm2*bbk, 0, nBlocksIFm, bc_lp, bk, lpb);
              cfg.upd_store_kernel( &copy_param );
              cfg.norm_to_vnni_kernel_wt( &format_param );
            }
          }
        }
      }
    }
    cfg.upd_tilerelease_kernel( NULL );
    libxsmm_barrier_wait(cfg.barrier, ltid);
  }
}

LIBXSMM_API void libxsmm_dnn_fc_bwd_exec_bf16( libxsmm_dnn_fc_bwd_config cfg,  const libxsmm_bfloat16* wt_ptr, libxsmm_bfloat16* din_act_ptr,
                          const libxsmm_bfloat16* dout_act_ptr, libxsmm_bfloat16* dwt_ptr, const libxsmm_bfloat16* in_act_ptr,
                          libxsmm_bfloat16* dbias_ptr, const unsigned char* relu_ptr, libxsmm_dnn_fc_pass pass, int start_tid, int my_tid, void* scratch )
{
  libxsmm_dnn_fc_bwd_exec_bf16_vnni_format( cfg, wt_ptr, din_act_ptr, dout_act_ptr, dwt_ptr, in_act_ptr,
                                   dbias_ptr, relu_ptr, pass, start_tid, my_tid, scratch );
}

LIBXSMM_API void destroy_libxsmm_dnn_fc_fwd(libxsmm_dnn_fc_fwd_config* cfg) {
  libxsmm_barrier_destroy(cfg->barrier);
}

LIBXSMM_API void destroy_libxsmm_dnn_fc_bwd(libxsmm_dnn_fc_bwd_config* cfg) {
  libxsmm_barrier_destroy(cfg->barrier);
}
