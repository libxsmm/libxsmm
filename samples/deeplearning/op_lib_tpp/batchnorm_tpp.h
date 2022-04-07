/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Kirill Voronin (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <libxsmm_sync.h>
#include <libxsmm_intrinsics_x86.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

#define BITS_PER_CHAR (8)

typedef enum my_bn_fuse {
  MY_BN_FUSE_NONE = 0,
  MY_BN_FUSE_RELU = 1,
  MY_BN_FUSE_ELTWISE = 2,
  MY_BN_FUSE_ELTWISE_RELU = 3,
  MY_BN_FUSE_RELU_WITH_MASK = 4,
  MY_BN_FUSE_ELTWISE_RELU_WITH_MASK = 5
} my_bn_fuse;

typedef enum my_bn_norm_type {
  MY_BN_FULL_NORM  = 0, /* stats + normalize for fwd, all grads for bwd */
  MY_BN_SCALE_ONLY = 1  /* normalize only for fwd, only input grad for bwd */
} my_bn_norm_type;

typedef struct my_bn_fwd_config {
  libxsmm_blasint  N;
  libxsmm_blasint  C;
  libxsmm_blasint  H;
  libxsmm_blasint  W;
  libxsmm_blasint  bc;
  libxsmm_blasint  CP;
  libxsmm_blasint  num_HW_blocks;
  libxsmm_blasint  threads;
  size_t           scratch_size;

  libxsmm_datatype datatype_in;
  libxsmm_datatype datatype_out;
  libxsmm_datatype datatype_comp;

  libxsmm_barrier* barrier;

  libxsmm_matrix_eqn_function  func10;
  libxsmm_meltwfunction_unary  reduce_HW_kernel;
  libxsmm_meltwfunction_unary  all_zero_kernel;
  libxsmm_meltwfunction_binary helper_add_kernel;
  my_bn_fuse        fuse_type;
} my_bn_fwd_config;

typedef struct my_bn_bwd_config {
  libxsmm_blasint  N;
  libxsmm_blasint  C;
  libxsmm_blasint  H;
  libxsmm_blasint  W;
  libxsmm_blasint  bc;
  libxsmm_blasint  CP;
  libxsmm_blasint  num_HW_blocks;
  libxsmm_blasint  threads;
  size_t           scratch_size;

  libxsmm_datatype datatype_in;
  libxsmm_datatype datatype_out;
  libxsmm_datatype datatype_comp;

  libxsmm_barrier* barrier;

  libxsmm_matrix_eqn_function  dgamma_func;
  libxsmm_matrix_eqn_function  dbeta_func;
  libxsmm_matrix_eqn_function  din_func;
  libxsmm_meltwfunction_unary  all_zero_kernel;
  libxsmm_meltwfunction_binary helper_add_kernel;
  libxsmm_meltwfunction_unary  helper_copy_kernel;
  libxsmm_meltwfunction_unary  inv_relu_kernel;
  libxsmm_meltwfunction_unary  ewise_copy_kernel;
  my_bn_fuse        fuse_type;
} my_bn_bwd_config;

my_bn_fwd_config setup_my_bn_fwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint bc,
                                 libxsmm_blasint threads, my_bn_fuse fuse_type,
                                 libxsmm_datatype datatype_in, libxsmm_datatype datatype_out, libxsmm_datatype datatype_comp ) {
  my_bn_fwd_config res;

  size_t sum_N_offset, sumsq_N_offset;

  libxsmm_meltw_unary_shape  unary_shape;
  libxsmm_meltw_binary_shape binary_shape;

  libxsmm_bitfield unary_flags;
  libxsmm_bitfield binary_flags;
  libxsmm_bitfield ternary_flags;

  libxsmm_blasint ldo = bc;
  libxsmm_blasint ld  = bc;
  libxsmm_blasint tmp_ld, tmp_ld2;
  libxsmm_blasint my_eqn10;

  libxsmm_meqn_arg_shape  eqn_out_arg_shape;
  libxsmm_meqn_arg_shape  arg_shape[128];

  libxsmm_matrix_arg_attributes arg_singular_attr;

  libxsmm_matrix_eqn_arg_metadata arg_metadata[128];
  libxsmm_matrix_eqn_op_metadata  op_metadata[128];

  arg_singular_attr.type = LIBXSMM_MATRIX_ARG_TYPE_SINGULAR;

  memset( &res,  0, sizeof(res));

  /* setting up some handle values */
  res.N  = N;
  res.C  = C;
  res.H  = H;
  res.W  = W;
  res.bc = bc;
  res.CP = res.C / res.bc;
  res.num_HW_blocks = (res.H > res.W ? res.H : res.W );
  res.threads       = threads;
  res.fuse_type     = fuse_type;

  /* when masking is on, bc must be divisible by 8 for compressing mask into char array (otherwise strides are wrong for relumask */
  if ( (res.fuse_type == 4 || res.fuse_type == 5) && (res.bc % BITS_PER_CHAR != 0)) {
    fprintf( stderr, "bc = %d is not divisible by BITS_PER_CHAR = %d. Bailing...!\n", res.bc, BITS_PER_CHAR);
    exit(-1);
  }

  res.datatype_in   = datatype_in;
  res.datatype_out  = datatype_out;
  res.datatype_comp = datatype_comp;

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* TPP creation */

  memset( &unary_shape,  0, sizeof(libxsmm_meltw_unary_shape));
  memset( &binary_shape, 0, sizeof(libxsmm_meltw_binary_shape));

  /* Eltwise TPPs  */

  unary_flags         = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  unary_shape         = libxsmm_create_meltw_unary_shape(res.bc, 1, res.bc, ldo, res.datatype_comp, res.datatype_comp, res.datatype_comp);
  res.all_zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
  if ( res.all_zero_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd all_zero_kernel failed. Bailing...!\n");
    exit(-1);
  }

  binary_shape          = libxsmm_create_meltw_binary_shape(res.bc, 1, ldo, ldo, ldo, res.datatype_comp, res.datatype_comp, res.datatype_comp, res.datatype_comp);
  binary_flags          = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  res.helper_add_kernel = libxsmm_dispatch_meltw_binary_v2(LIBXSMM_MELTW_TYPE_BINARY_ADD, binary_shape, binary_flags);
  if ( res.helper_add_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd helper_add_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* TPPs for reducing X and X2 in HW*/
  tmp_ld = bc;

  unary_shape          = libxsmm_create_meltw_unary_shape(res.bc, res.H*res.W / res.num_HW_blocks, ld, tmp_ld, res.datatype_in, res.datatype_comp, res.datatype_comp);
  unary_flags          = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  res.reduce_HW_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD, unary_shape, unary_flags);
  if ( res.reduce_HW_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd reduce_HW_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* TPP for scaling */
  ld      = bc;
  tmp_ld  = 1;
  tmp_ld2 = 1;

  my_eqn10 = libxsmm_matrix_eqn_create();                          /* y = relu ( ( (s*x + b)*gamma + beta ) + inp_add) */

  if (res.fuse_type == 1 || res.fuse_type == 3 || res.fuse_type == 4 || res.fuse_type == 5) {
    unary_flags                 = ( (res.fuse_type == 4 || res.fuse_type == 5) ? LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT : LIBXSMM_MELTW_FLAG_UNARY_NONE);
    op_metadata[3].eqn_idx      = my_eqn10;
    op_metadata[3].op_arg_pos   = -1;

    libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[3], LIBXSMM_MELTW_TYPE_UNARY_RELU, res.datatype_out, unary_flags);

    if (res.datatype_out == LIBXSMM_DATATYPE_BF16)
      libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[3], LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, res.datatype_out, LIBXSMM_MELTW_FLAG_UNARY_NONE);
  }

  if (res.fuse_type == 2 || res.fuse_type == 3 || res.fuse_type == 5) {
    binary_flags                = LIBXSMM_MELTW_FLAG_BINARY_NONE;
    op_metadata[2].eqn_idx      = my_eqn10;
    op_metadata[2].op_arg_pos   = -1;
    libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[2], LIBXSMM_MELTW_TYPE_BINARY_ADD, res.datatype_comp, binary_flags);
  }

  ternary_flags               = LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT;
  op_metadata[0].eqn_idx      = my_eqn10;
  op_metadata[0].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_ternary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_TERNARY_MULADD, res.datatype_comp, ternary_flags);

  arg_metadata[3].eqn_idx     = my_eqn10;
  arg_metadata[3].in_arg_pos  = 3;
  arg_shape[3].m    = res.bc;                                      /* gamma = [bc] */
  arg_shape[3].n    = 1;
  arg_shape[3].ld   = tmp_ld2;
  arg_shape[3].type = res.datatype_comp;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[3], arg_shape[3], arg_singular_attr);

  ternary_flags               = LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT;
  op_metadata[1].eqn_idx      = my_eqn10;
  op_metadata[1].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_ternary_op_v2(op_metadata[1], LIBXSMM_MELTW_TYPE_TERNARY_MULADD, res.datatype_comp, ternary_flags);

  arg_metadata[1].eqn_idx     = my_eqn10;
  arg_metadata[1].in_arg_pos  = 1;
  arg_shape[1].m    = res.bc;                                      /* s = [bc] */
  arg_shape[1].n    = 1;
  arg_shape[1].ld   = tmp_ld;
  arg_shape[1].type = res.datatype_comp;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[1], arg_shape[1], arg_singular_attr);

  arg_metadata[0].eqn_idx     = my_eqn10;
  arg_metadata[0].in_arg_pos  = 0;
  arg_shape[0].m    = res.bc;                                      /* x = [HW, bc] */
  arg_shape[0].n    = res.H*res.W /res.num_HW_blocks;
  arg_shape[0].ld   = ld;
  arg_shape[0].type = res.datatype_in;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[0], arg_shape[0], arg_singular_attr);

  arg_metadata[2].eqn_idx     = my_eqn10;
  arg_metadata[2].in_arg_pos  = 2;
  arg_shape[2].m    = res.bc;                                      /* b = [bc] */
  arg_shape[2].n    = 1;
  arg_shape[2].ld   = tmp_ld;
  arg_shape[2].type = res.datatype_comp;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[2], arg_shape[2], arg_singular_attr);

  arg_metadata[4].eqn_idx     = my_eqn10;
  arg_metadata[4].in_arg_pos  = 4;
  arg_shape[4].m    = res.bc;                                      /* beta = [bc] */
  arg_shape[4].n    = 1;
  arg_shape[4].ld   = tmp_ld2;
  arg_shape[4].type = res.datatype_comp;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[4], arg_shape[4], arg_singular_attr);

  if (res.fuse_type == 2 || res.fuse_type == 3 || res.fuse_type == 5) {
    arg_metadata[5].eqn_idx     = my_eqn10;
    arg_metadata[5].in_arg_pos  = 5;
    arg_shape[5].m    = res.bc;                                      /* inp_add = [HW, bc] */
    arg_shape[5].n    = res.H*res.W / res.num_HW_blocks;
    arg_shape[5].ld   = ld;
    arg_shape[5].type = res.datatype_in;
    libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[5], arg_shape[5], arg_singular_attr);
  }

  eqn_out_arg_shape.m    = res.bc;                                 /* y = [HW, bc] */
  eqn_out_arg_shape.n    = res.H*res.W / res.num_HW_blocks;
  eqn_out_arg_shape.ld   = ld;
  eqn_out_arg_shape.type = res.datatype_out;

  /* libxsmm_matrix_eqn_tree_print( my_eqn10 ); */
  /* libxsmm_matrix_eqn_rpn_print ( my_eqn10 ); */
  res.func10 = libxsmm_dispatch_matrix_eqn_v2( my_eqn10, eqn_out_arg_shape );
  if ( res.func10 == NULL) {
    fprintf( stderr, "JIT for TPP fwd func10 (eqn10) failed. Bailing...!\n");
    exit(-1);
  }

  /* init scratch */
  sum_N_offset   = LIBXSMM_UP2(res.CP * 2 * res.bc, 64);
  sumsq_N_offset = LIBXSMM_UP2(sum_N_offset + res.CP * res.N * res.bc, 64);
  res.scratch_size =  sizeof(float) * ( sumsq_N_offset /*sum_X_X2 + sumsq_N */ + LIBXSMM_UP2((size_t)res.CP * (size_t)res.N * (size_t)res.bc, 64) /* sumsq_N */ );

  return res;
}

my_bn_bwd_config setup_my_bn_bwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint bc,
                                 libxsmm_blasint threads, my_bn_fuse fuse_type,
                                 libxsmm_datatype datatype_in, libxsmm_datatype datatype_out, libxsmm_datatype datatype_comp ) {
  my_bn_bwd_config res;

  libxsmm_meltw_unary_shape  unary_shape;
  libxsmm_meltw_binary_shape binary_shape;

  libxsmm_bitfield unary_flags;
  libxsmm_bitfield binary_flags;
  libxsmm_bitfield ternary_flags;

  libxsmm_meqn_arg_shape  eqn_out_arg_shape;
  libxsmm_meqn_arg_shape  arg_shape[128];

  libxsmm_matrix_arg_attributes arg_singular_attr;

  libxsmm_matrix_eqn_arg_metadata arg_metadata[128];
  libxsmm_matrix_eqn_op_metadata  op_metadata[128];

  arg_singular_attr.type = LIBXSMM_MATRIX_ARG_TYPE_SINGULAR;

  size_t dbeta_N_offset;

  libxsmm_blasint ldo = bc;
  libxsmm_blasint ld  = bc;
  libxsmm_blasint tmp_ld2;
  libxsmm_blasint my_eqn11, my_eqn12, my_eqn16;

  memset( &res,  0, sizeof(res));

  /* setting up some handle values */
  res.N             = N;
  res.C             = C;
  res.H             = H;
  res.W             = W;
  res.bc            = bc;
  res.CP            = res.C / res.bc;
  res.num_HW_blocks = (res.H > res.W ? res.H : res.W );
  res.threads       = threads;
  res.fuse_type     = fuse_type;

  /* when masking is on, bc must be divisible by 8 for compressing mask into char array (otherwise strides are wrong for relumask */
  if ( (res.fuse_type == 4 || res.fuse_type == 5) && (res.bc % BITS_PER_CHAR != 0)) {
    fprintf( stderr, "bc = %d is not divisible by BITS_PER_CHAR = %d. Bailing...!\n", res.bc, BITS_PER_CHAR);
    exit(-1);
  }

  res.datatype_in   = datatype_in;
  res.datatype_out  = datatype_out;
  res.datatype_comp = datatype_comp;

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* TPP creation */

  memset( &unary_shape,  0, sizeof(libxsmm_meltw_unary_shape));
  memset( &binary_shape, 0, sizeof(libxsmm_meltw_binary_shape));

  /* Eltwise TPPs  */
  unary_shape         = libxsmm_create_meltw_unary_shape(res.bc, 1, res.bc, ldo, res.datatype_comp, res.datatype_comp, res.datatype_comp);
  unary_flags         = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.all_zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
  if ( res.all_zero_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd all_zero_kernel failed. Bailing...!\n");
    exit(-1);
  }

  unary_shape            = libxsmm_create_meltw_unary_shape(res.bc, 1, ldo, ldo, res.datatype_comp, res.datatype_comp, res.datatype_comp);
  unary_flags            = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.helper_copy_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, unary_flags);
  if ( res.helper_copy_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd helper_copy_kernel failed. Bailing...!\n");
    exit(-1);
  }

  binary_shape          = libxsmm_create_meltw_binary_shape(res.bc, 1, ldo, ldo, ldo, res.datatype_comp, res.datatype_comp, res.datatype_comp, res.datatype_comp);
  binary_flags          = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  res.helper_add_kernel = libxsmm_dispatch_meltw_binary_v2(LIBXSMM_MELTW_TYPE_BINARY_ADD, binary_shape, binary_flags);
  if ( res.helper_add_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd helper_add_kernel failed. Bailing...!\n");
    exit(-1);
  }

  if (res.fuse_type == 1 || res.fuse_type == 3 || res.fuse_type == 4 || res.fuse_type == 5) {
    unary_shape         = libxsmm_create_meltw_unary_shape(res.bc, res.H*res.W / res.num_HW_blocks, ldo, ldo, res.datatype_in, res.datatype_out, res.datatype_comp);
    unary_flags         = ( (res.fuse_type == 4 || res.fuse_type == 5) ? LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT : LIBXSMM_MELTW_FLAG_UNARY_NONE);
    res.inv_relu_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_RELU_INV, unary_shape, unary_flags);
    if ( res.inv_relu_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP bwd inv_relu_kernel failed. Bailing...!\n");
      exit(-1);
    }
  }

  if (res.fuse_type == 2 || res.fuse_type == 3 || res.fuse_type == 5) {
    unary_shape           = libxsmm_create_meltw_unary_shape(res.bc, res.H*res.W / res.num_HW_blocks, ldo, ldo, res.datatype_in, res.datatype_out, res.datatype_comp);
    unary_flags           = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    res.ewise_copy_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, unary_flags);
    if ( res.ewise_copy_kernel == NULL) {
      fprintf( stderr, "JIT for TPP bwd ewise_copy_kernel failed. Bailing...!\n");
      exit(-1);
    }
  }

  /* TPP equations for dgamma, dbeta and din */

  ld = bc;
  tmp_ld2 = 1;

  /* dgamma function  */
  my_eqn11 = libxsmm_matrix_eqn_create();                          /* dgamma = ((inp *a + b) * dout) + dgamma */

  binary_flags                = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  op_metadata[0].eqn_idx      = my_eqn11;
  op_metadata[0].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_BINARY_ADD, res.datatype_comp, binary_flags);

  unary_flags                 = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  op_metadata[1].eqn_idx      = my_eqn11;
  op_metadata[1].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[1], LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, res.datatype_comp, unary_flags);

  binary_flags                = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  op_metadata[2].eqn_idx      = my_eqn11;
  op_metadata[2].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[2], LIBXSMM_MELTW_TYPE_BINARY_MUL, res.datatype_comp, binary_flags);

  ternary_flags               = LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT;
  op_metadata[3].eqn_idx      = my_eqn11;
  op_metadata[3].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_ternary_op_v2(op_metadata[3], LIBXSMM_MELTW_TYPE_TERNARY_MULADD, res.datatype_comp, ternary_flags);

  arg_metadata[0].eqn_idx     = my_eqn11;
  arg_metadata[0].in_arg_pos  = 0;
  arg_shape[0].m    = res.bc;                                      /* inp [HW, bc] */
  arg_shape[0].n    = res.H*res.W /res.num_HW_blocks;
  arg_shape[0].ld   = ld;
  arg_shape[0].type = res.datatype_in;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[0], arg_shape[0], arg_singular_attr);

  arg_metadata[1].eqn_idx     = my_eqn11;
  arg_metadata[1].in_arg_pos  = 1;
  arg_shape[1].m    = res.bc;                                      /* a [bc] */
  arg_shape[1].n    = 1;
  arg_shape[1].ld   = tmp_ld2;
  arg_shape[1].type = res.datatype_comp;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[1], arg_shape[1], arg_singular_attr);

  arg_metadata[2].eqn_idx     = my_eqn11;
  arg_metadata[2].in_arg_pos  = 2;
  arg_shape[2].m    = res.bc;                                      /* b [bc] */
  arg_shape[2].n    = 1;
  arg_shape[2].ld   = tmp_ld2;
  arg_shape[2].type = res.datatype_comp;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[2], arg_shape[2], arg_singular_attr);

  arg_metadata[3].eqn_idx     = my_eqn11;
  arg_metadata[3].in_arg_pos  = 3;
  arg_shape[3].m    = res.bc;                                      /* dout [HW, bc] */
  arg_shape[3].n    = res.H*res.W/res.num_HW_blocks;
  arg_shape[3].ld   = ld;
  arg_shape[3].type = res.datatype_out;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[3], arg_shape[3], arg_singular_attr);

  arg_metadata[4].eqn_idx     = my_eqn11;
  arg_metadata[4].in_arg_pos  = 4;
  arg_shape[4].m    = res.bc;                                      /* dgamma [bc] */
  arg_shape[4].n    = 1;
  arg_shape[4].ld   = tmp_ld2;
  arg_shape[4].type = res.datatype_comp;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[4], arg_shape[4], arg_singular_attr);

  eqn_out_arg_shape.m    = res.bc;                                 /* dgamma [bc] */
  eqn_out_arg_shape.n    = 1;
  eqn_out_arg_shape.ld   = tmp_ld2;
  eqn_out_arg_shape.type = res.datatype_comp;

  /* libxsmm_matrix_eqn_tree_print( my_eqn11 ); */
  /* libxsmm_matrix_eqn_rpn_print ( my_eqn11 ); */
  res.dgamma_func = libxsmm_dispatch_matrix_eqn_v2( my_eqn11, eqn_out_arg_shape );
  if ( res.dgamma_func == NULL) {
    fprintf( stderr, "JIT for TPP bwd dgamma_func (eqn11) failed. Bailing...!\n");
    exit(-1);
  }

  /* dbeta function  */
  my_eqn12 = libxsmm_matrix_eqn_create();                         /* dbeta [bc] = dout [HW, bc] + dbeta [bc] */

  binary_flags                = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  op_metadata[0].eqn_idx      = my_eqn12;
  op_metadata[0].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_BINARY_ADD, res.datatype_comp, binary_flags); /* dbeta_tmp [HW, bc] */

  unary_flags                 = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  op_metadata[1].eqn_idx      = my_eqn12;
  op_metadata[1].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[1], LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, res.datatype_comp, unary_flags); /* [HW, bc] -> [bc] */

  arg_metadata[0].eqn_idx     = my_eqn12;
  arg_metadata[0].in_arg_pos  = 3;
  arg_shape[0].m    = res.bc;                                      /* dout [HW, bc] */
  arg_shape[0].n    = res.H*res.W /res.num_HW_blocks;
  arg_shape[0].ld   = ld;
  arg_shape[0].type = res.datatype_out;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[0], arg_shape[0], arg_singular_attr);

  arg_metadata[1].eqn_idx     = my_eqn12;
  arg_metadata[1].in_arg_pos  = 5;
  arg_shape[1].m    = res.bc;                                      /* dbeta [bc] */
  arg_shape[1].n    = 1;
  arg_shape[1].ld   = tmp_ld2;
  arg_shape[1].type = res.datatype_comp;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[1], arg_shape[1], arg_singular_attr);

  eqn_out_arg_shape.m    = res.bc;                                 /* dbeta [bc] */
  eqn_out_arg_shape.n    = 1;
  eqn_out_arg_shape.ld   = tmp_ld2;
  eqn_out_arg_shape.type = res.datatype_comp;

  /* libxsmm_matrix_eqn_tree_print( my_eqn12 ); */
  /* libxsmm_matrix_eqn_rpn_print ( my_eqn12 ); */

  res.dbeta_func = libxsmm_dispatch_matrix_eqn_v2( my_eqn12, eqn_out_arg_shape );
  if ( res.dbeta_func == NULL) {
    fprintf( stderr, "JIT for TPP bwd dbeta_func (eqn12) failed. Bailing...!\n");
    exit(-1);
  }

  /* din = gamma_ptr[v] * brstd_ptr[v] * recp_nhw * (nhw*del_output_ptr[v] - (del_beta_ptr[v] + (input_ptr[v] - bmean_ptr[v]) * del_gamma_ptr[v] * brstd_ptr[v])) */
  /* din = gamma_ptr[v] * brstd_ptr[v] *del_output_ptr[v] - gamma_ptr[v] * brstd_ptr[v] * recp_nhw * (del_beta_ptr[v] + (input_ptr[v] - bmean_ptr[v]) * del_gamma_ptr[v] * brstd_ptr[v])) */
  /* din = gamma_ptr[v] * brstd_ptr[v] *del_output_ptr[v] - gamma_ptr[v] * brstd_ptr[v] * recp_nhw * del_beta_ptr[v] + gamma_ptr[v] * brstd_ptr[v] * recp_nhw * (input_ptr[v] - bmean_ptr[v]) * del_gamma_ptr[v] * brstd_ptr[v]) */
  /* din = a * del_output_ptr[v] + b * input_ptr[v] + c */
  /* a = gamma_ptr[bc] * brstd_ptr[bc] */
  /* b = gamma_ptr[bc] *  del_gamma_ptr[v] * brstd_ptr[bc] * brstd_ptr[bc] * recp_nhw */
  /* c = -gamma_ptr[bc] * brstd_ptr[bc] * recp_nhw * del_beta_ptr[bc] + gamma_ptr[bc] * brstd_ptr[bc] * recp_nhw * bmean_ptr[bc] * del_gamma_ptr[bc] * brstd_ptr[bc]) */

  /* din long equation */
  my_eqn16 = libxsmm_matrix_eqn_create();                          /* din = a * dout + (b * inp + c) */

  ternary_flags               = LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT;
  op_metadata[0].eqn_idx      = my_eqn16;
  op_metadata[0].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_ternary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_TERNARY_MULADD, res.datatype_comp, ternary_flags);

  arg_metadata[0].eqn_idx     = my_eqn16;
  arg_metadata[0].in_arg_pos  = 1;
  arg_shape[0].m    = res.bc;                                      /* a [bc] */
  arg_shape[0].n    = 1;
  arg_shape[0].ld   = tmp_ld2;
  arg_shape[0].type = res.datatype_comp;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[0], arg_shape[0], arg_singular_attr);

  arg_metadata[1].eqn_idx     = my_eqn16;
  arg_metadata[1].in_arg_pos  = 3;
  arg_shape[1].m    = res.bc;                                      /* dout [HW, bc] */
  arg_shape[1].n    = res.H*res.W /res.num_HW_blocks;
  arg_shape[1].ld   = ld;
  arg_shape[1].type = res.datatype_out;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[1], arg_shape[1], arg_singular_attr);

  ternary_flags               = LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT;
  op_metadata[1].eqn_idx      = my_eqn16;
  op_metadata[1].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_ternary_op_v2(op_metadata[1], LIBXSMM_MELTW_TYPE_TERNARY_MULADD, res.datatype_comp, ternary_flags);

  arg_metadata[2].eqn_idx     = my_eqn16;
  arg_metadata[2].in_arg_pos  = 0;
  arg_shape[2].m    = res.bc;                                      /* inp [HW, bc] */
  arg_shape[2].n    = res.H*res.W /res.num_HW_blocks;
  arg_shape[2].ld   = ld;
  arg_shape[2].type = res.datatype_in;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[2], arg_shape[2], arg_singular_attr);

  arg_metadata[3].eqn_idx     = my_eqn16;
  arg_metadata[3].in_arg_pos  = 2;
  arg_shape[3].m    = res.bc;                                      /* b [bc] */
  arg_shape[3].n    = 1;
  arg_shape[3].ld   = tmp_ld2;
  arg_shape[3].type = res.datatype_comp;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[3], arg_shape[3], arg_singular_attr);

  arg_metadata[4].eqn_idx     = my_eqn16;
  arg_metadata[4].in_arg_pos  = 7;
  arg_shape[4].m    = res.bc;                                      /* c [bc] */
  arg_shape[4].n    = 1;
  arg_shape[4].ld   = tmp_ld2;
  arg_shape[4].type = res.datatype_comp;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[4], arg_shape[4], arg_singular_attr);

  eqn_out_arg_shape.m    = res.bc;                                 /* din [HW, bc] */
  eqn_out_arg_shape.n    = res.H*res.W/res.num_HW_blocks;
  eqn_out_arg_shape.ld   = ld;
  eqn_out_arg_shape.type = res.datatype_out;

  /* libxsmm_matrix_eqn_tree_print( my_eqn16 ); */
  /* libxsmm_matrix_eqn_rpn_print ( my_eqn16 ); */

  res.din_func = libxsmm_dispatch_matrix_eqn_v2( my_eqn16, eqn_out_arg_shape );
  if ( res.din_func == NULL) {
    fprintf( stderr, "JIT for TPP bwd din_func (eqn16) failed. Bailing...!\n");
    exit(-1);
  }

  /* init scratch */
  dbeta_N_offset = LIBXSMM_UP2(res.CP * res.N * res.bc, 64);
  res.scratch_size =  sizeof(float) * ( dbeta_N_offset /* dbeta_N*/ + LIBXSMM_UP2(res.CP * res.N * res.bc, 64) /*dgamma_N */ );

  return res;
}

void destroy_my_bn_fwd(my_bn_fwd_config* cfg) {
  libxsmm_barrier_destroy(cfg->barrier);

  /* when/if libxsmm_matrix_eqn_destroy gets added, destructords for equations should go here */
}

void destroy_my_bn_bwd(my_bn_bwd_config* cfg) {
  libxsmm_barrier_destroy(cfg->barrier);

  /* when/if libxsmm_matrix_eqn_destroy gets added, destructords for equations should go here */
}

void my_bn_fwd_exec_f32( my_bn_fwd_config cfg, const float *pinp, const float *pinp_add, const float *pgamma, const float *pbeta, float *mean, float *var, float *pout,
                         unsigned char *prelumask, float eps, int start_tid, int my_tid, void *scratch, my_bn_norm_type norm_type ) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint HW = cfg.H * cfg.W;
  const libxsmm_blasint bc = cfg.bc;
  const libxsmm_blasint num_HW_blocks = cfg.num_HW_blocks;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;

  /* number of tasks that could be run in parallel for 1d blocking */
  /* Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here? */
  const libxsmm_blasint work_dN = CP * N;
  /* compute chunk size */
  const libxsmm_blasint chunksize_dN = (work_dN % cfg.threads == 0) ?
    (work_dN / cfg.threads) : ((work_dN / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_dN = (ltid * chunksize_dN < work_dN) ? (ltid * chunksize_dN) : work_dN;
  const libxsmm_blasint thr_end_dN = ((ltid + 1) * chunksize_dN < work_dN) ? ((ltid + 1) * chunksize_dN) : work_dN;

  /* number of tasks that could be run in parallel for 1d blocking */
  /* Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here? */
  const libxsmm_blasint work_C = CP;
  /* compute chunk size */
  const libxsmm_blasint chunksize_C = (work_C % cfg.threads == 0) ?
    (work_C / cfg.threads) : ((work_C / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_C = (ltid * chunksize_C < work_C) ? (ltid * chunksize_C) : work_C;
  const libxsmm_blasint thr_end_C = ((ltid + 1) * chunksize_C < work_C) ? ((ltid + 1) * chunksize_C) : work_C;

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  LIBXSMM_VLA_DECL(4, const float,         inp,      pinp, CP, HW, bc);            /* [N, CP, HW, bc] */
  LIBXSMM_VLA_DECL(4,       float,         out,      pout, CP, HW, bc);            /* [N, CP, HW, bc] */
  LIBXSMM_VLA_DECL(2, const float,         gamma,    pgamma, bc);                  /* [CP, bc] */
  LIBXSMM_VLA_DECL(2, const float,         beta,     pbeta, bc);                   /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float,         mean,     mean,  bc);                   /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float,         var,      var,   bc);                   /* [CP, bc] */

  LIBXSMM_VLA_DECL(4, const float,         inp_add,  pinp_add, CP, HW, bc);        /* [N, CP, HW, bc] */

  float alpha = 0.0f;
  LIBXSMM_VLA_DECL(4,       unsigned char, relumask, prelumask, CP, HW, bc/BITS_PER_CHAR);    /* [N, CP, HW, bc/BITS_PER_CHAR] */

  const float scale = 1.0f /((float)N * HW);

  LIBXSMM_VLA_DECL(3, float, sum_X_X2, ((float*)scratch), CP, bc);  /* [2, CP, bc] */
  LIBXSMM_ASSUME_ALIGNED(sum_X_X2_, 64);
  const libxsmm_blasint sum_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + CP * 2 * bc), 64) - ((uintptr_t)(scratch))) / sizeof(float);
  LIBXSMM_VLA_DECL(3, float, sum_N, ((float*)scratch) + sum_N_offset, N, bc);  /* [CP, N, bc] */
  LIBXSMM_ASSUME_ALIGNED(sum_N_, 64);
  const libxsmm_blasint sumsq_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + sum_N_offset + CP * N * bc), 64) - ((uintptr_t)(scratch))) / sizeof(float);
  LIBXSMM_VLA_DECL(3, float, sumsq_N, ((float*)scratch) + sumsq_N_offset, N, bc);  /* [CP, N, bc] */
  LIBXSMM_ASSUME_ALIGNED(sumsq_N_, 64);

  libxsmm_meltw_unary_param  all_zero_param;
  libxsmm_meltw_binary_param add_param;
  libxsmm_meltw_unary_param  reduce_HW_param;
  libxsmm_meltw_unary_param  all_relu_param;

  libxsmm_matrix_arg arg_array[6];

  libxsmm_matrix_eqn_param eqn_param;

  memset( &all_zero_param,  0, sizeof(all_zero_param));
  memset( &add_param,       0, sizeof(add_param));
  memset( &reduce_HW_param, 0, sizeof(reduce_HW_param));
  memset( &all_relu_param,  0, sizeof(all_relu_param));

  memset( &eqn_param,       0, sizeof(eqn_param));

  LIBXSMM_ALIGNED(float s[bc], 64);
  LIBXSMM_ALIGNED(float b[bc], 64);
  int n, cp;

  int cpxnt;
  if (norm_type == MY_BN_FULL_NORM) {

    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      n  = cpxnt%N;
      cp = cpxnt/N;

      int hwb;

      float *sum_ncp_ptr   = &LIBXSMM_VLA_ACCESS(3, sum_N,   cp, n, 0, N, bc);
      float *sumsq_ncp_ptr = &LIBXSMM_VLA_ACCESS(3, sumsq_N, cp, n, 0, N, bc);

      all_zero_param.out.primary = sum_ncp_ptr;
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = sumsq_ncp_ptr;
      cfg.all_zero_kernel(&all_zero_param);

      /* #pragma omp simd  */
      /* for (int cb = 0; cb < bc; cb++) {  */
      /*   sum_ncp_ptr[cb] = 0.0f;    */
      /*   sumsq_ncp_ptr[cb] = 0.0f;  */
      /* } */


      LIBXSMM_ALIGNED(float lcl_sum_X_X2[2*bc], 64);
      reduce_HW_param.out.primary   = lcl_sum_X_X2;                                                         /* [2*bc]  */
      for(hwb=0; hwb < num_HW_blocks; hwb++){

        reduce_HW_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
        cfg.reduce_HW_kernel(&reduce_HW_param);                                                       /* [HW, bc] -----> [2 * bc] */

        add_param.in0.primary = sum_ncp_ptr;
        add_param.in1.primary = lcl_sum_X_X2;
        add_param.out.primary = sum_ncp_ptr;
        cfg.helper_add_kernel(&add_param);

        add_param.in0.primary = sumsq_ncp_ptr;
        add_param.in1.primary = &lcl_sum_X_X2[bc];
        add_param.out.primary = sumsq_ncp_ptr;
        cfg.helper_add_kernel(&add_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < bc; cb++) {  */
        /*   sum_ncp_ptr[cb] += lcl_sum_X_X2[cb];  */
        /*   sumsq_ncp_ptr[cb] += lcl_sum_X_X2[bc + cb];  */
        /* }  */
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);

    for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {

      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, 0, CP, bc);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, 0, CP, bc);
      cfg.all_zero_kernel(&all_zero_param);

      /* #pragma omp simd */
      /* for (int cb = 0; cb < bc; cb++) {  */
      /*   sum_X_X2[cp*bc + cb] = 0.0f;   */
      /*   sum_X_X2[CP*bc + (cp*bc + cb)] = 0.0f;  */
      /* } */

      int cb, ni;
      for(ni = 0; ni < N; ni++){

        add_param.in0.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, 0, CP, bc);
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, sum_N, cp, ni, 0, N, bc);
        add_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, 0, CP, bc);
        cfg.helper_add_kernel(&add_param);

        add_param.in0.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, 0, CP, bc);
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, sumsq_N, cp, ni, 0, N, bc);
        add_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, 0, CP, bc);
        cfg.helper_add_kernel(&add_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < bc; cb++) { */
        /*   sum_X_X2[cp*bc + cb] += sum_N[cp*N*bc + n*bc + cb]; */
        /*   sum_X_X2[CP*bc + (cp*bc + cb)] += sumsq_N[cp*N*bc + n*bc + cb]; */
        /* } */
      }

      for(cb = 0; cb < bc; cb++){
        mean[cp*bc + cb] = (LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, cb, CP, bc)) * scale;                 /* E[X] */
        var[cp*bc + cb] = ((LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, cb, CP, bc)) * scale) - (mean[cp*bc + cb]*mean[cp*bc + cb]);
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);

  } /* mean and var computation are for the full norm only */

  for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
    n  = cpxnt%N;
    cp = cpxnt/N;

    int hwb, cb;

    for(cb = 0; cb < bc; cb++){
      float lvar   = LIBXSMM_VLA_ACCESS(2, var,   cp, cb, bc);
      float lmean  = LIBXSMM_VLA_ACCESS(2, mean,  cp, cb, bc);

      s[cb] = 1.0f / ((float)sqrt(lvar + eps));                                 /* s = 1/sqrt(var(X) + eps)     [bc] */
      b[cb] = -1 * lmean * s[cb];                                               /* b = -E[X]/sqrt(var(X) + eps) [bc] */

      /* s[cb] = 1.0f / ((float)sqrt(var[cp*bc + cb] + eps)); */                /* s = 1/sqrt(var(X) + eps)     [bc] */
      /* b[cb] = -1 * mean[cp*bc + cb] * s[cb];               */                /* b = -E[X]/sqrt(var(X) + eps) [bc] */
    }
    arg_array[1].primary = s;                                                   /* [bc] */
    arg_array[2].primary = b;                                                   /* [bc] */
    arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);     /* [bc] */
    arg_array[4].primary = (void*)&LIBXSMM_VLA_ACCESS(2, beta,  cp, 0, bc);     /* [bc] */

    for(hwb=0; hwb < num_HW_blocks; hwb++){
      arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);              /* [HW, bc] */
      eqn_param.inputs = arg_array;
      eqn_param.output.primary   = &LIBXSMM_VLA_ACCESS(4, out, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);               /* [HW,bc] */

      if (cfg.fuse_type == MY_BN_FUSE_ELTWISE || cfg.fuse_type == MY_BN_FUSE_ELTWISE_RELU ||  cfg.fuse_type == MY_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
        arg_array[5].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp_add, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);        /* [HW, bc] */
      }

      if (cfg.fuse_type == MY_BN_FUSE_RELU || cfg.fuse_type == MY_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == MY_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
        eqn_param.output.secondary = ((cfg.fuse_type == MY_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == MY_BN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                        (void*)&LIBXSMM_VLA_ACCESS(4, relumask, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, (bc/BITS_PER_CHAR)) : NULL );
      }
      cfg.func10(&eqn_param);                                                   /* Normalization equation + relu + eltwise -> y = relu( ((s*x + b)*gamma + beta) + inp_add) */
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);
}


void my_bn_fwd_exec_bf16( my_bn_fwd_config cfg, const libxsmm_bfloat16 *pinp, const libxsmm_bfloat16 *pinp_add,
                          const float *pgamma, const float *pbeta, float *mean, float *var, libxsmm_bfloat16 *pout, unsigned char *prelumask,
                          float eps, int start_tid, int my_tid, void *scratch, my_bn_norm_type norm_type ) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint HW = cfg.H * cfg.W;
  const libxsmm_blasint bc = cfg.bc;
  const libxsmm_blasint num_HW_blocks = cfg.num_HW_blocks;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;

  /* number of tasks that could be run in parallel for 1d blocking */
  /* Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here? */
  const libxsmm_blasint work_dN = CP * N;
  /* compute chunk size */
  const libxsmm_blasint chunksize_dN = (work_dN % cfg.threads == 0) ?
    (work_dN / cfg.threads) : ((work_dN / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_dN = (ltid * chunksize_dN < work_dN) ? (ltid * chunksize_dN) : work_dN;
  const libxsmm_blasint thr_end_dN = ((ltid + 1) * chunksize_dN < work_dN) ? ((ltid + 1) * chunksize_dN) : work_dN;

  /* number of tasks that could be run in parallel for 1d blocking */
  /* Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here? */
  const libxsmm_blasint work_C = CP;
  /* compute chunk size */
  const libxsmm_blasint chunksize_C = (work_C % cfg.threads == 0) ?
    (work_C / cfg.threads) : ((work_C / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_C = (ltid * chunksize_C < work_C) ? (ltid * chunksize_C) : work_C;
  const libxsmm_blasint thr_end_C = ((ltid + 1) * chunksize_C < work_C) ? ((ltid + 1) * chunksize_C) : work_C;

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  LIBXSMM_VLA_DECL(4, const libxsmm_bfloat16,         inp,      pinp, CP, HW, bc);            /* [N, CP, HW, bc] */
  LIBXSMM_VLA_DECL(4,       libxsmm_bfloat16,         out,      pout, CP, HW, bc);            /* [N, CP, HW, bc] */
  LIBXSMM_VLA_DECL(2, const float,                    gamma,    pgamma, bc);                  /* [CP, bc] */
  LIBXSMM_VLA_DECL(2, const float,                    beta,     pbeta, bc);                   /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float,                    mean,     mean,  bc);                   /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float,                    var,      var,   bc);                   /* [CP, bc] */

  LIBXSMM_VLA_DECL(4, const libxsmm_bfloat16,         inp_add,  pinp_add, CP, HW, bc);        /* [N, CP, HW, bc] */

  float alpha = 0.0f;
  LIBXSMM_VLA_DECL(4,       unsigned char, relumask, prelumask, CP, HW, bc/BITS_PER_CHAR);    /* [N, CP, HW, bc/BITS_PER_CHAR] */

  libxsmm_meltw_binary_param add_param;
  libxsmm_meltw_unary_param  all_relu_param;

  libxsmm_matrix_arg arg_array[6];

  libxsmm_matrix_eqn_param eqn_param;

  memset( &add_param,       0, sizeof(add_param));
  memset( &all_relu_param,  0, sizeof(all_relu_param));
  memset( &eqn_param, 0, sizeof(eqn_param));

  LIBXSMM_ALIGNED(float s[bc], 64);
  LIBXSMM_ALIGNED(float b[bc], 64);
  int n, cp;

  int cpxnt;
  if (norm_type == MY_BN_FULL_NORM) {

    const float scale = 1.0f /((float)N * HW);

    LIBXSMM_VLA_DECL(3, float, sum_X_X2, ((float*)scratch), CP, bc);  /* [2, CP, bc] */
    LIBXSMM_ASSUME_ALIGNED(sum_X_X2_, 64);
    const libxsmm_blasint sum_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + CP * 2 * bc), 64) - ((uintptr_t)(scratch))) / sizeof(float);
    LIBXSMM_VLA_DECL(3, float, sum_N, ((float*)scratch) + sum_N_offset, N, bc);  /* [CP, N, bc] */
    LIBXSMM_ASSUME_ALIGNED(sum_N_, 64);
    const libxsmm_blasint sumsq_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + sum_N_offset + CP * N * bc), 64) - ((uintptr_t)(scratch))) / sizeof(float);
    LIBXSMM_VLA_DECL(3, float, sumsq_N, ((float*)scratch) + sumsq_N_offset, N, bc);  /* [CP, N, bc] */
    LIBXSMM_ASSUME_ALIGNED(sumsq_N_, 64);

    libxsmm_meltw_unary_param  all_zero_param;
    libxsmm_meltw_unary_param  reduce_HW_param;

    memset( &all_zero_param,  0, sizeof(all_zero_param));
    memset( &reduce_HW_param, 0, sizeof(reduce_HW_param));

    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      n  = cpxnt%N;
      cp = cpxnt/N;

      int hwb;

      float *sum_ncp_ptr   = &LIBXSMM_VLA_ACCESS(3, sum_N,   cp, n, 0, N, bc);
      float *sumsq_ncp_ptr = &LIBXSMM_VLA_ACCESS(3, sumsq_N, cp, n, 0, N, bc);

      all_zero_param.out.primary = sum_ncp_ptr;
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = sumsq_ncp_ptr;
      cfg.all_zero_kernel(&all_zero_param);

      /* #pragma omp simd  */
      /* for (int cb = 0; cb < bc; cb++) {  */
      /*   sum_ncp_ptr[cb] = 0.0f;    */
      /*   sumsq_ncp_ptr[cb] = 0.0f;  */
      /* } */


      LIBXSMM_ALIGNED(float lcl_sum_X_X2[2*bc], 64);
      reduce_HW_param.out.primary   = lcl_sum_X_X2;                                                         /* [2*bc]  */
      for(hwb=0; hwb < num_HW_blocks; hwb++){

        reduce_HW_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
        cfg.reduce_HW_kernel(&reduce_HW_param);                                                       /* [HW, bc] -----> [2 * bc] */

        add_param.in0.primary = sum_ncp_ptr;
        add_param.in1.primary = lcl_sum_X_X2;
        add_param.out.primary = sum_ncp_ptr;
        cfg.helper_add_kernel(&add_param);

        add_param.in0.primary = sumsq_ncp_ptr;
        add_param.in1.primary = &lcl_sum_X_X2[bc];
        add_param.out.primary = sumsq_ncp_ptr;
        cfg.helper_add_kernel(&add_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < bc; cb++) {  */
        /*   sum_ncp_ptr[cb] += lcl_sum_X_X2[cb];  */
        /*   sumsq_ncp_ptr[cb] += lcl_sum_X_X2[bc + cb];  */
        /* }  */
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);

    for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {

      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, 0, CP, bc);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, 0, CP, bc);
      cfg.all_zero_kernel(&all_zero_param);

      /* #pragma omp simd */
      /* for (int cb = 0; cb < bc; cb++) {  */
      /*   sum_X_X2[cp*bc + cb] = 0.0f;   */
      /*   sum_X_X2[CP*bc + (cp*bc + cb)] = 0.0f;  */
      /* } */

      int cb, ni;
      for(ni = 0; ni < N; ni++){

        add_param.in0.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, 0, CP, bc);
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, sum_N, cp, ni, 0, N, bc);
        add_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, 0, CP, bc);
        cfg.helper_add_kernel(&add_param);

        add_param.in0.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, 0, CP, bc);
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, sumsq_N, cp, ni, 0, N, bc);
        add_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, 0, CP, bc);
        cfg.helper_add_kernel(&add_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < bc; cb++) { */
        /*   sum_X_X2[cp*bc + cb] += sum_N[cp*N*bc + n*bc + cb]; */
        /*   sum_X_X2[CP*bc + (cp*bc + cb)] += sumsq_N[cp*N*bc + n*bc + cb]; */
        /* } */
      }

      for(cb = 0; cb < bc; cb++){
        mean[cp*bc + cb] = (LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, cb, CP, bc)) * scale;                 /* E[X] */
        var[cp*bc + cb] = ((LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, cb, CP, bc)) * scale) - (mean[cp*bc + cb]*mean[cp*bc + cb]);
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);
  } /* mean and var computation are for the full norm only */

  for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
    n  = cpxnt%N;
    cp = cpxnt/N;

    int hwb, cb;

    for(cb = 0; cb < bc; cb++){
      float lvar   = LIBXSMM_VLA_ACCESS(2, var,   cp, cb, bc);
      float lmean  = LIBXSMM_VLA_ACCESS(2, mean,  cp, cb, bc);

      s[cb] = 1.0f / ((float)sqrt(lvar + eps));                                 /* s = 1/sqrt(var(X) + eps)     [bc] */
      b[cb] = -1 * lmean * s[cb];                                               /* b = -E[X]/sqrt(var(X) + eps) [bc] */

      /* s[cb] = 1.0f / ((float)sqrt(var[cp*bc + cb] + eps)); */                /* s = 1/sqrt(var(X) + eps)     [bc] */
      /* b[cb] = -1 * mean[cp*bc + cb] * s[cb];               */                /* b = -E[X]/sqrt(var(X) + eps) [bc] */
    }
    arg_array[1].primary = s;                                                   /* [bc] */
    arg_array[2].primary = b;                                                   /* [bc] */
    arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);     /* [bc] */
    arg_array[4].primary = (void*)&LIBXSMM_VLA_ACCESS(2, beta,  cp, 0, bc);     /* [bc] */

    for(hwb=0; hwb < num_HW_blocks; hwb++){
      arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);          /* [HW, bc] */
      eqn_param.inputs = arg_array;
      eqn_param.output.primary   = &LIBXSMM_VLA_ACCESS(4, out, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);           /* [HW,bc] */

      if (cfg.fuse_type == MY_BN_FUSE_ELTWISE || cfg.fuse_type == MY_BN_FUSE_ELTWISE_RELU ||  cfg.fuse_type == MY_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
        arg_array[5].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp_add, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);    /* [HW, bc] */
      }

      if (cfg.fuse_type == MY_BN_FUSE_RELU || cfg.fuse_type == MY_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == MY_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
        eqn_param.output.secondary = ((cfg.fuse_type == MY_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == MY_BN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                      (void*)&LIBXSMM_VLA_ACCESS(4, relumask, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, (bc/BITS_PER_CHAR)) : NULL );
      }
      cfg.func10(&eqn_param);                                                   /* Normalization equation + relu + eltwise -> y = relu( ((s*x + b)*gamma + beta) + inp_add) */
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);
}


void my_bn_bwd_exec_f32( my_bn_bwd_config cfg, float *pdout, const float *pinp, const float *mean, const float *var, const float *pgamma, const unsigned char *prelumask,
                         float *pdin, float *pdin_add, float *pdgamma, float *pdbeta, float eps,
                         int start_tid, int my_tid, void *scratch, my_bn_norm_type norm_type) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint HW = cfg.H * cfg.W;
  const libxsmm_blasint bc = cfg.bc;
  const libxsmm_blasint num_HW_blocks = cfg.num_HW_blocks;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;

  /* number of tasks that could be run in parallel for 1d blocking */
  /* Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here? */
  const libxsmm_blasint work_dN = N * CP;
  /* compute chunk size */
  const libxsmm_blasint chunksize_dN = (work_dN % cfg.threads == 0) ?
    (work_dN / cfg.threads) : ((work_dN / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_dN = ( ltid      * chunksize_dN < work_dN) ? ( ltid      * chunksize_dN) : work_dN;
  const libxsmm_blasint thr_end_dN   = ((ltid + 1) * chunksize_dN < work_dN) ? ((ltid + 1) * chunksize_dN) : work_dN;

  /* number of tasks that could be run in parallel for 1d blocking */
  /* Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here? */
  const libxsmm_blasint work_C = CP;
  /* compute chunk size */
  const libxsmm_blasint chunksize_C = (work_C % cfg.threads == 0) ?
    (work_C / cfg.threads) : ((work_C / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_C = ( ltid      * chunksize_C < work_C) ? ( ltid      * chunksize_C) : work_C;
  const libxsmm_blasint thr_end_C   = ((ltid + 1) * chunksize_C < work_C) ? ((ltid + 1) * chunksize_C) : work_C;

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  const float scale = 1.0f / ((float)N*HW);                   /* Scaling parameter*/

  LIBXSMM_VLA_DECL(4,       float, din,    pdin, CP, HW, bc);          /* [N, CP, HW, bc] */
  LIBXSMM_VLA_DECL(4, const float, inp,    pinp, CP, HW, bc);          /* [N, CP, HW, bc] */
  LIBXSMM_VLA_DECL(4,       float, dout,   pdout, CP, HW, bc);         /* [N, CP, HW, bc] */
  LIBXSMM_VLA_DECL(2, const float, gamma,  pgamma, bc);                /* [CP, bc] */
  LIBXSMM_VLA_DECL(2, const float, mean,   mean,  bc);                 /* [CP, bc] */
  LIBXSMM_VLA_DECL(2, const float, var,    var,   bc);                 /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float, dgamma, pdgamma, bc);               /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float, dbeta,  pdbeta, bc);                /* [CP, bc] */

  LIBXSMM_VLA_DECL(4,       float, din_add, pdin_add, CP, HW, bc);     /* [N, CP, HW, bc] */

  float alpha = 0.0f;
  LIBXSMM_VLA_DECL(4, const unsigned char, relumask, prelumask, CP, HW, bc/BITS_PER_CHAR);    /* [N, CP, HW, bc/BITS_PER_CHAR] */

  const libxsmm_blasint dbeta_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + CP * N * bc), 64) - ((uintptr_t)(scratch))) / sizeof(float);
  LIBXSMM_VLA_DECL(3, float, dgamma_N, ((float*)scratch),                  N, bc);  /* [CP, N, bc] */
  LIBXSMM_ASSUME_ALIGNED(dgamma_N_, 64);
  LIBXSMM_VLA_DECL(3, float, dbeta_N,  ((float*)scratch) + dbeta_N_offset, N, bc);  /* [CP, N, bc] */
  LIBXSMM_ASSUME_ALIGNED(dbeta_N_, 64);

  libxsmm_meltw_unary_param  all_zero_param;
  libxsmm_meltw_binary_param add_param;
  libxsmm_meltw_unary_param  copy_param;
  libxsmm_meltw_unary_param  all_relu_param;
  libxsmm_meltw_unary_param  ewise_copy_param;

  memset( &all_zero_param,   0, sizeof(all_zero_param));
  memset( &add_param,        0, sizeof(add_param));
  memset( &copy_param,       0, sizeof(copy_param));
  memset( &all_relu_param,   0, sizeof(all_relu_param));
  memset( &ewise_copy_param, 0, sizeof(ewise_copy_param));

  libxsmm_matrix_arg arg_array[8];
  libxsmm_matrix_eqn_param eqn_param;

  memset( &eqn_param,        0, sizeof(eqn_param));

  LIBXSMM_ALIGNED(float a[bc], 64); /* could also get moved into the scratch but left on the private stack as these are small, same below */
  LIBXSMM_ALIGNED(float b[bc], 64);
  LIBXSMM_ALIGNED(float c[bc], 64);
  int n, cp;

  int cpxnt;

  if (norm_type == MY_BN_FULL_NORM) {
    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {

      n  = cpxnt%N;
      cp = cpxnt/N;

      int hwb, cb;

      LIBXSMM_ALIGNED(float lcl_dgamma_ptr[bc], 64);
      LIBXSMM_ALIGNED(float lcl_dbeta_ptr[bc], 64);

      float *dgamma_ncp_ptr = &LIBXSMM_VLA_ACCESS(3, dgamma_N, cp, n, 0, N, bc);
      float *dbeta_ncp_ptr  = &LIBXSMM_VLA_ACCESS(3, dbeta_N, cp, n, 0, N, bc);

      all_zero_param.out.primary = lcl_dgamma_ptr;
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = lcl_dbeta_ptr;
      cfg.all_zero_kernel(&all_zero_param);

      /* #pragma omp simd */
      /* for (int cb = 0; cb < bc; cb++) { */
      /*   lcl_dgamma_ptr[cb] = 0.0f; */
      /*   lcl_dbeta_ptr[cb] = 0.0f; */
      /* } */

      for(cb = 0; cb < bc; cb++){
        float lvar   = LIBXSMM_VLA_ACCESS(2, var,   cp, cb, bc);
        float lmean  = LIBXSMM_VLA_ACCESS(2, mean,  cp, cb, bc);

        a[cb] = 1.0f / ((float)sqrt(lvar + eps));
        b[cb] = -a[cb] * lmean;

        /* a[cb] = 1.0f / ((float)sqrt(var[cp*bc + cb] + eps)); */
        /* b[cb] = -a[cb]*mean[cp*bc + cb];                     */
      }

      arg_array[1].primary = a;
      arg_array[2].primary = b;
      arg_array[4].primary = lcl_dgamma_ptr;
      arg_array[5].primary = lcl_dbeta_ptr;
      arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);

      for(hwb=0; hwb < num_HW_blocks; hwb++){
        if (cfg.fuse_type == MY_BN_FUSE_ELTWISE ||
          cfg.fuse_type == MY_BN_FUSE_RELU || cfg.fuse_type == MY_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == MY_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
          if (cfg.fuse_type == MY_BN_FUSE_RELU || cfg.fuse_type == MY_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == MY_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
            all_relu_param.op.primary   = (void*)(&alpha);
            all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);      /* [HW,bc] */
            all_relu_param.in.secondary = ((cfg.fuse_type == MY_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == MY_BN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                             (void*)&LIBXSMM_VLA_ACCESS(4, relumask, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc/8)
                                             : NULL /*&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc) */ ); /* dout_fwd ? nonsense? */
            all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);      /* [HW,bc] */
            cfg.inv_relu_kernel(&all_relu_param);
          } /* ReLU/mask */
          if (cfg.fuse_type == MY_BN_FUSE_ELTWISE || cfg.fuse_type == MY_BN_FUSE_ELTWISE_RELU || cfg.fuse_type == MY_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
            ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(4, dout,    n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
            ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(4, din_add, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
            cfg.ewise_copy_kernel(&ewise_copy_param);
          } /* Eltwise */
        }
        arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
        arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);

        eqn_param.inputs = arg_array;
        eqn_param.output.primary = lcl_dgamma_ptr;
        cfg.dgamma_func(&eqn_param);                                                             /* dgamma += (a * inp + b) * dout */

        eqn_param.output.primary = lcl_dbeta_ptr;
        cfg.dbeta_func(&eqn_param);                                                              /* dbeta += dout */
      }

      copy_param.in.primary = lcl_dgamma_ptr;
      copy_param.out.primary = dgamma_ncp_ptr;
      cfg.helper_copy_kernel(&copy_param);

      copy_param.in.primary = lcl_dbeta_ptr;
      copy_param.out.primary = dbeta_ncp_ptr;
      cfg.helper_copy_kernel(&copy_param);

      /* #pragma omp simd */
      /* for (int cb = 0; cb < bc; cb++) { */
      /*   dgamma_ncp_ptr[cb] = lcl_dgamma_ptr[cb]; */
      /*   dbeta_ncp_ptr[cb] = lcl_dbeta_ptr[cb]; */
      /* } */
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);

    for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, bc);
      cfg.all_zero_kernel(&all_zero_param);

      /* #pragma omp simd */
      /* for (int cb = 0; cb < bc; cb++) { */
      /*   pdgamma[cp*bc + cb] = 0.0f; */
      /*   pdbeta[cp*bc + cb] = 0.0f; */
      /* } */

      int ni;
      for(ni = 0; ni < N; ni++){

        add_param.in0.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, cp, ni, 0, N, bc);
        add_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
        cfg.helper_add_kernel(&add_param);

        add_param.in0.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, bc);
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, cp, ni, 0, N, bc);
        add_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, bc);
        cfg.helper_add_kernel(&add_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < bc; cb++) { */
        /*   pdgamma[cp*bc + cb] += dgamma_N[cp*N*bc + n*bc + cb];  */
        /*   pdbeta[cp*bc + cb] += dbeta_N[cp*N*bc + n*bc + cb];  */
        /* } */
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);

  } /* this is only computed in case of full backward (norm_type ~ 0) */

  for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
    n  = cpxnt%N;
    cp = cpxnt/N;

    int hwb, cb;

    for(cb = 0; cb < bc; cb++){
      float lgamma  = LIBXSMM_VLA_ACCESS(2, gamma,  cp, cb, bc);
      float ldgamma = LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, bc);
      float lvar    = LIBXSMM_VLA_ACCESS(2, var,    cp, cb, bc);
      float lmean   = LIBXSMM_VLA_ACCESS(2, mean,   cp, cb, bc);
      float ldbeta  = LIBXSMM_VLA_ACCESS(2, dbeta,  cp, cb, bc);

      a[cb]        = lgamma / ((float)sqrt(lvar + eps));                            /* a = gamma_ptr[bc] * brstd_ptr[bc] */
      b[cb]        = -a[cb] * scale * ldgamma / ((float)sqrt(lvar + eps));          /* b = gamma_ptr[bc] * brstd_ptr[bc] * del_gamma_ptr[v] * brstd_ptr[bc] * recp_nhw */
      c[cb]        = -b[cb] * lmean - a[cb] * scale * ldbeta ;                      /* c = -gamma_ptr[bc] * brstd_ptr[bc] * recp_nhw * del_beta_ptr[bc] + gamma_ptr[bc] * brstd_ptr[bc] * recp_nhw * bmean_ptr[bc] * del_gamma_ptr[bc] * brstd_ptr[bc]) */
    }

    arg_array[1].primary = a;
    arg_array[2].primary = b;
    arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);
    arg_array[7].primary = c;

    for(hwb=0; hwb < num_HW_blocks; hwb++){
      arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
      arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);

      eqn_param.inputs = arg_array;
      eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, din, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
      cfg.din_func(&eqn_param);                                                                     /* din = dout * a + b * inp + c */

    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);
}


void my_bn_bwd_exec_bf16( my_bn_bwd_config cfg, libxsmm_bfloat16 *pdout, const libxsmm_bfloat16 *pinp, const float *mean, const float *var, const float *pgamma, const unsigned char *prelumask,
                         libxsmm_bfloat16 *pdin, libxsmm_bfloat16 *pdin_add, float *pdgamma, float *pdbeta, float eps,
                         int start_tid, int my_tid, void *scratch, my_bn_norm_type norm_type) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint HW = cfg.H * cfg.W;
  const libxsmm_blasint bc = cfg.bc;
  const libxsmm_blasint num_HW_blocks = cfg.num_HW_blocks;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;

  /* number of tasks that could be run in parallel for 1d blocking */
  /* Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here? */
  const libxsmm_blasint work_dN = N * CP;
  /* compute chunk size */
  const libxsmm_blasint chunksize_dN = (work_dN % cfg.threads == 0) ?
    (work_dN / cfg.threads) : ((work_dN / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_dN = ( ltid      * chunksize_dN < work_dN) ? ( ltid      * chunksize_dN) : work_dN;
  const libxsmm_blasint thr_end_dN   = ((ltid + 1) * chunksize_dN < work_dN) ? ((ltid + 1) * chunksize_dN) : work_dN;

  /* number of tasks that could be run in parallel for 1d blocking */
  /* Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here? */
  const libxsmm_blasint work_C = CP;
  /* compute chunk size */
  const libxsmm_blasint chunksize_C = (work_C % cfg.threads == 0) ?
    (work_C / cfg.threads) : ((work_C / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_C = ( ltid      * chunksize_C < work_C) ? ( ltid      * chunksize_C) : work_C;
  const libxsmm_blasint thr_end_C   = ((ltid + 1) * chunksize_C < work_C) ? ((ltid + 1) * chunksize_C) : work_C;

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  const float scale = 1.0f / ((float)N*HW);                                        /* Scaling parameter*/

  LIBXSMM_VLA_DECL(4,       libxsmm_bfloat16, din,    pdin,  CP, HW, bc);          /* [N, CP, HW, bc] */
  LIBXSMM_VLA_DECL(4, const libxsmm_bfloat16, inp,    pinp,  CP, HW, bc);          /* [N, CP, HW, bc] */
  LIBXSMM_VLA_DECL(4,       libxsmm_bfloat16, dout,   pdout, CP, HW, bc);          /* [N, CP, HW, bc] */
  LIBXSMM_VLA_DECL(2, const float,            gamma,  pgamma,  bc);                /* [CP, bc] */
  LIBXSMM_VLA_DECL(2, const float,            mean,   mean,    bc);                /* [CP, bc] */
  LIBXSMM_VLA_DECL(2, const float,            var,    var,     bc);                /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float,            dgamma, pdgamma, bc);                /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float,            dbeta,  pdbeta,  bc);                /* [CP, bc] */

  LIBXSMM_VLA_DECL(4,       libxsmm_bfloat16, din_add, pdin_add, CP, HW, bc);      /* [N, CP, HW, bc] */

  float alpha = 0.0f;
  LIBXSMM_VLA_DECL(4, const unsigned char, relumask, prelumask, CP, HW, bc/BITS_PER_CHAR);    /* [N, CP, HW, bc/BITS_PER_CHAR] */

  libxsmm_matrix_arg arg_array[8];
  libxsmm_matrix_eqn_param eqn_param;

  memset( &eqn_param,        0, sizeof(eqn_param));

  LIBXSMM_ALIGNED(float a[bc], 64); /* could also get moved into the scratch but left on the private stack as these are small, same below */
  LIBXSMM_ALIGNED(float b[bc], 64);
  LIBXSMM_ALIGNED(float c[bc], 64);
  int n, cp;

  int cpxnt;

  if (norm_type == MY_BN_FULL_NORM) {

    const libxsmm_blasint dbeta_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + CP * N * bc), 64) - ((uintptr_t)(scratch))) / sizeof(float);
    LIBXSMM_VLA_DECL(3, float, dgamma_N, ((float*)scratch),                  N, bc);  /* [CP, N, bc] */
    LIBXSMM_VLA_DECL(3, float, dbeta_N,  ((float*)scratch) + dbeta_N_offset, N, bc);  /* [CP, N, bc] */
    LIBXSMM_ASSUME_ALIGNED(dgamma_N_, 64);
    LIBXSMM_ASSUME_ALIGNED(dbeta_N_,  64);

    libxsmm_meltw_unary_param  all_zero_param;
    libxsmm_meltw_binary_param add_param;
    libxsmm_meltw_unary_param  copy_param;
    libxsmm_meltw_unary_param  all_relu_param;
    libxsmm_meltw_unary_param  ewise_copy_param;

    memset( &all_zero_param,   0, sizeof(all_zero_param));
    memset( &add_param,        0, sizeof(add_param));
    memset( &copy_param,       0, sizeof(copy_param));
    memset( &all_relu_param,   0, sizeof(all_relu_param));
    memset( &ewise_copy_param, 0, sizeof(ewise_copy_param));

    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {

      n  = cpxnt%N;
      cp = cpxnt/N;

      int hwb, cb;

      LIBXSMM_ALIGNED(float lcl_dgamma_ptr[bc], 64);
      LIBXSMM_ALIGNED(float lcl_dbeta_ptr[bc], 64);

      float *dgamma_ncp_ptr = &LIBXSMM_VLA_ACCESS(3, dgamma_N, cp, n, 0, N, bc);
      float *dbeta_ncp_ptr  = &LIBXSMM_VLA_ACCESS(3, dbeta_N, cp, n, 0, N, bc);

      all_zero_param.out.primary = lcl_dgamma_ptr;
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = lcl_dbeta_ptr;
      cfg.all_zero_kernel(&all_zero_param);

      /* #pragma omp simd */
      /* for (cb = 0; cb < bc; cb++) { */
      /*   lcl_dgamma_ptr[cb] = 0.0f; */
      /*   lcl_dbeta_ptr[cb] = 0.0f; */
      /* } */

      for(cb = 0; cb < bc; cb++){
        float lvar   = LIBXSMM_VLA_ACCESS(2, var,   cp, cb, bc);
        float lmean  = LIBXSMM_VLA_ACCESS(2, mean,  cp, cb, bc);

        a[cb] = 1.0f / ((float)sqrt(lvar + eps));
        b[cb] = -a[cb] * lmean;

        /* a[cb] = 1.0f / ((float)sqrt(var[cp*bc + cb] + eps)); */
        /* b[cb] = -a[cb]*mean[cp*bc + cb];                     */
      }

      arg_array[1].primary = a;
      arg_array[2].primary = b;
      arg_array[4].primary = lcl_dgamma_ptr;
      arg_array[5].primary = lcl_dbeta_ptr;
      arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);

      for(hwb=0; hwb < num_HW_blocks; hwb++){

        if (cfg.fuse_type == MY_BN_FUSE_ELTWISE ||
          cfg.fuse_type == MY_BN_FUSE_RELU || cfg.fuse_type == MY_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == MY_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
          if (cfg.fuse_type == MY_BN_FUSE_RELU || cfg.fuse_type == MY_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == MY_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
            all_relu_param.op.primary   = (void*)(&alpha);
            all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);      /* [HW,bc] */
            all_relu_param.in.secondary = ((cfg.fuse_type == MY_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == MY_BN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                             (void*)&LIBXSMM_VLA_ACCESS(4, relumask, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc/8)
                                             : NULL /*&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc) */ ); /* dout_fwd ? nonsense? */
            all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);      /* [HW,bc] */
            cfg.inv_relu_kernel(&all_relu_param);
          } /* ReLU/mask */
          if (cfg.fuse_type == MY_BN_FUSE_ELTWISE || cfg.fuse_type == MY_BN_FUSE_ELTWISE_RELU || cfg.fuse_type == MY_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
            ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(4, dout,    n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
            ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(4, din_add, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
            cfg.ewise_copy_kernel(&ewise_copy_param);
          } /* Eltwise */
        }

        arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp,  n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
        arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);

        eqn_param.inputs = arg_array;
        eqn_param.output.primary = lcl_dgamma_ptr;
        cfg.dgamma_func(&eqn_param);                                                             /* dgamma += (a * inp + b) * dout */

        eqn_param.output.primary = lcl_dbeta_ptr;
        cfg.dbeta_func(&eqn_param);                                                              /* dbeta += dout */
      }

      copy_param.in.primary = lcl_dgamma_ptr;
      copy_param.out.primary = dgamma_ncp_ptr;
      cfg.helper_copy_kernel(&copy_param);

      copy_param.in.primary = lcl_dbeta_ptr;
      copy_param.out.primary = dbeta_ncp_ptr;
      cfg.helper_copy_kernel(&copy_param);

      /* #pragma omp simd */
      /* for (int cb = 0; cb < bc; cb++) { */
      /*   dgamma_ncp_ptr[cb] = lcl_dgamma_ptr[cb]; */
      /*   dbeta_ncp_ptr[cb] = lcl_dbeta_ptr[cb]; */
      /* } */
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);

    for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, bc);
      cfg.all_zero_kernel(&all_zero_param);

      /* #pragma omp simd */
      /* for (int cb = 0; cb < bc; cb++) { */
      /*   pdgamma[cp*bc + cb] = 0.0f; */
      /*   pdbeta[cp*bc + cb] = 0.0f; */
      /* } */

      int ni;
      for(ni = 0; ni < N; ni++){

        add_param.in0.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, cp, ni, 0, N, bc);
        add_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
        cfg.helper_add_kernel(&add_param);

        add_param.in0.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, bc);
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, cp, ni, 0, N, bc);
        add_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, bc);
        cfg.helper_add_kernel(&add_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < bc; cb++) { */
        /*   pdgamma[cp*bc + cb] += dgamma_N[cp*N*bc + n*bc + cb];  */
        /*   pdbeta[cp*bc + cb] += dbeta_N[cp*N*bc + n*bc + cb];  */
        /* } */
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);

  } /* this is only computed in case of full backward (norm_type ~ 0) */

  for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
    n  = cpxnt%N;
    cp = cpxnt/N;

    int hwb, cb;

    for(cb = 0; cb < bc; cb++){
      float lgamma  = LIBXSMM_VLA_ACCESS(2, gamma,  cp, cb, bc);
      float ldgamma = LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, bc);
      float lvar    = LIBXSMM_VLA_ACCESS(2, var,    cp, cb, bc);
      float lmean   = LIBXSMM_VLA_ACCESS(2, mean,   cp, cb, bc);
      float ldbeta  = LIBXSMM_VLA_ACCESS(2, dbeta,  cp, cb, bc);

      a[cb]         = lgamma / ((float)sqrt(lvar + eps));                            /* a = gamma_ptr[bc] * brstd_ptr[bc] */
      b[cb]         = -a[cb] * scale * ldgamma / ((float)sqrt(lvar + eps));          /* b = gamma_ptr[bc] * brstd_ptr[bc] * del_gamma_ptr[v] * brstd_ptr[bc] * recp_nhw */
      c[cb]         = -b[cb] * lmean - a[cb] * scale * ldbeta ;                      /* c = -gamma_ptr[bc] * brstd_ptr[bc] * recp_nhw * del_beta_ptr[bc] + gamma_ptr[bc] * brstd_ptr[bc] * recp_nhw * bmean_ptr[bc] * del_gamma_ptr[bc] * brstd_ptr[bc]) */
    }

    arg_array[1].primary = a;
    arg_array[2].primary = b;
    arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);
    arg_array[7].primary = c;

    for(hwb=0; hwb < num_HW_blocks; hwb++){
      arg_array[0].primary     = (void*)&LIBXSMM_VLA_ACCESS(4, inp,  n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
      arg_array[3].primary     = (void*)&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
      eqn_param.output.primary = (void*)&LIBXSMM_VLA_ACCESS(4, din, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);;

      eqn_param.inputs = arg_array;
      cfg.din_func(&eqn_param);                                                     /* din = dout * a + b * inp + c */
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);
}
