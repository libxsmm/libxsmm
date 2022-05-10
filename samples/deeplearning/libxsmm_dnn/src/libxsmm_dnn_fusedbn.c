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

#include <libxsmm_dnn_fusedbn.h>

#define BITS_PER_CHAR (8)

LIBXSMM_API libxsmm_dnn_bn_fwd_config setup_libxsmm_dnn_bn_fwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint bc,
                                 libxsmm_blasint pad_h_in, libxsmm_blasint pad_w_in, libxsmm_blasint pad_h_out, libxsmm_blasint pad_w_out,
                                 libxsmm_blasint threads, libxsmm_dnn_bn_fuse fuse_type,
                                 libxsmm_datatype datatype_in, libxsmm_datatype datatype_out, libxsmm_datatype datatype_comp ) {
  libxsmm_dnn_bn_fwd_config res;

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

  memset( &res, 0, sizeof(res));

  /* setting up some handle values */
  res.N  = N;
  res.C  = C;
  res.H  = H;
  res.W  = W;
  res.bc = bc;
  res.CP = res.C / res.bc;
  res.num_HW_blocks = (res.H > res.W ? res.H : res.W );
  res.num_W_blocks  = (res.W % 64 == 0 ? res.W / 64 : 1); /* FIXME: Random heuristic */
  res.pad_h_in      = pad_h_in;
  res.pad_w_in      = pad_w_in;
  res.pad_h_out     = pad_h_out;
  res.pad_w_out     = pad_w_out;
  res.threads       = threads;
  res.fuse_type     = fuse_type;

  if (res.pad_h_in != 0 || res.pad_w_in != 0 || res.pad_h_out != 0 || res.pad_w_out != 0 )
    res.use_hw_blocking = 0; /* alternative is w blocking ([w, bc] blocks) */
  else
    res.use_hw_blocking = 1; /* using [hw, bc] blocks */

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

  if (res.pad_h_out != 0) {
    libxsmm_blasint ofwp   = res.W + 2 * res.pad_w_out;
    unary_flags            = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    unary_shape            = libxsmm_create_meltw_unary_shape(res.bc, (res.pad_h_out * ofwp), res.bc, ldo, res.datatype_comp, res.datatype_comp, res.datatype_comp);
    res.all_zero_hp_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
    if ( res.all_zero_hp_kernel == NULL) {
      fprintf( stderr, "JIT for TPP fwd all_zero_hp_kernel failed. Bailing...!\n");
      exit(-1);
    }
  }

  if (res.pad_w_out != 0) {
    unary_flags            = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    unary_shape            = libxsmm_create_meltw_unary_shape(res.bc, res.pad_w_out, res.bc, ldo, res.datatype_comp, res.datatype_comp, res.datatype_comp);
    res.all_zero_wp_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
    if ( res.all_zero_wp_kernel == NULL) {
      fprintf( stderr, "JIT for TPP fwd all_zero_wp_kernel failed. Bailing...!\n");
      exit(-1);
    }
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

  if (res.use_hw_blocking == 0)
    unary_shape     = libxsmm_create_meltw_unary_shape(res.bc, res.W / res.num_W_blocks, ld, tmp_ld, res.datatype_in, res.datatype_comp, res.datatype_comp);
  else
    unary_shape     = libxsmm_create_meltw_unary_shape(res.bc, res.H*res.W / res.num_HW_blocks, ld, tmp_ld, res.datatype_in, res.datatype_comp, res.datatype_comp);
  unary_flags       = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  res.reduce_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD, unary_shape, unary_flags);
  if ( res.reduce_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd reduce_kernel failed. Bailing...!\n");
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
  if (res.use_hw_blocking == 0)
    arg_shape[0].n  = res.W /res.num_W_blocks;
  else
    arg_shape[0].n  = res.H*res.W /res.num_HW_blocks;
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
    if (res.use_hw_blocking == 0)
      arg_shape[5].n  = res.W /res.num_W_blocks;
    else
      arg_shape[5].n  = res.H*res.W / res.num_HW_blocks;
    arg_shape[5].ld   = ld;
    arg_shape[5].type = res.datatype_in;
    libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[5], arg_shape[5], arg_singular_attr);
  }

  eqn_out_arg_shape.m    = res.bc;                                 /* y = [HW, bc] */
  if (res.use_hw_blocking == 0)
    eqn_out_arg_shape.n  = res.W /res.num_W_blocks;
  else
    eqn_out_arg_shape.n  = res.H*res.W / res.num_HW_blocks;
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

LIBXSMM_API libxsmm_dnn_bn_bwd_config setup_libxsmm_dnn_bn_bwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint bc,
                                 libxsmm_blasint pad_h_in, libxsmm_blasint pad_w_in, libxsmm_blasint pad_h_out, libxsmm_blasint pad_w_out,
                                 libxsmm_blasint threads, libxsmm_dnn_bn_fuse fuse_type,
                                 libxsmm_datatype datatype_in, libxsmm_datatype datatype_out, libxsmm_datatype datatype_comp ) {
  libxsmm_dnn_bn_bwd_config res;

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
  res.num_W_blocks  = (res.W % 64 == 0 ? res.W / 64 : 1); /* FIXME: Random heuristic [NEED TO TEST WITH nblocks != 1] */
  res.pad_h_in      = pad_h_in;
  res.pad_w_in      = pad_w_in;
  res.pad_h_out     = pad_h_out;
  res.pad_w_out     = pad_w_out;
  if (res.pad_h_in != 0 || res.pad_w_in != 0 || res.pad_h_out != 0 || res.pad_w_out != 0 )
    res.use_hw_blocking = 0; /* alternative is w blocking ([w, bc] blocks) */
  else
    res.use_hw_blocking = 1; /* using [hw, bc] blocks */

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
    fprintf( stderr, "JIT for TPP bwd all_zero_kernel failed. Bailing...!\n");
    exit(-1);
  }

  if (res.pad_h_in != 0) {
    libxsmm_blasint ifwp   = res.W + 2 * res.pad_w_in;
    unary_flags            = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    unary_shape            = libxsmm_create_meltw_unary_shape(res.bc, (res.pad_h_in * ifwp), res.bc, ldo, res.datatype_comp, res.datatype_comp, res.datatype_comp);
    res.all_zero_hp_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
    if ( res.all_zero_hp_kernel == NULL) {
      fprintf( stderr, "JIT for TPP bwd all_zero_hp_kernel failed. Bailing...!\n");
      exit(-1);
    }
  }

  if (res.pad_w_in != 0) {
    unary_flags            = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    unary_shape            = libxsmm_create_meltw_unary_shape(res.bc, res.pad_w_in, res.bc, ldo, res.datatype_comp, res.datatype_comp, res.datatype_comp);
    res.all_zero_wp_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
    if ( res.all_zero_wp_kernel == NULL) {
      fprintf( stderr, "JIT for TPP bwd all_zero_wp_kernel failed. Bailing...!\n");
      exit(-1);
    }
  }

  unary_shape            = libxsmm_create_meltw_unary_shape(res.bc, 1, ldo, ldo, res.datatype_comp, res.datatype_comp, res.datatype_comp);
  unary_flags            = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.helper_copy_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, unary_flags);
  if ( res.helper_copy_kernel == NULL) {
    fprintf( stderr, "JIT for TPP bwd helper_copy_kernel failed. Bailing...!\n");
    exit(-1);
  }

  binary_shape          = libxsmm_create_meltw_binary_shape(res.bc, 1, ldo, ldo, ldo, res.datatype_comp, res.datatype_comp, res.datatype_comp, res.datatype_comp);
  binary_flags          = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  res.helper_add_kernel = libxsmm_dispatch_meltw_binary_v2(LIBXSMM_MELTW_TYPE_BINARY_ADD, binary_shape, binary_flags);
  if ( res.helper_add_kernel == NULL) {
    fprintf( stderr, "JIT for TPP bwd helper_add_kernel failed. Bailing...!\n");
    exit(-1);
  }

  if (res.fuse_type == 1 || res.fuse_type == 3 || res.fuse_type == 4 || res.fuse_type == 5) {
    if (res.use_hw_blocking == 0)
      unary_shape       = libxsmm_create_meltw_unary_shape(res.bc, res.W / res.num_W_blocks, ldo, ldo, res.datatype_in, res.datatype_out, res.datatype_comp);
    else
      unary_shape       = libxsmm_create_meltw_unary_shape(res.bc, res.H*res.W / res.num_HW_blocks, ldo, ldo, res.datatype_in, res.datatype_out, res.datatype_comp);
    unary_flags         = ( (res.fuse_type == 4 || res.fuse_type == 5) ? LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT : LIBXSMM_MELTW_FLAG_UNARY_NONE);
    res.inv_relu_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_RELU_INV, unary_shape, unary_flags);
    if ( res.inv_relu_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP bwd inv_relu_kernel failed. Bailing...!\n");
      exit(-1);
    }
  }

  if (res.fuse_type == 2 || res.fuse_type == 3 || res.fuse_type == 5) {
    if (res.use_hw_blocking == 0)
      unary_shape         = libxsmm_create_meltw_unary_shape(res.bc, res.W / res.num_W_blocks, ldo, ldo, res.datatype_in, res.datatype_out, res.datatype_comp);
    else
      unary_shape         = libxsmm_create_meltw_unary_shape(res.bc, res.H*res.W / res.num_HW_blocks, ldo, ldo, res.datatype_in, res.datatype_out, res.datatype_comp);
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
  if (res.use_hw_blocking == 0)
    arg_shape[0].n    = res.W /res.num_W_blocks;
  else
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
  if (res.use_hw_blocking == 0)
    arg_shape[3].n    = res.W /res.num_W_blocks;
  else
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
  if (res.use_hw_blocking == 0)
    arg_shape[0].n    = res.W /res.num_W_blocks;
  else
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
  if (res.use_hw_blocking == 0)
    arg_shape[1].n    = res.W /res.num_W_blocks;
  else
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
  if (res.use_hw_blocking == 0)
    arg_shape[2].n    = res.W /res.num_W_blocks;
  else
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
  if (res.use_hw_blocking == 0)
    eqn_out_arg_shape.n    = res.W /res.num_W_blocks;
  else
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

LIBXSMM_API void destroy_libxsmm_dnn_bn_fwd(libxsmm_dnn_bn_fwd_config* cfg) {
  libxsmm_barrier_destroy(cfg->barrier);

  /* when/if libxsmm_matrix_eqn_destroy gets added, destructords for equations should go here */
}

LIBXSMM_API void destroy_libxsmm_dnn_bn_bwd(libxsmm_dnn_bn_bwd_config* cfg) {
  libxsmm_barrier_destroy(cfg->barrier);

  /* when/if libxsmm_matrix_eqn_destroy gets added, destructords for equations should go here */
}

LIBXSMM_API void libxsmm_dnn_bn_fwd_exec_f32( libxsmm_dnn_bn_fwd_config cfg, const float *pinp, const float *pinp_add, const float *pgamma, const float *pbeta, float *mean, float *var, float *pout,
                         unsigned char *prelumask, float eps, int start_tid, int my_tid, void *scratch, libxsmm_dnn_bn_norm_type norm_type ) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint H  = cfg.H;
  const libxsmm_blasint W  = cfg.W;
  const libxsmm_blasint HW = cfg.H * cfg.W;
  const libxsmm_blasint bc = cfg.bc;
  const libxsmm_blasint num_HW_blocks = cfg.num_HW_blocks;
  const libxsmm_blasint num_W_blocks  = cfg.num_W_blocks;

  const libxsmm_blasint hi_start      = cfg.pad_h_in;
  const libxsmm_blasint wi_start      = cfg.pad_w_in;
  const libxsmm_blasint ifhp = cfg.H + 2 * cfg.pad_h_in;
  const libxsmm_blasint ifwp = cfg.W + 2 * cfg.pad_w_in;

  const libxsmm_blasint ho_start      = cfg.pad_h_out;
  const libxsmm_blasint ho_end        = ho_start + cfg.H;
  const libxsmm_blasint wo_start      = cfg.pad_w_out;
  const libxsmm_blasint wo_end        = wo_start + cfg.W;
  const libxsmm_blasint ofhp = cfg.H + 2 * cfg.pad_h_out;
  const libxsmm_blasint ofwp = cfg.W + 2 * cfg.pad_w_out;

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

  LIBXSMM_VLA_DECL(5, const float,         inp ,     pinp     + (hi_start * ifwp + wi_start) * bc, CP, ifhp, ifwp, bc);      /* [N, CP, ifhp, ifwp, bc] + "padding" offset */
  LIBXSMM_VLA_DECL(5, const float,         inp_add,  pinp_add + (hi_start * ifwp + wi_start) * bc, CP, ifhp, ifwp, bc);      /* [N, CP, ifhp, ifwp, bc] + "padding" offset */
  LIBXSMM_VLA_DECL(5,       float,         out,      pout,      CP, ofhp, ofwp, bc);                                         /* [N, CP, ofhp, ofwp, bc] */
  LIBXSMM_VLA_DECL(5,       unsigned char, relumask, prelumask, CP, ofhp, ofwp, bc/BITS_PER_CHAR);                           /* [N, CP, ofhp, ofwp, bc/BITS_PER_CHAR] */

  LIBXSMM_VLA_DECL(2, const float,         gamma,    pgamma, bc);                   /* [CP, bc] */
  LIBXSMM_VLA_DECL(2, const float,         beta,     pbeta, bc);                    /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float,         mean,     mean,  bc);                    /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float,         var,      var,   bc);                    /* [CP, bc] */

  const float scale = 1.0f /((float)N * HW);

  LIBXSMM_VLA_DECL(3, float, sum_X_X2, ((float*)scratch), CP, bc);                  /* [2, CP, bc] */
  LIBXSMM_ASSUME_ALIGNED(sum_X_X2_, 64);
  const libxsmm_blasint sum_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + CP * 2 * bc), 64) - ((uintptr_t)(scratch))) / sizeof(float);
  LIBXSMM_VLA_DECL(3, float, sum_N, ((float*)scratch) + sum_N_offset, N, bc);       /* [CP, N, bc] */
  LIBXSMM_ASSUME_ALIGNED(sum_N_, 64);
  const libxsmm_blasint sumsq_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + sum_N_offset + CP * N * bc), 64) - ((uintptr_t)(scratch))) / sizeof(float);
  LIBXSMM_VLA_DECL(3, float, sumsq_N, ((float*)scratch) + sumsq_N_offset, N, bc);   /* [CP, N, bc] */
  LIBXSMM_ASSUME_ALIGNED(sumsq_N_, 64);

  libxsmm_meltw_unary_param  all_zero_param;
  libxsmm_meltw_binary_param add_param;
  libxsmm_meltw_unary_param  reduce_param;
  libxsmm_meltw_unary_param  all_relu_param;

  libxsmm_matrix_arg arg_array[6];

  libxsmm_matrix_eqn_param eqn_param;

  memset( &all_zero_param,  0, sizeof(all_zero_param));
  memset( &add_param,       0, sizeof(add_param));
  memset( &reduce_param,    0, sizeof(reduce_param));
  memset( &all_relu_param,  0, sizeof(all_relu_param));

  memset( &eqn_param,       0, sizeof(eqn_param));

  LIBXSMM_ALIGNED(float s[bc], 64);
  LIBXSMM_ALIGNED(float b[bc], 64);
  int n, cp;

  int cpxnt;
  if (norm_type == LIBXSMM_DNN_BN_FULL_NORM) {

    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      n  = cpxnt%N;
      cp = cpxnt/N;

      int hi, w, wb, hwb;

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
      if (cfg.use_hw_blocking == 0) { /* w-blocking */
        reduce_param.out.primary = lcl_sum_X_X2;                                                    /* [2*bc]  */
        for (hi = 0; hi < H; hi++) {
          for (wb = 0; wb < num_W_blocks; wb++) {
            reduce_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp, n, cp, hi, wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
            cfg.reduce_kernel(&reduce_param);                                                       /* [HW, bc] -----> [2 * bc] */

            add_param.in0.primary = sum_ncp_ptr;
            add_param.in1.primary = lcl_sum_X_X2;
            add_param.out.primary = sum_ncp_ptr;
            cfg.helper_add_kernel(&add_param);

            add_param.in0.primary = sumsq_ncp_ptr;
            add_param.in1.primary = &lcl_sum_X_X2[bc];
            add_param.out.primary = sumsq_ncp_ptr;
            cfg.helper_add_kernel(&add_param);
          }
        }
      } else { /* hw-blocking (implies no padding) */
        reduce_param.out.primary = lcl_sum_X_X2;                                                   /* [2*bc]  */
        for(hwb=0; hwb < num_HW_blocks; hwb++){
          hi = (hwb*(HW/num_HW_blocks))/W;
          w  = (hwb*(HW/num_HW_blocks))%W;
          reduce_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp, n, cp, hi, w, 0, CP, H, W, bc);
          cfg.reduce_kernel(&reduce_param);                                                       /* [HW, bc] -----> [2 * bc] */

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
        } /* loop over hw blocks */
      } /* if-else for the presence of input padding */
    } /* loop over cpxnt for temporary arrays */

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
    } /* loop over cp for computing mean and var */

    libxsmm_barrier_wait(cfg.barrier, ltid);

  } /* mean and var computation are for the full norm only */

  for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
    n  = cpxnt%N;
    cp = cpxnt/N;

    int hi, ho, w, wb, hwb, cb;

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

    if (cfg.use_hw_blocking == 0) { /* w-blocking */
      /* zeroing out strip [0, ho_start) x ofwp x bc */
      if (cfg.pad_h_out != 0) {
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, out, n, cp, 0, 0, 0, CP, ofhp, ofwp, bc);
        cfg.all_zero_hp_kernel(&all_zero_param);
      }
      for (hi = 0, ho = ho_start; hi < H; hi++, ho++) {
        /* zeroing out starting [0, wo_start) x bc and [wo_end, ofwp] x bc blocks for fixed ho */
        if (cfg.pad_w_out != 0) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, out, n, cp, ho, 0, 0, CP, ofhp, ofwp, bc);
          cfg.all_zero_wp_kernel(&all_zero_param);
        }
        for (wb = 0; wb < num_W_blocks; wb++) {
          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp, n, cp, hi, wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);             /* [bw, bc] */
          eqn_param.inputs = arg_array;
          eqn_param.output.primary   = &LIBXSMM_VLA_ACCESS(5, out, n, cp, ho, wo_start + wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);   /* [bw, bc] */

          if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU ||  cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
            arg_array[5].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp_add, n, cp, hi, wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);       /* [bw, bc] */
          }

          if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
            eqn_param.output.secondary = ((cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                            (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, wo_start + wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, (bc/BITS_PER_CHAR)) : NULL );
          }
          cfg.func10(&eqn_param);                                                   /* Normalization equation + relu + eltwise -> y = relu( ((s*x + b)*gamma + beta) + inp_add) */
        }
        /* zeroing out ending [wo_end, ofwp] x bc block for fixed ho */
        if (cfg.pad_w_out != 0) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, out, n, cp, ho, wo_end, 0, CP, ofhp, ofwp, bc);
          cfg.all_zero_wp_kernel(&all_zero_param);
        }
      }
      /* zeroing out strip [ho_end, ofhp) x ofwp x bc */
      if (cfg.pad_h_out != 0) {
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, out, n, cp, ho_end, 0, 0, CP, ofhp, ofwp, bc);
        cfg.all_zero_hp_kernel(&all_zero_param);
      }

    } else { /* hw-blocking (implies no padding) */
      for(hwb=0; hwb < num_HW_blocks; hwb++){
        hi = (hwb*(HW/num_HW_blocks))/W;
        ho = hi;
        w  = (hwb*(HW/num_HW_blocks))%W;
        arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp, n, cp, hi, w, 0, CP, H, W, bc);          /* [HW, bc] */
        eqn_param.inputs = arg_array;
        eqn_param.output.primary   = &LIBXSMM_VLA_ACCESS(5, out, n, cp, hi, w, 0, CP, H, W, bc);           /* [HW,bc] */

        if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU ||  cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
          arg_array[5].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp_add, n, cp, ho, w, 0, CP, H, W, bc);    /* [HW, bc] */
        }

        if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
          eqn_param.output.secondary = ((cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                          (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, w, 0, CP, H, W, (bc/BITS_PER_CHAR)) : NULL );
        }
        cfg.func10(&eqn_param);                                                   /* Normalization equation + relu + eltwise -> y = relu( ((s*x + b)*gamma + beta) + inp_add) */
      }
    } /* if-else for the presence of padding */
  } /* loop over cpxnt for computing din */

  libxsmm_barrier_wait(cfg.barrier, ltid);
}

LIBXSMM_API void libxsmm_dnn_bn_fwd_exec_bf16( libxsmm_dnn_bn_fwd_config cfg, const libxsmm_bfloat16 *pinp, const libxsmm_bfloat16 *pinp_add,
                          const float *pgamma, const float *pbeta, float *mean, float *var, libxsmm_bfloat16 *pout, unsigned char *prelumask,
                          float eps, int start_tid, int my_tid, void *scratch, libxsmm_dnn_bn_norm_type norm_type ) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint H  = cfg.H;
  const libxsmm_blasint W  = cfg.W;
  const libxsmm_blasint HW = cfg.H * cfg.W;
  const libxsmm_blasint bc = cfg.bc;
  const libxsmm_blasint num_HW_blocks = cfg.num_HW_blocks;
  const libxsmm_blasint num_W_blocks  = cfg.num_W_blocks;

  const libxsmm_blasint hi_start      = cfg.pad_h_in;
  const libxsmm_blasint wi_start      = cfg.pad_w_in;
  const libxsmm_blasint ifhp = cfg.H + 2 * cfg.pad_h_in;
  const libxsmm_blasint ifwp = cfg.W + 2 * cfg.pad_w_in;

  const libxsmm_blasint ho_start      = cfg.pad_h_out;
  const libxsmm_blasint ho_end        = ho_start + cfg.H;
  const libxsmm_blasint wo_start      = cfg.pad_w_out;
  const libxsmm_blasint wo_end        = wo_start + cfg.W;
  const libxsmm_blasint ofhp = cfg.H + 2 * cfg.pad_h_out;
  const libxsmm_blasint ofwp = cfg.W + 2 * cfg.pad_w_out;

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

  LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16,   inp ,     pinp     + (hi_start * ifwp + wi_start) * bc, CP, ifhp, ifwp, bc);      /* [N, CP, ifhp, ifwp, bc] + "padding" offset */
  LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16,   inp_add,  pinp_add + (hi_start * ifwp + wi_start) * bc, CP, ifhp, ifwp, bc);      /* [N, CP, ifhp, ifwp, bc] + "padding" offset */
  LIBXSMM_VLA_DECL(5,       libxsmm_bfloat16,   out,      pout,      CP, ofhp, ofwp, bc);                                         /* [N, CP, ofhp, ofwp, bc] */
  LIBXSMM_VLA_DECL(5,       unsigned char,      relumask, prelumask, CP, ofhp, ofwp, bc/BITS_PER_CHAR);                           /* [N, CP, ofhp, ofwp, bc/BITS_PER_CHAR] */

  LIBXSMM_VLA_DECL(2, const float,         gamma,    pgamma, bc);                    /* [CP, bc] */
  LIBXSMM_VLA_DECL(2, const float,         beta,     pbeta, bc);                     /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float,         mean,     mean,  bc);                     /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float,         var,      var,   bc);                     /* [CP, bc] */

  libxsmm_meltw_unary_param  all_zero_param;
  libxsmm_meltw_binary_param add_param;
  libxsmm_meltw_unary_param  all_relu_param;

  libxsmm_matrix_arg arg_array[6];

  libxsmm_matrix_eqn_param eqn_param;

  memset( &all_zero_param,  0, sizeof(all_zero_param));
  memset( &add_param,       0, sizeof(add_param));
  memset( &all_relu_param,  0, sizeof(all_relu_param));
  memset( &eqn_param, 0, sizeof(eqn_param));

  LIBXSMM_ALIGNED(float s[bc], 64);
  LIBXSMM_ALIGNED(float b[bc], 64);
  int n, cp;

  int cpxnt;
  if (norm_type == LIBXSMM_DNN_BN_FULL_NORM) {

    const float scale = 1.0f /((float)N * HW);

    LIBXSMM_VLA_DECL(3, float, sum_X_X2, ((float*)scratch), CP, bc);                 /* [2, CP, bc] */
    LIBXSMM_ASSUME_ALIGNED(sum_X_X2_, 64);
    const libxsmm_blasint sum_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + CP * 2 * bc), 64) - ((uintptr_t)(scratch))) / sizeof(float);
    LIBXSMM_VLA_DECL(3, float, sum_N, ((float*)scratch) + sum_N_offset, N, bc);      /* [CP, N, bc] */
    LIBXSMM_ASSUME_ALIGNED(sum_N_, 64);
    const libxsmm_blasint sumsq_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + sum_N_offset + CP * N * bc), 64) - ((uintptr_t)(scratch))) / sizeof(float);
    LIBXSMM_VLA_DECL(3, float, sumsq_N, ((float*)scratch) + sumsq_N_offset, N, bc);  /* [CP, N, bc] */
    LIBXSMM_ASSUME_ALIGNED(sumsq_N_, 64);

    libxsmm_meltw_unary_param  reduce_param;

    memset( &reduce_param,  0, sizeof(reduce_param));

    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      n  = cpxnt%N;
      cp = cpxnt/N;

      int hi, w, wb, hwb;

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
      if (cfg.use_hw_blocking == 0) { /* w-blocking */
        reduce_param.out.primary = lcl_sum_X_X2;                                                   /* [2*bc]  */
        for (hi = 0; hi < H; hi++) {
          for (wb = 0; wb < num_W_blocks; wb++) {
            reduce_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp, n, cp, hi, wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
            cfg.reduce_kernel(&reduce_param);                                                      /* [HW, bc] -----> [2 * bc] */

            add_param.in0.primary = sum_ncp_ptr;
            add_param.in1.primary = lcl_sum_X_X2;
            add_param.out.primary = sum_ncp_ptr;
            cfg.helper_add_kernel(&add_param);

            add_param.in0.primary = sumsq_ncp_ptr;
            add_param.in1.primary = &lcl_sum_X_X2[bc];
            add_param.out.primary = sumsq_ncp_ptr;
            cfg.helper_add_kernel(&add_param);
          }
        }
      } else { /* hw-blocking (implies no padding) */
        reduce_param.out.primary = lcl_sum_X_X2;                                                   /* [2*bc]  */
        for(hwb=0; hwb < num_HW_blocks; hwb++){
          hi = (hwb*(HW/num_HW_blocks))/W;
          w  = (hwb*(HW/num_HW_blocks))%W;
          reduce_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp, n, cp, hi, w, 0, CP, H, W, bc);
          cfg.reduce_kernel(&reduce_param);                                                       /* [HW, bc] -----> [2 * bc] */

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
        } /* loop over hw blocks */
      } /* if-else for the presence of input padding */
    } /* loop over cpxnt for temporary arrays */

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
    } /* loop over cp for computing mean and var */

    libxsmm_barrier_wait(cfg.barrier, ltid);
  } /* mean and var computation are for the full norm only */

  for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
    n  = cpxnt%N;
    cp = cpxnt/N;

    int hi, ho, w, wb, hwb, cb;

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

    if (cfg.use_hw_blocking == 0) { /* w-blocking */
      /* zeroing out strip [0, ho_start) x ofwp x bc */
      if (cfg.pad_h_out != 0) {
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, out, n, cp, 0, 0, 0, CP, ofhp, ofwp, bc);
        cfg.all_zero_hp_kernel(&all_zero_param);
      }
      for (hi = 0, ho = ho_start; hi < H; hi++, ho++) {
        /* zeroing out starting [0, wo_start) x bc block for fixed ho */
        if (cfg.pad_w_out != 0) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, out, n, cp, ho, 0, 0, CP, ofhp, ofwp, bc);
          cfg.all_zero_wp_kernel(&all_zero_param);
        }
        for (wb = 0; wb < num_W_blocks; wb++) {
          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp, n, cp, hi, wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);             /* [bw, bc] */
          eqn_param.inputs = arg_array;
          eqn_param.output.primary   = &LIBXSMM_VLA_ACCESS(5, out, n, cp, ho, wo_start + wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);   /* [bw, bc] */

          if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU ||  cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
            arg_array[5].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp_add, n, cp, hi, wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);       /* [bw, bc] */
          }

          if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
            eqn_param.output.secondary = ((cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                            (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, wo_start + wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, (bc/BITS_PER_CHAR)) : NULL );
          }
          cfg.func10(&eqn_param);                                                   /* Normalization equation + relu + eltwise -> y = relu( ((s*x + b)*gamma + beta) + inp_add) */
        }
        /* zeroing out ending [wo_end, ofwp] x bc block for fixed ho */
        if (cfg.pad_w_out != 0) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, out, n, cp, ho, wo_end, 0, CP, ofhp, ofwp, bc);
          cfg.all_zero_wp_kernel(&all_zero_param);
        }
      }
      /* zeroing out strip [ho_end, ofhp) x ofwp x bc */
      if (cfg.pad_h_out != 0) {
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, out, n, cp, ho_end, 0, 0, CP, ofhp, ofwp, bc);
        cfg.all_zero_hp_kernel(&all_zero_param);
      }
    } else { /* hw-blocking (implies no padding) */
      for(hwb=0; hwb < num_HW_blocks; hwb++){
        hi = (hwb*(HW/num_HW_blocks))/W;
        ho = hi;
        w  = (hwb*(HW/num_HW_blocks))%W;
        arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp, n, cp, hi, w, 0, CP, H, W, bc);          /* [HW, bc] */
        eqn_param.inputs = arg_array;
        eqn_param.output.primary   = &LIBXSMM_VLA_ACCESS(5, out, n, cp, hi, w, 0, CP, H, W, bc);           /* [HW,bc] */

        if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU ||  cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
          arg_array[5].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp_add, n, cp, ho, w, 0, CP, H, W, bc);    /* [HW, bc] */
        }

        if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
          eqn_param.output.secondary = ((cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                          (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, w, 0, CP, H, W, (bc/BITS_PER_CHAR)) : NULL );
        }
        cfg.func10(&eqn_param);                                                   /* Normalization equation + relu + eltwise -> y = relu( ((s*x + b)*gamma + beta) + inp_add) */
      }
    } /* if-else for the presence of padding */
  } /* loop over cpxnt for computing din */

  libxsmm_barrier_wait(cfg.barrier, ltid);
}


LIBXSMM_API void libxsmm_dnn_bn_bwd_exec_f32( libxsmm_dnn_bn_bwd_config cfg, float *pdout, const float *pinp, const float *mean, const float *var, const float *pgamma, const unsigned char *prelumask,
                         float *pdin, float *pdin_add, float *pdgamma, float *pdbeta, float eps,
                         int start_tid, int my_tid, void *scratch, libxsmm_dnn_bn_norm_type norm_type) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;

  const libxsmm_blasint H  = cfg.H;
  const libxsmm_blasint W  = cfg.W;
  const libxsmm_blasint HW = cfg.H * cfg.W;
  const libxsmm_blasint bc = cfg.bc;
  const libxsmm_blasint num_HW_blocks = cfg.num_HW_blocks;
  const libxsmm_blasint num_W_blocks  = cfg.num_W_blocks;

  const libxsmm_blasint hi_start      = cfg.pad_h_in;
  const libxsmm_blasint hi_end        = hi_start + cfg.H;
  const libxsmm_blasint wi_start      = cfg.pad_w_in;
  const libxsmm_blasint wi_end        = cfg.W + cfg.pad_w_in;
  const libxsmm_blasint ifhp = cfg.H + 2 * cfg.pad_h_in;
  const libxsmm_blasint ifwp = cfg.W + 2 * cfg.pad_w_in;

  const libxsmm_blasint ho_start      = cfg.pad_h_out;
  const libxsmm_blasint wo_start      = cfg.pad_w_out;
  /* const libxsmm_blasint wo_end        = wo_start + cfg.W; */
  const libxsmm_blasint ofhp = cfg.H + 2 * cfg.pad_h_out;
  const libxsmm_blasint ofwp = cfg.W + 2 * cfg.pad_w_out;

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

  LIBXSMM_VLA_DECL(5,       float,          din,      pdin,     CP, ifhp, ifwp, bc);    /* [N, CP, ifhp, ifwp, bc] */
  LIBXSMM_VLA_DECL(5,       float,          din_add,  pdin_add, CP, ifhp, ifwp, bc);    /* [N, CP, ifhp, ifwp, bc] */
  LIBXSMM_VLA_DECL(5, const float,          inp,      pinp,     CP, ifhp, ifwp, bc);    /* [N, CP, ifhp, ifwp, bc] */
  LIBXSMM_VLA_DECL(5,       float,          dout,     pdout     + (ho_start * ofwp + wo_start) * bc, CP, ofhp, ofwp, bc);                              /* [N, CP, ofhp, ofwp, bc] */
  LIBXSMM_VLA_DECL(5, const unsigned char, relumask , prelumask + (ho_start * ofwp + wo_start) * bc/BITS_PER_CHAR, CP, ofhp, ofwp, bc/BITS_PER_CHAR);  /* [N, CP, ofhp, ofwp, bc/BITS_PER_CHAR] */

  LIBXSMM_VLA_DECL(2, const float, gamma,   pgamma,  bc);                /* [CP, bc] */
  LIBXSMM_VLA_DECL(2, const float, mean,    mean,    bc);                /* [CP, bc] */
  LIBXSMM_VLA_DECL(2, const float, var,     var,     bc);                /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float, dgamma,  pdgamma, bc);                /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float, dbeta,   pdbeta,  bc);                /* [CP, bc] */

  float alpha = 0.0f;

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

  LIBXSMM_ALIGNED(float a[bc], 64); /* could also get moved into the scratch but left on the private stack as these are small, same below */
  LIBXSMM_ALIGNED(float b[bc], 64);
  LIBXSMM_ALIGNED(float c[bc], 64);
  int cpxnt;

  memset( &all_zero_param,   0, sizeof(all_zero_param));
  memset( &add_param,        0, sizeof(add_param));
  memset( &copy_param,       0, sizeof(copy_param));
  memset( &all_relu_param,   0, sizeof(all_relu_param));
  memset( &ewise_copy_param, 0, sizeof(ewise_copy_param));

  libxsmm_matrix_arg arg_array[8];
  libxsmm_matrix_eqn_param eqn_param;

  memset( &eqn_param,        0, sizeof(eqn_param));

  if (norm_type == LIBXSMM_DNN_BN_FULL_NORM) {
    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      int hi = 0, ho = 0, w = 0, wb = 0, hwb = 0, cb = 0;
      int n  = cpxnt%N;
      int cp = cpxnt/N;
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

      if (cfg.use_hw_blocking == 0) { /* w-blocking */
        /* Reminder: dout and relumask are already shifted by the offset (= point to the non-padded part already),
           while the other arrays are non-shifted (and hence accesses require offsets */
        /* Notice: Zeroing out the rim for din_add is not strictly necessary but for safety is done here */
        /* zeroing out strip [0, hi_start) */
        if (cfg.pad_h_in != 0 &&
              (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK)
            ) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, 0, 0, 0, CP, ifhp, ifwp, bc);
          cfg.all_zero_hp_kernel(&all_zero_param);
        }
        for (ho = 0, hi = hi_start; ho < H; ho++, hi++) {
          /* zeroing out starting [0, wi_start) x bc block for fixed hi */
          if (cfg.pad_w_in != 0 &&
              (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK)
              ) {
            all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, 0, 0, CP, ifhp, ifwp, bc);
            cfg.all_zero_wp_kernel(&all_zero_param);
          }
          for (wb = 0; wb < num_W_blocks; wb++) {
            if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE ||
              cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
              if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
                all_relu_param.op.primary   = (void*)(&alpha);
                all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);      /* [HW,bc] */
                all_relu_param.in.secondary = ((cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                                 (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc/8)
                                                 : NULL );
                all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);      /* [HW,bc] */
                cfg.inv_relu_kernel(&all_relu_param);
              } /* ReLU/mask */
              if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
                ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(5, dout,    n, cp, ho,            wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);
                ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
                cfg.ewise_copy_kernel(&ewise_copy_param);
              } /* Eltwise */
            }
            arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
            arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho,            wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);

            eqn_param.inputs = arg_array;
            eqn_param.output.primary = lcl_dgamma_ptr;
            cfg.dgamma_func(&eqn_param);                                                             /* dgamma += (a * inp + b) * dout */

            eqn_param.output.primary = lcl_dbeta_ptr;
            cfg.dbeta_func(&eqn_param);                                                              /* dbeta += dout */
          }

          /* zeroing out ending [wi_end, ifwp] x bc block for fixed hi */
          if (cfg.pad_w_in != 0 &&
              (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK)
              ) {
            all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, wi_end, 0, CP, ifhp, ifwp, bc);
            cfg.all_zero_wp_kernel(&all_zero_param);
          }

        }
        /* zeroing out strip [hi_end, ifhp) x ifwp x bc */
        if (cfg.pad_h_in != 0 &&
              (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK)
            ) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi_end, 0, 0, CP, ifhp, ifwp, bc);
          cfg.all_zero_hp_kernel(&all_zero_param);
        }

      } else { /* hw-blocking (implies no padding) */
        for(hwb=0; hwb < num_HW_blocks; hwb++){
          ho = (hwb*(HW/num_HW_blocks))/W;
          hi = ho;
          w  = (hwb*(HW/num_HW_blocks))%W;
          if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE ||
            cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
            if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
              all_relu_param.op.primary   = (void*)(&alpha);
              all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);      /* [HW,bc] */
              all_relu_param.in.secondary = ((cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                               (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, w, 0, CP, H, W, bc/8)
                                               : NULL /*&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc) */ ); /* dout_fwd ? nonsense? */
              all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);      /* [HW,bc] */
              cfg.inv_relu_kernel(&all_relu_param);
            } /* ReLU/mask */
            if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
              ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(5, dout,    n, cp, ho, w, 0, CP, H, W, bc);
              ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, w, 0, CP, H, W, bc);
              cfg.ewise_copy_kernel(&ewise_copy_param);
            } /* Eltwise */
          }
          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, w, 0, CP, H, W, bc);
          arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);

          eqn_param.inputs = arg_array;
          eqn_param.output.primary = lcl_dgamma_ptr;
          cfg.dgamma_func(&eqn_param);                                                             /* dgamma += (a * inp + b) * dout */

          eqn_param.output.primary = lcl_dbeta_ptr;
          cfg.dbeta_func(&eqn_param);                                                              /* dbeta += dout */
        } /* loop over hw blocks */
      } /* if-else for the presence of input padding */

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
    } /* loop over cpxnt for computing temporary n-local dbeta and dgamma */

    libxsmm_barrier_wait(cfg.barrier, ltid);

    {
      int cp = 0;
      int ni = 0;
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
      } /* loop over cp and nt for computing dbeta and dgamma */
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);

  } /* this is only computed in case of full backward (norm_type ~ 0) */

  for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
    int hi = 0, ho = 0, w = 0, wb = 0, hwb = 0, cb = 0;
    int n  = cpxnt%N;
    int cp = cpxnt/N;


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

    if (cfg.use_hw_blocking == 0) { /* w-blocking */
      /* Reminder: dout and relumask are already shifted by the offset (= point to the non-padded part already),
         while the other arrays are non-shifted (and hence accesses require offsets */
      /* Notice: Zeroing out the rim for din is not strictly necessary but for safety is done here */
      /* zeroing out strip [0, hi_start) x ifwp x bc */
      if (cfg.pad_h_in != 0) {
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, 0, 0, 0, CP, ifhp, ifwp, bc);
        cfg.all_zero_hp_kernel(&all_zero_param);
      }
      for (ho = 0, hi = hi_start; ho < H; ho++, hi++) {
        /* zeroing out starting [0, wi_start) x bc block for fixed hi */
        if (cfg.pad_w_in != 0) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, hi, 0, 0, CP, ifhp, ifwp, bc);
          cfg.all_zero_wp_kernel(&all_zero_param);
        }
        for (wb = 0; wb < num_W_blocks; wb++) {
          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp , n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
          arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho,            wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);

          eqn_param.inputs = arg_array;
          eqn_param.output.primary =    &LIBXSMM_VLA_ACCESS(5, din , n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
          cfg.din_func(&eqn_param);                                                                     /* din = dout * a + b * inp + c */
        }
        /* zeroing out ending [wi_end, ifwp] x bc block for fixed hi */
        if (cfg.pad_w_in != 0) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, hi, wi_end, 0, CP, ifhp, ifwp, bc);
          cfg.all_zero_wp_kernel(&all_zero_param);
        }
      }
      /* zeroing out strip [hi_end, ifhp) x ifwp x bc */
      if (cfg.pad_h_in != 0) {
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, hi_end, 0, 0, CP, ifhp, ifwp, bc);
        cfg.all_zero_hp_kernel(&all_zero_param);
      }
    } else { /* hw-blocking (implies no padding) */
      for(hwb=0; hwb < num_HW_blocks; hwb++){
        ho = (hwb*(HW/num_HW_blocks))/W;
        hi = ho;
        w  = (hwb*(HW/num_HW_blocks))%W;
        arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp , n, cp, hi, w, 0, CP, H, W, bc);
        arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);

        eqn_param.inputs = arg_array;
        eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, hi, w, 0, CP, H, W, bc);
        cfg.din_func(&eqn_param);                                                                     /* din = dout * a + b * inp + c */
      } /* loop over hw blocks */
    } /* if-else for the presence of input padding */
  } /* loop over cpxnt for computing din */

  libxsmm_barrier_wait(cfg.barrier, ltid);
}


LIBXSMM_API void libxsmm_dnn_bn_bwd_exec_bf16( libxsmm_dnn_bn_bwd_config cfg, libxsmm_bfloat16 *pdout, const libxsmm_bfloat16 *pinp, const float *mean, const float *var, const float *pgamma, const unsigned char *prelumask,
                         libxsmm_bfloat16 *pdin, libxsmm_bfloat16 *pdin_add, float *pdgamma, float *pdbeta, float eps,
                         int start_tid, int my_tid, void *scratch, libxsmm_dnn_bn_norm_type norm_type) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;

  const libxsmm_blasint H  = cfg.H;
  const libxsmm_blasint W  = cfg.W;
  const libxsmm_blasint HW = cfg.H * cfg.W;
  const libxsmm_blasint bc = cfg.bc;
  const libxsmm_blasint num_HW_blocks = cfg.num_HW_blocks;
  const libxsmm_blasint num_W_blocks  = cfg.num_W_blocks;

  const libxsmm_blasint hi_start      = cfg.pad_h_in;
  const libxsmm_blasint hi_end        = hi_start + cfg.H;
  const libxsmm_blasint wi_start      = cfg.pad_w_in;
  const libxsmm_blasint wi_end        = cfg.W + cfg.pad_w_in;
  const libxsmm_blasint ifhp = cfg.H + 2 * cfg.pad_h_in;
  const libxsmm_blasint ifwp = cfg.W + 2 * cfg.pad_w_in;

  const libxsmm_blasint ho_start      = cfg.pad_h_out;
  const libxsmm_blasint wo_start      = cfg.pad_w_out;
  const libxsmm_blasint ofhp = cfg.H + 2 * cfg.pad_h_out;
  const libxsmm_blasint ofwp = cfg.W + 2 * cfg.pad_w_out;

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

  LIBXSMM_VLA_DECL(5,       libxsmm_bfloat16, din,     pdin,     CP, ifhp, ifwp, bc);    /* [N, CP, ifhp, ifwp, bc] */
  LIBXSMM_VLA_DECL(5,       libxsmm_bfloat16, din_add, pdin_add, CP, ifhp, ifwp, bc);    /* [N, CP, ifhp, ifwp, bc] */
  LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16, inp,     pinp,     CP, ifhp, ifwp, bc);    /* [N, CP, ifhp, ifwp, bc] */
  LIBXSMM_VLA_DECL(5,       libxsmm_bfloat16, dout,    pdout   + (ho_start * ofwp + wo_start) * bc, CP, ofhp, ofwp, bc);                /* [N, CP, ofhp, ofwp, bc] */
  LIBXSMM_VLA_DECL(5, const unsigned char, relumask, prelumask + (ho_start * ofwp + wo_start) * bc/BITS_PER_CHAR, CP, ofhp, ofwp, bc/BITS_PER_CHAR);  /* [N, CP, ofhp, ofwp, bc/BITS_PER_CHAR] */

  LIBXSMM_VLA_DECL(2, const float,            gamma,  pgamma,  bc);                /* [CP, bc] */
  LIBXSMM_VLA_DECL(2, const float,            mean,   mean,    bc);                /* [CP, bc] */
  LIBXSMM_VLA_DECL(2, const float,            var,    var,     bc);                /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float,            dgamma, pdgamma, bc);                /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float,            dbeta,  pdbeta,  bc);                /* [CP, bc] */

  float alpha = 0.0f;
  int cpxnt = 0;

  libxsmm_meltw_unary_param  all_zero_param;
  LIBXSMM_ALIGNED(float a[bc], 64); /* could also get moved into the scratch but left on the private stack as these are small, same below */
  LIBXSMM_ALIGNED(float b[bc], 64);
  LIBXSMM_ALIGNED(float c[bc], 64);

  libxsmm_matrix_arg arg_array[8];
  libxsmm_matrix_eqn_param eqn_param;

  memset( &all_zero_param,   0, sizeof(all_zero_param));
  memset( &eqn_param,        0, sizeof(eqn_param));

  if (norm_type == LIBXSMM_DNN_BN_FULL_NORM) {

    const libxsmm_blasint dbeta_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + CP * N * bc), 64) - ((uintptr_t)(scratch))) / sizeof(float);
    LIBXSMM_VLA_DECL(3, float, dgamma_N, ((float*)scratch),                  N, bc);  /* [CP, N, bc] */
    LIBXSMM_VLA_DECL(3, float, dbeta_N,  ((float*)scratch) + dbeta_N_offset, N, bc);  /* [CP, N, bc] */
    LIBXSMM_ASSUME_ALIGNED(dgamma_N_, 64);
    LIBXSMM_ASSUME_ALIGNED(dbeta_N_,  64);

    libxsmm_meltw_binary_param add_param;
    libxsmm_meltw_unary_param  copy_param;
    libxsmm_meltw_unary_param  all_relu_param;
    libxsmm_meltw_unary_param  ewise_copy_param;

    memset( &add_param,        0, sizeof(add_param));
    memset( &copy_param,       0, sizeof(copy_param));
    memset( &all_relu_param,   0, sizeof(all_relu_param));
    memset( &ewise_copy_param, 0, sizeof(ewise_copy_param));

    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      int n  = cpxnt%N;
      int cp = cpxnt/N;

      int hi = 0, ho = 0, w = 0, wb = 0, hwb = 0, cb = 0;

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

      if (cfg.use_hw_blocking == 0) { /* w-blocking */
        /* Reminder: dout and relumask are already shifted by the offset (= point to the non-padded part already),
           while the other arrays are non-shifted (and hence accesses require offsets */
        /* Notice: Zeroing out the rim for din_add is not strictly necessary but for safety is done here */
        /* zeroing out strip [0, hi_start) */
        if (cfg.pad_h_in != 0 &&
              (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK)
            ) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, 0, 0, 0, CP, ifhp, ifwp, bc);
          cfg.all_zero_hp_kernel(&all_zero_param);
        }
        for (ho = 0, hi = hi_start; ho < H; ho++, hi++) {
          /* zeroing out starting [0, wi_start) x bc block for fixed hi */
          if (cfg.pad_w_in != 0 &&
              (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK)
              ) {
            all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, 0, 0, CP, ifhp, ifwp, bc);
            cfg.all_zero_wp_kernel(&all_zero_param);
          }
          for (wb = 0; wb < num_W_blocks; wb++) {
            if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE ||
              cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
              if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
                all_relu_param.op.primary   = (void*)(&alpha);
                all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);      /* [HW,bc] */
                all_relu_param.in.secondary = ((cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                                 (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc/8)
                                                 : NULL );
                all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);      /* [HW,bc] */
                cfg.inv_relu_kernel(&all_relu_param);
              } /* ReLU/mask */
              if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
                ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(5, dout,    n, cp, ho,            wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);
                ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
                cfg.ewise_copy_kernel(&ewise_copy_param);
              } /* Eltwise */
            }
            arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
            arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho,            wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);

            eqn_param.inputs = arg_array;
            eqn_param.output.primary = lcl_dgamma_ptr;
            cfg.dgamma_func(&eqn_param);                                                             /* dgamma += (a * inp + b) * dout */

            eqn_param.output.primary = lcl_dbeta_ptr;
            cfg.dbeta_func(&eqn_param);                                                              /* dbeta += dout */
          }
          /* zeroing out ending [wi_end, ifwp] x bc block for fixed hi */
          if (cfg.pad_w_in != 0 &&
              (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK)
              ) {
            all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, wi_end, 0, CP, ifhp, ifwp, bc);
            cfg.all_zero_wp_kernel(&all_zero_param);
          }

        }
        /* zeroing out strip [hi_end, ifhp) x ifwp x bc */
        if (cfg.pad_h_in != 0 &&
              (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK)
            ) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi_end, 0, 0, CP, ifhp, ifwp, bc);
          cfg.all_zero_hp_kernel(&all_zero_param);
        }

      } else { /* hw-blocking (implies no padding) */
        for(hwb=0; hwb < num_HW_blocks; hwb++){
          ho = (hwb*(HW/num_HW_blocks))/W;
          hi = hi;
          w  = (hwb*(HW/num_HW_blocks))%W;
          if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE ||
            cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
            if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
              all_relu_param.op.primary   = (void*)(&alpha);
              all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);      /* [HW,bc] */
              all_relu_param.in.secondary = ((cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                               (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, w, 0, CP, H, W, bc/8)
                                               : NULL /*&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc) */ ); /* dout_fwd ? nonsense? */
              all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);      /* [HW,bc] */
              cfg.inv_relu_kernel(&all_relu_param);
            } /* ReLU/mask */
            if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
              ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(5, dout,    n, cp, ho, w, 0, CP, H, W, bc);
              ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, w, 0, CP, H, W, bc);
              cfg.ewise_copy_kernel(&ewise_copy_param);
            } /* Eltwise */
          }
          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, w, 0, CP, H, W, bc);
          arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);

          eqn_param.inputs = arg_array;
          eqn_param.output.primary = lcl_dgamma_ptr;
          cfg.dgamma_func(&eqn_param);                                                             /* dgamma += (a * inp + b) * dout */

          eqn_param.output.primary = lcl_dbeta_ptr;
          cfg.dbeta_func(&eqn_param);                                                              /* dbeta += dout */
        } /* loop over hw blocks */
      } /* if-else for the presence of input padding */

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
    } /* loop over cpxnt for computing temporary n-local dbeta and dgamma */

    libxsmm_barrier_wait(cfg.barrier, ltid);

    {
      int cp = 0;
      int ni = 0;
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
      } /* loops over cp and nt for computing dbeta and dgamma */
    }
    libxsmm_barrier_wait(cfg.barrier, ltid);

  } /* this is only computed in case of full backward (norm_type ~ 0) */

  for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
    int n  = cpxnt%N;
    int cp = cpxnt/N;

    int hi = 0, ho = 0, w = 0, wb = 0, hwb = 0, cb = 0;

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

    if (cfg.use_hw_blocking == 0) { /* w-blocking */
      /* Reminder: dout and relumask are already shifted by the offset (= point to the non-padded part already),
         while the other arrays are non-shifted (and hence accesses require offsets */
        /* Notice: Zeroing out the rim for din is not strictly necessary but for safety is done here */
      /* zeroing out strip [0, hi_start) x ifwp x bc */
      if (cfg.pad_h_in != 0) {
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, 0, 0, 0, CP, ifhp, ifwp, bc);
        cfg.all_zero_hp_kernel(&all_zero_param);
      }
      for (ho = 0, hi = hi_start; ho < H; ho++, hi++) {
        /* zeroing out starting [0, wi_start) x bc block for fixed hi */
        if (cfg.pad_w_in != 0) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, hi, 0, 0, CP, ifhp, ifwp, bc);
          cfg.all_zero_wp_kernel(&all_zero_param);
        }
        for (wb = 0; wb < num_W_blocks; wb++) {
          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp , n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
          arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho,            wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);

          eqn_param.inputs = arg_array;
          eqn_param.output.primary =    &LIBXSMM_VLA_ACCESS(5, din , n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
          cfg.din_func(&eqn_param);                                                                     /* din = dout * a + b * inp + c */
        }
        /* zeroing out ending [wi_end, ifwp] x bc block for fixed hi */
        if (cfg.pad_w_in != 0) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, hi, wi_end, 0, CP, ifhp, ifwp, bc);
          cfg.all_zero_wp_kernel(&all_zero_param);
        }
      }
      /* zeroing out strip [hi_end, ifhp) x ifwp x bc */
      if (cfg.pad_h_in != 0) {
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, hi_end, 0, 0, CP, ifhp, ifwp, bc);
        cfg.all_zero_hp_kernel(&all_zero_param);
      }
    } else { /* hw-blocking (implies no padding) */
      for(hwb=0; hwb < num_HW_blocks; hwb++){
        ho = (hwb*(HW/num_HW_blocks))/W;
        hi = hi;
        w  = (hwb*(HW/num_HW_blocks))%W;
        arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp , n, cp, hi, w, 0, CP, H, W, bc);
        arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);

        eqn_param.inputs = arg_array;
        eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, hi, w, 0, CP, H, W, bc);
        cfg.din_func(&eqn_param);                                                                     /* din = dout * a + b * inp + c */
      }
    } /* if-else for the presence of input padding */
  } /* loop over cpxnt for computing dbeta and dgamma */

  libxsmm_barrier_wait(cfg.barrier, ltid);
}
