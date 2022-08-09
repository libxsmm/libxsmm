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

#include <libxsmm_dnn_fusedgn.h>

#define BITS_PER_CHAR (8)

LIBXSMM_API libxsmm_dnn_gn_fwd_config setup_libxsmm_dnn_gn_fwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint G, libxsmm_blasint bc,
                                 libxsmm_blasint pad_h_in, libxsmm_blasint pad_w_in, libxsmm_blasint pad_h_out, libxsmm_blasint pad_w_out,
                                 libxsmm_blasint threads, libxsmm_dnn_gn_fuse fuse_type,
                                 libxsmm_datatype datatype_in, libxsmm_datatype datatype_out, libxsmm_datatype datatype_comp ) {

  libxsmm_dnn_gn_fwd_config res;

  libxsmm_blasint ldo = bc;
  libxsmm_blasint ld  = bc;
  libxsmm_blasint tmp_ld, tmp_ld2;
  libxsmm_blasint my_eqn10;
  libxsmm_blasint group_size;

  libxsmm_meltw_unary_shape  unary_shape;
  libxsmm_meltw_binary_shape binary_shape;

  libxsmm_bitfield  unary_flags;
  libxsmm_bitfield  binary_flags;
  libxsmm_bitfield  ternary_flags;

  libxsmm_meqn_arg_shape  eqn_out_arg_shape;
  libxsmm_meqn_arg_shape  arg_shape[128];

  libxsmm_matrix_arg_attributes arg_singular_attr;

  libxsmm_matrix_eqn_arg_metadata arg_metadata[128];
  libxsmm_matrix_eqn_op_metadata  op_metadata[128];

  arg_singular_attr.type = LIBXSMM_MATRIX_ARG_TYPE_SINGULAR;

  memset( &res,  0, sizeof(res));

  /* setting up some handle values */
  res.N             = N;
  res.C             = C;
  res.G             = G;
  res.H             = H;
  res.W             = W;
  res.bc            = bc;
  res.CP            = res.C / res.bc;
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

  res.datatype_in   = datatype_in;
  res.datatype_out  = datatype_out;
  res.datatype_comp = datatype_comp;

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* TPP creation */
  ldo = res.G;
  unary_shape           = libxsmm_get_meltw_unary_shape(res.G, 1, res.G, ldo, res.datatype_comp, res.datatype_comp, res.datatype_comp);
  unary_flags           = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.all_zero_G_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
  if ( res.all_zero_G_kernel == NULL) {
    fprintf( stderr, "JIT for initialization by unary all zero group copy kernel failed for fwd. Bailing...!\n");
    exit(-1);
  }

  ldo = res.bc;
  unary_shape           = libxsmm_get_meltw_unary_shape(res.bc, 1, res.bc, ldo, res.datatype_comp, res.datatype_comp, res.datatype_comp);
  unary_flags           = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.all_zero_kernel   = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
  if ( res.all_zero_G_kernel == NULL) {
    fprintf( stderr, "JIT for initialization by unary all zero copy kernel failed for fwd. Bailing...!\n");
    exit(-1);
  }

  if (res.pad_h_out != 0) {
    libxsmm_blasint ofwp   = res.W + 2 * res.pad_w_out;
    unary_flags            = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    unary_shape            = libxsmm_get_meltw_unary_shape(res.bc, (res.pad_h_out * ofwp), res.bc, ldo, res.datatype_out, res.datatype_out, res.datatype_comp);
    res.all_zero_hp_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
    if ( res.all_zero_hp_kernel == NULL) {
      fprintf( stderr, "JIT for TPP fwd all_zero_hp_kernel failed. Bailing...!\n");
      exit(-1);
    }
  }

  if (res.pad_w_out != 0) {
    unary_flags            = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    unary_shape            = libxsmm_get_meltw_unary_shape(res.bc, res.pad_w_out, res.bc, ldo, res.datatype_out, res.datatype_out, res.datatype_comp);
    res.all_zero_wp_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
    if ( res.all_zero_wp_kernel == NULL) {
      fprintf( stderr, "JIT for TPP fwd all_zero_wp_kernel failed. Bailing...!\n");
      exit(-1);
    }
  }

  /* TPPs for reducing X and X2 in HW*/
  ld = res.bc;
  tmp_ld = res.bc;

  if (res.use_hw_blocking == 0)
    unary_shape     = libxsmm_get_meltw_unary_shape(res.bc, res.W / res.num_W_blocks, ld, tmp_ld, res.datatype_in, res.datatype_comp, res.datatype_comp);
  else
    unary_shape     = libxsmm_get_meltw_unary_shape(res.bc, res.H*res.W / res.num_HW_blocks, ld, tmp_ld, res.datatype_in, res.datatype_comp, res.datatype_comp);
  unary_flags        = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  res.reduce_kernel  = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD, unary_shape, unary_flags);
  if ( res.reduce_kernel == NULL) {
      fprintf( stderr, "JIT for initialization of reduce_kernel failed for fwd. Bailing...!\n");
      exit(-1);
  }

  binary_shape   = libxsmm_get_meltw_binary_shape(res.bc, 1, ld, ld, ld, res.datatype_comp, res.datatype_comp, res.datatype_comp, res.datatype_comp);
  binary_flags   = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  res.add_kernel = libxsmm_dispatch_meltw_binary_v2(LIBXSMM_MELTW_TYPE_BINARY_ADD, binary_shape, binary_flags);
  if ( res.add_kernel == NULL) {
      fprintf( stderr, "JIT for initialization of add_kernel failed for fwd. Bailing...!\n");
      exit(-1);
  }

  /* TPP for reducing groups */
  group_size = res.C/res.G;

  ld = group_size;                /* group_size = (CP*bc)/G */
  tmp_ld = 1;

  unary_shape              = libxsmm_get_meltw_unary_shape(group_size, 1, ld, tmp_ld, res.datatype_comp, res.datatype_comp, res.datatype_comp);
  unary_flags              = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS;
  res.reduce_groups_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, unary_shape, unary_flags);
  if ( res.reduce_groups_kernel == NULL) {
      fprintf( stderr, "JIT for initialization of reduce_groups_kernel failed for fwd. Bailing...!\n");
      exit(-1);
  }

  ld = res.bc;
  tmp_ld = 1;
  unary_shape            = libxsmm_get_meltw_unary_shape(res.bc, 1, ld, tmp_ld, res.datatype_comp, res.datatype_comp, res.datatype_comp);
  unary_flags            = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS;
  res.reduce_rows_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, unary_shape, unary_flags);
  if ( res.reduce_rows_kernel == NULL) {
      fprintf( stderr, "JIT for initialization of reduce_rows_kernel failed for fwd. Bailing...!\n");
      exit(-1);
  }

  /* TPP equation for forward */
  ld = res.bc;
  tmp_ld = 1;
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
    arg_shape[0].n  = res.W / res.num_W_blocks;
  else
    arg_shape[0].n  = res.H*res.W / res.num_HW_blocks;
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
      arg_shape[5].n  = res.W / res.num_W_blocks;
    else
      arg_shape[5].n  = res.H*res.W / res.num_HW_blocks;
    arg_shape[5].ld   = ld;
    arg_shape[5].type = res.datatype_in;
    libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[5], arg_shape[5], arg_singular_attr);
  }

  eqn_out_arg_shape.m    = res.bc;                                 /* y = [HW, bc] */
  if (res.use_hw_blocking == 0)
    eqn_out_arg_shape.n  = res.W / res.num_W_blocks;
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

  /* init scratch (currently is not needed for the groupnorm fwd) */
  res.scratch_size = 0;

  return res;
}

LIBXSMM_API libxsmm_dnn_gn_bwd_config setup_libxsmm_dnn_gn_bwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint G, libxsmm_blasint bc,
                                 libxsmm_blasint pad_h_in, libxsmm_blasint pad_w_in, libxsmm_blasint pad_h_out, libxsmm_blasint pad_w_out,
                                 libxsmm_blasint threads, libxsmm_dnn_gn_fuse fuse_type,
                                 libxsmm_datatype datatype_in, libxsmm_datatype datatype_out, libxsmm_datatype datatype_comp ) {

  libxsmm_dnn_gn_bwd_config res;

  size_t dbeta_N_offset;

  libxsmm_blasint ldo = bc;
  libxsmm_blasint ld  = bc;
  libxsmm_blasint tmp_ld2;
  libxsmm_blasint my_eqn11, my_eqn12, my_eqn13, my_eqn14, my_eqn15;

  libxsmm_meltw_unary_shape  unary_shape;

  libxsmm_bitfield unary_flags;
  libxsmm_bitfield binary_flags;
  libxsmm_bitfield ternary_flags;

  libxsmm_meqn_arg_shape  eqn_out_arg_shape;
  libxsmm_meqn_arg_shape  arg_shape[128];

  libxsmm_matrix_arg_attributes arg_singular_attr;

  libxsmm_matrix_eqn_arg_metadata arg_metadata[128];
  libxsmm_matrix_eqn_op_metadata  op_metadata[128];

  arg_singular_attr.type = LIBXSMM_MATRIX_ARG_TYPE_SINGULAR;

  memset( &res,  0, sizeof(res));

  /* setting up some handle values */
  res.N             = N;
  res.C             = C;
  res.G             = G;
  res.H             = H;
  res.W             = W;
  res.bc            = bc;
  res.CP            = res.C / res.bc;
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

  res.datatype_in   = datatype_in;
  res.datatype_out  = datatype_out;
  res.datatype_comp = datatype_comp;

  /* when masking is on, bc must be divisible by 8 for compressing mask into char array (otherwise strides are wrong for relumask */
  if ( (res.fuse_type == 4 || res.fuse_type == 5) && (res.bc % BITS_PER_CHAR != 0)) {
    fprintf( stderr, "bc = %d is not divisible by BITS_PER_CHAR = %d. Bailing...!\n", res.bc, BITS_PER_CHAR);
    exit(-1);
  }

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  ldo = res.bc;
  unary_shape         = libxsmm_get_meltw_unary_shape(res.bc, 1, res.bc, ldo, res.datatype_comp, res.datatype_comp, res.datatype_comp);
  unary_flags         = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.all_zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
  if ( res.all_zero_kernel == NULL) {
    fprintf( stderr, "JIT for initialization by unary all zero copy kernel failed for fwd. Bailing...!\n");
    exit(-1);
  }

  if (res.pad_h_in != 0) {
    libxsmm_blasint ifwp   = res.W + 2 * res.pad_w_in;
    unary_flags            = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    unary_shape            = libxsmm_get_meltw_unary_shape(res.bc, (res.pad_h_in * ifwp), res.bc, ldo, res.datatype_in, res.datatype_in, res.datatype_comp);
    res.all_zero_hp_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
    if ( res.all_zero_hp_kernel == NULL) {
      fprintf( stderr, "JIT for TPP bwd all_zero_hp_kernel failed. Bailing...!\n");
      exit(-1);
    }
  }

  if (res.pad_w_in != 0) {
    unary_flags            = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    unary_shape            = libxsmm_get_meltw_unary_shape(res.bc, res.pad_w_in, res.bc, ldo, res.datatype_in, res.datatype_in, res.datatype_comp);
    res.all_zero_wp_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
    if ( res.all_zero_wp_kernel == NULL) {
      fprintf( stderr, "JIT for TPP bwd all_zero_wp_kernel failed. Bailing...!\n");
      exit(-1);
    }
  }

  if (res.fuse_type == 1 || res.fuse_type == 3 || res.fuse_type == 4 || res.fuse_type == 5) {
    if (res.use_hw_blocking == 0)
      unary_shape       = libxsmm_get_meltw_unary_shape(res.bc, res.W / res.num_W_blocks, ldo, ldo, res.datatype_in, res.datatype_out, res.datatype_comp);
    else
      unary_shape       = libxsmm_get_meltw_unary_shape(res.bc, res.H*res.W / res.num_HW_blocks, ldo, ldo, res.datatype_in, res.datatype_out, res.datatype_comp);
    unary_flags         = ( (res.fuse_type == 4 || res.fuse_type == 5) ? LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT : LIBXSMM_MELTW_FLAG_UNARY_NONE);
    res.inv_relu_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_RELU_INV, unary_shape, unary_flags);
    if ( res.inv_relu_kernel == NULL ) {
      fprintf( stderr, "JIT for TPP bwd inv_relu_kernel failed. Bailing...!\n");
      exit(-1);
    }
  }

  if (res.fuse_type == 2 || res.fuse_type == 3 || res.fuse_type == 5) {
    if (res.use_hw_blocking == 0)
      unary_shape         = libxsmm_get_meltw_unary_shape(res.bc, res.W / res.num_W_blocks, ldo, ldo, res.datatype_in, res.datatype_out, res.datatype_comp);
    else
      unary_shape         = libxsmm_get_meltw_unary_shape(res.bc, res.H*res.W / res.num_HW_blocks, ldo, ldo, res.datatype_in, res.datatype_out, res.datatype_comp);
    unary_flags           = LIBXSMM_MELTW_FLAG_UNARY_NONE;
    res.ewise_copy_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, unary_flags);
    if ( res.ewise_copy_kernel == NULL) {
      fprintf( stderr, "JIT for TPP bwd ewise_copy_kernel failed. Bailing...!\n");
      exit(-1);
    }
  }

  /* Group norm equations */
  /* Create MatEq for bwd layernorm */

  ld = res.bc;
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
  arg_shape[0].m              = res.bc;                            /* inp [HW, bc] */
  if (res.use_hw_blocking == 0)
    arg_shape[0].n            = res.W / res.num_W_blocks;
  else
    arg_shape[0].n            = res.H*res.W / res.num_HW_blocks;
  arg_shape[0].ld             = ld;
  arg_shape[0].type           = res.datatype_in;
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
    arg_shape[3].n  = res.W / res.num_W_blocks;
  else
    arg_shape[3].n  = res.H*res.W / res.num_HW_blocks;
  arg_shape[3].ld   = ld;
  arg_shape[3].type = res.datatype_in;
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
    fprintf( stderr, "JIT for TPP fwd dgamma_func (eqn11) failed. Bailing...!\n");
    exit(-1);
  }

  /* dbeta function  */
  my_eqn12 = libxsmm_matrix_eqn_create();                          /* dbeta [bc] = dout [HW, bc] + dbeta [bc] */

  binary_flags                = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  op_metadata[0].eqn_idx      = my_eqn12;
  op_metadata[0].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_BINARY_ADD, res.datatype_comp, binary_flags);

  unary_flags                 = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  op_metadata[1].eqn_idx      = my_eqn12;
  op_metadata[1].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[1], LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, res.datatype_comp, unary_flags);

  arg_metadata[0].eqn_idx     = my_eqn12;
  arg_metadata[0].in_arg_pos  = 3;
  arg_shape[0].m    = res.bc;                                      /* dout [HW, bc] */
  if (res.use_hw_blocking == 0)
    arg_shape[0].n  = res.W / res.num_W_blocks;
  else
    arg_shape[0].n  = res.H*res.W / res.num_HW_blocks;
  arg_shape[0].ld   = ld;
  arg_shape[0].type = res.datatype_in;
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
    fprintf( stderr, "JIT for TPP fwd dbeta_func (eqn12) failed. Bailing...!\n");
    exit(-1);
  }

  /* db new equation */
  my_eqn13 = libxsmm_matrix_eqn_create();                          /* db [bc] = (dout * gamma) [HW, bc] + db [bc]*/

  binary_flags                = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  op_metadata[0].eqn_idx      = my_eqn13;
  op_metadata[0].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_BINARY_ADD, res.datatype_comp, binary_flags);

  unary_flags                 = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  op_metadata[1].eqn_idx      = my_eqn13;
  op_metadata[1].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[1], LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, res.datatype_comp, unary_flags);

  binary_flags                = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1;
  op_metadata[2].eqn_idx      = my_eqn13;
  op_metadata[2].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[2], LIBXSMM_MELTW_TYPE_BINARY_MUL, res.datatype_comp, binary_flags);

  arg_metadata[0].eqn_idx     = my_eqn13;
  arg_metadata[0].in_arg_pos  = 3;
  arg_shape[0].m    = res.bc;                                      /* dout [HW, bc] */
  if (res.use_hw_blocking == 0)
    arg_shape[0].n  = res.W / res.num_W_blocks;
  else
    arg_shape[0].n  = res.H*res.W / res.num_HW_blocks;
  arg_shape[0].ld   = ld;
  arg_shape[0].type = res.datatype_in;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[0], arg_shape[0], arg_singular_attr);

  arg_metadata[1].eqn_idx     = my_eqn13;
  arg_metadata[1].in_arg_pos  = 6;
  arg_shape[1].m    = res.bc;                                      /* gamma [bc] */
  arg_shape[1].n    = 1;
  arg_shape[1].ld   = tmp_ld2;
  arg_shape[1].type = res.datatype_comp;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[1], arg_shape[1], arg_singular_attr);

  arg_metadata[2].eqn_idx     = my_eqn13;
  arg_metadata[2].in_arg_pos  = 9;
  arg_shape[2].m    = res.bc;                                      /* db [bc] */
  arg_shape[2].n    = 1;
  arg_shape[2].ld   = tmp_ld2;
  arg_shape[2].type = res.datatype_comp;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[2], arg_shape[2], arg_singular_attr);

  eqn_out_arg_shape.m    = res.bc;                                 /* db [bc] */
  eqn_out_arg_shape.n    = 1;
  eqn_out_arg_shape.ld   = tmp_ld2;
  eqn_out_arg_shape.type = res.datatype_comp;

  /* libxsmm_matrix_eqn_tree_print( my_eqn13 ); */
  /* libxsmm_matrix_eqn_rpn_print ( my_eqn13 ); */

  res.db_func = libxsmm_dispatch_matrix_eqn_v2( my_eqn13, eqn_out_arg_shape );
  if ( res.db_func == NULL) {
    fprintf( stderr, "JIT for TPP fwd db_func (eqn13) failed. Bailing...!\n");
    exit(-1);
  }

  /* ds new equation */
  my_eqn14 = libxsmm_matrix_eqn_create();                          /* ds [bc] = ((dout * gamma) * inp) [HW, bc] + ds [bc] */

  binary_flags                = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  op_metadata[0].eqn_idx      = my_eqn14;
  op_metadata[0].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_BINARY_ADD, res.datatype_comp, binary_flags);

  unary_flags                 = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  op_metadata[1].eqn_idx      = my_eqn14;
  op_metadata[1].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[1], LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, res.datatype_comp, unary_flags);

  binary_flags                = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  op_metadata[2].eqn_idx      = my_eqn14;
  op_metadata[2].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[2], LIBXSMM_MELTW_TYPE_BINARY_MUL, res.datatype_comp, binary_flags);

  binary_flags                = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1;
  op_metadata[3].eqn_idx      = my_eqn14;
  op_metadata[3].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[3], LIBXSMM_MELTW_TYPE_BINARY_MUL, res.datatype_comp, binary_flags);

  arg_metadata[0].eqn_idx     = my_eqn14;
  arg_metadata[0].in_arg_pos  = 3;
  arg_shape[0].m    = res.bc;                                      /* dout [HW, bc] */
  if (res.use_hw_blocking == 0)
    arg_shape[0].n  = res.W / res.num_W_blocks;
  else
    arg_shape[0].n  = res.H*res.W / res.num_HW_blocks;
  arg_shape[0].ld   = ld;
  arg_shape[0].type = res.datatype_in;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[0], arg_shape[0], arg_singular_attr);

  arg_metadata[1].eqn_idx     = my_eqn14;
  arg_metadata[1].in_arg_pos  = 6;
  arg_shape[1].m    = res.bc;                                      /* gamma [bc] */
  arg_shape[1].n    = 1;
  arg_shape[1].ld   = tmp_ld2;
  arg_shape[1].type = res.datatype_comp;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[1], arg_shape[1], arg_singular_attr);

  arg_metadata[2].eqn_idx     = my_eqn14;
  arg_metadata[2].in_arg_pos  = 0;
  arg_shape[2].m              = res.bc;                            /* inp [HW, bc] */
  if (res.use_hw_blocking == 0)
    arg_shape[2].n            = res.W / res.num_W_blocks;
  else
    arg_shape[2].n            = res.H*res.W / res.num_HW_blocks;
  arg_shape[2].ld             = ld;
  arg_shape[2].type           = res.datatype_in;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[2], arg_shape[2], arg_singular_attr);

  arg_metadata[3].eqn_idx     = my_eqn14;
  arg_metadata[3].in_arg_pos  = 8;
  arg_shape[3].m    = res.bc;                                      /* ds [bc] */
  arg_shape[3].n    = 1;
  arg_shape[3].ld   = tmp_ld2;
  arg_shape[3].type = res.datatype_comp;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[3], arg_shape[3], arg_singular_attr);

  eqn_out_arg_shape.m    = res.bc;                                 /* ds [bc] */
  eqn_out_arg_shape.n    = 1;
  eqn_out_arg_shape.ld   = tmp_ld2;
  eqn_out_arg_shape.type = res.datatype_comp;

  /* libxsmm_matrix_eqn_tree_print( my_eqn14 ); */
  /* libxsmm_matrix_eqn_rpn_print ( my_eqn14 ); */

  res.ds_func = libxsmm_dispatch_matrix_eqn_v2( my_eqn14, eqn_out_arg_shape );
  if ( res.ds_func == NULL) {
    fprintf( stderr, "JIT for TPP fwd ds_func (eqn14) failed. Bailing...!\n");
    exit(-1);
  }

  /* din equation */
  my_eqn15 = libxsmm_matrix_eqn_create();                          /* din = ((gamma * a) * dout) + (inp * b + c) */

  ternary_flags               = LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT;
  op_metadata[0].eqn_idx      = my_eqn15;
  op_metadata[0].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_ternary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_TERNARY_MULADD, res.datatype_comp, ternary_flags);

  binary_flags                = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  op_metadata[2].eqn_idx      = my_eqn15;
  op_metadata[2].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[2], LIBXSMM_MELTW_TYPE_BINARY_MUL, res.datatype_comp, binary_flags);

  arg_metadata[0].eqn_idx     = my_eqn15;
  arg_metadata[0].in_arg_pos  = 6;
  arg_shape[0].m    = res.bc;                                      /* gamma [bc] */
  arg_shape[0].n    = 1;
  arg_shape[0].ld   = tmp_ld2;
  arg_shape[0].type = res.datatype_comp;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[0], arg_shape[0], arg_singular_attr);

  arg_metadata[1].eqn_idx     = my_eqn15;
  arg_metadata[1].in_arg_pos  = 1;
  arg_shape[1].m    = res.bc;                                      /* a [bc] */
  arg_shape[1].n    = 1;
  arg_shape[1].ld   = tmp_ld2;
  arg_shape[1].type = res.datatype_comp;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[1], arg_shape[1], arg_singular_attr);

  arg_metadata[2].eqn_idx     = my_eqn15;
  arg_metadata[2].in_arg_pos  = 3;
  arg_shape[2].m    = res.bc;                                      /* dout [HW, bc] */
  if (res.use_hw_blocking == 0)
    arg_shape[2].n  = res.W / res.num_W_blocks;
  else
    arg_shape[2].n  = res.H*res.W / res.num_HW_blocks;
  arg_shape[2].ld   = ld;
  arg_shape[2].type = res.datatype_in;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[2], arg_shape[2], arg_singular_attr);

  ternary_flags               = LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT;
  op_metadata[1].eqn_idx      = my_eqn15;
  op_metadata[1].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_ternary_op_v2(op_metadata[1], LIBXSMM_MELTW_TYPE_TERNARY_MULADD, res.datatype_comp, ternary_flags);

  arg_metadata[3].eqn_idx     = my_eqn15;
  arg_metadata[3].in_arg_pos  = 0;
  arg_shape[3].m              = res.bc;                            /* inp [HW, bc] */
  if (res.use_hw_blocking == 0)
    arg_shape[3].n            = res.W / res.num_W_blocks;
  else
    arg_shape[3].n            = res.H*res.W / res.num_HW_blocks;
  arg_shape[3].ld             = ld;
  arg_shape[3].type           = res.datatype_in;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[3], arg_shape[3], arg_singular_attr);

  arg_metadata[4].eqn_idx     = my_eqn15;
  arg_metadata[4].in_arg_pos  = 2;
  arg_shape[4].m    = res.bc;                                      /* b [bc] */
  arg_shape[4].n    = 1;
  arg_shape[4].ld   = tmp_ld2;
  arg_shape[4].type = res.datatype_comp;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[4], arg_shape[4], arg_singular_attr);

  arg_metadata[5].eqn_idx     = my_eqn15;
  arg_metadata[5].in_arg_pos  = 7;
  arg_shape[5].m    = res.bc;                                      /* c [bc] */
  arg_shape[5].n    = 1;
  arg_shape[5].ld   = tmp_ld2;
  arg_shape[5].type = res.datatype_comp;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[5], arg_shape[5], arg_singular_attr);

  eqn_out_arg_shape.m    = res.bc;                                 /* din [HW, bc] */
  if (res.use_hw_blocking == 0)
    eqn_out_arg_shape.n  = res.W / res.num_W_blocks;
  else
    eqn_out_arg_shape.n  = res.H*res.W / res.num_HW_blocks;
  eqn_out_arg_shape.ld   = ld;
  eqn_out_arg_shape.type = res.datatype_out;

  /* libxsmm_matrix_eqn_tree_print( my_eqn16 ); */
  /* libxsmm_matrix_eqn_rpn_print ( my_eqn16 ); */

  res.din_func = libxsmm_dispatch_matrix_eqn_v2( my_eqn15, eqn_out_arg_shape );
  if ( res.din_func == NULL) {
    fprintf( stderr, "JIT for TPP fwd din_func (eqn15) failed. Bailing...!\n");
    exit(-1);
  }

  /* init scratch */
  dbeta_N_offset = LIBXSMM_UP2(res.CP * res.N * res.bc, 64);
  res.scratch_size =  sizeof(float) * ( dbeta_N_offset /* dbeta_N*/ + LIBXSMM_UP2(res.CP * res.N * res.bc, 64) /*dgamma_N */ );

  return res;
}

LIBXSMM_API void destroy_libxsmm_dnn_gn_fwd(libxsmm_dnn_gn_fwd_config* cfg) {
  libxsmm_barrier_destroy(cfg->barrier);

  /* when/if libxsmm_matrix_eqn_destroy gets added, destructors for equations should go here */
}

LIBXSMM_API void destroy_libxsmm_dnn_gn_bwd(libxsmm_dnn_gn_bwd_config* cfg) {
  libxsmm_barrier_destroy(cfg->barrier);

}

LIBXSMM_API void libxsmm_dnn_gn_fwd_exec_f32( libxsmm_dnn_gn_fwd_config cfg, const float *pinp, const float *pinp_add, const float *pgamma, const float *pbeta, float *mean, float *var, float *pout, unsigned char *prelumask,
                         float eps, int start_tid, int my_tid, void *scratch ) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint G  = cfg.G;
  const libxsmm_blasint H  = cfg.H;
  const libxsmm_blasint W  = cfg.W;
  const libxsmm_blasint HW = cfg.H * cfg.W;
  const libxsmm_blasint bc = cfg.bc;
  const libxsmm_blasint num_HW_blocks = cfg.num_HW_blocks;
  const libxsmm_blasint num_W_blocks  = cfg.num_W_blocks;

  const libxsmm_blasint hi_start      = cfg.pad_h_in;
  const libxsmm_blasint wi_start      = cfg.pad_w_in;
  const libxsmm_blasint ifhp          = cfg.H + 2 * cfg.pad_h_in;
  const libxsmm_blasint ifwp          = cfg.W + 2 * cfg.pad_w_in;

  const libxsmm_blasint ho_start      = cfg.pad_h_out;
  const libxsmm_blasint ho_end        = ho_start + cfg.H;
  const libxsmm_blasint wo_start      = cfg.pad_w_out;
  const libxsmm_blasint wo_end        = wo_start + cfg.W;
  const libxsmm_blasint ofhp          = cfg.H + 2 * cfg.pad_h_out;
  const libxsmm_blasint ofwp          = cfg.W + 2 * cfg.pad_w_out;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;

  /* number of tasks that could be run in parallel for 1d blocking */
  /* Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here */
  const libxsmm_blasint work_dN = CP * N;
  /* compute chunk size */
  const libxsmm_blasint chunksize_dN = (work_dN % cfg.threads == 0) ?
    (work_dN / cfg.threads) : ((work_dN / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_dN = (ltid * chunksize_dN < work_dN) ? (ltid * chunksize_dN) : work_dN;
  const libxsmm_blasint thr_end_dN = ((ltid + 1) * chunksize_dN < work_dN) ? ((ltid + 1) * chunksize_dN) : work_dN;

  /* number of tasks that could be run in parallel for 1d blocking over N*/
  const libxsmm_blasint work_N = N;
  /* compute chunk size */
  const libxsmm_blasint chunksize_N = (work_N % cfg.threads == 0) ?
    (work_N / cfg.threads) : ((work_N / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_N = (ltid * chunksize_N < work_N) ? (ltid * chunksize_N) : work_N;
  const libxsmm_blasint thr_end_N = ((ltid + 1) * chunksize_N < work_N) ? ((ltid + 1) * chunksize_N) : work_N;

  LIBXSMM_VLA_DECL(5, const float,         inp ,     pinp     + (hi_start * ifwp + wi_start) * bc, CP, ifhp, ifwp, bc);      /* [N, CP, ifhp, ifwp, bc] + "padding" offset */
  LIBXSMM_VLA_DECL(5, const float,         inp_add,  pinp_add + (hi_start * ifwp + wi_start) * bc, CP, ifhp, ifwp, bc);      /* [N, CP, ifhp, ifwp, bc] + "padding" offset */
  LIBXSMM_VLA_DECL(5,       float,         out,      pout,      CP, ofhp, ofwp, bc);                                         /* [N, CP, ofhp, ofwp, bc] */
  LIBXSMM_VLA_DECL(5,       unsigned char, relumask, prelumask, CP, ofhp, ofwp, bc/BITS_PER_CHAR);                           /* [N, CP, ofhp, ofwp, bc/BITS_PER_CHAR] */

  LIBXSMM_VLA_DECL(2, const float,         gamma,    pgamma, bc);                   /* [CP, bc] */
  LIBXSMM_VLA_DECL(2, const float,         beta,     pbeta, bc);                    /* [CP, bc] */

  const int group_size = (CP*bc)/G;

  libxsmm_meltw_unary_param  all_zero_param;
  libxsmm_meltw_binary_param add_param;
  libxsmm_meltw_unary_param  reduce_param;
  libxsmm_meltw_unary_param  m_reduce_groups_param;
  libxsmm_meltw_unary_param  v_reduce_groups_param;
  libxsmm_meltw_unary_param  all_relu_param;

  libxsmm_matrix_arg arg_array[6];
  libxsmm_matrix_eqn_param eqn_param;

  LIBXSMM_UNUSED(scratch);

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  memset( &all_zero_param,        0, sizeof(all_zero_param));
  memset( &add_param,             0, sizeof(add_param));
  memset( &reduce_param,          0, sizeof(reduce_param));
  memset( &m_reduce_groups_param, 0, sizeof(m_reduce_groups_param));
  memset( &v_reduce_groups_param, 0, sizeof(v_reduce_groups_param));
  memset( &all_relu_param,        0, sizeof(all_relu_param));
  memset( &eqn_param,             0, sizeof(eqn_param));

  eqn_param.inputs = arg_array;

  if (group_size <= bc){
    int cp, n;
    int cpxnt;
    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      LIBXSMM_ALIGNED(float tmp[2*bc], 64);
      LIBXSMM_ALIGNED(float sum_X[G], 64);
      LIBXSMM_ALIGNED(float sum_X2[G], 64);
      LIBXSMM_ALIGNED(float s[bc], 64);
      LIBXSMM_ALIGNED(float b[bc], 64);

      int hi = 0, ho = 0, w = 0, wb = 0, hwb = 0;
      int i, j, g;

      n  = cpxnt/CP;
      cp = cpxnt%CP;

      all_zero_param.out.primary = tmp;
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &tmp[bc];
      cfg.all_zero_kernel(&all_zero_param);

      all_zero_param.out.primary = sum_X;
      cfg.all_zero_G_kernel(&all_zero_param);
      all_zero_param.out.primary = sum_X2;
      cfg.all_zero_G_kernel(&all_zero_param);

      LIBXSMM_ALIGNED(float new_tmp[2*bc], 64);

      reduce_param.out.primary   = new_tmp;                  /* [2*bc] */
      if (cfg.use_hw_blocking == 0) { /* w-blocking */
        for (hi = 0; hi < H; hi++) {
          for (wb = 0; wb < num_W_blocks; wb++) {
            reduce_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp, n, cp, hi, wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);      /* [HW_block, bc] -----> [2 * bc] */
            cfg.reduce_kernel(&reduce_param);

            add_param.in0.primary = tmp;
            add_param.in1.primary = new_tmp;
            add_param.out.primary = tmp;
            cfg.add_kernel(&add_param);

            add_param.in0.primary = &tmp[bc];
            add_param.in1.primary = &new_tmp[bc];
            add_param.out.primary = &tmp[bc];
            cfg.add_kernel(&add_param);

            /* for (cb = 0; cb < 2*bc; cb++) { */
            /*   tmp[cb] += new_tmp[cb]; */
            /* } */
          }
        }
      } else { /* hw-blocking (implies no padding) */
        for(hwb=0; hwb < num_HW_blocks; hwb++){
          hi = (hwb*(HW/num_HW_blocks))/W;
          w  = (hwb*(HW/num_HW_blocks))%W;
          reduce_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp, n, cp, hi, w, 0, CP, H, W, bc);      /* [HW_block, bc] -----> [2 * bc] */
          cfg.reduce_kernel(&reduce_param);

          add_param.in0.primary = tmp;
          add_param.in1.primary = new_tmp;
          add_param.out.primary = tmp;
          cfg.add_kernel(&add_param);

          add_param.in0.primary = &tmp[bc];
          add_param.in1.primary = &new_tmp[bc];
          add_param.out.primary = &tmp[bc];
          cfg.add_kernel(&add_param);

          /* for (cb = 0; cb < 2*bc; cb++) { */
          /*   tmp[cb] += new_tmp[cb]; */
          /* } */

        } /* loop over hw blocks */
      } /* if-else for the presence of input padding */

      for(i=0; i < bc; i += group_size){
        g = (cp*bc + i)/group_size;                                                                      /* determine current group */
        m_reduce_groups_param.in.primary    = &tmp[i];
        m_reduce_groups_param.out.primary   = &sum_X[g];
        v_reduce_groups_param.in.primary    = &tmp[bc + i];
        v_reduce_groups_param.out.primary   = &sum_X2[g];
        cfg.reduce_groups_kernel(&m_reduce_groups_param);
        cfg.reduce_groups_kernel(&v_reduce_groups_param);

        mean[n*G + g] = sum_X[g] / ((float)group_size * HW);
        var[n*G + g] = (sum_X2[g] / ((float)group_size * HW)) - (mean[n*G + g]*mean[n*G + g]);        /* var = E[X^2] - (E[X])^2 */

        for(j = 0; j < group_size; j++){
          s[i + j] = 1.0f / ((float)sqrt(var[n*G + g] + eps));                                        /* 1/sqrt(var(X) + eps) */
          b[i + j] = -1 * mean[n*G + g] * s[i + j];                                                   /* -E[X]/sqrt(var(X) + eps) */
        }
      }

      arg_array[1].primary = s;                                                                           /* [bc] */
      arg_array[2].primary = b;                                                                           /* [bc] */
      arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);                             /* [bc] */
      arg_array[4].primary = (void*)&LIBXSMM_VLA_ACCESS(2, beta,  cp, 0, bc);                             /* [bc] */

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

            if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU ||  cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
              arg_array[5].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp_add, n, cp, hi, wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);       /* [bw, bc] */
            }

            if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
              eqn_param.output.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
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
          eqn_param.output.primary    = &LIBXSMM_VLA_ACCESS(5, out, n, cp, ho, w, 0, CP, H, W, bc);          /* [HW,bc] */

          if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU ||  cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
            arg_array[5].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp_add, n, cp, hi, w, 0, CP, H, W, bc);    /* [HW, bc] */
          }

          if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
            eqn_param.output.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                            (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, w, 0, CP, H, W, (bc/BITS_PER_CHAR)) : NULL );
          }
          cfg.func10(&eqn_param);                                                   /* Normalization equation + relu + eltwise -> y = relu( ((s*x + b)*gamma + beta) + inp_add) */
        }
      } /* if-else for the presence of padding */
    } /* loop over cpxnt for computing out */
  } else {                                                         /* Case when group_size > bc */
    int n;
    for ( n = thr_begin_N; n < thr_end_N; ++n ) {
      LIBXSMM_ALIGNED(float tmp[2*bc], 64);
      LIBXSMM_ALIGNED(float sum_X[G], 64);
      LIBXSMM_ALIGNED(float sum_X2[G], 64);
      LIBXSMM_ALIGNED(float s[CP*bc], 64);
      LIBXSMM_ALIGNED(float b[CP*bc], 64);

      int hi = 0, ho = 0, w = 0, wb = 0, hwb = 0;
      int i, j, cp, g;
      float m, v;

      libxsmm_meltw_unary_param  m_reduce_rows_param;
      libxsmm_meltw_unary_param  v_reduce_rows_param;

      LIBXSMM_ALIGNED(float new_tmp[2*bc], 64);

      memset( &m_reduce_rows_param, 0, sizeof(m_reduce_rows_param));
      memset( &v_reduce_rows_param, 0, sizeof(v_reduce_rows_param));

      all_zero_param.out.primary = sum_X;
      cfg.all_zero_G_kernel(&all_zero_param);
      all_zero_param.out.primary = sum_X2;
      cfg.all_zero_G_kernel(&all_zero_param);

      for (cp = 0; cp < CP; cp++){                      /* [cp, HW, bc] */
        all_zero_param.out.primary = tmp;
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = &tmp[bc];
        cfg.all_zero_kernel(&all_zero_param);
        /* for (cb = 0; cb < 2*bc; cb++) { */
        /*   tmp[cb] = 0.0f; */
        /* } */

        reduce_param.out.primary   = new_tmp;                  /* [2*bc] */
        if (cfg.use_hw_blocking == 0) { /* w-blocking */
          for (hi = 0; hi < H; hi++) {
            for (wb = 0; wb < num_W_blocks; wb++) {
              reduce_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp, n, cp, hi, wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);      /* [HW_block, bc] -----> [2 * bc] */
              cfg.reduce_kernel(&reduce_param);

              add_param.in0.primary = tmp;
              add_param.in1.primary = new_tmp;
              add_param.out.primary = tmp;
              cfg.add_kernel(&add_param);

              add_param.in0.primary = &tmp[bc];
              add_param.in1.primary = &new_tmp[bc];
              add_param.out.primary = &tmp[bc];
              cfg.add_kernel(&add_param);

              /* for (cb = 0; cb < 2*bc; cb++) { */
              /*   tmp[cb] += new_tmp[cb]; */
              /* } */
            }
          }
        } else { /* hw-blocking (implies no padding) */
          for(hwb=0; hwb < num_HW_blocks; hwb++){
            hi = (hwb*(HW/num_HW_blocks))/W;
            w  = (hwb*(HW/num_HW_blocks))%W;
            reduce_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp, n, cp, hi, w, 0, CP, H, W, bc);      /* [HW_block, bc] -----> [2 * bc] */
            cfg.reduce_kernel(&reduce_param);

            add_param.in0.primary = tmp;
            add_param.in1.primary = new_tmp;
            add_param.out.primary = tmp;
            cfg.add_kernel(&add_param);

            add_param.in0.primary = &tmp[bc];
            add_param.in1.primary = &new_tmp[bc];
            add_param.out.primary = &tmp[bc];
            cfg.add_kernel(&add_param);

            /* for (cb = 0; cb < 2*bc; cb++) { */
            /*   tmp[cb] += new_tmp[cb]; */
            /* } */

          } /* loop over hw blocks */
        } /* if-else for the presence of input padding */

        if (group_size >= bc){                                 /* Group size >= block size  (Ex.- CP = 4, bc = 16, G = 2, group_size = 32) */
          g = (cp*bc)/group_size;                              /* determine current group */
          m_reduce_rows_param.in.primary    = tmp;
          m_reduce_rows_param.out.primary   = &m;
          v_reduce_rows_param.in.primary    = &tmp[bc];
          v_reduce_rows_param.out.primary   = &v;
          cfg.reduce_rows_kernel(&m_reduce_rows_param);
          cfg.reduce_rows_kernel(&v_reduce_rows_param);
          sum_X[g] += m;
          sum_X2[g] += v;
        }
        else{                                                 /* Group size < block size  (Ex.- CP = 4, bc = 16, G = 32, group_size = 2) */
          for(i=0; i < bc; i += group_size){
            m_reduce_groups_param.in.primary    = &tmp[i];
            m_reduce_groups_param.out.primary   = &sum_X[cp*(bc/group_size) + (i/group_size)];
            v_reduce_groups_param.in.primary    = &tmp[bc + i];
            v_reduce_groups_param.out.primary   = &sum_X2[cp*(bc/group_size) + (i/group_size)];
            cfg.reduce_groups_kernel(&m_reduce_groups_param);
            cfg.reduce_groups_kernel(&v_reduce_groups_param);
          }
        }
      }

      /* mean and variance calculation */
      for(g = 0; g < G; g++){
        mean[n*G + g] = sum_X[g] / ((float)group_size * HW);
        var[n*G + g] = (sum_X2[g] / ((float)group_size * HW)) - (mean[n*G + g]*mean[n*G + g]);        /* var = E[X^2] - (E[X])^2 */

        for(j = 0; j < group_size; j++){
          s[g*group_size + j] = 1.0f / ((float)sqrt(var[n*G + g] + eps));                             /* 1/sqrt(var(X) + eps) */
          b[g*group_size + j] = -1 * mean[n*G + g] * s[g*group_size + j];                             /* -E[X]/sqrt(var(X) + eps) */
        }
      }

      for (cp = 0; cp < CP; cp++){

        arg_array[1].primary = &s[cp*bc];                                                                /* [bc] */
        arg_array[2].primary = &b[cp*bc];                                                                /* [bc] */
        arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);                          /* [bc] */
        arg_array[4].primary = (void*)&LIBXSMM_VLA_ACCESS(2, beta, cp, 0, bc);                           /* [bc] */

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

              if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU ||  cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
                arg_array[5].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp_add, n, cp, hi, wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);       /* [bw, bc] */
              }

              if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
                eqn_param.output.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
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
            eqn_param.output.primary   =  &LIBXSMM_VLA_ACCESS(5, out, n, cp, ho, w, 0, CP, H, W, bc);          /* [HW,bc] */

            if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU ||  cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
              arg_array[5].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp_add, n, cp, hi, w, 0, CP, H, W, bc);    /* [HW, bc] */
            }

            if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
              eqn_param.output.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                              (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, w, 0, CP, H, W, (bc/BITS_PER_CHAR)) : NULL );
            }
            cfg.func10(&eqn_param);                                                   /* Normalization equation + relu + eltwise -> y = relu( ((s*x + b)*gamma + beta) + inp_add) */
          }
        } /* if-else for the presence of padding */
      } /* loop over cp for computing out */
    } /* loop over n */
  } /* if-else for the group_size */

  libxsmm_barrier_wait(cfg.barrier, ltid);

}

LIBXSMM_API void libxsmm_dnn_gn_fwd_exec_bf16( libxsmm_dnn_gn_fwd_config cfg, const libxsmm_bfloat16 *pinp, const libxsmm_bfloat16 *pinp_add,
                          const float *pgamma, const float *pbeta, float *mean, float *var,
                          libxsmm_bfloat16 *pout, unsigned char *prelumask,
                          float eps, int start_tid, int my_tid, void *scratch ) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint G  = cfg.G;
  const libxsmm_blasint H  = cfg.H;
  const libxsmm_blasint W  = cfg.W;
  const libxsmm_blasint HW = cfg.H * cfg.W;
  const libxsmm_blasint bc = cfg.bc;
  const libxsmm_blasint num_HW_blocks = cfg.num_HW_blocks;
  const libxsmm_blasint num_W_blocks  = cfg.num_W_blocks;

  const libxsmm_blasint hi_start      = cfg.pad_h_in;
  const libxsmm_blasint wi_start      = cfg.pad_w_in;
  const libxsmm_blasint ifhp          = cfg.H + 2 * cfg.pad_h_in;
  const libxsmm_blasint ifwp          = cfg.W + 2 * cfg.pad_w_in;

  const libxsmm_blasint ho_start      = cfg.pad_h_out;
  const libxsmm_blasint ho_end        = ho_start + cfg.H;
  const libxsmm_blasint wo_start      = cfg.pad_w_out;
  const libxsmm_blasint wo_end        = wo_start + cfg.W;
  const libxsmm_blasint ofhp          = cfg.H + 2 * cfg.pad_h_out;
  const libxsmm_blasint ofwp          = cfg.W + 2 * cfg.pad_w_out;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;

  /* number of tasks that could be run in parallel for 1d blocking */
  /* Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here */
  const libxsmm_blasint work_dN = CP * N;
  /* compute chunk size */
  const libxsmm_blasint chunksize_dN = (work_dN % cfg.threads == 0) ?
    (work_dN / cfg.threads) : ((work_dN / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_dN = (ltid * chunksize_dN < work_dN) ? (ltid * chunksize_dN) : work_dN;
  const libxsmm_blasint thr_end_dN = ((ltid + 1) * chunksize_dN < work_dN) ? ((ltid + 1) * chunksize_dN) : work_dN;

  /* number of tasks that could be run in parallel for 1d blocking over N*/
  const libxsmm_blasint work_N = N;
  /* compute chunk size */
  const libxsmm_blasint chunksize_N = (work_N % cfg.threads == 0) ?
    (work_N / cfg.threads) : ((work_N / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_N = (ltid * chunksize_N < work_N) ? (ltid * chunksize_N) : work_N;
  const libxsmm_blasint thr_end_N = ((ltid + 1) * chunksize_N < work_N) ? ((ltid + 1) * chunksize_N) : work_N;

  LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16, inp ,    pinp     + (hi_start * ifwp + wi_start) * bc, CP, ifhp, ifwp, bc);      /* [N, CP, ifhp, ifwp, bc] + "padding" offset */
  LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16, inp_add, pinp_add + (hi_start * ifwp + wi_start) * bc, CP, ifhp, ifwp, bc);      /* [N, CP, ifhp, ifwp, bc] + "padding" offset */
  LIBXSMM_VLA_DECL(5,       libxsmm_bfloat16,  out,    pout,      CP, ofhp, ofwp, bc);                                         /* [N, CP, ofhp, ofwp, bc] */
  LIBXSMM_VLA_DECL(5,       unsigned char, relumask,   prelumask, CP, ofhp, ofwp, bc/BITS_PER_CHAR);                           /* [N, CP, ofhp, ofwp, bc/BITS_PER_CHAR] */

  LIBXSMM_VLA_DECL(2, const float,         gamma,    pgamma, bc);                   /* [CP, bc] */
  LIBXSMM_VLA_DECL(2, const float,         beta,     pbeta, bc);                    /* [CP, bc] */

  const int group_size = (CP*bc)/G;

  libxsmm_meltw_unary_param  all_zero_param;
  libxsmm_meltw_binary_param add_param;
  libxsmm_meltw_unary_param  reduce_param;
  libxsmm_meltw_unary_param  m_reduce_groups_param;
  libxsmm_meltw_unary_param  v_reduce_groups_param;
  libxsmm_meltw_unary_param  all_relu_param;

  libxsmm_matrix_arg arg_array[6];
  libxsmm_matrix_eqn_param eqn_param;

  LIBXSMM_UNUSED(scratch);

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  memset( &all_zero_param,        0, sizeof(all_zero_param));
  memset( &add_param,             0, sizeof(add_param));
  memset( &reduce_param,          0, sizeof(reduce_param));
  memset( &m_reduce_groups_param, 0, sizeof(m_reduce_groups_param));
  memset( &v_reduce_groups_param, 0, sizeof(v_reduce_groups_param));
  memset( &all_relu_param,        0, sizeof(all_relu_param));
  memset( &eqn_param,             0, sizeof(eqn_param));

  eqn_param.inputs = arg_array;

  if (group_size <= bc){
    int n, cp;
    int cpxnt;
    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      LIBXSMM_ALIGNED(float tmp[2*bc], 64);
      LIBXSMM_ALIGNED(float sum_X[G], 64);
      LIBXSMM_ALIGNED(float sum_X2[G], 64);
      LIBXSMM_ALIGNED(float s[bc], 64);
      LIBXSMM_ALIGNED(float b[bc], 64);

      int hi = 0, ho = 0, w = 0, wb = 0, hwb = 0;
      int i, j, g;

      LIBXSMM_ALIGNED(float new_tmp[2*bc], 64);

      n  = cpxnt/CP;
      cp = cpxnt%CP;

      all_zero_param.out.primary = tmp;
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &tmp[bc];
      cfg.all_zero_kernel(&all_zero_param);

      all_zero_param.out.primary = sum_X;
      cfg.all_zero_G_kernel(&all_zero_param);
      all_zero_param.out.primary = sum_X2;
      cfg.all_zero_G_kernel(&all_zero_param);

      reduce_param.out.primary   = new_tmp;                  /* [2*bc] */
      if (cfg.use_hw_blocking == 0) { /* w-blocking */
        for (hi = 0; hi < H; hi++) {
          for (wb = 0; wb < num_W_blocks; wb++) {
            reduce_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp, n, cp, hi, wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);      /* [HW_block, bc] -----> [2 * bc] */
            cfg.reduce_kernel(&reduce_param);

            add_param.in0.primary = tmp;
            add_param.in1.primary = new_tmp;
            add_param.out.primary = tmp;
            cfg.add_kernel(&add_param);

            add_param.in0.primary = &tmp[bc];
            add_param.in1.primary = &new_tmp[bc];
            add_param.out.primary = &tmp[bc];
            cfg.add_kernel(&add_param);

            /* for (cb = 0; cb < 2*bc; cb++) { */
            /*   tmp[cb] += new_tmp[cb]; */
            /* } */
          }
        }
      } else { /* hw-blocking (implies no padding) */
        for(hwb=0; hwb < num_HW_blocks; hwb++){
          hi = (hwb*(HW/num_HW_blocks))/W;
          w  = (hwb*(HW/num_HW_blocks))%W;
          reduce_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp, n, cp, hi, w, 0, CP, H, W, bc);      /* [HW_block, bc] -----> [2 * bc] */
          cfg.reduce_kernel(&reduce_param);

          add_param.in0.primary = tmp;
          add_param.in1.primary = new_tmp;
          add_param.out.primary = tmp;
          cfg.add_kernel(&add_param);

          add_param.in0.primary = &tmp[bc];
          add_param.in1.primary = &new_tmp[bc];
          add_param.out.primary = &tmp[bc];
          cfg.add_kernel(&add_param);

          /* for (cb = 0; cb < 2*bc; cb++) { */
          /*   tmp[cb] += new_tmp[cb]; */
          /* } */

        } /* loop over hw blocks */
      } /* if-else for the presence of input padding */

      for(i=0; i < bc; i += group_size){
        g = (cp*bc + i)/group_size;                                                                      /* determine current group */
        m_reduce_groups_param.in.primary    = &tmp[i];
        m_reduce_groups_param.out.primary   = &sum_X[g];
        v_reduce_groups_param.in.primary    = &tmp[bc + i];
        v_reduce_groups_param.out.primary   = &sum_X2[g];
        cfg.reduce_groups_kernel(&m_reduce_groups_param);
        cfg.reduce_groups_kernel(&v_reduce_groups_param);

        mean[n*G + g] = sum_X[g] / ((float)group_size * HW);
        var[n*G + g] = (sum_X2[g] / ((float)group_size * HW)) - (mean[n*G + g]*mean[n*G + g]);           /* var = E[X^2] - (E[X])^2 */

        for(j = 0; j < group_size; j++){
          s[i + j] = 1.0f / ((float)sqrt(var[n*G + g] + eps));                                           /* 1/sqrt(var(X) + eps) */
          b[i + j] = -1 * mean[n*G + g] * s[i + j];                                                      /* -E[X]/sqrt(var(X) + eps) */
        }
      }

      arg_array[1].primary = s;                                                                           /* [bc] */
      arg_array[2].primary = b;                                                                           /* [bc] */
      arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);                             /* [bc] */
      arg_array[4].primary = (void*)&LIBXSMM_VLA_ACCESS(2, beta, cp, 0, bc);                              /* [bc] */

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

            if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU ||  cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
              arg_array[5].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp_add, n, cp, hi, wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);       /* [bw, bc] */
            }

            if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
              eqn_param.output.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
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
          eqn_param.output.primary    = &LIBXSMM_VLA_ACCESS(5, out, n, cp, ho, w, 0, CP, H, W, bc);          /* [HW,bc] */

          if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU ||  cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
            arg_array[5].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp_add, n, cp, hi, w, 0, CP, H, W, bc);    /* [HW, bc] */
          }

          if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
            eqn_param.output.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                            (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, w, 0, CP, H, W, (bc/BITS_PER_CHAR)) : NULL );
          }
          cfg.func10(&eqn_param);                                                   /* Normalization equation + relu + eltwise -> y = relu( ((s*x + b)*gamma + beta) + inp_add) */
        }
      } /* if-else for the presence of padding */
    } /* loop over cpxnt for computing out */
  } else {                                                         /* Case when group_size > bc */
    int n;
    for ( n = thr_begin_N; n < thr_end_N; ++n ) {
      LIBXSMM_ALIGNED(float tmp[2*bc], 64);
      LIBXSMM_ALIGNED(float sum_X[G], 64);
      LIBXSMM_ALIGNED(float sum_X2[G], 64);
      LIBXSMM_ALIGNED(float s[CP*bc], 64);
      LIBXSMM_ALIGNED(float b[CP*bc], 64);

      int hi = 0, ho = 0, w = 0, wb = 0, hwb = 0;
      int i, j, cp, g;
      float m, v;

      libxsmm_meltw_unary_param  m_reduce_rows_param;
      libxsmm_meltw_unary_param  v_reduce_rows_param;

      LIBXSMM_ALIGNED(float new_tmp[2*bc], 64);

      memset( &m_reduce_rows_param, 0, sizeof(m_reduce_rows_param));
      memset( &v_reduce_rows_param, 0, sizeof(v_reduce_rows_param));

      all_zero_param.out.primary = sum_X;
      cfg.all_zero_G_kernel(&all_zero_param);
      all_zero_param.out.primary = sum_X2;
      cfg.all_zero_G_kernel(&all_zero_param);

      for (cp = 0; cp < CP; cp++){                      /* [cp, HW, bc] */
        all_zero_param.out.primary = tmp;
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = &tmp[bc];
        cfg.all_zero_kernel(&all_zero_param);
        /* for (cb = 0; cb < 2*bc; cb++) { */
        /*   tmp[cb] = 0.0f; */
        /* } */

        reduce_param.out.primary   = new_tmp;                  /* [2*bc] */
        if (cfg.use_hw_blocking == 0) { /* w-blocking */
          for (hi = 0; hi < H; hi++) {
            for (wb = 0; wb < num_W_blocks; wb++) {
              reduce_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp, n, cp, hi, wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);      /* [HW_block, bc] -----> [2 * bc] */
              cfg.reduce_kernel(&reduce_param);

              add_param.in0.primary = tmp;
              add_param.in1.primary = new_tmp;
              add_param.out.primary = tmp;
              cfg.add_kernel(&add_param);

              add_param.in0.primary = &tmp[bc];
              add_param.in1.primary = &new_tmp[bc];
              add_param.out.primary = &tmp[bc];
              cfg.add_kernel(&add_param);

              /* for (cb = 0; cb < 2*bc; cb++) { */
              /*   tmp[cb] += new_tmp[cb]; */
              /* } */
            }
          }
        } else { /* hw-blocking (implies no padding) */
          for(hwb=0; hwb < num_HW_blocks; hwb++){
            hi = (hwb*(HW/num_HW_blocks))/W;
            w  = (hwb*(HW/num_HW_blocks))%W;
            reduce_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp, n, cp, hi, w, 0, CP, H, W, bc);      /* [HW_block, bc] -----> [2 * bc] */
            cfg.reduce_kernel(&reduce_param);

            add_param.in0.primary = tmp;
            add_param.in1.primary = new_tmp;
            add_param.out.primary = tmp;
            cfg.add_kernel(&add_param);

            add_param.in0.primary = &tmp[bc];
            add_param.in1.primary = &new_tmp[bc];
            add_param.out.primary = &tmp[bc];
            cfg.add_kernel(&add_param);

            /* for (cb = 0; cb < 2*bc; cb++) { */
            /*   tmp[cb] += new_tmp[cb]; */
            /* } */

          } /* loop over hw blocks */
        } /* if-else for the presence of input padding */

        if (group_size >= bc){                                 /* Group size >= block size  (Ex.- CP = 4, bc = 16, G = 2, group_size = 32) */
          g = (cp*bc)/group_size;                              /* determine current group */
          m_reduce_rows_param.in.primary    = tmp;
          m_reduce_rows_param.out.primary   = &m;
          v_reduce_rows_param.in.primary    = &tmp[bc];
          v_reduce_rows_param.out.primary   = &v;
          cfg.reduce_rows_kernel(&m_reduce_rows_param);
          cfg.reduce_rows_kernel(&v_reduce_rows_param);
          sum_X[g] += m;
          sum_X2[g] += v;
        }
        else{                                                 /* Group size < block size  (Ex.- CP = 4, bc = 16, G = 32, group_size = 2) */
          for(i=0; i < bc; i += group_size){
            m_reduce_groups_param.in.primary    = &tmp[i];
            m_reduce_groups_param.out.primary   = &sum_X[cp*(bc/group_size) + (i/group_size)];
            v_reduce_groups_param.in.primary    = &tmp[bc + i];
            v_reduce_groups_param.out.primary   = &sum_X2[cp*(bc/group_size) + (i/group_size)];
            cfg.reduce_groups_kernel(&m_reduce_groups_param);
            cfg.reduce_groups_kernel(&v_reduce_groups_param);
          }
        }
      }

      /* mean and variance calculation */
      for(g = 0; g < G; g++){
        mean[n*G + g] = sum_X[g] / ((float)group_size * HW);
        var[n*G + g] = (sum_X2[g] / ((float)group_size * HW)) - (mean[n*G + g]*mean[n*G + g]);           /* var = E[X^2] - (E[X])^2 */

        for(j = 0; j < group_size; j++){
          s[g*group_size + j] = 1.0f / ((float)sqrt(var[n*G + g] + eps));                                /* 1/sqrt(var(X) + eps) */
          b[g*group_size + j] = -1 * mean[n*G + g] * s[g*group_size + j];                                /* -E[X]/sqrt(var(X) + eps) */
        }
      }

      for (cp = 0; cp < CP; cp++){

        arg_array[1].primary = &s[cp*bc];                                                                /* [bc] */
        arg_array[2].primary = &b[cp*bc];                                                                /* [bc] */
        arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);                          /* [bc] */
        arg_array[4].primary = (void*)&LIBXSMM_VLA_ACCESS(2, beta, cp, 0, bc);                           /* [bc] */

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

              if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU ||  cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
                arg_array[5].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp_add, n, cp, hi, wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);       /* [bw, bc] */
              }

              if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
                eqn_param.output.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
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
            eqn_param.output.primary   =  &LIBXSMM_VLA_ACCESS(5, out, n, cp, ho, w, 0, CP, H, W, bc);          /* [HW,bc] */

            if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU ||  cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
              arg_array[5].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp_add, n, cp, hi, w, 0, CP, H, W, bc);    /* [HW, bc] */
            }

            if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
              eqn_param.output.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                              (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, w, 0, CP, H, W, (bc/BITS_PER_CHAR)) : NULL );
            }
            cfg.func10(&eqn_param);                                                   /* Normalization equation + relu + eltwise -> y = relu( ((s*x + b)*gamma + beta) + inp_add) */
          }
        } /* if-else for the presence of padding */
      } /* loop over cp for computing out */
    } /* loop over n */
  } /* if-else for the group_size */

  libxsmm_barrier_wait(cfg.barrier, ltid);

}


LIBXSMM_API void libxsmm_dnn_gn_bwd_exec_f32( libxsmm_dnn_gn_bwd_config cfg, float *pdout, const float *pinp, const float *mean, const float *var, const float *pgamma, const unsigned char *prelumask,
                         float *pdin, float *pdin_add, float *pdgamma, float *pdbeta, float eps,
                         int start_tid, int my_tid, void *scratch) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint G  = cfg.G;

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
  const libxsmm_blasint thr_begin_dN = (ltid * chunksize_dN < work_dN) ? (ltid * chunksize_dN) : work_dN;
  const libxsmm_blasint thr_end_dN = ((ltid + 1) * chunksize_dN < work_dN) ? ((ltid + 1) * chunksize_dN) : work_dN;

  /* number of tasks that could be run in parallel for 1d blocking over CP */
  const libxsmm_blasint work_C = CP;
  /* compute chunk size */
  const libxsmm_blasint chunksize_C = (work_C % cfg.threads == 0) ?
    (work_C / cfg.threads) : ((work_C / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_C = (ltid * chunksize_C < work_C) ? (ltid * chunksize_C) : work_C;
  const libxsmm_blasint thr_end_C = ((ltid + 1) * chunksize_C < work_C) ? ((ltid + 1) * chunksize_C) : work_C;

  /* number of tasks that could be run in parallel for 1d blocking over N */
  const libxsmm_blasint work_N = N;
  /* compute chunk size */
  const libxsmm_blasint chunksize_N = (work_N % cfg.threads == 0) ?
    (work_N / cfg.threads) : ((work_N / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_N = (ltid * chunksize_N < work_N) ? (ltid * chunksize_N) : work_N;
  const libxsmm_blasint thr_end_N = ((ltid + 1) * chunksize_N < work_N) ? ((ltid + 1) * chunksize_N) : work_N;

  libxsmm_meltw_unary_param  all_zero_param;
  libxsmm_meltw_unary_param  all_relu_param;
  libxsmm_meltw_unary_param  ewise_copy_param;

  libxsmm_matrix_arg arg_array[10];
  libxsmm_matrix_eqn_param eqn_param;

  const int group_size = (CP*bc)/G;

  const float scale = 1.0f / ((float)group_size * HW);

  LIBXSMM_VLA_DECL(5,       float,          din,      pdin,     CP, ifhp, ifwp, bc);    /* [N, CP, ifhp, ifwp, bc] */
  LIBXSMM_VLA_DECL(5,       float,          din_add,  pdin_add, CP, ifhp, ifwp, bc);    /* [N, CP, ifhp, ifwp, bc] */
  LIBXSMM_VLA_DECL(5, const float,          inp,      pinp,     CP, ifhp, ifwp, bc);    /* [N, CP, ifhp, ifwp, bc] */
  LIBXSMM_VLA_DECL(5,       float,          dout,     pdout     + (ho_start * ofwp + wo_start) * bc, CP, ofhp, ofwp, bc);                              /* [N, CP, ofhp, ofwp, bc] */
  LIBXSMM_VLA_DECL(5, const unsigned char, relumask , prelumask + (ho_start * ofwp + wo_start) * bc/BITS_PER_CHAR, CP, ofhp, ofwp, bc/BITS_PER_CHAR);  /* [N, CP, ofhp, ofwp, bc/BITS_PER_CHAR] */

  LIBXSMM_VLA_DECL(2, const float, gamma,   pgamma,  bc);                /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float, dgamma,  pdgamma, bc);                /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float, dbeta,   pdbeta,  bc);                /* [CP, bc] */

  float alpha = 0.0f;

  const libxsmm_blasint dbeta_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + N * CP * bc), 64) - ((uintptr_t)(scratch))) / sizeof(float);
  LIBXSMM_VLA_DECL(3, float, dgamma_N, ((float*)scratch),                  CP, bc);  /* [N, CP, bc] */
  LIBXSMM_VLA_DECL(3, float, dbeta_N,  ((float*)scratch) + dbeta_N_offset, CP, bc);  /* [N, CP, bc] */
  LIBXSMM_ASSUME_ALIGNED(dgamma_N_, 64);
  LIBXSMM_ASSUME_ALIGNED(dbeta_N_, 64);

  memset( &all_zero_param,        0, sizeof(all_zero_param));
  memset( &all_relu_param,        0, sizeof(all_relu_param));
  memset( &ewise_copy_param,      0, sizeof(ewise_copy_param));
  memset( &eqn_param,             0, sizeof(eqn_param));

  eqn_param.inputs = arg_array;

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  if (group_size <= bc){
    LIBXSMM_ALIGNED(float a[bc], 64);
    LIBXSMM_ALIGNED(float b[bc], 64);
    LIBXSMM_ALIGNED(float c[bc], 64);
    LIBXSMM_ALIGNED(float ds[bc], 64);
    LIBXSMM_ALIGNED(float db[bc], 64);

    int n, cp;
    int cpxnt;
    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      int hi = 0, ho = 0, w = 0, wb = 0, hwb = 0;
      int j, g, lg;

      n  = cpxnt/CP;
      cp = cpxnt%CP;

      /* for(j = 0; j < bc; j++){
          dgamma_N[n*CP*bc + cp*bc + j] = 0.0f;
          dbeta_N[n*CP*bc + cp*bc + j] = 0.0f;
       } */

      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, n, cp, 0, CP, bc);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, n, cp, 0, CP, bc);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = ds;
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = db;
      cfg.all_zero_kernel(&all_zero_param);

      /* compute a and b for each channel from group means and variance */
      for(g = (cp*bc)/group_size; g < ((cp+1)*bc)/group_size; g++){
        lg = g - (cp*bc)/group_size;
        for(j = 0; j < group_size; j++){
          a[lg*group_size + j] = 1.0f / ((float)sqrt(var[n*G + g] + eps));
          b[lg*group_size + j] = -a[lg*group_size + j]*mean[n*G + g];
        }
      }

      arg_array[1].primary = a;
      arg_array[2].primary = b;
      arg_array[4].primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, n, cp, 0, CP, bc);
      arg_array[5].primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, n, cp, 0, CP, bc);
      arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);
      arg_array[8].primary = ds;
      arg_array[9].primary = db;

      if (cfg.use_hw_blocking == 0) { /* w-blocking */
        /* Reminder: dout and relumask are already shifted by the offset (= point to the non-padded part already),
           while the other arrays are non-shifted (and hence accesses require offsets */
        /* Notice: Zeroing out the rim for din_add is not strictly necessary but for safety is done here */
        /* zeroing out strip [0, hi_start) */
        if (cfg.pad_h_in != 0 &&
              (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK)
            ) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, 0, 0, 0, CP, ifhp, ifwp, bc);
          cfg.all_zero_hp_kernel(&all_zero_param);
        }
        for (ho = 0, hi = hi_start; ho < H; ho++, hi++) {
          /* zeroing out starting [0, wi_start) x bc block for fixed hi */
          if (cfg.pad_w_in != 0 &&
              (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK)
              ) {
            all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, 0, 0, CP, ifhp, ifwp, bc);
            cfg.all_zero_wp_kernel(&all_zero_param);
          }
          for (wb = 0; wb < num_W_blocks; wb++) {
            if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE ||
              cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
              if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
                all_relu_param.op.primary   = (void*)(&alpha);
                all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);      /* [HW,bc] */
                all_relu_param.in.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                                 (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc/8)
                                                 : NULL );
                all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);      /* [HW,bc] */
                cfg.inv_relu_kernel(&all_relu_param);
              } /* ReLU/mask */
              if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
                ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(5, dout,    n, cp, ho,            wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);
                ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
                cfg.ewise_copy_kernel(&ewise_copy_param);
              } /* Eltwise */
            }

            arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
            arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho,            wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);

            eqn_param.output.primary = ds;
            cfg.ds_func(&eqn_param);

            eqn_param.output.primary = db;
            cfg.db_func(&eqn_param);

            eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, n, cp, 0, CP, bc);
            cfg.dgamma_func(&eqn_param);

            eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3,  dbeta_N, n, cp, 0, CP, bc);
            cfg.dbeta_func(&eqn_param);

          }

          /* zeroing out ending [wi_end, ifwp] x bc block for fixed hi */
          if (cfg.pad_w_in != 0 &&
              (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK)
              ) {
            all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, wi_end, 0, CP, ifhp, ifwp, bc);
            cfg.all_zero_wp_kernel(&all_zero_param);
          }

        }
        /* zeroing out strip [hi_end, ifhp) x ifwp x bc */
        if (cfg.pad_h_in != 0 &&
              (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK)
            ) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi_end, 0, 0, CP, ifhp, ifwp, bc);
          cfg.all_zero_hp_kernel(&all_zero_param);
        }

      } else { /* hw-blocking (implies no padding) */
        for(hwb=0; hwb < num_HW_blocks; hwb++){
          ho = (hwb*(HW/num_HW_blocks))/W;
          hi = ho;
          w  = (hwb*(HW/num_HW_blocks))%W;
          if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE ||
            cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
            if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
              all_relu_param.op.primary   = (void*)(&alpha);
              all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);      /* [HW,bc] */
              all_relu_param.in.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                               (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, w, 0, CP, H, W, bc/8)
                                               : NULL /*&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc) */ ); /* dout_fwd ? nonsense? */
              all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);      /* [HW,bc] */
              cfg.inv_relu_kernel(&all_relu_param);
            } /* ReLU/mask */
            if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
              ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(5, dout,    n, cp, ho, w, 0, CP, H, W, bc);
              ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, w, 0, CP, H, W, bc);
              cfg.ewise_copy_kernel(&ewise_copy_param);
            } /* Eltwise */
          }

          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, w, 0, CP, H, W, bc);
          arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);

          eqn_param.output.primary = ds;
          cfg.ds_func(&eqn_param);

          eqn_param.output.primary = db;
          cfg.db_func(&eqn_param);

          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, n, cp, 0, CP, bc);
          cfg.dgamma_func(&eqn_param);

          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3,  dbeta_N, n, cp, 0, CP, bc);
          cfg.dbeta_func(&eqn_param);

        } /* loop over hw blocks */
      } /* if-else for the presence of input padding */

      /* b = (db * mean[nb] - ds) * a * a * a * scale; */
      /* c = -b * mean[nb] - db * a * scale; */

      for(g = (cp*bc)/group_size; g < ((cp+1)*bc)/group_size; g++){            /* compute b and c for each channel from group means and variance */
        lg = g - (cp*bc)/group_size;
        float gds = 0.0f;
        float gdb = 0.0f;
        for(j = 0; j < group_size; j++){
          gds += ds[lg*group_size + j];                                        /* Group ds and db calculation */
          gdb += db[lg*group_size + j];
        }
        for(j = 0; j < group_size; j++){
          b[lg*group_size + j] = (gdb * mean[n*G + g] - gds) * a[lg*group_size + j] * a[lg*group_size + j] * a[lg*group_size + j] * scale;
          c[lg*group_size + j] = -b[lg*group_size + j] * mean[n*G + g] - gdb * a[lg*group_size + j] * scale;
        }
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

            arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
            arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho,            wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);
            eqn_param.output.primary =    &LIBXSMM_VLA_ACCESS(5, din,  n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
            cfg.din_func(&eqn_param);

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

          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, w, 0, CP, H, W, bc);
          arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);
          eqn_param.output.primary =    &LIBXSMM_VLA_ACCESS(5, din,  n, cp, hi, w, 0, CP, H, W, bc);
          cfg.din_func(&eqn_param);

        } /* loop over hw blocks */
      } /* if-else for the presence of input padding */
    } /* loop over cpxnt for computing din */

    libxsmm_barrier_wait(cfg.barrier, ltid); /* not needed? */

    for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, bc);
      cfg.all_zero_kernel(&all_zero_param);

      for (n=0; n < N; n++ ) {
        int cb;
        for(cb = 0; cb < bc; cb++){
          LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, bc) += LIBXSMM_VLA_ACCESS(3, dgamma_N, n, cp, cb, CP, bc);
          LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, bc)  += LIBXSMM_VLA_ACCESS(3, dbeta_N,  n, cp, cb, CP, bc);
        }
      }
    } /* loop over cp for finalizing dgamma and dbeta */

  } else { /* if-else for group_size */

    LIBXSMM_ALIGNED(float a[CP*bc], 64);
    LIBXSMM_ALIGNED(float b[CP*bc], 64);
    LIBXSMM_ALIGNED(float c[CP*bc], 64);
    LIBXSMM_ALIGNED(float ds[CP*bc], 64);
    LIBXSMM_ALIGNED(float db[CP*bc], 64);
    int n;

    for ( n = thr_begin_N; n < thr_end_N; ++n ) {
      int hi = 0, ho = 0, w = 0, wb = 0, hwb = 0, cp = 0;
      int j, g;

      /* for(j = 0; j < CP*bc; j++){ */
      /*   dgamma_N[n*CP*bc + j] = 0.0f; */
      /*   dbeta_N[n*CP*bc + j] = 0.0f; */
      /* } */

      for (cp = 0; cp < CP; cp++) {
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, n, cp, 0, CP, bc);
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, n, cp, 0, CP, bc);
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = &ds[cp*bc];
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = &db[cp*bc];
        cfg.all_zero_kernel(&all_zero_param);
      }

      for(g = 0; g < G; g++){                                                  /* compute a and b for each channel from group means and variance */
        for(j = 0; j < group_size; j++){
          a[g*group_size + j] = 1.0f / ((float)sqrt(var[n*G + g] + eps));
          b[g*group_size + j] = -a[g*group_size + j]*mean[n*G + g];
        }
      }

      for (cp = 0; cp < CP; cp++) {
        arg_array[1].primary = &a[cp*bc];
        arg_array[2].primary = &b[cp*bc];
        arg_array[4].primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, n, cp, 0, CP, bc);
        arg_array[5].primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N,  n, cp, 0, CP, bc);
        arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);
        arg_array[8].primary = &ds[cp*bc];
        arg_array[9].primary = &db[cp*bc];

        if (cfg.use_hw_blocking == 0) { /* w-blocking */
          /* Reminder: dout and relumask are already shifted by the offset (= point to the non-padded part already),
             while the other arrays are non-shifted (and hence accesses require offsets */
          /* Notice: Zeroing out the rim for din_add is not strictly necessary but for safety is done here */
          /* zeroing out strip [0, hi_start) */
          if (cfg.pad_h_in != 0 &&
                (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK)
              ) {
            all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, 0, 0, 0, CP, ifhp, ifwp, bc);
            cfg.all_zero_hp_kernel(&all_zero_param);
          }
          for (ho = 0, hi = hi_start; ho < H; ho++, hi++) {
            /* zeroing out starting [0, wi_start) x bc block for fixed hi */
            if (cfg.pad_w_in != 0 &&
                (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK)
                ) {
              all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, 0, 0, CP, ifhp, ifwp, bc);
              cfg.all_zero_wp_kernel(&all_zero_param);
            }
            for (wb = 0; wb < num_W_blocks; wb++) {
              if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE ||
                cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
                if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
                  all_relu_param.op.primary   = (void*)(&alpha);
                  all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);      /* [HW,bc] */
                  all_relu_param.in.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                                   (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc/8)
                                                   : NULL );
                  all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);      /* [HW,bc] */
                  cfg.inv_relu_kernel(&all_relu_param);
                } /* ReLU/mask */
                if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
                  ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(5, dout,    n, cp, ho,            wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);
                  ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
                  cfg.ewise_copy_kernel(&ewise_copy_param);
                } /* Eltwise */
              }

              arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
              arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho,            wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);

              eqn_param.output.primary = ds;
              cfg.ds_func(&eqn_param);

              eqn_param.output.primary = db;
              cfg.db_func(&eqn_param);

              eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, n, cp, 0, CP, bc);
              cfg.dgamma_func(&eqn_param);

              eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3,  dbeta_N, n, cp, 0, CP, bc);
              cfg.dbeta_func(&eqn_param);

            }

            /* zeroing out ending [wi_end, ifwp] x bc block for fixed hi */
            if (cfg.pad_w_in != 0 &&
                (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK)
                ) {
              all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, wi_end, 0, CP, ifhp, ifwp, bc);
              cfg.all_zero_wp_kernel(&all_zero_param);
            }

          }
          /* zeroing out strip [hi_end, ifhp) x ifwp x bc */
          if (cfg.pad_h_in != 0 &&
                (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK)
              ) {
            all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi_end, 0, 0, CP, ifhp, ifwp, bc);
            cfg.all_zero_hp_kernel(&all_zero_param);
          }

        } else { /* hw-blocking (implies no padding) */
          for(hwb=0; hwb < num_HW_blocks; hwb++){
            ho = (hwb*(HW/num_HW_blocks))/W;
            hi = ho;
            w  = (hwb*(HW/num_HW_blocks))%W;
            if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE ||
              cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
              if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
                all_relu_param.op.primary   = (void*)(&alpha);
                all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);      /* [HW,bc] */
                all_relu_param.in.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                                 (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, w, 0, CP, H, W, bc/8)
                                                 : NULL /*&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc) */ ); /* dout_fwd ? nonsense? */
                all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);      /* [HW,bc] */
                cfg.inv_relu_kernel(&all_relu_param);
              } /* ReLU/mask */
              if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
                ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(5, dout,    n, cp, ho, w, 0, CP, H, W, bc);
                ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, w, 0, CP, H, W, bc);
                cfg.ewise_copy_kernel(&ewise_copy_param);
              } /* Eltwise */
            }

            arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, w, 0, CP, H, W, bc);
            arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);

            eqn_param.output.primary = ds;
            cfg.ds_func(&eqn_param);

            eqn_param.output.primary = db;
            cfg.db_func(&eqn_param);

            eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, n, cp, 0, CP, bc);
            cfg.dgamma_func(&eqn_param);

            eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3,  dbeta_N, n, cp, 0, CP, bc);
            cfg.dbeta_func(&eqn_param);

          } /* loop over hw blocks */
        } /* if-else for the presence of input padding */
      } /* loop over cp for computing din */

      /* b = (db * mean[nb] - ds) * a * a * a * scale; */
      /* c = -b * mean[nb] - db * a * scale; */

      for(g = 0; g < G; g++){                                                 /* compute b and c for each channel from group means and variance */
        float gds = 0.0f;
        float gdb = 0.0f;
        for(j = 0; j < group_size; j++){
          gds += ds[g*group_size + j];                                        /* Group ds and db calculation */
          gdb += db[g*group_size + j];
        }
        for(j = 0; j < group_size; j++){
          b[g*group_size + j] = (gdb * mean[n*G + g] - gds) * a[g*group_size + j] * a[g*group_size + j] * a[g*group_size + j] * scale;
          c[g*group_size + j] = -b[g*group_size + j] * mean[n*G + g] - gdb * a[g*group_size + j] * scale;
        }
      }

      for (cp = 0; cp < CP; cp++) {

        arg_array[1].primary = &a[cp*bc];
        arg_array[2].primary = &b[cp*bc];
        arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);
        arg_array[7].primary = &c[cp*bc];

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

              arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
              arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho,            wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);
              eqn_param.output.primary =    &LIBXSMM_VLA_ACCESS(5, din,  n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
              cfg.din_func(&eqn_param);

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

            arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, w, 0, CP, H, W, bc);
            arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);
            eqn_param.output.primary =    &LIBXSMM_VLA_ACCESS(5, din,  n, cp, hi, w, 0, CP, H, W, bc);
            cfg.din_func(&eqn_param);

          } /* loop over hw blocks */
        } /* if-else for the presence of input padding */
      } /* loop over cp for computing din */
    } /* loop over n for computing din */

    libxsmm_barrier_wait(cfg.barrier, ltid);

    int cp;
    for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2,  dbeta, cp, 0, bc);
      cfg.all_zero_kernel(&all_zero_param);

      for (n=0; n < N; n++ ) {
        int cb;
        for(cb = 0; cb < bc; cb++){
          LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, bc) += LIBXSMM_VLA_ACCESS(3, dgamma_N, n, cp, cb, CP, bc);
          LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, bc)  += LIBXSMM_VLA_ACCESS(3, dbeta_N,  n, cp, cb, CP, bc);
        }
      }
    } /* loop over cp for finalizing dgamma and dbeta */
  } /* if-else for the group_size */

  libxsmm_barrier_wait(cfg.barrier, ltid);
}

LIBXSMM_API void libxsmm_dnn_gn_bwd_exec_bf16( libxsmm_dnn_gn_bwd_config cfg, libxsmm_bfloat16 *pdout, const libxsmm_bfloat16 *pinp, const float *mean, const float *var, const float *pgamma, const unsigned char *prelumask,
                          libxsmm_bfloat16 *pdin, libxsmm_bfloat16 *pdin_add, float *pdgamma, float *pdbeta, float eps,
                          int start_tid, int my_tid, void *scratch) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint G  = cfg.G;

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
  const libxsmm_blasint thr_begin_dN = (ltid * chunksize_dN < work_dN) ? (ltid * chunksize_dN) : work_dN;
  const libxsmm_blasint thr_end_dN = ((ltid + 1) * chunksize_dN < work_dN) ? ((ltid + 1) * chunksize_dN) : work_dN;

  /* number of tasks that could be run in parallel for 1d blocking over CP */
  const libxsmm_blasint work_C = CP;
  /* compute chunk size */
  const libxsmm_blasint chunksize_C = (work_C % cfg.threads == 0) ?
    (work_C / cfg.threads) : ((work_C / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_C = (ltid * chunksize_C < work_C) ? (ltid * chunksize_C) : work_C;
  const libxsmm_blasint thr_end_C = ((ltid + 1) * chunksize_C < work_C) ? ((ltid + 1) * chunksize_C) : work_C;

  /* number of tasks that could be run in parallel for 1d blocking over N */
  const libxsmm_blasint work_N = N;
  /* compute chunk size */
  const libxsmm_blasint chunksize_N = (work_N % cfg.threads == 0) ?
    (work_N / cfg.threads) : ((work_N / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_N = (ltid * chunksize_N < work_N) ? (ltid * chunksize_N) : work_N;
  const libxsmm_blasint thr_end_N = ((ltid + 1) * chunksize_N < work_N) ? ((ltid + 1) * chunksize_N) : work_N;

  libxsmm_meltw_unary_param  all_zero_param;
  libxsmm_meltw_unary_param  all_relu_param;
  libxsmm_meltw_unary_param  ewise_copy_param;

  libxsmm_matrix_arg arg_array[10];
  libxsmm_matrix_eqn_param eqn_param;

  memset( &all_zero_param,        0, sizeof(all_zero_param));
  memset( &all_relu_param,        0, sizeof(all_relu_param));
  memset( &ewise_copy_param,      0, sizeof(ewise_copy_param));
  memset( &eqn_param,             0, sizeof(eqn_param));

  eqn_param.inputs = arg_array;

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  const int group_size = (CP*bc)/G;

  const float scale = 1.0f / ((float)group_size * HW);

  LIBXSMM_VLA_DECL(5,       libxsmm_bfloat16, din,      pdin,     CP, ifhp, ifwp, bc);    /* [N, CP, ifhp, ifwp, bc] */
  LIBXSMM_VLA_DECL(5,       libxsmm_bfloat16, din_add,  pdin_add, CP, ifhp, ifwp, bc);    /* [N, CP, ifhp, ifwp, bc] */
  LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16, inp,      pinp,     CP, ifhp, ifwp, bc);    /* [N, CP, ifhp, ifwp, bc] */
  LIBXSMM_VLA_DECL(5,       libxsmm_bfloat16, dout,   pdout     + (ho_start * ofwp + wo_start) * bc, CP, ofhp, ofwp, bc);                              /* [N, CP, ofhp, ofwp, bc] */
  LIBXSMM_VLA_DECL(5, const unsigned char, relumask , prelumask + (ho_start * ofwp + wo_start) * bc/BITS_PER_CHAR, CP, ofhp, ofwp, bc/BITS_PER_CHAR);  /* [N, CP, ofhp, ofwp, bc/BITS_PER_CHAR] */

  LIBXSMM_VLA_DECL(2, const float, gamma,   pgamma,  bc);                /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float, dgamma,  pdgamma, bc);                /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float, dbeta,   pdbeta,  bc);                /* [CP, bc] */

  float alpha = 0.0f;

  const libxsmm_blasint dbeta_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + N * CP * bc), 64) - ((uintptr_t)(scratch))) / sizeof(float);
  LIBXSMM_VLA_DECL(3, float, dgamma_N, ((float*)scratch),                  CP, bc);  /* [N, CP, bc] */
  LIBXSMM_ASSUME_ALIGNED(dgamma_N_, 64);
  LIBXSMM_VLA_DECL(3, float, dbeta_N,  ((float*)scratch) + dbeta_N_offset, CP, bc);  /* [N, CP, bc] */
  LIBXSMM_ASSUME_ALIGNED(dbeta_N_, 64);

  if (group_size <= bc){
    LIBXSMM_ALIGNED(float a[bc], 64);
    LIBXSMM_ALIGNED(float b[bc], 64);
    LIBXSMM_ALIGNED(float c[bc], 64);
    LIBXSMM_ALIGNED(float ds[bc], 64);
    LIBXSMM_ALIGNED(float db[bc], 64);

    int n, cp;
    int cpxnt;
    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      n  = cpxnt/CP;
      cp = cpxnt%CP;

      int hi = 0, ho = 0, w = 0, wb = 0, hwb = 0;
      int j, g, lg;

      /* for(j = 0; j < bc; j++){
          dgamma_N[n*CP*bc + cp*bc + j] = 0.0f;
          dbeta_N[n*CP*bc + cp*bc + j] = 0.0f;
       } */

      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, n, cp, 0, CP, bc);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N,  n, cp, 0, CP, bc);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = ds;
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = db;
      cfg.all_zero_kernel(&all_zero_param);

      /* compute a and b for each channel from group means and variance */
      for(g = (cp*bc)/group_size; g < ((cp+1)*bc)/group_size; g++){
        lg = g - (cp*bc)/group_size;
        for(j = 0; j < group_size; j++){
          a[lg*group_size + j] = 1.0f / ((float)sqrt(var[n*G + g] + eps));
          b[lg*group_size + j] = -a[lg*group_size + j]*mean[n*G + g];
        }
      }

      arg_array[1].primary = a;
      arg_array[2].primary = b;
      arg_array[4].primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, n, cp, 0, CP, bc);
      arg_array[5].primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, n, cp, 0, CP, bc);
      arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);
      arg_array[8].primary = ds;
      arg_array[9].primary = db;

      if (cfg.use_hw_blocking == 0) { /* w-blocking */
        /* Reminder: dout and relumask are already shifted by the offset (= point to the non-padded part already),
           while the other arrays are non-shifted (and hence accesses require offsets */
        /* Notice: Zeroing out the rim for din_add is not strictly necessary but for safety is done here */
        /* zeroing out strip [0, hi_start) */
        if (cfg.pad_h_in != 0 &&
              (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK)
            ) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, 0, 0, 0, CP, ifhp, ifwp, bc);
          cfg.all_zero_hp_kernel(&all_zero_param);
        }
        for (ho = 0, hi = hi_start; ho < H; ho++, hi++) {
          /* zeroing out starting [0, wi_start) x bc block for fixed hi */
          if (cfg.pad_w_in != 0 &&
              (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK)
              ) {
            all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, 0, 0, CP, ifhp, ifwp, bc);
            cfg.all_zero_wp_kernel(&all_zero_param);
          }
          for (wb = 0; wb < num_W_blocks; wb++) {
            if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE ||
              cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
              if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
                all_relu_param.op.primary   = (void*)(&alpha);
                all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);      /* [HW,bc] */
                all_relu_param.in.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                                 (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc/8)
                                                 : NULL );
                all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);      /* [HW,bc] */
                cfg.inv_relu_kernel(&all_relu_param);
              } /* ReLU/mask */
              if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
                ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(5, dout,    n, cp, ho,            wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);
                ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
                cfg.ewise_copy_kernel(&ewise_copy_param);
              } /* Eltwise */
            }

            arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
            arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho,            wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);

            eqn_param.output.primary = ds;
            cfg.ds_func(&eqn_param);

            eqn_param.output.primary = db;
            cfg.db_func(&eqn_param);

            eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, n, cp, 0, CP, bc);
            cfg.dgamma_func(&eqn_param);

            eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3,  dbeta_N, n, cp, 0, CP, bc);
            cfg.dbeta_func(&eqn_param);

          }

          /* zeroing out ending [wi_end, ifwp] x bc block for fixed hi */
          if (cfg.pad_w_in != 0 &&
              (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK)
              ) {
            all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, wi_end, 0, CP, ifhp, ifwp, bc);
            cfg.all_zero_wp_kernel(&all_zero_param);
          }

        }
        /* zeroing out strip [hi_end, ifhp) x ifwp x bc */
        if (cfg.pad_h_in != 0 &&
              (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK)
            ) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi_end, 0, 0, CP, ifhp, ifwp, bc);
          cfg.all_zero_hp_kernel(&all_zero_param);
        }

      } else { /* hw-blocking (implies no padding) */
        for(hwb=0; hwb < num_HW_blocks; hwb++){
          ho = (hwb*(HW/num_HW_blocks))/W;
          hi = ho;
          w  = (hwb*(HW/num_HW_blocks))%W;
          if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE ||
            cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
            if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
              all_relu_param.op.primary   = (void*)(&alpha);
              all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);      /* [HW,bc] */
              all_relu_param.in.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                               (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, w, 0, CP, H, W, bc/8)
                                               : NULL /*&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc) */ ); /* dout_fwd ? nonsense? */
              all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);      /* [HW,bc] */
              cfg.inv_relu_kernel(&all_relu_param);
            } /* ReLU/mask */
            if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
              ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(5, dout,    n, cp, ho, w, 0, CP, H, W, bc);
              ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, w, 0, CP, H, W, bc);
              cfg.ewise_copy_kernel(&ewise_copy_param);
            } /* Eltwise */
          }

          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, w, 0, CP, H, W, bc);
          arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);

          eqn_param.output.primary = ds;
          cfg.ds_func(&eqn_param);

          eqn_param.output.primary = db;
          cfg.db_func(&eqn_param);

          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, n, cp, 0, CP, bc);
          cfg.dgamma_func(&eqn_param);

          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3,  dbeta_N, n, cp, 0, CP, bc);
          cfg.dbeta_func(&eqn_param);

        } /* loop over hw blocks */
      } /* if-else for the presence of input padding */

      /* b = (db * mean[nb] - ds) * a * a * a * scale; */
      /* c = -b * mean[nb] - db * a * scale; */

      for(g = (cp*bc)/group_size; g < ((cp+1)*bc)/group_size; g++){            /* compute b and c for each channel from group means and variance */
        lg = g - (cp*bc)/group_size;
        float gds = 0.0f;
        float gdb = 0.0f;
        for(j = 0; j < group_size; j++){
          gds += ds[lg*group_size + j];                                        /* Group ds and db calculation */
          gdb += db[lg*group_size + j];
        }
        for(j = 0; j < group_size; j++){
          b[lg*group_size + j] = (gdb * mean[n*G + g] - gds) * a[lg*group_size + j] * a[lg*group_size + j] * a[lg*group_size + j] * scale;
          c[lg*group_size + j] = -b[lg*group_size + j] * mean[n*G + g] - gdb * a[lg*group_size + j] * scale;
        }
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

            arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
            arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho,            wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);
            eqn_param.output.primary =    &LIBXSMM_VLA_ACCESS(5, din,  n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
            cfg.din_func(&eqn_param);

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

          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, w, 0, CP, H, W, bc);
          arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);
          eqn_param.output.primary =    &LIBXSMM_VLA_ACCESS(5, din,  n, cp, hi, w, 0, CP, H, W, bc);
          cfg.din_func(&eqn_param);

        } /* loop over hw blocks */
      } /* if-else for the presence of input padding */
    } /* loop over cpxnt for computing din */

    libxsmm_barrier_wait(cfg.barrier, ltid); /* not needed? */

    for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, bc);
      cfg.all_zero_kernel(&all_zero_param);

      for (n=0; n < N; n++ ) {
        int cb;
        for(cb = 0; cb < bc; cb++){
          LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, bc) += LIBXSMM_VLA_ACCESS(3, dgamma_N, n, cp, cb, CP, bc);
          LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, bc)  += LIBXSMM_VLA_ACCESS(3, dbeta_N,  n, cp, cb, CP, bc);
        }
      }
    } /* loop over cp for finalizing dgamma and dbeta */

  } else { /* if-else for group_size */

    LIBXSMM_ALIGNED(float a[CP*bc], 64);
    LIBXSMM_ALIGNED(float b[CP*bc], 64);
    LIBXSMM_ALIGNED(float c[CP*bc], 64);
    LIBXSMM_ALIGNED(float ds[CP*bc], 64);
    LIBXSMM_ALIGNED(float db[CP*bc], 64);
    int n;

    for ( n = thr_begin_N; n < thr_end_N; ++n ) {
      int hi = 0, ho = 0, w = 0, wb = 0, hwb = 0, cp = 0;
      int j, g;

      /* for(j = 0; j < CP*bc; j++){ */
      /*   dgamma_N[n*CP*bc + j] = 0.0f; */
      /*   dbeta_N[n*CP*bc + j] = 0.0f; */
      /* } */

      for (cp = 0; cp < CP; cp++) {
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, n, cp, 0, CP, bc);
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, n, cp, 0, CP, bc);
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = &ds[cp*bc];
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = &db[cp*bc];
        cfg.all_zero_kernel(&all_zero_param);
      }

      for(g = 0; g < G; g++){                                                  /* compute a and b for each channel from group means and variance */
        for(j = 0; j < group_size; j++){
          a[g*group_size + j] = 1.0f / ((float)sqrt(var[n*G + g] + eps));
          b[g*group_size + j] = -a[g*group_size + j]*mean[n*G + g];
        }
      }

      for (cp = 0; cp < CP; cp++) {
        arg_array[1].primary = &a[cp*bc];
        arg_array[2].primary = &b[cp*bc];
        arg_array[4].primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, n, cp, 0, CP, bc);
        arg_array[5].primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N,  n, cp, 0, CP, bc);
        arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);
        arg_array[8].primary = &ds[cp*bc];
        arg_array[9].primary = &db[cp*bc];

        if (cfg.use_hw_blocking == 0) { /* w-blocking */
          /* Reminder: dout and relumask are already shifted by the offset (= point to the non-padded part already),
             while the other arrays are non-shifted (and hence accesses require offsets */
          /* Notice: Zeroing out the rim for din_add is not strictly necessary but for safety is done here */
          /* zeroing out strip [0, hi_start) */
          if (cfg.pad_h_in != 0 &&
                (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK)
              ) {
            all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, 0, 0, 0, CP, ifhp, ifwp, bc);
            cfg.all_zero_hp_kernel(&all_zero_param);
          }
          for (ho = 0, hi = hi_start; ho < H; ho++, hi++) {
            /* zeroing out starting [0, wi_start) x bc block for fixed hi */
            if (cfg.pad_w_in != 0 &&
                (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK)
                ) {
              all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, 0, 0, CP, ifhp, ifwp, bc);
              cfg.all_zero_wp_kernel(&all_zero_param);
            }
            for (wb = 0; wb < num_W_blocks; wb++) {
              if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE ||
                cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
                if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
                  all_relu_param.op.primary   = (void*)(&alpha);
                  all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);      /* [HW,bc] */
                  all_relu_param.in.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                                   (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc/8)
                                                   : NULL );
                  all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);      /* [HW,bc] */
                  cfg.inv_relu_kernel(&all_relu_param);
                } /* ReLU/mask */
                if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
                  ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(5, dout,    n, cp, ho,            wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);
                  ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
                  cfg.ewise_copy_kernel(&ewise_copy_param);
                } /* Eltwise */
              }

              arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
              arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho,            wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);

              eqn_param.output.primary = ds;
              cfg.ds_func(&eqn_param);

              eqn_param.output.primary = db;
              cfg.db_func(&eqn_param);

              eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, n, cp, 0, CP, bc);
              cfg.dgamma_func(&eqn_param);

              eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3,  dbeta_N, n, cp, 0, CP, bc);
              cfg.dbeta_func(&eqn_param);

            }

            /* zeroing out ending [wi_end, ifwp] x bc block for fixed hi */
            if (cfg.pad_w_in != 0 &&
                (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK)
                ) {
              all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, wi_end, 0, CP, ifhp, ifwp, bc);
              cfg.all_zero_wp_kernel(&all_zero_param);
            }

          }
          /* zeroing out strip [hi_end, ifhp) x ifwp x bc */
          if (cfg.pad_h_in != 0 &&
                (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK)
              ) {
            all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi_end, 0, 0, CP, ifhp, ifwp, bc);
            cfg.all_zero_hp_kernel(&all_zero_param);
          }

        } else { /* hw-blocking (implies no padding) */
          for(hwb=0; hwb < num_HW_blocks; hwb++){
            ho = (hwb*(HW/num_HW_blocks))/W;
            hi = ho;
            w  = (hwb*(HW/num_HW_blocks))%W;
            if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE ||
              cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
              if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
                all_relu_param.op.primary   = (void*)(&alpha);
                all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);      /* [HW,bc] */
                all_relu_param.in.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                                 (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, w, 0, CP, H, W, bc/8)
                                                 : NULL /*&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc) */ ); /* dout_fwd ? nonsense? */
                all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);      /* [HW,bc] */
                cfg.inv_relu_kernel(&all_relu_param);
              } /* ReLU/mask */
              if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
                ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(5, dout,    n, cp, ho, w, 0, CP, H, W, bc);
                ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, w, 0, CP, H, W, bc);
                cfg.ewise_copy_kernel(&ewise_copy_param);
              } /* Eltwise */
            }

            arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, w, 0, CP, H, W, bc);
            arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);

            eqn_param.output.primary = ds;
            cfg.ds_func(&eqn_param);

            eqn_param.output.primary = db;
            cfg.db_func(&eqn_param);

            eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, n, cp, 0, CP, bc);
            cfg.dgamma_func(&eqn_param);

            eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3,  dbeta_N, n, cp, 0, CP, bc);
            cfg.dbeta_func(&eqn_param);

          } /* loop over hw blocks */
        } /* if-else for the presence of input padding */
      } /* loop over cp for computing din */

      /* b = (db * mean[nb] - ds) * a * a * a * scale; */
      /* c = -b * mean[nb] - db * a * scale; */

      for(g = 0; g < G; g++){                                                 /* compute b and c for each channel from group means and variance */
        float gds = 0.0f;
        float gdb = 0.0f;
        for(j = 0; j < group_size; j++){
          gds += ds[g*group_size + j];                                        /* Group ds and db calculation */
          gdb += db[g*group_size + j];
        }
        for(j = 0; j < group_size; j++){
          b[g*group_size + j] = (gdb * mean[n*G + g] - gds) * a[g*group_size + j] * a[g*group_size + j] * a[g*group_size + j] * scale;
          c[g*group_size + j] = -b[g*group_size + j] * mean[n*G + g] - gdb * a[g*group_size + j] * scale;
        }
      }

      for (cp = 0; cp < CP; cp++) {

        arg_array[1].primary = &a[cp*bc];
        arg_array[2].primary = &b[cp*bc];
        arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);
        arg_array[7].primary = &c[cp*bc];

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

              arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
              arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho,            wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);
              eqn_param.output.primary =    &LIBXSMM_VLA_ACCESS(5, din,  n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
              cfg.din_func(&eqn_param);

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

            arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, w, 0, CP, H, W, bc);
            arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);
            eqn_param.output.primary =    &LIBXSMM_VLA_ACCESS(5, din,  n, cp, hi, w, 0, CP, H, W, bc);
            cfg.din_func(&eqn_param);

          } /* loop over hw blocks */
        } /* if-else for the presence of input padding */
      } /* loop over cp for computing din */
    } /* loop over n for computing din */

    libxsmm_barrier_wait(cfg.barrier, ltid);

    int cp;
    for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, bc);
      cfg.all_zero_kernel(&all_zero_param);

      for (n=0; n < N; n++ ) {
        int cb;
        for(cb = 0; cb < bc; cb++){
          LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, bc) += LIBXSMM_VLA_ACCESS(3, dgamma_N, n, cp, cb, CP, bc);
          LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, bc)  += LIBXSMM_VLA_ACCESS(3, dbeta_N,  n, cp, cb, CP, bc);
        }
      }
    } /* loop over cp for finalizing dgamma and dbeta */
  } /* if-else for the group_size */

  libxsmm_barrier_wait(cfg.barrier, ltid);
}
