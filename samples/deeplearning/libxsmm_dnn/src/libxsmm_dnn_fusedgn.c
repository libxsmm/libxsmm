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

libxsmm_dnn_gn_fwd_config setup_libxsmm_dnn_gn_fwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint G, libxsmm_blasint bc,
                                 libxsmm_blasint threads, libxsmm_dnn_gn_fuse fuse_type,
                                 libxsmm_datatype datatype_in, libxsmm_datatype datatype_out, libxsmm_datatype datatype_comp ) {

  libxsmm_dnn_gn_fwd_config res;

  libxsmm_blasint ldo = bc;
  libxsmm_blasint ld  = bc;
  libxsmm_blasint tmp_ld, tmp_ld2;
  libxsmm_blasint my_eqn10;

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
  res.threads       = threads;
  res.fuse_type     = fuse_type;

  res.datatype_in   = datatype_in;
  res.datatype_out  = datatype_out;
  res.datatype_comp = datatype_comp;

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* TPP creation */
  ldo = res.G;
  unary_shape           = libxsmm_create_meltw_unary_shape(res.G, 1, res.G, ldo, res.datatype_comp, res.datatype_comp, res.datatype_comp);
  unary_flags           = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.all_zero_G_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
  if ( res.all_zero_G_kernel == NULL) {
    fprintf( stderr, "JIT for initialization by unary all zero group copy kernel failed for fwd. Bailing...!\n");
    exit(-1);
  }

  ldo = res.bc;
  unary_shape           = libxsmm_create_meltw_unary_shape(res.bc, 1, res.bc, ldo, res.datatype_comp, res.datatype_comp, res.datatype_comp);
  unary_flags           = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.all_zero_kernel   = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
  if ( res.all_zero_G_kernel == NULL) {
    fprintf( stderr, "JIT for initialization by unary all zero copy kernel failed for fwd. Bailing...!\n");
    exit(-1);
  }

  /* TPPs for reducing X and X2 in HW*/
  ld = res.bc;
  tmp_ld = res.bc;

  unary_shape           = libxsmm_create_meltw_unary_shape(res.bc, res.H*res.W / res.num_HW_blocks, ld, tmp_ld, res.datatype_in, res.datatype_comp, res.datatype_comp);
  unary_flags           = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  res.reduce_HW_kernel  = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD, unary_shape, unary_flags);
  if ( res.reduce_HW_kernel == NULL) {
      fprintf( stderr, "JIT for initialization of reduce_HW_kernel failed for fwd. Bailing...!\n");
      exit(-1);
  }

  binary_shape   = libxsmm_create_meltw_binary_shape(res.bc, 1, ld, ld, ld, res.datatype_comp, res.datatype_comp, res.datatype_comp, res.datatype_comp);
  binary_flags   = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  res.add_kernel = libxsmm_dispatch_meltw_binary_v2(LIBXSMM_MELTW_TYPE_BINARY_ADD, binary_shape, binary_flags);
  if ( res.add_kernel == NULL) {
      fprintf( stderr, "JIT for initialization of add_kernel failed for fwd. Bailing...!\n");
      exit(-1);
  }

  /* TPP for reducing groups */
  libxsmm_blasint group_size = res.C/res.G;

  ld = group_size;                /* group_size = (CP*bc)/G */
  tmp_ld = 1;

  unary_shape              = libxsmm_create_meltw_unary_shape(group_size, 1, ld, tmp_ld, res.datatype_comp, res.datatype_comp, res.datatype_comp);
  unary_flags              = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS;
  res.reduce_groups_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, unary_shape, unary_flags);
  if ( res.reduce_groups_kernel == NULL) {
      fprintf( stderr, "JIT for initialization of reduce_groups_kernel failed for fwd. Bailing...!\n");
      exit(-1);
  }

  ld = res.bc;
  tmp_ld = 1;
  unary_shape            = libxsmm_create_meltw_unary_shape(res.bc, 1, ld, tmp_ld, res.datatype_comp, res.datatype_comp, res.datatype_comp);
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

  /* init scratch (currently is not needed for the groupnorm fwd) */
  res.scratch_size = 0;

  return res;
}

libxsmm_dnn_gn_bwd_config setup_libxsmm_dnn_gn_bwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint G, libxsmm_blasint bc,
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
  res.threads       = threads;
  res.fuse_type     = fuse_type;

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
  unary_shape         = libxsmm_create_meltw_unary_shape(res.bc, 1, res.bc, ldo, res.datatype_comp, res.datatype_comp, res.datatype_comp);
  unary_flags         = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.all_zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
  if ( res.all_zero_kernel == NULL) {
    fprintf( stderr, "JIT for initialization by unary all zero copy kernel failed for fwd. Bailing...!\n");
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
  arg_shape[0].n              = res.H*res.W /res.num_HW_blocks;
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
  arg_shape[3].n    = res.H*res.W/res.num_HW_blocks;
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
  arg_shape[0].n    = res.H*res.W/res.num_HW_blocks;
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
  arg_shape[0].n    = res.H*res.W/res.num_HW_blocks;
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
  arg_shape[0].n    = res.H*res.W/res.num_HW_blocks;
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
  arg_shape[2].n              = res.H*res.W /res.num_HW_blocks;
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
  arg_shape[2].n    = res.H*res.W/res.num_HW_blocks;
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
  arg_shape[3].n              = res.H*res.W /res.num_HW_blocks;
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
  eqn_out_arg_shape.n    = res.H*res.W/res.num_HW_blocks;
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

void destroy_libxsmm_dnn_gn_fwd(libxsmm_dnn_gn_fwd_config* cfg) {
  libxsmm_barrier_destroy(cfg->barrier);

  /* when/if libxsmm_matrix_eqn_destroy gets added, destructors for equations should go here */
}

void destroy_libxsmm_dnn_gn_bwd(libxsmm_dnn_gn_bwd_config* cfg) {
  libxsmm_barrier_destroy(cfg->barrier);

}

void libxsmm_dnn_gn_fwd_exec_f32( libxsmm_dnn_gn_fwd_config cfg, const float *pinp, const float *pinp_add, const float *pgamma, const float *pbeta, float *mean, float *var, float *pout, unsigned char *prelumask,
                         float eps, int start_tid, int my_tid, void *scratch ) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint G  = cfg.G;
  const libxsmm_blasint HW = cfg.H * cfg.W;
  const libxsmm_blasint CB = cfg.bc;
  const libxsmm_blasint num_HW_blocks = cfg.num_HW_blocks;

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

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  LIBXSMM_VLA_DECL(4, const float, inp, pinp, CP, HW, CB);            /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4,       float, out, pout, CP, HW, CB);
  LIBXSMM_VLA_DECL(2, const float, gamma, pgamma, CB);                /* [CP,CB] */
  LIBXSMM_VLA_DECL(2, const float, beta, pbeta, CB);                  /* [CP,CB] */

  LIBXSMM_VLA_DECL(4, const float, inp_add, pinp_add, CP, HW, CB);   /* [N, CP, HW, bc] */

  LIBXSMM_VLA_DECL(4,       unsigned char, relumask, prelumask, CP, HW, CB/BITS_PER_CHAR);    /* [N, CP, HW, CB/BITS_PER_CHAR] */

  int np, group_size;
  group_size = (CP*CB)/G;

  libxsmm_meltw_unary_param  all_zero_param;
  libxsmm_meltw_binary_param add_param;
  libxsmm_meltw_unary_param  reduce_HW_param;
  libxsmm_meltw_unary_param  m_reduce_groups_param;
  libxsmm_meltw_unary_param  v_reduce_groups_param;
  libxsmm_meltw_unary_param  all_relu_param;

  libxsmm_matrix_arg arg_array[6];
  libxsmm_matrix_eqn_param eqn_param;

  memset( &all_zero_param,        0, sizeof(all_zero_param));
  memset( &add_param,             0, sizeof(add_param));
  memset( &reduce_HW_param,       0, sizeof(reduce_HW_param));
  memset( &m_reduce_groups_param, 0, sizeof(m_reduce_groups_param));
  memset( &v_reduce_groups_param, 0, sizeof(v_reduce_groups_param));
  memset( &all_relu_param,        0, sizeof(all_relu_param));
  memset( &eqn_param,             0, sizeof(eqn_param));

  eqn_param.inputs = arg_array;

  if (group_size <= CB){
    int cp;
    int cpxnt;
    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      np = cpxnt/CP;
      cp = cpxnt%CP;

      LIBXSMM_ALIGNED(float tmp[2*CB], 64);
      LIBXSMM_ALIGNED(float sum_X[G], 64);
      LIBXSMM_ALIGNED(float sum_X2[G], 64);
      LIBXSMM_ALIGNED(float s[CB], 64);
      LIBXSMM_ALIGNED(float b[CB], 64);

      int i, j, hwb, g;

      all_zero_param.out.primary = tmp;
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &tmp[CB];
      cfg.all_zero_kernel(&all_zero_param);

      all_zero_param.out.primary = sum_X;
      cfg.all_zero_G_kernel(&all_zero_param);
      all_zero_param.out.primary = sum_X2;
      cfg.all_zero_G_kernel(&all_zero_param);

      LIBXSMM_ALIGNED(float new_tmp[2*CB], 64);
      reduce_HW_param.out.primary   = new_tmp;                  /* [2*CB] */
      for(hwb=0; hwb < num_HW_blocks; hwb++){
        reduce_HW_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);      /* [HW_block, CB] -----> [2 * CB] */
        cfg.reduce_HW_kernel(&reduce_HW_param);

        add_param.in0.primary = tmp;
        add_param.in1.primary = new_tmp;
        add_param.out.primary = tmp;
        cfg.add_kernel(&add_param);

        add_param.in0.primary = &tmp[CB];
        add_param.in1.primary = &new_tmp[CB];
        add_param.out.primary = &tmp[CB];
        cfg.add_kernel(&add_param);
        /* for (cb = 0; cb < 2*CB; cb++) { */
        /*   tmp[cb] += new_tmp[cb]; */
        /* } */
      }

      for(i=0; i < CB; i += group_size){
        g = (cp*CB + i)/group_size;                                                                      /* determine current group */
        m_reduce_groups_param.in.primary    = &tmp[i];
        m_reduce_groups_param.out.primary   = &sum_X[g];
        v_reduce_groups_param.in.primary    = &tmp[CB + i];
        v_reduce_groups_param.out.primary   = &sum_X2[g];
        cfg.reduce_groups_kernel(&m_reduce_groups_param);
        cfg.reduce_groups_kernel(&v_reduce_groups_param);

        mean[np*G + g] = sum_X[g] / ((float)group_size * HW);
        var[np*G + g] = (sum_X2[g] / ((float)group_size * HW)) - (mean[np*G + g]*mean[np*G + g]);        /* var = E[X^2] - (E[X])^2 */

        for(j = 0; j < group_size; j++){
          s[i + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));                                          /* 1/sqrt(var(X) + eps) */
          b[i + j] = -1 * mean[np*G + g] * s[i + j];                                                     /* -E[X]/sqrt(var(X) + eps) */
        }
      }

      arg_array[1].primary = s;                                                                           /* [CB] */
      arg_array[2].primary = b;                                                                           /* [CB] */
      arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);                             /* [CB] */
      arg_array[4].primary = (void*)&LIBXSMM_VLA_ACCESS(2, beta, cp, 0, CB);                              /* [CB] */

      for(hwb=0; hwb < num_HW_blocks; hwb++){
        if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU ||  cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
          arg_array[5].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp_add, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);    /* [HW, CB] */
        }

        if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
          eqn_param.output.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                        (void*)&LIBXSMM_VLA_ACCESS(4, relumask, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, (CB/BITS_PER_CHAR)) : NULL );
        }

        arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB); /* [HW, CB] */
        eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, out, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);    /* [HW, CB] */
        cfg.func10(&eqn_param);                                                                                   /* Normalization equation -> y = relu(((s*x + b)*gamma + beta) + inp_add) */
      }
    }
  } else{                                                         /* Case when group_size > CB */
    for ( np = thr_begin_N; np < thr_end_N; ++np ) {
      LIBXSMM_ALIGNED(float tmp[2*CB], 64);
      LIBXSMM_ALIGNED(float sum_X[G], 64);
      LIBXSMM_ALIGNED(float sum_X2[G], 64);
      LIBXSMM_ALIGNED(float s[CP*CB], 64);
      LIBXSMM_ALIGNED(float b[CP*CB], 64);

      int i, j, cp, hwb, g;
      float m, v;

      libxsmm_meltw_unary_param  m_reduce_rows_param;
      libxsmm_meltw_unary_param  v_reduce_rows_param;

      memset( &m_reduce_rows_param, 0, sizeof(m_reduce_rows_param));
      memset( &v_reduce_rows_param, 0, sizeof(v_reduce_rows_param));

      all_zero_param.out.primary = sum_X;
      cfg.all_zero_G_kernel(&all_zero_param);
      all_zero_param.out.primary = sum_X2;
      cfg.all_zero_G_kernel(&all_zero_param);

      LIBXSMM_ALIGNED(float new_tmp[2*CB], 64);
      for (cp = 0; cp < CP; cp++){                      /* [cp, HW, CB] */
        all_zero_param.out.primary = tmp;
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = &tmp[CB];
        cfg.all_zero_kernel(&all_zero_param);
        /* for (cb = 0; cb < 2*CB; cb++) { */
        /*   tmp[cb] = 0.0f; */
        /* } */

        reduce_HW_param.out.primary = new_tmp;                  /* [2*CB] */
        for(hwb=0; hwb < num_HW_blocks; hwb++){
          reduce_HW_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);      /* [HW, CB] -----> [2 * CB] */
          cfg.reduce_HW_kernel(&reduce_HW_param);

          add_param.in0.primary = tmp;
          add_param.in1.primary = new_tmp;
          add_param.out.primary = tmp;
          cfg.add_kernel(&add_param);

          add_param.in0.primary = &tmp[CB];
          add_param.in1.primary = &new_tmp[CB];
          add_param.out.primary = &tmp[CB];
          cfg.add_kernel(&add_param);
          /* #pragma omp simd */
          /* for (cb = 0; cb < 2*CB; cb++) { */
          /*   tmp[cb] += new_tmp[cb]; */
          /* } */
        }

        if (group_size >= CB){                                 /* Group size >= block size  (Ex.- CP = 4, CB = 16, G = 2, group_size = 32) */
          g = (cp*CB)/group_size;                              /* determine current group */
          m_reduce_rows_param.in.primary    = tmp;
          m_reduce_rows_param.out.primary   = &m;
          v_reduce_rows_param.in.primary    = &tmp[CB];
          v_reduce_rows_param.out.primary   = &v;
          cfg.reduce_rows_kernel(&m_reduce_rows_param);
          cfg.reduce_rows_kernel(&v_reduce_rows_param);
          sum_X[g] += m;
          sum_X2[g] += v;
        }
        else{                                                 /* Group size < block size  (Ex.- CP = 4, CB = 16, G = 32, group_size = 2) */
          for(i=0; i < CB; i += group_size){
            m_reduce_groups_param.in.primary    = &tmp[i];
            m_reduce_groups_param.out.primary   = &sum_X[cp*(CB/group_size) + (i/group_size)];
            v_reduce_groups_param.in.primary    = &tmp[CB + i];
            v_reduce_groups_param.out.primary   = &sum_X2[cp*(CB/group_size) + (i/group_size)];
            cfg.reduce_groups_kernel(&m_reduce_groups_param);
            cfg.reduce_groups_kernel(&v_reduce_groups_param);
          }
        }
      }

      /* mean and variance calculation */
      for(g = 0; g < G; g++){
        mean[np*G + g] = sum_X[g] / ((float)group_size * HW);
        var[np*G + g] = (sum_X2[g] / ((float)group_size * HW)) - (mean[np*G + g]*mean[np*G + g]);        /* var = E[X^2] - (E[X])^2 */

        for(j = 0; j < group_size; j++){
          s[g*group_size + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));                               /* 1/sqrt(var(X) + eps) */
          b[g*group_size + j] = -1 * mean[np*G + g] * s[g*group_size + j];                               /* -E[X]/sqrt(var(X) + eps) */
        }
      }

      for (cp = 0; cp < CP; cp++){

        arg_array[1].primary = &s[cp*CB];                                                                /* [CB] */
        arg_array[2].primary = &b[cp*CB];                                                                /* [CB] */
        arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);                          /* [CB] */
        arg_array[4].primary = (void*)&LIBXSMM_VLA_ACCESS(2, beta, cp, 0, CB);                           /* [CB] */

        for(hwb=0; hwb < num_HW_blocks; hwb++){
          if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU ||  cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
            arg_array[5].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp_add, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);    /* [HW, CB] */
          }

          if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
            eqn_param.output.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                          (void*)&LIBXSMM_VLA_ACCESS(4, relumask, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, (CB/BITS_PER_CHAR)) : NULL );
          }

          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB); /* [HW, CB] */
          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, out, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);    /* [HW, CB] */
          cfg.func10(&eqn_param);                                                                                   /* Normalization equation -> y = relu(((s*x + b)*gamma + beta) + inp_add) */
        }
      }
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);

}

void libxsmm_dnn_gn_fwd_exec_bf16( libxsmm_dnn_gn_fwd_config cfg, const libxsmm_bfloat16 *pinp, const libxsmm_bfloat16 *pinp_add,
                          const float *pgamma, const float *pbeta, float *mean, float *var,
                          libxsmm_bfloat16 *pout, unsigned char *prelumask,
                          float eps, int start_tid, int my_tid, void *scratch ) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint G  = cfg.G;
  const libxsmm_blasint HW = cfg.H * cfg.W;
  const libxsmm_blasint CB = cfg.bc;
  const libxsmm_blasint num_HW_blocks = cfg.num_HW_blocks;

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

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  LIBXSMM_VLA_DECL(4, const libxsmm_bfloat16, inp, pinp, CP, HW, CB);           /* [N, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4,       libxsmm_bfloat16, out, pout, CP, HW, CB);
  LIBXSMM_VLA_DECL(2, const float, gamma, pgamma, CB);                          /* [CP,CB] */
  LIBXSMM_VLA_DECL(2, const float, beta, pbeta, CB);                            /* [CP,CB] */

  LIBXSMM_VLA_DECL(4, const libxsmm_bfloat16, inp_add, pinp_add, CP, HW, CB);   /* [N, CP, HW, bc] */

  LIBXSMM_VLA_DECL(4,       unsigned char, relumask, prelumask, CP, HW, CB/BITS_PER_CHAR);    /* [N, CP, HW, CB/BITS_PER_CHAR] */

  int np, group_size;
  group_size = (CP*CB)/G;

  libxsmm_meltw_unary_param  all_zero_param;
  libxsmm_meltw_binary_param add_param;
  libxsmm_meltw_unary_param  reduce_HW_param;
  libxsmm_meltw_unary_param  m_reduce_groups_param;
  libxsmm_meltw_unary_param  v_reduce_groups_param;
  libxsmm_meltw_unary_param  all_relu_param;

  libxsmm_matrix_arg arg_array[6];
  libxsmm_matrix_eqn_param eqn_param;

  memset( &all_zero_param,        0, sizeof(all_zero_param));
  memset( &add_param,             0, sizeof(add_param));
  memset( &reduce_HW_param,       0, sizeof(reduce_HW_param));
  memset( &m_reduce_groups_param, 0, sizeof(m_reduce_groups_param));
  memset( &v_reduce_groups_param, 0, sizeof(v_reduce_groups_param));
  memset( &all_relu_param,        0, sizeof(all_relu_param));
  memset( &eqn_param,             0, sizeof(eqn_param));

  eqn_param.inputs = arg_array;

  if (group_size <= CB){
    int cp;
    int cpxnt;
    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      np = cpxnt/CP;
      cp = cpxnt%CP;

      LIBXSMM_ALIGNED(float tmp[2*CB], 64);
      LIBXSMM_ALIGNED(float sum_X[G], 64);
      LIBXSMM_ALIGNED(float sum_X2[G], 64);
      LIBXSMM_ALIGNED(float s[CB], 64);
      LIBXSMM_ALIGNED(float b[CB], 64);

      int i, j, hwb, g;

      all_zero_param.out.primary = tmp;
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &tmp[CB];
      cfg.all_zero_kernel(&all_zero_param);

      all_zero_param.out.primary = sum_X;
      cfg.all_zero_G_kernel(&all_zero_param);
      all_zero_param.out.primary = sum_X2;
      cfg.all_zero_G_kernel(&all_zero_param);

      LIBXSMM_ALIGNED(float new_tmp[2*CB], 64);
      reduce_HW_param.out.primary   = new_tmp;                  /* [2*CB] */
      for(hwb=0; hwb < num_HW_blocks; hwb++){
        reduce_HW_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);      /* [HW_block, CB] -----> [2 * CB] */
        cfg.reduce_HW_kernel(&reduce_HW_param);

        add_param.in0.primary = tmp;
        add_param.in1.primary = new_tmp;
        add_param.out.primary = tmp;
        cfg.add_kernel(&add_param);

        add_param.in0.primary = &tmp[CB];
        add_param.in1.primary = &new_tmp[CB];
        add_param.out.primary = &tmp[CB];
        cfg.add_kernel(&add_param);
        /* for (cb = 0; cb < 2*CB; cb++) { */
        /*   tmp[cb] += new_tmp[cb]; */
        /* } */
      }

      for(i=0; i < CB; i += group_size){
        g = (cp*CB + i)/group_size;                                                                      /* determine current group */
        m_reduce_groups_param.in.primary    = &tmp[i];
        m_reduce_groups_param.out.primary   = &sum_X[g];
        v_reduce_groups_param.in.primary    = &tmp[CB + i];
        v_reduce_groups_param.out.primary   = &sum_X2[g];
        cfg.reduce_groups_kernel(&m_reduce_groups_param);
        cfg.reduce_groups_kernel(&v_reduce_groups_param);

        mean[np*G + g] = sum_X[g] / ((float)group_size * HW);
        var[np*G + g] = (sum_X2[g] / ((float)group_size * HW)) - (mean[np*G + g]*mean[np*G + g]);        /* var = E[X^2] - (E[X])^2 */

        for(j = 0; j < group_size; j++){
          s[i + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));                                          /* 1/sqrt(var(X) + eps) */
          b[i + j] = -1 * mean[np*G + g] * s[i + j];                                                     /* -E[X]/sqrt(var(X) + eps) */
        }
      }

      arg_array[1].primary = s;                                                                           /* [CB] */
      arg_array[2].primary = b;                                                                           /* [CB] */
      arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);                             /* [CB] */
      arg_array[4].primary = (void*)&LIBXSMM_VLA_ACCESS(2, beta, cp, 0, CB);                              /* [CB] */

      for(hwb=0; hwb < num_HW_blocks; hwb++){
        if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU ||  cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
          arg_array[5].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp_add, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);    /* [HW, CB] */
        }

        if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
          eqn_param.output.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                        (void*)&LIBXSMM_VLA_ACCESS(4, relumask, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, (CB/BITS_PER_CHAR)) : NULL );
        }

        arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB); /* [HW, CB] */
        eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, out, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);    /* [HW, CB] */
        cfg.func10(&eqn_param);                                                                                   /* Normalization equation -> y = relu(((s*x + b)*gamma + beta) + inp_add) */
      }
    }
  } else{                                                         /* Case when group_size > CB */
    for ( np = thr_begin_N; np < thr_end_N; ++np ) {
      LIBXSMM_ALIGNED(float tmp[2*CB], 64);
      LIBXSMM_ALIGNED(float sum_X[G], 64);
      LIBXSMM_ALIGNED(float sum_X2[G], 64);
      LIBXSMM_ALIGNED(float s[CP*CB], 64);
      LIBXSMM_ALIGNED(float b[CP*CB], 64);

      int i, j, cp, hwb, g;
      float m, v;

      libxsmm_meltw_unary_param  m_reduce_rows_param;
      libxsmm_meltw_unary_param  v_reduce_rows_param;

      memset( &m_reduce_rows_param, 0, sizeof(m_reduce_rows_param));
      memset( &v_reduce_rows_param, 0, sizeof(v_reduce_rows_param));

      all_zero_param.out.primary = sum_X;
      cfg.all_zero_G_kernel(&all_zero_param);
      all_zero_param.out.primary = sum_X2;
      cfg.all_zero_G_kernel(&all_zero_param);

      LIBXSMM_ALIGNED(float new_tmp[2*CB], 64);
      for (cp = 0; cp < CP; cp++){                      /* [cp, HW, CB] */
        all_zero_param.out.primary = tmp;
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = &tmp[CB];
        cfg.all_zero_kernel(&all_zero_param);
        /* for (cb = 0; cb < 2*CB; cb++) { */
        /*   tmp[cb] = 0.0f; */
        /* } */

        reduce_HW_param.out.primary = new_tmp;                  /* [2*CB] */
        for(hwb=0; hwb < num_HW_blocks; hwb++){
          reduce_HW_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);      /* [HW, CB] -----> [2 * CB] */
          cfg.reduce_HW_kernel(&reduce_HW_param);

          add_param.in0.primary = tmp;
          add_param.in1.primary = new_tmp;
          add_param.out.primary = tmp;
          cfg.add_kernel(&add_param);

          add_param.in0.primary = &tmp[CB];
          add_param.in1.primary = &new_tmp[CB];
          add_param.out.primary = &tmp[CB];
          cfg.add_kernel(&add_param);
          /* #pragma omp simd */
          /* for (cb = 0; cb < 2*CB; cb++) { */
          /*   tmp[cb] += new_tmp[cb]; */
          /* } */
        }

        if (group_size >= CB){                                 /* Group size >= block size  (Ex.- CP = 4, CB = 16, G = 2, group_size = 32) */
          g = (cp*CB)/group_size;                              /* determine current group */
          m_reduce_rows_param.in.primary    = tmp;
          m_reduce_rows_param.out.primary   = &m;
          v_reduce_rows_param.in.primary    = &tmp[CB];
          v_reduce_rows_param.out.primary   = &v;
          cfg.reduce_rows_kernel(&m_reduce_rows_param);
          cfg.reduce_rows_kernel(&v_reduce_rows_param);
          sum_X[g] += m;
          sum_X2[g] += v;
        }
        else{                                                 /* Group size < block size  (Ex.- CP = 4, CB = 16, G = 32, group_size = 2) */
          for(i=0; i < CB; i += group_size){
            m_reduce_groups_param.in.primary    = &tmp[i];
            m_reduce_groups_param.out.primary   = &sum_X[cp*(CB/group_size) + (i/group_size)];
            v_reduce_groups_param.in.primary    = &tmp[CB + i];
            v_reduce_groups_param.out.primary   = &sum_X2[cp*(CB/group_size) + (i/group_size)];
            cfg.reduce_groups_kernel(&m_reduce_groups_param);
            cfg.reduce_groups_kernel(&v_reduce_groups_param);
          }
        }
      }

      /* mean and variance calculation */
      for(g = 0; g < G; g++){
        mean[np*G + g] = sum_X[g] / ((float)group_size * HW);
        var[np*G + g] = (sum_X2[g] / ((float)group_size * HW)) - (mean[np*G + g]*mean[np*G + g]);        /* var = E[X^2] - (E[X])^2 */

        for(j = 0; j < group_size; j++){
          s[g*group_size + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));                               /* 1/sqrt(var(X) + eps) */
          b[g*group_size + j] = -1 * mean[np*G + g] * s[g*group_size + j];                               /* -E[X]/sqrt(var(X) + eps) */
        }
      }

      for (cp = 0; cp < CP; cp++){

        arg_array[1].primary = &s[cp*CB];                                                                /* [CB] */
        arg_array[2].primary = &b[cp*CB];                                                                /* [CB] */
        arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);                          /* [CB] */
        arg_array[4].primary = (void*)&LIBXSMM_VLA_ACCESS(2, beta, cp, 0, CB);                           /* [CB] */

        for(hwb=0; hwb < num_HW_blocks; hwb++){
          if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU ||  cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
            arg_array[5].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp_add, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);    /* [HW, CB] */
          }

          if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
            eqn_param.output.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                          (void*)&LIBXSMM_VLA_ACCESS(4, relumask, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, (CB/BITS_PER_CHAR)) : NULL );
          }

          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB); /* [HW, CB] */
          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, out, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);    /* [HW, CB] */
          cfg.func10(&eqn_param);                                                                                   /* Normalization equation -> y = relu(((s*x + b)*gamma + beta) + inp_add) */
        }
      }
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);

}


void libxsmm_dnn_gn_bwd_exec_f32( libxsmm_dnn_gn_bwd_config cfg, float *pdout, const float *pinp, const float *mean, const float *var, const float *pgamma, const unsigned char *prelumask,
                         float *pdin, float *pdin_add, float *pdgamma, float *pdbeta, float eps,
                         int start_tid, int my_tid, void *scratch) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint G  = cfg.G;
  const libxsmm_blasint HW = cfg.H * cfg.W;
  const libxsmm_blasint CB = cfg.bc;
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

  int group_size = (CP*CB)/G;

  const float scale = 1.0f / ((float)group_size * HW);

  LIBXSMM_VLA_DECL(4,       float, din, pdin, CP, HW, CB);
  LIBXSMM_VLA_DECL(4, const float, inp, pinp, CP, HW, CB);
  LIBXSMM_VLA_DECL(4,       float, dout, pdout, CP, HW, CB);
  LIBXSMM_VLA_DECL(2, const float, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2,       float, dgamma, pdgamma, CB);
  LIBXSMM_VLA_DECL(2,       float, dbeta, pdbeta, CB);

  LIBXSMM_VLA_DECL(4,       float, din_add, pdin_add, CP, HW, CB);     /* [N, CP, HW, bc] */

  float alpha = 0.0f;
  LIBXSMM_VLA_DECL(4, const unsigned char, relumask, prelumask, CP, HW, CB/BITS_PER_CHAR);    /* [N, CP, HW, CB/BITS_PER_CHAR] */

  const libxsmm_blasint dbeta_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + N * CP * CB), 64) - ((uintptr_t)(scratch))) / sizeof(float);
  LIBXSMM_VLA_DECL(3, float, dgamma_N, ((float*)scratch),                  CP, CB);  /* [N, CP, CB] */
  LIBXSMM_ASSUME_ALIGNED(dgamma_N_, 64);
  LIBXSMM_VLA_DECL(3, float, dbeta_N,  ((float*)scratch) + dbeta_N_offset, CP, CB);  /* [N, CP, CB] */
  LIBXSMM_ASSUME_ALIGNED(dbeta_N_, 64);

  if (group_size <= CB){
    LIBXSMM_ALIGNED(float a[CB], 64);
    LIBXSMM_ALIGNED(float b[CB], 64);
    LIBXSMM_ALIGNED(float c[CB], 64);
    LIBXSMM_ALIGNED(float ds[CB], 64);
    LIBXSMM_ALIGNED(float db[CB], 64);

    int np, cp;
    int cpxnt;
    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      np = cpxnt/CP;
      cp = cpxnt%CP;

      int j, g, hwb, lg;

      /* for(j = 0; j < CB; j++){
          dgamma_N[np*CP*CB + cp*CB + j] = 0.0f;
          dbeta_N[np*CP*CB + cp*CB + j] = 0.0f;
       } */

      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, np, cp, 0, CP, CB);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, np, cp, 0, CP, CB);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = ds;
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = db;
      cfg.all_zero_kernel(&all_zero_param);

      /* compute a and b for each channel from group means and variance */
      for(g = (cp*CB)/group_size; g < ((cp+1)*CB)/group_size; g++){
        lg = g - (cp*CB)/group_size;
        for(j = 0; j < group_size; j++){
          a[lg*group_size + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));
          b[lg*group_size + j] = -a[lg*group_size + j]*mean[np*G + g];
        }
      }

      arg_array[1].primary = a;
      arg_array[2].primary = b;
      arg_array[4].primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, np, cp, 0, CP, CB);
      arg_array[5].primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, np, cp, 0, CP, CB);
      arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
      arg_array[8].primary = ds;
      arg_array[9].primary = db;

      for(hwb=0; hwb < num_HW_blocks; hwb++){
        if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
          all_relu_param.op.primary   = (void*)(&alpha);
          all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);      /* [HW,CB] */
          all_relu_param.in.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                           (void*)&LIBXSMM_VLA_ACCESS(4, relumask, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB/8)
                                           : NULL /*&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB) */ ); /* dout_fwd ? nonsense? */
          all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);      /* [HW,CB] */
          cfg.inv_relu_kernel(&all_relu_param);
        } /* ReLU/mask */
        if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
          ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(4, dout,    np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(4, din_add, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          cfg.ewise_copy_kernel(&ewise_copy_param);
        } /* Eltwise */

        arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
        arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);

        eqn_param.output.primary = ds;
        cfg.ds_func(&eqn_param);

        eqn_param.output.primary = db;
        cfg.db_func(&eqn_param);

        eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, np, cp, 0, CP, CB);
        cfg.dgamma_func(&eqn_param);

        eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, np, cp, 0, CP, CB);
        cfg.dbeta_func(&eqn_param);
      }

      /* b = (db * mean[nb] - ds) * a * a * a * scale; */
      /* c = -b * mean[nb] - db * a * scale; */

      for(g = (cp*CB)/group_size; g < ((cp+1)*CB)/group_size; g++){            /* compute b and c for each channel from group means and variance */
        lg = g - (cp*CB)/group_size;
        float gds = 0.0f;
        float gdb = 0.0f;
        for(j = 0; j < group_size; j++){
          gds += ds[lg*group_size + j];                                        /* Group ds and db calculation */
          gdb += db[lg*group_size + j];
        }
        for(j = 0; j < group_size; j++){
          b[lg*group_size + j] = (gdb * mean[np*G + g] - gds) * a[lg*group_size + j] * a[lg*group_size + j] * a[lg*group_size + j] * scale;
          c[lg*group_size + j] = -b[lg*group_size + j] * mean[np*G + g] - gdb * a[lg*group_size + j] * scale;
        }
      }

      arg_array[1].primary = a;
      arg_array[2].primary = b;
      arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
      arg_array[7].primary = c;

      for(hwb=0; hwb < num_HW_blocks; hwb++){
        arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp,  np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
        arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
        eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, din, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
        cfg.din_func(&eqn_param);
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid); /* not needed? */

    for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, CB);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, CB);
      cfg.all_zero_kernel(&all_zero_param);

      for (np=0; np < N; np++ ) {
        int cb;
        for(cb = 0; cb < CB; cb++){
          LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, CB) += LIBXSMM_VLA_ACCESS(3, dgamma_N, np, cp, cb, CP, CB);
          LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, CB)  += LIBXSMM_VLA_ACCESS(3, dbeta_N,  np, cp, cb, CP, CB);
        }
      }
    }
  } else {
    LIBXSMM_ALIGNED(float a[CP*CB], 64);
    LIBXSMM_ALIGNED(float b[CP*CB], 64);
    LIBXSMM_ALIGNED(float c[CP*CB], 64);
    LIBXSMM_ALIGNED(float ds[CP*CB], 64);
    LIBXSMM_ALIGNED(float db[CP*CB], 64);
    int np;

    for ( np = thr_begin_N; np < thr_end_N; ++np ) {
      int j, g, cp, hwb;

      /* for(j = 0; j < CP*CB; j++){ */
      /*   dgamma_N[np*CP*CB + j] = 0.0f; */
      /*   dbeta_N[np*CP*CB + j] = 0.0f; */
      /* } */

      for (cp = 0; cp < CP; cp++) {
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, np, cp, 0, CP, CB);
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, np, cp, 0, CP, CB);
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = &ds[cp*CB];
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = &db[cp*CB];
        cfg.all_zero_kernel(&all_zero_param);
      }

      for(g = 0; g < G; g++){                                                  /* compute a and b for each channel from group means and variance */
        for(j = 0; j < group_size; j++){
          a[g*group_size + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));
          b[g*group_size + j] = -a[g*group_size + j]*mean[np*G + g];
        }
      }

      for (cp = 0; cp < CP; cp++) {
        arg_array[1].primary = &a[cp*CB];
        arg_array[2].primary = &b[cp*CB];
        arg_array[4].primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, np, cp, 0, CP, CB);
        arg_array[5].primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N,  np, cp, 0, CP, CB);
        arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
        arg_array[8].primary = &ds[cp*CB];
        arg_array[9].primary = &db[cp*CB];

        for(hwb=0; hwb < num_HW_blocks; hwb++){
          if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
            all_relu_param.op.primary   = (void*)(&alpha);
            all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);      /* [HW,CB] */
            all_relu_param.in.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                             (void*)&LIBXSMM_VLA_ACCESS(4, relumask, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB/8)
                                             : NULL /*&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB) */ ); /* dout_fwd ? nonsense? */
            all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);      /* [HW,CB] */
            cfg.inv_relu_kernel(&all_relu_param);
          } /* ReLU/mask */
          if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
            ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(4, dout,    np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
            ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(4, din_add, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
            cfg.ewise_copy_kernel(&ewise_copy_param);
          } /* Eltwise */

          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp,  np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);

          eqn_param.output.primary = &ds[cp*CB];
          cfg.ds_func(&eqn_param);

          eqn_param.output.primary = &db[cp*CB];
          cfg.db_func(&eqn_param);

          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, np, cp, 0, CP, CB);
          cfg.dgamma_func(&eqn_param);

          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, np, cp, 0, CP, CB);
          cfg.dbeta_func(&eqn_param);
        }
      }

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
          b[g*group_size + j] = (gdb * mean[np*G + g] - gds) * a[g*group_size + j] * a[g*group_size + j] * a[g*group_size + j] * scale;
          c[g*group_size + j] = -b[g*group_size + j] * mean[np*G + g] - gdb * a[g*group_size + j] * scale;
        }
      }

      for (cp = 0; cp < CP; cp++) {

        arg_array[1].primary = &a[cp*CB];
        arg_array[2].primary = &b[cp*CB];
        arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
        arg_array[7].primary = &c[cp*CB];

        for(hwb=0; hwb < num_HW_blocks; hwb++){
          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp,  np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, din, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          cfg.din_func(&eqn_param);
        }
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);

    int cp;
    for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, CB);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, CB);
      cfg.all_zero_kernel(&all_zero_param);

      for (np=0; np < N; np++ ) {
        int cb;
        for(cb = 0; cb < CB; cb++){
          LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, CB) += LIBXSMM_VLA_ACCESS(3, dgamma_N, np, cp, cb, CP, CB);
          LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, CB)  += LIBXSMM_VLA_ACCESS(3, dbeta_N,  np, cp, cb, CP, CB);
        }
      }
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);
}

void libxsmm_dnn_gn_bwd_exec_bf16( libxsmm_dnn_gn_bwd_config cfg, libxsmm_bfloat16 *pdout, const libxsmm_bfloat16 *pinp, const float *mean, const float *var, const float *pgamma, const unsigned char *prelumask,
                          libxsmm_bfloat16 *pdin, libxsmm_bfloat16 *pdin_add, float *pdgamma, float *pdbeta, float eps,
                          int start_tid, int my_tid, void *scratch) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint G  = cfg.G;
  const libxsmm_blasint HW = cfg.H * cfg.W;
  const libxsmm_blasint CB = cfg.bc;
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

  int group_size = (CP*CB)/G;

  const float scale = 1.0f / ((float)group_size * HW);

  LIBXSMM_VLA_DECL(4,       libxsmm_bfloat16, din, pdin, CP, HW, CB);
  LIBXSMM_VLA_DECL(4, const libxsmm_bfloat16, inp, pinp, CP, HW, CB);
  LIBXSMM_VLA_DECL(4,       libxsmm_bfloat16, dout, pdout, CP, HW, CB);
  LIBXSMM_VLA_DECL(2, const float, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2,       float, dgamma, pdgamma, CB);
  LIBXSMM_VLA_DECL(2,       float, dbeta, pdbeta, CB);

  LIBXSMM_VLA_DECL(4,       libxsmm_bfloat16, din_add, pdin_add, CP, HW, CB);     /* [N, CP, HW, bc] */

  float alpha = 0.0f;
  LIBXSMM_VLA_DECL(4, const unsigned char, relumask, prelumask, CP, HW, CB/BITS_PER_CHAR);    /* [N, CP, HW, CB/BITS_PER_CHAR] */

  const libxsmm_blasint dbeta_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + N * CP * CB), 64) - ((uintptr_t)(scratch))) / sizeof(float);
  LIBXSMM_VLA_DECL(3, float, dgamma_N, ((float*)scratch),                  CP, CB);  /* [N, CP, CB] */
  LIBXSMM_ASSUME_ALIGNED(dgamma_N_, 64);
  LIBXSMM_VLA_DECL(3, float, dbeta_N,  ((float*)scratch) + dbeta_N_offset, CP, CB);  /* [N, CP, CB] */
  LIBXSMM_ASSUME_ALIGNED(dbeta_N_, 64);

  if (group_size <= CB){
    LIBXSMM_ALIGNED(float a[CB], 64);
    LIBXSMM_ALIGNED(float b[CB], 64);
    LIBXSMM_ALIGNED(float c[CB], 64);
    LIBXSMM_ALIGNED(float ds[CB], 64);
    LIBXSMM_ALIGNED(float db[CB], 64);

    int np, cp;
    int cpxnt;
    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      np = cpxnt/CP;
      cp = cpxnt%CP;

      int j, g, hwb, lg;

      /* for(j = 0; j < CB; j++){
          dgamma_N[np*CP*CB + cp*CB + j] = 0.0f;
          dbeta_N[np*CP*CB + cp*CB + j] = 0.0f;
       } */

      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, np, cp, 0, CP, CB);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, np, cp, 0, CP, CB);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = ds;
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = db;
      cfg.all_zero_kernel(&all_zero_param);

      /* compute a and b for each channel from group means and variance */
      for(g = (cp*CB)/group_size; g < ((cp+1)*CB)/group_size; g++){
        lg = g - (cp*CB)/group_size;
        for(j = 0; j < group_size; j++){
          a[lg*group_size + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));
          b[lg*group_size + j] = -a[lg*group_size + j]*mean[np*G + g];
        }
      }

      arg_array[1].primary = a;
      arg_array[2].primary = b;
      arg_array[4].primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, np, cp, 0, CP, CB);
      arg_array[5].primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, np, cp, 0, CP, CB);
      arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
      arg_array[8].primary = ds;
      arg_array[9].primary = db;

      for(hwb=0; hwb < num_HW_blocks; hwb++){
        if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
          all_relu_param.op.primary   = (void*)(&alpha);
          all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);      /* [HW,CB] */
          all_relu_param.in.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                           (void*)&LIBXSMM_VLA_ACCESS(4, relumask, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB/8)
                                           : NULL /*&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB) */ ); /* dout_fwd ? nonsense? */
          all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);      /* [HW,CB] */
          cfg.inv_relu_kernel(&all_relu_param);
        } /* ReLU/mask */
        if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
          ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(4, dout,    np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(4, din_add, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          cfg.ewise_copy_kernel(&ewise_copy_param);
        } /* Eltwise */

        arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
        arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);

        eqn_param.output.primary = ds;
        cfg.ds_func(&eqn_param);

        eqn_param.output.primary = db;
        cfg.db_func(&eqn_param);

        eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, np, cp, 0, CP, CB);
        cfg.dgamma_func(&eqn_param);

        eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, np, cp, 0, CP, CB);
        cfg.dbeta_func(&eqn_param);
      }

      /* b = (db * mean[nb] - ds) * a * a * a * scale; */
      /* c = -b * mean[nb] - db * a * scale; */

      for(g = (cp*CB)/group_size; g < ((cp+1)*CB)/group_size; g++){            /* compute b and c for each channel from group means and variance */
        lg = g - (cp*CB)/group_size;
        float gds = 0.0f;
        float gdb = 0.0f;
        for(j = 0; j < group_size; j++){
          gds += ds[lg*group_size + j];                                        /* Group ds and db calculation */
          gdb += db[lg*group_size + j];
        }
        for(j = 0; j < group_size; j++){
          b[lg*group_size + j] = (gdb * mean[np*G + g] - gds) * a[lg*group_size + j] * a[lg*group_size + j] * a[lg*group_size + j] * scale;
          c[lg*group_size + j] = -b[lg*group_size + j] * mean[np*G + g] - gdb * a[lg*group_size + j] * scale;
        }
      }

      arg_array[1].primary = a;
      arg_array[2].primary = b;
      arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
      arg_array[7].primary = c;

      for(hwb=0; hwb < num_HW_blocks; hwb++){
        arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp,  np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
        arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
        eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, din, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
        cfg.din_func(&eqn_param);
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid); /* not needed? */

    for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, CB);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, CB);
      cfg.all_zero_kernel(&all_zero_param);

      for (np=0; np < N; np++ ) {
        int cb;
        for(cb = 0; cb < CB; cb++){
          LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, CB) += LIBXSMM_VLA_ACCESS(3, dgamma_N, np, cp, cb, CP, CB);
          LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, CB)  += LIBXSMM_VLA_ACCESS(3, dbeta_N,  np, cp, cb, CP, CB);
        }
      }
    }
  } else {
    LIBXSMM_ALIGNED(float a[CP*CB], 64);
    LIBXSMM_ALIGNED(float b[CP*CB], 64);
    LIBXSMM_ALIGNED(float c[CP*CB], 64);
    LIBXSMM_ALIGNED(float ds[CP*CB], 64);
    LIBXSMM_ALIGNED(float db[CP*CB], 64);
    int np;

    for ( np = thr_begin_N; np < thr_end_N; ++np ) {
      int j, g, cp, hwb;

      /* for(j = 0; j < CP*CB; j++){ */
      /*   dgamma_N[np*CP*CB + j] = 0.0f; */
      /*   dbeta_N[np*CP*CB + j] = 0.0f; */
      /* } */

      for (cp = 0; cp < CP; cp++) {
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, np, cp, 0, CP, CB);
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, np, cp, 0, CP, CB);
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = &ds[cp*CB];
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = &db[cp*CB];
        cfg.all_zero_kernel(&all_zero_param);
      }

      for(g = 0; g < G; g++){                                                  /* compute a and b for each channel from group means and variance */
        for(j = 0; j < group_size; j++){
          a[g*group_size + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));
          b[g*group_size + j] = -a[g*group_size + j]*mean[np*G + g];
        }
      }

      for (cp = 0; cp < CP; cp++) {
        arg_array[1].primary = &a[cp*CB];
        arg_array[2].primary = &b[cp*CB];
        arg_array[4].primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, np, cp, 0, CP, CB);
        arg_array[5].primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N,  np, cp, 0, CP, CB);
        arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
        arg_array[8].primary = &ds[cp*CB];
        arg_array[9].primary = &db[cp*CB];

        for(hwb=0; hwb < num_HW_blocks; hwb++){
          if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
            all_relu_param.op.primary   = (void*)(&alpha);
            all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);      /* [HW,CB] */
            all_relu_param.in.secondary = ((cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                             (void*)&LIBXSMM_VLA_ACCESS(4, relumask, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB/8)
                                             : NULL /*&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB) */ ); /* dout_fwd ? nonsense? */
            all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);      /* [HW,CB] */
            cfg.inv_relu_kernel(&all_relu_param);
          } /* ReLU/mask */
          if (cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK) {
            ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(4, dout,    np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
            ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(4, din_add, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
            cfg.ewise_copy_kernel(&ewise_copy_param);
          } /* Eltwise */

          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp,  np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);

          eqn_param.output.primary = &ds[cp*CB];
          cfg.ds_func(&eqn_param);

          eqn_param.output.primary = &db[cp*CB];
          cfg.db_func(&eqn_param);

          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, np, cp, 0, CP, CB);
          cfg.dgamma_func(&eqn_param);

          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, np, cp, 0, CP, CB);
          cfg.dbeta_func(&eqn_param);
        }
      }

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
          b[g*group_size + j] = (gdb * mean[np*G + g] - gds) * a[g*group_size + j] * a[g*group_size + j] * a[g*group_size + j] * scale;
          c[g*group_size + j] = -b[g*group_size + j] * mean[np*G + g] - gdb * a[g*group_size + j] * scale;
        }
      }

      for (cp = 0; cp < CP; cp++) {

        arg_array[1].primary = &a[cp*CB];
        arg_array[2].primary = &b[cp*CB];
        arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
        arg_array[7].primary = &c[cp*CB];

        for(hwb=0; hwb < num_HW_blocks; hwb++){
          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp,  np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          arg_array[3].primary =        &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, din, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
          cfg.din_func(&eqn_param);
        }
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);

    int cp;
    for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, CB);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, CB);
      cfg.all_zero_kernel(&all_zero_param);

      for (np=0; np < N; np++ ) {
        int cb;
        for(cb = 0; cb < CB; cb++){
          LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, CB) += LIBXSMM_VLA_ACCESS(3, dgamma_N, np, cp, cb, CP, CB);
          LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, CB)  += LIBXSMM_VLA_ACCESS(3, dbeta_N,  np, cp, cb, CP, CB);
        }
      }
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);
}
