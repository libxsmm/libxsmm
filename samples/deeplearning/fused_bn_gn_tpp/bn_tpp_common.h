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

#define NUM_HW_BLOCKS (16)

/* include c-based dnn library */

#if 0

#ifdef CNN_HEADER
  #include "dnn_common.h"
#else /* LIBXSMM sample */
  #include "../common/dnn_common.h"
#endif

#endif

typedef enum my_normalization_fuse {
  MY_NORMALIZE_FUSE_NONE = 0,
  MY_NORMALIZE_FUSE_RELU = 1,
  MY_NORMALIZE_FUSE_ELTWISE = 2,
  MY_NORMALIZE_FUSE_ELTWISE_RELU = 3,
  MY_NORMALIZE_FUSE_RELU_WITH_MASK = 4,
  MY_NORMALIZE_FUSE_ELTWISE_RELU_WITH_MASK = 5
} my_normalization_fuse;

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

  libxsmm_barrier* barrier;

  libxsmm_matrix_eqn_function  func10;
  libxsmm_meltwfunction_unary  reduce_HW_kernel;
  libxsmm_meltwfunction_unary  all_zero_kernel;
  libxsmm_meltwfunction_binary add_kernel; // FIXME: rename since this is a helper not the true inp_add */
  libxsmm_meltwfunction_unary  copy_kernel;// FIXME: rename since this is a helper not the true inp_add */
  libxsmm_meltwfunction_unary  relu_kernel;
  libxsmm_meltwfunction_binary ewise_add_kernel;
  my_normalization_fuse        fuse_type;
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

  libxsmm_barrier* barrier;

  libxsmm_matrix_eqn_function  dgamma_func;
  libxsmm_matrix_eqn_function  dbeta_func;
  libxsmm_matrix_eqn_function  din_func;
  libxsmm_meltwfunction_unary  all_zero_kernel;
  libxsmm_meltwfunction_binary add_kernel; // FIXME: rename since this is a helper not the true inp_add */
  libxsmm_meltwfunction_unary  copy_kernel;// FIXME: rename since this is a helper not the true inp_add */
  libxsmm_meltwfunction_unary  relu_kernel;
  libxsmm_meltwfunction_binary ewise_add_kernel;
  libxsmm_meltwfunction_unary  ewise_copy_kernel;
  my_normalization_fuse        fuse_type;
} my_bn_bwd_config;

my_bn_fwd_config setup_my_bn_fwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint bc,
                                 libxsmm_blasint threads, my_normalization_fuse fuse_type ) {
  my_bn_fwd_config res;

  size_t sum_N_offset, sumsq_N_offset;

  libxsmm_meltw_unary_shape  unary_shape;
  libxsmm_meltw_binary_shape binary_shape;

  libxsmm_meltw_unary_flags   unary_flags;
  libxsmm_meltw_binary_flags  binary_flags;
  libxsmm_meltw_ternary_flags ternary_flags;

  libxsmm_blasint ldo = bc;
  libxsmm_blasint ld  = bc;
  libxsmm_blasint tmp_ld, tmp_ld2;
  libxsmm_blasint my_eqn10;

//  libxsmm_meltw_unary_flags jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
//  libxsmm_meltw_unary_type  unary_type;

  libxsmm_datatype  in_dt  = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype  out_dt = LIBXSMM_DATATYPE_F32;

  libxsmm_meqn_arg_shape  eqn_out_arg_shape;

  libxsmm_meqn_arg_shape  arg_shape[128];
  libxsmm_matrix_arg_attributes arg_set_attr0, arg_set_attr1;
  libxsmm_matrix_arg_attributes arg_singular_attr;

  libxsmm_matrix_eqn_arg_metadata arg_metadata[128];
  libxsmm_matrix_eqn_op_metadata  op_metadata[128];

  arg_singular_attr.type = LIBXSMM_MATRIX_ARG_TYPE_SINGULAR;

  /* setting up some handle values */
  res.N  = N;
  res.C  = C;
  res.H  = H;
  res.W  = W;
  res.bc = bc;
  res.CP = res.C / res.bc;
  res.num_HW_blocks = NUM_HW_BLOCKS; /* hardcoded for now */
  res.threads       = threads;
  res.fuse_type     = fuse_type;

  /* when masking is on, bc must be divisible by 8 for compressing mask into char array (otherwise strides are wrong for relumask */
  if ( (res.fuse_type == 4 || res.fuse_type == 5) && (res.bc % BITS_PER_CHAR != 0)) {
    fprintf( stderr, "bc = %d is not divisible by BITS_PER_CHAR = %d. Bailing...!\n", res.bc, BITS_PER_CHAR);
    exit(-1);
  }

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* TPP creation */

  memset( &unary_shape,  0, sizeof(libxsmm_meltw_unary_shape));
  memset( &binary_shape, 0, sizeof(libxsmm_meltw_binary_shape));

  /* Eltwise TPPs  */

//  res.all_zero_kernel = libxsmm_dispatch_meltw_unary(res.bc, 1, NULL, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  unary_shape.m   = res.bc;
  unary_shape.n   = 1;
  unary_shape.ldi = NULL;
  unary_shape.ldo = &ldo;
  unary_shape.in_type   = LIBXSMM_DATATYPE_F32;
  unary_shape.out_type  = LIBXSMM_DATATYPE_F32;
  unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.all_zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
  if ( res.all_zero_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd all_zero_kernel failed. Bailing...!\n");
    exit(-1);
  }

  //res.copy_kernel = libxsmm_dispatch_meltw_unary(res.bc, 1, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
  unary_shape.m   = res.bc;
  unary_shape.n   = 1;
  unary_shape.ldi = &ldo;
  unary_shape.ldo = &ldo;
  unary_shape.in_type   = LIBXSMM_DATATYPE_F32;
  unary_shape.out_type  = LIBXSMM_DATATYPE_F32;
  unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.copy_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, unary_flags);
  if ( res.copy_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd copy_kernel failed. Bailing...!\n");
    exit(-1);
  }

//  res.add_kernel = libxsmm_dispatch_meltw_binary(res.bc, 1, &ldo, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_MELTW_TYPE_BINARY_ADD);
  binary_shape.m         = res.bc;
  binary_shape.n         = 1;
  binary_shape.in_type   = LIBXSMM_DATATYPE_F32;
  binary_shape.comp_type = LIBXSMM_DATATYPE_F32;
  binary_shape.out_type  = LIBXSMM_DATATYPE_F32;
  binary_shape.ldi       = &ldo;
  binary_shape.ldi2      = &ldo;
  binary_shape.ldo       = &ldo;
  binary_flags           = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  res.add_kernel = libxsmm_dispatch_meltw_binary_v2(LIBXSMM_MELTW_TYPE_BINARY_ADD, binary_shape, binary_flags);
  if ( res.add_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd add_kernel failed. Bailing...!\n");
    exit(-1);
  }

//  res.relu_kernel = libxsmm_dispatch_meltw_unary(res.bc, res.H*res.W / res.num_HW_blocks, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
//                                                 /*LIBXSMM_MELTW_FLAG_UNARY_NONE*/ LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT, LIBXSMM_MELTW_TYPE_UNARY_RELU);
  unary_shape.m   = res.bc;
  unary_shape.n   = res.H*res.W / res.num_HW_blocks;
  unary_shape.ldi = &ldo;
  unary_shape.ldo = &ldo;
  unary_shape.in_type   = LIBXSMM_DATATYPE_F32;
  unary_shape.out_type  = LIBXSMM_DATATYPE_F32;
  unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT /*LIBXSMM_MELTW_FLAG_UNARY_NONE*/;
  res.relu_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_RELU, unary_shape, unary_flags);

  if ( res.relu_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP fwd_relu_kernel failed. Bailing...!\n");
    exit(-1);
  }

//  res.ewise_add_kernel = libxsmm_dispatch_meltw_binary(res.bc, res.H*res.W / res.num_HW_blocks, &ldo, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
//                                                       LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_MELTW_TYPE_BINARY_ADD);
  binary_shape.m         = res.bc;
  binary_shape.n         = res.H*res.W / res.num_HW_blocks;
  binary_shape.in_type   = LIBXSMM_DATATYPE_F32;
  binary_shape.comp_type = LIBXSMM_DATATYPE_F32;
  binary_shape.out_type  = LIBXSMM_DATATYPE_F32;
  binary_shape.ldi       = &ldo;
  binary_shape.ldi2      = &ldo;
  binary_shape.ldo       = &ldo;
  binary_flags           = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  res.ewise_add_kernel   = libxsmm_dispatch_meltw_binary_v2(LIBXSMM_MELTW_TYPE_BINARY_ADD, binary_shape, binary_flags);
  if ( res.ewise_add_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd ewise_add_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* TPPs for reducing X and X2 in HW*/
  tmp_ld = bc;

//  unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD;
//  jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
//  res.reduce_HW_kernel = libxsmm_dispatch_meltw_unary(res.bc, res.H*res.W/res.num_HW_blocks, &ld, &tmp_ld, in_dt, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);

  unary_shape.m   = res.bc;
  unary_shape.n   = res.H*res.W / res.num_HW_blocks;
  unary_shape.ldi = &ld;
  unary_shape.ldo = &tmp_ld;
  unary_shape.in_type   = LIBXSMM_DATATYPE_F32;
  unary_shape.out_type  = LIBXSMM_DATATYPE_F32;
  unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  res.reduce_HW_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD, unary_shape, unary_flags);

  if ( res.reduce_HW_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd reduce_HW_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* TPP for scaling */
  ld = bc;
  tmp_ld = 1;
  tmp_ld2 = 1;

/*-----------

LIBXSMM_API int libxsmm_matrix_eqn_push_back_arg( const libxsmm_blasint idx, const libxsmm_blasint m, const libxsmm_blasint n, const libxsmm_blasint ld,
                                                  const libxsmm_blasint in_pos, const libxsmm_blasint offs_in_pos, const libxsmm_datatype dtype );
LIBXSMM_API int libxsmm_matrix_eqn_push_back_unary_op( const libxsmm_blasint idx, const libxsmm_meltw_unary_type type, const libxsmm_meltw_unary_flags flags, const libxsmm_datatype dtype );
LIBXSMM_API int libxsmm_matrix_eqn_push_back_binary_op( const libxsmm_blasint idx, const libxsmm_meltw_binary_type type, const libxsmm_meltw_binary_flags flags, const libxsmm_datatype dtype );
LIBXSMM_API int libxsmm_matrix_eqn_push_back_ternary_op( const libxsmm_blasint idx, const libxsmm_meltw_ternary_type type, const libxsmm_meltw_ternary_flags flags, const libxsmm_datatype dtype );

LIBXSMM_API int libxsmm_matrix_eqn_push_back_arg_v2( const libxsmm_matrix_eqn_arg_metadata arg_metadata, const libxsmm_meqn_arg_shape arg_shape, libxsmm_matrix_arg_attributes arg_attr);
LIBXSMM_API int libxsmm_matrix_eqn_push_back_unary_op_v2( const libxsmm_matrix_eqn_op_metadata op_metadata, const libxsmm_meltw_unary_type type, const libxsmm_datatype dtype, const libxsmm_bitfield flags);
LIBXSMM_API int libxsmm_matrix_eqn_push_back_binary_op_v2( const libxsmm_matrix_eqn_op_metadata op_metadata, const libxsmm_meltw_binary_type type, const libxsmm_datatype dtype, const libxsmm_bitfield flags);
LIBXSMM_API int libxsmm_matrix_eqn_push_back_ternary_op_v2( const libxsmm_matrix_eqn_op_metadata op_metadata, const libxsmm_meltw_ternary_type type, const libxsmm_datatype dtype, const libxsmm_bitfield flags);


  libxsmm_meqn_arg_shape  arg_shape[128];
  libxsmm_matrix_arg_attributes arg_set_attr0, arg_set_attr1;
  libxsmm_matrix_arg_attributes arg_singular_attr;

  libxsmm_matrix_eqn_arg_metadata arg_metadata[128];
  libxsmm_matrix_eqn_op_metadata  op_metadata[128];

  arg_singular_attr.type = LIBXSMM_MATRIX_ARG_TYPE_SINGULAR;

  my_eqn0 = libxsmm_matrix_eqn_create();
  arg_metadata[0].eqn_idx     = my_eqn0;
  arg_metadata[0].in_arg_pos  = 0;
  arg_metadata[1].eqn_idx     = my_eqn0;
  arg_metadata[1].in_arg_pos  = 1;
  arg_metadata[2].eqn_idx     = my_eqn0;
  arg_metadata[2].in_arg_pos  = 2;
  arg_metadata[3].eqn_idx     = my_eqn0;
  arg_metadata[3].in_arg_pos  = 3;
  op_metadata[0].eqn_idx      = my_eqn0;
  op_metadata[0].op_arg_pos   = -1;

  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE);
  libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_UNARY_GELU, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE);
  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE);
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[0], arg_shape[0], arg_singular_attr);
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[1], arg_shape[1], arg_singular_attr);
  libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_UNARY_TANH, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE);
  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_BINARY_MATMUL, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE);
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[2], arg_shape[2], arg_singular_attr);
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[3], arg_shape[3], arg_singular_attr);
  libxsmm_matrix_eqn_tree_print( my_eqn0 );
  func0 = libxsmm_dispatch_matrix_eqn_v2( my_eqn0, arg_shape[n_tensors-1] );

    arg_shape[j].m = m_i[j];
    arg_shape[j].n = n_i[j];
    arg_shape[j].ld = &ld_i[j];
    if (j == n_tensors-1) {
      arg_shape[j].type = out_dt;
    } else {
      arg_shape[j].type = in_dt;
    }

  op_metadata[1].eqn_idx      = my_eqn1;
  op_metadata[1].op_arg_pos   = 7;
  libxsmm_matrix_eqn_push_back_ternary_op_v2(op_metadata[1], LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
-----------
*/

  my_eqn10 = libxsmm_matrix_eqn_create();                               /* y = (s*x + b)*gamma + beta */

  ternary_flags = LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT;
  op_metadata[0].eqn_idx      = my_eqn10;
  //op_metadata[0].op_arg_pos   = 7;
  libxsmm_matrix_eqn_push_back_ternary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_DATATYPE_F32, ternary_flags);

  ternary_flags = LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT;
  op_metadata[1].eqn_idx      = my_eqn10;
  //op_metadata[1].op_arg_pos   = 7;
  libxsmm_matrix_eqn_push_back_ternary_op_v2(op_metadata[1], LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_DATATYPE_F32, ternary_flags);

  arg_metadata[0].eqn_idx     = my_eqn10;
  arg_metadata[0].in_arg_pos  = 0;
  arg_shape[0].m    = res.bc;                                      /* x = [HW, bc] */
  arg_shape[0].n    = res.H*res.W /res.num_HW_blocks;
  arg_shape[0].ld   = &ld;
  arg_shape[0].type = LIBXSMM_DATATYPE_F32;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[0], arg_shape[0], arg_singular_attr);

  arg_metadata[1].eqn_idx     = my_eqn10;
  arg_metadata[1].in_arg_pos  = 1;
  arg_shape[1].m    = res.bc;                                      /* s = [bc] */
  arg_shape[1].n    = 1;
  arg_shape[1].ld   = &tmp_ld;
  arg_shape[1].type = LIBXSMM_DATATYPE_F32;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[1], arg_shape[1], arg_singular_attr);

  arg_metadata[2].eqn_idx     = my_eqn10;
  arg_metadata[2].in_arg_pos  = 2;
  arg_shape[2].m    = res.bc;                                      /* b = [bc] */
  arg_shape[2].n    = 1;
  arg_shape[2].ld   = &tmp_ld;
  arg_shape[2].type = LIBXSMM_DATATYPE_F32;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[2], arg_shape[2], arg_singular_attr);

  arg_metadata[3].eqn_idx     = my_eqn10;
  arg_metadata[3].in_arg_pos  = 3;
  arg_shape[3].m    = res.bc;                                      /* gamma = [bc] */
  arg_shape[3].n    = 1;
  arg_shape[3].ld   = &tmp_ld2;
  arg_shape[3].type = LIBXSMM_DATATYPE_F32;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[3], arg_shape[3], arg_singular_attr);

  arg_metadata[4].eqn_idx     = my_eqn10;
  arg_metadata[4].in_arg_pos  = 4;
  arg_shape[4].m    = res.bc;                                      /* beta = [bc] */
  arg_shape[4].n    = 1;
  arg_shape[4].ld   = &tmp_ld2;
  arg_shape[4].type = LIBXSMM_DATATYPE_F32;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[4], arg_shape[4], arg_singular_attr);

  eqn_out_arg_shape.m    = res.bc;                                      /* y = [HW, bc] */
  eqn_out_arg_shape.n    = res.H*res.W / res.num_HW_blocks;
  eqn_out_arg_shape.ld   = &ld;
  eqn_out_arg_shape.type = LIBXSMM_DATATYPE_F32;
  res.func10 = libxsmm_dispatch_matrix_eqn_v2( my_eqn10, eqn_out_arg_shape );

  libxsmm_matrix_eqn_tree_print( my_eqn10 );

#if 0
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn10, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn10, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, res.bc, res.H*res.W /res.num_HW_blocks, ld, 0, 0, in_dt );   /* x = [HW, bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, res.bc, 1, tmp_ld, 1, 0, LIBXSMM_DATATYPE_F32 );       /* s = [bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, res.bc, 1, tmp_ld, 2, 0, LIBXSMM_DATATYPE_F32 );       /* b = [bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, res.bc, 1, tmp_ld2, 3, 0, in_dt );                     /* gamma = [bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, res.bc, 1, tmp_ld2, 4, 0, in_dt );                     /* beta = [bc] */
  /* libxsmm_matrix_eqn_tree_print( my_eqn10 ); */
  /* libxsmm_matrix_eqn_rpn_print( my_eqn10 ); */
  res.func10 = libxsmm_dispatch_matrix_eqn( res.bc, res.H*res.W / res.num_HW_blocks, &ld, out_dt, my_eqn10 );   /* y = [HW, bc] */
#endif

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
                                 libxsmm_blasint threads, my_normalization_fuse fuse_type ) {
  my_bn_bwd_config res;

  libxsmm_meltw_unary_shape  unary_shape;
  libxsmm_meltw_binary_shape binary_shape;

  libxsmm_meltw_unary_flags   unary_flags;
  libxsmm_meltw_binary_flags  binary_flags;
  libxsmm_meltw_ternary_flags ternary_flags;

  libxsmm_meqn_arg_shape  eqn_out_arg_shape;

  libxsmm_meqn_arg_shape  arg_shape[128];
  libxsmm_matrix_arg_attributes arg_set_attr0, arg_set_attr1;
  libxsmm_matrix_arg_attributes arg_singular_attr;

  libxsmm_matrix_eqn_arg_metadata arg_metadata[128];
  libxsmm_matrix_eqn_op_metadata  op_metadata[128];

  arg_singular_attr.type = LIBXSMM_MATRIX_ARG_TYPE_SINGULAR;

  size_t dbeta_N_offset;

  libxsmm_blasint ldo = bc;
  libxsmm_blasint ld  = bc;
  libxsmm_blasint tmp_ld2;
  libxsmm_blasint my_eqn11, my_eqn12, my_eqn16;

  libxsmm_datatype  in_dt  = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype  out_dt = LIBXSMM_DATATYPE_F32;

  /* setting up some handle values */
  res.N  = N;
  res.C  = C;
  res.H  = H;
  res.W  = W;
  res.bc = bc;
  res.CP = res.C / res.bc;
  res.num_HW_blocks = NUM_HW_BLOCKS; /* hardcoded for now */
  res.threads       = threads;
  res.fuse_type     = fuse_type;

  /* when masking is on, bc must be divisible by 8 for compressing mask into char array (otherwise strides are wrong for relumask */
  if ( (res.fuse_type == 4 || res.fuse_type == 5) && (res.bc % BITS_PER_CHAR != 0)) {
    fprintf( stderr, "bc = %d is not divisible by BITS_PER_CHAR = %d. Bailing...!\n", res.bc, BITS_PER_CHAR);
    exit(-1);
  }

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* TPP creation */

  /* Eltwise TPPs  */
//  res.all_zero_kernel = libxsmm_dispatch_meltw_unary(res.bc, 1, NULL, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  unary_shape.m   = res.bc;
  unary_shape.n   = 1;
  unary_shape.ldi = NULL;
  unary_shape.ldo = &ldo;
  unary_shape.in_type   = LIBXSMM_DATATYPE_F32;
  unary_shape.out_type  = LIBXSMM_DATATYPE_F32;
  unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.all_zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
  if ( res.all_zero_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd all_zero_kernel failed. Bailing...!\n");
    exit(-1);
  }

  //res.copy_kernel = libxsmm_dispatch_meltw_unary(res.bc, 1, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
  unary_shape.m   = res.bc;
  unary_shape.n   = 1;
  unary_shape.ldi = &ldo;
  unary_shape.ldo = &ldo;
  unary_shape.in_type   = LIBXSMM_DATATYPE_F32;
  unary_shape.out_type  = LIBXSMM_DATATYPE_F32;
  unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.copy_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, unary_flags);
  if ( res.copy_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd copy_kernel failed. Bailing...!\n");
    exit(-1);
  }

//  res.add_kernel = libxsmm_dispatch_meltw_binary(res.bc, 1, &ldo, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_MELTW_TYPE_BINARY_ADD);
  binary_shape.m         = res.bc;
  binary_shape.n         = 1;
  binary_shape.in_type   = LIBXSMM_DATATYPE_F32;
  binary_shape.comp_type = LIBXSMM_DATATYPE_F32;
  binary_shape.out_type  = LIBXSMM_DATATYPE_F32;
  binary_shape.ldi       = &ldo;
  binary_shape.ldi2      = &ldo;
  binary_shape.ldo       = &ldo;
  binary_flags           = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  res.add_kernel = libxsmm_dispatch_meltw_binary_v2(LIBXSMM_MELTW_TYPE_BINARY_ADD, binary_shape, binary_flags);
  if ( res.add_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd add_kernel failed. Bailing...!\n");
    exit(-1);
  }

//  res.relu_kernel = libxsmm_dispatch_meltw_unary(res.bc, res.H*res.W / res.num_HW_blocks, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
//                                                 /*LIBXSMM_MELTW_FLAG_UNARY_NONE unsupported */ LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT, LIBXSMM_MELTW_TYPE_UNARY_RELU_INV);
  unary_shape.m   = res.bc;
  unary_shape.n   = res.H*res.W / res.num_HW_blocks;
  unary_shape.ldi = &ldo;
  unary_shape.ldo = &ldo;
  unary_shape.in_type   = LIBXSMM_DATATYPE_F32;
  unary_shape.out_type  = LIBXSMM_DATATYPE_F32;
  unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT /*LIBXSMM_MELTW_FLAG_UNARY_NONE*/;
  res.relu_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_RELU_INV, unary_shape, unary_flags);
  if ( res.relu_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP bwd relu_kernel failed. Bailing...!\n");
    exit(-1);
  }

//  res.ewise_add_kernel = libxsmm_dispatch_meltw_binary(res.bc, res.H*res.W / res.num_HW_blocks, &ldo, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
//                                                       LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_MELTW_TYPE_BINARY_ADD);
  binary_shape.m         = res.bc;
  binary_shape.n         = res.H*res.W / res.num_HW_blocks;
  binary_shape.in_type   = LIBXSMM_DATATYPE_F32;
  binary_shape.comp_type = LIBXSMM_DATATYPE_F32;
  binary_shape.out_type  = LIBXSMM_DATATYPE_F32;
  binary_shape.ldi       = &ldo;
  binary_shape.ldi2      = &ldo;
  binary_shape.ldo       = &ldo;
  binary_flags           = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  res.ewise_add_kernel   = libxsmm_dispatch_meltw_binary_v2(LIBXSMM_MELTW_TYPE_BINARY_ADD, binary_shape, binary_flags);

  if ( res.ewise_add_kernel == NULL) {
    fprintf( stderr, "JIT for TPP bwd ewise_add_kernel failed. Bailing...!\n");
    exit(-1);
  }

//  res.ewise_copy_kernel = libxsmm_dispatch_meltw_unary(/*1, 1 */res.bc, res.H*res.W / res.num_HW_blocks, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
//                                                       LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
  unary_shape.m   = res.bc;
  unary_shape.n   = res.H*res.W / res.num_HW_blocks;
  unary_shape.ldi = &ldo;
  unary_shape.ldo = &ldo;
  unary_shape.in_type   = LIBXSMM_DATATYPE_F32;
  unary_shape.out_type  = LIBXSMM_DATATYPE_F32;
  unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.ewise_copy_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, unary_flags);
  if ( res.ewise_copy_kernel == NULL) {
    fprintf( stderr, "JIT for TPP bwd ewise_copy_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* TPP equations for dgamma, dbeta and din */

  ld = bc;
  tmp_ld2 = 1;

  /* dgamma function  */
  my_eqn11 = libxsmm_matrix_eqn_create();                                                       /* dgamma = ((inp *a + b) * dout) + dgamma */

//#if 0
//----------------
  binary_flags                = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  op_metadata[0].eqn_idx      = my_eqn11;
  op_metadata[0].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_DATATYPE_F32, binary_flags);
  //libxsmm_matrix_eqn_push_back_binary_op(my_eqn11, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32);                   /* dgamma = ((inp *a + b) * dout) + dgamma */

  unary_flags                 = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  op_metadata[1].eqn_idx      = my_eqn11;
  op_metadata[1].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[1], LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_DATATYPE_F32, unary_flags);
  //libxsmm_matrix_eqn_push_back_unary_op(my_eqn11, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);   /* [HW, bc] -> [bc] */

  binary_flags                = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  op_metadata[2].eqn_idx      = my_eqn11;
  op_metadata[2].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[2], LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_DATATYPE_F32, binary_flags);
  //libxsmm_matrix_eqn_push_back_binary_op(my_eqn11, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32);                   /* ((inp *a + b) * dout) */

  ternary_flags               = LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT;
  op_metadata[3].eqn_idx      = my_eqn11;
  op_metadata[3].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_ternary_op_v2(op_metadata[3], LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_DATATYPE_F32, ternary_flags);
  //libxsmm_matrix_eqn_push_back_ternary_op( my_eqn11, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );

  arg_metadata[0].eqn_idx     = my_eqn11;
  arg_metadata[0].in_arg_pos  = 0;
  arg_shape[0].m    = res.bc;                                      /* inp [HW, bc] */
  arg_shape[0].n    = res.H*res.W /res.num_HW_blocks;
  arg_shape[0].ld   = &ld;
  arg_shape[0].type = LIBXSMM_DATATYPE_F32;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[0], arg_shape[0], arg_singular_attr);

  arg_metadata[1].eqn_idx     = my_eqn11;
  arg_metadata[1].in_arg_pos  = 1;
  arg_shape[1].m    = res.bc;                                      /* a [bc] */
  arg_shape[1].n    = 1;
  arg_shape[1].ld   = &tmp_ld2;
  arg_shape[1].type = LIBXSMM_DATATYPE_F32;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[1], arg_shape[1], arg_singular_attr);

  arg_metadata[2].eqn_idx     = my_eqn11;
  arg_metadata[2].in_arg_pos  = 2;
  arg_shape[2].m    = res.bc;                                      /* b [bc] */
  arg_shape[2].n    = 1;
  arg_shape[2].ld   = &tmp_ld2;
  arg_shape[2].type = LIBXSMM_DATATYPE_F32;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[2], arg_shape[2], arg_singular_attr);

  arg_metadata[3].eqn_idx     = my_eqn11;
  arg_metadata[3].in_arg_pos  = 3;
  arg_shape[3].m    = res.bc;                                      /* dout [HW, bc] */
  arg_shape[3].n    = res.H*res.W/res.num_HW_blocks;
  arg_shape[3].ld   = &ld;
  arg_shape[3].type = LIBXSMM_DATATYPE_F32;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[3], arg_shape[3], arg_singular_attr);

  arg_metadata[4].eqn_idx     = my_eqn11;
  arg_metadata[4].in_arg_pos  = 4;
  arg_shape[4].m    = res.bc;                                      /* dgamma [bc] */
  arg_shape[4].n    = 1;
  arg_shape[4].ld   = &tmp_ld2;
  arg_shape[4].type = LIBXSMM_DATATYPE_F32;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[4], arg_shape[4], arg_singular_attr);

  eqn_out_arg_shape.m    = res.bc;                                 /* dgamma [bc] */
  eqn_out_arg_shape.n    = 1;
  eqn_out_arg_shape.ld   = &tmp_ld2;
  eqn_out_arg_shape.type = LIBXSMM_DATATYPE_F32;
  libxsmm_matrix_eqn_tree_print( my_eqn11 );
  libxsmm_matrix_eqn_rpn_print( my_eqn11 );

  libxsmm_matrix_eqn_tree_print( my_eqn11 );
  libxsmm_matrix_eqn_rpn_print( my_eqn11 );

  res.dgamma_func = libxsmm_dispatch_matrix_eqn_v2( my_eqn11, eqn_out_arg_shape );


  //exit(-1);
//----------------
//#endif
#if 0
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn11, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32);                   /* dgamma = ((inp *a + b) * dout) + dgamma */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn11, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);   /* [HW, bc] -> [bc] */
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn11, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32);                   /* ((inp *a + b) * dout) */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn11, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, res.bc, res.H*res.W/res.num_HW_blocks, ld, 0, 0, in_dt );          /* inp [HW, bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, res.bc, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 );           /* a [bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, res.bc, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );           /* b [bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, res.bc, res.H*res.W/res.num_HW_blocks, ld, 3, 0, in_dt );          /* dout [HW, bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, res.bc, 1, 1, 4, 0, LIBXSMM_DATATYPE_F32 );           /* dgamma [bc] */
  libxsmm_matrix_eqn_tree_print( my_eqn11 );
  libxsmm_matrix_eqn_rpn_print( my_eqn11 );

//  exit(-1);

  res.dgamma_func = libxsmm_dispatch_matrix_eqn( res.bc, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn11 );      /* dgamma [bc] */
#endif

  if ( res.dgamma_func == NULL) {
    fprintf( stderr, "JIT for TPP bwd dgamma_func (eqn11) failed. Bailing...!\n");
    exit(-1);
  }

  /* dbeta function  */
  my_eqn12 = libxsmm_matrix_eqn_create();                                                       /* dbeta [bc] = dout [HW, bc] + dbeta [bc] */

//#if 0
  binary_flags                = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  op_metadata[0].eqn_idx      = my_eqn12;
  //op_metadata[0].op_arg_pos   = 7;
  libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_DATATYPE_F32, binary_flags);
  //libxsmm_matrix_eqn_push_back_binary_op( my_eqn12, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );                /* dbeta_tmp [HW, bc] */

  unary_flags                 = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  op_metadata[1].eqn_idx      = my_eqn12;
  //op_metadata[0].op_arg_pos   = 7;
  libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata[1], LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_DATATYPE_F32, unary_flags);
  //libxsmm_matrix_eqn_push_back_unary_op(my_eqn12, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);  /* [HW, bc] -> [bc] */

  arg_metadata[0].eqn_idx     = my_eqn12;
  arg_metadata[0].in_arg_pos  = 3;
  arg_shape[0].m    = res.bc;                                      /* dout [HW, bc] */
  arg_shape[0].n    = res.H*res.W /res.num_HW_blocks;
  arg_shape[0].ld   = &ld;
  arg_shape[0].type = LIBXSMM_DATATYPE_F32;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[0], arg_shape[0], arg_singular_attr);

  arg_metadata[1].eqn_idx     = my_eqn12;
  arg_metadata[1].in_arg_pos  = 5;
  arg_shape[1].m    = res.bc;                                      /* dbeta [bc] */
  arg_shape[1].n    = 1;
  arg_shape[1].ld   = &tmp_ld2;
  arg_shape[1].type = LIBXSMM_DATATYPE_F32;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[1], arg_shape[1], arg_singular_attr);

  eqn_out_arg_shape.m    = res.bc;                                 /* dbeta [bc] */
  eqn_out_arg_shape.n    = 1;
  eqn_out_arg_shape.ld   = &tmp_ld2;
  eqn_out_arg_shape.type = LIBXSMM_DATATYPE_F32;

  libxsmm_matrix_eqn_tree_print( my_eqn12 );
  libxsmm_matrix_eqn_rpn_print( my_eqn12 );

  res.dbeta_func = libxsmm_dispatch_matrix_eqn_v2( my_eqn12, eqn_out_arg_shape );

//#endif

#if 0
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn12, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );                /* dbeta_tmp [HW, bc] */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn12, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);  /* [HW, bc] -> [bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn12, res.bc, res.H*res.W/res.num_HW_blocks, ld, 3, 0, in_dt );          /* dout [HW, bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn12, res.bc, 1, 1, 5, 0, LIBXSMM_DATATYPE_F32 );           /* dbeta [bc] */
  libxsmm_matrix_eqn_tree_print( my_eqn12 );
  /* libxsmm_matrix_eqn_rpn_print( my_eqn12 ); */
  res.dbeta_func = libxsmm_dispatch_matrix_eqn( res.bc, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn12 );      /* dbeta [bc] */
#endif

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
  my_eqn16 = libxsmm_matrix_eqn_create();                                                       /* din = a * dout + (b * inp + c) */

//#if 0
  ternary_flags               = LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT;
  op_metadata[0].eqn_idx      = my_eqn16;
  //op_metadata[1].op_arg_pos   = 7;
  libxsmm_matrix_eqn_push_back_ternary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_DATATYPE_F32, ternary_flags);
  //libxsmm_matrix_eqn_push_back_ternary_op( my_eqn16, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );

  arg_metadata[0].eqn_idx     = my_eqn16;
  arg_metadata[0].in_arg_pos  = 1;
  arg_shape[0].m    = res.bc;                                      /* a [bc] */
  arg_shape[0].n    = 1;
  arg_shape[0].ld   = &tmp_ld2;
  arg_shape[0].type = LIBXSMM_DATATYPE_F32;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[0], arg_shape[0], arg_singular_attr);

  arg_metadata[1].eqn_idx     = my_eqn16;
  arg_metadata[1].in_arg_pos  = 3;
  arg_shape[1].m    = res.bc;                                      /* dout [HW, bc] */
  arg_shape[1].n    = res.H*res.W /res.num_HW_blocks;
  arg_shape[1].ld   = &ld;
  arg_shape[1].type = LIBXSMM_DATATYPE_F32;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[1], arg_shape[1], arg_singular_attr);

  ternary_flags               = LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT;
  op_metadata[1].eqn_idx      = my_eqn16;
  //op_metadata[1].op_arg_pos   = 7;
  libxsmm_matrix_eqn_push_back_ternary_op_v2(op_metadata[1], LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_DATATYPE_F32, ternary_flags);
  //libxsmm_matrix_eqn_push_back_ternary_op( my_eqn16, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );

  arg_metadata[2].eqn_idx     = my_eqn16;
  arg_metadata[2].in_arg_pos  = 0;
  arg_shape[2].m    = res.bc;                                      /* inp [HW, bc] */
  arg_shape[2].n    = res.H*res.W /res.num_HW_blocks;
  arg_shape[2].ld   = &ld;
  arg_shape[2].type = LIBXSMM_DATATYPE_F32;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[2], arg_shape[2], arg_singular_attr);

  arg_metadata[3].eqn_idx     = my_eqn16;
  arg_metadata[3].in_arg_pos  = 2;
  arg_shape[3].m    = res.bc;                                      /* b [bc] */
  arg_shape[3].n    = 1;
  arg_shape[3].ld   = &tmp_ld2;
  arg_shape[3].type = LIBXSMM_DATATYPE_F32;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[3], arg_shape[3], arg_singular_attr);

  arg_metadata[4].eqn_idx     = my_eqn16;
  arg_metadata[4].in_arg_pos  = 7;
  arg_shape[4].m    = res.bc;                                      /* c [bc] */
  arg_shape[4].n    = 1;
  arg_shape[4].ld   = &tmp_ld2;
  arg_shape[4].type = LIBXSMM_DATATYPE_F32;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[4], arg_shape[4], arg_singular_attr);

  eqn_out_arg_shape.m    = res.bc;                                 /* din [HW, bc] */
  eqn_out_arg_shape.n    = res.H*res.W/res.num_HW_blocks;
  eqn_out_arg_shape.ld   = &ld;
  eqn_out_arg_shape.type = LIBXSMM_DATATYPE_F32;

  libxsmm_matrix_eqn_tree_print( my_eqn16 );
  libxsmm_matrix_eqn_rpn_print( my_eqn16 );
  res.din_func = libxsmm_dispatch_matrix_eqn_v2( my_eqn16, eqn_out_arg_shape );

//  exit(-1);
//#endif

#if 0
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn16, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, res.bc, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 );           /* a [bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, res.bc, res.H*res.W/res.num_HW_blocks, ld, 3, 0, in_dt );          /* dout [HW, bc] */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn16, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, res.bc, res.H*res.W/res.num_HW_blocks, ld, 0, 0, in_dt );          /* inp [HW, bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, res.bc, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );           /* b [bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, res.bc, 1, 1, 7, 0, LIBXSMM_DATATYPE_F32 );           /* c [bc] */
  libxsmm_matrix_eqn_tree_print( my_eqn16 );
  libxsmm_matrix_eqn_rpn_print( my_eqn16 );
  res.din_func = libxsmm_dispatch_matrix_eqn( res.bc, res.H*res.W/res.num_HW_blocks, &ld, in_dt, my_eqn16 );           /* din [HW, bc] */
  exit(-1);
#endif

  if ( res.din_func == NULL) {
    fprintf( stderr, "JIT for TPP bwd din_func (eqn16) failed. Bailing...!\n");
    exit(-1);
  }

  /* init scratch */
  dbeta_N_offset = LIBXSMM_UP2(res.CP * res.N * res.bc, 64);
  res.scratch_size =  sizeof(float) * ( dbeta_N_offset /* dbeta_N*/ + LIBXSMM_UP2(res.CP * res.N * res.bc, 64) /*dgamma_N */ );

  return res;
}

