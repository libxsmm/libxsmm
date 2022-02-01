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
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

//#define NUM_HW_BLOCKS (16)

#define REFACTORED_FWD
#define REFACTORED_BWD

#define COMPUTE_FP64_REFERENCE

/* include c-based dnn library */
#include "../common/dnn_common.h"

#define TRUE_PARALLEL_BWD
#define TRUE_PARALLEL_FWD

#if defined(REFACTORED_BWD) || defined(REFACTORED_FWD)

#define BITS_PER_CHAR (8)


// FIXME: Either make bn and gn differ, or make macro guards to have only one definition in the final batch/groupnorm headers
typedef enum my_normalization_fuse {
  MY_NORMALIZE_FUSE_NONE = 0,
  MY_NORMALIZE_FUSE_RELU = 1,
  MY_NORMALIZE_FUSE_ELTWISE = 2,
  MY_NORMALIZE_FUSE_ELTWISE_RELU = 3,
  MY_NORMALIZE_FUSE_RELU_WITH_MASK = 4,
  MY_NORMALIZE_FUSE_ELTWISE_RELU_WITH_MASK = 5
} my_normalization_fuse;



#endif /* either REFACTORED_FWD or REFACTORED_BWD */

#if !defined(REFACTORED_BWD) || !defined(REFACTORED_FWD)

void tpp_groupnorm_fwd_fp32(long NP, long CP, long HW, long CB, long G, long num_HW_blocks, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout,
                            libxsmm_matrix_eqn_function func10, libxsmm_meltwfunction_unary reduce_HW_kernel, libxsmm_meltwfunction_unary reduce_rows_kernel,
                            libxsmm_meltwfunction_unary reduce_groups_kernel, libxsmm_meltwfunction_unary all_zero_G_kernel, libxsmm_meltwfunction_unary all_zero_kernel, libxsmm_meltwfunction_binary add_kernel, float eps);

void tpp_groupnorm_bwd_fp32(long NP, long CP, long HW, long CB, long G, long num_HW_blocks, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta,
    libxsmm_matrix_eqn_function dgamma_func, libxsmm_matrix_eqn_function dbeta_func, libxsmm_matrix_eqn_function db_func, libxsmm_matrix_eqn_function ds_func, libxsmm_matrix_eqn_function din_func,
    libxsmm_meltwfunction_unary all_zero_kernel, libxsmm_meltwfunction_binary add_kernel, float eps);


void scaler_groupnorm_bwd_fp32(long NP, long CP, long HW, long CB, long G, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta, float eps);
void scaler_groupnorm_fwd_fp32(long NP, long CP, long HW, long CB, long G, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout, float eps);

#endif

#ifdef REFACTORED_FWD

typedef struct my_gn_fwd_config {
  libxsmm_blasint  N;
  libxsmm_blasint  C;
  libxsmm_blasint  G;
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
  libxsmm_meltwfunction_unary  reduce_rows_kernel;
  libxsmm_meltwfunction_unary  reduce_groups_kernel;
  libxsmm_meltwfunction_unary  all_zero_G_kernel;
  libxsmm_meltwfunction_unary  all_zero_kernel;
  libxsmm_meltwfunction_binary add_kernel;
  my_normalization_fuse        fuse_type;
} my_gn_fwd_config;

#endif

#ifdef REFACTORED_BWD

typedef struct my_gn_bwd_config {
  libxsmm_blasint  N;
  libxsmm_blasint  C;
  libxsmm_blasint  G;
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
  libxsmm_matrix_eqn_function  db_func;
  libxsmm_matrix_eqn_function  ds_func;
  libxsmm_matrix_eqn_function  din_func;
  libxsmm_meltwfunction_unary  all_zero_kernel;
  libxsmm_meltwfunction_binary add_kernel;
  my_normalization_fuse        fuse_type;
} my_gn_bwd_config;
#endif

#ifdef REFACTORED_FWD

my_gn_fwd_config setup_my_gn_fwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint G, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint bc,
                                 libxsmm_blasint threads, my_normalization_fuse fuse_type ) {

  my_gn_fwd_config res;

  size_t sum_N_offset, sumsq_N_offset;

  libxsmm_blasint ldo = bc;
  libxsmm_blasint ld  = bc;
  libxsmm_blasint tmp_ld, tmp_ld2;
  libxsmm_blasint my_eqn10;

  libxsmm_meltw_unary_shape  unary_shape;
  libxsmm_meltw_binary_shape binary_shape;

  libxsmm_meltw_unary_flags   unary_flags;
  libxsmm_meltw_binary_flags  binary_flags;
  libxsmm_meltw_ternary_flags ternary_flags;

//  libxsmm_meltw_unary_flags jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
//  libxsmm_meltw_unary_type  unary_type;

//  libxsmm_datatype  in_dt  = LIBXSMM_DATATYPE_F32;
//  libxsmm_datatype  out_dt = LIBXSMM_DATATYPE_F32;

  libxsmm_datatype dtype = LIBXSMM_DATATYPE_F32;

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

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* TPP creation */
  ldo = res.G;
  unary_shape.m           = res.G;
  unary_shape.n           = 1;
  unary_shape.ldi         = NULL;
  unary_shape.ldo         = &ldo;
  unary_shape.in_type     = dtype;
  unary_shape.out_type    = dtype;
  unary_shape.comp_type   = dtype;
  unary_flags             = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.all_zero_G_kernel   = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
  //res.all_zero_G_kernel = libxsmm_dispatch_meltw_unary(res.G, 1, NULL, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  if ( res.all_zero_G_kernel == NULL) {
    fprintf( stderr, "JIT for initialization by unary all zero group copy kernel failed for fwd. Bailing...!\n");
    exit(-1);
  }

  ldo = res.bc;
  unary_shape.m         = res.bc;
  unary_shape.n         = 1;
  unary_shape.ldi       = NULL;
  unary_shape.ldo       = &ldo;
  unary_shape.in_type   = dtype;
  unary_shape.out_type  = dtype;
  unary_shape.comp_type = dtype;
  unary_flags           = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.all_zero_kernel   = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
  //res.all_zero_kernel = libxsmm_dispatch_meltw_unary(res.bc, 1, NULL, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  if ( res.all_zero_G_kernel == NULL) {
    fprintf( stderr, "JIT for initialization by unary all zero copy kernel failed for fwd. Bailing...!\n");
    exit(-1);
  }

  /* TPPs for reducing X and X2 in HW*/
  ld = res.bc;
  tmp_ld = res.bc;

  unary_shape.m         = res.bc;
  unary_shape.n         = res.H*res.W/res.num_HW_blocks;
  unary_shape.ldi       = &ld;
  unary_shape.ldo       = &tmp_ld;
  unary_shape.in_type   = dtype;
  unary_shape.out_type  = dtype;
  unary_shape.comp_type = dtype;
  unary_flags           = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  res.reduce_HW_kernel  = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD, unary_shape, unary_flags);
  //unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD;
  //jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  //res.reduce_HW_kernel = libxsmm_dispatch_meltw_unary(res.bc, res.H*res.W/res.num_HW_blocks, &ld, &tmp_ld, in_dt, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);
  if ( res.reduce_HW_kernel == NULL) {
      fprintf( stderr, "JIT for initialization of reduce_HW_kernel failed for fwd. Bailing...!\n");
      exit(-1);
  }

  binary_shape.m         = res.bc;
  binary_shape.n         = 1;
  binary_shape.in_type   = dtype;
  binary_shape.comp_type = dtype;
  binary_shape.out_type  = dtype;
  binary_shape.ldi       = &ld;
  binary_shape.ldi2      = &ld;
  binary_shape.ldo       = &ld;
  binary_flags           = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  res.add_kernel         = libxsmm_dispatch_meltw_binary_v2(LIBXSMM_MELTW_TYPE_BINARY_ADD, binary_shape, binary_flags);
  //res.add_kernel = libxsmm_dispatch_meltw_binary(res.bc, 1, &ld, &ld, &ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_MELTW_TYPE_BINARY_ADD);
  if ( res.add_kernel == NULL) {
      fprintf( stderr, "JIT for initialization of add_kernel failed for fwd. Bailing...!\n");
      exit(-1);
  }

  /* TPP for reducing groups */
  libxsmm_blasint group_size = res.C/res.G;//(res.CP*res.bc)/res.G;

  ld = group_size;                /* group_size = (CP*bc)/G */
  tmp_ld = 1;

  unary_shape.m            = group_size;
  unary_shape.n            = 1;
  unary_shape.ldi          = &ld;
  unary_shape.ldo          = &tmp_ld;
  unary_shape.in_type      = dtype;
  unary_shape.out_type     = dtype;
  unary_shape.comp_type    = dtype;
  unary_flags              = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS;
  res.reduce_groups_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, unary_shape, unary_flags);
//  unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD;
//  jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS;
//  res.reduce_groups_kernel = libxsmm_dispatch_meltw_unary(group_size, 1, &ld, &tmp_ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);
  if ( res.reduce_groups_kernel == NULL) {
      fprintf( stderr, "JIT for initialization of reduce_groups_kernel failed for fwd. Bailing...!\n");
      exit(-1);
  }

  ld = res.bc;
  tmp_ld = 1;
  unary_shape.m          = res.bc;
  unary_shape.n          = 1;
  unary_shape.ldi        = &ld;
  unary_shape.ldo        = &tmp_ld;
  unary_shape.in_type    = dtype;
  unary_shape.out_type   = dtype;
  unary_shape.comp_type  = dtype;
  unary_flags            = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS;
  res.reduce_rows_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, unary_shape, unary_flags);
  //unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD;
  //jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS;
  //res.reduce_rows_kernel = libxsmm_dispatch_meltw_unary(res.bc, 1, &ld, &tmp_ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);
  if ( res.reduce_rows_kernel == NULL) {
      fprintf( stderr, "JIT for initialization of reduce_rows_kernel failed for fwd. Bailing...!\n");
      exit(-1);
  }

  /* TPP for foward */
  ld = res.bc;
  tmp_ld = 1;
  tmp_ld2 = 1;
  my_eqn10 = libxsmm_matrix_eqn_create();                                                        /* y = (s*x + b)*gamma + beta */

  ternary_flags               = (libxsmm_meltw_ternary_flags)(LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
  op_metadata[0].eqn_idx      = my_eqn10;
  op_metadata[0].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_ternary_op_v2(op_metadata[0], LIBXSMM_MELTW_TYPE_TERNARY_MULADD, dtype, ternary_flags);
  //libxsmm_matrix_eqn_push_back_ternary_op( my_eqn10, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, (libxsmm_meltw_ternary_flags)(LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT), LIBXSMM_DATATYPE_F32);

  ternary_flags               = (libxsmm_meltw_ternary_flags)(LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT);
  op_metadata[1].eqn_idx      = my_eqn10;
  op_metadata[1].op_arg_pos   = -1;
  libxsmm_matrix_eqn_push_back_ternary_op_v2(op_metadata[1], LIBXSMM_MELTW_TYPE_TERNARY_MULADD, dtype, ternary_flags);
  //libxsmm_matrix_eqn_push_back_ternary_op( my_eqn10, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, (libxsmm_meltw_ternary_flags)(LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT), LIBXSMM_DATATYPE_F32);

  arg_metadata[0].eqn_idx     = my_eqn10;
  arg_metadata[0].in_arg_pos  = 0;
  arg_shape[0].m              = res.bc;                                      /* x = [HW, bc] */
  arg_shape[0].n              = res.H*res.W /res.num_HW_blocks;
  arg_shape[0].ld             = &ld;
  arg_shape[0].type           = dtype;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[0], arg_shape[0], arg_singular_attr);
  //libxsmm_matrix_eqn_push_back_arg( my_eqn10, res.bc, res.H*res.W/res.num_HW_blocks, ld, 0, 0, in_dt );                         /* x = [HW, bc] */

  arg_metadata[1].eqn_idx     = my_eqn10;
  arg_metadata[1].in_arg_pos  = 1;
  arg_shape[1].m    = res.bc;                                      /* s = [bc] */
  arg_shape[1].n    = 1;
  arg_shape[1].ld   = &tmp_ld;
  arg_shape[1].type = dtype;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[1], arg_shape[1], arg_singular_attr);
  //libxsmm_matrix_eqn_push_back_arg( my_eqn10, res.bc, 1, tmp_ld, 1, 0, LIBXSMM_DATATYPE_F32 );       /* s = [bc] */

  arg_metadata[2].eqn_idx     = my_eqn10;
  arg_metadata[2].in_arg_pos  = 2;
  arg_shape[2].m    = res.bc;                                      /* b = [bc] */
  arg_shape[2].n    = 1;
  arg_shape[2].ld   = &tmp_ld;
  arg_shape[2].type = dtype;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[2], arg_shape[2], arg_singular_attr);
  //libxsmm_matrix_eqn_push_back_arg( my_eqn10, res.bc, 1, tmp_ld, 2, 0, LIBXSMM_DATATYPE_F32 );       /* b = [bc] */

  arg_metadata[3].eqn_idx     = my_eqn10;
  arg_metadata[3].in_arg_pos  = 3;
  arg_shape[3].m    = res.bc;                                      /* gamma = [bc] */
  arg_shape[3].n    = 1;
  arg_shape[3].ld   = &tmp_ld2;
  arg_shape[3].type = dtype;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[3], arg_shape[3], arg_singular_attr);
  //libxsmm_matrix_eqn_push_back_arg( my_eqn10, res.bc, 1, tmp_ld2, 3, 0, in_dt );                     /* gamma = [bc] */

  arg_metadata[4].eqn_idx     = my_eqn10;
  arg_metadata[4].in_arg_pos  = 4;
  arg_shape[4].m    = res.bc;                                      /* beta = [bc] */
  arg_shape[4].n    = 1;
  arg_shape[4].ld   = &tmp_ld2;
  arg_shape[4].type = dtype;
  libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[4], arg_shape[4], arg_singular_attr);
  //libxsmm_matrix_eqn_push_back_arg( my_eqn10, res.bc, 1, tmp_ld2, 4, 0, in_dt );                     /* beta = [bc] */

  eqn_out_arg_shape.m    = res.bc;                                 /* y = [HW, bc] */
  eqn_out_arg_shape.n    = res.H*res.W / res.num_HW_blocks;
  eqn_out_arg_shape.ld   = &ld;
  eqn_out_arg_shape.type = dtype;

  /* libxsmm_matrix_eqn_tree_print( my_eqn10 ); */
  /* libxsmm_matrix_eqn_rpn_print ( my_eqn10 ); */
  res.func10 = libxsmm_dispatch_matrix_eqn_v2( my_eqn10, eqn_out_arg_shape );
  //res.func10 = libxsmm_dispatch_matrix_eqn( res.bc, res.H*res.W/res.num_HW_blocks, &ld, out_dt, my_eqn10 );                         /* y = [HW, bc] */
  if ( res.func10 == NULL) {
    fprintf( stderr, "JIT for TPP fwd func10 (eqn10) failed. Bailing...!\n");
    exit(-1);
  }

  // FIXME: Need to modify as this code is for the batchnorm
  /* init scratch */
  sum_N_offset   = LIBXSMM_UP2(res.CP * 2 * res.bc, 64);
  sumsq_N_offset = LIBXSMM_UP2(sum_N_offset + res.CP * res.N * res.bc, 64);
  res.scratch_size =  sizeof(float) * ( sumsq_N_offset /*sum_X_X2 + sumsq_N */ + LIBXSMM_UP2((size_t)res.CP * (size_t)res.N * (size_t)res.bc, 64) /* sumsq_N */ );

  return res;
}

// FIXME: Set const modifiers properly? Cannot put const as then reduce_HW_params.in.primary    = &LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB); complains
void my_gn_fwd_exec( my_gn_fwd_config cfg, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout, float eps, int start_tid, int my_tid, void *scratch ) {

  // FIXME: N vs NP?
  const libxsmm_blasint NP = cfg.N;
  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint G  = cfg.G;
  const libxsmm_blasint HW = cfg.H * cfg.W;
  const libxsmm_blasint CB = cfg.bc;
  const libxsmm_blasint num_HW_blocks = cfg.num_HW_blocks;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;
#ifdef TRUE_PARALLEL_FWD
  /* number of tasks that could be run in parallel for 1d blocking */
  // Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here?
  const libxsmm_blasint work_dN = CP * N;
  /* compute chunk size */
  const libxsmm_blasint chunksize_dN = (work_dN % cfg.threads == 0) ?
    (work_dN / cfg.threads) : ((work_dN / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_dN = (ltid * chunksize_dN < work_dN) ? (ltid * chunksize_dN) : work_dN;
  const libxsmm_blasint thr_end_dN = ((ltid + 1) * chunksize_dN < work_dN) ? ((ltid + 1) * chunksize_dN) : work_dN;

  /* number of tasks that could be run in parallel for 1d blocking over NP*/
  // Question: each thread should take a number of full (of length NP chunks) or can we really do a partial split here?
  const libxsmm_blasint work_N = NP;
  /* compute chunk size */
  const libxsmm_blasint chunksize_N = (work_N % cfg.threads == 0) ?
    (work_N / cfg.threads) : ((work_N / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_N = (ltid * chunksize_N < work_N) ? (ltid * chunksize_N) : work_N;
  const libxsmm_blasint thr_end_N = ((ltid + 1) * chunksize_N < work_N) ? ((ltid + 1) * chunksize_N) : work_N;
#endif

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

// new stuff for groupnorm fwd

  LIBXSMM_VLA_DECL(4, float, inp, pinp, CP, HW, CB);            /* [NP, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, float, out, pout, CP, HW, CB);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);                /* [CP,CB] */
  LIBXSMM_VLA_DECL(2, float, beta, pbeta, CB);                  /* [CP,CB] */

  int np, group_size;
  group_size = (CP*CB)/G;

  if (group_size <= CB){
    int cp;
#ifdef TRUE_PARALLEL_FWD
    int cpxnt;
    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      { /* stupid block to keep indentation */
        np = cpxnt/CP;
        cp = cpxnt%CP;
#else
    #pragma omp parallel for collapse(2)
    for(np = 0; np < NP; np++){
      for (cp = 0; cp < CP; cp++){
#endif
        LIBXSMM_ALIGNED(float tmp[2*CB], 64);
        LIBXSMM_ALIGNED(float sum_X[G], 64);
        LIBXSMM_ALIGNED(float sum_X2[G], 64);
        LIBXSMM_ALIGNED(float s[CB], 64);
        LIBXSMM_ALIGNED(float b[CB], 64);

        int i, j, hwb, g;
        libxsmm_matrix_eqn_param eqn_param;
        libxsmm_meltw_unary_param m_reduce_groups_params, v_reduce_groups_params, reduce_HW_params;
        libxsmm_meltw_unary_param all_zero_param;
        libxsmm_meltw_binary_param add_param;
        libxsmm_matrix_arg arg_array[5];

        all_zero_param.out.primary = tmp;
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = &tmp[CB];
        cfg.all_zero_kernel(&all_zero_param);

        all_zero_param.out.primary = sum_X;
        cfg.all_zero_G_kernel(&all_zero_param);
        all_zero_param.out.primary = sum_X2;
        cfg.all_zero_G_kernel(&all_zero_param);

/***************************  Process entire block code *****************************/
        LIBXSMM_ALIGNED(float new_tmp[2*CB], 64);
        reduce_HW_params.out.primary   = new_tmp;                  /* [2*CB] */
        for(hwb=0; hwb < num_HW_blocks; hwb++){
          reduce_HW_params.in.primary    = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);      /* [HW_block, CB] -----> [2 * CB] */
          cfg.reduce_HW_kernel(&reduce_HW_params);

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
          m_reduce_groups_params.in.primary    = &tmp[i];
          m_reduce_groups_params.out.primary   = &sum_X[g];
          v_reduce_groups_params.in.primary    = &tmp[CB + i];
          v_reduce_groups_params.out.primary   = &sum_X2[g];
          cfg.reduce_groups_kernel(&m_reduce_groups_params);
          cfg.reduce_groups_kernel(&v_reduce_groups_params);

          mean[np*G + g] = sum_X[g] / ((float)group_size * HW);
          var[np*G + g] = (sum_X2[g] / ((float)group_size * HW)) - (mean[np*G + g]*mean[np*G + g]);        /* var = E[X^2] - (E[X])^2 */

          for(j = 0; j < group_size; j++){
            s[i + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));                                          /* 1/sqrt(var(X) + eps) */
            b[i + j] = -1 * mean[np*G + g] * s[i + j];                                                     /* -E[X]/sqrt(var(X) + eps) */
          }
        }

        arg_array[1].primary = s;                                                                           /* [CB] */
        arg_array[2].primary = b;                                                                           /* [CB] */
        arg_array[3].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);                                    /* [CB] */
        arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, beta, cp, 0, CB);                                     /* [CB] */

        for(hwb=0; hwb < num_HW_blocks; hwb++){
          arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);            /* [HW, CB] */
          eqn_param.inputs = arg_array;
          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, out, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);        /* [HW, CB] */
          cfg.func10(&eqn_param);                                                                                           /* Normalization equation -> y = ((s*x + b)*gamma + beta) */
        }
      }
    }
  } else{                                                         /* Case when group_size > CB */
#ifdef TRUE_PARALLEL_FWD
    for ( np = thr_begin_N; np < thr_end_N; ++np ) {
#else
    #pragma omp parallel for
    for(np = 0; np < NP; np++){
#endif
      LIBXSMM_ALIGNED(float tmp[2*CB], 64);
      LIBXSMM_ALIGNED(float sum_X[G], 64);
      LIBXSMM_ALIGNED(float sum_X2[G], 64);
      LIBXSMM_ALIGNED(float s[CP*CB], 64);
      LIBXSMM_ALIGNED(float b[CP*CB], 64);

      int i, j, cp, hwb, g;
      float m, v;
      libxsmm_matrix_eqn_param eqn_param;
      libxsmm_meltw_unary_param m_reduce_rows_params, v_reduce_rows_params, m_reduce_groups_params, v_reduce_groups_params, reduce_HW_params;
      libxsmm_meltw_unary_param all_zero_param;
      libxsmm_meltw_binary_param add_param;
      libxsmm_matrix_arg arg_array[5];

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

        reduce_HW_params.out.primary   = new_tmp;                  /* [2*CB] */
        for(hwb=0; hwb < num_HW_blocks; hwb++){
          reduce_HW_params.in.primary    = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);      /* [HW, CB] -----> [2 * CB] */
          cfg.reduce_HW_kernel(&reduce_HW_params);

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
          m_reduce_rows_params.in.primary    = tmp;
          m_reduce_rows_params.out.primary   = &m;
          v_reduce_rows_params.in.primary    = &tmp[CB];
          v_reduce_rows_params.out.primary   = &v;
          cfg.reduce_rows_kernel(&m_reduce_rows_params);
          cfg.reduce_rows_kernel(&v_reduce_rows_params);
          sum_X[g] += m;
          sum_X2[g] += v;
        }
        else{                                                 /* Group size < block size  (Ex.- CP = 4, CB = 16, G = 32, group_size = 2) */
          for(i=0; i < CB; i += group_size){
            m_reduce_groups_params.in.primary    = &tmp[i];
            m_reduce_groups_params.out.primary   = &sum_X[cp*(CB/group_size) + (i/group_size)];
            v_reduce_groups_params.in.primary    = &tmp[CB + i];
            v_reduce_groups_params.out.primary   = &sum_X2[cp*(CB/group_size) + (i/group_size)];
            cfg.reduce_groups_kernel(&m_reduce_groups_params);
            cfg.reduce_groups_kernel(&v_reduce_groups_params);
          }
        }
      }

      for(g = 0; g < G; g++){                                                  /* mean and variance calculation */
        mean[np*G + g] = sum_X[g] / ((float)group_size * HW);
        var[np*G + g] = (sum_X2[g] / ((float)group_size * HW)) - (mean[np*G + g]*mean[np*G + g]);        /* var = E[X^2] - (E[X])^2 */

        for(j = 0; j < group_size; j++){
          s[g*group_size + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));                               /* 1/sqrt(var(X) + eps) */
          b[g*group_size + j] = -1 * mean[np*G + g] * s[g*group_size + j];                               /* -E[X]/sqrt(var(X) + eps) */
        }
      }

      for (cp = 0; cp < CP; cp++){

        arg_array[1].primary = &s[cp*CB];                                                                   /* [CB] */
        arg_array[2].primary = &b[cp*CB];                                                                   /* [CB] */
        arg_array[3].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);                                    /* [CB] */
        arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, beta, cp, 0, CB);                                     /* [CB] */

        for(hwb=0; hwb < num_HW_blocks; hwb++){
          arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);                       /* [HW, CB] */
          eqn_param.inputs = arg_array;
          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, out, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);                   /* [HW,CB] */
          cfg.func10(&eqn_param);                                                                                 /* Normalization equation -> y = ((s*x + b)*gamma + beta) */
        }
      }
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);

}


#endif /* REFACTORED_FWD */

#ifdef REFACTORED_BWD

//    tpp_groupnorm_bwd_fp32(NP, CP, HW, bc, G, num_HW_blocks, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func13, func14, func15, all_zero_kernel, add_kernel, eps);

//void tpp_groupnorm_bwd_fp32(long NP, long CP, long HW, long bc, long G, long num_HW_blocks, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta,
//    libxsmm_matrix_eqn_function dgamma_func, libxsmm_matrix_eqn_function dbeta_func, libxsmm_matrix_eqn_function db_func, libxsmm_matrix_eqn_function ds_func, libxsmm_matrix_eqn_function din_func,
//    libxsmm_meltwfunction_unary all_zero_kernel, libxsmm_meltwfunction_binary add_kernel, float eps) {


my_gn_bwd_config setup_my_gn_bwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint G, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint bc,
                                 libxsmm_blasint threads, my_normalization_fuse fuse_type ) {

  my_gn_bwd_config res;

  size_t dbeta_N_offset;

  libxsmm_blasint ldo = bc;
  libxsmm_blasint ld  = bc;
  libxsmm_blasint tmp_ld2;
  libxsmm_blasint my_eqn11, my_eqn12, my_eqn13, my_eqn14, my_eqn15;

  libxsmm_meltw_unary_shape  unary_shape;
  libxsmm_meltw_binary_shape binary_shape;

  libxsmm_meltw_unary_flags   unary_flags;
  libxsmm_meltw_binary_flags  binary_flags;
  libxsmm_meltw_ternary_flags ternary_flags;

//  libxsmm_meltw_unary_flags jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
//  libxsmm_meltw_unary_type  unary_type;

  libxsmm_datatype  in_dt  = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype  out_dt = LIBXSMM_DATATYPE_F32;

  libxsmm_datatype dtype = LIBXSMM_DATATYPE_F32;

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

  /* when masking is on, bc must be divisible by 8 for compressing mask into char array (otherwise strides are wrong for relumask */
  if ( (res.fuse_type == 4 || res.fuse_type == 5) && (res.bc % BITS_PER_CHAR != 0)) {
    fprintf( stderr, "bc = %d is not divisible by BITS_PER_CHAR = %d. Bailing...!\n", res.bc, BITS_PER_CHAR);
    exit(-1);
  }

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  ldo = res.bc;
  unary_shape.m         = res.bc;
  unary_shape.n         = 1;
  unary_shape.ldi       = NULL;
  unary_shape.ldo       = &ldo;
  unary_shape.in_type   = dtype;
  unary_shape.out_type  = dtype;
  unary_shape.comp_type = dtype;
  unary_flags           = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.all_zero_kernel   = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags);
  //res.all_zero_kernel = libxsmm_dispatch_meltw_unary(res.bc, 1, NULL, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  if ( res.all_zero_kernel == NULL) {
    fprintf( stderr, "JIT for initialization by unary all zero copy kernel failed for fwd. Bailing...!\n");
    exit(-1);
  }

  ld = res.bc;
  binary_shape.m         = res.bc;
  binary_shape.n         = 1;
  binary_shape.in_type   = dtype;
  binary_shape.comp_type = dtype;
  binary_shape.out_type  = dtype;
  binary_shape.ldi       = &ld;
  binary_shape.ldi2      = &ld;
  binary_shape.ldo       = &ld;
  binary_flags           = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  res.add_kernel         = libxsmm_dispatch_meltw_binary_v2(LIBXSMM_MELTW_TYPE_BINARY_ADD, binary_shape, binary_flags);
  //res.add_kernel = libxsmm_dispatch_meltw_binary(res.bc, 1, &ld, &ld, &ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_MELTW_TYPE_BINARY_ADD);
  if ( res.add_kernel == NULL) {
      fprintf( stderr, "JIT for initialization of add_kernel failed for fwd. Bailing...!\n");
      exit(-1);
  }

  /* Group norm equations */
  /* Create MatEq for bwd layernorm */

  ld = res.bc;
  tmp_ld2 = 1;

  /* dgamma function  */
  my_eqn11 = libxsmm_matrix_eqn_create();                                                       /* dgamma = ((inp *a + b) * dout) + dgamma */
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn11, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32);                   /* dgamma = ((inp *a + b) * dout) + dgamma */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn11, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);   /* [HW, bc] -> [bc] */
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn11, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32);                   /* ((inp *a + b) * dout) */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn11, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32);
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, res.bc, res.H*res.W/res.num_HW_blocks, ld, 0, 0, in_dt );                        /* inp [HW, bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, res.bc, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 );           /* a [bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, res.bc, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );           /* b [bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, res.bc, res.H*res.W/res.num_HW_blocks, ld, 3, 0, in_dt );                        /* dout [HW, bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, res.bc, 1, 1, 4, 0, LIBXSMM_DATATYPE_F32 );           /* dgamma [bc] */
  res.dgamma_func = libxsmm_dispatch_matrix_eqn( res.bc, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn11 );      /* dgamma [bc] */
  if ( res.dgamma_func == NULL) {
    fprintf( stderr, "JIT for TPP fwd dgamma_func (eqn11) failed. Bailing...!\n");
    exit(-1);
  }

  /* dbeta function  */
  my_eqn12 = libxsmm_matrix_eqn_create();                                                       /* dbeta [bc] = dout [HW, bc] + dbeta [bc] */
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn12, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );      /* dbeta_tmp [HW, bc] */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn12, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);  /* [HW, bc] -> [bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn12, res.bc, res.H*res.W/res.num_HW_blocks, ld, 3, 0, in_dt );                        /* dout [HW, bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn12, res.bc, 1, 1, 5, 0, LIBXSMM_DATATYPE_F32 );           /* dbeta [bc] */
  res.dbeta_func = libxsmm_dispatch_matrix_eqn( res.bc, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn12 );      /* dbeta [bc] */
  if ( res.dbeta_func == NULL) {
    fprintf( stderr, "JIT for TPP fwd dbeta_func (eqn12) failed. Bailing...!\n");
    exit(-1);
  }

  /* db new equation */
  my_eqn13 = libxsmm_matrix_eqn_create();                                                       /* db [bc] = (dout * gamma) [HW, bc] + db [bc]*/
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn13, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );                  /* db [bc] */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn13, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);   /* [HW, bc] -> [bc] */
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn13, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn13, res.bc, res.H*res.W/res.num_HW_blocks, ld, 3, 0, in_dt );                        /* dout [HW, bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn13, res.bc, 1, 1, 6, 0, in_dt );                          /* gamma [bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn13, res.bc, 1, 1, 9, 0, LIBXSMM_DATATYPE_F32 );           /* db [bc] */
  res.db_func = libxsmm_dispatch_matrix_eqn( res.bc, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn13 );      /* db [bc] */
  if ( res.db_func == NULL) {
    fprintf( stderr, "JIT for TPP fwd db_func (eqn13) failed. Bailing...!\n");
    exit(-1);
  }

  /* ds new equation */
  my_eqn14 = libxsmm_matrix_eqn_create();                                                       /* ds [bc] = ((dout * gamma) * inp) [HW, bc] + ds [bc] */
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn14, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );                  /* ds [bc] */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn14, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);   /* [HW, bc] -> [bc] */
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn14, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn14, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1, LIBXSMM_DATATYPE_F32 );       /*(dout * gamma)*/
  libxsmm_matrix_eqn_push_back_arg( my_eqn14, res.bc, res.H*res.W/res.num_HW_blocks, ld, 3, 0, in_dt );                        /* dout [HW, bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn14, res.bc, 1, 1, 6, 0, in_dt );                          /* gamma [bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn14, res.bc, res.H*res.W/res.num_HW_blocks, ld, 0, 0, in_dt );                        /* inp [HW, bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn14, res.bc, 1, 1, 8, 0, LIBXSMM_DATATYPE_F32 );           /* ds [bc] */
  res.ds_func = libxsmm_dispatch_matrix_eqn( res.bc, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn14 );      /* ds [bc] */
  if ( res.ds_func == NULL) {
    fprintf( stderr, "JIT for TPP fwd ds_func (eqn14) failed. Bailing...!\n");
    exit(-1);
  }

  /* din equation */
  my_eqn15 = libxsmm_matrix_eqn_create();                                                       /* din = ((gamma * a) * dout) + (inp * b + c) */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn15, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, (libxsmm_meltw_ternary_flags)(LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT), LIBXSMM_DATATYPE_F32);
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn15, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, res.bc, 1, 1, 6, 0, in_dt );                          /* gamma [bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, res.bc, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 );           /* a [bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, res.bc, res.H*res.W/res.num_HW_blocks, ld, 3, 0, in_dt );                        /* dout [HW, bc] */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn15, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, (libxsmm_meltw_ternary_flags)(LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT), LIBXSMM_DATATYPE_F32);
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, res.bc, res.H*res.W/res.num_HW_blocks, ld, 0, 0, in_dt );                        /* inp [HW, bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, res.bc, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );           /* b [bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, res.bc, 1, 1, 7, 0, LIBXSMM_DATATYPE_F32 );           /* c [bc] */
  res.din_func = libxsmm_dispatch_matrix_eqn( res.bc, res.H*res.W/res.num_HW_blocks, &ld, in_dt, my_eqn15 );                         /* din [HW, bc] */
  if ( res.din_func == NULL) {
    fprintf( stderr, "JIT for TPP fwd din_func (eqn15) failed. Bailing...!\n");
    exit(-1);
  }

  // FIXME: Need to modify as this code is for the batchnorm
  /* init scratch */
  dbeta_N_offset = LIBXSMM_UP2(res.CP * res.N * res.bc, 64);
  res.scratch_size =  sizeof(float) * ( dbeta_N_offset /* dbeta_N*/ + LIBXSMM_UP2(res.CP * res.N * res.bc, 64) /*dgamma_N */ );

  return res;
}


// FIXME: Set const modifiers properly?
// FIXME: Add a "my_pass" type of input argument to distinguish between backward over weights vs backward over data
void my_gn_bwd_exec( my_gn_bwd_config cfg, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta, float eps,
                     int start_tid, int my_tid, void *scratch) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint NP = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint G  = cfg.G;
  const libxsmm_blasint HW = cfg.H * cfg.W;
  const libxsmm_blasint CB = cfg.bc;
  const libxsmm_blasint num_HW_blocks = cfg.num_HW_blocks;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;
#ifdef TRUE_PARALLEL_BWD
  /* number of tasks that could be run in parallel for 1d blocking */
  // Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here?
  const libxsmm_blasint work_dN = N * CP;
  /* compute chunk size */
  const libxsmm_blasint chunksize_dN = (work_dN % cfg.threads == 0) ?
    (work_dN / cfg.threads) : ((work_dN / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_dN = (ltid * chunksize_dN < work_dN) ? (ltid * chunksize_dN) : work_dN;
  const libxsmm_blasint thr_end_dN = ((ltid + 1) * chunksize_dN < work_dN) ? ((ltid + 1) * chunksize_dN) : work_dN;

  /* number of tasks that could be run in parallel for 1d blocking over CP */
  // Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here?
  const libxsmm_blasint work_C = CP;
  /* compute chunk size */
  const libxsmm_blasint chunksize_C = (work_C % cfg.threads == 0) ?
    (work_C / cfg.threads) : ((work_C / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_C = (ltid * chunksize_C < work_C) ? (ltid * chunksize_C) : work_C;
  const libxsmm_blasint thr_end_C = ((ltid + 1) * chunksize_C < work_C) ? ((ltid + 1) * chunksize_C) : work_C;

  /* number of tasks that could be run in parallel for 1d blocking over NP */
  // Question: each thread should take a number of full (of length NP chunks) or can we really do a partial split here?
  const libxsmm_blasint work_N = NP;
  /* compute chunk size */
  const libxsmm_blasint chunksize_N = (work_N % cfg.threads == 0) ?
    (work_N / cfg.threads) : ((work_N / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_N = (ltid * chunksize_N < work_N) ? (ltid * chunksize_N) : work_N;
  const libxsmm_blasint thr_end_N = ((ltid + 1) * chunksize_N < work_N) ? ((ltid + 1) * chunksize_N) : work_N;
#endif

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

//// new stuff for bwd gn exec

  int group_size;
  group_size = (CP*CB)/G;

  const float scale = 1.0f / ((float)group_size * HW);

  LIBXSMM_VLA_DECL(4, float, din, pdin, CP, HW, CB);
  LIBXSMM_VLA_DECL(4, float, inp, pinp, CP, HW, CB);
  LIBXSMM_VLA_DECL(4, float, dout, pdout, CP, HW, CB);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, CB);

#ifdef TRUE_PARALLEL_BWD
  const libxsmm_blasint dbeta_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + N * CP * CB), 64) - ((uintptr_t)(scratch))) / sizeof(float);
  LIBXSMM_VLA_DECL(3, float, dgamma_NP, ((float*)scratch),                  CP, CB);  /* [N, CP, CB] */
  LIBXSMM_ASSUME_ALIGNED(dgamma_NP_, 64);
  LIBXSMM_VLA_DECL(3, float, dbeta_NP,  ((float*)scratch) + dbeta_N_offset, CP, CB);  /* [N, CP, CB] */
  LIBXSMM_ASSUME_ALIGNED(dbeta_NP_, 64);
#else
  LIBXSMM_ALIGNED(float dgamma_NP[NP*CP*CB], 64);
  LIBXSMM_ALIGNED(float dbeta_NP[NP*CP*CB], 64);
#endif
//  LIBXSMM_ALIGNED(float dgamma_NP[NP*CP*CB], 64);
//  LIBXSMM_ALIGNED(float dbeta_NP[NP*CP*CB], 64);

  if (group_size <= CB){
#ifdef TRUE_PARALLEL_BWD
    {
#else
    #pragma omp parallel
    {
#endif
      LIBXSMM_ALIGNED(float a[CB], 64);
      LIBXSMM_ALIGNED(float b[CB], 64);
      LIBXSMM_ALIGNED(float c[CB], 64);
      LIBXSMM_ALIGNED(float ds[CB], 64);
      LIBXSMM_ALIGNED(float db[CB], 64);

      int np, cp;
#ifdef TRUE_PARALLEL_BWD
      int cpxnt;
      for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
        { /* stupid block to keep indentation */
          np = cpxnt/CP;
          cp = cpxnt%CP;
#else
      #pragma omp for collapse(2)
      for (np = 0; np < NP; np++) {
        for (cp = 0; cp < CP; cp++) {
#endif
          int j, g, hwb, lg;

          libxsmm_matrix_eqn_param eqn_param;
          libxsmm_meltw_unary_param all_zero_param;
          libxsmm_matrix_arg arg_array[10];
          eqn_param.inputs = arg_array;

          /* for(j = 0; j < CB; j++){
              dgamma_NP[np*CP*CB + cp*CB + j] = 0.0f;
              dbeta_NP[np*CP*CB + cp*CB + j] = 0.0f;
           } */

          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_NP, np, cp, 0, CP, CB);//&dgamma_NP[np*CP*CB + cp*CB];
          cfg.all_zero_kernel(&all_zero_param);
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_NP, np, cp, 0, CP, CB);//&dbeta_NP[np*CP*CB + cp*CB];
          cfg.all_zero_kernel(&all_zero_param);
          all_zero_param.out.primary = ds;
          cfg.all_zero_kernel(&all_zero_param);
          all_zero_param.out.primary = db;
          cfg.all_zero_kernel(&all_zero_param);


          for(g = (cp*CB)/group_size; g < ((cp+1)*CB)/group_size; g++){                                                  /* compute a and b for each channel from group means and variance */
            lg = g - (cp*CB)/group_size;
            for(j = 0; j < group_size; j++){
              a[lg*group_size + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));
              b[lg*group_size + j] = -a[lg*group_size + j]*mean[np*G + g];
            }
          }

          arg_array[1].primary = a;
          arg_array[2].primary = b;
          arg_array[4].primary = &LIBXSMM_VLA_ACCESS(3, dgamma_NP, np, cp, 0, CP, CB);//&dgamma_NP[np*CP*CB + cp*CB];
          arg_array[5].primary = &LIBXSMM_VLA_ACCESS(3, dbeta_NP, np, cp, 0, CP, CB);//&dbeta_NP[np*CP*CB + cp*CB];
          arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
          arg_array[8].primary = ds;
          arg_array[9].primary = db;

          for(hwb=0; hwb < num_HW_blocks; hwb++){
            arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
            arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);

            eqn_param.output.primary = ds;
            cfg.ds_func(&eqn_param);

            eqn_param.output.primary = db;
            cfg.db_func(&eqn_param);

            eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_NP, np, cp, 0, CP, CB);//&dgamma_NP[np*CP*CB + cp*CB];
            cfg.dgamma_func(&eqn_param);

            eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_NP, np, cp, 0, CP, CB);//&dbeta_NP[np*CP*CB + cp*CB];
            cfg.dbeta_func(&eqn_param);
          }

          /* b = (db * mean[nb] - ds) * a * a * a * scale; */
          /* c = -b * mean[nb] - db * a * scale; */

          for(g = (cp*CB)/group_size; g < ((cp+1)*CB)/group_size; g++){                                                  /* compute b and c for each channel from group means and variance */
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
          arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
          arg_array[7].primary = c;

          for(hwb=0; hwb < num_HW_blocks; hwb++){
            arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
            arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
            eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, din, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
            cfg.din_func(&eqn_param);
          }
        }
      }

#ifdef TRUE_PARALLEL_BWD
      for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {
#else
        #pragma omp for
        for (cp = 0; cp < CP; cp++) {
#endif
//      #pragma omp for
//      for (cp = 0; cp < CP; cp++) {
        for (np=0; np < NP; np++ ) {
          int cb;
          for(cb = 0; cb < CB; cb++){
            LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, CB) += LIBXSMM_VLA_ACCESS(3, dgamma_NP, np, cp, cb, CP, CB);//dgamma_NP[np*CP*CB + cp*CB + cb];
            LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, CB)  += LIBXSMM_VLA_ACCESS(3, dbeta_NP,  np, cp, cb, CP, CB);//dbeta_NP[np*CP*CB + cp*CB + cb];
          }
        }
      }
    }
  }
  else{
#ifdef TRUE_PARALLEL_BWD
    {
#else
    #pragma omp parallel
    {
#endif
      LIBXSMM_ALIGNED(float a[CP*CB], 64);
      LIBXSMM_ALIGNED(float b[CP*CB], 64);
      LIBXSMM_ALIGNED(float c[CP*CB], 64);
      LIBXSMM_ALIGNED(float ds[CP*CB], 64);
      LIBXSMM_ALIGNED(float db[CP*CB], 64);
      int np;

#ifdef TRUE_PARALLEL_BWD
      for ( np = thr_begin_N; np < thr_end_N; ++np ) {
#else
      #pragma omp for
      for (np = 0; np < NP; np++) {
#endif
        int j, g, cp, hwb;

        libxsmm_matrix_eqn_param eqn_param;
        libxsmm_meltw_unary_param all_zero_param;
        libxsmm_matrix_arg arg_array[10];
        eqn_param.inputs = arg_array;

        /* for(j = 0; j < CP*CB; j++){ */
        /*   dgamma_NP[np*CP*CB + j] = 0.0f; */
        /*   dbeta_NP[np*CP*CB + j] = 0.0f; */
        /* } */

        for (cp = 0; cp < CP; cp++) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_NP, np, cp, 0, CP, CB);//&dgamma_NP[np*CP*CB + cp*CB];
          cfg.all_zero_kernel(&all_zero_param);
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_NP, np, cp, 0, CP, CB);//&dbeta_NP[np*CP*CB + cp*CB];
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
          arg_array[4].primary = &LIBXSMM_VLA_ACCESS(3, dgamma_NP, np, cp, 0, CP, CB);//&dgamma_NP[np*CP*CB + cp*CB];
          arg_array[5].primary = &LIBXSMM_VLA_ACCESS(3, dbeta_NP,  np, cp, 0, CP, CB);//&dbeta_NP[np*CP*CB + cp*CB];
          arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
          arg_array[8].primary = &ds[cp*CB];
          arg_array[9].primary = &db[cp*CB];

          for(hwb=0; hwb < num_HW_blocks; hwb++){
            arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
            arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);

            eqn_param.output.primary = &ds[cp*CB];
            cfg.ds_func(&eqn_param);

            eqn_param.output.primary = &db[cp*CB];
            cfg.db_func(&eqn_param);

            eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_NP, np, cp, 0, CP, CB);//&dgamma_NP[np*CP*CB + cp*CB];
            cfg.dgamma_func(&eqn_param);

            eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_NP, np, cp, 0, CP, CB);//&dbeta_NP[np*CP*CB + cp*CB];
            cfg.dbeta_func(&eqn_param);
          }
        }

        /* b = (db * mean[nb] - ds) * a * a * a * scale; */
        /* c = -b * mean[nb] - db * a * scale; */

        for(g = 0; g < G; g++){                                                  /* compute b and c for each channel from group means and variance */
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
          arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
          arg_array[7].primary = &c[cp*CB];

          for(hwb=0; hwb < num_HW_blocks; hwb++){
            arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
            arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
            eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, din, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
            cfg.din_func(&eqn_param);
          }
        }
      }

      int cp;
#ifdef TRUE_PARALLEL_BWD
      for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {
#else
      #pragma omp for
      for (cp = 0; cp < CP; cp++) {
#endif
        for (np=0; np < NP; np++ ) {
          int cb;
          for(cb = 0; cb < CB; cb++){
            LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, CB) += LIBXSMM_VLA_ACCESS(3, dgamma_NP, np, cp, cb, CP, CB);//dgamma_NP[np*CP*CB + cp*CB + cb];
            LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, CB)  += LIBXSMM_VLA_ACCESS(3, dbeta_NP,  np, cp, cb, CP, CB);//dbeta_NP[np*CP*CB + cp*CB + cb];
          }
        }
      }
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);
}

#endif /* for REFACTORED_BWD */


int main( int argc, char* argv[] ) {

#ifdef REFACTORED_FWD
  my_gn_fwd_config my_gn_fwd;
#endif

#ifdef REFACTORED_BWD
  my_gn_bwd_config my_gn_bwd;
#endif

#if defined(REFACTORED_BWD) || defined(REFACTORED_FWD)
  naive_fusedgroupnorm_t naive_param;
  void *scratch;
#endif


#if !defined(REFACTORED_BWD) || !defined(REFACTORED_FWD)
  // Some are unused if either FWD or BWD is defined
  libxsmm_blasint my_eqn10, my_eqn11, my_eqn12, my_eqn13, my_eqn14, my_eqn15;
  libxsmm_matrix_eqn_function func10, func11, func12, func13, func14, func15;
  libxsmm_meltw_unary_flags jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type  unary_type;
  libxsmm_meltwfunction_unary reduce_rows_kernel, reduce_HW_kernel, reduce_groups_kernel;

  libxsmm_datatype  in_dt  = LIBXSMM_DATATYPE_F32;
  libxsmm_datatype  out_dt = LIBXSMM_DATATYPE_F32;

  libxsmm_blasint ld, tmp_ld, tmp_ld2;
#endif

  const float eps = FLT_EPSILON;
  libxsmm_blasint i, it;
  float *inp, *out, *dinp, *dout, *eqn_dinp, *eqn_dout, *dbeta, *eqn_dbeta, *dgamma, *eqn_dgamma, *eqn_out, *gamma, *beta, *cache_fl, *mean, *var, sum = 0.0;
#if defined(REFACTORED_BWD) || defined(REFACTORED_FWD)
  float *naive_inp, *naive_out, *naive_rcpstdev, *naive_zeros, *naive_dinp, *naive_dout, *naive_dbeta, *naive_dgamma;

#ifdef COMPUTE_FP64_REFERENCE
  double *naive_inp_fp64, *naive_out_fp64, *naive_rcpstdev_fp64, *naive_zeros_fp64, *naive_dinp_fp64, *naive_dout_fp64, *naive_dbeta_fp64, *naive_dgamma_fp64;
  double *beta_fp64, *gamma_fp64, *mean_fp64, *var_fp64;
  double *dbeta_fp64, *dgamma_fp64;
  float *naive_out_fp64_downscaled_to_fp32, *out_fp64_downscaled_to_fp32;
  float *naive_dinp_fp64_downscaled_to_fp32, *dinp_fp64_downscaled_to_fp32;
  float *dgamma_fp64_downscaled_to_fp32;
  float *dbeta_fp64_downscaled_to_fp32;
#endif
#endif /* for new code REFACTORED_FWD or REFACTORED_BWD */

  int iters     = 100;
  int N         = 28;
  int C         = 2 * 64;
  int G         = 1;
  int H         = 0; /* defined later */
  int W         = 0; /* defined later */
  int HW        = 784;
  int bc        = 64;
  int CP        = 0; /* defined later */
  int stride    = 1; /* stride when accessing inputs */
  int pad_h_in  = 0; /* padding mode */
  int pad_w_in  = 0; /* padding mode */
  int pad_h_out = 0; /* padding mode */
  int pad_w_out = 0; /* padding mode */
  int norm_type = 0; /* 0: full batchnorm, 1: batch scaling only */
  int fuse_type = 5; /* 0: nothing fused, 1: relu fused, 2: ewise fused, 3: relu and ewise fused, 4: relu with mask, 5: relu and ewise with mask  */

  int stride_h = 0;  /* defined later */
  int stride_w = 0;  /* defined later */

  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 1 : atof(env_check));

#if defined(_OPENMP)
  int nThreads = omp_get_max_threads(); /* number of threads */
#else
  int nThreads = 1; /* number of threads */
#endif
  unsigned long long l_start, l_end;
  double l_total = 0, l_total2 = 0;
  double t_vec = 0, t_tpp = 0;

  libxsmm_matdiff_info norms_fwd, norms_bwd_d, norms_bwd_beta, norms_bwd_gamma;

  libxsmm_matdiff_clear(&norms_fwd);
  libxsmm_matdiff_clear(&norms_bwd_d);
  libxsmm_matdiff_clear(&norms_bwd_beta);
  libxsmm_matdiff_clear(&norms_bwd_gamma);

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters N CP HW bc num_HW_blocks\n", argv[0]);
    return 0;
  }

  libxsmm_rng_set_seed(1);

  /* reading new values from cli */
  i = 1;
  if ( argc > i ) iters = atoi(argv[i++]);
  if ( argc > i ) N = atoi(argv[i++]);
  if ( argc > i ) C = atoi(argv[i++]);
  if ( argc > i ) G = atoi(argv[i++]);
  if ( argc > i ) H  = atoi(argv[i++]);
  if ( argc > i ) W  = atoi(argv[i++]);
  if ( argc > i ) bc = atoi(argv[i++]);
  if ( argc > i ) pad_w_in   = atoi(argv[i++]);
  if ( argc > i ) pad_h_in   = atoi(argv[i++]);
  if ( argc > i ) pad_w_out  = atoi(argv[i++]);
  if ( argc > i ) pad_h_out  = atoi(argv[i++]);
  if ( argc > i ) stride     = atoi(argv[i++]);
  if ( argc > i ) norm_type  = atoi(argv[i++]);
  if ( argc > i ) fuse_type  = atoi(argv[i++]);

  CP = C / bc;

  /* if H and W are read from cli, redefine HW */
  if (H && W)
    HW = H*W;
  else { /* else, set formally H and W from the value of HW hardcoded above */
    H = HW;
    W = 1;
  }

  if (pad_w_in || pad_h_in || pad_w_out || pad_h_out) {
    printf("Padding is not supported (must be all 0)\n");
    return -1;
  }

  if ( stride != 1 ) {
    printf("Non-unit stride is not supported \n");
    return -1;
  }

  if ( norm_type != 0 ) {
    printf("Only full batchnorm (norm_type = 0) is supported \n");
    return -1;
  }

  if ((fuse_type == 4 || fuse_type == 5) && bc % 16 != 0) {
    fprintf( stderr, "Fused ReLU with a mask will not work for sizes which are not a multiple of 16 (2BYTE limitation). Bailing...!\n");
    return -1;
  }

  if (fuse_type != 0 && fuse_type != 2 && fuse_type != 4 && fuse_type != 5) {
    printf("Unsupported fuse_type %d was provided (0, 2, 4 and 5 are supported only)\n", fuse_type);
    return -1;
  }

  stride_w = stride;
  stride_h = stride;

#if defined(REFACTORED_BWD) || defined(REFACTORED_FWD)
  /* set struct for naive batch normalization */
  naive_param.N = N;
  naive_param.C = CP*bc;
  naive_param.G = G;
  naive_param.H = H;
  naive_param.W = W;
  naive_param.stride_h = 1;
  naive_param.stride_w = 1;
  naive_param.fuse_type = 0; /* nothing fused */

#if defined(__SSE3__)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

#endif

  /* print some summary */
  printf("##########################################\n");
  printf("#          Setting Up (Common)           #\n");
  printf("##########################################\n");
  printf("PARAMS: N:%d  C:%d  G:%d CP:%d bc:%d H:%d W:%d STRIDE:%d (PADDING: must be 0s)\n", N, CP*bc, G, CP, bc, H, W, stride);
  printf("PARAMS: FUSE TYPE:%d\n", fuse_type);
  printf("PARAMS: ITERS:%d", iters); if (LIBXSMM_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  printf("SIZE Input  (MB): %10.2f MiB\n", (double)(N*CP*HW*bc*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output (MB): %10.2f MiB\n", (double)(N*CP*HW*bc*sizeof(float))/(1024.0*1024.0) );

  /* allocate data */
  inp        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  out        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  dinp       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  dout       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  dgamma     = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  dbeta      = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  eqn_dinp   = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  eqn_dout   = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  eqn_dgamma = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  eqn_dbeta  = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  gamma      = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  beta       = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  mean       = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  var        = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  eqn_out    = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  cache_fl   = (float*) libxsmm_aligned_malloc( sizeof(float)*1024*1024,   2097152);

#if defined(REFACTORED_BWD) || defined(REFACTORED_FWD)
  naive_inp      = (float*) libxsmm_aligned_malloc( sizeof(float)*N*(CP*bc)*HW*1, 2097152);
  naive_out      = (float*) libxsmm_aligned_malloc( sizeof(float)*N*(CP*bc)*HW*1, 2097152);
  naive_dinp     = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  naive_dout     = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  naive_dgamma   = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  naive_dbeta    = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  naive_rcpstdev = (float*) libxsmm_aligned_malloc( sizeof(float)*(CP*bc),   2097152);
  naive_zeros    = (float*) libxsmm_aligned_malloc( sizeof(float)*N*(CP*bc)*HW*1, 2097152);

#ifdef COMPUTE_FP64_REFERENCE
  naive_inp_fp64      = (double*) libxsmm_aligned_malloc( sizeof(double)*N*(CP*bc)*HW*1, 2097152);
  naive_out_fp64      = (double*) libxsmm_aligned_malloc( sizeof(double)*N*(CP*bc)*HW*1, 2097152);
  naive_dinp_fp64     = (double*) libxsmm_aligned_malloc( sizeof(double)*N*CP*HW*bc,   2097152);
  naive_dout_fp64     = (double*) libxsmm_aligned_malloc( sizeof(double)*N*CP*HW*bc,   2097152);
  naive_dgamma_fp64   = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*bc,   2097152);
  naive_dbeta_fp64    = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*bc,   2097152);
  naive_rcpstdev_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*(CP*bc),   2097152);
  naive_zeros_fp64    = (double*) libxsmm_aligned_malloc( sizeof(double)*N*(CP*bc)*HW*1, 2097152);

  gamma_fp64  = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*bc,   2097152);
  beta_fp64   = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*bc,   2097152);
  mean_fp64   = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*bc,   2097152);
  var_fp64    = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*bc,   2097152);
  dgamma_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*bc,   2097152);
  dbeta_fp64  = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*bc,   2097152);

  naive_out_fp64_downscaled_to_fp32  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*(CP*bc)*HW*1, 2097152);
  naive_dinp_fp64_downscaled_to_fp32 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*(CP*bc)*HW*1, 2097152);

  out_fp64_downscaled_to_fp32        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*(CP*bc)*HW*1, 2097152);
  dinp_fp64_downscaled_to_fp32       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*(CP*bc)*HW*1, 2097152);
  dgamma_fp64_downscaled_to_fp32     = (float*) libxsmm_aligned_malloc( sizeof(float)*(CP*bc)*1, 2097152);
  dbeta_fp64_downscaled_to_fp32      = (float*) libxsmm_aligned_malloc( sizeof(float)*(CP*bc)*1, 2097152);
#endif

#endif

  /* initialize data */
  init_buf(inp,  N*CP*HW*bc, 1, 0);
  init_buf(out,  N*CP*HW*bc, 1, 0);
  init_buf(dinp, N*CP*HW*bc, 1, 0);
  init_buf(dout, N*CP*HW*bc, 1, 0);
  copy_buf(out,  eqn_out,  N*CP*HW*bc);
  copy_buf(dinp, eqn_dinp, N*CP*HW*bc);
  copy_buf(dout, eqn_dout, N*CP*HW*bc);
#if defined(REFACTORED_BWD) || defined(REFACTORED_FWD)
  zero_buf(naive_zeros, N*CP*HW*bc);
  zero_buf_fp64(naive_zeros_fp64, N*CP*HW*bc);
#endif

  init_buf(gamma,  CP*bc, 1, 0);
  init_buf(beta,   CP*bc, 1, 0);
  init_buf(dgamma, CP*bc, 1, 0);
  init_buf(dbeta,  CP*bc, 1, 0);
  copy_buf(dgamma, eqn_dgamma, CP*bc);
  copy_buf(dbeta,  eqn_dbeta,  CP*bc);
#ifdef COMPUTE_FP64_REFERENCE
  extend_buf_fp32_to_fp64(gamma, gamma_fp64, CP*bc);
  extend_buf_fp32_to_fp64(beta,  beta_fp64,  CP*bc);
#endif

  init_buf(cache_fl,  1024*1024, 1, 0);

  /* setup TPPs (standalone or through the configs) */
#if !defined(REFACTORED_BWD) || !defined(REFACTORED_FWD)

  int num_HW_blocks = (H > W ? H : W);//NUM_HW_BLOCKS;//16
  int CB = bc;
  int NP = N;

  libxsmm_blasint ldo = G;
  libxsmm_meltwfunction_unary all_zero_G_kernel = libxsmm_dispatch_meltw_unary(G, 1, NULL, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  if ( all_zero_G_kernel == NULL) {
    fprintf( stderr, "JIT for initialization by unary all zero group copy kernel failed. Bailing...!\n");
    exit(-1);
  }

  ldo = CB;
  libxsmm_meltwfunction_unary all_zero_kernel = libxsmm_dispatch_meltw_unary(CB, 1, NULL, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  if ( all_zero_G_kernel == NULL) {
    fprintf( stderr, "JIT for initialization by unary all zero copy kernel failed. Bailing...!\n");
    exit(-1);
  }

  libxsmm_meltwfunction_unary copy_kernel = libxsmm_dispatch_meltw_unary(CB, 1, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
  if ( copy_kernel == NULL) {
      fprintf( stderr, "JIT for initialization by copy kernel failed. Bailing...!\n");
      exit(-1);
  }

  /* TPPs for reducing X and X2 in HW*/
  ld = CB;
  tmp_ld = CB;

  unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD;
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  reduce_HW_kernel = libxsmm_dispatch_meltw_unary(CB, HW/num_HW_blocks, &ld, &tmp_ld, in_dt, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);

  libxsmm_blasint group_size = (CP*CB)/G;

  libxsmm_meltwfunction_binary add_kernel = libxsmm_dispatch_meltw_binary(CB, 1, &ld, &ld, &ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_MELTW_TYPE_BINARY_ADD);
  if ( add_kernel == NULL) {
      fprintf( stderr, "JIT for initialization of add kernel failed. Bailing...!\n");
      exit(-1);
  }


  /* TPP for reducing groups */
  ld = group_size;                /* group_size = (CP*CB)/G */
  tmp_ld = 1;
  unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD;
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS;
  reduce_groups_kernel = libxsmm_dispatch_meltw_unary(group_size, 1, &ld, &tmp_ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);

  ld = CB;
  tmp_ld = 1;
  unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD;
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS;
  reduce_rows_kernel = libxsmm_dispatch_meltw_unary(CB, 1, &ld, &tmp_ld, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);

#endif /* for #ifdef-else !REFACTORED_FWD or !REFACTORED_BWD */

#ifdef REFACTORED_FWD
#else

  /* TPP for foward */
  ld = CB;
  tmp_ld = 1;
  tmp_ld2 = 1;
  my_eqn10 = libxsmm_matrix_eqn_create();                                                        /* y = (s*x + b)*gamma + beta */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn10, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32);
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn10, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32);
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, CB, HW/num_HW_blocks, ld, 0, 0, in_dt );                         /* x = [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, CB, 1, tmp_ld, 1, 0, LIBXSMM_DATATYPE_F32 );       /* s = [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, CB, 1, tmp_ld, 2, 0, LIBXSMM_DATATYPE_F32 );       /* b = [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, CB, 1, tmp_ld2, 3, 0, in_dt );                     /* gamma = [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn10, CB, 1, tmp_ld2, 4, 0, in_dt );                     /* beta = [CB] */
  func10 = libxsmm_dispatch_matrix_eqn( CB, HW/num_HW_blocks, &ld, out_dt, my_eqn10 );                         /* y = [HW, CB] */
#endif

#ifdef REFACTORED_BWD
#else

  /* Group norm equations */
  /* Create MatEq for bwd layernorm */

  ld = CB;
  tmp_ld2 = 1;

  /* dgamma function  */
  my_eqn11 = libxsmm_matrix_eqn_create();                                                       /* dgamma = ((inp *a + b) * dout) + dgamma */
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn11, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32);                   /* dgamma = ((inp *a + b) * dout) + dgamma */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn11, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);   /* [HW, CB] -> [CB] */
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn11, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32);                   /* ((inp *a + b) * dout) */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn11, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32);
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, CB, HW/num_HW_blocks, ld, 0, 0, in_dt );                        /* inp [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, CB, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 );           /* a [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, CB, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );           /* b [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, CB, HW/num_HW_blocks, ld, 3, 0, in_dt );                        /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, CB, 1, 1, 4, 0, LIBXSMM_DATATYPE_F32 );           /* dgamma [CB] */
  func11 = libxsmm_dispatch_matrix_eqn( CB, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn11 );      /* dgamma [CB] */

  /* dbeta function  */
  my_eqn12 = libxsmm_matrix_eqn_create();                                                       /* dbeta [CB] = dout [HW, CB] + dbeta [CB] */
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn12, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );      /* dbeta_tmp [HW, CB] */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn12, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);  /* [HW, CB] -> [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn12, CB, HW/num_HW_blocks, ld, 3, 0, in_dt );                        /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn12, CB, 1, 1, 5, 0, LIBXSMM_DATATYPE_F32 );           /* dbeta [CB] */
  func12 = libxsmm_dispatch_matrix_eqn( CB, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn12 );      /* dbeta [CB] */

  /* db new equation */
  my_eqn13 = libxsmm_matrix_eqn_create();                                                       /* db [CB] = (dout * gamma) [HW, CB] + db [CB]*/
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn13, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );                  /* db [CB] */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn13, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);   /* [HW, CB] -> [CB] */
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn13, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn13, CB, HW/num_HW_blocks, ld, 3, 0, in_dt );                        /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn13, CB, 1, 1, 6, 0, in_dt );                          /* gamma [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn13, CB, 1, 1, 9, 0, LIBXSMM_DATATYPE_F32 );           /* db [CB] */
  func13 = libxsmm_dispatch_matrix_eqn( CB, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn13 );      /* db [CB] */

  /* ds new equation */
  my_eqn14 = libxsmm_matrix_eqn_create();                                                       /* ds [CB] = ((dout * gamma) * inp) [HW, CB] + ds [CB] */
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn14, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );                  /* ds [CB] */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn14, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);   /* [HW, CB] -> [CB] */
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn14, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn14, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1, LIBXSMM_DATATYPE_F32 );       /*(dout * gamma)*/
  libxsmm_matrix_eqn_push_back_arg( my_eqn14, CB, HW/num_HW_blocks, ld, 3, 0, in_dt );                        /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn14, CB, 1, 1, 6, 0, in_dt );                          /* gamma [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn14, CB, HW/num_HW_blocks, ld, 0, 0, in_dt );                        /* inp [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn14, CB, 1, 1, 8, 0, LIBXSMM_DATATYPE_F32 );           /* ds [CB] */
  func14 = libxsmm_dispatch_matrix_eqn( CB, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn14 );      /* ds [CB] */

  /* din equation */
  my_eqn15 = libxsmm_matrix_eqn_create();                                                       /* din = ((gamma * a) * dout) + (inp * b + c) */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn15, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32);
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn15, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, CB, 1, 1, 6, 0, in_dt );                          /* gamma [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, CB, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 );           /* a [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, CB, HW/num_HW_blocks, ld, 3, 0, in_dt );                        /* dout [HW, CB] */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn15, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32);
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, CB, HW/num_HW_blocks, ld, 0, 0, in_dt );                        /* inp [HW, CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, CB, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );           /* b [CB] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn15, CB, 1, 1, 7, 0, LIBXSMM_DATATYPE_F32 );           /* c [CB] */
  func15 = libxsmm_dispatch_matrix_eqn( CB, HW/num_HW_blocks, &ld, in_dt, my_eqn15 );                         /* din [HW, CB] */

#endif /* for REFACTORED_BWD */

#ifdef REFACTORED_FWD
  my_gn_fwd = setup_my_gn_fwd(N, C, G, H, W, bc, nThreads, fuse_type );
#endif
#ifdef REFACTORED_BWD
  my_gn_bwd = setup_my_gn_bwd(N, C, G, H, W, bc, nThreads, fuse_type );
#endif

#if defined(REFACTORED_BWD) || defined(REFACTORED_FWD)
  /* allocate and bind scratch */
  if ( my_gn_fwd.scratch_size > 0 || my_gn_bwd.scratch_size > 0 ) {
    size_t alloc_size = LIBXSMM_MAX( my_gn_fwd.scratch_size, my_gn_bwd.scratch_size);
    scratch = libxsmm_aligned_malloc( alloc_size, 2097152 );
    init_buf( (float*)(scratch), (alloc_size)/4, 0, 0 );
  }
#endif

  /* Check correctness */
  if (LIBXSMM_NEQ(0, check)) {
#ifdef REFACTORED_FWD
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_gn_fwd_exec( my_gn_fwd, inp, gamma, beta, mean, var, eqn_out, eps, 0, tid, scratch);
    }
#else
    tpp_groupnorm_fwd_fp32(N, CP, HW, bc, G, num_HW_blocks, inp, gamma, beta, mean, var, eqn_out, func10, reduce_HW_kernel, reduce_rows_kernel, reduce_groups_kernel,
                            all_zero_G_kernel, all_zero_kernel, add_kernel, eps);
#endif /* for #ifdef-else REFACTORED_FWD */

#ifdef REFACTORED_FWD
    tensor_copy_NCHWc_to_NCHW (inp, naive_inp, N, CP*bc, HW, 1, bc);

//LIBXSMM_INLINE void naive_fusedgroupnorm_fp(naive_fusedgroupnorm_t* param, const float* input_ptr, float* output_ptr, const float* input_add_ptr,
//                                     const float* beta_ptr, const float* gamma_ptr, float* expectval_ptr, float* rcpstddev_ptr, float* variance_ptr)

    naive_fusedgroupnorm_fp(&naive_param, naive_inp, naive_out, naive_zeros /*cannot pass NULL or &dummy due to VLA_ACCESS but should be unused when fuse = 0 const float* input_add_ptr*/,
                                        beta, gamma, eps, mean, naive_rcpstdev, var);

    tensor_copy_NCHW_to_NCHWc (naive_out, out, N, CP*bc, HW, 1, bc);

#ifdef COMPUTE_FP64_REFERENCE
    extend_buf_fp32_to_fp64 (naive_inp, naive_inp_fp64, N*CP*bc*HW);

    naive_fusedgroupnorm_fp_fp64(&naive_param, naive_inp_fp64, naive_out_fp64, naive_zeros_fp64 /*cannot pass NULL or &dummy due to VLA_ACCESS but should be unused when fuse = 0 const float* input_add_ptr*/,
                                        beta_fp64, gamma_fp64, eps, mean_fp64, naive_rcpstdev_fp64, var_fp64);

    truncate_buf_fp64_to_fp32 (naive_out_fp64, naive_out_fp64_downscaled_to_fp32, N*CP*bc*HW);

    tensor_copy_NCHW_to_NCHWc (naive_out_fp64_downscaled_to_fp32, out_fp64_downscaled_to_fp32, N, CP*bc, HW, 1, bc);
#endif

#else /* for REFACTORED_FWD */
    scaler_groupnorm_fwd_fp32(N, CP, HW, bc, G, inp, gamma, beta, mean, var, out, eps);
#endif /* for #ifdef REFACTORED_FWD */

    /* compare */
    printf("############################################\n");
    printf("# Correctness FP32 FWD groupnorm - Output  #\n");
    printf("############################################\n");
    libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, N*CP*HW*bc, 1, out, eqn_out, 0, 0);
    printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
    printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_fwd.normf_rel);

#ifdef COMPUTE_FP64_REFERENCE
    printf("##################################################\n");
    printf("# Correctness FP32 FWD groupnorm - Output (fp64) #\n");
    printf("##################################################\n");
    libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, N*CP*HW*bc, 1, out_fp64_downscaled_to_fp32, eqn_out, 0, 0);
    printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
    printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_fwd.normf_rel);
#endif
  } /* checking correctness for FWD */

  for (i = 0; i < 1024 * 1024; i++ ) {
    sum += cache_fl[i];
  }
#ifdef REFACTORED_FWD
  naive_fusedgroupnorm_fp(&naive_param, naive_inp, naive_out, naive_zeros /*cannot pass NULL or &dummy due to VLA_ACCESS but should be unused when fuse = 0 const float* input_add_ptr*/,
                                        beta, gamma, eps, mean, naive_rcpstdev, var);
#else
  scaler_groupnorm_fwd_fp32(N, CP, HW, bc, G, inp, gamma, beta, mean, var, out, eps);
#endif
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
#ifdef REFACTORED_FWD
    naive_fusedgroupnorm_fp(&naive_param, naive_inp, naive_out, naive_zeros /*cannot pass NULL or &dummy due to VLA_ACCESS but should be unused when fuse = 0 const float* input_add_ptr*/,
                                        beta, gamma, eps, mean, naive_rcpstdev, var);
#else
    scaler_groupnorm_fwd_fp32(N, CP, HW, bc, G, inp, gamma, beta, mean, var, out, eps);
#endif
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Scaler groupnorm time FWD  = %.5g\n", ((double)(l_total)));
  for (i = 0; i < 1024 * 1024; i++ ) {
    sum += cache_fl[i] + (float)l_total;
  }
#ifdef REFACTORED_FWD
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_gn_fwd_exec( my_gn_fwd, inp, gamma, beta, mean, var, eqn_out, eps, 0, tid, scratch);
    }
#else
  //tpp_groupnorm_fwd_fp32(N, CP, HW, bc, G, num_HW_blocks, inp, gamma, beta, mean, var, eqn_out, eps, func10, reduce_HW_kernel, all_zero_kernel, add_kernel, copy_kernel);
  tpp_groupnorm_fwd_fp32(N, CP, HW, bc, G, num_HW_blocks, inp, gamma, beta, mean, var, eqn_out, func10, reduce_HW_kernel, reduce_rows_kernel, reduce_groups_kernel,
                            all_zero_G_kernel, all_zero_kernel, add_kernel, eps);
#endif
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
#ifdef REFACTORED_FWD
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_gn_fwd_exec( my_gn_fwd, inp, gamma, beta, mean, var, eqn_out, eps, 0, tid, scratch );
    }
#else
    //tpp_groupnorm_fwd_fp32(N, CP, HW, bc, G, num_HW_blocks, inp, gamma, beta, mean, var, eqn_out, eps, func10, reduce_HW_kernel, all_zero_kernel, add_kernel, copy_kernel);
    tpp_groupnorm_fwd_fp32(N, CP, HW, bc, G, num_HW_blocks, inp, gamma, beta, mean, var, eqn_out, func10, reduce_HW_kernel, reduce_rows_kernel, reduce_groups_kernel,
                            all_zero_G_kernel, all_zero_kernel, add_kernel, eps);
#endif
  }
  l_end = libxsmm_timer_tick();
  l_total2 = libxsmm_timer_duration(l_start, l_end);
  printf("TPP groupnorm time FWD  = %.5g\n", ((double)(l_total2)));
  printf("Speedup FWD is %.5g\n", l_total/l_total2);

  if (LIBXSMM_NEQ(0, check)) {
#ifdef REFACTORED_BWD
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif

      my_gn_bwd_exec( my_gn_bwd, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, eps, 0, tid, scratch );
    }
#else
    tpp_groupnorm_bwd_fp32(NP, CP, HW, CB, G, num_HW_blocks, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta,
                            func11, func12, func13, func14, func15, all_zero_kernel, add_kernel, eps);
#endif

#ifdef REFACTORED_BWD
    tensor_copy_NCHWc_to_NCHW (inp,  naive_inp,   N, CP*bc, HW, 1, bc);
    tensor_copy_NCHWc_to_NCHW (out,  naive_out,   N, CP*bc, HW, 1, bc);
    tensor_copy_NCHWc_to_NCHW (dout, naive_dout,  N, CP*bc, HW, 1, bc);


//LIBXSMM_INLINE void naive_fusedgroupnorm_bp(naive_fusedgroupnorm_t* param, const float* input_ptr, float* dinput_ptr, const float* output_ptr, float* doutput_ptr, float* dinput_add_ptr,
//                                     const float* beta_ptr, float* del_beta_ptr, const float* gamma_ptr, float* del_gamma_ptr,
//                                     const float* expectval_ptr, const float* rcpstddev_ptr, const float* variance_ptr)


    naive_fusedgroupnorm_bp(&naive_param, naive_inp, naive_dinp, naive_out, naive_dout, naive_zeros /*cannot pass NULL or &dummy due to VLA_ACCESS but should be unsued when fuse = 0 const float* dinput_add_ptr*/,
                                       beta, dbeta, gamma, dgamma, mean, naive_rcpstdev, var);

    tensor_copy_NCHW_to_NCHWc (naive_dinp  , dinp  ,  N, CP*bc, HW, 1, bc);

#ifdef COMPUTE_FP64_REFERENCE
    extend_buf_fp32_to_fp64 (naive_inp,  naive_inp_fp64,  N*CP*bc*HW);
    extend_buf_fp32_to_fp64 (naive_out,  naive_out_fp64,  N*CP*bc*HW);
    extend_buf_fp32_to_fp64 (naive_dout, naive_dout_fp64, N*CP*bc*HW);

    naive_fusedgroupnorm_bp_fp64(&naive_param, naive_inp_fp64, naive_dinp_fp64, naive_out_fp64, naive_dout_fp64, naive_zeros_fp64 /*cannot pass NULL or &dummy due to VLA_ACCESS but should be unsued when fuse = 0 const float* dinput_add_ptr*/,
                                       beta_fp64, dbeta_fp64, gamma_fp64, dgamma_fp64, mean_fp64, naive_rcpstdev_fp64, var_fp64);

    truncate_buf_fp64_to_fp32 (naive_dinp_fp64,   naive_dinp_fp64_downscaled_to_fp32, N*CP*bc*HW);
    truncate_buf_fp64_to_fp32 (dgamma_fp64, dgamma_fp64_downscaled_to_fp32, CP*bc);
    truncate_buf_fp64_to_fp32 (dbeta_fp64,  dbeta_fp64_downscaled_to_fp32, CP*bc);

    tensor_copy_NCHW_to_NCHWc (naive_dinp_fp64_downscaled_to_fp32, dinp_fp64_downscaled_to_fp32,  N, CP*bc, HW, 1, bc);
#endif

#else /* for REFACTORED_BWD */
    scaler_groupnorm_bwd_fp32(N, CP, HW, bc, G, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
#endif

    /* compare */
    printf("############################################\n");
    printf("# Correctness FP32 BWD groupnorm - Dinput  #\n");
    printf("############################################\n");
    libxsmm_matdiff(&norms_bwd_d, LIBXSMM_DATATYPE_F32, N*CP*HW*bc, 1, dinp, eqn_dinp, 0, 0);
    printf("L1 reference  : %.25g\n", norms_bwd_d.l1_ref);
    printf("L1 test       : %.25g\n", norms_bwd_d.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd_d.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd_d.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd_d.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd_d.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_bwd_d.normf_rel);

#ifdef COMPUTE_FP64_REFERENCE
    printf("##################################################\n");
    printf("# Correctness FP32 BWD groupnorm - Dinput (fp64) #\n");
    printf("##################################################\n");
    libxsmm_matdiff(&norms_bwd_d, LIBXSMM_DATATYPE_F32, N*CP*HW*bc, 1, dinp_fp64_downscaled_to_fp32, eqn_dinp, 0, 0);
    printf("L1 reference  : %.25g\n", norms_bwd_d.l1_ref);
    printf("L1 test       : %.25g\n", norms_bwd_d.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd_d.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd_d.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd_d.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd_d.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_bwd_d.normf_rel);
#endif

    printf("###########################################\n");
    printf("# Correctness FP32 BWD groupnorm - Dbeta  #\n");
    printf("###########################################\n");
    libxsmm_matdiff(&norms_bwd_beta, LIBXSMM_DATATYPE_F32, CP*bc, 1, dbeta, eqn_dbeta, 0, 0);
    printf("L1 reference  : %.25g\n", norms_bwd_beta.l1_ref);
    printf("L1 test       : %.25g\n", norms_bwd_beta.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd_beta.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd_beta.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd_beta.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd_beta.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_bwd_beta.normf_rel);

#ifdef COMPUTE_FP64_REFERENCE
    printf("##################################################\n");
    printf("# Correctness FP32 BWD groupnorm - Dbeta (fp64)  #\n");
    printf("##################################################\n");
    libxsmm_matdiff(&norms_bwd_beta, LIBXSMM_DATATYPE_F32, CP*bc, 1, dbeta_fp64_downscaled_to_fp32, eqn_dbeta, 0, 0);
    printf("L1 reference  : %.25g\n", norms_bwd_beta.l1_ref);
    printf("L1 test       : %.25g\n", norms_bwd_beta.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd_beta.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd_beta.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd_beta.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd_beta.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_bwd_beta.normf_rel);
#endif

    printf("############################################\n");
    printf("# Correctness FP32 BWD groupnorm - Dgamma  #\n");
    printf("############################################\n");
    libxsmm_matdiff(&norms_bwd_gamma, LIBXSMM_DATATYPE_F32, CP*bc, 1, dgamma, eqn_dgamma, 0, 0);
    printf("L1 reference  : %.25g\n", norms_bwd_gamma.l1_ref);
    printf("L1 test       : %.25g\n", norms_bwd_gamma.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd_gamma.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd_gamma.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd_gamma.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd_gamma.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_bwd_gamma.normf_rel);

#ifdef COMPUTE_FP64_REFERENCE
    printf("##################################################\n");
    printf("# Correctness FP32 BWD groupnorm - Dgamma (fp64) #\n");
    printf("##################################################\n");
    libxsmm_matdiff(&norms_bwd_gamma, LIBXSMM_DATATYPE_F32, CP*bc, 1, dgamma_fp64_downscaled_to_fp32, eqn_dgamma, 0, 0);
    printf("L1 reference  : %.25g\n", norms_bwd_gamma.l1_ref);
    printf("L1 test       : %.25g\n", norms_bwd_gamma.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd_gamma.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd_gamma.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd_gamma.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd_gamma.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_bwd_gamma.normf_rel);
#endif
  } /* correctness for BWD */

  for (i = 0; i < 1024 * 1024; i++ ) {
    sum += cache_fl[i];
  }
#ifdef REFACTORED_BWD
  naive_fusedgroupnorm_bp(&naive_param, naive_inp, naive_dinp, naive_out, naive_dout, naive_zeros /*cannot pass NULL or &dummy due to VLA_ACCESS but should be unsued when fuse = 0 const float* dinput_add_ptr*/,
                                       beta, dbeta, gamma, dgamma, mean, naive_rcpstdev, var);
#else
  scaler_groupnorm_bwd_fp32(N, CP, HW, bc, G, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
#endif
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
#ifdef REFACTORED_BWD
    naive_fusedgroupnorm_bp(&naive_param, naive_inp, naive_dinp, naive_out, naive_dout, naive_zeros /*cannot pass NULL or &dummy due to VLA_ACCESS but should be unsued when fuse = 0 const float* dinput_add_ptr*/,
                                       beta, dbeta, gamma, dgamma, mean, naive_rcpstdev, var);
#else
    scaler_groupnorm_bwd_fp32(N, CP, HW, bc, G, dout, inp, mean, var, gamma, dinp, dgamma, dbeta, eps);
#endif
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Scaler groupnorm time BWD = %.5g\n", ((double)(l_total)));
  for (i = 0; i < 1024 * 1024; i++ ) {
    sum += cache_fl[i] + (float)l_total;
  }
#ifdef REFACTORED_BWD
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_gn_bwd_exec( my_gn_bwd, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, eps, 0, tid, scratch );
    }
#else
  tpp_groupnorm_bwd_fp32(NP, CP, HW, CB, G, num_HW_blocks, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta,
                          func11, func12, func13, func14, func15, all_zero_kernel, add_kernel, eps);
#endif
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
#ifdef REFACTORED_BWD
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_gn_bwd_exec( my_gn_bwd, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, eps, 0, tid, scratch );
    }
#else
    tpp_groupnorm_bwd_fp32(NP, CP, HW, CB, G, num_HW_blocks, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta,
                            func11, func12, func13, func14, func15, all_zero_kernel, add_kernel, eps);
#endif
  }
  l_end = libxsmm_timer_tick();
  l_total2 = libxsmm_timer_duration(l_start, l_end);
  printf("TPP groupnorm time BWD = %.5g\n", ((double)(l_total2)));
  printf("Speedup BWD is %.5g\n", l_total/l_total2);
  /* printf("Running sum is %.5f\n", sum); */

  t_tpp += l_total2;
  t_vec += l_total;

  printf("\n\n=================================\n");
  printf("Total Speedup via TPP Matrix equation is %.5g\n", t_vec/t_tpp);
  printf("=================================\n");

  /* deallocate data */
#if defined(REFACTORED_BWD) || defined(REFACTORED_FWD)
  if ( scratch != NULL ) {
    libxsmm_free(scratch);
  }
#endif
  libxsmm_free(inp);
  libxsmm_free(out);
  libxsmm_free(dinp);
  libxsmm_free(dout);
  libxsmm_free(eqn_dinp);
  libxsmm_free(eqn_dout);
  libxsmm_free(dgamma);
  libxsmm_free(dbeta);
  libxsmm_free(eqn_dgamma);
  libxsmm_free(eqn_dbeta);
  libxsmm_free(mean);
  libxsmm_free(var);
  libxsmm_free(gamma);
  libxsmm_free(beta);
  libxsmm_free(eqn_out);
  libxsmm_free(cache_fl);
#if defined(REFACTORED_BWD) || defined(REFACTORED_FWD)
  libxsmm_free(naive_inp);
  libxsmm_free(naive_out);
  libxsmm_free(naive_dinp);
  libxsmm_free(naive_dout);
  libxsmm_free(naive_dgamma);
  libxsmm_free(naive_dbeta);
  libxsmm_free(naive_rcpstdev);
  libxsmm_free(naive_zeros);
#endif

#ifdef COMPUTE_FP64_REFERENCE
  libxsmm_free(naive_inp_fp64);
  libxsmm_free(naive_out_fp64);
  libxsmm_free(naive_rcpstdev_fp64);
  libxsmm_free(naive_zeros_fp64);
  libxsmm_free(naive_dinp_fp64);
  libxsmm_free(naive_dout_fp64);
  libxsmm_free(naive_dbeta_fp64);
  libxsmm_free(naive_dgamma_fp64);

  libxsmm_free(beta_fp64);
  libxsmm_free(gamma_fp64);
  libxsmm_free(mean_fp64);
  libxsmm_free(var_fp64);

  libxsmm_free(dbeta_fp64);
  libxsmm_free(dgamma_fp64);

  libxsmm_free(naive_out_fp64_downscaled_to_fp32);
  libxsmm_free(out_fp64_downscaled_to_fp32);
  libxsmm_free(naive_dinp_fp64_downscaled_to_fp32);
  libxsmm_free(dinp_fp64_downscaled_to_fp32);
  libxsmm_free(dgamma_fp64_downscaled_to_fp32);
  libxsmm_free(dbeta_fp64_downscaled_to_fp32);
#endif

  return 0;
}

/* copied implementation from the old sample */
#if !defined(REFACTORED_BWD) || !defined(REFACTORED_FWD)

void tpp_groupnorm_fwd_fp32(long NP, long CP, long HW, long CB, long G, long num_HW_blocks, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout,
                            libxsmm_matrix_eqn_function func10, libxsmm_meltwfunction_unary reduce_HW_kernel, libxsmm_meltwfunction_unary reduce_rows_kernel,
                            libxsmm_meltwfunction_unary reduce_groups_kernel, libxsmm_meltwfunction_unary all_zero_G_kernel, libxsmm_meltwfunction_unary all_zero_kernel, libxsmm_meltwfunction_binary add_kernel, float eps) {


  LIBXSMM_VLA_DECL(4, float, inp, pinp, CP, HW, CB);            /* [NP, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, float, out, pout, CP, HW, CB);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);                /* [CP,CB] */
  LIBXSMM_VLA_DECL(2, float, beta, pbeta, CB);                  /* [CP,CB] */

  int np, group_size;
  group_size = (CP*CB)/G;

  if (group_size <= CB){
    int cp;
    #pragma omp parallel for collapse(2)
    for(np = 0; np < NP; np++){
      for (cp = 0; cp < CP; cp++){
        LIBXSMM_ALIGNED(float tmp[2*CB], 64);
        LIBXSMM_ALIGNED(float sum_X[G], 64);
        LIBXSMM_ALIGNED(float sum_X2[G], 64);
        LIBXSMM_ALIGNED(float s[CB], 64);
        LIBXSMM_ALIGNED(float b[CB], 64);

        int i, j, hwb, g;
        libxsmm_matrix_eqn_param eqn_param;
        libxsmm_meltw_unary_param m_reduce_groups_params, v_reduce_groups_params, reduce_HW_params;
        libxsmm_meltw_unary_param all_zero_param;
        libxsmm_meltw_binary_param add_param;
        libxsmm_matrix_arg arg_array[5];

        all_zero_param.out.primary = tmp;
        all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = &tmp[CB];
        all_zero_kernel(&all_zero_param);

        all_zero_param.out.primary = sum_X;
        all_zero_G_kernel(&all_zero_param);
        all_zero_param.out.primary = sum_X2;
        all_zero_G_kernel(&all_zero_param);

/***************************  Process entire block code *****************************/
        LIBXSMM_ALIGNED(float new_tmp[2*CB], 64);
        reduce_HW_params.out.primary   = new_tmp;                  /* [2*CB] */
        for(hwb=0; hwb < num_HW_blocks; hwb++){
          reduce_HW_params.in.primary    = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);      /* [HW_block, CB] -----> [2 * CB] */
          reduce_HW_kernel(&reduce_HW_params);

          add_param.in0.primary = tmp;
          add_param.in1.primary = new_tmp;
          add_param.out.primary = tmp;
          add_kernel(&add_param);

          add_param.in0.primary = &tmp[CB];
          add_param.in1.primary = &new_tmp[CB];
          add_param.out.primary = &tmp[CB];
          add_kernel(&add_param);
          /* for (cb = 0; cb < 2*CB; cb++) { */
          /*   tmp[cb] += new_tmp[cb]; */
          /* } */
        }

        for(i=0; i < CB; i += group_size){
          g = (cp*CB + i)/group_size;                                                                      /* determine current group */
          m_reduce_groups_params.in.primary    = &tmp[i];
          m_reduce_groups_params.out.primary   = &sum_X[g];
          v_reduce_groups_params.in.primary    = &tmp[CB + i];
          v_reduce_groups_params.out.primary   = &sum_X2[g];
          reduce_groups_kernel(&m_reduce_groups_params);
          reduce_groups_kernel(&v_reduce_groups_params);

          mean[np*G + g] = sum_X[g] / ((float)group_size * HW);
          var[np*G + g] = (sum_X2[g] / ((float)group_size * HW)) - (mean[np*G + g]*mean[np*G + g]);        /* var = E[X^2] - (E[X])^2 */

          for(j = 0; j < group_size; j++){
            s[i + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));                                          /* 1/sqrt(var(X) + eps) */
            b[i + j] = -1 * mean[np*G + g] * s[i + j];                                                     /* -E[X]/sqrt(var(X) + eps) */
          }
        }

        arg_array[1].primary = s;                                                                           /* [CB] */
        arg_array[2].primary = b;                                                                           /* [CB] */
        arg_array[3].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);                                    /* [CB] */
        arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, beta, cp, 0, CB);                                     /* [CB] */

        for(hwb=0; hwb < num_HW_blocks; hwb++){
          arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);            /* [HW, CB] */
          eqn_param.inputs = arg_array;
          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, out, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);        /* [HW, CB] */
          func10(&eqn_param);                                                                                           /* Normalization equation -> y = ((s*x + b)*gamma + beta) */
        }
      }
    }
  }
  else{                                                         /* Case when group_size > CB */
    #pragma omp parallel for
    for(np = 0; np < NP; np++){

      LIBXSMM_ALIGNED(float tmp[2*CB], 64);
      LIBXSMM_ALIGNED(float sum_X[G], 64);
      LIBXSMM_ALIGNED(float sum_X2[G], 64);
      LIBXSMM_ALIGNED(float s[CP*CB], 64);
      LIBXSMM_ALIGNED(float b[CP*CB], 64);

      int i, j, cp, hwb, g;
      float m, v;
      libxsmm_matrix_eqn_param eqn_param;
      libxsmm_meltw_unary_param m_reduce_rows_params, v_reduce_rows_params, m_reduce_groups_params, v_reduce_groups_params, reduce_HW_params;
      libxsmm_meltw_unary_param all_zero_param;
      libxsmm_meltw_binary_param add_param;
      libxsmm_matrix_arg arg_array[5];

      all_zero_param.out.primary = sum_X;
      all_zero_G_kernel(&all_zero_param);
      all_zero_param.out.primary = sum_X2;
      all_zero_G_kernel(&all_zero_param);

      LIBXSMM_ALIGNED(float new_tmp[2*CB], 64);
      for (cp = 0; cp < CP; cp++){                      /* [cp, HW, CB] */
        all_zero_param.out.primary = tmp;
        all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = &tmp[CB];
        all_zero_kernel(&all_zero_param);
        /* for (cb = 0; cb < 2*CB; cb++) { */
        /*   tmp[cb] = 0.0f; */
        /* } */

        reduce_HW_params.out.primary   = new_tmp;                  /* [2*CB] */
        for(hwb=0; hwb < num_HW_blocks; hwb++){
          reduce_HW_params.in.primary    = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);      /* [HW, CB] -----> [2 * CB] */
          reduce_HW_kernel(&reduce_HW_params);

          add_param.in0.primary = tmp;
          add_param.in1.primary = new_tmp;
          add_param.out.primary = tmp;
          add_kernel(&add_param);

          add_param.in0.primary = &tmp[CB];
          add_param.in1.primary = &new_tmp[CB];
          add_param.out.primary = &tmp[CB];
          add_kernel(&add_param);
          /* #pragma omp simd */
          /* for (cb = 0; cb < 2*CB; cb++) { */
          /*   tmp[cb] += new_tmp[cb]; */
          /* } */
        }

        if (group_size >= CB){                                 /* Group size >= block size  (Ex.- CP = 4, CB = 16, G = 2, group_size = 32) */
          g = (cp*CB)/group_size;                              /* determine current group */
          m_reduce_rows_params.in.primary    = tmp;
          m_reduce_rows_params.out.primary   = &m;
          v_reduce_rows_params.in.primary    = &tmp[CB];
          v_reduce_rows_params.out.primary   = &v;
          reduce_rows_kernel(&m_reduce_rows_params);
          reduce_rows_kernel(&v_reduce_rows_params);
          sum_X[g] += m;
          sum_X2[g] += v;
        }
        else{                                                 /* Group size < block size  (Ex.- CP = 4, CB = 16, G = 32, group_size = 2) */
          for(i=0; i < CB; i += group_size){
            m_reduce_groups_params.in.primary    = &tmp[i];
            m_reduce_groups_params.out.primary   = &sum_X[cp*(CB/group_size) + (i/group_size)];
            v_reduce_groups_params.in.primary    = &tmp[CB + i];
            v_reduce_groups_params.out.primary   = &sum_X2[cp*(CB/group_size) + (i/group_size)];
            reduce_groups_kernel(&m_reduce_groups_params);
            reduce_groups_kernel(&v_reduce_groups_params);
          }
        }
      }

      for(g = 0; g < G; g++){                                                  /* mean and variance calculation */
        mean[np*G + g] = sum_X[g] / ((float)group_size * HW);
        var[np*G + g] = (sum_X2[g] / ((float)group_size * HW)) - (mean[np*G + g]*mean[np*G + g]);        /* var = E[X^2] - (E[X])^2 */

        for(j = 0; j < group_size; j++){
          s[g*group_size + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));                               /* 1/sqrt(var(X) + eps) */
          b[g*group_size + j] = -1 * mean[np*G + g] * s[g*group_size + j];                               /* -E[X]/sqrt(var(X) + eps) */
        }
      }

      for (cp = 0; cp < CP; cp++){

        arg_array[1].primary = &s[cp*CB];                                                                   /* [CB] */
        arg_array[2].primary = &b[cp*CB];                                                                   /* [CB] */
        arg_array[3].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);                                    /* [CB] */
        arg_array[4].primary = &LIBXSMM_VLA_ACCESS(2, beta, cp, 0, CB);                                     /* [CB] */

        for(hwb=0; hwb < num_HW_blocks; hwb++){
          arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);                       /* [HW, CB] */
          eqn_param.inputs = arg_array;
          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, out, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);                   /* [HW,CB] */
          func10(&eqn_param);                                                                                 /* Normalization equation -> y = ((s*x + b)*gamma + beta) */
        }
      }
    }
  }
}

//    tpp_groupnorm_bwd_fp32(NP, CP, HW, CB, G, num_HW_blocks, eqn_dout, inp, mean, var, gamma, eqn_dinp, eqn_dgamma, eqn_dbeta, func11, func12, func13, func14, func15, all_zero_kernel, add_kernel, eps);

void tpp_groupnorm_bwd_fp32(long NP, long CP, long HW, long CB, long G, long num_HW_blocks, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta,
    libxsmm_matrix_eqn_function dgamma_func, libxsmm_matrix_eqn_function dbeta_func, libxsmm_matrix_eqn_function db_func, libxsmm_matrix_eqn_function ds_func, libxsmm_matrix_eqn_function din_func,
    libxsmm_meltwfunction_unary all_zero_kernel, libxsmm_meltwfunction_binary add_kernel, float eps) {

  int group_size;
  group_size = (CP*CB)/G;

  const float scale = 1.0f / ((float)group_size * HW);

  LIBXSMM_VLA_DECL(4, float, din, pdin, CP, HW, CB);
  LIBXSMM_VLA_DECL(4, float, inp, pinp, CP, HW, CB);
  LIBXSMM_VLA_DECL(4, float, dout, pdout, CP, HW, CB);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, CB);

  LIBXSMM_ALIGNED(float dgamma_NP[NP*CP*CB], 64);
  LIBXSMM_ALIGNED(float dbeta_NP[NP*CP*CB], 64);


  if (group_size <= CB){
    #pragma omp parallel
    {
      LIBXSMM_ALIGNED(float a[CB], 64);
      LIBXSMM_ALIGNED(float b[CB], 64);
      LIBXSMM_ALIGNED(float c[CB], 64);
      LIBXSMM_ALIGNED(float ds[CB], 64);
      LIBXSMM_ALIGNED(float db[CB], 64);

      int np, cp;
      #pragma omp for collapse(2)
      for (np = 0; np < NP; np++){
        for (cp = 0; cp < CP; cp++) {
          int j, g, hwb, lg;

          libxsmm_matrix_eqn_param eqn_param;
          libxsmm_meltw_unary_param all_zero_param;
          libxsmm_matrix_arg arg_array[10];
          eqn_param.inputs = arg_array;

          /* for(j = 0; j < CB; j++){
              dgamma_NP[np*CP*CB + cp*CB + j] = 0.0f;
              dbeta_NP[np*CP*CB + cp*CB + j] = 0.0f;
           } */

          all_zero_param.out.primary = &dgamma_NP[np*CP*CB + cp*CB];
          all_zero_kernel(&all_zero_param);
          all_zero_param.out.primary = &dbeta_NP[np*CP*CB + cp*CB];
          all_zero_kernel(&all_zero_param);
          all_zero_param.out.primary = ds;
          all_zero_kernel(&all_zero_param);
          all_zero_param.out.primary = db;
          all_zero_kernel(&all_zero_param);


          for(g = (cp*CB)/group_size; g < ((cp+1)*CB)/group_size; g++){                                                  /* compute a and b for each channel from group means and variance */
            lg = g - (cp*CB)/group_size;
            for(j = 0; j < group_size; j++){
              a[lg*group_size + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));
              b[lg*group_size + j] = -a[lg*group_size + j]*mean[np*G + g];
            }
          }

          arg_array[1].primary = a;
          arg_array[2].primary = b;
          arg_array[4].primary = &dgamma_NP[np*CP*CB + cp*CB];
          arg_array[5].primary = &dbeta_NP[np*CP*CB + cp*CB];
          arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
          arg_array[8].primary = ds;
          arg_array[9].primary = db;

          for(hwb=0; hwb < num_HW_blocks; hwb++){
            arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
            arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);

            eqn_param.output.primary = ds;
            ds_func(&eqn_param);

            eqn_param.output.primary = db;
            db_func(&eqn_param);

            eqn_param.output.primary = &dgamma_NP[np*CP*CB + cp*CB];
            dgamma_func(&eqn_param);

            eqn_param.output.primary = &dbeta_NP[np*CP*CB + cp*CB];
            dbeta_func(&eqn_param);
          }

          /* b = (db * mean[nb] - ds) * a * a * a * scale; */
          /* c = -b * mean[nb] - db * a * scale; */

          for(g = (cp*CB)/group_size; g < ((cp+1)*CB)/group_size; g++){                                                  /* compute b and c for each channel from group means and variance */
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
          arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
          arg_array[7].primary = c;

          for(hwb=0; hwb < num_HW_blocks; hwb++){
            arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
            arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
            eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, din, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
            din_func(&eqn_param);
          }
        }
      }

      #pragma omp for
      for (cp = 0; cp < CP; cp++) {
        for (np=0; np < NP; np++ ) {
          int cb;
          for(cb = 0; cb < CB; cb++){
            LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, CB) += dgamma_NP[np*CP*CB + cp*CB + cb];
            LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, CB) += dbeta_NP[np*CP*CB + cp*CB + cb];
          }
        }
      }
    }
  }
  else{
    #pragma omp parallel
    {
      LIBXSMM_ALIGNED(float a[CP*CB], 64);
      LIBXSMM_ALIGNED(float b[CP*CB], 64);
      LIBXSMM_ALIGNED(float c[CP*CB], 64);
      LIBXSMM_ALIGNED(float ds[CP*CB], 64);
      LIBXSMM_ALIGNED(float db[CP*CB], 64);
      int np;

      #pragma omp for
      for (np = 0; np < NP; np++) {
        int j, g, cp, hwb;

        libxsmm_matrix_eqn_param eqn_param;
        libxsmm_meltw_unary_param all_zero_param;
        libxsmm_matrix_arg arg_array[10];
        eqn_param.inputs = arg_array;

        /* for(j = 0; j < CP*CB; j++){ */
        /*   dgamma_NP[np*CP*CB + j] = 0.0f; */
        /*   dbeta_NP[np*CP*CB + j] = 0.0f; */
        /* } */

        for (cp = 0; cp < CP; cp++) {
          all_zero_param.out.primary = &dgamma_NP[np*CP*CB + cp*CB];
          all_zero_kernel(&all_zero_param);
          all_zero_param.out.primary = &dbeta_NP[np*CP*CB + cp*CB];
          all_zero_kernel(&all_zero_param);
          all_zero_param.out.primary = &ds[cp*CB];
          all_zero_kernel(&all_zero_param);
          all_zero_param.out.primary = &db[cp*CB];
          all_zero_kernel(&all_zero_param);
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
          arg_array[4].primary = &dgamma_NP[np*CP*CB + cp*CB];
          arg_array[5].primary = &dbeta_NP[np*CP*CB + cp*CB];
          arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
          arg_array[8].primary = &ds[cp*CB];
          arg_array[9].primary = &db[cp*CB];

          for(hwb=0; hwb < num_HW_blocks; hwb++){
            arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
            arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);

            eqn_param.output.primary = &ds[cp*CB];
            ds_func(&eqn_param);

            eqn_param.output.primary = &db[cp*CB];
            db_func(&eqn_param);

            eqn_param.output.primary = &dgamma_NP[np*CP*CB + cp*CB];
            dgamma_func(&eqn_param);

            eqn_param.output.primary = &dbeta_NP[np*CP*CB + cp*CB];
            dbeta_func(&eqn_param);
          }
        }

        /* b = (db * mean[nb] - ds) * a * a * a * scale; */
        /* c = -b * mean[nb] - db * a * scale; */

        for(g = 0; g < G; g++){                                                  /* compute b and c for each channel from group means and variance */
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
          arg_array[6].primary = &LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, CB);
          arg_array[7].primary = &c[cp*CB];

          for(hwb=0; hwb < num_HW_blocks; hwb++){
            arg_array[0].primary = &LIBXSMM_VLA_ACCESS(4, inp, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
            arg_array[3].primary = &LIBXSMM_VLA_ACCESS(4, dout, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
            eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, din, np, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, CB);
            din_func(&eqn_param);
          }
        }
      }

      int cp;
      #pragma omp for
      for (cp = 0; cp < CP; cp++) {
        for (np=0; np < NP; np++ ) {
          int cb;
          for(cb = 0; cb < CB; cb++){
            LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, CB) += dgamma_NP[np*CP*CB + cp*CB + cb];
            LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, CB) += dbeta_NP[np*CP*CB + cp*CB + cb];
          }
        }
      }
    }
  }
}

void scaler_groupnorm_fwd_fp32(long NP, long CP, long HW, long CB, long G, float *pinp, float *pgamma, float *pbeta, float *mean, float *var, float *pout, float eps){

  LIBXSMM_VLA_DECL(4, float, inp, pinp, CP, HW, CB);            /* [NP, CP, HW, CB] */
  LIBXSMM_VLA_DECL(4, float, out, pout, CP, HW, CB);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2, float, beta, pbeta, CB);

  int np, group_size;
  group_size = (CP*CB)/G;

  #pragma omp parallel for
  for(np = 0; np < NP; np++){

    LIBXSMM_ALIGNED(float sum_X[G], 64);
    LIBXSMM_ALIGNED(float sum_X2[G], 64);
    LIBXSMM_ALIGNED(float s[CP*CB], 64);
    LIBXSMM_ALIGNED(float b[CP*CB], 64);

    int i, j, cp, cb, hw, g;
    float m, v, value;

    for(g = 0; g < G; g++){
      sum_X[g] = 0.0f;
      sum_X2[g] = 0.0f;
    }
    for(cp = 0; cp < CP; cp++){                           /* Size = CP*HW*CB*4 */
      m = 0.0f;
      v = 0.0f;
      if (group_size >= CB){                                 /* Group size >= block size  (Ex.- CP = 4, CB = 16, G = 2, group_size = 32) */
        for(cb = 0; cb < CB; cb++){
          for(hw = 0; hw < HW; hw++){
            value = LIBXSMM_VLA_ACCESS(4, inp, np, cp, hw, cb, CP, HW, CB);
            m += value;
            v += (value*value);
          }
        }
        g = (cp*CB)/group_size;                              /* determine current group */
        sum_X[g] += m;
        sum_X2[g] += v;
      }
      else{
        for(i=0; i < CB; i += group_size){              /* Group size < block size  (Ex.- CP = 4, CB = 16, G = 32, group_size = 2) */
          for(j = 0; j < group_size; j++){
            for(hw = 0; hw < HW; hw++){
              value = LIBXSMM_VLA_ACCESS(4, inp, np, cp, hw, (i + j), CP, HW, CB);
              sum_X[cp*(CB/group_size) + (i/group_size)] += value;
              sum_X2[cp*(CB/group_size) + (i/group_size)] += (value*value);
            }
          }
        }
      }
    }

    for(g = 0; g < G; g++){                                                  /* mean and variance calculation */           /* Size = 2*CP*CB*4 */
      mean[np*G + g] = sum_X[g] / ((float)group_size * HW);
      var[np*G + g] = (sum_X2[g] / ((float)group_size * HW)) - (mean[np*G + g]*mean[np*G + g]);      /* var = E[X^2] - (E[X])^2        [G] */

      for(j = 0; j < group_size; j++){
        s[g*group_size + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));                               /* s = 1/sqrt(var(X) + eps)     [CP, CB] */
        b[g*group_size + j] = -1 * mean[np*G + g] * s[g*group_size + j];                               /* b = -E[X]/sqrt(var(X) + eps) [CP, CB] */
      }
    }

    for(cp = 0; cp < CP; cp++){                                                     /* Size = 2*CP*HW*CB*4 + 2*CP*CB*4 */
      for(cb = 0; cb < CB; cb++){
        for(hw = 0; hw < HW; hw++){
          value = LIBXSMM_VLA_ACCESS(4, inp, np, cp, hw, cb, CP, HW, CB);
          value = ((value * s[cp*CB + cb]) + b[cp*CB + cb]) * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB) + LIBXSMM_VLA_ACCESS(2, beta, cp, cb, CB);        /* Normalization equation -> y = ((s*x + b)*gamma + beta) */
          LIBXSMM_VLA_ACCESS(4, out, np, cp, hw, cb, CP, HW, CB) = value;
        }
      }
    }
  }                                         /*End multithreading loop*/
}

void scaler_groupnorm_bwd_fp32(long NP, long CP, long HW, long CB, long G, float *pdout, float *pinp, float *mean, float *var, float *pgamma, float *pdin, float *pdgamma, float *pdbeta, float eps) {

  int np, group_size;
  group_size = (CP*CB)/G;
  float scale = 1.0f / ((float)group_size * HW);

  LIBXSMM_VLA_DECL(4, float, din, pdin, CP, HW, CB);
  LIBXSMM_VLA_DECL(4, float, inp, pinp, CP, HW, CB);
  LIBXSMM_VLA_DECL(4, float, dout, pdout, CP, HW, CB);
  LIBXSMM_VLA_DECL(2, float, gamma, pgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, CB);
  LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, CB);

  LIBXSMM_ALIGNED(float dgamma_NP[NP*CP*CB], 64);
  LIBXSMM_ALIGNED(float dbeta_NP[NP*CP*CB], 64);

  #pragma omp parallel for
  for(np = 0; np < NP; np++){

    int j, cp, cb, hw, g;
    LIBXSMM_ALIGNED(float a[CP*CB], 64);
    LIBXSMM_ALIGNED(float b[CP*CB], 64);
    LIBXSMM_ALIGNED(float c[CP*CB], 64);
    LIBXSMM_ALIGNED(float ds[CP*CB], 64);
    LIBXSMM_ALIGNED(float db[CP*CB], 64);

    for(j = 0; j < CP*CB; j++){
      dgamma_NP[np*CP*CB + j] = 0.0f;
      dbeta_NP[np*CP*CB + j] = 0.0f;
    }

    for(g = 0; g < G; g++){                                                  /* compute a and b for each channel from group means and variance */
      for(j = 0; j < group_size; j++){
        a[g*group_size + j] = 1.0f / ((float)sqrt(var[np*G + g] + eps));
        b[g*group_size + j] = -a[g*group_size + j]*mean[np*G + g];
        ds[g*group_size + j] = 0.0f;
        db[g*group_size + j] = 0.0f;
      }
    }

    for (cp = 0; cp < CP; cp++) {                    /* dgamma += (a * inp + b) * dout , dbeta += dout, ds += dout * gamma * inp, db += dout * gamma */    /* Size = 2*CP*HW*CB*4 */
      for (cb = 0; cb < CB; cb++) {
        for (hw = 0; hw < HW; hw++){
          dgamma_NP[np*CP*CB + cp*CB + cb] += (a[cp*CB + cb] * LIBXSMM_VLA_ACCESS(4, inp, np, cp, hw, cb, CP, HW, CB) + b[cp*CB + cb]) * LIBXSMM_VLA_ACCESS(4, dout, np, cp, hw, cb, CP, HW, CB);
          dbeta_NP[np*CP*CB + cp*CB + cb] += LIBXSMM_VLA_ACCESS(4, dout, np, cp, hw, cb, CP, HW, CB);
          ds[cp*CB + cb] += LIBXSMM_VLA_ACCESS(4, dout, np, cp, hw, cb, CP, HW, CB) * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB) * LIBXSMM_VLA_ACCESS(4, inp, np, cp, hw, cb, CP, HW, CB);
          db[cp*CB + cb] += LIBXSMM_VLA_ACCESS(4, dout, np, cp, hw, cb, CP, HW, CB) * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB);
        }
      }
    }
    /* b = (db * mean[nb] - ds) * a * a * a * scale; */
    /* c = -b * mean[nb] - db * a * scale; */
    for(g = 0; g < G; g++){                                                  /* compute b and c for each channel from group means and variance */
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

    for (cp = 0; cp < CP; cp++) {                                                     /* din = dout * a * gamma + b * inp + c */  /* Size = 3*CP*HW*CB*4 */
      for (cb = 0; cb < CB; cb++) {
        for (hw = 0; hw < HW; hw++){
          LIBXSMM_VLA_ACCESS(4, din, np, cp, hw, cb, CP, HW, CB) = LIBXSMM_VLA_ACCESS(4, dout, np, cp, hw, cb, CP, HW, CB)  * a[cp*CB + cb] * LIBXSMM_VLA_ACCESS(2, gamma, cp, cb, CB) + b[cp*CB + cb] * LIBXSMM_VLA_ACCESS(4, inp, np, cp, hw, cb, CP, HW, CB) + c[cp*CB + cb];
        }
      }
    }
  }

  int cp;
  #pragma omp parallel for
  for (cp = 0; cp < CP; cp++) {
    for (np=0; np < NP; np++ ) {
      int cb;
      for(cb = 0; cb < CB; cb++){
        LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, CB) += dgamma_NP[np*CP*CB + cp*CB + cb];
        LIBXSMM_VLA_ACCESS(2, dbeta, cp, cb, CB) += dbeta_NP[np*CP*CB + cp*CB + cb];
      }
    }
  }
}



#endif

