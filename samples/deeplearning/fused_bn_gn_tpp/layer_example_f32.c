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

#define BITS_PER_CHAR (8)

#define NUM_HW_BLOCKS (16)

#define COMPUTE_FP64_REFERENCE

/* include c-based dnn library */
#ifdef CNN_HEADER
  #include "dnn_common.h"
#else /* LIBXSMM sample */
  #include "../common/dnn_common.h"
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

  libxsmm_blasint ldo = bc;
  libxsmm_blasint ld  = bc;
  libxsmm_blasint tmp_ld, tmp_ld2;
  libxsmm_blasint my_eqn10;

  libxsmm_meltw_unary_flags jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_meltw_unary_type  unary_type;

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
  res.all_zero_kernel = libxsmm_dispatch_meltw_unary(res.bc, 1, NULL, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  if ( res.all_zero_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd all_zero_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.copy_kernel = libxsmm_dispatch_meltw_unary(res.bc, 1, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
  if ( res.copy_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd copy_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.add_kernel = libxsmm_dispatch_meltw_binary(res.bc, 1, &ldo, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_MELTW_TYPE_BINARY_ADD);
  if ( res.add_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd add_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.relu_kernel = libxsmm_dispatch_meltw_unary(res.bc, res.H*res.W / res.num_HW_blocks, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
                                                 /*LIBXSMM_MELTW_FLAG_UNARY_NONE*/ LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT, LIBXSMM_MELTW_TYPE_UNARY_RELU);
  if ( res.relu_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP fwd_relu_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.ewise_add_kernel = libxsmm_dispatch_meltw_binary(res.bc, res.H*res.W / res.num_HW_blocks, &ldo, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
                                                       LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_MELTW_TYPE_BINARY_ADD);
  if ( res.ewise_add_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd ewise_add_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* TPPs for reducing X and X2 in HW*/
  tmp_ld = bc;

  unary_type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD;
  jit_reduce_flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS;
  res.reduce_HW_kernel = libxsmm_dispatch_meltw_unary(res.bc, res.H*res.W/res.num_HW_blocks, &ld, &tmp_ld, in_dt, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_reduce_flags, unary_type);
  if ( res.reduce_HW_kernel == NULL) {
    fprintf( stderr, "JIT for TPP fwd reduce_HW_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* TPP for scaling */
  ld = bc;
  tmp_ld = 1;
  tmp_ld2 = 1;

  my_eqn10 = libxsmm_matrix_eqn_create();                                                            /* y = (s*x + b)*gamma + beta */
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

void my_bn_fwd_exec( my_bn_fwd_config cfg, const float *pinp, const float *pinp_add, const float *pgamma, const float *pbeta, float *mean, float *var, float *pout, unsigned char *prelumask, float eps, int start_tid, int my_tid, void *scratch ) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint HW = cfg.H * cfg.W;
  const libxsmm_blasint bc = cfg.bc;
  const libxsmm_blasint num_HW_blocks = cfg.num_HW_blocks;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;

  /* number of tasks that could be run in parallel for 1d blocking */
  // Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here?
  const libxsmm_blasint work_dN = CP * N;
  /* compute chunk size */
  const libxsmm_blasint chunksize_dN = (work_dN % cfg.threads == 0) ?
    (work_dN / cfg.threads) : ((work_dN / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_dN = (ltid * chunksize_dN < work_dN) ? (ltid * chunksize_dN) : work_dN;
  const libxsmm_blasint thr_end_dN = ((ltid + 1) * chunksize_dN < work_dN) ? ((ltid + 1) * chunksize_dN) : work_dN;

  /* number of tasks that could be run in parallel for 1d blocking */
  // Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here?
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

  { /* stupid block to keep indentation */
    LIBXSMM_ALIGNED(float s[bc], 64);
    LIBXSMM_ALIGNED(float b[bc], 64);
    int n, cp;

    int cpxnt;
    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      { /* stupid block to keep indentation */
        n  = cpxnt%N;
        cp = cpxnt/N;

        int hwb;

        float *sum_ncp_ptr   = &LIBXSMM_VLA_ACCESS(3, sum_N, cp, n, 0, N, bc);
        float *sumsq_ncp_ptr = &LIBXSMM_VLA_ACCESS(3, sumsq_N, cp, n, 0, N, bc);

        libxsmm_meltw_unary_param all_zero_param;
        all_zero_param.out.primary = sum_ncp_ptr;
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = sumsq_ncp_ptr;
        cfg.all_zero_kernel(&all_zero_param);

        /* #pragma omp simd  */
        /* for (int cb = 0; cb < bc; cb++) {  */
        /*   sum_ncp_ptr[cb] = 0.0f;    */
        /*   sumsq_ncp_ptr[cb] = 0.0f;  */
        /* } */

        libxsmm_meltw_binary_param add_param;

        libxsmm_meltw_unary_param reduce_HW_params;       /*Private params and tmp array */
        LIBXSMM_ALIGNED(float lcl_sum_X_X2[2*bc], 64);
        reduce_HW_params.out.primary   = lcl_sum_X_X2;                                                         /* [2*bc]  */
        for(hwb=0; hwb < num_HW_blocks; hwb++){

          reduce_HW_params.in.primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
          cfg.reduce_HW_kernel(&reduce_HW_params);                                                       /* [HW, bc] -----> [2 * bc] */

          add_param.in0.primary = sum_ncp_ptr;
          add_param.in1.primary = lcl_sum_X_X2;
          add_param.out.primary = sum_ncp_ptr;
          cfg.add_kernel(&add_param);

          add_param.in0.primary = sumsq_ncp_ptr;
          add_param.in1.primary = &lcl_sum_X_X2[bc];
          add_param.out.primary = sumsq_ncp_ptr;
          cfg.add_kernel(&add_param);

          /* #pragma omp simd */
          /* for (int cb = 0; cb < bc; cb++) {  */
          /*   sum_ncp_ptr[cb] += lcl_sum_X_X2[cb];  */
          /*   sumsq_ncp_ptr[cb] += lcl_sum_X_X2[bc + cb];  */
          /* }  */
        }
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);

    for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {

      libxsmm_meltw_unary_param all_zero_param;
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, 0, CP, bc);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, 0, CP, bc);
      cfg.all_zero_kernel(&all_zero_param);

      /* #pragma omp simd */
      /* for (int cb = 0; cb < bc; cb++) {  */
      /*   sum_X_X2[cp*bc + cb] = 0.0f;   */
      /*   sum_X_X2[CP*bc + (cp*bc + cb)] = 0.0f;  */
      /* } */

      libxsmm_meltw_binary_param add_param;
      int cb, ni;
      for(ni = 0; ni < N; ni++){

        add_param.in0.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, 0, CP, bc);
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, sum_N, cp, ni, 0, N, bc);
        add_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, 0, CP, bc);
        cfg.add_kernel(&add_param);

        add_param.in0.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, 0, CP, bc);
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, sumsq_N, cp, ni, 0, N, bc);
        add_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, 0, CP, bc);
        cfg.add_kernel(&add_param);

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

    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      { /* stupid block to keep indentation */
        n  = cpxnt%N;
        cp = cpxnt/N;

        libxsmm_matrix_arg arg_array[5];                                                         /* private eqn args and params*/
        libxsmm_matrix_eqn_param eqn_param;
        int hwb, cb;

        for(cb = 0; cb < bc; cb++){
          s[cb] = 1.0f / ((float)sqrt(var[cp*bc + cb] + eps));                                 /* s = 1/sqrt(var(X) + eps)     [bc] */
          b[cb] = -1 * mean[cp*bc + cb] * s[cb];                                               /* b = -E[X]/sqrt(var(X) + eps) [bc] */
        }
        arg_array[1].primary = s;                                                              /* [bc] */
        arg_array[2].primary = b;                                                              /* [bc] */
        arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);                       /* [bc] */
        arg_array[4].primary = (void*)&LIBXSMM_VLA_ACCESS(2, beta,  cp, 0, bc);                       /* [bc] */

        for(hwb=0; hwb < num_HW_blocks; hwb++){

          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);           /* [HW, bc] */
          eqn_param.inputs = arg_array;
          eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(4, out, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);       /* [HW,bc] */
          cfg.func10(&eqn_param);                                                                    /* Normalization equation -> y = ((s*x + b)*gamma + beta) */

          /* Eltwise add */
          if (cfg.fuse_type == MY_NORMALIZE_FUSE_ELTWISE || cfg.fuse_type == MY_NORMALIZE_FUSE_ELTWISE_RELU_WITH_MASK) {
            libxsmm_meltw_binary_param add_param;
            add_param.in0.primary = (void*)&LIBXSMM_VLA_ACCESS(4, out,     n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
            add_param.in1.primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp_add, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
            add_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(4, out,     n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
            cfg.ewise_add_kernel(&add_param);
          }

          /* ReLU */
          if (cfg.fuse_type == MY_NORMALIZE_FUSE_RELU || cfg.fuse_type == MY_NORMALIZE_FUSE_RELU_WITH_MASK || cfg.fuse_type == MY_NORMALIZE_FUSE_ELTWISE_RELU_WITH_MASK) {
            libxsmm_meltw_unary_param all_relu_param;

            all_relu_param.op.primary   = (void*)(&alpha);
            all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(4, out, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);      /* [HW,bc] */
            all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(4, out, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);      /* [HW,bc] */
            all_relu_param.out.secondary = ((cfg.fuse_type == MY_NORMALIZE_FUSE_RELU_WITH_MASK || cfg.fuse_type == MY_NORMALIZE_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                              (void*)&LIBXSMM_VLA_ACCESS(4, relumask, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc/BITS_PER_CHAR) : NULL );
            cfg.relu_kernel(&all_relu_param);
          } /* ReLU */
        }
      }
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);

}

my_bn_bwd_config setup_my_bn_bwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint bc,
                                 libxsmm_blasint threads, my_normalization_fuse fuse_type ) {
  my_bn_bwd_config res;

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
  res.all_zero_kernel = libxsmm_dispatch_meltw_unary(res.bc, 1, NULL, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
                                                      LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR);
  if ( res.all_zero_kernel == NULL) {
    fprintf( stderr, "JIT for TPP bwd all_zero_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.copy_kernel = libxsmm_dispatch_meltw_unary(res.bc, 1, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
                                                  LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
  if ( res.copy_kernel == NULL) {
    fprintf( stderr, "JIT for TPP bwd copy_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.add_kernel = libxsmm_dispatch_meltw_binary(res.bc, 1, &ldo, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
                                                  LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_MELTW_TYPE_BINARY_ADD);
  if ( res.add_kernel == NULL) {
    fprintf( stderr, "JIT for TPP bwd add_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.relu_kernel = libxsmm_dispatch_meltw_unary(res.bc, res.H*res.W / res.num_HW_blocks, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
                                                 /*LIBXSMM_MELTW_FLAG_UNARY_NONE unsupported */ LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT, LIBXSMM_MELTW_TYPE_UNARY_RELU_INV);
  if ( res.relu_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP bwd relu_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.ewise_add_kernel = libxsmm_dispatch_meltw_binary(res.bc, res.H*res.W / res.num_HW_blocks, &ldo, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
                                                       LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_MELTW_TYPE_BINARY_ADD);
  if ( res.ewise_add_kernel == NULL) {
    fprintf( stderr, "JIT for TPP bwd ewise_add_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.ewise_copy_kernel = libxsmm_dispatch_meltw_unary(/*1, 1 */res.bc, res.H*res.W / res.num_HW_blocks, &ldo, &ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
                                                       LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
  if ( res.ewise_copy_kernel == NULL) {
    fprintf( stderr, "JIT for TPP bwd ewise_copy_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* TPP equations for dgamma, dbeta and din */

  ld = bc;
  tmp_ld2 = 1;

  /* dgamma function  */
  my_eqn11 = libxsmm_matrix_eqn_create();                                                       /* dgamma = ((inp *a + b) * dout) + dgamma */
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn11, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32);                   /* dgamma = ((inp *a + b) * dout) + dgamma */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn11, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);   /* [HW, bc] -> [bc] */
  libxsmm_matrix_eqn_push_back_binary_op(my_eqn11, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32);                   /* ((inp *a + b) * dout) */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn11, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, res.bc, res.H*res.W/res.num_HW_blocks, ld, 0, 0, in_dt );          /* inp [HW, bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, res.bc, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 );           /* a [bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, res.bc, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );           /* b [bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, res.bc, res.H*res.W/res.num_HW_blocks, ld, 3, 0, in_dt );          /* dout [HW, bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn11, res.bc, 1, 1, 4, 0, LIBXSMM_DATATYPE_F32 );           /* dgamma [bc] */
  /* libxsmm_matrix_eqn_tree_print( my_eqn11 ); */
  /* libxsmm_matrix_eqn_rpn_print( my_eqn11 ); */
  res.dgamma_func = libxsmm_dispatch_matrix_eqn( res.bc, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn11 );      /* dgamma [bc] */
  if ( res.dgamma_func == NULL) {
    fprintf( stderr, "JIT for TPP bwd dgamma_func (eqn11) failed. Bailing...!\n");
    exit(-1);
  }

  /* dbeta function  */
  my_eqn12 = libxsmm_matrix_eqn_create();                                                       /* dbeta [bc] = dout [HW, bc] + dbeta [bc] */
  libxsmm_matrix_eqn_push_back_binary_op( my_eqn12, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );                /* dbeta_tmp [HW, bc] */
  libxsmm_matrix_eqn_push_back_unary_op(my_eqn12, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32);  /* [HW, bc] -> [bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn12, res.bc, res.H*res.W/res.num_HW_blocks, ld, 3, 0, in_dt );          /* dout [HW, bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn12, res.bc, 1, 1, 5, 0, LIBXSMM_DATATYPE_F32 );           /* dbeta [bc] */
  /* libxsmm_matrix_eqn_tree_print( my_eqn12 ); */
  /* libxsmm_matrix_eqn_rpn_print( my_eqn12 ); */
  res.dbeta_func = libxsmm_dispatch_matrix_eqn( res.bc, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn12 );      /* dbeta [bc] */
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
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn16, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, res.bc, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 );           /* a [bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, res.bc, res.H*res.W/res.num_HW_blocks, ld, 3, 0, in_dt );          /* dout [HW, bc] */
  libxsmm_matrix_eqn_push_back_ternary_op( my_eqn16, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, res.bc, res.H*res.W/res.num_HW_blocks, ld, 0, 0, in_dt );          /* inp [HW, bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, res.bc, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );           /* b [bc] */
  libxsmm_matrix_eqn_push_back_arg( my_eqn16, res.bc, 1, 1, 7, 0, LIBXSMM_DATATYPE_F32 );           /* c [bc] */
  /* libxsmm_matrix_eqn_tree_print( my_eqn16 ); */
  /* libxsmm_matrix_eqn_rpn_print( my_eqn16 ); */
  res.din_func = libxsmm_dispatch_matrix_eqn( res.bc, res.H*res.W/res.num_HW_blocks, &ld, in_dt, my_eqn16 );           /* din [HW, bc] */
  if ( res.din_func == NULL) {
    fprintf( stderr, "JIT for TPP bwd din_func (eqn16) failed. Bailing...!\n");
    exit(-1);
  }

  /* init scratch */
  dbeta_N_offset = LIBXSMM_UP2(res.CP * res.N * res.bc, 64);
  res.scratch_size =  sizeof(float) * ( dbeta_N_offset /* dbeta_N*/ + LIBXSMM_UP2(res.CP * res.N * res.bc, 64) /*dgamma_N */ );

  return res;
}

void my_bn_bwd_exec( my_bn_bwd_config cfg, float *pdout, const float *pinp, const float *mean, const float *var, const float *pgamma, const unsigned char *prelumask,
                     float *pdin, float *pdin_add, float *pdgamma, float *pdbeta, float eps,
                     int start_tid, int my_tid, void *scratch) {

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

  LIBXSMM_VLA_DECL(4,       float, din, pdin, CP, HW, bc);          /* [N, CP, HW, bc] */
  LIBXSMM_VLA_DECL(4, const float, inp, pinp, CP, HW, bc);          /* [N, CP, HW, bc] */
  LIBXSMM_VLA_DECL(4,       float, dout, pdout, CP, HW, bc);        /* [N, CP, HW, bc] */
  LIBXSMM_VLA_DECL(2, const float, gamma, pgamma, bc);              /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float, dgamma, pdgamma, bc);            /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float, dbeta, pdbeta, bc);              /* [CP, bc] */

  LIBXSMM_VLA_DECL(4,       float, din_add, pdin_add, CP, HW, bc);          /* [N, CP, HW, bc] */

  float alpha = 0.0f;
  LIBXSMM_VLA_DECL(4,       unsigned char, relumask, prelumask, CP, HW, bc/BITS_PER_CHAR);    /* [N, CP, HW, bc/BITS_PER_CHAR] */

  const libxsmm_blasint dbeta_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + CP * N * bc), 64) - ((uintptr_t)(scratch))) / sizeof(float);
  LIBXSMM_VLA_DECL(3, float, dgamma_N, ((float*)scratch),                  N, bc);  /* [CP, N, bc] */
  LIBXSMM_ASSUME_ALIGNED(dgamma_N_, 64);
  LIBXSMM_VLA_DECL(3, float, dbeta_N,  ((float*)scratch) + dbeta_N_offset, N, bc);  /* [CP, N, bc] */
  LIBXSMM_ASSUME_ALIGNED(dbeta_N_, 64);

  { /* stupid block to keep indentation */
    LIBXSMM_ALIGNED(float a[bc], 64); /* could also get moved into the scratch but left on the private stack as these are small, same below */
    LIBXSMM_ALIGNED(float b[bc], 64);
    LIBXSMM_ALIGNED(float c[bc], 64);
    int n, cp;

    int cpxnt;

    /* ReLU/Mask/Eltwise */
    if (cfg.fuse_type == MY_NORMALIZE_FUSE_ELTWISE ||
        cfg.fuse_type == MY_NORMALIZE_FUSE_RELU || cfg.fuse_type == MY_NORMALIZE_FUSE_RELU_WITH_MASK || cfg.fuse_type == MY_NORMALIZE_FUSE_ELTWISE_RELU_WITH_MASK) {

      for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
        { /* stupid block to keep indentation */
          n  = cpxnt%N;
          cp = cpxnt/N;

          int hwb, cb;

          for(hwb=0; hwb < num_HW_blocks; hwb++){
            if (cfg.fuse_type == MY_NORMALIZE_FUSE_RELU || cfg.fuse_type == MY_NORMALIZE_FUSE_RELU_WITH_MASK || cfg.fuse_type == MY_NORMALIZE_FUSE_ELTWISE_RELU_WITH_MASK) {
              libxsmm_meltw_unary_param all_relu_param;

              all_relu_param.op.primary   = (void*)(&alpha);
              all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);      /* [HW,bc] */
              all_relu_param.in.secondary = ((cfg.fuse_type == MY_NORMALIZE_FUSE_RELU_WITH_MASK || cfg.fuse_type == MY_NORMALIZE_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                               (void*)&LIBXSMM_VLA_ACCESS(4, relumask, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc/8)
                                               : NULL /*&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc) */ ); /* dout_fwd ? nonsense? */
              all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);      /* [HW,bc] */
              cfg.relu_kernel(&all_relu_param);
            } /* ReLU/mask */
//#if 0
            if (cfg.fuse_type == MY_NORMALIZE_FUSE_ELTWISE || cfg.fuse_type == MY_NORMALIZE_FUSE_ELTWISE_RELU_WITH_MASK) {
#if 0
              int i;
              for (i = 0; i < bc * (HW/num_HW_blocks); ++i) {
                int index;
                index = n * (CP * HW * bc) + cp * (HW * bc) + (hwb*(HW/num_HW_blocks)) * (bc) + i;
                pdin_add[index] = pdout[index];
              }
#endif
//#if 0
              libxsmm_meltw_unary_param ewise_copy_param;
              ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(4, dout,    n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
              ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(4, din_add, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
              cfg.ewise_copy_kernel(&ewise_copy_param);
//#endif
            } /* Eltwise */
//#endif
          }
        }
      } /* loop over the 1d parallel blocking */
    }  /* ReLU/Mask/Eltwise */

    libxsmm_barrier_wait(cfg.barrier, ltid);

    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      { /* stupid block to keep indentation */
        n  = cpxnt%N;
        cp = cpxnt/N;

        int hwb, cb;
        libxsmm_matrix_arg arg_array[10];                                                           /* Private values of args and params */
        libxsmm_matrix_eqn_param eqn_param;

        LIBXSMM_ALIGNED(float lcl_dgamma_ptr[bc], 64);
        LIBXSMM_ALIGNED(float lcl_dbeta_ptr[bc], 64);

        float *dgamma_ncp_ptr = &LIBXSMM_VLA_ACCESS(3, dgamma_N, cp, n, 0, N, bc);
        float *dbeta_ncp_ptr  = &LIBXSMM_VLA_ACCESS(3, dbeta_N, cp, n, 0, N, bc);

        libxsmm_meltw_unary_param all_zero_param;
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
          a[cb] = 1.0f / ((float)sqrt(var[cp*bc + cb] + eps));
          b[cb] = -a[cb]*mean[cp*bc + cb];
        }

        arg_array[1].primary = a;
        arg_array[2].primary = b;
        arg_array[4].primary = lcl_dgamma_ptr;
        arg_array[5].primary = lcl_dbeta_ptr;
        arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);

        for(hwb=0; hwb < num_HW_blocks; hwb++){

          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(4, inp, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);
          arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc);

          eqn_param.inputs = arg_array;
          eqn_param.output.primary = lcl_dgamma_ptr;
          cfg.dgamma_func(&eqn_param);                                                             /* dgamma += (a * inp + b) * dout */

          eqn_param.output.primary = lcl_dbeta_ptr;
          cfg.dbeta_func(&eqn_param);                                                              /* dbeta += dout */
        }

        libxsmm_meltw_unary_param copy_param;
        copy_param.in.primary = lcl_dgamma_ptr;
        copy_param.out.primary = dgamma_ncp_ptr;
        cfg.copy_kernel(&copy_param);

        copy_param.in.primary = lcl_dbeta_ptr;
        copy_param.out.primary = dbeta_ncp_ptr;
        cfg.copy_kernel(&copy_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < bc; cb++) { */
        /*   dgamma_ncp_ptr[cb] = lcl_dgamma_ptr[cb]; */
        /*   dbeta_ncp_ptr[cb] = lcl_dbeta_ptr[cb]; */
        /* } */
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);

    for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {
      libxsmm_meltw_unary_param all_zero_param;
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, bc);
      cfg.all_zero_kernel(&all_zero_param);

      /* #pragma omp simd */
      /* for (int cb = 0; cb < bc; cb++) { */
      /*   pdgamma[cp*bc + cb] = 0.0f; */
      /*   pdbeta[cp*bc + cb] = 0.0f; */
      /* } */

      libxsmm_meltw_binary_param add_param;
      int ni;
      for(ni = 0; ni < N; ni++){

        add_param.in0.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, cp, ni, 0, N, bc);
        add_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
        cfg.add_kernel(&add_param);

        add_param.in0.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, bc);
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, cp, ni, 0, N, bc);
        add_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, bc);
        cfg.add_kernel(&add_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < bc; cb++) { */
        /*   pdgamma[cp*bc + cb] += dgamma_N[cp*N*bc + n*bc + cb];  */
        /*   pdbeta[cp*bc + cb] += dbeta_N[cp*N*bc + n*bc + cb];  */
        /* } */
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);

    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      { /* stupid block to keep indentation */
        n  = cpxnt%N;
        cp = cpxnt/N;

        libxsmm_matrix_arg arg_array[8];                                                               /* Private eqn args and params */
        libxsmm_matrix_eqn_param eqn_param;
        int hwb, cb;

        /* FIXME: Replace expressions for pgamma, pdgamma etc. with ACCESS? */
        for(cb = 0; cb < bc; cb++){
          a[cb] = pgamma[cp*bc + cb] / ((float)sqrt(var[cp*bc + cb] + eps));                            /* a = gamma_ptr[bc] * brstd_ptr[bc] */
          b[cb] = -a[cb] * scale * pdgamma[cp*bc + cb] / ((float)sqrt(var[cp*bc + cb] + eps));          /* b = gamma_ptr[bc] * brstd_ptr[bc] * del_gamma_ptr[v] * brstd_ptr[bc] * recp_nhw */
          c[cb] = -b[cb] * mean[cp*bc + cb] - a[cb] * scale * pdbeta[cp*bc + cb] ;                      /* c = -gamma_ptr[bc] * brstd_ptr[bc] * recp_nhw * del_beta_ptr[bc] + gamma_ptr[bc] * brstd_ptr[bc] * recp_nhw * bmean_ptr[bc] * del_gamma_ptr[bc] * brstd_ptr[bc]) */
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
          cfg.din_func(&eqn_param);                                                                        /* din = dout * a + b * inp + c */

        }
      }
    }
  } /* simple code block or parallel section for old code */

  libxsmm_barrier_wait(cfg.barrier, ltid);
}


#ifndef CNN_HEADER

int main( int argc, char* argv[] ) {

  my_bn_fwd_config my_bn_fwd;
  my_bn_bwd_config my_bn_bwd;

  naive_fusedbatchnorm_t naive_param;
  void *scratch;

  const float eps = FLT_EPSILON;
  libxsmm_blasint i, it;
  float *inp, *inp_add, *out, *dinp, *dout, *dinp_add, *eqn_dinp, *eqn_dout, *eqn_dinp_add, *dbeta, *eqn_dbeta, *dgamma, *eqn_dgamma, *eqn_out, *gamma, *beta, *cache_fl, *mean, *var, sum = 0.0;
  unsigned char *relumask_uncompressed, *relumask, *eqn_relumask;
  float *naive_inp, *naive_inp_add, *naive_out, *naive_rcpstdev, *naive_zeros, *naive_dinp, *naive_dout, *naive_dbeta, *naive_dgamma, *naive_dinp_add;
  unsigned char *naive_relumask;

#ifdef COMPUTE_FP64_REFERENCE
  double *naive_inp_fp64, *naive_inp_add_fp64, *naive_out_fp64, *naive_rcpstdev_fp64, *naive_zeros_fp64, *naive_dinp_fp64, *naive_dout_fp64, *naive_dbeta_fp64, *naive_dgamma_fp64, *naive_dinp_add_fp64;
  double *beta_fp64, *gamma_fp64, *mean_fp64, *var_fp64;
  double *dbeta_fp64, *dgamma_fp64;
  float *naive_out_fp64_downscaled_to_fp32, *out_fp64_downscaled_to_fp32;
  float *naive_dinp_fp64_downscaled_to_fp32, *dinp_fp64_downscaled_to_fp32;
  float *naive_dinp_add_fp64_downscaled_to_fp32, *dinp_add_fp64_downscaled_to_fp32;
  float *dgamma_fp64_downscaled_to_fp32;
  float *dbeta_fp64_downscaled_to_fp32;
#endif

  int iters     = 100;
  int N         = 28;
  int C         = 2 * 64;
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
    printf("Usage: %s iters N CP H W bc pad_w_in pad_h_in pad_w_out pad_h_out stride norm_type fuse_type (tail is optional) \n", argv[0]);
    return 0;
  }

  libxsmm_rng_set_seed(1);

  /* reading new values from cli */
  i = 1;
  if ( argc > i ) iters = atoi(argv[i++]);
  if ( argc > i ) N = atoi(argv[i++]);
  if ( argc > i ) C = atoi(argv[i++]);
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
  else { /* else, set formally H and W from the previously set HW */
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

/*
  if (fuse_type != 0 && fuse_type != 4 && fuse_type != 5) {
    printf("Unsupported fuse_type %d was provided (0, 4 and 5 are supported only)\n", fuse_type);
    return -1;
  }
*/

  stride_w = stride;
  stride_h = stride;

  /* set struct for naive batch normalization */
  naive_param.N = N;
  naive_param.C = CP*bc;
  naive_param.H = H;
  naive_param.W = W;
  naive_param.stride_h = stride_h;
  naive_param.stride_w = stride_w;
  naive_param.norm_type = norm_type; /* 0: full batchnorm, 1: batch scaling only */
  naive_param.fuse_type = fuse_type; /* 0: nothing fused, 1: relu fused, 2: elementwise fused, 3: relu and elementwise fused */

#if defined(__SSE3__)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

  /* print some summary */
  printf("##########################################\n");
  printf("#          Setting Up (Common)           #\n");
  printf("##########################################\n");
  printf("PARAMS: N:%d  C:%d  CP:%d bc:%d H:%d W:%d STRIDE:%d\n", N, CP*bc, CP, bc, H, W, stride);
  printf("PARAMS: FUSE TYPE:%d\n", fuse_type);
  printf("PARAMS: ITERS:%d", iters); if (LIBXSMM_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  printf("SIZE Input  (MB): %10.2f MiB\n", (double)(N*CP*HW*bc*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output (MB): %10.2f MiB\n", (double)(N*CP*HW*bc*sizeof(float))/(1024.0*1024.0) );

  /* allocate data */
  inp        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  out        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  inp_add    = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  dinp       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  dout       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  dinp_add   = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  dgamma     = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  dbeta      = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  eqn_dinp   = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  eqn_dout   = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  eqn_dgamma = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  eqn_dbeta  = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  eqn_dinp_add = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  gamma      = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  beta       = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  mean       = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  var        = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  eqn_out    = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  cache_fl   = (float*) libxsmm_aligned_malloc( sizeof(float)*1024*1024,   2097152);

  relumask     = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*CP*HW*bc, 2097152);
  relumask_uncompressed = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*CP*HW*bc, 2097152);
  eqn_relumask = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*CP*HW*bc, 2097152);

  naive_inp      = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_out      = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_inp_add  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_dinp     = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_dout     = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_dinp_add = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_dgamma   = (float*) libxsmm_aligned_malloc( sizeof(float)*C,       2097152);
  naive_dbeta    = (float*) libxsmm_aligned_malloc( sizeof(float)*C,       2097152);
  naive_rcpstdev = (float*) libxsmm_aligned_malloc( sizeof(float)*C,       2097152);
  naive_zeros    = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);

  naive_relumask = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*C*H*W, 2097152);

#ifdef COMPUTE_FP64_REFERENCE
  naive_inp_fp64      = (double*) libxsmm_aligned_malloc( sizeof(double)*N*C*H*W, 2097152);
  naive_inp_add_fp64  = (double*) libxsmm_aligned_malloc( sizeof(double)*N*C*H*W, 2097152);
  naive_out_fp64      = (double*) libxsmm_aligned_malloc( sizeof(double)*N*C*H*W, 2097152);
  naive_dinp_fp64     = (double*) libxsmm_aligned_malloc( sizeof(double)*N*C*H*W,   2097152);
  naive_dout_fp64     = (double*) libxsmm_aligned_malloc( sizeof(double)*N*C*H*W,   2097152);
  naive_dinp_add_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*N*C*H*W,   2097152);
  naive_dgamma_fp64   = (double*) libxsmm_aligned_malloc( sizeof(double)*C,   2097152);
  naive_dbeta_fp64    = (double*) libxsmm_aligned_malloc( sizeof(double)*C,   2097152);
  naive_rcpstdev_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*C,   2097152);
  naive_zeros_fp64    = (double*) libxsmm_aligned_malloc( sizeof(double)*N*C*H*W, 2097152);

  gamma_fp64  = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*bc,   2097152);
  beta_fp64   = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*bc,   2097152);
  mean_fp64   = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*bc,   2097152);
  var_fp64    = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*bc,   2097152);

  dgamma_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*bc,   2097152);
  dbeta_fp64  = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*bc,   2097152);
  dgamma_fp64_downscaled_to_fp32     = (float*) libxsmm_aligned_malloc( sizeof(float)*(CP*bc)*1, 2097152);
  dbeta_fp64_downscaled_to_fp32      = (float*) libxsmm_aligned_malloc( sizeof(float)*(CP*bc)*1, 2097152);

  naive_out_fp64_downscaled_to_fp32  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_dinp_fp64_downscaled_to_fp32 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_dinp_add_fp64_downscaled_to_fp32 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);

  out_fp64_downscaled_to_fp32        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*(CP*bc)*HW*1, 2097152);
  dinp_fp64_downscaled_to_fp32       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*(CP*bc)*HW*1, 2097152);
  dinp_add_fp64_downscaled_to_fp32   = (float*) libxsmm_aligned_malloc( sizeof(float)*N*(CP*bc)*HW*1, 2097152);

#endif

  /* initialize data */
  init_buf(inp,      N*CP*HW*bc, 1, 0);
  //init_buf(out,      N*CP*HW*bc, 1, 0);
  init_buf(inp_add,  N*CP*HW*bc, 1, 0);
  init_buf(dinp,     N*CP*HW*bc, 1, 0);
  init_buf(dout,     N*CP*HW*bc, 1, 0);
  //init_buf(dinp_add, N*CP*HW*bc, 1, 0);

  //copy_buf(out,      eqn_out,      N*CP*HW*bc);
  //copy_buf(dinp,     eqn_dinp,     N*CP*HW*bc);
  copy_buf(dout,     eqn_dout,     N*CP*HW*bc);

  zero_buf(naive_zeros, N*C*H*W);
#ifdef COMPUTE_FP64_REFERENCE
  zero_buf_fp64(naive_zeros_fp64, N*C*H*W);
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

  zero_buf_uint8(relumask, N*CP*HW*bc);
  zero_buf_uint8(relumask_uncompressed, N*CP*HW*bc);

  init_buf(cache_fl,  1024*1024, 1, 0);

  /* setup TPPs (standalone or through the configs) */

  my_bn_fwd = setup_my_bn_fwd(N, C, H, W, bc, nThreads, (my_normalization_fuse)fuse_type );
  my_bn_bwd = setup_my_bn_bwd(N, C, H, W, bc, nThreads, (my_normalization_fuse)fuse_type );

  /* allocate and bind scratch */
  if ( my_bn_fwd.scratch_size > 0 || my_bn_bwd.scratch_size > 0 ) {
    size_t alloc_size = LIBXSMM_MAX( my_bn_fwd.scratch_size, my_bn_bwd.scratch_size);
    scratch = libxsmm_aligned_malloc( alloc_size, 2097152 );
    init_buf( (float*)(scratch), (alloc_size)/4, 0, 0 );
  }

  /* Check correctness */
  if (LIBXSMM_NEQ(0, check)) {
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_bn_fwd_exec( my_bn_fwd, inp, inp_add, gamma, beta, mean, var, eqn_out, eqn_relumask, eps, 0, tid, scratch);
    }

    tensor_copy_NCHWc_to_NCHW (inp,     naive_inp,     N, C, H, W, bc);
    tensor_copy_NCHWc_to_NCHW (inp_add, naive_inp_add, N, C, H, W, bc);

    naive_fusedbatchnorm_fp(&naive_param, naive_inp, naive_out, naive_inp_add,
                                        beta, gamma, eps, mean, naive_rcpstdev, var, naive_relumask);

    tensor_copy_NCHW_to_NCHWc       (naive_out     , out,      N, C, H, W, bc);
    tensor_copy_NCHW_to_NCHWc_uint8 (naive_relumask, relumask_uncompressed, N, C, H, W, bc);
    mask_compress_uint8 (relumask_uncompressed, relumask, N*CP*H*W*bc);

#ifdef COMPUTE_FP64_REFERENCE
    extend_buf_fp32_to_fp64 (naive_inp,     naive_inp_fp64,     N*C*H*W);
    extend_buf_fp32_to_fp64 (naive_inp_add, naive_inp_add_fp64, N*C*H*W);

    naive_fusedbatchnorm_fp_fp64(&naive_param, naive_inp_fp64, naive_out_fp64, naive_inp_add_fp64,
                                        beta_fp64, gamma_fp64, eps, mean_fp64, naive_rcpstdev_fp64, var_fp64, naive_relumask);

    truncate_buf_fp64_to_fp32 (naive_out_fp64, naive_out_fp64_downscaled_to_fp32, N*C*H*W);

    tensor_copy_NCHW_to_NCHWc (naive_out_fp64_downscaled_to_fp32, out_fp64_downscaled_to_fp32, N, C, H, W, bc);

    tensor_copy_NCHW_to_NCHWc_uint8 (naive_relumask, relumask_uncompressed, N, C, H, W, bc);
    mask_compress_uint8 (relumask_uncompressed, relumask, N*CP*H*W*bc);
#endif

    /* compare */
    printf("############################################\n");
    printf("# Correctness FP32 FWD Batchnorm - Output  #\n");
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
    printf("# Correctness FP32 FWD Batchnorm - Output (fp64) #\n");
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

    if (fuse_type == 4 || fuse_type == 5) {
      printf("############################################\n");
      printf("# Correctness FP32 FWD Batchnorm - Relumask  #\n");
      printf("############################################\n");
      libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_I8, N*CP*HW*bc, 1, relumask, eqn_relumask, 0, 0);
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n\n", norms_fwd.normf_rel);
    }
  } /* checking correctness for FWD */

  for (i = 0; i < 1024 * 1024; i++ ) {
    sum += cache_fl[i];
  }
  naive_fusedbatchnorm_fp(&naive_param, naive_inp, naive_out, naive_inp_add,
                                        beta, gamma, eps, mean, naive_rcpstdev, var, naive_relumask);
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
    naive_fusedbatchnorm_fp(&naive_param, naive_inp, naive_out, naive_inp_add,
                                        beta, gamma, eps, mean, naive_rcpstdev, var, naive_relumask);
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Scaler batchnorm time FWD  = %.5g\n", ((double)(l_total)));
  for (i = 0; i < 1024 * 1024; i++ ) {
    sum += cache_fl[i] + (float)l_total;
  }
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_bn_fwd_exec( my_bn_fwd, inp, inp_add, gamma, beta, mean, var, eqn_out, eqn_relumask, eps, 0, tid, scratch);
    }
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_bn_fwd_exec( my_bn_fwd, inp, inp_add, gamma, beta, mean, var, eqn_out, eqn_relumask, eps, 0, tid, scratch );
    }
  }
  l_end = libxsmm_timer_tick();
  l_total2 = libxsmm_timer_duration(l_start, l_end);
  printf("TPP batchnorm time FWD  = %.5g\n", ((double)(l_total2)));
  printf("Speedup FWD is %.5g\n", l_total/l_total2);

  if (LIBXSMM_NEQ(0, check)) {
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif

      my_bn_bwd_exec( my_bn_bwd, eqn_dout, inp, mean, var, gamma, relumask, eqn_dinp, eqn_dinp_add, eqn_dgamma, eqn_dbeta, eps, 0, tid, scratch );
    }

    tensor_copy_NCHWc_to_NCHW (inp,  naive_inp,   N, C, H, W, bc);
    tensor_copy_NCHWc_to_NCHW (out,  naive_out,   N, C, H, W, bc);
    tensor_copy_NCHWc_to_NCHW (dout, naive_dout,  N, C, H, W, bc);

    naive_fusedbatchnorm_bp(&naive_param, naive_inp, naive_dinp, naive_out, naive_dout, naive_dinp_add,
                                       beta, dbeta, gamma, dgamma, mean, naive_rcpstdev);

    tensor_copy_NCHW_to_NCHWc (naive_dinp,     dinp,     N, C, H, W, bc);
    tensor_copy_NCHW_to_NCHWc (naive_dinp_add, dinp_add, N, C, H, W, bc);

#ifdef COMPUTE_FP64_REFERENCE
    extend_buf_fp32_to_fp64 (naive_inp,  naive_inp_fp64,  N*C*H*W);
    extend_buf_fp32_to_fp64 (naive_out,  naive_out_fp64,  N*C*H*W);
    extend_buf_fp32_to_fp64 (naive_dout, naive_dout_fp64, N*C*H*W);

    naive_fusedbatchnorm_bp_fp64(&naive_param, naive_inp_fp64, naive_dinp_fp64, naive_out_fp64, naive_dout_fp64, naive_dinp_add_fp64,
                                       beta_fp64, dbeta_fp64, gamma_fp64, dgamma_fp64, mean_fp64, naive_rcpstdev_fp64);

    truncate_buf_fp64_to_fp32 (naive_dinp_fp64,     naive_dinp_fp64_downscaled_to_fp32,     N*C*H*W);
    truncate_buf_fp64_to_fp32 (naive_dinp_add_fp64, naive_dinp_add_fp64_downscaled_to_fp32, N*C*H*W);
    truncate_buf_fp64_to_fp32 (dgamma_fp64, dgamma_fp64_downscaled_to_fp32, CP*bc);
    truncate_buf_fp64_to_fp32 (dbeta_fp64,  dbeta_fp64_downscaled_to_fp32, CP*bc);

    tensor_copy_NCHW_to_NCHWc (naive_dinp_fp64_downscaled_to_fp32,     dinp_fp64_downscaled_to_fp32,     N, C, H, W, bc);
    tensor_copy_NCHW_to_NCHWc (naive_dinp_add_fp64_downscaled_to_fp32, dinp_add_fp64_downscaled_to_fp32, N, C, H, W, bc);
#endif

    /* compare */
    printf("############################################\n");
    printf("# Correctness FP32 BWD Batchnorm - Dinput  #\n");
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
    printf("# Correctness FP32 BWD Batchnorm - Dinput (fp64) #\n");
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

    if (fuse_type == 2 || fuse_type == 5) {
      printf("################################################\n");
      printf("# Correctness FP32 BWD Batchnorm - Dinput add  #\n");
      printf("################################################\n");
      libxsmm_matdiff(&norms_bwd_d, LIBXSMM_DATATYPE_F32, N*CP*HW*bc, 1, dinp_add, eqn_dinp_add, 0, 0);
      printf("L1 reference  : %.25g\n", norms_bwd_d.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd_d.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd_d.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd_d.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd_d.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd_d.linf_rel);
      printf("Check-norm    : %.24f\n\n", norms_bwd_d.normf_rel);

#ifdef COMPUTE_FP64_REFERENCE
      printf("##################################################\n");
      printf("# Correctness FP32 BWD Batchnorm - Dinput add (fp64) #\n");
      printf("##################################################\n");
      libxsmm_matdiff(&norms_bwd_d, LIBXSMM_DATATYPE_F32, N*CP*HW*bc, 1, dinp_add_fp64_downscaled_to_fp32, eqn_dinp_add, 0, 0);
      printf("L1 reference  : %.25g\n", norms_bwd_d.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd_d.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd_d.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd_d.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd_d.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd_d.linf_rel);
      printf("Check-norm    : %.24f\n\n", norms_bwd_d.normf_rel);
#endif
    }

    printf("###########################################\n");
    printf("# Correctness FP32 BWD Batchnorm - Dbeta  #\n");
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
    printf("# Correctness FP32 BWD Batchnorm - Dbeta (fp64)  #\n");
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
    printf("# Correctness FP32 BWD Batchnorm - Dgamma  #\n");
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
    printf("# Correctness FP32 BWD Batchnorm - Dgamma (fp64) #\n");
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
  naive_fusedbatchnorm_bp(&naive_param, naive_inp, naive_dinp, naive_out, naive_dout, naive_dinp_add,
                                       beta, dbeta, gamma, dgamma, mean, naive_rcpstdev);
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
    naive_fusedbatchnorm_bp(&naive_param, naive_inp, naive_dinp, naive_out, naive_dout, naive_dinp_add,
                                       beta, dbeta, gamma, dgamma, mean, naive_rcpstdev);
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Scaler batchnorm time BWD = %.5g\n", ((double)(l_total)));
  for (i = 0; i < 1024 * 1024; i++ ) {
    sum += cache_fl[i] + (float)l_total;
  }
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_bn_bwd_exec( my_bn_bwd, eqn_dout, inp, mean, var, gamma, relumask, eqn_dinp, eqn_dinp_add, eqn_dgamma, eqn_dbeta, eps, 0, tid, scratch );
    }
  l_start = libxsmm_timer_tick();

  for (it = 0; it < iters; it++) {
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_bn_bwd_exec( my_bn_bwd, eqn_dout, inp, mean, var, gamma, relumask, eqn_dinp, eqn_dinp_add, eqn_dgamma, eqn_dbeta, eps, 0, tid, scratch );
    }
  }

  l_end = libxsmm_timer_tick();
  l_total2 = libxsmm_timer_duration(l_start, l_end);
  printf("TPP batchnorm time BWD = %.5g\n", ((double)(l_total2)));
  printf("Speedup BWD is %.5g\n", l_total/l_total2);
  /* printf("Running sum is %.5f\n", sum); */

  t_tpp += l_total2;
  t_vec += l_total;

  printf("\n\n=================================\n");
  printf("Total Speedup via TPP Matrix equation is %.5g\n", t_vec/t_tpp);
  printf("=================================\n");

  /* deallocate data */
  if ( scratch != NULL ) {
    libxsmm_free(scratch);
  }
  libxsmm_free(inp);
  libxsmm_free(out);
  libxsmm_free(inp_add);
  libxsmm_free(dinp);
  libxsmm_free(dout);
  libxsmm_free(dinp_add);
  libxsmm_free(eqn_dinp);
  libxsmm_free(eqn_dout);
  libxsmm_free(eqn_dinp_add);
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

  libxsmm_free(relumask);
  libxsmm_free(relumask_uncompressed);
  libxsmm_free(eqn_relumask);

  libxsmm_free(naive_inp);
  libxsmm_free(naive_out);
  libxsmm_free(naive_inp_add);
  libxsmm_free(naive_dinp);
  libxsmm_free(naive_dout);
  libxsmm_free(naive_dgamma);
  libxsmm_free(naive_dbeta);
  libxsmm_free(naive_rcpstdev);
  libxsmm_free(naive_zeros);

  libxsmm_free(naive_relumask);

#ifdef COMPUTE_FP64_REFERENCE
  libxsmm_free(naive_inp_fp64);
  libxsmm_free(naive_out_fp64);
  libxsmm_free(naive_inp_add_fp64);
  libxsmm_free(naive_rcpstdev_fp64);
  libxsmm_free(naive_zeros_fp64);
  libxsmm_free(naive_dinp_fp64);
  libxsmm_free(naive_dout_fp64);
  libxsmm_free(naive_dinp_add_fp64);
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
  libxsmm_free(naive_dinp_add_fp64_downscaled_to_fp32);
  libxsmm_free(dinp_add_fp64_downscaled_to_fp32);
  libxsmm_free(dgamma_fp64_downscaled_to_fp32);
  libxsmm_free(dbeta_fp64_downscaled_to_fp32);
#endif

  return 0;
}

#endif /* for #ifndef CNN_HEADER */

