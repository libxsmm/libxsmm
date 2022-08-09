/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Evangelos Georganas (Intel Corp.)
******************************************************************************/

#include <libxsmm_dnn_pool.h>

LIBXSMM_API libxsmm_dnn_pooling_fwd_config setup_libxsmm_dnn_pooling_fwd( const libxsmm_blasint N, const libxsmm_blasint C, const libxsmm_blasint H, const libxsmm_blasint W,
                                            const libxsmm_blasint R, const libxsmm_blasint S,
                                            const libxsmm_blasint stride_h, const libxsmm_blasint stride_w,
                                            const libxsmm_blasint pad_h, const libxsmm_blasint pad_w,
                                            const libxsmm_blasint pad_h_in, const libxsmm_blasint pad_w_in,
                                            const libxsmm_blasint pad_h_out, const libxsmm_blasint pad_w_out,
                                            const libxsmm_blasint bc, const libxsmm_blasint threads, const libxsmm_dnn_pooling_type pool_type,
                                            const libxsmm_datatype datatype_in, const libxsmm_datatype datatype_out, const libxsmm_datatype datatype_comp ) {
  libxsmm_dnn_pooling_fwd_config res;
  libxsmm_bitfield unary_flags  = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_bitfield binary_flags = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  libxsmm_meltw_unary_shape  unary_shape;
  libxsmm_meltw_binary_shape binary_shape;

  /* check supported precision */
  if ( !((datatype_in == LIBXSMM_DATATYPE_F32) && (datatype_out == LIBXSMM_DATATYPE_F32) && (datatype_comp == LIBXSMM_DATATYPE_F32)) &&
       !((datatype_in == LIBXSMM_DATATYPE_BF16) && (datatype_out == LIBXSMM_DATATYPE_BF16) && (datatype_comp == LIBXSMM_DATATYPE_F32)) ) {
    fprintf( stderr, "Unsupported precision for bwdupd pass Bailing...!\n");
    exit(-1);
  }

  /* setting args */
  res.N = N;
  res.C = C;
  res.H = H;
  res.W = W;
  res.R = R;
  res.S = S;
  res.bc = bc;
  res.Bc = C / bc;
  res.pool_type = pool_type;
  res.pass_type = LIBXSMM_DNN_POOLING_PASS_FWD;
  res.u = stride_h;
  res.v = stride_w;
  res.pad_h = pad_h;
  res.pad_w = pad_w;
  res.pad_h_in = pad_h_in;
  res.pad_w_in = pad_w_in;
  res.pad_h_out = pad_h_out;
  res.pad_w_out = pad_w_out;
  /* setting ofh and ofw */
  res.ofh = (H + 2 * pad_h - R) / stride_h + 1;
  res.ofw = (W + 2 * pad_w - S) / stride_w + 1;
   /* create barrier */
  res.threads = threads;
  res.barrier = libxsmm_barrier_create(threads, 1);
  /* datatype */
  res.datatype_in = datatype_in;
  res.datatype_out = datatype_out;
  res.datatype_comp = datatype_comp;
  /* calculate scratch size for local pooling copies of one feature map block per thread */
  res.scratch_size = (((R*S)+63)/64)*64*sizeof(int)*threads;

  /* Setup fwd kernels */
  unary_shape = libxsmm_create_meltw_unary_shape( res.bc, 0, res.bc, res.bc, datatype_in, datatype_out, LIBXSMM_DATATYPE_F32 );
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES | LIBXSMM_MELTW_FLAG_UNARY_REDUCE_NO_PREFETCH;
  if ( res.pool_type == LIBXSMM_DNN_POOLING_TYPE_MAX ) {
    unary_flags = unary_flags | LIBXSMM_MELTW_FLAG_UNARY_REDUCE_NEG_INF_ACC | LIBXSMM_MELTW_FLAG_UNARY_REDUCE_RECORD_ARGOP;
    res.fwd_pool_reduce_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MAX, unary_shape, unary_flags );
  } else if ( res.pool_type == LIBXSMM_DNN_POOLING_TYPE_MAX_NOMASK )  {
    unary_flags = unary_flags | LIBXSMM_MELTW_FLAG_UNARY_REDUCE_NEG_INF_ACC;
    res.fwd_pool_reduce_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MAX, unary_shape, unary_flags );
  } else {
    res.fwd_pool_reduce_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_ADD, unary_shape, unary_flags );
    binary_shape = libxsmm_create_meltw_binary_shape( res.bc, 1, res.bc, 1, res.bc, datatype_in, datatype_in, datatype_out, datatype_comp );
    binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1;
    res.fwd_scale_kernel = libxsmm_dispatch_meltw_binary_v2( LIBXSMM_MELTW_TYPE_BINARY_MUL, binary_shape, binary_flags );
  }

  return res;
}

LIBXSMM_API libxsmm_dnn_pooling_bwd_config setup_libxsmm_dnn_pooling_bwd( const libxsmm_blasint N, const libxsmm_blasint C, const libxsmm_blasint H, const libxsmm_blasint W,
                                            const libxsmm_blasint R, const libxsmm_blasint S,
                                            const libxsmm_blasint stride_h, const libxsmm_blasint stride_w,
                                            const libxsmm_blasint pad_h, const libxsmm_blasint pad_w,
                                            const libxsmm_blasint pad_h_in, const libxsmm_blasint pad_w_in,
                                            const libxsmm_blasint pad_h_out, const libxsmm_blasint pad_w_out,
                                            const libxsmm_blasint bc, const libxsmm_blasint threads, const libxsmm_dnn_pooling_type pool_type,
                                            const libxsmm_datatype datatype_in, const libxsmm_datatype datatype_out, const libxsmm_datatype datatype_comp ) {
  libxsmm_dnn_pooling_bwd_config res;
  libxsmm_matrix_eqn_arg_metadata arg_metadata[2];
  libxsmm_matrix_eqn_op_metadata  op_metadata;
  libxsmm_meqn_arg_shape          arg_shape;
  libxsmm_blasint                 eqn_idx = 0;
  libxsmm_matrix_arg_attributes   arg_singular_attr = libxsmm_create_matrix_arg_attributes( LIBXSMM_MATRIX_ARG_TYPE_SINGULAR, LIBXSMM_MATRIX_ARG_SET_TYPE_NONE, 0, 0);
  libxsmm_bitfield                unary_flags  = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  libxsmm_bitfield                binary_flags = LIBXSMM_MELTW_FLAG_BINARY_NONE;
  libxsmm_meltw_unary_shape       unary_shape;
  libxsmm_meltw_binary_shape      binary_shape;

  /* check supported precision */
  if ( !((datatype_in == LIBXSMM_DATATYPE_F32) && (datatype_out == LIBXSMM_DATATYPE_F32) && (datatype_comp == LIBXSMM_DATATYPE_F32)) &&
       !((datatype_in == LIBXSMM_DATATYPE_BF16) && (datatype_out == LIBXSMM_DATATYPE_BF16) && (datatype_comp == LIBXSMM_DATATYPE_F32)) ) {
    fprintf( stderr, "Unsupported precision for bwdupd pass Bailing...!\n");
    exit(-1);
  }

  /* setting args */
  res.N = N;
  res.C = C;
  res.H = H;
  res.W = W;
  res.R = R;
  res.S = S;
  res.bc = bc;
  res.Bc = C / bc;
  res.pool_type = pool_type;
  res.pass_type = LIBXSMM_DNN_POOLING_PASS_FWD;
  res.u = stride_h;
  res.v = stride_w;
  res.pad_h = pad_h;
  res.pad_w = pad_w;
  res.pad_h_in = pad_h_in;
  res.pad_w_in = pad_w_in;
  res.pad_h_out = pad_h_out;
  res.pad_w_out = pad_w_out;
  /* setting ofh and ofw */
  res.ofh = (H + 2 * pad_h - R) / stride_h + 1;
  res.ofw = (W + 2 * pad_w - S) / stride_w + 1;
   /* create barrier */
  res.threads = threads;
  res.barrier = libxsmm_barrier_create(threads, 1);
  /* datatype */
  res.datatype_in = datatype_in;
  res.datatype_out = datatype_out;
  res.datatype_comp = datatype_comp;

  /* calculate scratch size for local pooling copies of one feature map block per thread */
  res.scratch_size = 0;

  /* Setup bwd kernels */
  if ( res.pool_type == LIBXSMM_DNN_POOLING_TYPE_MAX ) {
    eqn_idx = libxsmm_matrix_eqn_create();
    arg_metadata[0] = libxsmm_create_matrix_eqn_arg_metadata(eqn_idx, 0);
    arg_metadata[1] = libxsmm_create_matrix_eqn_arg_metadata(eqn_idx, 1);
    op_metadata     = libxsmm_create_matrix_eqn_op_metadata(eqn_idx, -1);
    arg_shape       = libxsmm_create_meqn_arg_shape( bc, 1, bc, datatype_in );
    unary_flags     = LIBXSMM_MELTW_FLAG_UNARY_GS_OFFS | LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES;

    libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata, LIBXSMM_MELTW_TYPE_UNARY_SCATTER, datatype_out, unary_flags);
    if (datatype_in == LIBXSMM_DATATYPE_BF16) {
      libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, datatype_in, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    }
    libxsmm_matrix_eqn_push_back_binary_op_v2(op_metadata, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE);
    libxsmm_matrix_eqn_push_back_unary_op_v2(op_metadata, LIBXSMM_MELTW_TYPE_UNARY_GATHER, datatype_in, unary_flags);
    libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[0], arg_shape, arg_singular_attr);
    libxsmm_matrix_eqn_push_back_arg_v2(arg_metadata[1], arg_shape, arg_singular_attr);
    res.func_bwd_max_pool = libxsmm_dispatch_matrix_eqn_v2( eqn_idx, arg_shape );
  } else {
    binary_shape = libxsmm_create_meltw_binary_shape( res.bc, 1, res.bc, 1, res.bc, datatype_in, datatype_in, datatype_out, datatype_comp );
    binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1;
    res.func_bwd_avg_pool = libxsmm_dispatch_meltw_binary_v2( LIBXSMM_MELTW_TYPE_BINARY_MULADD, binary_shape, binary_flags );
  }
  unary_shape = libxsmm_create_meltw_unary_shape( res.bc*res.W, 1, res.bc*res.W, res.bc*res.W, datatype_in, datatype_out, datatype_comp );
  unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;
  res.bwd_zero_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, unary_flags );

  return res;
}

LIBXSMM_API void libxsmm_dnn_pooling_fwd_exec_f32( const libxsmm_dnn_pooling_fwd_config cfg, const float* in_act_ptr, float* out_act_ptr, int* mask_ptr,
                              const libxsmm_blasint start_tid, const libxsmm_blasint my_tid, void* scratch ) {
  /* size variables, all const */
  const libxsmm_blasint ofhp = cfg.ofh + 2*cfg.pad_h_out;
  const libxsmm_blasint ofwp = cfg.ofw + 2*cfg.pad_w_out;
  const libxsmm_blasint ifhp = cfg.H   + 2*cfg.pad_h_in;
  const libxsmm_blasint ifwp = cfg.W   + 2*cfg.pad_w_in;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;
  /* number of tasks that could be run in parallel */
  const libxsmm_blasint work = cfg.N * cfg.Bc * cfg.ofh * cfg.ofw;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

  /* loop variables */
  libxsmm_blasint img = 0;
  libxsmm_blasint fm = 0;
  libxsmm_blasint task = 0;
  libxsmm_blasint ho = 0;
  libxsmm_blasint wo = 0;
  libxsmm_blasint hi = 0;
  libxsmm_blasint wi = 0;
  libxsmm_blasint kh = 0;
  libxsmm_blasint kw = 0;
  libxsmm_blasint v = 0;
  libxsmm_blasint _ho = 0;
  libxsmm_blasint _wo = 0;

  /* only for average pooling */
  float recp_pool_size = 1.0f/((float)cfg.R*(float)cfg.S);

  /* multi-dim arrays declaration */
  LIBXSMM_VLA_DECL(5, const float,             input,  in_act_ptr, cfg.Bc,    ifhp,    ifwp, cfg.bc);
  LIBXSMM_VLA_DECL(5,       float,            output, out_act_ptr, cfg.Bc,    ofhp,    ofwp, cfg.bc);
  LIBXSMM_VLA_DECL(5,       int,                mask,    mask_ptr, cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc);
  libxsmm_meltw_unary_param  unary_param;
  libxsmm_meltw_binary_param binary_param;
  int *ind_array = (int*)scratch + (size_t)((((cfg.R*cfg.S)+63)/64)*64)*ltid;
  unsigned long long n;

  unary_param.in.secondary = ind_array;
  unary_param.in.tertiary  = &n;
  binary_param.in1.primary = (void*) &recp_pool_size;

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  for (task = thr_begin; task < thr_end; ++task) {
    img = task / (cfg.ofw * cfg.ofh * cfg.Bc);
    fm = (task % (cfg.ofw * cfg.ofh * cfg.Bc))/(cfg.ofw * cfg.ofh);
    _ho = ((task % (cfg.ofw * cfg.ofh * cfg.Bc))%(cfg.ofw * cfg.ofh))/cfg.ofw;
    _wo = ((task % (cfg.ofw * cfg.ofh * cfg.Bc))%(cfg.ofw * cfg.ofh))%cfg.ofw;
    ho = cfg.pad_h_out + _ho;
    hi = ((ho-cfg.pad_h_out) * cfg.u) - cfg.pad_h;
    wo = cfg.pad_w_out + _wo;
    wi = ((wo-cfg.pad_w_out) * cfg.v) - cfg.pad_w;

    /* Setup the reduce indicdes */
    n = 0;
    for ( kh = 0; kh < cfg.R; kh++ ) {
      if (hi+kh < 0 || hi+kh >= cfg.H) continue;
      for ( kw = 0; kw < cfg.S; kw++ ) {
        if (wi+kw < 0 || wi+kw >= cfg.W) {
          continue;
        } else {
          ind_array[n] = (hi+kh+cfg.pad_h_in) * ifwp + (wi+kw+cfg.pad_w_in);
          n++;
        }
      }
    }
    unary_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(5, input, img, fm, 0, 0, 0, cfg.Bc, ifhp, ifwp, cfg.bc);
    unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, fm, ho, wo, 0, cfg.Bc, ofhp, ofwp, cfg.bc);
    if (cfg.pool_type == LIBXSMM_DNN_POOLING_TYPE_MAX) {
      unary_param.out.secondary = (void*)&LIBXSMM_VLA_ACCESS(5, mask, img, fm, ho-cfg.pad_h_out, wo-cfg.pad_w_out, 0, cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc);
    }
    cfg.fwd_pool_reduce_kernel( & unary_param );

    if (cfg.pool_type == LIBXSMM_DNN_POOLING_TYPE_MAX) {
      for ( v = 0; v < cfg.bc; v++ ) {
        LIBXSMM_VLA_ACCESS(5, mask, img, fm, ho-cfg.pad_h_out, wo-cfg.pad_w_out, v, cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc) =
        LIBXSMM_VLA_ACCESS(5, mask, img, fm, ho-cfg.pad_h_out, wo-cfg.pad_w_out, v, cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc) * cfg.bc + v;
      }
    } else if (cfg.pool_type == LIBXSMM_DNN_POOLING_TYPE_AVG) {
      binary_param.in0.primary = (void*) &LIBXSMM_VLA_ACCESS(5, output, img, fm, ho, wo, 0, cfg.Bc, ofhp, ofwp, cfg.bc);
      binary_param.out.primary = (void*) &LIBXSMM_VLA_ACCESS(5, output, img, fm, ho, wo, 0, cfg.Bc, ofhp, ofwp, cfg.bc);
      cfg.fwd_scale_kernel( &binary_param );
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);
}

LIBXSMM_API void libxsmm_dnn_pooling_fwd_exec_bf16( const libxsmm_dnn_pooling_fwd_config cfg, const libxsmm_bfloat16* in_act_ptr, libxsmm_bfloat16* out_act_ptr, int* mask_ptr,
                               const libxsmm_blasint start_tid, const libxsmm_blasint my_tid, void* scratch ) {
  /* size variables, all const */
  const libxsmm_blasint ofhp = cfg.ofh + 2*cfg.pad_h_out;
  const libxsmm_blasint ofwp = cfg.ofw + 2*cfg.pad_w_out;
  const libxsmm_blasint ifhp = cfg.H   + 2*cfg.pad_h_in;
  const libxsmm_blasint ifwp = cfg.W   + 2*cfg.pad_w_in;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;
  /* number of tasks that could be run in parallel */
  const libxsmm_blasint work = cfg.N * cfg.Bc * cfg.ofh * cfg.ofw;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

  /* loop variables */
  libxsmm_blasint img = 0;
  libxsmm_blasint fm = 0;
  libxsmm_blasint task = 0;
  libxsmm_blasint ho = 0;
  libxsmm_blasint wo = 0;
  libxsmm_blasint hi = 0;
  libxsmm_blasint wi = 0;
  libxsmm_blasint kh = 0;
  libxsmm_blasint kw = 0;
  libxsmm_blasint v = 0;
  libxsmm_blasint _ho = 0;
  libxsmm_blasint _wo = 0;

  /* only for average pooling */
  float recp_pool_size_f32 = 1.0f/((float)cfg.R*(float)cfg.S);
  libxsmm_bfloat16 recp_pool_size;

  /* multi-dim arrays declaration */
  LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16,             input,  in_act_ptr, cfg.Bc,    ifhp,    ifwp, cfg.bc);
  LIBXSMM_VLA_DECL(5,       libxsmm_bfloat16,            output, out_act_ptr, cfg.Bc,    ofhp,    ofwp, cfg.bc);
  LIBXSMM_VLA_DECL(5,                    int,              mask,    mask_ptr, cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc);
  libxsmm_meltw_unary_param  unary_param;
  libxsmm_meltw_binary_param binary_param;
  int *ind_array = (int*)scratch + (size_t)((((cfg.R*cfg.S)+63)/64)*64)*ltid;
  unsigned long long n;
  unary_param.in.secondary = ind_array;
  unary_param.in.tertiary  = &n;
  binary_param.in1.primary = (void*) &recp_pool_size;
  libxsmm_rne_convert_fp32_bf16( &recp_pool_size_f32, &recp_pool_size, 1 );

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  for (task = thr_begin; task < thr_end; ++task) {
    img = task / (cfg.ofw * cfg.ofh * cfg.Bc);
    fm = (task % (cfg.ofw * cfg.ofh * cfg.Bc))/(cfg.ofw * cfg.ofh);
    _ho = ((task % (cfg.ofw * cfg.ofh * cfg.Bc))%(cfg.ofw * cfg.ofh))/cfg.ofw;
    _wo = ((task % (cfg.ofw * cfg.ofh * cfg.Bc))%(cfg.ofw * cfg.ofh))%cfg.ofw;
    ho = cfg.pad_h_out + _ho;
    hi = ((ho-cfg.pad_h_out) * cfg.u) - cfg.pad_h;
    wo = cfg.pad_w_out + _wo;
    wi = ((wo-cfg.pad_w_out) * cfg.v) - cfg.pad_w;
    /* Setup the reduce indicdes */
    n = 0;
    for ( kh = 0; kh < cfg.R; kh++ ) {
      if (hi+kh < 0 || hi+kh >= cfg.H) continue;
      for ( kw = 0; kw < cfg.S; kw++ ) {
        if (wi+kw < 0 || wi+kw >= cfg.W) {
          continue;
        } else {
          ind_array[n] = (hi+kh+cfg.pad_h_in) * ifwp + (wi+kw+cfg.pad_w_in);
          n++;
        }
      }
    }
    unary_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(5, input, img, fm, 0, 0, 0, cfg.Bc, ifhp, ifwp, cfg.bc);
    unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5, output, img, fm, ho, wo, 0, cfg.Bc, ofhp, ofwp, cfg.bc);
    if (cfg.pool_type == LIBXSMM_DNN_POOLING_TYPE_MAX) {
      unary_param.out.secondary = (void*)&LIBXSMM_VLA_ACCESS(5, mask, img, fm, ho-cfg.pad_h_out, wo-cfg.pad_w_out, 0, cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc);
    }
    cfg.fwd_pool_reduce_kernel( & unary_param );

    if (cfg.pool_type == LIBXSMM_DNN_POOLING_TYPE_MAX) {
      for ( v = 0; v < cfg.bc; v++ ) {
        LIBXSMM_VLA_ACCESS(5, mask, img, fm, ho-cfg.pad_h_out, wo-cfg.pad_w_out, v, cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc) =
        LIBXSMM_VLA_ACCESS(5, mask, img, fm, ho-cfg.pad_h_out, wo-cfg.pad_w_out, v, cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc) * cfg.bc + v;
      }
    } else if (cfg.pool_type == LIBXSMM_DNN_POOLING_TYPE_AVG) {
      binary_param.in0.primary = (void*) &LIBXSMM_VLA_ACCESS(5, output, img, fm, ho, wo, 0, cfg.Bc, ofhp, ofwp, cfg.bc);
      binary_param.out.primary = (void*) &LIBXSMM_VLA_ACCESS(5, output, img, fm, ho, wo, 0, cfg.Bc, ofhp, ofwp, cfg.bc);
      cfg.fwd_scale_kernel( &binary_param );
    }
  }
  libxsmm_barrier_wait(cfg.barrier, ltid);
}

LIBXSMM_API void libxsmm_dnn_pooling_bwd_exec_f32( const libxsmm_dnn_pooling_bwd_config cfg, float* din_act_ptr, const float* dout_act_ptr, const int* mask_ptr,
                              const libxsmm_blasint start_tid, const libxsmm_blasint my_tid, void* scratch ) {
  /* size variables, all const */
  const libxsmm_blasint ofhp = cfg.ofh + 2*cfg.pad_h_out;
  const libxsmm_blasint ofwp = cfg.ofw + 2*cfg.pad_w_out;
  const libxsmm_blasint ifhp = cfg.H   + 2*cfg.pad_h_in;
  const libxsmm_blasint ifwp = cfg.W   + 2*cfg.pad_w_in;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;
  /* number of tasks that could be run in parallel */
  const libxsmm_blasint work = cfg.N * cfg.Bc;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

  /* loop variables */
  libxsmm_blasint img = 0;
  libxsmm_blasint fm = 0;
  libxsmm_blasint imgfm = 0;
  libxsmm_blasint ho = 0;
  libxsmm_blasint wo = 0;
  libxsmm_blasint hi = 0;
  libxsmm_blasint wi = 0;
  libxsmm_blasint kh = 0;
  libxsmm_blasint kw = 0;
  float recp_pool_size = 1.0f/((float)cfg.R*(float)cfg.S);

  libxsmm_matrix_eqn_param eqn_param;
  libxsmm_matrix_arg arg_array[2];
  libxsmm_meltw_unary_param  unary_param;
  libxsmm_meltw_binary_param binary_param;

  /* multi-dim arrays declaration */
  LIBXSMM_VLA_DECL(5,       float,            dinput, din_act_ptr,  cfg.Bc,    ifhp,    ifwp, cfg.bc);
  LIBXSMM_VLA_DECL(5, const float,           doutput, dout_act_ptr, cfg.Bc,    ofhp,    ofwp, cfg.bc);
  LIBXSMM_VLA_DECL(5, const int,                mask, mask_ptr,     cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc);
  eqn_param.inputs = arg_array;
  binary_param.in1.primary = (void*)&recp_pool_size;

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  for (imgfm = thr_begin; imgfm < thr_end; ++imgfm) {
    img = imgfm / cfg.Bc;
    fm = imgfm % cfg.Bc;
    for ( hi = cfg.pad_h_in; hi < (cfg.H+cfg.pad_h_in); hi++ ) {
      unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5, dinput, img, fm, hi, cfg.pad_w_in, 0, cfg.Bc, ifhp, ifwp, cfg.bc);
      cfg.bwd_zero_kernel( &unary_param );
    }
    if (cfg.pool_type == LIBXSMM_DNN_POOLING_TYPE_MAX) {
      for ( ho = cfg.pad_h_out; ho < (cfg.ofh+cfg.pad_h_out); ho++ ) {
        for ( wo = cfg.pad_w_out; wo < (cfg.ofw+cfg.pad_w_out); wo++ ) {
          arg_array[0].primary    = (void*) &LIBXSMM_VLA_ACCESS(5, dinput, img, fm, 0, 0, 0, cfg.Bc, ifhp, ifwp, cfg.bc);
          arg_array[0].secondary  = (void*) &LIBXSMM_VLA_ACCESS(5, mask, img, fm, ho-cfg.pad_h_out, wo-cfg.pad_w_out, 0, cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc);
          arg_array[1].primary    = (void*) &LIBXSMM_VLA_ACCESS(5, doutput, img, fm, ho, wo, 0, cfg.Bc, ofhp, ofwp, cfg.bc);
          eqn_param.output        = arg_array[0];
          cfg.func_bwd_max_pool(&eqn_param);
        }
      }
    } else if (cfg.pool_type == LIBXSMM_DNN_POOLING_TYPE_AVG) {
      for ( ho = cfg.pad_h_out; ho < (cfg.ofh+cfg.pad_h_out); ho++ ) {
        hi = ((ho-cfg.pad_h_out) * cfg.u) - cfg.pad_h;
        for ( wo = cfg.pad_w_out; wo < (cfg.ofw+cfg.pad_w_out); wo++ ) {
          wi = ((wo-cfg.pad_w_out) * cfg.v) - cfg.pad_w;
          for ( kh = 0; kh < cfg.R; kh++ ) {
            if (hi+kh < 0 || hi+kh >= cfg.H) continue;
            for ( kw = 0; kw < cfg.S; kw++ ) {
              if (wi+kw < 0 || wi+kw >= cfg.W) {
                continue;
              } else {
                binary_param.in0.primary = (void*) &LIBXSMM_VLA_ACCESS(5, doutput, img, fm, ho, wo, 0, cfg.Bc, ofhp, ofwp, cfg.bc);
                binary_param.out.primary = (void*) &LIBXSMM_VLA_ACCESS(5, dinput, img, fm, hi+kh+cfg.pad_h_in, wi+kw+cfg.pad_w_in, 0, cfg.Bc, ifhp, ifwp, cfg.bc);
                cfg.func_bwd_avg_pool( &binary_param );
              }
            }
          }
        }
      }
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);
}

LIBXSMM_API void libxsmm_dnn_pooling_bwd_exec_bf16( const libxsmm_dnn_pooling_bwd_config cfg, libxsmm_bfloat16* din_act_ptr, const libxsmm_bfloat16* dout_act_ptr, const int* mask_ptr,
                               const libxsmm_blasint start_tid, const libxsmm_blasint my_tid, void* scratch ) {
  /* size variables, all const */
  const libxsmm_blasint ofhp = cfg.ofh + 2*cfg.pad_h_out;
  const libxsmm_blasint ofwp = cfg.ofw + 2*cfg.pad_w_out;
  const libxsmm_blasint ifhp = cfg.H   + 2*cfg.pad_h_in;
  const libxsmm_blasint ifwp = cfg.W   + 2*cfg.pad_w_in;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;
  /* number of tasks that could be run in parallel */
  const libxsmm_blasint work = cfg.N * cfg.Bc;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

  /* loop variables */
  libxsmm_blasint img = 0;
  libxsmm_blasint fm = 0;
  libxsmm_blasint imgfm = 0;
  libxsmm_blasint ho = 0;
  libxsmm_blasint wo = 0;
  libxsmm_blasint hi = 0;
  libxsmm_blasint wi = 0;
  libxsmm_blasint kh = 0;
  libxsmm_blasint kw = 0;
  float recp_pool_size_f32 = 1.0f/((float)cfg.R*(float)cfg.S);
  libxsmm_bfloat16 recp_pool_size;

  libxsmm_matrix_eqn_param eqn_param;
  libxsmm_matrix_arg arg_array[2];
  libxsmm_meltw_unary_param  unary_param;
  libxsmm_meltw_binary_param binary_param;

  /* multi-dim arrays declaration */
  LIBXSMM_VLA_DECL(5,       libxsmm_bfloat16,  dinput, din_act_ptr,  cfg.Bc,    ifhp,    ifwp, cfg.bc);
  LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16, doutput, dout_act_ptr, cfg.Bc,    ofhp,    ofwp, cfg.bc);
  LIBXSMM_VLA_DECL(5, const int,                 mask, mask_ptr,     cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc);
  eqn_param.inputs = arg_array;
  binary_param.in1.primary = (void*)&recp_pool_size;
  libxsmm_rne_convert_fp32_bf16( &recp_pool_size_f32, &recp_pool_size, 1 );

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  for (imgfm = thr_begin; imgfm < thr_end; ++imgfm) {
    img = imgfm / cfg.Bc;
    fm = imgfm % cfg.Bc;
    for ( hi = cfg.pad_h_in; hi < (cfg.H+cfg.pad_h_in); hi++ ) {
      unary_param.out.primary = (void*)&LIBXSMM_VLA_ACCESS(5, dinput, img, fm, hi, cfg.pad_w_in, 0, cfg.Bc, ifhp, ifwp, cfg.bc);
      cfg.bwd_zero_kernel( &unary_param );
    }
    if (cfg.pool_type == LIBXSMM_DNN_POOLING_TYPE_MAX) {
      for ( ho = cfg.pad_h_out; ho < (cfg.ofh+cfg.pad_h_out); ho++ ) {
        for ( wo = cfg.pad_w_out; wo < (cfg.ofw+cfg.pad_w_out); wo++ ) {
          arg_array[0].primary    = (void*) &LIBXSMM_VLA_ACCESS(5, dinput, img, fm, 0, 0, 0, cfg.Bc, ifhp, ifwp, cfg.bc);
          arg_array[0].secondary  = (void*) &LIBXSMM_VLA_ACCESS(5, mask, img, fm, ho-cfg.pad_h_out, wo-cfg.pad_w_out, 0, cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc);
          arg_array[1].primary    = (void*) &LIBXSMM_VLA_ACCESS(5, doutput, img, fm, ho, wo, 0, cfg.Bc, ofhp, ofwp, cfg.bc);
          eqn_param.output        = arg_array[0];
          cfg.func_bwd_max_pool(&eqn_param);
        }
      }
    } else if (cfg.pool_type == LIBXSMM_DNN_POOLING_TYPE_AVG) {
      for ( ho = cfg.pad_h_out; ho < (cfg.ofh+cfg.pad_h_out); ho++ ) {
        hi = ((ho-cfg.pad_h_out) * cfg.u) - cfg.pad_h;
        for ( wo = cfg.pad_w_out; wo < (cfg.ofw+cfg.pad_w_out); wo++ ) {
          wi = ((wo-cfg.pad_w_out) * cfg.v) - cfg.pad_w;
          for ( kh = 0; kh < cfg.R; kh++ ) {
            if (hi+kh < 0 || hi+kh >= cfg.H) continue;
            for ( kw = 0; kw < cfg.S; kw++ ) {
              if (wi+kw < 0 || wi+kw >= cfg.W) {
                continue;
              } else {
                binary_param.in0.primary = (void*) &LIBXSMM_VLA_ACCESS(5, doutput, img, fm, ho, wo, 0, cfg.Bc, ofhp, ofwp, cfg.bc);
                binary_param.out.primary = (void*) &LIBXSMM_VLA_ACCESS(5, dinput, img, fm, hi+kh+cfg.pad_h_in, wi+kw+cfg.pad_w_in, 0, cfg.Bc, ifhp, ifwp, cfg.bc);
                cfg.func_bwd_avg_pool( &binary_param );
              }
            }
          }
        }
      }
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);
}

LIBXSMM_API void destroy_libxsmm_dnn_pooling_fwd(libxsmm_dnn_pooling_fwd_config* cfg) {
  libxsmm_barrier_destroy(cfg->barrier);

}

LIBXSMM_API void destroy_libxsmm_dnn_pooling_bwd(libxsmm_dnn_pooling_bwd_config* cfg) {
  libxsmm_barrier_destroy(cfg->barrier);

}
