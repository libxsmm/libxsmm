/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <libxsmm_sync.h>

typedef enum my_pooling_pass {
  MY_POOLING_PASS_FWD = 1,
  MY_POOLING_PASS_BWD = 2
} my_pooling_pass;

typedef enum my_pooling_type {
  MY_POOLING_TYPE_AVG = 1,
  MY_POOLING_TYPE_MAX = 2
} my_pooling_type;

typedef struct my_pooling_fwd_config {
  libxsmm_blasint  N;
  libxsmm_blasint  C;
  libxsmm_blasint  H;
  libxsmm_blasint  W;
  libxsmm_blasint  R;
  libxsmm_blasint  S;
  libxsmm_blasint  bc;
  libxsmm_blasint  Bc;
  libxsmm_blasint  ofh;
  libxsmm_blasint  ofw;
  libxsmm_blasint  u;
  libxsmm_blasint  v;
  libxsmm_blasint  pad_h;
  libxsmm_blasint  pad_w;
  libxsmm_blasint  pad_h_in;
  libxsmm_blasint  pad_w_in;
  libxsmm_blasint  pad_h_out;
  libxsmm_blasint  pad_w_out;
  libxsmm_blasint  threads;
  libxsmm_datatype datatype_in;
  libxsmm_datatype datatype_out;
  libxsmm_datatype datatype_comp;
  my_pooling_type  pool_type;
  my_pooling_pass  pass_type;
  size_t           scratch_size;
  libxsmm_barrier* barrier;
} my_pooling_fwd_config;

typedef struct my_pooling_bwd_config {
  libxsmm_blasint  N;
  libxsmm_blasint  C;
  libxsmm_blasint  H;
  libxsmm_blasint  W;
  libxsmm_blasint  R;
  libxsmm_blasint  S;
  libxsmm_blasint  bc;
  libxsmm_blasint  Bc;
  libxsmm_blasint  ofh;
  libxsmm_blasint  ofw;
  libxsmm_blasint  u;
  libxsmm_blasint  v;
  libxsmm_blasint  pad_h;
  libxsmm_blasint  pad_w;
  libxsmm_blasint  pad_h_in;
  libxsmm_blasint  pad_w_in;
  libxsmm_blasint  pad_h_out;
  libxsmm_blasint  pad_w_out;
  libxsmm_blasint  threads;
  libxsmm_datatype datatype_in;
  libxsmm_datatype datatype_out;
  libxsmm_datatype datatype_comp;
  my_pooling_type  pool_type;
  my_pooling_pass  pass_type;
  size_t           scratch_size;
  libxsmm_barrier* barrier;
} my_pooling_bwd_config;

my_pooling_fwd_config setup_my_pooling_fwd( const libxsmm_blasint N, const libxsmm_blasint C, const libxsmm_blasint H, const libxsmm_blasint W,
                                            const libxsmm_blasint R, const libxsmm_blasint S,
                                            const libxsmm_blasint stride_h, const libxsmm_blasint stride_w,
                                            const libxsmm_blasint pad_h, const libxsmm_blasint pad_w,
                                            const libxsmm_blasint pad_h_in, const libxsmm_blasint pad_w_in,
                                            const libxsmm_blasint pad_h_out, const libxsmm_blasint pad_w_out,
                                            const libxsmm_blasint bc, const libxsmm_blasint threads, const my_pooling_type pool_type,
                                            const libxsmm_datatype datatype_in, const libxsmm_datatype datatype_out, const libxsmm_datatype datatype_comp ) {
  my_pooling_fwd_config res;

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
  res.pass_type = MY_POOLING_PASS_FWD;
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
  res.scratch_size = (sizeof(float) * ( (size_t)H + (size_t)LIBXSMM_MAX(pad_h_in, pad_h_out)*2 )
                                    * ( (size_t)W + (size_t)LIBXSMM_MAX(pad_w_in, pad_w_out)*2 )
                                    * bc * threads );

  return res;
}

my_pooling_bwd_config setup_my_pooling_bwd( const libxsmm_blasint N, const libxsmm_blasint C, const libxsmm_blasint H, const libxsmm_blasint W,
                                            const libxsmm_blasint R, const libxsmm_blasint S,
                                            const libxsmm_blasint stride_h, const libxsmm_blasint stride_w,
                                            const libxsmm_blasint pad_h, const libxsmm_blasint pad_w,
                                            const libxsmm_blasint pad_h_in, const libxsmm_blasint pad_w_in,
                                            const libxsmm_blasint pad_h_out, const libxsmm_blasint pad_w_out,
                                            const libxsmm_blasint bc, const libxsmm_blasint threads, const my_pooling_type pool_type,
                                            const libxsmm_datatype datatype_in, const libxsmm_datatype datatype_out, const libxsmm_datatype datatype_comp ) {
  my_pooling_bwd_config res;

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
  res.pass_type = MY_POOLING_PASS_FWD;
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
  res.scratch_size = (sizeof(float) * ( (size_t)H + (size_t)LIBXSMM_MAX(pad_h_in, pad_h_out)*2 )
                                    * ( (size_t)W + (size_t)LIBXSMM_MAX(pad_w_in, pad_w_out)*2 )
                                    * bc * threads );

  return res;
}

void my_pooling_fwd_exec_f32( const my_pooling_fwd_config cfg, const float* in_act_ptr, float* out_act_ptr, int* mask_ptr,
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
  libxsmm_blasint v = 0;

  /* only for average pooling */
  float recp_pool_size = 1.0f/((float)cfg.R*(float)cfg.S);

  /* multi-dim arrays declaration */
  float *const lcl_buffer_ptr = (float*)scratch + (size_t)cfg.ofh*cfg.ofw*cfg.bc*ltid;
  LIBXSMM_VLA_DECL(3,                 float, lcl_output, lcl_buffer_ptr,            cfg.ofw, cfg.bc);
  LIBXSMM_VLA_DECL(5, const float,             input,  in_act_ptr, cfg.Bc,    ifhp,    ifwp, cfg.bc);
  LIBXSMM_VLA_DECL(5,       float,            output, out_act_ptr, cfg.Bc,    ofhp,    ofwp, cfg.bc);
  LIBXSMM_VLA_DECL(5,       int,                mask,    mask_ptr, cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc);

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  for (imgfm = thr_begin; imgfm < thr_end; ++imgfm) {
    img = imgfm / cfg.Bc;
    fm = imgfm % cfg.Bc;

    LIBXSMM_PRAGMA_SIMD
    for ( v = 0; v < cfg.ofh*cfg.ofw*cfg.bc; v++ ) {
      if ( cfg.pool_type == MY_POOLING_TYPE_MAX ) {
        lcl_buffer_ptr[v] = -FLT_MAX;
      } else if ( cfg.pool_type == MY_POOLING_TYPE_AVG ) {
        lcl_buffer_ptr[v] = (float)0.0f;
      }
    }

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
              const float*          input_ptr = &LIBXSMM_VLA_ACCESS(5, input, img, fm, hi+kh+cfg.pad_h_in, wi+kw+cfg.pad_w_in, 0, cfg.Bc, ifhp, ifwp, cfg.bc);
                    float*     lcl_output_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_output, ho-cfg.pad_h_out, wo-cfg.pad_w_out, 0, cfg.ofw, cfg.bc);
              const int                   idx = (hi+kh)*cfg.W*cfg.bc + (wi+kw)*cfg.bc;
                                int* mask_ptr = &LIBXSMM_VLA_ACCESS(5, mask, img, fm, ho-cfg.pad_h_out, wo-cfg.pad_w_out, 0, cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc);
              LIBXSMM_PRAGMA_SIMD
              for ( v = 0; v < cfg.bc; v++ ) {
                if ( cfg.pool_type == MY_POOLING_TYPE_MAX ) {
                  if ( input_ptr[v] > lcl_output_ptr[v] ) {
                    lcl_output_ptr[v] =  input_ptr[v];
                    mask_ptr[v] = idx + v;
                  }
                } else if ( cfg.pool_type == MY_POOLING_TYPE_AVG ) {
                  lcl_output_ptr[v] += input_ptr[v];
                }
              }
            }
          }
        }
      }
    }

    /* copy the local buffer into output activations */
    for ( ho = cfg.pad_h_out; ho < (cfg.ofh+cfg.pad_h_out); ho++ ) {
      for ( wo = cfg.pad_w_out; wo < (cfg.ofw+cfg.pad_w_out); wo++ ) {
        float*     output_ptr = &LIBXSMM_VLA_ACCESS(5, output, img, fm, ho, wo, 0, cfg.Bc, ofhp, ofwp, cfg.bc);
        float* lcl_output_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_output, ho-cfg.pad_h_out, wo-cfg.pad_w_out, 0, cfg.ofw, cfg.bc);

        LIBXSMM_PRAGMA_SIMD
        for ( v = 0; v < cfg.bc; v++ ) {
          if (cfg.pool_type == MY_POOLING_TYPE_MAX) {
            output_ptr[v] = lcl_output_ptr[v];
          } else if ( cfg.pool_type == MY_POOLING_TYPE_AVG ) {
            output_ptr[v] = lcl_output_ptr[v] * recp_pool_size;
          }
        }
      }
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);
}

void my_pooling_fwd_exec_bf16( const my_pooling_fwd_config cfg, const libxsmm_bfloat16* in_act_ptr, libxsmm_bfloat16* out_act_ptr, int* mask_ptr,
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
  libxsmm_blasint v = 0;

  /* only for average pooling */
  float recp_pool_size = 1.0f/((float)cfg.R*(float)cfg.S);

  /* multi-dim arrays declaration */
  float *const lcl_buffer_ptr = (float*)scratch + (size_t)cfg.ofh*cfg.ofw*cfg.bc*ltid;
  LIBXSMM_VLA_DECL(3,                  float, lcl_output, lcl_buffer_ptr,            cfg.ofw, cfg.bc);
  LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16,             input,  in_act_ptr, cfg.Bc,    ifhp,    ifwp, cfg.bc);
  LIBXSMM_VLA_DECL(5,       libxsmm_bfloat16,            output, out_act_ptr, cfg.Bc,    ofhp,    ofwp, cfg.bc);
  LIBXSMM_VLA_DECL(5,                    int,              mask,    mask_ptr, cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc);

  union libxsmm_bfloat16_hp input_f32;
  input_f32.i[1] = 0;
  input_f32.i[0] = 0;

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  for (imgfm = thr_begin; imgfm < thr_end; ++imgfm) {
    img = imgfm / cfg.Bc;
    fm = imgfm % cfg.Bc;

    LIBXSMM_PRAGMA_SIMD
    for ( v = 0; v < cfg.ofh*cfg.ofw*cfg.bc; v++ ) {
      if ( cfg.pool_type == MY_POOLING_TYPE_MAX ) {
        lcl_buffer_ptr[v] = -FLT_MAX;
      } else if ( cfg.pool_type == MY_POOLING_TYPE_AVG ) {
        lcl_buffer_ptr[v] = (float)0.0f;
      }
    }

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
              const libxsmm_bfloat16* input_ptr = &LIBXSMM_VLA_ACCESS(5, input, img, fm, hi+kh+cfg.pad_h_in, wi+kw+cfg.pad_w_in, 0, cfg.Bc, ifhp, ifwp, cfg.bc);
                    float*       lcl_output_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_output, ho-cfg.pad_h_out, wo-cfg.pad_w_out, 0, cfg.ofw, cfg.bc);
              const int                     idx = (hi+kh)*cfg.W*cfg.bc + (wi+kw)*cfg.bc;
                    int*               mask_ptr = &LIBXSMM_VLA_ACCESS(5, mask, img, fm, ho-cfg.pad_h_out, wo-cfg.pad_w_out, 0, cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc);
              LIBXSMM_PRAGMA_SIMD
              for ( v = 0; v < cfg.bc; v++ ) {
                input_f32.i[1] = input_ptr[v];
                if ( cfg.pool_type == MY_POOLING_TYPE_MAX ) {
                  if ( input_f32.f > lcl_output_ptr[v] ) {
                    lcl_output_ptr[v] =  input_f32.f;
                    mask_ptr[v] = idx + v;
                  }
                } else if ( cfg.pool_type == MY_POOLING_TYPE_AVG ) {
                  lcl_output_ptr[v] += input_f32.f;
                }
              }
            }
          }
        }
      }
    }

    /* copy the local buffer into output activations */
    for ( ho = cfg.pad_h_out; ho < (cfg.ofh+cfg.pad_h_out); ho++ ) {
      for ( wo = cfg.pad_w_out; wo < (cfg.ofw+cfg.pad_w_out); wo++ ) {
        libxsmm_bfloat16* output_ptr = &LIBXSMM_VLA_ACCESS(5, output, img, fm, ho, wo, 0, cfg.Bc, ofhp, ofwp, cfg.bc);
        float*        lcl_output_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_output, ho-cfg.pad_h_out, wo-cfg.pad_w_out, 0, cfg.ofw, cfg.bc);

        LIBXSMM_PRAGMA_SIMD
        for ( v = 0; v < cfg.bc; v++ ) {
          if (cfg.pool_type == MY_POOLING_TYPE_MAX) {
            libxsmm_rne_convert_fp32_bf16( &(lcl_output_ptr[v]), &(output_ptr[v]), 1 );
          } else if ( cfg.pool_type == MY_POOLING_TYPE_AVG ) {
            float l_temp = lcl_output_ptr[v] * recp_pool_size;
            libxsmm_rne_convert_fp32_bf16( &l_temp, &(output_ptr[v]), 1 );
          }
        }
      }
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);
}

void my_pooling_bwd_exec_f32( const my_pooling_bwd_config cfg, float* din_act_ptr, const float* dout_act_ptr, const int* mask_ptr,
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
  libxsmm_blasint v = 0;
  libxsmm_blasint kh = 0;
  libxsmm_blasint kw = 0;
  float recp_pool_size = 1.0f/((float)cfg.R*(float)cfg.S);

  /* multi-dim arrays declaration */
  float *const lcl_buffer_ptr = (float*)scratch + (size_t)cfg.H*cfg.W*cfg.bc*ltid;
  LIBXSMM_VLA_DECL(3,       float,        lcl_dinput, lcl_buffer_ptr,                  cfg.W, cfg.bc);
  LIBXSMM_VLA_DECL(5,       float,            dinput, din_act_ptr,  cfg.Bc,    ifhp,    ifwp, cfg.bc);
  LIBXSMM_VLA_DECL(5, const float,           doutput, dout_act_ptr, cfg.Bc,    ofhp,    ofwp, cfg.bc);
  LIBXSMM_VLA_DECL(5, const int,                mask, mask_ptr,     cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc);

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  for (imgfm = thr_begin; imgfm < thr_end; ++imgfm) {
    img = imgfm / cfg.Bc;
    fm = imgfm % cfg.Bc;

    LIBXSMM_PRAGMA_SIMD
    for ( v = 0; v < cfg.H*cfg.W*cfg.bc; v++ ) {
      lcl_buffer_ptr[v] = (float)0;
    }

    if (cfg.pool_type == MY_POOLING_TYPE_MAX) {
      for ( ho = cfg.pad_h_out; ho < (cfg.ofh+cfg.pad_h_out); ho++ ) {
        for ( wo = cfg.pad_w_out; wo < (cfg.ofw+cfg.pad_w_out); wo++ ) {
          const float*           doutput_ptr = &LIBXSMM_VLA_ACCESS(5, doutput, img, fm, ho, wo, 0, cfg.Bc, ofhp, ofwp, cfg.bc);
          const int*                mask_ptr = &LIBXSMM_VLA_ACCESS(5, mask, img, fm, ho-cfg.pad_h_out, wo-cfg.pad_w_out, 0, cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc);

          LIBXSMM_PRAGMA_SIMD
          for ( v = 0; v < cfg.bc; v++ ) {
            lcl_buffer_ptr[mask_ptr[v]] += doutput_ptr[v];
          }
        }
      }
    } else if (cfg.pool_type == MY_POOLING_TYPE_AVG) {
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
                const float*    doutput_ptr = &LIBXSMM_VLA_ACCESS(5, doutput, img, fm, ho, wo, 0, cfg.Bc, ofhp, ofwp, cfg.bc);
                      float* lcl_dinput_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_dinput, hi+kh, wi+kw, 0, cfg.W, cfg.bc);

                LIBXSMM_PRAGMA_SIMD
                for ( v = 0; v < cfg.bc; v++ ) {
                  lcl_dinput_ptr[v] += (doutput_ptr[v] * recp_pool_size);
                }
              }
            }
          }
        }
      }
    }

    /* copy the local buffer into dinput activations */
    for ( hi = cfg.pad_h_in; hi < (cfg.H+cfg.pad_h_in); hi++ ) {
      for ( wi = cfg.pad_w_in; wi < (cfg.W+cfg.pad_w_in); wi++ ) {
        float*     dinput_ptr = &LIBXSMM_VLA_ACCESS(5, dinput, img, fm, hi, wi, 0, cfg.Bc, ifhp, ifwp, cfg.bc);
        float* lcl_dinput_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_dinput, hi-cfg.pad_h_in, wi-cfg.pad_w_in, 0, cfg.W, cfg.bc);

        LIBXSMM_PRAGMA_SIMD
        for ( v = 0; v < cfg.bc; v++ ) {
          dinput_ptr[v] = lcl_dinput_ptr[v];
        }
      }
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);
}

void my_pooling_bwd_exec_bf16( const my_pooling_bwd_config cfg, libxsmm_bfloat16* din_act_ptr, const libxsmm_bfloat16* dout_act_ptr, const int* mask_ptr,
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
  libxsmm_blasint v = 0;
  libxsmm_blasint kh = 0;
  libxsmm_blasint kw = 0;
  float recp_pool_size = 1.0f/((float)cfg.R*(float)cfg.S);

  /* multi-dim arrays declaration */
  float *const lcl_buffer_ptr = (float*)scratch + (size_t)cfg.H*cfg.W*cfg.bc*ltid;
  LIBXSMM_VLA_DECL(3,       float,         lcl_dinput, lcl_buffer_ptr,                  cfg.W, cfg.bc);
  LIBXSMM_VLA_DECL(5,       libxsmm_bfloat16,  dinput, din_act_ptr,  cfg.Bc,    ifhp,    ifwp, cfg.bc);
  LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16, doutput, dout_act_ptr, cfg.Bc,    ofhp,    ofwp, cfg.bc);
  LIBXSMM_VLA_DECL(5, const int,                 mask, mask_ptr,     cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc);

  union libxsmm_bfloat16_hp doutput_f32;
  doutput_f32.i[1] = 0;
  doutput_f32.i[0] = 0;

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  for (imgfm = thr_begin; imgfm < thr_end; ++imgfm) {
    img = imgfm / cfg.Bc;
    fm = imgfm % cfg.Bc;

    LIBXSMM_PRAGMA_SIMD
    for ( v = 0; v < cfg.H*cfg.W*cfg.bc; v++ ) {
      lcl_buffer_ptr[v] = (float)0;
    }

    if (cfg.pool_type == MY_POOLING_TYPE_MAX) {
      for ( ho = cfg.pad_h_out; ho < (cfg.ofh+cfg.pad_h_out); ho++ ) {
        for ( wo = cfg.pad_w_out; wo < (cfg.ofw+cfg.pad_w_out); wo++ ) {
          const libxsmm_bfloat16* doutput_ptr = &LIBXSMM_VLA_ACCESS(5, doutput, img, fm, ho, wo, 0, cfg.Bc, ofhp, ofwp, cfg.bc);
          const int*                 mask_ptr = &LIBXSMM_VLA_ACCESS(5, mask, img, fm, ho-cfg.pad_h_out, wo-cfg.pad_w_out, 0, cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc);

          LIBXSMM_PRAGMA_SIMD
          for ( v = 0; v < cfg.bc; v++ ) {
            doutput_f32.i[1] = doutput_ptr[v];
            lcl_buffer_ptr[mask_ptr[v]] += doutput_f32.f;
          }
        }
      }
    } else if (cfg.pool_type == MY_POOLING_TYPE_AVG) {
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
                const libxsmm_bfloat16* doutput_ptr = &LIBXSMM_VLA_ACCESS(5, doutput, img, fm, ho, wo, 0, cfg.Bc, ofhp, ofwp, cfg.bc);
                      float*         lcl_dinput_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_dinput, hi+kh, wi+kw, 0, cfg.W, cfg.bc);

                LIBXSMM_PRAGMA_SIMD
                for ( v = 0; v < cfg.bc; v++ ) {
                  doutput_f32.i[1] = doutput_ptr[v];
                  lcl_dinput_ptr[v] += (doutput_f32.f * recp_pool_size);
                }
              }
            }
          }
        }
      }
    }

    /* copy the local buffer into dinput activations */
    for ( hi = cfg.pad_h_in; hi < (cfg.H+cfg.pad_h_in); hi++ ) {
      for ( wi = cfg.pad_w_in; wi < (cfg.W+cfg.pad_w_in); wi++ ) {
        libxsmm_bfloat16* dinput_ptr = &LIBXSMM_VLA_ACCESS(5, dinput, img, fm, hi, wi, 0, cfg.Bc, ifhp, ifwp, cfg.bc);
        float*        lcl_dinput_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_dinput, hi-cfg.pad_h_in, wi-cfg.pad_w_in, 0, cfg.W, cfg.bc);

        LIBXSMM_PRAGMA_SIMD
        for ( v = 0; v < cfg.bc; v++ ) {
          libxsmm_rne_convert_fp32_bf16( &(lcl_dinput_ptr[v]), &(dinput_ptr[v]), 1 );
        }
      }
    }
  }

  libxsmm_barrier_wait(cfg.barrier, ltid);
}

void my_pooling_fwd_exec( const my_pooling_fwd_config cfg, const void* in_act_ptr, void* out_act_ptr, int* mask_ptr,
                          const libxsmm_blasint start_tid, const libxsmm_blasint my_tid, void* scratch ) {
  if ( (cfg.datatype_in == LIBXSMM_DATATYPE_F32) && (cfg.datatype_out == LIBXSMM_DATATYPE_F32) && (cfg.datatype_comp == LIBXSMM_DATATYPE_F32) ) {
    my_pooling_fwd_exec_f32( cfg, (const float*)in_act_ptr, (float*)out_act_ptr, mask_ptr, start_tid, my_tid, scratch );
  } else if ( (cfg.datatype_in == LIBXSMM_DATATYPE_BF16) && (cfg.datatype_out == LIBXSMM_DATATYPE_BF16) && (cfg.datatype_comp == LIBXSMM_DATATYPE_F32) ) {
    my_pooling_fwd_exec_bf16( cfg, (const libxsmm_bfloat16*)in_act_ptr, (libxsmm_bfloat16*)out_act_ptr, mask_ptr, start_tid, my_tid, scratch );
  } else {
    /* shouldn't happen */
  }
}

void my_pooling_bwd_exec( const my_pooling_bwd_config cfg, void* din_act_ptr, const void* dout_act_ptr, const int* mask_ptr,
                          const libxsmm_blasint start_tid, const libxsmm_blasint my_tid, void* scratch ) {
  if ( (cfg.datatype_in == LIBXSMM_DATATYPE_F32) && (cfg.datatype_out == LIBXSMM_DATATYPE_F32) && (cfg.datatype_comp == LIBXSMM_DATATYPE_F32) ) {
    my_pooling_bwd_exec_f32( cfg, (float*)din_act_ptr, (const float*)dout_act_ptr, mask_ptr, start_tid, my_tid, scratch );
  } else if ( (cfg.datatype_in == LIBXSMM_DATATYPE_BF16) && (cfg.datatype_out == LIBXSMM_DATATYPE_BF16) && (cfg.datatype_comp == LIBXSMM_DATATYPE_F32) ) {
    my_pooling_bwd_exec_bf16( cfg, (libxsmm_bfloat16*)din_act_ptr, (const libxsmm_bfloat16*)dout_act_ptr, mask_ptr, start_tid, my_tid, scratch );
  } else {
    /* shouldn't happen */
  }
}

