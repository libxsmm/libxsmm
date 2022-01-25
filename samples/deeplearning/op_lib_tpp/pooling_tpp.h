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
  my_pooling_type  pool_type;
  my_pooling_pass  pass_type;
  size_t           scratch_size;
  libxsmm_barrier* barrier;
} my_pooling_bwd_config;

my_pooling_fwd_config setup_my_pooling_fwd( libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W,
                                            libxsmm_blasint R, libxsmm_blasint S,
                                            libxsmm_blasint stride_h, libxsmm_blasint stride_w,
                                            libxsmm_blasint pad_h, libxsmm_blasint pad_w,
                                            libxsmm_blasint pad_h_in, libxsmm_blasint pad_w_in,
                                            libxsmm_blasint pad_h_out, libxsmm_blasint pad_w_out,
                                            libxsmm_blasint bc, libxsmm_blasint threads, my_pooling_type pool_type ) {
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
  /* calculate scratch size for local pooling copies of one feature map block per thread */
  res.scratch_size = (sizeof(float) * ( (size_t)H + (size_t)LIBXSMM_MAX(pad_h_in, pad_h_out)*2 )
                                    * ( (size_t)W + (size_t)LIBXSMM_MAX(pad_w_in, pad_w_out)*2 )
                                    * bc * threads );

  return res;
}

my_pooling_bwd_config setup_my_pooling_bwd( libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W,
                                            libxsmm_blasint R, libxsmm_blasint S,
                                            libxsmm_blasint stride_h, libxsmm_blasint stride_w,
                                            libxsmm_blasint pad_h, libxsmm_blasint pad_w,
                                            libxsmm_blasint pad_h_in, libxsmm_blasint pad_w_in,
                                            libxsmm_blasint pad_h_out, libxsmm_blasint pad_w_out,
                                            libxsmm_blasint bc, libxsmm_blasint threads, my_pooling_type pool_type ) {
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
  /* calculate scratch size for local pooling copies of one feature map block per thread */
  res.scratch_size = (sizeof(float) * ( (size_t)H + (size_t)LIBXSMM_MAX(pad_h_in, pad_h_out)*2 )
                                    * ( (size_t)W + (size_t)LIBXSMM_MAX(pad_w_in, pad_w_out)*2 )
                                    * bc * threads );

  return res;
}

void my_pooling_fwd_exec_f32( my_pooling_fwd_config cfg, const float* in_act_ptr, float* out_act_ptr, libxsmm_blasint* mask_ptr,
                              libxsmm_blasint start_tid, libxsmm_blasint my_tid, void* scratch ) {
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
  LIBXSMM_VLA_DECL(5,       libxsmm_blasint,    mask,    mask_ptr, cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc);

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
              const libxsmm_blasint       idx = (hi+kh)*cfg.W*cfg.bc + (wi+kw)*cfg.bc;
                    libxsmm_blasint* mask_ptr = &LIBXSMM_VLA_ACCESS(5, mask, img, fm, ho-cfg.pad_h_out, wo-cfg.pad_w_out, 0, cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc);
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

void my_pooling_fwd_exec_bf16( my_pooling_fwd_config cfg, const libxsmm_bfloat16* in_act_ptr, libxsmm_bfloat16* out_act_ptr, libxsmm_blasint* mask_ptr,
                               int start_tid, int my_tid, void* scratch ) {

}

void my_pooling_bwd_exec_f32( my_pooling_bwd_config cfg, float* din_act_ptr, const float* dout_act_ptr, const libxsmm_blasint* mask_ptr,
                              int start_tid, int my_tid, void* scratch ) {
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
  LIBXSMM_VLA_DECL(5, const libxsmm_blasint,    mask, mask_ptr,     cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc);

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
          const libxsmm_blasint*    mask_ptr = &LIBXSMM_VLA_ACCESS(5, mask, img, fm, ho-cfg.pad_h_out, wo-cfg.pad_w_out, 0, cfg.Bc, cfg.ofh, cfg.ofw, cfg.bc);

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

void my_pooling_bwd_exec_bf16( my_pooling_bwd_config cfg, libxsmm_bfloat16* din_act_ptr, const libxsmm_bfloat16* dout_act_ptr, const libxsmm_blasint* mask_ptr,
                               int start_tid, int my_tid, void* scratch ) {

}



