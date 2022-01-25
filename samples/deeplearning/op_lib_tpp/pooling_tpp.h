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
#include "libxsmm_dnn_pooling_backward.h"
#include "libxsmm_dnn_pooling_forward.h"
#include "libxsmm_main.h"

typedef enum my_pooling_pass {
  MY_POOLING_PASS_FWD = 1,
  MY_POOLING_PASS_BWD = 2
} my_pooling_pass;

typedef enum my_pooling_type {
  MY_POOLING_TYPE_AVG = 1,
  MY_POOLINT_TYPE_MAX = 2
} my_pooling_type;

typedef struct my_pooling_fwd_config {
  libxsmm_blasint  N;
  libxsmm_blasint  C;
  libxsmm_blasint  H;
  libxsmm_blasint  W;
  libxsmm_blasint  bc;
  libxsmm_blasint  Bc;
  libxsmm_blasint  ofh;
  libxsmm_blasint  ofw;
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
  libxsmm_blasint  bc;
  libxsmm_blasint  Bc;
  libxsmm_blasint  ofh;
  libxsmm_blasint  ofw;
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
  res.bc = bc;
  res.Bc = C / bc;
  res.pool_type = pool_type;
  res.pass_type = MY_POOLING_PASS_FWD;
  res.pad_h = pad_h;
  res.pad_w = pad_w;
  res.pad_h_in = pad_h_in;
  res.pad_w_in = pad_w_in;
  res.pad_h_out = pad_h_out;
  res.pad_w_out = pad_w_out;

  /* setting ofh and ofw */
  res.ofh = (handle->desc.H + 2 * handle->desc.pad_h - handle->desc.R) / handle->desc.u + 1;
  res.ofw = (handle->desc.W + 2 * handle->desc.pad_w - handle->desc.S) / handle->desc.v + 1;
   /* create barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);
  /* calculate scratch size for local pooling copies of one feature map block per thread */
  res.scratch_size = (sizeof(float) * ( (size_t)H + (size_t)LIBXSMM_MAX(pad_h_in, pad_h_out)*2 )
                                    * ( (size_t)W + (size_t)LIBXSMM_MAX(pad_w_in, pad_w_out)*2 )
                                    * bc * threads );

  return res;
}

my_pooling_bwd_config setup_my_pooling_bwd( libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W,
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
  res.bc = bc;
  res.Bc = C / bc;
  res.pool_type = pool_type;
  res.pass_type = MY_POOLING_PASS_FWD;
  res.pad_h = pad_h;
  res.pad_w = pad_w;
  res.pad_h_in = pad_h_in;
  res.pad_w_in = pad_w_in;
  res.pad_h_out = pad_h_out;
  res.pad_w_out = pad_w_out;

  /* setting ofh and ofw */
  res.ofh = (handle->desc.H + 2 * handle->desc.pad_h - handle->desc.R) / handle->desc.u + 1;
  res.ofw = (handle->desc.W + 2 * handle->desc.pad_w - handle->desc.S) / handle->desc.v + 1;
   /* create barrier */
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
  const int nImg = handle->desc.N;
  const int ifh = handle->desc.H;
  const int ifw = handle->desc.W;
  const int sh = handle->desc.u;
  const int sw = handle->desc.v;
  const int ofh = handle->ofh;
const int ofw = handle->ofw;
const int iph = handle->desc.pad_h_in;
const int ipw = handle->desc.pad_w_in;
const int oph = handle->desc.pad_h_out;
const int opw = handle->desc.pad_w_out;
const int ofhp = ofh + 2*oph;
const int ofwp = ofw + 2*opw;
const int ifhp = ifh + 2*iph;
const int ifwp = ifw + 2*ipw;
/* here we assume that input and output blocking is similar */
const int nBlocksFm = handle->blocksifm;
const int nFmBlock = handle->ifmblock;

  /* computing first logical thread */
  const libxsmm_blasint ltid = tid - start_thread;
  /* number of tasks that could be run in parallel */
  const libxsmm_blasint work = cfg.N * cfg.Bc;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
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
  float recp_pool_size = 1.0f/((float)handle->desc.R*(float)handle->desc.S);

  /* multi-dim arrays declaration */
  float *const lcl_buffer_ptr = (float*)handle->scratch + (size_t)ofh*ofw*nFmBlock*ltid;
  LIBXSMM_VLA_DECL(3,                 float, lcl_output, lcl_buffer_ptr, ofw, nFmBlock);
  LIBXSMM_VLA_DECL(5, const float,             input,  in_act_ptr, nBlocksFm, ifhp, ifwp, nFmBlock);
  LIBXSMM_VLA_DECL(5,       float,            output, out_act_ptr, nBlocksFm, ofhp, ofwp, nFmBlock);
  LIBXSMM_VLA_DECL(5,       libxsmm_blasint,    mask,    mask_ptr, nBlocksFm,  ofh,  ofw, nFmBlock);

  /* lazy barrier init */
  libxsmm_barrier_init(handle->barrier, ltid);

  for (imgfm = thr_begin; imgfm < thr_end; ++imgfm) {
    img = imgfm / nBlocksFm;
    fm = imgfm % nBlocksFm;

    LIBXSMM_PRAGMA_SIMD
    for ( v = 0; v < ofh*ofw*nFmBlock; v++ ) {
      if ( cfg.pool_type == MY_POOLINT_TYPE_MAX ) {
        lcl_buffer_ptr[v] = -FLT_MAX;
      } else if ( cfg.pool_type == MY_POOLINT_TYPE_AVG ) {
        lcl_buffer_ptr[v] = (float)0.0f;
      }
    }

    for ( ho = oph; ho < (ofh+oph); ho++ ) {
      hi = ((ho-oph) * sh) - handle->desc.pad_h;
      for ( wo = opw; wo < (ofw+opw); wo++ ) {
        wi = ((wo-opw) * sw) - handle->desc.pad_w;
        for ( kh = 0; kh < handle->desc.R; kh++ ) {
          if (hi+kh < 0 || hi+kh >= ifh) continue;
          for ( kw = 0; kw < handle->desc.S; kw++ ) {
            if (wi+kw < 0 || wi+kw >= ifw) {
              continue;
            } else {
              const float*          input_ptr = &LIBXSMM_VLA_ACCESS(5, input,      img, fm, hi+kh+iph, wi+kw+ipw, 0, nBlocksFm, ifhp, ifwp, nFmBlock);
                    float*     lcl_output_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_output,             ho-oph,    wo-opw, 0,                   ofw, nFmBlock);
              const libxsmm_blasint       idx = (hi+kh)*ifw*nFmBlock + (wi+kw)*nFmBlock;
                    libxsmm_blasint* mask_ptr = &LIBXSMM_VLA_ACCESS(5, mask,       img, fm,    ho-oph,    wo-opw, 0, nBlocksFm,  ofh,  ofw, nFmBlock);
              LIBXSMM_PRAGMA_SIMD
              for ( v = 0; v < nFmBlock; v++ ) {
                if ( cfg.pool_type == MY_POOLINT_TYPE_MAX ) {
                  if ( input_ptr[v] > lcl_output_ptr[v] ) {
                    lcl_output_ptr[v] =  input_ptr[v];
                    mask_ptr[v] = idx + v;
                  }
                } else if ( cfg.pool_type == MY_POOLINT_TYPE_AVG ) {
                  lcl_output_ptr[v] += input_ptr[v];
                }
              }
            }
          }
        }
      }
    }

    /* copy the local buffer into output activations */
    for ( ho = oph; ho < (ofh+oph); ho++ ) {
      for ( wo = opw; wo < (ofw+opw); wo++ ) {
        float*     output_ptr = &LIBXSMM_VLA_ACCESS(5, output,     img, fm,        ho,        wo, 0, nBlocksFm, ofhp, ofwp, nFmBlock);
        float* lcl_output_ptr = &LIBXSMM_VLA_ACCESS(3, lcl_output,             ho-oph,    wo-opw, 0,                   ofw, nFmBlock);

        LIBXSMM_PRAGMA_SIMD
        for ( v = 0; v < nFmBlock; v++ ) {
          if (cfg.pool_type == MY_POOLINT_TYPE_MAX) {
            output_ptr[v] = lcl_output_ptr[v];
          } else if ( cfg.pool_type == MY_POOLINT_TYPE_AVG ) {
            output_ptr[v] = lcl_output_ptr[v] * recp_pool_size;
          }
        }
      }
    }
  }

  libxsmm_barrier_wait(handle->barrier, ltid);
}

void my_pooling_fwd_exec_bf16( my_pooling_fwd_config cfg, const libxsmm_bfloat16* in_act_ptr, libxsmm_bfloat16* out_act_ptr, libxsmm_blasint* mask_ptr,
                               int start_tid, int my_tid, void* scratch ) {

}

void my_pooling_bwd_exec_f32( my_pooling_bwd_config cfg, float* din_act_ptr, const float* dout_act_ptr, const libxsmm_blasint* mask_ptr,
                              int start_tid, int my_tid, void* scratch ) {

}

void my_pooling_bwd_exec_bf16( my_pooling_bwd_config cfg, libxsmm_bfloat16* din_act_ptr, const libxsmm_bfloat16* dout_act_ptr, const libxsmm_blasint* mask_ptr,
                               int start_tid, int my_tid, void* scratch ) {

}



