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

#include <libxsmm_dnn_opt.h>

LIBXSMM_API libxsmm_dnn_opt_config setup_libxsmm_dnn_opt(libxsmm_blasint C, libxsmm_blasint K, libxsmm_blasint bc, libxsmm_blasint bk,
                           libxsmm_blasint threads, float lr, libxsmm_datatype datatype_in,
                           libxsmm_datatype datatype_out, libxsmm_datatype datatype_comp) {
  libxsmm_dnn_opt_config res;

  /* setting up some handle values */
  res.C = C;
  res.K = K;
  res.bc = bc;
  res.bk = bk;
  res.threads = threads;
  res.lr = lr;

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* init scratch */
  if ( (datatype_in == LIBXSMM_DATATYPE_F32) &&
       (datatype_out == LIBXSMM_DATATYPE_F32) &&
       (datatype_comp == LIBXSMM_DATATYPE_F32) ) {
    res.scratch_size = 0;
  } else if ( (datatype_in == LIBXSMM_DATATYPE_BF16) &&
              (datatype_out == LIBXSMM_DATATYPE_BF16) &&
              (datatype_comp == LIBXSMM_DATATYPE_BF16) ) {
    res.scratch_size = 0;
  } else {
    fprintf( stderr, "Unsupported precision for smax bwd pass Bailing...!\n");
    exit(-1);
  }

  return res;
}

LIBXSMM_API void libxsmm_dnn_opt_exec_f32( libxsmm_dnn_opt_config cfg, float* wt_ptr, const float* delwt_ptr, int start_tid, int my_tid, void* scratch ) {
  /* loop counters */
  libxsmm_blasint i;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;

  /* number of tasks that could run in parallel for the filters */
  const libxsmm_blasint work = cfg.C * cfg.K;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

  /* lazy barrier init */
  libxsmm_barrier_init( cfg.barrier, ltid );

  for ( i = thr_begin; i < thr_end; ++i ) {
    wt_ptr[i] = wt_ptr[i] - (cfg.lr*delwt_ptr[i]);
  }

  libxsmm_barrier_wait( cfg.barrier, ltid );
}

LIBXSMM_API void libxsmm_dnn_opt_exec_bf16( libxsmm_dnn_opt_config cfg, libxsmm_bfloat16* wt_ptr, float* master_wt_ptr, const libxsmm_bfloat16* delwt_ptr, int start_tid, int my_tid, void* scratch ) {
  /* loop counters */
  libxsmm_blasint i;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;

  /* number of tasks that could run in parallel for the filters */
  const libxsmm_blasint work = cfg.C * cfg.K;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

  /* lazy barrier init */
  libxsmm_barrier_init( cfg.barrier, ltid );

  for ( i = thr_begin; i < thr_end; ++i ) {
    libxsmm_bfloat16_hp t1, t2;
    t1.i[0] =0;
    t1.i[1] = delwt_ptr[i];
    master_wt_ptr[i] = master_wt_ptr[i] - (cfg.lr*t1.f);
    t2.f = master_wt_ptr[i];
    wt_ptr[i] = t2.i[1];
  }

  libxsmm_barrier_wait( cfg.barrier, ltid );
}

