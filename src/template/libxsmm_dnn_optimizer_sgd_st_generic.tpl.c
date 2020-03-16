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

#if defined(LIBXSMM_DNN_OPTIMIZER_SGD_BF16_AVX512)
#endif

/* loop counters */
libxsmm_blasint i;

/* computing first logical thread */
const int ltid = tid - start_thread;

/* number of tasks that could run in parallel for the filters */
const int work = handle->desc.C * handle->desc.K;
/* compute chunk size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

element_filter_type*  filter = handle->reg_filter->data;
element_filter_type* dfilter = handle->grad_filter->data;
#if defined(LIBXSMM_DNN_OPTIMIZER_SGD_BF16) || defined(LIBXSMM_DNN_OPTIMIZER_SGD_BF16_AVX512)
element_master_type*  master = handle->master_filter->data;
#endif

/* lazy barrier init */
libxsmm_barrier_init( handle->barrier, ltid );

for ( i = thr_begin; i < thr_end; ++i ) {
#if defined(LIBXSMM_DNN_OPTIMIZER_SGD_BF16) || defined(LIBXSMM_DNN_OPTIMIZER_SGD_BF16_AVX512)
  libxsmm_bfloat16_hp t1, t2;
  t1.i[0] =0;
  t1.i[1] = dfilter[i];
  master[i] = master[i] - (handle->desc.learning_rate*t1.f);
  t2.f = master[i];
  filter[i] = t2.i[1];
#else
  filter[i] = filter[i] - (handle->desc.learning_rate*dfilter[i]);
#endif
}

libxsmm_barrier_wait( handle->barrier, ltid );

