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
# define _mm512_load_fil(A)   _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(A))),16))
# define _mm512_store_fil(A,B)  _mm256_storeu_si256((__m256i*)(A),_mm512_cvtepi32_epi16(_mm512_srai_epi32(LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16((B)),16)))
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

element_filter_type*  filter = (element_filter_type*)handle->reg_filter->data;
element_filter_type* dfilter = (element_filter_type*)handle->grad_filter->data;
#if defined(LIBXSMM_DNN_OPTIMIZER_SGD_BF16) || defined(LIBXSMM_DNN_OPTIMIZER_SGD_BF16_AVX512)
element_master_type*  master = (element_master_type*)handle->master_filter->data;
#endif

/* lazy barrier init */
libxsmm_barrier_init( handle->barrier, ltid );

#if defined(LIBXSMM_DNN_OPTIMIZER_SGD_BF16) || defined(LIBXSMM_DNN_OPTIMIZER_SGD_BF16_AVX512)
#if defined(LIBXSMM_DNN_OPTIMIZER_SGD_BF16_AVX512)
{
  libxsmm_blasint iv = ( (thr_end-thr_begin)/16 ) * 16; /* compute iterations which are vectorizable */
  __m512 vlr = _mm512_set1_ps( handle->desc.learning_rate );
  for ( i = thr_begin; i < iv; i+=16 ) {
    __m512 newfilter = _mm512_sub_ps( _mm512_loadu_ps( master+i ), _mm512_mul_ps( vlr, _mm512_load_fil( dfilter + i ) ) );
    _mm512_store_fil( filter+i, newfilter );
    _mm512_storeu_ps( master+i, newfilter );
  }
  for ( i = iv; i < thr_end; ++i ) {
    libxsmm_bfloat16_hp t1, t2;
    t1.i[0] =0;
    t1.i[1] = dfilter[i];
    master[i] = master[i] - (handle->desc.learning_rate*t1.f);
    t2.f = master[i];
    filter[i] = t2.i[1];
  }
}
#undef _mm512_load_fil
#undef _mm512_store_fil
#else
for ( i = thr_begin; i < thr_end; ++i ) {
  libxsmm_bfloat16_hp t1, t2;
  t1.i[0] =0;
  t1.i[1] = dfilter[i];
  master[i] = master[i] - (handle->desc.learning_rate*t1.f);
  t2.f = master[i];
  filter[i] = t2.i[1];
}
#endif
#else
#if defined(LIBXSMM_DNN_OPTIMIZER_SGD_F32_AVX512)
{
  libxsmm_blasint iv = ( (thr_end-thr_begin)/16 ) * 16; /* compute iterations which are vectorizable */
  __m512 vlr = _mm512_set1_ps( handle->desc.learning_rate );
  for ( i = thr_begin; i < iv; i+=16 ) {
    _mm512_storeu_ps( filter+i, _mm512_sub_ps( _mm512_loadu_ps( filter+i ), _mm512_mul_ps( vlr, _mm512_loadu_ps( dfilter + i ) ) ) ) ;
  }
  for ( i = iv; i < thr_end; ++i ) {
    filter[i] = filter[i] - (handle->desc.learning_rate*dfilter[i]);
  }
}
#else
for ( i = thr_begin; i < thr_end; ++i ) {
  filter[i] = filter[i] - (handle->desc.learning_rate*dfilter[i]);
}
#endif
#endif

libxsmm_barrier_wait( handle->barrier, ltid );

