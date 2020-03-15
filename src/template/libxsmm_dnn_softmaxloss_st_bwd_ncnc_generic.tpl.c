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

#if defined(LIBXSMM_DNN_SOFTMAXLOSS_BWD_BF16_AVX512)
#define LIBXSMM_DNN_CONVERT_F32_BF16(in, out, length) do { \
  unsigned int full_chunks = length / 16; \
  unsigned int remainder = length % 16; \
  int __i = 0; \
  if (remainder == 0) { \
    for ( __i = 0; __i < length; __i+= 16) { \
      _mm256_storeu_si256((__m256i*)(out+__i), _mm512_cvtepi32_epi16( _mm512_srai_epi32( LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16( LIBXSMM_INTRINSICS_MM512_LOAD_PS((const float*)in+__i) ),16)) ); \
    } \
  } else { \
    unsigned int chunk; \
    for ( chunk = 0; chunk < full_chunks; chunk++) { \
      __i = chunk * 16; \
      _mm256_storeu_si256((__m256i*)(out+__i), _mm512_cvtepi32_epi16( _mm512_srai_epi32( LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16( LIBXSMM_INTRINSICS_MM512_LOAD_PS((const float*)in+__i) ),16)) ); \
    } \
    libxsmm_rne_convert_fp32_bf16((const float*)in+16*full_chunks, (libxsmm_bfloat16*)out+16*full_chunks, remainder); \
  } \
} while(0)

#define LIBXSMM_DNN_CONVERT_BF16_F32(in, out, length) do { \
  unsigned int full_chunks = length / 16; \
  unsigned int remainder = length % 16; \
  int __i = 0; \
  if (remainder == 0) { \
    for ( __i = 0; __i < length; __i+= 16) { \
      _mm512_storeu_ps( out+__i,  _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(in+__i))),16))); \
    } \
  } else { \
    unsigned int chunk; \
    for ( chunk = 0; chunk < full_chunks; chunk++) { \
      __i = chunk * 16; \
      _mm512_storeu_ps( out+__i, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(in+__i))),16))); \
    } \
    libxsmm_convert_bf16_f32( (const libxsmm_bfloat16*)in+16*full_chunks, (float*)out+16*full_chunks, remainder); \
  } \
} while(0)
#endif

libxsmm_blasint bn = handle->bn;
libxsmm_blasint Bn = handle->Bn;
libxsmm_blasint bc = handle->bc;
libxsmm_blasint Bc = handle->Bc;

/* loop counters */
int i = 0;
libxsmm_blasint img1, img2, ifm1, ifm2;

float rcp_N = 1.0f/handle->desc.N;

/* computing first logical thread */
const int ltid = tid - start_thread;

/* number of tasks that could run in parallel for the batch */
const int n_work = Bn * bn;
/* compute chunk size */
const int n_chunksize = (n_work % handle->desc.threads == 0) ? (n_work / handle->desc.threads) : ((n_work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int n_thr_begin = (ltid * n_chunksize < n_work) ? (ltid * n_chunksize) : n_work;
const int n_thr_end = ((ltid + 1) * n_chunksize < n_work) ? ((ltid + 1) * n_chunksize) : n_work;

#if defined(LIBXSMM_DNN_SOFTMAXLOSS_BWD_BF16) || defined(LIBXSMM_DNN_SOFTMAXLOSS_BWD_BF16_AVX512)
/* number of tasks that could run in parallel for the batch */
const int nc_work = Bn * bn;
/* compute chunk size */
const int nc_chunksize = (nc_work % handle->desc.threads == 0) ? (nc_work / handle->desc.threads) : ((nc_work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int nc_thr_begin = (ltid * nc_chunksize < nc_work) ? (ltid * nc_chunksize) : nc_work;
const int nc_thr_end = ((ltid + 1) * nc_chunksize < nc_work) ? ((ltid + 1) * nc_chunksize) : nc_work;

libxsmm_bfloat16* poutput_bf16 = (element_output_type*)handle->reg_output->data;
libxsmm_bfloat16* pdinput_bf16 = (element_input_type*)handle->grad_input->data;
float*            poutput_fp32 = (float*)handle->scratch;
float*            pdinput_fp32 = ((float*)handle->scratch)+(handle->desc.N*handle->desc.C);
LIBXSMM_VLA_DECL(4, const float, output, poutput_fp32, Bc, bn, bc);
LIBXSMM_VLA_DECL(4,       float, dinput, pdinput_fp32, Bc, bn, bc);
#else
LIBXSMM_VLA_DECL(4, const element_output_type, output, (element_output_type*)handle->reg_output->data,  Bc, bn, bc);
LIBXSMM_VLA_DECL(4,       element_input_type,  dinput,  (element_input_type*)handle->grad_input->data,  Bc, bn, bc);
#endif
LIBXSMM_VLA_DECL(2, const element_label_type,   label,  (element_label_type*)handle->label->data,               bn);

/* lazy barrier init */
libxsmm_barrier_init( handle->barrier, ltid );

#if defined(LIBXSMM_DNN_SOFTMAXLOSS_BWD_BF16)
for ( i = nc_thr_begin; i < nc_thr_end; ++i ) {
  libxsmm_bfloat16_hp out;
  out.i[0] = 0;
  out.i[1] = poutput_bf16[i];
  poutput_fp32[i] = out.f;
}

libxsmm_barrier_wait( handle->barrier, ltid );
#endif
#if defined(LIBXSMM_DNN_SOFTMAXLOSS_BWD_BF16_AVX512)
LIBXSMM_DNN_CONVERT_BF16_F32(poutput_bf16+nc_thr_begin, poutput_fp32+nc_thr_begin, nc_thr_end-nc_thr_begin);

libxsmm_barrier_wait( handle->barrier, ltid );
#endif

for ( i = n_thr_begin; i < n_thr_end; ++i ) {
  img1 = i/bn;
  img2 = i%bn;

  /* set output to input and set compute max per image */
  for ( ifm1 = 0; ifm1 < Bc; ++ifm1 ) {
    for ( ifm2 = 0; ifm2 < bc; ++ifm2 ) {
      if ( (ifm1*Bc)+ifm2 == (libxsmm_blasint)LIBXSMM_VLA_ACCESS( 2, label, img1, img2, bn ) ) {
        LIBXSMM_VLA_ACCESS( 4, dinput, img1, ifm1, img2, ifm2, Bc, bn, bc ) =
          ( LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc ) - 1.0f ) * rcp_N * handle->desc.loss_weight;
      } else {
        LIBXSMM_VLA_ACCESS( 4, dinput, img1, ifm1, img2, ifm2, Bc, bn, bc ) =
          LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc ) * rcp_N * handle->desc.loss_weight;
      }
    }
  }
}

libxsmm_barrier_wait( handle->barrier, ltid );

#if defined(LIBXSMM_DNN_SOFTMAXLOSS_BWD_BF16)
for ( i = nc_thr_begin; i < nc_thr_end; ++i ) {
  libxsmm_bfloat16_hp din;
  din.i[0] = 0;
  din.i[1] = pdinput_bf16[i];
  pdinput_fp32[i] = din.f;
}

libxsmm_barrier_wait( handle->barrier, ltid );
#endif
#if defined(LIBXSMM_DNN_SOFTMAXLOSS_BWD_BF16_AVX512)
LIBXSMM_DNN_CONVERT_F32_BF16(pdinput_fp32+nc_thr_begin, pdinput_bf16+nc_thr_begin, nc_thr_end-nc_thr_begin);

libxsmm_barrier_wait( handle->barrier, ltid );
#undef LIBXSMM_DNN_CONVERT_F32_BF16
#undef LIBXSMM_DNN_CONVERT_BF16_F32
#endif

