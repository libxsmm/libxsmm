/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/

#if defined(LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI)
# define LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH( A ) (__m256i)_mm512_cvtneps_pbh( A )
# define LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH( A, B ) (__m512i)_mm512_cvtne2ps_pbh( A, B )
#else
# define LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH( A ) LIBXSMM_INTRINSICS_MM512_CVT_FP32_BF16( A )
# define LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH( A, B ) LIBXSMM_INTRINSICS_MM512_CVT2_FP32_BF16( A, B )
#endif

#define LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16(in, out, length) do { \
  unsigned int full_chunks = length / 32; \
  unsigned int remainder = length % 32; \
  int __i = 0; \
  if (remainder == 0) { \
    for ( __i = 0; __i < length; __i+= 32) { \
      _mm512_storeu_si512((libxsmm_bfloat16*)out+__i, LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(LIBXSMM_INTRINSICS_MM512_LOAD_PS((const float*)in+__i+16), LIBXSMM_INTRINSICS_MM512_LOAD_PS((const float*)in+__i))); \
    } \
  } else { \
    unsigned int chunk; \
    for ( chunk = 0; chunk < full_chunks; chunk++) { \
      __i = chunk * 32; \
      _mm512_storeu_si512((libxsmm_bfloat16*)out+__i, LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(LIBXSMM_INTRINSICS_MM512_LOAD_PS((const float*)in+__i+16), LIBXSMM_INTRINSICS_MM512_LOAD_PS((const float*)in+__i))); \
    } \
    libxsmm_rne_convert_fp32_bf16((const float*)in+32*full_chunks, (element_output_type*)out+32*full_chunks, remainder); \
  } \
} while(0)

#define LIBXSMM_DNN_CONVERT_BUFFER_BF16_F32(in, out, length) do { \
  unsigned int full_chunks = length / 16; \
  unsigned int remainder = length % 16; \
  int __i = 0; \
  if (remainder == 0) { \
    for ( __i = 0; __i < length; __i+= 16) { \
      _mm512_storeu_ps( (float*)out+__i, LIBXSMM_INTRINSICS_MM512_CVTPBH_PS( _mm256_loadu_si256((__m256i*)((const libxsmm_bfloat16*)in+__i)))); \
    } \
  } else { \
    unsigned int chunk; \
    for ( chunk = 0; chunk < full_chunks; chunk++) { \
      __i = chunk * 16; \
      _mm512_storeu_ps( (float*)out+__i, LIBXSMM_INTRINSICS_MM512_CVTPBH_PS( _mm256_loadu_si256((__m256i*)((const libxsmm_bfloat16*)in+__i)))); \
    } \
    libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)in+16*full_chunks, (float*)out+16*full_chunks, remainder); \
  } \
} while(0)

