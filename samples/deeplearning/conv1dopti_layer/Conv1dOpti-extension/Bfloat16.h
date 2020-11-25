/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Sasikanth Avancha (Intel Corp.)
******************************************************************************/


/* create a type called bfloat16 – need to figure out a common header file */
typedef unsigned short bfloat16;

#define _FIXUP_INPUT_CODE_QNAN  0
#define _FIXUP_INPUT_CODE_SNAN  1
#define _FIXUP_INPUT_CODE_NINF  4
#define _FIXUP_INPUT_CODE_PINF  5
#define _FIXUP_OUTPUT_CODE_COPY_INPUT  1
#define _FIXUP_OUTPUT_CODE_QNAN_INPUT  2
#define ENCODE_FIXUP_SELECTOR(input,output) ((output) << (4*(input)))

static const int selector_int32 =
  ENCODE_FIXUP_SELECTOR(_FIXUP_INPUT_CODE_SNAN, _FIXUP_OUTPUT_CODE_QNAN_INPUT) |        /* Qnan input to Qnan output (presenrving input bits 0..21) */
  ENCODE_FIXUP_SELECTOR(_FIXUP_INPUT_CODE_QNAN, _FIXUP_OUTPUT_CODE_QNAN_INPUT) |        /* Snan input to Qnan output (presenrving input bits 0..21) */
  ENCODE_FIXUP_SELECTOR(_FIXUP_INPUT_CODE_NINF, _FIXUP_OUTPUT_CODE_COPY_INPUT) |        /* Neg Inf input copied to output */
  ENCODE_FIXUP_SELECTOR(_FIXUP_INPUT_CODE_PINF, _FIXUP_OUTPUT_CODE_COPY_INPUT);         /* Pos Inf input copied to output */

/* FP32 to Bf16 with round-to-nearest-even */
static __m512 fp32_to_bfp16_rne_adjustment_avx512f(__m512 vfp32) {
  const __m512i vrne_even = _mm512_set1_epi32(0x00007fff);
  const __m512i one = _mm512_set1_epi32(1);
  const __m512i selector = _mm512_set1_epi32(selector_int32);

  __m512i vfp32_as_int = _mm512_castps_si512(vfp32);
  __m512i odd = _mm512_and_si512(_mm512_srli_epi32(vfp32_as_int, 16), one);
  __m512i rounding_factor = _mm512_add_epi32(vrne_even, odd);
  vfp32_as_int = _mm512_add_epi32(vfp32_as_int, rounding_factor);
  return _mm512_fixupimm_ps(_mm512_castsi512_ps(vfp32_as_int), vfp32, selector, 0);
}

/* FP32 to BF16 with truncation */
static __m256i fp32_to_bfp16_truncate_avx512f(__m512 vfp32) {
  __m512i vbfp16_32 = _mm512_srai_epi32(_mm512_castps_si512(vfp32), 16);
  return _mm512_cvtepi32_epi16(vbfp16_32);
}

/* BF16 to FP32 conversion */
static __m512 bfp16_to_fp32_avx512f(__m256i vbfp16) {
  __m512i vbfp16_32 = _mm512_cvtepi16_epi32(vbfp16);
  return _mm512_castsi512_ps(_mm512_slli_epi32(vbfp16_32, 16));
}

