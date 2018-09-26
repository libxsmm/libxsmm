/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/

if (fuse_relu_externally && downconvert_to_bf16_externally) {
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  LIBXSMM_VLA_DECL(5, const element_input_type, input, (element_input_type*) handle->reg_input->data,  handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
  const element_input_type *orig_input_ptr = &LIBXSMM_VLA_ACCESS(5, input, img, /*ifm1*/code_stream[pc].aux_index, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
  element_input_type *input_dst = &LIBXSMM_VLA_ACCESS(5, del_input, img, code_stream[pc].aux_index/*ifm1*/, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
  float *scratch_ptr = accumulators_scratch;
  const __m512 zero_reg = _mm512_setzero_ps();
  if ( handle->f32_bf16_cvt_rne ) {
    const __m512i vnaninf = _mm512_set1_epi32( 0x7f800000 );
    const __m512i vrneadd = _mm512_set1_epi32( 0x00007fff );
    const __m512i vfixup = _mm512_set1_epi32( 0x00000001 );
    const __m512i vfixupmask = _mm512_set1_epi32( 0x00010000 );
    for ( ij = 0; ij < handle->desc.H; ij++ ) {
      for ( ii = 0; ii < handle->desc.W*handle->ifmblock_hp; ii+=16 ) {
        const __m512i vfp32 = _mm512_castps_si512( _mm512_loadu_ps(scratch_ptr+ii) );
        const __m256i orig_reg = _mm256_loadu_si256( (__m256i*) (orig_input_ptr + ii));
        const __m512i orig_reg_fp32 = _mm512_cvtepi16_epi32( orig_reg );
        const __mmask16 mask = _mm512_cmp_epi32_mask((__m512i)zero_reg, orig_reg_fp32, _MM_CMPINT_EQ);
        const __m512i vfp32_masked = _mm512_mask_blend_epi32(mask, vfp32, orig_reg_fp32);
        const __m512i vfp32nan = _mm512_and_epi32( vfp32_masked, vnaninf );
        const __m512i vfp32fixup = _mm512_and_epi32( vfp32_masked, vfixupmask );
        const __mmask16 rnemask = _mm512_cmp_epi32_mask( vfp32nan, vnaninf, _MM_CMPINT_NE );
        const __mmask16 fixupmask = _mm512_cmp_epi32_mask( vfp32fixup, vfixupmask, _MM_CMPINT_EQ );
        const __m512i vrnd = _mm512_mask_add_epi32( vrneadd , fixupmask, vrneadd, vfixup );
        const __m512i vfp32rne = _mm512_mask_add_epi32( vfp32_masked, rnemask, vfp32_masked, vrnd );
        const __m512i vbfp16_32 = _mm512_srai_epi32( vfp32rne, 16 );
        const __m256i vbfp16 = _mm512_cvtepi32_epi16( vbfp16_32 );
        _mm512_storeu_ps(scratch_ptr+ii, zero_reg);
        _mm256_storeu_si256( (__m256i*)(input_dst+ii), vbfp16 );
      }
      scratch_ptr += handle->desc.W*handle->ifmblock_hp;
      input_dst += handle->ifwp*handle->ifmblock_hp;
      orig_input_ptr += handle->ifwp * 16;
    }
  } else {
    for ( ij = 0; ij < handle->desc.H; ij++ ) {
      for ( ii = 0; ii < handle->desc.W*handle->ifmblock_hp; ii+=16 ) {
        const __m512 tmp = _mm512_loadu_ps(scratch_ptr+ii);
        const __m512i vfp32 = _mm512_castps_si512(tmp);
        const __m256i orig_reg = _mm256_loadu_si256( (__m256i*) (orig_input_ptr + ii));
        const __m512i orig_reg_fp32 = _mm512_cvtepi16_epi32( orig_reg );
        const __mmask16 mask = _mm512_cmp_epi32_mask((__m512i)zero_reg, orig_reg_fp32, _MM_CMPINT_EQ);
        const __m512i vfp32_masked = _mm512_mask_blend_epi32(mask, vfp32, orig_reg_fp32);
        const __m256i vbfp16 = _mm512_cvtepi32_epi16(_mm512_srai_epi32( vfp32_masked, 16));
        _mm512_storeu_ps(scratch_ptr+ii, zero_reg);
        _mm256_storeu_si256( (__m256i*)(input_dst+ii), vbfp16 );
      }
      scratch_ptr += handle->desc.W*handle->ifmblock_hp;
      input_dst += handle->ifwp*handle->ifmblock_hp;
      orig_input_ptr += handle->ifwp * 16;
    }
  }
#else /* won't happen as this code only runs on AVX512 platforms */
  LIBXSMM_ASSERT(0);
#endif
}

if (fuse_relu_externally && !downconvert_to_bf16_externally) {
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  LIBXSMM_VLA_DECL(5, element_input_type, input, (element_input_type*) handle->reg_input->data,  handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
  LIBXSMM_VLA_DECL(5, element_input_type, del_input_2, (element_input_type*) handle->grad_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
  element_input_type *orig_input_ptr;
  element_input_type *del_input_ptr;
  __m512 zero_reg  = _mm512_setzero_ps();
  __m512 orig_reg;
  __mmask16 mask;
  orig_input_ptr = &LIBXSMM_VLA_ACCESS(5, input, img, /*ifm1*/code_stream[pc].aux_index, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
  del_input_ptr = &LIBXSMM_VLA_ACCESS(5, del_input_2, img, /*ifm1*/code_stream[pc].aux_index, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
  for (ij = 0; ij < handle->desc.H; ij++) {
    for (ii = 0; ii < handle->desc.W * 16; ii += 16) {
      orig_reg  = LIBXSMM_INTRINSICS_MM512_LOAD_PS(orig_input_ptr + ii);
      mask = _mm512_cmp_ps_mask(zero_reg, orig_reg, _CMP_EQ_OQ);
      _mm512_mask_storeu_ps(del_input_ptr + ii, mask, zero_reg);
    }
    orig_input_ptr += handle->ifwp * 16;
    del_input_ptr += handle->ifwp *16;
  }
#else /* won't happen as this code only runs on AVX512 platforms */
  LIBXSMM_ASSERT(0);
#endif
}

if (!fuse_relu_externally && downconvert_to_bf16_externally) {
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  element_input_type *input_dst = &LIBXSMM_VLA_ACCESS(5, del_input, img, code_stream[pc].aux_index/*ifm1*/, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock_hp);
  float *scratch_ptr = accumulators_scratch;
  __m512 zero_reg = _mm512_setzero_ps();
  if ( handle->f32_bf16_cvt_rne ) {
    __m512i vnaninf = _mm512_set1_epi32( 0x7f800000 );
    __m512i vrneadd = _mm512_set1_epi32( 0x00007fff );
    __m512i vfixup = _mm512_set1_epi32( 0x00000001 );
    __m512i vfixupmask = _mm512_set1_epi32( 0x00010000 );
    for ( ij = 0; ij < handle->desc.H; ij++ ) {
      for ( ii = 0; ii < handle->desc.W*handle->ifmblock_hp; ii+=16 ) {
        __m512i vfp32     = _mm512_castps_si512( _mm512_loadu_ps(scratch_ptr+ii) );
        __m512i vfp32nan  = _mm512_and_epi32( vfp32, vnaninf );
        __m512i vfp32fixup  = _mm512_and_epi32( vfp32, vfixupmask );
        __mmask16 rnemask = _mm512_cmp_epi32_mask( vfp32nan, vnaninf, _MM_CMPINT_NE );
        __mmask16 fixupmask = _mm512_cmp_epi32_mask( vfp32fixup, vfixupmask, _MM_CMPINT_EQ );
        __m512i vrnd = _mm512_mask_add_epi32( vrneadd , fixupmask, vrneadd, vfixup );
        __m512i vfp32rne  = _mm512_mask_add_epi32( vfp32, rnemask, vfp32, vrnd );
        __m512i vbfp16_32 = _mm512_srai_epi32( vfp32rne, 16 );
        __m256i vbfp16    = _mm512_cvtepi32_epi16( vbfp16_32 );
        _mm512_storeu_ps(scratch_ptr+ii, zero_reg);
        _mm256_storeu_si256( (__m256i*)(input_dst+ii), vbfp16 );
      }
      scratch_ptr += handle->desc.W*handle->ifmblock_hp;
      input_dst += handle->ifwp*handle->ifmblock_hp;
    }
  } else {
    for ( ij = 0; ij < handle->desc.H; ij++ ) {
      for ( ii = 0; ii < handle->desc.W*handle->ifmblock_hp; ii+=16 ) {
        __m512 tmp = _mm512_loadu_ps(scratch_ptr+ii);
        __m256i vbfp16 =  _mm512_cvtepi32_epi16(_mm512_srai_epi32( _mm512_castps_si512( tmp ), 16));
        _mm512_storeu_ps(scratch_ptr+ii, zero_reg);
        _mm256_storeu_si256( (__m256i*)(input_dst+ii), vbfp16 );
      }
      scratch_ptr += handle->desc.W*handle->ifmblock_hp;
      input_dst += handle->ifwp*handle->ifmblock_hp;
    }
  }
#else /* won't happen as this code only runs on AVX512 platforms */
  LIBXSMM_ASSERT(0);
#endif
}
