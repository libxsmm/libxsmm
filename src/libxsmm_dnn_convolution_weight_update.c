/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
/* Rajkishore Barik, Alexander Heinecke, Ankush Mandal, Jason Sewall (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_convolution_weight_update.h"
#include <libxsmm_intrinsics_x86.h>
#include "libxsmm_main.h"
#include "stdio.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <string.h>
#include <math.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#define TRANSPOSE_W_CHUNK(img, src_ifm1, ij, w_offset, src_ifm2, dst_ifm1, dst_ifm2) \
        base_addr = &LIBXSMM_VLA_ACCESS(6, input_nopad, img, src_ifm1, ij, w_offset, src_ifm2, 0, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block); \
        gather_reg = _mm512_i32gather_epi32(vgindex, base_addr, 1); \
        lo_reg= _mm512_extracti64x4_epi64(gather_reg,0); \
        hi_reg= _mm512_extracti64x4_epi64(gather_reg,1); \
        compressed_low  = _mm256_unpacklo_epi16(lo_reg, hi_reg); \
        compressed_low =  _mm256_permutevar8x32_epi32(compressed_low, shuffler); \
        compressed_high  = _mm256_unpackhi_epi16(lo_reg, hi_reg); \
        compressed_high =  _mm256_permutevar8x32_epi32(compressed_high, shuffler); \
        compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_low,0), 0); \
        compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_high, 0), 1); \
        compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_low,1), 0); \
        compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_high, 1), 1); \
        _mm256_storeu_si256((union __m256i *) &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, dst_ifm1, ij, dst_ifm2, w_offset, BLOCKSIFM, handle->ifhp, handle->ifmblock, ifwp_extended), compressed_low_store); \
        _mm256_storeu_si256((union __m256i *) &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, dst_ifm1, ij, dst_ifm2+1, w_offset, BLOCKSIFM, handle->ifhp, handle->ifmblock, ifwp_extended), compressed_high_store);

#define TRANSPOSE_W_REMAINDER(img, src_ifm1, ij, w_offset, src_ifm2, dst_ifm1, dst_ifm2) \
        base_addr = &LIBXSMM_VLA_ACCESS(6, input_nopad, img, src_ifm1, ij, w_offset, src_ifm2, 0, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block); \
        gather_reg = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), gmask, vgindex, base_addr, 1); \
        lo_reg= _mm512_extracti64x4_epi64(gather_reg,0); \
        hi_reg= _mm512_extracti64x4_epi64(gather_reg,1); \
        compressed_low  = _mm256_unpacklo_epi16(lo_reg, hi_reg); \
        compressed_low =  _mm256_permutevar8x32_epi32(compressed_low, shuffler); \
        compressed_high  = _mm256_unpackhi_epi16(lo_reg, hi_reg); \
        compressed_high =  _mm256_permutevar8x32_epi32(compressed_high, shuffler); \
        compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_low,0), 0); \
        compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_high, 0), 1); \
        compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_low,1), 0); \
        compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_high, 1), 1); \
        _mm256_maskstore_epi32((int*) &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, dst_ifm1, ij, dst_ifm2, w_offset, BLOCKSIFM, handle->ifhp, handle->ifmblock, ifwp_extended), mask_reg, compressed_low_store); \
        _mm256_maskstore_epi32((int*) &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, dst_ifm1, ij, dst_ifm2+1, w_offset, BLOCKSIFM, handle->ifhp, handle->ifmblock, ifwp_extended), mask_reg, compressed_high_store); 

#define TRANSPOSE_W_CHUNK_RESIZED(img, src_ifm1, src_i, src_j, dst_i, dst_j, src_ifm2, dst_ifm1, dst_ifm2) \
        base_addr = &LIBXSMM_VLA_ACCESS(6, input_nopad, img, src_ifm1, src_j, src_i, src_ifm2, 0, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block); \
        gather_reg = _mm512_i32gather_epi32(vgindex, base_addr, 1); \
        lo_reg= _mm512_extracti64x4_epi64(gather_reg,0); \
        hi_reg= _mm512_extracti64x4_epi64(gather_reg,1); \
        compressed_low  = _mm256_unpacklo_epi16(lo_reg, hi_reg); \
        compressed_low =  _mm256_permutevar8x32_epi32(compressed_low, shuffler); \
        compressed_high  = _mm256_unpackhi_epi16(lo_reg, hi_reg); \
        compressed_high =  _mm256_permutevar8x32_epi32(compressed_high, shuffler); \
        compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_low,0), 0); \
        compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_high, 0), 1); \
        compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_low,1), 0); \
        compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_high, 1), 1); \
        _mm256_storeu_si256((union __m256i *) &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, dst_ifm1, dst_j, dst_ifm2, dst_i, BLOCKSIFM, handle->ifhp_resized, handle->ifmblock, ifwp_extended), compressed_low_store); \
        _mm256_storeu_si256((union __m256i *) &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, dst_ifm1, dst_j, dst_ifm2+1, dst_i, BLOCKSIFM, handle->ifhp_resized, handle->ifmblock, ifwp_extended), compressed_high_store);

#define TRANSPOSE_W_REMAINDER_RESIZED(img, src_ifm1, src_i, src_j, dst_i, dst_j, src_ifm2, dst_ifm1, dst_ifm2) \
        base_addr = &LIBXSMM_VLA_ACCESS(6, input_nopad, img, src_ifm1, src_j, src_i, src_ifm2, 0, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block); \
        gather_reg = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), gmask, vgindex, base_addr, 1); \
        lo_reg= _mm512_extracti64x4_epi64(gather_reg,0); \
        hi_reg= _mm512_extracti64x4_epi64(gather_reg,1); \
        compressed_low  = _mm256_unpacklo_epi16(lo_reg, hi_reg); \
        compressed_low =  _mm256_permutevar8x32_epi32(compressed_low, shuffler); \
        compressed_high  = _mm256_unpackhi_epi16(lo_reg, hi_reg); \
        compressed_high =  _mm256_permutevar8x32_epi32(compressed_high, shuffler); \
        compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_low,0), 0); \
        compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_high, 0), 1); \
        compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_low,1), 0); \
        compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_high, 1), 1); \
        _mm256_maskstore_epi32((int*) &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, dst_ifm1, dst_j, dst_ifm2, dst_i, BLOCKSIFM, handle->ifhp_resized, handle->ifmblock, ifwp_extended), mask_reg, compressed_low_store); \
        _mm256_maskstore_epi32((int*) &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, dst_ifm1, dst_j, dst_ifm2+1, dst_i, BLOCKSIFM, handle->ifhp_resized, handle->ifmblock, ifwp_extended), mask_reg, compressed_high_store);

#define TRANSPOSE_W_FULL_PAIR(img, ofm1, ij, ii, half_i) \
      even_addr_lo = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, ii, 0, 0,  handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block); \
      odd_addr_lo = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, ii+1, 0, 0,  handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block); \
      even_addr_hi = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, ii, 8, 0,  handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block); \
      odd_addr_hi = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, ii+1, 8, 0,  handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block); \
      even_pixel_lo = _mm256_stream_load_si256((const union __m256i *) even_addr_lo); \
      odd_pixel_lo = _mm256_stream_load_si256((const union __m256i *) odd_addr_lo); \
      even_pixel_hi = _mm256_stream_load_si256((const union __m256i *) even_addr_hi); \
      odd_pixel_hi = _mm256_stream_load_si256((const union __m256i *) odd_addr_hi); \
      compressed_lo  = _mm256_unpacklo_epi16(even_pixel_lo, odd_pixel_lo); \
      compressed_hi  = _mm256_unpackhi_epi16(even_pixel_lo, odd_pixel_lo); \
      compressed_lo_store = _mm256_insertf128_si256(compressed_lo_store, _mm256_extractf128_si256(compressed_lo,0), 0); \
      compressed_lo_store = _mm256_insertf128_si256(compressed_lo_store, _mm256_extractf128_si256(compressed_hi,0), 1); \
      compressed_hi_store = _mm256_insertf128_si256(compressed_hi_store, _mm256_extractf128_si256(compressed_lo,1), 0); \
      compressed_hi_store = _mm256_insertf128_si256(compressed_hi_store, _mm256_extractf128_si256(compressed_hi,1), 1); \
      dst_lo = &LIBXSMM_VLA_ACCESS(6,  tr_output, img, 2*ofm1, ij, half_i, 0, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2); \
      dst_hi = &LIBXSMM_VLA_ACCESS(6,  tr_output, img, 2*ofm1, ij, half_i, 8, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2); \
      _mm256_stream_si256((union __m256i *) dst_lo, compressed_lo_store); \
      _mm256_stream_si256((union __m256i *) dst_hi, compressed_hi_store); \
      compressed_lo  = _mm256_unpacklo_epi16(even_pixel_hi, odd_pixel_hi); \
      compressed_hi  = _mm256_unpackhi_epi16(even_pixel_hi, odd_pixel_hi); \
      compressed_lo_store = _mm256_insertf128_si256(compressed_lo_store, _mm256_extractf128_si256(compressed_lo,0), 0); \
      compressed_lo_store = _mm256_insertf128_si256(compressed_lo_store, _mm256_extractf128_si256(compressed_hi,0), 1); \
      compressed_hi_store = _mm256_insertf128_si256(compressed_hi_store, _mm256_extractf128_si256(compressed_lo,1), 0); \
      compressed_hi_store = _mm256_insertf128_si256(compressed_hi_store, _mm256_extractf128_si256(compressed_hi,1), 1); \
      dst_lo = &LIBXSMM_VLA_ACCESS(6,  tr_output, img, 2*ofm1+1, ij, half_i, 0, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2); \
      dst_hi = &LIBXSMM_VLA_ACCESS(6,  tr_output, img, 2*ofm1+1, ij, half_i, 8, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2); \
      _mm256_stream_si256((union __m256i *) dst_lo, compressed_lo_store); \
      _mm256_stream_si256((union __m256i *) dst_hi, compressed_hi_store);

#define TRANSPOSE_W_HALF_PAIR(img, ofm1, ij, ii, half_i) \
      even_addr_lo = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, ii, 0, 0,  handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block); \
      even_addr_hi = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, ii, 8, 0,  handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block); \
      even_pixel_lo = _mm256_stream_load_si256((const union __m256i *) even_addr_lo); \
      even_pixel_hi = _mm256_stream_load_si256((const union __m256i *) even_addr_hi); \
      odd_pixel_lo = _mm256_xor_si256(odd_pixel_lo,odd_pixel_lo); \
      odd_pixel_hi = odd_pixel_lo; \
      compressed_lo  = _mm256_unpacklo_epi16(even_pixel_lo, odd_pixel_lo); \
      compressed_hi  = _mm256_unpackhi_epi16(even_pixel_lo, odd_pixel_lo); \
      compressed_lo_store = _mm256_insertf128_si256(compressed_lo_store, _mm256_extractf128_si256(compressed_lo,0), 0); \
      compressed_lo_store = _mm256_insertf128_si256(compressed_lo_store, _mm256_extractf128_si256(compressed_hi,0), 1); \
      compressed_hi_store = _mm256_insertf128_si256(compressed_hi_store, _mm256_extractf128_si256(compressed_lo,1), 0); \
      compressed_hi_store = _mm256_insertf128_si256(compressed_hi_store, _mm256_extractf128_si256(compressed_hi,1), 1); \
      dst_lo = &LIBXSMM_VLA_ACCESS(6,  tr_output, img, 2*ofm1, ij, half_i, 0, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2); \
      dst_hi = &LIBXSMM_VLA_ACCESS(6,  tr_output, img, 2*ofm1, ij, half_i, 8, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2); \
      _mm256_stream_si256((union __m256i *) dst_lo, compressed_lo_store); \
      _mm256_stream_si256((union __m256i *) dst_hi, compressed_hi_store); \
      compressed_lo  = _mm256_unpacklo_epi16(even_pixel_hi, odd_pixel_hi); \
      compressed_hi  = _mm256_unpackhi_epi16(even_pixel_hi, odd_pixel_hi); \
      compressed_lo_store = _mm256_insertf128_si256(compressed_lo_store, _mm256_extractf128_si256(compressed_lo,0), 0); \
      compressed_lo_store = _mm256_insertf128_si256(compressed_lo_store, _mm256_extractf128_si256(compressed_hi,0), 1); \
      compressed_hi_store = _mm256_insertf128_si256(compressed_hi_store, _mm256_extractf128_si256(compressed_lo,1), 0); \
      compressed_hi_store = _mm256_insertf128_si256(compressed_hi_store, _mm256_extractf128_si256(compressed_hi,1), 1); \
      dst_lo = &LIBXSMM_VLA_ACCESS(6,  tr_output, img, 2*ofm1+1, ij, half_i, 0, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2); \
      dst_hi = &LIBXSMM_VLA_ACCESS(6,  tr_output, img, 2*ofm1+1, ij, half_i, 8, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2); \
      _mm256_stream_si256((union __m256i *) dst_lo, compressed_lo_store); \
      _mm256_stream_si256((union __m256i *) dst_hi, compressed_hi_store);

void lp_transpose_input_and_output(int img, libxsmm_dnn_layer* handle) {
  typedef short element_input_type;
  typedef short element_output_type;
  int w_chunks = handle->ifwp/16;
  int w_remainder = handle->ifwp%16;
  int w_i, w, c_i, ifm1, ij, ifm2;
  int BLOCKSIFM = handle->blocksifm;;
  int padded_w = (handle->padding_flag == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
  int ifwp_extended = padded_w + handle->qfma_input_pad;
  int dst_ifhp;
  element_input_type *base_addr;
  const __m512i vgindex = _mm512_set_epi32(960,832,448,320,  704,576,192,64,  896,768,384,256,  640,512,128,0);
  const int gather_offsets[16] = {960,832,448,320,  704,576,192,64,  896,768,384,256,  640,512,128,0};
  const __m256i shuffler = _mm256_set_epi32(7,5,3,1,6,4,2,0);
  unsigned int int_mask = 0xffffffff;
  const __mmask16 gmask = int_mask;
  int mask_remainder = (w_remainder+1)/2;
  unsigned int mask[8];
  for (c_i=0;c_i<16;c_i++) {
    if (gather_offsets[16-c_i-1] >= w_remainder*64) {
      int_mask = int_mask & ~(1 << c_i);
    }
  }
  for (c_i=0; c_i<mask_remainder; c_i++) {
    mask[c_i] = (1<<31);
  }
  for (c_i=mask_remainder; c_i<8; c_i++) {
    mask[c_i] = 0;
  }
  __m256i mask_reg = _mm256_loadu_si256((const union __m256i *) mask);
  __m512i gather_reg;
  __m256i lo_reg, hi_reg, compressed_low, compressed_high, compressed_low_store, compressed_high_store;

  /* Input transpose  */
  dst_ifhp = handle->ifhp;
  LIBXSMM_VLA_DECL(6, element_input_type, input_nopad, (element_input_type*)handle->reg_input->data, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
  LIBXSMM_VLA_DECL(5, element_input_type, tr_input_nopad, (element_input_type*)handle->scratch3, BLOCKSIFM, dst_ifhp, handle->ifmblock, ifwp_extended);

  if (w_remainder) {
    for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ++ifm1) {
      for (ij = 0; ij < handle->ifhp; ++ij) {
        /* Handle full chunks  */
        for (w = 0; w < w_chunks; w++) {
          for (ifm2 = 0; ifm2 < 8; ++ifm2) {
            TRANSPOSE_W_CHUNK(img, ifm1, ij, w*16, ifm2, 2*ifm1, 2*ifm2);
            TRANSPOSE_W_CHUNK(img, ifm1, ij, w*16, ifm2+8, 2*ifm1+1, 2*ifm2);
          }
        }
        /* Handle remainder */
        for (ifm2 = 0; ifm2 < 8; ++ifm2) {
          TRANSPOSE_W_REMAINDER(img, ifm1, ij, w_chunks*16, ifm2, 2*ifm1, 2*ifm2);
          TRANSPOSE_W_REMAINDER(img, ifm1, ij, w_chunks*16, ifm2+8,  2*ifm1+1, 2*ifm2); 
        }
      }
    }
  } else {
    for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ++ifm1) {
      for (ij = 0; ij < handle->ifhp; ++ij) {
        /* Handle full chunks  */
        for (w = 0; w < w_chunks; w++) {
          for (ifm2 = 0; ifm2 < 8; ++ifm2) {
            TRANSPOSE_W_CHUNK(img, ifm1, ij, w*16, ifm2, 2*ifm1, 2*ifm2);
            TRANSPOSE_W_CHUNK(img, ifm1, ij, w*16, ifm2+8, 2*ifm1+1, 2*ifm2);
          }
        }
      }
    }
  }

  element_output_type *even_addr_lo, *odd_addr_lo, *even_addr_hi, *odd_addr_hi;
  element_output_type *dst_lo, *dst_hi;
  int half_i, ofm1,  ii;
  int BLOCKSOFM = handle->blocksofm;
  int OFWP = handle->ofwp+handle->output_lp_padding;
  __m256i even_pixel_lo, even_pixel_hi, odd_pixel_hi, odd_pixel_lo, compressed_hi, compressed_lo, compressed_lo_store, compressed_hi_store;

  /* Output transpose */
  element_output_type *out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock * handle->fm_lp_block;
  LIBXSMM_VLA_DECL(6, element_output_type, tr_output,  (element_output_type*)handle->scratch6 , BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2);
  LIBXSMM_VLA_DECL(6, element_output_type, output, out, handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);

  for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ++ofm1) {
    for (ij = 0; ij < handle->ofhp; ++ij) {
      for (ii = 0, half_i=0 ; ii < handle->ofwp-1; ii+=2, half_i++) {
        TRANSPOSE_W_FULL_PAIR(img, ofm1, ij, ii, half_i);
      }
    }
  }
  if (handle->output_lp_padding != 0) {
    for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ++ofm1) {
      for (ij = 0; ij < handle->ofhp; ++ij) {
        ii = handle->ofwp-1;
        half_i = ii/2;
        TRANSPOSE_W_HALF_PAIR(img, ofm1, ij, ii, half_i);
      }
    }
  }
}

void lp_transpose_and_resize_input_and_output(int img, libxsmm_dnn_layer* handle) {
  typedef short element_input_type;
  typedef short element_output_type;

  int w_chunks = handle->ifwp_resized/16;
  int w_remainder = handle->ifwp_resized%16;
  int u = handle->desc.u;
  int w_i, w, c_i, ifm1, ij, ifm2;
  int dst_i, dst_j, src_i, src_j;
  int BLOCKSIFM = handle->blocksifm;;
  int padded_w = (handle->padding_flag == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
  int ifwp_extended = padded_w + handle->qfma_input_pad;
  int dst_ifhp;
  element_input_type *base_addr;
  const __m512i vgindex = _mm512_set_epi32(u*960,u*832,u*448,u*320,  u*704,u*576,u*192,u*64,  u*896,u*768,u*384,u*256,  u*640,u*512,u*128, u*0);
  const int gather_offsets[16] = {u*960,u*832,u*448,u*320,  u*704,u*576,u*192,u*64,  u*896,u*768,u*384,u*256,  u*640,u*512,u*128, u*0};
  const __m256i shuffler = _mm256_set_epi32(7,5,3,1,6,4,2,0);
  unsigned int int_mask = 0xffffffff;
  const __mmask16 gmask = int_mask;
  int mask_remainder = (w_remainder+1)/2;
  unsigned int mask[8];
  for (c_i=0;c_i<16;c_i++) {
    if (gather_offsets[16-c_i-1] >= (w_remainder*64)*u) {
      int_mask = int_mask & ~(1 << c_i);
    }
  }
  for (c_i=0; c_i<mask_remainder; c_i++) {
    mask[c_i] = (1<<31);
  }
  for (c_i=mask_remainder; c_i<8; c_i++) {
    mask[c_i] = 0;
  }
  __m256i mask_reg = _mm256_loadu_si256((const union __m256i *) mask);
  __m512i gather_reg;
  __m256i lo_reg, hi_reg, compressed_low, compressed_high, compressed_low_store, compressed_high_store;

  /* Input transpose  */
  ifwp_extended = handle->ifwp_resized + handle->qfma_input_pad;
  dst_ifhp = handle->ifhp_resized;
  LIBXSMM_VLA_DECL(6, element_input_type, input_nopad, (element_input_type*)handle->reg_input->data, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
  LIBXSMM_VLA_DECL(5, element_input_type, tr_input_nopad, (element_input_type*)handle->scratch3, BLOCKSIFM, dst_ifhp, handle->ifmblock, ifwp_extended);

  if (w_remainder) {
    for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ++ifm1) {
      for (dst_j=0; dst_j < handle->ifhp_resized; dst_j++) {
        src_j = dst_j * handle->desc.v;
        /* Handle full chunks  */
        for (w = 0; w < w_chunks; w++) {
          for (ifm2 = 0; ifm2 < 8; ++ifm2) {
            TRANSPOSE_W_CHUNK_RESIZED(img, ifm1, w*u*16, src_j, w*16, dst_j, ifm2, 2*ifm1, 2*ifm2);
            TRANSPOSE_W_CHUNK_RESIZED(img, ifm1, w*u*16, src_j, w*16, dst_j, ifm2+8, 2*ifm1+1, 2*ifm2);  
          }
        }
        /* Handle remainder */
        for (ifm2 = 0; ifm2 < 8; ++ifm2) {
          TRANSPOSE_W_REMAINDER_RESIZED(img, ifm1, w_chunks*u*16, src_j, w_chunks*16, dst_j, ifm2, 2*ifm1, 2*ifm2);  
          TRANSPOSE_W_REMAINDER_RESIZED(img, ifm1, w_chunks*u*16, src_j, w_chunks*16, dst_j, ifm2+8, 2*ifm1+1, 2*ifm2);
        }
      }
    }
  } else {
    for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ++ifm1) {
      for (dst_j=0; dst_j < handle->ifhp_resized; dst_j++) {
        src_j = dst_j * handle->desc.v;
        /* Handle full chunks  */
        for (w = 0; w < w_chunks; w++) {
          for (ifm2 = 0; ifm2 < 8; ++ifm2) {
            TRANSPOSE_W_CHUNK_RESIZED(img, ifm1, w*u*16, src_j, w*16, dst_j, ifm2, 2*ifm1, 2*ifm2);
            TRANSPOSE_W_CHUNK_RESIZED(img, ifm1, w*u*16, src_j, w*16, dst_j, ifm2+8, 2*ifm1+1, 2*ifm2);  
          }
        }
      }
    }
  }

  element_output_type *even_addr_lo, *odd_addr_lo, *even_addr_hi, *odd_addr_hi;
  element_output_type *dst_lo, *dst_hi;
  int half_i, ofm1,  ii;
  int BLOCKSOFM = handle->blocksofm;
  int OFWP = handle->ofwp+handle->output_lp_padding;
  __m256i even_pixel_lo, even_pixel_hi, odd_pixel_hi, odd_pixel_lo, compressed_hi, compressed_lo, compressed_lo_store, compressed_hi_store;
  /* Output transpose */
  element_output_type *out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock * handle->fm_lp_block;
  LIBXSMM_VLA_DECL(6, element_output_type, tr_output,  (element_output_type*)handle->scratch6 , BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2);
  LIBXSMM_VLA_DECL(6, element_output_type, output, out, handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);

  for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ++ofm1) {
    for (ij = 0; ij < handle->ofhp; ++ij) {
      for (ii = 0, half_i=0 ; ii < handle->ofwp-1; ii+=2, half_i++) {
        TRANSPOSE_W_FULL_PAIR(img, ofm1, ij, ii, half_i);
      }
    }
  }
  if (handle->output_lp_padding != 0) {
    for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ++ofm1) {
      for (ij = 0; ij < handle->ofhp; ++ij) {
        ii = handle->ofwp-1;
        half_i = ii/2;
        TRANSPOSE_W_HALF_PAIR(img, ofm1, ij, ii, half_i);
      }
    }
  }
}

#if 0
void lp_transpose_0_chunk_1_remainder_even_pixels(int img, libxsmm_dnn_layer* handle) {
  typedef short element_input_type;
  typedef short element_output_type;

  int w_chunks = handle->ifwp/16;
  int w_remainder = handle->ifwp%16;
  int w_i, w, c_i, ifm1, ij, ifm2;
  int BLOCKSIFM = handle->blocksifm;;
  int padded_w = (handle->padding_flag == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
  int ifwp_extended = padded_w + handle->qfma_input_pad;
  int dst_ifhp;
  element_input_type *base_addr;
  const __m512i vgindex = _mm512_set_epi32(960,832,448,320,  704,576,192,64,  896,768,384,256,  640,512,128,0);
  const int gather_offsets[16] = {960,832,448,320,  704,576,192,64,  896,768,384,256,  640,512,128,0};
  const __m256i shuffler = _mm256_set_epi32(7,5,3,1,6,4,2,0);
  unsigned int int_mask = 0xffffffff;
  const __mmask16 gmask = int_mask;
  int mask_remainder = (w_remainder+1)/2;
  unsigned int mask[8];
  for (c_i=0;c_i<16;c_i++) {
    if (gather_offsets[16-c_i-1] >= w_remainder*64) {
      int_mask = int_mask & ~(1 << c_i);
    }
  }
  for (c_i=0; c_i<mask_remainder; c_i++) {
    mask[c_i] = (1<<31);
  }
  for (c_i=mask_remainder; c_i<8; c_i++) {
    mask[c_i] = 0;
  }
  __m256i mask_reg = _mm256_loadu_si256((const union __m256i *) mask);
  __m512i gather_reg;
  __m256i lo_reg, hi_reg, compressed_low, compressed_high, compressed_low_store, compressed_high_store;

  /* Input transpose  */
  dst_ifhp = handle->ifhp;
  LIBXSMM_VLA_DECL(6, element_input_type, input_nopad, (element_input_type*)handle->reg_input->data, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
  LIBXSMM_VLA_DECL(5, element_input_type, tr_input_nopad, (element_input_type*)handle->scratch3, BLOCKSIFM, dst_ifhp, handle->ifmblock, ifwp_extended);
  for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ++ifm1) {
    for (ij = 0; ij < handle->ifhp; ++ij) {
      for (ifm2 = 0; ifm2 < 8; ++ifm2) {
        TRANSPOSE_W_REMAINDER(img, ifm1, ij, 0, ifm2, 2*ifm1, 2*ifm2);
        TRANSPOSE_W_REMAINDER(img, ifm1, ij, 0, ifm2+8,  2*ifm1+1, 2*ifm2);   
      }
    }
  }

  element_output_type *even_addr_lo, *odd_addr_lo, *even_addr_hi, *odd_addr_hi;
  element_output_type *dst_lo, *dst_hi;
  int half_i, ofm1,  ii;
  int BLOCKSOFM = handle->blocksofm;
  int OFWP = handle->ofwp+handle->output_lp_padding;
  __m256i even_pixel_lo, even_pixel_hi, odd_pixel_hi, odd_pixel_lo, compressed_hi, compressed_lo, compressed_lo_store, compressed_hi_store;

  /* Output transpose */
  element_output_type *out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock * handle->fm_lp_block;
  LIBXSMM_VLA_DECL(6, element_output_type, tr_output,  (element_output_type*)handle->scratch6 , BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2);
  LIBXSMM_VLA_DECL(6, element_output_type, output, out, handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);
  for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ++ofm1) {
    for (ij = 0; ij < handle->ofhp; ++ij) {
      for (ii = 0, half_i=0 ; ii < handle->ofwp-1; ii+=2, half_i++) {
        TRANSPOSE_W_FULL_PAIR(img, ofm1, ij, ii, half_i);
      }
    }
  }
}

void lp_transpose_0_chunk_1_remainder_odd_pixels(int img, libxsmm_dnn_layer* handle) {
  typedef short element_input_type;
  typedef short element_output_type;

  int w_chunks = handle->ifwp/16;
  int w_remainder = handle->ifwp%16;
  int w_i, w, c_i, ifm1, ij, ifm2;
  int BLOCKSIFM = handle->blocksifm;;
  int padded_w = (handle->padding_flag == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
  int ifwp_extended = padded_w + handle->qfma_input_pad;
  int dst_ifhp;
  element_input_type *base_addr;
  const __m512i vgindex = _mm512_set_epi32(960,832,448,320,  704,576,192,64,  896,768,384,256,  640,512,128,0);
  const int gather_offsets[16] = {960,832,448,320,  704,576,192,64,  896,768,384,256,  640,512,128,0};
  const __m256i shuffler = _mm256_set_epi32(7,5,3,1,6,4,2,0);
  unsigned int int_mask = 0xffffffff;
  const __mmask16 gmask = int_mask;
  int mask_remainder = (w_remainder+1)/2;
  unsigned int mask[8];
  for (c_i=0;c_i<16;c_i++) {
    if (gather_offsets[16-c_i-1] >= w_remainder*64) {
      int_mask = int_mask & ~(1 << c_i);
    }
  }
  for (c_i=0; c_i<mask_remainder; c_i++) {
    mask[c_i] = (1<<31);
  }
  for (c_i=mask_remainder; c_i<8; c_i++) {
    mask[c_i] = 0;
  }
  __m256i mask_reg = _mm256_loadu_si256((const union __m256i *) mask);
  __m512i gather_reg;
  __m256i lo_reg, hi_reg, compressed_low, compressed_high, compressed_low_store, compressed_high_store;

  /* Input transpose  */
  dst_ifhp = handle->ifhp;
  LIBXSMM_VLA_DECL(6, element_input_type, input_nopad, (element_input_type*)handle->reg_input->data, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
  LIBXSMM_VLA_DECL(5, element_input_type, tr_input_nopad, (element_input_type*)handle->scratch3, BLOCKSIFM, dst_ifhp, handle->ifmblock, ifwp_extended);
  for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ++ifm1) {
    for (ij = 0; ij < handle->ifhp; ++ij) {
      for (ifm2 = 0; ifm2 < 8; ++ifm2) {
        TRANSPOSE_W_REMAINDER(img, ifm1, ij, 0, ifm2, 2*ifm1, 2*ifm2);
        TRANSPOSE_W_REMAINDER(img, ifm1, ij, 0, ifm2+8,  2*ifm1+1, 2*ifm2);   
      }
    }
  }

  element_output_type *even_addr_lo, *odd_addr_lo, *even_addr_hi, *odd_addr_hi;
  element_output_type *dst_lo, *dst_hi;
  int half_i, ofm1,  ii;
  int BLOCKSOFM = handle->blocksofm;
  int OFWP = handle->ofwp+handle->output_lp_padding;
  __m256i even_pixel_lo, even_pixel_hi, odd_pixel_hi, odd_pixel_lo, compressed_hi, compressed_lo, compressed_lo_store, compressed_hi_store;

  /* Output transpose */
  element_output_type *out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock * handle->fm_lp_block;
  LIBXSMM_VLA_DECL(6, element_output_type, tr_output,  (element_output_type*)handle->scratch6 , BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2);
  LIBXSMM_VLA_DECL(6, element_output_type, output, out, handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);
  for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ++ofm1) {
    for (ij = 0; ij < handle->ofhp; ++ij) {
      for (ii = 0, half_i=0 ; ii < handle->ofwp-1; ii+=2, half_i++) {
        TRANSPOSE_W_FULL_PAIR(img, ofm1, ij, ii, half_i);
      }
    }
  }
  for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ++ofm1) {
    for (ij = 0; ij < handle->ofhp; ++ij) {
      ii = handle->ofwp-1;
      half_i = ii/2;
      TRANSPOSE_W_HALF_PAIR(img, ofm1, ij, ii, half_i);
    }
  }
}

void lp_transpose_1_chunk_0_remainder_even_pixels(int img, libxsmm_dnn_layer* handle) {
  typedef short element_input_type;
  typedef short element_output_type;

  int w_chunks = handle->ifwp/16;
  int w_remainder = handle->ifwp%16;
  int w_i, w, c_i, ifm1, ij, ifm2;
  int BLOCKSIFM = handle->blocksifm;;
  int padded_w = (handle->padding_flag == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
  int ifwp_extended = padded_w + handle->qfma_input_pad;
  int dst_ifhp;
  element_input_type *base_addr;
  const __m512i vgindex = _mm512_set_epi32(960,832,448,320,  704,576,192,64,  896,768,384,256,  640,512,128,0);
  const int gather_offsets[16] = {960,832,448,320,  704,576,192,64,  896,768,384,256,  640,512,128,0};
  const __m256i shuffler = _mm256_set_epi32(7,5,3,1,6,4,2,0);
  __m512i gather_reg;
  __m256i lo_reg, hi_reg, compressed_low, compressed_high, compressed_low_store, compressed_high_store;

  /* Input transpose  */
  dst_ifhp = handle->ifhp;
  LIBXSMM_VLA_DECL(6, element_input_type, input_nopad, (element_input_type*)handle->reg_input->data, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
  LIBXSMM_VLA_DECL(5, element_input_type, tr_input_nopad, (element_input_type*)handle->scratch3, BLOCKSIFM, dst_ifhp, handle->ifmblock, ifwp_extended);
  for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ++ifm1) {
    for (ij = 0; ij < handle->ifhp; ++ij) {
      for (ifm2 = 0; ifm2 < 8; ++ifm2) {
        TRANSPOSE_W_CHUNK(img, ifm1, ij, 0, ifm2, 2*ifm1, 2*ifm2);
        TRANSPOSE_W_CHUNK(img, ifm1, ij, 0, ifm2+8, 2*ifm1+1, 2*ifm2);
      }
    }
  }

  element_output_type *even_addr_lo, *odd_addr_lo, *even_addr_hi, *odd_addr_hi;
  element_output_type *dst_lo, *dst_hi;
  int half_i, ofm1,  ii;
  int BLOCKSOFM = handle->blocksofm;
  int OFWP = handle->ofwp+handle->output_lp_padding;
  __m256i even_pixel_lo, even_pixel_hi, odd_pixel_hi, odd_pixel_lo, compressed_hi, compressed_lo, compressed_lo_store, compressed_hi_store;

  /* Output transpose */
  element_output_type *out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock * handle->fm_lp_block;
  LIBXSMM_VLA_DECL(6, element_output_type, tr_output,  (element_output_type*)handle->scratch6 , BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2);
  LIBXSMM_VLA_DECL(6, element_output_type, output, out, handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);
  for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ++ofm1) {
    for (ij = 0; ij < handle->ofhp; ++ij) {
      for (ii = 0, half_i=0 ; ii < handle->ofwp-1; ii+=2, half_i++) {
        TRANSPOSE_W_FULL_PAIR(img, ofm1, ij, ii, half_i);
      }
    }
  }
}

void lp_transpose_1_chunk_1_remainder_even_pixels(int img, libxsmm_dnn_layer* handle) {
  typedef short element_input_type;
  typedef short element_output_type;

  int w_chunks = handle->ifwp/16;
  int w_remainder = handle->ifwp%16;
  int w_i, w, c_i, ifm1, ij, ifm2;
  int BLOCKSIFM = handle->blocksifm;;
  int padded_w = (handle->padding_flag == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
  int ifwp_extended = padded_w + handle->qfma_input_pad;
  int dst_ifhp;
  element_input_type *base_addr;
  const __m512i vgindex = _mm512_set_epi32(960,832,448,320,  704,576,192,64,  896,768,384,256,  640,512,128,0);
  const int gather_offsets[16] = {960,832,448,320,  704,576,192,64,  896,768,384,256,  640,512,128,0};
  const __m256i shuffler = _mm256_set_epi32(7,5,3,1,6,4,2,0);
  unsigned int int_mask = 0xffffffff;
  const __mmask16 gmask = int_mask;
  int mask_remainder = (w_remainder+1)/2;
  unsigned int mask[8];
  for (c_i=0;c_i<16;c_i++) {
    if (gather_offsets[16-c_i-1] >= w_remainder*64) {
      int_mask = int_mask & ~(1 << c_i);
    }
  }
  for (c_i=0; c_i<mask_remainder; c_i++) {
    mask[c_i] = (1<<31);
  }
  for (c_i=mask_remainder; c_i<8; c_i++) {
    mask[c_i] = 0;
  }
  __m256i mask_reg = _mm256_loadu_si256((const union __m256i *) mask);
  __m512i gather_reg;
  __m256i lo_reg, hi_reg, compressed_low, compressed_high, compressed_low_store, compressed_high_store;

  /* Input transpose  */
  dst_ifhp = handle->ifhp;
  LIBXSMM_VLA_DECL(6, element_input_type, input_nopad, (element_input_type*)handle->reg_input->data, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
  LIBXSMM_VLA_DECL(5, element_input_type, tr_input_nopad, (element_input_type*)handle->scratch3, BLOCKSIFM, dst_ifhp, handle->ifmblock, ifwp_extended);
  for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ++ifm1) {
    for (ij = 0; ij < handle->ifhp; ++ij) {
      for (ifm2 = 0; ifm2 < 8; ++ifm2) {
        TRANSPOSE_W_CHUNK(img, ifm1, ij, 0, ifm2, 2*ifm1, 2*ifm2);
        TRANSPOSE_W_CHUNK(img, ifm1, ij, 0, ifm2+8, 2*ifm1+1, 2*ifm2);
        TRANSPOSE_W_REMAINDER(img, ifm1, ij, 16, ifm2, 2*ifm1, 2*ifm2);
        TRANSPOSE_W_REMAINDER(img, ifm1, ij, 16, ifm2+8,  2*ifm1+1, 2*ifm2);   
      }
    }
  }

  element_output_type *even_addr_lo, *odd_addr_lo, *even_addr_hi, *odd_addr_hi;
  element_output_type *dst_lo, *dst_hi;
  int half_i, ofm1,  ii;
  int BLOCKSOFM = handle->blocksofm;
  int OFWP = handle->ofwp+handle->output_lp_padding;
  __m256i even_pixel_lo, even_pixel_hi, odd_pixel_hi, odd_pixel_lo, compressed_hi, compressed_lo, compressed_lo_store, compressed_hi_store;

  /* Output transpose */
  element_output_type *out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock * handle->fm_lp_block;
  LIBXSMM_VLA_DECL(6, element_output_type, tr_output,  (element_output_type*)handle->scratch6 , BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2);
  LIBXSMM_VLA_DECL(6, element_output_type, output, out, handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);
  for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ++ofm1) {
    for (ij = 0; ij < handle->ofhp; ++ij) {
      for (ii = 0, half_i=0 ; ii < handle->ofwp-1; ii+=2, half_i++) {
        TRANSPOSE_W_FULL_PAIR(img, ofm1, ij, ii, half_i);
      }
    }
  }
}

void lp_transpose_3_chunk_1_remainder_even_pixels(int img, libxsmm_dnn_layer* handle) {
  typedef short element_input_type;
  typedef short element_output_type;

  int w_chunks = handle->ifwp/16;
  int w_remainder = handle->ifwp%16;
  int w_i, w, c_i, ifm1, ij, ifm2;
  int BLOCKSIFM = handle->blocksifm;;
  int padded_w = (handle->padding_flag == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
  int ifwp_extended = padded_w + handle->qfma_input_pad;
  int dst_ifhp;
  element_input_type *base_addr;
  const __m512i vgindex = _mm512_set_epi32(960,832,448,320,  704,576,192,64,  896,768,384,256,  640,512,128,0);
  const int gather_offsets[16] = {960,832,448,320,  704,576,192,64,  896,768,384,256,  640,512,128,0};
  const __m256i shuffler = _mm256_set_epi32(7,5,3,1,6,4,2,0);
  unsigned int int_mask = 0xffffffff;
  const __mmask16 gmask = int_mask;
  int mask_remainder = (w_remainder+1)/2;
  unsigned int mask[8];
  for (c_i=0;c_i<16;c_i++) {
    if (gather_offsets[16-c_i-1] >= w_remainder*64) {
      int_mask = int_mask & ~(1 << c_i);
    }
  }
  for (c_i=0; c_i<mask_remainder; c_i++) {
    mask[c_i] = (1<<31);
  }
  for (c_i=mask_remainder; c_i<8; c_i++) {
    mask[c_i] = 0;
  }
  __m256i mask_reg = _mm256_loadu_si256((const union __m256i *) mask);
  __m512i gather_reg;
  __m256i lo_reg, hi_reg, compressed_low, compressed_high, compressed_low_store, compressed_high_store;

  /* Input transpose  */
  dst_ifhp = handle->ifhp;
  LIBXSMM_VLA_DECL(6, element_input_type, input_nopad, (element_input_type*)handle->reg_input->data, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
  LIBXSMM_VLA_DECL(5, element_input_type, tr_input_nopad, (element_input_type*)handle->scratch3, BLOCKSIFM, dst_ifhp, handle->ifmblock, ifwp_extended);
  for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ++ifm1) {
    for (ij = 0; ij < handle->ifhp; ++ij) {
      for (ifm2 = 0; ifm2 < 8; ++ifm2) {
        TRANSPOSE_W_CHUNK(img, ifm1, ij, 0, ifm2, 2*ifm1, 2*ifm2);
        TRANSPOSE_W_CHUNK(img, ifm1, ij, 0, ifm2+8, 2*ifm1+1, 2*ifm2);
        TRANSPOSE_W_CHUNK(img, ifm1, ij, 16, ifm2, 2*ifm1, 2*ifm2);
        TRANSPOSE_W_CHUNK(img, ifm1, ij, 16, ifm2+8, 2*ifm1+1, 2*ifm2);
        TRANSPOSE_W_CHUNK(img, ifm1, ij, 32, ifm2, 2*ifm1, 2*ifm2);
        TRANSPOSE_W_CHUNK(img, ifm1, ij, 32, ifm2+8, 2*ifm1+1, 2*ifm2);   
        TRANSPOSE_W_REMAINDER(img, ifm1, ij, 48, ifm2, 2*ifm1, 2*ifm2);
        TRANSPOSE_W_REMAINDER(img, ifm1, ij, 48, ifm2+8,  2*ifm1+1, 2*ifm2);   
      }
    }
  }

  element_output_type *even_addr_lo, *odd_addr_lo, *even_addr_hi, *odd_addr_hi;
  element_output_type *dst_lo, *dst_hi;
  int half_i, ofm1,  ii;
  int BLOCKSOFM = handle->blocksofm;
  int OFWP = handle->ofwp+handle->output_lp_padding;
  __m256i even_pixel_lo, even_pixel_hi, odd_pixel_hi, odd_pixel_lo, compressed_hi, compressed_lo, compressed_lo_store, compressed_hi_store;

  /* Output transpose */
  element_output_type *out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock * handle->fm_lp_block;
  LIBXSMM_VLA_DECL(6, element_output_type, tr_output,  (element_output_type*)handle->scratch6 , BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2);
  LIBXSMM_VLA_DECL(6, element_output_type, output, out, handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);
  for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ++ofm1) {
    for (ij = 0; ij < handle->ofhp; ++ij) {
      for (ii = 0, half_i=0 ; ii < handle->ofwp-1; ii+=2, half_i++) {
        TRANSPOSE_W_FULL_PAIR(img, ofm1, ij, ii, half_i);
      }
    }
  }
}

void lp_transpose_and_resize_0_chunk_1_remainder_even_pixels(int img, libxsmm_dnn_layer* handle) {
  typedef short element_input_type;
  typedef short element_output_type;

  int w_chunks = handle->ifwp_resized/16;
  int w_remainder = handle->ifwp_resized%16;
  int u = handle->desc.u;
  int w_i, w, c_i, ifm1, ij, ifm2;
  int dst_i, dst_j, src_i, src_j;
  int BLOCKSIFM = handle->blocksifm;;
  int padded_w = (handle->padding_flag == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
  int ifwp_extended = padded_w + handle->qfma_input_pad;
  int dst_ifhp;
  element_input_type *base_addr;
  const __m512i vgindex = _mm512_set_epi32(u*960,u*832,u*448,u*320,  u*704,u*576,u*192,u*64,  u*896,u*768,u*384,u*256,  u*640,u*512,u*128, u*0);
  const int gather_offsets[16] = {u*960,u*832,u*448,u*320,  u*704,u*576,u*192,u*64,  u*896,u*768,u*384,u*256,  u*640,u*512,u*128, u*0};
  const __m256i shuffler = _mm256_set_epi32(7,5,3,1,6,4,2,0);
  unsigned int int_mask = 0xffffffff;
  const __mmask16 gmask = int_mask;
  int mask_remainder = (w_remainder+1)/2;
  unsigned int mask[8];
  for (c_i=0;c_i<16;c_i++) {
    if (gather_offsets[16-c_i-1] >= (w_remainder*64)*u) {
      int_mask = int_mask & ~(1 << c_i);
    }
  }
  for (c_i=0; c_i<mask_remainder; c_i++) {
    mask[c_i] = (1<<31);
  }
  for (c_i=mask_remainder; c_i<8; c_i++) {
    mask[c_i] = 0;
  }
  __m256i mask_reg = _mm256_loadu_si256((const union __m256i *) mask);
  __m512i gather_reg;
  __m256i lo_reg, hi_reg, compressed_low, compressed_high, compressed_low_store, compressed_high_store;

  /* Input transpose  */
  ifwp_extended = handle->ifwp_resized + handle->qfma_input_pad;
  dst_ifhp = handle->ifhp_resized;
  LIBXSMM_VLA_DECL(6, element_input_type, input_nopad, (element_input_type*)handle->reg_input->data, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
  LIBXSMM_VLA_DECL(5, element_input_type, tr_input_nopad, (element_input_type*)handle->scratch3, BLOCKSIFM, dst_ifhp, handle->ifmblock, ifwp_extended);
  for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ++ifm1) {
    for (dst_j=0; dst_j < handle->ifhp_resized; dst_j++) {
      src_j = dst_j * handle->desc.v;
      for (ifm2 = 0; ifm2 < 8; ++ifm2) {
        TRANSPOSE_W_REMAINDER_RESIZED(img, ifm1, 0, src_j, 0, dst_j, ifm2, 2*ifm1, 2*ifm2);  
        TRANSPOSE_W_REMAINDER_RESIZED(img, ifm1, 0, src_j, 0, dst_j, ifm2+8, 2*ifm1+1, 2*ifm2);
      }
    }
  }

  element_output_type *even_addr_lo, *odd_addr_lo, *even_addr_hi, *odd_addr_hi;
  element_output_type *dst_lo, *dst_hi;
  int half_i, ofm1,  ii;
  int BLOCKSOFM = handle->blocksofm;
  int OFWP = handle->ofwp+handle->output_lp_padding;
  __m256i even_pixel_lo, even_pixel_hi, odd_pixel_hi, odd_pixel_lo, compressed_hi, compressed_lo, compressed_lo_store, compressed_hi_store;
  /* Output transpose */
  element_output_type *out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock * handle->fm_lp_block;
  LIBXSMM_VLA_DECL(6, element_output_type, tr_output,  (element_output_type*)handle->scratch6 , BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2);
  LIBXSMM_VLA_DECL(6, element_output_type, output, out, handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);
  for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ++ofm1) {
    for (ij = 0; ij < handle->ofhp; ++ij) {
      for (ii = 0, half_i=0 ; ii < handle->ofwp-1; ii+=2, half_i++) {
        TRANSPOSE_W_FULL_PAIR(img, ofm1, ij, ii, half_i);
      }
    }
  }
}

void lp_transpose_and_resize_0_chunk_1_remainder_odd_pixels(int img, libxsmm_dnn_layer* handle) {
  typedef short element_input_type;
  typedef short element_output_type;

  int w_chunks = handle->ifwp_resized/16;
  int w_remainder = handle->ifwp_resized%16;
  int u = handle->desc.u;
  int w_i, w, c_i, ifm1, ij, ifm2;
  int dst_i, dst_j, src_i, src_j;
  int BLOCKSIFM = handle->blocksifm;;
  int padded_w = (handle->padding_flag == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
  int ifwp_extended = padded_w + handle->qfma_input_pad;
  int dst_ifhp;
  element_input_type *base_addr;
  const __m512i vgindex = _mm512_set_epi32(u*960,u*832,u*448,u*320,  u*704,u*576,u*192,u*64,  u*896,u*768,u*384,u*256,  u*640,u*512,u*128, u*0);
  const int gather_offsets[16] = {u*960,u*832,u*448,u*320,  u*704,u*576,u*192,u*64,  u*896,u*768,u*384,u*256,  u*640,u*512,u*128, u*0};
  const __m256i shuffler = _mm256_set_epi32(7,5,3,1,6,4,2,0);
  unsigned int int_mask = 0xffffffff;
  const __mmask16 gmask = int_mask;
  int mask_remainder = (w_remainder+1)/2;
  unsigned int mask[8];
  for (c_i=0;c_i<16;c_i++) {
    if (gather_offsets[16-c_i-1] >= (w_remainder*64)*u) {
      int_mask = int_mask & ~(1 << c_i);
    }
  }
  for (c_i=0; c_i<mask_remainder; c_i++) {
    mask[c_i] = (1<<31);
  }
  for (c_i=mask_remainder; c_i<8; c_i++) {
    mask[c_i] = 0;
  }
  __m256i mask_reg = _mm256_loadu_si256((const union __m256i *) mask);
  __m512i gather_reg;
  __m256i lo_reg, hi_reg, compressed_low, compressed_high, compressed_low_store, compressed_high_store;

  /* Input transpose  */
  ifwp_extended = handle->ifwp_resized + handle->qfma_input_pad;
  dst_ifhp = handle->ifhp_resized;
  LIBXSMM_VLA_DECL(6, element_input_type, input_nopad, (element_input_type*)handle->reg_input->data, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
  LIBXSMM_VLA_DECL(5, element_input_type, tr_input_nopad, (element_input_type*)handle->scratch3, BLOCKSIFM, dst_ifhp, handle->ifmblock, ifwp_extended);
  for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ++ifm1) {
    for (dst_j=0; dst_j < handle->ifhp_resized; dst_j++) {
      src_j = dst_j * handle->desc.v;
      for (ifm2 = 0; ifm2 < 8; ++ifm2) {
        TRANSPOSE_W_REMAINDER_RESIZED(img, ifm1, 0, src_j, 0, dst_j, ifm2, 2*ifm1, 2*ifm2);  
        TRANSPOSE_W_REMAINDER_RESIZED(img, ifm1, 0, src_j, 0, dst_j, ifm2+8, 2*ifm1+1, 2*ifm2);
      }
    }
  }

  element_output_type *even_addr_lo, *odd_addr_lo, *even_addr_hi, *odd_addr_hi;
  element_output_type *dst_lo, *dst_hi;
  int half_i, ofm1,  ii;
  int BLOCKSOFM = handle->blocksofm;
  int OFWP = handle->ofwp+handle->output_lp_padding;
  __m256i even_pixel_lo, even_pixel_hi, odd_pixel_hi, odd_pixel_lo, compressed_hi, compressed_lo, compressed_lo_store, compressed_hi_store;
  /* Output transpose */
  element_output_type *out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock * handle->fm_lp_block;
  LIBXSMM_VLA_DECL(6, element_output_type, tr_output,  (element_output_type*)handle->scratch6 , BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2);
  LIBXSMM_VLA_DECL(6, element_output_type, output, out, handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);
  for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ++ofm1) {
    for (ij = 0; ij < handle->ofhp; ++ij) {
      for (ii = 0, half_i=0 ; ii < handle->ofwp-1; ii+=2, half_i++) {
        TRANSPOSE_W_FULL_PAIR(img, ofm1, ij, ii, half_i);
      }
    }
  }

  for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ++ofm1) {
    for (ij = 0; ij < handle->ofhp; ++ij) {
      ii = handle->ofwp-1;
      half_i = ii/2;
      TRANSPOSE_W_HALF_PAIR(img, ofm1, ij, ii, half_i);
    }
  }
}

void lp_transpose_and_resize_1_chunk_1_remainder_even_pixels(int img, libxsmm_dnn_layer* handle) {
  typedef short element_input_type;
  typedef short element_output_type;

  int w_chunks = handle->ifwp_resized/16;
  int w_remainder = handle->ifwp_resized%16;
  int u = handle->desc.u;
  int w_i, w, c_i, ifm1, ij, ifm2;
  int dst_i, dst_j, src_i, src_j;
  int BLOCKSIFM = handle->blocksifm;;
  int padded_w = (handle->padding_flag == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
  int ifwp_extended = padded_w + handle->qfma_input_pad;
  int dst_ifhp;
  element_input_type *base_addr;
  const __m512i vgindex = _mm512_set_epi32(u*960,u*832,u*448,u*320,  u*704,u*576,u*192,u*64,  u*896,u*768,u*384,u*256,  u*640,u*512,u*128, u*0);
  const int gather_offsets[16] = {u*960,u*832,u*448,u*320,  u*704,u*576,u*192,u*64,  u*896,u*768,u*384,u*256,  u*640,u*512,u*128, u*0};
  const __m256i shuffler = _mm256_set_epi32(7,5,3,1,6,4,2,0);
  unsigned int int_mask = 0xffffffff;
  const __mmask16 gmask = int_mask;
  int mask_remainder = (w_remainder+1)/2;
  unsigned int mask[8];
  for (c_i=0;c_i<16;c_i++) {
    if (gather_offsets[16-c_i-1] >= (w_remainder*64)*u) {
      int_mask = int_mask & ~(1 << c_i);
    }
  }
  for (c_i=0; c_i<mask_remainder; c_i++) {
    mask[c_i] = (1<<31);
  }
  for (c_i=mask_remainder; c_i<8; c_i++) {
    mask[c_i] = 0;
  }
  __m256i mask_reg = _mm256_loadu_si256((const union __m256i *) mask);
  __m512i gather_reg;
  __m256i lo_reg, hi_reg, compressed_low, compressed_high, compressed_low_store, compressed_high_store;

  /* Input transpose  */
  ifwp_extended = handle->ifwp_resized + handle->qfma_input_pad;
  dst_ifhp = handle->ifhp_resized;
  LIBXSMM_VLA_DECL(6, element_input_type, input_nopad, (element_input_type*)handle->reg_input->data, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
  LIBXSMM_VLA_DECL(5, element_input_type, tr_input_nopad, (element_input_type*)handle->scratch3, BLOCKSIFM, dst_ifhp, handle->ifmblock, ifwp_extended);
  for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ++ifm1) {
    for (dst_j=0; dst_j < handle->ifhp_resized; dst_j++) {
      src_j = dst_j * handle->desc.v;
      for (ifm2 = 0; ifm2 < 8; ++ifm2) {
        TRANSPOSE_W_CHUNK_RESIZED(img, ifm1, 0, src_j, 0, dst_j, ifm2, 2*ifm1, 2*ifm2);
        TRANSPOSE_W_CHUNK_RESIZED(img, ifm1, 0, src_j, 0, dst_j, ifm2+8, 2*ifm1+1, 2*ifm2);
        TRANSPOSE_W_REMAINDER_RESIZED(img, ifm1, u*16, src_j, 16, dst_j, ifm2, 2*ifm1, 2*ifm2);  
        TRANSPOSE_W_REMAINDER_RESIZED(img, ifm1, u*16, src_j, 16, dst_j, ifm2+8, 2*ifm1+1, 2*ifm2);
      }
    }
  }

  element_output_type *even_addr_lo, *odd_addr_lo, *even_addr_hi, *odd_addr_hi;
  element_output_type *dst_lo, *dst_hi;
  int half_i, ofm1,  ii;
  int BLOCKSOFM = handle->blocksofm;
  int OFWP = handle->ofwp+handle->output_lp_padding;
  __m256i even_pixel_lo, even_pixel_hi, odd_pixel_hi, odd_pixel_lo, compressed_hi, compressed_lo, compressed_lo_store, compressed_hi_store;
  /* Output transpose */
  element_output_type *out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock * handle->fm_lp_block;
  LIBXSMM_VLA_DECL(6, element_output_type, tr_output,  (element_output_type*)handle->scratch6 , BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2);
  LIBXSMM_VLA_DECL(6, element_output_type, output, out, handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);
  for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ++ofm1) {
    for (ij = 0; ij < handle->ofhp; ++ij) {
      for (ii = 0, half_i=0 ; ii < handle->ofwp-1; ii+=2, half_i++) {
        TRANSPOSE_W_FULL_PAIR(img, ofm1, ij, ii, half_i);
      }
    }
  }
}
#endif

#undef TRANSPOSE_W_CHUNK
#undef TRANSPOSE_W_REMAINDER
#undef TRANSPOSE_W_FULL_PAIR
#undef TRANSPOSE_W_HALF_PAIR
#undef TRANSPOSE_W_CHUNK_RESIZED
#undef TRANSPOSE_W_REMAINDER_RESIZED

#if 1
void gather_transpose_ps_16_56_56_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x00FF;
  int m;
#pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n;
#pragma unroll(3)
    for(n = 0; n < 3; ++n) {
      const __m512 tmp =  _mm512_i32gather_ps(vindex, src+m+n*256, 4);
      _mm512_store_ps((void*)(dst+m*56+n*16),tmp);
    }
    const __m512 tmprem =  _mm512_mask_i32gather_ps(_mm512_undefined(), Nremmask, vindex, src+m+n*256, 4);
    _mm512_mask_store_ps((void*)(dst+m*56+n*16),Nremmask,tmprem);
  }
}

void gather_transpose_ps_16_56_58_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x00FF;
  int m;
#pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n;
#pragma unroll(3)
    for(n = 0; n < 3; ++n) {
      const __m512 tmp =  _mm512_i32gather_ps(vindex, src+m+n*256, 4);
      _mm512_store_ps((void*)(dst+m*58+n*16),tmp);
    }
    const __m512 tmprem =  _mm512_mask_i32gather_ps(_mm512_undefined(), Nremmask, vindex, src+m+n*256, 4);
    _mm512_mask_store_ps((void*)(dst+m*58+n*16),Nremmask,tmprem);
  }
}

void gather_transpose_ps_16_58_60_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x03FF;
  int m;
#pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n;
#pragma unroll(3)
    for(n = 0; n < 3; ++n) {
      const __m512 tmp =  _mm512_i32gather_ps(vindex, src+m+n*256, 4);
      _mm512_store_ps((void*)(dst+m*60+n*16),tmp);
    }
    const __m512 tmprem =  _mm512_mask_i32gather_ps(_mm512_undefined(), Nremmask, vindex, src+m+n*256, 4);
    _mm512_mask_store_ps((void*)(dst+m*60+n*16),Nremmask,tmprem);
  }
}

void gather_transpose_ps_16_58_58_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x03FF;
  int m;
#pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n;
#pragma unroll(3)
    for(n = 0; n < 3; ++n) {
      const __m512 tmp =  _mm512_i32gather_ps(vindex, src+m+n*256, 4);
      _mm512_store_ps((void*)(dst+m*58+n*16),tmp);
    }
    const __m512 tmprem =  _mm512_mask_i32gather_ps(_mm512_undefined(), Nremmask, vindex, src+m+n*256, 4);
    _mm512_mask_store_ps((void*)(dst+m*58+n*16),Nremmask,tmprem);
  }
}

void gather_transpose_ps_16_28_28_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x0FFF;
  int m;
#pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmp =  _mm512_i32gather_ps(vindex, src+m+n*256, 4);
    _mm512_store_ps((void*)(dst+m*28+n*16),tmp);
    n = 1;
    const __m512 tmprem =  _mm512_mask_i32gather_ps(_mm512_undefined(), Nremmask, vindex, src+m+n*256, 4);
    _mm512_mask_store_ps((void*)(dst+m*28+n*16),Nremmask,tmprem);
  }
}

void gather_transpose_ps_16_28_30_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x0FFF;
  int m;
#pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmp =  _mm512_i32gather_ps(vindex, src+m+n*256, 4);
    _mm512_store_ps((void*)(dst+m*30+n*16),tmp);
    n = 1;
    const __m512 tmprem =  _mm512_mask_i32gather_ps(_mm512_undefined(), Nremmask, vindex, src+m+n*256, 4);
    _mm512_mask_store_ps((void*)(dst+m*30+n*16),Nremmask,tmprem);
  }
}

void gather_transpose_ps_16_30_32_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x3FFF;
  int m;
#pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmp =  _mm512_i32gather_ps(vindex, src+m+n*256, 4);
    _mm512_store_ps((void*)(dst+m*32+n*16),tmp);
    n = 1;
    const __m512 tmprem =  _mm512_mask_i32gather_ps(_mm512_undefined(), Nremmask, vindex, src+m+n*256, 4);
    _mm512_mask_store_ps((void*)(dst+m*32+n*16),Nremmask,tmprem);
  }
}

void gather_transpose_ps_16_30_30_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x3FFF;
  int m;
#pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmp =  _mm512_i32gather_ps(vindex, src+m+n*256, 4);
    _mm512_store_ps((void*)(dst+m*30+n*16),tmp);
    n = 1;
    const __m512 tmprem =  _mm512_mask_i32gather_ps(_mm512_undefined(), Nremmask, vindex, src+m+n*256, 4);
    _mm512_mask_store_ps((void*)(dst+m*30+n*16),Nremmask,tmprem);
  }
}
void gather_transpose_ps_16_16_16_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  int m;
#pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmp =  _mm512_i32gather_ps(vindex, src+m+n*256, 4);
    _mm512_store_ps((void*)(dst+m*16+n*16),tmp);
  }
}

void gather_transpose_ps_16_16_18_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  int m;
#pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmp =  _mm512_i32gather_ps(vindex, src+m+n*256, 4);
    _mm512_store_ps((void*)(dst+m*18+n*16),tmp);
  }
}

void gather_transpose_ps_16_14_16_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x3FFF;
  int m;
#pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmprem =  _mm512_mask_i32gather_ps(_mm512_undefined(), Nremmask, vindex, src+m+n*256, 4);
    _mm512_mask_store_ps((void*)(dst+m*16+n*16),Nremmask,tmprem);
  }
}

void gather_transpose_ps_16_14_18_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x3FFF;
  int m;
#pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmprem =  _mm512_mask_i32gather_ps(_mm512_undefined(), Nremmask, vindex, src+m+n*256, 4);
    _mm512_mask_store_ps((void*)(dst+m*18+n*16),Nremmask,tmprem);
  }
}

void gather_transpose_ps_16_7_8_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(0, 97, 81, 65, 49, 33, 17,  1,
      0, 96, 80, 64, 48, 32, 16,  0);
  const __mmask16 Nremmask = 0x7F7F;
  int m;
#pragma unroll_and_jam(4)
  for(m = 0; m < 8; ++m) {
    const __m512 tmprem =  _mm512_mask_i32gather_ps(_mm512_undefined(), Nremmask, vindex, src+m*2, 4);
    _mm512_mask_store_ps((void*)(dst+m*8*2),Nremmask,tmprem);
  }
}

void gather_transpose_ps_16_7_10_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x07F;
  int m;
#pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmprem =  _mm512_mask_i32gather_ps(_mm512_undefined(), Nremmask, vindex, src+m+n*256, 4);
    _mm512_mask_store_ps((void*)(dst+m*10+n*16),Nremmask,tmprem);
  }
}

void gather_transpose_ps_16_9_12_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x01FF;
  int m;
#pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmprem =  _mm512_mask_i32gather_ps(_mm512_undefined(), Nremmask, vindex, src+m+n*256, 4);
    _mm512_mask_store_ps((void*)(dst+m*12+n*16),Nremmask,tmprem);
  }
}

void gather_transpose_ps_16_9_10_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x01FF;
  int m;
#pragma unroll_and_jam(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmprem =  _mm512_mask_i32gather_ps(_mm512_undefined(), Nremmask, vindex, src+m+n*256, 4);
    _mm512_mask_store_ps((void*)(dst+m*10+n*16),Nremmask,tmprem);
  }
}

void transpose_fallback(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  const __m512i vindex_base = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
  const __m512i vindex = _mm512_mullo_epi32(_mm512_set1_epi32(ldS), vindex_base);

  const int whole16s = N/16;
  const int remainder = N-whole16s*16;
  const __mmask16 Nmask = (1<<remainder)-1;
  int i;
#pragma unroll_and_jam(2)
  for(i = 0; i < M; ++i) {
    int j;
#pragma unroll(4)
    for(j = 0; j < whole16s; ++j) {
      const __m512 res = _mm512_i32gather_ps(vindex, src+i+j*16*ldS, 4);
      _mm512_store_ps(dst + ldD*i+j*16, res);
    }
    if(remainder) {
      const __m512 res = _mm512_mask_i32gather_ps(_mm512_undefined(), Nmask, vindex, src+i+j*16*ldS, 4);
      _mm512_mask_store_ps(dst + ldD*i+j*16, Nmask, res);
    }
  }
}
#else
void transpose_fallback(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS) {
  int n, m;
  for (n = 0; n < N; ++n) {
    for (m = 0; m < M; ++m) {
      dst[m*ldD + n] = src[n*ldS + m];
    }
  }
}
#endif //defined(__AVX512F__)

typedef void (*transposer)(int M, int N, float *dst, int ldD, const float *src, int ldS);

transposer get_transposer(int M, int N, int ldD, int ldS) {
  if(M == 16 && N == 7 && ldD == 8 && ldS == 16) {
    return gather_transpose_ps_16_7_8_16;
  }
  if(M == 16 && N == 7 && ldD == 10 && ldS == 16) {
    return gather_transpose_ps_16_7_10_16;
  }
  if(M == 16 && N == 9 && ldD == 10 && ldS == 16) {
    return gather_transpose_ps_16_9_10_16;
  }
  if(M == 16 && N == 9 && ldD == 12 && ldS == 16) {
    return gather_transpose_ps_16_9_12_16;
  }
  if(M == 16 && N == 14 && ldD == 16 && ldS == 16) {
    return gather_transpose_ps_16_14_16_16;
  }
  if(M == 16 && N == 14 && ldD == 18 && ldS == 16) {
    return gather_transpose_ps_16_14_18_16;
  }
  if(M == 16 && N == 16 && ldD == 16 && ldS == 16) {
    return gather_transpose_ps_16_16_16_16;
  }
  if(M == 16 && N == 16 && ldD == 18 && ldS == 16) {
    return gather_transpose_ps_16_16_18_16;
  }
  if(M == 16 && N == 28 && ldD == 28 && ldS == 16) {
    return gather_transpose_ps_16_28_28_16;
  }
  if(M == 16 && N == 28 && ldD == 30 && ldS == 16) {
    return gather_transpose_ps_16_28_30_16;
  }
  if(M == 16 && N == 30 && ldD == 30 && ldS == 16) {
    return gather_transpose_ps_16_30_30_16;
  }
  if(M == 16 && N == 30 && ldD == 32 && ldS == 16) {
    return gather_transpose_ps_16_30_32_16;
  }
  if(M == 16 && N == 56 && ldD == 56 && ldS == 16) {
    return gather_transpose_ps_16_56_56_16;
  }
  if(M == 16 && N == 56 && ldD == 58 && ldS == 16) {
    return gather_transpose_ps_16_56_58_16;
  }
  if(M == 16 && N == 58 && ldD == 58 && ldS == 16) {
    return gather_transpose_ps_16_58_58_16;
  }
  if(M == 16 && N == 58 && ldD == 60 && ldS == 16) {
    return gather_transpose_ps_16_58_60_16;
  }

  return transpose_fallback;
}

LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_convolve_st_upd_custom_custom(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->reg_input == 0 || handle->grad_output == 0 || handle->grad_filter == 0 || handle->scratch3 == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we scratch for MB parallel execution */
  if ( (handle->upd_use_thread_fil == 1) && (handle->scratch4 == 0) ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if (handle->code_upd[0].xconv.sconv == 0) {
#if 0
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
# include "template/libxsmm_dnn_convolve_st_upd_custom_custom_fallback.tpl.c"
#undef INPUT_PADDING
      } else {
# include "template/libxsmm_dnn_convolve_st_upd_custom_custom_fallback.tpl.c"
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
#endif
  }
  else {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      if (handle->upd_use_thread_fil > 0) {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
        if (handle->padding_flag == 1) {
#define INPUT_PADDING
#define LIBXSMM_WU_PER_THREAD_ALLOCATION
# include "template/libxsmm_dnn_convolve_st_upd_custom_custom.tpl.c"
#undef LIBXSMM_WU_PER_THREAD_ALLOCATION
#undef INPUT_PADDING
        } else {
#define LIBXSMM_WU_PER_THREAD_ALLOCATION
# include "template/libxsmm_dnn_convolve_st_upd_custom_custom.tpl.c"
#undef LIBXSMM_WU_PER_THREAD_ALLOCATION
        }
      }
#if 1
      else {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
        if (handle->padding_flag == 1) {
#define INPUT_PADDING
          if ( (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM)
              && (handle->desc.v == 1) && (handle->upd_ofw_rb%4 == 0) )
          {
#define LIBXSMM_WU_TRANSPOSE_OFW_IFM
# include "template/libxsmm_dnn_convolve_st_upd_custom_custom.tpl.c"
#undef LIBXSMM_WU_TRANSPOSE_OFW_IFM
          } else {
# include "template/libxsmm_dnn_convolve_st_upd_custom_custom.tpl.c"
          }
#undef INPUT_PADDING
        } else {
          if ( (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM)
              && (handle->desc.v == 1) && (handle->upd_ofw_rb%4 == 0) )
          {
#define LIBXSMM_WU_TRANSPOSE_OFW_IFM
# include "template/libxsmm_dnn_convolve_st_upd_custom_custom.tpl.c"
#undef LIBXSMM_WU_TRANSPOSE_OFW_IFM
          } else {
# include "template/libxsmm_dnn_convolve_st_upd_custom_custom.tpl.c"
          }
        }
      }
#endif
    } else if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      if (handle->upd_use_thread_fil > 0) {
        typedef short element_input_type;
        typedef short element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_uwsconvfunction libxsmm_convfunction;
        if (handle->use_fastpath) {
          if ( handle->use_hybrid_wu_parallelism == 1) {
#include "template/libxsmm_dnn_convolve_st_upd_custom_custom_stream_lp.tpl.c"
          } else {
#include "template/libxsmm_dnn_convolve_st_upd_custom_custom_stream_opt_lp.tpl.c"
          }
        } 
#if 0
        if (handle->padding_flag == 1) {
#define INPUT_PADDING
#define LIBXSMM_WU_PER_THREAD_ALLOCATION
# include "template/libxsmm_dnn_convolve_st_upd_custom_custom.tpl.c"
#undef LIBXSMM_WU_PER_THREAD_ALLOCATION
#undef INPUT_PADDING
        } else {
#define LIBXSMM_WU_PER_THREAD_ALLOCATION
# include "template/libxsmm_dnn_convolve_st_upd_custom_custom.tpl.c"
#undef LIBXSMM_WU_PER_THREAD_ALLOCATION
        }
#endif
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_convolve_st_upd_nhwc_rsck(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->reg_input == 0 || handle->grad_output == 0 || handle->grad_filter == 0 || handle->scratch3 == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we scratch for MB parallel execution */
  if ( (handle->upd_use_thread_fil == 1) && (handle->scratch4 == 0) ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

#if 1
  /* @TODO FIXME */
  printf("no nhwc_rsck update\n");
  return(-1);
#else
  /* check if we have a kernel JITed */
  if (handle->code_upd[0].xconv.sconv == 0) {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_rsck_fallback.tpl.c"
#undef INPUT_PADDING
      } else {
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_rsck_fallback.tpl.c"
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      if (handle->upd_use_thread_fil > 0) {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
        if (handle->padding_flag == 1) {
#define LIBXSMM_WU_PER_THREAD_ALLOCATION
#define INPUT_PADDING
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_rsck.tpl.c"
#undef INPUT_PADDING
#undef LIBXSMM_WU_PER_THREAD_ALLOCATION
        } else {
#define LIBXSMM_WU_PER_THREAD_ALLOCATION
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_rsck.tpl.c"
#undef LIBXSMM_WU_PER_THREAD_ALLOCATION
        }
      }
      else {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
        if (handle->padding_flag == 1) {
#define INPUT_PADDING
          if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM )
          {
#define LIBXSMM_WU_TRANSPOSE_OFW_IFM
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_rsck.tpl.c"
#undef LIBXSMM_WU_TRANSPOSE_OFW_IFM
          } else {
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_rsck.tpl.c"
          }
#undef INPUT_PADDING
        } else {
          if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM )
          {
#define LIBXSMM_WU_TRANSPOSE_OFW_IFM
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_rsck.tpl.c"
#undef LIBXSMM_WU_TRANSPOSE_OFW_IFM
          } else {
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_rsck.tpl.c"
          }
        }
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
#endif

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_convolve_st_upd_nhwc_custom(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->reg_input == 0 || handle->grad_output == 0 || handle->grad_filter == 0 || handle->scratch3 == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we scratch for MB parallel execution */
  if ( (handle->upd_use_thread_fil == 1) && (handle->scratch4 == 0) ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

#if 1
  /* @TODO FIXME */
  printf("no nhwc_custom update\n");
  return(-1);
#else
  /* check if we have a kernel JITed */
  if (handle->code_upd[0].xconv.sconv == 0) {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_custom_fallback.tpl.c"
#undef INPUT_PADDING
      } else {
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_custom_fallback.tpl.c"
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      if (handle->upd_use_thread_fil > 0) {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
        if (handle->padding_flag == 1) {
#define INPUT_PADDING
#define LIBXSMM_WU_PER_THREAD_ALLOCATION
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_custom.tpl.c"
#undef LIBXSMM_WU_PER_THREAD_ALLOCATION
#undef INPUT_PADDING
        } else {
#define LIBXSMM_WU_PER_THREAD_ALLOCATION
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_custom.tpl.c"
#undef LIBXSMM_WU_PER_THREAD_ALLOCATION
        }
      }
      else {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
        if (handle->padding_flag == 1) {
#define INPUT_PADDING
          if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM )
          {
#define LIBXSMM_WU_TRANSPOSE_OFW_IFM
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_custom.tpl.c"
#undef LIBXSMM_WU_TRANSPOSE_OFW_IFM
          } else {
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_custom.tpl.c"
          }
#undef INPUT_PADDING
        } else {
          if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM )
          {
#define LIBXSMM_WU_TRANSPOSE_OFW_IFM
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_custom.tpl.c"
#undef LIBXSMM_WU_TRANSPOSE_OFW_IFM
          } else {
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_custom.tpl.c"
          }
        }
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
#endif

  return status;
}


