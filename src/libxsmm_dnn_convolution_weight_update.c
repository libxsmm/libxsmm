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
/* Rajkishore Barik, Alexander Heinecke, Ankush Mandal, Jason Sewall (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_convolution_weight_update.h"
#include <libxsmm_intrinsics_x86.h>
#include "libxsmm_main.h"
#include <libxsmm.h>
#include <stdio.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <string.h>
#include <math.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

/* function prototypes for below implementations */
LIBXSMM_API_INTERN void lp_transpose_input_and_output(int ltid, libxsmm_dnn_layer* handle);
LIBXSMM_API_INTERN void lp_transpose_and_resize_input_and_output(int ltid, libxsmm_dnn_layer* handle);
LIBXSMM_API_INTERN void transpose_fallback(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*transposer)(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS);
LIBXSMM_API_INTERN transposer get_transposer(int M, int N, int ldD, int ldS);

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_upd_custom_custom_f32_f32(libxsmm_dnn_layer* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_upd_custom_custom_i16_i32(libxsmm_dnn_layer* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_upd_custom_custom_i16_f32(libxsmm_dnn_layer* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_upd_custom_custom_bf16_bf16(libxsmm_dnn_layer* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_upd_custom_custom_i8_i32(libxsmm_dnn_layer* handle, int start_thread, int tid);

#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
#define TRANSPOSE_W_CHUNK(img, ifm1, ij, w_offset, ifm2) \
        base_addr = &LIBXSMM_VLA_ACCESS(6, input_nopad, img, ifm1, ij, w_offset, ifm2, 0, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block); \
        gather_reg = _mm512_i32gather_epi32(vgindex, (const int*)base_addr, 1); \
        lo_reg = LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(gather_reg,0); \
        hi_reg = LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(gather_reg,1); \
        compressed_low = _mm256_unpacklo_epi16(lo_reg, hi_reg); \
        compressed_low = _mm256_permutevar8x32_epi32(compressed_low, shuffler); \
        compressed_high = _mm256_unpackhi_epi16(lo_reg, hi_reg); \
        compressed_high = _mm256_permutevar8x32_epi32(compressed_high, shuffler); \
        compressed_low_store = _mm256_undefined_si256(); compressed_high_store = _mm256_undefined_si256(); \
        compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_low,0), 0); \
        compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_high, 0), 1); \
        compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_low,1), 0); \
        compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_high, 1), 1); \
        _mm256_storeu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, ifm1, ij, 2*ifm2, w_offset, BLOCKSIFM, handle->ifhp, handle->ifmblock_hp, ifwp_extended), compressed_low_store); \
        _mm256_storeu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, ifm1, ij, 2*ifm2+1, w_offset, BLOCKSIFM, handle->ifhp, handle->ifmblock_hp, ifwp_extended), compressed_high_store)

#define TRANSPOSE_W_REMAINDER(img, ifm1, ij, w_offset, ifm2) \
        base_addr = &LIBXSMM_VLA_ACCESS(6, input_nopad, img, ifm1, ij, w_offset, ifm2, 0, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block); \
        gather_reg = LIBXSMM_INTRINSICS_MM512_MASK_I32GATHER_EPI32(LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), gmask, vgindex, base_addr, 1); \
        lo_reg = LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(gather_reg,0); \
        hi_reg = LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(gather_reg,1); \
        compressed_low = _mm256_unpacklo_epi16(lo_reg, hi_reg); \
        compressed_low = _mm256_permutevar8x32_epi32(compressed_low, shuffler); \
        compressed_high = _mm256_unpackhi_epi16(lo_reg, hi_reg); \
        compressed_high = _mm256_permutevar8x32_epi32(compressed_high, shuffler); \
        compressed_low_store = _mm256_undefined_si256(); compressed_high_store = _mm256_undefined_si256(); \
        compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_low,0), 0); \
        compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_high, 0), 1); \
        compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_low,1), 0); \
        compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_high, 1), 1); \
        _mm256_maskstore_epi32((int*) &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, ifm1, ij, 2*ifm2, w_offset, BLOCKSIFM, handle->ifhp, handle->ifmblock_hp, ifwp_extended), mask_reg, compressed_low_store); \
        _mm256_maskstore_epi32((int*) &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, ifm1, ij, 2*ifm2+1, w_offset, BLOCKSIFM, handle->ifhp, handle->ifmblock_hp, ifwp_extended), mask_reg, compressed_high_store)

#define TRANSPOSE_W_CHUNK_RESIZED(img, ifm1, w_offset, ij, ifm2, dst_i, dst_j) \
        base_addr = &LIBXSMM_VLA_ACCESS(6, input_nopad, img, ifm1, ij, w_offset, ifm2, 0, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block); \
        gather_reg = _mm512_i32gather_epi32(vgindex, (const int*)base_addr, 1); \
        lo_reg = LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(gather_reg,0); \
        hi_reg = LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(gather_reg,1); \
        compressed_low = _mm256_unpacklo_epi16(lo_reg, hi_reg); \
        compressed_low = _mm256_permutevar8x32_epi32(compressed_low, shuffler); \
        compressed_high = _mm256_unpackhi_epi16(lo_reg, hi_reg); \
        compressed_high = _mm256_permutevar8x32_epi32(compressed_high, shuffler); \
        compressed_low_store = _mm256_undefined_si256(); compressed_high_store = _mm256_undefined_si256(); \
        compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_low,0), 0); \
        compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_high, 0), 1); \
        compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_low,1), 0); \
        compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_high, 1), 1); \
        _mm256_storeu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, ifm1, dst_j, 2*ifm2, dst_i, BLOCKSIFM, handle->ifhp_resized, handle->ifmblock_hp, ifwp_extended), compressed_low_store); \
        _mm256_storeu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, ifm1, dst_j, 2*ifm2+1, dst_i, BLOCKSIFM, handle->ifhp_resized, handle->ifmblock_hp, ifwp_extended), compressed_high_store)

#define TRANSPOSE_W_REMAINDER_RESIZED(img, ifm1, w_offset, ij, ifm2, dst_i, dst_j) \
        base_addr = &LIBXSMM_VLA_ACCESS(6, input_nopad, img, ifm1, ij, w_offset, ifm2, 0, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block); \
        gather_reg = LIBXSMM_INTRINSICS_MM512_MASK_I32GATHER_EPI32(LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), gmask, vgindex, base_addr, 1); \
        lo_reg = LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(gather_reg,0); \
        hi_reg = LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(gather_reg,1); \
        compressed_low = _mm256_unpacklo_epi16(lo_reg, hi_reg); \
        compressed_low = _mm256_permutevar8x32_epi32(compressed_low, shuffler); \
        compressed_high = _mm256_unpackhi_epi16(lo_reg, hi_reg); \
        compressed_high = _mm256_permutevar8x32_epi32(compressed_high, shuffler); \
        compressed_low_store = _mm256_undefined_si256(); compressed_high_store = _mm256_undefined_si256(); \
        compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_low,0), 0); \
        compressed_low_store = _mm256_insertf128_si256(compressed_low_store, _mm256_extractf128_si256(compressed_high, 0), 1); \
        compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_low,1), 0); \
        compressed_high_store = _mm256_insertf128_si256(compressed_high_store, _mm256_extractf128_si256(compressed_high, 1), 1); \
        _mm256_maskstore_epi32((int*) &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, ifm1, dst_j, 2*ifm2, dst_i, BLOCKSIFM, handle->ifhp_resized, handle->ifmblock_hp, ifwp_extended), mask_reg, compressed_low_store); \
        _mm256_maskstore_epi32((int*) &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, ifm1, dst_j, 2*ifm2+1, dst_i, BLOCKSIFM, handle->ifhp_resized, handle->ifmblock_hp, ifwp_extended), mask_reg, compressed_high_store)

#define TRANSPOSE_W_FULL_PAIR(img, ofm1, ij, ii, half_i) \
      pair_addr = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, ii, 0, 0,  handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block); \
      pair_pixels = _mm512_loadu_si512(pair_addr); \
      even_pixel = LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(pair_pixels, 0); \
      odd_pixel = LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(pair_pixels, 1); \
      compressed_lo = _mm256_unpacklo_epi16(even_pixel, odd_pixel); \
      compressed_hi = _mm256_unpackhi_epi16(even_pixel, odd_pixel); \
      compact = _mm512_inserti64x4(LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), compressed_lo, 0); \
      compact = _mm512_inserti64x4(compact, compressed_hi, 1); \
      compact = _mm512_permutexvar_epi32(permute_compact_idx, compact); \
      pair_addr_dst = &LIBXSMM_VLA_ACCESS(6,  tr_output, img, ofm1, ij, half_i, 0, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2); \
      LIBXSMM_INTRINSICS_MM512_STREAM_SI512((void*)pair_addr_dst, compact)

#define TRANSPOSE_W_HALF_PAIR(img, ofm1, ij, ii, half_i) \
      pair_addr = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, ii, 0, 0,  handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block); \
      even_pixel = _mm256_loadu_si256((const __m256i*)pair_addr); \
      odd_pixel = _mm256_xor_si256(odd_pixel, odd_pixel); \
      compressed_lo = _mm256_unpacklo_epi16(even_pixel, odd_pixel); \
      compressed_hi = _mm256_unpackhi_epi16(even_pixel, odd_pixel); \
      compact = _mm512_inserti64x4(LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), compressed_lo, 0); \
      compact = _mm512_inserti64x4(compact, compressed_hi, 1); \
      compact = _mm512_permutexvar_epi32(permute_compact_idx, compact); \
      pair_addr_dst = &LIBXSMM_VLA_ACCESS(6,  tr_output, img, ofm1, ij, half_i, 0, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2); \
      LIBXSMM_INTRINSICS_MM512_STREAM_SI512((void*)pair_addr_dst, compact)

#if defined(LIBXSMM_INTRINSICS_AVX512_CORE)
LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
void lp_transpose_input_and_output_vperm(int my_img_start, int my_img_end, libxsmm_dnn_layer* handle)
{
  typedef short element_input_type;
  typedef short element_output_type;
  int img;

  if (handle->trans_ofw_ifm == 1) {
    int w_chunks = handle->ifwp / 16;
    int w_remainder = handle->ifwp % 16;
    int w, c_i, ifm1, ij, ifm2;
    int BLOCKSIFM = handle->blocksifm;
    int padded_w = (handle->padding_flag == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
    int ifwp_extended = padded_w + handle->qfma_input_pad;
    int dst_ifhp = handle->ifhp;
    element_input_type *base_addr;
    const __m512i vgindex = _mm512_set_epi32(480, 416, 224, 160, 352, 288, 96, 32, 448, 384, 192, 128, 320, 256, 64, 0);
    const int gather_offsets[16] = { 480,416,224,160,  352,288,96,32,  448,384,192,128,  320,256,64,0 };
    const __m256i shuffler = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    unsigned int int_mask = 0xffffffff;
    const __mmask16 gmask = (__mmask16)int_mask;
    int mask_remainder = (w_remainder + 1) / 2;
    /* Input transpose  */
    LIBXSMM_VLA_DECL(6, element_input_type, input_nopad, (element_input_type*)handle->reg_input->data, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
    LIBXSMM_VLA_DECL(5, element_input_type, tr_input_nopad, (element_input_type*)handle->scratch3, BLOCKSIFM, dst_ifhp, handle->ifmblock_hp, ifwp_extended);
    __m256i mask_reg, lo_reg, hi_reg, compressed_low, compressed_high, compressed_low_store, compressed_high_store;
    __m512i gather_reg;
    unsigned int mask[8];
    LIBXSMM_UNUSED(dst_ifhp);

    for (c_i = 0; c_i<16; c_i++) {
      if (gather_offsets[16 - c_i - 1] >= w_remainder * 64) {
        int_mask = int_mask & ~(1 << c_i);
      }
    }
    for (c_i = 0; c_i<mask_remainder; c_i++) {
      mask[c_i] = (1U << 31);
    }
    for (c_i = mask_remainder; c_i<8; c_i++) {
      mask[c_i] = 0;
    }
    mask_reg = _mm256_loadu_si256((const __m256i*)mask);

    if (w_remainder) {
      for (img = my_img_start; img < my_img_end; img++) {
        for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ifm1 += 2) {
          for (ij = 0; ij < handle->ifhp; ++ij) {
            /* Handle full chunks  */
            for (w = 0; w < w_chunks; w++) {
              for (ifm2 = 0; ifm2 < 8; ++ifm2) {
                TRANSPOSE_W_CHUNK(img, ifm1, ij, w * 16, ifm2);
                TRANSPOSE_W_CHUNK(img, ifm1 + 1, ij, w * 16, ifm2);
              }
            }
            /* Handle remainder */
            for (ifm2 = 0; ifm2 < 8; ++ifm2) {
              TRANSPOSE_W_REMAINDER(img, ifm1, ij, w_chunks * 16, ifm2);
              TRANSPOSE_W_REMAINDER(img, ifm1 + 1, ij, w_chunks * 16, ifm2);
            }
          }
        }
      }
    }
    else {
      for (img = my_img_start; img < my_img_end; img++) {
        for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ifm1 += 2) {
          for (ij = 0; ij < handle->ifhp; ++ij) {
            /* Handle full chunks  */
            for (w = 0; w < w_chunks; w++) {
              for (ifm2 = 0; ifm2 < 8; ++ifm2) {
                TRANSPOSE_W_CHUNK(img, ifm1, ij, w * 16, ifm2);
                TRANSPOSE_W_CHUNK(img, ifm1 + 1, ij, w * 16, ifm2);
              }
            }
          }
        }
      }
    }
  }
  else {
    if (handle->avoid_input_trans == 0) {
      const __m512i perm_index = LIBXSMM_INTRINSICS_MM512_SET_EPI16(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8, 23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
      LIBXSMM_VLA_DECL(6, element_input_type, input_nopad, (element_input_type*)handle->reg_input->data, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
      LIBXSMM_VLA_DECL(6, element_input_type, tr_input_nopad, (element_input_type*)handle->scratch3, handle->blocksifm_lp, handle->ifhp, handle->ifwp / 2, handle->ifmblock_hp, 2);
      int ifm1, ij, ii;

      for (img = my_img_start; img < my_img_end; img++) {
        for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ifm1++) {
          for (ij = 0; ij < handle->ifhp; ij++) {
            for (ii = 0; ii < handle->ifwp; ii += 2) {
              element_input_type *addr = &LIBXSMM_VLA_ACCESS(6, input_nopad, img, ifm1, ij, ii, 0, 0, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
              element_input_type *dst_addr = &LIBXSMM_VLA_ACCESS(6, tr_input_nopad, img, ifm1, ij, ii / 2, 0, 0, handle->blocksifm_lp, handle->ifhp, handle->ifwp / 2, handle->ifmblock_hp, 2);
              __m512i cl = _mm512_loadu_si512(addr);
              __m512i permuted_reg = _mm512_permutexvar_epi16(perm_index, cl);
              _mm512_store_si512(dst_addr, permuted_reg);
            }
          }
        }
      }
    }
  }

  if (handle->avoid_output_trans == 0) {
    const __m512i perm_index = LIBXSMM_INTRINSICS_MM512_SET_EPI16(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8, 23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
    int ij, ii, ofm1;
    int OFWP = handle->ofwp + handle->output_lp_padding;
    element_output_type *out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock_lp * handle->fm_lp_block;
    LIBXSMM_VLA_DECL(6, element_output_type, tr_output, (element_output_type*)handle->scratch2, handle->blocksofm, handle->ofhp, OFWP / 2, handle->ofmblock, 2);
    LIBXSMM_VLA_DECL(6, element_output_type, output, out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block);

    if (handle->ofwp % 2 == 0) {
      for (img = my_img_start; img < my_img_end; img++) {
        for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ofm1++) {
          for (ij = 0; ij < handle->ofhp; ij++) {
            for (ii = 0; ii < handle->ofwp; ii += 2) {
              element_output_type *addr = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, ii, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block);
              element_output_type *dst_addr = &LIBXSMM_VLA_ACCESS(6, tr_output, img, ofm1, ij, ii / 2, 0, 0, handle->blocksofm, handle->ofhp, OFWP / 2, handle->ofmblock, 2);
              __m512i cl = _mm512_loadu_si512(addr);
              __m512i permuted_reg = _mm512_permutexvar_epi16(perm_index, cl);
              _mm512_store_si512(dst_addr, permuted_reg);
            }
          }
        }
      }
    }
    else {
      for (img = my_img_start; img < my_img_end; img++) {
        for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ofm1++) {
          for (ij = 0; ij < handle->ofhp; ij++) {
            for (ii = 0; ii < handle->ofwp - 1; ii += 2) {
              element_output_type *addr = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, ii, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block);
              element_output_type *dst_addr = &LIBXSMM_VLA_ACCESS(6, tr_output, img, ofm1, ij, ii / 2, 0, 0, handle->blocksofm, handle->ofhp, OFWP / 2, handle->ofmblock, 2);
              __m512i cl = _mm512_loadu_si512(addr);
              __m512i permuted_reg = _mm512_permutexvar_epi16(perm_index, cl);
              _mm512_store_si512(dst_addr, permuted_reg);
            }
            {
              element_output_type *addr = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, handle->ofwp - 1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block);
              element_output_type *dst_addr = &LIBXSMM_VLA_ACCESS(6, tr_output, img, ofm1, ij, ii / 2, 0, 0, handle->blocksofm, handle->ofhp, OFWP / 2, handle->ofmblock, 2);
              __m256i half_cl = _mm256_loadu_si256((const __m256i*)addr);
              __m512i cl = _mm512_inserti64x4(LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), half_cl, 0), permuted_reg;
              /*__m256i zero_pixel = _mm256_xor_si256(zero_pixel, zero_pixel);*/
              cl = _mm512_inserti64x4(cl, _mm256_set1_epi32(0)/*zero_pixel*/, 1);
              permuted_reg = _mm512_permutexvar_epi16(perm_index, cl);
              _mm512_store_si512(dst_addr, permuted_reg);
            }
          }
        }
      }
    }
  }
}
#endif /*defined(LIBXSMM_INTRINSICS_AVX512_CORE)*/

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
void lp_transpose_input_and_output(int ltid, libxsmm_dnn_layer* handle)
{
  typedef short element_input_type;
  typedef short element_output_type;

  const int imgpt = (handle->desc.N + handle->desc.threads - 1)/handle->desc.threads;
  const int my_img_start = LIBXSMM_MIN( ltid * imgpt, handle->desc.N);
  const int my_img_end = LIBXSMM_MIN( (ltid+1) * imgpt, handle->desc.N);

#if defined(LIBXSMM_INTRINSICS_AVX512_CORE)
  if (handle->use_vperm_transposes == 1) {
    lp_transpose_input_and_output_vperm(my_img_start, my_img_end, handle);
  }
  else
#endif
  {
    int w_chunks = handle->ifwp/16;
    int w_remainder = handle->ifwp%16;
    int w, c_i, ifm1, ij, ifm2;
    int BLOCKSIFM = handle->blocksifm;
    int padded_w = (handle->padding_flag == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
    int ifwp_extended = padded_w + handle->qfma_input_pad;
    int dst_ifhp = handle->ifhp;
    element_input_type *base_addr;
    const __m512i vgindex = _mm512_set_epi32(480,416,224,160,  352,288,96,32,  448,384,192,128,  320,256,64,0);
    const int gather_offsets[16] = {480,416,224,160,  352,288,96,32,  448,384,192,128,  320,256,64,0};
    const __m256i shuffler = _mm256_set_epi32(7,5,3,1,6,4,2,0);
    unsigned int int_mask = 0xffffffff;
    const __mmask16 gmask = (__mmask16)int_mask;
    int mask_remainder = (w_remainder+1)/2;
    /* Input transpose  */
    LIBXSMM_VLA_DECL(6, element_input_type, input_nopad, (element_input_type*)handle->reg_input->data, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
    LIBXSMM_VLA_DECL(5, element_input_type, tr_input_nopad, (element_input_type*)handle->scratch3, BLOCKSIFM, dst_ifhp, handle->ifmblock_hp, ifwp_extended);
    __m256i mask_reg, lo_reg, hi_reg, compressed_low, compressed_high, compressed_low_store, compressed_high_store;
    __m512i gather_reg;
    unsigned int mask[8];
    int img;
    LIBXSMM_UNUSED(dst_ifhp);

    for (c_i=0;c_i<16;c_i++) {
      if (gather_offsets[16-c_i-1] >= w_remainder*64) {
        int_mask = int_mask & ~(1 << c_i);
      }
    }
    for (c_i=0; c_i<mask_remainder; c_i++) {
      mask[c_i] = (1U << 31);
    }
    for (c_i=mask_remainder; c_i<8; c_i++) {
      mask[c_i] = 0;
    }
    mask_reg = _mm256_loadu_si256((const __m256i*)mask);

    if (w_remainder) {
      for (img = my_img_start; img < my_img_end; img++) {
        for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ifm1+=2) {
          for (ij = 0; ij < handle->ifhp; ++ij) {
            /* Handle full chunks  */
            for (w = 0; w < w_chunks; w++) {
              for (ifm2 = 0; ifm2 < 8; ++ifm2) {
                TRANSPOSE_W_CHUNK(img, ifm1, ij, w*16, ifm2);
                TRANSPOSE_W_CHUNK(img, ifm1+1, ij, w*16, ifm2);
              }
            }
            /* Handle remainder */
            for (ifm2 = 0; ifm2 < 8; ++ifm2) {
              TRANSPOSE_W_REMAINDER(img, ifm1, ij, w_chunks*16, ifm2);
              TRANSPOSE_W_REMAINDER(img, ifm1+1, ij, w_chunks*16, ifm2);
            }
          }
        }
      }
    }
    else {
      for (img = my_img_start; img < my_img_end; img++) {
        for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ifm1+=2) {
          for (ij = 0; ij < handle->ifhp; ++ij) {
            /* Handle full chunks  */
            for (w = 0; w < w_chunks; w++) {
              for (ifm2 = 0; ifm2 < 8; ++ifm2) {
                TRANSPOSE_W_CHUNK(img, ifm1, ij, w*16, ifm2);
                TRANSPOSE_W_CHUNK(img, ifm1+1, ij, w*16, ifm2);
              }
            }
          }
        }
      }
    }
    {
      element_output_type *pair_addr, *pair_addr_dst;
      int half_i, ofm1, ii;
      int BLOCKSOFM = handle->blocksofm;
      int OFWP = handle->ofwp+handle->output_lp_padding;
      __m256i compressed_hi, compressed_lo, even_pixel, odd_pixel;
      __m512i compact, pair_pixels;
      const __m512i permute_compact_idx = _mm512_set_epi32(15,14,13,12,  7,6,5,4,  11,10,9,8,  3,2,1,0);

      /* Output transpose */
      element_output_type *out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock_lp * handle->fm_lp_block;
      LIBXSMM_VLA_DECL(6, element_output_type, tr_output,  (element_output_type*)handle->scratch2 , BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2);
      LIBXSMM_VLA_DECL(6, element_output_type, output, out, handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block);

      for (img = my_img_start; img < my_img_end; img++) {
        for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ofm1++) {
          for (ij = 0; ij < handle->ofhp; ++ij) {
            for (ii = 0, half_i = 0; ii < handle->ofwp - 1; ii += 2, half_i++) {
              TRANSPOSE_W_FULL_PAIR(img, ofm1, ij, ii, half_i);
            }
          }
        }

        if (handle->output_lp_padding != 0) {
          /* Zero out the "output padding pixel" */
          for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ofm1++) {
            for (ij = 0; ij < handle->ofhp; ++ij) {
              ii = handle->ofwp-1;
              half_i = ii/2;
              TRANSPOSE_W_HALF_PAIR(img, ofm1, ij, ii, half_i);
            }
          }
        }
      }
    }
  }
}

#if defined(LIBXSMM_INTRINSICS_AVX512_CORE)
LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
void lp_transpose_and_resize_input_and_output_vperm(int my_img_start, int my_img_end, libxsmm_dnn_layer* handle)
{
  /*typedef short element_input_type;*/
  typedef short element_output_type;
  int img, ij;

  if (handle->avoid_output_trans == 0) {
    const __m512i perm_index = LIBXSMM_INTRINSICS_MM512_SET_EPI16(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8, 23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
    int ii, ofm1;
    int OFWP = handle->ofwp + handle->output_lp_padding;
    element_output_type *out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock_lp * handle->fm_lp_block;
    LIBXSMM_VLA_DECL(6, element_output_type, output, out, handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block);
    LIBXSMM_VLA_DECL(6, element_output_type, tr_output, (element_output_type*)handle->scratch2, handle->blocksofm, handle->ofhp, OFWP / 2, handle->ofmblock, 2);

    if (handle->ofwp % 2 == 0) {
      for (img = my_img_start; img < my_img_end; img++) {
        for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ofm1++) {
          for (ij = 0; ij < handle->ofhp; ij++) {
            for (ii = 0; ii < handle->ofwp; ii += 2) {
              element_output_type *addr = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, ii, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block);
              element_output_type *dst_addr = &LIBXSMM_VLA_ACCESS(6, tr_output, img, ofm1, ij, ii / 2, 0, 0, handle->blocksofm, handle->ofhp, OFWP / 2, handle->ofmblock, 2);
              __m512i cl = _mm512_loadu_si512(addr);
              __m512i permuted_reg = _mm512_permutexvar_epi16(perm_index, cl);
              _mm512_store_si512(dst_addr, permuted_reg);
            }
          }
        }
      }
    }
    else {
      for (img = my_img_start; img < my_img_end; img++) {
        for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ofm1++) {
          for (ij = 0; ij < handle->ofhp; ij++) {
            for (ii = 0; ii < handle->ofwp - 1; ii += 2) {
              element_output_type *addr = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, ii, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block);
              element_output_type *dst_addr = &LIBXSMM_VLA_ACCESS(6, tr_output, img, ofm1, ij, ii / 2, 0, 0, handle->blocksofm, handle->ofhp, OFWP / 2, handle->ofmblock, 2);
              __m512i cl = _mm512_loadu_si512(addr);
              __m512i permuted_reg = _mm512_permutexvar_epi16(perm_index, cl);
              _mm512_store_si512(dst_addr, permuted_reg);
            }
            {
              element_output_type *addr = &LIBXSMM_VLA_ACCESS(6, output, img, ofm1, ij, handle->ofwp - 1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block);
              element_output_type *dst_addr = &LIBXSMM_VLA_ACCESS(6, tr_output, img, ofm1, ij, ii / 2, 0, 0, handle->blocksofm, handle->ofhp, OFWP / 2, handle->ofmblock, 2);
              __m256i half_cl = _mm256_loadu_si256((const __m256i*)addr);
              __m512i cl = _mm512_inserti64x4(LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), half_cl, 0), permuted_reg;
              /*__m256i zero_pixel = _mm256_xor_si256(zero_pixel, zero_pixel);*/
              cl = _mm512_inserti64x4(cl, _mm256_set1_epi32(0)/*zero_pixel*/, 1);
              permuted_reg = _mm512_permutexvar_epi16(perm_index, cl);
              _mm512_store_si512(dst_addr, permuted_reg);
            }
          }
        }
      }
    }
  }
}
#endif /*defined(LIBXSMM_INTRINSICS_AVX512_CORE)*/

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
void lp_transpose_and_resize_input_and_output(int ltid, libxsmm_dnn_layer* handle)
{
  typedef short element_input_type;
  typedef short element_output_type;

  int img = ltid;
  int w_chunks = handle->ifwp_resized/16;
  int w_remainder = handle->ifwp_resized%16;
  int u = handle->desc.u;
  int w, c_i, ifm1, ij, ifm2;
  int dst_j, src_j;
  int BLOCKSIFM = handle->blocksifm;
  const int imgpt = (handle->desc.N + handle->desc.threads - 1)/handle->desc.threads;
  const int my_img_start = LIBXSMM_MIN( ltid * imgpt, handle->desc.N);
  const int my_img_end = LIBXSMM_MIN( (ltid+1) * imgpt, handle->desc.N);
#if 0
  int padded_w = (handle->padding_flag == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
  int ifwp_extended = padded_w + handle->qfma_input_pad;
#else /* Input transpose */
  int ifwp_extended = handle->ifwp_resized + handle->qfma_input_pad;
#endif
  int dst_ifhp = handle->ifhp_resized;
  LIBXSMM_VLA_DECL(6, element_input_type, input_nopad, (element_input_type*)handle->reg_input->data, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
  LIBXSMM_VLA_DECL(5, element_input_type, tr_input_nopad, (element_input_type*)handle->scratch3, BLOCKSIFM, dst_ifhp, handle->ifmblock_hp, ifwp_extended);
  element_input_type *base_addr;
  const __m512i vgindex = _mm512_set_epi32(u*480,u*416,u*224,u*160,  u*352,u*288,u*96,u*32,  u*448,u*384,u*192,u*128,  u*320,u*256,u*64, u*0);
  const int gather_offsets[16] = {480,416,224,160,  352,288,96,32,  448,384,192,128,  320,256,64, 0};
  const __m256i shuffler = _mm256_set_epi32(7,5,3,1,6,4,2,0);
  unsigned int int_mask = 0xffffffff;
  const __mmask16 gmask = (__mmask16)int_mask;
  int mask_remainder = (w_remainder+1)/2;
  __m256i mask_reg, lo_reg, hi_reg, compressed_low, compressed_high, compressed_low_store, compressed_high_store;
  __m512i gather_reg;
  unsigned int mask[8];
  LIBXSMM_UNUSED(dst_ifhp);

  for (c_i=0;c_i<16;c_i++) {
    if (gather_offsets[16-c_i-1] >= (w_remainder*64)) {
      int_mask = int_mask & ~(1 << c_i);
    }
  }
  for (c_i=0; c_i<mask_remainder; c_i++) {
    mask[c_i] = (1U << 31);
  }
  for (c_i=mask_remainder; c_i<8; c_i++) {
    mask[c_i] = 0;
  }
  mask_reg = _mm256_loadu_si256((const __m256i*)mask);

  if (w_remainder) {
    for (img = my_img_start; img < my_img_end; img++) {
      for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ifm1+=2) {
        for (dst_j=0; dst_j < handle->ifhp_resized; dst_j++) {
          src_j = dst_j * handle->desc.v;
          /* Handle full chunks  */
          for (w = 0; w < w_chunks; w++) {
            for (ifm2 = 0; ifm2 < 8; ++ifm2) {
              TRANSPOSE_W_CHUNK_RESIZED(img, ifm1, w*u*16, src_j, ifm2, w*16, dst_j);
              TRANSPOSE_W_CHUNK_RESIZED(img, ifm1+1, w*u*16, src_j, ifm2, w*16, dst_j);
            }
          }
          /* Handle remainder */
          for (ifm2 = 0; ifm2 < 8; ++ifm2) {
            TRANSPOSE_W_REMAINDER_RESIZED(img, ifm1, w_chunks*u*16, src_j, ifm2, w_chunks*16, dst_j);
            TRANSPOSE_W_REMAINDER_RESIZED(img, ifm1+1, w_chunks*u*16, src_j, ifm2, w_chunks*16, dst_j);
          }
        }
      }
    }
  }
  else {
    for (img = my_img_start; img < my_img_end; img++) {
      for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ifm1+=2) {
        for (dst_j=0; dst_j < handle->ifhp_resized; dst_j++) {
          src_j = dst_j * handle->desc.v;
          /* Handle full chunks  */
          for (w = 0; w < w_chunks; w++) {
            for (ifm2 = 0; ifm2 < 8; ++ifm2) {
              TRANSPOSE_W_CHUNK_RESIZED(img, ifm1, w*u*16, src_j, ifm2, w*16, dst_j);
              TRANSPOSE_W_CHUNK_RESIZED(img, ifm1+1, w*u*16, src_j, ifm2, w*16, dst_j);
            }
          }
        }
      }
    }
  }

#if defined(LIBXSMM_INTRINSICS_AVX512_CORE)
  if (handle->use_vperm_transposes == 1) {
    lp_transpose_and_resize_input_and_output_vperm(my_img_start, my_img_end, handle);
  }
  else
#endif
  {
    element_output_type *pair_addr, *pair_addr_dst;
    int half_i, ofm1, ii;
    int BLOCKSOFM = handle->blocksofm;
    int OFWP = handle->ofwp+handle->output_lp_padding;
    __m256i compressed_hi, compressed_lo, even_pixel, odd_pixel;
    __m512i compact, pair_pixels;
    const __m512i permute_compact_idx = _mm512_set_epi32(15,14,13,12,  7,6,5,4,  11,10,9,8,  3,2,1,0);

    /* Output transpose */
    element_output_type *out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock_lp * handle->fm_lp_block;
    LIBXSMM_VLA_DECL(6, element_output_type, tr_output,  (element_output_type*)handle->scratch2 , BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2);
    LIBXSMM_VLA_DECL(6, element_output_type, output, out, handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block);

    for (img = my_img_start; img < my_img_end; img++) {
      for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ofm1++) {
        for (ij = 0; ij < handle->ofhp; ++ij) {
          for (ii = 0, half_i = 0; ii < handle->ofwp - 1; ii += 2, half_i++) {
            TRANSPOSE_W_FULL_PAIR(img, ofm1, ij, ii, half_i);
          }
        }
      }

      if (handle->output_lp_padding != 0) {
        /* Zero out the "output padding pixel" */
        for (ofm1 = 0; ofm1 < handle->blocksofm_lp; ofm1++) {
          for (ij = 0; ij < handle->ofhp; ++ij) {
            ii = handle->ofwp-1;
            half_i = ii/2;
            TRANSPOSE_W_HALF_PAIR(img, ofm1, ij, ii, half_i);
          }
        }
      }
    }
  }
}
#else
LIBXSMM_API_INTERN void lp_transpose_and_resize_input_and_output(int ltid, libxsmm_dnn_layer* handle) {
  LIBXSMM_UNUSED(ltid); LIBXSMM_UNUSED(handle);
}
LIBXSMM_API_INTERN void lp_transpose_input_and_output(int ltid, libxsmm_dnn_layer* handle) {
  LIBXSMM_UNUSED(ltid); LIBXSMM_UNUSED(handle);
}
#endif /*defined(LIBXSMM_INTRINSICS_AVX512)*/

#undef TRANSPOSE_W_CHUNK
#undef TRANSPOSE_W_REMAINDER
#undef TRANSPOSE_W_FULL_PAIR
#undef TRANSPOSE_W_HALF_PAIR
#undef TRANSPOSE_W_CHUNK_RESIZED
#undef TRANSPOSE_W_REMAINDER_RESIZED

#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
void gather_transpose_ps_16_56_56_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS)
{
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x00FF;
  __m512 tmp;
  int m, n;
  LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ldD); LIBXSMM_UNUSED(ldS);
  LIBXSMM_PRAGMA_UNROLL_AND_JAM(4)
  for(m = 0; m < 16; ++m) {
    LIBXSMM_PRAGMA_UNROLL_N(3)
    for(n = 0; n < 3; ++n) {
      tmp = _mm512_i32gather_ps(vindex, (const float*)(src+m+n*256), 4);
      _mm512_store_ps((void*)(dst+m*56+n*16),tmp);
    }
    tmp = _mm512_mask_i32gather_ps(LIBXSMM_INTRINSICS_MM512_UNDEFINED(), Nremmask, vindex, (const float*)(src+m+n*256), 4);
    _mm512_mask_store_ps((void*)(dst+m*56+n*16),Nremmask,tmp);
  }
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
void gather_transpose_ps_16_56_58_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS)
{
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x00FF;
  __m512 tmp;
  int m, n;
  LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ldD); LIBXSMM_UNUSED(ldS);
  LIBXSMM_PRAGMA_UNROLL_AND_JAM(4)
  for(m = 0; m < 16; ++m) {
    LIBXSMM_PRAGMA_UNROLL_N(3)
    for(n = 0; n < 3; ++n) {
      tmp = _mm512_i32gather_ps(vindex, (const float*)(src+m+n*256), 4);
      _mm512_store_ps((void*)(dst+m*58+n*16),tmp);
    }
    tmp = _mm512_mask_i32gather_ps(LIBXSMM_INTRINSICS_MM512_UNDEFINED(), Nremmask, vindex, (const float*)(src+m+n*256), 4);
    _mm512_mask_store_ps((void*)(dst+m*58+n*16),Nremmask,tmp);
  }
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
void gather_transpose_ps_16_58_60_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS)
{
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x03FF;
  __m512 tmp;
  int m, n;
  LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ldD); LIBXSMM_UNUSED(ldS);
  LIBXSMM_PRAGMA_UNROLL_AND_JAM(4)
  for(m = 0; m < 16; ++m) {
    LIBXSMM_PRAGMA_UNROLL_N(3)
    for(n = 0; n < 3; ++n) {
      tmp = _mm512_i32gather_ps(vindex, (const float*)(src+m+n*256), 4);
      _mm512_store_ps((void*)(dst+m*60+n*16),tmp);
    }
    tmp = _mm512_mask_i32gather_ps(LIBXSMM_INTRINSICS_MM512_UNDEFINED(), Nremmask, vindex, (const float*)(src+m+n*256), 4);
    _mm512_mask_store_ps((void*)(dst+m*60+n*16),Nremmask,tmp);
  }
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
void gather_transpose_ps_16_58_58_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS)
{
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x03FF;
  __m512 tmp;
  int m, n;
  LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ldD); LIBXSMM_UNUSED(ldS);
  LIBXSMM_PRAGMA_UNROLL_AND_JAM(4)
  for(m = 0; m < 16; ++m) {
    LIBXSMM_PRAGMA_UNROLL_N(3)
    for(n = 0; n < 3; ++n) {
      tmp = _mm512_i32gather_ps(vindex, (const float*)(src+m+n*256), 4);
      _mm512_store_ps((void*)(dst+m*58+n*16),tmp);
    }
    tmp = _mm512_mask_i32gather_ps(LIBXSMM_INTRINSICS_MM512_UNDEFINED(), Nremmask, vindex, (const float*)(src+m+n*256), 4);
    _mm512_mask_store_ps((void*)(dst+m*58+n*16),Nremmask,tmp);
  }
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
void gather_transpose_ps_16_28_28_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS)
{
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x0FFF;
  int m;
  LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ldD); LIBXSMM_UNUSED(ldS);
  LIBXSMM_PRAGMA_UNROLL_AND_JAM(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    __m512 tmp = _mm512_i32gather_ps(vindex, (const float*)(src+m+n*256), 4);
    _mm512_store_ps((void*)(dst+m*28+n*16),tmp);
    n = 1;
    tmp = _mm512_mask_i32gather_ps(LIBXSMM_INTRINSICS_MM512_UNDEFINED(), Nremmask, vindex, (const float*)(src+m+n*256), 4);
    _mm512_mask_store_ps((void*)(dst+m*28+n*16),Nremmask,tmp);
  }
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
void gather_transpose_ps_16_28_30_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS)
{
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x0FFF;
  int m;
  LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ldD); LIBXSMM_UNUSED(ldS);
  LIBXSMM_PRAGMA_UNROLL_AND_JAM(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    __m512 tmp = _mm512_i32gather_ps(vindex, (const float*)(src+m+n*256), 4);
    _mm512_store_ps((void*)(dst+m*30+n*16),tmp);
    n = 1;
    tmp = _mm512_mask_i32gather_ps(LIBXSMM_INTRINSICS_MM512_UNDEFINED(), Nremmask, vindex, (const float*)(src+m+n*256), 4);
    _mm512_mask_store_ps((void*)(dst+m*30+n*16),Nremmask,tmp);
  }
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
void gather_transpose_ps_16_30_32_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS)
{
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x3FFF;
  int m;
  LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ldD); LIBXSMM_UNUSED(ldS);
  LIBXSMM_PRAGMA_UNROLL_AND_JAM(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    __m512 tmp = _mm512_i32gather_ps(vindex, (const float*)(src+m+n*256), 4);
    _mm512_store_ps((void*)(dst+m*32+n*16),tmp);
    n = 1;
    tmp = _mm512_mask_i32gather_ps(LIBXSMM_INTRINSICS_MM512_UNDEFINED(), Nremmask, vindex, (const float*)(src+m+n*256), 4);
    _mm512_mask_store_ps((void*)(dst+m*32+n*16),Nremmask,tmp);
  }
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
void gather_transpose_ps_16_30_30_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS)
{
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x3FFF;
  int m;
  LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ldD); LIBXSMM_UNUSED(ldS);
  LIBXSMM_PRAGMA_UNROLL_AND_JAM(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    __m512 tmp = _mm512_i32gather_ps(vindex, (const float*)(src+m+n*256), 4);
    _mm512_store_ps((void*)(dst+m*30+n*16),tmp);
    n = 1;
    tmp = _mm512_mask_i32gather_ps(LIBXSMM_INTRINSICS_MM512_UNDEFINED(), Nremmask, vindex, (const float*)(src+m+n*256), 4);
    _mm512_mask_store_ps((void*)(dst+m*30+n*16),Nremmask,tmp);
  }
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
void gather_transpose_ps_16_16_16_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS)
{
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  int m;
  LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ldD); LIBXSMM_UNUSED(ldS);
  LIBXSMM_PRAGMA_UNROLL_AND_JAM(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmp = _mm512_i32gather_ps(vindex, (const float*)(src+m+n*256), 4);
    _mm512_store_ps((void*)(dst+m*16+n*16),tmp);
  }
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
void gather_transpose_ps_16_16_18_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS)
{
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  int m;
  LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ldD); LIBXSMM_UNUSED(ldS);
  LIBXSMM_PRAGMA_UNROLL_AND_JAM(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmp = _mm512_i32gather_ps(vindex, (const float*)(src+m+n*256), 4);
    _mm512_store_ps((void*)(dst+m*18+n*16),tmp);
  }
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
void gather_transpose_ps_16_14_16_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS)
{
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x3FFF;
  int m;
  LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ldD); LIBXSMM_UNUSED(ldS);
  LIBXSMM_PRAGMA_UNROLL_AND_JAM(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmp = _mm512_mask_i32gather_ps(LIBXSMM_INTRINSICS_MM512_UNDEFINED(), Nremmask, vindex, (const float*)(src+m+n*256), 4);
    _mm512_mask_store_ps((void*)(dst+m*16+n*16),Nremmask,tmp);
  }
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
void gather_transpose_ps_16_14_18_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS)
{
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x3FFF;
  int m;
  LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ldD); LIBXSMM_UNUSED(ldS);
  LIBXSMM_PRAGMA_UNROLL_AND_JAM(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmp = _mm512_mask_i32gather_ps(LIBXSMM_INTRINSICS_MM512_UNDEFINED(), Nremmask, vindex, (const float*)(src+m+n*256), 4);
    _mm512_mask_store_ps((void*)(dst+m*18+n*16),Nremmask,tmp);
  }
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
void gather_transpose_ps_16_7_8_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS)
{
  const __m512i vindex = _mm512_set_epi32(0, 97, 81, 65, 49, 33, 17,  1,
      0, 96, 80, 64, 48, 32, 16,  0);
  const __mmask16 Nremmask = 0x7F7F;
  int m;
  LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ldD); LIBXSMM_UNUSED(ldS);
  LIBXSMM_PRAGMA_UNROLL_AND_JAM(4)
  for(m = 0; m < 8; ++m) {
    const __m512 tmp = _mm512_mask_i32gather_ps(LIBXSMM_INTRINSICS_MM512_UNDEFINED(), Nremmask, vindex, (const float*)(src+m*2), 4);
    _mm512_mask_store_ps((void*)(dst+m*8*2),Nremmask,tmp);
  }
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
void gather_transpose_ps_16_7_10_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS)
{
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x07F;
  int m;
  LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ldD); LIBXSMM_UNUSED(ldS);
  LIBXSMM_PRAGMA_UNROLL_AND_JAM(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmp = _mm512_mask_i32gather_ps(LIBXSMM_INTRINSICS_MM512_UNDEFINED(), Nremmask, vindex, (const float*)(src+m+n*256), 4);
    _mm512_mask_store_ps((void*)(dst+m*10+n*16),Nremmask,tmp);
  }
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
void gather_transpose_ps_16_9_12_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS)
{
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x01FF;
  int m;
  LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ldD); LIBXSMM_UNUSED(ldS);
  LIBXSMM_PRAGMA_UNROLL_AND_JAM(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmp = _mm512_mask_i32gather_ps(LIBXSMM_INTRINSICS_MM512_UNDEFINED(), Nremmask, vindex, (const float*)(src+m+n*256), 4);
    _mm512_mask_store_ps((void*)(dst+m*12+n*16),Nremmask,tmp);
  }
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
void gather_transpose_ps_16_9_10_16(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS)
{
  const __m512i vindex = _mm512_set_epi32(240,224,208,192,176,160,144,128,112,96,80,64,48,32,16,0);
  const __mmask16 Nremmask = 0x01FF;
  int m;
  LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ldD); LIBXSMM_UNUSED(ldS);
  LIBXSMM_PRAGMA_UNROLL_AND_JAM(4)
  for(m = 0; m < 16; ++m) {
    int n = 0;
    const __m512 tmp = _mm512_mask_i32gather_ps(LIBXSMM_INTRINSICS_MM512_UNDEFINED(), Nremmask, vindex, (const float*)(src+m+n*256), 4);
    _mm512_mask_store_ps((void*)(dst+m*10+n*16),Nremmask,tmp);
  }
}

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
void transpose_fallback(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS)
{
  const __m512i vindex_base = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
  const __m512i vindex = _mm512_mullo_epi32(_mm512_set1_epi32(ldS), vindex_base);
  const int whole16s = N/16, remainder = N-whole16s*16;
  const __mmask16 Nmask = (__mmask16)((1 << remainder) - 1);
  int i;
  LIBXSMM_PRAGMA_UNROLL_AND_JAM(2)
  for(i = 0; i < M; ++i) {
    int j;
    LIBXSMM_PRAGMA_UNROLL_N(4)
    for(j = 0; j < whole16s; ++j) {
      const __m512 res = _mm512_i32gather_ps(vindex, (const float*)(src+i+j*16*ldS), 4);
      _mm512_store_ps(dst + ldD*i+j*16, res);
    }
    if(remainder) {
      const __m512 res = _mm512_mask_i32gather_ps(LIBXSMM_INTRINSICS_MM512_UNDEFINED(), Nmask, vindex, (const float*)(src+i+j*16*ldS), 4);
      _mm512_mask_store_ps(dst + ldD*i+j*16, Nmask, res);
    }
  }
}
#else
LIBXSMM_API_INTERN void transpose_fallback(int M, int N, float *LIBXSMM_RESTRICT dst, int ldD, const float *LIBXSMM_RESTRICT src, int ldS)
{
  int n, m;
  for (n = 0; n < N; ++n) {
    for (m = 0; m < M; ++m) {
      dst[m*ldD + n] = src[n*ldS + m];
    }
  }
}
#endif

LIBXSMM_API_INTERN transposer get_transposer(int M, int N, int ldD, int ldS)
{
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
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
#else
  LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ldD); LIBXSMM_UNUSED(ldS);
#endif
  return transpose_fallback;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_convolve_st_upd_custom_custom_f32_f32(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  typedef libxsmm_sconvfunction libxsmm_convfunction;
# include "template/libxsmm_dnn_convolve_st_upd_custom_custom.tpl.c"
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_convolve_st_upd_custom_custom_i16_i32(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid); /* TODO */
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#else /* should not happen */
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_convolve_st_upd_custom_custom_i16_f32(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if (handle->upd_use_thread_fil > 0) {
    typedef short element_input_type;
    typedef short element_output_type;
    typedef float element_filter_type;
    typedef libxsmm_uwsconvfunction libxsmm_convfunction;
    if (handle->use_fastpath) {
      if ( handle->use_hybrid_wu_parallelism == 1) {
# include "template/libxsmm_dnn_convolve_st_upd_custom_custom_stream_lp.tpl.c"
      }
      else {
# include "template/libxsmm_dnn_convolve_st_upd_custom_custom_stream_opt_lp.tpl.c"
      }
    }
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_convolve_st_upd_custom_custom_bf16_bf16(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if (handle->upd_use_thread_fil > 0) {
    typedef libxsmm_bfloat16 element_input_type;
    typedef libxsmm_bfloat16 element_output_type;
    typedef float element_filter_type;
    typedef libxsmm_bf16f32convfunction libxsmm_convfunction;
    if (handle->use_fastpath) {
      if ( handle->use_hybrid_wu_parallelism == 1) {
# include "template/libxsmm_dnn_convolve_st_upd_custom_custom_stream_bf16.tpl.c"
      }
      else {
# include "template/libxsmm_dnn_convolve_st_upd_custom_custom_stream_opt_bf16.tpl.c"
      }
    }
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_convolve_st_upd_custom_custom_i8_i32(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if (handle->upd_use_thread_fil > 0) {
    typedef unsigned char element_input_type;
    typedef unsigned char element_output_type;
    typedef int element_filter_type;
    typedef libxsmm_bdbconvfunction libxsmm_convfunction;
    if (handle->use_fastpath) {
      if ( handle->use_hybrid_wu_parallelism == 1) {
# include "template/libxsmm_dnn_convolve_st_upd_custom_custom_stream_lp.tpl.c"
      }
      else {
# include "template/libxsmm_dnn_convolve_st_upd_custom_custom_stream_opt_lp.tpl.c"
      }
    }
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_upd_custom_custom(libxsmm_dnn_layer* handle, int start_thread, int tid)
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
  if ( handle->use_upd_generic != 0 ) {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      const libxsmm_blasint ldx     = (libxsmm_blasint)(handle->desc.W+(2*handle->desc.pad_w));
      const libxsmm_blasint ldx_alt = (libxsmm_blasint)(handle->desc.v*handle->ifmblock);
      const libxsmm_blasint ldb_alt = (libxsmm_blasint)handle->ofwp;
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxsmm_smmfunction gemm_function;
      /* let's do a ofmblock x ifmblock x ofw_rb GEMM :-) or in other words M=nbOfm, N=nbIfm, K=ofw (col-major) */
      gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ofmblock, handle->ifmblock, handle->ofw, NULL, &ldx, NULL, NULL, NULL, NULL, NULL);
      /* for strided convolutions with kernel size bigger than 1 the above GEMM doesn't work and we need to switch to more transposes and an
         alternative GEMM:
         let's do a ifmblock x ofmblock x ofw_rb GEMM :-) or in other words M=nbIfm, N=nbOfm, K=ofw (col-major) */
      gemm_function gemm_kernel_alt = libxsmm_smmdispatch(handle->ifmblock, handle->ofmblock, handle->ofw, &ldx_alt, &ldb_alt, NULL, NULL, NULL, NULL, NULL);
# include "template/libxsmm_dnn_convolve_st_upd_custom_custom_generic.tpl.c"
    }
    else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_convolve_st_upd_custom_custom_f32_f32( handle, start_thread, tid );
    }
    else if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_BF16 ) {
      status = libxsmm_dnn_convolve_st_upd_custom_custom_bf16_bf16( handle, start_thread, tid );
    }
    else if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_convolve_st_upd_custom_custom_i16_f32( handle, start_thread, tid );
    }
    else if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32 ) {
      status = libxsmm_dnn_convolve_st_upd_custom_custom_i8_i32( handle, start_thread, tid );
    }
    else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_upd_nhwc_custom(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->reg_input == 0 || handle->grad_output == 0 || handle->grad_filter == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if ( handle->use_upd_generic != 0 ) {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      const libxsmm_blasint lda     = (libxsmm_blasint)(handle->blocksofm*handle->ofmblock);
      const libxsmm_blasint ldb     = (libxsmm_blasint)(handle->desc.W+(2*handle->desc.pad_w));
      const libxsmm_blasint ldc     = (libxsmm_blasint)(handle->ofmblock);
      const libxsmm_blasint lda_alt = (libxsmm_blasint)((handle->desc.pad_h == handle->desc.pad_h_in && handle->desc.pad_w == handle->desc.pad_w_in)
                            ? (handle->desc.v*handle->blocksifm*handle->ifmblock) : (handle->desc.v*handle->ifmblock));
      const libxsmm_blasint ldb_alt = (libxsmm_blasint)(handle->ofwp);
      const libxsmm_blasint ldc_alt = (libxsmm_blasint)(handle->ifmblock);
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxsmm_smmfunction gemm_function;
      /* let's do a ofmblock x ifmblock x ofw_rb GEMM :-) or in other words M=nbOfm, N=nbIfm, K=ofw (col-major) */
      gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ofmblock, handle->ifmblock, handle->ofw, &lda, &ldb, &ldc, NULL, NULL, NULL, NULL);
      /* for strided convolutions with kernel size bigger than 1 the above GEMM doesn't work and we need to switch to more transposes and an
         alternative GEMM:
         let's do a ifmblock x ofmblock x ofw_rb GEMM :-) or in other words M=nbIfm, N=nbOfm, K=ofw (col-major) */
      gemm_function gemm_kernel_alt = libxsmm_smmdispatch(handle->ifmblock, handle->ofmblock, handle->ofw, &lda_alt, &ldb_alt, &ldc_alt, NULL, NULL, NULL, NULL);
#define LIBXSMM_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_CUSTOM
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_custom-rsck_generic.tpl.c"
#undef LIBXSMM_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_CUSTOM
    }
    else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    /* shouldn't happen */
  }

  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_upd_nhwc_rsck(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->reg_input == 0 || handle->grad_output == 0 || handle->grad_filter == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if ( handle->use_upd_generic != 0 ) {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      const libxsmm_blasint lda     = (libxsmm_blasint)(handle->blocksofm*handle->ofmblock);
      const libxsmm_blasint ldb     = (libxsmm_blasint)(handle->desc.W+(2*handle->desc.pad_w));
      const libxsmm_blasint ldc     = (libxsmm_blasint)(handle->blocksofm*handle->ofmblock);
      const libxsmm_blasint lda_alt = (libxsmm_blasint)((handle->desc.pad_h == handle->desc.pad_h_in && handle->desc.pad_w == handle->desc.pad_w_in)
                            ? (handle->desc.v*handle->blocksifm*handle->ifmblock) : (handle->desc.v*handle->ifmblock));
      const libxsmm_blasint ldb_alt = (libxsmm_blasint)(handle->ofwp);
      const libxsmm_blasint ldc_alt = (libxsmm_blasint)(handle->ifmblock);
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxsmm_smmfunction gemm_function;
      /* let's do a ofmblock x ifmblock x ofw_rb GEMM :-) or in other words M=nbOfm, N=nbIfm, K=ofw (col-major) */
      gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ofmblock, handle->ifmblock, handle->ofw, &lda, &ldb, &ldc, NULL, NULL, NULL, NULL);
      /* for strided convolutions with kernel size bigger than 1 the above GEMM doesn't work and we need to switch to more transposes and an
         alternative GEMM:
         let's do a ifmblock x ofmblock x ofw_rb GEMM :-) or in other words M=nbIfm, N=nbOfm, K=ofw (col-major) */
      gemm_function gemm_kernel_alt = libxsmm_smmdispatch(handle->ifmblock, handle->ofmblock, handle->ofw, &lda_alt, &ldb_alt, &ldc_alt, NULL, NULL, NULL, NULL);
#define LIBXSMM_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_RSCK
# include "template/libxsmm_dnn_convolve_st_upd_nhwc_custom-rsck_generic.tpl.c"
#undef LIBXSMM_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_RSCK
    }
    else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    /* shouldn't happen */
  }

  return status;
}

