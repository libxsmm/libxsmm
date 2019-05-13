/******************************************************************************
** Copyright (c) 2016-2019, Intel Corporation                                **
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
#define _mm512_roundbf16rne(A) LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(A)
#define _mm512_storecvtrne_fp32_bf16(A,B)  _mm256_stream_si256((__m256i*)(A),_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_roundbf16rne((B)),16)))
#define _mm512_loadcvt_bf16_fp32(A)   _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(A))),16))
#define _mm512_loadcvtrne_fp32_bf16(A) _mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_roundbf16rne(LIBXSMM_INTRINSICS_MM512_LOAD_PS(A)),16))

int img, my_img_start, my_img_end, ofmb, ifmb, ojb, ofm1, ifm1, ifm2, ofm2, oj, oi, ii, ij, kj, ki, ind, j_br, img_br, i, j, img_block_size = 1, my_ofm_start, my_ofm_end, my_ifm_start, my_ifm_end, block_ofm, block_ifm, pix, pixb;
/* computing first logical thread */
const int ltid = tid - start_thread;

element_output_type *const out = (element_output_type*)handle->grad_output->data + ((size_t)handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
LIBXSMM_VLA_DECL(5, const element_output_type, output, (const element_output_type*)out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
LIBXSMM_VLA_DECL(5, const element_input_type, input, (const element_input_type*)handle->reg_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);

LIBXSMM_VLA_DECL(7, element_filter_type, weight_global, (element_filter_type*)handle->grad_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock/2, handle->ofmblock, 2);
element_filter_type *weight_ptr = (element_filter_type*)handle->scratch7 + ltid * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S;
LIBXSMM_VLA_DECL(7, element_filter_type, weight_private, (element_filter_type*)weight_ptr, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock/2, handle->ofmblock, 2);
element_filter_type *filter_dst_ptr = (handle->weight_copies > 1) ? (element_filter_type*)weight_ptr : (element_filter_type*)handle->grad_filter->data;
LIBXSMM_VLA_DECL(7, element_filter_type, weight_dst, (element_filter_type*)filter_dst_ptr, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock/2, handle->ofmblock, 2);

/* This intermediate tensor is used when pixels are NOT fully accumulated  */
float *weight_ptr_f32 = (float*)handle->scratch2 + handle->desc.N * (handle->output_pixels/2) * handle->desc.K + ltid * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S;
LIBXSMM_VLA_DECL(6, float, weight_private_f32, (float*)weight_ptr_f32, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
/* Accumulation scratch is used when pixels are ully accumulated  */
element_filter_type *filter_scratch = (element_filter_type*)handle->scratch7 + handle->weight_copies * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S + ltid * handle->ofmblock * handle->ifmblock * 2;
LIBXSMM_VLA_DECL(2, float, filter_tmp, (float*)filter_scratch, handle->ofmblock);

element_input_type *scratch_tr_input = (element_input_type*)handle->scratch3;
element_input_type *zero_ptr_in;
LIBXSMM_VLA_DECL(4, element_input_type, tr_input, (element_input_type*) scratch_tr_input, handle->blocksifm, handle->ifmblock, handle->input_pixels);
LIBXSMM_VLA_DECL(5, element_input_type, tr_input_2, (element_input_type*) scratch_tr_input, handle->blocksifm, handle->ifmblock, handle->ifhp, handle->ifwp_extended);

element_output_type *scratch_tr_output = (element_input_type*)handle->scratch2;
LIBXSMM_VLA_DECL(5, element_output_type, tr_output, (element_output_type*) scratch_tr_output, handle->blocksofm, handle->output_pixels/2, handle->ofmblock, 2);
LIBXSMM_VLA_DECL(6, element_output_type, tr_output_2, (element_output_type*) scratch_tr_output, handle->blocksofm, handle->ofhp, handle->ofwp_extended/2, handle->ofmblock, 2);
element_output_type *out_ptr = (element_output_type*)handle->grad_output->data + ((size_t)handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
element_output_type *zero_ptr_out;

/* transpose, copy and reduce work-related variables  */
const int reduce_work = (handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S)/16 ;
const int reduce_chunksize = (reduce_work % handle->desc.threads == 0) ? (reduce_work / handle->desc.threads) : (reduce_work / handle->desc.threads) + 1;
const int reduce_thr_begin = (ltid * reduce_chunksize < reduce_work) ? (ltid * reduce_chunksize) : reduce_work;
const int reduce_thr_end = ((ltid + 1) * reduce_chunksize < reduce_work) ? ((ltid + 1) * reduce_chunksize) : reduce_work;

/* Batch reduce related variables */
const element_output_type *A_ptrs[1024];
const element_input_type  *B_ptrs[1024];
unsigned long long n_blocks = handle->batchreduce_h_pixels;

int LDA = handle->ofmblock;
int LDB = handle->input_pixels;
int LDC = handle->ofmblock;
int prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');

libxsmm_barrier_init(handle->barrier, ltid);

my_img_start = ltid;
my_img_end = ltid+1;

if (handle->upd_linearized_pixels == 1) {
  /* First transpose input and output */
  for (img = my_img_start; img < my_img_end; img++) {
    zero_ptr_in = (element_input_type*) &LIBXSMM_VLA_ACCESS(4, tr_input, img, 0, 0, 0, handle->blocksifm, handle->ifmblock, handle->input_pixels);
    memset(zero_ptr_in, 0, handle->desc.C * handle->input_pixels * sizeof(element_input_type));
    for (ifm1 = 0; ifm1 < handle->blocksifm; ifm1++) {
      for (ij = 0; ij < handle->ifhp; ij++) {
        for (ii = 0; ii < handle->ifwp; ii++) {
          for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
            LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, ifm2, ij * handle->ifwp + ii, handle->blocksifm, handle->ifmblock, handle->input_pixels) =
              LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij, ii, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          }
        }
      }
    }
  }
  /* Reformat output */
  for (img = my_img_start; img < my_img_end; img++) {
    zero_ptr_out = (element_output_type*) &LIBXSMM_VLA_ACCESS(5, tr_output, img, 0, 0, 0, 0, handle->blocksofm, handle->output_pixels/2, handle->ofmblock, 2);
    memset(zero_ptr_out, 0, handle->desc.K * handle->output_pixels * sizeof(element_output_type));
    for (ofm1 = 0; ofm1 < handle->blocksofm; ofm1++) {
      for (oi = 0; oi < handle->n_used_pixels; oi++) {
        for (ofm2 = 0; ofm2 < handle->ofmblock; ofm2++) {
          LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, oi/2, ofm2, oi%2, handle->blocksofm, handle->output_pixels/2, handle->ofmblock, 2) =
            *((element_output_type*)out_ptr + img * handle->blocksofm * handle->ofwp * handle->ofhp * handle->ofmblock + ofm1 * handle->ofwp * handle->ofhp * handle->ofmblock + oi * handle->ofmblock + ofm2);
        }
      }
    }
  }
} else {
  for (img = my_img_start; img < my_img_end; img++) {
    zero_ptr_in = (element_input_type*) &LIBXSMM_VLA_ACCESS(5, tr_input_2, img, 0, 0, 0, 0, handle->blocksifm, handle->ifmblock, handle->ifhp, handle->ifwp_extended);
    memset(zero_ptr_in, 0, handle->desc.C * handle->ifhp * handle->ifwp_extended * sizeof(element_input_type));
    for (ifm1 = 0; ifm1 < handle->blocksifm; ifm1++) {
      for (ij = 0; ij < handle->ifhp; ij++) {
        for (ii = 0; ii < handle->ifwp; ii++) {
          for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
              LIBXSMM_VLA_ACCESS(5, tr_input_2, img, ifm1, ifm2, ij, ii, handle->blocksifm, handle->ifmblock, handle->ifhp, handle->ifwp_extended) =
              LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij, ii, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          }
        }
      }
    }
  }
  for (img = my_img_start; img < my_img_end; img++) {
    for (ofm1 = 0; ofm1 < handle->blocksofm; ofm1++) {
      for (oj = 0; oj < handle->ofh; oj++) {
        for (oi = 0; oi < handle->ofw; oi++) {
          for (ofm2 = 0; ofm2 < handle->ofmblock; ofm2++) {
            LIBXSMM_VLA_ACCESS(6, tr_output_2, img, ofm1, oj, oi/2, ofm2, oi%2, handle->blocksofm, handle->ofhp, handle->ofwp_extended/2, handle->ofmblock, 2) =
              LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, ofm2, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
          }
        }
      }
    }
  }
}

/* Make sure we initialize intermediate weights to zero */
if (handle->use_intermediate_f32_wt_tensor) {
  memset(weight_ptr_f32, 0, handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S * sizeof(float));
}

const float beta = (handle->use_intermediate_f32_wt_tensor) ? 1.0 : 0.0;

float *dst_ptr;

__m256i c0, c1;
__m512i c01;
const __m512i perm_index = LIBXSMM_INTRINSICS_MM512_SET_EPI16(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8, 23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);

if (handle->upd_linearized_pixels == 0) {
  LDA = handle->ofmblock;
  LDB = handle->ifhp*handle->ifwp_extended;
  LDC = handle->ofmblock;
  prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
  l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  gemm_br_function br_gemm_kernel =  libxsmm_bsmmdispatch_reducebatch(handle->ofmblock, handle->ifmblock, handle->ofw, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);

  for (img = my_img_start; img < my_img_end; img++) {
    for (ofmb = 0; ofmb < handle->blocksofm; ofmb += handle->block_upd_ofm) {
      for (oj = 0; oj < handle->ofh; oj += handle->batchreduce_h_pixels){
        for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_upd_ifm) {
          for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_upd_ofm, handle->blocksofm); ofm1++ ) {
            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_upd_ifm, handle->blocksifm); ifm1++) {
              for (kj = 0; kj < handle->desc.R; ++kj) {
                for (ki = 0; ki < handle->desc.S; ++ki) {

                  /* Determine if destination is the accumulation scratch or the intermediate fp32 weight tensor */
                  if (handle->use_intermediate_f32_wt_tensor == 1) {
                    dst_ptr = (float*)&LIBXSMM_VLA_ACCESS(6, weight_private_f32, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
                  } else {
                    dst_ptr = (float*)&LIBXSMM_VLA_ACCESS(2, filter_tmp, 0, 0, handle->ofmblock);
                  }

                  /* Copy the input in such a way that we ignore "w-pixels" based on ki value  */
                  {

                  }

                  for (j_br = 0; j_br < handle->batchreduce_h_pixels; j_br++) {
                    A_ptrs[j_br] = (element_output_type*) &LIBXSMM_VLA_ACCESS(6, tr_output_2, img, ofm1, oj+j_br, 0, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp_extended/2, handle->ofmblock, 2);
                    B_ptrs[j_br] = (element_input_type*) &LIBXSMM_VLA_ACCESS(5, tr_input_2, img, ifm1, 0, (oj+j_br) * handle->desc.u + kj, ki, handle->blocksifm, handle->ifmblock, handle->ifhp, handle->ifwp_extended);
                  }

                  br_gemm_kernel(A_ptrs, B_ptrs, dst_ptr, &n_blocks);

                  /* Convert fully caccumulated buffer to bf16 weight buffer in case of full accumulation has happened */
                  if (oj + handle->batchreduce_h_pixels >= handle->ofh) {
                    LIBXSMM_VLA_DECL(2, float, filter_acc_buffer, (float*)dst_ptr, handle->ofmblock);
                    for (ij = 0; ij < handle->ifmblock; ij+=2) {
                      for (ii = 0; ii < handle->ofmblock; ii+=16) {
                        c0 = _mm512_loadcvtrne_fp32_bf16(&LIBXSMM_VLA_ACCESS(2, filter_acc_buffer, ij, ii, handle->ofmblock));
                        c1 = _mm512_loadcvtrne_fp32_bf16(&LIBXSMM_VLA_ACCESS(2, filter_acc_buffer, ij+1, ii, handle->ofmblock));
                        c01 = _mm512_inserti64x4 (c01, c0, 0);
                        c01 = _mm512_inserti64x4 (c01, c1, 1);
                        _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(7, weight_dst, ofm1, ifm1, kj, ki, ij/2, ii, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock/2, handle->ofmblock, 2), _mm512_permutexvar_epi16(perm_index, c01));
                      }
                    }
                  }

                }
              }
            }
          }
        }
      }
    }
  }
} else {
  LDA = handle->ofmblock;
  LDB = handle->input_pixels;
  LDC = handle->ofmblock;
  prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
  l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  gemm_function gemm_kernel = libxsmm_bsmmdispatch(handle->ofmblock, handle->ifmblock, handle->pixel_blocking, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);

  for (img = my_img_start; img < my_img_end; img++) {
    for (ofmb = 0; ofmb < handle->blocksofm; ofmb += handle->block_upd_ofm) {
      for (pix = 0; pix < handle->n_used_pixels; pix += handle->pixel_blocking){
        for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_upd_ifm) {
          for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_upd_ofm, handle->blocksofm); ofm1++ ) {
            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_upd_ifm, handle->blocksifm); ifm1++) {
              for (kj = 0; kj < handle->desc.R; ++kj) {
                for (ki = 0; ki < handle->desc.S; ++ki) {

                  /* Determine if destination is the accumulation scratch or the intermediate fp32 weight tensor */
                  if (handle->use_intermediate_f32_wt_tensor == 1) {
                    dst_ptr = (float*)&LIBXSMM_VLA_ACCESS(6, weight_private_f32, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
                  } else {
                    dst_ptr = (float*)&LIBXSMM_VLA_ACCESS(2, filter_tmp, 0, 0, handle->ofmblock);
                  }
                  gemm_kernel( &LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, pix/2, 0, 0, handle->blocksofm, handle->output_pixels/2, handle->ofmblock, 2),
                      &LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, 0, pix + kj * handle->ifwp + ki, handle->blocksifm, handle->ifmblock, handle->input_pixels),
                      dst_ptr);

                  /* Convert fully caccumulated buffer to bf16 weight buffer in case of full accumulation has happened */
                  if (pix + handle->pixel_blocking >= handle->n_used_pixels) {
                    LIBXSMM_VLA_DECL(2, float, filter_acc_buffer, (float*)dst_ptr, handle->ofmblock);
                    for (ij = 0; ij < handle->ifmblock; ij+=2) {
                      for (ii = 0; ii < handle->ofmblock; ii+=16) {
                        c0 = _mm512_loadcvtrne_fp32_bf16(&LIBXSMM_VLA_ACCESS(2, filter_acc_buffer, ij, ii, handle->ofmblock));
                        c1 = _mm512_loadcvtrne_fp32_bf16(&LIBXSMM_VLA_ACCESS(2, filter_acc_buffer, ij+1, ii, handle->ofmblock));
                        c01 = _mm512_inserti64x4 (c01, c0, 0);
                        c01 = _mm512_inserti64x4 (c01, c1, 1);
                        _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(7, weight_dst, ofm1, ifm1, kj, ki, ij/2, ii, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock/2, handle->ofmblock, 2), _mm512_permutexvar_epi16(perm_index, c01));
                      }
                    }
                  }

                }
              }
            }
          }
        }
      }
    }
  }
}

libxsmm_barrier_wait(handle->barrier, ltid);

if (handle->weight_copies > 1) {
  const int filter_size = handle->desc.R  * handle->desc.S * handle->desc.C * handle->desc.K;
  LIBXSMM_VLA_DECL(2, element_filter_type, weight_copies_buffer, (element_filter_type*)handle->scratch7, filter_size);
  element_filter_type *weight_global_ptr = (element_filter_type*) handle->grad_filter->data;
  for ( j = reduce_thr_begin; j < reduce_thr_end; j++) {
    __m512 weight_sum = _mm512_setzero_ps();
    for ( i = 0; i < handle->weight_copies; i++ ) {
      weight_sum = _mm512_add_ps(weight_sum, _mm512_loadcvt_bf16_fp32(&LIBXSMM_VLA_ACCESS(2, weight_copies_buffer, i, j*16, filter_size)));
    }
    _mm512_storecvtrne_fp32_bf16( ((libxsmm_bfloat16*) weight_global_ptr) + j*16, weight_sum);
  }
  libxsmm_barrier_wait(handle->barrier, ltid);
}

#undef _mm512_roundbf16rne
#undef _mm512_storecvtrne_fp32_bf16
#undef _mm512_loadcvt_bf16_fp32
#undef _mm512_loadcvtrne_fp32_bf16

