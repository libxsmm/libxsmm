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
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#define _mm512_roundbf16rne(A) LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(A)
#define _mm512_storecvtrne_fp32_bf16(A,B)  _mm256_stream_si256((__m256i*)(A),_mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_roundbf16rne((B)),16)))
#define _mm512_loadcvt_bf16_fp32(A)   _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(A))),16))
#define _mm512_loadcvtrne_fp32_bf16(A) _mm512_cvtepi32_epi16(_mm512_srai_epi32(_mm512_roundbf16rne(LIBXSMM_INTRINSICS_MM512_LOAD_PS(A)),16))

int img, my_img_start, my_img_end, ofmb, ifmb, ojb, ofm1, ifm1, ifm2, ofm2, oj, oi, ii, ij, kj, ki, ind, j_br, img_br, i, j, img_block_size = 1, my_ofm_start, my_ofm_end, my_ifm_start, my_ifm_end, block_ofm, block_ifm, pix, pixb;
/* computing first logical thread */
const int ltid = tid - start_thread;

#if 0
element_output_type *const out = (element_output_type*)handle->grad_output->data + ((size_t)handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
LIBXSMM_VLA_DECL(5, const element_output_type, output, (const element_output_type*)out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
#endif
LIBXSMM_VLA_DECL(5, const element_input_type, input, (const element_input_type*)handle->reg_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
LIBXSMM_VLA_DECL(7, element_filter_type, weight_global, (element_filter_type*)handle->grad_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock/2, handle->ofmblock, 2);
element_filter_type *weight_ptr = (element_filter_type*)handle->scratch7 + ltid * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S;
LIBXSMM_VLA_DECL(7, element_filter_type, weight_private, (element_filter_type*)weight_ptr, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock/2, handle->ofmblock, 2);

element_filter_type *filter_scratch = (element_filter_type*)handle->scratch7 + handle->weight_copies * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S + ltid * handle->ofmblock * handle->ifmblock * 2;
LIBXSMM_VLA_DECL(2, float, filter_tmp, (float*)filter_scratch, handle->ofmblock);

/* transpose, copy and reduce work-related variables  */
const int reduce_work = (handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S)/16 ;
const int reduce_chunksize = (reduce_work % handle->desc.threads == 0) ? (reduce_work / handle->desc.threads) : (reduce_work / handle->desc.threads) + 1;
const int reduce_thr_begin = (ltid * reduce_chunksize < reduce_work) ? (ltid * reduce_chunksize) : reduce_work;
const int reduce_thr_end = ((ltid + 1) * reduce_chunksize < reduce_work) ? ((ltid + 1) * reduce_chunksize) : reduce_work;


/* Batch reduce related variables */
const element_output_type *A_ptrs[1024];
const element_input_type  *B_ptrs[1024];
unsigned long long n_blocks;

libxsmm_barrier_init(handle->barrier, ltid);

my_img_start = ltid;
my_img_end = ltid+1;

/* First transpose input and output */
element_input_type *scratch_tr_input = (element_input_type*)handle->scratch3;
element_input_type *zero_ptr_in;
LIBXSMM_VLA_DECL(4, element_input_type, tr_input, (element_input_type*) scratch_tr_input, handle->blocksifm, handle->ifmblock, handle->input_pixels);
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

element_output_type *scratch_tr_output = (element_input_type*)handle->scratch2;
LIBXSMM_VLA_DECL(5, element_output_type, tr_output, (element_output_type*) scratch_tr_output, handle->blocksofm, handle->output_pixels/2, handle->ofmblock, 2);
element_output_type *out_ptr = (element_output_type*)handle->grad_output->data + ((size_t)handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
element_output_type *zero_ptr_out;
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

int LDA = handle->ofmblock;
int LDB = handle->input_pixels;
int LDC = handle->ofmblock;
int prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
const float beta = 0.0;
gemm_function gemm_kernel = libxsmm_bsmmdispatch(handle->ofmblock, handle->ifmblock, handle->pixel_blocking, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);

__m256i c0, c1;
__m512i c01;
const __m512i perm_index = LIBXSMM_INTRINSICS_MM512_SET_EPI16(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8, 23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);

for (img = my_img_start; img < my_img_end; img++) {
  for (ofmb = 0; ofmb < handle->blocksofm; ofmb += handle->block_upd_ofm) {
    for (pix = 0; pix < handle->n_used_pixels; pix += handle->pixel_blocking){
      for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_upd_ifm) {
        for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_upd_ofm, handle->blocksofm); ofm1++ ) {
          for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_upd_ifm, handle->blocksifm); ifm1++) {
            for (kj = 0; kj < handle->desc.R; ++kj) {
              for (ki = 0; ki < handle->desc.S; ++ki) {

                gemm_kernel( &LIBXSMM_VLA_ACCESS(5, tr_output, img, ofm1, pix/2, 0, 0, handle->blocksofm, handle->output_pixels/2, handle->ofmblock, 2),
                    &LIBXSMM_VLA_ACCESS(4, tr_input, img, ifm1, 0, pix + kj * handle->ifwp + ki, handle->blocksifm, handle->ifmblock, handle->input_pixels),
                    &LIBXSMM_VLA_ACCESS(2, filter_tmp, 0, 0, handle->ofmblock) );

                /* Convert scratch to bf16 output buffer  */
                if (pix + handle->pixel_blocking >= handle->n_used_pixels) {
                  for (ij = 0; ij < handle->ifmblock; ij+=2) {
                    for (ii = 0; ii < handle->ofmblock; ii+=16) {
                      c0 = _mm512_loadcvtrne_fp32_bf16(&LIBXSMM_VLA_ACCESS(2, filter_tmp, ij, ii, handle->ofmblock));
                      c1 = _mm512_loadcvtrne_fp32_bf16(&LIBXSMM_VLA_ACCESS(2, filter_tmp, ij+1, ii, handle->ofmblock));
                      c01 = _mm512_inserti64x4 (c01, c0, 0);
                      c01 = _mm512_inserti64x4 (c01, c1, 1);
                      _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(7, weight_private, ofm1, ifm1, kj, ki, ij/2, ii, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock/2, handle->ofmblock, 2), _mm512_permutexvar_epi16(perm_index, c01));
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

#if 0
if (handle->upd_use_batchreduce == 0 && handle->upd_linearized_tasklist == 0) {
  /* Parallelize over minibatch */
  const int img_work = handle->desc.N;
  const int img_chunksize = (img_work % handle->desc.threads == 0) ? (img_work / handle->desc.threads) : (img_work / handle->desc.threads) + 1;
  const float beta = ((img_chunksize == 1) && (handle->upd_ofh_rb == handle->ofh) && (handle->upd_ofw_rb == handle->ofw)) ? 0.0 : 1.0;
  gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ofmblock, handle->ifmblock, handle->upd_ofw_rb * handle->upd_ofh_rb, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);

  my_img_start = (ltid * img_chunksize < img_work) ? (ltid * img_chunksize) : img_work;
  my_img_end = ((ltid + 1) * img_chunksize < img_work) ? ((ltid + 1) * img_chunksize) : img_work;

  if (!((img_chunksize == 1) && (handle->upd_ofh_rb == handle->ofh) && (handle->upd_ofw_rb == handle->ofw))) {
    memset(weight_ptr, 0, handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S * sizeof(element_filter_type));
  }

  if (handle->upd_loop_order == 0) {
    for (img = my_img_start; img < my_img_end; img++) {
      for (ofmb = 0; ofmb < handle->blocksofm; ofmb += handle->block_upd_ofm) {
        for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_upd_ifm) {
          for (ojb = 0; ojb < handle->ofh; ojb += handle->upd_ofh_rb) {
            for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_upd_ofm, handle->blocksofm); ofm1++ ) {
              for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_upd_ifm, handle->blocksifm); ifm1++) {
                for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->upd_ofh_rb,handle->ofh); oj+= handle->upd_ofh_rb) {
                  for (oi = 0; oi < handle->ofw; oi += handle->upd_ofw_rb) {
                    for (kj = 0; kj < handle->desc.R; ++kj) {
                      for (ki = 0; ki < handle->desc.S; ++ki) {
                        ii = oi * handle->desc.u + ki;
                        ij = oj * handle->desc.v + kj;
                        gemm_kernel( &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                            &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij, ii, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, weight_private, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock) );
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
  if (handle->upd_loop_order == 1) {
    for (img = my_img_start; img < my_img_end; img++) {
      for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_upd_ifm) {
        for (ofmb = 0; ofmb < handle->blocksofm; ofmb += handle->block_upd_ofm) {
          for (ojb = 0; ojb < handle->ofh; ojb += handle->upd_ofh_rb) {
            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_upd_ifm, handle->blocksifm); ifm1++) {
              for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_upd_ofm, handle->blocksofm); ofm1++ ) {
                for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->upd_ofh_rb,handle->ofh); oj+= handle->upd_ofh_rb) {
                  for (oi = 0; oi < handle->ofw; oi += handle->upd_ofw_rb) {
                    for (kj = 0; kj < handle->desc.R; ++kj) {
                      for (ki = 0; ki < handle->desc.S; ++ki) {
                        ii = oi * handle->desc.u + ki;
                        ij = oj * handle->desc.v + kj;
                        gemm_kernel( &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                            &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij, ii, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, weight_private, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock) );
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
  if (handle->upd_linearized_tasklist == 1) {
    /* Amount of work when using linearized view of tasks */
    const int work = handle->desc.R * handle->desc.S * handle->blocksofm * handle->blocksifm;
    const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
    const int work_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
    const int work_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
    int work_item;
    int Cb = handle->blocksifm;
#if 0
    int Kb = handle->blocksofm;
#endif
    int R = handle->desc.R;
    int S = handle->desc.S;

    if (handle->upd_avoid_rim_fmas == 0) {
      const int IFH = (handle->upd_pack_input == 1) ? handle->ifhp/handle->desc.u : handle->ifhp;
      const int IFW = (handle->upd_pack_input == 1) ? handle->ifwp/handle->desc.v : handle->ifwp;
      element_input_type *input_ptr_base = (handle->upd_pack_input == 1) ? (element_input_type*)handle->scratch1 + handle->blocksifm * handle->ifmblock * handle->blocksofm * handle->ofmblock * handle->desc.R * handle->desc.S : (element_input_type*)handle->reg_input->data;
      LIBXSMM_VLA_DECL(5, element_input_type, input_use, (element_input_type*)input_ptr_base, handle->blocksifm, IFH, IFW, handle->ifmblock);
      const float beta = ((handle->desc.N == 1) && (handle->upd_ofh_rb == handle->ofh) && (handle->upd_ofw_rb == handle->ofw)) ? 0.0 : 1.0;
      gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ofmblock, handle->ifmblock, handle->upd_ofw_rb * handle->upd_ofh_rb, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);

      /* If requested, pack input to avoid strided accesses */
      if (handle->upd_pack_input == 1) {
        LIBXSMM_VLA_DECL(5, element_input_type, input_src, (element_input_type*)handle->reg_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
        const int img_chunk = (handle->desc.N % handle->desc.threads == 0) ? handle->desc.N/handle->desc.threads : (handle->desc.N/handle->desc.threads) + 1;
        const int img_copy_start = LIBXSMM_MIN(ltid*img_chunk, handle->desc.N);
        const int img_copy_end = LIBXSMM_MIN((ltid+1)*img_chunk, handle->desc.N);

        for (img = img_copy_start; img < img_copy_end; img++) {
          for (ifm1 = 0; ifm1 < handle->blocksifm; ifm1++) {
            for (oj = 0; oj < handle->ofh; oj++) {
              for (oi = 0; oi < handle->ofw; oi++) {
                ij = oj * handle->desc.u;
                ii = oi * handle->desc.v;
                LIBXSMM_PRAGMA_SIMD
                  for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
                    LIBXSMM_VLA_ACCESS(5, input_use, img, ifm1, oj, oi, ifm2, handle->blocksifm, IFH, IFW, handle->ifmblock) = LIBXSMM_VLA_ACCESS(5, input_src, img, ifm1, ij, ii, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                  }
              }
            }
          }
        }
        libxsmm_barrier_wait(handle->barrier, ltid);
      }

      /* Initialize weights to zero */
      if (!((handle->desc.N == 1) && (handle->upd_ofh_rb == handle->ofh) && (handle->upd_ofw_rb == handle->ofw))) {
        for (work_item = work_begin; work_item < work_end; work_item++) {
          ofm1 = work_item/(Cb*R*S);
          ifm1 = (work_item%(Cb*R*S))/(R*S);
          kj = ((work_item%(Cb*R*S))%(R*S))/S;
          ki = ((work_item%(Cb*R*S))%(R*S))%S;

          for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
            LIBXSMM_PRAGMA_SIMD
              for (ofm2 = 0; ofm2 < handle->ofmblock; ofm2++) {
                LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, ifm2, ofm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock) = (element_filter_type)0;
              }
          }
        }
      }

      for (img = 0; img < handle->desc.N; img++) {
        for (work_item = work_begin; work_item < work_end; work_item++) {
          ofm1 = work_item/(Cb*R*S);
          ifm1 = (work_item%(Cb*R*S))/(R*S);
          kj = ((work_item%(Cb*R*S))%(R*S))/S;
          ki = ((work_item%(Cb*R*S))%(R*S))%S;
          oi = 0;
          ii = ki;
          for (oj = 0; oj < handle->ofh; oj += handle->upd_ofh_rb) {
            ij = oj * handle->desc.u + kj;
            gemm_kernel( &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                &LIBXSMM_VLA_ACCESS(5, input_use, img, ifm1, ij, ii, 0, handle->blocksifm, IFH, IFW, handle->ifmblock),
                &LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock) );
          }
        }
      }
    } else {
      const float beta = ((handle->upd_ofh_rb == handle->ofh) && (handle->upd_ofw_rb == handle->ofw)) ? 0.0 : 1.0;
      gemm_br_function br_gemm_kernel = libxsmm_smmdispatch_reducebatch(handle->ofmblock, handle->ifmblock, handle->upd_ofw_rb, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);
      gemm_br_function br_gemm_kernel2 = libxsmm_smmdispatch_reducebatch(handle->ofmblock, handle->ifmblock, handle->upd_ofw_rb-1, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);

      for (work_item = work_begin; work_item < work_end; work_item++) {
        ofm1 = work_item/(Cb*R*S);
        ifm1 = (work_item%(Cb*R*S))/(R*S);
        kj = ((work_item%(Cb*R*S))%(R*S))/S;
        ki = ((work_item%(Cb*R*S))%(R*S))%S;
        oi = 0;
        oj = 0;
        ii = oi * handle->desc.u + ki;
        ij = oj * handle->desc.v + kj;
        img = 0;
        img_block_size = handle->desc.N;

        if (kj == 0) {
          ind = 0;
          for (img_br = 0; img_br < img_block_size; img_br++) {
            for (j_br = 1; j_br < handle->upd_ofh_rb; j_br++) {
              A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img + img_br, ofm1, oj + j_br, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
              B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img + img_br, ifm1, ij + j_br * handle->desc.u, ii, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              ind++;
            }
          }
          n_blocks = ind;
          br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
        } else if (ki == 0) {
          ind = 0;
          for (img_br = 0; img_br < img_block_size; img_br++) {
            for (j_br = 0; j_br < handle->upd_ofh_rb; j_br++) {
              A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img + img_br, ofm1, oj + j_br, oi + 1, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
              B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img + img_br, ifm1, ij + j_br * handle->desc.u, ii + 1, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              ind++;
            }
          }
          n_blocks = ind;
          br_gemm_kernel2(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
        } else if (oi == handle->ofw-handle->fwd_ofw_rb  && ki == handle->desc.S-1) {
          ind = 0;
          for (img_br = 0; img_br < img_block_size; img_br++) {
            for (j_br = 0; j_br < handle->upd_ofh_rb; j_br++) {
              A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img + img_br, ofm1, oj + j_br, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
              B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img + img_br, ifm1, ij + j_br * handle->desc.u, ii, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              ind++;
            }
          }
          n_blocks = ind;
          br_gemm_kernel2(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
        } else {
          if (kj == handle->desc.R-1) {
            ind = 0;
            for (img_br = 0; img_br < img_block_size; img_br++) {
              for (j_br = 0; j_br < handle->upd_ofh_rb-1; j_br++) {
                A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img + img_br, ofm1, oj + j_br, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img + img_br, ifm1, ij + j_br * handle->desc.u, ii, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                ind++;
              }
            }
            n_blocks = ind;
            br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
          } else {
            ind = 0;
            for (img_br = 0; img_br < img_block_size; img_br++) {
              for (j_br = 0; j_br < handle->upd_ofh_rb; j_br++) {
                A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img + img_br, ofm1, oj + j_br, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img + img_br, ifm1, ij + j_br * handle->desc.u, ii, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                ind++;
              }
            }
            n_blocks = ind;
            br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_global, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
          }
        }
      }
    }
  } else {
    /* Here we are using batch-reduce kernel and hybrid minibatch/FM parallelization */
    /* FIXME: Hardcoed logic for N=27  */
    int group_size = (handle->desc.threads == 27 && handle->desc.N == 27 && handle->ofw == 14 && handle->desc.R == 1 && handle->desc.u == 1 && ltid >= 24) ? 3 : ((handle->desc.threads+handle->weight_copies-1)/handle->weight_copies);
    int tile_id = ltid/( (handle->desc.threads+handle->weight_copies-1)/handle->weight_copies );
    int tiles = handle->weight_copies;
    int img_per_tile = (handle->desc.N+tiles-1)/tiles;
    int my_in_tile_id = ltid % group_size;
    int ifms_per_thread = (handle->blocksifm+group_size-1)/group_size;
    int ofms_per_thread = (handle->blocksofm+group_size-1)/group_size;
    int my_R_start = 0;
    int my_R_end = handle->desc.R;
    const float beta = ((handle->upd_ofh_rb == handle->ofh) && (handle->upd_ofw_rb == handle->ofw)) ? 0.0 : 1.0;
    gemm_br_function br_gemm_kernel = libxsmm_smmdispatch_reducebatch(handle->ofmblock, handle->ifmblock, handle->upd_ofw_rb, &LDA, &LDB, &LDC, NULL, &beta, &l_flags, &prefetch_mode);
    element_filter_type *weight_ptr_group = (handle->weight_copies > 1) ? (element_filter_type*)handle->scratch7 + tile_id * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S : (element_filter_type*)handle->grad_filter->data;
    LIBXSMM_VLA_DECL(6, element_filter_type, weight_private_group, (element_filter_type*)weight_ptr_group, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
    my_img_start = LIBXSMM_MIN( tile_id * img_per_tile, handle->desc.N);
    my_img_end = LIBXSMM_MIN( (tile_id+1) * img_per_tile, handle->desc.N);
    my_ifm_start = LIBXSMM_MIN( my_in_tile_id * ifms_per_thread, handle->blocksifm  );
    my_ifm_end = LIBXSMM_MIN( (my_in_tile_id+1) * ifms_per_thread, handle->blocksifm  );
    my_ofm_start = 0;
    my_ofm_end = handle->blocksofm;
    /* FIXME: Hardcoed logic for N=27  */
    if (handle->desc.threads == 27 && handle->desc.N == 27 && handle->desc.C == 256 && handle->desc.K == 1024 && handle->ofh == 14 && handle->desc.u == 1) {
      my_ofm_start = LIBXSMM_MIN( my_in_tile_id * ofms_per_thread, handle->blocksofm  );
      my_ofm_end = LIBXSMM_MIN( (my_in_tile_id+1) * ofms_per_thread, handle->blocksofm  );
      my_ifm_start = 0;
      my_ifm_end = handle->blocksifm;
    }
    if (handle->desc.threads == 27 && handle->desc.N == 27 && handle->desc.R == 3 && handle->desc.S == 3 && handle->ofh == 14) {
      int r_per_tile = (handle->desc.R+group_size-1)/group_size;
      my_ifm_start = 0;
      my_ifm_end = handle->blocksifm;
      my_ofm_start = 0;
      my_ofm_end = handle->blocksofm;
      my_R_start = LIBXSMM_MIN( my_in_tile_id * r_per_tile, handle->desc.R );
      my_R_end = LIBXSMM_MIN( (my_in_tile_id+1) * r_per_tile, handle->desc.R );
    }
    block_ofm = my_ofm_end-my_ofm_start+1;
    block_ifm = my_ifm_end-my_ifm_start+1;
    img_block_size = my_img_end - my_img_start;

    /* May need to initialized private weights to zero  */
    if (!((handle->upd_ofh_rb == handle->ofh) && (handle->upd_ofw_rb == handle->ofw))) {
      for (ofm1 = my_ofm_start; ofm1 < my_ofm_end; ofm1++ ) {
        for (ifm1 = my_ifm_start; ifm1 < my_ifm_end; ifm1++) {
          for (kj = my_R_start; kj < my_R_end; ++kj) {
            for (ki = 0; ki < handle->desc.S; ++ki) {
              for (ofm2 = 0; ofm2 < handle->ofmblock; ofm2++ ) {
                for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
                  LIBXSMM_VLA_ACCESS(6, weight_private_group, ofm1, ifm1, kj, ki, ifm2, ofm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock) = (element_filter_type)0;
                }
              }
            }
          }
        }
      }
    }

    if (handle->upd_loop_order == 0) {
      for (img = my_img_start; img < my_img_end; img += img_block_size) {
        for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += block_ofm) {
          for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += block_ifm) {
            for (ojb = 0; ojb < handle->ofh; ojb += handle->upd_ofh_rb) {
              for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+block_ofm, my_ofm_end); ofm1++ ) {
                for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+block_ifm, my_ifm_end); ifm1++) {
                  for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->upd_ofh_rb,handle->ofh); oj+= handle->upd_ofh_rb) {
                    for (oi = 0; oi < handle->ofw; oi += handle->upd_ofw_rb) {
                      for (kj = my_R_start; kj < my_R_end; ++kj) {
                        for (ki = 0; ki < handle->desc.S; ++ki) {
                          ii = oi * handle->desc.u + ki;
                          ij = oj * handle->desc.v + kj;
                          ind = 0;
                          for (img_br = 0; img_br < img_block_size; img_br++) {
                            for (j_br = 0; j_br < handle->upd_ofh_rb; j_br++) {
                              A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img + img_br, ofm1, oj + j_br, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                              B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img + img_br, ifm1, ij + j_br * handle->desc.u, ii, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                              ind++;
                            }
                          }
                          n_blocks = ind;
                          br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_private_group, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
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
      for (img = my_img_start; img < my_img_end; img += img_block_size) {
        for (ifmb = my_ifm_start; ifmb < my_ifm_end; ifmb += block_ifm) {
          for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += block_ofm) {
            for (ojb = 0; ojb < handle->ofh; ojb += handle->upd_ofh_rb) {
              for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+block_ifm, my_ifm_end); ifm1++) {
                for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+block_ofm, my_ofm_end); ofm1++ ) {
                  for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->upd_ofh_rb,handle->ofh); oj+= handle->upd_ofh_rb) {
                    for (oi = 0; oi < handle->ofw; oi += handle->upd_ofw_rb) {
                      for (kj = my_R_start; kj < my_R_end; ++kj) {
                        for (ki = 0; ki < handle->desc.S; ++ki) {
                          ii = oi * handle->desc.u + ki;
                          ij = oj * handle->desc.v + kj;
                          ind = 0;
                          for (img_br = 0; img_br < img_block_size; img_br++) {
                            for (j_br = 0; j_br < handle->upd_ofh_rb; j_br++) {
                              A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img + img_br, ofm1, oj + j_br, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                              B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img + img_br, ifm1, ij + j_br * handle->desc.u, ii, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                              ind++;
                            }
                          }
                          n_blocks = ind;
                          br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_private_group, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
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
}

if (handle->weight_copies > 1) {
  /* reduce work-related variables  */
  const int reduce_work = handle->blocksofm * handle->blocksifm * handle->desc.R * handle->desc.S * (handle->ofmblock/16) * handle->ifmblock;
  const int reduce_chunksize = (reduce_work % handle->desc.threads == 0) ? (reduce_work / handle->desc.threads) : (reduce_work / handle->desc.threads) + 1;
  const int reduce_thr_begin = (ltid * reduce_chunksize < reduce_work) ? (ltid * reduce_chunksize) : reduce_work;
  const int reduce_thr_end = ((ltid + 1) * reduce_chunksize < reduce_work) ? ((ltid + 1) * reduce_chunksize) : reduce_work;

  /* Perform reduction here  */
  libxsmm_barrier_wait(handle->barrier, ltid);

  for ( ij = reduce_thr_begin; ij < reduce_thr_end; ij++ ) {
    element_filter_type *weight_ptr_glb = (element_filter_type*) handle->grad_filter->data;
#if 1
    float weight_sum[16];
    unsigned int wtcnt = 0;

    LIBXSMM_PRAGMA_SIMD
      for ( wtcnt = 0; wtcnt < 16; ++wtcnt ) {
        weight_sum[wtcnt] = 0.0f;
      }

    for ( ii = 0; ii < handle->weight_copies; ii++ ) {
      element_filter_type *weight_ptr_src = (element_filter_type*)handle->scratch7 + ii * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S + ij * 16;
      LIBXSMM_PRAGMA_SIMD
        for ( wtcnt = 0; wtcnt < 16; ++wtcnt ) {
          weight_sum[wtcnt] += weight_ptr_src[wtcnt];
        }
    }

    LIBXSMM_PRAGMA_SIMD
      for ( wtcnt = 0; wtcnt < 16; ++wtcnt ) {
        weight_ptr_glb[(ij*16) + wtcnt] = weight_sum[wtcnt];
      }
#else
    __m512 weight_sum = _mm512_setzero_ps();
    for ( ii = 0; ii < handle->weight_copies; ii++ ) {
      element_filter_type *weight_ptr_src = (element_filter_type*)handle->scratch7 + ii * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S + ij * 16;
      weight_sum = _mm512_add_ps(weight_sum, LIBXSMM_INTRINSICS_MM512_LOAD_PS(weight_ptr_src));
    }
    _mm512_store_ps(&weight_ptr_glb[ij*16], weight_sum);
#endif
  }
}

libxsmm_barrier_wait(handle->barrier, ltid);

#endif

