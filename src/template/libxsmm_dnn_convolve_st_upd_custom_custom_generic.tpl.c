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
/* Evangelos Georganas, Rajkishore Barik, Alexander Heinecke (Intel Corp.)
******************************************************************************/

int img, my_img_start, my_img_end, ofmb, ifmb, ojb, ofm1, ifm1, oj, oi, ii, ij, kj, ki, ind, j_br;
/* computing first logical thread */
const int ltid = tid - start_thread;

element_output_type *const out = (element_output_type*)handle->grad_output->data + ((size_t)handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
LIBXSMM_VLA_DECL(5, const element_output_type, output, (const element_output_type*)out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
LIBXSMM_VLA_DECL(5, const element_input_type, input, (const element_input_type*)handle->reg_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
LIBXSMM_VLA_DECL(6, element_filter_type, weight_global, (element_filter_type*)handle->grad_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
element_filter_type *weight_ptr = (element_filter_type*)handle->scratch7 + ltid * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S;
LIBXSMM_VLA_DECL(6, element_filter_type, weight_private, (element_filter_type*)weight_ptr, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);

/* reduce work-related variables  */
const int reduce_work = handle->blocksofm * handle->blocksifm * handle->desc.R * handle->desc.S * handle->ofmblock * (handle->ifmblock/16);
const int reduce_chunksize = (reduce_work % handle->desc.threads == 0) ? (reduce_work / handle->desc.threads) : (reduce_work / handle->desc.threads) + 1;
const int reduce_thr_begin = (ltid * reduce_chunksize < reduce_work) ? (ltid * reduce_chunksize) : reduce_work;
const int reduce_thr_end = ((ltid + 1) * reduce_chunksize < reduce_work) ? ((ltid + 1) * reduce_chunksize) : reduce_work;

/* Batch reduce related variables */
const element_output_type *A_ptrs[1024];
const element_input_type  *B_ptrs[1024];
unsigned long long n_blocks;

my_img_start = ltid;
my_img_end = ltid+1;

libxsmm_barrier_init(handle->barrier, ltid);

if (handle->avoid_init_weights == 0) {
  memset(weight_ptr, 0, handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S * sizeof(element_filter_type));
}

if (handle->upd_use_batchreduce == 0) {
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
                      gemm_kernel( &LIBXSMM_VLA_ACCESS(5,      output,  img, ofm1, oj,      oi,  0,     handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                          &LIBXSMM_VLA_ACCESS(5,      input,  img, ifm1, ij,      ii,  0,     handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                          &LIBXSMM_VLA_ACCESS(6,      weight_private, ofm1, ifm1, kj,      ki, 0, 0,  handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock) );
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
                      ind = 0;
                      for (j_br = 0; j_br < handle->upd_ofh_rb; j_br++) {
                        A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj + j_br, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                        B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij + j_br, ii, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                        ind++;
                      }
                      n_blocks = ind;
                      br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(6, weight_private, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock), &n_blocks);
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

/* Perform reduction here  */
libxsmm_barrier_wait(handle->barrier, ltid);

for ( ij = reduce_thr_begin; ij < reduce_thr_end; ij++ ) {
  element_filter_type *weight_ptr = (element_filter_type*) handle->grad_filter->data;
  __m512 weight_sum = _mm512_setzero_ps();
  for ( ii = 0; ii < handle->desc.threads; ii++ ) {
    element_filter_type *weight_ptr_src = (element_filter_type*)handle->scratch7 + ii * handle->desc.C * handle->desc.K * handle->desc.R * handle->desc.S + ij * 16;
    weight_sum = _mm512_add_ps(weight_sum, LIBXSMM_INTRINSICS_MM512_LOAD_PS(weight_ptr_src));
  }
  _mm512_store_ps(&weight_ptr[ij*16], weight_sum);
}

libxsmm_barrier_wait(handle->barrier, ltid);


