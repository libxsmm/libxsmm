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

int img, my_img_start, my_img_end, ofmb, ifmb, ojb, ofm1, ifm1, oj, oi, ii, ij, kj, ki;
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

#if 0
int ofm1ifm1, img, ofm1, ifm1, oj, ij, oi, ii, kj, ki, ifm2, ofm2;
/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
const int work = handle->blocksifm * handle->blocksofm;
/* compute chunk size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* transpose + padding via stack allocated buffers for input */
const int padded_w = handle->desc.W + (2 * handle->desc.pad_w);
const int padded_h = handle->desc.H + (2 * handle->desc.pad_h);
const int size_tls1 = padded_h * padded_w * handle->ifmblock;
element_input_type *const input_scratch = (element_input_type*)(((char*)handle->scratch5) +
    ltid * LIBXSMM_UP2(size_tls1 * sizeof(element_input_type), LIBXSMM_CACHELINE));

/* transpose via stack allocated buffers for output and weights to control stride-GEMM issue
idea: we transpose grad_output and transpose filters when done */
const int scratch6_size = handle->ofhp * handle->ofwp * handle->ofmblock;
const int scratch7_size = handle->desc.R * handle->desc.S * handle->ifmblock * handle->ofmblock;
element_output_type *const output_scratch = (element_output_type*)(((char*)handle->scratch6) +
    ltid * LIBXSMM_UP2(scratch6_size * sizeof(element_output_type), LIBXSMM_CACHELINE));
element_filter_type *const filter_scratch = (element_filter_type*)(((char*)handle->scratch7) +
    ltid * LIBXSMM_UP2(scratch7_size * sizeof(element_filter_type), LIBXSMM_CACHELINE));

element_output_type *const out = (element_output_type*)handle->grad_output->data + ((size_t)handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
LIBXSMM_VLA_DECL(5, const element_output_type, output, (const element_output_type*)out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
LIBXSMM_VLA_DECL(5, const element_output_type, output_padded, (const element_output_type*)handle->grad_output->data, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
LIBXSMM_VLA_DECL(5, const element_input_type, input, (const element_input_type*)handle->reg_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
LIBXSMM_VLA_DECL(6, element_filter_type, weight, (element_filter_type*)handle->grad_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
LIBXSMM_VLA_DECL(3, element_input_type, input_trans, input_scratch, handle->ifmblock, padded_w);
LIBXSMM_VLA_DECL(3, element_input_type, input_padded, input_scratch, padded_w, handle->ifmblock);
LIBXSMM_VLA_DECL(3, element_output_type, output_trans, output_scratch, handle->ofmblock, handle->ofwp);
LIBXSMM_VLA_DECL(4, element_filter_type, weight_local, filter_scratch, handle->desc.S, handle->ofmblock, handle->ifmblock);

/* zeroing local scratch after declarations (not mixing declarations and code) */
for (ii = 0; ii < size_tls1; ++ii) { input_scratch[ii] = (element_input_type)0; }
for (oi = 0; oi < scratch6_size; ++oi) { output_scratch[oi] = (element_output_type)0; }
for (oi = 0; oi < scratch7_size; ++oi) { filter_scratch[oi] = (element_filter_type)0; }

for (ofm1ifm1 = thr_begin; ofm1ifm1 < thr_end; ++ofm1ifm1) {
  ofm1 = ofm1ifm1 / handle->blocksifm;
  ifm1 = ofm1ifm1 % handle->blocksifm;
  /* reset result buffer to zero when intent is to overwrite when first block
     of input channels should be convoluted */
  if ( ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ) {
    element_filter_type* temp_buf = &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);

    LIBXSMM_PRAGMA_SIMD
      for (ii = 0; ii < scratch7_size; ++ii) {
        temp_buf[ii] = (element_filter_type)0;
      }
  }

  for (img = 0; img < handle->desc.N; ++img) {
    /* we can only run GEMM based code for update in case of 1x1 convolutions or
       if the kernel size is bigger 1 and there is no stride */
    if (    ( (handle->desc.R == 1) && (handle->desc.S == 1) && (handle->desc.pad_h == handle->desc.pad_h_in) && (handle->desc.pad_w == handle->desc.pad_w_in) )
        || ( (handle->desc.u == 1) && (handle->desc.v == 1) ) ) {
      /* first we need to transpose in order to use a GEMM
         we also do, if needed, padding for the input activations */
      if ( (handle->desc.pad_h == handle->desc.pad_h_in) && (handle->desc.pad_w == handle->desc.pad_w_in) ) {
        for (ij = 0; ij < handle->ifhp/handle->desc.u; ++ij) {
          for (ii = 0; ii < handle->ifwp/handle->desc.v; ++ii) {
            for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
              LIBXSMM_VLA_ACCESS(3, input_trans, ij, ifm2, ii, handle->ifmblock, padded_w) =
                LIBXSMM_VLA_ACCESS(5,  input, img, ifm1, ij*handle->desc.u, ii*handle->desc.v, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
            }
          }
        }
      } else {
        for (ij = 0; ij < handle->desc.H/handle->desc.u; ++ij) {
          for (ii = 0; ii < handle->desc.W/handle->desc.v; ++ii) {
            for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
              LIBXSMM_VLA_ACCESS(3, input_trans, ij + handle->desc.pad_h, ifm2, ii + handle->desc.pad_w, handle->ifmblock, padded_w) =
                LIBXSMM_VLA_ACCESS(5,  input, img, ifm1, ij*handle->desc.u, ii*handle->desc.v, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
            }
          }
        }
      }

      for (oj = 0; oj < handle->ofh; ++oj) {
        ij = oj; oi = 0; ii = 0;
        for (kj = 0; kj < handle->desc.R; ++kj) {
          for (ki = 0; ki < handle->desc.S; ++ki) {
            /* let's do a 16x16xofw GEMM :-): M=nbOfm, N=nbIfm, K=ofw (col-major) */
            gemm_kernel( &LIBXSMM_VLA_ACCESS(5,      output,  img, ofm1, oj,      0,  0,     handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                &LIBXSMM_VLA_ACCESS(3, input_trans,             ij + kj, 0,  ki,    handle->ifmblock, padded_w),
                &LIBXSMM_VLA_ACCESS(6,      weight, ofm1, ifm1, kj,      ki, 0, 0,  handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock) );
          }
        }
      }
    } else {
      /* we need to set local weight copy to 0 */
      LIBXSMM_PRAGMA_SIMD
        for (oi = 0; oi < handle->desc.R*handle->desc.S*handle->ofmblock*handle->ifmblock; ++oi) {
          filter_scratch[oi] = (element_filter_type)0;
        }

      /* if we have physically padded buffer, we can directly operate on the input data */
      if ( (handle->desc.pad_h == handle->desc.pad_h_in) && (handle->desc.pad_w == handle->desc.pad_w_in) ) {
        /* we now transpose output to compute transposed filters */
        for (oj = 0; oj < handle->ofhp; ++oj) {
          for (oi = 0; oi < handle->ofwp; ++oi) {
            for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
              LIBXSMM_VLA_ACCESS(3, output_trans, oj, ofm2, oi, handle->ofmblock, handle->ofwp) =
                LIBXSMM_VLA_ACCESS(5,  output_padded, img, ofm1, oj, oi, ofm2, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
            }
          }
        }

        /* now we run a strided vector operation */
        for (oj = 0; oj < handle->ofh; ++oj) {
          ij = oj * handle->desc.u;
          oi = 0, ii = 0;
          for (kj = 0; kj < handle->desc.R; ++kj) {
            for (ki = 0; ki < handle->desc.S; ++ki) {
              gemm_kernel_alt( &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij+kj, ii+ki, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                  &LIBXSMM_VLA_ACCESS(3, output_trans, oj+handle->desc.pad_h_out, 0, oi+handle->desc.pad_w_out, handle->ofmblock, handle->ofwp ),
                  &LIBXSMM_VLA_ACCESS(4, weight_local, kj, ki, 0, 0, handle->desc.S, handle->ofmblock, handle->ifmblock) );
            }
          }
        }
      } else {
        /* padding is needed, we first padded input */
        for (ij = 0; ij < handle->desc.H; ++ij) {
          for (ii = 0; ii < handle->desc.W; ++ii) {
            LIBXSMM_PRAGMA_SIMD
              for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                LIBXSMM_VLA_ACCESS(3, input_padded, ij + handle->desc.pad_h, ii + handle->desc.pad_w, ifm2, padded_w, handle->ifmblock) =
                  LIBXSMM_VLA_ACCESS(5,  input, img, ifm1, ij, ii, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              }
          }
        }

        /* we now transpose output to compute transposed filters */
        for (oj = 0; oj < handle->ofh; ++oj) {
          for (oi = 0; oi < handle->ofw; ++oi) {
            for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
              LIBXSMM_VLA_ACCESS(3, output_trans, oj, ofm2, oi, handle->ofmblock, handle->ofwp) =
                LIBXSMM_VLA_ACCESS(5,  output, img, ofm1, oj, oi, ofm2, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
            }
          }
        }

        /* now we run a strided vector operation */
        for (oj = 0; oj < handle->ofh; ++oj) {
          ij = oj * handle->desc.u;
          oi = 0, ii = 0;
          for (kj = 0; kj < handle->desc.R; ++kj) {
            for (ki = 0; ki < handle->desc.S; ++ki) {
              gemm_kernel_alt( &LIBXSMM_VLA_ACCESS(3, input_padded,       ij+kj, ii+ki, 0, padded_w, handle->ifmblock),
                  &LIBXSMM_VLA_ACCESS(3, output_trans,       oj, 0, oi, handle->ofmblock, handle->ofwp ),
                  &LIBXSMM_VLA_ACCESS(4, weight_local, kj, ki, 0, 0, handle->desc.S, handle->ofmblock, handle->ifmblock) );
            }
          }
        }
      }
      /* transpose filter back and update the master copy */
      for (kj = 0; kj < handle->desc.R; ++kj) {
        for (ki = 0; ki < handle->desc.S; ++ki) {
          for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
            for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
              LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, kj, ki, ifm2, ofm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock) +=
                LIBXSMM_VLA_ACCESS(4, weight_local, kj, ki, ofm2, ifm2, handle->desc.S, handle->ofmblock, handle->ifmblock);
            }
          }
        }
      }
    }
  }
}

#endif

