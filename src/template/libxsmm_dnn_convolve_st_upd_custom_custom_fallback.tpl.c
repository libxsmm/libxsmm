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
/* Rajkishore Barik, Alexander Heinecke (Intel Corp.)
******************************************************************************/

int ofm1ifm1, img, ofm1, ifm1, oj, ij, oi, ii, kj, ki, ifm2, ofm2;
/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
const int work = handle->blocksifm * handle->blocksofm;
/* compute chunck size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* transpose + padding via stack allocated buffers */
const int padded_h = handle->desc.H + (2 * handle->desc.pad_h);
const int padded_w = handle->desc.W + (2 * handle->desc.pad_w);
element_input_type input_scratch[padded_h*padded_w*handle->ifmblock]; /* this is a [H][c-block][W] or [H][c-block][W] tensor */
for ( ii = 0; ii < padded_h*padded_w*handle->ifmblock; ++ii ) { input_scratch[ii] = (element_input_type)0; }

element_output_type *const out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
LIBXSMM_VLA_DECL(5, const element_output_type, output, out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
LIBXSMM_VLA_DECL(5, const element_input_type, input, (element_input_type*)handle->reg_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
LIBXSMM_VLA_DECL(6, element_filter_type, weight, (element_filter_type*)handle->grad_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
LIBXSMM_VLA_DECL(3, element_input_type, input_trans, input_scratch, handle->ifmblock, padded_w);
LIBXSMM_VLA_DECL(3, element_input_type, input_padded, input_scratch, padded_w, handle->ifmblock);

for (ofm1ifm1 = thr_begin; ofm1ifm1 < thr_end; ++ofm1ifm1) {
  ofm1 = ofm1ifm1 / handle->blocksifm;
  ifm1 = ofm1ifm1 % handle->blocksifm;
  /* reset result buffer to zero when intent is to overwrite when first block
     of input channels should be convoluted */
  if ( ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ) {
    element_filter_type* temp_buf = &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);

    LIBXSMM_PRAGMA_SIMD
    for (ii = 0; ii < handle->desc.R*handle->desc.S*handle->ifmblock*handle->ofmblock; ++ii) {
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
      /* if we have physically padded buffer, we can directly operate on the input data */
      if ( (handle->desc.pad_h == handle->desc.pad_h_in) && (handle->desc.pad_w == handle->desc.pad_w_in) ) {
        /* now we run a strided vector operation */
        for (oj = 0; oj < handle->ofh; ++oj) {
          ij = oj * handle->desc.u;
          for (kj = 0; kj < handle->desc.R; ++kj) {
            for (ki = 0; ki < handle->desc.S; ++ki) {
              for (oi = 0; oi < handle->ofw; ++oi) {
                ii = oi * handle->desc.v;
                for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                  LIBXSMM_PRAGMA_SIMD
                  for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
                    LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, kj, ki, ifm2, ofm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock) +=
                        LIBXSMM_VLA_ACCESS(5, input, img, ifm1, ij+kj, ii+ki, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock)
                      * LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi,  ofm2, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
                  }
                }
              }
            }
          }
        }
      } else {
        /* padding is needed */
        for (ij = 0; ij < handle->desc.H; ++ij) {
          for (ii = 0; ii < handle->desc.W; ++ii) {
            for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
              LIBXSMM_VLA_ACCESS(3, input_padded, ij + handle->desc.pad_h, ii + handle->desc.pad_w, ifm2, padded_w, handle->ifmblock) =
                LIBXSMM_VLA_ACCESS(5,  input, img, ifm1, ij, ii, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
            }
          }
        }

        /* now we run a strided vector operation */
        for (oj = 0; oj < handle->ofh; ++oj) {
          ij = oj * handle->desc.u;
          for (kj = 0; kj < handle->desc.R; ++kj) {
            for (ki = 0; ki < handle->desc.S; ++ki) {
              for (oi = 0; oi < handle->ofw; ++oi) {
                ii = oi * handle->desc.v;
                for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                  LIBXSMM_PRAGMA_SIMD
                  for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
                    LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, kj, ki, ifm2, ofm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock) +=
                        LIBXSMM_VLA_ACCESS(3, input_padded,       ij+kj, ii+ki, ifm2, padded_w, handle->ifmblock)
                      * LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi,  ofm2, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
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

