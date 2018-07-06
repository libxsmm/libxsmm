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
/* Rajkishore Barik, Ankush Mandal, Alexander Heinecke (Intel Corp.)
******************************************************************************/

int imgifm1, img, ofm1, ifm1, oj, ij, oi, ii, kj, ki, ifm2, ofm2, ifm1ofm1;
/* computing first logical thread */
const int ltid = tid - start_thread;

/* number of tasks that could be run in parallel */
const int work = handle->desc.N * handle->blocksifm;
/* compute chunk size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* number of tasks for transpose that could be run in parallel */
int transpose_work = handle->blocksifm * handle->blocksofm;
/* compute chunk size */
const int transpose_chunksize = (transpose_work % handle->desc.threads == 0) ? (transpose_work / handle->desc.threads) : ((transpose_work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
const int transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;

/* offset pointer in case of physical padding */
element_output_type *const out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;

/* Weight and transpose_weight tensor declaration */
LIBXSMM_VLA_DECL(6, element_filter_type, wt, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
LIBXSMM_VLA_DECL(6, element_filter_type, tr_wt, (element_filter_type*)handle->scratch1, handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
/* define weight pointer which has the correct format */
element_filter_type* weight_base = 0;

/* padding via stack allocated buffers */
const int padded_w = handle->desc.W + (2 * handle->desc.pad_w);
const int padded_h = handle->desc.H + (2 * handle->desc.pad_h);
const int size_tls1 = padded_h * padded_w * handle->ifmblock;
#if !defined(LIBXSMM_DNN_VLA_TLS1)
element_input_type *const del_input_scratch_padding = (element_input_type*)(((char*)handle->scratch5) +
  ltid * LIBXSMM_UP2(size_tls1 * sizeof(element_input_type), LIBXSMM_CACHELINE));
LIBXSMM_ASSERT(size_tls1 * sizeof(element_input_type) * handle->desc.threads <= handle->max_scratch5_size);
#else
element_input_type del_input_scratch_padding_array[size_tls1];
element_input_type *const del_input_scratch_padding = del_input_scratch_padding_array;
#endif
for ( ii = 0; ii < size_tls1; ++ii ) { del_input_scratch_padding[ii] = (element_input_type)0; }

/* transpose filters, if requested */
if ( (handle->options & LIBXSMM_DNN_CONV_OPTION_BWD_NO_FILTER_TRANSPOSE) > 0 ) {
  weight_base = (element_filter_type*)handle->reg_filter_tr->data;
} else {
  /* lazy barrier init */
  libxsmm_barrier_init(handle->barrier, ltid);

  for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
    ofm1 = ifm1ofm1 / handle->blocksifm;
    ifm1 = ifm1ofm1 % handle->blocksifm;
    for (kj=0; kj < handle->desc.R; kj++) {
      for (ki=0; ki < handle->desc.S; ki++) {
        for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
          for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
            LIBXSMM_VLA_ACCESS(6, tr_wt, ifm1, ofm1, handle->desc.R-1-kj , handle->desc.S-1-ki, ofm2, ifm2, handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock) =
              LIBXSMM_VLA_ACCESS(6, wt, ofm1, ifm1, kj, ki, ifm2, ofm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
          }
        }
      }
    }
  }
  weight_base = (element_filter_type*)handle->scratch1;

  /* wait for transpose to finish */
  libxsmm_barrier_wait(handle->barrier, ltid);
}

{/* open new scope for additional variable declarations (C89) */
LIBXSMM_VLA_DECL(5, element_input_type, del_input, (element_output_type*)handle->grad_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
LIBXSMM_VLA_DECL(3, element_input_type, del_input_padded, del_input_scratch_padding, padded_w, handle->ifmblock);
LIBXSMM_VLA_DECL(5, const element_output_type, output, out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
LIBXSMM_VLA_DECL(6, const element_filter_type, weight, weight_base, handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);

for (imgifm1 = thr_begin; imgifm1 < thr_end; ++imgifm1) {
  img = imgifm1 / handle->blocksifm;
  ifm1 = imgifm1 % handle->blocksifm;

  /* check if we need padding, for now we do physical padding on the fly, however we can play with N parameter of the GEMM */
  /* @TODO: add variant which deals with multiple GEMMS by varying N to deal with padding */
  if ( (handle->desc.pad_h == handle->desc.pad_h_in) && (handle->desc.pad_w == handle->desc.pad_w_in) ) {

    /* reset result buffer to zero when intent is to overwrite when first block
       of input channels should be convoluted */
    if ( ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ) {
      element_input_type* temp_ptr = &(LIBXSMM_VLA_ACCESS(  5, del_input, img, ifm1, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock));
      LIBXSMM_PRAGMA_SIMD
      for (ij = 0; ij < handle->ifhp*handle->ifwp*handle->ifmblock; ij++) {
        temp_ptr[ij] = (element_input_type)0;
      }
    }

    /* run convolution */
    for (ofm1 = 0; ofm1 < handle->blocksofm; ++ofm1) {
      for ( oj = 0; oj < handle->ofh; ++oj) {
        ij = oj * handle->desc.u;
        oi = 0; ii = 0;
        for (kj = 0; kj < handle->desc.R; ++kj) {
          for (ki = 0; ki < handle->desc.S; ++ki) {
            gemm_kernel( &LIBXSMM_VLA_ACCESS(6, weight, ifm1, ofm1, handle->desc.R-1-kj, handle->desc.S-1-ki, 0, 0,        handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                         &LIBXSMM_VLA_ACCESS(5, output,  img, ofm1, oj, oi, 0,           handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                         &LIBXSMM_VLA_ACCESS(5, del_input,  img, ifm1, ij + kj, ii + ki, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock) );
          }
        }
      }
    }

    /* zero rim in case of physical padding.... this code is extremely stupid and crappy as it requires a complicated if... */
    if (handle->desc.pad_h_in > 0 || handle->desc.pad_w_in > 0) {
      for ( ij = 0; ij < handle->ifhp; ij++ ) {
        for ( ii = 0; ii < handle->ifwp; ii++ ) {
          if ( (ij < handle->desc.pad_h_in) || (ij >= (handle->desc.H+handle->desc.pad_h_in)) ||
               (ii < handle->desc.pad_w_in) || (ii >= (handle->desc.W+handle->desc.pad_w_in)) ) {
            for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
              LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, ii, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock) = (element_input_type)0;
            }
          }
        }
      }
    }
  } else {
    /* reset result buffer to zero when intent is to overwrite when first block
       of input channels should be convoluted */
    if ( ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ) {
      LIBXSMM_PRAGMA_SIMD
      for (ij = 0; ij < size_tls1; ++ij) {
        del_input_scratch_padding[ij] = (element_output_type)0;
      }
    } else {
      for (ij = 0; ij < handle->desc.H; ij++) {
        for (ii = 0; ii < handle->desc.W; ii++) {
          LIBXSMM_PRAGMA_SIMD
          for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
              LIBXSMM_VLA_ACCESS(3, del_input_padded, ij + handle->desc.pad_h, ii + handle->desc.pad_w, ifm2, padded_w, handle->ifmblock) =
                LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, ii, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          }
        }
      }
    }

    /* run convolution */
    for (ofm1 = 0; ofm1 < handle->blocksofm; ++ofm1) {
      for ( oj = 0; oj < handle->ofh; ++oj) {
        ij = oj * handle->desc.u;
        oi = 0; ii = 0;
        for (kj = 0; kj < handle->desc.R; ++kj) {
          for (ki = 0; ki < handle->desc.S; ++ki) {
            gemm_kernel( &LIBXSMM_VLA_ACCESS(6, weight, ifm1, ofm1, handle->desc.R-1-kj, handle->desc.S-1-ki, 0, 0,        handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                         &LIBXSMM_VLA_ACCESS(5, output,  img, ofm1, oj, oi, 0,           handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                         &LIBXSMM_VLA_ACCESS(3, del_input_padded, ij + kj, ii + ki, 0, padded_w, handle->ifmblock) );
          }
        }
      }
    }

    /* input padding copy back */
    for (ij = 0; ij < handle->desc.H; ij++) {
      for (ii = 0; ii < handle->desc.W; ii++) {
        LIBXSMM_PRAGMA_SIMD
        for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
          LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, ii, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock) =
            LIBXSMM_VLA_ACCESS(3, del_input_padded, ij + handle->desc.pad_h, ii + handle->desc.pad_w, ifm2, padded_w, handle->ifmblock);
        }
      }
    }
  }
} /* end of imgifm1 loop */

} /* end of new scope for additional variable declarations (C89) */

