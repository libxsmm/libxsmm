
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
/* Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/

int imgofm1, img, ofm1, ifm1, oj, ij, oi, ii, kj, ki, ifm2, ofm2;
/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
const int work = handle->desc.N * handle->blocksofm;
/* compute chunck size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* regular/high precision */
element_output_type* out = 0;
/* low precision */
element_input_type* out_lp = 0;

/* select pointer based on precision */
if (handle->datatype != handle->datatype_itm) {
  out = ((element_output_type*)handle->scratch6) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
  out_lp = ((element_input_type*)handle->reg_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
} else {
  out = ((element_output_type*)handle->reg_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
  out_lp = 0;
}

{ /* open new scope for additional variable declarations (C89) */
  LIBXSMM_VLA_DECL(5, element_output_type, output, out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
  LIBXSMM_VLA_DECL(5, element_input_type, output_lp, out_lp, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
  LIBXSMM_VLA_DECL(5, const element_input_type, input, (element_input_type*)handle->reg_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
  LIBXSMM_VLA_DECL(6, const element_filter_type, weight, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
#if defined(INPUT_PADDING)
  /* Variables and initializations related to padding */
  const int padded_h = handle->ifhp + 2 * handle->desc.pad_h;
  const int padded_w = handle->ifwp + 2 * handle->desc.pad_w;
  LIBXSMM_VLA_DECL(3, element_input_type, input_buffer, ((element_input_type*)handle->scratch5) + ltid * padded_h * padded_w * handle->ifmblock, padded_w, handle->ifmblock);
  /* Reset input padding buffer to zero (in case it is not set to zero due to fwd/bwd computations) */
  memset(&LIBXSMM_VLA_ACCESS(3, input_buffer, 0, 0, 0, padded_w, handle->ifmblock), 0, padded_w * padded_h * handle->ifmblock * sizeof(element_input_type));
#endif


  /* perform convolution */
  for (imgofm1 = thr_begin; imgofm1 < thr_end; ++imgofm1) {
    img = imgofm1 / handle->blocksofm;
    ofm1 = imgofm1 % handle->blocksofm;
    /* up-convert */
    if (handle->datatype != handle->datatype_itm) {
      for (oj = 0; oj < handle->ofh; ++oj) {
        for (oi = 0; oi < handle->ofw; ++oi) {
          for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
            LIBXSMM_VLA_ACCESS(  5, output, img, ofm1, oj, oi, ofm2, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock) = (element_output_type)
              (LIBXSMM_VLA_ACCESS(  5, output_lp, img, ofm1, oj, oi, ofm2, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
          }
        }
      }
    }
    for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
#if defined(INPUT_PADDING)
      for (oj = 0; oj < handle->ifhp; ++oj) {
        for (oi = 0; oi < handle->ifwp; ++oi) {
          for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
            LIBXSMM_VLA_ACCESS(3, input_buffer, oj + handle->desc.pad_h, oi + handle->desc.pad_w, ifm2, padded_w, handle->ifmblock) =
              LIBXSMM_VLA_ACCESS(5,  input, img, ifm1, oj, oi, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          }
        }
      }
#endif
      /* reset result buffer to zero when intent is to overwrite when first block
         of input channels should be convoluted */
      if ( (ifm1 == 0) && ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ) {
        element_output_type* temp_ptr = &(LIBXSMM_VLA_ACCESS(  5, output, img, ofm1, 0, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
        LIBXSMM_PRAGMA_SIMD
        for (oj = 0; oj < handle->ofhp*handle->ofwp*handle->ofmblock; oj++) {
          temp_ptr[oj] = (element_output_type)0;
        }
      }
      for (oj = 0; oj < handle->ofh; ++oj) {
        ij = oj * handle->desc.u;
        for (oi = 0; oi < handle->ofw; ++oi) {
          ii = oi * handle->desc.v;
          for (kj = 0; kj < handle->desc.R; ++kj) {
            for (ki = 0; ki< handle->desc.S; ++ki) {
              for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {

                  LIBXSMM_VLA_ACCESS(  5, output, img, ofm1, oj, oi, ofm2, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock) += (element_output_type)(
#if defined(INPUT_PADDING)
                    LIBXSMM_VLA_ACCESS(3, input_buffer, ij + kj, ii + ki, ifm2, padded_w, handle->ifmblock)
#else
                    LIBXSMM_VLA_ACCESS(5,  input, img, ifm1, ij + kj, ii + ki, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock)
#endif
                  * LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, kj, ki, ifm2, ofm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock));
                }
              }
            }
          }
        }
      }
    }
    /* down-convert */
    if (handle->datatype != handle->datatype_itm) {
      for (oj = 0; oj < handle->ofh; ++oj) {
        for (oi = 0; oi < handle->ofw; ++oi) {
          for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
            LIBXSMM_VLA_ACCESS(  5, output_lp, img, ofm1, oj, oi, ofm2, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock) = (element_input_type)
              (LIBXSMM_VLA_ACCESS(  5, output, img, ofm1, oj, oi, ofm2, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
          }
        }
      }
    }
  }
}

