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
/* Rajkishore Barik, Ankush Mandal, Alexander Heinecke (Intel Corp.)
******************************************************************************/

int imgifm1, img, ofm1, ifm1, oj, ij, oi, ii, kj, ki, ifm2, ofm2, ifmlp, ofmlp;
/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
const int work = handle->desc.N * handle->blocksifm;
/* compute chunck size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* Input tensor declaration */
/* regular/high precision */
element_input_type* del_in = 0;
/* low precision */
element_output_type* del_in_lp = 0;
/* select pointer based on precision */
if (handle->datatype != handle->datatype_itm) {
  del_in = ((element_input_type*)handle->scratch7);
  del_in_lp = ((element_output_type*)handle->grad_input->data);
} else {
  del_in = ((element_input_type*)handle->grad_input->data);
  del_in_lp = 0;
}
{/* open new scope for additional variable declarations (C89) */
LIBXSMM_VLA_DECL(5, element_input_type, del_input, del_in, handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock);
LIBXSMM_VLA_DECL(6, element_output_type, del_input_lp, del_in_lp, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
element_output_type *const out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock * handle->fm_lp_block;
LIBXSMM_VLA_DECL(6, const element_output_type, output, out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);
LIBXSMM_VLA_DECL(7, const element_filter_type, weight, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);

#if defined(INPUT_PADDING)
/* Variables and initializations related to padding */
const int padded_h = handle->ifhp + 2 * handle->desc.pad_h;
const int padded_w = handle->ifwp + 2 * handle->desc.pad_w;
LIBXSMM_VLA_DECL(3, element_input_type, input_buffer, ((element_input_type*)handle->scratch5) + ltid * padded_h * padded_w * handle->ifmblock, padded_w, handle->ifmblock);
/* Reset input padding buffer to zero (in case it is not set to zero due to fwd/bwd computations) */
memset(&LIBXSMM_VLA_ACCESS(3, input_buffer, 0, 0, 0, padded_w, handle->ifmblock), 0, padded_w * padded_h * handle->ifmblock * sizeof(element_input_type));
#endif

for (imgifm1 = thr_begin; imgifm1 < thr_end; ++imgifm1) {
  img = imgifm1 / handle->blocksifm;
  ifm1 = imgifm1 % handle->blocksifm;

  for (ifmlp = 0; ifmlp < handle->fm_lp_block; ++ifmlp) {
    /* upconvert for low precision */
    if (handle->datatype != handle->datatype_itm) {
      for (ij = 0; ij < handle->ifhp; ++ij) {
        for (ii = 0; ii < handle->ifwp; ++ii) {
          for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
            LIBXSMM_VLA_ACCESS(5, del_input, img, (ifm1 * handle->fm_lp_block) + ifmlp, ij, ii, ifm2, handle->blocksifm*handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock) = (element_input_type)(LIBXSMM_VLA_ACCESS(6, del_input_lp, img, ifm1, ij, ii, ((handle->ifmblock/handle->fm_lp_block)*ifmlp)+(ifm2/handle->fm_lp_block), (ifm2%handle->fm_lp_block), handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block));
          }
        }
      }
    }/* end of upconvert for low precision */

    /* input padding copy */
#if defined(INPUT_PADDING)
    memset(&LIBXSMM_VLA_ACCESS(3, input_buffer, 0, 0, 0, padded_w, handle->ifmblock), 0, padded_w * padded_h * sizeof(element_input_type));
    for (ij = 0; ij < handle->ifhp; ij++) {
      for (ii = 0; ii < handle->ifwp; ii++) {
        for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
          LIBXSMM_VLA_ACCESS(3, input_buffer, ij + handle->desc.pad_h, ii + handle->desc.pad_w, ifm2, padded_w, handle->ifmblock) =
            LIBXSMM_VLA_ACCESS(5, del_input, img, (ifm1 * handle->fm_lp_block) + ifmlp, ij, ii, ifm2, handle->blocksifm*handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock);
        }
      }
    }
#endif

    for (ofm1 = 0; ofm1 < handle->blocksofm; ++ofm1) {
      for ( oj = 0; oj < handle->ofh; ++oj) {
        ij = oj * handle->desc.u;
        for (oi = 0; oi < handle->ofw; ++oi) {
          ii = oi * handle->desc.v;
          for (kj = 0; kj < handle->desc.R; ++kj) {
            for (ki = 0; ki < handle->desc.S; ++ki) {
              for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
                for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                  for (ofmlp = 0; ofmlp < handle->fm_lp_block; ++ofmlp) {
#if defined(INPUT_PADDING)
                    LIBXSMM_VLA_ACCESS(3, input_buffer, ij+kj, ii+ki, ifm2, padded_w, handle->ifmblock)
#else
                    LIBXSMM_VLA_ACCESS(5, del_input, img, (ifm1 * handle->fm_lp_block) + ifmlp, ij+kj, ii+ki, ifm2, handle->blocksifm*handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock)
#endif
                    += (element_input_type)(
                      LIBXSMM_VLA_ACCESS(6, output, img, ofm1, oj, oi, ofm2, ofmlp, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block)
                    * LIBXSMM_VLA_ACCESS(7, weight, (ofm1 * handle->fm_lp_block) + ofmlp, ifm1, kj, ki, ifm2, ofm2, ifmlp, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block));
                  }
                }
              }
            }
          }
        }
      }
    }

    /* input padding copy back */
#if defined(INPUT_PADDING)
    for (ij = 0; ij < handle->ifhp; ij++) {
      for (ii = 0; ii < handle->ifwp; ii++) {
        for (ifm2 = 0; ifm2 < handle->ifmblock; ifm2++) {
          LIBXSMM_VLA_ACCESS(5, del_input, img, (ifm1 * handle->fm_lp_block) + ifmlp, ij, ii, ifm2, handle->blocksifm*handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock) =
            LIBXSMM_VLA_ACCESS(3, input_buffer, ij + handle->desc.pad_h, ii + handle->desc.pad_w, ifm2, padded_w, handle->ifmblock);
        }
      }
    }
#else
#include "libxsmm_dnn_zero_rim_st_input_custom_fallback.tpl.c"
#endif

    /* down-convert for low precision */
    if (handle->datatype != handle->datatype_itm) {
      for (ij = 0; ij < handle->ifhp; ++ij) {
        for (ii = 0; ii < handle->ifwp; ++ii) {
          for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
            LIBXSMM_VLA_ACCESS(6, del_input_lp, img, ifm1, ij, ii, ((handle->ifmblock/handle->fm_lp_block)*ifmlp)+(ifm2/handle->fm_lp_block), (ifm2%handle->fm_lp_block), handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block) = (element_output_type)(LIBXSMM_VLA_ACCESS(5, del_input, img, (ifm1 * handle->fm_lp_block) + ifmlp, ij, ii, ifm2, handle->blocksifm*handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock));
          }
        }
      }
    } /* end of downconvert for low precision */

  } /* end of ifmlp loop */
} /* end of imgifm1 loop */

} /* end of new scope for additional variable declarations (C89) */
