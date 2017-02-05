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

element_output_type *const out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->blocksofm * handle->ofmblock;
LIBXSMM_VLA_DECL(5, const element_output_type, output, out, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
LIBXSMM_VLA_DECL(5, const element_input_type, input, (element_input_type*)handle->reg_input->data, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
LIBXSMM_VLA_DECL(6, element_filter_type, weight, (element_filter_type*)handle->grad_filter->data, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);

#if defined(INPUT_PADDING)
/* Variables and initializations related to padding */
const int padded_h = handle->ifhp + 2 * handle->desc.pad_h;
const int padded_w = handle->ifwp + 2 * handle->desc.pad_w;
LIBXSMM_VLA_DECL(4, element_input_type, input_buffer, ((element_input_type*)handle->scratch5) + ltid * padded_h * padded_w * handle->blocksifm * handle->ifmblock, padded_w, handle->blocksifm, handle->ifmblock);
/* Reset input padding buffer to zero (in case it is not set to zero due to fwd/bwd computations) */
memset(&LIBXSMM_VLA_ACCESS(4, input_buffer, 0, 0, 0, 0, padded_w, handle->blocksifm, handle->ifmblock), 0, padded_w * padded_h * handle->ifmblock * handle->blocksifm * sizeof(element_input_type));
#endif

for (ofm1ifm1 = thr_begin; ofm1ifm1 < thr_end; ++ofm1ifm1) {
  ofm1 = ofm1ifm1 / handle->blocksifm;
  ifm1 = ofm1ifm1 % handle->blocksifm;
  for (img = 0; img < handle->desc.N; ++img) {

#if defined(INPUT_PADDING)
    for (oj = 0; oj < handle->ifhp; ++oj) {
      for (oi = 0; oi < handle->ifwp; ++oi) {
        for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
          LIBXSMM_VLA_ACCESS(4, input_buffer, oj + handle->desc.pad_h , oi + handle->desc.pad_w, ifm1, ifm2, padded_w, handle->blocksifm, handle->ifmblock) =
          (element_input_type) LIBXSMM_VLA_ACCESS(5, input, img, oj, oi, ifm1, ifm2, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
        }
      }
    }
#endif

    for (oj = 0; oj < handle->ofh; ++oj) {
      ij = oj * handle->desc.u;
      for (oi = 0; oi < handle->ofw; ++oi) {
        ii = oi * handle->desc.v;
        for (kj = 0; kj < handle->desc.R; ++kj) {
          for (ki = 0; ki < handle->desc.S; ++ki) {
            for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
              for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
                LIBXSMM_VLA_ACCESS(6, weight, kj, ki, ifm1, ifm2, ofm1, ofm2, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock) += (element_filter_type)(
#if defined(INPUT_PADDING)
                    LIBXSMM_VLA_ACCESS(4,  input_buffer, ij + kj + handle->desc.pad_h, ii + ki + handle->desc.pad_w, ifm1, ifm2, padded_w, handle->blocksifm, handle->ifmblock)
#else
                    LIBXSMM_VLA_ACCESS(5,  input, img, ij + kj, ii + ki, ifm1, ifm2, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock)
#endif
                  * LIBXSMM_VLA_ACCESS(5, output, img, oj, oi, ofm1, ofm2, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock));
              }
            }
          }
        }
      }
    }
  }
}

