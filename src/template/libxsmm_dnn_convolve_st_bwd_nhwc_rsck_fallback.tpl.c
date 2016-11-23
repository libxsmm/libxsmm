/******************************************************************************
 ** Copyright (c) 2016, Intel Corporation                                     **
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
/* Rajkishore Barik (Intel Corp.), Alexander Heinecke (Intel Corp.)
 ******************************************************************************/

int imgifm1, img, ofm1, ifm1, oj, ij, oi, ii, kj, ki, ifm2, ofm2;
/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
const int work = handle->desc.N * handle->blocksifm;
/* compute chunck size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

element_output_type *const out = ((element_output_type*)handle->output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->blocksofm * handle->ofmblock;
LIBXSMM_VLA_DECL(5, const element_output_type, output, out, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
LIBXSMM_VLA_DECL(5, element_input_type, input, (element_input_type*)handle->input->data, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
LIBXSMM_VLA_DECL(6, const element_filter_type, weight, (element_filter_type*)handle->filter->data, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);

for (imgifm1 = thr_begin; imgifm1 < thr_end; ++imgifm1) {
  img = imgifm1 / handle->blocksifm;
  ifm1 = imgifm1 % handle->blocksifm;
  for (ofm1 = 0; ofm1 < handle->blocksofm; ++ofm1) {
    for( oj = 0; oj < handle->ofh; ++oj) {
      ij = oj * handle->desc.u;
      for (oi = 0; oi < handle->ofw; ++oi) {
        ii = oi * handle->desc.v;
        for (kj = 0; kj < handle->desc.R; ++kj) {
          for (ki = 0; ki < handle->desc.S; ++ki) {
            for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
              for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                LIBXSMM_VLA_ACCESS(5, input, img, ij+kj, ii+ki, ifm1, ifm2, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock) += (element_input_type)(
                  LIBXSMM_VLA_ACCESS(5, output, img, oj, oi, ofm1, ofm2, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock)
                * LIBXSMM_VLA_ACCESS(6, weight, kj, ki, ifm1, ifm2, ofm1, ofm2, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock));
              }
            }
          }
        }
      }
    }
  }
}
