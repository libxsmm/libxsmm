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
/* Alexander Heinecke, Evangelos Georganas, Hans Pabst (Intel Corp.)
 ******************************************************************************/

/* loop counters */
int img1, ofm1, ifm1, oj, oi, ij, ii, kj, ki, i;
const int blocksofm = handle->blocksofm, ofh = handle->ofh, ofw = handle->ofw, u = handle->desc.u, v = handle->desc.v, pad_h = handle->desc.pad_h, pad_w = handle->desc.pad_w, blocksifm = handle->blocksifm, R = handle->desc.R, S = handle->desc.S, ifhp = handle->ifhp, ifwp = handle->ifwp, nbImg = handle->nbImg, ifmblock = handle->ifmblock, ofhp = handle->ofhp, ofwp = handle->ofwp, ofmblock = handle->ofmblock, nBImg = handle->nBImg, ifh = handle->desc.H, ifw = handle->desc.W;
const int ltid = tid-start_thread;
const int work = nBImg * blocksofm;
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
LIBXSMM_VLA_DECL(6, element_output_type, output_t, ((element_output_type*) handle->reg_output->data) + (handle->desc.pad_w_out * handle->ofwp + handle->desc.pad_h_out), nBImg, ofhp, ofwp, nbImg, ofmblock);
LIBXSMM_VLA_DECL(6, const element_input_type,  input_t, ((const element_input_type*) handle->reg_input->data) + (handle->desc.pad_w_in * handle->ifwp + handle->desc.pad_h_in), nBImg, ifhp, ifwp, nbImg, ifmblock);
LIBXSMM_VLA_DECL(6, const element_filter_type, filter_t, (const element_filter_type*) handle->reg_filter->data, blocksifm, R, S, ifmblock, ofmblock);
libxsmm_mmfunction sixteen = (libxsmm_mmfunction) handle->code_fwd[0].smm;

for (i = thr_begin; i < thr_end; ++i) {
  img1 = i/blocksofm;
  ofm1 = i%blocksofm;
  for (ifm1 = 0; ifm1 < blocksifm; ++ifm1) {
    for (oj = 0; oj < ofh; ++oj) {
      for (oi = 0; oi < ofw; ++oi) {
        ij = oj * u - pad_h;
        ii = oi * v - pad_w;
        for (kj = 0; kj < R; ++kj) {
          if(ij+kj < 0 || ij+kj >= ifh) continue;
          for (ki = 0; ki < S; ++ki) {
            if(ii+ki < 0 || ii+ki >= ifw) continue;
            sixteen( &LIBXSMM_VLA_ACCESS(6, filter_t, ofm1, ifm1, kj,      ki,      0, 0, blocksifm, R, S, ifmblock, ofmblock) ,
                    &LIBXSMM_VLA_ACCESS(6,  input_t, ifm1, img1, ij + kj, ii + ki, 0, 0, nBImg, ifhp, ifwp, nbImg, ifmblock) ,
                    &LIBXSMM_VLA_ACCESS(6, output_t, ofm1, img1, oj,      oi,      0, 0, nBImg, ofhp, ofwp, nbImg, ofmblock) );
          }
        }
      }
    }
  }
}
