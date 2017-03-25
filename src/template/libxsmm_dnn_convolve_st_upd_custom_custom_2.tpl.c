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
int img1, ofm1, ifm1, img2, ifm2, oj, oi, ij, ii, kj, ki, i;
const int ifh = handle->desc.H;
const int ifw = handle->desc.W;
const int ltid = tid-start_thread;
/* number of tasks that could be run in parallel */
const int work = handle->blocksofm * handle->blocksifm;
/* compute chunck size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
const int transpose_work = handle->nBImg * handle->blocksifm * handle->ifhp * handle->ifwp;
const int transpose_chunksize = (transpose_work % handle->desc.threads == 0) ? (transpose_work / handle->desc.threads) : (transpose_work / handle->desc.threads) + 1;
const int trans_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
const int trans_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;

LIBXSMM_VLA_DECL(6, const element_output_type, output_t, ((const element_output_type*)handle->reg_output->data) + (handle->desc.pad_w_out * handle->ofwp + handle->desc.pad_h_out), handle->nBImg, handle->ofhp, handle->ofwp, handle->nbImg, handle->ofmblock);
LIBXSMM_VLA_DECL(6, const element_input_type,  input_t, ((const element_input_type*)handle->reg_input->data) + (handle->desc.pad_w_in * handle->ifwp + handle->desc.pad_h_in), handle->nBImg, handle->ifhp, handle->ifwp, handle->nbImg, handle->ifmblock);
LIBXSMM_VLA_DECL(6,  element_input_type,  tr_input_t, ((element_input_type*)handle->scratch3) + (handle->desc.pad_w_in * handle->ifwp + handle->desc.pad_h_in), handle->nBImg, handle->ifhp, handle->ifwp, handle->ifmblock, handle->nbImg);
LIBXSMM_VLA_DECL(6, element_filter_type, filter_t, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
libxsmm_mmfunction sixteen = (libxsmm_mmfunction) handle->code_upd[0].smm;

/* Transpose in parallel the input  */
libxsmm_barrier_init(handle->barrier, ltid);
for (i = trans_thr_begin; i < trans_thr_end; ++i) {
  img1 = i/(handle->blocksifm * handle->ifhp * handle->ifwp);
  ifm1 = (i%(handle->blocksifm * handle->ifhp * handle->ifwp))/(handle->ifhp * handle->ifwp);
  ij = ((i%(handle->blocksifm * handle->ifhp * handle->ifwp))%(handle->ifhp * handle->ifwp))/ handle->ifwp;
  ii = ((i%(handle->blocksifm * handle->ifhp * handle->ifwp))%(handle->ifhp * handle->ifwp))% handle->ifwp;
  for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
    for (img2 = 0; img2 < handle->nbImg; ++img2) {
      LIBXSMM_VLA_ACCESS(6,  tr_input_t, ifm1, img1, ij, ii, ifm2, img2, handle->nBImg, handle->ifhp, handle->ifwp, handle->ifmblock, handle->nbImg) =
      LIBXSMM_VLA_ACCESS(6,   input_t, ifm1, img1, ij, ii, img2, ifm2, handle->nBImg, handle->ifhp, handle->ifwp, handle->nbImg, handle->ifmblock);
    }
  }
}
libxsmm_barrier_wait(handle->barrier, ltid);

for (i = thr_begin; i < thr_end; ++i) {
  ofm1 = i/(handle->blocksifm);
  ifm1 = i%(handle->blocksifm);
  for (img1 = 0; img1 < handle->nBImg; ++img1) {
    for (oj = 0; oj < handle->ofh; ++oj) {
      for (oi = 0; oi < handle->ofw; ++oi) {
        ij = oj * handle->desc.u - handle->desc.pad_h;
        ii = oi * handle->desc.v - handle->desc.pad_w;
        for (kj = 0; kj < handle->desc.R; ++kj) {
          if(ij+kj < 0 || ij+kj >= ifh) continue;
          for (ki = 0; ki < handle->desc.S; ++ki) {
            if(ii+ki < 0 || ii+ki >= ifw) continue;
            sixteen( &LIBXSMM_VLA_ACCESS(6,   output_t, ofm1, img1, oj,      oi,      0, 0, handle->nBImg, handle->ofhp, handle->ofwp, handle->nbImg, handle->ofmblock) /* A */,
                    &LIBXSMM_VLA_ACCESS(6, tr_input_t, ifm1, img1, ij + kj, ii + ki, 0, 0, handle->nBImg, handle->ifhp, handle->ifwp, handle->ifmblock, handle->nbImg) /* B */,
                    &LIBXSMM_VLA_ACCESS(6,   filter_t, ofm1, ifm1, kj,      ki,      0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock) /* C */  );
          }
        }
      }
    }
  }
}
