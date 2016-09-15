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
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

  const element_input_type *const inp = ((const element_input_type*)handle->input->data);
  const element_filter_type *const wtp = ((element_filter_type*)handle->filter->data);
  element_output_type *const outp = ((element_output_type*)handle->output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
  int imgofm1, img, ofm1, ifm1, oj, ij, oi, ii, kj, ki, ifm2, ofm2;
  /* computing first logical thread */
  const int ltid = tid - start_thread;
  /* number of tasks that could be run in parallel */
  const int work = handle->desc.N * handle->blocksofm;
  /* compute chunck size */
  const int chunksize = (work % num_threads == 0) ? (work / num_threads) : ((work / num_threads) + 1);
  /* compute thr_begin and thr_end */
  const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
  const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
#if defined(LIBXSMM_VLA)
  typedef element_input_type (*LIBXSMM_RESTRICT input_data_type)[handle->blocksifm][handle->ifhp][handle->ifwp][handle->ifmblock];
  typedef element_filter_type (*LIBXSMM_RESTRICT weight_data_type)[handle->blocksifm][handle->desc.R][handle->desc.S][handle->ifmblock][handle->ofmblock];
  typedef element_output_type (*LIBXSMM_RESTRICT output_data_type)[handle->blocksofm][handle->ofhp][handle->ofwp][handle->ofmblock];
  const input_data_type input = (input_data_type)inp;
  const weight_data_type weight = (weight_data_type)wtp;
  const output_data_type output = (output_data_type)outp;
#else
  const element_input_type *LIBXSMM_RESTRICT input = (const element_input_type*)inp;
  const element_filter_type *LIBXSMM_RESTRICT weight = (const element_filter_type*)wtp;
  element_output_type *LIBXSMM_RESTRICT output = (element_output_type*)outp;
  unsigned int ishape[5], wshape[6], oshape[5];
  unsigned int indexi[5], indexw[6], indexo[5];
  /* arrays must be initialized separately to avoid warning about values not computable at init.-time */
  ishape[0] = handle->ifmblock; ishape[1] = handle->ifwp; ishape[2] = handle->ifhp; ishape[3] = handle->blocksifm; ishape[4] = (thr_end - thr_begin) / handle->blocksofm;
  wshape[0] = handle->ofmblock; wshape[1] = handle->ifmblock; wshape[2] = handle->desc.S; wshape[3] = handle->desc.R; wshape[4] = handle->blocksifm; wshape[5] = (thr_end - thr_begin) % handle->blocksofm;
  oshape[0] = handle->ofmblock; oshape[1] = handle->ofwp; oshape[2] = handle->ofhp; oshape[3] = handle->blocksofm; oshape[4] = ishape[4];
#endif
  for (imgofm1 = thr_begin; imgofm1 < thr_end; ++imgofm1) {
    img = imgofm1 / handle->blocksofm;
    ofm1 = imgofm1 % handle->blocksofm;
    for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
      for (oj = 0; oj < handle->ofh; ++oj) {
        ij = oj * handle->desc.u;
        for (oi = 0; oi < handle->ofw; ++oi) {
          ii = oi * handle->desc.v;
          for (kj = 0; kj < handle->desc.R; ++kj) {
            for (ki = 0; ki< handle->desc.S; ++ki) {
              for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
#if defined(LIBXSMM_VLA)
                  output[img][ofm1][oj][oi][ofm2] += (element_output_type)(input[img][ifm1][ij+kj][ii+ki][ifm2] * weight[ofm1][ifm1][kj][ki][ifm2][ofm2]);
#else /* index arrays must be initialized separately to avoid warning about values not computable at init.-time */
                  size_t i, w, o;
                  indexi[0] = ifm2; indexi[1] = ii + ki; indexi[2] = ij + kj; indexi[3] = ifm1; indexi[4] = img;
                  indexw[0] = ofm2; indexw[1] = ifm2; indexw[2] = ki; indexw[3] = kj; indexw[4] = ifm1; indexw[5] = ofm1;
                  indexo[0] = ofm2; indexo[1] = oi; indexo[2] = oj; indexo[3] = ofm1; indexo[4] = img;
                  LIBXSMM_CALC_INDEX1(size_t, i, 5, indexi, ishape);
                  LIBXSMM_CALC_INDEX1(size_t, w, 6, indexw, wshape);
                  LIBXSMM_CALC_INDEX1(size_t, o, 5, indexo, oshape);
                  output[o] += (element_output_type)(input[i] * weight[w]);
#endif
                }
              }
            }
          }
        }
      }
    }
  }
