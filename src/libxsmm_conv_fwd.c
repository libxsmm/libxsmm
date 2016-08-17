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
/* Alexander Heinecke (Intel Corp.), Hans Pabst (Intel Corp.)
******************************************************************************/
#include "libxsmm_conv_fwd.h"

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_convolve_st_fwd_fp32_fallback(libxsmm_conv_handle* handle, int start_thread, int tid, int num_threads)
{
  typedef float element_type;
  const element_type *const inp = ((const element_type*)handle->input->data), *const wtp = ((element_type*)handle->filter->data);
  element_type *const outp = ((element_type*)handle->output->data) + (handle->desc.pad_h * handle->ofwp + handle->desc.pad_w) * handle->ofmblock;
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
  typedef element_type (*LIBXSMM_RESTRICT input_data_type)[handle->blocksifm][handle->ifhp][handle->ifwp][handle->ifmblock];
  typedef element_type (*LIBXSMM_RESTRICT weight_data_type)[handle->blocksifm][handle->desc.R][handle->desc.S][handle->ifmblock][handle->ofmblock];
  typedef element_type (*LIBXSMM_RESTRICT output_data_type)[handle->blocksofm][handle->ofhp][handle->ofwp][handle->ofmblock];
  const input_data_type input = (input_data_type)inp;
  const weight_data_type weight = (weight_data_type)wtp;
  const output_data_type output = (output_data_type)outp;
#else
  const element_type *LIBXSMM_RESTRICT input = (const element_type*)inp;
  const element_type *LIBXSMM_RESTRICT weight = (const element_type*)wtp;
  element_type *LIBXSMM_RESTRICT output = (element_type*)outp;
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
                  output[img][ofm1][oj][oi][ofm2] += input[img][ifm1][ij+kj][ii+ki][ifm2] * weight[ofm1][ifm1][kj][ki][ifm2][ofm2];
#else /* index arrays must be initialized separately to avoid warning about values not computable at init.-time */
                  size_t i, w, o;
                  indexi[0] = ifm2; indexi[1] = ii + ki; indexi[2] = ij + kj; indexi[3] = ifm1; indexi[4] = img;
                  indexw[0] = ofm2; indexw[1] = ifm2; indexw[2] = ki; indexw[3] = kj; indexw[4] = ifm1; indexw[5] = ofm1;
                  indexo[0] = ofm2; indexo[1] = oi; indexo[2] = oj; indexo[3] = ofm1; indexo[4] = img;
                  LIBXSMM_CALC_INDEX1(size_t, i, 5, indexi, ishape);
                  LIBXSMM_CALC_INDEX1(size_t, w, 6, indexw, wshape);
                  LIBXSMM_CALC_INDEX1(size_t, o, 5, indexo, oshape);
                  output[o] += input[i] * weight[w];
#endif
                }
              }
            }
          }
        }
      }
    }
  }
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_convolve_st_fwd_fp32_opt(libxsmm_conv_handle* handle, int start_thread, int tid, int num_threads)
{
  typedef float element_type;
  const element_type *const inp = ((const element_type*)handle->input->data), *const wtp = ((const element_type*)handle->filter->data);
  element_type *const outp = ((element_type*)handle->output->data) + (handle->desc.pad_h * handle->ofwp + handle->desc.pad_w) * handle->ofmblock;
  int imgofm1, img, ofm1, ifm1, oj, ij, oi, ii;
  /* computing first logical thread */
  const int ltid = tid-start_thread;
  /* number of tasks that could be run in parallel */
  const int work = handle->desc.N*handle->blocksofm;
  /* compute chunck size */
  const int chunksize = (work % num_threads == 0) ? (work / num_threads) : (work / num_threads) + 1;
  /* compute thr_begin and thr_end */
  const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
  const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
  libxsmm_sconvfunction jitted_sconv_fp_noweight_pf = handle->code_fwd[1].sconv;
  libxsmm_sconvfunction jitted_sconv_fp_weight_pf = handle->code_fwd[2].sconv;
  /*libxsmm_sconvfunction jitted_sconv_fp_weightnooutput_pf = handle->code_fwd[3].sconv;*/
#if defined(LIBXSMM_CONV_NO_PREFETCH)
  libxsmm_sconvfunction jitted_sconv_fp_no_pf = handle->code_fwd[0].sconv;
#endif
  const element_type *l_input, *l_wt;
  element_type* l_output;
#if defined(LIBXSMM_VLA)
  typedef element_type (*LIBXSMM_RESTRICT input_data_type)[handle->blocksifm][handle->ifhp][handle->ifwp][handle->ifmblock];
  typedef element_type (*LIBXSMM_RESTRICT weight_data_type)[handle->blocksifm][handle->desc.R][handle->desc.S][handle->ifmblock][handle->ofmblock];
  typedef element_type (*LIBXSMM_RESTRICT output_data_type)[handle->blocksofm][handle->ofhp][handle->ofwp][handle->ofmblock];
  const input_data_type input = (input_data_type)inp;
  const weight_data_type weight = (weight_data_type)wtp;
  const output_data_type output = (output_data_type)outp;
#else
  const element_type *LIBXSMM_RESTRICT input = (const element_type*)inp;
  const element_type *LIBXSMM_RESTRICT weight = (const element_type*)wtp;
  element_type *LIBXSMM_RESTRICT output = (element_type*)outp;
  unsigned int ishape[5], wshape[6], oshape[5];
  unsigned int indexi[5], indexw[6], indexo[5];
  /* arrays must be initialized separately to avoid warning about values not computable at init.-time */
  ishape[0] = handle->ifmblock; ishape[1] = handle->ifwp; ishape[2] = handle->ifhp; ishape[3] = handle->blocksifm; ishape[4] = (thr_end - thr_begin) / handle->blocksofm;
  wshape[0] = handle->ofmblock; wshape[1] = handle->ifmblock; wshape[2] = handle->desc.S; wshape[3] = handle->desc.R; wshape[4] = handle->blocksifm; wshape[5] = (thr_end - thr_begin) % handle->blocksofm;
  oshape[0] = handle->ofmblock; oshape[1] = handle->ofwp; oshape[2] = handle->ofhp; oshape[3] = handle->blocksofm; oshape[4] = ishape[4];
#endif
  for (imgofm1 = thr_begin; imgofm1 < thr_end; ++imgofm1) {
    img = imgofm1/handle->blocksofm;
    ofm1 = imgofm1%handle->blocksofm;
    for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
      for (oj = 0; oj < handle->ofh; oj += handle->fwd_ofh_rb) {
        ij = oj * handle->desc.u;
        for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
          ii = oi * handle->desc.v;
#if defined(LIBXSMM_VLA)
          l_input = &(input[img][ifm1][ij][ii][0]);
          l_wt = &(weight[ofm1][ifm1][0][0][0][0]);
          l_output = &(output[img][ofm1][oj][oi][0]);
#else /* index arrays must be initialized separately to avoid warning about values not computable at init.-time */
          indexi[0] = 0; indexi[1] = ii; indexi[2] = ij; indexi[3] = ifm1; indexi[4] = img;
          indexw[0] = 0; indexw[1] = 0; indexw[2] = 0; indexw[3] = 0; indexw[4] = ifm1; indexw[5] = ofm1;
          indexo[0] = 0; indexo[1] = oi; indexo[2] = oj; indexo[3] = ofm1; indexo[4] = img;
          {
            size_t i, w, o;
            LIBXSMM_CALC_INDEX1(size_t, i, 5, indexi, ishape);
            LIBXSMM_CALC_INDEX1(size_t, w, 6, indexw, wshape);
            LIBXSMM_CALC_INDEX1(size_t, o, 5, indexo, oshape);
            l_input = input + i;
            l_wt = weight + w;
            l_output = output + o;
          }
#endif
#if !defined(LIBXSMM_CONV_NO_PREFETCH)
          /* check we are not at the end */
          if (oj < handle->ofh-handle->fwd_ofh_rb) {
# if defined(LIBXSMM_VLA)
            jitted_sconv_fp_noweight_pf(l_input, l_wt, l_output,
              &(input[img][ifm1][(oj+handle->fwd_ofh_rb)*handle->desc.u][ii][0]), NULL, &(output[img][ofm1][oj+handle->fwd_ofh_rb][oi][0]));
# else
            size_t pi, po;
            indexi[0] = 0; indexi[1] = ii; indexi[2] = (oj + handle->fwd_ofh_rb) * handle->desc.u; indexi[3] = ifm1; indexi[4] = img;
            indexo[0] = 0; indexo[1] = oi; indexo[2] = oj + handle->fwd_ofh_rb; indexo[3] = ofm1; indexo[4] = img;
            LIBXSMM_CALC_INDEX1(size_t, pi, 5, indexi, ishape);
            LIBXSMM_CALC_INDEX1(size_t, po, 5, indexo, oshape);
            jitted_sconv_fp_noweight_pf(l_input, l_wt, l_output, &(input[pi]), NULL, &(output[po]));
# endif
          }
          else {
            if ((ofm1+1 == handle->blocksofm) &&  (ifm1+1 == handle->blocksifm)) {
# if defined(LIBXSMM_VLA)
              jitted_sconv_fp_weight_pf(l_input, l_wt, l_output,
                &(input[img+1][0][0][0][0]), &(weight[0][0][0][0][0][0]), &(output[img+1][0][0][0][0]));
# else
              size_t pi, pw, po;
              indexi[0] = 0; indexi[1] = 0; indexi[2] = 0; indexi[3] = 0; indexi[4] = img + 1;
              /*indexw[0] = 0; indexw[1] = 0; indexw[2] = 0; indexw[3] = 0; indexw[4] = 0; indexw[5] = 0;*/
              indexo[0] = 0; indexo[1] = 0; indexo[2] = 0; indexo[3] = 0; indexo[4] = img + 1;
              LIBXSMM_CALC_INDEX1(size_t, pi, 5, indexi, ishape);
              pw = 0;/*LIBXSMM_CALC_INDEX1(size_t, pw, 6, indexw, wshape);*/
              LIBXSMM_CALC_INDEX1(size_t, po, 5, indexo, oshape);
              jitted_sconv_fp_weight_pf(l_input, l_wt, l_output, &(input[pi]), &(weight[pw]), &(output[po]));
# endif
            }
            else {
              if ((ifm1+1 == handle->blocksifm)) {
# if defined(LIBXSMM_VLA)
                jitted_sconv_fp_weight_pf(l_input, l_wt, l_output,
                  &(input[img][0][0][0][0]), &(weight[ofm1+1][0][0][0][0][0]), &(output[img][ofm1+1][0][0][0]));
# else
                size_t pi, pw, po;
                indexi[0] = 0; indexi[1] = 0; indexi[2] = 0; indexi[3] = 0; indexi[4] = img;
                indexw[0] = 0; indexw[1] = 0; indexw[2] = 0; indexw[3] = 0; indexw[4] = 0; indexw[5] = ofm1 + 1;
                indexo[0] = 0; indexo[1] = 0; indexo[2] = 0; indexo[3] = ofm1 + 1; indexo[4] = img;
                LIBXSMM_CALC_INDEX1(size_t, pi, 5, indexi, ishape);
                LIBXSMM_CALC_INDEX1(size_t, pw, 6, indexw, wshape);
                LIBXSMM_CALC_INDEX1(size_t, po, 5, indexo, oshape);
                jitted_sconv_fp_weight_pf(l_input, l_wt, l_output, &(input[pi]), &(weight[pw]), &(output[po]));
# endif
              }
              else {
# if defined(LIBXSMM_VLA)
                jitted_sconv_fp_weight_pf(l_input, l_wt, l_output,
                  &(input[img][ifm1+1][0][0][0]), &(weight[ofm1][ifm1+1][0][0][0][0]), &(output[img][ofm1][0][0][0]));
# else
                size_t pi, pw, po;
                indexi[0] = 0; indexi[1] = 0; indexi[2] = 0; indexi[3] = ifm1 + 1; indexi[4] = img;
                indexw[0] = 0; indexw[1] = 0; indexw[2] = 0; indexw[3] = 0; indexw[4] = ifm1 + 1; indexw[5] = ofm1;
                indexo[0] = 0; indexo[1] = 0; indexo[2] = 0; indexo[3] = ofm1; indexo[4] = img;
                LIBXSMM_CALC_INDEX1(size_t, pi, 5, indexi, ishape);
                LIBXSMM_CALC_INDEX1(size_t, pw, 6, indexw, wshape);
                LIBXSMM_CALC_INDEX1(size_t, po, 5, indexo, oshape);
                jitted_sconv_fp_weight_pf(l_input, l_wt, l_output, &(input[pi]), &(weight[pw]), &(output[po]));
# endif
              }
            }
          }
#else
          jitted_sconv_fp_no_pf(l_input, l_wt, l_output, NULL, NULL, NULL);
#endif
        }
      }
    }
  }
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_convolve_st_fwd_fp32_img_parallel_opt(libxsmm_conv_handle* handle, int start_thread, int tid, int num_threads)
{
  typedef float element_type;
  const element_type *const inp = ((element_type*)handle->input->data), *const wtp = ((element_type*)handle->filter->data);
  element_type *const outp = ((element_type*)handle->output->data) + (handle->desc.pad_h * handle->ofwp + handle->desc.pad_w) * handle->ofmblock;
  int ifm1, oj, ij, oi, ii;
  /* calculate local thread ids */
  const int ltid = tid - start_thread;
  /* calculate group sizes */
  const int l_l1 = handle->desc.N * handle->blocksofm;
  const int l_l3 = handle->ofh / handle->fwd_ofh_rb;
  /* number of threads need in the ofh loop (as we have l_l1 global parallel tasks) */
  const int l_l1_gs = num_threads / l_l1;
  /* number of elemens of ofh loop per thread */
  const int l_l2_ts = (l_l3 % l_l1_gs == 0) ? (l_l3 / l_l1_gs) : ((l_l3 / l_l1_gs) + 1);
  /* get group id */
  const int l_tidgroup = ltid / l_l1_gs;
  /* compute img and ofm1 based on group */
  const int img = l_tidgroup / handle->blocksofm;
  const int ofm1 = l_tidgroup % handle->blocksofm;
  int start_ofh = l_l2_ts * (ltid % l_l1_gs);
  const int end_ofh = ((start_ofh + l_l2_ts) <= handle->ofh) ? (start_ofh + l_l2_ts) : handle->ofh;
  libxsmm_sconvfunction jitted_sconv_fp_noweight_pf = handle->code_fwd[1].sconv;
  libxsmm_sconvfunction jitted_sconv_fp_weight_pf = handle->code_fwd[2].sconv;
  /*libxsmm_sconvfunction jitted_sconv_fp_weightnooutput_pf = handle->code_fwd[3].sconv;*/
#if defined(LIBXSMM_CONV_NO_PREFETCH)
  libxsmm_sconvfunction jitted_sconv_fp_no_pf = handle->code_fwd[0].sconv;
#endif
  const element_type *l_input, *l_wt;
  element_type* l_output;
#if defined(LIBXSMM_VLA)
  typedef element_type (*LIBXSMM_RESTRICT input_data_type)[handle->blocksifm][handle->ifhp][handle->ifwp][handle->ifmblock];
  typedef element_type (*LIBXSMM_RESTRICT weight_data_type)[handle->blocksifm][handle->desc.R][handle->desc.S][handle->ifmblock][handle->ofmblock];
  typedef element_type (*LIBXSMM_RESTRICT output_data_type)[handle->blocksofm][handle->ofhp][handle->ofwp][handle->ofmblock];
  const input_data_type input = (input_data_type)inp;
  const weight_data_type weight = (weight_data_type)wtp;
  const output_data_type output = (output_data_type)outp;
#else
  const element_type *LIBXSMM_RESTRICT input = (const element_type*)inp;
  const element_type *LIBXSMM_RESTRICT weight = (const element_type*)wtp;
  element_type *LIBXSMM_RESTRICT output = (element_type*)outp;
  unsigned int ishape[5], wshape[6], oshape[5];
  unsigned int indexi[5], indexw[6], indexo[5];
  /* arrays must be initialized separately to avoid warning about values not computable at init.-time */
  ishape[0] = handle->ifmblock; ishape[1] = handle->ifwp; ishape[2] = handle->ifhp; ishape[3] = handle->blocksifm; ishape[4] = num_threads / handle->blocksofm;
  wshape[0] = handle->ofmblock; wshape[1] = handle->ifmblock; wshape[2] = handle->desc.S; wshape[3] = handle->desc.R; wshape[4] = handle->blocksifm; wshape[5] = num_threads % handle->blocksofm;
  oshape[0] = handle->ofmblock; oshape[1] = handle->ofwp; oshape[2] = handle->ofhp; oshape[3] = handle->blocksofm; oshape[4] = ishape[4];
#endif
  /* avoid ouf of bounds (dirty) */
  start_ofh = (img < handle->desc.N && ofm1 < handle->blocksofm) ? start_ofh : handle->ofh;
  for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
    for (oj = start_ofh; oj < end_ofh; oj += handle->fwd_ofh_rb) {
      ij = oj * handle->desc.u;
      for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
        ii = oi * handle->desc.v;
#if defined(LIBXSMM_VLA)
        l_input = &(input[img][ifm1][ij][ii][0]);
        l_wt = &(weight[ofm1][ifm1][0][0][0][0]);
        l_output = &(output[img][ofm1][oj][oi][0]);
#else /* index arrays must be initialized separately to avoid warning about values not computable at init.-time */
        indexi[0] = 0; indexi[1] = ii; indexi[2] = ij; indexi[3] = ifm1; indexi[4] = img;
        indexw[0] = 0; indexw[1] = 0; indexw[2] = 0; indexw[3] = 0; indexw[4] = ifm1; indexw[5] = ofm1;
        indexo[0] = 0; indexo[1] = oi; indexo[2] = oj; indexo[3] = ofm1; indexo[4] = img;
        { size_t index1;
          LIBXSMM_CALC_INDEX1(size_t, index1, 5, indexi, ishape);
          l_input = input + index1;
          LIBXSMM_CALC_INDEX1(size_t, index1, 6, indexw, wshape);
          l_wt = weight + index1;
          LIBXSMM_CALC_INDEX1(size_t, index1, 5, indexo, oshape);
          l_output = output + index1;
        }
#endif
#if !defined(LIBXSMM_CONV_NO_PREFETCH)
        /* check we are not at the end, we prefetch inside the image */
        if (oi < handle->ofw-handle->fwd_ofw_rb) {
# if defined(LIBXSMM_VLA)
          jitted_sconv_fp_noweight_pf(l_input, l_wt, l_output,
            &(input[img][ifm1][ij][(oi+handle->fwd_ofw_rb)*handle->desc.v][0]), NULL, &(output[img][ofm1][oj][oi+handle->fwd_ofw_rb][0]));
# else
          size_t pi, po;
          indexi[0] = 0; indexi[1] = (oi + handle->fwd_ofw_rb) * handle->desc.v; indexi[2] = ij; indexi[3] = ifm1; indexi[4] = img;
          indexo[0] = 0; indexo[1] = oi; indexo[2] = oj + handle->fwd_ofh_rb; indexo[3] = ofm1; indexo[4] = img;
          LIBXSMM_CALC_INDEX1(size_t, pi, 5, indexi, ishape);
          LIBXSMM_CALC_INDEX1(size_t, po, 5, indexo, oshape);
          jitted_sconv_fp_noweight_pf(l_input, l_wt, l_output, &(input[pi]), NULL, &(output[po]));
# endif
        }
        else {
          if (oj < end_ofh-handle->fwd_ofh_rb) {
# if defined(LIBXSMM_VLA)
            jitted_sconv_fp_noweight_pf(l_input, l_wt, l_output,
              &(input[img][ifm1][(oj+handle->fwd_ofw_rb)*handle->desc.u][ii][0]), NULL, &(output[img][ofm1][oj+handle->fwd_ofw_rb][oi][0]));
# else
            size_t pi, po;
            indexi[0] = 0; indexi[1] = ii; indexi[2] = (oj + handle->fwd_ofw_rb) * handle->desc.u; indexi[3] = ifm1; indexi[4] = img;
            indexo[0] = 0; indexo[1] = oi; indexo[2] = oj + handle->fwd_ofw_rb; indexo[3] = ofm1; indexo[4] = img;
            LIBXSMM_CALC_INDEX1(size_t, pi, 5, indexi, ishape);
            LIBXSMM_CALC_INDEX1(size_t, po, 5, indexo, oshape);
            jitted_sconv_fp_noweight_pf(l_input, l_wt, l_output, &(input[pi]), NULL, &(output[po]));
# endif
          }
          else {
# if defined(LIBXSMM_VLA)
            jitted_sconv_fp_weight_pf(l_input, l_wt, l_output,
              &(input[img][ifm1+1][0][0][0]), &(weight[ofm1][ifm1+1][0][0][0][0]), &(output[img][ofm1][0][0][0]));
# else
            size_t pi, pw, po;
            indexi[0] = 0; indexi[1] = 0; indexi[2] = 0; indexi[3] = ifm1 + 1; indexi[4] = img;
            indexw[0] = 0; indexw[1] = 0; indexw[2] = 0; indexw[3] = 0; indexw[4] = ifm1 + 1; indexw[5] = ofm1;
            indexo[0] = 0; indexo[1] = 0; indexo[2] = 0; indexo[3] = ofm1; indexo[4] = img;
            LIBXSMM_CALC_INDEX1(size_t, pi, 5, indexi, ishape);
            LIBXSMM_CALC_INDEX1(size_t, pw, 6, indexw, wshape);
            LIBXSMM_CALC_INDEX1(size_t, po, 5, indexo, oshape);
            jitted_sconv_fp_weight_pf(l_input, l_wt, l_output, &(input[pi]), &(weight[pw]), &(output[po]));
# endif
          }
        }
#else
        jitted_sconv_fp_no_pf(l_input, l_wt, l_output, NULL, NULL, NULL);
#endif
      }
    }
  }
}


LIBXSMM_API_DEFINITION libxsmm_conv_err_t libxsmm_convolve_st_fwd(libxsmm_conv_handle* handle, int start_thread, int tid, int num_threads)
{
  libxsmm_conv_err_t status = LIBXSMM_CONV_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->input == 0 || handle->output == 0 || handle->filter == 0) {
    status = LIBXSMM_CONV_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if (handle->code_fwd[0].sconv == 0) {
    switch (handle->datatype) {
      case LIBXSMM_CONV_DATATYPE_FP32: {
        if (1 == handle->desc.splits) {
          internal_convolve_st_fwd_fp32_fallback(handle, start_thread, tid, num_threads);
        }
        else {
          status = LIBXSMM_CONV_ERR_GENERAL;
          return status;
        }
      } break;
      default: {
        status = LIBXSMM_CONV_ERR_UNSUPPORTED_DATATYPE;
        return status;
      }
    }
  }
  else {
    switch (handle->datatype) {
      case LIBXSMM_CONV_DATATYPE_FP32: {
        if (1 == handle->desc.splits) {
          if (handle->desc.N*handle->blocksofm >= num_threads) {
            internal_convolve_st_fwd_fp32_opt(handle, start_thread, tid, num_threads);
          }
          else {
            internal_convolve_st_fwd_fp32_img_parallel_opt(handle, start_thread, tid, num_threads);
          }
        }
        else {
          status = LIBXSMM_CONV_ERR_GENERAL;
          return status;
        }
      } break;
      default: {
        status = LIBXSMM_CONV_ERR_UNSUPPORTED_DATATYPE;
        return status;
      }
    }
  }

  return status;
}
