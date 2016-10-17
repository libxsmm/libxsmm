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

int imgofm1, img, ofm1, ifm1, oj, oi;
#if !defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE)
int ij, ii;
#endif
#if defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_SPLITS)
int split;
#endif
/* computing first logical thread */
const int ltid = tid-start_thread;
/* number of tasks that could be run in parallel, we handle splits as an additional image here */
const int work = handle->desc.N*handle->blocksofm*handle->desc.splits;
/* compute chunck size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
/* avoid warning by using the xconv.sconv sequence to get some fn. ptr. to act as source of the type-cast */
libxsmm_convfunction jitted_conv_fp_noweight_pf = (libxsmm_convfunction)handle->code_fwd[1].xconv.sconv;
libxsmm_convfunction jitted_conv_fp_weight_pf = (libxsmm_convfunction)handle->code_fwd[2].xconv.sconv;
/*libxsmm_convfunction jitted_conv_fp_weightnooutput_pf = (libxsmm_convfunction)handle->code_fwd[3].xconv.sconv;*/
#if defined(LIBXSMM_CONV_NO_PREFETCH)
libxsmm_convfunction jitted_conv_fp_no_pf = (libxsmm_convfunction)handle->code_fwd[0].xconv.sconv;
#endif
const element_input_type *l_input;
const element_filter_type *l_wt;
element_output_type* l_output;

element_output_type *const out = ((element_output_type*)handle->output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
LIBXSMM_VLA_DECL(5, element_output_type, output, out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
LIBXSMM_VLA_DECL(6, const element_input_type, input, (element_input_type*)handle->input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
#if defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_SPLITS)
LIBXSMM_VLA_DECL(8, const element_filter_type, weight, (element_filter_type*)handle->filter->data, handle->blocksofm, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
#else
LIBXSMM_VLA_DECL(7, const element_filter_type, weight, (element_filter_type*)handle->filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
#endif

for (imgofm1 = thr_begin; imgofm1 < thr_end; ++imgofm1) {
  img = imgofm1/handle->blocksofm;
  ofm1 = imgofm1%handle->blocksofm;
#if defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_SPLITS)
  split = img%handle->desc.splits;
#endif
  for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
    for (oj = 0; oj < handle->ofh; oj += handle->fwd_ofh_rb) {
#if !defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE)
      ij = oj * handle->desc.u;
#endif
      for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
#if !defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE)
        ii = oi * handle->desc.v;

        l_input  = &LIBXSMM_VLA_ACCESS(6, input, img, ifm1, ij, ii, 0, 0,
                    handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
#else
        l_input  = &LIBXSMM_VLA_ACCESS(6, input, img, ifm1, oj, oi, 0, 0,
                    handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
#endif
#if defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_SPLITS)
        l_wt     = &LIBXSMM_VLA_ACCESS(8, weight, split, ofm1, ifm1, 0, 0, 0, 0, 0,
                    handle->blocksofm, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
#else
        l_wt     = &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1, 0, 0, 0, 0, 0,
                    handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
#endif
        l_output = &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0,
                    handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
#if !defined(LIBXSMM_CONV_NO_PREFETCH)
        /* check we are not at the end */
        if (oj < handle->ofh-handle->fwd_ofh_rb) {
          jitted_conv_fp_noweight_pf(l_input, l_wt, l_output,
#if !defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE)
            &LIBXSMM_VLA_ACCESS(6, input, img, ifm1, (oj + handle->fwd_ofh_rb) * handle->desc.u, ii, 0, 0,
                                   handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block), NULL,
#else
            &LIBXSMM_VLA_ACCESS(6, input, img, ifm1, oj + handle->fwd_ofh_rb, oi, 0, 0,
                                   handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block), NULL,
#endif
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj + handle->fwd_ofh_rb, oi, 0,
                                   handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
        }
        else {
          if ((ofm1+1 == handle->blocksofm) &&  (ifm1+1 == handle->blocksifm)) {
            jitted_conv_fp_weight_pf(l_input, l_wt, l_output,
              &LIBXSMM_VLA_ACCESS(6, input, img + 1, 0, 0, 0, 0, 0,
                handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block),
#if defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_SPLITS)
              &LIBXSMM_VLA_ACCESS(8, weight, (split+1)%handle->desc.splits, 0, 0, 0, 0, 0, 0, 0,
                handle->blocksofm, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block),
#else
              &LIBXSMM_VLA_ACCESS(7, weight, 0, 0, 0, 0, 0, 0, 0,
                handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block),
#endif
              &LIBXSMM_VLA_ACCESS(5, output, img + 1, 0, 0, 0, 0,
                handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
          }
          else {
            if ((ifm1+1 == handle->blocksifm)) {
              jitted_conv_fp_weight_pf(l_input, l_wt, l_output,
                &LIBXSMM_VLA_ACCESS(6, input, img, 0, 0, 0, 0, 0,
                  handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block),
#if defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_SPLITS)
                &LIBXSMM_VLA_ACCESS(8, weight, split, ofm1 + 1, 0, 0, 0, 0, 0, 0,
                  handle-blocksofm, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block),
#else
                &LIBXSMM_VLA_ACCESS(7, weight, ofm1 + 1, 0, 0, 0, 0, 0, 0,
                  handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block),
#endif
                &LIBXSMM_VLA_ACCESS(5, output, img, ofm1 + 1, 0, 0, 0,
                  handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
            }
            else {
              jitted_conv_fp_weight_pf(l_input, l_wt, l_output,
                &LIBXSMM_VLA_ACCESS(6, input, ifm1 + 1, 0, 0, 0, 0, 0,
                  handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block),
#if defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_SPLITS)
                &LIBXSMM_VLA_ACCESS(8, weight, split, ofm1, ifm1 + 1, 0, 0, 0, 0, 0,
                  handle->blocksofm, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block),
#else
                &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1 + 1, 0, 0, 0, 0, 0,
                  handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block),
#endif
                &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, 0, 0, 0,
                  handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
            }
          }
        }
#else
        jitted_conv_fp_no_pf(l_input, l_wt, l_output, NULL, NULL, NULL);
#endif
      }
    }
  }
}
