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
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **d
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

int imgofm1, img, ofm1, ifm1, oj, ij, oi, ii;
/* computing first logical thread */
const int ltid = tid-start_thread;
/* number of tasks that could be run in parallel */
const int work = handle->desc.N*handle->blocksofm;
/* compute chunck size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
const element_input_type *l_input;
const element_filter_type *l_wt;
element_output_type* l_output;

element_output_type *const out = ((element_output_type*)handle->output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->blocksofm * handle->ofmblock;
LIBXSMM_VLA_DECL(5, element_output_type, output, out, handle->ofhp, handle->ofwp,handle->blocksofm,  handle->ofmblock);
LIBXSMM_VLA_DECL(6, const element_input_type, input, (element_input_type*)handle->input->data, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
LIBXSMM_VLA_DECL(7, const element_filter_type, weight, (element_filter_type*)handle->filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);

/* JIT kernel function pointers */
libxsmm_convfunction jitted_conv_fp_one, jitted_conv_fp_two, jitted_conv_fp_zero;

/* select kernels based on architecture */
if ( libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_MIC ||
     libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_CORE   ) {
  jitted_conv_fp_one = (libxsmm_convfunction)handle->code_fwd[1].xconv.sconv;
  jitted_conv_fp_two = (libxsmm_convfunction)handle->code_fwd[2].xconv.sconv;
#if defined(LIBXSMM_CONV_NO_PREFETCH)
  jitted_conv_fp_zero = (libxsmm_convfunction)handle->code_fwd[0].xconv.sconv;
#endif

  for (imgofm1 = thr_begin; imgofm1 < thr_end; ++imgofm1) {
    img = imgofm1/handle->blocksofm;
    ofm1 = imgofm1%handle->blocksofm;
    for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
      for (oj = 0; oj < handle->ofh; oj += handle->fwd_ofh_rb) {
        ij = oj * handle->desc.u;
        for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
          ii = oi * handle->desc.v;
          l_input  = &LIBXSMM_VLA_ACCESS(6, input, img, ij, ii, ifm1, 0, 0,
                      handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
          l_wt     = &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1, 0, 0, 0, 0, 0,
                      handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
          l_output = &LIBXSMM_VLA_ACCESS(5, output, img, oj, oi, ofm1, 0,
                      handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
#if !defined(LIBXSMM_CONV_NO_PREFETCH)
          /* check we are not at the end */
          if (oj < handle->ofh-handle->fwd_ofh_rb) {
            jitted_conv_fp_one(l_input, l_wt, l_output,
              &LIBXSMM_VLA_ACCESS(6, input, img, (oj + handle->fwd_ofh_rb) * handle->desc.u, ii, ifm1, 0, 0,
                                     handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block), NULL,
              &LIBXSMM_VLA_ACCESS(5, output, img, oj + handle->fwd_ofh_rb, oi, ofm1, 0,
                                     handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock));
          }
          else {
            if ((ofm1+1 == handle->blocksofm) &&  (ifm1+1 == handle->blocksifm)) {
              jitted_conv_fp_two(l_input, l_wt, l_output,
                &LIBXSMM_VLA_ACCESS(6, input, img + 1, 0, 0, 0, 0, 0,
                  handle->ifhp, handle->ifwp, handle->ifmblock, handle->blocksifm, handle->fm_lp_block),
                &LIBXSMM_VLA_ACCESS(7, weight, 0, 0, 0, 0, 0, 0, 0,
                  handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block),
                &LIBXSMM_VLA_ACCESS(5, output, img + 1, 0, 0, 0, 0,
                  handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock));
            }
            else {
              if ((ifm1+1 == handle->blocksifm)) {
                jitted_conv_fp_two(l_input, l_wt, l_output,
                  &LIBXSMM_VLA_ACCESS(6, input, img, 0, 0, 0, 0, 0,
                    handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block),
                  &LIBXSMM_VLA_ACCESS(7, weight, ofm1 + 1, 0, 0, 0, 0, 0, 0,
                    handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block),
                  &LIBXSMM_VLA_ACCESS(5, output, img, ofm1 + 1, 0, 0, 0,
                    handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock));
              }
              else {
                jitted_conv_fp_two(l_input, l_wt, l_output,
                  &LIBXSMM_VLA_ACCESS(6, input, 0, 0, 0, ifm1 + 1, 0, 0,
                    handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block),
                  &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1 + 1, 0, 0, 0, 0, 0,
                    handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block),
                  &LIBXSMM_VLA_ACCESS(5, output, img, 0, 0, ofm1, 0,
                    handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock));
              }
            }
          }
#else
          jitted_conv_fp_three(l_input, l_wt, l_output, NULL, NULL, NULL);
#endif
        }
      }
    }
  }
} else if ( libxsmm_get_target_archid() == LIBXSMM_X86_AVX2 ){
  jitted_conv_fp_zero = (libxsmm_convfunction)handle->code_fwd[0].xconv.sconv;
  jitted_conv_fp_one = (libxsmm_convfunction)handle->code_fwd[1].xconv.sconv;

  for (imgofm1 = thr_begin; imgofm1 < thr_end; ++imgofm1) {
    img = imgofm1/handle->blocksofm;
    ofm1 = imgofm1%handle->blocksofm;
    for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
      for (oj = 0; oj < handle->ofh; oj += handle->fwd_ofh_rb) {
        ij = oj * handle->desc.u;
        for (oi = 0; oi < (handle->ofw - handle->fwd_ofw_rb_2); oi += handle->fwd_ofw_rb) {
          ii = oi * handle->desc.v;
          l_input  = &LIBXSMM_VLA_ACCESS(6, input, img, ij, ii, ifm1, 0, 0,
                      handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
          l_wt     = &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1, 0, 0, 0, 0, 0,
                      handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
          l_output = &LIBXSMM_VLA_ACCESS(5, output, img, oj, oi, ofm1, 0,
                      handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
          jitted_conv_fp_zero(l_input, l_wt, l_output, NULL, NULL, NULL);
        }
        if (handle->fwd_ofw_rb_2 != 0) {
          ii = oi * handle->desc.v;
          l_input  = &LIBXSMM_VLA_ACCESS(6, input, img, ij, ii, ifm1, 0, 0,
                      handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
          l_wt     = &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1, 0, 0, 0, 0, 0,
                      handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
          l_output = &LIBXSMM_VLA_ACCESS(5, output, img, oj, oi, ofm1, 0,
                      handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
          jitted_conv_fp_one(l_input, l_wt, l_output, NULL, NULL, NULL);
        }
      }
    }
  }
/* should never happen, this is just an additional check */
} else {
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
}
