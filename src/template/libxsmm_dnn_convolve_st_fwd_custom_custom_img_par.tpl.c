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

int ifm1, oj, oi;
#if !defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE)
int ij, ii;
#endif
/* calculate local thread ids */
const int ltid = tid - start_thread;
/* calculate group sizes, we handle splits as additional images */
const int l_l1 = handle->desc.N * handle->blocksofm;
const int l_l3 = handle->ofh / handle->fwd_ofh_rb;
/* number of threads need in the ofh loop (as we have l_l1 global parallel tasks) */
const int l_l1_gs = handle->desc.threads / l_l1;
/* number of elemens of ofh loop per thread */
const int l_l2_ts = (l_l3 % l_l1_gs == 0) ? (l_l3 / l_l1_gs) : ((l_l3 / l_l1_gs) + 1);
/* get group id */
const int l_tidgroup = ltid / l_l1_gs;
/* compute img and ofm1 based on group */
const int img = l_tidgroup / handle->blocksofm;
const int ofm1 = l_tidgroup % handle->blocksofm;
int start_ofh = l_l2_ts * (ltid % l_l1_gs);
const int end_ofh = ((start_ofh + l_l2_ts) <= handle->ofh) ? (start_ofh + l_l2_ts) : handle->ofh;
const element_input_type *l_input;
const element_filter_type *l_wt;
element_output_type* l_output;

element_output_type *const out = ((element_output_type*)handle->output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
LIBXSMM_VLA_DECL(5, element_output_type, output, out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
LIBXSMM_VLA_DECL(6, const element_input_type, input, (element_input_type*)handle->input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
LIBXSMM_VLA_DECL(7, const element_filter_type, weight, (element_filter_type*)handle->filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);

/* JIT kernel function pointers */
libxsmm_convfunction jitted_conv_fp_one, jitted_conv_fp_two, jitted_conv_fp_zero;

/* avoid ouf of bounds (dirty) */
start_ofh = (img < handle->desc.N && ofm1 < handle->blocksofm) ? start_ofh : handle->ofh;

/* select kernels based on architecture */
if ( libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_MIC ||
     libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_CORE   ) {
  jitted_conv_fp_one = (libxsmm_convfunction)handle->code_fwd[1].xconv.sconv;
  jitted_conv_fp_two = (libxsmm_convfunction)handle->code_fwd[2].xconv.sconv;
#if defined(LIBXSMM_CONV_NO_PREFETCH)
  jitted_conv_fp_zero = (libxsmm_convfunction)handle->code_fwd[0].xconv.sconv;
#endif

  for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
    for (oj = start_ofh; oj < end_ofh; oj += handle->fwd_ofh_rb) {
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
        l_wt     = &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1, 0, 0, 0, 0, 0,
                    handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
        l_output = &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0,
                      handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
#if !defined(LIBXSMM_CONV_NO_PREFETCH)
        /* check we are not at the end, we prefetch inside the image */
        if (oi < handle->ofw-handle->fwd_ofw_rb) {
          jitted_conv_fp_one(l_input, l_wt, l_output,
#if !defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE)
            &LIBXSMM_VLA_ACCESS(6, input, img, ifm1, ij, (oi + handle->fwd_ofw_rb) * handle->desc.v,
              0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block),
#else
            &LIBXSMM_VLA_ACCESS(6, input, img, ifm1, oj, oi + handle->fwd_ofw_rb,
              0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block),
#endif
               NULL,
            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi + handle->fwd_ofw_rb, 0,
              handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
        }
        else {
          if (oj < end_ofh-handle->fwd_ofh_rb) {
            jitted_conv_fp_one(l_input, l_wt, l_output,
#if !defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE)
              &LIBXSMM_VLA_ACCESS(6, input, img, ifm1, (oj + handle->fwd_ofw_rb) * handle->desc.u, ii,
                0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block),
#else
              &LIBXSMM_VLA_ACCESS(6, input, img, ifm1, oj + handle->fwd_ofw_rb, oi,
                0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block),
#endif
                 NULL,
              &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj + handle->fwd_ofw_rb, oi, 0,
                handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
          }
          else {
            jitted_conv_fp_two(l_input, l_wt, l_output,
              &LIBXSMM_VLA_ACCESS(6, input, img, ifm1 + 1, 0, 0, 0, 0,
                handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block),
              &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1 + 1, 0, 0, 0, 0, 0,
                handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block),
              &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, 0, 0, 0,
                handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
          }
        }
#else
        jitted_conv_fp_zero(l_input, l_wt, l_output, NULL, NULL, NULL);
#endif
      }
    }
  }
} else if ( libxsmm_get_target_archid() == LIBXSMM_X86_AVX2 ){
  jitted_conv_fp_zero = (libxsmm_convfunction)handle->code_fwd[0].xconv.sconv;
  jitted_conv_fp_one = (libxsmm_convfunction)handle->code_fwd[1].xconv.sconv;

  for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
    for (oj = start_ofh; oj < end_ofh; oj += handle->fwd_ofh_rb) {
#if !defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE)
      ij = oj * handle->desc.u;
#endif
      for (oi = 0; oi < (handle->ofw - handle->fwd_ofw_rb_2); oi += handle->fwd_ofw_rb) {
#if !defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE)
        ii = oi * handle->desc.v;
        l_input  = &LIBXSMM_VLA_ACCESS(6, input, img, ifm1, ij, ii, 0, 0,
                    handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
#else
        l_input  = &LIBXSMM_VLA_ACCESS(6, input, img, ifm1, oj, oi, 0, 0,
                    handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
#endif
        l_wt     = &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1, 0, 0, 0, 0, 0,
                    handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
        l_output = &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0,
                      handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);

        jitted_conv_fp_zero(l_input, l_wt, l_output, NULL, NULL, NULL);
      }
      if (handle->fwd_ofw_rb_2 != 0) {
#if !defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE)
        ii = oi * handle->desc.v;
        l_input  = &LIBXSMM_VLA_ACCESS(6, input, img, ifm1, ij, ii, 0, 0,
                    handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
#else
        l_input  = &LIBXSMM_VLA_ACCESS(6, input, img, ifm1, oj, oi, 0, 0,
                    handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
#endif
        l_wt     = &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1, 0, 0, 0, 0, 0,
                    handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
        l_output = &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0,
                      handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);

        jitted_conv_fp_one(l_input, l_wt, l_output, NULL, NULL, NULL);
      }
    }
  }
/* should never happen, this is just an additional check */
} else {
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
}
