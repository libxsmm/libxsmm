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
/* Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/

int ifm1, oj, ij, oi, ii, ofm2;
/* calculate local thread ids */
const int ltid = tid - start_thread;
/* calculate group sizes */
const int l_l1 = handle->desc.N * handle->blocksofm;
const int l_l3 = handle->ofh / handle->fwd_ofh_rb;
/* number of threads need in the ofh loop (as we have l_l1 global parallel tasks) */
const int l_l1_gs = handle->desc.threads / l_l1;
/* number of elemens of ofh loop per thread */
const int l_l2_ts = (l_l3 % l_l1_gs == 0) ? ((l_l3 / l_l1_gs)*handle->fwd_ofh_rb) : (((l_l3 / l_l1_gs) + 1)*handle->fwd_ofh_rb);
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

element_output_type *const out = ((element_output_type*)handle->reg_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->blocksofm * handle->ofmblock;
LIBXSMM_VLA_DECL(5, element_output_type, output, out, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
LIBXSMM_VLA_DECL(6, const element_input_type, input, (element_input_type*)handle->reg_input->data, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
LIBXSMM_VLA_DECL(7, const element_filter_type, weight, (element_filter_type*)handle->reg_filter->data, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock, handle->fm_lp_block);

#if defined(INPUT_PADDING)
/* Variables and initializations related to padding */
const element_input_type *input_ptr;
element_input_type *copy_ptr;
const int padded_h = handle->ifhp + 2 * handle->desc.pad_h;
const int padded_w = handle->ifwp + 2 * handle->desc.pad_w;
LIBXSMM_VLA_DECL(5, element_input_type, input_buffer, ((element_input_type*)handle->scratch5) + ltid * handle->blocksifm * padded_h * padded_w * handle->ifmblock * handle->fm_lp_block, padded_w, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
libxsmm_xmatcopyfunction jitted_matcopy;
#endif

/* JIT kernel function pointers */
libxsmm_convfunction jitted_conv_fp_one, jitted_conv_fp_two, jitted_conv_fp_zero;

/* avoid out of bounds (unclean) */
start_ofh = (img < handle->desc.N && ofm1 < handle->blocksofm) ? start_ofh : handle->ofh;

/* select kernels based on architecture */
if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  ||
     libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ||
     libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM ) {
  jitted_conv_fp_one = (libxsmm_convfunction)handle->code_fwd[1].xconv.sconv;
  jitted_conv_fp_two = (libxsmm_convfunction)handle->code_fwd[2].xconv.sconv;
#if defined(LIBXSMM_CONV_NO_PREFETCH)
  jitted_conv_fp_zero = (libxsmm_convfunction)handle->code_fwd[0].xconv.sconv;
#endif

#if defined(INPUT_PADDING)
  jitted_matcopy = handle->matcopy_fwd[0].xmatcopy;
  input_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(6, input, img, 0, 0, 0, 0, 0, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
  copy_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_buffer, handle->desc.pad_h, handle->desc.pad_w, 0, 0, 0, padded_w, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
  jitted_matcopy(input_ptr, NULL, copy_ptr, NULL, NULL);
#endif

  for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
    /* reset result buffer to zero when intent is to overwrite when first block
      of input channels should be convoluted */
    if ( (ifm1 == 0) && ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ) {
      for (oj = start_ofh; oj < end_ofh; oj++) {
        element_output_type* temp_ptr = &LIBXSMM_VLA_ACCESS(5, output, img, oj, 0, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
        for (oi = 0; oi < handle->ofw; oi++) {
          LIBXSMM_PRAGMA_SIMD
          for (ofm2 = 0; ofm2 < handle->ofmblock; ofm2++) {
            temp_ptr[ofm2] = (element_output_type)0;
          }
          temp_ptr += handle->blocksofm*handle->ofmblock;
        }
      }
    }
    for (oj = start_ofh; oj < end_ofh; oj += handle->fwd_ofh_rb) {
      ij = oj * handle->desc.u;
      for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
        ii = oi * handle->desc.v;

#if defined(INPUT_PADDING)
        l_input  = &LIBXSMM_VLA_ACCESS(5, input_buffer, ij, ii, ifm1, 0, 0,
                                       padded_w, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
#else
        l_input  = &LIBXSMM_VLA_ACCESS(6, input, img, ij, ii, ifm1, 0, 0,
                    handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
#endif
        l_wt     = &LIBXSMM_VLA_ACCESS(7, weight, 0, 0, ifm1, 0, ofm1, 0, 0,
                    handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock, handle->fm_lp_block);
        l_output = &LIBXSMM_VLA_ACCESS(5, output, img, oj, oi, ofm1, 0,
                    handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
#if !defined(LIBXSMM_CONV_NO_PREFETCH)
        /* check we are not at the end, we prefetch inside the image */
        if (oi < handle->ofw-handle->fwd_ofw_rb) {
          jitted_conv_fp_one(l_input, l_wt, l_output,
#if defined(INPUT_PADDING)
            &LIBXSMM_VLA_ACCESS(5, input_buffer, ij, (oi + handle->fwd_ofw_rb) * handle->desc.v, ifm1, 0, 0,
                                                 padded_w, handle->blocksifm, handle->ifmblock, handle->fm_lp_block),
#else
            &LIBXSMM_VLA_ACCESS(6, input, img, ij, (oi + handle->fwd_ofw_rb) * handle->desc.v, ifm1, 0, 0,
              handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block),
#endif
            NULL,
            &LIBXSMM_VLA_ACCESS(5, output, img, oj, oi + handle->fwd_ofw_rb, ofm1, 0,
              handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock));
        }
        else {
          if (oj < end_ofh-handle->fwd_ofh_rb) {
            jitted_conv_fp_one(l_input, l_wt, l_output,
#if defined(INPUT_PADDING)
              &LIBXSMM_VLA_ACCESS(5, input_buffer, (oj + handle->fwd_ofw_rb) * handle->desc.u, ii, ifm1, 0, 0,
                                                   padded_w, handle->blocksifm, handle->ifmblock, handle->fm_lp_block),
#else
              &LIBXSMM_VLA_ACCESS(6, input, img, (oj + handle->fwd_ofw_rb) * handle->desc.u, ii, ifm1, 0, 0,
               handle->ifhp, handle->ifwp,  handle->blocksifm, handle->ifmblock, handle->fm_lp_block),
#endif
              NULL,
              &LIBXSMM_VLA_ACCESS(5, output, img, oj + handle->fwd_ofw_rb, oi, ofm1, 0,
               handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock));
          }
          else {
            jitted_conv_fp_two(l_input, l_wt, l_output,
#if defined(INPUT_PADDING)
              &LIBXSMM_VLA_ACCESS(5, input_buffer, 0, 0, ifm1 + 1, 0, 0,
                                 padded_w, handle->blocksifm, handle->ifmblock, handle->fm_lp_block),
#else
              &LIBXSMM_VLA_ACCESS(6, input, img, 0, 0, ifm1 + 1, 0, 0,
                handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block),
#endif
              &LIBXSMM_VLA_ACCESS(7, weight, 0, 0, ifm1 + 1, 0, ofm1, 0, 0,
                handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock, handle->fm_lp_block),
              &LIBXSMM_VLA_ACCESS(5, output, img, 0, 0, ofm1, 0,
                handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock));
          }
        }
#else
        jitted_conv_fp_zero(l_input, l_wt, l_output, NULL, NULL, NULL);
#endif
      }
    }
  }
} else if ( libxsmm_target_archid == LIBXSMM_X86_AVX2 ) {
  jitted_conv_fp_zero = (libxsmm_convfunction)handle->code_fwd[0].xconv.sconv;
  jitted_conv_fp_one = (libxsmm_convfunction)handle->code_fwd[1].xconv.sconv;

#if defined(INPUT_PADDING)
  jitted_matcopy = handle->matcopy_fwd[0].xmatcopy;
  input_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(6, input, img, 0, 0, 0, 0, 0, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
  copy_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_buffer, handle->desc.pad_h, handle->desc.pad_w, 0, 0, 0, padded_w, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
  jitted_matcopy(input_ptr, NULL, copy_ptr, NULL, NULL);
#endif

  for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
    /* reset result buffer to zero when intent is to overwrite when first block
      of input channels should be convoluted */
    if ( (ifm1 == 0) && ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ) {
      for (oj = start_ofh; oj < end_ofh; oj++) {
        element_output_type* temp_ptr = &LIBXSMM_VLA_ACCESS(5, output, img, oj, 0, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
        for (oi = 0; oi < handle->ofw; oi++) {
          LIBXSMM_PRAGMA_SIMD
          for (ofm2 = 0; ofm2 < handle->ofmblock; ofm2++) {
            temp_ptr[ofm2] = (element_output_type)0;
          }
          temp_ptr += handle->blocksofm*handle->ofmblock;
        }
      }
    }
    for (oj = start_ofh; oj < end_ofh; oj += handle->fwd_ofh_rb) {
      ij = oj * handle->desc.u;
      for (oi = 0; oi < (handle->ofw - handle->fwd_ofw_rb_2); oi += handle->fwd_ofw_rb) {
        ii = oi * handle->desc.v;
#if defined(INPUT_PADDING)
        l_input  = &LIBXSMM_VLA_ACCESS(5, input_buffer, ij, ii, ifm1,  0, 0,
                                       padded_w, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
#else
        l_input  = &LIBXSMM_VLA_ACCESS(6, input, img, ij, ii, ifm1, 0, 0,
                    handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
#endif
        l_wt     = &LIBXSMM_VLA_ACCESS(7, weight, 0, 0, ifm1, 0, ofm1, 0, 0,
                    handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock, handle->fm_lp_block);
        l_output = &LIBXSMM_VLA_ACCESS(5, output, img, oj, oi, ofm1, 0,
                    handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
        jitted_conv_fp_zero(l_input, l_wt, l_output, NULL, NULL, NULL);
      }
      if (handle->fwd_ofw_rb_2 != 0) {
        ii = oi * handle->desc.v;
#if defined(INPUT_PADDING)
        l_input  = &LIBXSMM_VLA_ACCESS(5, input_buffer, ij, ii, ifm1,  0, 0,
                                       padded_w, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
#else
        l_input  = &LIBXSMM_VLA_ACCESS(6, input, img, ij, ii, ifm1, 0, 0,
                    handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock, handle->fm_lp_block);
#endif
        l_wt     = &LIBXSMM_VLA_ACCESS(7, weight, 0, 0, ifm1, 0, ofm1, 0, 0,
                    handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock, handle->fm_lp_block);
        l_output = &LIBXSMM_VLA_ACCESS(5, output, img, oj, oi, ofm1, 0,
                    handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
        jitted_conv_fp_one(l_input, l_wt, l_output, NULL, NULL, NULL);
      }
    }
  }
/* should never happen, this is just an additional check */
} else {
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
}
