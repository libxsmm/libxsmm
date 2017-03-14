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

int imgofm1, img, ofm1, ifm1, oj, oi, ofm2;
#if !defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE)
int ij, ii;
#endif
/* computing first logical thread */
const int ltid = tid-start_thread;
/* number of tasks that could be run in parallel */
const int work = handle->desc.N*(handle->blocksofm*handle->fm_lp_block);
/* compute chunck size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

const element_input_type *l_input;
const element_filter_type *l_wt;
element_output_type* l_output;

/* regular/high precision */
element_output_type* out = 0;
/* low precision */
element_input_type* out_lp = 0;

#if defined(INPUT_PADDING)
/* Variables and initializations related to padding */
int prev_image = -1;
const element_input_type *input_ptr;
element_input_type *copy_ptr;
element_input_type *prefetch_ptr;
const int padded_h = handle->ifhp + 2 * handle->desc.pad_h;
const int padded_w = handle->ifwp + 2 * handle->desc.pad_w;
LIBXSMM_VLA_DECL(5, element_input_type, input_buffer, ((element_input_type*)handle->scratch5) + ltid * handle->blocksifm * padded_h * padded_w * handle->ifmblock * handle->fm_lp_block, padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
libxsmm_matcopyfunction jitted_matcopy;
#endif

/* select pointer based on precision */
if (handle->datatype != handle->datatype_itm) {
  out = ((element_output_type*)handle->scratch6) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * (handle->ofmblock);
  out_lp = ((element_input_type*)handle->reg_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * (handle->ofmblock * handle->fm_lp_block);
} else {
  out = ((element_output_type*)handle->reg_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * (handle->ofmblock);
  out_lp = 0;
}

{ /* open new scope for additional variable declarations (C89) */
  LIBXSMM_VLA_DECL(5, element_output_type, output, out, handle->blocksofm*handle->fm_lp_block, handle->ofhp, handle->ofwp, handle->ofmblock);
  LIBXSMM_VLA_DECL(6, element_input_type, output_lp, out_lp, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);
  LIBXSMM_VLA_DECL(6, const element_input_type, input, (element_input_type*)handle->reg_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
  LIBXSMM_VLA_DECL(7, const element_filter_type, weight, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
  /* JIT kernel function pointers */
  libxsmm_convfunction jitted_conv_fp_one, jitted_conv_fp_two, jitted_conv_fp_zero;
  
  /* select kernels based on architecture */
  if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  ||
      libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ||
      libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM ) {
    jitted_conv_fp_one = (libxsmm_convfunction)handle->code_fwd[1].xconv.sconv;
    /* on KNM prefetches are less costly, so let's avoid some branch mispredicts by running redundant weight prefeteches */
    if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM ) {
      jitted_conv_fp_one = (libxsmm_convfunction)handle->code_fwd[2].xconv.sconv;
    }
    jitted_conv_fp_two = (libxsmm_convfunction)handle->code_fwd[2].xconv.sconv;
#if defined(LIBXSMM_CONV_NO_PREFETCH)
    jitted_conv_fp_zero = (libxsmm_convfunction)handle->code_fwd[0].xconv.sconv;
#endif
    
#if defined(INPUT_PADDING)
    jitted_matcopy = (libxsmm_matcopyfunction)handle->matcopy_fwd[0].xmatcopy.smatcopy;
#endif
    
    for (imgofm1 = thr_begin; imgofm1 < thr_end; ++imgofm1) {
      img = imgofm1/(handle->blocksofm*handle->fm_lp_block);
      ofm1 = imgofm1%(handle->blocksofm*handle->fm_lp_block);
#if defined(INPUT_PADDING)
      if (prev_image != img) {
        /* The img has changed so we should copy all the ifms */
        /* Start copying form the end of the ifm1 index in order to copy last the ifm1 that will be used first in the computation */
        for (ifm1 = handle->blocksifm-1; ifm1 >= 0; ifm1--) {
          input_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(6, input, img, ifm1, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
          copy_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_buffer, ifm1, handle->desc.pad_h, handle->desc.pad_w, 0, 0, padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
          if ( ifm1-1 == -1) {
            prefetch_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(6, input, img+1, handle->blocksifm-1, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
          } else {
            prefetch_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(6, input, img, ifm1-1, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
          }
          jitted_matcopy(input_ptr, NULL, copy_ptr, NULL, prefetch_ptr);
        }
        prev_image = img;
      }
#endif
      /* up-convert */
      if (handle->datatype != handle->datatype_itm) {
        for (oj = 0; oj < handle->ofh; ++oj) {
          for (oi = 0; oi < handle->ofw; ++oi) {
            for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
              LIBXSMM_VLA_ACCESS(  5, output, img, ofm1, oj, oi, ofm2, handle->blocksofm*handle->fm_lp_block, handle->ofhp, handle->ofwp, handle->ofmblock) = (element_output_type)
              (LIBXSMM_VLA_ACCESS(  6, output_lp, img, ofm1/handle->fm_lp_block, oj, oi, ((handle->ofmblock/handle->fm_lp_block)*(ofm1%handle->fm_lp_block))+ofm2/handle->fm_lp_block, ofm2%handle->fm_lp_block, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block));
            }
          }
        }
      }
      for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
        for (oj = 0; oj < handle->ofh; oj += handle->fwd_ofh_rb) {
#if !defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE)
          ij = oj * handle->desc.u;
#endif
          for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
#if !defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE)
            ii = oi * handle->desc.v;

#if defined(INPUT_PADDING)
            l_input  = &LIBXSMM_VLA_ACCESS(5, input_buffer, ifm1, ij, ii, 0, 0,
                                           padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
#else
            l_input  = &LIBXSMM_VLA_ACCESS(6, input, img, ifm1, ij, ii, 0, 0,
                                           handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
#endif
            
#else
            
#if defined(INPUT_PADDING)
            l_input  = &LIBXSMM_VLA_ACCESS(5, input_buffer, ifm1, oj, oi, 0, 0,
                                           padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
#else
            l_input  = &LIBXSMM_VLA_ACCESS(6, input, img, ifm1, oj, oi, 0, 0,
                                           handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
#endif

#endif
            l_wt     = &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1, 0, 0, 0, 0, 0,
                                           handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
            l_output = &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0,
                                           handle->blocksofm*handle->fm_lp_block, handle->ofhp, handle->ofwp, handle->ofmblock);
#if !defined(LIBXSMM_CONV_NO_PREFETCH)
            /* check we are not at the end */
            if (oj < handle->ofh-handle->fwd_ofh_rb) {
              jitted_conv_fp_one(l_input, l_wt, l_output,
#if !defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE)

#if defined(INPUT_PADDING)
                                 &LIBXSMM_VLA_ACCESS(5, input_buffer, ifm1, (oj + handle->fwd_ofh_rb) * handle->desc.u, ii, 0, 0,
                                                     padded_h, padded_w, handle->ifmblock, handle->fm_lp_block),
#else
                                 &LIBXSMM_VLA_ACCESS(6, input, img, ifm1, (oj + handle->fwd_ofh_rb) * handle->desc.u, ii, 0, 0,
                                                     handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block),
#endif
                                 
#else
                                 

#if defined(INPUT_PADDING)
                                 &LIBXSMM_VLA_ACCESS(5, input_buffer, ifm1, oj + handle->fwd_ofh_rb, oi, 0, 0,
                                                     padded_h, padded_w, handle->ifmblock, handle->fm_lp_block),
#else
                                 &LIBXSMM_VLA_ACCESS(6, input, img, ifm1, oj + handle->fwd_ofh_rb, oi, 0, 0,
                                                     handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block),
#endif

#endif
                                 NULL,
                                 &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj + handle->fwd_ofh_rb, oi, 0,
                                                     handle->blocksofm*handle->fm_lp_block, handle->ofhp, handle->ofwp, handle->ofmblock));
            }
            else {
              if ((ofm1+1 == handle->blocksofm) &&  (ifm1+1 == handle->blocksifm)) {
                jitted_conv_fp_two(l_input, l_wt, l_output,
#if defined(INPUT_PADDING)
                                   &LIBXSMM_VLA_ACCESS(5, input_buffer, 0, 0, 0, 0, 0,
                                                       padded_h, padded_w, handle->ifmblock, handle->fm_lp_block),
#else
                                   &LIBXSMM_VLA_ACCESS(6, input, img + 1, 0, 0, 0, 0, 0,
                                                       handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block),
#endif
                                   &LIBXSMM_VLA_ACCESS(7, weight, 0, 0, 0, 0, 0, 0, 0,
                                                       handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block),
                                   &LIBXSMM_VLA_ACCESS(5, output, img + 1, 0, 0, 0, 0,
                                                       handle->blocksofm*handle->fm_lp_block, handle->ofhp, handle->ofwp, handle->ofmblock));
              }
              else {
                if ((ifm1+1 == handle->blocksifm)) {
                  jitted_conv_fp_two(l_input, l_wt, l_output,
#if defined(INPUT_PADDING)
                                     &LIBXSMM_VLA_ACCESS(5, input_buffer, 0, 0, 0, 0, 0,
                                                         padded_h, padded_w, handle->ifmblock, handle->fm_lp_block),
#else
                                     &LIBXSMM_VLA_ACCESS(6, input, img, 0, 0, 0, 0, 0,
                                                         handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block),
#endif
                                     &LIBXSMM_VLA_ACCESS(7, weight, ofm1 + 1, 0, 0, 0, 0, 0, 0,
                                                         handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block),
                                     &LIBXSMM_VLA_ACCESS(5, output, img, ofm1 + 1, 0, 0, 0,
                                                         handle->blocksofm*handle->fm_lp_block, handle->ofhp, handle->ofwp, handle->ofmblock));
                }
                else {
                  jitted_conv_fp_two(l_input, l_wt, l_output,
#if defined(INPUT_PADDING)
                                     &LIBXSMM_VLA_ACCESS(5, input_buffer, ifm1 + 1, 0, 0, 0, 0,
                                                         padded_h, padded_w, handle->ifmblock, handle->fm_lp_block),
#else
                                     &LIBXSMM_VLA_ACCESS(6, input, img, ifm1 + 1, 0, 0, 0, 0,
                                                         handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block),
#endif
                                     &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1 + 1, 0, 0, 0, 0, 0,
                                                         handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block),
                                     &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, 0, 0, 0,
                                                         handle->blocksofm*handle->fm_lp_block, handle->ofhp, handle->ofwp, handle->ofmblock));
                }
              }
            }
#else
            jitted_conv_fp_zero(l_input, l_wt, l_output, NULL, NULL, NULL);
#endif
          }
        }
      }
      /* down-convert */
      if (handle->datatype != handle->datatype_itm) {
        for (oj = 0; oj < handle->ofh; ++oj) {
          for (oi = 0; oi < handle->ofw; ++oi) {
            for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
              LIBXSMM_VLA_ACCESS(  6, output_lp, img, ofm1/handle->fm_lp_block, oj, oi, ((handle->ofmblock/handle->fm_lp_block)*(ofm1%handle->fm_lp_block)+ofm2/handle->fm_lp_block), ofm2%handle->fm_lp_block, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block)
              = (element_input_type)(LIBXSMM_VLA_ACCESS(  5, output, img, ofm1, oj, oi, ofm2, handle->blocksofm*handle->fm_lp_block, handle->ofhp, handle->ofwp, handle->ofmblock));
            }
          }
        }
      }
    }
  } else if ( libxsmm_target_archid == LIBXSMM_X86_AVX2 ) {
    jitted_conv_fp_zero = (libxsmm_convfunction)handle->code_fwd[0].xconv.sconv;
    jitted_conv_fp_one = (libxsmm_convfunction)handle->code_fwd[1].xconv.sconv;
#if defined(INPUT_PADDING)
    jitted_matcopy = (libxsmm_matcopyfunction)handle->matcopy_fwd[0].xmatcopy.smatcopy;
#endif
    for (imgofm1 = thr_begin; imgofm1 < thr_end; ++imgofm1) {
      img = imgofm1/handle->blocksofm;
      ofm1 = imgofm1%handle->blocksofm;
#if defined(INPUT_PADDING)
      if (prev_image != img) {
        /* The img has changed so we should copy all the ifms */
        /* Start copying form the end of the ifm1 index in order to copy last the ifm1 that will be used first in the computation */
        for (ifm1 = handle->blocksifm-1; ifm1 >= 0; ifm1--) {
          input_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(6, input, img, ifm1, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
          copy_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_buffer, ifm1, handle->desc.pad_h, handle->desc.pad_w, 0, 0, padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
          if ( ifm1-1 == -1) {
            prefetch_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(6, input, img+1, handle->blocksifm-1, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
          } else {
            prefetch_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(6, input, img, ifm1-1, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
          }
          jitted_matcopy(input_ptr, NULL, copy_ptr, NULL, prefetch_ptr);
        }
        prev_image = img;
      }
#endif
      for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
        for (oj = 0; oj < handle->ofh; oj += handle->fwd_ofh_rb) {
#if !defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE)
          ij = oj * handle->desc.u;
#endif
          for (oi = 0; oi < (handle->ofw - handle->fwd_ofw_rb_2); oi += handle->fwd_ofw_rb) {
#if !defined(LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE)
            ii = oi * handle->desc.v;
#if defined(INPUT_PADDING)
            l_input  = &LIBXSMM_VLA_ACCESS(5, input_buffer, ifm1, ij, ii, 0, 0,
                                           padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
#else
            l_input  = &LIBXSMM_VLA_ACCESS(6, input, img, ifm1, ij, ii, 0, 0,
                                           handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
#endif
            
#else
            

#if defined(INPUT_PADDING)
            l_input  = &LIBXSMM_VLA_ACCESS(5, input_buffer, ifm1, oj, oi, 0, 0,
                                           padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
#else
            l_input  = &LIBXSMM_VLA_ACCESS(6, input, img, ifm1, oj, oi, 0, 0,
                                           handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
#endif

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

#if defined(INPUT_PADDING)
            l_input  = &LIBXSMM_VLA_ACCESS(5, input_buffer, ifm1, ij, ii, 0, 0,
                                           padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
#else
            l_input  = &LIBXSMM_VLA_ACCESS(6, input, img, ifm1, ij, ii, 0, 0,
                                           handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
#endif
            
#else
            
#if defined(INPUT_PADDING)
            l_input  = &LIBXSMM_VLA_ACCESS(5, input_buffer, ifm1, oj, oi, 0, 0,
                                           padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
#else
            l_input  = &LIBXSMM_VLA_ACCESS(6, input, img, ifm1, oj, oi, 0, 0,
                                           handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
#endif
#endif
            l_wt     = &LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1, 0, 0, 0, 0, 0,
                                           handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
            l_output = &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0,
                                           handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
            jitted_conv_fp_one(l_input, l_wt, l_output, NULL, NULL, NULL);
          }
        }
      }
    }
    /* should never happen, this is just an additional check */
  } else {
    status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
  }
}
