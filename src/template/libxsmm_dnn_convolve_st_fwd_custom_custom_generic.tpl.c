/******************************************************************************
** Copyright (c) 2016-2019, Intel Corporation                                **
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
/* Evangelos Georganas, Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/

int img, ofm1, ofm2, ifm1, ifm2, oj, ij, oi, ii, kj, ki, oi_use, oj_use, ii_use, ij_use, ofmb, ifmb, ojb, myOfmId, nOfmBlocks, ind;
/* computing first logical thread */
const int ltid = tid - start_thread;
int imgpt = (handle->desc.N + handle->desc.threads - 1)/handle->desc.threads;
int threads_per_image = handle->desc.threads / handle->desc.N;
int my_img_start = LIBXSMM_MIN( ltid * imgpt, handle->desc.N);
int my_img_end = LIBXSMM_MIN( (ltid+1) * imgpt, handle->desc.N);
int my_ofm_start = 0;
int my_ofm_end = handle->blocksofm;

/* Batch reduce related variables */
element_filter_type *A_ptrs[1024];
element_input_type  *B_ptrs[1024];
unsigned long long n_blocks;

/* offset output pointer in case of physical output padding */
element_output_type* out = (element_output_type*)handle->reg_output->data + ((size_t)handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
LIBXSMM_VLA_DECL(5, element_output_type, output, out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
LIBXSMM_VLA_DECL(5, const element_input_type, input, (element_input_type*)handle->reg_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
LIBXSMM_VLA_DECL(6, const element_filter_type, weight, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);

if ( imgpt <= 1 ) {
  my_img_start = LIBXSMM_MIN( ltid / threads_per_image, handle->desc.N);
  my_img_end = LIBXSMM_MIN( my_img_start + 1, handle->desc.N);
  myOfmId = ltid % threads_per_image;
  nOfmBlocks = (handle->blocksofm + threads_per_image - 1) / threads_per_image;
  my_ofm_start = LIBXSMM_MIN(myOfmId * nOfmBlocks, handle->blocksofm);
  my_ofm_end = LIBXSMM_MIN((myOfmId+1) * nOfmBlocks, handle->blocksofm);
}

if (1) { // (loop_order == N_Kb_Cb_Hb_k_c_h_w) {
  for (img = my_img_start; img < my_img_end; img++) {
    for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
      for (ifmb = 0; ifmb < handle->blocksifm; ifmb += handle->block_fwd_ifm) {
        for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
          for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {

            if ( (ifmb == 0) && ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && handle->avoid_acc_load == 0 && ojb == 0) {
              /* set output feature map to zero */
              for (oj = 0; oj < handle->ofh; ++oj) {
                element_output_type* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, output, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
                for (oi = 0; oi < handle->ofw; ++oi) {
                  LIBXSMM_PRAGMA_SIMD
                    for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
                      temp_ptr[ofm2] = (element_output_type)0;
                    }
                  temp_ptr += handle->ofmblock;
                }
              }
            }

            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, handle->blocksifm); ifm1 += handle->blocksifm_blocking) {
              for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
                for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
                  /* Prepare batch-reduce kernel arguments */
                  ij_use = oj * handle->desc.u;
                  ii_use = oi * handle->desc.v;
                  oi_use = oi;
                  oj_use = oj;
                  ind = 0;
                  for (ifm2 = ifm1; ifm2 < ifm1 + handle->blocksifm_blocking; ifm2++) {
                    for (kj = 0; kj < handle->desc.R; kj++) {
                      for (ki = 0; ki < handle->desc.S; ki++) {
                        A_ptrs[ind] = &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm2, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
                        B_ptrs[ind] = &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm2, ij_use + kj, ii_use + ki, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
                        ind++;
                      }
                    }
                  }
                  n_blocks = ind;
                  br_gemm_kernel(A_ptrs, B_ptrs, &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj_use, oi_use, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks);
                }
              }
            }
          }
        }
      }
    }
  }
}

#if 0
if (loop_order == N_Kb_h_w_k_c ) {

}
#endif

#if 0
/* perform convolution */
for (imgofm1 = thr_begin; imgofm1 < thr_end; ++imgofm1) i{
  img = imgofm1 / handle->blocksofm;
  ofm1 = imgofm1 % handle->blocksofm;
  for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
    /* reset result buffer to zero when intent is to overwrite when first block
       of input channels should be convoluted */
    if ( (ifm1 == 0) && ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) == 0) ) {
      /* set output feature map to zero */
      for (oj = 0; oj < handle->ofh; ++oj) {
        element_output_type* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, output, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock));
        for (oi = 0; oi < handle->ofw; ++oi) {
          LIBXSMM_PRAGMA_SIMD
            for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
              temp_ptr[ofm2] = (element_output_type)0;
            }
          temp_ptr += handle->ofmblock;
        }
      }
    }
    /* run convolution */
    for (oj = 0; oj < handle->ofh; ++oj) {
      ij = oj * handle->desc.u;
      ii = 0; oi = 0;
      for (kj = 0; kj < handle->desc.R; ++kj) {
        for (ki = 0; ki< handle->desc.S; ++ki) {
          gemm_kernel( &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock),
              &LIBXSMM_VLA_ACCESS(5,  input,  img, ifm1, ij + kj, ii + ki, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
              &LIBXSMM_VLA_ACCESS(5, output,  img, ofm1, oj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock) );
        }
      }
    }
  }
}
#endif

