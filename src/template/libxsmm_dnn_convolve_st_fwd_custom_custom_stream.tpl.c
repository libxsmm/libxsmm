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
/* Evangelos Georganas (Intel Corp.)
 ******************************************************************************/

const int ltid = tid-start_thread;
const element_input_type *input_base;
const element_filter_type *weight_base;
element_output_type *output_base;
int offset_i, offset_o, offset_w, pi, po, pw, pc, i = 0;
int *stream = handle->compute_fwd_indices_ptrs[ltid];
libxsmm_convfunction kernel = (libxsmm_convfunction)handle->code_fwd[2].xconv.sconv;
int instr, n_segments; 
char *kernel_stream = handle->kernel_fwd_variant_ptrs[ltid];
int *code_stream;
int n_convs, conv_i, ifm1;
int img = handle->img_start[ltid];
#if 0
libxsmm_convfunction kernel_pool[4];
#endif

element_output_type *out = ((element_output_type*)handle->reg_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * (handle->ofmblock);
LIBXSMM_VLA_DECL(5, element_output_type, output, out, handle->blocksofm*handle->fm_lp_block, handle->ofhp, handle->ofwp, handle->ofmblock);
LIBXSMM_VLA_DECL(6, const element_input_type, input, (element_input_type*)handle->reg_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
LIBXSMM_VLA_DECL(7, const element_filter_type, weight, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);

const element_input_type *input_ptr;
element_input_type *copy_ptr;
element_input_type *prefetch_ptr;
const int padded_h = handle->ifhp + 2 * handle->desc.pad_h;
const int padded_w = handle->ifwp + 2 * handle->desc.pad_w;
LIBXSMM_VLA_DECL(5, element_input_type, input_buffer, ((element_input_type*)handle->scratch5) + ltid * handle->blocksifm * padded_h * padded_w * handle->ifmblock * handle->fm_lp_block, padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
libxsmm_xmatcopyfunction jitted_matcopy = handle->matcopy_fwd[0].xmatcopy;
libxsmm_xmatcopyfunction jitted_zero_overwrite = handle->matcopy_fwd[1].xmatcopy;

#if 0
kernel_pool[0] =  (libxsmm_convfunction)handle->code_fwd[0].xconv.sconv;
kernel_pool[1] =  (libxsmm_convfunction)handle->code_fwd[1].xconv.sconv;
kernel_pool[2] =  (libxsmm_convfunction)handle->code_fwd[2].xconv.sconv;
kernel_pool[3] =  (libxsmm_convfunction)handle->code_fwd[3].xconv.sconv;
#endif

if (handle->padding_flag == 1) {
  input_base  = &LIBXSMM_VLA_ACCESS(5, input_buffer, 0, 0, 0, 0, 0,
     padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
} else {
  input_base  = &LIBXSMM_VLA_ACCESS(6, input, 0, 0, 0, 0, 0, 0,
      handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
}
weight_base = &LIBXSMM_VLA_ACCESS(7, weight, 0, 0, 0, 0, 0, 0, 0,
    handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
output_base = &LIBXSMM_VLA_ACCESS(5, output, 0, 0, 0, 0, 0,
    handle->blocksofm*handle->fm_lp_block, handle->ofhp, handle->ofwp, handle->ofmblock);

instr = handle->n_entries_fwd[ltid];
n_segments = handle->n_fwd_code_segments[ltid];

#define IMG_LOOP_INIT 0
#define OFM_LOOP_INIT 1
#define OFM_LOOP_CLOSE 2
#define CONVOLUTION_KERNEL 3

if (n_segments) {
  code_stream = handle->fwd_code_segments[ltid]; 
  for (pc = 0; pc < 2*n_segments; pc += 2) {
    instr = code_stream[pc];
    n_convs = code_stream[pc+1];

    if (instr == IMG_LOOP_INIT) {
      /* Padding code via jitted matcopy kernel */
      for (ifm1 = handle->blocksifm-1; ifm1 >= 1; ifm1--) {
        input_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(6, input, img, ifm1, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
        copy_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_buffer, ifm1, handle->desc.pad_h, handle->desc.pad_w, 0, 0, padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
        prefetch_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(6, input, img, ifm1-1, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
        jitted_matcopy(input_ptr, NULL, copy_ptr, NULL, prefetch_ptr);
      }
      input_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(6, input, img, ifm1, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
      copy_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_buffer, ifm1, handle->desc.pad_h, handle->desc.pad_w, 0, 0, padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
      prefetch_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(6, input, img+1, handle->blocksifm-1, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
      jitted_matcopy(input_ptr, NULL, copy_ptr, NULL, prefetch_ptr);
      img++;

    } else if ( instr == OFM_LOOP_INIT ) {
      /* Overwrite output with zeros if requested */
      if ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) {
        jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);     
      }      
    } else {

    }

    for (conv_i = 0; conv_i < n_convs; conv_i++) {
      offset_i = stream[i];
      offset_w = stream[i+1];
      offset_o = stream[i+2];
      pi = stream[i+3];
      pw = stream[i+4];
      po = stream[i+5];
      kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po);
      i+=3;
#if 0 
      kernel_pool[kernel_stream[pc]]( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po);
#endif
    }
  }
} else {
  /* Run just the stream of convolutions  */
  for (pc = 0; pc < instr; pc++) {
    offset_i = stream[i];
    offset_w = stream[i+1];
    offset_o = stream[i+2];
    pi = stream[i+3];
    pw = stream[i+4];
    po = stream[i+5];
    kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po);
    i+=3;
#if 0 
    kernel_pool[kernel_stream[pc]]( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po);
#endif
  }
}

