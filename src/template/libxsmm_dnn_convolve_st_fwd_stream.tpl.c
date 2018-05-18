/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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
libxsmm_convfunction kernel = (libxsmm_convfunction)handle->code_fwd[0].xconv.sconv;
int instr;
char *kernel_stream = handle->kernel_fwd_variant_ptrs[ltid];
#if 0
libxsmm_convfunction kernel_pool[4];
#endif

element_output_type *out = ((element_output_type*)handle->reg_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * (handle->ofmblock);
LIBXSMM_VLA_DECL(5, element_output_type, output, out, handle->blocksofm*handle->fm_lp_block, handle->ofhp, handle->ofwp, handle->ofmblock);
LIBXSMM_VLA_DECL(6, const element_input_type, input, (element_input_type*)handle->reg_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
LIBXSMM_VLA_DECL(7, const element_filter_type, weight, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);

#if 0
kernel_pool[0] =  (libxsmm_convfunction)handle->code_fwd[0].xconv.sconv;
kernel_pool[1] =  (libxsmm_convfunction)handle->code_fwd[1].xconv.sconv;
kernel_pool[2] =  (libxsmm_convfunction)handle->code_fwd[2].xconv.sconv;
kernel_pool[3] =  (libxsmm_convfunction)handle->code_fwd[3].xconv.sconv;
#endif

input_base  = &LIBXSMM_VLA_ACCESS(6, input, 0, 0, 0, 0, 0, 0,
    handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
weight_base = &LIBXSMM_VLA_ACCESS(7, weight, 0, 0, 0, 0, 0, 0, 0,
    handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
output_base = &LIBXSMM_VLA_ACCESS(5, output, 0, 0, 0, 0, 0,
    handle->blocksofm*handle->fm_lp_block, handle->ofhp, handle->ofwp, handle->ofmblock);

#if defined(KINPUT_PADDING)
for (seg = 0; seg < n_segments; seg++) {
  /* Add copy code here  */

  /* Assign here the instr to be thr proper segment size  */

#else
  instr = handle->n_entries_fwd[ltid];
#endif
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
#if defined(KINPUT_PADDING)
}
#endif

