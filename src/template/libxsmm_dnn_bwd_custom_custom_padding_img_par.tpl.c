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

/* Padding code via jitted matcopy kernel */
img = code_stream[pc].aux_index;
for (ofm1 = handle->blocksofm-1; ofm1 >= 1; ofm1--) {
  for (ih = input_h_start; ih < input_h_end; ih++) {
    input_ptr = (element_output_type*)&LIBXSMM_VLA_ACCESS(6, del_out, img, ofm1, ih, 0, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block);
    copy_ptr = (element_output_type*)&LIBXSMM_VLA_ACCESS(5, output_buffer, ofm1, handle->desc.pad_h+ih, handle->desc.pad_w, 0, 0, padded_h, padded_w, handle->ofmblock_lp, handle->fm_lp_block);
    prefetch_ptr = (element_output_type*)&LIBXSMM_VLA_ACCESS(6, del_out, img, ofm1-1, ih, 0, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block);
    jitted_matcopy(input_ptr, NULL, copy_ptr, NULL, prefetch_ptr);
  }
}
for (ih = input_h_start; ih < input_h_end; ih++) {
  input_ptr = (element_output_type*)&LIBXSMM_VLA_ACCESS(6, del_out, img, ofm1, ih, 0, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block);
  copy_ptr = (element_output_type*)&LIBXSMM_VLA_ACCESS(5, output_buffer, ofm1, handle->desc.pad_h+ih, handle->desc.pad_w, 0, 0, padded_h, padded_w, handle->ofmblock_lp, handle->fm_lp_block);
  prefetch_ptr = (element_output_type*)&LIBXSMM_VLA_ACCESS(6, del_out, img+1, handle->blocksofm-1, ih, 0, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock_lp, handle->fm_lp_block);
  jitted_matcopy(input_ptr, NULL, copy_ptr, NULL, prefetch_ptr);
}

