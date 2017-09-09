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
#define IMG_LOOP_INIT 0
#define IFM_LOOP_INIT 1
#define IFM_LOOP_CLOSE 2
#define CONVOLUTION_KERNEL 3

const int ltid = tid-start_thread;

/* number of tasks for transpose that could be run in parallel */
const int transpose_work = handle->blocksofm * (handle->blocksifm * handle->fm_lp_block);
/* compute chunck size */
const int transpose_chunksize = (transpose_work % handle->desc.threads == 0) ? (transpose_work / handle->desc.threads) : ((transpose_work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
const int transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;
/* Pointer variables  */
const element_input_type *input_base;
element_input_type (* __restrict input_ptr);
const element_filter_type *weight_base;
element_filter_type *wt_trans_base, *wt_base;
element_output_type *output_base;
element_input_type (* __restrict copy_ptr);
element_input_type *prefetch_ptr;
const int padded_h = handle->ifhp + 2 * handle->desc.pad_h;
const int padded_w = handle->ifwp + 2 * handle->desc.pad_w;
LIBXSMM_VLA_DECL(3, element_input_type, input_buffer, ((element_input_type*)handle->scratch5) + ltid * padded_h * padded_w * handle->ifmblock, padded_w, handle->ifmblock);

libxsmm_convfunction kernel_bwd = (libxsmm_convfunction)handle->code_bwd[4].xconv.sconv;
libxsmm_convfunction kernel2_bwd = (libxsmm_convfunction)handle->code_bwd[5].xconv.sconv;
libxsmm_convfunction kernel_pool[2];
kernel_pool[0] = kernel_bwd;
kernel_pool[1] = kernel2_bwd;
char *variant = handle->kernel_bwd_variant_ptrs[ltid];

/* Input tensor declaration */
/* regular/high precision */
element_input_type* del_in = 0;
/* select pointer based on precision */
if (handle->datatype != handle->datatype_itm) {
  del_in = ((element_input_type*)handle->scratch7); /* + (handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in) * (handle->ifmblock); */
} else {
  del_in = ((element_input_type*)handle->grad_input->data); /* + (handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in) * (handle->ifmblock); */
}
{ /* open new scope for additional variable declarations (C89) */
  LIBXSMM_VLA_DECL(5, element_input_type, del_input, del_in, handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock);
  /* Ouput tensor declaration */
  element_output_type *const out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock * handle->fm_lp_block;
  LIBXSMM_VLA_DECL(6, element_output_type, del_out, out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);

  /* Weight and transpose_weight tensor declaration */
  LIBXSMM_VLA_DECL(7, element_filter_type, wt, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
  LIBXSMM_VLA_DECL(7, element_filter_type, tr_wt, (element_filter_type*)handle->scratch1, handle->blocksifm * handle->fm_lp_block, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block);
  LIBXSMM_VLA_DECL(7, element_filter_type, tr_wt2, (element_filter_type*)handle->scratch1, handle->blocksofm * handle->fm_lp_block, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block);


  /* Auxiliary integer variables   */
  int instr, n_segments, offset_i, offset_o, offset_w, pi, po, pw, pc, i,  n_convs, conv_i, ifm1, img = 0, ifm2, ij, ii, ifm1lpblock ;
  int ti, tj, trans_i, n_trans_tasks, trans_offset, trans_offset_dst;
  /* Stream related variables  */
  segment_t *code_stream;
  int *stream = handle->compute_bwd_indices_ptrs[ltid];
  int *trans_indices =  handle->transpose_bwd_indices_ptrs[ltid];
  element_filter_type  *mat, *matT;
  /* Kernel related variables  */
  libxsmm_convfunction kernel = (libxsmm_convfunction)handle->code_bwd[2].xconv.sconv;
  libxsmm_xmatcopyfunction jitted_matcopy = handle->matcopy_bwd[0].xmatcopy;
  libxsmm_xmatcopyfunction jitted_matcopyback = handle->matcopy_bwd[1].xmatcopy;
  /* Initialize base pointers */
  if ( handle->padding_flag == 1  ) {
    input_base = &LIBXSMM_VLA_ACCESS(3, input_buffer , 0, 0, 0,
        padded_w, handle->ifmblock);
    copy_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(3, input_buffer, handle->desc.pad_h, handle->desc.pad_w, 0, padded_w, handle->ifmblock);
    input_ptr = NULL;
    memset(&LIBXSMM_VLA_ACCESS(3, input_buffer, 0, 0, 0, padded_w, handle->ifmblock), 0, padded_w * padded_h * handle->ifmblock * sizeof(element_input_type));
  } else {
    input_base = &LIBXSMM_VLA_ACCESS(5, del_input, 0, 0, 0, 0, 0,
        handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock);
    copy_ptr = NULL;
  }
  weight_base = &LIBXSMM_VLA_ACCESS(7, tr_wt, 0, 0, 0, 0, 0, 0, 0,
      handle->blocksifm * handle->fm_lp_block, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block);
  output_base = &LIBXSMM_VLA_ACCESS(6, del_out, 0, 0, 0, 0, 0, 0,
      handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);
  wt_trans_base = &LIBXSMM_VLA_ACCESS(7, tr_wt, 0, 0, 0, 0, 0, 0, 0,
      handle->blocksifm * handle->fm_lp_block, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block);
  wt_base = &LIBXSMM_VLA_ACCESS(7, wt, 0, 0, 0, 0, 0, 0, 0,
      handle->blocksifm * handle->fm_lp_block, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block);

  instr = handle->n_entries_bwd[ltid];
  n_segments = handle->n_bwd_code_segments[ltid];
  i = 0;
  code_stream = handle->bwd_code_segments[ltid];
  n_trans_tasks =  handle->n_entries_trans_bwd[ltid];

  if ( handle->desc.R == 1 && handle->desc.S == 1 && handle->use_nts_bwd == 1 ) {
    output_base = &LIBXSMM_VLA_ACCESS(5, del_input, 0, 0, 0, 0, 0,
        handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock);
    copy_ptr = NULL;
    input_base = &LIBXSMM_VLA_ACCESS(6, del_out, 0, 0, 0, 0, 0, 0,
        handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);
  }

  /* lazy barrier init */
  libxsmm_barrier_init(handle->barrier, ltid);

  if ( handle->desc.R == 1 && handle->desc.S == 1 && handle->use_nts_bwd == 1 ) {
    for ( trans_i = 0; trans_i < n_trans_tasks; trans_i += 2 ) {
      trans_offset = trans_indices[trans_i];
      trans_offset_dst = trans_indices[trans_i+1];
      mat = wt_base + trans_offset;
      matT = wt_trans_base + trans_offset_dst;
#include "transpose.tpl.c"
      /*
      for (ti = 0; ti < handle->ofmblock; ti++) {
        for (tj = 0; tj < handle->ifmblock; tj++) {
          matT[ti+handle->ifmblock*tj] = mat[tj+handle->ofmblock*ti];
        }
      }
      */
    }
#if 0
    int ifm1ofm1, kj, ki, ofm1, ifm1, ifm2, ofm2;
    for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
      ofm1 = ifm1ofm1 / handle->blocksifm;
      ifm1 = ifm1ofm1 % handle->blocksifm;
      for (kj=0; kj < handle->desc.R; ++kj) {
        for (ki=0; ki < handle->desc.S; ++ki) {
          /* TODO: enable this later */
          /*transpose<VLEN,VLEN>(&wt[ofm1][ifm1][kj][ki][0][0],&tr_wt[ofm1][ifm1][kj][ki][0][0]);*/
          for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
            for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
              LIBXSMM_VLA_ACCESS(7, tr_wt2, ifm1, ofm1, kj, ki, ofm2, ifm2, 0, handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block) =
                LIBXSMM_VLA_ACCESS(7, wt, ofm1, ifm1, kj, ki, ifm2, ofm2, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
            }
          }
        }
      }
    }
#endif
  } else {
    for ( trans_i = 0; trans_i < n_trans_tasks; trans_i++ ) {
      trans_offset = trans_indices[trans_i];
      mat = wt_base + trans_offset;
      matT = wt_trans_base + trans_offset;
#include "transpose.tpl.c"
      /*
      for (ti = 0; ti < handle->ofmblock; ti++) {
        for (tj = 0; tj < handle->ifmblock; tj++) {
          matT[ti+handle->ifmblock*tj] = mat[tj+handle->ofmblock*ti];
        }
      }
      */
    }
  }

  
  libxsmm_barrier_wait(handle->barrier, ltid);

  if ( handle->desc.R == 1 && handle->desc.S == 1 && handle->use_nts_bwd == 1 ) {
    if (handle->ofwp == 7) {
      for (conv_i = 0; conv_i < instr; conv_i++) {
        offset_i = stream[i];
        offset_w = stream[i+1];
        offset_o = stream[i+2];
        pi = stream[i+3];
        pw = stream[i+4];
        po = stream[i+5];
        kernel_pool[variant[conv_i]]( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po);
        i+=3;
      }
    } else {
      for (conv_i = 0; conv_i < instr; conv_i++) {
        offset_i = stream[i];
        offset_w = stream[i+1];
        offset_o = stream[i+2];
        pi = stream[i+3];
        pw = stream[i+4];
        po = stream[i+5];
        kernel_bwd( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po);
        i+=3;
      }
    }
  } else {
    for (pc = 0; pc < n_segments; pc++) {
      instr = code_stream[pc].segment_type;
      n_convs = code_stream[pc].n_convs;

      if (instr == IMG_LOOP_INIT) {
        img = code_stream[pc].aux_index;
      }

      if ( instr == IFM_LOOP_CLOSE) {
        ifm1 = code_stream[pc].aux_index;
        ifm1lpblock = ifm1;
        if ( handle->padding_flag == 1 ) {
          input_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, 0, 0, 0, handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock);
          jitted_matcopyback(copy_ptr, NULL, input_ptr, NULL, NULL);
        } else {
#include "libxsmm_dnn_zero_rim_st_input_custom.tpl.c"
        }
      }

      if ( instr == IFM_LOOP_INIT )  {
        ifm1 = code_stream[pc].aux_index;
        if ( ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ) {
          element_input_type* temp_ptr;
          if ( handle->padding_flag == 1 ) {
            temp_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(3, input_buffer, 0, 0, 0, padded_w, handle->ifmblock);
            LIBXSMM_PRAGMA_SIMD
              for (ij = 0; ij < padded_h * padded_w * handle->ifmblock; ij++) {
                temp_ptr[ij] = (element_output_type)0;
              }   
          } else {
            temp_ptr = &(LIBXSMM_VLA_ACCESS(  5, del_input, img, ifm1, 0, 0, 0, handle->blocksifm*handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock));
            LIBXSMM_PRAGMA_SIMD
              for (ij = 0; ij < handle->ifhp*handle->ifwp*handle->ifmblock*handle->fm_lp_block; ij++) {
                temp_ptr[ij] = (element_output_type)0;
              }
          }
        } else if ( handle->padding_flag == 1 ) {
          input_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, 0, 0, 0, handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock);
          if (ifm1+1 != handle->blocksifm * handle->fm_lp_block) {
            /* Prefetch next ifm1, same image */
            prefetch_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1+1, 0, 0, 0, handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock);
          } else {
            /* Prefetch ifm1  0 from next image */
            prefetch_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, del_input, img+1, 0, 0, 0, 0, handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock);
          }
          jitted_matcopy(input_ptr, NULL, copy_ptr, NULL, prefetch_ptr);
        }
      }

      /* Run the stream of convolutions for this segment */
      for (conv_i = 0; conv_i < n_convs; conv_i++) {
        offset_i = stream[i];
        offset_w = stream[i+1];
        offset_o = stream[i+2];
        pi = stream[i+3];
        pw = stream[i+4];
        po = stream[i+5];
        kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po);
        i+=3;
      }
    }
  }

  libxsmm_barrier_wait(handle->barrier, ltid);
}
