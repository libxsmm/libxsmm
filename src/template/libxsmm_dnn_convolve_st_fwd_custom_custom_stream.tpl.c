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
#define OFM_LOOP_INIT 1
#define OFM_LOOP_CLOSE 2
#define CONVOLUTION_KERNEL 3

const int ltid = tid-start_thread;
int gs = 72; /*atoi(getenv("GSIZE"));*/
const int tile_id = ltid/gs;
/* Pointer variables  */
const element_input_type *input_base, *input_ptr;
const element_filter_type *weight_base;
element_output_type *output_base;
element_input_type *copy_ptr, *prefetch_ptr;
element_output_type *out = ((element_output_type*)handle->reg_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * (handle->ofmblock);
LIBXSMM_VLA_DECL(5, element_output_type, output, out, handle->blocksofm*handle->fm_lp_block, handle->ofhp, handle->ofwp, handle->ofmblock);
LIBXSMM_VLA_DECL(6, const element_input_type, input, (element_input_type*)handle->reg_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
/* LIBXSMM_VLA_DECL(7, const element_filter_type, weight, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);*/
LIBXSMM_VLA_DECL(7, const element_filter_type, weight, (element_filter_type*)handle->reg_filter->data + tile_id * handle->blocksifm * handle->blocksofm * handle->ifmblock * handle->ofmblock * handle->fm_lp_block *  handle->desc.R * handle->desc.S, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);

/* Auxiliary integer variables   */
int instr, n_segments, offset_i, offset_o, offset_w, pi, po, pw, pc, i, ih, n_convs, conv_i, ifm1, ofm1, ofm2, oj, img, input_h_start, input_h_end, my_h_out;
/* Stream related variables  */
segment_t *code_stream;
int *stream = handle->compute_fwd_indices_ptrs[ltid];
/* Padding related variables */
const int padded_h = handle->ifhp + 2 * handle->desc.pad_h;
const int padded_w = handle->ifwp + 2 * handle->desc.pad_w;
LIBXSMM_VLA_DECL(5, element_input_type, input_buffer, ((element_input_type*)handle->scratch5) + ltid * handle->blocksifm * padded_h * padded_w * handle->ifmblock * handle->fm_lp_block, padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
/* Kernel related variables  */
libxsmm_xmatcopyfunction jitted_matcopy = handle->matcopy_fwd[0].xmatcopy;
libxsmm_xmatcopyfunction jitted_zero_overwrite = handle->matcopy_fwd[1].xmatcopy;
libxsmm_convfunction kernel = (libxsmm_convfunction)handle->code_fwd[2].xconv.sconv;
libxsmm_convfunction kernel2 = (libxsmm_convfunction)handle->code_fwd[3].xconv.sconv;
libxsmm_convfunction kernel_pool[2];
kernel_pool[0] = kernel;
kernel_pool[1] = kernel2;
char *variant = handle->kernel_fwd_variant_ptrs[ltid];
int pool_index = 0;

/* Initialize base pointers */
if (handle->padding_flag == 1) {
  input_base = &LIBXSMM_VLA_ACCESS(5, input_buffer, 0, 0, 0, 0, 0,
      padded_h, padded_w, handle->ifmblock, handle->fm_lp_block);
} else {
  input_base = &LIBXSMM_VLA_ACCESS(6, input, 0, 0, 0, 0, 0, 0,
      handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
}
weight_base = &LIBXSMM_VLA_ACCESS(7, weight, 0, 0, 0, 0, 0, 0, 0,
    handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
output_base = &LIBXSMM_VLA_ACCESS(5, output, 0, 0, 0, 0, 0,
    handle->blocksofm*handle->fm_lp_block, handle->ofhp, handle->ofwp, handle->ofmblock);

instr = handle->n_entries_fwd[ltid];
n_segments = handle->n_fwd_code_segments[ltid];
i = 0;
/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

if (n_segments) {
  /* We have segmented the stream of convolutions since we need to inject different functionalities...  */
  code_stream = handle->fwd_code_segments[ltid];
  /* If we are in the img_par execution then avoid fine-grained copy in case of padding...  */
  if (handle->desc.N*handle->blocksofm >= handle->desc.threads) {
    if (handle->ofw == 7) {
      for (pc = 0; pc < n_segments; pc++) {
        instr = code_stream[pc].segment_type;
        n_convs = code_stream[pc].n_convs;

        if (instr == IMG_LOOP_INIT) {
          img = code_stream[pc].aux_index;
          /* Apply padding  */
          if (handle->padding_flag == 1) {
#include "libxsmm_dnn_fwd_custom_custom_padding.tpl.c"
          }
        }
        
        if ( instr == OFM_LOOP_INIT ) {
          /* Apply bias if requested  */
          if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"
          }
          /* Overwrite output with zeros if requested */
          if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_fwd != 1) ) {
            jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
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
          kernel_pool[variant[pool_index]]( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po);
          pool_index++;
          i+=3;
        }
      }
    } else {
      for (pc = 0; pc < n_segments; pc++) {
        instr = code_stream[pc].segment_type;
        n_convs = code_stream[pc].n_convs;
        if (instr == IMG_LOOP_INIT) {
          img = code_stream[pc].aux_index;
          /* Apply padding  */
          if (handle->padding_flag == 1) {
#include "libxsmm_dnn_fwd_custom_custom_padding.tpl.c"
          }
        }

        if ( instr == OFM_LOOP_INIT ) {
          /* Apply bias if requested  */
          if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"
          }
          /* Overwrite output with zeros if requested */
          if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_fwd != 1) ) {
            jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
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

  } else {
    /* Use fine-grained operations since we are in the img_par path, so update relevant kernel pointers... */
    jitted_matcopy = handle->matcopy_fwd[2].xmatcopy;
    jitted_zero_overwrite = handle->matcopy_fwd[3].xmatcopy;
    input_h_start = LIBXSMM_MAX(0,  handle->ofh_fwd_start[ltid] - handle->desc.R + 1);
    input_h_end = LIBXSMM_MIN( handle->ifhp, (handle->ofh_fwd_end[ltid] + handle->desc.R -1) * handle->desc.u ) ;
    my_h_out = handle->ofh_fwd_end[ltid]-handle->ofh_fwd_start[ltid];
    for (pc = 0; pc < n_segments; pc++) {
      instr = code_stream[pc].segment_type;
      n_convs = code_stream[pc].n_convs;
      if (instr == IMG_LOOP_INIT) {
        /* Padding code via jitted matcopy kernel */
#include "libxsmm_dnn_fwd_custom_custom_padding_img_par.tpl.c"
      }

      if ( instr == OFM_LOOP_INIT ) {
        /* Apply bias if requested  */
        if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
#include "libxsmm_dnn_fwd_custom_custom_bias_img_par.tpl.c"
        }
        /* Overwrite output with zeros if requested */
        if ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) {
          for ( ih = 0; ih < my_h_out * handle->ofmblock * handle->ofwp; ih += handle->ofmblock * handle->ofwp) {
            jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2] + ih, NULL, NULL);
          }
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
} else {
  /* Run the stream of convolutions, no extra operations are required... */
  if (handle->ofw == 7) {
    for (pc = 0; pc < instr; pc+=1) {
      offset_i = stream[i];
      offset_w = stream[i+1]; 
      offset_o = stream[i+2];
      pi = stream[i+3];
      pw = stream[i+4];
      po = stream[i+5];
      kernel_pool[variant[pc]]( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po);
      i+=3;  
    }
  } else { 
    for (pc = 0; pc < instr; pc++) {
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
