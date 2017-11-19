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

int BLOCKSIFM = handle->blocksifm_lp;
int BLOCKSOFM = handle->blocksofm_lp;
const int oKB = handle->desc.K/32;
const int iCB = handle->desc.C/32;

/* number of tasks for transpose that could be run in parallel */
int transpose_work;
if (handle->use_lp_kernel == 0) {
  transpose_work = BLOCKSOFM * (BLOCKSIFM * handle->fm_lp_block);
} else {
#if 0
  transpose_work = handle->desc.C * handle->desc.K;
#endif
  transpose_work = oKB * iCB;
}

/* compute chunck size */
const int transpose_chunksize = (transpose_work % handle->desc.threads == 0) ? (transpose_work / handle->desc.threads) : ((transpose_work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
const int transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;
/* Pointer variables  */
element_output_type *input_base;
element_output_type *input_ptr;
element_filter_type *weight_base;
element_input_type *output_base;
element_output_type *copy_ptr;
element_output_type *prefetch_ptr;

/* Padding related variables */
const int padded_h = handle->ofhp + 2 * handle->desc.pad_h;
const int padded_w = handle->ofwp + 2 * handle->desc.pad_w;
LIBXSMM_VLA_DECL(5, element_output_type, output_buffer, ((element_output_type*)handle->scratch5) + ltid * BLOCKSOFM * padded_h * padded_w * handle->ofmblock * handle->fm_lp_block, padded_h, padded_w, handle->ofmblock, handle->fm_lp_block);

libxsmm_xmatcopyfunction jitted_matcopy = handle->matcopy_bwd[0].xmatcopy;
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
if (handle->datatype_in != handle->datatype_out) {
  del_in = ((element_input_type*)handle->grad_input->data) + (handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in) * (handle->ifmblock);
} else {
  del_in = ((element_input_type*)handle->grad_input->data) + (handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in) * (handle->ifmblock); 
}

float scale_factor __attribute__((aligned(64)));
if (handle->use_lp_kernel == 1) {
  scale_factor = (float) pow(2.0, -1.0*((double)(handle->reg_filter->scf + handle->grad_output->scf)));
}

float *max_vals  __attribute__((aligned(64)));
__m512 max_abs;
if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {
  LIBXSMM_VLA_DECL(2, float, maxstats, handle->maxstats_bwd->data, handle->ifmblock);
  max_vals = (float*) &LIBXSMM_VLA_ACCESS(2, maxstats, ltid, 0, handle->ofmblock);
  max_abs = _mm512_setzero_ps();
  _mm512_store_ps(max_vals, max_abs);
}

{ /* open new scope for additional variable declarations (C89) */
  LIBXSMM_VLA_DECL(5, element_input_type, del_input, del_in, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
  /* Ouput tensor declaration */
  element_output_type *const out = ((element_output_type*)handle->grad_output->data) /* + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock * handle->fm_lp_block*/;
  LIBXSMM_VLA_DECL(6, element_output_type, del_out, out, BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);

  /* Weight and transpose_weight tensor declaration */
  LIBXSMM_VLA_DECL(7, element_filter_type, wt, (element_filter_type*)handle->reg_filter->data, BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
  LIBXSMM_VLA_DECL(7, element_filter_type, tr_wt2, (element_filter_type*)handle->scratch1, BLOCKSOFM, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block);

  /* Auxiliary integer variables   */
  int instr, n_segments, offset_i, offset_o, offset_w, pi, po, pw, pc, i,  n_convs, conv_i, ifm1, img = 0, ifm2, ij, ii, ifm1lpblock ;
  int ti, tj, trans_i, n_trans_tasks, trans_offset, trans_offset_dst;
  /* Stream related variables  */
  segment_t *code_stream;
  int *stream = handle->compute_bwd_indices_ptrs[ltid];
  int *trans_indices =  handle->transpose_bwd_indices_ptrs[ltid];
  int pool_index;
  element_filter_type  *mat, *matT;
  int ifm1ofm1, kj, ki, ofm2, ofm1;
  /* Kernel related variables  */
  libxsmm_convfunction kernel = (libxsmm_convfunction)handle->code_bwd[4].xconv.sconv;
  libxsmm_xmatcopyfunction jitted_matcopy = handle->matcopy_bwd[0].xmatcopy;
  libxsmm_xmatcopyfunction jitted_zero_overwrite = handle->matcopy_bwd[1].xmatcopy;

  /* Initialize base pointers */
  if ( handle->padding_flag == 1  ) {
    input_base = &LIBXSMM_VLA_ACCESS(5, output_buffer, 0, 0, 0, 0, 0,
        padded_h, padded_w, handle->ofmblock, handle->fm_lp_block);
    /* we need to set the scratch to zero */
    /* @TODO: we need to find a better/faster code here, e.g. just setting the rim */
    memset( input_base, 0, BLOCKSOFM * padded_h * padded_w * handle->ofmblock * handle->fm_lp_block * sizeof(element_output_type) );
  } else {
    input_base = &LIBXSMM_VLA_ACCESS(6, del_out, 0, 0, 0, 0, 0, 0,
        BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);
  }

  output_base = &LIBXSMM_VLA_ACCESS(5, del_input, 0, 0, 0, 0, 0,
      handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
  weight_base = &LIBXSMM_VLA_ACCESS(7, tr_wt2, 0, 0, 0, 0, 0, 0, 0,
      BLOCKSOFM, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block);

  instr = handle->n_entries_bwd[ltid];
  n_segments = handle->n_bwd_code_segments[ltid];
  i = 0;
  code_stream = handle->bwd_code_segments[ltid];
  n_trans_tasks =  handle->n_entries_trans_bwd[ltid];

  /* lazy barrier init */
  libxsmm_barrier_init(handle->barrier, ltid);

  if ( (handle->options & LIBXSMM_DNN_CONV_OPTION_BWD_NO_FILTER_TRANSPOSE) > 0 ) {
    weight_base = (element_filter_type*)handle->reg_filter_tr->data;
  } else {
    if (handle->use_lp_kernel == 0) {
      for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
        ofm1 = ifm1ofm1 / BLOCKSIFM;
        ifm1 = ifm1ofm1 % BLOCKSIFM;
        for (kj=0; kj < handle->desc.R; kj++) {
          for (ki=0; ki < handle->desc.S; ki++) {
            /* TODO: enable this later */
            /*transpose<VLEN,VLEN>(&wt[ofm1][ifm1][kj][ki][0][0],&tr_wt[ofm1][ifm1][kj][ki][0][0]);*/
            for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
              for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                LIBXSMM_VLA_ACCESS(7, tr_wt2, ifm1, ofm1, handle->desc.R-1-kj , handle->desc.S-1-ki, ofm2, ifm2, 0, BLOCKSOFM, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block) =
                  LIBXSMM_VLA_ACCESS(7, wt, ofm1, ifm1, kj, ki, ifm2, ofm2, 0, BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
              }
            }
          }
        }
      }
    } else {
#if 0
      int k, c, r, s;
      for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
        k = ifm1ofm1 / handle->desc.C;
        c = ifm1ofm1 % handle->desc.C;
        for ( r = 0; r < handle->desc.R; r++ ) {
          for ( s = 0; s < handle->desc.S; s++ ) {
            int i_c1, i_c2, i_c3, i_k1, i_k2;
            int o_c1, o_c2, o_k1, o_k2, o_k3;
            o_k1 = k/32;
            o_k2 = (k%32)/2;
            o_k3 = (k%32)%2;
            o_c1 = c/16;
            o_c2 = c%16;
            i_c1 = c/32;
            i_c2 = (c%32)/2;
            i_c3 = (c%32)%2;
            i_k1 = k/16;
            i_k2 = k%16;
            LIBXSMM_VLA_ACCESS(7, tr_wt2, o_c1, o_k1, handle->desc.R-1-r , handle->desc.S-1-s, o_k2, o_c2, o_k3, BLOCKSOFM, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block) =
              LIBXSMM_VLA_ACCESS(7, wt, i_k1, i_c1, r, s, i_c2, i_k2, i_c3, BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
          }
        }  
      }
#endif
      int icb, okb, kkb, ocbb, ikbb;
      element_filter_type  * __restrict o_b;
      const element_filter_type *__restrict i_b;
      const __m512i vgindex_base = _mm512_set_epi32(7,6,5,4,3,2,1,0,7,6,5,4,3,2,1,0);
      const __m512i vgindex_hi_offs = _mm512_mullo_epi32(_mm512_set1_epi32(2), _mm512_set_epi32(1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0));
      const __m512i vgindex = _mm512_add_epi32(vgindex_hi_offs, _mm512_mullo_epi32(_mm512_set1_epi32(16), vgindex_base));
      const __m512i vsindex_base = _mm512_mullo_epi32(_mm512_set1_epi32(2), _mm512_set_epi32(7,6,5,4,3,2,1,0,7,6,5,4,3,2,1,0));
      const __m512i vsindex_hi_offs = _mm512_mullo_epi32(_mm512_set1_epi32(16), _mm512_set_epi32(1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0));
      const __m512i vsindex = _mm512_add_epi32(vsindex_hi_offs, vsindex_base);

      for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
        icb = ifm1ofm1 / oKB;
        okb = ifm1ofm1 % oKB;
        for (kj=0; kj < handle->desc.R; kj++) {
          for (ki=0; ki < handle->desc.S; ki++) {
            for(ocbb = 0; ocbb < 2; ++ocbb) {
              for(ikbb = 0; ikbb < 2; ++ikbb) {
                i_b = &LIBXSMM_VLA_ACCESS(7, wt, okb*2+ikbb, icb, kj, ki, ocbb*8, 0, 0, BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block); 
                o_b = &LIBXSMM_VLA_ACCESS(7, tr_wt2, icb*2+ocbb, okb, handle->desc.R-1-kj , handle->desc.S-1-ki, ikbb*8, 0, 0, BLOCKSOFM, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block);
#pragma unroll(4)
                for(kkb = 0; kkb < 4; ++kkb) {
                  const __m512i inp = _mm512_i32gather_epi32(vgindex, i_b + 8*kkb, 4);
                  const __m512i inp2 = _mm512_i32gather_epi32(vgindex, i_b + 8*kkb+2, 4);
                  const __m512i zeros = _mm512_or_epi32(_mm512_and_epi32(inp, _mm512_set1_epi32(0x0000FFFF)), _mm512_slli_epi32(inp2, 16));
                  const __m512i ones = _mm512_or_epi32(_mm512_and_epi32(inp2, _mm512_set1_epi32(0xFFFF0000)), _mm512_srli_epi32(inp, 16));
                  _mm512_i32scatter_epi32(o_b + (kkb*4/2)*32,     vsindex, zeros, 4);
                  _mm512_i32scatter_epi32(o_b + (kkb*4/2)*32 + 2, vsindex, ones, 4);
                }
              }
            }
          }
        }
      }
    }
    weight_base = &LIBXSMM_VLA_ACCESS(7, tr_wt2, 0, 0, 0, 0, 0, 0, 0,
        BLOCKSOFM, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block);
    libxsmm_barrier_wait(handle->barrier, ltid);
  }

  pool_index = 0;
  i = 0;

  if (n_segments) {
    /* We have segmented the stream of convolutions since we need to inject different functionalities...  */
    code_stream = handle->bwd_code_segments[ltid];
    if (handle->perform_relu_in_kernel == 1) {/* do RELU stuff in the kernel  */
      LIBXSMM_VLA_DECL(5, element_input_type, original_input, ((element_input_type*)handle->reg_input->data) + (handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in * handle->ifmblock), handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      element_input_type *regular_input_base;
      regular_input_base = &LIBXSMM_VLA_ACCESS(5, original_input, 0, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);

      if (handle->ofw == 7) {
        for (pc = 0; pc < n_segments; pc++) {
          instr = code_stream[pc].segment_type;
          n_convs = code_stream[pc].n_convs;

          if (instr == IMG_LOOP_INIT) {
            img = code_stream[pc].aux_index;
            /* Apply padding  */
            if (handle->padding_flag == 1) {
#include "libxsmm_dnn_bwd_custom_custom_padding.tpl.c"
            }
          }

          if ( instr == IFM_LOOP_INIT ) {
            /* Apply bias if requested  */
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
              /*#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"*/
            }
            /* Overwrite output with zeros if requested */
            if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_bwd == 0) ) {
              jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
            }
          }

          if ( instr == IFM_LOOP_CLOSE) {
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {     
              ifm1 =  code_stream[pc].aux_index;
              element_input_type* cur_vec = &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, 0, 0, 0,
                  handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              for ( ij = 0; ij < handle->desc.H; ij++ ) {
                for ( ii = 0; ii < handle->desc.W*handle->ifmblock; ii+=16 ) {
                  max_abs = _mm512_max_ps(max_abs, _mm512_abs_ps(_mm512_load_ps(cur_vec+ii)));
                }
                cur_vec += handle->ifwp*handle->ifmblock;
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
            kernel_pool[variant[pool_index]]( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, regular_input_base + offset_o, &scale_factor, max_vals);
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
#include "libxsmm_dnn_bwd_custom_custom_padding.tpl.c"
            }
          }

          if ( instr == IFM_LOOP_INIT ) {
            /* Apply bias if requested  */
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
              /*#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"*/
            }
            /* Overwrite output with zeros if requested */
            if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_bwd == 0) ) {
              jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
            }
          }

          if ( instr == IFM_LOOP_CLOSE) {
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {     
              ifm1 =  code_stream[pc].aux_index;
              element_input_type* cur_vec = &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, 0, 0, 0,
                  handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              for ( ij = 0; ij < handle->desc.H; ij++ ) {
                for ( ii = 0; ii < handle->desc.W*handle->ifmblock; ii+=16 ) {
                  max_abs = _mm512_max_ps(max_abs, _mm512_abs_ps(_mm512_load_ps(cur_vec+ii)));
                }
                cur_vec += handle->ifwp*handle->ifmblock;
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
            kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, regular_input_base + offset_o, &scale_factor, max_vals);
            i+=3;
          }
        }
      }
    } else { /* We don't do RELU stuff in the kernel  */
      if (handle->ofw == 7) {
        for (pc = 0; pc < n_segments; pc++) {
          instr = code_stream[pc].segment_type;
          n_convs = code_stream[pc].n_convs;

          if (instr == IMG_LOOP_INIT) {
            img = code_stream[pc].aux_index;
            /* Apply padding  */
            if (handle->padding_flag == 1) {
#include "libxsmm_dnn_bwd_custom_custom_padding.tpl.c"
            }
          }

          if ( instr == IFM_LOOP_INIT ) {
            /* Apply bias if requested  */
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
              /*#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"*/
            }
            /* Overwrite output with zeros if requested */
            if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_bwd == 0) ) {
              jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
            }
          } 

          if ( instr == IFM_LOOP_CLOSE ) {
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_RELU_BWD) > 0) {   
              ifm1 = code_stream[pc].aux_index; 
              LIBXSMM_VLA_DECL(5, element_input_type, input, (element_input_type*) handle->reg_input->data,  handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              LIBXSMM_VLA_DECL(5, element_input_type, del_input_2, (element_input_type*) handle->grad_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);  
              element_input_type *orig_input_ptr;
              element_input_type *del_input_ptr;
              __m512 zero_reg  = _mm512_setzero_ps();  
              __m512 orig_reg;
              __mmask16 mask; 
              orig_input_ptr = &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              del_input_ptr = &LIBXSMM_VLA_ACCESS(5, del_input_2, img, ifm1, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              for (ij = 0; ij < handle->desc.H; ij++) {
                for (ii = 0; ii < handle->desc.W * 16; ii += 16) {
                  orig_reg  = _mm512_load_ps(orig_input_ptr + ii);
                  mask = _mm512_cmp_ps_mask(zero_reg, orig_reg, _CMP_EQ_OQ);
                  _mm512_mask_storeu_ps(del_input_ptr + ii, mask, zero_reg);
                }
                orig_input_ptr += handle->ifwp * 16;
                del_input_ptr += handle->ifwp *16;
              }
            }

            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {     
              ifm1 =  code_stream[pc].aux_index;
              element_input_type* cur_vec = &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, 0, 0, 0,
                  handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              for ( ij = 0; ij < handle->desc.H; ij++ ) {
                for ( ii = 0; ii < handle->desc.W*handle->ifmblock; ii+=16 ) {
                  max_abs = _mm512_max_ps(max_abs, _mm512_abs_ps(_mm512_load_ps(cur_vec+ii)));
                }
                cur_vec += handle->ifwp*handle->ifmblock;
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
            kernel_pool[variant[pool_index]]( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
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
#include "libxsmm_dnn_bwd_custom_custom_padding.tpl.c"
            }
          }

          if ( instr == IFM_LOOP_INIT ) {
            /* Apply bias if requested  */
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) {
              /*#include "libxsmm_dnn_fwd_custom_custom_bias.tpl.c"*/
            }
            /* Overwrite output with zeros if requested */
            if (((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_bwd == 0) ) {
              jitted_zero_overwrite(NULL, NULL, output_base + stream[i+2], NULL, NULL);
            }
          }

          if ( instr == IFM_LOOP_CLOSE ) {
            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_RELU_BWD) > 0) {   
              ifm1 = code_stream[pc].aux_index; 
              LIBXSMM_VLA_DECL(5, element_input_type, input, (element_input_type*) handle->reg_input->data,  handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              LIBXSMM_VLA_DECL(5, element_input_type, del_input_2, (element_input_type*) handle->grad_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);  
              element_input_type *orig_input_ptr;
              element_input_type *del_input_ptr;
              __m512 zero_reg  = _mm512_setzero_ps();  
              __m512 orig_reg;
              __mmask16 mask; 
              orig_input_ptr = &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              del_input_ptr = &LIBXSMM_VLA_ACCESS(5, del_input_2, img, ifm1, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              for (ij = 0; ij < handle->desc.H; ij++) {
                for (ii = 0; ii < handle->desc.W * 16; ii += 16) {
                  orig_reg  = _mm512_load_ps(orig_input_ptr + ii);
                  mask = _mm512_cmp_ps_mask(zero_reg, orig_reg, _CMP_EQ_OQ);
                  _mm512_mask_storeu_ps(del_input_ptr + ii, mask, zero_reg);
                }
                orig_input_ptr += handle->ifwp * 16;
                del_input_ptr += handle->ifwp *16;
              }
            }

            if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {     
              ifm1 =  code_stream[pc].aux_index;
              element_input_type* cur_vec = &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, 0, 0, 0,
                  handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
              for ( ij = 0; ij < handle->desc.H; ij++ ) {
                for ( ii = 0; ii < handle->desc.W*handle->ifmblock; ii+=16 ) {
                  max_abs = _mm512_max_ps(max_abs, _mm512_abs_ps(_mm512_load_ps(cur_vec+ii)));
                }
                cur_vec += handle->ifwp*handle->ifmblock;
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
            kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
            i+=3;
          }
        }
      }
    }
  } else {
    /* Run the stream of convolutions, no extra operations are required... */
    if (handle->perform_relu_in_kernel == 1) {/* do RELU stuff in the kernel  */
      LIBXSMM_VLA_DECL(5, element_input_type, original_input, ((element_input_type*)handle->reg_input->data) + (handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in * handle->ifmblock), BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
      element_input_type *regular_input_base;
      regular_input_base = &LIBXSMM_VLA_ACCESS(5, original_input, 0, 0, 0, 0, 0, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);

      if (handle->ofw == 7) {
        for (pc = 0; pc < instr; pc+=1) {
          offset_i = stream[i];
          offset_w = stream[i+1]; 
          offset_o = stream[i+2];
          pi = stream[i+3];
          pw = stream[i+4];
          po = stream[i+5];
          kernel_pool[variant[pc]]( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, regular_input_base + offset_o, &scale_factor, max_vals);
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
          kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, regular_input_base + offset_o, &scale_factor, max_vals);
          i+=3;
        }
      }
    } else {
      if (handle->ofw == 7) {
        for (pc = 0; pc < instr; pc+=1) {
          offset_i = stream[i];
          offset_w = stream[i+1]; 
          offset_o = stream[i+2];
          pi = stream[i+3];
          pw = stream[i+4];
          po = stream[i+5];
          kernel_pool[variant[pc]]( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
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
          kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor, max_vals);
          i+=3;
        }
      }    
    }
  }

  if ( ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) && (handle->use_lp_kernel == 1) && (handle->compute_max_in_kernel_bwd == 0) ) {   
    _mm512_store_ps(max_vals, max_abs);
  }
  libxsmm_barrier_wait(handle->barrier, ltid);

#if 0
  /* Fuse ReLu here*/
  if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_RELU_BWD) > 0) {
    int ii, ij, ifm1, ifm2, img;
    img = ltid;
    LIBXSMM_VLA_DECL(5, element_input_type, input, (element_input_type*) handle->reg_input->data,  BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
    LIBXSMM_VLA_DECL(5, element_input_type, del_input_2, (element_input_type*) handle->grad_input->data, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);  
    element_input_type *orig_input_ptr;
    element_input_type *del_input_ptr;
    __m512 zero_reg  = _mm512_setzero_ps();  
    __m512 orig_reg;
    __mmask16 mask; 
    for (ifm1 = 0; ifm1 < BLOCKSIFM; ifm1++ ) {
      orig_input_ptr = &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
      del_input_ptr = &LIBXSMM_VLA_ACCESS(5, del_input_2, img, ifm1, handle->desc.pad_h_in, handle->desc.pad_w_in, 0, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
      for (ij = 0; ij < handle->desc.H; ij++) {
        for (ii = 0; ii < handle->desc.W * 16; ii += 16) {
          orig_reg  = _mm512_load_ps(orig_input_ptr + ii);
          mask = _mm512_cmp_ps_mask(zero_reg, orig_reg, _CMP_EQ_OQ);
          _mm512_mask_storeu_ps(del_input_ptr + ii, mask, zero_reg);
        }
        orig_input_ptr += handle->ifwp * 16;
        del_input_ptr += handle->ifwp *16;
      }
    }
    libxsmm_barrier_wait(handle->barrier, ltid);
  }
#endif

}
