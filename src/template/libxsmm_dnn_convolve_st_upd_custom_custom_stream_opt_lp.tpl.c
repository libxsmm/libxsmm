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
/* Evangelos Georganas, John Pennycook, Jason Sewall (Intel Corp.)
******************************************************************************/
#define WEIGHT_INIT 0
#define UPDATE_KERNEL 1
#define WEIGHT_COPY 2

/* FIXME assignemnts here  */
int BLOCKSIFM = handle->blocksifm;
int BLOCKSOFM = handle->blocksofm;
int OFWP = handle->ofwp+handle->output_lp_padding;

/* computing first logical thread */
const int ltid = tid-start_thread;

/* Auxiliary integer variables   */
int img, ifm1, ifm2, imgifm1,ii, ij, i;
int j, k;
int ifmb;

int imgpt = (handle->desc.N + handle->desc.threads - 1)/handle->desc.threads;
int my_img_start = LIBXSMM_MIN( ltid * imgpt, handle->desc.N);
int my_img_end = LIBXSMM_MIN( (ltid+1) * imgpt, handle->desc.N);

/* traspose, copy and reduce work-related variables  */
const int reduce_work = BLOCKSOFM*BLOCKSIFM*handle->desc.R*handle->desc.S*handle->ifmblock;
const int reduce_chunksize = (reduce_work % handle->desc.threads == 0) ? (reduce_work / handle->desc.threads) : (reduce_work / handle->desc.threads) + 1;
const int reduce_thr_begin = (ltid * reduce_chunksize < reduce_work) ? (ltid * reduce_chunksize) : reduce_work;
const int reduce_thr_end = ((ltid + 1) * reduce_chunksize < reduce_work) ? ((ltid + 1) * reduce_chunksize) : reduce_work;
const int copywork = handle->desc.N*BLOCKSIFM;
const int copychunksize = (copywork % handle->desc.threads == 0) ? (copywork / handle->desc.threads) : (copywork / handle->desc.threads) + 1;
const int copy_thr_begin = (ltid * copychunksize < copywork) ? (ltid * copychunksize) : copywork;
const int copy_thr_end = ((ltid + 1) * copychunksize < copywork) ? ((ltid + 1) * copychunksize) : copywork;

/* Pointer related variables for output and weight */
element_output_type *out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock * handle->fm_lp_block;
LIBXSMM_VLA_DECL(6, element_output_type, tr_output,  (element_output_type*)handle->scratch6 , BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2);
LIBXSMM_VLA_DECL(6, element_output_type, output, out, handle->blocksofm_lp, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);
element_filter_type* weight_ptr = (element_filter_type*)handle->grad_filter->data;
element_filter_type* per_thread_weight_ptr = ((element_filter_type*)handle->scratch4) + (ltid*LIBXSMM_MIN(handle->block_upd_ofm,BLOCKSOFM)*LIBXSMM_MIN(handle->block_upd_ifm,BLOCKSIFM)*handle->desc.R*handle->desc.S*handle->ifmblock*handle->ofmblock);
LIBXSMM_VLA_DECL(2, element_filter_type, per_thread_weight, per_thread_weight_ptr, handle->ofmblock);
element_filter_type* reduction_weight_ptr = ((element_filter_type*)handle->scratch4) + (handle->desc.threads*LIBXSMM_MIN(handle->block_upd_ofm,BLOCKSOFM)*LIBXSMM_MIN(handle->block_upd_ifm,BLOCKSIFM)*handle->desc.R*handle->desc.S*handle->ifmblock*handle->ofmblock);
LIBXSMM_VLA_DECL(3, element_filter_type, reduction_weight, reduction_weight_ptr, handle->desc.threads, handle->ofmblock);

/* Pointer related variables for input */
element_input_type (* LIBXSMM_RESTRICT input_ptr);
element_input_type (* LIBXSMM_RESTRICT copy_ptr);
element_input_type *prefetch_ptr;
int padded_h = (handle->padding_flag == 1) ? handle->ifhp + 2 * handle->desc.pad_h : handle->ifhp;
int padded_w = (handle->padding_flag == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
int ifwp_extended = padded_w + handle->qfma_input_pad;
int dst_ifhp;

if (handle->resize_input == 1) {
  ifwp_extended = handle->ifwp_resized + handle->qfma_input_pad;
  dst_ifhp = handle->ifhp_resized;
} else {
  dst_ifhp = handle->ifhp;
}

LIBXSMM_VLA_DECL(6, element_input_type, input_nopad, (element_input_type*)handle->reg_input->data, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
LIBXSMM_VLA_DECL(5, element_input_type, tr_input_padded, (element_input_type*)handle->scratch5, BLOCKSIFM, padded_h, handle->ifmblock, ifwp_extended);
LIBXSMM_VLA_DECL(5, element_input_type, input_padded, (element_input_type*)handle->scratch5, BLOCKSIFM, padded_h, padded_w, handle->ifmblock);
LIBXSMM_VLA_DECL(5, element_input_type, tr_input_nopad, (element_input_type*)handle->scratch3, BLOCKSIFM, dst_ifhp, handle->ifmblock, ifwp_extended);

/* Stream related variables  */
segment_t *code_stream;
int *stream = handle->compute_upd_indices_ptrs[ltid];
int instr, n_segments, n_convs, conv_i, offset_i, offset_t, offset_o, offset_w, offset_s, pi, po, pw, pc;

/* Base pointers  */
element_input_type *input_base;
element_input_type *input_zero;
element_filter_type *weight_base;
element_output_type *output_base;

/* Kernel related variables  */
libxsmm_xmatcopyfunction jitted_matcopy = handle->matcopy_upd[0].xmatcopy;
libxsmm_xmatcopyfunction jitted_matzero = handle->matcopy_upd[1].xmatcopy;
libxsmm_xmatcopyfunction jitted_matzero_weights = handle->matcopy_upd[2].xmatcopy;
libxsmm_convfunction kernel = ( handle->trans_ofw_ifm == 0 || handle->ifmblock == 1 ) ? (libxsmm_convfunction)handle->code_upd[1].xconv.sconv : (libxsmm_convfunction)handle->code_upd[4].xconv.sconv;

transposer tp_func;
if ( handle->trans_ofw_ifm > 0 ) {
  if (handle->padding_flag == 1) {
    tp_func = get_transposer(handle->ifmblock, handle->ifwp, ifwp_extended, handle->ifmblock);
  }
  else
    tp_func = get_transposer(handle->ifmblock, handle->ifwp, ifwp_extended, handle->ifmblock);
}

#if 0
if(tp_func == 0) {
  fprintf(stderr, "Couldn't find transposer to match %d %d %d %d", handle->ifmblock, handle->ifwp, ifwp_extended, handle->ifmblock);
  exit(1);
}
#endif

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);
#if 0
/* If padding is requested, copy the entire minibatch upfront (only if trnaspose is not requested, otherwise we combine trnaspose with padding) */
if (handle->padding_flag == 1) {
  /* Initialize in parallel scratch5 to zero */
  for (imgifm1 = copy_thr_begin; imgifm1 < copy_thr_end; ++imgifm1) {
    img = imgifm1/BLOCKSIFM;
    ifm1 = imgifm1%BLOCKSIFM;
    copy_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_padded, img, ifm1, 0, 0, 0, BLOCKSIFM, padded_h, padded_w, handle->ifmblock);
    jitted_matzero(NULL, NULL, copy_ptr, NULL, NULL);
  }
  libxsmm_barrier_wait(handle->barrier, ltid);

  if ( handle->trans_ofw_ifm == 0 ) {
    for (imgifm1 = copy_thr_end-1; imgifm1 >= copy_thr_begin; imgifm1--) {
      img = imgifm1/BLOCKSIFM;
      ifm1 = imgifm1%BLOCKSIFM;
      input_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_nopad, img, ifm1, 0, 0, 0, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
      copy_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_padded, img, ifm1, handle->desc.pad_h, handle->desc.pad_w, 0, BLOCKSIFM, padded_h, padded_w, handle->ifmblock);
      prefetch_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_nopad, (imgifm1-1)/BLOCKSIFM, (imgifm1-1)%BLOCKSIFM, 0, 0, 0, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
      jitted_matcopy(input_ptr, NULL, copy_ptr, NULL, prefetch_ptr);
    }
    libxsmm_barrier_wait(handle->barrier, ltid);
  }
}
#endif

/* Initialize base pointers */
if (handle->padding_flag == 1) {
  input_base = &LIBXSMM_VLA_ACCESS(5, tr_input_padded, 0, 0, 0, 0, 0, BLOCKSIFM, padded_h, handle->ifmblock, ifwp_extended);
  input_zero = &LIBXSMM_VLA_ACCESS(5, tr_input_padded, ltid, 0, 0, 0, 0, BLOCKSIFM, padded_h, handle->ifmblock, ifwp_extended);
  memset( input_zero, 0, BLOCKSIFM * padded_h * ifwp_extended * handle->ifmblock * sizeof(element_input_type) );
} else {
  input_base = &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, 0, 0, 0, 0, 0, BLOCKSIFM, dst_ifhp, handle->ifmblock, ifwp_extended);
}

/* LP transformations */
{
  int img = ltid, ifm1, ij, ifm2, ii;
  int ofm1, ofm2, k, lp;
  int FM;
  int W;

  if (handle->padding_flag == 1) {
    for (ifm1 = 0; ifm1 < handle->blocksifm_lp; ++ifm1) {
      for (ij = 0; ij < handle->ifhp; ++ij) {
        for (ii = 0; ii < handle->ifwp; ++ii) {
          for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
            for (lp = 0; lp < handle->fm_lp_block; ++lp) {
              FM = ifm1 * handle->ifmblock * handle->fm_lp_block + ifm2 * handle->fm_lp_block + lp;
              LIBXSMM_VLA_ACCESS(5, tr_input_padded, img, FM/handle->ifmblock, ij+handle->desc.pad_h, FM%handle->ifmblock, ii+handle->desc.pad_w, BLOCKSIFM, padded_h, handle->ifmblock, ifwp_extended) =
                LIBXSMM_VLA_ACCESS(6, input_nopad, img, ifm1, ij, ii, ifm2, lp, handle->blocksifm_lp, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
            }
          }
        }
      }
    }  
  } else {
   if (handle->resize_input == 0) {
      if (handle->ifwp%16 == 0) {
#include "input_lp_transposer.tpl.c" 
      } else {
#include "input_lp_transposer_remainder.tpl.c"   
      }
    } else {
      if (handle->ifwp_resized%16 == 0) {
#include "input_lp_transposer_resizer.tpl.c"   
      } else {
#include "input_lp_transposer_resizer_remainder.tpl.c"     
      }
    }
  }
#include "output_lp_transposer.tpl.c"
}

libxsmm_barrier_wait(handle->barrier, ltid);

if ( handle->ofh == 28 || handle->ofh == 56 )
{
  weight_base = &LIBXSMM_VLA_ACCESS(2, per_thread_weight, 0, 0, handle->ofmblock); /* use thread-private scratchpad to accumulate weights */
} else {
  weight_base = &LIBXSMM_VLA_ACCESS(3, reduction_weight, 0, ltid, 0, handle->desc.threads, handle->ofmblock); /* weights are accumulated in registers and can be written straight to memory */
}

output_base = &LIBXSMM_VLA_ACCESS(6, tr_output, 0, 0, 0, 0, 0, 0, BLOCKSOFM, handle->ofhp, OFWP/2, handle->ofmblock, 2);

i = 0;
instr = handle->n_entries_upd[ltid];
n_segments = handle->n_upd_code_segments[ltid];

float scale_factor __attribute__((aligned(64)));
if (handle->use_lp_kernel == 1) {
  scale_factor = 1.0; // (float) pow(2.0, -1.0*(handle->reg_filter->exp + handle->reg_input->exp));
}

float *max_vals __attribute__((aligned(64)));
__m512 max_abs = _mm512_setzero_ps();
if ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0) {
  LIBXSMM_VLA_DECL(2, float, maxstats, handle->maxstats_upd->data, 16);
  max_vals = (float*) &LIBXSMM_VLA_ACCESS(2, maxstats, ltid, 0, 16);
}

if (n_segments) {
  /* We have segmented the stream of convolutions since we need to inject different functionalities... */
  code_stream = handle->upd_code_segments[ltid];
  for (pc = 0; pc < n_segments; pc++) {
    instr = code_stream[pc].segment_type;
    n_convs = code_stream[pc].n_convs;

    if (instr == WEIGHT_INIT) {
      offset_w = code_stream[pc].aux_index;
      for ( j = offset_w; j < offset_w + handle->desc.R*handle->desc.S*handle->ifmblock*handle->ofmblock; j += 16) {
        LIBXSMM_PRAGMA_VALIGNED
          LIBXSMM_PRAGMA_SIMD
          for ( k = 0; k < 16; ++k ) {
            weight_base[j + k] = (element_filter_type) 0;
          }
      }
    }

    if (instr == WEIGHT_COPY) {
      offset_w /= handle->desc.R * handle->desc.S * handle->ifmblock * handle->ofmblock;
      offset_w *= handle->desc.R * handle->desc.S * handle->ifmblock;
      offset_s = code_stream[pc].aux_index;
      for ( j = 0; j < handle->desc.R*handle->desc.S*handle->ifmblock; j++ ) {
        LIBXSMM_PRAGMA_NONTEMPORAL
          LIBXSMM_PRAGMA_VALIGNED
          LIBXSMM_PRAGMA_SIMD
          for ( k = 0; k < 16; k++ ) {
            LIBXSMM_VLA_ACCESS(3, reduction_weight, offset_s + j, ltid, k, handle->desc.threads, 16) = LIBXSMM_VLA_ACCESS(2, per_thread_weight, offset_w + j, k, 16);
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
      kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor);
      i+=3;
    }
  }
} else {
  /* Run the stream of convolutions, no extra operations are required...  */
  for (pc = 0; pc < instr; pc++)
  {
    offset_i = stream[i];
    offset_w = stream[i+1];
    offset_o = stream[i+2];
    pi = stream[i+3];
    pw = stream[i+4];
    po = stream[i+5];
    kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po, &scale_factor);
    i+=3;
  }
}

#define __AVX512F__
/* Perform reduction because we used thread private filters... */
if (handle->upd_use_external_reduce == 0) {
  libxsmm_barrier_wait(handle->barrier, ltid);
  for ( j = reduce_thr_begin; j < reduce_thr_end; j++ ) {
#ifdef __AVX512F__
    __m512 weight_sum = _mm512_setzero_ps();
    for ( i = 0; i < handle->desc.threads; i++ ) {
      weight_sum = _mm512_add_ps(weight_sum, _mm512_load_ps(&LIBXSMM_VLA_ACCESS(3, reduction_weight, j, i, 0, handle->desc.threads, 16)));
    }
    if ( ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ) {
      _mm512_stream_ps(&weight_ptr[j*16], weight_sum);
      max_abs = _mm512_max_ps(max_abs, _mm512_abs_ps(weight_sum));
    } else {
      __m512 new_result = _mm512_add_ps(weight_sum, _mm512_load_ps(&weight_ptr[j*16]));
      _mm512_store_ps(&weight_ptr[j*16], new_result);
      max_abs = _mm512_max_ps(max_abs, _mm512_abs_ps(new_result));
    }
#else
    element_filter_type weight_sum[16] LIBXSMM_ATTRIBUTE(aligned(64));
    LIBXSMM_PRAGMA_VALIGNED
      LIBXSMM_PRAGMA_SIMD
      for ( k = 0; k < 16; k++ ) {
        weight_sum[k] = (element_filter_type) 0;
      }
    for ( i = 0; i < handle->desc.threads; i++ ) {
      LIBXSMM_PRAGMA_VALIGNED
        LIBXSMM_PRAGMA_SIMD
        for ( k = 0; k < 16; k++ ) {
          weight_sum[k] += LIBXSMM_VLA_ACCESS(3, reduction_weight, j, i, k, handle->desc.threads, 16);
        }
    }
    if ( ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ) {
      LIBXSMM_PRAGMA_NONTEMPORAL
        LIBXSMM_PRAGMA_VALIGNED
        LIBXSMM_PRAGMA_SIMD
        for ( k = 0; k < 16; k++ ) {
          weight_ptr[j*16 + k] = weight_sum[k];
        }
    } else {
      LIBXSMM_PRAGMA_VALIGNED
        LIBXSMM_PRAGMA_SIMD
        for ( k = 0; k < 16; k++ ) {
          weight_ptr[j*16 + k] += weight_sum[k];
        }
    }
#endif
  }
#ifdef __AVX512F__
  _mm512_store_ps(max_vals, max_abs);
#endif
}
libxsmm_barrier_wait(handle->barrier, ltid);
#undef __AVX512F__
