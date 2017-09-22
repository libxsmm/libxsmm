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
#define TRANSPOSE_EXEC 3
#define LIBXSMM_UPD_STREAMS_TRANSPOSE_IFMB_SHIFT 12

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
const int reduce_work = handle->blocksofm*handle->blocksifm*handle->desc.R*handle->desc.S*handle->ifmblock;
const int reduce_chunksize = (reduce_work % handle->desc.threads == 0) ? (reduce_work / handle->desc.threads) : (reduce_work / handle->desc.threads) + 1;
const int reduce_thr_begin = (ltid * reduce_chunksize < reduce_work) ? (ltid * reduce_chunksize) : reduce_work;
const int reduce_thr_end = ((ltid + 1) * reduce_chunksize < reduce_work) ? ((ltid + 1) * reduce_chunksize) : reduce_work;
const int copywork = handle->desc.N*handle->blocksifm;
const int copychunksize = (copywork % handle->desc.threads == 0) ? (copywork / handle->desc.threads) : (copywork / handle->desc.threads) + 1;
const int copy_thr_begin = (ltid * copychunksize < copywork) ? (ltid * copychunksize) : copywork;
const int copy_thr_end = ((ltid + 1) * copychunksize < copywork) ? ((ltid + 1) * copychunksize) : copywork;

/* Pointer related variables for output and weight */
element_output_type *const out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
LIBXSMM_VLA_DECL(5, element_output_type, output, out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
element_filter_type* weight_ptr = (element_filter_type*)handle->grad_filter->data;
element_filter_type* per_thread_weight_ptr = ((element_filter_type*)handle->scratch4) + (ltid*handle->block_upd_ofm*handle->block_upd_ifm*handle->desc.R*handle->desc.S*handle->ifmblock*handle->ofmblock);
LIBXSMM_VLA_DECL(2, element_filter_type, per_thread_weight, per_thread_weight_ptr, handle->ofmblock);
element_filter_type* reduction_weight_ptr = ((element_filter_type*)handle->scratch4) + (handle->desc.threads*handle->block_upd_ofm*handle->block_upd_ifm*handle->desc.R*handle->desc.S*handle->ifmblock*handle->ofmblock);
LIBXSMM_VLA_DECL(3, element_filter_type, reduction_weight, reduction_weight_ptr, handle->desc.threads, handle->ofmblock);

/* Pointer related variables for input */
element_input_type (* LIBXSMM_RESTRICT input_ptr);
element_input_type (* LIBXSMM_RESTRICT copy_ptr);
element_input_type *prefetch_ptr;
int padded_h = (handle->padding_flag == 1) ? handle->ifhp + 2 * handle->desc.pad_h : handle->ifhp;
int padded_w = (handle->padding_flag == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
int ifwp_extended = padded_w + handle->qfma_input_pad;

LIBXSMM_VLA_DECL(5, const element_input_type, input_nopad, (element_input_type*)handle->reg_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
LIBXSMM_VLA_DECL(5, element_input_type, tr_input_padded, (element_input_type*)handle->scratch5, handle->blocksifm, padded_h, handle->ifmblock, ifwp_extended);
LIBXSMM_VLA_DECL(5, element_input_type, input_padded, (element_input_type*)handle->scratch5, handle->blocksifm, padded_h, padded_w, handle->ifmblock);
LIBXSMM_VLA_DECL(5, element_input_type, tr_input_nopad, (element_input_type*)handle->scratch3, handle->blocksifm, handle->ifhp, handle->ifmblock, ifwp_extended);

/* Stream related variables  */
segment_t *code_stream;
int *stream = handle->compute_upd_indices_ptrs[ltid];
int instr, n_segments, n_convs, conv_i, offset_i, offset_t, offset_o, offset_w, offset_s, pi, po, pw, pc;

/* Base pointers  */
const element_input_type *input_base;
element_filter_type *weight_base;
element_output_type *output_base;

/* Kernel related variables  */
libxsmm_xmatcopyfunction jitted_matcopy = handle->matcopy_upd[0].xmatcopy;
libxsmm_xmatcopyfunction jitted_matzero = handle->matcopy_upd[1].xmatcopy;
libxsmm_xmatcopyfunction jitted_matzero_weights = handle->matcopy_upd[2].xmatcopy;
libxsmm_convfunction kernel = ( handle->trans_ofw_ifm == 0 || handle->ifmblock == 1 ) ? (libxsmm_convfunction)handle->code_upd[1].xconv.sconv : (libxsmm_convfunction)handle->code_upd[4].xconv.sconv;

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

#if 0
/* If padding is requested, copy the entire minibatch upfront (only if trnaspose is not requested, otherwise we combine trnaspose with padding) */
if (handle->padding_flag == 1) {
  /* Initialize in parallel scratch5 to zero */
  for (imgifm1 = copy_thr_begin; imgifm1 < copy_thr_end; ++imgifm1) {
    img = imgifm1/handle->blocksifm;
    ifm1 = imgifm1%handle->blocksifm;
    copy_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_padded, img, ifm1, 0, 0, 0, handle->blocksifm, padded_h, padded_w, handle->ifmblock);
    jitted_matzero(NULL, NULL, copy_ptr, NULL, NULL);
  }
  libxsmm_barrier_wait(handle->barrier, ltid);

  if ( handle->trans_ofw_ifm == 0 ) {
    for (imgifm1 = copy_thr_end-1; imgifm1 >= copy_thr_begin; imgifm1--) {
      img = imgifm1/handle->blocksifm;
      ifm1 = imgifm1%handle->blocksifm;
      input_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_nopad, img, ifm1, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      copy_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_padded, img, ifm1, handle->desc.pad_h, handle->desc.pad_w, 0, handle->blocksifm, padded_h, padded_w, handle->ifmblock);
      prefetch_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_nopad, (imgifm1-1)/handle->blocksifm, (imgifm1-1)%handle->blocksifm, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      jitted_matcopy(input_ptr, NULL, copy_ptr, NULL, prefetch_ptr);
    }
    libxsmm_barrier_wait(handle->barrier, ltid);
  }
}
#endif

/* Initialize base pointers */
if (handle->padding_flag == 1) {
  if (handle->trans_ofw_ifm > 0) {
    input_base = &LIBXSMM_VLA_ACCESS(5, tr_input_padded, 0, 0, 0, 0, 0, handle->blocksifm, padded_h, handle->ifmblock, padded_w);
  } else {
    input_base = &LIBXSMM_VLA_ACCESS(5, input_padded, 0, 0, 0, 0, 0, handle->blocksifm, padded_h, padded_w, handle->ifmblock);
  }
} else {
  if (handle->trans_ofw_ifm > 0) {
    input_base = &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, 0, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifmblock, ifwp_extended);
    /*input_base = &LIBXSMM_VLA_ACCESS(5, input_nopad, 0, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock); */
  } else {
    input_base = &LIBXSMM_VLA_ACCESS(5, input_nopad, 0, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
  }
}
if ( handle->ofh == 28 || handle->ofh == 56 )
{
  weight_base = &LIBXSMM_VLA_ACCESS(2, per_thread_weight, 0, 0, handle->ofmblock); /* use thread-private scratchpad to accumulate weights */
} else {
  weight_base = &LIBXSMM_VLA_ACCESS(3, reduction_weight, 0, ltid, 0, handle->desc.threads, handle->ofmblock); /* weights are accumulated in registers and can be written straight to memory */
}
output_base = &LIBXSMM_VLA_ACCESS(5, output, 0, 0, 0, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);

i = 0;
instr = handle->n_entries_upd[ltid];
n_segments = handle->n_upd_code_segments[ltid];
if (n_segments) {
  /* We have segmented the stream of convolutions since we need to inject different functionalities... */
  code_stream = handle->upd_code_segments[ltid];
  for (pc = 0; pc < n_segments; pc++) {
    instr = code_stream[pc].segment_type;
    n_convs = code_stream[pc].n_convs;

    if (instr == TRANSPOSE_EXEC) {
      offset_t = code_stream[pc].aux_index;
      img = offset_t & ((1 << LIBXSMM_UPD_STREAMS_TRANSPOSE_IFMB_SHIFT)-1);
      ifmb = offset_t >> LIBXSMM_UPD_STREAMS_TRANSPOSE_IFMB_SHIFT;
      if ( handle->trans_ofw_ifm > 0 ) {
        if (handle->padding_flag == 1) {
          /* Transpose IFW and IFM into the padded buffer!*/
          for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_upd_ifm, handle->blocksifm); ifm1++) {
            for (ij=0; ij < handle->ifhp; ++ij) {
              float *dst = &(LIBXSMM_VLA_ACCESS(5, tr_input_padded, img, ifm1, ij + handle->desc.pad_h, 0, 0 + handle->desc.pad_w, handle->blocksifm, padded_h, handle->ifmblock, ifwp_extended));
              const float *src = &(LIBXSMM_VLA_ACCESS(5, input_nopad, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock));
              #if defined(__AVX512F__)
              block_gather_transpose_ps(handle->ifmblock, handle->ifwp, dst, ifwp_extended, src, handle->ifmblock);
              #else
              for (ii=0; ii < handle->ifwp; ++ii) {
                for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                  dst[ifm2*ifwp_extended + ii] = src[ii*handle->ifmblock + ifm2];
                }
              }
              #endif //defined(__AVX512F__)
            }
          }
        } else {
          /* Transpose IFW and IFM */
          for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_upd_ifm, handle->blocksifm); ifm1++) {
            for (ij=0; ij < handle->ifhp; ++ij) {
              float *dst = &(LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifmblock, ifwp_extended));
              const float *src = &(LIBXSMM_VLA_ACCESS(5, input_nopad, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock));
              #if defined(__AVX512F__)
              block_gather_transpose_ps(handle->ifmblock, handle->ifwp, dst, ifwp_extended, src, handle->ifmblock);
              #else
              for (ii=0; ii < handle->ifwp; ++ii) {
                for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                  dst[ifm2*ifwp_extended + ii] = src[ii*handle->ifmblock + ifm2];
                }
              }
              #endif //defined(__AVX512F__)
            }
          }
        }
      }
    }

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
      kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po );
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
      kernel( input_base + offset_i, weight_base + offset_w, output_base + offset_o, input_base + pi, weight_base + pw, output_base + po );
      i+=3;
  }
}

/* Perform reduction because we used thread private filters... */
if (handle->upd_use_external_reduce == 0) {
  libxsmm_barrier_wait(handle->barrier, ltid);
  for ( j = reduce_thr_begin; j < reduce_thr_end; j++ ) {
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
  }
}
libxsmm_barrier_wait(handle->barrier, ltid);