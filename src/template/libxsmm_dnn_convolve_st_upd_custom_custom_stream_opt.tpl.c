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
/* Evangelos Georganas, John Pennycook, Jason Sewall (Intel Corp.)
******************************************************************************/
#define WEIGHT_INIT 0
#define UPDATE_KERNEL 1
#define WEIGHT_COPY 2
#define TRANSPOSE_EXEC 3
#define LIBXSMM_UPD_STREAMS_TRANSPOSE_IFMB_SHIFT 12

/* FIXME assignments here */
int BLOCKSIFM = handle->blocksifm;
int BLOCKSOFM = handle->blocksofm;

/* computing first logical thread */
const int ltid = tid-start_thread;

/* Auxiliary integer variables   */
int img, ifm1, imgifm1, ij, i, j, k;
int ifmb;

/* transpose, copy and reduce work-related variables  */
const int reduce_work = BLOCKSOFM*BLOCKSIFM*handle->desc.R*handle->desc.S*handle->ifmblock;
const int reduce_chunksize = (reduce_work % handle->desc.threads == 0) ? (reduce_work / handle->desc.threads) : (reduce_work / handle->desc.threads) + 1;
const int reduce_thr_begin = (ltid * reduce_chunksize < reduce_work) ? (ltid * reduce_chunksize) : reduce_work;
const int reduce_thr_end = ((ltid + 1) * reduce_chunksize < reduce_work) ? ((ltid + 1) * reduce_chunksize) : reduce_work;
const int copywork = handle->desc.N*BLOCKSIFM;
const int copychunksize = (copywork % handle->desc.threads == 0) ? (copywork / handle->desc.threads) : (copywork / handle->desc.threads) + 1;
const int copy_thr_begin = (ltid * copychunksize < copywork) ? (ltid * copychunksize) : copywork;
const int copy_thr_end = ((ltid + 1) * copychunksize < copywork) ? ((ltid + 1) * copychunksize) : copywork;

/* Pointer related variables for output and weight */
element_output_type *const out = (element_output_type*)handle->grad_output->data + ((size_t)handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
LIBXSMM_VLA_DECL(5, element_output_type, output, out, BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock);
element_filter_type* weight_ptr = (element_filter_type*)handle->grad_filter->data;
element_filter_type* per_thread_weight_ptr = ((element_filter_type*)handle->scratch4) + (ltid*LIBXSMM_MIN(handle->block_upd_ofm,BLOCKSOFM)*LIBXSMM_MIN(handle->block_upd_ifm,BLOCKSIFM)*handle->desc.R*handle->desc.S*handle->ifmblock*handle->ofmblock);
LIBXSMM_VLA_DECL(2, element_filter_type, per_thread_weight, per_thread_weight_ptr, handle->ofmblock);
element_filter_type* reduction_weight_ptr = ((element_filter_type*)handle->scratch4) + (handle->desc.threads*LIBXSMM_MIN(handle->block_upd_ofm,BLOCKSOFM)*LIBXSMM_MIN(handle->block_upd_ifm,BLOCKSIFM)*handle->desc.R*handle->desc.S*handle->ifmblock*handle->ofmblock);
LIBXSMM_VLA_DECL(3, element_filter_type, reduction_weight, reduction_weight_ptr, handle->desc.threads, handle->ofmblock);

/* Pointer related variables for input */
element_input_type *LIBXSMM_RESTRICT input_ptr;
element_input_type *LIBXSMM_RESTRICT copy_ptr;
element_input_type *prefetch_ptr;
int padded_h = (handle->padding_flag == 1) ? handle->ifhp + 2 * handle->desc.pad_h : handle->ifhp;
int padded_w = (handle->padding_flag == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
int ifwp_extended = (handle->resize_input == 1 ? (handle->ifwp_resized + handle->qfma_input_pad) : (padded_w + handle->qfma_input_pad));
int dst_ifhp = (handle->resize_input == 1 ? handle->ifhp_resized : handle->ifhp);

LIBXSMM_VLA_DECL(5, const element_input_type, input_nopad, (element_input_type*)handle->reg_input->data, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
LIBXSMM_VLA_DECL(5, element_input_type, tr_input_padded, (element_input_type*)handle->scratch5, BLOCKSIFM, padded_h, handle->ifmblock, ifwp_extended);
LIBXSMM_VLA_DECL(5, element_input_type, input_padded, (element_input_type*)handle->scratch5, BLOCKSIFM, padded_h, padded_w, handle->ifmblock);
LIBXSMM_VLA_DECL(5, element_input_type, tr_input_nopad, (element_input_type*)handle->scratch3, BLOCKSIFM, dst_ifhp, handle->ifmblock, ifwp_extended);

/* Stream related variables  */
segment_t *code_stream;
int *stream = handle->compute_upd_indices_ptrs[ltid];
int instr, n_segments, n_convs, conv_i, offset_i, offset_t, offset_o, offset_w = 0, offset_s, pi, po, pw, pc;

/* Base pointers  */
const element_input_type *input_base;
element_input_type *input_zero;
element_filter_type *weight_base;
element_output_type *output_base;

/* Kernel related variables  */
libxsmm_xmcopyfunction jitted_matcopy = handle->matcopy_upd[0].xmatcopy;
libxsmm_xmcopyfunction jitted_matzero = handle->matcopy_upd[1].xmatcopy;
libxsmm_convfunction kernel = ( handle->trans_ofw_ifm == 0 ) ? (libxsmm_convfunction)handle->code_upd[0].xconv.sconv : (libxsmm_convfunction)handle->code_upd[1].xconv.sconv;

transposer tp_func = NULL;
if ( handle->trans_ofw_ifm > 0 ) {
  if (handle->padding_flag == 1) {
    tp_func = get_transposer(handle->ifmblock, handle->ifwp, ifwp_extended, handle->ifmblock);
  }
  else {
    tp_func = get_transposer(handle->ifmblock, handle->ifwp, ifwp_extended, handle->ifmblock);
  }
}

#if defined(_DEBUG) /* beyond !defined(NDEBUG) */
if (NULL == tp_func) {
  fprintf(stderr, "Couldn't find transposer to match %d %d %d %d", handle->ifmblock, handle->ifwp, ifwp_extended, handle->ifmblock);
  exit(1);
}
#endif

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);
/* If padding is requested, copy the entire minibatch upfront (only if transpose is not requested, otherwise we combine transpose with padding) */
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

/* Initialize base pointers */
if (handle->padding_flag == 1) {
  if (handle->trans_ofw_ifm > 0) {
    int imgs_per_thread = handle->desc.N/handle->desc.threads;
    input_base = &LIBXSMM_VLA_ACCESS(5, tr_input_padded, 0, 0, 0, 0, 0, BLOCKSIFM, padded_h, handle->ifmblock, ifwp_extended);
    input_zero = &LIBXSMM_VLA_ACCESS(5, tr_input_padded, imgs_per_thread*ltid, 0, 0, 0, 0, BLOCKSIFM, padded_h, handle->ifmblock, ifwp_extended);
    memset( input_zero, 0, imgs_per_thread*BLOCKSIFM * padded_h * ifwp_extended * handle->ifmblock * sizeof(element_input_type) );
  } else {
    input_base = &LIBXSMM_VLA_ACCESS(5, input_padded, 0, 0, 0, 0, 0, BLOCKSIFM, padded_h, padded_w, handle->ifmblock);
    input_zero = &LIBXSMM_VLA_ACCESS(5, input_padded, ltid, 0, 0, 0, 0, BLOCKSIFM, padded_h, padded_w, handle->ifmblock);
  }
  /* we need to set the scratch to zero */
  /* @TODO: we need to find a better/faster code here */

} else {
  if (handle->trans_ofw_ifm > 0) {
    input_base = &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, 0, 0, 0, 0, 0, BLOCKSIFM, dst_ifhp, handle->ifmblock, ifwp_extended);
    /*input_base = &LIBXSMM_VLA_ACCESS(5, input_nopad, 0, 0, 0, 0, 0, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock); */
  } else {
    input_base = &LIBXSMM_VLA_ACCESS(5, input_nopad, 0, 0, 0, 0, 0, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
  }
}
if ( handle->use_nts_upd == 0 ) {
  weight_base = &LIBXSMM_VLA_ACCESS(2, per_thread_weight, 0, 0, handle->ofmblock); /* use thread-private scratchpad to accumulate weights */
} else {
  weight_base = &LIBXSMM_VLA_ACCESS(3, reduction_weight, 0, ltid, 0, handle->desc.threads, handle->ofmblock); /* weights are accumulated in registers and can be written straight to memory */
}

output_base = &LIBXSMM_VLA_ACCESS(5, output, 0, 0, 0, 0, 0, BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock);

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
          LIBXSMM_ASSERT(NULL != tp_func);
          for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_upd_ifm, BLOCKSIFM); ifm1++) {
            for (ij=0; ij < handle->ifhp; ++ij) {
              float *dst = &(LIBXSMM_VLA_ACCESS(5, tr_input_padded, img, ifm1, ij + handle->desc.pad_h, 0, 0 + handle->desc.pad_w, BLOCKSIFM, padded_h, handle->ifmblock, ifwp_extended));
              const float *src = &(LIBXSMM_VLA_ACCESS(5, input_nopad, img, ifm1, ij, 0, 0, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock));
              tp_func(handle->ifmblock, handle->ifwp, dst, ifwp_extended, src, handle->ifmblock);
            }
          }
        } else {
          /* Transpose IFW and IFM */
          if (handle->resize_input == 0) {
            LIBXSMM_ASSERT(NULL != tp_func);
            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_upd_ifm, BLOCKSIFM); ifm1++) {
              for (ij=0; ij < handle->ifhp; ++ij) {
                float *dst = &(LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, ifm1, ij, 0, 0, BLOCKSIFM, handle->ifhp, handle->ifmblock, ifwp_extended));
                const float *src = &(LIBXSMM_VLA_ACCESS(5, input_nopad, img, ifm1, ij, 0, 0, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock));
                tp_func(handle->ifmblock, handle->ifwp, dst, ifwp_extended, src, handle->ifmblock);
              }
            }
          } else {
            int dst_i, dst_j, src_i, src_j, fm;
            for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_upd_ifm, BLOCKSIFM); ifm1++) {
              for (dst_j=0; dst_j < handle->ifhp_resized; dst_j++) {
                src_j = dst_j * handle->desc.v;
                for (dst_i=0; dst_i < handle->ifwp_resized; dst_i++) {
                  src_i = dst_i * handle->desc.u;
                  for (fm = 0; fm < handle->ifmblock; fm++){
                    LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, ifm1, dst_j, fm, dst_i, BLOCKSIFM, handle->ifhp_resized, handle->ifmblock, ifwp_extended) =
                      LIBXSMM_VLA_ACCESS(5, input_nopad, img, ifm1, src_j, src_i, fm, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
                  }
                }
              }
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
      if (libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  || libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM) {
        for ( j = 0; j < handle->desc.R*handle->desc.S*handle->ifmblock; j++ ) {
          LIBXSMM_PRAGMA_NONTEMPORAL
            LIBXSMM_PRAGMA_VALIGNED
            LIBXSMM_PRAGMA_SIMD
            for ( k = 0; k < 16; k++ ) {
              LIBXSMM_VLA_ACCESS(3, reduction_weight, offset_s + j, ltid, k, handle->desc.threads, 16) = LIBXSMM_VLA_ACCESS(2, per_thread_weight, offset_w + j, k, 16);
            }
        }
      } else {
        for ( j = 0; j < handle->desc.R*handle->desc.S*handle->ifmblock; j++ ) {
          LIBXSMM_PRAGMA_VALIGNED
            LIBXSMM_PRAGMA_SIMD
            for ( k = 0; k < 16; k++ ) {
              LIBXSMM_VLA_ACCESS(3, reduction_weight, offset_s + j, ltid, k, handle->desc.threads, 16) = LIBXSMM_VLA_ACCESS(2, per_thread_weight, offset_w + j, k, 16);
            }
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
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if (libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  || libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM) {
    if ( ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ) {
      for ( j = reduce_thr_begin; j < reduce_thr_end; j++ ) {
        __m512 weight_sum = _mm512_setzero_ps();
        for ( i = 0; i < handle->desc.threads; i++ ) {
          weight_sum = _mm512_add_ps(weight_sum, LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(3, reduction_weight, j, i, 0, handle->desc.threads, 16)));
        }
        LIBXSMM_INTRINSICS_MM512_STREAM_PS(&weight_ptr[j*16], weight_sum);
      }
    } else {
      for ( j = reduce_thr_begin; j < reduce_thr_end; j++ ) {
        __m512 weight_sum = _mm512_setzero_ps();
        for ( i = 0; i < handle->desc.threads; i++ ) {
          weight_sum = _mm512_add_ps(weight_sum, LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(3, reduction_weight, j, i, 0, handle->desc.threads, 16)));
        }
        _mm512_store_ps(&weight_ptr[j*16], _mm512_add_ps(weight_sum, LIBXSMM_INTRINSICS_MM512_LOAD_PS(&weight_ptr[j*16])));
      }
    }
  } else {
    if ( ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ) {
      for ( j = reduce_thr_begin; j < reduce_thr_end; j++ ) {
        __m512 weight_sum = _mm512_setzero_ps();
        for ( i = 0; i < handle->desc.threads; i++ ) {
          weight_sum = _mm512_add_ps(weight_sum, LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(3, reduction_weight, j, i, 0, handle->desc.threads, 16)));
        }
        _mm512_store_ps(&weight_ptr[j*16], weight_sum);
      }
    } else {
      for ( j = reduce_thr_begin; j < reduce_thr_end; j++ ) {
        __m512 weight_sum = _mm512_setzero_ps();
        for ( i = 0; i < handle->desc.threads; i++ ) {
          weight_sum = _mm512_add_ps(weight_sum, LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(3, reduction_weight, j, i, 0, handle->desc.threads, 16)));
        }
        _mm512_store_ps(&weight_ptr[j*16], _mm512_add_ps(weight_sum, LIBXSMM_INTRINSICS_MM512_LOAD_PS(&weight_ptr[j*16])));
      }
    }
  }
#else
/* should not happen */
#endif
}
libxsmm_barrier_wait(handle->barrier, ltid);

#undef WEIGHT_INIT
#undef UPDATE_KERNEL
#undef WEIGHT_COPY
#undef TRANSPOSE_EXEC
#undef LIBXSMM_UPD_STREAMS_TRANSPOSE_IFMB_SHIFT
