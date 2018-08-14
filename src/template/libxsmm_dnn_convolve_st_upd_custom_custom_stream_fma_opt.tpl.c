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

/* computing first logical thread */
const int ltid = tid-start_thread;

/* FIXME assignments here */
int BLOCKSIFM = handle->blocksifm;
int BLOCKSOFM = handle->blocksofm;

/* Auxiliary integer variables   */
int img, ifm1, ifm2, imgifm1,ii, ij, i, j;

int imgpt = (handle->desc.N + handle->desc.threads - 1)/handle->desc.threads;
int my_img_start = LIBXSMM_MIN( ltid * imgpt, handle->desc.N);
int my_img_end = LIBXSMM_MIN( (ltid+1) * imgpt, handle->desc.N);

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
element_filter_type* remote_weight_ptr = 0;
element_filter_type* weight_ptr = (element_filter_type*)handle->grad_filter->data;
element_filter_type* per_thread_weight_ptr = ((element_filter_type*)handle->scratch4) + (ltid*BLOCKSOFM*BLOCKSIFM*handle->desc.R*handle->desc.S*handle->ifmblock*handle->ofmblock);
LIBXSMM_VLA_DECL(6, element_filter_type, per_thread_weight, per_thread_weight_ptr, BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
/* Declare both variables for weights (private and global)  */
LIBXSMM_VLA_DECL(6, element_filter_type, opt_weight_ptr_per_thread, per_thread_weight, BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
/* Pointer related variables for input */
element_input_type *LIBXSMM_RESTRICT input_ptr;
element_input_type *LIBXSMM_RESTRICT copy_ptr;
element_input_type *prefetch_ptr;
int padded_h = (handle->padding_flag == 1) ? handle->ifhp + 2 * handle->desc.pad_h : handle->ifhp;
int padded_w = (handle->padding_flag == 1) ? handle->ifwp + 2 * handle->desc.pad_w : handle->ifwp;
LIBXSMM_VLA_DECL(5, const element_input_type, input_nopad, (element_input_type*)handle->reg_input->data, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
LIBXSMM_VLA_DECL(5, element_input_type, tr_input_padded, (element_input_type*)handle->scratch5, BLOCKSIFM, padded_h, handle->ifmblock, padded_w);
LIBXSMM_VLA_DECL(5, element_input_type, input_padded, (element_input_type*)handle->scratch5, BLOCKSIFM, padded_h, padded_w, handle->ifmblock);
LIBXSMM_VLA_DECL(5, element_input_type, tr_input_nopad, (element_input_type*)handle->scratch3, BLOCKSIFM, handle->ifhp, handle->ifmblock, handle->ifwp);

/* Stream related variables  */
int *stream = handle->compute_upd_indices_ptrs[ltid];
int instr, offset_i, offset_o, offset_w, pi, po, pw, pc;

/* Base pointers  */
const element_input_type *input_base;
const element_filter_type *weight_base;
element_output_type *output_base;

/* Kernel related variables  */
libxsmm_xmcopyfunction jitted_matcopy = handle->matcopy_upd[0].xmatcopy;
libxsmm_xmcopyfunction jitted_matzero = handle->matcopy_upd[1].xmatcopy;
libxsmm_xmcopyfunction jitted_matzero_weights = handle->matcopy_upd[2].xmatcopy;
libxsmm_convfunction kernel = ( handle->trans_ofw_ifm == 0 ) ? (libxsmm_convfunction)handle->code_upd[0].xconv.sconv : (libxsmm_convfunction)handle->code_upd[1].xconv.sconv;

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

/* We DO USE private weights, initialize them to zero...  */
jitted_matzero_weights(NULL, NULL, per_thread_weight_ptr, NULL, NULL);

/* Handle transpose of input  */
if ( handle->trans_ofw_ifm > 0 ) {
  if (handle->padding_flag == 1) {
    /* Transpose IFW and IFM into the padded buffer!*/
    for (img = my_img_start; img < my_img_end; img++) {
      for (ifm1 = 0; ifm1 < BLOCKSIFM; ifm1++) {
        for (ij=0; ij < handle->ifhp; ++ij) {
          for (ii=0; ii < handle->ifwp; ++ii) {
            for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
              LIBXSMM_VLA_ACCESS(5, tr_input_padded, img, ifm1, ij + handle->desc.pad_h, ifm2, ii + handle->desc.pad_w, BLOCKSIFM, padded_h, handle->ifmblock, padded_w)
                =  LIBXSMM_VLA_ACCESS(5, input_nopad, img, ifm1, ij, ii, ifm2, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
            }
          }
        }
      }
    }
  } else {
    /* Transpose IFW and IFM */
    for (img = my_img_start; img < my_img_end; img++) {
      for (ifm1 = 0; ifm1 < BLOCKSIFM; ifm1++) {
        for (ij=0; ij < handle->ifhp; ++ij) {
          for (ii=0; ii < handle->ifwp; ++ii) {
            for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
              LIBXSMM_VLA_ACCESS(5, tr_input_nopad, img, ifm1, ij, ifm2, ii, BLOCKSIFM, handle->ifhp, handle->ifmblock, handle->ifwp)
                =  LIBXSMM_VLA_ACCESS(5, input_nopad, img, ifm1, ij, ii, ifm2, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
            }
          }
        }
      }
    }
  }
}

/* Initialize base pointers */
if (handle->padding_flag == 1) {
  if (handle->trans_ofw_ifm > 0) {
    input_base = &LIBXSMM_VLA_ACCESS(5, tr_input_padded, 0, 0, 0, 0, 0, BLOCKSIFM, padded_h, handle->ifmblock, padded_w);
  } else {
    input_base = &LIBXSMM_VLA_ACCESS(5, input_padded, 0, 0, 0, 0, 0, BLOCKSIFM, padded_h, padded_w, handle->ifmblock);
  }
} else {
  if (handle->trans_ofw_ifm > 0) {
     input_base = &LIBXSMM_VLA_ACCESS(5, tr_input_nopad, 0, 0, 0, 0, 0, BLOCKSIFM, handle->ifhp, handle->ifmblock, handle->ifwp);
    /*input_base = &LIBXSMM_VLA_ACCESS(5, input_nopad, 0, 0, 0, 0, 0, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock); */
  } else {
    input_base = &LIBXSMM_VLA_ACCESS(5, input_nopad, 0, 0, 0, 0, 0, BLOCKSIFM, handle->ifhp, handle->ifwp, handle->ifmblock);
  }
}
weight_base = &LIBXSMM_VLA_ACCESS(6, opt_weight_ptr_per_thread, 0, 0, 0, 0, 0, 0, BLOCKSIFM, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
output_base = &LIBXSMM_VLA_ACCESS(5, output, 0, 0, 0, 0, 0, BLOCKSOFM, handle->ofhp, handle->ofwp, handle->ofmblock);

i = 0;
instr = handle->n_entries_upd[ltid];
/* Run the stream of convolutions, no extra operations are required...  */
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

libxsmm_barrier_wait(handle->barrier, ltid);
/* Perform reduction because we used thread private filters... */
if (handle->upd_use_external_reduce == 0) {
  const int total_filter_size = reduce_work * handle->ofmblock;

  if ( ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ) {
    for ( j = reduce_thr_begin; j < reduce_thr_end; j++) {
      for (pc = 0; pc < handle->ofmblock; pc++) {
        weight_ptr[(j*16)+pc] = (element_filter_type)0;
      }
    }
  }

  for ( i = 0; i < handle->desc.threads; i++ ) {
    remote_weight_ptr = ((element_filter_type*)handle->scratch4) + (i*total_filter_size);
    for ( j = reduce_thr_begin; j < reduce_thr_end; j++) {
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
      __m512 remote_weight;
      __m512 reduction_weight;
      __m512 sum_weight;
      remote_weight = LIBXSMM_INTRINSICS_MM512_LOAD_PS(remote_weight_ptr + (j*16));
      reduction_weight = LIBXSMM_INTRINSICS_MM512_LOAD_PS(weight_ptr + (j*16));
      sum_weight =  _mm512_add_ps( remote_weight, reduction_weight);
      _mm512_store_ps((void*) &weight_ptr[j*16] , sum_weight);
#endif
    }
  }
}
libxsmm_barrier_wait(handle->barrier, ltid);

