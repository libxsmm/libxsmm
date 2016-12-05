/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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
/* Rajkishore Barik (Intel Corp.)
******************************************************************************/

int img, ofm1, ifm1, ofm1ifm1, ki, kj;
#if defined(LIBXSMM_WU_PER_THREAD_ALLOCATION)
int i, j, ofm1ifm1img;
#endif
/* computing first logical thread */
const int ltid = tid-start_thread;

#if !defined(LIBXSMM_WU_PER_THREAD_ALLOCATION)
/* number of tasks that could be run in parallel */
const int work = handle->blocksifm*handle->blocksofm;
/* compute chunck size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
#else
/* number of tasks that could be run in parallel */
const int img_parallel_work = handle->blocksifm*handle->blocksofm*handle->desc.N;
/* compute chunck size */
const int img_parallel_chunksize = (img_parallel_work % handle->desc.threads == 0) ? (img_parallel_work / handle->desc.threads) : (img_parallel_work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
const int img_parallel_thr_begin = (ltid * img_parallel_chunksize < img_parallel_work) ? (ltid * img_parallel_chunksize) : img_parallel_work;
const int img_parallel_thr_end = ((ltid + 1) * img_parallel_chunksize < img_parallel_work) ? ((ltid + 1) * img_parallel_chunksize) : img_parallel_work;
#endif

/* avoid warning by using the xconv.sconv sequence to get some fn. ptr. to act as source of the type-cast */
libxsmm_convfunction jitted_conv_wu_no_pf = (libxsmm_convfunction)handle->code_upd[0].xconv.sconv;
#if 0
libxsmm_convfunction jitted_conv_wu_pf = (libxsmm_convfunction)handle->code_upd[1].xconv.sconv;
libxsmm_convfunction jitted_conv_wu_nooutput_pf = (libxsmm_convfunction)handle->code_upd[2].xconv.sconv;
#endif

element_output_type *const out = ((element_output_type*)handle->output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->blocksofm * handle->ofmblock;
LIBXSMM_VLA_DECL(5, element_output_type, output, out, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
LIBXSMM_VLA_DECL(5, element_input_type, input, (element_input_type*)handle->input->data, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
#if !defined(LIBXSMM_WU_PER_THREAD_ALLOCATION)
LIBXSMM_VLA_DECL(6, element_filter_type, weight, (element_filter_type*)handle->filter->data, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);
#endif

element_input_type *l_input;
element_filter_type *l_wt;
element_output_type* l_output;

#ifdef LIBXSMM_WU_PER_THREAD_ALLOCATION
element_filter_type* remote_weight_ptr = 0;
element_filter_type* weight_ptr = (element_filter_type*)handle->filter->data;
element_filter_type* per_thread_weight_ptr = ((element_filter_type*)handle->scratch4)
                                                + (ltid*handle->blocksofm*handle->blocksifm*handle->desc.R*handle->desc.S*handle->ifmblock*handle->ofmblock);
LIBXSMM_VLA_DECL(6, element_filter_type, per_thread_weight, per_thread_weight_ptr, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);
/* number of tasks that could be run in parallel */
const int reduce_work = handle->blocksofm*handle->blocksifm*handle->desc.R*handle->desc.S*handle->ifmblock*handle->ofmblock;
/* compute chunck size */
const int reduce_chunksize = (reduce_work % handle->desc.threads == 0) ? (reduce_work / handle->desc.threads) : (reduce_work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
const int reduce_thr_begin = (ltid * reduce_chunksize < reduce_work) ? (ltid * reduce_chunksize) : reduce_work;
const int reduce_thr_end = ((ltid + 1) * reduce_chunksize < reduce_work) ? ((ltid + 1) * reduce_chunksize) : reduce_work;
#endif

if ( libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_MIC ||
     libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_CORE  || /* ) {
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
} else if (*/ libxsmm_get_target_archid() == LIBXSMM_X86_AVX2 ){
#ifdef LIBXSMM_WU_PER_THREAD_ALLOCATION
  for(i=0; i<handle->blocksofm*handle->blocksifm*handle->desc.R*handle->desc.S*handle->ifmblock*handle->ofmblock; i++) {
    per_thread_weight_ptr[i] = (element_filter_type)0;
  }
  /* lazy barrier init */
  libxsmm_barrier_init((libxsmm_barrier*)handle->scratch2, ltid);
  for (ofm1ifm1img = img_parallel_thr_begin; ofm1ifm1img < img_parallel_thr_end; ++ofm1ifm1img) {
    img = ofm1ifm1img / (handle->blocksifm * handle->blocksofm);
    ofm1ifm1 = ofm1ifm1img % (handle->blocksifm * handle->blocksofm);
    ofm1 = ofm1ifm1 / handle->blocksifm;
    ifm1 = ofm1ifm1 % handle->blocksifm;
    for(kj=0; kj < handle->desc.R; ++kj) {
      for(ki=0; ki < handle->desc.S; ++ki) {
        l_input =  &LIBXSMM_VLA_ACCESS(5, input, img, kj, ki, ifm1, 0, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
        l_wt = &LIBXSMM_VLA_ACCESS(6, per_thread_weight, kj, ki, ifm1, 0, ofm1, 0, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);
        l_output = &LIBXSMM_VLA_ACCESS(5, output, img, 0, 0, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
        jitted_conv_wu_no_pf(l_input, l_wt, l_output, NULL, NULL, NULL );
      }
    }
  }
  libxsmm_barrier_wait((libxsmm_barrier*)handle->scratch2, ltid);
  /* reduce weights */
  for ( i = 0; i < handle->desc.threads; i++ ) {
    remote_weight_ptr = ((element_filter_type*)handle->scratch4) + (i*reduce_work);
    for ( j = reduce_thr_begin; j < reduce_thr_end; j++) {
      weight_ptr[j] += remote_weight_ptr[j];
    }
  }
#else
  for (ofm1ifm1 = thr_begin; ofm1ifm1 < thr_end; ++ofm1ifm1) {
    ofm1 = ofm1ifm1 / handle->blocksifm;
    ifm1 = ofm1ifm1 % handle->blocksifm;
    for(img = 0; img < handle->desc.N; ++img) {
      for(kj=0; kj < handle->desc.R; ++kj) {
        for(ki=0; ki < handle->desc.S; ++ki) {
          l_input =  &LIBXSMM_VLA_ACCESS(5, input, img, kj, ki, ifm1, 0, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
          l_wt = &LIBXSMM_VLA_ACCESS(6, weight, kj, ki, ifm1, 0, ofm1, 0, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);
          l_output = &LIBXSMM_VLA_ACCESS(5, output, img, 0, 0, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
          jitted_conv_wu_no_pf(l_input, l_wt, l_output, NULL, NULL, NULL );
        }
      }
    }
  }
#endif
/* should never happen, this is just an additional check */
} else {
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
}
