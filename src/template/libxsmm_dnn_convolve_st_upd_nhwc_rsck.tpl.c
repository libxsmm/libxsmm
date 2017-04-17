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
/* Rajkishore Barik, Ankush Mandal (Intel Corp.)
******************************************************************************/

int img, ofm1, ifm1, num_ofw_strips, num_ofh_strips, oi_, oj_, oi__, oj__, ii_, ij_, kh, kw, ofm1ifm1, ki, kj;
#if defined(LIBXSMM_WU_PER_THREAD_ALLOCATION) || defined(INPUT_PADDING)
int imgifm1;
#endif
#if defined(LIBXSMM_WU_TRANSPOSE_OFW_IFM)
int imgifhp;
#endif
#if defined(LIBXSMM_WU_PER_THREAD_ALLOCATION)
int i, j, ofm1ifm1img;
#endif
/* computing first logical thread */
const int ltid = tid - start_thread;

/* number of tasks that could be run in parallel */
const int work = handle->blocksifm*handle->blocksofm;
/* compute chunck size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

#if defined(LIBXSMM_WU_PER_THREAD_ALLOCATION)
/* number of tasks that could be run in parallel */
const int img_parallel_work = handle->blocksifm*handle->blocksofm*handle->desc.N;
/* compute chunck size */
const int img_parallel_chunksize = (img_parallel_work % handle->desc.threads == 0) ? (img_parallel_work / handle->desc.threads) : (img_parallel_work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
const int img_parallel_thr_begin = (ltid * img_parallel_chunksize < img_parallel_work) ? (ltid * img_parallel_chunksize) : img_parallel_work;
const int img_parallel_thr_end = ((ltid + 1) * img_parallel_chunksize < img_parallel_work) ? ((ltid + 1) * img_parallel_chunksize) : img_parallel_work;
#endif

#ifdef LIBXSMM_WU_TRANSPOSE_OFW_IFM
int ifm2;
/* number of tasks that could be run in parallel */
const int transpose_work = handle->desc.N*handle->ifhp;
/* compute chunck size */
const int transpose_chunksize = (transpose_work % handle->desc.threads == 0) ? (transpose_work / handle->desc.threads) : (transpose_work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
const int transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
const int transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;
#endif

/* avoid warning by using the xconv.sconv sequence to get some fn. ptr. to act as source of the type-cast */
libxsmm_convfunction jitted_conv_wu_no_pf = (libxsmm_convfunction)handle->code_upd[0].xconv.sconv;
#if !defined(LIBXSMM_CONV_NO_PREFETCH) && !defined(LIBXSMM_WU_TRANSPOSE_OFW_IFM)
libxsmm_convfunction jitted_conv_wu_nooutput_pf = (libxsmm_convfunction)handle->code_upd[2].xconv.sconv;
libxsmm_convfunction jitted_conv_wu_pf = (libxsmm_convfunction)handle->code_upd[1].xconv.sconv;
#endif
#ifdef LIBXSMM_WU_TRANSPOSE_OFW_IFM
#ifdef LIBXSMM_CONV_NO_PREFETCH
libxsmm_convfunction jitted_conv_wu_transpose_no_pf = (libxsmm_convfunction)handle->code_upd[3].xconv.sconv;
#endif
libxsmm_convfunction jitted_conv_wu_transpose_pf = (libxsmm_convfunction)handle->code_upd[4].xconv.sconv;
libxsmm_convfunction jitted_conv_wu_transpose_nooutput_pf = (libxsmm_convfunction)handle->code_upd[5].xconv.sconv;
#endif

element_output_type *const out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->blocksofm * handle->ofmblock;
LIBXSMM_VLA_DECL(5, element_output_type, output, out, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
LIBXSMM_VLA_DECL(6, element_filter_type, weight, (element_filter_type*)handle->grad_filter->data, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);

element_input_type *l_input;
element_filter_type *l_wt;
element_output_type* l_output;

unsigned int stride_w = handle->desc.v;
unsigned int stride_h = handle->desc.u;

#ifdef LIBXSMM_WU_PER_THREAD_ALLOCATION
element_filter_type* remote_weight_ptr = 0;
element_filter_type* weight_ptr = (element_filter_type*)handle->grad_filter->data;
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

#ifdef LIBXSMM_WU_PER_THREAD_ALLOCATION
LIBXSMM_VLA_DECL(6, element_filter_type, opt_weight_ptr, per_thread_weight, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);
#elif !defined(LIBXSMM_WU_TRANSPOSE_OFW_IFM)
LIBXSMM_VLA_DECL(6, element_filter_type, opt_weight_ptr, weight, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);
#endif

#if defined(INPUT_PADDING)
/* Define variables if padding is required */
element_input_type *LIBXSMM_RESTRICT copy_ptr;
const int padded_h = handle->ifhp + 2 * handle->desc.pad_h;
const int padded_w = handle->ifwp + 2 * handle->desc.pad_w;
libxsmm_xmatcopyfunction jitted_matzero = handle->matcopy_upd[1].xmatcopy;
LIBXSMM_VLA_DECL(5, const element_input_type, input_nopad, (element_input_type*)handle->reg_input->data, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
LIBXSMM_VLA_DECL(5, element_input_type, input_padded, (element_input_type*)handle->scratch5, padded_h, padded_w, handle->blocksifm, handle->ifmblock);
LIBXSMM_VLA_DECL(5, element_input_type, input, (element_input_type*)handle->scratch5, padded_h, padded_w, handle->blocksifm, handle->ifmblock);
#ifdef LIBXSMM_WU_TRANSPOSE_OFW_IFM
LIBXSMM_VLA_DECL(5, element_input_type, tr_input, (element_input_type*)handle->scratch5, padded_h, handle->blocksifm, handle->ifmblock, padded_w);
#else
element_input_type *LIBXSMM_RESTRICT input_ptr, *prefetch_ptr;
libxsmm_xmatcopyfunction jitted_matcopy = handle->matcopy_upd[0].xmatcopy;
const int copywork = handle->desc.N*handle->ifhp;
const int copychunksize = (copywork % handle->desc.threads == 0) ? (copywork / handle->desc.threads) : (copywork / handle->desc.threads) + 1;
const int copy_thr_begin = (ltid * copychunksize < copywork) ? (ltid * copychunksize) : copywork;
const int copy_thr_end = ((ltid + 1) * copychunksize < copywork) ? ((ltid + 1) * copychunksize) : copywork;
#endif
const int zerowork = handle->desc.N*padded_h;
const int zerochunksize = (zerowork % handle->desc.threads == 0) ? (zerowork / handle->desc.threads) : (zerowork / handle->desc.threads) + 1;
const int zero_thr_begin = (ltid * zerochunksize < zerowork) ? (ltid * zerochunksize) : zerowork;
const int zero_thr_end = ((ltid + 1) * zerochunksize < zerowork) ? ((ltid + 1) * zerochunksize) : zerowork;
#else
/* Define variables if padding is not required */
const int padded_h = handle->ifhp;
const int padded_w = handle->ifwp;
LIBXSMM_VLA_DECL(5, element_input_type, input, (element_input_type*)handle->reg_input->data, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
#ifdef LIBXSMM_WU_TRANSPOSE_OFW_IFM
LIBXSMM_VLA_DECL(5, element_input_type, tr_input, (element_input_type*)handle->scratch3, handle->ifhp, handle->blocksifm, handle->ifmblock, handle->ifwp);
#endif
#endif

unsigned int input_w = padded_w, input_h = padded_h;

kh = handle->desc.R;
kw = handle->desc.S;

#define LIBXSMM_JITTED_CONV_WU_NO_PF_NHWC_RSCK(input, i_img, i_ifm1, i_ij, i_ii, i_ifm2, \
                                     weight, w_ofm1, w_ifm1, w_kj, w_ki, w_ifm2, w_ofm2, \
                                     output, o_img, o_ofm1, o_oj, o_oi, o_ofm2) \
jitted_conv_wu_no_pf(  \
                       &LIBXSMM_VLA_ACCESS(5, input, (i_img), (i_ij), (i_ii), (i_ifm1), (i_ifm2), input_h, input_w, handle->blocksifm, handle->ifmblock), \
                       &LIBXSMM_VLA_ACCESS(6, weight, (w_kj), (w_ki), (w_ifm1), (w_ifm2), (w_ofm1), (w_ofm2), handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), \
                       &LIBXSMM_VLA_ACCESS(5, output, (o_img), (o_oj), (o_oi), (o_ofm1), (o_ofm2), handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock), \
                       NULL, \
                       NULL, \
                       NULL  \
                    )

#ifdef LIBXSMM_CONV_NO_PREFETCH
#define LIBXSMM_JITTED_CONV_WU_PF_NHWC_RSCK(input, i_img, i_ifm1, i_ij, i_ii, i_ifm2, \
                                  weight, w_ofm1, w_ifm1, w_kj, w_ki, w_ifm2, w_ofm2, \
                                  output, o_img, o_ofm1, o_oj, o_oi, o_ofm2, \
                                  pf_input, pi_img, pi_ifm1, pi_ij, pi_ii, pi_ifm2, \
                                  pf_weight, pw_ofm1, pw_ifm1, pw_kj, pw_ki, pw_ifm2, pw_ofm2, \
                                  pf_output, po_img, po_ofm1, po_oj, po_oi, po_ofm2) \
jitted_conv_wu_no_pf(  \
                       &LIBXSMM_VLA_ACCESS(5, input, (i_img), (i_ij), (i_ii), (i_ifm1), (i_ifm2), input_h, input_w, handle->blocksifm, handle->ifmblock), \
                       &LIBXSMM_VLA_ACCESS(6, weight, (w_kj), (w_ki), (w_ifm1), (w_ifm2), (w_ofm1), (w_ofm2), handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), \
                       &LIBXSMM_VLA_ACCESS(5, output, (o_img), (o_oj), (o_oi), (o_ofm1), (o_ofm2), handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock), \
                       NULL, \
                       NULL, \
                       NULL  \
                    )
#else
#define LIBXSMM_JITTED_CONV_WU_PF_NHWC_RSCK(input, i_img, i_ifm1, i_ij, i_ii, i_ifm2, \
                                  weight, w_ofm1, w_ifm1, w_kj, w_ki, w_ifm2, w_ofm2, \
                                  output, o_img, o_ofm1, o_oj, o_oi, o_ofm2, \
                                  pf_input, pi_img, pi_ifm1, pi_ij, pi_ii, pi_ifm2, \
                                  pf_weight, pw_ofm1, pw_ifm1, pw_kj, pw_ki, pw_ifm2, pw_ofm2, \
                                  pf_output, po_img, po_ofm1, po_oj, po_oi, po_ofm2) \
jitted_conv_wu_pf(  \
                    &LIBXSMM_VLA_ACCESS(5, input, (i_img), (i_ij), (i_ii), (i_ifm1), (i_ifm2), input_h, input_w, handle->blocksifm, handle->ifmblock), \
                    &LIBXSMM_VLA_ACCESS(6, weight, (w_kj), (w_ki), (w_ifm1), (w_ifm2), (w_ofm1), (w_ofm2), handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), \
                    &LIBXSMM_VLA_ACCESS(5, output, (o_img), (o_oj), (o_oi), (o_ofm1), (o_ofm2), handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock), \
                    &LIBXSMM_VLA_ACCESS(5, pf_input, (pi_img), (pi_ij), (pi_ii), (pi_ifm1), (pi_ifm2), input_h, input_w, handle->blocksifm, handle->ifmblock), \
                    &LIBXSMM_VLA_ACCESS(6, pf_weight, (pw_kj), (pw_ki), (pw_ifm1), (pw_ifm2), (pw_ofm1), (pw_ofm2), handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), \
                    &LIBXSMM_VLA_ACCESS(5, pf_output, (po_img), (po_oj), (po_oi), (po_ofm1), (po_ofm2), handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock) \
                 )
#endif

#ifdef LIBXSMM_CONV_NO_PREFETCH
#define LIBXSMM_JITTED_CONV_WU_NOOUTPUT_PF_NHWC_RSCK(input, i_img, i_ifm1, i_ij, i_ii, i_ifm2, \
                                           weight, w_ofm1, w_ifm1, w_kj, w_ki, w_ifm2, w_ofm2, \
                                           output, o_img, o_ofm1, o_oj, o_oi, o_ofm2, \
                                           pf_input, pi_img, pi_ifm1, pi_ij, pi_ii, pi_ifm2, \
                                           pf_weight, pw_ofm1, pw_ifm1, pw_kj, pw_ki, pw_ifm2, pw_ofm2) \
jitted_conv_wu_no_pf(  \
                       &LIBXSMM_VLA_ACCESS(5, input, (i_img), (i_ij), (i_ii), (i_ifm1), (i_ifm2), input_h, input_w, handle->blocksifm, handle->ifmblock), \
                       &LIBXSMM_VLA_ACCESS(6, weight, (w_kj), (w_ki), (w_ifm1), (w_ifm2), (w_ofm1), (w_ofm2), handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), \
                       &LIBXSMM_VLA_ACCESS(5, output, (o_img), (o_oj), (o_oi), (o_ofm1), (o_ofm2), handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock), \
                       NULL, \
                       NULL, \
                       NULL  \
                    )
#else
#define LIBXSMM_JITTED_CONV_WU_NOOUTPUT_PF_NHWC_RSCK(input, i_img, i_ifm1, i_ij, i_ii, i_ifm2, \
                                           weight, w_ofm1, w_ifm1, w_kj, w_ki, w_ifm2, w_ofm2, \
                                           output, o_img, o_ofm1, o_oj, o_oi, o_ofm2, \
                                           pf_input, pi_img, pi_ifm1, pi_ij, pi_ii, pi_ifm2, \
                                           pf_weight, pw_ofm1, pw_ifm1, pw_kj, pw_ki, pw_ifm2, pw_ofm2) \
jitted_conv_wu_nooutput_pf(  \
                             &LIBXSMM_VLA_ACCESS(5, input, (i_img), (i_ij), (i_ii), (i_ifm1), (i_ifm2), input_h, input_w, handle->blocksifm, handle->ifmblock), \
                             &LIBXSMM_VLA_ACCESS(6, weight, (w_kj), (w_ki), (w_ifm1), (w_ifm2), (w_ofm1), (w_ofm2), handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), \
                             &LIBXSMM_VLA_ACCESS(5, output, (o_img), (o_oj), (o_oi), (o_ofm1), (o_ofm2), handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock), \
                             &LIBXSMM_VLA_ACCESS(5, pf_input, (pi_img), (pi_ij), (pi_ii), (pi_ifm1), (pi_ifm2), input_h, input_w, handle->blocksifm, handle->ifmblock), \
                             &LIBXSMM_VLA_ACCESS(6, pf_weight, (pw_kj), (pw_ki), (pw_ifm1), (pw_ifm2), (pw_ofm1), (pw_ofm2), handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), \
                             NULL \
                          )
#endif

#ifdef LIBXSMM_WU_TRANSPOSE_OFW_IFM
#define LIBXSMM_JITTED_CONV_WU_TRANSPOSE_NO_PF_NHWC_RSCK(input, i_img, i_ifm1, i_ij, i_ii, i_ifm2, \
                                     weight, w_ofm1, w_ifm1, w_kj, w_ki, w_ifm2, w_ofm2, \
                                     output, o_img, o_ofm1, o_oj, o_oi, o_ofm2) \
jitted_conv_wu_transpose_no_pf(  \
                       &LIBXSMM_VLA_ACCESS(5, input, (i_img), (i_ij), (i_ifm1), (i_ifm2), (i_ii), input_h, handle->blocksifm, handle->ifmblock, input_w), \
                       &LIBXSMM_VLA_ACCESS(6, weight, (w_kj), (w_ki), (w_ifm1), (w_ifm2), (w_ofm1), (w_ofm2), handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), \
                       &LIBXSMM_VLA_ACCESS(5, output, (o_img), (o_oj), (o_oi), (o_ofm1), (o_ofm2), handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock), \
                       NULL, \
                       NULL, \
                       NULL  \
                    )

#ifdef LIBXSMM_CONV_NO_PREFETCH
#define LIBXSMM_JITTED_CONV_WU_TRANSPOSE_PF_NHWC_RSCK(input, i_img, i_ifm1, i_ij, i_ii, i_ifm2, \
                                  weight, w_ofm1, w_ifm1, w_kj, w_ki, w_ifm2, w_ofm2, \
                                  output, o_img, o_ofm1, o_oj, o_oi, o_ofm2, \
                                  pf_input, pi_img, pi_ifm1, pi_ij, pi_ii, pi_ifm2, \
                                  pf_weight, pw_ofm1, pw_ifm1, pw_kj, pw_ki, pw_ifm2, pw_ofm2, \
                                  pf_output, po_img, po_ofm1, po_oj, po_oi, po_ofm2) \
jitted_conv_wu_transpose_no_pf(  \
                       &LIBXSMM_VLA_ACCESS(5, input, (i_img), (i_ij), (i_ifm1), (i_ifm2), (i_ii), input_h, handle->blocksifm, handle->ifmblock, input_w), \
                       &LIBXSMM_VLA_ACCESS(6, weight, (w_kj), (w_ki), (w_ifm1), (w_ifm2), (w_ofm1), (w_ofm2), handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), \
                       &LIBXSMM_VLA_ACCESS(5, output, (o_img), (o_oj), (o_oi), (o_ofm1), (o_ofm2), handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock), \
                       NULL, \
                       NULL, \
                       NULL  \
                    )
#else
#define LIBXSMM_JITTED_CONV_WU_TRANSPOSE_PF_NHWC_RSCK(input, i_img, i_ifm1, i_ij, i_ii, i_ifm2, \
                                  weight, w_ofm1, w_ifm1, w_kj, w_ki, w_ifm2, w_ofm2, \
                                  output, o_img, o_ofm1, o_oj, o_oi, o_ofm2, \
                                  pf_input, pi_img, pi_ifm1, pi_ij, pi_ii, pi_ifm2, \
                                  pf_weight, pw_ofm1, pw_ifm1, pw_kj, pw_ki, pw_ifm2, pw_ofm2, \
                                  pf_output, po_img, po_ofm1, po_oj, po_oi, po_ofm2) \
jitted_conv_wu_transpose_pf(  \
                    &LIBXSMM_VLA_ACCESS(5, input, (i_img), (i_ij), (i_ifm1), (i_ifm2), (i_ii), input_h, handle->blocksifm, handle->ifmblock, input_w), \
                    &LIBXSMM_VLA_ACCESS(6, weight, (w_kj), (w_ki), (w_ifm1), (w_ifm2), (w_ofm1), (w_ofm2), handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), \
                    &LIBXSMM_VLA_ACCESS(5, output, (o_img), (o_oj), (o_oi), (o_ofm1), (o_ofm2), handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock), \
                    &LIBXSMM_VLA_ACCESS(5, pf_input, (pi_img), (pi_ij), (pi_ifm1), (pi_ifm2), (pi_ii), input_h, handle->blocksifm, handle->ifmblock, input_w), \
                    &LIBXSMM_VLA_ACCESS(6, pf_weight, (pw_kj), (pw_ki), (pw_ifm1), (pw_ifm2), (pw_ofm1), (pw_ofm2), handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), \
                    &LIBXSMM_VLA_ACCESS(5, pf_output, (po_img), (po_oj), (po_oi), (po_ofm1), (po_ofm2), handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock) \
                 )
#endif

#ifdef LIBXSMM_CONV_NO_PREFETCH
#define LIBXSMM_JITTED_CONV_WU_TRANSPOSE_NOOUTPUT_PF_NHWC_RSCK(input, i_img, i_ifm1, i_ij, i_ii, i_ifm2, \
                                           weight, w_ofm1, w_ifm1, w_kj, w_ki, w_ifm2, w_ofm2, \
                                           output, o_img, o_ofm1, o_oj, o_oi, o_ofm2, \
                                           pf_input, pi_img, pi_ifm1, pi_ij, pi_ii, pi_ifm2, \
                                           pf_weight, pw_ofm1, pw_ifm1, pw_kj, pw_ki, pw_ifm2, pw_ofm2) \
jitted_conv_wu_transpose_no_pf(  \
                       &LIBXSMM_VLA_ACCESS(5, input, (i_img), (i_ij), (i_ifm1), (i_ifm2), (i_ii), input_h, handle->blocksifm, handle->ifmblock, input_w), \
                       &LIBXSMM_VLA_ACCESS(6, weight, (w_kj), (w_ki), (w_ifm1), (w_ifm2), (w_ofm1), (w_ofm2), handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), \
                       &LIBXSMM_VLA_ACCESS(5, output, (o_img), (o_oj), (o_oi), (o_ofm1), (o_ofm2), handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock), \
                       NULL, \
                       NULL, \
                       NULL  \
                    )
#else
#define LIBXSMM_JITTED_CONV_WU_TRANSPOSE_NOOUTPUT_PF_NHWC_RSCK(input, i_img, i_ifm1, i_ij, i_ii, i_ifm2, \
                                           weight, w_ofm1, w_ifm1, w_kj, w_ki, w_ifm2, w_ofm2, \
                                           output, o_img, o_ofm1, o_oj, o_oi, o_ofm2, \
                                           pf_input, pi_img, pi_ifm1, pi_ij, pi_ii, pi_ifm2, \
                                           pf_weight, pw_ofm1, pw_ifm1, pw_kj, pw_ki, pw_ifm2, pw_ofm2) \
jitted_conv_wu_transpose_nooutput_pf(  \
                             &LIBXSMM_VLA_ACCESS(5, input, (i_img), (i_ij), (i_ifm1), (i_ifm2), (i_ii), input_h, handle->blocksifm, handle->ifmblock, input_w), \
                             &LIBXSMM_VLA_ACCESS(6, weight, (w_kj), (w_ki), (w_ifm1), (w_ifm2), (w_ofm1), (w_ofm2), handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), \
                             &LIBXSMM_VLA_ACCESS(5, output, (o_img), (o_oj), (o_oi), (o_ofm1), (o_ofm2), handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock), \
                             &LIBXSMM_VLA_ACCESS(5, pf_input, (pi_img), (pi_ij), (pi_ifm1), (pi_ifm2), (pi_ii), input_h, handle->blocksifm, handle->ifmblock, input_w), \
                             &LIBXSMM_VLA_ACCESS(6, pf_weight, (pw_kj), (pw_ki), (pw_ifm1), (pw_ifm2), (pw_ofm1), (pw_ofm2), handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock), \
                             NULL \
                          )
#endif
#endif

if (libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  ||
    libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ||
    libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM  || /* ) {
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
} else if (*/ libxsmm_target_archid == LIBXSMM_X86_AVX2) {

#if defined(INPUT_PADDING)
  libxsmm_barrier_init(handle->barrier, ltid);

  /* Initialize in parallel scratch5 to zero */
  for (imgifm1 = zero_thr_begin; imgifm1 < zero_thr_end; ++imgifm1) {
    const int ii = imgifm1 % padded_h;
    img = imgifm1 / padded_h;
    copy_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_padded, img, ii, 0, 0, 0, padded_h, padded_w, handle->blocksifm, handle->ifmblock);
    jitted_matzero(NULL, NULL, copy_ptr, NULL, NULL);
  }

  libxsmm_barrier_wait(handle->barrier, ltid);

  /* Copy the minibatch to a padded version only if no transpose is required -- otherwise we combine the transpose with the copying into the padded buffer */
#ifndef LIBXSMM_WU_TRANSPOSE_OFW_IFM
  for (imgifm1 = copy_thr_end - 1; imgifm1 >= copy_thr_begin; imgifm1--) {
    const int ii = imgifm1 % handle->ifhp;
    img = imgifm1 / handle->ifhp;
    input_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_nopad, img, ii, 0, 0, 0, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
    copy_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_padded, img, ii + handle->desc.pad_h, handle->desc.pad_w, 0, 0, padded_h, padded_w, handle->blocksifm, handle->ifmblock);
    if (ii != 0) {
      prefetch_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_nopad, img, ii - 1, 0, 0, 0, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
    }
    else {
      prefetch_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, input_nopad, img - 1, handle->ifhp - 1, 0, 0, 0, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
    }
    jitted_matcopy(input_ptr, NULL, copy_ptr, NULL, prefetch_ptr);
  }
  libxsmm_barrier_wait(handle->barrier, ltid);
#endif
#endif

  if (libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  ||
      libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ||
      libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM)
  {
    num_ofw_strips = handle->ofw / handle->upd_ofw_rb;
    num_ofh_strips = handle->ofh / handle->upd_ofh_rb;

#ifdef LIBXSMM_WU_TRANSPOSE_OFW_IFM
    libxsmm_barrier_init(handle->barrier, ltid); /* lazy barrier initialization */
#if defined(INPUT_PADDING)
    /* Transpose IFW and IFM into the padded buffer!*/
    for (imgifhp = transpose_thr_begin; imgifhp < transpose_thr_end; ++imgifhp) {
      int ij = imgifhp % handle->ifhp, ii;
      img = imgifhp / handle->ifhp;
      for (ii = 0; ii < handle->ifwp; ++ii) {
        for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
          for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
            LIBXSMM_VLA_ACCESS(5, tr_input, img, ij + handle->desc.pad_h, ifm1, ifm2, ii + handle->desc.pad_w, padded_h, handle->blocksifm, handle->ifmblock, padded_w)
              = LIBXSMM_VLA_ACCESS(5, input_nopad, img, ij, ii, ifm1, ifm2, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
          }
        }
      }
    }
#else
    /* First transpose IFW and IFM */
    for (imgifhp = transpose_thr_begin; imgifhp < transpose_thr_end; ++imgifhp) {
      int ij = imgifhp % handle->ifhp, ii;
      img = imgifhp / handle->ifhp;
      for (ii = 0; ii < handle->ifwp; ++ii) {
        for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
          for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
            LIBXSMM_VLA_ACCESS(5, tr_input, img, ij, ifm1, ifm2, ii, handle->ifhp, handle->blocksifm, handle->ifmblock, handle->ifwp)
              = LIBXSMM_VLA_ACCESS(5, input, img, ij, ii, ifm1, ifm2, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
          }
        }
      }
    }
#endif
    libxsmm_barrier_wait(handle->barrier, ltid);

    for (ofm1ifm1 = thr_begin; ofm1ifm1 < thr_end; ++ofm1ifm1) {
      ofm1 = ofm1ifm1 / handle->blocksifm;
      ifm1 = ofm1ifm1 % handle->blocksifm;
      for (img = 0; img < handle->desc.N; ++img) {
        for (oi__ = 0; oi__ < num_ofw_strips; ++oi__) {
          for (oj__ = 0; oj__ < num_ofh_strips; ++oj__) {
            oi_ = oi__*handle->upd_ofw_rb;
            oj_ = oj__*handle->upd_ofh_rb;
            ii_ = oi_*stride_w;
            ij_ = oj_*stride_h;
            for (kj = 0; kj < kh - 1; ++kj) {
              if (handle->ifmblock > 1) {
                for (ki = 0; ki < kw - 1; ++ki) {
                  /*jitted_conv_wu_nooutput_pf(l_input, l_wt, l_output, &(input[img][ij_+kj][ii_+ki+1][ifm1][0]), &(weight[ofm1][ifm1][kj][ki+1][0][0]), NULL);*/
                  LIBXSMM_JITTED_CONV_WU_TRANSPOSE_NOOUTPUT_PF_NHWC_RSCK(
                    tr_input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                    weight, ofm1, ifm1, kj, ki, 0, 0,
                    output, img, ofm1, oj_, oi_, 0,
                    tr_input, img, ifm1, ij_ + kj, ii_ + ki + 1, 0,
                    weight, ofm1, ifm1, kj, ki + 1, 0, 0
                  );
                }
                /* kw-1 */
                ki = kw - 1;
              }
              else {
                /*kw is unrolled when ifmblock == 1*/
                ki = 0;
              } /*end of ifmblock > 1 */
              /*jitted_conv_wu_nooutput_pf(l_input, l_wt, l_output, &(input[img][ij_+kj+1][ii_+0][ifm1][0]), &(weight[ofm1][ifm1][kj+1][ki][0][0]), NULL);*/
              LIBXSMM_JITTED_CONV_WU_TRANSPOSE_NOOUTPUT_PF_NHWC_RSCK(
                tr_input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                weight, ofm1, ifm1, kj, ki, 0, 0,
                output, img, ofm1, oj_, oi_, 0,
                tr_input, img, ifm1, ij_ + kj + 1, ii_ + 0, 0,
                weight, ofm1, ifm1, kj + 1, 0, 0, 0
              );
            }
            kj = kh - 1;
            if (handle->ifmblock > 1) {
              for (ki = 0; ki < kw - 1; ++ki) {
                /*jitted_conv_wu_nooutput_pf(l_input, l_wt, l_output, &(input[img][ij_+kj][ii_+ki+1][ifm1][0]), &(weight[ofm1][ifm1][kj][ki+1][0][0]), NULL);*/
                LIBXSMM_JITTED_CONV_WU_TRANSPOSE_NOOUTPUT_PF_NHWC_RSCK(
                  tr_input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                  weight, ofm1, ifm1, kj, ki, 0, 0,
                  output, img, ofm1, oj_, oi_, 0,
                  tr_input, img, ifm1, ij_ + kj, ii_ + ki + 1, 0,
                  weight, ofm1, ifm1, kj, ki + 1, 0, 0
                );
              }
              ki = kw - 1;
            }
            else {
              /*kw is unrolled when ifmblock == 1*/
              ki = 0;
            } /* end of ifmblock > 1 */
            if ((oi__ + 1 == num_ofw_strips) && (oj__ + 1 == num_ofh_strips)) {
              if ((img + 1 == handle->desc.N) && (ifm1 + 1 == handle->blocksifm)) {  /* prefetch next ofm1 */
                /* 1 - prefetch for kj=0, ki=0; */
                /*jitted_conv_wu_pf(l_input, l_wt, l_output, &(input[0][0][0][0][0]), &(weight[ofm1+1][0][0][0][0][0]), &(output[0][0][0][ofm1+1][0]));*/
                LIBXSMM_JITTED_CONV_WU_TRANSPOSE_PF_NHWC_RSCK(
                  tr_input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                  weight, ofm1, ifm1, kj, ki, 0, 0,
                  output, img, ofm1, oj_, oi_, 0,
                  tr_input, 0, 0, 0, 0, 0,
                  weight, ofm1 + 1, 0, 0, 0, 0, 0,
                  output, 0, ofm1 + 1, 0, 0, 0
                );
              }
              else {
                if (img + 1 == handle->desc.N) {
                  /* 1 - prefetch for kj=0, ki=0; */
                  /*jitted_conv_wu_pf(l_input, l_wt, l_output, &(input[0][0][0][ifm1+1][0]), &(weight[ofm1][ifm1+1][0][0][0][0]), &(output[0][0][0][ofm1][0]));*/
                  LIBXSMM_JITTED_CONV_WU_TRANSPOSE_PF_NHWC_RSCK(
                    tr_input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                    weight, ofm1, ifm1, kj, ki, 0, 0,
                    output, img, ofm1, oj_, oi_, 0,
                    tr_input, 0, ifm1 + 1, 0, 0, 0,
                    weight, ofm1, ifm1 + 1, 0, 0, 0, 0,
                    output, 0, ofm1, 0, 0, 0
                  );
                }
                else {
                  /* 1 - prefetch for kj=0, ki=0; */
                  /*jitted_conv_wu_pf(l_input, l_wt, l_output, &(input[img+1][0][0][ifm1][0]), &(weight[ofm1][ifm1][0][0][0][0]), &(output[img+1][0][0][ofm1][0]));*/
                  LIBXSMM_JITTED_CONV_WU_TRANSPOSE_PF_NHWC_RSCK(
                    tr_input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                    weight, ofm1, ifm1, kj, ki, 0, 0,
                    output, img, ofm1, oj_, oi_, 0,
                    tr_input, img + 1, ifm1, 0, 0, 0,
                    weight, ofm1, ifm1, 0, 0, 0, 0,
                    output, img + 1, ofm1, 0, 0, 0
                  );
                }
              }
            }
            else if (oj__ + 1 == num_ofh_strips) {  /* end of oj_*/
           /* 1 - prefetch for kj=0, ki=0; */
           /*jitted_conv_wu_pf(l_input, l_wt, l_output, &(input[img][0][((oi__+1)*handle->upd_ofw_rb)*stride_w][ifm1][0]), &(weight[ofm1][ifm1][0][0][0][0]), &(output[img][0][(oi__+1)*handle->upd_ofw_rb][ofm1][0]));*/
              LIBXSMM_JITTED_CONV_WU_TRANSPOSE_PF_NHWC_RSCK(
                tr_input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                weight, ofm1, ifm1, kj, ki, 0, 0,
                output, img, ofm1, oj_, oi_, 0,
                tr_input, img, ifm1, 0, ((oi__ + 1)*handle->upd_ofw_rb)*stride_w, 0,
                weight, ofm1, ifm1, 0, 0, 0, 0,
                output, img, ofm1, 0, (oi__ + 1)*handle->upd_ofw_rb, 0
              );
            }
            else { /* end of oi */
           /* 1 - prefetch for kj=0, ki=0; */
           /*jitted_conv_wu_pf(l_input, l_wt, l_output, &(input[img][((oj__+1)*handle->upd_ofh_rb)*stride_h][ii_][ifm1][0]), &(weight[ofm1][ifm1][0][0][0][0]), &(output[img][(oj__+1)*handle->upd_ofh_rb][ii_][ofm1][0]));*/
              LIBXSMM_JITTED_CONV_WU_TRANSPOSE_PF_NHWC_RSCK(
                tr_input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                weight, ofm1, ifm1, kj, ki, 0, 0,
                output, img, ofm1, oj_, oi_, 0,
                tr_input, img, ifm1, ((oj__ + 1)*handle->upd_ofh_rb)*stride_h, ii_, 0,
                weight, ofm1, ifm1, 0, 0, 0, 0,
                output, img, ofm1, (oj__ + 1)*handle->upd_ofh_rb, oi_, 0
              );
            } /* else end */
          } /* ofh_stip*/
        } /* ofw_strip */
      } /* img loop */
    } /* ifm1, ofm1 loop */
#else /* TRANSPOSE_OFW_IFM */
    if ((handle->blocksifm * handle->blocksofm) < (2 * handle->desc.threads)) { /* special case for not enough parallelism */
#ifdef LIBXSMM_WU_PER_THREAD_ALLOCATION
      for (i = 0; i < handle->blocksofm*handle->blocksifm*handle->desc.R*handle->desc.S*handle->ifmblock*handle->ofmblock; i++) {
        per_thread_weight_ptr[i] = (element_filter_type)0;
      }
      /* lazy barrier init */
      if (handle->upd_use_external_reduce == 0) {
        libxsmm_barrier_init(handle->barrier, ltid);
      }
#endif
#ifdef LIBXSMM_WU_PER_THREAD_ALLOCATION
      for (ofm1ifm1img = img_parallel_thr_begin; ofm1ifm1img < img_parallel_thr_end; ++ofm1ifm1img) {
        ofm1 = ofm1ifm1img / (handle->blocksifm * handle->desc.N);
        imgifm1 = ofm1ifm1img % (handle->blocksifm * handle->desc.N);
        ifm1 = imgifm1 / handle->desc.N;
        img = imgifm1 % handle->desc.N;
        {
#else
      for (ofm1ifm1 = thr_begin; ofm1ifm1 < thr_end; ++ofm1ifm1) {
        ofm1 = ofm1ifm1 / handle->blocksifm;
        ifm1 = ofm1ifm1 % handle->blocksifm;
        for (img = 0; img < handle->desc.N; img++) {
#endif
          for (oi__ = 0; oi__ < num_ofw_strips; ++oi__) {
            for (oj__ = 0; oj__ < num_ofh_strips; ++oj__) {
              oi_ = oi__*handle->upd_ofw_rb;
              oj_ = oj__*handle->upd_ofh_rb;
              ii_ = oi_*stride_w;
              ij_ = oj_*stride_h;
              for (kj = 0; kj < kh - 1; ++kj) {
                if (handle->ifmblock > 1) {
                  for (ki = 0; ki < kw - 1; ++ki) {
                    /*jitted_conv_wu_nooutput_pf(l_input, l_wt, l_output, &(input[img][ij_+kj][ii_+ki+1][ifm1][0]), &(weight[ofm1][ifm1][kj][ki+1][0][0]), NULL);*/
                    LIBXSMM_JITTED_CONV_WU_NOOUTPUT_PF_NHWC_RSCK(
                      input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                      opt_weight_ptr, ofm1, ifm1, kj, ki, 0, 0,
                      output, img, ofm1, oj_, oi_, 0,
                      input, img, ifm1, ij_ + kj, ii_ + ki + 1, 0,
                      opt_weight_ptr, ofm1, ifm1, kj, ki + 1, 0, 0
                    );
                  }
                  /* kw-1 */
                  ki = kw - 1;
                }
                else {
                  /*kw is unrolled when ifmblock == 1*/
                  ki = 0;
                } /* end of ifmblock > 1 */
                /*jitted_conv_wu_nooutput_pf(l_input, l_wt, l_output, &(input[img][ij_+kj+1][ii_+0][ifm1][0]), &(weight[ofm1][ifm1][kj+1][ki][0][0]), NULL);*/
                LIBXSMM_JITTED_CONV_WU_NOOUTPUT_PF_NHWC_RSCK(
                  input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                  opt_weight_ptr, ofm1, ifm1, kj, ki, 0, 0,
                  output, img, ofm1, oj_, oi_, 0,
                  input, img, ifm1, ij_ + kj + 1, ii_ + 0, 0,
                  opt_weight_ptr, ofm1, ifm1, kj + 1, 0, 0, 0
                );
              }
              kj = kh - 1;
              if (handle->ifmblock > 1) {
                for (ki = 0; ki < kw - 1; ++ki) {
                  /*jitted_conv_wu_nooutput_pf(l_input, l_wt, l_output, &(input[img][ij_+kj][ii_+ki+1][ifm1][0]), &(weight[ofm1][ifm1][kj][ki+1][0][0]), NULL);*/
                  LIBXSMM_JITTED_CONV_WU_NOOUTPUT_PF_NHWC_RSCK(
                    input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                    opt_weight_ptr, ofm1, ifm1, kj, ki, 0, 0,
                    output, img, ofm1, oj_, oi_, 0,
                    input, img, ifm1, ij_ + kj, ii_ + ki + 1, 0,
                    opt_weight_ptr, ofm1, ifm1, kj, ki + 1, 0, 0
                  );
                }
                ki = kw - 1;
              }
              else {
                /*kw is unrolled when ifmblock == 1*/
                ki = 0;
              } /* end of ifmblock > 1 */
              if ((oi__ + 1 == num_ofw_strips) && (oj__ + 1 == num_ofh_strips)) {
                if ((img + 1 == handle->desc.N) && (ifm1 + 1 == handle->blocksifm)) {  /* prefetch next ofm1 */
                  /* 1 - prefetch for kj=0, ki=0; */
                  /*jitted_conv_wu_pf(l_input, l_wt, l_output, &(input[0][0][0][0][0]), &(weight[ofm1+1][0][0][0][0][0]), &(output[0][0][0][ofm1+1][0]));*/
                  LIBXSMM_JITTED_CONV_WU_PF_NHWC_RSCK(
                    input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                    opt_weight_ptr, ofm1, ifm1, kj, ki, 0, 0,
                    output, img, ofm1, oj_, oi_, 0,
                    input, 0, 0, 0, 0, 0,
                    opt_weight_ptr, ofm1 + 1, 0, 0, 0, 0, 0,
                    output, 0, ofm1 + 1, 0, 0, 0
                  );
                }
                else {
                  if (img + 1 == handle->desc.N) {
                    /* 1 - prefetch for kj=0, ki=0; */
                    /*jitted_conv_wu_pf(l_input, l_wt, l_output, &(input[0][0][0][ifm1+1][0]), &(weight[ofm1][ifm1+1][0][0][0][0]), &(output[0][0][0][ofm1][0]));*/
                    LIBXSMM_JITTED_CONV_WU_PF_NHWC_RSCK(
                      input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                      opt_weight_ptr, ofm1, ifm1, kj, ki, 0, 0,
                      output, img, ofm1, oj_, oi_, 0,
                      input, 0, ifm1 + 1, 0, 0, 0,
                      opt_weight_ptr, ofm1, ifm1 + 1, 0, 0, 0, 0,
                      output, 0, ofm1, 0, 0, 0
                    );
                  }
                  else {
                    /* 1 - prefetch for kj=0, ki=0; */
                    /*jitted_conv_wu_pf(l_input, l_wt, l_output, &(input[img+1][0][0][ifm1][0]), &(weight[ofm1][ifm1][0][0][0][0]), &(output[img+1][0][0][ofm1][0]));*/
                    LIBXSMM_JITTED_CONV_WU_PF_NHWC_RSCK(
                      input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                      opt_weight_ptr, ofm1, ifm1, kj, ki, 0, 0,
                      output, img, ofm1, oj_, oi_, 0,
                      input, img + 1, ifm1, 0, 0, 0,
                      opt_weight_ptr, ofm1, ifm1, 0, 0, 0, 0,
                      output, img + 1, ofm1, 0, 0, 0
                    );
                  }
                }
              }
              else if (oj__ + 1 == num_ofh_strips) {  /* end of oj_*/
                /* 1 - prefetch for kj=0, ki=0; */
                /*jitted_conv_wu_pf(l_input, l_wt, l_output, &(input[img][0][((oi__+1)*handle->upd_ofw_rb)*stride_w][ifm1][0]), &(weight[ofm1][ifm1][0][0][0][0]), &(output[img][0][(oi__+1)*handle->upd_ofw_rb][ofm1][0]));*/
                LIBXSMM_JITTED_CONV_WU_PF_NHWC_RSCK(
                  input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                  opt_weight_ptr, ofm1, ifm1, kj, ki, 0, 0,
                  output, img, ofm1, oj_, oi_, 0,
                  input, img, ifm1, 0, ((oi__ + 1)*handle->upd_ofw_rb)*stride_w, 0,
                  opt_weight_ptr, ofm1, ifm1, 0, 0, 0, 0,
                  output, img, ofm1, 0, (oi__ + 1)*handle->upd_ofw_rb, 0
                );
              }
              else { /* end of oj */
                /* 1 - prefetch for kj=0, ki=0; */
                /*jitted_conv_wu_pf(l_input, l_wt, l_output, &(input[img][((oj__+1)*handle->upd_ofh_rb)*stride_h][ii_][ifm1][0]), &(weight[ofm1][ifm1][0][0][0][0]), &(output[img][(oj__+1)*handle->upd_ofh_rb][ii_][ofm1][0]));*/
                LIBXSMM_JITTED_CONV_WU_PF_NHWC_RSCK(
                  input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                  opt_weight_ptr, ofm1, ifm1, kj, ki, 0, 0,
                  output, img, ofm1, oj_, oi_, 0,
                  input, img, ifm1, ((oj__ + 1)*handle->upd_ofh_rb)*stride_h, ii_, 0,
                  opt_weight_ptr, ofm1, ifm1, 0, 0, 0, 0,
                  output, img, ofm1, (oj__ + 1)*handle->upd_ofh_rb, oi_, 0
                );
              } /* else end */
            } /* ofh_strip loop */
          } /* ofw_strip loop */
        } /* img loop */
      } /* ifm1, ofm1 loops */
#ifdef LIBXSMM_WU_PER_THREAD_ALLOCATION
      /* perform reduction */
      /* TODO COMPLETE THIS USING ATOMIC INCREMENTS PLEASE */
      if (handle->upd_use_external_reduce == 0) {
        libxsmm_barrier_wait(handle->barrier, ltid);
        for (i = 0; i < handle->desc.threads; i++) {
          remote_weight_ptr = ((element_filter_type*)handle->scratch4) + (i*reduce_work);
          for (j = reduce_thr_begin; j < reduce_thr_end; j++) {
            weight_ptr[j] += remote_weight_ptr[j];
          }
        }
      }
#endif
    }
    else { /* enough parallelism available */
      for (ofm1ifm1 = thr_begin; ofm1ifm1 < thr_end; ++ofm1ifm1) {
        ofm1 = ofm1ifm1 / handle->blocksifm;
        ifm1 = ofm1ifm1 % handle->blocksifm;
        for (img = 0; img < handle->desc.N; ++img) {
          for (oi__ = 0; oi__ < num_ofw_strips; ++oi__) {
            for (oj__ = 0; oj__ < num_ofh_strips; ++oj__) {
              oi_ = oi__*handle->upd_ofw_rb;
              oj_ = oj__*handle->upd_ofh_rb;
              ii_ = oi_*stride_w;
              ij_ = oj_*stride_h;
              for (kj = 0; kj < kh - 1; ++kj) {
                if (handle->ifmblock > 1) {
                  for (ki = 0; ki < kw - 1; ++ki) {
                    /*jitted_conv_wu_nooutput_pf(l_input, l_wt, l_output, &(input[img][ij_+kj][ii_+ki+1][ifm1][0]), &(weight[ofm1][ifm1][kj][ki+1][0][0]), NULL);*/
                    LIBXSMM_JITTED_CONV_WU_NOOUTPUT_PF_NHWC_RSCK(
                      input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                      weight, ofm1, ifm1, kj, ki, 0, 0,
                      output, img, ofm1, oj_, oi_, 0,
                      input, img, ifm1, ij_ + kj, ii_ + ki + 1, 0,
                      weight, ofm1, ifm1, kj, ki + 1, 0, 0
                    );
                  }
                  /* kw-1 */
                  ki = kw - 1;
                }
                else {
                  /*kw is unrolled when ifmblock == 1*/
                  ki = 0;
                } /*end of ifmblock > 1 */
                /*jitted_conv_wu_nooutput_pf(l_input, l_wt, l_output, &(input[img][ij_+kj+1][ii_+0][ifm1][0]), &(weight[ofm1][ifm1][kj+1][ki][0][0]), NULL);*/
                LIBXSMM_JITTED_CONV_WU_NOOUTPUT_PF_NHWC_RSCK(
                  input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                  weight, ofm1, ifm1, kj, ki, 0, 0,
                  output, img, ofm1, oj_, oi_, 0,
                  input, img, ifm1, ij_ + kj + 1, ii_ + 0, 0,
                  weight, ofm1, ifm1, kj + 1, 0, 0, 0
                );
              }
              kj = kh - 1;
              if (handle->ifmblock > 1) {
                for (ki = 0; ki < kw - 1; ++ki) {
                  /*jitted_conv_wu_nooutput_pf(l_input, l_wt, l_output, &(input[img][ij_+kj][ii_+ki+1][ifm1][0]), &(weight[ofm1][ifm1][kj][ki+1][0][0]), NULL);*/
                  LIBXSMM_JITTED_CONV_WU_NOOUTPUT_PF_NHWC_RSCK(
                    input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                    weight, ofm1, ifm1, kj, ki, 0, 0,
                    output, img, ofm1, oj_, oi_, 0,
                    input, img, ifm1, ij_ + kj, ii_ + ki + 1, 0,
                    weight, ofm1, ifm1, kj, ki + 1, 0, 0
                  );
                }
                ki = kw - 1;
              }
              else {
                /*kw is unrolled when ifmblock == 1*/
                ki = 0;
              } /* end of ifmblock > 1 */
              if ((oi__ + 1 == num_ofw_strips) && (oj__ + 1 == num_ofh_strips)) {
                if ((img + 1 == handle->desc.N) && (ifm1 + 1 == handle->blocksifm)) {  /* prefetch next ofm1 */
                  /* 1 - prefetch for kj=0, ki=0; */
                  /*jitted_conv_wu_pf(l_input, l_wt, l_output, &(input[0][0][0][0][0]), &(weight[ofm1+1][0][0][0][0][0]), &(output[0][0][0][ofm1+1][0]));*/
                  LIBXSMM_JITTED_CONV_WU_PF_NHWC_RSCK(
                    input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                    weight, ofm1, ifm1, kj, ki, 0, 0,
                    output, img, ofm1, oj_, oi_, 0,
                    input, 0, 0, 0, 0, 0,
                    weight, ofm1 + 1, 0, 0, 0, 0, 0,
                    output, 0, ofm1 + 1, 0, 0, 0
                  );
                }
                else {
                  if (img + 1 == handle->desc.N) {
                    /* 1 - prefetch for kj=0, ki=0; */
                    /*jitted_conv_wu_pf(l_input, l_wt, l_output, &(input[0][0][0][ifm1+1][0]), &(weight[ofm1][ifm1+1][0][0][0][0]), &(output[0][0][0][ofm1][0]));*/
                    LIBXSMM_JITTED_CONV_WU_PF_NHWC_RSCK(
                      input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                      weight, ofm1, ifm1, kj, ki, 0, 0,
                      output, img, ofm1, oj_, oi_, 0,
                      input, 0, ifm1 + 1, 0, 0, 0,
                      weight, ofm1, ifm1 + 1, 0, 0, 0, 0,
                      output, 0, ofm1, 0, 0, 0
                    );
                  }
                  else {
                    /* 1 - prefetch for kj=0, ki=0; */
                    /*jitted_conv_wu_pf(l_input, l_wt, l_output, &(input[img+1][0][0][ifm1][0]), &(weight[ofm1][ifm1][0][0][0][0]), &(output[img+1][0][0][ofm1][0]));*/
                    LIBXSMM_JITTED_CONV_WU_PF_NHWC_RSCK(
                      input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                      weight, ofm1, ifm1, kj, ki, 0, 0,
                      output, img, ofm1, oj_, oi_, 0,
                      input, img + 1, ifm1, 0, 0, 0,
                      weight, ofm1, ifm1, 0, 0, 0, 0,
                      output, img + 1, ofm1, 0, 0, 0
                    );
                  }
                }
              }
              else if (oj__ + 1 == num_ofh_strips) {  /* end of oj_*/
             /* 1 - prefetch for kj=0, ki=0; */
             /*jitted_conv_wu_pf(l_input, l_wt, l_output, &(input[img][0][((oi__+1)*handle->upd_ofw_rb)*stride_w][ifm1][0]), &(weight[ofm1][ifm1][0][0][0][0]), &(output[img][0][(oi__+1)*handle->upd_ofw_rb][ofm1][0]));*/
                LIBXSMM_JITTED_CONV_WU_PF_NHWC_RSCK(
                  input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                  weight, ofm1, ifm1, kj, ki, 0, 0,
                  output, img, ofm1, oj_, oi_, 0,
                  input, img, ifm1, 0, ((oi__ + 1)*handle->upd_ofw_rb)*stride_w, 0,
                  weight, ofm1, ifm1, 0, 0, 0, 0,
                  output, img, ofm1, 0, (oi__ + 1)*handle->upd_ofw_rb, 0
                );
              }
              else { /* end of oi */
             /* 1 - prefetch for kj=0, ki=0; */
             /*jitted_conv_wu_pf(l_input, l_wt, l_output, &(input[img][((oj__+1)*handle->upd_ofh_rb)*stride_h][ii_][ifm1][0]), &(weight[ofm1][ifm1][0][0][0][0]), &(output[img][(oj__+1)*handle->upd_ofh_rb][ii_][ofm1][0]));*/
                LIBXSMM_JITTED_CONV_WU_PF_NHWC_RSCK(
                  input, img, ifm1, ij_ + kj, ii_ + ki, 0,
                  weight, ofm1, ifm1, kj, ki, 0, 0,
                  output, img, ofm1, oj_, oi_, 0,
                  input, img, ifm1, ((oj__ + 1)*handle->upd_ofh_rb)*stride_h, ii_, 0,
                  weight, ofm1, ifm1, 0, 0, 0, 0,
                  output, img, ofm1, (oj__ + 1)*handle->upd_ofh_rb, oi_, 0
                );
              } /* else end */
            } /* ofh_stip*/
          } /* ofw_strip */
        } /* img loop */
      } /* ifm1, ofm1 loop */
    } /*if enough parallelism end*/
#endif /* TRANSPOSE_OFW_IFM */
  }
  else if (libxsmm_target_archid == LIBXSMM_X86_AVX2) {

#ifdef LIBXSMM_WU_PER_THREAD_ALLOCATION
    for (i = 0; i < handle->blocksofm*handle->blocksifm*handle->desc.R*handle->desc.S*handle->ifmblock*handle->ofmblock; i++) {
      per_thread_weight_ptr[i] = (element_filter_type)0;
    }
    /* lazy barrier init */
    if (handle->upd_use_external_reduce == 0) {
      libxsmm_barrier_init(handle->barrier, ltid);
    }
    for (ofm1ifm1img = img_parallel_thr_begin; ofm1ifm1img < img_parallel_thr_end; ++ofm1ifm1img) {
      img = ofm1ifm1img / (handle->blocksifm * handle->blocksofm);
      ofm1ifm1 = ofm1ifm1img % (handle->blocksifm * handle->blocksofm);
      ofm1 = ofm1ifm1 / handle->blocksifm;
      ifm1 = ofm1ifm1 % handle->blocksifm;
      for (kj = 0; kj < handle->desc.R; ++kj) {
        for (ki = 0; ki < handle->desc.S; ++ki) {
          l_input = &LIBXSMM_VLA_ACCESS(5, input, img, kj, ki, ifm1, 0, padded_h, padded_w, handle->blocksifm, handle->ifmblock);
          l_wt = &LIBXSMM_VLA_ACCESS(6, per_thread_weight, kj, ki, ifm1, 0, ofm1, 0, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);
          l_output = &LIBXSMM_VLA_ACCESS(5, output, img, 0, 0, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
          jitted_conv_wu_no_pf(l_input, l_wt, l_output, NULL, NULL, NULL);
        }
      }
    }
    if (handle->upd_use_external_reduce == 0) {
      libxsmm_barrier_wait(handle->barrier, ltid);
      /* reduce weights */
      for (i = 0; i < handle->desc.threads; i++) {
        remote_weight_ptr = ((element_filter_type*)handle->scratch4) + (i*reduce_work);
        for (j = reduce_thr_begin; j < reduce_thr_end; j++) {
          weight_ptr[j] += remote_weight_ptr[j];
        }
      }
    }
#else
    for (ofm1ifm1 = thr_begin; ofm1ifm1 < thr_end; ++ofm1ifm1) {
      ofm1 = ofm1ifm1 / handle->blocksifm;
      ifm1 = ofm1ifm1 % handle->blocksifm;
      for (img = 0; img < handle->desc.N; ++img) {
        for (kj = 0; kj < handle->desc.R; ++kj) {
          for (ki = 0; ki < handle->desc.S; ++ki) {
            l_input = &LIBXSMM_VLA_ACCESS(5, input, img, kj, ki, ifm1, 0, padded_h, padded_w, handle->blocksifm, handle->ifmblock);
            l_wt = &LIBXSMM_VLA_ACCESS(6, weight, kj, ki, ifm1, 0, ofm1, 0, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);
            l_output = &LIBXSMM_VLA_ACCESS(5, output, img, 0, 0, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
            jitted_conv_wu_no_pf(l_input, l_wt, l_output, NULL, NULL, NULL);
          }
        }
      }
    }
#endif
    if (ltid == (handle->desc.threads - 1)) {
      handle->filter_transposed = 0;
    }
  } /* end of LIBXSMM_X86_AVX512_KNM || LIBXSMM_X86_AVX2 */
  /* should never happen, this is just an additional check */
}
else {
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
}

#ifdef LIBXSMM_JITTED_CONV_WU_NO_PF_NHWC_RSCK
#undef LIBXSMM_JITTED_CONV_WU_NO_PF_NHWC_RSCK
#endif

#ifdef LIBXSMM_JITTED_CONV_WU_PF_NHWC_RSCK
#undef LIBXSMM_JITTED_CONV_WU_PF_NHWC_RSCK
#endif

#ifdef LIBXSMM_JITTED_CONV_WU_NOOUTPUT_PF_NHWC_RSCK
#undef LIBXSMM_JITTED_CONV_WU_NOOUTPUT_PF_NHWC_RSCK
#endif

#ifdef LIBXSMM_JITTED_CONV_WU_TRANSPOSE_NO_PF_NHWC_RSCK
#undef LIBXSMM_JITTED_CONV_WU_TRANSPOSE_NO_PF_NHWC_RSCK
#endif

#ifdef LIBXSMM_JITTED_CONV_WU_TRANSPOSE_PF_NHWC_RSCK
#undef LIBXSMM_JITTED_CONV_WU_TRANSPOSE_PF_NHWC_RSCK
#endif

#ifdef LIBXSMM_JITTED_CONV_WU_TRANSPOSE_NOOUTPUT_PF_NHWC_RSCK
#undef LIBXSMM_JITTED_CONV_WU_TRANSPOSE_NOOUTPUT_PF_NHWC_RSCK
#endif
