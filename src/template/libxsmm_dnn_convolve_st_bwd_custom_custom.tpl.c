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
int imgifm1, img, ofm1, ifm1, oj, ij, kj, ki, ifm2, ofm2, kh, kw, ifm1ofm1, ifh, ofh;

/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
const int work = handle->desc.N * handle->blocksifm;
/* compute chunck size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;


/* number of tasks that could be run in parallel */
const int transpose_work = handle->blocksofm * handle->blocksifm;
/* compute chunck size */
const int transpose_chunksize = (transpose_work % handle->desc.threads == 0) ? (transpose_work / handle->desc.threads) : ((transpose_work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
const int transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;

element_output_type *const out = ((element_output_type*)handle->output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;
LIBXSMM_VLA_DECL(5, element_output_type, del_out, out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
LIBXSMM_VLA_DECL(5, element_input_type, del_input, (element_input_type*)handle->input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
LIBXSMM_VLA_DECL(6, element_filter_type, wt, (element_filter_type*)handle->filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
LIBXSMM_VLA_DECL(6, element_filter_type, tr_wt, (element_filter_type*)handle->scratch1, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);

/* avoid warning by using the xconv.sconv sequence to get some fn. ptr. to act as source of the type-cast */
libxsmm_convfunction jitted_conv_bp_no_pf = (libxsmm_convfunction)handle->code_bwd[0].xconv.sconv;
#if defined(LIBXSMM_CONV_NO_PREFETCH)
libxsmm_convfunction jitted_conv_bp_peeled_no_pf = (libxsmm_convfunction)handle->code_bwd[2].xconv.sconv;
#else
libxsmm_convfunction jitted_conv_bp_pf = (libxsmm_convfunction)handle->code_bwd[1].xconv.sconv;
libxsmm_convfunction jitted_conv_bp_peeled_noweight_pf = (libxsmm_convfunction)handle->code_bwd[3].xconv.sconv;
#endif

element_input_type *l_input;
element_filter_type *l_wt;
element_output_type* l_output;

kh = handle->desc.R;
kw = handle->desc.S;

ifh=handle->desc.H;
ofh=handle->ofh;

/* lazy barrier init */
libxsmm_barrier_init((libxsmm_barrier*)handle->scratch2, ltid);

for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
  ofm1 = ifm1ofm1/handle->blocksifm;
  ifm1 = ifm1ofm1%handle->blocksifm;
  for(kj=0; kj < kh; ++kj) {
    for(ki=0; ki < kw; ++ki) {
      /* TODO: enable this later */
      /*transpose<VLEN,VLEN>(&wt[ofm1][ifm1][kj][ki][0][0],&tr_wt[ofm1][ifm1][kj][ki][0][0]);*/
      for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
        for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
          LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kj, ki, ofm2, ifm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock) =
            LIBXSMM_VLA_ACCESS(6, wt, ofm1, ifm1, kj, ki, ifm2, ofm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
        }
      }
    }
  }
}
libxsmm_barrier_wait((libxsmm_barrier*)handle->scratch2, ltid);

if ( libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_MIC ||
     libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_CORE   ) {
for (imgifm1 = thr_begin; imgifm1 < thr_end; ++imgifm1) {
  img = imgifm1/handle->blocksifm;
  ifm1 = imgifm1%handle->blocksifm;
  for (ofm1 = 0; ofm1 < handle->blocksofm; ++ofm1) {

#if !defined(LIBXSMM_CONV_NO_PREFETCH)
    /* NON PEELED PROLOGUE VERSION */
    if(kh == 3) { /* 3x3 convolution */

      /* Unroll 1 */
      ij = 0; kj=2, oj=ij-kh+kj+1;
      l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
      /* prefetch ij=2; kj=0 */
      /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][ij+2][0][0]), &(tr_wt[ofm1][ifm1][kh-0-1][0][0][0]), &(del_out[img][ofm1][ij+2-kh+0+1][0][0]));*/
      jitted_conv_bp_pf(l_input, l_wt, l_output,
                        &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij+2, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-0-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, ij+2-kh+0+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                       );

      /* Unroll 2 */
      ij = 1; kj=1, oj=ij-kh+kj+1;
      l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
      /* prefetch ij=2; kj=1 */
      /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][ij+1][0][0]), &(tr_wt[ofm1][ifm1][kh-1-1][0][0][0]), &(del_out[img][ofm1][ij+1-kh+1+1][0][0]));*/
      jitted_conv_bp_pf(l_input, l_wt, l_output,
                        &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij+1, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-1-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, ij+1-kh+1+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                       );

      /* Unroll 3 */
      ij = 1; kj=2, oj=ij-kh+kj+1;
      l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
      /* prefetch for ij = 2 and kj=2 */
      /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][ij+1][0][0]), &(tr_wt[ofm1][ifm1][kh-2-1][0][0][0]), &(del_out[img][ofm1][ij+1-kh+2+1][0][0]));*/
      jitted_conv_bp_pf(l_input, l_wt, l_output,
                        &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij+1, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-2-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, ij+1-kh+2+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                       );

    } else if (kh == 5) { /* kh=5 */
      /* Unroll 1 */
      ij = 0; kj=4; oj=ij-kh+kj+1;
      l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
      /* prefetch for ij= 2 kj=4 */
      /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][ij+2][0][0]), &(tr_wt[ofm1][ifm1][kh-4-1][0][0][0]), &(del_out[img][ofm1][ij+2-kh+4+1][0][0]));*/
      jitted_conv_bp_pf(l_input, l_wt, l_output,
                        &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij+2, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-4-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, ij+2-kh+4+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                       );

      /* Unroll 2 */
      ij = 1; kj=3; oj=ij-kh+kj+1;
      l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
      /* prefetch for ij= 3 kj=1 */
      /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][ij+2][0][0]), &(tr_wt[ofm1][ifm1][kh-1-1][0][0][0]), &(del_out[img][ofm1][ij+2-kh+1+1][0][0]));*/
      jitted_conv_bp_pf(l_input, l_wt, l_output,
                        &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij+2, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-1-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, ij+2-kh+1+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                       );

      /* Unroll 3 */
      ij = 1; kj=4; oj=ij-kh+kj+1;
      l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
      /* prefetch for ij= 3 kj=2 */
      /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][ij+2][0][0]), &(tr_wt[ofm1][ifm1][kh-2-1][0][0][0]), &(del_out[img][ofm1][ij+2-kh+2+1][0][0]));*/
      jitted_conv_bp_pf(l_input, l_wt, l_output,
                        &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij+2, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-2-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, ij+2-kh+2+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                       );

      /* Unroll 4 */
      ij = 2; kj=2; oj=ij-kh+kj+1;
      l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
      /* prefetch for ij= 3 kj=3 */
      /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][ij+1][0][0]), &(tr_wt[ofm1][ifm1][kh-3-1][0][0][0]), &(del_out[img][ofm1][ij+1-kh+3+1][0][0]));*/
      jitted_conv_bp_pf(l_input, l_wt, l_output,
                        &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij+1, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-3-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, ij+1-kh+3+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                       );

      /* Unroll 5 */
      ij = 2; kj=3; oj=ij-kh+kj+1;
      l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
      /* prefetch for ij= 3 kj=4 */
      /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][ij+1][0][0]), &(tr_wt[ofm1][ifm1][kh-4-1][0][0][0]), &(del_out[img][ofm1][ij+1-kh+4+1][0][0]));*/
      jitted_conv_bp_pf(l_input, l_wt, l_output,
                        &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij+1, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-4-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, ij+1-kh+4+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                       );

      /* Unroll 6 */
      ij = 2; kj=4; oj=ij-kh+kj+1;
      l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
      /* prefetch for ij= 4 kj=0 */
      /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][ij+2][0][0]), &(tr_wt[ofm1][ifm1][kh-0-1][0][0][0]), &(del_out[img][ofm1][ij+2-kh+0+1][0][0]));*/
      jitted_conv_bp_pf(l_input, l_wt, l_output,
                        &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij+2, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-0-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, ij+2-kh+0+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                       );

      /* Unroll 7 */
      ij = 3; kj=1; oj=ij-kh+kj+1;
      l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
      /* prefetch for ij= 4 kj=1 */
      /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][ij+1][0][0]), &(tr_wt[ofm1][ifm1][kh-1-1][0][0][0]), &(del_out[img][ofm1][ij+1-kh+1+1][0][0]));*/
      jitted_conv_bp_pf(l_input, l_wt, l_output,
                        &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij+1, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-1-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, ij+1-kh+1+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                       );

      /* Unroll 8 */
      ij = 3; kj=2; oj=ij-kh+kj+1;
      l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
      /* prefetch for ij= 4 kj=2 */
      /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][ij+1][0][0]), &(tr_wt[ofm1][ifm1][kh-2-1][0][0][0]), &(del_out[img][ofm1][ij+1-kh+2+1][0][0]));*/
      jitted_conv_bp_pf(l_input, l_wt, l_output,
                        &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij+1, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-2-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, ij+1-kh+2+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                       );

      /* Unroll 9 */
      ij = 3; kj=3; oj=ij-kh+kj+1;
      l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
      /* prefetch for ij= 4 kj=3 */
      /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][ij+1][0][0]), &(tr_wt[ofm1][ifm1][kh-3-1][0][0][0]), &(del_out[img][ofm1][ij+1-kh+3+1][0][0]));*/
      jitted_conv_bp_pf(l_input, l_wt, l_output,
                        &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij+1, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-3-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, ij+1-kh+3+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                       );

      /* Unroll 10 */
      ij = 3; kj=4; oj=ij-kh+kj+1;
      l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
      /* prefetch for ij= 4 kj=4 */
      /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][ij+1][0][0]), &(tr_wt[ofm1][ifm1][kh-4-1][0][0][0]), &(del_out[img][ofm1][ij+1-kh+4+1][0][0]));*/
      jitted_conv_bp_pf(l_input, l_wt, l_output,
                        &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij+1, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-4-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, ij+1-kh+4+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                       );
    } else {
      for(ij=0; ij < kh-1; ij++) {
        for(kj=0; kj < kh; kj++) {
          oj = ij - kh + kj + 1;
          if(oj >=0) {
            l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
            l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
            l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
            jitted_conv_bp_no_pf(l_input, l_wt, l_output, NULL, NULL, NULL );
          }
        }
      }
    }

    /* PEELED INNERMOST VERSION */
    for(ij=kh-1; ij < ifh-kh+1; ij++) {
      l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, ij-kh+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
      /*jitted_conv_bp_peeled_noweight_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][ij+1][0][0]), NULL, &(del_out[img][ofm1][ij+1-kh+1][0][0]));*/
      jitted_conv_bp_peeled_noweight_pf(l_input, l_wt, l_output,
                                        &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij+1, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                                        NULL,
                                        &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, ij+1-kh+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                                       );
    }


    /* NON PEELED EPILOGUE VERSION */
    if (kh==3) {

      if ( (ofm1+1 == handle->blocksofm) &&  (ifm1+1 == handle->blocksifm) ) { /* prefetch next img, kj=2, ij=0 */
        ij=ifh-2; kj=0; oj=ij-kh+kj+1; /* ifh-2-3+0+1 = ifh-4 */
        l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
        l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
        l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
        /* prefetch ij=0 and kj =2 */
        /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img+1][0][0][0][0]), &(tr_wt[0][0][kh-2-1][0][0][0]), &(del_out[img+1][0][-kh+2+1][0][0]));*/
        jitted_conv_bp_pf(l_input, l_wt, l_output,
                          &LIBXSMM_VLA_ACCESS(5, del_input, img+1, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                          &LIBXSMM_VLA_ACCESS(6, tr_wt, 0, 0, kh-2-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                          &LIBXSMM_VLA_ACCESS(5, del_out, img+1, 0, -kh+2+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                         );

        ij=ifh-2; kj=1; oj=ij-kh+kj+1; /* ifh-2-3+1+1 = ifh-3 */
        l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
        l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
        l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
        /* prefetch ij=1 and kj =1 */
        /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img+1][0][1][0][0]), &(tr_wt[0][0][kh-1-1][0][0][0]), &(del_out[img+1][0][1-kh+1+1][0][0]));*/
        jitted_conv_bp_pf(l_input, l_wt, l_output,
                          &LIBXSMM_VLA_ACCESS(5, del_input, img+1, 0, 1, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                          &LIBXSMM_VLA_ACCESS(6, tr_wt, 0, 0, kh-1-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                          &LIBXSMM_VLA_ACCESS(5, del_out, img+1, 0, 1-kh+1+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                         );

        ij=ifh-1; kj=0; oj=ij-kh+kj+1; /* ifh-1-3+0+1 = ifh-3 */
        l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
        l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
        l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
        /* prefetch ij=1, kj=2 */
        /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img+1][0][1][0][0]), &(tr_wt[0][0][kh-2-1][0][0][0]), &(del_out[img+1][0][1-kh+2+1][0][0]));*/
        jitted_conv_bp_pf(l_input, l_wt, l_output,
                          &LIBXSMM_VLA_ACCESS(5, del_input, img+1, 0, 1, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                          &LIBXSMM_VLA_ACCESS(6, tr_wt, 0, 0, kh-2-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                          &LIBXSMM_VLA_ACCESS(5, del_out, img+1, 0, 1-kh+2+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                         );
      } else {
        if (ofm1+1 == handle->blocksofm) { /* prefecth next ifm1, kj=2, ij = 0 */
          ij=ifh-2; kj=0; oj=ij-kh+kj+1; /* ifh-2-3+0+1 = ifh-4 */
          l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
          l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
          /* prefetch ij=0 and kj =2 */
          /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1+1][0][0][0]), &(tr_wt[0][ifm1+1][kh-2-1][0][0][0]), &(del_out[img][0][-kh+2+1][0][0]));*/
          jitted_conv_bp_pf(l_input, l_wt, l_output,
                            &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1+1, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-2-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(5, del_out, img, 0, -kh+2+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                           );
          ij=ifh-2; kj=1; oj=ij-kh+kj+1; /* ifh-2-3+1+1 = ifh-3 */
          l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
          l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
          /* prefetch ij=1 and kj =1 */
          /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1+1][1][0][0]), &(tr_wt[0][ifm1+1][kh-1-1][0][0][0]), &(del_out[img][0][1-kh+1+1][0][0]));*/
          jitted_conv_bp_pf(l_input, l_wt, l_output,
                            &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1+1, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-1-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(5, del_out, img, 0, 1-kh+1+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                           );
          ij=ifh-1; kj=0; oj=ij-kh+kj+1; /* ifh-1-3+0+1 = ifh-3 */
          l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
          l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
          /* prefetch ij=1, kj=2 */
          /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1+1][1][0][0]), &(tr_wt[0][ifm1+1][kh-2-1][0][0][0]), &(del_out[img][0][1-kh+2+1][0][0]));*/
          jitted_conv_bp_pf(l_input, l_wt, l_output,
                            &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1+1, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1,  kh-2-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(5, del_out, img, 0, 1-kh+2+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                           );
        } else {
          ij=ifh-2; kj=0; oj=ij-kh+kj+1; /* ifh-2-3+0+1 = ifh-4 */
          l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
          l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
          /* prefetch ij=0 and kj =2 */
          /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][0][0][0]), &(tr_wt[ofm1+1][ifm1][kh-2-1][0][0][0]), &(del_out[img][ofm1+1][-kh+2+1][0][0]));*/
          jitted_conv_bp_pf(l_input, l_wt, l_output,
                            &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1+1, ifm1,  kh-2-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1+1, -kh+2+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                           );
          ij=ifh-2; kj=1; oj=ij-kh+kj+1; /* ifh-2-3+1+1 = ifh-3 */
          l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
          l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
          /* prefetch ij=1 and kj =1 */
          /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][1][0][0]), &(tr_wt[ofm1+1][ifm1][kh-1-1][0][0][0]), &(del_out[img][ofm1+1][1-kh+1+1][0][0]));*/
          jitted_conv_bp_pf(l_input, l_wt, l_output,
                            &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, 1, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1+1, ifm1,  kh-1-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1+1, 1-kh+1+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                           );
          ij=ifh-1; kj=0; oj=ij-kh+kj+1; /* ifh-1-3+0+1 = ifh-3 */
          l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
          l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
          /* prefetch ij=1, kj=2 */
          /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][1][0][0]), &(tr_wt[ofm1+1][ifm1][kh-2-1][0][0][0]), &(del_out[img][ofm1+1][1-kh+2+1][0][0]));*/
          jitted_conv_bp_pf(l_input, l_wt, l_output,
                            &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, 1, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1+1, ifm1,  kh-2-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1+1, 1-kh+2+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                           );
        } /* if (ofm1+1 == nBOfm) */
      } /* if ( (ofm1+1 == nBOfm) &&  (ifm1+1 == nBIfm) ) */
    } else if (kh==5)  {  /* kh =5 */
      /* Unroll 1 */
      ij = ifh-4; kj=0; oj=ij-kh+kj+1; /* ifh-8 */
      l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
      /* prefetch for ij=ifh-3 kj=1 */
      /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][ifh-3][0][0]), &(tr_wt[ofm1][ifm1][kh-1-1][0][0][0]), &(del_out[img][ofm1][ifh-3-kh+1+1][0][0]));*/
      jitted_conv_bp_pf(l_input, l_wt, l_output,
                        &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ifh-3, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1,  kh-1-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, ifh-3-kh+1+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                       );

      /* Unroll 2 */
      ij = ifh-4; kj=1; oj=ij-kh+kj+1;
      l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
      /* prefetch for ij=ifh-3 kj=2 */
      /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][ifh-3][0][0]), &(tr_wt[ofm1][ifm1][kh-2-1][0][0][0]), &(del_out[img][ofm1][ifh-3-kh+2+1][0][0]));*/
      jitted_conv_bp_pf(l_input, l_wt, l_output,
                        &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ifh-3, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1,  kh-2-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, ifh-3-kh+2+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                       );

      /* Unroll 3 */
      ij = ifh-4; kj=2; oj=ij-kh+kj+1;
      l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
      /* prefetch for ij=ifh-2 kj=0 */
      /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][ifh-2][0][0]), &(tr_wt[ofm1][ifm1][kh-0-1][0][0][0]), &(del_out[img][ofm1][ifh-2-kh+0+1][0][0]));*/
      jitted_conv_bp_pf(l_input, l_wt, l_output,
                        &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ifh-2, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1,  kh-0-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, ifh-2-kh+0+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                       );

      /* Unroll 4 */
      ij = ifh-4; kj=3; oj=ij-kh+kj+1;
      l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
      /* prefetch for ij=ifh-2 kj=1 */
      /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][ifh-2][0][0]), &(tr_wt[ofm1][ifm1][kh-1-1][0][0][0]), &(del_out[img][ofm1][ifh-2-kh+1+1][0][0]));*/
      jitted_conv_bp_pf(l_input, l_wt, l_output,
                        &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ifh-2, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1,  kh-1-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, ifh-2-kh+1+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                       );

      /* Unroll 5 */
      ij = ifh-3; kj=0; oj=ij-kh+kj+1; /* ifh-4 */
      l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
      /* prefetch for ij=ifh-1 kj=0 */
      /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][ifh-1][0][0]), &(tr_wt[ofm1][ifm1][kh-0-1][0][0][0]), &(del_out[img][ofm1][ifh-1-kh+0+1][0][0]));*/
      jitted_conv_bp_pf(l_input, l_wt, l_output,
                        &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ifh-1, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1,  kh-0-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                        &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, ifh-1-kh+0+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                       );


      if ( (ofm1+1 == handle->blocksofm) &&  (ifm1+1 == handle->blocksifm) ) { /* prefetch next img, kj=4, ij=0 */
        /* Unroll 6 */
        ij = ifh-3; kj=1; oj=ij-kh+kj+1;
        l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
        l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
        l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
        /* prefetch for ij=0 kj=4 */
        /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img+1][0][0][0][0]), &(tr_wt[0][0][kh-4-1][0][0][0]), &(del_out[img+1][0][-kh+4+1][0][0]));*/
        jitted_conv_bp_pf(l_input, l_wt, l_output,
                          &LIBXSMM_VLA_ACCESS(5, del_input, img+1, 0, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                          &LIBXSMM_VLA_ACCESS(6, tr_wt, 0, 0,  kh-4-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                          &LIBXSMM_VLA_ACCESS(5, del_out, img+1, 0, -kh+4+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                         );

        /* Unroll 7 */
        ij = ifh-3; kj=2; oj=ij-kh+kj+1;
        l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
        l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
        l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
        /* prefetch for ij=1 kj=3 */
        /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img+1][0][1][0][0]), &(tr_wt[0][0][kh-3-1][0][0][0]), &(del_out[img+1][0][1-kh+3+1][0][0]));*/
        jitted_conv_bp_pf(l_input, l_wt, l_output,
                          &LIBXSMM_VLA_ACCESS(5, del_input, img+1, 0, 1, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                          &LIBXSMM_VLA_ACCESS(6, tr_wt, 0, 0,  kh-3-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                          &LIBXSMM_VLA_ACCESS(5, del_out, img+1, 0, 1-kh+3+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                         );

        /* Unroll 8 */
        ij = ifh-2; kj=0; oj=ij-kh+kj+1;
        l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
        l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
        l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
        /* prefetch for ij=1 kj=4 */
        /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img+1][0][1][0][0]), &(tr_wt[0][0][kh-4-1][0][0][0]), &(del_out[img+1][0][1-kh+4+1][0][0]));*/
        jitted_conv_bp_pf(l_input, l_wt, l_output,
                          &LIBXSMM_VLA_ACCESS(5, del_input, img+1, 0, 1, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                          &LIBXSMM_VLA_ACCESS(6, tr_wt, 0, 0,  kh-4-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                          &LIBXSMM_VLA_ACCESS(5, del_out, img+1, 0, 1-kh+4+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                         );

        /* Unroll 9 */
        ij = ifh-2; kj=1; oj=ij-kh+kj+1;
        l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
        l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
        l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
        /* prefetch for ij=2 kj=2 */
        /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img+1][0][2][0][0]), &(tr_wt[0][0][kh-2-1][0][0][0]), &(del_out[img+1][0][2-kh+2+1][0][0]));*/
        jitted_conv_bp_pf(l_input, l_wt, l_output,
                          &LIBXSMM_VLA_ACCESS(5, del_input, img+1, 0, 2, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                          &LIBXSMM_VLA_ACCESS(6, tr_wt, 0, 0,  kh-2-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                          &LIBXSMM_VLA_ACCESS(5, del_out, img+1, 0, 2-kh+2+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                         );

        /* Unroll 10 */
        ij = ifh-1; kj=0; oj=ij-kh+kj+1;
        l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
        l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
        l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
        /* ij=2 kj=3 */
        /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img+1][0][2][0][0]), &(tr_wt[0][0][kh-3-1][0][0][0]), &(del_out[img+1][0][2-kh+3+1][0][0]));*/
        jitted_conv_bp_pf(l_input, l_wt, l_output,
                          &LIBXSMM_VLA_ACCESS(5, del_input, img+1, 0, 2, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                          &LIBXSMM_VLA_ACCESS(6, tr_wt, 0, 0,  kh-3-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                          &LIBXSMM_VLA_ACCESS(5, del_out, img+1, 0, 2-kh+3+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                         );
      } else {
        if (ofm1+1 == handle->blocksofm) { /* prefecth next ifm1,  kj=4, ij=0 */
          /* Unroll 6 */
          ij = ifh-3; kj=1; oj=ij-kh+kj+1;
          l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
          l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
          /* prefetch for ij=0 kj=4 */
          /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1+1][0][0][0]), &(tr_wt[0][ifm1+1][kh-4-1][0][0][0]), &(del_out[img][0][-kh+4+1][0][0]));*/
          jitted_conv_bp_pf(l_input, l_wt, l_output,
                            &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1+1, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, tr_wt, 0, ifm1+1,  kh-4-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(5, del_out, img, 0, 0-kh+4+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                           );

          /* Unroll 7 */
          ij = ifh-3; kj=2; oj=ij-kh+kj+1;
          l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
          l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
          /* prefetch for ij=1 kj=3 */
          /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1+1][1][0][0]), &(tr_wt[0][ifm1+1][kh-3-1][0][0][0]), &(del_out[img][0][1-kh+3+1][0][0]));*/
          jitted_conv_bp_pf(l_input, l_wt, l_output,
                            &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1+1, 1, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, tr_wt, 0, ifm1+1,  kh-3-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(5, del_out, img, 0, 0-kh+3+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                           );

          /* Unroll 8 */
          ij = ifh-2; kj=0; oj=ij-kh+kj+1;
          l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
          l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
          /* prefetch for ij=1 kj=4 */
          /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1+1][1][0][0]), &(tr_wt[0][ifm1+1][kh-4-1][0][0][0]), &(del_out[img][0][1-kh+4+1][0][0]));*/
          jitted_conv_bp_pf(l_input, l_wt, l_output,
                            &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1+1, 1, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, tr_wt, 0, ifm1+1,  kh-4-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(5, del_out, img, 0, 1-kh+4+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                           );

          /* Unroll 9 */
          ij = ifh-2; kj=1; oj=ij-kh+kj+1;
          l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
          l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
          /* prefetch for ij=2 kj=2 */
          /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1+1][2][0][0]), &(tr_wt[0][ifm1+1][kh-2-1][0][0][0]), &(del_out[img][0][2-kh+2+1][0][0]));*/
          jitted_conv_bp_pf(l_input, l_wt, l_output,
                            &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1+1, 2, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, tr_wt, 0, ifm1+1,  kh-2-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(5, del_out, img, 0, 2-kh+2+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                           );

          /* Unroll 10 */
          ij = ifh-1; kj=0; oj=ij-kh+kj+1;
          l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
          l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
          /* ij=2 kj=3 */
          /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1+1][2][0][0]), &(tr_wt[0][ifm1+1][kh-3-1][0][0][0]), &(del_out[img][0][2-kh+3+1][0][0]));*/
          jitted_conv_bp_pf(l_input, l_wt, l_output,
                            &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1+1, 2, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, tr_wt, 0, ifm1+1,  kh-3-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(5, del_out, img, 0, 2-kh+3+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                           );
        } else {
          /* Unroll 6 */
          ij = ifh-3; kj=1; oj=ij-kh+kj+1;
          l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
          l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
          /* prefetch for ij=0 kj=4 */
          /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][0][0][0]), &(tr_wt[ofm1+1][ifm1][kh-4-1][0][0][0]), &(del_out[img][ofm1+1][-kh+4+1][0][0]));*/
          jitted_conv_bp_pf(l_input, l_wt, l_output,
                            &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1+1, ifm1,  kh-4-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1+1, -kh+4+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                           );
          /* Unroll 7 */
          ij = ifh-3; kj=2; oj=ij-kh+kj+1;
          l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
          l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
          /* prefetch for ij=1 kj=3 */
          /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][1][0][0]), &(tr_wt[ofm1+1][ifm1][kh-3-1][0][0][0]), &(del_out[img][ofm1+1][1-kh+3+1][0][0]));*/
          jitted_conv_bp_pf(l_input, l_wt, l_output,
                            &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, 1, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1+1, ifm1,  kh-3-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1+1, 1-kh+3+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                           );

          /* Unroll 8 */
          ij = ifh-2; kj=0; oj=ij-kh+kj+1;
          l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
          l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
          /* prefetch for ij=1 kj=4 */
          /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][1][0][0]), &(tr_wt[ofm1+1][ifm1][kh-4-1][0][0][0]), &(del_out[img][ofm1+1][1-kh+4+1][0][0]));*/
          jitted_conv_bp_pf(l_input, l_wt, l_output,
                            &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, 1, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1+1, ifm1,  kh-4-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1+1, 1-kh+4+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                           );

          /* Unroll 9 */
          ij = ifh-2; kj=1; oj=ij-kh+kj+1;
          l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
          l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
          /* prefetch for ij=2 kj=2 */
          /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][2][0][0]), &(tr_wt[ofm1+1][ifm1][kh-2-1][0][0][0]), &(del_out[img][ofm1+1][2-kh+2+1][0][0]));*/
          jitted_conv_bp_pf(l_input, l_wt, l_output,
                            &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, 2, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1+1, ifm1,  kh-2-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1+1, 2-kh+2+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                           );

          /* Unroll 10 */
          ij = ifh-1; kj=0; oj=ij-kh+kj+1;
          l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
          l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
          /* ij=2 kj=3 */
          /*jitted_conv_bp_pf(l_input, l_wt, l_output, &(del_input[img][ifm1][2][0][0]), &(tr_wt[ofm1+1][ifm1][kh-3-1][0][0][0]), &(del_out[img][ofm1+1][2-kh+3+1][0][0]));*/
          jitted_conv_bp_pf(l_input, l_wt, l_output,
                            &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, 2, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1+1, ifm1,  kh-3-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1+1, 2-kh+3+1, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock)
                           );
        }
      }
    } else {
      for(ij=ifh-kh +1; ij < ifh; ij++) {
        for(kj=0; kj < kh; kj++) {
          oj = ij - kh + kj + 1;
          if(oj < ofh) {
            l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
            l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
            l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
            jitted_conv_bp_no_pf(l_input, l_wt, l_output, NULL, NULL, NULL );
          }
        }
      }
    }
#else /* NO_PREFETCH */
    for(ij=0; ij < kh-1; ++ij) {
      for(kj=0; kj < kh; ++kj) {
        oj = ij - kh + kj + 1;
        if(oj >=0) {
          l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
          l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
          jitted_conv_bp_no_pf(l_input, l_wt, l_output, NULL, NULL, NULL );
        }
      }
    }
    for(ij=kh-1; ij < ifh-kh +1 ; ++ij) {
      oj = ij - kh + 1;
      l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
      l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
      l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
      jitted_conv_bp_peeled_no_pf(l_input, l_wt, l_output, NULL, NULL, NULL );
    }
    for(ij=ifh-kh +1 ; ij < ifh; ++ij) {
      for(kj=0; kj < kh; ++kj) {
        oj = ij - kh + kj + 1;
        if(oj < ofh) {
          l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
          l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
          l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
          jitted_conv_bp_no_pf(l_input, l_wt, l_output, NULL, NULL, NULL );
        }
      }
    }
#endif
  }
}
} else if ( libxsmm_get_target_archid() == LIBXSMM_X86_AVX2 ){
  for (imgifm1 = thr_begin; imgifm1 < thr_end; ++imgifm1) {
    img = imgifm1/handle->blocksifm;
    ifm1 = imgifm1%handle->blocksifm;
    for (ofm1 = 0; ofm1 < handle->blocksofm; ++ofm1) {
      for(ij= 0 ; ij < ifh; ++ij) {
        for(kj=0; kj < kh; ++kj) {
          oj = ij - kh + kj + 1;
          if(oj >= 0 && oj < ofh) {
            l_input =  &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
            l_wt = &LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kh-kj-1, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
            l_output = &LIBXSMM_VLA_ACCESS(5, del_out, img, ofm1, oj, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
            jitted_conv_bp_no_pf(l_input, l_wt, l_output, NULL, NULL, NULL );
          }
        }
      }
    }
  }
/* should never happen, this is just an additional check */
} else {
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
}
