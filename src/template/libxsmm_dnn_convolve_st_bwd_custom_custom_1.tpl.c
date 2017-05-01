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
/* Ankush Mandal, Rajkishore Barik, Alexander Heinecke (Intel Corp.)
******************************************************************************/

int imgifm1, img, ofm1, ifm1, oj, oi, ij, ii, kj, ki, ifm2, ofm2, ifm1ofm1, lp, ifm1lpblock;

/* computing first logical thread */
const int ltid = tid-start_thread;
/* number of tasks that could be run in parallel */
const int work = handle->desc.N * (handle->blocksifm * handle->fm_lp_block);
/* compute chunck size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : (work / handle->desc.threads) + 1;
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* number of tasks for transpose that could be run in parallel */
const int transpose_work = handle->blocksofm * (handle->blocksifm * handle->fm_lp_block);
/* compute chunck size */
const int transpose_chunksize = (transpose_work % handle->desc.threads == 0) ? (transpose_work / handle->desc.threads) : ((transpose_work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
const int transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;


/* Input tensor declaration */
/* regular/high precision */
element_input_type* del_in = 0;
/* low precision */
element_output_type* del_in_lp = 0;
/* select pointer based on precision */
if (handle->datatype != handle->datatype_itm) {
del_in = ((element_input_type*)handle->scratch7); /* + (handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in) * (handle->ifmblock); */
del_in_lp = ((element_output_type*)handle->grad_input->data); /* + (handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in) * (handle->ifmblock * handle->fm_lp_block); */
} else {
del_in = ((element_input_type*)handle->grad_input->data); /* + (handle->desc.pad_h_in * handle->ifwp + handle->desc.pad_w_in) * (handle->ifmblock); */
del_in_lp = 0;
}
{ /* open new scope for additional variable declarations (C89) */
LIBXSMM_VLA_DECL(5, element_input_type, del_input, del_in, handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock);
LIBXSMM_VLA_DECL(6, element_output_type, del_input_lp, del_in_lp, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block);
/* Ouput tensor declaration */
element_output_type *const out = ((element_output_type*)handle->grad_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock * handle->fm_lp_block;
LIBXSMM_VLA_DECL(6, element_output_type, del_out, out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block);

/* Weight and transpose_weight tensor declaration */
LIBXSMM_VLA_DECL(7, element_filter_type, wt, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
LIBXSMM_VLA_DECL(7, element_filter_type, tr_wt, (element_filter_type*)handle->scratch1, handle->blocksifm * handle->fm_lp_block, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block);

LIBXSMM_VLA_DECL(6, element_filter_type, temp_wt, (element_filter_type*)handle->scratch1 + (handle->blocksofm * handle->fm_lp_block * handle->blocksifm * handle->fm_lp_block * handle->desc.R * handle->desc.S * handle->ifmblock * handle->ofmblock), handle->blocksifm * handle->fm_lp_block, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);

/* JIT kernel function pointers */
libxsmm_convfunction jitted_conv_bp_no_pf, jitted_conv_bp_noweight_pf, jitted_conv_bp_pf;

#if defined(INPUT_PADDING)
element_input_type (* __restrict input_ptr);
element_input_type (* __restrict copy_ptr);
element_input_type *prefetch_ptr;
const int padded_h = handle->ifhp + 2 * handle->desc.pad_h;
const int padded_w = handle->ifwp + 2 * handle->desc.pad_w;
LIBXSMM_VLA_DECL(3, element_input_type, input_buffer, ((element_input_type*)handle->scratch5) + ltid * padded_h * padded_w * handle->ifmblock, padded_w, handle->ifmblock);
LIBXSMM_VLA_DECL(3, element_input_type, input_to_use, input_buffer, padded_w, handle->ifmblock);
libxsmm_xmatcopyfunction jitted_matcopy = handle->matcopy_bwd[0].xmatcopy;
libxsmm_xmatcopyfunction jitted_matcopyback = handle->matcopy_bwd[1].xmatcopy;
copy_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(3, input_buffer, handle->desc.pad_h, handle->desc.pad_w, 0, padded_w, handle->ifmblock);
input_ptr = NULL;
memset(&LIBXSMM_VLA_ACCESS(3, input_buffer, 0, 0, 0, padded_w, handle->ifmblock), 0, padded_w * padded_h * handle->ifmblock * sizeof(element_input_type));
#else
LIBXSMM_VLA_DECL(5, element_input_type, input_to_use, del_input, handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock);
#endif


#if 0
/* on KNM prefetches are less costly, so let's avoid some branch mispredicts by running redundant weight prefetches */
/* WARNING!! Currently nothing is being done for this in the code, i.e. NULL pointer is being passed on for extra prefetch */
if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM ) {
jitted_conv_bp_noweight_pf = (libxsmm_convfunction)handle->code_bwd[1].xconv.sconv;
} else {
jitted_conv_bp_noweight_pf = (libxsmm_convfunction)handle->code_bwd[3].xconv.sconv;
}
#endif


/* transpose last two dimensions of weight tensor for vectorization */
/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

if (handle->fm_lp_block == 1) {
  for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
    ofm1 = ifm1ofm1 / handle->blocksifm;
    ifm1 = ifm1ofm1 % handle->blocksifm;
    for (kj=0; kj < handle->desc.R; ++kj) {
      for (ki=0; ki < handle->desc.S; ++ki) {
        /* TODO: enable this later */
        /*transpose<VLEN,VLEN>(&wt[ofm1][ifm1][kj][ki][0][0],&tr_wt[ofm1][ifm1][kj][ki][0][0]);*/
        for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
          for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
              LIBXSMM_VLA_ACCESS(7, tr_wt, ofm1, ifm1, kj, ki, ofm2, ifm2, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block) =
              LIBXSMM_VLA_ACCESS(7, wt, ofm1, ifm1, kj, ki, ifm2, ofm2, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
          }
        }
      }
    }
  }
} else { /* If low precision */
  /* It's a little complicated for low precision */
  /* weight tensor is in [blocksofm * fm_lp_block][blocksifm][R][S][ifmblock][ofmblock][fm_lp_block] format */
  /* First we convert it to [blocksofm * fm_lp_block][blocksifm * fm_lp_block][R][S][ofm_block][ifmblock] format */
  for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
    ofm1 = ifm1ofm1 / handle->blocksifm; /* here ofm1 is iterator over (blocksofm * fm_lp_block) */
    ifm1 = ifm1ofm1 % handle->blocksifm;
    for (lp = 0; lp < handle->fm_lp_block; ++lp) {
      for (kj=0; kj < handle->desc.R; ++kj) {
        for (ki=0; ki < handle->desc.S; ++ki) {
          for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
            for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
              LIBXSMM_VLA_ACCESS(6, temp_wt, ofm1, (ifm1 * handle->fm_lp_block) + lp, kj, ki, ofm2, ifm2, handle->blocksifm * handle->fm_lp_block, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock) =
                LIBXSMM_VLA_ACCESS(7, wt, ofm1, ifm1, kj, ki, ((handle->ifmblock / handle->fm_lp_block) * lp) + (ifm2/handle->fm_lp_block), ofm2, (ifm2 % handle->fm_lp_block), handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock, handle->fm_lp_block);
            }
          }
        }
      }
    }
  }

  libxsmm_barrier_wait(handle->barrier, ltid);

  /* Then we convert the temporary weight tensor to
   * [blocksofm][blocksifm * fm_lp_block][R][S][ofm_block][ifm_block][fm_lp_block] format
   */
  for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
    ofm1 = ifm1ofm1 / (handle->blocksifm * handle->fm_lp_block);
    ifm1 = ifm1ofm1 % (handle->blocksifm * handle->fm_lp_block); /* here ifm1 is iterator over (blocksifm * fm_lp_block) */
    for (lp = 0; lp < handle->fm_lp_block; ++lp) {
      for (kj=0; kj < handle->desc.R; ++kj) {
        for (ki=0; ki < handle->desc.S; ++ki) {
          for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
            for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
              LIBXSMM_VLA_ACCESS(7, tr_wt, ofm1, ifm1, kj, ki, ((handle->ofmblock / handle->fm_lp_block) * lp) + (ofm2/handle->fm_lp_block), ifm2, (ofm2 % handle->fm_lp_block), handle->blocksifm * handle->fm_lp_block, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block) =
                LIBXSMM_VLA_ACCESS(6, temp_wt, (ofm1 *  handle->fm_lp_block) + lp, ifm1, kj, ki, ofm2, ifm2, handle->blocksifm * handle->fm_lp_block, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);
            }
          }
        }
      }
    }
  }
} /* end of if low precision check */

libxsmm_barrier_wait(handle->barrier, ltid);

/****************************************************************/
/*******Macros to call jitted function***************************/
/****************************************************************/
#if defined(INPUT_PADDING)

#define LIBXSMM_JITTED_CONV_BP_PF(input_to_use, i_img, i_ifm1, i_ij, i_ii, i_ifm2, \
                                  tr_wt, w_ofm1, w_ifm1, w_kj, w_ki, w_ofm2, w_ifm2, \
                                  del_out, o_img, o_ofm1, o_oj, o_oi, o_ofm2, \
                                  pf_input_to_use, pi_img, pi_ifm1, pi_ij, pi_ii, pi_ifm2, \
                                  pf_tr_wt, pw_ofm1, pw_ifm1, pw_kj, pw_ki, pw_ofm2, pw_ifm2, \
                                  pf_del_out, po_img, po_ofm1, po_oj, po_oi, po_ofm2) \
                    jitted_conv_bp_pf(  \
                        &LIBXSMM_VLA_ACCESS(3, input_to_use, (i_ij), (i_ii), (i_ifm2), padded_w, handle->ifmblock), \
                        &LIBXSMM_VLA_ACCESS(7, tr_wt, (w_ofm1), (w_ifm1), (w_kj), (w_ki), (w_ofm2), (w_ifm2), 0, handle->blocksifm * handle->fm_lp_block, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block), \
                        &LIBXSMM_VLA_ACCESS(6, del_out, (o_img), (o_ofm1), (o_oj), (o_oi), (o_ofm2), 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block), \
                        &LIBXSMM_VLA_ACCESS(3, pf_input_to_use, (pi_ij), (pi_ii), (pi_ifm2), padded_w, handle->ifmblock), \
                        &LIBXSMM_VLA_ACCESS(7, pf_tr_wt, (pw_ofm1), (pw_ifm1), (pw_kj), (pw_ki), (pw_ofm2), (pw_ifm2), 0, handle->blocksifm * handle->fm_lp_block, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block), \
                        &LIBXSMM_VLA_ACCESS(6, pf_del_out, po_img, po_ofm1, po_oj, po_oi, po_ofm2, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block) \
                       )
#define LIBXSMM_JITTED_CONV_BP_NOWEIGHT_PF(input_to_use, i_img, i_ifm1, i_ij, i_ii, i_ifm2, \
                                  tr_wt, w_ofm1, w_ifm1, w_kj, w_ki, w_ofm2, w_ifm2, \
                                  del_out, o_img, o_ofm1, o_oj, o_oi, o_ofm2, \
                                  pf_input_to_use, pi_img, pi_ifm1, pi_ij, pi_ii, pi_ifm2, \
                                  pf_del_out, po_img, po_ofm1, po_oj, po_oi, po_ofm2) \
                    jitted_conv_bp_noweight_pf(  \
                        &LIBXSMM_VLA_ACCESS(3, input_to_use, (i_ij), (i_ii), (i_ifm2), padded_w, handle->ifmblock), \
                        &LIBXSMM_VLA_ACCESS(7, tr_wt, (w_ofm1), (w_ifm1), (w_kj), (w_ki), (w_ofm2), (w_ifm2), 0, handle->blocksifm * handle->fm_lp_block, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block), \
                        &LIBXSMM_VLA_ACCESS(6, del_out, (o_img), (o_ofm1), (o_oj), (o_oi), (o_ofm2), 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block), \
                        &LIBXSMM_VLA_ACCESS(3, pf_input_to_use, (pi_ij), (pi_ii), (pi_ifm2), padded_w, handle->ifmblock), \
                        NULL, \
                        &LIBXSMM_VLA_ACCESS(6, pf_del_out, po_img, po_ofm1, po_oj, po_oi, po_ofm2, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block) \
                       )

#define LIBXSMM_JITTED_CONV_BP_NO_PF(input_to_use, i_img, i_ifm1, i_ij, i_ii, i_ifm2, \
                                  tr_wt, w_ofm1, w_ifm1, w_kj, w_ki, w_ofm2, w_ifm2, \
                                  del_out, o_img, o_ofm1, o_oj, o_oi, o_ofm2) \
                    jitted_conv_bp_no_pf(  \
                        &LIBXSMM_VLA_ACCESS(3, input_to_use, (i_ij), (i_ii), (i_ifm2), padded_w, handle->ifmblock), \
                        &LIBXSMM_VLA_ACCESS(7, tr_wt, (w_ofm1), (w_ifm1), (w_kj), (w_ki), (w_ofm2), (w_ifm2), 0, handle->blocksifm * handle->fm_lp_block, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block), \
                        &LIBXSMM_VLA_ACCESS(6, del_out, (o_img), (o_ofm1), (o_oj), (o_oi), (o_ofm2), 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block), \
                        NULL, \
                        NULL, \
                        NULL \
                       )

#else

#define LIBXSMM_JITTED_CONV_BP_PF(input_to_use, i_img, i_ifm1, i_ij, i_ii, i_ifm2, \
                                  tr_wt, w_ofm1, w_ifm1, w_kj, w_ki, w_ofm2, w_ifm2, \
                                  del_out, o_img, o_ofm1, o_oj, o_oi, o_ofm2, \
                                  pf_input_to_use, pi_img, pi_ifm1, pi_ij, pi_ii, pi_ifm2, \
                                  pf_tr_wt, pw_ofm1, pw_ifm1, pw_kj, pw_ki, pw_ofm2, pw_ifm2, \
                                  pf_del_out, po_img, po_ofm1, po_oj, po_oi, po_ofm2) \
                    jitted_conv_bp_pf(  \
                        &LIBXSMM_VLA_ACCESS(5, input_to_use, (i_img), (i_ifm1), (i_ij), (i_ii), (i_ifm2), handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock), \
                        &LIBXSMM_VLA_ACCESS(7, tr_wt, (w_ofm1), (w_ifm1), (w_kj), (w_ki), (w_ofm2), (w_ifm2), 0, handle->blocksifm * handle->fm_lp_block, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block), \
                        &LIBXSMM_VLA_ACCESS(6, del_out, (o_img), (o_ofm1), (o_oj), (o_oi), (o_ofm2), 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block), \
                        &LIBXSMM_VLA_ACCESS(5, pf_input_to_use, (pi_img), (pi_ifm1), (pi_ij), (pi_ii), (pi_ifm2), handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock), \
                        &LIBXSMM_VLA_ACCESS(7, pf_tr_wt, (pw_ofm1), (pw_ifm1), (pw_kj), (pw_ki), (pw_ofm2), (pw_ifm2), 0, handle->blocksifm * handle->fm_lp_block, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block), \
                        &LIBXSMM_VLA_ACCESS(6, pf_del_out, po_img, po_ofm1, po_oj, po_oi, po_ofm2, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block) \
                       )

#define LIBXSMM_JITTED_CONV_BP_NOWEIGHT_PF(input_to_use, i_img, i_ifm1, i_ij, i_ii, i_ifm2, \
                                  tr_wt, w_ofm1, w_ifm1, w_kj, w_ki, w_ofm2, w_ifm2, \
                                  del_out, o_img, o_ofm1, o_oj, o_oi, o_ofm2, \
                                  pf_input_to_use, pi_img, pi_ifm1, pi_ij, pi_ii, pi_ifm2, \
                                  pf_del_out, po_img, po_ofm1, po_oj, po_oi, po_ofm2) \
                    jitted_conv_bp_noweight_pf(  \
                        &LIBXSMM_VLA_ACCESS(5, input_to_use, (i_img), (i_ifm1), (i_ij), (i_ii), (i_ifm2), handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock), \
                        &LIBXSMM_VLA_ACCESS(7, tr_wt, (w_ofm1), (w_ifm1), (w_kj), (w_ki), (w_ofm2), (w_ifm2), 0, handle->blocksifm * handle->fm_lp_block, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block), \
                        &LIBXSMM_VLA_ACCESS(6, del_out, (o_img), (o_ofm1), (o_oj), (o_oi), (o_ofm2), 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block), \
                        &LIBXSMM_VLA_ACCESS(5, pf_input_to_use, (pi_img), (pi_ifm1), (pi_ij), (pi_ii), (pi_ifm2), handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock), \
                        NULL, \
                        &LIBXSMM_VLA_ACCESS(6, pf_del_out, po_img, po_ofm1, po_oj, po_oi, po_ofm2, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block) \
                       )

#define LIBXSMM_JITTED_CONV_BP_NO_PF(input_to_use, i_img, i_ifm1, i_ij, i_ii, i_ifm2, \
                                  tr_wt, w_ofm1, w_ifm1, w_kj, w_ki, w_ofm2, w_ifm2, \
                                  del_out, o_img, o_ofm1, o_oj, o_oi, o_ofm2) \
                    jitted_conv_bp_no_pf(  \
                        &LIBXSMM_VLA_ACCESS(5, input_to_use, (i_img), (i_ifm1), (i_ij), (i_ii), (i_ifm2), handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock), \
                        &LIBXSMM_VLA_ACCESS(7, tr_wt, (w_ofm1), (w_ifm1), (w_kj), (w_ki), (w_ofm2), (w_ifm2), 0, handle->blocksifm * handle->fm_lp_block, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock, handle->fm_lp_block), \
                        &LIBXSMM_VLA_ACCESS(6, del_out, (o_img), (o_ofm1), (o_oj), (o_oi), (o_ofm2), 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock, handle->fm_lp_block), \
                        NULL, \
                        NULL, \
                        NULL \
                       )

#endif

if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  ||
     libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ||
     libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM ) {
#if defined(LIBXSMM_CONV_NO_PREFETCH)
  jitted_conv_bp_no_pf = (libxsmm_convfunction)handle->code_bwd[0].xconv.sconv;
#else
  jitted_conv_bp_noweight_pf = (libxsmm_convfunction)handle->code_bwd[1].xconv.sconv;
  jitted_conv_bp_pf = (libxsmm_convfunction)handle->code_bwd[2].xconv.sconv;
#endif
  /* Placing the if statement here to reduce number of branch predictions */
  if (handle->bwd_ofw_rb == handle->ofw) {
    /* Inside oi loop prefetch for next oj */
    for (imgifm1 = thr_begin; imgifm1 < thr_end; ++imgifm1) {
      img = imgifm1/(handle->blocksifm*handle->fm_lp_block);
      ifm1lpblock = imgifm1%(handle->blocksifm*handle->fm_lp_block);
      ifm1 = ifm1lpblock / handle->fm_lp_block;
      lp = ifm1lpblock % handle->fm_lp_block;
      /* First upconvert for low precision */
      if (handle->datatype != handle->datatype_itm) {
        for (ij = 0; ij < handle->ifhp; ++ij) {
          for (ii = 0; ii < handle->ifwp; ++ii) {
            for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
              LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1lpblock, ij, ii, ifm2, handle->blocksifm*handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock) = (element_input_type)(LIBXSMM_VLA_ACCESS(6, del_input_lp, img, ifm1, ij, ii, ((handle->ifmblock/handle->fm_lp_block)*lp)+(ifm2/handle->fm_lp_block), (ifm2%handle->fm_lp_block), handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block));
            }
          }
        }
      }/* end of upconvert for low precision */

      /* Probably input padding copy here */
#if defined(INPUT_PADDING)
      input_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1lpblock, 0, 0, 0, handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock);
      if (ifm1lpblock+1 != handle->blocksifm * handle->fm_lp_block) {
        /* Prefetch next ifm1lpblock, same image */
        prefetch_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1lpblock+1, 0, 0, 0, handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock);
      } else {
        /* Prefetch ifm1lpblock 0 from next image */
        prefetch_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, del_input, img+1, 0, 0, 0, 0, handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock);
      }
      jitted_matcopy(input_ptr, NULL, copy_ptr, NULL, prefetch_ptr);
#endif
      for (ofm1 = 0; ofm1 < handle->blocksofm; ++ofm1) {
        for (oj = 0; oj < handle->ofh; oj+= handle->bwd_ofh_rb) {
          /* define ij */
          ij = oj * handle->desc.u;
          for (oi = 0; oi < handle->ofw; oi += handle->bwd_ofw_rb) {
            /* define ii */
            ii = oi * handle->desc.v;
#if !defined(LIBXSMM_CONV_NO_PREFETCH)
            /* check we are not at the end */
            if (oj < handle->ofh-handle->bwd_ofh_rb) {
              LIBXSMM_JITTED_CONV_BP_NOWEIGHT_PF(input_to_use, img, ifm1lpblock, ij, ii, 0,
                                                tr_wt, ofm1, ifm1lpblock, 0, 0, 0, 0,
                                                del_out, img, ofm1, oj, oi, 0,
                                                input_to_use, img, ifm1lpblock, (oj + handle->bwd_ofh_rb) * handle->desc.u, ii, 0,
                                                del_out, img, ofm1, (oj + handle->bwd_ofh_rb), oi, 0
                                                );
            } else {
              if ((ofm1+1 == handle->blocksofm) &&  ((ifm1lpblock+1) == (handle->blocksifm * handle->fm_lp_block))) {
                LIBXSMM_JITTED_CONV_BP_PF(input_to_use, img, ifm1lpblock, ij, ii, 0,
                                          tr_wt, ofm1, ifm1lpblock, 0, 0, 0, 0,
                                          del_out, img, ofm1, oj, oi, 0,
                                          input_to_use, img+1, 0, 0, 0, 0,
                                          tr_wt, 0, 0, 0, 0, 0, 0,
                                          del_out, img+1, 0, 0, 0, 0
                                         );
              } else {
                if (ofm1+1 == handle->blocksofm) {
                  LIBXSMM_JITTED_CONV_BP_PF(input_to_use, img, ifm1lpblock, ij, ii, 0,
                                            tr_wt, ofm1, ifm1lpblock, 0, 0, 0, 0,
                                            del_out, img, ofm1, oj, oi, 0,
                                            input_to_use, img, ifm1lpblock+1, 0, 0, 0,
                                            tr_wt, 0, ifm1lpblock+1, 0, 0, 0, 0,
                                            del_out, img, 0, 0, 0, 0
                                           );
                } else {
                  LIBXSMM_JITTED_CONV_BP_PF(input_to_use, img, ifm1lpblock, ij, ii, 0,
                                            tr_wt, ofm1, ifm1lpblock, 0, 0, 0, 0,
                                            del_out, img, ofm1, oj, oi, 0,
                                            input_to_use, img, ifm1lpblock, 0, 0, 0,
                                            tr_wt, ofm1+1, ifm1lpblock, 0, 0, 0, 0,
                                            del_out, img, ofm1+1, 0, 0, 0
                                           );
                } /* end of ofm1+1 == handle->blocksofm */
              } /* end of (ofm1+1 == handle->blocksofm) &&  (ifm1+1 == handle->blocksifm) */
            } /* end of oj < handle->ofh-handle->bwd_ofh_rb */
#else
            LIBXSMM_JITTED_CONV_BP_NO_PF(input_to_use, img, ifm1lpblock, ij, ii, 0,
                                         tr_wt, ofm1, ifm1lpblock, 0, 0, 0, 0,
                                         del_out, img, ofm1, oj, oi, 0
                                        );
#endif
          } /* end of oi loop */
        } /* end of oj loop */
      } /* end of ofm1 loop */
      /* Probably input padding copy back here */
#if defined(INPUT_PADDING)
      jitted_matcopyback(copy_ptr, NULL, input_ptr, NULL, NULL);
#else
#include "libxsmm_dnn_zero_rim_st_input_custom.tpl.c"
#endif
      /* down-convert for low precision*/
      if (handle->datatype != handle->datatype_itm) {
        for (ij = 0; ij < handle->ifhp; ++ij) {
          for (ii = 0; ii < handle->ifwp; ++ii) {
            for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
              LIBXSMM_VLA_ACCESS(6, del_input_lp, img, ifm1, ij, ii, ((handle->ifmblock/handle->fm_lp_block)*lp)+(ifm2/handle->fm_lp_block), (ifm2%handle->fm_lp_block), handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block) = (element_output_type)(LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1lpblock, ij, ii, ifm2, handle->blocksifm*handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock));
            }
          }
        }
      }/* end of downconvert for low precision */
    } /* end of imgifm1 loop */
  } else { /* If bwd_ofw_rb != ofw */
    /* Inside oi loop prefetch for next ofw_rb */
    for (imgifm1 = thr_begin; imgifm1 < thr_end; ++imgifm1) {
      img = imgifm1/(handle->blocksifm*handle->fm_lp_block);
      ifm1lpblock = imgifm1%(handle->blocksifm*handle->fm_lp_block);
      ifm1 = ifm1lpblock / handle->fm_lp_block;
      lp = ifm1lpblock % handle->fm_lp_block;
      /* First upconvert for low precision */
      if (handle->datatype != handle->datatype_itm) {
        for (ij = 0; ij < handle->ifhp; ++ij) {
          for (ii = 0; ii < handle->ifwp; ++ii) {
            for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
              LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1lpblock, ij, ii, ifm2, handle->blocksifm*handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock) = (element_input_type)(LIBXSMM_VLA_ACCESS(6, del_input_lp, img, ifm1, ij, ii, ((handle->ifmblock/handle->fm_lp_block)*lp)+(ifm2/handle->fm_lp_block), (ifm2%handle->fm_lp_block), handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block));
            }
          }
        }
      }/* end of upconvert for low precision */

      /* Probably input padding copy here */
#if defined(INPUT_PADDING)
      input_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1lpblock, 0, 0, 0, handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock);
      if (ifm1lpblock+1 != handle->blocksifm * handle->fm_lp_block) {
        /* Prefetch next ifm1lpblock, same image */
        prefetch_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1lpblock+1, 0, 0, 0, handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock);
      } else {
        /* Prefetch ifm1lpblock 0 from next image */
        prefetch_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, del_input, img+1, 0, 0, 0, 0, handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock);
      }
      jitted_matcopy(input_ptr, NULL, copy_ptr, NULL, prefetch_ptr);
#endif
      for (ofm1 = 0; ofm1 < handle->blocksofm; ++ofm1) {
        for (oj = 0; oj < handle->ofh; oj+= handle->bwd_ofh_rb) {
          /* define ij */
          ij = oj * handle->desc.u;
          for (oi = 0; oi < handle->ofw; oi += handle->bwd_ofw_rb) {
            /* define ii */
            ii = oi * handle->desc.v;
#if !defined(LIBXSMM_CONV_NO_PREFETCH)
            /* check we are not at the end */
            if ((oj < handle->ofh-handle->bwd_ofh_rb) && (oi < handle->ofw-handle->bwd_ofw_rb)) {
              LIBXSMM_JITTED_CONV_BP_NOWEIGHT_PF(input_to_use, img, ifm1lpblock, ij, ii, 0,
                                                tr_wt, ofm1, ifm1lpblock, 0, 0, 0, 0,
                                                del_out, img, ofm1, oj, oi, 0,
                                                input_to_use, img, ifm1lpblock, ij, (oi + handle->bwd_ofw_rb) * handle->desc.v, 0,
                                                del_out, img, ofm1, oj, oi + handle->bwd_ofw_rb, 0
                                                );

            } else if (oj < handle->ofh-handle->bwd_ofh_rb) {
              LIBXSMM_JITTED_CONV_BP_NOWEIGHT_PF(input_to_use, img, ifm1lpblock, ij, ii, 0,
                                                tr_wt, ofm1, ifm1lpblock, 0, 0, 0, 0,
                                                del_out, img, ofm1, oj, oi, 0,
                                                input_to_use, img, ifm1lpblock, (oj + handle->bwd_ofh_rb) * handle->desc.u, 0, 0,
                                                del_out, img, ofm1, (oj + handle->bwd_ofh_rb), 0, 0
                                                );
            } else {
              if ((ofm1+1 == handle->blocksofm) &&  ((ifm1lpblock+1) == (handle->blocksifm * handle->fm_lp_block))) {
                LIBXSMM_JITTED_CONV_BP_PF(input_to_use, img, ifm1lpblock, ij, ii, 0,
                                          tr_wt, ofm1, ifm1lpblock, 0, 0, 0, 0,
                                          del_out, img, ofm1, oj, oi, 0,
                                          input_to_use, img+1, 0, 0, 0, 0,
                                          tr_wt, 0, 0, 0, 0, 0, 0,
                                          del_out, img+1, 0, 0, 0, 0
                                         );
              } else {
                if (ofm1+1 == handle->blocksofm) {
                  LIBXSMM_JITTED_CONV_BP_PF(input_to_use, img, ifm1lpblock, ij, ii, 0,
                                            tr_wt, ofm1, ifm1lpblock, 0, 0, 0, 0,
                                            del_out, img, ofm1, oj, oi, 0,
                                            input_to_use, img, ifm1lpblock+1, 0, 0, 0,
                                            tr_wt, 0, ifm1lpblock+1, 0, 0, 0, 0,
                                            del_out, img, 0, 0, 0, 0
                                           );
                } else {
                  LIBXSMM_JITTED_CONV_BP_PF(input_to_use, img, ifm1lpblock, ij, ii, 0,
                                            tr_wt, ofm1, ifm1lpblock, 0, 0, 0, 0,
                                            del_out, img, ofm1, oj, oi, 0,
                                            input_to_use, img, ifm1lpblock, 0, 0, 0,
                                            tr_wt, ofm1+1, ifm1lpblock, 0, 0, 0, 0,
                                            del_out, img, ofm1+1, 0, 0, 0
                                           );
                } /* end of ofm1+1 == handle->blocksofm */
              } /* end of (ofm1+1 == handle->blocksofm) &&  (ifm1+1 == handle->blocksifm) */
            } /* end of oj < handle->ofh-handle->bwd_ofh_rb */
#else
            LIBXSMM_JITTED_CONV_BP_NO_PF(input_to_use, img, ifm1lpblock, ij, ii, 0,
                                         tr_wt, ofm1, ifm1lpblock, 0, 0, 0, 0,
                                         del_out, img, ofm1, oj, oi, 0
                                        );
#endif
          } /* end of oi loop */
        } /* end of oj loop */
      } /* end of ofm1 loop */
      /* Probably input padding copy back here */
#if defined(INPUT_PADDING)
      jitted_matcopyback(copy_ptr, NULL, input_ptr, NULL, NULL);
#else
#include "libxsmm_dnn_zero_rim_st_input_custom.tpl.c"
#endif
      /* down-convert for low precision*/
      if (handle->datatype != handle->datatype_itm) {
        for (ij = 0; ij < handle->ifhp; ++ij) {
          for (ii = 0; ii < handle->ifwp; ++ii) {
            for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
              LIBXSMM_VLA_ACCESS(6, del_input_lp, img, ifm1, ij, ii, ((handle->ifmblock/handle->fm_lp_block)*lp)+(ifm2/handle->fm_lp_block), (ifm2%handle->fm_lp_block), handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock, handle->fm_lp_block) = (element_output_type)(LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1lpblock, ij, ii, ifm2, handle->blocksifm*handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock));
            }
          }
        }
      }/* end of downconvert for low precision */
    } /* end of imgifm1 loop */
  } /* end of if bwd_ofw_rb == ofw */
} else if (libxsmm_target_archid == LIBXSMM_X86_AVX2 ) {
  jitted_conv_bp_no_pf = (libxsmm_convfunction)handle->code_bwd[0].xconv.sconv;
  for (imgifm1 = thr_begin; imgifm1 < thr_end; ++imgifm1) {
    img = imgifm1/(handle->blocksifm*handle->fm_lp_block);
    ifm1lpblock = imgifm1%(handle->blocksifm*handle->fm_lp_block);
    ifm1 = ifm1lpblock / handle->fm_lp_block;
    lp = ifm1lpblock % handle->fm_lp_block;

  /* Probably input padding copy here */
#if defined(INPUT_PADDING)
    input_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1lpblock, 0, 0, 0, handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock);
    if (ifm1lpblock+1 != handle->blocksifm * handle->fm_lp_block) {
      /* Prefetch next ifm1lpblock, same image */
      prefetch_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1lpblock+1, 0, 0, 0, handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock);
    } else {
      /* Prefetch ifm1lpblock 0 from next image */
      prefetch_ptr = (element_input_type*)&LIBXSMM_VLA_ACCESS(5, del_input, img+1, 0, 0, 0, 0, handle->blocksifm * handle->fm_lp_block, handle->ifhp, handle->ifwp, handle->ifmblock);
    }
    jitted_matcopy(input_ptr, NULL, copy_ptr, NULL, prefetch_ptr);
#endif
    for (ofm1 = 0; ofm1 < handle->blocksofm; ++ofm1) {
      for (oj = 0; oj < handle->ofh; oj+= handle->bwd_ofh_rb) {
        /* define ij */
        ij = oj * handle->desc.u;
        for (oi = 0; oi < handle->ofw; oi += handle->bwd_ofw_rb) {
          /* define ii */
          ii = oi * handle->desc.v;
          LIBXSMM_JITTED_CONV_BP_NO_PF(input_to_use, img, ifm1lpblock, ij, ii, 0,
                                       tr_wt, ofm1, ifm1lpblock, 0, 0, 0, 0,
                                       del_out, img, ofm1, oj, oi, 0
                                      );
        } /* end of oi loop */
      } /* end of oj loop */
    } /* end of ofm1 loop */
    /* Probably input padding copy back here */
#if defined(INPUT_PADDING)
    jitted_matcopyback(copy_ptr, NULL, input_ptr, NULL, NULL);
#else
#include "libxsmm_dnn_zero_rim_st_input_custom.tpl.c"
#endif
  } /* end of imgifm1 loop */
/* should never happen, this is just an additional check */
} else {
status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
} /* end of architecture check */

#undef LIBXSMM_JITTED_CONV_BP_PF
#undef LIBXSMM_JITTED_CONV_BP_NOWEIGHT_PF
#undef LIBXSMM_JITTED_CONV_BP_NO_PF
} /* end of new scope for additional variable declarations (C89) */
