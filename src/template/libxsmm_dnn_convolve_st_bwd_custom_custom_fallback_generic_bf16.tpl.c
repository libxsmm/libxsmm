/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/

int imgifm1, img, ofm1, ifm1, oj, ij, oi, ii, kj, ki, ifm2, ofm2;
/* computing first logical thread */
const int ltid = tid - start_thread;
/* auxiliary lp variables  */
int ofmblock_lp = handle->ofmblock/handle->fm_lp_block;
int ifmblock_lp = handle->ifmblock/handle->fm_lp_block;
int lpb = handle->fm_lp_block;
unsigned long long n_blocks = handle->blocksofm;

/* number of tasks that could be run in parallel */
int task;
const int work = handle->desc.N * handle->blocksifm;
/* compute chunk size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* number of tasks for transpose that could be run in parallel */
int transpose_work = handle->blocksifm * handle->blocksofm * handle->desc.R * handle->desc.S;
/* compute chunk size */
const int transpose_chunksize = (transpose_work % handle->desc.threads == 0) ? (transpose_work / handle->desc.threads) : ((transpose_work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
const int transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;

/* offset pointer in case of physical padding */
element_output_type *const out = (element_output_type*)handle->grad_output->data + ((size_t)handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock;

/* Weight and transpose_weight tensor declaration */
LIBXSMM_VLA_DECL(7, element_filter_type, wt, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, lpb);
LIBXSMM_VLA_DECL(7, element_filter_type, tr_wt, (element_filter_type*)((char*)handle->scratch + handle->bwd_filter_trans_scratch_offset), handle->blocksofm, handle->desc.R, handle->desc.S, ofmblock_lp, handle->ifmblock, lpb);

/* define weight pointer which has the correct format */
element_filter_type* weight_base = 0;

/* padding via stack allocated buffers */
const int padded_w = handle->desc.W + (2 * handle->desc.pad_w);
const int padded_h = handle->desc.H + (2 * handle->desc.pad_h);
const int size_tls1 = padded_h * padded_w * handle->ifmblock;
float *const del_input_scratch_padding = (float*)((char*)handle->scratch + handle->bwd_packing_padding_scratch_offset) + ltid * size_tls1;
for ( ii = 0; ii < size_tls1; ++ii ) { del_input_scratch_padding[ii] = (float)0.0; }

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

/* transpose filters, if requested */
if ( (handle->options & LIBXSMM_DNN_CONV_OPTION_BWD_NO_FILTER_TRANSPOSE) == 0 ) {
  for (task = transpose_thr_begin; task < transpose_thr_end; ++task) {
    ifm1 = task/(handle->blocksofm * handle->desc.R * handle->desc.S);
    ofm1 = (task%(handle->blocksofm * handle->desc.R * handle->desc.S))/(handle->desc.R * handle->desc.S);
    kj =   ((task%(handle->blocksofm * handle->desc.R * handle->desc.S))%(handle->desc.R * handle->desc.S))/handle->desc.S;
    ki =   ((task%(handle->blocksofm * handle->desc.R * handle->desc.S))%(handle->desc.R * handle->desc.S))%handle->desc.S;
    for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
      for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
        LIBXSMM_VLA_ACCESS(7, tr_wt, ifm1, ofm1, handle->desc.R-1-kj , handle->desc.S-1-ki, ofm2/lpb, ifm2, ofm2%lpb, handle->blocksofm, handle->desc.R, handle->desc.S, ofmblock_lp, handle->ifmblock, lpb) =
          LIBXSMM_VLA_ACCESS(7, wt, ofm1, ifm1, kj, ki, ifm2/lpb, ofm2, ifm2%lpb, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, lpb);
      }
    }
  }
  weight_base = (element_filter_type*)((char*)handle->scratch + handle->bwd_filter_trans_scratch_offset);

  /* wait for transpose to finish */
  libxsmm_barrier_wait(handle->barrier, ltid);
} else {
  weight_base = (element_filter_type*)handle->reg_filter_tr->data;
}

{/* open new scope for additional variable declarations (C89) */
LIBXSMM_VLA_DECL(5, element_input_type, del_input, (element_output_type*)handle->grad_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
LIBXSMM_VLA_DECL(3, float, del_input_padded, del_input_scratch_padding, padded_w, handle->ifmblock);
LIBXSMM_VLA_DECL(5, element_output_type, output, out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
LIBXSMM_VLA_DECL(7, element_filter_type, weight, weight_base, handle->blocksofm, handle->desc.R, handle->desc.S, ofmblock_lp, handle->ifmblock, lpb);
/* Auxiliary fp32 accumulators */
float *del_inp_fp32 = (float*)((char*)handle->scratch + handle->bwd_lp_input_full_scratch_offset);
LIBXSMM_VLA_DECL(5, float, del_input_fp32, del_inp_fp32, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);

for (imgifm1 = thr_begin; imgifm1 < thr_end; ++imgifm1) {
  img = imgifm1 / handle->blocksifm;
  ifm1 = imgifm1 % handle->blocksifm;

  /* check if we need padding, for now we do physical padding on the fly, however we can play with N parameter of the GEMM */
  /* @TODO: add variant which deals with multiple GEMMS by varying N to deal with padding */
  if ( (handle->desc.pad_h == handle->desc.pad_h_in) && (handle->desc.pad_w == handle->desc.pad_w_in) ) {

    /* reset result buffer to zero when intent is to overwrite when first block
       of input channels should be convoluted */
    if ( ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) ) {
      float* temp_ptr = &(LIBXSMM_VLA_ACCESS(  5, del_input_fp32, img, ifm1, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock));
      LIBXSMM_PRAGMA_SIMD
      for (ij = 0; ij < handle->ifhp*handle->ifwp*handle->ifmblock; ij++) {
        temp_ptr[ij] = (float)0.0;
      }
    }

    /* run convolution */
    for ( oj = 0; oj < handle->ofh; ++oj) {
      ij = oj * handle->desc.u;
      oi = 0; ii = 0;
      for (kj = 0; kj < handle->desc.R; ++kj) {
        for (ki = 0; ki < handle->desc.S; ++ki) {
          bf16fp32_brgemm_kernel( &LIBXSMM_VLA_ACCESS(7, weight, ifm1, 0, handle->desc.R-1-kj, handle->desc.S-1-ki, 0, 0, 0,        handle->blocksofm, handle->desc.R, handle->desc.S, ofmblock_lp, handle->ifmblock, lpb),
                       &LIBXSMM_VLA_ACCESS(5, output,  img, 0, oj, oi, 0,           handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                       &LIBXSMM_VLA_ACCESS(5, del_input_fp32,  img, ifm1, ij + kj, ii + ki, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock), &n_blocks );
        }
      }
    }

    /* Downconvert computed result to bf16 */
    LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16( &LIBXSMM_VLA_ACCESS(5, del_input_fp32, img, ifm1, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                                         &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                                         handle->ifhp * handle->ifwp * handle->ifmblock);

    /* zero rim in case of physical padding.... this code is extremely stupid and crappy as it requires a complicated if... */
    if (handle->desc.pad_h_in > 0 || handle->desc.pad_w_in > 0) {
      for ( ij = 0; ij < handle->ifhp; ij++ ) {
        for ( ii = 0; ii < handle->ifwp; ii++ ) {
          if ( (ij < handle->desc.pad_h_in) || (ij >= (handle->desc.H+handle->desc.pad_h_in)) ||
               (ii < handle->desc.pad_w_in) || (ii >= (handle->desc.W+handle->desc.pad_w_in)) ) {
            for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
              LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, ii, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock) = (element_input_type)0;
            }
          }
        }
      }
    }

  } else {
    /* reset result buffer to zero when intent is to overwrite when first block
       of input channels should be convoluted */
    LIBXSMM_PRAGMA_SIMD
    for (ij = 0; ij < size_tls1; ++ij) {
      del_input_scratch_padding[ij] = (float)0.0;
    }


    /* run convolution */
    for ( oj = 0; oj < handle->ofh; ++oj) {
      ij = oj * handle->desc.u;
      oi = 0; ii = 0;
      for (kj = 0; kj < handle->desc.R; ++kj) {
        for (ki = 0; ki < handle->desc.S; ++ki) {
          bf16fp32_brgemm_kernel( &LIBXSMM_VLA_ACCESS(7, weight, ifm1, 0, handle->desc.R-1-kj, handle->desc.S-1-ki, 0, 0, 0,       handle->blocksofm, handle->desc.R, handle->desc.S, ofmblock_lp, handle->ifmblock, lpb),
                       &LIBXSMM_VLA_ACCESS(5, output,  img, 0, oj, oi, 0,           handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock),
                       &LIBXSMM_VLA_ACCESS(3, del_input_padded, ij + kj, ii + ki, 0, padded_w, handle->ifmblock), &n_blocks );
        }
      }
    }

    /* input padding copy back */
    for (ij = 0; ij < handle->desc.H; ij++) {
      LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16(&LIBXSMM_VLA_ACCESS(3, del_input_padded, ij + handle->desc.pad_h, handle->desc.pad_w, 0, padded_w, handle->ifmblock),
                                          &LIBXSMM_VLA_ACCESS(5, del_input, img, ifm1, ij, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                                          handle->desc.W * handle->ifmblock);
    }
  }
} /* end of imgifm1 loop */

} /* end of new scope for additional variable declarations (C89) */

libxsmm_barrier_wait(handle->barrier, ltid);
