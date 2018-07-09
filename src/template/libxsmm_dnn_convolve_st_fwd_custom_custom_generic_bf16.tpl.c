/******************************************************************************
** Copyright (c) 2018, Intel Corporation                                     **
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
/* Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/

int imgofm1, img, ofm1, ifm1, oj, ij, oi, ii, kj, ki, ifm2, ofm2, lp;
/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
const int work = handle->desc.N * handle->blocksofm;
/* compute chunk size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* offset output pointer in case of physical output padding */
element_output_type* out = ((element_output_type*)handle->reg_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock*handle->fm_lp_block;

/* padding via stack allocated buffers */
const int padded_w = handle->desc.W + (2 * handle->desc.pad_w);
const int padded_h = handle->desc.H + (2 * handle->desc.pad_h);
const int size_tls1 = padded_h * padded_w * handle->ifmblock * handle->fm_lp_block;
element_filter_type input_scratch_padding_array[size_tls1];
element_filter_type *const input_scratch_padding = input_scratch_padding_array;
for ( ii = 0; ii < size_tls1; ++ii ) { input_scratch_padding[ii] = (element_input_type)0; }

{
  const int scratch6_size = handle->ofhp * handle->ofwp * handle->ofmblock * handle->fm_lp_block;
  const int scratch7_size = handle->ofmblock * handle->fm_lp_block * handle->ifmblock * handle->fm_lp_block * handle->desc.R * handle->desc.S;
  float tmpin[size_tls1];
  float tmpout[scratch6_size];
  float tmpwt[scratch7_size];
  /* open new scope for additional variable declarations (C89) */
  LIBXSMM_VLA_DECL(3, float, output_hp, tmpout, handle->ofwp, handle->ofmblock*handle->fm_lp_block);
  LIBXSMM_VLA_DECL(3, float, input_hp, tmpin, padded_w, handle->ifmblock*handle->fm_lp_block);
  LIBXSMM_VLA_DECL(4, float, weight_hp, tmpwt, handle->desc.S, handle->ifmblock*handle->fm_lp_block, handle->ofmblock*handle->fm_lp_block);
  LIBXSMM_VLA_DECL(5, element_output_type, output, out, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock*handle->fm_lp_block);
  LIBXSMM_VLA_DECL(5, const element_input_type, input, (element_input_type*)handle->reg_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock*handle->fm_lp_block);
  LIBXSMM_VLA_DECL(7, const element_filter_type, weight, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock*handle->fm_lp_block, handle->fm_lp_block);
  LIBXSMM_VLA_DECL(3, element_filter_type, input_padded, input_scratch_padding, padded_w, handle->ifmblock*handle->fm_lp_block);

  /* perform convolution */
  for (imgofm1 = thr_begin; imgofm1 < thr_end; ++imgofm1) {
    img = imgofm1 / handle->blocksofm;
    ofm1 = imgofm1 % handle->blocksofm;

    /* set output to zero */
    for ( ofm2 = 0 ; ofm2 < scratch6_size; ++ofm2 ) {
      tmpout[ofm2] = 0.0f;
    }

    for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
      /* copy weights from BFP16 into float */
      for (kj = 0; kj < handle->desc.R; ++kj) {
        for (ki = 0; ki< handle->desc.S; ++ki) {
          for( ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2 ) {
            for( ofm2 = 0; ofm2 < handle->ifmblock*handle->fm_lp_block; ++ofm2 ) {
              for ( lp = 0; lp < handle->fm_lp_block; ++lp ) {
                union libxsmm_bfloat16_hp trans;
                trans.i[0] = 0;
                trans.i[1] = LIBXSMM_VLA_ACCESS(7, weight, ofm1, ifm1, kj, ki, ifm2, ofm2, lp, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock*handle->fm_lp_block, handle->fm_lp_block);

                LIBXSMM_VLA_ACCESS(4, weight_hp, kj, ki, (ifm2*handle->fm_lp_block)+lp, ofm2, handle->desc.S, handle->ifmblock*handle->fm_lp_block, handle->ofmblock*handle->fm_lp_block) = trans.f;
              }
            }
          }
        }
      }

      /* check if we need padding, for now we do physical padding on the fly, however we can play with N parameter of the GEMM */
      /* @TODO: add variant which deals with multiple GEMMS by varying N to deal with padding */
      if ( (handle->desc.pad_h == handle->desc.pad_h_in) && (handle->desc.pad_w == handle->desc.pad_w_in) ) {
        /* copy inputs from BFP16 into float */
        libxsmm_convert_bf16_f32( &LIBXSMM_VLA_ACCESS(5, input, img, ifm1, 0, 0, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock*handle->fm_lp_block),
                                  tmpin, handle->ifhp*handle->ifwp*handle->ifmblock*handle->fm_lp_block);

        /* run convolution */
        for (oj = 0; oj < handle->ofh; ++oj) {
          ij = oj * handle->desc.u;
          ii = 0; oi = 0;
          for (kj = 0; kj < handle->desc.R; ++kj) {
            for (ki = 0; ki< handle->desc.S; ++ki) {
              gemm_kernel( &LIBXSMM_VLA_ACCESS(4, weight_hp, kj, ki, 0, 0, handle->desc.S, handle->ifmblock*handle->fm_lp_block, handle->ofmblock*handle->fm_lp_block),
                           &LIBXSMM_VLA_ACCESS(3,  input_hp, ij + kj, ii + ki, 0, padded_w, handle->ifmblock*handle->fm_lp_block),
                           &LIBXSMM_VLA_ACCESS(3, output_hp,  oj, oi, 0, handle->ofwp, handle->ofmblock*handle->fm_lp_block) );
            }
          }
        }
      } else {
        /* copy inputs from BFP16 into float */
        /* copy into stack buffer for physical padding */
        for (ij = 0; ij < handle->desc.H; ++ij) {
          for (ii = 0; ii < handle->desc.W; ++ii) {
            for (ifm2 = 0; ifm2 < handle->ifmblock*handle->fm_lp_block; ++ifm2) {
              LIBXSMM_VLA_ACCESS(3, input_padded, ij + handle->desc.pad_h, ii + handle->desc.pad_w, ifm2, padded_w, handle->ifmblock*handle->fm_lp_block) =
                LIBXSMM_VLA_ACCESS(5,  input, img, ifm1, ij, ii, ifm2, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock*handle->fm_lp_block);
            }
          }
        }
        /* copy inputs from BFP16 into float */
        libxsmm_convert_bf16_f32( input_scratch_padding, tmpin, padded_h*padded_w*handle->ifmblock*handle->fm_lp_block );

        /* run convolution */
        for (oj = 0; oj < handle->ofh; ++oj) {
          ij = oj * handle->desc.u;
          ii = 0; oi = 0;
          for (kj = 0; kj < handle->desc.R; ++kj) {
            for (ki = 0; ki< handle->desc.S; ++ki) {
              gemm_kernel( &LIBXSMM_VLA_ACCESS(4, weight_hp, kj, ki, 0, 0, handle->desc.S, handle->ifmblock*handle->fm_lp_block, handle->ofmblock*handle->fm_lp_block),
                           &LIBXSMM_VLA_ACCESS(3,  input_hp, ij + kj, ii + ki, 0, padded_w, handle->ifmblock*handle->fm_lp_block),
                           &LIBXSMM_VLA_ACCESS(3, output_hp,  oj, oi, 0, handle->ofwp, handle->ofmblock*handle->fm_lp_block) );
            }
          }
        }
      }
    }
    /* copy outputs from FP32 into BFP16 */
    libxsmm_truncate_convert_f32_bf16( tmpout,
      &LIBXSMM_VLA_ACCESS( 5, output, img, ofm1, 0, 0, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock*handle->fm_lp_block),
      scratch6_size);
  }
}

