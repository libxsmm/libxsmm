/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
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
/* Alexander Heinecke, Sasikanth Avancha (Intel Corp.)
******************************************************************************/

/* size variables, all const */
const int nImg = handle->desc.N;
/* here we assume that input and output blocking is similar */
const int nBlocksIFm = handle->blocksifm;
const int nIFmBlock = handle->fm_lp_block*handle->ifmblock;
const int nBlocksOFm = handle->blocksofm;
const int nOFmBlock = handle->ofmblock;

/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
const int work = nBlocksIFm * nBlocksOFm;
/* compute chunk size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* number of tasks for transpose that could be run in parallel */
const int transpose_work = nBlocksIFm;
/* compute chunk size */
const int transpose_chunksize = (transpose_work % handle->desc.threads == 0) ? (transpose_work / handle->desc.threads) : ((transpose_work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
const int transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;

/* loop variables */
int img2 = 0;
int ifm1ofm1 = 0;
int ofm1 = 0;
int ifm1 = 0;
int ifm2 = 0;

LIBXSMM_VLA_DECL(3, const element_input_type,  input,    (element_input_type* )handle->reg_input->data,   nBlocksIFm, nIFmBlock);
LIBXSMM_VLA_DECL(3, const element_output_type, doutput,  (element_output_type*)handle->grad_output->data, nBlocksOFm, nOFmBlock);
LIBXSMM_VLA_DECL(4,       element_filter_type, dfilter,  (element_filter_type*)handle->grad_filter->data, nBlocksIFm, nIFmBlock, nOFmBlock);
LIBXSMM_VLA_DECL(3,       element_input_type,  input_tr, (element_input_type* )handle->scratch,           nIFmBlock,  nImg);

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

for (ifm1 = transpose_thr_begin; ifm1 < transpose_thr_end; ++ifm1) {
  for (ifm2 = 0; ifm2 < nIFmBlock; ++ifm2) {
    for (img2 = 0; img2 < nImg; ++img2) {
      LIBXSMM_VLA_ACCESS(3, input_tr, ifm1, ifm2, img2, nIFmBlock, nImg) =
        LIBXSMM_VLA_ACCESS(3, input, img2, ifm1, ifm2, nBlocksIFm, nIFmBlock);
    }
  }
}

/* wait for transpose to finish */
libxsmm_barrier_wait(handle->barrier, ltid);

for ( ifm1ofm1 = thr_begin; ifm1ofm1 < thr_end; ++ifm1ofm1 ) {  /* outer GEMM m/n-loop */
  ofm1 = ifm1ofm1 / nBlocksIFm;
  ifm1 = ifm1ofm1 % nBlocksIFm;

  gemm_kernel( &LIBXSMM_VLA_ACCESS(3, doutput,  0, ofm1, 0, nBlocksOFm, nOFmBlock),
               &LIBXSMM_VLA_ACCESS(3, input_tr, ifm1, 0, 0, nIFmBlock, nImg),
               &LIBXSMM_VLA_ACCESS(4, dfilter,  ofm1, ifm1, 0, 0, nBlocksIFm, nIFmBlock, nOFmBlock) );
}

libxsmm_barrier_wait(handle->barrier, ltid);

