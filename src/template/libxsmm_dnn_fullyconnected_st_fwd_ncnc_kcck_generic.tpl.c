/******************************************************************************
** Copyright (c) 2017-2019, Intel Corporation                                **
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
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

/* size variables, all const */
/* here we assume that input and output blocking is similar */
const int nBlocksIFm = handle->desc.C / handle->bc;
const int nBlocksOFm = handle->desc.K / handle->bk;
const int nBlocksMB  = handle->desc.N / handle->bn;

/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
const int work = nBlocksOFm * nBlocksMB;
/* compute chunk size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* loop variables */
int mb1ofm1 = 0;
int ifm1 = 0;
int mb2 = 0;
int ofm2 = 0;

LIBXSMM_VLA_DECL(4, element_output_type,       output, (element_output_type*)handle->reg_output->data, nBlocksOFm, handle->bn, handle->bk);
LIBXSMM_VLA_DECL(4, const element_input_type,  input,  (element_input_type* )handle->reg_input->data,  nBlocksIFm, handle->bn, handle->bc);
LIBXSMM_VLA_DECL(4, const element_filter_type, filter, (element_filter_type*)handle->reg_filter->data, nBlocksIFm, handle->bc, handle->bk);

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);
/* Zero out C */
for ( mb1ofm1 = thr_begin; mb1ofm1 < thr_end; ++mb1ofm1 ) {
  int mb1  = mb1ofm1/nBlocksOFm;
  int ofm1 = mb1ofm1%nBlocksOFm;
  
  for ( mb2 = 0; mb2 < handle->bn; ++mb2 ) {
    for ( ofm2 = 0; ofm2 < handle->bk; ++ofm2 ) {
      LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk) = (element_output_type)0;
    }
  }
}
libxsmm_barrier_wait(handle->barrier, ltid);

for ( mb1ofm1 = thr_begin; mb1ofm1 < thr_end; ++mb1ofm1 ) {
  int mb1  = mb1ofm1/nBlocksOFm;
  int ofm1 = mb1ofm1%nBlocksOFm;
 
  for( ifm1 = 0; ifm1 < nBlocksIFm; ++ifm1 ) {
    gemm_kernel( &LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1, 0, 0, nBlocksIFm, handle->bc, handle->bk),
                 &LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1,  0, 0, nBlocksIFm, handle->bn, handle->bc),
                 &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1,  0, 0, nBlocksOFm, handle->bn, handle->bk) );
  }
}

libxsmm_barrier_wait(handle->barrier, ltid);

