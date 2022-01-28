/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Sasikanth Avancha (Intel Corp.)
******************************************************************************/

/* size variables, all const */
/* here we assume that input and output blocking is similar */
const int nBlocksIFm = handle->blocksifm;
const int nIFmBlock = handle->ifmblock;
const int nBlocksOFm = handle->blocksofm;
const int nOFmBlock = handle->ofmblock;

/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
const int work = nBlocksOFm;
/* compute chunk size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* loop variables */
int ofm1 = 0;

LIBXSMM_VLA_DECL(3, element_output_type,       output, (element_output_type*)handle->reg_output->data, nBlocksOFm, nOFmBlock);
#if defined(LIBXSMM_DNN_FULLYCONNECTED_FWD_BF16_F32)
float* input_f32_ptr = (float*)handle->scratch;
float* filter_f32_ptr = ((float*)handle->scratch)+((size_t)handle->desc.N*(size_t)handle->desc.C);
LIBXSMM_VLA_DECL(3, const float,  input, input_f32_ptr,  nBlocksIFm, nIFmBlock);
LIBXSMM_VLA_DECL(4, const float, filter, filter_f32_ptr, nBlocksIFm, nIFmBlock, nOFmBlock);

/* number of tasks that could be run in parallel */
const int work_input = handle->desc.N * handle->desc.C;
/* compute chunk size */
const int chunksize_input = (work_input % handle->desc.threads == 0) ? (work_input / handle->desc.threads) : ((work_input / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin_input = (ltid * chunksize_input < work_input) ? (ltid * chunksize_input) : work_input;
const int thr_end_input = ((ltid + 1) * chunksize_input < work_input) ? ((ltid + 1) * chunksize_input) : work_input;

/* number of tasks that could be run in parallel */
const int work_filter = handle->desc.C * handle->desc.K;
/* compute chunk size */
const int chunksize_filter = (work_filter % handle->desc.threads == 0) ? (work_filter / handle->desc.threads) : ((work_filter / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin_filter = (ltid * chunksize_filter < work_filter) ? (ltid * chunksize_filter) : work_filter;
const int thr_end_filter = ((ltid + 1) * chunksize_filter < work_filter) ? ((ltid + 1) * chunksize_filter) : work_filter;
#else
LIBXSMM_VLA_DECL(3, const element_input_type,  input,  (element_input_type* )handle->reg_input->data,  nBlocksIFm, nIFmBlock);
LIBXSMM_VLA_DECL(4, const element_filter_type, filter, (element_filter_type*)handle->reg_filter->data, nBlocksIFm, nIFmBlock, nOFmBlock);
#endif

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

#if defined(LIBXSMM_DNN_FULLYCONNECTED_FWD_BF16_F32)
libxsmm_convert_bf16_f32( ((element_input_type*)handle->reg_input->data)+thr_begin_input,   input_f32_ptr+thr_begin_input,   thr_end_input - thr_begin_input );
libxsmm_convert_bf16_f32( ((element_filter_type*)handle->reg_filter->data)+thr_begin_filter, filter_f32_ptr+thr_begin_filter, thr_end_filter - thr_begin_filter );

libxsmm_barrier_wait(handle->barrier, ltid);
#endif

for ( ofm1 = thr_begin; ofm1 < thr_end; ++ofm1 ) {  /* outer GEMM m-loop */
#if 1
  gemm_kernel( &LIBXSMM_VLA_ACCESS(4, filter, ofm1, 0, 0, 0, nBlocksIFm, nIFmBlock, nOFmBlock),
               &LIBXSMM_VLA_ACCESS(3, input,  0, 0, 0, nBlocksIFm, nIFmBlock),
               &LIBXSMM_VLA_ACCESS(3, output, 0, ofm1, 0, nBlocksOFm, nOFmBlock) );
#else
  {
    const int nImg = handle->desc.N;
    int img2, ifm1, ifm2, ofm2;

    /* this is a simple replacement code using regular loops */
    for ( img2 = 0; img2 < nImg; ++img2 ) {
      LIBXSMM_PRAGMA_SIMD
      for ( ofm2 = 0; ofm2 < nOFmBlock; ++ofm2 ) {
        LIBXSMM_VLA_ACCESS(3, output, img2, ofm1, ofm2, nBlocksOFm, nOFmBlock) = (element_output_type)0;
      }
    }
    for ( ifm1 = 0; ifm1 < nBlocksIFm; ++ifm1 ) {     /* outer GEMM k-loop */
      for ( ifm2 = 0; ifm2 < nIFmBlock; ++ifm2 ) {    /* GEMM K-loop */
        for ( img2 = 0; img2 < nImg; ++img2 ) {       /* GEMM n-loop */
          LIBXSMM_PRAGMA_SIMD
          for ( ofm2 = 0; ofm2 < nOFmBlock; ++ofm2 ) { /* GEMM m-loop */
            LIBXSMM_VLA_ACCESS(3, output, img2, ofm1, ofm2, nBlocksOFm, nOFmBlock) +=
              LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1, ifm2, ofm2, nBlocksIFm, nIFmBlock, nOFmBlock) * LIBXSMM_VLA_ACCESS(3, input, img2, ifm1, ifm2, nBlocksIFm, nIFmBlock);
          }
        }
      }
    }
  }
#endif
}

libxsmm_barrier_wait(handle->barrier, ltid);

