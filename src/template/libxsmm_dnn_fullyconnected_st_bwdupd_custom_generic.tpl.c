/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Sasikanth Avancha (Intel Corp.)
******************************************************************************/

if ( (kind == LIBXSMM_DNN_COMPUTE_KIND_BWD) || (kind == LIBXSMM_DNN_COMPUTE_KIND_BWDUPD) ) {
  /* size variables, all const */
  /* here we assume that input and output blocking is similar */
  const int nBlocksIFm = handle->blocksifm;
  const int nIFmBlock = handle->ifmblock;
  const int nBlocksOFm = handle->blocksofm;
  const int nOFmBlock = handle->ofmblock;

  /* computing first logical thread */
  const int ltid = tid - start_thread;
  /* number of tasks that could be run in parallel */
  const int work = nBlocksIFm;
  /* compute chunk size */
  const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
  /* compute thr_begin and thr_end */
  const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
  const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

  /* number of tasks for transpose that could be run in parallel */
  const int transpose_work = nBlocksIFm * nBlocksOFm;
  /* compute chunk size */
  const int transpose_chunksize = (transpose_work % handle->desc.threads == 0) ? (transpose_work / handle->desc.threads) : ((transpose_work / handle->desc.threads) + 1);
  /* compute thr_begin and thr_end */
  const int transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
  const int transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;

  /* loop variables */
  int ofm1 = 0;
  int ofm2 = 0;
  int ifm1 = 0;
  int ifm2 = 0;
  int ifm1ofm1 = 0;

  LIBXSMM_VLA_DECL(3, const element_output_type,   doutput, (element_output_type*)handle->grad_output->data, nBlocksOFm, nOFmBlock);
  LIBXSMM_VLA_DECL(4, const element_filter_type,    filter, (element_filter_type*)handle->reg_filter->data,  nBlocksIFm, nIFmBlock, nOFmBlock);
#if defined(LIBXSMM_DNN_FULLYCONNECTED_BWD_BF16_F32)
  float* dinput_f32_ptr = (float*)handle->scratch;
  float* filter_f32_ptr = ((float*)handle->scratch)+((size_t)handle->desc.N*(size_t)handle->desc.C);
  LIBXSMM_VLA_DECL(3,       float,    dinput, dinput_f32_ptr,  nBlocksIFm, nIFmBlock);
  LIBXSMM_VLA_DECL(4,       float, filter_tr, filter_f32_ptr, nBlocksOFm, nOFmBlock, nIFmBlock);

  /* number of tasks that could be run in parallel */
  const int work_input = handle->desc.N * handle->desc.C;
  /* compute chunk size */
  const int chunksize_input = (work_input % handle->desc.threads == 0) ? (work_input / handle->desc.threads) : ((work_input / handle->desc.threads) + 1);
  /* compute thr_begin and thr_end */
  const int thr_begin_input = (ltid * chunksize_input < work_input) ? (ltid * chunksize_input) : work_input;
  const int thr_end_input = ((ltid + 1) * chunksize_input < work_input) ? ((ltid + 1) * chunksize_input) : work_input;
#else
  LIBXSMM_VLA_DECL(3,        element_input_type,    dinput, (element_input_type* )handle->grad_input->data,  nBlocksIFm, nIFmBlock);
  LIBXSMM_VLA_DECL(4,       element_filter_type, filter_tr, (element_filter_type*)handle->scratch,           nBlocksOFm, nOFmBlock, nIFmBlock);
#endif
  libxsmm_meltw_unary_param trans_param;

  /* lazy barrier init */
  libxsmm_barrier_init(handle->barrier, ltid);


  for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
    ofm1 = ifm1ofm1 / nBlocksIFm;
    ifm1 = ifm1ofm1 % nBlocksIFm;

#if defined(LIBXSMM_DNN_FULLYCONNECTED_BWD_BF16_F32)
    for (ofm2 = 0; ofm2 < nOFmBlock; ++ofm2) {
      for (ifm2 = 0; ifm2 < nIFmBlock; ++ifm2) {
        union libxsmm_bfloat16_hp filter_f32;
        filter_f32.i[0] = 0;
        filter_f32.i[1] = LIBXSMM_VLA_ACCESS(4, filter,  ofm1, ifm1, ifm2, ofm2, nBlocksIFm, nIFmBlock, nOFmBlock);
        LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1, ofm2, ifm2, nBlocksOFm, nOFmBlock, nIFmBlock) = filter_f32.f;
      }
    }
#else
    trans_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(4, filter,  ofm1, ifm1, 0, 0, nBlocksIFm, nIFmBlock, nOFmBlock);
    trans_param.out.primary = & LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1, 0, 0, nBlocksOFm, nOFmBlock, nIFmBlock);
    handle->tr_kernel( &trans_param );
#endif
  }

  /* wait for transpose to finish */
  libxsmm_barrier_wait(handle->barrier, ltid);

  for ( ifm1 = thr_begin; ifm1 < thr_end; ++ifm1 ) {  /* outer GEMM m-loop */
#if 1
    gemm_kernel_bwd( &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, 0, 0, 0, nBlocksOFm, nOFmBlock, nIFmBlock),
                     &LIBXSMM_VLA_ACCESS(3, doutput,   0, 0, 0, nBlocksOFm, nOFmBlock),
                     &LIBXSMM_VLA_ACCESS(3, dinput,    0, ifm1, 0, nBlocksIFm, nIFmBlock) );
#else
    const int nImg = handle->desc.N;
    int img2;

    /* this is a simple replacement code using regular loops */
    for ( img2 = 0; img2 < nImg; ++img2 ) {
      LIBXSMM_PRAGMA_SIMD
      for ( ifm2 = 0; ifm2 < nIFmBlock; ++ifm2 ) {
        LIBXSMM_VLA_ACCESS(3, dinput, img2, ifm1, ifm2, nBlocksIFm, nIFmBlock) = (element_output_type)0;
      }
    }
    for ( ofm1 = 0; ofm1 < nBlocksOFm; ++ofm1 ) {     /* outer GEMM k-loop */
      for ( ofm2 = 0; ofm2 < nOFmBlock; ++ofm2 ) {    /* GEMM K-loop */
        for ( img2 = 0; img2 < nImg; ++img2 ) {       /* GEMM n-loop */
          LIBXSMM_PRAGMA_SIMD
          for ( ifm2 = 0; ifm2 < nIFmBlock; ++ifm2 ) { /* GEMM m-loop */
            LIBXSMM_VLA_ACCESS(3, dinput, img2, ifm1, ifm2, nBlocksIFm, nIFmBlock) +=
              LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1, ofm2, ifm2, nBlocksOFm, nOFmBlock, nIFmBlock) * LIBXSMM_VLA_ACCESS(3, doutput, img2, ofm1, ofm2, nBlocksOFm, nOFmBlock);
          }
        }
      }
    }
#endif
  }

#if defined(LIBXSMM_DNN_FULLYCONNECTED_BWD_BF16_F32)
  libxsmm_barrier_wait(handle->barrier, ltid);

  libxsmm_rne_convert_fp32_bf16( dinput_f32_ptr+thr_begin_input, ((element_input_type*)handle->grad_input->data)+thr_begin_input, thr_end_input-thr_begin_input );
#endif

  libxsmm_barrier_wait(handle->barrier, ltid);
}

if ( (kind == LIBXSMM_DNN_COMPUTE_KIND_UPD) || (kind == LIBXSMM_DNN_COMPUTE_KIND_BWDUPD) ) {
  /* size variables, all const */
  const int nImg = handle->desc.N;
  /* here we assume that input and output blocking is similar */
  const int nBlocksIFm = handle->blocksifm;
  const int nIFmBlock = handle->ifmblock;
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
#if defined(LIBXSMM_DNN_FULLYCONNECTED_UPD_BF16_F32)
  float* input_f32_ptr = (float*)handle->scratch;
  float* dfilter_f32_ptr = ((float*)handle->scratch)+((size_t)handle->desc.N*(size_t)handle->desc.C);
  LIBXSMM_VLA_DECL(3, float, input_tr, input_f32_ptr, nIFmBlock, nImg);
  LIBXSMM_VLA_DECL(4, float,  dfilter, dfilter_f32_ptr, nBlocksIFm, nIFmBlock, nOFmBlock);

  /* number of tasks that could be run in parallel */
  const int work_filter = handle->desc.C * handle->desc.K;
  /* compute chunk size */
  const int chunksize_filter = (work_filter % handle->desc.threads == 0) ? (work_filter / handle->desc.threads) : ((work_filter / handle->desc.threads) + 1);
  /* compute thr_begin and thr_end */
  const int thr_begin_filter = (ltid * chunksize_filter < work_filter) ? (ltid * chunksize_filter) : work_filter;
  const int thr_end_filter = ((ltid + 1) * chunksize_filter < work_filter) ? ((ltid + 1) * chunksize_filter) : work_filter;
#else
  LIBXSMM_VLA_DECL(4,       element_filter_type, dfilter,  (element_filter_type*)handle->grad_filter->data, nBlocksIFm, nIFmBlock, nOFmBlock);
  LIBXSMM_VLA_DECL(3,       element_input_type,  input_tr, (element_input_type* )handle->scratch,           nIFmBlock,  nImg);
#endif

  /* lazy barrier init */
  libxsmm_barrier_init(handle->barrier, ltid);

  for (ifm1 = transpose_thr_begin; ifm1 < transpose_thr_end; ++ifm1) {
    for (ifm2 = 0; ifm2 < nIFmBlock; ++ifm2) {
      for (img2 = 0; img2 < nImg; ++img2) {
#if defined(LIBXSMM_DNN_FULLYCONNECTED_UPD_BF16_F32)
        union libxsmm_bfloat16_hp input_f32;
        input_f32.i[0] = 0;
        input_f32.i[1] = LIBXSMM_VLA_ACCESS(3, input, img2, ifm1, ifm2, nBlocksIFm, nIFmBlock);
        LIBXSMM_VLA_ACCESS(3, input_tr, ifm1, ifm2, img2, nIFmBlock, nImg) = input_f32.f;
#else
        LIBXSMM_VLA_ACCESS(3, input_tr, ifm1, ifm2, img2, nIFmBlock, nImg) =
          LIBXSMM_VLA_ACCESS(3, input, img2, ifm1, ifm2, nBlocksIFm, nIFmBlock);
#endif
      }
    }
  }

  /* wait for transpose to finish */
  libxsmm_barrier_wait(handle->barrier, ltid);

  for ( ifm1ofm1 = thr_begin; ifm1ofm1 < thr_end; ++ifm1ofm1 ) {  /* outer GEMM m/n-loop */
    ofm1 = ifm1ofm1 / nBlocksIFm;
    ifm1 = ifm1ofm1 % nBlocksIFm;

#if 1
    gemm_kernel_upd( &LIBXSMM_VLA_ACCESS(3, doutput,  0, ofm1, 0, nBlocksOFm, nOFmBlock),
                     &LIBXSMM_VLA_ACCESS(3, input_tr, ifm1, 0, 0, nIFmBlock, nImg),
                     &LIBXSMM_VLA_ACCESS(4, dfilter,  ofm1, ifm1, 0, 0, nBlocksIFm, nIFmBlock, nOFmBlock) );
#else
    {
      const int nImg = handle->desc.N;
      int ifm2, ofm2;

      /* this is a simple replacement code using regular loops */
      for ( ifm2 = 0; ifm2 < nIFmBlock; ++ifm2 ) {
        LIBXSMM_PRAGMA_SIMD
        for ( ofm2 = 0; ofm2 < nOFmBlock; ++ofm2 ) {
          LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, ifm2, ofm2, nBlocksIFm, nIFmBlock, nOFmBlock) = (element_output_type)0;
        }
      }
      for ( img2 = 0; img2 < nImg; ++img2 ) {            /* GEMM k-loop */
        for ( ifm2 = 0; ifm2 < nIFmBlock; ++ifm2 ) {     /* GEMM n-loop */
          LIBXSMM_PRAGMA_SIMD
          for ( ofm2 = 0; ofm2 < nOFmBlock; ++ofm2 ) { /* GEMM m-loop */
            LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, ifm2, ofm2, nBlocksIFm, nIFmBlock, nOFmBlock) +=
              LIBXSMM_VLA_ACCESS(3, doutput, img2, ofm1, ofm2, nBlocksOFm, nOFmBlock) * LIBXSMM_VLA_ACCESS(3, input_tr, ifm1, ifm2, img2, nIFmBlock, nImg);
          }
        }
      }
    }
#endif
  }

#if defined(LIBXSMM_DNN_FULLYCONNECTED_UPD_BF16_F32)
  libxsmm_barrier_wait(handle->barrier, ltid);

  libxsmm_rne_convert_fp32_bf16( dfilter_f32_ptr+thr_begin_filter, ((element_input_type*)handle->grad_filter->data)+thr_begin_filter, thr_end_filter-thr_begin_filter );
#endif

  libxsmm_barrier_wait(handle->barrier, ltid);
}

