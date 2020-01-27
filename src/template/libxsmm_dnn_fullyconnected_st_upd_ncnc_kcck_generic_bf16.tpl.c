/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#define _mm512_cvt2_fp32_bf16(A, B) LIBXSMM_INTRINSICS_MM512_CVT2_FP32_BF16(A, B)
#if defined(LIBXSMM_DNN_FC_UPD_AVX512_CPX)
#define LIBXSMM_DNN_FC_UPD_CONVERT_F32_BF16(in, out, length) do { \
  unsigned int full_chunks = length / 32; \
  unsigned int remainder = length % 32; \
  int __i = 0; \
  if (remainder == 0) { \
    for ( __i = 0; __i < length; __i+= 32) { \
      _mm512_storeu_si512((libxsmm_bfloat16*)out+__i, (__m512i) _mm512_cvtne2ps_pbh(LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)in+__i+16), LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)in+__i))); \
    } \
  } else { \
    unsigned int chunk; \
    for ( chunk = 0; chunk < full_chunks; chunk++) { \
      __i = chunk * 32; \
      _mm512_storeu_si512((libxsmm_bfloat16*)out+__i, (__m512i) _mm512_cvtne2ps_pbh(LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)in+__i+16), LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)in+__i))); \
    } \
    libxsmm_rne_convert_fp32_bf16((float*)in+32*full_chunks, (element_output_type*)out+32*full_chunks, remainder); \
  } \
} while(0)
#else
#define LIBXSMM_DNN_FC_UPD_CONVERT_F32_BF16(in, out, length) do { \
  unsigned int full_chunks = length / 16; \
  unsigned int remainder = length % 16; \
  int __i = 0; \
  if (remainder == 0) { \
    for ( __i = 0; __i < length; __i+= 16) { \
      _mm256_storeu_si256((__m256i*)(out+__i), _mm512_cvtepi32_epi16( _mm512_srai_epi32( LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16( LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)in+__i) ),16)) ); \
    } \
  } else { \
    unsigned int chunk; \
    for ( chunk = 0; chunk < full_chunks; chunk++) { \
      __i = chunk * 16; \
      _mm256_storeu_si256((__m256i*)(out+__i), _mm512_cvtepi32_epi16( _mm512_srai_epi32( LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16( LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)in+__i) ),16)) ); \
    } \
    libxsmm_rne_convert_fp32_bf16((float*)in+16*full_chunks, (element_output_type*)out+16*full_chunks, remainder); \
  } \
} while(0)
#endif

/* size variables, all const */
const int bn = handle->bn;
const int bk = handle->bk;
const int bc = handle->bc;
const int nBlocksIFm = handle->desc.C / bc;
const int nBlocksOFm = handle->desc.K / bk;
const int nBlocksMB  = handle->desc.N / bn;
/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
const int ofm_subtasks = handle->ofm_subtasks;
const int ifm_subtasks = handle->ifm_subtasks;
const int bbk = bk/ofm_subtasks;
const int bbc = bc/ifm_subtasks;
const int work = nBlocksIFm * ifm_subtasks * nBlocksOFm * ofm_subtasks;
const int Cck_work = nBlocksIFm * ifm_subtasks * ofm_subtasks;
const int Cc_work = nBlocksIFm * ifm_subtasks;

/* 2D blocking parameters  */
int use_2d_blocking = handle->upd_2d_blocking;
int im_tasks_per_thread = 0, in_tasks_per_thread = 0, my_in_start = 0, my_in_end = 0, my_im_start = 0, my_im_end = 0, my_row_id = 0, my_col_id = 0, row_teams = 0, column_teams = 0;

/* compute chunk size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
int BF = handle->upd_bf;

/* loop variables */
int mb1 = 0, ifm1ofm1 = 0, ofm1 = 0, ifm1 = 0, ofm2 = 0, ifm2 = 0, bfn = 0, ii = 0, jj = 0, mb1ofm1 = 0, mb1ifm1 = 0, mb2 = 0, jc = 0, jk = 0;

/* Batch reduce related variables */
unsigned long long  blocks = nBlocksMB/BF;

LIBXSMM_VLA_DECL(4, const element_input_type,  input,    (element_input_type* )handle->reg_input->data, nBlocksIFm, bn, bc);
LIBXSMM_VLA_DECL(4, const element_output_type, doutput,  (element_output_type*)handle->grad_output->data, nBlocksOFm, bn, bk);
LIBXSMM_VLA_DECL(5,       element_filter_type, dfilter,  (element_filter_type*)handle->grad_filter->data, nBlocksIFm, bc/2, bk, 2);

/* Set up tensors for transposing/scratch before vnni reformatting dfilter */
element_output_type *tr_out_ptr = (element_output_type*)handle->scratch;
element_input_type  *tr_inp_ptr = (element_input_type*) ((element_output_type*)tr_out_ptr + handle->desc.N * handle->desc.K);
float               *dfilter_f32_ptr = (float*) ((element_input_type*)tr_inp_ptr + handle->desc.N * handle->desc.C);
element_filter_type *dfilter_scratch = (element_filter_type*) ((float*)dfilter_f32_ptr + handle->desc.C * handle->desc.K) + ltid * bc * bk;

LIBXSMM_VLA_DECL(4, element_input_type,  input_tr,    (element_input_type*)tr_inp_ptr, nBlocksMB, bc, bn);
LIBXSMM_VLA_DECL(5, element_output_type, doutput_tr,  (element_output_type*)tr_out_ptr, nBlocksMB, bn/2, bk, 2);
LIBXSMM_VLA_DECL(4,       float, dfilter_f32,  (float*)dfilter_f32_ptr, nBlocksIFm, bc, bk);
LIBXSMM_VLA_DECL(2, element_filter_type, dfilter_block,  (element_filter_type*)dfilter_scratch, bk);

const int tr_out_work = nBlocksMB * nBlocksOFm;
const int tr_out_chunksize = (tr_out_work % handle->desc.threads == 0) ? (tr_out_work / handle->desc.threads) : ((tr_out_work / handle->desc.threads) + 1);
const int tr_out_thr_begin = (ltid * tr_out_chunksize < tr_out_work) ? (ltid * tr_out_chunksize) : tr_out_work;
const int tr_out_thr_end = ((ltid + 1) * tr_out_chunksize < tr_out_work) ? ((ltid + 1) * tr_out_chunksize) : tr_out_work;

const int tr_inp_work = nBlocksMB * nBlocksIFm;
const int tr_inp_chunksize = (tr_inp_work % handle->desc.threads == 0) ? (tr_inp_work / handle->desc.threads) : ((tr_inp_work / handle->desc.threads) + 1);
const int tr_inp_thr_begin = (ltid * tr_inp_chunksize < tr_inp_work) ? (ltid * tr_inp_chunksize) : tr_inp_work;
const int tr_inp_thr_end = ((ltid + 1) * tr_inp_chunksize < tr_inp_work) ? ((ltid + 1) * tr_inp_chunksize) : tr_inp_work;

/* These are used for the vnni reformatting of the f32 output  */
__m256i c0, c1;
__m512 a01, b01;
__m512i c01 = LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32();
const __m512i perm_index = LIBXSMM_INTRINSICS_MM512_SET_EPI16(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8, 23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);

if (use_2d_blocking == 1) {
  row_teams = handle->upd_row_teams;
  column_teams = handle->upd_column_teams;
  my_col_id = ltid % column_teams;
  my_row_id = ltid / column_teams;
  im_tasks_per_thread = (nBlocksIFm + row_teams-1)/row_teams;
  in_tasks_per_thread = (nBlocksOFm + column_teams-1)/column_teams;
  my_im_start = LIBXSMM_MIN( my_row_id * im_tasks_per_thread, nBlocksIFm);
  my_im_end = LIBXSMM_MIN( (my_row_id+1) * im_tasks_per_thread, nBlocksIFm);
  my_in_start = LIBXSMM_MIN( my_col_id * in_tasks_per_thread, nBlocksOFm);
  my_in_end = LIBXSMM_MIN( (my_col_id+1) * in_tasks_per_thread, nBlocksOFm);
}

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

/* Required upfront tranposes */
if (bk % 32 == 0) {
  for (mb1ofm1 = tr_out_thr_begin; mb1ofm1 < tr_out_thr_end; mb1ofm1++) {
    mb1 = mb1ofm1/nBlocksOFm;
    ofm1 = mb1ofm1%nBlocksOFm;
    bf16_vnni_reformat((element_output_type*)&LIBXSMM_VLA_ACCESS(4, doutput,  mb1, ofm1, 0, 0, nBlocksOFm, bn, bk), &LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, mb1, 0, 0, 0, nBlocksMB, bn/2, bk, 2), bk, bn, bk, bn);
  }
} else {
  for (mb1ofm1 = tr_out_thr_begin; mb1ofm1 < tr_out_thr_end; mb1ofm1++) {
    mb1 = mb1ofm1/nBlocksOFm;
    ofm1 = mb1ofm1%nBlocksOFm;
    for (mb2 = 0; mb2 < bn; mb2++) {
      for (ofm2 = 0; ofm2 < bk; ofm2++) {
        LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, mb1, mb2/2, ofm2, mb2%2, nBlocksMB, bn/2, bk, 2) = LIBXSMM_VLA_ACCESS(4, doutput,  mb1, ofm1, mb2, ofm2, nBlocksOFm, bn, bk);
      }
    }
  }
}

if (bc % 32 == 0) {
  for (mb1ifm1 = tr_inp_thr_begin; mb1ifm1 < tr_inp_thr_end; mb1ifm1++) {
    mb1 = mb1ifm1/nBlocksIFm;
    ifm1 = mb1ifm1%nBlocksIFm;
    bf16_transpose((element_input_type*)&LIBXSMM_VLA_ACCESS(4, input, mb1, ifm1, 0, 0, nBlocksIFm, bn, bc), &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, mb1, 0, 0, nBlocksMB, bc, bn), bc, bn, bc, bn);
  }
} else {
  for (mb1ifm1 = tr_inp_thr_begin; mb1ifm1 < tr_inp_thr_end; mb1ifm1++) {
    mb1 = mb1ifm1/nBlocksIFm;
    ifm1 = mb1ifm1%nBlocksIFm;
    for (mb2 = 0; mb2 < bn; mb2++) {
      for (ifm2 = 0; ifm2 < bc; ifm2++) {
        LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, mb1, ifm2, mb2, nBlocksMB, bc, bn) = LIBXSMM_VLA_ACCESS(4, input, mb1, ifm1, mb2, ifm2, nBlocksIFm, bn, bc);
      }
    }
  }
}

libxsmm_barrier_wait(handle->barrier, ltid);

if (use_2d_blocking == 1) {
  ifm2 = 0;
  ofm2 = 0;
  if (BF == 1) {
    for (ofm1 = my_in_start; ofm1 < my_in_end; ++ofm1) {
      for (ifm1 = my_im_start; ifm1 < my_im_end; ++ifm1) {
        batchreduce_kernel_zerobeta(&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, 0, 0, ofm2*bbk, 0, nBlocksMB, bn/2, bk, 2), &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, 0, ifm2*bbc, 0, nBlocksMB, bc, bn), &LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc, ofm2*bbk, bk), &blocks);
        /* TODO: Make this vnni reformating in the kernel...  */
        /* Copy result back to vnni format */
        if ((bbc % 2 == 0) && (bbk % 16 == 0)) {
          for (jc = 0; jc < bbc; jc+=2) {
            for (jk = 0; jk < bbk; jk+=16) {
              c1 = _mm256_load_si256((__m256i*)&LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc+jc+1, ofm2*bbk+jk, bk));
              c0 = _mm256_load_si256((__m256i*)&LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc+jc, ofm2*bbk+jk, bk));
              c01 = _mm512_inserti64x4(c01, c0, 0);
              c01 = _mm512_inserti64x4(c01, c1, 1);
              _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc+jc)/2, ofm2*bbk+jk, 0, nBlocksIFm, bc/2, bk, 2), _mm512_permutexvar_epi16(perm_index, c01));
            }
          }
        } else {
          for (ii = 0; ii < bbc; ii++) {
            for (jj = 0; jj < bbk; jj++) {
              LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc+ii)/2, ofm2*bbk+jj, (ifm2*bbc+ii)%2, nBlocksIFm, bc/2, bk, 2) = LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc+ii, ofm2*bbk+jj, bk);
            }
          }
        }
      }
    }
  } else {
    for (bfn = 0; bfn < BF; bfn++) {
      for (ofm1 = my_in_start; ofm1 < my_in_end; ++ofm1) {
        for (ifm1 = my_im_start; ifm1 < my_im_end; ++ifm1) {
          /* initialize current work task to zero */
          if (bfn == 0) {
            for (ii = 0; ii<bbc; ii++) {
              for (jj = 0; jj<bbk; jj++) {
                LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+ii, ofm2*bbk+jj, nBlocksIFm, bc, bk) = 0;
              }
            }
          }
          batchreduce_kernel(&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, bfn*blocks, 0, ofm2*bbk, 0, nBlocksMB, bn/2, bk, 2), &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, bfn*blocks, ifm2*bbc, 0, nBlocksMB, bc, bn), &LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk), &blocks);
          /* Downconvert result to BF16 and vnni format */
          if (bfn == BF-1) {
            if ((bbc % 2 == 0) && (bbk % 16 == 0)) {
              for (jc = 0; jc < bbc; jc+=2) {
                for (jk = 0; jk < bbk; jk+=16) {
                  a01 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+jc+1, ofm2*bbk+jk, nBlocksIFm, bc, bk));
                  b01 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+jc, ofm2*bbk+jk, nBlocksIFm, bc, bk));
                  c01 = _mm512_cvt2_fp32_bf16(a01, b01);
                  _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc+jc)/2, ofm2*bbk+jk, 0, nBlocksIFm, bc/2, bk, 2), _mm512_permutexvar_epi16(perm_index, c01));
                }
              }
            } else {
              for (jc = 0; jc < bbc; jc++) {
                LIBXSMM_DNN_FC_UPD_CONVERT_F32_BF16(&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+jc, ofm2*bbk, nBlocksIFm, bc, bk), &LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc+jc, ofm2*bbk, bk), bbk);
              }
              for (ii = 0; ii < bbc; ii++) {
                for (jj = 0; jj < bbk; jj++) {
                  LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc+ii)/2, ofm2*bbk+jj, (ifm2*bbc+ii)%2, nBlocksIFm, bc/2, bk, 2) = LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc+ii, ofm2*bbk+jj, bk);
                }
              }
            }
          }
        }
      }
    }
  }
} else {
  if (BF == 1) {
    for ( ifm1ofm1 = thr_begin; ifm1ofm1 < thr_end; ++ifm1ofm1 ) {
      ofm1 = ifm1ofm1 / Cck_work;
      ofm2 = (ifm1ofm1 % Cck_work) / Cc_work;
      ifm1 = ((ifm1ofm1 % Cck_work) % Cc_work) / ifm_subtasks;
      ifm2 = ((ifm1ofm1 % Cck_work) % Cc_work) % ifm_subtasks;
      batchreduce_kernel_zerobeta(&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, 0, 0, ofm2*bbk, 0, nBlocksMB, bn/2, bk, 2), &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, 0, ifm2*bbc, 0, nBlocksMB, bc, bn), &LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc, ofm2*bbk, bk), &blocks);
      /* TODO: Make this vnni reformating in the kernel...  */
      /* Copy result back to vnni format */
      if ((bbc % 2 == 0) && (bbk % 16 == 0)) {
        for (jc = 0; jc < bbc; jc+=2) {
          for (jk = 0; jk < bbk; jk+=16) {
            c1 = _mm256_load_si256((__m256i*)&LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc+jc+1, ofm2*bbk+jk, bk));
            c0 = _mm256_load_si256((__m256i*)&LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc+jc, ofm2*bbk+jk, bk));
            c01 = _mm512_inserti64x4(c01, c0, 0);
            c01 = _mm512_inserti64x4(c01, c1, 1);
            _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc+jc)/2, ofm2*bbk+jk, 0, nBlocksIFm, bc/2, bk, 2), _mm512_permutexvar_epi16(perm_index, c01));
          }
        }
      } else {
        for (ii = 0; ii < bbc; ii++) {
          for (jj = 0; jj < bbk; jj++) {
            LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc+ii)/2, ofm2*bbk+jj, (ifm2*bbc+ii)%2, nBlocksIFm, bc/2, bk, 2) = LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc+ii, ofm2*bbk+jj, bk);
          }
        }
      }
    }
  } else {
    for (bfn = 0; bfn < BF; bfn++) {
      for ( ifm1ofm1 = thr_begin; ifm1ofm1 < thr_end; ++ifm1ofm1 ) {
        ofm1 = ifm1ofm1 / Cck_work;
        ofm2 = (ifm1ofm1 % Cck_work) / Cc_work;
        ifm1 = ((ifm1ofm1 % Cck_work) % Cc_work) / ifm_subtasks;
        ifm2 = ((ifm1ofm1 % Cck_work) % Cc_work) % ifm_subtasks;
        /* initialize current work task to zero */
        if (bfn == 0) {
          for (ii = 0; ii<bbc; ii++) {
            for (jj = 0; jj<bbk; jj++) {
              LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+ii, ofm2*bbk+jj, nBlocksIFm, bc, bk) = 0;
            }
          }
        }
        batchreduce_kernel(&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, bfn*blocks, 0, ofm2*bbk, 0, nBlocksMB, bn/2, bk, 2), &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, bfn*blocks, ifm2*bbc, 0, nBlocksMB, bc, bn), &LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk), &blocks);
        /* Downconvert result to BF16 and vnni format */
        if (bfn == BF-1) {
          if ((bbc % 2 == 0) && (bbk % 16 == 0)) {
            for (jc = 0; jc < bbc; jc+=2) {
              for (jk = 0; jk < bbk; jk+=16) {
                a01 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+jc+1, ofm2*bbk+jk, nBlocksIFm, bc, bk));
                b01 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+jc, ofm2*bbk+jk, nBlocksIFm, bc, bk));
                c01 = _mm512_cvt2_fp32_bf16(a01, b01);
                _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc+jc)/2, ofm2*bbk+jk, 0, nBlocksIFm, bc/2, bk, 2), _mm512_permutexvar_epi16(perm_index, c01));
              }
            }
          } else {
            for (jc = 0; jc < bbc; jc++) {
              LIBXSMM_DNN_FC_UPD_CONVERT_F32_BF16(&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+jc, ofm2*bbk, nBlocksIFm, bc, bk), &LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc+jc, ofm2*bbk, bk), bbk);
            }
            for (ii = 0; ii < bbc; ii++) {
              for (jj = 0; jj < bbk; jj++) {
                LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc+ii)/2, ofm2*bbk+jj, (ifm2*bbc+ii)%2, nBlocksIFm, bc/2, bk, 2) = LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc+ii, ofm2*bbk+jj, bk);
              }
            }
          }
        }
      }
    }
  }
}
libxsmm_barrier_wait(handle->barrier, ltid);

#undef _mm512_cvt2_fp32_bf16
#undef LIBXSMM_DNN_FC_UPD_CONVERT_F32_BF16

