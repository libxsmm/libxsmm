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
/* Rajkishore Barik, Ankush Mandal, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_convolution_backward.h"
#include <libxsmm_intrinsics_x86.h>
#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <string.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

inline void tran(float *mat, float *matT, float *pmat, float *pmatT) {
  __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;
  __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;

  int mask;
  int64_t idx1[8] __attribute__((aligned(64))) = {2, 3, 0, 1, 6, 7, 4, 5}; 
  int64_t idx2[8] __attribute__((aligned(64))) = {1, 0, 3, 2, 5, 4, 7, 6}; 
  int32_t idx3[16] __attribute__((aligned(64))) = {1, 0, 3, 2, 5 ,4 ,7 ,6 ,9 ,8 , 11, 10, 13, 12 ,15, 14};
  __m512i vidx1 = _mm512_load_epi64(idx1);
  __m512i vidx2 = _mm512_load_epi64(idx2);
  __m512i vidx3 = _mm512_load_epi32(idx3);
/*
  t0 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_castps_si256(_mm256_load_ps(&mat[ 0*16+0]))), _mm256_castps_si256(_mm256_load_ps(&mat[ 8*16+0])), 1);
  _mm_prefetch(pmat, _MM_HINT_T0);
  t1 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_castps_si256(_mm256_load_ps(&mat[ 1*16+0]))), _mm256_castps_si256(_mm256_load_ps(&mat[ 9*16+0])), 1);
  _mm_prefetch(pmat+16, _MM_HINT_T0);
  t2 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_castps_si256(_mm256_load_ps(&mat[ 2*16+0]))), _mm256_castps_si256(_mm256_load_ps(&mat[10*16+0])), 1);
  _mm_prefetch(pmat+32, _MM_HINT_T0);
  t3 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_castps_si256(_mm256_load_ps(&mat[ 3*16+0]))), _mm256_castps_si256(_mm256_load_ps(&mat[11*16+0])), 1);
  _mm_prefetch(pmat+48, _MM_HINT_T0);
  t4 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_castps_si256(_mm256_load_ps(&mat[ 4*16+0]))), _mm256_castps_si256(_mm256_load_ps(&mat[12*16+0])), 1);
  _mm_prefetch(pmat+64, _MM_HINT_T0);
  t5 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_castps_si256(_mm256_load_ps(&mat[ 5*16+0]))), _mm256_castps_si256(_mm256_load_ps(&mat[13*16+0])), 1);
  _mm_prefetch(pmat+80, _MM_HINT_T0);
  t6 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_castps_si256(_mm256_load_ps(&mat[ 6*16+0]))), _mm256_castps_si256(_mm256_load_ps(&mat[14*16+0])), 1);
  _mm_prefetch(pmat+96, _MM_HINT_T0);
  t7 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_castps_si256(_mm256_load_ps(&mat[ 7*16+0]))), _mm256_castps_si256(_mm256_load_ps(&mat[15*16+0])), 1);
  _mm_prefetch(pmat+112, _MM_HINT_T0);
  t8 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_castps_si256(_mm256_load_ps(&mat[ 0*16+8]))), _mm256_castps_si256(_mm256_load_ps(&mat[ 8*16+8])), 1);
  _mm_prefetch(pmat+128, _MM_HINT_T0);
  t9 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_castps_si256(_mm256_load_ps(&mat[ 1*16+8]))), _mm256_castps_si256(_mm256_load_ps(&mat[ 9*16+8])), 1);
  _mm_prefetch(pmat+144, _MM_HINT_T0);
  ta = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_castps_si256(_mm256_load_ps(&mat[ 2*16+8]))), _mm256_castps_si256(_mm256_load_ps(&mat[10*16+8])), 1);
  _mm_prefetch(pmat+160, _MM_HINT_T0);
  tb = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_castps_si256(_mm256_load_ps(&mat[ 3*16+8]))), _mm256_castps_si256(_mm256_load_ps(&mat[11*16+8])), 1);
  _mm_prefetch(pmat+176, _MM_HINT_T0);
  tc = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_castps_si256(_mm256_load_ps(&mat[ 4*16+8]))), _mm256_castps_si256(_mm256_load_ps(&mat[12*16+8])), 1);
  _mm_prefetch(pmat+192, _MM_HINT_T0);
  td = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_castps_si256(_mm256_load_ps(&mat[ 5*16+8]))), _mm256_castps_si256(_mm256_load_ps(&mat[13*16+8])), 1);
  _mm_prefetch(pmat+208, _MM_HINT_T0);
  te = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_castps_si256(_mm256_load_ps(&mat[ 6*16+8]))), _mm256_castps_si256(_mm256_load_ps(&mat[14*16+8])), 1);
  _mm_prefetch(pmat+224, _MM_HINT_T0);
  tf = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_castps_si256(_mm256_load_ps(&mat[ 7*16+8]))), _mm256_castps_si256(_mm256_load_ps(&mat[15*16+8])), 1);
  _mm_prefetch(pmat+240, _MM_HINT_T0);

  mask= 0xcc;
  r0 = _mm512_mask_permutexvar_epi64(t0, (__mmask8)mask, vidx1, t4);
  r1 = _mm512_mask_permutexvar_epi64(t1, (__mmask8)mask, vidx1, t5);
  r2 = _mm512_mask_permutexvar_epi64(t2, (__mmask8)mask, vidx1, t6);
  r3 = _mm512_mask_permutexvar_epi64(t3, (__mmask8)mask, vidx1, t7);
  r8 = _mm512_mask_permutexvar_epi64(t8, (__mmask8)mask, vidx1, tc);
  r9 = _mm512_mask_permutexvar_epi64(t9, (__mmask8)mask, vidx1, td);
  ra = _mm512_mask_permutexvar_epi64(ta, (__mmask8)mask, vidx1, te);
  rb = _mm512_mask_permutexvar_epi64(tb, (__mmask8)mask, vidx1, tf);

  mask= 0x33;
  r4 = _mm512_mask_permutexvar_epi64(t4, (__mmask8)mask, vidx1, t0);
  r5 = _mm512_mask_permutexvar_epi64(t5, (__mmask8)mask, vidx1, t1);
  r6 = _mm512_mask_permutexvar_epi64(t6, (__mmask8)mask, vidx1, t2);
  r7 = _mm512_mask_permutexvar_epi64(t7, (__mmask8)mask, vidx1, t3);
  rc = _mm512_mask_permutexvar_epi64(tc, (__mmask8)mask, vidx1, t8);
  rd = _mm512_mask_permutexvar_epi64(td, (__mmask8)mask, vidx1, t9);
  re = _mm512_mask_permutexvar_epi64(te, (__mmask8)mask, vidx1, ta);
  rf = _mm512_mask_permutexvar_epi64(tf, (__mmask8)mask, vidx1, tb);

  mask = 0xaa;
  t0 = _mm512_mask_permutexvar_epi64(r0, (__mmask8)mask, vidx2, r2);
  t1 = _mm512_mask_permutexvar_epi64(r1, (__mmask8)mask, vidx2, r3);
  t4 = _mm512_mask_permutexvar_epi64(r4, (__mmask8)mask, vidx2, r6);
  t5 = _mm512_mask_permutexvar_epi64(r5, (__mmask8)mask, vidx2, r7);
  t8 = _mm512_mask_permutexvar_epi64(r8, (__mmask8)mask, vidx2, ra);
  t9 = _mm512_mask_permutexvar_epi64(r9, (__mmask8)mask, vidx2, rb);
  tc = _mm512_mask_permutexvar_epi64(rc, (__mmask8)mask, vidx2, re);
  td = _mm512_mask_permutexvar_epi64(rd, (__mmask8)mask, vidx2, rf);

  mask = 0x55;
  t2 = _mm512_mask_permutexvar_epi64(r2, (__mmask8)mask, vidx2, r0);
  t3 = _mm512_mask_permutexvar_epi64(r3, (__mmask8)mask, vidx2, r1);
  t6 = _mm512_mask_permutexvar_epi64(r6, (__mmask8)mask, vidx2, r4);
  t7 = _mm512_mask_permutexvar_epi64(r7, (__mmask8)mask, vidx2, r5);
  ta = _mm512_mask_permutexvar_epi64(ra, (__mmask8)mask, vidx2, r8);
  tb = _mm512_mask_permutexvar_epi64(rb, (__mmask8)mask, vidx2, r9);
  te = _mm512_mask_permutexvar_epi64(re, (__mmask8)mask, vidx2, rc);
  tf = _mm512_mask_permutexvar_epi64(rf, (__mmask8)mask, vidx2, rd);

  mask = 0xaaaa;
  r0 = _mm512_mask_permutexvar_epi32(t0, (__mmask16)mask, vidx3, t1);
  r2 = _mm512_mask_permutexvar_epi32(t2, (__mmask16)mask, vidx3, t3);
  r4 = _mm512_mask_permutexvar_epi32(t4, (__mmask16)mask, vidx3, t5);
  r6 = _mm512_mask_permutexvar_epi32(t6, (__mmask16)mask, vidx3, t7);
  r8 = _mm512_mask_permutexvar_epi32(t8, (__mmask16)mask, vidx3, t9);
  ra = _mm512_mask_permutexvar_epi32(ta, (__mmask16)mask, vidx3, tb);
  rc = _mm512_mask_permutexvar_epi32(tc, (__mmask16)mask, vidx3, td);
  re = _mm512_mask_permutexvar_epi32(te, (__mmask16)mask, vidx3, tf);    

  mask = 0x5555;
  r1 = _mm512_mask_permutexvar_epi32(t1, (__mmask16)mask, vidx3, t0);
  r3 = _mm512_mask_permutexvar_epi32(t3, (__mmask16)mask, vidx3, t2);
  r5 = _mm512_mask_permutexvar_epi32(t5, (__mmask16)mask, vidx3, t4);
  r7 = _mm512_mask_permutexvar_epi32(t7, (__mmask16)mask, vidx3, t6);
  r9 = _mm512_mask_permutexvar_epi32(t9, (__mmask16)mask, vidx3, t8);  
  rb = _mm512_mask_permutexvar_epi32(tb, (__mmask16)mask, vidx3, ta);  
  rd = _mm512_mask_permutexvar_epi32(td, (__mmask16)mask, vidx3, tc);
  rf = _mm512_mask_permutexvar_epi32(tf, (__mmask16)mask, vidx3, te);

  _mm512_store_ps(&matT[ 0*16], _mm512_castsi512_ps(r0));
  _mm_prefetch(pmatT, _MM_HINT_T0);
  _mm512_store_ps(&matT[ 1*16], _mm512_castsi512_ps(r1));
  _mm_prefetch(pmatT+16, _MM_HINT_T0);
  _mm512_store_ps(&matT[ 2*16], _mm512_castsi512_ps(r2));
  _mm_prefetch(pmatT+32, _MM_HINT_T0);
  _mm512_store_ps(&matT[ 3*16], _mm512_castsi512_ps(r3));
  _mm_prefetch(pmatT+48, _MM_HINT_T0);
  _mm512_store_ps(&matT[ 4*16], _mm512_castsi512_ps(r4));
  _mm_prefetch(pmatT+64, _MM_HINT_T0);
  _mm512_store_ps(&matT[ 5*16], _mm512_castsi512_ps(r5));
  _mm_prefetch(pmatT+80, _MM_HINT_T0);
  _mm512_store_ps(&matT[ 6*16], _mm512_castsi512_ps(r6));
  _mm_prefetch(pmatT+96, _MM_HINT_T0);
  _mm512_store_ps(&matT[ 7*16], _mm512_castsi512_ps(r7));
  _mm_prefetch(pmatT+112, _MM_HINT_T0);
  _mm512_store_ps(&matT[ 8*16], _mm512_castsi512_ps(r8));
  _mm_prefetch(pmatT+128, _MM_HINT_T0);
  _mm512_store_ps(&matT[ 9*16], _mm512_castsi512_ps(r9));
  _mm_prefetch(pmatT+144, _MM_HINT_T0);
  _mm512_store_ps(&matT[10*16], _mm512_castsi512_ps(ra));
  _mm_prefetch(pmatT+160, _MM_HINT_T0);
  _mm512_store_ps(&matT[11*16], _mm512_castsi512_ps(rb));
  _mm_prefetch(pmatT+176, _MM_HINT_T0);
  _mm512_store_ps(&matT[12*16], _mm512_castsi512_ps(rc));
  _mm_prefetch(pmatT+192, _MM_HINT_T0);
  _mm512_store_ps(&matT[13*16], _mm512_castsi512_ps(rd));
  _mm_prefetch(pmatT+208, _MM_HINT_T0);
  _mm512_store_ps(&matT[14*16], _mm512_castsi512_ps(re));
  _mm_prefetch(pmatT+224, _MM_HINT_T0);
  _mm512_store_ps(&matT[15*16], _mm512_castsi512_ps(rf));
  _mm_prefetch(pmatT+240, _MM_HINT_T0);*/
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_custom_custom(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->grad_input == 0 || handle->grad_output == 0 || handle->reg_filter == 0 || handle->scratch1 == 0 ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* for low/mixed precision we need some scratch to be bound */
  if ( (handle->datatype != handle->datatype_itm) && (handle->scratch7 == 0) ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if (handle->code_bwd[0].xconv.sconv == 0) {
    if (handle->datatype == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback.tpl.c"
#undef INPUT_PADDING
      } else {
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback.tpl.c"
      }
    } else if (handle->datatype ==  LIBXSMM_DNN_DATATYPE_I16 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_I32 ) {
      typedef int element_input_type;
      typedef short element_output_type;
      typedef short element_filter_type;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback.tpl.c"
#undef INPUT_PADDING
      } else {
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback.tpl.c"
      }
    } else if (handle->datatype == LIBXSMM_DNN_DATATYPE_I8 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_I16 && (handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0 ) {
      typedef unsigned short element_input_type;
      typedef char element_output_type;
      typedef char element_filter_type;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback.tpl.c"
#undef INPUT_PADDING
      } else {
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback.tpl.c"
      }
    } else if (handle->datatype == LIBXSMM_DNN_DATATYPE_I8 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_I32 && (handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0 ) {
      typedef unsigned int element_input_type;
      typedef char element_output_type;
      typedef char element_filter_type;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback.tpl.c"
#undef INPUT_PADDING
      } else {
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback.tpl.c"
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
#if 0
      if (handle->desc.N*handle->blocksifm >= handle->desc.threads) {
#endif
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxsmm_sconvfunction libxsmm_convfunction;
      typedef libxsmm_smmfunction libxsmm_mmfunction;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom.tpl.c"
#undef INPUT_PADDING
      } else {
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom.tpl.c"
      }
#if 0
      }
      else {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_img_par.tpl.c"
      }
#endif
    } else if (handle->datatype ==  LIBXSMM_DNN_DATATYPE_I16 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_I32 ) {
      typedef int element_input_type;
      typedef short element_output_type;
      typedef short element_filter_type;
      typedef libxsmm_wconvfunction_bwd libxsmm_convfunction;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_1.tpl.c"
#undef INPUT_PADDING
      } else {
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_1.tpl.c"
      }
    } else if (handle->datatype == LIBXSMM_DNN_DATATYPE_I8 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_I16 && (handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0 ) {
      typedef unsigned short element_input_type;
      typedef char element_output_type;
      typedef char element_filter_type;
      typedef libxsmm_busconvfunction_bwd libxsmm_convfunction;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_1.tpl.c"
#undef INPUT_PADDING
      } else {
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_1.tpl.c"
      }
    } else if (handle->datatype == LIBXSMM_DNN_DATATYPE_I8 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_I32 && (handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0 ) {
      typedef unsigned int element_input_type;
      typedef char element_output_type;
      typedef char element_filter_type;
      typedef libxsmm_budconvfunction_bwd libxsmm_convfunction;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_1.tpl.c"
#undef INPUT_PADDING
      } else {
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_1.tpl.c"
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}

LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_nhwc_rsck(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->grad_input == 0 || handle->grad_output == 0 || handle->reg_filter == 0 || handle->scratch1 == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if (handle->code_bwd[0].xconv.sconv == 0) {
    if (handle->datatype == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
#include "template/libxsmm_dnn_convolve_st_bwd_nhwc_rsck_fallback.tpl.c"
#undef INPUT_PADDING
      } else {
# include "template/libxsmm_dnn_convolve_st_bwd_nhwc_rsck_fallback.tpl.c"
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
#if 0
      if (handle->desc.N*handle->blocksifm >= handle->desc.threads) {
#endif
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
        if (handle->padding_flag == 1) {
#define INPUT_PADDING
#include "template/libxsmm_dnn_convolve_st_bwd_nhwc_rsck.tpl.c"
#undef INPUT_PADDING
        } else {
# include "template/libxsmm_dnn_convolve_st_bwd_nhwc_rsck.tpl.c"
        }
#if 0
      }
      else {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
# include "template/libxsmm_dnn_convolve_st_bwd_nhwc_rsck_img_par.tpl.c"
      }
#endif
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}

LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_nhwc_custom(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->grad_input == 0 || handle->grad_output == 0 || handle->reg_filter == 0 || handle->scratch1 == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if (handle->code_bwd[0].xconv.sconv == 0) {
    if (handle->datatype == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
# include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom_fallback.tpl.c"
#undef INPUT_PADDING
      } else {
# include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom_fallback.tpl.c"
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
#if 0
      if (handle->desc.N*handle->blocksifm >= handle->desc.threads) {
#endif
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
        if (handle->padding_flag == 1) {
#define INPUT_PADDING
# include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom.tpl.c"
#undef INPUT_PADDING
        } else {
# include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom.tpl.c"
        }
#if 0
      }
      else {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
# include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom_img_par.tpl.c"
      }
#endif
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}

