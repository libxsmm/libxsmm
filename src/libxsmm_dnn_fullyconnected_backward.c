/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_fullyconnected_backward.h"
#include <libxsmm_intrinsics_x86.h>
#include "libxsmm_main.h"
#include <libxsmm.h>
#define STRIDE_BRGEMM

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <string.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwd_custom_f32_f32(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwd_ncnc_kcck_f32_f32(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwd_custom_bf16_f32(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwd_ncnc_kcck_bf16_bf16(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid);

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
void bf16_vnni_transpose_16x16(void* source_void, void* dest_void, int source_stride, int dest_stride)
{
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE)
  libxsmm_bfloat16 *source = (libxsmm_bfloat16*)source_void;
  libxsmm_bfloat16 *dest = (libxsmm_bfloat16*)dest_void;
  __m512i zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
  __m512i tmp0, tmp1, tmp2, tmp3;
  const __m512i abcdefgh_to_abefcdgh = _mm512_set4_epi32(0x0f0e0b0a, 0x0d0c0908, 0x07060302, 0x05040100);

  zmm0 = _mm512_load_epi32(source);
  zmm1 = _mm512_load_epi32(source + source_stride);
  zmm2 = _mm512_load_epi32(source + source_stride*2);
  zmm3 = _mm512_load_epi32(source + source_stride*3);
  zmm4 = _mm512_load_epi32(source + source_stride*4);
  zmm5 = _mm512_load_epi32(source + source_stride*5);
  zmm6 = _mm512_load_epi32(source + source_stride*6);
  zmm7 = _mm512_load_epi32(source + source_stride*7);

  zmm0 = _mm512_shuffle_epi8(zmm0, abcdefgh_to_abefcdgh);
  zmm1 = _mm512_shuffle_epi8(zmm1, abcdefgh_to_abefcdgh);
  zmm2 = _mm512_shuffle_epi8(zmm2, abcdefgh_to_abefcdgh);
  zmm3 = _mm512_shuffle_epi8(zmm3, abcdefgh_to_abefcdgh);
  zmm4 = _mm512_shuffle_epi8(zmm4, abcdefgh_to_abefcdgh);
  zmm5 = _mm512_shuffle_epi8(zmm5, abcdefgh_to_abefcdgh);
  zmm6 = _mm512_shuffle_epi8(zmm6, abcdefgh_to_abefcdgh);
  zmm7 = _mm512_shuffle_epi8(zmm7, abcdefgh_to_abefcdgh);

  tmp0 = _mm512_unpacklo_epi64(zmm0, zmm1);
  tmp1 = _mm512_unpackhi_epi64(zmm0, zmm1);
  tmp2 = _mm512_unpacklo_epi64(zmm2, zmm3);
  tmp3 = _mm512_unpackhi_epi64(zmm2, zmm3);
  zmm0 = _mm512_unpacklo_epi64(zmm4, zmm5);
  zmm1 = _mm512_unpackhi_epi64(zmm4, zmm5);
  zmm2 = _mm512_unpacklo_epi64(zmm6, zmm7);
  zmm3 = _mm512_unpackhi_epi64(zmm6, zmm7);

  zmm4 = _mm512_shuffle_i32x4(tmp0, tmp2, 0x88);
  zmm6 = _mm512_shuffle_i32x4(tmp0, tmp2, 0xdd);
  zmm5 = _mm512_shuffle_i32x4(tmp1, tmp3, 0x88);
  zmm7 = _mm512_shuffle_i32x4(tmp1, tmp3, 0xdd);
  tmp0 = _mm512_shuffle_i32x4(zmm0, zmm2, 0x88);
  tmp1 = _mm512_shuffle_i32x4(zmm0, zmm2, 0xdd);
  tmp2 = _mm512_shuffle_i32x4(zmm1, zmm3, 0x88);
  tmp3 = _mm512_shuffle_i32x4(zmm1, zmm3, 0xdd);

  zmm0 = _mm512_shuffle_i32x4(zmm4, tmp0, 0x88);
  zmm1 = _mm512_shuffle_i32x4(zmm5, tmp2, 0x88);
  zmm2 = _mm512_shuffle_i32x4(zmm6, tmp1, 0x88);
  zmm3 = _mm512_shuffle_i32x4(zmm7, tmp3, 0x88);
  zmm4 = _mm512_shuffle_i32x4(zmm4, tmp0, 0xdd);
  zmm5 = _mm512_shuffle_i32x4(zmm5, tmp2, 0xdd);
  zmm6 = _mm512_shuffle_i32x4(zmm6, tmp1, 0xdd);
  zmm7 = _mm512_shuffle_i32x4(zmm7, tmp3, 0xdd);

  _mm512_store_epi32(dest, zmm0);
  _mm512_store_epi32(dest + dest_stride, zmm1);
  _mm512_store_epi32(dest + dest_stride * 2, zmm2);
  _mm512_store_epi32(dest + dest_stride * 3, zmm3);
  _mm512_store_epi32(dest + dest_stride * 4, zmm4);
  _mm512_store_epi32(dest + dest_stride * 5, zmm5);
  _mm512_store_epi32(dest + dest_stride * 6, zmm6);
  _mm512_store_epi32(dest + dest_stride * 7, zmm7);
#else
  LIBXSMM_UNUSED(source_void); LIBXSMM_UNUSED(dest_void); LIBXSMM_UNUSED(source_stride); LIBXSMM_UNUSED(dest_stride);
#endif
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
void bf16_vnni_transpose(libxsmm_bfloat16* src, libxsmm_bfloat16* dst, int M, int N, int ld_in, int ld_out)
{
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE)
  const int _M = M/16, _N = N/16;
  int i = 0, j = 0;
  for (i = 0; i < _N; i++) {
    for (j = 0; j < _M; j++) {
      bf16_vnni_transpose_16x16((libxsmm_bfloat16*) src+i*16*ld_in+j*32, (libxsmm_bfloat16*) dst+j*16*ld_out+i*32, ld_in*2, ld_out*2);
    }
  }
#else
  LIBXSMM_UNUSED(src); LIBXSMM_UNUSED(dst); LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ld_in); LIBXSMM_UNUSED(ld_out);
#endif
}

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwd_custom_f32_f32(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  libxsmm_blasint lda = (libxsmm_blasint)handle->ifmblock;
  libxsmm_blasint ldb = (libxsmm_blasint)handle->desc.K;
  libxsmm_blasint ldc = (libxsmm_blasint)handle->desc.C;
  element_input_type alpha = (element_input_type)1;
  element_input_type beta = (element_input_type)0;

  if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
    typedef libxsmm_smmfunction gemm_function;
    gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ifmblock, handle->desc.N, handle->desc.K, &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
# include "template/libxsmm_dnn_fullyconnected_st_bwd_custom_generic.tpl.c"
  } else {
    status = LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwd_custom_bf16_f32(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef float element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;
  typedef libxsmm_smmfunction gemm_function;
  libxsmm_blasint lda = (libxsmm_blasint)handle->ifmblock;
  libxsmm_blasint ldb = (libxsmm_blasint)handle->desc.K;
  libxsmm_blasint ldc = (libxsmm_blasint)handle->desc.C;
  float alpha = (element_input_type)1;
  float beta = (element_input_type)0;

  if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
    gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ifmblock, handle->desc.N, handle->desc.K, &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
# define LIBXSMM_DNN_FULLYCONNECTED_BWD_BF16_F32
# include "template/libxsmm_dnn_fullyconnected_st_bwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FULLYCONNECTED_BWD_BF16_F32
  } else {
    status = LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwd_ncnc_kcck_f32_f32(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;

  if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
#ifdef ADDRESS_BRGEMM
    libxsmm_smmfunction_reducebatch_addr batchreduce_kernel = handle->gemm_bwd.xgemm.smra;
#endif
#ifdef OFFSET_BRGEMM
    libxsmm_smmfunction_reducebatch_offs batchreduce_kernel = handle->gemm_bwd.xgemm.smro;
#endif
#ifdef STRIDE_BRGEMM
    libxsmm_smmfunction_reducebatch_strd batchreduce_kernel = handle->gemm_bwd.xgemm.smrs;
#endif
# include "template/libxsmm_dnn_fullyconnected_st_bwd_ncnc_kcck_generic.tpl.c"
  } else {
    status = LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CPX)
libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwd_ncnc_kcck_bf16_bf16(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;

  if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
    libxsmm_bsmmfunction_reducebatch_strd batchreduce_kernel = handle->gemm_bwd.xgemm.bsmrs;
    libxsmm_bmmfunction_reducebatch_strd batchreduce_kernel_zerobeta = handle->gemm_bwd2.xgemm.bmrs;
# include "template/libxsmm_dnn_fullyconnected_st_bwd_ncnc_kcck_generic_bf16.tpl.c"
  } else {
    status = LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwd_custom(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if all required tensors are bound */
  if (handle->grad_input == 0 || handle->grad_output == 0 ||
      handle->reg_filter == 0 || handle->scratch == 0         ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on an AVX512 platform */
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( libxsmm_target_archid >= LIBXSMM_X86_AVX512 ) {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_fullyconnected_st_bwd_custom_f32_f32( handle, start_thread, tid);
    }
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__*/
    else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_fullyconnected_st_bwd_custom_bf16_f32( handle, start_thread, tid);
    }
#endif
    else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxsmm_smmfunction gemm_function;
      libxsmm_blasint lda = (libxsmm_blasint)handle->ifmblock;
      libxsmm_blasint ldb = (libxsmm_blasint)handle->desc.K;
      libxsmm_blasint ldc = (libxsmm_blasint)handle->desc.C;
      element_input_type alpha = (element_input_type)1;
      element_input_type beta = (element_input_type)0;

      if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
        gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ifmblock, handle->desc.N, handle->desc.K, &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
# include "template/libxsmm_dnn_fullyconnected_st_bwd_custom_generic.tpl.c"
      } else {
        status = LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION;
      }
    } else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef libxsmm_bfloat16 element_input_type;
      typedef float element_output_type;
      typedef libxsmm_bfloat16 element_filter_type;
      typedef libxsmm_smmfunction gemm_function;
      libxsmm_blasint lda = (libxsmm_blasint)handle->ifmblock;
      libxsmm_blasint ldb = (libxsmm_blasint)handle->desc.K;
      libxsmm_blasint ldc = (libxsmm_blasint)handle->desc.C;
      float alpha = (element_input_type)1;
      float beta = (element_input_type)0;

      if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
        gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ifmblock, handle->desc.N, handle->desc.K, &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
# define LIBXSMM_DNN_FULLYCONNECTED_BWD_BF16_F32
# include "template/libxsmm_dnn_fullyconnected_st_bwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FULLYCONNECTED_BWD_BF16_F32
      } else {
        status = LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION;
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwd_ncnc_kcck(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if all required tensors are bound */
  if (handle->grad_input == 0 || handle->grad_output == 0 ||
      handle->reg_filter == 0 || handle->scratch == 0         ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on an AVX512 platform */
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( libxsmm_target_archid >= LIBXSMM_X86_AVX512 ) {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_fullyconnected_st_bwd_ncnc_kcck_f32_f32( handle, start_thread, tid);
    }
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
    else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_CORE ) {
      status = libxsmm_dnn_fullyconnected_st_bwd_ncnc_kcck_bf16_bf16( handle, start_thread, tid);
    }
#endif
    else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;

      if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
#ifdef ADDRESS_BRGEMM
        libxsmm_smmfunction_reducebatch_addr batchreduce_kernel = handle->gemm_bwd.xgemm.smra;
#endif
#ifdef OFFSET_BRGEMM
        libxsmm_smmfunction_reducebatch_offs batchreduce_kernel = handle->gemm_bwd.xgemm.smro;
#endif
#ifdef STRIDE_BRGEMM
        libxsmm_smmfunction_reducebatch_strd batchreduce_kernel = handle->gemm_bwd.xgemm.smrs;
#endif
# include "template/libxsmm_dnn_fullyconnected_st_bwd_ncnc_kcck_generic.tpl.c"
      } else {
        status = LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION;
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwd_nhwc(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  LIBXSMM_UNUSED( handle );
  LIBXSMM_UNUSED( start_thread );
  LIBXSMM_UNUSED( tid );
  return status;
}

