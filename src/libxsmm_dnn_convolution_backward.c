/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_convolution_backward.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_custom_custom_f32_f32(libxsmm_dnn_layer* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_nhwc_custom_f32_f32(libxsmm_dnn_layer* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_nhwc_rsck_f32_f32(libxsmm_dnn_layer* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16_emu(libxsmm_dnn_layer* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16(libxsmm_dnn_layer* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16_emu_amx(libxsmm_dnn_layer* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16_amx(libxsmm_dnn_layer* handle, int start_thread, int tid);

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
void bf16_vnni_transpose_16x16_kernel(void* source_void, void* dest_void, int source_stride, int dest_stride)
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
void bf16_vnni_transpose_kernel(libxsmm_bfloat16* src, libxsmm_bfloat16* dst, int M, int N, int ld_in, int ld_out)
{
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE)
  const int _M = M/16, _N = N/16;
  int i = 0, j = 0;
  for (i = 0; i < _N; i++) {
    for (j = 0; j < _M; j++) {
      bf16_vnni_transpose_16x16_kernel((libxsmm_bfloat16*) src+i*16*ld_in+j*32, (libxsmm_bfloat16*) dst+j*16*ld_out+i*32, ld_in*2, ld_out*2);
    }
  }
#else
  LIBXSMM_UNUSED(src); LIBXSMM_UNUSED(dst); LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ld_in); LIBXSMM_UNUSED(ld_out);
#endif
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_custom_custom_f32_f32(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if (handle->use_fallback_bwd_loops == 0) {
    typedef float element_input_type;
    typedef float element_output_type;
    typedef float element_filter_type;
    typedef libxsmm_smmfunction_reducebatch_addr gemm_br_function;
    const libxsmm_blasint ldB = (libxsmm_blasint)handle->ofmblock;
    const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
    const libxsmm_blasint ldC = (handle->spread_input_bwd == 1) ? (libxsmm_blasint)(handle->ifmblock * handle->desc.v) : (libxsmm_blasint)handle->ifmblock;
    const float  beta = (handle->avoid_acc_load_bwd ? 0.f : 1.f);
    int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
    int prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
    int brgemm_pf_oob = 0;
    const char *const env_brgemm_pf_oob = getenv("BRGEMM_PF_OOB");
    if ( 0 == env_brgemm_pf_oob ) {
    } else {
      brgemm_pf_oob = atoi(env_brgemm_pf_oob);
    }
    if (brgemm_pf_oob > 0) {
      prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_BRGEMM_OOB);
    }
    { /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
      gemm_br_function br_gemm_kernel = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, &prefetch_mode);
      gemm_br_function br_gemm_kernel2 = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, &prefetch_mode);
#     include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_generic.tpl.c"
    }
  } else {
    typedef float element_input_type;
    typedef float element_output_type;
    typedef float element_filter_type;
    typedef libxsmm_smmfunction gemm_function;
    const libxsmm_blasint ldC = (libxsmm_blasint)(handle->desc.v*handle->ifmblock);
    { /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
      gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ifmblock, handle->ofw, handle->ofmblock, NULL, NULL, &ldC, NULL, NULL, NULL, NULL);
#     include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback_generic.tpl.c"
    }
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16_emu(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
  if (handle->use_fallback_bwd_loops == 0) {
    typedef libxsmm_bfloat16 element_input_type;
    typedef libxsmm_bfloat16 element_output_type;

    /* some portable macrros fof BF16 <-> FP32 */
#   include "template/libxsmm_dnn_bf16_macros_define.tpl.c"
    { typedef libxsmm_bfloat16 element_filter_type;
      typedef libxsmm_bsmmfunction_reducebatch_addr gemm_br_function;
      typedef libxsmm_bmmfunction_reducebatch_addr gemm_br_function_bf16bf16;
      const libxsmm_blasint ldB = (libxsmm_blasint)handle->ofmblock;
      const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
      const libxsmm_blasint ldC = (handle->spread_input_bwd == 1) ? (libxsmm_blasint)(handle->ifmblock * handle->desc.v) : (libxsmm_blasint)handle->ifmblock;
      const float  beta = (handle->avoid_acc_load_bwd ? 0.f : 1.f);
      int l_flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
      /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
      gemm_br_function br_gemm_kernel = libxsmm_bsmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
      gemm_br_function br_gemm_kernel2 = libxsmm_bsmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
      gemm_br_function_bf16bf16 br_gemm_kernel_bf16bf16 = libxsmm_bmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
      gemm_br_function_bf16bf16 br_gemm_kernel2_bf16bf16 = libxsmm_bmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
#     include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_generic_bf16.tpl.c"
#     include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
    }
  } else {
    const libxsmm_blasint ldC = (libxsmm_blasint)(handle->desc.v*handle->ifmblock);
    typedef libxsmm_bfloat16 element_input_type;
    typedef libxsmm_bfloat16 element_output_type;
    typedef libxsmm_bfloat16 element_filter_type;
    typedef libxsmm_bsmmfunction_reducebatch_strd brgemm_function;
    int l_flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
    int stride_a = handle->ifmblock * handle->desc.R * handle->desc.S * handle->ofmblock * sizeof(libxsmm_bfloat16);
    int stride_b = handle->ofmblock * handle->ofwp * handle->ofhp * sizeof(libxsmm_bfloat16);
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    brgemm_function bf16fp32_brgemm_kernel = libxsmm_bsmmdispatch_reducebatch_strd(handle->ifmblock, handle->ofw, handle->ofmblock, stride_a, stride_b, NULL, NULL, &ldC, NULL, NULL, &l_flags, NULL);

    /* some portable macrros fof BF16 <-> FP32 */
#   include "template/libxsmm_dnn_bf16_macros_define.tpl.c"
#   include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback_generic_bf16.tpl.c"
#   include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16_emu_amx(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
  if (handle->use_fallback_bwd_loops == 0) {
    typedef libxsmm_bfloat16 element_input_type;
    typedef libxsmm_bfloat16 element_output_type;

    /* some portable macrros fof BF16 <-> FP32 */
#   include "template/libxsmm_dnn_bf16_macros_define.tpl.c"
    {
      typedef libxsmm_bfloat16 element_filter_type;
      typedef libxsmm_bsmmfunction gemm_function;
      typedef libxsmm_bsmmfunction_reducebatch_offs gemm_br_function_offs;
      typedef libxsmm_bsmmfunction_reducebatch_strd gemm_br_function_strd;
      gemm_br_function_offs br_gemm_kernel_offs = handle->bwd_compute_kernel_offs;
      gemm_br_function_strd br_gemm_kernel_strd = handle->bwd_compute_kernel_strd;
      gemm_function tile_config_kernel = handle->bwd_config_kernel;
#     include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_generic_bf16_amx.tpl.c"
#     include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
    }
  } else {
    typedef libxsmm_bfloat16 element_input_type;
    typedef libxsmm_bfloat16 element_output_type;
    typedef libxsmm_bfloat16 element_filter_type;
    typedef libxsmm_bsmmfunction_reducebatch_strd brgemm_function;
    const libxsmm_blasint ldC = (libxsmm_blasint)(handle->desc.v*handle->ifmblock);
    int l_flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
    int stride_a = handle->ifmblock * handle->desc.R * handle->desc.S * handle->ofmblock * sizeof(libxsmm_bfloat16);
    int stride_b = handle->ofmblock * handle->ofwp * handle->ofhp * sizeof(libxsmm_bfloat16);
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    brgemm_function bf16fp32_brgemm_kernel = libxsmm_bsmmdispatch_reducebatch_strd(handle->ifmblock, handle->ofw, handle->ofmblock, stride_a, stride_b, NULL, NULL, &ldC, NULL, NULL, &l_flags, NULL);
    /* some portable macrros fof BF16 <-> FP32 */
#   include "template/libxsmm_dnn_bf16_macros_define.tpl.c"
#   include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback_generic_bf16.tpl.c"
#   include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

#if defined(LIBXSMM_INTRINSICS_AVX512_CPX)
  LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CPX)
libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
  if (handle->use_fallback_bwd_loops == 0) {
    typedef libxsmm_bfloat16 element_input_type;
    typedef libxsmm_bfloat16 element_output_type;
    typedef libxsmm_bfloat16 element_filter_type;

#   define LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
    /* some portable macrros fof BF16 <-> FP32 */
#   include "template/libxsmm_dnn_bf16_macros_define.tpl.c"
    {
      typedef libxsmm_bsmmfunction_reducebatch_addr gemm_br_function;
      typedef libxsmm_bmmfunction_reducebatch_addr gemm_br_function_bf16bf16;
      const libxsmm_blasint ldB = (libxsmm_blasint)handle->ofmblock;
      const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
      const libxsmm_blasint ldC = (handle->spread_input_bwd == 1) ? (libxsmm_blasint)(handle->ifmblock * handle->desc.v) : (libxsmm_blasint)handle->ifmblock;
      const float  beta = (handle->avoid_acc_load_bwd ? 0.f : 1.f);
      int l_flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
      /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
      gemm_br_function br_gemm_kernel = libxsmm_bsmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
      gemm_br_function br_gemm_kernel2 = libxsmm_bsmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
      gemm_br_function_bf16bf16 br_gemm_kernel_bf16bf16 = libxsmm_bmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
      gemm_br_function_bf16bf16 br_gemm_kernel2_bf16bf16 = libxsmm_bmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
#     include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_generic_bf16.tpl.c"
#     include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
    }
#   undef LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
  } else {
    typedef libxsmm_bfloat16 element_input_type;
    typedef libxsmm_bfloat16 element_output_type;
    typedef libxsmm_bfloat16 element_filter_type;
    typedef libxsmm_bsmmfunction_reducebatch_strd brgemm_function;
    const libxsmm_blasint ldC = (libxsmm_blasint)(handle->desc.v*handle->ifmblock);
    int l_flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
    int stride_a = handle->ifmblock * handle->desc.R * handle->desc.S * handle->ofmblock * sizeof(libxsmm_bfloat16);
    int stride_b = handle->ofmblock * handle->ofwp * handle->ofhp * sizeof(libxsmm_bfloat16);
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    brgemm_function bf16fp32_brgemm_kernel = libxsmm_bsmmdispatch_reducebatch_strd(handle->ifmblock, handle->ofw, handle->ofmblock, stride_a, stride_b, NULL, NULL, &ldC, NULL, NULL, &l_flags, NULL);
#   define LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
    /* some portable macrros fof BF16 <-> FP32 */
#   include "template/libxsmm_dnn_bf16_macros_define.tpl.c"
#   include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback_generic_bf16.tpl.c"
#   include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#   undef LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#else
  LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  return libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16_emu( handle, start_thread, tid );
}
#endif

#if defined(LIBXSMM_INTRINSICS_AVX512_CPX)
  LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CPX)
libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16_amx(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
  if (handle->use_fallback_bwd_loops == 0) {
    typedef libxsmm_bfloat16 element_input_type;
    typedef libxsmm_bfloat16 element_output_type;
    typedef libxsmm_bfloat16 element_filter_type;

#   define LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
    /* some portable macrros fof BF16 <-> FP32 */
#   include "template/libxsmm_dnn_bf16_macros_define.tpl.c"
    {
      typedef libxsmm_bsmmfunction gemm_function;
      typedef libxsmm_bsmmfunction_reducebatch_offs gemm_br_function_offs;
      typedef libxsmm_bsmmfunction_reducebatch_strd gemm_br_function_strd;
      gemm_br_function_offs br_gemm_kernel_offs = handle->bwd_compute_kernel_offs;
      gemm_br_function_strd br_gemm_kernel_strd = handle->bwd_compute_kernel_strd;
      gemm_function tile_config_kernel = handle->bwd_config_kernel;
#     include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_generic_bf16_amx.tpl.c"
#     include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
    }
#   undef LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
  } else {
    typedef libxsmm_bfloat16 element_input_type;
    typedef libxsmm_bfloat16 element_output_type;
    typedef libxsmm_bfloat16 element_filter_type;
    typedef libxsmm_bsmmfunction_reducebatch_strd brgemm_function;
    const libxsmm_blasint ldC = (libxsmm_blasint)(handle->desc.v*handle->ifmblock);
    int l_flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
    int stride_a = handle->ifmblock * handle->desc.R * handle->desc.S * handle->ofmblock * sizeof(libxsmm_bfloat16);
    int stride_b = handle->ofmblock * handle->ofwp * handle->ofhp * sizeof(libxsmm_bfloat16);
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    brgemm_function bf16fp32_brgemm_kernel = libxsmm_bsmmdispatch_reducebatch_strd(handle->ifmblock, handle->ofw, handle->ofmblock, stride_a, stride_b, NULL, NULL, &ldC, NULL, NULL, &l_flags, NULL);

#   define LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
    /* some portable macrros fof BF16 <-> FP32 */
#   include "template/libxsmm_dnn_bf16_macros_define.tpl.c"
#   include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback_generic_bf16.tpl.c"
#   include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#   undef LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#else
  LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16_amx(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  return libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16_emu_amx( handle, start_thread, tid );
}
#endif

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_nhwc_custom_f32_f32(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if (handle->use_fallback_bwd_loops == 0) {
    typedef float element_input_type;
    typedef float element_output_type;
    typedef float element_filter_type;
    typedef libxsmm_smmfunction_reducebatch_addr gemm_br_function;
    const libxsmm_blasint ldB = (libxsmm_blasint)(handle->blocksofm * handle->ofmblock);
    const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
    const libxsmm_blasint ldC = (handle->spread_input_bwd == 1) ? (libxsmm_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v) : (libxsmm_blasint)(handle->blocksifm * handle->ifmblock);
    const float  beta = (handle->avoid_acc_load_bwd ? 0.f : 1.f);
    int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
    int prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
    int brgemm_pf_oob = 0;
    const char *const env_brgemm_pf_oob = getenv("BRGEMM_PF_OOB");
    if ( 0 == env_brgemm_pf_oob ) {
    } else {
      brgemm_pf_oob = atoi(env_brgemm_pf_oob);
    }
    if (brgemm_pf_oob > 0) {
      prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_BRGEMM_OOB);
    }
    { /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
      gemm_br_function br_gemm_kernel = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, &prefetch_mode);
      gemm_br_function br_gemm_kernel2 = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, &prefetch_mode);
#     define LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
#     include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom-rsck_generic.tpl.c"
#     undef LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
    }
  } else {
    typedef float element_input_type;
    typedef float element_output_type;
    typedef float element_filter_type;
    typedef libxsmm_smmfunction gemm_function;
    const libxsmm_blasint ldB = (libxsmm_blasint)(handle->blocksofm * handle->ofmblock);
    const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
    const libxsmm_blasint ldC = ( (handle->desc.pad_h != handle->desc.pad_h_in) || (handle->desc.pad_w != handle->desc.pad_w_in) ) ?
                                  (libxsmm_blasint)(handle->ifmblock * handle->desc.v) :
                                  (libxsmm_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v);
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ifmblock, handle->ofw, handle->ofmblock, &ldA, &ldB, &ldC, NULL, NULL, NULL, NULL);
#   define LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
#   include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom-rsck_fallback_generic.tpl.c"
#   undef LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_nhwc_rsck_f32_f32(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if (handle->use_fallback_bwd_loops == 0) {
    typedef float element_input_type;
    typedef float element_output_type;
    typedef float element_filter_type;
    typedef libxsmm_smmfunction_reducebatch_addr gemm_br_function;
    const libxsmm_blasint ldB = (libxsmm_blasint)(handle->blocksofm * handle->ofmblock);
    const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
    const libxsmm_blasint ldC = (handle->spread_input_bwd == 1) ? (libxsmm_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v) : (libxsmm_blasint)(handle->blocksifm * handle->ifmblock);
    const float  beta = (handle->avoid_acc_load_bwd ? 0.f : 1.f);
    int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
    int prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
    int brgemm_pf_oob = 0;
    const char *const env_brgemm_pf_oob = getenv("BRGEMM_PF_OOB");
    if ( 0 == env_brgemm_pf_oob ) {
    } else {
      brgemm_pf_oob = atoi(env_brgemm_pf_oob);
    }
    if (brgemm_pf_oob > 0) {
      prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_BRGEMM_OOB);
    }
    { /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
      gemm_br_function br_gemm_kernel = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, &prefetch_mode);
      gemm_br_function br_gemm_kernel2 = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags,  &prefetch_mode);
#     define LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
#     include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom-rsck_generic.tpl.c"
#     undef LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
    }
  } else {
    typedef float element_input_type;
    typedef float element_output_type;
    typedef float element_filter_type;
    typedef libxsmm_smmfunction gemm_function;
    const libxsmm_blasint ldB = (libxsmm_blasint)(handle->blocksofm * handle->ofmblock);
    const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
    const libxsmm_blasint ldC = ( (handle->desc.pad_h != handle->desc.pad_h_in) || (handle->desc.pad_w != handle->desc.pad_w_in) ) ?
                                  (libxsmm_blasint)(handle->ifmblock * handle->desc.v) :
                                  (libxsmm_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v);
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ifmblock, handle->ofw, handle->ofmblock, &ldA, &ldB, &ldC, NULL, NULL, NULL, NULL);
#   define LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
#   include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom-rsck_fallback_generic.tpl.c"
#   undef LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_custom_custom(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->grad_input == 0 || handle->grad_output == 0 || handle->reg_filter == 0 || handle->scratch == 0 ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on AVX512 */
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( (handle->target_archid >= LIBXSMM_X86_AVX512) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT) ) {
    if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_convolve_st_bwd_custom_custom_f32_f32( handle, start_thread, tid);
    }
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
    else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && handle->target_archid >= LIBXSMM_X86_AVX512_CORE && handle->target_archid < LIBXSMM_X86_AVX512_CPX ) {
      status = libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16_emu( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && handle->target_archid >= LIBXSMM_X86_AVX512_CPX && handle->target_archid < LIBXSMM_X86_AVX512_SPR) {
      status = libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && handle->target_archid >= LIBXSMM_X86_AVX512_SPR) {
      status = libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16_amx( handle, start_thread, tid);
    }
#elif defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
    else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && handle->target_archid >= LIBXSMM_X86_AVX512_CORE && handle->target_archid < LIBXSMM_X86_AVX512_SPR) {
      status = libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16_emu( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && handle->target_archid >= LIBXSMM_X86_AVX512_SPR) {
      status = libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16_emu_amx( handle, start_thread, tid);
    }
#endif
    else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      if (handle->use_fallback_bwd_loops == 0) {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_smmfunction_reducebatch_addr gemm_br_function;
        const libxsmm_blasint ldx = ((libxsmm_blasint)handle->ofmblock);
        const libxsmm_blasint ldA = handle->ifmblock;
        const libxsmm_blasint ldC = (handle->spread_input_bwd == 1) ? handle->ifmblock * handle->desc.v : handle->ifmblock;
        const float beta = (handle->avoid_acc_load_bwd) ? 0.f : 1.f;
        int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
        int prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
        int brgemm_pf_oob = 0;
        const char *const env_brgemm_pf_oob = getenv("BRGEMM_PF_OOB");
        if ( 0 == env_brgemm_pf_oob ) {
        } else {
          brgemm_pf_oob = atoi(env_brgemm_pf_oob);
        }
        if (brgemm_pf_oob > 0) {
          prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_BRGEMM_OOB);
        }
        { /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
          gemm_br_function br_gemm_kernel = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, &prefetch_mode);
          gemm_br_function br_gemm_kernel2 = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, &prefetch_mode);
#         include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_generic.tpl.c"
        }
      } else {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_smmfunction gemm_function;
        const libxsmm_blasint ldx = ((libxsmm_blasint)handle->desc.v*handle->ifmblock);
        /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
        gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ifmblock, handle->ofw, handle->ofmblock, NULL, NULL, &ldx, NULL, NULL, NULL, NULL);
#       include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback_generic.tpl.c"
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_nhwc_rsck(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->grad_input == 0 || handle->grad_output == 0 || handle->reg_filter == 0 || handle->scratch == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on AVX512 */
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( (handle->target_archid >= LIBXSMM_X86_AVX512) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT) ) {
    if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_convolve_st_bwd_nhwc_rsck_f32_f32( handle, start_thread, tid);
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      if (handle->use_fallback_bwd_loops == 0) {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_smmfunction_reducebatch_addr gemm_br_function;
        const libxsmm_blasint ldB = (libxsmm_blasint)(handle->blocksofm * handle->ofmblock);
        const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
        const libxsmm_blasint ldC = (handle->spread_input_bwd == 1) ? (libxsmm_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v) : (libxsmm_blasint)(handle->blocksifm * handle->ifmblock);
        const float  beta = (handle->avoid_acc_load_bwd ? 0.f : 1.f);
        int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
        int prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
        int brgemm_pf_oob = 0;
        const char *const env_brgemm_pf_oob = getenv("BRGEMM_PF_OOB");
        if ( 0 == env_brgemm_pf_oob ) {
        } else {
          brgemm_pf_oob = atoi(env_brgemm_pf_oob);
        }
        if (brgemm_pf_oob > 0) {
          prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_BRGEMM_OOB);
        }
        { /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
          gemm_br_function br_gemm_kernel = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, &prefetch_mode);
          gemm_br_function br_gemm_kernel2 = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, &prefetch_mode);
#         define LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
#         include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom-rsck_generic.tpl.c"
#         undef LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
        }
      } else {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_smmfunction gemm_function;
        const libxsmm_blasint ldB = (libxsmm_blasint)(handle->blocksofm * handle->ofmblock);
        const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
        const libxsmm_blasint ldC = ( (handle->desc.pad_h != handle->desc.pad_h_in) || (handle->desc.pad_w != handle->desc.pad_w_in) ) ?
                                      (libxsmm_blasint)(handle->ifmblock * handle->desc.v) :
                                      (libxsmm_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v);
        /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
        gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ifmblock, handle->ofw, handle->ofmblock, &ldA, &ldB, &ldC, NULL, NULL, NULL, NULL);
#       define LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
#       include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom-rsck_fallback_generic.tpl.c"
#       undef LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_nhwc_custom(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->grad_input == 0 || handle->grad_output == 0 || handle->reg_filter == 0 || handle->scratch == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on AVX512 */
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( (handle->target_archid >= LIBXSMM_X86_AVX512) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT) ) {
    if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_convolve_st_bwd_nhwc_custom_f32_f32( handle, start_thread, tid);
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      if (handle->use_fallback_bwd_loops == 0) {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_smmfunction_reducebatch_addr gemm_br_function;
        const libxsmm_blasint ldB = (libxsmm_blasint)(handle->blocksofm * handle->ofmblock);
        const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
        const libxsmm_blasint ldC = (handle->spread_input_bwd == 1) ? (libxsmm_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v) : (libxsmm_blasint)(handle->blocksifm * handle->ifmblock);
        const float  beta = (handle->avoid_acc_load_bwd ? 0.f : 1.f);
        int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
        int prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_NONE);
        int brgemm_pf_oob = 0;
        const char *const env_brgemm_pf_oob = getenv("BRGEMM_PF_OOB");
        if ( 0 == env_brgemm_pf_oob ) {
        } else {
          brgemm_pf_oob = atoi(env_brgemm_pf_oob);
        }
        if (brgemm_pf_oob > 0) {
          prefetch_mode = libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_PREFETCH_BRGEMM_OOB);
        }
        { /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
          gemm_br_function br_gemm_kernel = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, &prefetch_mode);
          gemm_br_function br_gemm_kernel2 = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, &prefetch_mode);
#         define LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
#         include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom-rsck_generic.tpl.c"
#         undef LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
        }
      } else {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_smmfunction gemm_function;
        const libxsmm_blasint ldB = (libxsmm_blasint)(handle->blocksofm * handle->ofmblock);
        const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
        const libxsmm_blasint ldC = ( (handle->desc.pad_h != handle->desc.pad_h_in) || (handle->desc.pad_w != handle->desc.pad_w_in) ) ?
                                      (libxsmm_blasint)(handle->ifmblock * handle->desc.v) :
                                      (libxsmm_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v);
        /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
        gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ifmblock, handle->ofw, handle->ofmblock, &ldA, &ldB, &ldC, NULL, NULL, NULL, NULL);
#       define LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
#       include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom-rsck_fallback_generic.tpl.c"
#       undef LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}

