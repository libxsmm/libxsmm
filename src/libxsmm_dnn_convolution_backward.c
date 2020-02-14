/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Rajkishore Barik, Ankush Mandal, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_convolution_backward.h"
#include "libxsmm_main.h"


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_custom_custom_f32_f32(libxsmm_dnn_layer* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_nhwc_custom_f32_f32(libxsmm_dnn_layer* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_nhwc_rsck_f32_f32(libxsmm_dnn_layer* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16_emu(libxsmm_dnn_layer* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16(libxsmm_dnn_layer* handle, int start_thread, int tid);


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_custom_custom_f32_f32(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if (handle->use_fallback_bwd_loops == 0) {
    const libxsmm_blasint ldB = (libxsmm_blasint)handle->ofmblock;
    const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
    const libxsmm_blasint ldC = (handle->spread_input_bwd == 1) ? (libxsmm_blasint)(handle->ifmblock * handle->desc.v) : (libxsmm_blasint)handle->ifmblock;
    const float  beta = (handle->avoid_acc_load_bwd ? 0.f : 1.f);
    typedef float element_input_type;
    typedef float element_output_type;
    typedef float element_filter_type;
    typedef libxsmm_smmfunction_reducebatch_addr gemm_br_function;
    int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    gemm_br_function br_gemm_kernel = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
    gemm_br_function br_gemm_kernel2 = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
# include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_generic.tpl.c"
  } else {
    const libxsmm_blasint ldC = (libxsmm_blasint)(handle->desc.v*handle->ifmblock);
    typedef float element_input_type;
    typedef float element_output_type;
    typedef float element_filter_type;
    typedef libxsmm_smmfunction gemm_function;
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ifmblock, handle->ofw, handle->ofmblock, NULL, NULL, &ldC, NULL, NULL, NULL, NULL);
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback_generic.tpl.c"
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
    const libxsmm_blasint ldB = (libxsmm_blasint)handle->ofmblock;
    const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
    const libxsmm_blasint ldC = (handle->spread_input_bwd == 1) ? (libxsmm_blasint)(handle->ifmblock * handle->desc.v) : (libxsmm_blasint)handle->ifmblock;
    const float  beta = (handle->avoid_acc_load_bwd ? 0.f : 1.f);
    typedef libxsmm_bfloat16 element_input_type;
    typedef libxsmm_bfloat16 element_output_type;
    typedef libxsmm_bfloat16 element_filter_type;
    typedef libxsmm_bsmmfunction_reducebatch_addr gemm_br_function;
    typedef libxsmm_bmmfunction_reducebatch_addr gemm_br_function_bf16bf16;
    int l_flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    gemm_br_function br_gemm_kernel = libxsmm_bsmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
    gemm_br_function br_gemm_kernel2 = libxsmm_bsmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
    gemm_br_function_bf16bf16 br_gemm_kernel_bf16bf16 = libxsmm_bmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
    gemm_br_function_bf16bf16 br_gemm_kernel2_bf16bf16 = libxsmm_bmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
# include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_generic_bf16.tpl.c"
  } else {
    status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
    return status;
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
    const libxsmm_blasint ldB = (libxsmm_blasint)handle->ofmblock;
    const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
    const libxsmm_blasint ldC = (handle->spread_input_bwd == 1) ? (libxsmm_blasint)(handle->ifmblock * handle->desc.v) : (libxsmm_blasint)handle->ifmblock;
    const float  beta = (handle->avoid_acc_load_bwd ? 0.f : 1.f);
    typedef libxsmm_bfloat16 element_input_type;
    typedef libxsmm_bfloat16 element_output_type;
    typedef libxsmm_bfloat16 element_filter_type;
    typedef libxsmm_bsmmfunction_reducebatch_addr gemm_br_function;
    typedef libxsmm_bmmfunction_reducebatch_addr gemm_br_function_bf16bf16;
    int l_flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    gemm_br_function br_gemm_kernel = libxsmm_bsmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
    gemm_br_function br_gemm_kernel2 = libxsmm_bsmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
    gemm_br_function_bf16bf16 br_gemm_kernel_bf16bf16 = libxsmm_bmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
    gemm_br_function_bf16bf16 br_gemm_kernel2_bf16bf16 = libxsmm_bmmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
#define LIBXSMM_DNN_CONVOLUTION_BWD_AVX512_CPX
# include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_CONVOLUTION_BWD_AVX512_CPX
  } else {
    status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
    return status;
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


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_nhwc_custom_f32_f32(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if (handle->use_fallback_bwd_loops == 0) {
    const libxsmm_blasint ldB = (libxsmm_blasint)(handle->blocksofm * handle->ofmblock);
    const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
    const libxsmm_blasint ldC = (handle->spread_input_bwd == 1) ? (libxsmm_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v) : (libxsmm_blasint)(handle->blocksifm * handle->ifmblock);
    const float  beta = (handle->avoid_acc_load_bwd ? 0.f : 1.f);
    typedef float element_input_type;
    typedef float element_output_type;
    typedef float element_filter_type;
    typedef libxsmm_smmfunction_reducebatch_addr gemm_br_function;
    int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    gemm_br_function br_gemm_kernel = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
    gemm_br_function br_gemm_kernel2 = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
#define LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
# include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom-rsck_generic.tpl.c"
#undef LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
  } else {
    const libxsmm_blasint ldB = (libxsmm_blasint)(handle->blocksofm * handle->ofmblock);
    const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
    const libxsmm_blasint ldC = (libxsmm_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v);
    typedef float element_input_type;
    typedef float element_output_type;
    typedef float element_filter_type;
    typedef libxsmm_smmfunction gemm_function;
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ifmblock, handle->ofw, handle->ofmblock, &ldA, &ldB, &ldC, NULL, NULL, NULL, NULL);
#define LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
#include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom-rsck_fallback_generic.tpl.c"
#undef LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
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
    const libxsmm_blasint ldB = (libxsmm_blasint)(handle->blocksofm * handle->ofmblock);
    const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
    const libxsmm_blasint ldC = (handle->spread_input_bwd == 1) ? (libxsmm_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v) : (libxsmm_blasint)(handle->blocksifm * handle->ifmblock);
    const float  beta = (handle->avoid_acc_load_bwd ? 0.f : 1.f);
    typedef float element_input_type;
    typedef float element_output_type;
    typedef float element_filter_type;
    typedef libxsmm_smmfunction_reducebatch_addr gemm_br_function;
    int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    gemm_br_function br_gemm_kernel = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
    gemm_br_function br_gemm_kernel2 = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
#define LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
# include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom-rsck_generic.tpl.c"
#undef LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
  } else {
    const libxsmm_blasint ldB = (libxsmm_blasint)(handle->blocksofm * handle->ofmblock);
    const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
    const libxsmm_blasint ldC = (libxsmm_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v);
    typedef float element_input_type;
    typedef float element_output_type;
    typedef float element_filter_type;
    typedef libxsmm_smmfunction gemm_function;
    /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
    gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ifmblock, handle->ofw, handle->ofmblock, &ldA, &ldB, &ldC, NULL, NULL, NULL, NULL);
#define LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
#include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom-rsck_fallback_generic.tpl.c"
#undef LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
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
  if (handle->grad_input == 0 || handle->grad_output == 0 || handle->reg_filter == 0 || handle->scratch1 == 0 ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on AVX512 */
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( libxsmm_target_archid >= LIBXSMM_X86_AVX512 ) {
    if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_convolve_st_bwd_custom_custom_f32_f32( handle, start_thread, tid);
    }
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
    else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_CORE && libxsmm_target_archid < LIBXSMM_X86_AVX512_CPX ) {
      status = libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16_emu( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_CPX ) {
      status = libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16( handle, start_thread, tid);
    }
#elif defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
    else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_CORE ) {
      status = libxsmm_dnn_convolve_st_bwd_custom_custom_bf16_bf16_emu( handle, start_thread, tid);
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
        const libxsmm_blasint ldx = ((libxsmm_blasint)handle->ofmblock);
        const libxsmm_blasint ldA = handle->ifmblock;
        const libxsmm_blasint ldC = (handle->spread_input_bwd == 1) ? handle->ifmblock * handle->desc.v : handle->ifmblock;
        const float beta = (handle->avoid_acc_load_bwd) ? 0.f : 1.f;
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_smmfunction_reducebatch_addr gemm_br_function;
        int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
        /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
        gemm_br_function br_gemm_kernel = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
        gemm_br_function br_gemm_kernel2 = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldx, &ldC, NULL, &beta, &l_flags, NULL);
# include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_generic.tpl.c"
      } else {
        const libxsmm_blasint ldx = ((libxsmm_blasint)handle->desc.v*handle->ifmblock);
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_smmfunction gemm_function;
        /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
        gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ifmblock, handle->ofw, handle->ofmblock, NULL, NULL, &ldx, NULL, NULL, NULL, NULL);
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback_generic.tpl.c"
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
  if (handle->grad_input == 0 || handle->grad_output == 0 || handle->reg_filter == 0 || handle->scratch1 == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on AVX512 */
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( libxsmm_target_archid >= LIBXSMM_X86_AVX512 ) {
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
        const libxsmm_blasint ldB = (libxsmm_blasint)(handle->blocksofm * handle->ofmblock);
        const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
        const libxsmm_blasint ldC = (handle->spread_input_bwd == 1) ? (libxsmm_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v) : (libxsmm_blasint)(handle->blocksifm * handle->ifmblock);
        const float  beta = (handle->avoid_acc_load_bwd ? 0.f : 1.f);
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_smmfunction_reducebatch_addr gemm_br_function;
        int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
        /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
        gemm_br_function br_gemm_kernel = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
        gemm_br_function br_gemm_kernel2 = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
#define LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
# include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom-rsck_generic.tpl.c"
#undef LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
      } else {
        const libxsmm_blasint ldB = (libxsmm_blasint)(handle->blocksofm * handle->ofmblock);
        const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
        const libxsmm_blasint ldC = (libxsmm_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v);
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_smmfunction gemm_function;
        /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
        gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ifmblock, handle->ofw, handle->ofmblock, &ldA, &ldB, &ldC, NULL, NULL, NULL, NULL);
#define LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
#include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom-rsck_fallback_generic.tpl.c"
#undef LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_RSCK
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
  if (handle->grad_input == 0 || handle->grad_output == 0 || handle->reg_filter == 0 || handle->scratch1 == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on AVX512 */
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( libxsmm_target_archid >= LIBXSMM_X86_AVX512 ) {
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
        const libxsmm_blasint ldB = (libxsmm_blasint)(handle->blocksofm * handle->ofmblock);
        const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
        const libxsmm_blasint ldC = (handle->spread_input_bwd == 1) ? (libxsmm_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v) : (libxsmm_blasint)(handle->blocksifm * handle->ifmblock);
        const float  beta = (handle->avoid_acc_load_bwd ? 0.f : 1.f);
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_smmfunction_reducebatch_addr gemm_br_function;
        int l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
        /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
        gemm_br_function br_gemm_kernel = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*handle->bwd_ofw_rb, handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
        gemm_br_function br_gemm_kernel2 = libxsmm_smmdispatch_reducebatch_addr(handle->ifmblock, handle->bwd_ofh_rb*(handle->bwd_ofw_rb-1), handle->ofmblock, &ldA, &ldB, &ldC, NULL, &beta, &l_flags, NULL);
#define LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
# include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom-rsck_generic.tpl.c"
#undef LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
      } else {
        const libxsmm_blasint ldB = (libxsmm_blasint)(handle->blocksofm * handle->ofmblock);
        const libxsmm_blasint ldA = (libxsmm_blasint)handle->ifmblock;
        const libxsmm_blasint ldC = (libxsmm_blasint)(handle->blocksifm * handle->ifmblock * handle->desc.v);
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_smmfunction gemm_function;
        /* let's do a ifmblock x ofw_rb x ofmblock GEMM :-) or in other words M=nbIfm, N=ofw, K=nbOfm (col-major) */
        gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ifmblock, handle->ofw, handle->ofmblock, &ldA, &ldB, &ldC, NULL, NULL, NULL, NULL);
#define LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
#include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom-rsck_fallback_generic.tpl.c"
#undef LIBXSMM_DNN_TPL_BWD_DIRECT_GENERIC_NHWC_CUSTOM
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}

