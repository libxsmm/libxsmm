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
#include "libxsmm_dnn_fullyconnected_forward.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_fwd_custom_f32_f32(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_fwd_custom_bf16_f32(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_f32_f32(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16_emu(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid);

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_fwd_custom_f32_f32(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  typedef libxsmm_smmfunction gemm_function;
  element_input_type alpha = (element_input_type)1;
  element_input_type beta = (element_input_type)0;
  libxsmm_blasint lda = (libxsmm_blasint)handle->ofmblock;
  libxsmm_blasint ldb = (libxsmm_blasint)handle->desc.C;
  libxsmm_blasint ldc = (libxsmm_blasint)handle->desc.K;

  if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
    gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ofmblock, handle->desc.N, handle->desc.C, &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
# include "template/libxsmm_dnn_fullyconnected_st_fwd_custom_generic.tpl.c"
  } else {
    status = LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_fwd_custom_bf16_f32(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef float element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;
  typedef libxsmm_smmfunction gemm_function;
  libxsmm_blasint lda = (libxsmm_blasint)handle->ofmblock;
  libxsmm_blasint ldb = (libxsmm_blasint)handle->desc.C;
  libxsmm_blasint ldc = (libxsmm_blasint)handle->desc.K;
  float alpha = (element_input_type)1;
  float beta = (element_input_type)0;

  if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
    gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ofmblock, handle->desc.N, handle->desc.C, &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
# define LIBXSMM_DNN_FULLYCONNECTED_FWD_BF16_F32
# include "template/libxsmm_dnn_fullyconnected_st_fwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FULLYCONNECTED_FWD_BF16_F32
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
libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_f32_f32(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  libxsmm_smmfunction_reducebatch_strd batchreduce_kernel_beta     = handle->gemm_fwd.xgemm.smrs;
  libxsmm_smmfunction_reducebatch_strd batchreduce_kernel_zerobeta = handle->gemm_fwd2.xgemm.smrs;

#define LIBXSMM_DNN_FC_FWD_USE_AVX512
  if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_NONE
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_NONE
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_BIAS
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_RELU ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_RELU
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_RELU
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_SIGMOID ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_RELU ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_BIAS
#define LIBXSMM_DNN_FC_FWD_FUSE_RELU
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_RELU
#undef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_BIAS
#define LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
#undef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
  } else {
    status = LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }
#undef LIBXSMM_DNN_FC_FWD_USE_AVX512
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16_emu(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;
  libxsmm_bsmmfunction_reducebatch_strd batchreduce_kernel = handle->gemm_fwd.xgemm.bsmrs;
  libxsmm_bmmfunction_reducebatch_strd batchreduce_kernel_zerobeta = handle->gemm_fwd2.xgemm.bmrs;
  libxsmm_bmmfunction_reducebatch_strd batchreduce_kernel_beta = handle->gemm_fwd3.xgemm.bmrs;

  if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_NONE
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_NONE
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_BIAS
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_RELU ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_RELU
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_RELU
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_SIGMOID ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_RELU ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_BIAS
#define LIBXSMM_DNN_FC_FWD_FUSE_RELU
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_RELU
#undef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_BIAS
#define LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
#undef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
  } else {
    status = LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

#if defined(LIBXSMM_INTRINSICS_AVX512_CPX)
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CPX)
libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;
  libxsmm_bsmmfunction_reducebatch_strd batchreduce_kernel = handle->gemm_fwd.xgemm.bsmrs;
  libxsmm_bmmfunction_reducebatch_strd batchreduce_kernel_zerobeta = handle->gemm_fwd2.xgemm.bmrs;
  libxsmm_bmmfunction_reducebatch_strd batchreduce_kernel_beta = handle->gemm_fwd3.xgemm.bmrs;

#define LIBXSMM_DNN_FC_FWD_AVX512_CPX
  if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_NONE
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_NONE
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_BIAS
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_RELU ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_RELU
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_RELU
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_SIGMOID ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_RELU ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_BIAS
#define LIBXSMM_DNN_FC_FWD_FUSE_RELU
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_RELU
#undef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_BIAS
#define LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
#undef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
  } else {
    status = LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }
#undef LIBXSMM_DNN_FC_FWD_AVX512_CPX
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#else
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid)
{
  return libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16_emu( handle, start_thread, tid );
}
#endif

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_fwd_custom(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if all required tensors are bound */
  if (handle->reg_input == 0 || handle->reg_output == 0 ||
      handle->reg_filter == 0                              ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on an AVX512 platform */
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( libxsmm_target_archid >= LIBXSMM_X86_AVX512 ) {
    if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_fullyconnected_st_fwd_custom_f32_f32( handle, start_thread, tid);
    }
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
    else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_CORE ) {
      status = libxsmm_dnn_fullyconnected_st_fwd_custom_bf16_f32( handle, start_thread, tid);
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
      libxsmm_blasint lda = (libxsmm_blasint)handle->ofmblock;
      libxsmm_blasint ldb = (libxsmm_blasint)handle->desc.C;
      libxsmm_blasint ldc = (libxsmm_blasint)handle->desc.K;
      element_input_type beta = (element_input_type)0;
      element_input_type alpha = (element_input_type)1;

      if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
        gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ofmblock, handle->desc.N, handle->desc.C, &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
# include "template/libxsmm_dnn_fullyconnected_st_fwd_custom_generic.tpl.c"
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


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if all required tensors are bound */
  if (handle->reg_input == 0 || handle->reg_output == 0 ||
      handle->reg_filter == 0                              ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }
  if ( ((handle->desc.fuse_ops & LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS ) != 0) && ( handle->reg_bias == 0 ) )  {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }
  if ( ((handle->desc.fuse_ops & LIBXSMM_DNN_FULLYCONNECTED_FUSE_RELU ) != 0) && ( handle->relumask == 0 ) )  {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on an AVX512 platform */
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( libxsmm_target_archid >= LIBXSMM_X86_AVX512 ) {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_f32_f32( handle, start_thread, tid);
    }
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
    else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_CORE && libxsmm_target_archid < LIBXSMM_X86_AVX512_CPX) {
      status = libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16_emu( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_CPX ) {
      status = libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16( handle, start_thread, tid);
    }
#elif defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
    else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_CORE ) {
      status = libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_bf16_bf16_emu( handle, start_thread, tid);
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
      libxsmm_smmfunction_reducebatch_strd batchreduce_kernel_beta     = handle->gemm_fwd.xgemm.smrs;
      libxsmm_smmfunction_reducebatch_strd batchreduce_kernel_zerobeta = handle->gemm_fwd2.xgemm.smrs;

      if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_NONE
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_NONE
      } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_BIAS
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
      } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_RELU ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_RELU
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_RELU
      } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_SIGMOID ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
      } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_RELU ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_BIAS
#define LIBXSMM_DNN_FC_FWD_FUSE_RELU
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_RELU
#undef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
      } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID ) {
#define LIBXSMM_DNN_FC_FWD_FUSE_BIAS
#define LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
# include "template/libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
#undef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
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


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_fwd_nhwc(libxsmm_dnn_fullyconnected* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  LIBXSMM_UNUSED( handle );
  LIBXSMM_UNUSED( start_thread );
  LIBXSMM_UNUSED( tid );
  return status;
}

