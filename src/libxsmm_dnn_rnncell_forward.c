/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Kunal Banerjee (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_rnncell_forward.h"
#include "libxsmm_dnn_elementwise.h"
#include "libxsmm_main.h"


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_ck_f32_f32(libxsmm_dnn_rnncell* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_ck_bf16_bf16(libxsmm_dnn_rnncell* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_ck_bf16_bf16_emu(libxsmm_dnn_rnncell* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_ck_bf16_bf16_amx(libxsmm_dnn_rnncell* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_ncnc_kcck_f32_f32(libxsmm_dnn_rnncell* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_kcck_f32_f32(libxsmm_dnn_rnncell* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_kcck_bf16_bf16(libxsmm_dnn_rnncell* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_kcck_bf16_bf16_emu(libxsmm_dnn_rnncell* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_kcck_bf16_bf16_amx(libxsmm_dnn_rnncell* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_ncnc_kcck_bf16_bf16_amx(libxsmm_dnn_rnncell* handle, int start_thread, int tid);


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_ck_f32_f32(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
# define LIBXSMM_DNN_RNN_RELU_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_nc_ck_generic.tpl.c"
# undef LIBXSMM_DNN_RNN_RELU_FWD
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
# define LIBXSMM_DNN_RNN_SIGMOID_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_nc_ck_generic.tpl.c"
# undef LIBXSMM_DNN_RNN_SIGMOID_FWD
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
# define LIBXSMM_DNN_RNN_TANH_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_nc_ck_generic.tpl.c"
# undef LIBXSMM_DNN_RNN_TANH_FWD
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
#define LIBXSMM_RNN_CELL_AVX512
# include "template/libxsmm_dnn_rnncell_st_lstm_fwd_nc_ck_generic.tpl.c"
#undef LIBXSMM_RNN_CELL_AVX512
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
# include "template/libxsmm_dnn_rnncell_st_gru_fwd_nc_ck_generic.tpl.c"
  } else {
    /* should not happen */
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_ck_bf16_bf16_emu(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__, __AVX512BW__, __AVX512DQ__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;

  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
#define LIBXSMM_RNN_CELL_AVX512
# include "template/libxsmm_dnn_rnncell_st_lstm_fwd_nc_ck_generic_bf16.tpl.c"
#undef LIBXSMM_RNN_CELL_AVX512
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }

# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

#if defined(LIBXSMM_INTRINSICS_AVX512_CPX)
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CPX)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_ck_bf16_bf16(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__, __AVX512BW__, __AVX512DQ__, __AVX512BF16__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;

#define LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
#define LIBXSMM_RNN_CELL_AVX512
# include "template/libxsmm_dnn_rnncell_st_lstm_fwd_nc_ck_generic_bf16.tpl.c"
#undef LIBXSMM_RNN_CELL_AVX512
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }

# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#undef LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#else
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_ck_bf16_bf16(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  return libxsmm_dnn_rnncell_st_fwd_nc_ck_bf16_bf16_emu(handle, start_thread, tid);
}
#endif

#if defined(LIBXSMM_INTRINSICS_AVX512_CPX)
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CPX)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_ck_bf16_bf16_amx(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__, __AVX512BW__, __AVX512DQ__, __AVX512BF16__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;

#define LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
#define LIBXSMM_RNN_CELL_AVX512
# include "template/libxsmm_dnn_rnncell_st_lstm_fwd_nc_ck_generic_bf16_amx.tpl.c"
#undef LIBXSMM_RNN_CELL_AVX512
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }

# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#undef LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#else
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_ck_bf16_bf16_amx(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__, __AVX512BW__, __AVX512DQ__ */
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;

  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
#define LIBXSMM_RNN_CELL_AVX512
# include "template/libxsmm_dnn_rnncell_st_lstm_fwd_nc_ck_generic_bf16_amx.tpl.c"
#undef LIBXSMM_RNN_CELL_AVX512
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }

# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#endif

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_ncnc_kcck_f32_f32(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
# define LIBXSMM_DNN_RNN_RELU_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_ncnc_kcck.tpl.c"
# undef LIBXSMM_DNN_RNN_RELU_FWD
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
# define LIBXSMM_DNN_RNN_SIGMOID_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_ncnc_kcck.tpl.c"
# undef LIBXSMM_DNN_RNN_SIGMOID_FWD
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
# define LIBXSMM_DNN_RNN_TANH_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_ncnc_kcck.tpl.c"
# undef LIBXSMM_DNN_RNN_TANH_FWD
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_kcck_f32_f32(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
# define LIBXSMM_DNN_RNN_RELU_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_nc_kcck.tpl.c"
# undef LIBXSMM_DNN_RNN_RELU_FWD
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
# define LIBXSMM_DNN_RNN_SIGMOID_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_nc_kcck.tpl.c"
# undef LIBXSMM_DNN_RNN_SIGMOID_FWD
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
# define LIBXSMM_DNN_RNN_TANH_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_nc_kcck.tpl.c"
# undef LIBXSMM_DNN_RNN_TANH_FWD
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
#define LIBXSMM_RNN_CELL_AVX512
# include "template/libxsmm_dnn_rnncell_st_lstm_fwd_nc_kcck.tpl.c"
#undef LIBXSMM_RNN_CELL_AVX512
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
# include "template/libxsmm_dnn_rnncell_st_gru_fwd_nc_kcck.tpl.c"
  } else {
    /* should not happen */
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_kcck_bf16_bf16_emu(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;

  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
#define LIBXSMM_RNN_CELL_AVX512
# include "template/libxsmm_dnn_rnncell_st_lstm_fwd_nc_kcck_bf16.tpl.c"
#undef LIBXSMM_RNN_CELL_AVX512
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }

# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

#if defined(LIBXSMM_INTRINSICS_AVX512_CPX)
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CPX)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_kcck_bf16_bf16(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;

#define LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
#define LIBXSMM_RNN_CELL_AVX512
# include "template/libxsmm_dnn_rnncell_st_lstm_fwd_nc_kcck_bf16.tpl.c"
#undef LIBXSMM_RNN_CELL_AVX512
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }

# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#undef LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#else
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_kcck_bf16_bf16(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  return libxsmm_dnn_rnncell_st_fwd_nc_kcck_bf16_bf16_emu(handle, start_thread, tid);
}
#endif

#if defined(LIBXSMM_INTRINSICS_AVX512_CPX)
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CPX)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_ncnc_kcck_bf16_bf16_amx(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;

#define LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
#define LIBXSMM_RNN_CELL_AVX512
# include "template/libxsmm_dnn_rnncell_st_lstm_fwd_ncnc_kcck_bf16_amx.tpl.c"
#undef LIBXSMM_RNN_CELL_AVX512
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }

# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#undef LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#else
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_ncnc_kcck_bf16_bf16_amx(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;

  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
#define LIBXSMM_RNN_CELL_AVX512
# include "template/libxsmm_dnn_rnncell_st_lstm_fwd_ncnc_kcck_bf16_amx.tpl.c"
#undef LIBXSMM_RNN_CELL_AVX512
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }

# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#endif

#if defined(LIBXSMM_INTRINSICS_AVX512_CPX)
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CPX)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_kcck_bf16_bf16_amx(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;

#define LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
#define LIBXSMM_RNN_CELL_AVX512
# include "template/libxsmm_dnn_rnncell_st_lstm_fwd_nc_kcck_bf16_amx.tpl.c"
#undef LIBXSMM_RNN_CELL_AVX512
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }

# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#undef LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#else
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_kcck_bf16_bf16_amx(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__ */
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;

  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
#define LIBXSMM_RNN_CELL_AVX512
# include "template/libxsmm_dnn_rnncell_st_lstm_fwd_nc_kcck_bf16_amx.tpl.c"
#undef LIBXSMM_RNN_CELL_AVX512
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }

# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#endif

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_ck(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
#if 0
  if (handle->? == 0 ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }
#endif

  /* check if we are on AVX512 */
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( (libxsmm_target_archid >= LIBXSMM_X86_AVX512) && (libxsmm_target_archid <= LIBXSMM_X86_ALLFEAT) ) {
    if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_rnncell_st_fwd_nc_ck_f32_f32( handle, start_thread, tid);
    }
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
    else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_CORE && libxsmm_target_archid < LIBXSMM_X86_AVX512_CPX ) {
      status = libxsmm_dnn_rnncell_st_fwd_nc_ck_bf16_bf16_emu( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_CPX && libxsmm_target_archid < LIBXSMM_X86_AVX512_SPR ) {
      status = libxsmm_dnn_rnncell_st_fwd_nc_ck_bf16_bf16( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_SPR ) {
      status = libxsmm_dnn_rnncell_st_fwd_nc_ck_bf16_bf16_amx( handle, start_thread, tid);
    }
#elif defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
    else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_CORE && libxsmm_target_archid < LIBXSMM_X86_AVX512_CPX ) {
      status = libxsmm_dnn_rnncell_st_fwd_nc_ck_bf16_bf16_emu( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_CPX && libxsmm_target_archid < LIBXSMM_X86_AVX512_SPR ) {
      status = libxsmm_dnn_rnncell_st_fwd_nc_ck_bf16_bf16( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_SPR ) {
      status = libxsmm_dnn_rnncell_st_fwd_nc_ck_bf16_bf16_amx( handle, start_thread, tid);
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
      if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
#define LIBXSMM_DNN_RNN_RELU_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_nc_ck_generic.tpl.c"
#undef LIBXSMM_DNN_RNN_RELU_FWD
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
#define LIBXSMM_DNN_RNN_SIGMOID_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_nc_ck_generic.tpl.c"
#undef LIBXSMM_DNN_RNN_SIGMOID_FWD
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
#define LIBXSMM_DNN_RNN_TANH_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_nc_ck_generic.tpl.c"
#undef LIBXSMM_DNN_RNN_TANH_FWD
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
# include "template/libxsmm_dnn_rnncell_st_lstm_fwd_nc_ck_generic.tpl.c"
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
# include "template/libxsmm_dnn_rnncell_st_gru_fwd_nc_ck_generic.tpl.c"
      } else {
        /* should not happen */
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_ncnc_kcck(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
#if 0
  if (handle->? == 0 ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }
#endif

  /* check if we are on AVX512 */
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
  if ( (libxsmm_target_archid >= LIBXSMM_X86_AVX512) && (libxsmm_target_archid <= LIBXSMM_X86_ALLFEAT) ) {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_rnncell_st_fwd_ncnc_kcck_f32_f32( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_SPR ) {
      status = libxsmm_dnn_rnncell_st_fwd_ncnc_kcck_bf16_bf16_amx( handle, start_thread, tid);
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#elif defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( (libxsmm_target_archid >= LIBXSMM_X86_AVX512) && (libxsmm_target_archid <= LIBXSMM_X86_ALLFEAT) ) {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_rnncell_st_fwd_ncnc_kcck_f32_f32( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_SPR ) {
      status = libxsmm_dnn_rnncell_st_fwd_ncnc_kcck_bf16_bf16_amx( handle, start_thread, tid);
    } else {
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
      if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
#define LIBXSMM_DNN_RNN_RELU_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_ncnc_kcck.tpl.c"
#undef LIBXSMM_DNN_RNN_RELU_FWD
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
#define LIBXSMM_DNN_RNN_SIGMOID_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_ncnc_kcck.tpl.c"
#undef LIBXSMM_DNN_RNN_SIGMOID_FWD
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
#define LIBXSMM_DNN_RNN_TANH_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_ncnc_kcck.tpl.c"
#undef LIBXSMM_DNN_RNN_TANH_FWD
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
        status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
        status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
      } else {
        /* should not happen */
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_kcck(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
#if 0
  if (handle->? == 0 ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }
#endif

  /* check if we are on AVX512 */
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( (libxsmm_target_archid >= LIBXSMM_X86_AVX512) && (libxsmm_target_archid <= LIBXSMM_X86_ALLFEAT) ) {
    if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_rnncell_st_fwd_nc_kcck_f32_f32( handle, start_thread, tid);
    }
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
    else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_CORE && libxsmm_target_archid < LIBXSMM_X86_AVX512_CPX ) {
      status = libxsmm_dnn_rnncell_st_fwd_nc_kcck_bf16_bf16_emu( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_CPX && libxsmm_target_archid < LIBXSMM_X86_AVX512_SPR ) {
      status = libxsmm_dnn_rnncell_st_fwd_nc_kcck_bf16_bf16( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_SPR ) {
      status = libxsmm_dnn_rnncell_st_fwd_nc_kcck_bf16_bf16_amx( handle, start_thread, tid);
    }
#elif defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
    else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_CORE && libxsmm_target_archid < LIBXSMM_X86_AVX512_SPR) {
      status = libxsmm_dnn_rnncell_st_fwd_nc_kcck_bf16_bf16_emu( handle, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_SPR ) {
      status = libxsmm_dnn_rnncell_st_fwd_nc_kcck_bf16_bf16_amx( handle, start_thread, tid);
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
      if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
#define LIBXSMM_DNN_RNN_RELU_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_nc_kcck.tpl.c"
#undef LIBXSMM_DNN_RNN_RELU_FWD
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
#define LIBXSMM_DNN_RNN_SIGMOID_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_nc_kcck.tpl.c"
#undef LIBXSMM_DNN_RNN_SIGMOID_FWD
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
#define LIBXSMM_DNN_RNN_TANH_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_nc_kcck.tpl.c"
#undef LIBXSMM_DNN_RNN_TANH_FWD
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
# include "template/libxsmm_dnn_rnncell_st_lstm_fwd_nc_kcck.tpl.c"
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
# include "template/libxsmm_dnn_rnncell_st_gru_fwd_nc_kcck.tpl.c"
      } else {
        /* should not happen */
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}
