/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_DNN_H
#define LIBXSMM_DNN_H

#include "libxsmm_typedefs.h"

typedef unsigned int libxsmm_dnn_err_t;

/** Define error and warning codes */
#define LIBXSMM_DNN_SUCCESS                             0

#define LIBXSMM_DNN_WARN_FALLBACK                   90000
#define LIBXSMM_DNN_WARN_RNN_SUBOPTIMAL_N_BLOCKING  90001
#define LIBXSMM_DNN_WARN_RNN_SUBOPTIMAL_C_BLOCKING  90002
#define LIBXSMM_DNN_WARN_RNN_SUBOPTIMAL_K_BLOCKING  90003
#define LIBXSMM_DNN_WARN_FC_SUBOPTIMAL_N_BLOCKING   90004
#define LIBXSMM_DNN_WARN_FC_SUBOPTIMAL_C_BLOCKING   90005
#define LIBXSMM_DNN_WARN_FC_SUBOPTIMAL_K_BLOCKING   90006

#define LIBXSMM_DNN_ERR_GENERAL                    100000
#define LIBXSMM_DNN_ERR_CREATE_HANDLE              100001
#define LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE       100002
#define LIBXSMM_DNN_ERR_INVALID_BLOCKING           100003
#define LIBXSMM_DNN_ERR_INVALID_HANDLE             100004
#define LIBXSMM_DNN_ERR_DATA_NOT_BOUND             100005
#define LIBXSMM_DNN_ERR_CREATE_TENSOR              100006
#define LIBXSMM_DNN_ERR_INVALID_TENSOR             100007
#define LIBXSMM_DNN_ERR_MISMATCH_TENSOR            100008
#define LIBXSMM_DNN_ERR_INVALID_HANDLE_TENSOR      100009
#define LIBXSMM_DNN_ERR_INVALID_KIND               100010
#define LIBXSMM_DNN_ERR_INVALID_FORMAT_NCHW        100011
#define LIBXSMM_DNN_ERR_UNSUPPORTED_DST_FORMAT     100012
#define LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT     100013
#define LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE    100014
#define LIBXSMM_DNN_ERR_INVALID_FORMAT_KCRS        100015
#define LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL     100016
#define LIBXSMM_DNN_ERR_CREATE_LAYOUT              100017
#define LIBXSMM_DNN_ERR_INVALID_LAYOUT             100018
#define LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH           100019
#define LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED        100020
#define LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE        100021
#define LIBXSMM_DNN_ERR_INVALID_ALGO               100022
#define LIBXSMM_DNN_ERR_INVALID_PADDING            100023
#define LIBXSMM_DNN_ERR_UNKNOWN_BIAS_TYPE          100024
#define LIBXSMM_DNN_ERR_MISMATCH_BIAS              100025
#define LIBXSMM_DNN_ERR_INVALID_HANDLE_BIAS        100026
#define LIBXSMM_DNN_ERR_TIME_STEPS_TOO_SMALL       100027
#define LIBXSMM_DNN_ERR_CREATE_LAYOUT_ARRAYS       100028
#define LIBXSMM_DNN_ERR_NOT_IMPLEMENTED            100029
#define LIBXSMM_DNN_ERR_FUSEDBN_UNSUPPORTED_ORDER  100030
#define LIBXSMM_DNN_ERR_FUSEDBN_UNSUPPORTED_FUSION 100031
#define LIBXSMM_DNN_ERR_INVALID_FORMAT_FUSEDBN     100032
#define LIBXSMM_DNN_ERR_UNSUPPORTED_POOLING        100033
#define LIBXSMM_DNN_ERR_INVALID_FORMAT_FC          100034
#define LIBXSMM_DNN_ERR_INVALID_RNN_TYPE           100035
#define LIBXSMM_DNN_ERR_RNN_INVALID_SEQ_LEN        100036
#define LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_ORDER  100037
#define LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_FUSION 100038
#define LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION      100039

/** Kinds of supported compute flavor operations. */
typedef enum libxsmm_dnn_compute_kind {
  /** Forward path */
  LIBXSMM_DNN_COMPUTE_KIND_FWD,
  /** Backward path */
  LIBXSMM_DNN_COMPUTE_KIND_BWD,
  /** Updated weights. */
  LIBXSMM_DNN_COMPUTE_KIND_UPD,
  /** Backward and weightupdate combined, useful for RNNs */
  LIBXSMM_DNN_COMPUTE_KIND_BWDUPD,
  /** All routines, need for some init routines. */
  LIBXSMM_DNN_COMPUTE_KIND_ALL
} libxsmm_dnn_compute_kind;

/** these are some quantization definitions, not sure if we want to
    move them into some main part of LIBXSMM */
/* @TODO check position of these declarations and defines */
typedef union LIBXSMM_RETARGETABLE libxsmm_intfloat {
  unsigned int ui;
  float f;
} libxsmm_intfloat;

/* F32 masking defines */
#define LIBXSNN_DNN_MASK_SIGN_F32      0x80000000
#define LIBXSMM_DNN_MASK_EXP_F32       0x7f800000
#define LIBXSMM_DNN_MASK_MANT_F32      0x007fffff
#define LIBXSMM_DNN_MASK_ABS_F32       0x7fffffff
#define LIBXSMM_DNN_MASK_FULL_F32      0xffffffff
#define LIBXSMM_DNN_MANT_SZ_F32        23
#define LIBXSMM_DNN_SZ_F32             32

/* DFP16 masking defines */
#define LIBXSMM_DNN_MANT_DFP16         15
#define LIXSMMM_DNN_RES_DFP16          libxsmm_sexp2_i8i(-(LIBXSMM_DNN_MANT_DFP16))

/* Quantization Rounding Defines */
#define LIBXSMM_DNN_QUANT_NO_ROUND       80000
#define LIBXSMM_DNN_QUANT_BIAS_ROUND     80001
#define LIBXSMM_DNN_QUANT_STOCH_ROUND    80002
#define LIBXSMM_DNN_QUANT_NEAREST_ROUND  80003
#define LIBXSMM_DNN_QUANT_FPHW_ROUND     80004

/** get string of error code */
LIBXSMM_API const char* libxsmm_dnn_get_error(libxsmm_dnn_err_t code);
LIBXSMM_API size_t libxsmm_dnn_typesize(libxsmm_dnn_datatype datatype);
LIBXSMM_API size_t libxsmm_dnn_get_simd_width(libxsmm_dnn_datatype datatype);

/** some quantization helper functions,
    @TODO need to be integrated better for all different ways of quantizations */
LIBXSMM_API void libxsmm_dnn_quantize( float* in_buffer, short* out_buffer, int length, unsigned char add_shift, unsigned char* scf, int round_mode );
LIBXSMM_API void libxsmm_dnn_quantize_act( float* in_buffer, short* out_buffer, unsigned int N, unsigned int C, unsigned int H, unsigned int W, unsigned int cblk_f32, unsigned int cblk_i16, unsigned int lp_blk, unsigned char add_shift, unsigned char* scf, int round_mode );
LIBXSMM_API void libxsmm_dnn_quantize_fil( float* in_buffer, short* out_buffer, unsigned int K, unsigned int C, unsigned int R, unsigned int S, unsigned int cblk_f32, unsigned int cblk_i16, unsigned int kblk_f32, unsigned int kblk_i16, unsigned int lp_blk, unsigned char add_shift, unsigned char* scf, int round_mode );
LIBXSMM_API void libxsmm_dnn_dequantize( short* in_buffer, float* out_buffer, int length, unsigned char scf );

/** some BF16<->FP32 conversion functions
    @TODO we need to find a final place for those */
LIBXSMM_API void libxsmm_truncate_convert_f32_bf16(const float* in, libxsmm_bfloat16* out, unsigned int length);
LIBXSMM_API void libxsmm_rnaz_convert_fp32_bf16(const float* in, libxsmm_bfloat16* out, unsigned int len);
LIBXSMM_API void libxsmm_rne_convert_fp32_bf16(const float* in, libxsmm_bfloat16* out, unsigned int len);
LIBXSMM_API void libxsmm_convert_bf16_f32(const libxsmm_bfloat16* in, float* out, unsigned int length);
LIBXSMM_API void libxsmm_rne_convert_fp32_bf8(const float* in, libxsmm_bfloat8* out, unsigned int len);
LIBXSMM_API void libxsmm_convert_bf8_f32(const libxsmm_bfloat8* in, float* out, unsigned int length);
LIBXSMM_API void libxsmm_stochastic_convert_fp32_bf8(const float* in, libxsmm_bfloat8* out, unsigned int length);

#endif /*LIBXSMM_DNN_H*/

