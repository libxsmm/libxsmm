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
#ifndef LIBXSMM_UTILS_LPFLT_QUANT_H
#define LIBXSMM_UTILS_LPFLT_QUANT_H

#include "../libxsmm_typedefs.h"

/** these are some quantization definitions, not sure if we want to
    move them into some main part of LIBXSMM */

/* F32 masking defines */
#define LIBXSMM_MASK_SIGN_F32      0x80000000
#define LIBXSMM_MASK_EXP_F32       0x7f800000
#define LIBXSMM_MASK_MANT_F32      0x007fffff
#define LIBXSMM_MASK_ABS_F32       0x7fffffff
#define LIBXSMM_MASK_FULL_F32      0xffffffff
#define LIBXSMM_MANT_SZ_F32        23
#define LIBXSMM_SZ_F32             32

/* DFP16 masking defines */
#define LIBXSMM_MANT_DFP16         15
#define LIBXSMM_RES_DFP16          libxsmm_sexp2_i8i(-(LIBXSMM_MANT_DFP16))

/* Quantization Rounding Defines */
#define LIBXSMM_QUANT_NO_ROUND       80000
#define LIBXSMM_QUANT_BIAS_ROUND     80001
#define LIBXSMM_QUANT_STOCH_ROUND    80002
#define LIBXSMM_QUANT_NEAREST_ROUND  80003
#define LIBXSMM_QUANT_FPHW_ROUND     80004

/** some quantization helper functions,
    TODO: need to be integrated better for all different ways of quantizations */
LIBXSMM_API void libxsmm_quantize_i16( float* in_buffer, short* out_buffer, int length, unsigned char add_shift, unsigned char* scf, int round_mode );
LIBXSMM_API void libxsmm_dequantize_i16( short* in_buffer, float* out_buffer, int length, unsigned char scf );

/** BF16<->FP32 conversion functions */
LIBXSMM_API void libxsmm_truncate_convert_f32_bf16(const float* in, libxsmm_bfloat16* out, size_t length);
LIBXSMM_API void libxsmm_rnaz_convert_fp32_bf16(const float* in, libxsmm_bfloat16* out, size_t length);
LIBXSMM_API void libxsmm_rne_convert_fp32_bf16(const float* in, libxsmm_bfloat16* out, size_t length);
LIBXSMM_API void libxsmm_convert_bf16_f32(const libxsmm_bfloat16* in, float* out, size_t length);
/** FP16<->FP32 conversion functions */
LIBXSMM_API void libxsmm_rne_convert_fp32_f16(const float* in, libxsmm_float16* out, size_t length);
LIBXSMM_API void libxsmm_convert_f16_f32(const libxsmm_float16* in, float* out, size_t length);
/** BF8<->FP32 conversion functions */
LIBXSMM_API void libxsmm_rne_convert_fp32_bf8(const float* in, libxsmm_bfloat8* out, size_t length);
LIBXSMM_API void libxsmm_convert_bf8_f32(const libxsmm_bfloat8* in, float* out, size_t length);
LIBXSMM_API void libxsmm_stochastic_convert_fp32_bf8(const float* in, libxsmm_bfloat8* out, unsigned int length,
  void* rng_state, unsigned int start_seed_idx);
/** HF8<->FP32 conversion functions */
LIBXSMM_API void libxsmm_rne_convert_fp32_hf8(const float* in, libxsmm_hfloat8* out, size_t length);
LIBXSMM_API void libxsmm_convert_hf8_f32(const libxsmm_hfloat8* in, float* out, size_t length);

/** internal api : xoshiro128 lfsr implementation */
LIBXSMM_API_INTERN void libxsmm_lsfr_i32(unsigned int* rng_state, unsigned int* prng_out, unsigned int seed_idx);

#endif /*LIBXSMM_UTILS_LPFLT_QUANT_H*/
