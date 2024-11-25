/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_MATELTWISE_REFERENCE_IMPL_H
#define GENERATOR_MATELTWISE_REFERENCE_IMPL_H

LIBXSMM_API_INTERN libxsmm_float16 my_libxsmm_convert_f32_to_f16(float in);
LIBXSMM_API_INTERN float my_libxsmm_convert_f16_to_f32(libxsmm_float16 in);
LIBXSMM_API_INTERN void my_libxsmm_lsfr_i32(unsigned int* rng_state, unsigned int* prng_out, const unsigned int seed_idx);
LIBXSMM_API_INTERN void my_libxsmm_stochastic_convert_fp32_bf8(const float* in, libxsmm_bfloat8* out, unsigned int len, void *rng_state, unsigned int start_seed_idx);
LIBXSMM_API_INTERN float my_libxsmm_convert_bf8_to_f32(libxsmm_bfloat8 in);
LIBXSMM_API_INTERN float my_libxsmm_convert_hf8_to_f32(libxsmm_hfloat8 in);
LIBXSMM_API_INTERN float my_libxsmm_convert_bf16_to_f32(libxsmm_bfloat16 in);
LIBXSMM_API_INTERN libxsmm_bfloat16 my_libxsmm_convert_f32_to_bf16_truncate(float in);
LIBXSMM_API_INTERN libxsmm_bfloat16 my_libxsmm_convert_f32_to_bf16_rnaz(float in);
LIBXSMM_API_INTERN libxsmm_bfloat16 my_libxsmm_convert_f32_to_bf16_rne(float in);
LIBXSMM_API_INTERN libxsmm_bfloat8 my_libxsmm_convert_f32_to_bf8_stochastic(float in, unsigned int seed);
LIBXSMM_API_INTERN libxsmm_bfloat8 my_libxsmm_convert_f32_to_bf8_rne(float in);
LIBXSMM_API_INTERN libxsmm_hfloat8 my_libxsmm_convert_f32_to_hf8_rne(float in);

LIBXSMM_API_INTERN
void libxsmm_reference_unary_elementwise(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc);

LIBXSMM_API_INTERN
void libxsmm_reference_binary_elementwise(libxsmm_meltw_binary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc);

LIBXSMM_API_INTERN
void libxsmm_reference_ternary_elementwise(libxsmm_meltw_ternary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc);

LIBXSMM_API_INTERN
void libxsmm_reference_elementwise(void *param,  const libxsmm_meltw_descriptor *i_mateltwise_desc);

#endif /* GENERATOR_MATELTWISE_REFERENCE_IMPL_H */


