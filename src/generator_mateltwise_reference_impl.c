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
#include "generator_mateltwise_common.h"
#include "generator_common.h"
#include "generator_mateltwise_reference_impl.h"

void libxsmm_reference_elementwise(libxsmm_meltw_unary_param *param,  const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  if (1) {
    int i, j;
    float in, out;
    for ( j = 0; j < i_mateltwise_desc->n; ++j ) {
      for ( i = 0; i < i_mateltwise_desc->m; ++i ) {
        libxsmm_bfloat16* bf16_in = (libxsmm_bfloat16*)(param->in.primary);
        libxsmm_bfloat16* bf16_out = (libxsmm_bfloat16*)(param->out.primary);
        in = libxsmm_convert_bf16_to_f32(bf16_in[(j*i_mateltwise_desc->ldi) + i]);
        out = LIBXSMM_EXPF(in);
        bf16_out[(j*i_mateltwise_desc->ldo) + i] = libxsmm_convert_f32_to_bf16_rne(out);
      }
    }
  } else {
    int i, j;
    float *in, *out;
    in = (float*)(param->in.primary);
    out = (float*)(param->out.primary);

    for ( j = 0; j < i_mateltwise_desc->n; ++j ) {
      for ( i = 0; i < i_mateltwise_desc->m; ++i ) {
        float in_value = in[(j*i_mateltwise_desc->ldi) + i];
        float out_value = LIBXSMM_EXPF(in_value);
        out[(j*i_mateltwise_desc->ldo) + i] = out_value;
      }
    }
  }
  return;
}
