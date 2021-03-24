/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include "generator_mateltwise_transform_common.h"
#include "generator_mateltwise_transform_sse.h"
#include "generator_mateltwise_transform_avx.h"
#include "generator_mateltwise_transform_avx512.h"

LIBXSMM_API_INTERN
void libxsmm_generator_transform_microkernel( libxsmm_generated_code*                        io_generated_code,
                                              libxsmm_loop_label_tracker*                    io_loop_label_tracker,
                                              libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
                                              const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                              const libxsmm_meltw_descriptor*                i_mateltwise_desc,
                                              const unsigned char                            i_called_from_unary ) {
  libxsmm_meltw_descriptor l_mateltwise_desc_mod = *i_mateltwise_desc;

  /* @TODO clean this up later */
  if ( i_called_from_unary != 0 ) {
    l_mateltwise_desc_mod.operation = LIBXSMM_MELTW_OPERATION_TRANSFORM;
    if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI ) {
      l_mateltwise_desc_mod.flags = LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_VNNI;
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT ) {
      l_mateltwise_desc_mod.flags = LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_NORMT;
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI_TO_VNNIT ) {
      l_mateltwise_desc_mod.flags = LIBXSMM_MELTW_FLAG_TRANSFORM_VNNI_TO_VNNIT;
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNIT ) {
      l_mateltwise_desc_mod.flags = LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_VNNIT;
    } else if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI_PAD ) {
      l_mateltwise_desc_mod.flags = LIBXSMM_MELTW_FLAG_TRANSFORM_NORM_TO_VNNI_PAD;
    }
  }

  if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_CORE) &&  (io_generated_code->arch < LIBXSMM_X86_ALLFEAT) ) {
    libxsmm_generator_transform_avx512_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, &l_mateltwise_desc_mod, i_called_from_unary );
  } else if ( (io_generated_code->arch >= LIBXSMM_X86_AVX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_CORE) ) {
    libxsmm_generator_transform_avx_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, &l_mateltwise_desc_mod, i_called_from_unary );
  } else if ( (io_generated_code->arch >= LIBXSMM_X86_GENERIC) && (io_generated_code->arch < LIBXSMM_X86_AVX) ) {
    libxsmm_generator_transform_sse_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, &l_mateltwise_desc_mod, i_called_from_unary );
  } else {
    /* This should not happen  */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
}

