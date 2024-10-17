/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
*               Friedrich Schiller University Jena - All rights reserved.     *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.), Antonio Noack (FSU Jena)
******************************************************************************/
#include "generator_mateltwise_sse_avx_avx512.h"
#include "generator_mateltwise_transform_common.h"
#include "generator_mateltwise_transform_sse.h"
#include "generator_mateltwise_transform_avx.h"
#include "generator_mateltwise_transform_avx512.h"
#include "generator_mateltwise_transform_aarch64_asimd.h"
#include "generator_mateltwise_transform_aarch64_sve.h"

LIBXSMM_API_INTERN
void libxsmm_generator_transform_x86_microkernel( libxsmm_generated_code*                        io_generated_code,
                                                  libxsmm_loop_label_tracker*                    io_loop_label_tracker,
                                                  libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
                                                  const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                                  const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {
  if ( ((io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) &&  (io_generated_code->arch < LIBXSMM_X86_ALLFEAT))
       || ((io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) && ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8_PAD) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8_TO_VNNI8T) )
         && ( ( LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
                LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ||
              ( LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
                LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) ) ||
              ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
                LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision( i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT ) )  )) ) {
    libxsmm_generator_transform_avx512_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc );
  } else if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ) {
    /* we need to re-run some setting for KNx */
    /* TODO: find a better solution for KNx */
    const unsigned int l_save_arch = io_generated_code->arch;
    libxsmm_mateltwise_kernel_config l_new_micro_kernel_config = *i_micro_kernel_config;
    libxsmm_meltw_descriptor l_new_mateltwise_desc = *i_mateltwise_desc;
    io_generated_code->arch = LIBXSMM_X86_AVX2;
    libxsmm_generator_mateltwise_update_micro_kernel_config_dtype_aluinstr( io_generated_code, &l_new_micro_kernel_config, &l_new_mateltwise_desc);
    libxsmm_generator_transform_avx_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, &l_new_micro_kernel_config, &l_new_mateltwise_desc );
    io_generated_code->arch = l_save_arch;
  } else if ( io_generated_code->arch >= LIBXSMM_X86_AVX ) {
    libxsmm_generator_transform_avx_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc );
  } else if ( (io_generated_code->arch >= LIBXSMM_X86_GENERIC) /*&& (io_generated_code->arch < LIBXSMM_X86_AVX)*/) {
    libxsmm_generator_transform_sse_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc );
  } else {
    /* This should not happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_transform_aarch64_microkernel( libxsmm_generated_code*                        io_generated_code,
                                                      libxsmm_loop_label_tracker*                    io_loop_label_tracker,
                                                      libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
                                                      const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                                      const libxsmm_meltw_descriptor*                i_mateltwise_desc ) {
  if ( (io_generated_code->arch >= LIBXSMM_AARCH64_V81) && (io_generated_code->arch < LIBXSMM_AARCH64_SVE128) ) {
    libxsmm_generator_transform_aarch64_asimd_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc );
  } else if ( (io_generated_code->arch >= LIBXSMM_AARCH64_SVE128) && (io_generated_code->arch < LIBXSMM_AARCH64_ALLFEAT) ) {
    libxsmm_generator_transform_aarch64_sve_microkernel( io_generated_code, io_loop_label_tracker, i_gp_reg_mapping, i_micro_kernel_config, i_mateltwise_desc );
  } else {
    /* This should not happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
}

