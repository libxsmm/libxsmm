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
#ifndef GENERATOR_MATELTWISE_REDUCE_AVX_AVX512_H
#define GENERATOR_MATELTWISE_REDUCE_AVX_AVX512_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_reduce_set_lp_vlen_vname_vmove_x86( libxsmm_generated_code*                        io_generated_code,
                                                           const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                                           const libxsmm_meltw_descriptor*                i_mateltwise_desc,
                                                           unsigned int*                                  io_vlen,
                                                           char*                                          io_vname_in,
                                                           char*                                          io_vname_out,
                                                           unsigned int*                                  io_vmove_instruction_in,
                                                           unsigned int*                                  io_vmove_instruction_out );

LIBXSMM_API_INTERN
void libxsmm_generator_reduce_cols_ncnc_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_reduce_cols_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_reduce_rows_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_reduce_cols_index_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc );

#endif /* GENERATOR_MATELTWISE_REDUCE_AVX_AVX512_H */
