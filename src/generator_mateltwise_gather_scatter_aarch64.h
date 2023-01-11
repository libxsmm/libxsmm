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

#ifndef GENERATOR_MATELTWISE_GATHER_SCATTER_AARCH64_H
#define GENERATOR_MATELTWISE_GATHER_SCATTER_AARCH64_H

#include "generator_common.h"


LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_rows_aarch64_mn_loop_unrolled( libxsmm_generated_code*                        io_generated_code,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    unsigned int                                   m_unroll_factor,
    unsigned int                                   n_unroll_factor,
    unsigned int                                   idx_vreg_start,
    unsigned int                                   idx_tsize,
    unsigned int                                   idx_mask_reg,
    unsigned int                                   ld_idx_mat,
    unsigned int                                   gather_instr,
    unsigned int                                   scatter_instr,
    unsigned int                                   vlen,
    unsigned int                                   m_remainder_elements,
    unsigned int                                   mask_reg,
    unsigned int                                   mask_reg_full_frac_vlen,
    unsigned int                                   is_gather,
    unsigned int                                   gp_idx_mat_reg,
    unsigned int                                   gp_reg_mat_reg,
    unsigned int                                   dtype_size_reg_mat,
    unsigned int                                   ld_reg_mat );

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_rows_aarch64_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_offs_aarch64_mn_loop_unrolled( libxsmm_generated_code*                        io_generated_code,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    unsigned int                                   m_unroll_factor,
    unsigned int                                   n_unroll_factor,
    unsigned int                                   gp_idx_base_reg,
    unsigned int                                   idx_vlen,
    unsigned int                                   idx_tsize,
    unsigned int                                   idx_mask_reg,
    unsigned int                                   ld_idx,
    unsigned int                                   gather_instr,
    unsigned int                                   scatter_instr,
    unsigned int                                   vlen,
    unsigned int                                   m_remainder_elements,
    unsigned int                                   mask_reg,
    unsigned int                                   mask_reg_full_frac_vlen,
    unsigned int                                   is_gather,
    unsigned int                                   gp_idx_mat_reg,
    unsigned int                                   gp_reg_mat_reg,
    unsigned int                                   dtype_size_reg_mat,
    unsigned int                                   ld_reg_mat );

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_offs_aarch64_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_aarch64_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc );

#endif /* GENERATOR_MATELTWISE_GATHER_SCATTER_AARCH64_H */

