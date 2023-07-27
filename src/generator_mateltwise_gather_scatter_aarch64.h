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
void libxsmm_generator_gather_scatter_cols_aarch64_microkernel( libxsmm_generated_code*                        io_generated_code,
                                                                libxsmm_loop_label_tracker*                    io_loop_label_tracker,
                                                                libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
                                                                const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                                                const libxsmm_meltw_descriptor*                i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_rows_aarch64_mn_loop_unrolled( libxsmm_generated_code*                              io_generated_code,
                                                                     libxsmm_mateltwise_gp_reg_mapping*                   i_gp_reg_mapping,
                                                                     const unsigned int                                   i_m_unroll_factor,
                                                                     const unsigned int                                   i_n_unroll_factor,
                                                                     const unsigned int                                   i_idx_vreg_start,
                                                                     const unsigned int                                   i_idx_tsize,
                                                                     const unsigned int                                   i_idx_mask_reg,
                                                                     const unsigned int                                   i_ld_idx_mat,
                                                                     const unsigned int                                   i_gather_instr,
                                                                     const unsigned int                                   i_scatter_instr,
                                                                     const unsigned int                                   i_vlen,
                                                                     const unsigned int                                   i_m_remainder_elements,
                                                                     const unsigned int                                   i_mask_reg,
                                                                     const unsigned int                                   i_mask_reg_full_frac_vlen,
                                                                     const unsigned int                                   i_is_gather,
                                                                     const unsigned int                                   i_gp_idx_mat_reg,
                                                                     const unsigned int                                   i_gp_reg_mat_reg,
                                                                     const unsigned int                                   i_dtype_size_reg_mat,
                                                                     const unsigned int                                   i_ld_reg_mat );

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_rows_aarch64_microkernel( libxsmm_generated_code*                        io_generated_code,
                                                                libxsmm_loop_label_tracker*                    io_loop_label_tracker,
                                                                libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
                                                                const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                                                const libxsmm_meltw_descriptor*                i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_offs_aarch64_mn_loop_unrolled( libxsmm_generated_code*                              io_generated_code,
                                                                     libxsmm_mateltwise_gp_reg_mapping*                   i_gp_reg_mapping,
                                                                     const unsigned int                                   i_m_unroll_factor,
                                                                     const unsigned int                                   i_n_unroll_factor,
                                                                     const unsigned int                                   i_gp_idx_base_reg,
                                                                     const unsigned int                                   i_idx_vlen,
                                                                     const unsigned int                                   i_idx_tsize,
                                                                     const unsigned int                                   i_idx_mask_reg,
                                                                     const unsigned int                                   i_ld_idx,
                                                                     const unsigned int                                   i_gather_instr,
                                                                     const unsigned int                                   i_scatter_instr,
                                                                     const unsigned int                                   i_vlen,
                                                                     const unsigned int                                   i_m_remainder_elements,
                                                                     const unsigned int                                   i_mask_reg,
                                                                     const unsigned int                                   i_mask_reg_full_frac_vlen,
                                                                     const unsigned int                                   i_is_gather,
                                                                     const unsigned int                                   i_gp_idx_mat_reg,
                                                                     const unsigned int                                   i_gp_reg_mat_reg,
                                                                     const unsigned int                                   i_dtype_size_reg_mat,
                                                                     const unsigned int                                   i_ld_reg_mat );

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

