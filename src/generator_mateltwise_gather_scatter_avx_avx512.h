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
#ifndef GENERATOR_MATELTWISE_GATHER_SCATTER_AVX_AVX512_H
#define GENERATOR_MATELTWISE_GATHER_SCATTER_AVX_AVX512_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_unified_vec_move_ext( libxsmm_generated_code* io_generated_code,
                                                const unsigned int      i_vmove_instr,
                                                const unsigned int      i_gp_reg_base,
                                                const unsigned int      i_reg_idx,
                                                const unsigned int      i_scale,
                                                const int               i_displacement,
                                                const char              i_vector_name,
                                                const unsigned int      i_vec_reg_number_0,
                                                const unsigned int      i_use_masking,
                                                const unsigned int      i_mask_reg_number,
                                                const unsigned int      use_mask_move_instr,
                                                const unsigned int      use_m_scalar_loads_stores,
                                                const unsigned int      aux_gpr,
                                                const unsigned int      i_is_store );

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_rows_avx_avx512_mn_loop_unrolled( libxsmm_generated_code*                        io_generated_code,
    unsigned int                                   m_unroll_factor,
    unsigned int                                   n_unroll_factor,
    unsigned int                                   vload_instr,
    unsigned int                                   vstore_instr,
    unsigned int                                   vlen,
    char                                           vname_load,
    char                                           vname_store,
    unsigned int                                   use_m_masking,
    unsigned int                                   ones_mask_reg,
    unsigned int                                   mask_reg,
    unsigned int                                   help_mask_reg,
    unsigned int                                   is_gather,
    unsigned int                                   gp_idx_mat_reg,
    unsigned int                                   gp_reg_mat_reg,
    unsigned int                                   dtype_size_idx_mat,
    unsigned int                                   dtype_size_reg_mat,
    unsigned int                                   ld_idx_mat,
    unsigned int                                   ld_reg_mat );

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_cols_avx_avx512_m_loop( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    unsigned int                                   m_trips_loop,
    unsigned int                                   m_unroll_factor,
    unsigned int                                   peeled_m_trips,
    unsigned int                                   vload_instr,
    unsigned int                                   vstore_instr,
    unsigned int                                   pf_instr,
    unsigned int                                   vlen,
    char                                           vname,
    unsigned int                                   use_m_masking,
    unsigned int                                   use_mask_move_instr,
    unsigned int                                   use_m_scalar_loads_stores,
    unsigned int                                   mask_reg,
    unsigned int                                   pf_dist,
    unsigned int                                   is_gather,
    unsigned int                                   gp_idx_mat_reg,
    unsigned int                                   gp_idx_mat_pf_reg,
    unsigned int                                   gp_reg_mat_reg,
    unsigned int                                   dtype_size_idx_mat,
    unsigned int                                   dtype_size_reg_mat );

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_offs_avx_avx512_mn_loop_unrolled( libxsmm_generated_code*                        io_generated_code,
    unsigned int                                   m_unroll_factor,
    unsigned int                                   n_unroll_factor,
    unsigned int                                   idx_vload_instr,
    unsigned int                                   gp_idx_base_reg,
    unsigned int                                   idx_vlen,
    unsigned int                                   idx_tsize,
    char                                           idx_vname,
    unsigned int                                   idx_mask_reg,
    unsigned int                                   ld_idx,
    unsigned int                                   vload_instr,
    unsigned int                                   vstore_instr,
    unsigned int                                   vlen,
    char                                           vname_load,
    char                                           vname_store,
    unsigned int                                   use_m_masking,
    unsigned int                                   ones_mask_reg,
    unsigned int                                   mask_reg,
    unsigned int                                   help_mask_reg,
    unsigned int                                   is_gather,
    unsigned int                                   gp_idx_mat_reg,
    unsigned int                                   gp_reg_mat_reg,
    unsigned int                                   dtype_size_idx_mat,
    unsigned int                                   dtype_size_reg_mat,
    unsigned int                                   ld_reg_mat );

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_cols_avx_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_rows_scalar_x86_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_offs_scalar_x86_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc );


LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_rows_avx_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_avx_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc );


LIBXSMM_API_INTERN
void libxsmm_generator_gather_scatter_offs_avx_avx512_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc );

#endif /* GENERATOR_MATELTWISE_GATHER_SCATTER_AVX_AVX512_H */

