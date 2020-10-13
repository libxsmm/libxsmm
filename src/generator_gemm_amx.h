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

#ifndef GENERATOR_GEMM_AMX_H
#define GENERATOR_GEMM_AMX_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_reduceloop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_nloop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const unsigned int                 i_n_blocking);


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_mloop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const unsigned int                 i_m_blocking );


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_reduceloop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_nloop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    const unsigned int                 i_n_blocking,
    const unsigned int                 i_n_done,
    const unsigned int                 i_m_loop_exists);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_mloop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    const unsigned int                 i_m_blocking,
    const unsigned int                 i_m_done,
    const unsigned int                 i_k_unrolled );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_load_C_amx( libxsmm_generated_code*            io_generated_code,
    libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    libxsmm_blocking_info_t*           n_blocking_info,
    libxsmm_blocking_info_t*           m_blocking_info );


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_store_C_amx( libxsmm_generated_code*            io_generated_code,
    libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    libxsmm_blocking_info_t*           n_blocking_info,
    libxsmm_blocking_info_t*           m_blocking_info );

LIBXSMM_API_INTERN
void libxsmm_setup_tile( unsigned int tile_id, unsigned int n_rows, unsigned int n_cols, libxsmm_tile_config *tc);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_setup_fusion_infra( libxsmm_generated_code*            io_generated_code,
                                                    const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                    const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                                    libxsmm_micro_kernel_config*  i_micro_kernel_config );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_setup_stack_frame( libxsmm_generated_code*            io_generated_code,
                                                  const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                  const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                                  libxsmm_micro_kernel_config*        i_micro_kernel_config,
                                                  int                                 m_tiles,
                                                  int                                 n_tiles );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_destroy_stack_frame( libxsmm_generated_code*            io_generated_code,
                                                  const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                  const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                                  const libxsmm_micro_kernel_config*  i_micro_kernel_config );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_kernel( libxsmm_generated_code*        io_generated_code,
                                        const libxsmm_gemm_descriptor* i_xgemm_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_kernel_mloop( libxsmm_generated_code*            io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    libxsmm_blocking_info_t*           n_blocking_info,
    libxsmm_blocking_info_t*           m_blocking_info );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_kernel_nloop( libxsmm_generated_code*            io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    libxsmm_blocking_info_t*           n_blocking_info,
    libxsmm_blocking_info_t*           m_blocking_info );

#endif /* GENERATOR_GEMM_AMX_H */

