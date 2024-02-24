/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_GEMM_AMX_H
#define GENERATOR_GEMM_AMX_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_brgemm_amx_set_gp_reg_a( libxsmm_generated_code*             io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    libxsmm_micro_kernel_config*       i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    unsigned int                       i_unrolled_index );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_dequant_loop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    unsigned int                       cnt_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_dequant_loop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    unsigned int                       cnt_reg,
    unsigned int                       n_iters);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_decompress_i4_vreg ( libxsmm_generated_code*            io_generated_code,
                                                                    const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                    const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                    unsigned int                       i_zpt_vreg,
                                                                    unsigned int                       io_vreg0,
                                                                    unsigned int                       o_vreg1 );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_decompress_KxM_i4_tensor( libxsmm_generated_code*            io_generated_code,
                                                                         libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                         const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                         unsigned int                       i_m_tiles,
                                                                         unsigned int                       i_K,
                                                                         unsigned int                       i_ldi,
                                                                         unsigned int                       i_ldo,
                                                                         unsigned int                       i_gp_reg,
                                                                         unsigned int                       o_gp_reg );

LIBXSMM_API_INTERN
void libxsmm_get_tileinfo( unsigned int tile_id, unsigned int *n_rows, unsigned int *n_cols, libxsmm_tile_config *tc);

LIBXSMM_API_INTERN void libxsmm_generator_gemm_setup_f8_ABC_tensors_to_stack_for_amx(  libxsmm_generated_code*        io_generated_code,
                                                                                        libxsmm_loop_label_tracker*    io_loop_label_tracker,
                                                                                        libxsmm_gp_reg_mapping*        i_gp_reg_mapping,
                                                                                        libxsmm_micro_kernel_config*   i_micro_kernel_config,
                                                                                        libxsmm_gemm_descriptor*       i_xgemm_desc,
                                                                                        const libxsmm_gemm_descriptor* i_xgemm_desc_orig,
                                                                                        libxsmm_datatype               i_in_dtype );

LIBXSMM_API_INTERN void libxsmm_generator_gemm_emit_f8_eltwise_fusion(   libxsmm_generated_code*        io_generated_code,
                                                                          libxsmm_loop_label_tracker*    io_loop_label_tracker,
                                                                          libxsmm_micro_kernel_config*   i_micro_kernel_config,
                                                                          libxsmm_gemm_descriptor*       i_xgemm_desc,
                                                                          const libxsmm_gemm_descriptor* i_xgemm_desc_orig,
                                                                          unsigned int                   i_defer_c_vnni_format,
                                                                          unsigned int                   i_defer_relu_bitmask_compute,
                                                                          libxsmm_datatype               i_dtype );

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
void libxsmm_generator_gemm_amx_setup_masking_infra( libxsmm_generated_code* io_generated_code, libxsmm_micro_kernel_config*  i_micro_kernel_config );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_setup_fusion_infra( libxsmm_generated_code*            io_generated_code,
                                                    const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                    const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                                    libxsmm_micro_kernel_config*  i_micro_kernel_config );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_init_micro_kernel_config_tileblocking(libxsmm_gemm_descriptor*      i_xgemm_desc,
    libxsmm_micro_kernel_config*  i_micro_kernel_config,
    libxsmm_blocking_info_t*      m_blocking_info,
    libxsmm_blocking_info_t*      n_blocking_info,
    libxsmm_tile_config*          tile_config );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_adjust_m_advancement( libxsmm_generated_code* io_generated_code,
    libxsmm_loop_label_tracker*         io_loop_label_tracker,
    const libxsmm_gemm_descriptor*      i_xgemm_desc,
    const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
    const libxsmm_micro_kernel_config*  i_micro_kernel_config,
    libxsmm_blasint                     i_m_adjustment );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_adjust_n_advancement( libxsmm_generated_code* io_generated_code,
    libxsmm_loop_label_tracker*         io_loop_label_tracker,
    const libxsmm_gemm_descriptor*      i_xgemm_desc,
    const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
    const libxsmm_micro_kernel_config*  i_micro_kernel_config,
    libxsmm_blasint                     i_n_adjustment );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_kernel_wrapper( libxsmm_generated_code*        io_generated_code,
                                        const libxsmm_gemm_descriptor* i_xgemm_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_kernel( libxsmm_generated_code*            io_generated_code,
                                                                           libxsmm_loop_label_tracker*    io_loop_label_tracker,
                                                                           libxsmm_gp_reg_mapping*  i_gp_reg_mapping,
                                                                           const libxsmm_gemm_descriptor* i_xgemm_desc_const );

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

