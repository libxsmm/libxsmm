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
#ifndef GENERATOR_GEMM_AMX_MICROKERNEL_H
#define GENERATOR_GEMM_AMX_MICROKERNEL_H

#include "generator_common.h"
#include "generator_gemm_common.h"


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_decompress_loop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    unsigned int                       cnt_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_decompress_loop_amx( libxsmm_generated_code*             io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    unsigned int                       cnt_reg,
    unsigned int                       n_iters);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_fill_array_4_entries(int *array, int v0, int v1, int v2, int v3);

LIBXSMM_API_INTERN
int libxsmm_is_tile_in_last_tilerow(const libxsmm_micro_kernel_config* i_micro_kernel_config, int tile);

LIBXSMM_API_INTERN
void libxsmm_x86_cvtstore_tile_from_I32_to_F32( libxsmm_generated_code*            io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    int                                tile,
    int                                im_offset,
    int                                in_offset,
    int                                n_cols);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_prefetch_tile_in_L2(libxsmm_generated_code*     io_generated_code,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    unsigned int tile_cols,
    unsigned int LD,
    unsigned int base_reg,
    long long    offset);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_prefetch_tile_in_L1(libxsmm_generated_code*     io_generated_code,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    unsigned int tile_cols,
    unsigned int LD,
    unsigned int base_reg,
    long long    offset);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_prefetch_output( libxsmm_generated_code*            io_generated_code,
    unsigned int                       gpr_base,
    unsigned int                       ldc,
    unsigned int                       dtype_size,
    unsigned int                       offset,
    int                                n_cols );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_paired_tilestore( libxsmm_generated_code*            io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    int                                tile0,
    int                                tile1,
    int                                im_offset,
    int                                in_offset,
    int                                n_cols);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_single_tilestore( libxsmm_generated_code*            io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    int                                tile,
    int                                im_offset,
    int                                in_offset,
    int                                n_cols);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_decompress_32x32_A_block(libxsmm_generated_code*     io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    libxsmm_micro_kernel_config*       i_micro_kernel_config,
    long long                          a_offs,
    long long                          a_lookahead_offs,
    long long                          a_lookahead_br_index);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_normT_32x16_bf16_ext_buf(libxsmm_generated_code*     io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    libxsmm_micro_kernel_config*       i_micro_kernel_config_gemm,
    unsigned int                       i_gp_reg_in,
    long long                          i_offset_in,
    long long                          i_offset_out);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_microkernel( libxsmm_generated_code*            io_generated_code,
                                                     libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                     const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                     libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                     const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                     libxsmm_blocking_info_t*           n_blocking_info,
                                                     libxsmm_blocking_info_t*           m_blocking_info,
                                                     long long                          offset_A,
                                                     long long                          offset_B,
                                                     unsigned int                       is_last_k,
                                                     long long                          i_brgemm_loop,
                                                     unsigned int                       fully_unrolled_brloop  );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_kernel_kloop( libxsmm_generated_code*            io_generated_code,
                                                      libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                      const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                      libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                      const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                      libxsmm_blocking_info_t*           n_blocking_info,
                                                      libxsmm_blocking_info_t*           m_blocking_info,
                                                      long long                          A_offs,
                                                      long long                          B_offs,
                                                      unsigned int                       fully_unrolled_brloop );

#endif /* GENERATOR_GEMM_AMX_MICROKERNEL_H */

