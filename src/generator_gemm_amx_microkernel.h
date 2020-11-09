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

#ifndef GENERATOR_GEMM_AMX_MICROKERNEL_H
#define GENERATOR_GEMM_AMX_MICROKERNEL_H

#include "generator_common.h"
#include "generator_gemm_common.h"
#include "libxsmm_main.h"

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
void libxsmm_generator_gemm_tanh_ps_rational_78_avx512( libxsmm_generated_code*                        io_generated_code,
    const libxsmm_micro_kernel_config*             i_micro_kernel_config,
    const unsigned int                             i_vec_x,
    const unsigned int                             i_vec_x2,
    const unsigned int                             i_vec_nom,
    const unsigned int                             i_vec_denom,
    const unsigned int                             i_mask_hi,
    const unsigned int                             i_mask_lo,
    const unsigned int                             i_vec_c0,
    const unsigned int                             i_vec_c1,
    const unsigned int                             i_vec_c2,
    const unsigned int                             i_vec_c3,
    const unsigned int                             i_vec_c1_d,
    const unsigned int                             i_vec_c2_d,
    const unsigned int                             i_vec_c3_d,
    const unsigned int                             i_vec_hi_bound,
    const unsigned int                             i_vec_lo_bound,
    const unsigned int                             i_vec_ones,
    const unsigned int                             i_vec_neg_ones);

LIBXSMM_API_INTERN
void fill_array_4_entries(int *array, int v0, int v1, int v2, int v3);

LIBXSMM_API_INTERN
void prefetch_tile_in_L2(libxsmm_generated_code*     io_generated_code,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    unsigned int tile_cols,
    unsigned int LD,
    unsigned int base_reg,
    unsigned int offset);

LIBXSMM_API_INTERN
void paired_tilestore( libxsmm_generated_code*            io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    int                                tile0,
    int                                tile1,
    int                                im_offset,
    int                                in_offset,
    int                                n_cols);

LIBXSMM_API_INTERN
void single_tilestore( libxsmm_generated_code*            io_generated_code,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    const libxsmm_micro_kernel_config* i_micro_kernel_config,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    int                                tile,
    int                                im_offset,
    int                                in_offset,
    int                                n_cols);

LIBXSMM_API_INTERN
void decompress_32x32_A_block(libxsmm_generated_code*     io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
    libxsmm_micro_kernel_config*       i_micro_kernel_config,
    int                                a_offs,
    unsigned int                       a_lookahead_offs);

LIBXSMM_API_INTERN
void normT_32x16_bf16_ext_buf(libxsmm_generated_code*     io_generated_code,
    libxsmm_loop_label_tracker*        io_loop_label_tracker,
    const libxsmm_gemm_descriptor*     i_xgemm_desc,
    libxsmm_micro_kernel_config*       i_micro_kernel_config_gemm,
    unsigned int                       i_gp_reg_in,
    unsigned int                       i_offset_in,
    unsigned int                       i_offset_out);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_microkernel( libxsmm_generated_code*            io_generated_code,
                                                     libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                     const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                     libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                     const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                     libxsmm_blocking_info_t*           n_blocking_info,
                                                     libxsmm_blocking_info_t*           m_blocking_info,
                                                     unsigned int                       offset_A,
                                                     unsigned int                       offset_B,
                                                     unsigned int                       is_last_k,
                                                     int                                i_brgemm_loop,
                                                     unsigned int                       fully_unrolled_brloop  );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_amx_kernel_kloop( libxsmm_generated_code*            io_generated_code,
                                                      libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                      const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                      libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                      const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                      libxsmm_blocking_info_t*           n_blocking_info,
                                                      libxsmm_blocking_info_t*           m_blocking_info,
                                                      unsigned int                       A_offs,
                                                      unsigned int                       B_offs,
                                                      unsigned int                       fully_unrolled_brloop );

#endif /* GENERATOR_GEMM_AMX_MICROKERNEL_H */

