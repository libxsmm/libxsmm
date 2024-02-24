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

#ifndef GENERATOR_PACKED_SPGEMM_BCSC_BSPARSE_AVX_AVX2_AVX512_AMX_H
#define GENERATOR_PACKED_SPGEMM_BCSC_BSPARSE_AVX_AVX2_AVX512_AMX_H

#include <libxsmm_generator.h>
#include "generator_common.h"

LIBXSMM_API_INTERN
unsigned int  libxsmm_generator_x86_packed_spgemm_bcsc_pf_dist_B(void);

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_bcsc_prefetch_B_block_in_L1(libxsmm_generated_code*     io_generated_code,
    unsigned int i_size_in_bytes,
    unsigned int i_base_reg,
    long long    i_offset_in_bytes);

LIBXSMM_API_INTERN
void libxsmm_spgemm_max_mn_blocking_factors_x86(libxsmm_generated_code* io_generated_code, unsigned int i_bn, unsigned int *o_max_m_bf, unsigned int *o_max_n_bf);

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_bcsc_bsparse_avx_avx2_avx512_amx( libxsmm_generated_code*         io_generated_code,
                                                                       const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                                       const unsigned int              i_packed_width,
                                                                       const unsigned int              i_bk,
                                                                       const unsigned int              i_bn );
LIBXSMM_API_INTERN
void libxsmm_spgemm_setup_tile( unsigned int tile_id,
                                unsigned int n_rows,
                                unsigned int n_cols,
                                libxsmm_tile_config *tc);

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_bcsc_bsparse_config_tiles_amx( libxsmm_generated_code*         io_generated_code,
                                                                    const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                                    unsigned int                    i_simd_packed_remainder,
                                                                    unsigned int                    i_simd_packed_iters,
                                                                    unsigned int*                   i_packed_reg_block,
                                                                    const unsigned int              i_bk,
                                                                    const unsigned int              i_bn,
                                                                    unsigned int*                   io_a_tile_id_starts );

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_bcsc_bsparse_kloop_bfdot_avx512(libxsmm_generated_code*            io_generated_code,
                                                                     libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                     libxsmm_jump_label_tracker*        i_jump_label_tracker,
                                                                     const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                     const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                     const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                     const unsigned int                 i_packed_processed,
                                                                     const unsigned int                 i_packed_range,
                                                                     const unsigned int                 i_packed_blocking,
                                                                     const unsigned int                 i_packed_remainder,
                                                                     const unsigned int                 i_packed_width,
                                                                     const unsigned int                 i_simd_packed_width,
                                                                     const unsigned int                 i_bk,
                                                                     const unsigned int                 i_bn);

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_bcsc_bsparse_kloop_amx(         libxsmm_generated_code*            io_generated_code,
                                                                     libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                     libxsmm_jump_label_tracker*        i_jump_label_tracker,
                                                                     const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                     const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                     const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                     const unsigned int                 i_packed_processed,
                                                                     const unsigned int                 i_packed_range,
                                                                     const unsigned int                 i_packed_blocking,
                                                                     const unsigned int                 i_packed_remainder,
                                                                     const unsigned int                 i_packed_width,
                                                                     const unsigned int                 i_simd_packed_width,
                                                                     const unsigned int                 i_bk,
                                                                     const unsigned int                 i_bn,
                                                                     unsigned int                       i_split_tiles,
                                                                     unsigned int*                      i_a_tile_id_starts );

#endif /* GENERATOR_PACKED_SPGEMM_BCSC_BSPARSE_AVX_AVX2_AVX512_AMX_H */
