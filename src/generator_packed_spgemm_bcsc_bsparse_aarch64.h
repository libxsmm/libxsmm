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

#ifndef GENERATOR_PACKED_SPGEMM_BCSC_BSPARSE_AARCH64_H
#define GENERATOR_PACKED_SPGEMM_BCSC_BSPARSE_AARCH64_H

#include <libxsmm_generator.h>
#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_spgemm_max_mn_blocking_factors_aarch64(libxsmm_generated_code* io_generated_code, unsigned int i_use_mmla, unsigned int i_bn, unsigned int *o_max_m_bf, unsigned int *o_max_n_bf);

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_bcsc_bsparse_aarch64( libxsmm_generated_code*         io_generated_code,
                                                           const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                           const unsigned int              i_packed_width,
                                                           const unsigned int              i_bk,
                                                           const unsigned int              i_bn );

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_bcsc_bsparse_aarch64_kloop_mmla_sve( libxsmm_generated_code*            io_generated_code,
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
                                                                          const unsigned int                 i_bn );

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_bcsc_bsparse_aarch64_kloop_bfdot_sve(libxsmm_generated_code*            io_generated_code,
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
#endif /* GENERATOR_PACKED_SPGEMM_BCSC_BSPARSE_AARCH64_H */
