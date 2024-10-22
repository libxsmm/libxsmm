/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
*               Intel Corporation - All rights reserved                       *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena), Alexander Heinecke (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_GEMM_AARCH64_H
#define GENERATOR_GEMM_AARCH64_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_microkernel_asimd_neoverse( libxsmm_generated_code*            io_generated_code,
                                                                const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                const unsigned int                 i_m_blocking,
                                                                const unsigned int                 i_n_blocking,
                                                                const unsigned int                 i_k_index );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_microkernel_asimd_neoverse_v2( libxsmm_generated_code*            io_generated_code,
                                                                   const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                   const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                   const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                   const unsigned int                 i_m_blocking,
                                                                   const unsigned int                 i_n_blocking,
                                                                   const unsigned int                 i_k_index );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_microkernel_asimd_a64fx( libxsmm_generated_code*            io_generated_code,
                                                             const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                             const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                             const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                             const unsigned int                 i_m_blocking,
                                                             const unsigned int                 i_n_blocking,
                                                             const unsigned int                 i_k_index );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_microkernel_asimd_mmla( libxsmm_generated_code*            io_generated_code,
                                                            const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                            const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                            const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                            const unsigned int                 i_m_blocking,
                                                            const unsigned int                 i_n_blocking,
                                                            const unsigned int                 i_k_index );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_microkernel_sve_a64fx( libxsmm_generated_code*            io_generated_code,
                                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                           const unsigned int                 i_m_blocking,
                                                           const unsigned int                 i_n_blocking,
                                                           const unsigned int                 i_k_index );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_microkernel_sve_mmla( libxsmm_generated_code*            io_generated_code,
                                                          const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                          const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                          const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                          const unsigned int                 i_m_blocking,
                                                          const unsigned int                 i_n_blocking,
                                                          const unsigned int                 i_k_index );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_kloop( libxsmm_generated_code*            io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                           const unsigned int                 i_m_blocking,
                                           const unsigned int                 i_n_blocking );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_kernel( libxsmm_generated_code*        io_generated_code,
                                            const libxsmm_gemm_descriptor* i_xgemm_desc );

#endif /* GENERATOR_GEMM_AARCH64_H */

