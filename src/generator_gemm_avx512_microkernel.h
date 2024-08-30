/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_GEMM_AVX512_MICROKERNEL_H
#define GENERATOR_GEMM_AVX512_MICROKERNEL_H

#include "generator_common.h"
#include "generator_gemm_common.h"

LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_kloop_kernel( libxsmm_generated_code*            io_generated_code,
                                                                    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                    const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                    const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                    const unsigned int                 i_m_blocking,
                                                                    const unsigned int                 i_n_blocking,
                                                                    const unsigned int                 i_k_blocking );

LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_nofsdbcst( libxsmm_generated_code*            io_generated_code,
                                                                             const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                             const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                             const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                             const unsigned int                 i_m_blocking,
                                                                             const unsigned int                 i_n_blocking,
                                                                             const int                          i_offset );

LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_m8_nofsdbcst( libxsmm_generated_code*            io_generated_code,
                                                                                      const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                                      const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                      const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                      const unsigned int                 i_m_blocking,
                                                                                      const unsigned int                 i_n_blocking,
                                                                                      const int                          i_offset );

LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_i8_ss_uu_emu_nofsdbcst( libxsmm_generated_code*            io_generated_code,
                                                                                          const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                                          const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                          const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                          const unsigned int                 i_m_blocking,
                                                                                          const unsigned int                 i_n_blocking,
                                                                                          const int                          i_offset );

LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_m8_i8_ss_uu_emu_nofsdbcst( libxsmm_generated_code*            io_generated_code,
                                                                                             const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                                             const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                             const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                             const unsigned int                 i_m_blocking,
                                                                                             const unsigned int                 i_n_blocking,
                                                                                             const int                          i_offset );

LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_bf16_emu_nofsdbcst( libxsmm_generated_code*            io_generated_code,
                                                                                      const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                                      const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                      const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                      const unsigned int                 i_m_blocking,
                                                                                      const unsigned int                 i_n_blocking,
                                                                                      const int                          i_offset );

LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_m8_bf16_emu_nofsdbcst( libxsmm_generated_code*            io_generated_code,
                                                                                         const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                                         const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                         const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                         const unsigned int                 i_m_blocking,
                                                                                         const unsigned int                 i_n_blocking,
                                                                                         const int                          i_offset );

LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_bf8_emu_nofsdbcst( libxsmm_generated_code*            io_generated_code,
                                                                                     const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                                     const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                     const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                     const unsigned int                 i_m_blocking,
                                                                                     const unsigned int                 i_n_blocking,
                                                                                     const int                          i_offset );

LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_m8_bf8_emu_nofsdbcst( libxsmm_generated_code*            io_generated_code,
                                                                                        const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                                        const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                        const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                        const unsigned int                 i_m_blocking,
                                                                                        const unsigned int                 i_n_blocking,
                                                                                        const int                          i_offset );

LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_fsdbcst( libxsmm_generated_code*            io_generated_code,
                                                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                           const unsigned int                 i_n_blocking,
                                                                           const unsigned int                 i_k_blocking );

LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_loadNinterleave_A_pair_k_i8_to_bf16( libxsmm_generated_code*            io_generated_code,
                                                                                                       const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                                                       const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                                       const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                                       unsigned int                       io_A_vreg,
                                                                                                       unsigned int                       i_tmp_vreg,
                                                                                                       unsigned int                       i_interleave_vreg,
                                                                                                       unsigned int                       i_m_blocking,
                                                                                                       unsigned int                       i_m  );

LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_process_vreg_A( libxsmm_generated_code*            io_generated_code,
                                                                                  const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                  const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                  char                               i_vname_cvt,
                                                                                  unsigned int                       i_is_Ai8_Bf16_gemm,
                                                                                  unsigned int                       i_is_Abf8_Bf16_gemm,
                                                                                  unsigned int                       i_is_Af16_Bf16_gemm,
                                                                                  unsigned int                       i_use_f16_replacement_fma,
                                                                                  unsigned int                       i_use_f32_compute_with_f16_inp,
                                                                                  unsigned int                       i_m,
                                                                                  unsigned int                       i_m_blocking,
                                                                                  unsigned int                       io_A_vreg );

LIBXSMM_API_INTERN void libxsmm_generator_gemm_avx512_microkernel_process_vreg_A_for_i4i8 ( libxsmm_generated_code*            io_generated_code,
                                                                                  const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                                  const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                                  unsigned int                       i_m,
                                                                                  unsigned int                       i_m_blocking,
                                                                                  unsigned int                       io_A_vreg );

#endif /* GENERATOR_GEMM_AVX512_MICROKERNEL_H */

