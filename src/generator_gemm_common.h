/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

#ifndef GENERATOR_GEMM_COMMON_H
#define GENERATOR_GEMM_COMMON_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_init_micro_kernel_config_fullvector( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                                 const unsigned int             i_arch,
                                                                 const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                                 const unsigned int             i_use_masking_a_c );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_init_micro_kernel_config_halfvector( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                                 const unsigned int             i_arch,
                                                                 const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                                 const unsigned int             i_use_masking_a_c );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_init_micro_kernel_config_scalar( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                             const unsigned int             i_arch,
                                                             const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                             const unsigned int             i_use_masking_a_c );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_add_flop_counter( libxsmm_generated_code*         io_generated_code,
                                               const libxsmm_gemm_descriptor* i_xgemm_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_kloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const unsigned int                 i_m_blocking,
                                           const unsigned int                 i_k_blocking );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_kloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                           const unsigned int                 i_m_blocking,
                                           const unsigned int                 i_max_blocked_k,
                                           const unsigned int                 i_kloop_complete );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_reduceloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_reduceloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const libxsmm_gemm_descriptor*     i_xgemm_desc);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_nloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const unsigned int                 i_n_blocking );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_nloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                           const unsigned int                 i_n_blocking,
                                           const unsigned int                 i_n_done );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_header_mloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const unsigned int                 i_m_blocking );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_footer_mloop( libxsmm_generated_code*             io_generated_code,
                                           libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                           const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                           const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                           const unsigned int                 i_m_blocking,
                                           const unsigned int                 i_m_done );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_load_C( libxsmm_generated_code*             io_generated_code,
                                     const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                     const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                     const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                     const unsigned int                 i_m_blocking,
                                     const unsigned int                 i_n_blocking );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_store_C( libxsmm_generated_code*             io_generated_code,
                                      const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                      const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                      const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                      const unsigned int                 i_m_blocking,
                                      const unsigned int                 i_n_blocking );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_initialize_avx512_mask( libxsmm_generated_code*            io_generated_code,
                                                    const unsigned int                 i_gp_reg_tmp,
                                                    const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                    const unsigned int                 i_mask_count );

#endif /* GENERATOR_GEMM_COMMON_H */
