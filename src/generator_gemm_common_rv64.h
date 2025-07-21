/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved                       *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Siddharth Rai, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_GEMM_COMMON_RV64_H
#define GENERATOR_GEMM_COMMON_RV64_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_apply_relu_fusion_2dregblock_rv64(  libxsmm_generated_code*         io_generated_code,
                                                                const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                                libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                                const unsigned int              i_gp_reg_scratch0,
                                                                const unsigned int              i_gp_reg_scratch1,
                                                                const unsigned int              i_vec_length,
                                                                const unsigned int              i_vec_reg_count,
                                                                const unsigned int              i_m_blocking,
                                                                const unsigned int              i_n_blocking );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_apply_fusion_2dregblock_rv64(  libxsmm_generated_code*         io_generated_code,
                                                           const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                           libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                           const unsigned int              i_gp_reg_scratch0,
                                                           const unsigned int              i_gp_reg_scratch1,
                                                           const unsigned int              i_vec_length,
                                                           const unsigned int              i_vec_reg_count,
                                                           const unsigned int              i_m_blocking,
                                                           const unsigned int              i_n_blocking );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_load_add_colbias_2dregblock_rv64(  libxsmm_generated_code*        io_generated_code,
                                                               const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                               const unsigned int             i_gp_reg_addr,
                                                               const unsigned int             i_gp_reg_scratch0,
                                                               const unsigned int             i_vec_length,
                                                               const unsigned int             i_vec_reg_count,
                                                               libxsmm_datatype               colbias_precision,
                                                               const unsigned int             i_m_blocking,
                                                               const unsigned int             i_n_blocking,
                                                               const unsigned int             i_ld );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_getval_stack_var_rv64( libxsmm_generated_code*  io_generated_code,
                                                   libxsmm_gemm_stack_var   stack_var,
                                                   unsigned int             i_gp_reg );
LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setval_stack_var_rv64( libxsmm_generated_code*  io_generated_code,
                                                   libxsmm_gemm_stack_var   stack_var,
                                                   unsigned int             i_aux_reg,
                                                   unsigned int             i_gp_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setup_stack_frame_rv64( libxsmm_generated_code* io_generated_code,
    const libxsmm_gemm_descriptor*  i_xgemm_desc,
    const libxsmm_gp_reg_mapping*   i_gp_reg_mapping,
    libxsmm_micro_kernel_config*    i_micro_kernel_config );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_destroy_stack_frame_rv64( libxsmm_generated_code* io_generated_code);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_setup_stack_frame_fill_ext_gemm_stack_vars_rv64( libxsmm_generated_code* io_generated_code,
    const libxsmm_gemm_descriptor*  i_xgemm_desc,
    libxsmm_micro_kernel_config*    i_micro_kernel_config,
    const libxsmm_gp_reg_mapping*   i_gp_reg_mapping );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_init_micro_kernel_config_rv64( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                              const unsigned int             i_arch,
                                                              const libxsmm_gemm_descriptor* i_xgemm_desc );

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_rv64_get_max_n_blocking( const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                                             const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                             const unsigned int                  i_arch );

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_rv64_get_initial_m_blocking( libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                                 const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                                 const unsigned int              i_arch );

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_rv64_update_m_blocking( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                            const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                            const unsigned int             i_arch,
                                                            const unsigned int             i_current_m_blocking );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_rv64_setup_n_blocking( libxsmm_generated_code*        io_generated_code,
                                                   libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                   const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                   const unsigned int             i_arch,
                                                   unsigned int*                  o_n_N,
                                                   unsigned int*                  o_n_n);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_rv64_setup_k_strides( libxsmm_generated_code*            io_generated_code,
                                                  const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                  const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                  const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                  const unsigned int                 i_m_blocking,
                                                  const unsigned int                 i_n_blocking );

#endif /* GENERATOR_GEMM_COMMON_RV64_H */

