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
#ifndef GENERATOR_GEMM_RV64_H
#define GENERATOR_GEMM_RV64_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_rv64_microkernel_rvv( libxsmm_generated_code*            io_generated_code,
                                                  const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                  const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                  const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                  const unsigned int                 i_m_blocking,
                                                  const unsigned int                 i_n_blocking,
                                                  const int                          u_loop_index,
                                                  const int                          u_fma_index,
                                                  const unsigned int                 i_pipeline,
                                                  const unsigned int                 i_reg_gp );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_rv64_kloop( libxsmm_generated_code*            io_generated_code,
                                        libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                        const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                        const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                        const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                        const unsigned int                 i_m_blocking,
                                        const unsigned int                 i_n_blocking,
                                        const unsigned int                 i_reg_gp );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_rv64_kernel( libxsmm_generated_code*        io_generated_code,
                                         const libxsmm_gemm_descriptor* i_xgemm_desc );

#endif /* GENERATOR_GEMM_RV64_H */

