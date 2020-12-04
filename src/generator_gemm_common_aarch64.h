/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
*               Intel Corporation - All rights reserved                       *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena), Alexander Heinecke (Intel Corp.)
******************************************************************************/

#ifndef GENERATOR_GEMM_COMMON_AARCH64_H
#define GENERATOR_GEMM_COMMON_AARCH64_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_init_micro_kernel_config_aarch64( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                              const unsigned int             i_arch,
                                                              const libxsmm_gemm_descriptor* i_xgemm_desc );

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_aarch64_get_max_n_blocking( const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                                                const libxsmm_gemm_descriptor*      i_xgemm_desc,
                                                                const unsigned int                  i_arch );

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_aarch64_get_initial_m_blocking( libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                                    const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                                    const unsigned int              i_arch );

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_aarch64_update_m_blocking( libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                               const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                               const unsigned int             i_arch,
                                                               const unsigned int             i_current_m_blocking );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_setup_n_blocking( libxsmm_generated_code*        io_generated_code,
                                                      libxsmm_micro_kernel_config*   io_micro_kernel_config,
                                                      const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                      const unsigned int             i_arch,
                                                      unsigned int*                  o_n_N,
                                                      unsigned int*                  o_n_n);

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_setup_k_strides( libxsmm_generated_code*            io_generated_code,
                                                     const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                     const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                     const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                     const unsigned int                 i_m_blocking,
                                                     const unsigned int                 i_n_blocking );

#endif /* GENERATOR_GEMM_COMMON_AARCH64_H */

