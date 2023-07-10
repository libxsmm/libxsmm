/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena), Alexander Heinecke (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_PACKED_SPGEMM_CSR_BSPARSE_AARCH64_H
#define GENERATOR_PACKED_SPGEMM_CSR_BSPARSE_AARCH64_H

#include <libxsmm_generator.h>
#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csr_bsparse_aarch64( libxsmm_generated_code*         io_generated_code,
                                                          const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                          const unsigned int*             i_row_idx,
                                                          const unsigned int*             i_column_idx,
                                                          const void*                     i_values,
                                                          const unsigned int              i_packed_width );

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csr_bsparse_aarch64_kloop_asimd( libxsmm_generated_code*            io_generated_code,
                                                                      libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                      const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                      const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                      const unsigned int*                i_row_idx,
                                                                      const unsigned int*                i_column_idx,
                                                                      const void*                        i_values,
                                                                      const unsigned int                 i_n_processed,
                                                                      const unsigned int                 i_n_limit,
                                                                      const unsigned int                 i_packed_processed,
                                                                      const unsigned int                 i_packed_range,
                                                                      const unsigned int                 i_packed_blocking,
                                                                      const unsigned int                 i_packed_remainder,
                                                                      const unsigned int                 i_packed_width );

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csr_bsparse_aarch64_kloop_sve( libxsmm_generated_code*            io_generated_code,
                                                                    libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                    const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                    const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                    const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                    const unsigned int*                i_row_idx,
                                                                    const unsigned int*                i_column_idx,
                                                                    const void*                        i_values,
                                                                    const unsigned int                 i_n_processed,
                                                                    const unsigned int                 i_n_limit,
                                                                    const unsigned int                 i_packed_processed,
                                                                    const unsigned int                 i_packed_range,
                                                                    const unsigned int                 i_packed_blocking,
                                                                    const unsigned int                 i_packed_remainder,
                                                                    const unsigned int                 i_packed_width );

#endif /* GENERATOR_PACKE_SPGEMM_CSR_BSPARSE_AARCH64_H */

