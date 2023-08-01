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
#ifndef GENERATOR_PACKED_SPGEMM_CSR_ASPARSE_AVX_AVX2_AVX512_H
#define GENERATOR_PACKED_SPGEMM_CSR_ASPARSE_AVX_AVX2_AVX512_H

#include "generator_common.h"
#include <libxsmm_generator.h>

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csr_asparse_avx_avx2_avx512( libxsmm_generated_code*         io_generated_code,
                                                                  const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                                  const unsigned int*             i_row_idx,
                                                                  const unsigned int*             i_column_idx,
                                                                  const void*                     i_values,
                                                                  const unsigned int              i_packed_width );

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csr_asparse_avx_avx2_avx512_n_loop( libxsmm_generated_code*            io_generated_code,
                                                                         const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                         libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                         const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                         const unsigned int*                i_row_idx,
                                                                         const unsigned int*                i_column_idx,
                                                                         const void*                        i_values,
                                                                         const unsigned int                 i_n_max_block,
                                                                         const unsigned int                 i_packed_width,
                                                                         const unsigned int                 i_packed_mask );

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csr_asparse_avx_avx2_avx512_m_loop( libxsmm_generated_code*            io_generated_code,
                                                                         const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                         libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                         const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                         const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                         const unsigned int*                i_row_idx,
                                                                         const unsigned int*                i_column_idx,
                                                                         const void*                        i_values,
                                                                         const unsigned int                 i_gen_m_trips,
                                                                         const unsigned int                 i_a_is_dense,
                                                                         const unsigned int                 i_num_c_cols,
                                                                         const unsigned int                 i_packed_width,
                                                                         const unsigned int                 i_packed_mask );

#endif /* GENERATOR_PACKED_SPGEMM_CSR_ASPARSE_AVX_AVX2_AVX512_H */

