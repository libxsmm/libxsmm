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

#ifndef GENERATOR_SPGEMM_CSC_CSPARSE_SOA_H
#define GENERATOR_SPGEMM_CSC_CSPARSE_SOA_H

#include "generator_common.h"
#include <libxsmm_generator.h>

/* @TODO change int based architecture value */
LIBXSMM_API_INTERN
void libxsmm_generator_spgemm_csc_csparse_soa( libxsmm_generated_code*         io_generated_code,
                                               const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                               const char*                     i_arch,
                                               const unsigned int*             i_row_idx,
                                               const unsigned int*             i_column_idx,
                                               const void*                     i_values,
                                               const unsigned int              i_packed_width );

LIBXSMM_API_INTERN
void libxsmm_generator_spgemm_csc_csparse_soa_avx256_512( libxsmm_generated_code*         io_generated_code,
                                                          const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                          const unsigned int*             i_row_idx,
                                                          const unsigned int*             i_column_idx,
                                                          const void*                     i_values,
                                                          const unsigned int              i_packed_width );

LIBXSMM_API_INTERN
void libxsmm_generator_spgemm_csc_csparse_soa_axv256_512_single( libxsmm_generated_code*            io_generated_code,
                                                                 libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                 const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                 const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                 const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                 const unsigned int*                i_row_idx,
                                                                 const unsigned int*                i_column_idx,
                                                                 const unsigned int                 i_packed_width,
                                                                 const unsigned int                 i_n,
                                                                 const unsigned int                 i_m );

LIBXSMM_API_INTERN
void libxsmm_generator_spgemm_csc_csparse_soa_axv256_512_16accs( libxsmm_generated_code*            io_generated_code,
                                                                 libxsmm_loop_label_tracker*        io_loop_label_tracker,
                                                                 const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                                 const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                                 const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                                                 const unsigned int*                i_row_idx,
                                                                 const unsigned int*                i_column_idx,
                                                                 const unsigned int                 i_packed_width,
                                                                 const unsigned int                 i_n,
                                                                 const unsigned int                 i_m,
                                                                 const unsigned int                 i_m_blocking );

#endif /* GENERATOR_SPGEMM_CSC_CSPARSE_SOA_H */

