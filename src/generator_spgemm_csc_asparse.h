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

#ifndef GENERATOR_SPGEMM_CSC_ASPARSE_H
#define GENERATOR_SPGEMM_CSC_ASPARSE_H

#include <libxsmm_generator.h>

LIBXSMM_API_INTERN
void libxsmm_sparse_csc_asparse_innerloop_scalar( libxsmm_generated_code*        io_generated_code,
                                                  const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                  const unsigned int             i_k,
                                                  const unsigned int             i_z,
                                                  const unsigned int*            i_row_idx,
                                                  const unsigned int*            i_column_idx );

LIBXSMM_API_INTERN
void libxsmm_sparse_csc_asparse_innerloop_two_vector( libxsmm_generated_code*        io_generated_code,
                                                      const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                      const unsigned int             i_k,
                                                      const unsigned int             i_z,
                                                      const unsigned int*            i_row_idx,
                                                      const unsigned int*            i_column_idx );

LIBXSMM_API_INTERN
void libxsmm_sparse_csc_asparse_innerloop_four_vector( libxsmm_generated_code*        io_generated_code,
                                                       const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                       const unsigned int             i_k,
                                                       const unsigned int             i_z,
                                                       const unsigned int*            i_row_idx,
                                                       const unsigned int*            i_column_idx );

/* @TODO change int based architecture value */
LIBXSMM_API_INTERN
void libxsmm_generator_spgemm_csc_asparse( libxsmm_generated_code*        io_generated_code,
                                           const libxsmm_gemm_descriptor* i_xgemm_desc,
                                           const char*                    i_arch,
                                           const unsigned int*            i_row_idx,
                                           const unsigned int*            i_column_idx,
                                           const double*                  i_values );

#endif /* GENERATOR_SPGEMM_CSC_ASPARSE_H */

