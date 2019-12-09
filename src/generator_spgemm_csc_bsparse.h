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

#ifndef GENERATOR_SPGEMM_CSC_BSPARSE_H
#define GENERATOR_SPGEMM_CSC_BSPARSE_H

#include <libxsmm_generator.h>

/* @TODO change int based architecture value */
LIBXSMM_API_INTERN
void libxsmm_generator_spgemm_csc_bsparse( libxsmm_generated_code*         io_generated_code,
                                           const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                           const char*                     i_arch,
                                           const unsigned int*             i_row_idx,
                                           const unsigned int*             i_column_idx,
                                           const double*                   i_values );

#endif /* GENERATOR_SPGEMM_CSC_BSPARSE_H */

