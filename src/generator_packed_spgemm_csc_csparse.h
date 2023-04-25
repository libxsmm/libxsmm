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
#ifndef GENERATOR_PACKED_SPGEMM_CSC_CSPARSE_H
#define GENERATOR_PACKED_SPGEMM_CSC_CSPARSE_H

#include <libxsmm_generator.h>

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_csc_csparse( libxsmm_generated_code*         io_generated_code,
                                                  const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                  const unsigned int*             i_row_idx,
                                                  const unsigned int*             i_column_idx,
                                                  const void*                     i_values,
                                                  const unsigned int              i_packed_width );

#endif /* GENERATOR_PACKED_SPGEMM_CSC_CSPARSE_H */

