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

#ifndef GENERATOR_SPGEMM_CSC_READER_H
#define GENERATOR_SPGEMM_CSC_READER_H

#include <libxsmm_generator.h>
#include <libxsmm_macros.h>

LIBXSMM_API_INTERN
void libxsmm_sparse_csc_reader( libxsmm_generated_code* io_generated_code,
                                const char*             i_csc_file_in,
                                unsigned int**          o_row_idx,
                                unsigned int**          o_column_idx,
                                double**                o_values,
                                unsigned int*           o_row_count,
                                unsigned int*           o_column_count,
                                unsigned int*           o_element_count );

#endif /* GENERATOR_SPGEMM_CSC_READER_H */

