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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef EDGE_COMMON_H
#define EDGE_COMMON_H

void edge_sparse_csr_reader_double( const char*           i_csr_file_in,
                                    unsigned int**        o_row_idx,
                                    unsigned int**        o_column_idx,
                                    double**              o_values,
                                    unsigned int*         o_row_count,
                                    unsigned int*         o_column_count,
                                    unsigned int*         o_element_count );

void edge_sparse_csr_reader_float( const char*           i_csr_file_in,
                                   unsigned int**        o_row_idx,
                                   unsigned int**        o_column_idx,
                                   float**               o_values,
                                   unsigned int*         o_row_count,
                                   unsigned int*         o_column_count,
                                   unsigned int*         o_element_count );

#endif /* EDGE_COMMON_H */

