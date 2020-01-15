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

#ifndef GENERATOR_GEMM_AVX512_FSDBCST_H
#define GENERATOR_GEMM_AVX512_FSDBCST_H

#if 0
#include "generator_common.h"

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_avx512_fsdbcst_get_max_n_blocking( const libxsmm_gemm_descriptor* i_xgemm_desc,
                                                                       const unsigned int             i_arch );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_avx512_kernel_fsdbcst( libxsmm_generated_code*        io_generated_code,
                                                   const libxsmm_gemm_descriptor* i_xgemm_desc );
#endif

#endif /* GENERATOR_GEMM_AVX512_FSDBCST_H */

