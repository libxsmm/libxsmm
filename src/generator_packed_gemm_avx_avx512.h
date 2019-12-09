/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Greg Henry, Timothy Costa (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_PACKED_GEMM_AVX_AVX512_H
#define GENERATOR_PACKED_GEMM_AVX_AVX512_H

#include "generator_common.h"

#define GARBAGE_PARAMETERS

LIBXSMM_API_INTERN
void libxsmm_generator_packed_gemm_avx_avx512_kernel( libxsmm_generated_code*           io_generated_code,
                                                       const libxsmm_pgemm_descriptor*  i_packed_pgemm_desc,
                                                       const char*                      i_arch
#ifdef GARBAGE_PARAMETERS
                                                   ,   unsigned int                     iunroll,
                                                       unsigned int                     junroll,
                                                       unsigned int                     loopi,
                                                       unsigned int                     loopj
#endif
 );

#endif /*GENERATOR_PACKED_GEMM_AVX_AVX512_H*/

