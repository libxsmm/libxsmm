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
#ifndef GENERATOR_PACKED_TRSM_AVX_AVX512_H
#define GENERATOR_PACKED_TRSM_AVX_AVX512_H

#include "generator_common.h"


LIBXSMM_API_INTERN
void libxsmm_generator_packed_trsm_avx_avx512_kernel( libxsmm_generated_code*         io_generated_code,
                                                      const libxsmm_trsm_descriptor*  i_packed_trsm_desc,
                                                      const char*                     i_arch );

#endif /*GENERATOR_PACKED_TRSM_AVX_AVX512_H*/

