/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Greg Henry (Intel Corp.)
******************************************************************************/

#ifndef GENERATOR_TRANSPOSE_AVX_AVX512_H
#define GENERATOR_TRANSPOSE_AVX_AVX512_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_transpose_avx_avx512_kernel( libxsmm_generated_code*         io_generated_code,
                                                    const libxsmm_trans_descriptor* i_trans_desc,
                                                    int                             i_arch );

#endif /* GENERATOR_TRANSPOSE_AVX_AVX512_H */

