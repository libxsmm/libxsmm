/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evanelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/

#ifndef GENERATOR_MATEQUATION_AVX_AVX512_H
#define GENERATOR_MATEQUAITON_AVX_AVX512_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_avx_avx512_kernel( libxsmm_generated_code*        io_generated_code,
                                                      const libxsmm_meqn_descriptor* i_mateqn_desc );

#endif /* GENERATOR_MATEQUATION_AVX_AVX512_H */

