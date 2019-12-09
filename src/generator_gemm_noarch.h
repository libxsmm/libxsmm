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

#ifndef GENERATOR_GEMM_NOARCH_H
#define GENERATOR_GEMM_NOARCH_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_noarch_kernel( libxsmm_generated_code*         io_generated_code,
                                           const libxsmm_gemm_descriptor*  i_xgemm_desc );

#endif /* GENERATOR_GEMM_NOARCH_H */

