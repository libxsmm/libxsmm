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

#ifndef GENERATOR_GEMM_AVX_MICROKERNEL_H
#define GENERATOR_GEMM_AVX_MICROKERNEL_H

#include "generator_gemm_common.h"


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_avx_microkernel( libxsmm_generated_code*             io_generated_code,
                                              const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                              const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                              const libxsmm_gemm_descriptor*     i_xgemm_desc,
                                              const unsigned int                 i_m_blocking,
                                              const unsigned int                 i_n_blocking,
                                              const int                          i_offset );

#endif /* GENERATOR_GEMM_AVX_MICROKERNEL_H */

