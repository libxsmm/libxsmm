/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_AARCH64_REFERENCE_H
#define GENERATOR_AARCH64_REFERENCE_H

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_aarch64_reference_kernel( libxsmm_generated_code*         io_generated_code,
                                                        const libxsmm_meltw_descriptor* i_mateltw_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_reference_kernel( libxsmm_generated_code* io_generated_code,
                                                      const libxsmm_gemm_descriptor* i_gemm_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_aarch64_reference_kernel( libxsmm_generated_code*         io_generated_code,
                                                         const libxsmm_meqn_descriptor* i_mateqn_desc );
#endif /* GENERATOR_AARCH64_REFERENCE_H */

