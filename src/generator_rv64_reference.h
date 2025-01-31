/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Siddharth Rai, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_RV64_REFERENCE_H
#define GENERATOR_RV64_REFERENCE_H

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_rv64_reference_kernel( libxsmm_generated_code*         io_generated_code,
                                                         const libxsmm_meltw_descriptor* i_mateltw_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_rv64_reference_kernel( libxsmm_generated_code*        io_generated_code,
                                                   const libxsmm_gemm_descriptor* i_gemm_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_rv64_reference_kernel( libxsmm_generated_code*        io_generated_code,
                                                          const libxsmm_meqn_descriptor* i_mateqn_desc );
#endif /* GENERATOR_RV64_REFERENCE_H */

