/******************************************************************************
* Copyright (c), 2025 IBM Corporation - All rights reserved.                  *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Will Trojak
******************************************************************************/
#ifndef GENERATOR_S390X_REFERENCE_H
#define GENERATOR_S390X_REFERENCE_H

#include "generator_mateltwise_common.h"
#include "generator_s390x_instructions.h"
#include "generator_common.h"
#include "generator_mateltwise_reference_impl.h"
#include "generator_gemm_reference_impl.h"
#include "generator_matequation_avx_avx512.h"
#include "generator_matequation_reference_impl.h"
#include "libxsmm_matrixeqn.h"

LIBXSMM_API_INTERN
void libxsmm_generator_s390x_reference_kernel( libxsmm_generated_code* io_generated_code,
                                               const void*             i_desc,
                                               unsigned int            i_is_gemm_or_eltwise );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_s390x_reference_kernel( libxsmm_generated_code*        io_generated_code,
                                                    const libxsmm_gemm_descriptor* i_gemm_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_s390x_reference_kernel( libxsmm_generated_code*         io_generated_code,
                                                          const libxsmm_meltw_descriptor* i_mateltw_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_s390x_reference_kernel( libxsmm_generated_code*        io_generated_code,
                                                           const libxsmm_meqn_descriptor* i_mateqn_desc );

#endif /* GENERATOR_S390X_REFERENCE_H */
