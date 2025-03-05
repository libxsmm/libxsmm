/******************************************************************************
* Copyright (c) 2025 IBM Corp. - All rights reserved.                         *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Will Trojak (IBM Corp.)
******************************************************************************/
#ifndef GENERATOR_PPC64LE_REFERENCE_H
#define GENERATOR_PPC64LE_REFERENCE_H

#include "generator_mateltwise_common.h"
#include "generator_ppc64le_instructions.h"
#include "generator_common.h"
#include "generator_mateltwise_reference_impl.h"
#include "generator_gemm_reference_impl.h"
#include "generator_matequation_avx_avx512.h"
#include "generator_matequation_reference_impl.h"
#include "libxsmm_matrixeqn.h"

LIBXSMM_API_INTERN
void libxsmm_generator_ppc64le_reference_kernel( libxsmm_generated_code *io_generated_code,
                                                 const void             *i_desc,
                                                 unsigned int            i_is_gemm_or_eltwise );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_reference_kernel( libxsmm_generated_code        *io_generated_code,
                                                      const libxsmm_gemm_descriptor *i_gemm_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_ppc64le_reference_kernel( libxsmm_generated_code         *io_generated_code,
                                                            const libxsmm_meltw_descriptor *i_mateltw_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_ppc64le_reference_kernel( libxsmm_generated_code        *io_generated_code,
                                                             const libxsmm_meqn_descriptor *i_mateqn_desc );

#endif /* GENERATOR_PPC64LE_REFERENCE_H */
