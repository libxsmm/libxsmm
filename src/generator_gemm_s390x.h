/******************************************************************************
* Copyright (c), 2025 IBM Corporation - All rights reserved.                  *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Will Trojak (IBM Corp.)
******************************************************************************/

#ifndef GENERATOR_GEMM_S390X_H
#define GENERATOR_GEMM_S390X_H

#include "generator_common.h"
#include "generator_s390x_instructions.h"
#include "../include/libxsmm_typedefs.h"

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vecotr_m_loop( libxsmm_generated_code        *io_generated_code,
                                           const libxsmm_gemm_descriptor *i_xgemm_desc,
                                           libxsmm_s390x_reg             *io_reg_tracker,
                                           libxsmm_loop_label_tracker    *io_loop_labels );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vector_kernel( libxsmm_generated_code        *io_generated_code,
                                           const libxsmm_gemm_descriptor *i_xgemm_desc,
                                           libxsmm_s390x_reg             *io_reg_tracker );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_s390x_kernel( libxsmm_generated_code        *io_generated_code,
                                          const libxsmm_gemm_descriptor *i_xgemm_desc );

#endif
