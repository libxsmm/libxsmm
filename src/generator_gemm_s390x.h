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
#include "generator_gemm_vxrs_microkernel.h"
#include "../include/libxsmm_typedefs.h"

typedef void (*libxsmm_s390x_reg_func)( unsigned int const  i_vec_ele,
                                        unsigned int       *i_blocking,
                                        unsigned int       *o_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_s390x_vxrs_reg( unsigned int  i_vec_ele,
                                            unsigned int *i_blocking,
                                            unsigned int *o_reg );

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_gemm_s390x_reg_sum( unsigned int const     i_vec_ele,
                                                   unsigned int          *i_blocking,
                                                   libxsmm_s390x_reg_func i_reg_func );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_s390x_vxrs_blocking_maximise( libxsmm_generated_code        *io_generated_code,
                                                          const libxsmm_gemm_descriptor *i_xgemm_desc,
                                                          libxsmm_s390x_blocking        *io_blocking,
                                                          libxsmm_s390x_reg_func         i_reg_func );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_s390x_vxrs_blocking_init( libxsmm_generated_code        *io_generated_code,
                                                      const libxsmm_gemm_descriptor *i_xgemm_desc,
                                                      libxsmm_s390x_blocking        *io_blocking );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_s390x_vxrs_m_loop( libxsmm_generated_code        *io_generated_code,
                                               const libxsmm_gemm_descriptor *i_xgemm_desc,
                                               libxsmm_s390x_reg             *io_reg_tracker,
                                               libxsmm_s390x_blocking        *i_blocking,
                                               libxsmm_loop_label_tracker    *io_loop_labels,
                                               unsigned int                   i_a,
                                               unsigned int                   i_b,
                                               unsigned int                   i_c );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_s390x_vxrs_kernel( libxsmm_generated_code        *io_generated_code,
                                               const libxsmm_gemm_descriptor *i_xgemm_desc,
                                               libxsmm_s390x_reg             *io_reg_tracker,
                                               libxsmm_s390x_blocking        *i_blocking );

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_s390x_kernel( libxsmm_generated_code        *io_generated_code,
                                          const libxsmm_gemm_descriptor *i_xgemm_desc );

#endif
