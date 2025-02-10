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

#include "generator_gemm_s390x.h"


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_vector_kernel( libxsmm_generated_code *io_generated_code,
                                           const libxsmm_gemm_descriptor *i_xgemm_desc ) {
  libxsmm_s390x_instr_nop( io_generated_code );
  libxsmm_s390x_instr_return( io_generated_code );
}


LIBXSMM_API_INTERN
void libxsmm_generator_gemm_s390x_kernel( libxsmm_generated_code        *io_generated_code,
                                          const libxsmm_gemm_descriptor *i_xgemm_desc ) {
  libxsmm_generator_gemm_vector_kernel( io_generated_code,
                                        i_xgemm_desc );
  return;
}
