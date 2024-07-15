/******************************************************************************
* Copyright (c) 2024, IBM Corporation                                         *
* - All rights reserved.                                                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Will Trojak (IBM Corp.)
******************************************************************************/

#ifndef GENERATOR_PPC64LE_VSX_MICROKERNEL_H
#define GENERATOR_PPC64LE_VSX_MICROKERNEL_H

#include "generator_ppc64le_instructions.h"

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_vsx_mk_bytes( libxsmm_datatype const datatype );


LIBXSMM_API_INTERN
void libxsmm_generator_vsx_microkernel( libxsmm_generated_code *io_generated,
                                        libxsmm_gemm_descriptor const *i_xgemm_desc,
                                        unsigned char a_ptr_gpr,
                                        unsigned char b_ptr_gpr,
                                        unsigned char c_ptr_gpr,
                                        unsigned int  m_block,
                                        unsigned int  n_block,
                                        unsigned int  k_block );

#endif
