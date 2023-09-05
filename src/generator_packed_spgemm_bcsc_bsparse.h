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

#ifndef GENERATOR_PACKED_SPGEMM_BCSC_BSPARSE_H
#define GENERATOR_PACKED_SPGEMM_BCSC_BSPARSE_H

#include <libxsmm_generator.h>

LIBXSMM_API_INTERN
void libxsmm_generator_packed_spgemm_bcsc_bsparse( libxsmm_generated_code*         io_generated_code,
                                                   const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                   const unsigned int              i_packed_width,
                                                   const unsigned int              i_bk,
                                                   const unsigned int              i_bn );

#endif /* GENERATOR_PACKED_SPGEMM_BCSC_BSPARSE_H */

