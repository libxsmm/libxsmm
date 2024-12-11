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
#ifndef GENERATOR_MATELTWISE_REFERENCE_IMPL_H
#define GENERATOR_MATELTWISE_REFERENCE_IMPL_H

LIBXSMM_API_INTERN
unsigned char libxsmm_extract_bit(const char *bit_matrix, libxsmm_blasint i, libxsmm_blasint j, libxsmm_blasint ld);

LIBXSMM_API_INTERN
void libxsmm_reference_unary_elementwise(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc);

LIBXSMM_API_INTERN
void libxsmm_reference_binary_elementwise(libxsmm_meltw_binary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc);

LIBXSMM_API_INTERN
void libxsmm_reference_ternary_elementwise(libxsmm_meltw_ternary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc);

LIBXSMM_API_INTERN
void libxsmm_reference_elementwise(void *param,  const libxsmm_meltw_descriptor *i_mateltwise_desc);

#endif /* GENERATOR_MATELTWISE_REFERENCE_IMPL_H */


