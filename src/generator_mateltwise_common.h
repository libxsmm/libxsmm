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
#ifndef GENERATOR_MATELTWISE_COMMON_H
#define GENERATOR_MATELTWISE_COMMON_H

#include <libxsmm.h>
#include "generator_common.h"

LIBXSMM_API_INTERN
libxsmm_blasint libxsmm_generator_mateltwise_all_inp_comp_out_prec(const libxsmm_meltw_descriptor*   i_mateltwise_desc, libxsmm_datatype i_dtype );

LIBXSMM_API_INTERN
libxsmm_blasint libxsmm_generator_mateltwise_involves_prec(const libxsmm_meltw_descriptor*   i_mateltwise_desc, libxsmm_datatype i_dtype );

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_mateltwise_is_binary_cmp_op( const libxsmm_meltw_descriptor*         i_mateltwise_desc);

#endif /* GENERATOR_MATELTWISE_COMMON_H */
