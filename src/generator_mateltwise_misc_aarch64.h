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
#ifndef GENERATOR_MATELTWISE_MISC_AARCH64_H
#define GENERATOR_MATELTWISE_MISC_AARCH64_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_mn_code_block_replicate_col_var_aarch64( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    libxsmm_mateltwise_kernel_config*              i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc,
    unsigned int                                   vlen,
    unsigned int                                   m_trips_loop,
    unsigned int                                   m_unroll_factor,
    unsigned int                                   peeled_m_trips,
    unsigned int                                   i_use_masking,
    unsigned int                                   mask_inout );

LIBXSMM_API_INTERN
void libxsmm_generator_replicate_col_var_aarch64_microkernel( libxsmm_generated_code*                        io_generated_code,
    libxsmm_loop_label_tracker*                    io_loop_label_tracker,
    libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
    libxsmm_mateltwise_kernel_config*              i_micro_kernel_config,
    const libxsmm_meltw_descriptor*                i_mateltwise_desc );

#endif /* GENERATOR_MATELTWISE_MISC_AARCH64_H */

