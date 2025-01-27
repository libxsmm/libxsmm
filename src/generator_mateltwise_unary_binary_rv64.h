/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Siddharth Rai, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_MATELTWISE_UNARY_BINARY_RV64_H
#define GENERATOR_MATELTWISE_UNARY_BINARY_RV64_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_load_rv64_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
                                     libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                     const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                     const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                     unsigned int                            i_vlen,
                                     unsigned int                            i_avlen,
                                     unsigned int                            i_start_vreg,
                                     unsigned int                            i_m_blocking,
                                     unsigned int                            i_n_blocking,
                                     unsigned int                            i_mask_last_m_chunk,
                                     unsigned int                            i_mask_reg);

LIBXSMM_API_INTERN
void libxsmm_compute_binary_rv64_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
                                               libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                               const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                               const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                               unsigned int                            i_vlen,
                                               unsigned int                            i_avlen,
                                               unsigned int                            i_start_vreg,
                                               unsigned int                            i_m_blocking,
                                               unsigned int                            i_n_blocking,
                                               unsigned int                            i_mask_last_m_chunk,
                                               unsigned int                            i_mask_reg);

LIBXSMM_API_INTERN
void libxsmm_compute_unary_binary_rv64_2d_reg_block( libxsmm_generated_code*                 io_generated_code,
                                                     libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                     const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                     const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                     unsigned int                            i_vlen,
                                                     unsigned int                            i_avlen,
                                                     unsigned int                            i_start_vreg,
                                                     unsigned int                            i_m_blocking,
                                                     unsigned int                            i_n_blocking,
                                                     unsigned int                            i_mask_last_m_chunk,
                                                     unsigned int                            i_mask_reg);

LIBXSMM_API_INTERN
void libxsmm_generator_unary_binary_rv64_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                      libxsmm_mateltwise_gp_reg_mapping*      i_gp_reg_mapping,
                                                      libxsmm_mateltwise_kernel_config*       i_micro_kernel_config,
                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc );

#endif /* GENERATOR_MATELTWISE_UNARY_BINARY_RV64_H */
