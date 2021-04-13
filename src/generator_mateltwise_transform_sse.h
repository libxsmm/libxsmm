/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evanelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/

#ifndef GENERATOR_MATELTWISE_TRANSFORM_SSE_H
#define GENERATOR_MATELTWISE_TRANSFORM_SSE_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                            libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                            const unsigned int                      i_gp_reg_in,
                                                                            const unsigned int                      i_gp_reg_out,
                                                                            const unsigned int                      i_gp_reg_m_loop,
                                                                            const unsigned int                      i_gp_reg_n_loop,
                                                                            const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                            const libxsmm_meltw_descriptor*         i_mateltwise_desc );


LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_64bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc );


LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_32bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc );


LIBXSMM_API_INTERN
void libxsmm_generator_transform_sse_microkernel( libxsmm_generated_code*                        io_generated_code,
                                                  libxsmm_loop_label_tracker*                    io_loop_label_tracker,
                                                  libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
                                                  const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                                  const libxsmm_meltw_descriptor*                i_mateltwise_desc );

#endif /* GENERATOR_MATELTWISE_TRANSFORM_SSE_H */

