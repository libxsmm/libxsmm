/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
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
void libxsmm_generator_transform_norm_to_vnni2_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                            libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                            const unsigned int                      i_gp_reg_in,
                                                                            const unsigned int                      i_gp_reg_out,
                                                                            const unsigned int                      i_gp_reg_m_loop,
                                                                            const unsigned int                      i_gp_reg_n_loop,
                                                                            const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                            const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                            const unsigned int                      i_pad_vnni );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni2_to_vnni2t_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                              libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                              const unsigned int                      i_gp_reg_in,
                                                                              const unsigned int                      i_gp_reg_out,
                                                                              const unsigned int                      i_gp_reg_m_loop,
                                                                              const unsigned int                      i_gp_reg_n_loop,
                                                                              const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                              const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni4_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                            libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                            const unsigned int                      i_gp_reg_in,
                                                                            const unsigned int                      i_gp_reg_out,
                                                                            const unsigned int                      i_gp_reg_m_loop,
                                                                            const unsigned int                      i_gp_reg_n_loop,
                                                                            const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                            const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                            const unsigned int                      i_pad_vnni );
LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_norm_08bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_norm_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                            libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                            const unsigned int                      i_gp_reg_in,
                                                                            const unsigned int                      i_gp_reg_out,
                                                                            const unsigned int                      i_gp_reg_m_loop,
                                                                            const unsigned int                      i_gp_reg_n_loop,
                                                                            const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                            const libxsmm_meltw_descriptor*         i_mateltwise_desc );
LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni2_08bit_sse_microkernel( libxsmm_generated_code*                io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni2_mbit_scalar_sse_microkernel( libxsmm_generated_code*                io_generated_code,
                                                                            libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                            const unsigned int                      i_gp_reg_in,
                                                                            const unsigned int                      i_gp_reg_out,
                                                                            const unsigned int                      i_gp_reg_m_loop,
                                                                            const unsigned int                      i_gp_reg_n_loop,
                                                                            const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                            const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni4t_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                             libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                             const unsigned int                      i_gp_reg_in,
                                                                             const unsigned int                      i_gp_reg_out,
                                                                             const unsigned int                      i_gp_reg_m_loop,
                                                                             const unsigned int                      i_gp_reg_n_loop,
                                                                             const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                             const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni8_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                            libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                            const unsigned int                      i_gp_reg_in,
                                                                            const unsigned int                      i_gp_reg_out,
                                                                            const unsigned int                      i_gp_reg_m_loop,
                                                                            const unsigned int                      i_gp_reg_n_loop,
                                                                            const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                            const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                            const unsigned int                      i_pad_vnni );
LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni8_to_norm_08bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni8_to_norm_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                            libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                            const unsigned int                      i_gp_reg_in,
                                                                            const unsigned int                      i_gp_reg_out,
                                                                            const unsigned int                      i_gp_reg_m_loop,
                                                                            const unsigned int                      i_gp_reg_n_loop,
                                                                            const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                            const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni8_to_vnni8t_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                             libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                             const unsigned int                      i_gp_reg_in,
                                                                             const unsigned int                      i_gp_reg_out,
                                                                             const unsigned int                      i_gp_reg_m_loop,
                                                                             const unsigned int                      i_gp_reg_n_loop,
                                                                             const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                             const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_padnm_mod2_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                              libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                              const unsigned int                      i_gp_reg_in,
                                                                              const unsigned int                      i_gp_reg_out,
                                                                              const unsigned int                      i_gp_reg_m_loop,
                                                                              const unsigned int                      i_gp_reg_n_loop,
                                                                              const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                              const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_padnm_mod4_mbit_scalar_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                              libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                              const unsigned int                      i_gp_reg_in,
                                                                              const unsigned int                      i_gp_reg_out,
                                                                              const unsigned int                      i_gp_reg_m_loop,
                                                                              const unsigned int                      i_gp_reg_n_loop,
                                                                              const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                              const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_128bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
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
void libxsmm_generator_transform_norm_to_normt_16bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_08bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc );


LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni2_16bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                      const unsigned int                      i_pad_vnni );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni2_to_vnni2t_16bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                        libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                        const unsigned int                      i_gp_reg_in,
                                                                        const unsigned int                      i_gp_reg_out,
                                                                        const unsigned int                      i_gp_reg_m_loop,
                                                                        const unsigned int                      i_gp_reg_n_loop,
                                                                        const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                        const libxsmm_meltw_descriptor*         i_mateltwise_desc );
LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni4_16bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                      const unsigned int                      i_pad_vnni );
LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni4_08bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                      const unsigned int                      i_pad_vnni );
LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni4t_16bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                       libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                       const unsigned int                      i_gp_reg_in,
                                                                       const unsigned int                      i_gp_reg_out,
                                                                       const unsigned int                      i_gp_reg_m_loop,
                                                                       const unsigned int                      i_gp_reg_n_loop,
                                                                       const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                       const libxsmm_meltw_descriptor*         i_mateltwise_desc );
LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni4t_08bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                       libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                       const unsigned int                      i_gp_reg_in,
                                                                       const unsigned int                      i_gp_reg_out,
                                                                       const unsigned int                      i_gp_reg_m_loop,
                                                                       const unsigned int                      i_gp_reg_n_loop,
                                                                       const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                       const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni8_16bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                      const unsigned int                      i_pad_vnni );
LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni8_08bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                      const unsigned int                      i_gp_reg_in,
                                                                      const unsigned int                      i_gp_reg_out,
                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                      const unsigned int                      i_pad_vnni );
LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni8_to_vnni8t_16bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                       libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                       const unsigned int                      i_gp_reg_in,
                                                                       const unsigned int                      i_gp_reg_out,
                                                                       const unsigned int                      i_gp_reg_m_loop,
                                                                       const unsigned int                      i_gp_reg_n_loop,
                                                                       const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                       const libxsmm_meltw_descriptor*         i_mateltwise_desc );
LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni8_to_vnni8t_08bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                       libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                       const unsigned int                      i_gp_reg_in,
                                                                       const unsigned int                      i_gp_reg_out,
                                                                       const unsigned int                      i_gp_reg_m_loop,
                                                                       const unsigned int                      i_gp_reg_n_loop,
                                                                       const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                       const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_padnm_mod2_16bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                               libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                               const unsigned int                      i_gp_reg_in,
                                                                               const unsigned int                      i_gp_reg_out,
                                                                               const unsigned int                      i_gp_reg_m_loop,
                                                                               const unsigned int                      i_gp_reg_n_loop,
                                                                               const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                               const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_padnm_mod4_08bit_sse_microkernel( libxsmm_generated_code*                 io_generated_code,
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

