/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_MATELTWISE_TRANSFORM_AARCH64_SVE_H
#define GENERATOR_MATELTWISE_TRANSFORM_AARCH64_SVE_H

#include "generator_aarch64_instructions.h"
#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni2_16bit_aarch64_sve_microkernel_block( libxsmm_generated_code*                 io_generated_code,
                                                                                    libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                    const unsigned int                      i_gp_reg_in,
                                                                                    const unsigned int                      i_gp_reg_out,
                                                                                    const unsigned int                      i_gp_reg_m_loop,
                                                                                    const unsigned int                      i_gp_reg_n_loop,
                                                                                    const unsigned int                      i_gp_reg_scratch,
                                                                                    const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                    const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                                    unsigned int                            i_m_remainder,
                                                                                    unsigned int                            i_m_mask_in,
                                                                                    unsigned int                            i_m_mask_out,
                                                                                    unsigned int                            i_n_padding_vregs );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni2_16bit_aarch64_sve_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                              libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                              const unsigned int                      i_gp_reg_in,
                                                                              const unsigned int                      i_gp_reg_out,
                                                                              const unsigned int                      i_gp_reg_m_loop,
                                                                              const unsigned int                      i_gp_reg_n_loop,
                                                                              const unsigned int                      i_gp_reg_scratch,
                                                                              const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                              const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                              const unsigned int                      i_pad_vnni );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni4_16bit_aarch64_sve_microkernel_block( libxsmm_generated_code*                 io_generated_code,
                                                                                    libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                    const unsigned int                      i_gp_reg_in,
                                                                                    const unsigned int                      i_gp_reg_out,
                                                                                    const unsigned int                      i_gp_reg_m_loop,
                                                                                    const unsigned int                      i_gp_reg_n_loop,
                                                                                    const unsigned int                      i_gp_reg_scratch,
                                                                                    const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                    const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                                    unsigned int                            i_m_remainder,
                                                                                    unsigned int                            i_m_mask_in,
                                                                                    unsigned int                            i_m_mask_out,
                                                                                    unsigned int                            i_n_padding_vregs );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni4_16bit_aarch64_sve_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                              libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                              const unsigned int                      i_gp_reg_in,
                                                                              const unsigned int                      i_gp_reg_out,
                                                                              const unsigned int                      i_gp_reg_m_loop,
                                                                              const unsigned int                      i_gp_reg_n_loop,
                                                                              const unsigned int                      i_gp_reg_scratch,
                                                                              const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                              const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                              const unsigned int                      i_pad_vnni );


LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni2_to_vnni2t_Nmod8_16bit_aarch64_sve_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                      const unsigned int                      i_gp_reg_in,
                                                                                      const unsigned int                      i_gp_reg_out,
                                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                                      const unsigned int                      i_gp_reg_scratch,
                                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni2_to_vnni2t_16bit_aarch64_sve_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                const unsigned int                      i_gp_reg_in,
                                                                                const unsigned int                      i_gp_reg_out,
                                                                                const unsigned int                      i_gp_reg_m_loop,
                                                                                const unsigned int                      i_gp_reg_n_loop,
                                                                                const unsigned int                      i_gp_reg_scratch,
                                                                                const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni4t_Nmod16_16bit_aarch64_sve_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                       libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                       const unsigned int                      i_gp_reg_in,
                                                                                       const unsigned int                      i_gp_reg_out,
                                                                                       const unsigned int                      i_gp_reg_m_loop,
                                                                                       const unsigned int                      i_gp_reg_n_loop,
                                                                                       const unsigned int                      i_gp_reg_scratch,
                                                                                       const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                       const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni4_to_vnni4t_16bit_aarch64_sve_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                const unsigned int                      i_gp_reg_in,
                                                                                const unsigned int                      i_gp_reg_out,
                                                                                const unsigned int                      i_gp_reg_m_loop,
                                                                                const unsigned int                      i_gp_reg_n_loop,
                                                                                const unsigned int                      i_gp_reg_scratch,
                                                                                const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_aarch64_sve_microkernel( libxsmm_generated_code*                        io_generated_code,
                                                          libxsmm_loop_label_tracker*                    io_loop_label_tracker,
                                                          libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
                                                          const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                                          const libxsmm_meltw_descriptor*                i_mateltwise_desc );

#endif /* GENERATOR_MATELTWISE_TRANSFORM_AARCH64_SVE_H */
