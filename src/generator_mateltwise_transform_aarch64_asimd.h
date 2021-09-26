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

#ifndef GENERATOR_MATELTWISE_TRANSFORM_AARCH64_ASIMD_H
#define GENERATOR_MATELTWISE_TRANSFORM_AARCH64_ASIMD_H

#include "generator_common.h"

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_mbit_scalar_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                      const unsigned int                      i_gp_reg_in,
                                                                                      const unsigned int                      i_gp_reg_out,
                                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni_mbit_scalar_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                     libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                     const unsigned int                      i_gp_reg_in,
                                                                                     const unsigned int                      i_gp_reg_out,
                                                                                     const unsigned int                      i_gp_reg_m_loop,
                                                                                     const unsigned int                      i_gp_reg_n_loop,
                                                                                     const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                     const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                                     const unsigned int                      i_pad_vnni );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni_to_vnnit_mbit_scalar_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                      libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                      const unsigned int                      i_gp_reg_in,
                                                                                      const unsigned int                      i_gp_reg_out,
                                                                                      const unsigned int                      i_gp_reg_m_loop,
                                                                                      const unsigned int                      i_gp_reg_n_loop,
                                                                                      const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                      const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_padnm_mod2_mbit_scalar_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                        libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                        const unsigned int                      i_gp_reg_in,
                                                                                        const unsigned int                      i_gp_reg_out,
                                                                                        const unsigned int                      i_gp_reg_m_loop,
                                                                                        const unsigned int                      i_gp_reg_n_loop,
                                                                                        const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                        const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_64bit_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                const unsigned int                      i_gp_reg_in,
                                                                                const unsigned int                      i_gp_reg_out,
                                                                                const unsigned int                      i_gp_reg_m_loop,
                                                                                const unsigned int                      i_gp_reg_n_loop,
                                                                                const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_32bit_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                const unsigned int                      i_gp_reg_in,
                                                                                const unsigned int                      i_gp_reg_out,
                                                                                const unsigned int                      i_gp_reg_m_loop,
                                                                                const unsigned int                      i_gp_reg_n_loop,
                                                                                const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_16bit_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                const unsigned int                      i_gp_reg_in,
                                                                                const unsigned int                      i_gp_reg_out,
                                                                                const unsigned int                      i_gp_reg_m_loop,
                                                                                const unsigned int                      i_gp_reg_n_loop,
                                                                                const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_normt_08bit_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                const unsigned int                      i_gp_reg_in,
                                                                                const unsigned int                      i_gp_reg_out,
                                                                                const unsigned int                      i_gp_reg_m_loop,
                                                                                const unsigned int                      i_gp_reg_n_loop,
                                                                                const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                const libxsmm_meltw_descriptor*         i_mateltwise_desc );


LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_to_vnni_16bit_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                               libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                               const unsigned int                      i_gp_reg_in,
                                                                               const unsigned int                      i_gp_reg_out,
                                                                               const unsigned int                      i_gp_reg_m_loop,
                                                                               const unsigned int                      i_gp_reg_n_loop,
                                                                               const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                               const libxsmm_meltw_descriptor*         i_mateltwise_desc,
                                                                               const unsigned int                      i_pad_vnni );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_vnni_to_vnnit_16bit_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                const unsigned int                      i_gp_reg_in,
                                                                                const unsigned int                      i_gp_reg_out,
                                                                                const unsigned int                      i_gp_reg_m_loop,
                                                                                const unsigned int                      i_gp_reg_n_loop,
                                                                                const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_norm_padnm_mod2_16bit_aarch64_asimd_microkernel( libxsmm_generated_code*                 io_generated_code,
                                                                                  libxsmm_loop_label_tracker*             io_loop_label_tracker,
                                                                                  const unsigned int                      i_gp_reg_in,
                                                                                  const unsigned int                      i_gp_reg_out,
                                                                                  const unsigned int                      i_gp_reg_m_loop,
                                                                                  const unsigned int                      i_gp_reg_n_loop,
                                                                                  const libxsmm_mateltwise_kernel_config* i_micro_kernel_config,
                                                                                  const libxsmm_meltw_descriptor*         i_mateltwise_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_transform_aarch64_asimd_microkernel( libxsmm_generated_code*                        io_generated_code,
                                                            libxsmm_loop_label_tracker*                    io_loop_label_tracker,
                                                            libxsmm_mateltwise_gp_reg_mapping*             i_gp_reg_mapping,
                                                            const libxsmm_mateltwise_kernel_config*        i_micro_kernel_config,
                                                            const libxsmm_meltw_descriptor*                i_mateltwise_desc );

#endif /* GENERATOR_MATELTWISE_TRANSFORM_AARCH64_ASIMD_H */

