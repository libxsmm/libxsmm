/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Antonio Noack (Friedrich Schiller University Jena)
******************************************************************************/
#ifndef GENERATOR_MATELTWISE_AARCH64_SVE_H
#define GENERATOR_MATELTWISE_AARCH64_SVE_H

#include "generator_common.h"
#include "generator_aarch64_instructions.h"

LIBXSMM_API_INTERN
libxsmm_aarch64_sve_type libxsmm_generator_aarch64_get_sve_type(unsigned char i_size);

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_aarch64_sve_kernel( libxsmm_generated_code*         io_generated_code,
                                                      const libxsmm_meltw_descriptor* i_mateltw_desc );

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_aarch64_sve_init_micro_kernel_config_fullvector( libxsmm_generated_code*           io_generated_code,
                                                                                   libxsmm_mateltwise_kernel_config* io_micro_kernel_config,
                                                                                   const libxsmm_meltw_descriptor*   i_mateltwise_desc);

LIBXSMM_API_INTERN
void libxsmm_generator_meltw_setup_stack_frame_aarch64_sve( libxsmm_generated_code*            io_generated_code,
                                                            const libxsmm_meltw_descriptor*      i_mateltwise_desc,
                                                            libxsmm_mateltwise_gp_reg_mapping*   i_gp_reg_mapping,
                                                            libxsmm_mateltwise_kernel_config*    i_micro_kernel_config);

LIBXSMM_API_INTERN
void libxsmm_generator_meltw_destroy_stack_frame_aarch64_sve( libxsmm_generated_code*            io_generated_code,
                                                              const libxsmm_meltw_descriptor*     i_mateltwise_desc,
                                                              const libxsmm_mateltwise_kernel_config*  i_micro_kernel_config );

#endif /* GENERATOR_MATELTWISE_AARCH64_SVE_H */
