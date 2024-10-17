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
#ifndef GENERATOR_MATEQUATION_SCRATCH_AARCH64_H
#define GENERATOR_MATEQUATION_SCRATCH_AARCH64_H

#include "generator_common.h"


LIBXSMM_API_INTERN
void libxsmm_generator_matequation_set_input_in_stack_param_struct_aarch64( libxsmm_generated_code*   io_generated_code,
    libxsmm_matequation_kernel_config*                  i_micro_kernel_config,
    libxsmm_matequation_gp_reg_mapping*                 i_gp_reg_mapping,
    libxsmm_meqn_elem*                            cur_node,
    unsigned int                                        temp_reg,
    unsigned int                                        ptr_id );

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_set_output_in_stack_param_struct_aarch64(libxsmm_generated_code*   io_generated_code,
    libxsmm_matequation_kernel_config*                  i_micro_kernel_config,
    libxsmm_matequation_gp_reg_mapping*                 i_gp_reg_mapping,
    libxsmm_meqn_elem*                            cur_node,
    unsigned int                                        temp_reg,
    unsigned int                                        is_last_op );

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_tmp_stack_scratch_aarch64_kernel( libxsmm_generated_code* io_generated_code,
    const libxsmm_meqn_descriptor*          i_mateqn_desc,
    libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
    libxsmm_matequation_kernel_config*      i_micro_kernel_config,
    libxsmm_loop_label_tracker*             io_loop_label_tracker,
    libxsmm_matrix_eqn*                     eqn );

#endif
