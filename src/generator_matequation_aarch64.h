/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evanelos Georganas (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_MATEQUATION_AARCH64_H
#define GENERATOR_MATEQUATION_AARCH64_H

#include "generator_common.h"

#define JIT_STRATEGY_USING_TMP_SCRATCH_BLOCKS   0
#define JIT_STRATEGY_USING_TMP_REGISTER_BLOCKS  1
#define JIT_STRATEGY_HYBRID 2
#define M_ADJUSTMENT 0
#define N_ADJUSTMENT 1
#define UNARY_OP_POOL 0
#define BINARY_OP_POOL 1

LIBXSMM_API_INTERN
libxsmm_blasint libxsmm_generator_matequation_aarch64_valid_arch_precision( libxsmm_generated_code*           io_generated_code,
                                                                        libxsmm_matrix_eqn*               i_eqn,
                                                                        const libxsmm_meqn_descriptor*    i_mateqn_desc);

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_aarch64_init_micro_kernel_config( libxsmm_generated_code*         io_generated_code,
    libxsmm_matequation_kernel_config*    io_micro_kernel_config);

LIBXSMM_API_INTERN
int libxsmm_generator_mateqn_get_fp_relative_offset( libxsmm_meqn_stack_var stack_var );

LIBXSMM_API_INTERN
void libxsmm_generator_meqn_getaddr_stack_var_aarch64( libxsmm_generated_code*              io_generated_code,
                                                libxsmm_meqn_stack_var              stack_var,
                                                unsigned int                        i_gp_reg );
LIBXSMM_API_INTERN
void libxsmm_generator_meqn_getval_stack_var_aarch64( libxsmm_generated_code*               io_generated_code,
                                                libxsmm_meqn_stack_var              stack_var,
                                                unsigned int                        i_gp_reg );
LIBXSMM_API_INTERN
void libxsmm_generator_meqn_getaddr_stack_tmp_i_aarch64( libxsmm_generated_code*            io_generated_code,
                                                unsigned int                        i_tmp_offset_i,
                                                unsigned int                        i_gp_reg_scratch,
                                                unsigned int                        i_gp_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_meqn_getaddr_stack_tmpaddr_i_aarch64( libxsmm_generated_code*            io_generated_code,
                                                unsigned int                        i_tmp_offset_i,
                                                unsigned int                        i_gp_reg_scratch,
                                                unsigned int                        i_gp_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_meqn_setval_stack_var_aarch64( libxsmm_generated_code*               io_generated_code,
                                                libxsmm_meqn_stack_var              stack_var,
                                                unsigned int                        i_aux_reg,
                                                unsigned int                        i_gp_reg );
LIBXSMM_API_INTERN
void libxsmm_generator_matequation_setup_stack_frame_aarch64( libxsmm_generated_code*   io_generated_code,
                                              const libxsmm_meqn_descriptor*                      i_mateqn_desc,
                                              libxsmm_matequation_gp_reg_mapping*                 i_gp_reg_mapping,
                                              libxsmm_matequation_kernel_config*                  i_micro_kernel_config,
                                              libxsmm_matrix_eqn*                                 i_eqn,
                                              unsigned int                                        i_strategy );

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_destroy_stack_frame_aarch64( libxsmm_generated_code*                   io_generated_code,
                                              libxsmm_matequation_kernel_config*                  i_micro_kernel_config,
                                              libxsmm_matequation_gp_reg_mapping*                 i_gp_reg_mapping,
                                              unsigned int                                        i_strategy  );
LIBXSMM_API_INTERN
void libxsmm_generator_decompose_equation_tree_aarch64( libxsmm_matrix_eqn *eqn, libxsmm_matrix_eqn **jiting_queue, unsigned int *queue_size, libxsmm_meqn_fusion_knobs *fusion_knobs );

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_aarch64_kernel( libxsmm_generated_code*        io_generated_code,
                                                      const libxsmm_meqn_descriptor* i_mateqn_desc);

#endif /* GENERATOR_MATEQUATION_AARCH64_H */

