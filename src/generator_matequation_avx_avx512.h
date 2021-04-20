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

#ifndef GENERATOR_MATEQUATION_AVX_AVX512_H
#define GENERATOR_MATEQUATION_AVX_AVX512_H

#include "generator_common.h"

#define JIT_STRATEGY_USING_TMP_SCRATCH_BLOCKS   0
#define JIT_STRATEGY_USING_TMP_REGISTER_BLOCKS  1
#define JIT_STRATEGY_HYBRID 2
#define M_ADJUSTMENT 0
#define N_ADJUSTMENT 1
#define UNARY_OP_POOL 0
#define BINARY_OP_POOL 1

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_init_micro_kernel_config( libxsmm_generated_code*         io_generated_code,
    libxsmm_matequation_kernel_config*    io_micro_kernel_config);

LIBXSMM_API_INTERN
int libxsmm_generator_mateqn_get_rbp_relative_offset( libxsmm_meqn_stack_var stack_var );

LIBXSMM_API_INTERN
void libxsmm_generator_meqn_getaddr_stack_var( libxsmm_generated_code*              io_generated_code,
                                                libxsmm_meqn_stack_var              stack_var,
                                                unsigned int                        i_gp_reg );
LIBXSMM_API_INTERN
void libxsmm_generator_meqn_getval_stack_var( libxsmm_generated_code*               io_generated_code,
                                                libxsmm_meqn_stack_var              stack_var,
                                                unsigned int                        i_gp_reg );
LIBXSMM_API_INTERN
void libxsmm_generator_meqn_getaddr_stack_tmp_i( libxsmm_generated_code*            io_generated_code,
                                                unsigned int                        i_tmp_offset_i,
                                                unsigned int                        i_gp_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_meqn_getaddr_stack_tmpaddr_i( libxsmm_generated_code*            io_generated_code,
                                                unsigned int                        i_tmp_offset_i,
                                                unsigned int                        i_gp_reg );

LIBXSMM_API_INTERN
void libxsmm_generator_meqn_setval_stack_var( libxsmm_generated_code*               io_generated_code,
                                                libxsmm_meqn_stack_var              stack_var,
                                                unsigned int                        i_gp_reg );
LIBXSMM_API_INTERN
void libxsmm_generator_matequation_setup_stack_frame( libxsmm_generated_code*   io_generated_code,
                                              const libxsmm_meqn_descriptor*                      i_mateqn_desc,
                                              libxsmm_matequation_gp_reg_mapping*                 i_gp_reg_mapping,
                                              libxsmm_matequation_kernel_config*                  i_micro_kernel_config,
                                              libxsmm_matrix_eqn*                                 i_eqn,
                                              unsigned int                                        i_strategy );

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_destroy_stack_frame( libxsmm_generated_code*                   io_generated_code,
                                              libxsmm_matequation_kernel_config*                  i_micro_kernel_config,
                                              libxsmm_matequation_gp_reg_mapping*                 i_gp_reg_mapping,
                                              unsigned int                                        i_strategy  );
LIBXSMM_API_INTERN
libxsmm_matrix_eqn_elem* find_op_at_timestamp(libxsmm_matrix_eqn_elem* cur_node, libxsmm_blasint timestamp);

LIBXSMM_API_INTERN
int is_eqn_node_breaking_point(libxsmm_matrix_eqn_elem *node);

LIBXSMM_API_INTERN
void enqueue_equation(libxsmm_matrix_eqn *eqn, libxsmm_matrix_eqn **jiting_queue, unsigned int *queue_size);

LIBXSMM_API_INTERN
void libxsmm_generator_decompose_equation_tree( libxsmm_matrix_eqn *eqn, libxsmm_matrix_eqn **jiting_queue, unsigned int *queue_size );

LIBXSMM_API_INTERN
void libxsmm_generator_assign_new_timestamp(libxsmm_matrix_eqn_elem* cur_node, libxsmm_blasint *current_timestamp );

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_assign_timestamps(libxsmm_matrix_eqn *eqn);

LIBXSMM_API_INTERN
void libxsmm_generator_reoptimize_eqn(libxsmm_matrix_eqn *eqn);

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_avx_avx512_kernel( libxsmm_generated_code*        io_generated_code,
                                                      const libxsmm_meqn_descriptor* i_mateqn_desc );

#endif /* GENERATOR_MATEQUATION_AVX_AVX512_H */

