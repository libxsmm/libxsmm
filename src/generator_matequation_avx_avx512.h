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
libxsmm_blasint libxsmm_generator_matequation_x86_valid_arch_precision( libxsmm_generated_code*           io_generated_code,
                                                                        libxsmm_matrix_eqn*               i_eqn,
                                                                        const libxsmm_meqn_descriptor*    i_mateqn_desc);
LIBXSMM_API_INTERN
void libxsmm_generator_matequation_apply_fusion_pattern_transformation(libxsmm_meqn_fusion_pattern_type fusion_pattern,
                                               libxsmm_meqn_elem                *cur_node,
                                               libxsmm_meqn_elem                *new_arg_node,
                                               unsigned int                           *timestamp,
                                               unsigned int                           last_timestamp );

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_apply_xgemm_fusion_pattern_transformation(libxsmm_meqn_fusion_pattern_type fusion_pattern,
                                               libxsmm_meqn_elem                *cur_node,
                                               libxsmm_meqn_elem                *new_arg_node,
                                               unsigned int                           *timestamp,
                                               unsigned int                           last_timestamp );

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_apply_gather_fusion_pattern_transformation(libxsmm_meqn_fusion_pattern_type fusion_pattern,
                                               libxsmm_meqn_elem                *cur_node,
                                               libxsmm_meqn_elem                *new_arg_node,
                                               unsigned int                           *timestamp,
                                               unsigned int                           last_timestamp );

LIBXSMM_API_INTERN
int libxsmm_generator_matequation_find_in_pos_for_colbias(libxsmm_meqn_elem *colbias_add_node);

LIBXSMM_API_INTERN
libxsmm_datatype libxsmm_generator_matequation_find_dtype_for_colbias(libxsmm_meqn_elem *colbias_add_node);

LIBXSMM_API_INTERN
libxsmm_meqn_fusion_pattern_type libxsmm_generator_matequation_find_fusion_pattern_with_ancestors(libxsmm_meqn_elem *cur_node, libxsmm_meqn_fusion_knobs *fusion_knobs);

LIBXSMM_API_INTERN
libxsmm_meqn_fusion_pattern_type libxsmm_generator_matequation_find_xgemm_fusion_pattern_with_ancestors(libxsmm_meqn_elem *xgemm_node);

LIBXSMM_API_INTERN
libxsmm_meqn_fusion_pattern_type libxsmm_generator_matequation_find_gather_fusion_pattern_with_ancestors(libxsmm_meqn_elem *gather_node);

LIBXSMM_API_INTERN
int libxsmm_generator_matequation_is_gather_node(libxsmm_meqn_elem  *cur_node);

LIBXSMM_API_INTERN
int libxsmm_generator_matequation_is_xgemm_node_supporting_fusion(libxsmm_meqn_elem  *xgemm_node);

LIBXSMM_API_INTERN
int libxsmm_generator_matequation_is_xgemm_node(libxsmm_meqn_elem  *cur_node);

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
libxsmm_meqn_elem* libxsmm_generator_matequation_find_op_at_timestamp(libxsmm_meqn_elem* cur_node, libxsmm_blasint timestamp);

LIBXSMM_API_INTERN
int libxsmm_generator_matequation_is_eqn_node_breaking_point(libxsmm_meqn_elem *node, libxsmm_meqn_fusion_knobs *fusion_knobs);

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_enqueue_equation(libxsmm_matrix_eqn *eqn, libxsmm_matrix_eqn **jiting_queue, unsigned int *queue_size);

LIBXSMM_API_INTERN
void libxsmm_generator_decompose_equation_tree_x86( libxsmm_matrix_eqn *eqn, libxsmm_matrix_eqn **jiting_queue, unsigned int *queue_size, libxsmm_meqn_fusion_knobs *fusion_knobs );

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_avx_avx512_kernel( libxsmm_generated_code*        io_generated_code,
                                                      const libxsmm_meqn_descriptor* i_mateqn_desc );

LIBXSMM_API_INTERN int libxsmm_generator_matequation_is_unary_with_bcast(libxsmm_bitfield flags);
LIBXSMM_API_INTERN int libxsmm_generator_matequation_is_binary_with_bcast(libxsmm_bitfield flags);
LIBXSMM_API_INTERN int libxsmm_generator_matequation_is_ternary_with_bcast(libxsmm_bitfield flags);
LIBXSMM_API_INTERN int libxsmm_generator_matequation_is_unary_bcast_arg_an_inputarg(libxsmm_bitfield flags, libxsmm_meqn_elem *node);
LIBXSMM_API_INTERN int libxsmm_generator_matequation_is_binary_bcast_arg_an_inputarg(libxsmm_bitfield flags, libxsmm_meqn_elem *node);
LIBXSMM_API_INTERN int libxsmm_generator_matequation_is_ternary_bcast_arg_an_inputarg(libxsmm_bitfield flags, libxsmm_meqn_elem *node);
LIBXSMM_API_INTERN void libxsmm_meqn_are_nodes_pure_f32(libxsmm_meqn_elem *node, unsigned int *result);

#endif /* GENERATOR_MATEQUATION_AVX_AVX512_H */

