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
#ifndef LIBXSMM_MATRIXEQN_H
#define LIBXSMM_MATRIXEQN_H

#include <libxsmm.h>

#define LEFT 0
#define RIGHT 1
#define RIGHT2 2


LIBXSMM_EXTERN_C typedef enum libxsmm_meqn_node_type {
  LIBXSMM_MATRIX_EQN_NODE_NONE    = 0,
  LIBXSMM_MATRIX_EQN_NODE_UNARY   = 1,
  LIBXSMM_MATRIX_EQN_NODE_BINARY  = 2,
  LIBXSMM_MATRIX_EQN_NODE_TERNARY = 4,
  LIBXSMM_MATRIX_EQN_NODE_ARG     = 8
} libxsmm_meqn_node_type;

LIBXSMM_EXTERN_C typedef enum libxsmm_meqn_fusion_pattern_type {
  LIBXSMM_MATRIX_EQN_FUSION_PATTERN_NONE                    = 0,
  LIBXSMM_MATRIX_EQN_FUSION_PATTERN_XGEMM_UNARY             = 1,
  LIBXSMM_MATRIX_EQN_FUSION_PATTERN_XGEMM_COLBIAS_ADD       = 2,
  LIBXSMM_MATRIX_EQN_FUSION_PATTERN_XGEMM_COLBIAS_ADD_UNARY = 3,
  LIBXSMM_MATRIX_EQN_FUSION_PATTERN_GATHER_COLS_REDUCE_COLS = 4
} libxsmm_meqn_fusion_pattern_type;

LIBXSMM_EXTERN_C typedef enum libxsmm_meqn_bcast_type {
  LIBXSMM_MATRIX_EQN_BCAST_TYPE_NONE   = 0,
  LIBXSMM_MATRIX_EQN_BCAST_TYPE_ROW    = 1,
  LIBXSMM_MATRIX_EQN_BCAST_TYPE_COL    = 2,
  LIBXSMM_MATRIX_EQN_BCAST_TYPE_SCALAR = 4
} libxsmm_meqn_bcast_type;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_MAY_ALIAS libxsmm_meqn_unary_op {
  libxsmm_meltw_unary_type  type;
  libxsmm_bitfield          flags;
  libxsmm_datatype          dtype;
  libxsmm_blasint           op_arg_pos;
} libxsmm_meqn_unary_op;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_MAY_ALIAS libxsmm_meqn_binary_op {
  libxsmm_meltw_binary_type         type;
  libxsmm_bitfield                  flags;
  libxsmm_datatype                  dtype;
  libxsmm_blasint                   op_arg_pos;
  libxsmm_blasint                   is_matmul;
  libxsmm_blasint                   is_brgemm;
} libxsmm_meqn_binary_op;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_MAY_ALIAS libxsmm_meqn_ternary_op {
  libxsmm_meltw_ternary_type        type;
  libxsmm_bitfield                  flags;
  libxsmm_datatype                  dtype;
  libxsmm_blasint                   op_arg_pos;
  libxsmm_blasint                   is_matmul;
  libxsmm_blasint                   is_brgemm;
} libxsmm_meqn_ternary_op;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_MAY_ALIAS libxsmm_meqn_arg {
  libxsmm_blasint  m;
  libxsmm_blasint  n;
  libxsmm_blasint  ld;
  libxsmm_blasint  in_pos;
  libxsmm_blasint  offs_in_pos;
  libxsmm_datatype dtype;
  libxsmm_meqn_bcast_type   bcast_type;
  libxsmm_matrix_arg_attributes   arg_attr;
} libxsmm_meqn_arg;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_MAY_ALIAS libxsmm_meqn_tmp_info {
  libxsmm_blasint  id;
  libxsmm_blasint  m;
  libxsmm_blasint  n;
  libxsmm_blasint  ld;
  libxsmm_datatype dtype;
  libxsmm_meqn_bcast_type  bcast_type;
  libxsmm_blasint  m_s;
  libxsmm_blasint  n_s;
  libxsmm_blasint  ld_s;
  libxsmm_datatype dtype_s;
  libxsmm_meqn_bcast_type  bcast_type_s;
  libxsmm_blasint  m_t;
  libxsmm_blasint  n_t;
  libxsmm_blasint  ld_t;
  libxsmm_datatype dtype_t;
  libxsmm_meqn_bcast_type  bcast_type_t;
} libxsmm_meqn_tmp_info;

LIBXSMM_EXTERN_C typedef union libxsmm_meqn_info {
  libxsmm_meqn_unary_op   u_op;
  libxsmm_meqn_binary_op  b_op;
  libxsmm_meqn_ternary_op t_op;
  libxsmm_meqn_arg     arg;
} libxsmm_meqn_info;

LIBXSMM_EXTERN_C typedef struct libxsmm_meqn_xgemm_fusion_info {
  libxsmm_blasint   fused_sigmoid_op;
  libxsmm_blasint   fused_relu_op;
  libxsmm_blasint   fused_colbias_add_op;
  libxsmm_blasint   colbias_pos_in_arg;
  libxsmm_datatype  colbias_dtype;
} libxsmm_meqn_xgemm_fusion_info;

LIBXSMM_EXTERN_C typedef struct libxsmm_meqn_gather_fusion_info {
  libxsmm_blasint   fused_reduce_cols_add;
  libxsmm_blasint   fused_reduce_cols_max;
  libxsmm_blasint   idx_array_pos_in_arg;
  libxsmm_datatype  idx_dtype;
} libxsmm_meqn_gather_fusion_info;

LIBXSMM_EXTERN_C typedef struct libxsmm_meqn_fusion_knobs {
  libxsmm_blasint   may_fuse_xgemm;
} libxsmm_meqn_fusion_knobs;

LIBXSMM_EXTERN_C typedef union libxsmm_meqn_fusion_info {
  libxsmm_meqn_xgemm_fusion_info   xgemm;
  libxsmm_meqn_gather_fusion_info  gather;
} libxsmm_meqn_fusion_info;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_MAY_ALIAS libxsmm_meqn_elem {
  struct libxsmm_meqn_elem* le;
  struct libxsmm_meqn_elem* ri;
  struct libxsmm_meqn_elem* r2;
  struct libxsmm_meqn_elem* up;
  libxsmm_meqn_node_type    type;
  libxsmm_meqn_info         info;
  libxsmm_blasint                 reg_score;
  libxsmm_blasint                 visit_timestamp;
  libxsmm_meqn_tmp_info     tmp;
  libxsmm_blasint                 max_tmp_size;
  libxsmm_blasint                 n_args;
  libxsmm_blasint                 tree_max_comp_tsize;
  libxsmm_meqn_fusion_info  fusion_info;
} libxsmm_meqn_elem;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_MAY_ALIAS libxsmm_matrix_eqn {
  libxsmm_meqn_elem*        eqn_root;
  libxsmm_meqn_elem*        eqn_cur;
  libxsmm_blasint                 is_constructed;
  libxsmm_blasint                 is_optimized;
  libxsmm_blasint                 unary_only;
  libxsmm_blasint                 binary_only;
} libxsmm_matrix_eqn;

/* Helper functions for matrix equation handling */
LIBXSMM_API_INTERN libxsmm_matrix_eqn* libxsmm_meqn_get_equation( libxsmm_blasint eqn_idx );
LIBXSMM_API_INTERN int libxsmm_meqn_is_ready_for_jit( libxsmm_blasint eqn_idx );
LIBXSMM_API_INTERN void libxsmm_meqn_propagate_tmp_info( libxsmm_meqn_elem* cur_node );
LIBXSMM_API_INTERN void libxsmm_meqn_assign_reg_scores( libxsmm_meqn_elem* cur_node );
LIBXSMM_API_INTERN void libxsmm_meqn_create_exec_plan( libxsmm_meqn_elem* cur_node, libxsmm_blasint *global_timestamp, libxsmm_blasint n_max_tmp, libxsmm_blasint *tmp_storage_pool );
LIBXSMM_API_INTERN libxsmm_blasint libxsmm_meqn_reserve_tmp_storage(libxsmm_blasint n_max_tmp, libxsmm_blasint *tmp_storage_pool);
LIBXSMM_API_INTERN void libxsmm_meqn_assign_new_timestamp(libxsmm_meqn_elem* cur_node, libxsmm_blasint *current_timestamp );
LIBXSMM_API_INTERN void libxsmm_meqn_assign_timestamps(libxsmm_matrix_eqn *eqn);
LIBXSMM_API_INTERN void libxsmm_meqn_reoptimize(libxsmm_matrix_eqn *eqn);
LIBXSMM_API_INTERN void libxsmm_meqn_adjust_tmp_sizes( libxsmm_meqn_elem* cur_node );
LIBXSMM_API_INTERN int libxsmm_meqn_is_unary_opcode_reduce_cols_idx_kernel (unsigned int opcode);
LIBXSMM_API_INTERN int libxsmm_meqn_is_unary_opcode_reduce_kernel (unsigned int opcode);
LIBXSMM_API_INTERN int libxsmm_meqn_is_unary_opcode_transform_kernel (unsigned int opcode);
LIBXSMM_API_INTERN int libxsmm_meqn_is_unary_opcode_reduce_to_scalar (unsigned int opcode);
LIBXSMM_API_INTERN int libxsmm_meqn_is_binary_opcode_reduce_to_scalar (unsigned int opcode);

LIBXSMM_API_INTERN void libxsmm_meqn_tree_contains_opcode(libxsmm_meqn_elem *node, libxsmm_meltw_unary_type u_opcode, libxsmm_meltw_binary_type b_opcode, libxsmm_meltw_ternary_type t_opcode, unsigned int *result);

LIBXSMM_API_INTERN unsigned int libxsmm_meqn_contains_opcode(libxsmm_matrix_eqn *eqn, libxsmm_meltw_unary_type u_opcode, libxsmm_meltw_binary_type b_opcode, libxsmm_meltw_ternary_type t_opcode );

LIBXSMM_API_INTERN void libxsmm_meqn_tree_all_nodes_dtype(libxsmm_meqn_elem *node, libxsmm_datatype dtype, unsigned int *result);

LIBXSMM_API_INTERN unsigned int libxsmm_meqn_all_nodes_dtype(libxsmm_matrix_eqn *eqn, libxsmm_datatype dtype);

LIBXSMM_API_INTERN void libxsmm_meqn_tree_all_args_dtype(libxsmm_meqn_elem *node, libxsmm_datatype dtype, unsigned int *result);

LIBXSMM_API_INTERN unsigned int libxsmm_meqn_all_args_dtype(libxsmm_matrix_eqn *eqn, libxsmm_datatype dtype);

LIBXSMM_API_INTERN void libxsmm_meqn_tree_any_args_dtype(libxsmm_meqn_elem *node, libxsmm_datatype dtype, unsigned int *result);

LIBXSMM_API_INTERN unsigned int libxsmm_meqn_any_args_dtype(libxsmm_matrix_eqn *eqn, libxsmm_datatype dtype);

LIBXSMM_API_INTERN void libxsmm_meqn_are_nodes_pure_f32(libxsmm_meqn_elem *node, unsigned int *result);

LIBXSMM_API_INTERN
libxsmm_meqn_bcast_type libxsmm_meqn_get_bcast_type_unary(libxsmm_bitfield flags);

LIBXSMM_API_INTERN
libxsmm_meqn_bcast_type libxsmm_meqn_get_bcast_type_binary(libxsmm_bitfield flags, unsigned int side);

LIBXSMM_API_INTERN
libxsmm_meqn_bcast_type libxsmm_meqn_get_bcast_type_ternary(libxsmm_bitfield flags, unsigned int side);

LIBXSMM_API_INTERN void libxsmm_meqn_reassign_bcast_tmp(libxsmm_matrix_eqn *eqn);
LIBXSMM_API_INTERN void libxsmm_meqn_reassign_children_bcast_tmp(libxsmm_matrix_eqn *eqn, libxsmm_meqn_elem* cur_node);

LIBXSMM_API_INTERN void libxsmm_meqn_trv_dbg_print( libxsmm_meqn_elem* cur_node, libxsmm_blasint indent );
#endif /*LIBXSMM_MATRIXEQN_H*/
