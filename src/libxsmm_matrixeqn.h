/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_MATRIXEQN_H
#define LIBXSMM_MATRIXEQN_H

#define LEFT 0
#define RIGHT 1
#define RIGHT2 2

#include <libxsmm.h>
/**
 * TF includes src/libxsmm_main.h and uses LIBXSMM's sync primitives
 * without including libxsmm_sync. However, libxsmm_sync.h shall be
 * an explicit include separate from including libxsmm.h.
 */
#include "libxsmm_sync.h"

LIBXSMM_EXTERN_C typedef enum libxsmm_matrix_eqn_node_type {
  LIBXSMM_MATRIX_EQN_NODE_NONE    = 0,
  LIBXSMM_MATRIX_EQN_NODE_UNARY   = 1,
  LIBXSMM_MATRIX_EQN_NODE_BINARY  = 2,
  LIBXSMM_MATRIX_EQN_NODE_TERNARY = 4,
  LIBXSMM_MATRIX_EQN_NODE_ARG     = 8
} libxsmm_matrix_eqn_node_type;

LIBXSMM_EXTERN_C typedef enum libxsmm_matrix_eqn_fusion_pattern_type {
  LIBXSMM_MATRIX_EQN_FUSION_PATTERN_NONE                    = 0,
  LIBXSMM_MATRIX_EQN_FUSION_PATTERN_XGEMM_UNARY             = 1,
  LIBXSMM_MATRIX_EQN_FUSION_PATTERN_XGEMM_COLBIAS_ADD       = 2,
  LIBXSMM_MATRIX_EQN_FUSION_PATTERN_XGEMM_COLBIAS_ADD_UNARY = 3
} libxsmm_matrix_eqn_fusion_pattern_type;

LIBXSMM_EXTERN_C typedef enum libxsmm_matrix_eqn_bcast_type {
  LIBXSMM_MATRIX_EQN_BCAST_TYPE_NONE   = 0,
  LIBXSMM_MATRIX_EQN_BCAST_TYPE_ROW    = 1,
  LIBXSMM_MATRIX_EQN_BCAST_TYPE_COL    = 2,
  LIBXSMM_MATRIX_EQN_BCAST_TYPE_SCALAR = 4
} libxsmm_matrix_eqn_bcast_type;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE LIBXSMM_MAY_ALIAS libxsmm_matrix_eqn_unary_op {
  libxsmm_meltw_unary_type  type;
  libxsmm_bitfield          flags;
  libxsmm_datatype          dtype;
  libxsmm_blasint           op_arg_pos;
} libxsmm_matrix_eqn_unary_op;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE LIBXSMM_MAY_ALIAS libxsmm_matrix_eqn_binary_op {
  libxsmm_meltw_binary_type         type;
  libxsmm_bitfield                  flags;
  libxsmm_datatype                  dtype;
  libxsmm_blasint                   op_arg_pos;
  libxsmm_blasint                   is_matmul;
  libxsmm_blasint                   is_brgemm;
} libxsmm_matrix_eqn_binary_op;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE LIBXSMM_MAY_ALIAS libxsmm_matrix_eqn_ternary_op {
  libxsmm_meltw_ternary_type        type;
  libxsmm_bitfield                  flags;
  libxsmm_datatype                  dtype;
  libxsmm_blasint                   op_arg_pos;
  libxsmm_blasint                   is_matmul;
  libxsmm_blasint                   is_brgemm;
} libxsmm_matrix_eqn_ternary_op;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE LIBXSMM_MAY_ALIAS libxsmm_matrix_eqn_arg {
  libxsmm_blasint  m;
  libxsmm_blasint  n;
  libxsmm_blasint  ld;
  libxsmm_blasint  in_pos;
  libxsmm_blasint  offs_in_pos;
  libxsmm_datatype dtype;
  libxsmm_matrix_eqn_bcast_type  bcast_type;
} libxsmm_matrix_eqn_arg;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE LIBXSMM_MAY_ALIAS libxsmm_matrix_eqn_arg_v2 {
  libxsmm_blasint  m;
  libxsmm_blasint  n;
  libxsmm_blasint  ld;
  libxsmm_blasint  in_pos;
  libxsmm_blasint  offs_in_pos;
  libxsmm_datatype dtype;
  libxsmm_matrix_eqn_bcast_type   bcast_type;
  libxsmm_matrix_arg_attributes   arg_attr;
} libxsmm_matrix_eqn_arg_v2;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE LIBXSMM_MAY_ALIAS libxsmm_matrix_eqn_tmp_info {
  libxsmm_blasint  id;
  libxsmm_blasint  m;
  libxsmm_blasint  n;
  libxsmm_blasint  ld;
  libxsmm_datatype dtype;
  libxsmm_matrix_eqn_bcast_type  bcast_type;
  libxsmm_blasint  m_s;
  libxsmm_blasint  n_s;
  libxsmm_blasint  ld_s;
  libxsmm_datatype dtype_s;
  libxsmm_matrix_eqn_bcast_type  bcast_type_s;
  libxsmm_blasint  m_t;
  libxsmm_blasint  n_t;
  libxsmm_blasint  ld_t;
  libxsmm_datatype dtype_t;
  libxsmm_matrix_eqn_bcast_type  bcast_type_t;
} libxsmm_matrix_eqn_tmp_info;

LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE libxsmm_matrix_eqn_info {
  libxsmm_matrix_eqn_unary_op   u_op;
  libxsmm_matrix_eqn_binary_op  b_op;
  libxsmm_matrix_eqn_ternary_op t_op;
  libxsmm_matrix_eqn_arg_v2     arg;
} libxsmm_matrix_eqn_info;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_matrix_eqn_xgemm_fusion_info {
  libxsmm_blasint   fused_sigmoid_op;
  libxsmm_blasint   fused_relu_op;
  libxsmm_blasint   fused_colbias_add_op;
  libxsmm_blasint   colbias_pos_in_arg;
  libxsmm_datatype  colbias_dtype;
} libxsmm_matrix_eqn_xgemm_fusion_info;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_matrix_eqn_fusion_knobs {
  libxsmm_blasint   may_fuse_xgemm;
} libxsmm_matrix_eqn_fusion_knobs;

LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE libxsmm_matrix_eqn_fusion_info {
  libxsmm_matrix_eqn_xgemm_fusion_info   xgemm;
} libxsmm_matrix_eqn_fusion_info;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE LIBXSMM_MAY_ALIAS libxsmm_matrix_eqn_elem {
  struct libxsmm_matrix_eqn_elem* le;
  struct libxsmm_matrix_eqn_elem* ri;
  struct libxsmm_matrix_eqn_elem* r2;
  struct libxsmm_matrix_eqn_elem* up;
  libxsmm_matrix_eqn_node_type    type;
  libxsmm_matrix_eqn_info         info;
  libxsmm_blasint                 reg_score;
  libxsmm_blasint                 visit_timestamp;
  libxsmm_matrix_eqn_tmp_info     tmp;
  libxsmm_blasint                 max_tmp_size;
  libxsmm_blasint                 n_args;
  libxsmm_blasint                 tree_max_comp_tsize;
  libxsmm_matrix_eqn_fusion_info  fusion_info;
} libxsmm_matrix_eqn_elem;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE LIBXSMM_MAY_ALIAS libxsmm_matrix_eqn {
  libxsmm_matrix_eqn_elem*        eqn_root;
  libxsmm_matrix_eqn_elem*        eqn_cur;
  libxsmm_blasint                 is_constructed;
  libxsmm_blasint                 is_optimized;
  libxsmm_blasint                 unary_only;
  libxsmm_blasint                 binary_only;
} libxsmm_matrix_eqn;

/* Helper functions for matrix equation handling */
LIBXSMM_API_INTERN libxsmm_matrix_eqn* libxsmm_matrix_eqn_get_equation( libxsmm_blasint eqn_idx );
LIBXSMM_API_INTERN int libxsmm_matrix_eqn_is_ready_for_jit( libxsmm_blasint eqn_idx );
LIBXSMM_API_INTERN void libxsmm_matrix_eqn_assign_reg_scores( libxsmm_matrix_eqn_elem* cur_node );
LIBXSMM_API_INTERN void libxsmm_matrix_eqn_create_exec_plan( libxsmm_matrix_eqn_elem* cur_node, libxsmm_blasint *global_timestamp, libxsmm_blasint n_max_tmp, libxsmm_blasint *tmp_storage_pool );
LIBXSMM_API_INTERN libxsmm_blasint libxsmm_matrix_eqn_reserve_tmp_storage(libxsmm_blasint n_max_tmp, libxsmm_blasint *tmp_storage_pool);
LIBXSMM_API_INTERN void libxsmm_generator_assign_new_timestamp(libxsmm_matrix_eqn_elem* cur_node, libxsmm_blasint *current_timestamp );
LIBXSMM_API_INTERN void libxsmm_generator_matequation_assign_timestamps(libxsmm_matrix_eqn *eqn);
LIBXSMM_API_INTERN void libxsmm_generator_reoptimize_eqn(libxsmm_matrix_eqn *eqn);
LIBXSMM_API_INTERN void libxsmm_matrix_eqn_adjust_tmp_sizes( libxsmm_matrix_eqn_elem* cur_node );
LIBXSMM_API_INTERN int libxsmm_matrix_eqn_is_unary_opcode_reduce_kernel (unsigned int opcode);
LIBXSMM_API_INTERN int libxsmm_matrix_eqn_is_unary_opcode_transform_kernel (unsigned int opcode);
LIBXSMM_API_INTERN int libxsmm_matrix_eqn_is_unary_opcode_reduce_to_scalar (unsigned int opcode);
LIBXSMM_API_INTERN int libxsmm_matrix_eqn_is_binary_opcode_reduce_to_scalar (unsigned int opcode);

LIBXSMM_API_INTERN
libxsmm_matrix_eqn_bcast_type libxsmm_matrix_eqn_get_bcast_type_unary(libxsmm_bitfield flags);

LIBXSMM_API_INTERN
libxsmm_matrix_eqn_bcast_type libxsmm_matrix_eqn_get_bcast_type_binary(libxsmm_bitfield flags, unsigned int side);

LIBXSMM_API_INTERN
libxsmm_matrix_eqn_bcast_type libxsmm_matrix_eqn_get_bcast_type_ternary(libxsmm_bitfield flags, unsigned int side);

LIBXSMM_API_INTERN void libxsmm_matrix_eqn_reassign_bcast_tmp(libxsmm_matrix_eqn *eqn);
LIBXSMM_API_INTERN void libxsmm_matrix_eqn_reassign_children_bcast_tmp(libxsmm_matrix_eqn *eqn, libxsmm_matrix_eqn_elem* cur_node);

LIBXSMM_API_INTERN void libxsmm_matrix_eqn_trv_dbg_print( libxsmm_matrix_eqn_elem* cur_node, libxsmm_blasint indent );
#endif /*LIBXSMM_MATRIXEQN_H*/

