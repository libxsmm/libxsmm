/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Hans Pabst, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "libxsmm_matrixeqn.h"

/* aux struct for matrix equations */
LIBXSMM_APIVAR_DEFINE(libxsmm_matrix_eqn* libxsmm_matrix_eqns[LIBXSMM_MAX_EQN_COUNT]);
LIBXSMM_APIVAR_DEFINE(libxsmm_blasint libxsmm_matrix_eqns_init);
LIBXSMM_APIVAR_DEFINE(libxsmm_blasint libxsmm_matrix_eqns_count);

LIBXSMM_API_INTERN void libxsmm_meqn_tree_contains_opcode(libxsmm_meqn_elem *node, libxsmm_meltw_unary_type u_opcode, libxsmm_meltw_binary_type b_opcode, libxsmm_meltw_ternary_type t_opcode, unsigned int *result) {
  if ( node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    return;
  } else if ( node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    if (node->info.u_op.type == u_opcode) {
      *result = 1;
    }
    libxsmm_meqn_tree_contains_opcode(node->le, u_opcode, b_opcode, t_opcode, result);
  } else if ( node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    if (node->info.b_op.type == b_opcode) {
      *result = 1;
    }
    libxsmm_meqn_tree_contains_opcode(node->le, u_opcode, b_opcode, t_opcode, result);
    libxsmm_meqn_tree_contains_opcode(node->ri, u_opcode, b_opcode, t_opcode, result);
  } else if ( node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
    if (node->info.t_op.type == t_opcode) {
      *result = 1;
    }
    libxsmm_meqn_tree_contains_opcode(node->le, u_opcode, b_opcode, t_opcode, result);
    libxsmm_meqn_tree_contains_opcode(node->ri, u_opcode, b_opcode, t_opcode, result);
    libxsmm_meqn_tree_contains_opcode(node->r2, u_opcode, b_opcode, t_opcode, result);
  }
}

LIBXSMM_API_INTERN unsigned int libxsmm_meqn_contains_opcode(libxsmm_matrix_eqn *eqn, libxsmm_meltw_unary_type u_opcode, libxsmm_meltw_binary_type b_opcode, libxsmm_meltw_ternary_type t_opcode ) {
  unsigned int result = 0;
  libxsmm_meqn_tree_contains_opcode(eqn->eqn_root, u_opcode, b_opcode, t_opcode, &result);
  return result;
}

LIBXSMM_API_INTERN void libxsmm_meqn_tree_all_nodes_dtype(libxsmm_meqn_elem *node, libxsmm_datatype dtype, unsigned int *result) {
  if (node->tmp.dtype != dtype) {
    *result = 0;
  }
  if ( node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    return;
  } else if ( node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    libxsmm_meqn_tree_all_nodes_dtype(node->le, dtype, result);
  } else if ( node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    libxsmm_meqn_tree_all_nodes_dtype(node->le, dtype, result);
    libxsmm_meqn_tree_all_nodes_dtype(node->ri, dtype, result);
  } else if ( node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
    libxsmm_meqn_tree_all_nodes_dtype(node->le, dtype, result);
    libxsmm_meqn_tree_all_nodes_dtype(node->ri, dtype, result);
    libxsmm_meqn_tree_all_nodes_dtype(node->r2, dtype, result);
  }
}

LIBXSMM_API_INTERN unsigned int libxsmm_meqn_all_nodes_dtype(libxsmm_matrix_eqn *eqn, libxsmm_datatype dtype) {
  unsigned int result = 1;
  libxsmm_meqn_tree_all_nodes_dtype(eqn->eqn_root, dtype, &result);
  return result;
}

LIBXSMM_API_INTERN void libxsmm_meqn_tree_all_args_dtype(libxsmm_meqn_elem *node, libxsmm_datatype dtype, unsigned int *result) {
  if ( node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    if (node->info.arg.dtype != dtype) {
      *result = 0;
    }
  } else if ( node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    libxsmm_meqn_tree_all_args_dtype(node->le, dtype, result);
  } else if ( node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    libxsmm_meqn_tree_all_args_dtype(node->le, dtype, result);
    libxsmm_meqn_tree_all_args_dtype(node->ri, dtype, result);
  } else if ( node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
    libxsmm_meqn_tree_all_args_dtype(node->le, dtype, result);
    libxsmm_meqn_tree_all_args_dtype(node->ri, dtype, result);
    libxsmm_meqn_tree_all_args_dtype(node->r2, dtype, result);
  }
}

LIBXSMM_API_INTERN unsigned int libxsmm_meqn_all_args_dtype(libxsmm_matrix_eqn *eqn, libxsmm_datatype dtype) {
  unsigned int result = 1;
  libxsmm_meqn_tree_all_args_dtype(eqn->eqn_root, dtype, &result);
  return result;
}

LIBXSMM_API_INTERN void libxsmm_meqn_tree_any_args_dtype(libxsmm_meqn_elem *node, libxsmm_datatype dtype, unsigned int *result) {
  if ( node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    if (node->info.arg.dtype == dtype) {
      *result = 1;
    }
  } else if ( node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    libxsmm_meqn_tree_any_args_dtype(node->le, dtype, result);
  } else if ( node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    libxsmm_meqn_tree_any_args_dtype(node->le, dtype, result);
    libxsmm_meqn_tree_any_args_dtype(node->ri, dtype, result);
  } else if ( node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
    libxsmm_meqn_tree_any_args_dtype(node->le, dtype, result);
    libxsmm_meqn_tree_any_args_dtype(node->ri, dtype, result);
    libxsmm_meqn_tree_any_args_dtype(node->r2, dtype, result);
  }
}

LIBXSMM_API_INTERN unsigned int libxsmm_meqn_any_args_dtype(libxsmm_matrix_eqn *eqn, libxsmm_datatype dtype) {
  unsigned int result = 0;
  libxsmm_meqn_tree_any_args_dtype(eqn->eqn_root, dtype, &result);
  return result;
}

LIBXSMM_API_INTERN void libxsmm_meqn_are_nodes_pure_f32(libxsmm_meqn_elem *node, unsigned int *result) {
  libxsmm_meqn_tree_all_nodes_dtype(node, LIBXSMM_DATATYPE_F32, result);
}

LIBXSMM_API_INTERN libxsmm_matrix_eqn* libxsmm_meqn_get_equation( libxsmm_blasint eqn_idx ) {
  return libxsmm_matrix_eqns[eqn_idx];
}

LIBXSMM_API_INTERN
libxsmm_meqn_bcast_type libxsmm_meqn_get_bcast_type_unary(libxsmm_bitfield flags) {
  libxsmm_meqn_bcast_type  result = LIBXSMM_MATRIX_EQN_BCAST_TYPE_NONE;
  if ((flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW) > 0) {
    result = LIBXSMM_MATRIX_EQN_BCAST_TYPE_ROW;
  } else if ((flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL) > 0) {
    result = LIBXSMM_MATRIX_EQN_BCAST_TYPE_COL;
  } else if ((flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR) > 0) {
    result = LIBXSMM_MATRIX_EQN_BCAST_TYPE_SCALAR;
  }
  return result;
}

LIBXSMM_API_INTERN
libxsmm_meqn_bcast_type libxsmm_meqn_get_bcast_type_binary(libxsmm_bitfield flags, unsigned int side) {
  libxsmm_meqn_bcast_type  result = LIBXSMM_MATRIX_EQN_BCAST_TYPE_NONE;
  if (side == RIGHT) {
    if ((flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1) > 0) {
      result = LIBXSMM_MATRIX_EQN_BCAST_TYPE_ROW;
    } else if ((flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1) > 0) {
      result = LIBXSMM_MATRIX_EQN_BCAST_TYPE_COL;
    } else if ((flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1) > 0) {
      result = LIBXSMM_MATRIX_EQN_BCAST_TYPE_SCALAR;
    }
  }
  if (side == LEFT) {
    if ((flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0) > 0) {
      result = LIBXSMM_MATRIX_EQN_BCAST_TYPE_ROW;
    } else if ((flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0) > 0) {
      result = LIBXSMM_MATRIX_EQN_BCAST_TYPE_COL;
    } else if ((flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0) > 0) {
      result = LIBXSMM_MATRIX_EQN_BCAST_TYPE_SCALAR;
    }
  }
  return result;
}

LIBXSMM_API_INTERN
libxsmm_meqn_bcast_type libxsmm_meqn_get_bcast_type_ternary(libxsmm_bitfield flags, unsigned int side) {
  libxsmm_meqn_bcast_type  result = LIBXSMM_MATRIX_EQN_BCAST_TYPE_NONE;
  if (side == RIGHT2) {
    if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_2) > 0) {
      result = LIBXSMM_MATRIX_EQN_BCAST_TYPE_ROW;
    } else if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2) > 0) {
      result = LIBXSMM_MATRIX_EQN_BCAST_TYPE_COL;
    } else if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2) > 0) {
      result = LIBXSMM_MATRIX_EQN_BCAST_TYPE_SCALAR;
    }
  }
  if (side == RIGHT) {
    if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_1) > 0) {
      result = LIBXSMM_MATRIX_EQN_BCAST_TYPE_ROW;
    } else if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1) > 0) {
      result = LIBXSMM_MATRIX_EQN_BCAST_TYPE_COL;
    } else if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1) > 0) {
      result = LIBXSMM_MATRIX_EQN_BCAST_TYPE_SCALAR;
    }
  }
  if (side == LEFT) {
    if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_0) > 0) {
      result = LIBXSMM_MATRIX_EQN_BCAST_TYPE_ROW;
    } else if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0) > 0) {
      result = LIBXSMM_MATRIX_EQN_BCAST_TYPE_COL;
    } else if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_0) > 0) {
      result = LIBXSMM_MATRIX_EQN_BCAST_TYPE_SCALAR;
    }
  }
  return result;
}

LIBXSMM_API_INTERN libxsmm_blasint libxsmm_meqn_can_overwrite_unary_input(libxsmm_meqn_elem* cur_node);
LIBXSMM_API_INTERN libxsmm_blasint libxsmm_meqn_can_overwrite_unary_input(libxsmm_meqn_elem* cur_node) {
  libxsmm_blasint result = 1;
  if (cur_node->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) {
    result = 0;
  }
  if (((cur_node->le->tmp.dtype == LIBXSMM_DATATYPE_BF16) || (cur_node->le->tmp.dtype == LIBXSMM_DATATYPE_F16) || (cur_node->le->tmp.dtype == LIBXSMM_DATATYPE_BF8) || (cur_node->le->tmp.dtype == LIBXSMM_DATATYPE_HF8)) &&
       ((cur_node->tmp.dtype == LIBXSMM_DATATYPE_F32) || (cur_node->tmp.dtype == LIBXSMM_DATATYPE_F64))) {
    result = 0;
  }
  if (libxsmm_meqn_is_unary_opcode_transform_kernel(cur_node->info.u_op.type) > 0) {
    result = 0;
  }
  return result;
}

LIBXSMM_API_INTERN libxsmm_blasint libxsmm_meqn_can_overwrite_binary_input(libxsmm_meqn_elem* cur_node);
LIBXSMM_API_INTERN libxsmm_blasint libxsmm_meqn_can_overwrite_binary_input(libxsmm_meqn_elem* cur_node) {
  libxsmm_blasint result = 1;
  if ((cur_node->info.b_op.is_matmul == 1) || (cur_node->info.b_op.is_brgemm == 1)) {
    result = 0;
  }
  if (((cur_node->le->tmp.dtype == LIBXSMM_DATATYPE_BF16) || (cur_node->ri->tmp.dtype == LIBXSMM_DATATYPE_BF16) || (cur_node->le->tmp.dtype == LIBXSMM_DATATYPE_F16) || (cur_node->ri->tmp.dtype == LIBXSMM_DATATYPE_F16) || (cur_node->le->tmp.dtype == LIBXSMM_DATATYPE_HF8) || (cur_node->ri->tmp.dtype == LIBXSMM_DATATYPE_HF8) || (cur_node->le->tmp.dtype == LIBXSMM_DATATYPE_BF8) || (cur_node->ri->tmp.dtype == LIBXSMM_DATATYPE_BF8)) &&
      ((cur_node->tmp.dtype == LIBXSMM_DATATYPE_F32) || (cur_node->tmp.dtype == LIBXSMM_DATATYPE_F64))) {
    result = 0;
  }
  return result;
}

LIBXSMM_API_INTERN void libxsmm_meqn_trv_dbg_print( libxsmm_meqn_elem* cur_node, libxsmm_blasint indent ) {
  libxsmm_blasint i;
  libxsmm_blasint tree_print_indent = 4;

  for ( i = 0; i < indent; ++i ) {
    if ( i < indent - tree_print_indent ) {
      printf(" ");
    } else {
      if ( i % tree_print_indent == 0 ) {
        printf("|");
      } else {
        printf("-");
      }
    }
  }

  /* check if we are at an argument leaf, then we move up */
  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    libxsmm_blasint argid = cur_node->info.arg.in_pos;
    if ( (cur_node->le == NULL) && (cur_node->ri == NULL) ) {
      if (argid >= 0) {
        printf("ARG: M=%i, N=%i, LD=%i, arg_id=%i, dtype=%i\n", cur_node->info.arg.m, cur_node->info.arg.n, cur_node->info.arg.ld, cur_node->info.arg.in_pos, LIBXSMM_TYPESIZE(cur_node->info.arg.dtype) );
      } else {
        printf("ARG: M=%i, N=%i, LD=%i, arg_id is scratch=%i, dtype=%i\n", cur_node->info.arg.m, cur_node->info.arg.n, cur_node->info.arg.ld, -1-argid, LIBXSMM_TYPESIZE(cur_node->info.arg.dtype) );
      }
    } else {
      printf("ERROR: Arg cannot have left or right child!\n");
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    /* we have to push more in this branch */
    if ( cur_node->le != NULL ) {
      printf("UNARY: type=%i, flags=%i, timestamp=%i, out_tmp_id=%i, out_dtype=%i\n", (int)cur_node->info.u_op.type, (int)cur_node->info.u_op.flags, cur_node->visit_timestamp, cur_node->tmp.id, LIBXSMM_TYPESIZE(cur_node->tmp.dtype));
      libxsmm_meqn_trv_dbg_print( cur_node->le, indent+tree_print_indent );
    /* we have reached the root, as we are unary, there is no right branch */
    } else if ( (cur_node->ri != NULL) ) {
      printf("ERROR: Unary cannot have right childs!\n");
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    /* we have to push more in this branch */
    if ( (cur_node->le != NULL) && (cur_node->ri != NULL) ) {
      printf("BINARY: type=%i, flags=%i, timestamp=%i, out_tmp_id=%i, out_dtype=%i\n", (int)cur_node->info.b_op.type, (int)cur_node->info.b_op.flags, cur_node->visit_timestamp, cur_node->tmp.id, LIBXSMM_TYPESIZE(cur_node->tmp.dtype));
      libxsmm_meqn_trv_dbg_print( cur_node->le, indent+tree_print_indent );
      libxsmm_meqn_trv_dbg_print( cur_node->ri, indent+tree_print_indent );
    } else {
      printf("ERROR: Binary needs left and right child!\n");
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
    /* we have to push more in this branch */
    if ( (cur_node->le != NULL) && (cur_node->ri != NULL) && (cur_node->r2 != NULL)) {
      printf("TERNARY: type=%i, flags=%i, timestamp=%i, out_tmp_id=%i, out_dtype=%i\n", (int)cur_node->info.t_op.type, (int)cur_node->info.t_op.flags, cur_node->visit_timestamp, cur_node->tmp.id, LIBXSMM_TYPESIZE(cur_node->tmp.dtype));
      libxsmm_meqn_trv_dbg_print( cur_node->le, indent+tree_print_indent );
      libxsmm_meqn_trv_dbg_print( cur_node->ri, indent+tree_print_indent );
      libxsmm_meqn_trv_dbg_print( cur_node->r2, indent+tree_print_indent );
    } else {
      printf("ERROR: Ternary needs three children!\n");
    }
  } else {
    /* should not happen */
  }
}

LIBXSMM_API_INTERN void libxsmm_meqn_propagate_tmp_info( libxsmm_meqn_elem* cur_node ) {
  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    if ( (cur_node->le == NULL) && (cur_node->ri == NULL) ) {
      cur_node->tmp.dtype = cur_node->info.arg.dtype;
    }
    else {
      printf("ERROR: Arg cannot have left or right child!\n");
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    if ( cur_node->le != NULL ) {
      cur_node->tmp.dtype = cur_node->info.u_op.dtype;
      libxsmm_meqn_propagate_tmp_info( cur_node->le );
    } else if ( (cur_node->ri != NULL) ) {
      printf("ERROR: Unary cannot have right childs!\n");
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    if ( (cur_node->le != NULL) && (cur_node->ri != NULL) ) {
      cur_node->tmp.dtype = cur_node->info.b_op.dtype;
      libxsmm_meqn_propagate_tmp_info( cur_node->le );
      libxsmm_meqn_propagate_tmp_info( cur_node->ri );
    } else {
      printf("ERROR: Binary needs left and right child!\n");
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
    if ( (cur_node->le != NULL) && (cur_node->ri != NULL) && (cur_node->r2 != NULL) ) {
      cur_node->tmp.dtype = cur_node->info.t_op.dtype;
      libxsmm_meqn_propagate_tmp_info( cur_node->le );
      libxsmm_meqn_propagate_tmp_info( cur_node->ri );
      libxsmm_meqn_propagate_tmp_info( cur_node->r2 );
    } else {
      printf("ERROR: Ternary needs all three children!\n");
    }
  } else {
    /* should not happen */
  }
}

LIBXSMM_API_INTERN void libxsmm_meqn_assign_reg_scores( libxsmm_meqn_elem* cur_node ) {
  /* check if we are at an argument leaf, then we assign register score 0 */
  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    if ( (cur_node->le == NULL) && (cur_node->ri == NULL) ) {
      cur_node->reg_score = 0;
    }
    else {
      printf("ERROR: Arg cannot have left or right child!\n");
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    /* If the node is unary type we have the following cases:
     * 1) If the left child is an arg, we just set the score to 1 (we do not overwrite the input)
     * 2) if the left child is NOT an arg AND we can overwrite the tmp, we just propagate the register score from it (no additional tmp storage is needed)
     * 3) if the left child is NOT an arg AND we CAN NOT overwrite the tmp, we should make the register score at least 2
     * */
    if ( cur_node->le != NULL ) {
      libxsmm_meqn_assign_reg_scores( cur_node->le );
      if ( cur_node->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
        cur_node->reg_score = 1;
      } else {
        if (libxsmm_meqn_can_overwrite_unary_input(cur_node) > 0) {
          cur_node->reg_score = cur_node->le->reg_score;
        } else {
          cur_node->reg_score = LIBXSMM_MAX(2, cur_node->le->reg_score);
        }
      }
    /* we have reached the root, as we are unary, there is no right branch */
    } else if ( (cur_node->ri != NULL) ) {
      printf("ERROR: Unary cannot have right childs!\n");
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    if ( (cur_node->le != NULL) && (cur_node->ri != NULL) ) {
      libxsmm_meqn_assign_reg_scores( cur_node->le );
      libxsmm_meqn_assign_reg_scores( cur_node->ri );

      /* If left and right are args, we just need 1 tmp */
      if ( (cur_node->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->ri->type == LIBXSMM_MATRIX_EQN_NODE_ARG) ) {
        cur_node->reg_score = 1;
      } else {
        if (libxsmm_meqn_can_overwrite_binary_input(cur_node) > 0) {
        /* If the node is binary type we have two cases:
         * 1) If the left/right subtrees have the same register score, we have to increase it by one (i.e. we have to first compute one of the subtrees and keep the result in a tmp storage and then compute the other subtree, so we would need an extra tmp storage)
         * 2) If the left/right subtrees DO NOT have the same register score, then we assign  the maximum of the register scores (i.e. we would compute first the subtree with the maximum score and then the tree with the smallest score, thus no extra tmp storage is required) */
          if (cur_node->le->reg_score == cur_node->ri->reg_score) {
            cur_node->reg_score = cur_node->le->reg_score + 1;
          } else {
            cur_node->reg_score = LIBXSMM_MAX(cur_node->le->reg_score, cur_node->ri->reg_score);
          }
        } else {
          if (cur_node->le->reg_score == cur_node->ri->reg_score) {
            cur_node->reg_score = LIBXSMM_MAX(3, cur_node->le->reg_score + 1);
          } else {
            cur_node->reg_score = LIBXSMM_MAX(3, LIBXSMM_MAX(cur_node->le->reg_score, cur_node->ri->reg_score));
          }
        }
      }
    } else {
      printf("ERROR: Binary needs left and right child!\n");
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
    if ( (cur_node->le != NULL) && (cur_node->ri != NULL) && (cur_node->r2 != NULL) ) {
      int use_r2_as_output = ((cur_node->info.t_op.flags & LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT) > 0) ? 1 : 0;
      libxsmm_meqn_assign_reg_scores( cur_node->le );
      libxsmm_meqn_assign_reg_scores( cur_node->ri );
      libxsmm_meqn_assign_reg_scores( cur_node->r2 );
      /* If all children re args, we just need 1 tmp */
      if ( (cur_node->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->ri->type == LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->r2->type == LIBXSMM_MATRIX_EQN_NODE_ARG) ) {
        cur_node->reg_score = 1;
      } else {
        if (use_r2_as_output > 0) {
          cur_node->reg_score = LIBXSMM_MAX(3, LIBXSMM_MAX(LIBXSMM_MAX(cur_node->le->reg_score, cur_node->ri->reg_score), cur_node->r2->reg_score));
        } else {
          cur_node->reg_score = LIBXSMM_MAX(4, LIBXSMM_MAX(LIBXSMM_MAX(cur_node->le->reg_score, cur_node->ri->reg_score), cur_node->r2->reg_score));
        }
      }
    } else {
      printf("ERROR: Ternary needs all three children!\n");
    }
  } else {
    /* should not happen */
  }
}

LIBXSMM_API_INTERN
void libxsmm_meqn_assign_new_timestamp(libxsmm_meqn_elem* cur_node, libxsmm_blasint *current_timestamp ) {
  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    /* Do not increase the timestamp, this node is just an arg so it's not part of the execution */
    cur_node->visit_timestamp = -1;
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    libxsmm_meqn_assign_new_timestamp( cur_node->le, current_timestamp );
    cur_node->visit_timestamp = *current_timestamp;
    *current_timestamp = *current_timestamp + 1;
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    if (cur_node->le->reg_score >= cur_node->ri->reg_score) {
      libxsmm_meqn_assign_new_timestamp( cur_node->le, current_timestamp );
      libxsmm_meqn_assign_new_timestamp( cur_node->ri, current_timestamp );
    } else {
      libxsmm_meqn_assign_new_timestamp( cur_node->ri, current_timestamp );
      libxsmm_meqn_assign_new_timestamp( cur_node->le, current_timestamp );
    }
    cur_node->visit_timestamp = *current_timestamp;
    *current_timestamp = *current_timestamp + 1;
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
    if ((cur_node->le->reg_score >= cur_node->ri->reg_score) && (cur_node->le->reg_score >= cur_node->r2->reg_score) ) {
      libxsmm_meqn_assign_new_timestamp( cur_node->le, current_timestamp );
      if ( cur_node->ri->reg_score >= cur_node->r2->reg_score ) {
        libxsmm_meqn_assign_new_timestamp( cur_node->ri, current_timestamp );
        libxsmm_meqn_assign_new_timestamp( cur_node->r2, current_timestamp );
      } else {
        libxsmm_meqn_assign_new_timestamp( cur_node->r2, current_timestamp );
        libxsmm_meqn_assign_new_timestamp( cur_node->ri, current_timestamp );
      }
    } else if ((cur_node->ri->reg_score >= cur_node->le->reg_score) && (cur_node->ri->reg_score >= cur_node->r2->reg_score) ) {
      libxsmm_meqn_assign_new_timestamp( cur_node->ri, current_timestamp );
      if ( cur_node->le->reg_score >= cur_node->r2->reg_score ) {
        libxsmm_meqn_assign_new_timestamp( cur_node->le, current_timestamp );
        libxsmm_meqn_assign_new_timestamp( cur_node->r2, current_timestamp );
      } else {
        libxsmm_meqn_assign_new_timestamp( cur_node->r2, current_timestamp );
        libxsmm_meqn_assign_new_timestamp( cur_node->le, current_timestamp );
      }
    } else {
      libxsmm_meqn_assign_new_timestamp( cur_node->r2, current_timestamp );
      if ( cur_node->le->reg_score >= cur_node->ri->reg_score ) {
        libxsmm_meqn_assign_new_timestamp( cur_node->le, current_timestamp );
        libxsmm_meqn_assign_new_timestamp( cur_node->ri, current_timestamp );
      } else {
        libxsmm_meqn_assign_new_timestamp( cur_node->ri, current_timestamp );
        libxsmm_meqn_assign_new_timestamp( cur_node->le, current_timestamp );
      }
    }
    cur_node->visit_timestamp = *current_timestamp;
    *current_timestamp = *current_timestamp + 1;
  } else {
    /* should not happen */
  }
}

LIBXSMM_API_INTERN
void libxsmm_meqn_assign_timestamps(libxsmm_matrix_eqn *eqn) {
  libxsmm_blasint timestamp = 0;
  libxsmm_meqn_assign_new_timestamp(eqn->eqn_root, &timestamp );
}

LIBXSMM_API_INTERN libxsmm_blasint libxsmm_meqn_reserve_tmp_storage(libxsmm_blasint n_max_tmp, libxsmm_blasint *tmp_storage_pool) {
  libxsmm_blasint i;
  if ( tmp_storage_pool != NULL ) {
    for (i = 0; i < n_max_tmp; i++) {
      if (tmp_storage_pool[i] == 0) {
        tmp_storage_pool[i] = 1;
        return i;
      }
    }
  }
  return -1;
}

LIBXSMM_API_INTERN void libxsmm_meqn_exec_plan_configure_unary_tmp(libxsmm_meqn_elem* cur_node);
LIBXSMM_API_INTERN void libxsmm_meqn_exec_plan_configure_unary_tmp(libxsmm_meqn_elem* cur_node) {
  cur_node->tmp.m  = cur_node->le->tmp.m;
  cur_node->tmp.n  = cur_node->le->tmp.n;
  cur_node->tmp.ld  = cur_node->le->tmp.m;
  cur_node->tmp.dtype  = cur_node->info.u_op.dtype;
}

LIBXSMM_API_INTERN void libxsmm_meqn_exec_plan_configure_binary_tmp(libxsmm_meqn_elem* cur_node);
LIBXSMM_API_INTERN void libxsmm_meqn_exec_plan_configure_binary_tmp(libxsmm_meqn_elem* cur_node) {
  cur_node->tmp.m  = cur_node->le->tmp.m;
  cur_node->tmp.ld  = cur_node->le->tmp.m;
  if ((cur_node->info.b_op.is_matmul == 1) ||
      (cur_node->info.b_op.is_brgemm == 1)) {
    cur_node->tmp.n  = cur_node->ri->tmp.n;
  } else {
    cur_node->tmp.n  = cur_node->le->tmp.n;
  }
  cur_node->tmp.dtype  = cur_node->info.b_op.dtype;
}

LIBXSMM_API_INTERN void libxsmm_meqn_exec_plan_configure_ternary_tmp(libxsmm_meqn_elem* cur_node);
LIBXSMM_API_INTERN void libxsmm_meqn_exec_plan_configure_ternary_tmp(libxsmm_meqn_elem* cur_node) {
  cur_node->tmp.m  = cur_node->r2->tmp.m;
  cur_node->tmp.n  = cur_node->r2->tmp.n;
  cur_node->tmp.ld  = cur_node->r2->tmp.m;
  if ((cur_node->info.t_op.is_matmul == 1) ||
      (cur_node->info.t_op.is_brgemm == 1)) {
    cur_node->tmp.m  = cur_node->r2->tmp.m;
    cur_node->tmp.n  = cur_node->r2->tmp.n;
    cur_node->tmp.ld  = cur_node->r2->tmp.ld;
  }
  cur_node->tmp.dtype  = cur_node->info.t_op.dtype;
}

LIBXSMM_API_INTERN void libxsmm_meqn_exec_plan_visit_arg_node(libxsmm_meqn_elem* cur_node);
LIBXSMM_API_INTERN void libxsmm_meqn_exec_plan_visit_arg_node(libxsmm_meqn_elem* cur_node) {
  /* Do not increase the timestamp, this node is just an arg so it's not part of the execution */
  cur_node->visit_timestamp = -1;
  cur_node->n_args = 1;
  cur_node->max_tmp_size = cur_node->info.arg.m * cur_node->info.arg.n;
  cur_node->tmp.m  = cur_node->info.arg.m;
  cur_node->tmp.n  = cur_node->info.arg.n;
  cur_node->tmp.ld  = cur_node->info.arg.ld;
  cur_node->tmp.dtype  = cur_node->info.arg.dtype;
}

LIBXSMM_API_INTERN void libxsmm_meqn_exec_plan_visit_unary_node(libxsmm_meqn_elem* cur_node, libxsmm_blasint *global_timestamp, libxsmm_blasint n_max_tmp, libxsmm_blasint *tmp_storage_pool);
LIBXSMM_API_INTERN void libxsmm_meqn_exec_plan_visit_unary_node(libxsmm_meqn_elem* cur_node, libxsmm_blasint *global_timestamp, libxsmm_blasint n_max_tmp, libxsmm_blasint *tmp_storage_pool) {
  /* Assign timestamp and propagate info for n_args/max_tmp_size */
  cur_node->visit_timestamp = *global_timestamp;
  *global_timestamp = *global_timestamp + 1;
  cur_node->n_args = cur_node->le->n_args;
  cur_node->max_tmp_size = cur_node->le->max_tmp_size;
  /* When assigning the tmp output storage, we have two cases in the unary:
   * 1) The child is an arg, so we have to reserve a tmp storage
   * 2) The child is NOT an arg, so we just reuse the tmp storage of the child IF we are allowed to overwrite */
  if ( cur_node->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    cur_node->tmp.id = libxsmm_meqn_reserve_tmp_storage( n_max_tmp, tmp_storage_pool );
    cur_node->tree_max_comp_tsize = LIBXSMM_TYPESIZE( cur_node->info.u_op.dtype );
  } else {
    if (libxsmm_meqn_can_overwrite_unary_input(cur_node) > 0) {
      cur_node->tmp.id = cur_node->le->tmp.id;
    } else {
      cur_node->tmp.id = libxsmm_meqn_reserve_tmp_storage( n_max_tmp, tmp_storage_pool );
      if (cur_node->le->tmp.id >= 0) tmp_storage_pool[cur_node->le->tmp.id] = 0;
    }
    cur_node->tree_max_comp_tsize = LIBXSMM_MAX( LIBXSMM_TYPESIZE(cur_node->info.u_op.dtype), cur_node->le->tree_max_comp_tsize );
  }
  libxsmm_meqn_exec_plan_configure_unary_tmp( cur_node );
}

LIBXSMM_API_INTERN void libxsmm_meqn_exec_plan_visit_binary_node(libxsmm_meqn_elem* cur_node, libxsmm_blasint *global_timestamp, libxsmm_blasint n_max_tmp, libxsmm_blasint *tmp_storage_pool);
LIBXSMM_API_INTERN void libxsmm_meqn_exec_plan_visit_binary_node(libxsmm_meqn_elem* cur_node, libxsmm_blasint *global_timestamp, libxsmm_blasint n_max_tmp, libxsmm_blasint *tmp_storage_pool) {
  /* Assign timestamp and propagate info for n_args/max_tmp_size */
  cur_node->visit_timestamp = *global_timestamp;
  *global_timestamp = *global_timestamp + 1;
  cur_node->n_args = cur_node->le->n_args + cur_node->ri->n_args;
  cur_node->max_tmp_size = LIBXSMM_MAX(cur_node->le->max_tmp_size, cur_node->ri->max_tmp_size);
  /* Max tmp size has to be adjusted if it is a MATMUL op */
  if ((cur_node->info.b_op.is_matmul == 1) || (cur_node->info.b_op.is_brgemm == 1)) {
    libxsmm_blasint matmul_out_size = cur_node->le->tmp.m * cur_node->ri->tmp.n;
    cur_node->max_tmp_size = LIBXSMM_MAX(matmul_out_size, cur_node->max_tmp_size);
  }
  /* When assigning the tmp output storage, we have three cases in the binary:
   * 1) Both children are arg, so we have to reserve a tmp storage
   * 2) Both child are NOT arg, so we reuse the tmp storage of either one for our output and we make the other tmp storage available IF we are allowed to overwrite
   * 3) One child IS arg and the other child is NOT an arg, so we just reuse the tmp storage of the non-arg child IF we are allowed to overwrite */
  if ( (cur_node->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->ri->type == LIBXSMM_MATRIX_EQN_NODE_ARG) ) {
    cur_node->tmp.id = libxsmm_meqn_reserve_tmp_storage( n_max_tmp, tmp_storage_pool );
    cur_node->tree_max_comp_tsize = LIBXSMM_TYPESIZE( cur_node->info.b_op.dtype );
  } else if ( (cur_node->le->type != LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->ri->type != LIBXSMM_MATRIX_EQN_NODE_ARG) ) {
    if (libxsmm_meqn_can_overwrite_binary_input(cur_node) > 0) {
      cur_node->tmp.id = cur_node->le->tmp.id;
      if (cur_node->ri->tmp.id >= 0) tmp_storage_pool[cur_node->ri->tmp.id] = 0;
    } else {
      cur_node->tmp.id = libxsmm_meqn_reserve_tmp_storage( n_max_tmp, tmp_storage_pool );
      if (cur_node->le->tmp.id >= 0) tmp_storage_pool[cur_node->le->tmp.id] = 0;
      if (cur_node->ri->tmp.id >= 0) tmp_storage_pool[cur_node->ri->tmp.id] = 0;
    }
    cur_node->tree_max_comp_tsize = LIBXSMM_MAX( LIBXSMM_TYPESIZE( cur_node->info.b_op.dtype ), LIBXSMM_MAX( cur_node->ri->tree_max_comp_tsize, cur_node->le->tree_max_comp_tsize ));
  } else {
    if (cur_node->le->type != LIBXSMM_MATRIX_EQN_NODE_ARG) {
      if (libxsmm_meqn_can_overwrite_binary_input(cur_node) > 0) {
        cur_node->tmp.id = cur_node->le->tmp.id;
      } else {
        cur_node->tmp.id = libxsmm_meqn_reserve_tmp_storage( n_max_tmp, tmp_storage_pool );
        if (cur_node->le->tmp.id >= 0) tmp_storage_pool[cur_node->le->tmp.id] = 0;
      }
      cur_node->tree_max_comp_tsize = LIBXSMM_MAX( LIBXSMM_TYPESIZE(cur_node->info.b_op.dtype), cur_node->le->tree_max_comp_tsize );
    } else {
      if (libxsmm_meqn_can_overwrite_binary_input(cur_node) > 0) {
        cur_node->tmp.id = cur_node->ri->tmp.id;
      } else {
        cur_node->tmp.id = libxsmm_meqn_reserve_tmp_storage( n_max_tmp, tmp_storage_pool );
        if (cur_node->ri->tmp.id >= 0) tmp_storage_pool[cur_node->ri->tmp.id] = 0;
      }
      cur_node->tree_max_comp_tsize = LIBXSMM_MAX( LIBXSMM_TYPESIZE(cur_node->info.b_op.dtype), cur_node->ri->tree_max_comp_tsize );
    }
  }
  libxsmm_meqn_exec_plan_configure_binary_tmp( cur_node );
}

LIBXSMM_API_INTERN void libxsmm_meqn_exec_plan_visit_ternary_node(libxsmm_meqn_elem* cur_node, libxsmm_blasint *global_timestamp, libxsmm_blasint n_max_tmp, libxsmm_blasint *tmp_storage_pool);
LIBXSMM_API_INTERN void libxsmm_meqn_exec_plan_visit_ternary_node(libxsmm_meqn_elem* cur_node, libxsmm_blasint *global_timestamp, libxsmm_blasint n_max_tmp, libxsmm_blasint *tmp_storage_pool) {
  /* Assign timestamp and propagate info for n_args/max_tmp_size */
  int use_r2_as_output = ((cur_node->info.t_op.flags & LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT) > 0) ? 1 : 0;
  cur_node->visit_timestamp = *global_timestamp;
  *global_timestamp = *global_timestamp + 1;
  cur_node->n_args = cur_node->le->n_args + cur_node->ri->n_args + cur_node->r2->n_args;
  cur_node->max_tmp_size = LIBXSMM_MAX( LIBXSMM_MAX(cur_node->le->max_tmp_size, cur_node->ri->max_tmp_size), cur_node->r2->max_tmp_size);
  if ( (cur_node->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->ri->type == LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->r2->type == LIBXSMM_MATRIX_EQN_NODE_ARG) ) {
    if ((use_r2_as_output > 0) && ((cur_node->info.t_op.is_brgemm == 1) || (cur_node->info.t_op.is_matmul == 1))) {
      cur_node->tmp.id = -(cur_node->r2->info.arg.in_pos + 1);
    } else {
      cur_node->tmp.id = libxsmm_meqn_reserve_tmp_storage( n_max_tmp, tmp_storage_pool );
    }
    cur_node->tree_max_comp_tsize = LIBXSMM_TYPESIZE( cur_node->info.t_op.dtype );
  } else if ( (cur_node->le->type != LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->ri->type != LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->r2->type != LIBXSMM_MATRIX_EQN_NODE_ARG) ) {
    if (use_r2_as_output > 0 ) {
      cur_node->tmp.id = cur_node->r2->tmp.id;
      if (cur_node->le->tmp.id >= 0) tmp_storage_pool[cur_node->le->tmp.id] = 0;
      if (cur_node->ri->tmp.id >= 0) tmp_storage_pool[cur_node->ri->tmp.id] = 0;
    } else {
      cur_node->tmp.id = libxsmm_meqn_reserve_tmp_storage( n_max_tmp, tmp_storage_pool );
      if (cur_node->le->tmp.id >= 0) tmp_storage_pool[cur_node->le->tmp.id] = 0;
      if (cur_node->ri->tmp.id >= 0) tmp_storage_pool[cur_node->ri->tmp.id] = 0;
      if (cur_node->r2->tmp.id >= 0) tmp_storage_pool[cur_node->r2->tmp.id] = 0;
    }
    cur_node->tree_max_comp_tsize = LIBXSMM_MAX( LIBXSMM_TYPESIZE( cur_node->info.t_op.dtype ), LIBXSMM_MAX( cur_node->r2->tree_max_comp_tsize, LIBXSMM_MAX( cur_node->ri->tree_max_comp_tsize, cur_node->le->tree_max_comp_tsize )));
  } else if ( (cur_node->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->ri->type != LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->r2->type != LIBXSMM_MATRIX_EQN_NODE_ARG) ) {
    if (use_r2_as_output > 0 ) {
      cur_node->tmp.id = cur_node->r2->tmp.id;
      if (cur_node->ri->tmp.id >= 0) tmp_storage_pool[cur_node->ri->tmp.id] = 0;
    } else {
      cur_node->tmp.id = libxsmm_meqn_reserve_tmp_storage( n_max_tmp, tmp_storage_pool );
      if (cur_node->ri->tmp.id >= 0) tmp_storage_pool[cur_node->ri->tmp.id] = 0;
      if (cur_node->r2->tmp.id >= 0) tmp_storage_pool[cur_node->r2->tmp.id] = 0;
    }
    cur_node->tree_max_comp_tsize = LIBXSMM_MAX( LIBXSMM_TYPESIZE( cur_node->info.t_op.dtype ), LIBXSMM_MAX( cur_node->r2->tree_max_comp_tsize, LIBXSMM_MAX( cur_node->ri->tree_max_comp_tsize, 1 )));
  } else if ( (cur_node->le->type != LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->ri->type == LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->r2->type != LIBXSMM_MATRIX_EQN_NODE_ARG) ) {
    if (use_r2_as_output > 0 ) {
      cur_node->tmp.id = cur_node->r2->tmp.id;
      if (cur_node->le->tmp.id >= 0) tmp_storage_pool[cur_node->le->tmp.id] = 0;
    } else {
      cur_node->tmp.id = libxsmm_meqn_reserve_tmp_storage( n_max_tmp, tmp_storage_pool );
      if (cur_node->le->tmp.id >= 0) tmp_storage_pool[cur_node->le->tmp.id] = 0;
      if (cur_node->r2->tmp.id >= 0) tmp_storage_pool[cur_node->r2->tmp.id] = 0;
    }
    cur_node->tree_max_comp_tsize = LIBXSMM_MAX( LIBXSMM_TYPESIZE( cur_node->info.t_op.dtype ), LIBXSMM_MAX( cur_node->r2->tree_max_comp_tsize, LIBXSMM_MAX( 1, cur_node->le->tree_max_comp_tsize )));
  } else if ( (cur_node->le->type != LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->ri->type != LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->r2->type == LIBXSMM_MATRIX_EQN_NODE_ARG) ) {
    if ((use_r2_as_output > 0) && ((cur_node->info.t_op.is_brgemm == 1) || (cur_node->info.t_op.is_matmul == 1))) {
      cur_node->tmp.id = -(cur_node->r2->info.arg.in_pos + 1);
    } else {
      cur_node->tmp.id = libxsmm_meqn_reserve_tmp_storage( n_max_tmp, tmp_storage_pool );
    }
    if (cur_node->le->tmp.id >= 0) tmp_storage_pool[cur_node->le->tmp.id] = 0;
    if (cur_node->ri->tmp.id >= 0) tmp_storage_pool[cur_node->ri->tmp.id] = 0;
    cur_node->tree_max_comp_tsize = LIBXSMM_MAX( LIBXSMM_TYPESIZE( cur_node->info.t_op.dtype ), LIBXSMM_MAX( 1, LIBXSMM_MAX( cur_node->ri->tree_max_comp_tsize, cur_node->le->tree_max_comp_tsize )));
  } else if ( (cur_node->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->ri->type == LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->r2->type != LIBXSMM_MATRIX_EQN_NODE_ARG) ) {
    if (use_r2_as_output > 0 ) {
      cur_node->tmp.id = cur_node->r2->tmp.id;
    } else {
      cur_node->tmp.id = libxsmm_meqn_reserve_tmp_storage( n_max_tmp, tmp_storage_pool );
      if (cur_node->r2->tmp.id >= 0) tmp_storage_pool[cur_node->r2->tmp.id] = 0;
    }
    cur_node->tree_max_comp_tsize = LIBXSMM_MAX( LIBXSMM_TYPESIZE( cur_node->info.t_op.dtype ), LIBXSMM_MAX( cur_node->r2->tree_max_comp_tsize, LIBXSMM_MAX( 1, 1 )));
  } else if ( (cur_node->le->type != LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->ri->type == LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->r2->type == LIBXSMM_MATRIX_EQN_NODE_ARG) ) {
    if ((use_r2_as_output > 0) && ((cur_node->info.t_op.is_brgemm == 1) || (cur_node->info.t_op.is_matmul == 1))) {
      cur_node->tmp.id = -(cur_node->r2->info.arg.in_pos + 1);
    } else {
      cur_node->tmp.id = libxsmm_meqn_reserve_tmp_storage( n_max_tmp, tmp_storage_pool );
    }
    if (cur_node->le->tmp.id >= 0) tmp_storage_pool[cur_node->le->tmp.id] = 0;
    cur_node->tree_max_comp_tsize = LIBXSMM_MAX( LIBXSMM_TYPESIZE( cur_node->info.t_op.dtype ), LIBXSMM_MAX( 1, LIBXSMM_MAX( 1, cur_node->le->tree_max_comp_tsize )));
  } else if ( (cur_node->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->ri->type != LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->r2->type == LIBXSMM_MATRIX_EQN_NODE_ARG) ) {
    if ((use_r2_as_output > 0) && ((cur_node->info.t_op.is_brgemm == 1) || (cur_node->info.t_op.is_matmul == 1))) {
      cur_node->tmp.id = -(cur_node->r2->info.arg.in_pos + 1);
    } else {
      cur_node->tmp.id = libxsmm_meqn_reserve_tmp_storage( n_max_tmp, tmp_storage_pool );
    }
    if (cur_node->ri->tmp.id >= 0) tmp_storage_pool[cur_node->ri->tmp.id] = 0;
    cur_node->tree_max_comp_tsize = LIBXSMM_MAX( LIBXSMM_TYPESIZE( cur_node->info.t_op.dtype ), LIBXSMM_MAX( 1, LIBXSMM_MAX( cur_node->ri->tree_max_comp_tsize, 1)));
  }
  libxsmm_meqn_exec_plan_configure_ternary_tmp( cur_node );
}

LIBXSMM_API_INTERN void libxsmm_meqn_reassign_children_bcast_tmp(libxsmm_matrix_eqn *eqn, libxsmm_meqn_elem* cur_node) {
  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    /* Do nothing */
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    if ((cur_node->le->type != LIBXSMM_MATRIX_EQN_NODE_ARG) && (libxsmm_meqn_get_bcast_type_unary(cur_node->info.u_op.flags) != LIBXSMM_MATRIX_EQN_BCAST_TYPE_NONE)) {
      cur_node->le->tmp.id = eqn->eqn_root->reg_score;
      eqn->eqn_root->reg_score = eqn->eqn_root->reg_score + 1;
    }
    libxsmm_meqn_reassign_children_bcast_tmp(eqn, cur_node->le);
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    if ((cur_node->le->type != LIBXSMM_MATRIX_EQN_NODE_ARG) && (libxsmm_meqn_get_bcast_type_binary(cur_node->info.b_op.flags, LEFT) != LIBXSMM_MATRIX_EQN_BCAST_TYPE_NONE)) {
      cur_node->le->tmp.id = eqn->eqn_root->reg_score;
      eqn->eqn_root->reg_score = eqn->eqn_root->reg_score + 1;
    }
    if ((cur_node->ri->type != LIBXSMM_MATRIX_EQN_NODE_ARG) && (libxsmm_meqn_get_bcast_type_binary(cur_node->info.b_op.flags, RIGHT) != LIBXSMM_MATRIX_EQN_BCAST_TYPE_NONE)) {
      cur_node->ri->tmp.id = eqn->eqn_root->reg_score;
      eqn->eqn_root->reg_score = eqn->eqn_root->reg_score + 1;
    }
    libxsmm_meqn_reassign_children_bcast_tmp(eqn, cur_node->le);
    libxsmm_meqn_reassign_children_bcast_tmp(eqn, cur_node->ri);
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
    if ((cur_node->le->type != LIBXSMM_MATRIX_EQN_NODE_ARG) && (libxsmm_meqn_get_bcast_type_ternary(cur_node->info.t_op.flags, LEFT) != LIBXSMM_MATRIX_EQN_BCAST_TYPE_NONE)) {
      cur_node->le->tmp.id = eqn->eqn_root->reg_score;
      eqn->eqn_root->reg_score = eqn->eqn_root->reg_score + 1;
    }
    if ((cur_node->ri->type != LIBXSMM_MATRIX_EQN_NODE_ARG) && (libxsmm_meqn_get_bcast_type_ternary(cur_node->info.t_op.flags, RIGHT) != LIBXSMM_MATRIX_EQN_BCAST_TYPE_NONE)) {
      cur_node->ri->tmp.id = eqn->eqn_root->reg_score;
      eqn->eqn_root->reg_score = eqn->eqn_root->reg_score + 1;
    }
    if ((cur_node->r2->type != LIBXSMM_MATRIX_EQN_NODE_ARG) && (libxsmm_meqn_get_bcast_type_ternary(cur_node->info.t_op.flags, RIGHT2) != LIBXSMM_MATRIX_EQN_BCAST_TYPE_NONE)) {
      cur_node->r2->tmp.id = eqn->eqn_root->reg_score;
      eqn->eqn_root->reg_score = eqn->eqn_root->reg_score + 1;
    }
    libxsmm_meqn_reassign_children_bcast_tmp(eqn, cur_node->le);
    libxsmm_meqn_reassign_children_bcast_tmp(eqn, cur_node->ri);
    libxsmm_meqn_reassign_children_bcast_tmp(eqn, cur_node->r2);
  } else {
    /* This should not happen */
  }
}

LIBXSMM_API_INTERN void libxsmm_meqn_reassign_bcast_tmp(libxsmm_matrix_eqn *eqn) {
  libxsmm_meqn_elem* root = eqn->eqn_root;
  if ( root->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    libxsmm_meqn_reassign_children_bcast_tmp(eqn, root->le);
  }
  if ( root->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    libxsmm_meqn_reassign_children_bcast_tmp(eqn, root->le);
    libxsmm_meqn_reassign_children_bcast_tmp(eqn, root->ri);
  }
  if ( root->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
    libxsmm_meqn_reassign_children_bcast_tmp(eqn, root->le);
    libxsmm_meqn_reassign_children_bcast_tmp(eqn, root->ri);
    libxsmm_meqn_reassign_children_bcast_tmp(eqn, root->r2);
  }
}

LIBXSMM_API_INTERN void libxsmm_meqn_create_exec_plan( libxsmm_meqn_elem* cur_node, libxsmm_blasint *global_timestamp, libxsmm_blasint n_max_tmp, libxsmm_blasint *tmp_storage_pool ) {
  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    libxsmm_meqn_exec_plan_visit_arg_node(cur_node);
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    /* First visit left child tree */
    libxsmm_meqn_create_exec_plan( cur_node->le, global_timestamp, n_max_tmp, tmp_storage_pool );
    libxsmm_meqn_exec_plan_visit_unary_node(cur_node, global_timestamp, n_max_tmp, tmp_storage_pool);
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    /* First we visit the child tree with the maximum register score */
    if (cur_node->le->reg_score >= cur_node->ri->reg_score) {
      libxsmm_meqn_create_exec_plan( cur_node->le, global_timestamp, n_max_tmp, tmp_storage_pool );
      libxsmm_meqn_create_exec_plan( cur_node->ri, global_timestamp, n_max_tmp, tmp_storage_pool );
    } else {
      libxsmm_meqn_create_exec_plan( cur_node->ri, global_timestamp, n_max_tmp, tmp_storage_pool );
      libxsmm_meqn_create_exec_plan( cur_node->le, global_timestamp, n_max_tmp, tmp_storage_pool );
    }
    libxsmm_meqn_exec_plan_visit_binary_node(cur_node, global_timestamp, n_max_tmp, tmp_storage_pool);
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
    if ((cur_node->le->reg_score >= cur_node->ri->reg_score) && (cur_node->le->reg_score >= cur_node->r2->reg_score) ) {
      libxsmm_meqn_create_exec_plan( cur_node->le, global_timestamp, n_max_tmp, tmp_storage_pool );
      if ( cur_node->ri->reg_score >= cur_node->r2->reg_score ) {
        libxsmm_meqn_create_exec_plan( cur_node->ri, global_timestamp, n_max_tmp, tmp_storage_pool );
        libxsmm_meqn_create_exec_plan( cur_node->r2, global_timestamp, n_max_tmp, tmp_storage_pool );
      } else {
        libxsmm_meqn_create_exec_plan( cur_node->r2, global_timestamp, n_max_tmp, tmp_storage_pool );
        libxsmm_meqn_create_exec_plan( cur_node->ri, global_timestamp, n_max_tmp, tmp_storage_pool );
      }
    } else if ((cur_node->ri->reg_score >= cur_node->le->reg_score) && (cur_node->ri->reg_score >= cur_node->r2->reg_score) ) {
      libxsmm_meqn_create_exec_plan( cur_node->ri, global_timestamp, n_max_tmp, tmp_storage_pool );
      if ( cur_node->le->reg_score >= cur_node->r2->reg_score ) {
        libxsmm_meqn_create_exec_plan( cur_node->le, global_timestamp, n_max_tmp, tmp_storage_pool );
        libxsmm_meqn_create_exec_plan( cur_node->r2, global_timestamp, n_max_tmp, tmp_storage_pool );
      } else {
        libxsmm_meqn_create_exec_plan( cur_node->r2, global_timestamp, n_max_tmp, tmp_storage_pool );
        libxsmm_meqn_create_exec_plan( cur_node->le, global_timestamp, n_max_tmp, tmp_storage_pool );
      }
    } else {
      libxsmm_meqn_create_exec_plan( cur_node->r2, global_timestamp, n_max_tmp, tmp_storage_pool );
      if ( cur_node->le->reg_score >= cur_node->ri->reg_score ) {
        libxsmm_meqn_create_exec_plan( cur_node->le, global_timestamp, n_max_tmp, tmp_storage_pool );
        libxsmm_meqn_create_exec_plan( cur_node->ri, global_timestamp, n_max_tmp, tmp_storage_pool );
      } else {
        libxsmm_meqn_create_exec_plan( cur_node->ri, global_timestamp, n_max_tmp, tmp_storage_pool );
        libxsmm_meqn_create_exec_plan( cur_node->le, global_timestamp, n_max_tmp, tmp_storage_pool );
      }
    }
    libxsmm_meqn_exec_plan_visit_ternary_node(cur_node, global_timestamp, n_max_tmp, tmp_storage_pool);
  } else {
    /* This should not happen */
  }
}

LIBXSMM_API_INTERN
int libxsmm_meqn_is_unary_opcode_reduce_cols_idx_kernel (unsigned int opcode) {
  int result = 0;
  if ((opcode == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_ADD) ||
      (opcode == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MAX) ||
      (opcode == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MIN)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INTERN
int libxsmm_meqn_is_unary_opcode_reduce_kernel (unsigned int opcode) {
  int result = 0;
  if ((opcode == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD) ||
      (opcode == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX) ||
      (opcode == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ABSMAX) ||
      (opcode == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MIN) ||
      (opcode == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MUL) ||
      (opcode == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD) ||
      (opcode == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD_NCNC_FORMAT) ||
      (opcode == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INTERN
int libxsmm_meqn_is_unary_opcode_transform_kernel (unsigned int opcode) {
  int result = 0;
  if ( (opcode == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2)     ||
       (opcode == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT)     ||
       (opcode == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T)   ||
       (opcode == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2T)    ||
       (opcode == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD) ||
       (opcode == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4)     ||
       (opcode == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4T)    ||
       (opcode == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_NORM)     ||
       (opcode == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI2)    ||
       (opcode == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T)   ||
       (opcode == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4T_TO_NORM)    ||
       (opcode == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2T_TO_NORM)    ||
       (opcode == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD) ||
       (opcode == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2)         ||
       (opcode == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2)         ||
       (opcode == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2)        ||
       (opcode == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD4)         ||
       (opcode == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD4)         ||
       (opcode == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD4)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INTERN
int libxsmm_meqn_is_unary_opcode_reduce_to_scalar (unsigned int opcode) {
  int result = 0;
  if (opcode == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_TO_SCALAR_OP_ADD) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INTERN
int libxsmm_meqn_is_binary_opcode_reduce_to_scalar (unsigned int opcode) {
  int result = 0;
  if (opcode == LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INTERN void libxsmm_meqn_adjust_tmp_sizes( libxsmm_meqn_elem* cur_node ) {
  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    /* Do nothing */
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    libxsmm_meqn_adjust_tmp_sizes( cur_node->le );
    /* If it is reduce kernel, have to resize out tmp */
    if ( libxsmm_meqn_is_unary_opcode_reduce_kernel(cur_node->info.u_op.type) > 0 ) {
      if ((cur_node->info.u_op.flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS) > 0) {
        cur_node->tmp.m = cur_node->le->tmp.n;
        cur_node->tmp.n = 1;
        cur_node->tmp.ld = cur_node->le->tmp.n;
      } else if ((cur_node->info.u_op.flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS) > 0) {
        cur_node->tmp.m = cur_node->le->tmp.m;
        cur_node->tmp.n = 1;
        cur_node->tmp.ld = cur_node->le->tmp.m;
      }
    } else if ( libxsmm_meqn_is_unary_opcode_reduce_cols_idx_kernel(cur_node->info.u_op.type) > 0 ) {
      cur_node->tmp.m = cur_node->le->tmp.m;
      cur_node->tmp.n = 1;
      cur_node->tmp.ld = cur_node->le->tmp.m;
    } else if ( libxsmm_meqn_is_unary_opcode_reduce_to_scalar(cur_node->info.u_op.type) > 0 ) {
      cur_node->tmp.m = 1;
      cur_node->tmp.n = 1;
      cur_node->tmp.ld = 1;
    } else if ( libxsmm_meqn_is_unary_opcode_transform_kernel(cur_node->info.u_op.type) > 0 ) {
      cur_node->tmp.m = cur_node->le->tmp.n;
      cur_node->tmp.n = cur_node->le->tmp.m;
      cur_node->tmp.ld = cur_node->le->tmp.n;
    } else {
      cur_node->tmp.m = cur_node->le->tmp.m;
      cur_node->tmp.n = cur_node->le->tmp.n;
      cur_node->tmp.ld = cur_node->le->tmp.m;
    }
    cur_node->max_tmp_size = LIBXSMM_MAX(cur_node->max_tmp_size,  cur_node->tmp.m * cur_node->tmp.n);
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    libxsmm_meqn_adjust_tmp_sizes( cur_node->le);
    libxsmm_meqn_adjust_tmp_sizes( cur_node->ri);
    if ( libxsmm_meqn_is_binary_opcode_reduce_to_scalar(cur_node->info.b_op.type) > 0 ) {
      cur_node->tmp.m = 1;
      cur_node->tmp.n = 1;
      cur_node->tmp.ld = 1;
    } else if ((cur_node->info.b_op.is_matmul == 1) || (cur_node->info.b_op.is_brgemm == 1)) {
      cur_node->tmp.m = cur_node->le->tmp.m;
      cur_node->tmp.n = cur_node->ri->tmp.n;
      cur_node->tmp.ld = cur_node->le->tmp.m;
    } else {
      cur_node->tmp.m = LIBXSMM_MAX(cur_node->le->tmp.m, cur_node->ri->tmp.m);
      cur_node->tmp.n = LIBXSMM_MAX(cur_node->le->tmp.n, cur_node->ri->tmp.n);
      cur_node->tmp.ld = LIBXSMM_MAX(cur_node->le->tmp.m, cur_node->ri->tmp.m);
    }
    cur_node->max_tmp_size = LIBXSMM_MAX(cur_node->max_tmp_size,  cur_node->tmp.m * cur_node->tmp.n);
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
    libxsmm_meqn_adjust_tmp_sizes( cur_node->le );
    libxsmm_meqn_adjust_tmp_sizes( cur_node->ri);
    libxsmm_meqn_adjust_tmp_sizes( cur_node->r2);
    if ((cur_node->info.t_op.is_matmul == 1) || (cur_node->info.t_op.is_brgemm == 1)) {
      cur_node->tmp.m = cur_node->r2->tmp.m;
      cur_node->tmp.n = cur_node->r2->tmp.n;
      cur_node->tmp.ld = cur_node->r2->tmp.ld;
    } else {
      cur_node->tmp.m = LIBXSMM_MAX(cur_node->r2->tmp.m, LIBXSMM_MAX(cur_node->le->tmp.m, cur_node->ri->tmp.m));
      cur_node->tmp.n = LIBXSMM_MAX(cur_node->r2->tmp.n, LIBXSMM_MAX(cur_node->le->tmp.n, cur_node->ri->tmp.n));
      cur_node->tmp.ld = LIBXSMM_MAX( cur_node->r2->tmp.m, LIBXSMM_MAX(cur_node->le->tmp.m, cur_node->ri->tmp.m));
    }
    cur_node->max_tmp_size = LIBXSMM_MAX(cur_node->max_tmp_size,  cur_node->tmp.m * cur_node->tmp.n);
  }
}

LIBXSMM_API_INTERN void libxsmm_meqn_opt_exec_plan( libxsmm_blasint idx );
LIBXSMM_API_INTERN void libxsmm_meqn_opt_exec_plan( libxsmm_blasint idx ) {
  libxsmm_blasint global_timestamp = 0;
  libxsmm_blasint max_reg_score = 0;
  libxsmm_blasint *tmp_storage_pool = NULL;
  assert(NULL != libxsmm_matrix_eqns);
  if ( libxsmm_matrix_eqns[idx] == NULL ) {
    fprintf( stderr, "the requested equation does not exist, nothing to optimize!\n" );
    return;
  }
  else if ( libxsmm_matrix_eqns[idx]->is_constructed == 0 ) {
    fprintf( stderr, "the requested equation is not yet finalized, so cannot optimize!\n" );
  }
#if 0
  printf("\n");
  printf("Assigning register scores to find optimal traversal plan (i.e. that minimizes tmp storage)... \n");
#endif
  assert(NULL != libxsmm_matrix_eqns[idx]);
  libxsmm_meqn_propagate_tmp_info( libxsmm_matrix_eqns[idx]->eqn_root );
  libxsmm_meqn_assign_reg_scores( libxsmm_matrix_eqns[idx]->eqn_root );
  max_reg_score = libxsmm_matrix_eqns[idx]->eqn_root->reg_score;
  tmp_storage_pool = (libxsmm_blasint*) calloc(max_reg_score, sizeof(libxsmm_blasint));
  if (tmp_storage_pool == NULL) {
    fprintf(stderr, "Tmp storage allocation array failed...\n");
    return;
  }
#if 0
  printf("Optimal number of intermediate tmp storage is %d\n", max_reg_score);
#endif
  libxsmm_meqn_create_exec_plan( libxsmm_matrix_eqns[idx]->eqn_root, &global_timestamp, max_reg_score, tmp_storage_pool );
  libxsmm_meqn_adjust_tmp_sizes( libxsmm_matrix_eqns[idx]->eqn_root );
  libxsmm_meqn_reassign_bcast_tmp( libxsmm_matrix_eqns[idx] );
#if 0
  printf("Created optimal execution plan...\n");
#endif
  free(tmp_storage_pool);
#if 0
  printf("\n\n");
#endif
  libxsmm_matrix_eqns[idx]->is_optimized = 1;
}

LIBXSMM_API_INTERN
void libxsmm_meqn_reoptimize(libxsmm_matrix_eqn *eqn) {
  libxsmm_blasint max_reg_score = 0, global_timestamp = 0;
  libxsmm_blasint *tmp_storage_pool = NULL;
  libxsmm_meqn_assign_reg_scores( eqn->eqn_root );
  max_reg_score = eqn->eqn_root->reg_score;
  tmp_storage_pool = (libxsmm_blasint*) calloc(max_reg_score, sizeof(libxsmm_blasint));
  if (tmp_storage_pool == NULL) {
    fprintf(stderr, "Tmp storage allocation array failed...\n");
    return;
  }
  libxsmm_meqn_create_exec_plan( eqn->eqn_root, &global_timestamp, max_reg_score, tmp_storage_pool );
  libxsmm_meqn_adjust_tmp_sizes( eqn->eqn_root );
  if (tmp_storage_pool != NULL) {
    free(tmp_storage_pool);
  }
}

LIBXSMM_API_INTERN libxsmm_meqn_elem* libxsmm_meqn_add_node( libxsmm_meqn_elem* cur_node, libxsmm_meqn_node_type type, libxsmm_meqn_info info );
LIBXSMM_API_INTERN libxsmm_meqn_elem* libxsmm_meqn_add_node( libxsmm_meqn_elem* cur_node, libxsmm_meqn_node_type type, libxsmm_meqn_info info ) {
  if ( type == LIBXSMM_MATRIX_EQN_NODE_NONE ) {
    /* should not happen */
    fprintf( stderr, "wrong op node type to add!\n");
  }

  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    libxsmm_meqn_elem *node = (libxsmm_meqn_elem*) malloc( sizeof(libxsmm_meqn_elem) );
    assert(NULL != node);
    node->le = NULL;
    node->ri = NULL;
    node->r2 = NULL;
    node->up = cur_node;
    node->type = type;
    node->info = info;
    if ( cur_node->le == NULL ) {
      cur_node->le = node;
    } else {
      /* should not happen */
      fprintf( stderr, "this is not a leaf node, so we cannot add a node!\n");
      free( node );
      node = NULL;
    }
    return node;
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    libxsmm_meqn_elem *node = (libxsmm_meqn_elem*) malloc( sizeof(libxsmm_meqn_elem) );
    assert(NULL != node);
    node->le = NULL;
    node->ri = NULL;
    node->r2 = NULL;
    node->up = cur_node;
    node->type = type;
    node->info = info;
    if ( cur_node->le == NULL ) {
      cur_node->le = node;
    } else if ( cur_node->ri == NULL ) {
      cur_node->ri = node;
    } else {
      /* should not happen */
      fprintf( stderr, "this is not a leaf node, so we cannot add a node!\n");
      free( node );
      node = NULL;
    }
    return node;
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
    libxsmm_meqn_elem *node = (libxsmm_meqn_elem*) malloc( sizeof(libxsmm_meqn_elem) );
    assert(NULL != node);
    node->le = NULL;
    node->ri = NULL;
    node->r2 = NULL;
    node->up = cur_node;
    node->type = type;
    node->info = info;
    if ( cur_node->le == NULL ) {
      cur_node->le = node;
    } else if ( cur_node->ri == NULL ) {
      cur_node->ri = node;
    } else if ( cur_node->r2 == NULL ) {
      cur_node->r2 = node;
    } else {
      /* should not happen */
      fprintf( stderr, "this is not a leaf node, so we cannot add a node!\n");
      free( node );
      node = NULL;
    }
    return node;
  /* we converting the root */
  } else if ( (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_NONE) && (type != LIBXSMM_MATRIX_EQN_NODE_ARG) ) {
    cur_node->le = NULL;
    cur_node->ri = NULL;
    cur_node->r2 = NULL;
    cur_node->up = NULL;
    cur_node->type = type;
    cur_node->info = info;
    return cur_node;
  } else {
    /* should not happen */
    fprintf( stderr, "at this position we cannot add an op!\n");
  }
  return NULL;
}


LIBXSMM_API_INTERN libxsmm_meqn_elem* libxsmm_meqn_trv_head( libxsmm_meqn_elem* cur_node );
LIBXSMM_API_INTERN libxsmm_meqn_elem* libxsmm_meqn_trv_head( libxsmm_meqn_elem* cur_node ) {
  /* check if we are at an argument leaf, then we move up */
  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    return libxsmm_meqn_trv_head( cur_node->up );
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    /* we have to push more in this branch */
    if ( cur_node->le == NULL ) {
      return cur_node;
    /* we have reached the root, as we are unary, there is no right branch */
    } else if ( cur_node->up == NULL ) {
      return cur_node;
    /* we have to find another node */
    } else {
      return libxsmm_meqn_trv_head( cur_node->up );
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    /* we have to push more in this branch */
    if ( cur_node->le == NULL ) {
      return cur_node;
    } else if ( cur_node->ri == NULL ) {
      return cur_node;
    /* we have reached the root, as we are unary, there is no right branch */
    } else if ( cur_node->up == NULL ) {
      return cur_node;
    /* we have to find another node */
    } else {
      return libxsmm_meqn_trv_head( cur_node->up );
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
    /* we have to push more in this branch */
    if ( cur_node->le == NULL ) {
      return cur_node;
    } else if ( cur_node->ri == NULL ) {
      return cur_node;
    } else if ( cur_node->r2 == NULL ) {
      return cur_node;
    /* we have reached the root, as we are unary, there is no right branch */
    } else if ( cur_node->up == NULL ) {
      return cur_node;
    /* we have to find another node */
    } else {
      return libxsmm_meqn_trv_head( cur_node->up );
    }
   } else {
    /* should not happen */
  }

  return NULL;
}


LIBXSMM_API_INTERN void libxsmm_meqn_trv_print( libxsmm_meqn_elem* cur_node, libxsmm_blasint indent );
LIBXSMM_API_INTERN void libxsmm_meqn_trv_print( libxsmm_meqn_elem* cur_node, libxsmm_blasint indent ) {
  libxsmm_blasint i;
  libxsmm_blasint tree_print_indent = 4;

  for ( i = 0; i < indent; ++i ) {
    if ( i < indent - tree_print_indent ) {
      printf(" ");
    } else {
      if ( i % tree_print_indent == 0 ) {
        printf("|");
      } else {
        printf("-");
      }
    }
  }

  /* check if we are at an argument leaf, then we move up */
  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    if ( (cur_node->le == NULL) && (cur_node->ri == NULL) ) {
      printf("ARG: %i %i %i %i %i\n", cur_node->info.arg.m, cur_node->info.arg.n, cur_node->info.arg.ld, cur_node->info.arg.in_pos, cur_node->info.arg.offs_in_pos );
    } else {
      printf("ERROR: Arg cannot have left or right child!\n");
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    /* we have to push more in this branch */
    if ( cur_node->le != NULL ) {
      printf("UNARY: %i %i (timestamp = %i, tmp = %i)\n", (int)cur_node->info.u_op.type, (int)cur_node->info.u_op.flags, cur_node->visit_timestamp, cur_node->tmp.id );
      libxsmm_meqn_trv_print( cur_node->le, indent+tree_print_indent );
    /* we have reached the root, as we are unary, there is no right branch */
    } else if ( (cur_node->ri != NULL) ) {
      printf("ERROR: Unary cannot have right childs!\n");
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    /* we have to push more in this branch */
    if ( (cur_node->le != NULL) && (cur_node->ri != NULL) ) {
      printf("BINARY: %i %i (timestamp = %i, tmp = %i)\n", (int)cur_node->info.b_op.type, (int)cur_node->info.b_op.flags, cur_node->visit_timestamp, cur_node->tmp.id );
      libxsmm_meqn_trv_print( cur_node->le, indent+tree_print_indent );
      libxsmm_meqn_trv_print( cur_node->ri, indent+tree_print_indent );
    } else {
      printf("ERROR: Binary needs left and right child!\n");
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
    /* we have to push more in this branch */
    if ( (cur_node->le != NULL) && (cur_node->ri != NULL) && (cur_node->r2 != NULL) ) {
      printf("TERNARY: %i %i (timestamp = %i, tmp = %i)\n", (int)cur_node->info.t_op.type, (int)cur_node->info.t_op.flags, cur_node->visit_timestamp, cur_node->tmp.id );
      libxsmm_meqn_trv_print( cur_node->le, indent+tree_print_indent );
      libxsmm_meqn_trv_print( cur_node->ri, indent+tree_print_indent );
      libxsmm_meqn_trv_print( cur_node->r2, indent+tree_print_indent );
    } else {
      printf("ERROR: Ternary needs left, right and right2 child!\n");
    }
  } else {
    /* should not happen */
  }
}


LIBXSMM_API_INTERN void libxsmm_meqn_trv_rpn_print( libxsmm_meqn_elem* cur_node );
LIBXSMM_API_INTERN void libxsmm_meqn_trv_rpn_print( libxsmm_meqn_elem* cur_node ) {
  /* check if we are at an argument leaf, then we move up */
  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    if ( (cur_node->le == NULL) && (cur_node->ri == NULL) ) {
      printf("ARG ");
    } else {
      printf("ERROR: Arg cannot have left or right child!\n");
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    /* we have to push more in this branch */
    if ( cur_node->le != NULL ) {
      libxsmm_meqn_trv_rpn_print( cur_node->le );
      printf("UNARY-%i ", (int)cur_node->info.u_op.type );
    /* we have reached the root, as we are unary, there is no right branch */
    } else if ( (cur_node->ri != NULL) ) {
      printf("ERROR: Unary cannot have right childs!\n");
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    /* we have to push more in this branch */
    if ( (cur_node->le != NULL) && (cur_node->ri != NULL) ) {
      libxsmm_meqn_trv_rpn_print( cur_node->le );
      libxsmm_meqn_trv_rpn_print( cur_node->ri );
      printf("BINARY-%i ", (int)cur_node->info.b_op.type );
    } else {
      printf("ERROR: Binary needs left and right child!\n");
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
    /* we have to push more in this branch */
    if ( (cur_node->le != NULL) && (cur_node->ri != NULL) && (cur_node->r2 != NULL) ) {
      libxsmm_meqn_trv_rpn_print( cur_node->le );
      libxsmm_meqn_trv_rpn_print( cur_node->ri );
      libxsmm_meqn_trv_rpn_print( cur_node->r2 );
      printf("TERNARY-%i ", (int)cur_node->info.t_op.type );
    } else {
      printf("ERROR: Ternary needs left, right and right2 child!\n");
    }
  } else {
    /* should not happen */
  }
}


LIBXSMM_API_INTERN void libxsmm_meqn_mov_head( libxsmm_blasint idx );
LIBXSMM_API_INTERN void libxsmm_meqn_mov_head( libxsmm_blasint idx ) {
  assert(NULL !=libxsmm_matrix_eqns);
  if ( libxsmm_matrix_eqns[idx] == NULL ) {
    fprintf( stderr, "the requested equation does not exist!\n" );
    return;
  }
  if ( libxsmm_matrix_eqns[idx]->is_constructed == 1 ) {
    fprintf( stderr, "the requested equation is already finalized!\n" );
  }

  libxsmm_matrix_eqns[idx]->eqn_cur = libxsmm_meqn_trv_head( libxsmm_matrix_eqns[idx]->eqn_cur );

#if 0
  printf("cur node address: %lld\n", libxsmm_matrix_eqns[idx]->eqn_cur );
#endif

  /* let's see if we need seal the equation */
  if ( (libxsmm_matrix_eqns[idx]->eqn_cur == libxsmm_matrix_eqns[idx]->eqn_root) &&
       ( ((libxsmm_matrix_eqns[idx]->eqn_cur->type == LIBXSMM_MATRIX_EQN_NODE_UNARY)   && (libxsmm_matrix_eqns[idx]->eqn_cur->le != NULL)) ||
         ((libxsmm_matrix_eqns[idx]->eqn_cur->type == LIBXSMM_MATRIX_EQN_NODE_BINARY)  && (libxsmm_matrix_eqns[idx]->eqn_cur->ri != NULL)) ||
         ((libxsmm_matrix_eqns[idx]->eqn_cur->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) && (libxsmm_matrix_eqns[idx]->eqn_cur->r2 != NULL))    ) ) {
    libxsmm_matrix_eqns[idx]->is_constructed = 1;
    libxsmm_meqn_opt_exec_plan( idx );
  }
}


LIBXSMM_API_INTERN int libxsmm_meqn_is_ready_for_jit( libxsmm_blasint eqn_idx ) {
  if ( libxsmm_matrix_eqns[eqn_idx] == NULL ) {
    fprintf( stderr, "the requested equation does not exist!\n" );
    return 1;
  }
  if ( libxsmm_matrix_eqns[eqn_idx]->is_constructed == 0 ) {
    fprintf( stderr, "the requested equation is not finalized, yet!\n" );
    return 2;
  }
  if ( libxsmm_matrix_eqns[eqn_idx]->is_optimized == 0 ) {
    fprintf( stderr, "the requested equation is not optimized, yet!\n" );
    return 2;
  }

  return 0;
}


LIBXSMM_API libxsmm_blasint libxsmm_meqn_create(void) {
  libxsmm_blasint ret = libxsmm_matrix_eqns_count;
  libxsmm_meqn_elem* node;

  assert(NULL != libxsmm_matrix_eqns);
  /* lazy init of helper array */
  if ( libxsmm_matrix_eqns_init == 0 ) {
    libxsmm_blasint i;
    for ( i = 0; i < LIBXSMM_MAX_EQN_COUNT; ++i ) {
      libxsmm_matrix_eqns[i] = NULL;
    }
    libxsmm_matrix_eqns_count = 0;
    libxsmm_matrix_eqns_init = 1;
  }

  if (ret >= LIBXSMM_MAX_EQN_COUNT) {
    fprintf(stderr, "Exceeded maximum number of equations (%d). Can't create requested equation...\n", LIBXSMM_MAX_EQN_COUNT);
    return -1;
  }

  libxsmm_matrix_eqns_count++;

  libxsmm_matrix_eqns[ret] = (libxsmm_matrix_eqn*) malloc( sizeof(libxsmm_matrix_eqn) );

  node = (libxsmm_meqn_elem*) malloc( sizeof(libxsmm_meqn_elem) );
  assert(NULL != node);
  node->le = NULL;
  node->ri = NULL;
  node->up = NULL;
  node->type = LIBXSMM_MATRIX_EQN_NODE_NONE;
  libxsmm_matrix_eqns[ret]->eqn_root = node;
  libxsmm_matrix_eqns[ret]->eqn_cur = node;
  libxsmm_matrix_eqns[ret]->is_constructed = 0;
  libxsmm_matrix_eqns[ret]->is_optimized = 0;
  libxsmm_matrix_eqns[ret]->unary_only = 0;
  libxsmm_matrix_eqns[ret]->unary_only = 0;
#if 0
  printf("created equation no: %i\n", ret);
  printf("root node address: %lld\n", libxsmm_matrix_eqns[ret]->eqn_cur );
#endif

  return ret;
}

LIBXSMM_API libxsmm_meqn_arg_shape libxsmm_create_meqn_arg_shape( const libxsmm_blasint m, const libxsmm_blasint n, const libxsmm_blasint ld, const libxsmm_datatype type ) {
  libxsmm_meqn_arg_shape res /*= { 0 }*/;

  res.m = m;
  res.n = n;
  res.ld = ld;
  res.type = type;

  return res;
}

LIBXSMM_API libxsmm_matrix_arg_attributes libxsmm_create_matrix_arg_attributes( const libxsmm_matrix_arg_type type, const libxsmm_matrix_arg_set_type set_type, const libxsmm_blasint set_cardinality_hint, const libxsmm_blasint set_stride_hint ) {
  libxsmm_matrix_arg_attributes res /*= { {0} }*/;

  res.type = type;
  res.set_type = set_type;
  res.set_cardinality_hint = set_cardinality_hint;
  res.set_stride_hint = set_stride_hint;

  return res;
}

LIBXSMM_API libxsmm_meqn_arg_metadata libxsmm_create_meqn_arg_metadata( const libxsmm_blasint eqn_idx, const libxsmm_blasint in_arg_pos ) {
  libxsmm_meqn_arg_metadata res /*= { 0 }*/;

  res.eqn_idx = eqn_idx;
  res.in_arg_pos = in_arg_pos;

  return res;
}

LIBXSMM_API libxsmm_meqn_op_metadata libxsmm_create_meqn_op_metadata( const libxsmm_blasint eqn_idx, const libxsmm_blasint op_arg_pos ) {
  libxsmm_meqn_op_metadata res /*= { 0 }*/;

  res.eqn_idx = eqn_idx;
  res.op_arg_pos = op_arg_pos;

  return res;
}

LIBXSMM_API int libxsmm_meqn_push_back_arg( const libxsmm_meqn_arg_metadata arg_metadata, const libxsmm_meqn_arg_shape arg_shape, libxsmm_matrix_arg_attributes arg_attr ) {
  union libxsmm_meqn_info info /*= { 0 }*/;
  libxsmm_blasint idx = arg_metadata.eqn_idx;

  if ( libxsmm_matrix_eqns[idx] == NULL ) {
    fprintf( stderr, "the requested equation does not exist!\n" );
    return 1;
  }
  if ( libxsmm_matrix_eqns[idx]->is_constructed == 1 ) {
    fprintf( stderr, "the requested equation is already finalized!\n" );
    return 2;
  }

  info.arg.m = arg_shape.m;
  info.arg.n = arg_shape.n;
  info.arg.ld = arg_shape.ld;
  info.arg.in_pos = arg_metadata.in_arg_pos;
  info.arg.dtype = arg_shape.type;
  info.arg.arg_attr = arg_attr;
  libxsmm_matrix_eqns[idx]->eqn_cur = libxsmm_meqn_add_node( libxsmm_matrix_eqns[idx]->eqn_cur, LIBXSMM_MATRIX_EQN_NODE_ARG, info );
#if 0
  printf("added arg node: %lld %i %i %i %i %i %i\n", libxsmm_matrix_eqns[idx]->eqn_cur, M, N, ld, in_pos, offs_in_pos, dtype );
#endif

  /* move to the next head position in the tree */
  libxsmm_meqn_mov_head( idx );

  return 0;
}


LIBXSMM_API int libxsmm_meqn_push_back_unary_op(const libxsmm_meqn_op_metadata op_metadata, const libxsmm_meltw_unary_type type, const libxsmm_datatype dtype, const libxsmm_bitfield flags) {
  union libxsmm_meqn_info info /*= { 0 }*/;
  libxsmm_blasint idx = op_metadata.eqn_idx;

  if ( libxsmm_matrix_eqns[idx] == NULL ) {
    fprintf( stderr, "the requested equation does not exist!\n" );
    return 1;
  }
  if ( libxsmm_matrix_eqns[idx]->is_constructed == 1 ) {
    fprintf( stderr, "the requested equation is already finalized!\n" );
    return 2;
  }

  info.u_op.type  = type;
  info.u_op.flags = flags;
  info.u_op.dtype = (dtype == LIBXSMM_DATATYPE_IMPLICIT) ? LIBXSMM_DATATYPE_F32 : dtype;
  info.u_op.op_arg_pos = op_metadata.op_arg_pos;
  libxsmm_matrix_eqns[idx]->eqn_cur = libxsmm_meqn_add_node( libxsmm_matrix_eqns[idx]->eqn_cur, LIBXSMM_MATRIX_EQN_NODE_UNARY, info );
#if 0
  printf("added unary node: %lld %i %i %i\n", libxsmm_matrix_eqns[idx]->eqn_cur, type, flags, dtype );
#endif

  /* move to the next head position in the tree */
  libxsmm_meqn_mov_head( idx );

  return 0;
}

LIBXSMM_API int libxsmm_meqn_push_back_binary_op(const libxsmm_meqn_op_metadata op_metadata, const libxsmm_meltw_binary_type type, const libxsmm_datatype dtype, const libxsmm_bitfield flags) {
  union libxsmm_meqn_info info /*= { 0 }*/;
  libxsmm_blasint idx = op_metadata.eqn_idx;
  unsigned int is_brgemm = ((type ==  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM) ||
                            (type ==  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_B_TRANS) ||
                            (type ==  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_TRANS) ||
                            (type ==  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_TRANS_B_TRANS) ||
                            (type ==  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_VNNI) ||
                            (type ==  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_VNNI_B_TRANS) ||
                            (type ==  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_VNNI_TRANS) ||
                            (type ==  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_VNNI_TRANS_B_TRANS)) ? 1 : 0;

  unsigned int is_matmul = ((type ==  LIBXSMM_MELTW_TYPE_BINARY_MATMUL) ||
                            (type ==  LIBXSMM_MELTW_TYPE_BINARY_MATMUL_B_TRANS) ||
                            (type ==  LIBXSMM_MELTW_TYPE_BINARY_MATMUL_A_TRANS) ||
                            (type ==  LIBXSMM_MELTW_TYPE_BINARY_MATMUL_A_TRANS_B_TRANS) ||
                            (type ==  LIBXSMM_MELTW_TYPE_BINARY_MATMUL_A_VNNI) ||
                            (type ==  LIBXSMM_MELTW_TYPE_BINARY_MATMUL_A_VNNI_B_TRANS) ||
                            (type ==  LIBXSMM_MELTW_TYPE_BINARY_MATMUL_A_VNNI_TRANS) ||
                            (type ==  LIBXSMM_MELTW_TYPE_BINARY_MATMUL_A_VNNI_TRANS_B_TRANS)) ? 1 : 0;

  if ( libxsmm_matrix_eqns[idx] == NULL ) {
    fprintf( stderr, "the requested equation does not exist!\n" );
    return 1;
  }
  if ( libxsmm_matrix_eqns[idx]->is_constructed == 1 ) {
    fprintf( stderr, "the requested equation is already finalized!\n" );
    return 2;
  }

  info.b_op.type  = type;
  info.b_op.flags = flags;
  info.b_op.dtype = (dtype == LIBXSMM_DATATYPE_IMPLICIT) ? LIBXSMM_DATATYPE_F32 : dtype;
  info.b_op.op_arg_pos = op_metadata.op_arg_pos;
  info.b_op.is_matmul  = is_matmul;
  info.b_op.is_brgemm  = is_brgemm;
  libxsmm_matrix_eqns[idx]->eqn_cur = libxsmm_meqn_add_node( libxsmm_matrix_eqns[idx]->eqn_cur, LIBXSMM_MATRIX_EQN_NODE_BINARY, info );
#if 0
  printf("added binary node: %lld %i %i %i\n", libxsmm_matrix_eqns[idx]->eqn_cur, type, flags, dtype );
#endif

  /* move to the next head position in the tree */
  libxsmm_meqn_mov_head( idx );

  return 0;
}

LIBXSMM_API int libxsmm_meqn_push_back_ternary_op(const libxsmm_meqn_op_metadata op_metadata, const libxsmm_meltw_ternary_type type, const libxsmm_datatype dtype, const libxsmm_bitfield flags) {
  union libxsmm_meqn_info info /*= { 0 }*/;
  libxsmm_blasint idx = op_metadata.eqn_idx;
  unsigned int is_brgemm = ((type ==  LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM) ||
                            (type ==  LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM_B_TRANS) ||
                            (type ==  LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM_A_TRANS) ||
                            (type ==  LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM_A_TRANS_B_TRANS) ||
                            (type ==  LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM_A_VNNI) ||
                            (type ==  LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM_A_VNNI_B_TRANS) ||
                            (type ==  LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM_A_VNNI_TRANS) ||
                            (type ==  LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM_A_VNNI_TRANS_B_TRANS)) ? 1 : 0;

  unsigned int is_matmul = ((type ==  LIBXSMM_MELTW_TYPE_TERNARY_MATMUL) ||
                            (type ==  LIBXSMM_MELTW_TYPE_TERNARY_MATMUL_B_TRANS) ||
                            (type ==  LIBXSMM_MELTW_TYPE_TERNARY_MATMUL_A_TRANS) ||
                            (type ==  LIBXSMM_MELTW_TYPE_TERNARY_MATMUL_A_TRANS_B_TRANS) ||
                            (type ==  LIBXSMM_MELTW_TYPE_TERNARY_MATMUL_A_VNNI) ||
                            (type ==  LIBXSMM_MELTW_TYPE_TERNARY_MATMUL_A_VNNI_B_TRANS) ||
                            (type ==  LIBXSMM_MELTW_TYPE_TERNARY_MATMUL_A_VNNI_TRANS) ||
                            (type ==  LIBXSMM_MELTW_TYPE_TERNARY_MATMUL_A_VNNI_TRANS_B_TRANS)) ? 1 : 0;

  if ( libxsmm_matrix_eqns[idx] == NULL ) {
    fprintf( stderr, "the requested equation does not exist!\n" );
    return 1;
  }
  if ( libxsmm_matrix_eqns[idx]->is_constructed == 1 ) {
    fprintf( stderr, "the requested equation is already finalized!\n" );
    return 2;
  }

  info.t_op.type  = type;
  info.t_op.flags = flags;
  info.t_op.dtype = (dtype == LIBXSMM_DATATYPE_IMPLICIT) ? LIBXSMM_DATATYPE_F32 : dtype;
  info.t_op.op_arg_pos = op_metadata.op_arg_pos;
  info.t_op.is_matmul  = is_matmul;
  info.t_op.is_brgemm  = is_brgemm;
  libxsmm_matrix_eqns[idx]->eqn_cur = libxsmm_meqn_add_node( libxsmm_matrix_eqns[idx]->eqn_cur, LIBXSMM_MATRIX_EQN_NODE_TERNARY, info );
#if 0
  printf("added ternary node: %lld %i %i %i\n", libxsmm_matrix_eqns[idx]->eqn_cur, type, flags, dtype );
#endif

  /* move to the next head position in the tree */
  libxsmm_meqn_mov_head( idx );

  return 0;
}

LIBXSMM_API void libxsmm_meqn_tree_print( const libxsmm_blasint idx ) {
  assert(NULL != libxsmm_matrix_eqns);
  if ( libxsmm_matrix_eqns[idx] == NULL ) {
    fprintf( stderr, "the requested equation does not exist!\n" );
  }
  if ( libxsmm_matrix_eqns[idx]->is_constructed == 0 ) {
    fprintf( stderr, "the requested equation is not yet finalized!\n" );
  }

  printf("\n");
  printf("Schematic of the expression tree (Pre-order)\n");
  libxsmm_meqn_trv_print( libxsmm_matrix_eqns[idx]->eqn_root, 0 );
  printf("\n");
}


LIBXSMM_API void libxsmm_meqn_rpn_print( const libxsmm_blasint idx ) {
  assert(NULL != libxsmm_matrix_eqns);
  if ( libxsmm_matrix_eqns[idx] == NULL ) {
    fprintf( stderr, "the requested equation does not exist!\n" );
  }
  if ( libxsmm_matrix_eqns[idx]->is_constructed == 0 ) {
    fprintf( stderr, "the requested equation is not yet finalized!\n" );
  }

  printf("\n");
  printf("HP calculator (RPN) print of the expression tree (Post-order)\n");
  libxsmm_meqn_trv_rpn_print( libxsmm_matrix_eqns[idx]->eqn_root );
  printf("\n\n");
}
