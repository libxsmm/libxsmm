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
#include "generator_mateltwise_common.h"
#include "generator_common.h"
#include "generator_matequation_reference_impl.h"
#include "generator_mateltwise_reference_impl.h"

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_setup_out_ptrs( libxsmm_meqn_elem  *cur_node,
                                                   libxsmm_meqn_param *eqn_param,
                                                   unsigned int       is_last_op,
                                                   unsigned char      *scratch_ptr,
                                                   unsigned long long tmp_size,
                                                   unsigned char      **primary,
                                                   unsigned char      **secondary,
                                                   unsigned char      **tertiary,
                                                   unsigned char      **quaternary,
                                                   unsigned char      **quinary,
                                                   unsigned char      **senary) {
  if (is_last_op > 0) {
    *primary = (unsigned char*)eqn_param->output.primary;
  } else {
    if (cur_node->tmp.id >= 0) {
      *primary = (unsigned char*)scratch_ptr + (cur_node->tmp.id) * tmp_size;
    } else {
      libxsmm_blasint arg_tmp_id = -1-cur_node->tmp.id;
      *primary = (unsigned char*)eqn_param->inputs[arg_tmp_id].primary;
    }
  }

  /* Setup secondaries if need be */
  if ((cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) &&
      ((cur_node->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_UNZIP) ||
       (cur_node->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_SCATTER) ||
       ((cur_node->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_RELU) && ((cur_node->info.u_op.flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) ))) {
    if (is_last_op > 0) {
      *secondary = (unsigned char*)eqn_param->output.secondary;
    } else {
      if (cur_node->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_SCATTER) {
        fprintf( stderr, "The requested SCATTER operation can only be the head of the equation...\n" );
      } else if (cur_node->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_UNZIP) {
        fprintf( stderr, "The requested UNPACK_TO_BLOCKS operation can only be the head of the equation...\n" );
      } else if (cur_node->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_RELU) {
        fprintf( stderr, "The requested RELU operation with bitmask can only be the head of the equation...\n" );
      }
      return;
    }
  }

  /* If DUMP operator set secondary output in stack param struct */
  if ((cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (cur_node->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_DUMP)) {
    *secondary = (unsigned char*)eqn_param->ops_args[cur_node->info.u_op.op_arg_pos].primary;
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_setup_inp_ptrs( libxsmm_meqn_elem  *cur_node,
                                                   libxsmm_meqn_elem  *cur_node_up,
                                                   libxsmm_meqn_param *eqn_param,
                                                   unsigned char      *scratch_ptr,
                                                   unsigned long long tmp_size,
                                                   unsigned char      **primary,
                                                   unsigned char      **secondary,
                                                   unsigned char      **tertiary,
                                                   unsigned char      **quaternary,
                                                   unsigned char      **quinary,
                                                   unsigned char      **senary) {

  if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
    if (cur_node->info.arg.in_pos >= 0) {
      *primary = (unsigned char*)eqn_param->inputs[cur_node->info.arg.in_pos].primary;
    } else {
      *primary = (unsigned char*)scratch_ptr + (-1-cur_node->info.arg.in_pos) * tmp_size;
    }
  } else {
    if (cur_node->tmp.id >= 0) {
      *primary = (unsigned char*)scratch_ptr + (cur_node->tmp.id) * tmp_size;
    } else {
      libxsmm_blasint arg_tmp_id = -1-cur_node->tmp.id;
      *primary = (unsigned char*)eqn_param->inputs[arg_tmp_id].primary;
    }
  }

  if ((cur_node_up->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) &&
      ((cur_node_up->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_GATHER) ||
       (libxsmm_meqn_is_unary_opcode_reduce_cols_idx_kernel(cur_node_up->info.u_op.type) > 0))) {
    if ((cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->info.arg.in_pos >= 0)) {
      *secondary = (unsigned char*)eqn_param->inputs[cur_node->info.arg.in_pos].secondary;
    } else {
      fprintf( stderr, "The requested GATHER operation accepts arguments given by the user only...\n" );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_reference_matequation(void *param, void *i_execution_plan, void *scratchpad, unsigned long long tmp_size) {
  int i = 0;
  libxsmm_meqn_elem *unfolded_exec_tree = (libxsmm_meqn_elem*) i_execution_plan;
  libxsmm_descriptor_blob   blob;
  libxsmm_meltw_descriptor  *meltw_desc = NULL;
  for (i=0; ; ) {
    libxsmm_meqn_elem cur_tpp_node  = unfolded_exec_tree[5*i+0];
    libxsmm_meqn_elem left_tpp_node = unfolded_exec_tree[5*i+1];
    unsigned int is_last_tpp = (cur_tpp_node.reg_score == -1) ? 1 : 0;
    /* Here he discpatch the proper TPP by setting up the proper descriptor and the parameter */
    if (cur_tpp_node.type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
      libxsmm_meltw_unary_param unary_param;
      if ((libxsmm_meqn_is_unary_opcode_reduce_kernel(cur_tpp_node.info.u_op.type) > 0) || (libxsmm_meqn_is_unary_opcode_reduce_cols_idx_kernel(cur_tpp_node.info.u_op.type) > 0)) {
        meltw_desc = libxsmm_meltw_descriptor_init2(&blob, left_tpp_node.tmp.dtype, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
          cur_tpp_node.info.u_op.dtype, cur_tpp_node.tmp.dtype, left_tpp_node.tmp.m, left_tpp_node.tmp.n, left_tpp_node.tmp.ld, cur_tpp_node.tmp.ld, 0, 0,
          LIBXSMM_CAST_USHORT(cur_tpp_node.info.u_op.flags), LIBXSMM_CAST_USHORT(cur_tpp_node.info.u_op.type), LIBXSMM_MELTW_OPERATION_UNARY);
      } else {
        if ((cur_tpp_node.info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) && (left_tpp_node.tmp.dtype != cur_tpp_node.tmp.dtype)) {
          meltw_desc = libxsmm_meltw_descriptor_init2(&blob, left_tpp_node.tmp.dtype, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
            LIBXSMM_DATATYPE_F32, cur_tpp_node.tmp.dtype, cur_tpp_node.tmp.m, cur_tpp_node.tmp.n, left_tpp_node.tmp.ld, cur_tpp_node.tmp.ld, 0, 0,
            LIBXSMM_CAST_USHORT(cur_tpp_node.info.u_op.flags), LIBXSMM_CAST_USHORT(cur_tpp_node.info.u_op.type), LIBXSMM_MELTW_OPERATION_UNARY);
        } else {
          meltw_desc = libxsmm_meltw_descriptor_init2(&blob, left_tpp_node.tmp.dtype, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED,
            cur_tpp_node.info.u_op.dtype, cur_tpp_node.tmp.dtype, cur_tpp_node.tmp.m, cur_tpp_node.tmp.n, left_tpp_node.tmp.ld, cur_tpp_node.tmp.ld, 0, 0,
            LIBXSMM_CAST_USHORT(cur_tpp_node.info.u_op.flags), LIBXSMM_CAST_USHORT(cur_tpp_node.info.u_op.type), LIBXSMM_MELTW_OPERATION_UNARY);
        }
      }
      libxsmm_generator_matequation_setup_inp_ptrs( &left_tpp_node, &cur_tpp_node, (libxsmm_meqn_param*)param, (unsigned char*)scratchpad, tmp_size,
                                                   (unsigned char**)&unary_param.in.primary,
                                                   (unsigned char**)&unary_param.in.secondary,
                                                   (unsigned char**)&unary_param.in.tertiary,
                                                   (unsigned char**)&unary_param.in.quaternary,
                                                   (unsigned char**)&unary_param.in.quinary,
                                                   (unsigned char**)&unary_param.in.senary);
      libxsmm_generator_matequation_setup_out_ptrs( &cur_tpp_node, (libxsmm_meqn_param*)param, is_last_tpp, (unsigned char*)scratchpad, tmp_size,
                                                   (unsigned char**)&unary_param.out.primary,
                                                   (unsigned char**)&unary_param.out.secondary,
                                                   (unsigned char**)&unary_param.out.tertiary,
                                                   (unsigned char**)&unary_param.out.quaternary,
                                                   (unsigned char**)&unary_param.out.quinary,
                                                   (unsigned char**)&unary_param.out.senary);
      libxsmm_reference_elementwise((void*)&unary_param, (const libxsmm_meltw_descriptor*)meltw_desc);
    } else if (cur_tpp_node.type == LIBXSMM_MATRIX_EQN_NODE_BINARY) {
      libxsmm_meqn_elem right_tpp_node = unfolded_exec_tree[5*i+2];
      libxsmm_meltw_binary_param binary_param;
      if (libxsmm_meqn_is_binary_opcode_reduce_to_scalar(cur_tpp_node.info.b_op.type) > 0) {
        meltw_desc = libxsmm_meltw_descriptor_init2(&blob, left_tpp_node.tmp.dtype, right_tpp_node.tmp.dtype, LIBXSMM_DATATYPE_UNSUPPORTED,
          cur_tpp_node.info.b_op.dtype, cur_tpp_node.tmp.dtype, LIBXSMM_MAX(left_tpp_node.tmp.m, right_tpp_node.tmp.m), LIBXSMM_MAX(left_tpp_node.tmp.n, right_tpp_node.tmp.n), left_tpp_node.tmp.ld, cur_tpp_node.tmp.ld, right_tpp_node.tmp.ld, 0,
          LIBXSMM_CAST_USHORT(cur_tpp_node.info.b_op.flags), LIBXSMM_CAST_USHORT(cur_tpp_node.info.b_op.type), LIBXSMM_MELTW_OPERATION_BINARY);
      } else {
        meltw_desc = libxsmm_meltw_descriptor_init2(&blob, left_tpp_node.tmp.dtype, right_tpp_node.tmp.dtype, LIBXSMM_DATATYPE_UNSUPPORTED,
          cur_tpp_node.info.b_op.dtype, cur_tpp_node.tmp.dtype, cur_tpp_node.tmp.m, cur_tpp_node.tmp.n, left_tpp_node.tmp.ld, cur_tpp_node.tmp.ld, right_tpp_node.tmp.ld, 0,
          LIBXSMM_CAST_USHORT(cur_tpp_node.info.b_op.flags), LIBXSMM_CAST_USHORT(cur_tpp_node.info.b_op.type), LIBXSMM_MELTW_OPERATION_BINARY);
      }
      libxsmm_generator_matequation_setup_inp_ptrs( &left_tpp_node, &cur_tpp_node, (libxsmm_meqn_param*)param, (unsigned char*)scratchpad, tmp_size,
                                                   (unsigned char**)&binary_param.in0.primary,
                                                   (unsigned char**)&binary_param.in0.secondary,
                                                   (unsigned char**)&binary_param.in0.tertiary,
                                                   (unsigned char**)&binary_param.in0.quaternary,
                                                   (unsigned char**)&binary_param.in0.quinary,
                                                   (unsigned char**)&binary_param.in0.senary);
      libxsmm_generator_matequation_setup_inp_ptrs( &right_tpp_node, &cur_tpp_node, (libxsmm_meqn_param*)param, (unsigned char*)scratchpad, tmp_size,
                                                   (unsigned char**)&binary_param.in1.primary,
                                                   (unsigned char**)&binary_param.in1.secondary,
                                                   (unsigned char**)&binary_param.in1.tertiary,
                                                   (unsigned char**)&binary_param.in1.quaternary,
                                                   (unsigned char**)&binary_param.in1.quinary,
                                                   (unsigned char**)&binary_param.in1.senary);
      libxsmm_generator_matequation_setup_out_ptrs( &cur_tpp_node, (libxsmm_meqn_param*)param, is_last_tpp, (unsigned char*)scratchpad, tmp_size,
                                                   (unsigned char**)&binary_param.out.primary,
                                                   (unsigned char**)&binary_param.out.secondary,
                                                   (unsigned char**)&binary_param.out.tertiary,
                                                   (unsigned char**)&binary_param.out.quaternary,
                                                   (unsigned char**)&binary_param.out.quinary,
                                                   (unsigned char**)&binary_param.out.senary);
      libxsmm_reference_elementwise((void*)&binary_param, (const libxsmm_meltw_descriptor*)meltw_desc);
    } else {
      libxsmm_meqn_elem right_tpp_node  = unfolded_exec_tree[5*i+2];
      libxsmm_meqn_elem right2_tpp_node = unfolded_exec_tree[5*i+3];
      libxsmm_datatype in2_dtype = (cur_tpp_node.info.t_op.type == LIBXSMM_MELTW_TYPE_TERNARY_SELECT) ? LIBXSMM_DATATYPE_IMPLICIT : right2_tpp_node.tmp.dtype;

      libxsmm_meltw_ternary_param ternary_param;
      meltw_desc = libxsmm_meltw_descriptor_init2(&blob, left_tpp_node.tmp.dtype, right_tpp_node.tmp.dtype, in2_dtype,
        cur_tpp_node.info.t_op.dtype, cur_tpp_node.tmp.dtype, cur_tpp_node.tmp.m, cur_tpp_node.tmp.n, left_tpp_node.tmp.ld, cur_tpp_node.tmp.ld, right_tpp_node.tmp.ld, right2_tpp_node.tmp.ld,
        LIBXSMM_CAST_USHORT(cur_tpp_node.info.t_op.flags), LIBXSMM_CAST_USHORT(cur_tpp_node.info.t_op.type), LIBXSMM_MELTW_OPERATION_TERNARY);
      libxsmm_generator_matequation_setup_inp_ptrs( &left_tpp_node, &cur_tpp_node, (libxsmm_meqn_param*)param, (unsigned char*)scratchpad, tmp_size,
                                                   (unsigned char**)&ternary_param.in0.primary,
                                                   (unsigned char**)&ternary_param.in0.secondary,
                                                   (unsigned char**)&ternary_param.in0.tertiary,
                                                   (unsigned char**)&ternary_param.in0.quaternary,
                                                   (unsigned char**)&ternary_param.in0.quinary,
                                                   (unsigned char**)&ternary_param.in0.senary);
      libxsmm_generator_matequation_setup_inp_ptrs( &right_tpp_node, &cur_tpp_node, (libxsmm_meqn_param*)param, (unsigned char*)scratchpad, tmp_size,
                                                   (unsigned char**)&ternary_param.in1.primary,
                                                   (unsigned char**)&ternary_param.in1.secondary,
                                                   (unsigned char**)&ternary_param.in1.tertiary,
                                                   (unsigned char**)&ternary_param.in1.quaternary,
                                                   (unsigned char**)&ternary_param.in1.quinary,
                                                   (unsigned char**)&ternary_param.in1.senary);
      libxsmm_generator_matequation_setup_inp_ptrs( &right2_tpp_node, &cur_tpp_node, (libxsmm_meqn_param*)param, (unsigned char*)scratchpad, tmp_size,
                                                   (unsigned char**)&ternary_param.in2.primary,
                                                   (unsigned char**)&ternary_param.in2.secondary,
                                                   (unsigned char**)&ternary_param.in2.tertiary,
                                                   (unsigned char**)&ternary_param.in2.quaternary,
                                                   (unsigned char**)&ternary_param.in2.quinary,
                                                   (unsigned char**)&ternary_param.in2.senary);
      libxsmm_generator_matequation_setup_out_ptrs( &cur_tpp_node, (libxsmm_meqn_param*)param, is_last_tpp, (unsigned char*)scratchpad, tmp_size,
                                                   (unsigned char**)&ternary_param.out.primary,
                                                   (unsigned char**)&ternary_param.out.secondary,
                                                   (unsigned char**)&ternary_param.out.tertiary,
                                                   (unsigned char**)&ternary_param.out.quaternary,
                                                   (unsigned char**)&ternary_param.out.quinary,
                                                   (unsigned char**)&ternary_param.out.senary);
      libxsmm_reference_elementwise((void*)&ternary_param, (const libxsmm_meltw_descriptor*)meltw_desc);
    }
    i++;
    if (is_last_tpp > 0) {
      break;
    }
  }
}
