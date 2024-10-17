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
#include "generator_matequation_scratch_avx_avx512.h"
#include "generator_matequation_avx_avx512.h"

#include "generator_matequation_aarch64.h"
#include "generator_mateltwise_aarch64.h"
#include "generator_mateltwise_unary_binary_aarch64.h"
#include "generator_aarch64_instructions.h"
#include "generator_common.h"

#include "generator_common_aarch64.h"
#include "generator_matequation_scratch_aarch64.h"
#include "generator_mateltwise_gather_scatter_aarch64.h"
#include "generator_mateltwise_reduce_aarch64.h"
#include "generator_mateltwise_transform_common.h"


#if 0
LIBXSMM_API_INTERN
void libxsmm_generator_matequation_create_unary_descriptor(libxsmm_descriptor_blob *blob, libxsmm_meqn_elem *cur_op, libxsmm_meltw_descriptor **desc, libxsmm_datatype in_precision, libxsmm_datatype out_precision) {
  if (libxsmm_meqn_is_unary_opcode_transform_kernel(cur_op->info.u_op.type) > 0) {
    *desc = libxsmm_meltw_descriptor_init2(blob, in_precision, cur_op->info.u_op.dtype, out_precision, LIBXSMM_DATATYPE_UNSUPPORTED, cur_op->le->tmp.m, cur_op->le->tmp.n, cur_op->le->tmp.ld, cur_op->tmp.ld, 0, 0, (unsigned short)cur_op->info.u_op.flags, cur_op->info.u_op.type, LIBXSMM_MELTW_OPERATION_UNARY);
  } else if (libxsmm_meqn_is_unary_opcode_reduce_kernel(cur_op->info.u_op.type) > 0) {
    *desc = libxsmm_meltw_descriptor_init2(blob, in_precision, cur_op->info.u_op.dtype, out_precision, LIBXSMM_DATATYPE_UNSUPPORTED, cur_op->le->tmp.m, cur_op->le->tmp.n, cur_op->le->tmp.ld, cur_op->tmp.ld, 0, 0, (unsigned short)cur_op->info.u_op.flags, cur_op->info.u_op.type, LIBXSMM_MELTW_OPERATION_UNARY);
  } else {
    *desc = libxsmm_meltw_descriptor_init2(blob, in_precision, cur_op->info.u_op.dtype, out_precision, LIBXSMM_DATATYPE_UNSUPPORTED, cur_op->tmp.m, cur_op->tmp.n, cur_op->le->tmp.ld, cur_op->tmp.ld, 0, 0, (unsigned short)cur_op->info.u_op.flags, cur_op->info.u_op.type, LIBXSMM_MELTW_OPERATION_UNARY);
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_create_binary_descriptor(libxsmm_descriptor_blob *blob, libxsmm_meqn_elem *cur_op, libxsmm_meltw_descriptor **desc, libxsmm_datatype in_precision, libxsmm_datatype out_precision) {
  *desc = libxsmm_meltw_descriptor_init2(blob, in_precision, cur_op->info.b_op.dtype, out_precision, LIBXSMM_DATATYPE_UNSUPPORTED, cur_op->tmp.m, cur_op->tmp.n, cur_op->le->tmp.ld, cur_op->tmp.ld, cur_op->ri->tmp.ld, 0, (unsigned short)cur_op->info.b_op.flags, cur_op->info.b_op.type, LIBXSMM_MELTW_OPERATION_BINARY);
}
#endif

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_set_input_in_stack_param_struct_aarch64( libxsmm_generated_code*   io_generated_code,
    libxsmm_matequation_kernel_config*                  i_micro_kernel_config,
    libxsmm_matequation_gp_reg_mapping*                 i_gp_reg_mapping,
    libxsmm_meqn_elem*                            cur_node,
    unsigned int                                        temp_reg,
    unsigned int                                        ptr_id ) {

  if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
    if (cur_node->info.arg.in_pos >= 0) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 8, temp_reg );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, temp_reg, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg, (long long)cur_node->info.arg.in_pos*32 );
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, temp_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, temp_reg );
    } else {
      libxsmm_generator_meqn_getaddr_stack_tmp_i_aarch64( io_generated_code, (-1-cur_node->info.arg.in_pos) * i_micro_kernel_config->tmp_size, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg);
    }
  } else {
    libxsmm_generator_meqn_getaddr_stack_tmp_i_aarch64( io_generated_code, cur_node->tmp.id * i_micro_kernel_config->tmp_size, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg);
  }
  if (ptr_id == 0) {
    libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR4, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
  } else if (ptr_id == 1) {
    libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR8, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
  } else {
    libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR12, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
  }

  /* Setup secondaries if need be */
  if ((cur_node->up->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) &&
      ((cur_node->up->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_GATHER) ||
       (libxsmm_meqn_is_unary_opcode_reduce_cols_idx_kernel(cur_node->up->info.u_op.type) > 0))) {
    if ((cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG) && (cur_node->info.arg.in_pos >= 0)) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 8, temp_reg );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, temp_reg, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg, (long long)cur_node->info.arg.in_pos*32+8 );
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, temp_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, temp_reg );
      libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR5, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
    } else {
      fprintf( stderr, "The requested GATHER operation accepts arguments given by the user only...\n" );
      return;
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_set_output_in_stack_param_struct_aarch64(libxsmm_generated_code*   io_generated_code,
    libxsmm_matequation_kernel_config*                  i_micro_kernel_config,
    libxsmm_matequation_gp_reg_mapping*                 i_gp_reg_mapping,
    libxsmm_meqn_elem*                            cur_node,
    unsigned int                                        temp_reg,
    unsigned int                                        is_last_op ) {
  if (is_last_op > 0) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 16, temp_reg );
  } else {
    libxsmm_generator_meqn_getaddr_stack_tmp_i_aarch64( io_generated_code, cur_node->tmp.id * i_micro_kernel_config->tmp_size, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg);
  }
  if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
    libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR8, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
  } else if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ||
            (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY && (cur_node->info.t_op.type == LIBXSMM_MELTW_TYPE_TERNARY_MULADD ||  cur_node->info.t_op.type == LIBXSMM_MELTW_TYPE_TERNARY_NMULADD))) {
    libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR12, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
  } else if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) {
    libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR16, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
  } else {
    /* Should not happen */
  }

  /* Setup secondaries if need be */
  if ((cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) &&
      ((cur_node->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_UNZIP) ||
       (cur_node->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_SCATTER) ||
       ((cur_node->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_RELU) && ((cur_node->info.u_op.flags & LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT) > 0) ) )) {
    if (is_last_op > 0) {
      libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 24, temp_reg );
      libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR9, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
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
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_tmp_stack_scratch_aarch64_kernel( libxsmm_generated_code* io_generated_code,
    const libxsmm_meqn_descriptor*          i_mateqn_desc,
    libxsmm_matequation_gp_reg_mapping*     i_gp_reg_mapping,
    libxsmm_matequation_kernel_config*      i_micro_kernel_config,
    libxsmm_loop_label_tracker*             io_loop_label_tracker,
    libxsmm_matrix_eqn*                     eqn ) {

  libxsmm_descriptor_blob   blob;
  libxsmm_meltw_descriptor  *meltw_desc = NULL;
#if 0
  libxsmm_gemm_descriptor   *xgemm_desc = NULL;
#endif
  unsigned int timestamp = 0;
  unsigned int last_timestamp;
  unsigned int temp_reg = LIBXSMM_AARCH64_GP_REG_X26;

  i_gp_reg_mapping->temp_reg = temp_reg;
  i_gp_reg_mapping->gp_reg_scratch_0 = LIBXSMM_AARCH64_GP_REG_X27;

  if ( eqn == NULL ) {
    fprintf( stderr, "The requested equation does not exist... nothing to JIT,,,\n" );
    return;
  } else {
    last_timestamp = eqn->eqn_root->visit_timestamp;
  }

  i_gp_reg_mapping->gp_reg_mapping_eltwise.gp_reg_param_struct = LIBXSMM_AARCH64_GP_REG_X28;
  libxsmm_generator_meqn_getaddr_stack_var_aarch64(  io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR0, i_gp_reg_mapping->gp_reg_mapping_eltwise.gp_reg_param_struct );

  /* Iterate over the equation tree based on the optimal traversal order and call the proper JITer */
  for (timestamp = 0; timestamp <= last_timestamp; timestamp++) {
    libxsmm_meqn_elem *cur_op = libxsmm_generator_matequation_find_op_at_timestamp(eqn->eqn_root, timestamp);
#if 0
    libxsmm_datatype out_precision = (timestamp == last_timestamp) ? (libxsmm_datatype) LIBXSMM_GETENUM_OUT(i_mateqn_desc->datatype) : cur_op->tmp.dtype;
    libxsmm_datatype in_precision = cur_op->tmp.dtype;
#else
    /* FIXME: This approach that avoids intermediate converts needs extra tmps, because when input is BF16 and output is FP32 we cannot reuse/overwrite the same tmp scratch... */
    libxsmm_datatype out_precision = LIBXSMM_DATATYPE_F32;
    libxsmm_datatype in_precision = LIBXSMM_DATATYPE_F32;

    /* Find input precision of op */
    in_precision = cur_op->le->tmp.dtype;

    /* Find sibling if applicable. If it is an Arg, set output precision to  that precision... */
    if (timestamp == last_timestamp) {
      out_precision = (libxsmm_datatype) LIBXSMM_GETENUM_OUT(i_mateqn_desc->datatype);
      cur_op->tmp.ld = i_mateqn_desc->ldo;
    } else {
      out_precision = cur_op->tmp.dtype;
    }
    /* Adjust the tmp precision in the tree */
    /* printf("Node at timestamp %d has input precision %d and  output precision %d\n", timestamp, libxsmm_typesize(in_precision), libxsmm_typesize(out_precision)); */
#endif
#if 0
    if (((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) && (cur_op->info.b_op.type == LIBXSMM_MELTW_TYPE_BINARY_MATMUL)) ||
        ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) && (cur_op->info.t_op.type == LIBXSMM_MELTW_TYPE_TERNARY_MATMUL))) {
#endif
#if 0
      libxsmm_blasint m = 0, n = 0, k = 0, lda = 0, ldb = 0, ldc = 0, alpha = 1, beta = 0, prefetch = 0, gemm_flags = LIBXSMM_FLAGS;
      m = cur_op->tmp.m;
      n = cur_op->tmp.n;
      if (cur_op->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
        k = cur_op->le->info.arg.n;
        lda = cur_op->le->info.arg.ld;
      } else {
        k = cur_op->le->tmp.n;
        lda = cur_op->le->tmp.ld;
      }
      if (cur_op->ri->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
        ldb = cur_op->ri->info.arg.ld;
      } else {
        ldb = cur_op->ri->tmp.ld;
      }
      ldc = cur_op->tmp.ld;

      libxsmm_generator_matequation_set_input_in_stack_param_struct( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->le,
          temp_reg, 0);
      libxsmm_generator_matequation_set_input_in_stack_param_struct( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->ri,
          temp_reg, 1);
      libxsmm_generator_matequation_set_output_in_stack_param_struct( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op,
          temp_reg, (timestamp == last_timestamp) );
      libxsmm_generator_meqn_getval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_UNARY_BINARY_PARAM_STRUCT_PTR0, LIBXSMM_X86_GP_REG_R8);
      libxsmm_generator_meqn_getval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_UNARY_BINARY_PARAM_STRUCT_PTR1, LIBXSMM_X86_GP_REG_RAX);
      libxsmm_generator_meqn_getval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_UNARY_BINARY_PARAM_STRUCT_PTR2, LIBXSMM_X86_GP_REG_R9);

      xgemm_desc = (libxsmm_gemm_descriptor*) libxsmm_sgemm_descriptor_init(&blob, m, n, k, lda, ldb, ldc, alpha, beta, gemm_flags, libxsmm_get_gemm_prefetch(prefetch));

      libxsmm_generator_gemm_striped_sse_avx_avx2_avx512_kernel( io_generated_code, io_loop_label_tracker, xgemm_desc);
#endif
#if 0
    } else {
#endif
      if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) {
        if (cur_op->info.t_op.type == LIBXSMM_MELTW_TYPE_TERNARY_NMULADD) {
          int temp_scratch_id = eqn->eqn_root->reg_score;
          unsigned short bin_flags = 0;
          libxsmm_bitfield flags  = cur_op->info.t_op.flags;
          if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_0) > 0) {
            bin_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0;
          } else if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0) > 0) {
            bin_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
          } else if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_0) > 0) {
            bin_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0;
          }
          if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_2) > 0) {
            bin_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1;
          } else if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2) > 0) {
            bin_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1;
          } else if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2) > 0) {
            bin_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1;
          }

          /* Set up first a binary MUL: left x right2 -> temp_scratch */
          libxsmm_generator_matequation_set_input_in_stack_param_struct_aarch64( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->le, temp_reg, 0);
          libxsmm_generator_matequation_set_input_in_stack_param_struct_aarch64( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->r2, temp_reg, 1);
          libxsmm_generator_meqn_getaddr_stack_tmp_i_aarch64( io_generated_code, temp_scratch_id * i_micro_kernel_config->tmp_size, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg);
          libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR12, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
          meltw_desc = libxsmm_meltw_descriptor_init2(&blob, in_precision, cur_op->r2->tmp.dtype, LIBXSMM_DATATYPE_UNSUPPORTED, cur_op->info.t_op.dtype, out_precision,
              cur_op->tmp.m, cur_op->tmp.n, cur_op->le->tmp.ld, cur_op->tmp.m, cur_op->r2->tmp.ld, 0, (unsigned short)bin_flags, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_OPERATION_BINARY);
          libxsmm_generator_mateltwise_aarch64_init_micro_kernel_config_fullvector( io_generated_code, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
          libxsmm_generator_unary_binary_aarch64_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );

          bin_flags = 0;
          if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_1) > 0) {
            bin_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0;
          } else if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1) > 0) {
            bin_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
          } else if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1) > 0) {
            bin_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0;
          }
          /* Set up then a binary SUB: right - temp_scratch -> output */
          libxsmm_generator_matequation_set_input_in_stack_param_struct_aarch64( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->ri, temp_reg, 0);
          libxsmm_generator_meqn_getaddr_stack_tmp_i_aarch64( io_generated_code, temp_scratch_id * i_micro_kernel_config->tmp_size, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg);
          libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR8, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
          libxsmm_generator_matequation_set_output_in_stack_param_struct_aarch64( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op, temp_reg, (timestamp == last_timestamp) );
          meltw_desc = libxsmm_meltw_descriptor_init2(&blob, cur_op->ri->tmp.dtype, out_precision, LIBXSMM_DATATYPE_UNSUPPORTED, cur_op->info.t_op.dtype, out_precision,
              cur_op->tmp.m, cur_op->tmp.n, cur_op->ri->tmp.ld, cur_op->tmp.ld, cur_op->tmp.m, 0, (unsigned short)bin_flags, LIBXSMM_MELTW_TYPE_BINARY_SUB, LIBXSMM_MELTW_OPERATION_BINARY);
          libxsmm_generator_mateltwise_aarch64_init_micro_kernel_config_fullvector( io_generated_code, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
          libxsmm_generator_unary_binary_aarch64_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
        } else if (cur_op->info.t_op.type == LIBXSMM_MELTW_TYPE_TERNARY_MULADD) {
          int temp_scratch_id = eqn->eqn_root->reg_score;
          unsigned short bin_flags = 0;
          libxsmm_bitfield flags  = cur_op->info.t_op.flags;
          if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_0) > 0) {
            bin_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0;
          } else if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0) > 0) {
            bin_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
          } else if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_0) > 0) {
            bin_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0;
          }
          if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_1) > 0) {
            bin_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1;
          } else if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1) > 0) {
            bin_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1;
          } else if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1) > 0) {
            bin_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1;
          }

          /* Set up first a binary MUL: left x right -> temp_scratch */
          libxsmm_generator_matequation_set_input_in_stack_param_struct_aarch64( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->le, temp_reg, 0);
          libxsmm_generator_matequation_set_input_in_stack_param_struct_aarch64( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->ri, temp_reg, 1);
          libxsmm_generator_meqn_getaddr_stack_tmp_i_aarch64( io_generated_code, temp_scratch_id * i_micro_kernel_config->tmp_size, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg);
          libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR12, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
          meltw_desc = libxsmm_meltw_descriptor_init2(&blob, in_precision, cur_op->ri->tmp.dtype, LIBXSMM_DATATYPE_UNSUPPORTED, cur_op->info.t_op.dtype, out_precision,
              cur_op->tmp.m, cur_op->tmp.n, cur_op->le->tmp.ld, cur_op->tmp.m, cur_op->ri->tmp.ld, 0, (unsigned short)bin_flags, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_OPERATION_BINARY);
          libxsmm_generator_mateltwise_aarch64_init_micro_kernel_config_fullvector( io_generated_code, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
          libxsmm_generator_unary_binary_aarch64_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );

          bin_flags = 0;
          if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_2) > 0) {
            bin_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0;
          } else if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2) > 0) {
            bin_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
          } else if ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2) > 0) {
            bin_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0;
          }
          /* Set up then a binary ADD: right2 + temp_scratch -> output */
          libxsmm_generator_matequation_set_input_in_stack_param_struct_aarch64( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->r2, temp_reg, 0);
          libxsmm_generator_meqn_getaddr_stack_tmp_i_aarch64( io_generated_code, temp_scratch_id * i_micro_kernel_config->tmp_size, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg);
          libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR8, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
          libxsmm_generator_matequation_set_output_in_stack_param_struct_aarch64( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op, temp_reg, (timestamp == last_timestamp) );
          meltw_desc = libxsmm_meltw_descriptor_init2(&blob, cur_op->r2->tmp.dtype, out_precision, LIBXSMM_DATATYPE_UNSUPPORTED, cur_op->info.t_op.dtype, out_precision,
              cur_op->tmp.m, cur_op->tmp.n, cur_op->r2->tmp.ld, cur_op->tmp.ld, cur_op->tmp.m, 0, (unsigned short)bin_flags, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_OPERATION_BINARY);
          libxsmm_generator_mateltwise_aarch64_init_micro_kernel_config_fullvector( io_generated_code, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
          libxsmm_generator_unary_binary_aarch64_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
        } else {
          libxsmm_generator_matequation_set_input_in_stack_param_struct_aarch64( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->le,
              temp_reg, 0);
          libxsmm_generator_matequation_set_input_in_stack_param_struct_aarch64( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->ri,
              temp_reg, 1);
          libxsmm_generator_matequation_set_input_in_stack_param_struct_aarch64( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->r2,
              temp_reg, 2);
          libxsmm_generator_matequation_set_output_in_stack_param_struct_aarch64( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op,
              temp_reg, (timestamp == last_timestamp) );
          libxsmm_generator_matequation_create_ternary_descriptor( &blob, cur_op, &meltw_desc, in_precision, out_precision);
        }
      } else if ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) && (cur_op->info.b_op.type == LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD)) {
        int temp_scratch_id = eqn->eqn_root->reg_score;
        /* Set up binary MUL */
        libxsmm_generator_matequation_set_input_in_stack_param_struct_aarch64( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->le, temp_reg, 0);
        libxsmm_generator_matequation_set_input_in_stack_param_struct_aarch64( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->ri, temp_reg, 1);
        libxsmm_generator_meqn_getaddr_stack_tmp_i_aarch64( io_generated_code, temp_scratch_id * i_micro_kernel_config->tmp_size, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg);
        libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR12, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
        meltw_desc = libxsmm_meltw_descriptor_init2(&blob, in_precision, cur_op->ri->tmp.dtype, LIBXSMM_DATATYPE_UNSUPPORTED, cur_op->info.b_op.dtype, out_precision,
              cur_op->le->tmp.m, cur_op->le->tmp.n, cur_op->le->tmp.ld, cur_op->le->tmp.m, cur_op->ri->tmp.ld, 0, (unsigned short)cur_op->info.b_op.flags, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_OPERATION_BINARY);
        libxsmm_generator_mateltwise_aarch64_init_micro_kernel_config_fullvector( io_generated_code, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
        libxsmm_generator_unary_binary_aarch64_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
        /* Set up reduce cols kernel */
        libxsmm_generator_meqn_getaddr_stack_tmp_i_aarch64( io_generated_code, temp_scratch_id * i_micro_kernel_config->tmp_size, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg);
        libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR4, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
        libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR8, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
        meltw_desc = libxsmm_meltw_descriptor_init2(&blob, in_precision, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, cur_op->info.b_op.dtype, out_precision,
              cur_op->le->tmp.m, cur_op->le->tmp.n, cur_op->le->tmp.m, cur_op->le->tmp.m, 0, 0, (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_OPERATION_UNARY);
        libxsmm_generator_mateltwise_aarch64_init_micro_kernel_config_fullvector( io_generated_code, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
        libxsmm_generator_reduce_cols_aarch64_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
        /* Set up reduce rows kernel */
        libxsmm_generator_meqn_getaddr_stack_tmp_i_aarch64( io_generated_code, temp_scratch_id * i_micro_kernel_config->tmp_size, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg);
        libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR4, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
        if (timestamp == last_timestamp) {
          libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 16, temp_reg );
        } else {
          libxsmm_generator_meqn_getaddr_stack_tmp_i_aarch64( io_generated_code, cur_op->tmp.id * i_micro_kernel_config->tmp_size, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg);
        }
        libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR8, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
        meltw_desc = libxsmm_meltw_descriptor_init2(&blob, in_precision, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, cur_op->info.b_op.dtype, out_precision,
              cur_op->le->tmp.m, 1, cur_op->le->tmp.m, cur_op->tmp.ld, 0, 0, (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_OPERATION_UNARY);
        libxsmm_generator_mateltwise_aarch64_init_micro_kernel_config_fullvector( io_generated_code, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
        libxsmm_generator_reduce_rows_aarch64_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
      } else if ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (cur_op->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_TO_SCALAR_OP_ADD)) {
        int temp_scratch_id = eqn->eqn_root->reg_score;
        /* Set up reduce cols kernel */
        libxsmm_generator_matequation_set_input_in_stack_param_struct_aarch64( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->le, temp_reg, 0);
        libxsmm_generator_meqn_getaddr_stack_tmp_i_aarch64( io_generated_code, temp_scratch_id * i_micro_kernel_config->tmp_size, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg);
        libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR8, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
        meltw_desc = libxsmm_meltw_descriptor_init2(&blob, in_precision, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, cur_op->info.u_op.dtype, out_precision,
            cur_op->le->tmp.m, cur_op->le->tmp.n, cur_op->le->tmp.ld, cur_op->le->tmp.m, 0, 0, (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_OPERATION_UNARY);
        libxsmm_generator_mateltwise_aarch64_init_micro_kernel_config_fullvector( io_generated_code, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
        libxsmm_generator_reduce_cols_aarch64_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
        /* Set up reduce rows kernel */
        libxsmm_generator_meqn_getaddr_stack_tmp_i_aarch64( io_generated_code, temp_scratch_id * i_micro_kernel_config->tmp_size, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg);
        libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR4, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
        if (timestamp == last_timestamp) {
          libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 16, temp_reg );
        } else {
          libxsmm_generator_meqn_getaddr_stack_tmp_i_aarch64( io_generated_code, cur_op->tmp.id * i_micro_kernel_config->tmp_size, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg);
        }
        libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR8, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
        meltw_desc = libxsmm_meltw_descriptor_init2(&blob, in_precision, LIBXSMM_DATATYPE_UNSUPPORTED, LIBXSMM_DATATYPE_UNSUPPORTED, cur_op->info.u_op.dtype, out_precision,
              cur_op->le->tmp.m, 1, cur_op->le->tmp.m, cur_op->tmp.ld, 0, 0, (unsigned short)LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_OPERATION_UNARY);
        libxsmm_generator_mateltwise_aarch64_init_micro_kernel_config_fullvector( io_generated_code, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
        libxsmm_generator_reduce_rows_aarch64_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
      } else if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
        /* Prepare struct param */
        libxsmm_generator_matequation_set_input_in_stack_param_struct_aarch64( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->le,
            temp_reg, 0);
        libxsmm_generator_matequation_set_output_in_stack_param_struct_aarch64( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op,
            temp_reg, (timestamp == last_timestamp) );

        /* If need be, set n for reduce cols_idx which is a fusion result */
        if (libxsmm_meqn_is_unary_opcode_reduce_cols_idx_kernel(cur_op->info.u_op.type) > 0) {
          libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, temp_reg, (long long)cur_op->le->tmp.n);
          libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR7, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
          libxsmm_generator_meqn_getaddr_stack_var_aarch64(  io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR7, temp_reg );
          libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR6, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
        }

        /* If need, be set properly the offset param from scratch...  */
        if ((eqn->eqn_root->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (eqn->eqn_root->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_UNZIP) && (timestamp == last_timestamp)) {
          libxsmm_generator_meqn_getval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_CONST_9, temp_reg );
          libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR9, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
        }

        /* If DUMP operator set secondary output in stack param struct */
        if ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (cur_op->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_DUMP)) {
          libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, temp_reg );
          libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, temp_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, cur_op->info.u_op.op_arg_pos*32, temp_reg );
          libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR9, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
        }

        /* Prepare descriptor */
        libxsmm_generator_matequation_create_unary_descriptor( &blob, cur_op, &meltw_desc, in_precision, out_precision);
      } else if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) {
        libxsmm_generator_matequation_set_input_in_stack_param_struct_aarch64( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->le,
            temp_reg, 0);
        libxsmm_generator_matequation_set_input_in_stack_param_struct_aarch64( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->ri,
            temp_reg, 1);
        libxsmm_generator_matequation_set_output_in_stack_param_struct_aarch64( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op,
            temp_reg, (timestamp == last_timestamp) );
        libxsmm_generator_matequation_create_binary_descriptor( &blob, cur_op, &meltw_desc, in_precision, out_precision);
      } else {
        /* This should not happen */
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
        return;
      }
      /* Configure the unary-binary microkernel */
      libxsmm_generator_mateltwise_aarch64_init_micro_kernel_config_fullvector( io_generated_code, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
      /* Call proper JITer */
#if 0
      if ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (libxsmm_meqn_is_unary_opcode_reduce_kernel(meltw_desc->param) > 0)) {
        if ((meltw_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS) > 0) {
          libxsmm_generator_reduce_rows_avx512_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
        } else if ((meltw_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS) > 0) {
          libxsmm_generator_reduce_cols_avx512_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
        } else {
          /* This should not happen */
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
          return;
        }
      } else if ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (libxsmm_meqn_is_unary_opcode_transform_kernel(meltw_desc->param) > 0)) {
        libxsmm_generator_transform_x86_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
      } else {
#endif
#if 0
      }
#endif
#if 0
    }
#endif
    /* Call proper JITer */
    if ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (libxsmm_meqn_is_unary_opcode_reduce_kernel(meltw_desc->param) > 0)) {
      if ((meltw_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS) > 0) {
        libxsmm_generator_reduce_rows_aarch64_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
      } else if ((meltw_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS) > 0) {
        libxsmm_generator_reduce_cols_aarch64_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
      } else {
        /* This should not happen */
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
        return;
      }
    } else if ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (libxsmm_meqn_is_unary_opcode_reduce_cols_idx_kernel(meltw_desc->param) > 0)) {
        libxsmm_generator_reduce_cols_index_aarch64_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
    } else if ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (libxsmm_meqn_is_unary_opcode_transform_kernel(meltw_desc->param) > 0)) {
      libxsmm_generator_transform_aarch64_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
    } else if ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && ((cur_op->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_GATHER) || (cur_op->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_SCATTER) )) {
      libxsmm_generator_gather_scatter_aarch64_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
    } else if ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) ||
               ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) && (cur_op->info.b_op.type == LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD)) ||
               ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (cur_op->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_TO_SCALAR_OP_ADD))) {
      /* JITing already taken care of, do nothing... */
    } else {
      libxsmm_generator_unary_binary_aarch64_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
    }
  }
}
