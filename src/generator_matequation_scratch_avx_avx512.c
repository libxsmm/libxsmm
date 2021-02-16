/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_matequation_avx_avx512.h"
#include "generator_mateltwise_sse_avx_avx512.h"
#include "generator_mateltwise_unary_binary_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"
#include "generator_common_x86.h"
#include "generator_matequation_scratch_avx_avx512.h"
#include "generator_mateltwise_reduce_avx_avx512.h"

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_create_unary_descriptor(libxsmm_descriptor_blob *blob, libxsmm_matrix_eqn_elem *cur_op, libxsmm_meltw_descriptor **desc, libxsmm_datatype in_precision, libxsmm_datatype out_precision) {
  *desc = libxsmm_meltw_descriptor_init2(blob, in_precision, cur_op->info.u_op.dtype, out_precision, LIBXSMM_DATATYPE_UNSUPPORTED, cur_op->tmp.m, cur_op->tmp.n, cur_op->tmp.ld, cur_op->tmp.ld, 0, 0, (unsigned short)cur_op->info.u_op.flags, cur_op->info.u_op.type, LIBXSMM_MELTW_OPERATION_UNARY);
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_create_binary_descriptor(libxsmm_descriptor_blob *blob, libxsmm_matrix_eqn_elem *cur_op, libxsmm_meltw_descriptor **desc, libxsmm_datatype in_precision, libxsmm_datatype out_precision) {
  *desc = libxsmm_meltw_descriptor_init2(blob, in_precision, cur_op->info.b_op.dtype, out_precision, LIBXSMM_DATATYPE_UNSUPPORTED, cur_op->tmp.m, cur_op->tmp.n, cur_op->tmp.ld, cur_op->tmp.ld, 0, 0, (unsigned short)cur_op->info.b_op.flags, cur_op->info.b_op.type, LIBXSMM_MELTW_OPERATION_BINARY);
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_set_input_in_stack_param_struct( libxsmm_generated_code*   io_generated_code,
    libxsmm_matequation_kernel_config*                  i_micro_kernel_config,
    libxsmm_matequation_gp_reg_mapping*                 i_gp_reg_mapping,
    libxsmm_matrix_eqn_elem*                            cur_node,
    unsigned int                                        temp_reg,
    unsigned int                                        ptr_id ) {

  if (cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
    if (cur_node->info.arg.in_pos >= 0) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          0,
          temp_reg,
          0 );
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          temp_reg,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          cur_node->info.arg.in_pos*8,
          temp_reg,
          0 );
    } else {
      libxsmm_generator_meqn_getaddr_stack_tmp_i( io_generated_code, (-1-cur_node->info.arg.in_pos) * i_micro_kernel_config->tmp_size, temp_reg);
    }
  } else {
    libxsmm_generator_meqn_getaddr_stack_tmp_i( io_generated_code, cur_node->tmp.id * i_micro_kernel_config->tmp_size, temp_reg);
  }
  if (ptr_id == 0) {
    libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_UNARY_BINARY_PARAM_STRUCT_PTR0, temp_reg );
  } else {
    libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_UNARY_BINARY_PARAM_STRUCT_PTR1, temp_reg );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_set_output_in_stack_param_struct(libxsmm_generated_code*   io_generated_code,
    libxsmm_matequation_kernel_config*                  i_micro_kernel_config,
    libxsmm_matequation_gp_reg_mapping*                 i_gp_reg_mapping,
    libxsmm_matrix_eqn_elem*                            cur_node,
    unsigned int                                        temp_reg,
    unsigned int                                        is_last_op ) {
  if (is_last_op > 0) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        8,
        temp_reg,
        0 );
  } else {
    libxsmm_generator_meqn_getaddr_stack_tmp_i( io_generated_code, cur_node->tmp.id * i_micro_kernel_config->tmp_size, temp_reg);
  }
  if ((cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (is_unary_opcode_reduce_kernel(cur_node->info.u_op.type) > 0)) {
    libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_UNARY_BINARY_PARAM_STRUCT_PTR1, temp_reg );
  } else {
    libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_UNARY_BINARY_PARAM_STRUCT_PTR2, temp_reg );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_tmp_stack_scratch_avx_avx512_kernel( libxsmm_generated_code* io_generated_code,
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
  unsigned int temp_reg = LIBXSMM_X86_GP_REG_R8;

  if ( eqn == NULL ) {
    fprintf( stderr, "The requested equation doesn't exist... nothing to JIT,,,\n" );
    return;
  } else {
    last_timestamp = eqn->eqn_root->visit_timestamp;
  }

  i_gp_reg_mapping->gp_reg_mapping_eltwise.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RSI;
  libxsmm_generator_meqn_getaddr_stack_var(  io_generated_code, LIBXSMM_MEQN_STACK_VAR_UNARY_BINARY_PARAM_STRUCT_PTR0, LIBXSMM_X86_GP_REG_RSI );

  /* Iterate over the equation tree based on the optimal traversal order and call the proper JITer */
  for (timestamp = 0; timestamp <= last_timestamp; timestamp++) {
    libxsmm_matrix_eqn_elem *cur_op = find_op_at_timestamp(eqn->eqn_root, timestamp);
#if 0
    libxsmm_datatype out_precision = (timestamp == last_timestamp) ? (libxsmm_datatype) LIBXSMM_GETENUM_OUT(i_mateqn_desc->datatype) : cur_op->tmp.dtype;
    libxsmm_datatype in_precision = cur_op->tmp.dtype;
#else
    /* FIXME: This approach that avoids intermediate converts needs extra tmps, because when input is BF16 and output is FP32 we can't reuse/overwrite the same tmp scratch... */
    libxsmm_datatype out_precision = LIBXSMM_DATATYPE_F32;
    libxsmm_datatype in_precision = LIBXSMM_DATATYPE_F32;

    /* Find input precision of op */
    if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
      if (cur_op->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
        in_precision = cur_op->le->info.arg.dtype;
      } else {
        in_precision = cur_op->le->tmp.dtype;
      }
    } else if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) {
      if (cur_op->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
        in_precision = cur_op->le->info.arg.dtype;
      } else {
        in_precision = cur_op->le->tmp.dtype;
      }
    }
    /* Find sibling if applicable. If it is an Arg, set output precision to  that precision... */
    if (timestamp == last_timestamp) {
      out_precision = LIBXSMM_GETENUM_OUT(i_mateqn_desc->datatype);
    } else {
      out_precision = cur_op->tmp.dtype;
    }
    /* Adjust the tmp precision in the tree  */
    /* printf("Node at timestamp %d has input precision %d and  output precision %d\n", timestamp, libxsmm_typesize(in_precision), libxsmm_typesize(out_precision)); */
#endif

    if ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) && (cur_op->info.b_op.type == LIBXSMM_MELTW_TYPE_BINARY_MATMUL)) {
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

      xgemm_desc = (libxsmm_gemm_descriptor*) libxsmm_sgemm_descriptor_init(&blob, m, n, k, lda, ldb, ldc, alpha, beta, gemm_flags, libxsmm_get_gemm_xprefetch(&prefetch));

      libxsmm_generator_gemm_striped_sse_avx_avx2_avx512_kernel( io_generated_code, io_loop_label_tracker, xgemm_desc);
#endif
    } else {
      if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
        /* Prepare struct param */
        libxsmm_generator_matequation_set_input_in_stack_param_struct( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->le,
            temp_reg, 0);
        libxsmm_generator_matequation_set_output_in_stack_param_struct( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op,
            temp_reg, (timestamp == last_timestamp) );
        /* Prepare descriptor  */
        libxsmm_generator_matequation_create_unary_descriptor( &blob, cur_op, &meltw_desc, in_precision, out_precision);
      } else if (cur_op->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) {
        libxsmm_generator_matequation_set_input_in_stack_param_struct( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->le,
            temp_reg, 0);
        libxsmm_generator_matequation_set_input_in_stack_param_struct( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op->ri,
            temp_reg, 1);
        libxsmm_generator_matequation_set_output_in_stack_param_struct( io_generated_code, i_micro_kernel_config, i_gp_reg_mapping, cur_op,
            temp_reg, (timestamp == last_timestamp) );
        libxsmm_generator_matequation_create_binary_descriptor( &blob, cur_op, &meltw_desc, in_precision, out_precision);
      } else {
        /* This should not happen */
      }
      /* Configure the unary-binary microkernel */
      libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
      /* Call proper JITer */
      if ((cur_op->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (is_unary_opcode_reduce_kernel(meltw_desc->param) > 0)) {
        libxsmm_descriptor_blob   _blob;
        libxsmm_meltw_descriptor  *meltw_reduce_desc = NULL;
        libxsmm_generator_create_reduce_desc_from_unary_desc( &_blob, meltw_desc, &meltw_reduce_desc);
        if ((meltw_reduce_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_ROWS) > 0) {
          libxsmm_generator_reduce_rows_avx512_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_reduce_desc );
        } else if ((meltw_reduce_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_COLS) > 0) {
          libxsmm_generator_reduce_cols_avx512_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_reduce_desc );
        } else {
          /* This should not happen  */
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
          return;
        }
      } else {
        libxsmm_generator_unary_binary_avx512_microkernel( io_generated_code, io_loop_label_tracker, &i_gp_reg_mapping->gp_reg_mapping_eltwise, &i_micro_kernel_config->meltw_kernel_config, meltw_desc );
      }
    }
  }
}

