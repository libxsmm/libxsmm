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


#include "generator_mateltwise_aarch64.h"
#include "generator_mateltwise_transform_common.h"
#include "generator_mateltwise_unary_binary_aarch64.h"
#include "libxsmm_matrixeqn.h"
#include "generator_aarch64_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_meltw_setup_stack_frame_aarch64( libxsmm_generated_code*            io_generated_code,
                                              const libxsmm_meltw_descriptor*      i_mateltwise_desc,
                                              libxsmm_mateltwise_gp_reg_mapping*   i_gp_reg_mapping,
                                              libxsmm_mateltwise_kernel_config*    i_micro_kernel_config) {
  #if 0
  unsigned int temp_reg                 = LIBXSMM_X86_GP_REG_R10;
  unsigned int skip_pushpops_callee_gp_reg  = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_REDUCE_COLS_IDX) ||
                                          (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_OPREDUCE_VECS_IDX) ||
                                          (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) ||
                                          (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) ||
                                          (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_REDUCE) ) ? 1 : 0;

  /* TODO: Determine if we want to save stuff to stack */
  unsigned int save_args_to_stack = 0;
  unsigned int allocate_scratch = 0;
  unsigned int use_aux_stack_vars = ((io_generated_code->arch < LIBXSMM_X86_AVX512) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) &&
      ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV) )) ? 1 : 0;
  unsigned int use_stack_vars = ((save_args_to_stack > 0) || (allocate_scratch > 0) || (use_aux_stack_vars > 0)) ? 1 : 0;

  LIBXSMM_UNUSED(i_gp_reg_mapping);

  i_micro_kernel_config->skip_pushpops_callee_gp_reg = skip_pushpops_callee_gp_reg;
  i_micro_kernel_config->use_stack_vars              = use_stack_vars;

  if (use_stack_vars > 0) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBP );
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_RBP);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, 184 );
  }

  if ((io_generated_code->arch < LIBXSMM_X86_AVX512) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY)) {
    if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV) ) {
      i_micro_kernel_config->rbp_offs_thres = libxsmm_generator_meltw_get_rbp_relative_offset(LIBXSMM_MELTW_STACK_VAR_CONST_0);
      i_micro_kernel_config->rbp_offs_signmask = libxsmm_generator_meltw_get_rbp_relative_offset(LIBXSMM_MELTW_STACK_VAR_CONST_1);
      i_micro_kernel_config->rbp_offs_absmask = libxsmm_generator_meltw_get_rbp_relative_offset(LIBXSMM_MELTW_STACK_VAR_CONST_2);
      i_micro_kernel_config->rbp_offs_scale = libxsmm_generator_meltw_get_rbp_relative_offset(LIBXSMM_MELTW_STACK_VAR_CONST_3);
      i_micro_kernel_config->rbp_offs_shifter = libxsmm_generator_meltw_get_rbp_relative_offset(LIBXSMM_MELTW_STACK_VAR_CONST_4);
      i_micro_kernel_config->rbp_offs_half = libxsmm_generator_meltw_get_rbp_relative_offset(LIBXSMM_MELTW_STACK_VAR_CONST_5);
    }
  }

  /* Exemplary usage of how to store args to stack if need be  */
  if (save_args_to_stack > 0) {
  }

  if (allocate_scratch > 0) {
    /* TODO: Scratch size is kernel-dependent  */
    unsigned int scratch_size = 1024;

    /* make scratch size multiple of 64b */
    scratch_size = (scratch_size % 64 == 0) ? scratch_size : ((scratch_size + 63)/64) * 64;

    /* Now align RSP to 64 byte boundary  */
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code, i_micro_kernel_config->alu_mov_instruction, temp_reg, 0xFFFFFFFFFFFFFFC0 );
    libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_ANDQ, temp_reg, LIBXSMM_X86_GP_REG_RSP);

    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, scratch_size );
    libxsmm_generator_meltw_setval_stack_var( io_generated_code, LIBXSMM_MELTW_STACK_VAR_SCRATCH_PTR, LIBXSMM_X86_GP_REG_RSP );
  }

  /* Now push to RSP the callee-save registers  */
  if (skip_pushpops_callee_gp_reg == 0) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBX );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R13 );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R14 );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R15 );
  }
  #endif
}

LIBXSMM_API_INTERN
void libxsmm_generator_meltw_destroy_stack_frame_aarch64( libxsmm_generated_code*            io_generated_code,
    const libxsmm_meltw_descriptor*     i_mateltwise_desc,
    const libxsmm_mateltwise_kernel_config*  i_micro_kernel_config ) {

  LIBXSMM_UNUSED(i_mateltwise_desc);
  #if 0
  if (i_micro_kernel_config->skip_pushpops_callee_gp_reg == 0) {
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R15 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R14 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R13 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBX );
  }

  if (i_micro_kernel_config->use_stack_vars > 0) {
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_RSP);
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBP );
  }
  #endif
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_aarch64_kernel( libxsmm_generated_code*         io_generated_code,
                                                         const libxsmm_meltw_descriptor* i_mateltwise_desc ) {
  libxsmm_mateltwise_kernel_config  l_kernel_config;
  libxsmm_mateltwise_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_loop_label_tracker        l_loop_label_tracker;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define gp register mapping */
  memset(&l_gp_reg_mapping, 0, sizeof(l_gp_reg_mapping));
  libxsmm_reset_aarch64_gp_reg_mapping( &l_gp_reg_mapping );

  // /* open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xe0f );

  /* being BLAS aligned, for empty kermls, do nothing */
  if ( (i_mateltwise_desc->m > 0) && ((i_mateltwise_desc->n > 0) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) ) ) {
    /* Stack management for melt kernel */
    libxsmm_generator_meltw_setup_stack_frame_aarch64( io_generated_code, i_mateltwise_desc, &l_gp_reg_mapping, &l_kernel_config);

    libxsmm_generator_unary_binary_aarch64_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );

    /* Stack management formelt kernel */
    libxsmm_generator_meltw_destroy_stack_frame_aarch64(  io_generated_code, i_mateltwise_desc, &l_kernel_config );
  }

  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xe0f );
}

