/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
*               Friedrich Schiller University Jena - All rights reserved.     *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.), Antonio Noack (FSU Jena)
******************************************************************************/

#include "generator_mateltwise_aarch64.h"
#include "generator_mateltwise_transform_common.h"
#include "generator_mateltwise_unary_binary_aarch64.h"
#include "generator_mateltwise_reduce_aarch64.h"
#include "generator_mateltwise_misc_aarch64.h"
#include "libxsmm_matrixeqn.h"
#include "generator_aarch64_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_aarch64_update_micro_kernel_config_vectorlength( libxsmm_generated_code*   io_generated_code,
                                                                           libxsmm_mateltwise_kernel_config* io_micro_kernel_config,
                                                                           const libxsmm_meltw_descriptor*   i_mateltwise_desc) {
  /* this could be simplified, as can be seen from https://github.com/AntonioNoack/libxsmm/blob/00a6077e6b81556879032f0d9fbf8f84e283dd8d/src/generator_mateltwise_aarch64.c */
  /* we currently keep it as-is, because there may be additional logic/data type depending things in the future */
  if (
    io_generated_code->arch == LIBXSMM_AARCH64_V81 ||
    io_generated_code->arch == LIBXSMM_AARCH64_V82 ||
    io_generated_code->arch == LIBXSMM_AARCH64_APPL_M1 ||
    io_generated_code->arch == LIBXSMM_AARCH64_A64FX
  ) {
    io_micro_kernel_config->instruction_set = io_generated_code->arch;
    io_micro_kernel_config->vector_reg_count = 32;
    /* Configure input specific microkernel options */
    if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
    } else if ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) || (LIBXSMM_DATATYPE_I32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) ) {
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
    } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
    } else if ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
    } else if ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
    } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
    /* Configure output specific microkernel options */
    if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 8;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
    } else if ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) || (LIBXSMM_DATATYPE_I32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ) {
      io_micro_kernel_config->datatype_size_out = 4;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
    } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 2;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
    } else if ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 2;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
    } else if ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 2;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
    } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 1;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_AARCH64_INSTR_ASIMD_LDR_R;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
    if( io_generated_code->arch == LIBXSMM_AARCH64_A64FX ){
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_AARCH64_INSTR_SVE_LDR_Z_I_OFF;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_AARCH64_INSTR_SVE_STR_Z_I_OFF;
    }
    io_micro_kernel_config->alu_add_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
    io_micro_kernel_config->alu_sub_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
    io_micro_kernel_config->alu_cmp_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
    io_micro_kernel_config->alu_jmp_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
    io_micro_kernel_config->alu_mov_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
    io_micro_kernel_config->vxor_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
  } else {
    /* That should not happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_aarch64_init_micro_kernel_config_fullvector( libxsmm_generated_code*           io_generated_code,
                                                                               libxsmm_mateltwise_kernel_config* io_micro_kernel_config,
                                                                               const libxsmm_meltw_descriptor*   i_mateltwise_desc) {
  memset(io_micro_kernel_config, 0, sizeof(*io_micro_kernel_config)); /* avoid warning "maybe used uninitialized" */
  libxsmm_generator_mateltwise_aarch64_update_micro_kernel_config_vectorlength( io_generated_code, io_micro_kernel_config, i_mateltwise_desc );
}

LIBXSMM_API_INTERN
void libxsmm_generator_meltw_setup_stack_frame_aarch64( libxsmm_generated_code*            io_generated_code,
                                              const libxsmm_meltw_descriptor*      i_mateltwise_desc,
                                              libxsmm_mateltwise_gp_reg_mapping*   i_gp_reg_mapping,
                                              libxsmm_mateltwise_kernel_config*    i_micro_kernel_config) {
  LIBXSMM_UNUSED(io_generated_code);
  LIBXSMM_UNUSED(i_mateltwise_desc);
  LIBXSMM_UNUSED(i_gp_reg_mapping);
  LIBXSMM_UNUSED(i_micro_kernel_config);
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
  LIBXSMM_UNUSED(io_generated_code);
  LIBXSMM_UNUSED(i_micro_kernel_config);
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
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_AARCH64_GP_REG_X0;

  /* define mateltwise kernel config */
  libxsmm_generator_mateltwise_aarch64_init_micro_kernel_config_fullvector( io_generated_code, &l_kernel_config, i_mateltwise_desc );

  /* open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xe0f );

  /* being BLAS aligned, for empty kermls, do nothing */
  if ( (i_mateltwise_desc->m > 0) && ((i_mateltwise_desc->n > 0) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX)) ) {
    /* Stack management for melt kernel */
    libxsmm_generator_meltw_setup_stack_frame_aarch64( io_generated_code, i_mateltwise_desc, &l_gp_reg_mapping, &l_kernel_config);

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_OPREDUCE_VECS_IDX) {
      libxsmm_generator_opreduce_vecs_index_aarch64_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
    } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
      if (libxsmm_matrix_eqn_is_unary_opcode_reduce_kernel(i_mateltwise_desc->param) > 0) {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS) > 0) {
          libxsmm_generator_reduce_rows_aarch64_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
        } else if (((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD_NCNC_FORMAT)) {
          libxsmm_generator_reduce_cols_aarch64_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
        } else if (((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS) > 0) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD_NCNC_FORMAT)) {
          libxsmm_generator_reduce_cols_ncnc_aarch64_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
        } else {
          /* This should not happen  */
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_MISSING_REDUCE_FLAGS );
          return;
        }
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX) {
        libxsmm_generator_reduce_cols_index_aarch64_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) {
        libxsmm_generator_replicate_col_var_aarch64_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else  if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI)     ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT)    ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI_TO_VNNIT)    ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNIT)    ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI_PAD) ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2)        ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2)        ||
           (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2)          ) {
        libxsmm_generator_transform_aarch64_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else {
        libxsmm_generator_unary_binary_aarch64_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      }
    } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY ) {
      libxsmm_generator_unary_binary_aarch64_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
    } else  {
      /* This should not happen  */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNKNOWN_OPERATION );
      return;
    }

    /* Stack management formelt kernel */
    libxsmm_generator_meltw_destroy_stack_frame_aarch64(  io_generated_code, i_mateltwise_desc, &l_kernel_config );
  }

  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xe0f );
}