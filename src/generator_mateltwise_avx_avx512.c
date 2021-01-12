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

#include "generator_mateltwise_avx_avx512.h"
#include "generator_mateltwise_transform_avx_avx512.h"
#include "generator_mateltwise_relu_avx_avx512.h"
#include "generator_mateltwise_dropout_avx_avx512.h"
#include "generator_mateltwise_unary_avx_avx512.h"
#include "generator_mateltwise_reduce_avx_avx512.h"
#include "generator_mateltwise_scale_avx_avx512.h"
#include "generator_mateltwise_copy_avx_avx512.h"
#include "generator_mateltwise_cvtfp32bf16_act_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
int libxsmm_generator_meltw_get_rbp_relative_offset( libxsmm_meltw_stack_var stack_var ) {
  /* The stack at exit of setup looks like this:
   *
   *      Return address                            <-- RBP+8
   *      Entry/saved RBP                           <-- RBP
   *      Input ptr                                 <-- RBP-8
   *      Output ptr                                <-- RBP-16
   *      Mask ptr                                  <-- RBP-24
   *      Scratch ptr in stack (to be filled)       <-- RBP-32
   *      Placeholder for stack var                 <-- RBP-40
   *      Placeholder for stack var                 <-- RBP-48
   *      Placeholder for stack var                 <-- RBP-56
   *      Placeholder for stack var                 <-- RBP-64
   *      Placeholder for stack var                 <-- RBP-72
   *      Placeholder for stack var                 <-- RBP-80
   *
   * * */

  switch ( stack_var ) {
    case LIBXSMM_MELTW_STACK_VAR_INP_PTR:
      return -8;
    case LIBXSMM_MELTW_STACK_VAR_OUT_PTR:
      return -16;
    case LIBXSMM_MELTW_STACK_VAR_MASK_PTR:
      return -24;
    case LIBXSMM_MELTW_STACK_VAR_SCRATCH_PTR:
      return -32;
    default:
      return 0;
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_meltw_getval_stack_var( libxsmm_generated_code*              io_generated_code,
                                                libxsmm_meltw_stack_var            stack_var,
                                                unsigned int                        i_gp_reg ) {
  int offset = libxsmm_generator_meltw_get_rbp_relative_offset(stack_var);
  /* make sure we requested a legal stack var */
  if (offset == 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, offset, i_gp_reg, 0 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_meltw_setval_stack_var( libxsmm_generated_code*              io_generated_code,
                                                libxsmm_meltw_stack_var             stack_var,
                                                unsigned int                        i_gp_reg ) {
  int offset = libxsmm_generator_meltw_get_rbp_relative_offset(stack_var);
  /* make sure we requested to set  a legal stack var */
  if (offset >= 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, offset, i_gp_reg, 1 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_meltw_setup_stack_frame( libxsmm_generated_code*            io_generated_code,
                                              const libxsmm_meltw_descriptor*      i_mateltwise_desc,
                                              libxsmm_mateltwise_gp_reg_mapping*   i_gp_reg_mapping,
                                              libxsmm_mateltwise_kernel_config*    i_micro_kernel_config) {
  unsigned int temp_reg                 = LIBXSMM_X86_GP_REG_R10;
  unsigned int skip_pushpops_callee_gp_reg  = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_REDUCE_COLS_IDX) ||
                                          (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_OPREDUCE_VECS_IDX) ||
                                          (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_COPY) ||
                                          (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) ||
                                          (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_REDUCE) ||
                                          (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TRANSFORM) ||
                                          ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_SCALE) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_SCALE_ROWS_BCASTVAL_ACCUMULATE) > 0)) ||
                                          ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_CVTFP32BF16) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_CVT_VNNI_FORMAT) > 0) )) ? 1 : 0;

  /* TODO: Determine  if we want to save stuff to stack */
  unsigned int save_args_to_stack = 0;
  unsigned int allocate_scratch = 0;
  unsigned int use_aux_stack_vars = 0;
  unsigned int use_stack_vars = ((save_args_to_stack > 0) || (allocate_scratch > 0) || (use_aux_stack_vars > 0)) ? 1 : 0;

  i_micro_kernel_config->skip_pushpops_callee_gp_reg = skip_pushpops_callee_gp_reg;
  i_micro_kernel_config->use_stack_vars              = use_stack_vars;

  if (use_stack_vars > 0) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBP );
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_RBP);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, 80 );
  }

  /* Exemplary usage of how to store args to stack if need be  */
  if (save_args_to_stack > 0) {
    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_COPY) {
      if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_COPY_ZERO) == 0) {
        libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
            i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, temp_reg, 0 );
        libxsmm_generator_meltw_setval_stack_var( io_generated_code, LIBXSMM_MELTW_STACK_VAR_INP_PTR, temp_reg );
      }
      libxsmm_x86_instruction_alu_mem( io_generated_code, i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_X86_GP_REG_UNDEF, 0, 8, temp_reg, 0 );
      libxsmm_generator_meltw_setval_stack_var( io_generated_code, LIBXSMM_MELTW_STACK_VAR_OUT_PTR, temp_reg );
    }
  }

  /* The stack now looks like this:
   *
   *      Return address                            <-- RBP+8
   *      Entry/saved RBP                           <-- RBP
   *      Input ptr (to be filled)                  <-- RBP-8
   *      Output ptr (to be filled)                 <-- RBP-16
   *      Mask ptr (to be filled)                   <-- RBP-24
   *      Scratch ptr in stack (to be filled)       <-- RBP-32
   *      Placeholder for stack var                 <-- RBP-40
   *      Placeholder for stack var                 <-- RBP-48
   *      Placeholder for stack var                 <-- RBP-56
   *      Placeholder for stack var                 <-- RBP-64
   *      Placeholder for stack var                 <-- RBP-72
   *      Placeholder for stack var                 <-- RBP-80
   *
   * * */

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

  /* The stack at exit of setup looks like this:
   *
   *      Return address                            <-- RBP+8
   *      Entry/saved RBP                           <-- RBP
   *      Input ptr (to be filled)                  <-- RBP-8
   *      Output ptr (to be filled)                 <-- RBP-16
   *      Mask ptr (to be filled)                   <-- RBP-24
   *      Scratch ptr in stack (to be filled)       <-- RBP-32
   *      Placeholder for stack var                 <-- RBP-40
   *      Placeholder for stack var                 <-- RBP-48
   *      Placeholder for stack var                 <-- RBP-56
   *      Placeholder for stack var                 <-- RBP-64
   *      Placeholder for stack var                 <-- RBP-72
   *      Placeholder for stack var                 <-- RBP-80
   *      [ Potentianl  pad for 64b align ]
   *      Scratch, 64b aligned                      <-- (RBP-32) contains this address
   *      Callee-saved registers                    <-- RSP
   *
   * * */
}

LIBXSMM_API_INTERN
void libxsmm_generator_meltw_destroy_stack_frame( libxsmm_generated_code*            io_generated_code,
    const libxsmm_meltw_descriptor*     i_mateltwise_desc,
    const libxsmm_mateltwise_kernel_config*  i_micro_kernel_config ) {

  LIBXSMM_UNUSED(i_mateltwise_desc);
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
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_header_m_loop( libxsmm_generated_code*                io_generated_code,
                                              libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                              const libxsmm_mateltwise_kernel_config*   i_kernel_config,
                                              const unsigned int                        i_gp_reg_m_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_mov_instruction, i_gp_reg_m_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_add_instruction, i_gp_reg_m_loop, 1);
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_footer_m_loop( libxsmm_generated_code*                    io_generated_code,
                                              libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                              const libxsmm_mateltwise_kernel_config*       i_kernel_config,
                                              const unsigned int                            i_gp_reg_m_loop,
                                              const unsigned int                            i_m ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_cmp_instruction, i_gp_reg_m_loop, i_m );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_header_n_loop( libxsmm_generated_code*                io_generated_code,
                                              libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                              const libxsmm_mateltwise_kernel_config*   i_kernel_config,
                                              const unsigned int                        i_gp_reg_n_loop ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_add_instruction, i_gp_reg_n_loop, 1);
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_footer_n_loop( libxsmm_generated_code*                    io_generated_code,
                                              libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                              const libxsmm_mateltwise_kernel_config*       i_kernel_config,
                                              const unsigned int                            i_gp_reg_n_loop,
                                              const unsigned int                            i_n ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_cmp_instruction, i_gp_reg_n_loop, i_n );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_header_n_dyn_loop( libxsmm_generated_code*                io_generated_code,
                                              libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                              const libxsmm_mateltwise_kernel_config*   i_kernel_config,
                                              const unsigned int                        i_gp_reg_n_loop,
                                              int                                       skip_init ) {
  if (skip_init == 0) {
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_mov_instruction, i_gp_reg_n_loop, 0);
  }
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_footer_n_dyn_loop( libxsmm_generated_code*                    io_generated_code,
                                              libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                              const libxsmm_mateltwise_kernel_config*       i_kernel_config,
                                              const unsigned int                            i_gp_reg_n_loop,
                                              const unsigned int                            i_gp_reg_n_bound ) {
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_kernel_config->alu_add_instruction, i_gp_reg_n_loop, 1);
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_kernel_config->alu_cmp_instruction, i_gp_reg_n_bound, i_gp_reg_n_loop);
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, i_kernel_config->alu_jmp_instruction, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_initialize_avx512_mask( libxsmm_generated_code*            io_generated_code,
    const unsigned int                       i_gp_reg_tmp,
    const unsigned int                       i_mask_reg,
    const unsigned int                       i_mask_count,
    const unsigned int                       i_precision) {

  unsigned long long l_mask = 0;

  if ( i_precision == LIBXSMM_DATATYPE_F64 || i_precision == LIBXSMM_DATATYPE_I64 ) {
    l_mask = 0xff;
  } else if ( i_precision == LIBXSMM_DATATYPE_F32 || i_precision == LIBXSMM_DATATYPE_I32 ) {
    l_mask = 0xffff;
  } else if ( i_precision == LIBXSMM_DATATYPE_F16 || i_precision == LIBXSMM_DATATYPE_BF16 || i_precision == LIBXSMM_DATATYPE_I16 ) {
    l_mask = 0xffffffff;
  } else if ( i_precision == LIBXSMM_GEMM_PRECISION_I8 ) {
    l_mask = 0xffffffffffffffff;
  }
  /* shift right by "inverse" remainder */
  l_mask = l_mask >> i_mask_count;

  /* move mask to GP register */
  libxsmm_x86_instruction_alu_imm( io_generated_code,
      LIBXSMM_X86_INSTR_MOVQ,
      i_gp_reg_tmp,
      l_mask );

  if ( io_generated_code->arch >= LIBXSMM_X86_AVX512_CORE  ) {
    if ( i_precision == LIBXSMM_DATATYPE_F64 || i_precision == LIBXSMM_DATATYPE_I64 ) {
      libxsmm_x86_instruction_mask_move( io_generated_code,
          LIBXSMM_X86_INSTR_KMOVB_GPR_LD,
          i_gp_reg_tmp,
          i_mask_reg );
    } else if ( i_precision == LIBXSMM_DATATYPE_F32 || i_precision == LIBXSMM_DATATYPE_I32 ) {
      libxsmm_x86_instruction_mask_move( io_generated_code,
          LIBXSMM_X86_INSTR_KMOVW_GPR_LD,
          i_gp_reg_tmp,
          i_mask_reg );
    } else if ( i_precision == LIBXSMM_DATATYPE_F16 || i_precision == LIBXSMM_DATATYPE_BF16 || i_precision == LIBXSMM_DATATYPE_I16 ) {
      libxsmm_x86_instruction_mask_move( io_generated_code,
          LIBXSMM_X86_INSTR_KMOVD_GPR_LD,
          i_gp_reg_tmp,
          i_mask_reg );
    } else if ( i_precision == LIBXSMM_GEMM_PRECISION_I8 ) {
      libxsmm_x86_instruction_mask_move( io_generated_code,
          LIBXSMM_X86_INSTR_KMOVQ_GPR_LD,
          i_gp_reg_tmp,
          i_mask_reg );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  } else {
    /* shouldn't happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( libxsmm_generated_code*         io_generated_code,
    libxsmm_mateltwise_kernel_config*    io_micro_kernel_config,
    const unsigned int              i_arch,
    const libxsmm_meltw_descriptor* i_mateltwise_desc) {
  memset(io_micro_kernel_config, 0, sizeof(*io_micro_kernel_config)); /* avoid warning "maybe used uninitialized" */
  if ( i_arch >= LIBXSMM_X86_AVX512_CORE ) {
    io_micro_kernel_config->instruction_set = io_generated_code->arch;
    io_micro_kernel_config->vector_reg_count = 16;
    /* Configure input specific microkernel options */
    if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->vector_length_in = 8;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVUPD;
    } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->vector_length_in = 16;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVUPS;
    } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->vector_length_in = 32;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->vector_length_in = 16;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->vector_length_in = 64;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU8;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
    /* Configure output specific microkernel options */
    if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 8;
      io_micro_kernel_config->vector_length_out = 8;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVUPD;
    } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 4;
      io_micro_kernel_config->vector_length_out = 16;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVUPS;
    } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 2;
      io_micro_kernel_config->vector_length_out = 32;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 2;
      io_micro_kernel_config->vector_length_out = 16;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 1;
      io_micro_kernel_config->vector_length_out = 64;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVDQU8;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
    io_micro_kernel_config->alu_add_instruction = LIBXSMM_X86_INSTR_ADDQ;
    io_micro_kernel_config->alu_sub_instruction = LIBXSMM_X86_INSTR_SUBQ;
    io_micro_kernel_config->alu_cmp_instruction = LIBXSMM_X86_INSTR_CMPQ;
    io_micro_kernel_config->alu_jmp_instruction = LIBXSMM_X86_INSTR_JL;
    io_micro_kernel_config->alu_mov_instruction = LIBXSMM_X86_INSTR_MOVQ;
    io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
    io_micro_kernel_config->vector_name = 'z';
  } else {
    /* That should not happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_avx_avx512_kernel( libxsmm_generated_code*         io_generated_code,
    const libxsmm_meltw_descriptor* i_mateltwise_desc) {
  libxsmm_mateltwise_kernel_config  l_kernel_config;
  libxsmm_mateltwise_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_loop_label_tracker        l_loop_label_tracker;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define gp register mapping */
  memset(&l_gp_reg_mapping, 0, sizeof(l_gp_reg_mapping));
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RCX;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RDI;
#endif

  /* for now we only support AVX512_CORE and higher */
  if ( io_generated_code->arch < LIBXSMM_X86_AVX512_CORE ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  /* define mateltwise kernel config */
  libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_kernel_config, io_generated_code->arch, i_mateltwise_desc);

  /* open asm */
  libxsmm_x86_instruction_open_stream_mateltwise( io_generated_code, l_gp_reg_mapping.gp_reg_param_struct, 1 );

  /* Stack management for melt kernel */
  libxsmm_generator_meltw_setup_stack_frame( io_generated_code, i_mateltwise_desc, &l_gp_reg_mapping, &l_kernel_config);

  /* Depending on the elementwise function, dispatch the proper code JITer */
  if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_CVTFP32BF16) || (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_CVTFP32BF16_ACT) || (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_ACT_CVTFP32BF16)) {
    if ( (LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && (LIBXSMM_GEMM_PRECISION_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))) {
      if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_CVTFP32BF16) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_CVT_VNNI_FORMAT) > 0) ) {
        libxsmm_generator_cvtfp32bf16_vnni_format_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else {
        libxsmm_generator_cvtfp32bf16_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      }
    } else {
      /* This should not happen  */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_REDUCE) {
    if ( (LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && (LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))) {
      if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_ROWS) > 0) {
        libxsmm_generator_reduce_rows_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_COLS) > 0) {
        libxsmm_generator_reduce_cols_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else {
        /* This should not happen  */
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
        return;
      }
    } else {
      if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && ((LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) || (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype))) &&
           ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_COLS) > 0) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_ELTS) > 0) &&
           ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_ROWS) == 0) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_ELTS_SQUARED) == 0) &&
           ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_OP_ADD) > 0) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_NCNC_FORMAT) > 0) ) {
        libxsmm_generator_reduce_cols_ncnc_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else {
        /* This should not happen  */
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_COPY) {
    libxsmm_generator_copy_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_REDUCE_COLS_IDX) {
    libxsmm_generator_reduce_cols_index_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_OPREDUCE_VECS_IDX) {
    libxsmm_generator_opreduce_vecs_index_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_SCALE) {
    if ( (LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) && (LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))) {
      libxsmm_generator_scale_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
    } else {
      /* This should not happen  */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_RELU) {
    libxsmm_generator_relu_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TRANSFORM ) {
    libxsmm_generator_transform_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_DROPOUT ) {
    libxsmm_generator_dropout_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
    libxsmm_generator_unary_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
  } else  {
    /* This should not happen  */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  /* Stack management formelt kernel */
  libxsmm_generator_meltw_destroy_stack_frame(  io_generated_code, i_mateltwise_desc, &l_kernel_config );

  /* close asm */
  libxsmm_x86_instruction_close_stream_mateltwise( io_generated_code, 1);
}

