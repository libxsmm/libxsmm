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

#include "generator_mateltwise_sse_avx_avx512.h"
#include "generator_mateltwise_transform_common.h"
#include "generator_mateltwise_unary_binary_avx_avx512.h"
#include "generator_mateltwise_reduce_avx_avx512.h"
#include "generator_mateltwise_misc_avx_avx512.h"
#include "libxsmm_matrixeqn.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
int libxsmm_generator_meltw_get_rbp_relative_offset( libxsmm_meltw_stack_var stack_var ) {
  switch ( stack_var ) {
    case LIBXSMM_MELTW_STACK_VAR_INP0_PTR0:
      return -8;
    case LIBXSMM_MELTW_STACK_VAR_INP0_PTR1:
      return -16;
    case LIBXSMM_MELTW_STACK_VAR_INP0_PTR2:
      return -24;
    case LIBXSMM_MELTW_STACK_VAR_INP1_PTR0:
      return -32;
    case LIBXSMM_MELTW_STACK_VAR_INP1_PTR1:
      return -40;
    case LIBXSMM_MELTW_STACK_VAR_INP1_PTR2:
      return -48;
    case LIBXSMM_MELTW_STACK_VAR_INP2_PTR0:
      return -56;
    case LIBXSMM_MELTW_STACK_VAR_INP2_PTR1:
      return -64;
    case LIBXSMM_MELTW_STACK_VAR_INP2_PTR2:
      return -72;
    case LIBXSMM_MELTW_STACK_VAR_OUT_PTR0:
      return -80;
    case LIBXSMM_MELTW_STACK_VAR_OUT_PTR1:
      return -88;
    case LIBXSMM_MELTW_STACK_VAR_OUT_PTR2:
      return -96;
    case LIBXSMM_MELTW_STACK_VAR_SCRATCH_PTR:
      return -104;
    case LIBXSMM_MELTW_STACK_VAR_CONST_0:
      return -112;
    case LIBXSMM_MELTW_STACK_VAR_CONST_1:
      return -120;
    case LIBXSMM_MELTW_STACK_VAR_CONST_2:
      return -128;
    case LIBXSMM_MELTW_STACK_VAR_CONST_3:
      return -136;
    case LIBXSMM_MELTW_STACK_VAR_CONST_4:
      return -144;
    case LIBXSMM_MELTW_STACK_VAR_CONST_5:
      return -152;
    case LIBXSMM_MELTW_STACK_VAR_CONST_6:
      return -160;
    case LIBXSMM_MELTW_STACK_VAR_CONST_7:
      return -168;
    case LIBXSMM_MELTW_STACK_VAR_CONST_8:
      return -176;
    case LIBXSMM_MELTW_STACK_VAR_CONST_9:
      return -184;
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
void libxsmm_generator_mateltwise_initialize_avx_mask( libxsmm_generated_code*            io_generated_code,
    const unsigned int                       i_mask_reg,
    const unsigned int                       i_mask_count) {
  unsigned int mask_array[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  unsigned int i;
  for (i = 0; i < i_mask_count; i++) {
    mask_array[i] = 0xFFFFFFFF;
  }
  libxsmm_x86_instruction_full_vec_load_of_constants ( io_generated_code, (const unsigned char *) mask_array, "mask_array", 'y', i_mask_reg );
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

  if ( io_generated_code->arch >= LIBXSMM_X86_AVX512  ) {
    if ( i_precision == LIBXSMM_DATATYPE_F64 || i_precision == LIBXSMM_DATATYPE_I64 ) {
      libxsmm_x86_instruction_mask_move( io_generated_code,
          (io_generated_code->arch >= LIBXSMM_X86_AVX512_CORE) ? LIBXSMM_X86_INSTR_KMOVB_GPR_LD : LIBXSMM_X86_INSTR_KMOVW_GPR_LD,
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
void libxsmm_generator_mateltwise_update_micro_kernel_config_vectorlength( libxsmm_generated_code*           io_generated_code,
                                                                           libxsmm_mateltwise_kernel_config* io_micro_kernel_config,
                                                                           const libxsmm_meltw_descriptor*   i_mateltwise_desc) {
  if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512) && (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT) ) {
    io_micro_kernel_config->instruction_set = io_generated_code->arch;
    io_micro_kernel_config->vector_reg_count = 16;
    /* Configure input specific microkernel options */
    if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->vector_length_in = 8;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVUPD;
    } else if ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) || (LIBXSMM_DATATYPE_I32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) ) {
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->vector_length_in = 16;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVUPS;
    } else if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) || (LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ))) {
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
    } else if ( (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) || (LIBXSMM_DATATYPE_I32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) ) {
      io_micro_kernel_config->datatype_size_out = 4;
      io_micro_kernel_config->vector_length_out = 16;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVUPS;
    } else if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype )) || (LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))) {
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
  } else if ( (io_generated_code->arch >= LIBXSMM_X86_AVX) && (io_generated_code->arch < LIBXSMM_X86_AVX512) ) {
    io_micro_kernel_config->instruction_set = io_generated_code->arch;
    io_micro_kernel_config->vector_reg_count = 16;
    /* Configure input specific microkernel options */
    if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->vector_length_in = 4;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVUPD;
    } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->vector_length_in = 8;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVUPS;
#if 0
    } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->vector_length_in = 16;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->vector_length_in = 16;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->vector_length_in = 32;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU8;
#endif
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
    /* Configure output specific microkernel options */
    if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 8;
      io_micro_kernel_config->vector_length_out = 4;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVUPD;
    } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 4;
      io_micro_kernel_config->vector_length_out = 8;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVUPS;
#if 0
    } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 2;
      io_micro_kernel_config->vector_length_out = 16;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 2;
      io_micro_kernel_config->vector_length_out = 16;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 1;
      io_micro_kernel_config->vector_length_out = 32;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVDQU8;
#endif
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
    io_micro_kernel_config->vector_name = 'y';
  } else if ( (io_generated_code->arch >= LIBXSMM_X86_GENERIC) && (io_generated_code->arch < LIBXSMM_X86_AVX) ) {
    io_micro_kernel_config->instruction_set = io_generated_code->arch;
    io_micro_kernel_config->vector_reg_count = 16;
    /* Configure input specific microkernel options */
    if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->vector_length_in = 2;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_MOVUPD;
    } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->vector_length_in = 4;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_MOVUPS;
#if 0
    } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->vector_length_in = 8;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->vector_length_in = 8;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->vector_length_in = 16;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU8;
#endif
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
    /* Configure output specific microkernel options */
    if ( LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 8;
      io_micro_kernel_config->vector_length_out = 2;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_MOVUPD;
    } else if ( LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 4;
      io_micro_kernel_config->vector_length_out = 4;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_MOVUPS;
#if 0
    } else if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 2;
      io_micro_kernel_config->vector_length_out = 8;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( LIBXSMM_DATATYPE_F16 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 2;
      io_micro_kernel_config->vector_length_out = 8;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ) ) {
      io_micro_kernel_config->datatype_size_out = 1;
      io_micro_kernel_config->vector_length_out = 16;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_VMOVDQU8;
#endif
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
    io_micro_kernel_config->alu_add_instruction = LIBXSMM_X86_INSTR_ADDQ;
    io_micro_kernel_config->alu_sub_instruction = LIBXSMM_X86_INSTR_SUBQ;
    io_micro_kernel_config->alu_cmp_instruction = LIBXSMM_X86_INSTR_CMPQ;
    io_micro_kernel_config->alu_jmp_instruction = LIBXSMM_X86_INSTR_JL;
    io_micro_kernel_config->alu_mov_instruction = LIBXSMM_X86_INSTR_MOVQ;
    io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_XORPD;
    io_micro_kernel_config->vector_name = 'x';

  } else {
     /* That should not happen */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( libxsmm_generated_code*           io_generated_code,
                                                                       libxsmm_mateltwise_kernel_config* io_micro_kernel_config,
                                                                       const libxsmm_meltw_descriptor*   i_mateltwise_desc) {
  memset(io_micro_kernel_config, 0, sizeof(*io_micro_kernel_config)); /* avoid warning "maybe used uninitialized" */
  libxsmm_generator_mateltwise_update_micro_kernel_config_vectorlength( io_generated_code, io_micro_kernel_config, i_mateltwise_desc);
}

LIBXSMM_API_INTERN
void libxsmm_generator_create_reduce_desc_from_unary_desc(libxsmm_descriptor_blob *blob, const libxsmm_meltw_descriptor *in_desc, libxsmm_meltw_descriptor **out_desc) {

  unsigned short reduce_flags = 0;

  if ((in_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD) ||
      (in_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD) ||
      (in_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD_NCNC_FORMAT) ||
      (in_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD)) {
    reduce_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_REDUCE_OP_ADD;
  } else if (in_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX) {
    reduce_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_REDUCE_OP_MAX;
  } else if (in_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MUL) {
    reduce_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_REDUCE_OP_MUL;
  }

  if ((in_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD) ||
      (in_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX) ||
      (in_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MUL) ||
      (in_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD_NCNC_FORMAT) ||
      (in_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD)) {
    reduce_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_REDUCE_ELTS;
  }

  if ((in_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD) ||
      (in_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD)) {
    reduce_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_REDUCE_ELTS_SQUARED;
  }

  if (in_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD_NCNC_FORMAT) {
    reduce_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_REDUCE_NCNC_FORMAT;
  }

  if ((in_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS) > 0) {
    reduce_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_REDUCE_COLS;
  } else {
    reduce_flags |= (unsigned short) LIBXSMM_MELTW_FLAG_REDUCE_ROWS;
  }

  *out_desc = libxsmm_meltw_descriptor_init(blob, (libxsmm_datatype)LIBXSMM_GETENUM_INP(in_desc->datatype), (libxsmm_datatype)LIBXSMM_GETENUM_OUT(in_desc->datatype),
      in_desc->m, in_desc->n, in_desc->ldi, in_desc->ldo, (unsigned short)reduce_flags, 0, LIBXSMM_MELTW_OPERATION_REDUCE);
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_sse_avx_avx512_kernel( libxsmm_generated_code*         io_generated_code,
                                                         const libxsmm_meltw_descriptor* i_mateltwise_desc ) {
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
  if ( (io_generated_code->arch < LIBXSMM_X86_AVX512) &&
       ( !((LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) &&
          (LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))) &&
         !((LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( i_mateltwise_desc->datatype )) &&
          (LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_OUT( i_mateltwise_desc->datatype ))) ) ) {
    /* This should not happen  */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  /* define mateltwise kernel config */
  libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_kernel_config, i_mateltwise_desc);

  /* open asm */
  libxsmm_x86_instruction_open_stream_mateltwise( io_generated_code, l_gp_reg_mapping.gp_reg_param_struct, 1 );

  /* being BLAS aligned, for empty kermls, do nothing */
  if ( (i_mateltwise_desc->m > 0) && ((i_mateltwise_desc->n > 0) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) ) ) {
    /* Stack management for melt kernel */
    libxsmm_generator_meltw_setup_stack_frame( io_generated_code, i_mateltwise_desc, &l_gp_reg_mapping, &l_kernel_config);

    /* Depending on the elementwise function, dispatch the proper code JITer */
    if ( (io_generated_code->arch >= LIBXSMM_X86_AVX2) && (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT) ) {
      if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_REDUCE_COLS_IDX) {
        libxsmm_generator_reduce_cols_index_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_OPREDUCE_VECS_IDX) {
        libxsmm_generator_opreduce_vecs_index_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
        if (is_unary_opcode_reduce_kernel(i_mateltwise_desc->param) > 0) {
          libxsmm_descriptor_blob   blob;
          libxsmm_meltw_descriptor  *meltw_reduce_desc = NULL;
          libxsmm_generator_create_reduce_desc_from_unary_desc( &blob, i_mateltwise_desc, &meltw_reduce_desc);
          if ((meltw_reduce_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_ROWS) > 0) {
            libxsmm_generator_reduce_rows_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, meltw_reduce_desc );
          } else if (((meltw_reduce_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_COLS) > 0) && ((meltw_reduce_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_NCNC_FORMAT) == 0)) {
            libxsmm_generator_reduce_cols_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, meltw_reduce_desc );
          } else if (((meltw_reduce_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_COLS) > 0) && ((meltw_reduce_desc->flags & LIBXSMM_MELTW_FLAG_REDUCE_NCNC_FORMAT) > 0)) {
            libxsmm_generator_reduce_cols_ncnc_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, meltw_reduce_desc );
          } else {
            /* This should not happen  */
            LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
            return;
          }
        } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) {
          libxsmm_generator_replicate_col_var_avx_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
        } else if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI)     ||
                    (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT)    ||
                    (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI_TO_VNNIT)    ||
                    (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNIT)    ||
                    (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI_PAD)    ) {
          libxsmm_generator_transform_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
        } else {
          libxsmm_generator_unary_binary_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
        }
      } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY ) {
        libxsmm_generator_unary_binary_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY ) {
        libxsmm_generator_unary_binary_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else  {
        /* This should not happen  */
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
        return;
      }
    } else {
      /* This should not happen  */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
      return;
    }

    /* Stack management formelt kernel */
    libxsmm_generator_meltw_destroy_stack_frame(  io_generated_code, i_mateltwise_desc, &l_kernel_config );
  }

  /* close asm */
  libxsmm_x86_instruction_close_stream_mateltwise( io_generated_code, 1);
}

