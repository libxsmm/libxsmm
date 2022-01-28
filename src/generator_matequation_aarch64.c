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
/* TODO: Move common functionality in different file */
#include "generator_matequation_avx_avx512.h"
#include "generator_common_aarch64.h"
#include "generator_aarch64_instructions.h"
#include "generator_matequation_aarch64.h"
#include "generator_matequation_scratch_aarch64.h"
#include "generator_mateltwise_aarch64.h"
#include "generator_mateltwise_unary_binary_aarch64.h"
#include "generator_common.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_aarch64_init_micro_kernel_config( libxsmm_generated_code*         io_generated_code,
    libxsmm_matequation_kernel_config*    io_micro_kernel_config) {
  memset(io_micro_kernel_config, 0, sizeof(*io_micro_kernel_config)); /* avoid warning "maybe used uninitialized" */
  io_micro_kernel_config->instruction_set = io_generated_code->arch;
  io_micro_kernel_config->alu_add_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
  io_micro_kernel_config->alu_sub_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
  io_micro_kernel_config->alu_cmp_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
  io_micro_kernel_config->alu_jmp_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
  io_micro_kernel_config->alu_mov_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
  io_micro_kernel_config->vxor_instruction = LIBXSMM_AARCH64_INSTR_UNDEF;
}

LIBXSMM_API_INTERN
int libxsmm_generator_mateqn_get_fp_relative_offset( libxsmm_meqn_stack_var stack_var ) {
  /* The stack at exit of setup looks like this:
   *
   *      Return address                            <-- FP+8
   *      Entry/saved FP                            <-- FP
   *      Param_struct_ptr15                        <-- FP-8
   *      Param_struct_ptr14                        <-- FP-16
   *      Param_struct_ptr13                        <-- FP-24
   *      Param_struct_ptr12                        <-- FP-32
   *      Param_struct_ptr11                        <-- FP-40
   *      Param_struct_ptr10                        <-- FP-48
   *      Param_struct_ptr9                         <-- FP-56
   *      Param_struct_ptr8                         <-- FP-64
   *      Param_struct_ptr7                         <-- FP-72
   *      Param_struct_ptr6                         <-- FP-80
   *      Param_struct_ptr5                         <-- FP-88
   *      Param_struct_ptr4                         <-- FP-96
   *      Param_struct_ptr3                         <-- FP-104
   *      Param_struct_ptr2                         <-- FP-112
   *      Param_struct_ptr1                         <-- FP-120
   *      Param_struct_ptr0                         <-- FP-128
   *      Scratch ptr in stack (to be filled)       <-- FP-136
   *      Address scratch ptrin stack (to be filled)<-- FP-144
   *      Saved equation output ptr                 <-- FP-152
   *      Const_0                                   <-- FP-160
   *      ...
   *      Const_9                                   <-- FP-232
   *
   * * */

  switch ( stack_var ) {
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR0:
      return -256;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR1:
      return -248;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR2:
      return -240;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR3:
      return -232;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR4:
      return -224;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR5:
      return -216;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR6:
      return -208;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR7:
      return -200;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR8:
      return -192;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR9:
      return -184;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR10:
      return -176;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR11:
      return -168;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR12:
      return -160;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR13:
      return -152;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR14:
      return -144;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR15:
      return -136;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR16:
      return -128;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR17:
      return -120;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR18:
      return -112;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR19:
      return -104;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR20:
      return -96;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR21:
      return -88;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR22:
      return -80;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR23:
      return -72;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR24:
      return -64;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR25:
      return -56;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR26:
      return -48;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR27:
      return -40;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR28:
      return -32;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR29:
      return -24;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR30:
      return -16;
    case LIBXSMM_MEQN_STACK_VAR_PARAM_STRUCT_PTR31:
      return -8;
    case LIBXSMM_MEQN_STACK_VAR_SCRATCH_PTR:
      return -264;
    case LIBXSMM_MEQN_STACK_VAR_ADDR_SCRATCH_PTR:
      return -272;
    case LIBXSMM_MEQN_STACK_VAR_OUT_PTR:
      return -280;
    case LIBXSMM_MEQN_STACK_VAR_CONST_0:
      return -288;
    case LIBXSMM_MEQN_STACK_VAR_CONST_1:
      return -296;
    case LIBXSMM_MEQN_STACK_VAR_CONST_2:
      return -304;
    case LIBXSMM_MEQN_STACK_VAR_CONST_3:
      return -312;
    case LIBXSMM_MEQN_STACK_VAR_CONST_4:
      return -320;
    case LIBXSMM_MEQN_STACK_VAR_CONST_5:
      return -328;
    case LIBXSMM_MEQN_STACK_VAR_CONST_6:
      return -336;
    case LIBXSMM_MEQN_STACK_VAR_CONST_7:
      return -344;
    case LIBXSMM_MEQN_STACK_VAR_CONST_8:
      return -352;
    case LIBXSMM_MEQN_STACK_VAR_CONST_9:
      return -360;
    default:
      return 0;
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_meqn_getaddr_stack_var_aarch64( libxsmm_generated_code*              io_generated_code,
                                                libxsmm_meqn_stack_var              stack_var,
                                                unsigned int                        i_gp_reg ) {
  int offset = libxsmm_generator_mateqn_get_fp_relative_offset(stack_var);
  /* make sure we requested a legal stack var */
  if (offset == 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ORR_SR, LIBXSMM_AARCH64_GP_REG_X29, LIBXSMM_AARCH64_GP_REG_X29, i_gp_reg, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I, i_gp_reg, i_gp_reg, -offset, 0 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_meqn_getval_stack_var_aarch64( libxsmm_generated_code*               io_generated_code,
                                                libxsmm_meqn_stack_var              stack_var,
                                                unsigned int                        i_gp_reg ) {
  int offset = libxsmm_generator_mateqn_get_fp_relative_offset(stack_var);
  /* make sure we requested a legal stack var */
  if (offset == 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I, LIBXSMM_AARCH64_GP_REG_X29, i_gp_reg, -offset, 0 );
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg );
}

LIBXSMM_API_INTERN
void libxsmm_generator_meqn_getaddr_stack_tmp_i_aarch64( libxsmm_generated_code*            io_generated_code,
                                                unsigned int                        i_tmp_offset_i,
                                                unsigned int                        i_gp_reg_scratch,
                                                unsigned int                        i_gp_reg ) {
  libxsmm_generator_meqn_getval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_SCRATCH_PTR, i_gp_reg );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg, i_gp_reg_scratch, i_gp_reg, i_tmp_offset_i );
}

LIBXSMM_API_INTERN
void libxsmm_generator_meqn_getaddr_stack_tmpaddr_i_aarch64( libxsmm_generated_code*            io_generated_code,
                                                unsigned int                        i_tmp_offset_i,
                                                unsigned int                        i_gp_reg_scratch,
                                                unsigned int                        i_gp_reg ) {
  libxsmm_generator_meqn_getval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_ADDR_SCRATCH_PTR, i_gp_reg );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, i_gp_reg, i_gp_reg_scratch, i_gp_reg, i_tmp_offset_i );
}

LIBXSMM_API_INTERN
void libxsmm_generator_meqn_setval_stack_var_aarch64( libxsmm_generated_code*               io_generated_code,
                                                libxsmm_meqn_stack_var              stack_var,
                                                unsigned int                        i_aux_reg,
                                                unsigned int                        i_gp_reg ) {
  int offset = libxsmm_generator_mateqn_get_fp_relative_offset(stack_var);
  /* make sure we requested to set  a legal stack var */
  if (offset >= 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I, LIBXSMM_AARCH64_GP_REG_X29, i_aux_reg, -offset, 0 );
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, i_aux_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, i_gp_reg );
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_setup_stack_frame_aarch64( libxsmm_generated_code*   io_generated_code,
                                              const libxsmm_meqn_descriptor*                      i_mateqn_desc,
                                              libxsmm_matequation_gp_reg_mapping*                 i_gp_reg_mapping,
                                              libxsmm_matequation_kernel_config*                  i_micro_kernel_config,
                                              libxsmm_matrix_eqn*                                 i_eqn,
                                              unsigned int                                        i_strategy ) {

  unsigned int temp_reg = i_gp_reg_mapping->temp_reg;
  unsigned int allocate_scratch = 1;

  LIBXSMM_UNUSED(i_mateqn_desc);
#if 0
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ORR_SR, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_X29, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
#endif
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_X29, 0, 0 );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 360, 0 );

  /* The stack at exit of setup looks like this:
   *
   *      Return address                            <-- FP+8
   *      Entry/saved FP                            <-- FP
   *      Param_struct_ptr15                        <-- FP-8
   *      Param_struct_ptr14                        <-- FP-16
   *      Param_struct_ptr13                        <-- FP-24
   *      Param_struct_ptr12                        <-- FP-32
   *      Param_struct_ptr11                        <-- FP-40
   *      Param_struct_ptr10                        <-- FP-48
   *      Param_struct_ptr9                         <-- FP-56
   *      Param_struct_ptr8                         <-- FP-64
   *      Param_struct_ptr7                         <-- FP-72
   *      Param_struct_ptr6                         <-- FP-80
   *      Param_struct_ptr5                         <-- FP-88
   *      Param_struct_ptr4                         <-- FP-96
   *      Param_struct_ptr3                         <-- FP-104
   *      Param_struct_ptr2                         <-- FP-112
   *      Param_struct_ptr1                         <-- FP-120
   *      Param_struct_ptr0                         <-- FP-128
   *      Scratch ptr in stack (to be filled)       <-- FP-136
   *      Address scratch ptrin stack (to be filled)<-- FP-144
   *      Saved equation output ptr                 <-- FP-152
   *      Const_0                                   <-- FP-160
   *      ...
   *      Const_9                                   <-- FP-232
   *
   * * */

  if (allocate_scratch > 0) {
    unsigned int scratch_size = 0;
#if 0
    unsigned int addr_scratch_size = 0;
#endif
    /* Now align RSP to 64 byte boundary  */
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, temp_reg, 0xFFFFFFFFFFFFFFC0 );
    /* reg-reg instruction */
#if 0
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_AND_SR, LIBXSMM_AARCH64_GP_REG_XSP, temp_reg, LIBXSMM_AARCH64_GP_REG_XSP, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
#endif
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, i_gp_reg_mapping->gp_reg_scratch_0, 0, 0 );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_AND_SR, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg, i_gp_reg_mapping->gp_reg_scratch_0, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, i_gp_reg_mapping->gp_reg_scratch_0, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );


    if (i_strategy == JIT_STRATEGY_USING_TMP_SCRATCH_BLOCKS) {
      /*TODO: Now we allocate tmps with dsize float */
      /* Extra tmp for ternary accommodation */
      libxsmm_blasint n_tmp = i_eqn->eqn_root->reg_score + 1;
      libxsmm_blasint tmp_size = i_eqn->eqn_root->max_tmp_size * 4;
      tmp_size = (tmp_size % 64 == 0) ? tmp_size : ((tmp_size + 63)/64) * 64;
      scratch_size = tmp_size * n_tmp;
      i_micro_kernel_config->tmp_size = tmp_size;
      /* make scratch size multiple of 64b */
      scratch_size = (scratch_size % 64 == 0) ? scratch_size : ((scratch_size + 63)/64) * 64;
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, temp_reg, 0, 0 );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, temp_reg, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg, scratch_size );
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, temp_reg, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
      libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_SCRATCH_PTR, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
      if (libxsmm_verbosity < 0) {
        fprintf( stderr, "JITing Matrix Equation with STACK-ALLOCATED TEMPS (n_tmp = %d , stack_scratch_size = %.5g KB)\n", n_tmp, (1.0*scratch_size)/1024.0 );
      }
    } else {
      fprintf( stderr, "On Aarch64 for now JITing equations only using stack vars\n");
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
#if 0
    else if (i_strategy == JIT_STRATEGY_USING_TMP_REGISTER_BLOCKS){
      libxsmm_blasint n_args = i_eqn->eqn_root->n_args;
      i_micro_kernel_config->n_args = n_args;
      addr_scratch_size = n_args * 8;
      /* make addr scratch size multiple of 64b */
      addr_scratch_size = (addr_scratch_size % 64 == 0) ? addr_scratch_size : ((addr_scratch_size + 63)/64) * 64;
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, addr_scratch_size );
      libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_ADDR_SCRATCH_PTR, LIBXSMM_X86_GP_REG_RSP );
      if (libxsmm_verbosity < 0) {
        fprintf( stderr, "JITing Matrix Equation with REGISTER-BLOCK TEMPS (n_args = %d , addr_scratch_size = %.5g KB)\n", n_args, (1.0*addr_scratch_size)/1024.0 );
      }
    } else if (i_strategy == JIT_STRATEGY_HYBRID) {
      libxsmm_blasint n_tmp = i_eqn->eqn_root->reg_score;
      libxsmm_blasint tmp_size = i_eqn->eqn_root->max_tmp_size * 4;
      libxsmm_blasint n_args = i_eqn->eqn_root->n_args;
      tmp_size = (tmp_size % 64 == 0) ? tmp_size : ((tmp_size + 63)/64) * 64;
      i_micro_kernel_config->tmp_size = tmp_size;
      /* make scratch size multiple of 64b */
      scratch_size = tmp_size * n_tmp;
      scratch_size = (scratch_size % 64 == 0) ? scratch_size : ((scratch_size + 63)/64) * 64;
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, scratch_size );
      libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_SCRATCH_PTR, LIBXSMM_X86_GP_REG_RSP );
      /* make addr scratch size multiple of 64b */
      i_micro_kernel_config->n_args = n_args;
      addr_scratch_size = n_args * 8;
      addr_scratch_size = (addr_scratch_size % 64 == 0) ? addr_scratch_size : ((addr_scratch_size + 63)/64) * 64;
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, addr_scratch_size );
      libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_ADDR_SCRATCH_PTR, LIBXSMM_X86_GP_REG_RSP );
      if (libxsmm_verbosity < 0) {
        fprintf( stderr, "JITing Matrix Equation with HYBRID STRATEGY for TEMPS (n_tmp = %d , stack_scratch_size = %.5g KB , addr_scratch_size = %.5g KB)\n", n_tmp, (1.0*scratch_size)/1024.0, (1.0*addr_scratch_size)/1024.0 );
      }
    }
#endif
  }

  /* Store the out ptr in stack  */
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 16, temp_reg );
  libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_OUT_PTR, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );

#if 0
  /* Store the out ptr in stack  */i
  if (i_strategy == JIT_STRATEGY_HYBRID) {
    libxsmm_x86_instruction_alu_mem( io_generated_code,
        i_micro_kernel_config->alu_mov_instruction,
        i_gp_reg_mapping->gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        16,
        temp_reg,
        0 );
    libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_OUT_PTR, temp_reg );

    /* If head of equaiton is unpack_to_blocks, then make sure we store the block ofset in the stack */
    if ((i_eqn->eqn_root->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (i_eqn->eqn_root->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_UNPACK_TO_BLOCKS)) {
      libxsmm_x86_instruction_alu_mem( io_generated_code,
          i_micro_kernel_config->alu_mov_instruction,
          i_gp_reg_mapping->gp_reg_param_struct,
          LIBXSMM_X86_GP_REG_UNDEF, 0,
          24,
          temp_reg,
          0 );
      libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_CONST_9, temp_reg );
    }
  }
#endif
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_destroy_stack_frame_aarch64( libxsmm_generated_code*                   io_generated_code,
                                              libxsmm_matequation_kernel_config*                  i_micro_kernel_config,
                                              libxsmm_matequation_gp_reg_mapping*                 i_gp_reg_mapping,
                                              unsigned int                                        i_strategy  ) {
  LIBXSMM_UNUSED(i_gp_reg_mapping);
  LIBXSMM_UNUSED(i_micro_kernel_config);
  LIBXSMM_UNUSED(i_strategy);


  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_X29, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
#if 0
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ORR_SR, LIBXSMM_AARCH64_GP_REG_X29, LIBXSMM_AARCH64_GP_REG_X29, LIBXSMM_AARCH64_GP_REG_XSP, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
#endif
}

#if 0
LIBXSMM_API_INTERN
libxsmm_matrix_eqn_elem* find_op_at_timestamp_aarch64(libxsmm_matrix_eqn_elem* cur_node, libxsmm_blasint timestamp) {
  libxsmm_matrix_eqn_elem *result = NULL;
  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    result = NULL;
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    if (cur_node->visit_timestamp == timestamp) {
      result = cur_node;
    } else {
      result = find_op_at_timestamp(cur_node->le, timestamp);
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    if (cur_node->visit_timestamp == timestamp) {
      result = cur_node;
    } else {
      result = find_op_at_timestamp(cur_node->le, timestamp);
      if (result == NULL) {
        result = find_op_at_timestamp(cur_node->ri, timestamp);
      }
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
    if (cur_node->visit_timestamp == timestamp) {
      result = cur_node;
    } else {
      result = find_op_at_timestamp(cur_node->le, timestamp);
      if (result == NULL) {
        result = find_op_at_timestamp(cur_node->ri, timestamp);
        if (result == NULL) {
          result = find_op_at_timestamp(cur_node->r2, timestamp);
        }
      }
    }
  }
  return result;
}

LIBXSMM_API_INTERN
int is_eqn_node_breaking_point(libxsmm_matrix_eqn_elem *node) {
  int result = 0;
  if (node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
    if ( node->info.u_op.type  == LIBXSMM_MELTW_TYPE_UNARY_TANH ||
         node->info.u_op.type  == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV ||
         node->info.u_op.type  == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID ||
         node->info.u_op.type  == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV ||
         /*node->info.u_op.type  == LIBXSMM_MELTW_TYPE_UNARY_EXP ||*/
         node->info.u_op.type  == LIBXSMM_MELTW_TYPE_UNARY_GELU ||
         node->info.u_op.type  == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV ||
         node->info.u_op.type  == LIBXSMM_MELTW_TYPE_UNARY_IDENTITY ||
         is_unary_opcode_transform_kernel(node->info.u_op.type) ||
         is_unary_opcode_reduce_kernel(node->info.u_op.type) ) {
      result = 1;
    }
  }
  if (node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) {
    if ( node->info.b_op.type  == LIBXSMM_MELTW_TYPE_BINARY_MATMUL) {
      result = 1;
    }
  }
  if (node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) {
    if ( node->info.t_op.type  == LIBXSMM_MELTW_TYPE_TERNARY_MATMUL) {
      result = 1;
    }
  }
  return result;
}

LIBXSMM_API_INTERN
void enqueue_equation(libxsmm_matrix_eqn *eqn, libxsmm_matrix_eqn **jiting_queue, unsigned int *queue_size) {
  jiting_queue[*queue_size] = eqn;
  *queue_size = *queue_size + 1;
}

LIBXSMM_API_INTERN int is_unary_with_bcast(libxsmm_bitfield flags);
LIBXSMM_API_INTERN int is_unary_with_bcast(libxsmm_bitfield flags) {
  int result = 0;
  if ( ((flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR) > 0) ) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INTERN int is_binary_with_bcast(libxsmm_bitfield flags);
LIBXSMM_API_INTERN int is_binary_with_bcast(libxsmm_bitfield flags) {
  int result = 0;
  if ( ((flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1) > 0) ) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INTERN int is_ternary_with_bcast(libxsmm_bitfield flags);
LIBXSMM_API_INTERN int is_ternary_with_bcast(libxsmm_bitfield flags) {
  int result = 0;
  if ( ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_0) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_0) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_1) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_2) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2) > 0) ) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INTERN int is_unary_bcast_arg_an_inputarg(libxsmm_bitfield flags, libxsmm_matrix_eqn_elem *node);
LIBXSMM_API_INTERN int is_unary_bcast_arg_an_inputarg(libxsmm_bitfield flags, libxsmm_matrix_eqn_elem *node) {
  int result = 1;
  if ( ((flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR) > 0) ) {
    if (node->le->type != LIBXSMM_MATRIX_EQN_NODE_ARG) {
      result = 0;
    }
  }
  return result;
}

LIBXSMM_API_INTERN int is_binary_bcast_arg_an_inputarg(libxsmm_bitfield flags, libxsmm_matrix_eqn_elem *node);
LIBXSMM_API_INTERN int is_binary_bcast_arg_an_inputarg(libxsmm_bitfield flags, libxsmm_matrix_eqn_elem *node) {
  int result = 1;
  if ( ((flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0) > 0) ) {
    if (node->le->type != LIBXSMM_MATRIX_EQN_NODE_ARG) {
      result = 0;
    }
  }

  if ( ((flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1) > 0) ) {
    if (node->ri->type != LIBXSMM_MATRIX_EQN_NODE_ARG) {
      result = 0;
    }
  }
  return result;
}

LIBXSMM_API_INTERN int is_ternary_bcast_arg_an_inputarg(libxsmm_bitfield flags, libxsmm_matrix_eqn_elem *node);
LIBXSMM_API_INTERN int is_ternary_bcast_arg_an_inputarg(libxsmm_bitfield flags, libxsmm_matrix_eqn_elem *node) {
  int result = 1;
  if ( ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_0) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_0) > 0) ) {
    if (node->le->type != LIBXSMM_MATRIX_EQN_NODE_ARG) {
      result = 0;
    }
  }

  if ( ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_1) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1) > 0) ) {
    if (node->ri->type != LIBXSMM_MATRIX_EQN_NODE_ARG) {
      result = 0;
    }
  }

  if ( ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_2) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2) > 0) ) {
    if (node->r2->type != LIBXSMM_MATRIX_EQN_NODE_ARG) {
      result = 0;
    }
  }
  return result;
}
#endif

LIBXSMM_API_INTERN
void libxsmm_generator_decompose_equation_tree_aarch64( libxsmm_matrix_eqn *eqn, libxsmm_matrix_eqn **jiting_queue, unsigned int *queue_size ) {
#if 0
  libxsmm_matrix_eqn_elem *root = eqn->eqn_root;
  unsigned int last_timestamp = eqn->eqn_root->visit_timestamp, timestamp = 0;

  for (timestamp = 0; timestamp <= last_timestamp; timestamp++) {
    libxsmm_matrix_eqn_elem *cur_node = find_op_at_timestamp(root, timestamp);
    if ( (timestamp < last_timestamp) && ((is_eqn_node_breaking_point(cur_node) > 0) || (is_eqn_node_breaking_point(cur_node->up) > 0) ||
                                           ((cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (is_unary_opcode_reduce_to_scalar(cur_node->info.u_op.type) > 0)) ||
                                           ((cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) && (is_binary_opcode_reduce_to_scalar(cur_node->info.b_op.type) > 0)) ||
                                           ((cur_node->up->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (is_unary_with_bcast(cur_node->up->info.u_op.flags) > 0) && (is_unary_bcast_arg_an_inputarg(cur_node->up->info.u_op.flags, cur_node->up) == 0) ) ||
                                           ((cur_node->up->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) && (is_binary_with_bcast(cur_node->up->info.b_op.flags) > 0) && (is_binary_bcast_arg_an_inputarg(cur_node->up->info.b_op.flags, cur_node->up) == 0)) ||
                                           ((cur_node->up->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) && (is_ternary_with_bcast(cur_node->up->info.t_op.flags) > 0) && (is_ternary_bcast_arg_an_inputarg(cur_node->up->info.t_op.flags, cur_node->up) == 0)))) {

      libxsmm_matrix_eqn_elem       *new_arg_node = (libxsmm_matrix_eqn_elem*) malloc( sizeof(libxsmm_matrix_eqn_elem) );
      libxsmm_matrix_eqn            *new_eqn      = (libxsmm_matrix_eqn*) malloc( sizeof(libxsmm_matrix_eqn) );
      union libxsmm_matrix_eqn_info info;
      info.arg.m = cur_node->tmp.m;
      info.arg.n = cur_node->tmp.n;
      info.arg.ld = cur_node->tmp.ld;
      info.arg.in_pos = -(cur_node->tmp.id + 1);
      info.arg.dtype = cur_node->tmp.dtype;
      new_arg_node->le = NULL;
      new_arg_node->ri = NULL;
      new_arg_node->r2 = NULL;
      new_arg_node->up = cur_node->up;
      new_arg_node->type = LIBXSMM_MATRIX_EQN_NODE_ARG;
      new_arg_node->info = info;
      new_arg_node->reg_score = 0;
      new_arg_node->tmp.dtype = cur_node->tmp.dtype;
      new_arg_node->tmp.m = cur_node->tmp.m;
      new_arg_node->tmp.n = cur_node->tmp.n;
      new_arg_node->tmp.ld = cur_node->tmp.ld;

      if (cur_node->up->le == cur_node) {
        cur_node->up->le = new_arg_node;
      } else if (cur_node->up->ri == cur_node)  {
        cur_node->up->ri = new_arg_node;
      } else {
        cur_node->up->r2 = new_arg_node;
      }
      new_eqn->eqn_root = cur_node;
      new_eqn->is_constructed = 1;
      enqueue_equation(new_eqn, jiting_queue, queue_size);
    }
    if (timestamp == last_timestamp) {
      enqueue_equation(eqn, jiting_queue, queue_size);
    }
  }
#else
  enqueue_equation(eqn, jiting_queue, queue_size);
#endif
}

#if 0
LIBXSMM_API_INTERN void are_nodes_pure_f32(libxsmm_matrix_eqn_elem *node, unsigned int *result);
LIBXSMM_API_INTERN void are_nodes_pure_f32(libxsmm_matrix_eqn_elem *node, unsigned int *result) {
  if (node->tmp.dtype != LIBXSMM_DATATYPE_F32) {
    *result = 0;
  }
  if ( node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    return;
  } else if ( node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    are_nodes_pure_f32(node->le, result);
  } else if ( node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    are_nodes_pure_f32(node->le, result);
    are_nodes_pure_f32(node->ri, result);
  } else if ( node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
    are_nodes_pure_f32(node->le, result);
    are_nodes_pure_f32(node->ri, result);
    are_nodes_pure_f32(node->r2, result);
  }
}
#endif

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_aarch64_kernel( libxsmm_generated_code*        io_generated_code,
                                                      const libxsmm_meqn_descriptor* i_mateqn_desc) {
  libxsmm_matequation_gp_reg_mapping  l_gp_reg_mapping;
  libxsmm_matequation_kernel_config   l_kernel_config;
  libxsmm_loop_label_tracker          l_loop_label_tracker;
  unsigned int eqn_idx = i_mateqn_desc->eqn_idx;
  libxsmm_matrix_eqn *eqn = libxsmm_matrix_eqn_get_equation( eqn_idx );
  libxsmm_matrix_eqn **jiting_queue;
  unsigned int queue_size = 0;

  /* TODO: Use number of tree nodes as max size */
  unsigned int max_queue_size = 256;
  /* TODO: Now using only strategy with tmp scratch blocks on aarch64 */
  unsigned int strategy = JIT_STRATEGY_USING_TMP_SCRATCH_BLOCKS;
  unsigned int eqn_tree_id = 0;
  unsigned int all_nodes_f32 = 1;

  if ( eqn == NULL ) {
    fprintf( stderr, "The requested equation doesn't exist... nothing to JIT,,,\n" );
    return;
  }

  /* Check if equation is purely F32 */
  are_nodes_pure_f32(eqn->eqn_root, &all_nodes_f32);
  if ( !((LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_OUT( i_mateqn_desc->datatype )) && (all_nodes_f32 == 1))) {
    /* This should not happen  */
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
    return;
  }

  /* Some basic initialization of the config kernel */
  libxsmm_generator_matequation_aarch64_init_micro_kernel_config(io_generated_code, &l_kernel_config);

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define gp register mapping */
  memset(&l_gp_reg_mapping, 0, sizeof(l_gp_reg_mapping));
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_AARCH64_GP_REG_X0;
  l_gp_reg_mapping.temp_reg    = LIBXSMM_AARCH64_GP_REG_X26;
  l_gp_reg_mapping.gp_reg_scratch_0    = LIBXSMM_AARCH64_GP_REG_X27;

#if 0
  l_gp_reg_mapping.gp_reg_out = LIBXSMM_X86_GP_REG_RAX;
  l_gp_reg_mapping.temp_reg    = LIBXSMM_X86_GP_REG_RBX;
  l_gp_reg_mapping.temp_reg2   = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_m_loop = LIBXSMM_X86_GP_REG_R14;
  l_gp_reg_mapping.gp_reg_n_loop = LIBXSMM_X86_GP_REG_R15;
  l_kernel_config.n_avail_gpr = 8;
  l_kernel_config.gpr_pool[0] = LIBXSMM_X86_GP_REG_RSI; l_kernel_config.gpr_pool[1] = LIBXSMM_X86_GP_REG_RDX; l_kernel_config.gpr_pool[2] = LIBXSMM_X86_GP_REG_R8; l_kernel_config.gpr_pool[3] = LIBXSMM_X86_GP_REG_R9;
  l_kernel_config.gpr_pool[4] = LIBXSMM_X86_GP_REG_R10; l_kernel_config.gpr_pool[5] = LIBXSMM_X86_GP_REG_R11; l_kernel_config.gpr_pool[6] = LIBXSMM_X86_GP_REG_R12; l_kernel_config.gpr_pool[7] = LIBXSMM_X86_GP_REG_R13;

  if ((eqn->eqn_root->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (eqn->eqn_root->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_UNPACK_TO_BLOCKS)) {
    l_kernel_config.n_avail_gpr = l_kernel_config.n_avail_gpr - 1;
    l_gp_reg_mapping.gp_reg_offset = LIBXSMM_X86_GP_REG_R13;
  }
#endif

  jiting_queue = (libxsmm_matrix_eqn**) malloc(max_queue_size * sizeof(libxsmm_matrix_eqn*));
  libxsmm_generator_decompose_equation_tree_aarch64( eqn, jiting_queue, &queue_size );

  /* open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xe0f );
  /* Setup the stack */
  libxsmm_generator_matequation_setup_stack_frame_aarch64( io_generated_code, i_mateqn_desc, &l_gp_reg_mapping, &l_kernel_config, eqn, strategy);

  for (eqn_tree_id = 0; eqn_tree_id < queue_size; eqn_tree_id++) {
    libxsmm_matrix_eqn *cur_eqn = jiting_queue[eqn_tree_id];
    libxsmm_meqn_descriptor copy_mateqn_desc = *i_mateqn_desc;

    /* Determine the output and precision of current equation tree to be JITed */
    if (eqn_tree_id == (queue_size - 1)) {
      libxsmm_generator_meqn_getval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_OUT_PTR, l_gp_reg_mapping.temp_reg );
    } else {
      libxsmm_generator_meqn_getaddr_stack_tmp_i_aarch64( io_generated_code,  cur_eqn->eqn_root->tmp.id * l_kernel_config.tmp_size, l_gp_reg_mapping.gp_reg_scratch_0, l_gp_reg_mapping.temp_reg );
      copy_mateqn_desc.datatype = cur_eqn->eqn_root->tmp.dtype;
    }

    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, l_gp_reg_mapping.gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 16, l_gp_reg_mapping.temp_reg );

#if 0
    if (is_eqn_node_breaking_point(cur_eqn->eqn_root) > 0) {
      /* For these nodes use strategy via scratch  */
      /* Re assign visit_stamps to current equation tree  */
      libxsmm_generator_matequation_assign_timestamps(cur_eqn);
#if 0
      printf("\nJITing tree with scratch %d\n", eqn_tree_id);
      libxsmm_matrix_eqn_trv_dbg_print( cur_eqn->eqn_root, 0);
#endif
      if (eqn_tree_id < queue_size - 1) {
        copy_mateqn_desc.ldo = cur_eqn->eqn_root->tmp.m;
      }
      l_kernel_config.meltw_kernel_config.vector_name = l_kernel_config.vector_name;
      libxsmm_generator_matequation_tmp_stack_scratch_avx_avx512_kernel(io_generated_code, &copy_mateqn_desc, &l_gp_reg_mapping, &l_kernel_config, &l_loop_label_tracker, cur_eqn);
    } else {
      /* For these nodes use strategy via regblocks  */
      /* Re-optimize current tree  */
      if (((cur_eqn->eqn_root->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (is_unary_opcode_reduce_to_scalar(cur_eqn->eqn_root->info.u_op.type) > 0)) ||
          ((cur_eqn->eqn_root->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) && (is_binary_opcode_reduce_to_scalar(cur_eqn->eqn_root->info.b_op.type) > 0))) {
        copy_mateqn_desc.m = cur_eqn->eqn_root->le->tmp.m;
        copy_mateqn_desc.n = cur_eqn->eqn_root->le->tmp.n;
      } else {
        copy_mateqn_desc.m = cur_eqn->eqn_root->tmp.m;
        copy_mateqn_desc.n = cur_eqn->eqn_root->tmp.n;
      }
      if (eqn_tree_id < queue_size - 1) {
        copy_mateqn_desc.ldo = cur_eqn->eqn_root->tmp.m;
      }
      /* If head of equaiton is unpack_to_blocks, then make sure we load the block offset from the stack */
      if ((cur_eqn->eqn_root->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (cur_eqn->eqn_root->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_UNPACK_TO_BLOCKS)) {
        libxsmm_generator_meqn_getval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_CONST_9, l_gp_reg_mapping.gp_reg_offset);
      }

      libxsmm_generator_reoptimize_eqn(cur_eqn);
      memset(&(l_kernel_config.meltw_kernel_config), 0, sizeof(libxsmm_mateltwise_kernel_config));
#if 0
      printf("\nJITing tree with regblocks %d\n", eqn_tree_id);
      libxsmm_matrix_eqn_trv_dbg_print( cur_eqn->eqn_root, 0);
#endif
      l_kernel_config.meltw_kernel_config.vector_name = l_kernel_config.vector_name;
      libxsmm_generator_matequation_tmp_register_block_avx_avx512_kernel(io_generated_code, &copy_mateqn_desc, &l_gp_reg_mapping, &l_kernel_config, &l_loop_label_tracker, cur_eqn);
    }
#else
    /* For now JIT equations via scratch  */
    /* Re assign visit_stamps to current equation tree  */
    libxsmm_generator_matequation_assign_timestamps(cur_eqn);
    if (eqn_tree_id < queue_size - 1) {
      copy_mateqn_desc.ldo = cur_eqn->eqn_root->tmp.m;
    }
    l_kernel_config.meltw_kernel_config.vector_name = l_kernel_config.vector_name;
    libxsmm_generator_matequation_tmp_stack_scratch_aarch64_kernel(io_generated_code, &copy_mateqn_desc, &l_gp_reg_mapping, &l_kernel_config, &l_loop_label_tracker, cur_eqn);
#endif
  }

  /* Destroy stack frame */
  libxsmm_generator_matequation_destroy_stack_frame_aarch64(  io_generated_code,  &l_kernel_config, &l_gp_reg_mapping, strategy);
  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xe0f );

  free(jiting_queue);
}

