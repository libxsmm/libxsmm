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
#include "generator_matequation_regblocks_aarch64.h"


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

  switch ( (int)stack_var ) {
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

  unsigned int temp_reg = i_gp_reg_mapping->temp_reg3;
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
    unsigned int addr_scratch_size = 0;
    /* Now align RSP to 64 byte boundary */
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, temp_reg, 0xFFFFFFFFFFFFFFC0 );
    /* reg-reg instruction */
#if 0
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_AND_SR, LIBXSMM_AARCH64_GP_REG_XSP, temp_reg, LIBXSMM_AARCH64_GP_REG_XSP, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
#endif
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, i_gp_reg_mapping->gp_reg_scratch_0, 0, 0 );
    libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_AND_SR, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg, i_gp_reg_mapping->gp_reg_scratch_0, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, i_gp_reg_mapping->gp_reg_scratch_0, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );


    if (i_strategy == JIT_STRATEGY_USING_TMP_SCRATCH_BLOCKS) {
      /* TODO: Now we allocate tmps with dsize float */
      /* Extra tmp for ternary accommodation */
      int tree_max_comp_tsize = i_eqn->eqn_root->tree_max_comp_tsize;
      libxsmm_blasint n_tmp = i_eqn->eqn_root->reg_score + 1;
      libxsmm_blasint tmp_size = i_eqn->eqn_root->max_tmp_size * tree_max_comp_tsize;
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
    } else if (i_strategy == JIT_STRATEGY_USING_TMP_REGISTER_BLOCKS){
      libxsmm_blasint n_args = i_eqn->eqn_root->n_args;
      libxsmm_blasint n_max_opargs = i_eqn->eqn_root->visit_timestamp + 1;
      i_micro_kernel_config->n_args = n_args;
      addr_scratch_size = n_args * 8 + n_max_opargs * 32;
      /* make addr scratch size multiple of 64b */
      addr_scratch_size = (addr_scratch_size % 64 == 0) ? addr_scratch_size : ((addr_scratch_size + 63)/64) * 64;
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, temp_reg, 0, 0 );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, temp_reg, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg, addr_scratch_size );
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, temp_reg, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
      libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_ADDR_SCRATCH_PTR, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
      if (libxsmm_verbosity < 0) {
        fprintf( stderr, "JITing Matrix Equation with REGISTER-BLOCK TEMPS (n_args = %d , addr_scratch_size = %.5g KB)\n", n_args, (1.0*addr_scratch_size)/1024.0 );
      }
    } else if (i_strategy == JIT_STRATEGY_HYBRID) {
      int tree_max_comp_tsize = i_eqn->eqn_root->tree_max_comp_tsize;
      libxsmm_blasint n_tmp = i_eqn->eqn_root->reg_score;
      libxsmm_blasint tmp_size = i_eqn->eqn_root->max_tmp_size * tree_max_comp_tsize;
      libxsmm_blasint n_args = i_eqn->eqn_root->n_args;
      libxsmm_blasint n_max_opargs = i_eqn->eqn_root->visit_timestamp + 1;
      tmp_size = (tmp_size % 64 == 0) ? tmp_size : ((tmp_size + 63)/64) * 64;
      i_micro_kernel_config->tmp_size = tmp_size;
      /* make scratch size multiple of 64b */
      scratch_size = tmp_size * n_tmp;
      scratch_size = (scratch_size % 64 == 0) ? scratch_size : ((scratch_size + 63)/64) * 64;
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, temp_reg, 0, 0 );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, temp_reg, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg, scratch_size );
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, temp_reg, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
      libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_SCRATCH_PTR, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
      /* make addr scratch size multiple of 64b */
      i_micro_kernel_config->n_args = n_args;
      addr_scratch_size = n_args * 8 + n_max_opargs * 32;
      addr_scratch_size = (addr_scratch_size % 64 == 0) ? addr_scratch_size : ((addr_scratch_size + 63)/64) * 64;
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, temp_reg, 0, 0 );
      libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB, temp_reg, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg, addr_scratch_size );
      libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, temp_reg, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
      libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_ADDR_SCRATCH_PTR, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
      if (libxsmm_verbosity < 0) {
        fprintf( stderr, "JITing Matrix Equation with HYBRID STRATEGY for TEMPS (n_tmp = %d , stack_scratch_size = %.5g KB , addr_scratch_size = %.5g KB)\n", n_tmp, (1.0*scratch_size)/1024.0, (1.0*addr_scratch_size)/1024.0 );
      }
    } else {
      fprintf( stderr, "Should not happen, not supported matrix equation JITing mode...\n");
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  }

  /* Store the out ptr in stack */
  libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 16, temp_reg );
  libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_OUT_PTR, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );

  if ((i_eqn->eqn_root->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (i_eqn->eqn_root->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_UNZIP)) {
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, i_gp_reg_mapping->gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 24, temp_reg );
    libxsmm_generator_meqn_setval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_CONST_9, i_gp_reg_mapping->gp_reg_scratch_0, temp_reg );
  }
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
}

LIBXSMM_API_INTERN
void libxsmm_generator_decompose_equation_tree_aarch64( libxsmm_matrix_eqn *eqn, libxsmm_matrix_eqn **jiting_queue, unsigned int *queue_size, libxsmm_meqn_fusion_knobs *fusion_knobs) {
  /* For now jus call the same decomposition strategy as on x86 */
  libxsmm_generator_decompose_equation_tree_x86( eqn, jiting_queue, queue_size, fusion_knobs);
}

LIBXSMM_API_INTERN
libxsmm_blasint libxsmm_generator_matequation_aarch64_valid_arch_precision( libxsmm_generated_code*           io_generated_code,
                                                                        libxsmm_matrix_eqn*               i_eqn,
                                                                        const libxsmm_meqn_descriptor*    i_mateqn_desc) {
  libxsmm_blasint is_valid_arch_prec = 1;
  unsigned int has_inp_or_out_fp8 = ((libxsmm_meqn_any_args_dtype(i_eqn, LIBXSMM_DATATYPE_BF8) > 0) || (libxsmm_meqn_any_args_dtype(i_eqn, LIBXSMM_DATATYPE_HF8) > 0) ||
                                     (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_OUT( i_mateqn_desc->datatype )) || (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_OUT( i_mateqn_desc->datatype ))) ? 1 : 0;
  unsigned int has_inp_or_out_fp64= ((libxsmm_meqn_any_args_dtype(i_eqn, LIBXSMM_DATATYPE_F64) > 0) || (LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_OUT( i_mateqn_desc->datatype ))) ? 1 : 0;
  unsigned int has_inp_or_out_bf16= ((libxsmm_meqn_any_args_dtype(i_eqn, LIBXSMM_DATATYPE_BF16) > 0) || (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_OUT( i_mateqn_desc->datatype ))) ? 1 : 0;
  unsigned int all_nodes_fp64 = libxsmm_meqn_all_nodes_dtype(i_eqn, LIBXSMM_DATATYPE_F64);
  unsigned int all_args_fp64 = libxsmm_meqn_all_args_dtype(i_eqn, LIBXSMM_DATATYPE_F64);
  unsigned int all_fp64 = ((all_nodes_fp64 > 0) && (all_args_fp64 > 0) && (LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_OUT( i_mateqn_desc->datatype ))) ? 1 : 0;

  /* Unary not supported for fp64 */
  libxsmm_meltw_unary_type non_fp64_unary[21] = { LIBXSMM_MELTW_TYPE_UNARY_RELU,
                                                  LIBXSMM_MELTW_TYPE_UNARY_RELU_INV,
                                                  LIBXSMM_MELTW_TYPE_UNARY_TANH,
                                                  LIBXSMM_MELTW_TYPE_UNARY_TANH_INV,
                                                  LIBXSMM_MELTW_TYPE_UNARY_SIGMOID,
                                                  LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV,
                                                  LIBXSMM_MELTW_TYPE_UNARY_GELU,
                                                  LIBXSMM_MELTW_TYPE_UNARY_GELU_INV,
                                                  LIBXSMM_MELTW_TYPE_UNARY_EXP,
                                                  LIBXSMM_MELTW_TYPE_UNARY_DROPOUT,
                                                  LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV,
                                                  LIBXSMM_MELTW_TYPE_UNARY_UNZIP,
                                                  LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU,
                                                  LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV,
                                                  LIBXSMM_MELTW_TYPE_UNARY_ELU,
                                                  LIBXSMM_MELTW_TYPE_UNARY_ELU_INV,
                                                  LIBXSMM_MELTW_TYPE_UNARY_STOCHASTIC_ROUND,
                                                  LIBXSMM_MELTW_TYPE_UNARY_QUANT,
                                                  LIBXSMM_MELTW_TYPE_UNARY_DEQUANT,
                                                  LIBXSMM_MELTW_TYPE_UNARY_GATHER,
                                                  LIBXSMM_MELTW_TYPE_UNARY_SCATTER };

  /* Binary not supported for fp64 */
  libxsmm_meltw_binary_type non_fp64_binary[2] = { LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD,
                                                   LIBXSMM_MELTW_TYPE_BINARY_ZIP };
  /* TODO: check for SVE128! */
  if ((libxsmm_meqn_contains_opcode(i_eqn, LIBXSMM_MELTW_TYPE_UNARY_UNZIP, LIBXSMM_MELTW_TYPE_BINARY_ZIP, LIBXSMM_MELTW_TYPE_TERNARY_NONE) > 0) && (io_generated_code->arch != LIBXSMM_AARCH64_NEOV1)) {
    is_valid_arch_prec = 0;
  }
  if ((has_inp_or_out_bf16 > 0) && (io_generated_code->arch != LIBXSMM_AARCH64_NEOV1)) {
    is_valid_arch_prec = 0;
  }
  if (has_inp_or_out_fp8 > 0) {
    is_valid_arch_prec = 0;
  }
  if ((has_inp_or_out_fp64 > 0) && (all_fp64 == 0)) {
    is_valid_arch_prec = 0;
  }
  if (has_inp_or_out_fp64 > 0) {
    unsigned int i = 0;
    for (i = 0; i < 21; i++) {
      if (libxsmm_meqn_contains_opcode(i_eqn, non_fp64_unary[i], LIBXSMM_MELTW_TYPE_BINARY_NONE, LIBXSMM_MELTW_TYPE_TERNARY_NONE) > 0) {
        is_valid_arch_prec = 0;
        break;
      }
    }
    for (i = 0; i < 2; i++) {
      if (libxsmm_meqn_contains_opcode(i_eqn, LIBXSMM_MELTW_TYPE_UNARY_NONE, non_fp64_binary[i], LIBXSMM_MELTW_TYPE_TERNARY_NONE) > 0) {
        is_valid_arch_prec = 0;
        break;
      }
    }
  }
  return is_valid_arch_prec;
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_aarch64_kernel( libxsmm_generated_code*        io_generated_code,
                                                      const libxsmm_meqn_descriptor* i_mateqn_desc) {
  libxsmm_matequation_gp_reg_mapping  l_gp_reg_mapping;
  libxsmm_matequation_kernel_config   l_kernel_config;
  libxsmm_loop_label_tracker          l_loop_label_tracker;
  unsigned int eqn_idx = i_mateqn_desc->eqn_idx;
  libxsmm_matrix_eqn *eqn = libxsmm_meqn_get_equation( eqn_idx );
  libxsmm_matrix_eqn **jiting_queue;
  unsigned int queue_size = 0;

  /* TODO: Use number of tree nodes as max size */
  unsigned int max_queue_size = 256;
  /* TODO: Now using only strategy with tmp scratch blocks on aarch64 */
#if 1
  unsigned int strategy = JIT_STRATEGY_HYBRID;
#else
  unsigned int strategy = JIT_STRATEGY_USING_TMP_SCRATCH_BLOCKS;
#endif
  unsigned int eqn_tree_id = 0;
  unsigned int temp_reg = LIBXSMM_AARCH64_GP_REG_X6;
  libxsmm_meqn_fusion_knobs fusion_knobs;
  memset(&fusion_knobs, 0, sizeof(libxsmm_meqn_fusion_knobs));

  if ( eqn == NULL ) {
    fprintf( stderr, "The requested equation does not exist... nothing to JIT,,,\n" );
    return;
  }

  if ( libxsmm_generator_matequation_aarch64_valid_arch_precision( io_generated_code, eqn, i_mateqn_desc) == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
    return;
  }

  /* Some basic initialization of the config kernel */
  libxsmm_generator_matequation_aarch64_init_micro_kernel_config(io_generated_code, &l_kernel_config);

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define gp register mapping */
  memset(&l_gp_reg_mapping, 0, sizeof(l_gp_reg_mapping));
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_AARCH64_GP_REG_X0;
  l_gp_reg_mapping.temp_reg    = LIBXSMM_AARCH64_GP_REG_X6;
  l_gp_reg_mapping.gp_reg_scratch_0    = LIBXSMM_AARCH64_GP_REG_X6;
  l_gp_reg_mapping.gp_reg_out = LIBXSMM_AARCH64_GP_REG_X1;
  l_gp_reg_mapping.temp_reg2   = LIBXSMM_AARCH64_GP_REG_X2;
  l_gp_reg_mapping.gp_reg_scratch_1 = LIBXSMM_AARCH64_GP_REG_X2;
  l_gp_reg_mapping.temp_reg3   = LIBXSMM_AARCH64_GP_REG_X3;
  l_gp_reg_mapping.gp_reg_scratch_2 = LIBXSMM_AARCH64_GP_REG_X3;
  l_gp_reg_mapping.gp_reg_m_loop = LIBXSMM_AARCH64_GP_REG_X4;
  l_gp_reg_mapping.gp_reg_n_loop = LIBXSMM_AARCH64_GP_REG_X5;
  l_kernel_config.n_avail_gpr = 19;
  l_kernel_config.gpr_pool[0] = LIBXSMM_AARCH64_GP_REG_X7; l_kernel_config.gpr_pool[1] = LIBXSMM_AARCH64_GP_REG_X8;
  l_kernel_config.gpr_pool[2] = LIBXSMM_AARCH64_GP_REG_X9; l_kernel_config.gpr_pool[3] = LIBXSMM_AARCH64_GP_REG_X10;
  l_kernel_config.gpr_pool[4] = LIBXSMM_AARCH64_GP_REG_X11; l_kernel_config.gpr_pool[5] = LIBXSMM_AARCH64_GP_REG_X12;
  l_kernel_config.gpr_pool[6] = LIBXSMM_AARCH64_GP_REG_X13; l_kernel_config.gpr_pool[7] = LIBXSMM_AARCH64_GP_REG_X14;
  l_kernel_config.gpr_pool[8] = LIBXSMM_AARCH64_GP_REG_X15; l_kernel_config.gpr_pool[9] = LIBXSMM_AARCH64_GP_REG_X19;
  l_kernel_config.gpr_pool[10] = LIBXSMM_AARCH64_GP_REG_X20; l_kernel_config.gpr_pool[11] = LIBXSMM_AARCH64_GP_REG_X21;
  l_kernel_config.gpr_pool[12] = LIBXSMM_AARCH64_GP_REG_X22; l_kernel_config.gpr_pool[13] = LIBXSMM_AARCH64_GP_REG_X23;
  l_kernel_config.gpr_pool[14] = LIBXSMM_AARCH64_GP_REG_X24; l_kernel_config.gpr_pool[15] = LIBXSMM_AARCH64_GP_REG_X25;
  l_kernel_config.gpr_pool[16] = LIBXSMM_AARCH64_GP_REG_X26; l_kernel_config.gpr_pool[17] = LIBXSMM_AARCH64_GP_REG_X27;
  l_kernel_config.gpr_pool[18] = LIBXSMM_AARCH64_GP_REG_X28;

  if ((eqn->eqn_root->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (eqn->eqn_root->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_UNZIP)) {
    l_kernel_config.n_avail_gpr = l_kernel_config.n_avail_gpr - 1;
    l_gp_reg_mapping.gp_reg_offset = LIBXSMM_AARCH64_GP_REG_X28;
  }

  jiting_queue = (libxsmm_matrix_eqn**) malloc(max_queue_size * sizeof(libxsmm_matrix_eqn*));
  fusion_knobs.may_fuse_xgemm = 1;
  libxsmm_generator_decompose_equation_tree_aarch64( eqn, jiting_queue, &queue_size, &fusion_knobs);

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
      if (cur_eqn->eqn_root->tmp.id >= 0) {
        libxsmm_generator_meqn_getaddr_stack_tmp_i_aarch64( io_generated_code,
            cur_eqn->eqn_root->tmp.id * l_kernel_config.tmp_size, l_gp_reg_mapping.gp_reg_scratch_0, l_gp_reg_mapping.temp_reg );
      } else {
        libxsmm_blasint arg_tmp_id = -1-cur_eqn->eqn_root->tmp.id;
        libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, l_gp_reg_mapping.gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 8, temp_reg );
        libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_ADD, temp_reg, l_gp_reg_mapping.temp_reg2, temp_reg, (long long)arg_tmp_id*32);
        libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, temp_reg, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, temp_reg );
      }
      copy_mateqn_desc.datatype = LIBXSMM_CAST_UCHAR(cur_eqn->eqn_root->tmp.dtype);
    }
    libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STR_I_OFF, l_gp_reg_mapping.gp_reg_param_struct, LIBXSMM_AARCH64_GP_REG_UNDEF, 16, l_gp_reg_mapping.temp_reg );

    if (libxsmm_generator_matequation_is_eqn_node_breaking_point(cur_eqn->eqn_root, &fusion_knobs) > 0) {
      /* For these nodes use strategy via scratch */
      /* Re assign visit_stamps to current equation tree */
      libxsmm_meqn_assign_timestamps(cur_eqn);
      if (eqn_tree_id < queue_size - 1) {
        if ((cur_eqn->eqn_root->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) &&
            ((cur_eqn->eqn_root->info.t_op.is_matmul == 1) || (cur_eqn->eqn_root->info.t_op.is_brgemm == 1))) {
          copy_mateqn_desc.ldo = cur_eqn->eqn_root->tmp.ld;
        } else {
          copy_mateqn_desc.ldo = cur_eqn->eqn_root->tmp.m;
        }
      }
#if 0
      printf("\nJITing tree with scratch %d and ldo is %d\n", eqn_tree_id, copy_mateqn_desc.ldo);
      libxsmm_meqn_trv_dbg_print( cur_eqn->eqn_root, 0);
#endif
      l_kernel_config.meltw_kernel_config.vector_name = l_kernel_config.vector_name;
      libxsmm_generator_matequation_tmp_stack_scratch_aarch64_kernel(io_generated_code, &copy_mateqn_desc, &l_gp_reg_mapping, &l_kernel_config, &l_loop_label_tracker, cur_eqn);
    } else {
      /* For these nodes use strategy via regblocks */
      /* Re-optimize current tree */
      if (((cur_eqn->eqn_root->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (libxsmm_meqn_is_unary_opcode_reduce_to_scalar(cur_eqn->eqn_root->info.u_op.type) > 0)) ||
          ((cur_eqn->eqn_root->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) && (libxsmm_meqn_is_binary_opcode_reduce_to_scalar(cur_eqn->eqn_root->info.b_op.type) > 0))) {
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
      if ((cur_eqn->eqn_root->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (cur_eqn->eqn_root->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_UNZIP)) {
        libxsmm_generator_meqn_getval_stack_var_aarch64( io_generated_code, LIBXSMM_MEQN_STACK_VAR_CONST_9, l_gp_reg_mapping.gp_reg_offset );
        libxsmm_aarch64_instruction_alu_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_LDR_I_OFF, l_gp_reg_mapping.gp_reg_offset, LIBXSMM_AARCH64_GP_REG_UNDEF, 0, l_gp_reg_mapping.gp_reg_offset );
      }
      libxsmm_meqn_reoptimize(cur_eqn);
      memset(&(l_kernel_config.meltw_kernel_config), 0, sizeof(libxsmm_mateltwise_kernel_config));
#if 0
      printf("\nJITing tree with regblocks %d and ldo is %d\n", eqn_tree_id, copy_mateqn_desc.ldo);
      libxsmm_meqn_trv_dbg_print( cur_eqn->eqn_root, 0);
#endif
      l_kernel_config.meltw_kernel_config.vector_name = l_kernel_config.vector_name;
      libxsmm_generator_matequation_tmp_register_block_aarch64_kernel(io_generated_code, &copy_mateqn_desc, &l_gp_reg_mapping, &l_kernel_config, &l_loop_label_tracker, cur_eqn);
    }
  }

  /* Destroy stack frame */
  libxsmm_generator_matequation_destroy_stack_frame_aarch64(  io_generated_code,  &l_kernel_config, &l_gp_reg_mapping, strategy);
  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xe0f );

  free(jiting_queue);
}

