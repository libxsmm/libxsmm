/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_matequation_avx_avx512.h"
#include "generator_matequation_scratch_avx_avx512.h"
#include "generator_matequation_regblocks_avx_avx512.h"
#include "generator_mateltwise_sse_avx_avx512.h"
#include "generator_mateltwise_unary_binary_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "generator_common_x86.h"
#include "libxsmm_matrixeqn.h"


LIBXSMM_API_INTERN
void libxsmm_generator_matequation_init_micro_kernel_config( libxsmm_generated_code*         io_generated_code,
    libxsmm_matequation_kernel_config*    io_micro_kernel_config) {
  memset(io_micro_kernel_config, 0, sizeof(*io_micro_kernel_config)); /* avoid warning "maybe used uninitialized" */
  if ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX ) {
    io_micro_kernel_config->instruction_set = io_generated_code->arch;
    io_micro_kernel_config->alu_add_instruction = LIBXSMM_X86_INSTR_ADDQ;
    io_micro_kernel_config->alu_sub_instruction = LIBXSMM_X86_INSTR_SUBQ;
    io_micro_kernel_config->alu_cmp_instruction = LIBXSMM_X86_INSTR_CMPQ;
    io_micro_kernel_config->alu_jmp_instruction = LIBXSMM_X86_INSTR_JL;
    io_micro_kernel_config->alu_mov_instruction = LIBXSMM_X86_INSTR_MOVQ;
    io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
    io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVUPS;
    io_micro_kernel_config->vmove_instruction_out= LIBXSMM_X86_INSTR_VMOVUPS;
    io_micro_kernel_config->vector_name = 'z';
  } else {
    io_micro_kernel_config->instruction_set = io_generated_code->arch;
    io_micro_kernel_config->alu_add_instruction = LIBXSMM_X86_INSTR_ADDQ;
    io_micro_kernel_config->alu_sub_instruction = LIBXSMM_X86_INSTR_SUBQ;
    io_micro_kernel_config->alu_cmp_instruction = LIBXSMM_X86_INSTR_CMPQ;
    io_micro_kernel_config->alu_jmp_instruction = LIBXSMM_X86_INSTR_JL;
    io_micro_kernel_config->alu_mov_instruction = LIBXSMM_X86_INSTR_MOVQ;
    io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VPXORD;
    io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVUPS;
    io_micro_kernel_config->vmove_instruction_out= LIBXSMM_X86_INSTR_VMOVUPS;
    io_micro_kernel_config->vector_name = 'y';
  }
}

LIBXSMM_API_INTERN
int libxsmm_generator_mateqn_get_rbp_relative_offset( libxsmm_meqn_stack_var stack_var ) {
  /* The stack at exit of setup looks like this:
   *
   *      Return address                            <-- RBP+8
   *      Entry/saved RBP                           <-- RBP
   *      Param_struct_ptr15                        <-- RBP-8
   *      Param_struct_ptr14                        <-- RBP-16
   *      Param_struct_ptr13                        <-- RBP-24
   *      Param_struct_ptr12                        <-- RBP-32
   *      Param_struct_ptr11                        <-- RBP-40
   *      Param_struct_ptr10                        <-- RBP-48
   *      Param_struct_ptr9                         <-- RBP-56
   *      Param_struct_ptr8                         <-- RBP-64
   *      Param_struct_ptr7                         <-- RBP-72
   *      Param_struct_ptr6                         <-- RBP-80
   *      Param_struct_ptr5                         <-- RBP-88
   *      Param_struct_ptr4                         <-- RBP-96
   *      Param_struct_ptr3                         <-- RBP-104
   *      Param_struct_ptr2                         <-- RBP-112
   *      Param_struct_ptr1                         <-- RBP-120
   *      Param_struct_ptr0                         <-- RBP-128
   *      Scratch ptr in stack (to be filled)       <-- RBP-136
   *      Address scratch ptrin stack (to be filled)<-- RBP-144
   *      Saved equation output ptr                 <-- RBP-152
   *      Const_0                                   <-- RBP-160
   *      ...
   *      Const_9                                   <-- RBP-232
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
void libxsmm_generator_meqn_getaddr_stack_var( libxsmm_generated_code*              io_generated_code,
                                                libxsmm_meqn_stack_var              stack_var,
                                                unsigned int                        i_gp_reg ) {
  int offset = libxsmm_generator_mateqn_get_rbp_relative_offset(stack_var);
  /* make sure we requested a legal stack var */
  if (offset == 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, i_gp_reg);
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg, offset);
}

LIBXSMM_API_INTERN
void libxsmm_generator_meqn_getval_stack_var( libxsmm_generated_code*               io_generated_code,
                                                libxsmm_meqn_stack_var              stack_var,
                                                unsigned int                        i_gp_reg ) {
  int offset = libxsmm_generator_mateqn_get_rbp_relative_offset(stack_var);
  /* make sure we requested a legal stack var */
  if (offset == 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, offset, i_gp_reg, 0 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_meqn_getaddr_stack_tmp_i( libxsmm_generated_code*            io_generated_code,
                                                unsigned int                        i_tmp_offset_i,
                                                unsigned int                        i_gp_reg ) {
  libxsmm_generator_meqn_getval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_SCRATCH_PTR, i_gp_reg );
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg, i_tmp_offset_i);
}

LIBXSMM_API_INTERN
void libxsmm_generator_meqn_getaddr_stack_tmpaddr_i( libxsmm_generated_code*            io_generated_code,
                                                unsigned int                        i_tmp_offset_i,
                                                unsigned int                        i_gp_reg ) {
  libxsmm_generator_meqn_getval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_ADDR_SCRATCH_PTR, i_gp_reg );
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg, i_tmp_offset_i);
}

LIBXSMM_API_INTERN
void libxsmm_generator_meqn_setval_stack_var( libxsmm_generated_code*               io_generated_code,
                                                libxsmm_meqn_stack_var              stack_var,
                                                unsigned int                        i_gp_reg ) {
  int offset = libxsmm_generator_mateqn_get_rbp_relative_offset(stack_var);
  /* make sure we requested to set  a legal stack var */
  if (offset >= 0) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
  libxsmm_x86_instruction_alu_mem( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_UNDEF, 0, offset, i_gp_reg, 1 );
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_setup_stack_frame( libxsmm_generated_code*   io_generated_code,
                                              const libxsmm_meqn_descriptor*                      i_mateqn_desc,
                                              libxsmm_matequation_gp_reg_mapping*                 i_gp_reg_mapping,
                                              libxsmm_matequation_kernel_config*                  i_micro_kernel_config,
                                              libxsmm_matrix_eqn*                                 i_eqn,
                                              unsigned int                                        i_strategy ) {

  unsigned int temp_reg                     = LIBXSMM_X86_GP_REG_R8;
  unsigned int skip_pushpops_callee_gp_reg  = 0;
  unsigned int allocate_scratch = 1;

  LIBXSMM_UNUSED(i_mateqn_desc);

  i_micro_kernel_config->skip_pushpops_callee_gp_reg = skip_pushpops_callee_gp_reg;
  libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBP );
  libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_RBP);
  libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, 360 );

  /* The stack at exit of setup looks like this:
   *
   *      Return address                            <-- RBP+8
   *      Entry/saved RBP                           <-- RBP
   *      Param_struct_ptr15                        <-- RBP-8
   *      Param_struct_ptr14                        <-- RBP-16
   *      Param_struct_ptr13                        <-- RBP-24
   *      Param_struct_ptr12                        <-- RBP-32
   *      Param_struct_ptr11                        <-- RBP-40
   *      Param_struct_ptr10                        <-- RBP-48
   *      Param_struct_ptr9                         <-- RBP-56
   *      Param_struct_ptr8                         <-- RBP-64
   *      Param_struct_ptr7                         <-- RBP-72
   *      Param_struct_ptr6                         <-- RBP-80
   *      Param_struct_ptr5                         <-- RBP-88
   *      Param_struct_ptr4                         <-- RBP-96
   *      Param_struct_ptr3                         <-- RBP-104
   *      Param_struct_ptr2                         <-- RBP-112
   *      Param_struct_ptr1                         <-- RBP-120
   *      Param_struct_ptr0                         <-- RBP-128
   *      Scratch ptr in stack (to be filled)       <-- RBP-136
   *      Address scratch ptrin stack (to be filled)<-- RBP-144
   *      Saved equation output ptr                 <-- RBP-152
   *      Const_0                                   <-- RBP-160
   *      ...
   *      Const_9                                   <-- RBP-232
   *
   * * */

  if (allocate_scratch > 0) {
    unsigned int scratch_size = 0;
    unsigned int addr_scratch_size = 0;

    /* Now align RSP to 64 byte boundary */
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code, i_micro_kernel_config->alu_mov_instruction, temp_reg, 0xFFFFFFFFFFFFFFC0 );
    libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_ANDQ, temp_reg, LIBXSMM_X86_GP_REG_RSP);

    if (i_strategy == JIT_STRATEGY_USING_TMP_SCRATCH_BLOCKS) {
      /* TODO: Now we allocate tmps with dsize float */
      int tree_max_comp_tsize = i_eqn->eqn_root->tree_max_comp_tsize;
      libxsmm_blasint n_tmp = i_eqn->eqn_root->reg_score;
      libxsmm_blasint tmp_size = i_eqn->eqn_root->max_tmp_size * tree_max_comp_tsize;
      tmp_size = (tmp_size % 64 == 0) ? tmp_size : ((tmp_size + 63)/64) * 64;
      scratch_size = tmp_size * n_tmp;
      i_micro_kernel_config->tmp_size = tmp_size;
      /* make scratch size multiple of 64b */
      scratch_size = (scratch_size % 64 == 0) ? scratch_size : ((scratch_size + 63)/64) * 64;
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, scratch_size );
      libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_SCRATCH_PTR, LIBXSMM_X86_GP_REG_RSP );
      if (libxsmm_verbosity < 0) {
        fprintf( stderr, "JITing Matrix Equation with STACK-ALLOCATED TEMPS (n_tmp = %d , stack_scratch_size = %.5g KB)\n", n_tmp, (1.0*scratch_size)/1024.0 );
      }
    } else if (i_strategy == JIT_STRATEGY_USING_TMP_REGISTER_BLOCKS) {
      libxsmm_blasint n_args = i_eqn->eqn_root->n_args;
      libxsmm_blasint n_max_opargs = i_eqn->eqn_root->visit_timestamp + 1;
      i_micro_kernel_config->n_args = n_args;
      addr_scratch_size = n_args * 8 + n_max_opargs * 32;
      /* make addr scratch size multiple of 64b */
      addr_scratch_size = (addr_scratch_size % 64 == 0) ? addr_scratch_size : ((addr_scratch_size + 63)/64) * 64;
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, addr_scratch_size );
      libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_ADDR_SCRATCH_PTR, LIBXSMM_X86_GP_REG_RSP );
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
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, scratch_size );
      libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_SCRATCH_PTR, LIBXSMM_X86_GP_REG_RSP );
      /* make addr scratch size multiple of 64b */
      i_micro_kernel_config->n_args = n_args;
      addr_scratch_size = n_args * 8 + n_max_opargs * 32;
      addr_scratch_size = (addr_scratch_size % 64 == 0) ? addr_scratch_size : ((addr_scratch_size + 63)/64) * 64;
      libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, addr_scratch_size );
      libxsmm_generator_meqn_setval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_ADDR_SCRATCH_PTR, LIBXSMM_X86_GP_REG_RSP );
      if (libxsmm_verbosity < 0) {
        fprintf( stderr, "JITing Matrix Equation with HYBRID STRATEGY for TEMPS (n_tmp = %d , stack_scratch_size = %.5g KB , addr_scratch_size = %.5g KB)\n", n_tmp, (1.0*scratch_size)/1024.0, (1.0*addr_scratch_size)/1024.0 );
      }
    }
  }

  /* Now push to RSP the callee-save registers */
  /* on windows we also have to save xmm6-xmm15 */
#if defined(_WIN32) || defined(__CYGWIN__)
  {
    unsigned int l_i;
    unsigned int l_simd_store_instr = (io_generated_code->arch < LIBXSMM_X86_AVX) ? LIBXSMM_X86_INSTR_MOVUPS_ST
                                                                                  : LIBXSMM_X86_INSTR_VMOVUPS_ST;
    /* decrease rsp by 160 (10x16) */
    libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_SUBQ, LIBXSMM_X86_GP_REG_RSP, 160);
    /* save 10 xmm onto the stack */
    for (l_i = 0; l_i < 10; ++l_i) {
      libxsmm_x86_instruction_vec_compute_mem_1reg_mask(io_generated_code, l_simd_store_instr, 'x', LIBXSMM_X86_GP_REG_RSP,
        LIBXSMM_X86_GP_REG_UNDEF, 0, 144 - (l_i * 16), 0, 6 + l_i, 0, 0);
    }
  }
#endif
  if (skip_pushpops_callee_gp_reg == 0) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBX );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R13 );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R14 );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_R15 );
#if defined(_WIN32) || defined(__CYGWIN__)
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDI );
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RSI );
#endif
  }

  /* Store the out ptr in stack */
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
    if ((i_eqn->eqn_root->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (i_eqn->eqn_root->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_UNZIP)) {
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
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_destroy_stack_frame( libxsmm_generated_code*                   io_generated_code,
                                              libxsmm_matequation_kernel_config*                  i_micro_kernel_config,
                                              libxsmm_matequation_gp_reg_mapping*                 i_gp_reg_mapping,
                                              unsigned int                                        i_strategy  ) {
  LIBXSMM_UNUSED(i_gp_reg_mapping);
  LIBXSMM_UNUSED(i_strategy);
  if (i_micro_kernel_config->skip_pushpops_callee_gp_reg == 0) {
#if defined(_WIN32) || defined(__CYGWIN__)
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RSI );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RDI );
#endif
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R15 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R14 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R13 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_R12 );
    libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBX );
  }
  /* on windows we also have to restore xmm6-xmm15 */
#if defined(_WIN32) || defined(__CYGWIN__)
  {
    unsigned int l_i;
    unsigned int l_simd_load_instr = (io_generated_code->arch < LIBXSMM_X86_AVX) ? LIBXSMM_X86_INSTR_MOVUPS_LD
                                                                                 : LIBXSMM_X86_INSTR_VMOVUPS_LD;
    /* save 10 xmm onto the stack */
    for (l_i = 0; l_i < 10; ++l_i) {
      libxsmm_x86_instruction_vec_compute_mem_1reg_mask(io_generated_code, l_simd_load_instr, 'x', LIBXSMM_X86_GP_REG_RSP,
        LIBXSMM_X86_GP_REG_UNDEF, 0, 144 - (l_i * 16), 0, 6 + l_i, 0, 0);
    }
    /* increase rsp by 160 (10x16) */
    libxsmm_x86_instruction_alu_imm(io_generated_code, LIBXSMM_X86_INSTR_ADDQ, LIBXSMM_X86_GP_REG_RSP, 160);
  }
#endif

  libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_RSP);
  libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBP );
}

LIBXSMM_API_INTERN
libxsmm_meqn_elem* libxsmm_generator_matequation_find_op_at_timestamp(libxsmm_meqn_elem* cur_node, libxsmm_blasint timestamp) {
  libxsmm_meqn_elem *result = NULL;
  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG ) {
    result = NULL;
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY ) {
    if (cur_node->visit_timestamp == timestamp) {
      result = cur_node;
    } else {
      result = libxsmm_generator_matequation_find_op_at_timestamp(cur_node->le, timestamp);
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    if (cur_node->visit_timestamp == timestamp) {
      result = cur_node;
    } else {
      result = libxsmm_generator_matequation_find_op_at_timestamp(cur_node->le, timestamp);
      if (result == NULL) {
        result = libxsmm_generator_matequation_find_op_at_timestamp(cur_node->ri, timestamp);
      }
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
    if (cur_node->visit_timestamp == timestamp) {
      result = cur_node;
    } else {
      result = libxsmm_generator_matequation_find_op_at_timestamp(cur_node->le, timestamp);
      if (result == NULL) {
        result = libxsmm_generator_matequation_find_op_at_timestamp(cur_node->ri, timestamp);
        if (result == NULL) {
          result = libxsmm_generator_matequation_find_op_at_timestamp(cur_node->r2, timestamp);
        }
      }
    }
  }
  return result;
}

LIBXSMM_API_INTERN
int libxsmm_generator_matequation_is_xgemm_node_supporting_fusion(libxsmm_meqn_elem  *xgemm_node) {
  int result = 0;
  if (((xgemm_node->le->tmp.dtype == LIBXSMM_DATATYPE_BF16) && (xgemm_node->ri->tmp.dtype == LIBXSMM_DATATYPE_BF16)) ||
      ((xgemm_node->le->tmp.dtype == LIBXSMM_DATATYPE_F16) && (xgemm_node->ri->tmp.dtype == LIBXSMM_DATATYPE_F16)) ||
      ((xgemm_node->le->tmp.dtype == LIBXSMM_DATATYPE_BF8) && (xgemm_node->ri->tmp.dtype == LIBXSMM_DATATYPE_BF8)) ||
      ((xgemm_node->le->tmp.dtype == LIBXSMM_DATATYPE_F32) && (xgemm_node->ri->tmp.dtype == LIBXSMM_DATATYPE_F32))) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INTERN
int libxsmm_generator_matequation_is_xgemm_node(libxsmm_meqn_elem  *cur_node) {
  int result = 0;
  if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY ) {
    if ( (cur_node->info.b_op.is_matmul  == 1) ||
         (cur_node->info.b_op.is_brgemm  == 1)) {
      result = 1;
    }
  } else if ( cur_node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY ) {
    if ( (cur_node->info.t_op.is_matmul == 1) ||
         (cur_node->info.t_op.is_brgemm == 1) ) {
      result = 1;
    }
  } else {
    result = 0;
  }
  return result;
}

LIBXSMM_API_INTERN
int libxsmm_generator_matequation_is_gather_node(libxsmm_meqn_elem  *cur_node) {
  int result = 0;
  if ((cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (cur_node->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_GATHER)) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INTERN
int libxsmm_generator_matequation_is_eqn_node_breaking_point(libxsmm_meqn_elem *node, libxsmm_meqn_fusion_knobs *fusion_knobs) {
  int result = 0;
  if (node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
    if ( node->info.u_op.type  == LIBXSMM_MELTW_TYPE_UNARY_TANH ||
         node->info.u_op.type  == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV ||
         node->info.u_op.type  == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID ||
         node->info.u_op.type  == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV ||
         /*node->info.u_op.type  == LIBXSMM_MELTW_TYPE_UNARY_EXP ||*/
         node->info.u_op.type  == LIBXSMM_MELTW_TYPE_UNARY_GELU ||
         node->info.u_op.type  == LIBXSMM_MELTW_TYPE_UNARY_RELU ||
         node->info.u_op.type  == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV ||
         node->info.u_op.type  == LIBXSMM_MELTW_TYPE_UNARY_IDENTITY ||
         node->info.u_op.type  == LIBXSMM_MELTW_TYPE_UNARY_GATHER ||
         node->info.u_op.type  == LIBXSMM_MELTW_TYPE_UNARY_SCATTER ||
         libxsmm_meqn_is_unary_opcode_transform_kernel(node->info.u_op.type) ||
         libxsmm_meqn_is_unary_opcode_reduce_kernel(node->info.u_op.type) ||
         libxsmm_meqn_is_unary_opcode_reduce_cols_idx_kernel(node->info.u_op.type) ) {
      result = 1;
    }
  }

  if (node->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) {
    if (node->info.t_op.type == LIBXSMM_MELTW_TYPE_TERNARY_SELECT) {
      result = 1;
    }
  }

  if (node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) {
    if (node->info.b_op.type == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_GT ||
        node->info.b_op.type == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_GE ||
        node->info.b_op.type == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LT ||
        node->info.b_op.type == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LE ||
        node->info.b_op.type == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_EQ ||
        node->info.b_op.type == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_NE) {
      result = 1;
    }
  }

  if (libxsmm_generator_matequation_is_xgemm_node(node) > 0) {
    result = 1;
  }

  /* Allow to break this in order to enable potential fusion of colbias add in BRGEMM */
  if ((node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) && (node->info.b_op.type == LIBXSMM_MELTW_TYPE_BINARY_ADD) && (fusion_knobs->may_fuse_xgemm > 0)) {
    if (libxsmm_generator_matequation_is_xgemm_node(node->le) > 0) {
      if (libxsmm_generator_matequation_is_xgemm_node_supporting_fusion(node->le) > 0) {
        if ((node->info.b_op.flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1) > 0) {
          result = 1;
        }
      }
    } else if (libxsmm_generator_matequation_is_xgemm_node(node->ri) > 0) {
      if (libxsmm_generator_matequation_is_xgemm_node_supporting_fusion(node->ri) > 0) {
        if ((node->info.b_op.flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0) > 0) {
          result = 1;
        }
      }
    } else {
      /* Do nothing */
    }
  }

  return result;
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_enqueue_equation(libxsmm_matrix_eqn *eqn, libxsmm_matrix_eqn **jiting_queue, unsigned int *queue_size) {
  jiting_queue[*queue_size] = eqn;
  *queue_size = *queue_size + 1;
}

LIBXSMM_API_INTERN int libxsmm_generator_matequation_is_unary_with_bcast(libxsmm_bitfield flags) {
  int result = 0;
  if ( ((flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL) > 0) ||
       ((flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR) > 0) ) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INTERN int libxsmm_generator_matequation_is_binary_with_bcast(libxsmm_bitfield flags) {
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

LIBXSMM_API_INTERN int libxsmm_generator_matequation_is_ternary_with_bcast(libxsmm_bitfield flags) {
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

LIBXSMM_API_INTERN int libxsmm_generator_matequation_is_unary_bcast_arg_an_inputarg(libxsmm_bitfield flags, libxsmm_meqn_elem *node) {
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

LIBXSMM_API_INTERN int libxsmm_generator_matequation_is_binary_bcast_arg_an_inputarg(libxsmm_bitfield flags, libxsmm_meqn_elem *node) {
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

LIBXSMM_API_INTERN int libxsmm_generator_matequation_is_ternary_bcast_arg_an_inputarg(libxsmm_bitfield flags, libxsmm_meqn_elem *node) {
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

LIBXSMM_API_INTERN
libxsmm_meqn_fusion_pattern_type libxsmm_generator_matequation_find_xgemm_fusion_pattern_with_ancestors(libxsmm_meqn_elem *xgemm_node) {
  libxsmm_meqn_fusion_pattern_type result = LIBXSMM_MATRIX_EQN_FUSION_PATTERN_NONE;
  if (xgemm_node->up != NULL) {
    if (xgemm_node->up->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) {
      libxsmm_meqn_elem     *sibling_node = NULL;
      libxsmm_bitfield bcast_flag = LIBXSMM_MELTW_FLAG_BINARY_NONE;
      if (xgemm_node->up->le == xgemm_node) {
        sibling_node = xgemm_node->up->ri;
        bcast_flag = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1;
      } else {
        sibling_node = xgemm_node->up->le;
        bcast_flag = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
      }
      if ((xgemm_node->up->info.b_op.type == LIBXSMM_MELTW_TYPE_BINARY_ADD) &&
          (sibling_node->type == LIBXSMM_MATRIX_EQN_NODE_ARG) &&
          ((xgemm_node->up->info.b_op.flags & bcast_flag) == bcast_flag)) {
        /* For sure have add  colbias node above */
        if (xgemm_node->up->up != NULL) {
          if ((xgemm_node->up->up->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) &&
              ((xgemm_node->up->up->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_RELU) ||
               (xgemm_node->up->up->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID)) ) {
            result = LIBXSMM_MATRIX_EQN_FUSION_PATTERN_XGEMM_COLBIAS_ADD_UNARY;
          } else {
            result = LIBXSMM_MATRIX_EQN_FUSION_PATTERN_XGEMM_COLBIAS_ADD;
          }
        } else {
          result = LIBXSMM_MATRIX_EQN_FUSION_PATTERN_XGEMM_COLBIAS_ADD;
        }
      }
    } else if (xgemm_node->up->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
      if ( (xgemm_node->up->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_RELU) ||
           (xgemm_node->up->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID) ) {
        result = LIBXSMM_MATRIX_EQN_FUSION_PATTERN_XGEMM_UNARY;
      }
    } else {
      /* Do nothing... */
    }
  }
  return result;
}

LIBXSMM_API_INTERN
libxsmm_meqn_fusion_pattern_type libxsmm_generator_matequation_find_gather_fusion_pattern_with_ancestors(libxsmm_meqn_elem *gather_node) {
  libxsmm_meqn_fusion_pattern_type result = LIBXSMM_MATRIX_EQN_FUSION_PATTERN_NONE;
  if ((gather_node->info.u_op.flags & LIBXSMM_MELTW_FLAG_UNARY_GS_COLS) > 0) {
    if (gather_node->up != NULL) {
      if (gather_node->up->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) {
        if (gather_node->up->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD || gather_node->up->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX || gather_node->up->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MIN) {
          if ((gather_node->up->info.u_op.flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS) > 0) {
            result = LIBXSMM_MATRIX_EQN_FUSION_PATTERN_GATHER_COLS_REDUCE_COLS;
          }
        }
      }
    }
  }
  return result;
}

LIBXSMM_API_INTERN
libxsmm_meqn_fusion_pattern_type libxsmm_generator_matequation_find_fusion_pattern_with_ancestors(libxsmm_meqn_elem *cur_node, libxsmm_meqn_fusion_knobs *fusion_knobs) {
  libxsmm_meqn_fusion_pattern_type result = LIBXSMM_MATRIX_EQN_FUSION_PATTERN_NONE;
  /* Check for xgemm fusion patterns*/
  if ((libxsmm_generator_matequation_is_xgemm_node(cur_node) > 0) && (fusion_knobs->may_fuse_xgemm > 0) ) {
    if (libxsmm_generator_matequation_is_xgemm_node_supporting_fusion(cur_node) > 0) {
      result = libxsmm_generator_matequation_find_xgemm_fusion_pattern_with_ancestors( cur_node );
    }
  }

  /* Check for gather-op fusion pattern */
  if (libxsmm_generator_matequation_is_gather_node(cur_node) > 0) {
    result = libxsmm_generator_matequation_find_gather_fusion_pattern_with_ancestors( cur_node );
  }

  return result;
}

LIBXSMM_API_INTERN
int libxsmm_generator_matequation_find_in_pos_for_colbias(libxsmm_meqn_elem *colbias_add_node) {
  int result = 0;
  if (colbias_add_node->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
    result = colbias_add_node->le->info.arg.in_pos;
  } else {
    result = colbias_add_node->ri->info.arg.in_pos;
  }
  return result;
}

LIBXSMM_API_INTERN
libxsmm_datatype libxsmm_generator_matequation_find_dtype_for_colbias(libxsmm_meqn_elem *colbias_add_node) {
  libxsmm_datatype result = LIBXSMM_DATATYPE_F32;
  if (colbias_add_node->le->type == LIBXSMM_MATRIX_EQN_NODE_ARG) {
    result = colbias_add_node->le->info.arg.dtype;
  } else {
    result = colbias_add_node->ri->info.arg.dtype;
  }
  return result;
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_apply_xgemm_fusion_pattern_transformation(libxsmm_meqn_fusion_pattern_type fusion_pattern,
                                               libxsmm_meqn_elem                *cur_node,
                                               libxsmm_meqn_elem                *new_arg_node,
                                               unsigned int                           *timestamp,
                                               unsigned int                           last_timestamp ) {
  if (fusion_pattern == LIBXSMM_MATRIX_EQN_FUSION_PATTERN_XGEMM_COLBIAS_ADD) {
    /* Collapse parent ADD node and enhance info in GEMM node */
    if (libxsmm_verbosity < 0) {
      fprintf( stderr, "Fusing XGEMM with column-bias ADD\n");
    }
    cur_node->fusion_info.xgemm.fused_colbias_add_op = 1;
    cur_node->fusion_info.xgemm.colbias_pos_in_arg = libxsmm_generator_matequation_find_in_pos_for_colbias(cur_node->up);
    cur_node->fusion_info.xgemm.colbias_dtype = libxsmm_generator_matequation_find_dtype_for_colbias(cur_node->up);
    (*timestamp)++;
    if (*timestamp < last_timestamp) {
      new_arg_node->up = cur_node->up->up;
      if (cur_node->up->up->le == cur_node->up) {
        cur_node->up->up->le = new_arg_node;
      } else if (cur_node->up->up->ri == cur_node->up) {
        cur_node->up->up->ri = new_arg_node;
      } else {
        cur_node->up->up->r2 = new_arg_node;
      }
    }
  } else if (fusion_pattern == LIBXSMM_MATRIX_EQN_FUSION_PATTERN_XGEMM_COLBIAS_ADD_UNARY) {
    /* Collapse parent ADD node & UNARY NODE and enhance info in GEMM node */
    if (cur_node->up->up->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_RELU) {
      cur_node->fusion_info.xgemm.fused_relu_op = 1;
      if (libxsmm_verbosity < 0) {
        fprintf( stderr, "Fusing XGEMM with column-bias ADD and unary RELU\n");
      }
    }
    if (cur_node->up->up->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID) {
      cur_node->fusion_info.xgemm.fused_sigmoid_op = 1;
      if (libxsmm_verbosity < 0) {
        fprintf( stderr, "Fusing XGEMM with column-bias ADD and unary SIGMOID\n");
      }
    }
    cur_node->fusion_info.xgemm.fused_colbias_add_op = 1;
    cur_node->fusion_info.xgemm.colbias_pos_in_arg = libxsmm_generator_matequation_find_in_pos_for_colbias(cur_node->up);
    cur_node->fusion_info.xgemm.colbias_dtype = libxsmm_generator_matequation_find_dtype_for_colbias(cur_node->up);
    (*timestamp) += 2;
    if (*timestamp < last_timestamp) {
      new_arg_node->up = cur_node->up->up->up;
      if (cur_node->up->up->up->le == cur_node->up->up) {
        cur_node->up->up->up->le = new_arg_node;
      } else if (cur_node->up->up->up->ri == cur_node->up->up)  {
        cur_node->up->up->up->ri = new_arg_node;
      } else {
        cur_node->up->up->up->r2 = new_arg_node;
      }
    }
  } else if (fusion_pattern == LIBXSMM_MATRIX_EQN_FUSION_PATTERN_XGEMM_UNARY) {
    /* Collapse parent UNARY node and enhance info in GEMM node */
    if (cur_node->up->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_RELU) {
      cur_node->fusion_info.xgemm.fused_relu_op = 1;
      if (libxsmm_verbosity < 0) {
        fprintf( stderr, "Fusing XGEMM with unary RELU\n");
      }
    }
    if (cur_node->up->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID) {
      cur_node->fusion_info.xgemm.fused_sigmoid_op = 1;
      if (libxsmm_verbosity < 0) {
        fprintf( stderr, "Fusing XGEMM with unary SIGMOID\n");
      }
    }
    (*timestamp)++;
    if (*timestamp < last_timestamp) {
      new_arg_node->up = cur_node->up->up;
      if (cur_node->up->up->le == cur_node->up) {
        cur_node->up->up->le = new_arg_node;
      } else if (cur_node->up->up->ri == cur_node->up)  {
        cur_node->up->up->ri = new_arg_node;
      } else {
        cur_node->up->up->r2 = new_arg_node;
      }
    }
  } else {
    /* Should not happen */
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_apply_gather_fusion_pattern_transformation(libxsmm_meqn_fusion_pattern_type fusion_pattern,
                                               libxsmm_meqn_elem                *cur_node,
                                               libxsmm_meqn_elem                *new_arg_node,
                                               unsigned int                           *timestamp,
                                               unsigned int                           last_timestamp ) {
  if (fusion_pattern == LIBXSMM_MATRIX_EQN_FUSION_PATTERN_GATHER_COLS_REDUCE_COLS) {
    cur_node->fusion_info.gather.idx_dtype = ((cur_node->info.u_op.flags & LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES) > 0) ? LIBXSMM_DATATYPE_I32 : LIBXSMM_DATATYPE_I64;
    cur_node->fusion_info.gather.idx_array_pos_in_arg = cur_node->le->info.arg.in_pos;
    cur_node->info.u_op.flags = 0;
    /* Collapse parent UNARY node and enhance info in gather node */
    if (cur_node->up->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD) {
      cur_node->fusion_info.gather.fused_reduce_cols_add = 1;
      cur_node->info.u_op.type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_ADD;
      cur_node->info.u_op.flags = 0;
      if (cur_node->fusion_info.gather.idx_dtype == LIBXSMM_DATATYPE_I32) {
        cur_node->info.u_op.flags |=  LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES;
      } else {
        cur_node->info.u_op.flags |=  LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES;
      }
      if (libxsmm_verbosity < 0) {
        fprintf( stderr, "Fusing GATHER-COLS with ADD-REDUCE-COLS\n");
      }
    } else if (cur_node->up->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX) {
      cur_node->fusion_info.gather.fused_reduce_cols_max = 1;
      cur_node->info.u_op.type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MAX;
      cur_node->info.u_op.flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INF_ACC;
      if (cur_node->fusion_info.gather.idx_dtype == LIBXSMM_DATATYPE_I32) {
        cur_node->info.u_op.flags |=  LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES;
      } else {
        cur_node->info.u_op.flags |=  LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES;
      }
      if (libxsmm_verbosity < 0) {
        fprintf( stderr, "Fusing GATHER-COLS with MAX-REDUCE-COLS\n");
      }
    } else if (cur_node->up->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MIN) {
      cur_node->fusion_info.gather.fused_reduce_cols_max = 1;
      cur_node->info.u_op.type = LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MIN;
      cur_node->info.u_op.flags = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INF_ACC;
      if (cur_node->fusion_info.gather.idx_dtype == LIBXSMM_DATATYPE_I32) {
        cur_node->info.u_op.flags |=  LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES;
      } else {
        cur_node->info.u_op.flags |=  LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES;
      }
      if (libxsmm_verbosity < 0) {
        fprintf( stderr, "Fusing GATHER-COLS with MIN-REDUCE-COLS\n");
      }
    } else {
      /* Should not happen */
    }
    /* Update tmp size of new pseudo-arg */
    new_arg_node->tmp.n = 1;

    (*timestamp)++;
    if (*timestamp < last_timestamp) {
      new_arg_node->up = cur_node->up->up;
      if (cur_node->up->up->le == cur_node->up) {
        cur_node->up->up->le = new_arg_node;
      } else if (cur_node->up->up->ri == cur_node->up)  {
        cur_node->up->up->ri = new_arg_node;
      } else {
        cur_node->up->up->r2 = new_arg_node;
      }
    }
  } else {
    /* Should not happen */
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_apply_fusion_pattern_transformation(libxsmm_meqn_fusion_pattern_type fusion_pattern,
                                               libxsmm_meqn_elem                *cur_node,
                                               libxsmm_meqn_elem                *new_arg_node,
                                               unsigned int                           *timestamp,
                                               unsigned int                           last_timestamp ) {
  if (libxsmm_generator_matequation_is_xgemm_node(cur_node) > 0) {
    libxsmm_generator_matequation_apply_xgemm_fusion_pattern_transformation(fusion_pattern, cur_node, new_arg_node, timestamp, last_timestamp );
  }
  if (libxsmm_generator_matequation_is_gather_node(cur_node) > 0) {
    libxsmm_generator_matequation_apply_gather_fusion_pattern_transformation(fusion_pattern, cur_node, new_arg_node, timestamp, last_timestamp );
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_decompose_equation_tree_x86( libxsmm_matrix_eqn *eqn, libxsmm_matrix_eqn **jiting_queue, unsigned int *queue_size, libxsmm_meqn_fusion_knobs *fusion_knobs) {
  libxsmm_meqn_elem *root = eqn->eqn_root;
  unsigned int last_timestamp = eqn->eqn_root->visit_timestamp;
  unsigned int timestamp = 0;

  for (timestamp = 0; timestamp <= last_timestamp;) {
    libxsmm_meqn_elem *cur_node = libxsmm_generator_matequation_find_op_at_timestamp(root, timestamp);
    if (timestamp == last_timestamp) {
      libxsmm_generator_matequation_enqueue_equation(eqn, jiting_queue, queue_size);
    }
    if ( (timestamp < last_timestamp) && ((libxsmm_generator_matequation_is_eqn_node_breaking_point(cur_node, fusion_knobs) > 0) || (libxsmm_generator_matequation_is_eqn_node_breaking_point(cur_node->up, fusion_knobs) > 0) ||
                                           ((cur_node->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (libxsmm_meqn_is_unary_opcode_reduce_to_scalar(cur_node->info.u_op.type) > 0)) ||
                                           ((cur_node->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) && (libxsmm_meqn_is_binary_opcode_reduce_to_scalar(cur_node->info.b_op.type) > 0)) ||
                                           ((cur_node->up->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (libxsmm_generator_matequation_is_unary_with_bcast(cur_node->up->info.u_op.flags) > 0) && (libxsmm_generator_matequation_is_unary_bcast_arg_an_inputarg(cur_node->up->info.u_op.flags, cur_node->up) == 0) ) ||
                                           ((cur_node->up->type == LIBXSMM_MATRIX_EQN_NODE_BINARY) && (libxsmm_generator_matequation_is_binary_with_bcast(cur_node->up->info.b_op.flags) > 0) && (libxsmm_generator_matequation_is_binary_bcast_arg_an_inputarg(cur_node->up->info.b_op.flags, cur_node->up) == 0)) ||
                                           ((cur_node->up->type == LIBXSMM_MATRIX_EQN_NODE_TERNARY) && (libxsmm_generator_matequation_is_ternary_with_bcast(cur_node->up->info.t_op.flags) > 0) && (libxsmm_generator_matequation_is_ternary_bcast_arg_an_inputarg(cur_node->up->info.t_op.flags, cur_node->up) == 0)))) {

      libxsmm_meqn_fusion_pattern_type fusion_pattern = LIBXSMM_MATRIX_EQN_FUSION_PATTERN_NONE;
      libxsmm_meqn_elem                *new_arg_node  = NULL;
      libxsmm_matrix_eqn                     *new_eqn       = NULL;
#if !defined(__clang_analyzer__)
      new_arg_node = (libxsmm_meqn_elem*)malloc(sizeof(libxsmm_meqn_elem));
      new_eqn = (libxsmm_matrix_eqn*)malloc(sizeof(libxsmm_matrix_eqn));
#endif

      if (NULL != new_arg_node && NULL != new_eqn) {
        union libxsmm_meqn_info info;
        info.arg.m = cur_node->tmp.m;
        info.arg.n = cur_node->tmp.n;
        info.arg.ld = cur_node->tmp.ld;
        info.arg.in_pos = -(cur_node->tmp.id + 1); /*(cur_node->tmp.id >= 0) ? -(cur_node->tmp.id + 1) : cur_node->tmp.id;*/
        info.arg.dtype = cur_node->tmp.dtype;

        new_arg_node->le = NULL;
        new_arg_node->ri = NULL;
        new_arg_node->r2 = NULL;
        new_arg_node->type = LIBXSMM_MATRIX_EQN_NODE_ARG;
        new_arg_node->info = info;
        new_arg_node->reg_score = 0;
        new_arg_node->tmp.dtype = cur_node->tmp.dtype;
        new_arg_node->tmp.m = cur_node->tmp.m;
        new_arg_node->tmp.n = cur_node->tmp.n;
        new_arg_node->tmp.ld = cur_node->tmp.ld;

        fusion_pattern = libxsmm_generator_matequation_find_fusion_pattern_with_ancestors( cur_node, fusion_knobs );

        if (fusion_pattern != LIBXSMM_MATRIX_EQN_FUSION_PATTERN_NONE) {
          libxsmm_generator_matequation_apply_fusion_pattern_transformation( fusion_pattern, cur_node, new_arg_node, &timestamp, last_timestamp );
        } else {
          new_arg_node->up = cur_node->up;
          if (cur_node->up->le == cur_node) {
            cur_node->up->le = new_arg_node;
          } else if (cur_node->up->ri == cur_node)  {
            cur_node->up->ri = new_arg_node;
          } else {
            cur_node->up->r2 = new_arg_node;
          }
        }
        new_eqn->eqn_root = cur_node;
        new_eqn->is_constructed = 1;
        libxsmm_generator_matequation_enqueue_equation(new_eqn, jiting_queue, queue_size);
      }
      else { /* error */
        free(new_arg_node);
        free(new_eqn);
      }
    }
    timestamp++;
  }
}

LIBXSMM_API_INTERN
libxsmm_blasint libxsmm_generator_matequation_x86_valid_arch_precision( libxsmm_generated_code*           io_generated_code,
                                                                        libxsmm_matrix_eqn*               i_eqn,
                                                                        const libxsmm_meqn_descriptor*    i_mateqn_desc) {
  libxsmm_blasint is_valid_arch_prec = 1;
  unsigned int has_inp_or_out_fp8 = ((libxsmm_meqn_any_args_dtype(i_eqn, LIBXSMM_DATATYPE_BF8) > 0) || (libxsmm_meqn_any_args_dtype(i_eqn, LIBXSMM_DATATYPE_HF8) > 0) ||
                                     (LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_OUT( i_mateqn_desc->datatype )) || (LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_OUT( i_mateqn_desc->datatype ))) ? 1 : 0;
  unsigned int has_inp_or_out_fp64= ((libxsmm_meqn_any_args_dtype(i_eqn, LIBXSMM_DATATYPE_F64) > 0) || (LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_OUT( i_mateqn_desc->datatype ))) ? 1 : 0;
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

  if (io_generated_code->arch < LIBXSMM_X86_AVX) {
    is_valid_arch_prec = 0;
  }
  if ((libxsmm_meqn_contains_opcode(i_eqn, LIBXSMM_MELTW_TYPE_UNARY_UNZIP, LIBXSMM_MELTW_TYPE_BINARY_ZIP, LIBXSMM_MELTW_TYPE_TERNARY_NONE) > 0) && (io_generated_code->arch < LIBXSMM_X86_AVX512_VL128_SKX)) {
    is_valid_arch_prec = 0;
  }
  if ((libxsmm_meqn_contains_opcode(i_eqn, LIBXSMM_MELTW_TYPE_UNARY_GELU, LIBXSMM_MELTW_TYPE_BINARY_NONE, LIBXSMM_MELTW_TYPE_TERNARY_NONE) > 0) && (io_generated_code->arch < LIBXSMM_X86_AVX2)) {
    is_valid_arch_prec = 0;
  }
  if ((libxsmm_meqn_contains_opcode(i_eqn, LIBXSMM_MELTW_TYPE_UNARY_GELU_INV, LIBXSMM_MELTW_TYPE_BINARY_NONE, LIBXSMM_MELTW_TYPE_TERNARY_NONE) > 0) && (io_generated_code->arch < LIBXSMM_X86_AVX2)) {
    is_valid_arch_prec = 0;
  }
  if ((has_inp_or_out_fp8 > 0) && (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX)) {
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
void libxsmm_generator_matequation_avx_avx512_kernel( libxsmm_generated_code*        io_generated_code,
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
  unsigned int strategy = JIT_STRATEGY_HYBRID;
  unsigned int eqn_tree_id = 0;
  unsigned int temp_reg = LIBXSMM_X86_GP_REG_R8;
  libxsmm_meqn_fusion_knobs fusion_knobs;
  memset(&fusion_knobs, 0, sizeof(libxsmm_meqn_fusion_knobs));

  if ( eqn == NULL ) {
    fprintf( stderr, "The requested equation does not exist... nothing to JIT,,,\n" );
    return;
  }

  if ( libxsmm_generator_matequation_x86_valid_arch_precision( io_generated_code, eqn, i_mateqn_desc) == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
    return;
  }

  /* Some basic initialization of the config kernel */
  libxsmm_generator_matequation_init_micro_kernel_config(io_generated_code, &l_kernel_config);

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define gp register mapping */
  memset(&l_gp_reg_mapping, 0, sizeof(l_gp_reg_mapping));
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RCX;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RDI;
#endif
  l_gp_reg_mapping.gp_reg_out = LIBXSMM_X86_GP_REG_RAX;
  l_gp_reg_mapping.temp_reg    = LIBXSMM_X86_GP_REG_RBX;
  l_gp_reg_mapping.temp_reg2   = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_m_loop = LIBXSMM_X86_GP_REG_R14;
  l_gp_reg_mapping.gp_reg_n_loop = LIBXSMM_X86_GP_REG_R15;
  l_kernel_config.n_avail_gpr = 8;
  l_kernel_config.gpr_pool[0] = LIBXSMM_X86_GP_REG_RSI; l_kernel_config.gpr_pool[1] = LIBXSMM_X86_GP_REG_RDX; l_kernel_config.gpr_pool[2] = LIBXSMM_X86_GP_REG_R8; l_kernel_config.gpr_pool[3] = LIBXSMM_X86_GP_REG_R9;
  l_kernel_config.gpr_pool[4] = LIBXSMM_X86_GP_REG_R10; l_kernel_config.gpr_pool[5] = LIBXSMM_X86_GP_REG_R11; l_kernel_config.gpr_pool[6] = LIBXSMM_X86_GP_REG_R12; l_kernel_config.gpr_pool[7] = LIBXSMM_X86_GP_REG_R13;

  if ((eqn->eqn_root->type == LIBXSMM_MATRIX_EQN_NODE_UNARY) && (eqn->eqn_root->info.u_op.type == LIBXSMM_MELTW_TYPE_UNARY_UNZIP)) {
    l_kernel_config.n_avail_gpr = l_kernel_config.n_avail_gpr - 1;
    l_gp_reg_mapping.gp_reg_offset = LIBXSMM_X86_GP_REG_R13;
  }

  jiting_queue = (libxsmm_matrix_eqn**) malloc(max_queue_size * sizeof(libxsmm_matrix_eqn*));

  /* Turn on fusion knobs given arch */
  if (io_generated_code->arch >= LIBXSMM_X86_AVX) {
    fusion_knobs.may_fuse_xgemm = 1;
  }
  libxsmm_generator_decompose_equation_tree_x86( eqn, jiting_queue, &queue_size, &fusion_knobs);

  /* Open asm */
  libxsmm_x86_instruction_open_stream_alt( io_generated_code, l_gp_reg_mapping.gp_reg_param_struct, 1 );

  /* Setup the stack */
  libxsmm_generator_matequation_setup_stack_frame( io_generated_code, i_mateqn_desc, &l_gp_reg_mapping, &l_kernel_config, eqn, strategy);

  for (eqn_tree_id = 0; eqn_tree_id < queue_size; eqn_tree_id++) {
    libxsmm_matrix_eqn *cur_eqn = jiting_queue[eqn_tree_id];
    libxsmm_meqn_descriptor copy_mateqn_desc = *i_mateqn_desc;

    /* Determine the output and precision of current equation tree to be JITed */
    if (eqn_tree_id == (queue_size - 1)) {
      libxsmm_generator_meqn_getval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_OUT_PTR, temp_reg);
    } else {
      if (cur_eqn->eqn_root->tmp.id >= 0) {
        libxsmm_generator_meqn_getaddr_stack_tmp_i( io_generated_code,  cur_eqn->eqn_root->tmp.id * l_kernel_config.tmp_size, temp_reg);
      } else {
        libxsmm_blasint arg_tmp_id = -1-cur_eqn->eqn_root->tmp.id;
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            l_kernel_config.alu_mov_instruction,
            l_gp_reg_mapping.gp_reg_param_struct,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            8,
            temp_reg,
            0 );
        libxsmm_x86_instruction_alu_mem( io_generated_code,
            l_kernel_config.alu_mov_instruction,
            temp_reg,
            LIBXSMM_X86_GP_REG_UNDEF, 0,
            arg_tmp_id*32,
            temp_reg,
            0 );
      }
      copy_mateqn_desc.datatype = LIBXSMM_CAST_UCHAR(cur_eqn->eqn_root->tmp.dtype);
    }

    libxsmm_x86_instruction_alu_mem( io_generated_code,
        l_kernel_config.alu_mov_instruction,
        l_gp_reg_mapping.gp_reg_param_struct,
        LIBXSMM_X86_GP_REG_UNDEF, 0,
        16,
        temp_reg,
        1 );

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
      libxsmm_generator_matequation_tmp_stack_scratch_avx_avx512_kernel(io_generated_code, &copy_mateqn_desc, &l_gp_reg_mapping, &l_kernel_config, &l_loop_label_tracker, cur_eqn);
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
        libxsmm_generator_meqn_getval_stack_var( io_generated_code, LIBXSMM_MEQN_STACK_VAR_CONST_9, l_gp_reg_mapping.gp_reg_offset);
        libxsmm_x86_instruction_alu_mem( io_generated_code, l_kernel_config.alu_mov_instruction, l_gp_reg_mapping.gp_reg_offset, LIBXSMM_X86_GP_REG_UNDEF, 0, 0, l_gp_reg_mapping.gp_reg_offset, 0 );
      }

      libxsmm_meqn_reoptimize(cur_eqn);
      memset(&(l_kernel_config.meltw_kernel_config), 0, sizeof(libxsmm_mateltwise_kernel_config));
#if 0
      printf("\nJITing tree with regblocks %d and ldo is %d\n", eqn_tree_id, copy_mateqn_desc.ldo);
      libxsmm_meqn_trv_dbg_print( cur_eqn->eqn_root, 0);
#endif
      l_kernel_config.meltw_kernel_config.vector_name = l_kernel_config.vector_name;
      libxsmm_generator_matequation_tmp_register_block_avx_avx512_kernel(io_generated_code, &copy_mateqn_desc, &l_gp_reg_mapping, &l_kernel_config, &l_loop_label_tracker, cur_eqn);
    }
  }

  /* Destroy stack frame */
  libxsmm_generator_matequation_destroy_stack_frame(  io_generated_code,  &l_kernel_config, &l_gp_reg_mapping, strategy);

  /* Close asm */
  libxsmm_x86_instruction_close_stream_alt( io_generated_code, 1 );

  free(jiting_queue);
}
