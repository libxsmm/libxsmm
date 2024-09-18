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
#include "generator_mateltwise_sse_avx_avx512.h"
#include "generator_mateltwise_transform_common.h"
#include "generator_mateltwise_unary_binary_avx_avx512.h"
#include "generator_mateltwise_reduce_avx_avx512.h"
#include "generator_mateltwise_misc_avx_avx512.h"
#include "generator_mateltwise_gather_scatter_avx_avx512.h"
#include "libxsmm_matrixeqn.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "generator_mateltwise_common.h"


LIBXSMM_API_INTERN
int libxsmm_generator_meltw_get_rbp_relative_offset( libxsmm_meltw_stack_var stack_var ) {
  switch ( (int)stack_var ) {
    case LIBXSMM_MELTW_STACK_VAR_OP_ARG_0:
      return -8;
    case LIBXSMM_MELTW_STACK_VAR_OP_ARG_1:
      return -16;
    case LIBXSMM_MELTW_STACK_VAR_OP_ARG_2:
      return -24;
    case LIBXSMM_MELTW_STACK_VAR_OP_ARG_3:
      return -32;
    case LIBXSMM_MELTW_STACK_VAR_INP0_PTR0:
      return -40;
    case LIBXSMM_MELTW_STACK_VAR_INP0_PTR1:
      return -48;
    case LIBXSMM_MELTW_STACK_VAR_INP0_PTR2:
      return -56;
    case LIBXSMM_MELTW_STACK_VAR_INP1_PTR0:
      return -64;
    case LIBXSMM_MELTW_STACK_VAR_INP1_PTR1:
      return -72;
    case LIBXSMM_MELTW_STACK_VAR_INP1_PTR2:
      return -80;
    case LIBXSMM_MELTW_STACK_VAR_INP2_PTR0:
      return -88;
    case LIBXSMM_MELTW_STACK_VAR_INP2_PTR1:
      return -96;
    case LIBXSMM_MELTW_STACK_VAR_INP2_PTR2:
      return -104;
    case LIBXSMM_MELTW_STACK_VAR_OUT_PTR0:
      return -112;
    case LIBXSMM_MELTW_STACK_VAR_OUT_PTR1:
      return -120;
    case LIBXSMM_MELTW_STACK_VAR_OUT_PTR2:
      return -128;
    case LIBXSMM_MELTW_STACK_VAR_SCRATCH_PTR:
      return -136;
    case LIBXSMM_MELTW_STACK_VAR_CONST_0:
      return -144;
    case LIBXSMM_MELTW_STACK_VAR_CONST_1:
      return -152;
    case LIBXSMM_MELTW_STACK_VAR_CONST_2:
      return -160;
    case LIBXSMM_MELTW_STACK_VAR_CONST_3:
      return -168;
    case LIBXSMM_MELTW_STACK_VAR_CONST_4:
      return -176;
    case LIBXSMM_MELTW_STACK_VAR_CONST_5:
      return -184;
    case LIBXSMM_MELTW_STACK_VAR_CONST_6:
      return -192;
    case LIBXSMM_MELTW_STACK_VAR_CONST_7:
      return -200;
    case LIBXSMM_MELTW_STACK_VAR_CONST_8:
      return -208;
    case LIBXSMM_MELTW_STACK_VAR_CONST_9:
      return -216;
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
  unsigned int skip_pushpops_callee_gp_reg  = ( (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) ||
                                                (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY)) ? 1 : 0;

  /* TODO: Determine if we want to save stuff to stack */
  unsigned int save_args_to_stack = 0;
  unsigned int allocate_scratch = ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS) > 0) ? 1 : 0;
  unsigned int use_aux_stack_vars = ((io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) &&
      ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV) )) ? 1 : 0;
  unsigned int use_stack_vars = ((save_args_to_stack > 0) || (allocate_scratch > 0) || (use_aux_stack_vars > 0)) ? 1 : 0;

  LIBXSMM_UNUSED(i_gp_reg_mapping);
  LIBXSMM_UNUSED(temp_reg);

  i_micro_kernel_config->skip_pushpops_callee_gp_reg = skip_pushpops_callee_gp_reg;
  i_micro_kernel_config->use_stack_vars              = use_stack_vars;

  if (use_stack_vars > 0) {
    libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBP );
    libxsmm_x86_instruction_alu_reg( io_generated_code, i_micro_kernel_config->alu_mov_instruction, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_RBP);
    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, 216 );
  }

  if ((io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY)) {
    if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV) ) {
      i_micro_kernel_config->rbp_offs_thres = libxsmm_generator_meltw_get_rbp_relative_offset(LIBXSMM_MELTW_STACK_VAR_CONST_0);
      i_micro_kernel_config->rbp_offs_signmask = libxsmm_generator_meltw_get_rbp_relative_offset(LIBXSMM_MELTW_STACK_VAR_CONST_1);
      i_micro_kernel_config->rbp_offs_absmask = libxsmm_generator_meltw_get_rbp_relative_offset(LIBXSMM_MELTW_STACK_VAR_CONST_2);
      i_micro_kernel_config->rbp_offs_scale = libxsmm_generator_meltw_get_rbp_relative_offset(LIBXSMM_MELTW_STACK_VAR_CONST_3);
      i_micro_kernel_config->rbp_offs_shifter = libxsmm_generator_meltw_get_rbp_relative_offset(LIBXSMM_MELTW_STACK_VAR_CONST_4);
      i_micro_kernel_config->rbp_offs_half = libxsmm_generator_meltw_get_rbp_relative_offset(LIBXSMM_MELTW_STACK_VAR_CONST_5);
    }
  }


  /* Exemplary usage of how to store args to stack if need be */
  if (save_args_to_stack > 0) {
  }

  if (allocate_scratch > 0) {
    /* TODO: Scratch size is kernel-dependent */
    unsigned int scratch_size = ( (i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS) > 0) ? 128 : 1024;

    /* make scratch size multiple of 64b */
    scratch_size = LIBXSMM_UP(scratch_size, 64);

    /* Now align RSP to 64 byte boundary */
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code, i_micro_kernel_config->alu_mov_instruction, temp_reg, 0xFFFFFFFFFFFFFFC0 );
    libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_ANDQ, temp_reg, LIBXSMM_X86_GP_REG_RSP);

    libxsmm_x86_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, LIBXSMM_X86_GP_REG_RSP, scratch_size );
    libxsmm_generator_meltw_setval_stack_var( io_generated_code, LIBXSMM_MELTW_STACK_VAR_SCRATCH_PTR, LIBXSMM_X86_GP_REG_RSP );
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
}

LIBXSMM_API_INTERN
void libxsmm_generator_meltw_destroy_stack_frame( libxsmm_generated_code*            io_generated_code,
    const libxsmm_meltw_descriptor*     i_mateltwise_desc,
    const libxsmm_mateltwise_kernel_config*  i_micro_kernel_config ) {

  LIBXSMM_UNUSED(i_mateltwise_desc);
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
  LIBXSMM_UNUSED(i_kernel_config);
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_m_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_m_loop, 1);
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_footer_m_loop( libxsmm_generated_code*                    io_generated_code,
                                              libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                              const libxsmm_mateltwise_kernel_config*       i_kernel_config,
                                              const unsigned int                            i_gp_reg_m_loop,
                                              const unsigned int                            i_m ) {
  LIBXSMM_UNUSED(i_kernel_config);
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_CMPQ, i_gp_reg_m_loop, i_m );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, LIBXSMM_X86_INSTR_JL, io_loop_label_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_header_n_loop( libxsmm_generated_code*                io_generated_code,
                                              libxsmm_loop_label_tracker*               io_loop_label_tracker,
                                              const libxsmm_mateltwise_kernel_config*   i_kernel_config,
                                              const unsigned int                        i_gp_reg_n_loop ) {
  LIBXSMM_UNUSED(i_kernel_config);
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, i_gp_reg_n_loop, 0);
  libxsmm_x86_instruction_register_jump_back_label( io_generated_code, io_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, i_gp_reg_n_loop, 1);
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_footer_n_loop( libxsmm_generated_code*                    io_generated_code,
                                              libxsmm_loop_label_tracker*                   io_loop_label_tracker,
                                              const libxsmm_mateltwise_kernel_config*       i_kernel_config,
                                              const unsigned int                            i_gp_reg_n_loop,
                                              const unsigned int                            i_n ) {
  LIBXSMM_UNUSED(i_kernel_config);
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_CMPQ, i_gp_reg_n_loop, i_n );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, LIBXSMM_X86_INSTR_JL, io_loop_label_tracker );
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
unsigned int libxsmm_generator_mateltwise_select_store_instruction( const libxsmm_meltw_descriptor*   i_mateltwise_desc,
                                                                    const unsigned int i_vlen_bytes,
                                                                    const unsigned int i_dt_width,
                                                                    const unsigned int i_instr_st,
                                                                    const unsigned int i_instr_nts ) {
  unsigned int l_instr;

  if ( ( ( i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY )  && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_NTS_HINT)  == LIBXSMM_MELTW_FLAG_UNARY_NTS_HINT  ) ) ||
       ( ( i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY ) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_NTS_HINT) == LIBXSMM_MELTW_FLAG_BINARY_NTS_HINT ) )    ) {
    if ( (i_mateltwise_desc->m   % (i_vlen_bytes / i_dt_width ) == 0) &&
         (i_mateltwise_desc->ldo % (i_vlen_bytes / i_dt_width ) == 0)  ) {
      l_instr = i_instr_nts;
    } else {
      l_instr = i_instr_st;
    }
  } else {
    l_instr = i_instr_st;
  }

  return l_instr;
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_update_micro_kernel_config_dtype_aluinstr( libxsmm_generated_code*           io_generated_code,
                                                                           libxsmm_mateltwise_kernel_config* io_micro_kernel_config,
                                                                           const libxsmm_meltw_descriptor*   i_mateltwise_desc) {
  libxsmm_datatype dtype_in0 = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0);
  libxsmm_datatype dtype_in1 = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1);
  libxsmm_datatype dtype_out = (libxsmm_datatype)libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT);

  if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_SKX) && (io_generated_code->arch <= LIBXSMM_X86_ALLFEAT) ) {
    unsigned int l_vlen_bytes = 64;
    io_micro_kernel_config->instruction_set = io_generated_code->arch;
    io_micro_kernel_config->vector_reg_count = 16;
    /* Configure input specific microkernel options */
    if ( (LIBXSMM_DATATYPE_F64 == dtype_in0) || (LIBXSMM_DATATYPE_I64 == dtype_in0) ) {
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVUPD;
    } else if ( (LIBXSMM_DATATYPE_F32 == dtype_in0) || (LIBXSMM_DATATYPE_I32 == dtype_in0) || (LIBXSMM_DATATYPE_U32 == dtype_in0)  ) {
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVUPS;
    } else if ( (LIBXSMM_DATATYPE_BF16 == dtype_in0) || (LIBXSMM_DATATYPE_I16 == dtype_in0) || (LIBXSMM_DATATYPE_U16 == dtype_in0)  || (LIBXSMM_DATATYPE_F16 == dtype_in0) ) {
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( LIBXSMM_DATATYPE_F16 == dtype_in0 ) {
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( (LIBXSMM_DATATYPE_BF8 == dtype_in0) || (LIBXSMM_DATATYPE_HF8 == dtype_in0) || (LIBXSMM_DATATYPE_I8 == dtype_in0 )) {
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU8;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY ) {
      /* Configure input specific microkernel options */
      if ( (LIBXSMM_DATATYPE_F64 == dtype_in1) || (LIBXSMM_DATATYPE_I64 == dtype_in1) ) {
        io_micro_kernel_config->datatype_size_in1 = 8;
        io_micro_kernel_config->vmove_instruction_in1 = LIBXSMM_X86_INSTR_VMOVUPD;
      } else if ( (LIBXSMM_DATATYPE_F32 == dtype_in1) || (LIBXSMM_DATATYPE_I32 == dtype_in1) || (LIBXSMM_DATATYPE_U32 == dtype_in1) ) {
        io_micro_kernel_config->datatype_size_in1 = 4;
        io_micro_kernel_config->vmove_instruction_in1 = LIBXSMM_X86_INSTR_VMOVUPS;
      } else if ( (LIBXSMM_DATATYPE_BF16 == dtype_in1) || (LIBXSMM_DATATYPE_I16 == dtype_in1) || (LIBXSMM_DATATYPE_U16 == dtype_in1) || (LIBXSMM_DATATYPE_F16 == dtype_in1) ) {
        io_micro_kernel_config->datatype_size_in1 = 2;
        io_micro_kernel_config->vmove_instruction_in1 = LIBXSMM_X86_INSTR_VMOVDQU16;
      } else if ( LIBXSMM_DATATYPE_F16 == dtype_in1 ) {
        io_micro_kernel_config->datatype_size_in1 = 2;
        io_micro_kernel_config->vmove_instruction_in1 = LIBXSMM_X86_INSTR_VMOVDQU16;
      } else if ( (LIBXSMM_DATATYPE_BF8 == dtype_in1) || (LIBXSMM_DATATYPE_HF8 == dtype_in1) || (LIBXSMM_DATATYPE_I8 == dtype_in1) ) {
        io_micro_kernel_config->datatype_size_in1 = 1;
        io_micro_kernel_config->vmove_instruction_in1 = LIBXSMM_X86_INSTR_VMOVDQU8;
      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY ) {
      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_TERNARY_SELECT) {
        io_micro_kernel_config->datatype_size_in2 = 1;
        io_micro_kernel_config->vmove_instruction_in2 = LIBXSMM_X86_INSTR_VMOVDQU8;
      } else {
        /* Configure input specific microkernel options */
        if ( (LIBXSMM_DATATYPE_F64 == dtype_out) || (LIBXSMM_DATATYPE_I64 == dtype_out) ) {
          io_micro_kernel_config->datatype_size_in2 = 8;
          io_micro_kernel_config->vmove_instruction_in2 = LIBXSMM_X86_INSTR_VMOVUPD;
        } else if ( (LIBXSMM_DATATYPE_F32 == dtype_out) || (LIBXSMM_DATATYPE_I32 == dtype_out) || (LIBXSMM_DATATYPE_U32 == dtype_out) ) {
          io_micro_kernel_config->datatype_size_in2 = 4;
          io_micro_kernel_config->vmove_instruction_in2 = LIBXSMM_X86_INSTR_VMOVUPS;
        } else if ( (LIBXSMM_DATATYPE_BF16 == dtype_out) || (LIBXSMM_DATATYPE_I16 == dtype_out) || (LIBXSMM_DATATYPE_U16 == dtype_out)  || (LIBXSMM_DATATYPE_F16 == dtype_out) ) {
          io_micro_kernel_config->datatype_size_in2 = 2;
          io_micro_kernel_config->vmove_instruction_in2 = LIBXSMM_X86_INSTR_VMOVDQU16;
        } else if ( LIBXSMM_DATATYPE_F16 == dtype_out ) {
          io_micro_kernel_config->datatype_size_in2 = 2;
          io_micro_kernel_config->vmove_instruction_in2 = LIBXSMM_X86_INSTR_VMOVDQU16;
        } else if ( (LIBXSMM_DATATYPE_BF8 == dtype_out) || (LIBXSMM_DATATYPE_I8 == dtype_out) ) {
          io_micro_kernel_config->datatype_size_in2 = 1;
          io_micro_kernel_config->vmove_instruction_in2 = LIBXSMM_X86_INSTR_VMOVDQU8;
        } else {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
          return;
        }
      }
    }

    /* Configure output specific microkernel options */
    if ( (LIBXSMM_DATATYPE_F64 == dtype_out) || (LIBXSMM_DATATYPE_I64 == dtype_out) ) {
      io_micro_kernel_config->datatype_size_out = 8;
      io_micro_kernel_config->vmove_instruction_out = libxsmm_generator_mateltwise_select_store_instruction( i_mateltwise_desc, l_vlen_bytes, io_micro_kernel_config->datatype_size_out,
                                                                                                             LIBXSMM_X86_INSTR_VMOVUPD, LIBXSMM_X86_INSTR_VMOVNTPD );
    } else if ( (LIBXSMM_DATATYPE_F32 == dtype_out) || (LIBXSMM_DATATYPE_I32 == dtype_out) || (LIBXSMM_DATATYPE_U32 == dtype_out) ) {
      io_micro_kernel_config->datatype_size_out = 4;
      io_micro_kernel_config->vmove_instruction_out = libxsmm_generator_mateltwise_select_store_instruction( i_mateltwise_desc, l_vlen_bytes, io_micro_kernel_config->datatype_size_out,
                                                                                                             LIBXSMM_X86_INSTR_VMOVUPS, LIBXSMM_X86_INSTR_VMOVNTPS );
    } else if ( (LIBXSMM_DATATYPE_BF16 == dtype_out) || (LIBXSMM_DATATYPE_I16 == dtype_out) || (LIBXSMM_DATATYPE_U16 == dtype_out) || (LIBXSMM_DATATYPE_F16 == dtype_out) ) {
      io_micro_kernel_config->datatype_size_out = 2;
      io_micro_kernel_config->vmove_instruction_out = libxsmm_generator_mateltwise_select_store_instruction( i_mateltwise_desc, l_vlen_bytes, io_micro_kernel_config->datatype_size_out,
                                                                                                             LIBXSMM_X86_INSTR_VMOVDQU16, LIBXSMM_X86_INSTR_VMOVNTPS );
    } else if ( (LIBXSMM_DATATYPE_BF8 == dtype_out) || (LIBXSMM_DATATYPE_HF8 == dtype_out) || (LIBXSMM_DATATYPE_I8 == dtype_out) || (LIBXSMM_DATATYPE_IMPLICIT == dtype_out) ) {
      io_micro_kernel_config->datatype_size_out = 1;
      io_micro_kernel_config->vmove_instruction_out = libxsmm_generator_mateltwise_select_store_instruction( i_mateltwise_desc, l_vlen_bytes, io_micro_kernel_config->datatype_size_out,
                                                                                                             LIBXSMM_X86_INSTR_VMOVDQU8, LIBXSMM_X86_INSTR_VMOVNTPS );
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
  } else if ( (io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256_SKX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_SKX) ) {
    unsigned int l_vlen_bytes = 32;
    io_micro_kernel_config->instruction_set = io_generated_code->arch;
    io_micro_kernel_config->vector_reg_count = 32;
    /* Configure input specific microkernel options */
    if ( (LIBXSMM_DATATYPE_F64 == dtype_in0) || (LIBXSMM_DATATYPE_I64 == dtype_in0) ) {
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVUPD;
    } else if ( (LIBXSMM_DATATYPE_F32 == dtype_in0) || (LIBXSMM_DATATYPE_I32 == dtype_in0) || (LIBXSMM_DATATYPE_U32 == dtype_in0)  ) {
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVUPS;
    } else if ( (LIBXSMM_DATATYPE_BF16 == dtype_in0) || (LIBXSMM_DATATYPE_I16 == dtype_in0) || (LIBXSMM_DATATYPE_U16 == dtype_in0) || (LIBXSMM_DATATYPE_F16 == dtype_in0) ) {
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( (LIBXSMM_DATATYPE_BF8 == dtype_in0) || (LIBXSMM_DATATYPE_HF8 == dtype_in0) ||  (LIBXSMM_DATATYPE_I8 == dtype_in0) ) {
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU8;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY ) {
      /* Configure input specific microkernel options */
      if ( (LIBXSMM_DATATYPE_F64 == dtype_in1) || (LIBXSMM_DATATYPE_I64 == dtype_in1) ) {
        io_micro_kernel_config->datatype_size_in1 = 8;
        io_micro_kernel_config->vmove_instruction_in1 = LIBXSMM_X86_INSTR_VMOVUPD;
      } else if ( (LIBXSMM_DATATYPE_F32 == dtype_in1) || (LIBXSMM_DATATYPE_I32 == dtype_in1) || (LIBXSMM_DATATYPE_U32 == dtype_in1) ) {
        io_micro_kernel_config->datatype_size_in1 = 4;
        io_micro_kernel_config->vmove_instruction_in1 = LIBXSMM_X86_INSTR_VMOVUPS;
      } else if ( (LIBXSMM_DATATYPE_BF16 == dtype_in1) || (LIBXSMM_DATATYPE_I16 == dtype_in1) || (LIBXSMM_DATATYPE_U16 == dtype_in1)|| (LIBXSMM_DATATYPE_F16 == dtype_in1) ) {
        io_micro_kernel_config->datatype_size_in1 = 2;
        io_micro_kernel_config->vmove_instruction_in1 = LIBXSMM_X86_INSTR_VMOVDQU16;
      } else if (( LIBXSMM_DATATYPE_BF8 == dtype_in1) || ( LIBXSMM_DATATYPE_HF8 == dtype_in1) || (LIBXSMM_DATATYPE_I8 == dtype_in1) ) {
        io_micro_kernel_config->datatype_size_in1 = 1;
        io_micro_kernel_config->vmove_instruction_in1 = LIBXSMM_X86_INSTR_VMOVDQU8;
      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY ) {
      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_TERNARY_SELECT) {
        io_micro_kernel_config->datatype_size_in2 = 1;
        io_micro_kernel_config->vmove_instruction_in2 = LIBXSMM_X86_INSTR_VMOVDQU8;
      } else {
        /* Configure input specific microkernel options */
        if ( (LIBXSMM_DATATYPE_F64 == dtype_out) || (LIBXSMM_DATATYPE_I64 == dtype_out) ) {
          io_micro_kernel_config->datatype_size_in2 = 8;
          io_micro_kernel_config->vmove_instruction_in2 = LIBXSMM_X86_INSTR_VMOVUPD;
        } else if ( (LIBXSMM_DATATYPE_F32 == dtype_out) || (LIBXSMM_DATATYPE_I32 == dtype_out) || (LIBXSMM_DATATYPE_U32 == dtype_out) ) {
          io_micro_kernel_config->datatype_size_in2 = 4;
          io_micro_kernel_config->vmove_instruction_in2 = LIBXSMM_X86_INSTR_VMOVUPS;
        } else if ( (LIBXSMM_DATATYPE_BF16 == dtype_out) || (LIBXSMM_DATATYPE_I16 == dtype_out) || (LIBXSMM_DATATYPE_U16 == dtype_out) || (LIBXSMM_DATATYPE_F16 == dtype_out) ) {
          io_micro_kernel_config->datatype_size_in2 = 2;
          io_micro_kernel_config->vmove_instruction_in2 = LIBXSMM_X86_INSTR_VMOVDQU16;
        } else if (( LIBXSMM_DATATYPE_BF8 == dtype_out ) || ( LIBXSMM_DATATYPE_HF8 == dtype_out ) ||  ( LIBXSMM_DATATYPE_I8 == dtype_out) ) {
          io_micro_kernel_config->datatype_size_in2 = 1;
          io_micro_kernel_config->vmove_instruction_in2 = LIBXSMM_X86_INSTR_VMOVDQU8;
        } else {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
          return;
        }
      }
    }
    /* Configure output specific microkernel options */
    if ( (LIBXSMM_DATATYPE_F64 == dtype_out) || (LIBXSMM_DATATYPE_I64 == dtype_out) ) {
      io_micro_kernel_config->datatype_size_out = 8;
      io_micro_kernel_config->vmove_instruction_out = libxsmm_generator_mateltwise_select_store_instruction( i_mateltwise_desc, l_vlen_bytes, io_micro_kernel_config->datatype_size_out,
                                                                                                             LIBXSMM_X86_INSTR_VMOVUPD, LIBXSMM_X86_INSTR_VMOVNTPD );
    } else if ( (LIBXSMM_DATATYPE_F32 == dtype_out) || (LIBXSMM_DATATYPE_I32 == dtype_out) || (LIBXSMM_DATATYPE_U32 == dtype_out) ) {
      io_micro_kernel_config->datatype_size_out = 4;
      io_micro_kernel_config->vmove_instruction_out = libxsmm_generator_mateltwise_select_store_instruction( i_mateltwise_desc, l_vlen_bytes, io_micro_kernel_config->datatype_size_out,
                                                                                                             LIBXSMM_X86_INSTR_VMOVUPS, LIBXSMM_X86_INSTR_VMOVNTPS );
    } else if ( (LIBXSMM_DATATYPE_BF16 == dtype_out) || (LIBXSMM_DATATYPE_I16 == dtype_out) || (LIBXSMM_DATATYPE_U16 == dtype_out) || (LIBXSMM_DATATYPE_F16 == dtype_out) ) {
      io_micro_kernel_config->datatype_size_out = 2;
      io_micro_kernel_config->vmove_instruction_out = libxsmm_generator_mateltwise_select_store_instruction( i_mateltwise_desc, l_vlen_bytes, io_micro_kernel_config->datatype_size_out,
                                                                                                             LIBXSMM_X86_INSTR_VMOVDQU16, LIBXSMM_X86_INSTR_VMOVNTPS );
    } else if ( (LIBXSMM_DATATYPE_BF8 == dtype_out) || (LIBXSMM_DATATYPE_HF8 == dtype_out) || (LIBXSMM_DATATYPE_I8 == dtype_out) || (LIBXSMM_DATATYPE_IMPLICIT == dtype_out) ) {
      io_micro_kernel_config->datatype_size_out = 1;
      io_micro_kernel_config->vmove_instruction_out = libxsmm_generator_mateltwise_select_store_instruction( i_mateltwise_desc, l_vlen_bytes, io_micro_kernel_config->datatype_size_out,
                                                                                                             LIBXSMM_X86_INSTR_VMOVDQU8, LIBXSMM_X86_INSTR_VMOVNTPS );
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
  } else if ( (io_generated_code->arch >= LIBXSMM_X86_AVX) && (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) ) {
    unsigned int l_vlen_bytes = 32;
    io_micro_kernel_config->instruction_set = io_generated_code->arch;
    io_micro_kernel_config->vector_reg_count = 16;
    /* Configure input specific microkernel options */
    if ( (LIBXSMM_DATATYPE_F64 == dtype_in0) || (LIBXSMM_DATATYPE_I64 == dtype_in0) ) {
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVUPD;
    } else if ( (LIBXSMM_DATATYPE_F32 == dtype_in0) || (LIBXSMM_DATATYPE_I32 == dtype_in0) || (LIBXSMM_DATATYPE_U32 == dtype_in0)  ) {
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVUPS;
    } else if ( (LIBXSMM_DATATYPE_BF16 == dtype_in0) || (LIBXSMM_DATATYPE_I16 == dtype_in0) || (LIBXSMM_DATATYPE_U16 == dtype_in0) || (LIBXSMM_DATATYPE_F16 == dtype_in0) ) {
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU16;
    } else if ( (LIBXSMM_DATATYPE_BF8 == dtype_in0) || (LIBXSMM_DATATYPE_HF8 == dtype_in0) || (LIBXSMM_DATATYPE_I8 == dtype_in0) ) {
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_VMOVDQU8;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY ) {
      /* Configure input specific microkernel options */
      if ( (LIBXSMM_DATATYPE_F64 == dtype_in1) || (LIBXSMM_DATATYPE_I64 == dtype_in1) ) {
        io_micro_kernel_config->datatype_size_in1 = 8;
        io_micro_kernel_config->vmove_instruction_in1 = LIBXSMM_X86_INSTR_VMOVUPD;
      } else if ( (LIBXSMM_DATATYPE_F32 == dtype_in1) || (LIBXSMM_DATATYPE_I32 == dtype_in1) || (LIBXSMM_DATATYPE_U32 == dtype_in1) ) {
        io_micro_kernel_config->datatype_size_in1 = 4;
        io_micro_kernel_config->vmove_instruction_in1 = LIBXSMM_X86_INSTR_VMOVUPS;
      } else if ( (LIBXSMM_DATATYPE_BF16 == dtype_in1) || (LIBXSMM_DATATYPE_I16 == dtype_in1) || (LIBXSMM_DATATYPE_U16 == dtype_in1)  || (LIBXSMM_DATATYPE_F16 == dtype_in1) ) {
        io_micro_kernel_config->datatype_size_in1 = 2;
        io_micro_kernel_config->vmove_instruction_in1 = LIBXSMM_X86_INSTR_VMOVDQU16;
      } else if ( (LIBXSMM_DATATYPE_BF8 == dtype_in1) || (LIBXSMM_DATATYPE_HF8 == dtype_in1) || (LIBXSMM_DATATYPE_I8 == dtype_in1) ) {
        io_micro_kernel_config->datatype_size_in1 = 1;
        io_micro_kernel_config->vmove_instruction_in1 = LIBXSMM_X86_INSTR_UNDEF;
      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY ) {
      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_TERNARY_SELECT) {
        io_micro_kernel_config->datatype_size_in2 = 1;
        io_micro_kernel_config->vmove_instruction_in2 = LIBXSMM_X86_INSTR_VMOVDQU8;
      } else {
        /* Configure input specific microkernel options */
        if ( (LIBXSMM_DATATYPE_F64 == dtype_out) || (LIBXSMM_DATATYPE_I64 == dtype_out) ) {
          io_micro_kernel_config->datatype_size_in2 = 8;
          io_micro_kernel_config->vmove_instruction_in2 = LIBXSMM_X86_INSTR_VMOVUPD;
        } else if ( (LIBXSMM_DATATYPE_F32 == dtype_out) || (LIBXSMM_DATATYPE_I32 == dtype_out) || (LIBXSMM_DATATYPE_U32 == dtype_out)) {
          io_micro_kernel_config->datatype_size_in2 = 4;
          io_micro_kernel_config->vmove_instruction_in2 = LIBXSMM_X86_INSTR_VMOVUPS;
        } else if ( (LIBXSMM_DATATYPE_BF16 == dtype_out) || (LIBXSMM_DATATYPE_I16 == dtype_out) || (LIBXSMM_DATATYPE_U16 == dtype_out)  || (LIBXSMM_DATATYPE_F16 == dtype_out) ) {
          io_micro_kernel_config->datatype_size_in2 = 2;
          io_micro_kernel_config->vmove_instruction_in2 = LIBXSMM_X86_INSTR_VMOVDQU16;
        } else if ( (LIBXSMM_DATATYPE_BF8 == dtype_out) || (LIBXSMM_DATATYPE_HF8 == dtype_out) || (LIBXSMM_DATATYPE_I8 == dtype_out)) {
          io_micro_kernel_config->datatype_size_in2 = 1;
          io_micro_kernel_config->vmove_instruction_in2 = LIBXSMM_X86_INSTR_UNDEF;
        } else {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
          return;
        }
      }
    }

    /* Configure output specific microkernel options */
    if ( (LIBXSMM_DATATYPE_F64 == dtype_out) || (LIBXSMM_DATATYPE_I64 == dtype_out) ) {
      io_micro_kernel_config->datatype_size_out = 8;
      io_micro_kernel_config->vmove_instruction_out = libxsmm_generator_mateltwise_select_store_instruction( i_mateltwise_desc, l_vlen_bytes, io_micro_kernel_config->datatype_size_out,
                                                                                                             LIBXSMM_X86_INSTR_VMOVUPD, LIBXSMM_X86_INSTR_VMOVNTPD );
    } else if ( (LIBXSMM_DATATYPE_F32 == dtype_out) || (LIBXSMM_DATATYPE_I32 == dtype_out) || (LIBXSMM_DATATYPE_U32 == dtype_out) ) {
      io_micro_kernel_config->datatype_size_out = 4;
      io_micro_kernel_config->vmove_instruction_out = libxsmm_generator_mateltwise_select_store_instruction( i_mateltwise_desc, l_vlen_bytes, io_micro_kernel_config->datatype_size_out,
                                                                                                             LIBXSMM_X86_INSTR_VMOVUPS, LIBXSMM_X86_INSTR_VMOVNTPS );
    } else if ( (LIBXSMM_DATATYPE_BF16 == dtype_out) || (LIBXSMM_DATATYPE_I16 == dtype_out) || (LIBXSMM_DATATYPE_U16 == dtype_out) || (LIBXSMM_DATATYPE_F16 == dtype_out) ) {
      io_micro_kernel_config->datatype_size_out = 2;
      io_micro_kernel_config->vmove_instruction_out = libxsmm_generator_mateltwise_select_store_instruction( i_mateltwise_desc, l_vlen_bytes, io_micro_kernel_config->datatype_size_out,
                                                                                                             LIBXSMM_X86_INSTR_VMOVDQU16, LIBXSMM_X86_INSTR_VMOVNTPS );
    } else if ( (LIBXSMM_DATATYPE_BF8 == dtype_out) || (LIBXSMM_DATATYPE_HF8 == dtype_out) || (LIBXSMM_DATATYPE_I8 == dtype_out) || (LIBXSMM_DATATYPE_IMPLICIT == dtype_out) ) {
      io_micro_kernel_config->datatype_size_out = 1;
      io_micro_kernel_config->vmove_instruction_out = libxsmm_generator_mateltwise_select_store_instruction( i_mateltwise_desc, l_vlen_bytes, io_micro_kernel_config->datatype_size_out,
                                                                                                             LIBXSMM_X86_INSTR_VMOVDQU8, LIBXSMM_X86_INSTR_VMOVDQU8 );
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
    if ( (LIBXSMM_DATATYPE_F64 == dtype_in0) || (LIBXSMM_DATATYPE_I64 == dtype_in0) ) {
      io_micro_kernel_config->datatype_size_in = 8;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_MOVUPD;
    } else if ( (LIBXSMM_DATATYPE_F32 == dtype_in0) || (LIBXSMM_DATATYPE_I32 == dtype_in0) || (LIBXSMM_DATATYPE_U32 == dtype_in0) ) {
      io_micro_kernel_config->datatype_size_in = 4;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_MOVUPS;
    } else if ( (LIBXSMM_DATATYPE_BF16 == dtype_in0) || (LIBXSMM_DATATYPE_I16 == dtype_in0) || (LIBXSMM_DATATYPE_U16 == dtype_in0) || (LIBXSMM_DATATYPE_F16 == dtype_in0) ) {
      io_micro_kernel_config->datatype_size_in = 2;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_UNDEF;
    } else if ( (LIBXSMM_DATATYPE_BF8 == dtype_in0) || (LIBXSMM_DATATYPE_HF8 == dtype_in0) || (LIBXSMM_DATATYPE_I8 == dtype_in0) ) {
      io_micro_kernel_config->datatype_size_in = 1;
      io_micro_kernel_config->vmove_instruction_in = LIBXSMM_X86_INSTR_UNDEF;
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY || i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY ) {
      /* Configure input specific microkernel options */
      if ( (LIBXSMM_DATATYPE_F64 == dtype_in1) || (LIBXSMM_DATATYPE_I64 == dtype_in1) ) {
        io_micro_kernel_config->datatype_size_in1 = 8;
        io_micro_kernel_config->vmove_instruction_in1 = LIBXSMM_X86_INSTR_VMOVUPD;
      } else if ( (LIBXSMM_DATATYPE_F32 == dtype_in1) || (LIBXSMM_DATATYPE_I32 == dtype_in1) || (LIBXSMM_DATATYPE_U32 == dtype_in1) ) {
        io_micro_kernel_config->datatype_size_in1 = 4;
        io_micro_kernel_config->vmove_instruction_in1 = LIBXSMM_X86_INSTR_VMOVUPS;
      } else if ( (LIBXSMM_DATATYPE_BF16 == dtype_in1) || (LIBXSMM_DATATYPE_I16 == dtype_in1) || (LIBXSMM_DATATYPE_U16 == dtype_in1) || (LIBXSMM_DATATYPE_F16 == dtype_in1) ) {
        io_micro_kernel_config->datatype_size_in1 = 2;
        io_micro_kernel_config->vmove_instruction_in1 = LIBXSMM_X86_INSTR_UNDEF;
      } else if ( (LIBXSMM_DATATYPE_BF8 == dtype_in1) || (LIBXSMM_DATATYPE_HF8 == dtype_in1) || (LIBXSMM_DATATYPE_I8 == dtype_in1) ) {
        io_micro_kernel_config->datatype_size_in1 = 1;
        io_micro_kernel_config->vmove_instruction_in1 = LIBXSMM_X86_INSTR_UNDEF;
      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    }

    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY ) {
      /* Configure input specific microkernel options */
      if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_TERNARY_SELECT) {
        io_micro_kernel_config->datatype_size_in2 = 1;
        io_micro_kernel_config->vmove_instruction_in2 = LIBXSMM_X86_INSTR_VMOVDQU8;
      } else {
        if ( (LIBXSMM_DATATYPE_F64 == dtype_out) || (LIBXSMM_DATATYPE_I64 == dtype_out) ) {
          io_micro_kernel_config->datatype_size_in2 = 8;
          io_micro_kernel_config->vmove_instruction_in2 = LIBXSMM_X86_INSTR_VMOVUPD;
        } else if ( (LIBXSMM_DATATYPE_F32 == dtype_out) || (LIBXSMM_DATATYPE_I32 == dtype_out) || (LIBXSMM_DATATYPE_U32 == dtype_out) ) {
          io_micro_kernel_config->datatype_size_in2 = 4;
          io_micro_kernel_config->vmove_instruction_in2 = LIBXSMM_X86_INSTR_VMOVUPS;
        } else if ( (LIBXSMM_DATATYPE_BF16 == dtype_out) || (LIBXSMM_DATATYPE_I16 == dtype_out) || (LIBXSMM_DATATYPE_U16 == dtype_out) || (LIBXSMM_DATATYPE_F16 == dtype_out) ) {
          io_micro_kernel_config->datatype_size_in2 = 2;
          io_micro_kernel_config->vmove_instruction_in2 = LIBXSMM_X86_INSTR_UNDEF;
        } else if ( (LIBXSMM_DATATYPE_BF8 == dtype_out) || (LIBXSMM_DATATYPE_HF8 == dtype_out) || (LIBXSMM_DATATYPE_I8 == dtype_out) ) {
          io_micro_kernel_config->datatype_size_in2 = 1;
          io_micro_kernel_config->vmove_instruction_in2 = LIBXSMM_X86_INSTR_UNDEF;
        } else {
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
          return;
        }
      }
    }

    /* Configure output specific microkernel options */
    if ( (LIBXSMM_DATATYPE_F64 == dtype_out) || (LIBXSMM_DATATYPE_I64 == dtype_out) ) {
      io_micro_kernel_config->datatype_size_out = 8;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_MOVUPD;
    } else if ( (LIBXSMM_DATATYPE_F32 == dtype_out) || (LIBXSMM_DATATYPE_I32 == dtype_out) || (LIBXSMM_DATATYPE_U32 == dtype_out)) {
      io_micro_kernel_config->datatype_size_out = 4;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_MOVUPS;
    } else if ( (LIBXSMM_DATATYPE_BF16 == dtype_out) || (LIBXSMM_DATATYPE_I16 == dtype_out) || (LIBXSMM_DATATYPE_U16 == dtype_out) || (LIBXSMM_DATATYPE_F16 == dtype_out) ) {
      io_micro_kernel_config->datatype_size_out = 2;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_UNDEF;
    } else if ( (LIBXSMM_DATATYPE_BF8 == dtype_out) || ( LIBXSMM_DATATYPE_HF8 == dtype_out ) ||  (LIBXSMM_DATATYPE_I8 == dtype_out) || (LIBXSMM_DATATYPE_IMPLICIT == dtype_out) ) {
      io_micro_kernel_config->datatype_size_out = 1;
      io_micro_kernel_config->vmove_instruction_out = LIBXSMM_X86_INSTR_UNDEF;
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
  libxsmm_generator_mateltwise_update_micro_kernel_config_dtype_aluinstr( io_generated_code, io_micro_kernel_config, i_mateltwise_desc);
}

LIBXSMM_API_INTERN
libxsmm_blasint libxsmm_generator_mateltwise_x86_valid_arch_precision( libxsmm_generated_code*           io_generated_code,
                                                                       const libxsmm_meltw_descriptor*   i_mateltwise_desc) {
  libxsmm_blasint is_valid_arch_prec = 1;
  unsigned int is_transform_tpp = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY )  &&
                 ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2)     ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT)     ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T)   ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2T)    ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD) ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4)     ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_NORM)     ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI2)    ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T)   ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4T)    ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2T_TO_NORM)    ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4T_TO_NORM)    ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD) ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8)     ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8_TO_VNNI8T)   ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8T)    ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8_PAD) ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8T_TO_NORM)    ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2)         ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2)         ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2)        ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD4)         ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD4)         ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD4))       ) ? 1 : 0;
  unsigned int is_unary_simple_tpp = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY )  &&
                 ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_IDENTITY)     ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_XOR)     ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_X2)   ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SQRT)    ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_NEGATE) ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_INC)     ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL)     ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT)    ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD)   ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD)    ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD) ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DUMP)         ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ABSMAX) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MIN)   )       ) ? 1 : 0;
  unsigned int is_gather_scatter_tpp = ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY )  &&
                                       ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER)     ||
                                        (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SCATTER)) ) ? 1 : 0;
  unsigned int has_inp_or_out_fp8 = (libxsmm_generator_mateltwise_involves_prec(i_mateltwise_desc, LIBXSMM_DATATYPE_BF8) > 0 || libxsmm_generator_mateltwise_involves_prec(i_mateltwise_desc, LIBXSMM_DATATYPE_HF8) > 0) ? 1 : 0;
  unsigned int has_inp_or_out_fp64 = (libxsmm_generator_mateltwise_involves_prec(i_mateltwise_desc, LIBXSMM_DATATYPE_F64) > 0) ? 1 : 0;
  unsigned int has_all_inp_and_out_fp64 = (libxsmm_generator_mateltwise_all_inp_comp_out_prec(i_mateltwise_desc, LIBXSMM_DATATYPE_F64) > 0) ? 1 : 0;

  if ((is_transform_tpp == 0) && (is_gather_scatter_tpp == 0) &&
      !((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_XOR)) &&
      !((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU)) &&
      !((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_RELU_INV)) &&
      !((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR )) &&
      !((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_IDENTITY))) {
    if ( LIBXSMM_DATATYPE_BF16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
      LIBXSMM_DATATYPE_I16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
      LIBXSMM_DATATYPE_F16 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
      LIBXSMM_DATATYPE_HF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
      LIBXSMM_DATATYPE_BF8 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
      LIBXSMM_DATATYPE_I8  == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
      LIBXSMM_DATATYPE_I32 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
      LIBXSMM_DATATYPE_I64 == libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ) {
      is_valid_arch_prec = 0;
    }
  }

  if ((io_generated_code->arch < LIBXSMM_X86_AVX2) && (is_transform_tpp == 0)) {
    is_valid_arch_prec = 0;
  }
  if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV)) && (io_generated_code->arch < LIBXSMM_X86_AVX2)) {
    is_valid_arch_prec = 0;
  }
  if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_UNZIP) && (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX)) {
    is_valid_arch_prec = 0;
  }
  if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_ZIP) && (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) ) {
    is_valid_arch_prec = 0;
  }

  if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ( libxsmm_generator_mateltwise_is_binary_cmp_op(i_mateltwise_desc) > 0 ) &&
       (LIBXSMM_DATATYPE_F32 != libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) || LIBXSMM_DATATYPE_IMPLICIT != libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT) || ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BITMASK_2BYTEMULT) == 0)) ) {
    is_valid_arch_prec = 0;
  }

  if ( (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X2) ||
                                                                            (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3)    ) &&
       (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) ) {
    is_valid_arch_prec = 0;
  }
  if ((has_inp_or_out_fp8 > 0) && (io_generated_code->arch < LIBXSMM_X86_AVX512_VL256_SKX) && (is_transform_tpp == 0)) {
    is_valid_arch_prec = 0;
  }
  if ( (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (is_transform_tpp == 0) && (is_unary_simple_tpp == 0) && (has_inp_or_out_fp64 > 0)) {
    is_valid_arch_prec = 0;
  }
  if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && (libxsmm_meqn_is_unary_opcode_reduce_cols_idx_kernel(i_mateltwise_desc->param) > 0) && (has_inp_or_out_fp64 > 0)) {
    is_valid_arch_prec = 0;
  }
  if ((has_inp_or_out_fp64 > 0) && (has_all_inp_and_out_fp64 == 0)) {
    is_valid_arch_prec = 0;
  }
  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
    if (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_TERNARY_SELECT) {
      is_valid_arch_prec = 0;
    } else {
      if (LIBXSMM_DATATYPE_F32 != libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP)) {
        is_valid_arch_prec = 0;
      }
      if (LIBXSMM_DATATYPE_IMPLICIT != libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN2)) {
        is_valid_arch_prec = 0;
      }
      if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BITMASK_2BYTEMULT) == 0) {
        is_valid_arch_prec = 0;
      }
    }
  }

  return is_valid_arch_prec;
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_sse_avx_avx512_kernel( libxsmm_generated_code*         io_generated_code,
                                                         const libxsmm_meltw_descriptor* i_mateltwise_desc ) {
  libxsmm_mateltwise_kernel_config  l_kernel_config;
  libxsmm_mateltwise_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_loop_label_tracker        l_loop_label_tracker;

  /* Check if TPP is supported on current arch */
  if ( libxsmm_generator_mateltwise_x86_valid_arch_precision(io_generated_code, i_mateltwise_desc) == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
    return;
  }

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define gp register mapping */
  memset(&l_gp_reg_mapping, 0, sizeof(l_gp_reg_mapping));
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RCX;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RDI;
#endif

  /* define mateltwise kernel config */
  libxsmm_generator_mateltwise_init_micro_kernel_config_fullvector( io_generated_code, &l_kernel_config, i_mateltwise_desc);

  /* open asm */
  libxsmm_x86_instruction_open_stream_alt( io_generated_code, l_gp_reg_mapping.gp_reg_param_struct, 1 );

  /* being BLAS aligned, for empty kermls, do nothing */
  if ( (i_mateltwise_desc->m > 0) && ((i_mateltwise_desc->n > 0) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) || libxsmm_meqn_is_unary_opcode_reduce_cols_idx_kernel(i_mateltwise_desc->param)) ) {
    /* Stack management for melt kernel */
    libxsmm_generator_meltw_setup_stack_frame( io_generated_code, i_mateltwise_desc, &l_gp_reg_mapping, &l_kernel_config);

    /* Depending on the elementwise function, dispatch the proper code JITer */
    if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
      if (libxsmm_meqn_is_unary_opcode_reduce_kernel(i_mateltwise_desc->param) > 0) {
        if ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS) > 0) {
          libxsmm_generator_reduce_rows_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
        } else if (((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS) > 0) && (i_mateltwise_desc->param != LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD_NCNC_FORMAT)) {
          libxsmm_generator_reduce_cols_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
        } else if (((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS) > 0) && (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD_NCNC_FORMAT)) {
          libxsmm_generator_reduce_cols_ncnc_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
        } else {
          /* This should not happen */
          LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
          return;
        }
      } else if (libxsmm_meqn_is_unary_opcode_reduce_cols_idx_kernel(i_mateltwise_desc->param) > 0) {
        libxsmm_generator_reduce_cols_index_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) {
        libxsmm_generator_replicate_col_var_avx_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else if ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_GATHER) || (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_SCATTER)) {
        libxsmm_generator_gather_scatter_avx_avx512_microkernel ( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else if ( (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2)     ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT)     ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T)   ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2T)    ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD) ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4)     ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_NORM)     ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI2)    ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T)   ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4T)    ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2T_TO_NORM)    ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4T_TO_NORM)    ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD) ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8)     ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8_TO_VNNI8T)   ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8T)    ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8_PAD) ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8T_TO_NORM)    ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2)         ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2)         ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2)        ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD4)         ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD4)         ||
                  (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD4)          ) {
         libxsmm_generator_transform_x86_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      } else {
        libxsmm_generator_unary_binary_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
      }
    } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY ) {
      libxsmm_generator_unary_binary_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
    } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY ) {
      libxsmm_generator_unary_binary_avx512_microkernel( io_generated_code, &l_loop_label_tracker, &l_gp_reg_mapping, &l_kernel_config, i_mateltwise_desc );
    } else  {
      /* This should not happen */
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
      return;
    }

    /* Stack management formelt kernel */
    libxsmm_generator_meltw_destroy_stack_frame(  io_generated_code, i_mateltwise_desc, &l_kernel_config );
  }

  /* close asm */
  libxsmm_x86_instruction_close_stream_alt( io_generated_code, 1 );
}

