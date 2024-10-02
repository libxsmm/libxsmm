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
#include "generator_common_x86.h"
#include "generator_mateltwise_x86_reference.h"
#include "generator_mateltwise_common.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "generator_mateltwise_reference_impl.h"

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_x86_reference_kernel( libxsmm_generated_code*         io_generated_code,
                                                        const libxsmm_meltw_descriptor* i_mateltwise_desc ) {
  unsigned char *l_code_buffer = (unsigned char *) io_generated_code->generated_code;
  libxsmm_mateltwise_gp_reg_mapping l_gp_reg_mapping;
  /* define gp register mapping */
  memset(&l_gp_reg_mapping, 0, sizeof(l_gp_reg_mapping));
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RCX;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_param_struct = LIBXSMM_X86_GP_REG_RDI;
#endif
  /* open asm */
  libxsmm_x86_instruction_open_stream_alt( io_generated_code, l_gp_reg_mapping.gp_reg_param_struct, 1 );
  libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBP );
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_RBP);
  /* Here we add some arguments in the stack and pass them in the function by settung to rsi the relevant stack pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, LIBXSMM_X86_GP_REG_RSP, 32 );
  /* Now align RSP to 64 byte boundary */
  libxsmm_x86_instruction_alu_imm_i64( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RAX, 0xFFFFFFFFFFFFFFC0 );
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_ANDQ, LIBXSMM_X86_GP_REG_RAX, LIBXSMM_X86_GP_REG_RSP);
  /* Store the descriptor in stack  */
  if ( io_generated_code->arch < LIBXSMM_X86_AVX ) {
    libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)i_mateltwise_desc, "l_desc", 'x', 0);
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VMOVUPS, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0,  0, 'x', 0, 0, 0, 1 );
    libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)i_mateltwise_desc + 16, "l_desc", 'x', 0);
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VMOVUPS, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0,  16, 'x', 0, 0, 0, 1 );
  } else {
    libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)i_mateltwise_desc, "l_desc", 'y', 0);
    libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VMOVUPS, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0,  0, 'y', 0, 0, 0, 1 );
  }
#if defined(_WIN32) || defined(__CYGWIN__)
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_RDX);
#else /* match calling convention on Linux */
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_RSI);
#endif
  /* We set the address of the function  */
  libxsmm_x86_instruction_alu_imm_i64( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RAX, (unsigned long long) libxsmm_reference_elementwise );
  /* We call the function  */
  l_code_buffer[io_generated_code->code_size++] = 0xff;
  l_code_buffer[io_generated_code->code_size++] = 0xd0;
  /* Recover stack pointer and caller-saved register  */
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_RSP);
  libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBP );
  /* close asm */
  libxsmm_x86_instruction_close_stream_alt( io_generated_code, 1 );
}
