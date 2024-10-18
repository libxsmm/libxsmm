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
#include "generator_x86_reference.h"
#include "generator_mateltwise_common.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "generator_mateltwise_reference_impl.h"
#include "generator_gemm_reference_impl.h"

LIBXSMM_API_INTERN
void libxsmm_generator_x86_reference_kernel( libxsmm_generated_code*         io_generated_code,
                                             const void*                     i_desc,
                                             unsigned int                    i_is_gemm_or_eltwise ) {
  unsigned long long l_padded_desc_size = (i_is_gemm_or_eltwise == 0) ? (((sizeof(libxsmm_gemm_descriptor)+31)/32) * 32) : (((sizeof(libxsmm_meltw_descriptor)+31)/32) * 32);
  unsigned long long i = 0;
  unsigned char *l_padded_desc = NULL;
  unsigned char *l_code_buffer = (unsigned char *) io_generated_code->generated_code;
  unsigned int input_param_reg = 0;
#if defined(_WIN32) || defined(__CYGWIN__)
  input_param_reg = LIBXSMM_X86_GP_REG_RCX;
#else /* match calling convention on Linux */
  input_param_reg = LIBXSMM_X86_GP_REG_RDI;
#endif
  l_padded_desc = (unsigned char*) malloc(l_padded_desc_size);
  memset(l_padded_desc, 0, l_padded_desc_size);
  if (i_is_gemm_or_eltwise == 0) {
    libxsmm_gemm_descriptor* i_gemm_desc = (libxsmm_gemm_descriptor*)i_desc;
    libxsmm_gemm_descriptor* desc_dst = (libxsmm_gemm_descriptor*)l_padded_desc;
    *desc_dst = *i_gemm_desc;
  } else {
    libxsmm_meltw_descriptor* i_mateltwise_desc = (libxsmm_meltw_descriptor*)i_desc;
    libxsmm_meltw_descriptor* desc_dst = (libxsmm_meltw_descriptor*)l_padded_desc;
    *desc_dst = *i_mateltwise_desc;
  }
  /* open asm */
  libxsmm_x86_instruction_open_stream_alt( io_generated_code, input_param_reg, 1 );
  libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBP );
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_RBP);
  /* Here we add some arguments in the stack and pass them in the function by settung to rsi the relevant stack pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, LIBXSMM_X86_GP_REG_RSP, l_padded_desc_size );
  /* Now align RSP to 64 byte boundary */
  libxsmm_x86_instruction_alu_imm_i64( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RAX, 0xFFFFFFFFFFFFFFC0 );
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_ANDQ, LIBXSMM_X86_GP_REG_RAX, LIBXSMM_X86_GP_REG_RSP);
  /* Store the descriptor in stack  */
  for (i = 0; i < l_padded_desc_size/32; i++) {
    if ( io_generated_code->arch < LIBXSMM_X86_AVX ) {
      libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)l_padded_desc + 32*i + 0, "l_desc", 'x', 0);
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VMOVUPS, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0,  0 + 32*i, 'x', 0, 0, 0, 1 );
      libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)l_padded_desc + 32*i + 16, "l_desc", 'x', 0);
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VMOVUPS, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0, 16 + 32*i, 'x', 0, 0, 0, 1 );
    } else {
      libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)l_padded_desc + 32*i, "l_desc", 'y', 0);
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VMOVUPS, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0,  32*i, 'y', 0, 0, 0, 1 );
    }
  }
#if defined(_WIN32) || defined(__CYGWIN__)
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_RDX);
#else /* match calling convention on Linux */
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_RSI);
#endif
  /* We set the address of the function  */
  if (i_is_gemm_or_eltwise == 0) {
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RAX, (unsigned long long) libxsmm_reference_gemm );
  } else {
    libxsmm_x86_instruction_alu_imm_i64( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RAX, (unsigned long long) libxsmm_reference_elementwise );
  }
  /* We call the function  */
  l_code_buffer[io_generated_code->code_size++] = 0xff;
  l_code_buffer[io_generated_code->code_size++] = 0xd0;
  /* Recover stack pointer and caller-saved register  */
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_RSP);
  libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBP );
  /* close asm */
  libxsmm_x86_instruction_close_stream_alt( io_generated_code, 1 );
  free(l_padded_desc);
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_x86_reference_kernel( libxsmm_generated_code* io_generated_code, const libxsmm_meltw_descriptor* i_mateltwise_desc ) {
  libxsmm_generator_x86_reference_kernel( io_generated_code, (const void*) i_mateltwise_desc, 1);
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_x86_reference_kernel( libxsmm_generated_code* io_generated_code, const libxsmm_gemm_descriptor* i_gemm_desc ) {
  libxsmm_generator_x86_reference_kernel( io_generated_code, (const void*) i_gemm_desc, 0);
}


