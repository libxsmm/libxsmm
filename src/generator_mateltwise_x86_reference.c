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

void read_descriptor(libxsmm_meltw_unary_param *param,  const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  int i, j;
  float *in, *out;
  in = (float*)(param->in.primary);
  out = (float*)(param->out.primary);

  for ( j = 0; j < i_mateltwise_desc->n; ++j ) {
    for ( i = 0; i < i_mateltwise_desc->m; ++i ) {
      float in_value = in[(j*i_mateltwise_desc->ldi) + i];
      float out_value = LIBXSMM_EXPF(in_value);
      out[(j*i_mateltwise_desc->ldo) + i] = out_value;
    }
  }
#if 0
  binary_op_ref( i_mateltwise_desc->m, i_mateltwise_desc->n, i_mateltwise_desc->ldi, i_mateltwise_desc->ldi2, i_mateltwise_desc->ldo,
                 param->in0.primary, param->in1.primary, param->out.primary, 32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_IMPLICIT, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_BITMASK_2BYTEMULT, NULL );
  float *tmp_ptr = (float*)(param->in1.primary);
  float the_val = tmp_ptr[0];
  printf("The TPP was dispatched with m = %d and n = %d\n", i_mateltwise_desc->m, i_mateltwise_desc->n);
  printf("The struct pointer is %p and the argument pointer is %p and ldi2 is %d\n", param, tmp_ptr, i_mateltwise_desc->ldi2);
  printf("The value at TPP location m=4, n=2 of input 2 is %.5f\n", the_val);
#endif
  return;
}

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
  libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)i_mateltwise_desc, "l_desc_ptr", 'x', 0);
  libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VMOVUPS, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0,  0, 'x', 0, 0, 0, 1 );
  libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)i_mateltwise_desc + 16, "l_desc_ptr", 'x', 0);
  libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VMOVUPS, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0,  16, 'x', 0, 0, 0, 1 );
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_RSI);
  /* We set the address of the function  */
  libxsmm_x86_instruction_alu_imm_i64( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RAX, (unsigned long long) read_descriptor );
  /* We call the function  */
  l_code_buffer[io_generated_code->code_size++] = 0xff;
  l_code_buffer[io_generated_code->code_size++] = 0xd0;
  /* Recover stack pointer and caller-saved register  */
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_RSP);
  libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBP );
  /* close asm */
  libxsmm_x86_instruction_close_stream_alt( io_generated_code, 1 );
}
