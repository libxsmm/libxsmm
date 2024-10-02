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
#include "generator_common_aarch64.h"
#include "generator_mateltwise_aarch64_reference.h"
#include "generator_mateltwise_common.h"
#include "generator_aarch64_instructions.h"
#include "generator_common.h"

void my_libxsmm_reference_elementwise(libxsmm_meltw_unary_param *param, libxsmm_meltw_descriptor *i_mateltwise_desc) {
  if (1) {
    int i, j;
    float in, out;
    for ( j = 0; j < i_mateltwise_desc->n; ++j ) {
      for ( i = 0; i < i_mateltwise_desc->m; ++i ) {
        libxsmm_bfloat16* bf16_in = (libxsmm_bfloat16*)(param->in.primary);
        libxsmm_bfloat16* bf16_out = (libxsmm_bfloat16*)(param->out.primary);
        in = libxsmm_convert_bf16_to_f32(bf16_in[(j*i_mateltwise_desc->ldi) + i]);
        out = LIBXSMM_EXPF(in);
        bf16_out[(j*i_mateltwise_desc->ldo) + i] = libxsmm_convert_f32_to_bf16_rne(out);
      }
    }
  } else {
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
  }
  return;
}


LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_aarch64_reference_kernel( libxsmm_generated_code*         io_generated_code,
                                                        const libxsmm_meltw_descriptor*     i_mateltwise_desc ) {
  unsigned char *l_code_buffer = (unsigned char *) io_generated_code->generated_code;
  unsigned long long l_imm_array_ptr[4];
  unsigned int l_temp_reg = LIBXSMM_AARCH64_GP_REG_X1;
  unsigned int l_temp_reg2 = LIBXSMM_AARCH64_GP_REG_X2;
  memcpy(l_imm_array_ptr, i_mateltwise_desc, sizeof(libxsmm_meltw_descriptor));
  /* open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xc00 );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_X29, 0, 0 );
  /* Here we add some arguments in the stack and pass them in the function by setting to x1/l_temp_reg the relevant stack pointer */
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_SUB_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 32, 0 );
  /* Now align RSP to 64 byte boundary */
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_temp_reg, 0xFFFFFFFFFFFFFFC0 );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, l_temp_reg2, 0, 0 );
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_AND_SR, l_temp_reg2, l_temp_reg, l_temp_reg2, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, l_temp_reg2, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
  /* Store the descriptor in stack and set argument in x1 */
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_temp_reg, l_imm_array_ptr[0] );
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_temp_reg2, l_imm_array_ptr[1] );
  libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 0, l_temp_reg, l_temp_reg2 );
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_temp_reg, l_imm_array_ptr[2] );
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_temp_reg2, l_imm_array_ptr[3] );
  libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 16, l_temp_reg, l_temp_reg2 );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, l_temp_reg, 0, 0 );
  /* We set the address of the function */
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_temp_reg2, (unsigned long long) my_libxsmm_reference_elementwise  );
  /* We call the function  */
  l_code_buffer[io_generated_code->code_size++] = 0x40;
  l_code_buffer[io_generated_code->code_size++] = 0x00;
  l_code_buffer[io_generated_code->code_size++] = 0x3f;
  l_code_buffer[io_generated_code->code_size++] = 0xd6;
  /* Recover stack pointer and caller-saved register  */
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_X29, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xc00 );
}
