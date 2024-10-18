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
#include "generator_aarch64_reference.h"
#include "generator_mateltwise_common.h"
#include "generator_aarch64_instructions.h"
#include "generator_common.h"
#include "generator_mateltwise_reference_impl.h"
#include "generator_gemm_reference_impl.h"

LIBXSMM_API_INTERN
void libxsmm_generator_aarch64_reference_kernel( libxsmm_generated_code*         io_generated_code,
                                                 void*                           i_desc,
                                                 unsigned int                    i_is_gemm_or_eltwise ) {
  unsigned long long l_padded_desc_size = (i_is_gemm_or_eltwise == 0) ? (((sizeof(libxsmm_gemm_descriptor)+31)/32) * 32) : (((sizeof(libxsmm_meltw_descriptor)+31)/32) * 32);
  unsigned long long i = 0;
  unsigned char *l_padded_desc = NULL;
  unsigned long long *l_imm_array_ptr = NULL;
  unsigned char *l_code_buffer = (unsigned char *) io_generated_code->generated_code;
  unsigned int l_temp_reg = LIBXSMM_AARCH64_GP_REG_X1;
  unsigned int l_temp_reg2 = LIBXSMM_AARCH64_GP_REG_X2;

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

  l_imm_array_ptr = (unsigned long long*)l_padded_desc;

  /* open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xc00 );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_X29, 0, 0 );
  /* Here we add some arguments in the stack and pass them in the function by setting to x1/l_temp_reg the relevant stack pointer */
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, l_temp_reg2, 0, 0 );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 l_temp_reg2, l_temp_reg, l_temp_reg2,
                                                 (long long) l_padded_desc_size );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, l_temp_reg2, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
  /* Now align RSP to 64 byte boundary */
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_temp_reg, 0xFFFFFFFFFFFFFFC0 );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, l_temp_reg2, 0, 0 );
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_AND_SR, l_temp_reg2, l_temp_reg, l_temp_reg2, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, l_temp_reg2, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
  /* Store the descriptor in stack and set argument in x1 */
  for (i = 0; i < l_padded_desc_size/32; i++) {
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_temp_reg, l_imm_array_ptr[0] );
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_temp_reg2, l_imm_array_ptr[1] );
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 0, l_temp_reg, l_temp_reg2 );
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_temp_reg, l_imm_array_ptr[2] );
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_temp_reg2, l_imm_array_ptr[3] );
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 16, l_temp_reg, l_temp_reg2 );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 32, 0 );
    l_imm_array_ptr += 4;
  }
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, l_temp_reg2, 0, 0 );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 l_temp_reg2, l_temp_reg, l_temp_reg2,
                                                 (long long) l_padded_desc_size );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, l_temp_reg2, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, l_temp_reg, 0, 0 );
  /* We set the address of the function  */
  if (i_is_gemm_or_eltwise == 0) {
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_temp_reg2, (unsigned long long) libxsmm_reference_gemm  );
  } else {
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_temp_reg2, (unsigned long long) libxsmm_reference_elementwise  );
  }
  /* We call the function  */
  l_code_buffer[io_generated_code->code_size++] = 0x40;
  l_code_buffer[io_generated_code->code_size++] = 0x00;
  l_code_buffer[io_generated_code->code_size++] = 0x3f;
  l_code_buffer[io_generated_code->code_size++] = 0xd6;
  /* Recover stack pointer and caller-saved register  */
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_X29, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xc00 );
  free(l_padded_desc);
}


LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_aarch64_reference_kernel( libxsmm_generated_code* io_generated_code, const libxsmm_meltw_descriptor* i_mateltwise_desc ) {
  libxsmm_generator_aarch64_reference_kernel( io_generated_code, (const void*) i_mateltwise_desc, 1);
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_aarch64_reference_kernel( libxsmm_generated_code* io_generated_code, const libxsmm_gemm_descriptor* i_gemm_desc ) {
  libxsmm_generator_aarch64_reference_kernel( io_generated_code, (const void*) i_gemm_desc, 0);
}



