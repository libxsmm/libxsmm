/******************************************************************************
* Copyright (c) 2025 IBM Corp. - All rights reserved.                         *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Will Trojak (IBM Corp.)
******************************************************************************/
#include "generator_ppc64le_reference.h"


LIBXSMM_API_INTERN
void libxsmm_generator_ppc64le_reference_kernel( libxsmm_generated_code* io_generated_code,
                                                 const void*             i_desc,
                                                 unsigned int            i_is_gemm_or_eltwise ) {
  unsigned long l_padded_desc_size = ( i_is_gemm_or_eltwise == 0 ) ?
    ( ( ( sizeof(libxsmm_gemm_descriptor) + 31 ) / 32 ) * 32 ) : ( ( ( sizeof(libxsmm_meltw_descriptor) + 31 ) / 32 ) * 32 );
  unsigned long i = 0;
  unsigned int stack_offset = 0;
  unsigned char *l_padded_desc = NULL;
  unsigned long *l_imm_array_ptr = NULL;
  unsigned int l_temp_reg1, l_temp_reg2, l_sp_copy;
  libxsmm_ref_code_pointer code_ptr;
  libxsmm_ppc64le_reg l_reg_tracker;

  /* Initialise reg tracker */
  l_reg_tracker = libxsmm_ppc64le_reg_init();

  code_ptr.ptr = NULL;
  l_padded_desc = (unsigned char*) malloc(l_padded_desc_size);
  memset(l_padded_desc, 0, l_padded_desc_size);
  if (i_is_gemm_or_eltwise == 0) {
    const libxsmm_gemm_descriptor* i_gemm_desc = (const libxsmm_gemm_descriptor*)i_desc;
    libxsmm_gemm_descriptor* desc_dst = (libxsmm_gemm_descriptor*)l_padded_desc;
    *desc_dst = *i_gemm_desc;
  } else {
    const libxsmm_meltw_descriptor* i_mateltwise_desc = (const libxsmm_meltw_descriptor*)i_desc;
    libxsmm_meltw_descriptor* desc_dst = (libxsmm_meltw_descriptor*)l_padded_desc;
    *desc_dst = *i_mateltwise_desc;
  }

  l_imm_array_ptr = (unsigned long*)l_padded_desc;

  /* Open stream */
  libxsmm_ppc64le_instr_open_stream( io_generated_code, &l_reg_tracker );

  /* Set some registers as used */
  libxsmm_ppc64le_instr_used_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_R31 );
  l_sp_copy = LIBXSMM_PPC64LE_GPR_R31;
  libxsmm_ppc64le_instr_used_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_R30 );
  l_temp_reg1 = LIBXSMM_PPC64LE_GPR_R30;
  libxsmm_ppc64le_instr_used_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_R29 );
  l_temp_reg2 = LIBXSMM_PPC64LE_GPR_R29;
  libxsmm_ppc64le_instr_used_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_R3 );

  /* Increament stack pointer to store description struct */
  libxsmm_ppc64le_instr_add_value( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR_SP, l_sp_copy, 0 );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_STDU, LIBXSMM_PPC64LE_GPR_SP, LIBXSMM_PPC64LE_GPR_SP, ( -l_padded_desc_size ) >> 2 );

  /* Store the descriptor in stack and set argument in x1 */
  stack_offset = 0;
  for (i = 0; i < l_padded_desc_size / 32; i++) {
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, &l_reg_tracker, l_temp_reg1, l_imm_array_ptr[0] );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_STD, l_temp_reg1, LIBXSMM_PPC64LE_GPR_SP, ( stack_offset + 0 ) >> 2 );

    libxsmm_ppc64le_instr_set_imm64( io_generated_code, &l_reg_tracker, l_temp_reg2, l_imm_array_ptr[1] );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_STD, l_temp_reg2, LIBXSMM_PPC64LE_GPR_SP, ( stack_offset + 8 ) >> 2 );

    libxsmm_ppc64le_instr_set_imm64( io_generated_code, &l_reg_tracker, l_temp_reg1, l_imm_array_ptr[2] );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_STD, l_temp_reg1, LIBXSMM_PPC64LE_GPR_SP, ( stack_offset + 16 ) >> 2 );

    libxsmm_ppc64le_instr_set_imm64( io_generated_code, &l_reg_tracker, l_temp_reg2, l_imm_array_ptr[3] );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_STD, l_temp_reg2, LIBXSMM_PPC64LE_GPR_SP, ( stack_offset + 24 ) >> 2 );

    l_imm_array_ptr += 4;
    stack_offset += 32;
  }

  /* Set the address of the function  */
  if (i_is_gemm_or_eltwise == 0) {
    code_ptr.ptr_gemm_fn = libxsmm_reference_gemm;
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, &l_reg_tracker, l_temp_reg2, code_ptr.uval );
  } else {
    code_ptr.ptr_eltw_fn = libxsmm_reference_elementwise;
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, &l_reg_tracker, l_temp_reg2, code_ptr.uval );
  }

  /* Set argument register */
  libxsmm_ppc64le_instr_add_value( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR_SP, LIBXSMM_PPC64LE_GPR_R3, 0 );

  /* Call the function */
  libxsmm_ppc64le_instr_jump_lr( io_generated_code, l_temp_reg2 );

  /* Recover stack pointer */
  libxsmm_ppc64le_instr_add_value( io_generated_code, &l_reg_tracker, l_sp_copy, LIBXSMM_PPC64LE_GPR_SP, 0 );

  free(l_padded_desc);

  /* Free the register used */
  libxsmm_ppc64le_instr_free_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_R31 );
  libxsmm_ppc64le_instr_free_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_R30 );
  libxsmm_ppc64le_instr_free_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_R29 );
  libxsmm_ppc64le_instr_free_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_R3 );

  /* Close stream */
  libxsmm_ppc64le_instr_colapse_stack( io_generated_code, &l_reg_tracker );
}
