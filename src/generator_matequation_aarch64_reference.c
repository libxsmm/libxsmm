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
#include "generator_matequation_avx_avx512.h"
#include "generator_common_aarch64.h"
#include "generator_matequation_aarch64_reference.h"
#include "generator_aarch64_reference.h"
#include "generator_mateltwise_common.h"
#include "generator_aarch64_instructions.h"
#include "generator_common.h"
#include "generator_mateltwise_reference_impl.h"
#include "generator_matequation_reference_impl.h"
#include "libxsmm_matrixeqn.h"

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_aarch64_reference_kernel( libxsmm_generated_code*         io_generated_code,
                                                         const libxsmm_meqn_descriptor* i_mateqn_desc ) {
  unsigned char *l_code_buffer = (unsigned char *) io_generated_code->generated_code;
  unsigned int l_temp_reg = LIBXSMM_AARCH64_GP_REG_X1;
  unsigned int l_temp_reg2 = LIBXSMM_AARCH64_GP_REG_X2;
  unsigned int l_temp_reg3 = LIBXSMM_AARCH64_GP_REG_X3;
  unsigned int l_temp_reg4 = LIBXSMM_AARCH64_GP_REG_X4;
  unsigned int l_temp_reg5 = LIBXSMM_AARCH64_GP_REG_X5;
  unsigned int eqn_idx = i_mateqn_desc->eqn_idx;
  libxsmm_matrix_eqn *eqn = libxsmm_meqn_get_equation( eqn_idx );
  unsigned int timestamp = 0;
  unsigned int last_timestamp;
  libxsmm_meqn_elem *unfolded_exec_tree = NULL;
  unsigned long long padded_size = 0, i = 0;
  unsigned long long tree_max_comp_tsize = 0;
  unsigned long long n_tmp = 0;
  unsigned long long tmp_size = 0;
  unsigned long long scratch_size = 0;
  unsigned long long *l_imm_array_ptr;
  if ( eqn == NULL ) {
    fprintf( stderr, "The requested equation does not exist... nothing to JIT,,,\n" );
    return;
  } else {
    last_timestamp = eqn->eqn_root->visit_timestamp;
    padded_size = ((((last_timestamp+1)*5*sizeof(libxsmm_meqn_elem))+63)/64)*64;
    unfolded_exec_tree = (libxsmm_meqn_elem*) malloc(padded_size);
    l_imm_array_ptr = (unsigned long long*)unfolded_exec_tree;
    memset(unfolded_exec_tree, 0, padded_size);
    tree_max_comp_tsize = eqn->eqn_root->tree_max_comp_tsize;
    n_tmp = eqn->eqn_root->reg_score;
    tmp_size = eqn->eqn_root->max_tmp_size * tree_max_comp_tsize;
  }
  /* open asm */
  libxsmm_aarch64_instruction_open_stream( io_generated_code, 0xc00 );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_X29, 0, 0 );
  /* Now align RSP to 64 byte boundary */
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_temp_reg, 0xFFFFFFFFFFFFFFC0 );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, l_temp_reg2, 0, 0 );
  libxsmm_aarch64_instruction_alu_compute_shifted_reg( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_AND_SR, l_temp_reg2, l_temp_reg, l_temp_reg2, 0, LIBXSMM_AARCH64_SHIFTMODE_LSL );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, l_temp_reg2, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
  /* Here we unfold the optimized execution tree and store it in the stack */
  for (timestamp = 0; timestamp <= last_timestamp; timestamp++) {
    libxsmm_meqn_elem *cur_op = libxsmm_generator_matequation_find_op_at_timestamp(eqn->eqn_root, timestamp);
    if (cur_op     != NULL) unfolded_exec_tree[timestamp*5+0] = *cur_op;
    if (cur_op->le != NULL) unfolded_exec_tree[timestamp*5+1] = *(cur_op->le);
    if (cur_op->ri != NULL) unfolded_exec_tree[timestamp*5+2] = *(cur_op->ri);
    if (cur_op->r2 != NULL) unfolded_exec_tree[timestamp*5+3] = *(cur_op->r2);
    if (cur_op->up != NULL) unfolded_exec_tree[timestamp*5+4] = *(cur_op->up);
    if (timestamp == last_timestamp) {
      unfolded_exec_tree[timestamp*5+0].reg_score = -1;
      unfolded_exec_tree[timestamp*5+0].tmp.ld = i_mateqn_desc->ldo;
      unfolded_exec_tree[timestamp*5+0].tmp.dtype = (libxsmm_datatype) LIBXSMM_GETENUM_OUT(i_mateqn_desc->datatype);
    }
  }
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, l_temp_reg5, 0, 0 );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 l_temp_reg5, l_temp_reg, l_temp_reg5,
                                                 (long long) padded_size );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, l_temp_reg5, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
  /* Store the unfolded descriptor in stack and set x1  */
  for (i = 0; i < padded_size/32; i++) {
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_temp_reg, l_imm_array_ptr[0] );
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_temp_reg2, l_imm_array_ptr[1] );
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 0, l_temp_reg, l_temp_reg2 );
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_temp_reg, l_imm_array_ptr[2] );
    libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_temp_reg2, l_imm_array_ptr[3] );
    libxsmm_aarch64_instruction_alu_pair_move( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_STP_I_OFF, LIBXSMM_AARCH64_GP_REG_XSP, 16, l_temp_reg, l_temp_reg2 );
    libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, LIBXSMM_AARCH64_GP_REG_XSP, 32, 0 );
    l_imm_array_ptr += 4;
  }
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, l_temp_reg5, 0, 0 );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 l_temp_reg5, l_temp_reg, l_temp_reg5,
                                                 (long long) padded_size );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, l_temp_reg5, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, l_temp_reg, 0, 0 );
  /* Get scratchpad pointer and set 3rd argument */
  tmp_size = (tmp_size % 64 == 0) ? tmp_size : ((tmp_size + 63)/64) * 64;
  scratch_size = tmp_size * n_tmp;
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, l_temp_reg5, 0, 0 );
  libxsmm_aarch64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_META_SUB,
                                                 l_temp_reg5, l_temp_reg2, l_temp_reg5,
                                                 (long long) scratch_size );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, l_temp_reg5, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_XSP, l_temp_reg2, 0, 0 );

  if (libxsmm_verbosity < 0) {
    fprintf( stderr, "JITing Matrix Equation with reference code (n_tmp = %lld , stack_scratch_size = %.5g KB)\n", n_tmp, (1.0*scratch_size)/1024.0 );
  }

  /* Set size of temp in 4th argument */
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_temp_reg3, (unsigned long long)tmp_size );

  /* We set the address of the function */
  libxsmm_aarch64_instruction_alu_set_imm64( io_generated_code, l_temp_reg4, (unsigned long long) libxsmm_reference_matequation  );
  /* We call the function  */
  l_code_buffer[io_generated_code->code_size++] = 0x80;
  l_code_buffer[io_generated_code->code_size++] = 0x00;
  l_code_buffer[io_generated_code->code_size++] = 0x3f;
  l_code_buffer[io_generated_code->code_size++] = 0xd6;
  /* Recover stack pointer and caller-saved register  */
  libxsmm_aarch64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_AARCH64_INSTR_GP_ADD_I, LIBXSMM_AARCH64_GP_REG_X29, LIBXSMM_AARCH64_GP_REG_XSP, 0, 0 );
  /* close asm */
  libxsmm_aarch64_instruction_close_stream( io_generated_code, 0xc00 );
  free(unfolded_exec_tree);
}
