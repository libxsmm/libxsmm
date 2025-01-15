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
#include "generator_common_rv64.h"
#include "generator_rv64_reference.h"
#include "generator_mateltwise_common.h"
#include "generator_rv64_instructions.h"
#include "generator_common.h"
#include "generator_mateltwise_reference_impl.h"
#include "generator_gemm_reference_impl.h"
#include "generator_matequation_avx_avx512.h"
#include "generator_matequation_reference_impl.h"
#include "libxsmm_matrixeqn.h"

LIBXSMM_API_INTERN
void libxsmm_generator_rv64_reference_kernel( libxsmm_generated_code*         io_generated_code,
                                                 const void*                     i_desc,
                                                 unsigned int                    i_is_gemm_or_eltwise ) {
  unsigned long long l_padded_desc_size = (i_is_gemm_or_eltwise == 0) ? (((sizeof(libxsmm_gemm_descriptor)+31)/32) * 32) : (((sizeof(libxsmm_meltw_descriptor)+31)/32) * 32);
  unsigned long long i = 0;
  unsigned char *l_padded_desc = NULL;
  unsigned long long *l_imm_array_ptr = NULL;
#if 0
  unsigned char *l_code_buffer = (unsigned char *) io_generated_code->generated_code;
#endif
  unsigned int l_temp_reg = LIBXSMM_RV64_GP_REG_X28;
  unsigned int l_temp_reg2 = LIBXSMM_RV64_GP_REG_X29;
  libxsmm_ref_code_pointer code_ptr;
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

  l_imm_array_ptr = (unsigned long long*)l_padded_desc;

  /* open asm */
  libxsmm_rv64_instruction_open_stream( io_generated_code, 0xc00 );
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, LIBXSMM_RV64_GP_REG_X30, 0 );
  /* Here we add some arguments in the stack and pass them in the function by setting to x1/l_temp_reg the relevant stack pointer */
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, l_temp_reg2, 0 );
  libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB, l_temp_reg2, l_temp_reg, l_temp_reg2,
                                                 (long long) l_padded_desc_size );
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, l_temp_reg2, LIBXSMM_RV64_GP_REG_XSP, 0 );

  /* Now align RSP to 64 byte boundary */
  libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_temp_reg, 0xFFFFFFFFFFFFFFC0 );
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, l_temp_reg2, 0 );
  libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_AND, l_temp_reg2, l_temp_reg, l_temp_reg2 );
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, l_temp_reg2, LIBXSMM_RV64_GP_REG_XSP, 0 );

  /* Store the descriptor in stack and set argument in x1 */
  for (i = 0; i < l_padded_desc_size/32; i++) {
    libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_temp_reg, l_imm_array_ptr[0] );
    libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_temp_reg2, l_imm_array_ptr[1] );
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_SD, LIBXSMM_RV64_GP_REG_XSP, 0, l_temp_reg );
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_SD, LIBXSMM_RV64_GP_REG_XSP, 8, l_temp_reg2 );
    libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_temp_reg, l_imm_array_ptr[2] );
    libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_temp_reg2, l_imm_array_ptr[3] );
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_SD, LIBXSMM_RV64_GP_REG_XSP, 16, l_temp_reg );
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_SD, LIBXSMM_RV64_GP_REG_XSP, 24, l_temp_reg2 );
    libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, LIBXSMM_RV64_GP_REG_XSP, 32 );
    l_imm_array_ptr += 4;
  }

  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, l_temp_reg2, 0 );
  libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB, l_temp_reg2, l_temp_reg, l_temp_reg2,
                                                 (long long) l_padded_desc_size );
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, l_temp_reg2, LIBXSMM_RV64_GP_REG_XSP, 0 );
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, l_temp_reg, 0 );

  /* We set the address of the function  */
  if (i_is_gemm_or_eltwise == 0) {
    code_ptr.ptr_gemm_fn = libxsmm_reference_gemm;
    libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_temp_reg2, code_ptr.uval  );
  } else {
    code_ptr.ptr_eltw_fn = libxsmm_reference_elementwise;
    libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_temp_reg2, code_ptr.uval  );
  }

  /* We call the function  */
  libxsmm_rv64_instruction_jump_and_link(io_generated_code, LIBXSMM_RV64_INSTR_GP_JALR, LIBXSMM_RV64_GP_REG_X1, l_temp_reg2);

#if 0
  l_code_buffer[io_generated_code->code_size++] = 0x40;
  l_code_buffer[io_generated_code->code_size++] = 0x00;
  l_code_buffer[io_generated_code->code_size++] = 0x3f;
  l_code_buffer[io_generated_code->code_size++] = 0xd6;
#endif
  /* Recover stack pointer and caller-saved register  */
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_X30, LIBXSMM_RV64_GP_REG_XSP, 0 );

  /* close asm */
  libxsmm_rv64_instruction_close_stream( io_generated_code, 0xc000 );
  free(l_padded_desc);
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_rv64_reference_kernel( libxsmm_generated_code* io_generated_code, const libxsmm_meltw_descriptor* i_mateltwise_desc ) {
  libxsmm_generator_rv64_reference_kernel( io_generated_code, (const void*) i_mateltwise_desc, 1);
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_rv64_reference_kernel( libxsmm_generated_code* io_generated_code, const libxsmm_gemm_descriptor* i_gemm_desc ) {
  libxsmm_generator_rv64_reference_kernel( io_generated_code, (const void*) i_gemm_desc, 0);
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_rv64_reference_kernel( libxsmm_generated_code*         io_generated_code,
                                                         const libxsmm_meqn_descriptor* i_mateqn_desc ) {
#if 0
  unsigned char *l_code_buffer = (unsigned char *) io_generated_code->generated_code;
#endif
  unsigned int l_temp_reg = LIBXSMM_RV64_GP_REG_X5;
  unsigned int l_temp_reg2 = LIBXSMM_RV64_GP_REG_X6;
  unsigned int l_temp_reg3 = LIBXSMM_RV64_GP_REG_X7;
  unsigned int l_temp_reg4 = LIBXSMM_RV64_GP_REG_X28;
  unsigned int l_temp_reg5 = LIBXSMM_RV64_GP_REG_X29;
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
  libxsmm_ref_code_pointer code_ptr;
  code_ptr.ptr = NULL;
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
  libxsmm_rv64_instruction_open_stream( io_generated_code, 0x3000 );
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, LIBXSMM_RV64_GP_REG_X30, 0 );

  /* Now align RSP to 64 byte boundary */
  libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_temp_reg, 0xFFFFFFFFFFFFFFC0 );
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, l_temp_reg2, 0 );
  libxsmm_rv64_instruction_alu_compute( io_generated_code, LIBXSMM_RV64_INSTR_GP_AND, l_temp_reg2, l_temp_reg, l_temp_reg2 );
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, l_temp_reg2, LIBXSMM_RV64_GP_REG_XSP, 0 );

  /* Here we unfold the optimized execution tree and store it in the stack */
  for (timestamp = 0; timestamp <= last_timestamp; timestamp++) {
    libxsmm_meqn_elem *cur_op = libxsmm_generator_matequation_find_op_at_timestamp(eqn->eqn_root, timestamp);
    if (cur_op     != NULL) unfolded_exec_tree[timestamp*5+0] = *cur_op;
    if (cur_op !=NULL && cur_op->le != NULL) unfolded_exec_tree[timestamp*5+1] = *(cur_op->le);
    if (cur_op !=NULL && cur_op->ri != NULL) unfolded_exec_tree[timestamp*5+2] = *(cur_op->ri);
    if (cur_op !=NULL && cur_op->r2 != NULL) unfolded_exec_tree[timestamp*5+3] = *(cur_op->r2);
    if (cur_op !=NULL && cur_op->up != NULL) unfolded_exec_tree[timestamp*5+4] = *(cur_op->up);
    if (timestamp == last_timestamp) {
      unfolded_exec_tree[timestamp*5+0].reg_score = -1;
      unfolded_exec_tree[timestamp*5+0].tmp.ld = i_mateqn_desc->ldo;
      unfolded_exec_tree[timestamp*5+0].tmp.dtype = (libxsmm_datatype) LIBXSMM_GETENUM_OUT(i_mateqn_desc->datatype);
    }
  }

  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, l_temp_reg5, 0 );
  libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB,
                                                 l_temp_reg5, l_temp_reg, l_temp_reg5, (long long) padded_size );
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, l_temp_reg5, LIBXSMM_RV64_GP_REG_XSP, 0 );

  /* Store the unfolded descriptor in stack and set x1  */
  for (i = 0; i < padded_size/32; i++) {
    libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_temp_reg, l_imm_array_ptr[0] );
    libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_temp_reg2, l_imm_array_ptr[1] );
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_SD, LIBXSMM_RV64_GP_REG_XSP, 0, l_temp_reg );
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_SD, LIBXSMM_RV64_GP_REG_XSP, 8, l_temp_reg2 );
    libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_temp_reg, l_imm_array_ptr[2] );
    libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_temp_reg2, l_imm_array_ptr[3] );
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_SD, LIBXSMM_RV64_GP_REG_XSP, 16, l_temp_reg );
    libxsmm_rv64_instruction_alu_move( io_generated_code, LIBXSMM_RV64_INSTR_GP_SD, LIBXSMM_RV64_GP_REG_XSP, 24, l_temp_reg2 );
    libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, LIBXSMM_RV64_GP_REG_XSP, 32 );
    l_imm_array_ptr += 4;
  }
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, l_temp_reg5, 0 );
  libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB, l_temp_reg5, l_temp_reg, l_temp_reg5,
                                                 (long long) padded_size );
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, l_temp_reg5, LIBXSMM_RV64_GP_REG_XSP, 0 );
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, l_temp_reg, 0 );
  /* Get scratchpad pointer and set 3rd argument */
  tmp_size = (tmp_size % 64 == 0) ? tmp_size : ((tmp_size + 63)/64) * 64;
  scratch_size = tmp_size * n_tmp;
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, l_temp_reg5, 0 );
  libxsmm_rv64_instruction_alu_compute_imm64( io_generated_code, LIBXSMM_RV64_INSTR_GP_SUB, l_temp_reg5, l_temp_reg2, l_temp_reg5,
                                                 (long long) scratch_size );
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, l_temp_reg5, LIBXSMM_RV64_GP_REG_XSP, 0 );
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_XSP, l_temp_reg2, 0 );

  if (libxsmm_verbosity < 0) {
    fprintf( stderr, "JITing Matrix Equation with reference code (n_tmp = %lld , stack_scratch_size = %.5g KB)\n", n_tmp, (1.0*scratch_size)/1024.0 );
  }

  /* Set size of temp in 4th argument */
  libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_temp_reg3, (unsigned long long)tmp_size );

  /* We set the address of the function */
  code_ptr.ptr_meqn_fn = libxsmm_reference_matequation;
  libxsmm_rv64_instruction_alu_set_imm64( io_generated_code, l_temp_reg4, code_ptr.uval  );

  /* We call the function  */
  libxsmm_rv64_instruction_jump_and_link(io_generated_code, LIBXSMM_RV64_INSTR_GP_JALR, LIBXSMM_RV64_GP_REG_X1, l_temp_reg4);

#if 0
  /* We call the function  */
  l_code_buffer[io_generated_code->code_size++] = 0x80;
  l_code_buffer[io_generated_code->code_size++] = 0x00;
  l_code_buffer[io_generated_code->code_size++] = 0x3f;
  l_code_buffer[io_generated_code->code_size++] = 0xd6;
#endif

  /* Recover stack pointer and caller-saved register  */
  libxsmm_rv64_instruction_alu_compute_imm12( io_generated_code, LIBXSMM_RV64_INSTR_GP_ADDI, LIBXSMM_RV64_GP_REG_X30, LIBXSMM_RV64_GP_REG_XSP, 0 );
  /* close asm */
  libxsmm_rv64_instruction_close_stream( io_generated_code, 0x3000 );
  free(unfolded_exec_tree);
}
