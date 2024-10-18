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
#include "generator_matequation_scratch_avx_avx512.h"
#include "generator_matequation_regblocks_avx_avx512.h"
#include "generator_common_x86.h"
#include "generator_matequation_x86_reference.h"
#include "generator_x86_reference.h"
#include "generator_mateltwise_common.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "generator_mateltwise_reference_impl.h"
#include "generator_matequation_reference_impl.h"
#include "libxsmm_matrixeqn.h"

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_x86_reference_kernel( libxsmm_generated_code*         io_generated_code,
                                                         const libxsmm_meqn_descriptor* i_mateqn_desc ) {
  unsigned char *l_code_buffer = (unsigned char *) io_generated_code->generated_code;
  unsigned int input_param_reg = 0;
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
  if ( eqn == NULL ) {
    fprintf( stderr, "The requested equation does not exist... nothing to JIT,,,\n" );
    return;
  } else {
    last_timestamp = eqn->eqn_root->visit_timestamp;
    padded_size = ((((last_timestamp+1)*5*sizeof(libxsmm_meqn_elem))+63)/64)*64;
    unfolded_exec_tree = (libxsmm_meqn_elem*) malloc(padded_size);
    memset(unfolded_exec_tree, 0, padded_size);
    tree_max_comp_tsize = eqn->eqn_root->tree_max_comp_tsize;
    n_tmp = eqn->eqn_root->reg_score;
    tmp_size = eqn->eqn_root->max_tmp_size * tree_max_comp_tsize;
  }
#if defined(_WIN32) || defined(__CYGWIN__)
  input_param_reg = LIBXSMM_X86_GP_REG_RCX;
#else /* match calling convention on Linux */
  input_param_reg = LIBXSMM_X86_GP_REG_RDI;
#endif
  /* Open asm */
  libxsmm_x86_instruction_open_stream_alt( io_generated_code, input_param_reg, 1 );
  libxsmm_x86_instruction_push_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBP );
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_RBP);
  /* Setup the stack */
  /* Now align RSP to 64 byte boundary */
  libxsmm_x86_instruction_alu_imm_i64( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RAX, 0xFFFFFFFFFFFFFFC0 );
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_ANDQ, LIBXSMM_X86_GP_REG_RAX, LIBXSMM_X86_GP_REG_RSP);
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
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, LIBXSMM_X86_GP_REG_RSP, padded_size );
  /* Store the unfolded descriptor in stack and set RSI  */
  for (i = 0; i < padded_size/32; i++) {
    if ( io_generated_code->arch < LIBXSMM_X86_AVX ) {
      libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)unfolded_exec_tree + 0  + 32*i, "l_unfolded_exec_tree", 'x', 0);
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VMOVUPS, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0,  0  + 32*i, 'x', 0, 0, 0, 1 );
      libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)unfolded_exec_tree + 16 + 32*i, "l_unfolded_exec_tree", 'x', 0);
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VMOVUPS, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0,  16 + 32*i, 'x', 0, 0, 0, 1 );
    } else {
      libxsmm_x86_instruction_full_vec_load_of_constants( io_generated_code, (const unsigned char*)unfolded_exec_tree + 32*i, "l_unfolded_exec_tree", 'y', 0);
      libxsmm_x86_instruction_vec_move( io_generated_code, io_generated_code->arch, LIBXSMM_X86_INSTR_VMOVUPS, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_UNDEF, 0,  32*i, 'y', 0, 0, 0, 1 );
    }
  }
  /* Set 2nd argument */
#if defined(_WIN32) || defined(__CYGWIN__)
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_RDX);
#else /* match calling convention on Linux */
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_RSI);
#endif

  /* Get scratchpad pointer and set 3rd argument */
  tmp_size = (tmp_size % 64 == 0) ? tmp_size : ((tmp_size + 63)/64) * 64;
  scratch_size = tmp_size * n_tmp;
  libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, LIBXSMM_X86_GP_REG_RSP, scratch_size );
  if (libxsmm_verbosity < 0) {
    fprintf( stderr, "JITing Matrix Equation with reference code (n_tmp = %lld , stack_scratch_size = %.5g KB)\n", n_tmp, (1.0*scratch_size)/1024.0 );
  }
#if defined(_WIN32) || defined(__CYGWIN__)
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_R8);
#else /* match calling convention on Linux */
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSP, LIBXSMM_X86_GP_REG_RDX);
#endif

  /* Set size of temp in 4th argument */
#if defined(_WIN32) || defined(__CYGWIN__)
  libxsmm_x86_instruction_alu_imm_i64( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_R9, (unsigned long long) tmp_size  );
#else /* match calling convention on Linux */
  libxsmm_x86_instruction_alu_imm_i64( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RCX, (unsigned long long) tmp_size );
#endif

  /* We set the address of the function  */
  libxsmm_x86_instruction_alu_imm_i64( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RAX, (unsigned long long) libxsmm_reference_matequation );
  /* We call the function  */
  l_code_buffer[io_generated_code->code_size++] = 0xff;
  l_code_buffer[io_generated_code->code_size++] = 0xd0;
  /* Recover stack pointer and caller-saved register  */
  libxsmm_x86_instruction_alu_reg( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_RSP);
  libxsmm_x86_instruction_pop_reg( io_generated_code, LIBXSMM_X86_GP_REG_RBP );
  /* Close asm */
  libxsmm_x86_instruction_close_stream_alt( io_generated_code, 1 );
  free(unfolded_exec_tree);
}
