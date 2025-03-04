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
  unsigned int l_stack_offset = 0;
  unsigned char *l_padded_desc = NULL;
  unsigned long *l_imm_array_ptr = NULL;
  unsigned int l_temp_reg1, l_temp_reg2, l_sp_copy;
  libxsmm_ref_code_pointer l_code_ptr;
  libxsmm_ppc64le_reg l_reg_tracker;

  /* Initialise reg tracker */
  l_reg_tracker = libxsmm_ppc64le_reg_init();

  l_code_ptr.ptr = NULL;
  l_padded_desc = (unsigned char*) malloc( l_padded_desc_size );
  memset( l_padded_desc, 0, l_padded_desc_size );
  if ( 0 == i_is_gemm_or_eltwise ) {
    const libxsmm_gemm_descriptor* i_gemm_desc = (const libxsmm_gemm_descriptor*)i_desc;
    libxsmm_gemm_descriptor* l_desc_dst = (libxsmm_gemm_descriptor*)l_padded_desc;
    *l_desc_dst = *i_gemm_desc;
  } else {
    const libxsmm_meltw_descriptor* i_mateltwise_desc = (const libxsmm_meltw_descriptor*)i_desc;
    libxsmm_meltw_descriptor* l_desc_dst = (libxsmm_meltw_descriptor*)l_padded_desc;
    *l_desc_dst = *i_mateltwise_desc;
  }

  l_imm_array_ptr = (unsigned long*)l_padded_desc;

  /* Open stream */
  libxsmm_ppc64le_instr_open_stream( io_generated_code, &l_reg_tracker );

  /* preserve ARG0 as this is used in calling */
  libxsmm_ppc64le_used_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_ARG0 );

  /* Set some registers as used */
  libxsmm_ppc64le_used_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_R31 );
  l_sp_copy = LIBXSMM_PPC64LE_GPR_R31;
  libxsmm_ppc64le_used_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_ARG1 );
  l_temp_reg1 = LIBXSMM_PPC64LE_GPR_ARG1;
  libxsmm_ppc64le_used_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_R30 );
  l_temp_reg2 = LIBXSMM_PPC64LE_GPR_R30;

  /* Increament stack pointer to store description struct */
  libxsmm_ppc64le_instr_add_value( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR_SP, l_sp_copy, 0 );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_STDU, LIBXSMM_PPC64LE_GPR_SP, LIBXSMM_PPC64LE_GPR_SP, ( -l_padded_desc_size ) >> 2 );

  /* Store the descriptor in stack */
  l_stack_offset = 0;
  for ( i = 0; i < l_padded_desc_size / 32; ++i ) {
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, &l_reg_tracker, l_temp_reg1, l_imm_array_ptr[0] );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_STD, l_temp_reg1, LIBXSMM_PPC64LE_GPR_SP, ( l_stack_offset + 0 ) >> 2 );

    libxsmm_ppc64le_instr_set_imm64( io_generated_code, &l_reg_tracker, l_temp_reg2, l_imm_array_ptr[1] );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_STD, l_temp_reg2, LIBXSMM_PPC64LE_GPR_SP, ( l_stack_offset + 8 ) >> 2 );

    libxsmm_ppc64le_instr_set_imm64( io_generated_code, &l_reg_tracker, l_temp_reg1, l_imm_array_ptr[2] );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_STD, l_temp_reg1, LIBXSMM_PPC64LE_GPR_SP, ( l_stack_offset + 16 ) >> 2 );

    libxsmm_ppc64le_instr_set_imm64( io_generated_code, &l_reg_tracker, l_temp_reg2, l_imm_array_ptr[3] );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_STD, l_temp_reg2, LIBXSMM_PPC64LE_GPR_SP, ( l_stack_offset + 24 ) >> 2 );

    l_imm_array_ptr += 4;
    l_stack_offset += 32;
  }

  /* Set the address of the function  */
  if (i_is_gemm_or_eltwise == 0) {
    l_code_ptr.ptr_gemm_fn = libxsmm_reference_gemm;
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, &l_reg_tracker, l_temp_reg2, l_code_ptr.uval );
  } else {
    l_code_ptr.ptr_eltw_fn = libxsmm_reference_elementwise;
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, &l_reg_tracker, l_temp_reg2, l_code_ptr.uval );
  }

  /* Set stack pointer as argument 1 */
  libxsmm_ppc64le_instr_add_value( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR_SP, LIBXSMM_PPC64LE_GPR_ARG1, 0 );

  /* Call the function */
  libxsmm_ppc64le_instr_jump_lr( io_generated_code, l_temp_reg2 );

  /* Recover stack pointer */
  libxsmm_ppc64le_instr_add_value( io_generated_code, &l_reg_tracker, l_sp_copy, LIBXSMM_PPC64LE_GPR_SP, 0 );

  free(l_padded_desc);

  /* Free the register used */
  libxsmm_ppc64le_free_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_R31 );
  libxsmm_ppc64le_free_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_R30 );
  libxsmm_ppc64le_free_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_ARG0 );
  libxsmm_ppc64le_free_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_ARG1 );

  /* Close stream */
  libxsmm_ppc64le_instr_colapse_stack( io_generated_code, &l_reg_tracker );
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_ppc64le_reference_kernel( libxsmm_generated_code* io_generated_code,
                                                            const libxsmm_meltw_descriptor* i_mateltwise_desc ) {
  libxsmm_generator_ppc64le_reference_kernel( io_generated_code, (const void*) i_mateltwise_desc, 1);
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_ppc64le_reference_kernel( libxsmm_generated_code* io_generated_code,
                                                      const libxsmm_gemm_descriptor* i_gemm_desc ) {
  libxsmm_generator_ppc64le_reference_kernel( io_generated_code, (const void*) i_gemm_desc, 0);
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_ppc64le_reference_kernel( libxsmm_generated_code*        io_generated_code,
                                                             const libxsmm_meqn_descriptor* i_mateqn_desc ) {
  int l_stack_offset = 0;
  unsigned int l_sp_copy, l_temp_reg1, l_temp_reg2;
  unsigned int l_eqn_idx = i_mateqn_desc->eqn_idx;
  libxsmm_matrix_eqn *l_eqn = libxsmm_meqn_get_equation( l_eqn_idx );
  unsigned int last_timestamp, l_timestamp = 0;
  libxsmm_meqn_elem *l_unfolded_exec_tree = NULL;
  unsigned long l_padded_size = 0, i = 0;
  unsigned long tree_max_comp_tsize = 0;
  unsigned long n_tmp = 0, l_tmp_size = 0, l_scratch_size = 0;
  unsigned long *l_imm_array_ptr;
  libxsmm_ref_code_pointer l_code_ptr;
  libxsmm_ppc64le_reg l_reg_tracker;

  /* Initialise reg tracker */
  l_reg_tracker = libxsmm_ppc64le_reg_init();

  l_code_ptr.ptr = NULL;
  if ( NULL == l_eqn ) {
    fprintf( stderr, "The requested equation does not exist... nothing to JIT,,,\n" );
    return;
  } else {
    last_timestamp = l_eqn->eqn_root->visit_timestamp;
    l_padded_size = ( ( ( ( last_timestamp + 1 )*5*sizeof(libxsmm_meqn_elem) ) + 63 ) / 64 )*64;
    l_unfolded_exec_tree = (libxsmm_meqn_elem*)malloc(l_padded_size);
    l_imm_array_ptr = (unsigned long*)l_unfolded_exec_tree;
    memset( l_unfolded_exec_tree, 0, l_padded_size );

    tree_max_comp_tsize = l_eqn->eqn_root->tree_max_comp_tsize;
    n_tmp = l_eqn->eqn_root->reg_score;
    l_tmp_size = l_eqn->eqn_root->max_tmp_size * tree_max_comp_tsize;
  }

  /* Open stream */
  libxsmm_ppc64le_instr_open_stream( io_generated_code, &l_reg_tracker );

  /* Set some registers as used */
  libxsmm_ppc64le_used_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_R31 );
  l_sp_copy = LIBXSMM_PPC64LE_GPR_R31;
  libxsmm_ppc64le_used_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_R30 );
  l_temp_reg1 = LIBXSMM_PPC64LE_GPR_R30;
  libxsmm_ppc64le_used_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_R29 );
  l_temp_reg2 = LIBXSMM_PPC64LE_GPR_R29;
  libxsmm_ppc64le_used_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_ARG0 );
  libxsmm_ppc64le_used_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_ARG1 );
  libxsmm_ppc64le_used_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_ARG2 );
  libxsmm_ppc64le_used_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_ARG3 );

  /* Increament stack pointer to store description struct */
  libxsmm_ppc64le_instr_add_value( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR_SP, l_sp_copy, 0 );
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_STDU, LIBXSMM_PPC64LE_GPR_SP, LIBXSMM_PPC64LE_GPR_SP, ( -l_padded_size ) >> 2 );

  /* Here we unfold the optimized execution tree and store it in the stack */
  for ( l_timestamp = 0; l_timestamp <= last_timestamp; ++l_timestamp ) {
    libxsmm_meqn_elem *l_cur_op = libxsmm_generator_matequation_find_op_at_timestamp( l_eqn->eqn_root, l_timestamp );
    if ( NULL != l_cur_op ) l_unfolded_exec_tree[l_timestamp*5 + 0] = *l_cur_op;
    if ( NULL != l_cur_op && NULL != l_cur_op->le ) l_unfolded_exec_tree[l_timestamp*5 + 1] = *(l_cur_op->le);
    if ( NULL != l_cur_op && NULL != l_cur_op->ri ) l_unfolded_exec_tree[l_timestamp*5 + 2] = *(l_cur_op->ri);
    if ( NULL != l_cur_op && NULL != l_cur_op->r2 ) l_unfolded_exec_tree[l_timestamp*5 + 3] = *(l_cur_op->r2);
    if ( NULL != l_cur_op && NULL != l_cur_op->up ) l_unfolded_exec_tree[l_timestamp*5 + 4] = *(l_cur_op->up);
    if ( l_timestamp == last_timestamp ) {
      l_unfolded_exec_tree[l_timestamp*5 + 0].reg_score = -1;
      l_unfolded_exec_tree[l_timestamp*5 + 0].tmp.ld = i_mateqn_desc->ldo;
      l_unfolded_exec_tree[l_timestamp*5 + 0].tmp.dtype = (libxsmm_datatype)LIBXSMM_GETENUM_OUT( i_mateqn_desc->datatype );
    }
  }

  l_stack_offset = 0;
  /* Store the descriptor in stack and set ARG1 */
  for ( i = 0; i < l_padded_size/32; ++i ) {
    libxsmm_ppc64le_instr_set_imm64( io_generated_code, &l_reg_tracker, l_temp_reg1, l_imm_array_ptr[0] );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_STD, l_temp_reg1, LIBXSMM_PPC64LE_GPR_SP, ( l_stack_offset + 0 ) >> 2 );

    libxsmm_ppc64le_instr_set_imm64( io_generated_code, &l_reg_tracker, l_temp_reg2, l_imm_array_ptr[1] );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_STD, l_temp_reg2, LIBXSMM_PPC64LE_GPR_SP, ( l_stack_offset + 8 ) >> 2 );

    libxsmm_ppc64le_instr_set_imm64( io_generated_code, &l_reg_tracker, l_temp_reg1, l_imm_array_ptr[2] );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_STD, l_temp_reg1, LIBXSMM_PPC64LE_GPR_SP, ( l_stack_offset + 16 ) >> 2 );

    libxsmm_ppc64le_instr_set_imm64( io_generated_code, &l_reg_tracker, l_temp_reg2, l_imm_array_ptr[3] );
    libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_STD, l_temp_reg2, LIBXSMM_PPC64LE_GPR_SP, ( l_stack_offset + 24 ) >> 2 );

    l_imm_array_ptr += 4;
    l_stack_offset += 32;
  }
  libxsmm_ppc64le_instr_add_value( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR_SP, LIBXSMM_PPC64LE_GPR_ARG1, 0 );

  /* Get scratchpad pointer and set ARG2 */
  l_tmp_size = ( 0 == l_tmp_size % 64 ) ? l_tmp_size : ( ( l_tmp_size + 63 ) / 64 ) * 64;
  l_scratch_size = l_tmp_size * n_tmp;
  libxsmm_ppc64le_instr_3( io_generated_code, LIBXSMM_PPC64LE_INSTR_STDU, LIBXSMM_PPC64LE_GPR_SP, LIBXSMM_PPC64LE_GPR_SP, ( -l_scratch_size ) >> 2 );

  if ( libxsmm_verbosity < 0 ) {
    fprintf( stderr, "JITing Matrix Equation with reference code (n_tmp = %ld , stack_scratch_size = %.5g KB)\n", n_tmp, ( 1.0*l_scratch_size ) / 1024.0 );
  }
  libxsmm_ppc64le_instr_add_value( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR_SP, LIBXSMM_PPC64LE_GPR_ARG2, 0 );

  /* Set size of temp in ARG3 */
  libxsmm_ppc64le_instr_set_imm64( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR_ARG3, (unsigned long)l_tmp_size );

  /* We set the address of the function */
  l_code_ptr.ptr_meqn_fn = libxsmm_reference_matequation;
  libxsmm_ppc64le_instr_set_imm64( io_generated_code, &l_reg_tracker, l_temp_reg2, l_code_ptr.uval );

  /* Call the function */
  libxsmm_ppc64le_instr_jump_lr( io_generated_code, l_temp_reg2 );

  /* Recover stack pointer */
  libxsmm_ppc64le_instr_add_value( io_generated_code, &l_reg_tracker, l_sp_copy, LIBXSMM_PPC64LE_GPR_SP, 0 );

  /* Free the registers */
  libxsmm_ppc64le_free_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_R31 );
  libxsmm_ppc64le_free_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_R30 );
  libxsmm_ppc64le_free_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_R29 );
  libxsmm_ppc64le_free_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_ARG0 );
  libxsmm_ppc64le_free_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_ARG1 );
  libxsmm_ppc64le_free_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_ARG2 );
  libxsmm_ppc64le_free_reg( io_generated_code, &l_reg_tracker, LIBXSMM_PPC64LE_GPR, LIBXSMM_PPC64LE_GPR_ARG3 );

  /* Close stream */
  libxsmm_ppc64le_instr_colapse_stack( io_generated_code, &l_reg_tracker );

  free(l_unfolded_exec_tree);
}
