/******************************************************************************
* Copyright (c), 2025 IBM Corporation - All rights reserved.                  *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Will Trojak
******************************************************************************/
#include "generator_s390x_reference.h"

LIBXSMM_API_INTERN
void libxsmm_generator_s390x_reference_kernel( libxsmm_generated_code *io_generated_code,
                                               const void             *i_desc,
                                               unsigned int            i_is_gemm_or_eltwise ) {
  unsigned long l_padded_desc_size = (i_is_gemm_or_eltwise == 0) ? (((sizeof(libxsmm_gemm_descriptor)+31)/32) * 32) : (((sizeof(libxsmm_meltw_descriptor)+31)/32) * 32);
  unsigned long i = 0;
  unsigned char *l_padded_desc = NULL;
  unsigned int l_arg1, l_temp_reg, l_sp_copy;
  libxsmm_ref_code_pointer l_code_ptr;
  libxsmm_s390x_reg l_reg_tracker;

  l_code_ptr.ptr = NULL;

  l_padded_desc = (unsigned char*) malloc(l_padded_desc_size);
  memset(l_padded_desc, 0, l_padded_desc_size);
  if ( 0 == i_is_gemm_or_eltwise ) {
    const libxsmm_gemm_descriptor* i_gemm_desc = (const libxsmm_gemm_descriptor*)i_desc;
    libxsmm_gemm_descriptor* l_desc_dst = (libxsmm_gemm_descriptor*)l_padded_desc;
    *l_desc_dst = *i_gemm_desc;
  } else {
    const libxsmm_meltw_descriptor* i_mateltwise_desc = (const libxsmm_meltw_descriptor*)i_desc;
    libxsmm_meltw_descriptor* l_desc_dst = (libxsmm_meltw_descriptor*)l_padded_desc;
    *l_desc_dst = *i_mateltwise_desc;
  }

  /* Start tracking registers */
  l_reg_tracker = libxsmm_s390x_reg_init( io_generated_code );

  /* Open stream */
  libxsmm_s390x_instr_open_stack( io_generated_code );

  /* Set up some registers for temporaries and arguments */
  l_sp_copy = 11;
  libxsmm_s390x_reg_used( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR, l_sp_copy);
  libxsmm_s390x_reg_used( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR, LIBXSMM_S390X_GPR_ARG0);
  l_arg1 = LIBXSMM_S390X_GPR_ARG1;
  libxsmm_s390x_reg_used( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR, l_arg1);
  l_temp_reg = 10;
  libxsmm_s390x_reg_used( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR, l_temp_reg);

  /* Copy stack pointer and allocate stack memory */
  libxsmm_s390x_instr_gpr_copy( io_generated_code, LIBXSMM_S390X_GPR_SP, l_sp_copy );
  libxsmm_s390x_instr_gpr_add_value( io_generated_code, LIBXSMM_S390X_GPR_SP, LIBXSMM_S390X_GPR_SP, -l_padded_desc_size );

  /* Store the descriptor in stack */
  for ( i = 0; i < l_padded_desc_size; i++ ) {
    libxsmm_s390x_instr_move_imm8( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR_SP, i, l_padded_desc[i] );
  }

  /* Set input arg1 */
  libxsmm_s390x_instr_gpr_copy( io_generated_code, LIBXSMM_S390X_GPR_SP, l_arg1 );

  /* Set the address of the function */
  if ( 0 == i_is_gemm_or_eltwise ) {
    l_code_ptr.ptr_gemm_fn = libxsmm_reference_gemm;
    libxsmm_s390x_instr_gpr_imm64( io_generated_code, l_temp_reg, l_code_ptr.uval );
  } else {
    l_code_ptr.ptr_eltw_fn = libxsmm_reference_elementwise;
    libxsmm_s390x_instr_gpr_imm64( io_generated_code, l_temp_reg, l_code_ptr.uval );
  }

  /* Call the function */
  libxsmm_s390x_instr_call_jump( io_generated_code, l_temp_reg );

  /* Recover stack pointer */
  libxsmm_s390x_instr_gpr_copy( io_generated_code, l_sp_copy, LIBXSMM_S390X_GPR_SP );

  /* Free registers */
  libxsmm_s390x_reg_free( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR, l_sp_copy);
  libxsmm_s390x_reg_free( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR, l_arg1);
  libxsmm_s390x_reg_free( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR, LIBXSMM_S390X_GPR_ARG0);
  libxsmm_s390x_reg_free( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR, l_temp_reg);

  /* Close stream */
  libxsmm_s390x_instr_collapse_stack( io_generated_code );

  /* Free memory */
  libxsmm_s390x_reg_destroy( io_generated_code, &l_reg_tracker );
  free(l_padded_desc);
}

LIBXSMM_API_INTERN
void libxsmm_generator_mateltwise_s390x_reference_kernel( libxsmm_generated_code* io_generated_code,
                                                          const libxsmm_meltw_descriptor* i_mateltwise_desc ) {
  libxsmm_generator_s390x_reference_kernel( io_generated_code, (const void*) i_mateltwise_desc, 1);
}

LIBXSMM_API_INTERN
void libxsmm_generator_gemm_s390x_reference_kernel( libxsmm_generated_code* io_generated_code,
                                                    const libxsmm_gemm_descriptor* i_gemm_desc ) {
  libxsmm_generator_s390x_reference_kernel( io_generated_code, (const void*) i_gemm_desc, 0);
}

LIBXSMM_API_INTERN
void libxsmm_generator_matequation_s390x_reference_kernel( libxsmm_generated_code*        io_generated_code,
                                                           const libxsmm_meqn_descriptor* i_mateqn_desc ) {
  unsigned int eqn_idx = i_mateqn_desc->eqn_idx;
  libxsmm_matrix_eqn *l_eqn = libxsmm_meqn_get_equation( eqn_idx );
  libxsmm_meqn_elem *l_unfolded_exec_tree = NULL;
  unsigned int l_temp_reg, l_arg1, l_arg2, l_arg3, l_sp_copy;
  unsigned int l_timestamp = 0, l_last_timestamp;
  unsigned long l_padded_size = 0, i = 0;
  unsigned long l_n_tmp = 0, l_tmp_size = 0, l_scratch_size = 0;
  unsigned char *l_imm_array_ptr;
  libxsmm_ref_code_pointer l_code_ptr;
  libxsmm_s390x_reg l_reg_tracker;

  l_code_ptr.ptr = NULL;
  if ( NULL == l_eqn ) {
    fprintf( stderr, "The requested equation does not exist... nothing to JIT,,,\n" );
    return;
  } else {
    unsigned long l_tree_max_comp_tsize = 0;
    l_last_timestamp = l_eqn->eqn_root->visit_timestamp;

    l_padded_size = ( ( ( ( l_last_timestamp + 1 )*5*sizeof(libxsmm_meqn_elem) ) + 63 ) / 64 )*64;
    l_unfolded_exec_tree = (libxsmm_meqn_elem*) malloc( l_padded_size );
    l_imm_array_ptr = (unsigned char*)l_unfolded_exec_tree;
    memset( l_unfolded_exec_tree, 0, l_padded_size );

    l_tree_max_comp_tsize = l_eqn->eqn_root->tree_max_comp_tsize;
    l_n_tmp = l_eqn->eqn_root->reg_score;
    l_tmp_size = l_eqn->eqn_root->max_tmp_size * l_tree_max_comp_tsize;
  }

  /* Start tracking registers */
  l_reg_tracker = libxsmm_s390x_reg_init( io_generated_code );

  /* Open stream */
  libxsmm_s390x_instr_open_stack( io_generated_code );

  /* Set up some registers for temporaries and arguments */
  l_sp_copy = 11;
  libxsmm_s390x_reg_used( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR, l_sp_copy );
  libxsmm_s390x_reg_free( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR, LIBXSMM_S390X_GPR_ARG0 );
  l_arg1 = LIBXSMM_S390X_GPR_ARG1;
  libxsmm_s390x_reg_used( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR, l_arg1 );
  l_arg2 = LIBXSMM_S390X_GPR_ARG2;
  libxsmm_s390x_reg_used( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR, l_arg2 );
  l_arg3 = LIBXSMM_S390X_GPR_ARG3;
  libxsmm_s390x_reg_used( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR, l_arg3 );
  l_temp_reg = 10;
  libxsmm_s390x_reg_used( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR, l_temp_reg );

  /* Unfold the optimized execution tree */
  for ( l_timestamp = 0; l_timestamp <= l_last_timestamp; l_timestamp++ ) {
    libxsmm_meqn_elem *l_cur_op = libxsmm_generator_matequation_find_op_at_timestamp( l_eqn->eqn_root, l_timestamp );
    if ( NULL != l_cur_op ) l_unfolded_exec_tree[5*l_timestamp + 0] = *l_cur_op;
    if ( NULL != l_cur_op && NULL != l_cur_op->le ) l_unfolded_exec_tree[5*l_timestamp + 1] = *(l_cur_op->le);
    if ( NULL != l_cur_op && NULL != l_cur_op->ri ) l_unfolded_exec_tree[5*l_timestamp + 2] = *(l_cur_op->ri);
    if ( NULL != l_cur_op && NULL != l_cur_op->r2 ) l_unfolded_exec_tree[5*l_timestamp + 3] = *(l_cur_op->r2);
    if ( NULL != l_cur_op && NULL != l_cur_op->up ) l_unfolded_exec_tree[5*l_timestamp + 4] = *(l_cur_op->up);
    if ( l_timestamp == l_last_timestamp ) {
      l_unfolded_exec_tree[5*l_timestamp + 0].reg_score = -1;
      l_unfolded_exec_tree[5*l_timestamp + 0].tmp.ld = i_mateqn_desc->ldo;
      l_unfolded_exec_tree[5*l_timestamp + 0].tmp.dtype = (libxsmm_datatype) LIBXSMM_GETENUM_OUT( i_mateqn_desc->datatype );
    }
  }

  /* Copy stack pointer and allocate stack memory */
  libxsmm_s390x_instr_gpr_copy( io_generated_code, LIBXSMM_S390X_GPR_SP, l_sp_copy );
  libxsmm_s390x_instr_gpr_add_value( io_generated_code, LIBXSMM_S390X_GPR_SP, LIBXSMM_S390X_GPR_SP, -l_padded_size );

  /* Store the unfoled descriptor in stack */
  for ( i = 0; i < l_padded_size; i++ ) {
    libxsmm_s390x_instr_move_imm8( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR_SP, i, l_imm_array_ptr[i] );
  }

  /* Set input arg1 */
  libxsmm_s390x_instr_gpr_copy( io_generated_code, LIBXSMM_S390X_GPR_SP, l_arg1 );

  /* Get scratchpad pointer and set arg2 and arg3 */
  l_tmp_size = ( 0 == l_tmp_size % 64 ) ? l_tmp_size : ( ( l_tmp_size + 63 ) / 64) * 64;
  l_scratch_size = l_tmp_size * l_n_tmp;
  libxsmm_s390x_instr_gpr_add_value( io_generated_code, LIBXSMM_S390X_GPR_SP, LIBXSMM_S390X_GPR_SP, -l_scratch_size );
  libxsmm_s390x_instr_gpr_copy( io_generated_code, LIBXSMM_S390X_GPR_SP, l_arg2 );
  libxsmm_s390x_instr_gpr_set_value( io_generated_code, l_arg3, l_tmp_size );

  if (libxsmm_verbosity < 0) {
    fprintf( stderr, "JITing Matrix Equation with reference code (n_tmp = %ld , stack_scratch_size = %.5g KB)\n", l_n_tmp, (1.0*l_scratch_size)/1024.0 );
  }

  /* Set the address of the function */
  l_code_ptr.ptr_meqn_fn = libxsmm_reference_matequation;
  libxsmm_s390x_instr_gpr_imm64( io_generated_code, l_temp_reg, l_code_ptr.uval );

  /* Call the function  */
  libxsmm_s390x_instr_call_jump( io_generated_code, l_temp_reg );

  /* Recover stack pointer */
  libxsmm_s390x_instr_gpr_copy( io_generated_code, l_sp_copy, LIBXSMM_S390X_GPR_SP );

  /* Free registers */
  libxsmm_s390x_reg_free( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR, l_sp_copy);
  libxsmm_s390x_reg_free( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR, l_arg1);
  libxsmm_s390x_reg_free( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR, l_arg2);
  libxsmm_s390x_reg_free( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR, l_arg3);
  libxsmm_s390x_reg_free( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR, LIBXSMM_S390X_GPR_ARG0);
  libxsmm_s390x_reg_free( io_generated_code, &l_reg_tracker, LIBXSMM_S390X_GPR, l_temp_reg);

  /* Close stream */
  libxsmm_s390x_instr_collapse_stack( io_generated_code );

  /* Free memory */
  libxsmm_s390x_reg_destroy( io_generated_code, &l_reg_tracker );
  free(l_unfolded_exec_tree);
}
